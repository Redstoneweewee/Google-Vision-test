/**
 * Hybrid receipt-line reconstruction algorithm (Method 3: line-first).
 *
 * Pipeline:
 *   1a. Build tentative lines via neighbor graph (local y-overlap chaining)
 *   1b. Compute per-line baseline angle via least-squares fit
 *   1c. Fallback: estimate global angle for singletons
 *   2.  Per-line rotation + RANSAC-split any oversized cluster
 *   3.  Order words left→right inside each line
 *   4.  Detect the price column, assign prices, classify lines
 *   5.  Sort top→bottom, handle wrapped item names & discount/tax lines
 *
 * Why Method 3:
 *   A single global rotation angle fails when the receipt has a fold or
 *   local curvature — prices at the far right shift to a different
 *   rotated-Y and get assigned to the wrong line.  Method 3 builds
 *   tentative lines from LOCAL adjacency (y-overlap of nearby words),
 *   then derives a stable per-line angle so each line is de-skewed
 *   independently.  This is inherently robust to folds because adjacent
 *   words always have high y-overlap regardless of distant curvature.
 */

import { protos } from '@google-cloud/vision';

import {
  WordBox,
  Point,
  ReceiptLine,
  annotationToWordBox,
  median,
  rotatePoint,
  angleBetween,
  isPrice,
  classifyLine,
} from './utils';

import {
  ANGLE_PAIR_MAX_DY_FACTOR,
  ANGLE_PAIR_MIN_DX_FACTOR,
  ANGLE_PAIR_MAX_DX_FACTOR,
  ANGLE_MAX_ABS_RAD,
  ANGLE_HISTOGRAM_BIN_DEG,
  LINE_CLUSTER_Y_THRESHOLD_FACTOR,
  RANSAC_SPLIT_THRESHOLD_FACTOR,
  RANSAC_MIN_CLUSTER_SIZE,
  RANSAC_INLIER_DIST_FACTOR,
  RANSAC_MAX_ITERATIONS,
  RANSAC_MIN_INLIERS,
  FRAGMENT_MERGE_GAP_FACTOR,
  PRICE_COLUMN_MIN_PRICES,
  PRICE_COLUMN_TOLERANCE_FACTOR,
  WRAP_MAX_VERTICAL_GAP_FACTOR,
  WRAP_MAX_LEFT_ALIGN_FACTOR,
  NEIGHBOR_Y_OVERLAP_MIN,
  NEIGHBOR_MAX_X_GAP_FACTOR,
  NEIGHBOR_MAX_HEIGHT_RATIO,
} from './constants';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;

/* ── Internal type: WordBox + rotated coordinates ────────────────────────── */

interface RotatedBox extends WordBox {
  /** Rotated center point. */
  rc: Point;
  /** Rotated axis-aligned edges. */
  rLeft: number;
  rRight: number;
  rTop: number;
  rBottom: number;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 1a – Build tentative lines via neighbor graph
   ─────────────────────────────────────────────────────────────────────────
   For each word box, find its nearest right neighbor (smallest x-gap)
   among all boxes with sufficient y-overlap and similar height.
   Union-find groups transitively connected boxes into tentative lines.
   Each line is sorted left → right by center x.

   This works even on folded/curved receipts because:
   - Adjacent words on the same line are close in x (gap ~ 1–3 char widths)
   - The y-shift between adjacent words = Δx × sin(angle), which is tiny
     for small Δx, so y-overlap stays high regardless of local angle
   - The chain structure bridges from left to right across the full line;
     we never need to directly connect far-apart words
   ═══════════════════════════════════════════════════════════════════════════ */

class UnionFind {
  private parent: number[];
  private rank: number[];

  constructor(n: number) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x: number): number {
    if (this.parent[x] !== x) this.parent[x] = this.find(this.parent[x]);
    return this.parent[x];
  }

  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;
    if (this.rank[ra] < this.rank[rb]) {
      this.parent[ra] = rb;
    } else if (this.rank[ra] > this.rank[rb]) {
      this.parent[rb] = ra;
    } else {
      this.parent[rb] = ra;
      this.rank[ra]++;
    }
  }
}

function buildTentativeLines(boxes: WordBox[], medH: number): WordBox[][] {
  const n = boxes.length;
  if (n === 0) return [];

  const uf = new UnionFind(n);

  for (let i = 0; i < n; i++) {
    const A = boxes[i];
    let bestJ = -1;
    let bestXDist = Infinity;

    for (let j = 0; j < n; j++) {
      if (i === j) continue;
      const B = boxes[j];

      // B must be to the right of A
      if (B.center.x <= A.center.x) continue;

      // Y-overlap: intersection of vertical extents / smaller height
      const overlapTop = Math.max(A.top, B.top);
      const overlapBottom = Math.min(A.bottom, B.bottom);
      const overlap = Math.max(0, overlapBottom - overlapTop);
      const minH = Math.min(A.height, B.height);
      if (minH <= 0 || overlap / minH < NEIGHBOR_Y_OVERLAP_MIN) continue;

      // Height similarity
      const maxH = Math.max(A.height, B.height);
      if (maxH / minH > NEIGHBOR_MAX_HEIGHT_RATIO) continue;

      // X-gap (edge-to-edge, clamped to 0 for overlapping boxes)
      const xGap = Math.max(0, B.left - A.right);
      if (xGap > NEIGHBOR_MAX_X_GAP_FACTOR * medH) continue;

      // Pick the closest right neighbor (by center-x distance)
      const xDist = B.center.x - A.center.x;
      if (xDist < bestXDist) {
        bestXDist = xDist;
        bestJ = j;
      }
    }

    if (bestJ >= 0) {
      uf.union(i, bestJ);
    }
  }

  // Extract connected components
  const groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = uf.find(i);
    if (!groups.has(root)) groups.set(root, []);
    groups.get(root)!.push(i);
  }

  return Array.from(groups.values())
    .map((indices) => indices.map((idx) => boxes[idx]))
    .map((line) => [...line].sort((a, b) => a.center.x - b.center.x));
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 1b – Compute per-line baseline angle
   ─────────────────────────────────────────────────────────────────────────
   For each tentative line with ≥ 2 words, fit a line through the word
   centres using ordinary least squares and return the angle.  This is
   far more stable than per-word angles because it's averaged over many
   points (reducing OCR jitter).  Lines with a single word return 0
   (sentinel for "use the fallback global angle").
   ═══════════════════════════════════════════════════════════════════════════ */

function computeLineAngle(line: WordBox[]): number {
  if (line.length < 2) return 0;

  const n = line.length;
  const meanX = line.reduce((s, b) => s + b.center.x, 0) / n;
  const meanY = line.reduce((s, b) => s + b.center.y, 0) / n;

  let num = 0;
  let den = 0;
  for (const box of line) {
    const dx = box.center.x - meanX;
    const dy = box.center.y - meanY;
    num += dx * dy;
    den += dx * dx;
  }

  if (Math.abs(den) < 1e-9) return 0;
  return Math.atan2(num, den);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 1c – Global angle estimation (fallback for singletons)
   ─────────────────────────────────────────────────────────────────────────
   Retained from the original pipeline.  Used only when:
   - A tentative line has a single word (can't compute a per-line angle)
   - No multi-word lines exist at all (edge case)
   ═══════════════════════════════════════════════════════════════════════════ */

function estimateGlobalAngle(boxes: WordBox[]): number {
  if (boxes.length < 2) return 0;

  const medH = median(boxes.map((b) => b.height));
  if (medH <= 0) return 0;

  const angles: number[] = [];

  for (let i = 0; i < boxes.length; i++) {
    for (let j = i + 1; j < boxes.length; j++) {
      const a = boxes[i].center;
      const b = boxes[j].center;
      const dy = Math.abs(a.y - b.y);
      const dx = Math.abs(a.x - b.x);

      // Only consider nearby, horizontally-separated pairs
      if (
        dy > ANGLE_PAIR_MAX_DY_FACTOR * medH ||
        dx < ANGLE_PAIR_MIN_DX_FACTOR * medH ||
        dx > ANGLE_PAIR_MAX_DX_FACTOR * medH
      ) continue;

      // Compute angle and normalise into (-π/2, π/2]
      let angle = angleBetween(a, b);
      if (angle > Math.PI / 2) angle -= Math.PI;
      if (angle < -Math.PI / 2) angle += Math.PI;

      // Keep only roughly-horizontal angles
      if (Math.abs(angle) < ANGLE_MAX_ABS_RAD) {
        angles.push(angle);
      }
    }
  }

  if (angles.length === 0) return 0;

  // Histogram binning → pick peak
  const BIN = (ANGLE_HISTOGRAM_BIN_DEG * Math.PI) / 180;
  const bins = new Map<number, number>();
  for (const a of angles) {
    const key = Math.round(a / BIN);
    bins.set(key, (bins.get(key) ?? 0) + 1);
  }

  let bestBin = 0;
  let bestCount = 0;
  for (const [bin, count] of bins) {
    if (count > bestCount) {
      bestCount = count;
      bestBin = bin;
    }
  }

  return bestBin * BIN;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 2a – Rotate boxes (per-line)
   ─────────────────────────────────────────────────────────────────────────
   Same rotation logic as before, but now called once per tentative line
   with that line's own angle instead of a single global angle.
   ═══════════════════════════════════════════════════════════════════════════ */

function rotateBoxes(boxes: WordBox[], angle: number): RotatedBox[] {
  return boxes.map((box) => {
    const rc = rotatePoint(box.center, -angle);
    const rv = box.vertices.map((v) => rotatePoint(v, -angle));
    const rxs = rv.map((v) => v.x);
    const rys = rv.map((v) => v.y);
    return {
      ...box,
      rc,
      rLeft: Math.min(...rxs),
      rRight: Math.max(...rxs),
      rTop: Math.min(...rys),
      rBottom: Math.max(...rys),
    };
  });
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 2b – Y-cluster within a per-line rotation (unused in normal flow
   but retained for the RANSAC split fallback path)
   ═══════════════════════════════════════════════════════════════════════════ */

function clusterIntoLines(
  boxes: RotatedBox[],
  medH: number,
): RotatedBox[][] {
  if (boxes.length === 0) return [];

  const sorted = [...boxes].sort((a, b) => a.rc.y - b.rc.y);
  const threshold = LINE_CLUSTER_Y_THRESHOLD_FACTOR * medH;

  const clusters: { items: RotatedBox[]; meanY: number }[] = [];

  for (const box of sorted) {
    let best: (typeof clusters)[number] | null = null;
    let bestDist = Infinity;

    for (const c of clusters) {
      const d = Math.abs(box.rc.y - c.meanY);
      if (d < threshold && d < bestDist) {
        bestDist = d;
        best = c;
      }
    }

    if (best) {
      best.items.push(box);
      best.meanY =
        best.items.reduce((s, b) => s + b.rc.y, 0) / best.items.length;
    } else {
      clusters.push({ items: [box], meanY: box.rc.y });
    }
  }

  // Top → bottom
  clusters.sort((a, b) => a.meanY - b.meanY);
  return clusters.map((c) => c.items);
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 2c – RANSAC split for oversized tentative lines
   ─────────────────────────────────────────────────────────────────────────
   Now applied per-tentative-line: if the rotated-Y range of a tentative
   line (using that line's own angle) exceeds the threshold, RANSAC splits
   it.  This catches cases where the neighbor graph accidentally merged
   two close lines (tight spacing + large font size overlap).
   ═══════════════════════════════════════════════════════════════════════════ */

function ransacSplitCluster(
  cluster: RotatedBox[],
  medH: number,
): RotatedBox[][] {
  const ys = cluster.map((b) => b.rc.y);
  const yRange = Math.max(...ys) - Math.min(...ys);

  if (yRange < RANSAC_SPLIT_THRESHOLD_FACTOR * medH || cluster.length < RANSAC_MIN_CLUSTER_SIZE)
    return [cluster];

  const INLIER_DIST = RANSAC_INLIER_DIST_FACTOR * medH;
  const MAX_ITER = RANSAC_MAX_ITERATIONS;
  const lines: RotatedBox[][] = [];
  let remaining = [...cluster];

  while (remaining.length >= 2) {
    let bestInliers: RotatedBox[] = [];

    for (let iter = 0; iter < MAX_ITER; iter++) {
      // Pick 2 distinct random points
      const i1 = Math.floor(Math.random() * remaining.length);
      let i2 = Math.floor(Math.random() * (remaining.length - 1));
      if (i2 >= i1) i2++;

      const p1 = remaining[i1].rc;
      const p2 = remaining[i2].rc;

      // Line equation:  a·x + b·y + c = 0
      const a = p2.y - p1.y;
      const b = p1.x - p2.x;
      const c = p2.x * p1.y - p1.x * p2.y;
      const norm = Math.hypot(a, b);
      if (norm < 1e-9) continue;

      const inliers = remaining.filter((box) => {
        const dist = Math.abs(a * box.rc.x + b * box.rc.y + c) / norm;
        return dist < INLIER_DIST;
      });

      if (inliers.length > bestInliers.length) bestInliers = inliers;
    }

    if (bestInliers.length < RANSAC_MIN_INLIERS) {
      // Can't find a good line – attach leftovers to nearest existing line
      if (lines.length > 0) {
        attachToNearestLine(remaining, lines);
      } else {
        lines.push(remaining);
      }
      remaining = [];
      break;
    }

    lines.push(bestInliers);
    const inlierSet = new Set(bestInliers);
    remaining = remaining.filter((b) => !inlierSet.has(b));
  }

  // Remaining singletons → nearest line
  if (remaining.length > 0) {
    if (lines.length > 0) {
      attachToNearestLine(remaining, lines);
    } else {
      lines.push(remaining);
    }
  }

  // Sort sub-lines top → bottom
  lines.sort((a, b) => meanY(a) - meanY(b));
  return lines;
}

/** Attach each box in `boxes` to the nearest line (by mean rotated-Y). */
function attachToNearestLine(
  boxes: RotatedBox[],
  lines: RotatedBox[][],
): void {
  for (const box of boxes) {
    let nearest = lines[0];
    let nearestDist = Infinity;
    for (const line of lines) {
      const d = Math.abs(box.rc.y - meanY(line));
      if (d < nearestDist) {
        nearestDist = d;
        nearest = line;
      }
    }
    nearest.push(box);
  }
}

function meanY(line: RotatedBox[]): number {
  return line.reduce((s, b) => s + b.rc.y, 0) / line.length;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 3 – Order words left → right inside each line
   ═══════════════════════════════════════════════════════════════════════════ */

function orderWordsInLine(line: RotatedBox[]): RotatedBox[] {
  return [...line].sort((a, b) => a.rLeft - b.rLeft);
}

/**
 * Join ordered word boxes into a single text string.
 * Boxes whose horizontal gap is smaller than 0.3 × estimated character
 * width are concatenated without a space (handles OCR word-fragments like
 * "$" + "4.99" → "$4.99").
 */
function buildLineText(ordered: RotatedBox[], charW: number): string {
  if (ordered.length === 0) return '';
  let text = ordered[0].text;
  for (let i = 1; i < ordered.length; i++) {
    const gap = ordered[i].rLeft - ordered[i - 1].rRight;
    text += gap < charW * FRAGMENT_MERGE_GAP_FACTOR ? '' : ' ';
    text += ordered[i].text;
  }
  return text;
}

/* ═══════════════════════════════════════════════════════════════════════════
   Stage 4 – Detect the price column & assign prices
   ═══════════════════════════════════════════════════════════════════════════ */

function detectPriceColumnX(allLines: RotatedBox[][]): number | null {
  const priceRights: number[] = [];
  for (const line of allLines) {
    for (const box of line) {
      if (isPrice(box.text)) priceRights.push(box.right);
    }
  }
  return priceRights.length >= PRICE_COLUMN_MIN_PRICES ? median(priceRights) : null;
}

function assignPrice(
  ordered: RotatedBox[],
  priceColX: number | null,
  medH: number,
): { price: string | null; priceIndex: number } {
  // Walk right → left; pick the first (rightmost) price token
  for (let i = ordered.length - 1; i >= 0; i--) {
    if (!isPrice(ordered[i].text)) continue;

    // If a price column was detected, verify this token is near it.
    // Use original (unrotated) right edges so that per-line angle
    // differences don't shift the comparison — the physical right column
    // on the receipt is consistent in image space.
    if (priceColX !== null) {
      if (Math.abs(ordered[i].right - priceColX) > PRICE_COLUMN_TOLERANCE_FACTOR * medH) continue;
    }

    // Absorb trailing suffix tokens that OCR may have split from the price.
    // Known suffixes: "-" (discount), "A" (taxed item), "-A" (taxed item).
    let price = ordered[i].text.trim();
    for (let j = i + 1; j < ordered.length && j <= i + 2; j++) {
      const suf = ordered[j].text.trim();
      if (!/^(-|A|-A)$/i.test(suf)) break;

      // Try joining: direct concat, space, dash — pick first valid match
      const upper = suf.toUpperCase();
      const candidates = [
        price + upper,
        price + ' ' + upper,
        price + '-' + upper,
      ];
      const match = candidates.find((c) => isPrice(c));
      if (match) {
        price = match;
      } else {
        break;
      }
    }

    return { price, priceIndex: i };
  }
  return { price: null, priceIndex: -1 };
}

function removeNullItemNameItems(lines: ReceiptLine[]): void {
  for (const line of lines) {
    if (line.itemName === null) {
      lines.splice(lines.indexOf(line), 1);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Wrapped item names
   ─────────────────────────────────────────────────────────────────────────
   If a line has no price and is classified as "info", and the very next
   line below it has a price with a small vertical gap (< 1.5 × median
   height) and similar left-alignment (< 3 × median height difference),
   then it's a continuation of the item name.  We prepend the wrapped
   line's text to the next line's itemName and re-tag it as "wrapped".
   Iteration runs bottom-up so chained wraps (2+ continuation lines)
   cascade correctly.
   ═══════════════════════════════════════════════════════════════════════════ */

function handleWrappedNames(lines: ReceiptLine[], medH: number): void {
  for (let i = lines.length - 2; i >= 0; i--) {
    const cur = lines[i];
    const nxt = lines[i + 1];

    if (cur.lineType !== 'info' || cur.price !== null) continue;
    if (nxt.price === null || nxt.lineType === 'wrapped') continue;

    const curBoxes = cur.words;
    const nxtBoxes = nxt.words;
    if (curBoxes.length === 0 || nxtBoxes.length === 0) continue;

    // Vertical gap (original image coordinates)
    const curBottom = Math.max(...curBoxes.map((b) => b.bottom));
    const nxtTop = Math.min(...nxtBoxes.map((b) => b.top));
    const gap = nxtTop - curBottom;

    // Left-alignment check
    const curLeft = Math.min(...curBoxes.map((b) => b.left));
    const nxtLeft = Math.min(...nxtBoxes.map((b) => b.left));

    if (
      gap < WRAP_MAX_VERTICAL_GAP_FACTOR * medH &&
      Math.abs(curLeft - nxtLeft) < WRAP_MAX_LEFT_ALIGN_FACTOR * medH
    ) {
      nxt.itemName = [cur.text, nxt.itemName].filter(Boolean).join(' ');
      cur.lineType = 'wrapped';
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Orphan prices for keyword lines (subtotal / tax / total)
   ─────────────────────────────────────────────────────────────────────────
   On many receipts the keyword ("SUBTOTAL") and its price ("281.49") are
   far apart horizontally — often beyond the neighbor-graph's x-gap limit.
   They end up as separate tentative lines.  After classification we detect
   keyword lines with no price, and look for an adjacent price-only line
   (a line whose entire text is the price with no item name) immediately
   below.  If found, the price is absorbed into the keyword line and the
   orphan is hidden.
   ═══════════════════════════════════════════════════════════════════════════ */

function mergeOrphanPrices(lines: ReceiptLine[], medH: number): void {
  for (let i = 0; i < lines.length; i++) {
    const cur = lines[i];

    // Only apply to keyword lines that have no price
    if (!['subtotal', 'tax', 'total'].includes(cur.lineType) || cur.price !== null) continue;
    if (cur.words.length === 0) continue;

    const curTop = Math.min(...cur.words.map((w) => w.top));
    const curBottom = Math.max(...cur.words.map((w) => w.bottom));

    // ── Try the line BELOW first (original logic) ───────────────────────
    if (i < lines.length - 1) {
      const nxt = lines[i + 1];
      if (
        nxt.price !== null &&
        nxt.lineType !== 'wrapped' &&
        nxt.itemName === null &&
        nxt.words.length > 0
      ) {
        const nxtTop = Math.min(...nxt.words.map((w) => w.top));
        if (nxtTop - curBottom <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          cur.price = nxt.price;
          cur.text = cur.text + ' ' + nxt.text;
          cur.words = [...cur.words, ...nxt.words];
          cur.lineType = classifyLine(cur.text, cur.price);
          nxt.lineType = 'wrapped';
          continue;
        }
      }
    }

    // ── Try the line ABOVE (handles price-before-keyword layout) ────────
    if (i > 0) {
      const prev = lines[i - 1];
      if (
        prev.price !== null &&
        prev.lineType !== 'wrapped' &&
        prev.itemName === null &&
        prev.words.length > 0
      ) {
        const prevBottom = Math.max(...prev.words.map((w) => w.bottom));
        if (curTop - prevBottom <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          cur.price = prev.price;
          cur.text = prev.text + ' ' + cur.text;
          cur.words = [...prev.words, ...cur.words];
          cur.lineType = classifyLine(cur.text, cur.price);
          prev.lineType = 'wrapped';
          continue;
        }
      }
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Public entry point
   ═══════════════════════════════════════════════════════════════════════════ */

export interface ReconstructionResult {
  /** Ordered receipt lines, top-to-bottom. */
  lines: ReceiptLine[];
  /** Median of per-line angles (radians); representative global angle. */
  angle: number;
}

/**
 * Reconstruct ordered receipt lines from an unordered list of word-level
 * OCR annotations.  Pass `detections.slice(1)` (skip the full-text block).
 *
 * Uses Method 3 (line-first → per-line angles) to handle folds / local
 * rotation that a single global angle would get wrong.
 */
export function reconstructLines(
  annotations: TextAnnotation[],
): ReconstructionResult {
  const boxes = annotations.map(annotationToWordBox);
  if (boxes.length === 0) return { lines: [], angle: 0 };

  const medH = median(boxes.map((b) => b.height));

  // Estimate median character width (for fragment merging)
  const charW = median(
    boxes
      .map((b) => (b.text.length > 0 ? b.width / b.text.length : 0))
      .filter((w) => w > 0),
  );

  // ── Stage 1a: build tentative lines via neighbor graph ────────────────
  const tentativeLines = buildTentativeLines(boxes, medH);

  // ── Stage 1b: compute per-line angles ─────────────────────────────────
  const lineAngles = tentativeLines.map((line) => computeLineAngle(line));

  // ── Stage 1c: global angle fallback for singletons ────────────────────
  const validAngles = lineAngles.filter((a) => a !== 0);
  const globalAngle =
    validAngles.length > 0
      ? median(validAngles)
      : estimateGlobalAngle(boxes);
  const resolvedAngles = lineAngles.map((a) => (a === 0 ? globalAngle : a));

  // ── Stage 2: per-line rotation + RANSAC split ─────────────────────────
  interface LineBucket {
    boxes: RotatedBox[];
    angle: number;
  }
  const finalBuckets: LineBucket[] = [];

  for (let i = 0; i < tentativeLines.length; i++) {
    const angle = resolvedAngles[i];
    const rotated = rotateBoxes(tentativeLines[i], angle);
    const splits = ransacSplitCluster(rotated, medH);
    for (const split of splits) {
      finalBuckets.push({ boxes: split, angle });
    }
  }

  // ── Stage 3: order words left → right ─────────────────────────────────
  const orderedBuckets = finalBuckets.map(({ boxes: bxs, angle }) => ({
    ordered: orderWordsInLine(bxs),
    angle,
  }));

  // ── Stage 4: price column detection & assignment ──────────────────────
  const priceColX = detectPriceColumnX(
    orderedBuckets.map((b) => b.ordered),
  );

  const receiptLines: ReceiptLine[] = orderedBuckets.map(
    ({ ordered, angle }) => {
      const text = buildLineText(ordered, charW);
      const { price, priceIndex } = assignPrice(ordered, priceColX, medH);

      const itemName =
        priceIndex >= 0
          ? buildLineText(ordered.slice(0, priceIndex), charW).trim()
          : text.trim();

      const lineType = classifyLine(text, price);

      return {
        words: ordered,
        text,
        itemName: itemName || null,
        price,
        lineType,
        angle,
      };
    },
  );

  // ── Stage 5: sort top → bottom by original y, then edge cases ─────────
  receiptLines.sort((a, b) => {
    const aY =
      a.words.reduce((s, w) => s + w.center.y, 0) / (a.words.length || 1);
    const bY =
      b.words.reduce((s, w) => s + w.center.y, 0) / (b.words.length || 1);
    return aY - bY;
  });

  mergeOrphanPrices(receiptLines, medH);
  handleWrappedNames(receiptLines, medH);
  removeNullItemNameItems(receiptLines);

  return { lines: receiptLines, angle: globalAngle };
}

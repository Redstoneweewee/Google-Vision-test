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
  tryStripTaxFlag,
  classifyLine,
  parsePrice,
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
  ORPHAN_SEARCH_RADIUS,
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

  console.debug(`[DEBUG]    RANSAC triggered: yRange=${yRange.toFixed(1)} > threshold=${(RANSAC_SPLIT_THRESHOLD_FACTOR * medH).toFixed(1)}, ${cluster.length} boxes`);

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
      if (isPrice(box.text) || tryStripTaxFlag(box.text)) priceRights.push(box.right);
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
    const rawText = ordered[i].text;
    // Check if the token is a price directly, or a fused tax-flag+price (e.g. "F2.00")
    const stripped = tryStripTaxFlag(rawText);
    if (!isPrice(rawText) && !stripped) continue;

    // If a price column was detected, verify this token is near it.
    // Use original (unrotated) right edges so that per-line angle
    // differences don't shift the comparison — the physical right column
    // on the receipt is consistent in image space.
    if (priceColX !== null) {
      if (Math.abs(ordered[i].right - priceColX) > PRICE_COLUMN_TOLERANCE_FACTOR * medH) continue;
    }

    // Use the stripped (flag-free) price if that's how it matched
    let price = stripped ?? rawText.trim();

    // Absorb trailing suffix tokens that OCR may have split from the price.
    // Known suffixes: "-" (discount), "A" (taxed item), "-A" (taxed item).
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

/* ═══════════════════════════════════════════════════════════════════════════
   Voided entry handling
   ─────────────────────────────────────────────────────────────────────────
   Some receipts (esp. Walmart) have a `** VOIDED ENTRY` marker that
   means the previous item was scanned by mistake and should be removed.
   This function finds such markers and demotes the preceding item to
   'info' so it isn't counted.
   ═══════════════════════════════════════════════════════════════════════════ */

function handleVoidedEntries(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    handleVoidedEntries: scanning for void markers...`);
  for (let i = 0; i < lines.length; i++) {
    const lower = lines[i].text.toLowerCase();
    if (!/\bvoid(ed)?\b/.test(lower)) continue;

    console.debug(`[DEBUG]      Found void marker at line ${i}: "${lines[i].text}"`);
    // Demote the closest preceding item line
    for (let j = i - 1; j >= 0; j--) {
      if (['untaxed_item', 'taxed_item'].includes(lines[j].lineType)) {
        console.debug(`[DEBUG]        Voiding item at line ${j}: "${lines[j].text}" (${lines[j].price})`);
        lines[j].lineType = 'info';
        break;
      }
    }
    // The void marker itself becomes info
    if (lines[i].lineType !== 'info') {
      lines[i].lineType = 'info';
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Assign adjacent prices to priceless keyword / tender lines
   ─────────────────────────────────────────────────────────────────────────
   Some receipt formats (e.g. Trader Joe's) have keyword lines (SUBTOTAL,
   TOTAL) where the dollar amount appears as a separate OCR text block
   on an adjacent info line (e.g. "$38.68" on its own line).  This
   function detects that pattern and assigns the adjacent price.
   ═══════════════════════════════════════════════════════════════════════════ */

function assignAdjacentPricesToKeywords(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    assignAdjacentPricesToKeywords: scanning for priceless keyword lines...`);

  const isPriceOnlyLine = (l: ReceiptLine): boolean => {
    if (l.lineType !== 'info') return false;
    const t = l.text.trim().replace(/^\$/, '');
    return t.length > 0 && isPrice(t);
  };

  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!['subtotal', 'total', 'tax', 'tender'].includes(l.lineType)) continue;
    if (l.price !== null) continue; // already has a price

    // Check line BEFORE first (more reliable for interleaved layouts)
    if (i - 1 >= 0 && isPriceOnlyLine(lines[i - 1])) {
      l.price = lines[i - 1].text.trim().replace(/^\$/, '');
      lines[i - 1].lineType = 'wrapped'; // consumed
      console.debug(`[DEBUG]      Assigned adjacent price "${l.price}" to ${l.lineType} line ${i} ("${l.text}") from line before`);
      continue;
    }

    // Fallback: check line AFTER
    if (i + 1 < lines.length && isPriceOnlyLine(lines[i + 1])) {
      l.price = lines[i + 1].text.trim().replace(/^\$/, '');
      lines[i + 1].lineType = 'wrapped'; // consumed
      console.debug(`[DEBUG]      Assigned adjacent price "${l.price}" to ${l.lineType} line ${i} ("${l.text}") from line after`);
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Split combined TAX / BAL lines
   ─────────────────────────────────────────────────────────────────────────
   Some receipts (e.g. Whole Foods) combine tax and balance on one line:
       **** TAX .93 BAL 45.44
   This function detects the pattern and:
     1. Sets the tax line's price to the TAX amount
     2. Inserts a synthetic TOTAL line with the BAL amount
   ═══════════════════════════════════════════════════════════════════════════ */

function splitTaxBalLine(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    splitTaxBalLine: scanning for combined TAX/BAL lines...`);
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.lineType !== 'tax' || l.price !== null) continue;

    // Try patterns like "TAX .93 BAL 45.44" or "TAX 1.23 BAL 56.78"
    const m = l.text.match(/\btax\s+(\d*\.?\d+)\s+bal(?:ance)?\s+(\d+\.\d{2})\b/i);
    if (!m) continue;

    const taxVal = m[1];   // e.g. ".93" or "1.23"
    const balVal = m[2];   // e.g. "45.44"

    console.debug(`[DEBUG]      Line ${i}: splitting TAX=${taxVal}, BAL=${balVal} from "${l.text}"`);

    // Assign the tax amount to this line
    l.price = taxVal;

    // Insert a synthetic total line right after
    const syntheticTotal: ReceiptLine = {
      text: `BAL ${balVal}`,
      words: [],
      itemName: `BAL ${balVal}`,
      price: balVal,
      lineType: 'total',
      angle: l.angle,
    };
    lines.splice(i + 1, 0, syntheticTotal);
    // Skip the just-inserted line
    i++;
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Demote items that appear after the keyword block
   ─────────────────────────────────────────────────────────────────────────
   On many receipts the section below SUBTOTAL / TAX / TOTAL contains
   payment-method lines ("VISA CHARGE $44.11"), auth codes, or orphan
   amounts that the classifier couldn't recognise as payment info.
   These get classified as untaxed/taxed items and inflate the item sum.

   This function finds the LAST keyword line (subtotal / total)
   that has a price, and demotes every item line AFTER it to 'info'.
   ═══════════════════════════════════════════════════════════════════════════ */

function demotePostTotalItems(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    demotePostTotalItems: checking for items after keyword block...`);

  // Find the LAST total or subtotal line with a price.
  // Using the last (rather than first) avoids demoting real items
  // that appear between intermediate subtotals on multi-section receipts.
  let boundaryIdx = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (
      ['total', 'subtotal'].includes(lines[i].lineType) &&
      lines[i].price !== null &&
      lines[i].lineType !== 'wrapped'
    ) {
      boundaryIdx = i;
      break;
    }
  }

  if (boundaryIdx === -1) {
    console.debug(`[DEBUG]      No total/subtotal line with a price found, skipping`);
    return;
  }

  console.debug(`[DEBUG]      Boundary line: ${boundaryIdx} ("${lines[boundaryIdx].text}")`);

  for (let i = boundaryIdx + 1; i < lines.length; i++) {
    const l = lines[i];
    if (l.lineType === 'wrapped') continue;
    if (['untaxed_item', 'taxed_item'].includes(l.lineType)) {
      // Reclassify through classifyLine: if it matches tender keywords
      // it becomes 'tender'; otherwise demote to 'info'.
      const reclassified = classifyLine(l.text, l.price);
      const newType = reclassified === 'tender' ? 'tender' : 'info';
      console.debug(`[DEBUG]      Demoting post-keyword item line ${i}: "${l.text}" (${l.price}) → ${newType}`);
      l.lineType = newType;
    }
  }
}

/**
 * Correct mis-assigned keyword prices.
 *
 * When all three keyword lines (SUBTOTAL, TAX, TOTAL) have prices,
 * try the constraint SUBTOTAL + TAX = TOTAL to find the unique correct
 * assignment.  When only SUBTOTAL and TAX are present, tax should never
 * exceed subtotal so swap them if needed.
 */
function fixKeywordPriceAssignment(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    fixKeywordPriceAssignment: checking keyword price consistency...`);
  const sub = lines.find(l => l.lineType === 'subtotal' && l.price !== null);
  const tax = lines.find(l => l.lineType === 'tax' && l.price !== null);
  const tot = lines.find(l => l.lineType === 'total' && l.price !== null);

  // --- Case 1: all three present — constraint-based reassignment ----------
  if (sub && tax && tot) {
    const pairs: { str: string; val: number }[] = [
      { str: sub.price!, val: parsePrice(sub.price) },
      { str: tax.price!, val: parsePrice(tax.price) },
      { str: tot.price!, val: parsePrice(tot.price) },
    ];
    const sorted = [...pairs].sort((a, b) => a.val - b.val);

    // smallest + middle ≈ largest ?
    if (
      sorted[0].val >= 0 &&
      Math.abs(sorted[0].val + sorted[1].val - sorted[2].val) < 0.02
    ) {
      const idealTax = sorted[0].str;
      const idealSub = sorted[1].str;
      const idealTot = sorted[2].str;

      if (sub.price !== idealSub || tax.price !== idealTax || tot.price !== idealTot) {
        console.debug(
          `[DEBUG]      Reassigning: S=${sorted[1].val}, T=${sorted[0].val}, TOTAL=${sorted[2].val}`,
        );
        sub.price = idealSub;
        tax.price = idealTax;
        tot.price = idealTot;
      }
    }
    return;
  }

  // --- Case 2: only SUBTOTAL and TAX — simple swap if subtotal < tax ------
  if (sub && tax) {
    const subVal = parsePrice(sub.price);
    const taxVal = parsePrice(tax.price);
    if (subVal > 0 && taxVal > 0 && subVal < taxVal) {
      console.debug(`[DEBUG]      Swapping subtotal (${subVal}) ↔ tax (${taxVal})`);
      const tmp = sub.price;
      sub.price = tax.price;
      tax.price = tmp;
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Competing totals
   ─────────────────────────────────────────────────────────────────────────
   Some receipts print multiple total-like lines:
     "SALE TOTAL  $44.11"  then later  "TOTAL DUE  $44.11"
   or restaurant receipts:
     "TOTAL  $20.00"  then  "TOTAL WITH TIP  $24.00"

   When we find multiple 'total' lines with prices, we pick the best one:
   1. If one satisfies  SUBTOTAL + TAX ≈ TOTAL, prefer it.
   2. Otherwise, if all have the same price, keep the first.
   3. Otherwise, prefer the LAST total (usually the final charged amount).
   Demoted totals become 'info'.
   ═══════════════════════════════════════════════════════════════════════════ */

function resolveCompetingTotals(lines: ReceiptLine[]): void {
  const totalLines = lines.filter(
    (l) => l.lineType === 'total' && l.price !== null,
  );
  if (totalLines.length <= 1) return;

  console.debug(`[DEBUG]    resolveCompetingTotals: ${totalLines.length} total lines found`);

  const sub = lines.find((l) => l.lineType === 'subtotal' && l.price !== null);
  const tax = lines.find((l) => l.lineType === 'tax' && l.price !== null);

  let winner: ReceiptLine | null = null;

  // Strategy 1: prefer the total that satisfies S + T = TOTAL
  if (sub && tax) {
    const subVal = parsePrice(sub.price);
    const taxVal = parsePrice(tax.price);
    const expected = subVal + taxVal;
    for (const t of totalLines) {
      if (Math.abs(parsePrice(t.price) - expected) < 0.015) {
        winner = t;
        console.debug(`[DEBUG]      Winner (S+T match): "${t.text}" = ${t.price}`);
        break;
      }
    }
  }

  // Strategy 2: if all totals are the same value, keep the first
  if (!winner) {
    const vals = totalLines.map((t) => parsePrice(t.price));
    if (vals.every((v) => Math.abs(v - vals[0]) < 0.015)) {
      winner = totalLines[0];
      console.debug(`[DEBUG]      Winner (all same, keep first): "${winner.text}" = ${winner.price}`);
    }
  }

  // Strategy 3: prefer the last total (final charged amount)
  if (!winner) {
    winner = totalLines[totalLines.length - 1];
    console.debug(`[DEBUG]      Winner (last total): "${winner.text}" = ${winner.price}`);
  }

  // Demote all non-winners
  for (const t of totalLines) {
    if (t !== winner) {
      console.debug(`[DEBUG]      Demoting competing total: "${t.text}" (${t.price}) → info`);
      t.lineType = 'info';
    }
  }
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
  console.debug(`[DEBUG]    handleWrappedNames: scanning for info lines that wrap into priced lines...`);
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
      // Guard: if merging would reclassify the target line as info or tender
      // (e.g. "TOTAL NUMBER OF…" wrapping into an orphan price), skip.
      const mergedText = [cur.text, nxt.text].filter(Boolean).join(' ');
      const mergedType = classifyLine(mergedText, nxt.price);
      if (mergedType === 'info' || mergedType === 'tender') {
        console.debug(`[DEBUG]      Skipped wrap: "${cur.text}" into "${nxt.text}" — merged text classifies as ${mergedType}`);
        continue;
      }

      console.debug(`[DEBUG]      Wrapped: "${cur.text}" → prepended to "${nxt.text}" (gap=${gap.toFixed(1)}, leftDiff=${Math.abs(curLeft - nxtLeft).toFixed(1)})`);
      nxt.itemName = [cur.text, nxt.itemName].filter(Boolean).join(' ');
      cur.lineType = 'wrapped';
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Orphan prices for regular items
   ─────────────────────────────────────────────────────────────────────────
   When the receipt is tilted or held at an angle, the price at the far
   right of an item line may drift far enough in x that the neighbor graph
   can't connect it to the item name.  The result is two separate lines:
   one with the item name (classified "info", no price) and one with just
   the price (classified as an item with itemName === null).

   After sorting and keyword-line merging, this function finds priceless
   info lines (in the item section, above the first keyword line) and
   searches nearby lines for an orphan price-only line.  It searches UP
   first (up to ORPHAN_SEARCH_RADIUS lines) for the closest orphan price,
   then DOWN if nothing was found above.  The search uses angle-projected
   Y residuals to correctly match on tilted receipts.

   Processes from top to bottom so earlier items claim their prices first.
   ═══════════════════════════════════════════════════════════════════════════ */

function mergeOrphanItemPrices(lines: ReceiptLine[], medH: number): void {
  console.debug(`[DEBUG]    mergeOrphanItemPrices: scanning for orphan item prices...`);

  // Find boundary: first keyword line marks the end of the item section
  let firstKeywordIdx = lines.length;
  for (let i = 0; i < lines.length; i++) {
    if (['subtotal', 'tax', 'total'].includes(lines[i].lineType)) {
      firstKeywordIdx = i;
      break;
    }
  }

  /**
   * Check if an item name is trivial (null, empty, or just special chars /
   * a single letter).  Lines like "$ 6.79" or "9.00 F" have itemName "$"
   * or "F" — these are really just orphan prices, not real items.
   */
  function isTrivialItemName(name: string | null): boolean {
    if (name === null) return true;
    const cleaned = name.replace(/[^a-zA-Z0-9]/g, '');
    return cleaned.length <= 1;
  }

  /** Check if a line is an orphan price (has price, no real item name, not already merged). */
  function isOrphanPrice(l: ReceiptLine): boolean {
    return (
      l.price !== null &&
      isTrivialItemName(l.itemName) &&
      l.lineType !== 'wrapped' &&
      !['subtotal', 'tax', 'total'].includes(l.lineType) &&
      l.words.length > 0
    );
  }

  /** Check if a line is a priceless item (info with no price). */
  function isPricelessItem(l: ReceiptLine): boolean {
    return l.lineType === 'info' && l.price === null && l.words.length > 0;
  }

  /** Compute the angle-projected Y residual between an item line and an orphan price. */
  function computeResidual(item: ReceiptLine, orphan: ReceiptLine): number {
    const itemMeanX = item.words.reduce((s, w) => s + w.center.x, 0) / item.words.length;
    const itemMeanY = item.words.reduce((s, w) => s + w.center.y, 0) / item.words.length;
    const orphanMeanX = orphan.words.reduce((s, w) => s + w.center.x, 0) / orphan.words.length;
    const orphanMeanY = orphan.words.reduce((s, w) => s + w.center.y, 0) / orphan.words.length;
    const expectedY = itemMeanY + (orphanMeanX - itemMeanX) * Math.sin(item.angle);
    return Math.abs(orphanMeanY - expectedY);
  }

  // Process priceless items top-to-bottom
  for (let i = 0; i < firstKeywordIdx; i++) {
    const cur = lines[i];
    if (!isPricelessItem(cur)) continue;

    console.debug(`[DEBUG]      Checking item line ${i}: "${cur.text}" for orphan price...`);

    // Search UP first (up to ORPHAN_SEARCH_RADIUS lines)
    let bestCandidate: { idx: number; residual: number } | null = null;
    for (let j = i - 1; j >= Math.max(0, i - ORPHAN_SEARCH_RADIUS); j--) {
      if (j >= firstKeywordIdx) continue;
      if (!isOrphanPrice(lines[j])) continue;
      const residual = computeResidual(cur, lines[j]);
      if (residual <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
        if (bestCandidate === null || residual < bestCandidate.residual) {
          bestCandidate = { idx: j, residual };
        }
        console.debug(`[DEBUG]        ABOVE candidate line ${j}: "${lines[j].price}" residual=${residual.toFixed(1)}`);
      }
    }

    // If nothing found above, search DOWN
    if (bestCandidate === null) {
      for (let j = i + 1; j <= Math.min(lines.length - 1, i + ORPHAN_SEARCH_RADIUS); j++) {
        if (j >= firstKeywordIdx) continue;
        if (!isOrphanPrice(lines[j])) continue;
        const residual = computeResidual(cur, lines[j]);
        if (residual <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          // Take the first one found below (closest by line order)
          bestCandidate = { idx: j, residual };
          console.debug(`[DEBUG]        BELOW candidate line ${j}: "${lines[j].price}" residual=${residual.toFixed(1)}`);
          break;
        }
      }
    }

    if (bestCandidate !== null) {
      const orphan = lines[bestCandidate.idx];

      // Guard: if the merged text would reclassify as 'info' or 'tender'
      // (e.g. "TOTAL NUMBER OF..." absorbing an orphan price), skip the merge.
      const mergedText = cur.text + ' ' + orphan.text;
      const mergedType = classifyLine(mergedText, orphan.price);
      if (mergedType === 'info' || mergedType === 'tender') {
        console.debug(`[DEBUG]      Skipped merge: "${cur.text}" + "${orphan.price}" — merged text classifies as ${mergedType}`);
        continue;
      }

      console.debug(`[DEBUG]      Merged: "${cur.text}" + "${orphan.price}" (residual=${bestCandidate.residual.toFixed(1)})`);

      cur.price = orphan.price;
      cur.itemName = cur.text;
      cur.text = cur.text + ' ' + orphan.text;
      cur.words = [...cur.words, ...orphan.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      orphan.lineType = 'wrapped';
    }
  }
}

/* ═══════════════════════════════════════════════════════════════════════════
   Edge case – Orphan prices for keyword lines (subtotal / tax / total)
   ─────────────────────────────────────────────────────────────────────────
   On many receipts the keyword ("SUBTOTAL") and its price ("281.49") are
   far apart horizontally — often beyond the neighbor-graph's x-gap limit.
   They end up as separate tentative lines.  After classification we detect
   keyword lines with no price, and search nearby lines (within
   ORPHAN_SEARCH_RADIUS) for a price-only orphan line.

   Processing order: highest unmerged keyword first, then downward.
   For each keyword, search UP first (up to ORPHAN_SEARCH_RADIUS lines).
   If nothing is found above, search DOWN for the first orphan price.
   This handles curved/distorted receipts where all keywords are stacked
   above their corresponding prices (e.g. SUBTOTAL / TAX / TOTAL followed
   by 73.60 / 0.44 / 74.04).
   ═══════════════════════════════════════════════════════════════════════════ */

function mergeOrphanPrices(lines: ReceiptLine[], medH: number): void {
  console.debug(`[DEBUG]    mergeOrphanPrices: scanning ${lines.length} lines for keyword lines without prices...`);

  /**
   * Check if an item name is trivial — see mergeOrphanItemPrices for docs.
   */
  function isTrivialItemName(name: string | null): boolean {
    if (name === null) return true;
    const cleaned = name.replace(/[^a-zA-Z0-9]/g, '');
    return cleaned.length <= 1;
  }

  /** Check if a line is an orphan price (has price, no real item name, not already used). */
  function isOrphanPrice(l: ReceiptLine): boolean {
    return (
      l.price !== null &&
      l.lineType !== 'wrapped' &&
      isTrivialItemName(l.itemName) &&
      l.words.length > 0
    );
  }

  // Process keyword lines top-to-bottom
  for (let i = 0; i < lines.length; i++) {
    const cur = lines[i];

    // Only apply to keyword lines that have no price
    if (!['subtotal', 'tax', 'total'].includes(cur.lineType) || cur.price !== null) continue;
    if (cur.words.length === 0) continue;

    console.debug(`[DEBUG]      Checking keyword line ${i}: "${cur.text}" for orphan price...`);

    const curTop = Math.min(...cur.words.map((w) => w.top));
    const curBottom = Math.max(...cur.words.map((w) => w.bottom));

    // ── Search UP first (up to ORPHAN_SEARCH_RADIUS lines) ──────────────
    let bestAbove: { idx: number; gap: number } | null = null;
    for (let j = i - 1; j >= Math.max(0, i - ORPHAN_SEARCH_RADIUS); j--) {
      const candidate = lines[j];
      if (!isOrphanPrice(candidate)) continue;

      const candBottom = Math.max(...candidate.words.map((w) => w.bottom));
      const gap = curTop - candBottom;
      if (gap <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
        // Pick the closest above candidate
        if (bestAbove === null || gap < bestAbove.gap) {
          bestAbove = { idx: j, gap };
        }
        console.debug(`[DEBUG]        ABOVE candidate line ${j}: "${candidate.price}" gap=${gap.toFixed(1)}`);
      }
    }

    // ── If nothing above, search DOWN (up to ORPHAN_SEARCH_RADIUS) ──────
    let bestBelow: { idx: number; gap: number } | null = null;
    if (bestAbove === null) {
      for (let j = i + 1; j <= Math.min(lines.length - 1, i + ORPHAN_SEARCH_RADIUS); j++) {
        const candidate = lines[j];
        if (!isOrphanPrice(candidate)) continue;

        const candTop = Math.min(...candidate.words.map((w) => w.top));
        const gap = candTop - curBottom;
        if (gap <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          // Take the first valid one below (closest by line order)
          bestBelow = { idx: j, gap };
          console.debug(`[DEBUG]        BELOW candidate line ${j}: "${candidate.price}" gap=${gap.toFixed(1)}`);
          break;
        }
      }
    }

    // ── Merge the best candidate ────────────────────────────────────────
    if (bestAbove !== null) {
      const prev = lines[bestAbove.idx];
      console.debug(`[DEBUG]      Orphan merge: "${cur.text}" ← ABOVE price "${prev.price}" (gap=${bestAbove.gap.toFixed(1)})`);
      cur.price = prev.price;
      cur.text = prev.text + ' ' + cur.text;
      cur.words = [...prev.words, ...cur.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      prev.lineType = 'wrapped';
    } else if (bestBelow !== null) {
      const nxt = lines[bestBelow.idx];
      console.debug(`[DEBUG]      Orphan merge: "${cur.text}" ← BELOW price "${nxt.price}" (gap=${bestBelow.gap.toFixed(1)})`);
      cur.price = nxt.price;
      cur.text = cur.text + ' ' + nxt.text;
      cur.words = [...cur.words, ...nxt.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      nxt.lineType = 'wrapped';
    } else {
      console.debug(`[DEBUG]      No orphan price found for "${cur.text}"`);
    }
  }

  // ── Last-resort pass: keyword lines still without prices ──────────────
  // On heavily-tilted receipts, the price column can drift so far that
  // orphan prices end up well beyond ORPHAN_SEARCH_RADIUS.  Do a second
  // pass with no radius/gap limit, matching by nearest Y center.
  for (let i = 0; i < lines.length; i++) {
    const cur = lines[i];
    if (!['subtotal', 'tax', 'total'].includes(cur.lineType) || cur.price !== null) continue;
    if (cur.words.length === 0) continue;

    const curY = cur.words.reduce((s, w) => s + w.center.y, 0) / cur.words.length;
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let j = 0; j < lines.length; j++) {
      if (!isOrphanPrice(lines[j])) continue;
      const candY = lines[j].words.reduce((s, w) => s + w.center.y, 0) / lines[j].words.length;
      const dist = Math.abs(candY - curY);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = j;
      }
    }

    if (bestIdx >= 0) {
      const orphan = lines[bestIdx];
      console.debug(`[DEBUG]      Last-resort merge: "${cur.text}" ← orphan "${orphan.price}" (Ydist=${bestDist.toFixed(1)})`);
      cur.price = orphan.price;
      cur.text = cur.text + ' ' + orphan.text;
      cur.words = [...cur.words, ...orphan.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      orphan.lineType = 'wrapped';
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

  console.debug(`[DEBUG] ══════════════════════════════════════════════════`);
  console.debug(`[DEBUG] reconstructLines: ${boxes.length} word boxes, medH=${medH.toFixed(1)}, charW=${charW.toFixed(1)}`);

  // ── Stage 1a: build tentative lines via neighbor graph ────────────────
  console.debug(`[DEBUG] ── Stage 1a: Building tentative lines via neighbor graph...`);
  const tentativeLines = buildTentativeLines(boxes, medH);
  console.debug(`[DEBUG]    → ${tentativeLines.length} tentative lines formed`);
  for (let _i = 0; _i < tentativeLines.length; _i++) {
    const tl = tentativeLines[_i];
    console.debug(`[DEBUG]    Line ${_i}: ${tl.length} words → "${tl.map(w => w.text).join(' ')}"`);
  }

  // ── Stage 1b: compute per-line angles ─────────────────────────────────
  console.debug(`[DEBUG] ── Stage 1b: Computing per-line angles...`);
  const lineAngles = tentativeLines.map((line) => computeLineAngle(line));
  for (let _i = 0; _i < lineAngles.length; _i++) {
    const deg = (lineAngles[_i] * 180 / Math.PI).toFixed(2);
    console.debug(`[DEBUG]    Line ${_i}: angle=${deg}°${lineAngles[_i] === 0 ? ' (singleton → fallback)' : ''}`);
  }

  // ── Stage 1c: global angle fallback for singletons ────────────────────
  console.debug(`[DEBUG] ── Stage 1c: Global angle fallback...`);
  const validAngles = lineAngles.filter((a) => a !== 0);
  const globalAngle =
    validAngles.length > 0
      ? median(validAngles)
      : estimateGlobalAngle(boxes);
  const resolvedAngles = lineAngles.map((a) => (a === 0 ? globalAngle : a));
  console.debug(`[DEBUG]    ${validAngles.length} valid per-line angles, globalAngle=${(globalAngle * 180 / Math.PI).toFixed(2)}°`);
  const singletonCount = lineAngles.filter(a => a === 0).length;
  if (singletonCount > 0) console.debug(`[DEBUG]    ${singletonCount} singleton lines using global fallback`);

  // ── Stage 2: per-line rotation + RANSAC split ─────────────────────────
  console.debug(`[DEBUG] ── Stage 2: Per-line rotation + RANSAC split...`);
  interface LineBucket {
    boxes: RotatedBox[];
    angle: number;
  }
  const finalBuckets: LineBucket[] = [];

  for (let i = 0; i < tentativeLines.length; i++) {
    const angle = resolvedAngles[i];
    const rotated = rotateBoxes(tentativeLines[i], angle);
    const splits = ransacSplitCluster(rotated, medH);
    if (splits.length > 1) {
      console.debug(`[DEBUG]    RANSAC split tentative line ${i} (${tentativeLines[i].length} words) into ${splits.length} sub-lines`);
      for (let s = 0; s < splits.length; s++) {
        console.debug(`[DEBUG]      Sub-line ${s}: ${splits[s].length} words → "${splits[s].map(w => w.text).join(' ')}"`);
      }
    }
    for (const split of splits) {
      finalBuckets.push({ boxes: split, angle });
    }
  }
  console.debug(`[DEBUG]    → ${finalBuckets.length} final line buckets (from ${tentativeLines.length} tentative)`);

  // ── Stage 3: order words left → right ─────────────────────────────────
  const orderedBuckets = finalBuckets.map(({ boxes: bxs, angle }) => ({
    ordered: orderWordsInLine(bxs),
    angle,
  }));

  // ── Stage 3b: detect price column & split multi-price lines ───────────
  console.debug(`[DEBUG] ── Stage 3b: Detect price column & split multi-price lines...`);
  const priceColX = detectPriceColumnX(
    orderedBuckets.map((b) => b.ordered),
  );
  console.debug(`[DEBUG]    Price column X: ${priceColX !== null ? priceColX.toFixed(1) : '(not detected)'}`);

  // If a single line has multiple price tokens near the price column,
  // it probably merged multiple receipt lines (common with background text
  // overlapping receipts).  Split at each price token boundary.
  if (priceColX !== null) {
    for (let bi = orderedBuckets.length - 1; bi >= 0; bi--) {
      const { ordered, angle } = orderedBuckets[bi];
      // Find all price token indices near the price column
      const priceIndices: number[] = [];
      for (let wi = 0; wi < ordered.length; wi++) {
        if (
          (isPrice(ordered[wi].text) || tryStripTaxFlag(ordered[wi].text)) &&
          Math.abs(ordered[wi].right - priceColX) <= PRICE_COLUMN_TOLERANCE_FACTOR * medH
        ) {
          priceIndices.push(wi);
        }
      }
      if (priceIndices.length < 2) continue;

      // Split: each price ends a sub-line, leftovers go into the last group
      console.debug(`[DEBUG]    Multi-price split: line "${ordered.map(w => w.text).join(' ')}" has ${priceIndices.length} prices at indices [${priceIndices.join(',')}]`);
      const subLines: { ordered: RotatedBox[]; angle: number }[] = [];
      let start = 0;
      for (const pi of priceIndices) {
        subLines.push({ ordered: ordered.slice(start, pi + 1), angle });
        start = pi + 1;
      }
      // Any trailing words after the last price go into the last sub-line
      if (start < ordered.length) {
        subLines[subLines.length - 1].ordered = [
          ...subLines[subLines.length - 1].ordered,
          ...ordered.slice(start),
        ];
      }
      orderedBuckets.splice(bi, 1, ...subLines);
      for (let s = 0; s < subLines.length; s++) {
        console.debug(`[DEBUG]      Sub-line ${s}: ${subLines[s].ordered.length} words → "${subLines[s].ordered.map(w => w.text).join(' ')}"`);
      }
    }
  }

  // ── Stage 4: price assignment ─────────────────────────────────────────
  console.debug(`[DEBUG] ── Stage 4: Price column detection & assignment...`);

  const receiptLines: ReceiptLine[] = orderedBuckets.map(
    ({ ordered, angle }, _idx) => {
      const text = buildLineText(ordered, charW);
      const { price, priceIndex } = assignPrice(ordered, priceColX, medH);

      const itemName =
        priceIndex >= 0
          ? buildLineText(ordered.slice(0, priceIndex), charW).trim()
          : text.trim();

      const lineType = classifyLine(text, price);

      console.debug(
        `[DEBUG]    Line ${_idx}: type=${lineType.padEnd(12)} price=${(price ?? '(none)').toString().padEnd(10)} text="${text}"`
      );

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
  console.debug(`[DEBUG] ── Stage 5: Sort + edge cases...`);
  receiptLines.sort((a, b) => {
    const aY =
      a.words.reduce((s, w) => s + w.center.y, 0) / (a.words.length || 1);
    const bY =
      b.words.reduce((s, w) => s + w.center.y, 0) / (b.words.length || 1);
    return aY - bY;
  });

  // Item prices first so keyword lines don't steal item orphan prices
  mergeOrphanItemPrices(receiptLines, medH);
  handleWrappedNames(receiptLines, medH);
  // Split combined TAX/BAL lines (e.g. "**** TAX .93 BAL 45.44")
  splitTaxBalLine(receiptLines);
  // Assign prices from adjacent info lines to priceless keyword lines
  // BEFORE mergeOrphanPrices, so keywords don't get wrong orphan prices
  assignAdjacentPricesToKeywords(receiptLines);
  mergeOrphanPrices(receiptLines, medH);
  demotePostTotalItems(receiptLines);
  handleVoidedEntries(receiptLines);
  resolveCompetingTotals(receiptLines);
  fixKeywordPriceAssignment(receiptLines);
  removeNullItemNameItems(receiptLines);

  console.debug(`[DEBUG] ── Final output: ${receiptLines.length} lines (excluding wrapped)`);
  for (const rl of receiptLines) {
    if (rl.lineType === 'wrapped') continue;
    console.debug(`[DEBUG]    [${rl.lineType.padEnd(12)}] ${(rl.itemName ?? rl.text).padEnd(35)} ${rl.price ?? ''}`);
  }
  console.debug(`[DEBUG] ══════════════════════════════════════════════════`);

  return { lines: receiptLines, angle: globalAngle };
}

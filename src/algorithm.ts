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

import type { WordBox, Point, ReceiptLine, ReconstructionResult } from './types';
import {
  annotationToWordBox,
  median,
  rotatePoint,
  angleBetween,
  isPrice,
  tryStripTaxFlag,
  classifyLine,
} from './utils';
import {
  handleVoidedEntries,
  assignAdjacentPricesToKeywords,
  splitTaxBalLine,
  demotePostTotalItems,
  fixKeywordPriceAssignment,
  resolveCompetingTotals,
  removeNullItemNameItems,
  handleWrappedNames,
  mergeOrphanItemPrices,
  mergeOrphanPrices,
  reconstructBrokenTaxPrices,
  extractBarcodePrices,
  rescueBarcodeItems,
} from './edge-cases';

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
  NEIGHBOR_SHORT_GAP_FACTOR,
  NEIGHBOR_MAX_HEIGHT_RATIO,
  VECTOR_Y_TOLERANCE_FACTOR,
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

  // ── Phase 1: Short-range neighbor connections ──────────────────────────
  // Use a conservative x-gap to reliably group nearby text words without
  // accidentally linking distant prices to the wrong line.
  const shortGap = NEIGHBOR_SHORT_GAP_FACTOR * medH;

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
      if (xGap > shortGap) continue;

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

  // ── Phase 2: Vector-extended connections ───────────────────────────────
  // For each multi-word group from phase 1, compute its direction vector
  // and extend rightward to capture distant boxes (e.g. prices) that
  // align with the line's trajectory.  This is more accurate than raw
  // y-overlap for long-range connections because it follows the actual
  // line angle, handling tilted/curved receipts correctly.
  const longGap = NEIGHBOR_MAX_X_GAP_FACTOR * medH;
  const vectorTol = VECTOR_Y_TOLERANCE_FACTOR * medH;

  // Snapshot phase-1 roots and build groups
  const p1Root: number[] = new Array(n);
  for (let i = 0; i < n; i++) p1Root[i] = uf.find(i);

  const p1Groups = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const r = p1Root[i];
    if (!p1Groups.has(r)) p1Groups.set(r, []);
    p1Groups.get(r)!.push(i);
  }

  // Collect merge candidates: for each multi-word line, find distant
  // boxes whose position aligns with the line's direction vector.
  interface MergeCandidate {
    yError: number;
    lineRightmostIdx: number;
    targetP1Root: number;
    targetIdx: number;
    lineText: string;
    targetText: string;
  }
  const candidates: MergeCandidate[] = [];

  for (const [lineRoot, lineIndices] of p1Groups) {
    if (lineIndices.length < 2) {
      // ── Singleton fallback: use y-overlap (like Phase 1) with long gap ──
      // Singletons (e.g. "TOTAL", "CHANGE") can't compute a direction angle,
      // so we fall back to simple y-overlap matching within the long gap range.
      const sIdx = lineIndices[0];
      const S = boxes[sIdx];

      for (let i = 0; i < n; i++) {
        if (p1Root[i] === lineRoot) continue;
        const B = boxes[i];
        if (B.center.x <= S.center.x) continue;

        const xGap = Math.max(0, B.left - S.right);
        if (xGap <= shortGap) continue; // Already handled by Phase 1
        if (xGap > longGap) continue;

        // Height similarity
        const sMinH = Math.min(S.height, B.height);
        const sMaxH = Math.max(S.height, B.height);
        if (sMaxH / sMinH > NEIGHBOR_MAX_HEIGHT_RATIO) continue;

        // Y-overlap check (same criterion as Phase 1)
        const overlapTop = Math.max(S.top, B.top);
        const overlapBottom = Math.min(S.bottom, B.bottom);
        const overlap = Math.max(0, overlapBottom - overlapTop);
        const minH = Math.min(S.height, B.height);
        if (minH <= 0 || overlap / minH < NEIGHBOR_Y_OVERLAP_MIN) continue;

        const yError = Math.abs(B.center.y - S.center.y);
        candidates.push({
          yError,
          lineRightmostIdx: sIdx,
          targetP1Root: p1Root[i],
          targetIdx: i,
          lineText: S.text,
          targetText: B.text,
        });
      }
      continue;
    }

    const lineBoxes = lineIndices.map(i => boxes[i]);
    const angle = computeLineAngle(lineBoxes);

    // Find rightmost box in this line
    let rIdx = lineIndices[0];
    for (const idx of lineIndices) {
      if (boxes[idx].right > boxes[rIdx].right) rIdx = idx;
    }
    const R = boxes[rIdx];

    for (let i = 0; i < n; i++) {
      if (p1Root[i] === lineRoot) continue;

      const B = boxes[i];
      if (B.center.x <= R.center.x) continue; // Must be to the right

      const xGap = Math.max(0, B.left - R.right);
      if (xGap > longGap) continue;

      // Height similarity
      const bMinH = Math.min(B.height, R.height);
      const bMaxH = Math.max(B.height, R.height);
      if (bMaxH / bMinH > NEIGHBOR_MAX_HEIGHT_RATIO) continue;

      // Vector projection: predict expected y at B's x position
      const dx = B.center.x - R.center.x;
      const expectedY = R.center.y + dx * Math.tan(angle);
      const yError = Math.abs(B.center.y - expectedY);

      candidates.push({
        yError,
        lineRightmostIdx: rIdx,
        targetP1Root: p1Root[i],
        targetIdx: i,
        lineText: lineBoxes.map(b => b.text).join(' '),
        targetText: boxes[i].text,
      });
    }
  }

  // Sort by y-error (best first) and apply greedily.
  // The claimed set prevents a target group from being absorbed by
  // multiple source lines — first (best) match wins.
  candidates.sort((a, b) => a.yError - b.yError);
  const claimed = new Set<number>();

  for (const c of candidates) {
    if (c.yError > vectorTol) break;
    if (claimed.has(c.targetP1Root)) continue;
    if (uf.find(c.targetIdx) === uf.find(c.lineRightmostIdx)) continue;

    console.debug(`[DEBUG]    Phase 2 vector link: "${c.lineText}" → "${c.targetText}" (yErr=${c.yError.toFixed(1)})`);
    claimed.add(c.targetP1Root);
    uf.union(c.lineRightmostIdx, c.targetIdx);
  }

  // Extract final connected components
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
      // When OCR splits "$" and the number into separate words (e.g. "$" + "376.31"),
      // the number word extends further right than a fused "$376.31" word would.
      // Use the "$" word's right edge for the column check if present, since
      // it's comparable to fused "$xxx" words that established the price column.
      // Allow slightly extra tolerance for the gap between "$" and the number.
      let checkRight = ordered[i].right;
      let extraTol = 0;
      if (i > 0 && ordered[i - 1].text.trim() === '$') {
        // Only apply "$" fallback for keyword lines (total, subtotal, tax, etc.),
        // not standalone price fragments like "$ 38.68".
        const lineText = ordered.map(w => w.text).join(' ').toLowerCase();
        if (/\b(total|subtotal|sub.?total|balance|amount\s+due|tax|hst|gst|pst|net\s+sales)\b/.test(lineText)) {
          checkRight = ordered[i - 1].right;
          extraTol = medH;
        }
      }
      if (Math.abs(checkRight - priceColX) > PRICE_COLUMN_TOLERANCE_FACTOR * medH + extraTol) continue;
    }

    // Use the stripped (flag-free) price if that's how it matched
    let price = stripped ?? rawText.trim();

    // Absorb trailing suffix tokens that OCR may have split from the price.
    // Known suffixes: "-" (discount), single-letter tax flags (A, X, N, T, O, B, R, F),
    // or dash+flag combos like "-A".
    for (let j = i + 1; j < ordered.length && j <= i + 2; j++) {
      const suf = ordered[j].text.trim();
      if (!/^(-|[A-Z]|-[A-Z])$/i.test(suf)) break;

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
   Public entry point
   ═══════════════════════════════════════════════════════════════════════════ */

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

    // First, try Y-clustering to separate words from different physical lines.
    // The neighbor graph can merge adjacent receipt lines when line spacing is
    // tight; Y-clustering with the standard threshold catches these cases even
    // when the overall Y range is too small to trigger RANSAC.
    const yClusters = clusterIntoLines(rotated, medH);
    if (yClusters.length > 1) {
      console.debug(`[DEBUG]    Y-cluster split tentative line ${i} (${tentativeLines[i].length} words) into ${yClusters.length} sub-lines`);
      for (let s = 0; s < yClusters.length; s++) {
        console.debug(`[DEBUG]      Sub-line ${s}: ${yClusters[s].length} words → "${yClusters[s].map(w => w.text).join(' ')}"`);
      }
    }

    // Then, try RANSAC on each Y-cluster for any remaining merged lines
    for (const cluster of yClusters) {
      const splits = ransacSplitCluster(cluster, medH);
      if (splits.length > 1) {
        console.debug(`[DEBUG]    RANSAC split (post Y-cluster) line ${i} (${cluster.length} words) into ${splits.length} sub-lines`);
        for (let s = 0; s < splits.length; s++) {
          console.debug(`[DEBUG]      Sub-line ${s}: ${splits[s].length} words → "${splits[s].map(w => w.text).join(' ')}"`);
        }
      }
      for (const split of splits) {
        finalBuckets.push({ boxes: split, angle });
      }
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
  extractBarcodePrices(receiptLines);
  handleWrappedNames(receiptLines, medH);
  // Split combined TAX/BAL lines (e.g. "**** TAX .93 BAL 45.44")
  splitTaxBalLine(receiptLines);
  // Assign prices from adjacent info lines to priceless keyword lines
  // BEFORE mergeOrphanPrices, so keywords don't get wrong orphan prices
  assignAdjacentPricesToKeywords(receiptLines);
  reconstructBrokenTaxPrices(receiptLines);
  mergeOrphanPrices(receiptLines, medH);
  demotePostTotalItems(receiptLines);
  handleVoidedEntries(receiptLines);
  resolveCompetingTotals(receiptLines);
  fixKeywordPriceAssignment(receiptLines);
  rescueBarcodeItems(receiptLines);
  removeNullItemNameItems(receiptLines);

  console.debug(`[DEBUG] ── Final output: ${receiptLines.length} lines (excluding wrapped)`);
  for (const rl of receiptLines) {
    if (rl.lineType === 'wrapped') continue;
    console.debug(`[DEBUG]    [${rl.lineType.padEnd(12)}] ${(rl.itemName ?? rl.text).padEnd(35)} ${rl.price ?? ''}`);
  }
  console.debug(`[DEBUG] ══════════════════════════════════════════════════`);

  return { lines: receiptLines, angle: globalAngle };
}

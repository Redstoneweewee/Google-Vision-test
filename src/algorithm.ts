/**
 * Hybrid receipt-line reconstruction algorithm.
 *
 * Pipeline:
 *   1. Estimate global text rotation angle (histogram of pairwise angles)
 *   2. Rotate all boxes & cluster into lines by rotated-Y
 *   3. RANSAC-split any oversized cluster (perspective / curved receipts)
 *   4. Order words left→right inside each line (graph-style chaining)
 *   5. Detect the price column, assign prices, classify lines
 *   6. Handle wrapped item names & discount/tax lines
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
   Stage 1 – Estimate the dominant text angle
   ─────────────────────────────────────────────────────────────────────────
   For every pair of nearby, horizontally-separated word boxes we compute
   the angle of the vector between their centers.  A histogram (0.5° bins)
   reveals the peak near 0° which is the receipt's roll angle.  We ignore
   pairs that are too far apart vertically (> 3× median height) or too
   close horizontally (< 0.5× median height) to filter out cross-line and
   same-word noise.
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
   Stage 2 – Rotate boxes & cluster into lines by rotated-Y
   ─────────────────────────────────────────────────────────────────────────
   After rotating all centres by −θ the text becomes roughly horizontal.
   We sweep the boxes sorted by rotated-Y and greedily assign each box to
   the cluster whose mean-Y is closest, provided the distance is less than
   0.6 × median word height.  This threshold is tuned for receipt line
   spacing (thermal printers use ~1.2× text height between baselines, so
   0.6× puts the boundary right between two lines).
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
   Stage 3 – RANSAC split for messy (multi-row) clusters
   ─────────────────────────────────────────────────────────────────────────
   A cluster whose rotated-Y range exceeds 1.5 × median height likely
   merged two real lines (perspective distortion made them close).  We run
   RANSAC on the (x′, y′) centres: pick 2 random points, fit a line,
   count inliers within 0.4 × medH, keep the best set, remove them, and
   repeat until < 2 points remain.  Any singletons are attached to their
   nearest sub-line.
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
   Stage 4 – Order words left → right inside each line
   ─────────────────────────────────────────────────────────────────────────
   Simple sort by rotated left-edge.  For receipt lines (typically 3–8
   tokens) this is sufficient; a full graph-chain is overkill but the sort
   achieves the same result because there are no branching ambiguities on
   a single-column receipt.
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
   Stage 5 – Detect the price column & assign prices
   ─────────────────────────────────────────────────────────────────────────
   Most receipts right-align all prices in a single column.  We collect
   the rotated right-edge x of every price-like token and take the median;
   this is the "price column".  When assigning a price to a line, we walk
   tokens right-to-left and accept the first price-matching token whose
   right-edge is near the column (within 3 × median height).  This avoids
   picking up quantities like "2.00" from "2.00 kg" on the left side.
   ═══════════════════════════════════════════════════════════════════════════ */

function detectPriceColumnX(allLines: RotatedBox[][]): number | null {
  const priceRights: number[] = [];
  for (const line of allLines) {
    for (const box of line) {
      if (isPrice(box.text)) priceRights.push(box.rRight);
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

    // If a price column was detected, verify this token is near it
    if (priceColX !== null) {
      if (Math.abs(ordered[i].rRight - priceColX) > PRICE_COLUMN_TOLERANCE_FACTOR * medH) continue;
    }

    return { price: ordered[i].text, priceIndex: i };
  }
  return { price: null, priceIndex: -1 };
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
   Public entry point
   ═══════════════════════════════════════════════════════════════════════════ */

/**
 * Reconstruct ordered receipt lines from an unordered list of word-level
 * OCR annotations.  Pass `detections.slice(1)` (skip the full-text block).
 *
 * Returns one `ReceiptLine` per visual line, top-to-bottom.
 */
export function reconstructLines(
  annotations: TextAnnotation[],
): ReceiptLine[] {
  const boxes = annotations.map(annotationToWordBox);
  if (boxes.length === 0) return [];

  const medH = median(boxes.map((b) => b.height));

  // Estimate median character width (for fragment merging)
  const charW = median(
    boxes
      .map((b) => (b.text.length > 0 ? b.width / b.text.length : 0))
      .filter((w) => w > 0),
  );

  // ── Stage 1: global rotation ──────────────────────────────────────────
  const angle = estimateGlobalAngle(boxes);

  // ── Stage 2: rotate & cluster ─────────────────────────────────────────
  const rotated = rotateBoxes(boxes, angle);
  const rawLines = clusterIntoLines(rotated, medH);

  // ── Stage 3: RANSAC-split oversized clusters ──────────────────────────
  const splitLines: RotatedBox[][] = [];
  for (const cluster of rawLines) {
    splitLines.push(...ransacSplitCluster(cluster, medH));
  }

  // ── Stage 4: order words left → right ─────────────────────────────────
  const orderedLines = splitLines.map((line) => orderWordsInLine(line));

  // ── Detect the price column ───────────────────────────────────────────
  const priceColX = detectPriceColumnX(orderedLines);

  // ── Stage 5: build ReceiptLine objects ─────────────────────────────────
  const receiptLines: ReceiptLine[] = orderedLines.map((ordered) => {
    const text = buildLineText(ordered, charW);
    const { price, priceIndex } = assignPrice(ordered, priceColX, medH);

    // Item name = everything to the left of the price token
    const itemName =
      priceIndex >= 0
        ? buildLineText(ordered.slice(0, priceIndex), charW).trim()
        : text.trim();

    const lineType = classifyLine(text, price !== null);

    return {
      words: ordered,
      text,
      itemName: itemName || null,
      price,
      lineType,
    };
  });

  // ── Edge cases ────────────────────────────────────────────────────────
  handleWrappedNames(receiptLines, medH);

  return receiptLines;
}

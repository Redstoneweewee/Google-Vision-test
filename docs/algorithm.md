# Receipt Line Reconstruction Algorithm

## Overview

This algorithm takes unordered word-level OCR detections (text + bounding box) from Google Cloud Vision and reconstructs the original receipt lines — even when the image is slightly rotated, tilted, or has mild perspective distortion.

It uses a **hybrid approach** that combines three well-known techniques, applying them in order of increasing cost so the cheap method handles easy cases and the expensive method only fires when needed:

| Technique | When it runs | Cost |
|---|---|---|
| Global rotation + Y-clustering (Method A) | Always | O(n²) angle pairs, O(n log n) sort |
| RANSAC line fitting (Method B) | Only on oversized clusters | O(k · iterations) per cluster |
| Graph-style word chaining (Method C) | Within each line | O(n) per line |

---

## Pipeline

```
Raw OCR word boxes (unordered)
        │
        ▼
 ┌──────────────────────────┐
 │  Stage 1: Estimate       │  Pairwise angle histogram → dominant θ
 │  global rotation angle   │
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐
 │  Stage 2: Rotate all     │  Rotate by −θ, greedy sweep on
 │  boxes & cluster by Y'   │  rotated-Y to form line groups
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐
 │  Stage 3: RANSAC split   │  If a cluster's Y-range is too tall,
 │  oversized clusters      │  split it into sub-lines via RANSAC
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐
 │  Stage 4: Order words    │  Sort by rotated left edge,
 │  left → right            │  merge OCR fragments
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐
 │  Stage 5: Price column   │  Detect price column, assign prices,
 │  detection & assignment  │  classify lines semantically
 └──────────┬───────────────┘
            ▼
 ┌──────────────────────────┐
 │  Edge cases: Wrapped     │  Merge priceless continuation lines;
 │  names, discounts, tax   │  tag discount/tax/total lines
 └──────────┘───────────────┘
            ▼
      ReceiptLine[]
```

---

## Stage-by-Stage Detail

### Stage 1 — Estimate Global Rotation Angle

**Goal:** Find the single angle θ that represents the receipt's roll (the phone was held at a slight angle).

**Method:**
1. For every pair of word boxes that are horizontally separated (not too close, not too far) and vertically close, compute the angle of the vector between their centres.
2. Build a histogram of these angles using small bins (0.5°) and find the peak nearest 0°.
3. This peak is the dominant text direction.

**Why a histogram instead of PCA?** PCA would be pulled by outlier boxes (logos, barcodes, sideways text). A histogram with a locality constraint naturally ignores outliers because they don't form a peak.

**Why not skip straight to RANSAC?** Because RANSAC is iterative/stochastic and unnecessary for the common case. Most receipt photos have a single consistent tilt, so one global correction handles 90%+ of images.

### Stage 2 — Rotate & Cluster into Lines

**Goal:** Group word boxes into horizontal lines.

**Method:**
1. Rotate every box centre by −θ so text becomes roughly horizontal.
2. Sort all boxes by their rotated Y coordinate.
3. Walk through sorted boxes and assign each to the nearest cluster if the distance to the cluster's mean-Y is below a threshold (0.6 × median word height). Otherwise, start a new cluster.

**Why 0.6×?** Thermal printers use ~1.2× text height for line spacing. The threshold sits at exactly half the gap between two adjacent baselines — the tightest boundary that avoids merging consecutive lines while tolerating slight skew.

**Why greedy sweep, not DBSCAN?** Receipt lines are inherently 1D-separable in Y after rotation. The gaps are very consistent (fixed line spacing). A greedy sweep is O(n log n), deterministic, and has one intuitive parameter. DBSCAN would need epsilon + min_samples tuning and gives no ordering guarantee.

### Stage 3 — RANSAC Split for Perspective Distortion

**Goal:** Fix clusters that accidentally merged two lines due to perspective.

**When it runs:** Only if a cluster's rotated-Y range exceeds 1.5× median word height.

**Method:**
1. Pick 2 random box centres, fit a 2D line through them.
2. Count "inliers" — boxes whose perpendicular distance to the line is below 0.4× median height.
3. Repeat for up to 50 iterations, keep the best set of inliers.
4. Remove inliers from the cluster, repeat to find the next line.
5. Singletons get attached to their nearest sub-line.

**Why is the inlier threshold (0.4×) tighter than the clustering threshold (0.6×)?** Because at this point we know two lines are close together (that's why they merged). A tighter threshold ensures clean separation.

### Stage 4 — Order Words Left → Right

**Goal:** Produce the correct reading order within each line and merge OCR fragments.

**Method:**
1. Sort boxes by rotated left edge.
2. Walk left-to-right: if the gap between two consecutive boxes is less than 0.3× character width, concatenate without a space (handles fragments like `"$"` + `"4.99"` → `"$4.99"`). Otherwise insert a space.

**Why not a full graph chain?** Receipt lines are short (3–8 tokens). There are no branching ambiguities on a single-column receipt, so sort-by-X achieves the same result as a graph at O(n log n) instead of O(n²).

### Stage 5 — Price Column Detection & Assignment

**Goal:** Identify which token on each line is the price (not a quantity, date, or phone number).

**Method:**
1. Find all tokens matching the price regex (`$4.99`, `-$1.50`, `($3.00)`, etc.).
2. Collect their rotated right-edge X positions and take the median — this is the "price column".
3. For each line, walk tokens right-to-left. The first price-matching token whose right edge is near the column (within 3× median height) is the price. Everything to its left is the item name.

**Why detect a column?** Receipts universally right-align prices. Once you know the column's X position, you can distinguish `"4.50"` (a price at x=500) from `"2.00"` in `"2.00 KG CHICKEN"` (a quantity at x=120).

### Edge Cases

#### Wrapped Item Names
Long item descriptions wrap to multiple lines. Detection:
- A line has no price and is classified as `info`.
- The line immediately below it has a price.
- The vertical gap between them is small (< 1.5× median height).
- Their left edges are roughly aligned (< 3× median height apart).

When detected, the text from the wrapped line is prepended to the next line's `itemName`, and the wrapped line is tagged `"wrapped"` so it can be hidden from the final output.

**The algorithm iterates bottom-up** so chained wraps (2+ continuation lines for one item) cascade correctly.

#### Discount / Tax / Total Lines
Classified by keyword matching on the full line text. Check order matters — `"subtotal"` is checked before `"total"` to avoid `"SUBTOTAL"` being tagged as a `total` line.

Keywords recognized:
- **Subtotal:** subtotal, sub-total, sub total
- **Total:** total, amount due, balance, grand total
- **Tax:** tax, hst, gst, pst, vat
- **Discount:** disc, discount, off, save, savings, coupon, promo

---

## Constants Reference

All magic numbers live in `src/constants.ts`. Here's every constant, its current value, which stage uses it, and what happens if you change it.

### Stage 1 — Rotation Estimation

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `ANGLE_PAIR_MAX_DY_FACTOR` | `3` | Accepts pairs spanning more lines vertically → more angle samples but noisier | Stricter: only same-line pairs → fewer samples but more precise |
| `ANGLE_PAIR_MIN_DX_FACTOR` | `0.5` | Requires wider horizontal separation → rejects short-range noise but loses close pairs | Accepts very close pairs → more samples but OCR jitter dominates |
| `ANGLE_PAIR_MAX_DX_FACTOR` | `10` | Accepts very far-apart pairs → may cross receipt halves and pick up curvature | Only nearby pairs → less noise, but fewer samples on sparse receipts |
| `ANGLE_MAX_ABS_RAD` | `π/6` (30°) | Allows steeper angles into the histogram → tolerant of heavily rotated receipts | Only near-horizontal angles → cleaner histogram, fails on very rotated images |
| `ANGLE_HISTOGRAM_BIN_DEG` | `0.5` | Wider bins → smoother histogram but lower angular precision | Finer bins → more precision but the peak may fragment across bins |

### Stage 2 — Line Clustering

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `LINE_CLUSTER_Y_THRESHOLD_FACTOR` | `0.6` | More forgiving: merges lines that are close together → risk merging two real lines | Stricter: may split one real line into two clusters (especially with mixed font sizes) |

**Tuning guidance:** If you see lines being incorrectly merged (two items on one line), decrease toward `0.4`. If you see half-lines appearing (one item split across two output lines), increase toward `0.8`.

### Stage 3 — RANSAC Splitting

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `RANSAC_SPLIT_THRESHOLD_FACTOR` | `1.5` | Tolerates taller clusters before splitting → fewer RANSAC runs, but may miss real merges | Triggers RANSAC more aggressively → catches subtle merges, but may needlessly split valid lines |
| `RANSAC_MIN_CLUSTER_SIZE` | `3` | Requires more boxes to attempt RANSAC → skips tiny clusters (safe, fast) | Would try RANSAC on 2-box clusters → usually pointless |
| `RANSAC_INLIER_DIST_FACTOR` | `0.4` | More permissive: counts boxes farther from the line as inliers → may lump two close lines | Tighter: cleaner separation, but may leave too many outliers unattached |
| `RANSAC_MAX_ITERATIONS` | `50` | More attempts → higher probability of finding the optimal line, slower | Fewer attempts → faster, but may miss the best line on larger clusters |
| `RANSAC_MIN_INLIERS` | `2` | Requires more support for a valid line → more conservative | Would accept single-point "lines" → meaningless |

### Stage 4 — Fragment Merging

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `FRAGMENT_MERGE_GAP_FACTOR` | `0.3` | Merges boxes with larger gaps → may incorrectly merge separate words | Only merges very tight fragments → may leave `$` and `4.99` as separate words |

### Stage 5 — Price Assignment

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `PRICE_COLUMN_MIN_PRICES` | `2` | Needs more prices to establish a column → more robust but may miss single-item receipts | Accept a single price as a "column" → fragile, the column position is just one data point |
| `PRICE_COLUMN_TOLERANCE_FACTOR` | `3` | Accepts prices further from the detected column → tolerates perspective spread, but may pick up stray numbers | Strict column check → precise but may reject legitimate prices that are slightly offset |

### Edge Cases — Wrapped Names

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `WRAP_MAX_VERTICAL_GAP_FACTOR` | `1.5` | Allows wrapping across larger vertical gaps → catches widely-spaced continuations, but may falsely merge unrelated lines | Only merges lines that are immediately adjacent → misses wraps with extra padding |
| `WRAP_MAX_LEFT_ALIGN_FACTOR` | `3` | Permits significant horizontal offset → catches indented continuations, but may match unrelated lines | Requires near-exact left alignment → misses indented or slightly shifted continuations |

---

## File Structure

```
src/
├── constants.ts   ← All tunable magic numbers (this doc's "Constants Reference")
├── utils.ts       ← Shared types (WordBox, ReceiptLine, Point, LineType)
│                     + pure helpers (median, rotatePoint, angleBetween,
│                       isPrice, classifyLine, annotationToWordBox)
├── algorithm.ts   ← The 5-stage pipeline (reconstructLines entry point)
└── detect.ts      ← Google Vision API calls, console output, image annotation
```

### How to modify

- **Change a threshold:** Edit the value in `constants.ts`. The JSDoc on each constant tells you exactly which function uses it.
- **Add a new keyword category:** Add the keywords array in `utils.ts` and add the new `LineType` variant + check in `classifyLine`.
- **Change the price regex:** Edit `PRICE_REGEX` in `utils.ts`. The regex is intentionally strict (requires exactly 2 decimal places) to avoid matching dates, phone numbers, and zip codes.
- **Skip RANSAC entirely:** Set `RANSAC_SPLIT_THRESHOLD_FACTOR` to `Infinity` — clusters will never trigger the split.
- **Disable wrapped-name merging:** Set `WRAP_MAX_VERTICAL_GAP_FACTOR` to `0`.

---

## Limitations

1. **Single-receipt only.** The algorithm assumes one receipt per image. Two side-by-side receipts may interleave lines. (Could be addressed by detecting x-bimodality in box positions.)
2. **Single price column.** Receipts with multiple aligned price columns (unit price + total) will only detect one column. The rightmost price is always chosen.
3. **No full perspective rectification.** The rotation correction handles roll well. Mild pitch/yaw is handled by RANSAC splitting. Severe perspective (e.g. photographing at a 45° angle) would need a full projective transform, which is out of scope.
4. **Keyword-dependent classification.** Discount/tax/total detection relies on English keywords. For multilingual support, extend the keyword arrays in `utils.ts`.

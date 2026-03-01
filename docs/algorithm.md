# Receipt Line Reconstruction Algorithm

## Overview

This algorithm takes unordered word-level OCR detections (text + bounding box) from Google Cloud Vision and reconstructs the original receipt lines — even when the image is slightly rotated, tilted, folded, or has mild perspective distortion.

It uses **Method 3 (line-first → per-line angles)**, which builds tentative lines from local word adjacency and then computes a stable baseline angle per line. This makes it inherently robust to receipt folds and local curvature, where a single global rotation angle would cause prices at the far right to drift to the wrong line.

| Technique | When it runs | Cost |
|---|---|---|
| Neighbor-graph word chaining (union-find) | Always — Stage 1a | O(n²) pairwise checks |
| Least-squares per-line angle | Always — Stage 1b | O(k) per line |
| Global angle histogram (fallback) | Only for singleton words — Stage 1c | O(n²) angle pairs |
| RANSAC line fitting | Only on oversized tentative lines — Stage 2 | O(k · iterations) per line |
| Price-column median | Always — Stage 4 | O(n) scan |

### Why Method 3?

Four methods were considered:

1. **Angle-first clustering** — cluster words by their individual orientation, then line-cluster within each group. Fails because per-word angles from OCR bounding boxes are extremely noisy (a 1px vertex error on a 20px-wide word produces a ~3° error).

2. **Spatially-aware orientation clustering** — adds position to the clustering metric. Too many parameters to tune; still starts from noisy per-word angles.

3. **Line-first → per-line angles (chosen)** — builds tentative lines using local adjacency (y-overlap of nearby words), then derives a stable angle per line via least-squares fit through word centres. This works because: (a) adjacent words on the same line always have high y-overlap regardless of distant folds (the y-shift between adjacent words = Δx × sin θ, which is tiny for small Δx); (b) the union-find chain structure bridges left-to-right across the full line without needing to directly connect far-apart words; (c) per-line angles averaged over many points are far more stable than per-word angles.

4. **Orientation field** — models angle as a smooth function θ(y) of vertical position. Elegant for continuous curvature but complex to implement and debug; overkill for receipt folds where each line is still locally straight.

---

## Pipeline

```
Raw OCR word boxes (unordered)
        │
        ▼
 ┌──────────────────────────────┐
 │  Stage 1a: Neighbor graph    │  For each box, find nearest right
 │  → tentative lines           │  neighbor with y-overlap; union-find
 └──────────┬───────────────────┘  → connected components = lines
            ▼
 ┌──────────────────────────────┐
 │  Stage 1b: Per-line angle    │  Least-squares fit through word
 │  estimation                  │  centres → baseline angle per line
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │  Stage 1c: Global angle      │  Pairwise angle histogram → fallback
 │  fallback for singletons     │  θ for single-word lines
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │  Stage 2: Per-line rotation  │  Rotate each line by its own angle;
 │  + RANSAC split              │  RANSAC-split if Y-range too large
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │  Stage 3: Order words        │  Sort by rotated left edge,
 │  left → right                │  merge OCR fragments
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │  Stage 4: Price column       │  Detect price column, assign prices,
 │  detection & assignment      │  classify lines semantically
 └──────────┬───────────────────┘
            ▼
 ┌──────────────────────────────┐
 │  Stage 5: Sort, wrap, edge   │  Sort top→bottom by original Y;
 │  cases                       │  merge wrapped names; tag tax/total
 └──────────┘───────────────────┘
            ▼
      ReceiptLine[]
      (each with its own angle)
```

---

## Stage-by-Stage Detail

### Stage 1a — Build Tentative Lines via Neighbor Graph

**Goal:** Group words into tentative lines using purely local spatial relationships, without needing any global rotation estimate.

**Method:**
1. For every word box A, scan all other boxes to find A's **nearest right neighbor** B — the box with the smallest center-x distance among candidates satisfying:
   - B's center is to the right of A's center.
   - The vertical overlap between A and B (as a fraction of the smaller box's height) exceeds 30% (`NEIGHBOR_Y_OVERLAP_MIN`).
   - The edge-to-edge horizontal gap is at most 8× median word height (`NEIGHBOR_MAX_X_GAP_FACTOR`).
   - The height ratio (taller / shorter) is below 2.0 (`NEIGHBOR_MAX_HEIGHT_RATIO`).
2. When a valid nearest-right neighbor is found, **union** A and B in a union-find structure (with path compression and union by rank).
3. After processing all boxes, extract connected components. Each component is a tentative line, sorted left-to-right by center x.

**Why union-find instead of greedy chains?** Greedy chaining (A→B→C→D) can fail when multiple boxes claim the same "next" neighbor. Union-find handles many-to-one connections naturally: if both A and C point to B, all three end up in the same component, which is correct.

**Why only connect to the nearest right neighbor?** Connecting all valid pairs would over-link boxes on adjacent lines that happen to have slight y-overlap. Restricting to the nearest right neighbor builds a sparse, chain-like graph that follows the reading direction.

**Why does this work on folded receipts?** A fold changes the baseline angle across the receipt but doesn't move adjacent words apart. Two words sitting next to each other on the same physical line always have high y-overlap because their y-shift is Δx × sin(angle), and Δx is small for adjacent words. The chain of connections (A→B→C→…→price) bridges from the left side of the line to the right side transitively, even though the first and last words might have little y-overlap directly.

### Stage 1b — Per-Line Angle Estimation

**Goal:** Compute a stable baseline angle for each tentative line.

**Method:**
1. For tentative lines with ≥ 2 words, compute the ordinary least-squares regression line through the word centres.
2. The angle is `atan2(Σ(dx·dy), Σ(dx²))` where dx and dy are deviations from the centroid.
3. Lines with a single word return 0 (sentinel for "use the fallback angle from Stage 1c").

**Why least-squares instead of endpoint-to-endpoint?** For long lines with many words, least-squares averages over all points and is robust to one or two outliers (e.g. a word with a bad bounding box). Endpoint-to-endpoint depends on exactly two words and is sensitive to OCR jitter.

### Stage 1c — Global Angle Fallback

**Goal:** Provide a reasonable angle for single-word tentative lines that can't compute their own angle.

**Method (retained from original pipeline):**
1. For every pair of word boxes that are horizontally separated and vertically close, compute the angle of the vector between their centres.
2. Build a histogram of these angles using small bins (0.5°) and find the peak nearest 0°.
3. If no multi-word tentative lines exist, this global angle is the only estimate available.

**Resolution:** The median of all valid per-line angles (from Stage 1b) is used if available. The pairwise histogram (`estimateGlobalAngle`) fires only when no multi-word lines exist at all — an edge case for very sparse receipts.

### Stage 2 — Per-Line Rotation + RANSAC Split

**Goal:** Rotate each tentative line's boxes into a locally de-skewed coordinate system and split any that accidentally merged two physical lines.

**Method:**
1. For each tentative line, rotate its boxes by −θ_line (the line's own angle from Stage 1b/1c).
2. If the resulting rotated-Y range exceeds 1.5× median word height, the neighbor graph accidentally merged two close lines. Run RANSAC to split:
   - Pick 2 random box centres, fit a 2D line through them.
   - Count "inliers" — boxes whose perpendicular distance is below 0.4× median height.
   - Repeat for up to 50 iterations, keep the best inlier set.
   - Remove inliers, repeat to find more sub-lines.
   - Singletons attach to the nearest sub-line.

**Why is the RANSAC inlier threshold (0.4×) tighter than the original Y-clustering threshold (0.6×)?** Because at this point we know two lines are close together (that's why the neighbor graph merged them). A tighter threshold ensures clean separation.

**How do per-line angles interact with RANSAC?** After RANSAC splits a tentative line, each sub-line inherits the parent's angle. This is reasonable because the parent's angle was computed from a mixture of two close lines that are nearly parallel — the inherited angle is close enough for both.

### Stage 3 — Order Words Left → Right

**Goal:** Produce the correct reading order within each line and merge OCR fragments.

**Method:**
1. Sort boxes by rotated left edge (using that line's own angle).
2. Walk left-to-right: if the gap between two consecutive boxes is less than 0.3× character width, concatenate without a space (handles fragments like `"$"` + `"4.99"` → `"$4.99"`). Otherwise insert a space.

### Stage 4 — Price Column Detection & Assignment

**Goal:** Identify which token on each line is the price (not a quantity, date, or phone number).

**Method:**
1. Find all tokens matching the price regex (`$4.99`, `-$1.50`, `($3.00)`, etc.).
2. Collect their **rotated** right-edge X positions and take the median — this is the "price column".
3. For each line, walk tokens right-to-left. The first price-matching token whose right edge is near the column (within 3× median height) is the price. Everything to its left is the item name.

**Why detect a column?** Receipts universally right-align prices. Once you know the column's X position, you can distinguish `"4.50"` (a price at x=500) from `"2.00"` in `"2.00 KG CHICKEN"` (a quantity at x=120).

**Don't per-line angles break cross-line price column detection?** No. The `rRight` values are rotated by slightly different per-line angles, but the tolerance (`PRICE_COLUMN_TOLERANCE_FACTOR = 3`, i.e. 3× median height ≈ 60px) easily absorbs the spread. For angle differences of a few degrees across lines, the resulting `rRight` spread is ~25px — well within tolerance.

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

### Stage 1a — Neighbor Graph (Tentative Lines)

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `NEIGHBOR_Y_OVERLAP_MIN` | `0.3` | Stricter: requires more vertical overlap to connect two boxes → safer but may disconnect words on skewed lines with large gaps | More lenient: connects boxes with less overlap → may link boxes on adjacent lines |
| `NEIGHBOR_MAX_X_GAP_FACTOR` | `8` | Bridges larger horizontal gaps → reliably connects items to far-right prices, but may link boxes across lines on narrow receipts | Only connects nearby words → may leave prices as orphaned singletons |
| `NEIGHBOR_MAX_HEIGHT_RATIO` | `2.0` | Tolerates larger font-size variation → connects bold totals and normal items, but may link headers/logos | Stricter: only connects similarly-sized text → may split lines with mixed font sizes |

**Tuning guidance:** If prices appear as separate lines (not connected to their item), increase `NEIGHBOR_MAX_X_GAP_FACTOR`. If two adjacent lines are merging, increase `NEIGHBOR_Y_OVERLAP_MIN` toward `0.5`.

### Stage 1c — Global Angle Fallback

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `ANGLE_PAIR_MAX_DY_FACTOR` | `3` | Accepts pairs spanning more lines vertically → more angle samples but noisier | Stricter: only same-line pairs → fewer samples but more precise |
| `ANGLE_PAIR_MIN_DX_FACTOR` | `0.5` | Requires wider horizontal separation → rejects short-range noise but loses close pairs | Accepts very close pairs → more samples but OCR jitter dominates |
| `ANGLE_PAIR_MAX_DX_FACTOR` | `10` | Accepts very far-apart pairs → may cross receipt halves and pick up curvature | Only nearby pairs → less noise, but fewer samples on sparse receipts |
| `ANGLE_MAX_ABS_RAD` | `π/6` (30°) | Allows steeper angles into the histogram → tolerant of heavily rotated receipts | Only near-horizontal angles → cleaner histogram, fails on very rotated images |
| `ANGLE_HISTOGRAM_BIN_DEG` | `0.5` | Wider bins → smoother histogram but lower angular precision | Finer bins → more precision but the peak may fragment across bins |

> **Note:** These constants only affect the fallback path used for single-word tentative lines. On receipts with multiple multi-word lines, they have no effect.

### Stage 2 — RANSAC Splitting

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `RANSAC_SPLIT_THRESHOLD_FACTOR` | `1.5` | Tolerates taller clusters before splitting → fewer RANSAC runs, but may miss real merges | Triggers RANSAC more aggressively → catches subtle merges, but may needlessly split valid lines |
| `RANSAC_MIN_CLUSTER_SIZE` | `3` | Requires more boxes to attempt RANSAC → skips tiny clusters (safe, fast) | Would try RANSAC on 2-box clusters → usually pointless |
| `RANSAC_INLIER_DIST_FACTOR` | `0.4` | More permissive: counts boxes farther from the line as inliers → may lump two close lines | Tighter: cleaner separation, but may leave too many outliers unattached |
| `RANSAC_MAX_ITERATIONS` | `50` | More attempts → higher probability of finding the optimal line, slower | Fewer attempts → faster, but may miss the best line on larger clusters |
| `RANSAC_MIN_INLIERS` | `2` | Requires more support for a valid line → more conservative | Would accept single-point "lines" → meaningless |

### Stage 2 (retained) — Y-Clustering

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `LINE_CLUSTER_Y_THRESHOLD_FACTOR` | `0.6` | More forgiving: merges lines that are close together → risk merging two real lines | Stricter: may split one real line into two clusters (especially with mixed font sizes) |

> **Note:** `clusterIntoLines` is retained in the code but is not called in the main pipeline. It serves as a fallback building block for the RANSAC split path.

### Stage 3 — Fragment Merging

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `FRAGMENT_MERGE_GAP_FACTOR` | `0.3` | Merges boxes with larger gaps → may incorrectly merge separate words | Only merges very tight fragments → may leave `$` and `4.99` as separate words |

### Stage 4 — Price Assignment

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `PRICE_COLUMN_MIN_PRICES` | `2` | Needs more prices to establish a column → more robust but may miss single-item receipts | Accept a single price as a "column" → fragile, the column position is just one data point |
| `PRICE_COLUMN_TOLERANCE_FACTOR` | `3` | Accepts prices further from the detected column → tolerates perspective spread + per-line angle variation, but may pick up stray numbers | Strict column check → precise but may reject legitimate prices that are slightly offset |

### Stage 5 — Wrapped Names

| Constant | Value | Effect of increasing | Effect of decreasing |
|---|---|---|---|
| `WRAP_MAX_VERTICAL_GAP_FACTOR` | `1.5` | Allows wrapping across larger vertical gaps → catches widely-spaced continuations, but may falsely merge unrelated lines | Only merges lines that are immediately adjacent → misses wraps with extra padding |
| `WRAP_MAX_LEFT_ALIGN_FACTOR` | `3` | Permits significant horizontal offset → catches indented continuations, but may match unrelated lines | Requires near-exact left alignment → misses indented or slightly shifted continuations |

---

## File Structure

```
src/
├── constants.ts   ← All tunable magic numbers (this doc's "Constants Reference")
├── utils.ts       ← Shared types (WordBox, ReceiptLine with per-line angle,
│                     Point, LineType) + pure helpers (median, rotatePoint,
│                     angleBetween, isPrice, classifyLine, annotationToWordBox)
├── algorithm.ts   ← Method 3 pipeline: neighbor graph → per-line angles →
│                     RANSAC → price column (reconstructLines entry point)
│                     Includes UnionFind, buildTentativeLines, computeLineAngle,
│                     estimateGlobalAngle (fallback), rotateBoxes, ransacSplitCluster
└── detect.ts      ← Google Vision API calls, console output, image annotation
                      with per-line rotated bounding boxes
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
3. **No full perspective rectification.** Per-line angle correction handles roll and folds well. Mild pitch/yaw is handled by RANSAC splitting. Severe perspective (e.g. photographing at a 45° angle) would need a full projective transform, which is out of scope.
4. **Keyword-dependent classification.** Discount/tax/total detection relies on English keywords. For multilingual support, extend the keyword arrays in `utils.ts`.
5. **O(n²) neighbor search.** The `buildTentativeLines` function scans all pairs to find nearest-right neighbors. For typical receipts (~50–200 words) this is instant, but very large documents might benefit from a spatial index.

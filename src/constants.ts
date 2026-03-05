/**
 * All tunable numeric constants used by the receipt-line reconstruction
 * algorithm, gathered in one place for easy tweaking and documentation.
 *
 * Every constant is named after THE STAGE + ROLE it plays, and the
 * accompanying JSDoc explains *why* that particular value was chosen.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Stage 1 – Global rotation estimation
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `estimateGlobalAngle` – filtering word pairs.
 *
 * Maximum vertical distance (in multiples of median word height) between
 * two box centres for them to be considered "on roughly the same line".
 * Set to 3× so we accept pairs that span up to ~3 text lines vertically;
 * this is generous enough to capture the dominant angle even on receipts
 * with large line gaps, while still excluding extreme cross-receipt pairs
 * (e.g. top of the header to the bottom total).
 */
export const ANGLE_PAIR_MAX_DY_FACTOR = 3;

/**
 * **Used in:** `estimateGlobalAngle` – filtering word pairs.
 *
 * Minimum horizontal distance (in multiples of median word height) between
 * two box centres.  Pairs that are too close horizontally give noisy angles
 * because a few pixels of OCR jitter dominate.  0.5× median height ≈ one
 * character width on a typical receipt, which is a safe lower bound.
 */
export const ANGLE_PAIR_MIN_DX_FACTOR = 0.5;

/**
 * **Used in:** `estimateGlobalAngle` – filtering word pairs.
 *
 * Maximum horizontal distance (in multiples of median word height).
 * Very far-apart pairs (other side of the receipt) are more likely to be
 * on different lines, so their angle is misleading.  10× keeps pairs
 * within a comfortable neighbourhood.
 */
export const ANGLE_PAIR_MAX_DX_FACTOR = 10;

/**
 * **Used in:** `estimateGlobalAngle` – angle pre-filter.
 *
 * Maximum absolute angle (radians) to keep when building the histogram.
 * π/6 = 30°.  Any pair whose angle exceeds ±30° is almost certainly
 * cross-line noise (vertical text, logos, etc.) and would pollute the
 * histogram peak.
 */
export const ANGLE_MAX_ABS_RAD = Math.PI / 6;

/**
 * **Used in:** `estimateGlobalAngle` – histogram binning.
 *
 * Width of each histogram bin in degrees.  0.5° gives sub-degree
 * resolution, which is fine for receipt skew (rarely > 10°), while
 * keeping the bin count small (~120 bins across ±30°).  Finer bins would
 * fragment the peak; coarser bins would lose precision.
 */
export const ANGLE_HISTOGRAM_BIN_DEG = 0.5;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 2 – Line clustering
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `clusterIntoLines` – assigning a box to an existing cluster.
 *
 * Maximum distance between a box's rotated-Y centre and a cluster's mean
 * rotated-Y, expressed as a fraction of median word height.  Thermal
 * printers typically use ~1.2× text height for baseline-to-baseline
 * spacing, so 0.6× sits exactly at the midpoint between two adjacent
 * baselines — the tightest boundary that still avoids merging consecutive
 * lines.  Increase toward 0.8 if fonts are very uneven; decrease toward
 * 0.4 if the receipt uses tight single-spacing.
 */
export const LINE_CLUSTER_Y_THRESHOLD_FACTOR = 0.6;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 3 – RANSAC cluster splitting
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `ransacSplitCluster` – deciding whether a cluster needs
 * splitting at all.
 *
 * If a cluster's rotated-Y range (max − min) exceeds this factor × median
 * height, we assume two or more real text lines were merged and invoke
 * RANSAC.  1.5× is slightly above one full text height; a single
 * correctly-clustered line should never span more than ~1× height, so
 * 1.5× gives a comfortable margin before triggering the more expensive
 * RANSAC path.
 */
export const RANSAC_SPLIT_THRESHOLD_FACTOR = 1.5;

/**
 * **Used in:** `ransacSplitCluster` – minimum cluster size to attempt
 * RANSAC.
 *
 * With fewer than 3 boxes there aren't enough points for a meaningful
 * line fit, so we skip RANSAC and keep the cluster as-is.
 */
export const RANSAC_MIN_CLUSTER_SIZE = 3;

/**
 * **Used in:** `ransacSplitCluster` – counting inliers for a candidate
 * line.
 *
 * Maximum perpendicular distance (in multiples of median word height) from
 * a box centre to the candidate RANSAC line for the box to be counted as
 * an inlier.  0.4× is tighter than the clustering threshold (0.6×)
 * because we want RANSAC to cleanly separate two lines that are already
 * known to be close together.
 */
export const RANSAC_INLIER_DIST_FACTOR = 0.4;

/**
 * **Used in:** `ransacSplitCluster` – iteration budget.
 *
 * Number of random 2-point samples to try per RANSAC round.  50 is
 * plenty for small clusters (5–15 boxes); the probability of missing the
 * correct line after 50 draws is negligible when the inlier ratio is
 * typically > 40%.
 */
export const RANSAC_MAX_ITERATIONS = 50;

/**
 * **Used in:** `ransacSplitCluster` – minimum inlier count for a valid
 * RANSAC model.
 *
 * If the best model has fewer than this many inliers, we stop splitting
 * and attach remaining points to the nearest existing sub-line.  Set to
 * 2 because a "line" with a single point is meaningless.
 */
export const RANSAC_MIN_INLIERS = 2;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 4 – Intra-line word ordering / fragment merging
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `buildLineText` – deciding whether to insert a space
 * between two adjacent word boxes.
 *
 * If the horizontal gap between the right edge of box[i-1] and the left
 * edge of box[i] is smaller than this fraction of the estimated character
 * width, the two boxes are treated as fragments of one token and
 * concatenated without a space (e.g. "$" + "4.99" → "$4.99").  0.3× is
 * just under a third of a character; typical inter-word spacing is
 * ≥ 0.5×, so this cleanly separates fragments from true word gaps.
 */
export const FRAGMENT_MERGE_GAP_FACTOR = 0.3;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 5 – Price column detection & price assignment
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `detectPriceColumnX` – deciding whether enough prices were
 * found to establish a column.
 *
 * We need at least this many price-like tokens on the receipt before we
 * trust the median right-edge as a reliable "price column" position.
 * With only 1 price, the column is just a guess; with 2+ it's a trend.
 */
export const PRICE_COLUMN_MIN_PRICES = 2;

/**
 * **Used in:** `assignPrice` – verifying a price candidate belongs to the
 * price column instead of being a stray number (quantity, date, etc.).
 *
 * Maximum distance (in multiples of median word height) between a price
 * token's right edge and the detected price-column X position.  3× is
 * generous enough to accommodate slight misalignment from perspective
 * or font-size variation, while still rejecting far-left numeric tokens
 * like quantities ("2.00 kg CHICKEN").
 */
export const PRICE_COLUMN_TOLERANCE_FACTOR = 3;

// ═══════════════════════════════════════════════════════════════════════════
// Edge case – Wrapped item names
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `handleWrappedNames` – vertical gap check.
 *
 * Maximum vertical distance (in multiples of median word height) between
 * the bottom of a priceless "info" line and the top of the next priced
 * line for them to be considered a continuation.  1.5× allows for
 * slightly larger-than-normal gaps while excluding lines that are a full
 * blank line apart.
 */
export const WRAP_MAX_VERTICAL_GAP_FACTOR = 1.8;

/**
 * **Used in:** `handleWrappedNames` – left-alignment check.
 *
 * Maximum horizontal difference (in multiples of median word height)
 * between the left edges of two candidate wrapped lines.  3× is
 * permissive because indentation varies; if you tighten this you'll miss
 * slightly indented continuation lines.
 */
export const WRAP_MAX_LEFT_ALIGN_FACTOR = 3;

// ═══════════════════════════════════════════════════════════════════════════
// Stage 1b – Neighbor-graph tentative line building (Method 3)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `buildTentativeLines` – connecting two word boxes.
 *
 * Minimum vertical overlap between two boxes, expressed as a fraction of
 * the smaller box's height, for them to be considered "on the same line".
 * 0.3 (30%) is strict enough to reject boxes on adjacent lines (which
 * typically have zero overlap because line spacing > text height) while
 * tolerating slight skew between adjacent words.  For two words on the
 * same line separated by a small x-gap, the y-shift due to rotation is
 * tiny (Δy = Δx × sin θ ≈ a few pixels), so overlap stays well above 30%.
 */
export const NEIGHBOR_Y_OVERLAP_MIN = 0.6;

/**
 * **Used in:** `buildTentativeLines` – maximum horizontal gap.
 *
 * Maximum gap (in multiples of median word height) between the right edge
 * of one box and the left edge of the next for them to be linked as
 * neighbors.  8× is generous enough to bridge the gap between an item
 * name and its far-right price on a typical receipt (item-to-price gaps
 * are often 5–10× text height).  The y-overlap constraint prevents this
 * from accidentally linking boxes on different lines.
 */
export const NEIGHBOR_MAX_X_GAP_FACTOR = 8;

/**
 * **Used in:** `buildTentativeLines` – height-similarity filter.
 *
 * Maximum ratio of the taller box's height to the shorter box's height.
 * 2.0 allows moderate font-size variation (e.g. bold totals vs. normal
 * items) while rejecting connections between normal text and large headers
 * or logos.  Tighten to 1.5 if you're getting false connections to headers.
 */
export const NEIGHBOR_MAX_HEIGHT_RATIO = 2.0;

// ═══════════════════════════════════════════════════════════════════════════
// Edge case – Orphan price search radius
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `mergeOrphanPrices`, `mergeOrphanItemPrices` – how many
 * lines above/below to scan when looking for an orphan price.
 *
 * On distorted or curved receipts, the keyword lines (SUBTOTAL, TAX,
 * TOTAL) and their prices may each be separate lines, with all keywords
 * stacked above their corresponding prices.  In the worst observed case
 * there are 3 keywords followed by 3 prices, plus occasional noise lines
 * ("-", "=", etc.) interleaved.  A radius of 4 covers:
 *   - 3 consecutive keyword lines (SUBTOTAL → skip TAX, TOTAL → reach 73.60)
 *   - 1 extra for noise lines that sometimes appear between keywords and prices
 *
 * Increasing beyond 4 risks matching unrelated prices from the receipt
 * body or footer.  The vertical-gap check (WRAP_MAX_VERTICAL_GAP_FACTOR)
 * provides an additional safety net.
 */
export const ORPHAN_SEARCH_RADIUS = 4;

// ═══════════════════════════════════════════════════════════════════════════
// Image preprocessing
// ═══════════════════════════════════════════════════════════════════════════

/**
 * **Used in:** `detectTextLocal` – image resizing before OCR.
 *
 * Maximum dimension (width or height) in pixels for the image sent to the
 * Vision API.  Images larger than this are downscaled (preserving aspect
 * ratio) before detection.  2048 px is a good trade-off between OCR
 * accuracy and API cost/latency.
 */
export const MAX_IMAGE_DIMENSION = 2048;


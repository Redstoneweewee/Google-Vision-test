# Google Vision API — Receipt OCR & Parser (TypeScript)

Uses the [Google Cloud Vision API](https://cloud.google.com/vision/docs/ocr) to detect text in receipt images and reconstruct a fully structured, line-classified receipt — including items, prices, discounts, tax, subtotal, and total — with a confidence report and an annotated output image.

---

## How It Works

1. **OCR** — Sends a local image to Google Cloud Vision; gets back unordered word-level bounding boxes.
2. **Line reconstruction** — A custom hybrid algorithm groups words into lines using a neighbor graph + union-find, then independently de-skews each line via least-squares angle fitting and RANSAC splitting. This is robust to folds, local curvature, and mild perspective distortion.
3. **Semantic classification** — Each line is classified as `untaxed_item`, `taxed_item`, `discount`, `tax`, `subtotal`, `total`, or `info`.
4. **Item building** — Discount lines are applied to the item immediately above them.
5. **Confidence checks** — A composable pipeline validates that `TOTAL ≈ SUBTOTAL + TAX`, taxability balance, tax-rate plausibility, and more.
6. **Annotated image** — Saves `<input>_annotated<ext>` next to the source image with per-line bounding boxes and labels overlaid.

---

## Prerequisites

- **Node.js** ≥ 18
- A **service account key** in `service-account-key.json`. Ask me for the key.

---

## Setup

### 1. Install dependencies

```bash
npm install
```

### 2. Configure credentials

Place the service account JSON file in the project root as `service-account-key.json` and set the environment variable:

```bash
# .env (change the .env.example file to .env)
GOOGLE_APPLICATION_CREDENTIALS=./service-account-key.json
```

---

## Usage

### Run with npm

```bash
# Uses ./sample.jpg by default
npm run start ./path_to_image.jpg
```

---

## Output

For a receipt image `receipt.jpg`, the tool prints to stdout:

```
=== Parsed Receipt ===
  [UNTAXED_ITEM]  Organic Whole Milk               $4.99
  [TAXED_ITEM]    Paper Towels 2pk                 $6.49
  [DISCOUNT]      Member savings                   -$1.00
  [SUBTOTAL]      Subtotal                         $10.48
  [TAX]           Tax                              $0.58
  [TOTAL]         Total                            $11.06

=== Items (with discounts applied) ===
  Paper Towels 2pk                    5.49   disc: -1.00 [T]
  Organic Whole Milk                  4.99

=== Summary ===
  Total lines:          6
  Total items:          2
  ...

=== Confidence Report ===
  [ PASS ] TOTAL = SUBTOTAL + TAX
  [ PASS ] Taxability balance
  ...
```

And saves an annotated image: `receipt_annotated.jpg`.

---

## Repository Structure

```
package.json
README.md
service-account-key.json    ← your GCP credentials (git-ignored)
tsconfig.json
docs/
    algorithm.md            ← detailed algorithm documentation
src/
    detect.ts               ← entry point; OCR, image annotation, CLI
    algorithm.ts            ← line reconstruction algorithm (Method 3)
    checks.ts               ← composable confidence-check pipeline
    constants.ts            ← all tunable algorithm parameters
    utils.ts                ← shared types and geometry helpers
```

---

## Algorithm

See [docs/algorithm.md](docs/algorithm.md) for a full stage-by-stage walkthrough of the line reconstruction pipeline, including the motivation for Method 3 over three alternative approaches.

**Pipeline summary:**

| Stage | What it does |
|-------|-------------|
| 1a | Neighbor-graph word chaining (union-find) → tentative lines |
| 1b | Least-squares per-line angle estimation |
| 1c | Global angle histogram fallback for singleton words |
| 2  | Per-line rotation + RANSAC split for oversized clusters |
| 3  | Sort words left → right, merge OCR fragments |
| 4  | Price-column detection, price assignment, semantic classification |
| 5  | Sort top → bottom, handle wrapped names and discount/tax edge cases |

---

## Tuning

All numeric thresholds are centralized in [`src/constants.ts`](src/constants.ts) with inline JSDoc explaining why each value was chosen. Adjust them there to improve results for different receipt formats or scan qualities.

## Output example

```
Detecting text in remote image: gs://cloud-samples-data/vision/ocr/sign.jpg

=== Full detected text ===
WAITING?
PLEASE TURN OFF
YOUR IDLE ENGINE

=== Individual word detections ===
Word 1: "WAITING?" — Bounding box: [(50, 30), (230, 30), (230, 70), (50, 70)]
...
```

---

## References

- [Vision API OCR docs](https://cloud.google.com/vision/docs/ocr)
- [@google-cloud/vision npm package](https://www.npmjs.com/package/@google-cloud/vision)
- [Authentication setup](https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)

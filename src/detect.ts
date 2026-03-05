import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import { reconstructLines } from './algorithm';
import type { ReconstructionResult } from './algorithm';
import type { ReceiptLine, Point, Receipt, ReceiptItem, ReceiptConfidence, LineType, WordBox } from './utils';
import { rotatePoint, parsePrice } from './utils';
import { checkConfidence, formatConfidenceReport, colorBold, colorDim, colorPass, colorWarn, colorError, colorCyan } from './checks';
import { MAX_IMAGE_DIMENSION } from './constants';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;
type Vertex = protos.google.cloud.vision.v1.IVertex;

// Creates a client using an API key from the GOOGLE_API_KEY environment variable.
const client = new vision.ImageAnnotatorClient({
  apiKey: process.env.GOOGLE_API_KEY,
});

interface LineAnnotation {
  description: string;
  lineType: LineType;
  boundingPoly: { vertices: Required<Vertex>[] };
}

/**
 * Builds a polygon for a line by tracing the bounding boxes of each word:
 *   - Top edges from left to right (TL → TR for each word)
 *   - Bottom edges from right to left (BR → BL for each word)
 *
 * This produces a tight outline that follows each word's actual OCR box
 * rather than a single merged rectangle.
 */
function buildLinePolygon(
  words: WordBox[],
): LineAnnotation['boundingPoly'] {
  if (words.length === 0) {
    const zero = { x: 0, y: 0 } as Required<Vertex>;
    return { vertices: [zero, zero, zero, zero] };
  }

  // Sort words left to right
  const sorted = [...words].sort((a, b) => a.left - b.left);

  const topPath: Required<Vertex>[] = [];
  const bottomReversed: Required<Vertex>[] = [];

  for (const word of sorted) {
    const verts = (word.original.boundingPoly?.vertices ?? []).map((v) => ({
      x: v.x ?? 0,
      y: v.y ?? 0,
    })) as Required<Vertex>[];

    if (verts.length >= 4) {
      // Vertices: [TL(0), TR(1), BR(2), BL(3)]
      topPath.push(verts[0], verts[1]);
      bottomReversed.push(verts[3], verts[2]);
    }
  }

  // Reverse the bottom path so it goes right → left
  bottomReversed.reverse();

  return { vertices: [...topPath, ...bottomReversed] };
}

/**
 * Produce a short label for the annotated-image overlay.
 */
function formatLineLabel(line: ReceiptLine): string {
  if (line.lineType === 'wrapped') return `\u21a9 ${line.text}`;
  if (line.price) {
    if (line.lineType === 'untaxed_item' || line.lineType === 'taxed_item')
      return `${line.itemName ?? ''} \u2192 ${line.price}`;
    return `[${line.lineType.toUpperCase()}] ${line.itemName ?? ''} ${line.price}`;
  }
  return line.text;
}

/**
 * Draws per-line bounding boxes and labels onto the image and saves it.
 */
async function saveAnnotatedImage(
  inputPath: string,
  imageBuffer: Buffer,
  lineAnnotations: LineAnnotation[]
): Promise<void> {
  const meta = await sharp(imageBuffer).metadata();
  const { width, height } = meta;

  const polygons = lineAnnotations
    .map(({ description, lineType, boundingPoly: { vertices } }) => {
      const pts = vertices.map((v) => `${v.x},${v.y}`).join(' ');
      const lx = vertices[0].x ?? 0;
      const ly = vertices[0].y ?? 0;
      const label = description
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
      const color = (lineType === 'info' || lineType === 'tender') ? '#888888' : '#00FF00';
      return `
      <polygon points="${pts}"
        fill="none" stroke="${color}" stroke-width="2"/>
      <text x="${lx}" y="${Math.max(ly - 4, 12)}"
        font-family="monospace" font-size="13"
        fill="${color}" stroke="black" stroke-width="0.6"
        paint-order="stroke">${label}</text>`;
    })
    .join('\n');

  const svg = `<svg xmlns="http://www.w3.org/2000/svg"
    width="${width}" height="${height}">
    ${polygons}
  </svg>`;

  const ext = path.extname(inputPath);
  const base = path.basename(inputPath, ext);
  const dir = path.dirname(inputPath);
  const outPath = path.join(dir, `${base}_annotated${ext}`);

  await sharp(imageBuffer)
    .composite([{ input: Buffer.from(svg), top: 0, left: 0 }])
    .toFile(outPath);

  console.log(`\nAnnotated image saved to: ${outPath}`);
}

/**
 * Build the list of ReceiptItems from classified lines.
 * Discount lines are applied to the item immediately above them.
 */
function buildItems(lines: ReceiptLine[]): ReceiptItem[] {
  const items: ReceiptItem[] = [];
  const visibleLines = lines.filter((l) => l.lineType !== 'wrapped');

  for (let i = 0; i < visibleLines.length; i++) {
    const line = visibleLines[i];

    if (line.lineType === 'untaxed_item' || line.lineType === 'taxed_item') {
      const originalPrice = Math.abs(parsePrice(line.price));
      items.push({
        name: line.itemName ?? line.text,
        originalPrice,
        discount: 0,
        finalPrice: originalPrice,
        taxed: line.lineType === 'taxed_item',
        rawPrice: line.price ?? '',
        rawDiscount: null,
      });
    } else if (line.lineType === 'discount') {
      const discountValue = -Math.abs(parsePrice(line.price)); // always negative
      if (items.length > 0) {
        const target = items[items.length - 1];
        target.discount += discountValue;
        target.finalPrice = target.originalPrice + target.discount;
        target.rawDiscount = target.rawDiscount
          ? target.rawDiscount + ', ' + (line.price ?? '')
          : (line.price ?? '');
      }
    }
  }

  return items;
}

/**
 * Detects text in a local image file, reconstructs receipt lines, and
 * returns a fully parsed Receipt object.  Also saves an annotated image.
 */
export async function detectTextLocal(filePath: string): Promise<Receipt> {
  const resolvedPath = path.resolve(filePath);

  if (!fs.existsSync(resolvedPath)) {
    throw new Error(`File not found: ${resolvedPath}`);
  }

  // ── Resize image if needed ──────────────────────────────────────────
  const imageBuffer = await sharp(resolvedPath)
    .resize({
      width: MAX_IMAGE_DIMENSION,
      height: MAX_IMAGE_DIMENSION,
      fit: 'inside',
      withoutEnlargement: true,
    })
    .toBuffer();

  const [result] = await client.textDetection({ image: { content: imageBuffer } });
  const detections = result.textAnnotations;

  if (!detections || detections.length === 0) {
    throw new Error('No text detected in image.');
  }

  console.log(`Full text: ${result.fullTextAnnotation?.text}\n\n`);
  // ── Reconstruct receipt lines ─────────────────────────────────────────
  const { lines: receiptLines, angle } = reconstructLines(detections.slice(1));

  // ── Build items (with discounts applied) ──────────────────────────────
  const items = buildItems(receiptLines);

  // ── Counts ────────────────────────────────────────────────────────────
  const totalLines = receiptLines.filter((l) => l.lineType !== 'wrapped').length;
  const untaxedItems = items.filter((i) => !i.taxed);
  const taxedItems = items.filter((i) => i.taxed);

  const untaxedItemsValue = untaxedItems.reduce((s, i) => s + i.finalPrice, 0);
  const taxedItemsValue = taxedItems.reduce((s, i) => s + i.finalPrice, 0);

  // ── OCR summary values ────────────────────────────────────────────────
  const findOcrValue = (type: string): number | null => {
    const line = receiptLines.find(
      (l) => l.lineType === type && l.price !== null,
    );
    return line ? parsePrice(line.price) : null;
  };

  // Use the LAST subtotal (some receipts have intermediate subtotals)
  const ocrSubtotal = (() => {
    const subs = receiptLines.filter((l) => l.lineType === 'subtotal' && l.price !== null);
    return subs.length > 0 ? parsePrice(subs[subs.length - 1].price) : null;
  })();

  // Sum tax lines, but detect breakdown vs total pattern.
  // Costco: "TAX 3.52" (total) + "A 8.50% TAX 0.55" + "E 3.75% TAX 2.97" (breakdowns sum to 3.52)
  // Walmart: "TAX 1 1.15" + "TAX 4 0.30" (independent categories → sum)
  const ocrTax = (() => {
    const taxLines = receiptLines.filter((l) => l.lineType === 'tax' && l.price !== null);
    if (taxLines.length === 0) return null;
    if (taxLines.length === 1) return parsePrice(taxLines[0].price);

    const values = taxLines.map((l) => parsePrice(l.price));
    const maxVal = Math.max(...values);
    const maxIdx = values.indexOf(maxVal);
    const othersSum = values.reduce((s, v, i) => (i === maxIdx ? s : s + v), 0);

    // If the smaller lines sum ≈ the largest, the largest is a total line
    // and the others are breakdowns → use just the largest.
    if (Math.abs(othersSum - maxVal) < 0.02) {
      return maxVal;
    }

    // Otherwise they are independent tax categories → sum all.
    return values.reduce((s, v) => s + v, 0);
  })();

  const ocrTotal = findOcrValue('total');

  // ── Tax rate ──────────────────────────────────────────────────────────
  let taxRate: number | null = null;
  if (ocrTax !== null && taxedItemsValue > 0) {
    taxRate = ocrTax / taxedItemsValue;
    console.log(`Estimated tax rate from OCR tax line: ${(taxRate * 100).toFixed(2)}%`);
  }
  else if(ocrSubtotal !== null && ocrTotal !== null) {
    const impliedTax = ocrTotal - ocrSubtotal;
    if (impliedTax > 0 && taxedItemsValue > 0) {
      taxRate = impliedTax / taxedItemsValue;
    }
    console.log(`Estimated tax rate from OCR subtotal and total: ${taxRate !== null ? (taxRate * 100).toFixed(2) + '%' : '(not computable)'}`);
  }

  const calculatedSubtotal = untaxedItemsValue + taxedItemsValue;

  // ── Tender amount (largest payment line) ──────────────────────────────
  const tenderAmount: number | null = (() => {
    let best: number | null = null;
    for (const l of receiptLines) {
      if (l.lineType !== 'tender' || l.price === null) continue;
      const v = parsePrice(l.price);
      if (v !== null && (best === null || v > best)) best = v;
    }
    return best;
  })();

  // ── Confidence ────────────────────────────────────────────────────────
  const confidence = checkConfidence(
    calculatedSubtotal,
    taxedItemsValue,
    untaxedItemsValue,
    ocrSubtotal,
    ocrTax,
    ocrTotal,
    taxRate,
    tenderAmount,
  );

  // ── Save annotated image ──────────────────────────────────────────────
  const lineAnnotations: LineAnnotation[] = receiptLines.map((line) => ({
    description: formatLineLabel(line),
    lineType: line.lineType,
    boundingPoly: buildLinePolygon(line.words),
  }));
  await saveAnnotatedImage(resolvedPath, imageBuffer, lineAnnotations);

  return {
    lines: receiptLines,
    angle,
    items,
    totalLines,
    totalItems: items.length,
    totalUntaxedItems: untaxedItems.length,
    totalTaxedItems: taxedItems.length,
    untaxedItemsValue,
    taxedItemsValue,
    ocrSubtotal,
    ocrTax,
    ocrTotal,
    calculatedSubtotal,
    taxRate,
    tenderAmount,
    confidence,
  };
}

// ── Pretty-print helper ──────────────────────────────────────────────────────

function printReceipt(receipt: Receipt): void {
  console.log(`\n${colorBold('=== Parsed Receipt ===')}`);
  for (const line of receipt.lines) {
    if (line.lineType === 'wrapped') continue;
    const tag = `[${line.lineType.toUpperCase()}]`.padEnd(14);
    const name = (line.itemName ?? line.text).padEnd(35);
    const price = line.price ?? '';
    const lineColor =
      line.lineType === 'subtotal' || line.lineType === 'tax' || line.lineType === 'total'
        ? colorCyan
        : line.lineType === 'discount'
          ? colorWarn
          : line.lineType === 'untaxed_item'
            ? colorDim
            : line.lineType === 'tender'
              ? colorDim
              : (s: string) => s;
    console.log(`  ${lineColor(tag)} ${name} ${price}`);
  }

  console.log(`\n${colorBold('=== Items (with discounts applied) ===')}`);
  for (const item of receipt.items) {
    const disc = item.discount !== 0 ? colorWarn(`  disc: ${item.discount.toFixed(2)}`) : '';
    const taxFlag = item.taxed ? colorDim(' [T]') : '';
    console.log(`  ${item.name.padEnd(35)} ${item.finalPrice.toFixed(2)}${disc}${taxFlag}`);
  }

  console.log(`\n${colorBold('=== Summary ===')}`);
  console.log(`  Total lines:          ${receipt.totalLines}`);
  console.log(`  Total items:          ${receipt.totalItems}`);
  console.log(`  Untaxed items:        ${receipt.totalUntaxedItems}   value: $${receipt.untaxedItemsValue.toFixed(2)}`);
  console.log(`  Taxed items:          ${receipt.totalTaxedItems}    value: $${receipt.taxedItemsValue.toFixed(2)}`);
  console.log(`  Calculated subtotal:  $${receipt.calculatedSubtotal.toFixed(2)}`);
  console.log(`  OCR subtotal:         $${receipt.ocrSubtotal?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  OCR tax:              $${receipt.ocrTax?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  OCR total:            $${receipt.ocrTotal?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  Tax rate:             ${receipt.taxRate !== null ? (receipt.taxRate * 100).toFixed(2) + '%' : colorDim('(not computable)')}`);
  console.log(`  Tender amount:        $${receipt.tenderAmount?.toFixed(2) ?? colorDim('(not found)')}`);

  // ── Colored confidence report with suggestions ──────────────────────────
  console.log(formatConfidenceReport(receipt.confidence));
}

// ── Entry point ───────────────────────────────────────────────────────────────
(async () => {
  const filePath = process.argv[2] ?? './sample.jpg';
  const receipt = await detectTextLocal(filePath);
  printReceipt(receipt);
})().catch((err: Error) => {
  console.error('Error:', err.message ?? err);
  process.exit(1);
});

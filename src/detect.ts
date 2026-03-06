import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import { reconstructLines } from './algorithm';
import type { ReconstructionResult } from './algorithm';
import type { ReceiptLine, Point, Receipt, ReceiptItem, ReceiptConfidence, LineType, WordBox, TaxRateInfo, DebugReceipt } from './utils';
import { rotatePoint, parsePrice, parseTaxRateLine, extractTaxCode } from './utils';
import { checkConfidence, formatConfidenceReport, colorBold, colorDim, colorPass, colorWarn, colorError, colorCyan } from './checks';
import { MAX_IMAGE_DIMENSION } from './constants';
import { determineTaxGroups } from './stores';
import type { TaxGroupResult } from './stores';

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
        fill="none" stroke="${color}" stroke-width="2"/>`;
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
      const taxCode = extractTaxCode(line.price);
      items.push({
        name: line.itemName ?? line.text,
        originalPrice,
        discount: 0,
        finalPrice: originalPrice,
        taxed: line.lineType === 'taxed_item',
        taxCode,
        taxRate: null,
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
export async function detectTextLocal(filePath: string): Promise<DebugReceipt> {
  const start = Date.now();
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
  
  const end1 = Date.now();

  const [result] = await client.documentTextDetection({ image: { content: imageBuffer } });

  
  const end2 = Date.now();

  const detections = result.textAnnotations;

  if (!detections || detections.length === 0) {
    throw new Error('No text detected in image.');
  }

  console.log(`Full text: ${result.fullTextAnnotation?.text}\n\n`);

  // ── Reconstruct receipt lines ─────────────────────────────────────────
  const { lines: receiptLines, angle } = reconstructLines(detections.slice(1));

  // ── Parse explicit tax rate breakdown lines ───────────────────────────
  const allTaxRates: TaxRateInfo[] = [];
  for (const line of receiptLines) {
    if (line.lineType === 'tax') {
      const info = parseTaxRateLine(line.text, line.price);
      if (info) {
        allTaxRates.push(info);
      }
    }
  }

  // ── Build items (with discounts applied) ──────────────────────────────
  const items = buildItems(receiptLines);

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
  const ocrTax = (() => {
    const taxLines = receiptLines.filter((l) => l.lineType === 'tax' && l.price !== null);
    if (taxLines.length === 0) return null;
    if (taxLines.length === 1) return parsePrice(taxLines[0].price);

    const values = taxLines.map((l) => parsePrice(l.price));
    const maxVal = Math.max(...values);

    const nonMaxValues = values.filter((v) => Math.abs(v - maxVal) >= 0.01);
    const nonMaxSum = nonMaxValues.reduce((s, v) => s + v, 0);

    if (nonMaxValues.length > 0 && Math.abs(nonMaxSum - maxVal) < 0.02) {
      return maxVal;
    }

    return values.reduce((s, v) => s + v, 0);
  })();

  const ocrTotal = findOcrValue('total');

  // ── Dynamic tax group analysis ────────────────────────────────────────
  const taxGroupResult = determineTaxGroups(items, ocrTax, ocrSubtotal, ocrTotal, allTaxRates);

  // Build rate lookup from groups
  const groupRateMap = new Map<string, number | null>();
  for (const g of taxGroupResult.groups) {
    groupRateMap.set(g.code, g.taxed ? g.rate : null);
  }

  // Reclassify items and receipt lines based on determined tax groups
  const taxedCodeSet = new Set(taxGroupResult.taxedCodes);
  for (const item of items) {
    const code = item.taxCode ?? '';
    item.taxed = taxedCodeSet.has(code);
    item.taxRate = groupRateMap.get(code) ?? null;
  }
  for (const line of receiptLines) {
    if (line.lineType !== 'untaxed_item' || line.price === null) continue;
    const code = extractTaxCode(line.price) ?? '';
    if (taxedCodeSet.has(code)) {
      line.lineType = 'taxed_item';
    }
  }

  // Log tax group results
  if (taxGroupResult.groups.length > 0) {
    console.log(`Tax groups detected:`);
    for (const g of taxGroupResult.groups) {
      const label = g.code || '(no suffix)';
      const status = g.taxed ? `TAXED @ ${((g.rate ?? 0) * 100).toFixed(2)}%` : 'untaxed';
      console.log(`  Group "${label}": ${g.items.length} items, $${g.total.toFixed(2)} — ${status}`);
    }
  }
  if (taxGroupResult.effectiveTaxRate !== null) {
    console.log(`Effective tax rate: ${(taxGroupResult.effectiveTaxRate * 100).toFixed(2)}%`);
  } else {
    console.log(`Effective tax rate: (not computable)`);
  }
  if (taxGroupResult.explicitRates.length > 0) {
    console.log(`Explicit tax rates from receipt:`);
    for (const tr of taxGroupResult.explicitRates) {
      console.log(`  ${tr.code}: ${(tr.rate * 100).toFixed(2)}% = $${tr.amount.toFixed(2)}`);
    }
  }
  if (allTaxRates.length > 0 && allTaxRates.some(r => !taxGroupResult.explicitRates.some(er => er.amount === r.amount && er.rate === r.rate))) {
    console.log(`All tax rates from receipt:`);
    for (const tr of allTaxRates) {
      console.log(`  ${tr.code}: ${(tr.rate * 100).toFixed(2)}% = $${tr.amount.toFixed(2)}`);
    }
  }

  // ── Counts ────────────────────────────────────────────────────────────
  const totalLines = receiptLines.filter((l) => l.lineType !== 'wrapped').length;
  const untaxedItems = items.filter((i) => !i.taxed);
  const taxedItems = items.filter((i) => i.taxed);

  const untaxedItemsValue = untaxedItems.reduce((s, i) => s + i.finalPrice, 0);
  const taxedItemsValue = taxedItems.reduce((s, i) => s + i.finalPrice, 0);

  const taxRate = taxGroupResult.effectiveTaxRate;
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
    taxGroupResult.explicitRates,
    items,
  );


  const end3 = Date.now();

  // ── Save annotated image ──────────────────────────────────────────────
  const lineAnnotations: LineAnnotation[] = receiptLines.map((line) => ({
    description: formatLineLabel(line),
    lineType: line.lineType,
    boundingPoly: buildLinePolygon(line.words),
  }));
  await saveAnnotatedImage(resolvedPath, imageBuffer, lineAnnotations);

  

  const end4 = Date.now();

  console.log(`Total processing time: ${((end4 - start) / 1000).toFixed(2)}s`);
  return {
    lines: receiptLines,
    angle,
    detectedStore: null,
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
    taxRates: allTaxRates,
    tenderAmount,
    confidence,
    times: [
      { type: 'Image loaded and resized time', elapsed: end1-start},
      { type: 'OCR Text detection time', elapsed: end2-end1 },
      { type: 'Post-processing time', elapsed: end3-end2 },
      { type: 'Save annotation time', elapsed: end4-end3 }
    ]
  };
}

// ── Pretty-print helper ──────────────────────────────────────────────────────

function printReceipt(receipt: DebugReceipt): void {
  // Build rate lookup from items for line-level display
  const codeToRate = new Map<string, number | null>();
  for (const item of receipt.items) {
    const code = item.taxCode ?? '';
    if (!codeToRate.has(code)) {
      codeToRate.set(code, item.taxRate);
    }
  }

  console.log(`\n${colorBold('=== Parsed Receipt ===')}`);
  for (const line of receipt.lines) {
    if (line.lineType === 'wrapped') continue;
    let tag: string;
    if (line.lineType === 'taxed_item' || line.lineType === 'untaxed_item') {
      const code = extractTaxCode(line.price) ?? '';
      const label = code || '(none)';
      const rate = codeToRate.get(code);
      tag = rate !== null && rate !== undefined
        ? `[${label} @ ${(rate * 100).toFixed(2)}%]`
        : `[${label} UNTAXED]`;
    } else {
      tag = `[${line.lineType.toUpperCase()}]`;
    }
    tag = tag.padEnd(16);
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
    const rateLabel = item.taxRate !== null
      ? colorDim(` [${item.taxCode ?? '?'} @ ${(item.taxRate * 100).toFixed(2)}%]`)
      : (item.taxCode ? colorDim(` [${item.taxCode} untaxed]`) : '');
    console.log(`  ${item.name.padEnd(35)} ${item.finalPrice.toFixed(2)}${disc}${rateLabel}`);
  }

  console.log(`\n${colorBold('=== Summary ===')}`);
  console.log(`  Total lines:          ${receipt.totalLines}`);
  console.log(`  Total items:          ${receipt.totalItems}`);

  // Per-group breakdown
  const groupMap = new Map<string, { items: typeof receipt.items, rate: number | null }>();
  for (const item of receipt.items) {
    const key = item.taxCode ?? '';
    if (!groupMap.has(key)) groupMap.set(key, { items: [], rate: item.taxRate });
    groupMap.get(key)!.items.push(item);
  }
  console.log(`  Tax groups:`);
  for (const [code, group] of groupMap) {
    const label = code || '(no suffix)';
    const total = group.items.reduce((s, i) => s + i.finalPrice, 0);
    const rateStr = group.rate !== null
      ? `${(group.rate * 100).toFixed(2)}%`
      : 'untaxed';
    console.log(`    ${label} (${rateStr}): ${group.items.length} items, $${total.toFixed(2)}`);
  }

  console.log(`  Calculated subtotal:  $${receipt.calculatedSubtotal.toFixed(2)}`);
  console.log(`  OCR subtotal:         $${receipt.ocrSubtotal?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  OCR tax:              $${receipt.ocrTax?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  OCR total:            $${receipt.ocrTotal?.toFixed(2) ?? colorDim('(not found)')}`);
  console.log(`  Tax rate:             ${receipt.taxRate !== null ? (receipt.taxRate * 100).toFixed(2) + '%' : colorDim('(not computable)')}`);
  if (receipt.taxRates.length > 0) {
    console.log(`  Tax rate breakdown:`);
    for (const tr of receipt.taxRates) {
      console.log(`    Code ${tr.code}: ${(tr.rate * 100).toFixed(2)}% = $${tr.amount.toFixed(2)}`);
    }
  }
  console.log(`  Tender amount:        $${receipt.tenderAmount?.toFixed(2) ?? colorDim('(not found)')}`);
  
  for( const timeInfo of receipt.times) {
    console.log(`  ${timeInfo.type}: ${(timeInfo.elapsed / 1000).toFixed(2)}s`);
  }
  console.log(`total processing time: ${((receipt.times.reduce((s, t) => s + t.elapsed, 0)) / 1000).toFixed(2)}s`);

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

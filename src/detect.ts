import 'dotenv/config';

import vision, { protos } from '@google-cloud/vision';
import sharp from 'sharp';
import fs from 'fs';
import path from 'path';

import { reconstructLines } from './algorithm';
import type { ReceiptLine, ReceiptItem, TaxRateInfo, DebugReceipt } from './types';
import { parsePrice, parseTaxRateLine, extractTaxCode } from './utils';
import { checkConfidence, formatConfidenceReport } from './checks';
import { MAX_IMAGE_DIMENSION } from './constants';
import { determineTaxGroups } from './stores';
import type { TaxGroupResult } from './stores';
import { LineAnnotation, buildLinePolygon, formatLineLabel, saveAnnotatedImage } from './annotation';
import { printReceipt } from './printing';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;

// Creates a client using an API key from the GOOGLE_API_KEY environment variable.
const client = new vision.ImageAnnotatorClient({
  apiKey: process.env.GOOGLE_API_KEY,
});

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

// ── Entry point ───────────────────────────────────────────────────────────────
(async () => {
  const filePath = process.argv[2] ?? './sample.jpg';
  const receipt = await detectTextLocal(filePath);
  printReceipt(receipt);
})().catch((err: Error) => {
  console.error('Error:', err.message ?? err);
  process.exit(1);
});

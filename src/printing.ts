/**
 * Receipt output formatting for terminal display.
 */

import type { DebugReceipt } from './types';
import { colorBold, colorDim, colorPass, colorWarn, colorCyan } from './colors';
import { formatConfidenceReport } from './checks';
import { extractTaxCode } from './utils';

export function printReceipt(receipt: DebugReceipt): void {
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
  
  for (const timeInfo of receipt.times) {
    console.log(`  ${timeInfo.type}: ${(timeInfo.elapsed / 1000).toFixed(2)}s`);
  }
  console.log(`total processing time: ${((receipt.times.reduce((s, t) => s + t.elapsed, 0)) / 1000).toFixed(2)}s`);

  // Colored confidence report with suggestions
  console.log(formatConfidenceReport(receipt.confidence));
}

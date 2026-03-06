/**
 * Llama 3.2 receipt post-processing pipeline.
 *
 * Sends only item/discount lines to Llama 3.2 for name cleanup and
 * tax code identification. All other receipt values (subtotal, tax,
 * total, tender, tax rates, discounts) are computed algorithmically
 * from the already-classified ReceiptLines.
 *
 * Requirements:
 *   - Ollama must be running locally (default: http://localhost:11434)
 *   - The `llama3.2` model must be pulled: `ollama pull llama3.2`
 */

import http from 'http';
import https from 'https';
import type {
  ReceiptLine,
  ReceiptItem,
  TaxRateInfo,
  DebugReceipt,
} from './types';
import { checkConfidence } from './checks';
import { parsePrice, parseTaxRateLine, extractTaxCode } from './utils';
import { determineTaxGroups } from './stores';

// ── Configuration ────────────────────────────────────────────────────────────

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? 'http://localhost:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? 'llama3.2';

// ── Build minimal input: only item/discount lines ────────────────────────────

/**
 * Convert reconstructed ReceiptLines into a compact JSON representation
 * containing only item and discount lines — the only lines Llama needs
 * to interpret (name cleanup + tax code identification).
 *
 * Each entry includes the raw text, algorithmically-detected price, and
 * the line's center position for spatial context.
 */
export function buildLlamaInput(lines: ReceiptLine[]): string {
  const visibleLines = lines.filter((l) => l.lineType !== 'wrapped');
  const items: { t: string; price: string; type: string }[] = [];

  for (const line of visibleLines) {
    if (
      line.lineType !== 'untaxed_item' &&
      line.lineType !== 'taxed_item' &&
      line.lineType !== 'discount'
    ) {
      continue;
    }

    // Skip discount lines with no price — these are descriptive text
    // (e.g. "PROMOTIONAL DISCOUNT OR COUPON"), not actual discount transactions
    if (line.lineType === 'discount' && line.price == null) {
      continue;
    }

    const taxCode = extractTaxCode(line.price);
    const cleanPrice = (line.price ?? '?').replace(/\s*[A-Z]$/i, '').trim();
    const priceSuffix = taxCode ? `${cleanPrice} ${taxCode}` : cleanPrice;

    items.push({
      t: line.itemName ?? line.text,
      price: priceSuffix,
      type: line.lineType === 'discount' ? 'discount' : 'item',
    });
  }

  return JSON.stringify(items);
}

// ── Prompt ───────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You parse receipt items. Input is a JSON array of objects with:
- "t": raw OCR text of an item or discount line
- "price": the algorithmically-detected price (already correct — do NOT change it). If the price string ends with a letter (e.g. "5.48 N"), that letter is the tax code.
- "type": "item" or "discount"

Your job is ONLY to clean up the item name from the raw OCR text. Do NOT invent items, change prices, or add items not in the input.
Output one entry per input entry, in the same order, including discounts.
Respond with a SINGLE JSON object containing ONE "items" array:
{"items":[{"name":"Bananas","price":0.20,"taxCode":"N"},{"name":"Frappuccino","price":5.48,"taxCode":"N"},{"name":"Discount given","price":-0.57,"taxCode":null}]}
Rules:
- ALL items AND discounts go in ONE "items" array — one output per input, same order
- name: cleaned item name — fix OCR errors, remove barcodes/UPCs/quantities, use sentence case
- price: use the EXACT numeric price from the input. Discounts ("type":"discount") must be NEGATIVE.
- taxCode: if the "price" string ends with a letter (e.g. "0.20 N"), set taxCode to that letter (e.g. "N"). Otherwise null.
- Return ONLY the JSON, no extra text`;

// ── HTTP helper ──────────────────────────────────────────────────────────────

interface OllamaResponse {
  model: string;
  response: string;
  done: boolean;
}

function ollamaGenerate(prompt: string, systemPrompt: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const url = new URL('/api/generate', OLLAMA_BASE_URL);
    const body = JSON.stringify({
      model: OLLAMA_MODEL,
      prompt,
      system: systemPrompt,
      stream: false,
      options: {
        temperature: 0,
        num_predict: 2048,
      },
    });

    const isHttps = url.protocol === 'https:';
    const transport = isHttps ? https : http;

    const req = transport.request(
      {
        hostname: url.hostname,
        port: url.port || (isHttps ? 443 : 80),
        path: url.pathname,
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(body),
        },
        timeout: 120_000,
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk: Buffer) => chunks.push(chunk));
        res.on('end', () => {
          const raw = Buffer.concat(chunks).toString('utf-8');
          if (res.statusCode !== 200) {
            reject(new Error(`Ollama returned HTTP ${res.statusCode}: ${raw}`));
            return;
          }
          try {
            const parsed: OllamaResponse = JSON.parse(raw);
            resolve(parsed.response);
          } catch {
            reject(new Error(`Failed to parse Ollama response: ${raw.slice(0, 500)}`));
          }
        });
      },
    );

    req.on('error', (err) =>
      reject(new Error(`Ollama request failed (is Ollama running at ${OLLAMA_BASE_URL}?): ${err.message}`)),
    );
    req.on('timeout', () => {
      req.destroy();
      reject(new Error('Ollama request timed out (120s)'));
    });

    req.write(body);
    req.end();
  });
}

// ── Response parsing ─────────────────────────────────────────────────────────

interface LlamaItem {
  text: string;
  name: string;
  price: number;
  taxCode: string | null;
}

interface LlamaResponse {
  items: LlamaItem[];
}

/**
 * Extract JSON from a Llama response that may have markdown fences,
 * extra braces, or repeated "items" keys (malformed JSON where Llama
 * outputs each item as a separate "items" key).
 */
function extractJson(text: string): string {
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) text = fenceMatch[1].trim();

  const braceStart = text.indexOf('{');
  if (braceStart === -1) return text.trim();

  // Extract the full outer {...} using brace counting
  let rawJson: string;
  let depth = 0;
  let inString = false;
  let escape = false;
  let endIdx = -1;
  for (let i = braceStart; i < text.length; i++) {
    const ch = text[i];
    if (escape) { escape = false; continue; }
    if (ch === '\\' && inString) { escape = true; continue; }
    if (ch === '"') { inString = !inString; continue; }
    if (inString) continue;
    if (ch === '{') depth++;
    if (ch === '}') {
      depth--;
      if (depth === 0) { endIdx = i; break; }
    }
  }

  if (endIdx !== -1) {
    rawJson = text.slice(braceStart, endIdx + 1);
  } else {
    const braceEnd = text.lastIndexOf('}');
    rawJson = braceEnd > braceStart
      ? text.slice(braceStart, braceEnd + 1)
      : text.trim();
  }

  // Fix malformed JSON where Llama repeats "items":[...] keys.
  // Merge all arrays into one: {"items":[...],"items":[...]} → {"items":[...,...]}
  const repeatedKeyPattern = /"items"\s*:\s*\[/g;
  const matches = [...rawJson.matchAll(repeatedKeyPattern)];
  if (matches.length > 1) {
    // Extract each individual item object from all the repeated arrays
    const allItems: string[] = [];
    const itemPattern = /\{\s*"name"\s*:[^}]+\}/g;
    let m: RegExpExecArray | null;
    while ((m = itemPattern.exec(rawJson)) !== null) {
      allItems.push(m[0]);
    }
    if (allItems.length > 0) {
      return `{"items":[${allItems.join(',')}]}`;
    }
  }

  return rawJson;
}

// ── Algorithmic extraction from ReceiptLines ─────────────────────────────────

function extractAlgorithmicValues(lines: ReceiptLine[]) {
  const visibleLines = lines.filter((l) => l.lineType !== 'wrapped');

  // Subtotal: use last subtotal line
  const subtotalLines = visibleLines.filter((l) => l.lineType === 'subtotal' && l.price !== null);
  const ocrSubtotal = subtotalLines.length > 0
    ? parsePrice(subtotalLines[subtotalLines.length - 1].price)
    : null;

  // Tax: sum tax lines, detecting breakdown-vs-total pattern
  const ocrTax = (() => {
    const taxLines = visibleLines.filter((l) => l.lineType === 'tax' && l.price !== null);
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

  // Total
  const totalLine = visibleLines.find((l) => l.lineType === 'total' && l.price !== null);
  const ocrTotal = totalLine ? parsePrice(totalLine.price) : null;

  // Tender: largest tender line
  const tenderAmount = (() => {
    let best: number | null = null;
    for (const l of visibleLines) {
      if (l.lineType !== 'tender' || l.price === null) continue;
      const v = parsePrice(l.price);
      if (v !== null && (best === null || v > best)) best = v;
    }
    return best;
  })();

  // Tax rates from tax lines
  const taxRates: TaxRateInfo[] = [];
  for (const line of visibleLines) {
    if (line.lineType === 'tax') {
      const info = parseTaxRateLine(line.text, line.price);
      if (info) taxRates.push(info);
    }
  }

  return { ocrSubtotal, ocrTax, ocrTotal, tenderAmount, taxRates };
}

// ── Build items algorithmically (same logic as detect.ts) ────────────────────

function buildItemsFromLines(lines: ReceiptLine[]): ReceiptItem[] {
  const items: ReceiptItem[] = [];
  const visibleLines = lines.filter((l) => l.lineType !== 'wrapped');

  for (const line of visibleLines) {
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
    } else if (line.lineType === 'discount' && line.price != null) {
      const discountValue = -Math.abs(parsePrice(line.price));
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

// ── Main pipeline function ───────────────────────────────────────────────────

/**
 * Send item/discount lines to Llama 3.2 for name cleanup and tax code
 * identification, then compute all other values algorithmically.
 */
export async function processReceiptWithLlama(
  compactInput: string,
  lines: ReceiptLine[],
  angle: number,
): Promise<{
  receipt: Omit<DebugReceipt, 'times'>;
  llamaElapsed: number;
}> {
  const start = Date.now();

  const userPrompt = `Parse these receipt items:\n\n${compactInput}`;

  console.log('── Llama input ──────────────────────────────────────────');
  console.log(compactInput);
  console.log('─────────────────────────────────────────────────────────\n');

  let responseText: string;
  try {
    responseText = await ollamaGenerate(userPrompt, SYSTEM_PROMPT);
  } catch (err: any) {
    console.warn(`Llama call failed: ${err.message} — using algorithmic names only`);
    responseText = '';
  }

  console.log('── Llama raw response ───────────────────────────────────');
  console.log(responseText);
  console.log('─────────────────────────────────────────────────────────\n');

  const elapsed = Date.now() - start;

  // Parse Llama JSON response
  let parsed: LlamaResponse;
  if (responseText) {
    const jsonStr = extractJson(responseText);
    try {
      parsed = JSON.parse(jsonStr);
    } catch {
      console.warn(`Llama returned invalid JSON — using algorithmic names only`);
      parsed = { items: [] };
    }
  } else {
    parsed = { items: [] };
  }

  // ── Build items algorithmically (prices, tax codes, discounts) ────
  const items = buildItemsFromLines(lines);

  // ── Overlay Llama's cleaned names onto algorithmic items ────────────
  // Llama is only used for name cleanup. Prices, tax codes, and discounts
  // come from the algorithm which is more reliable.
  const llamaItems = parsed.items ?? [];

  // Build a list of non-discount Llama items (just cleaned names)
  const llamaNames = llamaItems
    .filter((li) => !(typeof li.price === 'number' && li.price < 0))
    .map((li) => li.name);

  if (llamaNames.length === items.length) {
    // Counts match: apply names 1:1
    for (let i = 0; i < items.length; i++) {
      items[i].name = llamaNames[i];
    }
    console.log(`Llama names applied 1:1 (${llamaNames.length} items)`);
  } else if (llamaNames.length > 0) {
    // Counts don't match: try to match by price
    const used = new Set<number>();
    for (const item of items) {
      for (let j = 0; j < llamaItems.length; j++) {
        if (used.has(j)) continue;
        const li = llamaItems[j];
        if (typeof li.price === 'number' && li.price >= 0 &&
            Math.abs(li.price - item.originalPrice) < 0.01) {
          item.name = li.name;
          used.add(j);
          break;
        }
      }
    }
    console.log(`Llama returned ${llamaNames.length} items vs ${items.length} algorithmic — matched by price`);
  } else {
    console.log(`Llama returned no usable items — using algorithmic names`);
  }

  // ── Algorithmic values from classified lines ────────────────────────
  const { ocrSubtotal, ocrTax, ocrTotal, tenderAmount, taxRates } =
    extractAlgorithmicValues(lines);

  // ── Dynamic tax group analysis (same as detect.ts) ─────────────────
  const taxGroupResult = determineTaxGroups(items, ocrTax, ocrSubtotal, ocrTotal, taxRates);

  // Build rate lookup from groups
  const groupRateMap = new Map<string, number | null>();
  for (const g of taxGroupResult.groups) {
    groupRateMap.set(g.code, g.taxed ? g.rate : null);
  }

  // Reclassify items based on determined tax groups
  const taxedCodeSet = new Set(taxGroupResult.taxedCodes);
  for (const item of items) {
    const code = item.taxCode ?? '';
    item.taxed = taxedCodeSet.has(code);
    item.taxRate = groupRateMap.get(code) ?? null;
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
  }

  // ── Derived values ──────────────────────────────────────────────────
  const untaxedItems = items.filter((i) => !i.taxed);
  const taxedItems = items.filter((i) => i.taxed);

  const untaxedItemsValue = untaxedItems.reduce((s, i) => s + i.finalPrice, 0);
  const taxedItemsValue = taxedItems.reduce((s, i) => s + i.finalPrice, 0);
  const calculatedSubtotal = untaxedItemsValue + taxedItemsValue;

  const taxRate = taxGroupResult.effectiveTaxRate;

  // ── Confidence ──────────────────────────────────────────────────────
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

  const receipt = {
    lines,
    angle,
    detectedStore: null,
    items,
    totalLines: lines.filter((l) => l.lineType !== 'wrapped').length,
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
    taxRates: taxGroupResult.explicitRates,
    tenderAmount,
    confidence,
  };

  return { receipt, llamaElapsed: elapsed };
}

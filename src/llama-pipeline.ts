/**
 * Llama 3.2 receipt post-processing pipeline.
 *
 * Sends only item names to Llama 3.2 for name cleanup. All other
 * receipt values (subtotal, tax, total, tender, tax rates, discounts,
 * prices, tax codes) are computed algorithmically from the
 * already-classified ReceiptLines.
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

// ── Build minimal input: only item names, numbered ───────────────────

/**
 * Extract raw item names (no discounts) from ReceiptLines as numbered lines.
 * Llama only needs the item names to clean up — everything else is algorithmic.
 * Numbered format makes 1:1 correspondence structurally explicit.
 */
export function buildLlamaInput(lines: ReceiptLine[]): string {
  const visibleLines = lines.filter((l) => l.lineType !== 'wrapped');
  const numbered: string[] = [];

  for (const line of visibleLines) {
    if (
      line.lineType !== 'untaxed_item' &&
      line.lineType !== 'taxed_item'
    ) {
      continue;
    }

    const name = line.itemName ?? line.text;
    numbered.push(`${numbered.length + 1}. ${name}`);
  }

  return numbered.join('\n');
}

// ── Prompt ───────────────────────────────────────────────────────────────────

const SYSTEM_PROMPT = `You clean up receipt item names. Input is a numbered list of raw OCR item names from a receipt.

Return a numbered list with the SAME numbers, one cleaned name per line.

Example input:
1. BNLS SKNLS CHKN BR
2. FRSH STRWBRRS 1LB
3. MBR DISC

Example output:
1. Boneless Skinless Chicken Breast
2. Fresh Strawberries
3. Member Discount

Rules:
- Return EXACTLY the same count of numbered lines as the input
- Each input line N maps to EXACTLY one output line N
- NEVER split one input line into multiple output lines
- NEVER merge multiple input lines into one output line
- NEVER skip or omit any input line
- Fix OCR errors and expand abbreviations (e.g. "pkg" → "Package", "org" → "Organic")
- Remove barcodes, UPC numbers, store codes, and quantity prefixes
- Use sentence case
- If a name is unrecognizable, clean it up as best you can
- Return ONLY the numbered list, no extra text`;

// ── HTTP helper ──────────────────────────────────────────────────────────────

interface OllamaResponse {
  model: string;
  response: string;
  done: boolean;
  prompt_eval_count?: number;  // input tokens
  eval_count?: number;         // output tokens
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
            if (parsed.prompt_eval_count !== undefined || parsed.eval_count !== undefined) {
              console.log(`Ollama tokens — input: ${parsed.prompt_eval_count ?? '?'}, output: ${parsed.eval_count ?? '?'}`);
            }
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

/**
 * Parse a numbered list response from Llama into an array of cleaned names.
 * Expects lines like "1. Cleaned Name" or "1) Cleaned Name".
 * Falls back to JSON array parsing if numbered format isn't found.
 */
function parseNumberedResponse(text: string): string[] {
  // Try numbered list format first: "1. Name" or "1) Name"
  const numberedLines = text
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => /^\d+[.)]/  .test(line))
    .map((line) => line.replace(/^\d+[.)\s]+/, '').trim());

  if (numberedLines.length > 0) return numberedLines;

  // Fallback: try JSON array
  const fenceMatch = text.match(/```(?:json)?\s*\n?([\s\S]*?)```/);
  if (fenceMatch) text = fenceMatch[1].trim();

  const bracketStart = text.indexOf('[');
  if (bracketStart !== -1) {
    let depth = 0;
    let inString = false;
    let escape = false;
    let endIdx = -1;
    for (let i = bracketStart; i < text.length; i++) {
      const ch = text[i];
      if (escape) { escape = false; continue; }
      if (ch === '\\' && inString) { escape = true; continue; }
      if (ch === '"') { inString = !inString; continue; }
      if (inString) continue;
      if (ch === '[') depth++;
      if (ch === ']') { depth--; if (depth === 0) { endIdx = i; break; } }
    }
    const rawJson = endIdx !== -1 ? text.slice(bracketStart, endIdx + 1) : text.trim();
    try {
      const parsed = JSON.parse(rawJson);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch { /* fall through */ }
  }

  // Fallback: try {"items":[...]} format
  try {
    const braceStart = text.indexOf('{');
    if (braceStart !== -1) {
      const obj = JSON.parse(text.slice(braceStart));
      if (Array.isArray(obj?.items)) return obj.items.map((i: any) => typeof i === 'string' ? i : i.name ?? String(i));
    }
  } catch { /* fall through */ }

  return [];
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
 * Send item names to Llama 3.2 for name cleanup, then compute all
 * other values algorithmically.
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

  const inputCount = compactInput.split('\n').length;
  const userPrompt = `Clean these ${inputCount} receipt item names:\n\n${compactInput}`;

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

  // Parse Llama response (numbered list of cleaned names)
  const llamaNames: string[] = responseText
    ? parseNumberedResponse(responseText)
    : [];

  // ── Build items algorithmically (prices, tax codes, discounts) ────
  const items = buildItemsFromLines(lines);

  // ── Overlay Llama's cleaned names onto algorithmic items ────────────
  if (llamaNames.length === items.length) {
    for (let i = 0; i < items.length; i++) {
      items[i].name = llamaNames[i];
    }
    console.log(`Llama names: ${llamaNames.length}/${items.length} OK`);
  } else if (llamaNames.length > items.length && items.length > 0) {
    // Llama returned extra names (hallucination) — use the first N
    for (let i = 0; i < items.length; i++) {
      items[i].name = llamaNames[i];
    }
    console.log(`Llama names: ${items.length}/${items.length} OK (truncated from ${llamaNames.length})`);
  } else if (llamaNames.length > 0) {
    console.log(`Llama names: ${llamaNames.length}/${items.length} MISMATCH`);
  } else {
    console.log(`Llama names: 0/${items.length} MISMATCH`);
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

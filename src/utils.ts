/**
 * Pure utility functions used by the receipt-line reconstruction algorithm.
 * Type definitions live in ./types.ts.
 */

import { protos } from '@google-cloud/vision';
import type { Point, WordBox, LineType } from './types';

// Re-export all types so existing imports from './utils' still work.
export type { Point, WordBox, LineType, ReceiptLine, Severity, CheckResult, TaxRateInfo, ReceiptItem, ReceiptConfidence, Receipt, DebugReceipt } from './types';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;

// ── Conversion ────────────────────────────────────────────────────────────────

/** Convert a Vision API word annotation into a `WordBox`. */
export function annotationToWordBox(ann: TextAnnotation): WordBox {
  const verts = (ann.boundingPoly?.vertices ?? []).map((v) => ({
    x: v.x ?? 0,
    y: v.y ?? 0,
  }));

  const xs = verts.map((v) => v.x);
  const ys = verts.map((v) => v.y);

  const left = Math.min(...xs);
  const right = Math.max(...xs);
  const top = Math.min(...ys);
  const bottom = Math.max(...ys);

  return {
    text: ann.description ?? '',
    center: { x: (left + right) / 2, y: (top + bottom) / 2 },
    left,
    right,
    top,
    bottom,
    width: right - left,
    height: bottom - top,
    vertices: verts,
    original: ann,
  };
}

// ── Math helpers ──────────────────────────────────────────────────────────────

/** Median of a numeric array (returns 0 for empty input). */
export function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

/** Rotate `p` by `angle` radians around `origin` (default: 0,0). */
export function rotatePoint(
  p: Point,
  angle: number,
  origin: Point = { x: 0, y: 0 },
): Point {
  const cos = Math.cos(angle);
  const sin = Math.sin(angle);
  const dx = p.x - origin.x;
  const dy = p.y - origin.y;
  return {
    x: origin.x + dx * cos - dy * sin,
    y: origin.y + dx * sin + dy * cos,
  };
}

/** Angle (radians) of the vector from `a` to `b`. */
export function angleBetween(a: Point, b: Point): number {
  return Math.atan2(b.y - a.y, b.x - a.x);
}

// ── Price detection ───────────────────────────────────────────────────────────

/**
 * Matches common receipt price formats:
 * `12.34`  `-12.34`  `12.34-`  `$1,234.56`  `-$1,234.56`  `(1,234.56)`  `($1,234.56)`  `(1,234.56)-`
 * `12.34 A`  `12.34-A`  `12.34 X`  `12.34 N`
 * `.88`  `.8888`  `.88 FS`
 *
 * Supports 2–4 decimal places.
 *
 * Trailing tax-flag letters: A (taxed), X (taxed, Walmart), N (non-taxable),
 * E, T, O, B, R, F (various store-specific flags).
 */
const PRICE_REGEX =
/^(?:(?:-?\$?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2,4})|\.\d{2,4}|\(\$?(?:\d{1,3}(?:,\d{3})*|\d+)\.\d{2,4}\))(?:-(?![A-Z]))?(?:[ -][A-Z]{1,2})?$/i;

export function isPrice(text: string): boolean {
  return PRICE_REGEX.test(text.trim());
}

/**
 * Try to strip a leading Walmart-style tax flag letter (F, N, T, X, B, O, R)
 * that OCR sometimes fuses with the price token.  E.g. "F2.00" → "2.00".
 * Returns the bare price string or null if no flag pattern is found.
 */
export function tryStripTaxFlag(text: string): string | null {
  const m = text.trim().match(/^[FNTXBOREP](\$?\d{0,3}(?:,\d{3})*\.\d{2})$/i);
  return m && isPrice(m[1]) ? m[1] : null;
}

// ── Line classification by keywords ───────────────────────────────────────────

const SUBTOTAL_KEYWORDS = ['subtotal', 'sub-total', 'sub total', 'net sales'];
const TOTAL_KEYWORDS = ['total', 'amount due', 'balance', 'grand total'];
const TAX_KEYWORDS_RE = [/\btax\d*\b/, /\bhst\b/, /\bgst\b/, /\bpst\b/, /\bvat\b/];
const DISCOUNT_KEYWORDS = [
  'disc',
  'discount',
  'off',
  'save',
  'savings',
  'coupon',
  'promo',
];
const DISCOUNT_KEYWORDS_PRICE = [
  '-'
];

/**
 * Payment-method keywords.  Lines containing these are credit/debit card
 * charges, cash-back, or auth-code receipts — never actual item lines.
 * Checked AFTER subtotal/total/tax/discount so those still take priority
 * (e.g. "VISA TOTAL $17.49" is classified as 'total', not payment).
 */
const PAYMENT_KEYWORDS = [
  'visa', 'mastercard', 'master card', 'amex', 'american express',
  'discover', 'debit', 'change due', 'bal due',
];

/**
 * Tender / payment-detail keywords.  Lines that describe HOW the receipt
 * was paid — cash tendered, change given, card details, tips, gratuity.
 * These are never item lines but carry useful cross-check data.
 *
 * NOTE: matched with word-boundary regex (\b) to avoid false positives
 * like "BAND-AID" matching 'aid' or "CHIPS" matching 'chip'.
 */
const TENDER_KEYWORDS_RE = [
  /\btender\b/, /\bcash\b/, /\btendered\b/, /\bamount tendered\b/, /\bchange\b/,
  /\btip\b/, /\bgratuity\b/, /\bservice charge\b/,
  /\bapproved\b/, /\bauth code\b/, /\baid\s*:/, /\bchip\b/,
  /\bpaid\b/, /\bpayment\b/, /\bactivation\b/, /\bredemption\b/,
  /\btend\b/,
];

/**
 * Classify a receipt line by its text content.
 * Check order matters: "subtotal" must be checked before "total".
 *
 * @param taxedCodes  Store-specific tax code letters that indicate a taxed item.
 *                    Defaults to ['A', 'X'] when no store is detected.
 */
export function classifyLine(text: string, price: string | null): LineType {
  const lower = text.toLowerCase();

  if (SUBTOTAL_KEYWORDS.some((k) => lower.includes(k))) return 'subtotal';
  if (TOTAL_KEYWORDS.some((k) => lower.includes(k))) {
    // "TOTAL NUMBER OF ITEMS SOLD" etc. are informational, not the receipt total
    if (lower.includes('number of')) return 'info';
    // EBT / food-stamp / loyalty balance lines are payment info, not the receipt total
    if (lower.includes('food stamps') || lower.includes('food stamp')) return 'info';
    if (lower.includes('ebt')) return 'info';
    if (lower.includes('cash balance')) return 'info';
    // "TOTAL DISCOUNTS" is a savings summary, not the receipt total
    if (lower.includes('discounts')) return 'info';
    // "TOTAL TAX" is a tax line, not the receipt total
    if (/\btotal\s+tax\b/.test(lower)) return 'tax';
    // "TOTAL SAVINGS" is a savings summary, not the receipt total
    if (/\btotal\s+savings?\b/.test(lower)) return 'info';
    return 'total';
  }
  // "VAT No" / "VAT Number" / "VAT Reg" = registration info, not a tax amount
  if (/\bvat\s*(no|number|reg)\b/.test(lower)) return 'info';
  // "VAT rate" is a label/header, not a tax amount line
  if (/\bvat\s+rate\b/.test(lower)) return 'info';
  // "TAX INVOICE" = document label, not a tax amount
  if (/\btax\s*invoice\b/.test(lower)) return 'info';
  if (TAX_KEYWORDS_RE.some((re) => re.test(lower))) return 'tax';
  // "Save money" (Walmart slogan) is informational, not a discount offer
  if (/\bsave\s+money\b/.test(lower)) return 'info';
  if (DISCOUNT_KEYWORDS.some((k) => lower.includes(k))) return 'discount';
  if (PAYMENT_KEYWORDS.some((k) => lower.includes(k))) return 'tender';
  // "Sale Price" lines show the post-discount price — informational, not a separate item
  if (lower.includes('sale price')) return 'info';
  if (TENDER_KEYWORDS_RE.some((re) => re.test(lower))) return 'tender';
  // Weight/tare description lines (e.g. "1.08 lb @ 1.99 /lb TARE = .01") are not items
  if (/\btare\b/.test(lower)) return 'info';
  if (!price) return 'info';
  if (DISCOUNT_KEYWORDS_PRICE.some((k) => price.includes(k))) return 'discount';
  return 'untaxed_item';
}

// ── Price value parsing ───────────────────────────────────────────────────────

/**
 * Extract the numeric value from a price string.
 * Handles `$`, commas, parentheses (negative), trailing `-`, leading `-`.
 * Returns 0 if the string cannot be parsed.
 *
 * Examples:
 *   "$12.34"    → 12.34
 *   "12.34-"    → -12.34  (trailing minus = discount)
 *   "-$1.50"    → -1.50
 *   "($3.00)"   → -3.00
 *   "1.00-A"    → -1.00
 *   "12.34 A"   → 12.34
 */
export function parsePrice(raw: string | null): number {
  if (!raw) return 0;
  let s = raw.trim();

  // Detect negativity from surrounding parens or trailing dash (before suffix)
  const hasParens = s.startsWith('(') && s.includes(')');
  // trailing minus: "12.34-" or "12.34-A" — the minus is between digits and optional A
  const hasTrailingMinus = /\d-/.test(s);
  const hasLeadingMinus = s.startsWith('-');

  // Strip everything except digits, dots, and commas
  s = s.replace(/[^0-9.,]/g, '');
  const value = parseFloat(s.replace(/,/g, ''));
  if (isNaN(value)) return 0;

  return (hasParens || hasTrailingMinus || hasLeadingMinus) ? -Math.abs(value) : value;
}

// ── Tax rate line parsing ─────────────────────────────────────────────────────

/**
 * Parse an explicit tax rate breakdown line.
 * Supports two formats:
 *   1) "A 8.50% TAX" — letter code before rate (e.g. Albertsons/Sprouts)
 *   2) "TAX 1 7.900 %" — TAX keyword then numeric code and rate (e.g. Walmart)
 * Returns a TaxRateInfo or null if the line doesn't match.
 */
export function parseTaxRateLine(text: string, price: string | null): { code: string; rate: number; amount: number } | null {
  if (price === null) return null;

  // Format 1: "A 8.50% TAX"
  let m = text.match(/\b([A-Z])\s+(\d+(?:\.\d+)?)%\s+TAX\b/i);
  if (m) {
    return {
      code: m[1].toUpperCase(),
      rate: parseFloat(m[2]) / 100,
      amount: parsePrice(price),
    };
  }

  // Format 2: "TAX 1 7.900 %" or "TAX3 4.350 %" (Walmart-style numeric tax code)
  m = text.match(/\bTAX\s*(\d+)\s+(\d+(?:\.\d+)?)\s*%/i);
  if (m) {
    return {
      code: m[1],
      rate: parseFloat(m[2]) / 100,
      amount: parsePrice(price),
    };
  }

  return null;
}

/**
 * Extract the tax code letter from a price suffix, e.g. "23.99 E" → "E".
 * Returns null if no tax code suffix is found.
 */
export function extractTaxCode(price: string | null): string | null {
  if (!price) return null;
  const m = price.trim().match(/[ -]([A-Z])$/i);
  return m ? m[1].toUpperCase() : null;
}


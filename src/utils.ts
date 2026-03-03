/**
 * Shared types and pure utility functions used by the receipt-line
 * reconstruction algorithm.
 */

import { protos } from '@google-cloud/vision';

type TextAnnotation = protos.google.cloud.vision.v1.IEntityAnnotation;

// ── Core geometry types ───────────────────────────────────────────────────────

export interface Point {
  x: number;
  y: number;
}

/**
 * Axis-aligned summary of a single OCR word box, plus a back-reference
 * to the original Vision API annotation (needed for bounding-box drawing).
 */
export interface WordBox {
  text: string;
  center: Point;
  left: number;
  right: number;
  top: number;
  bottom: number;
  width: number;
  height: number;
  vertices: Point[];
  /** The raw Vision API annotation this box was derived from. */
  original: TextAnnotation;
}

// ── Receipt-line types ────────────────────────────────────────────────────────

export type LineType =
  | 'untaxed_item'
  | 'taxed_item'
  | 'discount'
  | 'tax'
  | 'total'
  | 'subtotal'
  | 'info'
  | 'wrapped';

export interface ReceiptLine {
  /** Ordered word boxes (left → right) that make up this line. */
  words: WordBox[];
  /** Full text of the line (words joined by spaces / merged fragments). */
  text: string;
  /** The item-name portion (everything to the left of the price). */
  itemName: string | null;
  /** The detected price token, e.g. "$4.99", or null if none found. */
  price: string | null;
  /** Semantic classification of the line. */
  lineType: LineType;
  /** Per-line baseline angle (radians) estimated from word positions. */
  angle: number;
}

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
 * `12.34`  `-12.34`  `12.34-`  `$1,234.56`  `-$1,234.56`  `(1,234.56)`  `($1,234.56)`  `(1,234.56)-`  `12.34 A`  `12.34-A`
 */
const PRICE_REGEX = /^(?:(?:-?\$?\d{1,3}(?:,\d{3})*\.\d{2})|\(\$?\d{1,3}(?:,\d{3})*\.\d{2}\))(?:-(?!A))?(?:[ -]A)?$/;

export function isPrice(text: string): boolean {
  return PRICE_REGEX.test(text.trim());
}

// ── Line classification by keywords ───────────────────────────────────────────

const SUBTOTAL_KEYWORDS = ['subtotal', 'sub-total', 'sub total'];
const TOTAL_KEYWORDS = ['total', 'amount due', 'balance', 'grand total'];
const TAX_KEYWORDS = ['tax', 'hst', 'gst', 'pst', 'vat'];
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
const TAXED_ITEM_KEYWORDS_PRICE = [
  'A'
];

/**
 * Payment-method keywords.  Lines containing these are credit/debit card
 * charges, cash-back, or auth-code receipts — never actual item lines.
 * Checked AFTER subtotal/total/tax/discount so those still take priority
 * (e.g. "VISA TOTAL $17.49" is classified as 'total', not payment).
 */
const PAYMENT_KEYWORDS = [
  'visa', 'mastercard', 'master card', 'amex', 'american express',
  'discover', 'debit', 'charge', 'change due', 'auth', 'bal due',
];

/**
 * Classify a receipt line by its text content.
 * Check order matters: "subtotal" must be checked before "total".
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
    return 'total';
  }
  if (TAX_KEYWORDS.some((k) => lower.includes(k))) return 'tax';
  if (DISCOUNT_KEYWORDS.some((k) => lower.includes(k))) return 'discount';
  if (PAYMENT_KEYWORDS.some((k) => lower.includes(k))) return 'info';
  // "Sale Price" lines show the post-discount price — informational, not a separate item
  if (lower.includes('sale price')) return 'info';
  if (!price) return 'info';
  if (DISCOUNT_KEYWORDS_PRICE.some((k) => price.includes(k))) return 'discount';
  if (TAXED_ITEM_KEYWORDS_PRICE.some((k) => price.includes(k))) return 'taxed_item';
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

// ── Receipt result types ──────────────────────────────────────────────────────

export type Severity = 'info' | 'warn' | 'error';

/**
 * A single composable check result.
 * Follows the ChatGPT "small composable rules" pattern:
 * each check returns a severity, a delta (how far off in $), a message,
 * and a penalty used for overall scoring.
 */
export interface CheckResult {
  /** Machine-readable check identifier. */
  id: string;
  /** Severity: info = FYI, warn = something might be off, error = definite mismatch. */
  severity: Severity;
  /** Human-readable explanation of the result. */
  message: string;
  /** Dollar amount the check is off by (absolute value), if applicable. */
  delta?: number;
  /** Numeric penalty for overall scoring (0 = perfect, higher = worse). */
  penalty: number;
}

/** A receipt item with its name, price, and optional discount applied. */
export interface ReceiptItem {
  /** Item name from OCR. */
  name: string;
  /** Parsed numeric price before discount (always positive). */
  originalPrice: number;
  /** Discount amount (negative number, 0 if no discount). */
  discount: number;
  /** Final price after discount: originalPrice + discount. */
  finalPrice: number;
  /** Whether this item is taxed (price ends with 'A'). */
  taxed: boolean;
  /** Raw price string from OCR. */
  rawPrice: string;
  /** Raw discount price string from OCR, if a discount was applied. */
  rawDiscount: string | null;
}

/**
 * Confidence report — contains composable check results, legacy booleans,
 * an overall 0–1 confidence score, and human-readable notes/warnings.
 */
export interface ReceiptConfidence {
  // ── Legacy boolean flags (null = check not applicable) ──
  /** TOTAL − TAX === OCR SUBTOTAL (if all three exist). */
  totalMinusTaxEqualsOcrSubtotal: boolean | null;
  /** Calculated subtotal (sum of items after discounts) === OCR SUBTOTAL. */
  calculatedSubtotalEqualsOcrSubtotal: boolean | null;
  /** Calculated subtotal === TOTAL − TAX. */
  calculatedSubtotalEqualsTotalMinusTax: boolean | null;

  // ── New composable checks ──
  /** All individual check results (composable rules). */
  checks: CheckResult[];
  /** Overall confidence score: 0 = no confidence, 1 = perfect. */
  overallScore: number;

  /** Informational notes (e.g. inferred values). */
  notes: string[];
  /** Actionable warnings when something doesn't add up. */
  warnings: string[];
}

/** The fully parsed receipt returned by detectTextLocal. */
export interface Receipt {
  /** All reconstructed lines (for reference / annotation). */
  lines: ReceiptLine[];
  /** Estimated rotation angle (radians). */
  angle: number;

  /** Parsed line items with discounts applied. */
  items: ReceiptItem[];

  // ── Counts ──
  totalLines: number;
  totalItems: number;
  totalUntaxedItems: number;
  totalTaxedItems: number;

  // ── Values ──
  /** Sum of all untaxed item final prices (after discounts). */
  untaxedItemsValue: number;
  /** Sum of all taxed item final prices (after discounts). */
  taxedItemsValue: number;

  // ── OCR-extracted summary values (null if not present on receipt) ──
  ocrSubtotal: number | null;
  ocrTax: number | null;
  ocrTotal: number | null;

  /** Calculated subtotal: untaxedItemsValue + taxedItemsValue. */
  calculatedSubtotal: number;
  /** Calculated tax rate: ocrTax / taxedItemsValue, or null if not computable. */
  taxRate: number | null;

  /** Confidence checks. */
  confidence: ReceiptConfidence;
}

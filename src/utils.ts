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
  | 'item'
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
 *   4.99  $4.99  -$4.99  $4.99-  ($4.99)  $1,234.99
 */
const PRICE_REGEX = /^-?\(?\$?\d{1,3}(,\d{3})*\.\d{2}\)?-?$/;

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

/**
 * Classify a receipt line by its text content.
 * Check order matters: "subtotal" must be checked before "total".
 */
export function classifyLine(text: string, hasPrice: boolean): LineType {
  const lower = text.toLowerCase();

  if (SUBTOTAL_KEYWORDS.some((k) => lower.includes(k))) return 'subtotal';
  if (TOTAL_KEYWORDS.some((k) => lower.includes(k))) return 'total';
  if (TAX_KEYWORDS.some((k) => lower.includes(k))) return 'tax';
  if (DISCOUNT_KEYWORDS.some((k) => lower.includes(k))) return 'discount';
  if (!hasPrice) return 'info';
  return 'item';
}

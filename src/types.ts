/**
 * Shared type definitions for the receipt OCR parser.
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
  | 'tender'
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

// ── Confidence & check types ──────────────────────────────────────────────────

export type Severity = 'info' | 'warn' | 'error';

/**
 * A single composable check result.
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

/** Explicit tax rate information parsed from receipt tax breakdown lines. */
export interface TaxRateInfo {
  /** Tax code letter, e.g. "A", "E". */
  code: string;
  /** Tax rate as a decimal, e.g. 0.085 for 8.50%. */
  rate: number;
  /** Dollar amount of tax for this code. */
  amount: number;
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
  /** Whether this item is taxed. */
  taxed: boolean;
  /** Tax code letter if detected (e.g. "A", "E"), or null. */
  taxCode: string | null;
  /** Tax rate for this item's group (decimal, e.g. 0.085), or null if untaxed. */
  taxRate: number | null;
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
  /** TOTAL − TAX === OCR SUBTOTAL (if all three exist). */
  totalMinusTaxEqualsOcrSubtotal: boolean | null;
  /** Calculated subtotal (sum of items after discounts) === OCR SUBTOTAL. */
  calculatedSubtotalEqualsOcrSubtotal: boolean | null;
  /** Calculated subtotal === TOTAL − TAX. */
  calculatedSubtotalEqualsTotalMinusTax: boolean | null;
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
  /** Detected store name, or null if unknown. */
  detectedStore: string | null;
  /** Parsed line items with discounts applied. */
  items: ReceiptItem[];
  totalLines: number;
  totalItems: number;
  totalUntaxedItems: number;
  totalTaxedItems: number;
  /** Sum of all untaxed item final prices (after discounts). */
  untaxedItemsValue: number;
  /** Sum of all taxed item final prices (after discounts). */
  taxedItemsValue: number;
  ocrSubtotal: number | null;
  ocrTax: number | null;
  ocrTotal: number | null;
  /** Calculated subtotal: untaxedItemsValue + taxedItemsValue. */
  calculatedSubtotal: number;
  /** Calculated tax rate: ocrTax / taxedItemsValue, or null if not computable. */
  taxRate: number | null;
  /** Explicit tax rates parsed from receipt breakdown lines. */
  taxRates: TaxRateInfo[];
  /** Largest tender / payment amount detected, or null. */
  tenderAmount: number | null;
  /** Confidence checks. */
  confidence: ReceiptConfidence;
}

export interface DebugReceipt extends Receipt {
  times: {
    type: string;
    elapsed: number;
  }[];
}

// ── Algorithm types ───────────────────────────────────────────────────────────

export interface ReconstructionResult {
  /** Ordered receipt lines, top-to-bottom. */
  lines: ReceiptLine[];
  /** Median of per-line angles (radians); representative global angle. */
  angle: number;
}

/**
 * Composable confidence-check pipeline for receipt price validation.
 *
 * Each check is a small, independent function that returns a CheckResult.
 * After all checks run, an overall 0–1 score is computed, and a
 * user-friendly summary string is produced with colored output.
 */

import type { CheckResult, ReceiptConfidence, Severity, TaxRateInfo, ReceiptItem } from './types';
import {
  RESET, BOLD, GREEN, YELLOW, RED, BG_GREEN, BG_YELLOW, BG_RED,
  colorPass, colorWarn, colorError, colorBold, colorDim, colorCyan,
} from './colors';

export { colorPass, colorWarn, colorError, colorBold, colorDim, colorCyan };

function colorBySeverity(sev: Severity, text: string): string {
  switch (sev) {
    case 'info':  return colorPass(text);
    case 'warn':  return colorWarn(text);
    case 'error': return colorError(text);
  }
}

function severityLabel(sev: Severity): string {
  switch (sev) {
    case 'info':  return colorPass(`${BG_GREEN}${BOLD} PASS ${RESET}`);
    case 'warn':  return colorWarn(`${BG_YELLOW}${BOLD} WARN ${RESET}`);
    case 'error': return colorError(`${BG_RED}${BOLD} FAIL ${RESET}`);
  }
}

// ── Tolerance & constants ────────────────────────────────────────────────────

const CENTS_TOLERANCE = 0.015;
/** Maximum plausible US sales-tax rate (15%). */
const MAX_PLAUSIBLE_TAX_RATE = 0.15;

function approxEqual(a: number, b: number): boolean {
  return Math.abs(a - b) < CENTS_TOLERANCE;
}

// ── Check context ────────────────────────────────────────────────────────────

export interface CheckContext {
  calculatedSubtotal: number;
  taxedItemsValue: number;
  untaxedItemsValue: number;
  ocrSubtotal: number | null;
  ocrTax: number | null;
  ocrTotal: number | null;
  taxRate: number | null;
  /** Explicit tax rate breakdowns from receipt (e.g. "A 8.50% TAX 0.55"). */
  taxRates: TaxRateInfo[];
  /** Parsed items with tax codes. */
  items: ReceiptItem[];
  /** Tender amount from payment lines (card charge, cash, etc.) if found. */
  tenderAmount: number | null;
}

// ── Individual composable checks ─────────────────────────────────────────────

/** 1) T ≈ S + X — do the three OCR summary lines agree internally? */
function checkTotalEqualsSubtotalPlusTax(ctx: CheckContext): CheckResult | null {
  const { ocrTotal, ocrSubtotal, ocrTax } = ctx;
  if (ocrTotal === null || ocrSubtotal === null || ocrTax === null) return null;
  const expected = ocrSubtotal + ocrTax;
  const delta = Math.abs(ocrTotal - expected);
  const ok = delta < CENTS_TOLERANCE;
  return {
    id: 'total_eq_subtotal_plus_tax',
    severity: ok ? 'info' : 'error',
    message: ok
      ? `TOTAL ($${ocrTotal.toFixed(2)}) = SUBTOTAL ($${ocrSubtotal.toFixed(2)}) + TAX ($${ocrTax.toFixed(2)})`
      : `TOTAL ($${ocrTotal.toFixed(2)}) != SUBTOTAL ($${ocrSubtotal.toFixed(2)}) + TAX ($${ocrTax.toFixed(2)}) — off by $${delta.toFixed(2)}`,
    delta,
    penalty: ok ? 0 : 10 + delta * 100,
  };
}

/** 2) S ≈ TaxedBase + UntaxedBase — taxability balance. */
function checkTaxabilityBalance(ctx: CheckContext): CheckResult | null {
  const { ocrSubtotal, taxedItemsValue, untaxedItemsValue } = ctx;
  let reference = ocrSubtotal;
  if (reference === null && ctx.ocrTotal !== null && ctx.ocrTax !== null) {
    reference = ctx.ocrTotal - ctx.ocrTax;
  }
  if (reference === null) return null;

  const itemsSum = taxedItemsValue + untaxedItemsValue;
  const delta = Math.abs(itemsSum - reference);
  const ok = delta < CENTS_TOLERANCE;
  return {
    id: 'taxability_balance',
    severity: ok ? 'info' : (delta > 1 ? 'error' : 'warn'),
    message: ok
      ? `Taxed ($${taxedItemsValue.toFixed(2)}) + Untaxed ($${untaxedItemsValue.toFixed(2)}) = reference subtotal ($${reference.toFixed(2)})`
      : `Taxed ($${taxedItemsValue.toFixed(2)}) + Untaxed ($${untaxedItemsValue.toFixed(2)}) = $${itemsSum.toFixed(2)} != reference subtotal ($${reference.toFixed(2)}) — off by $${delta.toFixed(2)}`,
    delta,
    penalty: ok ? 0 : 5 + delta * 50,
  };
}

/** 3) Calculated subtotal ≈ OCR subtotal — did we find all the items? */
function checkCalcSubtotalVsOcrSubtotal(ctx: CheckContext): CheckResult | null {
  const { calculatedSubtotal, ocrSubtotal } = ctx;
  if (ocrSubtotal === null) return null;
  const delta = Math.abs(calculatedSubtotal - ocrSubtotal);
  const ok = delta < CENTS_TOLERANCE;
  const direction = calculatedSubtotal < ocrSubtotal ? 'less' : 'more';
  return {
    id: 'calc_subtotal_vs_ocr_subtotal',
    severity: ok ? 'info' : (delta > 5 ? 'error' : 'warn'),
    message: ok
      ? `Calculated subtotal ($${calculatedSubtotal.toFixed(2)}) matches OCR SUBTOTAL ($${ocrSubtotal.toFixed(2)})`
      : `Calculated subtotal ($${calculatedSubtotal.toFixed(2)}) is $${delta.toFixed(2)} ${direction} than OCR SUBTOTAL ($${ocrSubtotal.toFixed(2)}) — we may have ${direction === 'less' ? 'missed items' : 'extra items'}`,
    delta,
    penalty: ok ? 0 : 8 + delta * 80,
  };
}

/** 4) Calculated subtotal ≈ T − X — cross-check via total & tax. */
function checkCalcSubtotalVsTotalMinusTax(ctx: CheckContext): CheckResult | null {
  const { calculatedSubtotal, ocrTotal, ocrTax } = ctx;
  if (ocrTotal === null) return null;
  const reference = ocrTax !== null ? ocrTotal - ocrTax : ocrTotal;
  const label = ocrTax !== null ? `TOTAL - TAX ($${reference.toFixed(2)})` : `TOTAL ($${ocrTotal.toFixed(2)}) [no tax line]`;
  const delta = Math.abs(calculatedSubtotal - reference);
  const ok = delta < CENTS_TOLERANCE;
  const direction = calculatedSubtotal < reference ? 'less' : 'more';
  return {
    id: 'calc_subtotal_vs_total_minus_tax',
    severity: ok ? 'info' : (delta > 5 ? 'error' : 'warn'),
    message: ok
      ? `Calculated subtotal ($${calculatedSubtotal.toFixed(2)}) matches ${label}`
      : `Calculated subtotal ($${calculatedSubtotal.toFixed(2)}) is $${delta.toFixed(2)} ${direction} than ${label} — we may have ${direction === 'less' ? 'missed items' : 'extra items'}`,
    delta,
    penalty: ok ? 0 : 8 + delta * 80,
  };
}

/** 5) X ≈ TaxedBase × R — tax amount consistency. */
function checkTaxConsistency(ctx: CheckContext): CheckResult | null {
  // If explicit per-code tax rates exist, skip the single-rate check
  // (the per-code check handles it instead).
  if (ctx.taxRates.length > 0) return null;

  const { ocrTax, taxedItemsValue, taxRate } = ctx;
  if (ocrTax === null || taxRate === null || taxedItemsValue <= 0) return null;
  const expectedTax = Math.round(taxedItemsValue * taxRate * 100) / 100;
  const delta = Math.abs(ocrTax - expectedTax);
  const ok = delta < CENTS_TOLERANCE;
  return {
    id: 'tax_consistency',
    severity: ok ? 'info' : (delta > 0.1 ? 'error' : 'warn'),
    message: ok
      ? `OCR TAX ($${ocrTax.toFixed(2)}) ~= taxed items ($${taxedItemsValue.toFixed(2)}) x rate (${(taxRate * 100).toFixed(2)}%) = $${expectedTax.toFixed(2)}`
      : `OCR TAX ($${ocrTax.toFixed(2)}) != taxed items ($${taxedItemsValue.toFixed(2)}) x rate (${(taxRate * 100).toFixed(2)}%) = $${expectedTax.toFixed(2)} — off by $${delta.toFixed(2)} (possible per-line rounding)`,
    delta,
    penalty: ok ? 0 : 3 + delta * 200,
  };
}

/** 5b) Per-code tax consistency — verify each explicit tax code. */
function checkPerCodeTaxConsistency(ctx: CheckContext): CheckResult[] {
  if (ctx.taxRates.length === 0) return [];

  const results: CheckResult[] = [];
  for (const tr of ctx.taxRates) {
    // Sum items with this tax code (null tax code maps to "" group)
    const codeItems = ctx.items.filter((i) => (i.taxCode ?? '') === tr.code);
    const codeBase = codeItems.reduce((s, i) => s + i.finalPrice, 0);

    if (codeBase <= 0) {
      results.push({
        id: `tax_consistency_${tr.code}`,
        severity: 'warn',
        message: `Tax code ${tr.code} (${(tr.rate * 100).toFixed(2)}%): no items found with this code, but receipt shows $${tr.amount.toFixed(2)} tax`,
        penalty: 2,
      });
      continue;
    }

    const expectedTax = Math.round(codeBase * tr.rate * 100) / 100;
    const delta = Math.abs(tr.amount - expectedTax);
    const ok = delta < CENTS_TOLERANCE;
    results.push({
      id: `tax_consistency_${tr.code}`,
      severity: ok ? 'info' : (delta > 0.1 ? 'error' : 'warn'),
      message: ok
        ? `Tax code ${tr.code}: $${tr.amount.toFixed(2)} ~= items ($${codeBase.toFixed(2)}) x ${(tr.rate * 100).toFixed(2)}% = $${expectedTax.toFixed(2)}`
        : `Tax code ${tr.code}: $${tr.amount.toFixed(2)} != items ($${codeBase.toFixed(2)}) x ${(tr.rate * 100).toFixed(2)}% = $${expectedTax.toFixed(2)} — off by $${delta.toFixed(2)}`,
      delta,
      penalty: ok ? 0 : 3 + delta * 200,
    });
  }
  return results;
}

/** 6) Is the inferred tax rate plausible (0 – 15%)? */
function checkTaxRatePlausibility(ctx: CheckContext): CheckResult[] {
  // If explicit per-code rates exist, check each one
  if (ctx.taxRates.length > 0) {
    return ctx.taxRates.map((tr) => {
      const pct = tr.rate * 100;
      const ok = tr.rate >= 0 && tr.rate <= MAX_PLAUSIBLE_TAX_RATE;
      return {
        id: `tax_rate_plausibility_${tr.code}`,
        severity: 'info' as Severity,
        message: ok
          ? `Tax code ${tr.code} rate ${pct.toFixed(2)}% is plausible (0-15%)`
          : `Tax code ${tr.code} rate ${pct.toFixed(2)}% is outside plausible range (0-15%)`,
        penalty: 0,
      };
    });
  }

  const { taxRate } = ctx;
  if (taxRate === null) return [];
  const pct = taxRate * 100;
  const ok = taxRate >= 0 && taxRate <= MAX_PLAUSIBLE_TAX_RATE;
  return [{
    id: 'tax_rate_plausibility',
    severity: 'info',
    message: ok
      ? `Inferred tax rate ${pct.toFixed(2)}% is plausible (0-15%)`
      : `Inferred tax rate ${pct.toFixed(2)}% is outside plausible range (0-15%) — possible taxability misclassification or OCR error`,
    penalty: 0,
  }];
}

/** 7) Solve-for-missing-one — estimate what's missing. */
function checkMissingElement(ctx: CheckContext): CheckResult | null {
  const { calculatedSubtotal, ocrSubtotal, ocrTotal, ocrTax } = ctx;
  let reference: number | null = ocrSubtotal;
  if (reference === null && ocrTotal !== null) {
    reference = ocrTax !== null ? ocrTotal - ocrTax : ocrTotal;
  }
  if (reference === null) return null;

  const diff = reference - calculatedSubtotal;
  if (Math.abs(diff) < CENTS_TOLERANCE) return null;

  const absDiff = Math.abs(diff);
  const looksLikeMoney = Math.abs(absDiff - Math.round(absDiff * 100) / 100) < 0.001;

  if (diff > 0) {
    return {
      id: 'missing_item_estimate',
      severity: 'warn',
      message: `Subtotal mismatch suggests a missing item worth ~$${absDiff.toFixed(2)}${looksLikeMoney ? '' : ' (unusual amount)'}`,
      delta: absDiff,
      penalty: 6 + absDiff * 20,
    };
  } else {
    return {
      id: 'missing_discount_estimate',
      severity: 'warn',
      message: `Subtotal mismatch suggests a missing discount of ~$${absDiff.toFixed(2)} or an extra item was included`,
      delta: absDiff,
      penalty: 6 + absDiff * 20,
    };
  }
}

/** 8) Missing summary lines — note which OCR values couldn't be found. */
function checkMissingSummaryLines(ctx: CheckContext): CheckResult[] {
  const results: CheckResult[] = [];
  if (ctx.ocrSubtotal === null) {
    // Subtotal is derivable when we have TOTAL and TAX, or cross-checkable
    // when we at least have TOTAL.  In either case the missing line is
    // purely informational.
    const derivable = ctx.ocrTotal !== null && ctx.ocrTax !== null;
    const crossCheckable = ctx.ocrTotal !== null;
    results.push({
      id: 'missing_subtotal',
      severity: derivable || crossCheckable ? 'info' : 'warn',
      message: derivable
        ? 'No SUBTOTAL line found on receipt — using TOTAL − TAX as reference'
        : crossCheckable
          ? 'No SUBTOTAL line found on receipt — using TOTAL for cross-check'
          : 'No SUBTOTAL line found on receipt — cross-checks limited',
      penalty: 0,
    });
  }
  if (ctx.ocrTax === null) {
    // Tax is irrelevant when there are no taxed items, OR when the
    // calculated subtotal already matches the total (tax is zero or
    // included in prices, e.g. VAT-inclusive receipts).
    const irrelevant = ctx.taxedItemsValue <= 0 ||
      (ctx.ocrTotal !== null && Math.abs(ctx.calculatedSubtotal - ctx.ocrTotal) < CENTS_TOLERANCE);
    results.push({
      id: 'missing_tax',
      severity: 'info',
      message: irrelevant
        ? 'No TAX line found on receipt — no taxed items detected so tax is irrelevant'
        : 'No TAX line found on receipt — tax rate cannot be verified',
      penalty: irrelevant ? 0 : 1,
    });
  }
  if (ctx.ocrTotal === null) {
    results.push({
      id: 'missing_total',
      severity: 'warn',
      message: 'No TOTAL line found on receipt — cannot verify final amount',
      penalty: 3,
    });
  }
  return results;
}

/** 9) Tender cross-check — does the payment amount match the total? */
function checkTenderVsTotal(ctx: CheckContext): CheckResult | null {
  const { tenderAmount, ocrTotal } = ctx;
  if (tenderAmount === null || ocrTotal === null) return null;
  const delta = Math.abs(tenderAmount - ocrTotal);
  const ok = delta < CENTS_TOLERANCE;
  // Tender cross-check is informational only — split payments, cash back,
  // food stamps, and tips all cause legitimate mismatches.  A match is a
  // nice confirmation; a mismatch is just a note, never a score penalty.
  return {
    id: 'tender_vs_total',
    severity: 'info',
    message: ok
      ? `Tender amount ($${tenderAmount.toFixed(2)}) matches TOTAL ($${ocrTotal.toFixed(2)})`
      : `Tender amount ($${tenderAmount.toFixed(2)}) != TOTAL ($${ocrTotal.toFixed(2)}) — off by $${delta.toFixed(2)} (could indicate tip, split payment, or cash/change mismatch)`,
    delta,
    penalty: 0,
  };
}

// ── Overall confidence score ─────────────────────────────────────────────────

function computeOverallScore(checks: CheckResult[], ctx: CheckContext): number {
  const totalPenalty = checks.reduce((s, c) => s + c.penalty, 0);

  // A summary line is truly missing only when it can't be derived from the
  // others.  SUBTOTAL is covered when TOTAL is present (cross-checkable).
  // TAX is irrelevant when there are no taxed items, OR when the calculated
  // subtotal already matches the total (tax is zero or included in prices).
  const subtotalCoverable = ctx.ocrSubtotal === null && ctx.ocrTotal !== null;
  const taxIrrelevant = ctx.ocrTax === null && (
    ctx.taxedItemsValue <= 0 ||
    (ctx.ocrTotal !== null && Math.abs(ctx.calculatedSubtotal - ctx.ocrTotal) < CENTS_TOLERANCE)
  );
  const missingCount =
    (ctx.ocrSubtotal === null && !subtotalCoverable ? 1 : 0) +
    (ctx.ocrTax === null && !taxIrrelevant ? 1 : 0) +
    (ctx.ocrTotal === null ? 1 : 0);

  const fitQuality = Math.exp(-totalPenalty / 50);
  const missingPenalty = Math.exp(-missingCount / 3);

  const hasError = checks.some((c) => c.severity === 'error');
  const hasWarn = checks.some((c) => c.severity === 'warn');
  const severityCap = hasError ? 0.6 : hasWarn ? 0.85 : 1.0;

  return Math.min(fitQuality * missingPenalty, severityCap);
}

// ── Main confidence pipeline ─────────────────────────────────────────────────

/**
 * Run all composable confidence checks and produce a ReceiptConfidence report.
 */
export function checkConfidence(
  calculatedSubtotal: number,
  taxedItemsValue: number,
  untaxedItemsValue: number,
  ocrSubtotal: number | null,
  ocrTax: number | null,
  ocrTotal: number | null,
  taxRate: number | null,
  tenderAmount: number | null = null,
  taxRates: TaxRateInfo[] = [],
  items: ReceiptItem[] = [],
): ReceiptConfidence {
  const ctx: CheckContext = {
    calculatedSubtotal,
    taxedItemsValue,
    untaxedItemsValue,
    ocrSubtotal,
    ocrTax,
    ocrTotal,
    taxRate,
    taxRates,
    items,
    tenderAmount,
  };

  const allChecks: (CheckResult | null)[] = [
    checkTotalEqualsSubtotalPlusTax(ctx),
    checkTaxabilityBalance(ctx),
    checkCalcSubtotalVsOcrSubtotal(ctx),
    checkCalcSubtotalVsTotalMinusTax(ctx),
    checkTaxConsistency(ctx),
    ...checkPerCodeTaxConsistency(ctx),
    ...checkTaxRatePlausibility(ctx),
    checkTenderVsTotal(ctx),
    checkMissingElement(ctx),
    ...checkMissingSummaryLines(ctx),
  ];

  const checks = allChecks.filter((c): c is CheckResult => c !== null);

  const findBool = (id: string): boolean | null => {
    const c = checks.find((ch) => ch.id === id);
    return c ? c.severity === 'info' : null;
  };

  const totalMinusTaxEqualsOcrSubtotal = findBool('total_eq_subtotal_plus_tax');
  const calculatedSubtotalEqualsOcrSubtotal = findBool('calc_subtotal_vs_ocr_subtotal');
  const calculatedSubtotalEqualsTotalMinusTax = findBool('calc_subtotal_vs_total_minus_tax');

  const notes = checks
    .filter((c) => c.severity === 'info')
    .map((c) => c.message);
  const warnings = checks
    .filter((c) => c.severity === 'warn' || c.severity === 'error')
    .map((c) => `[${c.severity.toUpperCase()}] ${c.message}`);

  const overallScore = computeOverallScore(checks, ctx);

  return {
    totalMinusTaxEqualsOcrSubtotal,
    calculatedSubtotalEqualsOcrSubtotal,
    calculatedSubtotalEqualsTotalMinusTax,
    checks,
    overallScore,
    notes,
    warnings,
  };
}

// ── User-friendly suggestion mapping ─────────────────────────────────────────

/**
 * Returns a user-friendly suggestion for a given check result.
 * For passing checks this is an assurance statement.
 * For warnings/errors this is actionable guidance.
 */
function getSuggestion(chk: CheckResult): string {
  switch (chk.id) {
    // ── Check 1: T = S + X ──────────────────────────────────────────────
    case 'total_eq_subtotal_plus_tax':
      if (chk.severity === 'info') {
        return 'The receipt total, subtotal, and tax are internally consistent — these OCR values can be trusted.';
      }
      return 'The receipt\'s own subtotal + tax doesn\'t equal the total. This usually means the OCR misread one of these three lines. Visually verify the subtotal, tax, and total on the receipt image.';

    // ── Check 2: Taxability balance ─────────────────────────────────────
    case 'taxability_balance':
      if (chk.severity === 'info') {
        return 'Taxed + untaxed item totals match the expected subtotal — item categorization looks correct.';
      }
      if (chk.severity === 'warn') {
        return 'The taxed + untaxed item totals are slightly off from the expected subtotal. A small rounding difference or one misclassified item is likely. Review any items near the threshold.';
      }
      return 'The taxed + untaxed totals are significantly off. Some items may be misclassified as taxed/untaxed, or an item price was misread. Check items marked with [T] against the receipt.';

    // ── Check 3: Calc subtotal vs OCR subtotal ──────────────────────────
    case 'calc_subtotal_vs_ocr_subtotal':
      if (chk.severity === 'info') {
        return 'All detected item prices add up to the receipt subtotal — no items appear to be missing or duplicated.';
      }
      if (chk.severity === 'warn') {
        return `The items we found are $${chk.delta?.toFixed(2) ?? '?'} off from the receipt subtotal. Check the receipt image for any items that may have been missed or misread by OCR.`;
      }
      return `The items we found are $${chk.delta?.toFixed(2) ?? '?'} off from the receipt subtotal — a significant discrepancy. There may be missing items, duplicated items, or OCR misreads. Compare the item list against the original receipt.`;

    // ── Check 4: Calc subtotal vs T - X ─────────────────────────────────
    case 'calc_subtotal_vs_total_minus_tax':
      if (chk.severity === 'info') {
        return 'Item prices match the total minus tax — this cross-check confirms the subtotal is accurate.';
      }
      if (chk.severity === 'warn') {
        return `Items are $${chk.delta?.toFixed(2) ?? '?'} off from total minus tax. If the subtotal check also failed, consider that an item or discount may have been missed.`;
      }
      return `Items are $${chk.delta?.toFixed(2) ?? '?'} off from total minus tax — a significant gap. Cross-reference the printed receipt to find the discrepancy.`;

    // ── Check 5: Tax consistency ────────────────────────────────────────
    case 'tax_consistency':
      if (chk.severity === 'info') {
        return 'The tax amount matches what we\'d expect from the tax rate and taxed items — tax calculations look correct.';
      }
      if (chk.severity === 'warn') {
        return `The tax amount is off by $${chk.delta?.toFixed(2) ?? '?'} from the expected value. This is often caused by per-line tax rounding vs. global rounding and is usually not an issue.`;
      }
      return `The tax amount is significantly off from the expected value. Either the tax rate, the taxable item classification, or the OCR'd tax amount may be wrong. Verify the tax line on the receipt.`;

    // ── Check 6: Tax rate plausibility ──────────────────────────────────
    case 'tax_rate_plausibility':      if (chk.severity === 'info') {
        return 'The inferred tax rate is within the normal US range (0-15%) — looks reasonable.';
      }
      return 'The inferred tax rate is outside the normal range. This probably means some items are misclassified as taxed/untaxed, or the tax line was misread. Check items with [T] flags.';

    // ── Check 7: Missing element estimates ──────────────────────────────
    case 'missing_item_estimate':
      return `The subtotal mismatch suggests we may be missing an item worth ~$${chk.delta?.toFixed(2) ?? '?'}. Look for any item on the receipt that wasn't detected.`;
    case 'missing_discount_estimate':
      return `The subtotal mismatch suggests an undetected discount of ~$${chk.delta?.toFixed(2) ?? '?'}, or an extra item was incorrectly included. Check for coupon or promo lines.`;

    // ── Check 8: Missing summary lines ──────────────────────────────────
    case 'missing_subtotal':
      return 'No subtotal line was found, which limits our ability to cross-check item prices. The receipt may not print a subtotal, or it was cut off / unreadable.';
    case 'missing_tax':
      return 'No tax line was found. The receipt may be tax-exempt, or the tax line was cut off. Tax rate cannot be independently verified.';
    case 'missing_total':
      return 'No total line was found. The receipt may be cut off at the bottom. We cannot verify the final amount.';

    // ── Check 9: Tender cross-check ─────────────────────────────────────
    case 'tender_vs_total':
      if (chk.severity === 'info') {
        return 'The payment amount matches the receipt total — this confirms the final charged amount.';
      }
      return `The payment amount differs from the total by $${chk.delta?.toFixed(2) ?? '?'}. This may indicate a tip, split payment, or cash/change math error. Compare the payment section on the receipt.`;

    default:
      // Per-code tax consistency checks (tax_consistency_A, tax_consistency_E, etc.)
      if (chk.id.startsWith('tax_consistency_')) {
        if (chk.severity === 'info') {
          return 'Per-code tax amount matches the expected calculation — tax breakdown is accurate.';
        }
        if (chk.severity === 'warn') {
          return `Per-code tax is off by $${chk.delta?.toFixed(2) ?? '?'} — could be per-line rounding or a misclassified item.`;
        }
        return `Per-code tax is significantly off. Check that items are assigned the correct tax code.`;
      }
      // Per-code tax rate plausibility checks
      if (chk.id.startsWith('tax_rate_plausibility_')) {
        return 'The per-code tax rate is within the normal US range (0-15%) — looks reasonable.';
      }
      return chk.message;
  }
}

// ── Score color ──────────────────────────────────────────────────────────────

function scoreColor(score: number): string {
  if (score >= 0.85) return `${GREEN}${BOLD}`;
  if (score >= 0.5)  return `${YELLOW}${BOLD}`;
  return `${RED}${BOLD}`;
}

// ── Formatted output ─────────────────────────────────────────────────────────

/**
 * Produces a fully colored, user-friendly summary string for the confidence
 * report.  Includes:
 * - Overall score with color
 * - Each check with PASS/WARN/FAIL badge + suggestion
 * - Final verdict
 */
export function formatConfidenceReport(confidence: ReceiptConfidence): string {
  const lines: string[] = [];
  const score = confidence.overallScore;
  const pct = (score * 100).toFixed(0);

  lines.push('');
  lines.push(`${colorBold('=== Confidence Report ===')}  Score: ${scoreColor(score)}${pct}%${RESET}`);
  lines.push('');

  // ── Individual checks ─────────────────────────────────────────────────
  for (const chk of confidence.checks) {
    const badge = severityLabel(chk.severity);
    const deltaStr = chk.delta !== undefined ? colorDim(` (Δ $${chk.delta.toFixed(2)})`) : '';
    lines.push(`  ${badge} ${colorBySeverity(chk.severity, chk.message)}${deltaStr}`);

    // Suggestion indented below the check
    const suggestion = getSuggestion(chk);
    const arrow = chk.severity === 'info'
      ? colorPass('    ✓ ')
      : chk.severity === 'warn'
        ? colorWarn('    → ')
        : colorError('    ✗ ');
    lines.push(`${arrow}${colorDim(suggestion)}`);
    lines.push('');
  }

  // ── Final verdict ─────────────────────────────────────────────────────
  lines.push(colorBold('--- Verdict ---'));

  const errors = confidence.checks.filter((c) => c.severity === 'error');
  const warns  = confidence.checks.filter((c) => c.severity === 'warn');
  const passes = confidence.checks.filter((c) => c.severity === 'info');

  if (errors.length === 0 && warns.length === 0) {
    lines.push(colorPass(
      `  All ${passes.length} checks passed. The receipt data looks accurate and can be trusted.`
    ));
  } else if (errors.length === 0) {
    lines.push(colorWarn(
      `  ${passes.length} checks passed, ${warns.length} warning(s). The receipt is mostly accurate but some values may need manual review.`
    ));
  } else {
    lines.push(colorError(
      `  ${errors.length} error(s), ${warns.length} warning(s), ${passes.length} passed. Significant discrepancies detected — manual review recommended.`
    ));
  }

  lines.push('');
  return lines.join('\n');
}

/**
 * Dynamic tax group determination for receipt parsing.
 *
 * Instead of static per-store tax code profiles, this module groups items
 * by their price suffix codes (e.g. "A", "T", "", etc.) and uses receipt
 * totals to determine which groups are taxed and at what rate.
 *
 * If the receipt includes explicit tax rate lines (e.g. "A 8.50% TAX 0.55"),
 * those tax groups are used directly.  Otherwise, it assumes a single tax
 * rate and determines which group of items is most likely taxed.
 */

import type { ReceiptItem, TaxRateInfo } from './types';

// ── Types ────────────────────────────────────────────────────────────────────

/** A group of items sharing the same tax code suffix. */
export interface TaxGroup {
  /** The tax code letter (e.g. "A", "T"), or "" for no suffix. */
  code: string;
  /** Items in this group. */
  items: ReceiptItem[];
  /** Sum of final prices in this group. */
  total: number;
  /** Whether this group was determined to be taxed. */
  taxed: boolean;
  /** The tax rate for this group (decimal, e.g. 0.085), or null. */
  rate: number | null;
}

/** Result of dynamic tax group analysis. */
export interface TaxGroupResult {
  /** All tax groups found on the receipt. */
  groups: TaxGroup[];
  /** Tax codes determined to be taxed. */
  taxedCodes: string[];
  /** Inferred single effective tax rate (decimal), or null. */
  effectiveTaxRate: number | null;
  /** Explicit per-code tax rates from receipt, if any. */
  explicitRates: TaxRateInfo[];
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Maximum plausible US sales-tax rate (15%). */
const MAX_PLAUSIBLE_TAX_RATE = 0.15;

/** Round to nearest 0.05% increment. */
function roundRate(raw: number): number {
  const pct = raw * 100;
  return Math.round(pct / 0.05) * 0.05 / 100;
}

/**
 * Find the best tax rate near rawRate that explains the observed tax amount.
 * Generates candidate rates at 0.05% increments plus nearby 0.25% multiples,
 * then picks the one with the smallest residual (tax mismatch) and, among
 * ties, the one closest to a 0.25% multiple (common US rate granularity).
 */
function findBestRate(rawRate: number, base: number, taxAmount: number): number {
  const rawBps = rawRate * 10000; // basis points (8.75% → 875)

  // Candidate rates in basis points
  const bpsCandidates = new Set<number>();
  // Nearest / floor / ceil at 5-bps (0.05%) granularity
  bpsCandidates.add(Math.round(rawBps / 5) * 5);
  bpsCandidates.add(Math.floor(rawBps / 5) * 5);
  bpsCandidates.add(Math.ceil(rawBps / 5) * 5);
  // Nearest 25-bps (0.25%) multiples
  bpsCandidates.add(Math.floor(rawBps / 25) * 25);
  bpsCandidates.add(Math.ceil(rawBps / 25) * 25);

  let bestRate = roundRate(rawRate); // fallback
  let bestScore = Infinity;

  for (const bps of bpsCandidates) {
    if (bps <= 0 || bps > MAX_PLAUSIBLE_TAX_RATE * 10000) continue;
    const rate = bps / 10000;
    const expectedTax = Math.round(base * rate * 100) / 100;
    const residual = Math.abs(expectedTax - taxAmount);
    // Distance to nearest 0.25% (25 bps)
    const dist25 = Math.abs(bps - Math.round(bps / 25) * 25);
    // Residual dominates; among equal residuals prefer rounder rates
    const score = residual * 10000 + dist25;
    if (score < bestScore) {
      bestScore = score;
      bestRate = rate;
    }
  }

  return bestRate;
}

// ── Main function ────────────────────────────────────────────────────────────

/**
 * Group items by tax code and determine which groups are taxed.
 *
 * @param items        Parsed receipt items (all initially untaxed).
 * @param ocrTax       OCR-detected tax amount, or null.
 * @param ocrSubtotal  OCR-detected subtotal, or null.
 * @param ocrTotal     OCR-detected total, or null.
 * @param explicitRates Explicit tax rate lines from the receipt (e.g. "A 8.50% TAX 0.55").
 */
export function determineTaxGroups(
  items: ReceiptItem[],
  ocrTax: number | null,
  ocrSubtotal: number | null,
  ocrTotal: number | null,
  explicitRates: TaxRateInfo[],
): TaxGroupResult {
  // ── Step 1: Group items by their tax code suffix ──────────────────────
  const groupMap = new Map<string, ReceiptItem[]>();
  for (const item of items) {
    const code = item.taxCode ?? '';
    if (!groupMap.has(code)) groupMap.set(code, []);
    groupMap.get(code)!.push(item);
  }

  const groups: TaxGroup[] = Array.from(groupMap.entries()).map(([code, groupItems]) => ({
    code,
    items: groupItems,
    // Round each item to cents before summing — stores compute tax on rounded prices
    total: groupItems.reduce((s, i) => s + Math.round(i.finalPrice * 100) / 100, 0),
    taxed: false,
    rate: null,
  }));

  // ── Step 2: Determine which groups are taxed ──────────────────────────

  // Compute tax amount early — needed by multiple strategies
  let taxAmount: number | null = ocrTax;
  if (taxAmount === null && ocrSubtotal !== null && ocrTotal !== null) {
    const implied = ocrTotal - ocrSubtotal;
    if (implied > 0) taxAmount = implied;
  }

  // Separate letter-coded rates (direct match) from numeric-coded rates (Walmart-style)
  const letterRates = explicitRates.filter((r) => /^[A-Z]$/i.test(r.code));
  const numericRates = explicitRates.filter((r) => !/^[A-Z]$/i.test(r.code));

  if (letterRates.length > 0) {
    // Letter codes — direct matching (e.g. "A 8.50% TAX")
    return handleExplicitRates(groups, letterRates);
  }

  if (numericRates.length > 0) {
    // Numeric codes — try amount-based matching (e.g. "TAX 1 6.500 %")
    const result = tryMatchNumericRates(groups, numericRates);
    if (result) return result;
    // Matching failed — fall through to single-rate inference
  }

  if (taxAmount === null || taxAmount <= 0) {
    // No tax detected, everything is untaxed
    return {
      groups,
      taxedCodes: [],
      effectiveTaxRate: null,
      explicitRates: [],
    };
  }

  return handleSingleTaxRate(groups, taxAmount);
}

// ── Strategy: numeric tax code matching (Walmart-style) ─────────────────────

/**
 * Try to match numeric tax codes (e.g. "TAX 1 6.500 %") to item groups
 * by finding which group's total best explains each tax amount.
 * Returns null if matching fails (falls through to single-rate inference).
 */
function tryMatchNumericRates(
  groups: TaxGroup[],
  numericRates: TaxRateInfo[],
): TaxGroupResult | null {
  // For each rate, find the best-matching group by amount
  const matches = new Map<TaxGroup, TaxRateInfo[]>();

  for (const rate of numericRates) {
    if (rate.rate <= 0) continue;
    const expectedBase = rate.amount / rate.rate;

    let bestGroup: TaxGroup | null = null;
    let bestDelta = Infinity;

    for (const group of groups) {
      if (group.total <= 0) continue;
      const delta = Math.abs(group.total - expectedBase) / Math.max(group.total, expectedBase);
      if (delta < 0.03 && delta < bestDelta) {
        bestDelta = delta;
        bestGroup = group;
      }
    }

    if (bestGroup) {
      if (!matches.has(bestGroup)) matches.set(bestGroup, []);
      matches.get(bestGroup)!.push(rate);
    } else {
      // Rate couldn't be matched to any group — can't do clean assignment
      return null;
    }
  }

  // All rates matched — apply assignments
  const taxedCodes: string[] = [];
  const resolvedRates: TaxRateInfo[] = [];
  let totalTaxAmount = 0;
  let totalTaxedBase = 0;

  for (const [group, rates] of matches) {
    group.taxed = true;
    group.rate = rates.reduce((sum, r) => sum + r.rate, 0);
    taxedCodes.push(group.code);
    const groupTax = rates.reduce((sum, r) => sum + r.amount, 0);
    totalTaxAmount += groupTax;
    totalTaxedBase += group.total;

    for (const rate of rates) {
      resolvedRates.push({
        code: group.code,
        rate: rate.rate,
        amount: rate.amount,
      });
    }
  }

  const effectiveTaxRate = totalTaxedBase > 0 ? roundRate(totalTaxAmount / totalTaxedBase) : null;

  return {
    groups,
    taxedCodes,
    effectiveTaxRate,
    explicitRates: resolvedRates,
  };
}

// ── Strategy: explicit tax rate breakdown lines ──────────────────────────────

function handleExplicitRates(
  groups: TaxGroup[],
  explicitRates: TaxRateInfo[],
): TaxGroupResult {
  const taxedCodes: string[] = [];
  let totalTaxAmount = 0;
  let totalTaxedBase = 0;

  for (const rate of explicitRates) {
    const group = groups.find((g) => g.code === rate.code);
    if (group) {
      group.taxed = true;
      group.rate = rate.rate;
      taxedCodes.push(rate.code);
      totalTaxAmount += rate.amount;
      totalTaxedBase += group.total;
    }
  }

  let effectiveTaxRate: number | null = null;
  if (totalTaxedBase > 0) {
    effectiveTaxRate = roundRate(totalTaxAmount / totalTaxedBase);
  }

  return {
    groups,
    taxedCodes,
    effectiveTaxRate,
    explicitRates,
  };
}

// ── Strategy: single tax rate inference ──────────────────────────────────────

function handleSingleTaxRate(
  groups: TaxGroup[],
  taxAmount: number,
): TaxGroupResult {
  // If there's only one group (all items have the same code or no code),
  // that group is the taxed group.
  if (groups.length === 1) {
    const group = groups[0];
    if (group.total > 0) {
      const rate = findBestRate(taxAmount / group.total, group.total, taxAmount);
      if (rate > 0 && rate <= MAX_PLAUSIBLE_TAX_RATE) {
        group.taxed = true;
        group.rate = rate;
        return {
          groups,
          taxedCodes: [group.code],
          effectiveTaxRate: rate,
          explicitRates: [],
        };
      }
    }
    return { groups, taxedCodes: [], effectiveTaxRate: null, explicitRates: [] };
  }

  // Multiple groups exist — try each individual group and each combination
  // to find the subset whose total yields the most plausible tax rate.
  const candidates: {
    codes: string[];
    rate: number;
    roundness: number;
  }[] = [];

  // Generate all non-empty subsets of groups
  const groupArr = groups.filter((g) => g.total > 0);
  const n = groupArr.length;

  for (let mask = 1; mask < (1 << n); mask++) {
    let subsetTotal = 0;
    const codes: string[] = [];
    for (let bit = 0; bit < n; bit++) {
      if (mask & (1 << bit)) {
        subsetTotal += groupArr[bit].total;
        codes.push(groupArr[bit].code);
      }
    }

    if (subsetTotal <= 0) continue;

    const rawRate = taxAmount / subsetTotal;
    if (rawRate <= 0 || rawRate > MAX_PLAUSIBLE_TAX_RATE) continue;

    const rate = findBestRate(rawRate, subsetTotal, taxAmount);
    if (rate <= 0 || rate > MAX_PLAUSIBLE_TAX_RATE) continue;

    // Compute how well this rate explains the tax amount
    const expectedTax = Math.round(subsetTotal * rate * 100) / 100;
    const residual = Math.abs(expectedTax - taxAmount);

    // "Roundness": prefer rates that are multiples of 0.25% (common US rates)
    const rateBps = Math.round(rate * 10000);
    const roundness025 = Math.abs(rateBps - Math.round(rateBps / 25) * 25) / 100;

    candidates.push({
      codes,
      rate,
      roundness: residual + roundness025 * 0.01,
    });
  }

  if (candidates.length === 0) {
    return { groups, taxedCodes: [], effectiveTaxRate: null, explicitRates: [] };
  }

  // Pick the candidate with the smallest roundness score (best fit)
  candidates.sort((a, b) => a.roundness - b.roundness);
  const best = candidates[0];

  const taxedCodes = best.codes;
  for (const group of groups) {
    if (taxedCodes.includes(group.code)) {
      group.taxed = true;
      group.rate = best.rate;
    }
  }

  return {
    groups,
    taxedCodes,
    effectiveTaxRate: best.rate,
    explicitRates: [],
  };
}

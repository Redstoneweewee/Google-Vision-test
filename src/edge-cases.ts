/**
 * Post-processing edge-case handlers for receipt line reconstruction.
 *
 * These functions operate on already-classified ReceiptLine arrays and
 * handle special patterns: voided entries, orphan prices, wrapped item
 * names, competing totals, combined TAX/BAL lines, etc.
 */

import type { ReceiptLine } from './types';
import { classifyLine, isPrice, parsePrice } from './utils';
import {
  WRAP_MAX_VERTICAL_GAP_FACTOR,
  WRAP_MAX_LEFT_ALIGN_FACTOR,
  ORPHAN_SEARCH_RADIUS,
} from './constants';

// ── Shared helpers ───────────────────────────────────────────────────────────

/**
 * Check if an item name is trivial (null, empty, or just special chars /
 * a single letter).  Lines like "$ 6.79" or "9.00 F" have itemName "$"
 * or "F" — these are really just orphan prices, not real items.
 */
function isTrivialItemName(name: string | null): boolean {
  if (name === null) return true;
  const cleaned = name.replace(/[^a-zA-Z0-9]/g, '');
  return cleaned.length <= 1;
}

// ── Voided entries ───────────────────────────────────────────────────────────

/**
 * Handle `** VOIDED ENTRY` markers: demote the preceding item to 'info'.
 */
export function handleVoidedEntries(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    handleVoidedEntries: scanning for void markers...`);
  for (let i = 0; i < lines.length; i++) {
    const lower = lines[i].text.toLowerCase();
    if (!/\bvoid(ed)?\b/.test(lower)) continue;

    console.debug(`[DEBUG]      Found void marker at line ${i}: "${lines[i].text}"`);
    // Demote the closest preceding item line
    for (let j = i - 1; j >= 0; j--) {
      if (['untaxed_item', 'taxed_item'].includes(lines[j].lineType)) {
        console.debug(`[DEBUG]        Voiding item at line ${j}: "${lines[j].text}" (${lines[j].price})`);
        lines[j].lineType = 'info';
        break;
      }
    }
    // The void marker itself becomes info
    if (lines[i].lineType !== 'info') {
      lines[i].lineType = 'info';
    }
  }
}

// ── Adjacent prices for keywords ─────────────────────────────────────────────

/**
 * Assign adjacent price-only info lines to priceless keyword / tender lines.
 * Handles receipt formats where the dollar amount appears as a separate
 * OCR text block on an adjacent info line (e.g. Trader Joe's).
 */
export function assignAdjacentPricesToKeywords(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    assignAdjacentPricesToKeywords: scanning for priceless keyword lines...`);

  const isPriceOnlyLine = (l: ReceiptLine): boolean => {
    if (l.lineType !== 'info') return false;
    const t = l.text.trim().replace(/^\$/, '');
    return t.length > 0 && isPrice(t);
  };

  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (!['subtotal', 'total', 'tax', 'tender'].includes(l.lineType)) continue;
    if (l.price !== null) continue; // already has a price

    // Check line BEFORE first (more reliable for interleaved layouts)
    if (i - 1 >= 0 && isPriceOnlyLine(lines[i - 1])) {
      l.price = lines[i - 1].text.trim().replace(/^\$/, '');
      lines[i - 1].lineType = 'wrapped'; // consumed
      console.debug(`[DEBUG]      Assigned adjacent price "${l.price}" to ${l.lineType} line ${i} ("${l.text}") from line before`);
      continue;
    }

    // Fallback: check line AFTER
    if (i + 1 < lines.length && isPriceOnlyLine(lines[i + 1])) {
      l.price = lines[i + 1].text.trim().replace(/^\$/, '');
      lines[i + 1].lineType = 'wrapped'; // consumed
      console.debug(`[DEBUG]      Assigned adjacent price "${l.price}" to ${l.lineType} line ${i} ("${l.text}") from line after`);
    }
  }
}

// ── Split combined TAX / BAL lines ───────────────────────────────────────────

/**
 * Split combined TAX/BAL lines (e.g. "**** TAX .93 BAL 45.44"):
 * set the tax line's price and insert a synthetic TOTAL line.
 */
export function splitTaxBalLine(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    splitTaxBalLine: scanning for combined TAX/BAL lines...`);
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    if (l.lineType !== 'tax') continue;

    // Try patterns like "TAX .93 BAL 45.44" or "TAX 1.23 BAL 56.78"
    const m = l.text.match(/\btax\s+(\d*\.?\d+)\s+bal(?:ance)?\s+(\d+\.\d{2})\b/i);
    if (!m) continue;

    const taxVal = m[1];   // e.g. ".93" or "1.23"
    const balVal = m[2];   // e.g. "45.44"

    console.debug(`[DEBUG]      Line ${i}: splitting TAX=${taxVal}, BAL=${balVal} from "${l.text}"`);

    // Assign the tax amount to this line
    l.price = taxVal;

    // Insert a synthetic total line right after
    const syntheticTotal: ReceiptLine = {
      text: `BAL ${balVal}`,
      words: [],
      itemName: `BAL ${balVal}`,
      price: balVal,
      lineType: 'total',
      angle: l.angle,
    };
    lines.splice(i + 1, 0, syntheticTotal);
    // Skip the just-inserted line
    i++;
  }
}

// ── Demote post-total items ──────────────────────────────────────────────────

/**
 * Demote item lines that appear after the last keyword block (SUBTOTAL/TOTAL)
 * to 'info' or 'tender'. These are typically payment-method lines or orphan
 * amounts that got misclassified as items.
 */
export function demotePostTotalItems(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    demotePostTotalItems: checking for items after keyword block...`);

  // Find the LAST total or subtotal line with a price.
  let boundaryIdx = -1;
  for (let i = lines.length - 1; i >= 0; i--) {
    if (
      ['total', 'subtotal'].includes(lines[i].lineType) &&
      lines[i].price !== null &&
      lines[i].lineType !== 'wrapped'
    ) {
      boundaryIdx = i;
      break;
    }
  }

  if (boundaryIdx === -1) {
    console.debug(`[DEBUG]      No total/subtotal line with a price found, skipping`);
    return;
  }

  console.debug(`[DEBUG]      Boundary line: ${boundaryIdx} ("${lines[boundaryIdx].text}")`);

  for (let i = boundaryIdx + 1; i < lines.length; i++) {
    const l = lines[i];
    if (l.lineType === 'wrapped') continue;
    if (['untaxed_item', 'taxed_item'].includes(l.lineType)) {
      const reclassified = classifyLine(l.text, l.price);
      const newType = reclassified === 'tender' ? 'tender' : 'info';
      console.debug(`[DEBUG]      Demoting post-keyword item line ${i}: "${l.text}" (${l.price}) → ${newType}`);
      l.lineType = newType;
    }
  }
}

// ── Fix keyword price assignment ─────────────────────────────────────────────

/**
 * Correct mis-assigned keyword prices.
 *
 * When all three keyword lines (SUBTOTAL, TAX, TOTAL) have prices,
 * try the constraint SUBTOTAL + TAX = TOTAL to find the unique correct
 * assignment.  When only SUBTOTAL and TAX are present, tax should never
 * exceed subtotal so swap them if needed.
 */
export function fixKeywordPriceAssignment(lines: ReceiptLine[]): void {
  console.debug(`[DEBUG]    fixKeywordPriceAssignment: checking keyword price consistency...`);
  const sub = lines.find(l => l.lineType === 'subtotal' && l.price !== null);
  const tax = lines.find(l => l.lineType === 'tax' && l.price !== null);
  const tot = lines.find(l => l.lineType === 'total' && l.price !== null);

  // --- Case 1: all three present — constraint-based reassignment ----------
  if (sub && tax && tot) {
    const pairs: { str: string; val: number }[] = [
      { str: sub.price!, val: parsePrice(sub.price) },
      { str: tax.price!, val: parsePrice(tax.price) },
      { str: tot.price!, val: parsePrice(tot.price) },
    ];
    const sorted = [...pairs].sort((a, b) => a.val - b.val);

    // smallest + middle ≈ largest ?
    if (
      sorted[0].val >= 0 &&
      Math.abs(sorted[0].val + sorted[1].val - sorted[2].val) < 0.02
    ) {
      const idealTax = sorted[0].str;
      const idealSub = sorted[1].str;
      const idealTot = sorted[2].str;

      if (sub.price !== idealSub || tax.price !== idealTax || tot.price !== idealTot) {
        console.debug(
          `[DEBUG]      Reassigning: S=${sorted[1].val}, T=${sorted[0].val}, TOTAL=${sorted[2].val}`,
        );
        sub.price = idealSub;
        tax.price = idealTax;
        tot.price = idealTot;
      }
    }
    return;
  }

  // --- Case 2: only SUBTOTAL and TAX — simple swap if subtotal < tax ------
  if (sub && tax) {
    const subVal = parsePrice(sub.price);
    const taxVal = parsePrice(tax.price);
    if (subVal > 0 && taxVal > 0 && subVal < taxVal) {
      console.debug(`[DEBUG]      Swapping subtotal (${subVal}) ↔ tax (${taxVal})`);
      const tmp = sub.price;
      sub.price = tax.price;
      tax.price = tmp;
    }
  }
}

// ── Competing totals ─────────────────────────────────────────────────────────

/**
 * When multiple 'total' lines with prices exist, pick the best one using:
 * 1. If one satisfies SUBTOTAL + TAX ≈ TOTAL, prefer it.
 * 2. If all have the same price, keep the first.
 * 3. Otherwise, prefer the LAST total (usually the final charged amount).
 * Demoted totals become 'info'.
 */
export function resolveCompetingTotals(lines: ReceiptLine[]): void {
  const totalLines = lines.filter(
    (l) => l.lineType === 'total' && l.price !== null,
  );
  if (totalLines.length <= 1) return;

  console.debug(`[DEBUG]    resolveCompetingTotals: ${totalLines.length} total lines found`);

  const sub = lines.find((l) => l.lineType === 'subtotal' && l.price !== null);
  const tax = lines.find((l) => l.lineType === 'tax' && l.price !== null);

  let winner: ReceiptLine | null = null;

  // Strategy 1: prefer the total that satisfies S + T = TOTAL
  if (sub && tax) {
    const subVal = parsePrice(sub.price);
    const taxVal = parsePrice(tax.price);
    const expected = subVal + taxVal;
    for (const t of totalLines) {
      if (Math.abs(parsePrice(t.price) - expected) < 0.015) {
        winner = t;
        console.debug(`[DEBUG]      Winner (S+T match): "${t.text}" = ${t.price}`);
        break;
      }
    }
  }

  // Strategy 2: if all totals are the same value, keep the first
  if (!winner) {
    const vals = totalLines.map((t) => parsePrice(t.price));
    if (vals.every((v) => Math.abs(v - vals[0]) < 0.015)) {
      winner = totalLines[0];
      console.debug(`[DEBUG]      Winner (all same, keep first): "${winner.text}" = ${winner.price}`);
    }
  }

  // Strategy 3: prefer the last total (final charged amount)
  if (!winner) {
    winner = totalLines[totalLines.length - 1];
    console.debug(`[DEBUG]      Winner (last total): "${winner.text}" = ${winner.price}`);
  }

  // Demote all non-winners
  for (const t of totalLines) {
    if (t !== winner) {
      console.debug(`[DEBUG]      Demoting competing total: "${t.text}" (${t.price}) → info`);
      t.lineType = 'info';
    }
  }
}

// ── Remove null item-name items ──────────────────────────────────────────────

/**
 * Remove lines that have a null itemName (orphan fragments that couldn't
 * be assigned to any item).  Iterates backwards to safely splice.
 */
export function removeNullItemNameItems(lines: ReceiptLine[]): void {
  for (let i = lines.length - 1; i >= 0; i--) {
    if (lines[i].itemName === null) {
      lines.splice(i, 1);
    }
  }
}

// ── Wrapped item names ───────────────────────────────────────────────────────

/**
 * If a line has no price and is classified as "info", and the very next
 * line below it has a price with a small vertical gap and similar
 * left-alignment, then it's a continuation of the item name.  We prepend
 * the wrapped line's text to the next line's itemName and re-tag it as
 * "wrapped".  Iterates bottom-up so chained wraps cascade correctly.
 */
export function handleWrappedNames(lines: ReceiptLine[], medH: number): void {
  console.debug(`[DEBUG]    handleWrappedNames: scanning for info lines that wrap into priced lines...`);
  for (let i = lines.length - 2; i >= 0; i--) {
    const cur = lines[i];
    const nxt = lines[i + 1];

    if (cur.lineType !== 'info' || cur.price !== null) continue;
    if (nxt.price === null || nxt.lineType === 'wrapped') continue;

    const curBoxes = cur.words;
    const nxtBoxes = nxt.words;
    if (curBoxes.length === 0 || nxtBoxes.length === 0) continue;

    // Vertical gap (original image coordinates)
    const curBottom = Math.max(...curBoxes.map((b) => b.bottom));
    const nxtTop = Math.min(...nxtBoxes.map((b) => b.top));
    const gap = nxtTop - curBottom;

    // Left-alignment check
    const curLeft = Math.min(...curBoxes.map((b) => b.left));
    const nxtLeft = Math.min(...nxtBoxes.map((b) => b.left));

    if (
      gap < WRAP_MAX_VERTICAL_GAP_FACTOR * medH &&
      Math.abs(curLeft - nxtLeft) < WRAP_MAX_LEFT_ALIGN_FACTOR * medH
    ) {
      // Guard: if merging would reclassify the target line as info or tender
      // (e.g. "TOTAL NUMBER OF…" wrapping into an orphan price), skip.
      const mergedText = [cur.text, nxt.text].filter(Boolean).join(' ');
      const mergedType = classifyLine(mergedText, nxt.price);
      if (mergedType === 'info' || mergedType === 'tender') {
        console.debug(`[DEBUG]      Skipped wrap: "${cur.text}" into "${nxt.text}" — merged text classifies as ${mergedType}`);
        continue;
      }

      console.debug(`[DEBUG]      Wrapped: "${cur.text}" → prepended to "${nxt.text}" (gap=${gap.toFixed(1)}, leftDiff=${Math.abs(curLeft - nxtLeft).toFixed(1)})`);
      nxt.itemName = [cur.text, nxt.itemName].filter(Boolean).join(' ');
      cur.lineType = 'wrapped';
    }
  }
}

// ── Orphan item prices ───────────────────────────────────────────────────────

/**
 * Merge orphan price-only lines into priceless item lines in the receipt body.
 * Handles tilted receipts where the price drifts too far from the item name
 * for the neighbor graph to connect them.
 */
export function mergeOrphanItemPrices(lines: ReceiptLine[], medH: number): void {
  console.debug(`[DEBUG]    mergeOrphanItemPrices: scanning for orphan item prices...`);

  // Find boundary: first keyword line marks the end of the item section
  let firstKeywordIdx = lines.length;
  for (let i = 0; i < lines.length; i++) {
    if (['subtotal', 'tax', 'total'].includes(lines[i].lineType)) {
      firstKeywordIdx = i;
      break;
    }
  }

  /** Check if a line is an orphan price (has price, no real item name, not a keyword). */
  function isOrphanPrice(l: ReceiptLine): boolean {
    return (
      l.price !== null &&
      isTrivialItemName(l.itemName) &&
      l.lineType !== 'wrapped' &&
      !['subtotal', 'tax', 'total'].includes(l.lineType) &&
      l.words.length > 0
    );
  }

  /** Check if a line is a priceless item (info with no price). */
  function isPricelessItem(l: ReceiptLine): boolean {
    return l.lineType === 'info' && l.price === null && l.words.length > 0;
  }

  /** Compute the angle-projected Y residual between an item line and an orphan price. */
  function computeResidual(item: ReceiptLine, orphan: ReceiptLine): number {
    const itemMeanX = item.words.reduce((s, w) => s + w.center.x, 0) / item.words.length;
    const itemMeanY = item.words.reduce((s, w) => s + w.center.y, 0) / item.words.length;
    const orphanMeanX = orphan.words.reduce((s, w) => s + w.center.x, 0) / orphan.words.length;
    const orphanMeanY = orphan.words.reduce((s, w) => s + w.center.y, 0) / orphan.words.length;
    const expectedY = itemMeanY + (orphanMeanX - itemMeanX) * Math.sin(item.angle);
    return Math.abs(orphanMeanY - expectedY);
  }

  // Process priceless items top-to-bottom
  for (let i = 0; i < firstKeywordIdx; i++) {
    const cur = lines[i];
    if (!isPricelessItem(cur)) continue;

    console.debug(`[DEBUG]      Checking item line ${i}: "${cur.text}" for orphan price...`);

    // Search UP first (up to ORPHAN_SEARCH_RADIUS lines)
    let bestCandidate: { idx: number; residual: number } | null = null;
    for (let j = i - 1; j >= Math.max(0, i - ORPHAN_SEARCH_RADIUS); j--) {
      if (j >= firstKeywordIdx) continue;
      if (!isOrphanPrice(lines[j])) continue;
      const residual = computeResidual(cur, lines[j]);
      if (residual <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
        if (bestCandidate === null || residual < bestCandidate.residual) {
          bestCandidate = { idx: j, residual };
        }
        console.debug(`[DEBUG]        ABOVE candidate line ${j}: "${lines[j].price}" residual=${residual.toFixed(1)}`);
      }
    }

    // If nothing found above, search DOWN
    if (bestCandidate === null) {
      for (let j = i + 1; j <= Math.min(lines.length - 1, i + ORPHAN_SEARCH_RADIUS); j++) {
        if (j >= firstKeywordIdx) continue;
        if (!isOrphanPrice(lines[j])) continue;
        const residual = computeResidual(cur, lines[j]);
        if (residual <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          // Take the first one found below (closest by line order)
          bestCandidate = { idx: j, residual };
          console.debug(`[DEBUG]        BELOW candidate line ${j}: "${lines[j].price}" residual=${residual.toFixed(1)}`);
          break;
        }
      }
    }

    if (bestCandidate !== null) {
      const orphan = lines[bestCandidate.idx];

      // Guard: if the merged text would reclassify as 'info' or 'tender', skip.
      const mergedText = cur.text + ' ' + orphan.text;
      const mergedType = classifyLine(mergedText, orphan.price);
      if (mergedType === 'info' || mergedType === 'tender') {
        console.debug(`[DEBUG]      Skipped merge: "${cur.text}" + "${orphan.price}" — merged text classifies as ${mergedType}`);
        continue;
      }

      console.debug(`[DEBUG]      Merged: "${cur.text}" + "${orphan.price}" (residual=${bestCandidate.residual.toFixed(1)})`);

      cur.price = orphan.price;
      cur.itemName = cur.text;
      cur.text = cur.text + ' ' + orphan.text;
      cur.words = [...cur.words, ...orphan.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      orphan.lineType = 'wrapped';
    }
  }
}

// ── Orphan keyword prices ────────────────────────────────────────────────────

/**
 * Merge orphan price-only lines into priceless keyword lines (SUBTOTAL,
 * TAX, TOTAL).  Handles distorted receipts where keyword and price end up
 * on separate tentative lines.
 */
export function mergeOrphanPrices(lines: ReceiptLine[], medH: number): void {
  console.debug(`[DEBUG]    mergeOrphanPrices: scanning ${lines.length} lines for keyword lines without prices...`);

  /** Check if a line is an orphan price (has price, no real item name, not already used). */
  function isOrphanPrice(l: ReceiptLine): boolean {
    return (
      l.price !== null &&
      l.lineType !== 'wrapped' &&
      isTrivialItemName(l.itemName) &&
      l.words.length > 0
    );
  }

  // Process keyword lines top-to-bottom
  for (let i = 0; i < lines.length; i++) {
    const cur = lines[i];

    // Only apply to keyword lines that have no price
    if (!['subtotal', 'tax', 'total'].includes(cur.lineType) || cur.price !== null) continue;
    if (cur.words.length === 0) continue;

    console.debug(`[DEBUG]      Checking keyword line ${i}: "${cur.text}" for orphan price...`);

    const curTop = Math.min(...cur.words.map((w) => w.top));
    const curBottom = Math.max(...cur.words.map((w) => w.bottom));

    // ── Search UP first (up to ORPHAN_SEARCH_RADIUS lines) ──────────────
    let bestAbove: { idx: number; gap: number } | null = null;
    for (let j = i - 1; j >= Math.max(0, i - ORPHAN_SEARCH_RADIUS); j--) {
      const candidate = lines[j];
      if (!isOrphanPrice(candidate)) continue;

      const candBottom = Math.max(...candidate.words.map((w) => w.bottom));
      const gap = curTop - candBottom;
      if (gap <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
        // Pick the closest above candidate
        if (bestAbove === null || gap < bestAbove.gap) {
          bestAbove = { idx: j, gap };
        }
        console.debug(`[DEBUG]        ABOVE candidate line ${j}: "${candidate.price}" gap=${gap.toFixed(1)}`);
      }
    }

    // ── If nothing above, search DOWN (up to ORPHAN_SEARCH_RADIUS) ──────
    let bestBelow: { idx: number; gap: number } | null = null;
    if (bestAbove === null) {
      for (let j = i + 1; j <= Math.min(lines.length - 1, i + ORPHAN_SEARCH_RADIUS); j++) {
        const candidate = lines[j];
        if (!isOrphanPrice(candidate)) continue;

        const candTop = Math.min(...candidate.words.map((w) => w.top));
        const gap = candTop - curBottom;
        if (gap <= WRAP_MAX_VERTICAL_GAP_FACTOR * medH) {
          // Take the first valid one below (closest by line order)
          bestBelow = { idx: j, gap };
          console.debug(`[DEBUG]        BELOW candidate line ${j}: "${candidate.price}" gap=${gap.toFixed(1)}`);
          break;
        }
      }
    }

    // ── Merge the best candidate ────────────────────────────────────────
    if (bestAbove !== null) {
      const prev = lines[bestAbove.idx];
      console.debug(`[DEBUG]      Orphan merge: "${cur.text}" ← ABOVE price "${prev.price}" (gap=${bestAbove.gap.toFixed(1)})`);
      cur.price = prev.price;
      cur.text = prev.text + ' ' + cur.text;
      cur.words = [...prev.words, ...cur.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      prev.lineType = 'wrapped';
    } else if (bestBelow !== null) {
      const nxt = lines[bestBelow.idx];
      console.debug(`[DEBUG]      Orphan merge: "${cur.text}" ← BELOW price "${nxt.price}" (gap=${bestBelow.gap.toFixed(1)})`);
      cur.price = nxt.price;
      cur.text = cur.text + ' ' + nxt.text;
      cur.words = [...cur.words, ...nxt.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      nxt.lineType = 'wrapped';
    } else {
      console.debug(`[DEBUG]      No orphan price found for "${cur.text}"`);
    }
  }

  // ── Last-resort pass: keyword lines still without prices ──────────────
  // On heavily-tilted receipts, the price column can drift so far that
  // orphan prices end up well beyond ORPHAN_SEARCH_RADIUS.  Do a second
  // pass with no radius/gap limit, matching by nearest Y center.
  for (let i = 0; i < lines.length; i++) {
    const cur = lines[i];
    if (!['subtotal', 'tax', 'total'].includes(cur.lineType) || cur.price !== null) continue;
    if (cur.words.length === 0) continue;

    const curY = cur.words.reduce((s, w) => s + w.center.y, 0) / cur.words.length;
    let bestIdx = -1;
    let bestDist = Infinity;

    for (let j = 0; j < lines.length; j++) {
      if (!isOrphanPrice(lines[j])) continue;
      const candY = lines[j].words.reduce((s, w) => s + w.center.y, 0) / lines[j].words.length;
      const dist = Math.abs(candY - curY);
      if (dist < bestDist) {
        bestDist = dist;
        bestIdx = j;
      }
    }

    if (bestIdx >= 0) {
      const orphan = lines[bestIdx];
      console.debug(`[DEBUG]      Last-resort merge: "${cur.text}" ← orphan "${orphan.price}" (Ydist=${bestDist.toFixed(1)})`);
      cur.price = orphan.price;
      cur.text = cur.text + ' ' + orphan.text;
      cur.words = [...cur.words, ...orphan.words];
      cur.lineType = classifyLine(cur.text, cur.price);
      orphan.lineType = 'wrapped';
    }
  }
}

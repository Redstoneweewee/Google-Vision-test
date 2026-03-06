/**
 * run-tests-llama.ts — Batch test runner for the Llama receipt OCR pipeline.
 *
 * Identical to run-tests.ts but uses detect-llama.ts (Llama 3.2 pipeline)
 * instead of detect.ts (clustering algorithm only).
 *
 * Usage:
 *   npx ts-node src/run-tests-llama.ts              # defaults to ./tests
 *   npx ts-node src/run-tests-llama.ts ./my-tests   # custom folder
 */

import 'dotenv/config';

import fs from 'fs';
import path from 'path';

import { colorBold, colorPass, colorWarn, colorError, colorDim, colorCyan } from './colors';

// ── Image discovery ──────────────────────────────────────────────────────────

const IMAGE_EXTS = new Set(['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']);

/** Recursively find all image files under `dir`, skipping annotated copies. */
function discoverImages(dir: string): string[] {
  const results: string[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      results.push(...discoverImages(full));
    } else if (
      IMAGE_EXTS.has(path.extname(entry.name).toLowerCase()) &&
      !entry.name.includes('_annotated')
    ) {
      results.push(full);
    }
  }
  return results.sort();
}

// ── Strip ANSI escape codes (for the plain-text output file) ─────────────────

function stripAnsi(s: string): string {
  // eslint-disable-next-line no-control-regex
  return s.replace(/\x1b\[[0-9;]*m/g, '');
}

// ── Main ─────────────────────────────────────────────────────────────────────

interface TestResult {
  file: string;
  store: string;
  score: number;
  output: string;
  error?: string;
}

(async () => {
  const { execFile } = await import('child_process');
  const { promisify } = await import('util');
  const execFileAsync = promisify(execFile);

  const rootDir = process.argv[2]
    ? path.resolve(process.argv[2])
    : path.resolve(__dirname, '..', 'tests');

  if (!fs.existsSync(rootDir)) {
    console.error(`Test directory not found: ${rootDir}`);
    process.exit(1);
  }

  const images = discoverImages(rootDir);
  if (images.length === 0) {
    console.error(`No images found in ${rootDir}`);
    process.exit(1);
  }

  const projectRoot = path.resolve(__dirname, '..');
  const detectScript = path.join(__dirname, 'detect-llama.ts');

  console.log(colorBold(`\nRunning ${images.length} receipt test(s) from ${rootDir} [LLAMA PIPELINE]\n`));

  const results: TestResult[] = [];
  const totalStart = Date.now();

  for (let i = 0; i < images.length; i++) {
    const absPath = images[i];
    const relPath = path.relative(projectRoot, absPath).replace(/\\/g, '/');
    const store = path.basename(path.dirname(absPath));

    const label = `[${i + 1}/${images.length}]`.padEnd(8);
    process.stdout.write(`${colorDim(label)} ${relPath} … `);

    const start = Date.now();
    let result: TestResult;

    try {
      const { stdout, stderr } = await execFileAsync(
        'npx',
        ['ts-node', `"${detectScript}"`, `"${absPath}"`],
        {
          cwd: projectRoot,
          timeout: 180_000, // longer timeout for Llama
          maxBuffer: 10 * 1024 * 1024,
          shell: true,
          env: { ...process.env, FORCE_COLOR: '1' },
        },
      );

      const combined = stdout + (stderr ? `\n--- stderr ---\n${stderr}` : '');
      const clean = stripAnsi(combined);
      const m = clean.match(/Score:\s*(\d+)%/);
      const score = m ? parseInt(m[1], 10) : -1;

      result = { file: relPath, store, score, output: combined };
    } catch (err: any) {
      const msg = err.stderr || err.stdout || err.message || String(err);
      result = { file: relPath, store, score: -1, output: msg, error: msg };
    }

    const elapsed = ((Date.now() - start) / 1000).toFixed(1);

    const scoreStr =
      result.score === -1
        ? colorError('ERR')
        : result.score === 100
          ? colorPass(`${result.score}%`)
          : result.score >= 85
            ? colorWarn(`${result.score}%`)
            : colorError(`${result.score}%`);
    console.log(`${scoreStr}  ${colorDim(`(${elapsed}s)`)}`);

    results.push(result);
  }

  const totalElapsed = ((Date.now() - totalStart) / 1000).toFixed(1);

  // ── Summary table ──────────────────────────────────────────────────────

  console.log(colorBold(`\n${'═'.repeat(70)}`));
  console.log(colorBold('  RECEIPT OCR TEST RESULTS — LLAMA PIPELINE'));
  console.log(colorBold(`${'═'.repeat(70)}\n`));

  const stores = [...new Set(results.map((r) => r.store))];

  const COL_FILE = 45;
  const COL_SCORE = 8;
  const headerLine = `  ${'File'.padEnd(COL_FILE)} ${'Score'.padStart(COL_SCORE)}`;
  console.log(colorDim(headerLine));
  console.log(colorDim(`  ${'─'.repeat(COL_FILE)} ${'─'.repeat(COL_SCORE)}`));

  let totalPassed = 0;
  const totalTests = results.length;

  for (const store of stores) {
    const storeResults = results.filter((r) => r.store === store);
    console.log(`\n  ${colorBold(store.toUpperCase())}`);
    for (const r of storeResults) {
      const name = r.file.padEnd(COL_FILE);
      let scoreCell: string;
      if (r.score === -1) {
        scoreCell = colorError('  ERR'.padStart(COL_SCORE));
      } else if (r.score === 100) {
        scoreCell = colorPass(`${r.score}%`.padStart(COL_SCORE));
        totalPassed++;
      } else if (r.score >= 85) {
        scoreCell = colorWarn(`${r.score}%`.padStart(COL_SCORE));
      } else {
        scoreCell = colorError(`${r.score}%`.padStart(COL_SCORE));
      }
      console.log(`  ${name} ${scoreCell}`);
    }
  }

  console.log(colorDim(`\n  ${'─'.repeat(COL_FILE + COL_SCORE + 1)}`));

  const pctPassed = totalTests > 0 ? ((totalPassed / totalTests) * 100).toFixed(0) : '0';
  const summaryColor = totalPassed === totalTests ? colorPass : totalPassed > totalTests * 0.8 ? colorWarn : colorError;
  console.log(
    `  ${colorBold('Total:')} ${summaryColor(`${totalPassed}/${totalTests} at 100%`)}  (${pctPassed}%)   ${colorDim(`${totalElapsed}s`)}`,
  );
  console.log();

  // ── Write full output file ─────────────────────────────────────────────

  const outPath = path.join(projectRoot, `test_results_llama.txt`);
  const lines: string[] = [];

  lines.push('='.repeat(80));
  lines.push(`  RECEIPT OCR TEST RESULTS — LLAMA PIPELINE — ${new Date().toLocaleString()}`);
  lines.push('='.repeat(80));
  lines.push('');

  lines.push('SCORE SUMMARY');
  lines.push('-'.repeat(60));
  lines.push(`  ${'File'.padEnd(COL_FILE)} ${'Score'.padStart(COL_SCORE)}`);
  lines.push(`  ${'─'.repeat(COL_FILE)} ${'─'.repeat(COL_SCORE)}`);
  for (const store of stores) {
    const storeResults = results.filter((r) => r.store === store);
    lines.push('');
    lines.push(`  ${store.toUpperCase()}`);
    for (const r of storeResults) {
      const scoreText = r.score === -1 ? 'ERR' : `${r.score}%`;
      lines.push(`  ${r.file.padEnd(COL_FILE)} ${scoreText.padStart(COL_SCORE)}`);
    }
  }
  lines.push('');
  lines.push(`  Total: ${totalPassed}/${totalTests} at 100%  (${pctPassed}%)`);
  lines.push(`  Elapsed: ${totalElapsed}s`);
  lines.push('');
  lines.push('');

  lines.push('='.repeat(80));
  lines.push('  FULL OUTPUTS');
  lines.push('='.repeat(80));

  for (const r of results) {
    lines.push('');
    lines.push('─'.repeat(80));
    lines.push(`  ${r.file}  —  Score: ${r.score === -1 ? 'ERR' : r.score + '%'}`);
    lines.push('─'.repeat(80));
    lines.push('');
    lines.push(stripAnsi(r.output).trimEnd());
    lines.push('');
  }

  fs.writeFileSync(outPath, lines.join('\n'), 'utf-8');
  console.log(`Full output written to ${colorCyan(path.relative(projectRoot, outPath))}\n`);
})().catch((err) => {
  console.error('Test runner error:', err);
  process.exit(1);
});

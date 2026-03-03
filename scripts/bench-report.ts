#!/usr/bin/env npx tsx
/**
 * bench-report.ts — Unified brepkit vs OCCT comparison report generator.
 *
 * Reads:
 *   - Criterion JSON from target/criterion/<name>/new/estimates.json (nanoseconds)
 *   - JS benchmark JSON extracted from vitest output (milliseconds)
 *
 * Produces:
 *   - bench-results/report.md    — human-readable comparison table
 *   - bench-results/comparison.json — machine-readable data
 *
 * Usage:
 *   npx tsx scripts/bench-report.ts \
 *     --criterion-dir target/criterion \
 *     --js-json bench-results/js-bench.json \
 *     --output-dir bench-results
 */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'fs';
import { join } from 'path';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CriterionEstimates {
  median: { point_estimate: number }; // nanoseconds
  mean: { point_estimate: number };
}

interface JsBenchResult {
  name: string;
  kernel?: string;
  min: number;    // milliseconds
  median: number; // milliseconds
  mean: number;
  max: number;
  iterations: number;
}

interface ComparisonRow {
  display: string;
  occtMs: number | null;
  brepkitWasmMs: number | null;
  nativeUs: number | null;
  wasmOverOcct: number | null;    // ratio: brepkit-wasm / occt
  wasmOverhead: number | null;    // ratio: wasm / native
  winner: string;
}

// ---------------------------------------------------------------------------
// Name mapping: Criterion bench name → JS benchBoth name → display name
// ---------------------------------------------------------------------------

interface NameMapping {
  criterion: string | null;      // Criterion bench function name (null = no native bench)
  js: string;                    // JS benchBoth name (without [kernel] prefix)
  display: string;               // Human-readable display name
}

const NAME_MAP: NameMapping[] = [
  { criterion: 'makeBox(10,20,30) x100',  js: 'makeBox(10,20,30)',        display: 'makeBox (×100)' },
  { criterion: 'makeCylinder(5,20) x100',  js: 'makeCylinder(5,20)',       display: 'makeCylinder (×100)' },
  { criterion: 'makeSphere(10) x100',      js: 'makeSphere(10)',           display: 'makeSphere (×100)' },
  { criterion: 'fuse(box,box) x10',        js: 'fuse(box,box)',            display: 'fuse box∪box (×10)' },
  { criterion: 'cut(box,cyl) x10',         js: 'cut(box,cyl)',             display: 'cut box−cyl (×10)' },
  { criterion: 'intersect(box,sphere) x10', js: 'intersect(box,sphere)',   display: 'intersect box∩sphere (×10)' },
  { criterion: 'translate x1000',           js: 'translate ×1000',          display: 'translate (×1000)' },
  { criterion: 'rotate x100',              js: 'rotate ×100',              display: 'rotate (×100)' },
  { criterion: 'mesh box (tol=0.1)',        js: 'mesh box (tol=0.1)',       display: 'mesh box coarse' },
  { criterion: 'mesh sphere (tol=0.01)',    js: 'mesh sphere (tol=0.01)',   display: 'mesh sphere fine' },
  { criterion: 'volume x100',              js: 'volume ×100',              display: 'volume (×100)' },
  { criterion: 'boundingBox x100',          js: 'boundingBox ×100',         display: 'boundingBox (×100)' },
  { criterion: null,                        js: 'exportSTEP ×10',           display: 'exportSTEP (×10)' },
  { criterion: 'box+chamfer',              js: 'box+chamfer',              display: 'box + chamfer' },
  { criterion: 'box+fillet',               js: 'box+fillet',               display: 'box + fillet' },
  { criterion: 'multi-boolean model',       js: 'multi-boolean model',      display: 'multi-boolean (16 holes)' },
];

// ---------------------------------------------------------------------------
// CLI argument parsing
// ---------------------------------------------------------------------------

function parseArgs(): { criterionDir: string; jsJson: string; outputDir: string } {
  const args = process.argv.slice(2);
  let criterionDir = '';
  let jsJson = '';
  let outputDir = '';

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--criterion-dir' && args[i + 1]) criterionDir = args[++i]!;
    else if (args[i] === '--js-json' && args[i + 1]) jsJson = args[++i]!;
    else if (args[i] === '--output-dir' && args[i + 1]) outputDir = args[++i]!;
  }

  if (!criterionDir || !jsJson || !outputDir) {
    console.error('Usage: npx tsx bench-report.ts --criterion-dir <dir> --js-json <file> --output-dir <dir>');
    process.exit(1);
  }

  return { criterionDir, jsJson, outputDir };
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/** Read Criterion estimates for a given bench name. Returns median in nanoseconds. */
function readCriterionNs(criterionDir: string, benchName: string): number | null {
  // Criterion uses the bench name as directory name
  const estimatesPath = join(criterionDir, benchName, 'new', 'estimates.json');
  if (!existsSync(estimatesPath)) return null;

  try {
    const data: CriterionEstimates = JSON.parse(readFileSync(estimatesPath, 'utf-8'));
    return data.median.point_estimate;
  } catch {
    return null;
  }
}

/** Read all JS benchmark results, indexed by (kernel, name). */
function readJsResults(jsJsonPath: string): Map<string, JsBenchResult> {
  if (!existsSync(jsJsonPath)) return new Map();

  try {
    const results: JsBenchResult[] = JSON.parse(readFileSync(jsJsonPath, 'utf-8'));
    const map = new Map<string, JsBenchResult>();

    for (const r of results) {
      // Key format: "kernel:name"
      // The name in the JSON has the [kernel] prefix from benchKernel
      // e.g. "[occt] makeBox(10,20,30)" → kernel='occt', raw name='makeBox(10,20,30)'
      const kernel = r.kernel ?? extractKernelFromName(r.name);
      const rawName = r.name.replace(/^\[(occt|brepkit)\]\s*/, '');
      map.set(`${kernel}:${rawName}`, r);
    }

    return map;
  } catch {
    return new Map();
  }
}

/** Fallback: extract kernel from "[kernel] name" format. */
function extractKernelFromName(name: string): string {
  const match = name.match(/^\[(occt|brepkit)\]/);
  return match ? match[1]! : 'unknown';
}

// ---------------------------------------------------------------------------
// Report generation
// ---------------------------------------------------------------------------

function fmt(val: number | null, decimals: number): string {
  if (val === null) return '—';
  return val.toFixed(decimals);
}

function fmtRatio(val: number | null): string {
  if (val === null) return '—';
  return val < 0.01 ? '<0.01×' : `${val.toFixed(2)}×`;
}

function generateReport(
  criterionDir: string,
  jsResults: Map<string, JsBenchResult>
): { rows: ComparisonRow[]; markdown: string; json: object } {
  const rows: ComparisonRow[] = [];

  for (const mapping of NAME_MAP) {
    // Native (Criterion) — nanoseconds → microseconds for display
    const nativeNs = mapping.criterion ? readCriterionNs(criterionDir, mapping.criterion) : null;
    const nativeUs = nativeNs !== null ? nativeNs / 1000 : null;

    // JS OCCT result — milliseconds
    const occtResult = jsResults.get(`occt:${mapping.js}`);
    const occtMs = occtResult?.median ?? null;

    // JS brepkit-WASM result — milliseconds
    const brepkitResult = jsResults.get(`brepkit:${mapping.js}`);
    const brepkitWasmMs = brepkitResult?.median ?? null;

    // Ratios
    const wasmOverOcct =
      brepkitWasmMs !== null && occtMs !== null && occtMs > 0
        ? brepkitWasmMs / occtMs
        : null;

    const nativeMs = nativeUs !== null ? nativeUs / 1000 : null;
    const wasmOverhead =
      brepkitWasmMs !== null && nativeMs !== null && nativeMs > 0
        ? brepkitWasmMs / nativeMs
        : null;

    // Winner determination
    let winner = '—';
    if (brepkitWasmMs !== null && occtMs !== null) {
      if (brepkitWasmMs < occtMs * 0.95) winner = 'brepkit';
      else if (occtMs < brepkitWasmMs * 0.95) winner = 'OCCT';
      else winner = 'tie';
    }

    rows.push({
      display: mapping.display,
      occtMs,
      brepkitWasmMs,
      nativeUs,
      wasmOverOcct,
      wasmOverhead,
      winner,
    });
  }

  // Build markdown
  const lines: string[] = [];
  lines.push('# brepkit vs OCCT — Benchmark Comparison');
  lines.push('');
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push('');
  lines.push('## Results');
  lines.push('');
  lines.push(
    '| Operation | OCCT (ms) | brepkit-WASM (ms) | Native (µs) | WASM/OCCT | WASM overhead | Winner |'
  );
  lines.push(
    '|-----------|----------:|------------------:|------------:|----------:|--------------:|--------|'
  );

  for (const r of rows) {
    lines.push(
      `| ${r.display.padEnd(25)} | ${fmt(r.occtMs, 2).padStart(9)} | ${fmt(r.brepkitWasmMs, 2).padStart(17)} | ${fmt(r.nativeUs, 1).padStart(11)} | ${fmtRatio(r.wasmOverOcct).padStart(9)} | ${fmtRatio(r.wasmOverhead).padStart(13)} | ${r.winner.padEnd(7)}|`
    );
  }

  // Bottleneck analysis
  const bottlenecks = rows.filter(
    (r) => r.wasmOverOcct !== null && r.wasmOverOcct > 1.0
  );
  if (bottlenecks.length > 0) {
    lines.push('');
    lines.push('## Bottleneck Analysis');
    lines.push('');
    lines.push('Operations where brepkit-WASM is slower than OCCT:');
    lines.push('');
    for (const b of bottlenecks.sort((a, b) => (b.wasmOverOcct ?? 0) - (a.wasmOverOcct ?? 0))) {
      lines.push(
        `- **${b.display}**: ${fmtRatio(b.wasmOverOcct)} slower (${fmt(b.brepkitWasmMs, 2)} ms vs ${fmt(b.occtMs, 2)} ms)`
      );
    }
  }

  // WASM overhead analysis
  const overheadRows = rows.filter(
    (r) => r.wasmOverhead !== null
  );
  if (overheadRows.length > 0) {
    lines.push('');
    lines.push('## WASM Overhead (WASM / Native)');
    lines.push('');
    lines.push('FFI + serialization cost per operation:');
    lines.push('');
    for (const o of overheadRows.sort((a, b) => (b.wasmOverhead ?? 0) - (a.wasmOverhead ?? 0))) {
      lines.push(
        `- **${o.display}**: ${fmtRatio(o.wasmOverhead)} (${fmt(o.brepkitWasmMs, 2)} ms WASM, ${fmt(o.nativeUs, 1)} µs native)`
      );
    }
  }

  // Wins summary
  const brepkitWins = rows.filter((r) => r.winner === 'brepkit').length;
  const occtWins = rows.filter((r) => r.winner === 'OCCT').length;
  const ties = rows.filter((r) => r.winner === 'tie').length;
  const noData = rows.filter((r) => r.winner === '—').length;

  lines.push('');
  lines.push('## Summary');
  lines.push('');
  lines.push(`- **brepkit wins**: ${brepkitWins}`);
  lines.push(`- **OCCT wins**: ${occtWins}`);
  lines.push(`- **Ties** (within 5%): ${ties}`);
  if (noData > 0) lines.push(`- **No data**: ${noData}`);

  const markdown = lines.join('\n') + '\n';

  // JSON output
  const json = {
    generated: new Date().toISOString(),
    rows: rows.map((r) => ({
      operation: r.display,
      occt_ms: r.occtMs,
      brepkit_wasm_ms: r.brepkitWasmMs,
      native_us: r.nativeUs,
      wasm_over_occt: r.wasmOverOcct,
      wasm_overhead: r.wasmOverhead,
      winner: r.winner,
    })),
    summary: { brepkit_wins: brepkitWins, occt_wins: occtWins, ties, no_data: noData },
  };

  return { rows, markdown, json };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const { criterionDir, jsJson, outputDir } = parseArgs();

console.log(`Reading Criterion data from: ${criterionDir}`);
console.log(`Reading JS benchmark data from: ${jsJson}`);

const jsResults = readJsResults(jsJson);
console.log(`  Found ${jsResults.size} JS benchmark results`);

// Count available criterion benchmarks
let criterionCount = 0;
for (const m of NAME_MAP) {
  if (m.criterion && readCriterionNs(criterionDir, m.criterion) !== null) {
    criterionCount++;
  }
}
console.log(`  Found ${criterionCount} Criterion benchmark results`);

const { markdown, json } = generateReport(criterionDir, jsResults);

if (!existsSync(outputDir)) mkdirSync(outputDir, { recursive: true });

writeFileSync(join(outputDir, 'report.md'), markdown);
writeFileSync(join(outputDir, 'comparison.json'), JSON.stringify(json, null, 2) + '\n');

console.log('');
console.log('Report written to:');
console.log(`  ${join(outputDir, 'report.md')}`);
console.log(`  ${join(outputDir, 'comparison.json')}`);

// Print the report to stdout too
console.log('');
console.log(markdown);

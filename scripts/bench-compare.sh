#!/usr/bin/env bash
# bench-compare.sh — Unified brepkit vs OCCT benchmark comparison.
#
# Runs native Criterion benchmarks, builds brepkit WASM, installs into brepjs,
# runs the JS kernel-comparison suite, then produces a unified comparison report.
#
# Usage:
#   ./scripts/bench-compare.sh <path-to-brepjs>
#
# Example:
#   ./scripts/bench-compare.sh ~/Git/brepjs

set -euo pipefail

BREPKIT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="$BREPKIT_ROOT/bench-results"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

BREPJS_DIR="${1:-}"
if [[ -z "$BREPJS_DIR" ]]; then
    echo "Usage: $0 <path-to-brepjs>"
    echo "  e.g. $0 ~/Git/brepjs"
    exit 1
fi
BREPJS_DIR="$(cd "$BREPJS_DIR" && pwd)"

if [[ ! -f "$BREPJS_DIR/package.json" ]]; then
    echo "Error: $BREPJS_DIR does not contain a package.json"
    exit 1
fi

echo "=== brepkit vs OCCT Benchmark Comparison ==="
echo "  brepkit:  $BREPKIT_ROOT"
echo "  brepjs:   $BREPJS_DIR"
echo ""

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Step 1: Run Criterion benchmarks (native Rust)
# ---------------------------------------------------------------------------

echo "[1/5] Running Criterion benchmarks (native)..."
if cargo bench -p brepkit-operations 2>&1 | tee "$RESULTS_DIR/criterion.log"; then
    echo "  Criterion benchmarks complete."
else
    echo "  Warning: Criterion benchmarks failed. Native results will be missing."
fi

# ---------------------------------------------------------------------------
# Step 2: Build brepkit WASM via wasm-pack
# ---------------------------------------------------------------------------

echo ""
echo "[2/5] Building brepkit WASM (--target nodejs --release)..."

# Check wasm-pack is available
if ! command -v wasm-pack &>/dev/null; then
    echo "Error: wasm-pack not found. Install with: cargo install wasm-pack"
    exit 1
fi

# Fedora libbz2 workaround: wasm-pack binary needs libbz2.so.1.0
if [[ -f /etc/fedora-release ]] && ! ldconfig -p 2>/dev/null | grep -q 'libbz2.so.1.0 '; then
    if [[ -f /usr/lib64/libbz2.so.1.0.8 ]] && [[ ! -f /usr/lib64/libbz2.so.1.0 ]]; then
        echo ""
        echo "Warning: Fedora libbz2 compatibility issue detected."
        echo "wasm-pack needs libbz2.so.1.0 but only libbz2.so.1.0.8 exists."
        echo ""
        echo "Fix with one of:"
        echo "  sudo ln -s /usr/lib64/libbz2.so.1.0.8 /usr/lib64/libbz2.so.1.0"
        echo "  cargo install wasm-pack --force  (builds from source, avoids the issue)"
        echo ""
    fi
fi

WASM_PKG="$BREPKIT_ROOT/crates/wasm/pkg"

(
    cd "$BREPKIT_ROOT"
    wasm-pack build crates/wasm --target nodejs --release --out-dir "$WASM_PKG"
) 2>&1 | tee -a "$RESULTS_DIR/criterion.log"

echo "  WASM package built at: $WASM_PKG"

# ---------------------------------------------------------------------------
# Step 3: Install WASM package into brepjs
# ---------------------------------------------------------------------------

echo ""
echo "[3/5] Installing brepkit-wasm into brepjs..."
(
    cd "$BREPJS_DIR"
    npm install "$WASM_PKG" --no-save 2>&1
)
echo "  Installed."

# ---------------------------------------------------------------------------
# Step 4: Run JS kernel-comparison benchmarks
# ---------------------------------------------------------------------------

echo ""
echo "[4/5] Running JS kernel-comparison benchmarks..."

JS_LOG="$RESULTS_DIR/js-bench.log"
JS_JSON="$RESULTS_DIR/js-bench.json"

(
    cd "$BREPJS_DIR"
    BENCH_OUTPUT_JSON=1 npx vitest run benchmarks/kernel-comparison.bench.test.ts \
        --config vitest.bench.config.ts --reporter=verbose 2>&1
) | tee "$JS_LOG"

# Extract JSON from sentinel markers
if grep -q '--- BENCHMARK RESULTS JSON ---' "$JS_LOG"; then
    sed -n '/--- BENCHMARK RESULTS JSON ---/,/--- END BENCHMARK RESULTS ---/p' "$JS_LOG" \
        | sed '1d;$d' > "$JS_JSON"
    echo "  JS benchmark JSON extracted to: $JS_JSON"
else
    echo "  Warning: No JSON output found in JS benchmark log."
    echo "[]" > "$JS_JSON"
fi

# ---------------------------------------------------------------------------
# Step 5: Generate comparison report
# ---------------------------------------------------------------------------

echo ""
echo "[5/5] Generating comparison report..."

npx tsx "$BREPKIT_ROOT/scripts/bench-report.ts" \
    --criterion-dir "$BREPKIT_ROOT/target/criterion" \
    --js-json "$JS_JSON" \
    --output-dir "$RESULTS_DIR"

echo ""
echo "=== Done ==="
echo "  Report:     $RESULTS_DIR/report.md"
echo "  JSON data:  $RESULTS_DIR/comparison.json"
echo "  Criterion:  $RESULTS_DIR/criterion.log"
echo "  JS log:     $RESULTS_DIR/js-bench.log"

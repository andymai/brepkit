#!/usr/bin/env bash
set -euo pipefail

# Build brepkit-wasm with SIMD optimizations for release.
RUSTFLAGS="-C target-feature=+simd128" \
cargo build -p brepkit-wasm \
    --target wasm32-unknown-unknown \
    --release

echo "Built: target/wasm32-unknown-unknown/release/brepkit_wasm.wasm"

# Run wasm-opt if available
if command -v wasm-opt &>/dev/null; then
    WASM_FILE="target/wasm32-unknown-unknown/release/brepkit_wasm.wasm"
    wasm-opt -O3 "$WASM_FILE" -o "$WASM_FILE.opt"
    mv "$WASM_FILE.opt" "$WASM_FILE"
    echo "Optimized with wasm-opt"
fi

ls -lh target/wasm32-unknown-unknown/release/brepkit_wasm.wasm

#!/usr/bin/env bash
set -euo pipefail

BREPKIT=~/Git/brepkit
GRIDFINITY=~/Git/gridfinity-layout-tool
PKG=$BREPKIT/crates/wasm/pkg
DEST=$GRIDFINITY/node_modules/brepkit-wasm

echo "═══ Building WASM (debug) ═══"
(cd "$BREPKIT" && cargo xtask wasm-build --skip-opt)

echo ""
echo "═══ Copying to gridfinity node_modules ═══"
cp "$PKG"/{brepkit_wasm_bg.js,brepkit_wasm_bg.wasm,brepkit_wasm.d.ts,brepkit_wasm.js,brepkit_wasm_node.cjs,package.json} "$DEST/"
echo "Copied $(wc -c < "$DEST/brepkit_wasm_bg.wasm") bytes WASM"

echo ""
echo "═══ Running topology parity tests ═══"
echo "(OCCT results cached in .occt-topology-cache.json — delete to regenerate)"
cd "$GRIDFINITY"
pnpm exec vitest run --config vitest.profile.config.ts topologyParity

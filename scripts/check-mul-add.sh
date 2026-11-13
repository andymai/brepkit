#!/usr/bin/env bash
set -euo pipefail

# Library code must not call f64::mul_add.
#
# mul_add rounds once, which takes a fused multiply-add instruction.
# WebAssembly has none, so each call goes to a software routine that costs
# several times a multiply and an add: it was more than half the running
# time of the wasm build. Library code calls brepkit_math::madd::Madd::madd.
#
# Integration tests, examples and benches can keep mul_add, and so can the
# one unit-test file whose fixtures build exactly tangent surfaces with it.

ALLOWED="crates/math/src/nurbs/intersection/tests.rs"

found=$(grep -rn --include='*.rs' -F '.mul_add(' crates/*/src | grep -v "^${ALLOWED}:" || true)
if [ -n "$found" ]; then
  echo "❌ f64::mul_add in library code. Use brepkit_math::madd::Madd::madd:"
  echo "$found"
  exit 1
fi
echo "✅ No mul_add in library code."

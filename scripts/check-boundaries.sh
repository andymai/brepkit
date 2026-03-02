#!/usr/bin/env bash
set -euo pipefail

# Verify crate dependency graph matches layer rules.
#
# Layer rules:
#   L0 (math)       — no workspace deps
#   L1 (topology)   — depends on math only
#   L2 (operations) — depends on math, topology
#   L2 (io)         — depends on math, topology
#   L3 (wasm)       — depends on all

FAIL=0

check_deps() {
  local crate="$1"
  shift
  local allowed=("$@")

  local cargo_toml="crates/${crate}/Cargo.toml"
  if [ ! -f "$cargo_toml" ]; then
    echo "SKIP: $cargo_toml not found"
    return
  fi

  # Extract only [dependencies] section lines (not [package] name)
  local deps_section
  deps_section=$(sed -n '/^\[dependencies\]/,/^\[/p' "$cargo_toml" 2>/dev/null || true)

  for dep in brepkit-math brepkit-topology brepkit-operations brepkit-io; do
    if echo "$deps_section" | grep -q "${dep}"; then
      local is_allowed=false
      for a in "${allowed[@]}"; do
        if [ "$dep" = "$a" ]; then
          is_allowed=true
          break
        fi
      done
      if [ "$is_allowed" = false ]; then
        echo "VIOLATION: crates/${crate} depends on ${dep} (not allowed)"
        FAIL=1
      fi
    fi
  done
}

echo "Checking crate boundary rules..."

check_deps "math"       ""
check_deps "topology"   "brepkit-math"
check_deps "operations" "brepkit-math" "brepkit-topology"
check_deps "io"         "brepkit-math" "brepkit-topology" "brepkit-operations"
check_deps "wasm"       "brepkit-math" "brepkit-topology" "brepkit-operations" "brepkit-io"

if [ $FAIL -ne 0 ]; then
  echo "❌ Boundary check failed."
  exit 1
fi

echo "✅ All crate boundaries valid."

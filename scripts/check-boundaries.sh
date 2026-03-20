#!/usr/bin/env bash
set -euo pipefail

# Verify crate dependency graph matches layer rules.
#
# Layer rules:
#   L0   (math)       — no workspace deps
#   L0.5 (geometry)   — depends on math only
#   L1   (topology)   — depends on math only
#   L1.5 (algo)       — depends on math, topology, geometry
#   L1.5 (blend)      — depends on math, topology, geometry
#   L1.5 (heal)       — depends on math, topology, geometry
#   L1.5 (check)      — depends on math, topology, geometry
#   L2   (operations) — depends on math, topology, algo, blend, heal, check, geometry
#   L2   (io)         — depends on math, topology, operations, heal
#   L3   (wasm)       — depends on all

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

  # Extract both [dependencies] and [dev-dependencies] sections
  local deps_section
  deps_section=$(sed -n '/^\[dependencies\]/,/^\[/p; /^\[dev-dependencies\]/,/^\[/p' "$cargo_toml" 2>/dev/null || true)

  for dep in brepkit-math brepkit-topology brepkit-algo brepkit-blend brepkit-heal brepkit-check brepkit-geometry brepkit-offset brepkit-operations brepkit-io; do
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

check_deps "math"
check_deps "topology"   "brepkit-math"
check_deps "geometry"   "brepkit-math"
check_deps "algo"       "brepkit-math" "brepkit-topology" "brepkit-geometry"
check_deps "blend"      "brepkit-math" "brepkit-topology" "brepkit-geometry"
check_deps "heal"       "brepkit-math" "brepkit-topology" "brepkit-geometry"
check_deps "check"      "brepkit-math" "brepkit-topology" "brepkit-geometry"
check_deps "offset"     "brepkit-math" "brepkit-topology" "brepkit-geometry" "brepkit-algo"
check_deps "operations" "brepkit-math" "brepkit-topology" "brepkit-algo" "brepkit-blend" "brepkit-heal" "brepkit-check" "brepkit-geometry" "brepkit-offset"
check_deps "io"         "brepkit-math" "brepkit-topology" "brepkit-operations" "brepkit-heal"
check_deps "wasm"       "brepkit-math" "brepkit-topology" "brepkit-algo" "brepkit-blend" "brepkit-heal" "brepkit-check" "brepkit-geometry" "brepkit-operations" "brepkit-io" "brepkit-offset"

if [ $FAIL -ne 0 ]; then
  echo "❌ Boundary check failed."
  exit 1
fi

echo "✅ All crate boundaries valid."

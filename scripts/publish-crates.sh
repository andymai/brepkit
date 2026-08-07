#!/usr/bin/env bash
set -euo pipefail

# Publish every publishable crate to crates.io, in dependency order, skipping
# any already at this version.
#
# Used two ways:
#   - `.github/workflows/publish.yml` runs it on each release, authenticating
#     via OIDC trusted publishing.
#   - A human runs it once per crate name to bootstrap. crates.io only accepts
#     a trusted-publisher config for a crate that already exists, so the FIRST
#     version of each crate must be pushed manually with an API token:
#       CARGO_REGISTRY_TOKEN=<token> ./scripts/publish-crates.sh
#
# Skipping already-published crates is what makes a re-run after a partial
# failure safe, and it is why this does not just call
# `cargo publish --workspace` (which aborts the batch on the first crate that
# already exists).
#
# Order is cargo's own topological order; `cargo package --workspace` prints
# it. brepkit-wasm-macros is `publish = false` and is deliberately absent.
CRATES=(
  brepkit-math
  brepkit-geometry
  brepkit-topology
  brepkit-check
  brepkit-algo
  brepkit-blend
  brepkit-heal
  brepkit-offset
  brepkit-sketch
  brepkit-operations
  brepkit-io
  brepkit-render
  brepkit-wasm
)

for tool in cargo jq curl; do
  command -v "$tool" >/dev/null || {
    echo "❌ $tool is required but not installed."
    exit 1
  }
done

VERSION=$(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "brepkit-math") | .version')

# An empty VERSION is not cosmetic: it would turn the existence check below
# into a request for the crate's base endpoint, which answers 200 for any
# published crate. Every crate would then "already exist", the loop would skip
# all of them, and the script would exit 0 having published nothing.
if [ -z "$VERSION" ]; then
  echo "❌ Could not read the workspace version from cargo metadata."
  exit 1
fi

echo "Publishing brepkit crates at $VERSION"

for crate in "${CRATES[@]}"; do
  if curl -sf -o /dev/null -H 'User-Agent: brepkit-release' \
       "https://crates.io/api/v1/crates/$crate/$VERSION"; then
    echo "  $crate@$VERSION already published, skipping"
    continue
  fi
  echo "  publishing $crate@$VERSION"
  # cargo blocks until the new version appears in the index, so the next
  # crate in the list can resolve it.
  cargo publish -p "$crate"
done

echo "✅ All crates at $VERSION."

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

# crates.io allows a burst of 5 NEW crate names then one per 10 minutes. New
# versions of existing crates have a far higher allowance, so this only bites
# when crates are added to the workspace. Retrying is bounded: a run that adds
# more names than this should be finished by re-running the script rather than
# holding a CI runner idle for an hour.
MAX_RETRIES=${PUBLISH_MAX_RETRIES:-3}
RETRY_WAIT=${PUBLISH_RETRY_WAIT:-630}

publish_one() {
  local crate="$1" attempt=0 out status
  while :; do
    # Capture rather than stream so the 429 can be recognised; the output is
    # echoed either way so nothing is hidden. The `|| status=$?` form is load
    # bearing: a bare assignment from a failing command substitution trips
    # `set -e` before the status can be inspected.
    status=0
    out=$(cargo publish -p "$crate" 2>&1) || status=$?
    echo "$out"
    [ "$status" -eq 0 ] && return 0

    if ! grep -q '429 Too Many Requests' <<<"$out"; then
      return $status
    fi
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_RETRIES" ]; then
      echo "❌ Still rate-limited after $MAX_RETRIES retries."
      echo "   Crates published so far are safe. Re-run this script to resume;"
      echo "   it skips everything already on crates.io."
      return 1
    fi
    echo "  rate-limited on $crate, waiting ${RETRY_WAIT}s (retry $attempt/$MAX_RETRIES)"
    sleep "$RETRY_WAIT"
  done
}

for crate in "${CRATES[@]}"; do
  if curl -sf -o /dev/null -H 'User-Agent: brepkit-release' \
       "https://crates.io/api/v1/crates/$crate/$VERSION"; then
    echo "  $crate@$VERSION already published, skipping"
    continue
  fi
  echo "  publishing $crate@$VERSION"
  # cargo blocks until the new version appears in the index, so the next
  # crate in the list can resolve it.
  publish_one "$crate"
done

echo "✅ All crates at $VERSION."

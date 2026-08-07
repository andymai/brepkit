#!/usr/bin/env bash
set -euo pipefail

# Verify every crate and every inter-crate dependency sits at one version.
#
# release-please bumps `[workspace.package] version` and each
# `[workspace.dependencies] brepkit-*.version` through separate jsonpath
# entries in release-please-config.json. If one of those entries is dropped or
# stops matching, the versions drift silently and `cargo publish` resolves an
# older crate from crates.io instead of the sibling in this workspace. This
# check makes that drift fail on the release PR rather than at publish time.

for tool in cargo jq; do
  command -v "$tool" >/dev/null || {
    echo "❌ $tool is required but not installed."
    exit 1
  }
done

WS_VERSION=$(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "brepkit-math") | .version')

if [ -z "$WS_VERSION" ]; then
  echo "❌ Could not read the workspace version from cargo metadata."
  exit 1
fi

FAIL=0

# Every workspace member must be at the shared version (they all use
# `version.workspace = true`, so a mismatch means one opted out).
while read -r name version; do
  if [ "$version" != "$WS_VERSION" ]; then
    echo "❌ crate $name is at $version, expected $WS_VERSION"
    FAIL=1
  fi
done < <(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | "\(.name) \(.version)"')

# Every brepkit-to-brepkit dependency must request the shared version.
while read -r line; do
  dep=${line%% *}
  req=${line#* }
  if [ "$req" != "^$WS_VERSION" ]; then
    echo "❌ workspace dependency $dep requests '$req', expected '^$WS_VERSION'"
    FAIL=1
  fi
done < <(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[].dependencies[]
           | select(.name | startswith("brepkit-"))
           | select(.kind != "dev")
           | "\(.name) \(.req)"' | sort -u)

# Every publishable member must be listed in the publish script. Without this,
# adding a crate and forgetting the script means it silently never reaches
# crates.io while everything else keeps releasing.
while read -r name; do
  if ! grep -qx "  $name" scripts/publish-crates.sh; then
    echo "❌ $name is publishable but missing from scripts/publish-crates.sh"
    FAIL=1
  fi
done < <(cargo metadata --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.publish != []) | .name')

if [ $FAIL -ne 0 ]; then
  echo "❌ Version check failed. Reconcile Cargo.toml with release-please-config.json."
  exit 1
fi

echo "✅ All crates and inter-crate deps at $WS_VERSION."

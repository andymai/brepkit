#!/usr/bin/env bash
set -euo pipefail

# Verify every Rust source path named in agent-facing docs still resolves.
#
# CLAUDE.md and the skill files are loaded as ground truth at the start of a
# session, so a path that has gone stale sends a session to a file that no
# longer exists. Refactors that split a `foo.rs` into a `foo/` directory are
# the usual cause and nothing else catches them.
#
# Paths are matched as suffixes against the real file list, so both
# `nurbs/curve.rs` and `math/src/traits.rs` resolve.

DOCS=(
  "CLAUDE.md"
  .claude/skills/*.md
  .claude/skills/*/*.md
)

# Paths that do not resolve on main by design.
ALLOWLIST=(
  # Cookbook templates naming the file a contributor is about to create.
  "io/src/format/mod.rs"
  "operations/src/op_name.rs"
  # Lives on the parked branch fix/kumiko-corner-window-cut, which the roadmap
  # records as must-not-ship. Delete this entry if that branch ever lands.
  "crates/io/tests/kumiko_corner_window_inmem.rs"
  "kumiko_corner_window_inmem.rs"
  # Named by fillet-blend/reference.md as a pass to design, not existing code.
  # That paragraph states outright that no such file exists.
  "reconcile.rs"
)

FAIL=0

FILE_LIST=$(find crates xtask/src tests -name '*.rs' -type f -not -path '*/target/*' 2>/dev/null | sed 's|^|/|')

is_allowlisted() {
  local path="$1"
  for a in "${ALLOWLIST[@]}"; do
    [ "$path" = "$a" ] && return 0
  done
  return 1
}

for doc in "${DOCS[@]}"; do
  [ -f "$doc" ] || continue

  # Backticked tokens ending in .rs, e.g. `math/src/cdt.rs`
  paths=$(grep -oE '`[A-Za-z0-9_./-]+\.rs`' "$doc" | tr -d '`' | sort -u || true)
  [ -z "$paths" ] && continue

  while IFS= read -r p; do
    [ -z "$p" ] && continue
    is_allowlisted "$p" && continue

    # Herestrings, not `echo | grep -q`: under pipefail, grep -q exiting at
    # the first match can EPIPE the echo and flip the pipeline status to
    # failure, reporting an existing file as a violation (a CI-only flake).
    if ! grep -qF "/${p}" <<< "$FILE_LIST"; then
      echo "VIOLATION: $doc references '$p', which does not exist"
      # A split into a directory is the common case; point at it directly.
      dir="${p%.rs}"
      if grep -qE "/${dir}/[^/]+\.rs$" <<< "$FILE_LIST"; then
        members=$(grep -oE "/${dir}/[^/]+\.rs$" <<< "$FILE_LIST" | sed "s|.*/||; s|\.rs$||" | sort | tr '\n' ' ')
        echo "           it is now a directory: ${dir}/ (${members% })"
      fi
      FAIL=1
    fi
  done <<< "$paths"
done

if [ $FAIL -ne 0 ]; then
  echo "❌ Doc path check failed."
  echo "   Update the path, or add it to ALLOWLIST in scripts/check-doc-paths.sh"
  echo "   if it names a file the reader is meant to create."
  exit 1
fi

echo "✅ Doc paths OK."

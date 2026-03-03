#!/usr/bin/env bash
# PostToolUse hook: Warn when a file imports from a higher layer
set -euo pipefail

# Read JSON from stdin and extract file path
INPUT=$(cat)
FILE=$(jq -r '.tool_input.file_path // .tool_input.filePath // empty' 2>/dev/null <<< "$INPUT")
[ -z "$FILE" ] && exit 0

# Only check .rs files in crates/
[[ "$FILE" == */crates/*.rs ]] || exit 0

# Determine which crate this file belongs to
CRATE=""
case "$FILE" in
  */crates/math/*)       CRATE="math" ;;
  */crates/topology/*)   CRATE="topology" ;;
  */crates/operations/*) CRATE="operations" ;;
  */crates/io/*)         CRATE="io" ;;
  */crates/wasm/*)       CRATE="wasm" ;;
  *) exit 0 ;;
esac

VIOLATIONS=""

check_import() {
  local forbidden="$1"
  local label="$2"
  if grep -q "use ${forbidden}" "$FILE"; then
    VIOLATIONS="${VIOLATIONS}  - imports ${label} (not allowed at this layer)\n"
  fi
}

case "$CRATE" in
  math)
    check_import "brepkit_topology" "brepkit-topology"
    check_import "brepkit_operations" "brepkit-operations"
    check_import "brepkit_io" "brepkit-io"
    check_import "brepkit_wasm" "brepkit-wasm"
    ;;
  topology)
    check_import "brepkit_operations" "brepkit-operations"
    check_import "brepkit_io" "brepkit-io"
    check_import "brepkit_wasm" "brepkit-wasm"
    ;;
  operations)
    check_import "brepkit_io" "brepkit-io"
    check_import "brepkit_wasm" "brepkit-wasm"
    ;;
  io)
    check_import "brepkit_wasm" "brepkit-wasm"
    ;;
esac

if [ -n "$VIOLATIONS" ]; then
  echo "⚠️  LAYER BOUNDARY VIOLATION in ${CRATE} crate:"
  printf "%b" "$VIOLATIONS"
  echo "See CLAUDE.md 'Layer dependency rules' for allowed imports."
fi

#!/usr/bin/env bash
# PostToolUse hook: Warn when unwrap()/expect() appears outside test code
set -euo pipefail

# Read JSON from stdin and extract file path
INPUT=$(cat)
FILE=$(jq -r '.tool_input.file_path // .tool_input.filePath // empty' 2>/dev/null <<< "$INPUT")
[ -z "$FILE" ] && exit 0

# Only check .rs files
[[ "$FILE" == *.rs ]] || exit 0

# Skip test files, test modules, and benchmarks
[[ "$FILE" == *test_utils* ]] && exit 0
[[ "$FILE" == *tests/* ]] && exit 0
[[ "$FILE" == */benches/* ]] && exit 0

# Find the first #[cfg(test)] line — everything after it is test code
TEST_LINE=$(grep -n '#\[cfg(test)\]' "$FILE" 2>/dev/null | head -1 | cut -d: -f1 || true)

if [ -n "$TEST_LINE" ]; then
  # Only check lines before the test module
  REGION=$(head -n "$((TEST_LINE - 1))" "$FILE" || true)
else
  REGION=$(cat "$FILE")
fi

[ -z "$REGION" ] && exit 0

if grep -q '\.unwrap()' <<< "$REGION"; then
  echo "⚠️  Found .unwrap() in non-test code of $(basename "$FILE") — use Result instead."
fi

if grep -q '\.expect(' <<< "$REGION"; then
  echo "⚠️  Found .expect() in non-test code of $(basename "$FILE") — use Result instead."
fi

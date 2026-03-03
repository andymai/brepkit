#!/usr/bin/env bash
# PostToolUse hook: Remind about ripple-effect when editing enum definitions
set -euo pipefail

# Read JSON from stdin and extract file path
INPUT=$(cat)
FILE=$(jq -r '.tool_input.file_path // .tool_input.filePath // empty' 2>/dev/null <<< "$INPUT")
[ -z "$FILE" ] && exit 0

# Check if this is the EdgeCurve enum definition file
if [[ "$FILE" == */topology/src/edge.rs ]]; then
  echo "🔔 RIPPLE-EFFECT REMINDER: You edited edge.rs (contains EdgeCurve enum)."
  echo "   If you added/changed a variant, check all match sites in CLAUDE.md → 'Adding an EdgeCurve variant'"
  echo "   Key files: tessellate.rs, transform.rs, copy.rs, measure.rs, boolean.rs,"
  echo "   step/writer.rs, iges/writer.rs, kernel.rs (8 sites)"
fi

# Check if this is the FaceSurface enum definition file
if [[ "$FILE" == */topology/src/face.rs ]]; then
  echo "🔔 RIPPLE-EFFECT REMINDER: You edited face.rs (contains FaceSurface enum)."
  echo "   If you added/changed a variant, check all match sites in CLAUDE.md → 'Adding a FaceSurface variant'"
  echo "   Key files: tessellate.rs, transform.rs, copy.rs, section.rs, distance.rs,"
  echo "   boolean.rs (4 sites), step/writer.rs, iges/writer.rs, kernel.rs (8 sites)"
  echo "   ⚠️  offset_face.rs, step/writer.rs, iges/writer.rs have wildcard catch-alls!"
fi

# Check if this is the AnalyticSurface enum
if [[ "$FILE" == */math/src/analytic_intersection.rs ]]; then
  echo "🔔 RIPPLE-EFFECT REMINDER: You edited analytic_intersection.rs (contains AnalyticSurface enum)."
  echo "   Check match sites in analytic_intersection.rs (4 sites) and boolean.rs (4 sites)"
fi

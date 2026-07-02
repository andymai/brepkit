---
name: debugging-doctrine
description: The meta-method for debugging hard geometry bugs in brepkit, used when a boolean, fillet, offset, or tessellation produces wrong volume, wrong face count, open or non-manifold shells, mesh fallback, misclassified points, or any failure where the first diagnosis did not immediately hold. Also applies before starting any multi-pass investigation, when a repro behaves differently from the real case, or when successive root-cause theories keep overturning each other.
---

# Debugging Doctrine

The method that separates a converging investigation from a thrashing one. Every rule below was paid for: static guesses that skipped instrumentation were wrong, unfaithful repros burned ten-pass detours, and plausible diagnoses pointed at code that never executed.

For concrete GFA phase-by-phase instrumentation recipes, see the **boolean-debugging** skill. For watertightness and volume verification, see **solid-verification**. Detailed catalogs live in [reference.md](reference.md).

## Quick reference

```bash
cargo run --release --example debug_boolean -p brepkit-operations   # canonical dump harness, edit in place
cargo run --release --example approx_census -p brepkit-operations   # which ops fall back to mesh/NURBS
cargo test -p brepkit-io --test lipfuse_fixture                     # fixture-replay template
```

| Symptom | First suspect | Not this |
|---|---|---|
| Hundreds of all-planar faces | Mesh fallback fired: the GFA (general fuse algorithm, the boolean engine in `crates/algo`) result failed the gate in `crates/operations/src/boolean/mod.rs` | "Tessellation is broken" |
| Coincident-contact boolean wrong | Classification (interior sample on a rim) | Same-domain (SD) detection. Instrument `detect_same_domain` only to rule it out; it has been wrongly blamed twice |
| Boolean output malformed | The operation INPUTS (dump both operands first) | The boolean itself |
| Volume slightly off | Un-carved or double-counted region masked by tessellation error | "Measure is imprecise". Volume is never ground truth |
| Point classified OUT that looks inside | Winding classifier on a faceted solid | The solid. Use `classify_point` (ray-cast) as ground truth |
| Shell open / inside-out / Euler off after fuse | Degenerate SECTION inputs, two layers up (`build_section_edges`) | The shell assembler |
| Fix looks right but case unchanged | The patched path never executes | "The fix needs more work" |

## The procedure

### 1. Build a faithful repro, then VALIDATE it (before any debugging)

A repro that does not reproduce the real case is worse than none: it manufactures red herrings. A proxy with corner radius 5 once straddled a wall the real radius 1.485 never touches; ten passes of findings applied only to geometry the real case never produces.

- Capture the ACTUAL failing operands. STEP path: export from the tool (`k.exportSTEP` in a brepjs vitest), replay via `brepkit_io::step::reader::read_step`. Template: `crates/io/tests/lipfuse_fixture.rs`.
- If STEP renumbers away the failure (in-memory id-layout bugs) use the lossless arena snapshot: `serialize_solid` / `deserialize_solid` in `crates/io/src/arena_io.rs` (JS: `serializeSolid` / `deserializeSolid`). The `*_inmem.rs` tests in `crates/io/tests/` are the pattern.
- Checkpoint: repro must match the real case on operand surface types (analytic, not NURBS-reconstructed), face counts, and the exact failure signature. If any differ, fix the repro first. See [reference.md, Faithful fixtures](reference.md#faithful-fixtures).

### 2. Dump literal data before theorizing

Never reason from the symptom alone. Dump: total volume, per-face surface type and area, per-face signed volume, entity counts (V, E, F), free-edge and over-shared-edge counts, literal wire-edge traversals. `crates/operations/examples/debug_boolean.rs` is the editable template; expected output shape and every dump helper are in [reference.md, Instrumentation catalog](reference.md#instrumentation-catalog).

### 3. Vary ONE variable at a time

When a theory involves a parameter, sweep that parameter alone on a pristine primitive (`operations/src/primitives.rs`, bases at z=0) inside a `debug_boolean.rs`-style harness. Changing radius AND pocket-presence together once produced a bogus theory that a clean radius sweep on a bare box collapsed to a two-line repro.

### 4. Confirm the suspect path FIRES

Before trusting a diagnosis, gate the candidate code behind a temporary flag or early-return and check whether the failing case changes at all. A plausible static diagnosis once blamed an arc-split branch that executed zero times; the real fix was elsewhere in the same file. When toggling, check INTERMEDIATE state (shell closed, signed volume), not just the final error string. Checkpoint: toggle changes nothing observable means wrong code path, go back to step 2.

### 5. Trace below the gate

`operations::boolean::boolean` runs GFA, then an acceptance gate (`euler_ok && open_shell_ok && validate_boolean_result`), and on failure substitutes a mesh fallback (a warning is logged, but callers see Ok). An operations-level result can look plausible while the real GFA output was rejected. Verify at BOTH levels: `brepkit_algo::gfa::boolean` directly (below the gate) and through the full pipeline. Trust no "this one fix closes it" prediction, including from a subagent, until a direct raw-engine trace on the real operands confirms it; one such trace turned a single predicted fix into five.

### 6. Expect layers; peel and re-instrument

Hard cases are routinely three to five stacked bugs, each fix exposing the next (one cut needed five successive correct diagnoses). Peeling layers is normal progress, not failure, but only if you re-dump at every layer. Never assume the previous dump still describes the state after a fix.

### 7. Treat diagnosis instability as a finding

If several rigorous passes each overturn the previous root cause, you are in a multi-bug region or your repro is unfaithful. Stop. Do not run another same-shaped pass. Build better tooling instead: a lossless fixture (arena snapshot), targeted instrumentation, an intermediate-state dump. The arena serializer itself was built as the exit from exactly this trap, and it worked. Symptom chases that instrument the last stage (the assembler) while the bug is two layers up (section construction) look like instability; dump each stage's INPUTS first.

## Anti-patterns (do not conclude)

- "Volume matches, so the boolean is correct." Volume is tessellation-based (`measure/volume.rs`, deflection clamped to bbox_diag * 5e-5) and has masked a fully un-carved slot. Face count per surface type is the reliable tell.
- "Triangle count and validity look fine, so no fallback." Both mask mesh fallback. Only face count exposes it.
- "The winding classifier says OUT, so the point is outside." Wrong on faceted, stepped, and NURBS-heavy solids. Ground truth is `brepkit_check::classify::classify_point`.
- "Same-domain detection must have missed the overlap." Check the classifier first; SD was innocent both times it was blamed.
- "The fix compiles and the theory is coherent, so ship it." Verify the path fires (step 4) and the case closes on the real fixture, or revert. Unshippable partial progress goes to a branch, never to main.
- "Generalize while we are here." Solve the narrow problem. A general merge-key primitive was provably impossible where a per-case splitter-side midpoint split worked; two callers needed opposite behavior from the same key.
- "Handle the planar case now, curved later." No shortcuts in CAD algorithms: cover periodic, closed, degenerate, and inner-wire cases, or split into complete sub-components. Never ship an incomplete whole.
- Effort commentary. Report the technical situation only: what is wrong, what the fix is, what is verified. Keep driving until solved or genuinely blocked (blocked means a user decision or credential is needed).

## Priors worth memorizing

- Coincident-contact boolean bug: classification before same-domain. Two interior-sample paths exist (`sample_face_interior` in `builder/mod.rs` for unsplit faces, `interior_point_3d` in `builder/face_splitter/mod.rs` for split ones); a hazard fixed in one can be open in the other.
- Malformed boolean output: verify the INPUTS before blaming the operation. A "boolean bug" was once a malformed shell fed in.
- Simple fix at the layer that owns the artifact beats a clever fix at the wrong layer.
- Scorecards and face-count baselines go stale: re-probe after every GFA change.

Case studies, dump recipes, and the glossary: [reference.md](reference.md).

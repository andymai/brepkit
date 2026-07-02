---
name: numerical-robustness
description: Use when debugging floating-point failures in brepkit: a boolean or heal result flips pass/fail across runs or under tiny (1e-13) input nudges, closed/periodic geometry (circles, tori, seams) produces collapsed AABBs or wrong arcs, intersection points miss chord-discretized seams, near-duplicate vertices fail to merge, or when writing any code that compares floats, hashes coordinates, or iterates edges of closed curves.
---

# Numerical Robustness in a CAD Kernel

Deep catalog with verified paths, procedures, and case studies: [reference.md](reference.md).

## When to use

- A test passes on some runs and fails on others (same code, same input).
- A boolean fails, but translating an operand by ~1e-13 makes it pass (or vice versa).
- A face AABB, winding, or extent computation is wrong on a cylinder, sphere, or torus.
- An intersection or split "never fires" even though the geometry clearly crosses.
- Two vertices or edges that should be the same entity stay distinct (free edges, open shells).
- You are writing code that compares floats, buckets coordinates, or walks a closed curve.

## Symptom-to-cause table

| Symptom | Likely cause | See reference.md |
|---|---|---|
| Pass/fail varies across processes, stable within one | HashMap iteration order drives branching | §5 Nondeterminism |
| Knife-edge sensitivity: 7e-14 nudge passes, 1e-13 fails | Near-duplicates straddle adjacent quantization cells | §2 Quantization buckets |
| Free edges / open shell after a boolean, mesh fallback | Duplicate-edge merge missed ULP-apart vertices | §2 Quantization buckets |
| AABB of a curved face collapses to a line or point | Endpoint-only computation on a closed curve (start == end) | §3 Closed geometry |
| Wrong arc recovered (major vs minor) after copying an edge | Stored vertex order disagrees with curve parameterization | §3 Periodic copy hazard |
| Intersection against a seam finds zero crossings | Point lies on the true curve, off the stored chords (sagitta gap) | §4 Chord vs analytic |
| Triangulation self-contradicts or fails to converge | Tolerance comparison used where a sign decision was needed | §1 Exact predicates |
| Two co-endpoint edges wrongly collapse (or wrongly stay distinct) | Endpoint-pair merge key cannot see curve identity | §2 Merge-key lesson |

## The six rules

1. **Never `==` on floats.** Use `Tolerance::approx_eq` (scale-aware) for coordinates, `approx_eq_abs` for parameter-space values, `parametric()` to convert linear tolerance into parameter space. Defaults: linear 1e-7, angular 1e-12, relative 1e-10 (`crates/math/src/tolerance.rs`). For sign/topology decisions (which side, inside circle, segments cross) tolerance is the wrong tool: use exact predicates in `crates/math/src/predicates.rs` (`orient2d`, `orient3d`, `in_circle`, `point_in_polygon`) or the fast filtered variants in `crates/math/src/filtered.rs`. Details: reference.md §1.

2. **Quantized-coordinate keys are knife-edges.** Any scheme keying on `(coord * scale).round()` fails when two points a few ULPs apart (ULP: unit in the last place, the gap between adjacent f64 values, about 2e-16 relative) land in adjacent cells. The proven fix shape is to weld first with a snap radius larger than the bucket (`weld_coincident_vertices` in `crates/algo/src/builder/builder_solid.rs`, snap = 10x `MERGE_TOL`), not to widen the bucket. Hashes over quantized floats are also order-fragile: build sorted canonical forms (`compute_edge_set_quantized` in `crates/algo/src/builder/same_domain.rs` sorts its pair list). Details: reference.md §2.

3. **Closed curves have start == end.** Any endpoint-only computation (AABB, winding, extent, degeneracy test) silently collapses on a full circle or torus boundary. Always sample along the curve (`compute_face_bbox` in `crates/algo/src/pave_filler/phase_ff.rs` samples 9 points per edge). Degenerate seam edges `Edge::new(v0, v0, EdgeCurve::Line)` exist by construction on full tori (`make_torus` in `crates/operations/src/primitives.rs`). Copying a periodic curve onto a new edge requires vertex order to match the parameterization, or `domain_with_endpoints` recovers the complementary arc: see `reverse_edge_curve` in `crates/operations/src/extrude.rs`. Details: reference.md §3.

4. **Chords are not the curve.** A discretized seam (the sphere equator is stored as line chords) deviates from the true curve by the sagitta `r*(1 - cos(theta/2))` (`crates/math/src/chord.rs`). An intersection point computed against the true curve will not lie on the chords, and every 3D-based layer downstream inherits the gap. Solve such arrangements in UV space where the seam is exact, or against the analytic carrier (`sphere_seam_plane_crossings` in `phase_ff.rs`). Details: reference.md §4.

5. **Nondeterminism signature: stable within a process, varies across processes.** Cause is almost always std `HashMap` iteration order feeding a branch. Fix: sort before branching (`sort_unstable_by_key` on ids, `total_cmp` for float keys). Bisect procedure and in-repo examples: reference.md §5.

6. **Prove the suspect path fires.** Before trusting a diagnosis, add a hit counter or gate the fix behind a flag and confirm the code path executes on the failing input. A plausible static diagnosis once blamed a path that ran zero times. Procedure: reference.md §6.

## Quick procedures

**Confirm cross-process nondeterminism** (checkpoint: expect a mix of `ok` and `FAILED`):

```bash
for i in $(seq 20); do
  cargo test -p brepkit-operations --release the_failing_test -- --exact 2>&1 | tail -1
done
```

Twenty in-process repeats inside one `cargo test` invocation prove nothing; each fresh process reseeds the HashMap. If all 20 agree, the bug is not seed-driven: look for input-dependent tolerance straddle instead (reference.md §2).

**Probe a bucket-straddle** (checkpoint: expect non-monotonic pass/fail): translate the failing operand by a ladder of tiny offsets (1e-14, 7e-14, 1e-13, 2e-13, 1e-12) and rerun. Pass at 7e-14 and 2e-13 but fail at 1e-13 is the quantization-cell signature.

**Find the relevant sites:**

```bash
rg -n 'approx_eq' crates/ | head
rg -n 'pub fn' crates/math/src/predicates.rs crates/math/src/filtered.rs
rg -n 'quantize_point|MERGE_TOL' crates/algo/src/builder/
```

**Verify a boolean result geometrically:** use the ray-cast classifier `brepkit_check::classify::classify_point` (`crates/check/src/classify/mod.rs`), never volume alone (tessellated volume once read 1.4% high and nearly masked an un-carved slot) and never the winding classifier on faceted solids. See the solid-verification skill.

## Anti-patterns: what NOT to conclude

- Do NOT conclude "the tolerance is too tight, widen it." Widening a quantization bucket moves the knife-edge; it does not remove it. Weld below the bucket instead (rule 2).
- Do NOT conclude a test is flaky and retry-loop it. Cross-process variance is a real determinism bug with a known fix shape (rule 5).
- Do NOT trust a fix because the test went green. A fix once "worked" because a wrong parameter range rejected the bad case for the wrong reason. Confirm the mechanism: the suspect path fired, and toggling the fix flips the outcome (rule 6).
- Do NOT make a shared merge key "smarter" to resolve one workload. The endpoint-pair edge-merge key is load-bearing: one workload needs co-endpoint chord+arc to collapse, another needs them distinct. Control the geometry you emit (splitter-side midpoint split) instead of discriminating in the key (reference.md §2).
- Do NOT evaluate curve parameters over an assumed `[0,1]`. `evaluate_with_endpoints` takes the native parameter (radians for circles, knot values for NURBS); get the range from `domain_with_endpoints` (`crates/topology/src/edge.rs`).
- Do NOT assume "no mesh fallback" means correct. Analytic output can still be geometrically wrong; classify points (solid-verification skill).
- Do NOT patch a chord-vs-analytic gap one layer at a time. Four consecutive downstream layers hit the same sagitta gap in one case; move the computation to UV or the analytic carrier once (rule 4).

## Related skills

- **boolean-debugging**: GFA pipeline phases, where these failures surface.
- **solid-verification**: watertightness, classifier choice, volume traps.
- **debugging-doctrine**: vary one variable, dump literal data, fix at the owning layer.
- **analytic-preservation**: keeping exact surface types through operations.
- **testing**: property tests and golden files that catch these regressions.

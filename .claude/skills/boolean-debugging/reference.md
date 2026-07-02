# Boolean debugging reference

Companion to [SKILL.md](SKILL.md). Symbols are the durable locators; line numbers rot, so
find everything with `rg -n 'fn <symbol>' crates/...`.

## Glossary

- **GFA**: the boolean engine in `crates/algo` (general-fuse family). Entry `brepkit_algo::gfa::boolean(topo, op, a, b)`. Pipeline: PaveFiller, then Builder, then select/assemble.
- **PaveFiller** (`crates/algo/src/pave_filler/`): computes pairwise interferences between the operands' sub-shapes in phases VV/VE/EE/VF/EF/FF (vertex/edge/face pairs), producing paves and pave blocks.
- **Pave / PaveBlock** (`crates/algo/src/ds/pave.rs`): a split point on an edge / the edge fragment between two paves.
- **FF phase / sections** (`pave_filler/phase_ff.rs`): face x face intersection; produces the section curves along which faces are split. Most boolean bugs live here or just downstream.
- **Builder / face splitter** (`crates/algo/src/builder/`): splits each face in UV space along its sections (`face_splitter/`), classifies sub-faces In/Out/On against the opposing solid, then selects and assembles shells (`builder_solid.rs`, `assemble.rs`).
- **SD (same-domain)**: detection of coincident overlapping faces across operands, `detect_same_domain` in `builder/same_domain.rs`, so ops can dedupe or cancel them.
- **Mesh fallback**: `mesh_boolean_fallback` in `operations/src/boolean/mod.rs`, co-refinement on tessellated triangles; result is all-planar, all analytic surface types lost.
- **Euler check**: V-E+F acceptance, hole-aware (a face with L inner wires raises V-E+F by L) and hollow-aware (each cavity shell adds a surplus of 2); `euler_balanced` in `boolean/mod.rs`.
- **Free edge / over-shared edge**: edge used by exactly 1 face (open shell) / 3+ faces (non-manifold). Closed manifold means every edge used exactly twice.
- **Seam edge**: the u=0 equivalent u=2pi closure edge of a periodic surface. Closed edges have start == end, which breaks any endpoint-based logic.
- **Classifiers**: analytic closed-form (`algo/src/classifier/analytic.rs`), ray-cast parity (`check/src/classify/mod.rs::classify_point`, mirrored by `algo/src/classifier/ray_cast.rs`), winding number (`classify_point_winding`, wrong for faceted solids).
- **`brepkit_approx` probe**: `log::debug!` target fired whenever an approximation path is taken; captured by the `approx_census` example.
- **Arena serialization**: byte-exact solid capture/replay (`crates/io/src/arena_io.rs`, wasm `serializeSolid`) for repros that STEP's ~15-sig-fig serialization would normalize away.

## Bug-class catalog

Match your symptom here before inventing a new theory. "Fix location" is where the class was
owned and fixed; recurrences of the same class usually land in the same file.

| Bug class | Symptom | Root cause | Fix location / doctrine |
|---|---|---|---|
| Wire spurs | Fused solid non-manifold with inflated volume; wire traversal shows a consecutive same-edge forward-then-reverse pair; an edge shared by 3+ faces | GFA wire builder emits an out-and-back excursion on a face with a single-opening notch; zero area, but the edge is counted twice | `remove_wire_spurs` in `crates/operations/src/heal.rs`, called from the boolean gate. Always sound to strip. Caveat: a spur can mask a genuine missing adjacency; after stripping, the edge may drop to free, proving a deeper hole |
| Coincident-ring vertex weld | Coincident-contact fuse comes out non-manifold or free-edged; volume wrong only for asymmetric footprints; near-duplicate vertices ~1e-6 apart | PaveFiller places ring vertices slightly off pre-existing vertices; `merge_duplicate_edges` quantizes endpoints at `MERGE_TOL` (1e-7) so near-twins land in different cells and the boundary never unifies | `weld_coincident_vertices` in `crates/algo/src/builder/builder_solid.rs` (snap radius 10x MERGE_TOL, runs before edge split/merge) |
| Face-AABB collapse on closed edges | Faces perpendicular to a cylinder/cone axis vanish (free edges trace the missing rim) while axis-parallel walls survive; or `AssemblyFailed("no faces selected")` on a full torus | Face AABB built from boundary edge endpoints only; a closed circular edge has start == end so its box collapses to a point, and curved faces bulge past their boundary polygon | Surface-aware `compute_face_bbox` in `pave_filler/phase_ff.rs`, plus arc-trim AABBs in `emit_split_circle_arcs`. Doctrine: closed boundary edges (Circle, closed NURBS) must be sampled ALONG the curve, never endpoint-checked |
| Closed-circle sections emitted as NURBS | A coaxial analytic boolean mesh-falls-back; the section wire on a shared rim is full of tiny arcs or closed-NURBS loops; a wall loses its bottom boundary | The seam-adopt path (phase_ff closed-circle handling plus `pave_filler/link_existing.rs`) is gated on `EdgeCurve::Circle`; a NURBS rim gets adopted as a redundant fresh edge instead. The general marcher also fragments near-tangent circles into ~100 micro-curves | Emit exact circles from `crates/math/src/analytic_intersection.rs` and construct `EdgeCurve::Circle` in phase_ff. `exact_cone_cone` and `exact_sphere_cylinder` are `pub`; `exact_plane_cone` and `algebraic_cylinder_cylinder` are private, reached via the public dispatchers `exact_plane_analytic` and `intersect_analytic_analytic`. See the analytic-preservation skill for adding an exact arm |
| `merge_duplicate_edges` co-endpoint collapse | A circle split into two half-arcs sharing BOTH endpoints collapses to one edge; E is too small, Euler off, gate rejects, fallback | `merge_duplicate_edges` (`builder_solid.rs`) keys on quantized endpoint pairs only, with no curve discriminator | Doctrine fix is SPLITTER-SIDE: give co-endpoint seam arcs a midpoint vertex (the pi * 0.999 split divisor in `emit_split_circle_arcs`) so no two edges ever share both endpoints. Do NOT make the merge key smarter (type key or deviation threshold): that provably conflicts with the gridfinity lip case, where a chord and an arc with the same endpoints and millimeter-scale deviation MUST merge |
| Thin-face FF filters drop valid sections | A razor-thin face (~0.03 to 1 mm wide) that should split becomes ONE full-width sub-face, gets one classify sample, and is kept or dropped wholesale; a fuse shows a hole where the thin strip's center is missing while a thicker parallel strip split fine | Several conservative filters in `phase_ff.rs` reject sections on thin faces: the uniform-sample AABB pre-filter, the midpoint-in-both test (midpoint evaluated outside the thin band), the ellipse-only section gate | `restrict_curves_to_faces` and `trim_ellipse_to_boundary_crossings` in `phase_ff.rs`. Fixes are case-by-case and regression-prone; re-run the full parity probes after any change here |
| Interior sample lands on a coincident rim | Coincident-contact cut/fuse where an analytic cavity or contact wall vanishes; the analytic classifier returns None exactly on the rim, the ray-cast fallback says Outside, face dropped | A periodic wall's UV boundary polygon is lopsided, so the 2D interior sample lands on a v-extreme, which IS the shared rim | Two paths, check BOTH: `sample_face_interior` (`builder/mod.rs`, unsplit faces, mid-v sampling for periodic walls) and `interior_point_3d` (`builder/face_splitter/mod.rs`, split faces, snap-to-mid-v). A hazard fixed in one has historically stayed open in the other |
| Sphere chord-discretized equator seam | plane x sphere sections fail to split a hemisphere: true crossing points sit off the inscribed-polygon seam chords by the sagitta, far beyond tolerance | Spheres are built as two hemispheres joined along a CHORD-discretized equator seam; every 3D layer then fights the chords | Partially closed: `sphere_seam_plane_crossings` in `phase_ff.rs` (Newell-plane crossings, facet-independent) plus the `split_noseam_by_arrangement` DCEL walk in `face_splitter/special_cases.rs`. Known open residual: a lone plane x sphere halfspace cut never reaches the 2D splitter (section-creation gap in the pavefiller). Identified right architecture: a UV-space arrangement splitter, where the equator is a clean v=0 line and the problem is degeneracy-free. That is a dedicated multi-day component, not yet built |
| Degenerate FF sections | Open shell / inside-out / Euler-unbalanced result on a fuse with contact faces | Zero-span arc PaveBlock remnants, or open arcs re-tracing an existing inner-wire hole, injected as sections | `build_section_edges` in `builder/fill_images_faces.rs`. The shell/Euler symptoms were all downstream: dump each contact face's INPUT sections before instrumenting the assembler |
| Phantom arrangement chord-breaks | A convex arc boundary gains break points that are not on it, producing a keyhole loop and fallback | `split_plane_face_by_arrangement` (`face_splitter/mod.rs`) registered chord intersections as breaks on the arc itself | Gated by the `chord_break_on_arc` check in the same file. Related trap: `evaluate_with_endpoints` takes the curve's NATIVE parameter (radians, knots), not a normalized [0,1] |
| Shell sign / "no outer shell found (all shells classified as holes)" | `AssemblyFailed` with that exact message from `builder_solid.rs`; the recurring assembler death | Shell orientation sign taken from raw wire winding instead of the geometric surface normal, or a genuinely never-created face leaves large free-edge gaps so no shell closes | `signed_volume_of_shell` in `builder_solid.rs` (sign by surface normal). If free-edge gaps are large (millimeter scale, not near-twin vertices), the missing face is upstream in section creation, not a weld problem |

## `debug_boolean.rs`: what it does and how to extend it

`crates/operations/examples/debug_boolean.rs`. Run:

```bash
cargo run --release --example debug_boolean -p brepkit-operations
```

Current behavior: builds `make_box(50,30,10)` and `make_cylinder(5,20)`, translates the
cylinder to the box center with `Mat4::translation(25,15,-5)` (primitives put their base at
z=0), runs `boolean(Cut)` through the FULL operations gate, then prints:

- face count and per-face surface type (exhaustive `FaceSurface` match; Plane printed with normal and d), `reversed`, `inner_wires` count, area
- per-face signed volume via the divergence theorem over the tessellation, and the total
- `measure::solid_volume` vs the closed-form expected value

Caveat: it iterates `outer_shell()` only. For hollow results, switch to
`brepkit_topology::explorer::solid_faces` (CLAUDE.md, "Walking faces in a solid").

Extensions, in the order you usually need them:

1. **Swap operands.** Replace the primitives/transform with your repro. One variable per run.
2. **Bypass the gate.** Call `brepkit_algo::gfa::boolean(&mut topo, algo_op, a, b)` directly
   (op enum from `crates/algo/src/bop.rs`). This skips the operations-layer shortcuts, heal
   passes, and fallback, so you see the raw GFA artifact.
3. **Edge-usage census.** `has_free_edges` and `is_closed_manifold` are private to
   `operations/src/boolean/mod.rs`; reimplement: walk every face's wires, count usages per
   `EdgeId` in a map, report edges with count 1 (free) and count >= 3 (non-manifold).
4. **Wire traversal dump.** For each face, print the ordered (edge id, orientation) list per
   wire. A consecutive same-edge forward-then-reverse pair is a spur.
5. **Logger.** Add `env_logger::init()` at the top of `main` (already a dev-dependency;
   pattern in `examples/profile_boolean.rs`) and run with `RUST_LOG=debug` to see the
   "falling back" warnings and `brepkit_approx` probes.

The census example: `cargo run --release --example approx_census -p brepkit-operations` runs
`boolean_matrix()` over overlapping primitives plus offset/fillet/chamfer, captures the
`brepkit_approx` log target in-process, and reports exact vs which-fallback, wall time, and
face count per op. A census "exact" verdict only proves no fallback probe fired; the geometry
can still be wrong (an un-carved cut has scored "exact"). Follow with the ray-cast check.

## Faithful repro: full recipe

The fixture-tier catalog (native repro, STEP, arena `.bin`) and test templates live in the
testing skill. Boolean-specific rules on top of it:

1. Export each failing boolean's OPERANDS from the tool, one file per operand per step:
   `k.unwrap(k.exportSTEP(operand))`. Never capture a serialized batch result; a mid-bisect
   intermediate looks plausible and burns a full pass.
2. Read back with `brepkit_io::step::reader::read_step(input: &str, &mut topo)`: it takes
   the file CONTENTS as `&str` and returns `Result<Vec<SolidId>, IoError>`, unlike the
   generic reader shape in the CLAUDE.md cookbook.
3. **Gate: confirm the round-trip is analytic.** Run the face census (Rust `type_tag`, or
   tool-side `getSurfaceType`/`getEdgeCurveType`). Expect cylinder/cone/plane faces and
   Circle edges. NURBS where the source had analytics means the fixture is unfaithful and
   any fix derived on it will overfit and not transfer. Bugs that depend on sub-ULP vertex
   noise that STEP rounds away need the arena tier (`serializeSolid`,
   `brepkit_io::arena_io`); see the testing skill.
4. Replay the boolean on the imported operands via `gfa::boolean` direct, then via
   `operations::boolean`, and compare.
5. If you built a synthetic proxy instead (e.g. a hand-placed 1e-13 nudge), validate that it
   flips the same way as the real tool case BEFORE deep-debugging on it. Proxies have
   diverged from the real case and cost multiple non-transferring iterations.

## Gate internals worth knowing when a gate rejects

All in `crates/operations/src/boolean/mod.rs`; locate with
`rg -n 'euler_ok|open_shell_ok|hollow_ok|euler_balanced|inner_shell_surplus' crates/operations/src/boolean/mod.rs`.

- `open_shell_ok`: `op != Intersect || !has_free_edges`. Cut and Fuse deliberately accept
  open shells because an open GFA shell usually loses less than the mesh fallback would.
- `euler_eff`: V-E+F minus `inner_shell_surplus` (2 per cavity shell). `euler_balanced(euler, inner_wires)`
  accepts genus >= 0 with the inner-wire correction.
- `hollow_ok`: results with inner shells must be closed manifold.
- Accept: `euler_ok && open_shell_ok && validate_boolean_result(..)` (validator in
  `boolean/assembly.rs`).
- Multi-region Cut: accepted separately when components are disjoint, Euler is 2 per
  component, the result is closed manifold, AND `cut_safe` holds. The `cut_safe` check
  (each component center classifies as Outside operand B, so the tool's interior piece
  cannot survive) only runs when `try_build_analytic_classifier(topo, b)` succeeds. For a
  non-analytic B it defaults to true and the interior-piece guard is skipped.
- There is no heal-retry after a reject. The `unify_faces` (up to 3x) and
  `remove_wire_spurs` heals run pre-gate only. The real post-reject path: for Cut, when the
  INPUT solid A has two or more disjoint components, `boolean()` retries via
  `cut_multi_region_input` (per-component cuts, recombined); every other rejected or errored
  case goes straight to `mesh_boolean_fallback`.

When `euler_ok` fails, the productive question is which term is off: E too small points at
edge over-merging (co-endpoint collapse row above); F too small points at dropped faces
(AABB collapse, thin-face filters, rim misclassification rows); free edges point at a
never-created section or a weld miss.

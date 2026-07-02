# io-formats reference

Verified against current source. Locate symbols with the given `rg` patterns; line
numbers rot, symbols do not. Key files: `crates/io/src/lib.rs`,
`crates/io/src/step/{reader,writer}.rs`, `crates/io/src/iges/{reader,writer}.rs`,
`crates/io/src/stl/import.rs`, `crates/io/tests/cross_format.rs`,
`crates/wasm/src/bindings/io.rs`, `crates/operations/src/heal.rs`,
`crates/heal/src/custom/convert_to_elementary.rs`,
`crates/geometry/src/convert/recognize_{surface,curve}.rs`,
`crates/check/src/classify/mod.rs`, `crates/io/src/arena_io.rs`.

## 1. STEP reader anatomy

`rg -n 'pub fn read_step' crates/io/src/step/reader.rs`

Flow: `read_step` parses the `DATA;`..`ENDSEC;` block into a
`HashMap<u64, StepEntity { entity_type, attrs }>` (split on `;`, no regex), then
`StepBuilder::build_all_solids` assembles topology.

### Entity to topology mapping

`rg -n 'MANIFOLD_SOLID_BREP|CLOSED_SHELL|ADVANCED_FACE|EDGE_LOOP|ORIENTED_EDGE|EDGE_CURVE' crates/io/src/step/reader.rs`

- `MANIFOLD_SOLID_BREP` -> `Solid` with a single outer shell and no inner shells
  (`Solid::new(shell_id, Vec::new())`). This is the only entity that becomes a solid.
  Anything stored as `SHELL_BASED_SURFACE_MODEL`, `GEOMETRIC_SET`, `BREP_WITH_VOIDS`, or a
  wireframe rep is filtered out with no error (`build_all_solids` filters
  `entity_type == "MANIFOLD_SOLID_BREP"`).
- `CLOSED_SHELL` -> `Shell`.
- `ADVANCED_FACE` -> `build_face`. Reads the `.T.`/`.F.` reversed flag
  (`Face::new` vs `Face::new_reversed`). `FACE_OUTER_BOUND` gives the outer wire, other
  bounds give inner wires; if none is flagged outer, the first bound is used as outer.
- `EDGE_LOOP` -> `Wire`; `ORIENTED_EDGE` (`.T.` = forward) sets edge orientation in the
  wire.
- `EDGE_CURVE` -> `build_edge_curve` -> `Edge::new(start_vp, end_vp, curve)` with a 3D
  curve only. Vertices come from `VERTEX_POINT` / `CARTESIAN_POINT`.

### Surface / curve construction

`rg -n 'fn build_surface|fn build_curve_geometry' crates/io/src/step/reader.rs`

`build_surface` arms: `PLANE` -> `Plane`, `CYLINDRICAL_SURFACE` -> `Cylinder`,
`CONICAL_SURFACE` -> `Cone` (half_angle in radians), `SPHERICAL_SURFACE` -> `Sphere`,
`TOROIDAL_SURFACE` -> `Torus`, `B_SPLINE_SURFACE*` -> `Nurbs` (rational detected by
`attrs.contains("RATIONAL")`). `build_curve_geometry` arms: `LINE` -> `Line`,
`CIRCLE` -> `Circle`, `ELLIPSE` -> `Ellipse`, `B_SPLINE_CURVE_WITH_KNOTS` -> NURBS.

Both functions end in `_ => Err(IoError::UnsupportedEntity { entity })`. There is no
skip or substitute path: one unhandled type (for example `SURFACE_OF_REVOLUTION`,
`SURFACE_OF_LINEAR_EXTRUSION`, an offset surface, or a bounded-curve subtype) fails the
entire import.

### What the reader does NOT do

- No pcurve / seam-curve reading. STEP `PCURVE` / `SEAM_CURVE` records are ignored;
  trimming is topological only (vertex endpoints plus edge-loop order). Pcurves are
  recomputed downstream when an operation needs UV (`crates/algo/src/builder/pcurve_compute.rs`).
- No unit handling. `GLOBAL_UNIT_ASSIGNED_CONTEXT` / `SI_UNIT` are not parsed; coordinates
  are taken as authored (assumes mm). A file in inches or metres imports at wrong scale
  silently.
- No healing. Parse and build only.
- Deterministic within a solid (shells/faces/edges follow file order via `parse_list_refs`),
  but the returned `Vec<SolidId>` order across top-level solids is nondeterministic because
  brep ids are collected by iterating a `HashMap`.

## 2. IGES limitations

`rg -n 'pub fn read_iges' crates/io/src/iges/reader.rs` and
`rg -n 'pub fn write_iges' crates/io/src/iges/writer.rs`

- Writer: emits Plane, NURBS surface (entity 128), NURBS curve (126), and lines. Analytic
  Cylinder/Cone/Sphere/Torus are written only as a placeholder comment (dropped). Circle
  edges are written as a 32-segment polyline approximation.
- Reader: reconstructs only entity 108 (planes), building a fixed unit square face
  (0.5 half-extent, spanning -0.5..+0.5 in u and v) per plane "for visualization."
  Entities 110/126/128 are not reconstructed.
- Net: IGES round-trip does not preserve real geometry. Treat it as write-partial /
  read-stub. Use STEP or the arena format when fidelity matters.

## 3. Mesh import path

`rg -n 'pub fn import_mesh' crates/io/src/stl/import.rs`

All five mesh `*_solid` readers (STL, 3MF, OBJ, PLY, GLB) delegate to `import_mesh(topo,
&mesh, tolerance) -> SolidId`. It welds vertices by tolerance and auto-flips winding: for a
closed mesh, outward triangles give positive signed volume (divergence theorem); a negative
raw signed volume flips the whole mesh. If per-triangle normals are present they take
precedence, else the signed-volume heuristic applies. Output is a solid of planar-triangle
faces. This is faithful faceting, never analytic.

## 4. Heal after import

`rg -n 'pub fn heal_solid|pub fn repair_solid|pub fn convert_to_elementary' crates/operations/src/heal.rs`

- `heal_solid` / `repair_solid`: gaps, orientation, degenerate edges, coincident vertices,
  wire spurs.
- `convert_to_elementary(topo, solid, tolerance)`: wraps
  `brepkit_heal::custom::convert_to_elementary::{convert_to_elementary, convert_edges_to_elementary}`.
  Its doc: the inverse of convert-to-bspline; STEP/IGES imports that came in as NURBS
  (from CAD systems that export everything as B-splines) are normalized back into analytic
  forms. Two passes (surfaces then edges), non-transactional (may leave a partially
  converted state; checkpoint first if you need atomicity). Hyperbola and Parabola are
  recognized but have no `EdgeCurve` variant, so they stay NURBS.
- Lower-level recognition (used by `convert_to_elementary`, callable directly):
  `recognize_surface` / `recognize_curve` in `crates/geometry/src/convert/`
  (`rg -n 'pub fn recognize_surface' crates/geometry/src/convert/recognize_surface.rs`,
  same for `recognize_curve.rs`). Tolerance-driven; can reconstruct slightly different
  parameters than the original (overfit risk).
- `brepkit_heal::upgrade::unify_same_domain::unify_same_domain(topo, solid, &UnifyOptions)`:
  merge coplanar / co-cylindrical adjacent faces (wasm `unifyFaces`).

## 5. Fixture faithfulness and the arena escape hatch

Priority order when capturing a fixture from a real failing case:

1. **STEP** if the operands round-trip analytic. Confirm surface *type* survives
   (section 6). If the part came in as NURBS, the fixture is unfaithful: you would debug a
   different solid, and `convert_to_elementary` may overfit slightly different numbers.
2. **Arena binary dump** when STEP goes lossy or the bug is tied to specific in-memory
   ids. Lossless, no weld, no recognition:
   `rg -n 'pub fn serialize_solid|pub fn deserialize_solid' crates/io/src/arena_io.rs`;
   wasm `serializeSolid` / `deserializeSolid`.

## 6. Round-trip / conformance tests

`rg -n 'fn step_roundtrip|fn step_output_has|fn stl_' crates/io/tests/cross_format.rs`

- `step_roundtrip_cylinder_surface_type`: write a cylinder, `read_step`, assert a face
  `matches!(.., FaceSurface::Cylinder(_))`. The template for "surface type survived."
- `step_roundtrip_nurbs_edge_survives`, `step_roundtrip_preserves_vertex_positions`,
  `step_roundtrip_box_volume`, `step_roundtrip_center_of_mass`,
  `step_roundtrip_multiple_solids`: geometry-value round-trips.
- `step_output_has_valid_syntax`: checks the `ISO-10303-21;` / `HEADER;` / `DATA;` /
  `END-ISO-10303-21;` envelope.
- `stl_import_roundtrip_preserves_aabb`, `stl_roundtrip_preserves_triangle_count`: mesh
  formats assert AABB / triangle count within tolerance, not surface type.
- In-crate reader tests: `roundtrip_cylinder_preserves_surface`,
  `roundtrip_circle_edge_preserved`, `roundtrip_nurbs_*` in
  `crates/io/src/step/reader.rs` test module.

## 7. Writer output shape

`rg -n 'ISO-10303-21|FILE_SCHEMA|SI_UNIT|fmt_f64' crates/io/src/step/writer.rs`

Header envelope: `ISO-10303-21;` / `HEADER;` /
`FILE_DESCRIPTION(('brepkit STEP export'), '2;1');` / `FILE_NAME(..)` /
`FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'));` / `DATA;` .. `ENDSEC;` / `END-ISO-10303-21;`.
Units: length `SI_UNIT(.MILLI.,.METRE.)`, angle `SI_UNIT($,.RADIAN.)`, plus a solid-angle
unit. Product structure via `ADVANCED_BREP_SHAPE_REPRESENTATION` +
`PRODUCT_DEFINITION_SHAPE` + `SHAPE_DEFINITION_REPRESENTATION`. Surface/curve coverage is
exhaustive (Plane/Cylinder/Cone/Sphere/Torus/Nurbs and Line/Circle/Ellipse/NurbsCurve all
map to real STEP entities). Floats via `fmt_f64` = `{:.15E}` with a `<1e-15 -> "0."` guard.
`write_step` takes `&[SolidId]` and errors `InvalidTopology` on an empty slice.

## 8. wasm binding names

`rg -n 'js_name' crates/wasm/src/bindings/io.rs`

Import/export methods on `BrepKernel`: `exportStep`/`importStep`, `exportIges`/`importIges`,
`exportStl`/`exportStlAscii`/`importStl`, `export3mf`/`import3mf`, `exportObj`/`importObj`,
`exportGlb`/`importGlb`, `exportPly`, `importIndexedMesh`,
`serializeSolid`/`deserializeSolid`. The whole file is `#![cfg(feature = "io")]`; `io` is a
default feature (`crates/wasm/Cargo.toml`). None of these appear in `bindings/batch.rs`, so
I/O ops are direct methods only, not part of `executeBatch`.

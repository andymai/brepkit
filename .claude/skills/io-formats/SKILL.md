---
name: io-formats
description: Use when a STEP or other file imports wrong (missing solids, all-NURBS where analytic was expected, hard UnsupportedEntity errors), when verifying writer output round-trips, when capturing a faithful fixture from a real file, or when adding or extending a format in crates/io. Covers the format matrix, STEP reader anatomy, the analytic round-trip check, and heal-after-import.
---

# I/O Formats

The `brepkit-io` crate (L3) reads and writes STEP, IGES, STL, 3MF, OBJ, PLY, glTF,
and a native binary. Treat it as a product surface: a real file that imports wrong
poisons every downstream skill, because most debugging fixtures come from importing
a file. The reader must be trustworthy first.

## When to use

- A STEP import drops solids, errors on an entity, or returns NURBS where you expected
  cylinders/planes.
- Proving a written file re-reads with its geometry intact (writer conformance).
- Capturing a fixture from a real file and confirming it is faithful (analytic, not lossy).
- Adding or extending a format (deltas over CLAUDE.md Recipe 2 are here).

## Format matrix

`find crates/io/src -type f` for the modules. Signatures verified against current source.

| Format | Kind | Analytic fidelity | Reader entry | Writer entry |
|--------|------|-------------------|--------------|--------------|
| STEP | B-Rep | Faithful: all 6 `FaceSurface` + 4 `EdgeCurve` variants round-trip | `step::reader::read_step(&str, &mut Topology) -> Vec<SolidId>` | `step::write_step(&Topology, &[SolidId]) -> String` |
| IGES | B-Rep in name only | Not trustworthy (see below) | `iges::reader::read_iges(&str, &mut Topology) -> Vec<SolidId>` | `iges::writer::write_iges(&Topology, &[SolidId]) -> String` |
| STL | Mesh-only by design | n/a (faceted) | `stl::read_stl(&[u8]) -> TriangleMesh`; `stl::read_stl_solid(topo, data, tol)` | `stl::write_stl(topo, solids, deflection, StlFormat)` |
| 3MF | Mesh-only by design | n/a | `threemf::read_threemf(data)`; `read_threemf_solid(topo, data, tol)` | `threemf::write_threemf(topo, solids, deflection)` |
| OBJ | Mesh-only by design | n/a | `obj::read_obj(&str)`; `read_obj_solid(topo, input, tol)` | `obj::write_obj(topo, solids, deflection)` |
| PLY | Mesh-only by design | n/a | `ply::read_ply(data)`; `read_ply_solid(topo, data, tol)` | `ply::write_ply(topo, solids, deflection, PlyFormat)` |
| glTF (GLB) | Mesh-only by design | n/a | `gltf::read_glb(data)`; `read_glb_solid(topo, data, tol)` | `gltf::write_glb(topo, solids, deflection)` |
| arena (native) | B-Rep, lossless binary | Faithful (full arena dump) | `arena_io::deserialize_solid(bytes, topo)` | `arena_io::serialize_solid(topo, solid_id)` |

**Mesh-only is by design, not a bug.** All five mesh `*_solid` readers delegate to
`stl::import::import_mesh` (`crates/io/src/stl/import.rs`), which welds vertices by
tolerance and auto-flips winding by signed volume. The result is a solid whose every
face is a planar triangle. A high-face-count all-planar solid out of STL/OBJ/PLY/3MF/GLB
is correct output. Do not file it as analytic degradation.

**IGES is write-partial / read-stub.** The writer emits planes, NURBS, and lines but
drops Cylinder/Cone/Sphere/Torus (placeholder comment) and approximates Circle edges as
a 32-segment polyline. The reader reconstructs only entity 108 planes, and builds a
fixed unit square face (0.5 half-extent) per plane "for visualization." IGES round-trips do not
preserve real geometry. Use STEP or the arena format for anything load-bearing.

`StlFormat` has variants `Binary` and `Ascii` (`rg -n 'enum StlFormat'
crates/io/src/stl/writer.rs`). `PlyFormat` has variants `Ascii` and
`BinaryLittleEndian` (not `Binary`) (`rg -n 'enum PlyFormat'
crates/io/src/ply/writer.rs`). Errors
are one enum `IoError` in `crates/io/src/lib.rs` (`ParseError`, `UnsupportedEntity`,
`InvalidTopology`, plus transparent `Topology`/`Operations`/`Io`/`Zip`).

## STEP reader in one screen

Full anatomy in [reference.md](reference.md) section 1. The load-bearing facts:

- **Only `MANIFOLD_SOLID_BREP` becomes a solid** (`rg -n 'MANIFOLD_SOLID_BREP'
  crates/io/src/step/reader.rs`). `SHELL_BASED_SURFACE_MODEL`, `GEOMETRIC_SET`,
  `BREP_WITH_VOIDS`, wireframe reps are silently dropped: the #1 "STEP dropped my solids"
  cause. Fewer solids returned, no error.
- **Analytic surfaces round-trip as their exact type.** `PLANE`, `CYLINDRICAL_SURFACE`,
  `CONICAL_SURFACE`, `SPHERICAL_SURFACE`, `TOROIDAL_SURFACE` map to the matching
  `FaceSurface`; `LINE`/`CIRCLE`/`ELLIPSE` to `EdgeCurve`; `B_SPLINE_*` to NURBS.
- **Unsupported entity = hard error, whole import fails.** `build_surface` and
  `build_curve_geometry` end in `_ => Err(IoError::UnsupportedEntity {..})`. One
  `SURFACE_OF_REVOLUTION` or offset surface aborts the entire file. No best-effort mode.
- **No pcurves, no units.** Trimming is topological (vertex endpoints + edge-loop order);
  pcurve/seam records are ignored and recomputed downstream. Unit context is not parsed, so
  an inches-authored file imports at wrong scale silently.
- **Top-level solid order is nondeterministic** (brep ids from HashMap iteration). Sort if
  you depend on order. The reader does **no healing**: parse and build only.

## Procedure: "this STEP imports wrong"

Goal is to separate three failure classes: reader bug/gap, unhealed geometry, genuinely
bad file. Checkpoints in brackets.

1. **Read and census.** `let solids = step::reader::read_step(&s, &mut topo)?;`
   [If `solids.len()` is short of the file's `MANIFOLD_SOLID_BREP` count, the rest were
   dropped reps: reader gap, not your geometry.] Census surface types by walking
   `brepkit_topology::explorer::solid_faces(&topo, sid)?` and matching `face.surface()`
   (or the `type_tag` / `is_analytic` delegates from `math/src/traits.rs`).
   [All-NURBS where you expected analytic means either a lossy source file or the file
   genuinely stored B-splines; go to step 5.]
2. **Validate.** `brepkit_operations::validate::validate_solid(&topo, sid)?`; check
   `report.is_valid()`. [Errors here with surfaces that imported fine means unhealed
   geometry, go to step 6.]
3. **Tessellate and watertight-check.** `brepkit_operations::tessellate::tessellate_solid`
   then `is_watertight` / `boundary_edge_count`. See the solid-verification and
   tessellation siblings.
4. **Classify.** Use ray-cast `brepkit_check::classify::classify_point`
   (`crates/check/src/classify/mod.rs`). [Never trust `classify_point_winding` /
   `classify_point_robust` on faceted or NURBS solids; that mistake has cost prior
   debuggers a wrong theory.]
5. **Classify the failure:**
   - *Reader bug/gap*: `IoError::UnsupportedEntity { entity }` (grep the name against the
     `build_surface` / `build_curve_geometry` arms), or fewer solids than the file's brep
     count.
   - *Unhealed geometry*: imports, but `validate_solid` flags gaps/orientation, or
     surfaces came in as NURBS. Heal after import (step 6).
   - *Genuinely bad file*: open it in an external STEP viewer; if malformed there too,
     it is the file.
6. **Heal after import** (verified names, `crates/operations/src/heal.rs`):
   - `heal_solid` / `repair_solid`: gaps, orientation, degenerate edges, coincident
     vertices, wire spurs.
   - `convert_to_elementary(topo, solid, tolerance)`: analytic recovery, the inverse of
     convert-to-bspline. Recovers Cylinder/Cone/Sphere/Torus/Plane and Circle/Line/Ellipse
     from NURBS that were exported by systems that emit everything as B-splines. Caveat:
     recognition is tolerance-driven and can reconstruct slightly different parameters than
     the original (overfit risk), and it is non-transactional (checkpoint first if you need
     atomicity).
   - `brepkit_heal::upgrade::unify_same_domain::unify_same_domain`: merge coplanar /
     co-cylindrical adjacent faces (wasm `unifyFaces`).

## The analytic round-trip check (why the reader must be trustworthy)

Write, read back, assert the surface/curve *type* survived. Canonical test:
`step_roundtrip_cylinder_surface_type` in `crates/io/tests/cross_format.rs` writes a
cylinder, `read_step`s it, asserts a face `matches!(.., FaceSurface::Cylinder(_))`.

Why it matters: fixtures for every other skill lean on the reader. The workflow is build a
case in the brepjs tool, `exportSTEP(operand)` to a file, `read_step` it, debug the real
geometry. **Before trusting a STEP fixture, confirm the operands round-trip analytic
(CIRCLE/CYLINDER, not NURBS).** An all-NURBS import of an analytic part means you are
debugging a different solid than the one that failed. If a file stores an analytic part as
B-splines, `convert_to_elementary` (or `recognize_surface` / `recognize_curve` in
`crates/geometry/src/convert/`) recovers it, but can overfit slightly different numbers.
When a STEP fixture will not preserve the failing state at all (in-memory-id-specific bugs),
escalate to the lossless arena dump (`serializeSolid` / `deserializeSolid` in wasm;
`arena_io::serialize_solid` / `deserialize_solid` in Rust).

## Writer conformance basics

The STEP writer (`crates/io/src/step/writer.rs`) emits ISO-10303-21 with
`FILE_SCHEMA(('CONFIG_CONTROL_DESIGN'))`, length unit `SI_UNIT(.MILLI.,.METRE.)`
(millimetres), angle unit radians. Surface/curve coverage is exhaustive: every
`FaceSurface` and `EdgeCurve` variant has a real STEP entity (no drops, unlike IGES).
The writer takes `&[SolidId]` (multi-solid) and errors `InvalidTopology` on an empty slice.

Sanity-check output two ways: (a) re-read it and run the type + volume asserts above
(pattern `step_output_has_valid_syntax` checks the `ISO-10303-21;` / `HEADER;` / `DATA;` /
`END-ISO-10303-21;` envelope); (b) open it in an external STEP viewer.

## Adding a format: deltas over CLAUDE.md Recipe 2

Recipe 2 covers the module scaffold, `lib.rs` `pub mod`, and signatures. It omits or gets
wrong:

- **wasm bindings live in `bindings/io.rs`, not `kernel.rs`.** The whole file is
  `#![cfg(feature = "io")]` and `io` is a **default** feature; bindings are auto-gated, add
  no new cfg.
- **Skip batch dispatch.** `rg -in 'step|import|export'
  crates/wasm/src/bindings/batch.rs` has no hits; I/O ops are direct `BrepKernel` methods.
- **Round-trip / golden tests go in `crates/io/tests/`** (crate-level), model on
  `cross_format.rs`; golden data in `crates/io/tests/data/`.
- **Mesh reader convention:** expose both `read_<fmt>(data) -> TriangleMesh` and
  `read_<fmt>_solid(topo, data, tolerance) -> SolidId` (delegates to `import_mesh`).

## Pitfalls: what NOT to conclude

| Observation | Wrong conclusion | Reality |
|-------------|------------------|---------|
| Import returned fewer solids than the file has | Reader lost geometry | Only `MANIFOLD_SOLID_BREP` becomes a solid; other reps are dropped silently. Grep the file for its solid rep type. |
| OBJ/STL import is all planar triangles | Import degraded analytic surfaces | Mesh formats carry no analytic data; a faceted solid is correct output. |
| STEP surfaces all came in as NURBS | Reader bug | The file likely stored B-splines. Confirm in a viewer, then `convert_to_elementary`. |
| `UnsupportedEntity` error | The file is corrupt | The reader hit a surface/curve type with no arm. It is a reader gap; the file may be fine. |
| Fixture volume matches, so the fixture is faithful | Fixture reproduces the bug | If it round-tripped to NURBS you are debugging a different solid. Assert the surface *type*, or use the arena dump. |
| IGES import produced a shape | IGES import works | The reader only makes fixed unit squares (0.5 half-extent) from entity-108 planes. Use STEP or arena. |
| Multi-solid order is stable in one run | Order is deterministic | Top-level order comes from HashMap iteration; sort if you depend on it. |
| Coordinates look wrong scale | Reader has a bug | The reader ignores STEP unit context; a non-mm file imports unscaled. |

## Sibling skills

- **solid-verification**: the measure / validate / watertight / classify bar in depth.
- **analytic-preservation**: keeping and recovering typed analytic surfaces; the census.
- **tessellation**: watertightness and periodic-band meshing after import.
- **testing** and **parity-benchmarking**: capturing faithful fixtures (STEP or arena)
  from a real tool scenario.
- **wasm-bindings**: exposing a new format's import/export on `BrepKernel`.

## Reference

STEP entity-to-topology mapping, the exact match arms, and the healing-pass catalog:
[reference.md](reference.md).

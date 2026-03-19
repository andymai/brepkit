//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! The primary pipeline (`boolean_pipeline`) operates in 2D parameter space,
//! preserving all surface types (plane, cylinder, cone, sphere, torus,
//! NURBS) through the boolean. Falls back to the analytic path for
//! cases pipeline can't handle, then to mesh boolean as a last resort.

mod analytic;
mod assembly;
pub mod boolean_pipeline;
mod classify;
pub(crate) mod classify_2d;
mod compound;
pub(crate) mod face_splitter;
mod fragments;
mod intersect;
pub(crate) mod pcurve_compute;
pub(crate) mod pipeline;
pub(crate) mod plane_frame;
mod precompute;
mod split;
mod types;
pub(crate) mod wire_builder;
use analytic::{analytic_boolean, collect_face_signatures, has_torus, is_all_analytic};
use assembly::validate_boolean_result;
pub(crate) use assembly::{assemble_solid, assemble_solid_mixed};
pub use compound::compound_cut;
pub use precompute::face_polygon;
pub use types::{BooleanOp, BooleanOptions, FaceSpec};

// ---------------------------------------------------------------------------
// GFA pipeline entry point
// ---------------------------------------------------------------------------

/// Boolean via the GFA pipeline, with fallback to the existing pipeline.
///
/// Tries the GFA first; if it produces an empty result or fails,
/// falls back to the existing pipeline.
///
/// # Errors
///
/// Returns an error if both the GFA and fallback pipelines fail.
pub fn boolean_gfa(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    let faces_a = brepkit_topology::explorer::solid_faces(topo, a)
        .map(|f| f.len())
        .unwrap_or(0);
    let faces_b = brepkit_topology::explorer::solid_faces(topo, b)
        .map(|f| f.len())
        .unwrap_or(0);

    let algo_op = match op {
        BooleanOp::Fuse => brepkit_algo::bop::BooleanOp::Fuse,
        BooleanOp::Cut => brepkit_algo::bop::BooleanOp::Cut,
        BooleanOp::Intersect => brepkit_algo::bop::BooleanOp::Intersect,
    };

    // Safety: GFA may allocate new entities in the topology arena, but
    // arena allocation is append-only — the original solid IDs `a` and `b`
    // remain valid. The fallback pipeline only reads original faces via
    // `solid_faces(topo, a/b)`, which is unaffected by new allocations.
    let gfa_start = timer_now();
    match brepkit_algo::gfa::boolean(topo, algo_op, a, b) {
        Ok(result) => {
            // Validate: result must have faces
            let result_faces = brepkit_topology::explorer::solid_faces(topo, result)
                .map(|f| f.len())
                .unwrap_or(0);
            if result_faces == 0 {
                log::warn!(
                    "GFA produced empty solid in {:.1}ms, falling back",
                    timer_elapsed_ms(gfa_start)
                );
                return boolean(topo, op, a, b);
            }
            log::info!(
                "GFA boolean succeeded in {:.1}ms (faces: {faces_a}+{faces_b} → {result_faces})",
                timer_elapsed_ms(gfa_start)
            );
            Ok(result)
        }
        Err(e) => {
            log::warn!(
                "GFA boolean failed in {:.1}ms ({e}), falling back",
                timer_elapsed_ms(gfa_start)
            );
            boolean(topo, op, a, b)
        }
    }
}

// WASM-compatible timer: `std::time::Instant` panics on wasm32 targets.
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn timer_now() -> std::time::Instant {
    std::time::Instant::now()
}
#[cfg(not(target_arch = "wasm32"))]
pub(super) fn timer_elapsed_ms(t: std::time::Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
#[cfg(target_arch = "wasm32")]
pub(super) fn timer_now() -> () {}
#[cfg(target_arch = "wasm32")]
pub(super) fn timer_elapsed_ms(_t: ()) -> f64 {
    0.0
}

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation on two solids.
///
/// When both solids are composed entirely of analytic faces (planes,
/// cylinders, cones, spheres), uses an exact analytic path that preserves
/// surface types through the boolean. Falls back to the tessellated path
/// for NURBS faces or analytic-analytic face pairs.
///
/// # Errors
///
/// Returns an error if either solid contains NURBS faces, or if the operation
/// produces an empty or non-manifold result.
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    boolean_with_options(topo, op, a, b, BooleanOptions::default())
}

/// Perform a boolean operation with custom options.
///
/// See [`boolean`] for details. The `opts` parameter allows configuring
/// tessellation quality for non-planar faces.
///
/// # Errors
///
/// Returns an error if either solid is invalid or the operation produces
/// an empty or non-manifold result.
#[allow(clippy::too_many_lines)]
pub fn boolean_with_options(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let tol = opts.tolerance;

    log::debug!(
        "boolean {op:?}: solids ({}, {}), deflection={}",
        a.index(),
        b.index(),
        opts.deflection,
    );

    // ── Try analytic fast path ──────────────────────────────────────
    // Optimized for non-coplanar all-analytic solids (box-cylinder, etc.).
    // This is the most validated and fastest path for the common case.
    let try_analytic = {
        let both_analytic = is_all_analytic(topo, a)? && is_all_analytic(topo, b)?;
        let no_torus = !has_torus(topo, a)? && !has_torus(topo, b)?;
        both_analytic && no_torus
    };
    // Detect when BOTH inputs have complex topology (shelled + hollow).
    // The analytic assembly creates non-manifold edges when fusing a
    // shelled box (inner wires on rim face) with a hollow solid (reversed
    // curved faces from boolean cut, e.g., lip frustum).
    // Skip analytic only when BOTH conditions are present simultaneously.
    let both_complex = {
        // Detect polygonal inner wires (shell rim) vs circular (boolean hole).
        let has_polygonal_inner_wire = |solid: SolidId| -> Result<bool, crate::OperationsError> {
            let s = topo.solid(solid)?;
            let sh = topo.shell(s.outer_shell())?;
            for &fid in sh.faces() {
                for &iw_id in topo.face(fid)?.inner_wires() {
                    let iw = topo.wire(iw_id)?;
                    let all_lines = iw.edges().iter().all(|oe| {
                        topo.edge(oe.edge()).is_ok_and(|e| {
                            matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line)
                        })
                    });
                    if all_lines && iw.edges().len() >= 3 {
                        return Ok(true);
                    }
                }
            }
            Ok(false)
        };
        let has_reversed_curved = |solid: SolidId| -> Result<bool, crate::OperationsError> {
            let s = topo.solid(solid)?;
            let sh = topo.shell(s.outer_shell())?;
            for &fid in sh.faces() {
                let f = topo.face(fid)?;
                if f.is_reversed()
                    && !matches!(
                        f.surface(),
                        brepkit_topology::face::FaceSurface::Plane { .. }
                    )
                {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        // Check if one input is shelled and the other is hollow, or either is both.
        let a_wires = has_polygonal_inner_wire(a)?;
        let b_wires = has_polygonal_inner_wire(b)?;
        let a_curved = has_reversed_curved(a)?;
        let b_curved = has_reversed_curved(b)?;
        // Trigger when one input has reversed curved faces (hollow frustum)
        // and EITHER the other has polygonal inner wires (shelled)
        // OR the other has many faces (complex result from prior booleans).
        let a_faces = topo.shell(topo.solid(a)?.outer_shell())?.faces().len();
        let b_faces = topo.shell(topo.solid(b)?.outer_shell())?.faces().len();
        // A solid is "structurally complex" if it has polygonal inner wires
        // (shelled) OR reversed curved faces (from boolean cut of hollow shapes).
        // "Large" means many faces (from prior booleans or mesh tessellation).
        let a_structural = a_wires || a_curved;
        let b_structural = b_wires || b_curved;
        let a_large = a_faces > 6;
        let b_large = b_faces > 6;
        // Skip analytic when one input is structurally complex AND the other
        // is large (from prior booleans). This catches lip fuse (hollow lip +
        // large shelled-box result) without catching simple frustum cuts.
        (a_structural && b_large) || (b_structural && a_large)
    };

    let mut analytic_fallback: Option<SolidId> = None;

    if try_analytic && !both_complex {
        if let Ok(solid) = analytic_boolean(topo, op, a, b, tol, opts.deflection) {
            let _ = crate::heal::remove_degenerate_edges(topo, solid, tol.linear)?;
            if opts.unify_faces {
                let _ = crate::heal::unify_faces(topo, solid)?;
            }
            // Check manifold. Only reject if there are MANY non-manifold edges
            // (>3) indicating a systematic assembly issue (e.g., shelled solid).
            // Minor non-manifold (1-3 edges) from sphere/cone booleans is acceptable.
            let nm_count = {
                let sh = topo.shell(topo.solid(solid)?.outer_shell())?;
                let mut efc: std::collections::HashMap<usize, u32> =
                    std::collections::HashMap::new();
                for &fid in sh.faces() {
                    if let Ok(face) = topo.face(fid) {
                        for wid in std::iter::once(face.outer_wire())
                            .chain(face.inner_wires().iter().copied())
                        {
                            if let Ok(wire) = topo.wire(wid) {
                                for oe in wire.edges() {
                                    *efc.entry(oe.edge().index()).or_default() += 1;
                                }
                            }
                        }
                    }
                }
                efc.values().filter(|&&c| c > 2).count()
            };
            let is_manifold = nm_count <= 30;

            if is_manifold && validate_boolean_result(topo, solid).is_ok() {
                let solid = enforce_manifold_shell(topo, solid).unwrap_or(solid);
                return Ok(solid);
            }
            analytic_fallback = Some(solid);
        }
    }

    // ── Containment shortcut ─────────────────────────────────────────
    // Detect A⊂B or B⊂A (including A=B) and handle directly.
    // This catches identical solids that neither analytic nor pipeline handle
    // correctly (boundary-coincident intersections produce degenerate splits).
    {
        use classify::try_build_analytic_classifier;
        let ca = try_build_analytic_classifier(topo, a);
        let cb = try_build_analytic_classifier(topo, b);
        // Sample the AABB centroid of the solid — guaranteed to be in the
        // interior for convex solids (which all analytic primitives are).
        let sample_aabb_center = |topo: &Topology, solid: SolidId| -> Option<Point3> {
            let s = topo.solid(solid).ok()?;
            let sh = topo.shell(s.outer_shell()).ok()?;
            let mut min = Point3::new(f64::MAX, f64::MAX, f64::MAX);
            let mut max = Point3::new(f64::MIN, f64::MIN, f64::MIN);
            for &fid in sh.faces() {
                let f = topo.face(fid).ok()?;
                let w = topo.wire(f.outer_wire()).ok()?;
                for oe in w.edges() {
                    let e = topo.edge(oe.edge()).ok()?;
                    let p = topo.vertex(e.start()).ok()?.point();
                    min = Point3::new(min.x().min(p.x()), min.y().min(p.y()), min.z().min(p.z()));
                    max = Point3::new(max.x().max(p.x()), max.y().max(p.y()), max.z().max(p.z()));
                }
            }
            Some(Point3::new(
                (min.x() + max.x()) * 0.5,
                (min.y() + max.y()) * 0.5,
                (min.z() + max.z()) * 0.5,
            ))
        };
        let b_in_a = ca
            .as_ref()
            .zip(sample_aabb_center(topo, b))
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(types::FaceClass::Inside);
        let a_in_b = cb
            .as_ref()
            .zip(sample_aabb_center(topo, a))
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(types::FaceClass::Inside);
        if b_in_a || a_in_b {
            return match (op, b_in_a, a_in_b) {
                (BooleanOp::Fuse, true, _) => Ok(crate::copy::copy_solid(topo, a)?),
                (BooleanOp::Fuse, _, true) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Cut, true, _) => Err(crate::OperationsError::InvalidInput {
                    reason: "boolean Cut: B is inside A — result would have a void".into(),
                }),
                (BooleanOp::Cut, _, true) => Err(crate::OperationsError::InvalidInput {
                    reason: "boolean Cut: A is inside B — result is empty".into(),
                }),
                (BooleanOp::Intersect, true, _) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Intersect, _, true) => Ok(crate::copy::copy_solid(topo, a)?),
                _ => Err(crate::OperationsError::InvalidInput {
                    reason: "containment shortcut: unexpected state".into(),
                }),
            };
        }
    }

    // ── Try pipeline parameter-space pipeline ────────────────────────────────
    // pipeline handles coplanar faces, NURBS, and other cases analytic can't.
    // Preserves all surface types through the boolean.
    match boolean_pipeline::boolean_pipeline(topo, op, a, b) {
        Ok(solid) if validate_boolean_result(topo, solid).is_ok() => {
            let solid = enforce_manifold_shell(topo, solid).unwrap_or(solid);
            return Ok(solid);
        }
        Ok(_) => {
            log::debug!("boolean {op:?}: pipeline result failed validation, falling back");
        }
        Err(e) => {
            log::debug!("boolean {op:?}: pipeline failed ({e}), falling back");
        }
    }

    // ── Mesh boolean (last resort) ───────────────────────────────────
    // When both analytic and pipeline fail, fall back to mesh co-refinement.
    // All surface types are lost — output is planar triangles only.
    match mesh_boolean_fallback(topo, op, a, b, opts.deflection, tol, &opts) {
        Ok(result) => {
            let result = enforce_manifold_shell(topo, result).unwrap_or(result);
            return Ok(result);
        }
        Err(e) => {
            log::debug!("boolean {op:?}: mesh boolean also failed ({e})");
        }
    }

    // If the analytic path produced a result (even with non-manifold edges),
    // try to fix it with greedy flood-fill shell splitting before returning.
    if let Some(solid) = analytic_fallback {
        let fixed = enforce_manifold_shell(topo, solid).unwrap_or(solid);
        return Ok(fixed);
    }

    Err(crate::OperationsError::InvalidInput {
        reason: format!("boolean {op:?}: all paths failed (analytic, pipeline, mesh boolean)"),
    })
}

// ---------------------------------------------------------------------------
// Evolution-tracking wrapper
// ---------------------------------------------------------------------------

/// Perform a boolean operation and return an [`EvolutionMap`] tracking face
/// provenance.
///
/// This wraps [`boolean`] and uses a heuristic (normal + centroid similarity)
/// to match output faces back to their input faces. Faces whose best match
/// score exceeds the similarity threshold are classified as "modified";
/// unmatched input faces are classified as "deleted".
///
/// # Errors
///
/// Returns the same errors as [`boolean`].
pub fn boolean_with_evolution(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<(SolidId, crate::evolution::EvolutionMap), crate::OperationsError> {
    use crate::evolution::EvolutionMap;

    // Collect input face normals + centroids before the operation mutates topology.
    let input_faces_a = collect_face_signatures(topo, a)?;
    let input_faces_b = collect_face_signatures(topo, b)?;

    let mut input_faces: Vec<(usize, Vec3, Point3)> =
        Vec::with_capacity(input_faces_a.len() + input_faces_b.len());
    input_faces.extend(input_faces_a);
    input_faces.extend(input_faces_b);

    // Run the actual boolean.
    let result = boolean(topo, op, a, b)?;

    // Collect output face normals + centroids.
    let output_faces = collect_face_signatures(topo, result)?;

    // Build evolution map via heuristic matching.
    let mut evo = EvolutionMap::new();
    let mut matched_inputs: std::collections::HashSet<usize> = std::collections::HashSet::new();
    let mut unmatched_outputs: Vec<(usize, Vec3, Point3)> = Vec::new();

    // Normal dot threshold: cos(45deg) — relaxed to handle faces split by
    // booleans where normals may shift slightly.
    let normal_threshold = 0.707;
    // Maximum centroid distance squared for a match (generous).
    let centroid_dist_sq_max = 100.0;

    for &(out_idx, out_normal, out_centroid) in &output_faces {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_input: Option<usize> = None;

        for &(in_idx, in_normal, in_centroid) in &input_faces {
            let dot = out_normal.dot(in_normal);
            if dot < normal_threshold {
                continue;
            }

            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));

            if dist_sq > centroid_dist_sq_max {
                continue;
            }

            // Score: higher normal alignment + closer centroid = better.
            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_modified(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        } else {
            unmatched_outputs.push((out_idx, out_normal, out_centroid));
        }
    }

    // Unmatched output faces are "generated" — attribute them to the nearest
    // input face (the face most likely responsible for generating them, e.g.
    // intersection curves create new faces near the boundary).
    for &(out_idx, _out_normal, out_centroid) in &unmatched_outputs {
        let mut best_dist_sq = f64::MAX;
        let mut best_input: Option<usize> = None;

        for &(in_idx, _, in_centroid) in &input_faces {
            let dx = out_centroid.x() - in_centroid.x();
            let dy = out_centroid.y() - in_centroid.y();
            let dz = out_centroid.z() - in_centroid.z();
            let dist_sq = dx.mul_add(dx, dy.mul_add(dy, dz * dz));
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_input = Some(in_idx);
            }
        }

        if let Some(in_idx) = best_input {
            evo.add_generated(in_idx, out_idx);
            matched_inputs.insert(in_idx);
        }
    }

    // Any input face not matched to any output is deleted.
    for &(in_idx, _, _) in &input_faces {
        if !matched_inputs.contains(&in_idx) {
            evo.add_deleted(in_idx);
        }
    }

    Ok((result, evo))
}

// ---------------------------------------------------------------------------
// Mesh boolean helpers
// ---------------------------------------------------------------------------

/// Best-effort mesh boolean fallback for high face-count solids.
///
/// Tessellates both solids, runs mesh co-refinement, assembles the result,
/// and applies the same post-processing as the other boolean paths.
/// Returns `Err` on any failure so the caller can fall through to the
/// chord-based path.
fn mesh_boolean_fallback(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    deflection: f64,
    tol: brepkit_math::tolerance::Tolerance,
    opts: &BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let mesh_a = crate::tessellate::tessellate_solid(topo, a, deflection)?;
    let mesh_b = crate::tessellate::tessellate_solid(topo, b, deflection)?;

    let mb_result = crate::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, op, tol.linear)?;
    let face_specs = mesh_result_to_face_specs(&mb_result);
    if face_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "mesh boolean produced empty result".into(),
        });
    }
    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    if opts.unify_faces {
        let _ = crate::heal::unify_faces(topo, result)?;
    }
    if opts.heal_after_boolean {
        let _ = crate::heal::heal_solid(topo, result, tol.linear)?;
    }
    validate_boolean_result(topo, result)?;
    log::info!(
        "boolean {op:?}: mesh boolean path → solid {} ({} faces, surface types lost)",
        result.index(),
        face_specs.len()
    );
    Ok(result)
}

/// Convert a mesh boolean result into `FaceSpec` entries for solid assembly.
fn mesh_result_to_face_specs(result: &crate::mesh_boolean::MeshBooleanResult) -> Vec<FaceSpec> {
    let mut specs = Vec::new();
    for tri in result.mesh.indices.chunks_exact(3) {
        let v0 = result.mesh.positions[tri[0] as usize];
        let v1 = result.mesh.positions[tri[1] as usize];
        let v2 = result.mesh.positions[tri[2] as usize];

        let edge1 = v1 - v0;
        let edge2 = v2 - v0;
        let Ok(normal) = edge1.cross(edge2).normalize() else {
            continue;
        };
        let d = crate::dot_normal_point(normal, v0);
        specs.push(FaceSpec::Planar {
            vertices: vec![v0, v1, v2],
            normal,
            d,
            inner_wires: vec![],
        });
    }
    specs
}

/// Post-process a solid to enforce manifold topology via greedy flood-fill.
///
/// Detects non-manifold edges (shared by 3+ faces) and uses OCCT-style
/// greedy shell building to split the non-manifold shell into manifold
/// sub-shells. The largest sub-shell becomes the outer shell; smaller ones
/// become inner shells (cavities).
///
/// If the solid is already manifold, returns it unchanged.
#[allow(clippy::too_many_lines)]
fn enforce_manifold_shell(
    topo: &mut Topology,
    solid: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    use std::collections::{HashMap, HashSet, VecDeque};

    let shell_id = topo.solid(solid)?.outer_shell();
    let face_ids = topo.shell(shell_id)?.faces().to_vec();

    // Count edges per face.
    let mut edge_face_count: HashMap<usize, u32> = HashMap::new();
    for &fid in &face_ids {
        if let Ok(face) = topo.face(fid) {
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if let Ok(wire) = topo.wire(wid) {
                    for oe in wire.edges() {
                        *edge_face_count.entry(oe.edge().index()).or_default() += 1;
                    }
                }
            }
        }
    }

    // Only apply for significant non-manifold (>3 edges). Minor non-manifold
    // (1-3 edges) from sphere/cone intersections is tolerable and splitting
    // the shell at those edges breaks downstream operations (section, volume).
    let nm_count = edge_face_count.values().filter(|&&c| c > 2).count();
    if nm_count <= 3 {
        return Ok(solid);
    }

    log::debug!(
        "enforce_manifold_shell: {} non-manifold edges in {} faces",
        nm_count,
        face_ids.len()
    );

    // Build vertex-pair → face adjacency for neighbor discovery.
    let mut vpair_faces: HashMap<(usize, usize), Vec<brepkit_topology::face::FaceId>> =
        HashMap::new();
    for &fid in &face_ids {
        if let Ok(face) = topo.face(fid) {
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if let Ok(wire) = topo.wire(wid) {
                    for oe in wire.edges() {
                        if let Ok(e) = topo.edge(oe.edge()) {
                            let si = e.start().index();
                            let ei = e.end().index();
                            let key = if si <= ei { (si, ei) } else { (ei, si) };
                            vpair_faces.entry(key).or_default().push(fid);
                        }
                    }
                }
            }
        }
    }

    // Greedy flood-fill shell construction.
    let available: HashSet<brepkit_topology::face::FaceId> = face_ids.iter().copied().collect();
    let mut processed: HashSet<brepkit_topology::face::FaceId> = HashSet::new();
    let mut shells: Vec<Vec<brepkit_topology::face::FaceId>> = Vec::new();

    for &start_face in &face_ids {
        if processed.contains(&start_face) {
            continue;
        }

        let mut shell_faces = vec![start_face];
        processed.insert(start_face);

        // Track edge-ID usage within this shell.
        let mut shell_edge_count: HashMap<usize, u32> = HashMap::new();
        if let Ok(face) = topo.face(start_face) {
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if let Ok(wire) = topo.wire(wid) {
                    for oe in wire.edges() {
                        *shell_edge_count.entry(oe.edge().index()).or_default() += 1;
                    }
                }
            }
        }

        let mut queue = VecDeque::new();
        queue.push_back(start_face);

        while let Some(current) = queue.pop_front() {
            let Ok(face) = topo.face(current) else {
                continue;
            };
            // Collect (vpair, edge_id) from all wires.
            let mut all_edges = Vec::new();
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if let Ok(wire) = topo.wire(wid) {
                    for oe in wire.edges() {
                        if let Ok(e) = topo.edge(oe.edge()) {
                            let si = e.start().index();
                            let ei = e.end().index();
                            let key = if si <= ei { (si, ei) } else { (ei, si) };
                            all_edges.push((key, oe.edge()));
                        }
                    }
                }
            }

            for (vpair, edge_id) in all_edges {
                let eidx = edge_id.index();

                // Skip edges already manifold in this shell.
                if shell_edge_count.get(&eidx).copied().unwrap_or(0) >= 2 {
                    continue;
                }

                // Find candidate neighbor faces via vertex-pair.
                let candidates: Vec<brepkit_topology::face::FaceId> = vpair_faces
                    .get(&vpair)
                    .map(|fs| {
                        fs.iter()
                            .copied()
                            .filter(|&f| {
                                f != current && available.contains(&f) && !processed.contains(&f)
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                if candidates.is_empty() {
                    continue;
                }

                // Pick first candidate (simple heuristic — dihedral selection
                // would be better but requires surface normal evaluation).
                let selected = candidates[0];

                if processed.contains(&selected) {
                    continue;
                }

                processed.insert(selected);
                shell_faces.push(selected);
                queue.push_back(selected);

                // Update edge count.
                if let Ok(sel_face) = topo.face(selected) {
                    for wid in std::iter::once(sel_face.outer_wire())
                        .chain(sel_face.inner_wires().iter().copied())
                    {
                        if let Ok(wire) = topo.wire(wid) {
                            for sel_oe in wire.edges() {
                                *shell_edge_count.entry(sel_oe.edge().index()).or_default() += 1;
                            }
                        }
                    }
                }
            }
        }

        shells.push(shell_faces);
    }

    // Add any unprocessed faces to a final shell.
    let remaining: Vec<brepkit_topology::face::FaceId> = available
        .iter()
        .filter(|f| !processed.contains(f))
        .copied()
        .collect();
    if !remaining.is_empty() {
        shells.push(remaining);
    }

    if shells.len() <= 1 {
        // Single shell — nothing to split.
        return Ok(solid);
    }

    log::debug!(
        "enforce_manifold_shell: split into {} shells (sizes: {:?})",
        shells.len(),
        shells.iter().map(Vec::len).collect::<Vec<_>>(),
    );

    // Build the solid: largest shell is outer, rest are inner.
    let mut best_idx = 0;
    let mut best_count = 0;
    for (i, faces) in shells.iter().enumerate() {
        if faces.len() > best_count {
            best_count = faces.len();
            best_idx = i;
        }
    }

    let outer = brepkit_topology::shell::Shell::new(shells[best_idx].clone())
        .map_err(crate::OperationsError::Topology)?;
    let outer_id = topo.add_shell(outer);
    let mut inner_ids = Vec::new();
    for (i, faces) in shells.iter().enumerate() {
        if i != best_idx && !faces.is_empty() {
            if let Ok(inner) = brepkit_topology::shell::Shell::new(faces.clone()) {
                inner_ids.push(topo.add_shell(inner));
            }
        }
    }

    Ok(topo.add_solid(brepkit_topology::solid::Solid::new(outer_id, inner_ids)))
}

#[cfg(test)]
mod tests;

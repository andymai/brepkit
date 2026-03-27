//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! Uses the GFA pipeline (`brepkit_algo::gfa`) as the primary boolean engine,
//! with mesh boolean (co-refinement) as a fallback when GFA fails or produces
//! invalid results.

mod assembly;
mod classify;
mod types;
use assembly::validate_boolean_result;
pub(crate) use assembly::{assemble_solid, assemble_solid_mixed};
pub use types::{BooleanOp, BooleanOptions, FaceSpec};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation on two solids.
///
/// Uses the GFA pipeline as the primary engine, with mesh boolean
/// (co-refinement) as a fallback when GFA fails or produces invalid results.
///
/// # Errors
///
/// Returns an error if either solid is invalid or the operation produces
/// an empty or non-manifold result.
#[allow(clippy::too_many_lines)]
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    let tol = brepkit_math::tolerance::Tolerance::new();

    // ── Containment shortcut ─────────────────────────────────────────
    // Detect A⊂B or B⊂A (including A=B) and handle directly.
    // Only applies when BOTH solids have simple analytic classifiers.
    {
        use brepkit_algo::classifier::try_build_analytic_classifier;
        let ca = try_build_analytic_classifier(topo, a);
        let cb = try_build_analytic_classifier(topo, b);
        let sample_aabb = |topo: &Topology, solid: SolidId| -> Option<(Point3, Point3)> {
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
            Some((min, max))
        };
        let aabb_center = |min: &Point3, max: &Point3| -> Point3 {
            Point3::new(
                (min.x() + max.x()) * 0.5,
                (min.y() + max.y()) * 0.5,
                (min.z() + max.z()) * 0.5,
            )
        };
        let aabb_a = sample_aabb(topo, a);
        let aabb_b = sample_aabb(topo, b);
        let center_a = aabb_a.map(|(min, max)| aabb_center(&min, &max));
        let center_b = aabb_b.map(|(min, max)| aabb_center(&min, &max));
        // B⊂A requires: B's center is inside A AND A's center is outside B
        // (if A's center is also inside B, the solids overlap rather than one
        // containing the other).
        let b_center_in_a = ca
            .as_ref()
            .zip(center_b)
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(brepkit_algo::FaceClass::Inside);
        let a_center_in_b = cb
            .as_ref()
            .zip(center_a)
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(brepkit_algo::FaceClass::Inside);
        let a_center_outside_b = cb
            .as_ref()
            .zip(center_a)
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(brepkit_algo::FaceClass::Outside);
        let b_center_outside_a = ca
            .as_ref()
            .zip(center_b)
            .and_then(|(c, p)| c.classify(p, tol))
            == Some(brepkit_algo::FaceClass::Outside);
        // AABB containment fallback: when a classifier is unavailable (e.g.,
        // tessellated sphere), check if one AABB is strictly inside the other.
        // The inner AABB must fit within the outer's bounds (within tol.linear
        // margin), and the outer must be >10% larger in at least 2 dimensions
        // to avoid false positives on overlapping same-size solids.
        let aabb_contains =
            |inner: &Option<(Point3, Point3)>, outer: &Option<(Point3, Point3)>| -> bool {
                let Some(((i_min, i_max), (o_min, o_max))) = inner.zip(*outer) else {
                    return false;
                };
                let margin = tol.linear;
                let inside = i_min.x() >= o_min.x() - margin
                    && i_min.y() >= o_min.y() - margin
                    && i_min.z() >= o_min.z() - margin
                    && i_max.x() <= o_max.x() + margin
                    && i_max.y() <= o_max.y() + margin
                    && i_max.z() <= o_max.z() + margin;
                if !inside {
                    return false;
                }
                // Outer must be strictly larger in ≥2 dimensions
                let dims = [
                    (o_max.x() - o_min.x(), i_max.x() - i_min.x()),
                    (o_max.y() - o_min.y(), i_max.y() - i_min.y()),
                    (o_max.z() - o_min.z(), i_max.z() - i_min.z()),
                ];
                dims.iter()
                    .filter(|(outer_d, inner_d)| *outer_d > *inner_d * 1.1)
                    .count()
                    >= 2
            };

        // Use classifier when available. Fall back to AABB when the
        // CONTAINING solid's classifier is unavailable:
        //   b_in_a: A is the container → fallback when ca.is_none()
        //   a_in_b: B is the container → fallback when cb.is_none()
        // Containment requires BOTH center test AND AABB containment.
        // Center-inside alone is insufficient: A's center can be inside B
        // while A extends far beyond B (e.g., T-shape fuse).
        let b_in_a = (b_center_in_a && a_center_outside_b && aabb_contains(&aabb_b, &aabb_a))
            || (ca.is_none() && aabb_contains(&aabb_b, &aabb_a));
        let a_in_b = (a_center_in_b && b_center_outside_a && aabb_contains(&aabb_a, &aabb_b))
            || (cb.is_none() && aabb_contains(&aabb_a, &aabb_b));
        // Identical-solid shortcut: both centers inside each other AND
        // bounding boxes match ⇒ A ≡ B geometrically.
        let aabbs_match = aabb_a
            .zip(aabb_b)
            .map(|((a_min, a_max), (b_min, b_max))| {
                let eps = tol.linear;
                (a_min.x() - b_min.x()).abs() < eps
                    && (a_min.y() - b_min.y()).abs() < eps
                    && (a_min.z() - b_min.z()).abs() < eps
                    && (a_max.x() - b_max.x()).abs() < eps
                    && (a_max.y() - b_max.y()).abs() < eps
                    && (a_max.z() - b_max.z()).abs() < eps
            })
            .unwrap_or(false);
        if b_center_in_a && a_center_in_b && aabbs_match {
            return match op {
                BooleanOp::Fuse | BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, a)?),
                BooleanOp::Cut => Err(crate::OperationsError::InvalidInput {
                    reason: "Cut of identical solids produces empty result".into(),
                }),
            };
        }
        // For Cut with containment, defer to GFA (produces shelled solid).
        if (b_in_a || a_in_b) && op != BooleanOp::Cut {
            return match (op, b_in_a, a_in_b) {
                (BooleanOp::Fuse, true, _) => Ok(crate::copy::copy_solid(topo, a)?),
                (BooleanOp::Fuse, _, true) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Intersect, true, _) => Ok(crate::copy::copy_solid(topo, b)?),
                (BooleanOp::Intersect, _, true) => Ok(crate::copy::copy_solid(topo, a)?),
                _ => Err(crate::OperationsError::InvalidInput {
                    reason: "containment shortcut: unexpected state".into(),
                }),
            };
        }
    }

    // ── GFA pipeline ─────────────────────────────────────────────────
    let algo_op = match op {
        BooleanOp::Fuse => brepkit_algo::bop::BooleanOp::Fuse,
        BooleanOp::Cut => brepkit_algo::bop::BooleanOp::Cut,
        BooleanOp::Intersect => brepkit_algo::bop::BooleanOp::Intersect,
    };
    let gfa_start = timer_now();
    match brepkit_algo::gfa::boolean(topo, algo_op, a, b) {
        Ok(result) => {
            let result_faces = brepkit_topology::explorer::solid_faces(topo, result)
                .map(|f| f.len())
                .unwrap_or(0);
            if result_faces > 0 {
                let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
                // Check Euler before unify_faces — if already valid, skip
                // unify to avoid its face-merging bugs (non-manifold edges).
                let (f_pre, e_pre, v_pre) =
                    brepkit_topology::explorer::solid_entity_counts(topo, result)?;
                #[allow(clippy::cast_possible_wrap)]
                let euler_pre = (v_pre as i64) - (e_pre as i64) + (f_pre as i64);
                if euler_pre != 2 {
                    for _ in 0..3 {
                        if crate::heal::unify_faces(topo, result)? == 0 {
                            break;
                        }
                    }
                }
                let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(topo, result)?;
                #[allow(clippy::cast_possible_wrap)]
                let euler = (v as i64) - (e as i64) + (f as i64);
                if euler == 2 && validate_boolean_result(topo, result).is_ok() {
                    log::info!(
                        "GFA boolean succeeded in {:.1}ms ({result_faces} faces)",
                        timer_elapsed_ms(gfa_start)
                    );
                    return Ok(result);
                }
            }
            log::warn!(
                "GFA result failed validation in {:.1}ms (faces={result_faces}), falling back to mesh boolean",
                timer_elapsed_ms(gfa_start)
            );
        }
        Err(e) => {
            log::warn!(
                "GFA boolean failed in {:.1}ms ({e}), falling back to mesh boolean",
                timer_elapsed_ms(gfa_start)
            );
        }
    }

    // ── Mesh boolean fallback (no recursion) ─────────────────────────
    let opts = BooleanOptions::default();
    let raw = mesh_boolean_fallback(topo, op, a, b, opts.deflection, tol, &opts)?;
    let result = crate::copy::copy_solid(topo, raw)?;
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    for _ in 0..3 {
        if crate::heal::unify_faces(topo, result)? == 0 {
            break;
        }
    }
    Ok(enforce_manifold_shell(topo, result).unwrap_or(result))
}

/// Perform a boolean operation with custom options.
///
/// **Deprecated:** Options are currently ignored. All booleans route through
/// the GFA pipeline with mesh boolean fallback. This wrapper exists for
/// backward compatibility with callers that pass `BooleanOptions`.
///
/// # Errors
///
/// Returns the same errors as [`boolean`].
pub fn boolean_with_options(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    _opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    boolean(topo, op, a, b)
}

/// Sequential compound cut via GFA.
///
/// Cuts the `target` solid by each tool in order using sequential
/// `boolean(Cut)` calls.
///
/// # Errors
///
/// Returns an error if any individual cut fails.
pub fn compound_cut(
    topo: &mut Topology,
    target: SolidId,
    tools: &[SolidId],
    _opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let mut result = target;
    for &tool in tools {
        result = boolean(topo, BooleanOp::Cut, result, tool)?;
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Evolution-tracking wrapper
// ---------------------------------------------------------------------------

/// Perform a boolean operation and return an [`crate::evolution::EvolutionMap`] tracking face
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
/// Detects non-manifold edges (shared by 3+ faces) and uses greedy
/// shell building to split the non-manifold shell into manifold
/// sub-shells. The largest sub-shell becomes the outer shell; smaller ones
/// become inner shells (cavities).
///
/// If the solid is already manifold, returns it unchanged.
/// Check if a solid's shell has edge-manifold topology (every edge shared by exactly 2 faces).
#[allow(dead_code)]
fn is_edge_manifold(topo: &Topology, solid: SolidId) -> bool {
    let shell = match topo.solid(solid).and_then(|s| topo.shell(s.outer_shell())) {
        Ok(sh) => sh,
        Err(_) => return false,
    };
    let mut edge_count: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    for &fid in shell.faces() {
        let Ok(face) = topo.face(fid) else {
            return false;
        };
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let Ok(wire) = topo.wire(wid) else {
                continue;
            };
            for oe in wire.edges() {
                *edge_count.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    edge_count.values().all(|&n| n == 2)
}

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

// ---------------------------------------------------------------------------
// Shared utility functions (relocated from deleted files)
// ---------------------------------------------------------------------------

/// Sample `n` evenly-spaced points along a closed edge curve.
///
/// For `Circle` and `Ellipse`, samples at `TAU * i / n`.
/// For closed `NurbsCurve`, samples across the domain avoiding endpoint
/// duplication. Returns an empty vec for `Line` (no sampling possible).
pub(crate) fn sample_edge_curve(curve: &EdgeCurve, n: usize) -> Vec<Point3> {
    match curve {
        EdgeCurve::Circle(c) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                c.evaluate(t)
            })
            .collect(),
        EdgeCurve::Ellipse(e) => (0..n)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                e.evaluate(t)
            })
            .collect(),
        EdgeCurve::NurbsCurve(nc) => {
            let (u0, u1) = nc.domain();
            // For closed curves (start ~ end), use n as divisor to avoid
            // duplicating the first point at t=u_max.
            let start_pt = nc.evaluate(u0);
            let end_pt = nc.evaluate(u1);
            // 1e-6 m: closure detection threshold — if start and end points are
            // within 1 micron, treat the NURBS curve as closed to avoid
            // duplicating the first point at t=u_max.
            let is_closed = (start_pt - end_pt).length() < 1e-6;
            let divisor = if is_closed { n } else { n - 1 };
            (0..n)
                .map(|i| {
                    #[allow(clippy::cast_precision_loss)]
                    let t = u0 + (u1 - u0) * (i as f64) / (divisor as f64);
                    nc.evaluate(t)
                })
                .collect()
        }
        EdgeCurve::Line => vec![],
    }
}

/// Get a polygon approximation of a face by sampling curved edges.
///
/// Samples circle/ellipse edges into 32 points so faces with a
/// single closed-curve edge (e.g. cylinder caps) get a proper polygon.
///
/// # Errors
///
/// Returns an error if the face or its wire cannot be resolved.
pub fn face_polygon(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<Point3>, crate::OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let curve = edge.curve();
        // Sample closed parametric edges (start == end vertex).
        // Partial arcs fall through to the vertex-based path.
        let start_vid = edge.start();
        let end_vid = edge.end();
        let is_closed_edge = start_vid == end_vid
            && matches!(
                curve,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            );
        if is_closed_edge {
            // Must use CLOSED_CURVE_SAMPLES (not a larger value) — vertex count
            // must match create_band_fragments and inner-wire dedup for sharing.
            let mut sampled = sample_edge_curve(curve, types::CLOSED_CURVE_SAMPLES);
            if !oe.is_forward() {
                sampled.reverse();
            }
            pts.extend(sampled);
        } else {
            let vid = oe.oriented_start(edge);
            pts.push(topo.vertex(vid)?.point());
        }
    }

    Ok(pts)
}

/// Collect face signatures (index, normal, centroid) for evolution tracking.
///
/// For each face of the solid, computes a representative normal and centroid
/// from the face polygon. Used by [`boolean_with_evolution`] to match output
/// faces back to input faces.
///
/// # Errors
///
/// Returns an error if any face or wire cannot be resolved.
fn collect_face_signatures(
    topo: &Topology,
    solid_id: SolidId,
) -> Result<Vec<(usize, Vec3, Point3)>, crate::OperationsError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;
    let mut result = Vec::with_capacity(shell.faces().len());

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        let verts = face_polygon(topo, fid)?;
        let normal = if let FaceSurface::Plane { normal, .. } = face.surface() {
            *normal
        } else if verts.len() >= 3 {
            let e1 = verts[1] - verts[0];
            let e2 = verts[2] - verts[0];
            e1.cross(e2).normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0))
        } else {
            Vec3::new(0.0, 0.0, 1.0)
        };

        let centroid = classify::polygon_centroid(&verts);
        result.push((fid.index(), normal, centroid));
    }

    Ok(result)
}

#[cfg(test)]
mod tests;

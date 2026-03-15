//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! Supports both planar and NURBS faces. NURBS faces are tessellated
//! into planar triangles before clipping, enabling approximate boolean
//! operations on any solid geometry.

mod analytic;
mod assembly;
mod classify;
mod compound;
mod fragments;
mod intersect;
mod precompute;
mod split;
mod types;
use analytic::{analytic_boolean, collect_face_signatures, has_torus, is_all_analytic};
use assembly::validate_boolean_result;
pub(crate) use assembly::{assemble_solid, assemble_solid_mixed};
use classify::{
    build_face_bvh, classify_fragment, polygon_centroid, try_build_analytic_classifier,
};
pub use compound::compound_cut;
use intersect::{compute_analytic_segments, compute_intersection_segments};
pub use precompute::face_polygon;
use precompute::{collect_face_data, handle_disjoint, solid_aabb, try_containment_shortcut};
use split::split_face;
#[cfg(not(target_arch = "wasm32"))]
use types::PARALLEL_THRESHOLD;
use types::{BooleanContext, FaceClass, FaceFragment, Source, select_fragment};
pub use types::{BooleanOp, BooleanOptions, FaceSpec};

use std::collections::HashMap;

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
    let ctx = BooleanContext::from_options(&opts);

    log::debug!(
        "boolean {op:?}: solids ({}, {}), deflection={}, vertex_merge={}, classify_tol={}, degenerate_area={}",
        a.index(),
        b.index(),
        opts.deflection,
        ctx.vertex_merge,
        ctx.classify_tol,
        ctx.degenerate_area,
    );

    // ── Try analytic fast path ──────────────────────────────────────────
    // Use when both solids are all-analytic (no NURBS) and neither contains
    // torus faces. This covers cutting/drilling with cylinders/cones and
    // sphere intersections. Sphere faces are handled by tessellating them
    // into triangle fragments within the analytic boolean, with O(1)
    // point-in-sphere classification for the opposite solid.
    //
    // Torus faces are excluded because their parametric decomposition
    // (degree-4 intersection curves) is not yet implemented.
    let try_analytic = {
        let both_analytic = is_all_analytic(topo, a)? && is_all_analytic(topo, b)?;
        let no_torus = !has_torus(topo, a)? && !has_torus(topo, b)?;
        both_analytic && no_torus
    };
    if try_analytic {
        if let Ok(solid) = analytic_boolean(topo, op, a, b, tol, opts.deflection) {
            let _ = crate::heal::remove_degenerate_edges(topo, solid, tol.linear)?;
            if opts.unify_faces {
                let _ = crate::heal::unify_faces(topo, solid)?;
            }
            validate_boolean_result(topo, solid)?;
            return Ok(solid);
        }
        // Analytic path failed; fall back to tessellated boolean.
    }

    // ── Phase 0: Guard + Precompute ──────────────────────────────────────

    let faces_a = collect_face_data(topo, a, opts.deflection)?;
    let faces_b = collect_face_data(topo, b, opts.deflection)?;

    let aabb_a = solid_aabb(topo, &faces_a, tol)?;
    let aabb_b = solid_aabb(topo, &faces_b, tol)?;

    // Disjoint AABB shortcut.
    if !aabb_a.intersects(aabb_b) {
        log::debug!("boolean {op:?}: disjoint AABBs, shortcut");
        return handle_disjoint(topo, op, &faces_a, &faces_b);
    }

    // ── Containment shortcut ─────────────────────────────────────────
    // If one solid is entirely inside the other, skip expensive intersection
    // computation and go directly to the appropriate result.
    if let Some(result) = try_containment_shortcut(topo, op, a, b, &faces_a, &faces_b, tol)? {
        return Ok(result);
    }

    // ── Phase 1a: Analytic fast path ───────────────────────────────────

    let (analytic_segs, analytic_pairs) = compute_analytic_segments(topo, a, b, tol)?;

    // ── Phase 1b: Tessellated intersection (skip analytic pairs) ────────

    let tess_segs = compute_intersection_segments(&faces_a, &faces_b, tol, &analytic_pairs);

    let mut segments = analytic_segs;
    segments.extend(tess_segs);

    // Build chord map: FaceId → Vec<(Point3, Point3)>
    let mut chord_map: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
    for seg in &segments {
        chord_map
            .entry(seg.face_a.index())
            .or_default()
            .push((seg.p0, seg.p1));
        chord_map
            .entry(seg.face_b.index())
            .or_default()
            .push((seg.p0, seg.p1));
    }

    // ── Phase 3: Face splitting ──────────────────────────────────────────

    let mut fragments: Vec<FaceFragment> = Vec::new();

    for &(fid, ref verts, normal, d) in &faces_a {
        fragments.extend(split_face(
            fid,
            verts,
            normal,
            d,
            Source::A,
            &chord_map,
            tol,
        ));
    }
    for &(fid, ref verts, normal, d) in &faces_b {
        fragments.extend(split_face(
            fid,
            verts,
            normal,
            d,
            Source::B,
            &chord_map,
            tol,
        ));
    }

    // ── Phase 4: Classification ──────────────────────────────────────────

    // Build analytic classifiers for O(1) point-in-solid tests when possible.
    let analytic_a = try_build_analytic_classifier(topo, a);
    let analytic_b = try_build_analytic_classifier(topo, b);

    // Build BVH acceleration structures for face data.
    let bvh_a = build_face_bvh(&faces_a);
    let bvh_b = build_face_bvh(&faces_b);

    // Pre-expand opposing AABBs for classification. A centroid outside the
    // opposing solid's bounding box is definitively Outside, skipping expensive
    // ray-casts. The padding accounts for floating-point rounding in
    // `polygon_centroid` which may shift a boundary centroid by a few ULP.
    let padded_aabb_a = aabb_a.expanded(tol.linear);
    let padded_aabb_b = aabb_b.expanded(tol.linear);

    // Classification: parallelize when fragment count justifies rayon overhead.
    let classify_fn = |frag: &FaceFragment| -> FaceClass {
        let centroid = polygon_centroid(&frag.vertices);

        // AABB pre-filter: skip expensive ray-cast for fragments whose centroid
        // is outside the opposing solid's bounding box.
        let opposing_aabb = match frag.source {
            Source::A => padded_aabb_b,
            Source::B => padded_aabb_a,
        };
        if !opposing_aabb.contains_point(centroid) {
            return FaceClass::Outside;
        }

        let fast = match frag.source {
            Source::A => analytic_b.as_ref().and_then(|c| c.classify(centroid, tol)),
            Source::B => analytic_a.as_ref().and_then(|c| c.classify(centroid, tol)),
        };
        if let Some(class) = fast {
            return class;
        }
        let (opposite, bvh) = match frag.source {
            Source::A => (&faces_b, bvh_b.as_ref()),
            Source::B => (&faces_a, bvh_a.as_ref()),
        };
        classify_fragment(frag, opposite, bvh, tol)
    };

    // Rayon panics on wasm32 (no thread pool) — use sequential iteration only.
    #[cfg(not(target_arch = "wasm32"))]
    let classes: Vec<FaceClass> = if fragments.len() >= PARALLEL_THRESHOLD {
        use rayon::prelude::*;
        fragments.par_iter().map(classify_fn).collect()
    } else {
        fragments.iter().map(classify_fn).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let classes: Vec<FaceClass> = fragments.iter().map(classify_fn).collect();

    // ── Phase 5: Selection ───────────────────────────────────────────────

    let mut selected: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for (frag, &class) in fragments.iter().zip(classes.iter()) {
        if let Some(flip) = select_fragment(frag.source, class, op) {
            // When flipping, negate the plane and reverse winding to keep CCW from outside.
            let (verts, normal, d) = if flip {
                let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
                (rev, -frag.normal, -frag.d)
            } else {
                (frag.vertices.clone(), frag.normal, frag.d)
            };
            selected.push((verts, normal, d));
        }
    }

    if selected.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "boolean operation produced no faces (empty result)".into(),
        });
    }

    // ── Phase 6: Assembly ────────────────────────────────────────────────

    let result = assemble_solid(topo, &selected, tol)?;

    // ── Phase 6b: Post-boolean healing ─────────────────────────────────
    // Light healing: remove degenerate (zero-length) edges and fix face
    // orientations. We intentionally skip vertex merging and face removal
    // since they can corrupt valid boolean output with small features.
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    // Merge co-surface faces that were split by the boolean pipeline.
    // Without this, sequential booleans cause topology explosion (10-27× more
    // faces than necessary) because each boolean creates fragments that
    // accumulate.  This is analogous to OCCT's same-domain face merging.
    if opts.unify_faces {
        let _ = crate::heal::unify_faces(topo, result)?;
    }
    // Full shape healing: vertex merging, small face/edge removal, etc.
    // Only enabled for final results — healing can corrupt intermediates
    // fed into further booleans.
    if opts.heal_after_boolean {
        let _ = crate::heal::heal_solid(topo, result, tol.linear)?;
    }

    // ── Phase 7: Degenerate result check ──────────────────────────────
    validate_boolean_result(topo, result)?;

    log::info!(
        "boolean {op:?}: tessellated path succeeded → solid {} ({} faces)",
        result.index(),
        selected.len()
    );
    Ok(result)
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

#[cfg(test)]
mod tests;

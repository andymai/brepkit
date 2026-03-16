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
pub(crate) mod pcurve_compute;
pub(crate) mod pipeline;
pub(crate) mod plane_frame;
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

use std::collections::{HashMap, HashSet};

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
use brepkit_topology::face::FaceSurface;
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

    // ── Mesh boolean guard for high face counts ─────────────────────
    // When either solid has many topology faces (e.g. from NURBS/torus
    // tessellation in prior booleans or merged faces from unify_faces),
    // the chord-based path is O(N²). Fall back to mesh boolean
    // (co-refinement, O(N log N)).
    //
    // Two thresholds:
    // - Per-solid: >100 faces (catches individual complex solids, e.g.
    //   shelled+filleted geometry with merged faces)
    // - Combined: >500 faces (catches pairs of moderate-complexity solids)
    //
    // This check is O(1) and avoids the expensive collect_face_data
    // tessellation that would otherwise run first.
    {
        let count_a = topo.shell(topo.solid(a)?.outer_shell())?.faces().len();
        let count_b = topo.shell(topo.solid(b)?.outer_shell())?.faces().len();
        // Also force mesh boolean when either solid has torus faces and the
        // analytic path was skipped. The chord-based path tessellates torus
        // faces into hundreds of triangles, making O(N²) intersection
        // prohibitively slow. Mesh boolean handles this in O(N log N).
        let has_torus_faces = !try_analytic
            && (has_torus(topo, a).unwrap_or(false) || has_torus(topo, b).unwrap_or(false));
        if count_a > types::MESH_BOOLEAN_PER_SOLID_THRESHOLD
            || count_b > types::MESH_BOOLEAN_PER_SOLID_THRESHOLD
            || count_a + count_b > types::MESH_BOOLEAN_FACE_THRESHOLD
            || has_torus_faces
        {
            log::debug!(
                "boolean {op:?}: high face count ({count_a} + {count_b}) or torus, using mesh boolean"
            );
            match mesh_boolean_fallback(topo, op, a, b, opts.deflection, tol, &opts) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    log::debug!(
                        "boolean {op:?}: mesh boolean fallback failed ({e}), \
                         falling through to chord-based path"
                    );
                }
            }
        }
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

    // ── Surface preservation: unsplit faces keep their original surface ──
    //
    // The chord-based path tessellates non-planar faces (NURBS, cone, sphere)
    // into planar triangles for intersection testing. Faces that don't
    // intersect the other solid (no chords) are preserved with their original
    // surface type, avoiding face-count explosion from unnecessary tessellation.

    let split_fids: HashSet<usize> = chord_map.keys().copied().collect();

    // Build surface + polygon map for original non-planar faces.
    // Each entry: (FaceSurface, is_reversed, polygon_vertices).
    let build_surface_map =
        |topo: &Topology, solid: SolidId| -> HashMap<usize, (FaceSurface, bool, Vec<Point3>)> {
            let mut map = HashMap::new();
            let Ok(s) = topo.solid(solid) else { return map };
            let Ok(shell) = topo.shell(s.outer_shell()) else {
                return map;
            };
            for &fid in shell.faces() {
                let Ok(face) = topo.face(fid) else { continue };
                if matches!(face.surface(), FaceSurface::Plane { .. }) {
                    continue; // Planar faces don't need surface preservation.
                }
                if let Ok(polygon) = face_polygon(topo, fid) {
                    map.insert(
                        fid.index(),
                        (face.surface().clone(), face.is_reversed(), polygon),
                    );
                }
            }
            map
        };
    let surface_map_a = build_surface_map(topo, a);
    let surface_map_b = build_surface_map(topo, b);

    // ── Phase 3: Face splitting ──────────────────────────────────────────
    // Only split faces that actually have intersection chords. Unsplit faces
    // are handled separately below.

    let mut fragments: Vec<FaceFragment> = Vec::new();

    for &(fid, ref verts, normal, d) in &faces_a {
        if !split_fids.contains(&fid.index()) {
            continue;
        }
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
        if !split_fids.contains(&fid.index()) {
            continue;
        }
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

    let analytic_a = try_build_analytic_classifier(topo, a);
    let analytic_b = try_build_analytic_classifier(topo, b);

    let bvh_a = build_face_bvh(&faces_a);
    let bvh_b = build_face_bvh(&faces_b);

    let padded_aabb_a = aabb_a.expanded(tol.linear);
    let padded_aabb_b = aabb_b.expanded(tol.linear);

    let classify_fn = |frag: &FaceFragment| -> FaceClass {
        let centroid = polygon_centroid(&frag.vertices);
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

    let mut selected_specs: Vec<FaceSpec> = Vec::new();

    // 5a: Select split face fragments (planar triangles from tessellation).
    for (frag, &class) in fragments.iter().zip(classes.iter()) {
        if let Some(flip) = select_fragment(frag.source, class, op) {
            let (verts, normal, d) = if flip {
                let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
                (rev, -frag.normal, -frag.d)
            } else {
                (frag.vertices.clone(), frag.normal, frag.d)
            };
            selected_specs.push(FaceSpec::Planar {
                vertices: verts,
                normal,
                d,
                inner_wires: vec![],
            });
        }
    }

    // 5b: Select unsplit faces — classify once per face, preserve original surface.
    // For tessellated non-planar faces, collect_face_data produces multiple entries
    // per FaceId. We classify using the first entry's centroid and emit a single
    // FaceSpec with the original surface (instead of N planar triangles).
    let mut processed_unsplit: HashSet<(usize, u8)> = HashSet::new(); // (fid, source)

    let classify_unsplit = |centroid: Point3, source: Source| -> FaceClass {
        let opposing_aabb = match source {
            Source::A => padded_aabb_b,
            Source::B => padded_aabb_a,
        };
        if !opposing_aabb.contains_point(centroid) {
            return FaceClass::Outside;
        }
        let fast = match source {
            Source::A => analytic_b.as_ref().and_then(|c| c.classify(centroid, tol)),
            Source::B => analytic_a.as_ref().and_then(|c| c.classify(centroid, tol)),
        };
        if let Some(class) = fast {
            return class;
        }
        // Build a temporary fragment for ray-cast classification.
        let dummy = FaceFragment {
            vertices: vec![centroid],
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
            source,
        };
        let (opposite, bvh) = match source {
            Source::A => (&faces_b, bvh_b.as_ref()),
            Source::B => (&faces_a, bvh_a.as_ref()),
        };
        classify_fragment(&dummy, opposite, bvh, tol)
    };

    // Process unsplit faces from solid A.
    for &(fid, ref verts, normal, d) in &faces_a {
        if split_fids.contains(&fid.index()) {
            continue;
        }
        let key = (fid.index(), 0u8);
        if !processed_unsplit.insert(key) {
            continue;
        } // Skip duplicate tessellation entries.

        let centroid = polygon_centroid(verts);
        let class = classify_unsplit(centroid, Source::A);
        if let Some(flip) = select_fragment(Source::A, class, op) {
            if let Some((surface, reversed, polygon)) = surface_map_a.get(&fid.index()) {
                // Non-planar face: preserve original surface.
                let r = if flip { !reversed } else { *reversed };
                let face_verts = if flip {
                    polygon.iter().copied().rev().collect()
                } else {
                    polygon.clone()
                };
                selected_specs.push(FaceSpec::Surface {
                    vertices: face_verts,
                    surface: surface.clone(),
                    reversed: r,
                    inner_wires: vec![],
                });
            } else {
                // Planar face: emit as FaceSpec::Planar.
                let (v, n, dd) = if flip {
                    (verts.iter().copied().rev().collect(), -normal, -d)
                } else {
                    (verts.clone(), normal, d)
                };
                selected_specs.push(FaceSpec::Planar {
                    vertices: v,
                    normal: n,
                    d: dd,
                    inner_wires: vec![],
                });
            }
        }
    }

    // Process unsplit faces from solid B.
    for &(fid, ref verts, normal, d) in &faces_b {
        if split_fids.contains(&fid.index()) {
            continue;
        }
        let key = (fid.index(), 1u8);
        if !processed_unsplit.insert(key) {
            continue;
        }

        let centroid = polygon_centroid(verts);
        let class = classify_unsplit(centroid, Source::B);
        if let Some(flip) = select_fragment(Source::B, class, op) {
            if let Some((surface, reversed, polygon)) = surface_map_b.get(&fid.index()) {
                let r = if flip { !reversed } else { *reversed };
                let face_verts = if flip {
                    polygon.iter().copied().rev().collect()
                } else {
                    polygon.clone()
                };
                selected_specs.push(FaceSpec::Surface {
                    vertices: face_verts,
                    surface: surface.clone(),
                    reversed: r,
                    inner_wires: vec![],
                });
            } else {
                let (v, n, dd) = if flip {
                    (verts.iter().copied().rev().collect(), -normal, -d)
                } else {
                    (verts.clone(), normal, d)
                };
                selected_specs.push(FaceSpec::Planar {
                    vertices: v,
                    normal: n,
                    d: dd,
                    inner_wires: vec![],
                });
            }
        }
    }

    if selected_specs.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "boolean operation produced no faces (empty result)".into(),
        });
    }

    // ── Phase 6: Assembly ────────────────────────────────────────────────

    let result = assemble_solid_mixed(topo, &selected_specs, tol)?;

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
        selected_specs.len()
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

#[cfg(test)]
mod tests;

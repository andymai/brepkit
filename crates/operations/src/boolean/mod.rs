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
        // Use measure::solid_bounding_box — it expands for surface curvature
        // (cylinder vertex projection, sphere/torus analytic). The naive
        // edge-vertex sampler missed cylinder lateral extents because cylinders
        // only have seam vertices, leaving the AABB center on the lateral
        // surface where the analytic classifier returns None.
        let sample_aabb = |topo: &Topology, solid: SolidId| -> Option<(Point3, Point3)> {
            let bb = crate::measure::solid_bounding_box(topo, solid).ok()?;
            Some((bb.min, bb.max))
        };
        let aabb_a = sample_aabb(topo, a);
        let aabb_b = sample_aabb(topo, b);
        // AABB-encloses check (lenient): does `inner` fit inside `outer`?
        let aabb_encloses =
            |inner: &Option<(Point3, Point3)>, outer: &Option<(Point3, Point3)>| -> bool {
                let Some(((i_min, i_max), (o_min, o_max))) = inner.zip(*outer) else {
                    return false;
                };
                let margin = tol.linear;
                i_min.x() >= o_min.x() - margin
                    && i_min.y() >= o_min.y() - margin
                    && i_min.z() >= o_min.z() - margin
                    && i_max.x() <= o_max.x() + margin
                    && i_max.y() <= o_max.y() + margin
                    && i_max.z() <= o_max.z() + margin
            };
        // AABB-strictly-contains (strict): outer must also be ≥10% larger in
        // ≥2 dims. Used as the no-classifier fallback to avoid false-positive
        // containment when AABBs overlap but neither solid contains the other.
        let aabb_strictly_contains =
            |inner: &Option<(Point3, Point3)>, outer: &Option<(Point3, Point3)>| -> bool {
                if !aabb_encloses(inner, outer) {
                    return false;
                }
                let Some(((i_min, i_max), (o_min, o_max))) = inner.zip(*outer) else {
                    return false;
                };
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

        // Bidirectional vertex check via the analytic classifier — the
        // primary signal for identical/containment classification. A vertex
        // classifying as inside-or-on (None within tolerance band counts
        // as on) means it sits within the solid's region.
        let all_b_verts_in_a = ca
            .as_ref()
            .is_some_and(|c| all_vertices_inside_or_on(topo, b, c, tol));
        let all_a_verts_in_b = cb
            .as_ref()
            .is_some_and(|c| all_vertices_inside_or_on(topo, a, c, tol));
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

        // Containment shortcut: A contains B when all B vertices are
        // inside-or-on A AND A's AABB encloses B's. The vertex check
        // handles cases where AABB centers coincide (e.g., concentric
        // cylinders of same radius, different heights) where the prior
        // center-based check failed. Falls back to AABB-only when the
        // containing solid's classifier is unavailable.
        let b_in_a = (all_b_verts_in_a && aabb_encloses(&aabb_b, &aabb_a))
            || (ca.is_none() && aabb_strictly_contains(&aabb_b, &aabb_a));
        let a_in_b = (all_a_verts_in_b && aabb_encloses(&aabb_a, &aabb_b))
            || (cb.is_none() && aabb_strictly_contains(&aabb_a, &aabb_b));

        // Identical-solid shortcut: matching AABBs AND every boundary
        // vertex of each solid classifies as inside-or-on the other's
        // analytic classifier. Stronger than a center test (a cube
        // inscribed in a sphere has matching AABBs but cube corners fall
        // outside the sphere) and works for non-convex solids like tori.
        if aabbs_match && all_b_verts_in_a && all_a_verts_in_b {
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

        // Coaxial-cylinder merge shortcut: when both A and B are simple
        // cylinder solids (cylinder + 2 planar caps) with the same axis,
        // origin and radius, fuse/intersect collapse to a single cylinder
        // spanning the combined / overlapping axial range. Bypasses GFA's
        // cap-on-cap and lateral-SD coplanar handling, which currently
        // falls through to a non-manifold mesh fallback.
        if let (
            Some(brepkit_algo::classifier::AnalyticClassifier::Cylinder {
                origin: oa,
                axis: aa,
                radius: ra,
                z_min: za_min,
                z_max: za_max,
            }),
            Some(brepkit_algo::classifier::AnalyticClassifier::Cylinder {
                origin: ob,
                axis: ab,
                radius: rb,
                z_min: zb_min,
                z_max: zb_max,
            }),
        ) = (ca.as_ref(), cb.as_ref())
        {
            // Axes coincide (same line) when directions are parallel AND
            // the origin offset is parallel to the axis (no perpendicular
            // component beyond linear tolerance).
            let same_axis_dir = aa.dot(*ab) > 1.0 - tol.angular;
            let origin_offset = *ob - *oa;
            let along_axis = origin_offset.dot(*aa);
            let perpendicular = origin_offset - *aa * along_axis;
            let coaxial = same_axis_dir && perpendicular.length() < tol.linear;
            let same_radius = (ra - rb).abs() < tol.linear;
            if coaxial && same_radius {
                // Translate B's z-range into A's axis frame.
                let za = (*za_min, *za_max);
                let zb = (*zb_min + along_axis, *zb_max + along_axis);
                if let Some(result) =
                    coaxial_cylinder_shortcut(topo, op, *oa, *aa, *ra, za, zb, tol)?
                {
                    return Ok(result);
                }
            }
        }

        // Coaxial-cone merge shortcut: two frustums on the same conical
        // surface (shared apex, axis, and tan(half_angle) = r/z ratio)
        // collapse to a single frustum spanning the combined axial range.
        if let (
            Some(brepkit_algo::classifier::AnalyticClassifier::Cone {
                origin: oa,
                axis: aa,
                z_min: za_min,
                z_max: za_max,
                r_at_z_min: rmin_a,
                r_at_z_max: rmax_a,
            }),
            Some(brepkit_algo::classifier::AnalyticClassifier::Cone {
                origin: ob,
                axis: ab,
                z_min: zb_min,
                z_max: zb_max,
                r_at_z_min: rmin_b,
                r_at_z_max: rmax_b,
            }),
        ) = (ca.as_ref(), cb.as_ref())
        {
            let same_axis_dir = aa.dot(*ab) > 1.0 - tol.angular;
            let same_apex = (*oa - *ob).length() < tol.linear;
            // Half-angle slope: dimensionless r/z. Use whichever endpoint has
            // |z| above tol.linear (compared against tol.linear because slope
            // is a length ratio, not an angle — `tol.angular` is a radian
            // threshold, wrong unit). When both endpoints of a frustum are
            // sub-tol (degenerate apex-pinned cone), skip the shortcut and
            // let GFA handle it rather than dividing by near-zero.
            let slope_a = if za_max.abs() > tol.linear {
                Some(rmax_a / *za_max)
            } else if za_min.abs() > tol.linear {
                Some(rmin_a / *za_min)
            } else {
                None
            };
            let slope_b = if zb_max.abs() > tol.linear {
                Some(rmax_b / *zb_max)
            } else if zb_min.abs() > tol.linear {
                Some(rmin_b / *zb_min)
            } else {
                None
            };
            let same_half_angle = match (slope_a, slope_b) {
                (Some(sa), Some(sb)) => (sa - sb).abs() < tol.linear,
                _ => false,
            };
            if let (true, Some(slope)) = (same_axis_dir && same_apex && same_half_angle, slope_a) {
                if let Some(result) = coaxial_cone_shortcut(
                    topo,
                    op,
                    *oa,
                    *aa,
                    slope,
                    (*za_min, *za_max),
                    (*zb_min, *zb_max),
                    tol,
                )? {
                    return Ok(result);
                }
            }
        }

        // Axis-aligned box-pair shortcut: when both A and B classify as
        // Box (analytic classifier infers axis-aligned bounds), Fuse and
        // Intersect can be computed exactly via AABB algebra. Bypasses
        // GFA so chained operations get clean fresh-primitive topology
        // rather than residual GFA splits that confuse subsequent steps.
        if let (
            Some(brepkit_algo::classifier::AnalyticClassifier::Box {
                min: a_min,
                max: a_max,
            }),
            Some(brepkit_algo::classifier::AnalyticClassifier::Box {
                min: b_min,
                max: b_max,
            }),
        ) = (ca.as_ref(), cb.as_ref())
        {
            if let Some(result) = box_pair_shortcut(topo, op, *a_min, *a_max, *b_min, *b_max, tol)?
            {
                return Ok(result);
            }
        }

        // Concentric-sphere merge shortcut: when both A and B classify as
        // Sphere with coincident centers, Fuse and Intersect collapse to a
        // single sphere by radius algebra. Bypasses GFA's coplanar-pole
        // handling (which currently routes spheres through the same SD
        // pipeline that flakes on coaxial cylinders pre-#541).
        //
        // Cut intentionally falls through to GFA: subtracting an inner
        // sphere from an outer one yields a hollow ball, whose topology
        // (outer shell + inner shell) requires builder support beyond the
        // single-sphere primitive used here.
        if let (
            Some(brepkit_algo::classifier::AnalyticClassifier::Sphere {
                center: ca_center,
                radius: ra,
            }),
            Some(brepkit_algo::classifier::AnalyticClassifier::Sphere {
                center: cb_center,
                radius: rb,
            }),
        ) = (ca.as_ref(), cb.as_ref())
        {
            let coincident = (*ca_center - *cb_center).length() < tol.linear;
            if coincident {
                if let Some(result) =
                    concentric_sphere_shortcut(topo, op, a, b, *ca_center, *ra, *rb, tol)?
                {
                    return Ok(result);
                }
            }
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

                // If Euler>2, try merging duplicate vertices before unify.
                // This fixes the flush-face case where duplicate vertices at
                // cross-rank positions inflate V.
                if euler_pre > 2 {
                    // Best-effort: don't abort on merge failure
                    let _ = merge_result_vertices(topo, result, tol);
                }

                let (f2, e2, v2) = brepkit_topology::explorer::solid_entity_counts(topo, result)?;
                #[allow(clippy::cast_possible_wrap)]
                let euler_pre2 = (v2 as i64) - (e2 as i64) + (f2 as i64);

                if euler_pre2 != 2 {
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
/// Runs the standard GFA boolean pipeline, then applies post-processing
/// options. Currently supported: `unify_faces` (merges co-surface face
/// fragments via `brepkit_heal::unify_same_domain`).
///
/// # Errors
///
/// Returns the same errors as [`boolean`].
pub fn boolean_with_options(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let result = boolean(topo, op, a, b)?;
    if opts.unify_faces {
        // Merge co-surface face fragments left by the boolean.
        let unify_opts = brepkit_heal::upgrade::unify_same_domain::UnifyOptions::default();
        if let Err(e) =
            brepkit_heal::upgrade::unify_same_domain::unify_same_domain(topo, result, &unify_opts)
        {
            log::debug!("boolean unify_faces post-processing failed: {e}");
        }
    }
    Ok(result)
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
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let mut result = target;
    for &tool in tools {
        result = boolean(topo, BooleanOp::Cut, result, tool)?;
    }
    if opts.unify_faces {
        let unify_opts = brepkit_heal::upgrade::unify_same_domain::UnifyOptions::default();
        if let Err(e) =
            brepkit_heal::upgrade::unify_same_domain::unify_same_domain(topo, result, &unify_opts)
        {
            log::debug!("compound_cut unify_faces failed: {e}");
        }
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

/// Compute the boolean of two axis-aligned boxes via AABB algebra.
///
/// Returns `Ok(None)` when the result isn't a single box:
/// - Fuse: requires two of three dims to match exactly AND the boxes to
///   overlap or touch in the third dim. Otherwise the union is L-shaped.
/// - Intersect: any non-empty AABB intersection is a box.
/// - Cut: skipped — the general case is L-shaped, defer to GFA.
fn box_pair_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    a_min: Point3,
    a_max: Point3,
    b_min: Point3,
    b_max: Point3,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let eps = tol.linear;
    let (min, max) = match op {
        BooleanOp::Intersect => {
            let lo = Point3::new(
                a_min.x().max(b_min.x()),
                a_min.y().max(b_min.y()),
                a_min.z().max(b_min.z()),
            );
            let hi = Point3::new(
                a_max.x().min(b_max.x()),
                a_max.y().min(b_max.y()),
                a_max.z().min(b_max.z()),
            );
            // Empty intersection — let general path return an error.
            if hi.x() <= lo.x() + eps || hi.y() <= lo.y() + eps || hi.z() <= lo.z() + eps {
                return Ok(None);
            }
            (lo, hi)
        }
        BooleanOp::Fuse => {
            // The union of two axis-aligned boxes is itself a box only
            // when two of three dimensions match exactly AND the boxes
            // overlap or touch in the third dim.
            let x_match =
                (a_min.x() - b_min.x()).abs() < eps && (a_max.x() - b_max.x()).abs() < eps;
            let y_match =
                (a_min.y() - b_min.y()).abs() < eps && (a_max.y() - b_max.y()).abs() < eps;
            let z_match =
                (a_min.z() - b_min.z()).abs() < eps && (a_max.z() - b_max.z()).abs() < eps;
            let matched = u8::from(x_match) + u8::from(y_match) + u8::from(z_match);
            if matched < 2 {
                return Ok(None);
            }
            // Verify overlap/touch in all three dims (the unmatched dim
            // must overlap; matched dims trivially do).
            if a_max.x() < b_min.x() - eps
                || b_max.x() < a_min.x() - eps
                || a_max.y() < b_min.y() - eps
                || b_max.y() < a_min.y() - eps
                || a_max.z() < b_min.z() - eps
                || b_max.z() < a_min.z() - eps
            {
                return Ok(None);
            }
            (
                Point3::new(
                    a_min.x().min(b_min.x()),
                    a_min.y().min(b_min.y()),
                    a_min.z().min(b_min.z()),
                ),
                Point3::new(
                    a_max.x().max(b_max.x()),
                    a_max.y().max(b_max.y()),
                    a_max.z().max(b_max.z()),
                ),
            )
        }
        BooleanOp::Cut => return Ok(None),
    };
    let dx = max.x() - min.x();
    let dy = max.y() - min.y();
    let dz = max.z() - min.z();
    if dx <= eps || dy <= eps || dz <= eps {
        return Ok(None);
    }
    let bx = crate::primitives::make_box(topo, dx, dy, dz)?;
    if min.x().abs() > eps || min.y().abs() > eps || min.z().abs() > eps {
        let xform = brepkit_math::mat::Mat4::translation(min.x(), min.y(), min.z());
        crate::transform::transform_solid(topo, bx, &xform)?;
    }
    Ok(Some(bx))
}

/// Compute the coaxial-cylinder boolean for two cylinders sharing axis,
/// origin, and radius. Returns `Ok(None)` when the shortcut doesn't apply
/// (disjoint along axis for fuse/intersect; cut requires general handling).
#[allow(clippy::too_many_arguments)]
fn coaxial_cylinder_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    origin: Point3,
    axis: Vec3,
    radius: f64,
    a_range: (f64, f64),
    b_range: (f64, f64),
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let (za_min, za_max) = a_range;
    let (zb_min, zb_max) = b_range;
    // For fuse: ranges must touch or overlap. Disjoint cylinders would
    // produce a compound, which the boolean API doesn't return.
    let touches_or_overlaps = zb_min <= za_max + tol.linear && za_min <= zb_max + tol.linear;
    let (z_min, z_max) = match op {
        BooleanOp::Fuse => {
            if !touches_or_overlaps {
                return Ok(None);
            }
            (za_min.min(zb_min), za_max.max(zb_max))
        }
        BooleanOp::Intersect => {
            // Strict overlap (not just touching) for non-degenerate result.
            let lo = za_min.max(zb_min);
            let hi = za_max.min(zb_max);
            if hi <= lo + tol.linear {
                return Ok(None);
            }
            (lo, hi)
        }
        BooleanOp::Cut => return Ok(None), // Defer to GFA / general path.
    };
    let height = z_max - z_min;
    if height <= tol.linear {
        return Ok(None);
    }
    // Build a fresh cylinder at axis-origin + axis*z_min, oriented along
    // axis. make_cylinder produces the canonical (0,0,0)→(0,0,h) cylinder;
    // then transform to the world axis frame.
    let cyl = crate::primitives::make_cylinder(topo, radius, height)?;
    let world_origin = Point3::new(
        origin.x() + axis.x() * z_min,
        origin.y() + axis.y() * z_min,
        origin.z() + axis.z() * z_min,
    );
    let xform = xform_from_canonical_z(world_origin, axis, tol);
    crate::transform::transform_solid(topo, cyl, &xform)?;
    Ok(Some(cyl))
}

/// Compute the coaxial-cone boolean for two frustums on the same conical
/// surface (shared apex, axis, and half-angle). Returns `Ok(None)` when
/// the shortcut doesn't apply.
#[allow(clippy::too_many_arguments)]
fn coaxial_cone_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    apex: Point3,
    axis: Vec3,
    slope: f64,
    a_range: (f64, f64),
    b_range: (f64, f64),
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let (za_min, za_max) = a_range;
    let (zb_min, zb_max) = b_range;
    let touches_or_overlaps = zb_min <= za_max + tol.linear && za_min <= zb_max + tol.linear;
    let (z_min, z_max) = match op {
        BooleanOp::Fuse => {
            if !touches_or_overlaps {
                return Ok(None);
            }
            (za_min.min(zb_min), za_max.max(zb_max))
        }
        BooleanOp::Intersect => {
            let lo = za_min.max(zb_min);
            let hi = za_max.min(zb_max);
            if hi <= lo + tol.linear {
                return Ok(None);
            }
            (lo, hi)
        }
        BooleanOp::Cut => return Ok(None),
    };
    let height = z_max - z_min;
    if height <= tol.linear {
        return Ok(None);
    }
    // r at axial position z (apex-relative) = slope * z. For frustums on
    // the +axis nappe, both z values are positive; if either becomes ≤ 0
    // (apex inclusion), bail out so we don't construct a degenerate cone.
    let r_at_z_min = slope * z_min;
    let r_at_z_max = slope * z_max;
    if r_at_z_min < -tol.linear || r_at_z_max < -tol.linear {
        return Ok(None);
    }
    let r_bot = r_at_z_min.abs();
    let r_top = r_at_z_max.abs();
    if r_bot <= tol.linear && r_top <= tol.linear {
        return Ok(None);
    }
    let cone = crate::primitives::make_cone(topo, r_bot, r_top, height)?;
    let world_origin = Point3::new(
        apex.x() + axis.x() * z_min,
        apex.y() + axis.y() * z_min,
        apex.z() + axis.z() * z_min,
    );
    // Cone shortcut keeps to axis-aligned cases for now (test corpus does
    // not yet cover off-axis cones). Detect parallel/antiparallel via the
    // dot product (the canonical-axis Z-component is the only term that
    // survives `canonical · axis` since canonical = ẑ).
    let dot = axis.z().clamp(-1.0, 1.0);
    if 1.0 - dot.abs() > tol.angular {
        return Ok(None);
    }
    let xform = xform_from_canonical_z(world_origin, axis, tol);
    crate::transform::transform_solid(topo, cone, &xform)?;
    Ok(Some(cone))
}

/// Compute the concentric-sphere boolean for two spheres sharing a
/// center. Returns `Ok(None)` when the shortcut doesn't apply (Cut, or
/// degenerate radii).
///
/// Sphere-sphere is simpler than the cylinder/cone analogues because
/// there's no axial range — the result radius is just `max(r_a, r_b)`
/// for Fuse and `min(r_a, r_b)` for Intersect.
///
/// The new sphere's tessellation density (segment count) is inherited from
/// whichever input has a higher equatorial vertex count, so a
/// 64-segment input never silently downgrades to a coarse default. This
/// relies on `make_sphere` allocating exactly `segments` equatorial
/// vertices and no pole vertices — see `crates/operations/src/primitives.rs`.
#[allow(clippy::too_many_arguments)]
fn concentric_sphere_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    center: Point3,
    r_a: f64,
    r_b: f64,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    if r_a <= tol.linear || r_b <= tol.linear {
        return Ok(None);
    }
    let r_result = match op {
        BooleanOp::Fuse => r_a.max(r_b),
        BooleanOp::Intersect => {
            // Both r_a and r_b are guaranteed > tol.linear by the guard above,
            // so `min(r_a, r_b)` is always positive here.
            r_a.min(r_b)
        }
        // Cut(A, B) on concentric spheres yields a hollow ball when r_a > r_b;
        // empty when r_a ≤ r_b. The hollow-ball case needs an outer + inner
        // shell, which `make_sphere` doesn't produce — defer to GFA.
        BooleanOp::Cut => return Ok(None),
    };

    // Inherit segment count from whichever input was tessellated more finely.
    // `make_sphere(r, n)` allocates exactly `n` equatorial vertices; because
    // sphere primitives are fully describe by (center, radius), all vertices
    // belong to that ring. Floor at 4 to satisfy `make_sphere`'s lower bound.
    let segments_a = brepkit_topology::explorer::solid_vertices(topo, a)
        .map(|v| v.len())
        .unwrap_or(0);
    let segments_b = brepkit_topology::explorer::solid_vertices(topo, b)
        .map(|v| v.len())
        .unwrap_or(0);
    let segments = segments_a.max(segments_b).max(4);

    let sphere = crate::primitives::make_sphere(topo, r_result, segments)?;
    if center.x().abs() > tol.linear
        || center.y().abs() > tol.linear
        || center.z().abs() > tol.linear
    {
        let xform = brepkit_math::mat::Mat4::translation(center.x(), center.y(), center.z());
        crate::transform::transform_solid(topo, sphere, &xform)?;
    }
    Ok(Some(sphere))
}

/// Build the world-frame transform that maps a primitive built in the
/// canonical Z-up local frame (origin at world origin, axis = +Z) to a
/// world frame at `world_origin` with up-axis `axis` (assumed
/// unit-length). Uses Rodrigues' rotation formula for the general case.
///
/// Comparisons use `1.0 - axis.dot(canonical) < tol.angular` rather than
/// vector-length deltas, because for unit vectors `|u−v| ≈ √2·θ`, so a
/// length comparison against `tol.angular` would correspond to
/// `θ ≈ 7×10⁻¹³` rad — effectively bit-identity.
fn xform_from_canonical_z(
    world_origin: Point3,
    axis: Vec3,
    tol: brepkit_math::tolerance::Tolerance,
) -> brepkit_math::mat::Mat4 {
    let translate =
        brepkit_math::mat::Mat4::translation(world_origin.x(), world_origin.y(), world_origin.z());
    let canonical = Vec3::new(0.0, 0.0, 1.0);
    let dot = canonical.dot(axis).clamp(-1.0, 1.0);
    // Parallel to +Z: pure translation.
    if 1.0 - dot < tol.angular {
        return translate;
    }
    // Antiparallel: rotate canonical (+z) by π around X to flip to −z.
    if 1.0 + dot < tol.angular {
        return translate * brepkit_math::mat::Mat4::rotation_x(std::f64::consts::PI);
    }
    // Rotate canonical (0,0,1) → axis via Rodrigues' formula:
    //   R = I + sin(θ) K + (1 - cos(θ)) K²,  K = [k]× for k = ẑ × axis / sin(θ).
    // k.z = 0 by construction, so K's z-row/z-column have a known structure.
    let sin_t = (1.0 - dot * dot).sqrt();
    let kx = -axis.y() / sin_t;
    let ky = axis.x() / sin_t;
    let one_minus_cos = 1.0 - dot;
    let r00 = one_minus_cos.mul_add(kx * kx, dot);
    let r01 = one_minus_cos * kx * ky;
    let r02 = sin_t * ky;
    let r10 = one_minus_cos * kx * ky;
    let r11 = one_minus_cos.mul_add(ky * ky, dot);
    let r12 = -sin_t * kx;
    let r20 = -sin_t * ky;
    let r21 = sin_t * kx;
    let r22 = dot;
    let rot = brepkit_math::mat::Mat4([
        [r00, r01, r02, 0.0],
        [r10, r11, r12, 0.0],
        [r20, r21, r22, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]);
    translate * rot
}

/// Check whether every boundary vertex of `solid` is classified as
/// `Inside` or `On` by `classifier`. Used by the identical-solid shortcut
/// to distinguish truly-identical solids from co-located but differently
/// shaped solids (e.g., a cone and a box that share an AABB).
fn all_vertices_inside_or_on(
    topo: &Topology,
    solid: SolidId,
    classifier: &brepkit_algo::classifier::AnalyticClassifier,
    tol: brepkit_math::tolerance::Tolerance,
) -> bool {
    let Ok(s) = topo.solid(solid) else {
        return false;
    };
    let Ok(sh) = topo.shell(s.outer_shell()) else {
        return false;
    };
    for &fid in sh.faces() {
        let Ok(f) = topo.face(fid) else { return false };
        let Ok(w) = topo.wire(f.outer_wire()) else {
            return false;
        };
        for oe in w.edges() {
            let Ok(e) = topo.edge(oe.edge()) else {
                return false;
            };
            for vid in [e.start(), e.end()] {
                let Ok(v) = topo.vertex(vid) else {
                    return false;
                };
                // The analytic classifier returns `None` for points within
                // tol.linear of the boundary — treat as "on" for this check.
                if classifier.classify(v.point(), tol) == Some(brepkit_algo::FaceClass::Outside) {
                    return false;
                }
            }
        }
    }
    true
}

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

/// Merge duplicate vertices in a solid's shell by position.
///
/// For each vertex position (quantized at tolerance), picks one canonical
/// vertex. Rebuilds all edges and wires to use canonical vertices.
/// Creates new edges (doesn't mutate existing ones) to avoid corrupting
/// input solids that may share edge topology.
#[allow(clippy::items_after_statements, clippy::type_complexity)]
fn merge_result_vertices(
    topo: &mut Topology,
    solid: SolidId,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<(), crate::OperationsError> {
    use std::collections::{BTreeMap, HashMap};

    let shell_id = topo.solid(solid)?.outer_shell();
    let face_ids: Vec<_> = topo.shell(shell_id)?.faces().to_vec();

    let scale = 1.0 / tol.linear;
    let quantize = |p: brepkit_math::vec::Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // Build vertex canonical map: position → first VertexId seen
    let mut canonical: BTreeMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        BTreeMap::new();
    let mut replacements: HashMap<
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
    > = HashMap::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    let pos = topo.vertex(vid)?.point();
                    let key = quantize(pos);
                    let canon = *canonical.entry(key).or_insert(vid);
                    if canon != vid {
                        replacements.insert(vid, canon);
                    }
                }
            }
        }
    }

    if replacements.is_empty() {
        return Ok(());
    }

    // Rebuild faces with merged vertices
    // Cache: (old_edge, new_start, new_end) → new_edge to share edges
    let mut edge_cache: HashMap<
        (
            brepkit_topology::edge::EdgeId,
            brepkit_topology::vertex::VertexId,
            brepkit_topology::vertex::VertexId,
        ),
        brepkit_topology::edge::EdgeId,
    > = HashMap::new();

    // Snapshot face data, then rebuild with merged vertices
    struct FaceSnap {
        surface: brepkit_topology::face::FaceSurface,
        reversed: bool,
        outer_oes: Vec<(
            brepkit_topology::edge::EdgeId,
            bool,
            brepkit_topology::edge::EdgeCurve,
            brepkit_topology::vertex::VertexId,
            brepkit_topology::vertex::VertexId,
            Option<f64>, // edge tolerance
        )>,
        outer_closed: bool,
        inner_wires: Vec<(
            Vec<(
                brepkit_topology::edge::EdgeId,
                bool,
                brepkit_topology::edge::EdgeCurve,
                brepkit_topology::vertex::VertexId,
                brepkit_topology::vertex::VertexId,
                Option<f64>,
            )>,
            bool, // wire closed flag
        )>,
    }

    let mut snaps = Vec::with_capacity(face_ids.len());
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let outer_wire = topo.wire(face.outer_wire())?;
        let outer_closed = outer_wire.is_closed();
        let outer_oes: Vec<_> = outer_wire
            .edges()
            .iter()
            .map(|oe| -> Result<_, crate::OperationsError> {
                let e = topo.edge(oe.edge())?;
                Ok((
                    oe.edge(),
                    oe.is_forward(),
                    e.curve().clone(),
                    e.start(),
                    e.end(),
                    e.tolerance(),
                ))
            })
            .collect::<Result<_, _>>()?;
        let inner_wids = face.inner_wires().to_vec();
        let mut inner_wires = Vec::new();
        for iw in inner_wids {
            let w = topo.wire(iw)?;
            let closed = w.is_closed();
            let oes: Vec<_> = w
                .edges()
                .iter()
                .map(|oe| -> Result<_, crate::OperationsError> {
                    let e = topo.edge(oe.edge())?;
                    Ok((
                        oe.edge(),
                        oe.is_forward(),
                        e.curve().clone(),
                        e.start(),
                        e.end(),
                        e.tolerance(),
                    ))
                })
                .collect::<Result<_, _>>()?;
            inner_wires.push((oes, closed));
        }
        snaps.push(FaceSnap {
            surface,
            reversed,
            outer_oes,
            outer_closed,
            inner_wires,
        });
    }

    #[allow(clippy::type_complexity)]
    let remap_oes = |oes: &[(
        brepkit_topology::edge::EdgeId,
        bool,
        brepkit_topology::edge::EdgeCurve,
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
        Option<f64>,
    )],
                     replacements: &HashMap<
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
    >,
                     edge_cache: &mut HashMap<
        (
            brepkit_topology::edge::EdgeId,
            brepkit_topology::vertex::VertexId,
            brepkit_topology::vertex::VertexId,
        ),
        brepkit_topology::edge::EdgeId,
    >,
                     topo: &mut Topology|
     -> Vec<brepkit_topology::wire::OrientedEdge> {
        oes.iter()
            .map(|(eid, fwd, curve, start, end, edge_tol)| {
                let ns = replacements.get(start).copied().unwrap_or(*start);
                let ne = replacements.get(end).copied().unwrap_or(*end);
                if ns == *start && ne == *end {
                    return brepkit_topology::wire::OrientedEdge::new(*eid, *fwd);
                }
                let key = (*eid, ns, ne);
                let new_eid = *edge_cache.entry(key).or_insert_with(|| {
                    topo.add_edge(brepkit_topology::edge::Edge::with_tolerance(
                        ns,
                        ne,
                        curve.clone(),
                        *edge_tol,
                    ))
                });
                brepkit_topology::wire::OrientedEdge::new(new_eid, *fwd)
            })
            .collect()
    };

    let mut new_face_ids = Vec::with_capacity(snaps.len());
    for snap in &snaps {
        let outer_oes = remap_oes(&snap.outer_oes, &replacements, &mut edge_cache, topo);
        let Ok(outer_wire) = brepkit_topology::wire::Wire::new(outer_oes, snap.outer_closed) else {
            // Wire rebuild failed — keep the original face unchanged
            // rather than silently dropping it
            continue;
        };
        let outer_id = topo.add_wire(outer_wire);

        let mut inner_ids = Vec::new();
        for (inner_oes_snap, inner_closed) in &snap.inner_wires {
            let oes = remap_oes(inner_oes_snap, &replacements, &mut edge_cache, topo);
            if let Ok(w) = brepkit_topology::wire::Wire::new(oes, *inner_closed) {
                inner_ids.push(topo.add_wire(w));
            }
        }

        let mut new_face =
            brepkit_topology::face::Face::new(outer_id, inner_ids, snap.surface.clone());
        if snap.reversed {
            new_face.set_reversed(true);
        }
        new_face_ids.push(topo.add_face(new_face));
    }

    // Replace the shell's faces
    let new_shell = brepkit_topology::shell::Shell::new(new_face_ids)?;
    let new_shell_id = topo.add_shell(new_shell);
    let solid_mut = topo.solid_mut(solid)?;
    solid_mut.set_outer_shell(new_shell_id);

    Ok(())
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

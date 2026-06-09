//! Boolean operations on solids: fuse, cut, and intersect.
//!
//! Uses the GFA pipeline (`brepkit_algo::gfa`) as the primary boolean engine,
//! with mesh boolean (co-refinement) as a fallback when GFA fails or produces
//! invalid results.

pub mod assembly;
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
        // ALL 3 dims. Used as the no-classifier fallback to detect true
        // nested containment (e.g., a ring fully inside a shell's cavity)
        // without false-positives on sparse multi-shell solids (e.g., a
        // fuse of disjoint boxes whose AABB technically encloses another
        // solid's AABB while mostly being empty space).
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
                    .all(|(outer_d, inner_d)| *outer_d > *inner_d * 1.1)
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
        // inside-or-on A AND A's AABB encloses B's. Falls back to a
        // strict AABB-only check when the containing solid has no
        // classifier — the strict check requires ≥10% larger in ALL
        // three dims so that sparse multi-shell solids (e.g., a fuse
        // of two disjoint boxes) don't false-positive as "contains
        // another solid".
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
                BooleanOp::Cut => Err(crate::OperationsError::EmptyResult {
                    reason: "Cut of identical solids".into(),
                }),
            };
        }
        // Containment shortcuts:
        // - Fuse/Intersect with either containment direction: copy the
        //   appropriate solid.
        // - Cut with A ⊆ B: result is empty (A is fully removed). Return
        //   EmptyResult explicitly — without this short-circuit, GFA falls
        //   back to producing a degenerate vol=0 solid that callers
        //   mistake for a real result, breaking volume invariants like
        //   `vol((A-B) ∪ (A∩B)) = vol(A)`.
        // - Cut with B ⊂ A: defer to GFA (produces hollow solid).
        if op == BooleanOp::Cut && a_in_b && !b_in_a {
            return Err(crate::OperationsError::EmptyResult {
                reason: "Cut with target fully contained in tool".into(),
            });
        }
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

        // Box-sphere intersect shortcut: when one input classifies as an
        // axis-aligned `Box` and the other as a `Sphere`, the Intersect
        // result has a closed analytic form in two common cases:
        //   - sphere fully inside box → result is a copy of the sphere
        //   - exactly 3 of the 6 box planes cut the sphere (their meeting
        //     corner sits at or inside the sphere) → spherical "octant"
        //     bounded by 3 quarter-disc box sub-faces + 1 spherical patch
        // Other configurations fall through to GFA.
        //
        // Cut/Fuse aren't covered here yet — they need outer/inner shell
        // construction (Cut: box with spherical hole) or full periodic-
        // sphere handling (Fuse: box with spherical bulge), both larger
        // than this shortcut warrants.
        if op == BooleanOp::Intersect {
            let (box_args, sphere_args) = match (ca.as_ref(), cb.as_ref()) {
                (
                    Some(brepkit_algo::classifier::AnalyticClassifier::Box {
                        min: bmin,
                        max: bmax,
                    }),
                    Some(brepkit_algo::classifier::AnalyticClassifier::Sphere { center, radius }),
                ) => (Some((*bmin, *bmax)), Some((*center, *radius))),
                (
                    Some(brepkit_algo::classifier::AnalyticClassifier::Sphere { center, radius }),
                    Some(brepkit_algo::classifier::AnalyticClassifier::Box {
                        min: bmin,
                        max: bmax,
                    }),
                ) => (Some((*bmin, *bmax)), Some((*center, *radius))),
                _ => (None, None),
            };
            if let (Some((bmin, bmax)), Some((sc, sr))) = (box_args, sphere_args) {
                let segs = brepkit_topology::explorer::solid_vertices(topo, a)
                    .map(|v| v.len())
                    .unwrap_or(0)
                    .max(
                        brepkit_topology::explorer::solid_vertices(topo, b)
                            .map(|v| v.len())
                            .unwrap_or(0),
                    )
                    .max(16);
                if let Some(result) =
                    box_sphere_intersect_shortcut(topo, bmin, bmax, sc, sr, segs, tol)?
                {
                    return Ok(result);
                }
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

        // Coaxial-torus merge shortcut: when both A and B classify as Torus
        // with the same center, axis (parallel/antiparallel), and major
        // radius, Fuse and Intersect collapse to a single torus by minor
        // radius algebra. Same family as the concentric-sphere shortcut
        // above; sidesteps GFA's torus same-domain handling for the
        // common shared-major case.
        if let (
            Some(brepkit_algo::classifier::AnalyticClassifier::Torus {
                center: ca_center,
                axis: aa,
                major_radius: maj_a,
                minor_radius: min_a,
            }),
            Some(brepkit_algo::classifier::AnalyticClassifier::Torus {
                center: cb_center,
                axis: ab,
                major_radius: maj_b,
                minor_radius: min_b,
            }),
        ) = (ca.as_ref(), cb.as_ref())
        {
            let coincident = (*ca_center - *cb_center).length() < tol.linear;
            // Allow either axis orientation — a torus with axis +z is the
            // same surface as the same torus with axis -z (the small-circle
            // sweep is symmetric about the central plane).
            let coaxial = aa.dot(*ab).abs() > 1.0 - tol.angular;
            let same_major = (maj_a - maj_b).abs() < tol.linear;
            if coincident && coaxial && same_major {
                if let Some(result) = coaxial_torus_shortcut(
                    topo, op, a, b, *ca_center, *aa, *maj_a, *min_a, *min_b, tol,
                )? {
                    return Ok(result);
                }
            }
        }
    }

    // ── Broad-phase disjoint intersect ───────────────────────────────
    // If the curvature-aware AABBs of A and B are separated on any axis
    // by more than linear tolerance, the solids provably do not overlap
    // and their intersection is the empty set. Containment shortcuts have
    // already run above (a contained solid has overlapping, not separated,
    // AABBs), so reaching here with separated boxes is an exact witness.
    // The boxes are conservative outer bounds, so box non-overlap implies
    // solid non-overlap. Symmetric in A and B by construction.
    if op == BooleanOp::Intersect {
        let bb_a = crate::measure::solid_bounding_box(topo, a).ok();
        let bb_b = crate::measure::solid_bounding_box(topo, b).ok();
        if let Some((a_box, b_box)) = bb_a.zip(bb_b) {
            if aabbs_separated(&a_box, &b_box, tol.linear) {
                return Ok(topo.add_empty_solid());
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
            // Narrow-phase empty intersect: overlapping AABBs but the engine
            // selected no faces for the common region (e.g. boxes whose boxes
            // overlap by tolerance but whose interiors do not). This is the
            // authoritative witness of an empty intersection.
            if op == BooleanOp::Intersect && result_faces == 0 {
                log::info!(
                    "GFA intersect empty in {:.1}ms (no common faces)",
                    timer_elapsed_ms(gfa_start)
                );
                return Ok(topo.add_empty_solid());
            }
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

                // Run unify_faces if Euler is off (existing condition) OR if
                // the topology already has 3+-face junctions, which can occur
                // with Euler==2 when overlapping coplanar faces cancel in V-E+F
                // counting. The same-domain detection in the assembler only
                // pairs faces across opposing ranks with identical edge sets,
                // so within-rank or different-boundary overlaps slip through;
                // unify_faces is the safety net for those (issue #696).
                let needs_unify = euler_pre2 != 2 || !is_closed_manifold(topo, result)?;
                if needs_unify {
                    for _ in 0..3 {
                        if crate::heal::unify_faces(topo, result)? == 0 {
                            break;
                        }
                    }
                }
                let (f, e, v) = brepkit_topology::explorer::solid_entity_counts(topo, result)?;
                #[allow(clippy::cast_possible_wrap)]
                let euler = (v as i64) - (e as i64) + (f as i64);
                // Free edges in an Intersect result mean faces were dropped
                // (e.g. a tolerance-thin sliver kept only some of its
                // bounding faces) — reject even when Euler accidentally
                // balances. Cut and Fuse keep the legacy lenient gate: some
                // coplanar cut/fuse results carry boundary edges yet are
                // still the best available output (the mesh fallback loses
                // more volume than the open GFA shell does).
                let open_shell_ok = op != BooleanOp::Intersect || !has_free_edges(topo, result)?;
                // Hole-aware Euler gate: a face with L inner wire loops raises
                // V-E+F by L (Euler-Poincare: V-E+F-L = 2(1-g)), so a valid
                // genus-0 result with holed faces (e.g. a fuse leaving circular
                // holes in box faces) has euler = 2 + L. Naive euler == 2 would
                // reject it. The holed acceptance additionally requires a
                // closed manifold so that accidental cancellations (open shells
                // whose missing faces offset the inner-wire surplus) still
                // fail safe to the mesh fallback.
                let inner_wire_count = solid_inner_wire_count(topo, result)?;
                let euler_ok = euler == 2
                    || (inner_wire_count > 0
                        && euler - inner_wire_count == 2
                        && is_closed_manifold(topo, result)?);
                if euler_ok && open_shell_ok && validate_boolean_result(topo, result).is_ok() {
                    log::info!(
                        "GFA boolean succeeded in {:.1}ms ({result_faces} faces)",
                        timer_elapsed_ms(gfa_start)
                    );
                    return Ok(result);
                }
                // Multi-region manifold result (e.g., a Cut that splits a
                // solid into N spatially-disjoint pieces). N independently
                // closed manifolds have combined Euler = 2*N. Falling back
                // to mesh boolean would collapse the disjoint pieces into
                // a single region's volume (the `cut with simplify`
                // returning vol 166 instead of 1000 symptom).
                //
                // Gate: every edge must be shared by exactly 2 faces
                // (closed-manifold) AND the components must be pairwise
                // spatially disjoint (AABBs do not overlap). The latter
                // distinguishes a "cut into N pieces" from a hollow solid
                // (outer surface + cavity surface — same number of
                // components, same Euler relation, but AABBs overlap).
                let components_vec = crate::boolean::assembly::face_components(topo, result);
                let components = components_vec.len();
                #[allow(clippy::cast_possible_wrap)]
                let expected_euler = (components as i64) * 2;
                // For Cut, also verify no component is a "B-interior piece" —
                // GFA can produce N closed manifolds where one of them is the
                // tool's interior (sphere - cylinder example: 3 pieces =
                // top cap + bottom cap + cylinder interior). Sample a point
                // inside each component's AABB and classify against B; if any
                // sits inside B, the GFA result included the cut-out piece
                // and should be rejected. Fuse/Intersect don't have this
                // failure mode.
                let cut_safe = op != BooleanOp::Cut
                    || brepkit_algo::classifier::try_build_analytic_classifier(topo, b)
                        .as_ref()
                        .is_none_or(|cls_b| {
                            all_component_centers_outside(topo, &components_vec, cls_b, tol)
                        });
                if op == BooleanOp::Cut
                    && components >= 2
                    && euler == expected_euler
                    && components_are_disjoint_pieces(topo, &components_vec)
                    && cut_safe
                    && is_closed_manifold(topo, result).unwrap_or(false)
                    && validate_boolean_result(topo, result).is_ok()
                {
                    log::info!(
                        "GFA multi-region succeeded in {:.1}ms ({result_faces} faces, {components} pieces)",
                        timer_elapsed_ms(gfa_start)
                    );
                    return Ok(result);
                }
            }
            log::warn!(
                "GFA result failed validation in {:.1}ms (faces={result_faces}), falling back",
                timer_elapsed_ms(gfa_start)
            );
        }
        Err(e) => {
            log::warn!(
                "GFA boolean failed in {:.1}ms ({e}), falling back",
                timer_elapsed_ms(gfa_start)
            );
        }
    }

    // ── Multi-region input fallback (Cut only) ───────────────────────
    // When the input solid carries multiple disjoint pieces (a previous
    // cut split a solid into N parts), GFA's pavefiller can't process
    // them together — feeding the whole thing in loses regions. Splitting
    // into per-component cuts and recombining preserves the missing
    // pieces. Cut distributes over disjoint union; Fuse/Intersect have
    // more complex interaction semantics so we leave those to mesh.
    if op == BooleanOp::Cut {
        let components = crate::boolean::assembly::face_components(topo, a);
        if components.len() >= 2 && components_are_disjoint_pieces(topo, &components) {
            if let Ok(result) = cut_multi_region_input(topo, a, b, components.len()) {
                return Ok(result);
            }
        }
    }

    // ── Mesh boolean fallback (no recursion) ─────────────────────────
    let opts = BooleanOptions::default();
    let raw = match mesh_boolean_fallback(topo, op, a, b, opts.deflection, tol, &opts) {
        Ok(raw) => raw,
        // An empty mesh-boolean output for an intersect means the common
        // region is empty — return the empty-result sentinel rather than
        // surfacing the empty set as an error.
        Err(crate::OperationsError::EmptyResult { .. }) if op == BooleanOp::Intersect => {
            return Ok(topo.add_empty_solid());
        }
        Err(e) => return Err(e),
    };
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
        // Collect every input face whose normal+centroid is "close enough"
        // to the output face. When fuse simplifies two coincident-plane
        // inputs into one output (e.g. fuse of touching boxes via the
        // box-pair shortcut: the top faces of A and B both collapse onto
        // the result's top face), picking only the single best-scoring
        // input drops the other input's metadata. Recording every
        // qualifying input under the same output preserves both origins.
        let mut best_score = f64::NEG_INFINITY;
        let mut matches: Vec<(usize, f64)> = Vec::new();

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

            let score = dot - dist_sq / centroid_dist_sq_max;
            if score > best_score {
                best_score = score;
            }
            matches.push((in_idx, score));
        }

        if matches.is_empty() {
            unmatched_outputs.push((out_idx, out_normal, out_centroid));
            continue;
        }

        // Accept any match within a small tolerance of the best score —
        // ties (or near-ties) usually indicate two input faces that
        // legitimately contributed to the same output (e.g., the two
        // halves of a face that merged across a same-domain boundary).
        let score_tol = 0.05;
        for &(in_idx, score) in &matches {
            if score >= best_score - score_tol {
                evo.add_modified(in_idx, out_idx);
                matched_inputs.insert(in_idx);
            }
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
        BooleanOp::Cut => {
            // Cut shortcut: when B spans A in 2 of 3 dims (≥ A's extent
            // on both sides) and overlaps in the third, the result is
            // up-to-2 axis-aligned boxes (the leftover slabs on either
            // side of B in the cutting dim). This avoids routing through
            // GFA's same-domain handling which currently mishandles the
            // 4-coincident-face case (target's lateral walls + tool's
            // matching walls).
            return box_pair_cut_shortcut(topo, a_min, a_max, b_min, b_max, eps);
        }
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

/// Cut shortcut for two axis-aligned boxes: returns the leftover
/// portion(s) when B slices through A in one dimension while spanning
/// A in the other two dimensions. The result is 0, 1, or 2 axis-aligned
/// boxes packaged into a single multi-region Solid.
///
/// Returns `Ok(None)` when the shortcut doesn't fit — e.g., B doesn't
/// span A in any 2 dims, B touches only a corner, etc. The general path
/// (GFA) handles those cases.
fn box_pair_cut_shortcut(
    topo: &mut Topology,
    a_min: Point3,
    a_max: Point3,
    b_min: Point3,
    b_max: Point3,
    eps: f64,
) -> Result<Option<SolidId>, crate::OperationsError> {
    // B must span A in 2 of 3 dims (B_min ≤ A_min - eps AND B_max ≥ A_max + eps,
    // i.e., B's extent covers A's extent in that dim).
    let x_spans = b_min.x() <= a_min.x() + eps && b_max.x() >= a_max.x() - eps;
    let y_spans = b_min.y() <= a_min.y() + eps && b_max.y() >= a_max.y() - eps;
    let z_spans = b_min.z() <= a_min.z() + eps && b_max.z() >= a_max.z() - eps;
    let spans_count = u8::from(x_spans) + u8::from(y_spans) + u8::from(z_spans);
    if spans_count != 2 {
        return Ok(None);
    }
    // In the non-spanning dim, B must actually intersect A.
    let (a_lo, a_hi, b_lo, b_hi) = if !x_spans {
        (a_min.x(), a_max.x(), b_min.x(), b_max.x())
    } else if !y_spans {
        (a_min.y(), a_max.y(), b_min.y(), b_max.y())
    } else {
        (a_min.z(), a_max.z(), b_min.z(), b_max.z())
    };
    if b_hi <= a_lo + eps || b_lo >= a_hi - eps {
        return Ok(None);
    }

    // Build the leftover slabs. There are 0, 1, or 2 pieces depending on
    // whether B extends past A on each side in the cutting dim.
    let cuts: Vec<(f64, f64)> = {
        let mut pieces = Vec::with_capacity(2);
        if b_lo > a_lo + eps {
            pieces.push((a_lo, b_lo)); // slab before B
        }
        if b_hi < a_hi - eps {
            pieces.push((b_hi, a_hi)); // slab after B
        }
        pieces
    };
    if cuts.is_empty() {
        // B fully covers A in the cutting dim → cut leaves nothing.
        // Let the general path handle this (it errors).
        return Ok(None);
    }

    let piece_solids: Vec<SolidId> = cuts
        .iter()
        .map(|&(lo, hi)| -> Result<SolidId, crate::OperationsError> {
            let (dx, dy, dz, tx, ty, tz) = if !x_spans {
                (
                    hi - lo,
                    a_max.y() - a_min.y(),
                    a_max.z() - a_min.z(),
                    lo,
                    a_min.y(),
                    a_min.z(),
                )
            } else if !y_spans {
                (
                    a_max.x() - a_min.x(),
                    hi - lo,
                    a_max.z() - a_min.z(),
                    a_min.x(),
                    lo,
                    a_min.z(),
                )
            } else {
                (
                    a_max.x() - a_min.x(),
                    a_max.y() - a_min.y(),
                    hi - lo,
                    a_min.x(),
                    a_min.y(),
                    lo,
                )
            };
            let bx = crate::primitives::make_box(topo, dx, dy, dz)?;
            if tx.abs() > eps || ty.abs() > eps || tz.abs() > eps {
                let xform = brepkit_math::mat::Mat4::translation(tx, ty, tz);
                crate::transform::transform_solid(topo, bx, &xform)?;
            }
            Ok(bx)
        })
        .collect::<Result<_, _>>()?;

    if piece_solids.len() == 1 {
        return Ok(Some(piece_solids[0]));
    }

    // Combine pieces into a single multi-region solid.
    let mut all_faces: Vec<brepkit_topology::face::FaceId> = Vec::new();
    for &p in &piece_solids {
        let p_data = topo.solid(p)?;
        for &fid in topo.shell(p_data.outer_shell())?.faces() {
            all_faces.push(fid);
        }
    }
    Ok(Some(make_solid_from_face_subset(topo, &all_faces)?))
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
/// Box-sphere `Intersect` shortcut. Handles two configurations exactly,
/// returning `Ok(None)` to fall through to GFA otherwise:
///
/// 1. **Sphere fully inside box** — every box face plane has the sphere
///    on the box-interior side with margin ≥ `R` (`s ≤ -R + eps`). The
///    result is a fresh sphere primitive at `sphere_center` with radius
///    `sphere_radius`.
/// 2. **Spherical "octant"** — exactly 3 of the 6 box face planes cut
///    the sphere (`|s| < R - eps`) and the other 3 leave the sphere on
///    the box-interior side. The 3 cutting planes are mutually orthogonal
///    (axis-aligned box invariant) and meet at a single box corner `O`.
///    The result is the sphere region in the box-interior octant of `O`,
///    bounded by 3 quarter-disc box sub-faces and 1 spherical patch.
///
/// `s` is the signed distance from `sphere_center` to a face plane along
/// the face's outward normal (positive = sphere on box-exterior side).
/// If any face has `s ≥ R - eps` the result is empty (sphere doesn't
/// reach into the box from that side) — we return `None` rather than an
/// empty solid so the caller can produce the canonical `EmptyResult`
/// error via the regular path.
#[allow(clippy::too_many_arguments)]
fn box_sphere_intersect_shortcut(
    topo: &mut Topology,
    box_min: Point3,
    box_max: Point3,
    sphere_center: Point3,
    sphere_radius: f64,
    sphere_segments: usize,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    let r = sphere_radius;
    let eps = tol.linear;
    if r <= eps {
        return Ok(None);
    }
    // Sanity: degenerate or inverted box.
    if box_max.x() <= box_min.x() + eps
        || box_max.y() <= box_min.y() + eps
        || box_max.z() <= box_min.z() + eps
    {
        return Ok(None);
    }

    // For each of 6 box face planes, compute `s` (signed distance from
    // sphere center along outward normal). Classify each plane.
    let faces: [(Vec3, f64); 6] = [
        (Vec3::new(-1.0, 0.0, 0.0), -box_min.x()),
        (Vec3::new(1.0, 0.0, 0.0), box_max.x()),
        (Vec3::new(0.0, -1.0, 0.0), -box_min.y()),
        (Vec3::new(0.0, 1.0, 0.0), box_max.y()),
        (Vec3::new(0.0, 0.0, -1.0), -box_min.z()),
        (Vec3::new(0.0, 0.0, 1.0), box_max.z()),
    ];
    let signed_dist = |n: Vec3, d: f64| -> f64 {
        n.x() * sphere_center.x() + n.y() * sphere_center.y() + n.z() * sphere_center.z() - d
    };

    let mut cuts: Vec<usize> = Vec::new();
    for (i, &(n, d)) in faces.iter().enumerate() {
        let s = signed_dist(n, d);
        if s >= r - eps {
            // Sphere is fully on the exterior side of this plane → box ∩
            // sphere = empty. Defer to GFA which will surface an
            // EmptyResult error in its usual form.
            return Ok(None);
        }
        if s.abs() < r - eps {
            cuts.push(i);
        }
        // else: s ≤ -r + eps → sphere fully inside this plane, face
        // doesn't bound the result; nothing to do.
    }

    // Case 1: sphere fully inside box (no cutting planes).
    if cuts.is_empty() {
        let sphere = crate::primitives::make_sphere(topo, r, sphere_segments)?;
        if sphere_center.x().abs() > eps
            || sphere_center.y().abs() > eps
            || sphere_center.z().abs() > eps
        {
            let xform = brepkit_math::mat::Mat4::translation(
                sphere_center.x(),
                sphere_center.y(),
                sphere_center.z(),
            );
            crate::transform::transform_solid(topo, sphere, &xform)?;
        }
        return Ok(Some(sphere));
    }

    // Case 2: 3 cutting planes meeting at a box corner → spherical
    // octant. The 3 cut planes' outward normals are mutually orthogonal
    // (axis-aligned box invariant) so the in-box direction perpendicular
    // to each is the negated outward normal.
    if cuts.len() == 3 {
        return build_box_sphere_octant(topo, &faces, &cuts, sphere_center, r, tol);
    }

    // 1, 2, 4, 5, 6 cutting planes — more complex geometries (caps,
    // lenses, etc.). Out of scope for this shortcut; fall through.
    Ok(None)
}

/// Construct the result of `box ∩ sphere` when exactly 3 box face planes
/// cut the sphere and meet at a single corner `O`. The result topology
/// is 4 faces (3 quarter-discs + 1 spherical patch), 6 edges, 4 vertices.
fn build_box_sphere_octant(
    topo: &mut Topology,
    faces: &[(Vec3, f64); 6],
    cuts: &[usize],
    sphere_center: Point3,
    r: f64,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    // Cutting plane normals + their box-plane-d values.
    let cut_planes: Vec<(Vec3, f64)> = cuts.iter().map(|&i| faces[i]).collect();
    // The 3 outward normals must be mutually orthogonal (axis-aligned box).
    let n0 = cut_planes[0].0;
    let n1 = cut_planes[1].0;
    let n2 = cut_planes[2].0;
    if n0.dot(n1).abs() > tol.angular
        || n0.dot(n2).abs() > tol.angular
        || n1.dot(n2).abs() > tol.angular
    {
        // Not orthogonal — defer to GFA.
        return Ok(None);
    }
    // The corner O is at the intersection of the 3 cutting planes:
    //   n_i · O = d_i  for all 3 i.
    // Since the normals are axis-aligned (±x, ±y, ±z), we can pull each
    // coordinate of O directly off the matching plane's d.
    let coord_from_axis = |axis: Vec3, d: f64| -> f64 {
        if axis.x().abs() > 0.5 {
            d * axis.x().signum()
        } else if axis.y().abs() > 0.5 {
            d * axis.y().signum()
        } else {
            d * axis.z().signum()
        }
    };
    let mut o = [0.0_f64; 3];
    for &(n, d) in &cut_planes {
        if n.x().abs() > 0.5 {
            o[0] = coord_from_axis(n, d);
        } else if n.y().abs() > 0.5 {
            o[1] = coord_from_axis(n, d);
        } else {
            o[2] = coord_from_axis(n, d);
        }
    }
    let o = Point3::new(o[0], o[1], o[2]);

    // In-box direction perpendicular to each cutting plane = -n_i.
    let in_dirs: Vec<Vec3> = cut_planes
        .iter()
        .map(|&(n, _)| Vec3::new(-n.x(), -n.y(), -n.z()))
        .collect();

    // For each cutting plane i, the box edge from O in direction in_dirs[i]
    // is the intersection of the other two cutting planes. Find the sphere
    // intersection with this edge — the vertex on the sphere along the box
    // edge.
    //
    // Edge parameterised as O + t·d_i for t ≥ 0. Sphere: |P - C|² = R².
    //   (O + t·d_i - C) · (O + t·d_i - C) = R²
    //   Let v = O - C; expand:
    //     t² + 2 t (v · d_i) + |v|² - R² = 0
    //   So t = -v·d_i ± sqrt((v·d_i)² - |v|² + R²)
    let mut sphere_pts: [Point3; 3] = [Point3::new(0.0, 0.0, 0.0); 3];
    for (idx, &dir) in in_dirs.iter().enumerate() {
        let vx = o.x() - sphere_center.x();
        let vy = o.y() - sphere_center.y();
        let vz = o.z() - sphere_center.z();
        let v_dot_d = vx * dir.x() + vy * dir.y() + vz * dir.z();
        let v_sq = vx * vx + vy * vy + vz * vz;
        let disc = v_dot_d * v_dot_d - v_sq + r * r;
        if disc < -tol.linear * tol.linear {
            return Ok(None);
        }
        let t = -v_dot_d + disc.max(0.0).sqrt();
        if t <= tol.linear {
            return Ok(None);
        }
        sphere_pts[idx] = Point3::new(
            o.x() + t * dir.x(),
            o.y() + t * dir.y(),
            o.z() + t * dir.z(),
        );
    }

    // Topology: 4 vertices, 6 edges, 4 faces.
    let v_o = topo.add_vertex(Vertex::new(o, tol.linear));
    let v_x = topo.add_vertex(Vertex::new(sphere_pts[0], tol.linear));
    let v_y = topo.add_vertex(Vertex::new(sphere_pts[1], tol.linear));
    let v_z = topo.add_vertex(Vertex::new(sphere_pts[2], tol.linear));

    // 3 line edges from O along the box edges.
    let e_ox = topo.add_edge(Edge::new(v_o, v_x, EdgeCurve::Line));
    let e_oy = topo.add_edge(Edge::new(v_o, v_y, EdgeCurve::Line));
    let e_oz = topo.add_edge(Edge::new(v_o, v_z, EdgeCurve::Line));

    // 3 arc edges on the sphere. Each arc lies on one of the cutting planes:
    // the arc opposite vertex `i` (i.e., between the other two vertices)
    // sits on cutting plane `i` (normal `n_i`), because those two vertices
    // lie on edges perpendicular to the remaining two normals — and both
    // of those edges lie within the plane perpendicular to `n_i`.
    let mut build_arc_edge =
        |n: Vec3,
         p_start: Point3,
         p_end: Point3,
         start_vid,
         end_vid|
         -> Result<brepkit_topology::edge::EdgeId, crate::OperationsError> {
            let dist = n.x() * (sphere_center.x() - p_start.x())
                + n.y() * (sphere_center.y() - p_start.y())
                + n.z() * (sphere_center.z() - p_start.z());
            let circle_center = Point3::new(
                sphere_center.x() - dist * n.x(),
                sphere_center.y() - dist * n.y(),
                sphere_center.z() - dist * n.z(),
            );
            let circle_r = (r * r - dist * dist).max(0.0).sqrt();
            if circle_r <= tol.linear {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "box-sphere octant: degenerate arc radius".into(),
                });
            }
            let dx = p_start.x() - circle_center.x();
            let dy = p_start.y() - circle_center.y();
            let dz = p_start.z() - circle_center.z();
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            if len <= tol.linear {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "box-sphere octant: degenerate arc reference".into(),
                });
            }
            let u_ref = Vec3::new(dx / len, dy / len, dz / len);
            let circle =
                Circle3D::new_with_ref(circle_center, n, circle_r, u_ref).map_err(|e| {
                    crate::OperationsError::InvalidInput {
                        reason: format!("box-sphere octant: circle construction failed: {e}"),
                    }
                })?;
            let _ = p_end; // p_end is used only via end_vid (already pre-placed at the correct sphere point)
            Ok(topo.add_edge(Edge::new(start_vid, end_vid, EdgeCurve::Circle(circle))))
        };

    // Arc on cut plane 0 (between v_y and v_z, i.e., the edge "opposite" v_x).
    let arc_yz = build_arc_edge(n0, sphere_pts[1], sphere_pts[2], v_y, v_z)?;
    // Arc on cut plane 1 (between v_z and v_x).
    let arc_zx = build_arc_edge(n1, sphere_pts[2], sphere_pts[0], v_z, v_x)?;
    // Arc on cut plane 2 (between v_x and v_y).
    let arc_xy = build_arc_edge(n2, sphere_pts[0], sphere_pts[1], v_x, v_y)?;

    // Quarter-disc face on cut plane 0 (perpendicular to n0): bounded by
    // box edges O-Y and O-Z + arc Y→Z.
    let qd0_wire = Wire::new(
        vec![
            OrientedEdge::new(e_oy, true),   // O → Y
            OrientedEdge::new(arc_yz, true), // Y → Z (arc)
            OrientedEdge::new(e_oz, false),  // Z → O (reversed)
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let qd0_id = topo.add_wire(qd0_wire);
    let qd0_face = topo.add_face(Face::new(
        qd0_id,
        Vec::new(),
        FaceSurface::Plane {
            normal: n0,
            d: cut_planes[0].1,
        },
    ));

    let qd1_wire = Wire::new(
        vec![
            OrientedEdge::new(e_oz, true),   // O → Z
            OrientedEdge::new(arc_zx, true), // Z → X (arc)
            OrientedEdge::new(e_ox, false),  // X → O (reversed)
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let qd1_id = topo.add_wire(qd1_wire);
    let qd1_face = topo.add_face(Face::new(
        qd1_id,
        Vec::new(),
        FaceSurface::Plane {
            normal: n1,
            d: cut_planes[1].1,
        },
    ));

    let qd2_wire = Wire::new(
        vec![
            OrientedEdge::new(e_ox, true),   // O → X
            OrientedEdge::new(arc_xy, true), // X → Y (arc)
            OrientedEdge::new(e_oy, false),  // Y → O (reversed)
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let qd2_id = topo.add_wire(qd2_wire);
    let qd2_face = topo.add_face(Face::new(
        qd2_id,
        Vec::new(),
        FaceSurface::Plane {
            normal: n2,
            d: cut_planes[2].1,
        },
    ));

    // Spherical patch: bounded by the 3 arcs.
    // Wind so the sphere's outward normal matches the resulting volume
    // (outside the octant). With arcs going X→Y→Z→X around the patch,
    // the right-hand rule gives an outward normal pointing AWAY from O.
    let sph_wire = Wire::new(
        vec![
            OrientedEdge::new(arc_xy, true), // X → Y
            OrientedEdge::new(arc_yz, true), // Y → Z
            OrientedEdge::new(arc_zx, true), // Z → X
        ],
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let sph_wire_id = topo.add_wire(sph_wire);
    let sphere_surface = SphericalSurface::new(sphere_center, r).map_err(|e| {
        crate::OperationsError::InvalidInput {
            reason: format!("box-sphere octant: sphere surface construction failed: {e}"),
        }
    })?;
    let sphere_face = topo.add_face(Face::new(
        sph_wire_id,
        Vec::new(),
        FaceSurface::Sphere(sphere_surface),
    ));

    let shell = Shell::new(vec![qd0_face, qd1_face, qd2_face, sphere_face])
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = topo.add_solid(Solid::new(shell_id, Vec::new()));
    Ok(Some(solid))
}

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

/// Compute the coaxial-torus boolean for two tori sharing center, axis,
/// and major radius. Returns `Ok(None)` when the shortcut doesn't apply
/// (Cut, or degenerate radii / overlap).
///
/// Like the concentric-sphere shortcut, the result tessellation density
/// is inherited from the higher-quality input so a 64-segment input
/// torus never silently downgrades.
#[allow(clippy::too_many_arguments)]
fn coaxial_torus_shortcut(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    center: Point3,
    axis: Vec3,
    major_radius: f64,
    minor_a: f64,
    minor_b: f64,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<SolidId>, crate::OperationsError> {
    if minor_a <= tol.linear || minor_b <= tol.linear || major_radius <= tol.linear {
        return Ok(None);
    }
    let minor_result = match op {
        BooleanOp::Fuse => minor_a.max(minor_b),
        BooleanOp::Intersect => {
            // Both minors are guaranteed > tol by the guard above.
            minor_a.min(minor_b)
        }
        // Cut on coaxial tori with shared major produces a hollow torus
        // (outer + inner small-circle shells) when minor_a > minor_b.
        // `make_torus` doesn't build that topology — defer to GFA.
        BooleanOp::Cut => return Ok(None),
    };
    if minor_result >= major_radius {
        // make_torus rejects self-intersecting tori (minor >= major).
        return Ok(None);
    }

    // Inherit segment count from the higher-quality input. `make_torus`
    // accepts a `segments` param controlling u-direction discretization.
    // We'd ideally extract this from each input solid's vertex count, but
    // unlike make_sphere torus topology has internal seam vertices that
    // make the relationship less clean. Approximate by the larger vertex
    // count.
    let segments_a = brepkit_topology::explorer::solid_vertices(topo, a)
        .map(|v| v.len())
        .unwrap_or(0);
    let segments_b = brepkit_topology::explorer::solid_vertices(topo, b)
        .map(|v| v.len())
        .unwrap_or(0);
    let segments = segments_a.max(segments_b).max(8);

    // Build a fresh torus at the origin then transform to the shared
    // center / axis. `make_torus` builds with axis = +z by default.
    let torus = crate::primitives::make_torus(topo, major_radius, minor_result, segments)?;
    let xform = xform_from_canonical_z(center, axis, tol);
    crate::transform::transform_solid(topo, torus, &xform)?;
    Ok(Some(torus))
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

/// Returns `true` when two axis-aligned boxes are separated on at least
/// one axis by more than `margin` — i.e. their (margin-expanded) extents
/// do not overlap and the solids they bound provably do not intersect.
///
/// The `margin` shrinks the overlap test so boxes that only touch (or
/// nearly touch) within `margin` are treated as separated: a shared
/// face/edge/corner has zero overlap volume.
fn aabbs_separated(
    a: &brepkit_math::aabb::Aabb3,
    b: &brepkit_math::aabb::Aabb3,
    margin: f64,
) -> bool {
    a.max.x() < b.min.x() + margin
        || b.max.x() < a.min.x() + margin
        || a.max.y() < b.min.y() + margin
        || b.max.y() < a.min.y() + margin
        || a.max.z() < b.min.z() + margin
        || b.max.z() < a.min.z() + margin
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
    // Mesh density here is a boolean-robustness concern, independent of the
    // rendering angular tolerance; use the linear-only criterion so the
    // resulting face count is unaffected by the display deflection cap.
    let mesh_a = crate::tessellate::tessellate_solid_with_tolerance(topo, a, deflection, 0.0)?;
    let mesh_b = crate::tessellate::tessellate_solid_with_tolerance(topo, b, deflection, 0.0)?;

    // Compute per-triangle "is on a planar face" flags by matching each
    // triangle's centroid + normal against the input solid's planar
    // face equations. Used by mesh_boolean to drop tessellation-diagonal
    // artifacts on planar faces while keeping load-bearing intermediates
    // on curved surfaces (issue #696).
    let planar_a = infer_planar_triangle_flags(topo, a, &mesh_a, tol);
    let planar_b = infer_planar_triangle_flags(topo, b, &mesh_b, tol);

    let mb_result = crate::mesh_boolean::mesh_boolean_with_metadata(
        &mesh_a,
        Some(&planar_a),
        &mesh_b,
        Some(&planar_b),
        op,
        tol.linear,
    )?;
    let face_specs = mesh_result_to_face_specs(&mb_result);
    if face_specs.is_empty() {
        return Err(crate::OperationsError::EmptyResult {
            reason: "mesh boolean produced no output faces".into(),
        });
    }
    let result = assemble_solid_mixed(topo, &face_specs, tol)?;
    let _ = crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    if opts.unify_faces {
        let _ = crate::heal::unify_faces(topo, result)?;
    }
    // Cross-face symmetrization: tessellation diagonals that one face
    // dropped while its neighbour kept (#696) leave structurally
    // orphan collinear interior wire vertices. Collapse those so both
    // sides reference the same EdgeId for the shared 3D segment,
    // eliminating the residual non-manifold edges that `unify_faces`
    // can't symmetrize from per-face surface matching alone.
    let collapsed =
        brepkit_heal::upgrade::collapse_collinear_vertices::collapse_collinear_wire_vertices(
            topo, result, tol,
        )
        .unwrap_or_else(|e| {
            log::warn!("boolean {op:?}: collapse_collinear_wire_vertices failed: {e}");
            0
        });
    if collapsed > 0 {
        log::info!(
            "boolean {op:?}: collapsed {collapsed} collinear interior wire vertex/vertices post-mesh-assembly",
        );
    }
    // Mesh-fallback can glue two physically-separate holes into a
    // single figure-8 inner wire via diagonal "bridge" edges across
    // gap material (#696 cumulative pattern: a slab top with multiple
    // pocket cuts ends up with one self-intersecting inner wire that
    // visits each pocket region). Split such wires at every pinch
    // vertex so each physical hole is its own simple inner wire —
    // the resulting topology is well-formed for downstream
    // tessellation, validation, and STEP export, even when the
    // bridge edges themselves remain as boundary edges (those are a
    // separate cleanup).
    let wires_split =
        brepkit_heal::upgrade::split_self_intersecting_wires::split_self_intersecting_inner_wires(
            topo, result,
        )
        .unwrap_or_else(|e| {
            log::warn!("boolean {op:?}: split_self_intersecting_inner_wires failed: {e}");
            0
        });
    if wires_split > 0 {
        log::info!(
            "boolean {op:?}: split {wires_split} self-intersecting inner wire(s) post-mesh-assembly",
        );
    }
    if opts.heal_after_boolean {
        let _ = crate::heal::heal_solid(topo, result, tol.linear)?;
    }
    assembly::validate_boolean_result_lenient(topo, result)?;
    log::info!(
        "boolean {op:?}: mesh boolean path → solid {} ({} faces, surface types lost)",
        result.index(),
        face_specs.len()
    );
    Ok(result)
}

/// For each triangle in `mesh`, return `true` iff the triangle is coplanar
/// with one of `solid`'s planar topology faces. Used by mesh_boolean to
/// gate the collinear-midpoint drop (issue #696): the drop is safe only
/// for triangles on planar input faces, where the dropped intermediate is
/// a tessellation diagonal artifact rather than a load-bearing tessellation
/// vertex.
///
/// Matching criterion: triangle's face normal aligns with the topology
/// plane's normal AND the triangle centroid lies on the plane within
/// linear tolerance. Triangles whose source is curved (cylinder, sphere,
/// NURBS, etc.) won't match any planar face and get `false`.
fn infer_planar_triangle_flags(
    topo: &Topology,
    solid: SolidId,
    mesh: &crate::tessellate::TriangleMesh,
    tol: brepkit_math::tolerance::Tolerance,
) -> Vec<bool> {
    let tri_count = mesh.indices.len() / 3;
    let mut flags = vec![false; tri_count];

    // Collect planar topology faces with their plane equations, normalized
    // to unit normals so the comparisons below can use exact cos thresholds
    // and point-on-plane distances. `validate_solid_relaxed_with_options`
    // allows non-unit normals in the topology, so we can't assume the
    // stored `normal` has magnitude 1 — divide through here. Empty
    // collection ⇒ all flags stay false (boolean falls back to baseline).
    let mut planes: Vec<(brepkit_math::vec::Vec3, f64)> = Vec::new();
    if let Ok(face_ids) = brepkit_topology::explorer::solid_faces(topo, solid) {
        for fid in face_ids {
            if let Ok(face) = topo.face(fid) {
                if let brepkit_topology::face::FaceSurface::Plane { normal, d } = face.surface() {
                    let len = normal.dot(*normal).sqrt();
                    if len > tol.linear {
                        planes.push((*normal * (1.0 / len), *d / len));
                    }
                }
            }
        }
    }
    if planes.is_empty() {
        return flags;
    }

    let lin_tol = tol.linear;
    let ang_tol = tol.angular.max(1e-9);
    // Degenerate-area threshold: the cross-product magnitude is parallelogram
    // area (length²), so compare against length² to match dimensions. A
    // triangle with edge length below `lin_tol` has area below `lin_tol²`.
    let degen_area_sq = lin_tol * lin_tol;

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3] as usize;
        let i1 = mesh.indices[t * 3 + 1] as usize;
        let i2 = mesh.indices[t * 3 + 2] as usize;
        let v0 = mesh.positions[i0];
        let v1 = mesh.positions[i1];
        let v2 = mesh.positions[i2];
        let face_normal = (v1 - v0).cross(v2 - v0);
        let area_sq = face_normal.dot(face_normal);
        if area_sq < degen_area_sq {
            continue; // degenerate triangle: no reliable normal direction.
        }
        let fn_len = area_sq.sqrt();
        let unit = face_normal * (1.0 / fn_len);
        let centroid_x = (v0.x() + v1.x() + v2.x()) / 3.0;
        let centroid_y = (v0.y() + v1.y() + v2.y()) / 3.0;
        let centroid_z = (v0.z() + v1.z() + v2.z()) / 3.0;
        for &(plane_normal, d) in &planes {
            // Normals parallel (within angular tolerance, either direction).
            // Plane normal is unit-normalized above.
            let cos = plane_normal.dot(unit);
            if cos.abs() < 1.0 - ang_tol {
                continue;
            }
            // Centroid on plane: |n·c - d| ≤ lin_tol (both n and d are
            // unit-normalized, so this is the true point-to-plane distance).
            let dist = plane_normal.x() * centroid_x
                + plane_normal.y() * centroid_y
                + plane_normal.z() * centroid_z
                - d;
            if dist.abs() <= lin_tol {
                flags[t] = true;
                break;
            }
        }
    }
    flags
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

/// True when the outer-shell face components represent disjoint solid
/// pieces (e.g., a previous cut split one solid into N parts), false
/// when one component is concentric inside another (a hollow solid:
/// outer surface + cavity surface both live in the outer shell).
///
/// The check is AABB-based: if any component's bounding box is
/// strictly contained in another's, treat the whole solid as hollow
/// and skip the multi-region split path.
/// Check that every component's AABB centre classifies as outside the
/// supplied classifier. Used to reject multi-region GFA Cut results that
/// erroneously include the tool's interior as one of the pieces.
fn all_component_centers_outside(
    topo: &Topology,
    components: &[Vec<FaceId>],
    classifier: &brepkit_algo::classifier::AnalyticClassifier,
    tol: brepkit_math::tolerance::Tolerance,
) -> bool {
    use brepkit_algo::FaceClass;
    for comp in components {
        let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        for &fid in comp {
            let Ok(face) = topo.face(fid) else { continue };
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                let Ok(wire) = topo.wire(wid) else { continue };
                for oe in wire.edges() {
                    let Ok(edge) = topo.edge(oe.edge()) else {
                        continue;
                    };
                    for vid in [edge.start(), edge.end()] {
                        if let Ok(v) = topo.vertex(vid) {
                            let p = v.point();
                            min = Point3::new(
                                min.x().min(p.x()),
                                min.y().min(p.y()),
                                min.z().min(p.z()),
                            );
                            max = Point3::new(
                                max.x().max(p.x()),
                                max.y().max(p.y()),
                                max.z().max(p.z()),
                            );
                        }
                    }
                }
            }
        }
        let centre = Point3::new(
            (min.x() + max.x()) * 0.5,
            (min.y() + max.y()) * 0.5,
            (min.z() + max.z()) * 0.5,
        );
        if matches!(classifier.classify(centre, tol), Some(FaceClass::Inside)) {
            return false;
        }
    }
    true
}

fn components_are_disjoint_pieces(topo: &Topology, components: &[Vec<FaceId>]) -> bool {
    let aabbs: Vec<(Point3, Point3)> = components
        .iter()
        .map(|comp| {
            let mut min = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
            let mut max = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
            for &fid in comp {
                let Ok(face) = topo.face(fid) else {
                    continue;
                };
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    let Ok(wire) = topo.wire(wid) else {
                        continue;
                    };
                    for oe in wire.edges() {
                        let Ok(edge) = topo.edge(oe.edge()) else {
                            continue;
                        };
                        for vid in [edge.start(), edge.end()] {
                            if let Ok(v) = topo.vertex(vid) {
                                let p = v.point();
                                min = Point3::new(
                                    min.x().min(p.x()),
                                    min.y().min(p.y()),
                                    min.z().min(p.z()),
                                );
                                max = Point3::new(
                                    max.x().max(p.x()),
                                    max.y().max(p.y()),
                                    max.z().max(p.z()),
                                );
                            }
                        }
                    }
                }
            }
            (min, max)
        })
        .collect();

    let eps = 1e-7;
    for i in 0..aabbs.len() {
        for j in (i + 1)..aabbs.len() {
            let (i_min, i_max) = aabbs[i];
            let (j_min, j_max) = aabbs[j];
            let x_overlap = i_min.x().max(j_min.x()) + eps < i_max.x().min(j_max.x());
            let y_overlap = i_min.y().max(j_min.y()) + eps < i_max.y().min(j_max.y());
            let z_overlap = i_min.z().max(j_min.z()) + eps < i_max.z().min(j_max.z());
            if x_overlap && y_overlap && z_overlap {
                return false;
            }
        }
    }
    true
}

/// Merge duplicate vertices in a solid's shell by position.
///
/// Cut a multi-region input solid: split the components, cut each
/// against `b` independently, then combine the per-component results
/// back into a single multi-region solid.
///
/// This works around the GFA pavefiller's assumption of a single
/// connected input — feeding a 2-piece "solid" into GFA loses one piece
/// at a time as the cut proceeds (Category B `multiple cuts creating
/// three pieces` and gear bore are both downstream of this).
fn cut_multi_region_input(
    topo: &mut Topology,
    a: SolidId,
    b: SolidId,
    comp_count: usize,
) -> Result<SolidId, crate::OperationsError> {
    let components = crate::boolean::assembly::face_components(topo, a);
    debug_assert_eq!(components.len(), comp_count);

    let mut per_component_results: Vec<SolidId> = Vec::with_capacity(components.len());
    for comp_faces in components {
        // Copy the component's faces into a fresh single-component solid
        // so the boolean engine sees a connected manifold.
        let comp_solid_raw = make_solid_from_face_subset(topo, &comp_faces)?;
        // Deep-copy the component into a fresh solid so its faces/edges/
        // vertices have fresh IDs disjoint from the original multi-region
        // input — GFA's pavefiller can stumble on shared vertex IDs across
        // what it considers a single "solid A".
        let comp_solid = crate::copy::copy_solid(topo, comp_solid_raw)?;
        match boolean(topo, BooleanOp::Cut, comp_solid, b) {
            Ok(r) => per_component_results.push(r),
            Err(
                crate::OperationsError::EmptyResult { .. }
                | crate::OperationsError::InvalidInput { .. },
            ) => {
                per_component_results.push(comp_solid);
            }
            Err(e) => return Err(e),
        }
    }

    // Combine all per-component results into a single multi-region solid.
    // Collect every face from every result into one outer shell. The
    // results are pairwise disjoint by construction (each came from a
    // disjoint input component cut by the same tool), so a single shell
    // containing all their faces is a valid manifold representation.
    let mut all_faces: Vec<brepkit_topology::face::FaceId> = Vec::new();
    for &r in &per_component_results {
        let r_data = topo.solid(r)?;
        for &fid in topo.shell(r_data.outer_shell())?.faces() {
            all_faces.push(fid);
        }
    }
    make_solid_from_face_subset(topo, &all_faces)
}

/// Build a new solid whose outer shell consists exactly of the given
/// faces. Faces are referenced as-is (no copying) — the caller is
/// expected to pass faces that already form a closed manifold.
///
/// `reversed=true` faces are NORMALIZED on the way in: a fresh face is
/// created with the surface normal negated, the wires reversed, and
/// `reversed=false`. Boolean operations downstream are sensitive to the
/// `reversed` flag (cut1's output carries reversed faces that GFA can't
/// re-process cleanly even via deep-copy), so handing GFA an
/// orientation-normalized solid recovers the fresh-primitive code path.
fn make_solid_from_face_subset(
    topo: &mut Topology,
    faces: &[brepkit_topology::face::FaceId],
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut normalized: Vec<brepkit_topology::face::FaceId> = Vec::with_capacity(faces.len());
    for &fid in faces {
        let face = topo.face(fid)?;
        if !face.is_reversed() {
            normalized.push(fid);
            continue;
        }
        // Only Plane has a trivial negate-the-normal flip. Non-planar
        // reversed faces (cylinder/cone/sphere/torus/nurbs) cannot have
        // their surface negated cheaply — they hit surface-specific GFA
        // paths that don't suffer from the same reversed-flag sensitivity.
        // Exhaustive match so a new FaceSurface variant fails to compile
        // rather than silently passing through un-normalized.
        let flipped_surface = match face.surface() {
            FaceSurface::Plane { normal, d } => FaceSurface::Plane {
                normal: -*normal,
                d: -*d,
            },
            FaceSurface::Nurbs(_)
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => {
                normalized.push(fid);
                continue;
            }
        };
        let outer_wid = face.outer_wire();
        let inner_wids: Vec<_> = face.inner_wires().to_vec();
        let outer_wire = topo.wire(outer_wid)?;
        let outer_reversed: Vec<OrientedEdge> = outer_wire
            .edges()
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        let new_outer_wire =
            Wire::new(outer_reversed, true).map_err(crate::OperationsError::Topology)?;
        let new_outer_wid = topo.add_wire(new_outer_wire);
        let mut new_inner_wids = Vec::with_capacity(inner_wids.len());
        for iw in &inner_wids {
            let w = topo.wire(*iw)?;
            let rev: Vec<OrientedEdge> = w
                .edges()
                .iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
            let new_w = Wire::new(rev, true).map_err(crate::OperationsError::Topology)?;
            new_inner_wids.push(topo.add_wire(new_w));
        }
        let new_face = Face::new(new_outer_wid, new_inner_wids, flipped_surface);
        normalized.push(topo.add_face(new_face));
    }

    let shell = brepkit_topology::shell::Shell::new(normalized)
        .map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = brepkit_topology::solid::Solid::new(shell_id, Vec::new());
    Ok(topo.add_solid(solid))
}

/// Count inner wire loops across all faces of a solid (outer + inner shells).
fn solid_inner_wire_count(topo: &Topology, solid: SolidId) -> Result<i64, crate::OperationsError> {
    let mut count: i64 = 0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(fid)?;
        #[allow(clippy::cast_possible_wrap)]
        {
            count += face.inner_wires().len() as i64;
        }
    }
    Ok(count)
}

/// Check whether a solid's outer shell is a closed manifold: every edge
/// is shared by exactly 2 faces. Returns `false` for open shells
/// (boundary edges with count == 1) and non-manifold shells (count > 2).
///
/// Stricter than [`brepkit_topology::validation::validate_shell_manifold`],
/// which only rejects edges shared by *more* than two faces.
fn is_closed_manifold(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
    use std::collections::HashMap;

    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                *counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    if counts.is_empty() {
        return Ok(false);
    }
    Ok(counts.values().all(|&c| c == 2))
}

/// Check whether a solid's outer shell has free edges: edges used by only
/// one wire occurrence. A free edge means the shell is open (e.g. a phantom
/// membrane face left a circle edge unmatched), which is never a valid
/// boolean result even when Euler accidentally balances.
fn has_free_edges(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
    use std::collections::HashMap;

    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let mut counts: HashMap<usize, usize> = HashMap::new();
    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                *counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    Ok(counts.values().any(|&c| c == 1))
}

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

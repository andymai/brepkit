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

    // Detect A⊂B or B⊂A (including A=B) and handle directly.
    // Only applies when BOTH solids have simple analytic classifiers.
    {
        use brepkit_algo::classifier::try_build_analytic_classifier;
        let ca = try_build_analytic_classifier(topo, a);
        let cb = try_build_analytic_classifier(topo, b);
        let TrivialRelation {
            identical,
            a_in_b,
            b_in_a,
        } = detect_trivial_relation(topo, a, b, ca.as_ref(), cb.as_ref(), tol);

        // Identical-solid shortcut: matching AABBs AND every boundary
        // vertex of each solid classifies as inside-or-on the other's
        // analytic classifier. Stronger than a center test (a cube
        // inscribed in a sphere has matching AABBs but cube corners fall
        // outside the sphere) and works for non-convex solids like tori.
        if identical {
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
        // Cut with the tool strictly inside the blank: build the hollow result
        // (blank + a reversed copy of the tool as a cavity shell) directly.
        // GFA's no-intersection assembly drops fully-contained cone/torus
        // tools; the cavity is exactly the tool's reversed shell, so construct
        // it here for any simple tool whose vertices are all strictly inside.
        if op == BooleanOp::Cut
            && b_in_a
            && !a_in_b
            && let Some(classifier) = ca.as_ref()
        {
            let tool_simple = topo
                .solid(b)
                .map(|s| s.inner_shells().is_empty())
                .unwrap_or(false);
            if tool_simple
                && solid_strictly_inside(topo, b, classifier, tol)
                && let Ok(result) = build_contained_cut_hollow(topo, a, b)
                && validate_boolean_result(topo, result).is_ok()
            {
                return Ok(result);
            }
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
            if let (true, Some(slope)) = (same_axis_dir && same_apex && same_half_angle, slope_a)
                && let Some(result) = coaxial_cone_shortcut(
                    topo,
                    op,
                    *oa,
                    *aa,
                    slope,
                    (*za_min, *za_max),
                    (*zb_min, *zb_max),
                    tol,
                )?
            {
                return Ok(result);
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
            && let Some(result) = box_pair_shortcut(topo, op, *a_min, *a_max, *b_min, *b_max, tol)?
        {
            return Ok(result);
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
            if coincident
                && let Some(result) =
                    concentric_sphere_shortcut(topo, op, a, b, *ca_center, *ra, *rb, tol)?
            {
                return Ok(result);
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
            if coincident
                && coaxial
                && same_major
                && let Some(result) = coaxial_torus_shortcut(
                    topo, op, a, b, *ca_center, *aa, *maj_a, *min_a, *min_b, tol,
                )?
            {
                return Ok(result);
            }
        }
    }

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
        if let Some((a_box, b_box)) = bb_a.zip(bb_b)
            && aabbs_separated(&a_box, &b_box, tol.linear)
        {
            return Ok(topo.add_empty_solid());
        }
    }

    // Disjoint-fuse fast path: when A and B are provably spatially disjoint,
    // their union is a multi-region solid — the same result GFA produces for
    // disjoint inputs, but built by a cheap shell merge instead of the full
    // pavefiller/assembly pipeline. This is what makes a pairwise-accumulate
    // loop over many disjoint pieces (e.g. one tapered foot per gridfinity
    // cell) scale linearly: each fuse onto the growing accumulator short-
    // circuits here.
    //
    // Disjointness is decided per connected component (not per whole-solid
    // bbox): the accumulator spans many pieces, so its overall box overlaps
    // the next piece's box even when no piece actually touches. Component
    // boxes are conservative outer bounds, and the gap test uses a positive
    // tolerance margin, so the path only fires on a clear gap — touching or
    // overlapping operands fall through to GFA, which welds the shared
    // geometry. The result is independent of the inputs (each operand is
    // deep-copied before merging), preserving the boolean contract.
    if op == BooleanOp::Fuse && solids_provably_disjoint(topo, a, b, tol.linear) {
        let copy_a = crate::copy::copy_solid(topo, a)?;
        let copy_b = crate::copy::copy_solid(topo, b)?;
        let merged = crate::compound_ops::merge_disjoint_solids(topo, &[copy_a, copy_b])?;
        log::debug!("Fuse short-circuited via disjoint shell merge");
        return Ok(merged);
    }

    let algo_op = match op {
        BooleanOp::Fuse => brepkit_algo::bop::BooleanOp::Fuse,
        BooleanOp::Cut => brepkit_algo::bop::BooleanOp::Cut,
        BooleanOp::Intersect => brepkit_algo::bop::BooleanOp::Intersect,
    };
    // Recognise flat NURBS walls/edges as analytic planes/lines so the engine's
    // face-face intersections take the exact plane×plane path (the tool's
    // rounded-rect extrude emits straight cavity walls as planar B-splines).
    // Only an operand that actually carries flattenable NURBS is deep-copied
    // and rewritten; operands without any (the common case — primitives and
    // already-analytic solids) are passed through unchanged. This matters for
    // correctness, not just speed: the engine's downstream ordering is keyed on
    // entity ids, so needlessly deep-copying an operand (which renumbers its
    // ids) can perturb volume-sensitive cut/fuse results.
    let gfa_a = if solid_has_flattenable_nurbs(topo, a, tol.linear)? {
        let copy_a = crate::copy::copy_solid(topo, a)?;
        let _ = flatten_planar_nurbs_faces(topo, copy_a, tol.linear)?;
        copy_a
    } else {
        a
    };
    let gfa_b = if solid_has_flattenable_nurbs(topo, b, tol.linear)? {
        let copy_b = crate::copy::copy_solid(topo, b)?;
        let _ = flatten_planar_nurbs_faces(topo, copy_b, tol.linear)?;
        copy_b
    } else {
        b
    };
    let gfa_start = timer_now();
    match brepkit_algo::gfa::boolean(topo, algo_op, gfa_a, gfa_b) {
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
                // Strip out-and-back wire spurs left by the GFA wire builder on
                // U-shaped (single-opening-notch) faces — they over-connect the
                // opening edge and inflate volume (issue #801 slot fuse).
                let _ = crate::heal::remove_wire_spurs(topo, result)?;
                // A coincident-junction fuse can leave duplicate junction-wire
                // edges (one per argument) that differ by sub-micron loft noise
                // → free edges. Merge those coincident duplicates. Gated on the
                // shell actually being open so clean results keep exact topology.
                if has_free_edges(topo, result).unwrap_or(false) {
                    // Best-effort: an error here shouldn't abort the boolean,
                    // but it's useful signal on an already-broken shell.
                    if let Err(e) =
                        unify_coincident_boundary_edges(topo, result, (tol.linear * 10.0).max(1e-6))
                    {
                        log::debug!("unify_coincident_boundary_edges failed: {e}");
                    }
                }
                // Check Euler before unify_faces — if already valid, skip
                // unify to avoid its face-merging bugs (non-manifold edges).
                let (f_pre, e_pre, v_pre) =
                    brepkit_topology::explorer::solid_entity_counts(topo, result)?;
                #[allow(clippy::cast_possible_wrap)]
                let euler_pre = (v_pre as i64) - (e_pre as i64) + (f_pre as i64);

                // If Euler>2, try merging duplicate vertices before unify.
                // This fixes the flush-face case where duplicate vertices at
                // cross-rank positions inflate V.
                let merged_vertices = euler_pre > 2;
                if merged_vertices {
                    // Best-effort: don't abort on merge failure
                    let _ = merge_result_vertices(topo, result, tol);
                }

                // Re-count only when the merge above ran; otherwise the counts
                // are unchanged from the pre-merge measurement (the merge is the
                // only mutation in between).
                let (f2, e2, v2) = if merged_vertices {
                    brepkit_topology::explorer::solid_entity_counts(topo, result)?
                } else {
                    (f_pre, e_pre, v_pre)
                };
                #[allow(clippy::cast_possible_wrap)]
                let euler_pre2 = (v2 as i64) - (e2 as i64) + (f2 as i64);

                // Hollow results (a Cut whose tool sits strictly inside the
                // target) arrive from GFA with the cavity assembled as inner
                // shells. Each closed genus-0 cavity shell adds 2 to V-E+F,
                // so the Euler acceptance below must compare against
                // 2 + 2*K instead of 2. Entity counts above already include
                // inner-shell entities via `solid_entity_counts`.
                #[allow(clippy::cast_possible_wrap)]
                let inner_shell_surplus = 2 * (topo.solid(result)?.inner_shells().len() as i64);

                // Hole-aware Euler: a face with L inner wire loops raises V-E+F
                // by L (Euler-Poincare: V-E+F-L = 2(1-g)), so a valid genus-0
                // result with holed faces (e.g. a fuse leaving circular holes in
                // box faces) has euler = 2 + L. Compute the inner-wire surplus
                // once here so both the unify decision and the acceptance gate
                // use the same hole-aware balance — otherwise a result that
                // deviates from euler==2 solely because of inner wires would
                // still trigger an unnecessary unify_faces pass.
                let inner_wire_count_pre = solid_inner_wire_count(topo, result)?;
                let euler_balanced_pre = euler_pre2 - inner_shell_surplus == 2
                    || euler_balanced(euler_pre2 - inner_shell_surplus, inner_wire_count_pre);

                // Run unify_faces if the (hole-aware) Euler is off OR if the
                // topology has 3+-face junctions, which can occur with a
                // balanced Euler when overlapping coplanar faces cancel in
                // V-E+F counting. The same-domain detection in the assembler
                // only pairs faces across opposing ranks with identical edge
                // sets, so within-rank or different-boundary overlaps slip
                // through; unify_faces is the safety net for those (issue #696).
                // `is_closed_manifold` is a whole-solid walk. It is needed both
                // here (to decide unify) and again after unify (the acceptance
                // gate). Compute the pre-unify value at most once, and reuse it
                // for the gate when unify changes nothing. It is only evaluated
                // when `euler_balanced_pre` holds (otherwise `||` short-circuits
                // and `needs_unify` is already true).
                let manifold_pre = if euler_balanced_pre {
                    Some(is_closed_manifold(topo, result)?)
                } else {
                    None
                };
                let needs_unify = !euler_balanced_pre || manifold_pre == Some(false);
                let mut unified = false;
                if needs_unify {
                    for _ in 0..3 {
                        if crate::heal::unify_faces(topo, result)? == 0 {
                            break;
                        }
                        unified = true;
                    }
                }
                // Re-count only when unify actually merged faces; otherwise the
                // counts are unchanged from the (post-merge) measurement above.
                let (f, e, v) = if unified {
                    brepkit_topology::explorer::solid_entity_counts(topo, result)?
                } else {
                    (f2, e2, v2)
                };
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
                // Hole-aware Euler acceptance: re-measure the inner-wire surplus
                // after unify (which can merge faces and change wire counts) and
                // accept euler - L == 2 - 2g for genus g >= 0. The holed/genus
                // acceptance additionally requires a closed manifold so that
                // accidental cancellations (open shells whose missing faces
                // offset the inner-wire surplus) still fail safe to the mesh
                // fallback. Reuse the pre-unify count when unify made no change.
                let inner_wire_count = if unified {
                    solid_inner_wire_count(topo, result)?
                } else {
                    inner_wire_count_pre
                };
                // `is_closed_manifold` walks every face/edge of the result; the
                // hollow gate, the genus-acceptance gate, and the multi-region
                // gate below all need it on the same (post-unify) topology, so
                // compute it once. Reuse the pre-unify value when it was already
                // computed AND unify changed nothing — the only intervening
                // mutation. Propagating a topology-query error with `?` here is
                // equivalent to the old multi-region `unwrap_or(false)`: that
                // call ran on this same solid, so an error would have surfaced
                // at the hollow gate (reached first) regardless.
                let closed_manifold = match manifold_pre {
                    Some(m) if !unified => m,
                    _ => is_closed_manifold(topo, result)?,
                };
                // A hollow result must additionally have every shell closed:
                // a missing cavity face could otherwise cancel against the
                // inner-shell surplus and balance Euler by accident.
                let hollow_ok = inner_shell_surplus == 0 || closed_manifold;
                let euler_eff = euler - inner_shell_surplus;
                let euler_ok = hollow_ok
                    && (euler_eff == 2
                        || (euler_balanced(euler_eff, inner_wire_count) && closed_manifold));
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
                    // Reuse the `closed_manifold` computed above: nothing between
                    // it and here mutates the result (only read-only component
                    // and classifier queries run in between).
                    && closed_manifold
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

    // When the input solid carries multiple disjoint pieces (a previous
    // cut split a solid into N parts), GFA's pavefiller can't process
    // them together — feeding the whole thing in loses regions. Splitting
    // into per-component cuts and recombining preserves the missing
    // pieces. Cut distributes over disjoint union; Fuse/Intersect have
    // more complex interaction semantics so we leave those to mesh.
    if op == BooleanOp::Cut {
        let components = crate::boolean::assembly::face_components(topo, a);
        if components.len() >= 2
            && components_are_disjoint_pieces(topo, &components)
            && let Ok(result) = cut_multi_region_input(topo, a, b, components.len())
        {
            return Ok(result);
        }
    }

    // Mesh boolean fallback (no recursion).
    log::debug!(
        target: "brepkit_approx",
        "boolean {op:?}: GFA unusable — using mesh (co-refinement) fallback; analytic surface types will be lost"
    );
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

/// Perform a boolean operation and return an [`crate::evolution::EvolutionMap`]
/// tracking face provenance.
///
/// Prefers **faithful** provenance from the GFA builder — each result face
/// records the input face it was split/derived from
/// (`brepkit_algo::gfa::boolean_with_face_origins`). Because that path runs the
/// GFA directly, it can take a different route than [`boolean`] (which
/// short-circuits some cases via AABB/containment fast paths), so its result is
/// validated; on a GFA error or an invalid result — and for identical operands
/// — it falls back to [`boolean`] with the geometry heuristic (normal +
/// centroid). Either way, unmatched input faces are classified as "deleted";
/// synthesised result faces with no input origin are left unattributed.
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
    use brepkit_topology::explorer::solid_faces;

    // Faithful path: the GFA reports each result face's true input source.
    // Identical/contained operand pairs must NOT take it: those are the
    // fully-coincident-boundary configurations `boolean` short-circuits
    // precisely because the raw GFA mis-splits them — coincident walls drop
    // into an open shell whose position-duplicate free edges pass the
    // by-edge-id validation gate (every edge id used ≤ 2×), so the broken
    // result would be returned as "valid". Route them through `boolean`'s
    // shortcuts below; the geometry heuristic attributes a copied result's
    // faces exactly (normal + centroid match 1:1).
    let trivial = {
        use brepkit_algo::classifier::try_build_analytic_classifier;
        let tol = brepkit_math::tolerance::Tolerance::new();
        let ca = try_build_analytic_classifier(topo, a);
        let cb = try_build_analytic_classifier(topo, b);
        let rel = detect_trivial_relation(topo, a, b, ca.as_ref(), cb.as_ref(), tol);
        rel.identical || rel.a_in_b || rel.b_in_a
    };
    if a != b && !trivial {
        let input_indices: Vec<usize> = solid_faces(topo, a)?
            .into_iter()
            .chain(solid_faces(topo, b)?)
            .map(brepkit_topology::arena::Id::index)
            .collect();
        let algo_op = match op {
            BooleanOp::Fuse => brepkit_algo::bop::BooleanOp::Fuse,
            BooleanOp::Cut => brepkit_algo::bop::BooleanOp::Cut,
            BooleanOp::Intersect => brepkit_algo::bop::BooleanOp::Intersect,
        };
        if let Ok((result, origins)) =
            brepkit_algo::gfa::boolean_with_face_origins(topo, algo_op, a, b)
        {
            // Apply the face-id-preserving result heals so the evolution result
            // is as correct as the standard boolean (manifold, no #801 wire
            // spurs). These rewrite wires in place, so the provenance — keyed by
            // face ID — survives. `unify_faces` is intentionally NOT run here:
            // it merges coplanar faces into new entities, discarding the
            // per-face provenance this path exists to track.
            // A heal failure here is not fatal: fall through to boolean()'s
            // full pipeline rather than propagating (the result solid stays as
            // orphaned topology, which is harmless in the arena).
            let tol = brepkit_math::tolerance::Tolerance::default();
            let healed_ok = crate::heal::remove_degenerate_edges(topo, result, tol.linear).is_ok()
                && crate::heal::remove_wire_spurs(topo, result).is_ok();

            // Trust the faithful path only if its result is valid; otherwise
            // fall through to boolean()'s full pipeline (fast paths + mesh
            // fallback + validation), matching boolean()'s contract.
            if healed_ok && validate_boolean_result(topo, result).is_ok() {
                let mut evo = crate::evolution::EvolutionMap::new();
                let mut sourced: std::collections::HashSet<usize> =
                    std::collections::HashSet::new();
                for (out_idx, src) in origins {
                    if let Some(in_idx) = src {
                        evo.add_modified(in_idx, out_idx);
                        sourced.insert(in_idx);
                    }
                }
                for in_idx in input_indices {
                    if !sourced.contains(&in_idx) {
                        evo.add_deleted(in_idx);
                    }
                }
                return Ok((result, evo));
            }
        }
    }

    // Fallback: geometry heuristic over the standard boolean result. Reached
    // for identical operands, a GFA error, or a GFA result that failed
    // validation — the EvolutionMap is then approximate, not faithful.
    log::debug!("boolean_with_evolution: faithful GFA provenance unavailable, using heuristic");
    let input_faces_a = collect_face_signatures(topo, a)?;
    let input_faces_b = collect_face_signatures(topo, b)?;

    let mut input_faces: Vec<(usize, Vec3, Point3)> =
        Vec::with_capacity(input_faces_a.len() + input_faces_b.len());
    input_faces.extend(input_faces_a);
    input_faces.extend(input_faces_b);

    let result = boolean(topo, op, a, b)?;

    let output_faces = collect_face_signatures(topo, result)?;

    let evo = crate::evolution::build_evolution_by_geometry(&input_faces, &output_faces);

    Ok((result, evo))
}

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

/// Returns `true` when two axis-aligned boxes have a *clear gap* exceeding
/// `margin` on at least one axis — i.e. they are separated by a real positive
/// distance, not merely touching.
///
/// This is intentionally stricter than [`aabbs_separated`]: a shared
/// face/edge/corner (zero gap) returns `false` here. Touching solids must NOT
/// be treated as disjoint by the fuse fast path — their shared geometry has to
/// be welded by GFA.
fn aabbs_clear_gap(
    a: &brepkit_math::aabb::Aabb3,
    b: &brepkit_math::aabb::Aabb3,
    margin: f64,
) -> bool {
    b.min.x() - a.max.x() > margin
        || a.min.x() - b.max.x() > margin
        || b.min.y() - a.max.y() > margin
        || a.min.y() - b.max.y() > margin
        || b.min.z() - a.max.z() > margin
        || a.min.z() - b.max.z() > margin
}

/// Returns `true` when solids `a` and `b` are provably spatially disjoint with
/// a clear gap: every connected face component of `a` is separated from every
/// connected face component of `b` by more than `margin` on some axis.
///
/// Soundness: component AABBs come from [`crate::measure::face_set_bounding_box`],
/// which is a conservative *outer* bound (vertices plus surface-curvature
/// expansion). If two components' true geometry overlapped or touched, their
/// boxes would touch or overlap and [`aabbs_clear_gap`] would (correctly)
/// return `false`. So a `true` result guarantees a real positive gap between
/// the two solids — never a false "disjoint" for touching/coincident inputs,
/// which must still go through GFA to weld shared geometry.
///
/// Component-level (rather than whole-solid) granularity is essential: a
/// multi-region solid (e.g. an accumulator of several already-merged disjoint
/// pieces) has a single outer shell whose overall box overlaps a nearby piece,
/// yet none of its pieces actually touch that piece. [`assembly::face_components`]
/// recovers the individual pieces from the merged shell.
///
/// Returns `false` on any topology error or empty operand (fall through to the
/// general path) rather than risking an unsound merge.
fn solids_provably_disjoint(topo: &Topology, a: SolidId, b: SolidId, margin: f64) -> bool {
    let comps_a = assembly::face_components(topo, a);
    let comps_b = assembly::face_components(topo, b);
    if comps_a.is_empty() || comps_b.is_empty() {
        return false;
    }
    let boxes = |comps: &[Vec<FaceId>]| -> Option<Vec<brepkit_math::aabb::Aabb3>> {
        comps
            .iter()
            .map(|faces| crate::measure::face_set_bounding_box(topo, faces).ok())
            .collect()
    };
    let (Some(boxes_a), Some(boxes_b)) = (boxes(&comps_a), boxes(&comps_b)) else {
        return false;
    };
    boxes_a
        .iter()
        .all(|ba| boxes_b.iter().all(|bb| aabbs_clear_gap(ba, bb, margin)))
}

/// The trivial operand relationships that let [`boolean`] short-circuit
/// without running the GFA: identical solids and full containment.
struct TrivialRelation {
    /// Matching AABBs AND every boundary vertex of each solid classifies
    /// as inside-or-on the other's analytic classifier.
    identical: bool,
    /// A is fully contained in B.
    a_in_b: bool,
    /// B is fully contained in A.
    b_in_a: bool,
}

/// Detect the trivial operand relationships (identical / contained).
///
/// [`boolean`] uses this to take copy/empty shortcuts. [`boolean_with_evolution`]
/// consults the same detection BEFORE its faithful raw-GFA provenance path:
/// these are exactly the fully-coincident-boundary configurations the raw GFA
/// mis-splits (coincident walls dropped into an open shell whose
/// position-duplicate free edges slip past the by-edge-id validation gate), so
/// the evolution path must route them through [`boolean`]'s shortcuts instead.
fn detect_trivial_relation(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    ca: Option<&brepkit_algo::classifier::AnalyticClassifier>,
    cb: Option<&brepkit_algo::classifier::AnalyticClassifier>,
    tol: brepkit_math::tolerance::Tolerance,
) -> TrivialRelation {
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

    // AABB enclosure is necessary but NOT sufficient for solid
    // containment: a non-convex container (notched or hollow) can
    // AABB-enclose a solid that actually lies in its empty region.
    // Issue #801: `(a − b) ∪ (a ∩ b)` dropped the `a ∩ b` operand
    // because the unit cube's bbox fits inside the notched `a − b`'s
    // bbox, yet the cube lives in the carved-out notch. Confirm the
    // AABB-only fallback with a real point-in-solid test: reject when
    // the inner solid's center is provably inside `inner` yet outside
    // `outer`. By the containment lemma (inner ⊆ outer ⇒ every point
    // of inner is in outer), that witness can only occur for genuine
    // non-containment, so it never rejects a true containment.
    let center_outside =
        |topo: &Topology, inner: SolidId, outer: SolidId, bb: &Option<(Point3, Point3)>| -> bool {
            let Some((lo, hi)) = *bb else { return false };
            let c = Point3::new(
                0.5 * (lo.x() + hi.x()),
                0.5 * (lo.y() + hi.y()),
                0.5 * (lo.z() + hi.z()),
            );
            let (dx, dy, dz) = (hi.x() - lo.x(), hi.y() - lo.y(), hi.z() - lo.z());
            let defl = (dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt() * 0.01).max(1e-6);
            // Conservative by design: when the AABB center falls in `inner`'s own
            // concavity (a C/U-shaped solid), `inside_inner` is false and the
            // witness is disabled, so a false-positive containment could still
            // slip through. That only ever fails to *reject* — it never rejects a
            // true containment — so the shortcut stays sound, just not complete.
            let inside_inner = matches!(
                crate::classify::classify_point(topo, inner, c, defl, tol.linear),
                Ok(crate::classify::PointClassification::Inside)
            );
            let outside_outer = matches!(
                crate::classify::classify_point(topo, outer, c, defl, tol.linear),
                Ok(crate::classify::PointClassification::Outside)
            );
            inside_inner && outside_outer
        };

    // Bidirectional vertex check via the analytic classifier — the
    // primary signal for identical/containment classification. A vertex
    // classifying as inside-or-on (None within tolerance band counts
    // as on) means it sits within the solid's region.
    let all_b_verts_in_a = ca.is_some_and(|c| all_vertices_inside_or_on(topo, b, c, tol));
    let all_a_verts_in_b = cb.is_some_and(|c| all_vertices_inside_or_on(topo, a, c, tol));
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

    // Containment: A contains B when all B vertices are inside-or-on A AND
    // A's AABB encloses B's. Falls back to a strict AABB-only check when the
    // containing solid has no classifier — the strict check requires ≥10%
    // larger in ALL three dims so that sparse multi-shell solids (e.g., a
    // fuse of two disjoint boxes) don't false-positive as "contains another
    // solid".
    // Both the analytic-classifier term and the AABB-only fallback can
    // false-positive when the container is non-convex: the analytic
    // classifier may mis-report notch points as inside-or-on, and an
    // AABB encloses a notch's empty volume. Guard the whole determination
    // with the `center_outside` witness — sound for every path because it
    // only fires on proven non-containment (see the lemma above).
    let b_in_a = ((all_b_verts_in_a && aabb_encloses(&aabb_b, &aabb_a))
        || (ca.is_none() && aabb_strictly_contains(&aabb_b, &aabb_a)))
        && !center_outside(topo, b, a, &aabb_b);
    let a_in_b = ((all_a_verts_in_b && aabb_encloses(&aabb_a, &aabb_b))
        || (cb.is_none() && aabb_strictly_contains(&aabb_a, &aabb_b)))
        && !center_outside(topo, a, b, &aabb_a);

    TrivialRelation {
        identical: aabbs_match && all_b_verts_in_a && all_a_verts_in_b,
        a_in_b,
        b_in_a,
    }
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

/// True when every outer-shell vertex of `inner` classifies as *strictly*
/// `Inside` (not on the boundary) of `classifier`. A strictly-contained tool
/// has no surface contact with the blank, so `Cut(blank, tool)` is a clean
/// internal cavity rather than a notch through the boundary.
fn solid_strictly_inside(
    topo: &Topology,
    inner: SolidId,
    classifier: &brepkit_algo::classifier::AnalyticClassifier,
    tol: brepkit_math::tolerance::Tolerance,
) -> bool {
    let Ok(s) = topo.solid(inner) else {
        return false;
    };
    let Ok(sh) = topo.shell(s.outer_shell()) else {
        return false;
    };
    let mut saw_vertex = false;
    for &fid in sh.faces() {
        let Ok(f) = topo.face(fid) else { return false };
        // Check the outer wire and any inner (hole) wires — a hole boundary on
        // a simple solid's face can also reach the blank's surface.
        let mut wires = vec![f.outer_wire()];
        wires.extend_from_slice(f.inner_wires());
        for wid in wires {
            let Ok(w) = topo.wire(wid) else {
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
                    if classifier.classify(v.point(), tol) != Some(brepkit_algo::FaceClass::Inside)
                    {
                        return false;
                    }
                    saw_vertex = true;
                }
            }
        }
    }
    saw_vertex
}

/// Build `Cut(blank, tool)` for a tool strictly contained in the blank: the
/// result is the blank with a tool-shaped internal cavity. Deep-copies the
/// blank and the tool, reverses every copied tool face in place so the cavity
/// boundary faces into the void, and attaches the reversed tool shell to the
/// copied blank as an inner shell. Bypasses GFA, whose no-intersection assembly
/// drops fully-contained cone/torus tools.
fn build_contained_cut_hollow(
    topo: &mut Topology,
    blank: SolidId,
    tool: SolidId,
) -> Result<SolidId, crate::OperationsError> {
    let result = crate::copy::copy_solid(topo, blank)?;

    // Deep-copy the tool as a whole solid so the cavity shell shares edges and
    // vertices between adjacent faces (a per-face copy would duplicate shared
    // boundary edges and leave the cavity non-manifold — wrong Euler, though
    // per-face volume is unaffected). Reverse each copied face in place and
    // reuse the copied outer shell directly as the cavity inner shell, so no
    // duplicate faces or extra result solid are created.
    let tool_copy = crate::copy::copy_solid(topo, tool)?;
    let cavity_shell = topo.solid(tool_copy)?.outer_shell();
    let cavity_faces = topo.shell(cavity_shell)?.faces().to_vec();
    for fid in cavity_faces {
        let face = topo.face_mut(fid)?;
        let flipped = !face.is_reversed();
        face.set_reversed(flipped);
    }
    topo.solid_mut(result)?.add_inner_shell(cavity_shell);
    Ok(result)
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
    // rendering tolerance: use the linear-only criterion (angular_tol 0.0) so
    // the face count is unaffected by the display deflection cap, AND keep the
    // circle curvature floor so co-refinement gets the denser circular sampling
    // it needs (display tessellation drops that floor for triangle count).
    let mesh_a = crate::tessellate::tessellate_solid_for_boolean(topo, a, deflection, 0.0)?;
    let mesh_b = crate::tessellate::tessellate_solid_for_boolean(topo, b, deflection, 0.0)?;

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
            if let Ok(face) = topo.face(fid)
                && let brepkit_topology::face::FaceSurface::Plane { normal, d } = face.surface()
            {
                let len = normal.dot(*normal).sqrt();
                if len > tol.linear {
                    planes.push((*normal * (1.0 / len), *d / len));
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

/// Genus-aware Euler balance for a closed orientable surface with holed faces.
///
/// Euler-Poincare for a closed surface of genus `g`: `V - E + F - L = 2(1 - g)`,
/// so the inner-wire surplus `euler - L` equals `2 - 2g` and is valid when it
/// is an even number no greater than 2: `2` for genus 0, `0` for genus 1, and
/// negative even values for genus >= 2 (e.g. a thin wall pierced by N
/// through-holes has genus N). Odd or > 2 surpluses indicate a miscounted
/// shell. Callers must pair this with a closed-manifold check — the relation
/// only holds for closed surfaces.
const fn euler_balanced(euler: i64, inner_wires: i64) -> bool {
    let surplus = euler - inner_wires;
    surplus <= 2 && surplus % 2 == 0
}

/// Count edge uses across ALL shells of a solid (outer + inner cavity
/// shells). Hollow solids keep cavity faces in inner shells — an
/// outer-shell-only walk silently misses their edges, letting open or
/// non-manifold cavity shells pass the acceptance gates.
fn solid_edge_use_counts(
    topo: &Topology,
    solid: SolidId,
) -> Result<std::collections::HashMap<usize, usize>, crate::OperationsError> {
    let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                *counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    Ok(counts)
}

/// Check whether every shell of a solid is a closed manifold: every edge
/// is shared by exactly 2 faces within its shell. Returns `false` for open
/// shells (boundary edges with count == 1) and non-manifold shells
/// (count > 2). Walks inner (cavity) shells as well as the outer shell —
/// each shell is an independent closed surface, so a single pooled count
/// per shell is correct.
///
/// Stricter than [`brepkit_topology::validation::validate_shell_manifold`],
/// which only rejects edges shared by *more* than two faces.
fn is_closed_manifold(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
    let s = topo.solid(solid)?;
    let shell_ids: Vec<_> = std::iter::once(s.outer_shell())
        .chain(s.inner_shells().iter().copied())
        .collect();
    for shell_id in shell_ids {
        let shell = topo.shell(shell_id)?;
        if !shell_is_closed_manifold(topo, shell)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn shell_is_closed_manifold(
    topo: &Topology,
    shell: &brepkit_topology::shell::Shell,
) -> Result<bool, crate::OperationsError> {
    use std::collections::HashMap;

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

/// Check whether a solid's boundary has free edges: edges used by only
/// one wire occurrence. A free edge means the shell is open (e.g. a phantom
/// membrane face left a circle edge unmatched), which is never a valid
/// boolean result even when Euler accidentally balances.
fn has_free_edges(topo: &Topology, solid: SolidId) -> Result<bool, crate::OperationsError> {
    let counts = solid_edge_use_counts(topo, solid)?;
    Ok(counts.values().any(|&c| c == 1))
}

/// Cheap read-only test for whether [`flatten_planar_nurbs_faces`] would change
/// anything: does `solid` carry a planar NURBS face or a straight NURBS edge?
/// Used to gate the deep-copy-and-flatten pre-pass so analytic operands are
/// passed to the engine unchanged (a needless deep copy renumbers entity ids
/// and can perturb the engine's id-keyed ordering on volume-sensitive cuts).
///
/// `tol` must match the linear tolerance passed to [`flatten_planar_nurbs_faces`]
/// so the gate and the pass agree: a looser default here could report "nothing
/// to flatten" while the pass (run at the operation tolerance) would in fact
/// rewrite geometry, reintroducing the NURBS-vs-plane fragmentation.
fn solid_has_flattenable_nurbs(
    topo: &Topology,
    solid: SolidId,
    tol: f64,
) -> Result<bool, crate::OperationsError> {
    use brepkit_geometry::convert::{
        RecognizedCurve, RecognizedSurface, recognize_curve, recognize_surface,
    };
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::solid_faces;

    let mut seen = std::collections::HashSet::new();
    for fid in solid_faces(topo, solid)? {
        let face = topo.face(fid)?;
        if let FaceSurface::Nurbs(nurbs) = face.surface()
            && matches!(
                recognize_surface(nurbs, tol),
                RecognizedSurface::Plane { .. }
            )
        {
            return Ok(true);
        }
        for &wid in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if !seen.insert(eid.index()) {
                    continue;
                }
                if let EdgeCurve::NurbsCurve(nurbs) = topo.edge(eid)?.curve()
                    && matches!(recognize_curve(nurbs, tol), RecognizedCurve::Line { .. })
                {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Replace planar NURBS faces of `solid` with analytic `Plane` surfaces, and
/// straight NURBS boundary edges with the `Line` variant.
///
/// A NURBS surface whose every control point lies within `tol` of a single
/// plane is geometrically a plane; the tool's rounded-rect extrude emits the
/// straight cavity walls as planar B-splines, and the boolean engine's
/// face-face intersections only take the exact (same-domain) plane×plane path
/// when both operands are `FaceSurface::Plane`. Recognising the flat walls as
/// planes before the boolean lets coincident/abutting wall regions merge
/// analytically instead of fragmenting through the NURBS surface-intersection
/// path.
///
/// The same extrude also leaves the straight cavity-floor/wall boundary edges
/// as NURBS curves. A planar-arrangement splitter treats every non-`Line` edge
/// as an arc and bails when one is split mid-edge by a coplanar section, so a
/// straight NURBS floor edge crossed by the scoop footprint forces the floor
/// face to a self-crossing trace. Recognising those straight NURBS edges as
/// `Line` lets the arrangement split them exactly.
///
/// Genuinely curved NURBS surfaces/edges (and all other analytic geometry) are
/// left untouched. Returns the number of faces flattened.
fn flatten_planar_nurbs_faces(
    topo: &mut Topology,
    solid: SolidId,
    tol: f64,
) -> Result<usize, crate::OperationsError> {
    use brepkit_geometry::convert::{
        RecognizedCurve, RecognizedSurface, recognize_curve, recognize_surface,
    };
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_faces;

    let face_ids = solid_faces(topo, solid)?;
    // Snapshot the surfaces first (immutable borrow), then mutate.
    let planar: Vec<(FaceId, Vec3, f64)> = face_ids
        .iter()
        .filter_map(|&fid| {
            let face = topo.face(fid).ok()?;
            let FaceSurface::Nurbs(nurbs) = face.surface() else {
                return None;
            };
            match recognize_surface(nurbs, tol) {
                RecognizedSurface::Plane { normal, d } => {
                    // `recognize_surface` derives the plane normal from a
                    // control-point cross product, whose sign can OPPOSE the
                    // NURBS surface's own du×dv normal. A `FaceSurface::Plane`
                    // is read with its normal flipped by `is_reversed`, so an
                    // opposed sign silently inverts the face's effective
                    // outward direction. Align to the surface du×dv normal at
                    // the domain midpoint.
                    let (u0, u1) = nurbs.domain_u();
                    let (v0, v1) = nurbs.domain_v();
                    let mid_n = nurbs.normal(0.5 * (u0 + u1), 0.5 * (v0 + v1)).ok();
                    let (normal, d) = match mid_n {
                        Some(n) if normal.dot(n) < 0.0 => (-normal, -d),
                        _ => (normal, d),
                    };
                    Some((fid, normal, d))
                }
                _ => None,
            }
        })
        .collect();
    let count = planar.len();
    for (fid, normal, d) in planar {
        topo.face_mut(fid)?
            .set_surface(FaceSurface::Plane { normal, d });
    }

    // Straighten NURBS edges that are geometrically lines.
    let mut straight_edges: Vec<EdgeId> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        for &wid in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if !seen.insert(eid.index()) {
                    continue;
                }
                let EdgeCurve::NurbsCurve(nurbs) = topo.edge(eid)?.curve() else {
                    continue;
                };
                if matches!(recognize_curve(nurbs, tol), RecognizedCurve::Line { .. }) {
                    straight_edges.push(eid);
                }
            }
        }
    }
    for eid in straight_edges {
        topo.edge_mut(eid)?.set_curve(EdgeCurve::Line);
    }

    Ok(count)
}

/// Test-only access to [`flatten_planar_nurbs_faces`] so integration tests can
/// reproduce the exact operand preprocessing the boolean applies before handing
/// the operands to the GFA engine.
#[doc(hidden)]
pub fn flatten_planar_nurbs_faces_for_tests(
    topo: &mut Topology,
    solid: SolidId,
    tol: f64,
) -> Result<usize, crate::OperationsError> {
    flatten_planar_nurbs_faces(topo, solid, tol)
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

/// Merge geometrically-coincident duplicate boundary edges on the outer shell.
///
/// A coincident-junction fuse (e.g. a box stacked on a tapered loft that share
/// a cap face) annihilates the shared cap but leaves each argument's faces
/// carrying their OWN copy of the junction-wire edges. Because the two copies
/// come from independently-built solids their endpoints differ by sub-micron
/// numerical noise (loft re-parameterization), so the tight-tolerance vertex
/// merge above leaves them as distinct edges — each used once → free edges that
/// open the shell.
///
/// This snaps vertices at `tol_merge` (looser than the default linear
/// tolerance, to absorb that noise), then rebuilds every wire against a global
/// canonical-edge map keyed by *unordered canonical endpoints + curve type +
/// geometric midpoint* — so a straight line and a bulged arc between the same
/// endpoints stay distinct, while true duplicates collapse to one shared edge.
/// Edges whose endpoints merge to a single vertex (degenerate) are dropped.
///
/// Returns `true` if anything changed. Run only on already-broken results
/// (free edges / non-manifold) so clean booleans keep their exact topology.
#[allow(
    clippy::too_many_lines,
    clippy::type_complexity,
    clippy::items_after_statements
)]
fn unify_coincident_boundary_edges(
    topo: &mut Topology,
    solid: SolidId,
    tol_merge: f64,
) -> Result<bool, crate::OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::vertex::VertexId;
    use brepkit_topology::wire::{OrientedEdge, Wire, WireId};
    use std::collections::HashMap;

    let shell_id = topo.solid(solid)?.outer_shell();
    let face_ids: Vec<_> = topo.shell(shell_id)?.faces().to_vec();

    let scale = 1.0 / tol_merge;
    let q = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };

    // 1. Canonical vertex per quantized position (first VertexId seen wins).
    let mut vcanon: HashMap<(i64, i64, i64), VertexId> = HashMap::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    let key = q(topo.vertex(vid)?.point());
                    vcanon.entry(key).or_insert(vid);
                }
            }
        }
    }

    // 2. Snapshot each face's wires (edge id, fwd, curve, endpoints, tol).
    type OeSnap = (EdgeId, bool, EdgeCurve, VertexId, VertexId, Option<f64>);
    struct FaceSnap {
        surface: FaceSurface,
        reversed: bool,
        outer: Vec<OeSnap>,
        outer_closed: bool,
        inners: Vec<(Vec<OeSnap>, bool)>,
    }
    let snap_wire =
        |topo: &Topology, wid: WireId| -> Result<(Vec<OeSnap>, bool), crate::OperationsError> {
            let w = topo.wire(wid)?;
            let closed = w.is_closed();
            let oes = w
                .edges()
                .iter()
                .map(|oe| -> Result<OeSnap, crate::OperationsError> {
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
            Ok((oes, closed))
        };
    let mut snaps = Vec::with_capacity(face_ids.len());
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let (outer, outer_closed) = snap_wire(topo, face.outer_wire())?;
        let mut inners = Vec::new();
        for iw in face.inner_wires() {
            inners.push(snap_wire(topo, *iw)?);
        }
        snaps.push(FaceSnap {
            surface,
            reversed,
            outer,
            outer_closed,
            inners,
        });
    }

    // 3. Rebuild wires against a global canonical-edge map.
    //    Key: (lo endpoint q, hi endpoint q, midpoint q, curve type tag).
    type EdgeKey = (
        (i64, i64, i64),
        (i64, i64, i64),
        (i64, i64, i64),
        &'static str,
    );
    let mut ecanon: HashMap<EdgeKey, (EdgeId, VertexId, VertexId)> = HashMap::new();
    let mut changed = false;

    let canon_vid = |topo: &Topology, vid: VertexId| -> Result<VertexId, crate::OperationsError> {
        Ok(*vcanon.get(&q(topo.vertex(vid)?.point())).unwrap_or(&vid))
    };

    let rebuild = |topo: &mut Topology,
                   oes: &[OeSnap],
                   ecanon: &mut HashMap<EdgeKey, (EdgeId, VertexId, VertexId)>,
                   changed: &mut bool|
     -> Result<Vec<OrientedEdge>, crate::OperationsError> {
        let mut out = Vec::with_capacity(oes.len());
        for (eid, fwd, curve, start, end, etol) in oes {
            let cs = canon_vid(topo, *start)?;
            let ce = canon_vid(topo, *end)?;
            if cs == ce {
                // Endpoints collapsed to a single vertex → degenerate, drop it.
                *changed = true;
                continue;
            }
            let sp = topo.vertex(*start)?.point();
            let ep = topo.vertex(*end)?.point();
            let (t0, t1) = curve.domain_with_endpoints(sp, ep);
            let mid = curve.evaluate_with_endpoints((t0 + t1) * 0.5, sp, ep);
            let (cs_q, ce_q) = (q(topo.vertex(cs)?.point()), q(topo.vertex(ce)?.point()));
            let (lo, hi) = if cs_q <= ce_q {
                (cs_q, ce_q)
            } else {
                (ce_q, cs_q)
            };
            let key = (lo, hi, q(mid), curve.type_tag());

            // Physical traversal start vertex (after canonicalization).
            let trav_start = if *fwd { cs } else { ce };
            if let Some(&(c_eid, c_start, _c_end)) = ecanon.get(&key) {
                // A duplicate of an already-seen edge → merge onto the keeper.
                *changed = true;
                out.push(OrientedEdge::new(c_eid, c_start == trav_start));
            } else {
                // First edge with this key. Reuse the original edge when its
                // endpoints didn't move; only allocate (and flag a change) when
                // a vertex was snapped — so an already-clean shell is a no-op.
                let (eid_use, e_start) = if cs == *start && ce == *end {
                    (*eid, *start)
                } else {
                    *changed = true;
                    (
                        topo.add_edge(Edge::with_tolerance(cs, ce, curve.clone(), *etol)),
                        cs,
                    )
                };
                ecanon.insert(key, (eid_use, e_start, ce));
                out.push(OrientedEdge::new(eid_use, e_start == trav_start));
            }
        }
        Ok(out)
    };

    let mut new_face_ids = Vec::with_capacity(snaps.len());
    for snap in &snaps {
        let outer_oes = rebuild(topo, &snap.outer, &mut ecanon, &mut changed)?;
        let Ok(outer_wire) = Wire::new(outer_oes, snap.outer_closed) else {
            // Keep original face if the rebuilt wire is invalid.
            return Ok(false);
        };
        let outer_id = topo.add_wire(outer_wire);
        let mut inner_ids = Vec::new();
        for (inner_oes, inner_closed) in &snap.inners {
            let oes = rebuild(topo, inner_oes, &mut ecanon, &mut changed)?;
            let Ok(w) = Wire::new(oes, *inner_closed) else {
                // A dropped hole silently changes topology (and removes free
                // edges, so the downstream gate can't catch it). Bail like the
                // outer-wire case, leaving the original solid untouched.
                return Ok(false);
            };
            inner_ids.push(topo.add_wire(w));
        }
        let mut new_face =
            brepkit_topology::face::Face::new(outer_id, inner_ids, snap.surface.clone());
        if snap.reversed {
            new_face.set_reversed(true);
        }
        new_face_ids.push(topo.add_face(new_face));
    }

    if !changed {
        return Ok(false);
    }

    let new_shell = brepkit_topology::shell::Shell::new(new_face_ids)?;
    let new_shell_id = topo.add_shell(new_shell);
    topo.solid_mut(solid)?.set_outer_shell(new_shell_id);
    Ok(true)
}

/// Post-process a solid to enforce manifold topology via greedy flood-fill.
///
/// Detects non-manifold edges (shared by 3+ faces) and uses greedy
/// shell building to split the non-manifold shell into manifold
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
        if i != best_idx
            && !faces.is_empty()
            && let Ok(inner) = brepkit_topology::shell::Shell::new(faces.clone())
        {
            inner_ids.push(topo.add_shell(inner));
        }
    }

    Ok(topo.add_solid(brepkit_topology::solid::Solid::new(outer_id, inner_ids)))
}

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
/// Snapshot each outer-shell face as `(index, face normal, centroid)` — the
/// signature [`crate::evolution::build_evolution_by_geometry`] matches on. The
/// normal is the stored plane normal (or a polygon-derived normal for
/// non-planar faces), not re-oriented by the face's `reversed` flag; matching
/// stays consistent because input and output faces use the same convention.
pub fn collect_face_signatures(
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

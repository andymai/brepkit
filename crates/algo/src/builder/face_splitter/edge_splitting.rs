//! Curve-specific edge splitting at 3D intersection points.

use brepkit_math::madd::Madd;
use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

use super::super::pcurve_compute::{
    compute_boundary_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface,
    shorter_arc_delta,
};
use super::super::plane_frame::PlaneFrame;
use super::super::split_types::OrientedPCurveEdge;
use super::sampling::normalize_angle_in_span;

/// Split boundary edges at 3D points where section edges start/end.
///
/// Handles Line, Circle, and Ellipse edges. For curved edges, projects
/// split points onto the curve via `Circle3D::project` / `Ellipse3D::project`
/// and checks distance from the curve. Creates sub-arc edges with pcurves
/// computed via sampling.
pub(super) fn split_boundary_edges_at_3d_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    frame: Option<&PlaneFrame>,
    surface: &FaceSurface,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    split_edges_at_3d_points(edges, split_pts_3d, frame, surface, tol, false)
}

/// [`split_boundary_edges_at_3d_points`] for a hole woven into a curved
/// face's arrangement, at the ends of the sections that reach it. A spline
/// edge splits where projection onto the part of the spline the edge runs on
/// finds the point, as the weave's contact test finds it.
pub(super) fn split_hole_edges_at_3d_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    frame: Option<&PlaneFrame>,
    surface: &FaceSurface,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    split_edges_at_3d_points(edges, split_pts_3d, frame, surface, tol, true)
}

#[allow(clippy::too_many_lines)]
fn split_edges_at_3d_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    frame: Option<&PlaneFrame>,
    surface: &FaceSurface,
    tol: f64,
    nurbs_by_projection: bool,
) -> Vec<OrientedPCurveEdge> {
    let mut result = Vec::new();
    for edge in edges {
        let mut splits = match &edge.curve_3d {
            EdgeCurve::Circle(circle) => {
                find_splits_on_circle(circle, &edge, split_pts_3d, surface, tol)
            }
            EdgeCurve::Ellipse(ellipse) => {
                find_splits_on_ellipse(ellipse, &edge, split_pts_3d, tol)
            }
            // A marched-NURBS boundary edge (a plane×cone conic minted by an
            // earlier boolean) must split where a neighbouring face's
            // partition anchors on it — the chord-based line fallback rejects
            // on-curve points by the sagitta, leaving the shared edge whole on
            // one face and split on the other (cross-face desync).
            EdgeCurve::NurbsCurve(_) if nurbs_by_projection => {
                find_splits_on_nurbs_by_projection(&edge, split_pts_3d, tol).unwrap_or_else(|| {
                    find_splits_on_nurbs_section(&edge, split_pts_3d, tol, false)
                })
            }
            EdgeCurve::NurbsCurve(_) => {
                find_splits_on_nurbs_section(&edge, split_pts_3d, tol, false)
            }
            EdgeCurve::Line => find_splits_on_line(&edge, split_pts_3d, tol),
        };

        if splits.is_empty() {
            result.push(edge);
            continue;
        }
        // A closed plane rim split once would leave two complementary arcs
        // sharing both endpoints (with each other, and with a chord section
        // across the rim); pave both at their middles exactly as
        // `make_blocks` does for the edge images, so this face's pieces
        // match its neighbours' and the assembler's endpoint-keyed merge
        // never conflates the two arcs. Only a holed planar face re-splits a
        // raw closed rim here (its outer wire is not image-expanded); every
        // other face, the wall sharing the rim included, takes the images
        // and so already carries the `make_blocks` midpoint.
        if frame.is_some() && splits.len() == 1 && (edge.start_3d - edge.end_3d).length() < tol {
            use std::f64::consts::TAU;
            match &edge.curve_3d {
                EdgeCurve::Circle(c) => {
                    let a0 = c.project(edge.start_3d);
                    let a1 = c.project(splits[0].1);
                    let first = (a1 - a0).rem_euclid(TAU);
                    let second = TAU - first;
                    for m in [a0 + 0.5 * first, a1 + 0.5 * second] {
                        let delta = if edge.forward { m - a0 } else { a0 - m };
                        let t_mid = delta.rem_euclid(TAU) / TAU;
                        if t_mid > tol && t_mid < 1.0 - tol {
                            splits.push((t_mid, c.evaluate(m)));
                        }
                    }
                }
                // An ellipse split is placed along its own parameter span
                // (`find_splits_on_ellipse`), so its midpoints are too.
                EdgeCurve::Ellipse(e) => {
                    let (t0, t1) = edge
                        .curve_3d
                        .domain_with_endpoints(edge.start_3d, edge.end_3d);
                    let a0 = e.project(edge.start_3d);
                    let a1 = e.project(splits[0].1);
                    let first = (a1 - a0).rem_euclid(TAU);
                    let second = TAU - first;
                    for m in [a0 + 0.5 * first, a1 + 0.5 * second] {
                        let t_mid = normalize_angle_in_span(m, t0, t1 - t0);
                        if t_mid > tol && t_mid < 1.0 - tol {
                            splits.push((t_mid, e.evaluate(m)));
                        }
                    }
                }
                EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => {}
            }
            splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        }
        // Split two or more times, a closed plane rim keeps a piece between
        // every two consecutive splits, and the chord section between those
        // two points shares both its endpoints with that piece: the
        // endpoint-keyed edge merge then collapses the two (a keep box's
        // diagonal face cutting a pin's rim kept the removed half's rim in
        // place of the chord). A chord's endpoints bound two arcs, one
        // through the seam (already a vertex) and one through the seam's
        // antipode, which the periodic path always puts on the wall sharing
        // this rim (`split_face_2d_impl`); put the cap's split there too.
        if frame.is_some()
            && splits.len() >= 2
            && let EdgeCurve::Circle(c) = &edge.curve_3d
            && (edge.start_3d - edge.end_3d).length() < tol
        {
            let anti_t = 0.5;
            let anti = c.evaluate(c.project(edge.start_3d) + std::f64::consts::PI);
            let inside_piece = splits
                .windows(2)
                .any(|w| w[0].0 < anti_t && anti_t < w[1].0);
            if inside_piece && splits.iter().all(|(_, p)| (*p - anti).length() > tol) {
                splits.push((anti_t, anti));
                splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        let circle_iso_v_rim = matches!(edge.curve_3d, EdgeCurve::Circle(_))
            && circle_edge_is_iso_v_rim(&edge, surface, tol);
        // A CLOSED rim on a cylinder/cone spans a whole period in the edge's own
        // unwrapped u (`start_u` -> `start_u ± 2π`, signed by the traversal). A
        // raw principal-value projection of each split point would drop the
        // pieces back into `[0, 2π)`, so the interior joints lose phase
        // coherence with the neighbouring boundary edges and the ring stops
        // reading as a monotone walk across the cut-open strip. Interpolate in
        // the edge's own span instead — the closed-edge counterpart of the
        // `circle_iso_v_rim` branch, which covers only OPEN rims.
        // The ring's own samples run the curve's sense in u (`end_uv`
        // follows the curve, not the traversal), which is opposite the
        // surface's u when the circle's normal opposes the axis (an
        // apex-up cone's rims, a mirrored cylinder's).
        let closed_ring_dir = (matches!(edge.curve_3d, EdgeCurve::Circle(_))
            && (matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
                || matches!(surface, FaceSurface::Torus(_)) && is_parallel_ring(&edge))
            && (edge.start_3d - edge.end_3d).length() < tol)
            .then(|| {
                let native = if edge.end_uv.x() < edge.start_uv.x() {
                    -1.0_f64
                } else {
                    1.0
                };
                if edge.forward { native } else { -native }
            });
        let mut prev_uv = edge.start_uv;
        let mut prev_3d = edge.start_3d;
        // A woven hole keeps its new vertices and its pieces' pcurves on the
        // copy of the period its edge is laid out on: the weave then moves
        // the whole hole to the face's window by one shift.
        let periods = crate::builder::pcurve_compute::surface_periods(surface);
        let on_copy_of = |pcurve: brepkit_math::curves2d::Curve2D,
                          uv: brepkit_math::vec::Point2,
                          periods: (Option<f64>, Option<f64>)| {
            let (du, dv) = pcurve_shift_to(&pcurve, uv, periods);
            if du == 0.0 && dv == 0.0 {
                pcurve
            } else {
                shifted_pcurve(&pcurve, du, dv).unwrap_or(pcurve)
            }
        };
        for &(t, pt) in &splits {
            // Circle splits carry the exact on-curve foot; re-evaluating via
            // `evaluate_edge_at_t` would re-apply the CCW-span convention
            // that iso-v rim splits deliberately bypass. NURBS splits carry
            // the CANDIDATE point itself (validated within `tol` of the
            // curve): it is the neighbouring face's anchor, and adopting it
            // verbatim gives both faces the identical split vertex.
            let split_3d = if matches!(
                edge.curve_3d,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            ) {
                pt
            } else {
                evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t)
            };
            let split_uv = if let Some(f) = frame {
                f.project(split_3d)
            } else if let Some(dir) = closed_ring_dir {
                brepkit_math::vec::Point2::new(
                    std::f64::consts::TAU.madd(dir * t, edge.start_uv.x()),
                    edge.start_uv.y(),
                )
            } else if circle_iso_v_rim {
                // Interpolate within the edge's own UV span: an iso-v rim's
                // pcurve is u-linear, and a raw principal-value projection
                // would break phase coherence with the neighbouring
                // boundary edges' unwrapped u.
                brepkit_math::vec::Point2::new(
                    (edge.end_uv.x() - edge.start_uv.x()).madd(t, edge.start_uv.x()),
                    edge.start_uv.y(),
                )
            } else if let (EdgeCurve::Line, FaceSurface::Cylinder(_)) = (&edge.curve_3d, surface) {
                // A line on a cylinder is a ruling at one u: a split on a
                // seam copy keeps that copy's u, and only v runs along it.
                brepkit_math::vec::Point2::new(
                    edge.start_uv.x(),
                    (edge.end_uv.y() - edge.start_uv.y()).madd(t, edge.start_uv.y()),
                )
            } else if let (EdgeCurve::Line, FaceSurface::Cone(cone)) = (&edge.curve_3d, surface) {
                // A line on a cone is a ruling at one u: a split on a seam
                // copy keeps that copy's u (the apex end's own u is its
                // ruling's, or arbitrary), and only v runs along it.
                let u = if (edge.start_3d - cone.apex()).length() < tol {
                    edge.end_uv.x()
                } else {
                    edge.start_uv.x()
                };
                brepkit_math::vec::Point2::new(
                    u,
                    (edge.end_uv.y() - edge.start_uv.y()).madd(t, edge.start_uv.y()),
                )
            } else {
                project_point_on_surface(split_3d, surface, &[], None)
            };
            // On the copy the edge runs on at the split. A spline's parameter
            // need not run evenly in u (its knots), so walk along it from the
            // edge's start; an arc's, a ring's and a ruling's parameter runs
            // evenly in the surface's, so the split's place in the edge's own
            // unwrapped span reads the copy, also on an arc over half a turn.
            // A walk that cannot finish falls back to that span.
            let split_uv = if nurbs_by_projection {
                let span_end = match closed_ring_dir {
                    Some(dir) => brepkit_math::vec::Point2::new(
                        std::f64::consts::TAU.madd(dir, edge.start_uv.x()),
                        edge.start_uv.y(),
                    ),
                    None => edge.end_uv,
                };
                let along = brepkit_math::vec::Point2::new(
                    (span_end.x() - edge.start_uv.x()).madd(t, edge.start_uv.x()),
                    (span_end.y() - edge.start_uv.y()).madd(t, edge.start_uv.y()),
                );
                let near = match edge.curve_3d {
                    EdgeCurve::NurbsCurve(_) => {
                        walked_uv(&edge, t, surface, periods).unwrap_or(along)
                    }
                    EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => along,
                };
                on_copy_near(split_uv, near, periods)
            } else {
                split_uv
            };
            let mut pcurve = boundary_piece_pcurve(&edge, prev_3d, split_3d, surface, frame);
            if nurbs_by_projection {
                pcurve = on_copy_of(pcurve, prev_uv, periods);
            }
            result.push(OrientedPCurveEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve,
                start_uv: prev_uv,
                end_uv: split_uv,
                start_3d: prev_3d,
                end_3d: split_3d,
                forward: edge.forward,
                source_edge_idx: None,
                pave_block_id: None,
            });
            prev_uv = split_uv;
            prev_3d = split_3d;
        }
        let mut pcurve = boundary_piece_pcurve(&edge, prev_3d, edge.end_3d, surface, frame);
        if nurbs_by_projection {
            pcurve = on_copy_of(pcurve, prev_uv, periods);
        }
        // The stored `end_uv` of a closed rim follows the CURVE's direction
        // (`sample_edge_to_uv` ignores orientation), so a reverse-traversed ring
        // would close a period on the wrong side of its own start.
        let tail_end_uv = match closed_ring_dir {
            Some(dir) => brepkit_math::vec::Point2::new(
                std::f64::consts::TAU.madd(dir, edge.start_uv.x()),
                edge.start_uv.y(),
            ),
            None => edge.end_uv,
        };
        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve_3d.clone(),
            pcurve,
            start_uv: prev_uv,
            end_uv: tail_end_uv,
            start_3d: prev_3d,
            end_3d: edge.end_3d,
            forward: edge.forward,
            source_edge_idx: None,
            pave_block_id: None,
        });
    }
    result
}

/// The pcurve of the piece of boundary edge `edge` from `from` to `to`
/// (in traversal order): along the edge's own sense, so a piece of half a
/// turn or more traces itself rather than its complement.
pub(super) fn boundary_piece_pcurve(
    edge: &OrientedPCurveEdge,
    from: Point3,
    to: Point3,
    surface: &FaceSurface,
    frame: Option<&PlaneFrame>,
) -> brepkit_math::curves2d::Curve2D {
    let (stored_start, stored_end) = if edge.forward { (from, to) } else { (to, from) };
    compute_boundary_pcurve_on_surface(
        &edge.curve_3d,
        stored_start,
        stored_end,
        edge.forward,
        surface,
        &[],
        frame,
    )
}

/// Find split parameters on a line edge. Returns `(t, split_3d)` sorted by `t`.
pub(super) fn find_splits_on_line(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let edge_dir = edge.end_3d - edge.start_3d;
    let edge_len_sq = edge_dir.dot(edge_dir);
    if edge_len_sq < tol * tol {
        return Vec::new();
    }
    let edge_len = edge_len_sq.sqrt();
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let to_pt = sp - edge.start_3d;
        let t = to_pt.dot(edge_dir) / edge_len_sq;
        if t * edge_len <= tol || (1.0 - t) * edge_len <= tol {
            continue;
        }
        let closest = edge.start_3d + edge_dir * t;
        let dist = (sp - closest).length();
        // Weld-scale acceptance, not exact-tol: a plane-plane section
        // endpoint carries clip/trim rounding of a few tol — the kumiko
        // lattice chain ends measured 2.1e-7 and 2.3e-7 off their boundary
        // edges — and rejecting the anchor leaves the section chain a
        // pendant the arrangement rightly refuses, so the face under-splits
        // and the fuse aborts on an open growth shell. The split point uses
        // the foot on the line, so boundary pieces stay exact.
        if dist < tol * 100.0 {
            // Return the FOOT, not the raw candidate: the candidate may sit
            // anywhere in the weld band, and consumers thread the returned
            // point into wires (section T-junction splits use it verbatim).
            splits.push((t, closest));
        }
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Weld-scale 3D dedup: two candidates within the weld band are copies of
    // the SAME junction carrying different fit error (~1e-6); a parameter-only
    // exact-tol dedup keeps both and mints an untrackable micro boundary piece
    // between them.
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol || (a.1 - b.1).length() < tol * 100.0);
    splits
}

/// Find split parameters on a circle edge. Uses `Circle3D::project` for angular
/// projection, then normalizes into the edge's `[0, 1]` parameter range.
///
/// Note: `domain_with_endpoints` for full circles (start approx end) returns the
/// full `(-pi, pi]` domain. For true arcs, it uses endpoint projection -- this
/// is correct for the boundary edges produced by `make_cylinder`/`make_cone`.
pub(super) fn find_splits_on_circle(
    circle: &brepkit_math::curves::Circle3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    surface: &FaceSurface,
    tol: f64,
) -> Vec<(f64, Point3)> {
    let (t0, t1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
    let span = t1 - t0;
    if span.abs() < 1e-14 {
        return Vec::new();
    }
    // `domain_with_endpoints` always returns the CCW span between the stored
    // endpoints, but an OPEN boundary arc traversed clockwise covers the
    // COMPLEMENT arc — a rim quarter-arc walked CW reads as its 270°
    // complement, so an on-arc split point normalizes outside [0,1] and the
    // split is dropped (the section dangles, the pendant filter removes it,
    // and the rim never splits at section endpoints — the A1-corner
    // doubled-dovetail nub). Where the edge is an iso-v rim on a
    // cylinder/cone, its own UV span IS the true arc (u varies linearly at
    // constant v), so test containment and parameterize directly in the
    // edge's unwrapped u-range instead of guessing between the CCW span and
    // its complement. Everywhere else — plane-face arcs (whose UV image is
    // an arc, not a segment) and closed circles — keep the original CCW
    // convention untouched; the d-series lip fuses are calibrated to it.
    let is_iso_v_rim = circle_edge_is_iso_v_rim(edge, surface, tol);
    // A CLOSED circle has no span between its endpoints, so
    // `domain_with_endpoints` hands back the circle's intrinsic full domain,
    // anchored at the circle's own angular origin instead of at the edge's
    // start point. The consumer chains the pieces from `start_3d` in ascending
    // `t`, so whenever those two anchors differ (a cylinder seam away from the
    // origin) the start point itself reads as an interior split and the pieces
    // neither tile the ring nor stay disjoint — the wedge between the origin
    // and the seam comes out covered twice, once inside the leading piece and
    // again as a trailing forward/reverse pair. Anchor at the edge's start
    // angle, signed by the traversal direction, so `t` is monotone along the
    // walk (a periodic rim is traversed CW on the top cap, CCW on the bottom).
    //
    // Cylinder/cone rims consume these `t` values through the UV-span branch
    // in `split_boundary_edges_at_3d_points`; a closed rim on a plane (a
    // barrel's end annulus, whose seam vertex sits on the bracket plane that
    // sections it) has the same origin-vs-start mismatch and projects its
    // split UVs through the frame instead. Anchoring both at the start keeps
    // the seam from reading as an interior split (a zero-length piece the
    // arrangement then traces as its own loop). A whole circle on a torus
    // or a sphere projects its split UVs, so the parameter only orders the
    // splits, and from the origin a ring seamed elsewhere came out covered
    // twice.
    let closed_anchor = (matches!(
        surface,
        FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Plane { .. }
            | FaceSurface::Torus(_)
            | FaceSurface::Sphere(_)
    ) && (edge.start_3d - edge.end_3d).length() < tol)
        .then(|| circle.project(edge.start_3d));
    let u_span = edge.end_uv.x() - edge.start_uv.x();
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = if let Some(base) = closed_anchor {
            let delta = if edge.forward {
                angle - base
            } else {
                base - angle
            };
            delta.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU
        } else if is_iso_v_rim {
            // Unwrap the surface u of the split point into the edge's own
            // u-range (shift by whole turns toward the range midpoint), then
            // parameterize linearly along the traversal.
            let Some((u_raw, _)) = surface.project_point(closest) else {
                continue;
            };
            let mid = f64::midpoint(edge.start_uv.x(), edge.end_uv.x());
            let turns = ((mid - u_raw) / std::f64::consts::TAU).round();
            let u = std::f64::consts::TAU.madd(turns, u_raw);
            (u - edge.start_uv.x()) / u_span
        } else if !matches!(surface, FaceSurface::Nurbs(_)) && !edge.forward {
            // A reversed-traversal arc covers ccw(end→start); the
            // ccw(start→end) span below is its COMPLEMENT, so on-arc points
            // would normalize outside [0,1] and drop (a bay ring absorbed
            // into a plane's outer boundary is walked CW, so its section
            // crossings never split it), and a point on the complement
            // would split it (the half of a torus meridian walked back to
            // the seam split at the meridian's far end, a ball's rim walked
            // back to its pole split where a section crosses the rim's other
            // half, or a wall's rim piece split where a section ends on the
            // rest of the rim). Normalize in the
            // physical span and map back to traversal order. A NURBS face
            // keeps the start-to-end reading: the scooped fillet's clip to
            // its tapered envelope (`fillet_taper_clip_inmem.rs`) is exact
            // with it and falls back without it.
            let (p0, p1) = edge
                .curve_3d
                .domain_with_endpoints(edge.end_3d, edge.start_3d);
            let pspan = p1 - p0;
            if pspan.abs() < 1e-14 {
                continue;
            }
            1.0 - normalize_angle_in_span(angle, p0, pspan)
        } else {
            normalize_angle_in_span(angle, t0, span)
        };
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, closest));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Whether a closed circle on a torus is a parallel, a turn of `u` at one
/// tube angle `v`, and not a meridian, a turn of `v` at one `u`.
fn is_parallel_ring(edge: &OrientedPCurveEdge) -> bool {
    (edge.start_uv.y() - edge.end_uv.y()).abs() < 1e-9
        && ((edge.end_uv.x() - edge.start_uv.x()).abs() - std::f64::consts::TAU).abs() < 1e-6
}

/// True when a circle boundary edge is an open iso-v rim on a cylinder/cone —
/// the gate under which `find_splits_on_circle` parameterizes splits in the
/// edge's own UV span (and the split consumer must interpolate UV the same
/// way to stay phase-coherent).
pub(super) fn circle_edge_is_iso_v_rim(
    edge: &OrientedPCurveEdge,
    surface: &FaceSurface,
    tol: f64,
) -> bool {
    let u_span = edge.end_uv.x() - edge.start_uv.x();
    (edge.start_3d - edge.end_3d).length() >= tol
        && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
        && (edge.start_uv.y() - edge.end_uv.y()).abs() < 1e-9
        && u_span.abs() > 1e-9
        && u_span.abs() < std::f64::consts::TAU - 1e-9
}

/// Find split parameters on a marched-NURBS SECTION edge by sampled
/// point-to-curve projection.
///
/// The chord-based `find_splits_on_line` misses a junction point that lies on
/// the CURVE but off its chord (a plane×cone conic bulges millimetres past the
/// chord), so a section chain meeting the conic mid-span never splits it and
/// the weave breaks (the dovetail tongue-relief cone cap). Parameters use the
/// same normalized-[0,1]-over-`domain_with_endpoints` convention as
/// `evaluate_edge_at_t`'s NURBS arm, returned in order ALONG THE EDGE
/// (descending `t` when the stored curve runs `end_3d` → `start_3d`).
/// `pcurve` moved by `(du, dv)` in its surface's parameters. `None` when
/// the moved curve cannot be built.
pub(super) fn shifted_pcurve(
    pcurve: &brepkit_math::curves2d::Curve2D,
    du: f64,
    dv: f64,
) -> Option<brepkit_math::curves2d::Curve2D> {
    use brepkit_math::curves2d::{Circle2D, Curve2D, Ellipse2D, Line2D, NurbsCurve2D};
    use brepkit_math::vec::Vec2;
    use std::f64::consts::{FRAC_PI_2, PI, TAU};
    let by = Vec2::new(du, dv);
    match pcurve {
        Curve2D::Line(line) => {
            let origin = line.evaluate(0.0);
            Line2D::new(origin + by, line.evaluate(1.0) - origin)
                .ok()
                .map(Curve2D::Line)
        }
        Curve2D::Circle(circle) => {
            let centre = circle.evaluate(0.0) + (circle.evaluate(PI) - circle.evaluate(0.0)) * 0.5;
            Circle2D::new(centre + by, circle.circumference() / TAU)
                .ok()
                .map(Curve2D::Circle)
        }
        Curve2D::Ellipse(ellipse) => {
            let centre =
                ellipse.evaluate(0.0) + (ellipse.evaluate(PI) - ellipse.evaluate(0.0)) * 0.5;
            let (major, minor) = (
                ellipse.evaluate(0.0) - centre,
                ellipse.evaluate(FRAC_PI_2) - centre,
            );
            Ellipse2D::new(
                centre + by,
                major.length(),
                minor.length(),
                major.y().atan2(major.x()),
            )
            .ok()
            .map(Curve2D::Ellipse)
        }
        Curve2D::Nurbs(nurbs) => NurbsCurve2D::new(
            nurbs.degree(),
            nurbs.knots().to_vec(),
            nurbs.control_points().iter().map(|&p| p + by).collect(),
            nurbs.weights().to_vec(),
        )
        .ok()
        .map(Curve2D::Nurbs),
    }
}

/// The surface parameters of `edge` at fraction `t` of its domain, unwrapped
/// continuously from the edge's start UV over a walk along its curve. `None`
/// on a surface with no period, and when the walk cannot finish: a value
/// that is not finite, a projection that fails, a step too small to move,
/// or more than `MAX_STEPS` steps.
///
/// A step is taken when its middle and its end each move less than an
/// eighth of a period from the point before, and halved when not: a span of
/// the curve dense in u on few of its parameters, or one that goes round a
/// whole period, does not read as a short step.
fn walked_uv(
    edge: &OrientedPCurveEdge,
    t: f64,
    surface: &FaceSurface,
    periods: (Option<f64>, Option<f64>),
) -> Option<brepkit_math::vec::Point2> {
    use brepkit_math::vec::Point2;
    const FIRST_STEPS: f64 = 16.0;
    const MAX_HALVINGS: u32 = 16;
    const MAX_STEPS: u32 = 4096;
    let finite = |p: Point2| p.x().is_finite() && p.y().is_finite();
    if periods.0.is_none() && periods.1.is_none() || !t.is_finite() || !finite(edge.start_uv) {
        return None;
    }
    let at =
        crate::builder::pcurve_compute::edge_point_at(&edge.curve_3d, edge.start_3d, edge.end_3d);
    // The fractions run from the domain's first end, which is the edge's
    // end when the domain runs against the edge.
    let from = if (at(1.0) - edge.start_3d).length() < (at(0.0) - edge.start_3d).length() {
        1.0
    } else {
        0.0
    };
    let uv_at = |f: f64, near: Point2| {
        surface
            .project_point(at(f))
            .map(|(u, v)| Point2::new(u, v))
            .filter(|&p| finite(p))
            .map(|p| on_copy_near(p, near, periods))
    };
    let small = |a: Point2, b: Point2| {
        periods.0.is_none_or(|p| (b.x() - a.x()).abs() < 0.125 * p)
            && periods.1.is_none_or(|p| (b.y() - a.y()).abs() < 0.125 * p)
    };
    let (mut f, mut uv) = (from, edge.start_uv);
    let first_step = (t - from) / FIRST_STEPS;
    let mut step = first_step;
    let mut halvings = 0;
    for _ in 0..MAX_STEPS {
        let tiny = f64::EPSILON * f.abs().max(1.0);
        if (t - f).abs() <= tiny {
            return Some(uv);
        }
        // A step at least `tiny` long always moves `f`.
        if step.abs() < tiny {
            return None;
        }
        let last = (t - f).abs() <= step.abs();
        let next = if last { t } else { f + step };
        let m = uv_at(0.5 * (f + next), uv)?;
        let n = uv_at(next, m)?;
        if halvings < MAX_HALVINGS && !(small(uv, m) && small(m, n)) {
            step *= 0.5;
            halvings += 1;
            continue;
        }
        if last {
            return Some(n);
        }
        (f, uv) = (next, n);
        halvings = 0;
        step = (2.0 * step)
            .abs()
            .min(first_step.abs())
            .copysign(first_step);
    }
    None
}

/// `uv` moved by whole periods to the copy nearest `near`.
fn on_copy_near(
    uv: brepkit_math::vec::Point2,
    near: brepkit_math::vec::Point2,
    (u_period, v_period): (Option<f64>, Option<f64>),
) -> brepkit_math::vec::Point2 {
    let lift = |x: f64, to: f64, period: Option<f64>| match period {
        Some(p) => x + ((to - x) / p).round() * p,
        None => x,
    };
    brepkit_math::vec::Point2::new(
        lift(uv.x(), near.x(), u_period),
        lift(uv.y(), near.y(), v_period),
    )
}

/// The whole periods that bring `pcurve` onto the copy where `uv` lies:
/// of the copies a few periods either way, the one nearest `uv`.
fn pcurve_shift_to(
    pcurve: &brepkit_math::curves2d::Curve2D,
    uv: brepkit_math::vec::Point2,
    (u_period, v_period): (Option<f64>, Option<f64>),
) -> (f64, f64) {
    use brepkit_math::curves2d::Curve2D;
    let distance = |p: brepkit_math::vec::Point2| match pcurve {
        Curve2D::Line(line) => line.distance_to_point(p),
        Curve2D::Circle(circle) => (circle.evaluate(circle.project(p)) - p).length(),
        Curve2D::Ellipse(ellipse) => (0..=64)
            .map(|k| (ellipse.evaluate(std::f64::consts::TAU * f64::from(k) / 64.0) - p).length())
            .fold(f64::INFINITY, f64::min),
        Curve2D::Nurbs(nurbs) => {
            let (t0, t1) = nurbs.domain();
            (0..=64)
                .map(|k| (nurbs.evaluate((t1 - t0).madd(f64::from(k) / 64.0, t0)) - p).length())
                .fold(f64::INFINITY, f64::min)
        }
    };
    let shifts = |period: Option<f64>| -> Vec<f64> {
        period.map_or_else(
            || vec![0.0],
            |p| (-3..=3).map(|k| f64::from(k) * p).collect(),
        )
    };
    let mut best = (0.0, 0.0, f64::INFINITY);
    for du in shifts(u_period) {
        for dv in shifts(v_period) {
            let d = distance(brepkit_math::vec::Point2::new(uv.x() - du, uv.y() - dv));
            if d < best.2 {
                best = (du, dv, d);
            }
        }
    }
    (best.0, best.1)
}

/// Split parameters on a spline edge by projection onto the part of the
/// spline the edge runs on, in the fractions of its domain that
/// [`find_splits_on_nurbs_section`] gives, and in the edge's order. `None`
/// when the edge is not a spline or its part cannot be cut out.
fn find_splits_on_nurbs_by_projection(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Option<Vec<(f64, Point3)>> {
    use brepkit_math::nurbs::knot_ops::curve_split;
    use brepkit_math::nurbs::projection::project_point_to_curve;
    use brepkit_math::traits::ParametricCurve;
    let EdgeCurve::NurbsCurve(whole) = &edge.curve_3d else {
        return None;
    };
    let (dom0, dom1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
    let dom_span = dom1 - dom0;
    if dom_span.abs() <= f64::EPSILON {
        return None;
    }
    let (lo, hi) = (dom0.min(dom1), dom0.max(dom1));
    let (d0, d1) = ParametricCurve::domain(whole);
    let slack = 1e-9 * (d1 - d0);
    let mut span = whole.clone();
    if lo > d0 + slack {
        span = curve_split(&span, lo).ok()?.1;
    }
    if hi < d1 - slack {
        span = curve_split(&span, hi).ok()?.0;
    }
    // The weave's contact test reads a point within the weld band of the
    // edge as on it (a fitted curve's error), so the split takes the same
    // band. With positive weights the control polygon holds the part, so a
    // point farther than that from its box is off the edge.
    let band = tol * 100.0;
    let boxed = span.weights().iter().all(|&w| w.is_finite() && w > 0.0);
    let (mut bmin, mut bmax) = (span.control_points()[0], span.control_points()[0]);
    for p in span.control_points() {
        bmin = Point3::new(
            bmin.x().min(p.x()),
            bmin.y().min(p.y()),
            bmin.z().min(p.z()),
        );
        bmax = Point3::new(
            bmax.x().max(p.x()),
            bmax.y().max(p.y()),
            bmax.z().max(p.z()),
        );
    }
    let outside = |p: Point3| {
        p.x() < bmin.x() - band
            || p.y() < bmin.y() - band
            || p.z() < bmin.z() - band
            || p.x() > bmax.x() + band
            || p.y() > bmax.y() + band
            || p.z() > bmax.z() + band
    };
    let mut splits: Vec<(f64, Point3)> = Vec::new();
    for &sp in split_pts_3d {
        if boxed && outside(sp) {
            continue;
        }
        let Ok(hit) = project_point_to_curve(&span, sp, tol * 1e-3) else {
            continue;
        };
        if hit.distance > band {
            continue;
        }
        let t = (hit.parameter - dom0) / dom_span;
        if t <= tol || t >= 1.0 - tol {
            continue;
        }
        splits.push((t, sp));
    }
    splits.sort_by(|a, b| a.0.total_cmp(&b.0));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    let first = edge
        .curve_3d
        .evaluate_with_endpoints(dom0, edge.start_3d, edge.end_3d);
    let last = edge
        .curve_3d
        .evaluate_with_endpoints(dom1, edge.start_3d, edge.end_3d);
    if (last - edge.start_3d).length() < (first - edge.start_3d).length() {
        splits.reverse();
    }
    Some(splits)
}

pub(super) fn find_splits_on_nurbs_section(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
    junction_bands: bool,
) -> Vec<(f64, Point3)> {
    let n_samples = 64usize;
    // Hoist the domain: `evaluate_edge_at_t` re-derives `domain_with_endpoints`
    // on EVERY call, and for a trimmed NURBS sub-span that is two iterative
    // point-to-curve projections per evaluation — this finder makes ~115
    // evaluations per candidate point, which turned each pad-fuse face split
    // into ~700ms (the lite magnet scenarios' timeout class). One domain
    // computation per edge, then direct parametric evaluation.
    let (dom0, dom1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
    let dom_span = dom1 - dom0;
    let eval_at = |t: f64| -> Point3 {
        edge.curve_3d
            .evaluate_with_endpoints(t.madd(dom_span, dom0), edge.start_3d, edge.end_3d)
    };
    let samples: Vec<Point3> = (0..=n_samples)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            eval_at(i as f64 / n_samples as f64)
        })
        .collect();
    // T-junction candidates are OTHER SECTIONS' ENDPOINTS: genuine on-curve
    // junctions whose nearest of 64 samples can sit half a sample step away,
    // and whose positions carry marched (~1e-6) rounding — so that caller
    // widens the broad phase to sampling resolution and the fine gate to the
    // weld scale. The boundary-split caller keeps the narrow bands: its
    // candidate set is arbitrary and the honeycomb pcut1 thin walls carry
    // real structure at the widened scale (foil-refuted, twice).
    let max_gap = if junction_bands {
        samples
            .windows(2)
            .map(|w| (w[1] - w[0]).length())
            .fold(0.0_f64, f64::max)
    } else {
        0.0
    };
    let broad_band = max_gap.max(tol * 100.0);
    let fine_band = if junction_bands { tol * 100.0 } else { tol };
    let mut splits: Vec<(f64, Point3)> = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        // Nearest sample, then ternary-refine the distance over the two
        // neighbouring segments.
        let (mut best_i, mut best_d) = (0usize, f64::MAX);
        for (i, s) in samples.iter().enumerate() {
            let d = (*s - sp).length();
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        if best_d > broad_band {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let (mut lo, mut hi) = (
            best_i.saturating_sub(1) as f64 / n_samples as f64,
            (best_i + 1).min(n_samples) as f64 / n_samples as f64,
        );
        for _ in 0..50 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if (eval_at(m1) - sp).length() < (eval_at(m2) - sp).length() {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        let t = f64::midpoint(lo, hi);
        if (eval_at(t) - sp).length() > fine_band {
            continue;
        }
        if t <= tol || t >= 1.0 - tol {
            continue;
        }
        splits.push((t, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    // `t` runs over the natural knot domain, which for the REVERSE twin of a
    // section pair runs end→start relative to the edge. The piece-building
    // loop walks `start_3d` → `end_3d`, so hand it splits in EDGE order —
    // ascending-t pieces on a reversed twin overlap once there are ≥2 splits.
    if (samples[n_samples] - edge.start_3d).length() < (samples[0] - edge.start_3d).length() {
        splits.reverse();
    }
    splits
}

/// Find split parameters on an open CIRCLE SECTION edge, using the
/// SHORTER-arc convention that `evaluate_edge_at_t` uses.
///
/// Circle section arcs are ≤ π by construction (the FF closed-circle emitter
/// splits longer spans), and `split_face_2d` pushes each section as a
/// forward/reverse PAIR. `domain_with_endpoints` assumes CCW traversal, so for
/// the REVERSE twin it returns the LONG complement span (e.g. 315 deg for a
/// 45 deg corner arc) — under which a point on the circle but OUTSIDE the arc
/// normalizes to an interior `t`, and the split evaluator (shorter-arc) then
/// mints a phantom vertex on the true arc's interior, breaking partition
/// alignment between coincident caps. The shorter-arc parameterization here
/// matches the evaluator for both twins. Boundary-edge splitting keeps the
/// CCW-domain convention (`find_splits_on_circle`) — boundary arcs may
/// genuinely exceed π, as may ellipse sections (no π-split guarantee), so
/// both stay on the domain-based finders.
pub(super) fn find_splits_on_section_arc(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let EdgeCurve::Circle(circle) = &edge.curve_3d else {
        return Vec::new();
    };
    let a0 = circle.project(edge.start_3d);
    let delta = shorter_arc_delta(circle.project(edge.end_3d) - a0);
    if delta.abs() < 1e-14 {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = shorter_arc_delta(angle - a0) / delta;
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Find split parameters on an ELLIPSE SECTION edge — the ellipse twin of
/// [`find_splits_on_section_arc`].
///
/// Sections are pushed as forward/reverse pairs, and the CCW
/// `domain_with_endpoints` span for the reverse twin is the LONG complement.
/// A point on the ellipse but OUTSIDE the section's own arc — e.g. an
/// endpoint of ANOTHER window of the same multi-window ellipse (a socket
/// chamfer plane cutting a pad cylinder yields one closed ellipse with
/// several in-face windows) — then normalizes to an interior `t`, and the
/// shorter-arc split evaluator mints a phantom vertex inside the true arc.
/// Only the reverse twin is affected, so the pair desyncs: the wire builders
/// trace zero-area lens cells between the whole forward copy and the split
/// reverse pieces, the cells attach as degenerate out-and-back hole wires,
/// and the shell opens. The shorter-arc parameterization here matches
/// `evaluate_edge_at_t` for both twins; an ellipse section spanning > π is
/// already outside that evaluator's contract, so matching it is strictly
/// more consistent than the domain-based normalization.
pub(super) fn find_splits_on_section_ellipse(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let EdgeCurve::Ellipse(ellipse) = &edge.curve_3d else {
        return Vec::new();
    };
    let a0 = ellipse.project(edge.start_3d);
    let delta = shorter_arc_delta(ellipse.project(edge.end_3d) - a0);
    if delta.abs() < 1e-14 {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let angle = ellipse.project(sp);
        let closest = ellipse.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = shorter_arc_delta(angle - a0) / delta;
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Find split parameters on an ellipse edge.
pub(super) fn find_splits_on_ellipse(
    ellipse: &brepkit_math::curves::Ellipse3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    // An open arc spans its stored edge's counter-clockwise range; read in
    // the traversal's order, a reversed arc's span is its complement, and a
    // point on the ellipse just past the arc's end (a strut notch's arc ending
    // a hair short of the seam vertex) split it into a detour through there.
    let closed = (edge.start_3d - edge.end_3d).length() < tol;
    let (stored_start, stored_end) = if edge.forward || closed {
        (edge.start_3d, edge.end_3d)
    } else {
        (edge.end_3d, edge.start_3d)
    };
    let (t0, t1) = edge
        .curve_3d
        .domain_with_endpoints(stored_start, stored_end);
    let span = t1 - t0;
    if span.abs() < 1e-14 {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let angle = ellipse.project(sp);
        let closest = ellipse.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let along = normalize_angle_in_span(angle, t0, span);
        let t_norm = if edge.forward || closed {
            along
        } else {
            1.0 - along
        };
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::nurbs::fitting::interpolate;
    use brepkit_math::vec::{Point2, Vec2};

    fn parabola_section_edge(reversed: bool) -> OrientedPCurveEdge {
        let pts: Vec<Point3> = (0..=8)
            .map(|k| {
                let x = -2.0 + 4.0 * f64::from(k) / 8.0;
                Point3::new(x, x * x, 0.0)
            })
            .collect();
        let nurbs = interpolate(&pts, 3).unwrap();
        let (start_3d, end_3d) = if reversed {
            (pts[8], pts[0])
        } else {
            (pts[0], pts[8])
        };
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(nurbs),
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(1.0, 0.0),
            start_3d,
            end_3d,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// The junction-band discriminant: an on-curve point midway between the
    /// finder's 64 samples sits ~half a sample step from its nearest sample,
    /// so the narrow broad phase (boundary-split calibration, honeycomb
    /// foil) must reject it while the junction bands (T-splits at other
    /// sections' endpoints — the wedge-corner junction) must accept it.
    #[test]
    fn junction_bands_accept_on_curve_points_between_samples() {
        let edge = parabola_section_edge(false);
        let eval = |t: f64| evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
        let sp = eval(0.507);
        let tol = 1e-7;
        assert!(
            find_splits_on_nurbs_section(&edge, &[sp], tol, false).is_empty(),
            "narrow bands must reject a mid-segment candidate"
        );
        let splits = find_splits_on_nurbs_section(&edge, &[sp], tol, true);
        assert_eq!(splits.len(), 1, "junction bands must accept the junction");
        assert!((splits[0].1 - sp).length() < 1e-6);
    }

    /// A split point's distance from the ends is a length: on a 166 mm
    /// edge, a point 12.8 microns short of the end splits it, and one within
    /// the tolerance of the end is the end itself.
    #[test]
    fn line_splits_read_their_distance_from_the_ends_as_a_length() {
        let (start_3d, end_3d) = (Point3::new(0.48, 0.0, 0.0), Point3::new(166.63, 0.0, 0.0));
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(166.15, 0.0),
            start_3d,
            end_3d,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let near = Point3::new(166.63 - 1.28e-5, 0.0, 0.0);
        let at = Point3::new(166.63 - 5e-8, 0.0, 0.0);
        let splits = find_splits_on_line(&edge, &[near, at], 1e-7);
        assert_eq!(splits.len(), 1, "{splits:?}");
        assert!((splits[0].1 - near).length() < 1e-9);
    }

    #[test]
    fn nurbs_section_splits_ordered_along_forward_edge() {
        let edge = parabola_section_edge(false);
        let eval = |t: f64| evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
        let (sp_a, sp_b) = (eval(0.3), eval(0.7));
        let splits = find_splits_on_nurbs_section(&edge, &[sp_b, sp_a], 1e-3, false);
        assert_eq!(splits.len(), 2);
        assert!((splits[0].1 - sp_a).length() < 1e-6);
        assert!((splits[1].1 - sp_b).length() < 1e-6);
    }

    fn ellipse_section_edge(
        ellipse: &brepkit_math::curves::Ellipse3D,
        a0: f64,
        a1: f64,
        forward: bool,
    ) -> OrientedPCurveEdge {
        let (s, e) = if forward {
            (ellipse.evaluate(a0), ellipse.evaluate(a1))
        } else {
            (ellipse.evaluate(a1), ellipse.evaluate(a0))
        };
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Ellipse(ellipse.clone()),
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(1.0, 0.0),
            start_3d: s,
            end_3d: e,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    /// A multi-window ellipse (one closed cylinder×plane ellipse emitted as
    /// several in-face windows): a point from ANOTHER window lies on the
    /// ellipse but outside this section's own arc. The CCW-domain finder
    /// normalized it to an interior `t` on the REVERSE twin only (the long
    /// complement span), minting a phantom split that desynced the twins.
    #[test]
    fn ellipse_section_reverse_twin_ignores_other_window_points() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::vec::Vec3;

        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            1.0,
        )
        .unwrap();
        let other_window_pt = ellipse.evaluate(3.0);
        let junction = ellipse.evaluate(0.6);

        for forward in [true, false] {
            let edge = ellipse_section_edge(&ellipse, 0.2, 1.0, forward);
            let phantom = find_splits_on_section_ellipse(&edge, &[other_window_pt], 1e-7);
            assert!(
                phantom.is_empty(),
                "other-window point must not split the {} twin",
                if forward { "forward" } else { "reverse" }
            );

            let real = find_splits_on_section_ellipse(&edge, &[junction], 1e-7);
            assert_eq!(real.len(), 1);
            let minted = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, real[0].0);
            assert!(
                (minted - junction).length() < 1e-9,
                "split evaluator must mint the junction point on both twins"
            );
        }

        // The boundary finder reads a reversed arc by its stored span too.
        let rev = ellipse_section_edge(&ellipse, 0.2, 1.0, false);
        let boundary = find_splits_on_ellipse(&ellipse, &rev, &[other_window_pt], 1e-7);
        assert!(
            boundary.is_empty(),
            "a point off a reversed boundary arc must not split it"
        );
    }

    /// A CLOSED rim on a cylinder, seamed at 3pi/2 (i.e. NOT at the circle's
    /// own angular origin), with split points bracketing that seam.
    fn closed_rim_edge(forward: bool) -> (FaceSurface, OrientedPCurveEdge, Vec<Point3>) {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::TAU;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let seam_angle = 1.5 * std::f64::consts::PI;
        let seam = circle.evaluate(seam_angle);
        let (u_seam, _) = surface.project_point(seam).unwrap();
        // `sample_edge_to_uv` walks the CURVE, ignoring the traversal flag, so a
        // closed rim always reports its span as `u_seam -> u_seam + 2pi`.
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle.clone()),
            pcurve: Curve2D::Line(
                Line2D::new(Point2::new(u_seam, 0.0), Vec2::new(1.0, 0.0)).unwrap(),
            ),
            start_uv: Point2::new(u_seam, 0.0),
            end_uv: Point2::new(u_seam + TAU, 0.0),
            start_3d: seam,
            end_3d: seam,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        // Two splits straddling the seam, plus the seam-antipodal one the
        // periodic path always contributes.
        let splits = vec![
            circle.evaluate(seam_angle - 0.3),
            circle.evaluate(seam_angle + 0.3),
            circle.evaluate(seam_angle + std::f64::consts::PI),
        ];
        (surface, edge, splits)
    }

    /// Total angular sweep of a chain of rim pieces, measured in the traversal
    /// direction. A correct partition of a full ring sweeps exactly 2pi.
    fn chain_sweep(pieces: &[OrientedPCurveEdge], surface: &FaceSurface, forward: bool) -> f64 {
        use std::f64::consts::TAU;
        pieces
            .iter()
            .map(|p| {
                let (u0, _) = surface.project_point(p.start_3d).unwrap();
                let (u1, _) = surface.project_point(p.end_3d).unwrap();
                let d = if forward { u1 - u0 } else { u0 - u1 };
                d.rem_euclid(TAU)
            })
            .sum()
    }

    #[test]
    fn closed_rim_splits_tile_the_ring_exactly_once() {
        use std::f64::consts::TAU;
        let (surface, edge, splits) = closed_rim_edge(true);
        let start_u = edge.start_uv.x();
        let pieces = split_boundary_edges_at_3d_points(vec![edge], &splits, None, &surface, 1e-7);

        assert_eq!(pieces.len(), 4, "3 splits must yield 4 rim pieces");
        // The pieces must chain end-to-end and cover the ring exactly once.
        // Anchoring the split parameters at the circle's angular origin instead
        // of the edge's own start point double-covers the wedge between the two
        // anchors, which shows up here as a 4pi sweep.
        for w in pieces.windows(2) {
            assert!(
                (w[0].end_3d - w[1].start_3d).length() < 1e-9,
                "rim pieces must chain end-to-end"
            );
        }
        let sweep = chain_sweep(&pieces, &surface, true);
        assert!(
            (sweep - TAU).abs() < 1e-9,
            "rim pieces must sweep exactly 2pi, got {sweep}"
        );
        // Stored UV must walk one full period in the traversal direction with
        // shared joints — a raw principal-value projection would drop the
        // interior joints back into [0, 2pi) and break the monotone strip.
        assert!((pieces[0].start_uv.x() - start_u).abs() < 1e-9);
        assert!(
            (pieces[3].end_uv.x() - (start_u + TAU)).abs() < 1e-9,
            "forward rim must close a period ABOVE its start, got {}",
            pieces[3].end_uv.x()
        );
        for w in pieces.windows(2) {
            assert!((w[0].end_uv.x() - w[1].start_uv.x()).abs() < 1e-9);
            assert!(
                w[1].start_uv.x() > w[0].start_uv.x(),
                "forward rim UV must increase monotonically"
            );
        }
    }

    #[test]
    fn closed_rim_splits_tile_the_ring_on_a_reversed_traversal() {
        // The top rim of a cylinder is traversed CW (`forward = false`) while
        // its stored UV span still runs CCW, so the piece UVs must follow the
        // TRAVERSAL or the ring stops reading as a monotone walk across the
        // cut-open strip.
        use std::f64::consts::TAU;
        let (surface, edge, splits) = closed_rim_edge(false);
        let start_u = edge.start_uv.x();
        let pieces = split_boundary_edges_at_3d_points(vec![edge], &splits, None, &surface, 1e-7);

        assert_eq!(pieces.len(), 4, "3 splits must yield 4 rim pieces");
        let sweep = chain_sweep(&pieces, &surface, false);
        assert!(
            (sweep - TAU).abs() < 1e-9,
            "rim pieces must sweep exactly 2pi, got {sweep}"
        );
        // Stored UV walks one full period in the traversal direction, and each
        // joint is shared, so the strip is monotone decreasing.
        assert!((pieces[0].start_uv.x() - start_u).abs() < 1e-9);
        assert!(
            (pieces[3].end_uv.x() - (start_u - TAU)).abs() < 1e-9,
            "reversed rim must close a period BELOW its start, got {}",
            pieces[3].end_uv.x()
        );
        for w in pieces.windows(2) {
            assert!((w[0].end_uv.x() - w[1].start_uv.x()).abs() < 1e-9);
            assert!(
                w[1].start_uv.x() < w[0].start_uv.x(),
                "reversed rim UV must decrease monotonically"
            );
        }
    }

    /// A closed rim whose circle turns against the cylinder's u (its normal
    /// opposes the axis, as on a mirrored cylinder), seamed at its own
    /// parameter 3pi/2: its samples run from `u_seam` down to `u_seam - 2pi`.
    fn closed_rim_edge_against_u(forward: bool) -> (FaceSurface, OrientedPCurveEdge, Vec<Point3>) {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::TAU;

        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 5.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let circle = Circle3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            5.0,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let seam_angle = 1.5 * std::f64::consts::PI;
        let seam = circle.evaluate(seam_angle);
        let (u_seam, _) = surface.project_point(seam).unwrap();
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle.clone()),
            pcurve: Curve2D::Line(
                Line2D::new(Point2::new(u_seam, 0.0), Vec2::new(-1.0, 0.0)).unwrap(),
            ),
            start_uv: Point2::new(u_seam, 0.0),
            end_uv: Point2::new(u_seam - TAU, 0.0),
            start_3d: seam,
            end_3d: seam,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let splits = vec![
            circle.evaluate(seam_angle - 0.3),
            circle.evaluate(seam_angle + 0.3),
            circle.evaluate(seam_angle + std::f64::consts::PI),
        ];
        (surface, edge, splits)
    }

    #[test]
    fn a_rim_turning_against_u_splits_where_its_points_project() {
        use std::f64::consts::TAU;
        for forward in [true, false] {
            let (surface, edge, splits) = closed_rim_edge_against_u(forward);
            let start_u = edge.start_uv.x();
            let pieces =
                split_boundary_edges_at_3d_points(vec![edge], &splits, None, &surface, 1e-7);
            assert_eq!(pieces.len(), 4, "3 splits must yield 4 rim pieces");
            // Forward, the rim runs down u; reversed, up.
            let sign: f64 = if forward { -1.0 } else { 1.0 };
            assert!(
                (pieces[3].end_uv.x() - sign.madd(TAU, start_u)).abs() < 1e-9,
                "forward {forward}: the rim closes at {}",
                pieces[3].end_uv.x()
            );
            for piece in &pieces {
                // Each joint's stored u is where its point projects, a whole
                // number of periods away.
                let (u, _) = surface.project_point(piece.end_3d).unwrap();
                let turns = (piece.end_uv.x() - u) / TAU;
                assert!(
                    (turns - turns.round()).abs() < 1e-9,
                    "forward {forward}: a joint at u {} projects to {u}",
                    piece.end_uv.x()
                );
            }
        }
    }

    #[test]
    fn nurbs_section_splits_ordered_along_reversed_twin() {
        // The reverse twin stores the SAME curve but swapped endpoints, so
        // ascending natural-domain `t` runs end→start, while the edge's own
        // fraction runs start→end; the splits must come back ordered from
        // `start_3d` (nearest first) or the piece-building loop emits
        // overlapping pieces.
        let edge = parabola_section_edge(true);
        let eval = |t: f64| evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
        assert!((eval(0.0) - edge.start_3d).length() < 1e-9);
        let (sp_a, sp_b) = (eval(0.3), eval(0.7));
        let splits = find_splits_on_nurbs_section(&edge, &[sp_b, sp_a], 1e-3, false);
        assert_eq!(splits.len(), 2);
        assert!((splits[0].1 - sp_a).length() < 1e-6);
        assert!((splits[1].1 - sp_b).length() < 1e-6);
        let d0 = (splits[0].1 - edge.start_3d).length();
        let d1 = (splits[1].1 - edge.start_3d).length();
        assert!(d0 < d1, "splits must walk start_3d → end_3d");
    }

    #[test]
    fn line_split_accepts_weld_band_probe_and_returns_the_foot() {
        // A probe 5 tol off the line must still anchor (section endpoints
        // carry a few tol of clip rounding), and the returned point must be
        // the exact foot on the line so wires stay on-edge.
        let tol = 1e-7;
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Line,
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: Point2::new(0.0, 0.0),
            end_uv: Point2::new(10.0, 0.0),
            start_3d: Point3::new(0.0, 0.0, 0.0),
            end_3d: Point3::new(10.0, 0.0, 0.0),
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let probe = Point3::new(4.0, 5.0 * tol, 0.0);
        let splits = find_splits_on_line(&edge, &[probe], tol);
        assert_eq!(splits.len(), 1, "weld-band probe must anchor");
        let (t, foot) = splits[0];
        assert!((t - 0.4).abs() < 1e-9);
        assert!(
            foot.y().abs() < 1e-15,
            "returned point must be the foot on the line, got y={}",
            foot.y()
        );
        // Beyond the weld band: rejected.
        let far = Point3::new(4.0, 200.0 * tol, 0.0);
        assert!(find_splits_on_line(&edge, &[far], tol).is_empty());
    }

    /// A closed planar ellipse split once, away from its vertex, comes out
    /// in four pieces, so neither half shares both endpoints with the other
    /// or with a chord across the rim.
    #[test]
    fn closed_planar_ellipse_split_once_is_paved_at_both_halves() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::vec::Vec3;

        let normal = Vec3::new(0.0, 0.0, 1.0);
        let ellipse = Ellipse3D::new(Point3::new(0.0, 0.0, 0.0), normal, 4.0, 2.0).unwrap();
        let vertex = ellipse.evaluate(0.7);
        let opposite = ellipse.evaluate(0.7 + std::f64::consts::PI);
        let surface = FaceSurface::Plane { normal, d: 0.0 };
        let frame = PlaneFrame::from_plane_face(normal, &[vertex, opposite]);
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Ellipse(ellipse.clone()),
            pcurve: Curve2D::Line(Line2D::new(Point2::new(0.0, 0.0), Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: frame.project(vertex),
            end_uv: frame.project(vertex),
            start_3d: vertex,
            end_3d: vertex,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let pieces = split_boundary_edges_at_3d_points(
            vec![edge],
            &[opposite],
            Some(&frame),
            &surface,
            1e-7,
        );
        assert_eq!(pieces.len(), 4);
        for (i, a) in pieces.iter().enumerate() {
            assert!(
                (a.start_3d - a.end_3d).length() > 1e-6,
                "piece {i} is degenerate"
            );
            for b in &pieces[i + 1..] {
                let same = (a.start_3d - b.start_3d).length() < 1e-9
                    && (a.end_3d - b.end_3d).length() < 1e-9
                    || (a.start_3d - b.end_3d).length() < 1e-9
                        && (a.end_3d - b.start_3d).length() < 1e-9;
                assert!(!same, "two pieces share both endpoints");
            }
            for p in [a.start_3d, a.end_3d] {
                assert!((ellipse.evaluate(ellipse.project(p)) - p).length() < 1e-9);
            }
        }
    }

    /// A whole circle of radius 25 round the z axis (a torus's outer
    /// equator, a sphere's equator) seamed away from its circle's own zero
    /// angle and split three times tiles the ring once, walked either way:
    /// the split parameters run from the edge's start, not from the
    /// circle's origin.
    #[test]
    fn a_closed_ring_split_from_its_seam_tiles_the_ring_once() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{SphericalSurface, ToroidalSurface};
        use brepkit_math::vec::Vec3;
        use std::f64::consts::{PI, TAU};

        let origin = Point3::new(0.0, 0.0, 0.0);
        for surface in [
            FaceSurface::Torus(ToroidalSurface::new(origin, 20.0, 5.0).unwrap()),
            FaceSurface::Sphere(SphericalSurface::new(origin, 25.0).unwrap()),
        ] {
            let circle = Circle3D::new_with_ref(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                25.0,
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap();
            let seam_angle = 1.5 * PI;
            let seam = circle.evaluate(seam_angle);
            let splits = vec![
                circle.evaluate(seam_angle - 0.3),
                circle.evaluate(seam_angle + 0.3),
                circle.evaluate(seam_angle + PI),
            ];
            for forward in [true, false] {
                let edge = OrientedPCurveEdge {
                    curve_3d: EdgeCurve::Circle(circle.clone()),
                    pcurve: Curve2D::Line(
                        Line2D::new(Point2::new(seam_angle, 0.0), Vec2::new(1.0, 0.0)).unwrap(),
                    ),
                    start_uv: Point2::new(seam_angle, 0.0),
                    end_uv: Point2::new(seam_angle + TAU, 0.0),
                    start_3d: seam,
                    end_3d: seam,
                    forward,
                    source_edge_idx: None,
                    pave_block_id: None,
                };
                let pieces =
                    split_boundary_edges_at_3d_points(vec![edge], &splits, None, &surface, 1e-7);
                assert_eq!(
                    pieces.len(),
                    4,
                    "forward {forward}: {} pieces",
                    pieces.len()
                );
                for w in pieces.windows(2) {
                    assert!(
                        (w[0].end_3d - w[1].start_3d).length() < 1e-9,
                        "forward {forward}: pieces do not chain"
                    );
                }
                let sweep = chain_sweep(&pieces, &surface, forward);
                assert!(
                    (sweep - TAU).abs() < 1e-9,
                    "{} forward {forward}: the pieces sweep {sweep}",
                    surface.type_tag()
                );
            }
        }
    }

    /// A quarter of a torus meridian walked against its circle, from the
    /// tube's top down to its outer equator, splits at a point on that
    /// quarter and at none on the other three.
    #[test]
    fn a_reversed_meridian_splits_only_on_its_own_arc() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_math::vec::Vec3;
        use std::f64::consts::FRAC_PI_2;

        let surface = FaceSurface::Torus(
            ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 20.0, 5.0).unwrap(),
        );
        // Counterclockwise about -y the circle turns from +x up to +z, so
        // the stored arc from the equator to the top is the quarter.
        let circle =
            Circle3D::new(Point3::new(20.0, 0.0, 0.0), Vec3::new(0.0, -1.0, 0.0), 5.0).unwrap();
        let (top, equator) = (Point3::new(20.0, 0.0, 5.0), Point3::new(25.0, 0.0, 0.0));
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle),
            pcurve: Curve2D::Line(
                Line2D::new(Point2::new(0.0, FRAC_PI_2), Vec2::new(0.0, -1.0)).unwrap(),
            ),
            start_uv: Point2::new(0.0, FRAC_PI_2),
            end_uv: Point2::new(0.0, 0.0),
            start_3d: top,
            end_3d: equator,
            forward: false,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let inner_equator = Point3::new(15.0, 0.0, 0.0);
        let pieces = split_boundary_edges_at_3d_points(
            vec![edge.clone()],
            &[inner_equator],
            None,
            &surface,
            1e-7,
        );
        assert_eq!(pieces.len(), 1, "split at a point off the quarter");

        let half = std::f64::consts::FRAC_1_SQRT_2 * 5.0;
        let between = Point3::new(20.0 + half, 0.0, half);
        let pieces =
            split_boundary_edges_at_3d_points(vec![edge], &[between], None, &surface, 1e-7);
        assert_eq!(pieces.len(), 2, "the quarter is not split where it runs");
        assert!((pieces[0].start_3d - top).length() < 1e-9);
        assert!((pieces[0].end_3d - between).length() < 1e-9);
        assert!((pieces[1].start_3d - between).length() < 1e-9);
        assert!((pieces[1].end_3d - equator).length() < 1e-9);
    }
}

#[cfg(test)]
mod hole_copy_tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_math::vec::{Point2, Vec3};

    /// The end of `pcurve` nearest `uv`, read in the curve's own terms.
    fn nearest_end(pcurve: &Curve2D, uv: Point2) -> Point2 {
        match pcurve {
            Curve2D::Line(line) => line.evaluate(line.project(uv)),
            Curve2D::Nurbs(n) => {
                let k = n.knots();
                let (a, b) = (n.evaluate(k[0]), n.evaluate(k[k.len() - 1]));
                if (a - uv).length() <= (b - uv).length() {
                    a
                } else {
                    b
                }
            }
            Curve2D::Circle(c) => c.evaluate(c.project(uv)),
            Curve2D::Ellipse(e) => (0..=360)
                .map(|k| e.evaluate(f64::from(k).to_radians()))
                .min_by(|p, q| (*p - uv).length().total_cmp(&(*q - uv).length()))
                .unwrap(),
        }
    }

    /// A woven hole's spline edge and a section's end 5e-7 off it, the error
    /// of a fitted curve: the weave's contact test takes the end as on the
    /// edge, and the split makes the junction there.
    #[test]
    fn a_hole_spline_splits_at_a_section_end_its_fit_error_off_it() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let angle = 0.2_f64;
        let (a, b) = (
            Point3::new(angle.cos(), angle.sin(), 0.0),
            Point3::new(angle.cos(), angle.sin(), 2.0),
        );
        let uv = |p: Point3| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u, v)
        };
        let ruling =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![a, b], vec![1.0, 1.0]).unwrap();
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(ruling),
            pcurve: Curve2D::Line(Line2D::new(uv(a), uv(b) - uv(a)).unwrap()),
            start_uv: uv(a),
            end_uv: uv(b),
            start_3d: a,
            end_3d: b,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let off = Point3::new(angle.cos() * (1.0 + 5e-7), angle.sin() * (1.0 + 5e-7), 1.0);
        let pieces = split_hole_edges_at_3d_points(vec![edge], &[off], None, &surface, 1e-7);
        assert_eq!(
            pieces.len(),
            2,
            "the spline did not split at the section's end"
        );
        assert!((pieces[0].end_3d - off).length() < 1e-12);
    }

    /// A spline rim on a unit cylinder at z = 3 over u = 0 to 5, through
    /// poles at every half unit of u, whose first four fifths in u lie on
    /// the first tenth of its knots; with its poles.
    fn uneven_rim(surface: &FaceSurface) -> (OrientedPCurveEdge, Vec<Point3>) {
        let z = 3.0;
        let poles: Vec<Point3> = (0..=10)
            .map(|k| surface.evaluate(0.5 * f64::from(k), z).unwrap())
            .collect();
        let mut knots = vec![0.0, 0.0];
        knots.extend((1..=8).map(|k| 0.0125 * f64::from(k)));
        knots.extend([0.55, 1.0, 1.0]);
        let rim = NurbsCurve::new(1, knots, poles.clone(), vec![1.0; poles.len()]).unwrap();
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(rim),
            pcurve: Curve2D::Line(
                Line2D::new(
                    Point2::new(0.0, z),
                    Point2::new(5.0, z) - Point2::new(0.0, z),
                )
                .unwrap(),
            ),
            start_uv: Point2::new(0.0, z),
            end_uv: Point2::new(5.0, z),
            start_3d: poles[0],
            end_3d: poles[10],
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        (edge, poles)
    }

    fn unit_cylinder() -> FaceSurface {
        FaceSurface::Cylinder(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap(),
        )
    }

    /// A woven hole's spline rim with uneven knots ([`uneven_rim`]): a split
    /// at u = 3.9 sits at a tenth of the way by parameter, and still takes
    /// the copy of the period the rim runs on there.
    #[test]
    fn a_hole_spline_with_uneven_knots_splits_on_its_own_copy() {
        let surface = unit_cylinder();
        let (edge, poles) = uneven_rim(&surface);
        let split = poles[7] + (poles[8] - poles[7]) * 0.8;
        let pieces = split_hole_edges_at_3d_points(vec![edge], &[split], None, &surface, 1e-7);
        assert_eq!(pieces.len(), 2);
        let u = pieces[0].end_uv.x();
        assert!((u - 3.9).abs() < 0.05, "the split took u = {u}");
        assert!((pieces[1].start_uv.x() - u).abs() < 1e-12);
    }

    /// The walk along a spline returns for a fraction or a start UV that is
    /// not finite, and gives no copy; for a finite fraction it gives one.
    #[test]
    fn the_walk_along_a_spline_returns_on_values_that_are_not_finite() {
        let surface = unit_cylinder();
        let periods = crate::builder::pcurve_compute::surface_periods(&surface);
        let (edge, _) = uneven_rim(&surface);
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(walked_uv(&edge, t, &surface, periods).is_none(), "t = {t}");
        }
        let mut lost = edge.clone();
        lost.start_uv = Point2::new(f64::NAN, 3.0);
        assert!(walked_uv(&lost, 0.05, &surface, periods).is_none());
        let walked = walked_uv(&edge, 0.1, &surface, periods).unwrap();
        assert!(
            (walked.x() - 4.0).abs() < 1e-9,
            "the walk took u = {}",
            walked.x()
        );
    }

    /// A woven hole's rim arc on a cylinder that runs from u = 0 to u = 5,
    /// more than half a turn, split at four fifths of the way: the split
    /// stays at u = 4, inside the arc's own span, and both pieces run on in
    /// u as the arc does.
    #[test]
    fn a_hole_arc_over_half_a_turn_splits_inside_its_own_span() {
        use brepkit_math::curves::Circle3D;
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let z = 3.0;
        let on = |u: f64| surface.evaluate(u, z).unwrap();
        let (a, b) = (on(0.0), on(5.0));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let (ta, tb) = (circle.project(a), circle.project(b));
        let rim = EdgeCurve::Circle(circle);
        assert!((tb - ta).rem_euclid(std::f64::consts::TAU) > std::f64::consts::PI);
        let edge = OrientedPCurveEdge {
            curve_3d: rim,
            pcurve: Curve2D::Line(
                Line2D::new(
                    Point2::new(0.0, z),
                    Point2::new(5.0, z) - Point2::new(0.0, z),
                )
                .unwrap(),
            ),
            start_uv: Point2::new(0.0, z),
            end_uv: Point2::new(5.0, z),
            start_3d: a,
            end_3d: b,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let pieces = split_hole_edges_at_3d_points(vec![edge], &[on(4.0)], None, &surface, 1e-7);
        assert_eq!(pieces.len(), 2);
        let us: Vec<f64> = pieces
            .iter()
            .flat_map(|p| [p.start_uv.x(), p.end_uv.x()])
            .collect();
        assert!(
            us.windows(2).all(|w| w[1] >= w[0] - 1e-9) && (us[1] - 4.0).abs() < 1e-9,
            "{us:?}"
        );
    }

    /// A woven hole's spline edge, a ruling of a cylinder laid out on the
    /// copy of the period at u = 6.5, split where a section meets it: the
    /// new vertex and both pieces' pcurves stay on that copy.
    #[test]
    fn a_hole_edge_on_a_later_copy_of_the_period_splits_on_that_copy() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let surface = FaceSurface::Cylinder(cyl);
        let angle = 0.2_f64;
        let (a, b) = (
            Point3::new(angle.cos(), angle.sin(), 0.0),
            Point3::new(angle.cos(), angle.sin(), 2.0),
        );
        // The ruling laid out a period past the surface's own u for it.
        let lift = |p: Point3| {
            let (u, v) = surface.project_point(p).unwrap();
            Point2::new(u + std::f64::consts::TAU, v)
        };
        let (ua, ub) = (lift(a), lift(b));
        let u = ua.x();
        let ruling =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![a, b], vec![1.0, 1.0]).unwrap();
        let edge = OrientedPCurveEdge {
            curve_3d: EdgeCurve::NurbsCurve(ruling),
            pcurve: Curve2D::Line(Line2D::new(ua, ub - ua).unwrap()),
            start_uv: ua,
            end_uv: ub,
            start_3d: a,
            end_3d: b,
            forward: true,
            source_edge_idx: None,
            pave_block_id: None,
        };
        let middle = Point3::new(angle.cos(), angle.sin(), 1.0);
        let pieces = split_hole_edges_at_3d_points(vec![edge], &[middle], None, &surface, 1e-7);
        assert_eq!(pieces.len(), 2);
        for piece in &pieces {
            for uv in [piece.start_uv, piece.end_uv] {
                assert!(
                    (uv.x() - u).abs() < 1e-9,
                    "{:?} -> {:?}",
                    piece.start_uv,
                    piece.end_uv
                );
                assert!(
                    (nearest_end(&piece.pcurve, uv) - uv).length() < 1e-6,
                    "the pcurve of {:?} -> {:?} is off its ends",
                    piece.start_uv,
                    piece.end_uv
                );
            }
        }
    }
}

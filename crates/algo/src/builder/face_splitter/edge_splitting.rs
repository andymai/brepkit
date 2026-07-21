//! Curve-specific edge splitting at 3D intersection points.

use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

use super::super::pcurve_compute::{
    compute_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface, shorter_arc_delta,
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
#[allow(clippy::too_many_lines)]
pub(super) fn split_boundary_edges_at_3d_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    frame: Option<&PlaneFrame>,
    surface: &FaceSurface,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    let mut result = Vec::new();
    for edge in edges {
        let splits = match &edge.curve_3d {
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
            EdgeCurve::NurbsCurve(_) => find_splits_on_nurbs_section(&edge, split_pts_3d, tol),
            EdgeCurve::Line => find_splits_on_line(&edge, split_pts_3d, tol),
        };

        if splits.is_empty() {
            result.push(edge);
            continue;
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
        let closed_ring_dir = (matches!(edge.curve_3d, EdgeCurve::Circle(_))
            && matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
            && (edge.start_3d - edge.end_3d).length() < tol)
            .then(|| if edge.forward { 1.0_f64 } else { -1.0_f64 });
        let mut prev_uv = edge.start_uv;
        let mut prev_3d = edge.start_3d;
        for &(t, pt) in &splits {
            // Circle splits carry the exact on-curve foot; re-evaluating via
            // `evaluate_edge_at_t` would re-apply the CCW-span convention
            // that iso-v rim splits deliberately bypass. NURBS splits carry
            // the CANDIDATE point itself (validated within `tol` of the
            // curve): it is the neighbouring face's anchor, and adopting it
            // verbatim gives both faces the identical split vertex.
            let split_3d = if matches!(
                edge.curve_3d,
                EdgeCurve::Circle(_) | EdgeCurve::NurbsCurve(_)
            ) {
                pt
            } else {
                evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t)
            };
            let split_uv = if let Some(f) = frame {
                f.project(split_3d)
            } else if let Some(dir) = closed_ring_dir {
                brepkit_math::vec::Point2::new(
                    std::f64::consts::TAU.mul_add(dir * t, edge.start_uv.x()),
                    edge.start_uv.y(),
                )
            } else if circle_iso_v_rim {
                // Interpolate within the edge's own UV span: an iso-v rim's
                // pcurve is u-linear, and a raw principal-value projection
                // would break phase coherence with the neighbouring
                // boundary edges' unwrapped u.
                brepkit_math::vec::Point2::new(
                    (edge.end_uv.x() - edge.start_uv.x()).mul_add(t, edge.start_uv.x()),
                    edge.start_uv.y(),
                )
            } else {
                project_point_on_surface(split_3d, surface, &[], None)
            };
            let pcurve =
                compute_pcurve_on_surface(&edge.curve_3d, prev_3d, split_3d, surface, &[], frame);
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
        let pcurve =
            compute_pcurve_on_surface(&edge.curve_3d, prev_3d, edge.end_3d, surface, &[], frame);
        // The stored `end_uv` of a closed rim follows the CURVE's direction
        // (`sample_edge_to_uv` ignores orientation), so a reverse-traversed ring
        // would close a period on the wrong side of its own start.
        let tail_end_uv = match closed_ring_dir {
            Some(dir) => brepkit_math::vec::Point2::new(
                std::f64::consts::TAU.mul_add(dir, edge.start_uv.x()),
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
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        crate::perf::bump_face_split_probe();
        let to_pt = sp - edge.start_3d;
        let t = to_pt.dot(edge_dir) / edge_len_sq;
        if t <= tol || t >= 1.0 - tol {
            continue;
        }
        let closest = edge.start_3d + edge_dir * t;
        let dist = (sp - closest).length();
        if dist < tol {
            splits.push((t, sp));
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
    // Scoped to cylinder/cone rims, matching the UV-span branch in
    // `split_boundary_edges_at_3d_points` that consumes these `t` values. The
    // same origin-vs-start mismatch is latent for a closed circle on any other
    // surface (a bore rim on a plane), but only the periodic-rim case is
    // exercised and verified here, so the wider change stays out of scope.
    let closed_anchor = (matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_))
        && (edge.start_3d - edge.end_3d).length() < tol)
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
            let u = std::f64::consts::TAU.mul_add(turns, u_raw);
            (u - edge.start_uv.x()) / u_span
        } else if matches!(surface, FaceSurface::Plane { .. }) && !edge.forward {
            // A reversed-traversal plane arc covers ccw(end→start); the
            // ccw(start→end) span below is its COMPLEMENT, so on-arc points
            // would normalize outside [0,1] and drop (a bay ring absorbed
            // into the outer boundary is walked CW — its section crossings
            // never split it). Normalize in the physical span and map back
            // to traversal order.
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
pub(super) fn find_splits_on_nurbs_section(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let n_samples = 64usize;
    let eval_at =
        |t: f64| -> Point3 { evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t) };
    let samples: Vec<Point3> = (0..=n_samples)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            eval_at(i as f64 / n_samples as f64)
        })
        .collect();
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
        if best_d > tol * 100.0 {
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
        if (eval_at(t) - sp).length() > tol {
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
    let (t0, t1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
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
        let t_norm = normalize_angle_in_span(angle, t0, span);
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

    #[test]
    fn nurbs_section_splits_ordered_along_forward_edge() {
        let edge = parabola_section_edge(false);
        let eval = |t: f64| evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
        let (sp_a, sp_b) = (eval(0.3), eval(0.7));
        let splits = find_splits_on_nurbs_section(&edge, &[sp_b, sp_a], 1e-3);
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

        // Pin the divergence that motivated the section-specific finder: the
        // domain-based `find_splits_on_ellipse` (still correct for boundary
        // edges) DOES phantom-split the reverse twin on the other-window
        // point. If it ever becomes twin-safe, delete this assertion and
        // consider re-unifying the finders.
        let rev = ellipse_section_edge(&ellipse, 0.2, 1.0, false);
        let old = find_splits_on_ellipse(&ellipse, &rev, &[other_window_pt], 1e-7);
        assert_eq!(
            old.len(),
            1,
            "domain-based finder is expected to phantom-split the reverse twin"
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

    #[test]
    fn nurbs_section_splits_ordered_along_reversed_twin() {
        // The reverse twin stores the SAME curve but swapped endpoints, so
        // ascending natural-domain `t` runs end→start; the splits must come
        // back ordered from `start_3d` (nearest first) or the piece-building
        // loop emits overlapping pieces.
        let edge = parabola_section_edge(true);
        let eval = |t: f64| evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
        let (sp_a, sp_b) = (eval(0.3), eval(0.7));
        let splits = find_splits_on_nurbs_section(&edge, &[sp_a, sp_b], 1e-3);
        assert_eq!(splits.len(), 2);
        assert!((splits[0].1 - sp_b).length() < 1e-6);
        assert!((splits[1].1 - sp_a).length() < 1e-6);
        let d0 = (splits[0].1 - edge.start_3d).length();
        let d1 = (splits[1].1 - edge.start_3d).length();
        assert!(d0 < d1, "splits must walk start_3d → end_3d");
    }
}

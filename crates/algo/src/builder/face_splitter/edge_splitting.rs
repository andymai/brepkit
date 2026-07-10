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
            EdgeCurve::Circle(circle) => find_splits_on_circle(circle, &edge, split_pts_3d, tol),
            EdgeCurve::Ellipse(ellipse) => {
                find_splits_on_ellipse(ellipse, &edge, split_pts_3d, tol)
            }
            _ => find_splits_on_line(&edge, split_pts_3d, tol),
        };

        if splits.is_empty() {
            result.push(edge);
            continue;
        }

        let mut prev_uv = edge.start_uv;
        let mut prev_3d = edge.start_3d;
        for &(t, _) in &splits {
            let split_3d = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
            let split_uv = if let Some(f) = frame {
                f.project(split_3d)
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
        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve_3d.clone(),
            pcurve,
            start_uv: prev_uv,
            end_uv: edge.end_uv,
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
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
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
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
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

/// Find split parameters on a marched-NURBS SECTION edge by sampled
/// point-to-curve projection.
///
/// The chord-based `find_splits_on_line` misses a junction point that lies on
/// the CURVE but off its chord (a plane×cone conic bulges millimetres past the
/// chord), so a section chain meeting the conic mid-span never splits it and
/// the weave breaks (the dovetail tongue-relief cone cap). Parameters use the
/// same normalized-[0,1]-over-`domain_with_endpoints` convention as
/// `evaluate_edge_at_t`'s NURBS arm.
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

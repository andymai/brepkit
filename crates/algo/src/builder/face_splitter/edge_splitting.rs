//! Curve-specific edge splitting at 3D intersection points.

use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceSurface;

use super::super::pcurve_compute::{
    compute_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface, shorter_arc_delta,
};
use super::super::plane_frame::PlaneFrame;
use super::super::split_types::OrientedPCurveEdge;

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

/// Map an angular crossing on an arc edge to the edge's `[0, 1]` parameter using
/// the SAME shorter-arc convention as `evaluate_edge_at_t`, then validate the
/// result in 3D so a crossing is accepted only if it actually lies on this
/// edge's trimmed span.
///
/// Why this and not `normalize_angle_in_span(angle, t0, span)` with `span` from
/// `domain_with_endpoints`: `domain_with_endpoints` reports the CCW (positive,
/// `0..2π`) arc from start to end, but `evaluate_edge_at_t` traces the SHORTER
/// arc (signed delta in `(-π, π]`). When the CCW arc is the long one the two
/// disagree — a `t` computed under the CCW convention is then evaluated
/// downstream against the shorter arc and lands at the wrong 3D point. Worse,
/// when one `Circle3D` is shared by two complementary arc edges (e.g. the two
/// halves of a z-step rim), a genuine crossing projects onto both arcs: the arc
/// it does NOT belong to either gets a spurious split at a bogus point, or
/// returns a just-negative `t` for the real split and rejects it.
///
/// Computing `t` under the shorter-arc convention and round-tripping it through
/// `evaluate_edge_at_t` selects the arc the crossing truly belongs to: the 3D
/// round-trip only matches on the correct arc, so the right edge splits and the
/// wrong one does not. `a_start` is the curve angle of `start`, `a_cross` the
/// angle of the candidate crossing.
fn split_param_on_arc(
    curve: &EdgeCurve,
    start: Point3,
    end: Point3,
    a_start: f64,
    a_cross: f64,
    sp: Point3,
    tol: f64,
) -> Option<f64> {
    // Shorter-arc delta from start to end (the edge's full traced span) and
    // from start to the crossing — both under the same convention used by
    // `evaluate_edge_at_t`.
    let span = shorter_arc_delta_for_endpoints(curve, start, end);
    if span.abs() < 1e-14 {
        return None;
    }
    let delta = shorter_arc_delta(a_cross - a_start);
    let t = delta / span;
    if t <= tol || t >= 1.0 - tol {
        return None;
    }
    // Self-consistency: the point this `t` maps to (under the downstream
    // `evaluate_edge_at_t` convention) must coincide with the crossing. This is
    // the decisive arc selector — when one `Circle3D` is shared by two
    // complementary arc edges and the crossing projects onto both, the arc the
    // point does NOT belong to maps `t` to a different 3D location and is
    // rejected here.
    let back = evaluate_edge_at_t(curve, start, end, t);
    if (back - sp).length() > tol {
        return None;
    }
    // Reject a split that lands on (or numerically on top of) an endpoint: it
    // makes a zero-length sub-edge that downstream over-shares the boundary.
    // The split point is a section endpoint which is itself only vertex-snapped
    // (≈1e-6 noise here), so a hair-past-endpoint crossing yields `t` just
    // inside `(0, 1)` (e.g. `t ≈ 1.3e-7` against a 1e-7 tol). Test the actual
    // mapped 3D point against the endpoints with a snap-scale tolerance so the
    // degenerate split is dropped regardless of the crossing's own noise.
    let snap = (tol * 100.0).max(1e-6);
    if (back - start).length() <= snap || (back - end).length() <= snap {
        return None;
    }
    Some(t)
}

/// Shorter-arc angular span from `start` to `end` for a Circle/Ellipse edge,
/// matching `evaluate_edge_at_t`'s open-arc tracing.
fn shorter_arc_delta_for_endpoints(curve: &EdgeCurve, start: Point3, end: Point3) -> f64 {
    match curve {
        EdgeCurve::Circle(c) => shorter_arc_delta(c.project(end) - c.project(start)),
        EdgeCurve::Ellipse(e) => shorter_arc_delta(e.project(end) - e.project(start)),
        _ => 0.0,
    }
}

/// Find split parameters on a circle edge. Uses `Circle3D::project` for angular
/// projection, then maps into `[0, 1]` via `split_param_on_arc` (shorter-arc
/// convention + 3D self-consistency check).
pub(super) fn find_splits_on_circle(
    circle: &brepkit_math::curves::Circle3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    // Closed full circle (start == end): no shorter-arc span; the legacy
    // domain-based mapping is correct because span is the full 2π.
    if (edge.start_3d - edge.end_3d).length() < 1e-9 {
        return find_splits_on_closed_circle(circle, edge, split_pts_3d, tol);
    }
    let a_start = circle.project(edge.start_3d);
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        if let Some(t) = split_param_on_arc(
            &edge.curve_3d,
            edge.start_3d,
            edge.end_3d,
            a_start,
            angle,
            sp,
            tol,
        ) {
            splits.push((t, sp));
        }
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Split mapping for a closed full-circle edge (start == end). The edge spans
/// the full domain; map each crossing's angle to `[0, 1]` via the curve domain.
fn find_splits_on_closed_circle(
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
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = super::sampling::normalize_angle_in_span(angle, t0, span);
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Find split parameters on an ellipse edge. Mirrors `find_splits_on_circle`.
pub(super) fn find_splits_on_ellipse(
    ellipse: &brepkit_math::curves::Ellipse3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    if (edge.start_3d - edge.end_3d).length() < 1e-9 {
        return find_splits_on_closed_ellipse(ellipse, edge, split_pts_3d, tol);
    }
    let a_start = ellipse.project(edge.start_3d);
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        let angle = ellipse.project(sp);
        let closest = ellipse.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        if let Some(t) = split_param_on_arc(
            &edge.curve_3d,
            edge.start_3d,
            edge.end_3d,
            a_start,
            angle,
            sp,
            tol,
        ) {
            splits.push((t, sp));
        }
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Split mapping for a closed full-ellipse edge (start == end).
fn find_splits_on_closed_ellipse(
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
        let angle = ellipse.project(sp);
        let closest = ellipse.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = super::sampling::normalize_angle_in_span(angle, t0, span);
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
    use brepkit_math::curves::Circle3D;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::vec::{Point2, Vec2, Vec3};
    use std::f64::consts::PI;

    fn arc_edge(circle: &Circle3D, a0: f64, a1: f64) -> OrientedPCurveEdge {
        let start_3d = circle.evaluate(a0);
        let end_3d = circle.evaluate(a1);
        OrientedPCurveEdge {
            curve_3d: EdgeCurve::Circle(circle.clone()),
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

    /// Two complementary arcs of ONE circle: the upper semicircle (angle 0→π,
    /// through the top) and the lower semicircle (angle π→2π, through the
    /// bottom). A split point on the top must split ONLY the upper arc; a point
    /// on the bottom must split ONLY the lower arc. The old CCW-span mapping
    /// mis-assigned these because both arcs project the same circle.
    #[test]
    fn co_circular_arcs_split_only_the_owning_arc() {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let tol = 1e-9;

        // Upper arc: start at angle 0 = (1,0,0), end at angle π = (-1,0,0),
        // through the top (0,1,0).
        let upper = arc_edge(&circle, 0.0, PI);
        // Lower arc edge of the SAME circle: start angle π, end angle 2π,
        // through the bottom (0,-1,0).
        let lower = arc_edge(&circle, PI, 2.0 * PI);

        let top = circle.evaluate(PI / 2.0); // (0,1,0)
        let bottom = circle.evaluate(3.0 * PI / 2.0); // (0,-1,0)

        // Top point splits the upper arc near its midpoint, not the lower.
        let up_top = find_splits_on_circle(&circle, &upper, &[top], tol);
        let lo_top = find_splits_on_circle(&circle, &lower, &[top], tol);
        assert_eq!(up_top.len(), 1, "top point must split the upper arc");
        assert!((up_top[0].0 - 0.5).abs() < 1e-6, "split at arc midpoint");
        assert!(lo_top.is_empty(), "top point must NOT split the lower arc");

        // Bottom point splits the lower arc, not the upper.
        let up_bot = find_splits_on_circle(&circle, &upper, &[bottom], tol);
        let lo_bot = find_splits_on_circle(&circle, &lower, &[bottom], tol);
        assert!(
            up_bot.is_empty(),
            "bottom point must NOT split the upper arc"
        );
        assert_eq!(lo_bot.len(), 1, "bottom point must split the lower arc");
        assert!((lo_bot[0].0 - 0.5).abs() < 1e-6, "split at arc midpoint");
    }

    /// A crossing that coincides with an arc's endpoint (within vertex-snap
    /// noise) must not produce a degenerate zero-length split.
    #[test]
    fn endpoint_coincident_crossing_is_not_split() {
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let tol = 1e-7;
        let arc = arc_edge(&circle, 0.0, PI / 2.0);
        // A point a hair off the start vertex (≈1e-8 noise), on the circle.
        let near_start = Point3::new(circle.evaluate(0.0).x(), 1e-8, 0.0);
        let splits = find_splits_on_circle(&circle, &arc, &[near_start], tol);
        assert!(
            splits.is_empty(),
            "near-endpoint crossing must not create a degenerate split, got {splits:?}"
        );
    }
}

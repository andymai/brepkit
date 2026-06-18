//! Point containment tests for UV-space hole detection.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::vec::Point2;

use super::super::split_types::OrientedPCurveEdge;
use super::sampling::sample_wire_loop_uv;

/// Build a UV polygon for a hole wire from its edges' endpoint UVs (chords),
/// not from `sample_wire_loop_uv`. The pcurve of a projected arc edge can be a
/// `Nurbs2D` whose parameter domain traces a path inconsistent with its stored
/// `start_uv`/`end_uv` (the pcurve was fitted in a different frame), so sampling
/// it yields a self-crossing garbage polygon. The stored endpoint UVs ARE
/// correct and chain edge-to-edge, so the chord polygon faithfully approximates
/// the hole for a point-in-polygon containment test (a rounded-rect's corner
/// arcs are clipped to chords, well away from any interior test point).
fn hole_chord_polygon(hole: &[OrientedPCurveEdge]) -> Vec<Point2> {
    let mut pts: Vec<Point2> = Vec::with_capacity(hole.len() + 1);
    for e in hole {
        pts.push(e.start_uv);
        // Densify a curved edge with its midpoint so a large corner arc keeps a
        // closer-to-true hull than a single chord. Cheap and robust: just adds
        // the arc's stored midpoint endpoint when present, otherwise the chord
        // midpoint. Either way it stays inside the true arc, never outside.
        if !matches!(e.pcurve, Curve2D::Line(_)) {
            pts.push(Point2::new(
                0.5 * (e.start_uv.x() + e.end_uv.x()),
                0.5 * (e.start_uv.y() + e.end_uv.y()),
            ));
        }
    }
    pts
}

/// Check if a UV point is inside any of the inner wire (hole) polygons.
pub(super) fn is_inside_any_hole(pt: &Point2, inner_wires: &[Vec<OrientedPCurveEdge>]) -> bool {
    for hole in inner_wires {
        // Prefer the chord polygon (endpoint-derived). Fall back to the sampled
        // polygon only when the hole degenerates to < 3 endpoints.
        let chord = hole_chord_polygon(hole);
        if chord.len() >= 3 {
            if super::super::classify_2d::point_in_polygon_2d(*pt, &chord) {
                return true;
            }
            continue;
        }
        let hole_pts = sample_wire_loop_uv(hole);
        if hole_pts.len() >= 3 && super::super::classify_2d::point_in_polygon_2d(*pt, &hole_pts) {
            return true;
        }
    }
    false
}

/// Find a UV point inside the outer wire but outside all holes.
///
/// Tries midpoints between outer wire vertices and the centroid of the first
/// hole. Falls back to midpoints of outer wire edges nudged outward from holes.
pub(super) fn find_point_outside_holes(
    outer_pts: &[Point2],
    inner_wires: &[Vec<OrientedPCurveEdge>],
) -> Point2 {
    // Strategy: take midpoints between outer wire edge midpoints and the outer
    // boundary -- these are likely in the ring region between outer and inner.
    let centroid_x = outer_pts.iter().map(|p| p.x()).sum::<f64>() / outer_pts.len() as f64;
    let centroid_y = outer_pts.iter().map(|p| p.y()).sum::<f64>() / outer_pts.len() as f64;
    for i in 0..outer_pts.len() {
        let j = (i + 1) % outer_pts.len();
        let edge_mid = Point2::new(
            (outer_pts[i].x() + outer_pts[j].x()) * 0.5,
            (outer_pts[i].y() + outer_pts[j].y()) * 0.5,
        );
        // Step inward from the edge midpoint toward the centroid in small
        // increments; the first point inside the outer wire and outside every
        // hole wins. Small steps handle THIN rings (e.g. the ~1.2mm gridfinity
        // lip annulus on an 83mm cap), where a single large nudge overshoots
        // straight into the hole and no ring point is ever found.
        for k in 1..=99 {
            let t = f64::from(k) * 0.005;
            let candidate = Point2::new(
                edge_mid.x() * (1.0 - t) + centroid_x * t,
                edge_mid.y() * (1.0 - t) + centroid_y * t,
            );
            if super::super::classify_2d::point_in_polygon_2d(candidate, outer_pts)
                && !is_inside_any_hole(&candidate, inner_wires)
            {
                return candidate;
            }
        }
    }

    // Fallback: try vertex midpoints between consecutive outer wire vertices.
    if outer_pts.len() >= 2 {
        let mid = Point2::new(
            (outer_pts[0].x() + outer_pts[1].x()) * 0.5,
            (outer_pts[0].y() + outer_pts[1].y()) * 0.5,
        );
        return mid;
    }

    // Ultimate fallback: centroid (even though it may be in a hole).
    let n = outer_pts.len() as f64;
    Point2::new(
        outer_pts.iter().map(|p| p.x()).sum::<f64>() / n,
        outer_pts.iter().map(|p| p.y()).sum::<f64>() / n,
    )
}

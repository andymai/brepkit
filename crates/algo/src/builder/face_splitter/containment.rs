//! Point containment tests for UV-space hole detection.

use brepkit_math::vec::Point2;

use super::super::split_types::OrientedPCurveEdge;
use super::sampling::sample_wire_loop_uv;

/// Check if a UV point is inside any of the inner wire (hole) polygons.
pub(super) fn is_inside_any_hole(pt: &Point2, inner_wires: &[Vec<OrientedPCurveEdge>]) -> bool {
    for hole in inner_wires {
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
        // Nudge the edge midpoint slightly toward the centroid.
        let candidate = Point2::new(
            edge_mid.x() * 0.9 + centroid_x * 0.1,
            edge_mid.y() * 0.9 + centroid_y * 0.1,
        );
        if super::super::classify_2d::point_in_polygon_2d(candidate, outer_pts)
            && !is_inside_any_hole(&candidate, inner_wires)
        {
            return candidate;
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

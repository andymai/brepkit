//! 2D interior point sampling and polygon classification.
//!
//! Given a closed wire loop in (u,v) parameter space, find a point
//! guaranteed to be inside the loop for solid classification.

#![allow(dead_code)] // Used by later boolean_pipeline pipeline stages.

use brepkit_math::vec::{Point2, Vec2};

/// Sample a point guaranteed to be inside a closed 2D polygon.
///
/// Strategy: try the centroid first (works for convex polygons). If the
/// centroid is outside (non-convex loop), walk inward from the midpoint
/// of a boundary edge along the inward normal.
///
/// `loop_pts` must be the vertices of a closed polygon (≥3 points).
pub fn sample_interior_point(loop_pts: &[Point2]) -> Point2 {
    // Graceful fallback for degenerate inputs instead of panicking.
    if loop_pts.is_empty() {
        return Point2::new(0.0, 0.0);
    }
    if loop_pts.len() < 3 {
        let n = loop_pts.len() as f64;
        let cx = loop_pts.iter().map(|p| p.x()).sum::<f64>() / n;
        let cy = loop_pts.iter().map(|p| p.y()).sum::<f64>() / n;
        return Point2::new(cx, cy);
    }

    // 1. Compute centroid.
    let n = loop_pts.len() as f64;
    let cx = loop_pts.iter().map(|p| p.x()).sum::<f64>() / n;
    let cy = loop_pts.iter().map(|p| p.y()).sum::<f64>() / n;
    let centroid = Point2::new(cx, cy);

    // 2. If centroid is inside, use it.
    if point_in_polygon_2d(centroid, loop_pts) {
        return centroid;
    }

    // 3. Fallback: try midpoint of each edge, nudged inward.
    let area = signed_area_2d(loop_pts);
    for i in 0..loop_pts.len() {
        let j = (i + 1) % loop_pts.len();
        let mid = Point2::new(
            (loop_pts[i].x() + loop_pts[j].x()) * 0.5,
            (loop_pts[i].y() + loop_pts[j].y()) * 0.5,
        );
        let edge_dir = Vec2::new(
            loop_pts[j].x() - loop_pts[i].x(),
            loop_pts[j].y() - loop_pts[i].y(),
        );
        let len = edge_dir.length();
        if len < 1e-15 {
            continue;
        }
        // Inward normal depends on winding:
        // CCW (positive area) → rotate edge 90° CW → (dy, -dx)
        // CW (negative area) → rotate edge 90° CCW → (-dy, dx)
        let inward = if area > 0.0 {
            Vec2::new(-edge_dir.y(), edge_dir.x())
        } else {
            Vec2::new(edge_dir.y(), -edge_dir.x())
        };
        let nudge = 1e-4;
        let candidate = Point2::new(
            mid.x() + inward.x() / len * nudge,
            mid.y() + inward.y() / len * nudge,
        );
        if point_in_polygon_2d(candidate, loop_pts) {
            return candidate;
        }
    }
    // All edge midpoints failed — return centroid as last resort.
    centroid
}

/// Test whether a 2D point is inside a closed polygon using ray-casting
/// (even-odd rule).
pub fn point_in_polygon_2d(p: Point2, polygon: &[Point2]) -> bool {
    let n = polygon.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let pi = polygon[i];
        let pj = polygon[j];
        // Check if a horizontal ray from p to +∞ crosses edge (pi, pj).
        if (pi.y() > p.y()) != (pj.y() > p.y()) {
            let x_intersect = (pj.x() - pi.x()) * (p.y() - pi.y()) / (pj.y() - pi.y()) + pi.x();
            if p.x() < x_intersect {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

/// Compute the signed area of a 2D polygon (Shoelace formula).
///
/// Positive area = counterclockwise winding, negative = clockwise.
pub fn signed_area_2d(pts: &[Point2]) -> f64 {
    let n = pts.len();
    if n < 3 {
        return 0.0;
    }
    let mut area = 0.0;
    let mut j = n - 1;
    for i in 0..n {
        area += (pts[j].x() - pts[i].x()) * (pts[j].y() + pts[i].y());
        j = i;
    }
    area * 0.5
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    #[test]
    fn interior_of_rectangle_is_inside() {
        let pts = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let interior = sample_interior_point(&pts);
        assert!(interior.x() > 0.5 && interior.x() < 9.5);
        assert!(interior.y() > 0.5 && interior.y() < 9.5);
        assert!(point_in_polygon_2d(interior, &pts));
    }

    #[test]
    fn point_in_polygon_simple_square() {
        let sq = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        assert!(point_in_polygon_2d(Point2::new(5.0, 5.0), &sq));
        assert!(!point_in_polygon_2d(Point2::new(15.0, 5.0), &sq));
        assert!(!point_in_polygon_2d(Point2::new(-1.0, 5.0), &sq));
    }

    #[test]
    fn signed_area_ccw_is_positive() {
        // CCW square: should be positive area.
        let ccw = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let area = signed_area_2d(&ccw);
        assert!((area - 100.0).abs() < 1e-10, "area = {area}");
    }

    #[test]
    fn signed_area_cw_is_negative() {
        // CW square: should be negative area.
        let cw = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 10.0),
            Point2::new(10.0, 10.0),
            Point2::new(10.0, 0.0),
        ];
        let area = signed_area_2d(&cw);
        assert!((area + 100.0).abs() < 1e-10, "area = {area}");
    }

    #[test]
    fn interior_of_l_shaped_polygon() {
        // L-shape (non-convex): centroid may be outside.
        let l_shape = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 5.0),
            Point2::new(5.0, 5.0),
            Point2::new(5.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        let interior = sample_interior_point(&l_shape);
        assert!(
            point_in_polygon_2d(interior, &l_shape),
            "interior ({}, {}) should be inside L-shape",
            interior.x(),
            interior.y()
        );
    }
}

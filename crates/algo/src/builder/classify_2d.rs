//! 2D interior point sampling and polygon classification.
//!
//! Given a closed wire loop in (u,v) parameter space, find a point
//! guaranteed to be inside the loop for solid classification.

use brepkit_math::vec::{Point2, Vec2};

/// Sample a point guaranteed to be inside a closed 2D polygon.
///
/// Strategy: try the centroid first (works for convex polygons). If the
/// centroid is outside (non-convex loop), walk inward from the midpoint
/// of a boundary edge along the inward normal.
///
/// `loop_pts` must be the vertices of a closed polygon (>=3 points).
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

    let n = loop_pts.len() as f64;
    let cx = loop_pts.iter().map(|p| p.x()).sum::<f64>() / n;
    let cy = loop_pts.iter().map(|p| p.y()).sum::<f64>() / n;
    let centroid = Point2::new(cx, cy);

    // If centroid is inside, use it. A centroid that lands on the
    // boundary itself (e.g. the reflex corner of an L-shaped loop) passes
    // the even-odd test unpredictably and is not a safe interior sample.
    if point_in_polygon_2d(centroid, loop_pts)
        && distance_to_polygon_boundary(centroid, loop_pts) > boundary_eps(loop_pts)
    {
        return centroid;
    }

    // Edge-midpoint fallback for non-convex loops. For every boundary edge,
    // cast a ray inward from its midpoint and take the midpoint of the resulting
    // interior chord (the segment from the edge to the first opposite boundary
    // crossing). Keep the candidate with the LONGEST such chord — the deepest,
    // most robustly-interior point.
    //
    // Returning the *deepest* candidate rather than the *first* valid one is
    // what makes this rotation-invariant: the wire builder emits each loop with
    // a per-process nondeterministic starting edge (the rim boundary wire's
    // rotation tracks hash order), so a first-match scan picks a different edge
    // — and on a non-convex slice (a notched annular rim) a different pocket —
    // each run, flipping the sub-face's IN/OUT classification and producing an
    // intermittent mesh fallback. The longest-chord midpoint depends only on
    // the loop's geometry, not its starting index, and lands far from every
    // boundary so the downstream point-in-solid test is stable.
    let area = signed_area_2d(loop_pts);
    let n = loop_pts.len();
    let mut best: Option<(Point2, f64)> = None;
    for i in 0..n {
        let j = (i + 1) % n;
        let a = loop_pts[i];
        let b = loop_pts[j];
        let mid = Point2::new((a.x() + b.x()) * 0.5, (a.y() + b.y()) * 0.5);
        let edge_dir = Vec2::new(b.x() - a.x(), b.y() - a.y());
        let len = edge_dir.length();
        if len < 1e-15 {
            continue;
        }
        // Inward normal depends on winding (left normal of the edge for a
        // CCW loop): CCW (positive area) -> (-dy, dx); CW -> (dy, -dx).
        let inward = if area > 0.0 {
            Vec2::new(-edge_dir.y() / len, edge_dir.x() / len)
        } else {
            Vec2::new(edge_dir.y() / len, -edge_dir.x() / len)
        };
        // First crossing of the inward ray with any other boundary edge
        // (every edge except the current one `i`).
        let mut t_hit = f64::INFINITY;
        for k in 0..n {
            if k == i {
                continue;
            }
            let c = loop_pts[k];
            let d = loop_pts[(k + 1) % n];
            if let Some(t) = ray_segment_param(mid, inward, c, d)
                && t > 1e-9
                && t < t_hit
            {
                t_hit = t;
            }
        }
        if !t_hit.is_finite() {
            continue;
        }
        let cand = Point2::new(
            mid.x() + inward.x() * t_hit * 0.5,
            mid.y() + inward.y() * t_hit * 0.5,
        );
        // Keep the longest interior chord; break exact ties by the candidate's
        // lexicographic coordinates so the result is loop-rotation-invariant
        // (the chosen start edge is HashMap-order-dependent upstream).
        let better = best.is_none_or(|(bp, bt)| {
            t_hit > bt + 1e-12
                || ((t_hit - bt).abs() <= 1e-12 && (cand.x(), cand.y()) < (bp.x(), bp.y()))
        });
        if point_in_polygon_2d(cand, loop_pts) && better {
            best = Some((cand, t_hit));
        }
    }
    if let Some((pt, _)) = best {
        return pt;
    }
    // All edge midpoints failed -- return centroid as last resort.
    centroid
}

/// Parameter `t >= 0` at which the ray `origin + t*dir` first crosses segment
/// `[s0, s1]`, or `None` if it does not. `dir` is assumed unit-length; `t` is a
/// world-space distance along the ray.
fn ray_segment_param(origin: Point2, dir: Vec2, s0: Point2, s1: Point2) -> Option<f64> {
    let e = Vec2::new(s1.x() - s0.x(), s1.y() - s0.y());
    let denom = dir.x() * e.y() - dir.y() * e.x();
    if denom.abs() < 1e-15 {
        return None; // parallel
    }
    let diff = Vec2::new(s0.x() - origin.x(), s0.y() - origin.y());
    // t along the ray, u along the segment.
    let t = (diff.x() * e.y() - diff.y() * e.x()) / denom;
    let u = (diff.x() * dir.y() - diff.y() * dir.x()) / denom;
    if t >= 0.0 && (0.0..=1.0).contains(&u) {
        Some(t)
    } else {
        None
    }
}

/// Scale-aware boundary tolerance for a 2D loop: a small fraction of the
/// loop's bounding-box diagonal, floored to avoid collapsing to zero for
/// degenerate inputs.
pub fn boundary_eps(loop_pts: &[Point2]) -> f64 {
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for p in loop_pts {
        min_x = min_x.min(p.x());
        min_y = min_y.min(p.y());
        max_x = max_x.max(p.x());
        max_y = max_y.max(p.y());
    }
    let diag = ((max_x - min_x).powi(2) + (max_y - min_y).powi(2)).sqrt();
    (diag * 1e-6).max(1e-12)
}

/// Minimum distance from a 2D point to the polygon's boundary segments.
pub fn distance_to_polygon_boundary(p: Point2, loop_pts: &[Point2]) -> f64 {
    let mut best = f64::INFINITY;
    let n = loop_pts.len();
    for i in 0..n {
        let a = loop_pts[i];
        let b = loop_pts[(i + 1) % n];
        let ab = Vec2::new(b.x() - a.x(), b.y() - a.y());
        let ap = Vec2::new(p.x() - a.x(), p.y() - a.y());
        let len_sq = ab.x() * ab.x() + ab.y() * ab.y();
        let t = if len_sq < 1e-30 {
            0.0
        } else {
            ((ap.x() * ab.x() + ap.y() * ab.y()) / len_sq).clamp(0.0, 1.0)
        };
        let dx = p.x() - (a.x() + ab.x() * t);
        let dy = p.y() - (a.y() + ab.y() * t);
        best = best.min((dx * dx + dy * dy).sqrt());
    }
    best
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
        // Check if a horizontal ray from p to +inf crosses edge (pi, pj).
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

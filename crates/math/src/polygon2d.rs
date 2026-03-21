//! 2D polygon operations: clipping, filleting, chamfering, and segment detection.

use crate::vec::Point2;

/// Cross product of vectors (b-a) and (c-a).
#[must_use]
pub fn cross_2d(a: Point2, b: Point2, c: Point2) -> f64 {
    (b.x() - a.x()) * (c.y() - a.y()) - (b.y() - a.y()) * (c.x() - a.x())
}

fn line_intersect_2d(a1: Point2, a2: Point2, b1: Point2, b2: Point2) -> Option<Point2> {
    let dx_a = a2.x() - a1.x();
    let dy_a = a2.y() - a1.y();
    let dx_b = b2.x() - b1.x();
    let dy_b = b2.y() - b1.y();
    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < 1e-15 {
        return None;
    }
    let t = ((b1.x() - a1.x()) * dy_b - (b1.y() - a1.y()) * dx_b) / denom;
    Some(Point2::new(a1.x() + t * dx_a, a1.y() + t * dy_a))
}

fn point_to_line_dist_sq_2d(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < 1e-30 {
        let ex = p.x() - a.x();
        let ey = p.y() - a.y();
        return ex * ex + ey * ey;
    }
    let cross = (p.x() - a.x()) * dy - (p.y() - a.y()) * dx;
    (cross * cross) / len_sq
}

/// Sutherland-Hodgman polygon clipping algorithm.
#[must_use]
pub fn sutherland_hodgman_clip(subject: &[Point2], clip: &[Point2]) -> Vec<Point2> {
    let mut output: Vec<Point2> = subject.to_vec();

    for i in 0..clip.len() {
        if output.is_empty() {
            return output;
        }
        let edge_start = clip[i];
        let edge_end = clip[(i + 1) % clip.len()];
        let input = output;
        output = Vec::new();

        for j in 0..input.len() {
            let current = input[j];
            let previous = input[(j + input.len() - 1) % input.len()];

            let curr_inside = cross_2d(edge_start, edge_end, current) >= 0.0;
            let prev_inside = cross_2d(edge_start, edge_end, previous) >= 0.0;

            if curr_inside {
                if !prev_inside {
                    if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                        output.push(p);
                    }
                }
                output.push(current);
            } else if prev_inside {
                if let Some(p) = line_intersect_2d(previous, current, edge_start, edge_end) {
                    output.push(p);
                }
            }
        }
    }

    output
}

/// Find common (collinear, overlapping) edges between two polygons.
#[must_use]
pub fn find_common_segments(a: &[Point2], b: &[Point2], tolerance: f64) -> Vec<(Point2, Point2)> {
    let mut results = Vec::new();
    let tol_sq = tolerance * tolerance;

    for i in 0..a.len() {
        let a1 = a[i];
        let a2 = a[(i + 1) % a.len()];
        for j in 0..b.len() {
            let b1 = b[j];
            let b2 = b[(j + 1) % b.len()];

            // Check if edge A and edge B are collinear and overlapping.
            // Both endpoints of B must be close to line through A, or vice versa.
            let dist_b1 = point_to_line_dist_sq_2d(b1, a1, a2);
            let dist_b2 = point_to_line_dist_sq_2d(b2, a1, a2);

            if dist_b1 < tol_sq && dist_b2 < tol_sq {
                // Edges are collinear. Check for overlap by projecting onto A's direction.
                let dx = a2.x() - a1.x();
                let dy = a2.y() - a1.y();
                let len_sq = dx * dx + dy * dy;
                if len_sq < tol_sq {
                    continue;
                }
                let t1 = ((b1.x() - a1.x()) * dx + (b1.y() - a1.y()) * dy) / len_sq;
                let t2 = ((b2.x() - a1.x()) * dx + (b2.y() - a1.y()) * dy) / len_sq;
                let t_min = t1.min(t2).max(0.0);
                let t_max = t1.max(t2).min(1.0);
                if t_max - t_min > tolerance / len_sq.sqrt() {
                    results.push((
                        Point2::new(a1.x() + t_min * dx, a1.y() + t_min * dy),
                        Point2::new(a1.x() + t_max * dx, a1.y() + t_max * dy),
                    ));
                }
            }
        }
    }
    results
}

/// Round all corners of a 2D polygon with arc approximations.
#[must_use]
pub fn fillet_polygon_2d(polygon: &[Point2], radius: f64) -> Vec<Point2> {
    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let arc_segments = 8; // Number of segments per fillet arc
    let mut result = Vec::with_capacity(n * (arc_segments + 1));

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let max_r = (d_prev.min(d_next) / 2.0).min(radius);

        if max_r < 1e-10 {
            result.push(curr);
            continue;
        }

        // Direction vectors from corner to adjacent vertices
        let dir_prev_x = (prev.x() - curr.x()) / d_prev;
        let dir_prev_y = (prev.y() - curr.y()) / d_prev;
        let dir_next_x = (next.x() - curr.x()) / d_next;
        let dir_next_y = (next.y() - curr.y()) / d_next;

        // Tangent points on edges
        let t1 = Point2::new(curr.x() + dir_prev_x * max_r, curr.y() + dir_prev_y * max_r);
        let t2 = Point2::new(curr.x() + dir_next_x * max_r, curr.y() + dir_next_y * max_r);

        // Generate arc points from t1 to t2
        for k in 0..=arc_segments {
            let t = k as f64 / arc_segments as f64;
            let x = t2.x().mul_add(t, t1.x() * (1.0 - t));
            let y = t2.y().mul_add(t, t1.y() * (1.0 - t));

            // Push point toward the arc center for a circular approximation
            let mid_x = f64::midpoint(t1.x(), t2.x());
            let mid_y = f64::midpoint(t1.y(), t2.y());
            let to_corner_x = curr.x() - mid_x;
            let to_corner_y = curr.y() - mid_y;
            let corner_dist = (to_corner_x * to_corner_x + to_corner_y * to_corner_y).sqrt();

            if corner_dist > 1e-10 {
                // Compute the bulge: how much to push along the corner bisector
                let chord_half =
                    ((t2.x() - t1.x()).powi(2) + (t2.y() - t1.y()).powi(2)).sqrt() / 2.0;
                let sagitta = if max_r > chord_half {
                    max_r - (max_r * max_r - chord_half * chord_half).sqrt()
                } else {
                    0.0
                };

                // Blend factor: maximum at midpoint (t=0.5), zero at endpoints
                let blend = 4.0 * t * (1.0 - t); // parabolic blend
                let push = sagitta * blend;

                let nx = to_corner_x / corner_dist;
                let ny = to_corner_y / corner_dist;
                result.push(Point2::new(x + nx * push, y + ny * push));
            } else {
                result.push(Point2::new(x, y));
            }
        }
    }

    result
}

/// Cut all corners of a 2D polygon with flat bevels.
#[must_use]
pub fn chamfer_polygon_2d(polygon: &[Point2], distance: f64) -> Vec<Point2> {
    let n = polygon.len();
    if n < 3 {
        return polygon.to_vec();
    }

    let mut result = Vec::with_capacity(n * 2);

    for i in 0..n {
        let prev = polygon[(i + n - 1) % n];
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];

        let d_prev = ((prev.x() - curr.x()).powi(2) + (prev.y() - curr.y()).powi(2)).sqrt();
        let d_next = ((next.x() - curr.x()).powi(2) + (next.y() - curr.y()).powi(2)).sqrt();

        let d = (d_prev.min(d_next) / 2.0).min(distance);

        if d < 1e-10 {
            result.push(curr);
            continue;
        }

        // Two chamfer points: one on previous edge, one on next edge
        result.push(Point2::new(
            curr.x() + (prev.x() - curr.x()) / d_prev * d,
            curr.y() + (prev.y() - curr.y()) / d_prev * d,
        ));
        result.push(Point2::new(
            curr.x() + (next.x() - curr.x()) / d_next * d,
            curr.y() + (next.y() - curr.y()) / d_next * d,
        ));
    }

    result
}

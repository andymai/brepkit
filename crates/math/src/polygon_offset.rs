//! 2D polygon offset via parallel edge translation and miter joins.
//!
//! Given a closed polygon in 2D, offsets each edge by a signed distance
//! (positive = outward, negative = inward) and computes new vertex
//! positions at the intersection of adjacent offset lines.

use crate::MathError;
use crate::vec::Point2;

/// Default miter limit: if a miter join extends further than this factor
/// times the offset distance, it is clamped to a bevel join.
const DEFAULT_MITER_LIMIT: f64 = 2.0;

/// Offset a closed 2D polygon by a signed distance.
///
/// Positive `distance` offsets outward (assuming CCW winding),
/// negative offsets inward. Applies a miter limit of 2× the offset
/// distance to prevent spikes at sharp corners, and removes
/// self-intersections from the result.
///
/// # Algorithm
///
/// For each edge, compute a parallel offset line by translating along
/// the outward normal. Intersect adjacent offset lines to find new
/// vertex positions (miter join). Degenerate intersections (parallel
/// edges) fall back to simple translation.
///
/// # Errors
///
/// Returns an error if fewer than 3 vertices are provided.
pub fn offset_polygon_2d(
    vertices: &[Point2],
    distance: f64,
    tolerance: f64,
) -> Result<Vec<Point2>, MathError> {
    let n = vertices.len();
    if n < 3 {
        return Err(MathError::EmptyInput);
    }

    let miter_limit_dist = distance.abs() * DEFAULT_MITER_LIMIT;

    // Compute offset edge lines: each edge shifts by distance along its outward normal.
    // For a CCW polygon, the outward normal of edge (A→B) is (dy, -dx) normalized.
    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let prev = (i + n - 1) % n;
        let next = (i + 1) % n;

        // Previous edge: prev → i
        let (nx0, ny0) = edge_outward_normal(vertices[prev], vertices[i]);
        // Current edge: i → next
        let (nx1, ny1) = edge_outward_normal(vertices[i], vertices[next]);

        // Offset lines:
        // Line 0 passes through (vertices[prev] + d*n0) with direction (vertices[i] - vertices[prev])
        // Line 1 passes through (vertices[i] + d*n1) with direction (vertices[next] - vertices[i])
        let p0 = Point2::new(
            vertices[i].x() + distance * nx0,
            vertices[i].y() + distance * ny0,
        );
        let p1 = Point2::new(
            vertices[i].x() + distance * nx1,
            vertices[i].y() + distance * ny1,
        );

        let d0x = vertices[i].x() - vertices[prev].x();
        let d0y = vertices[i].y() - vertices[prev].y();
        let d1x = vertices[next].x() - vertices[i].x();
        let d1y = vertices[next].y() - vertices[i].y();

        // Intersect two lines: p0 + t*d0 = p1 + s*d1
        // Cross product for 2D line intersection
        let cross = d0x * d1y - d0y * d1x;

        if cross.abs() < tolerance {
            // Parallel edges — just translate the vertex
            let avg_nx = (nx0 + nx1) * 0.5;
            let avg_ny = (ny0 + ny1) * 0.5;
            result.push(Point2::new(
                vertices[i].x() + distance * avg_nx,
                vertices[i].y() + distance * avg_ny,
            ));
        } else {
            // Solve for t: (p1 - p0) × d1 / (d0 × d1)
            let dx = p1.x() - p0.x();
            let dy = p1.y() - p0.y();
            let t = (dx * d1y - dy * d1x) / cross;
            let miter_pt = Point2::new(p0.x() + t * d0x, p0.y() + t * d0y);

            // Miter limit: if the miter point is too far from the original
            // vertex, clamp to prevent spikes at near-parallel edges.
            let miter_dist = ((miter_pt.x() - vertices[i].x()).powi(2)
                + (miter_pt.y() - vertices[i].y()).powi(2))
            .sqrt();

            if miter_dist > miter_limit_dist && miter_limit_dist > tolerance {
                // Bevel: use the average of the two offset points.
                result.push(Point2::new(
                    (p0.x() + p1.x()) * 0.5,
                    (p0.y() + p1.y()) * 0.5,
                ));
            } else {
                result.push(miter_pt);
            }
        }
    }

    // Remove self-intersections by detecting and clipping crossing edges.
    remove_self_intersections(&mut result, tolerance);

    Ok(result)
}

/// Remove self-intersections from a polygon by detecting crossing edges
/// and keeping only the largest non-self-intersecting loop.
fn remove_self_intersections(polygon: &mut Vec<Point2>, tolerance: f64) {
    if polygon.len() < 4 {
        return;
    }

    let n = polygon.len();
    let tol_sq = tolerance * tolerance;

    // Check all non-adjacent edge pairs for intersections.
    // When found, remove the smaller loop by cutting out the vertices between
    // the crossing edges.
    let mut i = 0;
    while i < polygon.len().saturating_sub(2) {
        let n_cur = polygon.len();
        let a1 = polygon[i];
        let a2 = polygon[(i + 1) % n_cur];
        let mut found = false;

        // Only check edges that are at least 2 apart (non-adjacent).
        let mut j = i + 2;
        while j < n_cur {
            // Skip the last edge wrapping back to edge 0 if i == 0.
            if i == 0 && j == n_cur - 1 {
                j += 1;
                continue;
            }

            let b1 = polygon[j];
            let b2 = polygon[(j + 1) % n_cur];

            if let Some(_pt) = segment_intersection_2d(a1, a2, b1, b2, tol_sq) {
                // Self-intersection found between edges i and j.
                // Remove the shorter loop (vertices between i+1 and j).
                let loop_len = j - i - 1;
                let other_len = n_cur - loop_len;

                if loop_len <= other_len {
                    // Remove vertices i+1..j
                    polygon.drain((i + 1)..j);
                } else {
                    // Remove vertices j+1..end and 0..i
                    let kept: Vec<Point2> = polygon[i..=j].to_vec();
                    *polygon = kept;
                }
                found = true;
                break;
            }
            j += 1;
        }

        if !found {
            i += 1;
        }
        // If found, restart from same i since polygon was modified.

        // Safety: prevent infinite loop if polygon degenerates.
        if polygon.len() < 3 || polygon.len() > n * 2 {
            break;
        }
    }
}

/// Compute the intersection point of two 2D line segments, if any.
fn segment_intersection_2d(
    a1: Point2,
    a2: Point2,
    b1: Point2,
    b2: Point2,
    _tol_sq: f64,
) -> Option<Point2> {
    let dx_a = a2.x() - a1.x();
    let dy_a = a2.y() - a1.y();
    let dx_b = b2.x() - b1.x();
    let dy_b = b2.y() - b1.y();

    let denom = dx_a * dy_b - dy_a * dx_b;
    if denom.abs() < 1e-15 {
        return None; // Parallel
    }

    let dx_ab = b1.x() - a1.x();
    let dy_ab = b1.y() - a1.y();
    let t = (dx_ab * dy_b - dy_ab * dx_b) / denom;
    let s = (dx_ab * dy_a - dy_ab * dx_a) / denom;

    // Both parameters must be in (0, 1) (exclusive to avoid endpoint touches).
    let eps = 1e-10;
    if t > eps && t < 1.0 - eps && s > eps && s < 1.0 - eps {
        Some(Point2::new(a1.x() + t * dx_a, a1.y() + t * dy_a))
    } else {
        None
    }
}

/// Compute the outward normal of a 2D edge (assuming CCW winding).
///
/// For edge A→B, the outward normal is the left-perpendicular of (B-A),
/// normalized to unit length. Returns `(0, 0)` for degenerate edges.
fn edge_outward_normal(a: Point2, b: Point2) -> (f64, f64) {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-15 {
        return (0.0, 0.0);
    }
    // Left perpendicular = (dy, -dx) / len
    (dy / len, -dx / len)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn offset_square_outward() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(1.0, 0.0),
            Point2::new(1.0, 1.0),
            Point2::new(0.0, 1.0),
        ];
        let result = offset_polygon_2d(&square, 0.5, 1e-10).unwrap();
        assert_eq!(result.len(), 4);
        // Outward offset of unit square by 0.5 → vertices at (-0.5, -0.5), (1.5, -0.5), etc.
        assert!((result[0].x() - (-0.5)).abs() < 1e-10);
        assert!((result[0].y() - (-0.5)).abs() < 1e-10);
        assert!((result[1].x() - 1.5).abs() < 1e-10);
        assert!((result[1].y() - (-0.5)).abs() < 1e-10);
        assert!((result[2].x() - 1.5).abs() < 1e-10);
        assert!((result[2].y() - 1.5).abs() < 1e-10);
        assert!((result[3].x() - (-0.5)).abs() < 1e-10);
        assert!((result[3].y() - 1.5).abs() < 1e-10);
    }

    #[test]
    fn offset_square_inward() {
        let square = vec![
            Point2::new(0.0, 0.0),
            Point2::new(2.0, 0.0),
            Point2::new(2.0, 2.0),
            Point2::new(0.0, 2.0),
        ];
        let result = offset_polygon_2d(&square, -0.5, 1e-10).unwrap();
        assert_eq!(result.len(), 4);
        assert!((result[0].x() - 0.5).abs() < 1e-10);
        assert!((result[0].y() - 0.5).abs() < 1e-10);
        assert!((result[1].x() - 1.5).abs() < 1e-10);
        assert!((result[1].y() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn offset_triangle() {
        let tri = vec![
            Point2::new(0.0, 0.0),
            Point2::new(4.0, 0.0),
            Point2::new(2.0, 3.0),
        ];
        let result = offset_polygon_2d(&tri, 0.1, 1e-10).unwrap();
        assert_eq!(result.len(), 3);
        // All vertices should be further from the centroid
        let cx = (tri[0].x() + tri[1].x() + tri[2].x()) / 3.0;
        let cy = (tri[0].y() + tri[1].y() + tri[2].y()) / 3.0;
        for (orig, off) in tri.iter().zip(result.iter()) {
            let d_orig = ((orig.x() - cx).powi(2) + (orig.y() - cy).powi(2)).sqrt();
            let d_off = ((off.x() - cx).powi(2) + (off.y() - cy).powi(2)).sqrt();
            assert!(
                d_off > d_orig,
                "offset vertex should be farther from centroid"
            );
        }
    }

    #[test]
    fn too_few_vertices() {
        let pts = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        assert!(offset_polygon_2d(&pts, 0.1, 1e-10).is_err());
    }
}

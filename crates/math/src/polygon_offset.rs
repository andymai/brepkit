//! 2D polygon offset via parallel edge translation and miter joins.
//!
//! Given a closed polygon in 2D, offsets each edge by a signed distance
//! (positive = outward, negative = inward) and computes new vertex
//! positions at the intersection of adjacent offset lines.

use crate::MathError;
use crate::vec::Point2;

/// Offset a closed 2D polygon by a signed distance.
///
/// Positive `distance` offsets outward (assuming CCW winding),
/// negative offsets inward.
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
            result.push(Point2::new(p0.x() + t * d0x, p0.y() + t * d0y));
        }
    }

    Ok(result)
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

//! Geometry-aware snap tolerance for vertex/edge deduplication during fillet assembly.
//!
//! Replaces hardcoded tolerance multipliers with values that scale relative to
//! local geometry dimensions, preventing unintended vertex merges on small features.

use brepkit_math::vec::Point3;

/// Compute geometry-aware snap tolerance for vertex deduplication.
///
/// Scales the tolerance relative to the smallest local geometry dimension,
/// ensuring vertices are merged only when they're genuinely coincident
/// (not just "close" relative to coarse global tolerance).
///
/// # Arguments
/// - `edge_length`: length of the shortest edge at the vertex
/// - `radius`: fillet radius
///
/// # Returns
/// A snap tolerance in the range `[1e-10, 1e-4]`.
#[must_use]
pub fn snap_tolerance(edge_length: f64, radius: f64) -> f64 {
    let from_edge = edge_length * 0.001;
    let from_radius = radius * 0.01;
    from_edge.min(from_radius).clamp(1e-10, 1e-4)
}

/// Check if two vertices should be merged at the given tolerance.
#[must_use]
pub fn should_merge_vertices(p1: Point3, p2: Point3, tol: f64) -> bool {
    (p1 - p2).length() < tol
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn test_snap_tolerance_scales_with_edge_length() {
        // edge_length * 0.001 is the limiting factor (smaller than radius * 0.01)
        let t1 = snap_tolerance(0.1, 1.0);
        let t2 = snap_tolerance(0.01, 1.0);
        // Shorter edge -> smaller tolerance
        assert!(t2 < t1, "tolerance should decrease with shorter edges");
    }

    #[test]
    fn test_snap_tolerance_scales_with_radius() {
        // radius * 0.01 is the limiting factor (smaller than edge_length * 0.001)
        let t1 = snap_tolerance(1.0, 0.01);
        let t2 = snap_tolerance(1.0, 0.001);
        // Smaller radius -> smaller tolerance
        assert!(t2 < t1, "tolerance should decrease with smaller radius");
    }

    #[test]
    fn test_snap_tolerance_clamped_to_range() {
        // Very large inputs -> clamped to 1e-4
        let t_large = snap_tolerance(1000.0, 1000.0);
        assert!(t_large <= 1e-4, "tolerance must not exceed 1e-4");

        // Very small inputs -> clamped to 1e-10
        let t_small = snap_tolerance(1e-12, 1e-12);
        assert!(t_small >= 1e-10, "tolerance must not go below 1e-10");
    }

    #[test]
    fn test_should_merge_close_points() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1e-8, 0.0, 0.0);
        assert!(should_merge_vertices(p1, p2, 1e-6));
    }

    #[test]
    fn test_should_not_merge_distant_points() {
        let p1 = Point3::new(0.0, 0.0, 0.0);
        let p2 = Point3::new(1.0, 0.0, 0.0);
        assert!(!should_merge_vertices(p1, p2, 1e-4));
    }
}

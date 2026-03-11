//! Winding direction utilities for consistent CCW vertex ordering.
//!
//! Operations that create solid faces from profile wires (extrude, sweep,
//! loft, pipe, revolve) rely on CCW-wound profiles to produce outward-facing
//! normals via cross-product formulas. When callers provide CW-wound profiles
//! (common with brepjs polygon approximations), the resulting normals point
//! inward, creating inside-out solids.
//!
//! This module provides shared detection and correction utilities used by
//! all profile-based operations.

use brepkit_math::vec::{Point3, Vec3};

/// Compute the Newell normal of a polygon.
///
/// The Newell normal is proportional to twice the signed area of the polygon
/// and points in the direction consistent with right-hand-rule traversal.
/// For CCW-wound vertices (viewed from the normal direction), the normal
/// points toward the viewer.
pub fn newell_normal(verts: &[Point3]) -> Vec3 {
    let m = verts.len();
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    for i in 0..m {
        let curr = verts[i];
        let next = verts[(i + 1) % m];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    Vec3::new(nx, ny, nz)
}

/// Compute the centroid of a polygon's vertices.
#[allow(clippy::cast_precision_loss)]
pub fn polygon_centroid(verts: &[Point3]) -> Point3 {
    let inv = 1.0 / verts.len() as f64;
    let (sx, sy, sz) = verts.iter().fold((0.0, 0.0, 0.0), |(x, y, z), p| {
        (x + p.x(), y + p.y(), z + p.z())
    });
    Point3::new(sx * inv, sy * inv, sz * inv)
}

/// Check if vertices wind CW relative to a reference direction.
///
/// Projects the Newell normal onto the reference direction. A negative
/// dot product means the polygon's traversal is CW when viewed from
/// the reference direction — i.e., the normals point opposite.
///
/// Returns `true` if the vertices are CW and need reversal for CCW convention.
pub fn is_cw_winding(positions: &[Point3], reference_dir: &Vec3) -> bool {
    if positions.len() < 3 {
        return false;
    }
    let newell = newell_normal(positions);
    let dot = newell.dot(*reference_dir);
    dot < -1e-30
}

/// Reverse positions in-place if they wind CW relative to the reference direction.
///
/// Returns `true` if the positions were reversed.
pub fn ensure_ccw_positions(positions: &mut [Point3], reference_dir: Vec3) -> bool {
    if is_cw_winding(positions, &reference_dir) {
        positions.reverse();
        true
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    #[test]
    fn newell_normal_ccw_square() {
        // CCW square on XY plane viewed from +Z
        let verts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let n = newell_normal(&verts);
        assert!(n.z() > 0.0, "CCW square should have +Z Newell normal");
    }

    #[test]
    fn newell_normal_cw_square() {
        // CW square on XY plane viewed from +Z
        let verts = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let n = newell_normal(&verts);
        assert!(n.z() < 0.0, "CW square should have -Z Newell normal");
    }

    #[test]
    fn is_cw_detects_cw_winding() {
        let cw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        assert!(is_cw_winding(&cw, &up));
    }

    #[test]
    fn is_cw_rejects_ccw_winding() {
        let ccw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        assert!(!is_cw_winding(&ccw, &up));
    }

    #[test]
    fn ensure_ccw_reverses_cw() {
        let mut cw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        let reversed = ensure_ccw_positions(&mut cw, up);
        assert!(reversed);
        // After reversal, should be CCW
        assert!(!is_cw_winding(&cw, &up));
    }

    #[test]
    fn ensure_ccw_preserves_ccw() {
        let mut ccw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let up = Vec3::new(0.0, 0.0, 1.0);
        let reversed = ensure_ccw_positions(&mut ccw, up);
        assert!(!reversed);
    }
}

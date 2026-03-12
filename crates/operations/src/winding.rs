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

/// Dot-product threshold for winding classification.
/// Values below this magnitude are treated as degenerate (coplanar/perpendicular).
const WINDING_DOT_TOL: f64 = 1e-30;

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
    dot < -WINDING_DOT_TOL
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

/// Determine if a polygon (inner wire) is CW when viewed from the
/// extrusion direction.
///
/// Uses the signed area projected onto the extrusion axis: negative = CW,
/// positive = CCW. This generalizes to non-axis-aligned extrusions.
/// When the projected area is near-zero (polygon nearly perpendicular to
/// the extrusion axis), defaults to CW (the standard B-Rep hole convention).
pub fn inner_wire_is_cw(positions: &[Point3], offset: &Vec3) -> bool {
    if positions.len() < 3 {
        return true; // degenerate — default to CW
    }
    let axis = offset.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let p0 = positions[0];
    let mut signed_area_2 = 0.0;
    for i in 1..positions.len() - 1 {
        let a = positions[i] - p0;
        let b = positions[i + 1] - p0;
        signed_area_2 += a.cross(b).dot(axis);
    }
    // Use a tolerance threshold to avoid floating-point noise flipping
    // the classification for polygons nearly perpendicular to the axis.
    // Default to CW (standard B-Rep hole convention) when ambiguous.
    signed_area_2 < -f64::EPSILON
}

/// Reverse all profiles if they wind CW relative to the stacking direction.
///
/// Stacking direction is inferred from first→last profile centroids.
/// Returns `true` if profiles were reversed.
pub fn ensure_ccw_profiles(profile_verts: &mut [Vec<Point3>]) -> bool {
    let c0 = polygon_centroid(&profile_verts[0]);
    let c1 = polygon_centroid(&profile_verts[profile_verts.len() - 1]);
    let stack_dir = c1 - c0;
    let newell = newell_normal(&profile_verts[0]);
    let dot = newell.dot(stack_dir);

    // Degenerate case: profiles share the same centroid (coplanar loft) or
    // the Newell normal is zero (degenerate polygon). Cannot determine
    // winding — leave unchanged.
    if dot.abs() < WINDING_DOT_TOL {
        return false;
    }

    if dot < 0.0 {
        for verts in profile_verts.iter_mut() {
            verts.reverse();
        }
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

    // ── inner_wire_is_cw tests ──

    #[test]
    fn inner_wire_cw_detected() {
        // CW square on XY plane, extrusion along +Z
        let cw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        assert!(inner_wire_is_cw(&cw, &Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn inner_wire_ccw_detected() {
        // CCW square on XY plane, extrusion along +Z
        let ccw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        assert!(!inner_wire_is_cw(&ccw, &Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn inner_wire_degenerate_defaults_cw() {
        // Fewer than 3 vertices — should default to CW
        let degen = vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)];
        assert!(inner_wire_is_cw(&degen, &Vec3::new(0.0, 0.0, 1.0)));
    }

    #[test]
    fn inner_wire_perpendicular_defaults_cw() {
        // Square on XZ plane, extrusion along +Y (perpendicular to polygon)
        // Signed area projected onto Y is ~zero → default CW
        let sq = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        assert!(inner_wire_is_cw(&sq, &Vec3::new(0.0, 1.0, 0.0)));
    }

    // ── ensure_ccw_profiles tests ──

    #[test]
    fn ensure_ccw_profiles_reverses_cw() {
        // Two CW squares stacked along +Z
        let cw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        ];
        let cw2 = cw.iter().map(|p| *p + Vec3::new(0.0, 0.0, 2.0)).collect();
        let mut profiles = vec![cw, cw2];
        let reversed = ensure_ccw_profiles(&mut profiles);
        assert!(reversed, "CW profiles should be reversed");
        // After reversal, Newell normal of first profile should agree with stacking dir
        let n = newell_normal(&profiles[0]);
        let stack = polygon_centroid(&profiles[1]) - polygon_centroid(&profiles[0]);
        assert!(n.dot(stack) > 0.0);
    }

    #[test]
    fn ensure_ccw_profiles_preserves_ccw() {
        let ccw = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let ccw2 = ccw.iter().map(|p| *p + Vec3::new(0.0, 0.0, 2.0)).collect();
        let mut profiles = vec![ccw, ccw2];
        let reversed = ensure_ccw_profiles(&mut profiles);
        assert!(!reversed, "CCW profiles should not be reversed");
    }

    #[test]
    fn ensure_ccw_profiles_coplanar_unchanged() {
        // Same centroid — degenerate, should not reverse
        let sq = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        ];
        let mut profiles = vec![sq.clone(), sq];
        let reversed = ensure_ccw_profiles(&mut profiles);
        assert!(!reversed);
    }
}

//! Surface type conversion utilities.
//!
//! Provides conversions between analytic surface types and their NURBS
//! representations. Currently implements plane-to-NURBS; cylinder, cone,
//! sphere, and torus conversions are stubbed for future implementation.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};

use crate::HealError;

/// Convert a plane to a degree 1x1 NURBS surface.
///
/// The plane is defined by its normal and signed distance from the origin.
/// The resulting surface has 4 corner control points spanning the given
/// `u_range` and `v_range` in the plane's local coordinate system.
///
/// # Parameters
///
/// - `normal` -- plane normal (must be unit-length)
/// - `d` -- signed distance from origin along normal
/// - `u_range` -- parameter range in the u direction
/// - `v_range` -- parameter range in the v direction
///
/// # Errors
///
/// Returns [`HealError`] if the NURBS construction fails.
pub fn plane_to_nurbs(
    normal: Vec3,
    d: f64,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> Result<NurbsSurface, HealError> {
    // Build a local frame on the plane.
    let origin = Point3::new(0.0, 0.0, 0.0) + normal * d;
    let (u_axis, v_axis) = plane_frame_axes(normal);

    let (u0, u1) = u_range;
    let (v0, v1) = v_range;

    // 4 corner control points: 2 rows (u) x 2 cols (v).
    let cp = vec![
        vec![
            origin + u_axis * u0 + v_axis * v0,
            origin + u_axis * u0 + v_axis * v1,
        ],
        vec![
            origin + u_axis * u1 + v_axis * v0,
            origin + u_axis * u1 + v_axis * v1,
        ],
    ];

    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

    let knots_u = vec![u0, u0, u1, u1];
    let knots_v = vec![v0, v0, v1, v1];

    let surface = NurbsSurface::new(1, 1, knots_u, knots_v, cp, weights)?;
    Ok(surface)
}

// TODO: Implement `cylinder_to_nurbs` -- rational degree 2 x 1 NURBS
// (periodic in u, linear in v). Requires similar quarter-arc decomposition
// as `circle_to_nurbs` in the u direction.

// TODO: Implement `cone_to_nurbs` -- rational degree 2 x 1 NURBS
// with v-dependent radius scaling.

// TODO: Implement `sphere_to_nurbs` -- rational degree 2 x 2 NURBS
// (periodic in u, semicircular arcs in v).

// TODO: Implement `torus_to_nurbs` -- rational degree 2 x 2 NURBS
// (periodic in both u and v).

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build orthonormal UV axes from a plane normal.
fn plane_frame_axes(normal: Vec3) -> (Vec3, Vec3) {
    let seed = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_raw = normal.cross(seed);
    let u_axis = u_raw.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v_axis = normal.cross(u_axis);
    (u_axis, v_axis)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use brepkit_math::traits::ParametricSurface;

    use super::*;

    #[test]
    fn plane_to_nurbs_evaluates_on_plane() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 5.0;
        let surface = plane_to_nurbs(normal, d, (-10.0, 10.0), (-10.0, 10.0)).unwrap();

        // All evaluated points should have z = 5.0.
        for &u in &[-10.0, -5.0, 0.0, 5.0, 10.0] {
            for &v in &[-10.0, -5.0, 0.0, 5.0, 10.0] {
                let p = ParametricSurface::evaluate(&surface, u, v);
                assert!(
                    (p.z() - d).abs() < 1e-10,
                    "at ({u}, {v}): z={}, expected {d}",
                    p.z()
                );
            }
        }
    }

    #[test]
    fn plane_to_nurbs_corners_match() {
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let d = 0.0;
        let surface = plane_to_nurbs(normal, d, (0.0, 1.0), (0.0, 1.0)).unwrap();

        // Corner at (0, 0) should be the origin.
        let p00 = ParametricSurface::evaluate(&surface, 0.0, 0.0);
        assert!(p00.z().abs() < 1e-10);

        // Corner at (1, 1) should be 1 unit away in both u and v.
        let p11 = ParametricSurface::evaluate(&surface, 1.0, 1.0);
        assert!(p11.z().abs() < 1e-10);
        // Distance from p00 to p11 should be sqrt(2).
        let dist = (p11 - p00).length();
        assert!(
            (dist - std::f64::consts::SQRT_2).abs() < 1e-10,
            "diagonal distance = {dist}, expected sqrt(2)"
        );
    }
}

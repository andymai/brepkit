//! Convert analytic surfaces to NURBS representation.
//!
//! These are thin wrappers around the `to_nurbs()` methods on each
//! analytic surface type from `brepkit-math`, propagating errors as
//! [`GeomError`].
//!
//! # Exact vs sampled
//!
//! - [`cylinder_to_nurbs`] is **geometrically exact** (the math layer's
//!   `CylindricalSurface::to_nurbs` uses the standard 9-CP rational
//!   representation).
//! - [`cone_to_nurbs`], [`sphere_to_nurbs`], and [`torus_to_nurbs`] are
//!   **sampled approximations** (33 × 9 degree-1 grid via the math
//!   layer's `analytic_to_nurbs_sampled`, intended for intersection
//!   seed-finding — chord-height error ~0.5–7% of radius depending on
//!   parameter direction).
//!
//! For exact rational NURBS conversions of cone/sphere/torus suitable
//! for downstream NURBS pipelines (e.g. transform under non-uniform
//! scale), use the corresponding functions in
//! `brepkit_heal::construct::convert_surface`.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};

use crate::GeomError;

/// Convert a [`CylindricalSurface`] patch to a NURBS surface.
///
/// Geometrically covers a full revolution around the cylinder axis in
/// the u-direction. `v_range` sets the axial extent. The NURBS parameter
/// domain is determined by the underlying `to_nurbs()` implementation.
///
/// # Errors
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn cylinder_to_nurbs(
    cyl: &CylindricalSurface,
    v_range: (f64, f64),
) -> Result<NurbsSurface, GeomError> {
    Ok(cyl.to_nurbs(v_range.0, v_range.1)?)
}

/// Convert a [`SphericalSurface`] to a NURBS surface (sampled
/// approximation, ~5% chord deviation in v).
///
/// For an exact rational form, use
/// `brepkit_heal::construct::convert_surface::sphere_to_nurbs`.
///
/// # Errors
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn sphere_to_nurbs(sphere: &SphericalSurface) -> Result<NurbsSurface, GeomError> {
    Ok(sphere.to_nurbs()?)
}

/// Convert a [`ConicalSurface`] patch to a NURBS surface (sampled
/// approximation; the math layer's underlying `to_nurbs` uses
/// `analytic_to_nurbs_sampled`).
///
/// The result spans the full angular circle in the u-direction.
/// `v_range` sets the extent along the cone's generator direction from
/// the apex.
///
/// For an exact rational form, use
/// `brepkit_heal::construct::convert_surface::cone_to_nurbs`.
///
/// # Errors
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn cone_to_nurbs(
    cone: &ConicalSurface,
    v_range: (f64, f64),
) -> Result<NurbsSurface, GeomError> {
    Ok(cone.to_nurbs(v_range.0, v_range.1)?)
}

/// Convert a [`ToroidalSurface`] to a NURBS surface (sampled
/// approximation, ~7% chord deviation of minor radius).
///
/// For an exact rational form, use
/// `brepkit_heal::construct::convert_surface::torus_to_nurbs`.
///
/// # Errors
///
/// Returns [`GeomError::Math`] if NURBS construction fails.
pub fn torus_to_nurbs(torus: &ToroidalSurface) -> Result<NurbsSurface, GeomError> {
    Ok(torus.to_nurbs()?)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[test]
    fn cylinder_to_nurbs_evaluates() {
        let cyl = CylindricalSurface::new(origin(), z_axis(), 2.0).unwrap();
        let nurbs = cylinder_to_nurbs(&cyl, (0.0, 5.0)).unwrap();
        // The domain should be defined.
        let (u0, u1) = nurbs.domain_u();
        let (v0, v1) = nurbs.domain_v();
        assert!(u1 > u0);
        assert!(v1 > v0);
        // Evaluate and check reasonable position.
        let pt = nurbs.evaluate((u0 + u1) * 0.5, (v0 + v1) * 0.5);
        let dist_from_axis = (Vec3::new(pt.x(), pt.y(), 0.0)).length();
        assert!((dist_from_axis - 2.0).abs() < 0.1);
    }

    #[test]
    fn sphere_to_nurbs_evaluates() {
        let sphere = SphericalSurface::new(origin(), 3.0).unwrap();
        let nurbs = sphere_to_nurbs(&sphere).unwrap();
        let (u0, u1) = nurbs.domain_u();
        let (v0, v1) = nurbs.domain_v();
        assert!(u1 > u0);
        assert!(v1 > v0);
        let pt = nurbs.evaluate((u0 + u1) * 0.5, (v0 + v1) * 0.5);
        let r = Vec3::new(pt.x(), pt.y(), pt.z()).length();
        assert!((r - 3.0).abs() < 0.2);
    }

    #[test]
    fn cone_to_nurbs_evaluates() {
        let half_angle = std::f64::consts::FRAC_PI_4;
        let cone = ConicalSurface::new(origin(), z_axis(), half_angle).unwrap();
        let nurbs = cone_to_nurbs(&cone, (0.0, 5.0)).unwrap();
        let (u0, u1) = nurbs.domain_u();
        let (v0, v1) = nurbs.domain_v();
        assert!(u1 > u0);
        assert!(v1 > v0);
        // Just check it evaluates without panic.
        let _pt = nurbs.evaluate((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    }

    #[test]
    fn torus_to_nurbs_evaluates() {
        let torus = ToroidalSurface::new(origin(), 4.0, 1.0).unwrap();
        let nurbs = torus_to_nurbs(&torus).unwrap();
        let (u0, u1) = nurbs.domain_u();
        let (v0, v1) = nurbs.domain_v();
        assert!(u1 > u0);
        assert!(v1 > v0);
        let _pt = nurbs.evaluate((u0 + u1) * 0.5, (v0 + v1) * 0.5);
    }
}

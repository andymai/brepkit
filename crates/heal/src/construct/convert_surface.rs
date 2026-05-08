//! Surface type conversion utilities.
//!
//! Provides conversions between analytic surface types and their NURBS
//! representations.
//!
//! * `plane_to_nurbs`, `cylinder_to_nurbs`, `cone_to_nurbs` produce
//!   **geometrically exact** rational NURBS (the rational forms exactly
//!   reproduce the analytic surface within floating-point tolerance).
//! * `sphere_to_nurbs`, `torus_to_nurbs` currently delegate to
//!   `brepkit-math`'s sampled approximation (33 × 9 degree-1×1 grid).
//!   Chord-height error: sphere ~0.5% in u, ~2% in v (8 spans across
//!   180° of latitude); torus ~0.5% in u, ~7–10% of *minor* radius
//!   in v (8 spans across the full 360° tube cross-section, so the
//!   per-span angular width is twice the sphere's). The exact
//!   rational forms (rotational sweep of a rational arc) are
//!   tracked as a future improvement.

use std::f64::consts::FRAC_1_SQRT_2;

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
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

/// Convert a cylindrical surface to a rational NURBS surface.
///
/// The result is degree 2 in u (full revolution as four quarter-arcs)
/// and degree 1 in v (axial direction over `[v_min, v_max]`).
///
/// # Errors
///
/// Returns [`HealError`] if NURBS construction fails.
pub fn cylinder_to_nurbs(
    cyl: &CylindricalSurface,
    v_range: (f64, f64),
) -> Result<NurbsSurface, HealError> {
    Ok(cyl.to_nurbs(v_range.0, v_range.1)?)
}

/// Convert a conical surface patch to a **geometrically exact**
/// rational NURBS surface.
///
/// Uses the same 9-CP × degree 2 rational representation in u as the
/// cylinder (four exact 90° arcs), with the radial scaling varying
/// linearly with v along the cone generator. The result is degree
/// `(2, 1)` and exactly reproduces the cone within floating-point
/// tolerance — finer than the sampled approximation in
/// `brepkit_math::ConicalSurface::to_nurbs`.
///
/// `v_range = (v_min, v_max)` is measured from the apex along the
/// cone-generator direction (NOT axial). Both endpoints must be
/// strictly positive to avoid the apex degeneracy; passing
/// `v_min ≤ 0` returns [`HealError`].
///
/// # Errors
///
/// Returns [`HealError`] if `v_min ≤ 0`, `v_max ≤ v_min`, or NURBS
/// construction fails.
pub fn cone_to_nurbs(
    cone: &ConicalSurface,
    v_range: (f64, f64),
) -> Result<NurbsSurface, HealError> {
    let (v_min, v_max) = v_range;
    // Apex degeneracy: v_min must be strictly positive (the rational
    // form requires a non-zero radius row).
    if v_min <= 0.0 {
        return Err(brepkit_math::MathError::ParameterOutOfRange {
            value: v_min,
            min: f64::EPSILON,
            max: f64::INFINITY,
        }
        .into());
    }
    // Empty/inverted range: v_max must lie strictly above v_min.
    if v_max <= v_min {
        return Err(brepkit_math::MathError::ParameterOutOfRange {
            value: v_max,
            min: v_min,
            max: f64::INFINITY,
        }
        .into());
    }

    let apex = cone.apex();
    let axis = cone.axis();
    let x_axis = cone.x_axis();
    let y_axis = cone.y_axis();
    let (sin_a, cos_a) = cone.half_angle().sin_cos();

    let circle_weights = [
        1.0,
        FRAC_1_SQRT_2,
        1.0,
        FRAC_1_SQRT_2,
        1.0,
        FRAC_1_SQRT_2,
        1.0,
        FRAC_1_SQRT_2,
        1.0,
    ];
    let dirs: [(f64, f64); 9] = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
    ];

    let r_lo = v_min * cos_a;
    let r_hi = v_max * cos_a;
    let z_lo = v_min * sin_a;
    let z_hi = v_max * sin_a;

    let mut cps = Vec::with_capacity(9);
    let mut ws = Vec::with_capacity(9);
    for (i, &(dx, dy)) in dirs.iter().enumerate() {
        let dir = x_axis * dx + y_axis * dy;
        let p_lo = apex + dir * r_lo + axis * z_lo;
        let p_hi = apex + dir * r_hi + axis * z_hi;
        cps.push(vec![p_lo, p_hi]);
        ws.push(vec![circle_weights[i], circle_weights[i]]);
    }

    let knots_u = vec![
        0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
    ];
    // Carry the user's `v_range` into the v-knots so `domain_v()`
    // matches the input. Lets callers evaluate at physical-v
    // coordinates (e.g. v=2.5 within (1.0, 4.0)) and join cleanly
    // with adjacent faces sharing boundary parameters.
    let knots_v = vec![v_min, v_min, v_max, v_max];
    Ok(NurbsSurface::new(2, 1, knots_u, knots_v, cps, ws)?)
}

/// Convert a spherical surface to a NURBS surface.
///
/// **Approximate.** Currently delegates to `brepkit-math`'s sampled
/// representation (33 × 9 grid of points, degree 1 × 1 NURBS). The
/// chord-height error is ~0.5% of radius along u (33 samples on
/// 360° revolution) and ~2% along v (9 samples on 180° latitude).
/// The exact rational form (degree 2 × 2, rotational sweep of a
/// rational semi-circle arc) is a tracked improvement.
///
/// # Errors
///
/// Returns [`HealError`] if NURBS construction fails.
pub fn sphere_to_nurbs(sphere: &SphericalSurface) -> Result<NurbsSurface, HealError> {
    Ok(sphere.to_nurbs()?)
}

/// Convert a toroidal surface to a NURBS surface.
///
/// **Approximate.** Currently delegates to `brepkit-math`'s sampled
/// representation (33 × 9 grid of points, degree 1 × 1 NURBS — chord
/// deviation ~7% of minor radius mid-span). The exact rational form
/// (degree 2 × 2, rotational sweep of a rational tube arc) is a
/// tracked improvement.
///
/// # Errors
///
/// Returns [`HealError`] if NURBS construction fails.
pub fn torus_to_nurbs(torus: &ToroidalSurface) -> Result<NurbsSurface, HealError> {
    Ok(torus.to_nurbs()?)
}

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
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::float_cmp
)]
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
    fn cylinder_to_nurbs_evaluates_on_cylinder() {
        let center = Point3::new(0.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let radius = 2.5_f64;
        let cyl = CylindricalSurface::new(center, axis, radius).unwrap();
        let surface = cylinder_to_nurbs(&cyl, (0.0, 5.0)).unwrap();

        // Sample across the surface; every point must satisfy
        // x² + y² == r² and 0 <= z <= 5 (within fp tol).
        let u_dom = surface.domain_u();
        let v_dom = surface.domain_v();
        let n = 10;
        for i in 0..=n {
            for j in 0..=n {
                let u = u_dom.0 + (u_dom.1 - u_dom.0) * f64::from(i) / f64::from(n);
                let v = v_dom.0 + (v_dom.1 - v_dom.0) * f64::from(j) / f64::from(n);
                let p = ParametricSurface::evaluate(&surface, u, v);
                let r = (p.x().powi(2) + p.y().powi(2)).sqrt();
                assert!(
                    (r - radius).abs() < 1e-9,
                    "({u}, {v}): r={r}, expected {radius}"
                );
                assert!(
                    p.z() >= -1e-9 && p.z() <= 5.0 + 1e-9,
                    "z out of range: {}",
                    p.z()
                );
            }
        }
    }

    #[test]
    fn cone_to_nurbs_rejects_apex_v_min() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::PI / 6.0,
        )
        .unwrap();
        let err = cone_to_nurbs(&cone, (0.0, 4.0)).unwrap_err();
        // The diagnostic should report v_min as `value` and
        // `max=f64::INFINITY` (NOT v_max), since the failure mode is
        // "apex degeneracy", not "range too small".
        match err {
            crate::HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value,
                max,
                ..
            }) => {
                assert!(value <= 0.0);
                assert!(max.is_infinite());
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn cone_to_nurbs_rejects_inverted_range() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::PI / 6.0,
        )
        .unwrap();
        let err = cone_to_nurbs(&cone, (3.0, 1.0)).unwrap_err();
        // For inverted range (v_max < v_min), the diagnostic must
        // describe v_max coherently — `value: v_max, min: v_min`,
        // not v_min as value.
        match err {
            crate::HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value,
                min,
                ..
            }) => {
                assert!(
                    (value - 1.0).abs() < 1e-12,
                    "value should be v_max=1.0, got {value}"
                );
                assert!(
                    (min - 3.0).abs() < 1e-12,
                    "min should be v_min=3.0, got {min}"
                );
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn cone_to_nurbs_carries_v_range_into_domain() {
        // Regression for greptile P1: v-knots used to be hard-coded to
        // [0,0,1,1] regardless of v_range, so domain_v() returned
        // (0.0, 1.0) for any input — surface couldn't be evaluated at
        // physical v-coordinates and didn't join cleanly with adjacent
        // patches sharing boundary parameters.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::PI / 6.0,
        )
        .unwrap();
        let surface = cone_to_nurbs(&cone, (1.0, 4.0)).unwrap();
        let (v_lo, v_hi) = surface.domain_v();
        assert!(
            (v_lo - 1.0).abs() < 1e-12,
            "v_lo should match v_min, got {v_lo}"
        );
        assert!(
            (v_hi - 4.0).abs() < 1e-12,
            "v_hi should match v_max, got {v_hi}"
        );
        // Also: evaluating at the geometric midpoint v=2.5 (inside
        // the user's range) should not panic.
        let _ = ParametricSurface::evaluate(&surface, 0.0, 2.5);
    }

    #[test]
    fn cone_to_nurbs_evaluates_on_cone() {
        let apex = Point3::new(0.0, 0.0, 0.0);
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let half_angle = std::f64::consts::PI / 6.0;
        let cone = ConicalSurface::new(apex, axis, half_angle).unwrap();
        let surface = cone_to_nurbs(&cone, (1.0, 4.0)).unwrap();

        // brepkit's `half_angle` is measured from the radial plane (NOT
        // axis), so the cone surface satisfies `r = z · cot(α)` (with
        // apex at origin, axis +z): radial component is v·cos(α), axial
        // is v·sin(α), yielding r/z = cos(α)/sin(α) = cot(α).
        let cot_a = half_angle.cos() / half_angle.sin();
        let u_dom = surface.domain_u();
        let v_dom = surface.domain_v();
        let n = 10;
        for i in 0..=n {
            for j in 0..=n {
                let u = u_dom.0 + (u_dom.1 - u_dom.0) * f64::from(i) / f64::from(n);
                let v = v_dom.0 + (v_dom.1 - v_dom.0) * f64::from(j) / f64::from(n);
                let p = ParametricSurface::evaluate(&surface, u, v);
                let r = (p.x().powi(2) + p.y().powi(2)).sqrt();
                let z = p.z();
                assert!(z > 0.0, "cone NURBS sample below apex: z={z}");
                assert!(
                    (r / z - cot_a).abs() < 1e-9,
                    "({u}, {v}): r/z = {} != cot(α) = {cot_a}",
                    r / z
                );
            }
        }
    }

    #[test]
    fn sphere_to_nurbs_approximates_sphere() {
        // Sampled NURBS (33 × 9 degree-1) — tolerance reflects chord
        // deviation, not floating-point drift. Document max observed
        // residual so a tightening is detectable when an exact rational
        // form is implemented.
        let center = Point3::new(0.0, 0.0, 0.0);
        let radius = 1.5_f64;
        let sphere = SphericalSurface::new(center, radius).unwrap();
        let surface = sphere_to_nurbs(&sphere).unwrap();

        let u_dom = surface.domain_u();
        let v_dom = surface.domain_v();
        let n = 12;
        let mut max_err = 0.0_f64;
        for i in 0..=n {
            for j in 0..=n {
                let u = u_dom.0 + (u_dom.1 - u_dom.0) * f64::from(i) / f64::from(n);
                let v = v_dom.0 + (v_dom.1 - v_dom.0) * f64::from(j) / f64::from(n);
                let p = ParametricSurface::evaluate(&surface, u, v);
                let r = (p - center).length();
                max_err = max_err.max((r - radius).abs());
            }
        }
        // Sampled-NURBS approximation: 9 latitude samples across 180°
        // = 8 spans of 22.5°, max chord deviation per span
        // = R(1 − cos(11.25°)) ≈ 1.9% of radius. Bound at 3% to
        // detect regressions while allowing for fp jitter and the
        // sample grid's worst-case non-equator latitudes.
        assert!(
            max_err < 0.03 * radius,
            "max sphere residual {max_err} exceeds 3% of radius"
        );
    }

    #[test]
    fn torus_to_nurbs_approximates_torus() {
        // Sampled NURBS — tolerance reflects chord deviation.
        let center = Point3::new(0.0, 0.0, 0.0);
        let major = 3.0_f64;
        let minor = 0.5_f64;
        let torus = ToroidalSurface::new(center, major, minor).unwrap();
        let surface = torus_to_nurbs(&torus).unwrap();

        let u_dom = surface.domain_u();
        let v_dom = surface.domain_v();
        let n = 12;
        let mut max_err = 0.0_f64;
        for i in 0..=n {
            for j in 0..=n {
                let u = u_dom.0 + (u_dom.1 - u_dom.0) * f64::from(i) / f64::from(n);
                let v = v_dom.0 + (v_dom.1 - v_dom.0) * f64::from(j) / f64::from(n);
                let p = ParametricSurface::evaluate(&surface, u, v);
                let rxy = (p.x().powi(2) + p.y().powi(2)).sqrt();
                let resid = ((rxy - major).powi(2) + p.z().powi(2)).sqrt();
                max_err = max_err.max((resid - minor).abs());
            }
        }
        // Chord deviation is ~9–10% of minor radius at v-mid-span
        // (degree-1 piecewise-linear interpolation across 8 spans of
        // the cross-section circle).
        assert!(
            max_err < 0.10 * minor,
            "max torus tube residual {max_err} exceeds 10% of minor radius"
        );
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

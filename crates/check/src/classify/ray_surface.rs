//! Ray-surface intersection for all analytic surface types and NURBS.
//!
//! Each function takes a ray (origin + direction) and a surface, returning
//! the positive-t intersection parameters along the ray.

// Functions used by boundary.rs for ray-face crossing counts.

use smallvec::SmallVec;

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};

use crate::CheckError;

// ---------------------------------------------------------------------------
// Tolerance constants
// ---------------------------------------------------------------------------

/// Denominator below this is treated as zero (parallel ray).
const NEAR_ZERO: f64 = 1e-15;

/// Minimum positive t to accept (reject self-intersections at origin).
const RAY_T_MIN: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Ray-plane
// ---------------------------------------------------------------------------

/// Intersect a ray with a plane defined by `normal . X = d`.
///
/// Returns `Some(t)` if the ray hits the plane at `origin + t * direction`
/// with `t > RAY_T_MIN`, or `None` if the ray is parallel or behind.
pub fn ray_plane(origin: Point3, direction: Vec3, normal: Vec3, d: f64) -> Option<f64> {
    let denom = normal.dot(direction);
    if denom.abs() < NEAR_ZERO {
        return None;
    }
    let origin_vec = Vec3::new(origin.x(), origin.y(), origin.z());
    let t = (d - normal.dot(origin_vec)) / denom;
    if t <= RAY_T_MIN { None } else { Some(t) }
}

// ---------------------------------------------------------------------------
// Ray-cylinder
// ---------------------------------------------------------------------------

/// Intersect a ray with an infinite cylindrical surface.
///
/// Returns up to 2 positive-t values.
pub fn ray_cylinder(
    origin: Point3,
    direction: Vec3,
    cyl: &CylindricalSurface,
) -> SmallVec<[f64; 4]> {
    let ov = origin - cyl.origin();
    let axis = cyl.axis();
    let ov_perp = ov - axis * ov.dot(axis);
    let d_perp = direction - axis * direction.dot(axis);

    let a = d_perp.dot(d_perp);
    let b = 2.0 * ov_perp.dot(d_perp);
    let c = ov_perp.dot(ov_perp) - cyl.radius() * cyl.radius();

    solve_quadratic(a, b, c)
}

// ---------------------------------------------------------------------------
// Ray-cone
// ---------------------------------------------------------------------------

/// Intersect a ray with an infinite conical surface.
///
/// Returns up to 2 positive-t values (both nappes of the double cone).
#[allow(clippy::similar_names)]
pub fn ray_cone(origin: Point3, direction: Vec3, cone: &ConicalSurface) -> SmallVec<[f64; 4]> {
    let ov = origin - cone.apex();
    let axis = cone.axis();

    let cos_a = cone.half_angle().cos();
    let cos2 = cos_a * cos_a;
    let sin2 = 1.0 - cos2;

    let d_dot_a = direction.dot(axis);
    let ov_dot_a = ov.dot(axis);

    let a = cos2 * d_dot_a * d_dot_a - sin2 * (direction.dot(direction) - d_dot_a * d_dot_a);
    let half_b = cos2 * d_dot_a * ov_dot_a - sin2 * (direction.dot(ov) - d_dot_a * ov_dot_a);
    let c = cos2 * ov_dot_a * ov_dot_a - sin2 * (ov.dot(ov) - ov_dot_a * ov_dot_a);

    solve_quadratic(a, 2.0 * half_b, c)
}

// ---------------------------------------------------------------------------
// Ray-sphere
// ---------------------------------------------------------------------------

/// Intersect a ray with a sphere.
///
/// Returns up to 2 positive-t values.
pub fn ray_sphere(origin: Point3, direction: Vec3, sph: &SphericalSurface) -> SmallVec<[f64; 4]> {
    let ov = origin - sph.center();
    let a = direction.dot(direction);
    let b = 2.0 * ov.dot(direction);
    let c = ov.dot(ov) - sph.radius() * sph.radius();

    solve_quadratic(a, b, c)
}

// ---------------------------------------------------------------------------
// Ray-torus
// ---------------------------------------------------------------------------

/// Intersect a ray with a torus.
///
/// Returns up to 4 positive-t values (quartic equation). Delegates to the
/// residual-verified quartic root finder in `brepkit_math` — a local Ferrari
/// solver previously both missed real roots and emitted off-surface spurious
/// ones for oblique rays at moderate radii, flipping crossing parity.
pub fn ray_torus(origin: Point3, direction: Vec3, tor: &ToroidalSurface) -> SmallVec<[f64; 4]> {
    brepkit_math::analytic_intersection::intersect_line_torus(tor, origin, direction)
        .into_iter()
        .filter(|&t| t > RAY_T_MIN)
        .collect()
}

// ---------------------------------------------------------------------------
// Ray-NURBS
// ---------------------------------------------------------------------------

/// Intersect a ray with a NURBS surface.
///
/// Returns a vector of `(t, u, v)` where `t` is the ray parameter and
/// `(u, v)` are the surface parameters at each hit.
///
/// # Errors
/// Propagates math errors from the NURBS intersection routine.
pub fn ray_nurbs(
    origin: Point3,
    direction: Vec3,
    surface: &NurbsSurface,
    n_samples: usize,
) -> Result<Vec<(f64, f64, f64)>, CheckError> {
    use brepkit_math::nurbs::intersection::intersect_line_nurbs;

    let hits = intersect_line_nurbs(surface, origin, direction, n_samples)?;
    let dir_dot_dir = direction.dot(direction);
    let mut results = Vec::new();
    for hit in &hits {
        let diff = hit.point - origin;
        let t = diff.dot(direction) / dir_dot_dir;
        if t > RAY_T_MIN {
            results.push((t, hit.param1.0, hit.param1.1));
        }
    }
    Ok(results)
}

// ===========================================================================
// Polynomial solvers
// ===========================================================================

/// Solve `a*t^2 + b*t + c = 0` returning positive roots (t > `RAY_T_MIN`).
pub fn solve_quadratic(a: f64, b: f64, c: f64) -> SmallVec<[f64; 4]> {
    let all = solve_quadratic_all(a, b, c);
    all.into_iter().filter(|&t| t > RAY_T_MIN).collect()
}

/// Solve `a*t^2 + b*t + c = 0` returning ALL real roots (no positive filter).
fn solve_quadratic_all(a: f64, b: f64, c: f64) -> SmallVec<[f64; 4]> {
    let mut roots = SmallVec::new();

    if a.abs() < NEAR_ZERO {
        if b.abs() < NEAR_ZERO {
            return roots;
        }
        roots.push(-c / b);
        return roots;
    }

    let disc = b * b - 4.0 * a * c;
    if disc < -NEAR_ZERO {
        return roots;
    }

    if disc.abs() < NEAR_ZERO {
        roots.push(-b / (2.0 * a));
        return roots;
    }

    let sqrt_disc = disc.max(0.0).sqrt();
    let q = if b < 0.0 {
        -0.5 * (b - sqrt_disc)
    } else {
        -0.5 * (b + sqrt_disc)
    };

    roots.push(q / a);
    if q.abs() > NEAR_ZERO {
        roots.push(c / q);
    }

    if roots.len() == 2 && roots[0] > roots[1] {
        roots.swap(0, 1);
    }

    roots
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn ray_sphere_through_center() {
        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let origin = Point3::new(0.0, 0.0, -5.0);
        let dir = Vec3::new(0.0, 0.0, 1.0);
        let hits = ray_sphere(origin, dir, &sph);
        assert_eq!(hits.len(), 2, "expected 2 hits, got {}", hits.len());
        assert!((hits[0] - 4.0).abs() < TOL, "t0={}, expected 4.0", hits[0]);
        assert!((hits[1] - 6.0).abs() < TOL, "t1={}, expected 6.0", hits[1]);
    }

    #[test]
    fn ray_plane_parallel() {
        let origin = Point3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(1.0, 0.0, 0.0); // parallel to XY plane
        let normal = Vec3::new(0.0, 0.0, 1.0);
        assert!(ray_plane(origin, dir, normal, 0.0).is_none());
    }

    #[test]
    fn ray_plane_forward() {
        let origin = Point3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let t = ray_plane(origin, dir, normal, 0.0).expect("should hit plane");
        assert!((t - 5.0).abs() < TOL, "t={t}, expected 5.0");
    }

    #[test]
    fn ray_cylinder_perpendicular() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let origin = Point3::new(5.0, 0.0, 0.0);
        let dir = Vec3::new(-1.0, 0.0, 0.0);
        let hits = ray_cylinder(origin, dir, &cyl);
        assert_eq!(hits.len(), 2, "expected 2 hits, got {}", hits.len());
        assert!((hits[0] - 4.0).abs() < TOL, "t0={}, expected 4.0", hits[0]);
        assert!((hits[1] - 6.0).abs() < TOL, "t1={}, expected 6.0", hits[1]);
    }

    #[test]
    fn ray_torus_through_center() {
        // Torus R=3, r=1 centered at origin, lying in XY plane.
        let tor = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let origin = Point3::new(-5.0, 0.0, 0.0);
        let dir = Vec3::new(1.0, 0.0, 0.0);
        let hits = ray_torus(origin, dir, &tor);
        // Ray along X through torus center: hits at x=-4, -2, 2, 4
        assert_eq!(
            hits.len(),
            4,
            "expected 4 hits, got {}: {:?}",
            hits.len(),
            hits
        );
        // t values: origin.x=-5, so t = x - (-5) = x + 5
        assert!(
            (hits[0] - 1.0).abs() < TOL,
            "t0={}, expected 1.0 (x=-4)",
            hits[0]
        );
        assert!(
            (hits[1] - 3.0).abs() < TOL,
            "t1={}, expected 3.0 (x=-2)",
            hits[1]
        );
        assert!(
            (hits[2] - 7.0).abs() < TOL,
            "t2={}, expected 7.0 (x=2)",
            hits[2]
        );
        assert!(
            (hits[3] - 9.0).abs() < TOL,
            "t3={}, expected 9.0 (x=4)",
            hits[3]
        );
    }

    #[test]
    fn solve_quadratic_discriminant_zero() {
        // t^2 - 2t + 1 = 0 => t = 1 (double root)
        let roots = solve_quadratic(1.0, -2.0, 1.0);
        assert_eq!(
            roots.len(),
            1,
            "expected 1 root, got {}: {:?}",
            roots.len(),
            roots
        );
        assert!(
            (roots[0] - 1.0).abs() < TOL,
            "root={}, expected 1.0",
            roots[0]
        );
    }

    #[test]
    fn solve_quadratic_no_real_roots() {
        // t^2 + 1 = 0 => no real roots
        let roots = solve_quadratic(1.0, 0.0, 1.0);
        assert!(roots.is_empty(), "expected no roots, got {:?}", roots);
    }

    #[test]
    fn ray_torus_off_axis() {
        // Torus R=3, r=1 in XY plane. Vertical ray through tube center at (3,0,0).
        // Tube cross-section at z=±1, so hits at t=4 and t=6 from origin z=5.
        let tor = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0).unwrap();
        let origin = Point3::new(3.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);

        let hits = ray_torus(origin, dir, &tor);
        assert_eq!(
            hits.len(),
            2,
            "vertical ray through tube should hit twice, got {}: {:?}",
            hits.len(),
            hits
        );
        // t = 5 - z_hit. Tube at (3,0,0) has z = ±1, so hits at z=1 (t=4) and z=-1 (t=6).
        assert!((hits[0] - 4.0).abs() < 0.1, "t0={}, expected ~4.0", hits[0]);
        assert!((hits[1] - 6.0).abs() < 0.1, "t1={}, expected ~6.0", hits[1]);
        // Verify hits are on the torus surface
        for &t in &hits {
            let p = origin + dir * t;
            let pv = p - tor.center();
            let z = pv.dot(tor.z_axis());
            let r_major = (pv - tor.z_axis() * z).length();
            let dist_to_tube = ((r_major - tor.major_radius()).powi(2) + z * z).sqrt();
            assert!(
                (dist_to_tube - tor.minor_radius()).abs() < 1e-6,
                "hit not on torus surface"
            );
        }
    }

    /// Regression: oblique irrational rays from inside the tube of an
    /// R=6, r=2 torus. The old Ferrari solver returned zero roots for some of
    /// these rays and off-surface spurious roots for others (hits at |z| > r
    /// on a torus spanning z in `[-r, r]`), flipping classification parity.
    #[test]
    fn ray_torus_oblique_from_inside_tube() {
        let tor = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0, 2.0).unwrap();
        let dirs = [
            Vec3::new(
                0.573_576_436_351_046,
                0.740_535_693_464_567_5,
                0.350_889_803_483_932_2,
            ),
            Vec3::new(
                -0.350_889_803_483_932_2,
                0.573_576_436_351_046,
                0.740_535_693_464_567_5,
            ),
            Vec3::new(
                0.740_535_693_464_567_5,
                -0.350_889_803_483_932_2,
                0.573_576_436_351_046,
            ),
        ];
        // Tube-center origins at several azimuths, plus off-center interiors.
        let mut origins = Vec::new();
        for theta in [0.05_f64, std::f64::consts::PI / 3.0, 2.0, 4.5] {
            origins.push(Point3::new(6.0 * theta.cos(), 6.0 * theta.sin(), 0.0));
            origins.push(Point3::new(4.5 * theta.cos(), 4.5 * theta.sin(), 0.5));
        }
        for origin in origins {
            for dir in dirs {
                let hits = ray_torus(origin, dir, &tor);
                // A forward ray from inside the tube must exit: at least one hit.
                assert!(
                    !hits.is_empty(),
                    "no roots for origin {origin:?} dir {dir:?}"
                );
                for &t in &hits {
                    let p = origin + dir * t;
                    let pv = p - tor.center();
                    let z = pv.dot(tor.z_axis());
                    let r_major = (pv - tor.z_axis() * z).length();
                    let dist_to_tube = ((r_major - tor.major_radius()).powi(2) + z * z).sqrt();
                    assert!(
                        (dist_to_tube - tor.minor_radius()).abs() < 1e-6,
                        "spurious off-surface root t={t} for origin {origin:?} dir {dir:?}"
                    );
                }
            }
        }
    }
}

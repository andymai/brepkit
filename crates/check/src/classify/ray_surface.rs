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
/// Returns up to 4 positive-t values (quartic equation).
#[allow(clippy::similar_names)]
pub fn ray_torus(origin: Point3, direction: Vec3, tor: &ToroidalSurface) -> SmallVec<[f64; 4]> {
    let z_axis = tor.z_axis();
    let x_axis = tor.x_axis();
    let y_axis = tor.y_axis();

    let ov = origin - tor.center();
    let o = Vec3::new(ov.dot(x_axis), ov.dot(y_axis), ov.dot(z_axis));
    let d = Vec3::new(
        direction.dot(x_axis),
        direction.dot(y_axis),
        direction.dot(z_axis),
    );

    let big_r = tor.major_radius();
    let small_r = tor.minor_radius();

    let sum_d_sq = d.dot(d);
    let sum_od = o.dot(d);
    let sum_o_sq = o.dot(o);
    let k = sum_o_sq - small_r * small_r - big_r * big_r;

    let c4 = sum_d_sq * sum_d_sq;
    let c3 = 4.0 * sum_d_sq * sum_od;
    let c2 = 2.0 * sum_d_sq * k + 4.0 * sum_od * sum_od + 4.0 * big_r * big_r * d.z() * d.z();
    let c1 = 4.0 * k * sum_od + 8.0 * big_r * big_r * o.z() * d.z();
    let c0 = k * k - 4.0 * big_r * big_r * (small_r * small_r - o.z() * o.z());

    solve_quartic(c4, c3, c2, c1, c0)
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
///
/// Used internally by `solve_quartic` where the quadratic variable `u` is not
/// the ray parameter — the positive filter is applied later after the shift.
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

/// Find one real root of `t^3 + p*t + q = 0` (depressed cubic).
fn solve_cubic_one_real(p: f64, q: f64) -> f64 {
    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        let u = (-q / 2.0 + sqrt_disc).cbrt();
        let v = (-q / 2.0 - sqrt_disc).cbrt();
        u + v
    } else {
        // Three real roots — use trigonometric method, return first.
        let r = (-p * p * p / 27.0).sqrt();
        let theta = (-q / (2.0 * r)).acos();
        2.0 * r.cbrt() * (theta / 3.0).cos()
    }
}

/// Solve `a*t^3 + b*t^2 + c*t + d = 0` returning positive roots.
pub fn solve_cubic(a: f64, b: f64, c: f64, d: f64) -> SmallVec<[f64; 4]> {
    if a.abs() < NEAR_ZERO {
        return solve_quadratic(b, c, d);
    }

    // Normalize: t^3 + pt^2 + qt + r = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Depress: substitute t = u - p/3
    let p_shift = q - p * p / 3.0;
    let q_shift = 2.0 * p * p * p / 27.0 - p * q / 3.0 + r;

    let disc = q_shift * q_shift / 4.0 + p_shift * p_shift * p_shift / 27.0;

    let mut roots = SmallVec::new();

    if disc > NEAR_ZERO {
        // One real root.
        let u = solve_cubic_one_real(p_shift, q_shift);
        let t = u - p / 3.0;
        if t > RAY_T_MIN {
            roots.push(t);
        }
    } else {
        // Three real roots via trigonometric method.
        let r_mag = (-p_shift * p_shift * p_shift / 27.0).sqrt();
        if r_mag.abs() < NEAR_ZERO {
            let t = -p / 3.0;
            if t > RAY_T_MIN {
                roots.push(t);
            }
            return roots;
        }
        let theta = (-q_shift / (2.0 * r_mag)).clamp(-1.0, 1.0).acos();
        let cbrt_r = r_mag.cbrt();
        for k in 0..3 {
            #[allow(clippy::cast_precision_loss)]
            let angle = (theta + 2.0 * std::f64::consts::PI * k as f64) / 3.0;
            let t = 2.0 * cbrt_r * angle.cos() - p / 3.0;
            if t > RAY_T_MIN {
                roots.push(t);
            }
        }
    }

    roots.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

/// Solve `a*t^3 + b*t^2 + c*t + d = 0` returning ALL real roots (no positive filter).
///
/// Used internally by `solve_quartic` for the resolvent cubic, where the
/// cubic variable is not the ray parameter and negative roots are valid.
fn solve_cubic_all(a: f64, b: f64, c: f64, d: f64) -> SmallVec<[f64; 4]> {
    if a.abs() < NEAR_ZERO {
        return solve_quadratic_all(b, c, d);
    }

    // Normalize: t^3 + pt^2 + qt + r = 0
    let p = b / a;
    let q = c / a;
    let r = d / a;

    // Depress: substitute t = u - p/3
    let p_shift = q - p * p / 3.0;
    let q_shift = 2.0 * p * p * p / 27.0 - p * q / 3.0 + r;

    let disc = q_shift * q_shift / 4.0 + p_shift * p_shift * p_shift / 27.0;

    let mut roots = SmallVec::new();

    if disc > NEAR_ZERO {
        // One real root.
        let u = solve_cubic_one_real(p_shift, q_shift);
        roots.push(u - p / 3.0);
    } else {
        // Three real roots via trigonometric method.
        let r_mag = (-p_shift * p_shift * p_shift / 27.0).sqrt();
        if r_mag.abs() < NEAR_ZERO {
            roots.push(-p / 3.0);
            return roots;
        }
        let theta = (-q_shift / (2.0 * r_mag)).clamp(-1.0, 1.0).acos();
        let cbrt_r = r_mag.cbrt();
        for k in 0..3 {
            #[allow(clippy::cast_precision_loss)]
            let angle = (theta + 2.0 * std::f64::consts::PI * k as f64) / 3.0;
            roots.push(2.0 * cbrt_r * angle.cos() - p / 3.0);
        }
    }

    roots.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    roots
}

/// Solve `c4*t^4 + c3*t^3 + c2*t^2 + c1*t + c0 = 0` returning positive real roots.
///
/// Uses depressed quartic reduction followed by Ferrari's method.
#[allow(clippy::many_single_char_names)]
pub fn solve_quartic(c4: f64, c3: f64, c2: f64, c1: f64, c0: f64) -> SmallVec<[f64; 4]> {
    if c4.abs() < NEAR_ZERO {
        return solve_cubic(c3, c2, c1, c0);
    }

    // Normalize: t^4 + a*t^3 + b*t^2 + c*t + d = 0
    let a = c3 / c4;
    let b = c2 / c4;
    let c = c1 / c4;
    let d = c0 / c4;

    // Depressed quartic via substitution t = u - a/4:
    // u^4 + p*u^2 + q*u + r = 0
    let a2 = a * a;
    let p = b - 3.0 * a2 / 8.0;
    let q = c - a * b / 2.0 + a2 * a / 8.0;
    let r = d - a * c / 4.0 + a2 * b / 16.0 - 3.0 * a2 * a2 / 256.0;
    let shift = -a / 4.0;

    let mut roots = SmallVec::new();

    if q.abs() < NEAR_ZERO {
        // Biquadratic: u^4 + p*u^2 + r = 0
        let inner_roots = solve_quadratic_all(1.0, p, r);
        for ir in &inner_roots {
            if *ir >= 0.0 {
                let s = ir.sqrt();
                let t1 = s + shift;
                let t2 = -s + shift;
                if t1 > RAY_T_MIN {
                    roots.push(t1);
                }
                if t2 > RAY_T_MIN {
                    roots.push(t2);
                }
            }
        }
    } else {
        // Ferrari's method: find y from the resolvent cubic
        // y^3 - (p/2)*y^2 - r*y + (r*p/2 - q^2/8) = 0
        let cubic_roots = solve_cubic_all(1.0, -p / 2.0, -r, r * p / 2.0 - q * q / 8.0);

        // Pick the largest real root for numerical stability
        let y = cubic_roots.iter().copied().reduce(f64::max).unwrap_or(0.0);

        // Factor: (u^2 + s*u + t1)(u^2 - s*u + t2) = 0
        // where s = sqrt(2y - p), t1 = y + q/(2s), t2 = y - q/(2s)
        let disc = (2.0 * y - p).max(0.0);
        let s = disc.sqrt();

        if s.abs() < NEAR_ZERO {
            // Degenerate: try biquadratic with adjusted r
            let inner_roots = solve_quadratic_all(1.0, p, r);
            for ir in &inner_roots {
                if *ir >= 0.0 {
                    let sq = ir.sqrt();
                    let t1 = sq + shift;
                    let t2 = -sq + shift;
                    if t1 > RAY_T_MIN {
                        roots.push(t1);
                    }
                    if t2 > RAY_T_MIN {
                        roots.push(t2);
                    }
                }
            }
        } else {
            let w = q / (2.0 * s);
            let quad1 = solve_quadratic_all(1.0, s, y + w);
            let quad2 = solve_quadratic_all(1.0, -s, y - w);

            for &u in quad1.iter().chain(quad2.iter()) {
                let t = u + shift;
                if t > RAY_T_MIN {
                    roots.push(t);
                }
            }
        }
    }

    roots.sort_by(|x: &f64, y_val: &f64| x.partial_cmp(y_val).unwrap_or(std::cmp::Ordering::Equal));
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
}

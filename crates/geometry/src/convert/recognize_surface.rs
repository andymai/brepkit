//! Recognize NURBS surfaces as elementary analytic forms.
//!
//! Ported from `brepkit-heal/analysis/canonical.rs` but expressed entirely
//! in terms of `brepkit-math` types (no topology dependency). The result is
//! a [`RecognizedSurface`] enum describing the best-fit analytic surface, or
//! [`RecognizedSurface::NotRecognized`] when no match is found.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};

/// The analytic surface form recognized from a NURBS surface.
#[derive(Debug, Clone, PartialEq)]
pub enum RecognizedSurface {
    /// Recognized as a plane.
    Plane {
        /// Outward normal (unit vector).
        normal: Vec3,
        /// Signed distance from origin: `normal · (any point on plane)`.
        d: f64,
    },
    /// Recognized as a cylinder.
    Cylinder {
        /// A point on the cylinder axis.
        origin: Point3,
        /// Axis direction (unit vector).
        axis: Vec3,
        /// Cylinder radius.
        radius: f64,
    },
    /// Recognized as a sphere.
    Sphere {
        /// Center of the sphere.
        center: Point3,
        /// Sphere radius.
        radius: f64,
    },
    /// The surface could not be matched to any elementary form.
    NotRecognized,
}

/// Attempt to recognize a NURBS surface as an elementary analytic surface.
///
/// Tries recognition in order: plane, cylinder, sphere. Returns the first
/// match whose maximum sample deviation is within `tolerance`.
#[must_use]
pub fn recognize_surface(surface: &NurbsSurface, tolerance: f64) -> RecognizedSurface {
    if let Some((normal, d)) = try_recognize_plane(surface, tolerance) {
        return RecognizedSurface::Plane { normal, d };
    }
    if let Some((origin, axis, radius)) = try_recognize_cylinder(surface, tolerance) {
        return RecognizedSurface::Cylinder {
            origin,
            axis,
            radius,
        };
    }
    if let Some((center, radius)) = try_recognize_sphere(surface, tolerance) {
        return RecognizedSurface::Sphere { center, radius };
    }
    RecognizedSurface::NotRecognized
}

// ── Plane recognition ─────────────────────────────────────────────────────────

/// Check if all control points of a NURBS surface lie on a single plane.
///
/// Returns `(normal, d)` if recognized, where `d = normal · p0`.
fn try_recognize_plane(surface: &NurbsSurface, tolerance: f64) -> Option<(Vec3, f64)> {
    let cps = surface.control_points();
    if cps.is_empty() || cps[0].is_empty() {
        return None;
    }

    // Collect all control points.
    let mut all_pts: Vec<Point3> = Vec::new();
    for row in cps {
        for pt in row {
            all_pts.push(*pt);
        }
    }

    if all_pts.len() < 3 {
        return None;
    }

    // Find a normal from the first 3 non-collinear points.
    let p0 = all_pts[0];
    let mut normal: Option<Vec3> = None;
    'outer: for i in 1..all_pts.len() {
        let v1 = all_pts[i] - p0;
        for pt in all_pts.iter().skip(i + 1) {
            let v2 = *pt - p0;
            let n = v1.cross(v2);
            if n.length() > tolerance {
                if let Ok(normalized) = n.normalize() {
                    normal = Some(normalized);
                    break 'outer;
                }
            }
        }
    }

    let n = normal?;
    let d = n.dot(Vec3::new(p0.x(), p0.y(), p0.z()));

    // Check all control points lie within tolerance of the plane.
    for pt in &all_pts {
        let dist = n.dot(Vec3::new(pt.x(), pt.y(), pt.z())) - d;
        if dist.abs() > tolerance {
            return None;
        }
    }

    Some((n, d))
}

// ── Cylinder recognition ──────────────────────────────────────────────────────

/// Check if a NURBS surface is a cylinder.
///
/// Estimates the axis from the v-direction within each control-point row
/// (averaged across all rows), then verifies that an 8×8 sample grid lies
/// at a consistent radial distance from that axis.
///
/// This handles both the exact rational form (9 u-rows × 2 v-columns) and
/// the sampled bilinear form (nu rows × nv columns).
#[allow(clippy::items_after_statements)]
fn try_recognize_cylinder(surface: &NurbsSurface, tolerance: f64) -> Option<(Point3, Vec3, f64)> {
    let cps = surface.control_points();
    if cps.len() < 2 {
        return None;
    }
    for row in cps {
        if row.len() < 2 {
            return None;
        }
    }

    // Estimate axis as average of (last_col - first_col) across all rows.
    let mut axis_sum = Vec3::new(0.0, 0.0, 0.0);
    for row in cps {
        let v = row[row.len() - 1] - row[0];
        axis_sum += v;
    }
    #[allow(clippy::cast_precision_loss)]
    let axis_avg = axis_sum * (1.0 / cps.len() as f64);
    let axis_len = axis_avg.length();
    if axis_len < tolerance {
        return None;
    }
    let axis = axis_avg.normalize().ok()?;

    // Sample at an 8×8 grid of evaluated surface points.
    // We use evaluated points (not control points) for the axis origin because
    // rational NURBS control points are NOT on the surface — the centroid of
    // weighted CPs would be skewed.
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    const N: usize = 8;

    let mut samples: Vec<Point3> = Vec::with_capacity(N * N);
    for iu in 0..N {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (iu as f64) / ((N - 1) as f64);
        for iv in 0..N {
            #[allow(clippy::cast_precision_loss)]
            let v = v0 + (v1 - v0) * (iv as f64) / ((N - 1) as f64);
            samples.push(surface.evaluate(u, v));
        }
    }

    // Find the axis position by least-squares circle fitting in the plane
    // perpendicular to the axis. Project each sample to 2D (removing the
    // axial component), then solve the algebraic circle equation:
    //   x² + y² = 2·cx·x + 2·cy·y + (r² - cx² - cy²)
    // This is linear in (cx, cy, C) and gives the circle center.
    let ref_pt = samples[0];

    // Build a 2D coordinate system perpendicular to the axis.
    let perp1 = {
        let trial = if axis.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        let p = trial - axis * axis.dot(trial);
        p.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0))
    };
    let perp2 = axis.cross(perp1);

    // Project samples to 2D (perpendicular to axis).
    let pts_2d: Vec<(f64, f64)> = samples
        .iter()
        .map(|pt| {
            let v = *pt - ref_pt;
            (perp1.dot(v), perp2.dot(v))
        })
        .collect();

    // Solve least-squares: for each (x,y), x²+y² = 2*cx*x + 2*cy*y + C
    // ATA * [cx, cy, C/2] = ATb where A[i] = [2x, 2y, 1] and b[i] = x²+y²
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [0.0_f64; 3];
    for &(x, y) in &pts_2d {
        let rhs = x * x + y * y;
        let row = [2.0 * x, 2.0 * y, 1.0];
        for i in 0..3 {
            for j in 0..3 {
                ata[i][j] += row[i] * row[j];
            }
            atb[i] += row[i] * rhs;
        }
    }

    let sol = solve_3x3(ata, atb)?;
    let cx = sol[0];
    let cy = sol[1];
    // Recover axis origin in 3D.
    let origin = ref_pt + perp1 * cx + perp2 * cy;

    let mut radii: Vec<f64> = Vec::with_capacity(samples.len());
    for pt in &samples {
        let to_pt = *pt - origin;
        let along = axis.dot(to_pt);
        let radial = to_pt - axis * along;
        radii.push(radial.length());
    }

    if radii.is_empty() {
        return None;
    }

    let sum: f64 = radii.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_radius = sum / radii.len() as f64;

    if mean_radius < tolerance {
        return None; // Degenerate — axis passes through all points.
    }

    let max_dev = radii
        .iter()
        .map(|r| (r - mean_radius).abs())
        .fold(0.0_f64, f64::max);
    if max_dev > tolerance {
        return None;
    }

    Some((origin, axis, mean_radius))
}

// ── Sphere recognition ────────────────────────────────────────────────────────

/// Check if a NURBS surface is a sphere.
///
/// Samples an 8×8 grid, estimates the center by solving a 3×3 least-squares
/// system, then verifies all sample points are equidistant from that center.
#[allow(clippy::items_after_statements)]
fn try_recognize_sphere(surface: &NurbsSurface, tolerance: f64) -> Option<(Point3, f64)> {
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    const N: usize = 8;

    let mut samples: Vec<Point3> = Vec::with_capacity(N * N);

    for iu in 0..N {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (iu as f64) / ((N - 1) as f64);
        for iv in 0..N {
            #[allow(clippy::cast_precision_loss)]
            let v = v0 + (v1 - v0) * (iv as f64) / ((N - 1) as f64);
            samples.push(surface.evaluate(u, v));
        }
    }

    if samples.len() < 4 {
        return None;
    }

    // Solve least-squares for center using algebraic approach.
    // For each pair (p0, pi), the difference equation eliminates R²:
    //   2*(pi - p0) · c = pi² - p0²
    let sq = |p: Point3| p.x() * p.x() + p.y() * p.y() + p.z() * p.z();

    let n = samples.len();
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [0.0_f64; 3];

    let p0 = samples[0];
    let sq0 = sq(p0);

    for i in 1..n {
        let pi = samples[i];
        let a_row = [
            2.0 * (pi.x() - p0.x()),
            2.0 * (pi.y() - p0.y()),
            2.0 * (pi.z() - p0.z()),
        ];
        let bi = sq(pi) - sq0;

        for r in 0..3 {
            for c in 0..3 {
                ata[r][c] += a_row[r] * a_row[c];
            }
            atb[r] += a_row[r] * bi;
        }
    }

    let center = solve_3x3(ata, atb)?;
    let center_pt = Point3::new(center[0], center[1], center[2]);

    let mut distances: Vec<f64> = Vec::with_capacity(n);
    for pt in &samples {
        let d = Vec3::new(
            pt.x() - center_pt.x(),
            pt.y() - center_pt.y(),
            pt.z() - center_pt.z(),
        )
        .length();
        distances.push(d);
    }

    let sum: f64 = distances.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_radius = sum / distances.len() as f64;

    if mean_radius < tolerance {
        return None;
    }

    let max_dev = distances
        .iter()
        .map(|d| (d - mean_radius).abs())
        .fold(0.0_f64, f64::max);

    if max_dev > tolerance {
        return None;
    }

    Some((center_pt, mean_radius))
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Solve a 3×3 linear system `A * x = b` via Cramer's rule.
///
/// Returns `None` if the determinant is near zero (singular system).
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-30 {
        return None;
    }

    let inv = 1.0 / det;

    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv;

    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv;

    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv;

    Some([x0, x1, x2])
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;
    use crate::convert::surface_to_nurbs::{cylinder_to_nurbs, sphere_to_nurbs};

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    #[test]
    fn recognize_cylinder_round_trip() {
        let cyl = CylindricalSurface::new(origin(), z_axis(), 3.0).unwrap();
        let nurbs = cylinder_to_nurbs(&cyl, (0.0, 5.0)).unwrap();

        let result = recognize_surface(&nurbs, 1e-4);
        match result {
            RecognizedSurface::Cylinder { radius, .. } => {
                assert!((radius - 3.0).abs() < 0.01, "radius {radius} != 3.0");
            }
            other => panic!("expected Cylinder, got {other:?}"),
        }
    }

    #[test]
    fn recognize_sphere_round_trip() {
        let sphere = SphericalSurface::new(origin(), 5.0).unwrap();
        let nurbs = sphere_to_nurbs(&sphere).unwrap();

        let result = recognize_surface(&nurbs, 0.1);
        match result {
            RecognizedSurface::Sphere { center, radius } => {
                let dist = Vec3::new(center.x(), center.y(), center.z()).length();
                assert!(dist < 0.5, "center too far from origin: {dist}");
                assert!((radius - 5.0).abs() < 0.5, "radius {radius} != 5.0");
            }
            other => panic!("expected Sphere, got {other:?}"),
        }
    }
}

//! Canonical form analysis — recognize NURBS surfaces as elementary.
//!
//! Attempts to identify NURBS surfaces that represent known analytic
//! surfaces (planes, cylinders, cones, spheres) and returns the
//! corresponding [`RecognizedSurface`] variant.

use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::face::FaceSurface;

/// Result of surface recognition.
pub enum RecognizedSurface {
    /// Recognized as a plane.
    Plane {
        /// Normal vector.
        normal: Vec3,
        /// Signed distance from origin.
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
    /// Could not be recognized as an elementary surface.
    NotRecognized,
}

/// Attempt to recognize a NURBS surface as an elementary analytic surface.
///
/// Tries recognition in order: plane, cylinder, sphere. Returns the first
/// match that fits within tolerance.
#[must_use]
pub fn recognize_surface(surface: &NurbsSurface, tolerance: &Tolerance) -> RecognizedSurface {
    if let Some(FaceSurface::Plane { normal, d }) = try_recognize_plane(surface, tolerance) {
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

/// Check if all control points of a NURBS surface lie on a single plane.
fn try_recognize_plane(surface: &NurbsSurface, tolerance: &Tolerance) -> Option<FaceSurface> {
    let cps = surface.control_points();
    if cps.is_empty() || cps[0].is_empty() {
        return None;
    }

    // Collect all control points.
    let mut all_pts = Vec::new();
    for row in cps {
        for pt in row {
            all_pts.push(*pt);
        }
    }

    if all_pts.len() < 3 {
        return None;
    }

    // Find a normal from first 3 non-collinear points.
    let p0 = all_pts[0];
    let mut normal: Option<Vec3> = None;
    'outer: for i in 1..all_pts.len() {
        let v1 = all_pts[i] - p0;
        for pt in all_pts.iter().skip(i + 1) {
            let v2 = *pt - p0;
            let n = v1.cross(v2);
            if n.length() > tolerance.linear {
                if let Ok(normalized) = n.normalize() {
                    normal = Some(normalized);
                    break 'outer;
                }
            }
        }
    }

    let n = normal?;

    // Check all points lie on the plane.
    let d = n.dot(Vec3::new(p0.x(), p0.y(), p0.z()));
    for pt in &all_pts {
        let dist = n.dot(Vec3::new(pt.x(), pt.y(), pt.z())) - d;
        if dist.abs() > tolerance.linear {
            return None;
        }
    }

    Some(FaceSurface::Plane { normal: n, d })
}

/// Check if a NURBS surface is a cylinder.
///
/// Uses control points to estimate the axis direction (from edge-row
/// centroids), then checks that all sample points lie at a consistent
/// distance from that axis.
fn try_recognize_cylinder(
    surface: &NurbsSurface,
    tolerance: &Tolerance,
) -> Option<(Point3, Vec3, f64)> {
    let cps = surface.control_points();
    if cps.len() < 2 {
        return None;
    }
    for row in cps {
        if row.is_empty() {
            return None;
        }
    }

    // Compute centroids of the first and last control point rows.
    let first_row = &cps[0];
    let last_row = &cps[cps.len() - 1];

    let centroid_first = row_centroid(first_row);
    let centroid_last = row_centroid(last_row);

    let axis_vec = centroid_last - centroid_first;
    let axis_len = axis_vec.length();
    if axis_len < tolerance.linear {
        return None;
    }
    let axis = match axis_vec.normalize() {
        Ok(a) => a,
        Err(_) => return None,
    };

    // Use the midpoint of the two centroids as the axis origin.
    let origin = Point3::new(
        (centroid_first.x() + centroid_last.x()) * 0.5,
        (centroid_first.y() + centroid_last.y()) * 0.5,
        (centroid_first.z() + centroid_last.z()) * 0.5,
    );

    // Sample the surface at an 8x8 grid and measure distance to axis.
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let n_u = 8;
    let n_v = 8;

    let mut radii = Vec::with_capacity(n_u * n_v);

    for iu in 0..n_u {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (iu as f64) / ((n_u - 1) as f64);
        for iv in 0..n_v {
            #[allow(clippy::cast_precision_loss)]
            let v = v0 + (v1 - v0) * (iv as f64) / ((n_v - 1) as f64);
            let pt = surface.evaluate(u, v);
            let to_pt = Vec3::new(
                pt.x() - origin.x(),
                pt.y() - origin.y(),
                pt.z() - origin.z(),
            );
            let along_axis = axis.dot(to_pt);
            let radial = to_pt - axis * along_axis;
            radii.push(radial.length());
        }
    }

    if radii.is_empty() {
        return None;
    }

    // Compute mean radius.
    let sum: f64 = radii.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_radius = sum / radii.len() as f64;

    if mean_radius < tolerance.linear {
        return None; // Degenerate — all points on the axis.
    }

    // Check max deviation from the mean radius.
    let max_dev = radii
        .iter()
        .map(|r| (r - mean_radius).abs())
        .fold(0.0_f64, f64::max);

    if max_dev > tolerance.linear {
        return None;
    }

    Some((origin, axis, mean_radius))
}

/// Check if a NURBS surface is a sphere.
///
/// Samples the surface at an 8x8 grid, finds the center via
/// least-squares (average of all points), then checks that all
/// points are equidistant from that center within tolerance.
fn try_recognize_sphere(surface: &NurbsSurface, tolerance: &Tolerance) -> Option<(Point3, f64)> {
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let n_u = 8;
    let n_v = 8;

    let mut samples = Vec::with_capacity(n_u * n_v);

    for iu in 0..n_u {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (iu as f64) / ((n_u - 1) as f64);
        for iv in 0..n_v {
            #[allow(clippy::cast_precision_loss)]
            let v = v0 + (v1 - v0) * (iv as f64) / ((n_v - 1) as f64);
            samples.push(surface.evaluate(u, v));
        }
    }

    if samples.len() < 4 {
        return None;
    }

    // Find center: solve least-squares for the point equidistant from
    // all samples. Use the algebraic approach: minimize sum of
    // (|p - c|^2 - R^2)^2 by setting up the linear system.
    //
    // Expanding |p - c|^2 = R^2:
    //   px^2 - 2*px*cx + cx^2 + py^2 - 2*py*cy + cy^2 + pz^2 - 2*pz*cz + cz^2 = R^2
    //   -2*px*cx - 2*py*cy - 2*pz*cz + (cx^2 + cy^2 + cz^2 - R^2) = -(px^2 + py^2 + pz^2)
    //
    // Let w = cx^2 + cy^2 + cz^2 - R^2
    // For each pair of points (i, j), subtract to eliminate w:
    //   2*(pj_x - pi_x)*cx + 2*(pj_y - pi_y)*cy + 2*(pj_z - pi_z)*cz = pj^2 - pi^2
    //
    // Build overdetermined system A*[cx,cy,cz]^T = b and solve via ATA * x = ATb.

    let n = samples.len();
    let mut ata = [[0.0_f64; 3]; 3];
    let mut atb = [0.0_f64; 3];

    let sq = |p: Point3| p.x() * p.x() + p.y() * p.y() + p.z() * p.z();

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

    // Solve 3x3 system via Cramer's rule.
    let center = solve_3x3(ata, atb)?;
    let center_pt = Point3::new(center[0], center[1], center[2]);

    // Compute distances from center to all samples.
    let mut distances = Vec::with_capacity(n);
    for pt in &samples {
        let d = Vec3::new(
            pt.x() - center_pt.x(),
            pt.y() - center_pt.y(),
            pt.z() - center_pt.z(),
        )
        .length();
        distances.push(d);
    }

    // Mean radius.
    let sum: f64 = distances.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_radius = sum / distances.len() as f64;

    if mean_radius < tolerance.linear {
        return None; // Degenerate.
    }

    // Check max deviation.
    let max_dev = distances
        .iter()
        .map(|d| (d - mean_radius).abs())
        .fold(0.0_f64, f64::max);

    if max_dev > tolerance.linear {
        return None;
    }

    Some((center_pt, mean_radius))
}

/// Compute the centroid of a row of control points.
fn row_centroid(row: &[Point3]) -> Point3 {
    if row.is_empty() {
        return Point3::new(0.0, 0.0, 0.0);
    }
    let mut sx = 0.0;
    let mut sy = 0.0;
    let mut sz = 0.0;
    for p in row {
        sx += p.x();
        sy += p.y();
        sz += p.z();
    }
    #[allow(clippy::cast_precision_loss)]
    let inv = 1.0 / row.len() as f64;
    Point3::new(sx * inv, sy * inv, sz * inv)
}

/// Solve a 3x3 linear system `A * x = b` using Cramer's rule.
///
/// Returns `None` if the determinant is near zero.
fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);

    if det.abs() < 1e-30 {
        return None;
    }

    let inv_det = 1.0 / det;

    let x0 = (b[0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (b[1] * a[2][2] - a[1][2] * b[2])
        + a[0][2] * (b[1] * a[2][1] - a[1][1] * b[2]))
        * inv_det;

    let x1 = (a[0][0] * (b[1] * a[2][2] - a[1][2] * b[2])
        - b[0] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * b[2] - b[1] * a[2][0]))
        * inv_det;

    let x2 = (a[0][0] * (a[1][1] * b[2] - b[1] * a[2][1])
        - a[0][1] * (a[1][0] * b[2] - b[1] * a[2][0])
        + b[0] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]))
        * inv_det;

    Some([x0, x1, x2])
}

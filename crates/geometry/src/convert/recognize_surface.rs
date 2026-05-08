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
    /// Recognized as a cone.
    Cone {
        /// The cone's apex (point where radius = 0).
        apex: Point3,
        /// Cone axis direction (from apex into the cone, unit vector).
        axis: Vec3,
        /// Half-angle from the radial plane to the cone generator
        /// (radians, in `(0, π/2)`).
        half_angle: f64,
    },
    /// The surface could not be matched to any elementary form.
    NotRecognized,
}

/// Attempt to recognize a NURBS surface as an elementary analytic surface.
///
/// Tries recognition in order: plane, cylinder, sphere, cone. Returns
/// the first match whose maximum sample deviation is within
/// `tolerance`. Cylinder is tested before cone so that constant-radius
/// surfaces are classified as `Cylinder`, not as `Cone` with apex at
/// infinity.
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
    if let Some((apex, axis, half_angle)) = try_recognize_cone(surface, tolerance) {
        return RecognizedSurface::Cone {
            apex,
            axis,
            half_angle,
        };
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

// ── Cone recognition ──────────────────────────────────────────────────────────

/// Check if all sampled surface points lie on a cone.
///
/// Estimates the axis from the average of "last column − first column"
/// across all CP rows (same as cylinder, since cone has the same
/// rotational structure). Then verifies samples lie on a cone by
/// checking that:
///
/// 1. The axial-component vs radial-component relationship is linear
///    (samples lie on a 2D wedge in `(axial, radial)` space).
/// 2. The radial component is consistent for all u at each fixed v
///    (each iso-v line is a circle around the axis).
///
/// The slope of the radial-vs-axial line gives `cot(half_angle)`; the
/// apex is the (axial, 0) intercept extrapolated from this line.
fn try_recognize_cone(surface: &NurbsSurface, tolerance: f64) -> Option<(Point3, Vec3, f64)> {
    const N: usize = 8;
    let cps = surface.control_points();
    if cps.len() < 2 {
        return None;
    }
    for row in cps {
        if row.len() < 2 {
            return None;
        }
    }

    // Estimate axis direction. For cones (unlike cylinders), the
    // (last_col - first_col) vector at row i has both an axial AND a
    // radial component (cos_a · radial_dir(u_i) + sin_a · axis) — so
    // averaging across u must cancel the radial part. The 33×9 CP
    // grid produced by `analytic_to_nurbs_sampled` duplicates the
    // u=0 and u=2π seam (CP[0] and CP[N-1] at the same 3D point),
    // which biases the unweighted sum. Skip the last row to remove
    // the duplicate before averaging.
    let n_rows = cps.len();
    let row_count = if n_rows >= 3 && (cps[0][0] - cps[n_rows - 1][0]).length() < tolerance {
        n_rows - 1
    } else {
        n_rows
    };
    let mut axis_sum = Vec3::new(0.0, 0.0, 0.0);
    for row in cps.iter().take(row_count) {
        let v = row[row.len() - 1] - row[0];
        axis_sum += v;
    }
    #[allow(clippy::cast_precision_loss)]
    let axis_avg = axis_sum * (1.0 / row_count as f64);
    if axis_avg.length() < tolerance {
        return None;
    }
    let axis = axis_avg.normalize().ok()?;

    // Sample at an 8×8 grid. CRITICAL: use OPEN range in u to avoid
    // duplicating the closing seam point (u_nurbs=0 and u_nurbs=1
    // coincide for full-revolution surfaces). Duplicates bias the
    // centroid off-axis, which throws off the radial-component
    // computation for samples near the duplicate.
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let mut samples: Vec<Point3> = Vec::with_capacity(N * N);
    for iu in 0..N {
        #[allow(clippy::cast_precision_loss)]
        let u = u0 + (u1 - u0) * (iu as f64 + 0.5) / (N as f64);
        for iv in 0..N {
            #[allow(clippy::cast_precision_loss)]
            let v = v0 + (v1 - v0) * (iv as f64) / ((N - 1) as f64);
            samples.push(surface.evaluate(u, v));
        }
    }

    // Estimate the apex (axis origin) by linear-fitting (axial,
    // radial) pairs. For each sample, `axial = axis · (p − sample[0])`
    // is a relative axial distance; `radial` is the perpendicular
    // distance from sample[0]'s axial projection. Wait — for cone
    // recognition we need a robust BUT axis-relative reference. Use
    // the centroid of all samples as the "anchor" for axial measurement.
    #[allow(clippy::cast_precision_loss)]
    let inv_n = 1.0 / samples.len() as f64;
    let mut anchor_x = 0.0_f64;
    let mut anchor_y = 0.0_f64;
    let mut anchor_z = 0.0_f64;
    for p in &samples {
        anchor_x += p.x();
        anchor_y += p.y();
        anchor_z += p.z();
    }
    let anchor = Point3::new(anchor_x * inv_n, anchor_y * inv_n, anchor_z * inv_n);

    // Measure axial and radial offsets from anchor along the axis.
    // For each sample, compute axial = axis · (p − anchor) and
    // radial = |(p − anchor) − axial · axis|. For a true cone with
    // apex at (anchor + axial_apex · axis), the radial component is
    // a linear function of axial: radial = |slope · (axial − axial_apex)|.
    let mut axials: Vec<f64> = Vec::with_capacity(samples.len());
    let mut radials: Vec<f64> = Vec::with_capacity(samples.len());
    for p in &samples {
        let to_p = *p - anchor;
        let along = axis.dot(to_p);
        let radial_vec = to_p - axis * along;
        axials.push(along);
        radials.push(radial_vec.length());
    }

    // Reject degenerate (all radials zero or all the same): would be
    // a line/cylinder, not a cone.
    let max_r = radials.iter().fold(0.0_f64, |m, &r| m.max(r));
    let min_r = radials.iter().fold(f64::INFINITY, |m, &r| m.min(r));
    if max_r - min_r < tolerance {
        return None; // Constant radius → cylinder (handled earlier).
    }

    // Linear fit: radial = m · axial + b. Then cone apex is at
    // axial_apex = -b / m, with radial_apex = 0. For axisymmetry,
    // the radial side should be an ABSOLUTE value (always >= 0); we
    // exploit the fact that radial is a vector magnitude, so on the
    // cone the relationship `radial = slope · (axial − axial_apex)`
    // holds with `slope > 0` for axials > axial_apex.
    //
    // We use unsigned-radial least-squares: pick the slope from a
    // simple linear regression of (axial, radial). For a true cone
    // the residual should be near zero.
    let n_f = samples.len() as f64;
    let sum_a: f64 = axials.iter().sum();
    let sum_r: f64 = radials.iter().sum();
    let mean_a = sum_a / n_f;
    let mean_r = sum_r / n_f;
    let mut s_aa = 0.0_f64;
    let mut s_ar = 0.0_f64;
    for i in 0..samples.len() {
        let da = axials[i] - mean_a;
        let dr = radials[i] - mean_r;
        s_aa += da * da;
        s_ar += da * dr;
    }
    if s_aa < 1e-30 {
        return None;
    }
    let slope = s_ar / s_aa;
    let intercept = mean_r - slope * mean_a;
    if slope.abs() < tolerance {
        return None; // Slope ≈ 0 → cylinder.
    }
    // Apex axial position relative to anchor.
    let axial_apex = -intercept / slope;

    // Verify residuals.
    for i in 0..samples.len() {
        let pred = slope * axials[i] + intercept;
        if (radials[i] - pred).abs() > tolerance {
            return None;
        }
    }

    // Compute half-angle from slope. The cone equation in local
    // (axial, radial) coords is `radial = (axial - axial_apex) ·
    // |slope|` for axial > axial_apex. The slope equals
    // cos(half_angle) / sin(half_angle) = cot(half_angle), so
    // half_angle = atan(1 / |slope|).
    //
    // brepkit's half_angle convention is the angle from the RADIAL
    // plane to the generator, so half_angle ∈ (0, π/2).
    let half_angle = (1.0 / slope.abs()).atan();
    if !(0.0 < half_angle && half_angle < std::f64::consts::FRAC_PI_2) {
        return None;
    }

    // Apex in 3D. Cone axis points from apex INTO the cone (positive
    // axial direction). If our slope is negative (radial decreases
    // with positive axial), the apex is in the +axial direction;
    // axis should point AWAY from the apex (negative-axial-from-apex
    // direction, i.e., positive `slope` convention).
    let apex_offset = axis * axial_apex;
    let apex = anchor + apex_offset;
    let cone_axis = if slope > 0.0 { axis } else { -axis };

    Some((apex, cone_axis, half_angle))
}

// ── Utilities ─────────────────────────────────────────────────────────────────

/// Solve a 3×3 linear system `A * x = b` via Cramer's rule.
///
/// Returns `None` if the determinant is near zero (singular system).
/// Exposed at `pub(super)` so [`super::recognize_curve`] can reuse it
/// (avoids duplicating the same 3×3 solver).
pub(super) fn solve_3x3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
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

// ── Lightweight detection ────────────────────────────────────────────────────

/// Detected geometric kind of a NURBS surface (without recovering full analytic
/// parameters). Cheaper than [`recognize_surface`] when you only need a type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedSurfaceKind {
    /// All sampled points lie on a plane.
    Plane,
    /// All sampled points are equidistant from a center (sphere).
    Sphere,
    /// All sampled points are equidistant from an axis (cylinder).
    Cylinder,
    /// Generic B-spline surface.
    BSpline,
}

impl DetectedSurfaceKind {
    /// Returns the lowercase string tag for this surface kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Plane => "plane",
            Self::Sphere => "sphere",
            Self::Cylinder => "cylinder",
            Self::BSpline => "bspline",
        }
    }
}

/// Detect the geometric kind of a NURBS surface by sampling.
///
/// Samples an 8x8 grid and checks for sphere (equidistant from centroid) or
/// cylinder (equidistant from a PCA axis). Falls back to `BSpline`.
///
/// This is a lightweight heuristic — use [`recognize_surface`] for full analytic
/// parameter recovery.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_surface_kind(surface: &NurbsSurface) -> DetectedSurfaceKind {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let n = 8; // 8x8 grid = 64 sample points

    let mut points = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let u = u_min + (u_max - u_min) * (i as f64) / ((n - 1) as f64);
            let v = v_min + (v_max - v_min) * (j as f64) / ((n - 1) as f64);
            points.push(surface.evaluate(u, v));
        }
    }

    // Compute center as average.
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let np = points.len() as f64;
    let center = Point3::new(cx / np, cy / np, cz / np);

    // Plane test: check if all points are coplanar.
    // Find a normal from the first non-degenerate cross product.
    let mut plane_normal = None;
    for i in 1..points.len() {
        for j in (i + 1)..points.len() {
            let v0 = points[i] - center;
            let v1 = points[j] - center;
            let n = v0.cross(v1);
            if let Ok(normalized) = n.normalize() {
                plane_normal = Some(normalized);
                break;
            }
        }
        if plane_normal.is_some() {
            break;
        }
    }
    if let Some(normal) = plane_normal {
        let is_plane = points
            .iter()
            .all(|p| (*p - center).dot(normal).abs() < 1e-6);
        if is_plane {
            return DetectedSurfaceKind::Plane;
        }
    }

    // Check if all points equidistant from center (sphere test).
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / np;

    if avg_dist < 1e-10 {
        return DetectedSurfaceKind::BSpline;
    }

    let tol = avg_dist * 1e-3; // 0.1% relative tolerance
    let is_sphere = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_sphere {
        return DetectedSurfaceKind::Sphere;
    }

    // Cylinder test: points should be equidistant from an axis line.
    if let Some(axis_dir) = estimate_cylinder_axis(&points, center) {
        let projected_distances: Vec<f64> = points
            .iter()
            .map(|p| {
                let v = *p - center;
                let along_axis = v.dot(axis_dir);
                let radial = Vec3::new(
                    v.x() - axis_dir.x() * along_axis,
                    v.y() - axis_dir.y() * along_axis,
                    v.z() - axis_dir.z() * along_axis,
                );
                radial.length()
            })
            .collect();

        let avg_r = projected_distances.iter().sum::<f64>() / np;
        if avg_r > 1e-10 {
            let r_tol = avg_r * 1e-3;
            let is_cylinder = projected_distances
                .iter()
                .all(|d| (d - avg_r).abs() < r_tol);
            if is_cylinder {
                return DetectedSurfaceKind::Cylinder;
            }
        }
    }

    DetectedSurfaceKind::BSpline
}

/// Estimate the cylinder axis direction from a set of surface sample points
/// using a simple PCA-like approach (direction of maximum variance).
fn estimate_cylinder_axis(points: &[Point3], center: Point3) -> Option<Vec3> {
    // Build covariance matrix.
    let mut cxx = 0.0_f64;
    let mut cxy = 0.0_f64;
    let mut cxz = 0.0_f64;
    let mut cyy = 0.0_f64;
    let mut cyz = 0.0_f64;
    let mut czz = 0.0_f64;

    for p in points {
        let dx = p.x() - center.x();
        let dy = p.y() - center.y();
        let dz = p.z() - center.z();
        cxx += dx * dx;
        cxy += dx * dy;
        cxz += dx * dz;
        cyy += dy * dy;
        cyz += dy * dz;
        czz += dz * dz;
    }

    // Power iteration to find the principal eigenvector.
    let mut v = Vec3::new(1.0, 0.0, 0.0);
    for _ in 0..20 {
        let new_v = Vec3::new(
            v.x().mul_add(cxx, v.y().mul_add(cxy, v.z() * cxz)),
            v.x().mul_add(cxy, v.y().mul_add(cyy, v.z() * cyz)),
            v.x().mul_add(cxz, v.y().mul_add(cyz, v.z() * czz)),
        );
        let len = new_v.length();
        if len < 1e-15 {
            return None;
        }
        v = Vec3::new(new_v.x() / len, new_v.y() / len, new_v.z() / len);
    }
    Some(v)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface};
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;
    use crate::convert::surface_to_nurbs::{cone_to_nurbs, cylinder_to_nurbs, sphere_to_nurbs};

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

    #[test]
    fn recognize_cone_round_trip() {
        // Cone with apex at origin, axis +z, half-angle π/6 (from
        // radial plane). At v=1 from apex (along generator),
        // radial = cos(π/6) ≈ 0.866, axial = sin(π/6) = 0.5.
        let half_angle = std::f64::consts::PI / 6.0;
        let cone = ConicalSurface::new(origin(), z_axis(), half_angle).unwrap();
        let nurbs = cone_to_nurbs(&cone, (1.0, 4.0)).unwrap();

        match recognize_surface(&nurbs, 0.05) {
            RecognizedSurface::Cone {
                apex,
                axis,
                half_angle: ha,
            } => {
                // Apex should be at origin within tolerance.
                assert!(
                    Vec3::new(apex.x(), apex.y(), apex.z()).length() < 0.05,
                    "apex {apex:?}"
                );
                // Axis should be along +z (or -z; both describe the same cone).
                assert!(
                    axis.dot(z_axis()).abs() > 1.0 - 1e-3,
                    "axis {axis:?} not aligned with z"
                );
                assert!(
                    (ha - half_angle).abs() < 1e-3,
                    "half_angle {ha} vs {half_angle}"
                );
            }
            other => panic!("expected Cone, got {other:?}"),
        }
    }

    #[test]
    fn cylinder_is_recognized_as_cylinder_not_cone() {
        // True cylinders should match Cylinder (tested first), not
        // fall through to Cone.
        let cyl = CylindricalSurface::new(origin(), z_axis(), 2.0).unwrap();
        let nurbs = cylinder_to_nurbs(&cyl, (0.0, 5.0)).unwrap();
        assert!(matches!(
            recognize_surface(&nurbs, 1e-4),
            RecognizedSurface::Cylinder { .. }
        ));
    }
}

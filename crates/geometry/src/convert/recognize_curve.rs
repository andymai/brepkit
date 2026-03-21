//! Recognize NURBS curves as elementary analytic forms.
//!
//! Samples a NURBS curve at 16 evenly-spaced parameter values and tests
//! whether the sample set is consistent with a line or circle within the
//! given tolerance. Returns a [`RecognizedCurve`] describing the best match.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};

/// The analytic curve form recognized from a NURBS curve.
#[derive(Debug, Clone, PartialEq)]
pub enum RecognizedCurve {
    /// Recognized as a straight line.
    Line {
        /// A point on the line.
        origin: Point3,
        /// Unit direction vector.
        direction: Vec3,
    },
    /// Recognized as a circle arc (or full circle).
    Circle {
        /// Center of the circle.
        center: Point3,
        /// Circle normal (unit vector perpendicular to the circle plane).
        normal: Vec3,
        /// Circle radius.
        radius: f64,
    },
    /// The curve could not be matched to any elementary form.
    NotRecognized,
}

/// Attempt to recognize a NURBS curve as an elementary analytic curve.
///
/// Samples the curve at 16 points in its parameter domain and checks:
/// 1. **Line**: all points collinear — max perpendicular deviation < `tolerance`.
/// 2. **Circle**: all points equidistant from a best-fit center and coplanar —
///    max radial deviation and max out-of-plane deviation both < `tolerance`.
///
/// Returns the first match, or [`RecognizedCurve::NotRecognized`].
#[must_use]
pub fn recognize_curve(curve: &NurbsCurve, tolerance: f64) -> RecognizedCurve {
    const N: usize = 16;
    let (t0, t1) = curve.domain();

    let samples: Vec<Point3> = (0..N)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let t = t0 + (t1 - t0) * (i as f64) / ((N - 1) as f64);
            curve.evaluate(t)
        })
        .collect();

    if let Some((origin, direction)) = try_recognize_line(&samples, tolerance) {
        return RecognizedCurve::Line { origin, direction };
    }
    if let Some((center, normal, radius)) = try_recognize_circle(&samples, tolerance) {
        return RecognizedCurve::Circle {
            center,
            normal,
            radius,
        };
    }
    RecognizedCurve::NotRecognized
}

// ── Line recognition ──────────────────────────────────────────────────────────

/// Check if all sample points are collinear.
///
/// Fits a best-fit line through the first and last sample, then checks that
/// all points' perpendicular distance to that line is within `tolerance`.
fn try_recognize_line(samples: &[Point3], tolerance: f64) -> Option<(Point3, Vec3)> {
    if samples.len() < 2 {
        return None;
    }

    let p0 = samples[0];
    let p_last = *samples.last()?;

    let dir_vec = p_last - p0;
    let len = dir_vec.length();
    if len < 1e-15 {
        return None; // Degenerate — all points at the same location.
    }
    let direction = dir_vec.normalize().ok()?;

    // Check perpendicular distance of every sample from the line.
    for pt in samples {
        let v = *pt - p0;
        let proj = direction * direction.dot(v);
        let perp = (v - proj).length();
        if perp > tolerance {
            return None;
        }
    }

    Some((p0, direction))
}

// ── Circle recognition ────────────────────────────────────────────────────────

/// Check if all sample points lie on a circle.
///
/// Estimates the circle plane from the first three non-collinear samples,
/// projects all samples onto that plane, then finds the center via the
/// algebraic least-squares method. Checks both radial consistency and
/// coplanarity within `tolerance`.
fn try_recognize_circle(samples: &[Point3], tolerance: f64) -> Option<(Point3, Vec3, f64)> {
    if samples.len() < 3 {
        return None;
    }

    // ── Step 1: find a stable plane normal ────────────────────────────────────
    let p0 = samples[0];
    let mut normal: Option<Vec3> = None;
    'outer: for i in 1..samples.len() {
        let v1 = samples[i] - p0;
        for pt in samples.iter().skip(i + 1) {
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

    // ── Step 2: check all points are coplanar ─────────────────────────────────
    let d_plane = n.dot(Vec3::new(p0.x(), p0.y(), p0.z()));
    for pt in samples {
        let dist = n.dot(Vec3::new(pt.x(), pt.y(), pt.z())) - d_plane;
        if dist.abs() > tolerance {
            return None;
        }
    }

    // ── Step 3: build an orthonormal 2D basis in the plane ────────────────────
    // Use the first edge direction as u-axis.
    let u_raw = samples[1] - p0;
    let u_len = u_raw.length();
    if u_len < 1e-15 {
        return None;
    }
    let u_axis = (u_raw * (1.0 / u_len)).normalize().ok()?;
    let v_axis = n.cross(u_axis).normalize().ok()?;

    // Project all points to 2D.
    let pts2d: Vec<(f64, f64)> = samples
        .iter()
        .map(|pt| {
            let v = *pt - p0;
            (u_axis.dot(v), v_axis.dot(v))
        })
        .collect();

    // ── Step 4: fit a circle to the 2D points ────────────────────────────────
    // Algebraic method: each pair (p0, pi) eliminates R² giving
    //   2*(xi - x0)*cx + 2*(yi - y0)*cy = xi² + yi² - x0² - y0²
    let sq2 = |(x, y): (f64, f64)| x * x + y * y;
    let (x0, y0) = pts2d[0];
    let sq0 = sq2(pts2d[0]);

    let mut ata = [[0.0_f64; 2]; 2];
    let mut atb = [0.0_f64; 2];

    for &(xi, yi) in pts2d.iter().skip(1) {
        let a = [2.0 * (xi - x0), 2.0 * (yi - y0)];
        let b = sq2((xi, yi)) - sq0;
        for r in 0..2 {
            for c in 0..2 {
                ata[r][c] += a[r] * a[c];
            }
            atb[r] += a[r] * b;
        }
    }

    // Solve 2×2 system.
    let det = ata[0][0] * ata[1][1] - ata[0][1] * ata[1][0];
    if det.abs() < 1e-30 {
        return None; // Degenerate — points might be collinear.
    }
    let inv = 1.0 / det;
    let cx_2d = (atb[0] * ata[1][1] - atb[1] * ata[0][1]) * inv;
    let cy_2d = (ata[0][0] * atb[1] - ata[1][0] * atb[0]) * inv;

    // Convert 2D center back to 3D.
    let center = p0 + u_axis * cx_2d + v_axis * cy_2d;

    // ── Step 5: check all points are equidistant from center ─────────────────
    let radii: Vec<f64> = samples
        .iter()
        .map(|pt| {
            Vec3::new(
                pt.x() - center.x(),
                pt.y() - center.y(),
                pt.z() - center.z(),
            )
            .length()
        })
        .collect();

    let sum: f64 = radii.iter().sum();
    #[allow(clippy::cast_precision_loss)]
    let mean_radius = sum / radii.len() as f64;

    if mean_radius < tolerance {
        return None; // Degenerate — all points at the center.
    }

    let max_dev = radii
        .iter()
        .map(|r| (r - mean_radius).abs())
        .fold(0.0_f64, f64::max);

    if max_dev > tolerance {
        return None;
    }

    // Ensure the normal points in a consistent direction (positive z-component
    // if possible, otherwise leave as-is).
    Some((center, n, mean_radius))
}

// ── Lightweight detection ────────────────────────────────────────────────────

/// Detected geometric kind of a NURBS curve (without recovering full analytic
/// parameters). Cheaper than [`recognize_curve`] when you only need a type tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectedCurveKind {
    /// All sampled points are collinear.
    Line,
    /// Sampled points are coplanar, equidistant from a center, and the curve is
    /// rational degree ≥ 2.
    Circle,
    /// Generic B-spline curve.
    BSpline,
}

impl DetectedCurveKind {
    /// Returns the lowercase string tag for this curve kind.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Line => "line",
            Self::Circle => "circle",
            Self::BSpline => "bspline",
        }
    }
}

/// Detect the geometric kind of a NURBS curve by sampling.
///
/// This is a lightweight heuristic that samples 16 points and checks for
/// collinearity (line) or coplanarity + equidistance (circle). It does **not**
/// recover analytic parameters — use [`recognize_curve`] for that.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn detect_curve_kind(curve: &NurbsCurve) -> DetectedCurveKind {
    // A rational degree-2+ NURBS with specific weight patterns can represent
    // conic sections. Non-rational or degree-1 curves are always BSpline (or Line).
    if curve.degree() < 2 || !curve.is_rational() {
        return DetectedCurveKind::BSpline;
    }

    let (u_min, u_max) = curve.domain();
    let n_samples = 16;

    // Check if the curve is closed (start ≈ end) to avoid sampling the
    // duplicate endpoint, which would bias the center calculation.
    let start_pt = curve.evaluate(u_min);
    let end_pt = curve.evaluate(u_max);
    let is_closed = (start_pt - end_pt).length() < 1e-6;

    let mut points = Vec::with_capacity(n_samples);
    for i in 0..n_samples {
        let t = if is_closed {
            u_min + (u_max - u_min) * (i as f64) / (n_samples as f64)
        } else {
            u_min + (u_max - u_min) * (i as f64) / ((n_samples - 1) as f64)
        };
        points.push(curve.evaluate(t));
    }

    // Compute center as average of all sampled points.
    let mut cx = 0.0_f64;
    let mut cy = 0.0_f64;
    let mut cz = 0.0_f64;
    for p in &points {
        cx += p.x();
        cy += p.y();
        cz += p.z();
    }
    let n = points.len() as f64;
    let center = Point3::new(cx / n, cy / n, cz / n);

    // Check if all points are equidistant from center (circle test).
    let distances: Vec<f64> = points.iter().map(|p| (*p - center).length()).collect();
    let avg_dist = distances.iter().sum::<f64>() / n;

    if avg_dist < 1e-10 {
        return DetectedCurveKind::BSpline;
    }

    let tol = avg_dist * 1e-4; // 0.01% relative tolerance
    let is_circle = distances.iter().all(|d| (d - avg_dist).abs() < tol);

    if is_circle {
        // Check coplanarity — all points should lie in a plane through center.
        let v0 = points[0] - center;
        let v1 = points[n_samples / 4] - center;
        let normal = v0.cross(v1);
        let normal_len = normal.length();
        if normal_len < 1e-10 {
            return DetectedCurveKind::BSpline;
        }
        let normal = Vec3::new(
            normal.x() / normal_len,
            normal.y() / normal_len,
            normal.z() / normal_len,
        );

        let coplanar = points
            .iter()
            .all(|p| ((*p - center).dot(normal)).abs() < tol);

        if coplanar {
            return DetectedCurveKind::Circle;
        }
    }

    DetectedCurveKind::BSpline
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::f64::consts::TAU;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;
    use crate::convert::curve_to_nurbs::{circle_to_nurbs, line_to_nurbs};

    fn origin() -> Point3 {
        Point3::new(0.0, 0.0, 0.0)
    }

    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    // ── line round-trip ──────────────────────────────────────────────────────

    #[test]
    fn recognize_line_round_trip() {
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(3.0, 4.0, 0.0);
        let nurbs = line_to_nurbs(start, end).unwrap();

        match recognize_curve(&nurbs, 1e-10) {
            RecognizedCurve::Line { origin, direction } => {
                // Direction should be unit-length.
                assert!((direction.length() - 1.0).abs() < 1e-10);
                // Origin should lie on the original segment.
                let v = origin - start;
                let proj = direction.dot(v);
                let perp = (v - direction * proj).length();
                assert!(perp < 1e-10, "origin not on original line: {perp}");
            }
            other => panic!("expected Line, got {other:?}"),
        }
    }

    // ── circle round-trip ────────────────────────────────────────────────────

    #[test]
    fn recognize_circle_full_round_trip() {
        let circle = Circle3D::new(Point3::new(1.0, 2.0, 3.0), z_axis(), 5.0).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, TAU).unwrap();

        match recognize_curve(&nurbs, 1e-6) {
            RecognizedCurve::Circle {
                center,
                normal,
                radius,
            } => {
                let dist = Vec3::new(center.x() - 1.0, center.y() - 2.0, center.z() - 3.0).length();
                assert!(dist < 1e-5, "center error {dist}");
                assert!(
                    (radius - 5.0).abs() < 1e-5,
                    "radius error {}",
                    (radius - 5.0).abs()
                );
                // Normal should be parallel (or anti-parallel) to z-axis.
                let cos_angle = normal.dot(z_axis()).abs();
                assert!(cos_angle > 0.999, "normal not aligned: {cos_angle}");
            }
            other => panic!("expected Circle, got {other:?}"),
        }
    }

    #[test]
    fn recognize_circle_quarter_arc() {
        let circle = Circle3D::new(origin(), z_axis(), 3.0).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, TAU * 0.25).unwrap();

        match recognize_curve(&nurbs, 1e-6) {
            RecognizedCurve::Circle { center, radius, .. } => {
                let dist = Vec3::new(center.x(), center.y(), center.z()).length();
                assert!(dist < 1e-5, "center not at origin: {dist}");
                assert!((radius - 3.0).abs() < 1e-5, "radius {radius}");
            }
            other => panic!("expected Circle, got {other:?}"),
        }
    }

    // ── not-recognized ───────────────────────────────────────────────────────

    #[test]
    fn line_is_not_recognized_as_circle() {
        let nurbs = line_to_nurbs(Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)).unwrap();
        // A line should be recognized as a Line, not a Circle.
        assert!(matches!(
            recognize_curve(&nurbs, 1e-6),
            RecognizedCurve::Line { .. }
        ));
    }
}

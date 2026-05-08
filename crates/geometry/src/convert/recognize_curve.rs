//! Recognize NURBS curves as elementary analytic forms.
//!
//! Samples a NURBS curve at 16 evenly-spaced parameter values and
//! tests whether the sample set is consistent with a line, circle,
//! ellipse, or hyperbola within the given tolerance. Returns a
//! [`RecognizedCurve`] describing the best match.

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
    /// Recognized as an ellipse (or elliptic arc, or full ellipse).
    Ellipse {
        /// Center of the ellipse.
        center: Point3,
        /// Ellipse normal (unit vector perpendicular to the ellipse plane).
        normal: Vec3,
        /// Direction of the semi-major axis (unit vector in the ellipse plane).
        u_axis: Vec3,
        /// Larger semi-axis length.
        semi_major: f64,
        /// Smaller semi-axis length.
        semi_minor: f64,
    },
    /// Recognized as a hyperbolic arc.
    Hyperbola {
        /// Center of the hyperbola.
        center: Point3,
        /// Hyperbola normal (unit vector perpendicular to the plane).
        normal: Vec3,
        /// Direction of the real (semi-major) axis.
        u_axis: Vec3,
        /// Real semi-axis length (distance from center to vertex).
        semi_major: f64,
        /// Imaginary semi-axis length.
        semi_minor: f64,
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
/// 3. **Ellipse**: all points coplanar and lie on a best-fit ellipse — max
///    residual against `(local_x/a)² + (local_y/b)² = 1` < `tolerance`. Tested
///    after circle so that a true circle is reported as `Circle`, not as
///    `Ellipse` with `a == b`.
/// 4. **Hyperbola**: all points coplanar and lie on a best-fit hyperbola —
///    max residual against `(local_x/a)² − (local_y/b)² = 1` < `tolerance`.
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
    if let Some((center, normal, u_axis, semi_major, semi_minor)) =
        try_recognize_ellipse(&samples, tolerance)
    {
        return RecognizedCurve::Ellipse {
            center,
            normal,
            u_axis,
            semi_major,
            semi_minor,
        };
    }
    if let Some((center, normal, u_axis, semi_major, semi_minor)) =
        try_recognize_hyperbola(&samples, tolerance)
    {
        return RecognizedCurve::Hyperbola {
            center,
            normal,
            u_axis,
            semi_major,
            semi_minor,
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

// ── Ellipse recognition ──────────────────────────────────────────────────────

/// Check if all sample points lie on an ellipse.
///
/// 1. Find a stable plane normal (same as `try_recognize_circle`).
/// 2. Require coplanarity within `tolerance`.
/// 3. Project to 2D using the plane's `(u_seed, v_seed)` basis.
/// 4. Solve the algebraic conic fit `A·x² + B·xy + C·y² + D·x + E·y = 1`
///    (5 unknowns, ≥5 samples) via least-squares normal equations.
/// 5. Recover canonical `(center, axes, semi_major, semi_minor)` from
///    the conic coefficients.
/// 6. Verify all samples satisfy `(local_x/a)² + (local_y/b)² = 1`
///    within `tolerance`.
///
/// This direct algebraic fit avoids the bias of using the sample
/// centroid as a center estimate — which is incorrect when samples
/// aren't uniformly-in-angle on the ellipse (e.g., NURBS-uniform
/// parameter sampling, which is what `recognize_curve` provides).
fn try_recognize_ellipse(
    samples: &[Point3],
    tolerance: f64,
) -> Option<(Point3, Vec3, Vec3, f64, f64)> {
    if samples.len() < 5 {
        return None;
    }

    // ── Step 1-2: plane + coplanarity ────────────────────────────────────────
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

    let d_plane = n.dot(Vec3::new(p0.x(), p0.y(), p0.z()));
    for pt in samples {
        let dist = n.dot(Vec3::new(pt.x(), pt.y(), pt.z())) - d_plane;
        if dist.abs() > tolerance {
            return None;
        }
    }

    // ── Step 3: project to 2D, shifting origin to the sample
    //    centroid. This centroid is NOT necessarily the ellipse
    //    center (sampling may be non-uniform-in-angle), but it
    //    moves all samples far from the 2D origin so the F=-1
    //    normalization in step 4 doesn't degenerate. (If we used
    //    p0 as the origin, p0 is on the ellipse so F=0 and the
    //    A·x² + … = 1 fit can't represent the conic.) ──────────────
    let u_seed = (samples[1] - p0).normalize().ok()?;
    let v_seed = n.cross(u_seed).normalize().ok()?;

    let raw_pts2d: Vec<(f64, f64)> = samples
        .iter()
        .map(|pt| {
            let v = *pt - p0;
            (u_seed.dot(v), v_seed.dot(v))
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n_f = raw_pts2d.len() as f64;
    let shift_x = raw_pts2d.iter().map(|p| p.0).sum::<f64>() / n_f;
    let shift_y = raw_pts2d.iter().map(|p| p.1).sum::<f64>() / n_f;
    let pts2d: Vec<(f64, f64)> = raw_pts2d
        .iter()
        .map(|&(x, y)| (x - shift_x, y - shift_y))
        .collect();

    // ── Step 4: algebraic conic fit (least-squares) ──────────────────────────
    // Solve A·x² + B·xy + C·y² + D·x + E·y = 1 via normal equations.
    // 5×5 symmetric system M·θ = b where θ = [A, B, C, D, E]ᵀ.
    let mut mat = [[0.0_f64; 5]; 5];
    let mut rhs = [0.0_f64; 5];
    for &(x, y) in &pts2d {
        let row = [x * x, x * y, y * y, x, y];
        for i in 0..5 {
            for j in 0..5 {
                mat[i][j] += row[i] * row[j];
            }
            rhs[i] += row[i];
        }
    }
    let theta = solve_5x5(&mat, &rhs)?;
    let (a_c, b_c, c_c, d_c, e_c) = (theta[0], theta[1], theta[2], theta[3], theta[4]);

    // For an ellipse: B² < 4AC (positive discriminant for ellipse).
    let disc = b_c * b_c - 4.0 * a_c * c_c;
    if disc >= -tolerance {
        return None; // Not an ellipse (could be parabola/hyperbola/degenerate).
    }

    // ── Step 5: recover canonical form ───────────────────────────────────────
    // Center: solve [2A B; B 2C] [cx; cy] = -[D; E].
    let m_det = 4.0 * a_c * c_c - b_c * b_c;
    if m_det.abs() < 1e-30 {
        return None;
    }
    let cx_2d = (-2.0 * c_c * d_c + b_c * e_c) / m_det;
    let cy_2d = (b_c * d_c - 2.0 * a_c * e_c) / m_det;

    // Translate to center: A·u² + B·uv + C·v² = K where
    // K = 1 - (A·cx² + B·cx·cy + C·cy² + D·cx + E·cy).
    let k = 1.0
        - (a_c * cx_2d * cx_2d
            + b_c * cx_2d * cy_2d
            + c_c * cy_2d * cy_2d
            + d_c * cx_2d
            + e_c * cy_2d);
    if k <= 0.0 {
        return None;
    }

    // Eigendecompose [A B/2; B/2 C] / K to get principal axes and
    // semi-axis lengths. λ_max = 1/semi_minor², λ_min = 1/semi_major².
    let aa = a_c / k;
    let bb = b_c / k;
    let cc = c_c / k;
    let trace_half = 0.5 * (aa + cc);
    let diff_half = 0.5 * (aa - cc);
    let radical = diff_half.hypot(0.5 * bb);
    let lambda1 = trace_half + radical;
    let lambda2 = trace_half - radical;
    if lambda1 <= 0.0 || lambda2 <= 0.0 {
        return None;
    }
    // Smaller eigenvalue → larger semi-axis (semi_major).
    let semi_major = 1.0 / lambda2.sqrt();
    let semi_minor = 1.0 / lambda1.sqrt();

    // Major axis = eigenvector for the SMALLER eigenvalue (since
    // λ_min = 1/semi_major²). The standard formula
    //   2θ = atan2(B, A − C)
    // gives the eigenvector for the LARGER eigenvalue (semi-minor
    // direction); we add π/2 to rotate to the major-axis direction.
    let theta_axis = 0.5 * bb.atan2(aa - cc) + std::f64::consts::FRAC_PI_2;
    let (sin_t, cos_t) = theta_axis.sin_cos();
    let u_local = (cos_t, sin_t);

    // ── Step 6: verify residual against the implicit equation ────────────────
    for &(x, y) in &pts2d {
        let dx = x - cx_2d;
        let dy = y - cy_2d;
        let lu = dx * u_local.0 + dy * u_local.1;
        let lv = -dx * u_local.1 + dy * u_local.0;
        let resid = (lu / semi_major).hypot(lv / semi_minor) - 1.0;
        if resid.abs() > tolerance {
            return None;
        }
    }

    // ── Convert center + axes back to 3D ─────────────────────────────────────
    // cx_2d/cy_2d are in the SHIFTED coords; add back the shift to
    // get coords in the (u_seed, v_seed) basis with origin p0.
    let center = p0 + u_seed * (cx_2d + shift_x) + v_seed * (cy_2d + shift_y);
    let u_axis_3d = (u_seed * u_local.0 + v_seed * u_local.1).normalize().ok()?;

    Some((center, n, u_axis_3d, semi_major, semi_minor))
}

// ── Hyperbola recognition ────────────────────────────────────────────────────

/// Check if all sample points lie on a hyperbolic arc.
///
/// Same algebraic conic fit as `try_recognize_ellipse`, but with
/// `B² − 4AC > 0` (hyperbolic discriminant) and a different canonical-
/// form recovery: after centering, the 2×2 quadratic form has one
/// positive and one negative eigenvalue. The semi-major (real) axis is
/// along the positive-eigenvalue eigenvector; semi-minor is the
/// imaginary axis along the negative-eigenvalue direction. Verification
/// uses the implicit equation `(local_x/a)² − (local_y/b)² = 1`.
fn try_recognize_hyperbola(
    samples: &[Point3],
    tolerance: f64,
) -> Option<(Point3, Vec3, Vec3, f64, f64)> {
    if samples.len() < 5 {
        return None;
    }

    // ── Step 1-2: plane + coplanarity (same as ellipse) ─────────────────────
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
    let d_plane = n.dot(Vec3::new(p0.x(), p0.y(), p0.z()));
    for pt in samples {
        let dist = n.dot(Vec3::new(pt.x(), pt.y(), pt.z())) - d_plane;
        if dist.abs() > tolerance {
            return None;
        }
    }

    // ── Step 3: project to 2D, shift origin to centroid ──────────────────────
    let u_seed = (samples[1] - p0).normalize().ok()?;
    let v_seed = n.cross(u_seed).normalize().ok()?;
    let raw_pts2d: Vec<(f64, f64)> = samples
        .iter()
        .map(|pt| {
            let v = *pt - p0;
            (u_seed.dot(v), v_seed.dot(v))
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let n_f = raw_pts2d.len() as f64;
    let shift_x = raw_pts2d.iter().map(|p| p.0).sum::<f64>() / n_f;
    let shift_y = raw_pts2d.iter().map(|p| p.1).sum::<f64>() / n_f;
    let pts2d: Vec<(f64, f64)> = raw_pts2d
        .iter()
        .map(|&(x, y)| (x - shift_x, y - shift_y))
        .collect();

    // ── Step 4: algebraic conic fit ──────────────────────────────────────────
    let mut mat = [[0.0_f64; 5]; 5];
    let mut rhs = [0.0_f64; 5];
    for &(x, y) in &pts2d {
        let row = [x * x, x * y, y * y, x, y];
        for i in 0..5 {
            for j in 0..5 {
                mat[i][j] += row[i] * row[j];
            }
            rhs[i] += row[i];
        }
    }
    let theta = solve_5x5(&mat, &rhs)?;
    let (a_c, b_c, c_c, d_c, e_c) = (theta[0], theta[1], theta[2], theta[3], theta[4]);

    // For a hyperbola: B² − 4AC > 0.
    let disc = b_c * b_c - 4.0 * a_c * c_c;
    if disc <= tolerance {
        return None;
    }

    // ── Step 5: recover center (same formula as ellipse) ─────────────────────
    let m_det = 4.0 * a_c * c_c - b_c * b_c;
    if m_det.abs() < 1e-30 {
        return None;
    }
    let cx_2d = (-2.0 * c_c * d_c + b_c * e_c) / m_det;
    let cy_2d = (b_c * d_c - 2.0 * a_c * e_c) / m_det;

    // K = 1 - (A·cx² + B·cx·cy + C·cy² + D·cx + E·cy).
    let k = 1.0
        - (a_c * cx_2d * cx_2d
            + b_c * cx_2d * cy_2d
            + c_c * cy_2d * cy_2d
            + d_c * cx_2d
            + e_c * cy_2d);
    if k.abs() < 1e-30 {
        return None;
    }

    // ── Step 6: eigendecompose [A B/2; B/2 C] / K. For a hyperbola
    //    the two eigenvalues have OPPOSITE signs. The positive
    //    eigenvalue corresponds to the real (semi-major) axis;
    //    negative corresponds to imaginary (semi-minor). ──────────────────────
    let aa = a_c / k;
    let bb = b_c / k;
    let cc = c_c / k;
    let trace_half = 0.5 * (aa + cc);
    let diff_half = 0.5 * (aa - cc);
    let radical = diff_half.hypot(0.5 * bb);
    let lambda_pos = trace_half + radical;
    let lambda_neg = trace_half - radical;
    if lambda_pos <= 0.0 || lambda_neg >= 0.0 {
        // Both eigenvalues same sign → not a hyperbola in this
        // normalization. Could happen if K has the wrong sign;
        // reject conservatively.
        return None;
    }
    let semi_major = 1.0 / lambda_pos.sqrt();
    let semi_minor = 1.0 / (-lambda_neg).sqrt();

    // Major-axis direction = eigenvector for the LARGER (positive)
    // eigenvalue. The standard formula 2θ = atan2(B, A−C) gives this
    // eigenvector directly (no π/2 rotation needed, unlike ellipse
    // where we wanted the SMALLER eigenvalue's direction).
    let theta_axis = 0.5 * bb.atan2(aa - cc);
    let (sin_t, cos_t) = theta_axis.sin_cos();
    let u_local = (cos_t, sin_t);

    // ── Step 7: verify residual against (lu/a)² − (lv/b)² = 1 ────────────────
    for &(x, y) in &pts2d {
        let dx = x - cx_2d;
        let dy = y - cy_2d;
        let lu = dx * u_local.0 + dy * u_local.1;
        let lv = -dx * u_local.1 + dy * u_local.0;
        let lhs = (lu / semi_major).powi(2) - (lv / semi_minor).powi(2);
        if (lhs - 1.0).abs() > tolerance {
            return None;
        }
    }

    let center = p0 + u_seed * (cx_2d + shift_x) + v_seed * (cy_2d + shift_y);
    let u_axis_3d = (u_seed * u_local.0 + v_seed * u_local.1).normalize().ok()?;

    Some((center, n, u_axis_3d, semi_major, semi_minor))
}

/// Solve a 5×5 linear system via Gaussian elimination with partial
/// pivoting. Returns `None` if the matrix is singular.
fn solve_5x5(mat: &[[f64; 5]; 5], rhs: &[f64; 5]) -> Option<[f64; 5]> {
    let mut m = *mat;
    let mut b = *rhs;
    for i in 0..5 {
        // Partial pivot.
        let mut max_row = i;
        let mut max_val = m[i][i].abs();
        for k in (i + 1)..5 {
            if m[k][i].abs() > max_val {
                max_val = m[k][i].abs();
                max_row = k;
            }
        }
        if max_val < 1e-30 {
            return None;
        }
        if max_row != i {
            m.swap(i, max_row);
            b.swap(i, max_row);
        }
        // Eliminate below.
        for k in (i + 1)..5 {
            let factor = m[k][i] / m[i][i];
            for j in i..5 {
                m[k][j] -= factor * m[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    // Back substitute.
    let mut x = [0.0_f64; 5];
    for i in (0..5).rev() {
        let mut sum = b[i];
        for j in (i + 1)..5 {
            sum -= m[i][j] * x[j];
        }
        x[i] = sum / m[i][i];
    }
    Some(x)
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
    // Degree-1 non-rational curves are lines by definition.
    if curve.degree() < 2 && !curve.is_rational() {
        return DetectedCurveKind::Line;
    }

    // Non-rational degree-2+ curves cannot represent conics.
    if !curve.is_rational() {
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

    use brepkit_math::curves::{Circle3D, Ellipse3D, Hyperbola3D};
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};

    use super::*;
    use crate::convert::curve_to_nurbs::{circle_to_nurbs, ellipse_to_nurbs, line_to_nurbs};

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

    // ── ellipse round-trip ───────────────────────────────────────────────────

    #[test]
    fn recognize_full_ellipse_round_trip() {
        // Build a NURBS for a full ellipse, recognize it, and verify
        // semi-axes / center / normal match within tolerance.
        let center = Point3::new(2.0, -1.0, 5.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let a = 3.0_f64;
        let b = 1.5_f64;
        let ellipse = Ellipse3D::new(center, normal, a, b).unwrap();
        let nurbs = ellipse_to_nurbs(&ellipse, 0.0, TAU).unwrap();

        match recognize_curve(&nurbs, 1e-5) {
            RecognizedCurve::Ellipse {
                center: c,
                normal: n,
                semi_major,
                semi_minor,
                ..
            } => {
                assert!(
                    (c - center).length() < 1e-4,
                    "center mismatch: {c:?} vs {center:?}"
                );
                assert!(
                    (n.dot(normal).abs() - 1.0).abs() < 1e-6,
                    "normal mismatch (cos angle {})",
                    n.dot(normal)
                );
                assert!(
                    (semi_major - a).abs() < 1e-4,
                    "semi_major {semi_major} vs {a}"
                );
                assert!(
                    (semi_minor - b).abs() < 1e-4,
                    "semi_minor {semi_minor} vs {b}"
                );
            }
            other => panic!("expected Ellipse, got {other:?}"),
        }
    }

    #[test]
    fn circle_is_recognized_as_circle_not_ellipse() {
        // True circles should match Circle (which is tested first), not
        // be downgraded to an Ellipse with a == b.
        let circle = Circle3D::new(origin(), z_axis(), 2.5).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, TAU).unwrap();
        assert!(matches!(
            recognize_curve(&nurbs, 1e-6),
            RecognizedCurve::Circle { .. }
        ));
    }

    // ── hyperbola round-trip ──────────────────────────────────────────────────

    /// Build a rational degree-2 NURBS for a hyperbolic arc, mirroring
    /// heal's `hyperbola_to_nurbs` (geometry/curve_to_nurbs doesn't
    /// have one yet).
    fn hyperbola_to_nurbs_inline(hyp: &Hyperbola3D, t_min: f64, t_max: f64) -> NurbsCurve {
        let center = hyp.center();
        let u = hyp.u_axis();
        let v = hyp.v_axis();
        let a = hyp.semi_major();
        let b = hyp.semi_minor();

        let p0 = hyp.evaluate(t_min);
        let p2 = hyp.evaluate(t_max);
        let half = 0.5 * (t_max - t_min);
        let tanh_b = half.tanh();
        let p1_x = a * (t_min.cosh() + tanh_b * t_min.sinh());
        let p1_y = b * (t_min.sinh() + tanh_b * t_min.cosh());
        let p1 = center + u * p1_x + v * p1_y;
        let w1 = half.cosh();

        NurbsCurve::new(
            2,
            vec![t_min, t_min, t_min, t_max, t_max, t_max],
            vec![p0, p1, p2],
            vec![1.0, w1, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn recognize_hyperbola_round_trip() {
        let center = Point3::new(2.0, -1.0, 5.0);
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let a = 3.0_f64;
        let b = 1.5_f64;
        let hyp = Hyperbola3D::new(center, normal, a, b).unwrap();
        let nurbs = hyperbola_to_nurbs_inline(&hyp, -1.5, 1.5);

        match recognize_curve(&nurbs, 1e-5) {
            RecognizedCurve::Hyperbola {
                center: c,
                normal: n,
                semi_major,
                semi_minor,
                ..
            } => {
                assert!(
                    (c - center).length() < 1e-4,
                    "center mismatch: {c:?} vs {center:?}"
                );
                assert!(
                    (n.dot(normal).abs() - 1.0).abs() < 1e-6,
                    "normal mismatch (cos angle {})",
                    n.dot(normal)
                );
                assert!(
                    (semi_major - a).abs() < 1e-4,
                    "semi_major {semi_major} vs {a}"
                );
                assert!(
                    (semi_minor - b).abs() < 1e-4,
                    "semi_minor {semi_minor} vs {b}"
                );
            }
            other => panic!("expected Hyperbola, got {other:?}"),
        }
    }

    #[test]
    fn ellipse_is_not_recognized_as_hyperbola() {
        // Ellipse must hit the Ellipse path (B²−4AC < 0), not fall
        // through to Hyperbola (B²−4AC > 0).
        let ellipse = Ellipse3D::new(origin(), z_axis(), 3.0, 1.5).unwrap();
        let nurbs = ellipse_to_nurbs(&ellipse, 0.0, TAU).unwrap();
        assert!(matches!(
            recognize_curve(&nurbs, 1e-6),
            RecognizedCurve::Ellipse { .. }
        ));
    }
}

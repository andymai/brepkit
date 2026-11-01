//! Bezier decomposition and degree elevation for NURBS curves and surfaces.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::knot_ops::{surface_knot_insert_u, surface_knot_insert_v};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

/// Decompose a NURBS curve into Bezier segments.
///
/// Each segment is a NURBS curve of the same degree with a clamped knot vector
/// over one knot span of the curve's domain, in order: segment *i* runs from
/// the domain's *i*-th distinct knot to the next. Each is read off its span's
/// de Boor points ([`curve_bezier_between`]) with no knot inserted, so a knot
/// vector that is not clamped, a degree of zero, or an interior knot repeated
/// past the degree needs no special case.
///
/// # Errors
///
/// Returns an error if a segment cannot be built.
pub fn curve_to_bezier_segments(curve: &NurbsCurve) -> Result<Vec<NurbsCurve>, MathError> {
    let p = curve.degree();
    let (d0, d1) = curve.domain();
    let mut cuts: Vec<f64> = std::iter::once(d0)
        .chain(curve.knots().iter().copied().filter(|&k| k > d0 && k < d1))
        .chain(std::iter::once(d1))
        .collect();
    // By value: -0 and 0 are one knot.
    cuts.dedup_by(|a, b| (*a - *b).abs() <= 0.0);
    if cuts.len() < 2 {
        return Ok(vec![curve.clone()]);
    }
    let mut segments = Vec::with_capacity(cuts.len() - 1);
    for w in cuts.windows(2) {
        let homogeneous = curve_bezier_between(curve, w[0], w[1])?;
        let (points, weights): (Vec<Point3>, Vec<f64>) =
            homogeneous.iter().map(|q| (cartesian(*q), q[3])).unzip();
        let mut knots = vec![w[0]; p + 1];
        knots.extend(std::iter::repeat_n(w[1], p + 1));
        segments.push(NurbsCurve::new(p, knots, points, weights)?);
    }
    Ok(segments)
}

/// A homogeneous point `[w x, w y, w z, w]` as its point, or its first three
/// coordinates for a weight of zero, as knot insertion reads one.
fn cartesian(q: [f64; 4]) -> Point3 {
    if q[3] == 0.0 {
        Point3::new(q[0], q[1], q[2])
    } else {
        Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])
    }
}

/// The rational Bezier form of `curve` from parameter `a` to parameter `b`.
///
/// `a` and `b` lie in one knot span of the domain. The result is `degree + 1`
/// control points in homogeneous form `[w x, w y, w z, w]`, the first at `a`
/// and the last at `b`, so in reverse when `a > b`.
///
/// Control point `j` is the curve's blossom at `a` taken `degree - j` times
/// and `b` taken `j` times, read off the span's de Boor points. No knot is
/// inserted, so a piece of a span and an unclamped knot vector need no
/// special case. When every returned weight is positive the piece lies in
/// the convex hull of the points.
///
/// # Errors
///
/// Returns [`MathError::ParameterOutOfRange`] if `a` and `b` do not lie in
/// one knot span of the curve's domain.
pub fn curve_bezier_between(
    curve: &NurbsCurve,
    a: f64,
    b: f64,
) -> Result<Vec<[f64; 4]>, MathError> {
    let p = curve.degree();
    let knots = curve.knots();
    let n = curve.control_points().len();
    let (lo, hi) = (a.min(b), a.max(b));
    let (d0, d1) = curve.domain();
    let out_of_range = MathError::ParameterOutOfRange {
        value: if lo < d0 { lo } else { hi },
        min: d0,
        max: d1,
    };
    if n <= p {
        return Err(out_of_range);
    }
    let i = p + knots[p..n].partition_point(|&k| k <= lo).saturating_sub(1);
    if !(knots[i] <= lo && hi <= knots[i + 1] && knots[i] < knots[i + 1]) {
        return Err(out_of_range);
    }
    let homogeneous: Vec<[f64; 4]> = curve.control_points()[i - p..=i]
        .iter()
        .zip(&curve.weights()[i - p..=i])
        .map(|(pt, &w)| [pt.x() * w, pt.y() * w, pt.z() * w, w])
        .collect();
    let blossom = |at_b: usize| {
        let mut d = homogeneous.clone();
        for level in 1..=p {
            let u = if level <= at_b { b } else { a };
            for j in (level..=p).rev() {
                let (left, right) = (knots[i - p + j], knots[i + 1 + j - level]);
                let alpha = (u - left) / (right - left);
                let prev = d[j - 1];
                for (c, q) in d[j].iter_mut().zip(prev) {
                    *c = (1.0 - alpha).mul_add(q, alpha * *c);
                }
            }
        }
        d[p]
    };
    Ok((0..=p).map(blossom).collect())
}

/// A single Bezier patch extracted from a NURBS surface decomposition.
///
/// Each patch has `(degree_u + 1) × (degree_v + 1)` control points and
/// corresponds to a rectangular region in the original parameter space.
#[derive(Debug, Clone)]
pub struct BezierPatch {
    /// The Bezier patch as a NURBS surface (clamped knots, single span).
    pub surface: NurbsSurface,
    /// Parameter range in u on the original surface: `(u_start, u_end)`.
    pub u_range: (f64, f64),
    /// Parameter range in v on the original surface: `(v_start, v_end)`.
    pub v_range: (f64, f64),
}

impl BezierPatch {
    /// Compute the axis-aligned bounding box from control point extrema.
    ///
    /// For a Bezier patch (single-span), the convex hull property guarantees
    /// the geometry lies entirely within the control point AABB.
    #[must_use]
    pub fn aabb(&self) -> Aabb3 {
        self.surface.aabb()
    }

    /// Compute the diagonal length of the control-point AABB.
    #[must_use]
    pub fn diagonal(&self) -> f64 {
        let aabb = self.aabb();
        let dx = aabb.max.x() - aabb.min.x();
        let dy = aabb.max.y() - aabb.min.y();
        let dz = aabb.max.z() - aabb.min.z();
        dx.mul_add(dx, dy.mul_add(dy, dz * dz)).sqrt()
    }

    /// Compute the midpoint in the u parameter range.
    #[must_use]
    pub fn u_mid(&self) -> f64 {
        (self.u_range.0 + self.u_range.1) * 0.5
    }

    /// Compute the midpoint in the v parameter range.
    #[must_use]
    pub fn v_mid(&self) -> f64 {
        (self.v_range.0 + self.v_range.1) * 0.5
    }
}

/// Decompose a NURBS surface into Bezier patches.
///
/// Inserts knots in both u and v directions until all interior knots reach
/// full multiplicity (equal to degree), then extracts individual patches.
/// Each patch has `(degree_u + 1) × (degree_v + 1)` control points.
///
/// # Errors
///
/// Returns an error if knot insertion fails or the surface is degenerate.
#[allow(clippy::too_many_lines)]
pub fn surface_to_bezier_patches(surface: &NurbsSurface) -> Result<Vec<BezierPatch>, MathError> {
    let pu = surface.degree_u();
    let pv = surface.degree_v();

    // Step 1: Insert u-knots to full multiplicity.
    let mut refined = surface.clone();
    {
        let knots_u = refined.knots_u().to_vec();
        let mut i = pu + 1;
        while i < knots_u.len() - pu - 1 {
            let u = knots_u[i];
            let mut mult = 1;
            while i + mult < knots_u.len() - pu - 1 && (knots_u[i + mult] - u).abs() < 1e-15 {
                mult += 1;
            }
            let deficit = pu.saturating_sub(mult);
            if deficit > 0 {
                refined = surface_knot_insert_u(&refined, u, deficit)?;
            }
            i += mult;
        }
    }

    // Step 2: Insert v-knots to full multiplicity.
    {
        let knots_v = refined.knots_v().to_vec();
        let mut i = pv + 1;
        while i < knots_v.len() - pv - 1 {
            let v = knots_v[i];
            let mut mult = 1;
            while i + mult < knots_v.len() - pv - 1 && (knots_v[i + mult] - v).abs() < 1e-15 {
                mult += 1;
            }
            let deficit = pv.saturating_sub(mult);
            if deficit > 0 {
                refined = surface_knot_insert_v(&refined, v, deficit)?;
            }
            i += mult;
        }
    }

    // Step 3: Extract unique knot spans in u and v.
    let ref_knots_u = refined.knots_u();
    let ref_knots_v = refined.knots_v();
    let ref_cps = refined.control_points();
    let ref_ws = refined.weights();

    let n_rows = ref_cps.len();
    let n_cols = if n_rows > 0 { ref_cps[0].len() } else { 0 };

    let n_patches_u = if n_rows > pu { (n_rows - 1) / pu } else { 0 };
    let n_patches_v = if n_cols > pv { (n_cols - 1) / pv } else { 0 };

    if n_patches_u == 0 || n_patches_v == 0 {
        // Single patch: return the whole surface.
        let (u_start, u_end) = refined.domain_u();
        let (v_start, v_end) = refined.domain_v();
        return Ok(vec![BezierPatch {
            surface: refined,
            u_range: (u_start, u_end),
            v_range: (v_start, v_end),
        }]);
    }

    let mut patches = Vec::with_capacity(n_patches_u * n_patches_v);

    for iu in 0..n_patches_u {
        let row_start = iu * pu;
        let row_end = row_start + pu + 1;
        if row_end > n_rows {
            break;
        }
        // Knot values for this u-span
        let u_start = ref_knots_u[row_start + pu];
        let u_end = ref_knots_u[row_end];

        for iv in 0..n_patches_v {
            let col_start = iv * pv;
            let col_end = col_start + pv + 1;
            if col_end > n_cols {
                break;
            }
            let v_start = ref_knots_v[col_start + pv];
            let v_end = ref_knots_v[col_end];

            // Extract (pu+1) × (pv+1) control point sub-grid.
            let mut patch_cps = Vec::with_capacity(pu + 1);
            let mut patch_ws = Vec::with_capacity(pu + 1);

            for row in row_start..row_end {
                patch_cps.push(ref_cps[row][col_start..col_end].to_vec());
                patch_ws.push(ref_ws[row][col_start..col_end].to_vec());
            }

            // Build Bezier knot vectors: (p+1) copies of start, (p+1) copies of end.
            let mut knots_u = vec![u_start; pu + 1];
            knots_u.extend(std::iter::repeat_n(u_end, pu + 1));

            let mut knots_v = vec![v_start; pv + 1];
            knots_v.extend(std::iter::repeat_n(v_end, pv + 1));

            let patch_surface = NurbsSurface::new(pu, pv, knots_u, knots_v, patch_cps, patch_ws)?;

            patches.push(BezierPatch {
                surface: patch_surface,
                u_range: (u_start, u_end),
                v_range: (v_start, v_end),
            });
        }
    }

    Ok(patches)
}

/// Elevate the degree of a NURBS curve by `t` (A5.9).
///
/// The resulting curve has degree `p + t` and represents the same geometry.
///
/// # Errors
///
/// Returns an error if the resulting curve cannot be constructed.
pub fn curve_degree_elevate(curve: &NurbsCurve, t: usize) -> Result<NurbsCurve, MathError> {
    if t == 0 {
        return Ok(curve.clone());
    }

    let p = curve.degree();
    let new_p = p + t;

    let segments = curve_to_bezier_segments(curve)?;
    let n_segs = segments.len();

    let bezalfs = compute_bezalfs(p, t);

    let mut all_cps: Vec<[f64; 4]> = Vec::new();
    let mut all_knots: Vec<f64> = Vec::new();

    // Whether each piece ends where the next starts, read off the knot's
    // repeats: the curve is continuous at a knot repeated at most its degree
    // times. One repeated more (or any knot of a degree-zero curve) may jump
    // there, so both ends stay and the elevated knot repeats once more.
    // Comparing the ends' coordinates instead would hang on the weights'
    // scale, which leaves the curve unchanged.
    let joined: Vec<bool> = segments
        .iter()
        .take(n_segs.saturating_sub(1))
        .map(|seg| {
            let u = seg.knots()[p + 1];
            curve
                .knots()
                .iter()
                .filter(|&&k| (k - u).abs() <= 0.0)
                .count()
                <= p
        })
        .collect();

    for (seg_idx, seg) in segments.iter().enumerate() {
        let seg_cps = seg.control_points();
        let seg_ws = seg.weights();
        let seg_pw: Vec<[f64; 4]> = seg_cps
            .iter()
            .zip(seg_ws.iter())
            .map(|(pt, &w)| [pt.x() * w, pt.y() * w, pt.z() * w, w])
            .collect();

        let mut elevated = vec![[0.0; 4]; new_p + 1];
        for i in 0..=new_p {
            for j in i.saturating_sub(t)..=(i.min(p)) {
                if j <= p {
                    let coeff = bezalfs[i][j];
                    elevated[i][0] += coeff * seg_pw[j][0];
                    elevated[i][1] += coeff * seg_pw[j][1];
                    elevated[i][2] += coeff * seg_pw[j][2];
                    elevated[i][3] += coeff * seg_pw[j][3];
                }
            }
        }

        if seg_idx == 0 {
            // First segment: add all control points and start knot.
            let u_start = seg.knots()[0];
            for _ in 0..=new_p {
                all_knots.push(u_start);
            }
            all_cps.extend_from_slice(&elevated);
        } else if joined[seg_idx - 1] {
            // Subsequent segments: skip first control point (shared with previous).
            all_cps.extend_from_slice(&elevated[1..]);
        } else {
            all_cps.extend_from_slice(&elevated);
        }

        if seg_idx == n_segs - 1 {
            // Last segment: add end knot.
            let u_end = seg.knots()[p + 1];
            for _ in 0..=new_p {
                all_knots.push(u_end);
            }
        } else {
            // Interior break: new_p copies of the break knot, one more
            // where the curve jumps.
            let u_break = seg.knots()[p + 1];
            let copies = if joined[seg_idx] { new_p } else { new_p + 1 };
            for _ in 0..copies {
                all_knots.push(u_break);
            }
        }
    }

    // Handle single-segment case (no segments produced)
    if segments.is_empty() {
        // Should not happen if curve is valid, but handle gracefully.
        return Ok(curve.clone());
    }

    // Convert back from homogeneous.
    let new_cps: Vec<Point3> = all_cps
        .iter()
        .map(|h| {
            if h[3] == 0.0 {
                Point3::new(h[0], h[1], h[2])
            } else {
                Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
            }
        })
        .collect();
    let new_ws: Vec<f64> = all_cps.iter().map(|h| h[3]).collect();

    NurbsCurve::new(new_p, all_knots, new_cps, new_ws)
}

/// Compute the Bezier degree elevation coefficients.
///
/// `bezalfs[i][j] = C(p,j) * C(t, i-j) / C(p+t, i)`
#[allow(clippy::cast_precision_loss)]
fn compute_bezalfs(p: usize, t: usize) -> Vec<Vec<f64>> {
    let new_p = p + t;
    let mut bezalfs = vec![vec![0.0; p + 1]; new_p + 1];

    for (i, row) in bezalfs.iter_mut().enumerate().take(new_p + 1) {
        let j_min = i.saturating_sub(t);
        let j_max = i.min(p);
        for (j, val) in row.iter_mut().enumerate().take(j_max + 1).skip(j_min) {
            *val = binomial(p, j) as f64 * binomial(t, i - j) as f64 / binomial(new_p, i) as f64;
        }
    }

    bezalfs
}

use super::basis::binomial;

#[cfg(test)]
#[allow(clippy::expect_used, clippy::cast_lossless, clippy::suboptimal_flops)]
mod tests {
    use super::*;

    fn cubic_bezier() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid")
    }

    fn multi_span_curve() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(2.0, 2.0, 0.0),
                Point3::new(3.0, 1.0, 0.0),
                Point3::new(4.0, 0.0, 0.0),
            ],
            vec![1.0, 1.0, 1.0, 1.0, 1.0],
        )
        .expect("valid")
    }

    #[test]
    fn bezier_decompose_single_span() {
        let c = cubic_bezier();
        let segs = curve_to_bezier_segments(&c).expect("valid");
        assert_eq!(segs.len(), 1);
        // The single segment should match the original curve.
        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = segs[0].evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-12 && (p1.y() - p2.y()).abs() < 1e-12,
                "mismatch at u={u}"
            );
        }
    }

    #[test]
    fn bezier_decompose_multi_span() {
        let c = multi_span_curve();
        let segs = curve_to_bezier_segments(&c).expect("valid");
        assert!(segs.len() >= 2, "expected at least 2 segments");

        // Each segment should be degree 3 with 4 control points.
        for seg in &segs {
            assert_eq!(seg.degree(), 3);
            assert_eq!(seg.control_points().len(), 4);
        }
    }

    #[test]
    fn degree_elevate_preserves_shape() {
        let c = cubic_bezier();
        let elevated = curve_degree_elevate(&c, 1).expect("valid");
        assert_eq!(elevated.degree(), 4);

        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = elevated.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-10 && (p1.y() - p2.y()).abs() < 1e-10,
                "mismatch at u={u}: ({},{}) vs ({},{})",
                p1.x(),
                p1.y(),
                p2.x(),
                p2.y()
            );
        }
    }

    #[test]
    fn degree_elevate_by_two() {
        let c = cubic_bezier();
        let elevated = curve_degree_elevate(&c, 2).expect("valid");
        assert_eq!(elevated.degree(), 5);

        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = elevated.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-9 && (p1.y() - p2.y()).abs() < 1e-9,
                "mismatch at u={u}"
            );
        }
    }

    // ── Surface Bezier decomposition tests ────────────────

    fn bilinear_surface() -> NurbsSurface {
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid")
    }

    fn quadratic_surface_multi_span() -> NurbsSurface {
        // Degree 2×2 with an interior knot at u=0.5 and v=0.5
        NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![
                vec![
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(0.25, 0.0, 0.1),
                    Point3::new(0.5, 0.0, 0.0),
                    Point3::new(1.0, 0.0, 0.0),
                ],
                vec![
                    Point3::new(0.0, 0.25, 0.1),
                    Point3::new(0.25, 0.25, 0.5),
                    Point3::new(0.5, 0.25, 0.1),
                    Point3::new(1.0, 0.25, 0.1),
                ],
                vec![
                    Point3::new(0.0, 0.5, 0.0),
                    Point3::new(0.25, 0.5, 0.1),
                    Point3::new(0.5, 0.5, 0.0),
                    Point3::new(1.0, 0.5, 0.0),
                ],
                vec![
                    Point3::new(0.0, 1.0, 0.0),
                    Point3::new(0.25, 1.0, 0.0),
                    Point3::new(0.5, 1.0, 0.0),
                    Point3::new(1.0, 1.0, 0.0),
                ],
            ],
            vec![vec![1.0; 4]; 4],
        )
        .expect("valid")
    }

    #[test]
    fn bezier_patches_single_span() {
        let surf = bilinear_surface();
        let patches = surface_to_bezier_patches(&surf).expect("valid");
        assert_eq!(patches.len(), 1, "single-span surface = 1 patch");

        // Patch should evaluate identically to original
        for i in 0..=4 {
            for j in 0..=4 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let p1 = surf.evaluate(u, v);
                let p2 = patches[0].surface.evaluate(u, v);
                assert!(
                    (p1.x() - p2.x()).abs() < 1e-10
                        && (p1.y() - p2.y()).abs() < 1e-10
                        && (p1.z() - p2.z()).abs() < 1e-10,
                    "mismatch at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn bezier_patches_cover_surface() {
        let surf = quadratic_surface_multi_span();
        let patches = surface_to_bezier_patches(&surf).expect("valid");

        // With interior knots at u=0.5 and v=0.5, should produce 2×2 = 4 patches
        assert!(
            patches.len() >= 4,
            "expected at least 4 patches, got {}",
            patches.len()
        );

        // Each patch should be degree 2×2 with 3×3 = 9 control points
        for patch in &patches {
            assert_eq!(patch.surface.degree_u(), 2);
            assert_eq!(patch.surface.degree_v(), 2);
            assert_eq!(patch.surface.control_points().len(), 3);
            assert_eq!(patch.surface.control_points()[0].len(), 3);
        }

        // Evaluate at patch midpoints and verify they match original surface
        for patch in &patches {
            let u_mid = (patch.u_range.0 + patch.u_range.1) * 0.5;
            let v_mid = (patch.v_range.0 + patch.v_range.1) * 0.5;

            let p_orig = surf.evaluate(u_mid, v_mid);
            // Map to local patch parameter
            let patch_u = (u_mid - patch.u_range.0) / (patch.u_range.1 - patch.u_range.0);
            let patch_v = (v_mid - patch.v_range.0) / (patch.v_range.1 - patch.v_range.0);
            // Bezier patch uses its own knot domain
            let pu = patch.surface.domain_u();
            let pv = patch.surface.domain_v();
            let local_u = pu.0 + patch_u * (pu.1 - pu.0);
            let local_v = pv.0 + patch_v * (pv.1 - pv.0);
            let p_patch = patch.surface.evaluate(local_u, local_v);

            assert!(
                (p_orig.x() - p_patch.x()).abs() < 1e-8
                    && (p_orig.y() - p_patch.y()).abs() < 1e-8
                    && (p_orig.z() - p_patch.z()).abs() < 1e-8,
                "patch mismatch at ({u_mid}, {v_mid}): orig=({},{},{}), patch=({},{},{})",
                p_orig.x(),
                p_orig.y(),
                p_orig.z(),
                p_patch.x(),
                p_patch.y(),
                p_patch.z()
            );
        }
    }

    #[test]
    fn bezier_patch_aabb_and_diagonal() {
        let surf = bilinear_surface();
        let patches = surface_to_bezier_patches(&surf).expect("valid");
        let patch = &patches[0];
        let diag = patch.diagonal();
        // Unit square on XY plane: diagonal ≈ sqrt(2)
        assert!(
            (diag - std::f64::consts::SQRT_2).abs() < 0.1,
            "expected diagonal ≈ √2, got {diag}"
        );
    }

    /// The point at `s` along homogeneous Bezier control points `q`.
    fn bezier_at(q: &[[f64; 4]], s: f64) -> Point3 {
        let mut d = q.to_vec();
        for r in 1..d.len() {
            for j in 0..d.len() - r {
                for c in 0..4 {
                    d[j][c] = (1.0 - s) * d[j][c] + s * d[j + 1][c];
                }
            }
        }
        let [x, y, z, w] = d[0];
        Point3::new(x / w, y / w, z / w)
    }

    /// A rational cubic with an unclamped knot vector: domain `[0.2, 1.0]`,
    /// spans `[0.2, 0.5]`, `[0.5, 0.7]`, `[0.7, 1.0]`.
    fn unclamped_rational_cubic() -> NurbsCurve {
        NurbsCurve::new(
            3,
            vec![-0.3, -0.1, 0.0, 0.2, 0.5, 0.7, 1.0, 1.2, 1.4, 1.5],
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.5),
                Point3::new(2.0, -1.0, 1.0),
                Point3::new(3.0, 2.0, 0.0),
                Point3::new(4.0, 0.5, -1.0),
                Point3::new(5.0, 1.0, 0.0),
            ],
            vec![1.0, 0.7, 1.3, 0.9, 1.1, 0.8],
        )
        .expect("valid")
    }

    /// The Bezier form between two parameters traces the curve between them,
    /// in either direction, on part of a span, a whole span, and the span
    /// ending the domain.
    #[test]
    fn bezier_between_traces_the_curve_between_its_parameters() {
        let curve = unclamped_rational_cubic();
        for (a, b) in [
            (0.25, 0.45),
            (0.45, 0.25),
            (0.5, 0.7),
            (0.7, 1.0),
            (1.0, 0.7),
        ] {
            let q = curve_bezier_between(&curve, a, b).expect("one span");
            assert_eq!(q.len(), 4);
            for k in 0..=8 {
                let s = f64::from(k) / 8.0;
                let on_curve = curve.evaluate(a + s * (b - a));
                let gap = (bezier_at(&q, s) - on_curve).length();
                assert!(gap < 1e-12, "({a}, {b}) at {s}: {gap}");
            }
        }
    }

    /// Parameters across a knot or outside the domain have no single span.
    #[test]
    fn bezier_between_refuses_more_than_one_span() {
        let curve = unclamped_rational_cubic();
        assert!(curve_bezier_between(&curve, 0.4, 0.6).is_err());
        assert!(curve_bezier_between(&curve, 0.1, 0.3).is_err());
        assert!(curve_bezier_between(&curve, 0.9, 1.1).is_err());
    }

    /// A curve's Bezier pieces trace it: each at 7 points inside its span
    /// holds the curve to 1e-12 there (a curve of degree zero jumps at its
    /// knots).
    fn pieces_trace(c: &NurbsCurve, spans: usize) {
        let segs = curve_to_bezier_segments(c).expect("valid");
        assert_eq!(segs.len(), spans);
        for seg in &segs {
            let (a, b) = seg.domain();
            for k in 1..8 {
                let t = (b - a).mul_add(f64::from(k) / 8.0, a);
                let off = (seg.evaluate(t) - c.evaluate(t)).length();
                assert!(off < 1e-12, "at {t}: {off}");
            }
        }
    }

    /// A degree-1 side on knots 0, 0, 0.5, 0.5, 1, 1 through its ends and its
    /// middle twice, an interior knot repeated past the degree: two pieces.
    #[test]
    fn a_knot_repeated_past_the_degree_cuts_two_pieces() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            vec![
                p(1.0, 1.0, 0.0),
                p(0.0, 1.0, 0.0),
                p(0.0, 1.0, 0.0),
                p(-1.0, 1.0, 0.0),
            ],
            vec![1.0; 4],
        )
        .expect("valid");
        pieces_trace(&c, 2);
    }

    /// A degree-0 curve on knots 0, 0.5, 1: two constant pieces.
    #[test]
    fn a_degree_zero_curve_cuts_into_its_spans() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            0,
            vec![0.0, 0.5, 1.0],
            vec![p(0.0, 0.0, 0.0), p(1.0, 0.0, 0.0)],
            vec![1.0; 2],
        )
        .expect("valid");
        pieces_trace(&c, 2);
    }

    /// A cubic on the uniform knots 0 to 7, not clamped: the pieces of its
    /// domain, 3 to 4, cut at its knots.
    #[test]
    fn a_curve_not_clamped_cuts_into_its_domains_spans() {
        let c = NurbsCurve::new(
            3,
            (0..8).map(f64::from).collect(),
            vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 2.0, 0.0),
                Point3::new(3.0, -1.0, 1.0),
                Point3::new(4.0, 1.0, 0.0),
            ],
            vec![1.0, 2.0, 0.5, 1.0],
        )
        .expect("valid");
        pieces_trace(&c, 1);
    }

    /// A quadratic weighted 1, 0, 1: its piece evaluates as it does, the
    /// zero-weight pole kept rather than divided by.
    #[test]
    fn a_zero_weight_pole_cuts_a_piece_that_traces_the_curve() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![p(0.0, 0.0, 0.0), p(1.0, 1.0, 0.0), p(2.0, 0.0, 0.0)],
            vec![1.0, 0.0, 1.0],
        )
        .expect("valid");
        pieces_trace(&c, 1);
    }

    /// A quadratic on knots -1, -1, -1, -0, 0, 1, 1, 1: -0 and 0 are one
    /// knot, so two pieces and none of no width.
    #[test]
    fn a_signed_zero_knot_cuts_no_piece_of_no_width() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            2,
            vec![-1.0, -1.0, -1.0, -0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(1.0, 1.0, 0.0),
                p(2.0, 0.0, 0.0),
                p(3.0, 1.0, 0.0),
                p(4.0, 0.0, 0.0),
            ],
            vec![1.0; 5],
        )
        .expect("valid");
        pieces_trace(&c, 2);
    }

    /// A degree-1 curve that jumps at 0.5, its knot repeated past its
    /// degree: raised to degree 3, it still jumps there, and traces the
    /// curve inside each span.
    #[test]
    fn a_curve_raised_in_degree_keeps_its_jump() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            1,
            vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(1.0, 0.0, 0.0),
                p(1.0, 2.0, 0.0),
                p(2.0, 2.0, 0.0),
            ],
            vec![1.0; 4],
        )
        .expect("valid");
        let raised = curve_degree_elevate(&c, 2).expect("valid");
        assert_eq!(raised.degree(), 3);
        for t in [0.1, 0.25, 0.4, 0.6, 0.75, 0.9] {
            let off = (raised.evaluate(t) - c.evaluate(t)).length();
            assert!(off < 1e-12, "at {t}: {off}");
        }
    }

    #[test]
    fn a_jump_raised_in_degree_does_not_hang_on_the_weights_scale() {
        let p = Point3::new;
        for w in [1.0, 1e-14, 1e14] {
            let c = NurbsCurve::new(
                1,
                vec![0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
                vec![
                    p(0.0, 0.0, 0.0),
                    p(1.0, 0.0, 0.0),
                    p(10.0, 0.0, 0.0),
                    p(11.0, 0.0, 0.0),
                ],
                vec![w; 4],
            )
            .expect("valid");
            let raised = curve_degree_elevate(&c, 2).expect("valid");
            for t in [0.1, 0.25, 0.4, 0.6, 0.75, 0.9] {
                let off = (raised.evaluate(t) - c.evaluate(t)).length();
                assert!(off < 1e-9, "weights {w}, at {t}: {off}");
            }
        }
    }

    #[test]
    fn a_curve_continuous_at_its_knots_raised_in_degree_stays_joined() {
        let p = Point3::new;
        let c = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0],
            vec![
                p(0.0, 0.0, 0.0),
                p(1.0, 1.0, 0.0),
                p(2.0, 0.0, 0.0),
                p(3.0, 1.0, 0.0),
                p(4.0, 0.0, 0.0),
            ],
            vec![1.0, 2.0, 1.0, 0.5, 1.0],
        )
        .expect("valid");
        let raised = curve_degree_elevate(&c, 1).expect("valid");
        assert_eq!(raised.control_points().len(), 7);
        for t in [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9] {
            let off = (raised.evaluate(t) - c.evaluate(t)).length();
            assert!(off < 1e-12, "at {t}: {off}");
        }
    }
}

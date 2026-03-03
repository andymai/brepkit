//! Bezier decomposition and degree elevation for NURBS curves and surfaces.

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::knot_ops::{curve_knot_insert, surface_knot_insert_u, surface_knot_insert_v};
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

/// Decompose a NURBS curve into Bezier segments.
///
/// Each segment is a NURBS curve of the same degree with a clamped knot vector
/// (all interior knots at full multiplicity). The segments are parametrically
/// contiguous — segment *i* goes from knot *i* to knot *i+1*.
///
/// # Errors
///
/// Returns an error if internal knot insertion fails.
pub fn curve_to_bezier_segments(curve: &NurbsCurve) -> Result<Vec<NurbsCurve>, MathError> {
    let p = curve.degree();
    let knots = curve.knots();

    // Collect unique interior knots and their deficiencies.
    let mut unique_interior = Vec::new();
    let mut i = p + 1;
    while i < knots.len() - p - 1 {
        let u = knots[i];
        let mut mult = 1;
        while i + mult < knots.len() - p - 1 && (knots[i + mult] - u).abs() < 1e-15 {
            mult += 1;
        }
        let deficit = p - mult;
        if deficit > 0 {
            unique_interior.push((u, deficit));
        }
        i += mult;
    }

    // Insert knots to make all interior knots have multiplicity p.
    let mut refined = curve.clone();
    for &(u, deficit) in &unique_interior {
        refined = curve_knot_insert(&refined, u, deficit)?;
    }

    // Now extract Bezier segments: each spans p+1 consecutive control points.
    let ref_knots = refined.knots();
    let ref_cps = refined.control_points();
    let ref_ws = refined.weights();

    let n_segments = (ref_cps.len() - 1) / p;
    if n_segments == 0 {
        return Ok(vec![refined]);
    }

    let mut segments = Vec::with_capacity(n_segments);
    for seg in 0..n_segments {
        let start_cp = seg * p;
        let end_cp = start_cp + p + 1;
        if end_cp > ref_cps.len() {
            break;
        }

        let seg_cps: Vec<Point3> = ref_cps[start_cp..end_cp].to_vec();
        let seg_ws: Vec<f64> = ref_ws[start_cp..end_cp].to_vec();

        // Knot vector for a Bezier: p+1 copies of start, p+1 copies of end
        let u_start = ref_knots[start_cp + p];
        let u_end = ref_knots[end_cp];
        let mut seg_knots = vec![u_start; p + 1];
        seg_knots.extend(std::iter::repeat_n(u_end, p + 1));

        segments.push(NurbsCurve::new(p, seg_knots, seg_cps, seg_ws)?);
    }

    Ok(segments)
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

    // First decompose into Bezier segments.
    let segments = curve_to_bezier_segments(curve)?;
    let n_segs = segments.len();

    // Precompute binomial coefficients for degree elevation.
    let bezalfs = compute_bezalfs(p, t);

    let mut all_cps: Vec<[f64; 4]> = Vec::new();
    let mut all_knots: Vec<f64> = Vec::new();

    for (seg_idx, seg) in segments.iter().enumerate() {
        let seg_cps = seg.control_points();
        let seg_ws = seg.weights();
        let seg_pw: Vec<[f64; 4]> = seg_cps
            .iter()
            .zip(seg_ws.iter())
            .map(|(pt, &w)| [pt.x() * w, pt.y() * w, pt.z() * w, w])
            .collect();

        // Degree-elevate this Bezier segment.
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
        } else {
            // Subsequent segments: skip first control point (shared with previous).
            all_cps.extend_from_slice(&elevated[1..]);
        }

        if seg_idx == n_segs - 1 {
            // Last segment: add end knot.
            let u_end = seg.knots()[p + 1];
            for _ in 0..=new_p {
                all_knots.push(u_end);
            }
        } else {
            // Interior break: add new_p copies of the break knot.
            let u_break = seg.knots()[p + 1];
            for _ in 0..new_p {
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
}

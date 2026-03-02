//! Bezier decomposition and degree elevation for NURBS curves.

use crate::MathError;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::knot_ops::curve_knot_insert;
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
}

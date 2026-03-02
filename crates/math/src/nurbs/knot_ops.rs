//! Knot insertion, refinement, and curve splitting.
//!
//! Algorithm numbers refer to Piegl & Tiller, *The NURBS Book*.

use crate::MathError;
use crate::nurbs::basis;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

/// Insert knot `u` into the curve `r` times (Boehm's algorithm, A5.1).
///
/// # Errors
///
/// Returns an error if the resulting curve cannot be constructed.
pub fn curve_knot_insert(curve: &NurbsCurve, u: f64, r: usize) -> Result<NurbsCurve, MathError> {
    if r == 0 {
        return Ok(curve.clone());
    }

    let p = curve.degree();
    let knots = curve.knots();
    let cps = curve.control_points();
    let ws = curve.weights();

    // Count existing multiplicity of u.
    let s = knots.iter().filter(|&&kv| (kv - u).abs() < 1e-15).count();

    let r = r.min(p.saturating_sub(s)); // Can't insert beyond full multiplicity
    if r == 0 {
        return Ok(curve.clone());
    }

    // Homogeneous control points.
    let qw: Vec<[f64; 4]> = cps
        .iter()
        .zip(ws.iter())
        .map(|(pt, &w)| [pt.x() * w, pt.y() * w, pt.z() * w, w])
        .collect();

    // Iterative single-knot insertion.
    let mut current_qw = qw;
    let mut current_knots: Vec<f64> = knots.to_vec();

    for _ins in 0..r {
        let cn = current_qw.len();
        let ck = basis::find_span(cn, p, u, &current_knots);

        let mut new_qw = Vec::with_capacity(cn + 1);

        // Copy unaffected points at the beginning.
        new_qw.extend_from_slice(&current_qw[..=ck.saturating_sub(p)]);

        // Compute new points in the affected region.
        for i in (ck - p + 1)..=ck {
            let denom = current_knots[i + p] - current_knots[i];
            let alpha = if denom.abs() < 1e-15 {
                0.0
            } else {
                (u - current_knots[i]) / denom
            };
            let prev = current_qw[i - 1];
            let curr = current_qw[i];
            new_qw.push([
                alpha * curr[0] + (1.0 - alpha) * prev[0],
                alpha * curr[1] + (1.0 - alpha) * prev[1],
                alpha * curr[2] + (1.0 - alpha) * prev[2],
                alpha * curr[3] + (1.0 - alpha) * prev[3],
            ]);
        }

        // Copy unaffected points at the end.
        new_qw.extend_from_slice(&current_qw[ck..cn]);

        // Update knot vector.
        let mut new_kv = Vec::with_capacity(current_knots.len() + 1);
        new_kv.extend_from_slice(&current_knots[..=ck]);
        new_kv.push(u);
        new_kv.extend_from_slice(&current_knots[ck + 1..]);

        current_qw = new_qw;
        current_knots = new_kv;
    }

    // Convert back from homogeneous.
    let new_cps: Vec<Point3> = current_qw
        .iter()
        .map(|h| {
            if h[3] == 0.0 {
                Point3::new(h[0], h[1], h[2])
            } else {
                Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
            }
        })
        .collect();
    let new_ws: Vec<f64> = current_qw.iter().map(|h| h[3]).collect();

    NurbsCurve::new(p, current_knots, new_cps, new_ws)
}

/// Insert knot `u` in the u-direction of a surface `r` times.
///
/// # Errors
///
/// Returns an error if the resulting surface cannot be constructed.
#[allow(clippy::similar_names)]
pub fn surface_knot_insert_u(
    surface: &NurbsSurface,
    u: f64,
    r: usize,
) -> Result<NurbsSurface, MathError> {
    let pu = surface.degree_u();
    let pv = surface.degree_v();
    let n_cols = surface.control_points()[0].len();

    // Treat each column of control points as a curve in u, insert knot.
    let mut new_rows: Option<Vec<Vec<Point3>>> = None;
    let mut new_wrows: Option<Vec<Vec<f64>>> = None;
    let mut new_knots_u = Vec::new();

    for col in 0..n_cols {
        let col_cps: Vec<Point3> = surface
            .control_points()
            .iter()
            .map(|row| row[col])
            .collect();
        let col_ws: Vec<f64> = surface.weights().iter().map(|row| row[col]).collect();
        let col_curve = NurbsCurve::new(pu, surface.knots_u().to_vec(), col_cps, col_ws)?;
        let inserted = curve_knot_insert(&col_curve, u, r)?;

        if new_knots_u.is_empty() {
            new_knots_u = inserted.knots().to_vec();
        }

        let n_new_rows = inserted.control_points().len();
        let rows = new_rows.get_or_insert_with(|| vec![Vec::with_capacity(n_cols); n_new_rows]);
        let wrows = new_wrows.get_or_insert_with(|| vec![Vec::with_capacity(n_cols); n_new_rows]);

        for (i, (pt, &w)) in inserted
            .control_points()
            .iter()
            .zip(inserted.weights().iter())
            .enumerate()
        {
            rows[i].push(*pt);
            wrows[i].push(w);
        }
    }

    let rows = new_rows.ok_or(MathError::EmptyInput)?;
    let wrows = new_wrows.ok_or(MathError::EmptyInput)?;

    NurbsSurface::new(pu, pv, new_knots_u, surface.knots_v().to_vec(), rows, wrows)
}

/// Insert knot `v` in the v-direction of a surface `r` times.
///
/// # Errors
///
/// Returns an error if the resulting surface cannot be constructed.
#[allow(clippy::similar_names)]
pub fn surface_knot_insert_v(
    surface: &NurbsSurface,
    v: f64,
    r: usize,
) -> Result<NurbsSurface, MathError> {
    let pu = surface.degree_u();
    let pv = surface.degree_v();

    // Treat each row of control points as a curve in v, insert knot.
    let mut new_rows = Vec::with_capacity(surface.control_points().len());
    let mut new_wrows = Vec::with_capacity(surface.weights().len());
    let mut new_knots_v = Vec::new();

    for (row_cps, row_ws) in surface
        .control_points()
        .iter()
        .zip(surface.weights().iter())
    {
        let row_curve = NurbsCurve::new(
            pv,
            surface.knots_v().to_vec(),
            row_cps.clone(),
            row_ws.clone(),
        )?;
        let inserted = curve_knot_insert(&row_curve, v, r)?;

        if new_knots_v.is_empty() {
            new_knots_v = inserted.knots().to_vec();
        }

        new_rows.push(inserted.control_points().to_vec());
        new_wrows.push(inserted.weights().to_vec());
    }

    NurbsSurface::new(
        pu,
        pv,
        surface.knots_u().to_vec(),
        new_knots_v,
        new_rows,
        new_wrows,
    )
}

/// Refine a curve by inserting multiple knots at once (A5.4).
///
/// `knots_to_insert` should be sorted in non-decreasing order.
///
/// # Errors
///
/// Returns an error if the resulting curve cannot be constructed.
pub fn curve_knot_refine(
    curve: &NurbsCurve,
    knots_to_insert: &[f64],
) -> Result<NurbsCurve, MathError> {
    if knots_to_insert.is_empty() {
        return Ok(curve.clone());
    }

    let mut result = curve.clone();
    for &u in knots_to_insert {
        result = curve_knot_insert(&result, u, 1)?;
    }
    Ok(result)
}

/// Split a curve at parameter `u` into two curves.
///
/// Inserts the knot to full multiplicity (degree + 1) at `u`, then
/// partitions the control polygon.
///
/// # Errors
///
/// Returns an error if the resulting curves cannot be constructed.
pub fn curve_split(curve: &NurbsCurve, u: f64) -> Result<(NurbsCurve, NurbsCurve), MathError> {
    let p = curve.degree();

    // Insert knot to multiplicity p (C^0 continuity = interpolatory).
    let refined = curve_knot_insert(curve, u, p)?;
    let knots = refined.knots();
    let cps = refined.control_points();
    let ws = refined.weights();

    // Find the first and last indices of u in the knot vector.
    let first_u = knots
        .iter()
        .position(|&k| (k - u).abs() < 1e-15)
        .ok_or(MathError::EmptyInput)?;

    let mut last_u = first_u;
    while last_u + 1 < knots.len() && (knots[last_u + 1] - u).abs() < 1e-15 {
        last_u += 1;
    }

    // The split control point index: with multiplicity p, the curve passes
    // through the CP at index (last_u - p). Both halves share this point.
    let split_cp = last_u - p;

    // Left curve: CPs 0..=split_cp, knots up to first_u then add p+1-mult
    // copies of u to form a proper clamped end.
    let mult = last_u - first_u + 1;
    let mut left_knots: Vec<f64> = knots[..=last_u].to_vec();
    // Need p+1 copies of u at the end; we have `mult` already.
    for _ in 0..(p + 1 - mult) {
        left_knots.push(u);
    }
    let left_cps: Vec<Point3> = cps[..=split_cp].to_vec();
    let left_ws: Vec<f64> = ws[..=split_cp].to_vec();

    // Right curve: p+1 copies of u, then remaining knots. CPs from split_cp onward.
    let mut right_knots: Vec<f64> = Vec::new();
    for _ in 0..(p + 1 - mult) {
        right_knots.push(u);
    }
    right_knots.extend_from_slice(&knots[first_u..]);
    let right_cps: Vec<Point3> = cps[split_cp..].to_vec();
    let right_ws: Vec<f64> = ws[split_cp..].to_vec();

    let left = NurbsCurve::new(p, left_knots, left_cps, left_ws)?;
    let right = NurbsCurve::new(p, right_knots, right_cps, right_ws)?;

    Ok((left, right))
}

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
    fn knot_insert_preserves_shape() {
        let c = multi_span_curve();
        let inserted = curve_knot_insert(&c, 0.25, 1).expect("valid");

        assert_eq!(
            inserted.control_points().len(),
            c.control_points().len() + 1
        );
        assert_eq!(inserted.knots().len(), c.knots().len() + 1);

        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = inserted.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-12
                    && (p1.y() - p2.y()).abs() < 1e-12
                    && (p1.z() - p2.z()).abs() < 1e-12,
                "shape mismatch at u={u}"
            );
        }
    }

    #[test]
    fn knot_insert_multiple() {
        let c = multi_span_curve();
        let inserted = curve_knot_insert(&c, 0.25, 2).expect("valid");

        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = inserted.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-11
                    && (p1.y() - p2.y()).abs() < 1e-11
                    && (p1.z() - p2.z()).abs() < 1e-11,
                "shape mismatch at u={u}"
            );
        }
    }

    #[test]
    fn knot_refine_preserves_shape() {
        let c = cubic_bezier();
        let refined = curve_knot_refine(&c, &[0.25, 0.5, 0.75]).expect("valid");

        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let p1 = c.evaluate(u);
            let p2 = refined.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-11 && (p1.y() - p2.y()).abs() < 1e-11,
                "shape mismatch at u={u}"
            );
        }
    }

    #[test]
    fn curve_split_basic() {
        let c = cubic_bezier();
        let (left, right) = curve_split(&c, 0.5).expect("valid split");

        let pl = left.evaluate(0.0);
        let pc_start = c.evaluate(0.0);
        assert!((pl.x() - pc_start.x()).abs() < 1e-12);

        let pr = right.evaluate(1.0);
        let pc_end = c.evaluate(1.0);
        assert!((pr.x() - pc_end.x()).abs() < 1e-12);
    }

    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_knot_insert_preserves_shape(u_insert in 0.01f64..0.99) {
            let c = multi_span_curve();
            let inserted = curve_knot_insert(&c, u_insert, 1).expect("valid");

            for i in 0..=10 {
                let u = i as f64 / 10.0;
                let p1 = c.evaluate(u);
                let p2 = inserted.evaluate(u);
                prop_assert!(
                    (p1.x() - p2.x()).abs() < 1e-10
                        && (p1.y() - p2.y()).abs() < 1e-10,
                    "mismatch at u={}: ({},{}) vs ({},{})",
                    u, p1.x(), p1.y(), p2.x(), p2.y()
                );
            }
        }
    }
}

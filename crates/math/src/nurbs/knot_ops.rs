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

/// Remove one occurrence of knot `u` from the curve (Piegl-Tiller A5.8).
///
/// If the knot does not exist in the knot vector the curve is returned
/// unchanged. The removal is accepted only when the deviation between the
/// two trial control-point sequences (computed from each end toward the
/// middle) is within `tolerance`.
///
/// # Errors
///
/// Returns [`MathError::ConvergenceFailure`] when the knot cannot be removed
/// within the requested tolerance (the curve geometry depends on that knot
/// too strongly).
/// Returns an error if the resulting curve cannot be constructed.
#[allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::suboptimal_flops
)]
pub fn curve_knot_remove(
    curve: &NurbsCurve,
    u: f64,
    tolerance: f64,
) -> Result<NurbsCurve, MathError> {
    let p = curve.degree();
    let knots = curve.knots();
    let cps = curve.control_points();
    let ws = curve.weights();
    let n = cps.len();

    // Find the last index of knot u in the knot vector.
    let mut r_last: Option<usize> = None;
    for (i, &kv) in knots.iter().enumerate() {
        if (kv - u).abs() < 1e-15 {
            r_last = Some(i);
        }
    }

    // Knot not found — return unchanged.
    let Some(r) = r_last else {
        return Ok(curve.clone());
    };

    // Homogeneous control points [x*w, y*w, z*w, w].
    let pw: Vec<[f64; 4]> = cps
        .iter()
        .zip(ws.iter())
        .map(|(pt, &w)| [pt.x() * w, pt.y() * w, pt.z() * w, w])
        .collect();

    // The exact inverse of single knot insertion. When curve_knot_insert
    // inserts u into the curve, it finds span k = find_span(n_old, p, u, knots_old),
    // then for each i in (k-p+1)..=k computes:
    //   Q[i] = alpha_i * P_old[i] + (1-alpha_i) * P_old[i-1]
    // where alpha_i = (u - knots_old[i]) / (knots_old[i+p] - knots_old[i]).
    //
    // The new knot vector has u inserted after knots_old[k], and the new
    // control points are: P_old[0..=k-p], Q[k-p+1..=k], P_old[k..].
    //
    // To reverse: we have the CURRENT (post-insert) knot vector and CPs.
    // We want to remove one copy of u. The removed knot vector is obtained
    // by deleting one u. Using the removed knot vector, we can compute
    // what alpha values were used, then solve for P_old.
    //
    // The span k in the REMOVED knot vector corresponds to the span where
    // u sits. In the current knot vector, the first occurrence of u is at
    // index r_first. In the removed knot vector (one fewer u), the span k
    // satisfies: removed_knots[k] <= u < removed_knots[k+1], which means
    // k = r_first - 1 if we remove the first occurrence, or more precisely
    // the span is the last knot index < u.

    // Build removed knot vector first (remove the LAST occurrence of u,
    // which is what we'll reconstruct from).
    let mut removed_knots: Vec<f64> = Vec::with_capacity(knots.len() - 1);
    {
        let mut skipped = false;
        for (idx, &kv) in knots.iter().enumerate() {
            if !skipped && idx == r {
                skipped = true;
                continue;
            }
            removed_knots.push(kv);
        }
    }

    // Find the span k in the removed knot vector.
    let n_old = n - 1; // number of old control points
    let k = basis::find_span(n_old, p, u, &removed_knots);

    // The affected range in the NEW (current) CP array is (k-p+1)..=k+1
    // (p+1 new CPs replaced p old CPs), but the unaffected prefix is 0..=k-p
    // and unaffected suffix is k+1.. (with +1 shift). Let me trace through
    // curve_knot_insert to be precise.
    //
    // In curve_knot_insert (single iteration):
    //   new_qw[0..=k-p]           = current_qw[0..=k-p]        (prefix copy)
    //   new_qw[k-p+1..=k]         = blended points              (p points)
    //   new_qw[k+1..=n_old+1]     = current_qw[k..=n_old-1]   (suffix copy, note k not k+1!)
    //
    // Wait, looking at the actual code:
    //   new_qw.extend_from_slice(&current_qw[..=ck.saturating_sub(p)]);  // 0..=k-p
    //   for i in (ck - p + 1)..=ck { ... blend ... }                     // p new points
    //   new_qw.extend_from_slice(&current_qw[ck..cn]);                   // k..=n_old-1
    //
    // So new has: (k-p+1) + p + (n_old - k) = n_old + 1 points. Correct.
    // new[0..=k-p] = old[0..=k-p]
    // new[k-p+1..=k] = blended
    // new[k+1..=n_old] = old[k..=n_old-1]
    //
    // The blend formula for new[i] where i in (k-p+1)..=k:
    //   alpha = (u - removed_knots[i]) / (removed_knots[i+p] - removed_knots[i])
    //   new[i] = alpha * old[i] + (1-alpha) * old[i-1]
    //
    // But note: in the old indexing, old[i] for i in (k-p+1)..=k corresponds to
    // the same as new[i] for i <= k-p, and old[i] = new[i+1] for i >= k.
    //
    // To recover old[i] from new, we work from both ends:
    //   From left:  old[i] = (new[i] - (1-alpha_i) * old[i-1]) / alpha_i
    //   From right: old[i-1] = (new[i] - alpha_i * old[i]) / (1-alpha_i)
    //
    // The prefix gives us old[0..=k-p] = new[0..=k-p] directly.
    // The suffix gives us old[k..=n_old-1] = new[k+1..=n_old] directly.
    // We need to recover old[k-p+1..=k-1] — that's p-1 unknowns from p equations.

    // Recover from the left: starting with old[k-p] = new[k-p] (known).
    let num_affected = p; // indices k-p+1 ..= k in the new array
    let mut left_pts = Vec::with_capacity(num_affected);
    let mut right_pts = Vec::with_capacity(num_affected);

    // From the left: recover old[k-p+1], old[k-p+2], ...
    {
        let mut prev = pw[k - p]; // old[k-p] = new[k-p]
        for idx in (k - p + 1)..=(k) {
            let denom = removed_knots[idx + p] - removed_knots[idx];
            let alpha = if denom.abs() < 1e-15 {
                0.0
            } else {
                (u - removed_knots[idx]) / denom
            };
            let qi = pw[idx]; // new[idx]
            let old_i = if alpha.abs() < 1e-15 {
                qi
            } else {
                [
                    (qi[0] - (1.0 - alpha) * prev[0]) / alpha,
                    (qi[1] - (1.0 - alpha) * prev[1]) / alpha,
                    (qi[2] - (1.0 - alpha) * prev[2]) / alpha,
                    (qi[3] - (1.0 - alpha) * prev[3]) / alpha,
                ]
            };
            left_pts.push(old_i);
            prev = old_i;
        }
    }

    // From the right: recover old[k-1], old[k-2], ...
    {
        let mut next = pw[k + 1]; // old[k] = new[k+1]
        for idx in ((k - p + 1)..=(k)).rev() {
            let denom = removed_knots[idx + p] - removed_knots[idx];
            let alpha = if denom.abs() < 1e-15 {
                0.0
            } else {
                (u - removed_knots[idx]) / denom
            };
            let qi = pw[idx]; // new[idx]
            let old_im1 = if (1.0 - alpha).abs() < 1e-15 {
                qi
            } else {
                [
                    (qi[0] - alpha * next[0]) / (1.0 - alpha),
                    (qi[1] - alpha * next[1]) / (1.0 - alpha),
                    (qi[2] - alpha * next[2]) / (1.0 - alpha),
                    (qi[3] - alpha * next[3]) / (1.0 - alpha),
                ]
            };
            right_pts.push(old_im1);
            next = old_im1;
        }
    }
    right_pts.reverse();
    // left_pts[i] = recovered old[k-p+1+i] from left sweep
    // right_pts[i] = recovered old[k-p+i] from right sweep (shifted by 1)
    // Actually right_pts recovers old[i-1] for each equation index i.
    // For idx = k: recovers old[k-1]. For idx = k-1: old[k-2]. etc.
    // After reverse: right_pts[0] = old[k-p], right_pts[1] = old[k-p+1], ..., right_pts[p-1] = old[k-1]
    // So right_pts[j] = old[k-p+j] for j in 0..p.
    // And left_pts[j] = old[k-p+1+j] for j in 0..p.
    //
    // Overlap: left_pts[j] should equal right_pts[j+1] for j in 0..p-1.
    // The tolerance check: compare them in the middle.

    // Check tolerance: compare left and right in the middle of the overlap.
    // left_pts[j] = old[k-p+1+j], right_pts[j+1] = old[k-p+1+j]
    // They should be identical for a perfectly removable knot.
    // Check all overlapping points.
    for idx in 0..p.saturating_sub(1) {
        let lp = left_pts[idx];
        let rp = right_pts[idx + 1];
        let dist = if lp[3].abs() > 1e-15 && rp[3].abs() > 1e-15 {
            let dx = lp[0] / lp[3] - rp[0] / rp[3];
            let dy = lp[1] / lp[3] - rp[1] / rp[3];
            let dz = lp[2] / lp[3] - rp[2] / rp[3];
            (dx * dx + dy * dy + dz * dz).sqrt()
        } else {
            let dx = lp[0] - rp[0];
            let dy = lp[1] - rp[1];
            let dz = lp[2] - rp[2];
            let dw = lp[3] - rp[3];
            (dx * dx + dy * dy + dz * dz + dw * dw).sqrt()
        };
        if dist > tolerance {
            return Err(MathError::ConvergenceFailure { iterations: 0 });
        }
    }

    // Build new control points (n_old = n-1 points).
    let mut new_pw: Vec<[f64; 4]> = Vec::with_capacity(n_old);

    // Prefix: old[0..=k-p] = new[0..=k-p].
    new_pw.extend_from_slice(&pw[..=k - p]);

    // Affected region: old[k-p+1..=k-1]. Use average of left and right.
    for idx in 0..(p - 1) {
        let lp = left_pts[idx];
        let rp = right_pts[idx + 1];
        new_pw.push([
            (lp[0] + rp[0]) * 0.5,
            (lp[1] + rp[1]) * 0.5,
            (lp[2] + rp[2]) * 0.5,
            (lp[3] + rp[3]) * 0.5,
        ]);
    }

    // Suffix: old[k..=n_old-1] = new[k+1..=n].
    new_pw.extend_from_slice(&pw[k + 1..]);

    // Build new knot vector.
    let new_knots = removed_knots;

    // Convert back from homogeneous.
    let new_cps: Vec<Point3> = new_pw
        .iter()
        .map(|h| {
            if h[3] == 0.0 {
                Point3::new(h[0], h[1], h[2])
            } else {
                Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])
            }
        })
        .collect();
    let new_ws: Vec<f64> = new_pw.iter().map(|h| h[3]).collect();

    NurbsCurve::new(p, new_knots, new_cps, new_ws)
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

    #[test]
    fn knot_insert_zero_times_returns_clone() {
        let c = cubic_bezier();
        let inserted = curve_knot_insert(&c, 0.5, 0).expect("valid");
        assert_eq!(inserted.knots().len(), c.knots().len());
    }

    #[test]
    fn surface_knot_insert_u_preserves_shape() {
        // Bilinear surface
        let surf = NurbsSurface::new(
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
        .expect("valid surface");

        let refined = surface_knot_insert_u(&surf, 0.5, 1).expect("valid insert");

        // The shape should be preserved at sample points
        for i in 0..=4 {
            for j in 0..=4 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let p1 = surf.evaluate(u, v);
                let p2 = refined.evaluate(u, v);
                assert!(
                    (p1.x() - p2.x()).abs() < 1e-10
                        && (p1.y() - p2.y()).abs() < 1e-10
                        && (p1.z() - p2.z()).abs() < 1e-10,
                    "shape mismatch at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn surface_knot_insert_v_preserves_shape() {
        let surf = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .expect("valid surface");

        let refined = surface_knot_insert_v(&surf, 0.5, 1).expect("valid insert");

        for i in 0..=4 {
            for j in 0..=4 {
                let u = i as f64 / 4.0;
                let v = j as f64 / 4.0;
                let p1 = surf.evaluate(u, v);
                let p2 = refined.evaluate(u, v);
                assert!(
                    (p1.x() - p2.x()).abs() < 1e-10
                        && (p1.y() - p2.y()).abs() < 1e-10
                        && (p1.z() - p2.z()).abs() < 1e-10,
                    "shape mismatch at ({u}, {v})"
                );
            }
        }
    }

    #[test]
    fn curve_split_preserves_endpoints() {
        let c = cubic_bezier();
        let (left, right) = curve_split(&c, 0.5).expect("valid split");

        // Left half should start where original starts
        let l_start = left.evaluate(0.0);
        let c_start = c.evaluate(0.0);
        assert!(
            (l_start.x() - c_start.x()).abs() < 1e-10,
            "left start should match original start"
        );

        // Right half should end where original ends
        let r_end = right.evaluate(1.0);
        let c_end = c.evaluate(1.0);
        assert!(
            (r_end.x() - c_end.x()).abs() < 1e-10,
            "right end should match original end"
        );
    }

    #[test]
    fn knot_refine_empty_list() {
        let c = cubic_bezier();
        let refined = curve_knot_refine(&c, &[]).expect("valid");
        assert_eq!(refined.knots().len(), c.knots().len());
    }

    #[test]
    fn knot_remove_roundtrip() {
        let c = multi_span_curve();
        let u_ins = 0.25;
        let inserted = curve_knot_insert(&c, u_ins, 1).expect("insert valid");
        let removed = curve_knot_remove(&inserted, u_ins, 1e-6).expect("remove valid");

        assert_eq!(
            removed.control_points().len(),
            c.control_points().len(),
            "control point count should match original"
        );
        assert_eq!(
            removed.knots().len(),
            c.knots().len(),
            "knot count should match original"
        );

        for i in 0..=20 {
            let u = i as f64 / 20.0;
            let p1 = c.evaluate(u);
            let p2 = removed.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-6
                    && (p1.y() - p2.y()).abs() < 1e-6
                    && (p1.z() - p2.z()).abs() < 1e-6,
                "shape mismatch at u={u}: original=({},{},{}), removed=({},{},{})",
                p1.x(),
                p1.y(),
                p1.z(),
                p2.x(),
                p2.y(),
                p2.z()
            );
        }
    }

    #[test]
    fn knot_remove_nonexistent_is_noop() {
        let c = multi_span_curve();
        let result = curve_knot_remove(&c, 0.123_456, 1e-6).expect("should succeed");

        assert_eq!(result.knots().len(), c.knots().len());
        assert_eq!(result.control_points().len(), c.control_points().len());

        for i in 0..=10 {
            let u = i as f64 / 10.0;
            let p1 = c.evaluate(u);
            let p2 = result.evaluate(u);
            assert!(
                (p1.x() - p2.x()).abs() < 1e-14
                    && (p1.y() - p2.y()).abs() < 1e-14
                    && (p1.z() - p2.z()).abs() < 1e-14,
                "noop should preserve shape exactly at u={u}"
            );
        }
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

        #[test]
        fn prop_knot_insert_remove_roundtrip(u_insert in 0.01f64..0.99) {
            let c = multi_span_curve();
            let inserted = curve_knot_insert(&c, u_insert, 1).expect("insert valid");
            let removed = curve_knot_remove(&inserted, u_insert, 1e-4).expect("remove valid");

            for i in 0..=10 {
                let u = i as f64 / 10.0;
                let p1 = c.evaluate(u);
                let p2 = removed.evaluate(u);
                prop_assert!(
                    (p1.x() - p2.x()).abs() < 1e-4
                        && (p1.y() - p2.y()).abs() < 1e-4
                        && (p1.z() - p2.z()).abs() < 1e-4,
                    "roundtrip mismatch at u={}: ({},{},{}) vs ({},{},{})",
                    u, p1.x(), p1.y(), p1.z(), p2.x(), p2.y(), p2.z()
                );
            }
        }
    }
}

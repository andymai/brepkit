//! Split surfaces along iso-parameter lines at continuity breaks.
//!
//! Provides detection of u- and v-parameter values where a NURBS surface
//! has insufficient continuity, and splitting at a chosen iso-parameter
//! line (`split_surface_at_u`, `split_surface_at_v`).
//!
//! The split algorithm mirrors `brepkit_math::nurbs::knot_ops::curve_split`
//! generalized to the 2D control grid: insert the knot to full multiplicity
//! along the split direction, then partition rows (or columns) of the CP
//! grid into the two sub-surfaces. The knots in the orthogonal direction
//! are unchanged.

use brepkit_math::nurbs::knot_ops::{surface_knot_insert_u, surface_knot_insert_v};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::Point3;

use crate::HealError;

/// Tolerance for comparing knot values.
const KNOT_EPS: f64 = 1e-15;

/// Find u-parameters where a NURBS surface has continuity breaks.
///
/// A break occurs at an internal u-knot whose multiplicity exceeds the
/// threshold `degree_u - min_continuity`.
///
/// - `min_continuity = 0` reports C0 breaks (multiplicity = degree)
/// - `min_continuity = 1` reports C1 or worse breaks
#[must_use]
pub fn find_surface_breaks_u(surface: &NurbsSurface, min_continuity: usize) -> Vec<f64> {
    find_knot_breaks(surface.knots_u(), surface.degree_u(), min_continuity)
}

/// Find v-parameters where a NURBS surface has continuity breaks.
///
/// A break occurs at an internal v-knot whose multiplicity exceeds the
/// threshold `degree_v - min_continuity`.
#[must_use]
pub fn find_surface_breaks_v(surface: &NurbsSurface, min_continuity: usize) -> Vec<f64> {
    find_knot_breaks(surface.knots_v(), surface.degree_v(), min_continuity)
}

/// Common implementation: find continuity breaks in a knot vector.
fn find_knot_breaks(knots: &[f64], degree: usize, min_continuity: usize) -> Vec<f64> {
    if knots.is_empty() || degree == 0 {
        return Vec::new();
    }

    // Continuity at multiplicity m is C^(degree - m).
    // Break when degree - m <= min_continuity, i.e., m >= degree - min_continuity.
    let break_threshold = degree.saturating_sub(min_continuity);

    let mut breaks = Vec::new();
    let mut i = degree + 1;
    let end = knots.len().saturating_sub(degree + 1);

    while i < end {
        let u = knots[i];
        let mut mult = 1;
        while i + mult < end && (knots[i + mult] - u).abs() < KNOT_EPS {
            mult += 1;
        }

        if mult >= break_threshold {
            breaks.push(u);
        }

        i += mult;
    }

    breaks
}

/// Split a NURBS surface at the given u-parameter, returning
/// `(left, right)` sub-surfaces sharing the iso-u line as a common
/// boundary.
///
/// `u` must lie strictly inside the open knot domain
/// `(knots_u[degree_u], knots_u[n_rows])`. Both sub-surfaces preserve
/// degree, v-knots, and column count from the input — only the row
/// (u) direction is partitioned.
///
/// # Errors
///
/// Returns [`HealError`] if `u` is outside the open domain or if knot
/// insertion / sub-surface construction fails.
pub fn split_surface_at_u(
    surface: &NurbsSurface,
    u: f64,
) -> Result<(NurbsSurface, NurbsSurface), HealError> {
    let degree_u = surface.degree_u();
    let degree_v = surface.degree_v();

    let (u_lo, u_hi) = surface.domain_u();
    if u <= u_lo + KNOT_EPS || u >= u_hi - KNOT_EPS {
        return Err(HealError::Math(
            brepkit_math::MathError::ParameterOutOfRange {
                value: u,
                min: u_lo,
                max: u_hi,
            },
        ));
    }

    // Insert u to multiplicity = degree_u (gives interpolatory C0 along
    // that iso-line, so the split row of CPs is shared between halves).
    let refined = surface_knot_insert_u(surface, u, degree_u)?;
    let knots_u = refined.knots_u();
    let cps = refined.control_points();
    let ws = refined.weights();

    let (first_u, last_u, mult) = locate_knot_run(knots_u, u)?;
    // Number of u-knots to PAD with at the boundary so each half has
    // a proper clamped knot vector of multiplicity `degree_u + 1`.
    // checked_sub guards against malformed input where mult > degree_u
    // + 1 (NurbsSurface::new doesn't validate knot multiplicity, so a
    // wrap-around `pad = usize::MAX` here would attempt a giant
    // allocation and abort).
    let pad = (degree_u + 1)
        .checked_sub(mult)
        .ok_or_else(|| degenerate_knot_error("u", mult, degree_u))?;
    // The split CP row: with multiplicity `degree_u`, the surface
    // passes through the row at index `last_u - degree_u`. Both halves
    // share this row.
    let split_row = last_u
        .checked_sub(degree_u)
        .ok_or_else(|| degenerate_knot_error("u", mult, degree_u))?;

    // Left: rows 0..=split_row, knots_u[..=last_u] + `pad` copies of u.
    let mut left_knots: Vec<f64> = knots_u[..=last_u].to_vec();
    left_knots.extend(std::iter::repeat_n(u, pad));
    let left_cps: Vec<Vec<Point3>> = cps[..=split_row].to_vec();
    let left_ws: Vec<Vec<f64>> = ws[..=split_row].to_vec();

    // Right: `pad` copies of u + knots_u[first_u..], rows split_row..
    let mut right_knots: Vec<f64> = std::iter::repeat_n(u, pad).collect();
    right_knots.extend_from_slice(&knots_u[first_u..]);
    let right_cps: Vec<Vec<Point3>> = cps[split_row..].to_vec();
    let right_ws: Vec<Vec<f64>> = ws[split_row..].to_vec();

    let knots_v = refined.knots_v().to_vec();

    let left = NurbsSurface::new(
        degree_u,
        degree_v,
        left_knots,
        knots_v.clone(),
        left_cps,
        left_ws,
    )?;
    let right = NurbsSurface::new(
        degree_u,
        degree_v,
        right_knots,
        knots_v,
        right_cps,
        right_ws,
    )?;
    Ok((left, right))
}

/// Split a NURBS surface at the given v-parameter, returning
/// `(low, high)` sub-surfaces sharing the iso-v line as a common
/// boundary.
///
/// Symmetric counterpart to [`split_surface_at_u`].
///
/// # Errors
///
/// Returns [`HealError`] if `v` is outside the open domain or if knot
/// insertion / sub-surface construction fails.
pub fn split_surface_at_v(
    surface: &NurbsSurface,
    v: f64,
) -> Result<(NurbsSurface, NurbsSurface), HealError> {
    let degree_u = surface.degree_u();
    let degree_v = surface.degree_v();

    let (v_lo, v_hi) = surface.domain_v();
    if v <= v_lo + KNOT_EPS || v >= v_hi - KNOT_EPS {
        return Err(HealError::Math(
            brepkit_math::MathError::ParameterOutOfRange {
                value: v,
                min: v_lo,
                max: v_hi,
            },
        ));
    }

    let refined = surface_knot_insert_v(surface, v, degree_v)?;
    let knots_v = refined.knots_v();
    let cps = refined.control_points();
    let ws = refined.weights();

    let (first_v, last_v, mult) = locate_knot_run(knots_v, v)?;
    let pad = (degree_v + 1)
        .checked_sub(mult)
        .ok_or_else(|| degenerate_knot_error("v", mult, degree_v))?;
    let split_col = last_v
        .checked_sub(degree_v)
        .ok_or_else(|| degenerate_knot_error("v", mult, degree_v))?;

    let mut low_knots: Vec<f64> = knots_v[..=last_v].to_vec();
    low_knots.extend(std::iter::repeat_n(v, pad));
    let low_cps: Vec<Vec<Point3>> = cps.iter().map(|row| row[..=split_col].to_vec()).collect();
    let low_ws: Vec<Vec<f64>> = ws.iter().map(|row| row[..=split_col].to_vec()).collect();

    let mut high_knots: Vec<f64> = std::iter::repeat_n(v, pad).collect();
    high_knots.extend_from_slice(&knots_v[first_v..]);
    let high_cps: Vec<Vec<Point3>> = cps.iter().map(|row| row[split_col..].to_vec()).collect();
    let high_ws: Vec<Vec<f64>> = ws.iter().map(|row| row[split_col..].to_vec()).collect();

    let knots_u = refined.knots_u().to_vec();

    let low = NurbsSurface::new(
        degree_u,
        degree_v,
        knots_u.clone(),
        low_knots,
        low_cps,
        low_ws,
    )?;
    let high = NurbsSurface::new(degree_u, degree_v, knots_u, high_knots, high_cps, high_ws)?;
    Ok((low, high))
}

/// Locate the contiguous run of `knot` in a knot vector after
/// insertion to full multiplicity. Returns `(first_idx, last_idx,
/// multiplicity)`.
///
/// On a missing knot, returns [`HealError::UpgradeFailed`] (NOT
/// `MathError::ParameterOutOfRange`) — by the time this runs, the
/// caller has already verified that the parameter is in-domain and
/// inserted it; if it's still missing, that's an internal invariant
/// failure (knot insertion didn't take), not a user-facing range
/// problem.
fn locate_knot_run(knots: &[f64], knot: f64) -> Result<(usize, usize, usize), HealError> {
    let first = knots
        .iter()
        .position(|&k| (k - knot).abs() < KNOT_EPS)
        .ok_or_else(|| {
            HealError::UpgradeFailed(format!(
                "split_surface: knot {knot:e} not present in knot vector after \
                 insertion (internal invariant failure)"
            ))
        })?;
    let mut last = first;
    while last + 1 < knots.len() && (knots[last + 1] - knot).abs() < KNOT_EPS {
        last += 1;
    }
    Ok((first, last, last - first + 1))
}

/// Build the typed error for a degenerate knot-multiplicity case
/// (`mult > degree + 1`, which `NurbsSurface::new` doesn't reject).
fn degenerate_knot_error(direction: &str, mult: usize, degree: usize) -> HealError {
    HealError::UpgradeFailed(format!(
        "split_surface: {direction}-knot multiplicity {mult} exceeds \
         degree+1 ({}) — input surface has malformed knot vector",
        degree + 1
    ))
}

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::float_cmp
    )]

    use super::*;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::Point3;

    fn make_surface_with_u_break() -> NurbsSurface {
        // Degree 2 in u, degree 1 in v.
        // Internal u-knot at 0.5 with multiplicity 2 (C0 break for degree 2).
        let degree_u = 2;
        let degree_v = 1;
        let knots_u = vec![0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0];
        let knots_v = vec![0.0, 0.0, 1.0, 1.0];
        // 5 rows (u) x 2 cols (v)
        let control_points = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
            vec![Point3::new(1.0, 0.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            vec![Point3::new(2.0, 0.0, 1.0), Point3::new(2.0, 1.0, 1.0)],
            vec![Point3::new(3.0, 0.0, 0.0), Point3::new(3.0, 1.0, 0.0)],
            vec![Point3::new(4.0, 0.0, 0.0), Point3::new(4.0, 1.0, 0.0)],
        ];
        let weights = vec![vec![1.0; 2]; 5];
        NurbsSurface::new(
            degree_u,
            degree_v,
            knots_u,
            knots_v,
            control_points,
            weights,
        )
        .unwrap()
    }

    #[test]
    fn find_u_c0_break() {
        let surface = make_surface_with_u_break();
        let breaks = find_surface_breaks_u(&surface, 0);
        assert_eq!(breaks.len(), 1);
        assert!((breaks[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn no_v_breaks() {
        let surface = make_surface_with_u_break();
        let breaks = find_surface_breaks_v(&surface, 0);
        assert!(breaks.is_empty());
    }

    fn make_smooth_patch() -> NurbsSurface {
        // 4×3 degree (3, 2) NURBS over [0, 1] × [0, 1].
        // 4 + 3 + 1 = 8 u-knots; 3 + 2 + 1 = 6 v-knots.
        let knots_u = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let knots_v = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let mut cps: Vec<Vec<Point3>> = Vec::with_capacity(4);
        for i in 0..4 {
            let u = f64::from(i) / 3.0;
            let mut row = Vec::with_capacity(3);
            for j in 0..3 {
                let v = f64::from(j) / 2.0;
                row.push(Point3::new(u, v, (u + v).sin()));
            }
            cps.push(row);
        }
        let weights = vec![vec![1.0; 3]; 4];
        NurbsSurface::new(3, 2, knots_u, knots_v, cps, weights).unwrap()
    }

    #[test]
    fn split_at_u_preserves_evaluation() {
        // Splitting the surface at u=u_split must produce two patches
        // whose evaluations on either side of the split match the
        // original surface within tolerance.
        use brepkit_math::traits::ParametricSurface;

        let surface = make_smooth_patch();
        let u_split = 0.4_f64;
        let (left, right) = split_surface_at_u(&surface, u_split).unwrap();

        // Sample on the left side: u ∈ (0, u_split).
        for (iu, iv) in (0..5).flat_map(|i| (0..5).map(move |j| (i, j))) {
            let u_orig = f64::from(iu) / 4.0 * u_split;
            let v_param = f64::from(iv) / 4.0;
            let p_orig = ParametricSurface::evaluate(&surface, u_orig, v_param);
            // Map to left's domain: left covers [0, u_split].
            let p_left = ParametricSurface::evaluate(&left, u_orig, v_param);
            assert!(
                (p_orig - p_left).length() < 1e-9,
                "left mismatch at ({u_orig}, {v_param}): {p_orig:?} vs {p_left:?}"
            );
        }

        // Sample on the right side: u ∈ (u_split, 1).
        for (iu, iv) in (0..5).flat_map(|i| (0..5).map(move |j| (i, j))) {
            let u_orig = u_split + f64::from(iu) / 4.0 * (1.0 - u_split);
            let v_param = f64::from(iv) / 4.0;
            let p_orig = ParametricSurface::evaluate(&surface, u_orig, v_param);
            let p_right = ParametricSurface::evaluate(&right, u_orig, v_param);
            assert!(
                (p_orig - p_right).length() < 1e-9,
                "right mismatch at ({u_orig}, {v_param}): {p_orig:?} vs {p_right:?}"
            );
        }
    }

    #[test]
    fn split_at_v_preserves_evaluation() {
        use brepkit_math::traits::ParametricSurface;

        let surface = make_smooth_patch();
        let v_split = 0.6_f64;
        let (low, high) = split_surface_at_v(&surface, v_split).unwrap();

        for (iu, iv) in (0..5).flat_map(|i| (0..5).map(move |j| (i, j))) {
            let u_param = f64::from(iu) / 4.0;
            let v_orig = f64::from(iv) / 4.0 * v_split;
            let p_orig = ParametricSurface::evaluate(&surface, u_param, v_orig);
            let p_low = ParametricSurface::evaluate(&low, u_param, v_orig);
            assert!(
                (p_orig - p_low).length() < 1e-9,
                "low mismatch at ({u_param}, {v_orig}): {p_orig:?} vs {p_low:?}"
            );
        }
        for (iu, iv) in (0..5).flat_map(|i| (0..5).map(move |j| (i, j))) {
            let u_param = f64::from(iu) / 4.0;
            let v_orig = v_split + f64::from(iv) / 4.0 * (1.0 - v_split);
            let p_orig = ParametricSurface::evaluate(&surface, u_param, v_orig);
            let p_high = ParametricSurface::evaluate(&high, u_param, v_orig);
            assert!(
                (p_orig - p_high).length() < 1e-9,
                "high mismatch at ({u_param}, {v_orig}): {p_orig:?} vs {p_high:?}"
            );
        }
    }

    #[test]
    fn split_at_u_then_v_gives_4_patches_covering_original() {
        // Splitting in u then v of each result yields 4 sub-patches
        // that together evaluate identically to the original.
        use brepkit_math::traits::ParametricSurface;

        let surface = make_smooth_patch();
        let (l, r) = split_surface_at_u(&surface, 0.5).unwrap();
        let (ll, lh) = split_surface_at_v(&l, 0.5).unwrap();
        let (rl, rh) = split_surface_at_v(&r, 0.5).unwrap();

        let cases: [(f64, f64, &NurbsSurface); 4] = [
            (0.2, 0.2, &ll),
            (0.2, 0.7, &lh),
            (0.7, 0.2, &rl),
            (0.7, 0.7, &rh),
        ];
        for (u, v, sub) in cases {
            let p_orig = ParametricSurface::evaluate(&surface, u, v);
            let p_sub = ParametricSurface::evaluate(sub, u, v);
            assert!(
                (p_orig - p_sub).length() < 1e-9,
                "sub-patch mismatch at ({u}, {v}): {p_orig:?} vs {p_sub:?}"
            );
        }
    }

    #[test]
    fn split_at_u_rejects_out_of_domain() {
        let surface = make_smooth_patch();
        // Domain is [0, 1]; 1.5 is outside.
        let err = split_surface_at_u(&surface, 1.5).unwrap_err();
        // Out-of-domain must surface as the typed range error so
        // callers can react programmatically (not just the catch-all
        // UpgradeFailed string).
        match err {
            HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value,
                min,
                max,
                ..
            }) => {
                assert!(
                    (value - 1.5).abs() < 1e-12,
                    "value should be 1.5, got {value}"
                );
                assert!((min - 0.0).abs() < 1e-12, "min should be 0.0, got {min}");
                assert!((max - 1.0).abs() < 1e-12, "max should be 1.0, got {max}");
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn split_at_v_rejects_out_of_domain() {
        // Symmetric to split_at_u_rejects_out_of_domain — ensures the
        // v-direction domain validation produces the same typed error
        // shape (catches regressions where someone copies the u logic
        // and forgets to update one of the field references).
        let surface = make_smooth_patch();
        let err = split_surface_at_v(&surface, -0.3).unwrap_err();
        match err {
            HealError::Math(brepkit_math::MathError::ParameterOutOfRange {
                value,
                min,
                max,
                ..
            }) => {
                assert!(
                    (value + 0.3).abs() < 1e-12,
                    "value should be -0.3, got {value}"
                );
                assert!((min - 0.0).abs() < 1e-12, "min should be 0.0, got {min}");
                assert!((max - 1.0).abs() < 1e-12, "max should be 1.0, got {max}");
            }
            other => panic!("expected ParameterOutOfRange, got {other:?}"),
        }
    }
}

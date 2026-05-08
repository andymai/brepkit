//! Split curves at continuity breaks.
//!
//! Provides utilities for finding parameter values where a NURBS curve
//! has insufficient continuity (based on internal knot multiplicity) and
//! for splitting the curve at those parameters.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::curve_split;

use crate::HealError;

/// Tolerance for comparing knot values.
const KNOT_EPS: f64 = 1e-10;

/// Find parameters where a NURBS curve has continuity breaks.
///
/// A break occurs at an internal knot whose multiplicity exceeds the
/// threshold `degree - min_continuity`. For example, with degree 3 and
/// `min_continuity = 1` (C1), any internal knot with multiplicity >= 3
/// is a break.
///
/// - `min_continuity = 0` reports C0 breaks (multiplicity = degree)
/// - `min_continuity = 1` reports C1 or worse breaks (multiplicity >= degree - 1)
/// - `min_continuity = 2` reports C2 or worse breaks (multiplicity >= degree - 2)
#[must_use]
pub fn find_continuity_breaks(curve: &NurbsCurve, min_continuity: usize) -> Vec<f64> {
    let degree = curve.degree();
    let knots = curve.knots();

    if knots.is_empty() || degree == 0 {
        return Vec::new();
    }

    // The continuity at a knot with multiplicity m is C^(degree - m).
    // We want to find knots where continuity <= min_continuity,
    // i.e., degree - m <= min_continuity, i.e., m >= degree - min_continuity.
    let break_threshold = degree.saturating_sub(min_continuity);

    // Iterate unique internal knots.
    let mut breaks = Vec::new();
    let mut i = degree + 1; // Skip the first (degree+1) clamped knots.
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

/// Split a NURBS curve at the given parameter values.
///
/// Returns sub-curves whose union is geometrically identical to the
/// input — produced by repeatedly applying
/// [`brepkit_math::nurbs::knot_ops::curve_split`] at each split
/// parameter. Each sub-curve preserves the original degree, weights,
/// and knot structure exactly (no fit-through-samples drift).
///
/// Split parameters that fall outside the curve's open knot domain or
/// duplicate other entries are ignored.
///
/// # Errors
///
/// Returns [`HealError::UpgradeFailed`] if `curve_split` fails on
/// any sub-curve (e.g., due to malformed knot vectors).
pub fn split_curve_at_params(
    curve: &NurbsCurve,
    params: &[f64],
) -> Result<Vec<NurbsCurve>, HealError> {
    if params.is_empty() {
        return Ok(vec![curve.clone()]);
    }

    // Sort and deduplicate split parameters.
    let mut sorted_params: Vec<f64> = params.to_vec();
    sorted_params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_params.dedup_by(|a, b| (*a - *b).abs() < KNOT_EPS);

    // Drop split params outside the open domain — they would be no-ops
    // at best and produce invalid knot vectors at worst.
    let knots = curve.knots();
    let degree = curve.degree();
    let u_lo = knots[degree];
    let u_hi = knots[knots.len() - degree - 1];
    sorted_params.retain(|&u| u > u_lo + KNOT_EPS && u < u_hi - KNOT_EPS);

    if sorted_params.is_empty() {
        return Ok(vec![curve.clone()]);
    }

    // Repeatedly split: each call to curve_split returns (left, right),
    // we keep splitting `right` at the next sorted parameter.
    let mut segments: Vec<NurbsCurve> = Vec::with_capacity(sorted_params.len() + 1);
    let mut remaining = curve.clone();
    for &u in &sorted_params {
        let (left, right) = curve_split(&remaining, u)
            .map_err(|e| HealError::UpgradeFailed(format!("curve_split failed at u={u}: {e}")))?;
        segments.push(left);
        remaining = right;
    }
    segments.push(remaining);
    Ok(segments)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Point3;

    fn make_cubic_with_c0_break() -> NurbsCurve {
        // Degree 3, with an internal knot at 0.5 with multiplicity 3 (C0 break).
        let degree = 3;
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0];
        let control_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(3.0, 0.5, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(5.0, -1.0, 0.0),
            Point3::new(6.0, 0.0, 0.0),
        ];
        let weights = vec![1.0; 7];
        NurbsCurve::new(degree, knots, control_points, weights).unwrap()
    }

    #[test]
    fn find_c0_breaks() {
        let curve = make_cubic_with_c0_break();
        let breaks = find_continuity_breaks(&curve, 0);
        assert_eq!(breaks.len(), 1);
        assert!((breaks[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn find_c1_breaks_includes_c0() {
        let curve = make_cubic_with_c0_break();
        // C1 should also report the C0 break (multiplicity 3 > 3 - 1 = 2).
        let breaks = find_continuity_breaks(&curve, 1);
        assert_eq!(breaks.len(), 1);
    }

    #[test]
    fn no_breaks_for_smooth_curve() {
        let degree = 3;
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let control_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(4.0, -1.0, 0.0),
        ];
        let weights = vec![1.0; 5];
        let curve = NurbsCurve::new(degree, knots, control_points, weights).unwrap();

        let breaks = find_continuity_breaks(&curve, 0);
        assert!(breaks.is_empty());
    }

    #[test]
    fn split_at_c0_break() {
        let curve = make_cubic_with_c0_break();
        let breaks = find_continuity_breaks(&curve, 0);
        assert_eq!(breaks, vec![0.5]);

        let segments = split_curve_at_params(&curve, &breaks).unwrap();
        assert_eq!(
            segments.len(),
            2,
            "expected 2 segments after splitting at C0 break"
        );
        assert_eq!(segments[0].degree(), 3);
        assert_eq!(segments[1].degree(), 3);
    }

    #[test]
    fn split_empty_params_returns_original() {
        let curve = make_cubic_with_c0_break();
        let segments = split_curve_at_params(&curve, &[]).unwrap();
        assert_eq!(segments.len(), 1);
    }

    #[test]
    fn split_preserves_evaluation_exactly() {
        // The exact-split implementation must reproduce the original
        // curve's evaluation on each sub-piece within fp tolerance,
        // NOT just within a fitting-error bound.
        let curve = make_cubic_with_c0_break();
        let segments = split_curve_at_params(&curve, &[0.5]).unwrap();
        assert_eq!(segments.len(), 2);

        // Sample the first segment over its domain and compare with the
        // original curve at the same parameter.
        let s0 = &segments[0];
        let (s0_lo, s0_hi) = (
            s0.knots()[s0.degree()],
            s0.knots()[s0.knots().len() - s0.degree() - 1],
        );
        for k in 0..=10 {
            #[allow(clippy::cast_precision_loss)]
            let t = s0_lo + (s0_hi - s0_lo) * (k as f64 / 10.0);
            let p_seg = s0.evaluate(t);
            let p_orig = curve.evaluate(t);
            assert!(
                (p_seg - p_orig).length() < 1e-9,
                "left segment mismatch at t={t}: seg {p_seg:?} vs orig {p_orig:?}"
            );
        }

        let s1 = &segments[1];
        let (s1_lo, s1_hi) = (
            s1.knots()[s1.degree()],
            s1.knots()[s1.knots().len() - s1.degree() - 1],
        );
        for k in 0..=10 {
            #[allow(clippy::cast_precision_loss)]
            let t = s1_lo + (s1_hi - s1_lo) * (k as f64 / 10.0);
            let p_seg = s1.evaluate(t);
            let p_orig = curve.evaluate(t);
            assert!(
                (p_seg - p_orig).length() < 1e-9,
                "right segment mismatch at t={t}: seg {p_seg:?} vs orig {p_orig:?}"
            );
        }
    }

    #[test]
    fn split_at_multiple_params_yields_n_plus_one_segments() {
        // Smooth degree-3 curve, splittable at any interior parameter.
        let degree = 3;
        let knots = vec![0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0];
        let cps: Vec<Point3> = (0..7)
            .map(|i| {
                let t = f64::from(i) / 6.0;
                Point3::new(t, t * t, 0.0)
            })
            .collect();
        let weights = vec![1.0; 7];
        let curve = NurbsCurve::new(degree, knots, cps, weights).unwrap();

        let segments = split_curve_at_params(&curve, &[0.3, 0.6]).unwrap();
        assert_eq!(segments.len(), 3, "two splits → three segments");
    }

    #[test]
    fn split_ignores_out_of_domain_params() {
        let curve = make_cubic_with_c0_break();
        // 1.5 is outside [0, 1] — should be silently ignored.
        let segments = split_curve_at_params(&curve, &[1.5]).unwrap();
        assert_eq!(segments.len(), 1);
    }
}

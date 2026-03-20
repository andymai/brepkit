//! Split curves at continuity breaks.
//!
//! Provides utilities for finding parameter values where a NURBS curve
//! has insufficient continuity (based on internal knot multiplicity) and
//! for splitting the curve at those parameters.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::curve_knot_insert;

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
/// Inserts knots at each split parameter until multiplicity reaches
/// `degree + 1` (full break), then extracts the sub-curves between
/// consecutive break points.
///
/// # Errors
///
/// Returns [`HealError::UpgradeFailed`] if knot insertion or curve
/// construction fails.
pub fn split_curve_at_params(
    curve: &NurbsCurve,
    params: &[f64],
) -> Result<Vec<NurbsCurve>, HealError> {
    if params.is_empty() {
        return Ok(vec![curve.clone()]);
    }

    let degree = curve.degree();

    // Sort and deduplicate split parameters.
    let mut sorted_params: Vec<f64> = params.to_vec();
    sorted_params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_params.dedup_by(|a, b| (*a - *b).abs() < KNOT_EPS);

    // Insert knots at each split point to full multiplicity (degree + 1).
    let mut refined = curve.clone();
    for &u in &sorted_params {
        // Count current multiplicity of u in the refined curve.
        let current_mult = refined
            .knots()
            .iter()
            .filter(|&&kv| (kv - u).abs() < KNOT_EPS)
            .count();

        // Need multiplicity = degree for a C0 split (curve passes through CP).
        let needed = degree.saturating_sub(current_mult);
        if needed > 0 {
            let result = curve_knot_insert(&refined, u, needed);
            match result {
                Ok(new_curve) => {
                    refined = new_curve;
                }
                Err(e) => {
                    return Err(HealError::UpgradeFailed(format!(
                        "knot insertion failed at u={u}: {e}"
                    )));
                }
            }
        }
    }

    // Extract sub-curves: find unique knot values with full multiplicity
    // (degree + 1) — these are the segment boundaries. Each segment's
    // control points span between consecutive boundaries.
    let ref_knots = refined.knots();

    // Find all unique knot values that have multiplicity >= degree.
    // These are the segment boundaries (including domain endpoints,
    // which have multiplicity = degree + 1 in clamped B-splines).
    let mut boundary_knots: Vec<f64> = Vec::new();
    let mut i = 0;
    while i < ref_knots.len() {
        let u = ref_knots[i];
        let mut mult = 1;
        while i + mult < ref_knots.len() && (ref_knots[i + mult] - u).abs() < KNOT_EPS {
            mult += 1;
        }
        if mult >= degree {
            boundary_knots.push(u);
        }
        i += mult;
    }

    if boundary_knots.len() < 2 {
        return Ok(vec![curve.clone()]);
    }

    // Extract sub-curves between consecutive boundary knots.
    // At each boundary with multiplicity m, the curve passes through
    // control point cp[k-m] (where k is the span index). We build
    // clamped sub-curves by repeating the boundary knot to full
    // multiplicity (degree+1) on each side.
    let n_segments = boundary_knots.len() - 1;
    let mut segments = Vec::with_capacity(n_segments);

    for seg_idx in 0..n_segments {
        let u_start = boundary_knots[seg_idx];
        let u_end = boundary_knots[seg_idx + 1];

        // Evaluate the refined curve at several points to get the sub-curve
        // geometry, then fit a new NURBS through those points.
        // This is simpler and more robust than partitioning knots/CPs.
        let n_pts = degree + 4; // enough points for a good fit
        let mut pts = Vec::with_capacity(n_pts);
        for k in 0..n_pts {
            #[allow(clippy::cast_precision_loss)]
            let t = u_start + (u_end - u_start) * (k as f64 / (n_pts - 1) as f64);
            pts.push(refined.evaluate(t));
        }

        let sub_curve = brepkit_math::nurbs::fitting::interpolate(&pts, degree).map_err(|e| {
            HealError::UpgradeFailed(format!("failed to fit sub-curve [{u_start}, {u_end}]: {e}"))
        })?;

        segments.push(sub_curve);
    }

    if segments.is_empty() {
        Ok(vec![curve.clone()])
    } else {
        Ok(segments)
    }
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
}

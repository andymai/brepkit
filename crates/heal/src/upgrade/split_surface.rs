//! Split surfaces along iso-parameter lines at continuity breaks.
//!
//! Provides detection of u- and v-parameter values where a NURBS surface
//! has insufficient continuity. Actual surface splitting (which requires
//! knot insertion in both directions and sub-patch extraction) is not yet
//! implemented.

use brepkit_math::nurbs::surface::NurbsSurface;

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

// TODO: Implement actual surface splitting.
//
// `split_surface_at_u` and `split_surface_at_v` would need to:
// 1. Insert knots in the appropriate direction to full multiplicity
// 2. Extract sub-patches (rows of control points for u-splits, columns for v-splits)
// 3. Construct valid NurbsSurface for each sub-patch
//
// For splitting in both u and v, apply u-splits first, then v-splits
// to each resulting sub-surface (or vice versa).

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

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
}

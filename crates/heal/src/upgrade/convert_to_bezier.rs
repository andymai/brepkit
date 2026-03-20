//! Convert B-spline geometry to Bezier segments.
//!
//! Delegates to [`brepkit_math::nurbs::decompose::curve_to_bezier_segments`]
//! for the actual decomposition, wrapping the result in the heal error type.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_to_bezier_segments;

use crate::HealError;

/// Decompose a NURBS curve into Bezier segments.
///
/// Inserts knots at every internal knot to full multiplicity, then
/// extracts each span as a separate Bezier curve (a NURBS curve with
/// clamped knots and no internal knots).
///
/// Each output curve has degree equal to the input and `degree + 1`
/// control points.
///
/// # Errors
///
/// Returns [`HealError::UpgradeFailed`] if knot insertion or curve
/// construction fails.
pub fn decompose_curve_to_bezier(curve: &NurbsCurve) -> Result<Vec<NurbsCurve>, HealError> {
    curve_to_bezier_segments(curve)
        .map_err(|e| HealError::UpgradeFailed(format!("Bezier decomposition failed: {e}")))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Point3;

    #[test]
    fn decompose_single_span_returns_one() {
        // A single-span Bezier curve (no internal knots) should return itself.
        let degree = 3;
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let control_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(2.0, 2.0, 0.0),
            Point3::new(3.0, 0.0, 0.0),
        ];
        let weights = vec![1.0; 4];
        let curve = NurbsCurve::new(degree, knots, control_points, weights).unwrap();

        let segments = decompose_curve_to_bezier(&curve).unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].degree(), 3);
        assert_eq!(segments[0].control_points().len(), 4);
    }

    #[test]
    fn decompose_two_span_returns_two() {
        // Degree 2 with one internal knot at 0.5 (multiplicity 1 < degree).
        let degree = 2;
        let knots = vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0];
        let control_points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(3.0, -1.0, 0.0),
        ];
        let weights = vec![1.0; 4];
        let curve = NurbsCurve::new(degree, knots, control_points, weights).unwrap();

        let segments = decompose_curve_to_bezier(&curve).unwrap();
        assert_eq!(segments.len(), 2);
        for seg in &segments {
            assert_eq!(seg.degree(), 2);
            assert_eq!(seg.control_points().len(), 3); // degree + 1 for Bezier
        }
    }
}

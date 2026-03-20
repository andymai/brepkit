//! Curve analysis — arc length, degeneracy, continuity breaks.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::traits::ParametricCurve;

use crate::status::Status;

/// Number of sample points for arc length approximation.
const ARC_LENGTH_SAMPLES: usize = 32;

/// Result of analyzing a curve.
#[derive(Debug, Clone)]
pub struct CurveAnalysis {
    /// Approximate arc length computed by chord-length summation.
    pub approx_length: f64,
    /// Whether the curve is degenerate (zero length).
    pub is_degenerate: bool,
    /// Parameter values where continuity drops (C0 breaks at internal knots).
    pub continuity_breaks: Vec<f64>,
    /// Outcome status flags.
    pub status: Status,
}

/// Analyze a NURBS curve for length, degeneracy, and continuity.
///
/// Continuity breaks are detected at internal knots where the knot
/// multiplicity equals the curve degree (C0 break).
#[must_use]
pub fn analyze_curve(curve: &NurbsCurve, tolerance: &Tolerance) -> CurveAnalysis {
    let (t_min, t_max) = ParametricCurve::domain(curve);

    // Approximate arc length.
    let mut length = 0.0;
    let mut prev = ParametricCurve::evaluate(curve, t_min);
    for i in 1..=ARC_LENGTH_SAMPLES {
        let t = t_min + (t_max - t_min) * (i as f64 / ARC_LENGTH_SAMPLES as f64);
        let pt = ParametricCurve::evaluate(curve, t);
        length += (pt - prev).length();
        prev = pt;
    }

    let is_degenerate = length < tolerance.linear;

    // Detect C0 continuity breaks: internal knots with multiplicity == degree.
    let continuity_breaks = detect_continuity_breaks(curve);

    let mut status = Status::OK;
    if is_degenerate {
        status = status.merge(Status::DONE1);
    }
    if !continuity_breaks.is_empty() {
        status = status.merge(Status::DONE2);
    }

    CurveAnalysis {
        approx_length: length,
        is_degenerate,
        continuity_breaks,
        status,
    }
}

/// Find internal knots where the multiplicity equals the degree (C0 breaks).
fn detect_continuity_breaks(curve: &NurbsCurve) -> Vec<f64> {
    let knots = curve.knots();
    let degree = curve.degree();
    let (t_min, t_max) = ParametricCurve::domain(curve);

    let mut breaks = Vec::new();
    let mut i = 0;
    while i < knots.len() {
        let val = knots[i];
        // Count multiplicity.
        let mut mult = 0;
        while i + mult < knots.len() && (knots[i + mult] - val).abs() < 1e-15 {
            mult += 1;
        }
        // Internal knot with multiplicity == degree means C0 break.
        if val > t_min + 1e-15 && val < t_max - 1e-15 && mult >= degree {
            breaks.push(val);
        }
        i += mult;
    }
    breaks
}

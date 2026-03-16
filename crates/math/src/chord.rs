//! Chord deviation computation for circular arc discretization.
//!
//! Given a circle of radius `r`, the chord deviation (sag) at the midpoint
//! of an arc subtending angle `θ` is `r*(1 - cos(θ/2))`. This module
//! provides the inverse: given a maximum deflection, compute the number
//! of segments needed to discretize an arc.

/// Compute the number of segments needed to discretize a circular arc
/// so that the chord-height deviation stays below `deflection`.
///
/// For a circle of radius `r`, the maximum sag of a chord subtending
/// angle `θ` is `r*(1 - cos(θ/2))`. Solving for the number of segments
/// `n` over an arc range: `n = ceil(range / θ)` where
/// `θ = 2*acos(1 - deflection/r)`.
///
/// Returns at least 4 segments. For degenerate inputs (non-positive
/// radius, deflection, or arc range) returns 8 as a safe default.
#[must_use]
pub fn segments_for_chord_deviation(radius: f64, arc_range: f64, deflection: f64) -> usize {
    if radius <= 0.0 || deflection <= 0.0 || arc_range <= 0.0 {
        return 8;
    }
    let ratio = (deflection / radius).min(0.5); // clamp to avoid near-degenerate arcs
    let theta = 2.0 * (1.0 - ratio).acos(); // max angle per segment
    if theta <= 0.0 {
        return 8;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (arc_range / theta).ceil() as usize;
    // Minimum segment count for doubly-curved surfaces (e.g. spheres) where
    // the geometric formula under-samples because it only considers single-
    // direction curvature. Scales with sqrt(radius/deflection) so larger
    // radii correctly produce more segments.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n_min = (arc_range * (radius / deflection).sqrt()).ceil() as usize;
    n.max(n_min).max(4)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use std::f64::consts::TAU;

    #[test]
    fn degenerate_inputs() {
        assert_eq!(segments_for_chord_deviation(0.0, TAU, 0.01), 8);
        assert_eq!(segments_for_chord_deviation(1.0, 0.0, 0.01), 8);
        assert_eq!(segments_for_chord_deviation(1.0, TAU, 0.0), 8);
        assert_eq!(segments_for_chord_deviation(-1.0, TAU, 0.01), 8);
    }

    #[test]
    fn minimum_four_segments() {
        // Even for very coarse deflection, at least 4 segments.
        let n = segments_for_chord_deviation(1.0, TAU, 10.0);
        assert!(n >= 4, "got {n}");
    }

    #[test]
    fn finer_deflection_more_segments() {
        let coarse = segments_for_chord_deviation(1.0, TAU, 0.1);
        let fine = segments_for_chord_deviation(1.0, TAU, 0.01);
        assert!(fine > coarse, "fine={fine} should be > coarse={coarse}");
    }

    #[test]
    fn larger_radius_more_segments() {
        let small = segments_for_chord_deviation(1.0, TAU, 0.05);
        let large = segments_for_chord_deviation(10.0, TAU, 0.05);
        assert!(large > small, "large={large} should be > small={small}");
    }
}

//! Chord deviation computation for circular arc discretization.
//!
//! Given a circle of radius `r`, the chord deviation (sag) at the midpoint
//! of an arc subtending angle `θ` is `r*(1 - cos(θ/2))`. This module
//! provides the inverse: given a maximum deflection, compute the number
//! of segments needed to discretize an arc.
//!
//! Two independent tolerances drive density: the linear sag `deflection`
//! and an angular cap `angular_tol` (max tangent turn per segment). The
//! linear criterion alone lets a small-radius arc take a near-`π` turn per
//! segment, under-tessellating sharp rounded features; the angular cap
//! floors the segment count independently of radius.

/// Default angular deflection cap (radians) when a caller has no preference.
///
/// ~20° — chosen so small fillet/chamfer arcs reach reference density.
pub const DEFAULT_ANGULAR_TOL: f64 = 0.35;

/// Compute the number of segments needed to discretize a circular arc
/// so that the chord-height deviation stays below `deflection`.
///
/// Equivalent to [`segments_for_chord_deviation_with_angle`] with the
/// default angular cap and no minimum-edge-length clamp.
///
/// Returns at least 4 segments. For degenerate inputs (non-positive
/// radius, deflection, or arc range) returns 8 as a safe default.
#[must_use]
pub fn segments_for_chord_deviation(radius: f64, arc_range: f64, deflection: f64) -> usize {
    segments_for_chord_deviation_with_angle(radius, arc_range, deflection, DEFAULT_ANGULAR_TOL, 0.0)
}

/// Compute the number of segments to discretize a circular arc so that both
/// the chord-height deviation stays below `deflection` and the per-segment
/// tangent turn stays below `angular_tol`.
///
/// The per-segment angle is `θ_step = max(min(θ_lin, α), θ_minsize)` where:
/// - `θ_lin = 2*acos(clamp(1 - deflection/radius, 0, 1))` is the linear sag angle,
/// - `α = angular_tol` is the angular cap,
/// - `θ_minsize = min(min_len/radius, π/2)` (when `min_len > 0`) floors the
///   step so a vanishingly small radius cannot demand sub-`min_len` edges.
///
/// The segment count is `ceil(arc_range / θ_step)`, kept at least 4.
/// For degenerate inputs (non-positive radius, deflection, or arc range)
/// returns 8 as a safe default. A non-positive `angular_tol` is treated as
/// "no angular cap" (linear-only behaviour).
#[must_use]
pub fn segments_for_chord_deviation_with_angle(
    radius: f64,
    arc_range: f64,
    deflection: f64,
    angular_tol: f64,
    min_len: f64,
) -> usize {
    use std::f64::consts::FRAC_PI_2;

    if radius <= 0.0 || deflection <= 0.0 || arc_range <= 0.0 {
        return 8;
    }

    let theta_lin = 2.0 * (1.0 - deflection / radius).clamp(0.0, 1.0).acos();

    let mut theta_step = if angular_tol > 0.0 {
        theta_lin.min(angular_tol)
    } else {
        theta_lin
    };

    if min_len > 0.0 {
        let theta_minsize = (min_len / radius).min(FRAC_PI_2);
        theta_step = theta_step.max(theta_minsize);
    }

    if theta_step <= 0.0 {
        return 8;
    }

    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (arc_range / theta_step).ceil() as usize;

    // Legacy curvature floor for doubly-curved surfaces. Retained as a lower
    // bound (never reduces the count) so existing watertight tessellations
    // stay bit-identical: it dominates only for large radii where it demands
    // MORE segments than the angular cap, preserving the shared-boundary
    // vertex counts that adjacent faces stitch against.
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
    fn larger_radius_more_segments_linear_only() {
        // With the angular cap disabled the linear sag criterion dominates,
        // so a larger radius (smaller per-segment turn) yields more segments.
        let small = segments_for_chord_deviation_with_angle(1.0, TAU, 0.05, 0.0, 0.0);
        let large = segments_for_chord_deviation_with_angle(10.0, TAU, 0.05, 0.0, 0.0);
        assert!(large > small, "large={large} should be > small={small}");
    }

    #[test]
    fn angular_cap_floors_small_radius() {
        let n = segments_for_chord_deviation_with_angle(0.5, TAU, 0.1, 0.35, 0.0);
        let floor = (TAU / 0.35).ceil() as usize;
        assert!(n >= floor, "got {n}, expected >= {floor}");
    }

    #[test]
    fn large_angular_cap_matches_linear_only() {
        // alpha large => angular cap inactive => identical to linear-only.
        for (r, d) in [(1.0, 0.05), (10.0, 0.01), (0.4, 0.02)] {
            let capped = segments_for_chord_deviation_with_angle(r, TAU, d, 10.0, 0.0);
            let linear = segments_for_chord_deviation_with_angle(r, TAU, d, 0.0, 0.0);
            assert_eq!(capped, linear, "r={r} d={d}");
        }
    }

    #[test]
    fn angular_degenerate_inputs_return_default() {
        assert_eq!(
            segments_for_chord_deviation_with_angle(0.0, TAU, 0.1, 0.35, 0.0),
            8
        );
        assert_eq!(
            segments_for_chord_deviation_with_angle(1.0, 0.0, 0.1, 0.35, 0.0),
            8
        );
        assert_eq!(
            segments_for_chord_deviation_with_angle(1.0, TAU, 0.0, 0.35, 0.0),
            8
        );
    }

    #[test]
    fn min_len_caps_blow_up_on_tiny_radius() {
        // Tiny radius with a tight angular cap would demand huge counts;
        // min_len floors the per-segment angle so the count stays bounded.
        let unbounded = segments_for_chord_deviation_with_angle(1e-4, TAU, 1e-6, 0.05, 0.0);
        let bounded = segments_for_chord_deviation_with_angle(1e-4, TAU, 1e-6, 0.05, 0.1);
        assert!(
            bounded < unbounded,
            "bounded={bounded} should be < unbounded={unbounded}"
        );
    }

    #[test]
    fn min_size_angle_capped_at_half_pi() {
        // min_len much larger than radius must not exceed pi/2 per segment;
        // a full circle then needs at least 4 segments.
        let n = segments_for_chord_deviation_with_angle(0.1, TAU, 1e-6, 0.01, 100.0);
        let floor = (TAU / std::f64::consts::FRAC_PI_2).ceil() as usize;
        assert!(n >= floor, "got {n}, expected >= {floor}");
    }
}

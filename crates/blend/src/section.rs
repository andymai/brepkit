//! Cross-section of a blend surface at a spine parameter.

use brepkit_math::vec::Point3;

/// A circular cross-section of the blend at a given spine parameter.
#[derive(Debug, Clone)]
pub struct CircSection {
    /// Contact point on surface 1.
    pub p1: Point3,
    /// Contact point on surface 2.
    pub p2: Point3,
    /// Center of the rolling ball.
    pub center: Point3,
    /// Fillet radius at this section.
    pub radius: f64,
    /// Surface 1 parameters (u, v) at the contact point.
    pub uv1: (f64, f64),
    /// Surface 2 parameters (u, v) at the contact point.
    pub uv2: (f64, f64),
    /// Spine parameter where this section was computed.
    pub t: f64,
}

impl CircSection {
    /// Half-angle of the fillet arc (angle from center to each contact).
    #[must_use]
    pub fn half_angle(&self) -> f64 {
        let d = (self.p1 - self.p2).length();
        if self.radius < f64::EPSILON {
            return 0.0;
        }
        (d / (2.0 * self.radius)).clamp(-1.0, 1.0).asin()
    }
}

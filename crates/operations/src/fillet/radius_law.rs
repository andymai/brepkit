//! Fillet radius law definitions for variable-radius fillets.

/// Law governing how fillet radius varies along an edge.
#[derive(Debug, Clone)]
pub enum FilletRadiusLaw {
    /// Constant radius (same as basic [`super::fillet`]).
    Constant(f64),
    /// Linear interpolation from `start_radius` to `end_radius`.
    Linear {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
    /// Smooth S-curve (sinusoidal) interpolation between two radii.
    SCurve {
        /// Radius at the start of the edge.
        start: f64,
        /// Radius at the end of the edge.
        end: f64,
    },
}

impl FilletRadiusLaw {
    /// Evaluate the radius at parameter `t ∈ [0, 1]` along the edge.
    #[must_use]
    pub fn evaluate(&self, t: f64) -> f64 {
        let t = t.clamp(0.0, 1.0);
        match self {
            Self::Constant(r) => *r,
            Self::Linear { start, end } => (end - start).mul_add(t, *start),
            Self::SCurve { start, end } => {
                // Smooth step: 3t² - 2t³ (Hermite interpolation)
                let s = t * t * (-2.0f64).mul_add(t, 3.0);
                (end - start).mul_add(s, *start)
            }
        }
    }
}

//! Compositional geometric properties accumulator.
//!
//! `GProps` stores mass (volume or area), center of mass, and inertia tensor.
//! Multiple `GProps` can be combined using Huygens' parallel-axis theorem.

use brepkit_math::vec::Point3;

/// Accumulated geometric properties of a shape.
///
/// For solid properties, `mass` is volume. For surface properties, `mass` is area.
/// The inertia tensor is stored as 6 components of the symmetric matrix:
/// `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]`.
#[derive(Debug, Clone)]
pub struct GProps {
    /// Total mass (volume for solids, area for surfaces).
    pub mass: f64,
    /// Center of mass.
    pub center: Point3,
    /// Inertia tensor components `[Ixx, Iyy, Izz, Ixy, Ixz, Iyz]` at center of mass.
    pub inertia: [f64; 6],
}

impl GProps {
    /// Create empty properties (zero mass, origin center, zero inertia).
    #[must_use]
    pub fn new() -> Self {
        Self {
            mass: 0.0,
            center: Point3::new(0.0, 0.0, 0.0),
            inertia: [0.0; 6],
        }
    }

    /// Combine another `GProps` into this one using Huygens' parallel-axis theorem.
    ///
    /// The inertia tensors are shifted to the new combined center of mass.
    pub fn add(&mut self, other: &Self) {
        let m_total = self.mass + other.mass;
        if m_total.abs() < 1e-30 {
            return;
        }

        // New center of mass (weighted average)
        let cx = (self.mass * self.center.x() + other.mass * other.center.x()) / m_total;
        let cy = (self.mass * self.center.y() + other.mass * other.center.y()) / m_total;
        let cz = (self.mass * self.center.z() + other.mass * other.center.z()) / m_total;
        let new_center = Point3::new(cx, cy, cz);

        // Shift both inertia tensors to new center via parallel-axis theorem:
        // I_shifted = I_original + m * (d^2 * I_3 - d (x) d)
        // where d = old_center - new_center
        let i_self = shift_inertia(&self.inertia, self.mass, self.center, new_center);
        let i_other = shift_inertia(&other.inertia, other.mass, other.center, new_center);

        self.mass = m_total;
        self.center = new_center;
        for k in 0..6 {
            self.inertia[k] = i_self[k] + i_other[k];
        }
    }

    /// Return the 3x3 symmetric inertia matrix.
    #[must_use]
    pub fn matrix_of_inertia(&self) -> [[f64; 3]; 3] {
        let [ixx, iyy, izz, ixy, ixz, iyz] = self.inertia;
        [[ixx, -ixy, -ixz], [-ixy, iyy, -iyz], [-ixz, -iyz, izz]]
    }
}

impl Default for GProps {
    fn default() -> Self {
        Self::new()
    }
}

/// Shift inertia tensor from `old_center` to `new_center` using parallel-axis theorem.
fn shift_inertia(
    inertia: &[f64; 6],
    mass: f64,
    old_center: Point3,
    new_center: Point3,
) -> [f64; 6] {
    let dx = old_center.x() - new_center.x();
    let dy = old_center.y() - new_center.y();
    let dz = old_center.z() - new_center.z();
    let d_sq = dx * dx + dy * dy + dz * dz;

    [
        inertia[0] + mass * (d_sq - dx * dx), // Ixx + m*(dy^2+dz^2)
        inertia[1] + mass * (d_sq - dy * dy), // Iyy + m*(dx^2+dz^2)
        inertia[2] + mass * (d_sq - dz * dz), // Izz + m*(dx^2+dy^2)
        inertia[3] + mass * dx * dy,          // Ixy + m*dx*dy
        inertia[4] + mass * dx * dz,          // Ixz + m*dx*dz
        inertia[5] + mass * dy * dz,          // Iyz + m*dy*dz
    ]
}

//! Geometric properties: volume, area, center of mass, inertia tensor.

pub mod accumulator;
pub mod analytic;
pub mod bbox;

pub use accumulator::GProps;

use brepkit_math::aabb::Aabb3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::CheckError;

/// Options for property computation.
#[derive(Debug, Clone)]
pub struct PropertiesOptions {
    /// Gauss quadrature order (default 5).
    pub gauss_order: usize,
    /// Adaptive integration tolerance (default 1e-6).
    pub adaptive_eps: f64,
    /// Maximum adaptive subdivision depth (default 8).
    pub max_depth: usize,
}

impl Default for PropertiesOptions {
    fn default() -> Self {
        Self {
            gauss_order: 5,
            adaptive_eps: 1e-6,
            max_depth: 8,
        }
    }
}

/// Compute the bounding box of a solid.
///
/// # Errors
///
/// Returns an error if any topology entity is missing or the solid has no vertices.
pub fn bounding_box(topo: &Topology, solid: SolidId) -> Result<Aabb3, CheckError> {
    bbox::bounding_box(topo, solid)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use brepkit_math::vec::Point3;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    #[test]
    fn gprops_accumulator_two_cubes() {
        // Two unit cubes side by side along x-axis
        let a = analytic::box_props(1.0, 1.0, 1.0);
        let mut b = analytic::box_props(1.0, 1.0, 1.0);
        // Shift b's center to (1.5, 0.5, 0.5) — as if placed at x=1
        b.center = Point3::new(1.5, 0.5, 0.5);

        let mut combined = a;
        combined.add(&b);

        // Total volume = 2
        assert!((combined.mass - 2.0).abs() < 1e-12);
        // Combined center = (1.0, 0.5, 0.5)
        assert!((combined.center.x() - 1.0).abs() < 1e-12);
        assert!((combined.center.y() - 0.5).abs() < 1e-12);
        assert!((combined.center.z() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn box_props_volume_and_com() {
        let props = analytic::box_props(2.0, 3.0, 4.0);
        assert!((props.mass - 24.0).abs() < 1e-12);
        assert!((props.center.x() - 1.0).abs() < 1e-12);
        assert!((props.center.y() - 1.5).abs() < 1e-12);
        assert!((props.center.z() - 2.0).abs() < 1e-12);
        // Ixx = 24/12 * (9 + 16) = 50
        assert!((props.inertia[0] - 50.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_props_volume() {
        let props = analytic::sphere_props(1.0);
        let expected = 4.0 / 3.0 * std::f64::consts::PI;
        assert!((props.mass - expected).abs() < 1e-12);
        assert!((props.center.x()).abs() < 1e-12);
        assert!((props.center.y()).abs() < 1e-12);
        assert!((props.center.z()).abs() < 1e-12);
    }

    #[test]
    fn cylinder_props_volume_and_com() {
        let props = analytic::cylinder_props(1.0, 2.0);
        let expected_v = std::f64::consts::PI * 2.0;
        assert!((props.mass - expected_v).abs() < 1e-12);
        assert!((props.center.z() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cone_full_volume() {
        let props = analytic::cone_props(1.0, 0.0, 3.0);
        let expected_v = std::f64::consts::PI * 3.0 / 3.0; // pi * h/3 * r^2
        assert!((props.mass - expected_v).abs() < 1e-12);
        // CoM of full cone at h/4 from base
        assert!((props.center.z() - 0.75).abs() < 1e-12);
    }

    #[test]
    fn torus_props_volume() {
        let props = analytic::torus_props(3.0, 1.0);
        let expected_v = 2.0 * std::f64::consts::PI * std::f64::consts::PI * 3.0;
        assert!((props.mass - expected_v).abs() < 1e-12);
    }

    #[test]
    fn box_surface_area() {
        let area = analytic::box_area(2.0, 3.0, 4.0);
        // 2*(6 + 12 + 8) = 52
        assert!((area - 52.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_surface_area() {
        let area = analytic::sphere_area(2.0);
        let expected = 4.0 * std::f64::consts::PI * 4.0;
        assert!((area - expected).abs() < 1e-12);
    }

    #[test]
    fn inertia_matrix_symmetric() {
        let mut props = GProps::new();
        props.inertia = [10.0, 20.0, 30.0, 1.0, 2.0, 3.0];
        let mat = props.matrix_of_inertia();
        // Off-diagonal symmetry
        assert!((mat[0][1] - mat[1][0]).abs() < 1e-15);
        assert!((mat[0][2] - mat[2][0]).abs() < 1e-15);
        assert!((mat[1][2] - mat[2][1]).abs() < 1e-15);
        // Diagonal values
        assert!((mat[0][0] - 10.0).abs() < 1e-15);
        assert!((mat[1][1] - 20.0).abs() < 1e-15);
        assert!((mat[2][2] - 30.0).abs() < 1e-15);
    }

    #[test]
    fn bounding_box_unit_cube() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let aabb = bounding_box(&topo, solid).unwrap();
        // Unit cube at origin: min=(0,0,0), max=(1,1,1)
        assert!((aabb.min.x()).abs() < 1e-12);
        assert!((aabb.min.y()).abs() < 1e-12);
        assert!((aabb.min.z()).abs() < 1e-12);
        assert!((aabb.max.x() - 1.0).abs() < 1e-12);
        assert!((aabb.max.y() - 1.0).abs() < 1e-12);
        assert!((aabb.max.z() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn accumulator_default_is_zero() {
        let props = GProps::default();
        assert!((props.mass).abs() < 1e-15);
        assert!((props.center.x()).abs() < 1e-15);
        assert!((props.center.y()).abs() < 1e-15);
        assert!((props.center.z()).abs() < 1e-15);
        for &c in &props.inertia {
            assert!(c.abs() < 1e-15);
        }
    }
}

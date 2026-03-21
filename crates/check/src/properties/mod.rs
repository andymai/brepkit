//! Geometric properties: volume, area, center of mass, inertia tensor.

pub mod accumulator;
pub mod analytic;
pub mod bbox;
pub mod face_integrator;

pub use accumulator::GProps;

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
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

/// Compute the volume of a solid via face integration.
///
/// Uses the divergence theorem: V = (1/3) sum of integral P dot N dA
/// over all faces of the outer shell.
///
/// # Errors
///
/// Returns an error if any topology entity is missing or integration fails.
pub fn solid_volume(
    topo: &Topology,
    solid: SolidId,
    options: &PropertiesOptions,
) -> Result<f64, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_volume = 0.0;
    for &fid in shell.faces() {
        let contrib = face_integrator::integrate_face(topo, fid, options.gauss_order)?;
        total_volume += contrib.volume;
    }
    Ok(total_volume)
}

/// Compute the total surface area of a solid.
///
/// Sums the area of each face in the solid's outer shell.
///
/// # Errors
///
/// Returns an error if any topology entity is missing or integration fails.
pub fn solid_area(
    topo: &Topology,
    solid: SolidId,
    options: &PropertiesOptions,
) -> Result<f64, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_area = 0.0;
    for &fid in shell.faces() {
        let contrib = face_integrator::integrate_face(topo, fid, options.gauss_order)?;
        total_area += contrib.area;
    }
    Ok(total_area)
}

/// Compute the center of mass of a solid.
///
/// Uses the divergence theorem: for each coordinate axis, integrates
/// `(1/2) x_i^2 * n_i` over the solid's boundary, then divides by total
/// volume to obtain the volumetric centroid (solid CoM).
///
/// # Errors
///
/// Returns an error if any topology entity is missing, integration fails,
/// or the solid has zero volume.
pub fn center_of_mass(
    topo: &Topology,
    solid: SolidId,
    options: &PropertiesOptions,
) -> Result<Point3, CheckError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut total_volume = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;

    for &fid in shell.faces() {
        let contrib = face_integrator::integrate_face(topo, fid, options.gauss_order)?;
        total_volume += contrib.volume;
        mx += contrib.volume_moment_x;
        my += contrib.volume_moment_y;
        mz += contrib.volume_moment_z;
    }

    if total_volume.abs() < 1e-30 {
        return Err(CheckError::IntegrationFailed(
            "solid has zero volume".into(),
        ));
    }

    Ok(Point3::new(
        mx / total_volume,
        my / total_volume,
        mz / total_volume,
    ))
}

/// Compute the v-range for an analytic surface by projecting face wire
/// vertices onto the given axis.
///
/// Iterates over all wires (outer + inner) of `face_id`, projects each vertex
/// position onto `axis` relative to `origin`, and returns `(v_min, v_max)`.
/// If the face has no distinguishable range (e.g. a single vertex),
/// returns `(-1.0, 1.0)` as a fallback.
///
/// # Errors
///
/// Returns an error if any topology entity is missing.
pub fn axial_v_range(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    axis: Vec3,
) -> Result<(f64, f64), CheckError> {
    let face_data = topo.face(face_id)?;
    let outer = topo.wire(face_data.outer_wire())?;

    let mut v_min = f64::MAX;
    let mut v_max = f64::MIN;

    // Chain outer wire and inner wires.
    let inner_wires: Vec<_> = face_data
        .inner_wires()
        .iter()
        .filter_map(|&wid| topo.wire(wid).ok())
        .collect();

    for wire in std::iter::once(outer).chain(inner_wires.iter().copied()) {
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [oe.oriented_start(edge), oe.oriented_end(edge)] {
                let pt = topo.vertex(vid)?.point();
                let to_pt = Vec3::new(
                    pt.x() - origin.x(),
                    pt.y() - origin.y(),
                    pt.z() - origin.z(),
                );
                let v = axis.dot(to_pt);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }

    if v_min < v_max {
        Ok((v_min, v_max))
    } else {
        Ok((-1.0, 1.0))
    }
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
    fn gauss_volume_matches_analytic() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let options = PropertiesOptions::default();
        let vol = solid_volume(&topo, solid, &options).unwrap();
        // Unit cube volume = 1.0
        assert!((vol - 1.0).abs() < 1e-10, "expected volume 1.0, got {vol}");
    }

    #[test]
    fn gauss_area_matches_analytic() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let options = PropertiesOptions::default();
        let area = solid_area(&topo, solid, &options).unwrap();
        // Unit cube surface area = 6.0
        assert!((area - 6.0).abs() < 1e-10, "expected area 6.0, got {area}");
    }

    #[test]
    fn gauss_com_matches_analytic() {
        let mut topo = Topology::new();
        let solid = make_unit_cube_manifold(&mut topo);
        let options = PropertiesOptions::default();
        let com = center_of_mass(&topo, solid, &options).unwrap();
        // Unit cube CoM at (0.5, 0.5, 0.5)
        assert!((com.x() - 0.5).abs() < 1e-10, "com.x = {}", com.x());
        assert!((com.y() - 0.5).abs() < 1e-10, "com.y = {}", com.y());
        assert!((com.z() - 0.5).abs() < 1e-10, "com.z = {}", com.z());
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

//! Geometric properties: volume, area, center of mass, inertia tensor.

pub mod accumulator;
pub mod analytic;
pub mod bbox;
mod boundary;
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
/// Uses the divergence theorem: V = (1/3) sum of integral (P - C) dot N dA
/// over every face of the solid, cavity shells included, with C the origin
/// for a solid near it and the centre of its box for one far away.
///
/// # Errors
///
/// Returns an error if any topology entity is missing or integration fails.
pub fn solid_volume(
    topo: &Topology,
    solid: SolidId,
    options: &PropertiesOptions,
) -> Result<f64, CheckError> {
    let about = volume_anchor(topo, solid);
    let mut total_volume = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let contrib = face_integrator::integrate_face_about(topo, fid, options.gauss_order, about)?;
        total_volume += contrib.volume;
    }
    Ok(total_volume)
}

/// [`solid_volume`] when every face integrates exactly, else `None`.
///
/// A plane face integrates along its edges, a cylinder, cone or sphere face
/// along its wires. `None` for a solid with a torus or NURBS face, or a face
/// neither path reads (a wire that does not chain, a cone face across its
/// apex), for a caller that would otherwise measure it from a mesh.
///
/// # Errors
///
/// Returns an error if any topology entity is missing.
pub fn exact_solid_volume(topo: &Topology, solid: SolidId) -> Result<Option<f64>, CheckError> {
    use brepkit_topology::face::FaceSurface;
    let about = volume_anchor(topo, solid);
    let mut total = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(fid)?;
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        let contribution = match face.surface() {
            FaceSurface::Plane { normal, .. } => {
                face_integrator::planar_face_by_edges(topo, fid, *normal * sign, about)?
            }
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => boundary::curved_face_by_boundary(topo, face, sign, about)?,
            FaceSurface::Nurbs(_) => None,
        };
        let Some(contribution) = contribution else {
            return Ok(None);
        };
        total += contribution.volume;
    }
    Ok(Some(total))
}

/// The point [`solid_volume`] sums about: the origin, as it always has, unless
/// the solid sits more than ten of its own half-diagonals from it, where the
/// far-off coordinates would swamp the flux. A solid without vertices (an
/// empty boolean result) has no box and sums about the origin.
fn volume_anchor(topo: &Topology, solid: SolidId) -> Vec3 {
    let Ok(b) = bbox::bounding_box(topo, solid) else {
        return Vec3::new(0.0, 0.0, 0.0);
    };
    let centre = Vec3::new(
        f64::midpoint(b.min.x(), b.max.x()),
        f64::midpoint(b.min.y(), b.max.y()),
        f64::midpoint(b.min.z(), b.max.z()),
    );
    let reach = (b.max - b.min).length() * 0.5;
    if centre.length() <= 10.0 * reach.max(1.0) {
        Vec3::new(0.0, 0.0, 0.0)
    } else {
        centre
    }
}

/// Compute the total surface area of a solid.
///
/// Sums the area of every face of the solid, cavity shells included.
///
/// # Errors
///
/// Returns an error if any topology entity is missing or integration fails.
pub fn solid_area(
    topo: &Topology,
    solid: SolidId,
    options: &PropertiesOptions,
) -> Result<f64, CheckError> {
    let mut total_area = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
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
    let mut total_volume = 0.0;
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;

    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
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
    fn empty_solid_has_no_volume() {
        let mut topo = Topology::new();
        let empty = topo.add_empty_solid();
        let volume = solid_volume(&topo, empty, &PropertiesOptions::default()).unwrap();
        assert!(volume.abs() < f64::EPSILON);
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

    /// A band of the ring torus of radii 4 and 1 about the z axis between
    /// its tube's circles at `u0` and `u1`, each wire closed at its own
    /// angle round the tube. Each wire carries `u` on from where it starts,
    /// so a band across the surface's `u = 0` starts its wires a turn
    /// apart; it still holds `(u1 - u0)` of the turn's area `2 pi R r` and of
    /// its volume term `pi R r^2`, wherever its wires start.
    #[test]
    fn a_torus_band_between_tube_circles_holds_its_share() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        use std::f64::consts::{FRAC_PI_2, PI};
        let (big, small) = (4.0_f64, 1.0_f64);
        // The tube's circle at `u`, run toward growing `v` when `up`, closed
        // at `v = from`.
        let tube_circle = |topo: &mut Topology, u: f64, from: f64, up: bool| {
            let out = Vec3::new(u.cos(), u.sin(), 0.0);
            let z = Vec3::new(0.0, 0.0, 1.0);
            let c = Circle3D::new_with_ref(
                Point3::new(0.0, 0.0, 0.0) + out * big,
                out.cross(z),
                small,
                out * from.cos() + z * from.sin(),
            )
            .unwrap();
            let v = topo.add_vertex(Vertex::new(c.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(c)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, up)], true).unwrap())
        };
        let starts = [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2];
        for (u0, u1) in [(-0.1, 0.1), (0.3, 0.5), (-0.5, 2.0)] {
            for (a, b) in starts
                .iter()
                .flat_map(|&a| starts.iter().map(move |&b| (a, b)))
            {
                let mut topo = Topology::new();
                let high = tube_circle(&mut topo, u1, a, true);
                let low = tube_circle(&mut topo, u0, b, false);
                let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small).unwrap();
                let face = Face::new(high, vec![low], FaceSurface::Torus(torus));
                let c =
                    boundary::curved_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
                        .unwrap()
                        .unwrap();
                let (area, volume) = (
                    (u1 - u0) * 2.0 * PI * big * small,
                    (u1 - u0) * PI * big * small * small,
                );
                assert!(
                    (c.area - area).abs() < 1e-9 * area
                        && (c.volume - volume).abs() < 1e-9 * volume,
                    "u {u0}..{u1}, starts {a} and {b}: area {} of {area}, volume {} of {volume}",
                    c.area,
                    c.volume
                );
            }
        }
    }
}

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
            FaceSurface::Nurbs(_) => {
                match boundary::nurbs_face_over_domain(topo, face, sign, about)? {
                    Some(whole) => Some(whole),
                    None => boundary::nurbs_face_by_boundary(topo, face, sign, about)?,
                }
            }
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

    /// The band of the ring torus of radii 4 and 1 between its tube's
    /// circles at `u = 0.3` and `0.5`, each circle drawn as rational quadratic
    /// arcs: four quarter turns whose weights 1, 100/sqrt 2 and 10000 bunch
    /// each arc's parameter at its start, or three third turns of one knot
    /// span each. It holds 0.2 of the turn's area `2 pi R r` and of its
    /// volume term `pi R r^2`.
    #[test]
    fn a_torus_band_between_rational_tube_circles_holds_its_share() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};
        let (big, small) = (4.0_f64, 1.0_f64);
        // The tube's circle at `u` in `arcs` arcs, each from `v` to the next
        // with its middle weight `w1` and its last `w2`, run toward growing
        // `v` when `up`.
        let tube_circle = |topo: &mut Topology, u: f64, arcs: u32, w1: f64, w2: f64, up: bool| {
            let out = Vec3::new(u.cos(), u.sin(), 0.0);
            let z = Vec3::new(0.0, 0.0, 1.0);
            let centre = Point3::new(0.0, 0.0, 0.0) + out * big;
            let at = |v: f64| centre + (out * v.cos() + z * v.sin()) * small;
            let step = TAU / f64::from(arcs);
            let vertices: Vec<_> = (0..arcs)
                .map(|k| topo.add_vertex(Vertex::new(at(step * f64::from(k)), 1e-7)))
                .collect();
            let mut edges: Vec<OrientedEdge> = (0..arcs)
                .map(|k| {
                    let (a, b) = (step * f64::from(k), step * f64::from(k + 1));
                    let corner = centre
                        + (out * f64::midpoint(a, b).cos() + z * f64::midpoint(a, b).sin())
                            * (small / (0.5 * step).cos());
                    let curve = NurbsCurve::new(
                        2,
                        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                        vec![at(a), corner, at(b)],
                        vec![1.0, w1, w2],
                    )
                    .unwrap();
                    let (start, end) = (vertices[k as usize], vertices[((k + 1) % arcs) as usize]);
                    let e = topo.add_edge(Edge::new(start, end, EdgeCurve::NurbsCurve(curve)));
                    OrientedEdge::new(e, up)
                })
                .collect();
            if !up {
                edges.reverse();
            }
            topo.add_wire(Wire::new(edges, true).unwrap())
        };
        let (u0, u1) = (0.3, 0.5);
        for (arcs, w1, w2) in [(4, 100.0 * FRAC_1_SQRT_2, 10_000.0), (3, 0.5, 1.0)] {
            let mut topo = Topology::new();
            let high = tube_circle(&mut topo, u1, arcs, w1, w2, true);
            let low = tube_circle(&mut topo, u0, arcs, w1, w2, false);
            let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small).unwrap();
            let face = Face::new(high, vec![low], FaceSurface::Torus(torus));
            let c = boundary::curved_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
                .unwrap()
                .unwrap();
            let (area, volume) = (
                (u1 - u0) * 2.0 * PI * big * small,
                (u1 - u0) * PI * big * small * small,
            );
            assert!(
                (c.area - area).abs() < 1e-9 * area && (c.volume - volume).abs() < 1e-9 * volume,
                "{arcs} arcs: area {} of {area}, volume {} of {volume}",
                c.area,
                c.volume
            );
        }
    }

    /// A quarter of a cone of half-angle pi/4 between generator lengths 1 and
    /// 3, its generators drawn as lines or as polynomial NURBS lines of degree
    /// 12 run as `t^12` (every control point at the start but the last).
    /// Those lie on the cone exactly, but the area and volume terms, linear
    /// in the generator length, reach degree 23 along them, past one Gauss
    /// rule's reach; read either way the quarter holds the same area and
    /// volume term.
    #[test]
    fn a_cone_quarter_reads_its_generators_however_they_run() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_math::traits::ParametricSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        use std::f64::consts::{FRAC_PI_2, FRAC_PI_4};
        let read = |polynomial: bool| {
            let mut topo = Topology::new();
            let z = Vec3::new(0.0, 0.0, 1.0);
            let cone = ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), z, FRAC_PI_4).unwrap();
            let at = |u: f64, v: f64| ParametricSurface::evaluate(&cone, u, v);
            let corners = [
                at(0.0, 1.0),
                at(FRAC_PI_2, 1.0),
                at(FRAC_PI_2, 3.0),
                at(0.0, 3.0),
            ];
            let [a, b, c, d] = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let rim = |v: f64, up: f64| {
                let (sin, cos) = FRAC_PI_4.sin_cos();
                EdgeCurve::Circle(
                    brepkit_math::curves::Circle3D::new(
                        Point3::new(0.0, 0.0, v * sin),
                        z * up,
                        v * cos,
                    )
                    .unwrap(),
                )
            };
            let generator = |from: Point3, to: Point3| {
                if polynomial {
                    let mut points = vec![from; 12];
                    points.push(to);
                    let mut knots = vec![0.0; 13];
                    knots.extend([1.0; 13]);
                    EdgeCurve::NurbsCurve(
                        NurbsCurve::new(12, knots, points, vec![1.0; 13]).unwrap(),
                    )
                } else {
                    EdgeCurve::Line
                }
            };
            let edges = [
                Edge::new(a, b, rim(1.0, 1.0)),
                Edge::new(b, c, generator(corners[1], corners[2])),
                Edge::new(c, d, rim(3.0, -1.0)),
                Edge::new(d, a, generator(corners[3], corners[0])),
            ]
            .map(|e| OrientedEdge::new(topo.add_edge(e), true));
            let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
            let face = Face::new(wire, vec![], FaceSurface::Cone(cone));
            boundary::curved_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.5, -1.0, -2.0))
                .unwrap()
                .unwrap()
        };
        let (lines, polynomials) = (read(false), read(true));
        assert!(
            (polynomials.area - lines.area).abs() < 1e-9 * lines.area
                && (polynomials.volume - lines.volume).abs() < 1e-9 * lines.volume.abs(),
            "area {} of {}, volume {} of {}",
            polynomials.area,
            lines.area,
            polynomials.volume,
            lines.volume
        );
    }

    /// A band of the ring torus of radii 4 and 1 between two of its circles
    /// about the axis, at tube angles `v0` and `v1`, each wire closed at its
    /// own angle about the axis. A band across the inner equator (`v = pi`)
    /// starts its wires a turn of `v` apart; it still holds the band's area
    /// `2 pi r (R dv + r d(sin v))` and its volume term, wherever its wires
    /// start.
    #[test]
    fn a_torus_band_between_circles_about_its_axis_holds_its_share() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ToroidalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        use std::f64::consts::{FRAC_PI_2, PI, TAU};
        let (big, small) = (4.0_f64, 1.0_f64);
        // The circle about the axis at tube angle `v`, run toward growing `u`
        // when `forward`, closed at `u = from`.
        let latitude = |topo: &mut Topology, v: f64, from: f64, forward: bool| {
            let c = Circle3D::new_with_ref(
                Point3::new(0.0, 0.0, small * v.sin()),
                Vec3::new(0.0, 0.0, 1.0),
                small.mul_add(v.cos(), big),
                Vec3::new(from.cos(), from.sin(), 0.0),
            )
            .unwrap();
            let vertex = topo.add_vertex(Vertex::new(c.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(vertex, vertex, EdgeCurve::Circle(c)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, forward)], true).unwrap())
        };
        let starts = [0.0, FRAC_PI_2, PI, 3.0 * FRAC_PI_2];
        for (v0, v1) in [(3.0_f64, 3.3_f64), (0.3, 0.5), (-0.5, 2.0)] {
            let dv = v1 - v0;
            let area = TAU * small * big.mul_add(dv, small * (v1.sin() - v0.sin()));
            let volume = TAU * small / 3.0
                * (big.mul_add(big, small * small) * (v1.sin() - v0.sin())
                    + big * small * (dv / 2.0 + ((2.0 * v1).sin() - (2.0 * v0).sin()) / 4.0)
                    + small * big * dv);
            for (a, b) in starts
                .iter()
                .flat_map(|&a| starts.iter().map(move |&b| (a, b)))
            {
                let mut topo = Topology::new();
                let low = latitude(&mut topo, v0, a, true);
                let high = latitude(&mut topo, v1, b, false);
                let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small).unwrap();
                let face = Face::new(low, vec![high], FaceSurface::Torus(torus));
                let c =
                    boundary::curved_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
                        .unwrap()
                        .unwrap();
                assert!(
                    (c.area - area).abs() < 1e-9 * area
                        && (c.volume - volume).abs() < 1e-9 * volume.abs(),
                    "v {v0}..{v1}, starts {a} and {b}: area {} of {area}, volume {} of {volume}",
                    c.area,
                    c.volume
                );
            }
        }
    }

    /// The unit cube with its top face's surface swapped for `top`, a
    /// NURBS surface over the same unit square at z = 1.
    fn cube_topped_by(top: brepkit_math::nurbs::surface::NurbsSurface) -> (Topology, SolidId) {
        use brepkit_topology::face::FaceSurface;
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let shell = topo.solid(cube).unwrap().outer_shell();
        let lid = topo
            .shell(shell)
            .unwrap()
            .faces()
            .iter()
            .copied()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                    if normal.z() > 0.5 && (d - 1.0).abs() < 1e-9)
            })
            .unwrap();
        topo.face_mut(lid)
            .unwrap()
            .set_surface(FaceSurface::Nurbs(top));
        (topo, cube)
    }

    /// A top `(100u / (1 + 99u), v, 1)`, a unit square run at a hundredth of
    /// its pace at one side, still holds the cube's 1: the integrals are
    /// halved until they settle, where a fixed rule gave 0.9445.
    #[test]
    fn a_cube_under_a_crowded_rational_top_holds_its_volume() {
        let p = Point3::new;
        let top = brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![p(0.0, 0.0, 1.0), p(0.0, 1.0, 1.0)],
                vec![p(1.0, 0.0, 1.0), p(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0, 1.0], vec![100.0, 100.0]],
        )
        .unwrap();
        let (topo, cube) = cube_topped_by(top);
        let volume = exact_solid_volume(&topo, cube).unwrap().unwrap();
        assert!((volume.abs() - 1.0).abs() < 1e-9, "{volume}");
    }

    /// A flat top whose `v` runs twice as fast past its knot at 0.25: the
    /// cube's side edges, read at eighths, land on that knot line at their
    /// middles, and the cube still holds 1.
    #[test]
    fn a_cube_under_a_top_with_a_sampled_knot_holds_its_volume() {
        let p = Point3::new;
        let top = brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.25, 1.0, 1.0],
            vec![
                vec![p(0.0, 0.0, 1.0), p(0.0, 0.5, 1.0), p(0.0, 1.0, 1.0)],
                vec![p(1.0, 0.0, 1.0), p(1.0, 0.5, 1.0), p(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0; 3]; 2],
        )
        .unwrap();
        let (topo, cube) = cube_topped_by(top);
        let volume = exact_solid_volume(&topo, cube).unwrap().unwrap();
        assert!((volume.abs() - 1.0).abs() < 1e-12, "{volume}");
    }

    /// The flat NURBS patch `(u, v, 1)` over the unit square.
    fn flat_top() -> brepkit_math::nurbs::surface::NurbsSurface {
        let p = Point3::new;
        brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![p(0.0, 0.0, 1.0), p(0.0, 1.0, 1.0)],
                vec![p(1.0, 0.0, 1.0), p(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0; 2]; 2],
        )
        .unwrap()
    }

    /// A closed wire at z = 1 through `corners` in turn, each edge a line
    /// unless `curve` gives the curve of the edge from that corner.
    fn wire_through(
        topo: &mut Topology,
        corners: &[(f64, f64)],
        curve: impl Fn(usize) -> Option<brepkit_topology::edge::EdgeCurve>,
    ) -> brepkit_topology::wire::WireId {
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let vertices: Vec<_> = corners
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 1.0), 1e-7)))
            .collect();
        let edges = (0..vertices.len())
            .map(|k| {
                let (a, b) = (vertices[k], vertices[(k + 1) % vertices.len()]);
                let e = topo.add_edge(Edge::new(a, b, curve(k).unwrap_or(EdgeCurve::Line)));
                OrientedEdge::new(e, true)
            })
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    /// The flat patch with a centred half-unit square hole holds three
    /// quarters of the square, and a third of that for its volume term,
    /// whichever way each of its wires runs.
    #[test]
    fn a_nurbs_face_takes_its_hole_away_whichever_way_it_runs() {
        use brepkit_topology::face::{Face, FaceSurface};
        let square = |lo: f64, hi: f64, ccw: bool| {
            let mut corners = vec![(lo, lo), (hi, lo), (hi, hi), (lo, hi)];
            if !ccw {
                corners.reverse();
            }
            corners
        };
        for (outer_ccw, hole_ccw) in [(true, false), (true, true), (false, true), (false, false)] {
            let mut topo = Topology::new();
            let outer = wire_through(&mut topo, &square(0.0, 1.0, outer_ccw), |_| None);
            let hole = wire_through(&mut topo, &square(0.25, 0.75, hole_ccw), |_| None);
            let face = Face::new(outer, vec![hole], FaceSurface::Nurbs(flat_top()));
            let c = boundary::nurbs_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
                .unwrap()
                .unwrap();
            assert!(
                (c.area - 0.75).abs() < 1e-12 && (c.volume - 0.25).abs() < 1e-12,
                "outer ccw {outer_ccw}, hole ccw {hole_ccw}: area {}, volume {}",
                c.area,
                c.volume
            );
        }
    }

    /// The twisted patch with corners (0, 0, 0), (1, 0, 0), (0, 1, 0) and
    /// (1, 1, 1), bounded by its four sides: read over its domain it holds
    /// what it holds along its boundary, area, volume and moments alike.
    #[test]
    fn a_whole_nurbs_patch_reads_over_its_domain_as_along_its_boundary() {
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let p = Point3::new;
        let twisted = brepkit_math::nurbs::surface::NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![p(0.0, 0.0, 0.0), p(0.0, 1.0, 0.0)],
                vec![p(1.0, 0.0, 0.0), p(1.0, 1.0, 1.0)],
            ],
            vec![vec![1.0; 2]; 2],
        )
        .unwrap();
        let mut topo = Topology::new();
        let corners: Vec<_> = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 0.0),
        ]
        .iter()
        .map(|&c| topo.add_vertex(Vertex::new(c, 1e-7)))
        .collect();
        let edges = (0..4)
            .map(|k| {
                let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = Face::new(wire, vec![], FaceSurface::Nurbs(twisted));
        let about = Vec3::new(0.3, -0.2, 0.5);
        let whole = boundary::nurbs_face_over_domain(&topo, &face, 1.0, about)
            .unwrap()
            .unwrap();
        let along = boundary::nurbs_face_by_boundary(&topo, &face, 1.0, about)
            .unwrap()
            .unwrap();
        let read = |c: &face_integrator::FaceContribution| {
            [
                c.area,
                c.volume,
                c.volume_moment_x,
                c.volume_moment_y,
                c.volume_moment_z,
                c.centroid_x,
                c.centroid_y,
                c.centroid_z,
            ]
        };
        for (x, y) in read(&whole).iter().zip(read(&along)) {
            assert!(
                (x - y).abs() < 1e-10,
                "{:?} for {:?}",
                read(&whole),
                read(&along)
            );
        }
    }

    /// The flat patch cut down to the square from 0.25 to 0.75, or holed by
    /// it, is not its whole surface and is left to its boundary; the whole
    /// patch is read over its domain, a unit square and a third for its
    /// volume term.
    #[test]
    fn only_a_whole_nurbs_patch_is_read_over_its_domain() {
        use brepkit_topology::face::{Face, FaceSurface};
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let middle = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)];
        let mut topo = Topology::new();
        let inner = wire_through(&mut topo, &middle, |_| None);
        let cut = Face::new(inner, vec![], FaceSurface::Nurbs(flat_top()));
        assert!(
            boundary::nurbs_face_over_domain(&topo, &cut, 1.0, origin)
                .unwrap()
                .is_none()
        );
        let outer = wire_through(&mut topo, &unit, |_| None);
        let hole = wire_through(&mut topo, &middle, |_| None);
        let holed = Face::new(outer, vec![hole], FaceSurface::Nurbs(flat_top()));
        assert!(
            boundary::nurbs_face_over_domain(&topo, &holed, 1.0, origin)
                .unwrap()
                .is_none()
        );
        let outer = wire_through(&mut topo, &unit, |_| None);
        let whole = Face::new(outer, vec![], FaceSurface::Nurbs(flat_top()));
        let c = boundary::nurbs_face_over_domain(&topo, &whole, 1.0, origin)
            .unwrap()
            .unwrap();
        assert!(
            (c.area - 1.0).abs() < 1e-12 && (c.volume - 1.0 / 3.0).abs() < 1e-12,
            "area {}, volume {}",
            c.area,
            c.volume
        );
    }

    /// The `(x, y)` of a control point at `v = 0` and of its partner at
    /// `v = 1`.
    type Column = ((f64, f64), (f64, f64));

    /// A patch at z = 1 of degree 1 along `v`, through `columns` of control
    /// points along `u`, weighted by `weights`.
    fn patch_at_one(
        knots_u: Vec<f64>,
        columns: &[Column],
        weights: &[f64],
    ) -> brepkit_math::nurbs::surface::NurbsSurface {
        let p = |(x, y): (f64, f64)| Point3::new(x, y, 1.0);
        brepkit_math::nurbs::surface::NurbsSurface::new(
            knots_u.len() - columns.len() - 1,
            1,
            knots_u,
            vec![0.0, 0.0, 1.0, 1.0],
            columns.iter().map(|&(a, b)| vec![p(a), p(b)]).collect(),
            weights.iter().map(|&w| vec![w; 2]).collect(),
        )
        .unwrap()
    }

    /// Faces the domain's integral cannot vouch for are left to their
    /// boundaries: a patch 1e6 by 10 trimmed half a unit in from its long
    /// sides; a degree-1 side that dips to y = -0.2 at its knot 0.125 under
    /// a straight edge along y = 0; a side that is a curve; and a rational
    /// patch weighted 1e16 over half of it.
    #[test]
    fn a_nurbs_face_not_shown_whole_is_left_to_its_boundary() {
        use brepkit_topology::edge::EdgeCurve;
        use brepkit_topology::face::{Face, FaceSurface};
        let origin = Vec3::new(0.0, 0.0, 0.0);
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let mut topo = Topology::new();
        let mut faces = Vec::new();
        let wide = patch_at_one(
            vec![0.0, 0.0, 1.0, 1.0],
            &[((0.0, 0.0), (0.0, 10.0)), ((1e6, 0.0), (1e6, 10.0))],
            &[1.0, 1.0],
        );
        let trimmed = [(0.0, 0.5), (1e6, 0.5), (1e6, 9.5), (0.0, 9.5)];
        faces.push(Face::new(
            wire_through(&mut topo, &trimmed, |_| None),
            vec![],
            FaceSurface::Nurbs(wide),
        ));
        let dipped = patch_at_one(
            vec![0.0, 0.0, 0.125, 0.25, 1.0, 1.0],
            &[
                ((0.0, 0.0), (0.0, 1.0)),
                ((0.125, -0.2), (0.125, 1.0)),
                ((0.25, 0.0), (0.25, 1.0)),
                ((1.0, 0.0), (1.0, 1.0)),
            ],
            &[1.0; 4],
        );
        faces.push(Face::new(
            wire_through(&mut topo, &unit, |_| None),
            vec![],
            FaceSurface::Nurbs(dipped),
        ));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)],
            vec![1.0, 1.0],
        )
        .unwrap();
        faces.push(Face::new(
            wire_through(&mut topo, &unit, |k| {
                (k == 1).then(|| EdgeCurve::NurbsCurve(line.clone()))
            }),
            vec![],
            FaceSurface::Nurbs(flat_top()),
        ));
        let heavy = patch_at_one(
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            &[
                ((0.0, 0.0), (0.0, 1.0)),
                ((0.5, 0.0), (0.5, 1.0)),
                ((1.0, 0.0), (1.0, 1.0)),
            ],
            &[1.0, 1e16, 1e16],
        );
        faces.push(Face::new(
            wire_through(&mut topo, &unit, |_| None),
            vec![],
            FaceSurface::Nurbs(heavy),
        ));
        for (k, face) in faces.iter().enumerate() {
            assert!(
                boundary::nurbs_face_over_domain(&topo, face, 1.0, origin)
                    .unwrap()
                    .is_none(),
                "face {k} read over its domain"
            );
        }
    }

    /// The flat unit patch whose `u` knots run 0, 1 - 2^-43, 1, its last span
    /// that sliver of `u` carrying half the square: read over its domain, a
    /// unit square and a third for its volume term.
    #[test]
    fn a_whole_nurbs_patch_keeps_a_sliver_span_by_its_end() {
        use brepkit_topology::face::{Face, FaceSurface};
        let sliver = patch_at_one(
            vec![0.0, 0.0, 1.0 - 2.0_f64.powi(-43), 1.0, 1.0],
            &[
                ((0.0, 0.0), (0.0, 1.0)),
                ((0.5, 0.0), (0.5, 1.0)),
                ((1.0, 0.0), (1.0, 1.0)),
            ],
            &[1.0; 3],
        );
        let mut topo = Topology::new();
        let unit = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let face = Face::new(
            wire_through(&mut topo, &unit, |_| None),
            vec![],
            FaceSurface::Nurbs(sliver),
        );
        let c = boundary::nurbs_face_over_domain(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (c.area - 1.0).abs() < 1e-9 && (c.volume - 1.0 / 3.0).abs() < 1e-9,
            "area {}, volume {}",
            c.area,
            c.volume
        );
    }

    /// The flat patch with its side at u = 1 the rational line
    /// `(1, 100t / (1 + 99t), 1)`, weights 1 and 100, which runs at a
    /// hundredth of its pace at one end: a third for its volume term, read
    /// from the curve's own velocity.
    #[test]
    fn a_nurbs_face_reads_a_crowded_edge_by_its_velocity() {
        use brepkit_topology::edge::EdgeCurve;
        use brepkit_topology::face::{Face, FaceSurface};
        let rational = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![Point3::new(1.0, 0.0, 1.0), Point3::new(1.0, 1.0, 1.0)],
            vec![1.0, 100.0],
        )
        .unwrap();
        let mut topo = Topology::new();
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let outer = wire_through(&mut topo, &corners, |k| {
            (k == 1).then(|| EdgeCurve::NurbsCurve(rational.clone()))
        });
        let face = Face::new(outer, vec![], FaceSurface::Nurbs(flat_top()));
        let c = boundary::nurbs_face_by_boundary(&topo, &face, 1.0, Vec3::new(0.0, 0.0, 0.0))
            .unwrap()
            .unwrap();
        assert!(
            (c.area - 1.0).abs() < 1e-12 && (c.volume - 1.0 / 3.0).abs() < 1e-12,
            "area {}, volume {}",
            c.area,
            c.volume
        );
    }
}

//! Prisms over notched outlines (an L, a V, a W) intersected with a cylinder
//! whose wall crosses all their side faces: the section loop on the wall
//! takes the outline's shape, its centroid in a notch, and the piece of wall
//! it bounds is kept.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 21.0;

const L: &[(f64, f64)] = &[
    (0.0, 0.0),
    (8.0, 0.0),
    (8.0, 2.0),
    (2.0, 2.0),
    (2.0, 7.0),
    (0.0, 7.0),
];

const V: &[(f64, f64)] = &[
    (0.0, 7.0),
    (4.0, 0.0),
    (8.0, 7.0),
    (6.0, 7.0),
    (4.0, 3.0),
    (2.0, 7.0),
];

const W: &[(f64, f64)] = &[
    (0.0, 7.0),
    (1.5, 0.0),
    (3.0, 0.0),
    (4.0, 5.0),
    (5.0, 0.0),
    (6.5, 0.0),
    (8.0, 7.0),
    (6.6, 7.0),
    (5.6, 2.5),
    (4.6, 7.0),
    (3.4, 7.0),
    (2.4, 2.5),
    (1.4, 7.0),
];

/// The volume of the prism over `outline` (in x, z) that lies in the
/// cylinder: the integral of sqrt(R^2 - x^2) over the outline, by Green's
/// theorem with F(x) = integral of sqrt(R^2 - t^2) dt from 0 to x.
fn exact_volume(outline: &[(f64, f64)]) -> f64 {
    let r2 = RADIUS * RADIUS;
    let f = |x: f64| x.mul_add((r2 - x * x).sqrt(), r2 * (x / RADIUS).asin()) / 2.0;
    let h = |x: f64| x.mul_add(f(x), (r2 - x * x).powf(1.5) / 3.0);
    let mut sum = 0.0;
    for (i, &(x0, z0)) in outline.iter().enumerate() {
        let (x1, z1) = outline[(i + 1) % outline.len()];
        sum += if (x1 - x0).abs() < 1e-12 {
            f(x0) * (z1 - z0)
        } else {
            (z1 - z0) * (h(x1) - h(x0)) / (x1 - x0)
        };
    }
    sum.abs()
}

/// A prism 22 deep along -y from y = 0 over `outline`, posed by `pose`.
fn prism(topo: &mut Topology, outline: &[(f64, f64)], pose: Pose) -> SolidId {
    let points: Vec<Point3> = outline
        .iter()
        .map(|&(x, z)| Point3::new(x, 0.0, z))
        .collect();
    let wire = make_polygon_wire(topo, &points, 1e-7).unwrap();
    let face = make_face_from_wire(topo, wire).unwrap();
    let solid = extrude(topo, face, Vec3::new(0.0, -1.0, 0.0), 22.0).unwrap();
    match pose {
        Pose::Turned(angle) => {
            transform_solid(topo, solid, &Mat4::rotation_z(angle.to_radians())).unwrap();
            solid
        }
        Pose::Mirrored => mirror(
            topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap(),
    }
}

#[derive(Clone, Copy, Debug)]
enum Pose {
    /// Turned about the cylinder's axis by this many degrees.
    Turned(f64),
    /// Mirrored across the plane x = 0.
    Mirrored,
}

/// The cylinder's seam runs along +x and its parameter origin along +y: 80
/// and 170 degrees put the outline across each. Across the seam the wall
/// comes out in two pieces, one either side of it.
const POSES: &[Pose] = &[
    Pose::Turned(0.0),
    Pose::Turned(37.0),
    Pose::Turned(80.0),
    Pose::Turned(170.0),
    Pose::Turned(-123.0),
    Pose::Mirrored,
];

/// A cylinder of radius 21 standing on z = -30, intersected with the prism
/// over `outline` in every pose: exact (no mesh fallback), valid, made of the
/// prism's planes and the cylinder's wall only, and holding the exact volume.
fn assert_intersect_is_exact(outline: &[(f64, f64)]) {
    let expected = exact_volume(outline);
    for &pose in POSES {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, outline, pose);
        let cylinder = make_cylinder(&mut topo, RADIUS, 100.0).unwrap();
        transform_solid(&mut topo, cylinder, &Mat4::translation(0.0, 0.0, -30.0)).unwrap();
        let before = mesh_fallback_count();
        let result = boolean(&mut topo, BooleanOp::Intersect, solid, cylinder).unwrap();
        assert_eq!(
            mesh_fallback_count(),
            before,
            "{pose:?}: the intersect stays exact"
        );
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{pose:?}: {:?}", report.issues);
        let faces = solid_faces(&topo, result).unwrap();
        let walls = faces
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        let planes = faces
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
            .count();
        assert_eq!(
            planes + walls,
            faces.len(),
            "{pose:?}: only planes and the cylinder's wall"
        );
        assert!(walls > 0, "{pose:?}: the cylinder's wall is kept");
        let volume = solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            ((volume - expected) / expected).abs() < 1e-6,
            "{pose:?}: volume {volume}, exact {expected}"
        );
    }
}

#[test]
fn an_l_prism_intersects_a_cylinder_exactly() {
    assert_intersect_is_exact(L);
}

#[test]
fn a_v_prism_intersects_a_cylinder_exactly() {
    assert_intersect_is_exact(V);
}

#[test]
fn a_w_prism_intersects_a_cylinder_exactly() {
    assert_intersect_is_exact(W);
}

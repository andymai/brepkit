//! Offsets and shells that keep the input's topology are built exact: every
//! face stays on its surface's analytic offset, the solid is valid and
//! meshes watertight, and it measures its closed-form volume, however the
//! input is placed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::offset_v2::offset_solid_v2;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::shell_op::shell;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

type Make = fn(&mut Topology) -> SolidId;

/// A frustum's volume from its height and end radii.
fn frustum(h: f64, a: f64, b: f64) -> f64 {
    PI * h / 3.0 * b.mul_add(b, a.mul_add(a, a * b))
}

/// `make_cone(5, 2, 10)`'s wall moves `1 / cos` of its slant per unit of
/// offset, measured across the axis.
fn cone_shift() -> f64 {
    1.09_f64.sqrt()
}

/// Upright, turned and moved, and mirrored.
fn poses() -> [(&'static str, Mat4); 3] {
    [
        ("upright", Mat4::identity()),
        (
            "turned",
            Mat4::translation(1.5, -2.0, 0.7) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.4),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// Every check the tests share: valid, watertight, no face refit as NURBS,
/// and the volume within `1e-9` of `truth`.
fn assert_exact(topo: &Topology, solid: SolidId, truth: f64, label: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    for f in solid_faces(topo, solid).unwrap() {
        let surface = topo.face(f).unwrap().surface();
        assert!(
            !matches!(surface, FaceSurface::Nurbs(_)),
            "{label}: a face was refit as NURBS"
        );
    }
    let volume = solid_volume(topo, solid, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "{label}: volume {volume}, truth {truth}"
    );
}

/// The planar face whose outward normal points along `up` and lies furthest
/// along it.
fn top_face(topo: &Topology, solid: SolidId, up: Vec3) -> FaceId {
    let mut best: Option<(f64, FaceId)> = None;
    for f in solid_faces(topo, solid).unwrap() {
        let face = topo.face(f).unwrap();
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let n = if face.is_reversed() {
                -*normal
            } else {
                *normal
            };
            if n.dot(up) > 0.9 {
                let height = d * normal.dot(up);
                if best.is_none_or(|(h, _)| height > h) {
                    best = Some((height, f));
                }
            }
        }
    }
    best.unwrap().1
}

/// A box, a cylinder, a frustum, a sphere and a torus offset one unit in and
/// out, upright, turned and mirrored: each offset is exact.
#[test]
fn primitives_offset_exactly() {
    let k = cone_shift();
    let cases: [(&str, Make, f64, f64); 5] = [
        (
            "box",
            |t| make_box(t, 10.0, 10.0, 10.0).unwrap(),
            512.0,
            1728.0,
        ),
        (
            "cylinder",
            |t| make_cylinder(t, 5.0, 10.0).unwrap(),
            PI * 16.0 * 8.0,
            PI * 36.0 * 12.0,
        ),
        (
            "frustum",
            |t| make_cone(t, 5.0, 2.0, 10.0).unwrap(),
            frustum(8.0, 4.7 - k, 2.3 - k),
            frustum(12.0, 5.3 + k, 1.7 + k),
        ),
        (
            "sphere",
            |t| make_sphere(t, 5.0, 32).unwrap(),
            4.0 / 3.0 * PI * 64.0,
            4.0 / 3.0 * PI * 216.0,
        ),
        (
            "torus",
            |t| make_torus(t, 6.0, 2.0, 32).unwrap(),
            2.0 * PI * PI * 6.0,
            2.0 * PI * PI * 54.0,
        ),
    ];
    for (name, make, inward, outward) in cases {
        for (pose, place) in poses() {
            for (distance, truth) in [(-1.0, inward), (1.0, outward)] {
                let mut topo = Topology::new();
                let solid = make(&mut topo);
                transform_solid(&mut topo, solid, &place).unwrap();
                let offset = offset_solid_v2(&mut topo, solid, distance).unwrap();
                assert_exact(&topo, offset, truth, &format!("{name} {pose} {distance}"));
            }
        }
    }
}

/// An L of two 10 x 4 x 4 bars (a concave edge along its inner corner) and
/// a 10 x 10 x 4 plate bored through by a radius-2 hole, offset in and out:
/// both keep their topology and are exact, the hole's wall moving the other
/// way to the plate's.
#[test]
fn concave_and_holed_solids_offset_exactly() {
    let ell: Make = |t| {
        let a = make_box(t, 10.0, 4.0, 4.0).unwrap();
        let b = make_box(t, 4.0, 10.0, 4.0).unwrap();
        boolean(t, BooleanOp::Fuse, a, b).unwrap()
    };
    let plate: Make = |t| {
        let b = make_box(t, 10.0, 10.0, 4.0).unwrap();
        let c = make_cylinder(t, 2.0, 10.0).unwrap();
        transform_solid(t, c, &Mat4::translation(5.0, 5.0, -3.0)).unwrap();
        boolean(t, BooleanOp::Cut, b, c).unwrap()
    };
    let cases: [(&str, Make, f64, f64); 4] = [
        ("ell", ell, -1.0, 2.0 * 8.0 * 2.0 * 2.0 - 2.0 * 2.0 * 2.0),
        ("ell", ell, 1.0, 2.0 * 12.0 * 6.0 * 6.0 - 6.0 * 6.0 * 6.0),
        ("plate", plate, -0.5, 9.0 * 9.0 * 3.0 - PI * 2.5 * 2.5 * 3.0),
        (
            "plate",
            plate,
            0.5,
            11.0 * 11.0 * 5.0 - PI * 1.5 * 1.5 * 5.0,
        ),
    ];
    for (name, make, distance, truth) in cases {
        for (pose, place) in poses() {
            let mut topo = Topology::new();
            let solid = make(&mut topo);
            transform_solid(&mut topo, solid, &place).unwrap();
            let offset = offset_solid_v2(&mut topo, solid, distance).unwrap();
            assert_exact(&topo, offset, truth, &format!("{name} {pose} {distance}"));
        }
    }
}

/// A cylinder and a frustum shelled one unit thick with their tops open,
/// and a sphere and a torus hollowed one unit thick, upright, turned and
/// mirrored: each wall stays on the input's analytic surfaces and their
/// offsets, and the solid is exact. The cups used to come back with their
/// inner walls' shared edges running one way (a cylinder cup read 344.45
/// against 333.01).
#[test]
fn curved_solids_shell_exactly() {
    let k = cone_shift();
    let cases: [(&str, Make, bool, f64); 4] = [
        (
            "cylinder",
            |t| make_cylinder(t, 5.0, 10.0).unwrap(),
            true,
            PI * 25.0 * 10.0 - PI * 16.0 * 9.0,
        ),
        (
            "frustum",
            |t| make_cone(t, 5.0, 2.0, 10.0).unwrap(),
            true,
            frustum(10.0, 5.0, 2.0) - frustum(9.0, 4.7 - k, 2.0 - k),
        ),
        (
            "sphere",
            |t| make_sphere(t, 5.0, 32).unwrap(),
            false,
            4.0 / 3.0 * PI * 61.0,
        ),
        (
            "torus",
            |t| make_torus(t, 6.0, 2.0, 32).unwrap(),
            false,
            2.0 * PI * PI * 18.0,
        ),
    ];
    for (name, make, open_top, truth) in cases {
        for (pose, place) in poses() {
            let mut topo = Topology::new();
            let solid = make(&mut topo);
            transform_solid(&mut topo, solid, &place).unwrap();
            let up = place.mul_point(Point3::new(0.0, 0.0, 1.0))
                - place.mul_point(Point3::new(0.0, 0.0, 0.0));
            let open = if open_top {
                vec![top_face(&topo, solid, up)]
            } else {
                vec![]
            };
            let hollow = shell(&mut topo, solid, 1.0, &open).unwrap();
            assert_exact(&topo, hollow, truth, &format!("{name} {pose}"));
        }
    }
}

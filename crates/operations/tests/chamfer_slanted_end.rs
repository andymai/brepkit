//! A prism's bottom edge chamfered with two distances where the prism's end
//! face stands at a slant to the edge. The chamfer is a plane through its
//! two contact lines, and it ends on its line in the slanted end face: each
//! contact runs on (or stops short) to the end face's plane. With the end
//! face `x = A + k z`, the chamfer takes off `d1 d2 (A / 2 + k d2 / 6)`: the
//! prism of `d1 d2 / 2` per unit length to `x = A`, and a tetrahedron past
//! it (or short of it, for `k < 0`).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::chamfer::chamfer_two_on;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 54.5;
const HEIGHT: f64 = 33.0;
const THICK: f64 = 9.0;

/// The prism: its profile in the plane `y = 0` runs along x for `LENGTH`
/// on the floor `z = 0`, then up the end face `x = LENGTH + k z`, and back
/// over the top; extruded `THICK` along y.
fn prism(topo: &mut Topology, k: f64, pose: &Mat4) -> SolidId {
    let corners = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(0.0, 0.0, HEIGHT),
        Point3::new(LENGTH + k * HEIGHT, 0.0, HEIGHT),
        Point3::new(LENGTH, 0.0, 0.0),
    ];
    let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
    let face = make_face_from_wire(topo, wire).unwrap();
    let bar = extrude(topo, face, Vec3::new(0.0, 1.0, 0.0), THICK).unwrap();
    transform_solid(topo, bar, pose).unwrap();
    bar
}

/// The floor's edge along the far side (`y = THICK`, `z = 0`), and the floor.
fn floor_edge(topo: &Topology, bar: SolidId, pose: &Mat4) -> (EdgeId, FaceId) {
    let inverse = pose.inverse().unwrap();
    let local = |p: Point3| inverse.mul_point(p);
    let edge = solid_edges(topo, bar)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let ed = topo.edge(e).unwrap();
            [ed.start(), ed.end()].iter().all(|&v| {
                let p = local(topo.vertex(v).unwrap().point());
                (p.y() - THICK).abs() < 1e-9 && p.z().abs() < 1e-9
            })
        })
        .unwrap();
    let floor = solid_faces(topo, bar)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let wire = topo.wire(topo.face(f).unwrap().outer_wire()).unwrap();
            wire.edges().iter().any(|oe| oe.edge() == edge)
                && wire.edges().iter().all(|oe| {
                    let ed = topo.edge(oe.edge()).unwrap();
                    local(topo.vertex(ed.start()).unwrap().point()).z().abs() < 1e-9
                })
        })
        .unwrap();
    (edge, floor)
}

fn check(k: f64, d1: f64, d2: f64, pose: &Mat4) {
    let label = format!("k {k}, d1 {d1}, d2 {d2}");
    let mut topo = Topology::new();
    let bar = prism(&mut topo, k, pose);
    let (edge, floor) = floor_edge(&topo, bar, pose);
    let before = solid_volume(&topo, bar, 0.01).unwrap();
    let result = chamfer_two_on(&mut topo, bar, &[(edge, floor)], d1, d2)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    assert_eq!(
        solid_faces(&topo, result).unwrap().len(),
        7,
        "{label}: faces"
    );
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = d1 * d2 * (LENGTH / 2.0 + k * d2 / 6.0);
    assert!(
        (removed - want).abs() <= 1e-9 * before,
        "{label}: took off {removed}, want {want}"
    );
}

#[test]
fn a_chamfer_ends_on_a_slanted_end_face() {
    // The end face leaning out over the floor's end, as a prism's next side
    // turning up at 13.6 degrees, and leaning back over the edge.
    for k in [16.0 / 3.862, 0.5, -0.4] {
        for (d1, d2) in [(4.0, 1.865_230_632_619_994_3), (2.0, 5.0)] {
            check(k, d1, d2, &Mat4::identity());
        }
    }
}

#[test]
fn a_turned_prism_chamfers_alike() {
    let pose = Mat4::translation(-89.0, 3.0, 47.5) * Mat4::rotation_y(0.6) * Mat4::rotation_x(-0.9);
    check(16.0 / 3.862, 4.0, 1.865_230_632_619_994_3, &pose);
    check(-0.4, 2.0, 5.0, &pose);
}

//! A bar standing on an L-shaped profile, a straight edge along its arm
//! chamfered where the arm meets the L's wider leg: the edge ends at the
//! L's inside corner. The chamfer's contact on the floor ends inside the
//! floor, over the wider leg's side, and the wedge it takes off leaves a
//! triangle of that side bare. The floor is trimmed back to the corner and
//! the side takes the triangle. The result is valid, watertight, and takes
//! off a prism of d1 d2 / 2 per unit length along the edge.
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

const ARM: f64 = 25.0;
const HEIGHT: f64 = 68.0;
/// The arm runs along x from the inside corner at (0, 6), 2.88 wide; the
/// wider leg stands at x < 0, from y = 1 to the arm's far side.
const PROFILE: [(f64, f64); 6] = [
    (-25.0, 1.0),
    (0.0, 1.0),
    (0.0, 6.0),
    (ARM, 6.0),
    (ARM, 8.88),
    (-25.0, 8.88),
];
const AREA: f64 = 25.0 * 7.88 + ARM * 2.88;

fn l_bar(topo: &mut Topology, pose: &Mat4) -> SolidId {
    let corners: Vec<Point3> = PROFILE
        .iter()
        .map(|&(x, y)| Point3::new(x, y, 0.0))
        .collect();
    let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
    let face = make_face_from_wire(topo, wire).unwrap();
    let bar = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), HEIGHT).unwrap();
    transform_solid(topo, bar, pose).unwrap();
    bar
}

/// The arm's bottom edge along its near side (y = 6, z = 0), and the floor
/// face (z = 0) beside it.
fn arm_edge(topo: &Topology, bar: SolidId, pose: &Mat4) -> (EdgeId, FaceId) {
    let inverse = pose.inverse().unwrap();
    let local = |p: Point3| inverse.mul_point(p);
    let edge = solid_edges(topo, bar)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let ed = topo.edge(e).unwrap();
            [ed.start(), ed.end()].iter().all(|&v| {
                let p = local(topo.vertex(v).unwrap().point());
                (p.y() - 6.0).abs() < 1e-9 && p.z().abs() < 1e-9 && p.x() > -1e-9
            })
        })
        .unwrap();
    let floor = solid_faces(topo, bar)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let face = topo.face(f).unwrap();
            topo.wire(face.outer_wire())
                .unwrap()
                .edges()
                .iter()
                .any(|oe| oe.edge() == edge)
                && topo
                    .wire(face.outer_wire())
                    .unwrap()
                    .edges()
                    .iter()
                    .all(|oe| {
                        let ed = topo.edge(oe.edge()).unwrap();
                        local(topo.vertex(ed.start()).unwrap().point()).z().abs() < 1e-9
                    })
        })
        .unwrap();
    (edge, floor)
}

fn check(pose: &Mat4, d1: f64, d2: f64) {
    let mut topo = Topology::new();
    let bar = l_bar(&mut topo, pose);
    let (edge, floor) = arm_edge(&topo, bar, pose);
    let label = format!("d1 {d1}, d2 {d2}");
    let result = chamfer_two_on(&mut topo, bar, &[(edge, floor)], d1, d2)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    // The floor and the side it runs on lose a strip each, the wider leg's
    // side gains a triangle, and the chamfer adds one face.
    assert_eq!(
        solid_faces(&topo, result).unwrap().len(),
        solid_faces(&topo, bar).unwrap().len() + 1,
        "{label}: faces"
    );
    let want = AREA * HEIGHT - 0.5 * d1 * d2 * ARM;
    let got = solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (got - want).abs() <= 1e-6 * want,
        "{label}: volume {got}, want {want}"
    );
}

#[test]
fn a_chamfer_into_an_inside_corner_bares_a_triangle_of_the_wider_leg() {
    for (d1, d2) in [(2.5, 12.861_385_039_925_768), (1.0, 1.0), (2.0, 30.0)] {
        check(&Mat4::identity(), d1, d2);
    }
}

#[test]
fn a_turned_bar_chamfers_alike() {
    let pose =
        Mat4::translation(-110.16, 3.0, -22.47) * Mat4::rotation_z(0.4) * Mat4::rotation_x(1.1);
    check(&pose, 2.5, 12.861_385_039_925_768);
}

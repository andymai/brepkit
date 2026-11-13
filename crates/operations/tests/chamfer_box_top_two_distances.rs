//! A box's four top edges chamfered with two distances, `a` on the top and
//! `h` down the sides. The chamfers meet in mitres at the top's corners,
//! which the blend engine's chamfer cannot close; a box of planes alone is
//! rebuilt with flat bevels, as the equal chamfer rebuilds it. The top
//! layer of depth `h` is a frustum whose sides step in by `a`, so the
//! chamfer takes off `h (a (W + D) - 4 a^2 / 3)`.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::chamfer::chamfer_two_on;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};

const W: f64 = 68.0;
const D: f64 = 154.0;
const HEIGHT: f64 = 21.0;

fn check(a: f64, h: f64, pose: &Mat4) {
    let label = format!("a {a}, h {h}");
    let mut topo = Topology::new();
    let block = make_box(&mut topo, W, D, HEIGHT).unwrap();
    transform_solid(&mut topo, block, pose).unwrap();
    let inverse = pose.inverse().unwrap();
    let on_top = |p: Point3| (inverse.mul_point(p).z() - HEIGHT).abs() < 1e-9;
    let top = solid_faces(&topo, block)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let wire = topo.wire(topo.face(f).unwrap().outer_wire()).unwrap();
            wire.edges().iter().all(|oe| {
                let e = topo.edge(oe.edge()).unwrap();
                on_top(topo.vertex(e.start()).unwrap().point())
            })
        })
        .unwrap();
    let picks: Vec<_> = solid_edges(&topo, block)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let ed = topo.edge(e).unwrap();
            on_top(topo.vertex(ed.start()).unwrap().point())
                && on_top(topo.vertex(ed.end()).unwrap().point())
        })
        .map(|e| (e, top))
        .collect();
    assert_eq!(picks.len(), 4);
    let before = solid_volume(&topo, block, 0.01).unwrap();
    let result =
        chamfer_two_on(&mut topo, block, &picks, a, h).unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    assert_eq!(
        solid_faces(&topo, result).unwrap().len(),
        10,
        "{label}: faces"
    );
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = h * (a * (W + D) - 4.0 * a * a / 3.0);
    assert!(
        (removed - want).abs() <= 1e-9 * before,
        "{label}: took off {removed}, want {want}"
    );
}

#[test]
fn a_box_top_chamfered_all_round_with_two_distances() {
    for (a, h) in [(3.0, 16.0), (16.0, 3.0), (5.0, 5.0)] {
        check(a, h, &Mat4::identity());
    }
}

#[test]
fn a_turned_box_chamfers_alike() {
    let pose =
        Mat4::translation(-120.0, 0.0, -77.0) * Mat4::rotation_x(-1.2) * Mat4::rotation_z(0.3);
    check(3.0, 16.0, &pose);
}

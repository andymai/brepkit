//! A drilled plate's bottom edges at its two ends chamfered by the blend
//! engine by exactly the plate's thickness: each end face is consumed
//! whole. The chamfer runs from the bottom up to the end's top edge, which
//! it shares with the top face, and the end face goes. The result is valid,
//! watertight, and takes off a prism of `t^2 / 2` per unit length at each
//! end.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{all_blended, chamfer_v2};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};

const LENGTH: f64 = 200.0;
const WIDTH: f64 = 69.0;
const THICK: f64 = 10.0;
const HOLE: f64 = 5.75;

fn check(pose: &Mat4) {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, LENGTH, WIDTH, THICK).unwrap();
    let drill = make_cylinder(&mut topo, HOLE, THICK + 2.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(100.0, 24.0, -1.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
    transform_solid(&mut topo, drilled, pose).unwrap();
    let inverse = pose.inverse().unwrap();
    let ends: Vec<_> = solid_edges(&topo, drilled)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let ed = topo.edge(e).unwrap();
            let (a, b) = (
                inverse.mul_point(topo.vertex(ed.start()).unwrap().point()),
                inverse.mul_point(topo.vertex(ed.end()).unwrap().point()),
            );
            let on_end = |p: Point3| p.x().abs() < 1e-9 || (p.x() - LENGTH).abs() < 1e-9;
            on_end(a) && (a.x() - b.x()).abs() < 1e-9 && a.z().abs() < 1e-9 && b.z().abs() < 1e-9
        })
        .collect();
    assert_eq!(ends.len(), 2);
    let faces = solid_faces(&topo, drilled).unwrap().len();
    let result = chamfer_v2(&mut topo, drilled, &ends, THICK, THICK)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Each end face goes and a chamfer face takes its place.
    assert_eq!(solid_faces(&topo, result).unwrap().len(), faces);
    let want = LENGTH * WIDTH * THICK - PI * HOLE * HOLE * THICK - THICK * THICK * WIDTH;
    let got = solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (got - want).abs() <= 1e-6 * want,
        "volume {got}, want {want}"
    );
}

#[test]
fn a_chamfer_as_wide_as_the_plate_is_thick_takes_the_end_faces() {
    check(&Mat4::identity());
}

#[test]
fn a_turned_plate_chamfers_alike() {
    check(
        &(Mat4::translation(-100.0, -32.0, -10.0) * Mat4::rotation_z(0.5) * Mat4::rotation_y(1.0)),
    );
}

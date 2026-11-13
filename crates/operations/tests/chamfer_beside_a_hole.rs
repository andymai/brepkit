//! A drilled box's straight top edges chamfered: the hole gives the body
//! curved edges, which the flat-bevel rebuild cannot write, so the chamfer
//! goes to the blend engine, which leaves the hole as it is. The result is
//! valid, watertight, and takes off a prism of d²/2 per unit length along
//! each edge.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 100.0;
const WIDTH: f64 = 60.0;
const HEIGHT: f64 = 20.0;
const HOLE: f64 = 5.0;

fn drilled_box(topo: &mut Topology) -> SolidId {
    let block = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let drill = make_cylinder(topo, HOLE, HEIGHT + 2.0).unwrap();
    transform_solid(
        topo,
        drill,
        &Mat4::translation(0.5 * LENGTH, 0.5 * WIDTH, -1.0),
    )
    .unwrap();
    boolean(topo, BooleanOp::Cut, block, drill).unwrap()
}

/// The box's two long top edges.
fn long_top_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            (a.z() - HEIGHT).abs() < 1e-9
                && (b.z() - HEIGHT).abs() < 1e-9
                && ((a - b).length() - LENGTH).abs() < 1e-9
        })
        .collect()
}

#[test]
fn a_drilled_box_s_straight_edges_take_the_chamfer() {
    let turned = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.3);
    for pose in [Mat4::identity(), turned] {
        for d in [2.0, 4.0, 9.5] {
            let mut topo = Topology::new();
            let part = drilled_box(&mut topo);
            let edges = long_top_edges(&topo, part);
            assert_eq!(edges.len(), 2);
            transform_solid(&mut topo, part, &pose).unwrap();
            let result = chamfer(&mut topo, part, &edges, d).unwrap();
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "d {d}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result, 0.1).unwrap();
            assert!(is_watertight(&mesh), "d {d}: mesh not watertight");
            let truth =
                (PI * HOLE * HOLE).mul_add(-HEIGHT, LENGTH * WIDTH * HEIGHT) - LENGTH * d * d;
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-6 * truth,
                "d {d}: volume {volume}, truth {truth}"
            );
        }
    }
}

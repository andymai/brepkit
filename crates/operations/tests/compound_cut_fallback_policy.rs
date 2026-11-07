//! `compound_cut` when no cut stays exact: a ring cut by a rod along y across
//! its tube and by a small box over its top. Cut one by one, the rod's cut
//! falls back to a mesh and the box is then cut against that mesh, which falls
//! back again; the batch must instead be cut once, with a single fallback.
//!
//! Kept in its own file: the fallback counter is process-wide.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn centred_box(topo: &mut Topology, centre: [f64; 3], size: [f64; 3]) -> SolidId {
    let b = make_box(topo, size[0], size[1], size[2]).unwrap();
    let corner = Mat4::translation(
        centre[0] - size[0] / 2.0,
        centre[1] - size[1] / 2.0,
        centre[2] - size[2] / 2.0,
    );
    transform_solid(topo, b, &corner).unwrap();
    b
}

fn ring_and_tools(topo: &mut Topology) -> (SolidId, [SolidId; 2]) {
    let ring = make_torus(topo, 4.0, 1.5, 32).unwrap();
    // A rod along y whose top rulings pass over the tube, which the ruling
    // trace cannot take, and a small box on what is left.
    let rod = make_cylinder(topo, 0.6, 20.0).unwrap();
    let along_y = Mat4::translation(0.5, 10.0, 1.0) * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
    transform_solid(topo, rod, &along_y).unwrap();
    let tools = [rod, centred_box(topo, [0.0, 4.0, 1.0], [1.0, 2.0, 1.0])];
    (ring, tools)
}

#[test]
fn compound_cut_takes_one_fallback_when_no_cut_stays_exact() {
    let mut topo = Topology::new();
    let (ring, tools) = ring_and_tools(&mut topo);
    let before = boolean::mesh_fallback_count();
    let mut cur = ring;
    for t in tools {
        cur = boolean::boolean(&mut topo, BooleanOp::Cut, cur, t).unwrap();
    }
    assert_eq!(
        boolean::mesh_fallback_count() - before,
        2,
        "premise: each tool's own cut must fall back, or this pins nothing"
    );

    let mut topo = Topology::new();
    let (ring, tools) = ring_and_tools(&mut topo);
    let before = boolean::mesh_fallback_count();
    boolean::compound_cut(&mut topo, ring, &tools, BooleanOptions::default()).unwrap();
    assert_eq!(boolean::mesh_fallback_count() - before, 1);
}

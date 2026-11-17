//! Straight edges left in pieces after a fuse join again. Fusing a block
//! onto the end of a thinner one merges their coplanar side and bottom
//! faces, and each long edge along them is left split where the blocks
//! met. The edge merge ran per wire, giving each merged face its own copy
//! of the joined edge; the face across the edge then no longer shared it,
//! so the merge was undone and the pieces stayed: picking such an edge
//! picked one piece, and a chamfer along the pieces failed.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::solid::SolidId;

/// A fuse with the default options: coplanar fragments unified afterwards.
fn fuse(topo: &mut Topology, a: SolidId, b: SolidId) -> SolidId {
    boolean_with_options(topo, BooleanOp::Fuse, a, b, BooleanOptions::default()).unwrap()
}

fn placed_box(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
    let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
    b
}

/// (faces, edges, vertices) of a valid solid of the given volume.
fn counts(topo: &Topology, s: SolidId, volume: f64) -> (usize, usize, usize) {
    let report = validate_solid(topo, s).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let v = solid_volume(topo, s, 0.001).unwrap();
    assert!(
        (v - volume).abs() < 1e-6 * volume,
        "volume {v}, want {volume}"
    );
    (
        solid_faces(topo, s).unwrap().len(),
        solid_edges(topo, s).unwrap().len(),
        solid_vertices(topo, s).unwrap().len(),
    )
}

#[test]
fn a_step_keeps_whole_edges_along_its_sides() {
    // an L in profile: a 2 mm block, a 5 mm block on its end
    let mut topo = Topology::new();
    let a = placed_box(&mut topo, [0.0, 0.0, 0.0], [10.0, 10.0, 2.0]);
    let b = placed_box(&mut topo, [0.0, 10.0, 0.0], [10.0, 20.0, 5.0]);
    let l = fuse(&mut topo, a, b);
    assert_eq!(counts(&topo, l, 200.0 + 500.0), (8, 18, 12));
}

#[test]
fn boxes_fused_side_by_side_make_a_plain_box() {
    let mut topo = Topology::new();
    let a = placed_box(&mut topo, [-100.0, -40.0, -20.0], [100.0, -36.0, 0.0]);
    let b = placed_box(&mut topo, [-100.0, -36.0, -20.0], [100.0, 28.0, 0.0]);
    let c = placed_box(&mut topo, [-100.0, 28.0, -20.0], [100.0, 40.0, 0.0]);
    let ab = fuse(&mut topo, a, b);
    let plate = fuse(&mut topo, ab, c);
    assert_eq!(counts(&topo, plate, 200.0 * 80.0 * 20.0), (6, 12, 8));
}

#[test]
fn a_cross_keeps_its_corners() {
    // two bars crossing flush: every vertex is a real corner
    let mut topo = Topology::new();
    let a = placed_box(&mut topo, [-10.0, -2.0, 0.0], [10.0, 2.0, 3.0]);
    let b = placed_box(&mut topo, [-2.0, -10.0, 0.0], [2.0, 10.0, 3.0]);
    let x = fuse(&mut topo, a, b);
    assert_eq!(
        counts(&topo, x, 2.0 * 20.0 * 4.0 * 3.0 - 4.0 * 4.0 * 3.0),
        (14, 36, 24)
    );
}

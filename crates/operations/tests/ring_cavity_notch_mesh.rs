//! A rod with a ring cavity, less a box whose floor runs through the middle
//! of the ring: the box's floor meets the cavity's torus face in arcs at one
//! tube angle, and the torus face's mesh grid has a row of points there. Each
//! such point split the arc in the torus face alone, so the faces on either
//! side met along different points and the mesh opened.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::tessellate::{boundary_edge_count, is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

fn check(op: BooleanOp, pose: &Mat4, label: &str) {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 6.0, 16.0).unwrap();
    let ring = make_torus(&mut topo, 5.5, 0.32, 32).unwrap();
    transform_solid(&mut topo, ring, &Mat4::translation(0.0, 0.0, 2.0)).unwrap();
    let block = make_box(&mut topo, 3.0, 6.0, 3.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(5.0, -3.0, 2.0)).unwrap();
    for solid in [rod, ring, block] {
        transform_solid(&mut topo, solid, pose).unwrap();
    }
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let hollow = boolean_with_options(&mut topo, BooleanOp::Cut, rod, ring, exact).unwrap();
    let solid = boolean_with_options(&mut topo, op, hollow, block, exact)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    assert!(validate_solid(&topo, solid).unwrap().is_valid(), "{label}");
    for deflection in [0.1, 0.03, 0.01, 0.003, 0.001] {
        let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "{label} at {deflection}: {} boundary edges",
            boundary_edge_count(&mesh)
        );
    }
}

#[test]
fn a_notch_through_the_middle_of_a_ring_cavity_meshes_closed() {
    let mirrored = Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4);
    let turned =
        Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    for op in [BooleanOp::Cut, BooleanOp::Intersect, BooleanOp::Fuse] {
        for (pose, name) in [
            (Mat4::identity(), "upright"),
            (turned, "turned"),
            (mirrored, "mirrored"),
        ] {
            check(op, &pose, &format!("{op:?}, {name}"));
        }
    }
}

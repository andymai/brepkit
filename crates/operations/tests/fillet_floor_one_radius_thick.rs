//! A U-shaped part (two uprights joined by a floor exactly a radius thick),
//! its long bottom edges rounded: on each side face the round's contact runs
//! along the floor's top edge between the uprights, so the side face is left
//! as the two uprights joined by a strip of no width. Each side face splits
//! into its two uprights, and the round meets the floor's top face along
//! that edge: the result is valid, its mesh is watertight, and its volume is
//! the part's less what the rounds take off.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 120.0;
const WIDTH: f64 = 20.0;
const HEIGHT: f64 = 34.0;
/// The gap between the uprights.
const GAP: f64 = 40.0;

/// The part, its floor `floor` thick.
fn u_part(topo: &mut Topology, floor: f64) -> SolidId {
    let block = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let gap = make_box(topo, GAP, WIDTH + 2.0, HEIGHT).unwrap();
    transform_solid(
        topo,
        gap,
        &Mat4::translation(0.5 * (LENGTH - GAP), -1.0, floor),
    )
    .unwrap();
    boolean(topo, BooleanOp::Cut, block, gap).unwrap()
}

/// The part's two edges along its whole length at the bottom.
fn bottom_edges(topo: &Topology, part: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, part)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            a.z().abs() < 1e-9 && b.z().abs() < 1e-9 && ((a - b).length() - LENGTH).abs() < 1e-9
        })
        .collect()
}

/// Rounds the bottom edges of the part with a floor `floor` thick by `r`,
/// in the pose `pose`, and checks the result.
fn round(floor: f64, r: f64, pose: &Mat4, faces: usize) {
    let mut topo = Topology::new();
    let part = u_part(&mut topo, floor);
    let edges = bottom_edges(&topo, part);
    assert_eq!(edges.len(), 2);
    transform_solid(&mut topo, part, pose).unwrap();
    let result = fillet_v2(&mut topo, part, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), faces);
    for deflection in [0.5, 0.05] {
        let mesh = tessellate_solid(&topo, result.solid, deflection).unwrap();
        assert!(
            is_watertight(&mesh),
            "floor {floor}, r {r}: mesh not watertight at {deflection}"
        );
    }
    let part_volume = LENGTH * WIDTH * HEIGHT - GAP * WIDTH * (HEIGHT - floor);
    let truth = 2.0f64.mul_add(-(LENGTH * (1.0 - PI / 4.0) * r * r), part_volume);
    let volume = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "floor {floor}, r {r}: volume {volume}, truth {truth}"
    );
}

#[test]
fn a_floor_one_radius_thick_splits_each_side_into_its_uprights() {
    let turned = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.3);
    for pose in [Mat4::identity(), turned] {
        // Two side faces become four: 10 faces, two rounds.
        round(2.0, 2.0, &pose, 14);
    }
}

#[test]
fn a_floor_thicker_than_the_radius_keeps_each_side_whole() {
    round(2.0, 1.5, &Mat4::identity(), 12);
}

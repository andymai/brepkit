//! A T-joint of two equal cylinders: a bar lying across a rod's top, its axis
//! through the rod's axis at the rod's top, so the bar's lower half sits in the
//! rod and the rod's top disc is cut by the bar's wall along a chord.
//!
//! On the rod's wall the section is a pair of half ellipses meeting at the
//! two points where the walls touch, both on the rod's rim. The patch of the
//! wall between them runs rim to rim, has no line in its outline and lies
//! clear of any region it could be a hole of, so it has to be told an outer
//! loop by its winding. Read as a hole, it was dropped with its twin and the
//! boolean fell back to a mesh whenever the bar's seam turned to the side or
//! the bottom.
//!
//! With the rod's seam turned a quarter, one touch point is the seam vertex
//! and the other halves the rim: the wall's rim is cut there, and the top
//! disc's rim, whole, has to be cut there too, or the two rims never pair.
//!
//! The shared volume is the half Steinmetz solid, 8 r^3 / 3.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

const R: f64 = 10.0;
const ROD: f64 = 22.0;

fn check(op: BooleanOp, rod_turn: f64, bar_turn: f64) {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, R, ROD).unwrap();
    transform_solid(&mut topo, rod, &Mat4::rotation_z(rod_turn)).unwrap();
    let bar = make_cylinder(&mut topo, R, 2.0 * R).unwrap();
    transform_solid(&mut topo, bar, &Mat4::rotation_z(bar_turn)).unwrap();
    transform_solid(&mut topo, bar, &Mat4::rotation_y(FRAC_PI_2)).unwrap();
    transform_solid(&mut topo, bar, &Mat4::translation(-R, 0.0, ROD)).unwrap();

    let pose = format!("{op:?}, rod turned {rod_turn:.2}, bar turned {bar_turn:.2}");
    let before = mesh_fallback_count();
    let result = boolean(&mut topo, op, rod, bar).unwrap();
    assert_eq!(mesh_fallback_count(), before, "{pose}: fell back");

    let mut uses: HashMap<usize, usize> = HashMap::new();
    for fid in solid_faces(&topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
    }
    assert!(
        uses.values().all(|&n| n == 2),
        "{pose}: free or over-shared edges"
    );

    let shared = 8.0 * R * R * R / 3.0;
    let (rod_v, bar_v) = (PI * R * R * ROD, PI * R * R * 2.0 * R);
    let want = match op {
        BooleanOp::Fuse => rod_v + bar_v - shared,
        BooleanOp::Cut => rod_v - shared,
        BooleanOp::Intersect => shared,
    };
    let got = solid_volume(&topo, result, 1e-3).unwrap();
    assert!(
        ((got - want) / want).abs() < 1e-6,
        "{pose}: volume {got}, closed form {want}"
    );
}

const BAR_TURNS: [f64; 4] = [0.0, 1.0, FRAC_PI_2, PI];

#[test]
fn a_t_joint_of_equal_cylinders_is_exact_in_every_seam_pose() {
    for op in [BooleanOp::Fuse, BooleanOp::Cut, BooleanOp::Intersect] {
        for rod_turn in [0.0, 1.0, FRAC_PI_2, PI] {
            for bar_turn in BAR_TURNS {
                check(op, rod_turn, bar_turn);
            }
        }
    }
}

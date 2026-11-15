//! Captured operands of three calls that fail in the brepjs playground
//! examples (#2372), each a ready repro for its own issue. Each test states
//! what the call must give, and is ignored until its issue closes.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::blend_ops::{all_blended, chamfer_v2};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

fn load(name: &str, topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn assert_holds(topo: &Topology, solid: SolidId) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.05).unwrap()),
        "mesh not watertight"
    );
}

/// The exact fuse, which must be a valid solid that keeps a curved face.
fn fuse_exactly(topo: &mut Topology, a: SolidId, b: SolidId) -> SolidId {
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let fused = boolean_with_options(topo, BooleanOp::Fuse, a, b, exact).unwrap();
    assert_holds(topo, fused);
    assert!(
        solid_faces(topo, fused).unwrap().iter().any(|&f| !topo
            .face(f)
            .unwrap()
            .surface()
            .is_planar()),
        "no curved face is left"
    );
    fused
}

/// `mars-rover`: a wheel's thin shell (radii 249.25 and 250) with four other
/// parts of the wheel fused into it, and a grouser: a box turned 9 degrees
/// about the radius, that passes through the shell's wall.
#[test]
#[ignore = "TODO: enable when #2487 closes"]
fn a_grouser_fuses_onto_a_wheel() {
    let mut topo = Topology::new();
    let wheel = load("mars_rover_wheel.bin", &mut topo);
    let grouser = load("mars_rover_grouser.bin", &mut topo);
    fuse_exactly(&mut topo, wheel, grouser);
}

/// `robot-arm`: the turret's cap, a cylinder with a cone on it, and a yoke
/// whose arms have rounded edges. The yoke's flat bottom lies in the plane
/// where the cylinder meets the cone.
#[test]
#[ignore = "TODO: enable when #2488 closes"]
fn a_yoke_with_rounded_arms_fuses_onto_the_turrets_cap() {
    let mut topo = Topology::new();
    let cap = load("robot_arm_shoulder_cap.bin", &mut topo);
    let yoke = load("robot_arm_shoulder_yoke.bin", &mut topo);
    fuse_exactly(&mut topo, cap, yoke);
}

/// `pocket-watch`: a bridge plate whose top outline, at z = 3.15, is 6 lines
/// and 7 circle arcs.
#[test]
#[ignore = "TODO: enable when #2490 closes"]
fn a_plates_top_outline_of_lines_and_arcs_takes_a_chamfer() {
    let mut topo = Topology::new();
    let plate = load("pocket_watch_bridge.bin", &mut topo);
    let outline: Vec<_> = solid_edges(&topo, plate)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - 3.15).abs() < 1e-9)
        })
        .collect();
    assert_eq!(outline.len(), 13);
    let chamfered = chamfer_v2(&mut topo, plate, &outline, 0.1, 0.1)
        .and_then(all_blended)
        .unwrap();
    assert_holds(&topo, chamfered);
}

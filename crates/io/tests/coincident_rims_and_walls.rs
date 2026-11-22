//! Coincident walls and rims read from STEP, each case's two operands as
//! written.
//!
//! A pin cut by a part whose hole has the pin's radius: the pin's wall and
//! the hole's lie on one cylinder, their seams at different angles. Each
//! seam cuts the other wall, so the two walls split alike and pair up; a
//! rule that dropped a seam from a wall on the same cylinder left the hole's
//! wall split and the pin's whole, and the cut fell back. Its rims, split
//! where the seam left them, rejoin into whole circles before meshing. With
//! the pin through the hole and out the far side, the pin's wall below the
//! hole is bounded by its end rim, written as two halves of exactly half a
//! turn: lifted vertex by vertex the face trace read a half either way
//! round, found that band winding with no area, and dropped it. With the
//! pin ending inside the hole, the trace read the hole's rim half as
//! heading back and walked the rim out of the face.
//!
//! Two pieces fused along a rim they share, an arc the other piece splits at
//! its middle: the section along that arc is split there too and shares the
//! rim's two edges, where it had stayed one edge beside them and bounded a
//! sliver of no area that no point inside could classify.
//!
//! A pin fused into a blind socket of its radius in a hub's side: the
//! socket's seam cuts the closed curve where the pin enters the hub, on the
//! pin's wall and the hub's alike, and the pin's own seam lies in the
//! socket's wall over the few millimetres from that curve to the floor. The
//! curve's two pieces join the same two points and stay two edges, and the
//! one run backward keeps its own arc rather than the rest of the loop.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

fn exact(topo: &mut Topology, op: BooleanOp, a: SolidId, b: SolidId) -> SolidId {
    let before = mesh_fallback_count();
    let result = boolean(topo, op, a, b).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    result
}

#[test]
fn a_pin_cut_by_a_hole_of_its_radius_stays_whole() {
    let mut topo = Topology::new();
    let pin = load(&mut topo, "pin_in_same_radius_hole_pin.step");
    let part = load(&mut topo, "pin_in_same_radius_hole_part.step");
    let want = solid_volume(&topo, pin, 0.001).unwrap();
    let result = exact(&mut topo, BooleanOp::Cut, pin, part);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}

#[test]
fn a_pin_through_a_hole_of_its_radius_stays_whole() {
    let mut topo = Topology::new();
    let pin = load(&mut topo, "pin_through_same_radius_hole_pin.step");
    let part = load(&mut topo, "pin_through_same_radius_hole_part.step");
    let want = solid_volume(&topo, pin, 0.001).unwrap();
    let result = exact(&mut topo, BooleanOp::Cut, pin, part);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}

#[test]
fn a_pin_ending_in_a_hole_of_its_radius_stays_whole() {
    let mut topo = Topology::new();
    let pin = load(&mut topo, "pin_ending_in_same_radius_hole_pin.step");
    let part = load(&mut topo, "pin_ending_in_same_radius_hole_part.step");
    let want = solid_volume(&topo, pin, 0.001).unwrap();
    let result = exact(&mut topo, BooleanOp::Cut, pin, part);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}

#[test]
fn pieces_sharing_a_split_rim_fuse_exactly() {
    let mut topo = Topology::new();
    let a = load(&mut topo, "split_rim_piece_a.step");
    let b = load(&mut topo, "split_rim_piece_b.step");
    let want = solid_volume(&topo, a, 0.001).unwrap() + solid_volume(&topo, b, 0.001).unwrap();
    let result = exact(&mut topo, BooleanOp::Fuse, a, b);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
}

#[test]
#[ignore = "each wall's seam cuts the shared loop on the other wall, and the loop's piece ends where the fitted curve passes nearest the cut, 6e-7 from the seam's end, so the two never meet; passing a closed loop's cuts on to later faces broke fuses where both walls' seams cross the loop, so it waits for every face's loop cuts to be gathered, at the cut points themselves, before any face is split"]
fn a_pin_fused_into_a_blind_socket_of_its_radius_fills_it() {
    let mut topo = Topology::new();
    let pin = load(&mut topo, "pin_in_blind_socket_pin.step");
    let hub = load(&mut topo, "pin_in_blind_socket_hub.step");
    let want = solid_volume(&topo, pin, 0.001).unwrap() + solid_volume(&topo, hub, 0.001).unwrap();
    let result = exact(&mut topo, BooleanOp::Fuse, pin, hub);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    // The curve where the pin enters the hub is fitted, not exact.
    assert!(
        (volume - want).abs() < 1e-5 * want,
        "volume {volume}, want {want}"
    );
}

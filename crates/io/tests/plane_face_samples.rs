//! Two booleans whose plane faces were split right but sampled or traced
//! wrong, both read from STEP as a file gives them.
//!
//! A plate whose end is notched round a ring of the notch's radius, its
//! outline B-spline curves and the notch's arc, a hole at its far end,
//! fused with the ring. The plate's faces are untouched but for their
//! edges, and a face's sample point was picked off its outline's chords
//! with the holes left out, then moved clear of the hole on chord polygons
//! again: it landed 3e-3 inside the notch's arc, the face read inside the
//! ring and was dropped.
//!
//! A block with a counterbore cut by a part whose top runs across the
//! counterbore's floor through the bore's axis, a tab standing above it.
//! The floor's walk took the floor's two sides of the tab round the bore as
//! one loop, crossing each section twice, and the floor came out whole.
//!
//! Two more that took the first fixes wrong, each falling back to the mesh
//! boolean: a bored peg cut by a rod, whose end face carries the rod's hole
//! twice (in one parity the copies cancelled, and the face's point landed in
//! the hole); and a drum fused into a plate's notch of its own radius, its
//! axis 0.77 off the notch's (its rim's arc in the gap is kept on the plate's
//! sixteen-chord outline, and the fuse depends on it).
//!
//! Two ignored: they pass only where the loop sampler reads a reversed edge
//! by its flag, against the end points it runs between. A fuse whose
//! sections' pcurves wander past their B-spline face's parameter box, and a
//! peg whose B-spline wall closes round its axis but is laid out as open.
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

/// The fallback counter is process-wide: one test at a time reads it.
static COUNTER: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn check(op: BooleanOp, a: &str, b: &str, want: impl Fn(f64, f64) -> f64) {
    let _one_at_a_time = COUNTER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut topo = Topology::new();
    let a = load(&mut topo, a);
    let b = load(&mut topo, b);
    let (va, vb) = (
        solid_volume(&topo, a, 0.001).unwrap(),
        solid_volume(&topo, b, 0.001).unwrap(),
    );
    let before = mesh_fallback_count();
    let result = boolean(&mut topo, op, a, b).unwrap();
    assert_eq!(mesh_fallback_count(), before, "mesh fallback");
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let want = want(va, vb);
    let volume = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (volume - want).abs() < 1e-6 * want,
        "volume {volume}, want {want}"
    );
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "open mesh");
}

#[test]
fn a_plate_notched_round_a_ring_fuses_with_it_exactly() {
    // The ring sits in the notch: the two only touch.
    check(
        BooleanOp::Fuse,
        "notch_ring_fuse_ring.step",
        "notch_ring_fuse_plate.step",
        |ring, plate| ring + plate,
    );
}

#[test]
fn a_counterbore_cut_across_its_floor_is_exact() {
    check(
        BooleanOp::Cut,
        "counterbore_slot_cut_block.step",
        "counterbore_slot_cut_tool.step",
        |_, _| COUNTERBORE_CUT,
    );
}

/// The cut's volume, as brepkit and a reference kernel both measure it (as
/// are the three below).
const COUNTERBORE_CUT: f64 = 94.356_313_854;

#[test]
fn a_bored_peg_cut_by_its_rod_is_exact() {
    check(
        BooleanOp::Cut,
        "bored_peg_cut_peg.step",
        "bored_peg_cut_rod.step",
        |_, _| 2_900.844_9,
    );
}

#[test]
fn a_drum_fused_into_an_offset_notch_is_exact() {
    check(
        BooleanOp::Fuse,
        "offset_rims_fuse_drum.step",
        "offset_rims_fuse_plate.step",
        |_, _| 6_711_891.448_8,
    );
}

#[test]
#[ignore = "falls back: section pcurves outside their B-spline face's parameter box"]
fn a_fuse_with_section_pcurves_against_their_flags_is_exact() {
    check(
        BooleanOp::Fuse,
        "spline_section_fuse_a.step",
        "spline_section_fuse_b.step",
        |_, _| 827_622.216_4,
    );
}

#[test]
#[ignore = "falls back: a closed B-spline wall laid out as open"]
fn a_peg_cut_across_its_closed_spline_wall_is_exact() {
    // brepkit's own measure; a reference kernel measures this result and
    // its own both at 351.794_5, the B-spline wall integrated otherwise.
    check(
        BooleanOp::Cut,
        "peg_cone_cut_peg.step",
        "peg_cone_cut_block.step",
        |_, _| 351.820_608_6,
    );
}

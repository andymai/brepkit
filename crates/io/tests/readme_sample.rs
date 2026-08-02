//! Pins the Rust example in README.md so an API change cannot silently rot it.
//!
//! The calls and their arguments mirror the README fence; the assertions below
//! are extra. If a call or argument here has to change, change the README too.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_io::step::write_step;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_topology::Topology;

#[test]
fn readme_rust_example_compiles_and_runs() -> Result<(), Box<dyn std::error::Error>> {
    let mut topo = Topology::new();

    // Primitives are anchored at the origin, so this cylinder rounds off the
    // block's corner. Use `transform_solid` to place it somewhere else.
    let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
    let cutter = make_cylinder(&mut topo, 5.0, 15.0)?;
    let notched = boolean(&mut topo, BooleanOp::Cut, block, cutter)?;

    // Measure and export
    let vol = solid_volume(&topo, notched, 0.1)?;
    let step = write_step(&topo, &[notched])?;

    // Only the quarter of the cylinder inside the +x+y octant overlaps the block,
    // so the cut takes a quarter-round scallop off one vertical corner. Pin that,
    // not just "something was removed", or the README prose can drift back to
    // claiming a through-hole.
    let box_volume = 30.0 * 20.0 * 10.0;
    let quarter_cylinder = std::f64::consts::PI * 5.0 * 5.0 * 10.0 / 4.0;
    let removed = box_volume - vol;
    assert!(
        (removed - quarter_cylinder).abs() < 0.05 * quarter_cylinder,
        "expected a quarter-cylinder notch (~{quarter_cylinder:.1}), removed {removed:.1}"
    );

    assert!(step.starts_with("ISO-10303-21;"), "not a STEP part 21 file");

    Ok(())
}

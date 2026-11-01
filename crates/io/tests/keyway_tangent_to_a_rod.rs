//! A keyway cut into a rod, captured from brepjs (the last row of #2166):
//! a slot 8 wide from z = 45 to 75 whose ends curve in as half circles of
//! radius 4, extruded from x = 8.5 to 12.5 against a rod of radius 12.5, so
//! the slot's outer face lies in a plane tangent to the rod along y = 0.
//! `keyway_tool_nudged.bin` draws the end arcs through mid points 4.9e-16
//! off the axis, as brepjs's round-off leaves them.
//!
//! Two faults broke the cut, depending on where the rod's seam lay:
//! - each rounded end is half a turn of a cylinder, so its vertices sit on
//!   two rulings half a turn apart, and the window its splitter laid it out
//!   in was picked by rounding, sometimes on the half it does not cover;
//! - the outer face's rim arcs, which touch the rod only where they cross
//!   the tangent line, were taken as lying in it and split it off the rod.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

/// The rod less the slot: for each `|y| <= 4` the slot takes the rod from
/// x = 8.5 out to its wall over `30 - 2 sqrt(16 - y^2)` of z, integrated with
/// `y = 4 sin t`.
const CUT_VOLUME: f64 = 58_189.792_027_939_23;

#[test]
fn a_keyway_tangent_to_a_rod_cuts_exactly_wherever_its_seam_lies() {
    for tool in ["keyway_tool.bin", "keyway_tool_nudged.bin"] {
        for spin in [0.0_f64, 1.0, 10.0, 17.0, 30.0, 90.0, 180.0, 359.0] {
            let mut topo = Topology::new();
            let rod = std::fs::read(fixture("keyway_rod.bin")).unwrap();
            let key = std::fs::read(fixture(tool)).unwrap();
            let rod = deserialize_solid(&rod, &mut topo).unwrap();
            let key = deserialize_solid(&key, &mut topo).unwrap();
            // Turning the rod about its axis moves only its seam.
            transform_solid(&mut topo, rod, &Mat4::rotation_z(spin.to_radians())).unwrap();

            let before = mesh_fallback_count();
            let cut = boolean(&mut topo, BooleanOp::Cut, rod, key).unwrap();
            let case = format!("{tool}, seam at {spin} degrees");
            assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");
            assert_eq!(solid_faces(&topo, cut).unwrap().len(), 8, "{case}");
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let volume = solid_volume(&topo, cut, 0.001).unwrap();
            assert!(
                (volume - CUT_VOLUME).abs() < 1e-6 * CUT_VOLUME,
                "{case}: volume {volume}"
            );
            let classify =
                |x, y, z| classify_point(&topo, cut, Point3::new(x, y, z), 0.0, 1e-7).unwrap();
            // In the slot, in the slot beside a rounded end, and out in the
            // rod past the end's bulge.
            for (x, y, z) in [(11.0, 0.0, 60.0), (12.4, 0.0, 60.0), (11.0, 3.9, 46.0)] {
                assert_eq!(
                    classify(x, y, z),
                    PointClassification::Outside,
                    "{case}: ({x}, {y}, {z})"
                );
            }
            for (x, y, z) in [(5.0, 0.0, 60.0), (11.0, 0.0, 46.0), (12.4, 0.0, 47.0)] {
                assert_eq!(
                    classify(x, y, z),
                    PointClassification::Inside,
                    "{case}: ({x}, {y}, {z})"
                );
            }
        }
    }
}

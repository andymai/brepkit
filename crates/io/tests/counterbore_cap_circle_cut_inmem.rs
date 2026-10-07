//! Captured-operand pin for a bin base's magnet and screw counterbore,
//! captured from the gridfinity layout tool's split export ("exported STL
//! pieces with magnet+screw base have full geometry"), where the base is cut
//! by 48 of these tools at once.
//!
//! The tool's underside, flush with the base's, is an annulus and a disc
//! sharing the screw hole's r = 1.5 circle. The base's underside is split by
//! that circle first, and the magnet pocket's r = 3.25 rim circle then has to
//! go to the piece holding it: the underside with the r = 1.5 hole, whose
//! hole the circle's centre lies in.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

#[test]
fn a_counterbore_flush_with_the_base_underside_cuts_exactly() {
    let mut topo = Topology::new();
    let base = load(&mut topo, "magnet_screw_base.bin");
    let tool = load(&mut topo, "magnet_screw_counterbore.bin");

    let before = mesh_fallback_count();
    let cut = boolean(&mut topo, BooleanOp::Cut, base, tool).unwrap();
    assert_eq!(mesh_fallback_count(), before, "the cut fell back to a mesh");
    assert!(validate_solid(&topo, cut).unwrap().is_valid());

    let volume = |s: SolidId| oriented_solid_volume(&topo, s, 0.01).unwrap();
    let removed = volume(base) - volume(cut);
    assert!(
        (removed - volume(tool)).abs() <= 0.01,
        "the cut removes {removed}, the tool is {}",
        volume(tool)
    );

    // The magnet pocket, the screw hole above it, the annulus between the two
    // rims on the underside, and the base around the pocket.
    for (p, inside) in [
        (Point3::new(-118.0, -34.0, -4.7), false),
        (Point3::new(-116.0, -34.0, -4.7), false),
        (Point3::new(-118.0, -34.0, -1.0), false),
        (Point3::new(-121.6, -34.0, -4.7), true),
        (Point3::new(-118.0, -38.0, -4.7), true),
    ] {
        assert_eq!(
            classify_point(&topo, cut, p, 0.01, 1e-7).unwrap() == PointClassification::Inside,
            inside,
            "cut at {p:?}"
        );
    }
}

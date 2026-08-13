//! Regression for the 4x4 bin base fuse with an insert pocket at a four-foot
//! junction (the tool's "4x4 with everything enabled" row, #1517): the bin
//! body's underside is one large plate whose center insert pocket opens as a
//! circular hole exactly where four feet meet, so each adjacent foot's top
//! gets a quarter-circle bite whose arc crosses the foot's own boundary.
//! Three defects stacked here:
//!
//! - the pocket rim entered the plate's arrangement as pave-image boundary
//!   pieces, and the wire builder traced the in-hole island as a MATERIAL
//!   region — a phantom disc over the pocket mouth;
//! - the disconnected island trace meant the plate's surviving piece emerged
//!   with no inner wire, over-covering the opening (the walker cannot reach
//!   an island from the containing cell's angular walk);
//! - the SD geometric group (one plate, sixteen coincident foot tops) demoted
//!   every same-rank member to a within-rank duplicate of one representative,
//!   annihilating the pocket bites — the only members NOT covered by the
//!   plate — and leaving a 33-face open foot.
//!
//! The fuse previously burned a fast analytic failure plus a 26s mesh
//! fallback whose output was non-manifold (over=36); it is exact in ~100ms
//! with the island handed to hole matching, in-hole regions discarded, and
//! uncovered group members exempt from demotion.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

#[test]
fn gridbin4x4_feet_fuse_is_exact_and_strictly_valid() {
    let mut topo = Topology::new();
    let body = load("gridbin4x4_body.bin", &mut topo);
    let feet = load("gridbin4x4_feet.bin", &mut topo);

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let fused = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Fuse,
        body,
        feet,
    )
    .unwrap();
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "the 4x4 base fuse must stay on the exact path"
    );

    // The captured body operand arrives with 34 same-direction shared edges
    // of its own (compartment-divider winding rot at z 6.2..42, far from the
    // z=5 interface; it is still mesh-watertight — flags are not a reliable
    // orientation datum). The fuse must add no orientation issues of its own
    // and be otherwise strictly valid.
    let report = brepkit_operations::validate::validate_solid(&topo, fused).unwrap();
    for issue in &report.issues {
        let inherited = issue
            .description
            .strip_suffix(" shared edges have inconsistent face orientations")
            .and_then(|n| n.parse::<usize>().ok())
            .is_some_and(|n| n <= 34);
        assert!(inherited, "fuse must validate: {issue:?}");
    }

    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, fused, 0.05).unwrap();
    assert!(
        (vol - 193_857.25).abs() < 5.0,
        "exact fuse volume expected ~193857, got {vol}"
    );
}

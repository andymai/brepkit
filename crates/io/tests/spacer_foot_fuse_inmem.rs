//! Regression for the 1u spacer foot fuse (#1570 tail): the spacer body
//! (op3's exact output, carrying a foot-shaped underside recess whose corner
//! bores and crescent ceilings sit entirely inside the foot plate's hole)
//! fused with the baseplate-profile foot block. Three independent defects
//! stacked on this pair:
//!
//! - the face splitter carried sections lying wholly inside the foot plate's
//!   hole into the arrangement whenever the hole weave bailed (a section
//!   crossing the hole's corner ARC cannot be woven), closing four phantom
//!   corner lens loops — garbage faces plus punched holes;
//! - `classify_coincident_coplanar`'s depth probes walked tip → vertex
//!   centroid, which for the annular foot plate lands in its own hole (the
//!   genuinely open foot cavity): every probe read air-air and the buried
//!   ring was declared Outside;
//! - the shell outward-orientation vote sampled edges at raw t ∈ [0,1]
//!   (extrapolating marched NURBS edges to astronomical coordinates) and
//!   signed faces by stored-normal × reversed flag, misreading the body's
//!   legal flag-rewound faces — the assembled shell then classified as a
//!   hole shell and the whole fuse failed assembly.
//!
//! Operands are the tool-side capture of the spacer scenario's final boolean
//! (`{width:1, depth:1, height:1, heightUnitMm:3, spacer:true}`), which
//! previously burned a 12-15s doomed analytic attempt before the mesh
//! fallback (x3 in wasm = the tool's 46s test timeout).

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
fn spacer_foot_fuse_is_exact_and_strictly_valid() {
    let mut topo = Topology::new();
    let body = load("spacer_body.bin", &mut topo);
    let foot = load("spacer_foot.bin", &mut topo);

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let fused = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Fuse,
        body,
        foot,
    )
    .unwrap();
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "the spacer foot fuse must stay on the exact path"
    );

    let report = brepkit_operations::validate::validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "fuse must validate: {report:?}");

    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, fused, 0.01).unwrap();
    assert!(
        (vol - 2401.65).abs() < 0.5,
        "exact fuse volume expected ~2401.65, got {vol}"
    );
}

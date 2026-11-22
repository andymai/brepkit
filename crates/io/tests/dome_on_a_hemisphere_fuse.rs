//! A hemisphere fused flat to flat with a B-spline dome, both read from
//! STEP. The dome's one face is opened along a seam from its rim to its
//! pole, its wire the seam up, the rim, and the seam back down; the result's
//! cleanup took the seam for an out-and-back spur, cut it away and dropped
//! the face, and the fuse came back exact as the hemisphere's lone sphere
//! face, a shell of no volume. The seam is kept now, as a sphere cap's is.
//! Whatever path the fuse takes, an exact result holds both halves' volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
}

#[test]
fn a_dome_fused_to_a_hemisphere_is_never_an_empty_shell() {
    let mut topo = Topology::new();
    let ball = load(&mut topo, "dome_fuse_hemisphere.step");
    let dome = load(&mut topo, "dome_fuse_bspline_dome.step");
    let want =
        solid_volume(&topo, ball, 0.001).unwrap() + solid_volume(&topo, dome, 0.001).unwrap();
    let before = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, ball, dome).unwrap();
    if mesh_fallback_count() == before {
        let volume = solid_volume(&topo, fused, 0.001).unwrap();
        assert!(
            (volume - want).abs() < 1e-3 * want,
            "exact fuse of volume {volume}, want {want}"
        );
    }
}

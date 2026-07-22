//! Captured-operand regression for the gridfinity "2×2 lite + half sockets"
//! export fuse (body + deferred 16-cup lite base).
//!
//! Operands captured from the live gridfinity tool via the `serializeSolid`
//! wasm binding. Adjacent half-socket corner geometry makes several analytic
//! face pairs meet in POINT tangencies; the exact-intersection sampler
//! returns those as one point repeated N times, and the degree-3 NURBS
//! interpolation through the duplicates hit a singular matrix — aborting the
//! whole GFA into a ~65s mesh fallback (the scenario's 190s tool timeout).
//! With point-tangency dedup the fuse is analytic in well under a second.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

#[test]
fn halfsockets_export_fuse_is_analytic_and_watertight() {
    let mut topo = Topology::new();
    let body = load("halfsockets_export_body.bin", &mut topo);
    let base = load("halfsockets_export_deferred.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, body, base).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    let mut curved = 0;
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        if face.surface().type_tag() != "plane" {
            curved += 1;
        }
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&c| c == 1).count();
    assert_eq!(free, 0, "export fuse must be closed, got {free} free edges");

    // The fallback tell: the mesh path re-emits everything as planes
    // (~17k facets); the analytic path keeps the cups' cylinders/cones.
    assert!(
        curved >= 400 && faces.len() < 2500,
        "analytic result expected, got {curved} curved of {} faces",
        faces.len()
    );

    // Volume pins the geometry (analytic reference from the fixed engine).
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - 23241.3).abs() < 10.0,
        "fused volume out of band: got {vol}"
    );
}

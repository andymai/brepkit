//! Captured-operand pin for the 6x4 baseplate pocket stage: one compound cut
//! of the slab against 24 pitch-aligned pocket frustums that TOUCH rim-to-rim
//! (38 grid adjacencies) but never interpenetrate.
//!
//! This is the shape the contact-thin compound-cut shortcut exists for: the
//! tool union's boundary is the concatenation of the tool shells, so the cut
//! runs as ONE arrangement instead of a fuse per tool. Before the shortcut,
//! `fuse_n` computed the welded union, the by-edge-id gate rejected it (a
//! touching union is genuinely non-manifold), and `fuse_cluster` fell back to
//! 23 pairwise accumulator fuses: ~24 wasted GFA runs, 501ms native for a cut
//! the shortcut does in ~31ms with an identical result.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
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
fn bp64_pocket_compound_cut_is_exact_and_analytic() {
    let mut topo = Topology::new();
    let slab = load("bp64_pocket_slab.bin", &mut topo);
    let tools: Vec<SolidId> = (1..=24)
        .map(|i| load(&format!("bp64_pocket_tool_{i}.bin"), &mut topo))
        .collect();

    let before = brepkit_operations::boolean::mesh_fallback_count();
    let result = brepkit_operations::boolean::compound_cut(
        &mut topo,
        slab,
        &tools,
        brepkit_operations::boolean::BooleanOptions::default(),
    )
    .expect("compound cut should succeed");
    assert_eq!(
        brepkit_operations::boolean::mesh_fallback_count(),
        before,
        "pocket compound cut degraded to the mesh fallback"
    );

    let faces = solid_faces(&topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() == "cone")
        .count();
    assert_eq!(cones, 96, "each of the 24 pockets should keep its 4 cones");

    let mut uses: std::collections::HashMap<brepkit_topology::edge::EdgeId, usize> =
        std::collections::HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    let free = uses.values().filter(|&&n| n == 1).count();
    assert_eq!(free, 0, "pocket cut left an open shell");

    // Slab 317520.0 minus what the 24 pockets carve: 139457.4 at 0.01
    // deflection. A dropped pocket moves the reading by ~+7400, far outside
    // this band; fallback-vs-exact is decided by the counter and cone-count
    // asserts above, not by the volume.
    let vol = brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (139_440.0..=139_475.0).contains(&vol),
        "unexpected pocket-cut volume {vol}"
    );
}

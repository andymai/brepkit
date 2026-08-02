//! Fusing two kumiko lattice bands aborts with "open growth shell with 67
//! faces" in 364 ms, from two clean planar operands.
//!
//! This is the smallest reduction of the mitsukude wall-pattern failure, which
//! the matrix files under `divider patterns` but which has nothing to do with
//! dividers: a 2x2x6 bin with the mitsukude wall pattern and NO compartments
//! and NO dividers already exports `bnd=4 nm=9`.
//!
//! How the export gets here (31 booleans, all replayed):
//!   - ops 0-28 are clean, analytic, `free=0 over=0`.
//!   - `op29` cuts the bin body (F=78, 12 cones + 24 cylinders) with 8 lattice
//!     tools and takes **99 s** to return `F=5716 ALL-PLANAR free=2 over=3` —
//!     every curved face gone, the canonical mesh-fallback tell.
//!   - `op30` then fuses that already-broken body and is pure GIGO.
//!
//! `compound_cut` batches by merging its tools into ONE solid first, so those
//! merges are Fuses — and inside op29 they fail **82 times**, 80 of them with
//! `open growth shell`, each dropping to a mesh fallback whose own output is
//! not a closed 2-manifold. This fixture is one of those merges.
//!
//! Same `open growth shell` family as the kumiko corner campaign, so the
//! `divider` and `kumiko` matrix families share machinery rather than being
//! independent. NOTE the parked `fix/kumiko-corner-window-cut` branch does NOT
//! fix this: the kernel that produced these captures was built from that
//! branch and contains all five of its roots.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> brepkit_topology::solid::SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// (free, over, total faces)
fn health(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize, usize) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    (
        uses.values().filter(|&&c| c == 1).count(),
        uses.values().filter(|&&c| c > 2).count(),
        faces.len(),
    )
}

/// Guards the fixture: a replayed capture must be validated before anything is
/// concluded from it, and a magnitude-only volume cannot see an inverted shell.
#[test]
fn lattice_band_operands_are_clean_and_outward() {
    let mut topo = Topology::new();
    for name in ["kumiko_lattice_band_a.bin", "kumiko_lattice_band_b.bin"] {
        let sid = load(name, &mut topo);
        let (free, over, faces) = health(&topo, sid);
        assert_eq!(free, 0, "{name}: operand must be closed, got {free} free");
        assert_eq!(over, 0, "{name}: operand must be manifold, got {over} over");
        assert!(faces > 0, "{name}: operand has no faces");
        assert!(
            brepkit_operations::measure::oriented_solid_volume(&topo, sid, 0.05).unwrap() > 0.0,
            "{name}: operand must be OUTWARD oriented"
        );
    }
}

#[test]
#[ignore = "ready repro: GFA aborts with 'open growth shell with 67 faces' fusing two clean lattice bands"]
fn kumiko_lattice_bands_fuse_closed() {
    let mut topo = Topology::new();
    let a = load("kumiko_lattice_band_a.bin", &mut topo);
    let b = load("kumiko_lattice_band_b.bin", &mut topo);

    let vol_a = brepkit_operations::measure::oriented_solid_volume(&topo, a, 0.01).unwrap();
    let vol_b = brepkit_operations::measure::oriented_solid_volume(&topo, b, 0.01).unwrap();

    let result = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, a, b)
        .expect("analytic fuse should not abort");

    let (free, over, faces) = health(&topo, result);
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");

    // A union is bounded below by the larger operand and above by their sum.
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        vol >= vol_a.max(vol_b) - 1.0 && vol <= vol_a + vol_b + 1.0,
        "fuse volume {vol} outside [{}, {}] for {faces} faces",
        vol_a.max(vol_b),
        vol_a + vol_b
    );
}

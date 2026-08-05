//! Fusing two half-socket pieces of the 3x3 O-shape bin must stay analytic.
//! FIXED by the clean/suspicious conflict re-cast in the ray-cast classifier
//! (votes_from_geoms): the fuse is watertight, keeps the full analytic mix,
//! and its volume equals the operand sum exactly (the halves are
//! complementary and non-overlapping). Historically it aborted with "open
//! growth shell with 45 faces", fell to the mesh fallback, and the fallback
//! output poisoned the next fuse (whose other operand replays as 1022
//! all-planar faces), ending in the export's 8 non-manifold edges (the
//! `3x3 O-shape + half sockets` export-integrity failure).
//!
//! Both operands are clean 49-face socket pieces: one all-analytic
//! (12 cones + 12 cylinders), the other carrying 12 NURBS faces (the
//! quarter-socket pieces the per-cell dispatch produces). The failure
//! reproduces identically on kernels before and after the 2026-08-04/05
//! engine work; the trigger is the tool's generator changes (the #3223-#3227
//! era) reshaping this configuration.
//!
//! Sibling finding, same capture session: the `2x2 mixed-detail per-cell
//! half sockets` export failure (bnd=259) replays ENTIRELY CLEAN through all
//! nine of its booleans — its leak is post-boolean (tessellation/export or
//! an op class the boolean capture does not hook), the "not every scenario
//! failure is a boolean fallback" class.
//!
//! Operands captured 2026-08-05 via the kernel-test boolean monkey-patch
//! (call 006 of the export chain; call 007 is its downstream collateral).
//!
//! BK_OPEN_SHELL characterization: the open 45-face growth shell has
//! POSITIVE volume 3628.8 and is the socket ring itself — alternating plane
//! and NURBS faces at the ring positions (x,y around +-18.7/+-22.45,
//! z=-2.6): the quarter-socket NURBS pieces fail to pair with their plane
//! neighbours after selection.
//!
//! The unpaired edges are operand B's OWN corner rims: the top rims of its
//! four corner cylinders (Circle arcs, len 5.30, at z=0, faces Id 17/19/21/
//! 23 spanning z -1.2..0) plus matching rims at z=0.7 and the slanted 1.21
//! connectors. Operand A has NO geometry at the corner radius (nothing at
//! x or y near 23.85), so this is not a coincident-twin mismatch between
//! operands: the fuse drops or fails to select the B faces ABOVE those rims
//! (or their split pieces), leaving B's rim edges single-used in the
//! assembled shell.
//!
//! ROOT CLASS MEASURED (BK_CLS3 + point oracles): B's chamfer band (local
//! faces 33-40, z 0..0.7, builder ids 83-90) alternates NURBS corner pieces
//! (classified Outside, kept) with thin slanted PLANE strips, and three of
//! the four strips (builder ids 84/86/88) classify Inside AS WHOLE FACES
//! and drop, with ZERO FF sections (has_sections=false for each), unsplit.
//! The independent ray-cast point oracle (POINT_IN) says A=Outside at every
//! strip sample, so the Inside verdicts are MISCLASSIFICATIONS: the builder
//! ray-cast classifier is unstable for thin 45-degree strips near operand
//! A's socket cones (near-tangent crossings flip the parity). Dropping the
//! strips orphans the corner rims below and the connectors above, producing
//! the 45-face open growth shell.
//!
//! PARITY FLIP MEASURED (BK_RAY_POINT): the strip samples land at
//! z=0.6999999999999998, in operand A's rim plane at z=0.7. From there the
//! two horizontal cardinal rays graze A's structure, each reporting ONE
//! crossing and flagging itself suspicious, while the vertical ray is clean
//! with zero crossings; the 2-vote suspicious pair outvoted the 1 clean ray
//! (the re-cast fired only when ALL THREE rays were suspicious). One
//! micrometre off the plane (z=0.701, face 90's sample) the same rays count
//! two clean crossings each. The fix re-casts on that conflict signature
//! and adopts the generic vote only when unanimous (here 0/3: Outside).

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

fn health(topo: &Topology, sid: brepkit_topology::solid::SolidId) -> (usize, usize, usize) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
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
    let over = uses.values().filter(|&&c| c > 2).count();
    (free, over, curved)
}

#[test]
fn oshape_operands_are_clean() {
    let mut topo = Topology::new();
    for name in ["oshape_socket_a.bin", "oshape_socket_b.bin"] {
        let sid = load(name, &mut topo);
        let (free, over, curved) = health(&topo, sid);
        assert_eq!((free, over), (0, 0), "{name} must be closed and manifold");
        assert!(curved > 0, "{name} must keep analytic curved faces");
    }
}

#[test]
fn oshape_socket_fuse_is_analytic_watertight() {
    let mut topo = Topology::new();
    let a = load("oshape_socket_a.bin", &mut topo);
    let b = load("oshape_socket_b.bin", &mut topo);
    let vol_a = brepkit_operations::measure::oriented_solid_volume(&topo, a, 0.05).unwrap();
    let vol_b = brepkit_operations::measure::oriented_solid_volume(&topo, b, 0.05).unwrap();

    let result = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, a, b)
        .expect("analytic fuse should not abort");

    let (free, over, curved) = health(&topo, result);
    assert!(curved > 0, "all-planar output is the mesh-fallback tell");
    assert_eq!(over, 0, "fuse must stay manifold, got {over} over-shared");
    assert_eq!(free, 0, "fuse must be closed, got {free} free edges");

    // The halves are complementary and non-overlapping, so the fuse volume
    // must equal the operand sum (measured 8391.860 = 6151.772 + 2240.088).
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, result, 0.05).unwrap();
    assert!(
        (vol - (vol_a + vol_b)).abs() < 0.01,
        "fuse volume {vol:.3} must equal the operand sum {:.3}",
        vol_a + vol_b
    );
}

//! Captured-operand regression + documented residual for the gridfinity "A2"
//! bin: width 2, depth 2, height 4, 2×1 compartments, stacking lip, honeycomb
//! wall pattern, four u-shape wall cutouts.
//!
//! Operands captured from the live gridfinity tool via the `serializeSolid`
//! wasm binding (byte-exact in-memory arena, no geometry re-derivation):
//!   a2hcomb_prewall_body = body box + stacking lip (shell stage output)
//!   a2hcomb_wcut0/1      = the two wall-cutout cut tools
//!   a2hcomb_pcut0..3     = the four honeycomb wall-pattern cut tools
//!
//! The tool's boolean stage runs, in order: first the wall cutouts
//! `Cut(Cut(body, wcut0), wcut1)`, then the honeycomb pattern as one
//! `cutAllBisect` pass `Cut(result, pcut0..3)`.
//!
//! #923 WIN (regression guard). The wall cutouts are now WATERTIGHT and
//! analytic after #923: `Cut(Cut(prewall_body, wcut0), wcut1)` gives 156
//! faces, 52 curved, free=0. Before #923 the second wall-cut dropped a
//! contained back-wall, leaving the body open (the A2 bin mesh-fell-back).
//! `wallcut_step_is_watertight` locks this in for the exact A2 honeycomb-bin
//! operands (a different capture than `gridfinity_wallcut_seq_inmem`, which
//! uses the same tool but no honeycomb).
//!
//! OPEN RESIDUAL (documented, not yet fixed). The honeycomb pattern cut on the
//! (now watertight) wall-cut body is the NEXT bug, surfaced only now that #923
//! makes the body it cuts watertight. At the raw GFA level, `pcut0` (a
//! full-footprint rounded-rect cutter whose outer wall is COINCIDENT with the
//! body's ±41.75 outer wall) makes the assembler fail outright with
//! AssemblyFailed "no outer shell found (all shells classified as holes)" —
//! the #899/#911/#923 coincident-wall family. `pcut1`/`pcut2`/`pcut3` succeed
//! but leave 15 to 52 free edges (open shells). In-tool this manifests as a
//! multi-minute hang: `cutAllBisect` retries the failed n-way honeycomb cut by
//! recursively bisecting (re-running the expensive NURBS-faced `pcut1` each
//! time), never converging.
//!
//! `honeycomb_cut_residual_documented` pins the CURRENT (broken) behavior so a
//! future fix is forced to update it (flip the asserts to free=0 / Ok).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_algo::gfa;
use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

/// (free edges, over-shared edges) by quantized endpoint position.
fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    let mut occ: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
                *occ.entry(key).or_insert(0) += 1;
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

/// Build the clean wall-cut body the honeycomb pattern cut operates on.
fn build_wallcut_body(topo: &mut Topology) -> SolidId {
    let body = load("a2hcomb_prewall_body.bin", topo);
    let w0 = load("a2hcomb_wcut0.bin", topo);
    let after0 = gfa::boolean(topo, BooleanOp::Cut, body, w0).unwrap();
    let w1 = load("a2hcomb_wcut1.bin", topo);
    gfa::boolean(topo, BooleanOp::Cut, after0, w1).unwrap()
}

#[test]
fn wallcut_step_is_watertight() {
    // #923 regression guard for the A2 honeycomb-bin operands: the wall-cutout
    // sequence the honeycomb pattern then cuts must be watertight + analytic.
    let mut topo = Topology::new();
    let body = build_wallcut_body(&mut topo);
    let (free, over) = edge_health(&topo, body);
    let faces = solid_faces(&topo, body).unwrap().len();
    assert_eq!(
        free, 0,
        "A2 wall-cut body must be watertight (got free={free})"
    );
    assert_eq!(
        over, 0,
        "A2 wall-cut body must be manifold (got over={over})"
    );
    assert_eq!(faces, 156, "A2 wall-cut body face count (got {faces})");
    assert!(
        curved_count(&topo, body) >= 52,
        "A2 wall-cut body preserves the lip cones + corner cylinders"
    );
}

#[test]
fn honeycomb_cut_residual_documented() {
    // DOCUMENTED RESIDUAL: the honeycomb pattern cut on the (now watertight)
    // wall-cut body is broken. pcut0 fails the assembler; the others leave
    // free edges. This pins the current behavior; a fix must flip these.
    let mut topo = Topology::new();
    let body = build_wallcut_body(&mut topo);

    // pcut0: full-footprint coincident-wall cutter → assembler fails.
    let p0 = load("a2hcomb_pcut0.bin", &mut topo);
    let r0 = gfa::boolean(&mut topo, BooleanOp::Cut, body, p0);
    assert!(
        r0.is_err(),
        "RESIDUAL: honeycomb pcut0 currently fails the assembler; \
         a fix should make this Ok + watertight"
    );

    // pcut1..3: succeed but leave the shell open (free edges).
    for tool in [
        "a2hcomb_pcut1.bin",
        "a2hcomb_pcut2.bin",
        "a2hcomb_pcut3.bin",
    ] {
        let mut t = Topology::new();
        let b = build_wallcut_body(&mut t);
        let p = load(tool, &mut t);
        let r = gfa::boolean(&mut t, BooleanOp::Cut, b, p).unwrap();
        let (free, _over) = edge_health(&t, r);
        assert!(
            free > 0,
            "RESIDUAL: honeycomb {tool} currently leaves free edges; \
             a fix should make free=0"
        );
    }
}

//! The mixed-cell top half-socket cut: the FIRST boolean of the
//! mixed-detail export chain (stage-capture call 000). Both operands are
//! orientation-clean by every oracle, yet the cut result carries 3
//! COHERENTLY DOUBLE-FLIPPED cylinder bands — effective surface normal
//! pointing INTO the material with the wire winding flipped to match, so
//! edge-sense pairing (check_shell_orientation) passes while the face is
//! geometrically inside-out. These bands survive the rest of the chain
//! into the body operand of `mixed_socket_tess_inmem.rs` and own all 116
//! unmatched half-edges of that fixture's export-tolerance mesh.
//!
//! Only an OUTWARDNESS oracle sees the class: classify points offset
//! along the effective normal; plus-side Inside with minus-side Outside
//! marks an inverted face (majority vote over spread samples — a single
//! centroid near thin material flips verdicts).
//!
//! Probes: `crates/io/examples/audit_bin.rs` (per-.bin audit and the
//! BOOL_A/BOOL_B native-boolean mode), `fuse_orient.rs`. Captured
//! 2026-08-07 via the kernel-test boolean monkey-patch
//! (mixedSocketStageCapture, tool-side, untracked).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::classify::{PointClassification, classify_point};
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

fn inverted_faces(topo: &Topology, solid: SolidId) -> Vec<String> {
    let (mesh, offsets) = brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
        topo,
        solid,
        0.05,
        10.0_f64.to_radians(),
    )
    .unwrap();
    let faces = solid_faces(topo, solid).unwrap();
    let mut inverted = Vec::new();
    for (fi, &fid) in faces.iter().enumerate() {
        let face = topo.face(fid).unwrap();
        let start = offsets[fi] as usize;
        let end = offsets[fi + 1] as usize;
        if end <= start {
            continue;
        }
        let tris = (end - start) / 3;
        let mut votes_in = 0usize;
        let mut votes_out = 0usize;
        for k in 0..5usize {
            let mid = start + ((tris * (2 * k + 1) / 10).min(tris.saturating_sub(1))) * 3;
            let Some(t) = mesh.indices.get(mid..mid + 3) else {
                continue;
            };
            let (pa, pb, pc) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let centroid = brepkit_math::vec::Point3::new(
                (pa.x() + pb.x() + pc.x()) / 3.0,
                (pa.y() + pb.y() + pc.y()) / 3.0,
                (pa.z() + pb.z() + pc.z()) / 3.0,
            );
            let Some((u, v)) = face.surface().project_point(centroid) else {
                continue;
            };
            let sn = face.surface().normal(u, v);
            let eff = if face.is_reversed() { -1.0 } else { 1.0 };
            let Ok(n_eff) = (sn * eff).normalize() else {
                continue;
            };
            for off in [0.02, 0.05] {
                match (
                    classify_point(topo, solid, centroid + n_eff * off, 0.01, 1e-6),
                    classify_point(topo, solid, centroid - n_eff * off, 0.01, 1e-6),
                ) {
                    (Ok(PointClassification::Inside), Ok(PointClassification::Outside)) => {
                        votes_in += 1;
                    }
                    (Ok(PointClassification::Outside), Ok(PointClassification::Inside)) => {
                        votes_out += 1;
                    }
                    _ => {}
                }
            }
        }
        if votes_in > votes_out && votes_in >= 2 {
            inverted.push(format!(
                "{fid:?} {} rev={} votes={votes_in}-{votes_out}",
                face.surface().type_tag(),
                face.is_reversed()
            ));
        }
    }
    inverted
}

#[test]
fn topsocket_cut_operands_are_outward_clean() {
    // Both operands pass the outwardness oracle, so any inverted face in
    // the result is minted by the cut itself.
    let mut topo = Topology::new();
    let base = load("topsocket_cut_base.bin", &mut topo);
    let inv = inverted_faces(&topo, base);
    assert!(
        inv.is_empty(),
        "base operand must be outward-clean, got {inv:?}"
    );
    let tool = load("topsocket_cut_tool.bin", &mut topo);
    let inv = inverted_faces(&topo, tool);
    assert!(
        inv.is_empty(),
        "tool operand must be outward-clean, got {inv:?}"
    );
}

#[test]
fn topsocket_cut_emits_double_flipped_bands() {
    // ACTIVE pin of the live defect: the cut of two outward-clean operands
    // emits inverted cylinder bands (3 at capture). A fix must flip this
    // pin to assert zero and un-ignore the clean pin below.
    let mut topo = Topology::new();
    let base = load("topsocket_cut_base.bin", &mut topo);
    let tool = load("topsocket_cut_tool.bin", &mut topo);
    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
            .expect("cut must succeed");
    let inv = inverted_faces(&topo, result);
    assert!(
        !inv.is_empty(),
        "documented defect vanished — flip this pin and un-ignore the clean pin"
    );
    assert!(
        inv.iter().all(|s| s.contains("cylinder")),
        "documented inverted faces are cylinder bands, got {inv:?}"
    );
}

#[test]
#[ignore = "ready-repro: the cut of two outward-clean operands must emit zero inverted \
            faces; currently 3 double-flipped cylinder bands (see the header)"]
fn topsocket_cut_result_is_outward_clean() {
    let mut topo = Topology::new();
    let base = load("topsocket_cut_base.bin", &mut topo);
    let tool = load("topsocket_cut_tool.bin", &mut topo);
    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
            .expect("cut must succeed");
    assert_eq!(inverted_faces(&topo, result), Vec::<String>::new());
}

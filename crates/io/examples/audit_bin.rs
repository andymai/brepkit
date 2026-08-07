//! Outwardness audit over captured arena `.bin` solids: report faces whose
//! effective surface normal points INTO the material (majority vote of
//! classified points offset along the normal). Finds coherently
//! double-flipped faces that edge-sense pairing cannot see.
//!
//! ```sh
//! cargo run --release -p brepkit-io --example audit_bin -- /tmp/stages/*.bin
//! ```
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn main() {
    // BOOL_A=<path> BOOL_B=<path> BOOL_OP=cut|fuse: run the boolean natively
    // and audit its result instead of loading files.
    if let (Ok(pa), Ok(pb)) = (std::env::var("BOOL_A"), std::env::var("BOOL_B")) {
        let op = match std::env::var("BOOL_OP").as_deref() {
            Ok("fuse") => brepkit_algo::bop::BooleanOp::Fuse,
            _ => brepkit_algo::bop::BooleanOp::Cut,
        };
        let mut topo = Topology::new();
        let load = |path: &str, topo: &mut Topology| {
            std::fs::read(path)
                .ok()
                .and_then(|bytes| deserialize_solid(&bytes, topo).ok())
        };
        let (Some(a), Some(b)) = (load(&pa, &mut topo), load(&pb, &mut topo)) else {
            println!("BOOL mode: could not load {pa} / {pb} as solids");
            return;
        };
        match brepkit_algo::gfa::boolean(&mut topo, op, a, b) {
            Ok(result) => audit_one(&topo, result, "native boolean result"),
            Err(e) => println!("BOOL mode: boolean failed: {e}"),
        }
        return;
    }
    for path in std::env::args().skip(1) {
        let mut topo = Topology::new();
        let Ok(bytes) = std::fs::read(&path) else {
            println!("{path}: unreadable");
            continue;
        };
        let Ok(solid) = deserialize_solid(&bytes, &mut topo) else {
            println!("{path}: not a solid");
            continue;
        };
        audit_one(&topo, solid, &path);
    }
}

fn audit_one(topo: &Topology, solid: brepkit_topology::solid::SolidId, label: &str) {
    {
        let Ok((mesh, offsets)) =
            brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
                topo,
                solid,
                0.05,
                10.0_f64.to_radians(),
            )
        else {
            println!("{label}: tessellation failed");
            return;
        };
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
        println!(
            "{label}: F={} inverted={} {:?}",
            faces.len(),
            inverted.len(),
            inverted
        );
    }
}

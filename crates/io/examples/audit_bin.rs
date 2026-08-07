//! Oracles over captured arena `.bin` solids.
//!
//! `HALFEDGE=1`: directed half-edge count at export tolerance — the
//! AUTHORITATIVE oracle for orientation mismatches (a mesh edge traversed
//! the same way by two triangles has no opposite twin).
//!
//! Default mode: the offset-classification "outwardness" audit. CAUTION:
//! it returns unanimous false positives near concave cylinder corners
//! (a directed-watertight cut result audited "3 inverted, 10-0") — never
//! trust it without the HALFEDGE cross-check.
//!
//! `LIST=1`: dump each face's type and reversal flag.
//! `BOOL_A`/`BOOL_B`/`BOOL_OP`: run a native boolean on two captures and
//! validate + audit the result.
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
            Ok(result) => {
                let opts = brepkit_operations::validate::ValidationOptions {
                    check_orientation: true,
                    ..Default::default()
                };
                match brepkit_operations::validate::validate_solid_with_options(
                    &topo, result, &opts,
                ) {
                    Ok(report) => {
                        for i in &report.issues {
                            println!("validate: {}", i.description);
                        }
                        if report.issues.is_empty() {
                            println!("validate: clean");
                        }
                    }
                    Err(e) => println!("validate failed: {e}"),
                }
                if let Ok(mesh) = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
                    &topo,
                    result,
                    0.01,
                    5.0_f64.to_radians(),
                ) {
                    let mut half = std::collections::HashMap::new();
                    for t in mesh.indices.chunks(3) {
                        for k in 0..3 {
                            *half.entry((t[k], t[(k + 1) % 3])).or_insert(0usize) += 1;
                        }
                    }
                    let unmatched = half
                        .keys()
                        .filter(|&&(x, y)| !half.contains_key(&(y, x)))
                        .count();
                    println!("mesh: {unmatched} unmatched half-edges");
                }
                audit_one(&topo, result, "native boolean result");
            }
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
        if std::env::var("HALFEDGE").is_ok() {
            match brepkit_operations::tessellate::tessellate_solid_with_tolerance(
                &topo,
                solid,
                0.01,
                5.0_f64.to_radians(),
            ) {
                Ok(mesh) => {
                    let mut half = std::collections::HashMap::new();
                    for t in mesh.indices.chunks(3) {
                        for k in 0..3 {
                            *half.entry((t[k], t[(k + 1) % 3])).or_insert(0usize) += 1;
                        }
                    }
                    let unmatched = half
                        .keys()
                        .filter(|&&(x, y)| !half.contains_key(&(y, x)))
                        .count();
                    println!("{path}: directed unmatched half-edges = {unmatched}");
                    if unmatched > 0 && std::env::var("OWNERS").is_ok() {
                        let unmatched_set: std::collections::HashSet<(u32, u32)> = half
                            .keys()
                            .filter(|&&(x, y)| !half.contains_key(&(y, x)))
                            .copied()
                            .collect();
                        let (gmesh, offsets) =
                            brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
                                &topo,
                                solid,
                                0.01,
                                5.0_f64.to_radians(),
                            )
                            .unwrap();
                        let mut ghalf = std::collections::HashMap::new();
                        for t in gmesh.indices.chunks(3) {
                            for k in 0..3 {
                                *ghalf.entry((t[k], t[(k + 1) % 3])).or_insert(0usize) += 1;
                            }
                        }
                        let gset: std::collections::HashSet<(u32, u32)> = ghalf
                            .keys()
                            .filter(|&&(x, y)| !ghalf.contains_key(&(y, x)))
                            .copied()
                            .collect();
                        let faces = solid_faces(&topo, solid).unwrap();
                        for (fi, &fid) in faces.iter().enumerate() {
                            let mut n = 0;
                            for t in gmesh.indices[offsets[fi] as usize..offsets[fi + 1] as usize]
                                .chunks(3)
                            {
                                for k in 0..3 {
                                    if gset.contains(&(t[k], t[(k + 1) % 3])) {
                                        n += 1;
                                    }
                                }
                            }
                            if n > 0 {
                                let face = topo.face(fid).unwrap();
                                println!(
                                    "  owner {fid:?} {} rev={} : {n}",
                                    face.surface().type_tag(),
                                    face.is_reversed()
                                );
                            }
                        }
                        let _ = unmatched_set;
                    }
                }
                Err(e) => println!("{path}: tessellation failed: {e}"),
            }
            continue;
        }
        if std::env::var("LIST").is_ok() {
            for fid in solid_faces(&topo, solid).unwrap() {
                let face = topo.face(fid).unwrap();
                println!(
                    "  {fid:?} {} rev={} inner_wires={}",
                    face.surface().type_tag(),
                    face.is_reversed(),
                    face.inner_wires().len()
                );
            }
            continue;
        }
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
                // Vertex-based extents: arcs can bulge past their endpoints.
                let mut lo = [f64::MAX; 3];
                let mut hi = [f64::MIN; 3];
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    for oe in topo.wire(wid).unwrap().edges() {
                        let e = topo.edge(oe.edge()).unwrap();
                        for vid in [e.start(), e.end()] {
                            let p = topo.vertex(vid).unwrap().point();
                            let c = [p.x(), p.y(), p.z()];
                            for k in 0..3 {
                                lo[k] = lo[k].min(c[k]);
                                hi[k] = hi[k].max(c[k]);
                            }
                        }
                    }
                }
                inverted.push(format!(
                    "{fid:?} {} rev={} votes={votes_in}-{votes_out} vbox x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]",
                    face.surface().type_tag(),
                    face.is_reversed(),
                    lo[0],
                    hi[0],
                    lo[1],
                    hi[1],
                    lo[2],
                    hi[2]
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

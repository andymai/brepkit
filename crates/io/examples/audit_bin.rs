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
    // PRIM=octant: build box(10) corner + sphere(8) center natively, run the
    // requested boolean, and report faces + volume — the box-sphere intersect
    // wrong-region repro without fixture files.
    if std::env::var("PRIM").as_deref() == Ok("octant") {
        let mut topo = Topology::new();
        let b = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let s = brepkit_operations::primitives::make_sphere(&mut topo, 8.0, 32).unwrap();
        let op = match std::env::var("BOOL_OP").as_deref() {
            Ok("cut") => brepkit_algo::bop::BooleanOp::Cut,
            Ok("fuse") => brepkit_algo::bop::BooleanOp::Fuse,
            _ => brepkit_algo::bop::BooleanOp::Intersect,
        };
        let r = if std::env::var("VIA").as_deref() == Ok("ops") {
            let op2 = match op {
                brepkit_algo::bop::BooleanOp::Cut => brepkit_operations::boolean::BooleanOp::Cut,
                brepkit_algo::bop::BooleanOp::Fuse => brepkit_operations::boolean::BooleanOp::Fuse,
                brepkit_algo::bop::BooleanOp::Intersect => {
                    brepkit_operations::boolean::BooleanOp::Intersect
                }
            };
            brepkit_operations::boolean::boolean(&mut topo, op2, b, s).unwrap()
        } else {
            brepkit_algo::gfa::boolean(&mut topo, op, b, s).unwrap()
        };
        let vol = brepkit_operations::measure::solid_volume(&topo, r, 0.01).unwrap();
        let ovol = brepkit_operations::measure::oriented_solid_volume(&topo, r, 0.01).unwrap();
        println!("result vol={vol:.3} oriented={ovol:.3}");
        let opts = brepkit_operations::validate::ValidationOptions {
            check_orientation: true,
            ..Default::default()
        };
        let report =
            brepkit_operations::validate::validate_solid_with_options(&topo, r, &opts).unwrap();
        println!(
            "validate issues: {:?}",
            report
                .issues
                .iter()
                .map(|i| i.description.clone())
                .collect::<Vec<_>>()
        );
        if let Ok(mesh) = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
            &topo,
            r,
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
        let c = brepkit_operations::classify::classify_point(
            &topo,
            r,
            brepkit_math::vec::Point3::new(1.0, 1.0, 1.0),
            0.01,
            1e-7,
        );
        println!("classify(1,1,1) = {c:?}");
        for fid in solid_faces(&topo, r).unwrap() {
            let face = topo.face(fid).unwrap();
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid).unwrap().edges() {
                    let e = topo.edge(oe.edge()).unwrap();
                    for vid in [e.start(), e.end()] {
                        let pt = topo.vertex(vid).unwrap().point();
                        let c = [pt.x(), pt.y(), pt.z()];
                        for k in 0..3 {
                            lo[k] = lo[k].min(c[k]);
                            hi[k] = hi[k].max(c[k]);
                        }
                    }
                }
            }
            if std::env::var("ARCS").is_ok() {
                for wid in
                    std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
                {
                    for oe in topo.wire(wid).unwrap().edges() {
                        let e = topo.edge(oe.edge()).unwrap();
                        if let brepkit_topology::edge::EdgeCurve::Circle(_) = e.curve() {
                            let ps = topo.vertex(e.start()).unwrap().point();
                            let pe = topo.vertex(e.end()).unwrap().point();
                            let (t0, t1) = e.curve().domain_with_endpoints(ps, pe);
                            let mid =
                                e.curve()
                                    .evaluate_with_endpoints(f64::midpoint(t0, t1), ps, pe);
                            println!(
                                "    arc {:?} span={:.2}deg mid=({:.2},{:.2},{:.2})",
                                oe.edge(),
                                (t1 - t0).to_degrees(),
                                mid.x(),
                                mid.y(),
                                mid.z()
                            );
                        }
                    }
                }
            }
            println!(
                "  {fid:?} {} rev={} inner={} vbox x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]",
                face.surface().type_tag(),
                face.is_reversed(),
                face.inner_wires().len(),
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                lo[2],
                hi[2]
            );
        }
        return;
    }
    // LOFT_A/LOFT_B: thin profile extrusions (capture artifacts). Extract
    // each one's TOP cap (the extrude top carries the input profile
    // faithfully), loft them natively, validate + count directed
    // mismatches on the result.
    if let (Ok(pa), Ok(pb)) = (std::env::var("LOFT_A"), std::env::var("LOFT_B")) {
        let mut topo = Topology::new();
        let load = |path: &str, topo: &mut Topology| {
            std::fs::read(path)
                .ok()
                .and_then(|bytes| deserialize_solid(&bytes, topo).ok())
        };
        let (Some(a), Some(b)) = (load(&pa, &mut topo), load(&pb, &mut topo)) else {
            println!("LOFT mode: could not load {pa} / {pb} as solids");
            return;
        };
        let top_face = |topo: &Topology, sid: brepkit_topology::solid::SolidId| {
            let faces = solid_faces(topo, sid).unwrap();
            let mut best: Option<(brepkit_topology::face::FaceId, f64)> = None;
            for &fid in &faces {
                let face = topo.face(fid).unwrap();
                if !matches!(
                    face.surface(),
                    brepkit_topology::face::FaceSurface::Plane { .. }
                ) {
                    continue;
                }
                let mut z_min = f64::MAX;
                for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                    let e = topo.edge(oe.edge()).unwrap();
                    for vid in [e.start(), e.end()] {
                        z_min = z_min.min(topo.vertex(vid).unwrap().point().z());
                    }
                }
                if best.is_none_or(|(_, bz)| z_min > bz) {
                    best = Some((fid, z_min));
                }
            }
            best.map(|(fid, _)| fid)
        };
        let (Some(fa), Some(fb)) = (top_face(&topo, a), top_face(&topo, b)) else {
            println!("LOFT mode: could not find top caps");
            return;
        };
        match brepkit_operations::loft::loft(&mut topo, &[fa, fb]) {
            Ok(result) => {
                let mesh = brepkit_operations::tessellate::tessellate_solid_with_tolerance(
                    &topo,
                    result,
                    0.01,
                    5.0_f64.to_radians(),
                )
                .unwrap();
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
                println!("native loft: directed unmatched half-edges = {unmatched}");
            }
            Err(e) => println!("LOFT mode: loft failed: {e}"),
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

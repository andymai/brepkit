//! Fuse two captured arena `.bin` operands and validate the RESULT's shell
//! orientation, then report which faces own the tessellation's unmatched
//! half-edges.
//!
//! ```sh
//! A=a.bin B=b.bin cargo run --release -p brepkit-io --example fuse_orient
//! ```
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::collections::{HashMap, HashSet};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::validate::{ValidationOptions, validate_solid_with_options};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let a_path = std::env::var("A").expect("A=<path>");
    let b_path = std::env::var("B").expect("B=<path>");
    let mut topo = Topology::new();
    let a = deserialize_solid(&std::fs::read(&a_path).unwrap(), &mut topo).unwrap();
    let b = deserialize_solid(&std::fs::read(&b_path).unwrap(), &mut topo).unwrap();

    // FACES=0,121: dump stored normal / reversal / winding of operand faces
    // before the fuse mutates anything.
    if let Ok(list) = std::env::var("FACES") {
        for tok in list.split(',') {
            let Ok(idx) = tok.trim().parse::<usize>() else {
                continue;
            };
            let Some(fid) = topo.face_id_from_index(idx) else {
                println!("operand face Id({idx}): not found");
                continue;
            };
            let face = topo.face(fid).unwrap();
            print!(
                "operand face Id({idx}) {} rev={}",
                face.surface().type_tag(),
                face.is_reversed()
            );
            if let brepkit_topology::face::FaceSurface::Plane { normal, d } = face.surface() {
                let wire = topo.wire(face.outer_wire()).unwrap();
                let mut pts: Vec<brepkit_math::vec::Point3> = Vec::new();
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let vid = if oe.is_forward() {
                        edge.start()
                    } else {
                        edge.end()
                    };
                    pts.push(topo.vertex(vid).unwrap().point());
                }
                if face.is_reversed() {
                    pts.reverse();
                }
                if pts.len() < 3 {
                    println!(" degenerate outer wire ({} points)", pts.len());
                    continue;
                }
                let mut area2 = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
                for w in 1..pts.len().saturating_sub(1) {
                    let u = pts[w] - pts[0];
                    let v = pts[w + 1] - pts[0];
                    area2 += u.cross(v);
                }
                print!(
                    " n=({:.2},{:.2},{:.2}) d={:.3} outer_edges={} eff_signed_area={:.4}",
                    normal.x(),
                    normal.y(),
                    normal.z(),
                    d,
                    wire.edges().len(),
                    area2.dot(*normal) * 0.5
                );
            }
            println!(" inner_wires={}", face.inner_wires().len());
        }
    }

    // Outwardness audit: faces whose effective surface normal points INTO the
    // material (plus-side classifies Inside). Invisible to the combinatorial
    // same-sense check when the wire winding is coherently double-flipped.
    let audit = |topo: &Topology, solid: brepkit_topology::solid::SolidId, label: &str| {
        let (mesh, offsets) =
            brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
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
            // Majority vote over spread samples; a single centroid near thin
            // material or a boundary flips verdicts between runs.
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
                    let plus = centroid + n_eff * off;
                    let minus = centroid - n_eff * off;
                    match (
                        classify_point(topo, solid, plus, 0.01, 1e-6),
                        classify_point(topo, solid, minus, 0.01, 1e-6),
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
                // Vertex-based extents: arcs can bulge past their endpoints.
                println!(
                    "  inverted {fid:?} {} rev={} votes={votes_in}-{votes_out} vbox x[{:.2},{:.2}] y[{:.2},{:.2}] z[{:.2},{:.2}]",
                    face.surface().type_tag(),
                    face.is_reversed(),
                    lo[0],
                    hi[0],
                    lo[1],
                    hi[1],
                    lo[2],
                    hi[2]
                );
                inverted.push((
                    fid,
                    face.surface().type_tag(),
                    face.is_reversed(),
                    votes_in,
                    votes_out,
                ));
            }
        }
        println!("{label}: {} inverted faces {:?}", inverted.len(), inverted);
    };
    audit(&topo, a, "operand A");
    audit(&topo, b, "operand B");

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, a, b).unwrap();
    audit(&topo, result, "fuse result");

    let opts = ValidationOptions {
        check_orientation: true,
        ..Default::default()
    };
    let report = validate_solid_with_options(&topo, result, &opts).unwrap();
    println!("fuse result validation issues: {}", report.issues.len());
    for i in &report.issues {
        println!("  {}", i.description);
    }

    // Per-edge attribution of same-sense pairs in the B-Rep itself.
    let faces = solid_faces(&topo, result).unwrap();
    let mut edge_uses: HashMap<brepkit_topology::edge::EdgeId, Vec<(usize, bool, bool)>> =
        HashMap::new();
    for (fi, &fid) in faces.iter().enumerate() {
        let face = topo.face(fid).unwrap();
        let face_reversed = face.is_reversed();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let eff = oe.is_forward() != face_reversed;
                edge_uses
                    .entry(oe.edge())
                    .or_default()
                    .push((fi, eff, oe.is_forward()));
            }
        }
    }
    for (eid, uses) in &edge_uses {
        if uses.len() == 2 && uses[0].1 == uses[1].1 {
            let edge = topo.edge(*eid).unwrap();
            let (p0, p1) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            println!(
                "same-sense {eid:?} curve={} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3})",
                edge.curve().type_tag(),
                p0.x(),
                p0.y(),
                p0.z(),
                p1.x(),
                p1.y(),
                p1.z()
            );
            for &(fi, eff, raw) in uses {
                let fid = faces[fi];
                let face = topo.face(fid).unwrap();
                println!(
                    "    {fid:?} {} rev={} raw_fwd={raw} eff_fwd={eff}",
                    face.surface().type_tag(),
                    face.is_reversed()
                );
            }
        }
    }

    // Effective winding of every planar face in the same-sense set: shoelace
    // over the wire's vertex chain projected on the plane normal. A reversed
    // face's effective boundary is the wire in REVERSE ORDER.
    let mut suspects: Vec<usize> = Vec::new();
    for (_, uses) in edge_uses
        .iter()
        .filter(|(_, u)| u.len() == 2 && u[0].1 == u[1].1)
    {
        for &(fi, _, _) in uses {
            if !suspects.contains(&fi) {
                suspects.push(fi);
            }
        }
    }
    for fi in suspects {
        let fid = faces[fi];
        let face = topo.face(fid).unwrap();
        let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() else {
            continue;
        };
        let wire = topo.wire(face.outer_wire()).unwrap();
        let mut pts: Vec<brepkit_math::vec::Point3> = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge()).unwrap();
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            pts.push(topo.vertex(vid).unwrap().point());
        }
        if face.is_reversed() {
            pts.reverse();
        }
        if pts.len() < 3 {
            continue;
        }
        let n = *normal;
        let origin = pts[0];
        let mut area2 = brepkit_math::vec::Vec3::new(0.0, 0.0, 0.0);
        for w in 1..pts.len().saturating_sub(1) {
            let a = pts[w] - origin;
            let b = pts[w + 1] - origin;
            area2 += a.cross(b);
        }
        println!(
            "winding {fid:?} rev={} n=({:.2},{:.2},{:.2}) signed_area={:.4} edges={}",
            face.is_reversed(),
            n.x(),
            n.y(),
            n.z(),
            area2.dot(n) * 0.5,
            wire.edges().len()
        );
    }

    let (mesh, face_offsets) =
        brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
            &topo,
            result,
            0.01,
            5.0_f64.to_radians(),
        )
        .unwrap();
    let mut half: HashMap<(u32, u32), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            *half.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
        }
    }
    let unmatched_set: HashSet<(u32, u32)> = half
        .keys()
        .filter(|&&(x, y)| !half.contains_key(&(y, x)))
        .copied()
        .collect();
    println!("mesh: {} unmatched half-edges", unmatched_set.len());

    let faces = solid_faces(&topo, result).unwrap();
    let mut rows: Vec<(usize, usize)> = Vec::new();
    for fi in 0..faces.len() {
        let start = face_offsets[fi] as usize;
        let end = face_offsets[fi + 1] as usize;
        let mut n = 0;
        for t in mesh.indices[start..end].chunks(3) {
            for k in 0..3 {
                if unmatched_set.contains(&(t[k], t[(k + 1) % 3])) {
                    n += 1;
                }
            }
        }
        if n > 0 {
            rows.push((fi, n));
        }
    }
    rows.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    for &(fi, n) in rows.iter().take(16) {
        let fid = faces[fi];
        let face = topo.face(fid).unwrap();
        // Mesh-orientation check: average (triangle normal . effective face
        // normal at the triangle centroid). Negative = the face's mesh is
        // wound against its own effective orientation.
        let start = face_offsets[fi] as usize;
        let end = face_offsets[fi + 1] as usize;
        let mut dot_sum = 0.0;
        let mut tri_n = 0usize;
        for t in mesh.indices[start..end].chunks(3) {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let tn = (b - a).cross(c - a);
            if tn.length() < 1e-12 {
                continue;
            }
            let centroid = brepkit_math::vec::Point3::new(
                (a.x() + b.x() + c.x()) / 3.0,
                (a.y() + b.y() + c.y()) / 3.0,
                (a.z() + b.z() + c.z()) / 3.0,
            );
            if let Some((u, v)) = face.surface().project_point(centroid) {
                let sn = face.surface().normal(u, v);
                let eff = if face.is_reversed() { -1.0 } else { 1.0 };
                dot_sum += tn.normalize().map(|t| t.dot(sn) * eff).unwrap_or(0.0);
                tri_n += 1;
            }
        }
        // Geometric outwardness: classify points offset along the effective
        // normal from a mid-face triangle centroid. plus=Inside means the
        // effective normal points INTO the material (inverted face).
        let mut outward = String::from("n/a");
        let mid = start + ((end - start) / 6) * 3;
        if let Some(t) = mesh.indices.get(mid..mid + 3) {
            let (a, b, c) = (
                mesh.positions[t[0] as usize],
                mesh.positions[t[1] as usize],
                mesh.positions[t[2] as usize],
            );
            let centroid = brepkit_math::vec::Point3::new(
                (a.x() + b.x() + c.x()) / 3.0,
                (a.y() + b.y() + c.y()) / 3.0,
                (a.z() + b.z() + c.z()) / 3.0,
            );
            if let Some((u, v)) = face.surface().project_point(centroid) {
                let sn = face.surface().normal(u, v);
                let eff = if face.is_reversed() { -1.0 } else { 1.0 };
                if let Ok(n_eff) = (sn * eff).normalize() {
                    let plus = centroid + n_eff * 0.05;
                    let minus = centroid - n_eff * 0.05;
                    let cp = brepkit_operations::classify::classify_point(
                        &topo, result, plus, 0.01, 1e-6,
                    );
                    let cm = brepkit_operations::classify::classify_point(
                        &topo, result, minus, 0.01, 1e-6,
                    );
                    outward = format!("plus={cp:?} minus={cm:?}");
                }
            }
        }
        println!(
            "  {fid:?} {} reversed={} : {n} unmatched half-edges, mesh_orient={:.3} over {tri_n} tris, {outward}",
            face.surface().type_tag(),
            face.is_reversed(),
            if tri_n > 0 {
                dot_sum / tri_n as f64
            } else {
                f64::NAN
            }
        );
    }
    for (x, y) in unmatched_set.iter().take(6) {
        let pa = mesh.positions[*x as usize];
        let pb = mesh.positions[*y as usize];
        println!(
            "  half-edge ({:.3},{:.3},{:.3}) -> ({:.3},{:.3},{:.3})",
            pa.x(),
            pa.y(),
            pa.z(),
            pb.x(),
            pb.y(),
            pb.z()
        );
    }
}

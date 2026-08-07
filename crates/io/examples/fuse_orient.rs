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

    let result =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, a, b).unwrap();

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
        println!(
            "  {fid:?} {} reversed={} : {n} unmatched half-edges",
            face.surface().type_tag(),
            face.is_reversed()
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

//! TEMPORARY replay — do not commit. Replays the doubled-dovetail connector
//! pipeline step by step from captured operands.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    missing_docs,
    clippy::print_stdout,
    clippy::type_complexity,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::items_after_statements,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::or_fun_call
)]

use std::collections::HashMap;

use brepkit_operations::boolean::{BooleanOp, boolean_with_evolution};
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DIR: &str = "/tmp/claude-1000/-home-andy-Git-brepkit/ed01ede9-8360-4c92-93f3-dbd6ad5bd177/scratchpad/interior_stages";

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let data = std::fs::read(format!("{DIR}/{name}")).unwrap();
    brepkit_io::arena_io::deserialize_solid(&data, topo).unwrap()
}

fn mesh_health(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let mesh = tessellate_solid_with_tolerance(topo, solid, 0.01, 5.0_f64.to_radians()).unwrap();
    let q = |v: f64| (v * 1.0e4).round() as i64;
    let mut occ: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            let a = t[k] as usize;
            let b = t[(k + 1) % 3] as usize;
            let vs = &mesh.positions;
            let pa = (q(vs[a].x()), q(vs[a].y()), q(vs[a].z()));
            let pb = (q(vs[b].x()), q(vs[b].y()), q(vs[b].z()));
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            *occ.entry(key).or_default() += 1;
        }
    }
    let bnd = occ.values().filter(|&&c| c == 1).count();
    let nm = occ.values().filter(|&&c| c > 2).count();
    (mesh.indices.len() / 3, bnd, nm)
}

#[test]
fn probe_single_nub_fuse() {
    use brepkit_math::vec::Point3;
    use brepkit_topology::explorer::solid_faces;

    let mut topo = Topology::new();
    let base = load(&mut topo, "stage_03_preConnector.bin");
    let nub = load(&mut topo, "stage_04_nub_0.bin");

    println!("=== nub_0 geometry ===");
    for fid in solid_faces(&topo, nub).unwrap() {
        let face = topo.face(fid).unwrap();
        println!("-- face {fid:?} [{}]:", face.surface().type_tag());
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (s, t) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                println!(
                    "   [{}] ({:.4},{:.4},{:.4}) -> ({:.4},{:.4},{:.4})",
                    e.curve().type_tag(),
                    s.x(),
                    s.y(),
                    s.z(),
                    t.x(),
                    t.y(),
                    t.z()
                );
            }
        }
    }

    let (t, b, n) = mesh_health(&topo, nub);
    println!("nub_0 alone: tris={t} bnd={b} nm={n}");
    let (t, b, n) = mesh_health(&topo, base);
    println!("base alone: tris={t} bnd={b} nm={n}");

    let (result, _) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, base, nub).unwrap();

    // B-Rep free/over edges by quantized position.
    type Q = (i64, i64, i64);
    let q = |p: Point3| -> Q {
        (
            (p.x() * 1.0e5).round() as i64,
            (p.y() * 1.0e5).round() as i64,
            (p.z() * 1.0e5).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    let mut ends: HashMap<(Q, Q), (Point3, Point3, String)> = HashMap::new();
    for fid in solid_faces(&topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (s, t) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                let (a, b) = (q(s), q(t));
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
                ends.entry(key)
                    .or_insert((s, t, format!("{fid:?} {}", face.surface().type_tag())));
            }
        }
    }
    let faces = solid_faces(&topo, result).unwrap();
    println!("=== fuse result: {} faces ===", faces.len());
    for (key, &c) in &occ {
        if c == 1 {
            let (s, t, ref owner) = ends[key];
            println!(
                "FREE edge on {owner}: ({:.4},{:.4},{:.4}) -> ({:.4},{:.4},{:.4})",
                s.x(),
                s.y(),
                s.z(),
                t.x(),
                t.y(),
                t.z()
            );
        }
        if c > 2 {
            let (s, t, ref owner) = ends[key];
            println!(
                "OVER edge x{c} on {owner}: ({:.4},{:.4},{:.4}) -> ({:.4},{:.4},{:.4})",
                s.x(),
                s.y(),
                s.z(),
                t.x(),
                t.y(),
                t.z()
            );
        }
    }
}

#[test]
fn replay_connector_pipeline() {
    let mut topo = Topology::new();
    let mut current = load(&mut topo, "stage_03_preConnector.bin");
    let (t, b, n) = mesh_health(&topo, current);
    println!("preConnector: tris={t} bnd={b} nm={n}");

    for i in 0..12 {
        let nub = load(&mut topo, &format!("stage_{:02}_nub_{i}.bin", 4 + i));
        match boolean_with_evolution(&mut topo, BooleanOp::Fuse, current, nub) {
            Ok((r, _)) => {
                current = r;
                let (t, b, n) = mesh_health(&topo, current);
                println!("after fuse nub_{i}: tris={t} bnd={b} nm={n}");
            }
            Err(e) => {
                println!("fuse nub_{i} FAILED: {e}");
                return;
            }
        }
    }
    for i in 0..12 {
        let hole = load(&mut topo, &format!("stage_{:02}_hole_{i}.bin", 16 + i));
        match boolean_with_evolution(&mut topo, BooleanOp::Cut, current, hole) {
            Ok((r, _)) => {
                current = r;
                let (t, b, n) = mesh_health(&topo, current);
                println!("after cut hole_{i}: tris={t} bnd={b} nm={n}");
            }
            Err(e) => {
                println!("cut hole_{i} FAILED: {e}");
                return;
            }
        }
    }
}

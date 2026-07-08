//! TEMPORARY replay — do not commit. Replays the doubled-dovetail tongue
//! relief cut (tongue − socket pocket) from captured tool operands.

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

use brepkit_operations::boolean::{BooleanOp, boolean, boolean_with_evolution};
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const DIR: &str = "/tmp/claude-1000/-home-andy-Git-brepkit/ed01ede9-8360-4c92-93f3-dbd6ad5bd177/scratchpad/relief_ops";

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
fn probe_relief_cut_detail() {
    use brepkit_math::vec::Point3;
    use brepkit_topology::explorer::solid_faces;

    let _ = env_logger::try_init();

    let mut topo = Topology::new();
    let tongue = load(&mut topo, "relief_tongue_-38.00.bin");
    let cutter = load(&mut topo, "relief_cutter_-38.00_0.bin");

    println!("=== tongue ===");
    for fid in solid_faces(&topo, tongue).unwrap() {
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
    let cutter_faces = solid_faces(&topo, cutter).unwrap();
    let mut tags: HashMap<&str, usize> = HashMap::new();
    for &f in &cutter_faces {
        *tags
            .entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    println!("=== cutter: {} faces {tags:?} ===", cutter_faces.len());

    let result = boolean(&mut topo, BooleanOp::Cut, tongue, cutter).unwrap();
    type Q = (i64, i64, i64);
    let q = |p: Point3| -> Q {
        (
            (p.x() * 1.0e5).round() as i64,
            (p.y() * 1.0e5).round() as i64,
            (p.z() * 1.0e5).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    let mut ends: HashMap<(Q, Q), (Point3, Point3, String, String)> = HashMap::new();
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
                ends.entry(key).or_insert((
                    s,
                    t,
                    format!("{fid:?}"),
                    e.curve().type_tag().to_string(),
                ));
            }
        }
    }
    let faces = solid_faces(&topo, result).unwrap();
    println!("=== cut result: {} faces ===", faces.len());
    for fid in &faces {
        let face = topo.face(*fid).unwrap();
        let ne: usize = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .map(|wid| topo.wire(wid).unwrap().edges().len())
            .sum();
        println!("  face {fid:?} [{}] edges={ne}", face.surface().type_tag());
    }
    for (key, &c) in &occ {
        if c != 2 {
            let (s, t, ref owner, ref ctag) = ends[key];
            println!(
                "x{c} edge [{ctag}] on {owner}: ({:.4},{:.4},{:.4}) -> ({:.4},{:.4},{:.4})",
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
fn replay_relief_cuts() {
    let mut names: Vec<String> = std::fs::read_dir(DIR)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.starts_with("relief_tongue_"))
        .collect();
    names.sort();
    for tongue_name in names {
        let bp = tongue_name
            .trim_start_matches("relief_tongue_")
            .trim_end_matches(".bin")
            .to_string();
        let mut topo = Topology::new();
        let tongue = load(&mut topo, &tongue_name);
        let (t0, b0, n0) = mesh_health(&topo, tongue);
        let cutter = load(&mut topo, &format!("relief_cutter_{bp}_0.bin"));
        let (tc, bc, nc) = mesh_health(&topo, cutter);
        match boolean(&mut topo, BooleanOp::Cut, tongue, cutter) {
            Ok(r) => {
                let (t, b, n) = mesh_health(&topo, r);
                println!(
                    "bp={bp}: tongue(tris={t0},bnd={b0},nm={n0}) cutter(tris={tc},bnd={bc},nm={nc}) -> boolean cut tris={t} bnd={b} nm={n}"
                );
            }
            Err(e) => println!("bp={bp}: boolean cut FAILED: {e}"),
        }
        // Also via the evolution path (the tool goes through cutAll → which entry?).
        let mut topo2 = Topology::new();
        let tongue2 = load(&mut topo2, &tongue_name);
        let cutter2 = load(&mut topo2, &format!("relief_cutter_{bp}_0.bin"));
        match boolean_with_evolution(&mut topo2, BooleanOp::Cut, tongue2, cutter2) {
            Ok((r, _)) => {
                let (t, b, n) = mesh_health(&topo2, r);
                println!("bp={bp}: evolution cut tris={t} bnd={b} nm={n}");
            }
            Err(e) => println!("bp={bp}: evolution cut FAILED: {e}"),
        }
    }
}

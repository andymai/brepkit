//! Replay a captured pair through raw GFA and print every face wire whose
//! edge chain does not close positionally, plus the directed-use count of
//! every shared edge (the halfedge winding oracle). Written for the #1538
//! deep-cutout case, where validation reports unclosed wires the free-edge
//! census cannot see.
//!
//! `A=<a.bin> B=<b.bin> OP=cut cargo run --release -p brepkit-io \
//!   --example replay_wire_audit`
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

type Q3 = (i64, i64, i64);

use std::collections::HashMap;
use std::path::PathBuf;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let a_path = PathBuf::from(std::env::var_os("A").expect("A=<path>"));
    let b_path = PathBuf::from(std::env::var_os("B").expect("B=<path>"));
    let op = match std::env::var("OP").as_deref() {
        Ok("cut") => brepkit_algo::bop::BooleanOp::Cut,
        Ok("intersect") => brepkit_algo::bop::BooleanOp::Intersect,
        _ => brepkit_algo::bop::BooleanOp::Fuse,
    };

    let mut topo = Topology::new();
    let a = deserialize_solid(&std::fs::read(&a_path).unwrap(), &mut topo).unwrap();
    let b = deserialize_solid(&std::fs::read(&b_path).unwrap(), &mut topo).unwrap();
    let result = brepkit_algo::gfa::boolean(&mut topo, op, a, b).unwrap();
    if let Some(save) = std::env::var_os("SAVE") {
        std::fs::write(
            save,
            brepkit_io::arena_io::serialize_solid(&topo, result).unwrap(),
        )
        .unwrap();
    }

    let mut directed: HashMap<(Q3, Q3), i64> = HashMap::new();
    let q = |p: brepkit_math::vec::Point3| -> Q3 {
        let f = |v: f64| (v * 1e6).round() as i64;
        (f(p.x()), f(p.y()), f(p.z()))
    };
    for fid in solid_faces(&topo, result).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            let mut chain: Vec<(brepkit_math::vec::Point3, brepkit_math::vec::Point3)> = Vec::new();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (s3, e3) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                let (s3, e3) = if oe.is_forward() { (s3, e3) } else { (e3, s3) };
                chain.push((s3, e3));
                *directed.entry((q(s3), q(e3))).or_default() += 1;
            }
            let mut breaks = Vec::new();
            for i in 0..chain.len() {
                let gap = (chain[(i + 1) % chain.len()].0 - chain[i].1).length();
                if gap > 1e-6 {
                    breaks.push((i, gap));
                }
            }
            if !breaks.is_empty() {
                println!(
                    "UNCLOSED {fid:?} wire {wid:?} ({} edges, {} breaks)",
                    chain.len(),
                    breaks.len()
                );
                for (s3, e3) in &chain {
                    println!(
                        "    ({:7.3},{:7.3},{:7.3}) -> ({:7.3},{:7.3},{:7.3})",
                        s3.x(),
                        s3.y(),
                        s3.z(),
                        e3.x(),
                        e3.y(),
                        e3.z()
                    );
                }
            }
        }
    }
    let mut same_dir = 0usize;
    for ((sk, ek), n) in &directed {
        if *n > 1 && sk != ek {
            same_dir += 1;
            println!(
                "SAMEDIR x{n} ({:.3},{:.3},{:.3}) -> ({:.3},{:.3},{:.3})",
                sk.0 as f64 / 1e6,
                sk.1 as f64 / 1e6,
                sk.2 as f64 / 1e6,
                ek.0 as f64 / 1e6,
                ek.1 as f64 / 1e6,
                ek.2 as f64 / 1e6
            );
        }
    }
    println!("directed edge keys used more than once (same direction): {same_dir}");
}

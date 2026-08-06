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

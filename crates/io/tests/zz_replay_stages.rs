//! TEMPORARY replay harness — do not commit. Deserializes captured baseplate
//! stages and reports mesh health at export tolerance.

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
    clippy::many_single_char_names
)]

use std::collections::HashMap;

use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;

#[test]
fn replay_interior_stages() {
    let dir = std::env::var("STAGE_DIR").unwrap_or_else(|_| {
        "/tmp/claude-1000/-home-andy-Git-brepkit/ed01ede9-8360-4c92-93f3-dbd6ad5bd177/scratchpad/interior_stages".into()
    });
    let mut files: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "bin"))
        .collect();
    files.sort();
    for f in files {
        let mut topo = Topology::new();
        let data = std::fs::read(&f).unwrap();
        let solid = match brepkit_io::arena_io::deserialize_solid(&data, &mut topo) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "{}: DESERIALIZE FAILED: {e}",
                    f.file_name().unwrap().to_string_lossy()
                );
                continue;
            }
        };
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        let curved = faces
            .iter()
            .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
            .count();
        match tessellate_solid_with_tolerance(&topo, solid, 0.01, 5.0_f64.to_radians()) {
            Ok(mesh) => {
                let q = |v: f64| (v * 1.0e4).round() as i64;
                let mut occ: HashMap<((i64, i64, i64), (i64, i64, i64)), usize> = HashMap::new();
                let idx = &mesh.indices;
                let vs = &mesh.positions;
                for t in idx.chunks(3) {
                    for k in 0..3 {
                        let a = t[k] as usize;
                        let b = t[(k + 1) % 3] as usize;
                        let pa = (q(vs[a].x()), q(vs[a].y()), q(vs[a].z()));
                        let pb = (q(vs[b].x()), q(vs[b].y()), q(vs[b].z()));
                        let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
                        *occ.entry(key).or_default() += 1;
                    }
                }
                let bnd = occ.values().filter(|&&c| c == 1).count();
                let nm = occ.values().filter(|&&c| c > 2).count();
                println!(
                    "{}: faces={} curved={curved} tris={} bnd={bnd} nm={nm}",
                    f.file_name().unwrap().to_string_lossy(),
                    faces.len(),
                    idx.len() / 3
                );
            }
            Err(e) => {
                println!(
                    "{}: faces={} curved={curved} TESSELLATION FAILED: {e}",
                    f.file_name().unwrap().to_string_lossy(),
                    faces.len()
                );
            }
        }
    }
}

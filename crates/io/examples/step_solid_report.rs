#![allow(clippy::print_stdout, clippy::expect_used, missing_docs)]
//! Report B-Rep and mesh health for a STEP file: face/surface mix, by-edge-id
//! use counts, and position-welded mesh edge use counts.
//!
//! Distinguishes "the B-Rep is broken" from "the B-Rep is fine but its
//! tessellation folds" — the two have identical STL symptoms.

use std::collections::HashMap;

use brepkit_io::step::reader::read_step;
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let path = std::env::args()
        .nth(1)
        .expect("usage: step_solid_report <file.step>");
    let text = std::fs::read_to_string(&path).expect("read");

    let mut topo = Topology::new();
    let solids = read_step(&text, &mut topo).expect("read_step");
    println!("{path}: {} solid(s)", solids.len());

    for (n, &sid) in solids.iter().enumerate() {
        let faces = solid_faces(&topo, sid).expect("faces");

        let mut mix: HashMap<&str, usize> = HashMap::new();
        let mut uses: HashMap<EdgeId, usize> = HashMap::new();
        for &fid in &faces {
            let face = topo.face(fid).expect("face");
            *mix.entry(face.surface().type_tag()).or_default() += 1;
            for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wid).expect("wire").edges() {
                    *uses.entry(oe.edge()).or_default() += 1;
                }
            }
        }
        let free = uses.values().filter(|&&c| c == 1).count();
        let over = uses.values().filter(|&&c| c > 2).count();

        let mesh = tessellate_solid_with_tolerance(&topo, sid, 0.01, 0.5).expect("mesh");
        let q = |v: f64| (v * 1e4).round() as i64;
        let mut edge_use: HashMap<[i64; 6], usize> = HashMap::new();
        for tri in mesh.indices.chunks(3) {
            let p: Vec<[i64; 3]> = tri
                .iter()
                .map(|&i| {
                    let pt = mesh.positions[i as usize];
                    [q(pt.x()), q(pt.y()), q(pt.z())]
                })
                .collect();
            for i in 0..3 {
                let (a, b) = (p[i], p[(i + 1) % 3]);
                let k = if a <= b {
                    [a[0], a[1], a[2], b[0], b[1], b[2]]
                } else {
                    [b[0], b[1], b[2], a[0], a[1], a[2]]
                };
                *edge_use.entry(k).or_default() += 1;
            }
        }
        let mesh_bnd = edge_use.values().filter(|&&c| c == 1).count();
        let mesh_nm = edge_use.values().filter(|&&c| c > 2).count();

        if std::env::var("DUMP_NM").is_ok() {
            let mut bad: Vec<_> = edge_use
                .iter()
                .filter(|&(_, &c)| c > 2)
                .map(|(k, &c)| (*k, c))
                .collect();
            bad.sort_unstable();
            for (k, c) in bad.iter().take(40) {
                println!(
                    "    nm x{c}: ({:.3},{:.3},{:.3}) -> ({:.3},{:.3},{:.3})",
                    k[0] as f64 / 1e4,
                    k[1] as f64 / 1e4,
                    k[2] as f64 / 1e4,
                    k[3] as f64 / 1e4,
                    k[4] as f64 / 1e4,
                    k[5] as f64 / 1e4,
                );
            }
        }

        if let Ok(zs) = std::env::var("MESH_AREA_AT") {
            let target: f64 = zs.parse().expect("MESH_AREA_AT");
            let mut area = 0.0;
            let mut n = 0;
            for tri in mesh.indices.chunks(3) {
                let p: Vec<_> = tri.iter().map(|&i| mesh.positions[i as usize]).collect();
                if p.iter().all(|q| (q.z() - target).abs() < 1e-6) {
                    let u = p[1] - p[0];
                    let v = p[2] - p[0];
                    area += u.cross(v).length() * 0.5;
                    n += 1;
                }
            }
            println!("    mesh@{target}: tris={n} area={area:.3}");
        }

        if let Ok(zs) = std::env::var("PLANES_AT") {
            let target: f64 = zs.parse().expect("PLANES_AT");
            for &fid in &faces {
                let face = topo.face(fid).expect("face");
                if face.surface().type_tag() != "plane" {
                    continue;
                }
                let wire = topo.wire(face.outer_wire()).expect("wire");
                let zvals: Vec<f64> = wire
                    .edges()
                    .iter()
                    .flat_map(|oe| {
                        let e = topo.edge(oe.edge()).expect("edge");
                        [e.start(), e.end()].map(|v| topo.vertex(v).expect("vertex").point().z())
                    })
                    .collect();
                let zmin = zvals.iter().copied().fold(f64::MAX, f64::min);
                let zmax = zvals.iter().copied().fold(f64::MIN, f64::max);
                if (zmax - zmin).abs() < 1e-6 && (zmin - target).abs() < 1e-3 {
                    let area = brepkit_operations::measure::face_area(&topo, fid, 0.01)
                        .unwrap_or(f64::NAN);
                    println!(
                        "    plane@{target}: {fid:?} outer_edges={} inners={} area={area:.3}",
                        wire.edges().len(),
                        face.inner_wires().len(),
                    );
                }
            }
        }

        let mut mix: Vec<_> = mix.into_iter().collect();
        mix.sort_unstable();
        println!(
            "  solid[{n}]: F={} mix={mix:?} brep_free={free} brep_over={over} tris={} mesh_bnd={mesh_bnd} mesh_nm={mesh_nm}",
            faces.len(),
            mesh.indices.len() / 3,
        );
    }
}

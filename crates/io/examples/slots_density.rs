//! Per-surface-type triangle census of the slots cut at export tolerance.
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]
use std::collections::HashMap;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

struct Tap;
impl log::Log for Tap {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }
    fn log(&self, r: &log::Record) {
        println!("[{}] {}", r.level(), r.args());
    }
    fn flush(&self) {}
}
static TAP: Tap = Tap;

fn main() {
    let _ = log::set_logger(&TAP);
    log::set_max_level(log::LevelFilter::Debug);
    let mut topo = Topology::new();
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let body = deserialize_solid(
        &std::fs::read(dir.join("slots_lip_body.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    let tool = deserialize_solid(
        &std::fs::read(dir.join("slots_lip_tool.bin")).unwrap(),
        &mut topo,
    )
    .unwrap();
    let result = brepkit_operations::boolean::boolean(
        &mut topo,
        brepkit_operations::boolean::BooleanOp::Cut,
        body,
        tool,
    )
    .unwrap();

    let deflection: f64 = std::env::var("DEFL")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.01);
    let angular: f64 = std::env::var("ANG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5.0_f64.to_radians());
    let (mesh, face_offsets) =
        brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance(
            &topo, result, deflection, angular,
        )
        .unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let mut by_type: HashMap<&'static str, (usize, usize)> = HashMap::new();
    let mut max_faces: Vec<(usize, String)> = Vec::new();
    for (i, fid) in faces.iter().enumerate() {
        let tag = topo.face(*fid).unwrap().surface().type_tag();
        let tris = ((face_offsets[i + 1] - face_offsets[i]) / 3) as usize;
        let e = by_type.entry(tag).or_insert((0, 0));
        e.0 += 1;
        e.1 += tris;
        max_faces.push((tris, format!("{fid:?}:{tag}")));
    }
    max_faces.sort_unstable_by_key(|a| std::cmp::Reverse(a.0));
    let total = mesh.indices.len() / 3;
    println!("total tris={total} defl={deflection} ang={angular:.4}");
    let mut rows: Vec<_> = by_type.into_iter().collect();
    rows.sort_unstable();
    for (tag, (n, t)) in rows {
        println!("  {tag}: faces={n} tris={t} avg={:.0}", t as f64 / n as f64);
    }
    println!("top faces:");
    for (t, f) in max_faces.iter().take(8) {
        println!("  {f} tris={t}");
    }
    // Detail the densest cylinder: distinct vertices and per-boundary-edge
    // sample counts.
    if let Some((_, label)) = max_faces.first() {
        let want: usize = label
            .trim_start_matches("Id(")
            .split(')')
            .next()
            .unwrap()
            .parse()
            .unwrap();
        for (i, fid) in faces.iter().enumerate() {
            if fid.index() != want {
                continue;
            }
            let lo = face_offsets[i] as usize;
            let hi = face_offsets[i + 1] as usize;
            let verts: std::collections::HashSet<u32> =
                mesh.indices[lo..hi].iter().copied().collect();
            println!(
                "detail {label}: tris={} verts={}",
                (hi - lo) / 3,
                verts.len()
            );
            let face = topo.face(*fid).unwrap();
            for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (a, b) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                let extra = match e.curve() {
                    brepkit_topology::edge::EdgeCurve::Circle(c) => format!(
                        " r={:.3} c=({:.2},{:.2},{:.2})",
                        c.radius(),
                        c.center().x(),
                        c.center().y(),
                        c.center().z()
                    ),
                    _ => String::new(),
                };
                println!(
                    "  edge e{} {}{extra} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
                    oe.edge().index(),
                    e.curve().type_tag(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z()
                );
            }
        }
    }
}

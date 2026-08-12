//! Print every wire of every face of a serialized solid, with oriented edge
//! endpoint chains, filtered to faces whose AABB overlaps an optional box.
//!
//! `F=<path.bin> BOX=x0,x1,y0,y1,z0,z1 cargo run --release -p brepkit-io \
//!   --example dump_face_wires`
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn main() {
    let path = std::env::var_os("F").expect("F=<path>");
    let bounds: Option<Vec<f64>> = std::env::var("BOX")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect());

    let mut topo = Topology::new();
    let sid = deserialize_solid(&std::fs::read(&path).unwrap(), &mut topo).unwrap();
    if std::env::var("VALIDATE").is_ok() {
        let report = brepkit_operations::validate::validate_solid(&topo, sid).unwrap();
        println!("validate: valid={} issues={:?}", report.is_valid(), report);
        return;
    }
    for fid in solid_faces(&topo, sid).unwrap() {
        let face = topo.face(fid).unwrap();
        let wires: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        let mut pts = Vec::new();
        for &wid in &wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                pts.push(topo.vertex(e.start()).unwrap().point());
                pts.push(topo.vertex(e.end()).unwrap().point());
            }
        }
        if let Some(b) = &bounds
            && b.len() == 6
        {
            let inside = pts.iter().any(|p| {
                p.x() >= b[0]
                    && p.x() <= b[1]
                    && p.y() >= b[2]
                    && p.y() <= b[3]
                    && p.z() >= b[4]
                    && p.z() <= b[5]
            });
            if !inside {
                continue;
            }
        }
        println!("{fid:?} {}", face.surface().type_tag());
        for (wi, &wid) in wires.iter().enumerate() {
            println!("  w{wi} {wid:?}");
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (s3, e3) = (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                );
                let (s3, e3) = if oe.is_forward() { (s3, e3) } else { (e3, s3) };
                println!(
                    "    {:?} fwd={} ({:7.3},{:7.3},{:7.3}) -> ({:7.3},{:7.3},{:7.3})",
                    oe.edge(),
                    oe.is_forward(),
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

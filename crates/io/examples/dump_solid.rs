//! Print every face of a serialized solid: surface type, outer-wire vertex
//! count, and the outer wire's vertex positions.
//!
//! Usage: `F=<path.bin> cargo run --release -p brepkit-io --example dump_solid`

#![allow(clippy::print_stdout, clippy::unwrap_used, clippy::expect_used)]

use brepkit_io::arena_io::deserialize_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

fn dump_edge_uses(topo: &Topology, sid: brepkit_topology::solid::SolidId) {
    use std::collections::HashMap;
    let mut uses: HashMap<brepkit_topology::edge::EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, sid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                *uses.entry(oe.edge()).or_default() += 1;
            }
        }
    }
    for (eid, n) in &uses {
        if *n != 2 {
            let e = topo.edge(*eid).unwrap();
            let a = topo.vertex(e.start()).unwrap().point();
            let b = topo.vertex(e.end()).unwrap().point();
            println!(
                "EDGE {eid:?} uses={n} ({:.2},{:.2},{:.2})->({:.2},{:.2},{:.2})",
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

fn main() {
    let path = std::env::var_os("F").expect("F=<path>");
    let data = std::fs::read(path).unwrap();
    let mut topo = Topology::new();
    let sid = deserialize_solid(&data, &mut topo).unwrap();
    dump_edge_uses(&topo, sid);
    for fid in solid_faces(&topo, sid).unwrap() {
        let face = topo.face(fid).unwrap();
        for (wi, wid) in std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .enumerate()
        {
            let wire = topo.wire(wid).unwrap();
            let mut pts = Vec::new();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let v = topo.vertex(oe.oriented_start(e)).unwrap().point();
                pts.push(format!(
                    "e{}({:.2},{:.2},{:.2})",
                    oe.edge().index(),
                    v.x(),
                    v.y(),
                    v.z()
                ));
            }
            println!(
                "{fid:?} {} rev={} w{wi} n={} {}",
                face.surface().type_tag(),
                face.is_reversed(),
                pts.len(),
                pts.join(" ")
            );
        }
    }
}

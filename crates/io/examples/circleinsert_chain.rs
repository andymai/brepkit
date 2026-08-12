//! #1538 circleinsert chain probe: cut the captured pre-pocket bin with a
//! canonical quartered-cylinder tool, then fuse the captured socket assembly
//! across the coplanar z=0 interface. The cut validates; the fuse is the
//! open residue (free edges + non-manifold at the pocket-mouth region).
//!
//! `FREE_EDGES=1` prints every non-2-use edge of the fuse with its owners.
//! `SAVE_BODY=<path>` serializes the holed body after the cut.
#![allow(clippy::print_stdout, clippy::expect_used, clippy::unwrap_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

fn load(name: &str, topo: &mut Topology) -> SolidId {
    let path: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn cw_tool(topo: &mut Topology) -> SolidId {
    let (r, z0) = (10.0, 0.0);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7))
    };
    let v0 = v(topo, r, 0.0);
    let v1 = v(topo, 0.0, r);
    let v2 = v(topo, -r, 0.0);
    let v3 = v(topo, 0.0, -r);
    let arc = |topo: &mut Topology, a, b| {
        let circle = Circle3D::new(Point3::new(0.0, 0.0, z0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        arc(topo, v0, v1),
        arc(topo, v1, v2),
        arc(topo, v2, v3),
        arc(topo, v3, v0),
    ];
    let oes: Vec<OrientedEdge> = edges
        .iter()
        .rev()
        .map(|&e| OrientedEdge::new(e, false))
        .collect();
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        },
    ));
    brepkit_operations::extrude::extrude(topo, fid, z, 5.0).unwrap()
}

fn report(topo: &Topology, sid: SolidId, label: &str) {
    let faces = solid_faces(topo, sid).unwrap();
    let mut mix: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
    let mut uses: HashMap<EdgeId, Vec<brepkit_topology::face::FaceId>> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        *mix.entry(face.surface().type_tag()).or_default() += 1;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                uses.entry(oe.edge()).or_default().push(fid);
            }
        }
    }
    let free = uses.values().filter(|f| f.len() == 1).count();
    let over = uses.values().filter(|f| f.len() > 2).count();
    if std::env::var("FREE_EDGES").is_ok() {
        for (eid, owners) in &uses {
            if owners.len() != 2
                && let Ok(e) = topo.edge(*eid)
                && let (Ok(a), Ok(b)) = (topo.vertex(e.start()), topo.vertex(e.end()))
            {
                let (a, b) = (a.point(), b.point());
                let os: Vec<String> = owners
                    .iter()
                    .map(|f| {
                        let tag = topo
                            .face(*f)
                            .map(|fc| fc.surface().type_tag())
                            .unwrap_or("?");
                        format!("{f:?}:{tag}")
                    })
                    .collect();
                println!(
                    "  use{} {eid:?} ({:.3},{:.3},{:.3})->({:.3},{:.3},{:.3}) {}",
                    owners.len(),
                    a.x(),
                    a.y(),
                    a.z(),
                    b.x(),
                    b.y(),
                    b.z(),
                    os.join(" ")
                );
            }
        }
    }
    let vol = brepkit_operations::measure::oriented_solid_volume(topo, sid, 0.05).unwrap();
    println!(
        "{label}: F={} mix={mix:?} free={free} over={over} vol={vol:.3}",
        faces.len()
    );
}

fn main() {
    let mut topo = Topology::new();
    let base = load("circleinsert_base.bin", &mut topo);
    let tool = cw_tool(&mut topo);
    let body = brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, base, tool)
        .unwrap();
    report(&topo, body, "pocket cut");
    if let Some(save) = std::env::var_os("SAVE_BODY") {
        std::fs::write(
            save,
            brepkit_io::arena_io::serialize_solid(&topo, body).unwrap(),
        )
        .unwrap();
    }
    let sockets = load("circleinsert_sockets.bin", &mut topo);
    report(&topo, sockets, "sockets");
    let fused =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Fuse, body, sockets)
            .unwrap();
    report(&topo, fused, "socket fuse");
}

//! Faithful regression guard: the halfSockets bin × socket-assembly fuse at
//! the z=5 base interface. Operands captured from the live gridfinity tool
//! via the boolean-capture probe kernel (arena-serialized at the boolean
//! entry): the final export fuse of the `2×2 halfSockets` bin (a 4×4 socket
//! grid whose four interior outlines touch nothing else — the disconnected
//! case; smaller bins have every outline on the bin boundary and never hit
//! it).
//!
//! Half sockets tile the base at half pitch, so the interior socket outlines
//! sit strictly inside the bin bottom, touching neither the bin outline nor
//! each other. When such a face is split by the planar arrangement, each
//! interior outline is a DISCONNECTED component of the trace graph and its
//! cycle is traced twice — once per orientation. Flat emission shipped both
//! traces (duplicate overlapping discs) and left the surrounding web region
//! without those outlines as inner wires, so the hole-less web geometrically
//! covered the duplicates. Same-domain detection then glued web, duplicate
//! discs, and the socket tops into one group and dropped every piece: the
//! web vanished, and the assembler's cap fill patched the openings with
//! membranes lying inside solid material. The mesh showed same-direction
//! half-edge pairs along every interior cell rim (boundary edges with zero
//! non-manifold edges) and a −13% signed volume.
//!
//! Fixed by resolving twin cycle pairs in the arrangement's flat emission:
//! a disconnected loop is emitted once as a solid region and its reversed
//! twin becomes an inner wire of the region that geometrically contains it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean_with_evolution};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    faces_per_edge.values().filter(|f| f.len() == 1).count()
}

fn assert_fuse_health(topo: &Topology, r: SolidId, min_curved: usize, max_faces: usize) {
    let n_faces = solid_faces(topo, r).unwrap().len();
    assert!(
        n_faces < max_faces,
        "fuse returned {n_faces} faces — mesh fallback fired?"
    );
    let curved = curved_count(topo, r);
    assert!(
        curved >= min_curved,
        "only {curved} curved faces — analytic surfaces lost?"
    );
    assert_eq!(free_edge_count(topo, r), 0, "free B-Rep edges in result");

    let mesh = tessellate_solid_with_tolerance(topo, r, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "boundary edges in export mesh"
    );
    assert_eq!(
        non_manifold_edge_count(&mesh),
        0,
        "non-manifold edges in export mesh"
    );
}

#[test]
fn hs2x2_bin_socket_fuse_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let bin = load("hs2x2_socketfuse_body.bin", &mut topo);
    let sockets = load("hs2x2_socketfuse_sockets.bin", &mut topo);

    // The tool's export chain routes fuses through the evolution path, which
    // ships the GFA result without the plain `boolean` gates — exercise it.
    let (r, _evo) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, bin, sockets).unwrap();
    assert_fuse_health(&topo, r, 200, 700);
}

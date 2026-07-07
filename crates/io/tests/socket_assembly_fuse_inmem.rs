//! Faithful regression guard: the bin × socket-assembly fuse at the z=5 base
//! interface. Operands captured from the live gridfinity tool via the
//! boolean-capture probe kernel (arena-serialized at the boolean entry).
//!
//! The export chain fuses the bin body onto the assembly of base sockets. The
//! two coincide across the whole bottom plane: each socket's top face is a
//! chamfered-outline square that same-domain-cancels against the matching
//! region of the bin bottom, and the webbing between sockets stays exposed.
//! At the four bin corners the bin's rounded-corner arc (r = 3.75) overhangs
//! the socket outline's chamfer, leaving a ~0.1 mm crescent of bin bottom
//! that must survive as a real face.
//!
//! The failure this pins: the wire builder hands the corner crescents back as
//! CW loops, and the hole-promotion pass probed a SINGLE interior point to
//! decide nesting. On a 0.1 mm crescent the probe slips across the shared
//! boundary into the adjacent socket-square outer, so the crescent stayed a
//! "hole"; hole matching then probed its first vertex (exactly ON a
//! neighboring boundary — unpredictable strict ray-cast) and dumped all four
//! crescents onto the first sub-face, one of them geometrically unrelated.
//! That face is same-domain-dropped, so every corner crescent vanished:
//! free edges at all four corners. The compartmented variant regressed
//! further — GFA's result failed validation and the whole fuse fell back to
//! the mesh boolean, whose output was itself non-manifold (the `1×4 2×8`
//! compartments, `1.5×6 ±40°` and `2×6 halfSockets ±40°` scenario failures).
//!
//! Fixed by deciding hole nesting from the whole sampled boundary (a loop is
//! nested only when every sampled point is inside-or-on the outer) in both the
//! promotion pass and hole matching.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean, boolean_with_evolution};
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
fn compartment_bin_socket_fuse_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let bin = load("socketfuse_comps_bin.bin", &mut topo);
    let sockets = load("socketfuse_comps_sockets.bin", &mut topo);

    let r = boolean(&mut topo, BooleanOp::Fuse, bin, sockets).unwrap();
    assert_fuse_health(&topo, r, 20, 600);
}

#[test]
fn tilted_divider_bin_socket_fuse_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let bin = load("socketfuse_nohs_bin.bin", &mut topo);
    let sockets = load("socketfuse_nohs_sockets.bin", &mut topo);

    // The tool's export chain routes fuses through the evolution path, which
    // ships the GFA result without the plain `boolean` gates — exercise it.
    let (r, _evo) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, bin, sockets).unwrap();
    assert_fuse_health(&topo, r, 40, 900);
}

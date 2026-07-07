//! Faithful regression guard: the halfSockets base-clip cut. Operands captured
//! from the live gridfinity tool via the boolean-capture probe kernel
//! (arena-serialized at the boolean entry).
//!
//! A 2×6 halfSockets bin trims its base slab (a rounded-rect spanning
//! z ∈ [−2.6, 4.4]) with a flared clip tool: inset rounded rect at the bottom
//! (corner r = 2.55, 1.2 mm inside the slab's r = 3.75), full slab footprint at
//! the top, chamfer cones between. The cut leaves a 1.2 mm perimeter wall that
//! pinches out at the top rim.
//!
//! The failure this pins: the slab's bottom face splits into the inset disc
//! (coincident with the tool bottom — correctly same-domain-cancelled) and the
//! 1.2 mm ring that must remain as the wall's floor. The ring's classifier
//! seed was chosen against chord-approximated hole polygons; at the inset
//! rect's corner arcs the chord under-covers the hole by a ~0.75 mm sagitta,
//! so the seed landed between chord and arc — inside the true hole — and the
//! ring classified Inside the tool and was discarded. The open shell failed
//! validation and the whole cut fell back to the mesh boolean (515 planar
//! faces, all analytic surfaces lost), which poisoned every downstream fuse of
//! the export chain (the 2×6-halfSockets and crossing-tilt compartment
//! scenarios' non-manifold STL exports).
//!
//! Fixed by making the seed search's hole polygons arc-true: curved hole edges
//! are densified from their 3D curve through the plane frame instead of their
//! endpoint chords.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
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

#[test]
fn halfsockets_clip_cut_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let body = load("hsclip_cut_body.bin", &mut topo);
    let tool = load("hsclip_cut_tool.bin", &mut topo);

    let r = boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();

    let n_faces = solid_faces(&topo, r).unwrap().len();
    assert!(
        n_faces < 100,
        "clip cut returned {n_faces} faces — mesh fallback fired?"
    );
    let curved = curved_count(&topo, r);
    assert!(
        curved >= 10,
        "only {curved} curved faces — analytic surfaces lost?"
    );
    assert_eq!(
        free_edge_count(&topo, r),
        0,
        "free B-Rep edges in cut result"
    );

    let mesh = tessellate_solid_with_tolerance(&topo, r, 0.01, 5.0_f64.to_radians()).unwrap();
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

//! Faithful regression guard: the halfSockets bin body × stacking-lip fuse.
//! Operands captured from the live gridfinity tool via the boolean-capture
//! probe kernel (arena-serialized at the boolean entry).
//!
//! The body (a `2×6 halfSockets ±40°` bin at the stage where only its south
//! compartment cavity has been cut) carries a top ledge whose single hole is
//! the cavity opening, bounded on one side by the tilted divider's diagonal.
//! The lip fused onto it imprints its inner profile across that ledge: two
//! lines that cross the hole's diagonal edge mid-span, splitting the ledge
//! into the ring under the lip band and the region visible through the lip
//! throat.
//!
//! The failure this pins: the body's cavity cut emitted the ledge hole wound
//! the SAME way as the outer wire, and the splitter's hole weave trusts the
//! stored orientation. At the T-junctions where the lip profile crosses the
//! hole's diagonal, the angular wire builder then traced a double-cover: a
//! spurious CCW loop spanning the whole opening (kept as a membrane across
//! the bin throat) and the real throat-ledge region wound CW (hole-matched
//! onto a face that same-domain dropping erased). Every edge of the membrane
//! was left unpaired — `free=11`, 46 boundary edges and a non-manifold edge
//! in the export mesh — and the defect propagated through the final bin ×
//! socket-assembly fuse into the `2×6 halfSockets ±40` scenario's
//! non-manifold STL.
//!
//! Fixed by normalizing inner-wire winding (opposite the outer wire in UV)
//! where the wires enter the face splitter.

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

#[test]
fn halfsockets_body_lip_fuse_is_watertight_and_analytic() {
    let mut topo = Topology::new();
    let body = load("hslipfuse_body.bin", &mut topo);
    let lip = load("hslipfuse_lip.bin", &mut topo);

    // The tool's export chain routes fuses through the evolution path, which
    // ships the GFA result without the plain `boolean` gates — exercise it.
    let (r, _evo) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, body, lip).unwrap();

    let n_faces = solid_faces(&topo, r).unwrap().len();
    assert!(
        n_faces < 200,
        "fuse returned {n_faces} faces — mesh fallback fired?"
    );
    let curved = curved_count(&topo, r);
    assert!(
        curved >= 30,
        "only {curved} curved faces — analytic surfaces lost?"
    );
    assert_eq!(free_edge_count(&topo, r), 0, "free B-Rep edges in result");

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

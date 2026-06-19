//! Faithful coverage guard: the gridfinity multi-cavity shell cut is CLEAN.
//!
//! A 2×2 compartmented bin builds its divider walls via the "multi-cavity cut"
//! shell path: a rounded-corner box (`drawRoundedRectangle(outerW, outerD,
//! BOX_CORNER_RADIUS).extrude(wallHeight)`) is cut by one extruded cavity per
//! compartment, leaving the divider walls as the residue between cuts.
//!
//! Operands captured from the live tool via `serializeSolid` (#915): the
//! rounded box (10 faces, 4 corner cylinders) and four cavity solids (8 faces
//! each, one rounded exterior corner). This guard replays the production
//! sequential cut and asserts it stays watertight + analytic at every step.
//!
//! This is the negative half of the divider-lip bug
//! (`gridfinity_lipfuse_dividers_inmem.rs`): the cut is correct; the subsequent
//! lip fuse onto this body is what mesh-falls-back. Locking the cut down as
//! clean keeps the diagnosis honest — a regression here would mean the body
//! itself degraded, not the lip fuse.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_algo::gfa;
use brepkit_io::arena_io::deserialize_solid;
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

fn free_and_over(topo: &Topology, solid: SolidId) -> (usize, usize) {
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
    let free = faces_per_edge.values().filter(|f| f.len() == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

#[test]
fn multi_cavity_shell_cut_stays_watertight_and_analytic() {
    let mut topo = Topology::new();
    let mut body = load("cavitycut2x2_inmem_box.bin", &mut topo);

    let (free, over) = free_and_over(&topo, body);
    assert_eq!(
        (free, over),
        (0, 0),
        "rounded box must start watertight/manifold"
    );
    assert!(
        curved_count(&topo, body) >= 4,
        "rounded box must have its 4 vertical corner cylinders"
    );

    for i in 0..4 {
        let cavity = load(&format!("cavitycut2x2_inmem_cavity{i}.bin"), &mut topo);
        body = gfa::boolean(&mut topo, BooleanOp::Cut, body, cavity).unwrap();
        let (free, over) = free_and_over(&topo, body);
        assert_eq!(
            (free, over),
            (0, 0),
            "multi-cavity cut step {i} must stay watertight + manifold"
        );
    }

    // After 4 cuts: 4 box corner cylinders + 4 cavity exterior-corner cylinders.
    assert!(
        curved_count(&topo, body) >= 8,
        "final compartmented body must preserve all 8 corner cylinders (no mesh fallback)"
    );
    let faces = solid_faces(&topo, body).unwrap().len();
    assert!(
        faces < 60,
        "final compartmented body must be a compact analytic solid, got {faces} faces"
    );
}

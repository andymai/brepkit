//! Faithful regression guard for the baseplate dovetail groove cut.
//!
//! The gridfinity tool builds connecting baseplate tiles with dovetail joints:
//! a male trapezoidal tongue is fused onto one tile and a matching female groove
//! is cut into the neighbour. With `preferIdenticalPieces` a single tile carries
//! both, so the connector pipeline runs the groove `Cut` on a slab whose outer
//! wall is the gridfinity baseplate's slanted draft face.
//!
//! These `.bin` fixtures are the tool's EXACT in-memory operands for the groove
//! `Cut` (the 2×2 A1-canonical doubled-dovetail config), captured via the
//! `serializeSolid` wasm binding (byte-exact f64) and replayed through
//! `brepkit_io::arena_io::deserialize_solid`. The `_x` / `_y` pair are the two
//! join edges (right + back) of the corner tile.
//!
//! The bug: the groove trapezoid pokes through the slab's SLANTED outer wall.
//! When that wall (a planar face) is split, the planar arrangement is fed
//! foreign section edges — the groove's z=0 top edge and its tip walls, whose
//! endpoints sit up to ~1.3 mm OFF the slanted wall's plane. The arrangement
//! weaves those into a spurious corner-triangle sub-region that DUPLICATES the
//! groove flank's own triangle on the adjacent surface. After edge-merging the
//! two triangles reference the identical three edges, so every one of those
//! edges becomes incident to 3-4 faces (non-manifold over-share) — the tool
//! reported hundreds of non-manifold edges on the doubled-dovetail tiles and
//! fell back to a slow mesh repair.
//!
//! The fix drops doubled faces (two selected faces whose outer wires reference
//! the identical merged-edge set) in `builder_solid::build_solid`. This guard
//! asserts the groove cut is watertight (no free edges), manifold (no
//! over-shared edges), stays analytic (all-planar, no mesh fallback), and is
//! compact.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_algo::gfa;
use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
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

/// Free (used-once) and over-shared (incident to >2 distinct faces) boundary
/// edge counts, keyed by an orientation-independent quantized endpoint pair.
fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    let mut occ: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

fn assert_clean_groove_cut(base_name: &str, tool_name: &str, label: &str) {
    let mut topo = Topology::new();
    let base = load(base_name, &mut topo);
    let tool = load(tool_name, &mut topo);

    let result = gfa::boolean(&mut topo, BooleanOp::Cut, base, tool).unwrap();

    let (free, over) = edge_health(&topo, result);
    let faces = solid_faces(&topo, result).unwrap();
    let curved = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count();

    assert_eq!(
        over,
        0,
        "{label}: dovetail groove cut must be manifold (no over-shared edges); \
         got {} faces, {curved} curved, {free} free",
        faces.len()
    );
    assert_eq!(
        free,
        0,
        "{label}: dovetail groove cut must be watertight (no free edges); \
         got {} faces",
        faces.len()
    );
    assert_eq!(
        curved, 0,
        "{label}: the dovetail groove cut is all-planar; a curved face signals \
         a degraded result"
    );
    assert!(
        faces.len() < 40,
        "{label}: expected a compact analytic result, got {} faces (mesh fallback?)",
        faces.len()
    );
}

#[test]
fn dovetail_groove_cut_right_edge_is_watertight() {
    assert_clean_groove_cut(
        "dovetail_groove_base_x.bin",
        "dovetail_groove_tool_x.bin",
        "right-edge groove",
    );
}

#[test]
fn dovetail_groove_cut_back_edge_is_watertight() {
    assert_clean_groove_cut(
        "dovetail_groove_base_y.bin",
        "dovetail_groove_tool_y.bin",
        "back-edge groove",
    );
}

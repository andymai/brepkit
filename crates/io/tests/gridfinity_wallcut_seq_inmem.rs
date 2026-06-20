//! Faithful regression guard: sequential wall-cutout cuts on a 2×1-compartment
//! bin WITH stacking lip stay watertight. Before the #923 fix the second cut
//! dropped a contained back-wall and left free edges (the body mesh-fell-back).
//!
//! Operands captured from the live gridfinity tool (bin params: 2×2 width/depth,
//! height 4, 2×1 compartments, stacking lip, honeycomb wall pattern, four
//! u-shape wall cutouts) via the `serializeSolid` wasm binding.
//!
//! The tool's feature stage turns the four enabled u-shape wall cutouts into
//! TWO cut tools (`cut0`, `cut1`) and the honeycomb pattern into four pattern
//! cuts, then runs `Cut(Cut(body, cut0), cut1)` followed by the honeycomb cuts.
//!
//! Each wall-cutout cut is CLEAN on its own:
//!     Cut(body, cut0) → 150 faces, 52 curved, watertight
//!     Cut(body, cut1) → 120 faces, 44 curved, watertight
//! Both cutouts reach the same back-plane (x = ±28.385): cut0's tool face there
//! spans a wider y-range than cut1's, so once cut0 has carved its notch the
//! body carries a back-wall face that is fully contained in cut1's coincident
//! tool face. That opposite-oriented coincident pair used to hit the Cut
//! same-domain "overlapping → discard both" branch, dropping the body wall and
//! leaving its notch outline open:
//!     Cut(Cut(body, cut0), cut1) → 150 faces, **62 FREE edges**
//! The honeycomb pattern cut then ran on this non-watertight body and collapsed
//! it to 13 faces (tool mesh volume dropped ~60%, 21221 → 8497 mm³, and fell
//! back to mesh). The fix keeps the minuend's wall for an opposite-oriented
//! coincident Cut pair regardless of containment, so the sequence stays
//! watertight:
//!     Cut(Cut(body, cut0), cut1) → 156 faces, 0 free edges, analytic.

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

fn edge_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
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
                *occ.entry(key).or_insert(0) += 1;
            }
        }
    }
    let free = occ.values().filter(|&&c| c == 1).count();
    let over = faces_per_edge.values().filter(|f| f.len() > 2).count();
    (free, over)
}

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

#[test]
fn wallcut_each_cut_is_clean_individually() {
    // Each wall-cutout cut, applied alone to a fresh body, is watertight.
    for tool in ["wallcuthcomb2x1_cut0.bin", "wallcuthcomb2x1_cut1.bin"] {
        let mut topo = Topology::new();
        let body = load("wallcuthcomb2x1_body.bin", &mut topo);
        let t = load(tool, &mut topo);
        let r = gfa::boolean(&mut topo, BooleanOp::Cut, body, t).unwrap();
        let (free, over) = edge_health(&topo, r);
        assert_eq!(free, 0, "{tool} alone must leave a watertight body");
        assert_eq!(over, 0, "{tool} alone must leave a manifold body");
        assert!(
            curved_count(&topo, r) >= 44,
            "{tool} alone preserves the lip cones + corner cylinders"
        );
    }
}

#[test]
fn wallcut_sequential_cuts_stay_watertight() {
    let mut topo = Topology::new();
    let body = load("wallcuthcomb2x1_body.bin", &mut topo);
    let cut0 = load("wallcuthcomb2x1_cut0.bin", &mut topo);
    let cut1 = load("wallcuthcomb2x1_cut1.bin", &mut topo);

    let after0 = gfa::boolean(&mut topo, BooleanOp::Cut, body, cut0).unwrap();
    let (free0, over0) = edge_health(&topo, after0);
    assert_eq!(free0, 0, "first wall-cutout cut is clean");
    assert_eq!(over0, 0, "first wall-cutout cut is manifold");

    let after1 = gfa::boolean(&mut topo, BooleanOp::Cut, after0, cut1).unwrap();
    let (free1, over1) = edge_health(&topo, after1);

    // The second wall-cutout cut, sequenced after the first, must reconcile the
    // back-wall the first cut left coincident with the second tool's face.
    assert_eq!(
        free1, 0,
        "sequential wall-cutout cuts must stay watertight; got free={free1}"
    );
    assert_eq!(
        over1, 0,
        "sequential wall-cutout cuts must stay manifold; got over={over1}"
    );
    assert!(
        curved_count(&topo, after1) >= 44,
        "analytic surfaces (lip cones + corner cylinders) must be preserved"
    );
}

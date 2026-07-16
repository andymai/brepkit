//! Faithful guard for the fit-offset groove-mouth sliver family.
//!
//! A fit-offset baseplate export cuts six connector grooves into the plate's
//! interior walls. Each groove's mouth clips the corners of the two adjacent
//! socket-pocket openings, leaving zero-width slivers between the groove
//! outline and the pockets' r=4 rim circles on the top face. Three variants
//! of the same root appear as the chain progresses, because each cut absorbs
//! its mouth pockets' rings into the top face's OUTER wire (bays):
//!
//! - hole 0: both mouth rings are still INNER wires — the promotion pass must
//!   pull them into the planar arrangement (their pave-split vertices lie ON
//!   the section lines), and same-component regions must not be re-attached
//!   as holes (double cover) so the sliver is emitted and classified out.
//! - hole 1: one ring is a bay — the bay arc must be split at its true
//!   circle×section crossings (the arc's CHORD stays clear of the sections,
//!   so the arrangement's chord-based crossing detection misses them by the
//!   sagitta) or the notch region traces to a phantom corner inside air.
//! - hole 5: both rings are bays — nothing integrates, so the bay-mouth
//!   arrangement entry (boundary-arc crossings + multi-hole cap) must fire.
//!
//! Also exercised: the fourth-quadrant corner cones' seam-side UV endpoints
//! (projected to u=0 instead of 2π, flipping the boundary window to its
//! complement so the cut's conics dangled and the cone never split), and the
//! reversed-traversal plane-arc split normalization.
//!
//! Fixtures are the tool's EXACT serialized operands (3×3 fit-offset loose
//! +0.2 plate after all nub fuses, plus the six groove cutters).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::tessellate::tessellate_solid_with_tolerance;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn mesh_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let mesh = tessellate_solid_with_tolerance(topo, solid, 0.02, 6.0_f64.to_radians()).unwrap();
    let q = |v: f64| (v * 1.0e4).round() as i64;
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for t in mesh.indices.chunks(3) {
        for k in 0..3 {
            let a = t[k] as usize;
            let b = t[(k + 1) % 3] as usize;
            let vs = &mesh.positions;
            let pa = (q(vs[a].x()), q(vs[a].y()), q(vs[a].z()));
            let pb = (q(vs[b].x()), q(vs[b].y()), q(vs[b].z()));
            let key = if pa <= pb { (pa, pb) } else { (pb, pa) };
            *occ.entry(key).or_default() += 1;
        }
    }
    let bnd = occ.values().filter(|&&c| c == 1).count();
    let nm = occ.values().filter(|&&c| c > 2).count();
    (bnd, nm)
}

#[test]
fn groove_chain_stays_analytic_and_watertight() {
    let mut topo = Topology::new();
    let mut current = load("fitoffset_groove_base.bin", &mut topo);
    for i in 0..6 {
        let hole = load(&format!("fitoffset_groove_hole{i}.bin"), &mut topo);
        current = boolean(&mut topo, BooleanOp::Cut, current, hole).unwrap();
        let faces = solid_faces(&topo, current).unwrap().len();
        assert!(
            faces < 400,
            "groove cut {i}: face count {faces} signals a mesh fallback \
             (analytic chain stays under ~211 faces)"
        );
        let (bnd, nm) = mesh_health(&topo, current);
        assert_eq!(
            (bnd, nm),
            (0, 0),
            "groove cut {i}: must be watertight and manifold at export \
             tolerance; got bnd={bnd} nm={nm} ({faces} faces)"
        );
    }
}

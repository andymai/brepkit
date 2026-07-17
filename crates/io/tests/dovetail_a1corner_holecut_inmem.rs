//! Faithful guard for the A1-corner dovetail connector-recess hole cuts.
//!
//! The export chain cuts two rotated recess boxes into the fused corner plate.
//! Each box's slanted side wall receives a 4-section web from the doubled
//! tongue it crosses: a U-chain of three lines plus a plane×cone conic whose
//! far end lands mid-span of the z=0 line section (a T-junction). Two fit-error
//! conditioning gaps broke the split:
//!
//! - the marched conic's endpoint carries ~1e-6 of curve-fit error, so the
//!   T-junction sat above the 1e-7 vertex tolerance and the junction never
//!   formed (the weld now projects unmatched endpoints onto other Line
//!   sections' interiors);
//! - the planar-arrangement rescue bailed because its arc on-plane round-trip
//!   demanded 1e-7 while the fitted conic lies in the plane only to ~1e-6
//!   (the band is now the weld scale, still rejecting genuine straddle arcs).
//!
//! Un-rescued, the angular wire builder walked the CW-boundary slit-web as one
//! grand circuit (every section out-and-back), the cut failed the analytic
//! gate, and the mesh fallback exported a doubled coincident face pair (the
//! scenario's nm=2 STL pin).
//!
//! Fixtures are the tool's EXACT serialized operands (forExport=false variant:
//! post-nub-fuse corner plate + both recess boxes, cut in tool order).

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
fn recess_hole_cuts_stay_analytic_and_watertight() {
    let mut topo = Topology::new();
    let mut current = load("dovetail_a1corner_noexp_plate.bin", &mut topo);
    for name in ["dovetail_a1corner_hole0.bin", "dovetail_a1corner_hole1.bin"] {
        let hole = load(name, &mut topo);
        current = boolean(&mut topo, BooleanOp::Cut, current, hole).unwrap();
        let faces = solid_faces(&topo, current).unwrap().len();
        assert!(
            faces < 200,
            "{name}: face count {faces} signals a mesh fallback \
             (the analytic chain stays under ~70 faces)"
        );
        let (bnd, nm) = mesh_health(&topo, current);
        assert_eq!(
            (bnd, nm),
            (0, 0),
            "{name}: must stay watertight and manifold at export tolerance; \
             got bnd={bnd} nm={nm} ({faces} faces)"
        );
    }
}

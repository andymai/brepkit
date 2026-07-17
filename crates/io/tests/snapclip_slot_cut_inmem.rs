//! Faithful guard for the snap-slot hole cut into a socketed plate edge.
//!
//! A snapClip baseplate cuts 44 plain-box slot cutters into its join edges
//! after the socket pockets. The first slot's cut exercised four stacked
//! section-machinery gaps at the plate's edge-junction web (the face between
//! two socket-bite circles at a join-edge cell corner):
//!
//! - the per-face section clip picked the OUTERMOST crossing pair, which is
//!   only right for convex corner arcs — the web's INWARD-bulging bite
//!   circles made the chord crossings overshoot into air (sections extended
//!   0.65-4.6 past the true material);
//! - a section crossing the face in TWO material windows kept only one;
//! - plane×band (cylinder/cone) Line sections were never clipped to the
//!   band's v-window, arriving overlong on the slot wall;
//! - marched-NURBS section endpoints differ from their exact chain partners
//!   by the curve-fit error (~1e-6), above the 1e-7 vertex quantization, so
//!   the wall's silhouette chain never formed junctions and pendant removal
//!   dropped it whole.
//!
//! Together: 17 unpaired edges → GFA gate → mesh fallback (F=7356) whose
//! open output poisoned the remaining 43 slot cuts. Fixtures are the tool's
//! EXACT serialized operands (5×4 snapClip plate after all pocket cuts, plus
//! the first slot cutter).

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

#[test]
fn snap_slot_cut_stays_analytic_and_watertight() {
    type Q = (i64, i64, i64);
    let mut topo = Topology::new();
    let plate = load("snapclip_slot_plate.bin", &mut topo);
    let hole = load("snapclip_slot_hole.bin", &mut topo);
    let result = boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    let faces = solid_faces(&topo, result).unwrap().len();
    assert!(
        faces < 700,
        "slot cut: face count {faces} signals a mesh fallback (analytic result is ~603 faces)"
    );
    let mesh = tessellate_solid_with_tolerance(&topo, result, 0.02, 6.0_f64.to_radians()).unwrap();
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
    assert_eq!(
        (bnd, nm),
        (0, 0),
        "slot cut must be watertight and manifold; got bnd={bnd} nm={nm} ({faces} faces)"
    );
}

//! Faithful guard for the completed 4-way socket-junction disc.
//!
//! A baseplate's socket outline carries an r=4 relief circle CENTERED on each
//! cell corner; every cell's pocket contributes one quarter-cylinder of that
//! bore (a blind recess in the top band). Cutting the FOURTH cell around a
//! shared corner completes the circle, and the plate top must keep a DISC
//! face over it (outer wire = the four quarter rim arcs).
//!
//! The angular wire builder traces that completed circle as a standalone
//! two-arc closed loop whose pcurve-sampled polygon folds to ~zero area, so
//! the classifier's sliver guard silently dropped it — the loop COUNT matched
//! the planar arrangement's region count, the arrangement was declined, the
//! disc vanished, and the four rims went unpaired (GFA gate → mesh fallback
//! whose open output poisoned every later cut of the plate: the snapClip
//! join-edges export chain reached bnd≈1100). The arrangement gate now also
//! fires when any traced loop is area-degenerate.
//!
//! Fixtures are the tool's EXACT serialized operands: the 5×4 snapClip plate
//! slab and the four pockets around one interior cell corner, cut in tool
//! order (the fourth completes the junction).

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
fn completing_the_fourth_socket_keeps_the_junction_disc() {
    let mut topo = Topology::new();
    let mut current = load("snapclip_junction_slab.bin", &mut topo);
    for i in 0..4 {
        let pocket = load(&format!("snapclip_junction_pocket{i}.bin"), &mut topo);
        current = boolean(&mut topo, BooleanOp::Cut, current, pocket).unwrap();
        let faces = solid_faces(&topo, current).unwrap().len();
        assert!(
            faces < 200,
            "pocket cut {i}: face count {faces} signals a mesh fallback \
             (analytic chain stays under ~130 faces)"
        );
        let (bnd, nm) = mesh_health(&topo, current);
        assert_eq!(
            (bnd, nm),
            (0, 0),
            "pocket cut {i}: must stay watertight and manifold; got bnd={bnd} nm={nm} ({faces} faces)"
        );
    }
}

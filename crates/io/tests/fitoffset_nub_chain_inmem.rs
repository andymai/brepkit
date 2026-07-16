//! Faithful guard for the compound-relieved dovetail tongue (fit-offset tile).
//!
//! A 4×4 all-join interior baseplate relieves each interior tongue against
//! BOTH neighbouring cells' socket pockets — two sequential cuts. Fixing the
//! first cut (the tangency-cap family, see `dovetail_dblcorner_nub_inmem.rs`)
//! un-masked the second: the intermediate nub's plane faces now carry CURVED
//! boundary edges from the first cut (the pocket's rim arcs and conic), and
//! `trim_open_curve_to_plane_face_lines` declined such faces outright — its
//! all-straight-edges gate — so the second pocket's flank×cone conic fell to
//! the generic sample-clip, was dropped as a graze, the cone cap chain broke
//! (a triple-wound top-rim arc plus unpaired band rims), GFA failed
//! validation, and the mesh fallback emitted an OPEN mesh that poisoned the
//! export (fit-offset loose bnd=54). The gate now tolerates curved boundary
//! edges, computing exact crossings only against straight segments and
//! declining if a kept piece strays outside the sampled polygon mid-span
//! (which would have needed a crossing against a curved edge).
//!
//! Fixtures are the tool's EXACT serialized operands for one interior tongue
//! and both of its neighbour pockets.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
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

fn free_edges(topo: &Topology, solid: SolidId) -> usize {
    type Q = (i64, i64, i64);
    let s = 1.0e5;
    let q = |p: Point3| -> Q {
        (
            (p.x() * s).round() as i64,
            (p.y() * s).round() as i64,
            (p.z() * s).round() as i64,
        )
    };
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    occ.values().filter(|&&c| c != 2).count()
}

fn mesh_health(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let mesh = tessellate_solid_with_tolerance(topo, solid, 0.01, 5.0_f64.to_radians()).unwrap();
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

fn check(
    topo: &Topology,
    solid: SolidId,
    faces_expected: usize,
    cones: usize,
    cyls: usize,
    label: &str,
) {
    let faces = solid_faces(topo, solid).unwrap();
    let count_of = |tag: &str| {
        faces
            .iter()
            .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == tag)
            .count()
    };
    let unpaired = free_edges(topo, solid);
    assert_eq!(
        unpaired,
        0,
        "{label}: must be watertight and manifold; got {unpaired} position-edges not used \
         exactly twice ({} faces)",
        faces.len()
    );
    assert_eq!(faces.len(), faces_expected, "{label}: face count");
    assert_eq!(count_of("cone"), cones, "{label}: analytic cone caps");
    assert_eq!(
        count_of("cylinder"),
        cyls,
        "{label}: analytic cylinder caps"
    );
    let (bnd, nm) = mesh_health(topo, solid);
    assert_eq!(
        (bnd, nm),
        (0, 0),
        "{label}: mesh must be watertight/manifold"
    );
}

#[test]
fn compound_relief_chain_stays_analytic() {
    let mut topo = Topology::new();
    let tongue = load("fitoffset_nub_tongue.bin", &mut topo);
    let p0 = load("fitoffset_nub_pocket_0.bin", &mut topo);
    let p1 = load("fitoffset_nub_pocket_1.bin", &mut topo);

    let first = boolean(&mut topo, BooleanOp::Cut, tongue, p0).unwrap();
    check(&topo, first, 8, 1, 1, "first relief cut");

    let second = boolean(&mut topo, BooleanOp::Cut, first, p1).unwrap();
    check(&topo, second, 10, 2, 2, "second relief cut");
}

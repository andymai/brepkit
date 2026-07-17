//! Faithful guard for the A1-corner dovetail nub fuse.
//!
//! The dovetail baseplate's corner nub is fused onto a plate whose cell-corner
//! junction carries an r=4 relief circle TANGENT to the plate wall x=42 at
//! (42,−4). The fuse must drop the plate wall's notched middle strip whole:
//! nub material sits flush against its entire east side (the bore touches the
//! wall only along the tangency meridian), so the strip is interior.
//!
//! The strip's splitter-computed interior point lands on (42, −4, −1.75) — the
//! intersection of THREE axis-aligned feature planes (the wall, the tangency /
//! profile-seam meridian, the dovetail flare plane). Every cardinal
//! classification ray from there runs along edges, seams, and the tangency
//! line, so all three parities were garbage and the strip classified Outside —
//! a kept interior membrane, a non-manifold analytic result, and a mesh
//! fallback for the whole plate chain. The classifier now detects that all
//! three cardinal rays grazed degenerate structure and re-votes with fixed
//! generic directions.
//!
//! Fixtures are the tool's EXACT serialized operands (corner-intersected 2×2
//! plate + the relieved corner nub).

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
fn a1_corner_nub_fuse_stays_analytic_and_watertight() {
    let mut topo = Topology::new();
    let plate = load("dovetail_a1corner_plate.bin", &mut topo);
    let nub = load("dovetail_a1corner_nub.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, plate, nub).unwrap();

    let faces = solid_faces(&topo, result).unwrap().len();
    assert!(
        faces < 300,
        "nub fuse: face count {faces} signals a mesh fallback \
         (the analytic result stays under ~150 faces)"
    );
    let (bnd, nm) = mesh_health(&topo, result);
    assert_eq!(
        (bnd, nm),
        (0, 0),
        "nub fuse must be watertight and manifold at export tolerance; \
         got bnd={bnd} nm={nm} ({faces} faces)"
    );
}

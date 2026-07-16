//! Faithful guard for the seam-edge (flush-wall) pocket cut.
//!
//! A fractional-width baseplate tile on a split seam extends its seam-edge
//! pockets all the way to the tile boundary, so the pocket's straight wall is
//! EXACTLY coplanar with the slab's outer wall and the opening must merge
//! into the boundary as a notch. The cut mesh-fell-back (F=122, 44 open
//! boundary edges at export tolerance) and poisoned the whole fractional
//! plate build (the dovetail `5×4.5 edge-y-1` scenario, nm=28 exported).
//!
//! Root: `find_point_outside_holes` built its hole-rejection polygon from the
//! stored `start_uv` of each hole edge, and a hole-wire vertex whose UV was
//! fitted in a DIFFERENT plane frame corrupted the polygon — the even-odd
//! test then accepted classifier seeds INSIDE the opening, both top
//! sub-faces sampled the cutter's interior, and the entire slab top was
//! dropped (Euler gate -> mesh fallback). With a frame available every
//! vertex is now derived from 3D, the same doctrine the function already
//! applied to curved-edge sampling.
//!
//! Fixtures are the tool's EXACT serialized operands (slab + the seam-edge
//! pocket of the 5×4.5 fractional tile).

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
fn seam_edge_pocket_cut_stays_analytic() {
    let mut topo = Topology::new();
    let slab = load("fracplate_slab.bin", &mut topo);
    let pocket = load("fracplate_seam_pocket.bin", &mut topo);
    let result = boolean(&mut topo, BooleanOp::Cut, slab, pocket).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
        .count();
    assert_eq!(
        faces.len(),
        14,
        "flush-wall pocket cut must stay analytic (got {} faces)",
        faces.len()
    );
    assert_eq!(cones, 4, "the pocket's corner cones must survive");
    let (bnd, nm) = mesh_health(&topo, result);
    assert_eq!((bnd, nm), (0, 0), "result must be watertight/manifold");
}

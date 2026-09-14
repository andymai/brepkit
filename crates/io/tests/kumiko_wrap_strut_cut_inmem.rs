//! Kumiko corner wrap from the gridfinity tool (`slideRailBuilder.test.ts`,
//! "is not carved away by a kumiko wrap either": a 3x2x6 bin with the
//! mitsukude wall pattern and interior slide rails), captured on brepkit-wasm
//! 3.4.0 and reproduced identically on brepjs 19.0.4.
//!
//! The corner band is a 6-face solid (2 cylinders, 4 planes). Each tilted
//! corner strut is a rectangle swept along a helix segment: 16 NURBS wall
//! patches per wall, 64 NURBS faces plus 2 planar caps. Cutting the band by
//! one strut spends 4.2 s in the face-face phase, the marcher reports section
//! curves deviating up to 1.8 mm from the surfaces before re-fitting, and the
//! exact result has 42 free edges (35 on the strut's NURBS sub-faces), so the
//! op falls back to a 167-face planar mesh. The export runs 21 such compound
//! cuts (532 s, 23 fallbacks). The reference kernel exports it watertight.
//!
//! Data: `kumiko_wrap_band.bin` (cut base), `kumiko_wrap_strut.bin` (one
//! of the three tools of the captured compound cut).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(topo: &mut Topology, name: &str) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn surface_census(topo: &Topology, solid: SolidId) -> HashMap<&'static str, usize> {
    let mut census = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let tag = match topo.face(fid).unwrap().surface() {
            FaceSurface::Plane { .. } => "plane",
            FaceSurface::Cylinder(_) => "cylinder",
            FaceSurface::Cone(_) => "cone",
            FaceSurface::Sphere(_) => "sphere",
            FaceSurface::Torus(_) => "torus",
            FaceSurface::Nurbs(_) => "nurbs",
        };
        *census.entry(tag).or_insert(0) += 1;
    }
    census
}

fn edge_midpoint(topo: &Topology, eid: EdgeId) -> Point3 {
    let e = topo.edge(eid).unwrap();
    let (a, b) = (
        topo.vertex(e.start()).unwrap().point(),
        topo.vertex(e.end()).unwrap().point(),
    );
    let (t0, t1) = e.curve().domain_with_endpoints(a, b);
    e.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b)
}

/// Edge uses keyed by quantized geometry (endpoints and midpoint), so two ids
/// on one curve count as one edge.
#[allow(clippy::cast_possible_truncation)]
fn positional_edge_uses(topo: &Topology, solid: SolidId) -> HashMap<[i64; 9], usize> {
    let q = |p: Point3| {
        [
            (p.x() * 1e6).round() as i64,
            (p.y() * 1e6).round() as i64,
            (p.z() * 1e6).round() as i64,
        ]
    };
    let mut uses = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let (mut a, mut b) = (
                    q(topo.vertex(e.start()).unwrap().point()),
                    q(topo.vertex(e.end()).unwrap().point()),
                );
                if a > b {
                    std::mem::swap(&mut a, &mut b);
                }
                let m = q(edge_midpoint(topo, oe.edge()));
                *uses
                    .entry([a[0], a[1], a[2], b[0], b[1], b[2], m[0], m[1], m[2]])
                    .or_insert(0) += 1;
            }
        }
    }
    uses
}

#[test]
fn kumiko_corner_fixture_is_faithful() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_band.bin");
    let strut = load(&mut topo, "kumiko_wrap_strut.bin");
    let band_census = surface_census(&topo, band);
    assert_eq!(band_census.get("cylinder"), Some(&2));
    assert_eq!(band_census.get("plane"), Some(&4));
    let strut_census = surface_census(&topo, strut);
    assert_eq!(strut_census.get("nurbs"), Some(&64));
    assert_eq!(strut_census.get("plane"), Some(&2));
}

#[test]
#[ignore = "ready repro: the band cut by a helical-sweep strut must stay exact (the marcher's sections drift off the surfaces); the 4.4 s it takes today is the bench harness's concern"]
fn kumiko_corner_strut_cut_stays_exact() {
    let mut topo = Topology::new();
    let band = load(&mut topo, "kumiko_wrap_band.bin");
    let strut = load(&mut topo, "kumiko_wrap_strut.bin");
    let before = boolean::mesh_fallback_count();
    let result = boolean::boolean(&mut topo, BooleanOp::Cut, band, strut).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        before,
        "the cut took the mesh fallback"
    );
    let census = surface_census(&topo, result);
    assert!(
        census.get("cylinder").copied().unwrap_or(0) >= 2
            && census.get("nurbs").copied().unwrap_or(0) > 0,
        "cut result lost its typed faces: {census:?}"
    );
    let uses = positional_edge_uses(&topo, result);
    assert_eq!(
        uses.values().filter(|&&c| c != 2).count(),
        0,
        "cut result must be manifold by position"
    );
}

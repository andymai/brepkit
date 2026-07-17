//! Guard for the plane×cone exact CIRCLE arc at a relief-cutter back corner
//! (the snapClip export-variant op-cut-3 root).
//!
//! The export baseplate uses the simplified tapered pockets, so the deep
//! relief cutter's back corner lands on the pocket's corner CONE. The cutter
//! contributes three sections to the cone face: two marched conics (cone ×
//! back wall, cone × side wall) and the cone × cutter-top CIRCLE arc closing
//! the chain. That horizontal-plane section is an exact `EdgeCurve::Circle`,
//! and `trim_ellipse_to_boundary_crossings` — the exact-arc path that
//! bypasses the sampling pre-filters — only accepted Ellipse sections, so
//! the ~0.11-long arc (2% of the full circle) was dropped by the 16-sample
//! in-both filter. The cone face then received an OPEN 2-conic chain, the
//! internal-loops splitter rejected it, the face stayed unsplit, and the
//! analytic-but-leaky result poisoned the export chain into mesh fallback
//! two cuts later (final bnd=111 nm=6 at export tolerance).
//!
//! Fixtures are the tool's serialized operands (2026-07-17 export-variant
//! capture: plate after op-cut-2, first deep relief cutter).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_algo::bop::BooleanOp;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    brepkit_io::arena_io::deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

type Q = (i64, i64, i64);

#[test]
fn export_corner_cut_pairs_every_brep_edge() {
    let mut topo = Topology::new();
    let plate = load("snapclip_export_corner_plate.bin", &mut topo);
    let cutter = load("snapclip_export_corner_cutter.bin", &mut topo);

    let result = brepkit_algo::gfa::boolean(&mut topo, BooleanOp::Cut, plate, cutter).unwrap();

    let faces = solid_faces(&topo, result).unwrap();
    let cones = faces
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cone(_)))
        .count();
    assert!(
        cones >= 80,
        "tapered pockets must stay analytic, got {cones} cones"
    );

    // Position-quantized B-Rep edge pairing: every edge used exactly twice.
    // The missing circle arc left the corner cones unsplit — their partners'
    // conic sections and the cutter-top ledge rims dangled (posBad=8).
    let sc = 1.0e5;
    let q = |v: f64| (v * sc).round() as i64;
    let qp = |p: Point3| (q(p.x()), q(p.y()), q(p.z()));
    let mut occ: HashMap<(Q, Q), usize> = HashMap::new();
    for &fid in &faces {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = qp(topo.vertex(e.start()).unwrap().point());
                let b = qp(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *occ.entry(key).or_default() += 1;
            }
        }
    }
    let bad: Vec<_> = occ.iter().filter(|&(_, &c)| c != 2).collect();
    assert!(
        bad.is_empty(),
        "unpaired/overused B-Rep edges at the cutter back corner: {} entries",
        bad.len()
    );
}

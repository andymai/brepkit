//! Regression test for the gridfinity front/back multi-cavity compartment cut.
//!
//! Cutting two adjacent rounded-rect cavities (exported from the gridfinity
//! layout tool, hence NURBS-cornered) through the top cap of a bin box leaves
//! a thin shared divider. The second cut turns the top cap into a frame with
//! two holes; classifying that frame used to bounce its interior sample into
//! the second hole, drop the frame, and leave the result non-manifold — which
//! sent the operation to the mesh-boolean fallback and ballooned the face
//! count (~26 analytic faces -> thousands of planar facets).
//!
//! The fixtures are the tool's literal operands; the bug is sensitive to the
//! exact (spline-cornered) cavity geometry, so a native arc-cornered rebuild
//! does not reproduce it.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn read_one(name: &str, topo: &mut Topology) -> SolidId {
    let text = std::fs::read_to_string(fixture(name)).unwrap();
    let solids = brepkit_io::step::reader::read_step(&text, topo).unwrap();
    assert_eq!(solids.len(), 1, "expected exactly one solid in {name}");
    solids[0]
}

/// Count boundary edges used by exactly one wire occurrence (free edges).
/// Keys edges by their quantized, orientation-independent endpoint pair so
/// distinct topological edges sharing a curve still count as shared.
fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e5;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut counts: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts.values().filter(|&&c| c == 1).count()
}

#[test]
fn gridfinity_frontback_multicavity_cut_is_watertight() {
    let mut topo = Topology::new();
    let bin = read_one("multicavity_bin_box.step", &mut topo);
    let cav0 = read_one("multicavity_cavity_0.step", &mut topo);
    let cav1 = read_one("multicavity_cavity_1.step", &mut topo);

    let cut1 = boolean(&mut topo, BooleanOp::Cut, bin, cav0).unwrap();
    assert_eq!(
        free_edge_count(&topo, cut1),
        0,
        "first cavity cut must be watertight"
    );

    let result = boolean(&mut topo, BooleanOp::Cut, cut1, cav1).unwrap();
    assert_eq!(
        free_edge_count(&topo, result),
        0,
        "two-cavity cut must be watertight (no mesh-boolean fallback)"
    );

    // A clean analytic result is a few dozen faces; the mesh-boolean fallback
    // produces thousands of planar facets. Guard the regression with a generous
    // ceiling that still distinguishes the two by orders of magnitude.
    let face_count = solid_faces(&topo, result).unwrap().len();
    assert!(
        face_count < 100,
        "expected a compact analytic result, got {face_count} faces (mesh fallback?)"
    );
}

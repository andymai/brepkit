//! Regression test for the gridfinity scoop-fuse-into-compartmented-bin
//! non-manifold (corner-cylinder half).
//!
//! A faceted concave scoop-ramp prism is fused into a no-lip compartmented
//! bin. The scoop's sharp staircase pokes into the bin's rounded corner
//! cylinders, so each tread plane meets a corner cylinder in a sub-millimetre
//! ellipse arc. Those per-tread arcs must assemble into one continuous split
//! curve so the cylinder splits into a lower band (engulfed by the scoop →
//! dropped) and an upper band (faces the open cavity → kept). When the arcs do
//! not chain, the cylinder stays whole, its interior sample lands inside the
//! scoop, and it is dropped entirely → free edges → mesh-boolean fallback.
//!
//! The bin walls are exported by STEP as B-splines; the native tool builds
//! them as `Plane`+`Cylinder` via rounded-rect extrude, so `convert_to_elementary`
//! recovers the faithful (plane-walled) geometry before the fuse.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::heal::convert_to_elementary;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
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

type QPoint = (i64, i64, i64);
fn qp(p: brepkit_math::vec::Point3) -> QPoint {
    let s = 1.0e5;
    (
        (p.x() * s).round() as i64,
        (p.y() * s).round() as i64,
        (p.z() * s).round() as i64,
    )
}

/// Count how many faces each undirected edge (by quantized endpoints) borders.
fn edge_incidence(topo: &Topology, solid: SolidId) -> HashMap<(QPoint, QPoint), usize> {
    let mut counts: HashMap<(QPoint, QPoint), usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid).unwrap();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = qp(topo.vertex(e.start()).unwrap().point());
                let b = qp(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                *counts.entry(key).or_insert(0) += 1;
            }
        }
    }
    counts
}

#[test]
fn scoop_fuse_corner_cylinders_are_watertight() {
    let mut topo = Topology::new();
    let bin = read_one("scoop_comp_bin.step", &mut topo);
    let scoop = read_one("scoop_scoop_0.step", &mut topo);

    // Recover the faithful plane-walled bin (STEP export turned the planar
    // walls into B-splines; the corners stay analytic cylinders).
    convert_to_elementary(&mut topo, bin, 1e-5).unwrap();

    let result = boolean(&mut topo, BooleanOp::Fuse, bin, scoop).unwrap();

    // Watertight: every edge bordered by exactly two faces.
    let counts = edge_incidence(&topo, result);
    let free = counts.values().filter(|&&c| c == 1).count();
    let over = counts.values().filter(|&&c| c > 2).count();
    assert_eq!(
        free, 0,
        "fuse result has {free} free edges (corner cylinders dropped)"
    );
    assert_eq!(over, 0, "fuse result has {over} over-shared edges");

    // No mesh-boolean fallback: the GFA result must survive as analytic
    // geometry. The fallback tessellates everything to planes (hundreds of
    // faces); the analytic fuse keeps the 8 corner cylinders and ~50 planes.
    let faces = solid_faces(&topo, result).unwrap();
    let cylinders = faces
        .iter()
        .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(
        cylinders, 8,
        "expected 8 analytic corner cylinders to survive (got {cylinders}); \
         a mesh-boolean fallback would tessellate them away"
    );
    assert!(
        faces.len() < 100,
        "expected a small analytic result, got {} faces (mesh fallback?)",
        faces.len()
    );
}

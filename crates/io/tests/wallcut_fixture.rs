//! Regression test for the gridfinity "2×2 wall cutouts" (U-notch) bin.
//!
//! Each of the four bin walls is carved by a rounded-rect prism that opens at
//! the rim (the cut runs from inside the wall up through the top edge). The four
//! prisms are fused into one multi-shell tool and cut from the shelled body in a
//! single boolean. The wall faces then carry a U-notch — a rectangular bite with
//! rounded bottom corners — whose top edge IS the rim boundary (open at the
//! top). The planar arrangement splitter used to reject those faces (their
//! corner arcs are curved) and fall back to the angular wire builder, which
//! traced one self-crossing loop per wall, left ~96 free edges, and forced the
//! whole cut to the mesh-boolean fallback (158 all-planar facets instead of an
//! analytic solid).
//!
//! The fixtures are the tool's literal operands (brepjs sketch+extrude geometry,
//! exported via STEP). A native arc-cornered rebuild does not reproduce the
//! exact prism/seam structure, so the regression is guarded with these.

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

/// Count boundary edges used by exactly one wire occurrence (free) and by more
/// than two (over-shared), keyed by the quantized orientation-independent
/// endpoint pair so distinct topological edges on the same curve still match.
fn edge_use(topo: &Topology, solid: SolidId) -> (usize, usize) {
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
    let free = counts.values().filter(|&&c| c == 1).count();
    let over = counts.values().filter(|&&c| c > 2).count();
    (free, over)
}

fn cylinder_face_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() == "cylinder")
        .count()
}

#[test]
fn gridfinity_wall_cutouts_cut_is_watertight() {
    let mut topo = Topology::new();
    let body = read_one("wallcut_body.step", &mut topo);
    let tool = read_one("wallcut_tool_0.step", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Cut, body, tool).unwrap();

    let (free, over) = edge_use(&topo, result);
    assert_eq!(
        free, 0,
        "wall-cutout cut must be watertight (no free edges)"
    );
    assert_eq!(
        over, 0,
        "wall-cutout cut must be manifold (no over-shared edges)"
    );

    // A clean analytic result is a few dozen faces with the corner cylinders
    // preserved; the mesh-boolean fallback produces ~158 all-planar facets.
    let faces = solid_faces(&topo, result).unwrap().len();
    assert!(
        faces < 100,
        "expected a compact analytic result, got {faces} faces (mesh fallback?)"
    );
    assert!(
        cylinder_face_count(&topo, result) >= 8,
        "wall-cutout cut must stay analytic (body + notch corner cylinders preserved)"
    );
}

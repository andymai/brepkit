//! Faithful regression guards: stacking-lip fuse onto compartmented bodies
//! whose FF sections re-trace the lip ring's own boundary. Operands captured
//! from the live gridfinity tool via the boolean-capture probe kernel
//! (arena-serialized at `boolean_with_evolution` entry).
//!
//! Two operand pairs pin two failure modes of the same root — feeding a
//! section that lies ON a face's existing boundary edge into the planar
//! arrangement:
//!
//! 1. **Tilted-divider bin** (`tiltdiv_*`): a 1.5×6×6 bin with a ±40 mm tilted
//!    divider. The body's corner cylinders meet the lip's bottom plane exactly
//!    at its flush outer rim, so the FF intersection arcs re-trace the lip
//!    ring's own outer corner arcs. Threading them wove a snake wire that
//!    multiply-traversed the rim; the shell flood-fill saw every junction edge
//!    as already-manifold and orphaned an 85-face fragment (lip + divider +
//!    cavity walls), which the open-hole-shell guard then dropped — the fuse
//!    returned the 14-face body exterior with the entire interior missing.
//!    Two further defects hid behind it: the divider cap's bridge sections
//!    were discarded as "pure air" because uniform hole probes missed the
//!    1.2 mm cap sliver, and the cap's tilted cavity rims are straight
//!    NurbsCurve edges the hole weave refused to split.
//!
//! 2. **Top-row-merged bin** (`toprow_*`): a 2×2 bin with the top row merged
//!    into one compartment. The body's inner-wall planes intersect the lip's
//!    bottom plane exactly along its flush inner rim; those PaveBlock sections
//!    re-traced the ring's hole boundary, the arrangement wove a degenerate
//!    region, and the whole fuse fell back to the mesh boolean (1276 planar
//!    faces), poisoning every downstream op of the export chain.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load(name: &str, topo: &mut Topology) -> SolidId {
    deserialize_solid(&std::fs::read(fixture(name)).unwrap(), topo).unwrap()
}

fn curved_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() != "plane")
        .count()
}

fn free_edge_count(topo: &Topology, solid: SolidId) -> usize {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
    let q = |p: brepkit_math::vec::Point3| -> QPoint {
        (
            (p.x() * scale).round() as i64,
            (p.y() * scale).round() as i64,
            (p.z() * scale).round() as i64,
        )
    };
    let mut faces_per_edge: HashMap<(QPoint, QPoint), HashSet<FaceId>> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                let a = q(topo.vertex(e.start()).unwrap().point());
                let b = q(topo.vertex(e.end()).unwrap().point());
                let key = if a <= b { (a, b) } else { (b, a) };
                faces_per_edge.entry(key).or_default().insert(fid);
            }
        }
    }
    faces_per_edge.values().filter(|f| f.len() == 1).count()
}

fn assert_lipfuse_clean(body_fix: &str, lip_fix: &str, min_faces: usize, label: &str) {
    let mut topo = Topology::new();
    let body = load(body_fix, &mut topo);
    let lip = load(lip_fix, &mut topo);

    let r = boolean(&mut topo, BooleanOp::Fuse, body, lip).unwrap();

    let n_faces = solid_faces(&topo, r).unwrap().len();
    assert!(
        n_faces >= min_faces,
        "{label}: fuse returned {n_faces} faces (< {min_faces}) — interior faces dropped?"
    );
    let curved = curved_count(&topo, r);
    assert!(
        curved >= 10,
        "{label}: only {curved} curved faces — mesh fallback fired?"
    );
    assert_eq!(
        free_edge_count(&topo, r),
        0,
        "{label}: free B-Rep edges in fuse result"
    );

    // Export-tolerance mesh must be watertight (0.01 mm / 5° matches the
    // gridfinity tool's STL export tier).
    let mesh = tessellate_solid_with_tolerance(&topo, r, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(
        boundary_edge_count(&mesh),
        0,
        "{label}: boundary edges in export-tolerance mesh"
    );
    assert_eq!(
        non_manifold_edge_count(&mesh),
        0,
        "{label}: non-manifold edges in export-tolerance mesh"
    );
}

#[test]
fn tilted_divider_lip_fuse_is_watertight_and_analytic() {
    assert_lipfuse_clean(
        "tiltdiv_lipfuse_body.bin",
        "tiltdiv_lipfuse_lip.bin",
        90,
        "tilted-divider lip fuse",
    );
}

#[test]
fn toprow_merged_lip_fuse_is_watertight_and_analytic() {
    assert_lipfuse_clean(
        "toprow_lipfuse_body.bin",
        "toprow_lipfuse_lip.bin",
        100,
        "top-row-merged lip fuse",
    );
}

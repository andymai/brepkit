//! Faithful regression guard for the body × stacking-lip fuse (Failure A).
//!
//! The `lipfuse_3x3_*.step` fixtures (see `lipfuse_fixture.rs`) fuse CLEAN
//! because STEP's ~15-sig-fig serialization rounds the kernel's in-memory
//! sub-ULP vertex noise away. The gridfinity tool's actual in-memory operands
//! do NOT — before this fix the body×lip fuse fell back to a 238-facet
//! all-planar mesh (213 free edges).
//!
//! These `.bin` fixtures are the tool's EXACT in-memory operands, captured via
//! the `serializeSolid` wasm binding (byte-exact f64) and replayed here through
//! `brepkit_io::arena_io::deserialize_solid`. They are the first faithful
//! committed scoop-family guard: unlike the STEP fixtures, they reproduce the
//! production fallback.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_io::arena_io::deserialize_solid;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name)
}

fn load_inmem(name: &str, topo: &mut Topology) -> SolidId {
    let bytes = std::fs::read(fixture(name)).unwrap();
    deserialize_solid(&bytes, topo).unwrap()
}

/// Free (used-once) and over-shared (used >2×) boundary-edge counts, keyed by
/// an orientation-independent quantized endpoint pair.
fn edge_use(topo: &Topology, solid: SolidId) -> (usize, usize) {
    type QPoint = (i64, i64, i64);
    let scale = 1.0e6;
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

fn curved_face_count(topo: &Topology, solid: SolidId) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .filter(|&&fid| topo.face(fid).unwrap().surface().type_tag() != "plane")
        .count()
}

#[test]
fn gridfinity_lip_fuse_3x3_inmem_is_watertight() {
    let mut topo = Topology::new();
    let body = load_inmem("lip3x3_inmem_body.bin", &mut topo);
    let lip = load_inmem("lip3x3_inmem_lip.bin", &mut topo);

    let result = boolean(&mut topo, BooleanOp::Fuse, body, lip).unwrap();

    let (free, over) = edge_use(&topo, result);
    let faces = solid_faces(&topo, result).unwrap().len();
    let curved = curved_face_count(&topo, result);

    assert_eq!(
        free, 0,
        "in-memory 3×3 lip fuse must be watertight (no free edges); got {faces} faces, {curved} curved"
    );
    assert_eq!(
        over, 0,
        "in-memory 3×3 lip fuse must be manifold (no over-shared edges)"
    );
    assert!(
        faces < 120,
        "expected a compact analytic result, got {faces} faces (mesh fallback?)"
    );
    assert!(
        curved >= 24,
        "in-memory 3×3 lip fuse must stay analytic (corner cylinders + lip cones preserved); got {curved} curved"
    );
}

//! Guard for the gridfinity stacking-lip corner near-duplicate vertices,
//! replayed from the tool's literal kernel operands (captured via the arena
//! serializer during a tool export probe).
//!
//! The lip solid's peak corner ARC ends exactly on the tangent line where the
//! body's coincident outer-wall PLANE touches the lip's corner cylinder. The
//! EF (edge-face) interference phase intersects that arc with the wall plane;
//! the contact is tangential (grazing), so the numeric crossing's position
//! along the arc is only accurate to sqrt-of-residual — it solved to a point
//! ~3.1 µm along the arc from the true endpoint (residual ~1e-12) and minted
//! a NEW vertex there, 3.1 µm from the arc's own endpoint vertex. The mesh
//! stays index-watertight (a micron-wide sliver triangle bridges the pair),
//! but any consumer that welds vertices at practical tolerance — the STL
//! quantization the gridfinity tool and slicers use — sees the sliver's two
//! long edges collapse onto one another: 8 "non-manifold" STL edges, two per
//! lip corner (the compartment scenario export failures).
//!
//! The fix widens the EF endpoint-contact drop window by the crossing angle
//! (`tol / |tangent · normal|`, capped), so a grazing contact at an endpoint
//! is recognized as the vertex-face incidence it is.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean_with_evolution};
use brepkit_operations::tessellate::{
    TriangleMesh, boundary_edge_count, non_manifold_edge_count, tessellate_solid_with_tolerance,
};
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

/// Count B-Rep vertex pairs that are distinct positions yet closer than
/// `dist` — the near-duplicate class that collapses under consumer welding.
fn near_duplicate_vertex_pairs(topo: &Topology, solid: SolidId, dist: f64) -> usize {
    let mut positions: Vec<Point3> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                for vid in [e.start(), e.end()] {
                    if seen.insert(vid.index()) {
                        positions.push(topo.vertex(vid).unwrap().point());
                    }
                }
            }
        }
    }
    let mut pairs = 0;
    for i in 0..positions.len() {
        for j in (i + 1)..positions.len() {
            let d = (positions[i] - positions[j]).length();
            if d > 1e-9 && d < dist {
                pairs += 1;
            }
        }
    }
    pairs
}

/// The gridfinity tool's STL manifold oracle: quantize vertices at 1e-4 and
/// count edges used once (boundary) or more than twice (non-manifold).
fn quantized_mesh_defects(mesh: &TriangleMesh, quantize: f64) -> (usize, usize) {
    type Q = (i64, i64, i64);
    let q = |p: Point3| -> Q {
        (
            (p.x() / quantize).round() as i64,
            (p.y() / quantize).round() as i64,
            (p.z() / quantize).round() as i64,
        )
    };
    let mut counts: HashMap<(Q, Q), usize> = HashMap::new();
    for t in mesh.indices.chunks_exact(3) {
        let keys = [
            q(mesh.positions[t[0] as usize]),
            q(mesh.positions[t[1] as usize]),
            q(mesh.positions[t[2] as usize]),
        ];
        for i in 0..3 {
            let (a, b) = (keys[i], keys[(i + 1) % 3]);
            if a == b {
                continue; // degenerate under quantization; counted via its long edges
            }
            let key = if a <= b { (a, b) } else { (b, a) };
            *counts.entry(key).or_default() += 1;
        }
    }
    let boundary = counts.values().filter(|&&c| c == 1).count();
    let non_manifold = counts.values().filter(|&&c| c > 2).count();
    (boundary, non_manifold)
}

#[test]
fn lip_fuse_has_no_near_duplicate_corner_vertices() {
    let mut topo = Topology::new();
    let body = load("lipcorner_tangent_body.bin", &mut topo);
    let lip = load("lipcorner_tangent_lip.bin", &mut topo);

    // The tool's export pipeline runs the provenance-tracking fuse.
    let (result, _evo) = boolean_with_evolution(&mut topo, BooleanOp::Fuse, body, lip).unwrap();

    assert_eq!(
        near_duplicate_vertex_pairs(&topo, result, 1e-4),
        0,
        "fuse minted vertices closer than the STL quantization step"
    );

    // Kernel-mesh health at the tool's export tier, both by index and under
    // the tool's 1e-4 STL quantization (the oracle that caught the sliver).
    let mesh = tessellate_solid_with_tolerance(&topo, result, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "one-sided mesh edges");
    assert_eq!(non_manifold_edge_count(&mesh), 0, "non-manifold mesh edges");
    let (bnd_q, nm_q) = quantized_mesh_defects(&mesh, 1e-4);
    assert_eq!(nm_q, 0, "quantized non-manifold STL edges");
    assert_eq!(bnd_q, 0, "quantized boundary STL edges");
}

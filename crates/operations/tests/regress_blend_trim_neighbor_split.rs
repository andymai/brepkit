//! Regression: the v2 blend trimmer must propagate boundary-edge splits into
//! neighbor faces that are not themselves trimmed.
//!
//! A fillet's contact curve crosses the trimmed face's boundary mid-edge; the
//! crossed edge is shared with a cap face (here: the box end faces). Before
//! the fix, `split_edge_at` rebuilt only the trimmed face's wire, so the end
//! faces kept referencing the stale unsplit edge: the stale edge and the kept
//! sub-edge were each used by a single face, opening the shell along the
//! shared span (16 free B-Rep edges, 28 boundary mesh edges at export
//! tolerance for this configuration).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid_with_tolerance};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

fn edge_use_counts(topo: &Topology, solid: SolidId) -> HashMap<EdgeId, usize> {
    let mut counts: HashMap<EdgeId, usize> = HashMap::new();
    for fid in solid_faces(topo, solid).unwrap() {
        let face = topo.face(fid).unwrap();
        let mut wires = vec![face.outer_wire()];
        wires.extend(face.inner_wires().iter().copied());
        for wid in wires {
            for oe in topo.wire(wid).unwrap().edges() {
                *counts.entry(oe.edge()).or_insert(0) += 1;
            }
        }
    }
    counts
}

fn assert_wire_connected(topo: &Topology, fid: FaceId) {
    let wire = topo.wire(topo.face(fid).unwrap().outer_wire()).unwrap();
    let oes = wire.edges();
    for i in 0..oes.len() {
        let cur = topo.edge(oes[i].edge()).unwrap();
        let next_oe = oes[(i + 1) % oes.len()];
        let next = topo.edge(next_oe.edge()).unwrap();
        assert_eq!(
            oes[i].oriented_end(cur),
            next_oe.oriented_start(next),
            "wire of face {fid:?} is disconnected at position {i}"
        );
    }
}

fn wire_has_vertex_at(topo: &Topology, fid: FaceId, p: Point3) -> bool {
    let wire = topo.wire(topo.face(fid).unwrap().outer_wire()).unwrap();
    wire.edges().iter().any(|oe| {
        let e = topo.edge(oe.edge()).unwrap();
        [e.start(), e.end()]
            .iter()
            .any(|&vid| (topo.vertex(vid).unwrap().point() - p).length() < 1e-9)
    })
}

#[test]
fn fillet_v2_box_edge_propagates_boundary_splits() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    // Top-front edge: (0,0,10) -> (10,0,10).
    let fillet_edge = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            let e = topo.edge(eid).unwrap();
            let s = topo.vertex(e.start()).unwrap().point();
            let t = topo.vertex(e.end()).unwrap().point();
            (s.z() - 10.0).abs() < 1e-9
                && (t.z() - 10.0).abs() < 1e-9
                && s.y().abs() < 1e-9
                && t.y().abs() < 1e-9
        })
        .expect("top front edge");

    // The four edges sharing exactly one endpoint with the fillet edge are
    // the ones the contact curves cross mid-edge (at distance r = 1 from the
    // corner). Each is shared between a trimmed face and an untouched end
    // face.
    let fe = topo.edge(fillet_edge).unwrap();
    let fe_verts = [fe.start(), fe.end()];
    let split_candidates: Vec<EdgeId> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&eid| {
            let e = topo.edge(eid).unwrap();
            let shared = usize::from(fe_verts.contains(&e.start()))
                + usize::from(fe_verts.contains(&e.end()));
            eid != fillet_edge && shared == 1
        })
        .collect();
    assert_eq!(split_candidates.len(), 4, "box corner adjacency");

    let result = fillet_v2(&mut topo, solid, &[fillet_edge], 1.0).unwrap();
    assert_eq!(result.succeeded, vec![fillet_edge]);
    assert!(result.failed.is_empty());

    let counts = edge_use_counts(&topo, result.solid);

    // The stale unsplit edges must not be referenced by ANY face of the
    // result: every wire that referenced them (including the untouched end
    // faces) must have been rebuilt onto the two sub-edges.
    for eid in &split_candidates {
        assert_eq!(
            counts.get(eid).copied().unwrap_or(0),
            0,
            "stale pre-split edge {eid:?} still referenced by a result face"
        );
    }

    // No edge may be over-shared, and every face wire must remain a
    // head-to-tail connected loop after the in-place neighbor rebuilds.
    assert!(
        counts.values().all(|&c| c <= 2),
        "over-shared edge after split propagation"
    );
    let faces = solid_faces(&topo, result.solid).unwrap();
    for &fid in &faces {
        assert_wire_connected(&topo, fid);
    }

    // The two end faces (planes x=0 and x=10) each gained both split
    // vertices: (x, 1, 10) from the top-face trim and (x, 0, 9) from the
    // front-face trim, growing from 4 to 6 boundary edges.
    for x in [0.0, 10.0] {
        let end_face = faces
            .iter()
            .copied()
            .find(|&fid| {
                let f = topo.face(fid).unwrap();
                matches!(
                    f.surface(),
                    FaceSurface::Plane { normal, .. } if normal.x().abs() > 0.99
                ) && wire_has_vertex_at(&topo, fid, Point3::new(x, 10.0, 0.0))
            })
            .expect("end face");
        let n_edges = topo
            .wire(topo.face(end_face).unwrap().outer_wire())
            .unwrap()
            .edges()
            .len();
        assert_eq!(n_edges, 6, "end face at x={x} should gain 2 edges");
        assert!(
            wire_has_vertex_at(&topo, end_face, Point3::new(x, 1.0, 10.0)),
            "end face at x={x} missing top-trim split vertex"
        );
        assert!(
            wire_has_vertex_at(&topo, end_face, Point3::new(x, 0.0, 9.0)),
            "end face at x={x} missing front-trim split vertex"
        );
    }

    // Export-tolerance mesh (0.01 mm / 5 deg). Pre-fix this configuration
    // produced 28 boundary mesh edges: the shared spans tessellated as
    // unwelded T-junction cracks on both sides. The remaining openings are
    // the characterized separate v2 gaps (untrimmed end-face notches,
    // trimmed-face/blend contact edges built as duplicate edge ids, and the
    // keep-side selection defect), not the propagation defect.
    let mesh =
        tessellate_solid_with_tolerance(&topo, result.solid, 0.01, 5.0_f64.to_radians()).unwrap();
    let bnd = boundary_edge_count(&mesh);
    assert!(
        bnd < 28,
        "T-junction cracks along propagated splits should be gone; bnd = {bnd}"
    );
}

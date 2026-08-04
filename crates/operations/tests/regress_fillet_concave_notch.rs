//! A fillet on a CONCAVE (reflex-notch) edge must fill the corner with a
//! small sliver, not carve material. Two roots closed here: the analytic
//! plane-plane fillet placed its whole cross-section down the inward-normal
//! bisector (the material side; a convex and a concave edge are
//! indistinguishable from the inward normals alone, so a face-extent witness
//! supplies the missing bit), and the trim keep-side keyed on which side of
//! the PLANE the ball centre sits, which flips for concave edges even though
//! the in-plane keep side does not (now resolved inside the trimmer via
//! TrimKeep::AwayFrom, since its Left/Right frame follows wire traversal).

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
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

#[test]
fn fillet_v2_concave_notch_adds_only_the_fillet_sliver() {
    let mut topo = Topology::new();
    // A shallow ridge: the vertex at (5, 0.05) makes the two top laterals
    // meet at ~178.9 degrees after extrusion.
    let profile = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(4.0, 0.0, 0.0),
            Point3::new(5.0, -1.0, 0.0),
            Point3::new(6.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, -3.0, 0.0),
            Point3::new(0.0, -3.0, 0.0),
        ],
        1e-7,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        profile,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 8.0).unwrap();

    // The ridge edge runs along (5, 0.05, z).
    let ridge = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .find(|&eid| {
            let e = topo.edge(eid).unwrap();
            let s = topo.vertex(e.start()).unwrap().point();
            let t = topo.vertex(e.end()).unwrap().point();
            (s.x() - 5.0).abs() < 1e-9 && (t.x() - 5.0).abs() < 1e-9 && (s.z() - t.z()).abs() > 1.0
        })
        .expect("ridge edge");

    let before = brepkit_operations::measure::solid_volume(&topo, solid, 0.05).unwrap();
    let result = fillet_v2(&mut topo, solid, &[ridge], 0.02).unwrap();
    assert_eq!(result.succeeded, vec![ridge]);

    // A 0.2 chamfer on this ridge removes a sliver; a wrong tangent branch
    // cuts a macroscopic wedge or adds material.
    let after = brepkit_operations::measure::solid_volume(&topo, result.solid, 0.05).unwrap();
    // Chamfering a CONCAVE edge fills the notch corner: volume grows by a
    // small sliver. The external tangent branch instead cuts outward or
    // collapses the notch walls.
    let added = after - before;
    assert!(
        added > 0.0 && added < before * 0.02,
        "concave fillet should add a small sliver: before={before:.4} after={after:.4}"
    );

    // No stale or over-shared edges.
    let counts = edge_use_counts(&topo, result.solid);
    assert!(
        counts.values().all(|&c| c <= 2),
        "over-shared edge after near-tangent trim"
    );
}

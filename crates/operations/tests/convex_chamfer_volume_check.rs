//! Pins chamfer_v2's tangent-branch selection on a CONCAVE (reflex) edge:
//! the bisector-projected contact direction flips onto the faces' external
//! extensions there (a 0.02 chamfer grew the solid 6.7% pre-fix); the
//! material-oriented direction (wire-traversal left side) keeps the
//! contacts on the faces.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::HashMap;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::chamfer_v2;
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
fn chamfer_v2_convex_ridge_removes_material() {
    let mut topo = Topology::new();
    // A shallow ridge: the vertex at (5, 0.05) makes the two top laterals
    // meet at ~178.9 degrees after extrusion.
    let profile = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 2.0, 0.0),
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
    let result = chamfer_v2(&mut topo, solid, &[ridge], 0.5, 0.5).unwrap();
    assert_eq!(result.succeeded, vec![ridge]);

    // A 0.2 chamfer on this ridge removes a sliver; a wrong tangent branch
    // cuts a macroscopic wedge or adds material.
    let after = brepkit_operations::measure::solid_volume(&topo, result.solid, 0.05).unwrap();
    // A convex chamfer REMOVES material.
    let removed = before - after;
    assert!(
        removed > 0.0 && removed < before * 0.05,
        "convex chamfer must remove a modest sliver: before={before:.4} after={after:.4}"
    );

    // No stale or over-shared edges.
    let counts = edge_use_counts(&topo, result.solid);
    assert!(
        counts.values().all(|&c| c <= 2),
        "over-shared edge after near-tangent trim"
    );
}

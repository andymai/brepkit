//! A near-tangent ridge fillet must barely change the solid. The 12% loss
//! this pinned was not keep-side selection at all: `dihedral_half_angle`
//! returned half the angle BETWEEN the normals where the geometry needs the
//! material wedge half-angle `(pi - angle)/2`. The two coincide only at a
//! 90-degree dihedral (both 45), so every box-calibrated case passed while a
//! 178.9-degree ridge got contacts r/tan(0.55 deg) = 100*r from the edge
//! instead of r*tan(0.55 deg), trimming away a fifth of each top face.

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
fn fillet_v2_near_tangent_ridge_keeps_correct_sides() {
    let mut topo = Topology::new();
    // A shallow ridge: the vertex at (5, 0.05) makes the two top laterals
    // meet at ~178.9 degrees after extrusion.
    let profile = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.05, 0.0),
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

    // A wrong keep-side discards a whole lateral: the volume collapses or
    // balloons. A correct near-tangent fillet changes the volume by well
    // under 1 percent.
    let after = brepkit_operations::measure::solid_volume(&topo, result.solid, 0.05).unwrap();
    let rel = ((after - before) / before).abs();
    assert!(
        rel < 0.01,
        "near-tangent fillet should barely change volume: before={before:.4} after={after:.4}"
    );

    // No stale or over-shared edges.
    let counts = edge_use_counts(&topo, result.solid);
    assert!(
        counts.values().all(|&c| c <= 2),
        "over-shared edge after near-tangent trim"
    );
}

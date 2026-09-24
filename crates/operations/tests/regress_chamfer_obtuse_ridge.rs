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
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::validate::validate_solid;
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
fn chamfer_v2_concave_notch_adds_only_the_chamfer_sliver() {
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
    let result = chamfer_v2(&mut topo, solid, &[ridge], 0.02, 0.02).unwrap();
    assert_eq!(result.succeeded, vec![ridge]);

    // Chamfering a CONCAVE edge fills the notch corner: the walls meet at a
    // right angle, so the 0.02 chamfer adds a right triangle of legs 0.02
    // along the 8-long ridge. The external tangent branch instead cuts
    // outward or collapses the notch walls, and a chamfer face stored
    // facing into the material reads a volume that moves with the point it
    // is summed about.
    let after = brepkit_operations::measure::solid_volume(&topo, result.solid, 0.05).unwrap();
    let sliver = 0.5 * 0.02 * 0.02 * 8.0;
    assert!(
        (before - 232.0).abs() < 1e-9 && (after - (before + sliver)).abs() < 1e-9,
        "concave chamfer should add {sliver}: before={before} after={after}"
    );
    let mesh = oriented_solid_volume(&topo, result.solid, 0.05).unwrap();
    assert!(
        (mesh - after).abs() < 1e-9,
        "mesh volume {mesh}, exact {after}"
    );
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    // No stale or over-shared edges.
    let counts = edge_use_counts(&topo, result.solid);
    assert!(
        counts.values().all(|&c| c <= 2),
        "over-shared edge after near-tangent trim"
    );
}

fn edge_length(topo: &Topology, edge: EdgeId) -> f64 {
    let e = topo.edge(edge).unwrap();
    (topo.vertex(e.end()).unwrap().point() - topo.vertex(e.start()).unwrap().point()).length()
}

fn assert_closed_at(topo: &Topology, solid: SolidId, expected: f64, what: &str) {
    let counts = edge_use_counts(topo, solid);
    assert!(
        counts.values().all(|&c| c == 2),
        "{what}: open or over-shared edges"
    );
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let volume = brepkit_operations::measure::solid_volume(topo, solid, 0.01).unwrap();
    assert!(
        (volume - expected).abs() < 1e-9,
        "{what}: volume {volume}, expected {expected}"
    );
}

/// Every edge of a box, as built and mirrored, symmetric and not: the
/// chamfer's end faces take its cross edges, so the solid closes and loses
/// exactly the right triangular prism, `d1 d2 / 2` times the edge length.
#[test]
fn chamfer_v2_closes_on_every_box_edge() {
    for mirrored in [false, true] {
        for (d1, d2) in [(0.5, 0.5), (0.3, 0.6)] {
            for index in 0..12 {
                let mut topo = Topology::new();
                let b = brepkit_operations::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
                if mirrored {
                    brepkit_operations::transform::transform_solid(
                        &mut topo,
                        b,
                        &brepkit_math::mat::Mat4::scale(-1.0, 1.0, 1.0),
                    )
                    .unwrap();
                }
                let edge = solid_edges(&topo, b).unwrap()[index];
                let length = edge_length(&topo, edge);
                let result = chamfer_v2(&mut topo, b, &[edge], d1, d2).unwrap();
                assert_eq!(result.succeeded, vec![edge]);
                assert_closed_at(
                    &topo,
                    result.solid,
                    24.0 - 0.5 * d1 * d2 * length,
                    &format!("edge {index} ({d1}, {d2}) mirrored={mirrored}"),
                );
            }
        }
    }
}

/// Two parallel edges at once: each chamfer closes its own ends.
#[test]
fn chamfer_v2_closes_two_parallel_edges() {
    let mut topo = Topology::new();
    let b = brepkit_operations::primitives::make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
    let edges = solid_edges(&topo, b).unwrap();
    let first = edges[0];
    let (s0, t0) = {
        let e = topo.edge(first).unwrap();
        (
            topo.vertex(e.start()).unwrap().point(),
            topo.vertex(e.end()).unwrap().point(),
        )
    };
    let parallel = edges
        .iter()
        .copied()
        .find(|&e| {
            let d = {
                let e = topo.edge(e).unwrap();
                (
                    topo.vertex(e.start()).unwrap().point(),
                    topo.vertex(e.end()).unwrap().point(),
                )
            };
            let shares = [d.0, d.1]
                .iter()
                .any(|p| (*p - s0).length() < 1e-9 || (*p - t0).length() < 1e-9);
            (d.1 - d.0)
                .normalize()
                .unwrap()
                .cross((t0 - s0).normalize().unwrap())
                .length()
                < 1e-9
                && !shares
        })
        .unwrap();
    let length = edge_length(&topo, first);
    let result = chamfer_v2(&mut topo, b, &[first, parallel], 0.4, 0.4).unwrap();
    assert_eq!(result.succeeded.len(), 2);
    assert_closed_at(
        &topo,
        result.solid,
        24.0 - 2.0 * 0.08 * length,
        "two parallel edges",
    );
}

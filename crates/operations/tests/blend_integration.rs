//! End-to-end integration tests for the v2 fillet and chamfer engine.
//!
//! Each test creates fresh geometry from primitives and exercises the
//! walking-based blend pipeline (`blend_ops::fillet_v2`, `chamfer_v2`,
//! `chamfer_distance_angle`).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_operations::blend_ops::{chamfer_distance_angle, chamfer_v2, fillet_v2};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;

const BOX_VOLUME: f64 = 1000.0; // 10 x 10 x 10

/// Create a 10x10x10 box and fillet a single edge.
#[test]
fn fillet_box_single_edge() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    assert!(!edges.is_empty(), "box must have edges");

    let result = fillet_v2(&mut topo, solid, &edges[..1], 1.0).unwrap();

    // Fillet adds faces (original 6 + at least 1 blend surface).
    let faces = solid_faces(&topo, result.solid).unwrap();
    assert!(
        faces.len() > 6,
        "filleted box should have more than 6 faces"
    );

    // At least the one edge should have succeeded.
    assert!(
        !result.succeeded.is_empty(),
        "at least one edge should succeed"
    );

    // Volume changes when edges are filleted.
    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "filleted volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Fillet 4 edges of a box (e.g. the first 4 found).
#[test]
fn fillet_box_multiple_edges() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let n = edges.len().min(4);
    let target = &edges[..n];

    let result = fillet_v2(&mut topo, solid, target, 0.5).unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "at least some edges should succeed"
    );

    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "filleted volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Symmetric chamfer on a single edge.
#[test]
fn chamfer_box_symmetric() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let result = chamfer_v2(&mut topo, solid, &edges[..1], 1.0, 1.0).unwrap();

    let faces = solid_faces(&topo, result.solid).unwrap();
    assert!(
        faces.len() > 6,
        "chamfered box should have more than 6 faces"
    );

    let vol = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (vol - BOX_VOLUME).abs() > 0.01,
        "chamfered volume {vol} should differ from original {BOX_VOLUME}"
    );
}

/// Distance-angle chamfer on a single edge.
#[test]
fn chamfer_box_distance_angle() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let edges = solid_edges(&topo, solid).unwrap();
    let result = chamfer_distance_angle(
        &mut topo,
        solid,
        &edges[..1],
        1.0,
        std::f64::consts::FRAC_PI_4,
    )
    .unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "distance-angle chamfer should succeed on at least one edge"
    );
}

/// Zero radius should be rejected.
#[test]
fn fillet_zero_radius_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edges = solid_edges(&topo, solid).unwrap();

    let err = fillet_v2(&mut topo, solid, &edges[..1], 0.0);
    assert!(err.is_err(), "zero radius should return an error");
}

/// Zero distance should be rejected.
#[test]
fn chamfer_zero_distance_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let edges = solid_edges(&topo, solid).unwrap();

    let err = chamfer_v2(&mut topo, solid, &edges[..1], 0.0, 1.0);
    assert!(err.is_err(), "zero distance should return an error");
}

/// Empty edge list should be rejected.
#[test]
fn fillet_empty_edges_error() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

    let err = fillet_v2(&mut topo, solid, &[], 1.0);
    assert!(err.is_err(), "empty edges should return an error");
}

/// Plane-cylinder fillet on a primitive cylinder. The bottom-cap circle
/// edge sits where the plane (cap) meets the cylinder lateral; the analytic
/// dispatcher should produce an exact toroidal fillet face for the convex
/// "post on plate" geometry.
#[test]
fn fillet_cylinder_base_circle_produces_torus() {
    let mut topo = Topology::new();
    // Cylinder of radius 2, height 4 — convex base circle is the spine.
    let solid = make_cylinder(&mut topo, 2.0, 4.0).unwrap();

    // The two `EdgeCurve::Circle` edges on a primitive cylinder are the top
    // and bottom rims; pick whichever the explorer surfaces first.
    let circle_edges: Vec<_> = solid_edges(&topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&eid| matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::Circle(_)))
        .collect();
    assert!(
        !circle_edges.is_empty(),
        "cylinder must have at least one circular rim edge"
    );

    let result = fillet_v2(&mut topo, solid, &circle_edges[..1], 0.3).unwrap();

    assert!(
        !result.succeeded.is_empty(),
        "cylinder rim fillet must produce at least one stripe; failed = {:?}",
        result.failed
    );

    // The new blend face should be exactly a Torus when the analytic path
    // fired (vs a NURBS approximation when the walker fallback was used).
    let new_faces: Vec<_> = solid_faces(&topo, result.solid).unwrap();
    let torus = new_faces.iter().find_map(|&fid| {
        if let FaceSurface::Torus(t) = topo.face(fid).unwrap().surface() {
            Some(t.clone())
        } else {
            None
        }
    });
    let torus = torus.expect("analytic fast path should produce a Torus face");

    // Torus geometry: minor radius == fillet radius, major radius ==
    // r_cylinder + r_fillet (convex case), axis parallel to cylinder axis.
    assert!(
        (torus.minor_radius() - 0.3).abs() < 1e-9,
        "torus minor radius should equal fillet radius 0.3, got {}",
        torus.minor_radius()
    );
    assert!(
        (torus.major_radius() - 2.3).abs() < 1e-9,
        "torus major radius should equal cylinder radius + fillet radius (2.3), got {}",
        torus.major_radius()
    );
}

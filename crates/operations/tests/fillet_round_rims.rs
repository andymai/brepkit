//! Fillets of closed circular rims where a flat cap meets a coaxial cylinder:
//! a rod's top, whose cap is a disc, and a tube's outer mouth, whose cap is a
//! ring. Each removes a ring whose cross-section is the square of side `r` at
//! the rim less a quarter disc, so its volume is that area times the path of
//! its centroid. The corner blend is a torus of major radius `R - r`: a ring
//! torus below half the cylinder's radius and a spindle torus above it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 5.0;
const HEIGHT: f64 = 6.0;

/// The closed circle edge of radius `rim` at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, rim: f64, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c) if (c.radius() - rim).abs() < 1e-9)
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .unwrap()
}

/// The ring a fillet of radius `r` removes from a convex rim of radius `rim`.
fn rounded_off(rim: f64, r: f64) -> f64 {
    let area = (1.0 - PI / 4.0) * r * r;
    let centroid_from_corner = r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI));
    2.0 * PI * (rim - centroid_from_corner) * area
}

/// Fillets the outer top rim of `solid` and checks it is valid, watertight,
/// and short by exactly the ring the fillet removes.
fn assert_rim_rounds_off(topo: &mut Topology, solid: SolidId, r: f64) {
    let rim = rim_at(topo, solid, RADIUS, HEIGHT);
    let before = solid_volume(topo, solid, 0.001).unwrap();
    let result = fillet_v2(topo, solid, &[rim], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let removed = before - solid_volume(topo, result.solid, 0.001).unwrap();
    let truth = rounded_off(RADIUS, r);
    assert!(
        (removed - truth).abs() < 1e-6 * before,
        "r = {r}: removed {removed}, truth {truth}"
    );
}

#[test]
fn a_rods_top_rim_rounds_off_on_ring_and_spindle_tori() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
        assert_rim_rounds_off(&mut topo, rod, r);
    }
}

/// The tube's top is a ring, not a disc, and its material still lies toward
/// the axis from the outer rim.
#[test]
fn a_tubes_outer_mouth_rounds_off_on_ring_and_spindle_tori() {
    for r in [1.0, 3.0] {
        let mut topo = Topology::new();
        let outer = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
        let bore = make_cylinder(&mut topo, 1.0, HEIGHT + 2.0).unwrap();
        transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
        let tube = boolean(&mut topo, BooleanOp::Cut, outer, bore).unwrap();
        assert_rim_rounds_off(&mut topo, tube, r);
    }
}

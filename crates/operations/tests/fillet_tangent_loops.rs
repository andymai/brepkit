//! Fillets along tangent loops of lines and arcs: the floor of a pocket whose
//! corners are already rounded, and the top of a block whose vertical edges
//! are. Along each straight run the fillet fills or removes a prism whose
//! section is the square of side `r` less a quarter disc; round each corner
//! arc it turns that section a quarter of the way about the arc's axis.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

fn section(r: f64) -> f64 {
    (1.0 - PI / 4.0) * r * r
}

/// The section's centroid lies this far from its corner along each face.
fn centroid_from_corner(r: f64) -> f64 {
    r * 3.0f64.mul_add(-PI, 10.0) / (3.0 * (4.0 - PI))
}

fn ends(topo: &Topology, edge: EdgeId) -> [brepkit_math::vec::Point3; 2] {
    let edge = topo.edge(edge).unwrap();
    [
        topo.vertex(edge.start()).unwrap().point(),
        topo.vertex(edge.end()).unwrap().point(),
    ]
}

/// The vertical edges standing at one of `corners`.
fn verticals_at(topo: &Topology, solid: SolidId, corners: &[(f64, f64)]) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let [a, b] = ends(topo, e);
            corners.iter().any(|&(x, y)| {
                [a, b]
                    .iter()
                    .all(|p| (p.x() - x).abs() < 1e-9 && (p.y() - y).abs() < 1e-9)
            })
        })
        .collect()
}

/// The edges whose ends both satisfy `keep`.
fn edges_where(
    topo: &Topology,
    solid: SolidId,
    keep: impl Fn(brepkit_math::vec::Point3) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| ends(topo, e).into_iter().all(&keep))
        .collect()
}

/// Fillets `edges` and checks the result is valid, watertight, and changed in
/// volume by exactly `change`.
fn fillet_changes_by(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    r: f64,
    change: f64,
) -> SolidId {
    let before = solid_volume(topo, solid, 0.001).unwrap();
    let result = fillet_v2(topo, solid, edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let changed = solid_volume(topo, result.solid, 0.001).unwrap() - before;
    assert!(
        (changed - change).abs() < 1e-6 * before,
        "r = {r}: changed by {changed}, truth {change}"
    );
    result.solid
}

/// An 80 x 60 x 20 block less a 50 x 30 x 8 pocket, its pocket's corners
/// filleted r 5.
fn pocket_with_round_corners(topo: &mut Topology) -> SolidId {
    let block = make_box(topo, 80.0, 60.0, 20.0).unwrap();
    let pocket = make_box(topo, 50.0, 30.0, 8.0).unwrap();
    transform_solid(topo, pocket, &Mat4::translation(15.0, 15.0, 12.0)).unwrap();
    let body = boolean(topo, BooleanOp::Cut, block, pocket).unwrap();
    let corners = [(15.0, 15.0), (65.0, 15.0), (65.0, 45.0), (15.0, 45.0)];
    let verticals = verticals_at(topo, body, &corners);
    assert_eq!(verticals.len(), 4);
    let filled = 4.0 * section(5.0) * 8.0;
    fillet_changes_by(topo, body, &verticals, 5.0, filled)
}

fn on_pocket_floor(p: brepkit_math::vec::Point3) -> bool {
    (p.z() - 12.0).abs() < 1e-9 && p.x() > 14.0 && p.x() < 66.0 && p.y() > 14.0 && p.y() < 46.0
}

#[test]
fn a_pockets_floor_loop_fills_round_its_rounded_corners() {
    let mut topo = Topology::new();
    let body = pocket_with_round_corners(&mut topo);
    let floor = edges_where(&topo, body, on_pocket_floor);
    assert_eq!(floor.len(), 8);
    let r = 2.0;
    // Straight runs of 40, 20, 40 and 20, and four quarter turns about the
    // corner cylinders, the section's centroid inside their radius 5.
    let filled = section(r) * 120.0 + 2.0 * PI * (5.0 - centroid_from_corner(r)) * section(r);
    fillet_changes_by(&mut topo, body, &floor, r, filled);
}

/// One corner arc of the floor alone: its fillet ends on the straight runs it
/// is tangent to, and turns its section a quarter of the way round.
#[test]
fn a_pockets_floor_corner_arc_fills_its_quarter_turn() {
    let mut topo = Topology::new();
    let body = pocket_with_round_corners(&mut topo);
    let arcs: Vec<EdgeId> = edges_where(&topo, body, on_pocket_floor)
        .into_iter()
        .filter(|&e| {
            let [a, b] = ends(&topo, e);
            (a.x() - b.x()).abs() > 1e-9 && (a.y() - b.y()).abs() > 1e-9
        })
        .collect();
    assert_eq!(arcs.len(), 4);
    let r = 2.0;
    let filled = PI / 2.0 * (5.0 - centroid_from_corner(r)) * section(r);
    for &arc in &arcs {
        let mut t = topo.clone();
        fillet_changes_by(&mut t, body, &[arc], r, filled);
    }
}

/// A 50 x 40 x 30 block with its vertical edges filleted R 8: the loop round
/// its top, four lines and four arcs, rounds off at R 4.
#[test]
fn a_blocks_top_loop_rounds_off_beside_its_rounded_corners() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 50.0, 40.0, 30.0).unwrap();
    let corners = [(0.0, 0.0), (50.0, 0.0), (50.0, 40.0), (0.0, 40.0)];
    let verticals = verticals_at(&topo, block, &corners);
    assert_eq!(verticals.len(), 4);
    let removed = 4.0 * section(8.0) * 30.0;
    let body = fillet_changes_by(&mut topo, block, &verticals, 8.0, -removed);
    let top = edges_where(&topo, body, |p| (p.z() - 30.0).abs() < 1e-9);
    assert_eq!(top.len(), 8);
    let r = 4.0;
    let removed = section(r) * 116.0 + 2.0 * PI * (8.0 - centroid_from_corner(r)) * section(r);
    fillet_changes_by(&mut topo, body, &top, r, -removed);
}

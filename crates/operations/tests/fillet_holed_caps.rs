//! Fillets whose ends meet a planar cap that has holes. A shelled box's rim
//! is such a cap: its outer corners end the fillets of the outer vertical
//! edges, and the corners of its hole end those of the cavity's inner
//! vertical edges. The top of a block with a blind pocket is another, its hole
//! the pocket's mouth. Each straight fillet removes (convex) or fills
//! (concave) a prism whose section is the square of side `r` less a quarter
//! disc.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::shell_op::shell;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

/// The vertical line edges whose foot lies at `(x, y)` for one of `feet`.
fn verticals_at(topo: &Topology, solid: SolidId, feet: &[(f64, f64)]) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let a = topo.vertex(edge.start()).unwrap().point();
            let b = topo.vertex(edge.end()).unwrap().point();
            (a.z() - b.z()).abs() > 1e-6
                && feet.iter().any(|&(x, y)| {
                    [a, b]
                        .iter()
                        .all(|p| (p.x() - x).abs() < 1e-9 && (p.y() - y).abs() < 1e-9)
                })
        })
        .collect()
}

/// A 60 x 40 x 30 box shelled 3 with its top open.
fn shelled_box(topo: &mut Topology) -> SolidId {
    let block = make_box(topo, 60.0, 40.0, 30.0).unwrap();
    let top = solid_faces(topo, block)
        .unwrap()
        .into_iter()
        .find(|&f| {
            let wire = topo.face(f).unwrap().outer_wire();
            topo.wire(wire).unwrap().edges().iter().all(|oe| {
                let start = topo.edge(oe.edge()).unwrap().start();
                (topo.vertex(start).unwrap().point().z() - 30.0).abs() < 1e-9
            })
        })
        .unwrap();
    shell(topo, block, 3.0, &[top]).unwrap()
}

/// Fillets `edges` and checks the result is valid, watertight, and changed in
/// volume by exactly `change`.
fn assert_fillet_changes_by(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    r: f64,
    change: f64,
) {
    let before = solid_volume(topo, solid, 0.001).unwrap();
    let result = fillet_v2(topo, solid, edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.1).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let changed = solid_volume(topo, result.solid, 0.001).unwrap() - before;
    assert!(
        (changed - change).abs() < 1e-6 * before,
        "r = {r}: changed by {changed}, truth {change}"
    );
}

fn corner_section(r: f64) -> f64 {
    (1.0 - PI / 4.0) * r * r
}

#[test]
fn a_shelled_boxs_outer_corners_round_off_at_its_rim() {
    let mut topo = Topology::new();
    let cup = shelled_box(&mut topo);
    let edges = verticals_at(
        &topo,
        cup,
        &[(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)],
    );
    assert_eq!(edges.len(), 4);
    let removed = 4.0 * corner_section(6.0) * 30.0;
    assert_fillet_changes_by(&mut topo, cup, &edges, 6.0, -removed);
}

#[test]
fn a_shelled_boxs_inner_corners_fill_up_to_its_rim() {
    let corners = [(3.0, 3.0), (57.0, 3.0), (57.0, 37.0), (3.0, 37.0)];
    let mut topo = Topology::new();
    let cup = shelled_box(&mut topo);
    let edges = verticals_at(&topo, cup, &corners);
    assert_eq!(edges.len(), 4);
    let filled = 4.0 * corner_section(3.0) * 27.0;
    assert_fillet_changes_by(&mut topo, cup, &edges, 3.0, filled);

    let mut topo = Topology::new();
    let cup = shelled_box(&mut topo);
    let edges = verticals_at(&topo, cup, &corners[..1]);
    assert_fillet_changes_by(&mut topo, cup, &edges, 3.0, corner_section(3.0) * 27.0);
}

#[test]
fn a_blind_pockets_corner_fills_up_to_its_mouth() {
    for r in [1.0, 5.0] {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 80.0, 60.0, 20.0).unwrap();
        let pocket = make_box(&mut topo, 50.0, 30.0, 8.0).unwrap();
        transform_solid(&mut topo, pocket, &Mat4::translation(15.0, 15.0, 12.0)).unwrap();
        let body = boolean(&mut topo, BooleanOp::Cut, block, pocket).unwrap();
        let edges = verticals_at(&topo, body, &[(15.0, 15.0)]);
        assert_eq!(edges.len(), 1);
        assert_fillet_changes_by(&mut topo, body, &edges, r, corner_section(r) * 8.0);
    }
}

/// A fillet wider than the cup's wall would carry its corner across the
/// cavity's mouth: the rim keeps no notch that crosses its hole, so the
/// fillet is refused or the result is a valid, watertight solid.
#[test]
fn a_shelled_boxs_corner_fillet_wider_than_its_wall_never_crosses_the_mouth() {
    let mut topo = Topology::new();
    let cup = shelled_box(&mut topo);
    let edges = verticals_at(
        &topo,
        cup,
        &[(0.0, 0.0), (60.0, 0.0), (60.0, 40.0), (0.0, 40.0)],
    );
    let Ok(result) = fillet_v2(&mut topo, cup, &edges, 12.0) else {
        return;
    };
    if !result.failed.is_empty() || result.solid == cup {
        return;
    }
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.1).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
}

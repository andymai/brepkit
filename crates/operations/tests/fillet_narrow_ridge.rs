//! Rounds along both long edges of a narrow face: a box's top `W` wide.
//!
//! A round of radius `r` along a right-angled edge sets its contact back by
//! `r` on each face. With both edges of the top rounded, the two contacts on
//! it meet at `2r = W`; past that, each lies nearer the other round's edge
//! than its own, and the top between the swapped contacts would be a loop
//! that crosses itself. Such rounds are refused; below `W / 2` the two
//! rounds leave a strip of the top between them.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const L: f64 = 10.0;
const W: f64 = 2.0;
const H: f64 = 10.0;

/// The box, `[0, L] x [0, W] x [0, H]`, and its two top edges along `x`.
fn ridge(topo: &mut Topology) -> (SolidId, Vec<EdgeId>) {
    let solid = make_box(topo, L, W, H).unwrap();
    let edges = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            (a.z() - H).abs() < 1e-9 && (b.z() - H).abs() < 1e-9 && (a.y() - b.y()).abs() < 1e-9
        })
        .collect();
    (solid, edges)
}

#[test]
fn rounds_that_leave_a_strip_between_them() {
    let r = 0.9;
    let mut topo = Topology::new();
    let (solid, edges) = ridge(&mut topo);
    assert_eq!(edges.len(), 2);
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh));
    let removed = L * W * H - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let strips = 2.0 * L * r * r * (1.0 - PI / 4.0);
    assert!(
        (removed - strips).abs() <= 1e-6 * L * W * H,
        "removed {removed}, the two strips {strips}"
    );
}

#[test]
fn rounds_whose_contacts_pass_each_other_are_refused() {
    for r in [1.02, 1.5] {
        let mut topo = Topology::new();
        let (solid, edges) = ridge(&mut topo);
        let refused = match fillet_v2(&mut topo, solid, &edges, r) {
            Err(_) => true,
            Ok(result) => !result.failed.is_empty(),
        };
        assert!(refused, "r = {r} on a top {W} wide must be refused");
    }
}

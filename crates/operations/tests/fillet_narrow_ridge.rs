//! Rounds along both long edges of a narrow face: a box's top `W` wide.
//!
//! A round of radius `r` along a right-angled edge sets its contact back by
//! `r` on each face. With both edges of the top rounded, the two contacts on
//! it meet at `2r = W`; below that the two rounds leave a strip of the top
//! between them. Past it, each contact lies nearer the other round's edge
//! than its own: the first round is kept, and the second rests on its side
//! and pivots on the first's contact on the top, so the top goes and the
//! rounds meet along that line.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const L: f64 = 10.0;
const W: f64 = 2.0;
const H: f64 = 10.0;

/// The box, `[0, L] x [0, W] x [0, H]`, and its two top edges along `x`,
/// the box then moved by `pose`.
fn ridge(topo: &mut Topology, pose: Option<Mat4>) -> (SolidId, Vec<EdgeId>) {
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
    if let Some(pose) = pose {
        transform_solid(topo, solid, &pose).unwrap();
    }
    (solid, edges)
}

/// The box turned about two axes and moved off the origin.
fn turned() -> Mat4 {
    Mat4::translation(37.0, 483.0, -11.0) * Mat4::rotation_z(0.4229) * Mat4::rotation_x(1.0)
}

fn assert_strip_left(pose: Option<Mat4>) {
    let r = 0.9;
    let mut topo = Topology::new();
    let (solid, edges) = ridge(&mut topo, pose);
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

/// The volume the two rounds take off where they cross: the first's corner
/// strip, and under the second, pivoting with its centre `h` below the
/// top, from the first's contact (`y = r`) to its own side, by Simpson's
/// rule in its angle about its axis.
fn crossing_removed(r: f64) -> f64 {
    const N: usize = 2000;
    let h = (r * r - (W - 2.0 * r).powi(2)).sqrt();
    let reach = ((2.0 * r - W) / r).acos();
    let f = |t: f64| (h - r * t.sin()) * r * t.sin();
    let step = reach / N as f64;
    let mut sum = f(0.0) + f(reach);
    for i in 1..N {
        sum += f(i as f64 * step) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    L * (r * r * (1.0 - PI / 4.0) + sum * step / 3.0)
}

fn assert_pivoted(pose: Option<Mat4>, r: f64) {
    let mut topo = Topology::new();
    let (solid, edges) = ridge(&mut topo, pose);
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh open");
    let removed = L * W * H - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let expected = crossing_removed(r);
    assert!(
        (removed - expected).abs() <= 1e-6 * L * W * H,
        "r = {r}: removed {removed}, the pivoting round takes off {expected}"
    );
}

#[test]
fn rounds_that_leave_a_strip_between_them() {
    assert_strip_left(None);
}

#[test]
fn rounds_whose_contacts_pass_each_other_pivot() {
    for r in [1.02, 1.5] {
        assert_pivoted(None, r);
    }
}

#[test]
fn a_turned_ridge_keeps_its_strip_or_pivots() {
    assert_strip_left(Some(turned()));
    for r in [1.02, 1.5] {
        assert_pivoted(Some(turned()), r);
    }
}

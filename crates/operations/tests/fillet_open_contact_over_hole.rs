//! An open round whose contact on a drilled face passes a hole.
//!
//! A box drilled down through its top near one top edge, that edge then
//! rounded. A hole the round's contact on the top passes on the edge's
//! side lies under the band: no trim builds the top round it (it was built
//! with the hole's loop left in the trimmed top, outside its outline, then
//! refused), so the round is made as on the undrilled box and the hole cut
//! back through it. A hole past the contact keeps its place in the top.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const SIDE: f64 = 20.0;
const HEIGHT: f64 = 10.0;
/// The hole's axis, `HOLE_X` in from the rounded edge.
const HOLE_X: f64 = 4.0;
const HOLE_R: f64 = 1.0;

/// The drilled box, and its top edge along `y` at `x = 0`.
fn drilled_box(topo: &mut Topology) -> (SolidId, EdgeId) {
    let block = make_box(topo, SIDE, SIDE, HEIGHT).unwrap();
    let drill = make_cylinder(topo, HOLE_R, HEIGHT + 2.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(HOLE_X, SIDE / 2.0, -1.0)).unwrap();
    let solid = boolean(topo, BooleanOp::Cut, block, drill).unwrap();
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            matches!(edge.curve(), EdgeCurve::Line)
                && [a, b]
                    .iter()
                    .all(|p| p.x().abs() < 1e-9 && (p.z() - HEIGHT).abs() < 1e-9)
        })
        .expect("the top edge at x = 0");
    (solid, edge)
}

/// The round's contact on the top stands `r` in from the edge: a hole
/// beyond it is kept, and the round takes off its quarter-round strip.
#[test]
fn a_hole_past_the_contact_is_kept() {
    for r in [1.0, 2.5] {
        let mut topo = Topology::new();
        let (solid, edge) = drilled_box(&mut topo);
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh open");
        let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        let expected = (1.0 - std::f64::consts::FRAC_PI_4) * r * r * SIDE;
        assert!(
            (lost - expected).abs() <= 1e-4 * expected,
            "r = {r}: removed {lost}, the round takes off {expected}"
        );
    }
}

/// The part of the round's strip (depth `r - sqrt(r^2 - (r - x)^2)` below
/// the top at `x` from the edge) inside the hole.
fn strip_in_hole(r: f64) -> f64 {
    let (lo, hi) = (HOLE_X - HOLE_R, (HOLE_X + HOLE_R).min(r));
    let n = 20_000;
    let h = (hi - lo) / f64::from(n);
    let f = |x: f64| {
        let chord = 2.0
            * (HOLE_R * HOLE_R - (x - HOLE_X) * (x - HOLE_X))
                .max(0.0)
                .sqrt();
        chord * (r - (r * r - (r - x) * (r - x)).max(0.0).sqrt())
    };
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        sum += f(lo + h * f64::from(i)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

/// A contact past the hole on its far side: the round is cut back by the
/// hole, and takes off its strip less the part of it in the hole.
#[test]
fn a_hole_under_the_band_is_cut_back_through_it() {
    for r in [HOLE_X + HOLE_R + 0.5, 6.0] {
        let mut topo = Topology::new();
        let (solid, edge) = drilled_box(&mut topo);
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
        assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r = {r}: mesh open");
        let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        let expected = (1.0 - std::f64::consts::FRAC_PI_4) * r * r * SIDE - strip_in_hole(r);
        assert!(
            (lost - expected).abs() <= 1e-4 * expected,
            "r = {r}: removed {lost}, the round takes off {expected}"
        );
    }
}

//! A round running over a through-bore's mouth: a box drilled through near
//! one top edge, that edge rounded wider than the bore stands from it. The
//! round's contact on the top crosses the bore, which the trimmer refuses.
//! The round is made as on the box with the bore filled, and the bore is cut
//! back through it. The round takes off its strip, (1 - pi / 4) r^2 per
//! unit length, less the part of the strip already in the bore.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 40.0;
const WIDTH: f64 = 20.0;
const HEIGHT: f64 = 15.0;

/// A box drilled through its height, the bore's axis at (`cx`, `cy`), and
/// its top edge along x at y = 0.
fn drilled_box(topo: &mut Topology, cx: f64, cy: f64, rho: f64) -> (SolidId, EdgeId) {
    let block = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let drill = make_cylinder(topo, rho, HEIGHT + 2.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(cx, cy, -1.0)).unwrap();
    let solid = boolean(topo, BooleanOp::Cut, block, drill).unwrap();
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - HEIGHT).abs() < 1e-9
            })
        })
        .unwrap();
    (solid, edge)
}

/// The part of the round's strip (depth `r - sqrt(r^2 - (r - y)^2)` below
/// the top at `y` from the side) inside the bore.
fn strip_in_bore(r: f64, cy: f64, rho: f64) -> f64 {
    let (lo, hi) = ((cy - rho).max(0.0), (cy + rho).min(r));
    if hi <= lo {
        return 0.0;
    }
    let n = 20_000;
    let h = (hi - lo) / f64::from(n);
    let f = |y: f64| {
        let chord = 2.0 * (rho * rho - (y - cy) * (y - cy)).max(0.0).sqrt();
        let depth = r - (r * r - (r - y) * (r - y)).max(0.0).sqrt();
        chord * depth
    };
    let mut sum = f(lo) + f(hi);
    for i in 1..n {
        let y = lo + h * f64::from(i);
        sum += f(y) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn check(cy: f64, rho: f64, r: f64, pose: &Mat4) {
    let label = format!("bore at {cy} radius {rho}, r {r}");
    let mut topo = Topology::new();
    let (solid, edge) = drilled_box(&mut topo, 0.5 * LENGTH, cy, rho);
    transform_solid(&mut topo, solid, pose).unwrap();
    let before = solid_volume(&topo, solid, 0.01).unwrap();
    let faces = solid_faces(&topo, solid).unwrap().len();
    let result = fillet_v2(&mut topo, solid, &[edge], r)
        .and_then(all_blended)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{label}: mesh not watertight");
    assert_eq!(
        solid_faces(&topo, result).unwrap().len(),
        faces + 1,
        "{label}: faces"
    );
    let removed = before - solid_volume(&topo, result, 0.01).unwrap();
    let want = (1.0 - std::f64::consts::FRAC_PI_4) * r * r * LENGTH - strip_in_bore(r, cy, rho);
    assert!(
        (removed - want).abs() <= 1e-4 * want,
        "{label}: removed {removed}, want {want}"
    );
}

#[test]
fn a_round_over_a_bore_cuts_back_through_it() {
    // The bore wholly inside the round's reach, and straddling its contact.
    for (cy, rho, r) in [(5.0, 2.0, 8.0), (6.0, 2.0, 7.0), (4.0, 1.5, 5.0)] {
        check(cy, rho, r, &Mat4::identity());
    }
}

#[test]
fn a_turned_bore_rounds_alike() {
    let pose =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(6.0, 2.0, 7.0, &pose);
}

//! A rod whose top rim is rounded, bored through along its axis wider than
//! the round leaves its top: the bore cuts the round's torus band round the
//! axis along one circle. The band is split there into two bands, as a
//! cylinder wall is by a full circle (it split into nothing, and the cut
//! fell back to a mesh). What is left is the solid of revolution between
//! the bore and the rod.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::explorer::{solid_edges, solid_faces};

const RADIUS: f64 = 3.0;
const HEIGHT: f64 = 5.0;

/// The rounded rod's volume outside radius `bore`: its height at `s` from
/// the axis is `HEIGHT`, less the round's drop past `RADIUS - r`.
fn outside(bore: f64, r: f64) -> f64 {
    let height = |s: f64| {
        let t = s - (RADIUS - r);
        if t <= 0.0 {
            HEIGHT
        } else {
            HEIGHT - r + (r * r - t * t).max(0.0).sqrt()
        }
    };
    let n = 200_000;
    let h = (RADIUS - bore) / f64::from(n);
    let f = |s: f64| 2.0 * std::f64::consts::PI * s * height(s);
    let mut sum = f(bore) + f(RADIUS);
    for i in 1..n {
        sum += f(bore + h * f64::from(i)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * h / 3.0
}

fn check(bore: f64, r: f64, pose: &Mat4) {
    let label = format!("bore {bore}, r {r}");
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, RADIUS, HEIGHT).unwrap();
    let rim = solid_edges(&topo, rod)
        .unwrap()
        .into_iter()
        .find(|&e| {
            matches!(topo.edge(e).unwrap().curve(),
                EdgeCurve::Circle(c) if (c.center().z() - HEIGHT).abs() < 1e-9)
        })
        .unwrap();
    let round = fillet_v2(&mut topo, rod, &[rim], r)
        .and_then(all_blended)
        .unwrap();
    let drill = make_cylinder(&mut topo, bore, HEIGHT + 2.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    transform_solid(&mut topo, round, pose).unwrap();
    transform_solid(&mut topo, drill, pose).unwrap();
    let opts = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let cut = boolean_with_options(&mut topo, BooleanOp::Cut, round, drill, opts)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, cut, 0.01).unwrap()),
        "{label}: mesh not watertight"
    );
    // The bottom ring, the rod's wall, what is left of the band, the bore.
    assert_eq!(solid_faces(&topo, cut).unwrap().len(), 4, "{label}: faces");
    let volume = solid_volume(&topo, cut, 0.001).unwrap();
    let want = outside(bore, r);
    assert!(
        (volume - want).abs() <= 1e-5 * want,
        "{label}: volume {volume}, want {want}"
    );
}

#[test]
fn a_coaxial_bore_through_a_rounded_rim_splits_its_band() {
    for (bore, r) in [(2.2, 1.0), (2.5, 1.0), (2.8, 1.0), (2.9, 2.0)] {
        check(bore, r, &Mat4::identity());
    }
}

#[test]
fn a_turned_coaxial_bore_splits_alike() {
    let pose =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(2.5, 1.0, &pose);
}

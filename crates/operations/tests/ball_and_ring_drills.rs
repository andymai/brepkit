//! A cylindrical drill into a ball off its axis and through a ring's tube
//! parallel to its axis, on the surface's seam meridian and off it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// The integral over a disc of radius `r` about `(cx, cy)` of `height(x, y)`,
/// by a midpoint rule in polar coordinates.
fn over_disc(cx: f64, cy: f64, r: f64, height: impl Fn(f64, f64) -> f64) -> f64 {
    const N: u32 = 600;
    let mut sum = 0.0;
    for i in 0..N {
        let rho = r * ((f64::from(i) + 0.5) / f64::from(N)).sqrt();
        for j in 0..N {
            let theta = 2.0 * PI * (f64::from(j) + 0.5) / f64::from(N);
            sum += height(cx + rho * theta.cos(), cy + rho * theta.sin());
        }
    }
    sum * PI * r * r / f64::from(N * N)
}

/// Validity, an analytic census, the exact volume, a closed mesh, and the
/// drilled points carved away with the material around them kept.
fn check(
    topo: &Topology,
    result: SolidId,
    census: &[(&str, usize)],
    truth: f64,
    carved: &[Point3],
    kept: &[Point3],
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mut faces: BTreeMap<&str, usize> = BTreeMap::new();
    for face in solid_faces(topo, result).unwrap() {
        *faces
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(
        faces,
        census.iter().copied().collect(),
        "the cut stays analytic"
    );
    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - truth).abs() < 1e-7 * truth,
        "volume {exact}, truth {truth}"
    );
    assert!(is_watertight(
        &tessellate_solid(topo, result, 0.01).unwrap()
    ));
    let classify =
        |p: Point3| classify_point(topo, result, p, &ClassifyOptions::default()).unwrap();
    for &p in carved {
        assert_eq!(classify(p), PointClassification::Outside, "{p:?} is carved");
    }
    for &p in kept {
        assert_eq!(classify(p), PointClassification::Inside, "{p:?} is kept");
    }
}

/// A radius 0.2 drill from `(x, y, 1)` up through a radius 2 ball.
fn drill_ball(x: f64, y: f64) {
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, 2.0, 16).unwrap();
    let undrilled = oriented_solid_volume(&topo, ball, 0.001).unwrap();
    let drill = make_cylinder(&mut topo, 0.2, 2.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(x, y, 1.0)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, ball, drill).unwrap();

    let plug = over_disc(x, y, 0.2, |px, py| (4.0 - px * px - py * py).sqrt() - 1.0);
    check(
        &topo,
        result,
        &[("cylinder", 1), ("plane", 1), ("sphere", 2)],
        4.0 / 3.0 * PI * 8.0 - plug,
        &[Point3::new(x, y, 1.1), Point3::new(x, y, 1.8)],
        &[
            Point3::new(x, y, 0.9),
            Point3::new(x * 1.8, y * 1.8, 1.5),
            Point3::new(0.0, 0.0, -1.5),
        ],
    );
    // The hemispheres meet on a chordal equator, which the mesh follows: it
    // loses the plug from the undrilled ball's mesh.
    let meshed = oriented_solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (meshed - (undrilled - plug)).abs() < 1e-3,
        "mesh volume {meshed}, undrilled mesh {undrilled} less the plug {plug}"
    );
}

/// A radius 0.3 drill parallel to the axis of a (5, 1) ring, through its
/// tube at `(x, y)`.
fn drill_ring(x: f64, y: f64) {
    let mut topo = Topology::new();
    let ring = make_torus(&mut topo, 5.0, 1.0, 16).unwrap();
    let drill = make_cylinder(&mut topo, 0.3, 4.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(x, y, -2.0)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, ring, drill).unwrap();

    let plug = over_disc(x, y, 0.3, |px, py| {
        let reach = 1.0 - (px.hypot(py) - 5.0).powi(2);
        2.0 * reach.max(0.0).sqrt()
    });
    let truth = 2.0 * PI * PI * 5.0 - plug;
    let (ux, uy) = (x / 5.0, y / 5.0);
    check(
        &topo,
        result,
        &[("cylinder", 1), ("torus", 1)],
        truth,
        &[Point3::new(x, y, 0.0), Point3::new(x, y, 0.8)],
        &[
            Point3::new(x - uy * 0.6, y + ux * 0.6, 0.0),
            Point3::new(-x, -y, 0.0),
            Point3::new(x + ux * 0.6, y + uy * 0.6, 0.0),
        ],
    );
    let meshed = oriented_solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "mesh volume {meshed}, truth {truth}"
    );
}

#[test]
fn ball_drilled_across_its_seam() {
    drill_ball(0.5, 0.0);
}

#[test]
fn ball_drilled_clear_of_its_seam() {
    drill_ball(0.0, 0.5);
}

#[test]
fn ring_drilled_across_its_seam() {
    drill_ring(5.0, 0.0);
}

#[test]
fn ring_drilled_clear_of_its_seam() {
    drill_ring(0.0, 5.0);
}

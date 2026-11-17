//! A shaft turned from a profile: a cylinder that tapers along a cone to a
//! smaller end disc, the outside of a universal joint's yoke (the brepjs
//! playground example `universal-joint`). Its analytic classifier must not
//! read the pipe of the cylinder over the taper, and a block across the
//! taper intersects it exactly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_algo::FaceClass;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::primitives::make_box;
use brepkit_operations::revolve::revolve;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// The profile `(x, r)` revolved about the x axis: a cylinder of radius 47
/// from x = -46 to 22, and a cone from it down to radius 34 at x = -62.
fn shaft(topo: &mut Topology) -> SolidId {
    let points: Vec<Point3> = [
        (-62.0, 0.0),
        (22.0, 0.0),
        (22.0, 47.0),
        (-46.0, 47.0),
        (-62.0, 34.0),
    ]
    .iter()
    .map(|&(x, r)| Point3::new(x, r, 0.0))
    .collect();
    let wire = make_polygon_wire(topo, &points, 1e-7).unwrap();
    let face = make_planar_face_from_wire(topo, wire).unwrap();
    revolve(
        topo,
        face,
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        2.0 * PI,
    )
    .unwrap()
}

/// The radius of the shaft at `x`.
fn radius_at(x: f64) -> f64 {
    if x < -46.0 {
        34.0 + (x + 62.0) * 13.0 / 16.0
    } else {
        47.0
    }
}

#[test]
fn the_analytic_classifier_reads_the_taper() {
    let mut topo = Topology::new();
    let solid = shaft(&mut topo);
    // Beside the cone and inside the cylinder's radius, then inside the cone.
    for (p, inside) in [
        (Point3::new(-50.0, 0.0, 44.5), false),
        (Point3::new(-55.0, 10.0, 40.0), false),
        (Point3::new(-50.0, 0.0, 43.5), true),
        (Point3::new(-47.0, 0.0, 44.5), true),
    ] {
        let read = brepkit_algo::classifier::classify_analytic(&topo, solid, p);
        assert!(
            read.is_none_or(|c| c
                == if inside {
                    FaceClass::Inside
                } else {
                    FaceClass::Outside
                }),
            "{p:?} reads {read:?}"
        );
    }
}

/// The yoke's bridge, a block over x = -62 to -38, |y| <= 21, |z| <= 44.5,
/// across the taper: its corners at |z| = 44.5 stand outside the cone.
#[test]
fn a_block_across_the_taper_intersects_it_exactly() {
    let mut topo = Topology::new();
    let solid = shaft(&mut topo);
    let block = make_box(&mut topo, 24.0, 42.0, 89.0).unwrap();
    transform_solid(
        &mut topo,
        block,
        &brepkit_math::mat::Mat4::translation(-62.0, -21.0, -44.5),
    )
    .unwrap();
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let common = boolean_with_options(&mut topo, BooleanOp::Intersect, block, solid, exact)
        .map_err(|e| e.to_string())
        .unwrap();
    assert!(validate_solid(&topo, common).unwrap().is_valid());
    assert!(is_watertight(
        &tessellate_solid(&topo, common, 0.01).unwrap()
    ));
    assert!(
        solid_faces(&topo, common).unwrap().iter().any(|&f| topo
            .face(f)
            .unwrap()
            .surface()
            .type_tag()
            == "cone"),
        "the cone is gone"
    );
    let n = 16;
    let mut read = 0;
    #[allow(clippy::cast_precision_loss)]
    for (i, j, k) in (0..n).flat_map(|i| (0..n).flat_map(move |j| (0..n).map(move |k| (i, j, k)))) {
        let at = |s: usize, lo: f64, hi: f64| lo + (hi - lo) * (s as f64 + 0.5) / n as f64;
        let p = Point3::new(at(i, -63.0, -37.0), at(j, -22.0, 22.0), at(k, -46.0, 46.0));
        let r = p.y().hypot(p.z());
        let truth = p.x() > -62.0
            && p.x() < -38.0
            && p.y().abs() < 21.0
            && p.z().abs() < 44.5
            && r < radius_at(p.x());
        let margin = [
            (p.x() + 62.0).abs(),
            (p.x() + 38.0).abs(),
            (p.y().abs() - 21.0).abs(),
            (p.z().abs() - 44.5).abs(),
            (r - radius_at(p.x())).abs(),
        ]
        .into_iter()
        .fold(f64::INFINITY, f64::min);
        if margin < 0.05 {
            continue;
        }
        let got = classify_point(&topo, common, p, 0.01, 1e-7).unwrap();
        let want = if truth {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(got, want, "at {p:?}");
        read += 1;
    }
    assert!(read > n * n * n / 2, "only {read} points read");
}

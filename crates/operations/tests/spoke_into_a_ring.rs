//! A spoke fused into a ring from its bore, the spoke's bottom in the plane
//! of the ring's bottom cap. The spoke's end stops in the ring's material,
//! so its outline crosses the cap's hole twice and ends in the cap. The
//! hole's rim is split at the two crossings into a short arc and one of most
//! of a turn, and the cap divides into the patch under the spoke and the
//! rest of the ring (the long arc was read as its short complement, the walk
//! ran on round the rim past the spoke's outline, and the fuse fell back to
//! a mesh).
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

const OUTER: f64 = 7.025;
const BORE: f64 = 6.25;
const RING_H: f64 = 0.45;
const HALF_W: f64 = 0.35;

fn exact() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

/// The area between the chord `x = to` and the circle of radius `r` about
/// the origin, over `|y| <= HALF_W`.
fn past_circle(r: f64, to: f64) -> f64 {
    let under = |y: f64| 0.5 * (y * (r * r - y * y).sqrt() + r * r * (y / r).asin());
    2.0 * (to * HALF_W - under(HALF_W))
}

/// A spoke from `x0` to `x1`, `height` high from the ring's bottom, fused
/// into the ring.
fn check(x0: f64, x1: f64, height: f64, pose: &Mat4) {
    let label = format!("spoke {x0}..{x1}, height {height}");
    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, OUTER, RING_H).unwrap();
    let drill = make_cylinder(&mut topo, BORE, RING_H + 0.2).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(0.0, 0.0, -0.1)).unwrap();
    let ring = boolean_with_options(&mut topo, BooleanOp::Cut, disc, drill, exact()).unwrap();
    let spoke = make_box(&mut topo, x1 - x0, 2.0 * HALF_W, height).unwrap();
    transform_solid(&mut topo, spoke, &Mat4::translation(x0, -HALF_W, 0.0)).unwrap();
    transform_solid(&mut topo, ring, pose).unwrap();
    transform_solid(&mut topo, spoke, pose).unwrap();

    let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, ring, spoke, exact())
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, fused).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, fused, 0.01).unwrap()),
        "{label}: mesh not watertight"
    );
    let faces = solid_faces(&topo, fused).unwrap();
    let walls = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    assert_eq!(walls, 2, "{label}: the ring's two walls");
    assert_eq!(
        faces
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
            .count(),
        faces.len() - 2,
        "{label}: every other face is flat"
    );

    // The spoke adds its part in the bore, up to the ring's inner wall.
    let ring_volume = PI * (OUTER * OUTER - BORE * BORE) * RING_H;
    let in_ring = past_circle(BORE, x1.min(OUTER)) * height.min(RING_H);
    let want = ring_volume + (x1 - x0) * 2.0 * HALF_W * height - in_ring;
    let volume = solid_volume(&topo, fused, 0.001).unwrap();
    assert!(
        (volume - want).abs() <= 1e-7 * want,
        "{label}: volume {volume}, want {want}"
    );

    let tol = brepkit_math::tolerance::Tolerance::new();
    let mid = 0.5 * height.min(RING_H);
    for (p, inside) in [
        (Point3::new(3.0, 0.0, mid), true),
        (Point3::new(6.3, 0.0, mid), true),
        (Point3::new(0.0, 6.6, mid), true),
        (Point3::new(-6.6, 0.0, mid), true),
        (Point3::new(3.0, 1.0, mid), false),
        (Point3::new(-3.0, 0.0, mid), false),
        (Point3::new(3.0, 0.0, height + 0.1), false),
    ] {
        let class = classify_point(&topo, fused, pose.mul_point(p), 0.01, tol.linear).unwrap();
        let want = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(class, want, "{label}: point {p:?}");
    }
}

#[test]
fn a_spoke_ends_in_a_ring_flush_with_its_bottom() {
    check(0.7, 6.4, 0.3, &Mat4::identity());
}

#[test]
fn a_spoke_as_high_as_the_ring_ends_in_it() {
    check(0.7, 6.4, RING_H, &Mat4::identity());
}

#[test]
fn a_spoke_that_ends_near_either_wall() {
    check(0.7, 6.26, 0.3, &Mat4::identity());
    check(0.7, 7.0, 0.3, &Mat4::identity());
}

#[test]
fn a_turned_and_a_mirrored_ring_take_the_spoke() {
    let turned =
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    check(0.7, 6.4, 0.3, &turned);
    let mirrored = Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_z(2.3);
    check(0.7, 6.4, 0.3, &mirrored);
}

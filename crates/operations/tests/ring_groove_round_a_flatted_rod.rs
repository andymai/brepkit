//! A rod with one or two flats and a groove: a torus ring round its axis.
//!
//! Flats first, then the ring: the ring's circles on the rod's wall are cut
//! by the flats into arcs, one for each part of the wall. A whole torus has
//! no outline, so its face's box was a point, every arc read as off the
//! ring, and the cut fell back to a mesh.
//!
//! The ring first, then the flats: each flat meets the groove's torus band
//! in a loop round the tube, of which the band keeps the part past half the
//! loop. The section's curve was the whole loop, so its reversed copy read
//! the whole loop through its own ends; and the band's pieces, whose seam is
//! a meridian arc and not a line, read as holes. The cut fell back to a mesh.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 6.0;
const HEIGHT: f64 = 16.0;
const FLAT: f64 = 5.4;
const TUBE: f64 = 0.32;
const GROOVE_Z: f64 = 2.0;

fn exact() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

fn cut(topo: &mut Topology, a: SolidId, b: SolidId, label: &str) -> SolidId {
    boolean_with_options(topo, BooleanOp::Cut, a, b, exact())
        .unwrap_or_else(|e| panic!("{label}: {e}"))
}

/// The cuts in the order they are made.
#[derive(Clone, Copy, Debug)]
enum Order {
    FlatsFirst,
    RingFirst,
}

/// The rod with `flats` flats (past `y = FLAT`, then past `y = -FLAT`) and
/// the ring's groove.
fn build(
    topo: &mut Topology,
    major: f64,
    flats: usize,
    order: Order,
    pose: &Mat4,
    label: &str,
) -> SolidId {
    let rod = make_cylinder(topo, RADIUS, HEIGHT).unwrap();
    let ring = make_torus(topo, major, TUBE, 32).unwrap();
    transform_solid(topo, ring, &Mat4::translation(0.0, 0.0, GROOVE_Z)).unwrap();
    let blocks: Vec<SolidId> = [FLAT, -20.0 - FLAT][..flats]
        .iter()
        .map(|&y| {
            let block = make_box(topo, 20.0, 20.0, 20.0).unwrap();
            transform_solid(topo, block, &Mat4::translation(-10.0, y, -2.0)).unwrap();
            block
        })
        .collect();
    for &solid in [rod, ring].iter().chain(&blocks) {
        transform_solid(topo, solid, pose).unwrap();
    }
    match order {
        Order::FlatsFirst => {
            let flatted = blocks
                .iter()
                .fold(rod, |solid, &block| cut(topo, solid, block, label));
            cut(topo, flatted, ring, label)
        }
        Order::RingFirst => {
            let grooved = cut(topo, rod, ring, label);
            blocks
                .iter()
                .fold(grooved, |solid, &block| cut(topo, solid, block, label))
        }
    }
}

/// The turn about the axis that the flatted rod covers at radius `s`.
fn turn_at(s: f64, flats: usize) -> f64 {
    let cut_off = 2.0 * (FLAT / s).min(1.0).acos();
    2.0f64.mul_add(std::f64::consts::PI, -(flats as f64) * cut_off)
}

/// The volume of the flatted rod less the ring's tube. At radius
/// `major + TUBE sin(a)` the tube is `2 TUBE cos(a)` high.
fn grooved_volume(major: f64, flats: usize) -> f64 {
    let segment =
        RADIUS * RADIUS * (FLAT / RADIUS).acos() - FLAT * (RADIUS * RADIUS - FLAT * FLAT).sqrt();
    let flatted = (std::f64::consts::PI * RADIUS * RADIUS - flats as f64 * segment) * HEIGHT;
    let last = ((RADIUS - major) / TUBE).clamp(-1.0, 1.0).asin();
    let first = -std::f64::consts::FRAC_PI_2;
    let f = |a: f64| {
        let s = TUBE.mul_add(a.sin(), major);
        2.0 * TUBE * TUBE * a.cos() * a.cos() * s * turn_at(s, flats)
    };
    let n = 20_000;
    let h = (last - first) / f64::from(n);
    let mut sum = f(first) + f(last);
    for i in 1..n {
        sum += f(h.mul_add(f64::from(i), first)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    flatted - sum * h / 3.0
}

fn check(major: f64, flats: usize, order: Order, pose: &Mat4) {
    let label = format!("ring of radius {major}, {flats} flats, {order:?}");
    let mut topo = Topology::new();
    let solid = build(&mut topo, major, flats, order, pose, &label);

    let report = validate_solid(&topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    assert!(
        is_watertight(&tessellate_solid(&topo, solid, 0.01).unwrap()),
        "{label}: mesh not watertight"
    );
    let faces = solid_faces(&topo, solid).unwrap();
    let count = |want: fn(&FaceSurface) -> bool| {
        faces
            .iter()
            .filter(|&&f| want(topo.face(f).unwrap().surface()))
            .count()
    };
    // A piece of groove between each two flats, the two ends and the flats.
    assert_eq!(
        count(|s| matches!(s, FaceSurface::Torus(_))),
        flats,
        "{label}"
    );
    assert_eq!(
        count(|s| matches!(s, FaceSurface::Plane { .. })),
        2 + flats,
        "{label}"
    );
    assert_eq!(
        count(|s| matches!(s, FaceSurface::Cylinder(_))),
        faces.len() - 2 - 2 * flats,
        "{label}: every other face is a part of the wall"
    );

    let volume = solid_volume(&topo, solid, 0.001).unwrap();
    let want = grooved_volume(major, flats);
    assert!(
        (volume - want).abs() <= 1e-6 * want,
        "{label}: volume {volume}, want {want}"
    );

    let tol = brepkit_math::tolerance::Tolerance::new();
    let in_tube = major - 0.5 * TUBE;
    for (p, inside) in [
        (Point3::new(0.0, 0.0, GROOVE_Z), true),
        (Point3::new(RADIUS - 0.1, 0.0, 8.0), true),
        (
            Point3::new(in_tube.min(RADIUS - 0.01), 0.0, GROOVE_Z),
            false,
        ),
        (
            Point3::new(-in_tube.min(RADIUS - 0.01), 0.0, GROOVE_Z),
            false,
        ),
        (Point3::new(0.0, FLAT + 0.1, 8.0), false),
        (Point3::new(0.0, FLAT - 0.1, GROOVE_Z), true),
        (Point3::new(0.0, -FLAT - 0.1, 8.0), flats == 1),
        (
            Point3::new(0.0, -in_tube.min(RADIUS - 0.01), GROOVE_Z),
            false,
        ),
    ] {
        let class = classify_point(&topo, solid, pose.mul_point(p), 0.01, tol.linear).unwrap();
        let want = if inside {
            PointClassification::Inside
        } else {
            PointClassification::Outside
        };
        assert_eq!(class, want, "{label}: point {p:?}");
    }
}

fn turned() -> Mat4 {
    Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9)
}

fn mirrored() -> Mat4 {
    Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4)
}

#[test]
fn a_ring_on_the_wall_grooves_a_flatted_rod() {
    check(RADIUS, 2, Order::FlatsFirst, &Mat4::identity());
}

#[test]
fn a_ring_wider_or_narrower_than_the_rod_grooves_it() {
    check(RADIUS + 0.1, 2, Order::FlatsFirst, &Mat4::identity());
    check(RADIUS - 0.1, 2, Order::FlatsFirst, &Mat4::identity());
}

#[test]
fn a_turned_and_a_mirrored_rod_take_the_groove() {
    check(RADIUS, 2, Order::FlatsFirst, &turned());
    check(RADIUS, 2, Order::FlatsFirst, &mirrored());
}

#[test]
fn a_flat_cuts_through_the_groove() {
    check(RADIUS, 1, Order::RingFirst, &Mat4::identity());
    check(RADIUS, 2, Order::RingFirst, &Mat4::identity());
}

#[test]
fn a_flat_cuts_through_a_groove_wider_or_narrower_than_the_rod() {
    for flats in [1, 2] {
        check(RADIUS + 0.1, flats, Order::RingFirst, &Mat4::identity());
        check(RADIUS - 0.1, flats, Order::RingFirst, &Mat4::identity());
    }
}

#[test]
fn a_flat_cuts_through_the_groove_of_a_turned_and_a_mirrored_rod() {
    for major in [RADIUS, RADIUS + 0.1, RADIUS - 0.1] {
        for flats in [1, 2] {
            check(major, flats, Order::RingFirst, &turned());
            check(major, flats, Order::RingFirst, &mirrored());
        }
    }
}

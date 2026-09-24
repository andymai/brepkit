//! Rods cut by planes along their axis: a half rod, a D-shaft (a flat
//! ground off a rod) and a quarter rod. Their caps are
//! bounded by arcs and chords and their walls by rims and rulings, so their
//! volumes and cap areas have closed forms.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

/// Cut `solid` by an axis-aligned box from `min` to `max`.
fn cut_box(topo: &mut Topology, solid: SolidId, min: [f64; 3], max: [f64; 3]) -> SolidId {
    let block = make_box(topo, max[0] - min[0], max[1] - min[1], max[2] - min[2]).unwrap();
    transform_solid(topo, block, &Mat4::translation(min[0], min[1], min[2])).unwrap();
    boolean(topo, BooleanOp::Cut, solid, block).unwrap()
}

/// A valid solid of one cylinder wall and planes, the given points kept and
/// cut, `solid_volume` to 1e-9 and, for each cap perpendicular to z,
/// `face_area` to 1e-12 of one of `cap_areas`.
fn check(
    topo: &Topology,
    solid: SolidId,
    volume: f64,
    cap_areas: &[f64],
    kept: &[Point3],
    cut: &[Point3],
) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let faces = solid_faces(topo, solid).unwrap();
    let cylinders = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .count();
    let planes = faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
        .count();
    assert!(
        cylinders >= 1 && cylinders + planes == faces.len(),
        "the cut stays analytic: {cylinders} cylinders, {planes} planes of {}",
        faces.len()
    );
    let classify = |p: Point3| classify_point(topo, solid, p, &ClassifyOptions::default()).unwrap();
    for &p in kept {
        assert_eq!(classify(p), PointClassification::Inside, "{p:?} is kept");
    }
    for &p in cut {
        assert_eq!(classify(p), PointClassification::Outside, "{p:?} is cut");
    }
    let measured = solid_volume(topo, solid, 0.01).unwrap();
    assert!(
        (measured - volume).abs() < 1e-9 * volume,
        "volume {measured}, truth {volume}"
    );
    let mut caps = 0;
    for face in solid_faces(topo, solid).unwrap() {
        if let FaceSurface::Plane { normal, .. } = topo.face(face).unwrap().surface()
            && normal.z().abs() > 0.5
        {
            caps += 1;
            let area = face_area(topo, face, 0.01).unwrap();
            assert!(
                cap_areas
                    .iter()
                    .any(|&cap| (area - cap).abs() < 1e-12 * cap),
                "cap area {area}, truths {cap_areas:?}"
            );
        }
    }
    assert!(caps > 0, "no cap measured");
}

#[test]
fn half_rod() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let half = cut_box(&mut topo, rod, [-5.0, -5.0, -1.0], [0.0, 5.0, 4.0]);
    check(
        &topo,
        half,
        6.0 * PI,
        &[2.0 * PI],
        &[Point3::new(1.0, 0.5, 1.5), Point3::new(1.9, -0.2, 0.1)],
        &[Point3::new(-0.5, 0.0, 1.5), Point3::new(-1.9, 0.2, 2.9)],
    );
}

/// The flat at x = 1 cuts a segment of angle 2 pi / 3 off each cap.
#[test]
fn d_shaft() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let shaft = cut_box(&mut topo, rod, [1.0, -5.0, -1.0], [5.0, 5.0, 4.0]);
    let angle = 2.0 * PI / 3.0;
    let cap = 4.0 * PI - 2.0 * (angle - angle.sin());
    check(
        &topo,
        shaft,
        3.0 * cap,
        &[cap],
        &[Point3::new(0.9, 0.0, 1.5), Point3::new(-1.9, 0.0, 1.5)],
        &[Point3::new(1.1, 0.0, 1.5), Point3::new(1.5, 1.0, 2.0)],
    );
}

#[test]
fn quarter_rod() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 2.0, 3.0).unwrap();
    let half = cut_box(&mut topo, rod, [-5.0, -5.0, -1.0], [0.0, 5.0, 4.0]);
    let quarter = cut_box(&mut topo, half, [-5.0, -5.0, -1.0], [5.0, 0.0, 4.0]);
    check(
        &topo,
        quarter,
        3.0 * PI,
        &[PI],
        &[Point3::new(1.0, 1.0, 1.5)],
        &[
            Point3::new(-1.0, 1.0, 1.5),
            Point3::new(1.0, -1.0, 1.5),
            Point3::new(-1.0, -1.0, 1.5),
        ],
    );
}

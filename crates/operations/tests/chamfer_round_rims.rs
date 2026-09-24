//! Chamfers of closed circular rims, where a flat cap meets a coaxial
//! cylinder or cone wall: a rod's ends, a tube's mouth, a frustum's top and
//! a hole's countersink. Each removes a ring whose cross-section is a
//! triangle, so its volume is the triangle's area times the path of its
//! centroid.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

/// The closed circle edges of `solid` at height `z`.
fn rims_at(topo: &Topology, solid: SolidId, z: f64) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(_))
                && edge.start() == edge.end()
                && (topo.vertex(edge.start()).unwrap().point().z() - z).abs() < 1e-9
        })
        .collect()
}

/// The ring a chamfer cuts from a rim of radius `rim`: the triangle between
/// the rim point `(rim, 0)`, the cap point `(rim + cap, 0)` and the wall
/// point `(rim + wall.0, wall.1)`, revolved.
fn ring(rim: f64, cap: f64, wall: (f64, f64)) -> f64 {
    let area = 0.5 * (cap * wall.1).abs();
    let centroid = rim + (cap + wall.0) / 3.0;
    2.0 * PI * centroid * area
}

fn check(
    topo: &Topology,
    result: SolidId,
    expected: f64,
    census: &[(&str, usize)],
    inside: &[Point3],
    outside: &[Point3],
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mut faces: BTreeMap<&str, usize> = BTreeMap::new();
    for face in solid_faces(topo, result).unwrap() {
        *faces
            .entry(topo.face(face).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(faces, census.iter().copied().collect(), "face census");

    let classify =
        |p: Point3| classify_point(topo, result, p, &ClassifyOptions::default()).unwrap();
    for &p in inside {
        assert_eq!(classify(p), PointClassification::Inside, "{p:?} is kept");
    }
    for &p in outside {
        assert_eq!(classify(p), PointClassification::Outside, "{p:?} is cut");
    }

    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - expected).abs() < 1e-7 * expected,
        "volume {exact}, truth {expected}"
    );
    let mesh = tessellate_solid(topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let meshed = oriented_solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected).abs() < 2e-3 * expected,
        "mesh volume {meshed}, truth {expected}"
    );
}

#[test]
fn rod_end_rim() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let rims = rims_at(&topo, rod, 10.0);
    let result = chamfer(&mut topo, rod, &rims, 1.0).unwrap();
    check(
        &topo,
        result,
        90.0 * PI - ring(3.0, -1.0, (0.0, -1.0)),
        &[("cone", 1), ("cylinder", 1), ("plane", 2)],
        &[Point3::new(0.0, 0.0, 9.9), Point3::new(2.4, 0.0, 9.5)],
        &[Point3::new(2.9, 0.0, 9.9), Point3::new(0.0, -2.8, 9.8)],
    );
}

#[test]
fn rod_both_rims() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let mut rims = rims_at(&topo, rod, 10.0);
    rims.extend(rims_at(&topo, rod, 0.0));
    let result = chamfer(&mut topo, rod, &rims, 1.0).unwrap();
    check(
        &topo,
        result,
        90.0 * PI - 2.0 * ring(3.0, -1.0, (0.0, -1.0)),
        &[("cone", 2), ("cylinder", 1), ("plane", 2)],
        &[Point3::new(2.4, 0.0, 0.5), Point3::new(2.4, 0.0, 9.5)],
        &[Point3::new(2.9, 0.0, 0.1), Point3::new(2.9, 0.0, 9.9)],
    );
}

#[test]
fn tube_mouth_rims() {
    let mut topo = Topology::new();
    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let inner = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert_eq!(rims.len(), 2);
    let d = 1.0 / 3.0;
    let result = chamfer(&mut topo, tube, &rims, d).unwrap();
    check(
        &topo,
        result,
        50.0 * PI - ring(3.0, -d, (0.0, -d)) - ring(2.0, d, (0.0, -d)),
        &[("cone", 2), ("cylinder", 2), ("plane", 2)],
        &[Point3::new(2.5, 0.0, 9.9), Point3::new(0.0, 2.5, 5.0)],
        &[
            Point3::new(2.95, 0.0, 9.95),
            Point3::new(0.0, 2.05, 9.95),
            Point3::new(0.0, 0.0, 9.0),
        ],
    );
}

/// A tube counterbored at its mouth: the cap is the ring between the outer
/// wall and the counterbore, and the counterbore's wall ends at its step.
#[test]
fn counterbored_tube_mouth_rims() {
    let mut topo = Topology::new();
    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 1.5, 12.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, bore).unwrap();
    let counterbore = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, counterbore, &Mat4::translation(0.0, 0.0, 7.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, tube, counterbore).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert_eq!(rims.len(), 2);
    let d = 1.0 / 3.0;
    let result = chamfer(&mut topo, tube, &rims, d).unwrap();
    check(
        &topo,
        result,
        PI * (6.75 * 10.0 - 1.75 * 3.0) - ring(3.0, -d, (0.0, -d)) - ring(2.0, d, (0.0, -d)),
        &[("cone", 2), ("cylinder", 3), ("plane", 3)],
        &[Point3::new(2.5, 0.0, 9.9), Point3::new(0.0, 1.7, 5.0)],
        &[
            Point3::new(2.95, 0.0, 9.95),
            Point3::new(0.0, 2.05, 9.95),
            Point3::new(0.0, 1.8, 8.0),
        ],
    );
}

/// The wall runs from the top rim (radius 2 at z = 6) down to radius 3 at
/// z = 0, so the chamfer's wall point moves `d` along that slant.
#[test]
fn frustum_top_rim() {
    let mut topo = Topology::new();
    let frustum = make_cone(&mut topo, 3.0, 2.0, 6.0).unwrap();
    let rims = rims_at(&topo, frustum, 6.0);
    let d = 0.5;
    let slant = 37.0_f64.sqrt();
    let result = chamfer(&mut topo, frustum, &rims, d).unwrap();
    check(
        &topo,
        result,
        38.0 * PI - ring(2.0, -d, (d / slant, -6.0 * d / slant)),
        &[("cone", 2), ("plane", 2)],
        &[Point3::new(1.4, 0.0, 5.9), Point3::new(2.3, 0.0, 3.0)],
        &[Point3::new(1.95, 0.0, 5.97)],
    );
}

/// A through hole in a block: the chamfer's cap circle grows into the top
/// face and its wall circle moves down the bore.
#[test]
fn hole_countersink() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(5.0, 5.0, -1.0)).unwrap();
    let holed = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
    let rims = rims_at(&topo, holed, 10.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, holed, &rims, d).unwrap();
    check(
        &topo,
        result,
        1000.0 - 40.0 * PI - ring(2.0, d, (0.0, -d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 6)],
        &[Point3::new(7.6, 5.0, 9.9), Point3::new(5.0, 7.3, 5.0)],
        &[Point3::new(7.05, 5.0, 9.95), Point3::new(5.0, 5.0, 9.9)],
    );
}

/// The floor rim of a blind hole is concave: its chamfer fills the corner,
/// adding the ring instead of removing it.
#[test]
fn blind_hole_floor_rim() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 2.0, 6.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(5.0, 5.0, 5.0)).unwrap();
    let holed = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
    let rims = rims_at(&topo, holed, 5.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, holed, &rims, d).unwrap();
    check(
        &topo,
        result,
        1000.0 - 20.0 * PI + ring(2.0, -d, (0.0, d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 7)],
        &[Point3::new(6.9, 5.0, 5.05), Point3::new(5.0, 5.0, 4.9)],
        &[Point3::new(6.6, 5.0, 5.3), Point3::new(5.0, 5.0, 5.1)],
    );
}

/// A closed cylindrical cavity inside a block: its rims bound an inner
/// shell, and the chamfer's band joins that shell, filling the corner.
#[test]
fn enclosed_cavity_rim() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let cavity = make_cylinder(&mut topo, 2.0, 4.0).unwrap();
    transform_solid(&mut topo, cavity, &Mat4::translation(5.0, 5.0, 3.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, block, cavity).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);
    let rims = rims_at(&topo, hollow, 7.0);
    assert_eq!(rims.len(), 1);
    let d = 0.5;
    let result = chamfer(&mut topo, hollow, &rims, d).unwrap();
    assert_eq!(topo.solid(result).unwrap().inner_shells().len(), 1);
    check(
        &topo,
        result,
        1000.0 - 16.0 * PI + ring(2.0, -d, (0.0, -d)),
        &[("cone", 1), ("cylinder", 1), ("plane", 8)],
        &[Point3::new(6.9, 5.0, 6.95), Point3::new(8.0, 5.0, 5.0)],
        &[Point3::new(6.6, 5.0, 6.6), Point3::new(5.0, 5.0, 5.0)],
    );
}

/// A distance that consumes the cap or the wall is refused, not built.
#[test]
fn oversized_distances_are_refused() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let rims = rims_at(&topo, rod, 10.0);
    assert!(chamfer(&mut topo, rod, &rims, 3.5).is_err());

    let short = make_cylinder(&mut topo, 3.0, 1.5).unwrap();
    let mut rims = rims_at(&topo, short, 1.5);
    rims.extend(rims_at(&topo, short, 0.0));
    assert!(chamfer(&mut topo, short, &rims, 1.0).is_err());

    let outer = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    let inner = make_cylinder(&mut topo, 2.0, 12.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    let tube = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let rims = rims_at(&topo, tube, 10.0);
    assert!(chamfer(&mut topo, tube, &rims, 0.5).is_err());
}

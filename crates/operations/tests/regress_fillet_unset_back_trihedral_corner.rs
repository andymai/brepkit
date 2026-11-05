//! Regression: the trihedral corner of a box, whatever its proportions, sets
//! its stripes back to the rolling ball, so its patch lies on that ball with
//! contacts `r * sqrt(2)` from the vertex.
//!
//! A guard once left any corner whose longest stripe was over twice its shortest
//! un-set-back, with contacts `r` from the vertex and a cap of radius
//! `r * sqrt(2)`. The result read closed and valid, so this test passed on it, but
//! a 90 x 60 x 10 box at r = 0.5 measured 48982.38 against 53965.83 with an open
//! mesh, which is why these tests also read the volume. The un-set-back
//! classification and cap halves still live in `crates/blend/src/corner.rs` and
//! `crates/blend/src/spherical_triangle.rs`, for a corner no setback ball fits.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;
use brepkit_topology::test_utils::make_unit_cube_manifold;

const GEOMETRY_TOL: f64 = 1e-4;

/// A unit-cube manifold scaled to `dimensions`, with the face planes'
/// offsets rewritten to match.
fn scaled_box(topo: &mut Topology, dimensions: Vec3) -> SolidId {
    let solid = make_unit_cube_manifold(topo);
    for vertex_id in solid_vertices(topo, solid).unwrap() {
        let point = topo.vertex(vertex_id).unwrap().point();
        topo.vertex_mut(vertex_id).unwrap().set_point(Point3::new(
            point.x() * dimensions.x(),
            point.y() * dimensions.y(),
            point.z() * dimensions.z(),
        ));
    }
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            unreachable!("test box faces are planar")
        };
        let normal = *normal;
        let d = normal.x().max(0.0) * dimensions.x()
            + normal.y().max(0.0) * dimensions.y()
            + normal.z().max(0.0) * dimensions.z();
        topo.face_mut(face_id)
            .unwrap()
            .set_surface(FaceSurface::Plane { normal, d });
    }
    solid
}

/// The corner patch face at the origin corner: the face whose outer wire's
/// unique vertices are exactly the three contacts, each at `distance`
/// (within `GEOMETRY_TOL`) from the origin. Exactly one must exist.
///
/// The patch's wire carries six oriented edges — each terminal arc appears
/// twice, forward and reversed, because the corner face and the adjacent
/// band face share the boundary through the registry — so the search keys
/// on unique vertex positions, not wire-edge count.
fn origin_corner_face(topo: &Topology, solid: SolidId, distance: f64) -> [Point3; 3] {
    let origin = Point3::new(0.0, 0.0, 0.0);
    let mut matches = Vec::new();
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        let wire = topo.wire(face.outer_wire()).unwrap();
        let mut points = Vec::new();
        for oriented in wire.edges() {
            let edge = topo.edge(oriented.edge()).unwrap();
            let start = topo.vertex(edge.start()).unwrap().point();
            let end = topo.vertex(edge.end()).unwrap().point();
            if !points.contains(&start) {
                points.push(start);
            }
            if !points.contains(&end) {
                points.push(end);
            }
        }
        if points.len() != 3 {
            continue;
        }
        if points
            .iter()
            .all(|point| ((*point - origin).length() - distance).abs() <= GEOMETRY_TOL)
        {
            matches.push([points[0], points[1], points[2]]);
        }
    }
    assert_eq!(
        matches.len(),
        1,
        "expected exactly one three-vertex origin corner patch at distance {distance}, got {}",
        matches.len()
    );
    matches.pop().unwrap()
}

fn assert_builds_all(topo: &mut Topology, solid: SolidId, radius: f64, expect: usize) -> SolidId {
    let edges = solid_edges(topo, solid).unwrap();
    assert_eq!(edges.len(), expect);
    let result = fillet_v2(topo, solid, &edges, radius)
        .unwrap_or_else(|error| panic!("all-edge capture must succeed, got {error:?}"));
    assert_eq!(result.succeeded.len(), edges.len());
    assert!(result.failed.is_empty(), "failed: {:?}", result.failed);
    assert!(!result.is_partial);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(
        report.is_valid(),
        "result must validate: {} issues",
        report.issues.len()
    );
    result.solid
}

/// A plain 90 x 60 x 10 box, all twelve edges, at r = 0.5 and r = 0.25: builds
/// 12/12, its origin corner patch's vertices are `r * sqrt(2)` from the vertex,
/// and it measures as the smaller box swept by the ball.
#[test]
fn a_long_box_sets_its_corners_back() {
    for radius in [0.5, 0.25] {
        let mut topo = Topology::new();
        let solid = scaled_box(&mut topo, Vec3::new(90.0, 60.0, 10.0));
        let result = assert_builds_all(&mut topo, solid, radius, 12);

        let side = radius * std::f64::consts::SQRT_2;
        let contacts = origin_corner_face(&topo, result, side);
        for pair in [(0, 1), (1, 2), (2, 0)] {
            let separation = (contacts[pair.0] - contacts[pair.1]).length();
            assert!(
                (separation - side).abs() <= GEOMETRY_TOL,
                "set-back contacts must be mutually r*sqrt(2) = {side} apart, got {separation}"
            );
        }

        let (a, b, c) = (
            90.0 - 2.0 * radius,
            60.0 - 2.0 * radius,
            10.0 - 2.0 * radius,
        );
        let pi = std::f64::consts::PI;
        let expected = a * b * c
            + 2.0 * radius * (a * b + b * c + a * c)
            + pi * radius * radius * (a + b + c)
            + 4.0 / 3.0 * pi * radius * radius * radius;
        let volume = brepkit_operations::measure::solid_volume(&topo, result, 0.001).unwrap();
        assert!(
            (volume - expected).abs() < 1e-3,
            "radius {radius}: volume {volume}, expected {expected}"
        );
    }
}

/// Equal sides take the same construction: 30^3 and 60^3 boxes, all twelve
/// edges at r = 0.5, build 12/12 with the origin corner patch's vertices at
/// `r * sqrt(2)` from the vertex, the tangency-point contact convention
/// (`|contact - centre| == r`).
#[test]
fn near_equal_controls_stay_on_the_setback_path() {
    for dimensions in [Vec3::new(30.0, 30.0, 30.0), Vec3::new(60.0, 60.0, 60.0)] {
        let radius = 0.5;
        let mut topo = Topology::new();
        let solid = scaled_box(&mut topo, dimensions);
        let result = assert_builds_all(&mut topo, solid, radius, 12);

        let contacts = origin_corner_face(&topo, result, radius * std::f64::consts::SQRT_2);
        let side = radius * std::f64::consts::SQRT_2;
        for pair in [(0, 1), (1, 2), (2, 0)] {
            let separation = (contacts[pair.0] - contacts[pair.1]).length();
            assert!(
                (separation - side).abs() <= GEOMETRY_TOL,
                "set-back contacts must be mutually r*sqrt(2) = {side} apart, got {separation}"
            );
        }
    }
}

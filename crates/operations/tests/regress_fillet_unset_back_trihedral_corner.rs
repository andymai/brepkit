//! OttoCAD N424: the deliberate un-set-back trihedral corner builds again.
//!
//! N423 diagnosed the `cross_one_row_fillet_inmem` `CornerFailure { vertex:
//! Id(0) }` as overreach: N392's requested-radius check in
//! `compute_sphere_center` fired on the class upstream's own
//! near-equal-stripe guard (`a7ba558d` / #1631, named for the 90:60:10
//! cross-one-row ratio) excludes from setbacks. That class builds with a
//! different, exact contact convention — contacts at distance `radius` from
//! the *vertex*, cap sphere radius `radius * sqrt(2)` — and N424 gates the
//! check on the new `setback_trimmed == Some(true)` classification.
//!
//! This file is the build-level half of the regression: on the shipping tree
//! `9dee6009` the 90 x 60 x 10 all-twelve-edge capture fails with
//! `Blend(CornerFailure { vertex: Id(0) })` (N423 §4, reproduced below as
//! the failing-before proof); after the fix it builds 12/12 with the
//! un-set-back contact convention pinned from the built geometry. The
//! classification/cap-radius half lives in
//! `crates/blend/src/corner.rs` (`n424_unset_back_junction_*` and
//! `n424_setback_junction_*`) and
//! `crates/blend/src/spherical_triangle.rs`
//! (`test_sphere_center_accepts_unset_back_rsqrt2_contacts`), independent of
//! any fixture file.

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
/// offsets rewritten to match (the same helper N423's minimal repro used).
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

/// The minimal repro N423 §4 isolated from the 186-edge fixture: a plain
/// 90 x 60 x 10 box, all twelve edges, at r = 0.5 and r = 0.25. Builds
/// 12/12, and the origin corner patch's three vertices are the un-set-back
/// contacts — at exactly `r` from the vertex (the set-back convention would
/// put them at `r * sqrt(2)`; N423 §2.2 measured `d_vpos_minus_radius =
/// 1.11e-16` here).
#[test]
fn unset_back_90_60_10_all_edges_builds_with_vertex_radius_contacts() {
    for radius in [0.5, 0.25] {
        let mut topo = Topology::new();
        let solid = scaled_box(&mut topo, Vec3::new(90.0, 60.0, 10.0));
        let result = assert_builds_all(&mut topo, solid, radius, 12);

        let contacts = origin_corner_face(&topo, result, radius);
        let side = radius * std::f64::consts::SQRT_2;
        for pair in [(0, 1), (1, 2), (2, 0)] {
            let separation = (contacts[pair.0] - contacts[pair.1]).length();
            assert!(
                (separation - side).abs() <= GEOMETRY_TOL,
                "un-set-back contacts must be mutually r*sqrt(2) = {side} apart, got {separation}"
            );
        }
    }
}

/// The near-equal controls keep their set-back construction: 30^3 and 60^3
/// boxes, all twelve edges at r = 0.5, still build 12/12 with the origin
/// corner patch's vertices at `r * sqrt(2)` from the vertex — the
/// tangency-point contact convention (`|contact - centre| == r`, N423 §2.4),
/// not the un-set-back convention above.
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

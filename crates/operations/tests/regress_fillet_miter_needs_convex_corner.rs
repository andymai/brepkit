//! Regression: the sharp-mitered two-edge corner is only built at a convex
//! trihedral corner.
//!
//! The miter recognizer keys on two equal-radius stripes meeting at a vertex
//! with three mutually perpendicular planar faces. Two concave floor-to-wall
//! edges of a pocket and the two cap edges at an L-profile's re-entrant
//! vertex satisfy that too, but the miter's retained-edge vertex
//! `vertex - n*r` then sits inside the material and the splice cannot be
//! built. Those junctions must keep the ordinary junction fan.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_box;
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid_with_tolerance};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::builder::make_polygon_wire;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;

fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let close = |p: Point3, q: Point3| (p - q).length() < 1e-6;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&id| {
            let edge = topo.edge(id).unwrap();
            let s = topo.vertex(edge.start()).unwrap().point();
            let e = topo.vertex(edge.end()).unwrap().point();
            (close(s, a) && close(e, b)) || (close(s, b) && close(e, a))
        })
        .unwrap_or_else(|| panic!("no edge between {a:?} and {b:?}"))
}

fn assert_pair_fillets_watertight(
    topo: &mut Topology,
    solid: SolidId,
    edges: [EdgeId; 2],
    radius: f64,
) -> SolidId {
    let result = fillet_v2(topo, solid, &edges, radius)
        .unwrap_or_else(|error| panic!("r={radius}: fillet errored: {error}"));
    assert_eq!(
        result.succeeded.len(),
        2,
        "r={radius}: failed edges {:?}",
        result.failed
    );
    let validation = validate_solid(topo, result.solid).unwrap();
    assert!(validation.is_valid(), "r={radius}: {:?}", validation.issues);
    let mesh =
        tessellate_solid_with_tolerance(topo, result.solid, 0.01, 5.0_f64.to_radians()).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0, "r={radius}: open mesh edges");
    result.solid
}

/// A pocket sunk into the top of a block: the two floor-to-wall edges at a
/// pocket corner are concave, and the walls' outward normals point into the
/// pocket void. The two rounds meet along the curve where they cross, from
/// where their contacts on the floor cross, so the floor ends there. Each
/// adds its strip along its 16 long edge, and the two strips overlap in the
/// corner where a point lies outside both balls: with `w` its depth under
/// the balls' axes' plane, both its other two offsets from those axes run
/// from `sqrt(r^2 - w^2)` to `r`, so the overlap is `r^3 (5/3 - pi/2)`.
#[test]
fn concave_pocket_corner_pair_keeps_the_junction_fan() {
    let poses = [
        Mat4::identity(),
        Mat4::translation(-3.0, 18.0, 250.0) * Mat4::rotation_y(0.8) * Mat4::rotation_x(-0.3),
        Mat4::scale(1.0, 1.0, -1.0),
    ];
    for pose in poses {
        for r in [2.0, 3.5] {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 40.0, 40.0, 20.0).unwrap();
            let pocket = make_box(&mut topo, 16.0, 16.0, 10.0).unwrap();
            transform_solid(&mut topo, pocket, &Mat4::translation(12.0, 12.0, 10.0)).unwrap();
            let solid = boolean(&mut topo, BooleanOp::Cut, block, pocket).unwrap();
            transform_solid(&mut topo, solid, &pose).unwrap();

            let at = |p: Point3| pose.mul_point(p);
            let corner = at(Point3::new(12.0, 12.0, 10.0));
            let along_x = edge_between(&topo, solid, corner, at(Point3::new(28.0, 12.0, 10.0)));
            let along_y = edge_between(&topo, solid, corner, at(Point3::new(12.0, 28.0, 10.0)));
            let filleted = assert_pair_fillets_watertight(&mut topo, solid, [along_x, along_y], r);
            let strip = r * r * (1.0 - PI / 4.0);
            let overlap = r * r * r * (5.0 / 3.0 - PI / 2.0);
            let want = 40.0 * 40.0 * 20.0 - 16.0 * 16.0 * 10.0 + 2.0 * 16.0 * strip - overlap;
            let held = solid_volume(&topo, filleted, 1e-4).unwrap();
            assert!(
                (held - want).abs() <= 1e-5,
                "r={r}: volume {held}, want {want}"
            );
        }
    }
}

/// An L profile extruded along z: the two bottom-cap edges at the
/// re-entrant vertex are convex, but the retained vertical edge there is
/// concave, so the corner is a saddle rather than an octant.
#[test]
fn saddle_corner_pair_keeps_the_junction_fan() {
    let mut topo = Topology::new();
    let profile = make_polygon_wire(
        &mut topo,
        &[
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(30.0, 0.0, 0.0),
            Point3::new(30.0, 15.0, 0.0),
            Point3::new(15.0, 15.0, 0.0),
            Point3::new(15.0, 30.0, 0.0),
            Point3::new(0.0, 30.0, 0.0),
        ],
        1e-7,
    )
    .unwrap();
    let face = topo.add_face(Face::new(
        profile,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 20.0).unwrap();

    let reentrant = Point3::new(15.0, 15.0, 0.0);
    let along_x = edge_between(&topo, solid, reentrant, Point3::new(30.0, 15.0, 0.0));
    let along_y = edge_between(&topo, solid, reentrant, Point3::new(15.0, 30.0, 0.0));
    assert_pair_fillets_watertight(&mut topo, solid, [along_x, along_y], 2.0);
}

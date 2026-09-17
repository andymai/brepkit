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

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
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
) {
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
}

/// A pocket sunk into the top of a block: the two floor-to-wall edges at a
/// pocket corner are concave, and the walls' outward normals point into the
/// pocket void.
#[test]
fn concave_pocket_corner_pair_keeps_the_junction_fan() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 40.0, 40.0, 20.0).unwrap();
    let pocket = make_box(&mut topo, 16.0, 16.0, 10.0).unwrap();
    transform_solid(&mut topo, pocket, &Mat4::translation(12.0, 12.0, 10.0)).unwrap();
    let solid = boolean(&mut topo, BooleanOp::Cut, block, pocket).unwrap();

    let corner = Point3::new(12.0, 12.0, 10.0);
    let along_x = edge_between(&topo, solid, corner, Point3::new(28.0, 12.0, 10.0));
    let along_y = edge_between(&topo, solid, corner, Point3::new(12.0, 28.0, 10.0));
    assert_pair_fillets_watertight(&mut topo, solid, [along_x, along_y], 2.0);
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

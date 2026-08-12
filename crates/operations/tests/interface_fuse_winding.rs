//! #1538 coplanar-interface family: cutting a through-hole out of a plate and
//! fusing the plate onto a block across the shared plane must produce strictly
//! valid winding (the free/over census cannot see same-direction shared
//! edges; only `validate_solid`'s orientation check can).
//!
//! Four winding emitters have failed here, each invisible to watertightness:
//! - the internal-loops splitter normalized disc/hole loops with a signed
//!   area taken in the surface's own parameterization, which inverts the
//!   verdict on a down-facing plane (fixed: local-frame areas);
//! - `merge_duplicate_edges` never flipped closed edges, silently reversing
//!   winding when two coincident circles parameterize opposite ways (fixed:
//!   tangent comparison at the shared point — parameter-frame evaluation is
//!   NOT valid for closed curves, whose domains anchor at their own
//!   reference directions);
//! - `rebuild_face_with_cb_edges` degenerated to `forward=true` for a closed
//!   rim replaced by its CommonBlock circle (fixed: same comparison);
//! - `extrude` compensated a CW-wound profile by flipping surface normals
//!   while emitting the mirrored wires as-is, so every face's wire wound
//!   against its flags — valid to the pairwise-opposition check, but the
//!   face splitter trusts effective wire winding and mints same-direction
//!   rim arcs in any later boolean (fixed: profile wires are rewound
//!   up front).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn plate_and_block(topo: &mut Topology) -> (SolidId, SolidId) {
    let plate = brepkit_operations::primitives::make_box(topo, 80.0, 80.0, 5.0).unwrap();
    brepkit_operations::transform::transform_solid(topo, plate, &Mat4::translation(0.0, 0.0, 5.0))
        .unwrap();
    let block = brepkit_operations::primitives::make_box(topo, 80.0, 80.0, 5.0).unwrap();
    (plate, block)
}

fn assert_strictly_valid(topo: &Topology, sid: SolidId, label: &str) {
    let report = brepkit_operations::validate::validate_solid(topo, sid).unwrap();
    assert!(report.is_valid(), "{label} must validate: {report:?}");
}

#[test]
fn rect_through_hole_cut_and_interface_fuse_have_valid_winding() {
    let mut topo = Topology::new();
    let (plate, block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 7.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(30.0, 30.0, 4.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "rect through-hole cut");

    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_strictly_valid(&topo, fused, "rect interface fuse");
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, fused, 0.05).unwrap();
    assert!((vol - 62000.0).abs() < 0.5, "fuse volume {vol:.3}");
}

#[test]
fn circle_through_hole_cut_and_interface_fuse_have_valid_winding() {
    let mut topo = Topology::new();
    let (plate, block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 7.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(40.0, 40.0, 4.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "circle through-hole cut");

    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_strictly_valid(&topo, fused, "circle interface fuse");
}

/// A cutter whose bottom cap is COINCIDENT with the plate's bottom plane
/// (the circle-insert cutDepth == floor configuration). The kept band's
/// original rim merges with the section circle parameterized the other way;
/// the direction map must come from tangents at the shared point, not from
/// parameter-frame evaluation (a closed circle's domain anchors at its own
/// reference direction).
#[test]
fn coincident_cap_pocket_cut_has_valid_winding() {
    let mut topo = Topology::new();
    let (plate, _block) = plate_and_block(&mut topo);
    let hole = brepkit_operations::primitives::make_cylinder(&mut topo, 10.0, 6.0).unwrap();
    brepkit_operations::transform::transform_solid(
        &mut topo,
        hole,
        &Mat4::translation(40.0, 40.0, 5.0),
    )
    .unwrap();
    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "coincident-cap pocket cut");
}

/// A CW-wound 4-arc circle profile (the way the layout tool's extruded
/// insert profiles arrive) extruded into the coincident-cap cutter of
/// `coincident_cap_pocket_cut_has_valid_winding`. Extrude must rewind the
/// profile: compensating with flipped surface normals leaves every wire
/// winding against its face flags, and the cut then mints eight
/// same-direction rim arcs (the captured circle-insert floor cut).
#[test]
fn cw_wound_extruded_profile_cut_has_valid_winding() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let (plate, block) = plate_and_block(&mut topo);

    let (cx, cy, r, z0) = (40.0, 40.0, 10.0, 5.0);
    let z = Vec3::new(0.0, 0.0, 1.0);
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7))
    };
    let v0 = v(&mut topo, cx + r, cy);
    let v1 = v(&mut topo, cx, cy + r);
    let v2 = v(&mut topo, cx - r, cy);
    let v3 = v(&mut topo, cx, cy - r);
    let arc = |topo: &mut Topology, a, b| {
        let circle = Circle3D::new(Point3::new(cx, cy, z0), z, r).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let edges = [
        arc(&mut topo, v0, v1),
        arc(&mut topo, v1, v2),
        arc(&mut topo, v2, v3),
        arc(&mut topo, v3, v0),
    ];
    // CW traversal of CCW-stored arcs: v0 -> v3 -> v2 -> v1 -> v0.
    let oes: Vec<OrientedEdge> = edges
        .iter()
        .rev()
        .map(|&e| OrientedEdge::new(e, false))
        .collect();
    let wid = topo.add_wire(Wire::new(oes, true).unwrap());
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        },
    ));
    let hole = brepkit_operations::extrude::extrude(&mut topo, fid, z, 6.0).unwrap();
    assert_strictly_valid(&topo, hole, "cw-profile extruded tool");

    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "cw-profile coincident-cap cut");

    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_strictly_valid(&topo, fused, "cw-profile interface fuse");
}

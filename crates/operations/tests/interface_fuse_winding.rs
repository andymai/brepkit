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

/// A rounded-rect plate centered at the origin with a quartered-cylinder
/// through-hole at its center, fused onto a block covering only ONE quadrant
/// of the plate: the hole lands exactly at the block's corner, so only some
/// of its rim sections cross the fused interface. Pre-#1581 the
/// vertex-coincidence promotion path spliced those hole-split sections
/// without the full weave; the pocket-mouth cells traced as disconnected
/// islands and emitted as phantom material (raw fuse free=15, wrong volume,
/// ops paying an all-planar mesh fallback that still validated clean — which
/// is why this pins the fallback count and volume sum, not just validity).
#[test]
fn partial_overlap_corner_hole_interface_fuse_is_exact() {
    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let mut topo = Topology::new();
    let z = Vec3::new(0.0, 0.0, 1.0);
    let z0 = 5.0;

    // Rounded 80x80 plate spanning z 5..10, centered at the origin.
    let (hw, r) = (40.0, 3.75);
    let c = hw - r;
    let v = |topo: &mut Topology, x: f64, y: f64| {
        topo.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7))
    };
    let corners = [
        (hw, -c),
        (hw, c),
        (c, hw),
        (-c, hw),
        (-hw, c),
        (-hw, -c),
        (-c, -hw),
        (c, -hw),
    ];
    let vs: Vec<_> = corners.iter().map(|&(x, y)| v(&mut topo, x, y)).collect();
    let arc_centers = [(c, c), (-c, c), (-c, -c), (c, -c)];
    let mut edges = Vec::new();
    for i in 0..4 {
        edges.push(topo.add_edge(Edge::new(vs[2 * i], vs[2 * i + 1], EdgeCurve::Line)));
        let (ax, ay) = arc_centers[i];
        let circle = Circle3D::new(Point3::new(ax, ay, z0), z, r).unwrap();
        edges.push(topo.add_edge(Edge::new(
            vs[2 * i + 1],
            vs[(2 * i + 2) % 8],
            EdgeCurve::Circle(circle),
        )));
    }
    let wid = topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let fid = topo.add_face(Face::new(
        wid,
        vec![],
        FaceSurface::Plane { normal: z, d: z0 },
    ));
    let plate = brepkit_operations::extrude::extrude(&mut topo, fid, z, 5.0).unwrap();

    // Quartered r=10 cylinder through-cut at the origin, bottom cap
    // coincident with the plate bottom.
    let hr = 10.0;
    let hv = [
        v(&mut topo, hr, 0.0),
        v(&mut topo, 0.0, hr),
        v(&mut topo, -hr, 0.0),
        v(&mut topo, 0.0, -hr),
    ];
    let harc = |topo: &mut Topology, a, b| {
        let circle = Circle3D::new(Point3::new(0.0, 0.0, z0), z, hr).unwrap();
        topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle)))
    };
    let hedges = [
        harc(&mut topo, hv[0], hv[1]),
        harc(&mut topo, hv[1], hv[2]),
        harc(&mut topo, hv[2], hv[3]),
        harc(&mut topo, hv[3], hv[0]),
    ];
    let hwid = topo.add_wire(
        Wire::new(
            hedges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let hfid = topo.add_face(Face::new(
        hwid,
        vec![],
        FaceSurface::Plane { normal: z, d: z0 },
    ));
    let hole = brepkit_operations::extrude::extrude(&mut topo, hfid, z, 6.0).unwrap();

    let holed = boolean::boolean(&mut topo, BooleanOp::Cut, plate, hole).unwrap();
    assert_strictly_valid(&topo, holed, "corner-hole cut");

    // Block spanning (0..80, 0..80, z 0..5): one quadrant of the plate.
    let block = brepkit_operations::primitives::make_box(&mut topo, 80.0, 80.0, 5.0).unwrap();

    let deflection = 0.05;
    let holed_vol =
        brepkit_operations::measure::oriented_solid_volume(&topo, holed, deflection).unwrap();
    let block_vol =
        brepkit_operations::measure::oriented_solid_volume(&topo, block, deflection).unwrap();

    let fallbacks_before = boolean::mesh_fallback_count();
    let fused = boolean::boolean(&mut topo, BooleanOp::Fuse, holed, block).unwrap();
    assert_eq!(
        boolean::mesh_fallback_count(),
        fallbacks_before,
        "partial-overlap corner-hole fuse must not mesh-fallback"
    );
    assert_strictly_valid(&topo, fused, "partial-overlap corner-hole fuse");

    let cylinders = brepkit_topology::explorer::solid_faces(&topo, fused)
        .unwrap()
        .iter()
        .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cylinder")
        .count();
    assert_eq!(cylinders, 8, "all quarter-cylinder hole walls must survive");

    // The operands share only the z=5 interface plane, so the fuse volume is
    // exactly their sum (compared at one deflection to cancel tessellation).
    let vol = brepkit_operations::measure::oriented_solid_volume(&topo, fused, deflection).unwrap();
    assert!(
        (vol - (holed_vol + block_vol)).abs() < 0.5,
        "fuse volume {vol:.3} vs operand sum {:.3}",
        holed_vol + block_vol
    );
}

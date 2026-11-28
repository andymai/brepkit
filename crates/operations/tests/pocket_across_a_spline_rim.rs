//! A drilled plate whose outline runs round a B-spline arch, pocketed from
//! below by a box that crosses the arch. The box cuts the plate's bottom face
//! along a U whose two legs end on the arch's B-spline edges. The bottom face
//! has a hole, so its outline reaches the splitter whole, and its B-spline
//! edges were never split where the legs end on them (only lines, circles and
//! ellipses were): the legs dangled, were dropped as pendants, and the face
//! stood whole, so the cut fell back to a mesh. The cut is exact, valid, and
//! with the intersection makes up the plate.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The plate: a 50 wide outline from y = -24 up to y = 43.35, closed above by
/// an arch rising to y = 74 (two B-spline halves meeting at its top), 4 thick,
/// drilled through below the arch.
fn drilled_arch_plate(topo: &mut Topology) -> SolidId {
    let arch = |x: f64| 43.35 + 30.65 * (1.0 - (x / 25.0).powi(2)).sqrt();
    let corners = [
        Point3::new(-25.0, -24.0, 0.0),
        Point3::new(25.0, -24.0, 0.0),
        Point3::new(25.0, 43.35, 0.0),
        Point3::new(0.0, 74.0, 0.0),
        Point3::new(-25.0, 43.35, 0.0),
    ];
    let v: Vec<_> = corners
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let half = |from: f64, to: f64| {
        let pts: Vec<Point3> = (0..=8)
            .map(|i| {
                let x = from + (to - from) * f64::from(i) / 8.0;
                Point3::new(x, arch(x), 0.0)
            })
            .collect();
        brepkit_math::nurbs::fitting::interpolate(&pts, 3).unwrap()
    };
    let edges = [
        topo.add_edge(Edge::new(v[0], v[1], EdgeCurve::Line)),
        topo.add_edge(Edge::new(v[1], v[2], EdgeCurve::Line)),
        topo.add_edge(Edge::new(
            v[2],
            v[3],
            EdgeCurve::NurbsCurve(half(25.0, 0.0)),
        )),
        topo.add_edge(Edge::new(
            v[3],
            v[4],
            EdgeCurve::NurbsCurve(half(0.0, -25.0)),
        )),
        topo.add_edge(Edge::new(v[4], v[0], EdgeCurve::Line)),
    ];
    let wire = topo.add_wire(
        Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap(),
    );
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    let plate = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
    let drill = make_cylinder(topo, 3.0, 8.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(0.0, -10.0, -2.0)).unwrap();
    boolean(topo, BooleanOp::Cut, plate, drill).unwrap()
}

/// A box over `x` in [-8, 8] from `y` = 5 past the arch's top, standing 2 into
/// the plate from below.
fn pocket(topo: &mut Topology) -> SolidId {
    let b = make_box(topo, 16.0, 80.0, 4.0).unwrap();
    transform_solid(topo, b, &Mat4::translation(-8.0, 5.0, -2.0)).unwrap();
    b
}

#[test]
fn a_pocket_across_a_spline_arch_cuts_exactly() {
    let mut topo = Topology::new();
    let plate = drilled_arch_plate(&mut topo);
    let plate_volume = solid_volume(&topo, plate, 0.001).unwrap();

    let before = mesh_fallback_count();
    let tool = pocket(&mut topo);
    let cut = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
    assert_eq!(mesh_fallback_count(), before, "cut fell back to a mesh");
    let report = validate_solid(&topo, cut).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    let plate = drilled_arch_plate(&mut topo);
    let tool = pocket(&mut topo);
    let common = boolean(&mut topo, BooleanOp::Intersect, plate, tool).unwrap();
    assert_eq!(
        mesh_fallback_count(),
        before,
        "intersection fell back to a mesh"
    );

    let cut_volume = solid_volume(&topo, cut, 0.001).unwrap();
    let common_volume = solid_volume(&topo, common, 0.001).unwrap();
    assert!(common_volume > 100.0, "pocket volume {common_volume}");
    assert!(
        (cut_volume + common_volume - plate_volume).abs() < 1e-6 * plate_volume,
        "cut {cut_volume} + pocket {common_volume} != plate {plate_volume}"
    );
}

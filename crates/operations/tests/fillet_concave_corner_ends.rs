//! A fillet of the concave edge where an arm leaves a plate's side.
//!
//! The plate is 3 thick; the arm leaves its side along one of its ends, its
//! top and bottom rising gently away from the plate's. The edge filleted is
//! the plate's vertical edge in the corner between the plate's side and the
//! arm's end face, so the fillet adds material, and each of its ends meets
//! four faces: the plate's side, the arm's end, and the plate's and the
//! arm's top (or bottom). Neither end's cap can be notched, so each end is
//! closed by a runout patch, whose edge on the plate's side lies along the
//! plate's bottom (or top) edge. The plate's side runs along that edge
//! once, on the patch's edge and then the edge's far piece, as the plate's
//! bottom (or top) does, so their meshes meet along it.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The plate's thickness, and the corner the arm leaves it at.
const T: f64 = 3.0;
const CORNER: (f64, f64) = (-100.0, 35.0);

/// A prism: the polygon `points` (in a plane through them with normal
/// `normal`, wound either way) extruded `distance` along `normal`.
fn prism(topo: &mut Topology, points: &[Point3], normal: Vec3, distance: f64) -> SolidId {
    // Wind the polygon counter-clockwise about `normal`.
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    for i in 1..points.len() - 1 {
        area += (points[i] - points[0]).cross(points[i + 1] - points[0]);
    }
    let mut points = points.to_vec();
    if area.dot(normal) < 0.0 {
        points.reverse();
    }
    let ids: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], EdgeCurve::Line);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let d = normal.dot(Vec3::new(points[0].x(), points[0].y(), points[0].z()));
    let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }));
    extrude(topo, face, normal, distance).unwrap()
}

/// The plate, `z` in `[-T, 0]`, with the arm fused along its side `x =
/// CORNER.0` for `y` past `CORNER.1`: the arm's bottom and top rise by
/// `rise` over its 25 length.
fn plate_with_arm(topo: &mut Topology, rise: f64) -> SolidId {
    let (x0, y0) = CORNER;
    let plate = prism(
        topo,
        &[
            Point3::new(x0, -35.0, -T),
            Point3::new(100.0, -35.0, -T),
            Point3::new(100.0, 59.0, -T),
            Point3::new(x0, 59.0, -T),
        ],
        Vec3::new(0.0, 0.0, 1.0),
        T,
    );
    let arm = prism(
        topo,
        &[
            Point3::new(x0, y0, -T),
            Point3::new(x0 - 25.0, y0, -T + rise),
            Point3::new(x0 - 25.0, y0, rise),
            Point3::new(x0, y0, 0.0),
        ],
        Vec3::new(0.0, 1.0, 0.0),
        59.0 - y0,
    );
    boolean(topo, BooleanOp::Fuse, plate, arm).unwrap()
}

/// The vertical edge at the corner.
fn corner_edge(topo: &Topology, solid: SolidId) -> EdgeId {
    let at_corner = |p: Point3| (p.x() - CORNER.0).abs() < 1e-9 && (p.y() - CORNER.1).abs() < 1e-9;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            at_corner(a) && at_corner(b)
        })
        .expect("the corner's vertical edge")
}

/// Fillets the corner with radius `r` and checks the result is valid,
/// watertight, and holds the corner's strip of `T r^2 (1 - pi / 4)` more,
/// to within the caps' tilt against the arm's sloping faces.
fn assert_corner_fills(rise: f64, r: f64) {
    let mut topo = Topology::new();
    let solid = plate_with_arm(&mut topo, rise);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let edge = corner_edge(&topo, solid);
    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let added = solid_volume(&topo, result.solid, 1e-3).unwrap() - before;
    let strip = T * r * r * (1.0 - PI / 4.0);
    // Each cap lies in the plate's face, the arm's face rising from it by up
    // to `rise r / 25` over the cap.
    let tilt = 2.0 * r * r * rise * r / 25.0;
    assert!(
        (added - strip).abs() <= tilt + 1e-6,
        "r = {r}: added {added}, the strip {strip}"
    );
}

#[test]
fn concave_corner_between_a_plate_and_a_flat_arm() {
    for r in [1.0, 2.0] {
        assert_corner_fills(0.0, r);
    }
}

#[test]
fn concave_corner_between_a_plate_and_a_rising_arm() {
    for r in [1.0, 2.0] {
        assert_corner_fills(1.0, r);
    }
}

/// The plate with its top a cylinder about an axis along `y`, its radius
/// `RHO`, through the plate's top edge at `x = CORNER.0`, where it leaves
/// `z = 0` falling away at `kink` radians, and the flat arm fused along its
/// side as in [`plate_with_arm`].
fn curved_plate_with_arm(topo: &mut Topology, kink: f64) -> SolidId {
    const RHO: f64 = 400.0;
    const LENGTH: f64 = 40.0;
    const Y_LOW: f64 = -35.0;
    let (x0, y0) = CORNER;
    let (cx, cz) = (RHO.mul_add(-kink.sin(), x0), -RHO * kink.cos());
    let x1 = x0 + LENGTH;
    let z1 = (RHO * RHO - (x1 - cx) * (x1 - cx)).sqrt() + cz;
    // Counter-clockwise about `+y`: up the side, along the arc, down, back.
    let corners = [(x0, -T), (x0, 0.0), (x1, z1), (x1, -T)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, z)| topo.add_vertex(Vertex::new(Point3::new(x, Y_LOW, z), 1e-7)))
        .collect();
    let arc = Circle3D::new(Point3::new(cx, Y_LOW, cz), Vec3::new(0.0, 1.0, 0.0), RHO).unwrap();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = if i == 1 {
                EdgeCurve::Circle(arc.clone())
            } else {
                EdgeCurve::Line
            };
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 1.0, 0.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: Y_LOW },
    ));
    let plate = extrude(topo, face, normal, 59.0 - Y_LOW).unwrap();
    let arm = prism(
        topo,
        &[
            Point3::new(x0, y0, -T),
            Point3::new(x0 - 25.0, y0, -T),
            Point3::new(x0 - 25.0, y0, 0.0),
            Point3::new(x0, y0, 0.0),
        ],
        Vec3::new(0.0, 1.0, 0.0),
        59.0 - y0,
    );
    boolean(topo, BooleanOp::Fuse, plate, arm).unwrap()
}

/// The fillet's top end is closed by a patch in `z = 0`, beside the plate's
/// curved top across the plate's top edge. The plate's side runs along that
/// edge once, the patch's edge then its far piece, where the patch carries
/// on the curved top: level with it, or at a slight kink.
#[test]
fn concave_corner_beside_a_curved_plate_top() {
    for kink in [0.0, 0.01] {
        for r in [1.0, 2.0] {
            let mut topo = Topology::new();
            let solid = curved_plate_with_arm(&mut topo, kink);
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let edge = corner_edge(&topo, solid);
            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            assert!(
                result.failed.is_empty(),
                "kink {kink} r {r}: {:?}",
                result.failed
            );
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "kink {kink} r {r}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "kink {kink} r {r}: mesh open");
            let added = solid_volume(&topo, result.solid, 1e-3).unwrap() - before;
            let strip = T * r * r * (1.0 - PI / 4.0);
            assert!(
                (added - strip).abs() <= 1e-9 * before,
                "kink {kink} r {r}: added {added}, the strip {strip}"
            );
        }
    }
}

//! Extruding a face whose boundary holds a circular arc past half a turn, as
//! a circle or as a rational NURBS stored either way round: the arc's wall
//! faces outward, and the solid measures its face's area times the height.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{PI, TAU};

use brepkit_check::properties::PropertiesOptions;
use brepkit_check::properties::face_integrator::integrate_face;
use brepkit_geometry::convert::circle_to_nurbs;
use brepkit_math::curves::Circle3D;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// How the arc side is stored.
#[derive(Clone, Copy)]
enum Arc {
    Circle,
    /// A rational NURBS running from the edge's start to its end.
    Nurbs,
    /// The same NURBS run from the edge's end back to its start.
    NurbsEndToStart,
}

/// The arc of the circle about `center` (radius `radius`, normal
/// `(0, 0, turn)`) counter-clockwise about that normal from `a` to `b`.
fn arc_curve(center: Point3, turn: f64, radius: f64, a: Point3, b: Point3, arc: Arc) -> EdgeCurve {
    let circle = |turn: f64| Circle3D::new(center, Vec3::new(0.0, 0.0, turn), radius).unwrap();
    let nurbs = |c: &Circle3D, from: Point3, to: Point3| {
        let t0 = c.project(from);
        let span = (c.project(to) - t0).rem_euclid(TAU);
        EdgeCurve::NurbsCurve(circle_to_nurbs(c, t0, t0 + span).unwrap())
    };
    match arc {
        Arc::Circle => EdgeCurve::Circle(circle(turn)),
        Arc::Nurbs => nurbs(&circle(turn), a, b),
        Arc::NurbsEndToStart => nurbs(&circle(-turn), b, a),
    }
}

/// A face in `z = 0` through `corners`, every side a line but the one after
/// corner `arc_after`, an arc of the circle about `center` (radius `radius`,
/// normal `(0, 0, turn)`).
fn face(
    topo: &mut Topology,
    corners: &[(f64, f64)],
    arc_after: usize,
    center: (f64, f64),
    turn: f64,
    radius: f64,
    arc: Arc,
) -> FaceId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for (i, &a) in ids.iter().enumerate() {
        let b = ids[(i + 1) % ids.len()];
        let curve = if i == arc_after {
            let center = Point3::new(center.0, center.1, 0.0);
            let (pa, pb) = (
                topo.vertex(a).unwrap().point(),
                topo.vertex(b).unwrap().point(),
            );
            arc_curve(center, turn, radius, pa, pb, arc)
        } else {
            EdgeCurve::Line
        };
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(a, b, curve)),
            true,
        ));
    }
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    topo.add_face(Face::new(wire, vec![], plane))
}

/// The angle the arc below the chord turns through (323 degrees).
fn arc_angle() -> f64 {
    let radius = 2.5_f64.sqrt();
    2.0f64.mul_add(-(0.5 / radius).asin(), TAU)
}

/// The circle about `(0, 1.5)` through `(±0.5, 3)`, and the area of its part
/// below its chord `y = 3` (the arc of 323 degrees and its chord).
fn chamber() -> (f64, f64) {
    let radius = 2.5_f64.sqrt();
    let angle = TAU - arc_angle();
    let cap = radius * radius / 2.0 * (angle - angle.sin());
    (radius, PI.mul_add(radius * radius, -cap))
}

/// Valid, watertight, measuring `area * 0.2` both by integration (to `1e-4`)
/// and by its mesh (to `1e-3`), its wall's area read to `1e-6` by the check
/// crate's integrator.
fn check(topo: &Topology, face_area: f64, solid: brepkit_topology::solid::SolidId, name: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
    let truth = face_area * 0.2;
    let volume = solid_volume(topo, solid, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-4 * truth,
        "{name}: volume {volume}, truth {truth}"
    );
    let (radius, _) = chamber();
    let wall_truth = radius * arc_angle() * 0.2;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        if !matches!(topo.face(fid).unwrap().surface(), FaceSurface::Cylinder(_)) {
            continue;
        }
        let wall = integrate_face(topo, fid, PropertiesOptions::default().gauss_order).unwrap();
        assert!(
            (wall.area - wall_truth).abs() < 1e-6 * wall_truth,
            "{name}: checked wall area {}, truth {wall_truth}",
            wall.area
        );
    }
    let meshed = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (meshed - truth).abs() < 1e-3 * truth,
        "{name}: mesh volume {meshed}, truth {truth}"
    );
}

/// A square with a keyhole notch: its top side opens into the circle's
/// chamber through a mouth 1 wide, the notch's arc running 323 degrees.
#[test]
fn a_keyhole_notch_extrudes_to_its_area() {
    for (arc, name) in [
        (Arc::Circle, "keyhole"),
        (Arc::Nurbs, "NURBS keyhole"),
        (Arc::NurbsEndToStart, "end-to-start NURBS keyhole"),
    ] {
        keyhole(arc, name);
    }
}

fn keyhole(arc: Arc, name: &str) {
    let (radius, chamber) = chamber();
    let mut topo = Topology::new();
    let corners = [
        (-3.0, -3.0),
        (3.0, -3.0),
        (3.0, 3.0),
        (0.5, 3.0),
        (-0.5, 3.0),
        (-3.0, 3.0),
    ];
    let keyhole = face(&mut topo, &corners, 3, (0.0, 1.5), -1.0, radius, arc);
    let solid = extrude(&mut topo, keyhole, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
    check(&topo, 36.0 - chamber, solid, name);
}

/// The circle's part below its chord as a face of its own: a chord and an
/// arc of 323 degrees bulging out.
#[test]
fn a_major_segment_extrudes_to_its_area() {
    for (arc, name) in [
        (Arc::Circle, "major segment"),
        (Arc::Nurbs, "NURBS major segment"),
        (Arc::NurbsEndToStart, "end-to-start NURBS major segment"),
    ] {
        let (radius, chamber) = chamber();
        let mut topo = Topology::new();
        let corners = [(0.5, 3.0), (-0.5, 3.0)];
        let segment = face(&mut topo, &corners, 1, (0.0, 1.5), 1.0, radius, arc);
        let solid = extrude(&mut topo, segment, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
        check(&topo, chamber, solid, name);
    }
}

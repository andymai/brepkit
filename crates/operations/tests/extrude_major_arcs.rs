//! Extruding a face whose boundary holds a circular arc past half a turn:
//! the arc's wall faces outward, and the solid measures its face's area times
//! the height.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

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
) -> FaceId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for (i, &a) in ids.iter().enumerate() {
        let b = ids[(i + 1) % ids.len()];
        let curve = if i == arc_after {
            let axis = Vec3::new(0.0, 0.0, turn);
            let circle = Circle3D::new(Point3::new(center.0, center.1, 0.0), axis, radius);
            EdgeCurve::Circle(circle.unwrap())
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

/// The circle about `(0, 1.5)` through `(±0.5, 3)`, and the area of its part
/// below its chord `y = 3` (the arc of 323 degrees and its chord).
fn chamber() -> (f64, f64) {
    let radius = 2.5_f64.sqrt();
    let angle = 2.0 * (0.5 / radius).asin();
    let cap = radius * radius / 2.0 * (angle - angle.sin());
    (radius, PI.mul_add(radius * radius, -cap))
}

/// Valid, watertight, and measuring `area * 0.2` both by integration (to
/// `1e-4`) and by its mesh (to `1e-3`).
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
    let keyhole = face(&mut topo, &corners, 3, (0.0, 1.5), -1.0, radius);
    let solid = extrude(&mut topo, keyhole, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
    check(&topo, 36.0 - chamber, solid, "keyhole");
}

/// The circle's part below its chord as a face of its own: a chord and an
/// arc of 323 degrees bulging out.
#[test]
fn a_major_segment_extrudes_to_its_area() {
    let (radius, chamber) = chamber();
    let mut topo = Topology::new();
    let segment = face(
        &mut topo,
        &[(0.5, 3.0), (-0.5, 3.0)],
        1,
        (0.0, 1.5),
        1.0,
        radius,
    );
    let solid = extrude(&mut topo, segment, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
    check(&topo, chamber, solid, "major segment");
}

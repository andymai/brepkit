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
use brepkit_operations::measure::{face_area, oriented_solid_volume, solid_volume};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

/// How the arc sides are stored.
#[derive(Clone, Copy)]
enum Arc {
    Circle,
    /// A rational NURBS running from the edge's start to its end.
    Nurbs,
    /// The same NURBS run from the edge's end back to its start.
    NurbsEndToStart,
}

/// The arc of the circle about `center` with normal `(0, 0, turn)`
/// counter-clockwise about that normal from `a` to `b`.
fn arc_curve(center: Point3, turn: f64, a: Point3, b: Point3, arc: Arc) -> EdgeCurve {
    let radius = (a - center).length();
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

/// A closed wire in `z = 0` through `corners`, every side a line but those
/// listed in `arcs` as `(side, center, turn)`: the side after corner `side`
/// is an arc of the circle about `center` with normal `(0, 0, turn)`.
fn wire(
    topo: &mut Topology,
    corners: &[(f64, f64)],
    arcs: &[(usize, (f64, f64), f64)],
    arc: Arc,
) -> WireId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for (i, &a) in ids.iter().enumerate() {
        let b = ids[(i + 1) % ids.len()];
        let curve = match arcs.iter().find(|s| s.0 == i) {
            Some(&(_, (cx, cy), turn)) => {
                let (pa, pb) = (
                    topo.vertex(a).unwrap().point(),
                    topo.vertex(b).unwrap().point(),
                );
                arc_curve(Point3::new(cx, cy, 0.0), turn, pa, pb, arc)
            }
            None => EdgeCurve::Line,
        };
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(a, b, curve)),
            true,
        ));
    }
    topo.add_wire(Wire::new(edges, true).unwrap())
}

/// A face in `z = 0` bounded by `outer` with `holes`.
fn face(topo: &mut Topology, outer: WireId, holes: Vec<WireId>) -> FaceId {
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    topo.add_face(Face::new(outer, holes, plane))
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

/// Mesh area of one face tessellated on its own.
fn meshed_area(topo: &Topology, face: FaceId) -> f64 {
    let mesh = tessellate(topo, face, 0.001).unwrap();
    mesh.indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            0.5 * (b - a).cross(c - a).length()
        })
        .sum()
}

/// The solid extruded 0.2 from a face of `area` whose arcs turn `wall`
/// (their radius times their angle) is valid and watertight and measures
/// `area * 0.2` by integration (to `1e-4`) and by its mesh (to `2e-3`, the
/// chords of a unit circle at deflection `0.001` holding 0.13% less); the
/// check crate's integrator reads its walls' area to `1e-6`, and each plane
/// face tessellated on its own meshes its area to `2e-3`.
fn check(topo: &Topology, area: f64, wall: f64, solid: SolidId, name: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
    let truth = area * 0.2;
    let volume = solid_volume(topo, solid, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-4 * truth,
        "{name}: volume {volume}, truth {truth}"
    );
    let meshed = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (meshed - truth).abs() < 2e-3 * truth,
        "{name}: mesh volume {meshed}, truth {truth}"
    );
    let wall_truth = wall * 0.2;
    let mut walls = 0.0;
    for fid in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        match topo.face(fid).unwrap().surface() {
            FaceSurface::Cylinder(_) => {
                let gauss = PropertiesOptions::default().gauss_order;
                walls += integrate_face(topo, fid, gauss).unwrap().area;
            }
            FaceSurface::Plane { .. } => {
                let (own, exact) = (meshed_area(topo, fid), face_area(topo, fid, 0.01).unwrap());
                assert!(
                    (own - exact).abs() < 2e-3 * exact,
                    "{name}: face meshed alone has area {own}, truth {exact}"
                );
            }
            _ => {}
        }
    }
    assert!(
        (walls - wall_truth).abs() < 1e-6 * wall_truth,
        "{name}: checked wall area {walls}, truth {wall_truth}"
    );
}

const ARCS: [(Arc, &str); 3] = [
    (Arc::Circle, "circle"),
    (Arc::Nurbs, "NURBS"),
    (Arc::NurbsEndToStart, "end-to-start NURBS"),
];

/// A square with a keyhole notch: its top side opens into the circle's
/// chamber through a mouth 1 wide, the notch's arc running 323 degrees.
#[test]
fn a_keyhole_notch_extrudes_to_its_area() {
    let (radius, chamber) = chamber();
    for (arc, kind) in ARCS {
        let mut topo = Topology::new();
        let corners = [
            (-3.0, -3.0),
            (3.0, -3.0),
            (3.0, 3.0),
            (0.5, 3.0),
            (-0.5, 3.0),
            (-3.0, 3.0),
        ];
        let outer = wire(&mut topo, &corners, &[(3, (0.0, 1.5), -1.0)], arc);
        let keyhole = face(&mut topo, outer, vec![]);
        let solid = extrude(&mut topo, keyhole, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
        let name = format!("{kind} keyhole");
        check(&topo, 36.0 - chamber, radius * arc_angle(), solid, &name);
    }
}

/// The circle's part below its chord as a face of its own: a chord and an
/// arc of 323 degrees bulging out.
#[test]
fn a_major_segment_extrudes_to_its_area() {
    let (radius, chamber) = chamber();
    for (arc, kind) in ARCS {
        let mut topo = Topology::new();
        let corners = [(0.5, 3.0), (-0.5, 3.0)];
        let outer = wire(&mut topo, &corners, &[(1, (0.0, 1.5), 1.0)], arc);
        let segment = face(&mut topo, outer, vec![]);
        let solid = extrude(&mut topo, segment, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
        let name = format!("{kind} major segment");
        check(&topo, chamber, radius * arc_angle(), solid, &name);
    }
}

/// A square whose top side opens into a half-disc notch: the arc turns
/// exactly half a turn, where the radius at its start runs along its chord.
#[test]
fn a_half_turn_notch_extrudes_to_its_area() {
    for (arc, kind) in ARCS {
        let mut topo = Topology::new();
        let corners = [
            (-2.0, -2.0),
            (2.0, -2.0),
            (2.0, 2.0),
            (1.0, 2.0),
            (-1.0, 2.0),
            (-2.0, 2.0),
        ];
        let outer = wire(&mut topo, &corners, &[(3, (0.0, 2.0), -1.0)], arc);
        let notched = face(&mut topo, outer, vec![]);
        let solid = extrude(&mut topo, notched, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
        let name = format!("{kind} half-turn notch");
        check(&topo, 16.0 - PI / 2.0, PI, solid, &name);
    }
}

/// A square plate with a hole shaped as the major segment, its wire run
/// either way round.
#[test]
fn a_major_segment_hole_extrudes_to_its_area() {
    let (radius, chamber) = chamber();
    for (arc, kind) in ARCS {
        for (corners, turn, way) in [
            ([(0.5, 3.0), (-0.5, 3.0)], 1.0, "counter-clockwise"),
            ([(-0.5, 3.0), (0.5, 3.0)], -1.0, "clockwise"),
        ] {
            let mut topo = Topology::new();
            let square = [(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)];
            let outer = wire(&mut topo, &square, &[], arc);
            let hole = wire(&mut topo, &corners, &[(1, (0.0, 1.5), turn)], arc);
            let plate = face(&mut topo, outer, vec![hole]);
            let solid = extrude(&mut topo, plate, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
            let name = format!("{kind} {way} major segment hole");
            check(&topo, 100.0 - chamber, radius * arc_angle(), solid, &name);
        }
    }
}

/// A unit disc bounded by an arc of 300 degrees and one of 60.
#[test]
fn a_disc_of_two_arcs_extrudes_to_its_area() {
    for (arc, kind) in ARCS {
        let mut topo = Topology::new();
        let corners = [(1.0, 0.0), (0.5, -(0.75_f64.sqrt()))];
        let outer = wire(
            &mut topo,
            &corners,
            &[(0, (0.0, 0.0), 1.0), (1, (0.0, 0.0), 1.0)],
            arc,
        );
        let disc = face(&mut topo, outer, vec![]);
        let solid = extrude(&mut topo, disc, Vec3::new(0.0, 0.0, 1.0), 0.2).unwrap();
        let name = format!("{kind} two-arc disc");
        check(&topo, PI, TAU, solid, &name);
    }
}

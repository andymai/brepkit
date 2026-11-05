//! IGES export writes each face as a trimmed surface (144) over its plane (108)
//! or B-spline surface (128), each boundary as a curve on that surface (142)
//! holding a composite (102) of lines (110) and B-spline curves (126). Import
//! reads them back and sews the faces by position, so a solid survives the
//! round trip.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeSet;
use std::f64::consts::PI;

use brepkit_io::iges::reader::read_iges;
use brepkit_io::iges::writer::write_iges;
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::{oriented_solid_volume, solid_surface_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// Writes `solids` and reads them back into a fresh arena.
fn round_trip(topo: &Topology, solids: &[SolidId]) -> (Topology, Vec<SolidId>) {
    let text = write_iges(topo, solids).unwrap();
    let mut read = Topology::new();
    let back = read_iges(&text, &mut read).unwrap();
    (read, back)
}

/// The entity types a file's directory section lists.
fn entity_types(text: &str) -> BTreeSet<u32> {
    text.lines()
        .filter(|line| line.as_bytes().get(72) == Some(&b'D'))
        .map(|line| line[..8].trim().parse().unwrap())
        .collect()
}

fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, tolerance: f64, what: &str) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let measured = solid_volume(topo, solid, 0.001).unwrap();
    assert!(
        (measured - volume).abs() < tolerance,
        "{what}: volume {measured}, expected {volume}"
    );
    let oriented = oriented_solid_volume(topo, solid, 0.001).unwrap();
    assert!(oriented > 0.0, "{what}: reads inside out, {oriented}");
    assert!(
        is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
}

/// The issue's box: 24000 and 5200, not 30 and 6.
#[test]
fn a_box_survives_the_round_trip() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 40.0, 30.0, 20.0).unwrap();

    let text = write_iges(&topo, &[solid]).unwrap();
    let types = entity_types(&text);
    for wanted in [102, 108, 110, 142, 144] {
        assert!(types.contains(&wanted), "no entity {wanted} in {types:?}");
    }

    let mut read = Topology::new();
    let back = read_iges(&text, &mut read).unwrap();

    assert_eq!(back.len(), 1);
    assert_sound(&read, back[0], 24000.0, 1e-9, "box");
    assert!((solid_surface_area(&read, back[0], 0.01).unwrap() - 5200.0).abs() < 1e-9);
    assert_eq!(solid_faces(&read, back[0]).unwrap().len(), 6);
}

/// Coordinates keep every digit: a sixth-decimal format loses a third.
#[test]
fn a_box_keeps_its_exact_coordinates() {
    let mut topo = Topology::new();
    let side = 1.0 / 3.0;
    let solid = make_box(&mut topo, side, side, side).unwrap();

    let (read, back) = round_trip(&topo, &[solid]);

    assert_sound(&read, back[0], side * side * side, 1e-15, "a third");
}

/// A turned box, whose plane coefficients and corners are all long fractions.
#[test]
fn a_turned_box_stays_closed() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
    let turn = Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3);
    transform_solid(&mut topo, solid, &turn).unwrap();

    let (read, back) = round_trip(&topo, &[solid]);

    assert_sound(&read, back[0], 24.0, 1e-9, "turned box");
}

/// A plate with a square hole: the cap's inner wire comes back as its hole, and
/// the boolean's reversed hole walls come back facing into the hole.
#[test]
fn a_hole_survives_the_round_trip() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 4.0).unwrap();
    let tool = make_box(&mut topo, 4.0, 4.0, 8.0).unwrap();
    transform_solid(&mut topo, tool, &Mat4::translation(8.0, 8.0, -2.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();

    let (read, back) = round_trip(&topo, &[drilled]);

    assert_eq!(back.len(), 1);
    assert_sound(&read, back[0], 1600.0 - 64.0, 1e-9, "drilled plate");
}

/// A closed cavity comes back as an inner shell of the one solid.
#[test]
fn a_cavity_survives_the_round_trip() {
    let mut topo = Topology::new();
    let outer = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let core = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
    transform_solid(&mut topo, core, &Mat4::translation(3.0, 3.0, 3.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, outer, core).unwrap();
    assert_eq!(topo.solid(hollow).unwrap().inner_shells().len(), 1);

    let (read, back) = round_trip(&topo, &[hollow]);

    assert_eq!(back.len(), 1, "the cavity is no solid of its own");
    assert_eq!(read.solid(back[0]).unwrap().inner_shells().len(), 1);
    assert_sound(&read, back[0], 1000.0 - 64.0, 1e-9, "hollow box");
}

/// Solids apart from each other come back as separate solids.
#[test]
fn separate_solids_stay_separate() {
    let mut topo = Topology::new();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(5.0, 0.0, 0.0)).unwrap();

    let (read, back) = round_trip(&topo, &[a, b]);

    assert_eq!(back.len(), 2);
    let mut volumes: Vec<f64> = back
        .iter()
        .map(|&s| solid_volume(&read, s, 0.01).unwrap())
        .collect();
    volumes.sort_by(f64::total_cmp);
    assert!((volumes[0] - 1.0).abs() < 1e-9 && (volumes[1] - 8.0).abs() < 1e-9);
}

/// An obround prism: arcs on flat caps, and walls that are planes and half
/// cylinders. Arcs go out exactly, as rational curves, not as 32 chords.
#[test]
fn arcs_survive_exactly() {
    let mut topo = Topology::new();
    let at = |topo: &mut Topology, x, y| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7));
    let (a, b, c, d) = (
        at(&mut topo, 0.0, -8.0),
        at(&mut topo, 40.0, -8.0),
        at(&mut topo, 40.0, 8.0),
        at(&mut topo, 0.0, 8.0),
    );
    let arc = |cx: f64| {
        EdgeCurve::Circle(
            Circle3D::new(Point3::new(cx, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 8.0).unwrap(),
        )
    };
    let edges = [
        (a, b, EdgeCurve::Line),
        (b, c, arc(40.0)),
        (c, d, EdgeCurve::Line),
        (d, a, arc(0.0)),
    ]
    .into_iter()
    .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true))
    .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let plane = FaceSurface::Plane {
        normal: Vec3::new(0.0, 0.0, 1.0),
        d: 0.0,
    };
    let face = topo.add_face(Face::new(wire, Vec::new(), plane));
    let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 10.0).unwrap();

    let (read, back) = round_trip(&topo, &[solid]);

    assert_eq!(back.len(), 1);
    // Measured off a mesh of its B-spline walls, the volume reads to the chords.
    assert_sound(
        &read,
        back[0],
        (640.0 + 64.0 * PI) * 10.0,
        1.0,
        "obround prism",
    );
    // Every arc came back on its own circle, not as chords of it.
    let mut arcs = 0;
    for edge in solid_edges(&read, back[0]).unwrap() {
        let data = read.edge(edge).unwrap();
        let EdgeCurve::NurbsCurve(curve) = data.curve() else {
            continue;
        };
        arcs += 1;
        let (lo, hi) = curve.domain();
        for k in 0..=32 {
            let p = curve.evaluate(lo + (hi - lo) * f64::from(k) / 32.0);
            let reach = |cx: f64| (p.x() - cx).hypot(p.y());
            assert!(
                (reach(0.0) - 8.0).abs().min((reach(40.0) - 8.0).abs()) < 1e-12,
                "{p:?} is off both circles"
            );
        }
    }
    assert_eq!(arcs, 4, "two arcs on each cap");
}

/// Curved faces go out as exact rational B-spline surfaces: closed by seams, at
/// poles and at an apex.
#[test]
fn curved_solids_survive_the_round_trip() {
    let mut topo = Topology::new();
    let cases: Vec<(&str, SolidId, f64)> = vec![
        (
            "cylinder",
            make_cylinder(&mut topo, 5.0, 10.0).unwrap(),
            PI * 25.0 * 10.0,
        ),
        (
            "frustum",
            make_cone(&mut topo, 5.0, 2.0, 6.0).unwrap(),
            PI * 6.0 * (25.0 + 10.0 + 4.0) / 3.0,
        ),
        (
            "ball",
            make_sphere(&mut topo, 5.0, 32).unwrap(),
            4.0 / 3.0 * PI * 125.0,
        ),
        (
            "ring",
            make_torus(&mut topo, 4.0, 1.5, 32).unwrap(),
            2.0 * PI * PI * 4.0 * 1.5 * 1.5,
        ),
    ];
    for (name, solid, volume) in cases {
        let text = write_iges(&topo, &[solid]).unwrap();
        assert!(
            entity_types(&text).contains(&128),
            "{name}: curved faces go out as entity 128"
        );

        let mut read = Topology::new();
        let back = read_iges(&text, &mut read).unwrap();

        assert_eq!(back.len(), 1, "{name}");
        let measured = solid_volume(&read, back[0], 0.001).unwrap();
        assert!(
            (measured - volume).abs() < 2e-3 * volume,
            "{name}: volume {measured}, expected {volume}"
        );
        assert!(
            is_watertight(&tessellate_solid(&read, back[0], 0.01).unwrap()),
            "{name}: its mesh is open"
        );
    }
}

/// A boolean's reversed curved wall: a block less a cylinder keeps its bore
/// facing into the bore, in a B-spline surface with no flag to say so.
#[test]
fn a_reversed_curved_wall_faces_the_same_way() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
    let rod = make_cylinder(&mut topo, 3.0, 14.0).unwrap();
    transform_solid(&mut topo, rod, &Mat4::translation(10.0, 10.0, -2.0)).unwrap();
    let drilled = boolean(&mut topo, BooleanOp::Cut, block, rod).unwrap();

    let (read, back) = round_trip(&topo, &[drilled]);

    assert_eq!(back.len(), 1);
    let volume = 4000.0 - PI * 9.0 * 10.0;
    let measured = oriented_solid_volume(&read, back[0], 0.001).unwrap();
    assert!(
        (measured - volume).abs() < 2e-3 * volume,
        "volume {measured}, expected {volume}"
    );
    assert!(is_watertight(
        &tessellate_solid(&read, back[0], 0.01).unwrap()
    ));
}

/// A plane with no boundary in a file bounds nothing: it is left out rather
/// than turned into a unit square at the plane's origin.
#[test]
fn an_unbounded_plane_is_no_face() {
    let text = [
        format!("{:<72}S{:>7}", "brepkit test", 1),
        format!(
            "{:<72}G{:>7}",
            "1H,,1H;,7Hbrepkit,5Ha.igs,7Hbrepkit,7Hbrepkit,32,38,6,38,15;", 1
        ),
        format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            108, 1, 0, 0, 0, 0, 0, 0, 0, 1
        ),
        format!(
            "{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            108, 0, 0, 1, 0, " ", " ", " ", 0, 2
        ),
        format!("{:<64}{:>8}P{:>7}", "108,0.,0.,1.,5.,0,0.,0.,0.;", 1, 1),
    ]
    .join("\n");

    let mut topo = Topology::new();
    assert!(read_iges(&text, &mut topo).unwrap().is_empty());
}

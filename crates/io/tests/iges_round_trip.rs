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

/// An IGES file from entity lines: `(type, parameters)`, numbered in order.
fn iges_file(entities: &[(u32, String)]) -> String {
    let mut directory = Vec::new();
    let mut parameters = Vec::new();
    for (i, (kind, params)) in entities.iter().enumerate() {
        let de = u32::try_from(2 * i + 1).unwrap();
        let mut count = 0_u32;
        let first = parameters.len() + 1;
        let mut line = String::new();
        for token in params.split_inclusive([',', ';']) {
            if line.len() + token.len() > 64 {
                parameters.push(format!("{line:<64}{de:>8}P{:>7}", parameters.len() + 1));
                count += 1;
                line.clear();
            }
            line.push_str(token);
        }
        parameters.push(format!("{line:<64}{de:>8}P{:>7}", parameters.len() + 1));
        count += 1;
        directory.push(format!(
            "{kind:>8}{first:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{de:>7}",
            0, 0, 0, 0, 0, 0, 0
        ));
        directory.push(format!(
            "{kind:>8}{:>8}{:>8}{count:>8}{:>8}{:>8}{:>8}{:>8}{:>8}D{:>7}",
            0,
            0,
            0,
            " ",
            " ",
            " ",
            0,
            de + 1
        ));
    }
    let mut lines = vec![format!("{:<72}S{:>7}", "brepkit test", 1)];
    lines.extend(directory);
    lines.extend(parameters);
    lines.join("\n")
}

/// A malformed file costs its bad entities, not the process: a composite holding
/// itself, two holding each other, and counts past anything the file could hold.
#[test]
fn a_malformed_file_is_read_safely() {
    let plane = (108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string());
    let bad = [
        vec![
            plane.clone(),
            (102, "102,1,3;".to_string()),
            (144, "144,1,1,0,3;".to_string()),
        ],
        vec![
            plane.clone(),
            (102, "102,1,5;".to_string()),
            (102, "102,1,3;".to_string()),
            (144, "144,1,1,0,3;".to_string()),
        ],
        vec![plane.clone(), (144, "144,1,1,1E30,1;".to_string())],
        vec![plane.clone(), (144, "144,1,1,-3,1;".to_string())],
        vec![plane.clone(), (144, "144,1,1,2.5,1;".to_string())],
        // A lone spline is read by nothing, so each is reached through a face.
        vec![
            (128, "128,1E30,1E30,1,1,0,0,0,0,0;".to_string()),
            (144, "144,1,0,0,0;".to_string()),
        ],
        vec![
            plane,
            (126, "126,1E30,2,0,0,0,0;".to_string()),
            (144, "144,1,1,0,3;".to_string()),
        ],
    ];
    for entities in &bad {
        let mut topo = Topology::new();
        let solids = read_iges(&iges_file(entities), &mut topo).unwrap();
        assert!(solids.is_empty(), "{entities:?}");
    }
}

/// A spline evaluation cannot walk costs its face, not the process: a degree past
/// its points runs off the knots, and knots that fall can search forever.
#[test]
fn a_spline_evaluation_cannot_walk_is_refused() {
    let plane = (108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string());
    let curve_in_a_face = |params: &str| {
        vec![
            plane.clone(),
            (126, params.to_string()),
            (144, "144,1,1,0,3;".to_string()),
        ]
    };
    let surface_as_a_face =
        |params: &str| vec![(128, params.to_string()), (144, "144,1,0,0,0;".to_string())];
    let corners = "0.,0.,0.,1.,0.,0.,0.,1.,0.,1.,1.,0.";
    let bad = [
        curve_in_a_face("126,1,5,0,0,0,0,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,0.,0.,0.,1.,0.,0.,0.,1.;"),
        curve_in_a_face("126,1,1,0,0,0,0,1.,0.5,0.,-1.,1.,1.,0.,0.,0.,1.,0.,0.,0.,1.;"),
        surface_as_a_face(&format!(
            "128,1,1,5,5,0,0,0,0,0,0.,0.,0.,0.,1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.,\
             1.,1.,1.,1.,{corners},0.,1.,0.,1.;"
        )),
        surface_as_a_face(&format!(
            "128,1,1,1,1,0,0,0,0,0,1.,0.5,0.,-1.,1.,0.5,0.,-1.,\
             1.,1.,1.,1.,{corners},0.,1.,0.,1.;"
        )),
    ];
    for entities in &bad {
        let mut topo = Topology::new();
        let solids = read_iges(&iges_file(entities), &mut topo).unwrap();
        assert!(solids.is_empty(), "{entities:?}");
    }
}

/// A composite nested far past any real file costs its face, not the stack.
#[test]
fn a_long_chain_of_composites_is_refused() {
    let depth = 50_000_usize;
    let mut entities = vec![(108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string())];
    for level in 0..depth {
        entities.push((102, format!("102,1,{};", 2 * level + 5)));
    }
    entities.push((102, "102,0;".to_string()));
    entities.push((144, "144,1,1,0,3;".to_string()));

    let mut topo = Topology::new();
    assert!(
        read_iges(&iges_file(&entities), &mut topo)
            .unwrap()
            .is_empty()
    );
}

/// Forty composites, each holding the next one twice, would ask for 2^40 visits
/// from a file of a few dozen lines. No cycle is involved, and nothing is emitted.
#[test]
fn composites_cannot_multiply_the_work_of_reading_them() {
    let levels = 40_usize;
    let mut entities = vec![(108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string())];
    for level in 0..levels {
        let next = 2 * level + 5;
        entities.push((102, format!("102,2,{next},{next};")));
    }
    entities.push((102, "102,0;".to_string()));
    entities.push((144, "144,1,1,0,3;".to_string()));

    let mut topo = Topology::new();
    assert!(
        read_iges(&iges_file(&entities), &mut topo)
            .unwrap()
            .is_empty()
    );
}

/// One solid named twice is one solid: its faces written twice would read back
/// with every edge on four faces.
#[test]
fn a_solid_named_twice_is_written_once() {
    let mut topo = Topology::new();
    let solid = make_box(&mut topo, 10.0, 20.0, 30.0).unwrap();

    let once = write_iges(&topo, &[solid]).unwrap();
    let twice = write_iges(&topo, &[solid, solid]).unwrap();

    assert_eq!(once, twice);
}

/// Two closed circles sharing a start and a far point, in planes at right angles,
/// are two edges: those points alone would sew them into one.
#[test]
fn closed_curves_through_the_same_points_stay_apart() {
    let root = std::f64::consts::FRAC_1_SQRT_2;
    let circle = |flat: [(f64, f64); 9], y: bool| {
        let weights: Vec<String> = (0..9)
            .map(|k| {
                if k % 2 == 0 {
                    "1.".into()
                } else {
                    format!("{root}")
                }
            })
            .collect();
        let points: Vec<String> = flat
            .iter()
            .map(|&(a, b)| {
                if y {
                    format!("{a},{b},0.")
                } else {
                    format!("{a},0.,{b}")
                }
            })
            .collect();
        format!(
            "126,8,2,0,0,0,0,0.,0.,0.,1.,1.,2.,2.,3.,3.,4.,4.,4.,{},{},0.,4.,0.,0.,0.;",
            weights.join(","),
            points.join(",")
        )
    };
    let flat = [
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (-1.0, 1.0),
        (-1.0, 0.0),
        (-1.0, -1.0),
        (0.0, -1.0),
        (1.0, -1.0),
        (1.0, 0.0),
    ];
    let entities = vec![
        (126, circle(flat, true)),
        (126, circle(flat, false)),
        (108, "108,0.,0.,1.,0.,1,0.,0.,0.;".to_string()),
        (108, "108,0.,1.,0.,0.,3,0.,0.,0.;".to_string()),
    ];

    let mut topo = Topology::new();
    let solids = read_iges(&iges_file(&entities), &mut topo).unwrap();

    assert_eq!(solids.len(), 2, "each disc is its own, sharing no edge");
}

/// A hollow box standing in the void of a larger hollow box: each keeps its own
/// cavity, whichever the file lists first.
#[test]
fn nested_hollow_solids_keep_their_own_cavities() {
    let mut topo = Topology::new();
    let hollow = |topo: &mut Topology, side: f64, wall: f64, at: f64| {
        let outer = make_box(topo, side, side, side).unwrap();
        let core = make_box(
            topo,
            side - 2.0 * wall,
            side - 2.0 * wall,
            side - 2.0 * wall,
        )
        .unwrap();
        transform_solid(topo, core, &Mat4::translation(wall, wall, wall)).unwrap();
        let shell = boolean(topo, BooleanOp::Cut, outer, core).unwrap();
        transform_solid(topo, shell, &Mat4::translation(at, at, at)).unwrap();
        shell
    };
    let large = hollow(&mut topo, 20.0, 2.0, 0.0);
    let small = hollow(&mut topo, 6.0, 1.0, 7.0);
    let volume = |side: f64, wall: f64| side.powi(3) - (side - 2.0 * wall).powi(3);

    for order in [[large, small], [small, large]] {
        let (read, back) = round_trip(&topo, &order);

        assert_eq!(back.len(), 2);
        let mut sizes: Vec<(f64, usize)> = back
            .iter()
            .map(|&s| {
                (
                    solid_volume(&read, s, 0.01).unwrap(),
                    read.solid(s).unwrap().inner_shells().len(),
                )
            })
            .collect();
        sizes.sort_by(|p, q| p.0.total_cmp(&q.0));
        assert!((sizes[0].0 - volume(6.0, 1.0)).abs() < 1e-9, "{sizes:?}");
        assert!((sizes[1].0 - volume(20.0, 2.0)).abs() < 1e-9, "{sizes:?}");
        assert_eq!((sizes[0].1, sizes[1].1), (1, 1), "{sizes:?}");
    }
}

/// A whole ring has no boundary to write: it takes its surface's own, and reads
/// back with its two seams each used twice, not one edge used four times.
#[test]
fn a_whole_ring_takes_its_surfaces_own_boundary() {
    let mut topo = Topology::new();
    let solid = make_torus(&mut topo, 4.0, 1.5, 32).unwrap();

    let (read, back) = round_trip(&topo, &[solid]);

    assert_eq!(back.len(), 1);
    let faces = solid_faces(&read, back[0]).unwrap();
    assert_eq!(faces.len(), 1);
    let wire = read
        .wire(read.face(faces[0]).unwrap().outer_wire())
        .unwrap();
    let edges: BTreeSet<_> = wire.edges().iter().map(|oe| oe.edge().index()).collect();
    assert_eq!((wire.edges().len(), edges.len()), (4, 2));
}

/// A flat face over `corners`, in the order they run, appended to `entities`:
/// its edges as lines in a composite, over the plane they lie in. The corners
/// wind counter-clockwise about the face's outward normal, `(a, b, c)`, and
/// `d` is the plane's offset along it.
fn push_quad(entities: &mut Vec<(u32, String)>, corners: [[f64; 3]; 4], normal: [f64; 3], d: f64) {
    let de = |index: usize| 2 * index + 1;
    let first = entities.len();
    for (i, a) in corners.iter().enumerate() {
        let b = corners[(i + 1) % 4];
        entities.push((
            110,
            format!(
                "110,{:?},{:?},{:?},{:?},{:?},{:?};",
                a[0], a[1], a[2], b[0], b[1], b[2]
            ),
        ));
    }
    let lines: Vec<String> = (first..first + 4).map(|i| de(i).to_string()).collect();
    let composite = entities.len();
    entities.push((102, format!("102,4,{};", lines.join(","))));
    let plane = entities.len();
    entities.push((
        108,
        format!(
            "108,{:?},{:?},{:?},{d:?},0,0.,0.,0.;",
            normal[0], normal[1], normal[2]
        ),
    ));
    entities.push((144, format!("144,{},1,0,{};", de(plane), de(composite))));
}

/// The six faces of the box from `-half` to `half`, facing out.
fn push_box(entities: &mut Vec<(u32, String)>, half: f64) {
    let h = half;
    push_quad(
        entities,
        [[-h, -h, h], [h, -h, h], [h, h, h], [-h, h, h]],
        [0.0, 0.0, 1.0],
        h,
    );
    push_quad(
        entities,
        [[-h, h, -h], [h, h, -h], [h, -h, -h], [-h, -h, -h]],
        [0.0, 0.0, -1.0],
        h,
    );
    push_quad(
        entities,
        [[h, -h, -h], [h, h, -h], [h, h, h], [h, -h, h]],
        [1.0, 0.0, 0.0],
        h,
    );
    push_quad(
        entities,
        [[-h, -h, h], [-h, h, h], [-h, h, -h], [-h, -h, -h]],
        [-1.0, 0.0, 0.0],
        h,
    );
    push_quad(
        entities,
        [[h, h, -h], [-h, h, -h], [-h, h, h], [h, h, h]],
        [0.0, 1.0, 0.0],
        h,
    );
    push_quad(
        entities,
        [[-h, -h, -h], [h, -h, -h], [h, -h, h], [-h, -h, h]],
        [0.0, -1.0, 0.0],
        h,
    );
}

/// A closed polyline as a degree 1 curve through the square's corners, read
/// over `from..to`, and the face it bounds in `z = 0`.
fn closed_square_face(points: &[[f64; 3]], from: f64, to: f64) -> Vec<(u32, String)> {
    let n = points.len();
    let mut knots = vec![0.0_f64];
    knots.extend((0..n).map(|i| i as f64));
    knots.push((n - 1) as f64);
    let list = |values: Vec<f64>| {
        values
            .iter()
            .map(|v| format!("{v:?}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let weights = vec![1.0; n];
    let coordinates: Vec<f64> = points.iter().flatten().copied().collect();
    vec![
        (108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string()),
        (
            126,
            format!(
                "126,{},1,0,0,0,0,{},{},{},{from:?},{to:?},0.,0.,1.;",
                n - 1,
                list(knots),
                list(weights),
                list(coordinates)
            ),
        ),
        (144, "144,1,1,0,3;".to_string()),
    ]
}

/// The point `fraction` of the way along a face's first edge, the way the
/// face's wire runs it.
fn along_first_edge(topo: &Topology, solid: SolidId, fraction: f64) -> Point3 {
    let shell = topo
        .shell(topo.solid(solid).unwrap().outer_shell())
        .unwrap();
    let face = topo.face(shell.faces()[0]).unwrap();
    let oriented = topo.wire(face.outer_wire()).unwrap().edges()[0];
    let edge = topo.edge(oriented.edge()).unwrap();
    let (a, b) = (
        topo.vertex(edge.start()).unwrap().point(),
        topo.vertex(edge.end()).unwrap().point(),
    );
    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
    let f = if oriented.is_forward() {
        fraction
    } else {
        1.0 - fraction
    };
    edge.curve()
        .evaluate_with_endpoints(t0 + (t1 - t0) * f, a, b)
}

/// A closed curve keeps the span a file reads it over, and the way it runs:
/// both are lost when the edge stores the whole curve, which reads forward.
#[test]
fn a_closed_curve_keeps_its_span_and_direction() {
    let square = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
    ];
    for (name, points, from, to, quarter) in [
        ("forward", square.to_vec(), 0.0, 4.0, (1.0, 0.0)),
        ("backward", square.to_vec(), 4.0, 0.0, (0.0, 1.0)),
        (
            "a tail past its closing point",
            [&square[..], &[[5.0, 5.0, 0.0]]].concat(),
            0.0,
            4.0,
            (1.0, 0.0),
        ),
        (
            "a tail, run backward",
            [&square[..], &[[5.0, 5.0, 0.0]]].concat(),
            4.0,
            0.0,
            (0.0, 1.0),
        ),
    ] {
        let mut topo = Topology::new();
        let solids = read_iges(
            &iges_file(&closed_square_face(&points, from, to)),
            &mut topo,
        )
        .unwrap();
        assert_eq!(solids.len(), 1, "{name}");
        let at = along_first_edge(&topo, solids[0], 0.25);
        assert!(
            (at - Point3::new(quarter.0, quarter.1, 0.0)).length() < 1e-9,
            "{name}: a quarter along reads {at:?}"
        );
    }
}

/// A line and an S across it share their ends and their middle, and are not
/// one edge: the faces on them are separate solids.
#[test]
fn a_line_and_an_s_across_it_stay_apart() {
    let mut entities = Vec::new();
    push_quad(
        &mut entities,
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [0.0, 0.0, 1.0],
        0.0,
    );
    let curve = entities.len();
    entities.push((
        126,
        "126,3,3,0,0,1,0,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,\
         0.,0.,0.,1.,1.,0.,2.,-1.,0.,3.,0.,0.,0.,1.,0.,0.,1.;"
            .to_string(),
    ));
    let mut side = |a: [f64; 3], b: [f64; 3]| {
        entities.push((
            110,
            format!(
                "110,{:?},{:?},{:?},{:?},{:?},{:?};",
                a[0], a[1], a[2], b[0], b[1], b[2]
            ),
        ));
        2 * (entities.len() - 1) + 1
    };
    let down = side([3.0, 0.0, 0.0], [3.0, -5.0, 0.0]);
    let across = side([3.0, -5.0, 0.0], [0.0, -5.0, 0.0]);
    let up = side([0.0, -5.0, 0.0], [0.0, 0.0, 0.0]);
    let composite = entities.len();
    entities.push((
        102,
        format!("102,4,{},{down},{across},{up};", 2 * curve + 1),
    ));
    let plane = entities.len();
    entities.push((108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string()));
    entities.push((
        144,
        format!("144,{},1,0,{};", 2 * plane + 1, 2 * composite + 1),
    ));

    let mut topo = Topology::new();
    let solids = read_iges(&iges_file(&entities), &mut topo).unwrap();
    assert_eq!(solids.len(), 2);
}

/// An open sheet facing down inside a box reads a negative signed volume, but
/// it has no inside: it stays its own solid, and not a cavity of the box.
#[test]
fn an_open_sheet_is_no_cavity() {
    let mut entities = Vec::new();
    push_box(&mut entities, 2.0);
    push_quad(
        &mut entities,
        [
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, -1.0, 1.0],
        ],
        [0.0, 0.0, -1.0],
        -1.0,
    );
    let mut topo = Topology::new();
    let solids = read_iges(&iges_file(&entities), &mut topo).unwrap();
    assert_eq!(solids.len(), 2);
    for &solid in &solids {
        assert!(topo.solid(solid).unwrap().inner_shells().is_empty());
    }
}

/// A hollow column through a ring's hole: its cavity lies in the ring's box,
/// the smaller of the two, but not in the ring.
#[test]
fn a_cavity_in_a_rings_box_stays_with_its_own_solid() {
    let mut topo = Topology::new();
    let ring = make_torus(&mut topo, 4.0, 1.5, 16).unwrap();
    let solid = make_box(&mut topo, 2.0, 2.0, 1000.0).unwrap();
    transform_solid(&mut topo, solid, &Mat4::translation(-1.0, -1.0, -500.0)).unwrap();
    let void = make_box(&mut topo, 0.5, 0.5, 0.5).unwrap();
    transform_solid(&mut topo, void, &Mat4::translation(-0.25, -0.25, -0.25)).unwrap();
    let column = boolean(&mut topo, BooleanOp::Cut, solid, void).unwrap();
    assert_eq!(topo.solid(column).unwrap().inner_shells().len(), 1);

    let (read, solids) = round_trip(&topo, &[column, ring]);
    assert_eq!(solids.len(), 2);
    for &solid in &solids {
        let cavities = read.solid(solid).unwrap().inner_shells().len();
        let bounds = brepkit_operations::measure::solid_bounding_box(&read, solid).unwrap();
        let is_column = bounds.max.z() > 100.0;
        assert_eq!(cavities, usize::from(is_column), "column: {is_column}");
    }
}

/// Coordinates far past what a vertex's cell can count saturate it: the cells
/// around it must not overflow.
#[test]
fn a_huge_coordinate_is_read_safely() {
    let mut entities = Vec::new();
    push_quad(
        &mut entities,
        [
            [1e13, 1e13, 0.0],
            [1e300, 1e13, 0.0],
            [1e300, 1e300, 0.0],
            [-1e300, 1e300, 0.0],
        ],
        [0.0, 0.0, 1.0],
        0.0,
    );
    let mut topo = Topology::new();
    assert!(read_iges(&iges_file(&entities), &mut topo).is_ok());
}

/// A boundary that does not close bounds nothing: an open line, and two lines
/// that do not meet, leave no face.
#[test]
fn a_boundary_that_does_not_close_is_no_face() {
    let line = |a: f64, b: f64| format!("110,{a:?},0.,0.,{b:?},0.,0.;");
    for lines in [vec![line(0.0, 1.0)], vec![line(0.0, 1.0), line(5.0, 6.0)]] {
        let mut entities = vec![(108, "108,0.,0.,1.,0.,0,0.,0.,0.;".to_string())];
        let first = entities.len();
        let count = lines.len();
        entities.extend(lines.into_iter().map(|l| (110, l)));
        let pointers: Vec<String> = (first..first + count)
            .map(|i| (2 * i + 1).to_string())
            .collect();
        let composite = entities.len();
        entities.push((102, format!("102,{count},{};", pointers.join(","))));
        entities.push((144, format!("144,1,1,0,{};", 2 * composite + 1)));
        let mut topo = Topology::new();
        assert!(
            read_iges(&iges_file(&entities), &mut topo)
                .unwrap()
                .is_empty(),
            "{count} lines"
        );
    }
}

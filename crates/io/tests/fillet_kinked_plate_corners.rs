//! A plate 80 thick whose outline is two B-spline sides meeting at two
//! shallow kinks (4.4 and 17.4 degrees), its side walls split at mid-height
//! (a part fused from two halves). All its rims and its four vertical kink
//! edges are rounded at r = 30. (The file's plate has two bores through it;
//! they are left out, the top and bottom taking their outlines alone.)
//!
//! At each corner where a kink meets a rim, three walked rounds of one
//! radius meet on three faces. They are set back to the one ball touching
//! all three, solved on the faces, and the corner is a patch of its sphere;
//! left running on to the corner, the rounds' ends crossed and no corner
//! closed them ("corner failure"). Where the upper and lower kink rounds
//! meet at mid-height, their contacts end on the walls' curved split, which
//! both halves' trims now split at the same points: before, each made its
//! own, and the rounds' shared section was closed by two slivers of plane.
//!
//! Read from STEP, as a file gives the part.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::{all_blended, fillet_v2};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

const RADIUS: f64 = 30.0;

/// The file's plate without its bores: the bores' walls left out, the top
/// and bottom without their holes.
fn load(topo: &mut Topology) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/kinked_plate_halves.step");
    let bored = read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0];
    let mut faces = Vec::new();
    for f in solid_faces(topo, bored).unwrap() {
        let face = topo.face(f).unwrap().clone();
        match face.surface() {
            FaceSurface::Cylinder(_) => {}
            _ if face.inner_wires().is_empty() => faces.push(f),
            surface => {
                let mut whole = Face::new(face.outer_wire(), Vec::new(), surface.clone());
                whole.set_reversed(face.is_reversed());
                faces.push(topo.add_face(whole));
            }
        }
    }
    let shell = topo.add_shell(Shell::new(faces).unwrap());
    topo.add_solid(Solid::new(shell, Vec::new()))
}

/// The edge of `solid` passing nearest `p`.
fn edge_through(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
    let off = |e: EdgeId| {
        let edge = topo.edge(e).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=2000)
            .map(|k| {
                let t = t0 + (t1 - t0) * f64::from(k) / 2000.0;
                (edge.curve().evaluate_with_endpoints(t, a, b) - p).length()
            })
            .fold(f64::INFINITY, f64::min)
    };
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap()
}

/// The four kink edges (two each side of mid-height).
const KINKS: [[f64; 3]; 4] = [
    [622.524_066_232_872_9, -829.491_116_542_209_6, -20.0],
    [601.563_011_294_882, -920.050_424_148_843_3, -20.0],
    [622.524_066_232_872_9, -829.491_116_542_209_6, 20.0],
    [601.563_011_294_882, -920.050_424_148_843_3, 20.0],
];
/// A point on each side's rim, top and bottom.
const RIMS: [[f64; 3]; 4] = [
    [569.063_648_292_335_9, -865.176_946_023_415_3, -40.0],
    [806.394_894_125_862_4, -923.066_706_269_628_9, -40.0],
    [569.063_648_292_335_9, -865.176_946_023_415_3, 40.0],
    [806.394_894_125_862_4, -923.066_706_269_628_9, 40.0],
];

fn rounded(points: &[[f64; 3]]) -> (Topology, SolidId, SolidId) {
    let mut topo = Topology::new();
    let plate = load(&mut topo);
    let edges: Vec<EdgeId> = points
        .iter()
        .map(|p| edge_through(&topo, plate, Point3::new(p[0], p[1], p[2])))
        .collect();
    let result = all_blended(fillet_v2(&mut topo, plate, &edges, RADIUS).unwrap()).unwrap();
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let mesh = tessellate_solid(&topo, result, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    (topo, plate, result)
}

fn count(topo: &Topology, solid: SolidId, kind: fn(&FaceSurface) -> bool) -> usize {
    solid_faces(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&f| kind(topo.face(f).unwrap().surface()))
        .count()
}

#[test]
fn the_kink_rounds_meet_on_one_section_at_mid_height() {
    let (topo, _, result) = rounded(&KINKS);
    // Only the plate's top and bottom: no sliver closing a round's section.
    assert_eq!(
        count(&topo, result, |s| matches!(s, FaceSurface::Plane { .. })),
        2
    );
}

#[test]
fn three_walked_rounds_meet_in_a_ball_at_each_corner() {
    let mut points = KINKS.to_vec();
    points.extend_from_slice(&RIMS);
    let (topo, plate, result) = rounded(&points);
    // A ball's patch at each of the four corners: of the radius, touching
    // the top or the bottom (40 off mid-height).
    let balls: Vec<_> = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .filter_map(|f| match topo.face(f).unwrap().surface() {
            FaceSurface::Sphere(sphere) => Some(sphere.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(balls.len(), 4, "a ball's patch at each corner");
    for ball in &balls {
        assert!(
            (ball.radius() - RADIUS).abs() < 1e-9,
            "radius {}",
            ball.radius()
        );
        let height = ball.center().z().abs();
        assert!(
            (height - (40.0 - RADIUS)).abs() < 1e-6,
            "centre at height {height}"
        );
    }
    assert!(solid_volume(&topo, plate, 0.1).unwrap() > solid_volume(&topo, result, 0.1).unwrap());
}

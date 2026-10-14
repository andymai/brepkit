//! A STEP surface model reads as shapes too: each shell of a
//! `SHELL_BASED_SURFACE_MODEL`, open or closed, and each face set of a
//! `FACE_BASED_SURFACE_MODEL`, comes back as a solid of that one shell.
#![allow(clippy::unwrap_used, clippy::expect_used)]

#[path = "support/step_text.rs"]
mod step_text;

use std::f64::consts::PI;

use brepkit_io::step::reader::read_step;
use brepkit_math::chord::DEFAULT_ANGULAR_TOL;
use brepkit_math::vec::Vec3;
use brepkit_operations::tessellate::{
    is_watertight, tessellate_solid, tessellate_solid_grouped_with_tolerance,
};
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use step_text::StepText;

const R: f64 = 10.0;
const H: f64 = 30.0;

/// A cup: a floor of radius `R` and a wall of height `H`, no lid. Returns
/// the floor and the wall.
fn cup(s: &mut StepText) -> (u64, u64) {
    let a = s.placement([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let surface = s.add(format!("CYLINDRICAL_SURFACE('',#{a},{R:?})"));
    let (v0, v1) = (s.vertex([R, 0.0, 0.0]), s.vertex([R, 0.0, H]));
    let c0 = s.circle([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let c1 = s.circle([0.0, 0.0, H], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], R);
    let (e0, e1) = (s.edge(v0, v0, c0), s.edge(v1, v1, c1));
    let up = s.line([R, 0.0, 0.0], [0.0, 0.0, 1.0]);
    let seam = s.edge(v0, v1, up);
    let floor_loop = s.edge_loop(&[(e0, false)]);
    let floor_plane = s.plane([0.0; 3], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let floor = s.face(&[floor_loop], floor_plane, false);
    let wall_loop = s.edge_loop(&[(e0, true), (seam, true), (e1, false), (seam, false)]);
    let wall = s.face(&[wall_loop], surface, true);
    (floor, wall)
}

/// Read `step`'s one shape and check it is the cup, its floor facing `down`.
fn assert_cup(step: &str, down: bool, label: &str) {
    let mut topo = Topology::new();
    let solids = read_step(step, &mut topo).unwrap();
    assert_eq!(solids.len(), 1, "{label}: one shape");
    let solid = solids[0];
    let faces = solid_faces(&topo, solid).unwrap();
    assert_eq!(faces.len(), 2, "{label}: floor and wall");

    let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
    assert!(!is_watertight(&mesh), "{label}: the cup is open");

    let (mesh, offsets) =
        tessellate_solid_grouped_with_tolerance(&topo, solid, 0.01, DEFAULT_ANGULAR_TOL).unwrap();
    let mut areas = Vec::new();
    for (k, &f) in faces.iter().enumerate() {
        let mut area = 0.0;
        let mut facing = Vec3::new(0.0, 0.0, 0.0);
        for t in mesh.indices[offsets[k] as usize..offsets[k + 1] as usize].chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| mesh.positions[t[i] as usize]);
            let n = (b - a).cross(c - a);
            area += 0.5 * n.length();
            facing += n;
        }
        if matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }) {
            assert_eq!(facing.z() < 0.0, down, "{label}: floor faces the wrong way");
        }
        areas.push(area);
    }
    let truth = PI * R * R + 2.0 * PI * R * H;
    let total: f64 = areas.iter().sum();
    assert!(
        (total - truth).abs() < 2e-3 * truth,
        "{label}: area {total}, truth {truth}"
    );
}

#[test]
fn an_open_shell_reads_as_a_shape() {
    let mut s = StepText::new();
    let (floor, wall) = cup(&mut s);
    let shell = s.add(format!("OPEN_SHELL('',(#{floor},#{wall}))"));
    let model = s.add(format!("SHELL_BASED_SURFACE_MODEL('',(#{shell}))"));
    s.representation("MANIFOLD_SURFACE_SHAPE_REPRESENTATION", &[model]);
    assert_cup(&s.finish(), true, "open shell");
}

#[test]
fn a_reversed_open_shell_turns_its_faces_over() {
    let mut s = StepText::new();
    let (floor, wall) = cup(&mut s);
    let shell = s.add(format!("OPEN_SHELL('',(#{floor},#{wall}))"));
    let oriented = s.add(format!("ORIENTED_OPEN_SHELL('',*,#{shell},.F.)"));
    let model = s.add(format!("SHELL_BASED_SURFACE_MODEL('',(#{oriented}))"));
    s.representation("MANIFOLD_SURFACE_SHAPE_REPRESENTATION", &[model]);
    assert_cup(&s.finish(), false, "reversed open shell");
}

#[test]
fn a_face_set_reads_as_a_shape() {
    let mut s = StepText::new();
    let (floor, wall) = cup(&mut s);
    let set = s.add(format!("CONNECTED_FACE_SET('',(#{floor},#{wall}))"));
    let model = s.add(format!("FACE_BASED_SURFACE_MODEL('',(#{set}))"));
    s.representation(
        "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
        &[model],
    );
    assert_cup(&s.finish(), true, "face set");
}

#[test]
fn shells_sharing_an_edge_are_each_placed_once() {
    // An inch file whose floor and wall are separate shells meeting on one
    // edge: each shell is its own solid, and both are scaled to millimetres
    // once.
    let mut s = StepText::with_unit(step_text::Unit::Inch);
    let (floor, wall) = cup(&mut s);
    let floor_shell = s.add(format!("OPEN_SHELL('',(#{floor}))"));
    let wall_shell = s.add(format!("OPEN_SHELL('',(#{wall}))"));
    let model = s.add(format!(
        "SHELL_BASED_SURFACE_MODEL('',(#{floor_shell},#{wall_shell}))"
    ));
    s.representation("MANIFOLD_SURFACE_SHAPE_REPRESENTATION", &[model]);
    let mut topo = Topology::new();
    let shapes = read_step(&s.finish(), &mut topo).unwrap();
    assert_eq!(shapes.len(), 2, "one shape per shell");
    for shape in shapes {
        for f in solid_faces(&topo, shape).unwrap() {
            let face = topo.face(f).unwrap();
            for oe in topo.wire(face.outer_wire()).unwrap().edges() {
                let e = topo.edge(oe.edge()).unwrap();
                for v in [e.start(), e.end()] {
                    let p = topo.vertex(v).unwrap().point();
                    let radius = p.x().hypot(p.y());
                    assert!(
                        (radius - R * 25.4).abs() < 1e-9,
                        "vertex {p:?} scaled other than once"
                    );
                }
            }
        }
    }
}

#[test]
fn a_sheet_the_reader_cannot_build_leaves_the_solids() {
    // A cup and, beside it, a sheet on a surface the reader does not know:
    // the cup still comes through.
    let mut s = StepText::new();
    let (floor, wall) = cup(&mut s);
    let solid_shell = s.add(format!("OPEN_SHELL('',(#{floor},#{wall}))"));
    let solid_model = s.add(format!("SHELL_BASED_SURFACE_MODEL('',(#{solid_shell}))"));
    let basis = s.plane([0.0, 0.0, 50.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]);
    let offset = s.add(format!("OFFSET_SURFACE('',#{basis},2.,.F.)"));
    let sheet_vertex = s.vertex([5.0, 0.0, 52.0]);
    let sheet_circle = s.circle([0.0, 0.0, 52.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], 5.0);
    let sheet_edge = s.edge(sheet_vertex, sheet_vertex, sheet_circle);
    let sheet_loop = s.edge_loop(&[(sheet_edge, true)]);
    let sheet = s.face(&[sheet_loop], offset, true);
    let sheet_shell = s.add(format!("OPEN_SHELL('',(#{sheet}))"));
    let sheet_model = s.add(format!("SHELL_BASED_SURFACE_MODEL('',(#{sheet_shell}))"));
    s.representation(
        "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        &[solid_model, sheet_model],
    );
    let mut topo = Topology::new();
    let shapes = read_step(&s.finish(), &mut topo).unwrap();
    assert_eq!(shapes.len(), 1, "the cup comes through, the sheet does not");
}

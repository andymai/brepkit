//! A combustor liner hollowed out, from operands captured from brepjs's
//! `turbofan-engine` playground example.
//!
//! Each operand is the revolve of a face with one closed spline edge, sixteen
//! NURBS faces, and the tool is the blank scaled to 0.78 about the middle of
//! its profile: it lies wholly inside the blank, a millimetre from it at the
//! nearest. No pair of their faces meets, and the boolean searched every one
//! of the 192 whose boxes overlap for a section anyway, by grid and along
//! each boundary curve, then meshed the result's faces to validate it for
//! the log: 1.7 s, where the cut is the blank with the tool's shell for a
//! cavity.
//!
//! Data: `spline_can_outer.bin` (the blank) and `spline_can_inner.bin` (the
//! tool).

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::Path;

use brepkit_io::arena_io::deserialize_solid;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{self, BooleanOp};
use brepkit_operations::classify::{PointClassification, classify_point};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn load(topo: &mut Topology, name: &str) -> SolidId {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(name);
    deserialize_solid(&std::fs::read(path).unwrap(), topo).unwrap()
}

fn census(topo: &Topology, solid: SolidId) -> BTreeMap<&'static str, usize> {
    let mut m = BTreeMap::new();
    for f in brepkit_topology::explorer::solid_faces(topo, solid).unwrap() {
        *m.entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    m
}

/// The liner's profile is centred on radius 28 at z = 250: the blank reaches
/// 5.5 out from there and the tool 4.3.
fn at_radius(radius: f64, degrees: f64) -> Point3 {
    let a = degrees.to_radians();
    Point3::new(radius * a.cos(), radius * a.sin(), 250.0)
}

fn hollow(topo: &Topology, liner: SolidId) {
    assert_eq!(census(topo, liner), BTreeMap::from([("nurbs", 32)]));
    assert_eq!(topo.solid(liner).unwrap().inner_shells().len(), 1);
    let report = validate_solid(topo, liner).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    for degrees in [10.0, 100.0, 190.0, 280.0] {
        let class = |radius: f64| {
            classify_point(topo, liner, at_radius(radius, degrees), 0.01, 1e-7).unwrap()
        };
        assert_eq!(
            class(28.0),
            PointClassification::Outside,
            "the cavity at {degrees}"
        );
        assert_eq!(
            class(32.5),
            PointClassification::Inside,
            "the outer wall at {degrees}"
        );
        assert_eq!(
            class(23.5),
            PointClassification::Inside,
            "the inner wall at {degrees}"
        );
        assert_eq!(
            class(40.0),
            PointClassification::Outside,
            "outside at {degrees}"
        );
        assert_eq!(
            class(10.0),
            PointClassification::Outside,
            "the bore at {degrees}"
        );
    }
}

#[test]
fn a_tool_inside_the_liner_leaves_a_cavity() {
    let mut topo = Topology::new();
    let blank = load(&mut topo, "spline_can_outer.bin");
    let tool = load(&mut topo, "spline_can_inner.bin");

    let before = boolean::mesh_fallback_count();
    let liner = boolean::boolean(&mut topo, BooleanOp::Cut, blank, tool).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");
    hollow(&topo, liner);
}

/// The cut that records where each face came from takes the same route: the
/// result's faces are the operands' own, one for one.
#[test]
fn the_cut_with_face_history_copies_every_face_once() {
    let mut topo = Topology::new();
    let blank = load(&mut topo, "spline_can_outer.bin");
    let tool = load(&mut topo, "spline_can_inner.bin");
    let inputs: Vec<usize> = [blank, tool]
        .into_iter()
        .flat_map(|s| brepkit_topology::explorer::solid_faces(&topo, s).unwrap())
        .map(brepkit_topology::arena::Id::index)
        .collect();

    let (liner, history) =
        boolean::boolean_with_evolution(&mut topo, BooleanOp::Cut, blank, tool).unwrap();
    hollow(&topo, liner);

    assert!(history.deleted.is_empty(), "{:?}", history.deleted);
    let mut images: Vec<usize> = Vec::new();
    for input in &inputs {
        let image = history.modified.get(input).expect("an image for each face");
        assert_eq!(image.len(), 1, "face {input}");
        let from = topo.face(topo.face_id_from_index(*input).unwrap()).unwrap();
        let to = topo
            .face(topo.face_id_from_index(image[0]).unwrap())
            .unwrap();
        assert_eq!(
            format!("{:?}", from.surface()),
            format!("{:?}", to.surface()),
            "face {input}"
        );
        images.push(image[0]);
    }
    images.sort_unstable();
    images.dedup();
    assert_eq!(images.len(), 32);
}

#[test]
fn the_engine_finds_the_same_cavity_with_no_section_to_find() {
    let mut topo = Topology::new();
    let blank = load(&mut topo, "spline_can_outer.bin");
    let tool = load(&mut topo, "spline_can_inner.bin");

    let liner =
        brepkit_algo::gfa::boolean(&mut topo, brepkit_algo::bop::BooleanOp::Cut, blank, tool)
            .unwrap();
    hollow(&topo, liner);
}

#[test]
fn a_tool_in_two_pieces_takes_no_cavity_for_the_piece_outside_the_liner() {
    // One shell holding the tool and a fifth-size copy of the blank about the
    // axis: a ring in the liner's bore, inside its box and outside its wall.
    // A vertex of the first piece lies inside the liner, which says nothing
    // of the second.
    let mut topo = Topology::new();
    let blank = load(&mut topo, "spline_can_outer.bin");
    let inside = load(&mut topo, "spline_can_inner.bin");
    let in_the_bore = brepkit_operations::copy::copy_solid(&mut topo, blank).unwrap();
    let about = |m: brepkit_math::mat::Mat4| {
        brepkit_math::mat::Mat4::translation(0.0, 0.0, 250.0)
            * m
            * brepkit_math::mat::Mat4::translation(0.0, 0.0, -250.0)
    };
    brepkit_operations::transform::transform_solid(
        &mut topo,
        in_the_bore,
        &about(brepkit_math::mat::Mat4::scale(0.2, 0.2, 0.2)),
    )
    .unwrap();
    let faces: Vec<_> = [inside, in_the_bore]
        .into_iter()
        .flat_map(|s| brepkit_topology::explorer::solid_faces(&topo, s).unwrap())
        .collect();
    let shell = topo.add_shell(brepkit_topology::shell::Shell::new(faces).unwrap());
    let tool = topo.add_solid(brepkit_topology::solid::Solid::new(shell, vec![]));

    let liner = boolean::boolean(&mut topo, BooleanOp::Cut, blank, tool).unwrap();
    for degrees in [10.0, 100.0, 190.0, 280.0] {
        let class = |radius: f64| {
            classify_point(&topo, liner, at_radius(radius, degrees), 0.01, 1e-7).unwrap()
        };
        assert_eq!(
            class(28.0),
            PointClassification::Outside,
            "the cavity at {degrees}"
        );
        assert_eq!(
            class(32.5),
            PointClassification::Inside,
            "the outer wall at {degrees}"
        );
        assert_eq!(
            class(23.5),
            PointClassification::Inside,
            "the inner wall at {degrees}"
        );
        // In the wall of the ring in the bore, and beside it.
        assert_eq!(
            class(5.6),
            PointClassification::Outside,
            "the ring in the bore at {degrees}"
        );
        assert_eq!(
            class(10.0),
            PointClassification::Outside,
            "the bore at {degrees}"
        );
    }
}

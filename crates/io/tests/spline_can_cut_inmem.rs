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

//! A hollow combustor liner opened by a wedge, from operands captured from
//! brepjs's `turbofan-engine` playground example.
//!
//! The liner is the revolve of a closed spline profile with a cavity of the
//! same shape inside it: 32 NURBS faces. The wedge is 130 degrees of a
//! cylinder about the liner's axis, so each of its two flat faces lies in a
//! plane through that axis and cuts the liner along meridians of its
//! surfaces: curves the surfaces hold themselves.
//!
//! Data: `spline_liner.bin` (the blank) and `wedge_across_seam.bin` (the
//! tool, from 65 degrees either side of the x axis).

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

/// The liner's profile is centred on radius 28 at z = 250: its wall runs
/// from 4.3 to 5.5 out from there.
fn at_radius(radius: f64, degrees: f64) -> Point3 {
    let a = degrees.to_radians();
    Point3::new(radius * a.cos(), radius * a.sin(), 250.0)
}

#[test]
fn a_wedge_through_the_axis_opens_the_liner_along_its_meridians() {
    let mut topo = Topology::new();
    let liner = load(&mut topo, "spline_liner.bin");
    let wedge = load(&mut topo, "wedge_across_seam.bin");

    let before = boolean::mesh_fallback_count();
    let opened = boolean::boolean(&mut topo, BooleanOp::Cut, liner, wedge).unwrap();
    assert_eq!(boolean::mesh_fallback_count(), before, "mesh fallback");

    let mut census: BTreeMap<&'static str, usize> = BTreeMap::new();
    for f in brepkit_topology::explorer::solid_faces(&topo, opened).unwrap() {
        *census
            .entry(topo.face(f).unwrap().surface().type_tag())
            .or_default() += 1;
    }
    assert_eq!(census, BTreeMap::from([("nurbs", 32), ("plane", 2)]));
    let report = validate_solid(&topo, opened).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    let class = |radius: f64, degrees: f64| {
        classify_point(&topo, opened, at_radius(radius, degrees), 0.01, 1e-7).unwrap()
    };
    for degrees in [-40.0, 0.0, 40.0] {
        assert_eq!(
            class(32.5, degrees),
            PointClassification::Outside,
            "the wall the wedge took at {degrees}"
        );
    }
    for degrees in [90.0, 180.0, 270.0] {
        assert_eq!(
            class(32.5, degrees),
            PointClassification::Inside,
            "the outer wall at {degrees}"
        );
        assert_eq!(
            class(23.5, degrees),
            PointClassification::Inside,
            "the inner wall at {degrees}"
        );
        assert_eq!(
            class(28.0, degrees),
            PointClassification::Outside,
            "the cavity at {degrees}"
        );
    }
}

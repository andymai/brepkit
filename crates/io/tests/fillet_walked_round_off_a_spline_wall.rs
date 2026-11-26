//! A plate's two rim edges along its B-spline wall, rounded at r = 7.
//!
//! The rounds are walked between the plate's faces and its wall, read from
//! STEP. Each contact on the wall is a fit through the walk's sections,
//! and it stands off the wall's surface farther than a pcurve fitted along
//! it is allowed to. Such a contact takes the line between its ends'
//! parameters on the wall as its pcurve, as before contacts were fitted
//! along their edges; the trim had refused it ("trimming failure").

#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::Path;

use brepkit_io::step::reader::read_step;
use brepkit_math::vec::Point3;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

/// The edge of `solid` passing nearest `p`.
fn edge_near(topo: &Topology, solid: SolidId, p: Point3) -> EdgeId {
    let off = |e: EdgeId| {
        let edge = topo.edge(e).unwrap();
        let (a, b) = (
            topo.vertex(edge.start()).unwrap().point(),
            topo.vertex(edge.end()).unwrap().point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
        (0..=4000)
            .map(|k| {
                let t = (t1 - t0).mul_add(f64::from(k) / 4000.0, t0);
                (edge.curve().evaluate_with_endpoints(t, a, b) - p).length()
            })
            .fold(f64::MAX, f64::min)
    };
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .min_by(|&a, &b| off(a).total_cmp(&off(b)))
        .unwrap()
}

#[test]
fn walked_rounds_whose_contacts_stand_off_a_spline_wall_trim_it() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/plate_round_off_its_spline_wall.step");
    let mut topo = Topology::new();
    let plate = read_step(&std::fs::read_to_string(path).unwrap(), &mut topo).unwrap()[0];
    let picks = [
        Point3::new(-2.223_480_299_731_921, 61.040_233_415_118_39, 18.0),
        Point3::new(-2.223_480_299_731_899, 61.040_233_415_118_3, -18.0),
    ]
    .map(|p| edge_near(&topo, plate, p));
    let result = fillet_v2(&mut topo, plate, &picks, 7.0).unwrap();
    assert!(result.failed.is_empty(), "{:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(solid_faces(&topo, result.solid).unwrap().len(), 5);
    let mesh = tessellate_solid(&topo, result.solid, 0.5).unwrap();
    assert!(is_watertight(&mesh), "mesh not watertight");
    // Another kernel's measure of its result of the same fillet.
    let volume = solid_volume(&topo, result.solid, 0.1).unwrap();
    let expected = 538_136.734;
    assert!(
        (volume - expected).abs() < 1e-4 * expected,
        "volume {volume}, expected {expected}"
    );
}

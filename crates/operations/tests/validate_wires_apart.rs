//! `validate_solid` reports a face whose hole crosses its outline. A round
//! on a drilled box's top edge, made by the direct fillet with the bore
//! across the round's contact, trims the top face's outline through the
//! bore's mouth: the mouth stays a hole of the face, and the outline crosses
//! it.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use brepkit_blend::fillet_builder::FilletBuilder;
use brepkit_math::mat::Mat4;
use brepkit_operations::blend_ops::all_blended;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::solid::SolidId;

const LENGTH: f64 = 40.0;
const WIDTH: f64 = 20.0;
const HEIGHT: f64 = 15.0;

/// A box drilled through its height at (20, `cy`), and its top edge along x
/// at y = 0.
fn drilled_box(topo: &mut Topology, cy: f64, rho: f64) -> (SolidId, EdgeId) {
    let block = make_box(topo, LENGTH, WIDTH, HEIGHT).unwrap();
    let drill = make_cylinder(topo, rho, HEIGHT + 2.0).unwrap();
    transform_solid(topo, drill, &Mat4::translation(0.5 * LENGTH, cy, -1.0)).unwrap();
    let solid = boolean(topo, BooleanOp::Cut, block, drill).unwrap();
    let edge = solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()].iter().all(|&v| {
                let p = topo.vertex(v).unwrap().point();
                p.y().abs() < 1e-9 && (p.z() - HEIGHT).abs() < 1e-9
            })
        })
        .unwrap();
    (solid, edge)
}

fn direct_fillet(topo: &mut Topology, solid: SolidId, edge: EdgeId, r: f64) -> SolidId {
    let mut builder = FilletBuilder::new(topo, solid);
    builder.add_edges(&[edge], r);
    all_blended(builder.build().unwrap()).unwrap()
}

fn crossing_errors(topo: &Topology, solid: SolidId) -> Vec<String> {
    validate_solid(topo, solid)
        .unwrap()
        .issues
        .into_iter()
        .map(|i| i.description)
        .filter(|d| d.contains("inner wire"))
        .collect()
}

#[test]
fn a_round_whose_contact_crosses_a_bore_is_not_valid() {
    for pose in [
        Mat4::identity(),
        Mat4::translation(-37.2, 5.5, 140.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9),
        Mat4::scale(-1.0, 1.0, 1.0),
    ] {
        let mut topo = Topology::new();
        let (solid, edge) = drilled_box(&mut topo, 6.0, 2.0);
        transform_solid(&mut topo, solid, &pose).unwrap();
        assert!(crossing_errors(&topo, solid).is_empty(), "the drilled box");
        let round = direct_fillet(&mut topo, solid, edge, 7.0);
        let errors = crossing_errors(&topo, round);
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("crosses the outer wire"), "{errors:?}");
        assert!(!validate_solid(&topo, round).unwrap().is_valid());
    }
}

#[test]
fn a_round_short_of_a_bore_is_valid() {
    let mut topo = Topology::new();
    let (solid, edge) = drilled_box(&mut topo, 6.0, 2.0);
    let round = direct_fillet(&mut topo, solid, edge, 3.0);
    let report = validate_solid(&topo, round).unwrap();
    assert!(report.is_valid(), "{report:?}");
}

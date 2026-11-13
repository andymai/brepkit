//! A closed rim chamfered at exactly its wall's height (a disc as thick as
//! the distance, a pin as tall, a blind hole as deep), or at exactly its
//! disc cap's radius (a rod's end). The wall or the cap goes whole: the
//! chamfer's cone runs from the cap to the wall's far rim, or closes to a
//! point on the axis. Both were refused ("consumes the wall", "consumes the
//! cap"). Each result is valid, watertight, and takes off the revolved
//! triangle, pi d^2 (R -/+ d/3).
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::solid::SolidId;

const R: f64 = 4.0;
const D: f64 = 2.0;

/// The closed circle of radius `R` at height `z`.
fn rim_at(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c)
                if edge.start() == edge.end()
                    && (c.radius() - R).abs() < 1e-9
                    && (c.center().z() - z).abs() < 1e-9)
        })
        .expect("the rim")
}

fn check(topo: &mut Topology, solid: SolidId, rim: EdgeId, faces: usize, removed: f64) {
    let before = solid_volume(topo, solid, 0.01).unwrap();
    let result = chamfer(topo, solid, &[rim], D).unwrap();
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    assert_eq!(solid_faces(topo, result).unwrap().len(), faces);
    for deflection in [0.5, 0.05] {
        let mesh = tessellate_solid(topo, result, deflection).unwrap();
        assert!(is_watertight(&mesh), "mesh not watertight at {deflection}");
    }
    let truth = before - removed;
    let volume = solid_volume(topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
}

#[test]
fn a_disc_as_thick_as_the_chamfer_becomes_a_frustum() {
    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, R, D).unwrap();
    let rim = rim_at(&topo, disc, D);
    // The top disc, the cone and the bottom disc.
    check(&mut topo, disc, rim, 3, PI * D * D * (R - D / 3.0));
}

#[test]
fn a_pin_as_tall_as_the_chamfer_becomes_a_frustum_on_its_plate() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 5.0).unwrap();
    let pin = make_cylinder(&mut topo, R, D).unwrap();
    transform_solid(&mut topo, pin, &Mat4::translation(10.0, 10.0, 5.0)).unwrap();
    let part = boolean(&mut topo, BooleanOp::Fuse, plate, pin).unwrap();
    let rim = rim_at(&topo, part, 5.0 + D);
    // The plate's six faces, the pin's top and its cone.
    check(&mut topo, part, rim, 8, PI * D * D * (R - D / 3.0));
}

#[test]
fn a_blind_hole_as_deep_as_the_chamfer_becomes_a_countersink() {
    let mut topo = Topology::new();
    let plate = make_box(&mut topo, 20.0, 20.0, 5.0).unwrap();
    let drill = make_cylinder(&mut topo, R, D + 1.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(10.0, 10.0, 5.0 - D)).unwrap();
    let part = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
    let rim = rim_at(&topo, part, 5.0);
    // The plate's six faces, the hole's floor and the countersink.
    check(&mut topo, part, rim, 8, PI * D * D * (R + D / 3.0));
}

#[test]
fn a_rod_chamfered_at_its_radius_comes_to_a_point() {
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, D, 10.0).unwrap();
    let rim = solid_edges(&topo, rod)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c)
                if edge.start() == edge.end() && (c.center().z() - 10.0).abs() < 1e-9)
        })
        .unwrap();
    // The point, the wall and the bottom disc.
    check(&mut topo, rod, rim, 3, PI * D * D * (D - D / 3.0));
}

#[test]
fn a_cap_and_a_wall_both_consumed_are_refused() {
    // A disc of radius and thickness both the distance: no wall and no cap
    // are left for the cone to meet.
    let mut topo = Topology::new();
    let disc = make_cylinder(&mut topo, D, D).unwrap();
    let rim = solid_edges(&topo, disc)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            matches!(edge.curve(), EdgeCurve::Circle(c)
                if edge.start() == edge.end() && (c.center().z() - D).abs() < 1e-9)
        })
        .unwrap();
    assert!(chamfer(&mut topo, disc, &[rim], D).is_err());
}

#[test]
fn a_drilled_cap_chamfered_to_its_centre_is_refused() {
    // The cap's hole is in the cone's way.
    let mut topo = Topology::new();
    let rod = make_cylinder(&mut topo, R, 10.0).unwrap();
    let drill = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(0.0, 0.0, 8.0)).unwrap();
    let part = boolean(&mut topo, BooleanOp::Cut, rod, drill).unwrap();
    let rim = rim_at(&topo, part, 10.0);
    assert!(chamfer(&mut topo, part, &[rim], R).is_err());
}

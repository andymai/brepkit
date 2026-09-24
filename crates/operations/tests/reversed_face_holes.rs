//! A face flipped into a cavity by a cut keeps its wires and carries the
//! reversed flag, so its stored outer wire still runs counter-clockwise about
//! the surface normal. A hole cut into it later must run against that wire.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

/// A bore's wall is a reversed cylinder that keeps its wires, so a pocket cut
/// through it later leaves a hole wound the same way as the wall's outer wire.
#[test]
fn pocket_through_a_bore_wall_is_exact() {
    let mut topo = Topology::new();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let bore = make_cylinder(&mut topo, 3.0, 10.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(5.0, 5.0, 0.0)).unwrap();
    let bored = boolean(&mut topo, BooleanOp::Cut, block, bore).unwrap();
    let pocket = make_box(&mut topo, 1.0, 2.5, 2.0).unwrap();
    transform_solid(&mut topo, pocket, &Mat4::translation(4.5, 7.0, 4.0)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, bored, pocket).unwrap();

    let wall = solid_faces(&topo, result)
        .unwrap()
        .into_iter()
        .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
        .unwrap();
    let wall_face = topo.face(wall).unwrap();
    assert!(wall_face.is_reversed() && wall_face.inner_wires().len() == 1);
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);

    // The block less the bore, less the pocket's material beyond the bore:
    // 2 ∫ (4.5 - sqrt(9 - t²)) dt over t in [-0.5, 0.5], 2 deep.
    let chord =
        |t: f64| 0.5 * (t * (3.0 * 3.0 - t * t).sqrt()) + 0.5 * 3.0 * 3.0 * (t / 3.0).asin();
    let pocket_material = 2.0 * (4.5 - (chord(0.5) - chord(-0.5)));
    let truth = 1000.0 - std::f64::consts::PI * 3.0 * 3.0 * 10.0 - pocket_material;
    let exact = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (exact - truth).abs() < 1e-9 * truth,
        "solid_volume {exact}, closed form {truth}"
    );
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, result, deflection).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
    }
    let fine = oriented_solid_volume(&topo, result, 0.0005).unwrap();
    assert!(
        (fine - truth).abs() < 1e-3 * truth,
        "mesh volume {fine}, closed form {truth}"
    );
}

/// The floor of a box's cavity is a reversed plane; a drill through the
/// bottom that stops inside the cavity leaves a circular hole in it.
#[test]
fn drill_into_a_cavity_floor_is_valid() {
    let mut topo = Topology::new();
    let outer = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let inner = make_box(&mut topo, 6.0, 6.0, 6.0).unwrap();
    transform_solid(&mut topo, inner, &Mat4::translation(2.0, 2.0, 2.0)).unwrap();
    let hollow = boolean(&mut topo, BooleanOp::Cut, outer, inner).unwrap();
    let drill = make_cylinder(&mut topo, 1.0, 6.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(5.0, 5.0, -1.0)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, hollow, drill).unwrap();

    assert!(
        solid_faces(&topo, result).unwrap().iter().any(|&f| {
            let face = topo.face(f).unwrap();
            face.is_reversed() && !face.inner_wires().is_empty()
        }),
        "the cavity floor carries the drill's hole"
    );
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{:?}", report.issues);
    let truth = 1000.0 - 216.0 - 2.0 * std::f64::consts::PI;
    let exact = solid_volume(&topo, result, 0.001).unwrap();
    assert!(
        (exact - truth).abs() < 1e-9 * truth,
        "solid_volume {exact}, closed form {truth}"
    );
    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
    assert_eq!(boundary_edge_count(&mesh), 0);
}

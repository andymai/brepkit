//! Solids that touch at a single point: two boxes sharing a corner and two
//! balls touching. Their fuse keeps both pieces as they are, joined at the
//! point, and later booleans and meshing take it like any other solid.

#![allow(clippy::unwrap_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

/// The faces' surface kinds, and whether the solid validates.
fn check(topo: &Topology, solid: SolidId, label: &str) -> Vec<&'static str> {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{label}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
    solid_faces(topo, solid)
        .unwrap()
        .iter()
        .map(|&f| topo.face(f).unwrap().surface().type_tag())
        .collect()
}

#[test]
fn boxes_sharing_a_corner_fuse_exactly() {
    let mut topo = Topology::default();
    let a = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    let b = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
    transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.0, 1.0)).unwrap();
    let fallbacks = mesh_fallback_count();
    let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
    assert_eq!(mesh_fallback_count(), fallbacks, "fuse fell back");
    assert_eq!(check(&topo, fused, "fuse"), vec!["plane"; 12]);
    assert!((solid_volume(&topo, fused, 0.001).unwrap() - 2.0).abs() < 1e-12);

    // A cube of side 0.2 centred on the first box's far corner takes an
    // eighth of itself.
    let notch = make_box(&mut topo, 0.2, 0.2, 0.2).unwrap();
    transform_solid(&mut topo, notch, &Mat4::translation(-0.1, -0.1, -0.1)).unwrap();
    let cut = boolean(&mut topo, BooleanOp::Cut, fused, notch).unwrap();
    assert_eq!(mesh_fallback_count(), fallbacks, "cut fell back");
    assert_eq!(check(&topo, cut, "cut"), vec!["plane"; 15]);
    assert!((solid_volume(&topo, cut, 0.001).unwrap() - 1.999).abs() < 1e-12);
}

#[test]
fn balls_touching_fuse_exactly() {
    let ball = 4.0 * PI / 3.0;
    for segments in [16, 32] {
        let mut topo = Topology::default();
        let a = make_sphere(&mut topo, 1.0, segments).unwrap();
        let b = make_sphere(&mut topo, 1.0, segments).unwrap();
        transform_solid(&mut topo, b, &Mat4::translation(2.0, 0.0, 0.0)).unwrap();
        let fallbacks = mesh_fallback_count();
        let fused = boolean(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        assert_eq!(mesh_fallback_count(), fallbacks, "fuse fell back");
        assert_eq!(check(&topo, fused, "fuse"), vec!["sphere"; 4]);
        let got = solid_volume(&topo, fused, 0.001).unwrap();
        assert!((got - 2.0 * ball).abs() < 1e-9, "fuse volume {got}");

        // The half-space x < -0.5 takes a cap 0.5 high off the first ball.
        let slab = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, slab, &Mat4::translation(-10.5, -5.0, -5.0)).unwrap();
        let cut = boolean(&mut topo, BooleanOp::Cut, fused, slab).unwrap();
        assert_eq!(mesh_fallback_count(), fallbacks, "cut fell back");
        let kinds = check(&topo, cut, "cut");
        assert!(
            kinds.contains(&"plane") && kinds.contains(&"sphere"),
            "{kinds:?}"
        );
        let cap = PI * 0.25 * (3.0 - 0.5) / 3.0;
        let got = solid_volume(&topo, cut, 0.001).unwrap();
        assert!((got - (2.0 * ball - cap)).abs() < 1e-9, "cut volume {got}");
    }
}

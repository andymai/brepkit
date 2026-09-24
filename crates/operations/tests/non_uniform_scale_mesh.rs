//! A non-uniform scale turns spheres and tori into exact NURBS images. The
//! ellipsoid primitive's hemispheres are caps bounded by the equator alone, a
//! boundary that winds the periodic direction once and is closed by a pole
//! row; a torus's one face closes on itself in both directions.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::measure::oriented_solid_volume;
use brepkit_operations::primitives::{make_sphere, make_torus};
use brepkit_operations::tessellate::{
    boundary_edge_count, non_manifold_edge_count, tessellate, tessellate_solid,
};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;

#[test]
fn ellipsoid_meshes_watertight_at_its_volume() {
    let _ = env_logger::builder().is_test(true).try_init();
    let mut topo = Topology::new();
    let unit = make_sphere(&mut topo, 1.0, 16).unwrap();
    let unit_volume = oriented_solid_volume(&topo, unit, 0.001).unwrap();

    let ellipsoid = make_sphere(&mut topo, 1.0, 16).unwrap();
    transform_solid(&mut topo, ellipsoid, &Mat4::scale(2.0, 3.0, 4.0)).unwrap();
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, ellipsoid, deflection).unwrap();
        assert!(!mesh.indices.is_empty(), "empty mesh at {deflection}");
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh at {deflection}"
        );
    }
    // The equator is a polygon of chords, so compare against the unit
    // sphere built the same way, scaled by the map's determinant.
    let volume = oriented_solid_volume(&topo, ellipsoid, 0.001).unwrap();
    let expected = 24.0 * unit_volume;
    assert!(
        (volume - expected).abs() < 5e-3 * expected,
        "ellipsoid volume {volume}, expected {expected}"
    );

    for fid in solid_faces(&topo, ellipsoid).unwrap() {
        let mesh = tessellate(&topo, fid, 0.01).unwrap();
        assert!(!mesh.indices.is_empty(), "face {fid:?} meshes on its own");
    }
}

#[test]
fn squashed_torus_meshes_watertight_at_its_volume() {
    let mut topo = Topology::new();
    let torus = make_torus(&mut topo, 5.0, 1.5, 16).unwrap();
    transform_solid(
        &mut topo,
        torus,
        &(Mat4::rotation_z(0.3) * Mat4::scale(2.0, 1.0, 1.0)),
    )
    .unwrap();
    for deflection in [0.01, 0.001] {
        let mesh = tessellate_solid(&topo, torus, deflection).unwrap();
        assert!(!mesh.indices.is_empty(), "empty mesh at {deflection}");
        assert_eq!(boundary_edge_count(&mesh), 0, "open mesh at {deflection}");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "non-manifold mesh at {deflection}"
        );
    }
    let exact = 2.0 * 2.0 * std::f64::consts::PI.powi(2) * 5.0 * 1.5 * 1.5;
    let volume = oriented_solid_volume(&topo, torus, 0.001).unwrap();
    assert!(
        (volume - exact).abs() < 1e-3 * exact,
        "torus volume {volume}, expected {exact}"
    );
}

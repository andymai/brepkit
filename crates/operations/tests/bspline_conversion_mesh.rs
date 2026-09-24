//! A solid converted to B-splines keeps its shape: a watertight mesh, a valid
//! solid and the same volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::heal::convert_to_bspline;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_cylinder, make_sphere};
use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// `bounds` pairs each deflection with the relative volume error its mesh may
/// carry (the rims' chord loss); `solid_volume` meshes a NURBS solid at its
/// own fine deflection and gets `exact_bound`, when given.
fn assert_keeps(
    topo: &Topology,
    solid: SolidId,
    truth: f64,
    bounds: &[(f64, f64)],
    exact_bound: Option<f64>,
    what: &str,
) {
    let report = validate_solid(topo, solid).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    for &(deflection, bound) in bounds {
        let mesh = tessellate_solid(topo, solid, deflection).unwrap();
        assert_eq!(
            boundary_edge_count(&mesh),
            0,
            "{what}: open mesh at {deflection}"
        );
        let volume = oriented_solid_volume(topo, solid, deflection).unwrap();
        assert!(
            (volume - truth).abs() < bound * truth,
            "{what}: mesh volume {volume} at {deflection}, truth {truth}"
        );
    }
    if let Some(bound) = exact_bound {
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - truth).abs() < bound * truth,
            "{what}: solid_volume {exact}, truth {truth}"
        );
    }
}

/// The cylinder's seam vertex stays on both rims and on the seam, and the
/// caps' patches contain their discs.
#[test]
fn converted_cylinder_keeps_its_shape() {
    let mut topo = Topology::new();
    let cylinder = make_cylinder(&mut topo, 1.5, 4.0).unwrap();
    convert_to_bspline(&mut topo, cylinder).unwrap();
    let truth = std::f64::consts::PI * 1.5 * 1.5 * 4.0;
    assert_keeps(
        &topo,
        cylinder,
        truth,
        &[(0.01, 5e-3), (0.001, 1e-3)],
        Some(3e-4),
        "cylinder",
    );
}

/// A napkin ring: each spherical band lies between two loops that wind once
/// around the axis, and the tunnel stays open.
#[test]
fn converted_bored_sphere_keeps_its_tunnel() {
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, 6.0, 24).unwrap();
    let bore = make_cylinder(&mut topo, 3.0, 20.0).unwrap();
    transform_solid(&mut topo, bore, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
    let ring = boolean(&mut topo, BooleanOp::Cut, sphere, bore).unwrap();
    let truth = solid_volume(&topo, ring, 0.001).unwrap();
    convert_to_bspline(&mut topo, ring).unwrap();
    assert_keeps(&topo, ring, truth, &[(0.01, 3e-3)], None, "bored sphere");
}

//! `BooleanOptions::mesh_fallback`: with `false`, a boolean that needs the
//! mesh fallback returns an error and is counted as declined, building no
//! mesh; a boolean that stays exact is unchanged.
//!
//! Kept in its own file: the fallback counters are process-wide.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{
    self, BooleanOp, BooleanOptions, boolean_with_options, mesh_fallback_count,
    mesh_fallback_declined_count,
};
use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A ring and a rod along y whose top rulings pass over the tube: the cut
/// falls back to the mesh boolean (see `compound_cut_fallback_policy.rs`).
fn ring_and_rod(topo: &mut Topology) -> (SolidId, SolidId) {
    let ring = make_torus(topo, 4.0, 1.5, 32).unwrap();
    let rod = make_cylinder(topo, 0.6, 20.0).unwrap();
    let along_y = Mat4::translation(0.5, 10.0, 1.0) * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
    transform_solid(topo, rod, &along_y).unwrap();
    (ring, rod)
}

fn decline() -> BooleanOptions {
    BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    }
}

#[test]
fn a_declined_fallback_errors_and_is_counted() {
    let mut topo = Topology::new();
    let (ring, rod) = ring_and_rod(&mut topo);
    let before = mesh_fallback_count();
    boolean_with_options(
        &mut topo,
        BooleanOp::Cut,
        ring,
        rod,
        BooleanOptions::default(),
    )
    .unwrap();
    assert_eq!(
        mesh_fallback_count() - before,
        1,
        "premise: the cut must fall back, or this pins nothing"
    );

    let mut topo = Topology::new();
    let (ring, rod) = ring_and_rod(&mut topo);
    let (built, declined) = (mesh_fallback_count(), mesh_fallback_declined_count());
    assert!(boolean_with_options(&mut topo, BooleanOp::Cut, ring, rod, decline()).is_err());
    assert_eq!(mesh_fallback_declined_count() - declined, 1);
    assert_eq!(mesh_fallback_count(), built, "no mesh may be built");
}

#[test]
fn an_exact_boolean_is_the_same_with_the_fallback_declined() {
    let cut = |opts: BooleanOptions| {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let pin = make_cylinder(&mut topo, 2.0, 20.0).unwrap();
        transform_solid(&mut topo, pin, &Mat4::translation(5.0, 5.0, -5.0)).unwrap();
        let result =
            boolean::boolean_with_options(&mut topo, BooleanOp::Cut, block, pin, opts).unwrap();
        brepkit_operations::measure::solid_volume(&topo, result, 0.01).unwrap()
    };
    let (allowed, declined) = (cut(BooleanOptions::default()), cut(decline()));
    assert!(
        (allowed - declined).abs() < 1e-12,
        "{allowed} != {declined}"
    );
    let expected = 1000.0 - std::f64::consts::PI * 4.0 * 10.0;
    assert!((declined - expected).abs() / expected < 1e-6);
}

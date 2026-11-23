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
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// A block (40 x 6 x 70) less a bore of radius 2.25 through it, its seam
/// turned 270 degrees, and a pin of radius 1.7 standing across the bore's
/// mouth, its seam turned 180 degrees: a cut that falls back to the mesh
/// boolean (`a_pin_across_a_bores_mouth_with_both_seams_at_the_fold` in
/// `pin_through_hole_mouth.rs`).
fn bored_block_and_pin(topo: &mut Topology) -> (SolidId, SolidId) {
    let block = make_box(topo, 40.0, 6.0, 70.0).unwrap();
    transform_solid(topo, block, &Mat4::translation(-20.0, -360.0, 751.0)).unwrap();
    let bore = make_cylinder(topo, 2.25, 10.0).unwrap();
    let across = Mat4::translation(0.0, -352.0, 796.0)
        * Mat4::rotation_x(std::f64::consts::FRAC_PI_2)
        * Mat4::rotation_z(270f64.to_radians());
    transform_solid(topo, bore, &across).unwrap();
    let part = boolean::boolean(topo, BooleanOp::Cut, block, bore).unwrap();
    let pin = make_cylinder(topo, 1.7, 444.0).unwrap();
    let standing = Mat4::translation(0.0, -355.0, 706.2) * Mat4::rotation_z(180f64.to_radians());
    transform_solid(topo, pin, &standing).unwrap();
    (part, pin)
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
    let (part, pin) = bored_block_and_pin(&mut topo);
    let before = mesh_fallback_count();
    boolean_with_options(
        &mut topo,
        BooleanOp::Cut,
        part,
        pin,
        BooleanOptions::default(),
    )
    .unwrap();
    assert_eq!(
        mesh_fallback_count() - before,
        1,
        "premise: the cut must fall back, or this pins nothing"
    );

    let mut topo = Topology::new();
    let (part, pin) = bored_block_and_pin(&mut topo);
    let (built, declined) = (mesh_fallback_count(), mesh_fallback_declined_count());
    assert!(boolean_with_options(&mut topo, BooleanOp::Cut, part, pin, decline()).is_err());
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

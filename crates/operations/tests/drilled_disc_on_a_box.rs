//! A drilled disc fused flat against a box that swallows its middle: the
//! box's side lies on the disc's back face, its top and sides cross the
//! disc, and every hole lies inside the box, so the union is the box with
//! the disc's arch standing out of it, and no hole survives.
//!
//! The disc's back face faces against its rim circle's axis, so its rim
//! arcs reach the split with pcurves running in the wire's order while
//! their flag says reversed. The pieces' outlines were read backwards along
//! those arcs, the arch's outline swallowed the hole, and the hole went to
//! the arch: a result that failed validation was accepted. With two or more
//! holes the face took the chord-read arrangement, whose long rim arcs'
//! chords cross the box's top, and fell back to the mesh boolean.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use std::f64::consts::{FRAC_PI_2, PI};

/// Disc r = 25 along x (x 46..52, centre (0, 45) in y, z), its holes
/// r = 1.7 through it, fused with a box x 46..58, y -15..15, z -155..60.
fn fuse(holes: &[(f64, f64)], turn: f64) -> (u64, Topology, brepkit_topology::solid::SolidId) {
    let mut topo = Topology::new();
    let before = mesh_fallback_count();
    let mut disc = make_cylinder(&mut topo, 25.0, 6.0).unwrap();
    let lay =
        Mat4::translation(46.0, 0.0, 45.0) * Mat4::rotation_y(FRAC_PI_2) * Mat4::rotation_z(turn);
    transform_solid(&mut topo, disc, &lay).unwrap();
    for &(y, z) in holes {
        let drill = make_cylinder(&mut topo, 1.7, 10.0).unwrap();
        let at = Mat4::translation(44.0, y, z) * Mat4::rotation_y(FRAC_PI_2);
        transform_solid(&mut topo, drill, &at).unwrap();
        disc = boolean(&mut topo, BooleanOp::Cut, disc, drill).unwrap();
    }
    let block = make_box(&mut topo, 12.0, 30.0, 215.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(46.0, -15.0, -155.0)).unwrap();
    let fused = boolean(&mut topo, BooleanOp::Fuse, disc, block).unwrap();
    (mesh_fallback_count() - before, topo, fused)
}

#[test]
fn drilled_disc_against_a_box_fuses_exactly() {
    // The box, and the disc's arch outside it: the disc less its part with
    // |y| <= 15 and z <= 60, whose section is 750 + 625 asin(3/5).
    let want = 6.0f64.mul_add(
        625.0f64.mul_add(PI, -625.0f64.mul_add((0.6f64).asin(), 750.0)),
        77400.0,
    );
    let sets: [&[(f64, f64)]; 4] = [
        &[(0.0, 30.0)],
        &[(-5.0, 53.66)],
        &[(5.0, 36.34), (10.0, 45.0)],
        &[(5.0, 36.34), (10.0, 45.0), (-5.0, 53.66), (-10.0, 30.0)],
    ];
    for holes in sets {
        for turn in [0.0, 1.0, 2.5] {
            let (fallbacks, topo, fused) = fuse(holes, turn);
            let tag = format!("{} holes, turn {turn}", holes.len());
            assert_eq!(fallbacks, 0, "{tag}: mesh fallback");
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{tag}: {:?}", report.issues);
            let volume = solid_volume(&topo, fused, 0.001).unwrap();
            assert!(
                (volume - want).abs() < 1e-6 * want,
                "{tag}: volume {volume}, want {want}"
            );
            let mesh = tessellate_solid(&topo, fused, 0.05).unwrap();
            assert!(is_watertight(&mesh), "{tag}: open mesh");
        }
    }
}

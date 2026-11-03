//! A torus of radii 5 and 4 (its tube 1 to 9 from the axis) and a box 2 tall
//! spanning the torus's hole, its corners, edges and (off centre) its middle
//! all in the tube. The box is not inside the torus: the intersect keeps the
//! box less the hole's part of it, the cut the torus less that, in every
//! pose.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

fn placed_box(topo: &mut Topology, lo: (f64, f64, f64), hi: (f64, f64, f64)) -> SolidId {
    let b = make_box(topo, hi.0 - lo.0, hi.1 - lo.1, hi.2 - lo.2).unwrap();
    transform_solid(topo, b, &Mat4::translation(lo.0, lo.1, lo.2)).unwrap();
    b
}

fn posed(topo: &mut Topology, s: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            let m =
                Mat4::translation(1.0, -2.0, 0.5) * Mat4::rotation_x(0.6) * Mat4::rotation_z(0.4);
            transform_solid(topo, s, &m).unwrap();
            s
        }
        "mirrored" => mirror(
            topo,
            s,
            Point3::new(0.3, 0.0, 0.0),
            Vec3::new(1.0, 0.3, 0.2),
        )
        .unwrap(),
        _ => s,
    }
}

#[test]
fn a_box_across_a_torus_hole_is_not_inside_the_torus() {
    let torus_volume = 2.0 * PI * PI * 5.0 * 16.0;
    // The hole inside the box: pi * integral over |z| <= 1 of
    // (5 - sqrt(16 - z^2))^2.
    let hole = PI * (50.0 - 10.0 * (15.0_f64.sqrt() + 16.0 * 0.25_f64.asin()) + 32.0 - 2.0 / 3.0);
    for (label, lo, hi) in [
        ("off centre", (-2.0, -2.0, -1.0), (6.0, 2.0, 1.0)),
        ("centred", (-2.0, -2.0, -1.0), (2.0, 2.0, 1.0)),
    ] {
        let box_volume = (hi.0 - lo.0) * (hi.1 - lo.1) * (hi.2 - lo.2);
        for pose in ["upright", "turned", "mirrored"] {
            for op in [BooleanOp::Intersect, BooleanOp::Cut] {
                let case = format!("{label} box, {pose}, {op:?}");
                let mut topo = Topology::new();
                let torus = make_torus(&mut topo, 5.0, 4.0, 16).unwrap();
                let tool = placed_box(&mut topo, lo, hi);
                let torus = posed(&mut topo, torus, pose);
                let tool = posed(&mut topo, tool, pose);
                let before = mesh_fallback_count();
                let result = boolean(&mut topo, op, torus, tool).unwrap();
                assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{case}: {:?}", report.issues);
                let want = match op {
                    BooleanOp::Intersect => box_volume - hole,
                    _ => torus_volume - (box_volume - hole),
                };
                let volume = solid_volume(&topo, result, 0.001).unwrap();
                assert!(
                    (volume - want).abs() < 1e-6 * want,
                    "{case}: volume {volume}, want {want}"
                );
            }
        }
    }

    // Thin boxes crossing the axis at a slant, their stretch of the axis a
    // few hundredths long, are not inside the torus either: the hole takes
    // about a tenth of each. The general boolean falls back to a mesh on
    // these slivers, so only the intersect's volume is read here, and
    // these run after the exact cases above, whose fallback counter they
    // would otherwise disturb.
    for (lo, hi, turn, about) in [
        ((-2.0, -2.0, -0.01), (5.5, 2.0, 0.01), 0.5, (0.0, 0.0, 0.0)),
        (
            (-2.0, -2.0, -0.001),
            (6.0, 2.0, 0.001),
            0.2,
            (2.0, 0.0, 0.0),
        ),
    ] {
        let case = format!("box {lo:?}..{hi:?} turned {turn} about {about:?}");
        let box_volume = (hi.0 - lo.0) * (hi.1 - lo.1) * (hi.2 - lo.2);
        let mut topo = Topology::new();
        let torus = make_torus(&mut topo, 5.0, 4.0, 16).unwrap();
        let tool = placed_box(&mut topo, lo, hi);
        let m = Mat4::translation(about.0, about.1, about.2)
            * Mat4::rotation_y(turn)
            * Mat4::translation(-about.0, -about.1, -about.2);
        transform_solid(&mut topo, tool, &m).unwrap();
        let result = boolean(&mut topo, BooleanOp::Intersect, torus, tool).unwrap();
        let volume = solid_volume(&topo, result, 0.0001).unwrap();
        assert!(
            volume < 0.95 * box_volume && volume > 0.75 * box_volume,
            "{case}: volume {volume} of a box of {box_volume}"
        );
    }
}

//! A rod with a crossbar fused on its top: a shorter, wider cylinder lying
//! across the rod's end, its axis on the rod's top face, its end caps
//! cutting down through the rod's wall. On each cap the rod's wall leaves
//! two vertical lines, and the cap's rim, split at their feet, keeps a major
//! arc whose chord crosses one of them. The cap's trial arrangement broke
//! that line at the chord and registered the break before its pieces were
//! dropped for the walker's, so the rod's wall split the line there alone
//! and its pieces met nothing. In every turn of the crossbar's seam the fuse
//! is exact, valid and at its integrated volume.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

const ROD_R: f64 = 6.5;
const ROD_H: f64 = 30.0;
const BAR_R: f64 = 10.0;
const BAR_HALF: f64 = 4.5;

/// The rod, and the crossbar's lower half (its upper half stands clear),
/// less the rod's part inside the crossbar's lower half: integrated in x
/// across the rod over strips.
fn fused_volume() -> f64 {
    let n = 20_000;
    let dx = 2.0 * BAR_HALF / f64::from(n);
    let mut in_rod = 0.0;
    for i in 0..n {
        let x = -BAR_HALF + (f64::from(i) + 0.5) * dx;
        let half_y = (ROD_R * ROD_R - x * x).sqrt();
        // The strip of the rod's section at this x, inside the bar's lower
        // half: depth below the axis sqrt(R^2 - y^2), over |y| < half_y.
        let m = 400;
        let dy = 2.0 * half_y / f64::from(m);
        for j in 0..m {
            let y = -half_y + (f64::from(j) + 0.5) * dy;
            in_rod += (BAR_R * BAR_R - y * y).sqrt() * dy * dx;
        }
    }
    let rod = std::f64::consts::PI * ROD_R * ROD_R * ROD_H;
    let bar = std::f64::consts::PI * BAR_R * BAR_R * 2.0 * BAR_HALF;
    rod + bar - in_rod
}

#[test]
fn a_rod_under_a_crossbar_fuses_exactly() {
    let want = fused_volume();
    for turn_rod in [90.0_f64, 270.0] {
        for turn_bar in [0.0_f64, 90.0, 180.0, 270.0] {
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, ROD_R, ROD_H).unwrap();
            transform_solid(&mut topo, rod, &Mat4::rotation_z(turn_rod.to_radians())).unwrap();
            let bar = make_cylinder(&mut topo, BAR_R, 2.0 * BAR_HALF).unwrap();
            transform_solid(
                &mut topo,
                bar,
                &(Mat4::translation(-BAR_HALF, 0.0, ROD_H)
                    * Mat4::rotation_y(FRAC_PI_2)
                    * Mat4::rotation_z(turn_bar.to_radians())),
            )
            .unwrap();
            let before = mesh_fallback_count();
            let fused = boolean(&mut topo, BooleanOp::Fuse, rod, bar).unwrap();
            assert_eq!(
                mesh_fallback_count(),
                before,
                "rod {turn_rod} bar {turn_bar}: mesh fallback"
            );
            let report = validate_solid(&topo, fused).unwrap();
            assert!(
                report.is_valid(),
                "rod {turn_rod} bar {turn_bar}: {:?}",
                report.issues
            );
            let volume = solid_volume(&topo, fused, 0.001).unwrap();
            assert!(
                (volume - want).abs() < 1e-5 * want,
                "rod {turn_rod} bar {turn_bar}: volume {volume}, want {want}"
            );
        }
    }
}

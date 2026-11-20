//! A rod standing across a ring's inner wall: a cylinder of radius 0.2813
//! whose axis lies 4.3375 out, between the ring's bore (4.42) and its
//! outside (4.66), so it stands half in the ring and half in the bore.
//! Each cap of the ring is an annulus that the rod's circle cuts only near
//! its hole, and the ring's inner rim, between the rod's two crossings, is
//! a long arc that the rod's circle leaves at a shallow angle.
//!
//! These fell back to the mesh boolean in most poses: the fitted pcurve of
//! that long rim arc turned away from the arc at its end, so the wire walk
//! read the rim as turning off where the rod's arc does; the annulus outline
//! was sampled too coarsely to tell from the rod's circle; and the annulus,
//! whose interior point (read without its hole) lies in the hole, was taken
//! for an island in the hole and dropped. Every pose is now exact.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

const OUTER: f64 = 4.66;
const BORE: f64 = 4.42;
const ROD: f64 = 0.2813;
const AT: f64 = 4.3375;
const H: f64 = 1.5;

/// The area of a disc of radius `r` whose centre is `d` from the centre of
/// a disc of radius `big`, inside that disc (the two circles crossing).
fn lens(big: f64, r: f64, d: f64) -> f64 {
    let a = ((d * d + big * big - r * r) / (2.0 * d * big)).acos();
    let b = ((d * d + r * r - big * big) / (2.0 * d * r)).acos();
    let k = ((-d + big + r) * (d + big - r) * (d - big + r) * (d + big + r)).sqrt();
    big * big * a + r * r * b - 0.5 * k
}

#[test]
fn a_rod_across_a_ring_s_bore_is_exact_in_every_pose() {
    let in_bore = lens(BORE, ROD, AT) * H;
    let ring = PI * (OUTER * OUTER - BORE * BORE) * H;
    let rod = PI * ROD * ROD * H;
    for (op, want) in [
        (BooleanOp::Fuse, ring + in_bore),
        (BooleanOp::Cut, in_bore),
        (BooleanOp::Intersect, rod - in_bore),
    ] {
        for bore_turn in [0.0, 1.0] {
            for rod_turn in [0.0, 2.0] {
                let mut topo = Topology::new();
                let outer = make_cylinder(&mut topo, OUTER, H).unwrap();
                let bore = make_cylinder(&mut topo, BORE, H).unwrap();
                transform_solid(&mut topo, bore, &Mat4::rotation_z(bore_turn)).unwrap();
                let ring_id = boolean(&mut topo, BooleanOp::Cut, outer, bore).unwrap();
                let rod_id = make_cylinder(&mut topo, ROD, H).unwrap();
                transform_solid(&mut topo, rod_id, &Mat4::rotation_z(rod_turn)).unwrap();
                transform_solid(
                    &mut topo,
                    rod_id,
                    &Mat4::translation(AT * 0.8, AT * 0.6, 0.0),
                )
                .unwrap();

                let pose = format!("{op:?}, bore turned {bore_turn}, rod turned {rod_turn}");
                let before = mesh_fallback_count();
                let result = boolean(&mut topo, op, rod_id, ring_id).unwrap();
                assert_eq!(mesh_fallback_count(), before, "{pose}: mesh fallback");
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{pose}: {:?}", report.issues);
                let volume = solid_volume(&topo, result, 1e-4).unwrap();
                assert!(
                    (volume - want).abs() < 1e-6 * want,
                    "{pose}: volume {volume}, want {want}"
                );
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{pose}: open mesh");
            }
        }
    }
}

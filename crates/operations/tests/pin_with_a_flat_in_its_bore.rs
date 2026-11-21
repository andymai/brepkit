//! A pin with a flat on its side, fused into a bore of its own radius, its
//! ends flush with the plate's faces. On each face the flat's chord crosses
//! the bore's rim, which leaves the rim in two loops, the chord between
//! them: the pin's end and the sliver of bore beside the flat. Both are
//! opening on the plate's face, but the face took them as two holes using
//! the chord twice, and the fuse fell back to the mesh boolean.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

const R: f64 = 2.5;
const H: f64 = 6.0;

/// A 12 x 12 x 6 plate bored through its middle at radius R, fused with a
/// pin of radius R as long as the plate is thick, flattened `sag` deep and
/// turned `turn` about its axis.
fn fuse(sag: f64, turn: f64) -> (u64, Topology, brepkit_topology::solid::SolidId) {
    let mut topo = Topology::new();
    let before = mesh_fallback_count();
    let mut plate = make_box(&mut topo, 12.0, 12.0, H).unwrap();
    transform_solid(&mut topo, plate, &Mat4::translation(-6.0, -6.0, 0.0)).unwrap();
    let drill = make_cylinder(&mut topo, R, H + 2.0).unwrap();
    transform_solid(&mut topo, drill, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
    plate = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
    let mut pin = make_cylinder(&mut topo, R, H).unwrap();
    let flat = make_box(&mut topo, 2.0, 2.0 * R + 2.0, H + 2.0).unwrap();
    transform_solid(&mut topo, flat, &Mat4::translation(R - sag, -R - 1.0, -1.0)).unwrap();
    pin = boolean(&mut topo, BooleanOp::Cut, pin, flat).unwrap();
    transform_solid(&mut topo, pin, &Mat4::rotation_z(turn)).unwrap();
    let fused = boolean(&mut topo, BooleanOp::Fuse, plate, pin).unwrap();
    (mesh_fallback_count() - before, topo, fused)
}

#[test]
fn a_flatted_pin_in_its_own_bore_fuses_exactly() {
    for sag in [1e-3, 0.05, 0.4] {
        // The plate less the sliver of bore the flat leaves open.
        let d = R - sag;
        let sliver = R * R * (d / R).acos() - d * (R * R - d * d).sqrt();
        let want = H.mul_add(-sliver, 144.0 * H);
        for turn in [0.0, 0.7, 2.0, 4.1] {
            let tag = format!("sag {sag}, turn {turn}");
            let (fallbacks, topo, fused) = fuse(sag, turn);
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

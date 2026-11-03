//! A plate with a round through hole, fused with a slab standing on the
//! plate across the hole. The slab's bottom edges lie on the plate's top
//! and are cut by the hole's rim into chords sharing both ends with the
//! rim's arcs: the fuse is exact, valid and at its closed-form volume in
//! every pose, wherever the hole's seam lies.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const HOLE_R: f64 = 5.0;
const SLAB_W: f64 = 3.0;
const SLAB_D: f64 = 30.0;
const SLAB_H: f64 = 20.0;

fn centred_box(topo: &mut Topology, size: (f64, f64, f64), c: (f64, f64, f64)) -> SolidId {
    let b = make_box(topo, size.0, size.1, size.2).unwrap();
    let to = Mat4::translation(c.0 - size.0 / 2.0, c.1 - size.1 / 2.0, c.2 - size.2 / 2.0);
    transform_solid(topo, b, &to).unwrap();
    b
}

fn posed(topo: &mut Topology, s: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            let m =
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6);
            transform_solid(topo, s, &m).unwrap();
            s
        }
        "mirrored" => mirror(
            topo,
            s,
            Point3::new(0.3, 0.0, 0.0),
            Vec3::new(1.0, 0.3, 0.0),
        )
        .unwrap(),
        _ => s,
    }
}

#[test]
#[ignore = "ready repro: the chord the hole's rim cuts from the slab's bottom edge welds with the rim's arc (roadmap OPEN)"]
fn a_slab_across_a_through_hole_fuses_exactly() {
    for pose in ["upright", "turned", "mirrored"] {
        for spin in [0.0_f64, 45.0, 90.0, 180.0, 270.0] {
            let case = format!("{pose}, hole seam turned {spin}");
            let mut topo = Topology::new();
            // Plate x -30..30, y -20..20, z 0..5, holed about (20, 0); slab
            // x 18.5..21.5, y -15..15, z 5..25.
            let plate = centred_box(&mut topo, (60.0, 40.0, 5.0), (0.0, 0.0, 2.5));
            let tool = make_cylinder(&mut topo, HOLE_R, 7.0).unwrap();
            let m = Mat4::translation(20.0, 0.0, -1.0) * Mat4::rotation_z(spin.to_radians());
            transform_solid(&mut topo, tool, &m).unwrap();
            let slab = centred_box(
                &mut topo,
                (SLAB_W, SLAB_D, SLAB_H),
                (20.0, 0.0, 5.0 + SLAB_H / 2.0),
            );
            let plate = posed(&mut topo, plate, pose);
            let tool = posed(&mut topo, tool, pose);
            let slab = posed(&mut topo, slab, pose);

            let holed = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
            let before = mesh_fallback_count();
            let fused = boolean(&mut topo, BooleanOp::Fuse, holed, slab).unwrap();
            assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");

            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let faces = solid_faces(&topo, fused).unwrap();
            let surfaces: Vec<&FaceSurface> = faces
                .iter()
                .map(|&f| topo.face(f).unwrap().surface())
                .collect();
            assert!(
                !surfaces.iter().any(|s| matches!(s, FaceSurface::Nurbs(_))),
                "{case}: a NURBS face"
            );
            assert!(
                surfaces
                    .iter()
                    .any(|s| matches!(s, FaceSurface::Cylinder(_))),
                "{case}: the hole lost its wall"
            );

            let want = 60.0 * 40.0 * 5.0 - PI * HOLE_R * HOLE_R * 5.0 + SLAB_W * SLAB_D * SLAB_H;
            let volume = solid_volume(&topo, fused, 0.001).unwrap();
            assert!(
                (volume - want).abs() < 1e-6 * want,
                "{case}: volume {volume}, want {want}"
            );
        }
    }
}

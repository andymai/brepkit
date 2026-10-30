//! A plate with a boss on it, fused with a slab that runs through the boss
//! down to the plate. The slab's sides cross the rim where the boss meets the
//! plate, which is a hole in the plate's top face, so the slab's footprint on
//! that face is no loop inside it: the strip over the boss's foot is not the
//! face's material. The fuse is exact, valid and at its closed-form volume in
//! every pose, wherever the round boss's seam lies, for a slab standing on the
//! plate or sunk into it, and for a square boss too.
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

const BOSS_R: f64 = 5.0;
const BOSS_H: f64 = 15.0;
const SLAB_W: f64 = 3.0;
const SLAB_D: f64 = 30.0;
const SLAB_TOP: f64 = 25.0;

#[derive(Debug, Clone, Copy)]
enum Boss {
    /// Round, turned about its own axis by this many degrees.
    Round(f64),
    /// Square, `2 * BOSS_R` on a side.
    Square,
}

#[derive(Debug, Clone, Copy)]
enum Pose {
    Upright,
    Turned,
    Mirrored,
}

fn centred_box(topo: &mut Topology, size: (f64, f64, f64), c: (f64, f64, f64)) -> SolidId {
    let b = make_box(topo, size.0, size.1, size.2).unwrap();
    let to = Mat4::translation(c.0 - size.0 / 2.0, c.1 - size.1 / 2.0, c.2 - size.2 / 2.0);
    transform_solid(topo, b, &to).unwrap();
    b
}

fn posed(topo: &mut Topology, s: SolidId, pose: Pose) -> SolidId {
    match pose {
        Pose::Upright => s,
        Pose::Turned => {
            let m = Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_z(0.6);
            transform_solid(topo, s, &m).unwrap();
            s
        }
        Pose::Mirrored => mirror(
            topo,
            s,
            Point3::new(0.3, 0.0, 0.0),
            Vec3::new(1.0, 0.3, 0.0),
        )
        .unwrap(),
    }
}

/// The boss's cross-section within the slab's width, `|x| <= SLAB_W / 2`
/// about its axis.
fn boss_strip_area(boss: Boss) -> f64 {
    let h = SLAB_W / 2.0;
    match boss {
        Boss::Round(_) => {
            let r2 = BOSS_R * BOSS_R;
            let f = |x: f64| x * (r2 - x * x).sqrt() + r2 * (x / BOSS_R).asin();
            f(h) - f(-h)
        }
        Boss::Square => SLAB_W * 2.0 * BOSS_R,
    }
}

fn boss_area(boss: Boss) -> f64 {
    match boss {
        Boss::Round(_) => PI * BOSS_R * BOSS_R,
        Boss::Square => 4.0 * BOSS_R * BOSS_R,
    }
}

#[test]
fn a_slab_through_a_boss_to_its_plate_fuses_exactly() {
    let bosses = [
        Boss::Round(0.0),
        Boss::Round(45.0),
        Boss::Round(90.0),
        Boss::Round(180.0),
        Boss::Round(270.0),
        Boss::Square,
    ];
    for pose in [Pose::Upright, Pose::Turned, Pose::Mirrored] {
        for boss_kind in bosses {
            // The slab standing on the plate's top face, and sunk into it.
            for slab_bottom in [5.0, 4.5] {
                let case = format!("{pose:?} {boss_kind:?} slab from z = {slab_bottom}");
                let mut topo = Topology::new();
                // Plate x -30..30, y -20..20, z 0..5; boss about (20, 0) up to
                // z = 20; slab x 18.5..21.5, y -15..15, up to z = 25.
                let plate = centred_box(&mut topo, (60.0, 40.0, 5.0), (0.0, 0.0, 2.5));
                let boss = match boss_kind {
                    Boss::Round(spin) => {
                        let c = make_cylinder(&mut topo, BOSS_R, BOSS_H).unwrap();
                        let m =
                            Mat4::translation(20.0, 0.0, 5.0) * Mat4::rotation_z(spin.to_radians());
                        transform_solid(&mut topo, c, &m).unwrap();
                        c
                    }
                    Boss::Square => centred_box(
                        &mut topo,
                        (2.0 * BOSS_R, 2.0 * BOSS_R, BOSS_H),
                        (20.0, 0.0, 5.0 + BOSS_H / 2.0),
                    ),
                };
                let slab = centred_box(
                    &mut topo,
                    (SLAB_W, SLAB_D, SLAB_TOP - slab_bottom),
                    (20.0, 0.0, f64::midpoint(SLAB_TOP, slab_bottom)),
                );
                let plate = posed(&mut topo, plate, pose);
                let boss = posed(&mut topo, boss, pose);
                let slab = posed(&mut topo, slab, pose);

                let body = boolean(&mut topo, BooleanOp::Fuse, plate, boss).unwrap();
                let before = mesh_fallback_count();
                let fused = boolean(&mut topo, BooleanOp::Fuse, body, slab).unwrap();
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
                if matches!(boss_kind, Boss::Round(_)) {
                    assert!(
                        surfaces
                            .iter()
                            .any(|s| matches!(s, FaceSurface::Cylinder(_))),
                        "{case}: the boss lost its cylinder"
                    );
                }

                // The slab adds its part above the plate less the strip of it
                // inside the boss.
                let want = 60.0 * 40.0 * 5.0
                    + boss_area(boss_kind) * BOSS_H
                    + SLAB_W * SLAB_D * (SLAB_TOP - 5.0)
                    - boss_strip_area(boss_kind) * BOSS_H;
                let volume = solid_volume(&topo, fused, 0.001).unwrap();
                assert!(
                    (volume - want).abs() < 1e-6 * want,
                    "{case}: volume {volume}, want {want}"
                );
            }
        }
    }
}

//! A narrow frustum standing on a wide frustum's base, inside it, whose wall
//! leaves the wide frustum's wall: their axes are parallel, and the section
//! is a closed loop, a hole in the wide wall that winds the narrow one. It
//! comes in two branches that meet where the walls' circles touch, at both
//! ends. In every pose each operation is exact, valid and watertight, and
//! holds the volume integrated from the lens-shaped sections.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::make_cone;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const OFFSET: f64 = 59.0;
const HEIGHT: f64 = 84.0;

/// The area two coplanar discs of radii `r1` and `r2`, `d` apart, share.
fn lens(r1: f64, r2: f64, d: f64) -> f64 {
    if r1 <= 0.0 || r2 <= 0.0 || d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        return PI * r1.min(r2).powi(2);
    }
    let a = r1 * r1 * ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos();
    let b = r2 * r2 * ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos();
    let c = 0.5 * ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).sqrt();
    a + b - c
}

/// The volume the two frustums share: Simpson's rule over the heights of
/// the lens their sections make.
fn shared() -> f64 {
    let n = 200_000;
    let h = HEIGHT / f64::from(n);
    let mut total = 0.0;
    for i in 0..=n {
        let z = h * f64::from(i);
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        total += w * lens(72.0 - 22.0 * z / HEIGHT, 10.0 - 2.0 * z / HEIGHT, OFFSET);
    }
    total * h / 3.0
}

fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
    match pose {
        "turned" => {
            let m =
                Mat4::translation(4.0, -3.0, 2.0) * Mat4::rotation_x(0.7) * Mat4::rotation_z(0.3);
            transform_solid(topo, solid, &m).unwrap();
            solid
        }
        "mirrored" => mirror(
            topo,
            solid,
            Point3::new(3.0, 0.0, 0.0),
            Vec3::new(1.0, 0.2, 0.1),
        )
        .unwrap(),
        _ => solid,
    }
}

/// `make_cone(72, 50, 84)` and `make_cone(10, 8, 84)` 59 from its axis, at
/// every 16th of a turn round it. The walls' circles touch from inside at
/// z = 12.6 and from outside at z = 80.5, so the loop stays clear of both
/// frustums' rims; at the turns 0 and 1/2 its ends lie on the narrow
/// frustum's seam.
#[test]
fn a_frustum_through_a_frustums_wall_is_exact() {
    let wide = PI * HEIGHT / 3.0 * 72.0f64.mul_add(72.0 + 50.0, 2500.0);
    let narrow = PI * HEIGHT / 3.0 * 10.0f64.mul_add(10.0 + 8.0, 64.0);
    let both = shared();
    for pose in ["upright", "turned", "mirrored"] {
        for k in 0..16 {
            let a = 2.0 * PI * f64::from(k) / 16.0;
            for (op, truth) in [
                (BooleanOp::Cut, wide - both),
                (BooleanOp::Intersect, both),
                (BooleanOp::Fuse, wide + narrow - both),
            ] {
                let label = format!("at {a:.4}, {pose} {op:?}");
                let mut topo = Topology::new();
                let large = make_cone(&mut topo, 72.0, 50.0, HEIGHT).unwrap();
                let small = make_cone(&mut topo, 10.0, 8.0, HEIGHT).unwrap();
                let at = Mat4::translation(OFFSET * a.cos(), OFFSET * a.sin(), 0.0);
                transform_solid(&mut topo, small, &at).unwrap();
                let (large, small) = (posed(&mut topo, large, pose), posed(&mut topo, small, pose));
                let opts = BooleanOptions {
                    mesh_fallback: false,
                    ..BooleanOptions::default()
                };
                let result = boolean_with_options(&mut topo, op, large, small, opts);
                let result = result.expect(&label);
                let faces = solid_faces(&topo, result).unwrap();
                let cones = faces
                    .iter()
                    .filter(|&&f| topo.face(f).unwrap().surface().type_tag() == "cone")
                    .count();
                assert!(
                    faces.len() <= 6 && cones == 2,
                    "{label}: {} faces, {cones} cones",
                    faces.len()
                );
                assert!(
                    validate_solid(&topo, result).unwrap().is_valid(),
                    "{label}: invalid"
                );
                let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
                assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                let volume = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (volume - truth).abs() < 1e-7 * truth,
                    "{label}: volume {volume}, truth {truth}"
                );
            }
        }
    }
}

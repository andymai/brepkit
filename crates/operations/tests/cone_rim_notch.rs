//! A frustum standing on z = 0 with a smaller solid standing on the same
//! plane in its wall, the smaller solid's axis 70 from the frustum's at any
//! angle round it, so the smaller solid notches the frustum's base rim. The
//! sections cross either solid's seam at some angles, and the smaller cone's
//! section arches up from the rim to where the two walls' circles touch. In
//! every pose each operation is exact, valid and watertight, and holds the
//! volume integrated from the lens-shaped sections.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_cone, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

const OFFSET: f64 = 70.0;
const HEIGHT: f64 = 44.0;

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

/// The volume the frustum and the smaller solid (radius 10 at its base,
/// `top` at its top) share: Simpson's rule over the heights of the lens
/// their sections make.
fn shared(top: f64) -> f64 {
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
        let big = 72.0 - 22.0 * z / 84.0;
        let small = 10.0 + (top - 10.0) * z / HEIGHT;
        total += w * lens(big, small, OFFSET);
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

/// The frustum `make_cone(72, 50, 84)` with a cylinder of radius 10, or a
/// cone of radius 10 to 5, 44 tall, at every 64th of a turn round the
/// frustum's axis and at the angle 2, in `pose`.
fn assert_notch_is_exact(cylinder: bool, pose: &str) {
    let top: f64 = if cylinder { 10.0 } else { 5.0 };
    let frustum = PI * 84.0 / 3.0 * (72.0f64.mul_add(72.0 + 50.0, 2500.0));
    let small = PI * HEIGHT / 3.0 * top.mul_add(top + 10.0, 100.0);
    let both = shared(top);
    let wall = if cylinder { "cylinder" } else { "cone" };
    let angles = (0..64)
        .map(|k| 2.0 * PI * f64::from(k) / 64.0)
        .chain(std::iter::once(2.0));
    for a in angles {
        for (op, truth) in [
            (BooleanOp::Cut, frustum - both),
            (BooleanOp::Intersect, both),
            (BooleanOp::Fuse, frustum + small - both),
        ] {
            let label = format!("{wall} at {a:.4}, {pose} {op:?}");
            let mut topo = Topology::new();
            let large = make_cone(&mut topo, 72.0, 50.0, 84.0).unwrap();
            let notch = if cylinder {
                make_cylinder(&mut topo, 10.0, HEIGHT).unwrap()
            } else {
                make_cone(&mut topo, 10.0, top, HEIGHT).unwrap()
            };
            let at = Mat4::translation(OFFSET * a.cos(), OFFSET * a.sin(), 0.0);
            transform_solid(&mut topo, notch, &at).unwrap();
            let (large, notch) = (posed(&mut topo, large, pose), posed(&mut topo, notch, pose));
            let opts = BooleanOptions {
                mesh_fallback: false,
                ..BooleanOptions::default()
            };
            let result = boolean_with_options(&mut topo, op, large, notch, opts);
            let result = result.expect(&label);
            let faces = solid_faces(&topo, result).unwrap();
            let tags: Vec<&str> = faces
                .iter()
                .map(|&f| topo.face(f).unwrap().surface().type_tag())
                .collect();
            assert!(
                faces.len() <= 5 && tags.contains(&"cone") && tags.contains(&wall),
                "{label}: surfaces {tags:?}"
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

#[test]
fn a_cylinder_notching_a_frustum_rim_is_exact_upright() {
    assert_notch_is_exact(true, "upright");
}

#[test]
fn a_cylinder_notching_a_frustum_rim_is_exact_turned() {
    assert_notch_is_exact(true, "turned");
}

#[test]
fn a_cylinder_notching_a_frustum_rim_is_exact_mirrored() {
    assert_notch_is_exact(true, "mirrored");
}

#[test]
fn a_cone_notching_a_frustum_rim_is_exact_upright() {
    assert_notch_is_exact(false, "upright");
}

#[test]
fn a_cone_notching_a_frustum_rim_is_exact_turned() {
    assert_notch_is_exact(false, "turned");
}

#[test]
fn a_cone_notching_a_frustum_rim_is_exact_mirrored() {
    assert_notch_is_exact(false, "mirrored");
}

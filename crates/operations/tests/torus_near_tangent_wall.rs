//! A box whose wall lies just inside a torus's outer equator, where the
//! wall's plane is nearly tangent to the tube. The plane meets the tube in a
//! small oval round the equator, and the box's edge across the wall crosses
//! the oval. The oval is a fitted spline: near the tangent, a sample a
//! little off both surfaces lies far off their intersection, since the two
//! surfaces meet at a small angle. The edge's exact crossings lay 1.7e-5
//! off the oval on a ring of radii 400 and 150, past what the trim took for
//! the oval's error, so it left the oval uncut at the box's edge.
//!
//! The same scene at a hundredth of the size, with the wall a hundredth as
//! deep, holds the same angles: its piece has 1e-6 of the volume.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::{make_box, make_torus};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;

/// Tanh-sinh quadrature over `[a, b]`, for an integrand with square-root
/// ends.
fn tanh_sinh(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    let h = 1.0 / 64.0;
    let half = 0.5 * (b - a);
    let mut sum = 0.0;
    for k in -240..=240 {
        let t = f64::from(k) * h;
        let s = 0.5 * PI * t.sinh();
        let weight = 0.5 * PI * t.cosh() / (s.cosh() * s.cosh());
        let gap = half / (s.abs().exp() * s.cosh());
        let x = if s >= 0.0 { b - gap } else { a + gap };
        if x > a && x < b {
            sum += weight * f(x);
        }
    }
    sum * h * half
}

/// The ring's part with `x > c` and `y > y0`: at the distance `rho` from
/// the axis, the turn between `asin(y0 / rho)` and `acos(c / rho)` times the
/// tube's height there.
fn piece(major: f64, minor: f64, c: f64, y0: f64) -> f64 {
    let f = |rho: f64| {
        let height = 2.0
            * minor
                .mul_add(minor, -(rho - major) * (rho - major))
                .max(0.0)
                .sqrt();
        let turn = (c / rho).min(1.0).acos() - (y0 / rho).clamp(-1.0, 1.0).asin();
        rho * turn.max(0.0) * height
    };
    tanh_sinh(&f, c.hypot(y0), major + minor)
}

/// The piece of the ring inside the box, for a ring of radii `400 scale`
/// and `150 scale`, the box's wall `0.01 scale` inside the outer equator
/// and its edge at `y = y0 scale`.
fn check(scale: f64, y0: f64, op: BooleanOp, pose: &Mat4, label: &str) -> f64 {
    let label = format!("{label}, scale {scale}, y0 {y0}, {op:?}");
    let (major, minor) = (400.0 * scale, 150.0 * scale);
    let c = 0.01f64.mul_add(-scale, major + minor);
    let (y0, size) = (y0 * scale, 100.0 * scale);
    let mut topo = Topology::new();
    let ring = make_torus(&mut topo, major, minor, 32).unwrap();
    let block = make_box(&mut topo, size, size, 2.0 * size).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(c, y0, -size)).unwrap();
    for solid in [ring, block] {
        transform_solid(&mut topo, solid, pose).unwrap();
    }
    let exact = BooleanOptions {
        mesh_fallback: false,
        ..BooleanOptions::default()
    };
    let result = boolean_with_options(&mut topo, op, ring, block, exact)
        .unwrap_or_else(|e| panic!("{label}: {e}"));
    let report = validate_solid(&topo, result).unwrap();
    assert!(report.is_valid(), "{label}: invalid: {report:?}");
    // The cut keeps the whole ring, which a fine mesh would take long over.
    let deflection = if op == BooleanOp::Intersect {
        0.01
    } else {
        1.0
    } * scale;
    assert!(
        is_watertight(&tessellate_solid(&topo, result, deflection).unwrap()),
        "{label}: mesh not watertight"
    );

    // Measured where the ring was made: the turned piece is small against
    // its distance from the origin, and the measure's sums cancel there.
    if let Ok(back) = pose.inverse() {
        transform_solid(&mut topo, result, &back).unwrap();
    }
    let inside = piece(major, minor, c, y0);
    let whole = 2.0 * PI * PI * major * minor * minor;
    let volume = solid_volume(&topo, result, 1e-4 * scale).unwrap();
    // The section near the tangent is a fitted spline, 1e-5 of the oval's
    // size off the true one. The cut's volume sums the whole ring's faces,
    // whose rounding is near 1e-14 of the ring's volume: 1e-13 of it is
    // left for that.
    let (taken, allowance) = match op {
        BooleanOp::Intersect => (volume, 0.0),
        _ => (whole - volume, 1e-13 * whole),
    };
    assert!(
        (taken - inside).abs() <= 2e-5f64.mul_add(inside, allowance),
        "{label}: the box takes {taken}, want {inside}"
    );
    taken
}

#[test]
fn a_box_wall_near_a_tangent_to_the_tube_ends_the_oval_at_its_edge() {
    let turned =
        Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9);
    for y0 in [0.3, 1.0] {
        for op in [BooleanOp::Intersect, BooleanOp::Cut] {
            for (pose, name) in [(Mat4::identity(), "upright"), (turned, "turned")] {
                let large = check(1.0, y0, op, &pose, name);
                let small = check(0.01, y0, op, &pose, name);
                if op == BooleanOp::Intersect {
                    assert!(
                        (large * 1e-6 - small).abs() <= 1e-5 * small,
                        "y0 {y0}, {name}: {large} at full size against {small} at a hundredth"
                    );
                }
            }
        }
    }
}

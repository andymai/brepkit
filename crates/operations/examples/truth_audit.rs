//! Booleans between primitives and simple tools, each result's volume read
//! against the truth, in three poses. A pose sweep compares poses with each
//! other, so a result wrong in every pose reads alike in all of them; here
//! each is measured against its closed form:
//!
//! ```text
//! cargo run --release --example truth_audit -p brepkit-operations
//! ```
//!
//! Every primitive is a solid of revolution about `z` (a cylinder, a pointed
//! cone, a frustum, a ball and a ring), so its part inside a tool is its
//! section at each height (a disc or an annulus) cut by the tool, integrated
//! over the heights the tool spans: a half-space `x > 0.5`, the corner
//! `x > 1, y > 1.2, z > 0.8`, a rod of radius 0.6 along `y` through
//! `(0.5, ., 1)`, and the slab `1 < z < 2`. Each line reads the case, the
//! truth, and per pose (upright, turned, mirrored through a slanted plane)
//! `x` for an exact result or `F` for the mesh boolean's, `~` when
//! `validate_solid` rejects it, `!` when its volume misses the truth by more
//! than 1e-4 of it, and that relative error.
#![allow(
    clippy::unwrap_used,
    clippy::print_stdout,
    clippy::cast_precision_loss,
    missing_docs
)]

use std::f64::consts::PI;
use std::fmt::Write as _;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

const PRIMITIVES: [&str; 5] = ["cylinder", "cone", "frustum", "ball", "ring"];
const TOOLS: [&str; 4] = ["half x>0.5", "corner", "rod along y", "slab z 1..2"];

fn primitive(topo: &mut Topology, k: usize) -> SolidId {
    let down = Mat4::translation(0.0, 0.0, -3.0);
    match k {
        0 => {
            let c = make_cylinder(topo, 2.0, 6.0).unwrap();
            transform_solid(topo, c, &down).unwrap();
            c
        }
        1 | 2 => {
            let c = make_cone(topo, 3.0, if k == 1 { 0.0 } else { 1.5 }, 6.0).unwrap();
            transform_solid(topo, c, &down).unwrap();
            c
        }
        3 => make_sphere(topo, 3.0, 32).unwrap(),
        _ => make_torus(topo, 4.0, 1.5, 32).unwrap(),
    }
}

fn tool(topo: &mut Topology, k: usize) -> SolidId {
    let (solid, place) = match k {
        0 => (
            make_box(topo, 20.0, 20.0, 20.0).unwrap(),
            Mat4::translation(0.5, -10.0, -10.0),
        ),
        1 => (
            make_box(topo, 10.0, 10.0, 10.0).unwrap(),
            Mat4::translation(1.0, 1.2, 0.8),
        ),
        2 => (
            make_cylinder(topo, 0.6, 20.0).unwrap(),
            Mat4::translation(0.5, 10.0, 1.0) * Mat4::rotation_x(std::f64::consts::FRAC_PI_2),
        ),
        _ => (
            make_box(topo, 20.0, 20.0, 1.0).unwrap(),
            Mat4::translation(-10.0, -10.0, 1.0),
        ),
    };
    transform_solid(topo, solid, &place).unwrap();
    solid
}

/// The primitive's volume.
fn whole(k: usize) -> f64 {
    [24.0 * PI, 18.0 * PI, 31.5 * PI, 36.0 * PI, 18.0 * PI * PI][k]
}

/// The primitive's heights.
fn heights(k: usize) -> (f64, f64) {
    if k == 4 { (-1.5, 1.5) } else { (-3.0, 3.0) }
}

/// Inner and outer radius of the primitive's section at `z`.
fn section(k: usize, z: f64) -> (f64, f64) {
    match k {
        0 => (0.0, 2.0),
        1 => (0.0, 0.5 * (3.0 - z)),
        2 => (0.0, 0.25f64.mul_add(-(z + 3.0), 3.0)),
        3 => (0.0, z.mul_add(-z, 9.0).max(0.0).sqrt()),
        _ => {
            let s = z.mul_add(-z, 2.25).max(0.0).sqrt();
            (4.0 - s, 4.0 + s)
        }
    }
}

/// `∫ 2 sqrt(r^2 - x^2) dx` over `[a, b]` clamped to the disc of radius `r`.
fn strip(r: f64, a: f64, b: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    let f = |x: f64| {
        let x = x.clamp(-r, r);
        x.mul_add((r * r - x * x).max(0.0).sqrt(), r * r * (x / r).asin())
    };
    let (a, b) = (a.max(-r), b.min(r));
    if b > a { f(b) - f(a) } else { 0.0 }
}

/// The area of the disc of radius `r` inside the tool's section.
fn disc_in_tool(t: usize, r: f64) -> f64 {
    match t {
        0 => strip(r, 0.5, r),
        1 => {
            if r * r <= 2.44 {
                return 0.0;
            }
            let x_max = (r * r - 1.44).sqrt();
            0.5f64.mul_add(strip(r, 1.0, x_max), -1.2 * (x_max - 1.0))
        }
        _ => PI * r * r,
    }
}

fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let h = (b - a) / n as f64;
    let inner: f64 = (1..n)
        .map(|i| f(h.mul_add(i as f64, a)) * if i % 2 == 1 { 4.0 } else { 2.0 })
        .sum();
    (f(a) + f(b) + inner) * h / 3.0
}

/// The primitive's volume inside the tool.
fn inside(p: usize, t: usize) -> f64 {
    let (lo, hi) = heights(p);
    if t == 2 {
        // Across the rod at each height of its axis's section, in its angle.
        return simpson(
            |phi| {
                let (z, half) = (0.6f64.mul_add(phi.sin(), 1.0), 0.6 * phi.cos());
                if z < lo || z > hi {
                    return 0.0;
                }
                let (ri, ro) = section(p, z);
                (strip(ro, 0.5 - half, 0.5 + half) - strip(ri, 0.5 - half, 0.5 + half)) * half
            },
            -PI / 2.0,
            PI / 2.0,
            20_000,
        );
    }
    // Only over the tool's own heights (its bounds are jumps), with
    // `z = lo + (hi - lo)(1 - cos s) / 2` smoothing the ends.
    let (lo, hi) = match t {
        1 => (lo.max(0.8), hi),
        3 => (lo.max(1.0), hi.min(2.0)),
        _ => (lo, hi),
    };
    simpson(
        |s| {
            let z = (hi - lo).mul_add(0.5 * (1.0 - s.cos()), lo);
            let (ri, ro) = section(p, z);
            let area = disc_in_tool(t, ro) - if ri > 0.0 { disc_in_tool(t, ri) } else { 0.0 };
            area * (hi - lo) * 0.5 * s.sin()
        },
        0.0,
        PI,
        40_000,
    )
}

fn main() {
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
    for p in 0..PRIMITIVES.len() {
        for t in 0..TOOLS.len() {
            let within = inside(p, t);
            for op in [BooleanOp::Cut, BooleanOp::Intersect] {
                let truth = if op == BooleanOp::Cut {
                    whole(p) - within
                } else {
                    within
                };
                let mut line = format!(
                    "{:9} {:12} {:9} {truth:>10.5}",
                    PRIMITIVES[p],
                    TOOLS[t],
                    format!("{op:?}")
                );
                for pose in 0..3 {
                    let mut topo = Topology::new();
                    let (mut a, mut b) = (primitive(&mut topo, p), tool(&mut topo, t));
                    if pose == 1 {
                        transform_solid(&mut topo, a, &turn).unwrap();
                        transform_solid(&mut topo, b, &turn).unwrap();
                    } else if pose == 2 {
                        a = mirror(&mut topo, a, at, normal).unwrap();
                        b = mirror(&mut topo, b, at, normal).unwrap();
                    }
                    let before = mesh_fallback_count();
                    let tag = match boolean(&mut topo, op, a, b) {
                        Err(_) if truth.abs() < 1e-9 => "empty".to_string(),
                        Err(_) => "error".to_string(),
                        Ok(s) => {
                            let fallback = mesh_fallback_count() != before;
                            let valid = validate_solid(&topo, s).is_ok_and(|r| r.is_valid());
                            let volume = solid_volume(&topo, s, 0.01).unwrap_or(f64::NAN);
                            let rel = (volume - truth).abs() / truth.abs().max(1e-9);
                            format!(
                                "{}{}{}({rel:.0e})",
                                if fallback { "F" } else { "x" },
                                if valid { "" } else { "~" },
                                if rel > 1e-4 { "!" } else { "" }
                            )
                        }
                    };
                    let _ = write!(line, " {tag:>13}");
                }
                println!("{line}");
            }
        }
    }
}

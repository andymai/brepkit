//! The area of a sphere face that a box trims, measured exactly and by the
//! face's own mesh. A ball of radius 3 meets the box over `x > 1`, `y > 1.2`,
//! `z > 0.8` in a three-sided patch, and the box over the positive octant in
//! a patch whose sides are two meridians and the equator. A tilted rod
//! through the ball keeps two caps whose flux measures the rod's piece
//! exactly.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_math::mat::Mat4;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::tessellate;
use brepkit_operations::transform::transform_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;

const RADIUS: f64 = 3.0;

/// The sphere faces of the ball less or within a box at `corner`, with their
/// exact and meshed areas, in the order the result lists them.
fn sphere_faces(op: BooleanOp, corner: (f64, f64, f64)) -> Vec<(f64, f64)> {
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(
        &mut topo,
        block,
        &Mat4::translation(corner.0, corner.1, corner.2),
    )
    .unwrap();
    let piece = boolean(&mut topo, op, ball, block).unwrap();
    solid_faces(&topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .map(|f| {
            let mesh = tessellate(&topo, f, 0.005).unwrap();
            let meshed: f64 = mesh
                .indices
                .chunks_exact(3)
                .map(|t| {
                    let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
                    (b - a).cross(c - a).length() / 2.0
                })
                .sum();
            (face_area(&topo, f, 0.005).unwrap(), meshed)
        })
        .collect()
}

/// The patch past `x = 1`, `y = 1.2` and `z = 0.8`, projected onto the
/// `xy` plane, where the sphere's area element is `R / z`: across `y` it
/// integrates to `R asin(y / c)` with `c = sqrt(R² - x²)`, leaving one
/// Simpson integral in `x`, substituted where the patch pinches off.
fn corner_patch() -> f64 {
    let x_end = (RADIUS * RADIUS - 0.64 - 1.44).sqrt();
    let across = |x: f64| {
        let c = RADIUS.mul_add(RADIUS, -(x * x)).sqrt();
        let y_end = (RADIUS * RADIUS - 0.64 - x * x).max(0.0).sqrt();
        RADIUS * ((y_end / c).asin() - (1.2 / c).asin())
    };
    let f = |s: f64| across(s.mul_add(-s, x_end)) * 2.0 * s;
    let (n, span) = (800_u32, (x_end - 1.0).sqrt());
    let step = span / f64::from(n);
    let mut sum = f(0.0) + f(span);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step * f64::from(k));
    }
    sum * step / 3.0
}

#[test]
fn sphere_face_bitten_by_a_box_corner() {
    let truth = corner_patch();
    let faces = sphere_faces(BooleanOp::Intersect, (1.0, 1.2, 0.8));
    assert_eq!(faces.len(), 1, "one sphere face");
    let (area, meshed) = faces[0];
    assert!(
        (area - truth).abs() < 1e-9 * truth,
        "area {area}, truth {truth}"
    );
    assert!(
        (meshed - truth).abs() < 1e-2 * truth,
        "mesh area {meshed}, truth {truth}"
    );
}

#[test]
fn sphere_octant() {
    let octant = PI * RADIUS * RADIUS / 2.0;
    let within = sphere_faces(BooleanOp::Intersect, (0.0, 0.0, 0.0));
    assert_eq!(within.len(), 1, "one sphere face");
    // Less the octant, the upper hemisphere keeps three quarters of itself.
    let mut less = sphere_faces(BooleanOp::Cut, (0.0, 0.0, 0.0));
    less.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(less.len(), 2, "two sphere faces");
    for ((area, meshed), truth) in [
        (within[0], octant),
        (less[0], 3.0 * octant),
        (less[1], 4.0 * octant),
    ] {
        assert!(
            (area - truth).abs() < 1e-9 * truth,
            "area {area}, truth {truth}"
        );
        assert!(
            (meshed - truth).abs() < 1e-2 * truth,
            "mesh area {meshed}, truth {truth}"
        );
    }
}

/// The ball above the plane `z = 1.5 + 0.2 x`, clear of its equator: a cap
/// under one tilted circle, `2 pi R (R - d)` of the sphere with `d` the
/// plane's distance from the centre, which its own mesh covers too.
#[test]
fn cap_under_a_tilted_circle() {
    let slope = 0.2_f64;
    let d = 1.5 / slope.hypot(1.0);
    let truth = 2.0 * PI * RADIUS * (RADIUS - d);
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
    let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
    let place = Mat4::translation(0.0, 0.0, 1.5)
        * Mat4::rotation_y(-slope.atan())
        * Mat4::translation(-10.0, -10.0, 0.0);
    transform_solid(&mut topo, lid, &place).unwrap();
    let cap = boolean(&mut topo, BooleanOp::Intersect, ball, lid).unwrap();
    let faces: Vec<_> = solid_faces(&topo, cap)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(faces.len(), 1, "one sphere face");
    let area = face_area(&topo, faces[0], 0.005).unwrap();
    assert!(
        (area - truth).abs() < 1e-9 * truth,
        "area {area}, truth {truth}"
    );
    let mesh = tessellate(&topo, faces[0], 0.005).unwrap();
    let meshed: f64 = mesh
        .indices
        .chunks_exact(3)
        .map(|t| {
            let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize]);
            (b - a).cross(c - a).length() / 2.0
        })
        .sum();
    assert!(
        (meshed - truth).abs() < 1e-2 * truth,
        "mesh area {meshed}, truth {truth}"
    );
}

/// A rod of radius 0.8 through the ball, its axis tilted and 1.08 from the
/// centre: the piece inside is, over the rod's cross-section, the chord
/// through the ball, integrated by Simpson in polar coordinates about the
/// axis. Less it, the ball keeps the rest.
#[test]
fn tilted_rod_through_a_ball() {
    let (rod, tilt, foot) = (0.8_f64, 0.6_f64, (1.0_f64, 0.5_f64));
    // The axis runs along (0, -sin, cos) through (1, 0.5, 0).
    let along = foot.1 * -tilt.sin();
    let axis_gap = (foot.0 * foot.0 + foot.1 * foot.1 - along * along).sqrt();
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let inside = simpson(200, 0.0, rod, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), axis_gap), r * th.sin());
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    for (op, truth) in [
        (BooleanOp::Intersect, inside),
        (BooleanOp::Cut, ball - inside),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let cylinder = make_cylinder(&mut topo, rod, 10.0).unwrap();
        let place = Mat4::translation(foot.0, foot.1, 0.0)
            * Mat4::rotation_x(tilt)
            * Mat4::translation(0.0, 0.0, -5.0);
        transform_solid(&mut topo, cylinder, &place).unwrap();
        let piece = boolean(&mut topo, op, sphere, cylinder).unwrap();
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-7 * truth,
            "{op:?}: volume {volume}, truth {truth}"
        );
    }
}

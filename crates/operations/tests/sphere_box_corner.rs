//! A ball less a box whose corner pokes into it keeps a three-sided pocket:
//! the hemisphere the corner reaches gets a hole bounded by three arcs, which
//! must wind against the hemisphere's outer wire. Each piece is an exact,
//! valid, watertight solid whose volume matches the corner's integral, above
//! or below the equator. The box shortcut's octant feeds a second boolean,
//! and it steps aside when the box's corner lies outside the ball.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::solid_volume;
use brepkit_operations::mirror::mirror;
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const RADIUS: f64 = 3.0;

/// Whether a result is exact rather than a mesh fallback, which is all planes
/// and hundreds of them: the results here keep a sphere face among a few
/// dozen at most. (The fallback counter is process-wide, and other tests in
/// this binary may fall back while one runs.)
fn exact(topo: &Topology, solid: SolidId) -> bool {
    let faces = solid_faces(topo, solid).unwrap();
    faces.len() <= 24
        && faces
            .iter()
            .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
}

/// The ball's piece past `x = a`, `y = b` and `z = c`: across `y` the height
/// `sqrt(R² - x² - y²) - c` integrates in closed form, leaving a Simpson
/// integral in `x`, substituted `x = end - s²` where the piece pinches off.
fn corner_piece(a: f64, b: f64, c: f64) -> f64 {
    let x_end = (RADIUS * RADIUS - b * b - c * c).sqrt();
    let across = |x: f64| {
        let c2 = RADIUS.mul_add(RADIUS, -(x * x));
        let y_end = (c2 - c * c).max(0.0).sqrt();
        let g = |y: f64| {
            0.5 * y.mul_add(
                y.mul_add(-y, c2).max(0.0).sqrt(),
                c2 * (y / c2.sqrt()).asin(),
            ) - c * y
        };
        g(y_end) - g(b)
    };
    let f = |s: f64| across(s.mul_add(-s, x_end)) * 2.0 * s;
    let (n, span) = (800_u32, (x_end - a).sqrt());
    let step = span / f64::from(n);
    let mut sum = f(0.0) + f(span);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step * f64::from(k));
    }
    sum * step / 3.0
}

#[test]
fn ball_less_a_box_corner() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    // A box of side 10 with its near corner at (a, b, c) reaching into the
    // ball, below the equator when `flip` mirrors it through z = 0.
    for (corner, flip) in [
        ((1.0, 1.0, 1.0), false),
        ((1.0, 1.2, 0.8), false),
        ((0.5, 0.5, 0.5), false),
        ((1.5, 0.2, 0.3), false),
        ((1.0, 1.2, 0.8), true),
    ] {
        let (a, b, c) = corner;
        let piece = corner_piece(a, b, c);
        let z0 = if flip { -c - 10.0 } else { c };
        for (op, truth) in [
            (BooleanOp::Cut, ball - piece),
            (BooleanOp::Intersect, piece),
        ] {
            let label = format!("{corner:?} flip {flip} {op:?}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, block, &Mat4::translation(a, b, z0)).unwrap();
            let result = boolean(&mut topo, op, sphere, block).unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            // Just inside the corner, and on the ball's far side.
            let z_in = if flip { -c - 0.05 } else { c + 0.05 };
            let at =
                |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
            let (pocket, far) = if op == BooleanOp::Cut {
                (PointClassification::Outside, PointClassification::Inside)
            } else {
                (PointClassification::Inside, PointClassification::Outside)
            };
            assert_eq!(
                at(Point3::new(a + 0.05, b + 0.05, z_in)),
                pocket,
                "{label}: corner"
            );
            assert_eq!(at(Point3::new(-2.0, -1.0, 0.3)), far, "{label}: far side");
        }
    }
}

/// The ball's octant feeds a second boolean: less a rod of radius 0.4 along
/// `z` through `(1, 1)` it loses the column over the rod's disc (by Simpson in
/// polar coordinates), and less a box corner at `(1, 1, ±1)` exactly the
/// corner's piece, whether the box shortcut built the octant or the boolean
/// engine did (box and ball turned about `z`, out of the shortcut's reach).
#[test]
fn box_octant_feeds_a_second_boolean() {
    let octant = PI * RADIUS.powi(3) / 6.0;
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let column = simpson(200, 0.0, 0.4, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), 1.0), r.mul_add(th.sin(), 1.0));
            RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    for (lower, spin) in [(false, 0.0), (true, 0.0), (false, 0.3), (true, 0.3)] {
        let label = format!("lower {lower} spin {spin}");
        let turn = Mat4::rotation_z(spin);
        let octant_of = |topo: &mut Topology| {
            let sphere = make_sphere(topo, RADIUS, 32).unwrap();
            let block = make_box(topo, 10.0, 10.0, 10.0).unwrap();
            let z0 = if lower { -10.0 } else { 0.0 };
            transform_solid(topo, block, &(turn * Mat4::translation(0.0, 0.0, z0))).unwrap();
            transform_solid(topo, sphere, &turn).unwrap();
            let piece = boolean(topo, BooleanOp::Intersect, sphere, block).unwrap();
            assert!(
                validate_solid(topo, piece).unwrap().is_valid(),
                "{label}: invalid octant"
            );
            piece
        };
        {
            let mut topo = Topology::new();
            let piece = octant_of(&mut topo);
            let rod = make_cylinder(&mut topo, 0.4, 20.0).unwrap();
            transform_solid(&mut topo, rod, &(turn * Mat4::translation(1.0, 1.0, -10.0))).unwrap();
            let result = boolean(&mut topo, BooleanOp::Cut, piece, rod).unwrap();
            assert!(exact(&topo, result), "{label} rod: fell back to a mesh");
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label} rod: invalid"
            );
            let (volume, truth) = (solid_volume(&topo, result, 0.01).unwrap(), octant - column);
            assert!(
                (volume - truth).abs() < 1e-7 * truth,
                "{label} rod: volume {volume}, truth {truth}"
            );
        }
        let mut topo = Topology::new();
        let piece = octant_of(&mut topo);
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let z0 = if lower { -11.0 } else { 1.0 };
        transform_solid(&mut topo, block, &(turn * Mat4::translation(1.0, 1.0, z0))).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, piece, block).unwrap();
        assert!(exact(&topo, result), "{label}: fell back to a mesh");
        assert!(
            validate_solid(&topo, result).unwrap().is_valid(),
            "{label}: invalid"
        );
        // Just inside the removed corner, and the octant's material beside it.
        let side = if lower { -1.0 } else { 1.0 };
        for (p, class) in [
            (
                Point3::new(1.3, 1.3, 1.3 * side),
                PointClassification::Outside,
            ),
            (
                Point3::new(0.6, 0.6, 0.6 * side),
                PointClassification::Inside,
            ),
            (
                Point3::new(1.3, 0.6, 1.3 * side),
                PointClassification::Inside,
            ),
        ] {
            assert_eq!(
                classify_point(
                    &topo,
                    result,
                    turn.mul_point(p),
                    &ClassifyOptions::default()
                )
                .unwrap(),
                class,
                "{label}: {p:?}"
            );
        }
        let (volume, truth) = (
            solid_volume(&topo, result, 0.01).unwrap(),
            octant - corner_piece(1.0, 1.0, 1.0),
        );
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "{label}: volume {volume}, truth {truth}"
        );
    }
}

/// The ball within the box over an octant, both turned about `z`, is the
/// octant from the boolean engine above and below the equator: four faces,
/// valid, `pi R³ / 6`. Below it, the box's top face meets the ball on the
/// faceted equator, whose arc both hemispheres need.
#[test]
fn turned_octants_are_exact() {
    let truth = PI * RADIUS.powi(3) / 6.0;
    for spin in [0.3, 1.1, -0.4] {
        for lower in [false, true] {
            let label = format!("lower {lower} spin {spin}");
            let turn = Mat4::rotation_z(spin);
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let z0 = if lower { -10.0 } else { 0.0 };
            transform_solid(&mut topo, block, &(turn * Mat4::translation(0.0, 0.0, z0))).unwrap();
            transform_solid(&mut topo, sphere, &turn).unwrap();
            let piece = boolean(&mut topo, BooleanOp::Intersect, sphere, block).unwrap();
            assert_eq!(
                solid_faces(&topo, piece).unwrap().len(),
                4,
                "{label}: faces"
            );
            assert!(exact(&topo, piece), "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, piece).unwrap().is_valid(),
                "{label}: invalid"
            );
            let volume = solid_volume(&topo, piece, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-9 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// The ball less, and fused with, the box over an octant, both turned 0.3
/// about `z`, above and below the equator: exact and valid, the ball less its
/// octant or joined to the box by the rest of it. The volume bound covers the
/// chordal equator's measure (the roadmap's sphere measure row).
#[test]
fn ball_cut_and_fused_with_a_turned_octant_box() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let octant = PI * RADIUS.powi(3) / 6.0;
    for (op, truth) in [
        (BooleanOp::Cut, ball - octant),
        (BooleanOp::Fuse, 1000.0 + ball - octant),
    ] {
        for lower in [false, true] {
            let label = format!("{op:?} lower {lower}");
            let turn = Mat4::rotation_z(0.3);
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let z0 = if lower { -10.0 } else { 0.0 };
            transform_solid(&mut topo, block, &(turn * Mat4::translation(0.0, 0.0, z0))).unwrap();
            transform_solid(&mut topo, sphere, &turn).unwrap();
            let result = boolean(&mut topo, op, sphere, block).unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 2e-4 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// A ball turned about an oblique axis, or mirrored through a slanted plane,
/// with the tool moved alongside: a box corner, a slab and a rod stay exact
/// and valid. A turned hemisphere's equator is a rounding sliver of `v`, not
/// an extent; the slab's latitudes nest on one hemisphere; the rod's tunnel
/// leaves a shell whose hemispheres' flux decides its orientation. The
/// corner and slab Cut bounds cover the chordal equator's measure (the
/// roadmap's sphere measure row).
#[test]
fn turned_and_mirrored_balls_stay_exact() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let corner = corner_piece(1.0, 1.2, 0.8);
    // The ball between z = 1 and z = 2.
    let zone = PI * (RADIUS.powi(2) * 1.0 - (8.0 - 1.0) / 3.0);
    // The rod of radius 0.6 along y through (0.5, _, 1): the ball's chord
    // along y over the rod's disc, by Simpson in polar coordinates.
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let rod = simpson(200, 0.0, 0.6, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, z) = (r.mul_add(th.cos(), 0.5), r.mul_add(th.sin(), 1.0));
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + z * z)).sqrt()
        })
    });
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    let at = Point3::new(0.3, 0.0, 0.0);
    let normal = brepkit_math::vec::Vec3::new(1.0, 0.2, 0.1);
    let unit = normal.normalize().unwrap();
    for pose in ["turned", "mirrored"] {
        let place = |p: Point3| {
            if pose == "turned" {
                turn.mul_point(p)
            } else {
                p - unit * (2.0 * (p - at).dot(unit))
            }
        };
        for (tool, op, truth, bound) in [
            ("corner", BooleanOp::Intersect, corner, 1e-7),
            ("corner", BooleanOp::Cut, ball - corner, 2e-4),
            ("slab", BooleanOp::Intersect, zone, 1e-9),
            ("slab", BooleanOp::Cut, ball - zone, 2e-4),
            ("rod", BooleanOp::Intersect, rod, 1e-6),
            ("rod", BooleanOp::Cut, ball - rod, 1e-7),
        ] {
            let label = format!("{pose} {tool} {op:?}");
            let mut topo = Topology::new();
            let mut sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let mut block = match tool {
                "corner" => {
                    let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                    transform_solid(&mut topo, b, &Mat4::translation(1.0, 1.2, 0.8)).unwrap();
                    b
                }
                "slab" => {
                    let b = make_box(&mut topo, 20.0, 20.0, 1.0).unwrap();
                    transform_solid(&mut topo, b, &Mat4::translation(-10.0, -10.0, 1.0)).unwrap();
                    b
                }
                _ => {
                    let c = make_cylinder(&mut topo, 0.6, 20.0).unwrap();
                    let place = Mat4::translation(0.5, 10.0, 1.0)
                        * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
                    transform_solid(&mut topo, c, &place).unwrap();
                    c
                }
            };
            if pose == "turned" {
                transform_solid(&mut topo, sphere, &turn).unwrap();
                transform_solid(&mut topo, block, &turn).unwrap();
            } else {
                sphere = mirror(&mut topo, sphere, at, normal).unwrap();
                block = mirror(&mut topo, block, at, normal).unwrap();
            }
            let result = boolean(&mut topo, op, sphere, block).unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            assert!(
                validate_solid(&topo, result).unwrap().is_valid(),
                "{label}: invalid"
            );
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < bound * truth,
                "{label}: volume {volume}, truth {truth}"
            );
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            // A point in the tool's part of the ball and one in the rest,
            // placed like the solids.
            let (in_tool, in_rest) = match tool {
                "corner" => (Point3::new(1.05, 1.25, 0.85), Point3::new(-2.0, -1.0, 0.3)),
                "slab" => (Point3::new(0.2, -0.3, 1.5), Point3::new(0.2, -0.3, 0.5)),
                _ => (Point3::new(0.5, 0.0, 1.0), Point3::new(-1.5, 0.0, -1.0)),
            };
            let (tool_side, rest_side) = if op == BooleanOp::Cut {
                (PointClassification::Outside, PointClassification::Inside)
            } else {
                (PointClassification::Inside, PointClassification::Outside)
            };
            let at_point =
                |p: Point3| classify_point(&topo, result, place(p), &ClassifyOptions::default());
            assert_eq!(at_point(in_tool).unwrap(), tool_side, "{label}: tool side");
            assert_eq!(at_point(in_rest).unwrap(), rest_side, "{label}: rest");
        }
    }
}

/// A ball of radius 3 at `(1.8, 1.8, 1.8)` within the box over the positive
/// octant meets all three of the box's corner planes, but the corner lies
/// outside the ball, so the region is not an octant: the result keeps the
/// z-extent above `z = 0` over `x, y >= 0` (Simpson in two dimensions,
/// within 2%, a mesh allowed) rather than the octant shortcut's 82.09.
#[test]
fn corner_outside_the_ball_is_not_an_octant() {
    let (c, r) = (1.8_f64, RADIUS);
    let n = 600_u32;
    let step = (c + r) / f64::from(n);
    let weight = |k: u32| {
        if k == 0 || k == n {
            1.0
        } else if k % 2 == 1 {
            4.0
        } else {
            2.0
        }
    };
    let mut sum = 0.0;
    for i in 0..=n {
        let x = step * f64::from(i);
        for j in 0..=n {
            let y = step * f64::from(j);
            let rho2 = (x - c).powi(2) + (y - c).powi(2);
            if rho2 < r * r {
                let h = (r * r - rho2).sqrt();
                sum += weight(i) * weight(j) * (c + h - (c - h).max(0.0));
            }
        }
    }
    let truth = sum * step * step / 9.0;
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, r, 32).unwrap();
    transform_solid(&mut topo, sphere, &Mat4::translation(c, c, c)).unwrap();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let result = boolean(&mut topo, BooleanOp::Intersect, sphere, block).unwrap();
    let volume = solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 2e-2 * truth,
        "volume {volume}, truth {truth}"
    );
}

/// The area of the disc of radius `r` about the axis where `x > a` and
/// `y > b`: across `x`, the chord above `y = b` (or the whole chord where the
/// line misses the disc below it) in closed form.
fn disc_corner_area(r: f64, a: f64, b: f64) -> f64 {
    if r <= 0.0 {
        return 0.0;
    }
    // ∫ sqrt(r² − x²) dx from 0.
    let half_chord = |x: f64| {
        let x = x.clamp(-r, r);
        0.5 * x.mul_add((r * r - x * x).max(0.0).sqrt(), r * r * (x / r).asin())
    };
    let x_b = (r * r - b * b).max(0.0).sqrt();
    let from = a.max(-r);
    let mut area = 0.0;
    for (p, q, crossed) in [(-r, -x_b, false), (-x_b, x_b, true), (x_b, r, false)] {
        let p = p.max(from);
        if q <= p {
            continue;
        }
        let under = half_chord(q) - half_chord(p);
        if crossed {
            area += b.mul_add(-(q - p), under);
        } else if b < 0.0 {
            area += 2.0 * under;
        }
    }
    area
}

/// The ball's volume over `x > a`, `y > b` and `z0 < z < z1`, by Simpson in
/// `z` over the closed-form slices.
fn ball_in_box(a: f64, b: f64, z0: f64, z1: f64) -> f64 {
    let (lo, hi) = (z0.max(-RADIUS), z1.min(RADIUS));
    if hi <= lo {
        return 0.0;
    }
    let n = 4000_u32;
    let step = (hi - lo) / f64::from(n);
    let slice = |z: f64| disc_corner_area(RADIUS.mul_add(RADIUS, -(z * z)).max(0.0).sqrt(), a, b);
    let mut sum = slice(lo) + slice(hi);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * slice(step.mul_add(f64::from(k), lo));
    }
    sum * step / 3.0
}

/// A box whose corner lies above the equator but on the far side of the axis
/// takes a patch around the pole: the section loop on the upper hemisphere
/// winds the axis, and the patch's interior sample has to land near the pole
/// rather than at its loop's centroid, which lies in the ring around it. The
/// Cut and the Intersect are exact, valid and watertight; a point by the pole
/// is removed, one below the box and one in a lune beside it are kept. The
/// Intersect measures its closed form; the Cut is held to the ball less it
/// only loosely, for the measure of a sphere face whose hole winds the pole
/// (the roadmap's sphere measure row). With the corner at `(-2.5, -2.5)` the
/// box's walls cut lens faces from the ball whose arcs and lines share both
/// ends. The first box mirrored through `z = 0` holds the south pole.
#[test]
fn a_corner_holding_the_pole_cuts_its_patch() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    for (a, b, c, up) in [
        (-0.7, -1.1, 0.1, 1.0),
        (-0.3, -0.4, 0.2, 1.0),
        (-0.3, -0.4, 1.2, 1.0),
        (-2.5, -2.5, 0.1, 1.0),
        (-0.7, -1.1, 0.1, -1.0),
    ] {
        let label = format!("corner ({a}, {b}, {c}) side {up}");
        let piece = ball_in_box(a, b, c, c + 10.0);
        for (op, truth, bound) in [
            (BooleanOp::Cut, ball - piece, 5e-3),
            (BooleanOp::Intersect, piece, 1e-4),
        ] {
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let z0 = if up > 0.0 { c } else { -c - 10.0 };
            transform_solid(&mut topo, block, &Mat4::translation(a, b, z0)).unwrap();
            let result = boolean(&mut topo, op, sphere, block).unwrap();
            assert!(exact(&topo, result), "{label} {op:?}: fell back to a mesh");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label} {op:?}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(
                is_watertight(&mesh),
                "{label} {op:?}: open or non-manifold mesh"
            );
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < bound * truth,
                "{label} {op:?}: volume {volume}, truth {truth}"
            );
            let at =
                |p: Point3| classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
            let (by_pole, kept) = if op == BooleanOp::Cut {
                (PointClassification::Outside, PointClassification::Inside)
            } else {
                (PointClassification::Inside, PointClassification::Outside)
            };
            for (p, class, place) in [
                (Point3::new(0.1, 0.1, 2.9 * up), by_pole, "by the pole"),
                (Point3::new(0.2, -0.3, -up), kept, "past the box's face"),
                (Point3::new(-2.717, 0.202, 0.735 * up), kept, "in a lune"),
            ] {
                assert_eq!(at(p), class, "{label} {op:?}: {place}");
            }
        }
    }
}

/// The ball's patch around the pole, cut again by a box over `z > 2.95`:
/// the second cut takes the patch's cap, and the piece left is exact.
#[test]
fn a_patch_around_the_pole_takes_a_second_cut() {
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, block, &Mat4::translation(-0.7, -1.1, 0.1)).unwrap();
    let patch = boolean(&mut topo, BooleanOp::Intersect, sphere, block).unwrap();
    let top = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    transform_solid(&mut topo, top, &Mat4::translation(-5.0, -5.0, 2.95)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Cut, patch, top).unwrap();
    assert!(exact(&topo, result), "fell back to a mesh");
    assert!(validate_solid(&topo, result).unwrap().is_valid(), "invalid");
    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
    assert!(is_watertight(&mesh), "open or non-manifold mesh");
    let truth = ball_in_box(-0.7, -1.1, 0.1, 2.95);
    let volume = solid_volume(&topo, result, 0.01).unwrap();
    assert!(
        (volume - truth).abs() < 2e-4 * truth,
        "volume {volume}, truth {truth}"
    );
}

/// The ball bored along `z` by a rod of radius 0.3 at `(x, y)`, with the box
/// over `(-0.7, -1.1, 0.1)`, which holds the pole and the rod's mouth (at
/// `(0.2, 0)` the bore holds the pole too, its rim a marched loop round it): the
/// ball's face around the pole keeps its bore's rim as a hole through the
/// box's section, so the box less the bored ball keeps the rod's piece as a
/// post and the bored ball within the box keeps the bore. Each result is
/// exact, valid and watertight, measures its truth (the ball's piece of the
/// box by slices, the rod's piece of the ball by the lens its disc shares
/// with each slice), and classifies a point in the post and one beside it.
/// The first box mirrored through `z = 0` holds the south pole, and each is
/// turned and mirrored with the ball too.
#[test]
fn a_bored_ball_keeps_its_bore_in_a_box_holding_the_pole() {
    const ROD: f64 = 0.3;
    let (a, b, c) = (-0.7, -1.1, 0.1);
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let piece = ball_in_box(a, b, c, c + 10.0);
    // The rod's disc, `d` off the axis, within a slice of radius `rho`.
    let lens = |rho: f64, d: f64| {
        if rho <= 0.0 || d >= rho + ROD {
            0.0
        } else if d + ROD <= rho {
            PI * ROD * ROD
        } else if d + rho <= ROD {
            PI * rho * rho
        } else {
            let (r2, p2) = (ROD * ROD, rho * rho);
            let kite =
                ((-d + ROD + rho) * (d + ROD - rho) * (d - ROD + rho) * (d + ROD + rho)).sqrt();
            r2 * ((d * d + r2 - p2) / (2.0 * d * ROD)).acos()
                + p2 * ((d * d + p2 - r2) / (2.0 * d * rho)).acos()
                - 0.5 * kite
        }
    };
    let rod_above = |d: f64, z0: f64| {
        let n = 20_000_u32;
        let step = (RADIUS - z0) / f64::from(n);
        let slice = |z: f64| lens(RADIUS.mul_add(RADIUS, -(z * z)).max(0.0).sqrt(), d);
        let mut sum = slice(z0) + slice(RADIUS);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * slice(step.mul_add(f64::from(k), z0));
        }
        sum * step / 3.0
    };
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    let at = Point3::new(0.3, 0.0, 0.0);
    let normal = brepkit_math::vec::Vec3::new(1.0, 0.2, 0.1);
    let unit = normal.normalize().unwrap();
    for (x, y) in [(0.0, 0.0), (0.2, 0.0), (1.0, 0.0), (-0.3, 0.4)] {
        let d = f64::hypot(x, y);
        let post = rod_above(d, c);
        let bored = ball - rod_above(d, -RADIUS);
        for up in [1.0, -1.0] {
            for pose in ["upright", "turned", "mirrored"] {
                let place = |p: Point3| match pose {
                    "turned" => turn.mul_point(p),
                    "mirrored" => p - unit * (2.0 * (p - at).dot(unit)),
                    _ => p,
                };
                let mut topo = Topology::new();
                let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
                let rod = make_cylinder(&mut topo, ROD, 10.0).unwrap();
                transform_solid(&mut topo, rod, &Mat4::translation(x, y, -5.0)).unwrap();
                let mut ball_bored = boolean(&mut topo, BooleanOp::Cut, sphere, rod).unwrap();
                let mut block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
                let z0 = if up > 0.0 { c } else { -c - 10.0 };
                transform_solid(&mut topo, block, &Mat4::translation(a, b, z0)).unwrap();
                match pose {
                    "turned" => {
                        transform_solid(&mut topo, ball_bored, &turn).unwrap();
                        transform_solid(&mut topo, block, &turn).unwrap();
                    }
                    "mirrored" => {
                        ball_bored = mirror(&mut topo, ball_bored, at, normal).unwrap();
                        block = mirror(&mut topo, block, at, normal).unwrap();
                    }
                    _ => {}
                }
                for (name, truth, bound) in [
                    ("box less bored", 1000.0 - piece + post, 1e-4),
                    ("bored within box", piece - post, 1e-4),
                    ("bored less box", bored - piece + post, 5e-3),
                ] {
                    let label = format!("rod ({x}, {y}) side {up} {pose}: {name}");
                    let result = match name {
                        "box less bored" => boolean(&mut topo, BooleanOp::Cut, block, ball_bored),
                        "bored within box" => {
                            boolean(&mut topo, BooleanOp::Intersect, ball_bored, block)
                        }
                        _ => boolean(&mut topo, BooleanOp::Cut, ball_bored, block),
                    }
                    .unwrap();
                    assert!(exact(&topo, result), "{label}: fell back to a mesh");
                    let report = validate_solid(&topo, result).unwrap();
                    assert!(report.is_valid(), "{label}: {:?}", report.issues);
                    let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                    assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
                    let volume = solid_volume(&topo, result, 0.01).unwrap();
                    assert!(
                        (volume - truth).abs() < bound * truth,
                        "{label}: volume {volume}, truth {truth}"
                    );
                    let (in_post, beside) = match name {
                        "box less bored" => {
                            (PointClassification::Inside, PointClassification::Outside)
                        }
                        "bored within box" => {
                            (PointClassification::Outside, PointClassification::Inside)
                        }
                        _ => (PointClassification::Outside, PointClassification::Outside),
                    };
                    let class = |p: Point3| {
                        classify_point(&topo, result, place(p), &ClassifyOptions::default())
                            .unwrap()
                    };
                    assert_eq!(
                        class(Point3::new(x, y, 1.5 * up)),
                        in_post,
                        "{label}: in the post"
                    );
                    assert_eq!(
                        class(Point3::new(x + 0.6, y, 1.5 * up)),
                        beside,
                        "{label}: beside the post"
                    );
                }
            }
        }
    }
}

/// The box over `(-0.7, -1.1, 0.1)` takes the ball's patch around the pole
/// and leaves a band between the chordal equator and its loop, and a hole
/// already in the band stays there: a bore at `(-1.5, 0)`, clear of the box,
/// or a pocket the box over `x < -2.2`, `y > -0.5`, `z > 1` bit out. The band
/// then has a hole winding the axis and another beside it. The ball less the
/// box is exact, valid and watertight, and measures the ball less both
/// pieces by slices, or less the box's piece and the bore's.
#[test]
fn a_band_keeps_its_other_holes() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let piece = ball_in_box(-0.7, -1.1, 0.1, 10.1);
    // The rod's chord through the ball over its disc, by Simpson in polar
    // coordinates about its axis at distance 1.5 from the ball's.
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let bore = simpson(200, 0.0, 0.3, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), -1.5), r * th.sin());
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    // Mirrored in `x`, the pocket is the ball over x > 2.2, y > -0.5, z > 1.
    let pocket = ball_in_box(2.2, -0.5, 1.0, 11.0);
    for (name, truth) in [
        ("bore", ball - bore - piece),
        ("pocket", ball - pocket - piece),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let tool = if name == "bore" {
            let rod = make_cylinder(&mut topo, 0.3, 10.0).unwrap();
            transform_solid(&mut topo, rod, &Mat4::translation(-1.5, 0.0, -5.0)).unwrap();
            rod
        } else {
            let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, b, &Mat4::translation(-12.2, -0.5, 1.0)).unwrap();
            b
        };
        let holed = boolean(&mut topo, BooleanOp::Cut, sphere, tool).unwrap();
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(-0.7, -1.1, 0.1)).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, holed, block).unwrap();
        assert!(exact(&topo, result), "{name}: fell back to a mesh");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-3 * truth,
            "{name}: volume {volume}, truth {truth}"
        );
    }
}

/// The ball less a square column through both poles, wider than the ball at
/// its corners, is four caps `pi h² (3R - h) / 3`, `h = R - a`, however the
/// column turns about the poles and wherever the pair sits. Each cap is two
/// lunes and a piece of wall, all cornered on the wall, and assembles as a
/// solid piece rather than a cavity. Each result is exact, valid and
/// watertight, measures the caps, and classifies a point in a cap inside and
/// the centre and a pole outside.
#[test]
fn a_ball_less_a_turned_column_keeps_its_four_caps() {
    for (a, turn, x) in [(2.5, 0.3, 0.0), (2.2, 0.3, 0.0), (2.2, 0.0, 10.0)] {
        let label = format!("column {a} turned {turn} at x = {x}");
        let h = RADIUS - a;
        let caps = 4.0 * PI * h * h * h.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let column = make_box(&mut topo, 2.0 * a, 2.0 * a, 10.0).unwrap();
        transform_solid(&mut topo, column, &Mat4::translation(-a, -a, -5.0)).unwrap();
        transform_solid(&mut topo, column, &Mat4::rotation_z(turn)).unwrap();
        transform_solid(&mut topo, sphere, &Mat4::translation(x, 0.0, 0.0)).unwrap();
        transform_solid(&mut topo, column, &Mat4::translation(x, 0.0, 0.0)).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, sphere, column).unwrap();
        assert!(exact(&topo, result), "{label}: fell back to a mesh");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - caps).abs() < 1e-3 * caps,
            "{label}: volume {volume}, truth {caps}"
        );
        let in_cap = 0.5 * (a + RADIUS);
        for (p, want) in [
            (
                Point3::new(in_cap.mul_add(turn.cos(), x), in_cap * turn.sin(), 0.1),
                PointClassification::Inside,
            ),
            (Point3::new(x, 0.0, 0.0), PointClassification::Outside),
            (
                Point3::new(x + 0.1, 0.1, -2.9),
                PointClassification::Outside,
            ),
        ] {
            let got = classify_point(&topo, result, p, &ClassifyOptions::default());
            assert_eq!(got.unwrap(), want, "{label}: {p:?}");
        }
    }
}

/// The ball within the column over `-2.5 < x, y < 2`, whose corner at
/// `(2, 2)` lies inside the ball, and the column less the ball, the column
/// through the ball or ending in it at `z = 2.8`: the walls `x = 2` and
/// `y = 2` meet on the sphere, the region past them owns a seam arc longer
/// than half a turn and is bounded by two wall circles, and the column's top
/// leaves a latitude hole in the collar. Each is exact, valid and watertight,
/// and within `1e-3` of the ball's chords over the column (by Simpson over
/// `x`, each chord's integral over `y` in closed form) less the polar cap.
#[test]
fn a_column_with_a_corner_in_the_ball_keeps_its_collars() {
    let r2 = RADIUS * RADIUS;
    let chords = |x: f64| {
        let c2 = r2 - x * x;
        if c2 <= 0.0 {
            return 0.0;
        }
        let c = c2.sqrt();
        let part = |y: f64| {
            let y = y.clamp(-c, c);
            y.mul_add((c2 - y * y).max(0.0).sqrt(), c2 * (y / c).asin())
        };
        part(2.0) - part(-2.5)
    };
    let n = 2000;
    let step = 4.5 / f64::from(n);
    let mut within = chords(-2.5) + chords(2.0);
    for k in 1..n {
        within += if k % 2 == 1 { 4.0 } else { 2.0 } * chords(step.mul_add(f64::from(k), -2.5));
    }
    within *= step / 3.0;
    let polar = PI * 0.04 * 0.2f64.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
    for (name, height, truth) in [
        ("within", 10.0, within),
        ("column less ball", 10.0, 202.5 - within),
        ("within, ending at z = 2.8", 7.8, within - polar),
        (
            "column ending at z = 2.8 less ball",
            7.8,
            4.5f64.mul_add(4.5 * 7.8, -(within - polar)),
        ),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let column = make_box(&mut topo, 4.5, 4.5, height).unwrap();
        transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
        let result = if name.starts_with("within") {
            boolean(&mut topo, BooleanOp::Intersect, sphere, column)
        } else {
            boolean(&mut topo, BooleanOp::Cut, column, sphere)
        }
        .unwrap();
        assert!(exact(&topo, result), "{name}: fell back to a mesh");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-3 * truth,
            "{name}: volume {volume}, truth {truth}"
        );
    }
}

/// The ball with a square column through both poles and a thin rod along
/// `z` through one of its caps, fused into one tool: the ball less it, within
/// it and the tool less the ball. Each hemisphere's cap past the wall
/// `x = 2.5` holds the rod's section as a hole, and the section bounds a
/// patch of its own. Each is exact, valid and watertight and within `1e-3` of
/// its closed form (the rod's chord by Simpson in polar coordinates), and the
/// ball less the tool reads the rod outside and the cap beside it inside to
/// both classifiers.
#[test]
fn a_rod_through_a_cap_leaves_its_hole_in_the_cap() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let h: f64 = 0.5;
    let caps = 4.0 * PI * h * h * h.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let bore = simpson(200, 0.0, 0.1, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), 2.75), r * th.sin());
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    let within = ball - caps + bore;
    let tool_volume = PI.mul_add(0.01 * 10.0, 250.0);
    for (name, truth) in [
        ("ball less tool", caps - bore),
        ("ball within tool", within),
        ("tool less ball", tool_volume - within),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
        transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
        let rod = make_cylinder(&mut topo, 0.1, 10.0).unwrap();
        transform_solid(&mut topo, rod, &Mat4::translation(2.75, 0.0, -5.0)).unwrap();
        let tool = boolean(&mut topo, BooleanOp::Fuse, column, rod).unwrap();
        let result = match name {
            "ball less tool" => boolean(&mut topo, BooleanOp::Cut, sphere, tool),
            "ball within tool" => boolean(&mut topo, BooleanOp::Intersect, sphere, tool),
            _ => boolean(&mut topo, BooleanOp::Cut, tool, sphere),
        }
        .unwrap();
        assert!(exact(&topo, result), "{name}: fell back to a mesh");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-3 * truth,
            "{name}: volume {volume}, truth {truth}"
        );
        if name != "ball less tool" {
            continue;
        }
        for (p, inside) in [
            (Point3::new(2.75, 0.0, 0.5), false),
            (Point3::new(2.75, 0.0, -0.5), false),
            (Point3::new(2.75, 0.4, 0.5), true),
            (Point3::new(2.75, -0.4, -0.5), true),
        ] {
            let (want_check, want_engine) = if inside {
                (PointClassification::Inside, brepkit_algo::FaceClass::Inside)
            } else {
                (
                    PointClassification::Outside,
                    brepkit_algo::FaceClass::Outside,
                )
            };
            let got = classify_point(&topo, result, p, &ClassifyOptions::default()).unwrap();
            assert_eq!(got, want_check, "{p:?}");
            let got = brepkit_algo::classifier::classify_ray_cast(&topo, result, p).unwrap();
            assert_eq!(got, want_engine, "engine: {p:?}");
        }
    }
}

/// The ball and a square column wider than the ball at its corners that
/// ends inside it at `z = 2.8`: the column's top cuts the upper hemisphere
/// in a latitude circle, a hole of the collar that winds the axis, and the
/// polar cap over it `pi h² (3R - h) / 3`, `h = 0.2`, is a patch of its own.
/// The ball less the column keeps the four caps past the walls and the polar
/// cap, the ball within it and the column less the ball lose it; each is
/// exact, valid, watertight and within `1e-3` of its closed form.
#[test]
fn a_column_ending_in_the_ball_keeps_the_cap_over_it() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let side_caps = 4.0 * PI * 0.25 * 0.5f64.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
    let polar = PI * 0.04 * 0.2f64.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
    let within = ball - side_caps - polar;
    for (name, truth) in [
        ("ball less column", side_caps + polar),
        ("ball within column", within),
        ("column less ball", 5.0 * 5.0 * 7.8 - within),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let column = make_box(&mut topo, 5.0, 5.0, 7.8).unwrap();
        transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
        let result = match name {
            "ball less column" => boolean(&mut topo, BooleanOp::Cut, sphere, column),
            "ball within column" => boolean(&mut topo, BooleanOp::Intersect, sphere, column),
            _ => boolean(&mut topo, BooleanOp::Cut, column, sphere),
        }
        .unwrap();
        assert!(exact(&topo, result), "{name}: fell back to a mesh");
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{name}: open or non-manifold mesh");
        let volume = solid_volume(&topo, result, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-3 * truth,
            "{name}: volume {volume}, truth {truth}"
        );
    }
}

/// The ball within the square column `|x|, |y| < 2.5` through both poles, the
/// column less the ball and the ball less the column, bored by a rod at
/// `(0.5, 0.5)` or not: each hemisphere keeps the collar inside the column's
/// four walls, or the four lunes past them. Each is exact, valid and
/// watertight, and measures the ball less its four caps `pi h² (3R - h) / 3`,
/// `h = 0.5`, and the bore's chord, the column less that, or the caps, which
/// classify on the right side.
#[test]
fn a_column_through_both_poles_keeps_its_collars() {
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    let caps = 4.0 * PI * 0.25 * 0.5f64.mul_add(-1.0, 3.0 * RADIUS) / 3.0;
    let simpson = |n: u32, lo: f64, hi: f64, f: &dyn Fn(f64) -> f64| {
        let step = (hi - lo) / f64::from(n);
        let mut sum = f(lo) + f(hi);
        for k in 1..n {
            sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(step.mul_add(f64::from(k), lo));
        }
        sum * step / 3.0
    };
    let bore = simpson(200, 0.0, 0.3, &|r: f64| {
        r * simpson(200, 0.0, 2.0 * PI, &|th: f64| {
            let (x, y) = (r.mul_add(th.cos(), 0.5), r.mul_add(th.sin(), 0.5));
            2.0 * RADIUS.mul_add(RADIUS, -(x * x + y * y)).sqrt()
        })
    });
    for bored in [false, true] {
        let within = ball - caps - if bored { bore } else { 0.0 };
        for (name, op, truth) in [
            ("within", BooleanOp::Intersect, within),
            ("column less ball", BooleanOp::Cut, 250.0 - within),
            ("ball less column", BooleanOp::Cut, caps),
        ] {
            let label = format!("bored {bored} {name}");
            let mut topo = Topology::new();
            let mut sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            if bored {
                let rod = make_cylinder(&mut topo, 0.3, 10.0).unwrap();
                transform_solid(&mut topo, rod, &Mat4::translation(0.5, 0.5, -5.0)).unwrap();
                sphere = boolean(&mut topo, BooleanOp::Cut, sphere, rod).unwrap();
            }
            let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
            transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
            let result = if name == "column less ball" {
                boolean(&mut topo, op, column, sphere)
            } else {
                boolean(&mut topo, op, sphere, column)
            }
            .unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
            if name == "ball less column" {
                for (p, want) in [
                    (Point3::new(2.8, 0.0, 0.1), PointClassification::Inside),
                    (Point3::new(-0.1, -2.8, -0.2), PointClassification::Inside),
                    (Point3::new(0.0, 0.0, 0.0), PointClassification::Outside),
                    (Point3::new(0.2, 0.0, 2.9), PointClassification::Outside),
                    (Point3::new(2.2, 2.2, 0.0), PointClassification::Outside),
                ] {
                    let got = classify_point(&topo, result, p, &ClassifyOptions::default());
                    assert_eq!(got.unwrap(), want, "{label}: {p:?}");
                }
            }
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{label}: open or non-manifold mesh");
            let volume = solid_volume(&topo, result, 0.01).unwrap();
            assert!(
                (volume - truth).abs() < 1e-3 * truth,
                "{label}: volume {volume}, truth {truth}"
            );
        }
    }
}

/// The ball within the square column `|x|, |y| < 2.5` keeps a collar on each
/// hemisphere, bounded by the four walls' arcs and four arcs of the equator.
/// The engine's ray cast reads the collar by its planes: every point of a
/// grid over the ball, clear of its surfaces by 0.05, lands on the right side.
#[test]
fn the_engine_reads_a_collar_by_its_planes() {
    let mut topo = Topology::new();
    let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
    let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
    transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
    let result = boolean(&mut topo, BooleanOp::Intersect, sphere, column).unwrap();
    assert!(exact(&topo, result), "fell back to a mesh");
    let steps = 13;
    let mut checked = 0;
    for i in 0..steps {
        for j in 0..steps {
            for k in 0..steps {
                let at = |n: i32| -3.3 + 6.6 * f64::from(n) / f64::from(steps - 1);
                let p = Point3::new(at(i), at(j), at(k));
                let rho = (p - Point3::new(0.0, 0.0, 0.0)).length();
                let near = (rho - RADIUS).abs() < 0.05
                    || (p.x().abs() - 2.5).abs() < 0.05
                    || (p.y().abs() - 2.5).abs() < 0.05;
                if near {
                    continue;
                }
                let inside = rho < RADIUS && p.x().abs() < 2.5 && p.y().abs() < 2.5;
                let got = brepkit_algo::classifier::classify_ray_cast(&topo, result, p).unwrap();
                let want = if inside {
                    brepkit_algo::FaceClass::Inside
                } else {
                    brepkit_algo::FaceClass::Outside
                };
                assert_eq!(got, want, "{p:?}");
                checked += 1;
            }
        }
    }
    assert!(checked > 1000, "{checked} points checked");
}

/// The ball less a square column that enters it from below (`z > -1`) or
/// from above (`z < 1`): the far hemisphere keeps a hole of four wall arcs,
/// whose wall circles also run past the equator, where the hemisphere's own
/// boundary already ends the face. The engine's ray cast reads the hole by
/// its planes and the walls by their arcs: every point of a grid over the
/// ball, clear of its surfaces by 0.05, lands on the right side, as does a
/// point whose rays pass 0.029 under a wall's arc and over its chord.
#[test]
fn the_engine_reads_a_hole_by_its_planes() {
    for half in [0.1, 0.6, 1.6] {
        for end in [-1.0, 1.0] {
            let label = format!("{half} to z = {end}");
            let mut topo = Topology::new();
            let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let column = make_box(&mut topo, 2.0 * half, 2.0 * half, 10.0).unwrap();
            let bottom = if end < 0.0 { end } else { end - 10.0 };
            transform_solid(&mut topo, column, &Mat4::translation(-half, -half, bottom)).unwrap();
            let result = boolean(&mut topo, BooleanOp::Cut, sphere, column).unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            let in_column = |p: Point3| {
                p.x().abs() < half
                    && p.y().abs() < half
                    && if end < 0.0 { p.z() > end } else { p.z() < end }
            };
            let read = |p: Point3| {
                let got = brepkit_algo::classifier::classify_ray_cast(&topo, result, p).unwrap();
                let rho = (p - Point3::new(0.0, 0.0, 0.0)).length();
                let want = if rho < RADIUS && !in_column(p) {
                    brepkit_algo::FaceClass::Inside
                } else {
                    brepkit_algo::FaceClass::Outside
                };
                assert_eq!(got, want, "{label}: {p:?}");
            };
            let steps = 13;
            let mut checked = 0;
            for i in 0..steps {
                for j in 0..steps {
                    for k in 0..steps {
                        let at = |n: i32| -3.3 + 6.6 * f64::from(n) / f64::from(steps - 1);
                        let p = Point3::new(at(i), at(j), at(k));
                        let rho = (p - Point3::new(0.0, 0.0, 0.0)).length();
                        let near = (rho - RADIUS).abs() < 0.05
                            || (p.x().abs() - half).abs() < 0.05
                            || (p.y().abs() - half).abs() < 0.05
                            || (p.z() - end).abs() < 0.05;
                        if !near {
                            read(p);
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 1000, "{label}: {checked} points checked");
            if half > 1.0 && end < 0.0 {
                read(Point3::new(0.4125, 0.4125, 2.475));
            }
        }
    }
}

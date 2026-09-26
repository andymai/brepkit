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
/// and dozens of them: the results here keep a sphere face among a handful.
/// (The fallback counter is process-wide, and other tests in this binary may
/// fall back while one runs.)
fn exact(topo: &Topology, solid: SolidId) -> bool {
    let faces = solid_faces(topo, solid).unwrap();
    faces.len() <= 12
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

/// The ball within the square column `|x|, |y| < 2.5` through both poles, and
/// the column less the ball, bored by a rod at `(0.5, 0.5)` or not: each
/// hemisphere keeps the collar inside the column's four walls. Each is exact,
/// valid and watertight, and measures the ball less its four caps
/// `pi h² (3R - h) / 3`, `h = 0.5`, and the bore's chord, or the column less
/// that.
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
        for (op, truth) in [
            (BooleanOp::Intersect, within),
            (BooleanOp::Cut, 250.0 - within),
        ] {
            let label = format!("bored {bored} {op:?}");
            let mut topo = Topology::new();
            let mut sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
            if bored {
                let rod = make_cylinder(&mut topo, 0.3, 10.0).unwrap();
                transform_solid(&mut topo, rod, &Mat4::translation(0.5, 0.5, -5.0)).unwrap();
                sphere = boolean(&mut topo, BooleanOp::Cut, sphere, rod).unwrap();
            }
            let column = make_box(&mut topo, 5.0, 5.0, 10.0).unwrap();
            transform_solid(&mut topo, column, &Mat4::translation(-2.5, -2.5, -5.0)).unwrap();
            let result = if op == BooleanOp::Cut {
                boolean(&mut topo, op, column, sphere)
            } else {
                boolean(&mut topo, op, sphere, column)
            }
            .unwrap();
            assert!(exact(&topo, result), "{label}: fell back to a mesh");
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{label}: {:?}", report.issues);
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

//! The area of a sphere face that a box trims, measured exactly and by the
//! face's own mesh. A ball of radius 3 meets the box over `x > 1`, `y > 1.2`,
//! `z > 0.8` in a three-sided patch, and the box over the positive octant in
//! a patch whose sides are two meridians and the equator. A tilted rod
//! through the ball keeps two caps whose flux measures the rod's piece
//! exactly, and a pocket over the pole leaves a hole that meets the pole.
//! Each piece's volume is held to an independent integral too.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::measure::{face_area, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
use brepkit_operations::tessellate::{is_watertight, tessellate, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const RADIUS: f64 = 3.0;

/// The sphere faces of the ball less or within a box at `corner`, with their
/// exact and meshed areas, in the order the result lists them, and the
/// piece's volume when `measure` asks for it.
fn sphere_faces(
    op: BooleanOp,
    corner: (f64, f64, f64),
    measure: bool,
) -> (Vec<(f64, f64)>, Option<f64>) {
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
    let faces = solid_faces(&topo, piece)
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
        .collect();
    let volume = measure.then(|| solid_volume(&topo, piece, 0.01).unwrap());
    (faces, volume)
}

/// Simpson's rule over `n` (even) panels, substituted `x = end - s²` so the
/// integrand stays smooth where the region pinches off at `end`.
fn toward_pinch(from: f64, end: f64, f: &dyn Fn(f64) -> f64) -> f64 {
    let g = |s: f64| f(s.mul_add(-s, end)) * 2.0 * s;
    let (n, span) = (800_u32, (end - from).sqrt());
    let step = span / f64::from(n);
    let mut sum = g(0.0) + g(span);
    for k in 1..n {
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * g(step * f64::from(k));
    }
    sum * step / 3.0
}

/// The patch past `x = 1`, `y = 1.2` and `z = 0.8`, projected onto the
/// `xy` plane, where the sphere's area element is `R / z`: across `y` it
/// integrates to `R asin(y / c)` with `c = sqrt(R² - x²)`, leaving one
/// Simpson integral in `x`, substituted where the patch pinches off.
fn corner_patch() -> f64 {
    let x_end = (RADIUS * RADIUS - 0.64 - 1.44).sqrt();
    toward_pinch(1.0, x_end, &|x: f64| {
        let c = RADIUS.mul_add(RADIUS, -(x * x)).sqrt();
        let y_end = (RADIUS * RADIUS - 0.64 - x * x).max(0.0).sqrt();
        RADIUS * ((y_end / c).asin() - (1.2 / c).asin())
    })
}

/// The ball's piece past the same three planes: across `y` the height
/// `sqrt(c² - y²) - 0.8` integrates in closed form, leaving the same
/// integral in `x`.
fn corner_piece() -> f64 {
    let x_end = (RADIUS * RADIUS - 0.64 - 1.44).sqrt();
    toward_pinch(1.0, x_end, &|x: f64| {
        let c2 = RADIUS.mul_add(RADIUS, -(x * x));
        let y_end = (c2 - 0.64).max(0.0).sqrt();
        let g = |y: f64| {
            0.5 * y.mul_add(
                y.mul_add(-y, c2).max(0.0).sqrt(),
                c2 * (y / c2.sqrt()).asin(),
            ) - 0.8 * y
        };
        g(y_end) - g(1.2)
    })
}

#[test]
fn sphere_face_bitten_by_a_box_corner() {
    let truth = corner_patch();
    let (faces, volume) = sphere_faces(BooleanOp::Intersect, (1.0, 1.2, 0.8), true);
    let (volume, piece) = (volume.unwrap(), corner_piece());
    assert!(
        (volume - piece).abs() < 1e-9 * piece,
        "volume {volume}, truth {piece}"
    );
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
    let (within, volume) = sphere_faces(BooleanOp::Intersect, (0.0, 0.0, 0.0), true);
    let (volume, piece) = (volume.unwrap(), PI * RADIUS.powi(3) / 6.0);
    assert!(
        (volume - piece).abs() < 1e-9 * piece,
        "volume {volume}, truth {piece}"
    );
    assert_eq!(within.len(), 1, "one sphere face");
    // Less the octant, the upper hemisphere keeps three quarters of itself.
    let (mut less, _) = sphere_faces(BooleanOp::Cut, (0.0, 0.0, 0.0), false);
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

/// The ball above the plane `z = 1.5 + slope x`, clear of its equator: a cap
/// under one circle, level or tilted, `2 pi R (R - d)` of the sphere with `d`
/// the plane's distance from the centre, which its own mesh covers too.
#[test]
fn cap_under_a_level_or_tilted_circle() {
    for slope in [0.0_f64, 0.2] {
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
        assert_eq!(faces.len(), 1, "slope {slope}: one sphere face");
        let area = face_area(&topo, faces[0], 0.005).unwrap();
        assert!(
            (area - truth).abs() < 1e-9 * truth,
            "slope {slope}: area {area}, truth {truth}"
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
            "slope {slope}: mesh area {meshed}, truth {truth}"
        );
    }
}

/// A rod of radius 0.8 through the ball, its axis tilted and 1.08 from the
/// centre: the piece inside is, over the rod's cross-section, the chord
/// through the ball, integrated by Simpson in polar coordinates about the
/// axis. Less it, the ball keeps the rest: a valid, watertight solid that
/// holds the ball's far side and not the rod's axis.
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
        let report = validate_solid(&topo, piece).unwrap();
        assert!(report.is_valid(), "{op:?}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, piece, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{op:?}: open or non-manifold mesh");
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-7 * truth,
            "{op:?}: volume {volume}, truth {truth}"
        );
        let at = |p: Point3| classify_point(&topo, piece, p, &ClassifyOptions::default()).unwrap();
        let (on_axis, far_side) = if op == BooleanOp::Intersect {
            (PointClassification::Inside, PointClassification::Outside)
        } else {
            (PointClassification::Outside, PointClassification::Inside)
        };
        assert_eq!(
            at(Point3::new(foot.0, foot.1, 0.0)),
            on_axis,
            "{op:?}: rod axis"
        );
        assert_eq!(
            at(Point3::new(-2.0, 0.0, 0.0)),
            far_side,
            "{op:?}: far side"
        );
    }
}

/// The box over `x > 0`, `y > 0`, `z > 2` takes a quarter of the cap above
/// `z = 2`, whose loop runs up one meridian to the pole and down another:
/// the cap is `h = 1` high, so `2 pi R h` of the sphere and `pi h² (3R - h) / 3`
/// of the ball.
#[test]
fn pocket_through_a_pole() {
    let (cap_area, cap_volume) = (2.0 * PI * RADIUS, PI * (3.0 * RADIUS - 1.0) / 3.0);
    let ball = 4.0 / 3.0 * PI * RADIUS.powi(3);
    for (op, truth, areas) in [
        (BooleanOp::Intersect, cap_volume / 4.0, vec![cap_area / 4.0]),
        (
            BooleanOp::Cut,
            ball - cap_volume / 4.0,
            vec![
                2.0 * PI * RADIUS * RADIUS - cap_area / 4.0,
                2.0 * PI * RADIUS * RADIUS,
            ],
        ),
    ] {
        let mut topo = Topology::new();
        let sphere = make_sphere(&mut topo, RADIUS, 32).unwrap();
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(0.0, 0.0, 2.0)).unwrap();
        let piece = boolean(&mut topo, op, sphere, block).unwrap();
        let volume = solid_volume(&topo, piece, 0.01).unwrap();
        assert!(
            (volume - truth).abs() < 1e-9 * truth,
            "{op:?}: volume {volume}, truth {truth}"
        );
        let mut found: Vec<f64> = solid_faces(&topo, piece)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
            .map(|f| face_area(&topo, f, 0.005).unwrap())
            .collect();
        found.sort_by(f64::total_cmp);
        assert_eq!(found.len(), areas.len(), "{op:?}: sphere faces");
        for (area, truth) in found.into_iter().zip(areas) {
            assert!(
                (area - truth).abs() < 1e-9 * truth,
                "{op:?}: area {area}, truth {truth}"
            );
        }
    }
}

/// A box over the positive octant less the ball keeps the octant's patch
/// turned inward, a quarter of a hemisphere, and the box less an eighth of
/// the ball.
#[test]
fn box_less_the_balls_octant() {
    let mut topo = Topology::new();
    let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
    let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
    let piece = boolean(&mut topo, BooleanOp::Cut, block, ball).unwrap();
    let faces: Vec<_> = solid_faces(&topo, piece)
        .unwrap()
        .into_iter()
        .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
        .collect();
    assert_eq!(faces.len(), 1, "one sphere face");
    assert!(topo.face(faces[0]).unwrap().is_reversed(), "turned inward");
    let (area, truth) = (
        face_area(&topo, faces[0], 0.005).unwrap(),
        PI * RADIUS * RADIUS / 2.0,
    );
    assert!(
        (area - truth).abs() < 1e-9 * truth,
        "area {area}, truth {truth}"
    );
    let (volume, truth) = (
        solid_volume(&topo, piece, 0.01).unwrap(),
        1000.0 - PI * RADIUS.powi(3) / 6.0,
    );
    assert!(
        (volume - truth).abs() < 1e-9 * truth,
        "volume {volume}, truth {truth}"
    );
}

/// Half the cap above `z = 1`, built by hand: the latitude on the `y > 0`
/// side, then the arc in `y = 0` back over the pole to a vertex on the far
/// side (at `z = 2`, or half a degree past the pole, inside the arc's last
/// probe), which it passes between its two vertices off the middle of its
/// span, and down to the latitude. Half the cap is `pi R h` with `h = R - 1`.
#[test]
fn half_cap_whose_arc_runs_over_the_pole() {
    let rim = RADIUS.mul_add(RADIUS, -1.0).sqrt();
    let half_degree = 0.5_f64.to_radians();
    for peak in [
        Point3::new(RADIUS.mul_add(RADIUS, -4.0).sqrt(), 0.0, 2.0),
        Point3::new(RADIUS * half_degree.sin(), 0.0, RADIUS * half_degree.cos()),
    ] {
        let mut topo = Topology::new();
        let o = Point3::new(0.0, 0.0, 0.0);
        let east = topo.add_vertex(Vertex::new(Point3::new(rim, 0.0, 1.0), 1e-7));
        let west = topo.add_vertex(Vertex::new(Point3::new(-rim, 0.0, 1.0), 1e-7));
        let peak_id = topo.add_vertex(Vertex::new(peak, 1e-7));
        let latitude =
            Circle3D::new(Point3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0), rim).unwrap();
        let meridian = Circle3D::new(o, Vec3::new(0.0, 1.0, 0.0), RADIUS).unwrap();
        let edges = [
            topo.add_edge(Edge::new(east, west, EdgeCurve::Circle(latitude))),
            topo.add_edge(Edge::new(
                west,
                peak_id,
                EdgeCurve::Circle(meridian.clone()),
            )),
            topo.add_edge(Edge::new(peak_id, east, EdgeCurve::Circle(meridian))),
        ];
        let wire = Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let sphere = SphericalSurface::new(o, RADIUS).unwrap();
        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Sphere(sphere)));
        let truth = PI * RADIUS * (RADIUS - 1.0);
        let area = face_area(&topo, face, 0.005).unwrap();
        assert!(
            (area - truth).abs() < 1e-9 * truth,
            "peak {peak:?}: area {area}, truth {truth}"
        );
    }
}

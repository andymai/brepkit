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
use brepkit_math::nurbs::curve::NurbsCurve;
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

/// A box whose corner lies above the equator on the far side of the axis
/// keeps a patch of the ball around the pole: at longitude `u` it spans from
/// the highest latitude any of the box's three planes allows to the pole, so
/// its area is `R² ∫ (1 − sin v_b(u)) du` (Simpson). The patch's loop winds
/// the axis, which a patch's `(u, v)` area does not cover.
#[test]
fn patch_around_the_pole() {
    let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
    for (a, b, c) in [(-0.7, -1.1, 0.1), (-0.3, -0.4, 0.2), (-0.3, -0.4, 1.2)] {
        let v_b = |u: f64| {
            let mut lo = (c / RADIUS).asin();
            for (comp, bound) in [(u.cos(), a), (u.sin(), b)] {
                let ratio = bound / (RADIUS * comp);
                if comp < 0.0 && ratio < 1.0 {
                    lo = lo.max(ratio.acos());
                }
            }
            lo
        };
        let n = 20_000_u32;
        let step = 2.0 * PI / f64::from(n);
        let mut sum = 0.0;
        for k in 0..=n {
            let weight = if k == 0 || k == n {
                1.0
            } else if k % 2 == 1 {
                4.0
            } else {
                2.0
            };
            sum += weight * (1.0 - v_b(step * f64::from(k)).sin());
        }
        let truth = RADIUS * RADIUS * sum * step / 3.0;
        // The south pole's patch mirrors the north's, and the block less the
        // ball keeps the same patch reversed.
        for (south, dimple, turned) in (0..8).map(|k| (k & 1 == 1, k & 2 == 2, k & 4 == 4)) {
            let label = format!("({a}, {b}, {c}) south {south} dimple {dimple} turned {turned}");
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let z = if south { -c - 10.0 } else { c };
            transform_solid(&mut topo, block, &Mat4::translation(a, b, z)).unwrap();
            if turned {
                transform_solid(&mut topo, ball, &turn).unwrap();
                transform_solid(&mut topo, block, &turn).unwrap();
            }
            let piece = if dimple {
                boolean(&mut topo, BooleanOp::Cut, block, ball).unwrap()
            } else {
                boolean(&mut topo, BooleanOp::Intersect, ball, block).unwrap()
            };
            let patches: Vec<_> = solid_faces(&topo, piece)
                .unwrap()
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                .collect();
            assert_eq!(patches.len(), 1, "{label}: sphere faces");
            let area = face_area(&topo, patches[0], 0.01).unwrap();
            assert!(
                (area - truth).abs() < 1e-6 * truth,
                "{label}: area {area}, truth {truth}"
            );
        }
    }
}

/// The circle where the plane `n · p = d` (`n` tilted `tilt` from `z`) meets
/// the ball, as three rational quadratic arcs running about `n`. `flip` stores
/// one arc's curve from its end vertex to its start, `over` stretches each
/// curve past its vertices by that far, and `reverse` runs the loop backwards.
fn arcs_around_the_pole(
    topo: &mut Topology,
    (tilt, d): (f64, f64),
    flip: Option<usize>,
    over: f64,
    reverse: bool,
) -> brepkit_topology::wire::WireId {
    let n = Vec3::new(tilt.sin(), 0.0, tilt.cos());
    let e1 = Vec3::new(tilt.cos(), 0.0, -tilt.sin());
    let e2 = Vec3::new(0.0, 1.0, 0.0);
    let c = Point3::new(0.0, 0.0, 0.0) + n * d;
    let rho = (RADIUS * RADIUS - d * d).sqrt();
    let at = |a: f64| c + e1 * (rho * a.cos()) + e2 * (rho * a.sin());
    let angles = [0.3, 0.3 + 2.0 * PI / 3.0, 0.3 + 4.0 * PI / 3.0];
    let verts: Vec<_> = angles
        .iter()
        .map(|&a| topo.add_vertex(Vertex::new(at(a), 1e-7)))
        .collect();
    let mut edges = Vec::new();
    for k in 0..3 {
        let (a0, a1) = (
            angles[k] - over / rho,
            angles[k] + 2.0 * PI / 3.0 + over / rho,
        );
        let half = 0.5 * (a1 - a0);
        let mid = 0.5 * (a0 + a1);
        let peak = c + (e1 * mid.cos() + e2 * mid.sin()) * (rho / half.cos());
        let mut points = vec![at(a0), peak, at(a1)];
        if flip == Some(k) {
            points.reverse();
        }
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            points,
            vec![1.0, half.cos(), 1.0],
        )
        .unwrap();
        edges.push(topo.add_edge(Edge::new(
            verts[k],
            verts[(k + 1) % 3],
            EdgeCurve::NurbsCurve(curve),
        )));
    }
    let oes: Vec<_> = if reverse {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    } else {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    };
    topo.add_wire(Wire::new(oes, true).unwrap())
}

#[test]
fn nurbs_loop_around_the_pole() {
    let sphere =
        || FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), RADIUS).unwrap());
    let cap = |d: f64| 2.0 * PI * RADIUS * (RADIUS - d);
    // A curve that runs against its edge is walked from the edge's start.
    for flip in [None, Some(0), Some(1)] {
        let mut topo = Topology::new();
        let outer = arcs_around_the_pole(&mut topo, (0.3, 1.0), flip, 0.0, false);
        let face = topo.add_face(Face::new(outer, vec![], sphere()));
        let area = face_area(&topo, face, 0.01).unwrap();
        assert!(
            (area - cap(1.0)).abs() < 1e-6 * cap(1.0),
            "flip {flip:?}: {area} against {}",
            cap(1.0)
        );

        let mut topo = Topology::new();
        let hole = arcs_around_the_pole(&mut topo, (0.3, 1.0), flip, 0.0, true);
        let z0 = -2.0_f64;
        let r0 = (RADIUS * RADIUS - z0 * z0).sqrt();
        let rim = Circle3D::new(Point3::new(0.0, 0.0, z0), Vec3::new(0.0, 0.0, 1.0), r0).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r0, 0.0, z0), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(rim)));
        let outer = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let face = topo.add_face(Face::new(outer, vec![hole], sphere()));
        let area = face_area(&topo, face, 0.01).unwrap();
        let truth = cap(z0) - cap(1.0);
        assert!(
            (area - truth).abs() < 1e-6 * truth,
            "hole flip {flip:?}: {area} against {truth}"
        );
    }
    // Curve ends may miss their vertices by up to 1e-6, which near the pole
    // leaves the turn short by more than a rounding error.
    for over in [3e-7, 9e-7] {
        let mut topo = Topology::new();
        let outer = arcs_around_the_pole(&mut topo, (0.05, 2.8), None, over, false);
        let face = topo.add_face(Face::new(outer, vec![], sphere()));
        let area = face_area(&topo, face, 0.01).unwrap();
        assert!(
            (area - cap(2.8)).abs() < 1e-4 * cap(2.8),
            "over {over}: {area} against {}",
            cap(2.8)
        );
    }
}

#[test]
fn latitude_loop_split_at_opposite_longitudes() {
    let z = 1.0_f64;
    let rho = (RADIUS * RADIUS - z * z).sqrt();
    let north = 2.0 * PI * RADIUS * (RADIUS - z);
    for (reverse, truth) in [(false, north), (true, 4.0 * PI * RADIUS * RADIUS - north)] {
        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, z), Vec3::new(0.0, 0.0, 1.0), rho).unwrap();
        let ends = [
            topo.add_vertex(Vertex::new(Point3::new(rho, 0.0, z), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(-rho, 0.0, z), 1e-7)),
        ];
        let halves = [
            topo.add_edge(Edge::new(
                ends[0],
                ends[1],
                EdgeCurve::Circle(circle.clone()),
            )),
            topo.add_edge(Edge::new(ends[1], ends[0], EdgeCurve::Circle(circle))),
        ];
        let oes = if reverse {
            vec![
                OrientedEdge::new(halves[1], false),
                OrientedEdge::new(halves[0], false),
            ]
        } else {
            vec![
                OrientedEdge::new(halves[0], true),
                OrientedEdge::new(halves[1], true),
            ]
        };
        let outer = topo.add_wire(Wire::new(oes, true).unwrap());
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), RADIUS).unwrap();
        let face = topo.add_face(Face::new(outer, vec![], FaceSurface::Sphere(sphere)));
        let area = face_area(&topo, face, 0.01).unwrap();
        assert!(
            (area - truth).abs() < 1e-6 * truth,
            "reverse {reverse}: {area} against {truth}"
        );
    }
}

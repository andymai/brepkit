//! A disc of radius `R`, `H` high, with a slot `2 C` wide cut across its top
//! down to `FLOOR`: the cap of the brepjs `panel-fuse-holder` example. Each
//! end of the slot meets the disc's wall along two straight edges, rulings of
//! the wall between it and a slot side parallel to the axis.
//!
//! A chamfer of `d` on such an edge takes the region between the slot side,
//! the wall and the plane through the two contacts: `d` into the side, and
//! the ruling a chord `d` round the wall. Its cross-section is the triangle
//! between the edge and the two contacts and the segment of the disc between
//! the chord and the wall. A chamfer of the top rim's arcs takes the ring
//! above the cone `r + z = R + H - d`. Where the two meet, at the slot's top
//! corners, they cut each other off, and the cone also takes a sliver of the
//! slot side. The result's volume is read off its cross-sections across the
//! axis, each the disc's two segments beyond the slot, cut back to the
//! cone's radius at that height, less the four corners the rulings' chamfers
//! take.
#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::f64::consts::PI;

use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
use brepkit_math::mat::Mat4;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Point3;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::chamfer::chamfer;
use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

const R: f64 = 9.4;
const H: f64 = 3.0;
const C: f64 = 0.8;
const FLOOR: f64 = 2.0;

/// The disc with its slot, placed by `pose`.
fn slotted_disc(topo: &mut Topology, pose: &Mat4) -> SolidId {
    let disc = make_cylinder(topo, R, H).unwrap();
    let slot = make_box(topo, 30.0, 2.0 * C, H - FLOOR).unwrap();
    transform_solid(topo, slot, &Mat4::translation(-15.0, -C, FLOOR)).unwrap();
    let cap = boolean(topo, BooleanOp::Cut, disc, slot).unwrap();
    transform_solid(topo, cap, pose).unwrap();
    cap
}

/// The top rim's arcs (when `rim`) and the four rulings from the floor to the
/// top at the slot's ends, read in the disc's own frame.
fn pick(topo: &Topology, cap: SolidId, pose: &Mat4, rim: bool) -> Vec<EdgeId> {
    let inverse = pose.inverse().unwrap();
    let local = |v| inverse.mul_point(topo.vertex(v).unwrap().point());
    solid_edges(topo, cap)
        .unwrap()
        .into_iter()
        .filter(|&e| {
            let edge = topo.edge(e).unwrap();
            let (a, b) = (local(edge.start()), local(edge.end()));
            let r = |p: Point3| p.x().hypot(p.y());
            match edge.curve() {
                EdgeCurve::Circle(_) => rim && (a.z() - H).abs() < 1e-9 && (b.z() - H).abs() < 1e-9,
                EdgeCurve::Line => {
                    (r(a) - R).abs() < 1e-6
                        && (r(b) - R).abs() < 1e-6
                        && (a.z() - b.z()).abs() > 0.5
                        && a.z().min(b.z()) > FLOOR - 1e-9
                }
                _ => false,
            }
        })
        .collect()
}

/// Upright, turned about one axis and about all three, and mirrored.
fn poses() -> Vec<(&'static str, Mat4)> {
    vec![
        ("upright", Mat4::translation(0.0, 0.0, 0.0)),
        ("turned", Mat4::rotation_z(0.7)),
        (
            "tilted",
            Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
        ),
        ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
    ]
}

/// The area of the disc of radius `rho` beyond the slot's side, `y >= C`.
fn segment_beyond_slot(rho: f64) -> f64 {
    rho * rho * (C / rho).acos() - C * (rho * rho - C * C).sqrt()
}

/// The slot side's contact `s` and the wall's contact `w` of a ruling's
/// chamfer at the corner `(x0, C)`, and the area of the triangle between
/// the three with the disc's segment between the corner and `w`.
fn ruling_section(d: f64) -> ([f64; 2], [f64; 2], f64) {
    let x0 = (R * R - C * C).sqrt();
    let turn = 2.0 * (0.5 * d / R).asin();
    let at = (C / R).asin() + turn;
    let s = [x0 - d, C];
    let w = [R * at.cos(), R * at.sin()];
    let triangle = 0.5 * ((s[0] - x0) * (w[1] - C) - (s[1] - C) * (w[0] - x0)).abs();
    (s, w, triangle + 0.5 * R * R * (turn - turn.sin()))
}

/// The area of the disc of radius `rho` beyond the slot's side and beyond
/// the line from `s` through `w`, on the corner's side: the triangle from
/// `s` to where the side and the line leave the disc, with the disc's
/// segment between those two.
fn corner_cut(rho: f64, s: [f64; 2], w: [f64; 2]) -> f64 {
    let x1 = (rho * rho - C * C).sqrt();
    if x1 <= s[0] {
        return 0.0;
    }
    let e1 = [x1, C];
    let u = [w[0] - s[0], w[1] - s[1]];
    let a = u[0] * u[0] + u[1] * u[1];
    let b = 2.0 * (s[0] * u[0] + s[1] * u[1]);
    let c = s[0] * s[0] + s[1] * s[1] - rho * rho;
    let t = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);
    let e2 = [s[0] + u[0] * t, s[1] + u[1] * t];
    let triangle = 0.5 * ((e1[0] - s[0]) * (e2[1] - s[1]) - (e1[1] - s[1]) * (e2[0] - s[0])).abs();
    let phi = e2[1].atan2(e2[0]) - e1[1].atan2(e1[0]);
    triangle + 0.5 * rho * rho * (phi - phi.sin())
}

/// `f` integrated from `a` to `b`, over which it is smooth.
fn integral(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    gauss_legendre_points(20)
        .iter()
        .map(|g| 0.5 * (b - a) * g.w * f(0.5 * (b - a) * g.x + 0.5 * (a + b)))
        .sum()
}

/// The chamfered cap's volume: below the floor the whole disc, above it the
/// cross-section at each height. With `rim`, the cone cuts the disc back to
/// `rho(z)` above `H - d`, and stops cutting the corners where `rho` falls
/// to the side's contact's distance from the axis.
fn chamfered_volume(d: f64, rim: bool) -> f64 {
    let (s, w, _) = ruling_section(d);
    let rho = |z: f64| {
        if rim && z > H - d {
            R - (z - (H - d))
        } else {
            R
        }
    };
    let section = |z: f64| {
        let r = rho(z);
        2.0 * (segment_beyond_slot(r) - 2.0 * corner_cut(r, s, w))
    };
    let lower = PI * R * R * FLOOR;
    if !rim {
        return lower + integral(FLOOR, H, section);
    }
    let reach = H - d + (R - s[0].hypot(s[1]));
    lower
        + integral(FLOOR, H - d, section)
        + integral(H - d, reach, section)
        + integral(reach, H, section)
}

fn plain_volume() -> f64 {
    let band = 2.0 * (C * (R * R - C * C).sqrt() + R * R * (C / R).asin());
    PI * R * R * H - band * (H - FLOOR)
}

/// How many of the faces of `solid` are cones, cylinders and planes.
fn census(topo: &Topology, solid: SolidId) -> (usize, usize, usize) {
    let mut counts = (0, 0, 0);
    for f in solid_faces(topo, solid).unwrap() {
        match topo.face(f).unwrap().surface() {
            FaceSurface::Cone(_) => counts.0 += 1,
            FaceSurface::Cylinder(_) => counts.1 += 1,
            FaceSurface::Plane { .. } => counts.2 += 1,
            other => panic!("unexpected {}", other.type_tag()),
        }
    }
    counts
}

/// Valid, watertight, oriented outward, `expected` in volume, and holding
/// each of `kept` but none of `cut`, given in the disc's own frame.
fn assert_exact(
    topo: &Topology,
    result: SolidId,
    expected: f64,
    pose: &Mat4,
    kept: &[[f64; 3]],
    cut: &[[f64; 3]],
    what: &str,
) {
    let report = validate_solid(topo, result).unwrap();
    assert!(report.is_valid(), "{what}: {:?}", report.issues);
    let exact = solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (exact - expected).abs() < 1e-9 * expected,
        "{what}: volume {exact}, expected {expected}"
    );
    assert!(
        is_watertight(&tessellate_solid(topo, result, 0.01).unwrap()),
        "{what}: its mesh is open"
    );
    let meshed = oriented_solid_volume(topo, result, 0.001).unwrap();
    assert!(
        (meshed - expected).abs() < 1e-3 * expected,
        "{what}: meshed volume {meshed}, expected {expected}"
    );
    let class = |p: [f64; 3]| {
        let at = pose.mul_point(Point3::new(p[0], p[1], p[2]));
        classify_point(topo, result, at, &ClassifyOptions::default()).unwrap()
    };
    for &p in kept {
        assert_eq!(
            class(p),
            PointClassification::Inside,
            "{what}: {p:?} is kept"
        );
    }
    for &p in cut {
        assert_eq!(
            class(p),
            PointClassification::Outside,
            "{what}: {p:?} is cut"
        );
    }
}

/// The example's pick: the top rim's arcs and the four rulings at the slot's
/// ends, chamfered by 0.6. Each slot corner's cone and plane meet in a
/// hyperbola, and the cone takes a sliver of the slot side.
#[test]
fn the_top_rim_and_the_slots_end_edges_take_one_chamfer() {
    let d = 0.6;
    let expected = chamfered_volume(d, true);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let cap = slotted_disc(&mut topo, &pose);
        let edges = pick(&topo, cap, &pose, true);
        assert_eq!(edges.len(), 8, "{name}: four arcs and four rulings");

        let result = chamfer(&mut topo, cap, &edges, d).unwrap();

        let x0 = (R * R - C * C).sqrt();
        assert_exact(
            &topo,
            result,
            expected,
            &pose,
            &[
                [0.0, 8.7, 2.9],
                [x0 - 0.65, 1.0, 2.5],
                [x0 - 0.3, 0.9, 1.9],
                [-x0 + 0.65, -1.0, 2.5],
            ],
            &[
                [0.0, 9.3, 2.95],
                [x0 - 0.05, 0.85, 2.5],
                [-(x0 - 0.05), -0.85, 2.5],
                [x0 - 0.05, 0.85, 2.95],
            ],
            name,
        );
        assert_eq!(census(&topo, result), (4, 1, 10), "{name}");
    }
}

/// The rulings alone: each takes off its prism, which ends square on the top
/// and, at the slot's floor, hands the floor the triangle it bares.
#[test]
fn the_slots_end_edges_alone_take_off_their_prisms() {
    let d = 0.6;
    let (_, _, area) = ruling_section(d);
    let expected = plain_volume() - 4.0 * area * (H - FLOOR);
    assert!((expected - chamfered_volume(d, false)).abs() < 1e-9 * expected);
    for (name, pose) in poses() {
        let mut topo = Topology::new();
        let cap = slotted_disc(&mut topo, &pose);
        let edges = pick(&topo, cap, &pose, false);
        assert_eq!(edges.len(), 4, "{name}");

        let result = chamfer(&mut topo, cap, &edges, d).unwrap();

        let x0 = (R * R - C * C).sqrt();
        assert_exact(
            &topo,
            result,
            expected,
            &pose,
            &[[x0 - 0.65, 1.0, 2.5], [x0 - 0.1, 0.85, 1.95]],
            &[[x0 - 0.05, 0.85, 2.5], [x0 - 0.05, 0.85, 2.99]],
            name,
        );
        assert_eq!(census(&topo, result), (0, 1, 10), "{name}");
    }
}

/// A smaller chamfer on the same pick, whose corners meet in the same way.
#[test]
fn a_smaller_chamfer_of_the_same_pick_meets_the_same_way() {
    let d = 0.25;
    let pose = Mat4::translation(0.0, 0.0, 0.0);
    let mut topo = Topology::new();
    let cap = slotted_disc(&mut topo, &pose);
    let edges = pick(&topo, cap, &pose, true);

    let result = chamfer(&mut topo, cap, &edges, d).unwrap();

    assert_exact(
        &topo,
        result,
        chamfered_volume(d, true),
        &pose,
        &[[0.0, 9.0, 2.9]],
        &[[0.0, 9.35, 2.97]],
        "d 0.25",
    );
}

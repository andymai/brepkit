//! Two rounds along a prism's top outline meeting at a corner whose walls
//! are not square to each other, the corner's vertical edge left sharp.
//!
//! The rounds meet as a moulding mitred at the corner: each round is its
//! cross-section swept along its edge up to the plane bisecting the
//! corner, and the two meet along their section by that plane, from the
//! top face down to the pinch a radius below the corner, and no corner
//! patch closes it. Each round's contact on the top face ends where the
//! two contacts cross, `r cot(theta / 2)` along its edge between plane
//! walls: that much of the edge the corner takes.
//!
//! The prism is a kite: corner `V` of angle `theta` between the rounded
//! edges, the edges' far ends square to the walls after them, so each
//! round ends square on its end wall and notches it. Along edge `a`, a
//! point of the round's cross-section at inset `w` from its wall runs from
//! the bisector, `w cot(theta / 2)` from `V`, to the far end, `L` from `V`.
//! The cross-section (a square of side `r` less its quarter disc) has area
//! `r^2 (1 - pi / 4)` and first moment in `w` of `r^3 (5 / 6 - pi / 4)`,
//! so the two rounds take off
//! `2 L r^2 (1 - pi / 4) - 2 cot(theta / 2) r^3 (5 / 6 - pi / 4)`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_blend::BlendError;
use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::OperationsError;
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::transform::transform_solid;
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const L: f64 = 10.0;
const H: f64 = 6.0;

/// The polygon `points` in `z = 0`, wound counter-clockwise, extruded `H`
/// up `z`.
fn prism(topo: &mut Topology, points: &[Point3]) -> SolidId {
    let ids: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], EdgeCurve::Line);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The edge of `solid` from `a` to `b` (either way round).
fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
    let near = |p: Point3, q: Point3| (p - q).length() < 1e-9;
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            (near(p, a) && near(q, b)) || (near(p, b) && near(q, a))
        })
        .expect("the edge")
}

/// The kite with its corner `V` at the origin, of angle `theta` between
/// its edges `V -> A` and `V -> B`, each `l` long, square to the walls
/// `A -> C` and `B -> C`.
fn kite(theta: f64, l: f64) -> [Point3; 4] {
    let h = theta / 2.0;
    [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(l * h.cos(), -l * h.sin(), 0.0),
        Point3::new(l / h.cos(), 0.0, 0.0),
        Point3::new(l * h.cos(), l * h.sin(), 0.0),
    ]
}

/// What the two mitred rounds of radius `r` take off the kite of angle
/// `theta` and edges `l` long.
fn kite_rounds(theta: f64, l: f64, r: f64) -> f64 {
    let cot = 1.0 / (theta / 2.0).tan();
    2.0 * l * r * r * (1.0 - PI / 4.0) - 2.0 * cot * r.powi(3) * (5.0 / 6.0 - PI / 4.0)
}

/// `∫_0^r f(w) dt` over a round's quarter, where `w = r - sqrt(r^2 - (r -
/// t)^2)` is how far the round is inset from its wall at depth `t` below
/// the top: the volume between the top and a radius below it of the
/// section whose area at inset `w` is `f(w)`.
fn over_the_round(r: f64, f: impl Fn(f64) -> f64) -> f64 {
    const NODES: [(f64, f64); 8] = [
        (-0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
        (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
    ];
    let panels = 64;
    let width = FRAC_PI_2 / f64::from(panels);
    let mut total = 0.0;
    for panel in 0..panels {
        let mid = width * (f64::from(panel) + 0.5);
        for (x, weight) in NODES {
            // `t = r (1 - sin phi)`, so `w = r (1 - cos phi)`.
            let phi = mid + 0.5 * width * x;
            total += 0.5 * width * weight * f(r * (1.0 - phi.cos())) * r * phi.cos();
        }
    }
    total
}

/// Rounds `edges` of `solid` by `r` and checks the result: valid,
/// watertight, `faces` faces (the rounds alone added, no corner patch),
/// and `volume` within `1e-6` of it.
fn assert_rounds_to(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    r: f64,
    faces: usize,
    volume: f64,
    case: &str,
) {
    let result = fillet_v2(topo, solid, edges, r).unwrap();
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    assert_eq!(
        solid_faces(topo, result.solid).unwrap().len(),
        faces,
        "{case}: face count"
    );
    let got = solid_volume(topo, result.solid, 1e-3).unwrap();
    assert!(
        (got - volume).abs() <= 1e-6 * volume,
        "{case}: volume {got} against {volume}"
    );
}

/// The edges of `solid` along its top face.
fn top_edges(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - H).abs() < 1e-9)
        })
        .collect()
}

#[test]
fn rounds_meeting_at_a_corner_not_square_are_mitred() {
    for theta_deg in [60.0_f64, 75.0, 120.0, 140.0] {
        let theta = theta_deg.to_radians();
        for r in [1.0, 2.0] {
            let case = format!("theta = {theta_deg}, r = {r}");
            let mut topo = Topology::new();
            let [v, b, _, a] = kite(theta, L);
            let solid = prism(&mut topo, &kite(theta, L));
            let up = Vec3::new(0.0, 0.0, H);
            let edges = [
                edge_between(&topo, solid, v + up, a + up),
                edge_between(&topo, solid, v + up, b + up),
            ];
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let faces_before = solid_faces(&topo, solid).unwrap().len();

            let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
            assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{case}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{case}: mesh not watertight");
            assert_eq!(
                solid_faces(&topo, result.solid).unwrap().len(),
                faces_before + 2,
                "{case}: the two rounds alone are added, no corner patch"
            );

            let removed = kite_rounds(theta, L, r);
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            assert!(
                (lost - removed).abs() <= 1e-6 * before,
                "{case}: lost {lost}, the mitred rounds {removed}"
            );
        }
    }
}

/// A regular hexagonal prism, side 2, all six top edges rounded: each
/// round is mitred at both its ends, each end taking `r cot(60 deg)` of its
/// edge, so a radius past half the side still fits. At inset `w` the
/// section is the hexagon of inradius `sqrt(3) - w`.
#[test]
fn a_hexagon_s_top_rounds_mitre_at_both_ends() {
    let hexagon: Vec<Point3> = (0..6)
        .map(|k| {
            let a = f64::from(k) * PI / 3.0;
            Point3::new(2.0 * a.cos(), 2.0 * a.sin(), 0.0)
        })
        .collect();
    let area = |w: f64| 2.0 * 3.0_f64.sqrt() * (3.0_f64.sqrt() - w).powi(2);
    for r in [1.0, 1.5] {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &hexagon);
        let edges = top_edges(&topo, solid);
        assert_eq!(edges.len(), 6);
        let volume = area(0.0) * (H - r) + over_the_round(r, area);
        assert_rounds_to(&mut topo, solid, &edges, r, 14, volume, &format!("r = {r}"));
    }
}

/// A kite of 140 deg, its rounded edges 1.5 long, rounded by 1.6: the
/// mitre takes `1.6 cot(70 deg)`, about 0.58, of each edge, so the rounds
/// fit though the radius is longer than the edges.
#[test]
fn a_short_obtuse_kite_takes_a_radius_longer_than_its_edges() {
    let (theta, l, r) = (140.0_f64.to_radians(), 1.5, 1.6);
    let mut topo = Topology::new();
    let [v, b, _, a] = kite(theta, l);
    let solid = prism(&mut topo, &kite(theta, l));
    let up = Vec3::new(0.0, 0.0, H);
    let edges = [
        edge_between(&topo, solid, v + up, a + up),
        edge_between(&topo, solid, v + up, b + up),
    ];
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let volume = before - kite_rounds(theta, l, r);
    assert_rounds_to(&mut topo, solid, &edges, r, 8, volume, "kite");
}

/// A rhombus of side 10 and corners of 60 and 120 deg, its four top edges
/// rounded: each edge's two ends take `r (cot(30 deg) + cot(60 deg))`, all
/// of it at `r = 5 sqrt(3) / 2`, the rhombus's inradius. A radius past that
/// is refused before the build starts, naming that limit.
#[test]
fn a_rhombus_refuses_a_radius_its_mitres_overrun() {
    let s3 = 3.0_f64.sqrt();
    let rhombus = [
        Point3::new(0.0, 0.0, 0.0),
        Point3::new(10.0, 0.0, 0.0),
        Point3::new(15.0, 5.0 * s3, 0.0),
        Point3::new(5.0, 5.0 * s3, 0.0),
    ];
    let limit = 2.5 * s3;
    for r in [4.5, 6.0] {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &rhombus);
        let edges = top_edges(&topo, solid);
        assert_eq!(edges.len(), 4);
        let result = fillet_v2(&mut topo, solid, &edges, r);
        let max_radius = match &result {
            Err(OperationsError::Blend(BlendError::RadiusTooLarge { max_radius, .. })) => {
                Some(*max_radius)
            }
            _ => None,
        };
        assert!(
            max_radius.is_some_and(|max| (max - limit).abs() <= 1e-9 * limit),
            "r = {r}: {max_radius:?} against the limit {limit}, {:?}",
            result.map(|built| built.failed)
        );
    }
    let area = |w: f64| 10.0 * 5.0 * s3 * (1.0 - w / limit).powi(2);
    let r = 4.0;
    let mut topo = Topology::new();
    let solid = prism(&mut topo, &rhombus);
    let edges = top_edges(&topo, solid);
    let volume = area(0.0) * (H - r) + over_the_round(r, area);
    assert_rounds_to(&mut topo, solid, &edges, r, 10, volume, "r = 4");
}

/// The kite of 75 deg, turned off every axis and moved off the origin
/// before its rounds: the mitre reads its corner in the part's own frame.
#[test]
fn a_turned_kite_s_rounds_are_mitred() {
    let theta = 75.0_f64.to_radians();
    let place = Mat4::translation(12.5, -4.25, 7.75)
        * Mat4::rotation_x(0.7)
        * Mat4::rotation_z(0.4)
        * Mat4::rotation_y(-1.1);
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let [v, b, _, a] = kite(theta, L);
        let solid = prism(&mut topo, &kite(theta, L));
        transform_solid(&mut topo, solid, &place).unwrap();
        let up = Vec3::new(0.0, 0.0, H);
        let at = |p: Point3| place.mul_point(p + up);
        let edges = [
            edge_between(&topo, solid, at(v), at(a)),
            edge_between(&topo, solid, at(v), at(b)),
        ];
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let volume = before - kite_rounds(theta, L, r);
        assert_rounds_to(&mut topo, solid, &edges, r, 8, volume, &format!("r = {r}"));
    }
}

/// A prism whose corner `V` joins a flat wall and a round one: the
/// profile runs from `V` along `x` to `A`, up to the round wall's axis `O`,
/// out along a radius to `B`, and round the wall, a circle of radius 10
/// about `O`, back to `V`, meeting the flat wall there at 120 deg. Both
/// walls end square to their round's edge, at `A` and at `B`. At inset `w`
/// the section is the profile with its flat wall moved to `y = w` and its
/// round wall's radius cut by `w`: the triangle `V' A' O` and the sector of
/// `O` from `B'` to `V'`.
#[test]
fn a_flat_and_a_round_wall_s_rounds_are_mitred() {
    let (big, theta) = (10.0_f64, 120.0_f64.to_radians());
    let centre = (big * theta.sin(), -big * theta.cos());
    let to_b = theta + FRAC_PI_2 - PI / 3.0;
    let b = (
        big.mul_add(to_b.cos(), centre.0),
        big.mul_add(to_b.sin(), centre.1),
    );
    let area = |w: f64| {
        let reach = big - w;
        let x = centre.0 - (reach * reach - (w - centre.1).powi(2)).sqrt();
        let to_v = (w - centre.1).atan2(x - centre.0) + 2.0 * PI;
        0.5 * (centre.0 - x) * (centre.1 - w) + 0.5 * reach * reach * (to_v - to_b)
    };
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let corner = |topo: &mut Topology, (x, y): (f64, f64)| {
            topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7))
        };
        let v = corner(&mut topo, (0.0, 0.0));
        let a = corner(&mut topo, (centre.0, 0.0));
        let o = corner(&mut topo, centre);
        let b_id = corner(&mut topo, b);
        let wall = Circle3D::new(
            Point3::new(centre.0, centre.1, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            big,
        )
        .unwrap();
        let edges: Vec<_> = [
            (v, a, EdgeCurve::Line),
            (a, o, EdgeCurve::Line),
            (o, b_id, EdgeCurve::Line),
            (b_id, v, EdgeCurve::Circle(wall)),
        ]
        .into_iter()
        .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true))
        .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal, d: 0.0 },
        ));
        let solid = extrude(&mut topo, face, normal, H).unwrap();
        let top_v = Point3::new(0.0, 0.0, H);
        let rounded: Vec<EdgeId> = top_edges(&topo, solid)
            .into_iter()
            .filter(|&edge_id| {
                let edge = topo.edge(edge_id).unwrap();
                [edge.start(), edge.end()]
                    .iter()
                    .any(|&end| (topo.vertex(end).unwrap().point() - top_v).length() < 1e-9)
            })
            .collect();
        assert_eq!(rounded.len(), 2);
        let volume = area(0.0) * (H - r) + over_the_round(r, area);
        assert_rounds_to(
            &mut topo,
            solid,
            &rounded,
            r,
            8,
            volume,
            &format!("r = {r}"),
        );
    }
}

/// Splits the edge of `solid` from `a` to `b` (either way round) at the
/// point `along` from `a`, in every face it bounds, as a file can carry an
/// edge in collinear pieces: the pieces keep the same two faces, so the
/// fillet's plan joins them into one contour.
fn split_edge(topo: &mut Topology, solid: SolidId, (a, b): (Point3, Point3), along: f64) {
    let edge_id = edge_between(topo, solid, a, b);
    let edge = topo.edge(edge_id).unwrap().clone();
    let (start, end) = (
        topo.vertex(edge.start()).unwrap().point(),
        topo.vertex(edge.end()).unwrap().point(),
    );
    let from_a = (start - a).length() < 1e-9;
    let split = a + (b - a) * (along / (b - a).length());
    let middle = topo.add_vertex(Vertex::new(split, 1e-7));
    let first = topo.add_edge(Edge::new(edge.start(), middle, EdgeCurve::Line));
    let second = topo.add_edge(Edge::new(middle, edge.end(), EdgeCurve::Line));
    assert!(from_a || (end - a).length() < 1e-9);
    for face_id in solid_faces(topo, solid).unwrap() {
        let wire_id = topo.face(face_id).unwrap().outer_wire();
        let edges = topo.wire(wire_id).unwrap().edges().to_vec();
        let Some(at) = edges.iter().position(|oe| oe.edge() == edge_id) else {
            continue;
        };
        let pieces = if edges[at].is_forward() {
            [
                OrientedEdge::new(first, true),
                OrientedEdge::new(second, true),
            ]
        } else {
            [
                OrientedEdge::new(second, false),
                OrientedEdge::new(first, false),
            ]
        };
        let mut split_edges = edges[..at].to_vec();
        split_edges.extend(pieces);
        split_edges.extend_from_slice(&edges[at + 1..]);
        let wire = topo.add_wire(Wire::new(split_edges, true).unwrap());
        topo.face_mut(face_id).unwrap().set_outer_wire(wire);
    }
}

/// The kite of 120 deg, its edge `V -> A` carried in two pieces split 0.25
/// from the corner: the mitre takes `cot(60 deg)`, about 0.58, of the edge
/// at r = 1, past the first piece, and fits the whole edge as it does
/// unsplit.
#[test]
fn a_kite_s_round_split_near_its_corner_is_mitred() {
    let theta = 120.0_f64.to_radians();
    for r in [1.0, 2.0] {
        let mut topo = Topology::new();
        let [v, b, _, a] = kite(theta, L);
        let solid = prism(&mut topo, &kite(theta, L));
        let up = Vec3::new(0.0, 0.0, H);
        split_edge(&mut topo, solid, (v + up, a + up), 0.25);
        let pieces: Vec<EdgeId> = top_edges(&topo, solid)
            .into_iter()
            .filter(|&edge_id| {
                let edge = topo.edge(edge_id).unwrap();
                let [p, q] =
                    [edge.start(), edge.end()].map(|end| topo.vertex(end).unwrap().point());
                let on = |s: Point3| ((s - (v + up)).cross(a - v)).length() < 1e-9;
                on(p) && on(q)
            })
            .collect();
        assert_eq!(pieces.len(), 2);
        let mut edges = pieces;
        edges.push(edge_between(&topo, solid, v + up, b + up));
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let volume = before - kite_rounds(theta, L, r);
        assert_rounds_to(&mut topo, solid, &edges, r, 8, volume, &format!("r = {r}"));
    }
}

/// The hexagon of side 2 with one side carried in two pieces split 0.3
/// from a corner: at r = 1.5 that corner's mitre takes about 0.87 of the
/// side, past the short piece, and the rounds still fit.
#[test]
fn a_hexagon_with_a_split_side_rounds_as_whole() {
    let hexagon: Vec<Point3> = (0..6)
        .map(|k| {
            let a = f64::from(k) * PI / 3.0;
            Point3::new(2.0 * a.cos(), 2.0 * a.sin(), 0.0)
        })
        .collect();
    let area = |w: f64| 2.0 * 3.0_f64.sqrt() * (3.0_f64.sqrt() - w).powi(2);
    let up = Vec3::new(0.0, 0.0, H);
    for r in [1.0, 1.5] {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &hexagon);
        split_edge(&mut topo, solid, (hexagon[0] + up, hexagon[1] + up), 0.3);
        let edges = top_edges(&topo, solid);
        assert_eq!(edges.len(), 7);
        let volume = area(0.0) * (H - r) + over_the_round(r, area);
        assert_rounds_to(&mut topo, solid, &edges, r, 14, volume, &format!("r = {r}"));
    }
}

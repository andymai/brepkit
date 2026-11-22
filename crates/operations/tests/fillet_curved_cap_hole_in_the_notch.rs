//! A round run on to a curved end face, with a hole in that face near the
//! corner.
//!
//! Two parts are rounded. One is a D: an arc of radius `R` about the `z`
//! axis, from angle `-alpha` to `alpha`, closed by a wall through `(M, 0)`,
//! and extruded `H` up `z`. The wall is a circle's arc (a cylinder face) or
//! a quadratic Bezier (a B-spline face). Its top arc edge is rounded by a
//! torus. The other is the prism of `fillet_walked_tilted_caps.rs` with its
//! side from `B` to `C` an arc that bows out (a cylinder face). Its top
//! B-spline edge is rounded by a walked band. Each round runs on to the
//! curved wall at its end, and the wall is notched along the band's end.
//!
//! A blind hole drilled square into the wall inside that notch lies in the
//! material the round takes off. The notch would leave the hole's wire
//! outside the wall's new outline, so the round is refused. The same hole
//! drilled well clear of the notch stays, and the round takes off as much
//! as from the part without the hole.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::boolean::{BooleanOp, boolean};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::primitives::make_cylinder;
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

const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

/// A prism `height` up `z` on the plan through `corners`, counter-clockwise
/// about `+z`: `curves[i]` runs from corner `i` to the next.
fn prism(topo: &mut Topology, corners: &[Point3], curves: Vec<EdgeCurve>, height: f64) -> SolidId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let edges: Vec<_> = curves
        .into_iter()
        .enumerate()
        .map(|(i, curve)| {
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal: UP, d: 0.0 },
    ));
    extrude(topo, face, UP, height).unwrap()
}

/// `solid` drilled square into a wall at `point` (on the wall, at `z = 0`),
/// `inward` the wall's normal into the part there: a hole of radius `rho`,
/// its axis `down` below `top`, `depth` deep.
fn drilled(
    topo: &mut Topology,
    solid: SolidId,
    (point, inward): (Point3, Vec3),
    top: f64,
    down: f64,
    rho: f64,
    depth: f64,
) -> SolidId {
    // The drill starts outside the wall and runs `depth` in.
    let drill = make_cylinder(topo, rho, depth + 1.0).unwrap();
    transform_solid(topo, drill, &Mat4::rotation_y(FRAC_PI_2)).unwrap();
    transform_solid(topo, drill, &Mat4::rotation_z(inward.y().atan2(inward.x()))).unwrap();
    let start = point - inward;
    transform_solid(
        topo,
        drill,
        &Mat4::translation(start.x(), start.y(), top - down),
    )
    .unwrap();
    boolean(topo, BooleanOp::Cut, solid, drill).unwrap()
}

/// The edges of `solid` at height `top` whose curve `keep` takes.
fn top_edges(
    topo: &Topology,
    solid: SolidId,
    top: f64,
    keep: impl Fn(&EdgeCurve) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            keep(edge.curve())
                && [edge.start(), edge.end()]
                    .iter()
                    .all(|&v| (topo.vertex(v).unwrap().point().z() - top).abs() < 1e-9)
        })
        .collect()
}

/// Rounds `edges` at `r` and checks the round is refused for the hole in
/// its notch.
fn refused(topo: &mut Topology, solid: SolidId, edges: &[EdgeId], r: f64, case: &str) {
    assert!(
        validate_solid(topo, solid).unwrap().is_valid(),
        "{case}: the drilled part"
    );
    let result = fillet_v2(topo, solid, edges, r);
    assert!(
        result
            .as_ref()
            .is_err_and(|e| e.to_string().contains("cross another boundary")),
        "{case}: {:?}",
        result.map(|blended| blended.failed)
    );
}

/// Rounds `edges` at `r` and checks the result: valid, watertight, the
/// round its one new face, and `want` taken off.
fn rounded(topo: &mut Topology, solid: SolidId, edges: &[EdgeId], r: f64, want: f64, case: &str) {
    let before = solid_volume(topo, solid, 1e-4).unwrap();
    let faces_before = solid_faces(topo, solid).unwrap().len();
    let result = fillet_v2(topo, solid, edges, r).unwrap();
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    let faces = solid_faces(topo, result.solid).unwrap().len();
    assert_eq!(faces, faces_before + 1, "{case}: the round alone is added");
    let lost = before - solid_volume(topo, result.solid, 1e-4).unwrap();
    assert!(
        (lost - want).abs() <= 1e-4 * want,
        "{case}: lost {lost}, without the hole {want}"
    );
}

/// The D, its top arc rounded by a torus.
mod d {
    use super::*;

    const R: f64 = 10.0;
    pub const H: f64 = 6.0;
    const M: f64 = 3.0;
    const ALPHA_DEG: f64 = 60.0;

    /// The arc's ends: at `alpha`, where the holes go, and at `-alpha`.
    fn ends() -> (Point3, Point3) {
        let alpha = ALPHA_DEG.to_radians();
        let (c, s) = (R * alpha.cos(), R * alpha.sin());
        (Point3::new(c, s, 0.0), Point3::new(c, -s, 0.0))
    }

    /// The circular wall's centre on the `x` axis and its radius: the
    /// circle through both arc ends and `(M, 0)`.
    fn wall_circle() -> (f64, f64) {
        let (end, _) = ends();
        // `(x - k)^2 + y^2 = (M - k)^2` at the arc's end.
        let k = (end.x() * end.x() + end.y() * end.y() - M * M) / (2.0 * (end.x() - M));
        (k, k - M)
    }

    /// The Bezier wall's control points, from the arc's end at `alpha` to
    /// the one at `-alpha`: the middle of a quadratic Bezier is a quarter
    /// of each end plus half the control point.
    fn bezier() -> [Point3; 3] {
        let (top, bottom) = ends();
        [
            top,
            Point3::new(2.0f64.mul_add(M, -top.x()), 0.0, 0.0),
            bottom,
        ]
    }

    pub fn solid(topo: &mut Topology, spline: bool) -> SolidId {
        let (top, bottom) = ends();
        let arc = Circle3D::new(Point3::new(0.0, 0.0, 0.0), UP, R).unwrap();
        let wall = if spline {
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    bezier().to_vec(),
                    vec![1.0, 1.0, 1.0],
                )
                .unwrap(),
            )
        } else {
            let (k, radius) = wall_circle();
            EdgeCurve::Circle(Circle3D::new(Point3::new(k, 0.0, 0.0), UP, radius).unwrap())
        };
        prism(topo, &[bottom, top], vec![EdgeCurve::Circle(arc), wall], H)
    }

    /// The wall's point `along` its plan from the arc's end at `alpha`,
    /// and its normal into the part there.
    pub fn on_wall(spline: bool, along: f64) -> (Point3, Vec3) {
        let (top, _) = ends();
        if !spline {
            let (k, radius) = wall_circle();
            let theta = top.y().atan2(top.x() - k) + along / radius;
            let (s, c) = theta.sin_cos();
            return (
                Point3::new(radius.mul_add(c, k), radius * s, 0.0),
                Vec3::new(-c, -s, 0.0),
            );
        }
        let [b0, b1, b2] = bezier();
        let at = |t: f64| {
            let s = 1.0 - t;
            Point3::new(
                s * s * b0.x() + 2.0 * s * t * b1.x() + t * t * b2.x(),
                s * s * b0.y() + 2.0 * s * t * b1.y() + t * t * b2.y(),
                0.0,
            )
        };
        let step = 1e-6;
        let (mut t, mut length) = (0.0, 0.0);
        for _ in 0..1_000_000 {
            if length >= along {
                break;
            }
            length += (at(t + step) - at(t)).length();
            t += step;
        }
        let tangent = (at(t + step) - at(t)).normalize().unwrap();
        // The part lies to the left of the wall's run.
        (at(t), Vec3::new(-tangent.y(), tangent.x(), 0.0))
    }

    /// The top arc's pieces: a drill's cut splits it in two.
    pub fn top_arc(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
        top_edges(
            topo,
            solid,
            H,
            |curve| matches!(curve, EdgeCurve::Circle(circle) if (circle.radius() - R).abs() < 1e-9),
        )
    }
}

/// The walked prism with its side from `B` to `C` bowed out, its top
/// B-spline edge rounded by a walked band.
mod walked {
    use super::*;

    pub const H: f64 = 6.0;
    const A: (f64, f64) = (0.0, 0.0);
    const K: (f64, f64) = (10.0, -2.0);
    const B: (f64, f64) = (20.0, 0.0);
    const C: (f64, f64) = (22.0, 8.0);
    const D: (f64, f64) = (3.0, 8.0);
    /// How far the side from `B` to `C` bows out at its middle.
    const SAG: f64 = 0.5;

    fn point((x, y): (f64, f64)) -> Point3 {
        Point3::new(x, y, 0.0)
    }

    /// The circle of the side from `B` to `C`: its centre and radius.
    fn side_circle() -> (Point3, f64) {
        let chord = (C.0 - B.0).hypot(C.1 - B.1);
        let (tx, ty) = ((C.0 - B.0) / chord, (C.1 - B.1) / chord);
        let radius = chord.mul_add(chord / 4.0, SAG * SAG) / (2.0 * SAG);
        // The centre lies inward of the chord's middle, the part's side.
        let inset = radius - SAG;
        let middle = (0.5 * (B.0 + C.0), 0.5 * (B.1 + C.1));
        (
            Point3::new(
                ty.mul_add(-inset, middle.0),
                tx.mul_add(inset, middle.1),
                0.0,
            ),
            radius,
        )
    }

    pub fn solid(topo: &mut Topology) -> SolidId {
        let spline = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![point(A), point(K), point(B)],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        let (centre, radius) = side_circle();
        let side = Circle3D::new(centre, UP, radius).unwrap();
        prism(
            topo,
            &[point(A), point(B), point(C), point(D)],
            vec![
                EdgeCurve::NurbsCurve(spline),
                EdgeCurve::Circle(side),
                EdgeCurve::Line,
                EdgeCurve::Line,
            ],
            H,
        )
    }

    /// The bowed side's point `along` it from `B`, and its normal into the
    /// part there.
    pub fn on_side(along: f64) -> (Point3, Vec3) {
        let (centre, radius) = side_circle();
        let theta = (B.1 - centre.y()).atan2(B.0 - centre.x()) + along / radius;
        let (s, c) = theta.sin_cos();
        (
            Point3::new(
                radius.mul_add(c, centre.x()),
                radius.mul_add(s, centre.y()),
                0.0,
            ),
            Vec3::new(-c, -s, 0.0),
        )
    }

    pub fn top_spline(topo: &Topology, solid: SolidId) -> Vec<EdgeId> {
        top_edges(topo, solid, H, |curve| {
            matches!(curve, EdgeCurve::NurbsCurve(_))
        })
    }
}

#[test]
fn a_hole_in_a_curved_walls_notch_refuses_a_torus_round() {
    // Each hole lies wholly between the round and the corner, as in a
    // square corner: its centre `a` from both corner edges, and
    // `sqrt(2) (r - a) > r + rho`.
    for spline in [false, true] {
        for (r, a, rho) in [(2.0, 0.4, 0.1), (3.0, 0.5, 0.2)] {
            let case = format!("spline wall {spline}, r = {r}");
            let mut topo = Topology::new();
            let solid = d::solid(&mut topo, spline);
            let solid = drilled(&mut topo, solid, d::on_wall(spline, a), d::H, a, rho, 0.5);
            let edges = d::top_arc(&topo, solid);
            refused(&mut topo, solid, &edges, r, &case);
        }
    }
}

#[test]
fn a_hole_clear_of_a_curved_walls_notch_stays_under_a_torus_round() {
    for spline in [false, true] {
        for r in [2.0, 3.0] {
            let case = format!("spline wall {spline}, r = {r}");
            let mut topo = Topology::new();
            let plain = d::solid(&mut topo, spline);
            let before = solid_volume(&topo, plain, 1e-4).unwrap();
            let edges = d::top_arc(&topo, plain);
            let result = fillet_v2(&mut topo, plain, &edges, r).unwrap();
            let want = before - solid_volume(&topo, result.solid, 1e-4).unwrap();

            let mut topo = Topology::new();
            let solid = d::solid(&mut topo, spline);
            let solid = drilled(
                &mut topo,
                solid,
                d::on_wall(spline, 4.5),
                d::H,
                4.0,
                0.3,
                1.0,
            );
            let edges = d::top_arc(&topo, solid);
            rounded(&mut topo, solid, &edges, r, want, &case);
        }
    }
}

#[test]
fn a_hole_in_a_curved_walls_notch_refuses_a_walked_round() {
    let r = 2.0;
    let (a, rho) = (0.3, 0.1);
    let mut topo = Topology::new();
    let solid = walked::solid(&mut topo);
    let solid = drilled(&mut topo, solid, walked::on_side(a), walked::H, a, rho, 0.5);
    let edges = walked::top_spline(&topo, solid);
    refused(&mut topo, solid, &edges, r, "walked round");
}

#[test]
fn a_hole_clear_of_a_curved_walls_notch_stays_under_a_walked_round() {
    for r in [1.0, 2.0] {
        let case = format!("walked round, r = {r}");
        let mut topo = Topology::new();
        let plain = walked::solid(&mut topo);
        let before = solid_volume(&topo, plain, 1e-4).unwrap();
        let edges = walked::top_spline(&topo, plain);
        let result = fillet_v2(&mut topo, plain, &edges, r).unwrap();
        let want = before - solid_volume(&topo, result.solid, 1e-4).unwrap();

        let mut topo = Topology::new();
        let solid = walked::solid(&mut topo);
        let solid = drilled(
            &mut topo,
            solid,
            walked::on_side(5.0),
            walked::H,
            3.5,
            0.3,
            1.0,
        );
        let edges = walked::top_spline(&topo, solid);
        rounded(&mut topo, solid, &edges, r, want, &case);
    }
}

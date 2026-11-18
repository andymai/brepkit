//! A plate whose B-spline side runs into its round end at a kink, the side's
//! rims rounded.
//!
//! The plate is the profile below, extruded `H` up: a B-spline side from
//! `B` to `A`, a straight end at `x = 0` and a straight side at `y = -45`,
//! and a round end, an arc of the circle of radius `R` about `O` from its
//! bottom round to `B`, where the B-spline side leaves it at about 37
//! degrees, outside the circle. A round along the side's rim ends there: its
//! contact on the plate's face runs on into the circle, so the round is cut
//! back where its contact crosses the circle and ends on its intersection
//! with the end's cylinder, which runs on over the notch, keeping the round
//! end whole.
//!
//! The side stands square to the plate's faces, so a round at height `z`
//! is the side's trace set in by `w(z)` within the plate's slice there. The
//! volume a round takes off is that strip, clipped to the plate (`x >= 0`)
//! and to outside the circle, integrated over `z`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::{solid_edges, solid_faces};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const H: f64 = 20.0;
const R: f64 = 25.0;
const O: (f64, f64) = (100.0, -20.0);

/// The B-spline side, from `B` on the circle at 120 degrees to `A`.
fn side() -> NurbsCurve {
    let angle = 120.0_f64.to_radians();
    let b = Point3::new(
        R.mul_add(angle.cos(), O.0),
        R.mul_add(angle.sin(), O.1),
        0.0,
    );
    NurbsCurve::new(
        3,
        vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        vec![
            b,
            Point3::new(60.0, 5.0, 0.0),
            Point3::new(30.0, 9.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ],
        vec![1.0; 4],
    )
    .unwrap()
}

fn plate(topo: &mut Topology) -> SolidId {
    let side = side();
    let b = side.evaluate(0.0);
    let points = [
        Point3::new(0.0, 10.0, 0.0),
        Point3::new(0.0, O.1 - R, 0.0),
        Point3::new(O.0, O.1 - R, 0.0),
        b,
    ];
    let v: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
        .collect();
    let end = Circle3D::new(Point3::new(O.0, O.1, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let edges = [
        Edge::new(v[0], v[1], EdgeCurve::Line),
        Edge::new(v[1], v[2], EdgeCurve::Line),
        Edge::new(v[2], v[3], EdgeCurve::Circle(end)),
        Edge::new(v[3], v[0], EdgeCurve::NurbsCurve(side)),
    ];
    let edges: Vec<_> = edges
        .into_iter()
        .map(|edge| OrientedEdge::new(topo.add_edge(edge), true))
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

/// The side's rim at height `z`.
fn rim(topo: &Topology, solid: SolidId, z: f64) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && [edge.start(), edge.end()]
                    .iter()
                    .all(|&v| (topo.vertex(v).unwrap().point().z() - z).abs() < 1e-9)
        })
        .expect("the side's rim")
}

/// The area of the side's strip set in by `w`, clipped to `x >= 0` and to
/// outside the circle: across the side `p(u) + t n(u)`, `n` its inward unit
/// normal, the area element is `det(p' + t n', n)` and each line in from
/// the side is clipped exactly.
fn strip(w: f64) -> f64 {
    if w <= 0.0 {
        return 0.0;
    }
    let side = side();
    let at = |u: f64| {
        let p = side.evaluate(u);
        (p.x(), p.y())
    };
    let normal = |u: f64| {
        let t = side.tangent(u).unwrap();
        // The side runs from `B` to `A`, the plate on its left.
        let (x, y) = (-t.y(), t.x());
        let l = x.hypot(y);
        (x / l, y / l)
    };
    let row = |u: f64| -> f64 {
        let h = 1e-6;
        let (u0, u1) = ((u - h).max(0.0), (u + h).min(1.0));
        let (pa, pb) = (at(u0), at(u1));
        let (na, nb) = (normal(u0), normal(u1));
        let dp = ((pb.0 - pa.0) / (u1 - u0), (pb.1 - pa.1) / (u1 - u0));
        let dn = ((nb.0 - na.0) / (u1 - u0), (nb.1 - na.1) / (u1 - u0));
        let p = at(u);
        let n = normal(u);
        let det = |a: (f64, f64)| (a.0 * n.1 - a.1 * n.0).abs();
        let (a, b) = (det(dp), det(dn));
        // `t` in `[0, w]` with `x >= 0` and outside the circle.
        let mut lo = 0.0_f64;
        let mut hi = w;
        if n.0 < 0.0 {
            hi = hi.min(-p.0 / n.0);
        }
        let (dx, dy) = (p.0 - O.0, p.1 - O.1);
        let (qb, qc) = (dx * n.0 + dy * n.1, dx.mul_add(dx, dy * dy) - R * R);
        let disc = qb.mul_add(qb, -qc);
        let length = |l0: f64, l1: f64| {
            if l1 > l0 {
                a * (l1 - l0) - b * (l1 * l1 - l0 * l0) / 2.0
            } else {
                0.0
            }
        };
        if disc > 0.0 {
            let (t0, t1) = (-qb - disc.sqrt(), -qb + disc.sqrt());
            let before = length(lo, hi.min(t0));
            lo = lo.max(t1);
            before + length(lo, hi)
        } else {
            length(lo, hi)
        }
    };
    adaptive(&row, 0.0, 1.0, 1e-7)
}

/// Adaptive Simpson's rule for `f` over `[a, b]` to `eps`: the strip's rows
/// change kind where a row starts to reach the circle.
fn adaptive(f: &dyn Fn(f64) -> f64, a: f64, b: f64, eps: f64) -> f64 {
    fn step(
        f: &dyn Fn(f64) -> f64,
        (a, fa): (f64, f64),
        (m, fm): (f64, f64),
        (b, fb): (f64, f64),
        whole: f64,
        eps: f64,
        depth: usize,
    ) -> f64 {
        let (l, r) = (f64::midpoint(a, m), f64::midpoint(m, b));
        let (fl, fr) = (f(l), f(r));
        let left = (m - a) / 6.0 * (4.0f64.mul_add(fl, fa) + fm);
        let right = (b - m) / 6.0 * (4.0f64.mul_add(fr, fm) + fb);
        if depth == 0 || (left + right - whole).abs() <= 15.0 * eps {
            return left + right + (left + right - whole) / 15.0;
        }
        step(f, (a, fa), (l, fl), (m, fm), left, eps / 2.0, depth - 1)
            + step(f, (m, fm), (r, fr), (b, fb), right, eps / 2.0, depth - 1)
    }
    let m = f64::midpoint(a, b);
    let (fa, fm, fb) = (f(a), f(m), f(b));
    let whole = (b - a) / 6.0 * (4.0f64.mul_add(fm, fa) + fb);
    step(f, (a, fa), (m, fm), (b, fb), whole, eps, 20)
}

/// Simpson's rule for `f` over `[a, b]`.
fn simpson(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const N: usize = 200;
    let step = (b - a) / N as f64;
    let mut sum = f(a) + f(b);
    for i in 1..N {
        sum += f((i as f64).mul_add(step, a)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * step / 3.0
}

fn check(name: &str, picks: &[f64], r: f64, removed: f64) {
    let mut topo = Topology::new();
    let solid = plate(&mut topo);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let edges: Vec<EdgeId> = picks.iter().map(|&z| rim(&topo, solid, z)).collect();
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "{name}: {:?}", result.failed);
    let faces = solid_faces(&topo, result.solid).unwrap();
    // The plate's six faces and one round, or two rounds and no side
    // (crossing, they meet along the first's contact): no runout patch.
    assert_eq!(faces.len(), 7, "{name}: faces");
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{name}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.05).unwrap();
    assert!(is_watertight(&mesh), "{name}: mesh open");
    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    // A walked band stands off its sections between them by about 2e-5
    // of the volume here; a round ended by a flat runout patch instead
    // takes off 1.8% too much.
    assert!(
        (lost - removed).abs() <= 1e-4 * removed,
        "{name}: removed {lost}, the round takes off {removed}"
    );
}

#[test]
fn a_round_running_into_a_round_end_at_a_kink_keeps_the_end_whole() {
    for r in [6.0, 10.0] {
        // At `z = H - r + r sin t` the round is set in `r (1 - cos t)`.
        let removed = simpson(0.0, FRAC_PI_2, |t| strip(r * (1.0 - t.cos())) * r * t.cos());
        check(&format!("top rim, r = {r}"), &[H], r, removed);
    }
}

#[test]
fn crossing_rounds_running_into_a_round_end_at_a_kink_keep_the_end_whole() {
    for r in [12.0, 14.0] {
        let first = simpson(0.0, FRAC_PI_2, |t| strip(r * (1.0 - t.cos())) * r * t.cos());
        // The bottom round rests on the bottom face and pivots on the top
        // round's contact on the side, `H - r` up: its ball's centre stands
        // `reach` in from the side, and at `z = r - r cos t` it is set in
        // `reach - r sin t`, up to the pivot.
        let reach = (r * r - (H - 2.0 * r).powi(2)).sqrt();
        let pivot = ((2.0 * r - H) / r).acos();
        let second = simpson(0.0, pivot, |t| {
            strip(r.mul_add(-t.sin(), reach)) * r * t.sin()
        });
        check(&format!("both rims, r = {r}"), &[H, 0.0], r, first + second);
    }
}

//! A round of a B-spline edge whose ends meet tilted end faces.
//!
//! The part is a prism whose plan has a B-spline side (a quadratic Bezier
//! from `A` to `B`, bowing away from the part) between two straight sides
//! that lean, one making an acute corner with the spline and the other an
//! obtuse one, closed by a straight back, and extruded `H` up `z`. Its top
//! B-spline edge is rounded: the band between the top and the B-spline wall
//! is walked and fitted. At each end it meets a straight side, a plane
//! tilted to the band's end section.
//!
//! The band ended in its end section, closed by a runout patch. At the
//! acute corner a contact ran on past the end face; at the obtuse one a
//! contact stopped short of it. Now the band is carried on past its end
//! along the ball's own sections where a contact falls short, and each
//! contact is cut where it crosses the end face, on the face's corner edge;
//! the band ends on its intersection with the face, which is notched along
//! it.
//!
//! The volume taken off: the top is a plane and the wall stands square to
//! it, so at depth `h` below the top and inset `d` from the wall the round
//! takes off the plan's inset curve at `d`, clipped by the two sides,
//! wherever `(d, h)` lies in the corner square of side `r` outside its
//! quarter disc. That is the integral over `d` of the inset curve's length
//! times the profile's height there, `r - sqrt(r^2 - (r - d)^2)`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

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

const H: f64 = 6.0;

/// The plan's corners: the spline from `A` to `B` (its control point
/// `K`), then `B` to `C`, `C` to `D` and `D` back to `A`, counter-clockwise.
const A: (f64, f64) = (0.0, 0.0);
const K: (f64, f64) = (10.0, -2.0);
const B: (f64, f64) = (20.0, 0.0);
const C: (f64, f64) = (22.0, 8.0);
const D: (f64, f64) = (3.0, 8.0);

fn point((x, y): (f64, f64)) -> Point3 {
    Point3::new(x, y, 0.0)
}

/// The prism.
fn prism(topo: &mut Topology) -> SolidId {
    let corners = [A, B, C, D].map(|p| topo.add_vertex(Vertex::new(point(p), 1e-7)));
    let spline = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![point(A), point(K), point(B)],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    let edges = [
        Edge::new(corners[0], corners[1], EdgeCurve::NurbsCurve(spline)),
        Edge::new(corners[1], corners[2], EdgeCurve::Line),
        Edge::new(corners[2], corners[3], EdgeCurve::Line),
        Edge::new(corners[3], corners[0], EdgeCurve::Line),
    ]
    .map(|edge| OrientedEdge::new(topo.add_edge(edge), true));
    let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
    let normal = Vec3::new(0.0, 0.0, 1.0);
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane { normal, d: 0.0 },
    ));
    extrude(topo, face, normal, H).unwrap()
}

/// The prism's top B-spline edge.
fn top_spline(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
                && [edge.start(), edge.end()]
                    .iter()
                    .all(|&v| (topo.vertex(v).unwrap().point().z() - H).abs() < 1e-9)
        })
        .expect("the top spline edge")
}

/// Which side of the line from `p` to `q` the point `(x, y)` lies on:
/// positive to its left, inside the plan.
fn left_of(p: (f64, f64), q: (f64, f64), (x, y): (f64, f64)) -> f64 {
    (q.0 - p.0) * (y - p.1) - (q.1 - p.1) * (x - p.0)
}

/// The length of the spline's inset curve at `d`, clipped by the sides
/// `D A` and `B C`.
fn inset_length(d: f64) -> f64 {
    const STEPS: usize = 20_000;
    let at = |t: f64| {
        let s = 1.0 - t;
        let x = s * s * A.0 + 2.0 * s * t * K.0 + t * t * B.0;
        let y = s * s * A.1 + 2.0 * s * t * K.1 + t * t * B.1;
        let dx = 2.0 * s * (K.0 - A.0) + 2.0 * t * (B.0 - K.0);
        let dy = 2.0 * s * (K.1 - A.1) + 2.0 * t * (B.1 - K.1);
        let len = dx.hypot(dy);
        // The inside is to the left of the spline's run from A to B.
        (x - dy / len * d, y + dx / len * d)
    };
    let inside = |p: (f64, f64)| left_of(D, A, p) >= 0.0 && left_of(B, C, p) >= 0.0;
    // Past the ends the inset runs on along the end tangents, so sample a
    // little beyond the spline's span as lines.
    let extended = |t: f64| {
        if (0.0..=1.0).contains(&t) {
            at(t)
        } else {
            let (base, from) = if t < 0.0 {
                (0.0, 1e-6)
            } else {
                (1.0, 1.0 - 1e-6)
            };
            let p = at(base);
            let q = at(from);
            let scale = (t - base) / (from - base);
            (p.0 + (q.0 - p.0) * scale, p.1 + (q.1 - p.1) * scale)
        }
    };
    let mut length = 0.0;
    #[allow(clippy::cast_precision_loss)]
    let t_at = |k: usize| -0.25 + 1.5 * (k as f64) / (STEPS as f64);
    let mut previous = extended(t_at(0));
    for k in 1..=STEPS {
        let next = extended(t_at(k));
        let middle = (0.5 * (previous.0 + next.0), 0.5 * (previous.1 + next.1));
        if inside(middle) {
            length += (next.0 - previous.0).hypot(next.1 - previous.1);
        }
        previous = next;
    }
    length
}

/// The volume the round takes off.
fn removed(r: f64) -> f64 {
    const STEPS: usize = 400;
    let f = |d: f64| {
        let h = r - (r * r - (r - d) * (r - d)).max(0.0).sqrt();
        inset_length(d) * h
    };
    #[allow(clippy::cast_precision_loss)]
    let step = r / STEPS as f64;
    let mut sum = f(0.0) + f(r);
    for k in 1..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let d = step * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(d);
    }
    sum * step / 3.0
}

fn round_and_check(r: f64) {
    let mut topo = Topology::new();
    let solid = prism(&mut topo);
    let edge = top_spline(&topo, solid);
    let before = solid_volume(&topo, solid, 1e-3).unwrap();
    let faces_before = solid_faces(&topo, solid).unwrap().len();

    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: mesh not watertight");
    let faces = solid_faces(&topo, result.solid).unwrap().len();
    assert_eq!(
        faces,
        faces_before + 1,
        "r = {r}: the round alone is added, no runout patch"
    );

    let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
    let want = removed(r);
    assert!(
        (lost - want).abs() <= 3e-4 * want,
        "r = {r}: lost {lost}, the round's sweep {want}"
    );
}

#[test]
fn a_walked_round_runs_on_and_back_to_tilted_end_faces() {
    for r in [1.0, 2.0] {
        round_and_check(r);
    }
}

//! A ring whose top face is narrower than two rounds, both its rims rounded.
//!
//! The ring is the annulus between circles of radius `R` and `R - W` about
//! the z axis, `H` thick, its two walls B-spline faces (so each round is a
//! walked band) and each rim one closed B-spline edge. Rounded at `r`, with
//! `W < 2 r`, the two rounds' contacts on the top would cross. The round
//! picked first is made as alone, and the second rests on its own wall and
//! pivots on the first's contact on the top, which then lies on it all
//! round: the top goes, the two rounds meeting along that contact.
//!
//! The removed volume is a solid of revolution: at a distance `d` in from
//! the first wall the first round takes `r - sqrt(r^2 - (r - d)^2)` off the
//! top, and at `e` in from the second wall the second, its ball's centre
//! `r` in and `k = sqrt(r^2 - (2 r - W)^2)` down, takes
//! `k - sqrt(r^2 - (r - e)^2)` up to the pivot, `e = W - r`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::{FRAC_PI_2, PI};

use brepkit_geometry::convert::ellipse_to_nurbs;
use brepkit_math::curves::Ellipse3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::fillet_v2;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{
    is_watertight, tessellate_solid, tessellate_solid_grouped_with_tolerance,
};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

const R: f64 = 10.0;
const W: f64 = 3.0;
const H: f64 = 4.0;

/// A wall of radius `radius`: its bottom and top rims as closed B-spline
/// edges and the wall ruled between them, its seam a line where the rims
/// start. An inner wall faces the axis.
struct Wall {
    face: brepkit_topology::face::FaceId,
    bottom: EdgeId,
    top: EdgeId,
}

fn wall(topo: &mut Topology, radius: f64, inner: bool) -> Wall {
    let circle = Ellipse3D::new(
        Point3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        radius,
        radius,
    )
    .unwrap();
    let spline = ellipse_to_nurbs(&circle, 0.0, 2.0 * PI).unwrap();
    let up = Vec3::new(0.0, 0.0, H);
    let lifted = NurbsCurve::new(
        spline.degree(),
        spline.knots().to_vec(),
        spline.control_points().iter().map(|&p| p + up).collect(),
        spline.weights().to_vec(),
    )
    .unwrap();
    let surface = NurbsSurface::new(
        spline.degree(),
        1,
        spline.knots().to_vec(),
        vec![0.0, 0.0, 1.0, 1.0],
        spline
            .control_points()
            .iter()
            .map(|&p| vec![p, p + up])
            .collect(),
        spline.weights().iter().map(|&w| vec![w, w]).collect(),
    )
    .unwrap();
    let start = spline.evaluate(spline.domain().0);
    let vb = topo.add_vertex(Vertex::new(start, 1e-7));
    let vt = topo.add_vertex(Vertex::new(start + up, 1e-7));
    let bottom = topo.add_edge(Edge::new(vb, vb, EdgeCurve::NurbsCurve(spline)));
    let top = topo.add_edge(Edge::new(vt, vt, EdgeCurve::NurbsCurve(lifted)));
    let seam = topo.add_edge(Edge::new(vb, vt, EdgeCurve::Line));
    let edges = [(bottom, true), (seam, true), (top, false), (seam, false)]
        .iter()
        .map(|&(e, forward)| OrientedEdge::new(e, forward))
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let surface = FaceSurface::Nurbs(surface);
    let face = topo.add_face(if inner {
        Face::new_reversed(wire, vec![], surface)
    } else {
        Face::new(wire, vec![], surface)
    });
    Wall { face, bottom, top }
}

/// The ring, with its outer and inner top rims.
fn ring(topo: &mut Topology) -> (SolidId, EdgeId, EdgeId) {
    let outer = wall(topo, R, false);
    let inner = wall(topo, R - W, true);
    let mut wire = |edges: &[(EdgeId, bool)]| {
        let edges = edges
            .iter()
            .map(|&(e, forward)| OrientedEdge::new(e, forward))
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    };
    let (floor_outer, floor_hole) = (
        wire(&[(outer.bottom, false)]),
        wire(&[(inner.bottom, true)]),
    );
    let (lid_outer, lid_hole) = (wire(&[(outer.top, true)]), wire(&[(inner.top, false)]));
    let floor = topo.add_face(Face::new(
        floor_outer,
        vec![floor_hole],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: 0.0,
        },
    ));
    let lid = topo.add_face(Face::new(
        lid_outer,
        vec![lid_hole],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: H,
        },
    ));
    let shell = topo.add_shell(Shell::new(vec![outer.face, inner.face, floor, lid]).unwrap());
    (
        topo.add_solid(Solid::new(shell, vec![])),
        outer.top,
        inner.top,
    )
}

/// Simpson's rule for `f` over `[a, b]`.
fn simpson(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
    const N: usize = 400;
    let step = (b - a) / N as f64;
    let mut sum = f(a) + f(b);
    for i in 1..N {
        sum += f((i as f64).mul_add(step, a)) * if i % 2 == 1 { 4.0 } else { 2.0 };
    }
    sum * step / 3.0
}

/// The volume the two rounds take off, the first on the wall of radius
/// `first` (the other wall's radius `second`), each distance in from a wall
/// carried to its radius by `at`.
fn removed(r: f64, outer_first: bool) -> f64 {
    let (first_wall, inward): (f64, f64) = if outer_first { (R, -1.0) } else { (R - W, 1.0) };
    let second_wall = if outer_first { R - W } else { R };
    let at_first = |d: f64| inward.mul_add(d, first_wall);
    let at_second = |e: f64| (-inward).mul_add(e, second_wall);
    // The first round: `d = r (1 - cos t)`, taking off `r (1 - sin t)`.
    let first = simpson(0.0, FRAC_PI_2, |t| {
        let d = r * (1.0 - t.cos());
        2.0 * PI * at_first(d) * r * (1.0 - t.sin()) * r * t.sin()
    });
    // The second: `e = r (1 - cos t)`, taking off `k - r sin t`, up to the
    // pivot.
    let k = (r * r - (2.0 * r - W).powi(2)).sqrt();
    let pivot = ((2.0 * r - W) / r).acos();
    let second = simpson(0.0, pivot, |t| {
        let e = r * (1.0 - t.cos());
        2.0 * PI * at_second(e) * r.mul_add(-t.sin(), k) * r * t.sin()
    });
    first + second
}

#[test]
fn rounds_wider_than_a_rings_top_meet_along_the_first_s_contact() {
    for r in [1.8, 2.0] {
        for outer_first in [true, false] {
            let name = format!("r = {r}, outer first: {outer_first}");
            let mut topo = Topology::new();
            let (solid, outer, inner) = ring(&mut topo);
            assert!(validate_solid(&topo, solid).unwrap().is_valid());
            let before = solid_volume(&topo, solid, 1e-3).unwrap();
            let picks = if outer_first {
                [outer, inner]
            } else {
                [inner, outer]
            };
            let result = fillet_v2(&mut topo, solid, &picks, r).unwrap();
            assert!(result.failed.is_empty(), "{name}: {:?}", result.failed);
            // The two walls, the floor and the two rounds: the top is gone.
            let faces = solid_faces(&topo, result.solid).unwrap();
            assert_eq!(faces.len(), 5, "{name}: faces");
            let report = validate_solid(&topo, result.solid).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result.solid, 0.02).unwrap();
            assert!(is_watertight(&mesh), "{name}: mesh open");
            // Coarsely too: each closed band reads as periodic.
            let (coarse, _) =
                tessellate_solid_grouped_with_tolerance(&topo, result.solid, 0.5, 0.5).unwrap();
            assert!(is_watertight(&coarse), "{name}: coarse mesh open");
            let lost = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
            let expected = removed(r, outer_first);
            assert!(
                (lost - expected).abs() <= 1e-4 * expected,
                "{name}: removed {lost}, the rounds take off {expected}"
            );
        }
    }
}

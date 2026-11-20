//! A round of an arc edge whose ends meet a curved (B-spline) end face.
//!
//! The part is a D whose straight side is a B-spline bowing in towards the
//! arc's axis: an arc of radius `R` about the `z` axis, from angle `-alpha`
//! to `alpha`, closed by a quadratic Bezier from one end to the other with
//! its middle at `x = M`, and extruded `H` up `z`. Its top arc edge is
//! rounded: a torus about the axis. At each end the round meets the
//! B-spline wall, curved and tilted to the end section.
//!
//! The round's end stood in the section plane and was closed by a runout
//! patch. Each contact, a circle about the axis, now runs on to where it
//! crosses the wall's corner edges (on the top, the B-spline; on the arc's
//! cylinder, the vertical line), and the band ends on its intersection with
//! the wall, which the wall is notched along.
//!
//! A file may store an edge a little off its face (one stood 8e-5 off its
//! plane on a 500 mm part, a fitted section 3e-5 off a cylinder): the
//! second test moves the top B-spline edge down by that much, the third
//! bows the vertical edge where the walls meet off the arc's cylinder, and
//! each contact must still end on its edge.
//!
//! The volume taken off is the round's profile (the corner square of side
//! `r` less its quarter disc, in `(rho, z)`) swept about the axis between
//! the two crossings of the wall at radius `rho`, at angles `+-phi(rho)`:
//! the profile's integral of `2 rho phi(rho)`.
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

const R: f64 = 10.0;
const H: f64 = 6.0;
const M: f64 = 3.0;
const ALPHA_DEG: f64 = 60.0;

/// The Bezier's three control points, from the arc's end at `alpha` back
/// to its start at `-alpha`, in `z = 0`.
fn bezier() -> [Point3; 3] {
    let alpha = ALPHA_DEG.to_radians();
    let c = R * alpha.cos();
    // The middle of a quadratic Bezier is a quarter of each end plus half
    // the control point.
    let k = 2.0 * M - c;
    [
        Point3::new(c, R * alpha.sin(), 0.0),
        Point3::new(k, 0.0, 0.0),
        Point3::new(c, -R * alpha.sin(), 0.0),
    ]
}

/// The D, its arc counter-clockwise about `+z`.
fn d_prism(topo: &mut Topology) -> SolidId {
    let [b0, b1, b2] = bezier();
    let ids = [b2, b0].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
    let arc = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), R).unwrap();
    let side = NurbsCurve::new(
        2,
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        vec![b0, b1, b2],
        vec![1.0, 1.0, 1.0],
    )
    .unwrap();
    let edges = [
        Edge::new(ids[0], ids[1], EdgeCurve::Circle(arc)),
        Edge::new(ids[1], ids[0], EdgeCurve::NurbsCurve(side)),
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

/// The prism's top edge of `kind` at height `H`.
fn top_edge(topo: &Topology, solid: SolidId, arc: bool) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let kind = if arc {
                matches!(edge.curve(), EdgeCurve::Circle(_))
            } else {
                matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
            };
            kind && [edge.start(), edge.end()]
                .iter()
                .all(|&v| (topo.vertex(v).unwrap().point().z() - H).abs() < 1e-9)
        })
        .expect("the top edge")
}

/// The wall's angle at radius `rho`: where the Bezier's first half (from
/// the arc's end at `alpha`, its radius falling to `M`) is `rho` from the
/// axis, by bisection.
fn phi(rho: f64) -> f64 {
    let [b0, b1, b2] = bezier();
    let at = |t: f64| {
        let s = 1.0 - t;
        let x = s * s * b0.x() + 2.0 * s * t * b1.x() + t * t * b2.x();
        let y = s * s * b0.y() + 2.0 * s * t * b1.y() + t * t * b2.y();
        (x.hypot(y), y.atan2(x))
    };
    let (mut lo, mut hi) = (0.0, 0.5);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if at(mid).0 > rho {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    at(0.5 * (lo + hi)).1
}

/// The volume the round takes off: the profile's integral of
/// `2 rho phi(rho)`, over `rho = R - r + r sin(theta)`, where the
/// profile's height at `rho` is `r (1 - cos(theta))`.
fn removed(r: f64) -> f64 {
    const STEPS: usize = 2000;
    let f = |theta: f64| {
        let rho = r.mul_add(theta.sin(), R - r);
        let height = r * (1.0 - theta.cos());
        2.0 * rho * phi(rho) * height * r * theta.cos()
    };
    #[allow(clippy::cast_precision_loss)]
    let h = FRAC_PI_2 / STEPS as f64;
    let mut sum = f(0.0) + f(FRAC_PI_2);
    for k in 1..STEPS {
        #[allow(clippy::cast_precision_loss)]
        let theta = h * k as f64;
        sum += if k % 2 == 1 { 4.0 } else { 2.0 } * f(theta);
    }
    sum * h / 3.0
}

/// The vertical edge where the arc's cylinder meets the B-spline wall at
/// the arc's end at `alpha`.
fn vertical_edge(topo: &Topology, solid: SolidId) -> EdgeId {
    let alpha = ALPHA_DEG.to_radians();
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            matches!(edge.curve(), EdgeCurve::Line)
                && (a.x() - b.x()).abs() < 1e-9
                && (a.y() - b.y()).abs() < 1e-9
                && (a.x() - R * alpha.cos()).abs() < 1e-9
                && (a.y() - R * alpha.sin()).abs() < 1e-9
        })
        .expect("the vertical edge")
}

/// Rounds the top arc of the D at `r`, the top B-spline edge first moved
/// down `drop` and the vertical edge at the arc's end bowed `bow` out from
/// the arc's cylinder at its middle, and checks the result.
fn round_and_check(r: f64, drop: f64, bow: f64) {
    let case = format!("r = {r}, top spline edge {drop} off its plane, vertical edge bowed {bow}");
    let mut topo = Topology::new();
    let solid = d_prism(&mut topo);
    if bow > 0.0 {
        // A quadratic through the edge's ends whose middle control point
        // stands `2 bow` out, so the curve's middle stands `bow` out.
        let vertical = vertical_edge(&topo, solid);
        let edge = topo.edge(vertical).unwrap();
        let [a, b] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
        let out = Vec3::new(a.x(), a.y(), 0.0) * (2.0 * bow / R);
        let middle = Point3::new(
            0.5 * (a.x() + b.x()),
            0.5 * (a.y() + b.y()),
            0.5 * (a.z() + b.z()),
        ) + out;
        let curve = NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![a, middle, b],
            vec![1.0, 1.0, 1.0],
        )
        .unwrap();
        topo.edge_mut(vertical)
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(curve));
    }
    if drop > 0.0 {
        let spline = top_edge(&topo, solid, false);
        let EdgeCurve::NurbsCurve(curve) = topo.edge(spline).unwrap().curve().clone() else {
            unreachable!("the top spline edge");
        };
        let moved = NurbsCurve::new(
            curve.degree(),
            curve.knots().to_vec(),
            curve
                .control_points()
                .iter()
                .map(|&p| p - Vec3::new(0.0, 0.0, drop))
                .collect(),
            curve.weights().to_vec(),
        )
        .unwrap();
        topo.edge_mut(spline)
            .unwrap()
            .set_curve(EdgeCurve::NurbsCurve(moved));
    }
    let edge = top_edge(&topo, solid, true);
    // Measured finer than the 1e-4 asked of the round: the wall's notch is
    // a curved trim, and the measure's slivers along its chords grow with
    // the mesh's size (at 1e-3 they came to 5.7e-4).
    let before = solid_volume(&topo, solid, 1e-4).unwrap();
    let faces_before = solid_faces(&topo, solid).unwrap().len();

    let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
    assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{case}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{case}: mesh not watertight");
    let faces = solid_faces(&topo, result.solid).unwrap();
    assert_eq!(
        faces.len(),
        faces_before + 1,
        "{case}: the round alone is added, no runout patch"
    );
    assert!(
        faces
            .iter()
            .any(|&face| matches!(topo.face(face).unwrap().surface(), FaceSurface::Torus(_))),
        "{case}: the round is a torus"
    );

    let lost = before - solid_volume(&topo, result.solid, 1e-4).unwrap();
    let want = removed(r);
    // A top edge dropped off its plane leaves the top face and the wall
    // disagreeing by up to `drop` along the spline; the trim moves each
    // piece's end to its vertex, off the curve, and so moves part of that
    // disagreement, at most a strip `drop` deep and `r` wide along the
    // spline's chord.
    let chord = 2.0 * R * ALPHA_DEG.to_radians().sin();
    assert!(
        (lost - want).abs() <= 1e-4 * want.max(1.0) + drop * chord * r,
        "{case}: lost {lost}, the round's sweep {want}"
    );
}

#[test]
fn a_torus_round_runs_on_to_a_spline_end_face() {
    for r in [1.0, 2.0] {
        round_and_check(r, 0.0, 0.0);
    }
}

#[test]
fn a_torus_round_ends_on_a_spline_edge_off_its_plane() {
    for r in [1.0, 2.0] {
        round_and_check(r, 8e-5, 0.0);
    }
}

#[test]
fn a_torus_round_ends_on_a_fitted_edge_off_its_wall() {
    for r in [1.0, 2.0] {
        round_and_check(r, 0.0, 3e-5);
    }
}

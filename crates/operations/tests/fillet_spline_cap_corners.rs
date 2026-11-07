//! A round ending on a flat cap whose corner edge there is a B-spline.
//!
//! The part is a prism: the profile `x = 0`, `y = -B`, `y = B` closed by
//! a lopsided bulge out to `x = X0 + 3` stored as a cubic B-spline (as a
//! file writes a spline profile), extruded `H` up `z`. Its wall over the
//! bulge is a B-spline face, and the vertical edges at `x = X0`, where a
//! flat side meets that wall, round with the walked blend.
//!
//! At each end the round's section lies in the cap, its contacts ending on
//! the cap's two corner edges: the flat side's line and the wall's
//! B-spline rim. The cap notch took lines only, so the cap was left whole
//! and the mesh open there. It now splits the B-spline rim where the
//! contact ends on it, as it splits a line.
//!
//! The removed volume is `H` times the corner's area between the side, the
//! spline and the round's circle, found from the spline itself: the
//! circle's centre is `r` from the side and `r` from the spline.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::fitting::interpolate;
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

const B: f64 = 5.0;
const X0: f64 = 10.0;
const H: f64 = 6.0;

/// The bulge's profile curve, from `(X0, -B)` to `(X0, B)`.
fn bulge() -> NurbsCurve {
    let points: Vec<Point3> = (0..=20)
        .map(|k| {
            let s = f64::from(k) / 10.0 - 1.0;
            let x = X0 + 3.0 * (1.0 - s * s) * (1.0 + 0.4 * s);
            Point3::new(x, B * s, 0.0)
        })
        .collect();
    interpolate(&points, 3).unwrap()
}

/// The prism, its bulge counter-clockwise about `+z`.
fn spline_prism(topo: &mut Topology) -> SolidId {
    let corners = [(X0, -B), (X0, B), (0.0, B), (0.0, -B)];
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let spline = bulge();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = if i == 0 {
                EdgeCurve::NurbsCurve(spline.clone())
            } else {
                EdgeCurve::Line
            };
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
            OrientedEdge::new(topo.add_edge(edge), true)
        })
        .collect();
    let wire = topo.add_wire(Wire::new(edges, true).unwrap());
    let face = topo.add_face(Face::new(
        wire,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        },
    ));
    extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), H).unwrap()
}

/// The area the round of radius `r` takes off the corner at `(X0, B)`.
fn corner_area(r: f64) -> f64 {
    const N: u32 = 20_000;
    let spline = bulge();
    let (t0, t1) = spline.domain();
    let at = |k: u32| spline.evaluate(t0 + (t1 - t0) * f64::from(k) / f64::from(N));
    let samples: Vec<Point3> = (0..=N).map(at).collect();
    let nearest = |x: f64| {
        samples
            .iter()
            .enumerate()
            .map(|(k, p)| (((p.x() - x).powi(2) + (p.y() - (B - r)).powi(2)).sqrt(), k))
            .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a })
    };
    // The round's centre `(x, B - r)`: as far out as stays `r` from the
    // spline (bisection, the distance falling as `x` grows).
    let (mut low, mut high) = (X0 - 2.0 * r, X0 + 3.0);
    for _ in 0..80 {
        let mid = 0.5 * (low + high);
        if nearest(mid).0 > r {
            low = mid;
        } else {
            high = mid;
        }
    }
    let fx = 0.5 * (low + high);
    let contact = nearest(fx).1;
    // Green's theorem round the region: the side from the round's contact to
    // the corner, the spline back from the corner to its contact, the round's
    // arc back to the side.
    let mut loop_points = vec![Point3::new(fx, B, 0.0)];
    loop_points.extend(samples[contact..].iter().rev().copied());
    let c = samples[contact];
    let a0 = (c.y() - (B - r)).atan2(c.x() - fx);
    let a1 = std::f64::consts::FRAC_PI_2;
    for k in 1..N {
        let a = a0 + (a1 - a0) * f64::from(k) / f64::from(N);
        loop_points.push(Point3::new(fx + r * a.cos(), B - r + r * a.sin(), 0.0));
    }
    let mut twice = 0.0;
    for (k, p) in loop_points.iter().enumerate() {
        let q = loop_points[(k + 1) % loop_points.len()];
        twice += p.x() * q.y() - q.x() * p.y();
    }
    (0.5 * twice).abs()
}

fn arc_corner_edge(topo: &Topology, solid: SolidId) -> EdgeId {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .find(|&e| {
            let edge = topo.edge(e).unwrap();
            let [p, q] = [edge.start(), edge.end()].map(|v| topo.vertex(v).unwrap().point());
            [p, q]
                .iter()
                .all(|p| (p.x() - X0).abs() < 1e-9 && (p.y() - B).abs() < 1e-9)
                && (p.z() - q.z()).abs() > 1.0
        })
        .expect("the vertical edge at (X0, B)")
}

#[test]
fn a_walked_round_ends_on_a_caps_spline_rim() {
    for r in [0.5, 1.0] {
        let mut topo = Topology::new();
        let solid = spline_prism(&mut topo);
        let faces = solid_faces(&topo, solid).unwrap();
        assert!(
            faces
                .iter()
                .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_))),
            "the wall over the bulge is a B-spline face"
        );
        let before = solid_volume(&topo, solid, 1e-3).unwrap();
        let edge = arc_corner_edge(&topo, solid);
        let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
        assert!(result.failed.is_empty(), "r {r}: {:?}", result.failed);
        let report = validate_solid(&topo, result.solid).unwrap();
        assert!(report.is_valid(), "r {r}: {:?}", report.issues);
        let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "r {r}: mesh open at the caps");
        assert_eq!(
            solid_faces(&topo, result.solid).unwrap().len(),
            faces.len() + 1,
            "r {r}: the round alone is new"
        );
        let removed = before - solid_volume(&topo, result.solid, 1e-3).unwrap();
        let expected = H * corner_area(r);
        assert!(
            (removed - expected).abs() <= 1e-5 * expected,
            "r {r}: removed {removed}, the corner's closed form {expected}"
        );
    }
}

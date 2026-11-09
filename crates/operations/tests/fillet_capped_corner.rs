//! Two rounds meeting at an extrusion's end beside a sharp edge.
//!
//! An extruded profile's wall `C` meets its end plane `P` and a flat side
//! `Q` at a corner. The edge where `P` meets `C` is rounded, and so is the
//! extrusion's edge where `C` meets `Q`; the edge where `P` meets `Q` is
//! left sharp, and the wall meets `Q` at more than a right angle.
//!
//! At depth `t` below `P` the round along `P` is the wall's trace inset by
//! `w = r - sqrt(r^2 - (r - t)^2)`, and the extrusion's round is one
//! circle. So the part's section there is the profile with its wall inset
//! by `w` and the corner cut by that circle. The two rounds meet along a
//! crease from where their contacts on the wall cross round the
//! extrusion round's quarter to `Q`, and from there the round along `P`
//! runs on to `Q`, notching it, up to the sharp edge. Neither round ends
//! on its own section at the corner, and no patch closes it. The crease
//! and the notch are ellipses, so every edge lies on the faces it bounds.
//!
//! The volumes are integrated over those sections, each in closed form.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

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

/// The prisms' height.
const H: f64 = 12.0;

/// A prism of `profile` (counter-clockwise about `+z`), its edge `i` from
/// corner `i` to the next.
fn prism(topo: &mut Topology, corners: &[(f64, f64)]) -> SolidId {
    prism_along(topo, corners, Vec3::new(0.0, 0.0, 1.0))
}

/// The prism of `profile` swept by `H` times `direction`.
fn prism_along(topo: &mut Topology, corners: &[(f64, f64)], direction: Vec3) -> SolidId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
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
    extrude(topo, face, direction, H).unwrap()
}

/// The edges of `solid` whose two ends `keep` takes.
fn edges_where(
    topo: &Topology,
    solid: SolidId,
    keep: impl Fn(Point3, Point3) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            keep(
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            )
        })
        .collect()
}

/// Round `edges` of `solid` at `r` and check the result: valid, watertight,
/// `faces` faces with none a patch, `volume`, and no edge further than
/// `slip` off a face it bounds.
fn check(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    (r, faces, volume): (f64, usize, f64),
    slip: f64,
) {
    let result = fillet_v2(topo, solid, edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
    let report = validate_solid(topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(topo, result.solid, 0.02).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: the mesh is open");
    let result_faces = solid_faces(topo, result.solid).unwrap();
    let patches = result_faces
        .iter()
        .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
        .count();
    assert_eq!(patches, 0, "r = {r}: a corner was closed by a patch");
    assert_eq!(result_faces.len(), faces, "r = {r}: face count");
    let got = solid_volume(topo, result.solid, 0.005).unwrap();
    assert!(
        (got - volume).abs() <= 1e-7 * volume,
        "r = {r}: volume {got} against {volume}"
    );
    let off = edges_off_their_faces(topo, result.solid);
    assert!(
        off <= slip,
        "r = {r}: an edge strays {off} off a face it bounds"
    );
}

/// The furthest any edge of `solid` strays from a face it bounds.
fn edges_off_their_faces(topo: &Topology, solid: SolidId) -> f64 {
    let mut worst = 0.0_f64;
    for face_id in solid_faces(topo, solid).unwrap() {
        let face = topo.face(face_id).unwrap();
        let surface = face.surface();
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            for oriented in topo.wire(wire_id).unwrap().edges() {
                let edge = topo.edge(oriented.edge()).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                for k in 0..=64 {
                    let t = t0 + (t1 - t0) * f64::from(k) / 64.0;
                    let p = edge.curve().evaluate_with_endpoints(t, a, b);
                    let off = if let FaceSurface::Plane { normal, d } = surface {
                        (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs() / normal.length()
                    } else {
                        let (u, v) = surface.project_point(p).unwrap();
                        (surface.evaluate(u, v).unwrap() - p).length()
                    };
                    worst = worst.max(off);
                }
            }
        }
    }
    worst
}

/// `∫ sqrt(a^2 - u^2) du` from 0 to `u`.
fn quarter(a: f64, u: f64) -> f64 {
    let u = u.clamp(-a, a);
    0.5 * (u * (a * a - u * u).sqrt() + a * a * (u / a).asin())
}

/// `∫_0^r f(r - sqrt(r^2 - (r - t)^2)) dt`, over the round's quarter.
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

/// The trapezoid `(0, 0)`, `(L, 0)`, `(L + s, T)`, `(0, T)`. Rounding the
/// top edge along `y = 0` and the vertical edge at `(L, 0)` leaves the
/// top's slanted edge, where the top meets the slanted side at more than
/// a right angle, sharp; the round along the top ends square at `x = 0`.
mod trapezoid {
    use super::*;

    pub const L: f64 = 20.0;
    pub const S: f64 = 6.0;
    pub const T: f64 = 10.0;

    /// The run `s` that makes the wall `y = 0` meet the slanted side at
    /// `alpha` degrees.
    pub fn run_for(alpha: f64) -> f64 {
        T / (180.0 - alpha).to_radians().tan()
    }

    pub fn corners(s: f64) -> [(f64, f64); 4] {
        [(0.0, 0.0), (L, 0.0), (L + s, T), (0.0, T)]
    }

    pub fn solid(topo: &mut Topology, s: f64) -> SolidId {
        prism(topo, &corners(s))
    }

    /// The edge along the top at `y = 0` and the vertical edge at `(L, 0)`,
    /// within `tol` of where they are.
    pub fn rounded(topo: &Topology, solid: SolidId, tol: f64) -> Vec<EdgeId> {
        edges_where(topo, solid, |p, q| {
            let top_side = (p.z() - H).abs() < tol
                && (q.z() - H).abs() < tol
                && p.y().abs() < tol
                && q.y().abs() < tol;
            let vertical = (p.x() - L).abs() < tol
                && (q.x() - L).abs() < tol
                && p.y().abs() < tol
                && q.y().abs() < tol;
            top_side || vertical
        })
    }

    /// The section's area with the side `y = 0` inset by `w` and the
    /// corner at `(L, 0)` cut by the vertical round's circle.
    pub fn area(s: f64, r: f64, w: f64) -> f64 {
        let slant = s.hypot(T);
        // The vertical round's centre, `r` from both sides, and the height
        // where it touches the slanted side.
        let kx = L - r * (slant - s) / T;
        let yq = r - r * s / slant;
        // `∫ x_right dy`, the slanted side's `L + s y / T` above `yq`, the
        // circle's `kx + sqrt(r^2 - (y - r)^2)` below.
        let line = |y: f64| L * y + s * y * y / (2.0 * T);
        let round = |y: f64| kx * y + quarter(r, y - r);
        if w >= yq {
            line(T) - line(w)
        } else {
            line(T) - line(yq) + round(yq) - round(w)
        }
    }

    pub fn volume(s: f64, r: f64) -> f64 {
        area(s, r, 0.0) * (H - r) + over_the_round(r, |w| area(s, r, w))
    }
}

#[test]
fn a_trapezoid_s_top_edge_and_vertical_meet_at_the_sharp_slant() {
    for r in [1.0, 2.0, 3.0] {
        let mut topo = Topology::new();
        let solid = trapezoid::solid(&mut topo, trapezoid::S);
        let edges = trapezoid::rounded(&topo, solid, 1e-9);
        assert_eq!(edges.len(), 2);
        // The top, the bottom, the four sides and the two rounds.
        let volume = trapezoid::volume(trapezoid::S, r);
        check(&mut topo, solid, &edges, (r, 8, volume), 1e-7);
    }
}

/// The wall meeting the slanted side at 150 and 165 deg: the cap round's
/// notch on the slanted side starts nearly where the round meets the wall,
/// where its depth below the top steepens without bound, and still runs on
/// the round.
#[test]
fn a_trapezoid_s_wider_slant_keeps_the_notch_on_the_round() {
    for alpha in [150.0, 165.0] {
        let s = trapezoid::run_for(alpha);
        for r in [1.0, 3.0] {
            let mut topo = Topology::new();
            let solid = trapezoid::solid(&mut topo, s);
            let edges = trapezoid::rounded(&topo, solid, 1e-9);
            assert_eq!(edges.len(), 2);
            check(
                &mut topo,
                solid,
                &edges,
                (r, 8, trapezoid::volume(s, r)),
                1e-7,
            );
        }
    }
}

/// The trapezoid swept a hair off the top's normal, so its slanted side
/// stands off square to the top by a cosine of `cos`, as a part read from
/// a file can: inside what the sharp mitre takes as square, the corner is
/// built as on the square part, its points moved by no more than the
/// mitre's `1e-6`.
#[test]
fn a_trapezoid_s_slant_a_hair_off_square_meets_the_same() {
    let (s, r) = (trapezoid::S, 2.0);
    for cos in [5e-9, 5e-8] {
        let lean = cos * s.hypot(trapezoid::T) / trapezoid::T;
        let mut topo = Topology::new();
        let solid = prism_along(&mut topo, &trapezoid::corners(s), Vec3::new(lean, 0.0, 1.0));
        let edges = trapezoid::rounded(&topo, solid, 1e-5);
        assert_eq!(edges.len(), 2, "cos = {cos}");
        check(
            &mut topo,
            solid,
            &edges,
            (r, 8, trapezoid::volume(s, r)),
            1e-6,
        );
    }
}

//! Two rounds meeting at a D prism's end beside its sharp chord.
//!
//! The D's major arc is its wall `C`, its chord the flat side `Q`. The top
//! arc and both vertical edges at the chord are rounded, the top chord left
//! sharp: the top's round is a torus, and at depth `t` below the top it is
//! the wall's trace inset by `w = r - sqrt(r^2 - (r - t)^2)`, the vertical
//! rounds one circle each. See `fillet_capped_corner.rs` for the straight
//! wall; this is the same corner on an arc.
//!
//! The volumes are integrated over those sections, each in closed form.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::FRAC_PI_2;

use brepkit_math::curves::Circle3D;
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
/// corner `i` to the next, a circle about `+z` where `arcs` gives one.
fn prism(topo: &mut Topology, corners: &[(f64, f64)], arcs: &[Option<Circle3D>]) -> SolidId {
    let ids: Vec<_> = corners
        .iter()
        .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
        .collect();
    let edges: Vec<_> = (0..ids.len())
        .map(|i| {
            let curve = arcs[i].clone().map_or(EdgeCurve::Line, EdgeCurve::Circle);
            let edge = Edge::new(ids[i], ids[(i + 1) % ids.len()], curve);
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

/// The edges of `solid` whose two ends, and whether it is straight, `keep`
/// takes.
fn edges_where(
    topo: &Topology,
    solid: SolidId,
    keep: impl Fn(Point3, Point3, bool) -> bool,
) -> Vec<EdgeId> {
    solid_edges(topo, solid)
        .unwrap()
        .into_iter()
        .filter(|&edge_id| {
            let edge = topo.edge(edge_id).unwrap();
            keep(
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
                matches!(edge.curve(), EdgeCurve::Line),
            )
        })
        .collect()
}

/// Round `edges` of `solid` at `r` and check the result: valid, watertight,
/// `faces` faces with none a patch, and `volume`.
fn check(topo: &mut Topology, solid: SolidId, edges: &[EdgeId], r: f64, faces: usize, volume: f64) {
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

/// The D: a circle of radius `RC` about the origin, cut by the chord
/// `x = -A`, its major arc the wall. Rounding the top arc and both
/// vertical edges at the chord leaves the top chord, where the top meets
/// the flat side at more than a right angle, sharp.
mod d {
    use super::*;

    pub const RC: f64 = 10.0;
    pub const A: f64 = 4.0;

    pub fn solid(topo: &mut Topology) -> SolidId {
        let b = (RC * RC - A * A).sqrt();
        let arc = Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), RC).unwrap();
        prism(topo, &[(-A, -b), (-A, b)], &[Some(arc), None])
    }

    /// The section's area with the wall inset by `w` and both corners cut
    /// by the vertical rounds' circles.
    pub fn area(r: f64, w: f64) -> f64 {
        let rho = RC - w;
        let (kx, ky) = (-A + r, ((RC - r).powi(2) - (r - A).powi(2)).sqrt());
        // Where the vertical round's circle touches the wall.
        let xt = kx * RC / (RC - r);
        let circle = |x: f64| (rho * rho - x * x).max(0.0).sqrt();
        let corner = |x: f64| ky + (r * r - (x - kx).powi(2)).max(0.0).sqrt();
        // The section's upper half: under the inset arc, and under the
        // corner's circle up to where it touches the wall.
        let mut cuts = vec![-A];
        let samples = 4000;
        let h = |x: f64| circle(x) - corner(x);
        for i in 0..samples {
            let (x0, x1) = (
                -A + (xt + A) * f64::from(i) / f64::from(samples),
                -A + (xt + A) * f64::from(i + 1) / f64::from(samples),
            );
            if h(x0) * h(x1) < 0.0 {
                let (mut lo, mut hi) = (x0, x1);
                for _ in 0..100 {
                    let mid = 0.5 * (lo + hi);
                    if h(lo) * h(mid) <= 0.0 {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                cuts.push(0.5 * (lo + hi));
            }
        }
        cuts.push(xt);
        let mut half = 0.0;
        for pair in cuts.windows(2) {
            let (x0, x1) = (pair[0], pair[1]);
            let mid = 0.5 * (x0 + x1);
            half += if circle(mid) <= corner(mid) {
                quarter(rho, x1) - quarter(rho, x0)
            } else {
                ky * (x1 - x0) + quarter(r, x1 - kx) - quarter(r, x0 - kx)
            };
        }
        half += quarter(rho, rho) - quarter(rho, xt);
        2.0 * half
    }

    pub fn volume(r: f64) -> f64 {
        area(r, 0.0) * (H - r) + over_the_round(r, |w| area(r, w))
    }
}

#[test]
fn a_d_s_top_arc_and_its_verticals_meet_at_the_sharp_chord() {
    let b = (d::RC * d::RC - d::A * d::A).sqrt();
    for r in [1.0, 2.0, 3.0] {
        let mut topo = Topology::new();
        let solid = d::solid(&mut topo);
        let edges = edges_where(&topo, solid, |p, q, straight| {
            let top_arc = (p.z() - H).abs() < 1e-9 && (q.z() - H).abs() < 1e-9 && !straight;
            let vertical = (p.x() + d::A).abs() < 1e-9
                && (q.x() + d::A).abs() < 1e-9
                && (p.y().abs() - b).abs() < 1e-9
                && (p.z() - q.z()).abs() > 1e-6;
            top_arc || vertical
        });
        assert_eq!(edges.len(), 3);
        // The top, the bottom, the flat side, the wall, the torus and the
        // two vertical rounds.
        check(&mut topo, solid, &edges, r, 7, d::volume(r));
    }
}

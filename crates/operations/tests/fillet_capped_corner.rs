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
//! runs on to `Q`, notching it, up to the sharp edge. Both rounds used to
//! end on their own sections, with a patch closing the gap between them,
//! the wall left a loop crossing itself and the mesh open.
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
    extrude(topo, face, normal, H).unwrap()
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

/// The trapezoid `(0, 0)`, `(L, 0)`, `(L + S, T)`, `(0, T)`. Rounding the
/// top edge along `y = 0` and the vertical edge at `(L, 0)` leaves the
/// top's slanted edge, where the top meets the slanted side at more than
/// a right angle, sharp; the round along the top ends square at `x = 0`.
mod trapezoid {
    use super::*;

    pub const L: f64 = 20.0;
    pub const S: f64 = 6.0;
    pub const T: f64 = 10.0;

    pub fn solid(topo: &mut Topology) -> SolidId {
        prism(topo, &[(0.0, 0.0), (L, 0.0), (L + S, T), (0.0, T)])
    }

    /// The section's area with the side `y = 0` inset by `w` and the
    /// corner at `(L, 0)` cut by the vertical round's circle.
    pub fn area(r: f64, w: f64) -> f64 {
        let slant = S.hypot(T);
        // The vertical round's centre, `r` from both sides, and the height
        // where it touches the slanted side.
        let kx = L - r * (slant - S) / T;
        let yq = r - r * S / slant;
        // `∫ x_right dy`, the slanted side's `L + S y / T` above `yq`, the
        // circle's `kx + sqrt(r^2 - (y - r)^2)` below.
        let line = |y: f64| L * y + S * y * y / (2.0 * T);
        let round = |y: f64| kx * y + quarter(r, y - r);
        if w >= yq {
            line(T) - line(w)
        } else {
            line(T) - line(yq) + round(yq) - round(w)
        }
    }

    pub fn volume(r: f64) -> f64 {
        area(r, 0.0) * (H - r) + over_the_round(r, |w| area(r, w))
    }
}

#[test]
fn a_trapezoid_s_top_edge_and_vertical_meet_at_the_sharp_slant() {
    for r in [1.0, 2.0, 3.0] {
        let mut topo = Topology::new();
        let solid = trapezoid::solid(&mut topo);
        let edges = edges_where(&topo, solid, |p, q| {
            let top_side = (p.z() - H).abs() < 1e-9
                && (q.z() - H).abs() < 1e-9
                && p.y().abs() < 1e-9
                && q.y().abs() < 1e-9;
            let vertical = (p.x() - trapezoid::L).abs() < 1e-9
                && (q.x() - trapezoid::L).abs() < 1e-9
                && p.y().abs() < 1e-9
                && q.y().abs() < 1e-9;
            top_side || vertical
        });
        assert_eq!(edges.len(), 2);
        // The top, the bottom, the four sides and the two rounds.
        check(&mut topo, solid, &edges, r, 8, trapezoid::volume(r));
    }
}

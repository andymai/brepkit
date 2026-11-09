//! Two rounds along a face's outline meeting where the outline turns in, a
//! reflex corner, the edge there between the walls (concave) left sharp.
//!
//! An arch: a block with a channel through it, rounded along the channel's
//! outline on the top face. At each corner of the channel's ceiling the
//! ball rolling along one edge pivots on the sharp concave edge and rolls
//! on along the other, so the rounds join through a horn torus about that
//! edge, which pinches to a point on it a radius below the top. The
//! concave edge must end there: the walls' trims split it where their
//! contacts end on it, so no wall and no torus runs along it to the corner
//! and back.
//!
//! Each straight round takes off its cross-section, `r^2 (1 - pi / 4)`,
//! along its edge. Past both edges at a corner the ball pivots: a point
//! at distance `rho < r` from the concave edge, at depth `a` below the
//! top, is cut away where `a > sqrt(r^2 - (r - rho)^2)`, so each corner
//! takes off `(pi / 2) (5 / 6 - pi / 4) r^3`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::f64::consts::PI;

use brepkit_blend::radius_law::RadiusLaw;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::blend_ops::{fillet_v2, fillet_v2_variable};
use brepkit_operations::extrude::extrude;
use brepkit_operations::measure::solid_volume;
use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
use brepkit_operations::validate::validate_solid;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_edges;
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

/// The block's length, width and height.
const L: f64 = 20.0;
const W: f64 = 12.0;
const H: f64 = 6.0;
/// The channel runs from `x = A` to `x = B` and `C` in from `y = 0`.
const A: f64 = 6.0;
const B: f64 = 14.0;
const C: f64 = 7.0;

/// The arch's outline, counter-clockwise.
const OUTLINE: [(f64, f64); 8] = [
    (0.0, 0.0),
    (A, 0.0),
    (A, C),
    (B, C),
    (B, 0.0),
    (L, 0.0),
    (L, W),
    (0.0, W),
];

/// The arch's outline in `z = 0`, counter-clockwise, extruded `H` up `z`.
fn arch(topo: &mut Topology) -> SolidId {
    let ids: Vec<_> = OUTLINE
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

/// Round the top face's edges between the outline's points `chain` (by
/// index into [`OUTLINE`], in order, closed when `closed`) at `r`, and
/// check the result against `removed`.
fn round_top_outline(chain: &[usize], closed: bool, heights: &[f64], r: f64, removed: f64) {
    let mut topo = Topology::new();
    let solid = arch(&mut topo);
    let pairs = chain.len() - usize::from(!closed);
    let mut edges: Vec<EdgeId> = Vec::new();
    for &z in heights {
        let at = |i: usize| Point3::new(OUTLINE[i].0, OUTLINE[i].1, z);
        edges.extend(
            (0..pairs).map(|k| {
                edge_between(&topo, solid, at(chain[k]), at(chain[(k + 1) % chain.len()]))
            }),
        );
    }
    let result = fillet_v2(&mut topo, solid, &edges, r).unwrap();
    assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);

    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "r = {r}: the mesh is open");

    let expected = H * (L * W - (B - A) * C) - removed;
    let volume = solid_volume(&topo, result.solid, 0.01).unwrap();
    assert!(
        (volume - expected).abs() < 1e-6 * expected,
        "r = {r}: volume {volume}, expected {expected}"
    );
}

/// The straight rounds' cross-section.
fn section(r: f64) -> f64 {
    r * r * (1.0 - PI / 4.0)
}

/// What a reflex corner's pivot takes off past both its edges.
fn pivot(r: f64) -> f64 {
    PI / 2.0 * (5.0 / 6.0 - PI / 4.0) * r.powi(3)
}

#[test]
fn rounds_meeting_at_a_reflex_corner_pivot_on_its_concave_edge() {
    for r in [1.0_f64, 2.0] {
        // The channel's three edges, their far ends square on the arch's
        // front face.
        round_top_outline(
            &[1, 2, 3, 4],
            false,
            &[H],
            r,
            section(r) * (2.0 * C + (B - A)) + 2.0 * pivot(r),
        );
    }
}

#[test]
fn a_whole_outline_rounded_pivots_at_its_reflex_corners() {
    // Every top edge: the six convex corners are mitred, each keeping
    // `2 r^3 (5 / 6 - pi / 4)` of its two rounds (see
    // `fillet_extruded_mitre.rs`), and the two reflex corners pivot.
    for r in [1.0_f64, 2.0] {
        let perimeter = 2.0 * (L + W + C);
        let mitres = 6.0 * 2.0 * (5.0 / 6.0 - PI / 4.0) * r.powi(3);
        round_top_outline(
            &[0, 1, 2, 3, 4, 5, 6, 7],
            true,
            &[H],
            r,
            section(r) * perimeter - mitres + 2.0 * pivot(r),
        );
    }
}

#[test]
fn both_ends_of_a_concave_edge_pivot() {
    // The whole outline at the top and at the bottom, so the concave
    // edges end at a pivot at both ends.
    for r in [1.0_f64, 2.0] {
        let perimeter = 2.0 * (L + W + C);
        let mitres = 6.0 * 2.0 * (5.0 / 6.0 - PI / 4.0) * r.powi(3);
        let one = section(r) * perimeter - mitres + 2.0 * pivot(r);
        round_top_outline(&[0, 1, 2, 3, 4, 5, 6, 7], true, &[0.0, H], r, 2.0 * one);
    }
}

/// Rounds the arch's whole outlines at the top and the bottom, the top of
/// the channel's ceiling at `ceiling` and every other edge at `r`, and
/// returns the volume taken off, or `None` where the fillet is refused.
fn removed_with_ceiling(r: f64, ceiling: f64) -> Option<f64> {
    let mut topo = Topology::new();
    let solid = arch(&mut topo);
    let mut laws = Vec::new();
    for (z, top) in [(0.0, false), (H, true)] {
        let at = |i: usize| Point3::new(OUTLINE[i].0, OUTLINE[i].1, z);
        for k in 0..OUTLINE.len() {
            let edge = edge_between(&topo, solid, at(k), at((k + 1) % OUTLINE.len()));
            let radius = if top && k == 2 { ceiling } else { r };
            laws.push((edge, RadiusLaw::Constant(radius)));
        }
    }
    let result = fillet_v2_variable(&mut topo, solid, laws).ok()?;
    if !result.failed.is_empty() {
        return None;
    }
    let report = validate_solid(&topo, result.solid).unwrap();
    assert!(report.is_valid(), "{r}/{ceiling}: {:?}", report.issues);
    let mesh = tessellate_solid(&topo, result.solid, 0.01).unwrap();
    assert!(is_watertight(&mesh), "{r}/{ceiling}: the mesh is open");
    Some(H * (L * W - (B - A) * C) - solid_volume(&topo, result.solid, 0.01).unwrap())
}

#[test]
fn a_reflex_corner_of_unequal_rounds_is_right_or_refused() {
    // The ceiling's round at 2 meets the walls' at 1 at both reflex corners.
    // The corners take off more than the even rounds' and less than that
    // with the ceiling's wider section along its length and two wider pivots.
    let Some(mixed) = removed_with_ceiling(1.0, 2.0) else {
        return;
    };
    let even = removed_with_ceiling(1.0, 1.0).unwrap();
    let bound = even + (section(2.0) - section(1.0)) * (B - A) + 2.0 * pivot(2.0);
    assert!(
        even < mixed && mixed < bound,
        "removed {mixed}, between {even} and {bound}"
    );
}

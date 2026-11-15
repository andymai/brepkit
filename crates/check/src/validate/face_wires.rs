//! Whether a face's wires keep clear of each other: no hole crosses the
//! outline or another hole, and on a plane no hole lies outside the outline
//! or inside another hole.
//!
//! The edges' own curves are read. Where two wires meet, each is followed a
//! short way to both sides of the meeting point: the wires cross when one
//! leaves the point on the other side of the other wire than it came in on.
//! Wires that meet at a vertex, or touch, come and leave on one side.

use std::f64::consts::TAU;

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::{PlanarRegion, wire_region_2d};
use brepkit_topology::wire::WireId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;

/// The chords a curved edge is read in when meeting points are looked for.
const CHORDS: usize = 16;

/// The angle within which a direction is taken to lie along a wire, on
/// neither side of it.
const ALONG: f64 = 1e-6;

/// One edge of a wire, in the wire's direction.
struct Arc<'a> {
    edge: EdgeId,
    curve: &'a EdgeCurve,
    ends: (Point3, Point3),
    span: (f64, f64),
    forward: bool,
    /// The edge's points at even fractions of its span.
    points: Vec<Point3>,
    /// A bound on how far the edge stands off its chords.
    sag: f64,
    length: f64,
    lo: [f64; 3],
    hi: [f64; 3],
}

impl<'a> Arc<'a> {
    fn of(topo: &'a Topology, edge_id: EdgeId, forward: bool) -> Result<Self, CheckError> {
        let edge = topo.edge(edge_id)?;
        let ends = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let curve = edge.curve();
        let mut arc = Self {
            edge: edge_id,
            curve,
            ends,
            span: curve.domain_with_endpoints(ends.0, ends.1),
            forward,
            points: Vec::new(),
            sag: 0.0,
            length: 0.0,
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        };
        let chords = if matches!(curve, EdgeCurve::Line) {
            1
        } else {
            CHORDS
        };
        arc.points = (0..=chords).map(|k| arc.at(fraction(k, chords))).collect();
        for k in 0..chords {
            let (a, b) = (arc.points[k], arc.points[k + 1]);
            arc.length += (b - a).length();
            if chords > 1 {
                let on = arc.at((fraction(k, chords) + fraction(k + 1, chords)) * 0.5);
                arc.sag = arc.sag.max(2.0 * (on - (a + (b - a) * 0.5)).length());
            }
        }
        for p in &arc.points {
            for (k, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                arc.lo[k] = arc.lo[k].min(c);
                arc.hi[k] = arc.hi[k].max(c);
            }
        }
        Ok(arc)
    }

    /// The edge's point at the fraction `f` of its way along the wire.
    fn at(&self, f: f64) -> Point3 {
        let s = if self.forward { f } else { 1.0 - f };
        self.curve.evaluate_with_endpoints(
            self.span.0 + (self.span.1 - self.span.0) * s,
            self.ends.0,
            self.ends.1,
        )
    }

    fn chords(&self) -> usize {
        self.points.len() - 1
    }

    /// Whether the two edges' boxes lie farther apart than `reach`.
    fn clear_of(&self, other: &Self, reach: f64) -> bool {
        (0..3).any(|k| self.lo[k] > other.hi[k] + reach || other.lo[k] > self.hi[k] + reach)
    }
}

#[allow(clippy::cast_precision_loss)]
fn fraction(k: usize, of: usize) -> f64 {
    k as f64 / of as f64
}

/// The fraction of the way from `a` to `b` at which `p` lies.
fn along(a: Point3, b: Point3, p: Point3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

/// The nearest the two edges come to each other from the fractions `fa` and
/// `fb`: the two fractions and the gap there.
fn nearest(a: &Arc<'_>, b: &Arc<'_>, mut fa: f64, mut fb: f64) -> (f64, f64, f64) {
    const STEP: f64 = 1e-6;
    let slope = |arc: &Arc<'_>, f: f64| {
        let (f0, f1) = ((f - STEP).max(0.0), (f + STEP).min(1.0));
        (arc.at(f1) - arc.at(f0)) * (1.0 / (f1 - f0))
    };
    let gap = |fa: f64, fb: f64| (a.at(fa) - b.at(fb)).length();
    let mut best = (fa, fb, gap(fa, fb));
    for _ in 0..40 {
        let r = a.at(fa) - b.at(fb);
        let (da, db) = (slope(a, fa), slope(b, fb));
        let (g11, g12, g22) = (da.dot(da), -da.dot(db), db.dot(db));
        let (r1, r2) = (-r.dot(da), r.dot(db));
        let det = g11 * g22 - g12 * g12;
        let (x, y) = if det > 1e-12 * g11 * g22 {
            ((r1 * g22 - r2 * g12) / det, (r2 * g11 - r1 * g12) / det)
        } else if g11 > 0.0 {
            (r1 / g11, 0.0)
        } else {
            break;
        };
        let (na, nb) = ((fa + x).clamp(0.0, 1.0), (fb + y).clamp(0.0, 1.0));
        let moved = (na - fa).abs() + (nb - fb).abs();
        (fa, fb) = (na, nb);
        let now = gap(fa, fb);
        if now < best.2 {
            best = (fa, fb, now);
        }
        if moved < 1e-14 {
            break;
        }
    }
    best
}

/// The fractions at which the two edges come within `tol` of each other.
fn meetings(a: &Arc<'_>, b: &Arc<'_>, tol: f64) -> Vec<(f64, f64)> {
    let reach = a.sag + b.sag + tol;
    let mut found: Vec<(f64, f64)> = Vec::new();
    if a.clear_of(b, reach) {
        return found;
    }
    for i in 0..a.chords() {
        let (a0, a1) = (a.points[i], a.points[i + 1]);
        for j in 0..b.chords() {
            let (b0, b1) = (b.points[j], b.points[j + 1]);
            let (gap, pa, pb) = crate::distance::edge::segment_segment_distance(a0, a1, b0, b1);
            if gap > reach {
                continue;
            }
            let fa = fraction(i, a.chords()) + along(a0, a1, pa) / fraction(a.chords(), 1);
            let fb = fraction(j, b.chords()) + along(b0, b1, pb) / fraction(b.chords(), 1);
            let (fa, fb, gap) = nearest(a, b, fa, fb);
            let known = found
                .iter()
                .any(|&(ga, gb)| (ga - fa).abs() < 1e-6 && (gb - fb).abs() < 1e-6);
            if gap < tol && !known {
                found.push((fa, fb));
            }
        }
    }
    found
}

/// The wire's point the distance `d` from `p`, the first met when the wire
/// is followed from the fraction `f` of its edge `index`, in its direction
/// or against it.
fn arm(wire: &[Arc<'_>], index: usize, f: f64, onward: bool, p: Point3, d: f64) -> Option<Point3> {
    let (mut i, mut from) = (index, f);
    for _ in 0..=wire.len() {
        let arc = &wire[i];
        let n = arc.chords();
        let steps: Vec<f64> = if onward {
            (0..=n)
                .map(|k| fraction(k, n))
                .filter(|&g| g > from)
                .collect()
        } else {
            (0..=n)
                .rev()
                .map(|k| fraction(k, n))
                .filter(|&g| g < from)
                .collect()
        };
        let mut near = from;
        for far in steps {
            if (arc.at(far) - p).length() >= d {
                let (mut lo, mut hi) = (near, far);
                for _ in 0..50 {
                    let mid = 0.5 * (lo + hi);
                    if (arc.at(mid) - p).length() < d {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                return Some(arc.at(0.5 * (lo + hi)));
            }
            near = far;
        }
        if onward {
            i = (i + 1) % wire.len();
            from = 0.0;
        } else {
            i = (i + wire.len() - 1) % wire.len();
            from = 1.0;
        }
    }
    None
}

/// Two unit directions square to `n` and to each other.
fn frame(n: Vec3) -> Option<(Vec3, Vec3)> {
    let axis = if n.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let x = n.cross(axis).normalize().ok()?;
    let y = n.cross(x).normalize().ok()?;
    Some((x, y))
}

/// Whether the direction at the angle `at` lies to the left of a path that
/// comes in along the direction at `back` (read from the path's point) and
/// leaves at `out`. `None` when it lies along the path.
fn left_of(out: f64, back: f64, at: f64) -> Option<bool> {
    let wedge = (back - out).rem_euclid(TAU);
    let turn = (at - out).rem_euclid(TAU);
    if !(ALONG..=TAU - ALONG).contains(&wedge) {
        None
    } else if turn > ALONG && turn < wedge - ALONG {
        Some(true)
    } else if turn > wedge + ALONG && turn < TAU - ALONG {
        Some(false)
    } else {
        None
    }
}

/// Whether wire `a` crosses wire `b` where its edge `ia` meets `b`'s edge
/// `ib` at the fractions `fa` and `fb`.
fn crosses(
    surface: &FaceSurface,
    a: &[Arc<'_>],
    (ia, fa): (usize, f64),
    b: &[Arc<'_>],
    (ib, fb): (usize, f64),
    tol: f64,
) -> bool {
    let p = b[ib].at(fb);
    let d = 1e-2 * a[ia].length.min(b[ib].length);
    if d < 100.0 * tol {
        return false;
    }
    let normal = surface
        .project_point(p)
        .map_or_else(|| surface.normal(0.0, 0.0), |(u, v)| surface.normal(u, v));
    let Some((x, y)) = frame(normal) else {
        return false;
    };
    let angle = |q: Point3| {
        let w = q - p;
        w.dot(y).atan2(w.dot(x))
    };
    let arms = [
        arm(b, ib, fb, true, p, d),
        arm(b, ib, fb, false, p, d),
        arm(a, ia, fa, false, p, d),
        arm(a, ia, fa, true, p, d),
    ];
    let [Some(out), Some(back), Some(before), Some(after)] = arms.map(|q| q.map(angle)) else {
        return false;
    };
    matches!(
        (left_of(out, back, before), left_of(out, back, after)),
        (Some(came), Some(left)) if came != left
    )
}

fn issue(face_id: FaceId, description: String) -> ValidationIssue {
    ValidationIssue {
        check: CheckId::FaceWiresIntersect,
        severity: Severity::Error,
        entity: EntityRef::Face(face_id),
        description,
        deviation: None,
    }
}

/// A wire's box, grown by how far its edges stand off their chords.
struct Bounds {
    lo: [f64; 3],
    hi: [f64; 3],
}

impl Bounds {
    fn of(wire: &[Arc<'_>]) -> Self {
        let mut bounds = Self {
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        };
        for arc in wire {
            for k in 0..3 {
                bounds.lo[k] = bounds.lo[k].min(arc.lo[k] - arc.sag);
                bounds.hi[k] = bounds.hi[k].max(arc.hi[k] + arc.sag);
            }
        }
        bounds
    }

    fn within(&self, outer: &Self) -> bool {
        (0..3).all(|k| self.lo[k] >= outer.lo[k] && self.hi[k] <= outer.hi[k])
    }

    fn clear_of(&self, other: &Self, reach: f64) -> bool {
        (0..3).any(|k| self.lo[k] > other.hi[k] + reach || other.lo[k] > self.hi[k] + reach)
    }
}

/// On a plane face, the first hole that lies outside the outline or inside
/// another hole.
fn misplaced_hole(
    topo: &Topology,
    wire_ids: &[WireId],
    wires: &[Vec<Arc<'_>>],
    bounds: &[Bounds],
    normal: Vec3,
    tol: f64,
) -> Result<Option<String>, CheckError> {
    let Some((x, y)) = frame(normal) else {
        return Ok(None);
    };
    let Some(origin) = wires
        .first()
        .and_then(|w| w.first())
        .map(|arc| arc.points[0])
    else {
        return Ok(None);
    };
    let flat = |p: Point3| Point2::new((p - origin).dot(x), (p - origin).dot(y));
    let region = |k: usize| wire_region_2d(topo, wire_ids[k], origin, x, y, tol);
    // Where a hole stands to a wire's region, read at the first of its
    // edges' middles that lies clear of that wire.
    let side = |hole: usize, of: &PlanarRegion| {
        wires[hole]
            .iter()
            .filter(|arc| arc.length > tol)
            .find_map(|arc| of.side(flat(arc.at(0.5)), tol))
    };
    let Some(outline) = region(0)? else {
        return Ok(None);
    };
    for hole in 1..wires.len() {
        if side(hole, &outline) == Some(false) {
            return Ok(Some(format!(
                "inner wire {} lies outside the outer wire",
                wire_ids[hole].index()
            )));
        }
    }
    for other in 1..wires.len() {
        let mut around: Option<Option<PlanarRegion>> = None;
        for hole in 1..wires.len() {
            if hole == other || !bounds[hole].within(&bounds[other]) {
                continue;
            }
            if around.is_none() {
                around = Some(region(other)?);
            }
            if let Some(Some(around)) = &around
                && side(hole, around) == Some(true)
            {
                return Ok(Some(format!(
                    "inner wire {} lies inside inner wire {}",
                    wire_ids[hole].index(),
                    wire_ids[other].index()
                )));
            }
        }
    }
    Ok(None)
}

/// Check that a face's wires keep clear of each other.
///
/// Reports a hole that crosses the outline or another hole, on a face of
/// any surface. On a plane face, also reports a hole that lies outside the
/// outline or inside another hole. Wires that share a vertex or an edge, or
/// touch without crossing, are not reported. `tolerance` is the gap within
/// which two edges meet.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_face_wires_apart(
    topo: &Topology,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let face = topo.face(face_id)?;
    if face.inner_wires().is_empty() {
        return Ok(vec![]);
    }
    let wire_ids: Vec<WireId> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    let mut wires = Vec::with_capacity(wire_ids.len());
    for &wid in &wire_ids {
        let arcs = topo
            .wire(wid)?
            .edges()
            .iter()
            .map(|oe| Arc::of(topo, oe.edge(), oe.is_forward()))
            .collect::<Result<Vec<_>, _>>()?;
        wires.push(arcs);
    }
    let bounds: Vec<Bounds> = wires.iter().map(|w| Bounds::of(w)).collect();
    for (j, b) in wires.iter().enumerate() {
        for (i, a) in wires.iter().enumerate().skip(j + 1) {
            if bounds[i].clear_of(&bounds[j], tolerance) {
                continue;
            }
            for (ia, arc_a) in a.iter().enumerate() {
                if arc_a.length <= tolerance {
                    continue;
                }
                for (ib, arc_b) in b.iter().enumerate() {
                    if arc_b.length <= tolerance || arc_a.edge == arc_b.edge {
                        continue;
                    }
                    for (fa, fb) in meetings(arc_a, arc_b, tolerance) {
                        if crosses(face.surface(), a, (ia, fa), b, (ib, fb), tolerance) {
                            let name = if j == 0 {
                                "the outer wire"
                            } else {
                                "another inner wire"
                            };
                            return Ok(vec![issue(
                                face_id,
                                format!("inner wire {} crosses {name}", wire_ids[i].index()),
                            )]);
                        }
                    }
                }
            }
        }
    }
    if let FaceSurface::Plane { normal, .. } = face.surface()
        && let Some(description) =
            misplaced_hole(topo, &wire_ids, &wires, &bounds, *normal, tolerance)?
    {
        return Ok(vec![issue(face_id, description)]);
    }
    Ok(vec![])
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::surfaces::CylindricalSurface;
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::{Vertex, VertexId};
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const Z: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    fn vertex(topo: &mut Topology, p: [f64; 3]) -> VertexId {
        topo.add_vertex(Vertex::new(Point3::new(p[0], p[1], p[2]), 1e-7))
    }

    /// A closed loop of line edges through `corners`.
    fn polygon(topo: &mut Topology, corners: &[VertexId]) -> WireId {
        let edges = (0..corners.len())
            .map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % corners.len()]);
                OrientedEdge::new(topo.add_edge(Edge::new(a, b, EdgeCurve::Line)), true)
            })
            .collect();
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    fn points(topo: &mut Topology, corners: &[[f64; 3]]) -> Vec<VertexId> {
        corners.iter().map(|&p| vertex(topo, p)).collect()
    }

    /// A circle about `center` square to z, as one closed edge.
    fn circle_edge(topo: &mut Topology, center: [f64; 3], r: f64, nurbs: bool) -> EdgeId {
        let circle = Circle3D::new(Point3::new(center[0], center[1], center[2]), Z, r).unwrap();
        let start = circle.evaluate(0.0);
        let v = vertex(topo, [start.x(), start.y(), start.z()]);
        let curve = if nurbs {
            EdgeCurve::NurbsCurve(
                brepkit_geometry::convert::curve_to_nurbs::circle_to_nurbs(&circle, 0.0, TAU)
                    .unwrap(),
            )
        } else {
            EdgeCurve::Circle(circle)
        };
        topo.add_edge(Edge::new(v, v, curve))
    }

    fn circle(topo: &mut Topology, center: [f64; 3], r: f64) -> WireId {
        let e = circle_edge(topo, center, r, false);
        topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
    }

    fn square(topo: &mut Topology) -> WireId {
        let corners = points(
            topo,
            &[
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        polygon(topo, &corners)
    }

    fn flat(topo: &mut Topology, outer: WireId, holes: Vec<WireId>) -> FaceId {
        topo.add_face(Face::new(
            outer,
            holes,
            FaceSurface::Plane { normal: Z, d: 0.0 },
        ))
    }

    fn found(topo: &Topology, face: FaceId) -> Option<String> {
        check_face_wires_apart(topo, face, 1e-6)
            .unwrap()
            .into_iter()
            .next()
            .map(|i| i.description)
    }

    #[test]
    fn a_hole_within_the_outline_passes() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let holes = vec![
            circle(&mut topo, [3.0, 3.0, 0.0], 2.0),
            circle(&mut topo, [7.0, 7.0, 0.0], 2.0),
        ];
        let face = flat(&mut topo, outer, holes);
        assert_eq!(found(&topo, face), None);
    }

    #[test]
    fn a_hole_across_the_outline_is_reported() {
        for nurbs in [false, true] {
            let mut topo = Topology::default();
            let outer = square(&mut topo);
            let e = circle_edge(&mut topo, [10.0, 5.0, 0.0], 2.0, nurbs);
            let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
            let face = flat(&mut topo, outer, vec![hole]);
            let got = found(&topo, face).unwrap();
            assert!(
                got.contains("crosses the outer wire"),
                "nurbs {nurbs}: {got}"
            );
        }
    }

    #[test]
    fn a_hole_outside_the_outline_is_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let hole = circle(&mut topo, [20.0, 5.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("outside the outer wire")
        );
    }

    #[test]
    fn a_hole_outside_an_outline_of_splines_is_reported() {
        let mut topo = Topology::default();
        let e = circle_edge(&mut topo, [0.0, 0.0, 0.0], 5.0, true);
        let outer = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let inside = circle(&mut topo, [1.0, 1.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![inside]);
        assert_eq!(found(&topo, face), None);
        let outside = circle(&mut topo, [9.0, 0.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![outside]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("outside the outer wire")
        );
    }

    #[test]
    fn two_holes_that_cross_are_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let holes = vec![
            circle(&mut topo, [4.0, 5.0, 0.0], 2.0),
            circle(&mut topo, [6.0, 5.0, 0.0], 2.0),
        ];
        let face = flat(&mut topo, outer, holes);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses another inner wire")
        );
    }

    #[test]
    fn a_hole_inside_a_hole_is_reported() {
        for swap in [false, true] {
            let mut topo = Topology::default();
            let outer = square(&mut topo);
            let mut holes = vec![
                circle(&mut topo, [5.0, 5.0, 0.0], 3.0),
                circle(&mut topo, [5.5, 5.0, 0.0], 1.0),
            ];
            if swap {
                holes.reverse();
            }
            let face = flat(&mut topo, outer, holes);
            assert!(
                found(&topo, face)
                    .unwrap()
                    .contains("lies inside inner wire")
            );
        }
    }

    /// A hole that meets the outline at a vertex they share, at a corner of
    /// the hole or where the hole's circle touches the outline's edge.
    #[test]
    fn a_hole_that_shares_a_vertex_with_the_outline_passes() {
        let mut topo = Topology::default();
        let corners = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let outer = polygon(&mut topo, &corners);
        let more = points(&mut topo, &[[7.0, 4.0, 0.0], [3.0, 4.0, 0.0]]);
        let triangle = polygon(&mut topo, &[corners[1], more[0], more[1]]);
        let face = flat(&mut topo, outer, vec![triangle]);
        assert_eq!(found(&topo, face), None);

        let touching = Circle3D::new(Point3::new(5.0, 2.0, 0.0), Z, 2.0).unwrap();
        // The circle's start, at angle zero, is not the touching point: turn
        // it so its vertex is the outline's.
        let turned = Circle3D::with_axes(
            touching.center(),
            Z,
            2.0,
            Vec3::new(0.0, -1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let e = topo.add_edge(Edge::new(corners[1], corners[1], EdgeCurve::Circle(turned)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let face = flat(&mut topo, outer, vec![hole]);
        assert_eq!(found(&topo, face), None);
    }

    #[test]
    fn a_hole_that_touches_the_outline_between_its_vertices_passes() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let hole = circle(&mut topo, [5.0, 2.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![hole]);
        assert_eq!(found(&topo, face), None);
    }

    /// A cylinder's band between z = 0 and 10, with its seam, and a window
    /// of two arcs and two rulings from `low` to `high` over a quarter turn
    /// from the angle `from`.
    fn windowed_band(topo: &mut Topology, from: f64, low: f64, high: f64) -> FaceId {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let ring = |z: f64| Circle3D::new(Point3::new(0.0, 0.0, z), Z, 3.0).unwrap();
        let (bottom, top) = (ring(0.0), ring(10.0));
        let a = bottom.evaluate(0.0);
        let b = top.evaluate(0.0);
        let va = vertex(topo, [a.x(), a.y(), a.z()]);
        let vb = vertex(topo, [b.x(), b.y(), b.z()]);
        let rim_a = topo.add_edge(Edge::new(va, va, EdgeCurve::Circle(bottom)));
        let rim_b = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(top)));
        let seam = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
        let outer = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(rim_a, true),
                    OrientedEdge::new(seam, true),
                    OrientedEdge::new(rim_b, false),
                    OrientedEdge::new(seam, false),
                ],
                true,
            )
            .unwrap(),
        );
        let (lower, upper) = (ring(low), ring(high));
        let to = from + std::f64::consts::FRAC_PI_2;
        let corner = |topo: &mut Topology, c: &Circle3D, t: f64| {
            let p = c.evaluate(t);
            vertex(topo, [p.x(), p.y(), p.z()])
        };
        let (p0, p1) = (corner(topo, &lower, from), corner(topo, &lower, to));
        let (q0, q1) = (corner(topo, &upper, from), corner(topo, &upper, to));
        let edges = [
            (
                topo.add_edge(Edge::new(p0, p1, EdgeCurve::Circle(lower))),
                true,
            ),
            (topo.add_edge(Edge::new(p1, q1, EdgeCurve::Line)), true),
            (
                topo.add_edge(Edge::new(q0, q1, EdgeCurve::Circle(upper))),
                false,
            ),
            (topo.add_edge(Edge::new(p0, q0, EdgeCurve::Line)), false),
        ];
        let window = topo.add_wire(
            Wire::new(
                edges
                    .iter()
                    .map(|&(e, f)| OrientedEdge::new(e, f))
                    .collect(),
                true,
            )
            .unwrap(),
        );
        let surface = FaceSurface::Cylinder(CylindricalSurface::new(origin, Z, 3.0).unwrap());
        topo.add_face(Face::new(outer, vec![window], surface))
    }

    #[test]
    fn a_window_in_a_cylinders_band_passes() {
        let mut topo = Topology::default();
        let face = windowed_band(&mut topo, 1.0, 3.0, 7.0);
        assert_eq!(found(&topo, face), None);
    }

    #[test]
    fn a_window_across_a_cylinders_rim_is_reported() {
        let mut topo = Topology::default();
        let face = windowed_band(&mut topo, 1.0, 3.0, 12.0);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }

    #[test]
    fn a_window_across_a_cylinders_seam_is_reported() {
        let mut topo = Topology::default();
        let face = windowed_band(&mut topo, -0.5, 3.0, 7.0);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }
}

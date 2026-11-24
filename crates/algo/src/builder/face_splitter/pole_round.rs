//! A ball face that holds one pole of its chart, its other pole off the
//! face, and the loops that split it.
//!
//! In the chart's `(u, v)`, a loop that winds the axis bounds no area, so
//! its pieces cannot be read there. Seen from the pole the face holds, with
//! the other pole off the face, the face is a region of a plane: the
//! azimuthal equal-area map about the held pole takes the whole ball, less
//! the other pole, onto a disc. There each loop is a closed curve that
//! winds as an outline or as a hole, and holes nest in their pieces.

use brepkit_math::madd::Madd;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::super::classify_2d::{
    boundary_eps, distance_to_polygon_boundary, point_in_polygon_2d, sample_interior_point_where,
};
use super::super::split_types::OrientedPCurveEdge;

/// Samples per curved edge.
const SAMPLES: u32 = 64;

/// The chord below which an edge's ends are one point and it runs its
/// whole curve, as [`EdgeCurve::domain_with_endpoints`] reads it.
const CLOSED: f64 = 1e-9;

/// The plane about a ball's held pole: the azimuthal equal-area map, which
/// keeps areas (the whole ball maps onto a disc of radius 2, area `4 pi`).
pub(super) struct PoleView {
    center: Point3,
    radius: f64,
    north: bool,
    pole: Vec3,
    x: Vec3,
    y: Vec3,
}

impl PoleView {
    /// The view of `sphere` about its north pole, or its south pole when
    /// `north` is false.
    pub(super) fn new(sphere: &SphericalSurface, north: bool) -> Self {
        let z = sphere.z_axis();
        Self {
            center: sphere.center(),
            radius: sphere.radius(),
            north,
            pole: if north { z } else { -z },
            x: sphere.x_axis(),
            y: sphere.y_axis(),
        }
    }

    /// Where `p` maps. `None` within about 1.4e-6 rad of the far pole,
    /// which has no image.
    fn map(&self, p: Point3) -> Option<Point2> {
        let d = (p - self.center) * (1.0 / self.radius);
        let rise = 1.0 + d.dot(self.pole);
        if rise <= 1e-12 {
            return None;
        }
        let k = (2.0 / rise).sqrt();
        Some(Point2::new(k * d.dot(self.x), k * d.dot(self.y)))
    }

    /// Points on the ball, mapped.
    fn polygon(&self, pts: &[Point3]) -> Option<Vec<Point2>> {
        pts.iter().map(|&p| self.map(p)).collect()
    }

    /// Twice the signed area a run of edges adds to a loop it lies in: the
    /// shoelace terms of each edge's samples, its end included. The terms of
    /// an edge and of its twin, walked the other way on the same samples,
    /// cancel.
    fn twice_area(&self, edges: &[OrientedPCurveEdge]) -> Option<f64> {
        let mut sum = 0.0;
        for e in edges {
            let mut pts = Vec::new();
            walk_edge(e, &mut pts);
            pts.push(e.end_3d);
            let mapped = self.polygon(&pts)?;
            sum += mapped
                .windows(2)
                .map(|w| w[0].x() * w[1].y() - w[1].x() * w[0].y())
                .sum::<f64>();
        }
        Some(sum)
    }

    /// The signed area of a loop; 0 where it does not map.
    pub(super) fn loop_area(&self, lp: &[OrientedPCurveEdge]) -> f64 {
        self.twice_area(lp).map_or(0.0, |a| 0.5 * a)
    }

    /// Whether the face whose outline is `boundary` (its edges as they are
    /// walked) lies on the outline's left in the chart's `(u, v)`. The face
    /// lies inside its outline here, on the outline's left where the outline
    /// runs counter-clockwise. The map keeps the way round of `(u, v)` about
    /// the north pole, whose distance from the centre falls as `v` rises,
    /// and turns it about the south pole.
    pub(super) fn face_left_in_uv(&self, boundary: &[OrientedPCurveEdge]) -> Option<bool> {
        let area = self.twice_area(boundary)?;
        (area.abs() > 1e-9).then_some((area > 0.0) == self.north)
    }

    /// A loop of edges, each sampled along its own curve in the way the
    /// loop walks it, mapped.
    fn loop_polygon(&self, lp: &[OrientedPCurveEdge]) -> Option<Vec<Point2>> {
        let mut pts = Vec::new();
        for e in lp {
            walk_edge(e, &mut pts);
        }
        self.polygon(&pts)
    }
}

/// `e`'s start, then samples along its curve toward its end, in the way the
/// loop walks it.
fn walk_edge(e: &OrientedPCurveEdge, out: &mut Vec<Point3>) {
    use std::f64::consts::TAU;
    out.push(e.start_3d);
    if matches!(e.curve_3d, EdgeCurve::Line) {
        return;
    }
    let fraction = |k: u32| f64::from(k) / f64::from(SAMPLES);
    let closed = (e.start_3d - e.end_3d).length() < CLOSED;
    match &e.curve_3d {
        EdgeCurve::Circle(c) if closed => {
            let (a0, turn) = (c.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..SAMPLES).map(|k| c.evaluate(turn.madd(fraction(k), a0))));
        }
        EdgeCurve::Ellipse(el) if closed => {
            let (a0, turn) = (el.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..SAMPLES).map(|k| el.evaluate(turn.madd(fraction(k), a0))));
        }
        curve => {
            let (s3, e3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            let (t0, t1) = curve.domain_with_endpoints(s3, e3);
            let mut samples: Vec<Point3> = (1..SAMPLES)
                .map(|k| curve.evaluate_with_endpoints((t1 - t0).madd(fraction(k), t0), s3, e3))
                .collect();
            // A closed curve (a spline loop) runs its whole domain, from its
            // vertex back to it either way: the walk's sense picks the way.
            // A marched curve's domain can run from either end.
            let backward = if closed {
                !e.forward
            } else {
                let at_t0 = curve.evaluate_with_endpoints(t0, s3, e3);
                (at_t0 - e.start_3d).length() > (at_t0 - e.end_3d).length()
            };
            if backward {
                samples.reverse();
            }
            out.extend(samples);
        }
    }
}

/// A point of the region inside `outline` and outside each of `holes`,
/// clear of all of them: the polygon's own interior point where it serves,
/// else one stepped off the middle of a side of the outline or a hole, a
/// quarter of the side's length or a twentieth, a hundredth or a thousandth
/// (a region thinner than its sides are long).
fn region_point(outline: &[Point2], holes: &[Vec<Point2>]) -> Option<Point2> {
    let walls = || std::iter::once(outline).chain(holes.iter().map(Vec::as_slice));
    let holds = |p: Point2| {
        point_in_polygon_2d(p, outline)
            && !holes.iter().any(|h| point_in_polygon_2d(p, h))
            && walls().all(|w| distance_to_polygon_boundary(p, w) > boundary_eps(w))
    };
    if let Some(p) = sample_interior_point_where(outline, holds) {
        return Some(p);
    }
    for wall in walls() {
        for (k, &a) in wall.iter().enumerate() {
            let b = wall[(k + 1) % wall.len()];
            let side = b - a;
            let length = side.length();
            if length <= 0.0 {
                continue;
            }
            let middle = Point2::new(0.5 * (a.x() + b.x()), 0.5 * (a.y() + b.y()));
            let normal = Vec2::new(-side.y(), side.x()) * (1.0 / length);
            for share in [0.25, 0.05, 0.01, 1e-3] {
                for way in [1.0, -1.0] {
                    let p = middle + normal * (way * share * length);
                    if holds(p) {
                        return Some(p);
                    }
                }
            }
        }
    }
    None
}

/// The 3D point half way along `e`'s curve.
fn edge_middle(e: &OrientedPCurveEdge) -> Point3 {
    let (s3, e3) = if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    };
    let (t0, t1) = e.curve_3d.domain_with_endpoints(s3, e3);
    e.curve_3d
        .evaluate_with_endpoints(f64::midpoint(t0, t1), s3, e3)
}

/// Whether two loop edges are one edge of the face, in either sense.
fn same_edge(a: &OrientedPCurveEdge, b: &OrientedPCurveEdge, near: f64) -> bool {
    let ends = ((a.start_3d - b.start_3d).length() <= near
        && (a.end_3d - b.end_3d).length() <= near)
        || ((a.start_3d - b.end_3d).length() <= near && (a.end_3d - b.start_3d).length() <= near);
    ends && (edge_middle(a) - edge_middle(b)).length() <= near
}

/// A piece: its outline and its holes.
pub(super) type Piece = (Vec<OrientedPCurveEdge>, Vec<Vec<OrientedPCurveEdge>>);

/// The pieces that `loops` bound on a ball face whose outline is the run of
/// edges `boundary`, cut by the section halves `sections` (each section run
/// both ways), seen in `view`.
///
/// The loops must run every piece of the outline and every section half
/// once: a section the walk left out would leave the face standing whole
/// across it.
///
/// A loop that winds as the outline does is a piece's outline, one that
/// winds against it a hole. Each hole goes to the smallest outline that
/// holds the middle of its first edge, of those that do not run along that
/// edge (a hole's twin, the outline of the piece inside it), and that
/// outline must run none of the hole's edges. The areas
/// must balance: the pieces' outlines less their holes cover the face's
/// own area. Read on the same samples, each section the pieces run both
/// ways cancels and each piece of the outline counts once, so a piece left
/// out or taken twice shows.
///
/// `None` when a loop does not map, bounds no area, runs an edge both ways
/// or passes a vertex twice (two regions toured as one, touching there), a
/// hole finds no outline or one that shares an edge with it, the areas do
/// not balance, or two pieces overlap.
pub(super) fn pieces(
    view: &PoleView,
    boundary: &[OrientedPCurveEdge],
    sections: &[OrientedPCurveEdge],
    loops: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let face_area = 0.5 * view.twice_area(boundary)?;
    if face_area.abs() <= 1e-9 {
        return None;
    }
    let runs = |on_section: bool| {
        loops
            .iter()
            .flatten()
            .filter(|e| e.source_edge_idx.is_some() == on_section)
            .count()
    };
    if sections.is_empty() || runs(true) != sections.len() || runs(false) != boundary.len() {
        return None;
    }
    let near = 100.0 * tol;
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, Vec<Point2>, f64)> = Vec::new();
    let mut holes: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    for lp in loops {
        let both_ways = lp
            .iter()
            .enumerate()
            .any(|(i, a)| lp[i + 1..].iter().any(|b| same_edge(a, b, near)));
        let pinched = lp.iter().enumerate().any(|(i, a)| {
            lp[i + 1..]
                .iter()
                .any(|b| (a.start_3d - b.start_3d).length() <= near)
        });
        if both_ways || pinched {
            return None;
        }
        let poly = view.loop_polygon(&lp)?;
        let area = 0.5 * view.twice_area(&lp)? * face_area.signum();
        let perimeter: f64 = poly
            .iter()
            .zip(poly.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        // The plane's lengths are the ball's over its radius.
        if area.abs() <= perimeter * tol / view.radius {
            return None;
        }
        if area > 0.0 {
            outers.push((lp, poly, area));
        } else {
            holes.push((lp, -area));
        }
    }
    let covered = outers.iter().map(|o| o.2).sum::<f64>() - holes.iter().map(|h| h.1).sum::<f64>();
    if (covered - face_area.abs()).abs() > 1e-6 * face_area.abs() {
        return None;
    }
    let mut pieces: Vec<Piece> = outers
        .iter()
        .map(|(lp, _, _)| (lp.clone(), Vec::new()))
        .collect();
    for (hole, _) in holes {
        let first = hole.first()?;
        let probe = view.map(edge_middle(first))?;
        let owner = outers
            .iter()
            .enumerate()
            .filter(|(_, (lp, poly, _))| {
                !lp.iter().any(|e| same_edge(e, first, near)) && point_in_polygon_2d(probe, poly)
            })
            .min_by(|a, b| a.1.2.total_cmp(&b.1.2))
            .map(|(i, _)| i)?;
        // A hole lies inside its piece's outline: an edge on both bounds
        // the piece on both sides.
        if outers[owner]
            .0
            .iter()
            .any(|e| hole.iter().any(|h| same_edge(e, h, near)))
        {
            return None;
        }
        pieces[owner].1.push(hole);
    }
    // Loops that use each half once and balance the area can still overlap:
    // two section loops that cross, read as their union and their lens with
    // both whole loops as holes. A piece's region (its outline less its
    // holes) holds a point of its own that no other piece's region holds.
    let regions: Vec<(&[Point2], Vec<Vec<Point2>>)> = outers
        .iter()
        .zip(&pieces)
        .map(|((_, poly, _), (_, piece_holes))| {
            let holes = piece_holes
                .iter()
                .map(|h| view.loop_polygon(h))
                .collect::<Option<Vec<_>>>()?;
            Some((poly.as_slice(), holes))
        })
        .collect::<Option<_>>()?;
    let holds = |(outline, holes): &(&[Point2], Vec<Vec<Point2>>), p: Point2| {
        point_in_polygon_2d(p, outline) && !holes.iter().any(|h| point_in_polygon_2d(p, h))
    };
    for (i, (outline, holes)) in regions.iter().enumerate() {
        let own = region_point(outline, holes)?;
        if regions
            .iter()
            .enumerate()
            .any(|(j, other)| j != i && holds(other, own))
        {
            return None;
        }
    }
    Some(pieces)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::nurbs::curve::NurbsCurve;

    use super::*;

    fn edge(curve: &EdgeCurve, from: Point3, to: Point3, forward: bool) -> OrientedPCurveEdge {
        let origin = Point2::new(0.0, 0.0);
        OrientedPCurveEdge {
            curve_3d: curve.clone(),
            pcurve: Curve2D::Line(Line2D::new(origin, Vec2::new(1.0, 0.0)).unwrap()),
            start_uv: origin,
            end_uv: origin,
            start_3d: from,
            end_3d: to,
            forward,
            source_edge_idx: None,
            pave_block_id: None,
        }
    }

    fn twin(e: &OrientedPCurveEdge) -> OrientedPCurveEdge {
        OrientedPCurveEdge {
            start_3d: e.end_3d,
            end_3d: e.start_3d,
            forward: !e.forward,
            ..e.clone()
        }
    }

    fn reversed(lp: &[OrientedPCurveEdge]) -> Vec<OrientedPCurveEdge> {
        lp.iter().rev().map(twin).collect()
    }

    /// A small circle of the ball of radius `r` about the unit direction
    /// `d`, `rho` radians round.
    fn small_circle(r: f64, d: Vec3, rho: f64) -> Circle3D {
        let c = Point3::new(0.0, 0.0, 0.0) + d * (r * rho.cos());
        Circle3D::new(c, d, r * rho.sin()).unwrap()
    }

    /// A ball of radius `r` less the cap above `z = 0.95 r`, seen about its
    /// south pole, and its outline: one closed circle.
    fn capped_ball(r: f64) -> (PoleView, OrientedPCurveEdge) {
        let ball = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), r).unwrap();
        let rim = small_circle(r, Vec3::new(0.0, 0.0, 1.0), 0.95_f64.acos());
        let at = rim.evaluate(0.0);
        (
            PoleView::new(&ball, false),
            edge(&EdgeCurve::Circle(rim), at, at, true),
        )
    }

    /// `lp` walked so that it winds as the outline does where `outline`,
    /// against it otherwise.
    fn oriented(
        view: &PoleView,
        boundary: &OrientedPCurveEdge,
        lp: Vec<OrientedPCurveEdge>,
        outline: bool,
    ) -> Vec<OrientedPCurveEdge> {
        let face = view.loop_area(std::slice::from_ref(boundary));
        if (view.loop_area(&lp) * face > 0.0) == outline {
            lp
        } else {
            reversed(&lp)
        }
    }

    /// Two small circles about the south pole of a ball of radius `r`,
    /// `rho` radians round, their middles `apart` radians apart, so that
    /// they cross twice. Their four arcs: the first circle's outside the
    /// second's disc, its inside, then the second circle's outside and
    /// inside. Each arc is walked from the crossing with `y > 0` and carries
    /// its index as its section.
    fn crossing_arcs(r: f64, rho: f64, apart: f64) -> [OrientedPCurveEdge; 4] {
        let (s, c) = (0.5 * apart).sin_cos();
        let d1 = Vec3::new(s, 0.0, -c);
        let d2 = Vec3::new(-s, 0.0, -c);
        let z = -rho.cos() / c;
        let y = (1.0 - z * z).sqrt();
        let (p, q) = (
            Point3::new(0.0, y * r, z * r),
            Point3::new(0.0, -y * r, z * r),
        );
        let arcs = |d: Vec3, other: Vec3| {
            let curve = EdgeCurve::Circle(small_circle(r, d, rho));
            // The arc that runs the circle's way from `p` to `q`, and the
            // one from `q` to `p`.
            let one = edge(&curve, p, q, true);
            let other_way = edge(&curve, q, p, true);
            let inside = |e: &OrientedPCurveEdge| {
                (edge_middle(e) - Point3::new(0.0, 0.0, 0.0)).dot(other) > r * rho.cos()
            };
            let (outer, inner) = if inside(&one) {
                (twin(&other_way), one)
            } else {
                (one, twin(&other_way))
            };
            [outer, inner]
        };
        let [o1, i1] = arcs(d1, d2);
        let [o2, i2] = arcs(d2, d1);
        let mut out = [o1, i1, o2, i2];
        for (k, e) in out.iter_mut().enumerate() {
            e.source_edge_idx = Some(k);
        }
        out
    }

    fn halves(arcs: &[OrientedPCurveEdge]) -> Vec<OrientedPCurveEdge> {
        arcs.iter().flat_map(|e| [e.clone(), twin(e)]).collect()
    }

    /// Arcs `a` from the crossing with `y > 0` and `b` back to it.
    fn round(a: &OrientedPCurveEdge, b: &OrientedPCurveEdge) -> Vec<OrientedPCurveEdge> {
        vec![a.clone(), twin(b)]
    }

    #[test]
    fn crossing_loops_read_as_their_union_and_lens_are_refused() {
        let (view, rim) = capped_ball(1.0);
        let arcs = crossing_arcs(1.0, 0.2, 0.3);
        let [o1, i1, o2, i2] = &arcs;
        let sections = halves(&arcs);
        let at = |lp, outline| oriented(&view, &rim, lp, outline);
        // A loop turned to start on `arc`.
        let from = |mut lp: Vec<OrientedPCurveEdge>, arc: &OrientedPCurveEdge| {
            if lp[0].source_edge_idx != arc.source_edge_idx {
                lp.rotate_left(1);
            }
            lp
        };
        // Each whole circle as a hole, starting on its arc outside the
        // other disc, so both go to the face.
        let overlapping = vec![
            vec![rim.clone()],
            at(round(o1, o2), true),
            at(round(i1, i2), true),
            from(at(round(o1, i1), false), o1),
            from(at(round(o2, i2), false), o2),
        ];
        assert!(
            pieces(
                &view,
                std::slice::from_ref(&rim),
                &sections,
                overlapping,
                1e-7
            )
            .is_none()
        );
        // Each whole circle as an outline, the union and the lens as holes.
        // The lens starts on the first circle's arc inside the second disc,
        // so it goes to the second circle, whose outline runs the lens's
        // other arc.
        let paired = vec![
            vec![rim.clone()],
            at(round(o1, i1), true),
            at(round(o2, i2), true),
            from(at(round(o1, o2), false), o1),
            from(at(round(i1, i2), false), i1),
        ];
        assert!(pieces(&view, std::slice::from_ref(&rim), &sections, paired, 1e-7).is_none());

        let partition = vec![
            vec![rim.clone()],
            at(round(o1, o2), false),
            at(round(o1, i2), true),
            at(round(o2, i1), true),
            at(round(i1, i2), true),
        ];
        let split = pieces(
            &view,
            std::slice::from_ref(&rim),
            &sections,
            partition,
            1e-7,
        )
        .unwrap();
        assert_eq!(split.len(), 4);
        assert_eq!(split.iter().map(|p| p.1.len()).sum::<usize>(), 1);
    }

    /// A closed spline circle about the south pole of the unit ball, `rho`
    /// radians round: a rational quadratic in nine points.
    fn spline_loop(rho: f64) -> OrientedPCurveEdge {
        let circle = small_circle(1.0, Vec3::new(0.0, 0.0, -1.0), rho);
        let (c, r) = (circle.center(), circle.radius());
        let (u, v) = (circle.u_axis(), circle.v_axis());
        let w = std::f64::consts::FRAC_1_SQRT_2;
        let corners = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
        ];
        let points = corners.map(|(a, b)| c + u * (r * a) + v * (r * b)).to_vec();
        let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
        let knots = vec![
            0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
        ];
        let curve = NurbsCurve::new(2, knots, points, weights).unwrap();
        let at = c + u * r;
        let mut e = edge(&EdgeCurve::NurbsCurve(curve), at, at, true);
        e.source_edge_idx = Some(0);
        e
    }

    #[test]
    fn a_closed_spline_walked_both_ways_cancels() {
        let (view, rim) = capped_ball(1.0);
        let one_way = spline_loop(0.3);
        let other_way = twin(&one_way);
        let area = view.loop_area(std::slice::from_ref(&one_way));
        assert!(area.abs() > 0.1, "{area}");
        let sum = area + view.loop_area(std::slice::from_ref(&other_way));
        assert!(sum.abs() < 1e-12 * area.abs(), "{sum}");

        // The face less the disc round the pole, and the disc.
        let sections = vec![one_way.clone(), other_way];
        let loops = vec![
            vec![rim.clone()],
            oriented(&view, &rim, vec![one_way.clone()], false),
            oriented(&view, &rim, vec![one_way], true),
        ];
        let split = pieces(&view, std::slice::from_ref(&rim), &sections, loops, 1e-7).unwrap();
        assert_eq!(split.len(), 2);
    }

    #[test]
    fn a_lens_a_thousandth_wide_on_a_ball_ten_thousand_round_is_a_piece() {
        // On a ball of radius 1e4 the lens is 1e-3 wide, far over the 1e-7
        // tolerance, though its plane image is 1e-7 wide.
        let r = 1e4;
        let rho = 0.05;
        let (view, rim) = capped_ball(r);
        let arcs = crossing_arcs(r, rho, 2.0 * rho - 1e-7);
        let [o1, i1, o2, i2] = &arcs;
        let sections = halves(&arcs);
        let at = |lp, outline| oriented(&view, &rim, lp, outline);
        let partition = vec![
            vec![rim.clone()],
            at(round(o1, o2), false),
            at(round(o1, i2), true),
            at(round(o2, i1), true),
            at(round(i1, i2), true),
        ];
        let split = pieces(
            &view,
            std::slice::from_ref(&rim),
            &sections,
            partition,
            1e-7,
        )
        .unwrap();
        assert_eq!(split.len(), 4);
    }
}

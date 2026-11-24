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
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::super::classify_2d::{distance_to_polygon_boundary, point_in_polygon_2d};
use super::super::split_types::OrientedPCurveEdge;
use super::super::wire_builder::build_wire_loops_dcel_flat;
use super::special_cases::reverse_loop;

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

    /// Whether a loop of signed area `area`, sampled as `poly`, bounds no
    /// area: its area is within its perimeter times `tol`, a length on the
    /// ball, which the plane reads over the ball's radius.
    fn flat_reading(&self, area: f64, poly: &[Point2], tol: f64) -> bool {
        let perimeter: f64 = poly
            .iter()
            .zip(poly.iter().cycle().skip(1))
            .map(|(a, b)| (*b - *a).length())
            .sum();
        area.abs() <= perimeter * tol / self.radius
    }

    /// Whether a loop bounds no area ([`Self::flat_reading`]), or does not
    /// map.
    fn flat(&self, lp: &[OrientedPCurveEdge], tol: f64) -> bool {
        match (self.loop_polygon(lp), self.twice_area(lp)) {
            (Some(poly), Some(twice)) => self.flat_reading(0.5 * twice, &poly, tol),
            _ => true,
        }
    }

    /// The signed area of a loop; 0 where it does not map.
    #[cfg(test)]
    fn loop_area(&self, lp: &[OrientedPCurveEdge]) -> f64 {
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

    /// A bound on how far the image of each point of a side's hull
    /// ([`Side::hull`]) lies from the segment between the images of the
    /// side's ends; infinite where the hull reaches the far pole's tangent
    /// plane. `None` where the side's piece has no hull.
    ///
    /// The map is `f`, `m` the side's middle, and `A` the affine map that
    /// agrees with `f` and its derivative at `m`. Each hull point `h` lies
    /// within half the most of `|D2 f [u, u]|` (`u = h - m`) of `A(h)`;
    /// `A(h)` within `|Df(m)|` times the most that a hull point lies off the
    /// side of the image under `A` of the side; and that within half the
    /// same most of the segment between the ends' images.
    fn side_bound(&self, side: &Side) -> Option<f64> {
        let hull = side.hull()?;
        let (a, b) = (side.from, side.to);
        let m = a + (b - a) * 0.5;
        let r = self.radius;
        // In the ball's own units, about the held pole `n`: `f(d) = k(w) (d
        // x, d y)` with `w = 1 + d n` and `k(w) = sqrt(2 / w)`, so that
        // `|Df [v]| <= |k'| |(d x, d y)| |v| + k |v|` and `|D2 f [u, u]| <=
        // k'' (n u)^2 |(d x, d y)| + 2 |k'| |n u| |u|`. Along a segment from
        // `m`, `w` falls by at most `|n u|`, and `k`, `|k'|` and `k''` rise
        // as `w` falls.
        let off = hull
            .iter()
            .map(|&h| segment_distance(h, a, b))
            .fold(0.0, f64::max)
            / r;
        let reach = hull.iter().map(|&h| (h - m).length()).fold(0.0, f64::max) / r;
        let rise = hull
            .iter()
            .map(|&h| (h - m).dot(self.pole).abs())
            .fold(0.0, f64::max)
            / r;
        let d = (m - self.center) * (1.0 / r);
        let w = 1.0 + d.dot(self.pole);
        let low = w - rise;
        if low <= 0.0 {
            return Some(f64::INFINITY);
        }
        let k = |w: f64| (2.0 / w).sqrt();
        let k1 = |w: f64| k(w) / (2.0 * w);
        let k2 = |w: f64| 3.0 * k(w) / (4.0 * w * w);
        let level = d.dot(self.x).hypot(d.dot(self.y));
        let slope = k1(w).madd(level, k(w));
        let bend = (k2(low) * (level + reach) * rise).madd(rise, 2.0 * k1(low) * rise * reach);
        Some(slope.madd(off, bend))
    }

    /// Whether the point `at` on the ball lies inside the loop of `sides`:
    /// read on a polygon of their ends, each side cut in two (up to
    /// [`CUTS`] times) until the point lies farther from it than twice its
    /// bound ([`Self::side_bound`]). The loop and the polygon differ by a
    /// closed curve along each side within its bound, which winds no point
    /// farther than that from the side, so both wind the point alike.
    /// `None` where a side stays too near, or the point does not map.
    fn inside(&self, at: Point3, sides: &[Side]) -> Option<bool> {
        let q = self.map(at)?;
        let mut poly = Vec::new();
        for side in sides {
            self.clear_side(q, *side, CUTS, &mut poly)?;
        }
        Some(point_in_polygon_2d(q, &poly))
    }

    /// Pushes onto `poly` the image of the start of `side`, or of each part
    /// it is cut into, so that `q` lies farther than twice its bound from
    /// each. `None` where that takes more than `cuts` cuts, or a part has no
    /// bound or does not map.
    fn clear_side(&self, q: Point2, side: Side, cuts: u32, poly: &mut Vec<Point2>) -> Option<()> {
        let (from, to) = (self.map(side.from)?, self.map(side.to)?);
        let d = distance_to_polygon_boundary(q, &[from, to]);
        if d > 2.0 * self.side_bound(&side)? && d > 1e-12 {
            poly.push(from);
            return Some(());
        }
        if cuts == 0 {
            return None;
        }
        let (first, second) = side.halves();
        self.clear_side(q, first, cuts - 1, poly)?;
        self.clear_side(q, second, cuts - 1, poly)
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
    walk_edge_in(e, SAMPLES, out);
}

/// [`walk_edge`] in `steps` steps.
fn walk_edge_in(e: &OrientedPCurveEdge, steps: u32, out: &mut Vec<Point3>) {
    use std::f64::consts::TAU;
    out.push(e.start_3d);
    if matches!(e.curve_3d, EdgeCurve::Line) {
        return;
    }
    let fraction = |k: u32| f64::from(k) / f64::from(steps);
    let closed = (e.start_3d - e.end_3d).length() < CLOSED;
    match &e.curve_3d {
        EdgeCurve::Circle(c) if closed => {
            let (a0, turn) = (c.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..steps).map(|k| c.evaluate(turn.madd(fraction(k), a0))));
        }
        EdgeCurve::Ellipse(el) if closed => {
            let (a0, turn) = (el.project(e.start_3d), if e.forward { TAU } else { -TAU });
            out.extend((1..steps).map(|k| el.evaluate(turn.madd(fraction(k), a0))));
        }
        curve => {
            let (s3, e3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            let (t0, t1) = curve.domain_with_endpoints(s3, e3);
            let mut samples: Vec<Point3> = (1..steps)
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

/// A side of a loop's polygon: the piece of an edge's curve from parameter
/// `a` to `b`, from the vertex `from` to `to`.
#[derive(Clone, Copy)]
struct Side<'a> {
    edge: &'a OrientedPCurveEdge,
    a: f64,
    b: f64,
    from: Point3,
    to: Point3,
}

impl Side<'_> {
    /// Points whose convex hull holds the side's piece of curve, its ends,
    /// and the segments from them to the curve's ends ([`piece_hull`]).
    fn hull(&self) -> Option<Vec<Point3>> {
        let mut hull = piece_hull(&self.edge.curve_3d, self.a, self.b)?;
        hull.extend([self.from, self.to]);
        Some(hull)
    }

    /// The side cut in two at its middle parameter.
    fn halves(&self) -> (Self, Self) {
        let t = f64::midpoint(self.a, self.b);
        let (s3, e3) = curve_ends(self.edge);
        let at = self.edge.curve_3d.evaluate_with_endpoints(t, s3, e3);
        (
            Self {
                b: t,
                to: at,
                ..*self
            },
            Self {
                a: t,
                from: at,
                ..*self
            },
        )
    }
}

/// A loop's sides: a vertex at each step of each edge's curve, and on a
/// spline at each knot as well ([`edge_sides`]). An edge's last side
/// runs on to the next edge's start, which can lie a weld off the edge's
/// end.
fn loop_sides(lp: &[OrientedPCurveEdge]) -> Vec<Side<'_>> {
    let mut sides: Vec<Side> = lp.iter().flat_map(|e| edge_sides(e, 2 * SAMPLES)).collect();
    let n = sides.len();
    for k in 0..n {
        sides[k].to = sides[(k + 1) % n].from;
    }
    sides
}

/// The most times a side is cut in two to read a point's side of a loop.
const CUTS: u32 = 24;

/// `e`'s ends in the way its curve runs.
fn curve_ends(e: &OrientedPCurveEdge) -> (Point3, Point3) {
    if e.forward {
        (e.start_3d, e.end_3d)
    } else {
        (e.end_3d, e.start_3d)
    }
}

/// The sides of `e`'s polygon in the way the loop walks it, from its start
/// to its end: a vertex at each of `steps` even steps of its curve's
/// parameter (one step on a line), and on a spline at each knot as well,
/// so that each side spans part of one Bezier piece.
fn edge_sides(e: &OrientedPCurveEdge, steps: u32) -> Vec<Side<'_>> {
    use std::f64::consts::TAU;
    let curve = &e.curve_3d;
    let steps = if matches!(curve, EdgeCurve::Line) {
        1
    } else {
        steps
    };
    let fraction = |k: u32| f64::from(k) / f64::from(steps);
    let closed = (e.start_3d - e.end_3d).length() < CLOSED;
    let (s3, e3) = curve_ends(e);
    let turn = if e.forward { TAU } else { -TAU };
    // The curve's parameters in the way the loop walks it, as
    // `walk_edge_in` takes them.
    let ts: Vec<f64> = match curve {
        EdgeCurve::Circle(c) if closed => {
            let a0 = c.project(e.start_3d);
            (0..=steps).map(|k| turn.madd(fraction(k), a0)).collect()
        }
        EdgeCurve::Ellipse(el) if closed => {
            let a0 = el.project(e.start_3d);
            (0..=steps).map(|k| turn.madd(fraction(k), a0)).collect()
        }
        _ => {
            let (t0, t1) = curve.domain_with_endpoints(s3, e3);
            let mut ts: Vec<f64> = (0..=steps)
                .map(|k| (t1 - t0).madd(fraction(k), t0))
                .collect();
            if let EdgeCurve::NurbsCurve(n) = curve {
                let (lo, hi) = (t0.min(t1), t0.max(t1));
                ts.extend(n.knots().iter().filter(|&&k| lo < k && k < hi));
                ts.sort_by(f64::total_cmp);
                ts.dedup();
                if t1 < t0 {
                    ts.reverse();
                }
            }
            let backward = if closed {
                !e.forward
            } else {
                let at_t0 = curve.evaluate_with_endpoints(t0, s3, e3);
                (at_t0 - e.start_3d).length() > (at_t0 - e.end_3d).length()
            };
            if backward {
                ts.reverse();
            }
            ts
        }
    };
    let last = ts.len() - 1;
    let point = |k: usize| match k {
        0 => e.start_3d,
        k if k == last => e.end_3d,
        k => curve.evaluate_with_endpoints(ts[k], s3, e3),
    };
    (0..last)
        .map(|k| Side {
            edge: e,
            a: ts[k],
            b: ts[k + 1],
            from: point(k),
            to: point(k + 1),
        })
        .collect()
}

/// Points whose convex hull holds `curve`'s piece from parameter `a` to
/// `b`. A spline's piece, within one knot span, lies in the hull of its
/// Bezier points where their weights are positive. A conic's piece is the
/// image of an arc of the unit circle, which lies within `1 - cos(h)` of
/// its chord toward its middle, `h` half its span: the hull is the chord,
/// and the chord moved that far toward the middle. `None` for a spline's
/// piece with a weight that is not positive, or a conic's piece over half
/// a turn.
fn piece_hull(curve: &EdgeCurve, a: f64, b: f64) -> Option<Vec<Point3>> {
    let conic = |at: &dyn Fn(f64) -> Point3, center: Point3| {
        let half = 0.5 * (b - a).abs();
        if half > std::f64::consts::FRAC_PI_2 {
            return None;
        }
        let rise = (at(f64::midpoint(a, b)) - center) * (1.0 - half.cos());
        let (p, q) = (at(a), at(b));
        Some(vec![p, q, p + rise, q + rise])
    };
    match curve {
        EdgeCurve::Line => Some(Vec::new()),
        EdgeCurve::Circle(c) => conic(&|t| c.evaluate(t), c.center()),
        EdgeCurve::Ellipse(el) => conic(&|t| el.evaluate(t), el.center()),
        EdgeCurve::NurbsCurve(n) => curve_bezier_between(n, a, b)
            .ok()?
            .iter()
            .map(|&[x, y, z, w]| (w > 0.0).then(|| Point3::new(x / w, y / w, z / w)))
            .collect(),
    }
}

/// How far `p` lies from the segment from `a` to `b`.
fn segment_distance(p: Point3, a: Point3, b: Point3) -> f64 {
    let ab = b - a;
    let len2 = ab.dot(ab);
    let t = if len2 > 0.0 {
        ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p - (a + ab * t)).length()
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

/// Whether two loop edges are one half of an edge of the face: one edge,
/// walked the same way.
fn same_half(a: &OrientedPCurveEdge, b: &OrientedPCurveEdge, near: f64) -> bool {
    (a.start_3d - b.start_3d).length() <= near
        && (a.end_3d - b.end_3d).length() <= near
        && a.forward == b.forward
        && (edge_middle(a) - edge_middle(b)).length() <= near
}

/// A piece: its outline and its holes.
pub(super) type Piece = (Vec<OrientedPCurveEdge>, Vec<Vec<OrientedPCurveEdge>>);

/// The pieces of a ball face round the pole `view` holds, whose outline is
/// `edges`' first `n_boundary` and whose section halves follow: read from
/// the walk's loops `walked` ([`pieces`]), or where they do not read, from
/// the face trace of `edges`. The walk can tour the regions about a
/// crossing of sections as one loop; the trace's successor is a function of
/// the graph, and reads them apart. The trace welds its nodes in `(u, v)`
/// by `tol`, and drops a loop as bounding no area by the same reading in
/// the plane as [`pieces`]. It keeps each region on its left in `(u, v)`:
/// where the face lies on its outline's right there, each of its loops is
/// turned round.
pub(super) fn read(
    view: &PoleView,
    edges: &[OrientedPCurveEdge],
    n_boundary: usize,
    walked: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let (boundary, sections) = edges.split_at(n_boundary);
    pieces(view, boundary, sections, walked, tol).or_else(|| {
        let left = view.face_left_in_uv(boundary)?;
        let flat = |lp: &[OrientedPCurveEdge]| view.flat(lp, tol);
        let traced = build_wire_loops_dcel_flat(edges, tol, true, false, Some(left), &flat);
        let traced = if left {
            traced
        } else {
            traced.iter().map(|lp| reverse_loop(lp)).collect()
        };
        pieces(view, boundary, sections, traced, tol)
    })
}

/// The pieces that `loops` bound on a ball face whose outline is the run of
/// edges `boundary`, cut by the section halves `sections` (each section run
/// both ways), seen in `view`.
///
/// The loops must be closed, pass no vertex twice, and between them run
/// each piece of the outline and each section half once: a section left
/// out would leave the face standing whole across it. The edges of the
/// face's arrangement cross nowhere but at its vertices, so each loop is a
/// simple closed curve in the plane. A loop that winds as the outline does
/// is a piece's outline, and one that winds against it a hole. Each hole
/// goes to the smallest outline that holds the middle of its first edge,
/// of those that do not run along that edge (a hole's twin, the outline of
/// the piece inside it).
///
/// That the pieces tile the face is then proved, not sampled. Each piece's
/// mark (one inside its outline, less one inside each of its holes) sums
/// over the pieces to the windings of all the loops: each section runs once
/// each way and cancels, so the sum is the outline's winding, one inside
/// the face and none outside. The pieces tile the face where each piece's
/// mark is one or none everywhere: where each hole lies inside its outline
/// and no two holes of a piece overlap ([`tiled`]).
///
/// `None` when a loop does not map, is not closed, bounds no area, runs an
/// edge both ways or passes a vertex twice (two regions toured as one,
/// touching there), the loops do not run each half once, a hole finds no
/// outline, or the tiling is not proved.
pub(super) fn pieces(
    view: &PoleView,
    boundary: &[OrientedPCurveEdge],
    sections: &[OrientedPCurveEdge],
    loops: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let face_area = 0.5 * view.twice_area(boundary)?;
    if face_area.abs() <= 1e-9 || sections.is_empty() {
        return None;
    }
    let near = 100.0 * tol;
    let halves: Vec<&OrientedPCurveEdge> = boundary.iter().chain(sections).collect();
    let mut free = vec![true; halves.len()];
    for e in loops.iter().flatten() {
        let k = (0..halves.len()).find(|&k| free[k] && same_half(halves[k], e, near))?;
        free[k] = false;
    }
    if free.contains(&true) {
        return None;
    }
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, Vec<Point2>, f64)> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for lp in loops {
        let closed = lp
            .iter()
            .zip(lp.iter().cycle().skip(1))
            .all(|(a, b)| (a.end_3d - b.start_3d).length() <= near);
        let both_ways = lp
            .iter()
            .enumerate()
            .any(|(i, a)| lp[i + 1..].iter().any(|b| same_edge(a, b, near)));
        let pinched = lp.iter().enumerate().any(|(i, a)| {
            lp[i + 1..]
                .iter()
                .any(|b| (a.start_3d - b.start_3d).length() <= near)
        });
        if !closed || both_ways || pinched {
            return None;
        }
        let poly = view.loop_polygon(&lp)?;
        let area = 0.5 * view.twice_area(&lp)? * face_area.signum();
        if view.flat_reading(area, &poly, tol) {
            return None;
        }
        if area > 0.0 {
            outers.push((lp, poly, area));
        } else {
            holes.push(lp);
        }
    }
    let mut pieces: Vec<Piece> = outers
        .iter()
        .map(|(lp, _, _)| (lp.clone(), Vec::new()))
        .collect();
    for hole in holes {
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
        pieces[owner].1.push(hole);
    }
    tiled(view, &pieces, near)?;
    Some(pieces)
}

/// Whether each piece's mark is one or none everywhere: each hole lies
/// inside its outline, and no two holes of a piece overlap. Edges of the
/// arrangement cross nowhere but at vertices, so each edge of one loop lies
/// all inside or all outside another loop that does not run it, and its
/// middle tells which ([`PoleView::inside`]). A hole lies inside its
/// outline where it shares no edge with it and each of its edges does; two
/// holes are apart where they share no edge and each edge of either lies
/// outside the other. `None` where one of these fails or a middle lies too
/// near a loop to read.
fn tiled(view: &PoleView, pieces: &[Piece], near: f64) -> Option<()> {
    let off = |e: &OrientedPCurveEdge, lp: &[OrientedPCurveEdge]| {
        !lp.iter().any(|f| same_edge(e, f, near))
    };
    for (outline, holes) in pieces {
        let read = loop_sides(outline);
        let hole_reads: Vec<Vec<Side>> = holes.iter().map(|h| loop_sides(h)).collect();
        for (i, hole) in holes.iter().enumerate() {
            for e in hole {
                if !(off(e, outline) && view.inside(edge_middle(e), &read)?) {
                    return None;
                }
            }
            for (j, other) in holes.iter().enumerate().skip(i + 1) {
                for (a, b, b_read) in [(hole, other, &hole_reads[j]), (other, hole, &hole_reads[i])]
                {
                    for e in a {
                        if !off(e, b) || view.inside(edge_middle(e), b_read)? {
                            return None;
                        }
                    }
                }
            }
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::curves2d::{Curve2D, Line2D};
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::vec::Vec2;

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

    #[test]
    fn a_thin_crescent_on_a_large_ball_is_read_by_the_face_trace() {
        use super::super::super::pcurve_compute::compute_pcurve_on_surface;
        use super::super::conversion::uv_endpoints_from_pcurve;
        use brepkit_topology::face::FaceSurface;
        use std::f64::consts::{PI, TAU};

        // A ball of radius 1e4 less its cap above z = 0.95 r, charted about
        // z. Two arcs from p to q, 20 apart on its southern half clear of
        // the chart's seam and poles, bulge 2 and 2 less 2e-3 the same way:
        // the crescent between them is 2e-3 wide at most. That is far over
        // the tolerance of 1e-7, but under it times the crescent's
        // perimeter in `(u, v)`, the band the face trace once read it by.
        let r = 1e4;
        let (view, _) = capped_ball(r);
        let chart =
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), r).unwrap());
        // The rim runs its circle's own way, toward -u.
        let rim = small_circle(r, Vec3::new(0.0, 0.0, -1.0), PI - 0.95_f64.acos());
        let at = rim.evaluate(0.0);
        let (u0, v0) = chart.project_point(at).unwrap();
        let mut outline = edge(&EdgeCurve::Circle(rim), at, at, true);
        outline.start_uv = Point2::new(u0, v0);
        outline.end_uv = Point2::new(u0 - TAU, v0);
        outline.pcurve =
            Curve2D::Line(Line2D::new(outline.start_uv, Vec2::new(-TAU, 0.0)).unwrap());
        // `p` and `q` half a chord `h` either side of `m` along `t`; each
        // arc's circle has its middle on the bisector through `m` along `b`.
        let (lat_s, lat_c) = (-0.8_f64).sin_cos();
        let (lon_s, lon_c) = 1.5_f64.sin_cos();
        let m = Vec3::new(lat_c * lon_c, lat_c * lon_s, lat_s);
        let t = Vec3::new(-lon_s, lon_c, 0.0);
        let b = m.cross(t);
        let h = 1e-3_f64;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let (p, q) = (
            origin + (m * h.cos() + t * h.sin()) * r,
            origin + (m * h.cos() - t * h.sin()) * r,
        );
        // The arc from `p` to `q` that bulges `s` toward -b: its circle's
        // middle lies `rho - s` along +b, `rho` from `p`.
        let arc = |s: f64, k: usize| -> OrientedPCurveEdge {
            let rho = ((1.0 - h.cos() * s.cos()) / (h.cos() * s.sin())).atan();
            let (bs, bc) = (rho - s).sin_cos();
            let curve = EdgeCurve::Circle(small_circle(r, m * bc + b * bs, rho));
            let one = edge(&curve, p, q, true);
            let mut out = if (edge_middle(&one) - origin).dot(b) < 0.0 {
                one
            } else {
                twin(&edge(&curve, q, p, true))
            };
            out.source_edge_idx = Some(k);
            out
        };
        let arcs = [arc(2e-4, 0), arc(2e-4 - 2e-7, 1)];
        // Each half laid out in the chart as the splitter lays a section:
        // its pcurve from its curve's own way, its ends read on it.
        let lay = |e: &OrientedPCurveEdge| {
            let (s3, e3) = if e.forward {
                (e.start_3d, e.end_3d)
            } else {
                (e.end_3d, e.start_3d)
            };
            let pcurve = compute_pcurve_on_surface(&e.curve_3d, s3, e3, &chart, &[], None);
            assert!(matches!(pcurve, Curve2D::Nurbs(_)));
            let (a, c) = uv_endpoints_from_pcurve(&pcurve, s3, e3, &chart, &[]);
            let (start_uv, end_uv) = if e.forward { (a, c) } else { (c, a) };
            OrientedPCurveEdge {
                pcurve,
                start_uv,
                end_uv,
                ..e.clone()
            }
        };
        let mut edges = vec![outline];
        edges.extend(halves(&arcs).iter().map(lay));
        assert_eq!(view.face_left_in_uv(&edges[..1]), Some(true));
        // With no loops from the walk, `pieces` refuses, so the face trace
        // gives the pieces: the face round the crescent, and the crescent.
        assert!(pieces(&view, &edges[..1], &edges[1..], Vec::new(), 1e-7).is_none());
        let split = read(&view, &edges, 1, Vec::new(), 1e-7).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split.iter().map(|p| p.1.len()).sum::<usize>(), 1);
    }

    /// A closed circle edge of the unit ball round the unit direction `d`,
    /// `rho` radians round, walked its own way, with the section `k`.
    fn ring(d: Vec3, rho: f64, k: usize) -> OrientedPCurveEdge {
        let circle = small_circle(1.0, d, rho);
        let at = circle.evaluate(0.0);
        let mut e = edge(&EdgeCurve::Circle(circle), at, at, true);
        e.source_edge_idx = Some(k);
        e
    }

    fn unit(x: f64, y: f64, z: f64) -> Vec3 {
        Vec3::new(x, y, z).normalize().unwrap()
    }

    #[test]
    fn a_hole_outside_its_outline_or_inside_another_hole_is_refused() {
        let (view, rim) = capped_ball(1.0);
        let at = |lp, outline| oriented(&view, &rim, lp, outline);
        // Two discs apart on the face, round the south pole.
        let a = ring(unit(0.3, 0.0, -1.0), 0.1, 0);
        let b = ring(unit(-0.3, 0.0, -1.0), 0.1, 1);
        let [a_in, a_out] = [at(vec![a.clone()], true), at(vec![a.clone()], false)];
        let [b_in, b_out] = [at(vec![b.clone()], true), at(vec![b.clone()], false)];
        let sections = halves(&[a, b]);
        // Read by `pieces`, each disc's hole goes to the face, whatever
        // order the loops come in.
        let loops = vec![
            a_in.clone(),
            b_out.clone(),
            vec![rim.clone()],
            b_in.clone(),
            a_out.clone(),
        ];
        let split = pieces(&view, std::slice::from_ref(&rim), &sections, loops, 1e-7).unwrap();
        assert_eq!(split.len(), 3);
        let face = split
            .iter()
            .find(|p| p.0.len() == 1 && p.0[0].source_edge_idx.is_none());
        assert_eq!(face.unwrap().1.len(), 2);
        assert!(tiled(&view, &split, 1e-5).is_some());
        // Given by hand: the first disc less the second, which lies
        // outside it (marked less one there), the face less the first disc,
        // and the second disc. Each half runs once and the marks sum to one
        // over the face, but the face's piece and the second disc both
        // cover the second disc.
        let outside: Vec<Piece> = vec![
            (a_in.clone(), vec![b_out]),
            (vec![rim.clone()], vec![a_out.clone()]),
            (b_in, Vec::new()),
        ];
        assert!(tiled(&view, &outside, 1e-5).is_none());
        // A disc inside another: the face with both as holes nests them.
        let c = ring(unit(0.3, 0.0, -1.0), 0.03, 2);
        let nested: Vec<Piece> = vec![
            (vec![rim.clone()], vec![a_out, at(vec![c.clone()], false)]),
            (a_in, Vec::new()),
            (at(vec![c], true), Vec::new()),
        ];
        assert!(tiled(&view, &nested, 1e-5).is_none());
    }

    /// A closed quadratic spline in the plane `z = -1`, which the view about
    /// the unit ball's south pole maps onto itself: twelve points round a
    /// circle 0.1 round, with a lobe out to `x = -0.3` between the sixth and
    /// the seventh. The lobe's two points act on four knot spans only, each
    /// a ten-thousandth as long as each of the nine others, so that steps of
    /// a 128th of the domain step over the lobe.
    fn lobed_spline() -> OrientedPCurveEdge {
        use std::f64::consts::TAU;
        let mut points: Vec<Point3> = (0..12)
            .map(|k| {
                let (s, c) = (TAU * f64::from(k) / 12.0).sin_cos();
                Point3::new(0.1 * c, 0.1 * s, -1.0)
            })
            .collect();
        points.splice(
            7..7,
            [
                Point3::new(-0.3, 0.04, -1.0),
                Point3::new(-0.3, -0.09, -1.0),
            ],
        );
        points.push(points[0]);
        let spans: Vec<f64> = (0..points.len() - 2)
            .map(|k| if (5..=8).contains(&k) { 1e-4 } else { 1.0 })
            .collect();
        let total: f64 = spans.iter().sum();
        let mut knots = vec![0.0; 3];
        let mut run = 0.0;
        for span in &spans[..spans.len() - 1] {
            run += span;
            knots.push(run / total);
        }
        knots.extend([1.0; 3]);
        let weights = vec![1.0; points.len()];
        let at = points[0];
        let curve = NurbsCurve::new(2, knots, points, weights).unwrap();
        let mut e = edge(&EdgeCurve::NurbsCurve(curve), at, at, true);
        e.source_edge_idx = Some(0);
        e
    }

    #[test]
    fn a_hole_in_a_spline_lobe_that_the_samples_step_over_is_not_read_apart() {
        let (view, rim) = capped_ball(1.0);
        let spline = lobed_spline();
        let middle = Point3::new(-0.22, -0.03, -1.0);
        let disc = Circle3D::new(middle, Vec3::new(0.0, 0.0, 1.0), 0.01).unwrap();
        let at = disc.evaluate(0.0);
        let mut ring = edge(&EdgeCurve::Circle(disc), at, at, true);
        ring.source_edge_idx = Some(1);

        // The lobe holds the disc: on the spline sampled finely over its
        // domain and over the lobe's spans, the disc's middle lies inside,
        // farther from the spline than the disc's radius.
        let EdgeCurve::NurbsCurve(curve) = &spline.curve_3d else {
            unreachable!()
        };
        let (lobe_from, lobe_to) = (curve.knots()[7], curve.knots()[11]);
        let mut ts: Vec<f64> = (0..=100_000).map(|k| f64::from(k) / 1e5).collect();
        ts.extend((0..=10_000).map(|k| lobe_from + (lobe_to - lobe_from) * f64::from(k) / 1e4));
        ts.sort_by(f64::total_cmp);
        let fine: Vec<Point2> = ts
            .iter()
            .map(|&t| {
                let p = curve.evaluate(t);
                Point2::new(p.x(), p.y())
            })
            .collect();
        let probe = Point2::new(middle.x(), middle.y());
        assert!(point_in_polygon_2d(probe, &fine));
        assert!(distance_to_polygon_boundary(probe, &fine) > 0.03);

        // Read against the spline's loop, the disc's middle lies inside it:
        // the loop's sides meet at each knot, and those near the middle are
        // cut until it lies clear of their bounds.
        let spline_hole = oriented(&view, &rim, vec![spline.clone()], false);
        let read = loop_sides(&spline_hole);
        assert_eq!(view.inside(edge_middle(&ring), &read), Some(true));

        // The spline's 64 samples step over the lobe, so the disc's hole
        // goes to the face beside the spline's hole, which holds it. That
        // must be refused, or the disc's hole go to the spline's disc.
        let at = |lp, outline| oriented(&view, &rim, lp, outline);
        let loops = vec![
            vec![rim.clone()],
            at(vec![spline.clone()], false),
            at(vec![spline.clone()], true),
            at(vec![ring.clone()], false),
            at(vec![ring.clone()], true),
        ];
        let sections = halves(&[spline, ring]);
        let split = pieces(&view, std::slice::from_ref(&rim), &sections, loops, 1e-7);
        assert!(split.is_none_or(|s| s.iter().all(|p| p.1.len() <= 1)));
    }

    #[test]
    fn each_side_bound_holds_the_image_of_its_curve() {
        let (view, rim) = capped_ball(1.0);
        // Steps over the lobe, crossing arcs, and circles near the far pole:
        // the rim round it, and one that passes 0.05 rad from it.
        let mut edges = vec![lobed_spline(), rim, ring(unit(0.25, 0.0, 1.0), 0.3, 1)];
        edges.extend(crossing_arcs(1.0, 0.2, 0.3));
        for e in &edges {
            let (s3, e3) = curve_ends(e);
            let mut sides = loop_sides(std::slice::from_ref(e));
            // Cut each side a few times, to reach the far pole's bounds.
            for _ in 0..4 {
                sides = sides
                    .iter()
                    .flat_map(|side| {
                        let (first, second) = side.halves();
                        [first, second]
                    })
                    .collect();
            }
            for side in &sides {
                let bound = view.side_bound(side).unwrap();
                if !bound.is_finite() {
                    continue;
                }
                let chord = [view.map(side.from).unwrap(), view.map(side.to).unwrap()];
                for k in 0..=200 {
                    let t = side.a + (side.b - side.a) * f64::from(k) / 200.0;
                    let at = e.curve_3d.evaluate_with_endpoints(t, s3, e3);
                    let off = distance_to_polygon_boundary(view.map(at).unwrap(), &chord);
                    assert!(off <= bound, "{off} over {bound}");
                }
            }
        }
    }
}

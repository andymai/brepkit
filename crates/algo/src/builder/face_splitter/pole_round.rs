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
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
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
    #[cfg(test)]
    fn polygon(&self, pts: &[Point3]) -> Option<Vec<Point2>> {
        pts.iter().map(|&p| self.map(p)).collect()
    }

    /// Twice the signed area a run of edges adds to a loop it lies in, read
    /// on [`SAMPLES`] samples of each edge: the shoelace terms of each edge's
    /// samples, its end included. The terms of an edge and of its twin,
    /// walked the other way on the same samples, cancel.
    #[cfg(test)]
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

    /// Whether a loop bounds no area: whether its reading
    /// ([`Self::area_reading`]) proves its area within its perimeter times
    /// `tol`, a length on the ball, which the plane reads over the ball's
    /// radius. A loop the reading leaves unresolved stands; one that does
    /// not map bounds none.
    fn flat(&self, lp: &[OrientedPCurveEdge], tol: f64) -> bool {
        self.area_reading(lp, Some(tol))
            .is_none_or(|r| r.area.abs() + r.error <= r.perimeter * tol / self.radius)
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
        let area = self.proved_area(boundary)?;
        (area.abs() > 5e-10).then_some((area > 0.0) == self.north)
    }

    /// How a side's hull ([`Side::hull`]) lies about the side's middle, in
    /// the ball's own units.
    fn spread(&self, side: &Side, hull: &[Point3]) -> Spread {
        let (a, b) = (side.from, side.to);
        let m = a + (b - a) * 0.5;
        let r = self.radius;
        let most = |f: &dyn Fn(Point3) -> f64| hull.iter().map(|&h| f(h)).fold(0.0, f64::max) / r;
        let d = (m - self.center) * (1.0 / r);
        let w = 1.0 + d.dot(self.pole);
        let rise = most(&|h| (h - m).dot(self.pole).abs());
        Spread {
            middle: d,
            w,
            low: w - rise,
            level: d.dot(self.x).hypot(d.dot(self.y)),
            off: most(&|h| segment_distance(h, a, b)),
            reach: most(&|h| (h - m).length()),
            rise,
        }
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
        // About the held pole `n`: `f(d) = k(w) (d x, d y)` with `w = 1 + d
        // n` and `k(w) = sqrt(2 / w)`, so that `|Df [v]| <= |k'| |(d x, d
        // y)| |v| + k |v|` and `|D2 f [u, u]| <= k'' (n u)^2 |(d x, d y)| + 2
        // |k'| |n u| |u|`. Along a segment from `m`, `w` falls by at most
        // `|n u|`, and `k`, `|k'|` and `k''` rise as `w` falls.
        let s = self.spread(side, &hull);
        if s.low <= 0.0 {
            return Some(f64::INFINITY);
        }
        let slope = k1(s.w).madd(s.level, k(s.w));
        let bend = (k2(s.low) * (s.level + s.reach) * s.rise)
            .madd(s.rise, 2.0 * k1(s.low) * s.rise * s.reach);
        Some(slope.madd(s.off, bend))
    }

    /// Whether the image of a side's piece of curve runs one way along the
    /// segment between the images of the side's ends. The curve runs in a
    /// direction of the cone of [`piece_tangents`] at a point of the hull;
    /// it does where the map's derivative there takes each of the cone's
    /// edges `v` onto a direction with a positive part along the segment.
    /// That part differs from its value at the middle by at most the most
    /// of `|D2 f [u, v]| <= k'' |n u| |n v| |(d x, d y)| + |k'| (|n u| |v| +
    /// |n v| |u|)` ([`Self::side_bound`]). `None` where the side has no
    /// hull or its ends do not map.
    fn runs_along(&self, side: &Side) -> Option<bool> {
        let hull = side.hull()?;
        let s = self.spread(side, &hull);
        let along = self.map(side.to)? - self.map(side.from)?;
        let length = along.length();
        if s.low <= 0.0 || length <= 0.0 {
            return Some(false);
        }
        let along = along * (1.0 / length);
        let level = Vec2::new(s.middle.dot(self.x), s.middle.dot(self.y));
        for v in piece_tangents(side)? {
            let v = v * (1.0 / self.radius);
            let (nv, size) = (v.dot(self.pole).abs(), v.length());
            if size <= 0.0 {
                continue;
            }
            let image = level * (-k1(s.w) * v.dot(self.pole))
                + Vec2::new(v.dot(self.x), v.dot(self.y)) * k(s.w);
            let drift = (k2(s.low) * s.rise * nv).madd(
                s.level + s.reach,
                k1(s.low) * s.rise.madd(size, nv * s.reach),
            );
            if image.dot(along) <= drift {
                return Some(false);
            }
        }
        Some(true)
    }

    /// A bound on the signed area between a side and its piece of curve.
    /// Put `x` along the segment `K` between the images of the side's ends
    /// and `y` across it. The piece's image, with straight joins from `K`'s
    /// ends to the image's ends (`l_a` and `l_b` long), and `K` walked back,
    /// close a curve `C` whose area is `-∮ y dx`. Along `K`, `y` is 0;
    /// elsewhere `|y|` is at most the side's bound `B` ([`Self::side_bound`]).
    /// Where the image runs one way along `K` ([`Self::runs_along`]), `x`
    /// varies along the joins and the image by at most `|K| + 2 (l_a +
    /// l_b)`, so `|area(C)| <= B (|K| + 2 (l_a + l_b))`. Elsewhere the bound
    /// is infinite: within a band about `K`, a curve can wind a point more
    /// than once. `None` where the side has no hull or does not map.
    fn side_area_bound(&self, side: &Side) -> Option<f64> {
        let bound = self.side_bound(side)?;
        if !bound.is_finite() || !self.runs_along(side)? {
            return Some(f64::INFINITY);
        }
        let (s3, e3) = curve_ends(side.edge);
        let curve = &side.edge.curve_3d;
        let (from, to) = (self.map(side.from)?, self.map(side.to)?);
        let join = |t: f64, end: Point2| {
            self.map(curve.evaluate_with_endpoints(t, s3, e3))
                .map(|p| (p - end).length())
        };
        let joins = join(side.a, from)? + join(side.b, to)?;
        Some(bound * 2.0_f64.madd(joins, (to - from).length()))
    }

    /// A loop's image read on a polygon of its sides' ends, the sides cut
    /// in two until the reading proves the area's sign (the area exceeds
    /// twice the error) or, given `tol`, proves that the loop bounds no area
    /// (the area and the error together lie within the perimeter times
    /// `tol`, over the ball's radius). Each round cuts each side whose bound
    /// ([`Self::side_area_bound`]) exceeds the easier of the two targets
    /// over twice the number of sides: where none does, the error is at
    /// most half that target. The reading stops unresolved where a side
    /// would be cut more than [`CUTS`] times, the sides would grow past
    /// [`MOST_SIDES`], or no side is cut. `None` where a side does not map.
    fn area_reading(&self, lp: &[OrientedPCurveEdge], tol: Option<f64>) -> Option<AreaReading> {
        let mut sides: Vec<(Side, u32)> = loop_sides(lp).into_iter().map(|s| (s, 0)).collect();
        loop {
            let images: Vec<Point2> = sides
                .iter()
                .map(|(s, _)| self.map(s.from))
                .collect::<Option<_>>()?;
            let pairs = || images.iter().zip(images.iter().cycle().skip(1));
            let reading = AreaReading {
                area: 0.5
                    * pairs()
                        .map(|(p, q)| p.x() * q.y() - q.x() * p.y())
                        .sum::<f64>(),
                error: sides
                    .iter()
                    .map(|(s, _)| self.side_area_bound(s))
                    .sum::<Option<f64>>()?,
                perimeter: pairs().map(|(p, q)| (*q - *p).length()).sum(),
            };
            let band = tol.map_or(0.0, |t| reading.perimeter * t / self.radius);
            let size = reading.area.abs();
            if size > 2.0 * reading.error || (tol.is_some() && size + reading.error <= band) {
                return Some(reading);
            }
            let target = (0.5 * size).max(band - size);
            let share = target / (2.0 * f64::from(u32::try_from(sides.len()).ok()?));
            let mut next = Vec::with_capacity(sides.len());
            for (side, cuts) in &sides {
                if self.side_area_bound(side)? > share {
                    if *cuts >= CUTS {
                        return Some(reading);
                    }
                    let (first, second) = side.halves();
                    next.extend([(first, cuts + 1), (second, cuts + 1)]);
                } else {
                    next.push((*side, *cuts));
                }
            }
            if next.len() > MOST_SIDES || next.len() == sides.len() {
                return Some(reading);
            }
            sides = next;
        }
    }

    /// The signed area of a loop's image where its reading proves its sign
    /// ([`Self::area_reading`]).
    fn proved_area(&self, lp: &[OrientedPCurveEdge]) -> Option<f64> {
        self.area_reading(lp, None)
            .filter(|r| r.area.abs() > 2.0 * r.error)
            .map(|r| r.area)
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
}

/// `e`'s start, then samples along its curve toward its end, in the way the
/// loop walks it.
#[cfg(test)]
fn walk_edge(e: &OrientedPCurveEdge, out: &mut Vec<Point3>) {
    walk_edge_in(e, SAMPLES, out);
}

/// [`walk_edge`] in `steps` steps.
#[cfg(test)]
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
    // The curve's parameters in the way the loop walks it.
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

/// A loop's image read on a polygon of its sides' ends: the polygon's
/// signed area, a bound on how far the loop's area differs from it (the
/// sum of [`PoleView::side_area_bound`] over the sides), and the polygon's
/// perimeter.
struct AreaReading {
    area: f64,
    error: f64,
    perimeter: f64,
}

/// How a side's hull lies about the side's middle, in the ball's own
/// units: where the middle lies from the ball's centre, `w = 1 + d n` there
/// (`n` the held pole), the least `w` on a segment from it to the hull,
/// how far the middle lies from the axis, and the most that a hull point
/// lies off the side, from the middle, and along the axis from the middle.
struct Spread {
    middle: Vec3,
    w: f64,
    low: f64,
    level: f64,
    off: f64,
    reach: f64,
    rise: f64,
}

/// The equal-area map's stretch at `w` ([`PoleView::side_bound`]):
/// `k(w) = sqrt(2 / w)`, and the sizes of its first and second derivatives.
fn k(w: f64) -> f64 {
    (2.0 / w).sqrt()
}

fn k1(w: f64) -> f64 {
    k(w) / (2.0 * w)
}

fn k2(w: f64) -> f64 {
    3.0 * k(w) / (4.0 * w * w)
}

/// The most sides a loop is cut into to prove its area's sign.
const MOST_SIDES: usize = 1 << 16;

/// Directions whose sums with weights not below zero hold the direction
/// in which a side's piece of curve runs, from parameter `a` to `b`. A
/// rational Bezier piece with positive weights runs along a sum of its
/// points' differences `P_j - P_i` (`i < j`) with weights not below zero,
/// so along a sum of the differences of neighbours. A conic's piece of
/// under half a turn runs between the directions at its ends. `None` where
/// the piece has no hull ([`piece_hull`]).
fn piece_tangents(side: &Side) -> Option<Vec<Vec3>> {
    let (a, b) = (side.a, side.b);
    let way = if b < a { -1.0 } else { 1.0 };
    let conic = |at_a: Vec3, at_b: Vec3| {
        (0.5 * (b - a).abs() <= std::f64::consts::FRAC_PI_2).then(|| vec![at_a * way, at_b * way])
    };
    match &side.edge.curve_3d {
        EdgeCurve::Line => {
            let (s3, e3) = curve_ends(side.edge);
            Some(vec![(e3 - s3) * way])
        }
        EdgeCurve::Circle(c) => conic(c.tangent(a), c.tangent(b)),
        EdgeCurve::Ellipse(el) => conic(el.tangent(a), el.tangent(b)),
        EdgeCurve::NurbsCurve(_) => {
            let points = piece_hull(&side.edge.curve_3d, a, b)?;
            Some(points.windows(2).map(|w| w[1] - w[0]).collect())
        }
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
/// the face trace of `edges`. The loops must run each half of `received`
/// once: the outline, then every section half the face received before
/// any was pruned from `edges`. The walk can tour the regions about a
/// crossing of sections as one loop; the trace's successor is a function of
/// the graph, and reads them apart. The trace welds its nodes in `(u, v)`
/// by `tol`, and drops a loop as bounding no area by the same reading in
/// the plane as [`pieces`]. It keeps each region on its left in `(u, v)`:
/// where the face lies on its outline's right there, each of its loops is
/// turned round.
pub(super) fn read(
    view: &PoleView,
    edges: &[OrientedPCurveEdge],
    received: &[OrientedPCurveEdge],
    n_boundary: usize,
    walked: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let (boundary, sections) = received.split_at(n_boundary);
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
/// is a piece's outline, and one that winds against it a hole: the way
/// each winds is the sign of its area, proved ([`PoleView::proved_area`]).
/// Each hole goes to the smallest outline that holds the middle of its
/// first edge ([`PoleView::inside`]), of those that do not run along that
/// edge (a hole's twin, the outline of the piece inside it).
///
/// That the pieces tile the face is then proved, not sampled. Each piece's
/// mark (one inside its outline, less one inside each of its holes) sums
/// over the pieces to the windings of all the loops, as each loop winds the
/// way its role says: each section runs once
/// each way and cancels, so the sum is the outline's winding, one inside
/// the face and none outside. The pieces tile the face where each piece's
/// mark is one or none everywhere: where each hole lies inside its outline
/// and no two holes of a piece overlap ([`tiled`]).
///
/// `None` when a loop does not map, is not closed, bounds no area, runs an
/// edge both ways or passes a vertex twice (two regions toured as one,
/// touching there), the loops do not run each half once, a loop's way
/// round is not proved, a hole finds no outline, or the tiling is not
/// proved.
pub(super) fn pieces(
    view: &PoleView,
    boundary: &[OrientedPCurveEdge],
    sections: &[OrientedPCurveEdge],
    loops: Vec<Vec<OrientedPCurveEdge>>,
    tol: f64,
) -> Option<Vec<Piece>> {
    let face_area = view.proved_area(boundary)?;
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
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
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
        let reading = view
            .area_reading(&lp, None)
            .filter(|r| r.area.abs() > 2.0 * r.error)?;
        let area = reading.area * face_area.signum();
        if area.abs() <= reading.perimeter * tol / view.radius {
            return None;
        }
        if area > 0.0 {
            outers.push((lp, area));
        } else {
            holes.push(lp);
        }
    }
    let mut pieces: Vec<Piece> = outers
        .iter()
        .map(|(lp, _)| (lp.clone(), Vec::new()))
        .collect();
    let reads: Vec<Vec<Side>> = outers.iter().map(|(lp, _)| loop_sides(lp)).collect();
    for hole in holes {
        let first = hole.first()?;
        let probe = edge_middle(first);
        let owner = outers
            .iter()
            .zip(&reads)
            .enumerate()
            .filter(|(_, ((lp, _), read))| {
                !lp.iter().any(|e| same_edge(e, first, near))
                    && view.inside(probe, read) == Some(true)
            })
            .min_by(|a, b| (a.1).0.1.total_cmp(&(b.1).0.1))
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
        let split = read(&view, &edges, &edges, 1, Vec::new(), 1e-7).unwrap();
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
        // Steps over a lobe, crossing arcs, and circles near the far pole:
        // the rim round it, and one that passes 0.05 rad from it.
        let mut edges = vec![
            lobed_spline(),
            flipped_spline(),
            twin(&flipped_spline()),
            rim,
            ring(unit(0.25, 0.0, 1.0), 0.3, 1),
        ];
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
                // Where the side is read as running one way along its
                // chord, each image lies farther along it than the last.
                // Both read to within the rounding of the points' images.
                let one_way = view.runs_along(side).unwrap();
                let along = chord[1] - chord[0];
                let mut last = f64::NEG_INFINITY;
                for k in 0..=200 {
                    let t = side.a + (side.b - side.a) * f64::from(k) / 200.0;
                    let at = view
                        .map(e.curve_3d.evaluate_with_endpoints(t, s3, e3))
                        .unwrap();
                    let off = distance_to_polygon_boundary(at, &chord);
                    assert!(off <= bound + 1e-15, "{off} over {bound}");
                    let x = (at - chord[0]).dot(along);
                    assert!(!one_way || x >= last - 1e-15, "{x} back from {last}");
                    last = x;
                }
            }
        }
    }

    /// A closed quadratic spline in the plane `z = -1`, which the view about
    /// the unit ball's south pole maps onto itself. It runs clockwise round
    /// a triangle about 0.1 across, from its top to near its top, then
    /// counter-clockwise round a lobe 0.15 round that holds the triangle,
    /// back to its start. The lobe's points act on the last ten knot spans
    /// only, each a ten-thousandth as long as each of the five others, so
    /// that steps of a 128th of the domain step over the lobe and read the
    /// spline as running clockwise.
    fn flipped_spline() -> OrientedPCurveEdge {
        use std::f64::consts::TAU;
        let corners = [(0.0, 0.06), (0.05, -0.03), (-0.05, -0.03)];
        let on = |(ax, ay): (f64, f64), (bx, by): (f64, f64), t: f64| {
            Point3::new(ax + (bx - ax) * t, ay + (by - ay) * t, -1.0)
        };
        let mut points: Vec<Point3> = (0..3)
            .flat_map(|k| {
                let (a, b) = (corners[k], corners[(k + 1) % 3]);
                [on(a, b, 0.0), on(a, b, 0.5)]
            })
            .collect();
        points.push(on(corners[2], corners[0], 0.85));
        let start = points[6].y().atan2(points[6].x());
        points.extend((1..10).map(|j| {
            let (s, c) = (TAU * f64::from(j)).madd(0.1, start).sin_cos();
            Point3::new(0.15 * c, 0.15 * s, -1.0)
        }));
        points.push(points[0]);
        let spans: Vec<f64> = (0..points.len() - 2)
            .map(|k| if k >= 5 { 1e-4 } else { 1.0 })
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

    /// The signed area that the closed spline `e` bounds in the plane `z =
    /// -1`, as it walks it: on its points at steps of 1e-5 of its domain,
    /// and of 1e-8 over its last 2e-4.
    fn fine_area(e: &OrientedPCurveEdge) -> f64 {
        let EdgeCurve::NurbsCurve(curve) = &e.curve_3d else {
            unreachable!()
        };
        let mut ts: Vec<f64> = (0..100_000).map(|k| f64::from(k) / 1e5).collect();
        ts.extend((0..20_000).map(|k| 1.0 - 2e-4 + f64::from(k) * 1e-8));
        ts.sort_by(f64::total_cmp);
        let fine: Vec<Point3> = ts.iter().map(|&t| curve.evaluate(t)).collect();
        let twice: f64 = fine
            .iter()
            .zip(fine.iter().cycle().skip(1))
            .map(|(p, q)| p.x() * q.y() - q.x() * p.y())
            .sum();
        if e.forward { 0.5 * twice } else { -0.5 * twice }
    }

    #[test]
    fn a_loop_whose_steps_wind_the_other_way_keeps_its_way_round() {
        let (view, rim) = capped_ball(1.0);
        let spline = flipped_spline();
        assert!(fine_area(&spline) > 0.04);
        assert!(view.loop_area(std::slice::from_ref(&spline)) < 0.0);
        assert!(
            view.proved_area(std::slice::from_ref(&spline))
                .is_none_or(|a| a > 0.0)
        );

        // The face less the spline's disc, and the disc. Read from the
        // steps, the loop that winds as the face does goes to the face as
        // its hole, and its twin stands as the disc's outline. That must be
        // refused, or the face's hole be the loop that winds against it.
        let face = view.loop_area(std::slice::from_ref(&rim));
        let with_face = if (fine_area(&spline) > 0.0) == (face > 0.0) {
            spline.clone()
        } else {
            twin(&spline)
        };
        let loops = vec![
            vec![rim.clone()],
            vec![with_face.clone()],
            vec![twin(&with_face)],
        ];
        let sections = halves(std::slice::from_ref(&spline));
        let split = pieces(&view, std::slice::from_ref(&rim), &sections, loops, 1e-7);
        assert!(split.is_none_or(|s| s.iter().all(|(outline, holes)| {
            holes
                .iter()
                .flatten()
                .all(|e| e.forward != with_face.forward)
                && (outline[0].source_edge_idx.is_none() || outline[0].forward == with_face.forward)
        })));
    }

    /// A section `e` laid out in the chart of the unit ball about `z`, as
    /// the splitter lays a section: its pcurve from its curve's own way, its
    /// ends read on it.
    fn laid(e: &OrientedPCurveEdge) -> OrientedPCurveEdge {
        use super::super::super::pcurve_compute::compute_pcurve_on_surface;
        use super::super::conversion::uv_endpoints_from_pcurve;
        use brepkit_topology::face::FaceSurface;
        let chart =
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap());
        let (s3, e3) = curve_ends(e);
        let pcurve = compute_pcurve_on_surface(&e.curve_3d, s3, e3, &chart, &[], None);
        let (a, c) = uv_endpoints_from_pcurve(&pcurve, s3, e3, &chart, &[]);
        let (start_uv, end_uv) = if e.forward { (a, c) } else { (c, a) };
        OrientedPCurveEdge {
            pcurve,
            start_uv,
            end_uv,
            ..e.clone()
        }
    }

    /// The capped unit ball's rim laid out in its chart, running its
    /// circle's own way, toward -u.
    fn laid_rim() -> OrientedPCurveEdge {
        use brepkit_topology::face::FaceSurface;
        use std::f64::consts::{PI, TAU};
        let chart =
            FaceSurface::Sphere(SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap());
        let rim = small_circle(1.0, Vec3::new(0.0, 0.0, -1.0), PI - 0.95_f64.acos());
        let at = rim.evaluate(0.0);
        let (u0, v0) = chart.project_point(at).unwrap();
        let mut outline = edge(&EdgeCurve::Circle(rim), at, at, true);
        outline.start_uv = Point2::new(u0, v0);
        outline.end_uv = Point2::new(u0 - TAU, v0);
        outline.pcurve =
            Curve2D::Line(Line2D::new(outline.start_uv, Vec2::new(-TAU, 0.0)).unwrap());
        outline
    }

    /// A small circle of the unit ball 0.3 rad round, as a closed rational
    /// quadratic spline in four arcs: three over the first 1e-6 of its
    /// domain run all but 1e-3 rad of the turn, and the fourth runs the rest
    /// over the remainder. Steps of a 64th of the domain all fall on the
    /// fourth arc.
    fn squeezed_circle() -> OrientedPCurveEdge {
        use std::f64::consts::TAU;
        let circle = small_circle(1.0, unit(0.6, 0.0, -1.0), 0.3);
        let (c, r) = (circle.center(), circle.radius());
        let (u, v) = (circle.u_axis(), circle.v_axis());
        let on = |a: f64| c + u * (r * a.cos()) + v * (r * a.sin());
        let last = 1e-3;
        let big = (TAU - last) / 3.0;
        let ends = [0.0, big, 2.0 * big, TAU - last, TAU];
        let mut points = vec![on(0.0)];
        let mut weights = vec![1.0];
        for w in ends.windows(2) {
            let half = 0.5 * (w[1] - w[0]);
            let apex = on(w[0] + half);
            points.push(c + (apex - c) * (1.0 / half.cos()));
            weights.push(half.cos());
            points.push(on(w[1]));
            weights.push(1.0);
        }
        let knots = vec![
            0.0,
            0.0,
            0.0,
            1e-6 / 3.0,
            1e-6 / 3.0,
            2e-6 / 3.0,
            2e-6 / 3.0,
            1e-6,
            1e-6,
            1.0,
            1.0,
            1.0,
        ];
        let at = points[0];
        let curve = NurbsCurve::new(2, knots, points, weights).unwrap();
        let mut e = edge(&EdgeCurve::NurbsCurve(curve), at, at, true);
        e.source_edge_idx = Some(0);
        e
    }

    #[test]
    fn a_squeezed_spline_circle_stands_in_the_face_trace() {
        let (view, _) = capped_ball(1.0);
        let circle = squeezed_circle();
        // Its 64 steps read no area; it bounds the cap's 2 pi (1 - cos 0.3).
        assert!(view.loop_area(std::slice::from_ref(&circle)).abs() < 1e-9);
        let cap = std::f64::consts::TAU * (1.0 - 0.3_f64.cos());
        let area = view.proved_area(std::slice::from_ref(&circle)).unwrap();
        // The reading proves the area within half its size.
        assert!((area.abs() - cap).abs() < 0.5 * area.abs(), "{area}");
        assert!(!view.flat(std::slice::from_ref(&circle), 1e-7));

        // With no loops from the walk, the face trace reads the face round
        // the circle's disc, and the disc.
        let mut edges = vec![laid_rim()];
        edges.extend(halves(std::slice::from_ref(&circle)).iter().map(laid));
        assert_eq!(view.face_left_in_uv(&edges[..1]), Some(true));
        let split = read(&view, &edges, &edges, 1, Vec::new(), 1e-7).unwrap();
        assert_eq!(split.len(), 2);
        assert_eq!(split.iter().map(|p| p.1.len()).sum::<usize>(), 1);
    }

    #[test]
    fn a_section_half_the_loops_leave_out_refuses_the_read() {
        let (view, rim) = capped_ball(1.0);
        let arcs = crossing_arcs(1.0, 0.2, 0.3);
        let [o1, i1, o2, i2] = &arcs;
        let lone = ring(unit(-0.3, 0.0, -1.0), 0.05, 4);
        let at = |lp, outline| oriented(&view, &rim, lp, outline);
        let partition = vec![
            vec![rim.clone()],
            at(round(o1, o2), false),
            at(round(o1, i2), true),
            at(round(o2, i1), true),
            at(round(i1, i2), true),
        ];
        let mut sections = halves(&arcs);
        let mut edges = vec![rim.clone()];
        edges.extend(sections.iter().cloned());
        assert!(read(&view, &edges, &edges, 1, partition.clone(), 1e-7).is_some());
        // The lone circle came to the face, but no loop runs it.
        sections.extend(halves(std::slice::from_ref(&lone)));
        let mut received = vec![rim];
        received.extend(sections);
        assert!(read(&view, &edges, &received, 1, partition, 1e-7).is_none());
    }
}

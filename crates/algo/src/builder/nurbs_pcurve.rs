//! Pcurves for edges on B-spline faces, each proved to lie within a given
//! distance of its edge.
//!
//! The pcurve runs the edge's own span: at the fraction `s` of its knots'
//! span it gives the (u, v) whose surface point stands off the edge's point
//! at the fraction `s` of the edge's span by no more than the bound, the
//! reading the validator's same-parameter check takes. It is made of cubic
//! pieces, each matching the edge's (u, v) and their rates at its ends, and
//! a piece is halved until a bound on `g(s) = S(f(s)) - C(s)` over it holds,
//! `S` the surface, `f` the piece and `C` the edge.
//!
//! The bound reads the derivatives of the surface over the box of the
//! piece's Bezier points, from the control points of the surface's
//! derivatives there: a B-spline's derivative is a B-spline whose control
//! points are differences of the first one's, and on the surface's domain
//! each of its points lies in the hull of the control points whose basis
//! functions do not vanish there. A rational surface `A / w` is read through
//! `A = w S`: each derivative of `S` is that of `A` less the products the
//! Leibniz rule gives, over the least weight.

use brepkit_math::curves2d::{Curve2D, NurbsCurve2D};
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::edge::EdgeCurve;

/// Even pieces a pcurve starts from, before the edge's own knots: a piece
/// whose bound fails is halved, so a straight run takes few.
const START_PIECES: usize = 2;

/// Most pieces one pcurve may take.
const MOST_PIECES: usize = 4096;

/// The highest derivative a bound reads.
const ORDER: usize = 4;

/// A control point in homogeneous form `[w x, w y, w z, w]`.
type Jet = [f64; 4];

/// A control point of a derivative and the same sums taken over the
/// absolute values of its terms, which bound the rounding of the point.
type Entry = (Jet, Jet);

/// Bounds on `|d^(i + j) S / du^i dv^j|`, by `[i][j]`.
type Table = [[f64; ORDER + 1]; ORDER + 1];

/// A pcurve on `surface` for the edge `curve` from `start` to `end`, on the
/// span `[0, 1]`, whose image is proved within `tolerance` of the edge at
/// the same fraction of both spans, as a reader evaluates the spline, or
/// `None`.
///
/// `None` on a periodic surface, a surface with a weight that is not a
/// positive normal float, an edge of no length, where a point of the edge
/// does not project onto the surface, where the surface's parameters do not
/// move along the edge (a pole), past [`MOST_PIECES`], where a term of the
/// bound underflows, or where the bound does not hold: an edge that lies
/// farther than `tolerance` from the surface gets none, and so does one on
/// a surface whose parameters lie so far from nothing that reading them
/// rounds by more than `tolerance` in space.
#[must_use]
pub fn proved_nurbs_pcurve(
    curve: &EdgeCurve,
    start: Point3,
    end: Point3,
    surface: &NurbsSurface,
    tolerance: f64,
) -> Option<Curve2D> {
    if surface.is_periodic_u()
        || surface.is_periodic_v()
        || !surface
            .weights()
            .iter()
            .flatten()
            .all(|&w| w > 0.0 && w.is_normal())
    {
        return None;
    }
    let run = Run::of(curve, start, end)?;
    let speed = run.bounds((0.0, 1.0), 1)?[1];
    if !(speed.is_finite() && speed > tolerance) {
        return None;
    }
    let allowed = tolerance - run.rounding(speed) - surface_rounding(surface);
    if allowed.is_nan() || allowed <= 0.0 {
        return None;
    }
    let jets = Jets::of(surface);

    // The edge's knots, and even stops not beside one.
    let mut stops: Vec<(f64, bool)> = (0..=START_PIECES)
        .map(|k| (k as f64 / START_PIECES as f64, false))
        .chain(run.knot_stops().into_iter().map(|s| (s, true)))
        .collect();
    stops.sort_by(|a, b| a.0.total_cmp(&b.0));
    stops.dedup_by(|later, earlier| {
        let beside = later.0 - earlier.0 < 1e-12;
        if beside && later.1 {
            *earlier = *later;
        }
        beside
    });
    let mut stops: Vec<f64> = stops.into_iter().map(|(s, _)| s).collect();
    if let Some(first) = stops.first_mut() {
        *first = 0.0;
    }
    if let Some(last) = stops.last_mut() {
        *last = 1.0;
    }

    let newton = (1e-3 * allowed).max(f64::MIN_POSITIVE);
    let first = run.at(0.0)?;
    let found =
        brepkit_math::nurbs::projection::project_point_to_surface(surface, first.0, newton).ok()?;
    let start = jets.node(&run, 0.0, Point2::new(found.u, found.v), newton)?;
    // A surface whose parameters lie far from nothing, beside its own
    // size, reads any spline there to more than the tolerance: the first
    // piece's bound holds at least this.
    let (su, sv) = partials_on(surface, start.uv, jets.spans(start.uv, [false, false]))?;
    let mut at_start: Table = [[0.0; ORDER + 1]; ORDER + 1];
    (at_start[1][0], at_start[0][1]) = (
        length(su.x(), su.y(), su.z()),
        length(sv.x(), sv.y(), sv.z()),
    );
    if read_rounding(&at_start, start.uv.x().abs(), start.uv.y().abs()) > allowed {
        return None;
    }
    let mut done = vec![start];
    // Each pending node is found once, the first time a piece ends there.
    let mut pending: Vec<(f64, Option<Node>)> =
        stops.into_iter().skip(1).rev().map(|s| (s, None)).collect();
    while let Some((s, found)) = pending.pop() {
        let a = done[done.len() - 1];
        let b = match found {
            Some(b) => b,
            None => jets.node(&run, s, a.uv + a.rate_out * (s - a.s), newton)?,
        };
        let err = jets.piece_bound(&run, &a, &b);
        if err.is_finite() && err <= allowed {
            done.push(b);
            continue;
        }
        if s - a.s < 1e-9 || done.len() + pending.len() >= MOST_PIECES {
            return None;
        }
        pending.push((s, Some(b)));
        pending.push((f64::midpoint(a.s, s), None));
    }

    let pieces = done.len() - 1;
    let mut points = Vec::with_capacity(3 * pieces + 1);
    let mut knots = vec![0.0; 4];
    points.push(done[0].uv);
    for (k, w) in done.windows(2).enumerate() {
        let [p1, p2, p3] = jets.inner_points(&w[0], &w[1]);
        points.extend([p1, p2, p3]);
        if k + 1 < pieces {
            knots.extend([w[1].s; 3]);
        }
    }
    knots.extend([1.0; 4]);
    let weights = vec![1.0; points.len()];
    NurbsCurve2D::new(3, knots, points, weights)
        .ok()
        .map(Curve2D::Nurbs)
}

/// How far a point of `surface` its evaluator returns can stand from the
/// exact one, as roundings of the largest coordinate of its control points:
/// a point sums `(p + 1)(q + 1)` weighted control points over the sum of
/// their weights, each read to a few roundings.
fn surface_rounding(surface: &NurbsSurface) -> f64 {
    let (p, q) = (surface.degree_u(), surface.degree_v());
    let roundings = (4 * (p + 1) * (q + 1) + 16 * (p + q + 1) + 8) as f64;
    let size = surface
        .control_points()
        .iter()
        .flatten()
        .fold(0.0_f64, |m, c| m.max(largest(*c)));
    roundings * f64::EPSILON * size
}

fn largest(p: Point3) -> f64 {
    p.x().abs().max(p.y().abs()).max(p.z().abs())
}

/// An edge by the fraction `s` of its span: `C(s)` is its curve at
/// `t0 + s (t1 - t0)`, `(t0, t1)` its span as its endpoints give it.
enum Run<'a> {
    Line(Point3, Point3),
    Circle(&'a brepkit_math::curves::Circle3D, f64, f64),
    Ellipse(&'a brepkit_math::curves::Ellipse3D, f64, f64),
    Spline(&'a NurbsCurve, f64, f64),
}

impl<'a> Run<'a> {
    fn of(curve: &'a EdgeCurve, start: Point3, end: Point3) -> Option<Self> {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        if !(t0.is_finite() && t1.is_finite()) {
            return None;
        }
        Some(match curve {
            EdgeCurve::Line => Self::Line(start, end),
            EdgeCurve::Circle(c) => Self::Circle(c, t0, t1),
            EdgeCurve::Ellipse(e) => Self::Ellipse(e, t0, t1),
            EdgeCurve::NurbsCurve(n) => {
                if !n.weights().iter().all(|&w| w > 0.0 && w.is_normal()) {
                    return None;
                }
                Self::Spline(n, t0, t1)
            }
        })
    }

    fn parameter(t0: f64, t1: f64, s: f64) -> f64 {
        (t1 - t0).madd(s, t0)
    }

    /// A B-spline edge's parameter at `s`, moved onto a knot it lies within
    /// a few roundings of: a node at a knot then reads the curve on either
    /// side of it, and no piece next to it runs across it by a rounding.
    /// The move is within the rounding [`Self::rounding`] counts.
    fn spline_parameter(n: &NurbsCurve, t0: f64, t1: f64, s: f64) -> f64 {
        let t = Self::parameter(t0, t1, s);
        let near = 2.0 * f64::EPSILON * (t0.abs() + t1.abs());
        n.knots()
            .iter()
            .copied()
            .filter(|k| (k - t).abs() <= near)
            .min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs()))
            .unwrap_or(t)
    }

    /// The edge's point at `s` and its rates in `s` coming in from below
    /// `s` and going out above it.
    fn at(&self, s: f64) -> Option<(Point3, Vec3, Vec3)> {
        Some(match self {
            Self::Line(a, b) => (*a + (*b - *a) * s, *b - *a, *b - *a),
            Self::Circle(c, t0, t1) => {
                let t = Self::parameter(*t0, *t1, s);
                let rate = c.tangent(t) * (c.radius() * (t1 - t0));
                (c.evaluate(t), rate, rate)
            }
            Self::Ellipse(e, t0, t1) => {
                let t = Self::parameter(*t0, *t1, s);
                let rate = e.tangent(t) * (t1 - t0);
                (e.evaluate(t), rate, rate)
            }
            Self::Spline(n, t0, t1) => {
                let t = Self::spline_parameter(n, *t0, *t1, s);
                let rising = t1 > t0;
                let rate_in = spline_rate(n, t, rising)? * (t1 - t0);
                let rate_out = spline_rate(n, t, !rising)? * (t1 - t0);
                (n.evaluate(t), rate_in, rate_out)
            }
        })
    }

    /// The order to which the edge is smooth over `[sa, sb]` (see
    /// [`smooth_order`]); a line or a conic is smooth throughout.
    fn smooth_order(&self, (sa, sb): (f64, f64)) -> usize {
        let Self::Spline(n, t0, t1) = self else {
            return ORDER;
        };
        let (ta, tb) = (
            Self::spline_parameter(n, *t0, *t1, sa),
            Self::spline_parameter(n, *t0, *t1, sb),
        );
        smooth_order(n.knots(), n.degree(), ta.min(tb), ta.max(tb))
    }

    /// Bounds on `|d^k C / ds^k|` over `[sa, sb]` for `k` up to `order`,
    /// by `k`; the first is unused.
    fn bounds(&self, (sa, sb): (f64, f64), order: usize) -> Option<[f64; ORDER + 1]> {
        let mut out = [0.0; ORDER + 1];
        match self {
            Self::Line(a, b) => out[1] = (*b - *a).length(),
            Self::Circle(_, t0, t1) | Self::Ellipse(_, t0, t1) => {
                let (a, b) = self.conic_axes()?;
                // `|a cos + b sin|` is at most `sqrt(|a|^2 + |b|^2)`, whatever
                // the lengths of the stored axes and the angle between them.
                let reach = length(a, b, 0.0) * (1.0 + 4.0 * f64::EPSILON);
                for (k, slot) in out.iter_mut().enumerate().skip(1) {
                    *slot = reach * power((t1 - t0).abs(), k);
                }
            }
            Self::Spline(n, t0, t1) => {
                let (ta, tb) = (
                    Self::spline_parameter(n, *t0, *t1, sa),
                    Self::spline_parameter(n, *t0, *t1, sb),
                );
                let found = curve_bounds(n, (ta.min(tb), ta.max(tb)), order)?;
                for (k, slot) in out.iter_mut().enumerate().skip(1).take(order) {
                    *slot = found[k] * power((t1 - t0).abs(), k);
                }
            }
        }
        out.iter().all(|b| b.is_finite()).then_some(out)
    }

    /// The lengths of a circle's or an ellipse's two terms as stored, `c +
    /// a cos t + b sin t`: each axis times its radius or semi-axis, the
    /// axes of whatever length the conic holds. `None` for a line or a
    /// B-spline, or where a length is not finite.
    fn conic_axes(&self) -> Option<(f64, f64)> {
        let (a, b) = match self {
            Self::Circle(c, ..) => (c.u_axis() * c.radius(), c.v_axis() * c.radius()),
            Self::Ellipse(e, ..) => (e.u_axis() * e.semi_major(), e.v_axis() * e.semi_minor()),
            Self::Line(..) | Self::Spline(..) => return None,
        };
        let (a, b) = (length(a.x(), a.y(), a.z()), length(b.x(), b.y(), b.z()));
        (a.is_finite() && b.is_finite()).then_some((a, b))
    }

    /// The fractions in `(0, 1)` at the knots of a B-spline edge, so that no
    /// piece runs across one, where the edge's derivatives can jump.
    fn knot_stops(&self) -> Vec<f64> {
        let Self::Spline(n, t0, t1) = self else {
            return Vec::new();
        };
        let span = t1 - t0;
        n.knots()
            .iter()
            .map(|&k| (k - t0) / span)
            .filter(|s| *s > 1e-9 && *s < 1.0 - 1e-9)
            .collect()
    }

    /// How far an evaluated point of the edge can stand from the exact one:
    /// its own evaluation's roundings of its largest coordinate, and the
    /// rounding of the parameter at `s`, which moves the point by up to the
    /// curve's speed in its parameter, `speed / |t1 - t0|`.
    fn rounding(&self, speed: f64) -> f64 {
        let eps = f64::EPSILON;
        let along = |t0: f64, t1: f64| {
            4.0 * eps * (t0.abs() + t1.abs()) * speed / (t1 - t0).abs().max(f64::MIN_POSITIVE)
        };
        match self {
            Self::Line(a, b) => 8.0 * eps * largest(*a).max(largest(*b)),
            Self::Circle(_, t0, t1) | Self::Ellipse(_, t0, t1) => {
                let centre = match self {
                    Self::Circle(c, ..) => c.center(),
                    Self::Ellipse(e, ..) => e.center(),
                    Self::Line(..) | Self::Spline(..) => return f64::NAN,
                };
                let Some((a, b)) = self.conic_axes() else {
                    return f64::NAN;
                };
                32.0 * eps * (largest(centre) + a + b) + along(*t0, *t1)
            }
            Self::Spline(n, t0, t1) => {
                let size = n
                    .control_points()
                    .iter()
                    .fold(0.0_f64, |m, c| m.max(largest(*c)));
                let roundings = (24 * (n.degree() + 1) + 8) as f64;
                roundings * eps * size + along(*t0, *t1)
            }
        }
    }
}

/// A node of a pcurve: the fraction `s` along the edge, the edge's (u, v)
/// there and their rates in `s` coming in and going out, the edge's rates
/// the same ways (they differ at a knot where a B-spline edge turns), and
/// how far the surface's point there stands off the edge's.
#[derive(Clone, Copy)]
struct Node {
    s: f64,
    uv: Point2,
    rate_in: Vec2,
    rate_out: Vec2,
    rate_3d_in: Vec3,
    rate_3d_out: Vec3,
    gap: f64,
}

/// A surface read for its nodes and its bounds.
struct Jets<'a> {
    surface: &'a NurbsSurface,
    rational: bool,
    domain: [f64; 4],
}

impl<'a> Jets<'a> {
    fn of(surface: &'a NurbsSurface) -> Self {
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        Self {
            surface,
            rational: surface.is_rational(),
            domain: [u0, u1, v0, v1],
        }
    }

    fn clamp(&self, p: Point2) -> Point2 {
        let [u0, u1, v0, v1] = self.domain;
        Point2::new(p.x().clamp(u0, u1), p.y().clamp(v0, v1))
    }

    /// The node at `s`, its (u, v) found by Newton from `guess`, and by a
    /// search of the whole surface where Newton fails. Each of its rates
    /// solves the surface's tangent plane for the edge's rate in the least
    /// squares, on the knot spans the rate leaves or comes in from: across a
    /// knot where the surface turns, its partials differ on either side.
    fn node(&self, run: &Run<'_>, s: f64, guess: Point2, newton: f64) -> Option<Node> {
        use brepkit_math::nurbs::projection::{
            project_point_to_surface, project_point_to_surface_from,
        };
        let (point, rate_3d_in, rate_3d_out) = run.at(s)?;
        let guess = self.clamp(guess);
        let found =
            project_point_to_surface_from(self.surface, point, (guess.x(), guess.y()), newton)
                .or_else(|_| project_point_to_surface(self.surface, point, newton))
                .ok()?;
        let uv = self.snap(self.clamp(Point2::new(found.u, found.v)));
        let q = self.surface.evaluate(uv.x(), uv.y());
        let solve = |rate: Vec3, spans: (usize, usize)| -> Option<Vec2> {
            let (su, sv) = partials_on(self.surface, uv, spans)?;
            let (a, b, c) = (su.dot(su), su.dot(sv), sv.dot(sv));
            let det = a.madd(c, -(b * b));
            if !(det.is_finite() && det > 1e-12 * a * c) {
                return None;
            }
            let (x, y) = (su.dot(rate), sv.dot(rate));
            Some(Vec2::new(
                c.madd(x, -(b * y)) / det,
                a.madd(y, -(b * x)) / det,
            ))
        };
        // A first reading on the spans above the node gives the way each
        // rate runs, and the spans that way give the rate.
        let above = self.spans(uv, [false, false]);
        let rough_in = solve(rate_3d_in, above)?;
        let rough_out = solve(rate_3d_out, above)?;
        let rate_in = solve(
            rate_3d_in,
            self.spans(uv, [rough_in.x() > 0.0, rough_in.y() > 0.0]),
        )?;
        let rate_out = solve(
            rate_3d_out,
            self.spans(uv, [rough_out.x() < 0.0, rough_out.y() < 0.0]),
        )?;
        let node = Node {
            s,
            uv,
            rate_in,
            rate_out,
            rate_3d_in,
            rate_3d_out,
            gap: (q - point).length(),
        };
        [
            uv.x(),
            uv.y(),
            rate_in.x(),
            rate_in.y(),
            rate_out.x(),
            rate_out.y(),
            node.gap,
        ]
        .iter()
        .all(|x| x.is_finite())
        .then_some(node)
    }

    /// `uv` with each parameter that lies within a few roundings of a knot
    /// of the surface moved onto it. An edge made along a knot line, or
    /// with a knot where the surface has one, then has nodes on that line,
    /// so no piece next to it runs across it by a rounding and is read as
    /// smooth only to the order the knot leaves.
    fn snap(&self, uv: Point2) -> Point2 {
        let onto = |x: f64, knots: &[f64], (lo, hi): (f64, f64)| {
            let near = 16.0 * f64::EPSILON * (x.abs() + (hi - lo));
            knots
                .iter()
                .copied()
                .filter(|k| (k - x).abs() <= near)
                .min_by(|a, b| (a - x).abs().total_cmp(&(b - x).abs()))
                .unwrap_or(x)
        };
        let [u0, u1, v0, v1] = self.domain;
        Point2::new(
            onto(uv.x(), self.surface.knots_u(), (u0, u1)),
            onto(uv.y(), self.surface.knots_v(), (v0, v1)),
        )
    }

    /// The knot spans in `u` and `v` that hold `uv`: the one below it where
    /// `below` is set for that parameter and it lies on a knot, else the one
    /// above.
    fn spans(&self, uv: Point2, below: [bool; 2]) -> (usize, usize) {
        let s = self.surface;
        let net = s.control_points();
        let columns = net.first().map_or(0, Vec::len);
        (
            span_at(s.knots_u(), s.degree_u(), net.len(), uv.x(), below[0]),
            span_at(s.knots_v(), s.degree_v(), columns, uv.y(), below[1]),
        )
    }

    /// The three Bezier points after the first of the cubic from `a` to
    /// `b`, its inner two held in the surface's domain.
    fn inner_points(&self, a: &Node, b: &Node) -> [Point2; 3] {
        let third = (b.s - a.s) / 3.0;
        [
            self.clamp(a.uv + a.rate_out * third),
            self.clamp(b.uv - b.rate_in * third),
            b.uv,
        ]
    }

    /// A bound on `|S(f(s)) - C(s)|` over the piece from `a` to `b`, or NaN
    /// where none is found.
    ///
    /// Let `k` be the order to which the surface is smooth over the piece's
    /// box (four inside one knot span each way, else one more than its
    /// continuity across the knots in the box), `H` the cubic Hermite
    /// interpolant of `g` from its ends, and `h` the piece's length. Then
    /// `|H| <= max(|g(a)|, |g(b)|) + 4 h / 27 (|g'(a)| + |g'(b)|)`, and
    /// `|g - H| <= c_k h^k sup |g^(k)|`, `c_4 = 1 / 384` and, from Taylor's
    /// remainder at `a` that `H` leaves, `c_3 = 11 / 27` and `c_2 = 31 / 27`.
    /// For `k = 1` the bound is `max(|g(a)|, |g(b)|) + h sup |g'| / 2`. The
    /// derivatives of `g` follow by Faa di Bruno from the surface's partials
    /// and the cubic's derivatives, read from its Bezier points (its fourth
    /// is nothing), and the edge's.
    fn piece_bound(&self, run: &Run<'_>, a: &Node, b: &Node) -> f64 {
        let h = b.s - a.s;
        let [p1, p2, p3] = self.inner_points(a, b);
        let p0 = a.uv;
        let d = [
            (p1 - p0) * (3.0 / h),
            (p2 - p1) * (3.0 / h),
            (p3 - p2) * (3.0 / h),
        ];
        let e = [(d[1] - d[0]) * (2.0 / h), (d[2] - d[1]) * (2.0 / h)];
        let g = (e[1] - e[0]) * (1.0 / h);
        let f1 = parts(&d);
        let f2 = parts(&e);
        let f3 = parts(&[g]);
        let held = [
            p0.x().min(p1.x()).min(p2.x()).min(p3.x()),
            p0.x().max(p1.x()).max(p2.x()).max(p3.x()),
            p0.y().min(p1.y()).min(p2.y()).min(p3.y()),
            p0.y().max(p1.y()).max(p2.y()).max(p3.y()),
        ];
        let s = self.surface;
        let order = smooth_order(s.knots_u(), s.degree_u(), held[0], held[1])
            .min(smooth_order(s.knots_v(), s.degree_v(), held[2], held[3]))
            .min(run.smooth_order((a.s, b.s)));
        if order == 0 {
            return f64::NAN;
        }
        let Some(m) = surface_bounds(s, self.rational, held, order) else {
            return f64::NAN;
        };
        let Some(c) = run.bounds((a.s, b.s), order) else {
            return f64::NAN;
        };
        let ml = |slots: &[(f64, f64)]| multilinear(&m, slots);
        // A reader evaluates the stored spline, not the exact cubic: the
        // (u, v) it gets stand off the exact ones by the rounding of its
        // sums, a share of the largest Bezier coordinate, and the surface
        // carries that into space by its partials over the box.
        let far_u = [p0, p1, p2, p3]
            .iter()
            .fold(0.0_f64, |m, q| max_or_nan(m, q.x().abs()));
        let far_v = [p0, p1, p2, p3]
            .iter()
            .fold(0.0_f64, |m, q| max_or_nan(m, q.y().abs()));
        let ends = max_or_nan(a.gap, b.gap) + read_rounding(&m, far_u + f1.0, far_v + f1.1);
        if order == 1 {
            return (0.5 * h).madd(ml(&[f1]) + c[1], ends);
        }
        let image = match order {
            2 => ml(&[f1, f1]) + ml(&[f2]),
            3 => 3.0f64.madd(ml(&[f1, f2]), ml(&[f1, f1, f1])) + ml(&[f3]),
            _ => 6.0f64.madd(
                ml(&[f1, f1, f2]),
                4.0f64.madd(ml(&[f1, f3]), 3.0f64.madd(ml(&[f2, f2]), ml(&[f1; 4]))),
            ),
        };
        let rest = match order {
            2 => 31.0 / 27.0 * h * h,
            3 => 11.0 / 27.0 * h * h * h,
            _ => h.powi(4) / 384.0,
        };
        // The cubic's rate at each end, held in the domain, against the
        // edge's: the derivative of `g` there. An end on a knot line at the
        // top of the box reads the surface's partials on the span below it,
        // where the piece lies; across a knot inside the box the partials
        // are continuous, as the order is two or more.
        let off = |n: &Node, f: Vec2, rate: Vec3| -> Option<f64> {
            let below = [
                held[0] < held[1] && n.uv.x() >= held[1],
                held[2] < held[3] && n.uv.y() >= held[3],
            ];
            let (su, sv) = partials_on(s, n.uv, self.spans(n.uv, below))?;
            Some((su * f.x() + sv * f.y() - rate).length())
        };
        let (Some(at_a), Some(at_b)) = (off(a, d[0], a.rate_3d_out), off(b, d[2], b.rate_3d_in))
        else {
            return f64::NAN;
        };
        let rates = at_a + at_b;
        // The partials and rates at the nodes round within a share of the
        // sizes of their terms.
        let (p, q) = (s.degree_u(), s.degree_v());
        let roundings = (16 * (4 * (p + 1) * (q + 1) + 16 * (p + q + 1))) as f64;
        let rate_rounding = roundings
            * f64::EPSILON
            * (2.0 * (m[1][0] * f1.0 + m[0][1] * f1.1)
                + a.rate_3d_out.length()
                + b.rate_3d_in.length());
        (4.0 / 27.0 * h).madd(rates + rate_rounding, rest.madd(image + c[order], ends))
    }
}

/// How far a point of the surface at (u, v) read from a stored cubic
/// spline can stand from the one at the exact (u, v), where the surface's
/// first partials are bounded by `m` and the spline's Bezier points reach
/// `far_u` and `far_v` from nothing: a coordinate of the spline sums four
/// points times basis values, each read to a few roundings, within
/// `ROUNDINGS` roundings of the largest coordinate. The caller adds the
/// spline's rates to the reaches for the rounding of the reader's
/// parameter, which lies in `[0, 1]`.
fn read_rounding(m: &Table, far_u: f64, far_v: f64) -> f64 {
    const ROUNDINGS: f64 = 64.0;
    ROUNDINGS * f64::EPSILON * m[1][0].madd(far_u, m[0][1] * far_v)
}

/// The most `|u|` and `|v|` parts among vectors.
fn parts(ws: &[Vec2]) -> (f64, f64) {
    ws.iter().fold((0.0_f64, 0.0_f64), |(u, v), w| {
        (max_or_nan(u, w.x().abs()), max_or_nan(v, w.y().abs()))
    })
}

/// `x` to the power `k`.
fn power(x: f64, k: usize) -> f64 {
    (0..k).fold(1.0, |acc, _| acc * x)
}

/// The larger of two numbers, NaN where either is.
fn max_or_nan(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// A bound on `|D^k S [w_1, ..., w_k]|`, `k = slots.len()`, each `w` with
/// `|u|` and `|v|` parts at most the slot's pair: the sum over each choice
/// of the slots read along `v` of the bound on that partial times the parts
/// chosen.
fn multilinear(m: &Table, slots: &[(f64, f64)]) -> f64 {
    let k = slots.len();
    let mut total = 0.0;
    for chosen in 0..(1_usize << k) {
        let j = chosen.count_ones() as usize;
        let mut term = m[k - j][j];
        for (slot, &(along_u, along_v)) in slots.iter().enumerate() {
            term *= if chosen >> slot & 1 == 1 {
                along_v
            } else {
                along_u
            };
        }
        total += term;
    }
    total
}

/// The order to which a B-spline of degree `p` on `knots` is smooth over
/// `[a, b]`, at most [`ORDER`]: one more than its continuity across each
/// interior knot strictly inside, `p - m` for a knot of multiplicity `m`,
/// and [`ORDER`] where none lies inside. Nothing where it breaks inside or
/// at an end of `[a, b]`, `a < b`: its value at that knot is the one beyond
/// it.
fn smooth_order(knots: &[f64], p: usize, a: f64, b: f64) -> usize {
    let (lo, hi) = (knots[p], knots[knots.len() - p - 1]);
    let mut order = ORDER;
    let mut i = 0;
    while i < knots.len() {
        let k = knots[i];
        let mut m = 1;
        // The knots do not fall, so the next is the same knot where it is
        // not above this one: `-0.0` and `0.0` are one knot.
        while i + m < knots.len() && knots[i + m] <= k {
            m += 1;
        }
        if k > lo && k < hi {
            if k > a && k < b {
                order = order.min((p + 1).saturating_sub(m));
            } else if a < b && k >= a && k <= b && m > p {
                order = 0;
            }
        }
        i += m;
    }
    order
}

/// The knot span of a B-spline of degree `p` with `count` control points
/// that holds `x`: the one below it where `below` is set and `x` lies on a
/// knot, else the one above.
fn span_at(knots: &[f64], p: usize, count: usize, x: f64, below: bool) -> usize {
    let mut span = brepkit_math::nurbs::basis::find_span(count, p, x, knots);
    if below {
        while span > p && knots[span] >= x {
            span -= 1;
        }
    }
    span
}

/// A surface's first partials at `uv`, read on the knot spans `spans`: the
/// polynomial pieces of those spans, at their ends where `uv` lies on a
/// knot. The control points are taken about one of their own, so the
/// rounding goes with the patch's size, not with where it lies.
fn partials_on(
    surface: &NurbsSurface,
    uv: Point2,
    (su, sv): (usize, usize),
) -> Option<(Vec3, Vec3)> {
    use brepkit_math::nurbs::basis::ders_basis_funs;
    let (p, q) = (surface.degree_u(), surface.degree_v());
    let net = surface.control_points();
    let weights = surface.weights();
    let du = ders_basis_funs(su, uv.x(), p, 1, surface.knots_u());
    let dv = ders_basis_funs(sv, uv.y(), q, 1, surface.knots_v());
    let (first_u, first_v) = (su.checked_sub(p)?, sv.checked_sub(q)?);
    let origin = *net.get(first_u)?.get(first_v)?;
    let scale = weight_scale(
        (0..=p)
            .flat_map(|i| (0..=q).map(move |j| (first_u + i, first_v + j)))
            .filter_map(|(i, j)| weights.get(i)?.get(j).copied()),
    )?;
    let zero = Vec3::new(0.0, 0.0, 0.0);
    let (mut a, mut a_u, mut a_v) = (zero, zero, zero);
    let (mut w, mut w_u, mut w_v) = (0.0, 0.0, 0.0);
    for i in 0..=p {
        for j in 0..=q {
            let point = *net.get(first_u + i)?.get(first_v + j)? - origin;
            let weight = *weights.get(first_u + i)?.get(first_v + j)? * scale;
            let (n_u, n_v) = (du.first()?.get(i)?, dv.first()?.get(j)?);
            let (d_u, d_v) = (du.get(1)?.get(i)?, dv.get(1)?.get(j)?);
            let (b, b_u, b_v) = (n_u * n_v * weight, d_u * n_v * weight, n_u * d_v * weight);
            a += point * b;
            a_u += point * b_u;
            a_v += point * b_v;
            w += b;
            w_u += b_u;
            w_v += b_v;
        }
    }
    if w.is_nan() || w <= 0.0 {
        return None;
    }
    let at = a * (1.0 / w);
    let s_u = (a_u - at * w_u) * (1.0 / w);
    let s_v = (a_v - at * w_v) * (1.0 / w);
    [s_u.x(), s_u.y(), s_u.z(), s_v.x(), s_v.y(), s_v.z()]
        .iter()
        .all(|x| x.is_finite())
        .then_some((s_u, s_v))
}

/// A B-spline curve's derivative at `t`, read on the knot span below `t`
/// where `below` is set and `t` is a knot, else on the span above it.
fn spline_rate(curve: &NurbsCurve, t: f64, below: bool) -> Option<Vec3> {
    use brepkit_math::nurbs::basis::ders_basis_funs;
    let p = curve.degree();
    let knots = curve.knots();
    let points = curve.control_points();
    let weights = curve.weights();
    let span = span_at(knots, p, points.len(), t, below);
    let ders = ders_basis_funs(span, t, p, 1, knots);
    let first = span.checked_sub(p)?;
    let scale = weight_scale(weights.get(first..=span)?.iter().copied())?;
    let (mut a, mut a1) = (Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0));
    let (mut w, mut w1) = (0.0, 0.0);
    for j in 0..=p {
        let i = (span + j).checked_sub(p)?;
        let (point, weight) = (*points.get(i)?, *weights.get(i)? * scale);
        let at = Vec3::new(point.x(), point.y(), point.z());
        let (n0, n1) = (
            ders.first()?.get(j)? * weight,
            ders.get(1)?.get(j)? * weight,
        );
        a += at * n0;
        a1 += at * n1;
        w += n0;
        w1 += n1;
    }
    if w.is_nan() || w <= 0.0 {
        return None;
    }
    let rate = (a1 - a * (w1 / w)) * (1.0 / w);
    [rate.x(), rate.y(), rate.z()]
        .iter()
        .all(|x| x.is_finite())
        .then_some(rate)
}

/// The first and last of the `count` basis functions of degree `p` on
/// `knots` that do not vanish everywhere on `[a, b]`.
fn active(knots: &[f64], p: usize, count: usize, a: f64, b: f64) -> Option<(usize, usize)> {
    let first = (0..count).find(|&i| knots[i + p + 1] >= a)?;
    let last = (0..count).rev().find(|&i| knots[i] <= b)?;
    (first <= last).then_some((first, last))
}

/// The control points of a B-spline's derivative of order `r` from those of
/// order `r - 1`, `net[k]` being control point `first + k`: point `i` is
/// `(p - r + 1) (Q[i + 1] - Q[i]) / (knots[i + p + 1] - knots[i + r])`, and
/// nothing where that span is empty (its basis function vanishes).
/// `None` where a difference or a product underflows: a term that rounds
/// to nothing, or below the least normal float, bounds nothing.
fn difference(
    net: &[Entry],
    knots: &[f64],
    p: usize,
    r: usize,
    first: usize,
) -> Option<Vec<Entry>> {
    net.windows(2)
        .enumerate()
        .map(|(k, w)| {
            let i = first + k;
            let span = knots[i + p + 1] - knots[i + r];
            let f = if span > 0.0 {
                (p - r + 1) as f64 / span
            } else {
                0.0
            };
            let mut value = [0.0; 4];
            let mut size = [0.0; 4];
            for c in 0..4 {
                let step = w[1].0[c] - w[0].0[c];
                value[c] = f * step;
                size[c] = f * (w[1].1[c] + w[0].1[c]);
                if f != 0.0 && step != 0.0 && !all_normal(&[step, value[c]]) {
                    return None;
                }
            }
            Some((value, size))
        })
        .collect()
}

/// Whether each of `terms` is a normal float: none has underflowed.
fn all_normal(terms: &[f64]) -> bool {
    terms.iter().all(|t| t.is_normal())
}

/// The length of `[x, y, z]`, read on its largest part so that no square
/// underflows or overflows: NaN where a part is.
fn length(x: f64, y: f64, z: f64) -> f64 {
    let most = x.abs().max(y.abs()).max(z.abs());
    if most.is_nan() || [x, y, z].iter().any(|t| t.is_nan()) {
        return f64::NAN;
    }
    if most == 0.0 || most.is_infinite() {
        return most;
    }
    let (a, b, c) = (x / most, y / most, z / most);
    most * a.madd(a, b.madd(b, c * c)).sqrt()
}

/// The power of two that brings the largest of `weights` to between one
/// and two: weights scaled by it alike give the same rational curve or
/// surface, and the scaling is exact while no weight leaves the normal
/// floats. `None` where the largest weight is not a positive normal float.
fn weight_scale(weights: impl IntoIterator<Item = f64>) -> Option<f64> {
    let most = weights.into_iter().fold(0.0_f64, f64::max);
    if !(most.is_normal() && most > 0.0) {
        return None;
    }
    let exponent = most.log2().floor();
    #[allow(clippy::cast_possible_truncation)]
    let scale = 2f64.powi(-(exponent as i32));
    scale.is_normal().then_some(scale)
}

/// The largest length of the `[x, y, z]` parts and the largest weight part
/// of a net's points of derivative order `level`, each grown by the
/// rounding its sizes bound: each order takes a difference, a quotient and
/// a product, `3 level + 2` roundings of the sizes with the first point's.
fn net_bound(net: &[Entry], level: usize) -> (f64, f64) {
    let grow = (3 * level + 2) as f64 * f64::EPSILON;
    net.iter()
        .fold((0.0_f64, 0.0_f64), |(m, w), (value, size)| {
            let xyz = length(value[0], value[1], value[2]);
            let xyz_size = length(size[0], size[1], size[2]);
            (
                max_or_nan(m, grow.madd(xyz_size, xyz)),
                max_or_nan(w, grow.madd(size[3], value[3].abs())),
            )
        })
}

/// A net's points in homogeneous form about `centre`, their weights times
/// `scale`, with their sizes. `None` where a product underflows.
fn homogeneous(
    points: &[Point3],
    weights: &[f64],
    centre: Point3,
    scale: f64,
) -> Option<Vec<Entry>> {
    points
        .iter()
        .zip(weights)
        .map(|(&p, &w)| {
            let d = p - centre;
            let w = w * scale;
            let value = [w * d.x(), w * d.y(), w * d.z(), w];
            let fine = [d.x(), d.y(), d.z()]
                .iter()
                .zip(&value)
                .all(|(&part, &product)| part == 0.0 || product.is_normal())
                && w.is_normal();
            fine.then(|| (value, value.map(f64::abs)))
        })
        .collect()
}

/// The derivatives of `S = A / w` from those of `A` and `w` over a region:
/// `a[i][j]` and `w[i][j]` bound `A`'s and `w`'s partials there, `reach`
/// bounds `|S - c|`, `least` the weight from below. By Leibniz,
/// `D^(i,j) S = (D^(i,j) A - sum C(i,a) C(j,b) D^(a,b) w D^(i-a,j-b) S) / w`
/// over `(a, b) != (0, 0)`, `A` taken about `c`.
fn quotient_bounds(a: &Table, w: &Table, reach: f64, least: f64, order: usize) -> Table {
    let choose = |n: usize, k: usize| -> f64 {
        (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64)
    };
    let mut m: Table = [[0.0; ORDER + 1]; ORDER + 1];
    m[0][0] = reach;
    let grow = 1.0 + 8.0 * f64::EPSILON;
    for total in 1..=order {
        for i in 0..=total {
            let j = total - i;
            let mut sum = a[i][j];
            for x in 0..=i {
                for y in 0..=j {
                    if x == 0 && y == 0 {
                        continue;
                    }
                    sum += choose(i, x) * choose(j, y) * w[x][y] * m[i - x][j - y];
                }
            }
            m[i][j] = sum / least * grow;
        }
    }
    m
}

/// Bounds on a surface's partials of total order up to `order` over the box
/// `[u0, u1, v0, v1]`, which must lie in its domain, or `None`.
fn surface_bounds(
    surface: &NurbsSurface,
    rational: bool,
    [u0, u1, v0, v1]: [f64; 4],
    order: usize,
) -> Option<Table> {
    let (p, q) = (surface.degree_u(), surface.degree_v());
    let net = surface.control_points();
    let weights = surface.weights();
    let (rows, columns) = (net.len(), net.first()?.len());
    let (i0, i1) = active(surface.knots_u(), p, rows, u0, u1)?;
    let (j0, j1) = active(surface.knots_v(), q, columns, v0, v1)?;
    let centre = brepkit_math::aabb::Aabb3::try_from_points(
        (i0..=i1).flat_map(|i| (j0..=j1).map(move |j| net[i][j])),
    )?
    .center();
    let scale = weight_scale((i0..=i1).flat_map(|i| (j0..=j1).map(move |j| weights[i][j])))?;
    let mut reach = 0.0_f64;
    let mut least = f64::INFINITY;
    for i in i0..=i1 {
        for j in j0..=j1 {
            let d = net[i][j] - centre;
            reach = max_or_nan(reach, length(d.x(), d.y(), d.z()));
            least = least.min(weights[i][j] * scale);
        }
    }
    if !(least.is_normal() && least > 0.0 && reach.is_finite()) {
        return None;
    }
    reach *= 1.0 + 4.0 * f64::EPSILON;
    // Rows along `u`, each a column of points along `v`.
    let base: Vec<Vec<Entry>> = (i0..=i1)
        .map(|i| homogeneous(&net[i][j0..=j1], &weights[i][j0..=j1], centre, scale))
        .collect::<Option<_>>()?;
    let mut a: Table = [[0.0; ORDER + 1]; ORDER + 1];
    let mut w: Table = [[0.0; ORDER + 1]; ORDER + 1];
    let mut along_u = base;
    for r in 0..=p.min(order) {
        if r > 0 {
            let columns = along_u.first().map_or(0, Vec::len);
            let mut next: Vec<Vec<Entry>> = vec![Vec::new(); along_u.len().saturating_sub(1)];
            for col in 0..columns {
                let column: Vec<Entry> = along_u.iter().map(|row| row[col]).collect();
                for (row, entry) in difference(&column, surface.knots_u(), p, r, i0)?
                    .into_iter()
                    .enumerate()
                {
                    next[row].push(entry);
                }
            }
            along_u = next;
        }
        let mut along_v = along_u.clone();
        for t in 0..=q.min(order - r) {
            if t > 0 {
                along_v = along_v
                    .iter()
                    .map(|row| difference(row, surface.knots_v(), q, t, j0))
                    .collect::<Option<_>>()?;
            }
            let flat: Vec<Entry> = along_v.iter().flatten().copied().collect();
            let (xyz, weight) = net_bound(&flat, r + t);
            a[r][t] = xyz;
            w[r][t] = if rational { weight } else { 0.0 };
        }
    }
    if !rational {
        least = 1.0;
    }
    let m = quotient_bounds(&a, &w, reach, least, order);
    m.iter().flatten().all(|x| x.is_finite()).then_some(m)
}

/// Bounds on a B-spline curve's derivatives in its own parameter up to
/// `order` over `[a, b]`, by order, or `None`.
fn curve_bounds(curve: &NurbsCurve, (a, b): (f64, f64), order: usize) -> Option<[f64; ORDER + 1]> {
    let p = curve.degree();
    let net = curve.control_points();
    let weights = curve.weights();
    let rational = weights.iter().any(|&w| w.to_bits() != 1.0_f64.to_bits());
    let (i0, i1) = active(curve.knots(), p, net.len(), a, b)?;
    let centre = brepkit_math::aabb::Aabb3::try_from_points(net[i0..=i1].iter().copied())?.center();
    let scale = weight_scale(weights[i0..=i1].iter().copied())?;
    let mut reach = 0.0_f64;
    let mut least = f64::INFINITY;
    for i in i0..=i1 {
        let d = net[i] - centre;
        reach = max_or_nan(reach, length(d.x(), d.y(), d.z()));
        least = least.min(weights[i] * scale);
    }
    if !(least.is_normal() && least > 0.0 && reach.is_finite()) {
        return None;
    }
    reach *= 1.0 + 4.0 * f64::EPSILON;
    let mut at: Table = [[0.0; ORDER + 1]; ORDER + 1];
    let mut wt: Table = [[0.0; ORDER + 1]; ORDER + 1];
    let mut level = homogeneous(&net[i0..=i1], &weights[i0..=i1], centre, scale)?;
    for r in 0..=p.min(order) {
        if r > 0 {
            level = difference(&level, curve.knots(), p, r, i0)?;
        }
        let (xyz, weight) = net_bound(&level, r);
        at[r][0] = xyz;
        wt[r][0] = if rational { weight } else { 0.0 };
    }
    if !rational {
        least = 1.0;
    }
    let m = quotient_bounds(&at, &wt, reach, least, order);
    let mut out = [0.0; ORDER + 1];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = m[k][0];
    }
    out.iter().all(|x| x.is_finite()).then_some(out)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    use brepkit_math::curves::Circle3D;

    use brepkit_math::nurbs::basis::{basis_funs, find_span};

    use super::*;

    /// A rigid motion: a turn about `(1, 2, 3)` by `angle`, then a shift.
    #[derive(Clone, Copy)]
    struct Pose {
        angle: f64,
        shift: Vec3,
    }

    impl Pose {
        const POSES: [Self; 3] = [
            Self {
                angle: 0.0,
                shift: Vec3::new(0.0, 0.0, 0.0),
            },
            Self {
                angle: 0.7,
                shift: Vec3::new(30.0, -12.0, 5.0),
            },
            Self {
                angle: 2.3,
                shift: Vec3::new(4.0e4, 1.0e4, -2.0e4),
            },
        ];

        fn turn(self, w: Vec3) -> Vec3 {
            let k = Vec3::new(1.0, 2.0, 3.0).normalize().unwrap();
            let (sin, cos) = self.angle.sin_cos();
            w * cos + k.cross(w) * sin + k * (k.dot(w) * (1.0 - cos))
        }

        fn point(self, p: Point3) -> Point3 {
            let at = self.turn(Vec3::new(p.x(), p.y(), p.z())) + self.shift;
            Point3::new(at.x(), at.y(), at.z())
        }

        fn surface(self, s: &NurbsSurface) -> NurbsSurface {
            NurbsSurface::new(
                s.degree_u(),
                s.degree_v(),
                s.knots_u().to_vec(),
                s.knots_v().to_vec(),
                s.control_points()
                    .iter()
                    .map(|row| row.iter().map(|&p| self.point(p)).collect())
                    .collect(),
                s.weights().to_vec(),
            )
            .unwrap()
        }

        fn curve(self, c: &EdgeCurve) -> EdgeCurve {
            match c {
                EdgeCurve::Line => EdgeCurve::Line,
                EdgeCurve::Circle(c) => EdgeCurve::Circle(
                    Circle3D::with_axes(
                        self.point(c.center()),
                        self.turn(c.normal()),
                        c.radius(),
                        self.turn(c.u_axis()),
                        self.turn(c.v_axis()),
                    )
                    .unwrap(),
                ),
                EdgeCurve::Ellipse(_) => unreachable!(),
                EdgeCurve::NurbsCurve(n) => EdgeCurve::NurbsCurve(
                    NurbsCurve::new(
                        n.degree(),
                        n.knots().to_vec(),
                        n.control_points().iter().map(|&p| self.point(p)).collect(),
                        n.weights().to_vec(),
                    )
                    .unwrap(),
                ),
            }
        }
    }

    fn clamped(spans: usize, degree: usize, lo: f64, hi: f64) -> Vec<f64> {
        let mut knots = vec![lo; degree + 1];
        for k in 1..spans {
            knots.push(lo + (hi - lo) * k as f64 / spans as f64);
        }
        knots.extend(vec![hi; degree + 1]);
        knots
    }

    /// A bicubic sheet over `[-50, 50]` squared with four spans each way,
    /// bent by `bend` along `z`.
    fn sheet(bend: f64) -> NurbsSurface {
        let n = 7;
        let points = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let (x, y) = (
                            -50.0 + 100.0 * i as f64 / (n - 1) as f64,
                            -50.0 + 100.0 * j as f64 / (n - 1) as f64,
                        );
                        Point3::new(x, y, bend * ((x / 30.0).sin() + (y / 20.0).cos()))
                    })
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            3,
            3,
            clamped(4, 3, 0.0, 1.0),
            clamped(4, 3, 0.0, 1.0),
            points,
            vec![vec![1.0; n]; n],
        )
        .unwrap()
    }

    /// The curve of `surface` along `v` at `u`, as a B-spline of its own.
    fn iso_v(surface: &NurbsSurface, u: f64) -> NurbsCurve {
        let p = surface.degree_u();
        let net = surface.control_points();
        let span = find_span(net.len(), p, u, surface.knots_u());
        let basis = basis_funs(span, u, p, surface.knots_u());
        let columns = net[0].len();
        let points = (0..columns)
            .map(|j| {
                let mut at = Vec3::new(0.0, 0.0, 0.0);
                for (k, b) in basis.iter().enumerate() {
                    let q = net[span - p + k][j];
                    at += Vec3::new(q.x(), q.y(), q.z()) * *b;
                }
                Point3::new(at.x(), at.y(), at.z())
            })
            .collect();
        NurbsCurve::new(
            surface.degree_v(),
            surface.knots_v().to_vec(),
            points,
            vec![1.0; columns],
        )
        .unwrap()
    }

    /// The farthest the pcurve's image stands off the edge at the same
    /// fraction of both spans, over dense samples.
    fn stray(
        pcurve: &Curve2D,
        curve: &EdgeCurve,
        start: Point3,
        end: Point3,
        s: &NurbsSurface,
    ) -> f64 {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        (0..=4000)
            .map(|k| {
                let f = f64::from(k) / 4000.0;
                let uv = pcurve.evaluate(f);
                let on = s.evaluate(uv.x(), uv.y());
                let at = curve.evaluate_with_endpoints(t0 + (t1 - t0) * f, start, end);
                (on - at).length()
            })
            .fold(0.0, f64::max)
    }

    fn pieces(pcurve: &Curve2D) -> usize {
        match pcurve {
            Curve2D::Nurbs(n) => (n.control_points().len() - 1) / 3,
            _ => 0,
        }
    }

    /// A curve along `v` of a bent bicubic sheet, a straight line across
    /// it (which runs across its knots and bends in its parameters) and a
    /// whole circle on the flat sheet each get a pcurve whose image lies
    /// within the tolerance of the edge at every sample, in three poses,
    /// one of them 4e4 from the origin.
    #[test]
    fn edges_on_a_bspline_sheet_get_pcurves_within_the_tolerance() {
        let tol = 1e-7;
        let bent = sheet(4.0);
        let flat = sheet(0.0);
        let iso = EdgeCurve::NurbsCurve(iso_v(&bent, 0.3));
        let circle = EdgeCurve::Circle(
            Circle3D::with_axes(
                Point3::new(3.0, -2.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                20.0,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            )
            .unwrap(),
        );
        for pose in Pose::POSES {
            let (bent, flat) = (pose.surface(&bent), pose.surface(&flat));
            let iso = pose.curve(&iso);
            let EdgeCurve::NurbsCurve(n) = &iso else {
                unreachable!()
            };
            let (iso_start, iso_end) = (n.evaluate(0.0), n.evaluate(1.0));
            let line = (
                pose.point(Point3::new(-40.0, -31.0, 0.0)),
                pose.point(Point3::new(37.0, 44.0, 0.0)),
            );
            let circle = pose.curve(&circle);
            let EdgeCurve::Circle(c) = &circle else {
                unreachable!()
            };
            let rim = c.evaluate(0.0);
            for (curve, start, end, surface) in [
                (&iso, iso_start, iso_end, &bent),
                (&EdgeCurve::Line, line.0, line.1, &flat),
                (&circle, rim, rim, &flat),
            ] {
                let pcurve = proved_nurbs_pcurve(curve, start, end, surface, tol)
                    .expect("a pcurve for each edge");
                let far = stray(&pcurve, curve, start, end, surface);
                assert!(far <= tol, "{far:e} with {} pieces", pieces(&pcurve));
            }
        }
    }

    /// An edge that stands 1e-5 off the sheet gets no pcurve, and one that
    /// stands 5e-8 off gets one within the tolerance.
    #[test]
    fn an_edge_off_the_sheet_gets_none() {
        let flat = sheet(0.0);
        for (lift, found) in [(1e-5, false), (5e-8, true)] {
            let (a, b) = (
                Point3::new(-40.0, -31.0, lift),
                Point3::new(37.0, 44.0, lift),
            );
            let pcurve = proved_nurbs_pcurve(&EdgeCurve::Line, a, b, &flat, 1e-7);
            assert_eq!(pcurve.is_some(), found, "lift {lift}");
            if let Some(pcurve) = pcurve {
                assert!(stray(&pcurve, &EdgeCurve::Line, a, b, &flat) <= 1e-7);
            }
        }
    }

    /// On a quarter rod as a rational surface, its rims (circular arcs)
    /// and its straight sides get pcurves within the tolerance.
    #[test]
    fn a_rational_quarter_rod_gets_pcurves_on_its_rims_and_sides() {
        let r = 12.0;
        let points = vec![
            vec![Point3::new(r, 0.0, 0.0), Point3::new(r, 0.0, 30.0)],
            vec![Point3::new(r, r, 0.0), Point3::new(r, r, 30.0)],
            vec![Point3::new(0.0, r, 0.0), Point3::new(0.0, r, 30.0)],
        ];
        let rod = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            points,
            vec![vec![1.0, 1.0], vec![FRAC_1_SQRT_2; 2], vec![1.0, 1.0]],
        )
        .unwrap();
        let rim = |z: f64| {
            EdgeCurve::Circle(
                Circle3D::with_axes(
                    Point3::new(0.0, 0.0, z),
                    Vec3::new(0.0, 0.0, 1.0),
                    r,
                    Vec3::new(1.0, 0.0, 0.0),
                    Vec3::new(0.0, 1.0, 0.0),
                )
                .unwrap(),
            )
        };
        for pose in Pose::POSES {
            let rod = pose.surface(&rod);
            for z in [0.0, 30.0] {
                let arc = pose.curve(&rim(z));
                let (a, b) = (
                    pose.point(Point3::new(r, 0.0, z)),
                    pose.point(Point3::new(0.0, r, z)),
                );
                let pcurve = proved_nurbs_pcurve(&arc, a, b, &rod, 1e-7).expect("rim");
                assert!(stray(&pcurve, &arc, a, b, &rod) <= 1e-7);
            }
            let (a, b) = (
                pose.point(Point3::new(
                    r * (0.25 * PI).cos(),
                    r * (0.25 * PI).sin(),
                    2.0,
                )),
                pose.point(Point3::new(
                    r * (0.25 * PI).cos(),
                    r * (0.25 * PI).sin(),
                    27.0,
                )),
            );
            let pcurve = proved_nurbs_pcurve(&EdgeCurve::Line, a, b, &rod, 1e-7).expect("side");
            assert!(stray(&pcurve, &EdgeCurve::Line, a, b, &rod) <= 1e-7);
        }
    }

    /// A bicubic sheet over `[-30, 30]` squared, its `u` knot at the middle
    /// taken three times: along that line, `x = 0`, it is only continuous,
    /// and it folds there by `fold` in `z` per unit of `|x|`.
    fn folded(fold: f64) -> NurbsSurface {
        let points = (0..7)
            .map(|i| {
                (0..7)
                    .map(|j| {
                        let (x, y) = (-30.0 + 10.0 * f64::from(i), -30.0 + 10.0 * f64::from(j));
                        Point3::new(x, y, fold * x.abs() + 3.0 * (y / 15.0).sin())
                    })
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            3,
            3,
            vec![0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0],
            clamped(4, 3, 0.0, 1.0),
            points,
            vec![vec![1.0; 7]; 7],
        )
        .unwrap()
    }

    /// The curve of `surface` along `u` at `v`, as a B-spline of its own.
    fn iso_u(surface: &NurbsSurface, v: f64) -> NurbsCurve {
        let q = surface.degree_v();
        let net = surface.control_points();
        let span = find_span(net[0].len(), q, v, surface.knots_v());
        let basis = basis_funs(span, v, q, surface.knots_v());
        let points = net
            .iter()
            .map(|row| {
                let mut at = Vec3::new(0.0, 0.0, 0.0);
                for (k, b) in basis.iter().enumerate() {
                    let c = row[span - q + k];
                    at += Vec3::new(c.x(), c.y(), c.z()) * *b;
                }
                Point3::new(at.x(), at.y(), at.z())
            })
            .collect();
        NurbsCurve::new(
            surface.degree_u(),
            surface.knots_u().to_vec(),
            points,
            vec![1.0; net.len()],
        )
        .unwrap()
    }

    /// On a sheet that folds along a knot line, the curve along the fold,
    /// a curve across it that folds with it, and its piece that ends on the
    /// fold get pcurves within the tolerance: each piece reads the
    /// surface's partials on its own side of the fold. On the same sheet
    /// unfolded, a straight line across the knot line gets one too.
    #[test]
    fn edges_along_across_and_up_to_a_fold_get_pcurves() {
        let tol = 1e-7;
        for pose in Pose::POSES {
            let sheet = pose.surface(&folded(0.4));
            let along = pose.curve(&EdgeCurve::NurbsCurve(iso_v(&folded(0.4), 0.5)));
            let across = pose.curve(&EdgeCurve::NurbsCurve(iso_u(&folded(0.4), 0.37)));
            let ends = |c: &EdgeCurve, a: f64, b: f64| match c {
                EdgeCurve::NurbsCurve(n) => (n.evaluate(a), n.evaluate(b)),
                _ => unreachable!(),
            };
            for (curve, (start, end)) in [
                (&along, ends(&along, 0.0, 1.0)),
                (&across, ends(&across, 0.0, 1.0)),
                (&across, ends(&across, 0.1, 0.5)),
            ] {
                let pcurve = proved_nurbs_pcurve(curve, start, end, &sheet, tol)
                    .expect("a pcurve for each edge");
                assert!(stray(&pcurve, curve, start, end, &sheet) <= tol);
            }
            // Along `u` the unfolded sheet runs straight: its control points
            // lie evenly along `x` on either side of the knot.
            let flat = pose.surface(&folded(0.0));
            let row = iso_u(&folded(0.0), 0.21);
            let (a, b) = (
                pose.point(row.evaluate(0.1)),
                pose.point(row.evaluate(0.83)),
            );
            let line = EdgeCurve::Line;
            let pcurve = proved_nurbs_pcurve(&line, a, b, &flat, tol).expect("line across");
            assert!(stray(&pcurve, &line, a, b, &flat) <= tol);
        }
    }

    /// Over pieces of many lengths, the bound on a piece is never less than
    /// the farthest its sampled image stands off the edge: on a bent sheet
    /// along a curve across its knots, and on a rational quarter rod.
    #[test]
    fn a_piece_bound_holds_its_sampled_stray() {
        let bent = sheet(4.0);
        let iso = EdgeCurve::NurbsCurve(iso_v(&bent, 0.3));
        let EdgeCurve::NurbsCurve(n) = &iso else {
            unreachable!()
        };
        let (start, end) = (n.evaluate(0.0), n.evaluate(1.0));
        let cases = [
            (
                EdgeCurve::Line,
                Point3::new(-40.0, -31.0, 0.0),
                Point3::new(37.0, 44.0, 0.0),
                sheet(0.0),
            ),
            (iso.clone(), start, end, bent.clone()),
            (
                EdgeCurve::Line,
                Point3::new(-40.0, -31.0, 1.0),
                Point3::new(37.0, 44.0, -3.0),
                bent,
            ),
        ];
        let mut read = 0;
        for (curve, a, b, surface) in &cases {
            let run = Run::of(curve, *a, *b).unwrap();
            let jets = Jets::of(surface);
            let first = run.at(0.0).unwrap().0;
            let found =
                brepkit_math::nurbs::projection::project_point_to_surface(surface, first, 1e-10)
                    .unwrap();
            let start = jets
                .node(&run, 0.0, Point2::new(found.u, found.v), 1e-10)
                .unwrap();
            for &(s0, h) in &[
                (0.0, 1.0),
                (0.0, 0.5),
                (0.1, 0.3),
                (0.37, 0.05),
                (0.6, 0.011),
                (0.2, 0.0017),
            ] {
                let lo = jets.node(&run, s0, start.uv, 1e-10).unwrap();
                let hi = jets
                    .node(&run, s0 + h, lo.uv + lo.rate_out * h, 1e-10)
                    .unwrap();
                let bound = jets.piece_bound(&run, &lo, &hi);
                let [p1, p2, p3] = jets.inner_points(&lo, &hi);
                let mut most: f64 = 0.0;
                for k in 0..=400 {
                    let x = f64::from(k) / 400.0;
                    let y = 1.0 - x;
                    let uv = Point2::new(
                        y * y * y * lo.uv.x()
                            + 3.0 * y * y * x * p1.x()
                            + 3.0 * y * x * x * p2.x()
                            + x * x * x * p3.x(),
                        y * y * y * lo.uv.y()
                            + 3.0 * y * y * x * p1.y()
                            + 3.0 * y * x * x * p2.y()
                            + x * x * x * p3.y(),
                    );
                    let on = surface.evaluate(uv.x(), uv.y());
                    let at = run.at(s0 + h * x).unwrap().0;
                    most = most.max((on - at).length());
                }
                if bound.is_finite() {
                    read += 1;
                    assert!(bound >= most, "bound {bound:e} under {most:e} at {s0} {h}");
                }
            }
        }
        assert!(read > 10, "{read}");
    }

    /// On the flat patch `S(u, v) = (u - 1e12, v, 0)` over `[1e12, 1e12 + 4]`
    /// by `[0, 4]`, a pcurve's (u, v) round by a unit of 1e12 wherever it
    /// is read, which the surface carries to 1e-4 in space. The line across
    /// the patch, whose cubic pieces are exact, gets a pcurve only where
    /// its image, read through the stored spline, lies within the tolerance
    /// of the edge; the same patch on `[0, 4]`, or on `[1e3, 1e3 + 4]`,
    /// gets one.
    #[test]
    fn a_patch_far_out_in_its_parameters_gets_a_pcurve_only_within_the_tolerance() {
        let tol = 1e-7;
        for shift in [0.0, 1.0e3, 1.0e12] {
            let patch = NurbsSurface::new(
                1,
                1,
                vec![shift, shift, shift + 4.0, shift + 4.0],
                vec![0.0, 0.0, 4.0, 4.0],
                vec![
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(0.0, 4.0, 0.0)],
                    vec![Point3::new(4.0, 0.0, 0.0), Point3::new(4.0, 4.0, 0.0)],
                ],
                vec![vec![1.0; 2]; 2],
            )
            .unwrap();
            let (a, b) = (Point3::new(0.5, 2.0, 0.0), Point3::new(3.5, 2.0, 0.0));
            let line = EdgeCurve::Line;
            let pcurve = proved_nurbs_pcurve(&line, a, b, &patch, tol);
            if let Some(pcurve) = &pcurve {
                let far = stray(pcurve, &line, a, b, &patch);
                assert!(far <= tol, "shift {shift}: {far:e}");
            }
            assert_eq!(pcurve.is_some(), shift < 1.0e6, "shift {shift}");
        }
    }

    /// A surface's weights scaled alike, by 1e-200 or by 1e200, leave its
    /// points where they were: the line across the flat sheet, which bends
    /// in its parameters, gets a pcurve within the tolerance from each, as
    /// from the unit weights.
    #[test]
    fn weights_scaled_alike_give_a_pcurve_within_the_tolerance() {
        let tol = 1e-7;
        let flat = sheet(0.0);
        let (a, b) = (Point3::new(-40.0, -31.0, 0.0), Point3::new(37.0, 44.0, 0.0));
        let line = EdgeCurve::Line;
        for scale in [1.0, 1e-200, 1e200] {
            let scaled = NurbsSurface::new(
                3,
                3,
                flat.knots_u().to_vec(),
                flat.knots_v().to_vec(),
                flat.control_points().to_vec(),
                flat.weights()
                    .iter()
                    .map(|row| row.iter().map(|w| w * scale).collect())
                    .collect(),
            )
            .unwrap();
            let pcurve = proved_nurbs_pcurve(&line, a, b, &scaled, tol);
            assert!(pcurve.is_some(), "weights scaled by {scale}: no pcurve");
            let far = stray(&pcurve.unwrap(), &line, a, b, &flat);
            assert!(far <= tol, "weights scaled by {scale}: {far:e}");
        }
    }

    /// Circles stored with radius 1e-8 on axes 1e8 long and with radius
    /// 1e-6 on axes 1e6 long, and an ellipse of semi-axes 0.5 and 0.2 on
    /// axes 40 and 25 long, each in the plane of the flat sheet: the stored
    /// curve is what the edge is, so each gets a pcurve within the
    /// tolerance of it, or none, and the larger two get one.
    #[test]
    fn conics_on_scaled_axes_get_pcurves_within_the_tolerance_of_their_stored_form() {
        use brepkit_math::curves::{Circle3D, Ellipse3D};
        let tol = 1e-7;
        let flat = sheet(0.0);
        let (x, y, z) = (
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        let centre = Point3::new(3.0, -2.0, 0.0);
        let circle = |radius: f64, axes: f64| {
            EdgeCurve::Circle(Circle3D::with_axes(centre, z, radius, x * axes, y * axes).unwrap())
        };
        let ellipse = EdgeCurve::Ellipse(
            Ellipse3D::with_axes(centre, z, 0.5, 0.2, x * 40.0, y * 25.0).unwrap(),
        );
        let mut proved = 0;
        for curve in [&circle(1e-8, 1e8), &circle(1e-6, 1e6), &ellipse] {
            let at = curve.evaluate_with_endpoints(0.3, centre, centre);
            let pcurve = proved_nurbs_pcurve(curve, at, at, &flat, tol);
            if let Some(pcurve) = pcurve {
                proved += 1;
                let far = stray(&pcurve, curve, at, at, &flat);
                assert!(far <= tol, "{curve:?}: {far:e}");
            }
        }
        assert!(proved >= 2, "{proved}");
    }

    /// Knots `-0.0` and `+0.0` are one knot: a cubic's three of them make it
    /// only continuous there, as three plain zeros do. On a sheet written
    /// with the zeros mixed, a line across that knot line, and a curve
    /// across the fold the sheet makes there when it is bent, get the same
    /// pcurves as on the sheet written with plain zeros.
    #[test]
    fn knots_of_either_signed_zero_count_as_one() {
        let mixed = [-1.0, -1.0, -1.0, -1.0, -0.0, -0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let plain = [-1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert_eq!(smooth_order(&plain, 3, -0.5, 0.5), 1);
        assert_eq!(smooth_order(&mixed, 3, -0.5, 0.5), 1);
        let sheet = |knots: &[f64], fold: f64| {
            let points = (0..7)
                .map(|i| {
                    (0..7)
                        .map(|j| {
                            let (x, y) = (-30.0 + 10.0 * f64::from(i), -30.0 + 10.0 * f64::from(j));
                            Point3::new(x, y, fold * x.abs() + 3.0 * (y / 15.0).sin())
                        })
                        .collect()
                })
                .collect();
            NurbsSurface::new(
                3,
                3,
                knots.to_vec(),
                clamped(4, 3, 0.0, 1.0),
                points,
                vec![vec![1.0; 7]; 7],
            )
            .unwrap()
        };
        for fold in [0.0, 0.4] {
            let (with_mixed, with_plain) = (sheet(&mixed, fold), sheet(&plain, fold));
            // Along `u` the unfolded sheet runs straight: its control points
            // lie evenly along `x` on either side of the knot.
            let row = iso_u(&with_plain, 0.21);
            let (a, b) = (row.evaluate(-0.8), row.evaluate(0.66));
            let curve = if fold > 0.0 {
                EdgeCurve::NurbsCurve(row)
            } else {
                EdgeCurve::Line
            };
            let from_plain =
                proved_nurbs_pcurve(&curve, a, b, &with_plain, 1e-7).expect("plain zeros");
            let from_mixed =
                proved_nurbs_pcurve(&curve, a, b, &with_mixed, 1e-7).expect("mixed zeros");
            assert!(
                stray(&from_mixed, &curve, a, b, &with_plain) <= 1e-7,
                "fold {fold}"
            );
            let (Curve2D::Nurbs(p), Curve2D::Nurbs(m)) = (&from_plain, &from_mixed) else {
                unreachable!()
            };
            assert_eq!(
                p.control_points().len(),
                m.control_points().len(),
                "fold {fold}"
            );
            assert!(
                p.control_points()
                    .iter()
                    .zip(m.control_points())
                    .all(|(x, y)| (*x - *y).length() <= 0.0),
                "fold {fold}"
            );
        }
    }
}

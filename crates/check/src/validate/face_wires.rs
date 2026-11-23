//! Whether a face's wires keep clear of each other: no hole crosses the
//! outline or another hole, lies outside the outline, or lies inside another
//! hole.
//!
//! The edges' own curves are read. Where two wires meet, each is followed a
//! short way to both sides of the meeting point: the wires cross when one
//! leaves the point on the other side of the other wire than it came in on.
//! Wires that meet at a vertex, or touch, come and leave on one side.

use std::f64::consts::TAU;

use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::planar::{PlanarRegion, wire_region_2d};
use brepkit_topology::wire::WireId;

use super::checks::{CheckId, EntityRef, Severity, ValidationIssue};
use crate::CheckError;
use crate::face_loops::{FaceLoops, Misplaced, Turns, wire_turns};

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
    /// The fractions of the edge's way along the wire at which it is read,
    /// from 0 to 1: a NURBS edge's knots are among them.
    fractions: Vec<f64>,
    /// The edge's points at `fractions`.
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
        Ok(Self::spanning(
            edge_id,
            curve,
            ends,
            curve.domain_with_endpoints(ends.0, ends.1),
            forward,
        ))
    }

    /// The edge's stretch over the parameters `span`, run as `forward` says.
    fn spanning(
        edge_id: EdgeId,
        curve: &'a EdgeCurve,
        ends: (Point3, Point3),
        span: (f64, f64),
        forward: bool,
    ) -> Self {
        let mut arc = Self {
            edge: edge_id,
            curve,
            ends,
            span,
            forward,
            fractions: Vec::new(),
            points: Vec::new(),
            sag: 0.0,
            length: 0.0,
            lo: [f64::INFINITY; 3],
            hi: [f64::NEG_INFINITY; 3],
        };
        arc.fractions = arc.breaks();
        arc.points = arc.fractions.iter().map(|&f| arc.at(f)).collect();
        for k in 0..arc.chords() {
            let (a, b) = (arc.points[k], arc.points[k + 1]);
            arc.length += (b - a).length();
            let (f0, f1) = (arc.fractions[k], arc.fractions[k + 1]);
            arc.sag = arc.sag.max(arc.stand_off(f0, f1, a, b));
        }
        for p in &arc.points {
            for (k, c) in [p.x(), p.y(), p.z()].into_iter().enumerate() {
                arc.lo[k] = arc.lo[k].min(c);
                arc.hi[k] = arc.hi[k].max(c);
            }
        }
        arc
    }

    /// The edge in stretches none of which can cross itself, in the wire's
    /// direction: a line, a circle or an ellipse whole, and a B-spline
    /// split until each stretch's control points run one way. A curve
    /// whose control points all step forward along one direction moves
    /// forward along it (its positive weights keep it from crossing a line
    /// more often than its control points do), so it cannot come back to
    /// cross itself.
    fn stretches(self) -> Vec<Self> {
        const DEPTH: u32 = 12;
        let EdgeCurve::NurbsCurve(nurbs) = self.curve else {
            return vec![self];
        };
        let one_way = |points: &[Point3]| {
            let (Some(&first), Some(&last)) = (points.first(), points.last()) else {
                return false;
            };
            let along = last - first;
            points
                .windows(2)
                .all(|pair| (pair[1] - pair[0]).dot(along) > 0.0)
        };
        let whole: Option<Vec<Point3>> = nurbs
            .weights()
            .iter()
            .all(|&w| w > 0.0)
            .then(|| nurbs.control_points().to_vec());
        if whole.is_some_and(|points| one_way(&points)) {
            return vec![self];
        }
        let (t0, t1) = self.span;
        let (lo, hi) = (t0.min(t1), t0.max(t1));
        let mut cuts: Vec<f64> = std::iter::once(lo)
            .chain(nurbs.knots().iter().copied().filter(|&k| k > lo && k < hi))
            .chain(std::iter::once(hi))
            .collect();
        cuts.dedup();
        let bezier = |a: f64, b: f64| -> Option<Vec<Point3>> {
            let hull = curve_bezier_between(nurbs, a, b).ok()?;
            hull.iter()
                .map(|&[x, y, z, w]| (w > 0.0).then(|| Point3::new(x / w, y / w, z / w)))
                .collect()
        };
        let mut spans = Vec::new();
        let mut stack: Vec<(f64, f64, u32)> = cuts
            .windows(2)
            .rev()
            .map(|pair| (pair[0], pair[1], 0))
            .collect();
        while let Some((a, b, depth)) = stack.pop() {
            if depth >= DEPTH || bezier(a, b).is_some_and(|points| one_way(&points)) {
                spans.push((a, b));
            } else {
                let mid = 0.5 * (a + b);
                stack.push((mid, b, depth + 1));
                stack.push((a, mid, depth + 1));
            }
        }
        // `span` runs from `t0`; the wire runs from `t0` when `forward`.
        let rising = t1 >= t0;
        if !rising {
            spans = spans.iter().rev().map(|&(a, b)| (b, a)).collect();
        }
        if !self.forward {
            spans.reverse();
        }
        spans
            .into_iter()
            .map(|span| Self::spanning(self.edge, self.curve, self.ends, span, self.forward))
            .collect()
    }

    /// The fractions at which the edge is read: the two ends of a line, and
    /// `CHORDS` even steps of a circle or an ellipse. A NURBS edge is read
    /// at each knot in its span, and at even steps between them, so each
    /// chord keeps to one knot span.
    fn breaks(&self) -> Vec<f64> {
        let even = |n: usize| (0..=n).map(|k| fraction(k, n)).collect::<Vec<_>>();
        match self.curve {
            EdgeCurve::Line => even(1),
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => even(CHORDS),
            EdgeCurve::NurbsCurve(nurbs) => {
                let (t0, t1) = self.span;
                let width = t1 - t0;
                if width.abs() <= 0.0 {
                    return even(CHORDS);
                }
                let (lo, hi) = (t0.min(t1), t0.max(t1));
                let mut cuts: Vec<f64> = std::iter::once(0.0)
                    .chain(
                        nurbs
                            .knots()
                            .iter()
                            .filter(|&&k| k > lo && k < hi)
                            .map(|&k| (k - t0) / width),
                    )
                    .chain(std::iter::once(1.0))
                    .collect();
                cuts.sort_by(f64::total_cmp);
                cuts.dedup();
                let steps = CHORDS.div_ceil(cuts.len() - 1);
                let mut fractions = vec![0.0];
                for pair in cuts.windows(2) {
                    for k in 1..=steps {
                        fractions.push(pair[0] + (pair[1] - pair[0]) * fraction(k, steps));
                    }
                }
                if !self.forward {
                    fractions = fractions.iter().rev().map(|s| 1.0 - s).collect();
                }
                fractions
            }
        }
    }

    /// A bound on how far the edge stands off its chord from `a`, at the
    /// fraction `fa`, to `b`, at `fb`, two fractions in one knot span. For
    /// a NURBS edge, the farthest control point of the chord's piece of the
    /// curve, which holds the piece when its weights are positive. Else
    /// twice the stand-off at the chord's middle, which a circle's is.
    fn stand_off(&self, fa: f64, fb: f64, a: Point3, b: Point3) -> f64 {
        if let EdgeCurve::NurbsCurve(nurbs) = self.curve {
            let hull = curve_bezier_between(nurbs, self.parameter(fa), self.parameter(fb));
            if let Ok(hull) = hull
                && hull.iter().all(|c| c[3] > 0.0)
            {
                return hull
                    .iter()
                    .map(|&[x, y, z, w]| {
                        let c = Point3::new(x / w, y / w, z / w);
                        (c - (a + (b - a) * along(a, b, c))).length()
                    })
                    .fold(0.0, f64::max);
            }
        }
        if matches!(self.curve, EdgeCurve::Line) {
            return 0.0;
        }
        let on = self.at(0.5 * (fa + fb));
        2.0 * (on - (a + (b - a) * 0.5)).length()
    }

    /// The curve's parameter at the fraction `f` of the edge's way along
    /// the wire.
    fn parameter(&self, f: f64) -> f64 {
        let s = if self.forward { f } else { 1.0 - f };
        self.span.0 + (self.span.1 - self.span.0) * s
    }

    /// The edge's point at the fraction `f` of its way along the wire.
    fn at(&self, f: f64) -> Point3 {
        self.curve
            .evaluate_with_endpoints(self.parameter(f), self.ends.0, self.ends.1)
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

/// An edge's stretch between two fractions, its chord, and a bound on how
/// far the stretch stands off the chord.
#[derive(Clone, Copy)]
struct Piece {
    fractions: (f64, f64),
    ends: (Point3, Point3),
    sag: f64,
}

impl Piece {
    fn of(arc: &Arc<'_>, fractions: (f64, f64), ends: (Point3, Point3)) -> Self {
        Self {
            fractions,
            ends,
            sag: arc.stand_off(fractions.0, fractions.1, ends.0, ends.1),
        }
    }

    /// The stretch's two halves, or the stretch itself when it is its chord.
    fn halves(self, arc: &Arc<'_>) -> Vec<Self> {
        if self.sag <= 0.0 {
            return vec![self];
        }
        let f = 0.5 * (self.fractions.0 + self.fractions.1);
        let p = arc.at(f);
        vec![
            Self::of(arc, (self.fractions.0, f), (self.ends.0, p)),
            Self::of(arc, (f, self.fractions.1), (p, self.ends.1)),
        ]
    }
}

/// How often a pair of chords is halved to tell meeting points apart.
const HALVINGS: usize = 5;

/// The most meeting points read between two edges: edges that meet more
/// often run along each other.
const MOST_MEETINGS: usize = 64;

/// Adds the points at which the stretches `pa` of `a` and `pb` of `b` come
/// within `tol` of each other. A pair of stretches whose chords come near
/// is halved, so two meeting points close to each other fall to different
/// pairs, each with a search of its own.
fn meet(
    (a, pa): (&Arc<'_>, Piece),
    (b, pb): (&Arc<'_>, Piece),
    halvings: usize,
    tol: f64,
    found: &mut Vec<(f64, f64, Point3)>,
) {
    if found.len() >= MOST_MEETINGS {
        return;
    }
    let (gap, qa, qb) =
        crate::distance::edge::segment_segment_distance(pa.ends.0, pa.ends.1, pb.ends.0, pb.ends.1);
    if gap > pa.sag + pb.sag + tol {
        return;
    }
    if halvings == 0 || pa.sag + pb.sag <= tol {
        let seed = |piece: Piece, q: Point3| {
            piece.fractions.0
                + along(piece.ends.0, piece.ends.1, q) * (piece.fractions.1 - piece.fractions.0)
        };
        let (fa, fb, gap) = nearest(a, b, seed(pa, qa), seed(pb, qb));
        let at = a.at(fa);
        if gap < tol
            && !found
                .iter()
                .any(|&(_, _, known)| (known - at).length() < tol)
        {
            found.push((fa, fb, at));
        }
        return;
    }
    for ha in pa.halves(a) {
        for hb in pb.halves(b) {
            meet((a, ha), (b, hb), halvings - 1, tol, found);
        }
    }
}

/// The fractions at which the two edges come within `tol` of each other.
fn meetings(a: &Arc<'_>, b: &Arc<'_>, tol: f64) -> Vec<(f64, f64)> {
    let mut found = Vec::new();
    if a.clear_of(b, a.sag + b.sag + tol) {
        return Vec::new();
    }
    let piece = |arc: &Arc<'_>, k: usize| {
        Piece::of(
            arc,
            (arc.fractions[k], arc.fractions[k + 1]),
            (arc.points[k], arc.points[k + 1]),
        )
    };
    let pieces_b: Vec<Piece> = (0..b.chords()).map(|j| piece(b, j)).collect();
    for i in 0..a.chords() {
        let pa = piece(a, i);
        for &pb in &pieces_b {
            meet((a, pa), (b, pb), HALVINGS, tol, &mut found);
        }
    }
    found.into_iter().map(|(fa, fb, _)| (fa, fb)).collect()
}

/// The wire's point the distance `d` from `p`, the first met when the wire
/// is followed from the fraction `f` of its edge `index`, in its direction
/// or against it.
fn arm(wire: &[Arc<'_>], index: usize, f: f64, onward: bool, p: Point3, d: f64) -> Option<Point3> {
    let (mut i, mut from) = (index, f);
    for _ in 0..=wire.len() {
        let arc = &wire[i];
        let steps: Vec<f64> = if onward {
            arc.fractions
                .iter()
                .copied()
                .filter(|&g| g > from)
                .collect()
        } else {
            arc.fractions
                .iter()
                .rev()
                .copied()
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
/// leaves at `out`. `None` when it lies within `along` of the path.
fn left_of(out: f64, back: f64, at: f64, along: f64) -> Option<bool> {
    let wedge = (back - out).rem_euclid(TAU);
    let turn = (at - out).rem_euclid(TAU);
    if !(along..=TAU - along).contains(&wedge) {
        None
    } else if turn > along && turn < wedge - along {
        Some(true)
    } else if turn > wedge + along && turn < TAU - along {
        Some(false)
    } else {
        None
    }
}

/// Where an edge of one wire meets an edge of another: each edge's place in
/// its wire, the fraction along it, and the point.
struct Meeting {
    a: (usize, f64),
    b: (usize, f64),
    at: Point3,
}

/// Whether wire `a` crosses wire `b` at `meeting`, read from the wires'
/// points the distance `d` from it.
fn crosses(
    surface: &FaceSurface,
    a: &[Arc<'_>],
    b: &[Arc<'_>],
    meeting: &Meeting,
    d: f64,
    tol: f64,
) -> bool {
    let ((ia, fa), (ib, fb), p) = (meeting.a, meeting.b, meeting.at);
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
    // A point within the tolerance of the other wire is on it, on neither
    // side.
    let along = ALONG.max(4.0 * tol / d);
    matches!(
        (left_of(out, back, before, along), left_of(out, back, after, along)),
        (Some(came), Some(left)) if came != left
    )
}

/// Whether wire `a` crosses wire `b`.
///
/// Each wire is followed from a meeting point a hundredth of the shorter
/// meeting edge, but no farther than a quarter of the way to the nearest
/// other meeting point, so a wire that crosses out and back in close by is
/// read between its two crossings.
fn wires_cross(surface: &FaceSurface, a: &[Arc<'_>], b: &[Arc<'_>], tol: f64) -> bool {
    let mut all: Vec<(Meeting, f64)> = Vec::new();
    for (ia, arc_a) in a.iter().enumerate() {
        if arc_a.length <= tol {
            continue;
        }
        for (ib, arc_b) in b.iter().enumerate() {
            if arc_b.length <= tol || arc_a.edge == arc_b.edge {
                continue;
            }
            let reach = (1e-2 * arc_a.length.min(arc_b.length)).max(100.0 * tol);
            for (fa, fb) in meetings(arc_a, arc_b, tol) {
                let meeting = Meeting {
                    a: (ia, fa),
                    b: (ib, fb),
                    at: arc_b.at(fb),
                };
                all.push((meeting, reach));
            }
        }
    }
    all.iter().any(|(meeting, reach)| {
        let d = all
            .iter()
            .map(|(other, _)| (other.at - meeting.at).length())
            .filter(|&gap| gap > 10.0 * tol)
            .fold(*reach, |d, gap| d.min(0.25 * gap));
        d >= 10.0 * tol && crosses(surface, a, b, meeting, d, tol)
    })
}

/// Whether `wire`, its edges cut into stretches that cannot cross
/// themselves ([`Arc::stretches`]), crosses itself, as [`wires_cross`]
/// reads a wire against itself, with the pairs of stretches it reads found
/// by a sweep. `uses[k]` is the place in the wire of the edge stretch `k`
/// belongs to, so the stretches of one B-spline that loops are read against
/// each other.
///
/// The edges are swept in the order their boxes start along the axis the
/// wire spans most, and an edge is read against the edges whose boxes,
/// grown by the edges' sags and `tol`, it still overlaps on that axis. Each
/// pair is read once. A meeting at the vertex two edges in a row share is
/// the wire running on, which never crosses: it is kept as a neighbour of
/// the other meetings, and its sides are not read.
fn wire_crosses_itself(surface: &FaceSurface, wire: &[Arc<'_>], uses: &[usize], tol: f64) -> bool {
    let count = wire.len();
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for arc in wire {
        for k in 0..3 {
            lo[k] = lo[k].min(arc.lo[k]);
            hi[k] = hi[k].max(arc.hi[k]);
        }
    }
    let axis = (0..3)
        .max_by(|&i, &j| (hi[i] - lo[i]).total_cmp(&(hi[j] - lo[j])))
        .unwrap_or(0);
    let start = |i: usize| wire[i].lo[axis] - wire[i].sag - 0.5 * tol;
    let end = |i: usize| wire[i].hi[axis] + wire[i].sag + 0.5 * tol;
    let mut order: Vec<usize> = (0..count).filter(|&i| wire[i].length > tol).collect();
    order.sort_by(|&i, &j| start(i).total_cmp(&start(j)).then(i.cmp(&j)));
    // Where the wire runs on from edge `i` to edge `j`, the vertex they share.
    let runs_on = |i: usize, j: usize| -> Option<Point3> {
        let (from, to) = (wire[i].at(1.0), wire[j].at(0.0));
        ((i + 1) % count == j && (from - to).length() <= tol).then_some(from)
    };
    let mut all: Vec<(Meeting, f64, bool)> = Vec::new();
    let mut active: Vec<usize> = Vec::new();
    for &j in &order {
        active.retain(|&i| end(i) >= start(j));
        for &i in &active {
            // Two uses of one edge (a seam) lie along each other.
            if wire[i].edge == wire[j].edge && uses[i] != uses[j] {
                continue;
            }
            let (a, b) = (i.min(j), i.max(j));
            let reach = (1e-2 * wire[a].length.min(wire[b].length)).max(100.0 * tol);
            let shared = [runs_on(a, b), runs_on(b, a)];
            for (fa, fb) in meetings(&wire[a], &wire[b], tol) {
                let at = wire[b].at(fb);
                let running_on = shared
                    .iter()
                    .flatten()
                    .any(|&vertex| (vertex - at).length() <= 10.0 * tol);
                all.push((
                    Meeting {
                        a: (a, fa),
                        b: (b, fb),
                        at,
                    },
                    reach,
                    running_on,
                ));
            }
        }
        active.push(j);
    }
    all.iter()
        .filter(|(_, _, running_on)| !running_on)
        .any(|(meeting, reach, _)| {
            let d = all
                .iter()
                .map(|(other, _, _)| (other.at - meeting.at).length())
                .filter(|&gap| gap > 10.0 * tol)
                .fold(*reach, |d, gap| d.min(0.25 * gap));
            let swapped = Meeting {
                a: meeting.b,
                b: meeting.a,
                at: meeting.at,
            };
            d >= 10.0 * tol
                && (crosses(surface, wire, wire, meeting, d, tol)
                    || crosses(surface, wire, wire, &swapped, d, tol))
        })
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

    fn clear_of(&self, other: &Self, reach: f64) -> bool {
        (0..3).any(|k| self.lo[k] > other.hi[k] + reach || other.lo[k] > self.hi[k] + reach)
    }
}

/// The fractions of an edge at which a hole's side of a wire is read.
const PROBES: [f64; 7] = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875];

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
    // Where a hole stands to a wire's region, read at the first of these
    // points of its edges that lies clear of that wire.
    let side = |hole: usize, of: &PlanarRegion| {
        wires[hole]
            .iter()
            .filter(|arc| arc.length > tol)
            .flat_map(|arc| PROBES.iter().map(move |&f| arc.at(f)))
            .find_map(|p| of.side(flat(p), tol))
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
            if hole == other || bounds[hole].clear_of(&bounds[other], tol) {
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
/// Reports a hole that crosses the outline or another hole, or lies outside
/// the outline or inside another hole. Wires that share a vertex or an edge,
/// or touch without crossing, are not reported. `tolerance` is the gap within
/// which two edges meet.
///
/// On a curved face the wires are read in a plane drawn from the surface's
/// parameters ([`FaceLoops`]). A closed surface has no inside and outside
/// of a wire as a plane has: there a hole is reported when the face's other
/// wires do not all lie on one side of it or of the outline, or when the
/// face reaches an open end of its surface.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_face_wires_apart(
    topo: &Topology,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    Ok(face_wires_issue(topo, face_id, tolerance, true)?
        .into_iter()
        .collect())
}

/// Check that a face's holes lie inside its outline and outside each other.
///
/// Reports what [`check_face_wires_apart`] reports for a hole outside the
/// outline or inside another hole. A face whose wires cross is not reported,
/// because crossing wires have no inside and outside to compare.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_face_holes_placed(
    topo: &Topology,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    Ok(face_wires_issue(topo, face_id, tolerance, false)?
        .into_iter()
        .collect())
}

/// Check that no wire of a face on a surface that closes on itself runs
/// round the surface more than once.
///
/// The surface is a cylinder, cone, sphere, torus, or a closed B-spline
/// surface. A wire that runs round more than once (see
/// [`Turns::bound_a_face`]) bounds no region of the surface. Two rims of one
/// wire that run the same way round, as where one of them is used the wrong
/// way along the wire, give one. The edges' own curves are read; a wire that
/// does not read is not reported.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_face_wire_turns(
    topo: &Topology,
    face_id: FaceId,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let Some(turns) = wire_turns(topo, face_id)? else {
        return Ok(vec![]);
    };
    let face = topo.face(face_id)?;
    let wire_ids = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
    for (wire_id, turns) in wire_ids.zip(turns) {
        let Some(turns) = turns.filter(|t| !t.bound_a_face()) else {
            continue;
        };
        let times = match turns {
            Turns::Round(k) => format!("{} times", k.unsigned_abs()),
            Turns::Torus(a, b) => format!(
                "{} and {} times along its two periods",
                a.unsigned_abs(),
                b.unsigned_abs()
            ),
        };
        return Ok(vec![ValidationIssue {
            check: CheckId::FaceWireTurns,
            severity: Severity::Error,
            entity: EntityRef::Face(face_id),
            description: format!(
                "wire {} runs round the surface {times}, so it bounds no region of it",
                wire_id.index()
            ),
            deviation: None,
        }]);
    }
    Ok(vec![])
}

fn face_wires_issue(
    topo: &Topology,
    face_id: FaceId,
    tolerance: f64,
    report_crossing: bool,
) -> Result<Option<ValidationIssue>, CheckError> {
    let face = topo.face(face_id)?;
    if face.inner_wires().is_empty() {
        return Ok(None);
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
            if wires_cross(face.surface(), a, b, tolerance) {
                let name = if j == 0 {
                    "the outer wire"
                } else {
                    "another inner wire"
                };
                return Ok(report_crossing.then(|| {
                    issue(
                        face_id,
                        format!("inner wire {} crosses {name}", wire_ids[i].index()),
                    )
                }));
            }
        }
    }
    if let FaceSurface::Plane { normal, .. } = face.surface() {
        if let Some(description) =
            misplaced_hole(topo, &wire_ids, &wires, &bounds, *normal, tolerance)?
        {
            return Ok(Some(issue(face_id, description)));
        }
    } else if let Some(misplaced) =
        FaceLoops::of(topo, face_id, tolerance)?.and_then(|loops| loops.misplaced_hole())
    {
        let description = match misplaced {
            Misplaced::Outside(hole) => format!(
                "inner wire {} lies outside the outer wire",
                wire_ids[hole].index()
            ),
            Misplaced::Inside(hole, other) => format!(
                "inner wire {} lies inside inner wire {}",
                wire_ids[hole].index(),
                wire_ids[other].index()
            ),
        };
        return Ok(Some(issue(face_id, description)));
    }
    Ok(None)
}

/// A wire's edges cut into stretches that cannot cross themselves, and for
/// each stretch the place of its edge in the wire.
fn wire_stretches(
    topo: &Topology,
    wire_id: WireId,
) -> Result<(Vec<Arc<'_>>, Vec<usize>), CheckError> {
    let mut stretches = Vec::new();
    let mut uses = Vec::new();
    for (place, oe) in topo.wire(wire_id)?.edges().iter().enumerate() {
        for stretch in Arc::of(topo, oe.edge(), oe.is_forward())?.stretches() {
            stretches.push(stretch);
            uses.push(place);
        }
    }
    Ok((stretches, uses))
}

/// Whether a wire of a plane face is a convex loop, which cannot cross
/// itself: its direction turns one way all along it, through one whole
/// turn, about the plane's `normal`. Lines and circles in the plane are
/// read, each circle turning through its span; any other edge, or a turn
/// back at a vertex, leaves the wire to the full search.
fn convex_loop(topo: &Topology, normal: Vec3, wire_id: WireId) -> Result<bool, CheckError> {
    // A turn smaller than this is read as none, either way.
    const STRAIGHT: f64 = 1e-9;
    let Ok(n) = normal.normalize() else {
        return Ok(false);
    };
    let signed = |from: Vec3, to: Vec3| from.cross(to).dot(n).atan2(from.dot(to));
    let mut turns = Vec::new();
    let mut first: Option<Vec3> = None;
    let mut last: Option<Vec3> = None;
    for oe in topo.wire(wire_id)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (start, end, turn) = match edge.curve() {
            EdgeCurve::Line => {
                let Ok(d) = (b - a).normalize() else {
                    return Ok(false);
                };
                let d = if oe.is_forward() { d } else { -d };
                (d, d, 0.0)
            }
            EdgeCurve::Circle(c) => {
                let along = c.u_axis().cross(c.v_axis()).dot(n);
                if (along.abs() - 1.0).abs() > STRAIGHT {
                    return Ok(false);
                }
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                let (s, e) = (c.tangent(t0), c.tangent(t1));
                let span = (t1 - t0) * along.signum();
                if oe.is_forward() {
                    (s, e, span)
                } else {
                    (-e, -s, -span)
                }
            }
            EdgeCurve::NurbsCurve(curve) => {
                let Some((s, e, turn)) = convex_spline(curve, (a, b), edge.curve(), n) else {
                    return Ok(false);
                };
                if oe.is_forward() {
                    (s, e, turn)
                } else {
                    (-e, -s, -turn)
                }
            }
            EdgeCurve::Ellipse(_) => return Ok(false),
        };
        let (Ok(start), Ok(end)) = (start.normalize(), end.normalize()) else {
            return Ok(false);
        };
        if let Some(previous) = last {
            turns.push(signed(previous, start));
        }
        first.get_or_insert(start);
        last = Some(end);
        turns.push(turn);
    }
    let (Some(first), Some(last)) = (first, last) else {
        return Ok(false);
    };
    turns.push(signed(last, first));
    // A turn of a half turn at a vertex is a cusp: no way round is told.
    if turns
        .iter()
        .any(|&t| (t.abs() - std::f64::consts::PI).abs() <= STRAIGHT)
    {
        return Ok(false);
    }
    let total: f64 = turns.iter().sum();
    let one_way = turns.iter().all(|&t| t * total.signum() >= -STRAIGHT);
    Ok(one_way && (total.abs() - TAU).abs() <= 1e-6)
}

/// A whole B-spline edge from `ends.0` to `ends.1` whose control polygon
/// lies in the plane square to `n` and turns one way, through no more than
/// a half turn where it is open: its direction at each end, in the edge's
/// stored direction, and how far it turns about `n` between them. With
/// positive weights the curve crosses no line more often than its control
/// polygon does, so it is a convex arc from the first leg's direction to
/// the last's. `None` for any other B-spline, or a piece of one.
fn convex_spline(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    ends: (Point3, Point3),
    carrier: &EdgeCurve,
    n: Vec3,
) -> Option<(Vec3, Vec3, f64)> {
    const STRAIGHT: f64 = 1e-9;
    let (d0, d1) = curve.domain();
    let (t0, t1) = carrier.domain_with_endpoints(ends.0, ends.1);
    let width = (d1 - d0).abs().max(f64::MIN_POSITIVE);
    let same = |x: f64, y: f64| (x - y).abs() <= 1e-12 * width;
    let stored_forward = if same(t0, d0) && same(t1, d1) {
        true
    } else if same(t0, d1) && same(t1, d0) {
        false
    } else {
        return None;
    };
    if !curve.weights().iter().all(|&w| w > 0.0) {
        return None;
    }
    let mut points = curve.control_points().to_vec();
    if !stored_forward {
        points.reverse();
    }
    let first = *points.first()?;
    let size = points
        .iter()
        .map(|&p| (p - first).length())
        .fold(0.0, f64::max);
    if points
        .iter()
        .any(|&p| (p - first).dot(n).abs() > STRAIGHT * size.max(1.0))
    {
        return None;
    }
    let legs: Vec<Vec3> = points
        .windows(2)
        .filter_map(|pair| {
            let leg = pair[1] - pair[0];
            (leg.length() > STRAIGHT * size.max(1.0))
                .then(|| leg.normalize().ok())
                .flatten()
        })
        .collect();
    let (&start, &end) = (legs.first()?, legs.last()?);
    let turns: Vec<f64> = legs
        .windows(2)
        .map(|pair| pair[0].cross(pair[1]).dot(n).atan2(pair[0].dot(pair[1])))
        .collect();
    let total: f64 = turns.iter().sum();
    let one_way = turns.iter().all(|&t| t * total.signum() >= -STRAIGHT);
    // A closed polygon turning one way is convex as a whole, through one
    // whole turn with its closing turn, which the wire reads at the vertex.
    let closed = points
        .last()
        .is_some_and(|&last| (last - first).length() <= STRAIGHT * size.max(1.0));
    let most = if closed {
        TAU
    } else {
        std::f64::consts::PI - STRAIGHT
    };
    (one_way && total.abs() <= most).then_some((start, end, total))
}

/// Check that no wire of a plane face crosses itself.
///
/// No edge of a wire may cross another edge of the same wire. Edges that
/// meet at a vertex, run along each other, or touch without crossing are
/// not reported. `tolerance` is the gap within which two edges meet. A
/// curved face is not read: on cones, frustums and tori the side test
/// reported crossings of wires that do not cross.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn check_face_wires_simple(
    topo: &Topology,
    face_id: FaceId,
    tolerance: f64,
) -> Result<Vec<ValidationIssue>, CheckError> {
    let face = topo.face(face_id)?;
    let FaceSurface::Plane { normal, .. } = face.surface() else {
        return Ok(vec![]);
    };
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        if convex_loop(topo, *normal, wire_id)? {
            continue;
        }
        let (stretches, uses) = wire_stretches(topo, wire_id)?;
        if stretches.len() > 1 && wire_crosses_itself(face.surface(), &stretches, &uses, tolerance)
        {
            return Ok(vec![issue(
                face_id,
                format!("wire {} crosses itself", wire_id.index()),
            )]);
        }
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

    fn crossed(topo: &Topology, face: FaceId) -> Option<String> {
        check_face_wires_simple(topo, face, 1e-6)
            .unwrap()
            .into_iter()
            .next()
            .map(|i| i.description)
    }

    /// A 10 x 6 rectangle with its corners rounded by `r`: lines along its
    /// sides and quarter circles about z, run counter-clockwise, or
    /// clockwise where `reverse`.
    fn rounded_rectangle(topo: &mut Topology, r: f64, reverse: bool) -> WireId {
        let (w, h) = (10.0, 6.0);
        let corners = [
            ([w - r, r], -std::f64::consts::FRAC_PI_2),
            ([w - r, h - r], 0.0),
            ([r, h - r], std::f64::consts::FRAC_PI_2),
            ([r, r], std::f64::consts::PI),
        ];
        let mut edges = Vec::new();
        let mut first = None;
        let mut last: Option<VertexId> = None;
        for ([cx, cy], from) in corners {
            let x = Vec3::new(1.0, 0.0, 0.0);
            let circle = Circle3D::new_with_ref(Point3::new(cx, cy, 0.0), Z, r, x).unwrap();
            let [a, b] = [from, from + std::f64::consts::FRAC_PI_2].map(|t| {
                let p = circle.evaluate(t);
                vertex(topo, [p.x(), p.y(), 0.0])
            });
            if let Some(end) = last {
                edges.push(OrientedEdge::new(
                    topo.add_edge(Edge::new(end, a, EdgeCurve::Line)),
                    true,
                ));
            }
            first.get_or_insert(a);
            edges.push(OrientedEdge::new(
                topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle))),
                true,
            ));
            last = Some(b);
        }
        let (first, last) = (first.unwrap(), last.unwrap());
        edges.push(OrientedEdge::new(
            topo.add_edge(Edge::new(last, first, EdgeCurve::Line)),
            true,
        ));
        if reverse {
            edges = edges
                .into_iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
        }
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    /// A rounded rectangle turns one way through one whole turn, either way
    /// round: it is read as convex, and does not cross itself. An L of lines
    /// turns back at its inner corner, and a bow tie turns through no whole
    /// turn: neither is read as convex.
    #[test]
    fn a_convex_loop_of_lines_and_arcs_is_read_without_a_search() {
        let mut topo = Topology::new();
        for reverse in [false, true] {
            let wire = rounded_rectangle(&mut topo, 1.5, reverse);
            assert!(convex_loop(&topo, Z, wire).unwrap(), "reverse {reverse}");
            assert!(convex_loop(&topo, -Z, wire).unwrap(), "reverse {reverse}");
            let face = flat(&mut topo, wire, vec![]);
            assert_eq!(crossed(&topo, face), None);
        }
        let l = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 2.0, 0.0],
                [2.0, 2.0, 0.0],
                [2.0, 4.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        );
        let l = polygon(&mut topo, &l);
        assert!(!convex_loop(&topo, Z, l).unwrap());
        let face = flat(&mut topo, l, vec![]);
        assert_eq!(crossed(&topo, face), None);
        let bow = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [4.0, 4.0, 0.0],
                [4.0, 0.0, 0.0],
                [0.0, 4.0, 0.0],
            ],
        );
        let bow = polygon(&mut topo, &bow);
        assert!(!convex_loop(&topo, Z, bow).unwrap());
        let face = flat(&mut topo, bow, vec![]);
        assert!(crossed(&topo, face).is_some_and(|d| d.contains("crosses itself")));
    }

    #[test]
    fn a_square_outline_does_not_cross_itself() {
        let mut topo = Topology::new();
        let outer = square(&mut topo);
        let hole = circle(&mut topo, [5.0, 5.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![hole]);
        assert_eq!(crossed(&topo, face), None);
    }

    /// The outline the issue's fillet left on the floor: its two contacts
    /// cross, so the outline runs out past one and back.
    #[test]
    fn an_outline_whose_contacts_cross_is_reported() {
        let mut topo = Topology::new();
        let corners = points(
            &mut topo,
            &[
                [20.0, 3.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 20.0, 0.0],
                [19.0, 20.0, 0.0],
                [19.0, 0.0, 0.0],
            ],
        );
        let outer = polygon(&mut topo, &corners);
        let face = flat(&mut topo, outer, vec![]);
        assert!(crossed(&topo, face).is_some_and(|d| d.contains("crosses itself")));
    }

    /// Two loops that meet at one vertex touch there and do not cross.
    #[test]
    fn an_outline_pinched_at_a_vertex_passes() {
        let mut topo = Topology::new();
        let c = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [10.0, 0.0, 0.0],
                [5.0, 5.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let outer = polygon(&mut topo, &[c[0], c[1], c[2], c[3], c[4], c[2]]);
        let face = flat(&mut topo, outer, vec![]);
        assert_eq!(crossed(&topo, face), None);
    }

    /// A bow tie: the outline crosses itself between its vertices.
    #[test]
    fn a_bow_tie_is_reported() {
        let mut topo = Topology::new();
        let corners = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [10.0, 10.0, 0.0],
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let outer = polygon(&mut topo, &corners);
        let face = flat(&mut topo, outer, vec![]);
        assert!(crossed(&topo, face).is_some());
    }

    /// A hole of circle and seam on a cylinder band, and the band's own
    /// outline that runs up its seam and back down, cross nothing.
    #[test]
    fn a_cylinder_band_and_its_window_do_not_cross_themselves() {
        let mut topo = Topology::new();
        let face = windowed_band(&mut topo, 0.3, 2.0, 3.0);
        assert_eq!(crossed(&topo, face), None);
    }

    /// A cubic and the line between its ends, on a plane.
    fn cubic_and_chord(topo: &mut Topology, control: [[f64; 2]; 4]) -> FaceId {
        let points: Vec<Point3> = control
            .iter()
            .map(|&[x, y]| Point3::new(x, y, 0.0))
            .collect();
        let cubic = brepkit_math::nurbs::curve::NurbsCurve::new(
            3,
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            points.clone(),
            vec![1.0; 4],
        )
        .unwrap();
        let a = vertex(topo, [points[0].x(), points[0].y(), 0.0]);
        let b = vertex(topo, [points[3].x(), points[3].y(), 0.0]);
        let curve = topo.add_edge(Edge::new(a, b, EdgeCurve::NurbsCurve(cubic)));
        let chord = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(curve, true),
                    OrientedEdge::new(chord, true),
                ],
                true,
            )
            .unwrap(),
        );
        flat(topo, wire, vec![])
    }

    /// A cubic whose control points run out and back loops over itself,
    /// crossing at (0.5, 0.5625): one edge, so no pair of edges shows it.
    #[test]
    fn a_cubic_that_loops_over_itself_is_reported() {
        let mut topo = Topology::new();
        let face = cubic_and_chord(&mut topo, [[0.0, 0.0], [3.0, 3.0], [-2.0, 3.0], [1.0, 0.0]]);
        assert!(crossed(&topo, face).is_some_and(|d| d.contains("crosses itself")));
    }

    /// Cubics whose control points run out and back without the curve
    /// coming back over itself, and one whose control points run one way.
    #[test]
    fn cubics_that_do_not_loop_pass() {
        for control in [
            [[0.0, 0.0], [-1.0, 1.0], [2.0, 1.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.3, 1.0], [0.7, 1.0], [1.0, 0.0]],
        ] {
            let mut topo = Topology::new();
            let face = cubic_and_chord(&mut topo, control);
            assert_eq!(crossed(&topo, face), None, "{control:?}");
        }
    }

    /// The swept search flags a wire exactly where the search of every pair
    /// of its edges does: random outlines of lines and arcs, many of them
    /// crossing themselves, on a plane.
    #[test]
    fn the_swept_search_flags_what_the_search_of_every_pair_flags() {
        let mut state: u64 = 0x2587;
        let mut next = move || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss)]
            let unit = (state >> 11) as f64 / (1_u64 << 53) as f64;
            unit
        };
        let (mut crossed, mut clear) = (0, 0);
        for round in 0..400 {
            let mut topo = Topology::new();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let count = 3 + (next() * 25.0) as usize;
            // Points round a circle, each pushed in or out, some swapped so
            // the outline doubles back across itself.
            let mut corners: Vec<[f64; 3]> = (0..count)
                .map(|k| {
                    #[allow(clippy::cast_precision_loss)]
                    let angle = TAU * (k as f64 + 0.4 * next()) / count as f64;
                    let r = 10.0 * (0.4 + next());
                    [r * angle.cos(), r * angle.sin(), 0.0]
                })
                .collect();
            if round % 3 == 0 && count > 4 {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let k = (next() * (count - 2) as f64) as usize;
                corners.swap(k, k + 2);
            }
            let vertices = points(&mut topo, &corners);
            let mut edges = Vec::with_capacity(count);
            for k in 0..count {
                let (a, b) = (vertices[k], vertices[(k + 1) % count]);
                let (pa, pb) = (
                    topo.vertex(a).unwrap().point(),
                    topo.vertex(b).unwrap().point(),
                );
                let curve = if next() < 0.4 {
                    // An arc through both ends that bows to one side.
                    let mid = pa + (pb - pa) * 0.5;
                    let chord = pb - pa;
                    let side = Z.cross(chord).normalize().unwrap();
                    let centre = mid + side * (chord.length() * (next() - 0.5) * 3.0);
                    let normal = (pa - centre).cross(pb - centre).normalize().unwrap();
                    EdgeCurve::Circle(
                        Circle3D::new(centre, normal, (pa - centre).length()).unwrap(),
                    )
                } else {
                    EdgeCurve::Line
                };
                edges.push(OrientedEdge::new(
                    topo.add_edge(Edge::new(a, b, curve)),
                    true,
                ));
            }
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            let face = flat(&mut topo, wire, vec![]);
            let surface = topo.face(face).unwrap().surface().clone();
            let arcs = topo
                .wire(wire)
                .unwrap()
                .edges()
                .iter()
                .map(|oe| Arc::of(&topo, oe.edge(), oe.is_forward()))
                .collect::<Result<Vec<_>, _>>()
                .unwrap();
            let every_pair = wires_cross(&surface, &arcs, &arcs, 1e-6);
            assert_eq!(
                wire_crosses_itself(&surface, &arcs, &(0..arcs.len()).collect::<Vec<_>>(), 1e-6),
                every_pair,
                "round {round}: {corners:?}"
            );
            if every_pair {
                crossed += 1;
            } else {
                clear += 1;
            }
        }
        assert!(
            crossed > 50 && clear > 50,
            "{crossed} crossed, {clear} clear"
        );
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
    fn only_a_misplaced_hole_reads_as_out_of_place() {
        let placed = |topo: &Topology, face| {
            check_face_holes_placed(topo, face, 1e-6)
                .unwrap()
                .into_iter()
                .next()
                .map(|i| i.description)
        };
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let crossing = vec![
            circle(&mut topo, [4.0, 5.0, 0.0], 2.0),
            circle(&mut topo, [6.0, 5.0, 0.0], 2.0),
        ];
        let face = flat(&mut topo, outer, crossing);
        assert!(found(&topo, face).is_some());
        assert_eq!(placed(&topo, face), None);
        let outside = circle(&mut topo, [20.0, 5.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![outside]);
        assert_eq!(placed(&topo, face), found(&topo, face));
        assert!(
            placed(&topo, face)
                .unwrap()
                .contains("outside the outer wire")
        );
        let nested = vec![
            circle(&mut topo, [5.0, 5.0, 0.0], 3.0),
            circle(&mut topo, [5.5, 5.0, 0.0], 1.0),
        ];
        let face = flat(&mut topo, outer, nested);
        assert!(
            placed(&topo, face)
                .unwrap()
                .contains("lies inside inner wire")
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

    /// A hole that stands out of the outline by a ten thousandth: its two
    /// crossings lie close to each other.
    #[test]
    fn a_hole_a_little_across_the_outline_is_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let hole = circle(&mut topo, [8.0001, 5.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }

    /// A hole of a line and a spline that leaves the outline in one narrow
    /// spike, between points that all lie inside.
    #[test]
    fn a_splines_narrow_spike_across_the_outline_is_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let n = 41;
        let control: Vec<Point3> = (0..n)
            .map(|k| {
                let x = 7.0 - 4.0 * fraction(k, n - 1);
                let y = if k == n / 2 { 30.0 } else { 6.0 };
                Point3::new(x, if k == 0 || k == n - 1 { 5.0 } else { y }, 0.0)
            })
            .collect();
        let mut knots = vec![0.0; 4];
        knots.extend((1..n - 3).map(|k| fraction(k, n - 3)));
        knots.extend([1.0; 4]);
        let spline =
            brepkit_math::nurbs::curve::NurbsCurve::new(3, knots, control, vec![1.0; n]).unwrap();
        let ends = points(&mut topo, &[[3.0, 5.0, 0.0], [7.0, 5.0, 0.0]]);
        let base = topo.add_edge(Edge::new(ends[0], ends[1], EdgeCurve::Line));
        let top = topo.add_edge(Edge::new(ends[1], ends[0], EdgeCurve::NurbsCurve(spline)));
        let hole = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(base, true), OrientedEdge::new(top, true)],
                true,
            )
            .unwrap(),
        );
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }

    /// A circle's box stands off its plane by its chords' bound, and a
    /// polygon's does not: the circle's box is not within the polygon's.
    #[test]
    fn a_round_hole_inside_a_square_hole_is_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let corners = points(
            &mut topo,
            &[
                [2.0, 2.0, 0.0],
                [8.0, 2.0, 0.0],
                [8.0, 8.0, 0.0],
                [2.0, 8.0, 0.0],
            ],
        );
        let holes = vec![
            polygon(&mut topo, &corners),
            circle(&mut topo, [5.0, 5.0, 0.0], 1.0),
        ];
        let face = flat(&mut topo, outer, holes);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("lies inside inner wire")
        );
    }

    /// A hole whose spline leaves the outline and comes back within one
    /// chord of the spline.
    #[test]
    fn a_splines_two_crossings_in_one_chord_are_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        // x = 3 + 4 t, y = 10.0001 - 5 (t - 0.45)^2, as a quadratic Bezier.
        let y = |t: f64| 10.0001 - 5.0 * (t - 0.45) * (t - 0.45);
        let slope = |t: f64| -10.0 * (t - 0.45);
        let control = vec![
            Point3::new(3.0, y(0.0), 0.0),
            Point3::new(5.0, y(0.0) + 0.5 * slope(0.0), 0.0),
            Point3::new(7.0, y(1.0), 0.0),
        ];
        let spline = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            control,
            vec![1.0; 3],
        )
        .unwrap();
        let ends = points(&mut topo, &[[3.0, y(0.0), 0.0], [7.0, y(1.0), 0.0]]);
        let top = topo.add_edge(Edge::new(ends[0], ends[1], EdgeCurve::NurbsCurve(spline)));
        let base = topo.add_edge(Edge::new(ends[1], ends[0], EdgeCurve::Line));
        let hole = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(top, true), OrientedEdge::new(base, true)],
                true,
            )
            .unwrap(),
        );
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }

    /// Two crossings 1e-4 apart on an outline 1000 long.
    #[test]
    fn two_crossings_close_on_a_long_outline_are_reported() {
        let mut topo = Topology::default();
        let corners = points(
            &mut topo,
            &[
                [0.0, 0.0, 0.0],
                [1000.0, 0.0, 0.0],
                [1000.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let outer = polygon(&mut topo, &corners);
        // A thin slot of a hole through the outline's long edge.
        let spike = points(
            &mut topo,
            &[
                [499.999_95, 5.0, 0.0],
                [499.999_95, -5.0, 0.0],
                [500.000_05, -5.0, 0.0],
                [500.000_05, 5.0, 0.0],
            ],
        );
        let hole = polygon(&mut topo, &spike);
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("crosses the outer wire")
        );
    }

    /// A circle outside the outline that touches it at the middle of the
    /// circle's edge, and one inside another hole that touches it there.
    /// A circle's edge starts on its +x side, so its middle is on its -x
    /// side.
    #[test]
    fn a_hole_misplaced_and_touching_at_its_edges_middle_is_reported() {
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let hole = circle(&mut topo, [12.0, 5.0, 0.0], 2.0);
        let face = flat(&mut topo, outer, vec![hole]);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("outside the outer wire")
        );

        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let holes = vec![
            circle(&mut topo, [5.0, 5.0, 0.0], 3.0),
            circle(&mut topo, [3.0, 5.0, 0.0], 1.0),
        ];
        let face = flat(&mut topo, outer, holes);
        assert!(
            found(&topo, face)
                .unwrap()
                .contains("lies inside inner wire")
        );
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
    fn a_thin_window_across_a_cylinders_rim_is_reported() {
        let mut topo = Topology::default();
        let face = windowed_band(&mut topo, 1.0, 9.999, 10.001);
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

    /// A closed edge on `curve` through `at`, as a wire run forward or not.
    fn closed(topo: &mut Topology, curve: EdgeCurve, at: Point3, forward: bool) -> WireId {
        let v = vertex(topo, [at.x(), at.y(), at.z()]);
        let e = topo.add_edge(Edge::new(v, v, curve));
        topo.add_wire(Wire::new(vec![OrientedEdge::new(e, forward)], true).unwrap())
    }

    /// The circle at height `z` about the z axis.
    fn ring(z: f64, r: f64) -> Circle3D {
        Circle3D::new(Point3::new(0.0, 0.0, z), Z, r).unwrap()
    }

    fn ring_wire(topo: &mut Topology, z: f64, r: f64, forward: bool) -> WireId {
        let c = ring(z, r);
        closed(topo, EdgeCurve::Circle(c.clone()), c.evaluate(0.0), forward)
    }

    /// A window of two arcs about the z axis from `low` to `high`, from the
    /// angle `from` a quarter turn on, joined by two lines, with the arcs'
    /// radii `r(z)`, run forward or not.
    fn window(
        topo: &mut Topology,
        from: f64,
        (low, high): (f64, f64),
        r: &dyn Fn(f64) -> f64,
        forward: bool,
    ) -> WireId {
        let (lower, upper) = (ring(low, r(low)), ring(high, r(high)));
        let to = from + std::f64::consts::FRAC_PI_2;
        let corner = |topo: &mut Topology, c: &Circle3D, t: f64| {
            let p = c.evaluate(t);
            vertex(topo, [p.x(), p.y(), p.z()])
        };
        let (p0, p1) = (corner(topo, &lower, from), corner(topo, &lower, to));
        let (q0, q1) = (corner(topo, &upper, from), corner(topo, &upper, to));
        let mut edges = vec![
            OrientedEdge::new(
                topo.add_edge(Edge::new(p0, p1, EdgeCurve::Circle(lower))),
                true,
            ),
            OrientedEdge::new(topo.add_edge(Edge::new(p1, q1, EdgeCurve::Line)), true),
            OrientedEdge::new(
                topo.add_edge(Edge::new(q0, q1, EdgeCurve::Circle(upper))),
                false,
            ),
            OrientedEdge::new(topo.add_edge(Edge::new(p0, q0, EdgeCurve::Line)), false),
        ];
        if !forward {
            edges = edges
                .iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
        }
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    /// A band about the z axis from `low` to `high`: its two rims and a seam
    /// between them, with the rims' radii `r(z)`.
    fn band(topo: &mut Topology, (low, high): (f64, f64), r: &dyn Fn(f64) -> f64) -> WireId {
        let (bottom, top) = (ring(low, r(low)), ring(high, r(high)));
        let (a, b) = (bottom.evaluate(0.0), top.evaluate(0.0));
        let va = vertex(topo, [a.x(), a.y(), a.z()]);
        let vb = vertex(topo, [b.x(), b.y(), b.z()]);
        let rim_a = topo.add_edge(Edge::new(va, va, EdgeCurve::Circle(bottom)));
        let rim_b = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(top)));
        let seam = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
        topo.add_wire(
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
        )
    }

    fn cylinder() -> FaceSurface {
        FaceSurface::Cylinder(CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Z, 3.0).unwrap())
    }

    const THREE: &dyn Fn(f64) -> f64 = &|_| 3.0;

    #[test]
    fn a_window_beyond_a_cylinders_band_is_reported() {
        for forward in [true, false] {
            let mut topo = Topology::default();
            let outer = band(&mut topo, (0.0, 10.0), THREE);
            let hole = window(&mut topo, 1.0, (12.0, 14.0), THREE, forward);
            let face = topo.add_face(Face::new(outer, vec![hole], cylinder()));
            assert!(
                found(&topo, face)
                    .unwrap()
                    .contains("outside the outer wire")
            );
        }
    }

    /// A band of a cylinder whose lower rim, a B-spline of degree one, dips
    /// to `z = -100` between the points the check samples along it: the
    /// samples all lie at `z = 0`. A window inside the band passes, and a
    /// window above it is reported.
    #[test]
    fn a_band_whose_rim_dips_between_its_samples_reads_its_windows() {
        for (window_at, outside) in [((3.0, 7.0), false), ((12.0, 14.0), true)] {
            let mut topo = Topology::default();
            let at = |angle: f64, z: f64| Point3::new(3.0 * angle.cos(), 3.0 * angle.sin(), z);
            let mut points = vec![at(0.0, 0.0), at(TAU / 32.0, -100.0)];
            let mut knots = vec![0.0, 0.0, 1.0 / 32.0];
            for k in 1..=16 {
                points.push(at(TAU * f64::from(k) / 16.0, 0.0));
                knots.push(f64::from(k) / 16.0);
            }
            knots.push(1.0);
            let weights = vec![1.0; points.len()];
            let dip =
                brepkit_math::nurbs::curve::NurbsCurve::new(1, knots, points, weights).unwrap();
            let va = vertex(&mut topo, [3.0, 0.0, 0.0]);
            let vb = vertex(&mut topo, [3.0, 0.0, 10.0]);
            let rim_a = topo.add_edge(Edge::new(va, va, EdgeCurve::NurbsCurve(dip)));
            let rim_b = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(ring(10.0, 3.0))));
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
            let hole = window(&mut topo, 1.0, window_at, THREE, true);
            let face = topo.add_face(Face::new(outer, vec![hole], cylinder()));
            let found = found(&topo, face);
            assert_eq!(
                found
                    .as_ref()
                    .is_some_and(|d| d.contains("outside the outer wire")),
                outside,
                "{window_at:?}: {found:?}"
            );
        }
    }

    #[test]
    fn a_window_inside_a_window_of_a_cylinder_is_reported() {
        let mut topo = Topology::default();
        let outer = band(&mut topo, (0.0, 10.0), THREE);
        let wide = window(&mut topo, 1.0, (2.0, 8.0), THREE, false);
        let narrow = window(&mut topo, 1.3, (4.0, 6.0), &|_| 3.0, false);
        // A quarter turn from 1.3 runs past the wide window's end at 1 + a
        // quarter turn: narrow it by turning the arcs' ends in.
        let _ = narrow;
        let narrow = {
            let (lower, upper) = (ring(4.0, 3.0), ring(6.0, 3.0));
            let corner = |topo: &mut Topology, c: &Circle3D, t: f64| {
                let p = c.evaluate(t);
                vertex(&mut *topo, [p.x(), p.y(), p.z()])
            };
            let (p0, p1) = (
                corner(&mut topo, &lower, 1.2),
                corner(&mut topo, &lower, 2.2),
            );
            let (q0, q1) = (
                corner(&mut topo, &upper, 1.2),
                corner(&mut topo, &upper, 2.2),
            );
            let edges = vec![
                OrientedEdge::new(
                    topo.add_edge(Edge::new(p0, p1, EdgeCurve::Circle(lower))),
                    true,
                ),
                OrientedEdge::new(topo.add_edge(Edge::new(p1, q1, EdgeCurve::Line)), true),
                OrientedEdge::new(
                    topo.add_edge(Edge::new(q0, q1, EdgeCurve::Circle(upper))),
                    false,
                ),
                OrientedEdge::new(topo.add_edge(Edge::new(p0, q0, EdgeCurve::Line)), false),
            ];
            topo.add_wire(Wire::new(edges, true).unwrap())
        };
        for holes in [vec![wide, narrow], vec![narrow, wide]] {
            let face = topo.add_face(Face::new(outer, holes, cylinder()));
            assert!(
                found(&topo, face)
                    .unwrap()
                    .contains("lies inside inner wire")
            );
        }
        let beside = window(&mut topo, 3.0, (2.0, 8.0), THREE, true);
        let face = topo.add_face(Face::new(outer, vec![wide, beside], cylinder()));
        assert_eq!(found(&topo, face), None);
    }

    /// A band bounded by its two rims alone, no seam: either rim may be the
    /// outer wire, and a window between them passes. A window beyond the
    /// rim that is a hole lies inside that hole, on its side away from the
    /// face; one beyond the outer wire lies outside it. A face with one rim
    /// is reported.
    #[test]
    fn a_seamless_band_of_a_cylinder_reads_its_windows() {
        let mut topo = Topology::default();
        let bottom = ring_wire(&mut topo, 0.0, 3.0, true);
        let top = ring_wire(&mut topo, 10.0, 3.0, false);
        let inside = window(&mut topo, 1.0, (3.0, 7.0), THREE, false);
        let above = window(&mut topo, 1.0, (12.0, 14.0), THREE, false);
        for (outer, rim, beyond) in [
            (bottom, top, "lies inside inner wire"),
            (top, bottom, "outside the outer wire"),
        ] {
            let face = topo.add_face(Face::new(outer, vec![rim, inside], cylinder()));
            assert_eq!(found(&topo, face), None);
            let face = topo.add_face(Face::new(outer, vec![rim, inside, above], cylinder()));
            assert!(found(&topo, face).is_some_and(|d| d.contains(beyond)));
        }
        let face = topo.add_face(Face::new(bottom, vec![inside], cylinder()));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    fn cone() -> (FaceSurface, Box<dyn Fn(f64) -> f64>) {
        let surface =
            brepkit_math::surfaces::ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Z, 0.6)
                .unwrap();
        // Height `z = v sin a` and radius `v cos a` along the generator.
        let slope = 0.6_f64.cos() / 0.6_f64.sin();
        (FaceSurface::Cone(surface), Box::new(move |z| z * slope))
    }

    #[test]
    fn a_cones_holes_read_as_a_discs() {
        let (surface, r) = cone();
        let mut topo = Topology::default();
        let outer = band(&mut topo, (2.0, 6.0), &*r);
        let inside = window(&mut topo, 1.0, (3.0, 5.0), &*r, true);
        let face = topo.add_face(Face::new(outer, vec![inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let above = window(&mut topo, 1.0, (7.0, 9.0), &*r, true);
        let face = topo.add_face(Face::new(outer, vec![inside, above], surface.clone()));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));

        // A seamless band: the wide rim holds the narrow one, about the apex.
        let narrow = ring_wire(&mut topo, 2.0, r(2.0), true);
        let wide = ring_wire(&mut topo, 6.0, r(6.0), false);
        let face = topo.add_face(Face::new(wide, vec![narrow, inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(narrow, vec![wide, inside], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    const R: f64 = 5.0;

    /// The circle on the sphere of radius `R` about the origin at the
    /// latitude `degrees`.
    fn latitude(topo: &mut Topology, degrees: f64, forward: bool) -> WireId {
        let (s, c) = degrees.to_radians().sin_cos();
        ring_wire(topo, R * s, R * c, forward)
    }

    /// The circle on the sphere ten degrees about the direction at latitude
    /// `degrees` on the meridian through +x.
    fn spot(topo: &mut Topology, degrees: f64, forward: bool) -> WireId {
        let (s, c) = degrees.to_radians().sin_cos();
        let axis = Vec3::new(c, 0.0, s);
        let (rs, rc) = 10.0_f64.to_radians().sin_cos();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0) + axis * (R * rc), axis, R * rs).unwrap();
        let at = circle.evaluate(0.0);
        closed(topo, EdgeCurve::Circle(circle), at, forward)
    }

    fn sphere() -> FaceSurface {
        FaceSurface::Sphere(
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), R).unwrap(),
        )
    }

    /// Whichever way round its wires run, a sphere face reads where its holes
    /// lie: all on one side of the outline, or on both; side by side, or one
    /// inside another, read from the outline's side.
    #[test]
    fn a_spheres_holes_read_whichever_way_round_its_wires_run() {
        for (outline_forward, holes_forward) in
            [(true, true), (true, false), (false, true), (false, false)]
        {
            let mut topo = Topology::default();
            let mut face = |outer: (f64, bool), holes: &[(f64, bool)]| {
                let outer = latitude(&mut topo, outer.0, outer.1 == outline_forward);
                let holes = holes
                    .iter()
                    .map(|&(at, round)| {
                        if round {
                            latitude(&mut topo, at, holes_forward)
                        } else {
                            spot(&mut topo, at, holes_forward)
                        }
                    })
                    .collect();
                let id = topo.add_face(Face::new(outer, holes, sphere()));
                found(&topo, id)
            };
            let case = (outline_forward, holes_forward);
            assert_eq!(
                face((0.0, true), &[(75.0, true), (30.0, false)]),
                None,
                "{case:?}"
            );
            assert!(
                face((0.0, true), &[(50.0, true), (-50.0, true)])
                    .is_some_and(|d| d.contains("outside the outer wire")),
                "{case:?}"
            );
            assert!(
                face((0.0, true), &[(50.0, true), (70.0, true)])
                    .is_some_and(|d| d.contains("lies inside inner wire")),
                "{case:?}"
            );
            // The face beyond a small cap round the north pole.
            assert_eq!(
                face((80.0, true), &[(-50.0, true), (0.0, false)]),
                None,
                "{case:?}"
            );
            assert!(
                face((80.0, true), &[(-50.0, true), (-70.0, false)])
                    .is_some_and(|d| d.contains("lies inside inner wire")),
                "{case:?}"
            );
        }
    }

    /// The same sphere as a closed B-spline surface with a pole at each end.
    #[test]
    fn a_spline_spheres_holes_are_read() {
        let surface = FaceSurface::Nurbs(
            brepkit_math::surfaces::SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), R)
                .unwrap()
                .to_rational_nurbs()
                .unwrap(),
        );
        let mut topo = Topology::default();
        let outer = latitude(&mut topo, 0.0, true);
        let holes = vec![
            latitude(&mut topo, 50.0, false),
            latitude(&mut topo, -50.0, false),
        ];
        let face = topo.add_face(Face::new(outer, holes, surface.clone()));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
        let holes = vec![
            latitude(&mut topo, 50.0, false),
            spot(&mut topo, 30.0, false),
        ];
        let face = topo.add_face(Face::new(outer, holes, surface));
        assert_eq!(found(&topo, face), None);
    }

    /// A band of a cylinder as a B-spline surface closed round its axis.
    #[test]
    fn a_spline_cylinders_window_beyond_its_band_is_reported() {
        let surface = FaceSurface::Nurbs(
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Z, 3.0)
                .unwrap()
                .to_nurbs(-5.0, 25.0)
                .unwrap(),
        );
        let mut topo = Topology::default();
        let outer = band(&mut topo, (0.0, 10.0), THREE);
        let inside = window(&mut topo, 1.0, (3.0, 7.0), THREE, true);
        let above = window(&mut topo, 1.0, (12.0, 14.0), THREE, true);
        let face = topo.add_face(Face::new(outer, vec![inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(outer, vec![inside, above], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// An open B-spline patch, flat, read in its own parameters.
    #[test]
    fn an_open_spline_patchs_hole_outside_its_outline_is_reported() {
        let control = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(20.0 * f64::from(i), 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                control,
                vec![vec![1.0; 2]; 2],
            )
            .unwrap(),
        );
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let inside = circle(&mut topo, [5.0, 5.0, 0.0], 2.0);
        let outside = circle(&mut topo, [15.0, 15.0, 0.0], 2.0);
        let face = topo.add_face(Face::new(outer, vec![inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(outer, vec![outside], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    fn torus() -> brepkit_math::surfaces::ToroidalSurface {
        brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0, 2.0).unwrap()
    }

    /// A loop of lines on the torus round `(u, v)`, `(a, b)` across.
    fn torus_loop(topo: &mut Topology, (u, v): (f64, f64), (a, b): (f64, f64)) -> WireId {
        let t = torus();
        let corners: Vec<[f64; 3]> = (0..48)
            .map(|k| {
                let s = TAU * f64::from(k) / 48.0;
                let p = t.evaluate(u + a * s.cos(), v + b * s.sin());
                [p.x(), p.y(), p.z()]
            })
            .collect();
        let ids = points(topo, &corners);
        polygon(topo, &ids)
    }

    /// The whole torus: its outer wire is its two seams, lines closed on
    /// one vertex.
    fn whole_torus_outline(topo: &mut Topology) -> WireId {
        let v = vertex(topo, [8.0, 0.0, 0.0]);
        let ea = topo.add_edge(Edge::new(v, v, EdgeCurve::Line));
        let eb = topo.add_edge(Edge::new(v, v, EdgeCurve::Line));
        topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(ea, true),
                    OrientedEdge::new(eb, true),
                    OrientedEdge::new(ea, false),
                    OrientedEdge::new(eb, false),
                ],
                true,
            )
            .unwrap(),
        )
    }

    #[test]
    fn a_toruss_holes_are_read_across_its_seams() {
        let mut topo = Topology::default();
        let outer = whole_torus_outline(&mut topo);
        // Holes round the seams, where `u` and `v` start again.
        let a = torus_loop(&mut topo, (0.0, 0.0), (0.3, 0.4));
        let b = torus_loop(&mut topo, (3.0, 3.0), (0.3, 0.4));
        let around_a = torus_loop(&mut topo, (0.0, 0.0), (0.6, 0.8));
        let face = topo.add_face(Face::new(outer, vec![a, b], FaceSurface::Torus(torus())));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(
            outer,
            vec![a, b, around_a],
            FaceSurface::Torus(torus()),
        ));
        assert!(found(&topo, face).is_some_and(|d| d.contains("lies inside inner wire")));

        // Two holes alone, one round the other.
        let face = topo.add_face(Face::new(
            outer,
            vec![a, around_a],
            FaceSurface::Torus(torus()),
        ));
        assert!(found(&topo, face).is_some_and(|d| d.contains("lies inside inner wire")));

        // A patch round the seams: a hole inside it passes, and with one
        // more beyond it, the two lie on two sides of the outline.
        let patch = torus_loop(&mut topo, (0.0, 0.0), (1.0, 1.0));
        let c = torus_loop(&mut topo, (0.1, 0.0), (0.3, 0.4));
        let face = topo.add_face(Face::new(patch, vec![c], FaceSurface::Torus(torus())));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(patch, vec![c, b], FaceSurface::Torus(torus())));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// A torus face between two tube circles, with no seam: holes in both of
    /// the bands between them are reported.
    #[test]
    fn a_toruss_holes_in_both_bands_between_two_tube_circles_are_reported() {
        let t = torus();
        let mut topo = Topology::default();
        let tube = |topo: &mut Topology, u: f64| {
            let (s, c) = u.sin_cos();
            let circle = Circle3D::new(
                Point3::new(6.0 * c, 6.0 * s, 0.0),
                Vec3::new(-s, c, 0.0),
                2.0,
            )
            .unwrap();
            let at = circle.evaluate(0.0);
            closed(topo, EdgeCurve::Circle(circle), at, true)
        };
        let outer = tube(&mut topo, 0.0);
        let rim = tube(&mut topo, std::f64::consts::PI);
        let near = torus_loop(&mut topo, (1.0, 1.0), (0.3, 0.4));
        let also_near = torus_loop(&mut topo, (2.0, 4.0), (0.3, 0.4));
        let far = torus_loop(&mut topo, (4.5, 1.0), (0.3, 0.4));
        let face = topo.add_face(Face::new(
            outer,
            vec![rim, near, also_near],
            FaceSurface::Torus(t),
        ));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(
            outer,
            vec![rim, near, far],
            FaceSurface::Torus(torus()),
        ));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// Turning a hole, or the outline, turns whether the hole runs the
    /// outline's way round, on a sphere, a cylinder and a torus. The
    /// equator and the circle at 50 degrees north, both counter-clockwise
    /// about +z, run one way round: the band between them lies to the left
    /// of the one and to the right of the other. So do a band of a cylinder
    /// and a window that both run counter-clockwise in `(u, v)`.
    #[test]
    fn a_turned_hole_runs_the_other_way_round() {
        let read = |topo: &Topology, face: FaceId| {
            crate::face_loops::FaceLoops::of(topo, face, 1e-6)
                .unwrap()
                .unwrap()
                .holes_running_with_outline()
        };
        let mut topo = Topology::default();
        let mut sphere_face = |outline: bool, hole: bool| {
            let outer = latitude(&mut topo, 0.0, outline);
            let hole = latitude(&mut topo, 50.0, hole);
            let face = topo.add_face(Face::new(outer, vec![hole], sphere()));
            read(&topo, face)
        };
        let same = sphere_face(true, true);
        assert_eq!(same, vec![Some(true)]);
        assert_eq!(sphere_face(false, false), same);
        assert_ne!(sphere_face(true, false), same);
        assert_ne!(sphere_face(false, true), same);

        let mut topo = Topology::default();
        let outer = band(&mut topo, (0.0, 10.0), THREE);
        let mut readings = Vec::new();
        for forward in [true, false] {
            let hole = window(&mut topo, 1.0, (3.0, 7.0), THREE, forward);
            let face = topo.add_face(Face::new(outer, vec![hole], cylinder()));
            readings.push(read(&topo, face));
        }
        assert_eq!(readings, vec![vec![Some(true)], vec![Some(false)]]);

        let mut topo = Topology::default();
        let patch = torus_loop(&mut topo, (0.0, 0.0), (1.0, 1.0));
        let hole = torus_loop(&mut topo, (0.1, 0.0), (0.3, 0.4));
        let turned: Vec<OrientedEdge> = topo
            .wire(hole)
            .unwrap()
            .edges()
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        let turned = topo.add_wire(Wire::new(turned, true).unwrap());
        let one = topo.add_face(Face::new(patch, vec![hole], FaceSurface::Torus(torus())));
        let other = topo.add_face(Face::new(patch, vec![turned], FaceSurface::Torus(torus())));
        let (one, other) = (read(&topo, one), read(&topo, other));
        assert!(matches!(one[..], [Some(_)]));
        assert_ne!(one, other);
    }

    /// A patch 20 across whose `v` spans 1e-10: each parameter is read as
    /// the fraction of its span, so the hole outside the outline reads clear
    /// of it.
    #[test]
    fn a_hole_outside_an_outline_on_a_patch_of_unequal_spans_is_reported() {
        let control = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(20.0 * f64::from(i), 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1e-10, 1e-10],
                control,
                vec![vec![1.0; 2]; 2],
            )
            .unwrap(),
        );
        let mut topo = Topology::default();
        let outer = square(&mut topo);
        let inside = circle(&mut topo, [5.0, 5.0, 0.0], 2.0);
        // Beyond the outline along `v` alone.
        let outside = circle(&mut topo, [5.0, 15.0, 0.0], 2.0);
        let face = topo.add_face(Face::new(outer, vec![inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(outer, vec![outside], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// An outline on a flat patch whose lower edge is one span of a quintic
    /// B-spline that bulges up between three points that lie on a line: a
    /// small hole under the bulge lies outside the outline.
    #[test]
    fn a_hole_under_a_splines_bulge_between_its_samples_is_reported() {
        let control = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(20.0 * f64::from(i), 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                control,
                vec![vec![1.0; 2]; 2],
            )
            .unwrap(),
        );
        let ys = [5.0, 25.0 / 9.0, 80.0 / 9.0, 10.0 / 9.0, 65.0 / 9.0, 5.0];
        let bulge = brepkit_math::nurbs::curve::NurbsCurve::new(
            5,
            [vec![0.0; 6], vec![1.0; 6]].concat(),
            (0..6)
                .map(|k| Point3::new(2.0 * f64::from(k), ys[k as usize], 0.0))
                .collect(),
            vec![1.0; 6],
        )
        .unwrap();
        let mut topo = Topology::default();
        let corners = points(
            &mut topo,
            &[
                [0.0, 5.0, 0.0],
                [10.0, 5.0, 0.0],
                [10.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let lower = topo.add_edge(Edge::new(
            corners[0],
            corners[1],
            EdgeCurve::NurbsCurve(bulge),
        ));
        let mut edges = vec![OrientedEdge::new(lower, true)];
        for k in 1..4 {
            let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
            edges.push(OrientedEdge::new(e, true));
        }
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let hole = circle(&mut topo, [8.5, 5.1, 0.0], 0.05);
        let face = topo.add_face(Face::new(outer, vec![hole], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// A small patch of a torus, 2e-5 across, with a hole inside it and one
    /// beside it, read at the seam `u = 0` and turned away from it.
    #[test]
    fn a_small_torus_patch_reads_alike_on_its_seam_and_off_it() {
        for u in [0.0, 1.0] {
            let mut topo = Topology::default();
            let patch = torus_loop(&mut topo, (u, 1.0), (1e-5, 1e-5));
            let inside = torus_loop(&mut topo, (u, 1.0), (2e-6, 2e-6));
            let beside = torus_loop(&mut topo, (u + 3e-5, 1.0), (2e-6, 2e-6));
            let face = topo.add_face(Face::new(patch, vec![inside], FaceSurface::Torus(torus())));
            let loops = crate::face_loops::FaceLoops::of(&topo, face, 1e-7)
                .unwrap()
                .unwrap();
            assert_eq!(loops.misplaced_hole(), None, "u {u}");
            let face = topo.add_face(Face::new(
                patch,
                vec![inside, beside],
                FaceSurface::Torus(torus()),
            ));
            let loops = crate::face_loops::FaceLoops::of(&topo, face, 1e-7)
                .unwrap()
                .unwrap();
            assert_eq!(
                loops.misplaced_hole(),
                Some(crate::face_loops::Misplaced::Outside(2)),
                "u {u}"
            );
        }
    }

    /// The quintic bulge on a patch stretched a hundredfold along x: the
    /// drawing stretches a length across the patch far more than along it.
    #[test]
    fn a_hole_under_a_stretched_splines_bulge_is_reported() {
        let control = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(2000.0 * f64::from(i), 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                control,
                vec![vec![1.0; 2]; 2],
            )
            .unwrap(),
        );
        let ys = [5.0, 25.0 / 9.0, 80.0 / 9.0, 10.0 / 9.0, 65.0 / 9.0, 5.0];
        let bulge = brepkit_math::nurbs::curve::NurbsCurve::new(
            5,
            [vec![0.0; 6], vec![1.0; 6]].concat(),
            (0..6)
                .map(|k| Point3::new(200.0 * f64::from(k), ys[k as usize], 0.0))
                .collect(),
            vec![1.0; 6],
        )
        .unwrap();
        let mut topo = Topology::default();
        let corners = points(
            &mut topo,
            &[
                [0.0, 5.0, 0.0],
                [1000.0, 5.0, 0.0],
                [1000.0, 10.0, 0.0],
                [0.0, 10.0, 0.0],
            ],
        );
        let lower = topo.add_edge(Edge::new(
            corners[0],
            corners[1],
            EdgeCurve::NurbsCurve(bulge),
        ));
        let mut edges = vec![OrientedEdge::new(lower, true)];
        for k in 1..4 {
            let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
            edges.push(OrientedEdge::new(e, true));
        }
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let hole = circle(&mut topo, [850.0, 5.1, 0.0], 0.05);
        let face = topo.add_face(Face::new(outer, vec![hole], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    /// A seamless band of a cylinder whose top rim does not read, a rational
    /// circle with every weight negated: the band's open ends are not
    /// inferred without it, and its window passes.
    #[test]
    fn a_band_with_a_rim_that_does_not_read_is_not_reported() {
        let mut topo = Topology::default();
        let bottom = ring_wire(&mut topo, 0.0, 3.0, true);
        let top = {
            let c = ring(10.0, 3.0);
            let n =
                brepkit_geometry::convert::curve_to_nurbs::circle_to_nurbs(&c, 0.0, TAU).unwrap();
            let negated = brepkit_math::nurbs::curve::NurbsCurve::new(
                n.degree(),
                n.knots().to_vec(),
                n.control_points().to_vec(),
                n.weights().iter().map(|w| -w).collect(),
            )
            .unwrap();
            closed(
                &mut topo,
                EdgeCurve::NurbsCurve(negated),
                c.evaluate(0.0),
                false,
            )
        };
        let inside = window(&mut topo, 1.0, (3.0, 7.0), THREE, false);
        let face = topo.add_face(Face::new(bottom, vec![top, inside], cylinder()));
        assert_eq!(found(&topo, face), None);
    }

    /// A flat patch whose `u` runs unevenly along x, a cubic crowded at both
    /// ends: a straight edge across it draws as a curve in `(u, v)`. Holes
    /// just either side of that edge read on their own sides of it.
    #[test]
    fn holes_beside_a_straight_edge_on_an_uneven_patch_read_their_sides() {
        let xs = [0.0, 0.5, 19.5, 20.0];
        let control = xs
            .iter()
            .map(|&x| {
                (0..2)
                    .map(|j| Point3::new(x, 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                3,
                1,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                control,
                vec![vec![1.0; 2]; 4],
            )
            .unwrap(),
        );
        let mut topo = Topology::default();
        let corners = points(
            &mut topo,
            &[[2.0, 2.0, 0.0], [18.0, 2.0, 0.0], [18.0, 18.0, 0.0]],
        );
        let outer = polygon(&mut topo, &corners);
        // The diagonal runs from (2, 2) to (18, 18); a point off it by 0.02.
        let inside = circle(&mut topo, [6.02, 5.98, 0.0], 0.01);
        let outside = circle(&mut topo, [5.98, 6.02, 0.0], 0.01);
        let face = topo.add_face(Face::new(outer, vec![inside], surface.clone()));
        assert_eq!(found(&topo, face), None);
        let face = topo.add_face(Face::new(outer, vec![outside], surface));
        assert!(found(&topo, face).is_some_and(|d| d.contains("outside the outer wire")));
    }

    fn turns(topo: &Topology, face: FaceId) -> Option<String> {
        check_face_wire_turns(topo, face)
            .unwrap()
            .into_iter()
            .next()
            .map(|i| i.description)
    }

    /// A band of a cylinder or a cone whose two rims run the same way round,
    /// as where one of them is used the wrong way, runs round the surface
    /// twice and is reported. The band as built turns no times, and each rim
    /// alone, as an outline and a hole, turns once: neither is reported.
    #[test]
    fn a_band_whose_rims_run_one_way_round_is_reported() {
        // A surface, its radius at a height, and the band's heights.
        type Band<'a> = (FaceSurface, &'a dyn Fn(f64) -> f64, (f64, f64));
        let (cone, on_cone) = cone();
        let surfaces: [Band<'_>; 2] = [
            (cylinder(), THREE, (0.0, 10.0)),
            (cone, &*on_cone, (2.0, 6.0)),
        ];
        for (surface, r, (low, high)) in surfaces {
            let mut topo = Topology::default();
            let outer = band(&mut topo, (low, high), r);
            let face = topo.add_face(Face::new(outer, vec![], surface.clone()));
            assert_eq!(turns(&topo, face), None);
            assert_eq!(
                crate::face_loops::wire_turns(&topo, face).unwrap(),
                Some(vec![Some(Turns::Round(0))])
            );

            let twisted: Vec<OrientedEdge> = topo
                .wire(outer)
                .unwrap()
                .edges()
                .iter()
                .enumerate()
                .map(|(k, oe)| OrientedEdge::new(oe.edge(), oe.is_forward() != (k == 2)))
                .collect();
            let twisted = topo.add_wire(Wire::new(twisted, true).unwrap());
            let face = topo.add_face(Face::new(twisted, vec![], surface.clone()));
            assert!(
                turns(&topo, face).is_some_and(|d| d.contains("2 times")),
                "{:?}",
                turns(&topo, face)
            );

            for (top, bottom) in [(true, false), (true, true)] {
                let outer = ring_wire(&mut topo, high, r(high), top);
                let hole = ring_wire(&mut topo, low, r(low), bottom);
                let face = topo.add_face(Face::new(outer, vec![hole], surface.clone()));
                assert_eq!(turns(&topo, face), None, "{top} {bottom}");
            }
        }
    }

    /// A band of a ring torus between two circles round its axis, joined by
    /// an arc of its tube: as built it turns no times; with the second
    /// circle run the same way as the first, it runs twice along the ring
    /// and is reported.
    #[test]
    fn a_torus_band_whose_rims_run_one_way_round_is_reported() {
        use brepkit_math::surfaces::ToroidalSurface;
        let (big, small) = (5.0, 1.0);
        let torus = || {
            FaceSurface::Torus(
                ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small).unwrap(),
            )
        };
        for twisted in [false, true] {
            let mut topo = Topology::default();
            let latitude = |v: f64| ring(small * v.sin(), big + small * v.cos());
            let (lower, upper) = (latitude(-0.5), latitude(0.5));
            let (a, b) = (lower.evaluate(0.0), upper.evaluate(0.0));
            let va = vertex(&mut topo, [a.x(), a.y(), a.z()]);
            let vb = vertex(&mut topo, [b.x(), b.y(), b.z()]);
            let rim_a = topo.add_edge(Edge::new(va, va, EdgeCurve::Circle(lower)));
            let rim_b = topo.add_edge(Edge::new(vb, vb, EdgeCurve::Circle(upper)));
            // The arc of the tube through both rims' vertices.
            let radial = Vec3::new(a.x(), a.y(), 0.0).normalize().unwrap();
            let centre = Point3::new(0.0, 0.0, 0.0) + radial * big;
            let tube = Circle3D::new(centre, radial.cross(Z), small).unwrap();
            let seam = topo.add_edge(Edge::new(va, vb, EdgeCurve::Circle(tube)));
            let wire = topo.add_wire(
                Wire::new(
                    vec![
                        OrientedEdge::new(rim_a, true),
                        OrientedEdge::new(seam, true),
                        OrientedEdge::new(rim_b, twisted),
                        OrientedEdge::new(seam, false),
                    ],
                    true,
                )
                .unwrap(),
            );
            let face = topo.add_face(Face::new(wire, vec![], torus()));
            let read = crate::face_loops::wire_turns(&topo, face).unwrap();
            if twisted {
                assert_eq!(read, Some(vec![Some(Turns::Torus(2, 0))]));
                assert!(turns(&topo, face).is_some_and(|d| d.contains("2 and 0 times")));
            } else {
                assert_eq!(read, Some(vec![Some(Turns::Torus(0, 0))]));
                assert_eq!(turns(&topo, face), None);
            }
        }
    }
}

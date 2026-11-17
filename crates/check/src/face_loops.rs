//! A curved face's wires drawn in a plane made from its surface's
//! parameters, so that the side of a wire a point lies on, and the way round
//! a wire runs, read on a curved face as on a flat one.
//!
//! Each wire is followed along its edges' own curves, and each point is
//! taken to the surface's `(u, v)` and drawn:
//!
//! - on a cylinder, or a B-spline surface closed in one parameter with no
//!   pole, as an annulus: the angle round the axis and the distance along it
//!   from below the wires, as polar coordinates; the two open ends of the
//!   surface are the origin and the far field;
//! - on a cone, or such a B-spline surface with a pole at one end, as a
//!   disc: polar coordinates with the apex or the pole at the origin;
//! - on a sphere, or such a B-spline surface with a pole at each end, as a
//!   disc: each point at its angle from one pole, round that pole; the other
//!   pole keeps clear of the wires and is drawn as the disc's rim;
//! - on an open B-spline surface, in its `(u, v)`;
//! - on a torus, or a B-spline surface closed in both parameters, in its
//!   `(u, v)` carried on across each seam, a wire that winds the surface
//!   ending a whole period from where it starts.
//!
//! In each drawing but the last, a seam needs nothing: a wire across it is
//! drawn on without a break. A point's side of a wire is then the parity of
//! a ray's crossings of the wire's chords. On a torus a point is read at
//! each of its copies a period apart.
//!
//! Which side of its outline a face lies on is not read from the wires'
//! direction: faces do not keep one convention for it. It is read from the
//! other wires, which all lie on the face's side.

use std::collections::HashMap;
use std::f64::consts::TAU;

use brepkit_math::madd::Madd;
use brepkit_math::region2d::{Boundary2, distance_to_boundary, point_in_region};
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::wire::WireId;

use crate::CheckError;
use crate::properties::boundary::{BoundaryEdge, edge_pieces, wires_of};

mod bound;

use bound::{Chord, Frame, Net, clearance, conic_off};

/// The fractions of an edge at which a wire's points are read against other
/// wires.
const PROBES: [f64; 7] = [0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875];

/// The most times a chord is halved to follow its curve.
const MOST_HALVINGS: u32 = 12;

/// How the drawing stands to the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Drawing {
    /// A disc whose far field is the surface's one open end or edge.
    Disc,
    /// An annulus whose origin and far field are the surface's two open ends
    /// or edges.
    Annulus,
    /// A closed surface drawn as a disc: no point of the drawing is off it.
    Sphere,
    /// A surface closed in both parameters, in its parameters carried on
    /// across each seam, with these periods.
    Torus(Vec2),
}

/// One wire's drawing.
struct Loop {
    /// The chords' points in the wire's direction. The last chord runs back
    /// to the first point, moved on by `shift`.
    points: Vec<Point2>,
    /// The chords as region pieces, the closing chord among them, for a wire
    /// that closes in the drawing.
    pieces: Vec<Boundary2>,
    /// How far a period the wire runs on a torus, or nothing.
    shift: Vec2,
    /// A bound on how far the chords stand off the wire's drawing.
    sag: f64,
    lo: Point2,
    hi: Point2,
    /// The wire's points read against other wires.
    probes: Vec<Point2>,
}

impl Loop {
    fn closes(&self) -> bool {
        self.shift.length() == 0.0
    }

    /// Whether `q` lies farther than `reach` from the chords' box.
    fn clear_of_box(&self, q: Point2, reach: f64) -> bool {
        q.x() < self.lo.x() - reach
            || q.x() > self.hi.x() + reach
            || q.y() < self.lo.y() - reach
            || q.y() > self.hi.y() + reach
    }

    /// Twice the area the closed drawing goes round, positive
    /// counter-clockwise.
    fn twice_area(&self) -> f64 {
        let Some(&o) = self.points.first() else {
            return 0.0;
        };
        let n = self.points.len();
        (0..n)
            .map(|k| {
                let (a, b) = (self.points[k] - o, self.points[(k + 1) % n] - o);
                a.x().madd(b.y(), -(a.y() * b.x()))
            })
            .sum()
    }
}

/// A curved face's wires in its drawing, outline first.
pub struct FaceLoops {
    drawing: Drawing,
    /// Each wire's drawing; `None` for a wire that does not read: one of
    /// seams alone, or one that does not close.
    loops: Vec<Option<Loop>>,
    /// The drawing's linear tolerance.
    reach: f64,
    /// The outline is the surface's seams alone, closed lines of no length:
    /// the face is the whole surface less its holes, which then lies
    /// outside the disc of each hole.
    whole: bool,
}

/// A point's side of a wire, read as the parity of crossings.
type Side = Option<bool>;

impl FaceLoops {
    /// The wires of a face on a cylinder, cone, sphere, ring torus or
    /// B-spline surface, in its drawing. `None` for a plane, a spindle torus,
    /// a cone face across its apex, a sphere face whose wires leave no pole
    /// clear, or a closed B-spline face whose wires reach both its poles.
    /// `tolerance` is the gap within which two edges meet.
    ///
    /// # Errors
    ///
    /// Returns an error if topology lookups fail.
    pub fn of(
        topo: &Topology,
        face_id: FaceId,
        tolerance: f64,
    ) -> Result<Option<Self>, CheckError> {
        let face = topo.face(face_id)?;
        let wire_ids: Vec<WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        let mut wires: Vec<Option<Vec<BoundaryEdge<'_>>>> = Vec::with_capacity(wire_ids.len());
        for &wid in &wire_ids {
            wires.push(wires_of(topo, &[wid])?.and_then(|mut w| w.pop()));
        }
        let all: Vec<Point3> = wires.iter().flatten().flat_map(|w| samples(w)).collect();
        if all.is_empty() {
            return Ok(None);
        }
        let edges: Vec<&BoundaryEdge<'_>> = wires.iter().flatten().flatten().collect();
        let Some(chart) = drawing_of(face.surface(), &all, &edges) else {
            return Ok(None);
        };
        let size = extent_3d(&all);
        // The drawing's extent, read wire by wire along each wire, so that a
        // wire across a seam reads its own extent; and the most the drawing
        // stretches a short length on the surface at the wires' points.
        let mut scale = 0.0_f64;
        for wire in wires.iter().flatten() {
            let mut before = None;
            let drawn: Vec<Point2> = samples(wire)
                .iter()
                .map(|&p| {
                    let q = (chart.draw)(p, before);
                    before = Some(q);
                    q
                })
                .collect();
            scale = scale.max(extent_2d(&drawn));
        }
        let stretch = all.iter().map(|&p| (chart.magnify)(p)).fold(0.0, f64::max);
        let scale = scale.max(f64::MIN_POSITIVE);
        let reach = (tolerance * stretch).max(1e-12 * scale);
        if !(scale.is_finite() && reach.is_finite()) {
            return Ok(None);
        }
        let sizes = Sizes {
            scale,
            reach,
            stretch,
            size,
        };
        let loops = wires
            .iter()
            .map(|w| w.as_ref().and_then(|w| trace(w, &chart, &sizes)))
            .collect();
        let outline = topo.wire(face.outer_wire())?.edges();
        let mut whole = !outline.is_empty();
        for oe in outline {
            let edge = topo.edge(oe.edge())?;
            whole &= edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Line);
        }
        Ok(Some(Self {
            drawing: chart.drawing,
            loops,
            reach,
            whole,
        }))
    }

    /// The side of wire `a` that `q` lies on: `Some(true)` on the side away
    /// from the drawing's far field, which on a torus is the side where a
    /// wire that does not wind the torus holds its disc. `None` where `q`
    /// lies on the wire, the ray does not settle, or the wire does not
    /// read or winds a torus.
    fn side(&self, a: usize, q: Point2) -> Side {
        let l = self.loops.get(a)?.as_ref()?;
        let margin = 2.0 * l.sag + self.reach;
        if let Drawing::Torus(period) = self.drawing {
            if !l.closes() {
                return None;
            }
            let mut inside = false;
            for copy in copies(q, period, l, margin)? {
                if distance_to_boundary(&l.pieces, copy) <= margin {
                    return None;
                }
                inside |= point_in_region(&l.pieces, copy, self.reach)?;
            }
            return Some(inside);
        }
        if l.clear_of_box(q, margin) {
            return Some(false);
        }
        if distance_to_boundary(&l.pieces, q) <= margin {
            return None;
        }
        point_in_region(&l.pieces, q, self.reach)
    }

    /// The side of wire `a` that wire `b` lies on, read at the first of
    /// `b`'s points that settles.
    fn side_of_wire(&self, a: usize, b: usize) -> Side {
        let probes = &self.loops.get(b)?.as_ref()?.probes;
        probes.iter().find_map(|&q| self.side(a, q))
    }

    /// The parity of crossings of the two torus wires `a` and `b` by a path
    /// from `q` to one fixed point clear of both: which of the two bands
    /// between them `q` lies in. Every path between two points crosses the
    /// pair as often, mod 2, so a path that passes a chord's end is
    /// replaced by one through a point aside.
    fn band(&self, (a, b): (usize, usize), q: Point2) -> Side {
        let Drawing::Torus(period) = self.drawing else {
            return None;
        };
        let (la, lb) = (self.loops[a].as_ref()?, self.loops[b].as_ref()?);
        let at = |(fu, fv): (f64, f64)| Point2::new(fu * period.x(), fv * period.y());
        // A point reads its side only clear of each wire's chords by more
        // than they may stand off the wire.
        let clear = |p: Point2, extra: f64| {
            [la, lb].iter().all(|l| {
                let margin = 2.0f64.madd(l.sag, self.reach) + extra;
                copies(p, period, l, margin).is_some_and(|near| {
                    near.iter()
                        .all(|&c| distance_to_boundary(&l.pieces, c) > margin)
                })
            })
        };
        let to = [
            (0.371, 0.613),
            (0.127, 0.883),
            (0.759, 0.241),
            (0.553, 0.067),
        ]
        .into_iter()
        .map(at)
        .find(|&r| clear(r, 100.0 * self.reach))?;
        let from = Point2::new(q.x().rem_euclid(period.x()), q.y().rem_euclid(period.y()));
        if !clear(from, 0.0) {
            return None;
        }
        let count = |p: Point2, r: Point2| {
            Some(
                crossings(la, p, r, period, self.reach)? + crossings(lb, p, r, period, self.reach)?,
            )
        };
        let aside = [(0.211, 0.457), (0.689, 0.811), (0.433, 0.163)].map(at);
        count(from, to)
            .or_else(|| {
                aside
                    .into_iter()
                    .find_map(|w| Some(count(from, w)? + count(w, to)?))
            })
            .map(|n| n % 2 == 1)
    }

    /// The wire's side of each separating wire: on a torus, the wires that
    /// do not wind it, each alone, and two that do, as a pair.
    fn separators(&self) -> (Vec<Separator>, Vec<usize>) {
        let mut separators = Vec::new();
        let mut winding = Vec::new();
        for (k, l) in self.loops.iter().enumerate() {
            match l {
                Some(l) if l.closes() => separators.push(Separator::One(k)),
                Some(_) => winding.push(k),
                None => {}
            }
        }
        if let [a, b] = winding[..] {
            separators.push(Separator::Two(a, b));
        }
        (separators, winding)
    }

    fn separator_side(&self, s: Separator, q: Point2) -> Side {
        match s {
            Separator::One(a) => self.side(a, q),
            Separator::Two(a, b) => self.band((a, b), q),
        }
    }

    /// The first hole that lies off the face: on a side of the outline or of
    /// another hole where the face's other wires are not, beyond the edge of
    /// a surface that is not closed, or, on a disc, outside the outline.
    #[must_use]
    pub fn misplaced_hole(&self) -> Option<Misplaced> {
        let n = self.loops.len();
        let readable: Vec<usize> = (0..n).filter(|&k| self.loops[k].is_some()).collect();
        let (separators, winding) = self.separators();
        // Three wires that wind a torus, which cross no other, cut it into
        // three bands, and a face lies in one band, between two of them.
        if winding.len() > 2 {
            let hole = winding.iter().copied().find(|&k| k > 0)?;
            return Some(Misplaced::Outside(hole));
        }
        if self.drawing == Drawing::Disc && self.loops[0].is_some() {
            for &hole in readable.iter().filter(|&&k| k > 0) {
                if self.side_of_wire(0, hole) == Some(false) {
                    return Some(Misplaced::Outside(hole));
                }
            }
        }
        // Each separator's side that holds the face: the side of all the
        // other wires.
        let mut face_sides: Vec<(Separator, bool)> = Vec::new();
        let mut all_settle = true;
        for &s in &separators {
            let mut sides: Vec<(usize, bool)> = Vec::new();
            for &b in readable.iter().filter(|&&b| !s.holds(b)) {
                let side = self.loops[b]
                    .as_ref()?
                    .probes
                    .iter()
                    .find_map(|&q| self.separator_side(s, q));
                if let Some(side) = side {
                    sides.push((b, side));
                }
            }
            let Some(&(_, first)) = sides.first() else {
                all_settle = false;
                continue;
            };
            let face_side = if s.holds(0) {
                if self.drawing == Drawing::Disc {
                    true
                } else {
                    majority(&sides).unwrap_or(first)
                }
            } else {
                sides
                    .iter()
                    .find(|&&(b, _)| b == 0)
                    .map(|&(_, side)| side)
                    .or_else(|| (self.whole && matches!(s, Separator::One(_))).then_some(false))
                    .or_else(|| majority(&sides))
                    .unwrap_or(first)
            };
            if let Some(&(b, _)) = sides.iter().find(|&&(b, side)| b > 0 && side != face_side) {
                return Some(if s.holds(0) {
                    Misplaced::Outside(b)
                } else {
                    Misplaced::Inside(b, s.first())
                });
            }
            face_sides.push((s, face_side));
        }
        // An open end of the surface on the face's side of every wire lies
        // in the face, which then has no end there. Every wire's side is
        // needed for that: a face with a wire that does not read, or whose
        // side the other wires do not settle, is let be.
        let ends: &[Point2] = match self.drawing {
            Drawing::Disc => &[FAR],
            Drawing::Annulus => &[FAR, Point2::new(0.0, 0.0)],
            Drawing::Sphere | Drawing::Torus(_) => &[],
        };
        if readable.len() == n && all_settle && face_sides.iter().any(|&(s, _)| s.holds(0)) {
            for &end in ends {
                let held = face_sides.iter().all(|&(s, face_side)| {
                    let side = if end == FAR {
                        Some(false)
                    } else {
                        self.separator_side(s, end)
                    };
                    side == Some(face_side)
                });
                if held {
                    let hole = readable.iter().copied().find(|&k| k > 0)?;
                    return Some(Misplaced::Outside(hole));
                }
            }
        }
        None
    }

    /// For each hole, whether it runs the same way round as the outline
    /// about the face: the face lies to the left of one and to the right of
    /// the other. `None` for a hole whose way round does not read: one that
    /// winds a torus, or whose side of the outline, or the outline's of it,
    /// does not settle.
    #[must_use]
    pub fn holes_running_with_outline(&self) -> Vec<Option<bool>> {
        let left_of = |a: usize, other: usize| -> Option<bool> {
            let l = self.loops[a].as_ref()?;
            if !l.closes() {
                return None;
            }
            let face_inside = self.side_of_wire(a, other)?;
            let area = l.twice_area();
            (area != 0.0).then_some((area > 0.0) == face_inside)
        };
        (1..self.loops.len())
            .map(|hole| Some(left_of(0, hole)? != left_of(hole, 0)?))
            .collect()
    }
}

/// Where a hole lies off its face.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Misplaced {
    /// The hole of this index (1 for the first hole) lies outside the
    /// outline.
    Outside(usize),
    /// The first hole lies inside the second.
    Inside(usize, usize),
}

/// A wire, or two wires that wind a torus, which part the surface.
#[derive(Clone, Copy, Debug)]
enum Separator {
    One(usize),
    Two(usize, usize),
}

impl Separator {
    fn holds(self, k: usize) -> bool {
        match self {
            Self::One(a) => a == k,
            Self::Two(a, b) => a == k || b == k,
        }
    }

    fn first(self) -> usize {
        match self {
            Self::One(a) | Self::Two(a, _) => a,
        }
    }
}

/// The far field of a drawing, which no wire reaches.
const FAR: Point2 = Point2::new(f64::INFINITY, f64::INFINITY);

/// The side most of `sides` lie on, or `None` on a tie.
fn majority(sides: &[(usize, bool)]) -> Option<bool> {
    let inside = sides.iter().filter(|&&(_, s)| s).count();
    let outside = sides.len() - inside;
    match inside.cmp(&outside) {
        std::cmp::Ordering::Greater => Some(true),
        std::cmp::Ordering::Less => Some(false),
        std::cmp::Ordering::Equal => None,
    }
}

/// Points along a wire, seventeen to an edge.
fn samples(wire: &[BoundaryEdge<'_>]) -> Vec<Point3> {
    wire.iter()
        .flat_map(|e| {
            (0..=16).map(move |k| {
                let t = (e.to - e.from).madd(f64::from(k) / 16.0, e.from);
                e.curve.evaluate_with_endpoints(t, e.start, e.end)
            })
        })
        .collect()
}

fn extent_3d(points: &[Point3]) -> f64 {
    let Some(&first) = points.first() else {
        return 0.0;
    };
    let (mut lo, mut hi) = (first, first);
    for p in points {
        lo = Point3::new(lo.x().min(p.x()), lo.y().min(p.y()), lo.z().min(p.z()));
        hi = Point3::new(hi.x().max(p.x()), hi.y().max(p.y()), hi.z().max(p.z()));
    }
    (hi - lo).length()
}

fn extent_2d(points: &[Point2]) -> f64 {
    let Some(&first) = points.first() else {
        return 0.0;
    };
    let (mut lo, mut hi) = (first, first);
    for p in points {
        lo = Point2::new(lo.x().min(p.x()), lo.y().min(p.y()));
        hi = Point2::new(hi.x().max(p.x()), hi.y().max(p.y()));
    }
    (hi - lo).length()
}

/// A map from a point on the surface to the drawing. On a torus it carries
/// the point on from the point drawn before it, if any, across each seam.
type Draw<'s> = Box<dyn Fn(Point3, Option<Point2>) -> Point2 + 's>;

/// `(ρ cos θ, ρ sin θ)`.
fn polar(rho: f64, theta: f64) -> Point2 {
    let (s, c) = theta.sin_cos();
    Point2::new(rho * c, rho * s)
}

/// A surface's drawing: how it stands to the surface, the map, how much it
/// stretches a short length on the surface at a point, and, on a B-spline
/// surface, how far the surface strays from straight along a chord of its
/// parameters.
struct Chart<'s> {
    drawing: Drawing,
    draw: Draw<'s>,
    magnify: Box<dyn Fn(Point3) -> f64 + 's>,
    /// On a B-spline surface, how far it strays from straight along a
    /// chord of its parameters; `None` on an analytic surface, whose
    /// drawing bends smoothly.
    bend: Option<Bend<'s>>,
}

/// A B-spline surface's parameters, each read as the fraction of its span,
/// at a point, how far the surface strays from straight along a chord of
/// them, and from that a bound on how far a drawn edge strays from its
/// chord: see [`bound`].
struct Bend<'s> {
    surface: &'s brepkit_math::nurbs::surface::NurbsSurface,
    /// The first value and the length of each parameter's span.
    start: (f64, f64),
    length: (f64, f64),
    /// Which parameters close on themselves, whose moves wrap.
    closed: (bool, bool),
    /// How the drawing stands to the parameters' fractions.
    frame: Frame,
    /// The power of two that brings the most weight to between one and two.
    unit: f64,
    /// The gap across each closed parameter's seam; `None` where the seam's
    /// weights differ.
    seams: [Option<f64>; 2],
    /// Each parameter's knots from the first to the last of its span, as
    /// fractions of the span; empty where the knots are not all finite and
    /// rising, which leaves the surface without a bound.
    fractions: [Vec<f64>; 2],
    /// Each knot cell's net, by the indices of its first knots, read once.
    nets: std::cell::RefCell<HashMap<(usize, usize), Net>>,
}

/// The most knot cells one chord runs through before it counts as
/// unbounded.
const MOST_CELLS: usize = 64;

#[cfg(test)]
thread_local! {
    /// The knot cells that chords ran through, by the indices of their
    /// first knots.
    static CELLS_READ: std::cell::RefCell<std::collections::HashSet<(usize, usize)>> =
        std::cell::RefCell::new(std::collections::HashSet::new());
}

impl Bend<'_> {
    /// How far the surface strays, along its parameters' chord from `from`
    /// by `by` (fractions, wrapped round a closed parameter), from the
    /// straight line between the chord's ends on it. `None` where the chord
    /// runs through more than `MOST_CELLS` knot cells, or a cell's weights
    /// along it are not all positive.
    ///
    /// In each knot cell the surface along the chord is a rational Bezier
    /// curve of the two degrees' sum. Its points come from the cell's net
    /// over the chord's corners, read off by blossoming, and the curve lies
    /// in their hull. The chord's curve runs from end to end, so every
    /// point of the line between its ends lies as near the curve as the
    /// farthest of those points.
    fn stray(&self, from: (f64, f64), by: (f64, f64)) -> Option<f64> {
        let s = self.surface;
        let (u0, v0) = self.start;
        let (pu, pv) = self.length;
        let (ua, va) = (pu.madd(from.0, u0), pv.madd(from.1, v0));
        let (du, dv) = (pu * by.0, pv * by.1);
        // Where the chord crosses a knot line, as fractions of the chord.
        let mut cuts = vec![0.0, 1.0];
        let (nu, nv) = (
            s.control_points().len(),
            s.control_points().first().map_or(0, Vec::len),
        );
        for (knots, (degree, count), (closed, period), (a, d)) in [
            (
                s.knots_u(),
                (s.degree_u(), nu),
                (self.closed.0, pu),
                (ua, du),
            ),
            (
                s.knots_v(),
                (s.degree_v(), nv),
                (self.closed.1, pv),
                (va, dv),
            ),
        ] {
            if d == 0.0 {
                continue;
            }
            let inside = knots.get(degree..=count)?;
            let (lo, hi) = (a.min(a + d), a.max(a + d));
            // A run that wraps crosses the knots a period on as well.
            let shifts: &[f64] = if closed {
                &[-period, 0.0, period]
            } else {
                &[0.0]
            };
            for &shift in shifts {
                let first = inside.partition_point(|&k| k + shift <= lo);
                let last = inside.partition_point(|&k| k + shift < hi);
                if last.saturating_sub(first) > MOST_CELLS {
                    return None;
                }
                cuts.extend(
                    inside
                        .get(first..last.max(first))?
                        .iter()
                        .map(|&k| ((k + shift - a) / d).clamp(0.0, 1.0)),
                );
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        if cuts.len() > MOST_CELLS + 1 {
            return None;
        }
        let mut points: Vec<Point3> = Vec::new();
        for t in cuts.windows(2) {
            let mid = f64::midpoint(t[0], t[1]);
            let (u_ends, iu) = piece_in_span(
                s.knots_u(),
                s.degree_u(),
                nu,
                (self.closed.0, pu),
                (du.madd(t[0], ua), du.madd(t[1], ua), du.madd(mid, ua)),
            )?;
            let (v_ends, iv) = piece_in_span(
                s.knots_v(),
                s.degree_v(),
                nv,
                (self.closed.1, pv),
                (dv.madd(t[0], va), dv.madd(t[1], va), dv.madd(mid, va)),
            )?;
            points.extend(self.surface_along((iu, iv), u_ends, v_ends)?);
        }
        let (a, b) = (*points.first()?, *points.last()?);
        let d = b - a;
        let len2 = d.dot(d);
        let stray = points
            .iter()
            .map(|&c| {
                let f = if len2 > 0.0 {
                    ((c - a).dot(d) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (c - (a + d * f)).length()
            })
            .fold(0.0, f64::max);
        stray.is_finite().then_some(stray)
    }
}

/// A piece of a chord's run in one parameter, `(start, end, middle)`,
/// wrapped into the domain of a closed parameter of period `period`, with
/// the index of the first knot of the knot span it lies in; its ends kept
/// within that span. `None` for a piece in an empty span.
fn piece_in_span(
    knots: &[f64],
    degree: usize,
    count: usize,
    (closed, period): (bool, f64),
    (a, b, mid): (f64, f64, f64),
) -> Option<((f64, f64), usize)> {
    let (first, last) = (*knots.get(degree)?, *knots.get(count)?);
    let shift = if closed {
        ((mid - first) / period).floor() * period
    } else {
        0.0
    };
    let mid = mid - shift;
    let i = degree
        + knots
            .get(degree..count)?
            .partition_point(|&k| k <= mid)
            .saturating_sub(1);
    let (lo, hi) = (*knots.get(i)?, *knots.get(i + 1)?);
    if lo >= hi || !(first..=last).contains(&mid) && !closed {
        return None;
    }
    Some((((a - shift).clamp(lo, hi), (b - shift).clamp(lo, hi)), i))
}

impl Bend<'_> {
    /// The points of the rational Bezier curve that the surface runs along
    /// between parameters `(u.0, v.0)` and `(u.1, v.1)` in the knot cell
    /// whose first knots have indices `(iu, iv)`. `None` where a weight is
    /// not positive.
    ///
    /// The cell's net over the chord's corners holds the blossoms with each
    /// parameter at its start or end, and the curve's point `k` is their
    /// mean over the ways of taking `k` of the curve's arguments at its end:
    /// `B_i^p B_j^q = C(p, i) C(q, j) / C(p + q, i + j) B_(i+j)^(p+q)`.
    fn surface_along(
        &self,
        (iu, iv): (usize, usize),
        u: (f64, f64),
        v: (f64, f64),
    ) -> Option<Vec<Point3>> {
        let s = self.surface;
        let (p, q) = (s.degree_u(), s.degree_v());
        let (ku, kv) = (s.knots_u(), s.knots_v());
        let share = |knots: &[f64], i: usize, (a, b): (f64, f64)| {
            let (lo, hi) = (*knots.get(i)?, *knots.get(i + 1)?);
            Some(((a - lo) / (hi - lo), (b - lo) / (hi - lo)))
        };
        let net = self.net(iu, iv)?;
        let net = bound::sub_net(&net, (p, q), share(ku, iu, u)?, share(kv, iv, v)?);
        if net.iter().any(|c| c[3] <= 0.0) {
            return None;
        }
        let choose = |n: usize, k: usize| {
            (0..k).fold(1.0, |c: f64, i| {
                #[allow(clippy::cast_precision_loss)]
                let (n, i) = ((n - i) as f64, (i + 1) as f64);
                c * n / i
            })
        };
        (0..=p + q)
            .map(|k| {
                let mut h = [0.0_f64; 4];
                for i in k.saturating_sub(q)..=k.min(p) {
                    let f = choose(p, i) * choose(q, k - i) / choose(p + q, k);
                    for (x, y) in h.iter_mut().zip(net[i * (q + 1) + k - i]) {
                        *x = f.madd(y, *x);
                    }
                }
                (h[3] > 0.0).then(|| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
            })
            .collect()
    }
}

/// The least and the most of `along · (p - origin)` over the points `p` of
/// the edges: exact for a line, and over the whole curve for a circle or an
/// ellipse. A B-spline lies in its control points' hull, which bounds it.
/// `None` for a B-spline with a weight that is not positive, or a bound that
/// is not finite.
fn reach(edges: &[&BoundaryEdge<'_>], origin: Point3, along: Vec3) -> Option<(f64, f64)> {
    let mut out = (f64::INFINITY, f64::NEG_INFINITY);
    let mut take = |lo: f64, hi: f64| out = (out.0.min(lo), out.1.max(hi));
    let at = |p: Point3| along.dot(p - origin);
    for e in edges {
        match e.curve {
            EdgeCurve::Line => {
                let (a, b) = (
                    at(e.curve.evaluate_with_endpoints(e.from, e.start, e.end)),
                    at(e.curve.evaluate_with_endpoints(e.to, e.start, e.end)),
                );
                take(a.min(b), a.max(b));
            }
            EdgeCurve::Circle(c) => {
                let mid = at(c.center());
                let half = c.radius() * (along - c.normal() * along.dot(c.normal())).length();
                take(mid - half, mid + half);
            }
            EdgeCurve::Ellipse(c) => {
                let mid = at(c.center());
                let (x, y) = (
                    c.semi_major() * along.dot(c.u_axis()),
                    c.semi_minor() * along.dot(c.v_axis()),
                );
                let half = x.hypot(y);
                take(mid - half, mid + half);
            }
            EdgeCurve::NurbsCurve(n) => {
                if n.weights().iter().any(|&w| w <= 0.0) {
                    return None;
                }
                for &p in n.control_points() {
                    let x = at(p);
                    take(x, x);
                }
            }
        }
    }
    (out.0.is_finite() && out.1.is_finite()).then_some(out)
}

/// The drawing of a face's surface, from points along its wires and the
/// wires' edges.
#[allow(clippy::too_many_lines)]
fn drawing_of<'s>(
    surface: &'s FaceSurface,
    samples: &[Point3],
    edges: &[&BoundaryEdge<'_>],
) -> Option<Chart<'s>> {
    match surface {
        FaceSurface::Plane { .. } => None,
        FaceSurface::Cylinder(s) => {
            // The wires' reach along the axis, which keeps the drawing's
            // radius positive along the whole of each edge.
            let (low, high) = reach(edges, s.origin(), s.axis())?;
            let r = s.radius();
            let base = (high - low).max(r) - low;
            // Along the axis one to one; round it `ρ / r`, at most where
            // the wires reach farthest.
            let most = (high + base).max(r) / r;
            Some(Chart {
                drawing: Drawing::Annulus,
                draw: Box::new(move |p, _| {
                    let (u, v) = s.project_point(p);
                    polar(v + base, u)
                }),
                magnify: Box::new(move |_| most),
                bend: None,
            })
        }
        FaceSurface::Cone(s) => {
            // The side of the apex the wires lie on, by their reach along
            // the axis, which `v` follows: `v` is the reach over `sin a`.
            let (low, high) = reach(edges, s.apex(), s.axis())?;
            let sin_a = s.half_angle().sin();
            let (low, high) = if sin_a < 0.0 {
                (-high, -low)
            } else {
                (low, high)
            };
            let slack = 1e-9 * low.abs().max(high.abs()).max(1.0);
            // A cone flat to its axis has `v` the distance from the apex.
            let sign = if sin_a.abs() <= 1e-15 || low >= -slack {
                1.0
            } else if high <= slack {
                -1.0
            } else {
                return None;
            };
            // Along a generator one to one; round the axis `1 / cos a`.
            let round = 1.0 / s.half_angle().cos();
            Some(Chart {
                drawing: Drawing::Disc,
                draw: Box::new(move |p, _| {
                    let (u, v) = s.project_point(p);
                    polar((sign * v).max(0.0), u)
                }),
                magnify: Box::new(move |_| round.max(1.0)),
                bend: None,
            })
        }
        FaceSurface::Sphere(s) => {
            let center = s.center();
            let radius = s.radius();
            let far = clear_direction(samples, center)?;
            let near = -far;
            // How near the wires come to the far pole, from their edges: the
            // angle `α` from the near pole is at most `π` less that.
            let closest = reach(edges, center, far)?.1 / radius;
            if closest >= 1.0 - 1e-9 {
                return None;
            }
            let farthest = std::f64::consts::PI - closest.max(-1.0).acos();
            let (e1, e2) = brepkit_math::frame::Frame3::from_normal(center, near)
                .map(|f| (f.x, f.y))
                .ok()?;
            let angle_of = move |p: Point3| {
                let w = (p - center).normalize().ok()?;
                let flat = Vec2::new(w.dot(e1), w.dot(e2));
                Some((flat, flat.length().atan2(w.dot(near))))
            };
            Some(Chart {
                drawing: Drawing::Sphere,
                draw: Box::new(move |p, _| {
                    let Some((flat, angle)) = angle_of(p) else {
                        return Point2::new(0.0, 0.0);
                    };
                    let across = flat.length();
                    if across <= 0.0 {
                        return Point2::new(angle, 0.0);
                    }
                    Point2::new(flat.x() * angle / across, flat.y() * angle / across)
                }),
                // Toward the pole `1 / R`; round it `α / (R sin α)`, which
                // grows with `α`.
                magnify: {
                    let round = if farthest > 1e-9 {
                        farthest / farthest.sin()
                    } else {
                        1.0
                    };
                    let most = round.max(1.0) / radius;
                    Box::new(move |_| most)
                },
                bend: None,
            })
        }
        FaceSurface::Torus(t) => {
            let (big, small) = (t.major_radius(), t.minor_radius());
            if small >= big {
                return None;
            }
            // Round the ring `1 / (R + r cos v)`, at most `1 / (R - r)`;
            // round the tube `1 / r`.
            let most = (1.0 / (big - small)).max(1.0 / small);
            Some(Chart {
                drawing: Drawing::Torus(Vec2::new(TAU, TAU)),
                draw: Box::new(move |p, before| carry(t.project_point(p), before, (TAU, TAU))),
                magnify: Box::new(move |_| most),
                bend: None,
            })
        }
        FaceSurface::Nurbs(s) => nurbs_drawing(s, edges),
    }
}

/// `(u, v)` carried on from `before` across each seam of the periods.
fn carry((u, v): (f64, f64), before: Option<Point2>, (pu, pv): (f64, f64)) -> Point2 {
    let Some(b) = before else {
        return Point2::new(u, v);
    };
    let near = |x: f64, from: f64, period: f64| {
        from + (x - from + 0.5 * period).rem_euclid(period) - 0.5 * period
    };
    Point2::new(near(u, b.x(), pu), near(v, b.y(), pv))
}

/// The direction from `center` farthest in angle from every sample, among a
/// spread of directions, if it keeps two degrees clear of them all.
fn clear_direction(samples: &[Point3], center: Point3) -> Option<Vec3> {
    const DIRECTIONS: u32 = 96;
    let units: Vec<Vec3> = samples
        .iter()
        .filter_map(|&p| (p - center).normalize().ok())
        .collect();
    let golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    (0..DIRECTIONS)
        .map(|k| {
            let z = 1.0 - 2.0 * (f64::from(k) + 0.5) / f64::from(DIRECTIONS);
            let r = (1.0 - z * z).max(0.0).sqrt();
            let (s, c) = (golden * f64::from(k)).sin_cos();
            Vec3::new(r * c, r * s, z)
        })
        .map(|d| {
            let nearest = units.iter().fold(-1.0_f64, |m, w| m.max(w.dot(d)));
            (nearest, d)
        })
        .filter(|&(nearest, _)| nearest <= 2.0_f64.to_radians().cos())
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, d)| d)
}

/// The drawing of a B-spline surface: by which parameters it closes on
/// itself, and at which ends of the other it shrinks to a point.
#[allow(clippy::too_many_lines)]
fn nurbs_drawing<'s>(
    s: &'s brepkit_math::nurbs::surface::NurbsSurface,
    edges: &[&BoundaryEdge<'_>],
) -> Option<Chart<'s>> {
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    // A span of the parameters too short to carry a point apart from its
    // ends reads nothing: a drawing on it would divide by next to nothing.
    let usable = |a: f64, b: f64| {
        let span = b - a;
        span.is_finite() && span > 1e-9 * a.abs().max(b.abs()).max(f64::MIN_POSITIVE.sqrt())
    };
    if !usable(u0, u1) || !usable(v0, v1) {
        return None;
    }
    let (closed_u, closed_v) = (s.is_periodic_u(), s.is_periodic_v());
    let rows = s.control_points();
    let same = |points: &[Point3]| {
        points
            .windows(2)
            .all(|w| (w[1] - w[0]).length_squared() < 1e-14)
    };
    // The sides `v = v0` and `v = v1` (rows' first and last points), and
    // `u = u0` and `u = u1` (the first and last rows).
    let pole_v0 = same(
        &rows
            .iter()
            .filter_map(|r| r.first().copied())
            .collect::<Vec<_>>(),
    );
    let pole_v1 = same(
        &rows
            .iter()
            .filter_map(|r| r.last().copied())
            .collect::<Vec<_>>(),
    );
    let pole_u0 = rows.first().is_some_and(|r| same(r));
    let pole_u1 = rows.last().is_some_and(|r| same(r));
    // Each parameter is read as the fraction of its span, so a span far
    // shorter or longer than the other weighs as much in the drawing.
    let (pu, pv) = (u1 - u0, v1 - v0);
    let closed = (closed_u, closed_v);
    // How much a drawing whose `(u, v)` stretch by `factors(u, v)` stretches
    // a short length on the surface: at most the root of
    // `(d1² G22 + d2² G11) / det G`, `G` the surface's metric there. A
    // point where the metric degenerates (a pole) reads nothing.
    let magnify_by = move |factors: Box<dyn Fn(f64, f64) -> (f64, f64) + 's>| {
        Box::new(move |p: Point3| {
            let (u, v) = s.project_point(p);
            let (su, sv) = (s.partial_u(u, v), s.partial_v(u, v));
            let (g11, g12, g22) = (su.dot(su), su.dot(sv), sv.dot(sv));
            let det = g11.madd(g22, -(g12 * g12));
            let (d1, d2) = factors(u, v);
            let m = ((d1 * d1).madd(g22, d2 * d2 * g11) / det).sqrt();
            if det > 0.0 && m.is_finite() { m } else { 0.0 }
        }) as Box<dyn Fn(Point3) -> f64 + 's>
    };
    if closed_u && closed_v {
        return Some(Chart {
            drawing: Drawing::Torus(Vec2::new(1.0, 1.0)),
            draw: Box::new(move |p, before| {
                let (u, v) = s.project_point(p);
                carry(((u - u0) / pu, (v - v0) / pv), before, (1.0, 1.0))
            }),
            magnify: magnify_by(Box::new(move |_, _| (1.0 / pu, 1.0 / pv))),
            bend: Some(Bend::new(s, closed, Frame::Flat)),
        });
    }
    if !closed_u && !closed_v {
        return Some(Chart {
            drawing: Drawing::Disc,
            draw: Box::new(move |p, _| {
                let (u, v) = s.project_point(p);
                Point2::new((u - u0) / pu, (v - v0) / pv)
            }),
            magnify: magnify_by(Box::new(move |_, _| (1.0 / pu, 1.0 / pv))),
            bend: Some(Bend::new(s, closed, Frame::Flat)),
        });
    }
    // Round the closed parameter, out along the other: `(angle, along,
    // span of along, pole at its start, pole at its end)`.
    let (round, span_round, along_span, pole_start, pole_end) = if closed_u {
        (0, (u0, u1), (v0, v1), pole_v0, pole_v1)
    } else {
        (1, (v0, v1), (u0, u1), pole_u0, pole_u1)
    };
    let turn = span_round.1 - span_round.0;
    let parts = move |p: Point3| {
        let (u, v) = s.project_point(p);
        let (a, b) = if round == 0 { (u, v) } else { (v, u) };
        (TAU * (a - span_round.0) / turn, b)
    };
    let (lo, hi) = along_span;
    let width = hi - lo;
    let (drawing, from_start) = match (pole_start, pole_end) {
        (false, false) => (Drawing::Annulus, None),
        (true, false) => (Drawing::Disc, Some(true)),
        (false, true) => (Drawing::Disc, Some(false)),
        (true, true) => {
            // The far pole is drawn as the disc's rim, so no wire may reach
            // it: it is the pole that the edges keep farther from.
            let pole = |at_end: bool| {
                let side: Vec<Point3> = match (round, at_end) {
                    (0, false) => rows.iter().filter_map(|r| r.first().copied()).collect(),
                    (0, true) => rows.iter().filter_map(|r| r.last().copied()).collect(),
                    (_, false) => rows.first().cloned().unwrap_or_default(),
                    (_, true) => rows.last().cloned().unwrap_or_default(),
                };
                side.first().copied()
            };
            let clear = |at_end: bool| {
                pole(at_end)
                    .and_then(|p| clearance(edges, p))
                    .unwrap_or(0.0)
            };
            let (from_lo, from_hi) = (clear(false), clear(true));
            if from_hi > 0.0 && from_hi >= from_lo {
                (Drawing::Sphere, Some(true))
            } else if from_lo > 0.0 {
                (Drawing::Sphere, Some(false))
            } else {
                return None;
            }
        }
    };
    let radius = move |b: f64| match from_start {
        None => (b - lo) / width + 1.0,
        Some(true) => (b - lo) / width,
        Some(false) => (hi - b) / width,
    };
    Some(Chart {
        drawing,
        draw: Box::new(move |p, _| {
            let (angle, b) = parts(p);
            polar(radius(b).max(0.0), angle)
        }),
        // Round the closed parameter `ρ 2π / turn`, along the other
        // `1 / width`.
        magnify: magnify_by(Box::new(move |u, v| {
            let (b, rho_scale) = if round == 0 { (v, 1) } else { (u, 0) };
            let around = radius(b).max(0.0) * TAU / turn;
            if rho_scale == 1 {
                (around, 1.0 / width)
            } else {
                (1.0 / width, around)
            }
        })),
        bend: {
            let (base, sign) = match from_start {
                None => (1.0, 1.0),
                Some(true) => (0.0, 1.0),
                Some(false) => (1.0, -1.0),
            };
            let mut bend = Bend::new(
                s,
                closed,
                Frame::Polar {
                    round,
                    base,
                    sign,
                    pole: None,
                },
            );
            if let Some(start) = from_start {
                bend.frame = Frame::Polar {
                    round,
                    base,
                    sign,
                    pole: bend.pole(round, !start),
                };
            }
            Some(bend)
        },
    })
}

/// The drawing's sizes: its extent, its linear tolerance, the most it
/// stretches a length on the surface, and the wires' extent in space.
struct Sizes {
    scale: f64,
    reach: f64,
    stretch: f64,
    size: f64,
}

/// How far the piece of `curve` from `ta` (at `a`) to `tb` (at `b`) can
/// stray from its chord: its farthest control point, `None` where a weight
/// is not positive or the piece leaves a knot span.
fn hull_off_chord(
    curve: &brepkit_math::nurbs::curve::NurbsCurve,
    ta: f64,
    tb: f64,
    a: Point3,
    b: Point3,
) -> Option<f64> {
    let hull = brepkit_math::nurbs::decompose::curve_bezier_between(curve, ta, tb).ok()?;
    if hull.iter().any(|c| c[3] <= 0.0) {
        return None;
    }
    let d = b - a;
    let len2 = d.dot(d);
    Some(
        hull.iter()
            .map(|&[x, y, z, w]| {
                let c = Point3::new(x / w, y / w, z / w);
                let f = if len2 > 0.0 {
                    ((c - a).dot(d) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (c - (a + d * f)).length()
            })
            .fold(0.0, f64::max),
    )
}

/// A wire's drawing, its chords halved until each follows its curve to
/// within a ten-thousandth of the drawing's `scale`. `None` for a wire
/// that does not close, or closes on a point.
fn trace(wire: &[BoundaryEdge<'_>], chart: &Chart<'_>, sizes: &Sizes) -> Option<Loop> {
    let (draw, drawing) = (&chart.draw, chart.drawing);
    let (scale, reach) = (sizes.scale, sizes.reach);
    let target = 1e-4 * scale;
    let target_3d = 1e-4 * sizes.size;
    let longest = match drawing {
        Drawing::Torus(period) => 0.125 * period.x().min(period.y()),
        _ => f64::INFINITY,
    };
    let mut points: Vec<Point2> = Vec::new();
    let mut sag = 0.0_f64;
    let mut probes = Vec::new();
    for edge in wire {
        let at = |t: f64| edge.curve.evaluate_with_endpoints(t, edge.start, edge.end);
        for f in PROBES {
            probes.push(draw(at((edge.to - edge.from).madd(f, edge.from)), None));
        }
        let mut params: HashMap<u64, ((f64, f64), f64)> = HashMap::new();
        for (ta, tb) in edge_pieces(edge) {
            let start = draw(at(ta), points.last().copied());
            if points
                .last()
                .is_none_or(|&last| (start - last).length() > reach)
            {
                points.push(start);
            }
            // Each stretch is carried on from the point traced last, which
            // is its start: a point carried from farther back can land a
            // period away.
            let mut pending = vec![(ta, tb, 0_u32)];
            while let Some((ta, tb, depth)) = pending.pop() {
                let a = *points.last()?;
                let b = draw(at(tb), Some(a));
                // Each point read against the chord's point at the same
                // fraction, so a drawing that runs unevenly along the chord
                // reads as off it too.
                let off = [0.13, 0.29, 0.5, 0.71, 0.87]
                    .map(|f| {
                        let p = draw(at((tb - ta).madd(f, ta)), Some(a));
                        (p - (a + (b - a) * f)).length()
                    })
                    .into_iter()
                    .fold(0.0_f64, f64::max);
                // A B-spline piece lies within its control points' hull, so
                // their distance from its chord bounds how far it strays
                // between the points read; three points alone do not.
                let hull = match edge.curve {
                    EdgeCurve::NurbsCurve(n) => hull_off_chord(n, ta, tb, at(ta), at(tb)),
                    _ => Some(0.0),
                };
                let off_3d = hull.unwrap_or(0.0);
                let hull_short = hull.is_none() || off_3d > target_3d;
                let halve = off > target || hull_short || (b - a).length() > longest;
                // On a B-spline surface, a bound over the chord on how far
                // the drawn edge strays from it; not read for a chord halved
                // anyway.
                let bound = chart
                    .bend
                    .as_ref()
                    .filter(|_| !halve || depth >= MOST_HALVINGS)
                    .map(|bend| {
                        let mut params_at = |t: f64| {
                            *params
                                .entry(t.to_bits())
                                .or_insert_with(|| bend.params(at(t)))
                        };
                        let (((ua, va), miss_a), ((ub, vb), miss_b)) =
                            (params_at(ta), params_at(tb));
                        let step = |d: f64, closed: bool| {
                            if closed {
                                (d + 0.5).rem_euclid(1.0) - 0.5
                            } else {
                                d
                            }
                        };
                        let chord = Chord {
                            from: (ua, va),
                            by: (step(ub - ua, bend.closed.0), step(vb - va, bend.closed.1)),
                            ends: (a, b),
                            ends_3d: (at(ta), at(tb)),
                            off: off_3d.max(conic_off(edge.curve, ta, tb)),
                            miss: miss_a.max(miss_b),
                        };
                        bend.sag(&chord, target)
                    });
                if (halve || bound.is_some_and(|x| x > target)) && depth < MOST_HALVINGS {
                    let mid = f64::midpoint(ta, tb);
                    // The second half is followed after the first.
                    pending.push((mid, tb, depth + 1));
                    pending.push((ta, mid, depth + 1));
                    continue;
                }
                // A piece whose stray is not bounded leaves the wire unread.
                hull?;
                // An analytic drawing bends smoothly: its chords' points read
                // on the curve bound it.
                let here = bound.unwrap_or_else(|| 2.0f64.madd(off, off_3d * sizes.stretch));
                if !here.is_finite() {
                    return None;
                }
                sag = sag.max(here);
                points.push(b);
            }
        }
    }
    if !points
        .iter()
        .chain(&probes)
        .all(|p| p.x().is_finite() && p.y().is_finite())
    {
        return None;
    }
    let (&first, &last) = (points.first()?, points.last()?);
    let gap = last - first;
    let shift = match drawing {
        Drawing::Torus(period) => Vec2::new(
            (gap.x() / period.x()).round() * period.x(),
            (gap.y() / period.y()).round() * period.y(),
        ),
        _ => Vec2::new(0.0, 0.0),
    };
    if !(shift.x().is_finite() && shift.y().is_finite())
        || (gap - shift).length() > 10.0 * sag + 100.0 * reach
    {
        return None;
    }
    points.pop();
    if points.len() < 2 || extent_2d(&points) <= 10.0 * reach {
        return None;
    }
    let n = points.len();
    let pieces = (0..n)
        .map(|k| {
            let to = if k + 1 == n {
                points[0] + shift
            } else {
                points[k + 1]
            };
            Boundary2::Segment(points[k], to)
        })
        .collect();
    // The box holds the closing chord's end, a shift on from the first
    // point.
    let (mut lo, mut hi) = (first, first);
    for p in points.iter().chain(std::iter::once(&(first + shift))) {
        lo = Point2::new(lo.x().min(p.x()), lo.y().min(p.y()));
        hi = Point2::new(hi.x().max(p.x()), hi.y().max(p.y()));
    }
    Some(Loop {
        points,
        pieces,
        shift,
        sag,
        lo,
        hi,
        probes,
    })
}

/// The most whole periods a range of copies may span: a face's wires span
/// a few periods of its surface at most.
const MOST_PERIODS: f64 = 16.0;

/// The whole numbers `k` with `k p` from `lo` to `hi`. `None` when a bound
/// is not finite or the range spans more than [`MOST_PERIODS`].
#[allow(clippy::cast_possible_truncation)]
fn periods_between(lo: f64, hi: f64, p: f64) -> Option<std::ops::RangeInclusive<i64>> {
    let (first, last) = ((lo / p).ceil(), (hi / p).floor());
    if !(first.is_finite() && last.is_finite()) || last - first > MOST_PERIODS {
        return None;
    }
    Some((first as i64)..=(last as i64))
}

/// The copies of `q` a whole number of periods apart that lie within
/// `margin` of the loop's box. `None` where they do not read: see
/// [`periods_between`].
#[allow(clippy::cast_precision_loss)]
fn copies(q: Point2, period: Vec2, l: &Loop, margin: f64) -> Option<Vec<Point2>> {
    let us = periods_between(
        l.lo.x() - margin - q.x(),
        l.hi.x() + margin - q.x(),
        period.x(),
    )?;
    let vs = periods_between(
        l.lo.y() - margin - q.y(),
        l.hi.y() + margin - q.y(),
        period.y(),
    )?;
    let mut out = Vec::new();
    for m in us {
        for n in vs.clone() {
            out.push(Point2::new(
                (m as f64).madd(period.x(), q.x()),
                (n as f64).madd(period.y(), q.y()),
            ));
        }
    }
    Some(out)
}

/// How often the segment from `from` to `to` crosses the copies of a torus
/// wire's chords a whole number of periods apart. `None` where it passes
/// within `reach` of a chord's end or runs along a chord.
#[allow(clippy::cast_precision_loss)]
fn crossings(l: &Loop, from: Point2, to: Point2, period: Vec2, reach: f64) -> Option<u32> {
    let seg_lo = Point2::new(from.x().min(to.x()), from.y().min(to.y()));
    let seg_hi = Point2::new(from.x().max(to.x()), from.y().max(to.y()));
    let d = to - from;
    let mut count = 0;
    // The box holds the closing chord's end already.
    let (lo, hi) = (l.lo, l.hi);
    let us = periods_between(
        seg_lo.x() - hi.x() - period.x(),
        seg_hi.x() - lo.x() + period.x(),
        period.x(),
    )?;
    let vs = periods_between(
        seg_lo.y() - hi.y() - period.y(),
        seg_hi.y() - lo.y() + period.y(),
        period.y(),
    )?;
    for m in us {
        for n in vs.clone() {
            let offset = Vec2::new(m as f64 * period.x(), n as f64 * period.y());
            for piece in &l.pieces {
                let Boundary2::Segment(a, b) = *piece else {
                    continue;
                };
                let (a, b) = (a + offset, b + offset);
                let e = b - a;
                let denom = d.x().madd(e.y(), -(d.y() * e.x()));
                let w = a - from;
                let s = w.x().madd(e.y(), -(w.y() * e.x()));
                let t = w.x().madd(d.y(), -(w.y() * d.x()));
                if denom.abs() <= f64::EPSILON * d.length() * e.length() {
                    // Parallel: a chord on the segment's line leaves the
                    // count unread.
                    if s.abs() <= reach * e.length() {
                        return None;
                    }
                    continue;
                }
                let (s, t) = (s / denom, t / denom);
                let near_end =
                    |x: f64, len: f64| x.abs() * len <= reach || (1.0 - x).abs() * len <= reach;
                if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&t) {
                    if near_end(t, e.length()) || near_end(s, d.length()) {
                        return None;
                    }
                    count += 1;
                }
            }
        }
    }
    Some(count)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    #[test]
    fn a_range_of_copies_too_long_or_unbounded_reads_nothing() {
        assert_eq!(periods_between(-0.5, 2.5, 1.0), Some(0..=2));
        assert_eq!(periods_between(0.0, 1e9, 1e-3), None);
        assert_eq!(periods_between(0.0, 1.0, 1e-320), None);
        assert_eq!(periods_between(0.0, f64::INFINITY, 1.0), None);
        assert_eq!(periods_between(f64::NAN, 1.0, 1.0), None);
        assert_eq!(periods_between(0.0, 1.0, 0.0), None);
    }

    /// A surface closed in both parameters whose `u` spans 1e-300: its
    /// drawing would divide by that span. The face is not read, and the
    /// validator reports nothing for it.
    #[test]
    fn a_closed_spline_surface_of_no_period_is_not_read() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(1.0, 0.0, 1.0);
        let c = Point3::new(0.0, 1.0, 1.0);
        let d = Point3::new(1.0, 1.0, 2.0);
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 5e-301, 1e-300, 1e-300],
            vec![0.0, 0.0, 0.5, 1.0, 1.0],
            vec![vec![a, b, a], vec![c, d, c], vec![a, b, a]],
            vec![vec![1.0; 3]; 3],
        )
        .unwrap();
        assert!(surface.is_periodic_u() && surface.is_periodic_v());
        let mut topo = Topology::default();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let mut ring = |r: f64| {
            let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z, r).unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
        };
        let (outer, hole) = (ring(5.0), ring(1.0));
        let face = topo.add_face(Face::new(outer, vec![hole], FaceSurface::Nurbs(surface)));
        assert!(FaceLoops::of(&topo, face, 1e-6).unwrap().is_none());
        assert!(
            crate::validate::face_wires::check_face_wires_apart(&topo, face, 1e-6)
                .unwrap()
                .is_empty()
        );
    }

    /// A loop round a torus's outer equator as one closed polynomial
    /// B-spline of a single span, its middle more than half a turn from its
    /// start: its drawing winds the torus once.
    #[test]
    fn a_single_span_spline_round_a_torus_winds_it() {
        let torus =
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0, 2.0)
                .unwrap();
        let degree = 8_usize;
        #[allow(clippy::cast_precision_loss)]
        let control: Vec<Point3> = (0..=degree)
            .map(|k| {
                let s = k as f64 / degree as f64;
                let a = TAU * 0.1f64.madd((std::f64::consts::PI * s).sin(), s);
                Point3::new(20.0 * a.cos(), 20.0 * a.sin(), 0.0)
            })
            .collect();
        let mut knots = vec![0.0; degree + 1];
        knots.extend(vec![1.0; degree + 1]);
        let rim = brepkit_math::nurbs::curve::NurbsCurve::new(
            degree,
            knots,
            control,
            vec![1.0; degree + 1],
        )
        .unwrap();
        let mut topo = Topology::default();
        let start = rim.evaluate(0.0);
        let v = topo.add_vertex(Vertex::new(start, 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(rim)));
        let outer = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let z = Vec3::new(0.0, 0.0, 1.0);
        let top = Circle3D::new(Point3::new(0.0, 0.0, 2.0), z, 6.0).unwrap();
        let w = topo.add_vertex(Vertex::new(top.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(w, w, EdgeCurve::Circle(top)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, false)], true).unwrap());
        let face = topo.add_face(Face::new(outer, vec![hole], FaceSurface::Torus(torus)));
        let loops = FaceLoops::of(&topo, face, 1e-6).unwrap().unwrap();
        let outline = loops.loops[0].as_ref().unwrap();
        assert!(
            (outline.shift.x().abs() - TAU).abs() < 1e-9,
            "{:?}",
            outline.shift
        );
        assert!(outline.shift.y().abs() < 1e-9, "{:?}", outline.shift);
    }

    /// Two tube circles of a torus, at `u = 0.5` and `u = 1.5`, part it into
    /// two bands. Every point of one band reads one side of the pair, also
    /// where its path to the fixed point passes the end of a chord.
    #[test]
    fn each_band_between_two_tube_circles_reads_one_side() {
        let (big, small) = (6.0, 2.0);
        let torus =
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small)
                .unwrap();
        let mut topo = Topology::default();
        let mut tube = |u: f64| {
            let (s, c) = u.sin_cos();
            let circle = Circle3D::new(
                Point3::new(big * c, big * s, 0.0),
                Vec3::new(-s, c, 0.0),
                small,
            )
            .unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
        };
        let (outer, hole) = (tube(0.5), tube(1.5));
        let face = topo.add_face(Face::new(outer, vec![hole], FaceSurface::Torus(torus)));
        let loops = FaceLoops::of(&topo, face, 1e-6).unwrap().unwrap();
        let mut sides = [Vec::new(), Vec::new()];
        for i in 0..40 {
            for j in 0..64 {
                let v = TAU * f64::from(j) / 64.0;
                let (inside, u) = if i < 20 {
                    (0, 0.6 + 0.8 * f64::from(i) / 19.0)
                } else {
                    (1, 1.6 + 4.9 * f64::from(i - 20) / 19.0)
                };
                if let Some(side) = loops.band((0, 1), Point2::new(u, v)) {
                    sides[inside].push(side);
                }
            }
        }
        // A point whose straight path to the fixed point passes the end of
        // a chord of a tube circle.
        for q in [
            Point2::new(1.0, 2.714_428_281_870_906_5),
            Point2::new(1.0, 4.0),
        ] {
            sides[0].push(loops.band((0, 1), q).unwrap());
        }
        for band in &sides {
            assert!(band.len() > 1200, "{}", band.len());
            assert!(band.iter().all(|&s| s == band[0]));
        }
        assert_ne!(sides[0][0], sides[1][0]);
    }

    /// Two wires that wind a torus once each way, as loops of lines, half a
    /// turn apart: a point on either wire, where its chords may stand off
    /// it, reads neither band. Each wire's box holds its closing chord's end.
    #[test]
    fn a_point_on_a_winding_wire_reads_no_band() {
        let torus =
            brepkit_math::surfaces::ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0, 2.0)
                .unwrap();
        let mut topo = Topology::default();
        let mut middles = Vec::new();
        let mut wire = |topo: &mut Topology, shift: f64, keep: bool| {
            let n = 24;
            let corners: Vec<Point3> = (0..n)
                .map(|k| {
                    let t = TAU * f64::from(k) / f64::from(n);
                    torus.evaluate(t + shift, 0.3f64.madd(t.sin(), t))
                })
                .collect();
            let ids: Vec<_> = corners
                .iter()
                .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
                .collect();
            let edges = (0..corners.len())
                .map(|k| {
                    let j = (k + 1) % corners.len();
                    if keep {
                        middles.push(corners[k] + (corners[j] - corners[k]) * 0.5);
                    }
                    OrientedEdge::new(
                        topo.add_edge(Edge::new(ids[k], ids[j], EdgeCurve::Line)),
                        true,
                    )
                })
                .collect();
            topo.add_wire(Wire::new(edges, true).unwrap())
        };
        let outer = wire(&mut topo, 0.0, true);
        let hole = wire(&mut topo, std::f64::consts::PI, false);
        let face = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Torus(torus.clone()),
        ));
        let loops = FaceLoops::of(&topo, face, 1e-6).unwrap().unwrap();
        for l in loops.loops.iter().flatten() {
            assert!(!l.closes());
            let end = l.points[0] + l.shift;
            assert!(end.x() >= l.lo.x() && end.x() <= l.hi.x());
            assert!(end.y() >= l.lo.y() && end.y() <= l.hi.y());
        }
        for m in middles {
            let (u, v) = torus.project_point(m);
            assert_eq!(loops.band((0, 1), Point2::new(u, v)), None, "{m:?}");
        }
    }

    /// A wire's drawing: a polygon round `center` of radius `radius`, with
    /// its chords standing off the wire by up to `sag`.
    fn round_loop(center: Point2, radius: f64, sag: f64) -> Loop {
        let points: Vec<Point2> = (0..64)
            .map(|k| {
                let at = polar(radius, TAU * f64::from(k) / 64.0);
                Point2::new(center.x() + at.x(), center.y() + at.y())
            })
            .collect();
        let pieces = (0..64)
            .map(|k| Boundary2::Segment(points[k], points[(k + 1) % 64]))
            .collect();
        let probes = points.clone();
        Loop {
            points,
            pieces,
            shift: Vec2::new(0.0, 0.0),
            sag,
            lo: center - Vec2::new(radius, radius),
            hi: center + Vec2::new(radius, radius),
            probes,
        }
    }

    /// A band of a cylinder between two rims, drawn round its axis 1e-3
    /// apart. The outer rim's chords may stand off it by 1e-2, so no point
    /// of the inner rim reads a side of it. The band's other end reads
    /// nothing, and the check reports no hole.
    #[test]
    fn a_rim_with_no_settled_side_leaves_the_open_ends_unread() {
        let origin = Point2::new(0.0, 0.0);
        let loops = FaceLoops {
            drawing: Drawing::Annulus,
            loops: vec![
                Some(round_loop(origin, 1.0, 1e-9)),
                Some(round_loop(origin, 1.001, 1e-2)),
            ],
            reach: 1e-9,
            whole: false,
        };
        assert_eq!(loops.side_of_wire(1, 0), None);
        assert_eq!(loops.side_of_wire(0, 1), Some(false));
        assert_eq!(loops.misplaced_hole(), None);
    }

    /// How far a surface strays from the straight line between a chord's
    /// ends, sampled finely along the chord.
    fn sampled_stray(s: &NurbsSurface, from: (f64, f64), to: (f64, f64)) -> f64 {
        let at = |t: f64| s.evaluate((to.0 - from.0) * t + from.0, (to.1 - from.1) * t + from.1);
        let (a, b) = (at(0.0), at(1.0));
        let d = b - a;
        (0..=2000)
            .map(|i| {
                let c = at(f64::from(i) / 2000.0);
                let f = ((c - a).dot(d) / d.dot(d)).clamp(0.0, 1.0);
                (c - (a + d * f)).length()
            })
            .fold(0.0, f64::max)
    }

    fn stray_on(s: &NurbsSurface, from: (f64, f64), to: (f64, f64)) -> Option<f64> {
        let bend = Bend::new(s, (false, false), Frame::Flat);
        bend.stray(from, (to.0 - from.0, to.1 - from.1))
    }

    /// A patch `(20 u, 20 v + f(u), 0)` whose `f''` is nothing at `u` = 0,
    /// 1/4, 1/2, 3/4 and 1 but not between: the derivatives at a grid of
    /// points miss its bend. The stray read from the patch's net holds along
    /// a chord across it, and so does that of a rational patch over two knot
    /// spans, within twice the stray along a short chord.
    #[test]
    fn a_chords_stray_holds_between_points_where_its_bend_is_nothing() {
        // `f'' = 1000 u (u - 1/4) (u - 1/2) (u - 3/4) (u - 1)`, from its
        // roots, then integrated twice and written over the Bernstein
        // polynomials of degree 7.
        let mut second = vec![1000.0];
        for root in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let mut next = vec![0.0; second.len() + 1];
            for (k, c) in second.iter().enumerate() {
                next[k + 1] += c;
                next[k] -= root * c;
            }
            second = next;
        }
        let mut power = [0.0; 8];
        for (k, c) in second.iter().enumerate() {
            power[k + 2] = c / f64::from(u32::try_from((k + 1) * (k + 2)).unwrap());
        }
        let choose =
            |n: usize, k: usize| (0..k).fold(1.0, |acc, i| acc * (n - i) as f64 / (i + 1) as f64);
        let bernstein: Vec<f64> = (0..8)
            .map(|j| {
                (0..=j)
                    .map(|i| choose(j, i) / choose(7, i) * power[i])
                    .sum()
            })
            .collect();
        let control = (0..8)
            .map(|j| {
                (0..2)
                    .map(|l| {
                        Point3::new(
                            20.0 * j as f64 / 7.0,
                            20.0 * f64::from(l) + bernstein[j],
                            0.0,
                        )
                    })
                    .collect()
            })
            .collect();
        let wavy = NurbsSurface::new(
            7,
            1,
            [vec![0.0; 8], vec![1.0; 8]].concat(),
            vec![0.0, 0.0, 1.0, 1.0],
            control,
            vec![vec![1.0; 2]; 8],
        )
        .unwrap();
        let s_uu = |u: f64| wavy.derivatives(u, 0.5, 2)[2][0].length();
        for i in 0..=4 {
            assert!(
                s_uu(f64::from(i) / 4.0) < 1e-9,
                "{i}: {}",
                s_uu(f64::from(i) / 4.0)
            );
        }
        for (from, to) in [((0.0, 0.5), (1.0, 0.5)), ((0.1, 0.2), (0.4, 0.9))] {
            let sampled = sampled_stray(&wavy, from, to);
            let read = stray_on(&wavy, from, to).unwrap();
            assert!(sampled > 1e-3 && read >= sampled, "{read} under {sampled}");
        }

        // A quarter of a tube bent along a quarter circle, rational in both
        // parameters, over two knot spans in `u`.
        let r = std::f64::consts::FRAC_1_SQRT_2;
        let arc = [(1.0, 0.0, 1.0), (1.0, 1.0, r), (0.0, 1.0, 1.0)];
        let ring = |(x, y, w): (f64, f64, f64), radius: f64| {
            arc.map(|(a, b, v)| (Point3::new(x + radius * a, y, radius * b), w * v))
        };
        let rows: Vec<_> = [
            (0.0, 0.0, 1.0),
            (3.0, 2.0, 0.6),
            (6.0, 0.0, 1.4),
            (9.0, 1.0, 1.0),
        ]
        .iter()
        .map(|&c| ring(c, 2.0))
        .collect();
        let tube = NurbsSurface::new(
            2,
            2,
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            rows.iter().map(|r| r.map(|(p, _)| p).to_vec()).collect(),
            rows.iter().map(|r| r.map(|(_, w)| w).to_vec()).collect(),
        )
        .unwrap();
        for (from, to) in [
            ((0.0, 0.0), (1.0, 1.0)),
            ((0.2, 0.9), (0.7, 0.1)),
            ((0.45, 0.3), (0.55, 0.6)),
        ] {
            let sampled = sampled_stray(&tube, from, to);
            let read = stray_on(&tube, from, to).unwrap();
            assert!(read >= sampled, "{from:?} {to:?}: {read} under {sampled}");
        }
        let (from, to) = ((0.3, 0.4), (0.31, 0.42));
        let (sampled, read) = (
            sampled_stray(&tube, from, to),
            stray_on(&tube, from, to).unwrap(),
        );
        assert!(
            sampled > 0.0 && read >= sampled && read <= 2.0 * sampled,
            "{read} against {sampled}"
        );
    }

    /// A flat spline of 40 by 40 knot cells, and a small face with a hole
    /// in one corner of it. Its wires read the bounds of only the cells
    /// they reach.
    #[test]
    fn a_face_reads_only_the_knot_cells_its_wires_reach() {
        let n = 42;
        let mut knots = vec![0.0; 3];
        knots.extend((1..40).map(|k| f64::from(k) / 40.0));
        knots.extend([1.0; 3]);
        // The Greville points of the knots, which draw `u` as itself.
        let greville: Vec<f64> = (0..n)
            .map(|i| f64::midpoint(knots[i + 1], knots[i + 2]))
            .collect();
        let control = greville
            .iter()
            .map(|&x| {
                greville
                    .iter()
                    .map(|&y| Point3::new(40.0 * x, 40.0 * y, 0.0))
                    .collect()
            })
            .collect();
        let sheet =
            NurbsSurface::new(2, 2, knots.clone(), knots, control, vec![vec![1.0; n]; n]).unwrap();
        let mut topo = Topology::default();
        let corners = [[2.0, 2.0], [6.0, 2.0], [6.0, 6.0], [2.0, 6.0]]
            .map(|c| topo.add_vertex(Vertex::new(Point3::new(c[0], c[1], 0.0), 1e-7)));
        let edges = (0..4)
            .map(|k| {
                let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let z = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new(Point3::new(4.0, 4.0, 0.0), z, 1.0).unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, false)], true).unwrap());
        let face = topo.add_face(Face::new(outer, vec![hole], FaceSurface::Nurbs(sheet)));
        CELLS_READ.with(|read| read.borrow_mut().clear());
        let loops = FaceLoops::of(&topo, face, 1e-6).unwrap().unwrap();
        assert!(loops.loops.iter().all(Option::is_some));
        assert_eq!(loops.misplaced_hole(), None);
        let cells = CELLS_READ.with(|read| read.borrow().clone());
        assert!(
            !cells.is_empty() && cells.len() <= 49,
            "{} cells",
            cells.len()
        );
        // The face lies in `u` and `v` from 0.05 to 0.15: in the spans
        // whose first knots are knots 4 to 8. Its outline runs on knot
        // lines, and the box round each chord reaches the spans beside them.
        assert!(
            cells
                .iter()
                .all(|&(iu, iv)| (3..=9).contains(&iu) && (3..=9).contains(&iv)),
            "{cells:?}"
        );
    }

    /// A flat patch `(x(u), 20 v, 0)` whose `x` crawls, within `w` of 10,
    /// through the knot span from `u = 0.495` to `0.505`, with a knot span
    /// `turn` long each side of it to go from its pace to the even pace of
    /// the rest; and its `x(u)` inverted by halving. `x` is quadratic in
    /// `u`, with `dx/du` the polyline through the paces at the knots. With
    /// `turn` nothing, each end of the crawl is a double knot, where `dx/du`
    /// jumps.
    fn crawling_patch(w: f64, turn: f64) -> (NurbsSurface, impl Fn(Point3) -> Point2) {
        let knots = [0.495 - turn, 0.495, 0.505, 0.505 + turn];
        let slow = 2.0 * w / (knots[2] - knots[1]);
        // `x(1) = 20`: the integral of the pace.
        let fast = slow.madd(-(knots[2] - knots[0]), 20.0) / (knots[1] + 1.0 - knots[3]);
        let paces = [fast, fast, slow, slow, fast, fast];
        let all = [
            0.0, 0.0, 0.0, knots[0], knots[1], knots[2], knots[3], 1.0, 1.0, 1.0,
        ];
        let mut xs = vec![0.0];
        for i in 0..6 {
            let last = xs[i];
            xs.push((paces[i] * (all[i + 3] - all[i + 1])).madd(0.5, last));
        }
        let control = xs
            .iter()
            .map(|&x| {
                (0..2)
                    .map(|j| Point3::new(x, 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let patch = NurbsSurface::new(
            2,
            1,
            all.to_vec(),
            vec![0.0, 0.0, 1.0, 1.0],
            control,
            vec![vec![1.0; 2]; 7],
        )
        .unwrap();
        assert!((patch.evaluate(0.5, 0.0).x() - 10.0).abs() < 1e-12);
        assert!((patch.evaluate(1.0, 0.0).x() - 20.0).abs() < 1e-9);
        let along = patch.clone();
        let lift = move |p: Point3| {
            let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
            for _ in 0..200 {
                let mid = f64::midpoint(lo, hi);
                if along.evaluate(mid, 0.0).x() < p.x() {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            Point2::new(f64::midpoint(lo, hi), p.y() / 20.0)
        };
        (patch, lift)
    }

    /// The outline's left side is a parabola whose vertex just touches
    /// `x = 10` on a [`crawling_patch`] with `w = 5e-6`, between the points
    /// at which the drawing's stretch is read: in the drawing it dips across
    /// the crawling span. The outline reads, each point of the drawn
    /// parabola lies within the bound its chords carry, and a point beside
    /// the vertex, on the face, does not read outside it.
    fn a_parabola_into_a_crawl_stays_within_its_chords_bound(turn: f64) {
        let w = 5e-6;
        let (patch, lift) = crawling_patch(w, turn);
        let (bend, vertex) = (0.1_f64, 0.5247_f64);
        let x0 = (4.0 * bend * vertex).madd(vertex, 10.0);
        let x2 = (4.0 * bend * (1.0 - vertex)).madd(1.0 - vertex, 10.0);
        let x1 = 0.5 * (x0 + x2 - 4.0 * bend);
        let rim = brepkit_math::nurbs::curve::NurbsCurve::new(
            2,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![
                Point3::new(x0, 5.0, 0.0),
                Point3::new(x1, 10.0, 0.0),
                Point3::new(x2, 15.0, 0.0),
            ],
            vec![1.0; 3],
        )
        .unwrap();
        let mut topo = Topology::default();
        let corners = [[x0, 5.0], [18.0, 5.0], [18.0, 15.0], [x2, 15.0]]
            .map(|c| topo.add_vertex(Vertex::new(Point3::new(c[0], c[1], 0.0), 1e-7)));
        let mut edges: Vec<OrientedEdge> = (0..3)
            .map(|k| {
                let e = topo.add_edge(Edge::new(corners[k], corners[k + 1], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let e = topo.add_edge(Edge::new(
            corners[0],
            corners[3],
            EdgeCurve::NurbsCurve(rim.clone()),
        ));
        edges.push(OrientedEdge::new(e, false));
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(outer, vec![], FaceSurface::Nurbs(patch)));
        let loops = FaceLoops::of(&topo, face, 1e-7).unwrap().unwrap();
        let beside = lift(Point3::new(
            0.4f64.madd(w, 10.0),
            10.0f64.madd(vertex, 5.0),
            0.0,
        ));
        assert!((beside.x() - 0.502).abs() < 1e-9, "{beside:?}");
        assert_ne!(loops.side(0, beside), Some(false));
        let outline = loops.loops[0].as_ref().unwrap();
        for k in 0..=4000 {
            let drawn = lift(rim.evaluate(f64::from(k) / 4000.0));
            let off = distance_to_boundary(&outline.pieces, drawn);
            assert!(
                off <= outline.sag + loops.reach,
                "{k}: {off} past {}",
                outline.sag
            );
        }
    }

    /// The crawl turns to the even pace over knot spans 5e-5 long, too
    /// short for the drawing's samples to find.
    #[test]
    fn a_parabola_into_a_crawling_knot_span_stays_within_its_chords_bound() {
        a_parabola_into_a_crawl_stays_within_its_chords_bound(5e-5);
    }

    /// The crawl starts and ends at double knots, where `dx/du` jumps
    /// between about 20 and 0.001.
    #[test]
    fn a_parabola_into_a_crawl_between_double_knots_stays_within_its_chords_bound() {
        a_parabola_into_a_crawl_stays_within_its_chords_bound(0.0);
    }

    /// A sphere of radius 5 as a closed B-spline surface with a pole at each
    /// end, and a face on it above the latitude -30 degrees. The face's hole
    /// is a circle 40 degrees across that runs through the north pole, half
    /// way between two of the points read on it. The edges keep clear of
    /// the south pole, which the drawing takes as its rim; the chords next
    /// to the north pole, at the drawing's origin, read their bound from it.
    #[test]
    fn a_hole_through_a_spline_spheres_pole_reads_from_that_pole() {
        let r = 5.0_f64;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let sphere = brepkit_math::surfaces::SphericalSurface::new(origin, r)
            .unwrap()
            .to_rational_nurbs()
            .unwrap();
        let north = Point3::new(0.0, 0.0, r);
        let (tilt, turn) = (20.0_f64.to_radians(), 0.7_f64);
        let toward = |polar: f64, azimuth: f64| {
            Vec3::new(
                polar.sin() * azimuth.cos(),
                polar.sin() * azimuth.sin(),
                polar.cos(),
            )
        };
        let axis = toward(tilt, turn);
        let center = origin + axis * (r * tilt.cos());
        let at_pole = (north - center).normalize().unwrap();
        let t_pole = 4.5 * TAU / 16.0;
        let first = at_pole * t_pole.cos() - axis.cross(at_pole) * t_pole.sin();
        let rim = Circle3D::new_with_ref(center, axis, r * tilt.sin(), first).unwrap();
        assert!((rim.evaluate(t_pole) - north).length() < 1e-12);
        let mut topo = Topology::default();
        let closed = |topo: &mut Topology, circle: Circle3D| {
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap())
        };
        let (s30, c30) = 30.0_f64.to_radians().sin_cos();
        let low = Circle3D::new(
            Point3::new(0.0, 0.0, -r * s30),
            Vec3::new(0.0, 0.0, 1.0),
            r * c30,
        )
        .unwrap();
        let outer = closed(&mut topo, low);
        let hole = closed(&mut topo, rim.clone());
        let face = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Nurbs(sphere.clone()),
        ));
        let loops = FaceLoops::of(&topo, face, 1e-7).unwrap().unwrap();
        assert_eq!(loops.drawing, Drawing::Sphere);
        assert!(loops.loops.iter().all(Option::is_some));
        assert_eq!(loops.misplaced_hole(), None);
        // Drawn from the north pole: out by the angle from it, round by `u`.
        let draw = |p: Point3| {
            let (u, v) = sphere.project_point(p);
            polar(0.5 - v / std::f64::consts::PI, u)
        };
        let inside = draw(origin + axis * r);
        let beyond = draw(origin + toward(50.0_f64.to_radians(), turn + 3.0) * r);
        assert_eq!(loops.side(1, inside), Some(true));
        assert_eq!(loops.side(1, beyond), Some(false));
        assert_eq!(loops.side(0, inside), Some(true));
        let drawn = loops.loops[1].as_ref().unwrap();
        for k in 0..=2000 {
            let p = rim.evaluate(TAU * f64::from(k) / 2000.0);
            let off = distance_to_boundary(&drawn.pieces, draw(p));
            assert!(
                off <= drawn.sag + loops.reach,
                "{k}: {off} past {}",
                drawn.sag
            );
        }
    }

    /// A flat quintic patch whose `u` runs very unevenly along x, crowded at
    /// both ends: a straight edge across it draws as a curve in `(u, v)`.
    /// The drawn edge stays within the bound the chords carry.
    #[test]
    fn a_straight_edge_on_an_uneven_patch_stays_within_its_chords_bound() {
        let xs = [0.0, 0.01, 0.01, 19.99, 19.99, 20.0];
        let control = xs
            .iter()
            .map(|&x| {
                (0..2)
                    .map(|j| Point3::new(x, 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let patch = NurbsSurface::new(
            5,
            1,
            [vec![0.0; 6], vec![1.0; 6]].concat(),
            vec![0.0, 0.0, 1.0, 1.0],
            control,
            vec![vec![1.0; 2]; 6],
        )
        .unwrap();
        let mut topo = Topology::default();
        let corners = [[2.0, 2.0], [18.0, 18.0], [2.0, 18.0]]
            .map(|c| topo.add_vertex(Vertex::new(Point3::new(c[0], c[1], 0.0), 1e-7)));
        let edges = (0..3)
            .map(|k| {
                let e = topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 3], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let outer = topo.add_wire(Wire::new(edges, true).unwrap());
        let z = Vec3::new(0.0, 0.0, 1.0);
        let circle = Circle3D::new(Point3::new(6.0, 12.0, 0.0), z, 1.0).unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let face = topo.add_face(Face::new(
            outer,
            vec![hole],
            FaceSurface::Nurbs(patch.clone()),
        ));
        let loops = FaceLoops::of(&topo, face, 1e-6).unwrap().unwrap();
        let outline = loops.loops[0].as_ref().unwrap();
        for k in 0..=400 {
            let t = f64::from(k) / 400.0;
            let p = Point3::new(2.0 + 16.0 * t, 2.0 + 16.0 * t, 0.0);
            let (u, v) = patch.project_point(p);
            let drawn = Point2::new(u, v);
            let off = distance_to_boundary(&outline.pieces, drawn);
            assert!(
                off <= outline.sag + loops.reach,
                "{t}: {off} past {}",
                outline.sag
            );
        }
    }
}

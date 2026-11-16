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
        let Some((drawing, draw)) = drawing_of(face.surface(), &all) else {
            return Ok(None);
        };
        let size = extent_3d(&all);
        // The drawing's size against the surface's, read wire by wire along
        // each wire, so that a wire across a seam reads its own extent.
        let (mut scale, mut stretch) = (0.0_f64, 0.0_f64);
        for wire in wires.iter().flatten() {
            let points = samples(wire);
            let mut before = None;
            let drawn: Vec<Point2> = points
                .iter()
                .map(|&p| {
                    let q = draw(p, before);
                    before = Some(q);
                    q
                })
                .collect();
            let (flat, solid) = (extent_2d(&drawn), extent_3d(&points));
            scale = scale.max(flat);
            if solid > 0.0 {
                stretch = stretch.max(flat / solid);
            }
        }
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
            .map(|w| w.as_ref().and_then(|w| trace(w, &*draw, drawing, &sizes)))
            .collect();
        let outline = topo.wire(face.outer_wire())?.edges();
        let mut whole = !outline.is_empty();
        for oe in outline {
            let edge = topo.edge(oe.edge())?;
            whole &= edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Line);
        }
        Ok(Some(Self {
            drawing,
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
        // in the face, which then has no end there. The outline's side is
        // needed for that: a face whose outline does not read is let be.
        let ends: &[Point2] = match self.drawing {
            Drawing::Disc => &[FAR],
            Drawing::Annulus => &[FAR, Point2::new(0.0, 0.0)],
            Drawing::Sphere | Drawing::Torus(_) => &[],
        };
        if face_sides.iter().any(|&(s, _)| s.holds(0)) {
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

/// The drawing of a face's surface, from points along its wires.
fn drawing_of<'s>(surface: &'s FaceSurface, samples: &[Point3]) -> Option<(Drawing, Draw<'s>)> {
    match surface {
        FaceSurface::Plane { .. } => None,
        FaceSurface::Cylinder(s) => {
            let vs: Vec<f64> = samples.iter().map(|&p| s.project_point(p).1).collect();
            let low = vs.iter().copied().fold(f64::INFINITY, f64::min);
            let high = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let base = (high - low).max(s.radius()) - low;
            Some((
                Drawing::Annulus,
                Box::new(move |p, _| {
                    let (u, v) = s.project_point(p);
                    polar(v + base, u)
                }),
            ))
        }
        FaceSurface::Cone(s) => {
            let vs: Vec<f64> = samples.iter().map(|&p| s.project_point(p).1).collect();
            let slack = 1e-9 * vs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let sign = if vs.iter().all(|&v| v >= -slack) {
                1.0
            } else if vs.iter().all(|&v| v <= slack) {
                -1.0
            } else {
                return None;
            };
            Some((
                Drawing::Disc,
                Box::new(move |p, _| {
                    let (u, v) = s.project_point(p);
                    polar((sign * v).max(0.0), u)
                }),
            ))
        }
        FaceSurface::Sphere(s) => {
            let center = s.center();
            let far = clear_direction(samples, center)?;
            let near = -far;
            let (e1, e2) = brepkit_math::frame::Frame3::from_normal(center, near)
                .map(|f| (f.x, f.y))
                .ok()?;
            Some((
                Drawing::Sphere,
                Box::new(move |p, _| {
                    let Ok(w) = (p - center).normalize() else {
                        return Point2::new(0.0, 0.0);
                    };
                    let flat = Vec2::new(w.dot(e1), w.dot(e2));
                    let across = flat.length();
                    let angle = across.atan2(w.dot(near));
                    if across <= 0.0 {
                        return Point2::new(angle, 0.0);
                    }
                    Point2::new(flat.x() * angle / across, flat.y() * angle / across)
                }),
            ))
        }
        FaceSurface::Torus(t) => {
            if t.minor_radius() >= t.major_radius() {
                return None;
            }
            Some((
                Drawing::Torus(Vec2::new(TAU, TAU)),
                Box::new(move |p, before| carry(t.project_point(p), before, (TAU, TAU))),
            ))
        }
        FaceSurface::Nurbs(s) => nurbs_drawing(s, samples),
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
fn nurbs_drawing<'s>(
    s: &'s brepkit_math::nurbs::surface::NurbsSurface,
    samples: &[Point3],
) -> Option<(Drawing, Draw<'s>)> {
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
    if closed_u && closed_v {
        return Some((
            Drawing::Torus(Vec2::new(1.0, 1.0)),
            Box::new(move |p, before| {
                let (u, v) = s.project_point(p);
                carry(((u - u0) / pu, (v - v0) / pv), before, (1.0, 1.0))
            }),
        ));
    }
    if !closed_u && !closed_v {
        return Some((
            Drawing::Disc,
            Box::new(move |p, _| {
                let (u, v) = s.project_point(p);
                Point2::new((u - u0) / pu, (v - v0) / pv)
            }),
        ));
    }
    // Round the closed parameter, out along the other: `(angle, along,
    // span of along, pole at its start, pole at its end)`.
    let (round, span_round, along_span, pole_start, pole_end) = if closed_u {
        (0, (u0, u1), (v0, v1), pole_v0, pole_v1)
    } else {
        (1, (v0, v1), (u0, u1), pole_u0, pole_u1)
    };
    let parts = move |p: Point3| {
        let (u, v) = s.project_point(p);
        let (a, b) = if round == 0 { (u, v) } else { (v, u) };
        (TAU * (a - span_round.0) / (span_round.1 - span_round.0), b)
    };
    let (lo, hi) = along_span;
    let width = hi - lo;
    // Which poles the wires come near, read in the parameter along.
    let near = |at: f64| {
        samples
            .iter()
            .any(|&p| (parts(p).1 - at).abs() <= 1e-3 * width)
    };
    let (drawing, from_start) = match (pole_start, pole_end) {
        (false, false) => (Drawing::Annulus, None),
        (true, false) => (Drawing::Disc, Some(true)),
        (false, true) => (Drawing::Disc, Some(false)),
        (true, true) if !near(hi) => (Drawing::Sphere, Some(true)),
        (true, true) if !near(lo) => (Drawing::Sphere, Some(false)),
        (true, true) => return None,
    };
    Some((
        drawing,
        Box::new(move |p, _| {
            let (angle, b) = parts(p);
            let rho = match from_start {
                None => (b - lo) / width + 1.0,
                Some(true) => (b - lo) / width,
                Some(false) => (hi - b) / width,
            };
            polar(rho.max(0.0), angle)
        }),
    ))
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
fn trace(
    wire: &[BoundaryEdge<'_>],
    draw: &dyn Fn(Point3, Option<Point2>) -> Point2,
    drawing: Drawing,
    sizes: &Sizes,
) -> Option<Loop> {
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
                let thirds = [1.0 / 3.0, 0.5, 2.0 / 3.0].map(|f| {
                    let p = draw(at((tb - ta).madd(f, ta)), Some(a));
                    let on = a + (b - a) * f;
                    (p - on).length()
                });
                let off = thirds.iter().fold(0.0_f64, |m, &x| m.max(x));
                // A B-spline piece lies within its control points' hull, so
                // their distance from its chord bounds how far it strays
                // between the points read; three points alone do not.
                let hull = match edge.curve {
                    EdgeCurve::NurbsCurve(n) => hull_off_chord(n, ta, tb, at(ta), at(tb)),
                    _ => Some(0.0),
                };
                let off_3d = hull.unwrap_or(0.0);
                let hull_short = hull.is_none() || off_3d > target_3d;
                if (off > target || hull_short || (b - a).length() > longest)
                    && depth < MOST_HALVINGS
                {
                    let mid = f64::midpoint(ta, tb);
                    // The second half is followed after the first.
                    pending.push((mid, tb, depth + 1));
                    pending.push((ta, mid, depth + 1));
                    continue;
                }
                // A piece whose stray is not bounded leaves the wire unread.
                hull?;
                sag = sag.max(2.0f64.madd(off, off_3d * sizes.stretch));
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
                let a = TAU * 0.1f64.mul_add((std::f64::consts::PI * s).sin(), s);
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
                    torus.evaluate(t + shift, 0.3f64.mul_add(t.sin(), t))
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
}

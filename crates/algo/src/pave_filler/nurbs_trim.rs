//! A NURBS face's pcurves read in its surface's parameters: on which side of
//! the face's wires a point of the surface lies.

use brepkit_math::curves2d::Curve2D;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceId;

use crate::error::AlgoError;

/// How many times a piece of a wire is halved before a point in its box
/// counts as on it: past this the pieces are no longer than the rounding of
/// their coordinates.
const MOST_HALVINGS: u32 = 56;

/// How many pieces one reading of a point may halve.
const MOST_PIECES: usize = 1 << 12;

/// The longest arc of a circle or an ellipse that one piece takes, in
/// radians.
const ARC_STEP: f64 = std::f64::consts::FRAC_PI_4;

/// How many times the pieces of a face's pcurves are halved, at most, to
/// bound how far their images stand from the face's edges.
const IMAGE_HALVINGS: usize = 256;

/// How many parts of the loops' box of parameters one proof that a point
/// lies off a face may read ([`NurbsFaceTrim::clear_of`]).
const MOST_PARTS: usize = 512;

/// A pcurve run over its parameters from the first to the second, and the
/// place in a trim's list of the edge whose pcurve it is.
type Run<'a> = (&'a Curve2D, (f64, f64), usize);

/// A box in a surface's parameters.
#[derive(Clone, Copy)]
struct Box2 {
    lo: [f64; 2],
    hi: [f64; 2],
}

impl Box2 {
    fn of(points: impl IntoIterator<Item = Point2>) -> Self {
        points.into_iter().fold(
            Self {
                lo: [f64::INFINITY; 2],
                hi: [f64::NEG_INFINITY; 2],
            },
            |b, p| Self {
                lo: [b.lo[0].min(p.x()), b.lo[1].min(p.y())],
                hi: [b.hi[0].max(p.x()), b.hi[1].max(p.y())],
            },
        )
    }

    fn grown(self, by: f64) -> Self {
        Self {
            lo: [self.lo[0] - by, self.lo[1] - by],
            hi: [self.hi[0] + by, self.hi[1] + by],
        }
    }
}

/// What holds a piece of a wire between its two ends.
#[derive(Clone)]
enum Hull {
    /// The straight segment between the ends.
    Segment,
    /// An arc of a circle or an ellipse over the parameters `t`, whose
    /// second derivative is never longer than `bend`: the arc lies within
    /// `bend Δt² / 8` of the chord between its ends.
    Conic {
        curve: Curve2D,
        t: (f64, f64),
        bend: f64,
    },
    /// A Bezier piece of a B-spline by its control points `[w x, w y, w]`,
    /// each weight positive: the piece lies in the box of the points.
    Bezier(Vec<[f64; 3]>),
}

/// A piece of a wire from `a` to `b`. Two pieces that meet share the one
/// computed point there, so the pieces of a wire run round a closed curve.
#[derive(Clone)]
struct Piece {
    a: Point2,
    b: Point2,
    hull: Hull,
    /// The edges the piece runs by, by their place in the trim's list: one
    /// edge twice for a piece of its pcurve, or the two edges a segment
    /// joins the pcurves of.
    owner: [usize; 2],
}

impl Piece {
    /// A box that holds the piece and its ends, grown by `slack`.
    fn bounds(&self, slack: f64) -> Box2 {
        let ends = Box2::of([self.a, self.b]);
        let held = match &self.hull {
            Hull::Segment => ends,
            Hull::Conic { t, bend, .. } => {
                let w = t.1 - t.0;
                ends.grown(bend * w * w / 8.0)
            }
            Hull::Bezier(net) => Box2::of(
                net.iter()
                    .map(|q| Point2::new(q[0] / q[2], q[1] / q[2]))
                    .chain([self.a, self.b]),
            ),
        };
        held.grown(slack)
    }

    /// The piece run the other way.
    fn reversed(&self) -> Self {
        Self {
            owner: [self.owner[1], self.owner[0]],
            a: self.b,
            b: self.a,
            hull: match &self.hull {
                Hull::Segment => Hull::Segment,
                Hull::Conic { curve, t, bend } => Hull::Conic {
                    curve: curve.clone(),
                    t: (t.1, t.0),
                    bend: *bend,
                },
                Hull::Bezier(net) => Hull::Bezier(net.iter().rev().copied().collect()),
            },
        }
    }

    /// The piece cut in two at the middle of its parameters.
    fn halves(&self) -> (Self, Self) {
        match &self.hull {
            Hull::Segment => {
                let m = Point2::new(
                    f64::midpoint(self.a.x(), self.b.x()),
                    f64::midpoint(self.a.y(), self.b.y()),
                );
                (
                    Self {
                        owner: self.owner,
                        a: self.a,
                        b: m,
                        hull: Hull::Segment,
                    },
                    Self {
                        owner: self.owner,
                        a: m,
                        b: self.b,
                        hull: Hull::Segment,
                    },
                )
            }
            Hull::Conic { curve, t, bend } => {
                let tm = f64::midpoint(t.0, t.1);
                let m = curve.evaluate(tm);
                (
                    Self {
                        owner: self.owner,
                        a: self.a,
                        b: m,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (t.0, tm),
                            bend: *bend,
                        },
                    },
                    Self {
                        owner: self.owner,
                        a: m,
                        b: self.b,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (tm, t.1),
                            bend: *bend,
                        },
                    },
                )
            }
            Hull::Bezier(net) => {
                let mut row = net.clone();
                let mut left = vec![row[0]];
                let mut right = vec![row[row.len() - 1]];
                for level in 1..row.len() {
                    for j in 0..row.len() - level {
                        let (p, q) = (row[j], row[j + 1]);
                        row[j] = [
                            f64::midpoint(p[0], q[0]),
                            f64::midpoint(p[1], q[1]),
                            f64::midpoint(p[2], q[2]),
                        ];
                    }
                    left.push(row[0]);
                    right.push(row[row.len() - 1 - level]);
                }
                right.reverse();
                let q = row[0];
                let m = Point2::new(q[0] / q[2], q[1] / q[2]);
                (
                    Self {
                        owner: self.owner,
                        a: self.a,
                        b: m,
                        hull: Hull::Bezier(left),
                    },
                    Self {
                        owner: self.owner,
                        a: m,
                        b: self.b,
                        hull: Hull::Bezier(right),
                    },
                )
            }
        }
    }
}

/// A NURBS face's wires in its surface's parameters, each the closed run of
/// its edges' pcurves ([`super::helpers::pcurve_loops`]), the end of each
/// joined to the start of the next by a segment. On a surface whose sides
/// keep apart, a point of the face is a point of the surface whose
/// parameters lie inside the loops.
///
/// A point's side is read by a ray from its parameters toward growing `v`:
/// each piece of a loop whose box lies past the point along `v` and across
/// its `u` is crossed an odd number of times where its ends lie on either
/// side of the point's `u`, and one whose box misses the ray is crossed by
/// it nowhere. A piece whose box holds the point is halved. Each box holds
/// its piece's own curve and the rounded points the pieces run through, so
/// the loops of the rounded points wind round the point as the face's wires
/// do, wherever the point lies outside the boxes.
///
/// The loops are the pcurves' images on the surface, which stand off the
/// face's edges by as much as the pcurves stray from them: a pcurve need
/// meet its edge only at its ends. How far each image can stand from its
/// edge is bounded from the surface's control points over the boxes of the
/// pcurve's pieces ([`Self::image_reaches`]), and a point outside the loops
/// is off the face only farther than that from every edge.
pub(super) struct NurbsFaceTrim {
    surface: NurbsSurface,
    pieces: Vec<Piece>,
    /// How far a computed point of a piece, a control point of a half or the
    /// end of a run can stand from where it lies exactly.
    slack: f64,
    /// The face's edges, by the place the pieces' owners name.
    edges: Vec<TrimEdge>,
    /// How far the image of each edge's pcurve can stand from the edge, read
    /// the first time a point is read outside the loops; `None` where it
    /// cannot be bounded.
    image_reaches: std::cell::OnceCell<Option<Vec<f64>>>,
    /// The box of the loops' parameters, which holds the parameters of every
    /// point of the face.
    bounds: Box2,
}

/// An edge of a face read by its trim.
struct TrimEdge {
    curve: EdgeCurve,
    start: Point3,
    end: Point3,
    /// How near the edge a point counts as on it ([`super::phase_ef::edge_reach`]).
    reach: f64,
}

impl NurbsFaceTrim {
    /// The trim of face `face_id` on `surface`. `None` on a periodic
    /// surface, or where [`super::helpers::pcurve_loops`] reads no loops
    /// within `reach`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if a topology lookup fails.
    pub(super) fn new(
        topo: &Topology,
        face_id: FaceId,
        surface: &NurbsSurface,
        tol: Tolerance,
        reach: f64,
    ) -> Result<Option<Self>, AlgoError> {
        if surface.is_periodic_u() || surface.is_periodic_v() {
            return Ok(None);
        }
        let Some(loops) = super::helpers::pcurve_loops(topo, face_id, surface, tol, reach)? else {
            return Ok(None);
        };
        // An edge run twice, out and back as along a slit, leaves its loop
        // no simple curve.
        let mut seen = std::collections::HashSet::new();
        if !loops
            .loops
            .iter()
            .flatten()
            .all(|(_, _, eid)| seen.insert(*eid))
        {
            return Ok(None);
        }
        let mut edges = Vec::new();
        let mut runs: Vec<Vec<Run<'_>>> = Vec::new();
        for wire in &loops.loops {
            let mut run = Vec::with_capacity(wire.len());
            for (pcurve, forward, eid) in wire {
                let edge = topo.edge(*eid)?;
                let (sv, ev) = (topo.vertex(edge.start())?, topo.vertex(edge.end())?);
                edges.push(TrimEdge {
                    curve: edge.curve().clone(),
                    start: sv.point(),
                    end: ev.point(),
                    reach: super::phase_ef::edge_reach(edge, sv, ev, tol),
                });
                let along = if *forward {
                    (pcurve.t_start(), pcurve.t_end())
                } else {
                    (pcurve.t_end(), pcurve.t_start())
                };
                run.push((pcurve.curve(), along, edges.len() - 1));
            }
            runs.push(run);
        }
        Ok(Self::of_runs(surface, &runs, edges))
    }

    /// The trim on `surface` of loops each of which runs along its curves
    /// over their parameters, in order, each curve the pcurve of the edge
    /// in `edges` that it names. `None` where a B-spline's piece cannot be
    /// read, or the loops hold no piece.
    fn of_runs(
        surface: &NurbsSurface,
        loops: &[Vec<Run<'_>>],
        edges: Vec<TrimEdge>,
    ) -> Option<Self> {
        let mut pieces: Vec<Piece> = Vec::new();
        let mut spans: Vec<(usize, usize)> = Vec::with_capacity(loops.len());
        let (mut size, mut degree) = (0.0_f64, 2_usize);
        for wire in loops {
            let first = pieces.len();
            for &(curve, run, edge) in wire {
                let mut found = pcurve_pieces(curve, run, &mut size, &mut degree)?;
                for piece in &mut found {
                    piece.owner = [edge; 2];
                }
                if let (Some(last), Some(next)) =
                    (pieces.get(first..).and_then(<[_]>::last), found.first())
                {
                    pieces.push(Piece {
                        owner: [last.owner[1], edge],
                        a: last.b,
                        b: next.a,
                        hull: Hull::Segment,
                    });
                }
                pieces.extend(found);
            }
            if let (Some(start), Some(end)) = (pieces.get(first), pieces.last()) {
                let (a, b) = (end.b, start.a);
                pieces.push(Piece {
                    owner: [end.owner[1], start.owner[0]],
                    a,
                    b,
                    hull: Hull::Segment,
                });
            }
            spans.push((first, pieces.len()));
        }
        if pieces.is_empty() || !size.is_finite() {
            return None;
        }
        // A halving rounds each coordinate of a net it blends by half a unit
        // of the net's largest; a projection divides by a weight, which a
        // net's sizes carry as the ratio of its weights; and a point of a
        // line or a conic is a few products and sums.
        #[allow(clippy::cast_precision_loss)]
        let roundings = (8 * (MOST_HALVINGS as usize + 4) * (degree + 2)) as f64;
        let slack = roundings * f64::EPSILON * size.max(f64::MIN_POSITIVE);
        let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
        let joint = 2e-6 * ((u1 - u0) + (v1 - v0)) + 4.0 * slack;
        if !loops_are_simple(&pieces, &spans, slack, joint) {
            return None;
        }
        let bounds = pieces
            .iter()
            .map(|piece| piece.bounds(slack))
            .reduce(|x, y| Box2 {
                lo: [x.lo[0].min(y.lo[0]), x.lo[1].min(y.lo[1])],
                hi: [x.hi[0].max(y.hi[0]), x.hi[1].max(y.hi[1])],
            })?;
        Some(Self {
            surface: surface.clone(),
            pieces,
            slack,
            edges,
            image_reaches: std::cell::OnceCell::new(),
            bounds,
        })
    }

    /// Whether `p`, a point within `near` of the face's surface, lies on the
    /// face: `Some(true)` where its parameters, where the surface's point
    /// there lies within `near` of it, lie inside the loops; `Some(false)`
    /// where they lie outside them, `p` lies farther from every edge than
    /// the edge's reach, how far its pcurve's image can stand from it, and
    /// how far `p` lies from the surface, and no point of the surface over
    /// the loops lies within the tolerance and that distance of `p`
    /// ([`Self::clear_of`]): a surface that runs back by `p` (a hairpin) can
    /// give it parameters off the face though the face runs by it too.
    /// `None` where no such parameters are found, they lie in the box of a
    /// piece of a loop halved as far as it goes, an image's reach cannot be
    /// bounded, `p` lies within those of an edge, or the surface over the
    /// loops cannot be proved clear of it.
    pub(super) fn holds(&self, p: Point3, near: f64, tol: Tolerance) -> Option<bool> {
        let foot =
            brepkit_math::nurbs::projection::project_point_to_surface(&self.surface, p, tol.linear)
                .ok()?;
        let off = (self.surface.evaluate(foot.u, foot.v) - p).length();
        if off.is_nan() || off > near {
            return None;
        }
        if self.holds_parameters(Point2::new(foot.u, foot.v))? {
            return Some(true);
        }
        let images = self
            .image_reaches
            .get_or_init(|| self.image_reaches(tol))
            .as_ref()?;
        for (edge, image) in self.edges.iter().zip(images) {
            let reach = edge.reach + image + off;
            if super::phase_ef::near_the_edge(&edge.curve, (edge.start, edge.end), p, reach, tol)? {
                return None;
            }
        }
        self.clear_of(p, tol.linear + off).then_some(false)
    }

    /// Whether every point of the surface over the loops lies farther than
    /// `reach` from `p`. The box of the loops' parameters is halved, its
    /// longer side first, and a part is dropped where the hull of the
    /// surface's control points cut to it ([`super::helpers::cut_net_points`])
    /// keeps farther than `reach` from `p` past a plane ([`parted`]), or where
    /// the part lies outside the loops ([`Self::outside_loops`]); a part that
    /// holds no area of the domain holds no point. Each plane is read with
    /// the rounding of the cut and of the coordinates' size. `false` where a
    /// part is left after [`MOST_PARTS`] are read.
    fn clear_of(&self, p: Point3, reach: f64) -> bool {
        let rounding = super::helpers::net_rounding(&self.surface)
            + 64.0 * f64::EPSILON * (p.x().abs() + p.y().abs() + p.z().abs());
        let mut parts = vec![self.bounds];
        let mut read = 0_usize;
        while let Some(part) = parts.pop() {
            if read == MOST_PARTS {
                return false;
            }
            read += 1;
            let range = [part.lo[0], part.hi[0], part.lo[1], part.hi[1]];
            let Some(points) = super::helpers::cut_net_points(&self.surface, range) else {
                continue;
            };
            if parted(&points, p, reach + rounding) || self.outside_loops(part) {
                continue;
            }
            let (du, dv) = (part.hi[0] - part.lo[0], part.hi[1] - part.lo[1]);
            let (first, second) = if du >= dv {
                let mid = f64::midpoint(part.lo[0], part.hi[0]);
                (
                    Box2 {
                        lo: part.lo,
                        hi: [mid, part.hi[1]],
                    },
                    Box2 {
                        lo: [mid, part.lo[1]],
                        hi: part.hi,
                    },
                )
            } else {
                let mid = f64::midpoint(part.lo[1], part.hi[1]);
                (
                    Box2 {
                        lo: part.lo,
                        hi: [part.hi[0], mid],
                    },
                    Box2 {
                        lo: [part.lo[0], mid],
                        hi: part.hi,
                    },
                )
            };
            parts.extend([first, second]);
        }
        true
    }

    /// Whether every point of the box of parameters `part` lies outside the
    /// loops: no piece of a loop meets it, halved where its box does, so the
    /// loops wind round each point of the part alike, and its middle reads
    /// outside them.
    fn outside_loops(&self, part: Box2) -> bool {
        let mut budget = MOST_PIECES;
        for piece in &self.pieces {
            if !self.misses(piece, part, &mut budget, 0) {
                return false;
            }
        }
        let middle = Point2::new(
            f64::midpoint(part.lo[0], part.hi[0]),
            f64::midpoint(part.lo[1], part.hi[1]),
        );
        self.holds_parameters(middle) == Some(false)
    }

    /// Whether `piece` keeps out of the box `part`, halved where its box
    /// meets it.
    fn misses(&self, piece: &Piece, part: Box2, budget: &mut usize, depth: u32) -> bool {
        let b = piece.bounds(self.slack);
        if b.lo[0] > part.hi[0]
            || b.hi[0] < part.lo[0]
            || b.lo[1] > part.hi[1]
            || b.hi[1] < part.lo[1]
        {
            return true;
        }
        if depth >= MOST_HALVINGS || *budget == 0 {
            return false;
        }
        *budget -= 1;
        let (left, right) = piece.halves();
        self.misses(&left, part, budget, depth + 1) && self.misses(&right, part, budget, depth + 1)
    }

    /// How far the image on the surface of each edge's pcurve can stand
    /// from the edge, and the segments that join the pcurves from either
    /// edge they join. Over the box of a piece of a pcurve, the surface lies
    /// in the hull of its control points cut to the box
    /// ([`super::helpers::cut_net_points`]), and the distance to a segment
    /// is convex, so it is no more than its most at those points: to a line
    /// edge itself, and to the chord of a circle or an ellipse between the
    /// places nearest the piece's ends, within the edge's range, which the
    /// arc between them stands off by no more than its second derivative's
    /// bound times the span squared over eight. Each bound is grown by the
    /// rounding of the cut and of the distances. The piece with the widest
    /// bound is halved until each is within half the tolerance, or
    /// [`IMAGE_HALVINGS`] halvings are spent. `None` where an edge is a
    /// B-spline, or a piece's box holds no area of the surface's domain.
    fn image_reaches(&self, tol: Tolerance) -> Option<Vec<f64>> {
        let bound = |piece: &Piece| -> Option<f64> {
            let [a, b] = piece.owner;
            let first = self.image_bound(piece, self.edges.get(a)?)?;
            if a == b {
                return Some(first);
            }
            Some(first.min(self.image_bound(piece, self.edges.get(b)?)?))
        };
        let mut bounded: Vec<(f64, Piece)> = self
            .pieces
            .iter()
            .map(|piece| Some((bound(piece)?, piece.clone())))
            .collect::<Option<_>>()?;
        for _ in 0..IMAGE_HALVINGS {
            let Some((widest, &(most, _))) = bounded
                .iter()
                .enumerate()
                .max_by(|x, y| x.1.0.total_cmp(&y.1.0))
            else {
                break;
            };
            if most <= 0.5 * tol.linear {
                break;
            }
            let (_, piece) = bounded.swap_remove(widest);
            let (left, right) = piece.halves();
            bounded.push((bound(&left)?, left));
            bounded.push((bound(&right)?, right));
        }
        let mut reaches = vec![0.0_f64; self.edges.len()];
        for (most, piece) in &bounded {
            for &e in &piece.owner {
                reaches[e] = reaches[e].max(*most);
            }
        }
        Some(reaches)
    }

    /// A bound on how far a point of the surface over the box of `piece`
    /// can stand from `edge` ([`Self::image_reaches`]).
    fn image_bound(&self, piece: &Piece, edge: &TrimEdge) -> Option<f64> {
        use std::f64::consts::TAU;
        let b = piece.bounds(self.slack);
        // A piece along `u` or `v` has a box of no width but its slack.
        let wider = |x: f64| 4.0 * f64::EPSILON * (1.0 + x.abs());
        let range = [
            b.lo[0] - wider(b.lo[0]),
            b.hi[0] + wider(b.hi[0]),
            b.lo[1] - wider(b.lo[1]),
            b.hi[1] + wider(b.hi[1]),
        ];
        let points = super::helpers::cut_net_points(&self.surface, range)?;
        let size = |q: Point3| q.x().abs() + q.y().abs() + q.z().abs();
        let farthest = |a: Point3, z: Point3| {
            points
                .iter()
                .map(|&q| super::phase_ef::point_to_segment(q, a, z))
                .fold(0.0_f64, f64::max)
        };
        let ends = [piece.a, piece.b].map(|q| self.surface.evaluate(q.x(), q.y()));
        let (far, scale) = match &edge.curve {
            EdgeCurve::Line => (
                farthest(edge.start, edge.end),
                size(edge.start).max(size(edge.end)),
            ),
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {
                let (bend, scale) = match &edge.curve {
                    EdgeCurve::Circle(c) => (
                        super::phase_ef::conic_reach(
                            c.radius(),
                            c.radius(),
                            c.u_axis(),
                            c.v_axis(),
                        ),
                        size(c.center()) + c.radius(),
                    ),
                    EdgeCurve::Ellipse(e) => (
                        super::phase_ef::conic_reach(
                            e.semi_major(),
                            e.semi_minor(),
                            e.u_axis(),
                            e.v_axis(),
                        ),
                        size(e.center()) + e.semi_major().max(e.semi_minor()),
                    ),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => return None,
                };
                let at = |t: f64| match &edge.curve {
                    EdgeCurve::Circle(c) => c.evaluate(t),
                    EdgeCurve::Ellipse(e) => e.evaluate(t),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => edge.start,
                };
                let place = |q: Point3| match &edge.curve {
                    EdgeCurve::Circle(c) => c.project(q),
                    EdgeCurve::Ellipse(e) => e.project(q),
                    EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => 0.0,
                };
                let (t0, t1) = edge.curve.domain_with_endpoints(edge.start, edge.end);
                let (lo, hi) = (t0.min(t1), t0.max(t1));
                let whole = hi - lo >= TAU * (1.0 - 1e-12);
                // A place within the arc: on a whole turn any place, else
                // the place round from the arc's start, or the nearer end.
                let within = |t: f64| {
                    if whole {
                        return t;
                    }
                    let t = lo + (t - lo).rem_euclid(TAU);
                    if t <= hi {
                        t
                    } else if t - hi <= lo + TAU - t {
                        hi
                    } else {
                        lo
                    }
                };
                let (s0, mut s1) = (within(place(ends[0])), within(place(ends[1])));
                if whole {
                    let turn = s1 - s0;
                    s1 = s0 + (turn - (turn / TAU).round() * TAU);
                }
                let span = s1 - s0;
                (farthest(at(s0), at(s1)) + bend * span * span / 8.0, scale)
            }
            EdgeCurve::NurbsCurve(_) => return None,
        };
        let largest = points.iter().map(|&q| size(q)).fold(scale, f64::max);
        Some(far + super::helpers::net_rounding(&self.surface) + 64.0 * f64::EPSILON * largest)
    }

    /// Whether the parameters `q` lie inside the loops, or `None` where they
    /// lie in the box of a piece halved as far as it goes.
    fn holds_parameters(&self, q: Point2) -> Option<bool> {
        let mut budget = MOST_PIECES;
        let mut odd = false;
        for piece in &self.pieces {
            odd ^= self.crossed(piece, q, &mut budget, 0)?;
        }
        Some(odd)
    }

    /// Whether the ray from `q` toward growing `v` crosses `piece` an odd
    /// number of times, a point read on the side of growing `u` where its
    /// `u` is that of `q`.
    fn crossed(&self, piece: &Piece, q: Point2, budget: &mut usize, depth: u32) -> Option<bool> {
        let b = piece.bounds(self.slack);
        if b.lo[0] > q.x() || b.hi[0] < q.x() || b.hi[1] < q.y() {
            return Some(false);
        }
        if b.lo[1] > q.y() {
            return Some((piece.a.x() >= q.x()) != (piece.b.x() >= q.x()));
        }
        if depth >= MOST_HALVINGS || *budget == 0 {
            return None;
        }
        *budget -= 1;
        let (left, right) = piece.halves();
        Some(
            self.crossed(&left, q, budget, depth + 1)?
                != self.crossed(&right, q, budget, depth + 1)?,
        )
    }
}

/// How many pairs of pieces one proof that a trim's loops are simple may
/// read ([`loops_are_simple`]).
const MOST_LOOP_PAIRS: usize = 1 << 14;

/// How many times a piece is halved, at most, in that proof.
const MOST_LOOP_DEPTH: u32 = 40;

/// The angle, in radians, by which the rounding of a direction's
/// coordinates can move the angle between two directions read by `acos`.
const ANGLE_SLACK: f64 = 1e-7;

/// Whether each of the loops (`spans`, ranges of `pieces`) is a simple
/// curve and no two meet: no two pieces meet, but two that follow each
/// other in a loop, at their joint. A parity read of loops that are each
/// simple reads every point inside the outer loop and no hole as inside,
/// whatever the holes do; a loop that runs round a region twice reads it
/// outside.
///
/// A piece whose box lies within `joint` of a point is read as part of the
/// joint of the pieces on either side of it. Two pieces that do not follow
/// each other are apart where their boxes, grown by `slack`, keep apart,
/// halved, the larger first, as far as needed. Two that follow each other
/// each lie in a cone from their joint ([`cone2`]): where the cones meet
/// only at the joint, the pieces meet only near it, within the joint's
/// size over the angle between them; else they are halved, the halves at
/// the joint read so again. Two ends of pieces within `2 slack` of each
/// other away from a joint show the pieces meet. A loop of fewer than three
/// pieces, more than [`MOST_LOOP_PAIRS`] pairs read, or a piece halved
/// [`MOST_LOOP_DEPTH`] times reads as not simple.
fn loops_are_simple(pieces: &[Piece], spans: &[(usize, usize)], slack: f64, joint: f64) -> bool {
    // The pieces each loop runs through past its joints, by their places,
    // and the place that follows each.
    let mut proper: Vec<usize> = Vec::new();
    let mut next: Vec<usize> = Vec::new();
    for &(first, end) in spans {
        let start = proper.len();
        for k in first..end {
            let b = pieces[k].bounds(0.0);
            if (b.hi[0] - b.lo[0]).hypot(b.hi[1] - b.lo[1]) > joint {
                proper.push(k);
            }
        }
        let count = proper.len() - start;
        if count < 3 {
            return false;
        }
        next.extend((start..proper.len()).map(|i| start + (i - start + 1) % count));
    }
    let boxes: Vec<Box2> = proper.iter().map(|&k| pieces[k].bounds(slack)).collect();
    let mut order: Vec<usize> = (0..proper.len()).collect();
    order.sort_by(|&x, &y| boxes[x].lo[0].total_cmp(&boxes[y].lo[0]));
    let mut read = 0_usize;
    let mut work: Vec<(Piece, Piece, bool, u32)> = Vec::new();
    for (place, &x) in order.iter().enumerate() {
        for &y in &order[place + 1..] {
            read += 1;
            if read > MOST_LOOP_PAIRS {
                return false;
            }
            if boxes[y].lo[0] > boxes[x].hi[0] {
                break;
            }
            if boxes[y].lo[1] > boxes[x].hi[1] || boxes[y].hi[1] < boxes[x].lo[1] {
                continue;
            }
            let (px, py) = (&pieces[proper[x]], &pieces[proper[y]]);
            // Two pieces that follow each other, read from their joint.
            if next[x] == y {
                work.push((px.reversed(), py.clone(), true, 0));
            } else if next[y] == x {
                work.push((py.reversed(), px.clone(), true, 0));
            } else {
                work.push((px.clone(), py.clone(), false, 0));
            }
        }
    }
    while let Some((x, y, corner, depth)) = work.pop() {
        read += 1;
        if read > MOST_LOOP_PAIRS {
            return false;
        }
        let (bx, by) = (x.bounds(slack), y.bounds(slack));
        let parted = bx.lo[0] > by.hi[0]
            || by.lo[0] > bx.hi[0]
            || bx.lo[1] > by.hi[1]
            || by.lo[1] > bx.hi[1];
        if parted || (corner && cones2_apart(&x, &y, slack)) {
            continue;
        }
        let meet = |p: Point2, q: Point2| (p.x() - q.x()).hypot(p.y() - q.y()) <= 2.0 * slack;
        let ends_meet = [x.a, x.b].into_iter().any(|p| meet(p, y.a) || meet(p, y.b));
        if (!corner && ends_meet) || depth >= MOST_LOOP_DEPTH {
            return false;
        }
        let size = |b: Box2| (b.hi[0] - b.lo[0]).hypot(b.hi[1] - b.lo[1]);
        let (cut_x, cut_y) = (
            corner || size(bx) >= size(by),
            corner || size(by) > size(bx),
        );
        let xs = if cut_x {
            let (l, r) = x.halves();
            vec![l, r]
        } else {
            vec![x]
        };
        let ys = if cut_y {
            let (l, r) = y.halves();
            vec![l, r]
        } else {
            vec![y]
        };
        for (i, xx) in xs.iter().enumerate() {
            for (j, yy) in ys.iter().enumerate() {
                work.push((
                    xx.clone(),
                    yy.clone(),
                    corner && i == 0 && j == 0,
                    depth + 1,
                ));
            }
        }
    }
    true
}

/// The axis and half-angle of a cone from a piece's start that holds the
/// piece: a segment's direction; the sector between a conic arc's tangent
/// at its start and its chord, which holds the arc, as it turns by less
/// than half a turn; or the directions to a Bezier piece's control points
/// from its first, as each point of the piece blends them with weights
/// that are not negative. Each direction is read to an angle of its
/// rounding over its length. `None` where a direction is no longer than a
/// few `slack`, or the half-angle reaches a right angle.
fn cone2(piece: &Piece, slack: f64) -> Option<((f64, f64), f64)> {
    let unit = |(x, y): (f64, f64), err: f64| -> Option<((f64, f64), f64)> {
        let length = x.hypot(y);
        (length > 4.0 * slack && length.is_finite())
            .then(|| ((x / length, y / length), err + 4.0 * slack / length))
    };
    let chord = (piece.b.x() - piece.a.x(), piece.b.y() - piece.a.y());
    let directions: Vec<((f64, f64), f64)> = match &piece.hull {
        Hull::Segment => vec![unit(chord, 0.0)?],
        Hull::Conic { curve, t, .. } => {
            let along = curve.tangent(t.0) * (t.1 - t.0).signum();
            vec![unit((along.x(), along.y()), 1e-9)?, unit(chord, 0.0)?]
        }
        Hull::Bezier(net) => {
            let first = net.first()?;
            let apex = (first[0] / first[2], first[1] / first[2]);
            net[1..]
                .iter()
                .map(|q| unit((q[0] / q[2] - apex.0, q[1] / q[2] - apex.1), 0.0))
                .collect::<Option<_>>()?
        }
    };
    let sum = directions
        .iter()
        .fold((0.0, 0.0), |s, ((x, y), _)| (s.0 + x, s.1 + y));
    let length = sum.0.hypot(sum.1);
    if length.is_nan() || length <= 0.0 {
        return None;
    }
    let axis = (sum.0 / length, sum.1 / length);
    let half = directions
        .iter()
        .map(|((x, y), err)| x.madd(axis.0, y * axis.1).clamp(-1.0, 1.0).acos() + err)
        .fold(0.0_f64, f64::max);
    (half < std::f64::consts::FRAC_PI_2).then_some((axis, half))
}

/// Whether two pieces that start at one joint leave it in cones that meet
/// only there ([`cone2`]).
fn cones2_apart(a: &Piece, b: &Piece, slack: f64) -> bool {
    let (Some((x, hx)), Some((y, hy))) = (cone2(a, slack), cone2(b, slack)) else {
        return false;
    };
    x.0.madd(y.0, x.1 * y.1).clamp(-1.0, 1.0).acos() > hx + hy + ANGLE_SLACK
}

/// Whether a plane keeps every one of `points`, and so their convex hull,
/// farther than `reach` from `p`: one square to an axis, to the points'
/// middle from `p`, or to the point nearest `p`.
fn parted(points: &[Point3], p: Point3, reach: f64) -> bool {
    use brepkit_math::vec::Vec3;
    let Some(nearest) = points
        .iter()
        .copied()
        .min_by(|a, b| (*a - p).length().total_cmp(&(*b - p).length()))
    else {
        return true;
    };
    #[allow(clippy::cast_precision_loss)]
    let share = 1.0 / points.len() as f64;
    let middle = points
        .iter()
        .fold(Vec3::new(0.0, 0.0, 0.0), |sum, &q| sum + (q - p) * share);
    let axes = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let toward = [middle, nearest - p]
        .into_iter()
        .filter_map(|d| d.normalize().ok());
    axes.into_iter()
        .flat_map(|a| [a, -a])
        .chain(toward)
        .any(|d| points.iter().all(|&q| d.dot(q - p) > reach))
}

/// The pieces of `curve` from parameter `run.0` to `run.1`, in that order:
/// a line whole, a circle or an ellipse in arcs of at most [`ARC_STEP`], and
/// a B-spline in its Bezier pieces, each between two of its knots. `size`
/// takes the largest coordinate the pieces' points reach, a B-spline's
/// scaled by the ratio of its weights, and `degree` the B-spline's degree.
/// `None` where a B-spline's piece cannot be read.
fn pcurve_pieces(
    curve: &Curve2D,
    run: (f64, f64),
    size: &mut f64,
    degree: &mut usize,
) -> Option<Vec<Piece>> {
    let largest = |p: Point2| p.x().abs().max(p.y().abs());
    match curve {
        Curve2D::Line(line) => {
            *size = size.max(largest(line.origin()) + run.0.abs().max(run.1.abs()));
            Some(vec![Piece {
                owner: [0; 2],
                a: curve.evaluate(run.0),
                b: curve.evaluate(run.1),
                hull: Hull::Segment,
            }])
        }
        Curve2D::Circle(_) | Curve2D::Ellipse(_) => {
            let (centre, bend) = match curve {
                Curve2D::Circle(c) => (c.center(), c.radius()),
                Curve2D::Ellipse(e) => (e.center(), e.semi_major().max(e.semi_minor())),
                Curve2D::Line(_) | Curve2D::Nurbs(_) => return None,
            };
            *size = size.max(largest(centre) + bend);
            let count = ((run.1 - run.0).abs() / ARC_STEP).ceil().max(1.0);
            if !count.is_finite() || count > 1e6 {
                return None;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let count = count as usize;
            #[allow(clippy::cast_precision_loss)]
            let at = |k: usize| {
                if k == count {
                    run.1
                } else {
                    (run.1 - run.0).madd(k as f64 / count as f64, run.0)
                }
            };
            let points: Vec<(f64, Point2)> = (0..=count)
                .map(|k| {
                    let t = at(k);
                    (t, curve.evaluate(t))
                })
                .collect();
            Some(
                points
                    .windows(2)
                    .map(|w| Piece {
                        owner: [0; 2],
                        a: w[0].1,
                        b: w[1].1,
                        hull: Hull::Conic {
                            curve: curve.clone(),
                            t: (w[0].0, w[1].0),
                            bend,
                        },
                    })
                    .collect(),
            )
        }
        Curve2D::Nurbs(spline) => {
            let weights = spline.weights();
            let (least, most) = weights
                .iter()
                .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &w| {
                    (lo.min(w), hi.max(w))
                });
            if !(least > 0.0 && least.is_normal()) {
                return None;
            }
            let ratio = most / least;
            for p in spline.control_points() {
                *size = size.max(largest(*p) * ratio);
            }
            *degree = (*degree).max(spline.degree());
            let lifted = brepkit_math::nurbs::curve::NurbsCurve::new(
                spline.degree(),
                spline.knots().to_vec(),
                spline
                    .control_points()
                    .iter()
                    .map(|p| Point3::new(p.x(), p.y(), 0.0))
                    .collect(),
                weights.to_vec(),
            )
            .ok()?;
            let (lo, hi) = (run.0.min(run.1), run.0.max(run.1));
            let mut breaks: Vec<f64> = std::iter::once(lo)
                .chain(spline.knots().iter().copied().filter(|&k| lo < k && k < hi))
                .chain(std::iter::once(hi))
                .collect();
            breaks.dedup_by(|a, b| a <= b);
            if run.0 > run.1 {
                breaks.reverse();
            }
            let mut pieces: Vec<Piece> = Vec::with_capacity(breaks.len());
            for w in breaks.windows(2) {
                let net: Vec<[f64; 3]> =
                    brepkit_math::nurbs::decompose::curve_bezier_between(&lifted, w[0], w[1])
                        .ok()?
                        .into_iter()
                        .map(|q| [q[0], q[1], q[3]])
                        .collect();
                let (first, last) = (net.first()?, net.last()?);
                if !net.iter().all(|q| q[2] > 0.0 && q[2].is_normal()) {
                    return None;
                }
                let a = pieces.last().map_or_else(
                    || Point2::new(first[0] / first[2], first[1] / first[2]),
                    |p| p.b,
                );
                let b = Point2::new(last[0] / last[2], last[1] / last[2]);
                pieces.push(Piece {
                    owner: [0; 2],
                    a,
                    b,
                    hull: Hull::Bezier(net),
                });
            }
            (!pieces.is_empty()).then_some(pieces)
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::curves2d::{Circle2D, Ellipse2D, Line2D, NurbsCurve2D};
    use brepkit_math::vec::Vec2;

    fn sheet() -> NurbsSurface {
        let rows = [0.0, 1.0]
            .iter()
            .map(|&x| [0.0, 1.0].iter().map(|&y| Point3::new(x, y, 0.0)).collect())
            .collect();
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        NurbsSurface::new(1, 1, knots.clone(), knots, rows, vec![vec![1.0; 2]; 2]).unwrap()
    }

    /// The circle about `c` of radius `r` as a rational quadratic B-spline
    /// over `[0, 1]`.
    fn spline_circle(c: Point2, r: f64) -> NurbsCurve2D {
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
        NurbsCurve2D::new(
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            corners
                .iter()
                .map(|&(x, y)| Point2::new(x.madd(r, c.x()), y.madd(r, c.y())))
                .collect(),
            (0..9).map(|k| if k % 2 == 1 { w } else { 1.0 }).collect(),
        )
        .unwrap()
    }

    /// The unit square less a round hole bounded by a B-spline and an
    /// elliptic hole, each run either way, read at a grid of points and at
    /// points `1e-3`, `1e-6` and `1e-9` to either side of each wire: each
    /// point is read on its own side of every wire, a point on the square's
    /// border, which the reading may put on either side, aside.
    #[test]
    fn parameters_read_inside_loops_of_lines_splines_and_ellipses_as_they_lie() {
        let (c, r) = (Point2::new(0.3, 0.3), 0.15);
        let (e, a, b, turn) = (Point2::new(0.7, 0.65), 0.2, 0.08, 0.5_f64);
        let round = Curve2D::Nurbs(spline_circle(c, r));
        let oval = Curve2D::Ellipse(Ellipse2D::new(e, a, b, turn).unwrap());
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let sides: Vec<Curve2D> = (0..4)
            .map(|k| {
                let (p, q) = (corners[k], corners[(k + 1) % 4]);
                Curve2D::Line(
                    Line2D::new(Point2::new(p.0, p.1), Vec2::new(q.0 - p.0, q.1 - p.1)).unwrap(),
                )
            })
            .collect();
        let inside_oval = |q: Point2| {
            let (s, co) = turn.sin_cos();
            let (dx, dy) = (q.x() - e.x(), q.y() - e.y());
            let (x, y) = (dx.madd(co, dy * s), dy.madd(co, -(dx * s)));
            (x / a).powi(2) + (y / b).powi(2) < 1.0
        };
        let truth = |q: Point2| {
            (0.0..1.0).contains(&q.x())
                && (0.0..1.0).contains(&q.y())
                && (q - c).length() > r
                && !inside_oval(q)
        };
        let mut points: Vec<Point2> = (0..=100)
            .flat_map(|i| (0..=100).map(move |j| (i, j)))
            .map(|(i, j)| {
                Point2::new(
                    0.011f64.madd(f64::from(i), -0.05),
                    0.011f64.madd(f64::from(j), -0.05),
                )
            })
            .collect();
        for k in 0..64 {
            let t = std::f64::consts::TAU * f64::from(k) / 64.0;
            for off in [1e-3, 1e-6, 1e-9, -1e-3, -1e-6, -1e-9] {
                points.push(Point2::new(
                    (r + off).madd(t.cos(), c.x()),
                    (r + off).madd(t.sin(), c.y()),
                ));
                let (s, co) = turn.sin_cos();
                let (x, y) = (a * t.cos(), b * t.sin());
                let normal = Vec2::new(b * t.cos(), a * t.sin());
                let n = normal * (1.0 / normal.length());
                let (nx, ny) = (n.x().madd(co, -(n.y() * s)), n.x().madd(s, n.y() * co));
                points.push(Point2::new(
                    x.madd(co, -(y * s)) + e.x() + nx * off,
                    x.madd(s, y * co) + e.y() + ny * off,
                ));
            }
        }
        let tau = std::f64::consts::TAU;
        for (round_run, oval_run) in [((0.0, 1.0), (0.0, tau)), ((1.0, 0.0), (tau, 0.0))] {
            let loops: Vec<Vec<Run<'_>>> = vec![
                sides.iter().map(|side| (side, (0.0, 1.0), 0)).collect(),
                vec![(&round, round_run, 0)],
                vec![(&oval, oval_run, 0)],
            ];
            let trim = NurbsFaceTrim::of_runs(&sheet(), &loops, Vec::new()).unwrap();
            let misread: Vec<(Point2, Option<bool>)> = points
                .iter()
                .filter(|q| {
                    let border = q
                        .x()
                        .abs()
                        .min(q.y().abs())
                        .min((1.0 - q.x()).abs())
                        .min((1.0 - q.y()).abs())
                        < 1e-12;
                    !border
                })
                .map(|&q| (q, trim.holds_parameters(q)))
                .filter(|&(q, read)| read != Some(truth(q)))
                .collect();
            assert!(misread.is_empty(), "{round_run:?}: {misread:?}");
        }
    }

    /// A circle read over a quarter turn, then on round the rest of it, as
    /// two pcurves: the gap closer between them and the arcs' boxes leave
    /// the reading of a point at its centre and just outside it unchanged.
    #[test]
    fn a_circle_in_two_runs_reads_its_inside() {
        let circle = Curve2D::Circle(Circle2D::new(Point2::new(0.5, 0.5), 0.25).unwrap());
        let quarter = std::f64::consts::FRAC_PI_2;
        let loops = vec![vec![
            (&circle, (0.0, quarter), 0),
            (&circle, (quarter, std::f64::consts::TAU), 0),
        ]];
        let trim = NurbsFaceTrim::of_runs(&sheet(), &loops, Vec::new()).unwrap();
        assert_eq!(trim.holds_parameters(Point2::new(0.5, 0.5)), Some(true));
        assert_eq!(
            trim.holds_parameters(Point2::new(0.75 + 1e-9, 0.5)),
            Some(false)
        );
        assert_eq!(
            trim.holds_parameters(Point2::new(0.75 - 1e-9, 0.5)),
            Some(true)
        );
        assert_eq!(
            trim.holds_parameters(Point2::new(0.5, 0.75 + 1e-9)),
            Some(false)
        );
    }

    /// Loops that are no simple curves take no trim: a figure eight, whose
    /// sides cross at its middle, and a square hole that crosses the outer
    /// square's side. The outer square with a hole inside it takes one.
    #[test]
    fn loops_that_cross_take_no_trim() {
        // Each wire's sides as lines read over their lengths.
        let sides = |corners: &[(f64, f64)]| -> Vec<(Curve2D, f64)> {
            (0..corners.len())
                .map(|k| {
                    let (p, q) = (corners[k], corners[(k + 1) % corners.len()]);
                    let along = Vec2::new(q.0 - p.0, q.1 - p.1);
                    let line = Line2D::new(Point2::new(p.0, p.1), along).unwrap();
                    (Curve2D::Line(line), along.length())
                })
                .collect()
        };
        let trimmed = |wires: &[&[(f64, f64)]]| {
            let lines: Vec<Vec<(Curve2D, f64)>> = wires.iter().map(|w| sides(w)).collect();
            let loops: Vec<Vec<Run<'_>>> = lines
                .iter()
                .map(|w| w.iter().map(|(c, len)| (c, (0.0, *len), 0)).collect())
                .collect();
            NurbsFaceTrim::of_runs(&sheet(), &loops, Vec::new()).is_some()
        };
        let outer: &[(f64, f64)] = &[(0.1, 0.1), (0.9, 0.1), (0.9, 0.9), (0.1, 0.9)];
        let eight: &[(f64, f64)] = &[(0.1, 0.1), (0.9, 0.9), (0.9, 0.1), (0.1, 0.9)];
        let across: &[(f64, f64)] = &[(0.8, 0.4), (0.8, 0.6), (0.95, 0.6), (0.95, 0.4)];
        let inside: &[(f64, f64)] = &[(0.4, 0.4), (0.4, 0.6), (0.6, 0.6), (0.6, 0.4)];
        assert!(!trimmed(&[eight]));
        assert!(!trimmed(&[outer, across]));
        assert!(trimmed(&[outer, inside]));
    }
}

//! A B-spline surface's knot spans indexed by their control boxes.
//!
//! Over one knot span a B-spline surface with positive weights lies within
//! the convex hull of the control points that shape it, so the box of those
//! points bounds the span. A fitted surface can have thousands of spans;
//! the index keeps their boxes in a bounding volume hierarchy, made once
//! per surface, so a search near a point reads only the spans near it.
//!
//! Each span can also be split into sub-patches by de Casteljau, whose own
//! nets hug the surface far closer than the span's: over a span turning
//! through a wide angle the span's box holds points of the surface far from
//! the span.
//!
//! A line meets the surface only in spans whose box it meets, and each such
//! span is solved on its own, so a fitted patch with hundreds of spans is
//! read as finely as its knots for exact line hits, however small a fixed
//! grid over its whole domain would leave each span.

use crate::aabb::Aabb3;
use crate::bvh::Bvh;
use crate::madd::Madd;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::{Point3, Vec3};

/// A parameter interval.
type Interval = (f64, f64);

/// One knot span of a surface: its parameter rectangle, and the indices of
/// the last row and column of the control points that shape it.
#[derive(Debug, Clone, Copy)]
struct Span {
    u: Interval,
    v: Interval,
    last_row: usize,
    last_col: usize,
}

/// Each span is split in halves this many times along `u` and `v` for its
/// sub-patch boxes.
const SUB_DEPTH: usize = 2;

/// Sub-patch boxes a span: `4^SUB_DEPTH`.
const SUB_BOXES: usize = 1 << (2 * SUB_DEPTH);

/// One span's sub-patch boxes.
type SubBoxes = [Aabb3; SUB_BOXES];

/// A surface's knot spans with a BVH over their control boxes.
#[derive(Debug, Clone)]
pub struct SpanIndex {
    /// The spans, along `u` and then, within each, along `v`.
    spans: Vec<Span>,
    /// Each span's control box, by span index, exactly the box of its
    /// control points.
    boxes: Vec<Aabb3>,
    /// The boxes padded for rounding, in a BVH.
    bvh: Bvh,
    /// The padding of each box for rounding: a share of the largest
    /// control box diagonal, the scale of the surface's pieces, and a
    /// share of the largest coordinate, which rounding moves points by.
    pad: f64,
    /// The largest control box diagonal, the scale of the surface's pieces.
    scale: f64,
    /// Each span's sub-patch boxes, by span index, each made the first time
    /// a search reads it.
    sub_boxes: std::sync::OnceLock<Box<[std::sync::OnceLock<SubBoxes>]>>,
    /// Each span's Bezier net, by span index, each made the first time it
    /// is read.
    nets: std::sync::OnceLock<Box<[std::sync::OnceLock<Vec<Hpoint>>]>>,
}

/// A point where a line meets the surface.
#[derive(Debug, Clone, Copy)]
pub struct LineHit {
    /// The line's parameter: the hit is `origin + dir * t`.
    pub t: f64,
    /// The surface's parameters at the hit.
    pub uv: (f64, f64),
    /// The hit itself.
    pub point: Point3,
    /// Whether the line meets the surface at a grazing angle there, so the
    /// hit may be a touch rather than a crossing.
    pub grazing: bool,
}

/// The distinct knot spans of one direction: each span's bounds and the
/// index of its last control point.
pub(crate) fn knot_spans(knots: &[f64], degree: usize, n: usize) -> Vec<(f64, f64, usize)> {
    (degree..n)
        .filter(|&i| knots[i + 1] > knots[i])
        .map(|i| (knots[i], knots[i + 1], i))
        .collect()
}

impl SpanIndex {
    /// Index `surface`'s knot spans. `None` when a weight is not positive
    /// (or not a number), where the control boxes need not bound the
    /// surface, or is too small for its products to keep their digits, or
    /// a weighted coordinate overflows.
    #[must_use]
    pub fn new(surface: &NurbsSurface) -> Option<Self> {
        if surface
            .weights()
            .iter()
            .flatten()
            .any(|&w| w <= 0.0 || !w.is_normal())
        {
            return None;
        }
        let cps = surface.control_points();
        // A weighted coordinate that overflows leaves the nets without
        // their points.
        let weighted = cps.iter().zip(surface.weights()).all(|(row, ws)| {
            row.iter().zip(ws).all(|(p, &w)| {
                [p.x() * w, p.y() * w, p.z() * w]
                    .iter()
                    .all(|c| c.is_finite())
            })
        });
        if !weighted {
            return None;
        }
        let rows = cps.len();
        let cols = cps.first()?.len();
        let (pu, pv) = (surface.degree_u(), surface.degree_v());
        let spans_u = knot_spans(surface.knots_u(), pu, rows);
        let spans_v = knot_spans(surface.knots_v(), pv, cols);
        let mut spans = Vec::with_capacity(spans_u.len() * spans_v.len());
        let mut boxes = Vec::with_capacity(spans.capacity());
        let mut scale = 0.0_f64;
        for &(u0, u1, last_row) in &spans_u {
            for &(v0, v1, last_col) in &spans_v {
                let aabb = Aabb3::from_points(
                    cps[last_row - pu..=last_row]
                        .iter()
                        .flat_map(|row| row[last_col - pv..=last_col].iter().copied()),
                );
                scale = scale.max((aabb.max - aabb.min).length());
                boxes.push(aabb);
                spans.push(Span {
                    u: (u0, u1),
                    v: (v0, v1),
                    last_row,
                    last_col,
                });
            }
        }
        // Pad each box for rounding, so the BVH holds every point a box
        // comes within a distance of.
        let largest = cps
            .iter()
            .flatten()
            .flat_map(|p| [p.x(), p.y(), p.z()])
            .fold(0.0_f64, |m, c| m.max(c.abs()));
        let pad = 1e-9 * scale.max(1.0) + 1e-12 * largest;
        let padded: Vec<(usize, Aabb3)> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| (i, b.expanded(pad)))
            .collect();
        Some(Self {
            spans,
            boxes,
            bvh: Bvh::build(&padded),
            pad,
            scale,
            sub_boxes: std::sync::OnceLock::new(),
            nets: std::sync::OnceLock::new(),
        })
    }

    /// How many spans the index holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether the index holds no span.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Span `i`'s parameter rectangle, `((u0, u1), (v0, v1))`.
    #[must_use]
    pub fn span(&self, i: usize) -> Option<(Interval, Interval)> {
        self.spans.get(i).map(|s| (s.u, s.v))
    }

    /// The spans whose control boxes come nearer `point` than `bound`, as
    /// `(span, distance to its box)`, nearest first and, among equally
    /// near, in span order: what reading every span's box and sorting them
    /// by their distance gives, from only the spans near the point.
    #[must_use]
    pub fn spans_within(&self, point: Point3, bound: f64) -> Vec<(usize, f64)> {
        let mut near: Vec<usize> = if bound.is_finite() {
            let reach = crate::vec::Vec3::new(bound, bound, bound);
            self.bvh.query_overlap(&Aabb3 {
                min: point - reach,
                max: point + reach,
            })
        } else {
            (0..self.spans.len()).collect()
        };
        near.sort_unstable();
        let mut found: Vec<(usize, f64)> = near
            .into_iter()
            .map(|i| (i, self.boxes[i].distance_squared_to_point(point).sqrt()))
            .filter(|&(_, d)| d < bound)
            .collect();
        found.sort_by(|a, b| a.1.total_cmp(&b.1));
        found
    }

    /// Whether every point of `surface` lies at least `reach` from `point`:
    /// no span whose control box comes nearer has a sub-patch box nearer.
    #[must_use]
    pub fn clear_of(&self, surface: &NurbsSurface, point: Point3, reach: f64) -> bool {
        self.spans_within(point, reach)
            .into_iter()
            .all(|(i, _)| self.sub_gap(surface, i, point) >= reach)
    }

    /// Whether no span's control box of this index comes within `margin` of
    /// one of `other`'s, read against its boxes padded for rounding: then no
    /// part of either surface's Bezier decomposition does, each part's net
    /// lying in its span's control box.
    #[must_use]
    pub fn boxes_apart(&self, other: &Self, margin: f64) -> bool {
        let mut near = Vec::new();
        self.boxes.iter().all(|b| {
            other
                .bvh
                .query_overlap_into(&b.expanded(margin + self.pad), &mut near);
            near.is_empty()
        })
    }

    /// Whether every point of `b` lies farther than `reach` from `surface`:
    /// none of the sub-patch boxes of the spans whose boxes come within
    /// `reach` of `b` does.
    #[must_use]
    pub fn box_clears(&self, surface: &NurbsSurface, b: &Aabb3, reach: f64) -> bool {
        self.bvh
            .query_overlap(&b.expanded(reach))
            .into_iter()
            .all(|i| {
                self.sub_boxes_of(surface, i)
                    .is_some_and(|boxes| boxes.iter().all(|sub| box_gap(sub, b) > reach))
            })
    }

    /// A lower bound on the distance from `point` to span `i`: its distance
    /// to the nearest of the span's sub-patch boxes, which hug the span far
    /// closer than its control box; 0 where they cannot be made.
    pub(crate) fn sub_gap(&self, surface: &NurbsSurface, i: usize, point: Point3) -> f64 {
        self.sub_boxes_of(surface, i).map_or(0.0, |boxes| {
            boxes
                .iter()
                .map(|b| b.distance_squared_to_point(point))
                .fold(f64::INFINITY, f64::min)
                .sqrt()
        })
    }

    /// Span `i`'s Bezier net (see [`span_net`]), made the first time it is
    /// read, with its parameter rectangle.
    pub(crate) fn bezier_net(
        &self,
        surface: &NurbsSurface,
        i: usize,
    ) -> Option<(&[Hpoint], Interval, Interval)> {
        let span = *self.spans.get(i)?;
        let net = self
            .nets
            .get_or_init(|| {
                (0..self.spans.len())
                    .map(|_| std::sync::OnceLock::new())
                    .collect()
            })
            .get(i)?
            .get_or_init(|| span_net(surface, span));
        Some((net, span.u, span.v))
    }

    /// Span `i`'s sub-patch boxes, made the first time they are read.
    fn sub_boxes_of(&self, surface: &NurbsSurface, i: usize) -> Option<&SubBoxes> {
        let pad = self.pad;
        let (net, _, _) = self.bezier_net(surface, i)?;
        Some(
            self.sub_boxes
                .get_or_init(|| {
                    (0..self.spans.len())
                        .map(|_| std::sync::OnceLock::new())
                        .collect()
                })
                .get(i)?
                .get_or_init(|| sub_patch_boxes(surface, net, pad)),
        )
    }

    /// The indices of the spans whose control boxes meet `aabb`, in span
    /// order.
    pub(crate) fn span_ids_meeting(&self, aabb: &Aabb3) -> Vec<usize> {
        let mut found = self.bvh.query_overlap(aabb);
        found.sort_unstable();
        found
    }

    /// The parameter rectangles `(u, v)` of the spans whose control boxes
    /// meet `aabb`, in span order.
    #[must_use]
    pub fn spans_meeting(&self, aabb: &Aabb3) -> Vec<((f64, f64), (f64, f64))> {
        let mut found = self.bvh.query_overlap(aabb);
        found.sort_unstable();
        found
            .into_iter()
            .map(|i| (self.spans[i].u, self.spans[i].v))
            .collect()
    }

    /// Every point where the line `origin + dir * t` meets `surface` (the
    /// surface this index was built from), each found once, in no order.
    ///
    /// Each span whose box the line meets is halved in both parameters,
    /// dropping each piece whose middle lies farther from the line than its
    /// half-widths along the surface's derivatives reach (with a margin for
    /// curvature), and each piece left at the finest level is searched by
    /// Newton's method on `S(u, v) = origin + dir * t` from its middle.
    #[must_use]
    pub fn line_hits(&self, surface: &NurbsSurface, origin: Point3, dir: Vec3) -> Vec<LineHit> {
        const DEPTH: u32 = 4;
        const REACH: f64 = 1.5;
        let mut hits: Vec<LineHit> = Vec::new();
        let dd = dir.dot(dir);
        if dd <= 1e-300 {
            return hits;
        }
        // The BVH's ray query keeps positive t only; a line runs both ways.
        let mut candidates = self.bvh.query_ray(origin, dir);
        candidates.extend(self.bvh.query_ray(origin, -dir));
        candidates.sort_unstable();
        candidates.dedup();
        let off_line = |p: Point3| {
            let w = p - origin;
            (w - dir * (w.dot(dir) / dd)).length()
        };
        let mut stack: Vec<(Interval, Interval, u32)> = Vec::new();
        for idx in candidates {
            let span = self.spans[idx];
            stack.push((span.u, span.v, 0));
            while let Some((u, v, depth)) = stack.pop() {
                let mid = (f64::midpoint(u.0, u.1), f64::midpoint(v.0, v.1));
                let (p, su, sv) = surface.first_derivatives(mid.0, mid.1);
                let reach = REACH * 0.5 * su.length().madd(u.1 - u.0, sv.length() * (v.1 - v.0));
                if off_line(Point3::new(p.x(), p.y(), p.z())) > reach + 1e-9 * self.scale.max(1.0) {
                    continue;
                }
                if depth < DEPTH {
                    let (um, vm) = mid;
                    for (uu, vv) in [
                        ((u.0, um), (v.0, vm)),
                        ((um, u.1), (v.0, vm)),
                        ((u.0, um), (vm, v.1)),
                        ((um, u.1), (vm, v.1)),
                    ] {
                        stack.push((uu, vv, depth + 1));
                    }
                    continue;
                }
                let Some(hit) = self.solve(surface, origin, dir, mid, (u, v)) else {
                    continue;
                };
                let slack = |(lo, hi): (f64, f64)| 1e-9 * (hi - lo);
                if hit.uv.0 < span.u.0 - slack(span.u)
                    || hit.uv.0 > span.u.1 + slack(span.u)
                    || hit.uv.1 < span.v.0 - slack(span.v)
                    || hit.uv.1 > span.v.1 + slack(span.v)
                {
                    continue;
                }
                // A root on a piece's edge is found from both sides of it.
                let same =
                    |h: &LineHit| (h.point - hit.point).length() <= 1e-9 * self.scale.max(1.0);
                if !hits.iter().any(same) {
                    hits.push(hit);
                }
            }
        }
        hits
    }

    /// The point of `surface` closest to `p`, as `(u, v, distance)`.
    ///
    /// Each span's own closest point is found by Gauss-Newton from its
    /// middle, kept within the span, and the spans are searched nearest box
    /// first: a span whose box lies farther than the best point found so far
    /// holds no closer one. `None` for an empty index.
    #[must_use]
    pub fn closest_point(&self, surface: &NurbsSurface, p: Point3) -> Option<(f64, f64, f64)> {
        let in_span = |i: usize| {
            let span = self.spans[i];
            closest_from(
                surface,
                p,
                (
                    f64::midpoint(span.u.0, span.u.1),
                    f64::midpoint(span.v.0, span.v.1),
                ),
                span.u,
                span.v,
            )
        };
        let best = self
            .bvh
            .query_closest_with_distance(p, &|i| in_span(i).2.powi(2))?;
        Some(in_span(best))
    }

    /// The point of `surface` closest to `p` found by Gauss-Newton from
    /// `guess` over the whole domain, as `(u, v, distance)`: the nearest
    /// point to a point beside one already placed (the next sample along a
    /// curve on the surface), which a search over the whole surface need not
    /// find on a surface with many spans.
    #[must_use]
    pub fn closest_point_near(
        surface: &NurbsSurface,
        p: Point3,
        guess: (f64, f64),
    ) -> (f64, f64, f64) {
        closest_from(surface, p, guess, surface.domain_u(), surface.domain_v())
    }

    /// Newton's method on `S(u, v) - (origin + dir * t)` from `seed`, given
    /// up once it leaves the piece `(u, v)` by more than the piece's width.
    fn solve(
        &self,
        surface: &NurbsSurface,
        origin: Point3,
        dir: Vec3,
        seed: (f64, f64),
        (pu, pv): ((f64, f64), (f64, f64)),
    ) -> Option<LineHit> {
        const MAX_ITER: usize = 30;
        let (u_lo, u_hi) = surface.domain_u();
        let (v_lo, v_hi) = surface.domain_v();
        let (du, dv) = (pu.1 - pu.0, pv.1 - pv.0);
        let u_win = ((pu.0 - du).max(u_lo), (pu.1 + du).min(u_hi));
        let v_win = ((pv.0 - dv).max(v_lo), (pv.1 + dv).min(v_hi));
        let dd = dir.dot(dir);
        let (mut u, mut v) = seed;
        let p = surface.evaluate(u, v);
        let mut t = (p - origin).dot(dir) / dd;
        let tol = 1e-12 * self.scale.max(1.0);
        for _ in 0..MAX_ITER {
            let (p, su, sv) = surface.first_derivatives(u, v);
            let r = Point3::new(p.x(), p.y(), p.z()) - (origin + dir * t);
            let back = -dir;
            let det = su.dot(sv.cross(back));
            let size = su.length() * sv.length() * dir.length();
            if r.length() <= tol {
                return Some(LineHit {
                    t,
                    uv: (u, v),
                    point: Point3::new(p.x(), p.y(), p.z()),
                    grazing: det.abs() <= 1e-3 * size,
                });
            }
            if det.abs() <= 1e-12 * size {
                return None;
            }
            // Solve su du + sv dv - dir dt = -r.
            let rhs = -r;
            u += rhs.dot(sv.cross(back)) / det;
            v += su.dot(rhs.cross(back)) / det;
            t += su.dot(sv.cross(rhs)) / det;
            if u < u_win.0 || u > u_win.1 || v < v_win.0 || v > v_win.1 {
                return None;
            }
        }
        None
    }
}

/// The distance between two boxes, 0 where they meet.
pub(crate) fn box_gap(a: &Aabb3, b: &Aabb3) -> f64 {
    let gap = |a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64| (a_lo - b_hi).max(b_lo - a_hi).max(0.0);
    let x = gap(a.min.x(), a.max.x(), b.min.x(), b.max.x());
    let y = gap(a.min.y(), a.max.y(), b.min.y(), b.max.y());
    let z = gap(a.min.z(), a.max.z(), b.min.z(), b.max.z());
    x.madd(x, y.madd(y, z * z)).sqrt()
}

/// A homogeneous control point: `(w x, w y, w z, w)`.
pub(crate) type Hpoint = [f64; 4];

/// `span`'s Bezier net, read off its control points by blossoming: rows
/// along `u` of homogeneous points, point `(i, j)` at `i * (degree_v + 1) +
/// j`.
fn span_net(surface: &NurbsSurface, span: Span) -> Vec<Hpoint> {
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let (nu, nv) = (pu + 1, pv + 1);
    let cps = surface.control_points();
    let ws = surface.weights();
    // The span's own control points, homogeneous, row by row.
    let local: Vec<Hpoint> = (span.last_row - pu..=span.last_row)
        .flat_map(|r| {
            (span.last_col - pv..=span.last_col).map(move |c| {
                let (q, w) = (cps[r][c], ws[r][c]);
                [q.x() * w, q.y() * w, q.z() * w, w]
            })
        })
        .collect();
    // Bezier form along u, column by column, then along v, row by row.
    let mut net = vec![[0.0; 4]; nu * nv];
    for c in 0..nv {
        let column: Vec<Hpoint> = (0..nu).map(|r| local[r * nv + c]).collect();
        let bez = span_bezier(&column, surface.knots_u(), span.last_row, span.u);
        for (r, h) in bez.into_iter().enumerate() {
            net[r * nv + c] = h;
        }
    }
    for r in 0..nu {
        let row: Vec<Hpoint> = net[r * nv..(r + 1) * nv].to_vec();
        let bez = span_bezier(&row, surface.knots_v(), span.last_col, span.v);
        net[r * nv..(r + 1) * nv].copy_from_slice(&bez);
    }
    net
}

/// A span's sub-patch boxes, each padded by `pad`: its Bezier net `net`,
/// split in halves along `u` and `v` [`SUB_DEPTH`] times, each piece's box
/// that of its net's points. Each piece lies in the hull of its net, its
/// weights being positive.
fn sub_patch_boxes(surface: &NurbsSurface, net: &[Hpoint], pad: f64) -> SubBoxes {
    let (nu, nv) = (surface.degree_u() + 1, surface.degree_v() + 1);
    let mut pieces = vec![net.to_vec()];
    for _ in 0..SUB_DEPTH {
        pieces = pieces
            .into_iter()
            .flat_map(|n| {
                let (a, b) = halve(&n, nu, nv, true);
                let (a0, a1) = halve(&a, nu, nv, false);
                let (b0, b1) = halve(&b, nu, nv, false);
                [a0, a1, b0, b1]
            })
            .collect();
    }
    std::array::from_fn(|k| {
        Aabb3::from_points(
            pieces[k]
                .iter()
                .map(|h| Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3])),
        )
        .expanded(pad)
    })
}

/// The Bezier control points of a B-spline curve over its knot span
/// `[a, b]`, the span `i` (`knots[i] <= a < knots[i + 1]`) shaped by the
/// homogeneous de Boor points `points` (the curve's `i - p` to `i`): point
/// `j` is the curve's blossom at `a` taken `p - j` times and `b` taken `j`
/// times.
pub(crate) fn span_bezier(
    points: &[Hpoint],
    knots: &[f64],
    i: usize,
    (a, b): Interval,
) -> Vec<Hpoint> {
    let p = points.len() - 1;
    let blossom = |at_b: usize| {
        let mut d = points.to_vec();
        for level in 1..=p {
            let u = if level <= at_b { b } else { a };
            for j in (level..=p).rev() {
                let (left, right) = (knots[i - p + j], knots[i + 1 + j - level]);
                let alpha = (u - left) / (right - left);
                let prev = d[j - 1];
                for (c, q) in d[j].iter_mut().zip(prev) {
                    *c = (1.0 - alpha).madd(q, alpha * *c);
                }
            }
        }
        d[p]
    };
    (0..=p).map(blossom).collect()
}

/// A Bezier net of `nu` by `nv` homogeneous points (row `i`, column `j` at
/// `i * nv + j`) split at the middle of `u` (rows) or of `v` (columns), by
/// de Casteljau: the two halves' nets.
pub(crate) fn halve(
    net: &[Hpoint],
    nu: usize,
    nv: usize,
    along_u: bool,
) -> (Vec<Hpoint>, Vec<Hpoint>) {
    let (lines, len) = if along_u { (nv, nu) } else { (nu, nv) };
    let at = |line: usize, k: usize| {
        if along_u {
            k * nv + line
        } else {
            line * nv + k
        }
    };
    let mut lo = vec![[0.0; 4]; net.len()];
    let mut hi = vec![[0.0; 4]; net.len()];
    let mut tmp: Vec<Hpoint> = Vec::with_capacity(len);
    for line in 0..lines {
        tmp.clear();
        tmp.extend((0..len).map(|k| net[at(line, k)]));
        lo[at(line, 0)] = tmp[0];
        hi[at(line, len - 1)] = tmp[len - 1];
        for r in 1..len {
            for k in 0..len - r {
                for c in 0..4 {
                    tmp[k][c] = f64::midpoint(tmp[k][c], tmp[k + 1][c]);
                }
            }
            lo[at(line, r)] = tmp[0];
            hi[at(line, len - 1 - r)] = tmp[len - 1 - r];
        }
    }
    (lo, hi)
}

/// Gauss-Newton for the point of `surface` nearest `p` from `start`, kept
/// within the parameter window `(u_win, v_win)`, as `(u, v, distance)`.
fn closest_from(
    surface: &NurbsSurface,
    p: Point3,
    start: (f64, f64),
    u_win: Interval,
    v_win: Interval,
) -> (f64, f64, f64) {
    const MAX_ITER: usize = 30;
    let (mut u, mut v) = start;
    let gap = |u: f64, v: f64| (surface.evaluate(u, v) - p).length();
    for _ in 0..MAX_ITER {
        let (s, su, sv) = surface.first_derivatives(u, v);
        let r = Vec3::new(s.x() - p.x(), s.y() - p.y(), s.z() - p.z());
        let (a11, a12, a22) = (su.dot(su), su.dot(sv), sv.dot(sv));
        let (b1, b2) = (-su.dot(r), -sv.dot(r));
        let det = a11.madd(a22, -(a12 * a12));
        if det.abs() <= 1e-300 {
            break;
        }
        let du = b1.madd(a22, -(b2 * a12)) / det;
        let dv = a11.madd(b2, -(a12 * b1)) / det;
        let (nu, nv) = (
            (u + du).clamp(u_win.0, u_win.1),
            (v + dv).clamp(v_win.0, v_win.1),
        );
        let moved = (nu - u).abs() + (nv - v).abs();
        (u, v) = (nu, nv);
        if moved <= 1e-15 * (1.0 + u.abs() + v.abs()) {
            break;
        }
    }
    (u, v, gap(u, v))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    /// A U-shaped wall: a rational cubic profile of two spans, each turning
    /// through about a right angle, run straight along `y` over [0, 1].
    fn u_wall() -> NurbsSurface {
        let profile = [
            (0.0, 4.0),
            (0.0, -1.0),
            (2.0, -1.5),
            (4.0, -1.0),
            (4.0, 4.0),
        ];
        let weights = [1.0, 0.7, 1.3, 0.7, 1.0];
        let cps = profile
            .iter()
            .map(|&(x, z)| vec![Point3::new(x, 0.0, z), Point3::new(x, 1.0, z)])
            .collect();
        let ws = weights.iter().map(|&w| vec![w, w]).collect();
        let knots_u = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0];
        NurbsSurface::new(3, 1, knots_u, vec![0.0, 0.0, 1.0, 1.0], cps, ws).unwrap()
    }

    /// A bicubic patch with nine spans whose middle bulges up.
    fn bump() -> NurbsSurface {
        let n = 6;
        let cps: Vec<Vec<Point3>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        let (x, y) = (f64::from(i) * 0.6, f64::from(j) * 0.6);
                        let z = if (1..n - 1).contains(&i) && (1..n - 1).contains(&j) {
                            1.0
                        } else {
                            0.0
                        };
                        Point3::new(x, y, z)
                    })
                    .collect()
            })
            .collect();
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0];
        NurbsSurface::new(3, 3, knots.clone(), knots, cps, vec![vec![1.0; 6]; 6]).unwrap()
    }

    /// The least distance from `p` to span `i` of `index`, sampled.
    fn sampled_gap(s: &NurbsSurface, index: &SpanIndex, i: usize, p: Point3) -> f64 {
        let span = index.spans[i];
        let n = 64;
        let mut best = f64::INFINITY;
        for a in 0..=n {
            for b in 0..=n {
                let u = span.u.0 + (span.u.1 - span.u.0) * f64::from(a) / f64::from(n);
                let v = span.v.0 + (span.v.1 - span.v.0) * f64::from(b) / f64::from(n);
                best = best.min((s.evaluate(u, v) - p).length());
            }
        }
        best
    }

    #[test]
    fn sub_patch_boxes_hold_their_span() {
        // Weights of 1e308 leave the surface as it is, and its homogeneous
        // points stay finite; the sum of two of them does not, so the
        // halvings must not form it.
        let heavy = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            [0.0, 0.5, 0.0]
                .iter()
                .enumerate()
                .map(|(i, &z)| {
                    let x = 0.05 * i as f64 + 0.9;
                    vec![Point3::new(x, 0.0, z), Point3::new(x, 1.0, z)]
                })
                .collect(),
            vec![vec![1e308; 2]; 3],
        )
        .unwrap();
        for s in [u_wall(), heavy] {
            sub_patch_boxes_hold_the_span_of(&s);
        }
    }

    /// A small surface far from the origin: rounding there is about a ten
    /// thousandth of a unit, far more than a share of the surface's size.
    #[test]
    fn sub_patch_boxes_hold_a_span_far_from_the_origin() {
        let s = u_wall();
        let far = NurbsSurface::new(
            s.degree_u(),
            s.degree_v(),
            s.knots_u().to_vec(),
            s.knots_v().to_vec(),
            s.control_points()
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|p| Point3::new(p.x() + 1e12, p.y() + 1e12, p.z() + 1e12))
                        .collect()
                })
                .collect(),
            s.weights().to_vec(),
        )
        .unwrap();
        sub_patch_boxes_hold_the_span_of(&far);
        let index = SpanIndex::new(&far).unwrap();
        let p = far.evaluate(0.25, 0.0);
        let b = Aabb3 { min: p, max: p };
        assert!(!index.box_clears(&far, &b, 8e-7));
    }

    /// Weights below the least normal float, and weights whose products
    /// with the coordinates overflow, leave a span's net without its
    /// points, so no index is made.
    #[test]
    fn weights_out_of_range_get_no_index() {
        let cps = (0..3)
            .map(|i| {
                (0..2)
                    .map(|j| {
                        Point3::new(f64::from(i), f64::from(j), if i == 1 { 1.0 } else { 0.0 })
                    })
                    .collect()
            })
            .collect();
        let s = NurbsSurface::new(
            2,
            1,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            cps,
            vec![vec![1e-323; 2]; 3],
        )
        .unwrap();
        assert!(SpanIndex::new(&s).is_none());
        // Weights of `1e308` on coordinates of 2 overflow.
        let heavy = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(2.0, 0.0, 0.0), Point3::new(0.0, 1.0, 0.0)],
                vec![Point3::new(0.0, 0.0, 1.0), Point3::new(0.0, 1.0, 1.0)],
            ],
            vec![vec![1e308; 2]; 2],
        )
        .unwrap();
        assert!(SpanIndex::new(&heavy).is_none());
    }

    fn sub_patch_boxes_hold_the_span_of(s: &NurbsSurface) {
        let s = s.clone();
        let index = SpanIndex::new(&s).unwrap();
        for i in 0..index.len() {
            let boxes = index.sub_boxes_of(&s, i).unwrap();
            let span = index.spans[i];
            for a in 0..=32 {
                for b in 0..=8 {
                    let u = span.u.0 + (span.u.1 - span.u.0) * f64::from(a) / 32.0;
                    let v = span.v.0 + (span.v.1 - span.v.0) * f64::from(b) / 8.0;
                    let p = s.evaluate(u, v);
                    assert!(
                        boxes.iter().any(|bx| bx.contains_point(p)),
                        "span {i} at ({u}, {v}): {p:?} in no sub-box"
                    );
                }
            }
        }
    }

    #[test]
    fn sub_patch_boxes_part_a_span_from_points_its_net_holds() {
        let s = u_wall();
        let index = SpanIndex::new(&s).unwrap();
        // Inside the U, within the first span's control box, about a unit
        // from the span itself.
        let p = Point3::new(1.0, 0.5, 1.0);
        assert!(index.boxes[0].distance_squared_to_point(p) <= 0.0);
        let gap = sampled_gap(&s, &index, 0, p);
        let nearest_sub = index
            .sub_boxes_of(&s, 0)
            .unwrap()
            .iter()
            .map(|b| b.distance_squared_to_point(p).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest_sub > 0.2 * gap && nearest_sub <= gap,
            "{nearest_sub} for {gap}"
        );
    }

    #[test]
    fn spans_within_matches_reading_every_span() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        assert_eq!(index.len(), 9);
        for &(x, y, z) in &[
            (0.4, 0.4, 2.0),
            (1.5, 1.5, 0.2),
            (5.0, -1.0, 0.0),
            (1.5, 0.1, -0.3),
        ] {
            let p = Point3::new(x, y, z);
            for bound in [0.05, 0.5, 1.0, 3.0, f64::INFINITY] {
                let mut every: Vec<(usize, f64)> = (0..index.len())
                    .map(|i| (i, index.boxes[i].distance_squared_to_point(p).sqrt()))
                    .filter(|&(_, d)| d < bound)
                    .collect();
                every.sort_by(|a, b| a.1.total_cmp(&b.1));
                assert_eq!(index.spans_within(p, bound), every, "{p:?} within {bound}");
            }
        }
    }

    #[test]
    fn box_clears_only_a_box_off_the_surface() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        let cube = |c: Point3, h: f64| Aabb3 {
            min: Point3::new(c.x() - h, c.y() - h, c.z() - h),
            max: Point3::new(c.x() + h, c.y() + h, c.z() + h),
        };
        // Above the bump's top, clear; around a point on it, not.
        let top = s.evaluate(1.5, 1.5);
        assert!(index.box_clears(&s, &cube(Point3::new(1.5, 1.5, top.z() + 1.0), 0.1), 0.05));
        assert!(!index.box_clears(&s, &cube(top, 0.01), 0.05));
        // A box within reach of a point of the surface is never clear,
        // wherever the point lies.
        for (u, v) in [(0.2, 0.2), (1.0, 2.0), (2.9, 0.4), (1.5, 1.5)] {
            let q = s.evaluate(u, v);
            let near = Point3::new(q.x(), q.y(), q.z() + 0.2);
            assert!(!index.box_clears(&s, &cube(near, 0.05), 0.16), "({u}, {v})");
        }
    }

    #[test]
    fn vertical_line_meets_the_bump_once() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        let hits = index.line_hits(&s, Point3::new(1.5, 1.5, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(hits.len(), 1, "{hits:?}");
        let h = hits[0];
        assert!((h.point - s.evaluate(h.uv.0, h.uv.1)).length() < 1e-9);
        assert!((h.point.x() - 1.5).abs() < 1e-9 && (h.point.y() - 1.5).abs() < 1e-9);
        assert!(!h.grazing);
    }

    #[test]
    fn a_hit_on_a_span_edge_counts_once() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        // u = 1 is an interior knot; find its x at v = 1.5 and shoot there.
        let p = s.evaluate(1.0, 1.5);
        let hits = index.line_hits(
            &s,
            Point3::new(p.x(), p.y(), -5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert_eq!(hits.len(), 1, "{hits:?}");
    }

    #[test]
    fn horizontal_line_through_the_bump_meets_it_twice() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        let top = s.evaluate(1.5, 1.5).z();
        let hits = index.line_hits(
            &s,
            Point3::new(-1.0, 1.5, 0.5 * top),
            Vec3::new(1.0, 0.0, 0.0),
        );
        assert_eq!(hits.len(), 2, "{hits:?}");
    }

    #[test]
    fn closest_point_near_finds_a_point_on_the_patch() {
        let s = bump();
        for &(u, v) in &[(0.3, 2.7), (1.5, 1.5), (2.0, 0.0), (1.0, 1.0)] {
            let q = s.evaluate(u, v);
            let (_, _, dn) = SpanIndex::closest_point_near(&s, q, (u + 0.05, v - 0.05));
            assert!(dn < 1e-9, "({u}, {v}) projected {dn} off from nearby");
        }
    }

    #[test]
    fn line_beside_the_patch_misses() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        let hits = index.line_hits(&s, Point3::new(5.0, 5.0, -5.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(hits.is_empty(), "{hits:?}");
    }

    /// A surface is clear of a point only where every point of it lies at
    /// least that far, and is never clear of a point on it.
    #[test]
    fn clear_of_only_a_point_the_surface_keeps_away_from() {
        let s = u_wall();
        let index = SpanIndex::new(&s).unwrap();
        for x in -2..=6 {
            for z in -3..=5 {
                let p = Point3::new(f64::from(x), 0.5, f64::from(z));
                let nearest = (0..index.len())
                    .map(|i| sampled_gap(&s, &index, i, p))
                    .fold(f64::INFINITY, f64::min);
                for reach in [0.1, 0.5, 1.0, 2.0] {
                    if index.clear_of(&s, p, reach) {
                        assert!(nearest >= reach - 1e-12, "{p:?} at {reach}: {nearest}");
                    }
                }
            }
        }
        let on = s.evaluate(0.7, 0.3);
        assert!(!index.clear_of(&s, on, 1e-9));
        assert!(index.clear_of(&s, Point3::new(2.0, 0.5, 30.0), 10.0));
    }

    /// Two surfaces' spans are apart only where the surfaces are.
    #[test]
    fn boxes_apart_only_for_surfaces_apart() {
        let s = bump();
        let index = SpanIndex::new(&s).unwrap();
        let moved = |dz: f64| {
            let cps: Vec<Vec<Point3>> = s
                .control_points()
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|p| Point3::new(p.x(), p.y(), p.z() + dz))
                        .collect()
                })
                .collect();
            NurbsSurface::new(
                3,
                3,
                s.knots_u().to_vec(),
                s.knots_v().to_vec(),
                cps,
                s.weights().to_vec(),
            )
            .unwrap()
        };
        let far = moved(5.0);
        assert!(index.boxes_apart(SpanIndex::new(&far).as_ref().unwrap(), 1.0));
        assert!(!index.boxes_apart(SpanIndex::new(&far).as_ref().unwrap(), 4.5));
        let near = moved(0.5);
        assert!(!index.boxes_apart(SpanIndex::new(&near).as_ref().unwrap(), 0.0));
    }
}

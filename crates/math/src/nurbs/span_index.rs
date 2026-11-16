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

use crate::aabb::Aabb3;
use crate::bvh::Bvh;
use crate::madd::Madd;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

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
    /// The largest control box diagonal, the scale of the surface's pieces.
    scale: f64,
    /// Each span's sub-patch boxes, by span index, each made the first time
    /// a search reads it.
    sub_boxes: std::sync::OnceLock<Box<[std::sync::OnceLock<SubBoxes>]>>,
    /// Each span's Bezier net, by span index, each made the first time it
    /// is read.
    nets: std::sync::OnceLock<Box<[std::sync::OnceLock<Vec<Hpoint>>]>>,
}

/// The distinct knot spans of one direction: each span's bounds and the
/// index of its last control point.
fn knot_spans(knots: &[f64], degree: usize, n: usize) -> Vec<(f64, f64, usize)> {
    (degree..n)
        .filter(|&i| knots[i + 1] > knots[i])
        .map(|i| (knots[i], knots[i + 1], i))
        .collect()
}

impl SpanIndex {
    /// Index `surface`'s knot spans. `None` when a weight is not positive
    /// (or not a number), where the control boxes need not bound the
    /// surface.
    #[must_use]
    pub fn new(surface: &NurbsSurface) -> Option<Self> {
        if surface
            .weights()
            .iter()
            .flatten()
            .any(|&w| w.is_nan() || w <= 0.0)
        {
            return None;
        }
        let cps = surface.control_points();
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
        let pad = 1e-9 * scale.max(1.0);
        let padded: Vec<(usize, Aabb3)> = boxes
            .iter()
            .enumerate()
            .map(|(i, b)| (i, b.expanded(pad)))
            .collect();
        Some(Self {
            spans,
            boxes,
            bvh: Bvh::build(&padded),
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
        let pad = 1e-9 * self.scale.max(1.0);
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
}

/// The distance between two boxes, 0 where they meet.
fn box_gap(a: &Aabb3, b: &Aabb3) -> f64 {
    let gap = |a_lo: f64, a_hi: f64, b_lo: f64, b_hi: f64| (a_lo - b_hi).max(b_lo - a_hi).max(0.0);
    let x = gap(a.min.x(), a.max.x(), b.min.x(), b.max.x());
    let y = gap(a.min.y(), a.max.y(), b.min.y(), b.max.y());
    let z = gap(a.min.z(), a.max.z(), b.min.z(), b.max.z());
    x.madd(x, y.madd(y, z * z)).sqrt()
}

/// A homogeneous control point: `(w x, w y, w z, w)`.
type Hpoint = [f64; 4];

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
fn span_bezier(points: &[Hpoint], knots: &[f64], i: usize, (a, b): Interval) -> Vec<Hpoint> {
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
fn halve(net: &[Hpoint], nu: usize, nv: usize, along_u: bool) -> (Vec<Hpoint>, Vec<Hpoint>) {
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
}

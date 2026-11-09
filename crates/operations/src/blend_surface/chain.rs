//! A chain of edges read as one curve run by arc length.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::traits::ParametricCurve;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};

use crate::OperationsError;

/// How far apart a chain's edges may end and still join: the same band a
/// sweep's path chain accepts.
const JOIN: f64 = 1e-5;

/// An edge shorter than this contributes no direction to run along.
const NO_LENGTH: f64 = 1e-9;

/// How far a tangent is read inside a piece, so that one read at a kink comes
/// from the piece's own side of it.
const INSIDE: f64 = 1e-12;

/// One edge's curve in the direction a chain runs it, with its arc length.
#[derive(Debug, Clone)]
pub(super) struct RunCurve {
    curve: EdgeCurve,
    start: Point3,
    end: Point3,
    t_from: f64,
    t_to: f64,
    arc: ArcLength,
}

#[derive(Debug, Clone)]
enum ArcLength {
    /// A line or circle: arc length is linear in the parameter.
    Linear(f64),
    /// Cumulative length at the ascending `nodes` of the run `[0, 1]`.
    Table { nodes: Vec<f64>, lengths: Vec<f64> },
}

impl RunCurve {
    pub fn new(curve: EdgeCurve, start: Point3, end: Point3, against: bool) -> Self {
        let (t0, t1) = curve.domain_with_endpoints(start, end);
        let (t_from, t_to) = if against { (t1, t0) } else { (t0, t1) };
        let mut run = Self {
            curve,
            start,
            end,
            t_from,
            t_to,
            arc: ArcLength::Linear(0.0),
        };
        run.arc = match &run.curve {
            EdgeCurve::Line => ArcLength::Linear((end - start).length()),
            EdgeCurve::Circle(c) => ArcLength::Linear(c.radius() * (t1 - t0).abs()),
            EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => run.table(),
        };
        run
    }

    fn table(&self) -> ArcLength {
        let mut nodes = vec![0.0, 1.0];
        nodes.extend(self.interior_knots());
        nodes.sort_by(f64::total_cmp);
        let mut refined = Vec::with_capacity(nodes.len() * 8);
        for pair in nodes.windows(2) {
            for k in 0..8 {
                refined.push(pair[0] + (pair[1] - pair[0]) * f64::from(k) / 8.0);
            }
        }
        refined.push(1.0);
        let mut lengths = Vec::with_capacity(refined.len());
        let mut sum = 0.0;
        lengths.push(0.0);
        for pair in refined.windows(2) {
            sum += self.integrate(pair[0], pair[1]);
            lengths.push(sum);
        }
        ArcLength::Table {
            nodes: refined,
            lengths,
        }
    }

    /// The curve's interior knots on this run, in `[0, 1]` along it.
    fn interior_knots(&self) -> Vec<f64> {
        let EdgeCurve::NurbsCurve(curve) = &self.curve else {
            return Vec::new();
        };
        let mut knots: Vec<f64> = curve
            .knots()
            .iter()
            .map(|k| (k - self.t_from) / (self.t_to - self.t_from))
            .filter(|w| *w > 1e-9 && *w < 1.0 - 1e-9)
            .collect();
        knots.sort_by(f64::total_cmp);
        knots.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
        knots
    }

    fn t_at(&self, w: f64) -> f64 {
        self.t_from + (self.t_to - self.t_from) * w
    }

    fn derivative(&self, t: f64) -> Vec3 {
        match &self.curve {
            EdgeCurve::Line => self.end - self.start,
            EdgeCurve::Circle(c) => ParametricCurve::tangent(c, t) * c.radius(),
            EdgeCurve::Ellipse(e) => ParametricCurve::tangent(e, t),
            EdgeCurve::NurbsCurve(n) => n.derivatives(t, 1)[1],
        }
    }

    fn speed(&self, w: f64) -> f64 {
        self.derivative(self.t_at(w)).length() * (self.t_to - self.t_from).abs()
    }

    fn integrate(&self, w0: f64, w1: f64) -> f64 {
        let half = 0.5 * (w1 - w0);
        gauss_legendre_points(16)
            .iter()
            .map(|g| g.w * half * self.speed(0.5 * (w0 + w1) + half * g.x))
            .sum()
    }

    pub fn length(&self) -> f64 {
        match &self.arc {
            ArcLength::Linear(length) => *length,
            ArcLength::Table { lengths, .. } => lengths[lengths.len() - 1],
        }
    }

    /// The run position `w` at arc-length fraction `x`.
    fn w_at(&self, x: f64) -> f64 {
        let ArcLength::Table { nodes, lengths } = &self.arc else {
            return x;
        };
        let total = lengths[lengths.len() - 1];
        let target = x * total;
        let k = lengths
            .partition_point(|l| *l <= target)
            .clamp(1, nodes.len() - 1);
        let (mut lo, mut hi) = (nodes[k - 1], nodes[k]);
        let base = lengths[k - 1];
        let mut w = lo + (hi - lo) * ((target - base) / (lengths[k] - base).max(f64::MIN_POSITIVE));
        for _ in 0..30 {
            let error = base + self.integrate(nodes[k - 1], w) - target;
            if error.abs() <= 1e-15 * total {
                break;
            }
            if error > 0.0 {
                hi = w;
            } else {
                lo = w;
            }
            let speed = self.speed(w);
            let step = w - error / speed;
            let newton = speed > 0.0 && step >= lo && step <= hi;
            w = if newton { step } else { 0.5 * (lo + hi) };
            // Newton squares the error, so this last step reaches rounding. A
            // fit amplifies noise in its data, and this is where it would start.
            if newton && error.abs() <= 1e-9 * total {
                break;
            }
        }
        w
    }

    /// The arc-length fraction at run position `w`.
    fn x_at(&self, w: f64) -> f64 {
        match &self.arc {
            ArcLength::Linear(_) => w,
            ArcLength::Table { nodes, lengths } => {
                let k = nodes.partition_point(|n| *n <= w).clamp(1, nodes.len() - 1);
                (lengths[k - 1] + self.integrate(nodes[k - 1], w)) / lengths[lengths.len() - 1]
            }
        }
    }

    pub fn point(&self, x: f64) -> Point3 {
        self.curve
            .evaluate_with_endpoints(self.t_at(self.w_at(x)), self.start, self.end)
    }

    /// The unit tangent in the direction of the run, `None` where the curve's
    /// derivative vanishes.
    pub fn tangent(&self, x: f64) -> Option<Vec3> {
        let t = self.t_at(self.w_at(x));
        let d = self.derivative(t) * (self.t_to - self.t_from).signum();
        d.normalize().ok()
    }

    /// The curve's interior knots on this run, as arc-length fractions.
    pub fn knot_fractions(&self) -> Vec<f64> {
        self.interior_knots()
            .into_iter()
            .map(|w| self.x_at(w))
            .collect()
    }
}

/// A piece of a chain: all of an edge unless a closed chain was cut inside it.
#[derive(Debug, Clone)]
pub(super) struct Segment {
    pub edge: Option<EdgeId>,
    pub against: bool,
    pub run: RunCurve,
    x0: f64,
    x1: f64,
    pub s0: f64,
    pub s1: f64,
}

impl Segment {
    fn length(&self) -> f64 {
        self.run.length() * (self.x1 - self.x0)
    }

    fn fraction(&self, s: f64) -> f64 {
        let along = ((s - self.s0) / (self.s1 - self.s0)).clamp(0.0, 1.0);
        self.x0 + (self.x1 - self.x0) * along
    }
}

/// Edges running end to end, parameterised by `s` in `[0, 1]`.
#[derive(Debug, Clone)]
pub(super) struct Chain {
    pub segments: Vec<Segment>,
    pub closed: bool,
    pub length: f64,
}

impl Chain {
    /// The chain through `edges` in the order given, each stored either way.
    pub fn from_edges(topo: &Topology, edges: &[EdgeId]) -> Result<Self, OperationsError> {
        if edges.is_empty() {
            return Err(OperationsError::InvalidInput {
                reason: "an edge chain needs at least one edge".into(),
            });
        }
        for (i, edge) in edges.iter().enumerate() {
            if edges[..i].contains(edge) {
                return Err(OperationsError::InvalidInput {
                    reason: format!("edge {i} of a chain repeats an earlier one"),
                });
            }
        }
        let mut ends = Vec::with_capacity(edges.len());
        for &id in edges {
            let edge = topo.edge(id)?;
            ends.push((
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            ));
        }
        let meet = |p: Point3, q: Point3| (p - q).length() <= JOIN;
        let mut against = Vec::with_capacity(edges.len());
        let mut tail: Option<Point3> = None;
        for (i, &(start, end)) in ends.iter().enumerate() {
            if edges.len() > 1 && meet(start, end) && topo.edge(edges[i])?.is_closed() {
                return Err(OperationsError::InvalidInput {
                    reason: format!("edge {i} of a chain is closed, so nothing can run on from it"),
                });
            }
            let backward = match tail {
                Some(at) if meet(start, at) || meet(end, at) => {
                    (end - at).length() < (start - at).length()
                }
                Some(_) => {
                    return Err(OperationsError::InvalidInput {
                        reason: format!("edge {i} of a chain does not meet the one before it"),
                    });
                }
                None => ends.get(1).is_some_and(|&(next_start, next_end)| {
                    let reaches = |p: Point3| meet(p, next_start) || meet(p, next_end);
                    !reaches(end) && reaches(start)
                }),
            };
            tail = Some(if backward { start } else { end });
            against.push(backward);
        }
        let head = if against[0] { ends[0].1 } else { ends[0].0 };
        let closed = tail.is_some_and(|at| meet(at, head));

        let mut segments = Vec::with_capacity(edges.len());
        for (i, &id) in edges.iter().enumerate() {
            let (start, end) = ends[i];
            let run = RunCurve::new(topo.edge(id)?.curve().clone(), start, end, against[i]);
            if run.length() > NO_LENGTH {
                segments.push(Segment {
                    edge: Some(id),
                    against: against[i],
                    run,
                    x0: 0.0,
                    x1: 1.0,
                    s0: 0.0,
                    s1: 0.0,
                });
            }
        }
        Self::assemble(segments, closed)
    }

    /// The chain of a wire's edges in the wire's own running order: `(edge,
    /// whether the wire runs it as stored)`. The caller vouches that they meet.
    pub fn from_walk(
        topo: &Topology,
        walk: &[(EdgeId, bool)],
        closed: bool,
    ) -> Result<Self, OperationsError> {
        let mut segments = Vec::with_capacity(walk.len());
        for &(id, forward) in walk {
            let edge = topo.edge(id)?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            let run = RunCurve::new(edge.curve().clone(), start, end, !forward);
            if run.length() > NO_LENGTH {
                segments.push(Segment {
                    edge: Some(id),
                    against: !forward,
                    run,
                    x0: 0.0,
                    x1: 1.0,
                    s0: 0.0,
                    s1: 0.0,
                });
            }
        }
        Self::assemble(segments, closed)
    }

    /// A chain of one curve that belongs to no edge.
    pub fn from_curve(
        curve: NurbsCurve,
        start: Point3,
        end: Point3,
    ) -> Result<Self, OperationsError> {
        let closed = (start - end).length() <= JOIN;
        let run = RunCurve::new(EdgeCurve::NurbsCurve(curve), start, end, false);
        Self::assemble(
            vec![Segment {
                edge: None,
                against: false,
                run,
                x0: 0.0,
                x1: 1.0,
                s0: 0.0,
                s1: 0.0,
            }],
            closed,
        )
    }

    fn assemble(segments: Vec<Segment>, closed: bool) -> Result<Self, OperationsError> {
        let length: f64 = segments.iter().map(Segment::length).sum();
        if segments.is_empty() || length <= NO_LENGTH {
            return Err(OperationsError::InvalidInput {
                reason: "an edge chain has no length".into(),
            });
        }
        let mut chain = Self {
            segments,
            closed,
            length,
        };
        chain.spread_by_length();
        Ok(chain)
    }

    /// Give each segment the share of `[0, 1]` its length earns.
    pub fn spread_by_length(&mut self) {
        let mut along = 0.0;
        let total = self.length;
        let last = self.segments.len() - 1;
        for (i, segment) in self.segments.iter_mut().enumerate() {
            segment.s0 = along / total;
            along += segment.run.length() * (segment.x1 - segment.x0);
            segment.s1 = if i == last { 1.0 } else { along / total };
        }
    }

    /// The `s` of each interior join between segments.
    pub fn joints(&self) -> Vec<f64> {
        self.segments[1..]
            .iter()
            .map(|segment| segment.s0)
            .collect()
    }

    /// The `s` of each interior join between different edges: where a boundary
    /// along this chain needs a vertex.
    pub fn edge_joints(&self) -> Vec<f64> {
        self.segments
            .windows(2)
            .filter(|pair| pair[0].edge != pair[1].edge)
            .map(|pair| pair[1].s0)
            .collect()
    }

    /// Put the interior joins at `joints`, one per join.
    pub fn set_joints(&mut self, joints: &[f64]) {
        let last = self.segments.len() - 1;
        for (i, segment) in self.segments.iter_mut().enumerate() {
            segment.s0 = if i == 0 { 0.0 } else { joints[i - 1] };
            segment.s1 = if i == last { 1.0 } else { joints[i] };
        }
    }

    pub fn reversed(&self) -> Self {
        let segments = self
            .segments
            .iter()
            .rev()
            .map(|segment| {
                let (x0, x1) = (1.0 - segment.x1, 1.0 - segment.x0);
                Segment {
                    edge: segment.edge,
                    against: !segment.against,
                    run: RunCurve::new(
                        segment.run.curve.clone(),
                        segment.run.start,
                        segment.run.end,
                        !segment.against,
                    ),
                    x0,
                    x1,
                    s0: 0.0,
                    s1: 0.0,
                }
            })
            .collect();
        let mut chain = Self {
            segments,
            closed: self.closed,
            length: self.length,
        };
        chain.spread_by_length();
        chain
    }

    /// This closed chain started at `s`, cutting the segment there unless the
    /// point is one of its joins.
    pub fn started_at(&self, s: f64) -> Self {
        let at = s.rem_euclid(1.0);
        let reach = 1e-6;
        let mut segments = self.segments.clone();
        let index = segments
            .iter()
            .position(|segment| at < segment.s1)
            .unwrap_or(segments.len() - 1);
        let segment = segments[index].clone();
        if at - segment.s0 <= reach {
            segments.rotate_left(index);
        } else if segment.s1 - at <= reach {
            let turn = (index + 1) % segments.len();
            segments.rotate_left(turn);
        } else {
            let cut = segment.fraction(at);
            let mut before = segment.clone();
            before.x1 = cut;
            let mut after = segment;
            after.x0 = cut;
            let mut order = vec![after];
            order.extend_from_slice(&segments[index + 1..]);
            order.extend_from_slice(&segments[..index]);
            order.push(before);
            segments = order;
        }
        let mut chain = Self {
            segments,
            closed: self.closed,
            length: self.length,
        };
        chain.spread_by_length();
        chain
    }

    /// The segment that holds `s`, a join reading as the one it starts.
    pub fn segment_at(&self, s: f64) -> &Segment {
        let index = self.segments.partition_point(|segment| segment.s1 <= s);
        &self.segments[index.min(self.segments.len() - 1)]
    }

    pub fn point(&self, s: f64) -> Point3 {
        let segment = self.segment_at(s);
        segment.run.point(segment.fraction(s))
    }

    /// The `s` an arc length `along` in from the start, or from the end.
    pub fn s_in_from(&self, along: f64, from_end: bool) -> f64 {
        let count = self.segments.len();
        let mut left = along;
        for k in 0..count {
            let segment = &self.segments[if from_end { count - 1 - k } else { k }];
            let length = segment.length();
            if left <= length {
                let step = (segment.s1 - segment.s0) * (left / length);
                return if from_end {
                    segment.s1 - step
                } else {
                    segment.s0 + step
                };
            }
            left -= length;
        }
        if from_end { 0.0 } else { 1.0 }
    }

    /// The arc length of the chain from `from` to `to`.
    pub fn length_between(&self, from: f64, to: f64) -> f64 {
        self.segments
            .iter()
            .map(|segment| {
                let (lo, hi) = (from.max(segment.s0), to.min(segment.s1));
                if hi > lo {
                    segment.length() * (hi - lo) / (segment.s1 - segment.s0)
                } else {
                    0.0
                }
            })
            .sum()
    }

    /// The point at `s`, the unit tangent there, and the point the tangent was
    /// read at, just inside the piece `(lo, hi)` that `s` is taken to belong
    /// to, so that what stands at a knot reads as this piece's side of it.
    pub fn point_and_tangent(
        &self,
        (lo, hi): (f64, f64),
        s: f64,
    ) -> Result<(Point3, Vec3, Point3), OperationsError> {
        let segment = self.segment_at(0.5 * (lo + hi));
        let point = segment.run.point(segment.fraction(s));
        let inward = if s > 0.5 * (lo + hi) {
            lo - hi
        } else {
            hi - lo
        };
        for step in [0.0, 1e-7, 1e-5, 1e-3] {
            let read = (s + step * inward).clamp(lo + INSIDE, hi - INSIDE);
            let at = segment.fraction(read);
            if let Some(tangent) = segment.run.tangent(at) {
                return Ok((point, tangent, segment.run.point(at)));
            }
        }
        Err(OperationsError::InvalidInput {
            reason: format!("an edge of a chain has no direction near s = {s}"),
        })
    }

    /// The `s` of the interior knots of the chain's spline edges.
    pub fn knot_breaks(&self) -> Vec<f64> {
        let mut breaks = Vec::new();
        for segment in &self.segments {
            for x in segment.run.knot_fractions() {
                if x > segment.x0 && x < segment.x1 {
                    let along = (x - segment.x0) / (segment.x1 - segment.x0);
                    breaks.push(segment.s0 + along * (segment.s1 - segment.s0));
                }
            }
        }
        breaks
    }

    /// The arc-length fraction along this chain of its only curve's parameter
    /// `t`.
    pub fn fraction_of_parameter(&self, t: f64) -> f64 {
        let run = &self.segments[0].run;
        run.x_at((t - run.t_from) / (run.t_to - run.t_from))
    }

    /// The parameter of this chain's only curve at arc-length fraction `x`.
    pub fn parameter_of_fraction(&self, x: f64) -> f64 {
        let run = &self.segments[0].run;
        run.t_at(run.w_at(x))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Ellipse3D;
    use brepkit_math::nurbs::fitting::interpolate;

    use super::*;

    #[test]
    fn an_ellipse_runs_by_arc_length() {
        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            1.0,
        )
        .unwrap();
        let start = ParametricCurve::evaluate(&ellipse, 0.0);
        let end = ParametricCurve::evaluate(&ellipse, std::f64::consts::PI);
        let run = RunCurve::new(EdgeCurve::Ellipse(ellipse), start, end, false);
        let total = run.length();
        let steps = 40;
        let mut last = run.point(0.0);
        for i in 1..=steps {
            let next = run.point(f64::from(i) / f64::from(steps));
            let chord = (next - last).length();
            assert!(chord <= total / f64::from(steps) + 1e-12);
            assert!(chord > 0.9 * total / f64::from(steps));
            last = next;
        }
        assert!((run.point(0.0) - start).length() < 1e-12);
        assert!((run.point(1.0) - end).length() < 1e-9);
    }

    #[test]
    fn a_spline_run_against_itself_mirrors_the_fraction() {
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(3.0, 2.0, 1.0),
            Point3::new(4.0, 0.0, 1.0),
            Point3::new(6.0, -1.0, 0.0),
        ];
        let curve = interpolate(&points, 3).unwrap();
        let (start, end) = (points[0], points[4]);
        let forward = RunCurve::new(EdgeCurve::NurbsCurve(curve.clone()), start, end, false);
        let backward = RunCurve::new(EdgeCurve::NurbsCurve(curve), start, end, true);
        for i in 0..=20 {
            let x = f64::from(i) / 20.0;
            assert!((forward.point(x) - backward.point(1.0 - x)).length() < 1e-9);
            let (a, b) = (
                forward.tangent(x).unwrap(),
                backward.tangent(1.0 - x).unwrap(),
            );
            assert!((a + b).length() < 1e-7, "{a:?} and {b:?}");
        }
    }
}

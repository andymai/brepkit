//! A proof that the march reaches every part of the section of two
//! analytic surfaces that crosses a patch of the first.
//!
//! The patch is cut into leaves, rectangles of its parameters `u` and `v`.
//! On a leaf, `g` is the second surface's signed distance read at the first
//! surface's point. Its second derivatives are bounded by the first
//! surface's derivatives and a bound on the curvature of the second
//! surface's distance near the leaf. A leaf is proved empty where `g` keeps
//! one sign over it. A leaf is simple where one partial derivative of `g`
//! keeps one sign over it, and each of the two edges along the other
//! parameter holds at most one zero of `g`. On a simple leaf, each line
//! along that derivative's parameter holds at most one zero, so the section
//! on the leaf is one arc or none, and each edge holds at most one point of
//! it, where `g` changes sign between the edge's ends. Two simple leaves
//! whose shared edge holds such a point hold one arc of the section, and are
//! linked in one class. Any other leaf is cut, until it is too small for a
//! fit within the linear tolerance of both surfaces to tell its arc from
//! another's.
//!
//! A section loop needs a point where `g`'s derivative along `u` is zero and
//! one where that along `v` is zero, so it cannot lie inside one simple
//! leaf. Each part of the section on the patch that reaches a leaf the fit
//! resolves runs through a class of linked leaves. A class is reached when a
//! marched point lies in it, when a marched curve's fit passes through it,
//! or when a seed from it settles on a marched curve or starts a march.

use std::collections::HashMap;

use super::AnalyticSurface;
use super::section_fit::Carrier;
use crate::MathError;
use crate::aabb::Aabb3;
use crate::madd::Madd;
use crate::vec::{Point3, Vec3};

/// The halvings of a grid cell's side down to the finest leaf.
const DEPTH: u32 = 40;

/// How far the leaves' lattice sits from the seeding grid along `u` and
/// `v`, as shares of a cell: odd counts of the finest steps, so no edge or
/// corner of a leaf falls on a line of the grid. A section of a symmetric
/// pair can run along such a line, or through its crossings, where no
/// edge's sign can be read.
const SHIFT: (f64, f64) = (0.3719, 0.2913);

/// The most leaves a patch is cut into before the proof is given up.
pub(super) const MOST_LEAVES: usize = 1 << 16;

/// How many times the linear tolerance over the sine of the crossing a
/// leaf must reach for a fit within the tolerance to tell its arc from
/// another's: [`Cover::holds`] places a point of a fit to within two and a
/// half times, on each side of a leaf's middle.
const RESOLVED: f64 = 16.0;

/// How many seeds of a class that settle are marched from before it is cut
/// or refused.
const SEED_TRIES: usize = 4;

/// A bound read up by this share, to cover the rounding of its terms.
const UP: f64 = 1.0 + 1e-9;

/// What a leaf holds, as far as its bounds prove.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// Not yet read.
    Open,
    /// No point of the section.
    Empty,
    /// No point of the section that the march could trace: off the region,
    /// or off the second surface's patch.
    Off,
    /// One arc of the section or none; `g`'s derivative along `v` keeps one
    /// sign when `along_u` is false, along `u` when true.
    Simple { along_u: bool },
    /// Too small for a fit within the linear tolerance to resolve its arc
    /// (see [`RESOLVED`]), or past the finest leaf.
    Fine,
    /// Cut into smaller leaves.
    Cut,
}

#[derive(Clone, Copy, Debug)]
struct Leaf {
    u: (u64, u64),
    v: (u64, u64),
    kind: Kind,
    /// Whether the leaf lies inside the region the march follows and on
    /// the second surface's patch, where a simple leaf links to others.
    inside: bool,
    /// A bound on how far a point of the leaf lies from its middle.
    reach: f64,
    /// The first surface's point at the leaf's middle, and the second
    /// surface's distance and normal there.
    middle: Point3,
    gap: f64,
    normal: Vec3,
    /// Whether a seed from the leaf was tried, or a marched curve runs
    /// through it.
    done: bool,
    /// Where a cut leaf's parts start among the leaves, and how many.
    first: usize,
    parts: usize,
}

/// The first surface's point, its two partial derivatives, `g` there, its
/// two partial derivatives, and a bound on the rounding of `g`.
#[derive(Clone, Copy)]
struct Reading {
    p: Point3,
    /// The sine of the angle between the surfaces' normals at `p`.
    sin: f64,
    g: f64,
    gu: f64,
    gv: f64,
    normal: Vec3,
    slack: f64,
}

/// Bounds over a leaf of the lengths of the first surface's first and
/// second partial derivatives.
#[derive(Clone, Copy)]
struct Rates {
    u: f64,
    v: f64,
    uu: f64,
    uv: f64,
    vv: f64,
}

/// Where a ball lies against a box or a patch.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Inside,
    Across,
    Outside,
}

/// The leaves of a patch of `a` against `b`, and the classes of simple
/// leaves that hold one arc of the section.
pub(super) struct Cover<'s, 'a> {
    a: &'s AnalyticSurface<'a>,
    b: &'s AnalyticSurface<'a>,
    u_range: (f64, f64),
    v_range: (f64, f64),
    b_v_range: (f64, f64),
    /// Integer steps across the patch along `u` and `v`.
    steps: u64,
    /// Where the leaves' lattice starts along `u` and `v`, in steps.
    shift: (u64, u64),
    u_periodic: bool,
    v_periodic: bool,
    /// The region the faces can meet in, and the larger one the march
    /// follows.
    region: Option<(Aabb3, Aabb3)>,
    floor: f64,
    /// The most leaves the patch is cut into.
    most: usize,
    leaves: Vec<Leaf>,
    roots: HashMap<(u64, u64), usize>,
    corners: HashMap<(u64, u64), Option<(f64, f64)>>,
}

impl<'s, 'a> Cover<'s, 'a> {
    /// The leaves of the cells `kept` (row along `u`, column along `v`) of
    /// a `grid` by `grid` cut of `a`'s patch, cut until each is empty,
    /// simple, set aside or too fine, or `most` are cut. `floor` is the
    /// linear tolerance.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        a: &'s AnalyticSurface<'a>,
        b: &'s AnalyticSurface<'a>,
        (u_range, v_range): ((f64, f64), (f64, f64)),
        b_v_range: (f64, f64),
        grid: usize,
        kept: &[(usize, usize)],
        region: Option<(Aabb3, Aabb3)>,
        floor: f64,
        most: usize,
    ) -> Self {
        let side = 1_u64 << DEPTH;
        let n = grid as u64;
        let full = |r: (f64, f64)| (r.1 - r.0 - std::f64::consts::TAU).abs() <= 1e-12;
        #[allow(
            clippy::cast_precision_loss,
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss
        )]
        let odd = |share: f64| ((share * side as f64) as u64) | 1;
        let mut cover = Self {
            a,
            b,
            u_range,
            v_range,
            b_v_range,
            steps: n * side,
            shift: (odd(SHIFT.0), odd(SHIFT.1)),
            u_periodic: full(u_range),
            v_periodic: matches!(a, AnalyticSurface::Torus(_)) && full(v_range),
            region,
            floor,
            most,
            leaves: Vec::with_capacity(4 * kept.len()),
            roots: HashMap::with_capacity(kept.len()),
            corners: HashMap::new(),
        };
        for &(i, j) in kept {
            let (i, j) = (i as u64, j as u64);
            for uu in [i * side, (i + 1) * side - 1] {
                for vv in [j * side, (j + 1) * side - 1] {
                    let (Some(u), Some(v)) = (cover.row(uu, true), cover.row(vv, false)) else {
                        continue;
                    };
                    if !cover.roots.contains_key(&(u.0, v.0)) {
                        cover.roots.insert((u.0, v.0), cover.leaves.len());
                        cover.push(u, v);
                    }
                }
            }
        }
        cover.settle_from(0);
        cover
    }

    /// The span of the lattice's cell along `u` (or `v`) that holds the
    /// step `x`: a whole cell where the parameter wraps round, and one cut
    /// at the patch's ends where it does not.
    fn row(&self, x: u64, along_u: bool) -> Option<(u64, u64)> {
        let side = 1_u64 << DEPTH;
        let (shift, periodic) = if along_u {
            (self.shift.0, self.u_periodic)
        } else {
            (self.shift.1, self.v_periodic)
        };
        if periodic {
            let x = if x < shift { x + self.steps } else { x };
            let lo = shift + ((x - shift) >> DEPTH << DEPTH);
            return Some((lo, lo + side));
        }
        if x >= self.steps {
            return None;
        }
        if x < shift {
            return Some((0, shift));
        }
        let lo = shift + ((x - shift) >> DEPTH << DEPTH);
        Some((lo, (lo + side).min(self.steps)))
    }

    fn push(&mut self, u: (u64, u64), v: (u64, u64)) {
        self.leaves.push(Leaf {
            u,
            v,
            kind: Kind::Open,
            inside: false,
            reach: f64::INFINITY,
            middle: Point3::new(0.0, 0.0, 0.0),
            gap: 0.0,
            normal: Vec3::new(0.0, 0.0, 0.0),
            done: false,
            first: 0,
            parts: 0,
        });
    }

    /// Whether every leaf was read: the cuts settled within the budget.
    pub(super) fn complete(&self) -> bool {
        self.leaves.iter().all(|l| l.kind != Kind::Open)
    }

    /// Reads and cuts every open leaf from `start` on, broadest first, up
    /// to the budget: past that the leaves left open prove nothing.
    fn settle_from(&mut self, start: usize) {
        let mut k = start;
        while k < self.leaves.len() && self.leaves.len() <= self.most {
            if self.leaves[k].kind == Kind::Open {
                self.read_leaf(k);
            }
            k += 1;
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn param(&self, uu: u64, vv: u64) -> (f64, f64) {
        let s = self.steps as f64;
        (
            (self.u_range.1 - self.u_range.0).madd(uu as f64 / s, self.u_range.0),
            (self.v_range.1 - self.v_range.0).madd(vv as f64 / s, self.v_range.0),
        )
    }

    /// The leaf's half widths in `u` and `v`, and a step more, as its
    /// middle rounds to a step.
    #[allow(clippy::cast_precision_loss)]
    fn half_widths(&self, leaf: &Leaf) -> (f64, f64) {
        let s = self.steps as f64;
        let half = |w: u64| (w as f64).madd(0.5, 1.0) / s;
        (
            (self.u_range.1 - self.u_range.0) * half(leaf.u.1 - leaf.u.0),
            (self.v_range.1 - self.v_range.0) * half(leaf.v.1 - leaf.v.0),
        )
    }

    fn read(&self, u: f64, v: f64) -> Option<Reading> {
        let (p, au, av) = frame(self.a, u, v);
        let (g, normal) = self.b.distance(p)?;
        if !(g.is_finite() && normal.length().is_finite()) {
            return None;
        }
        let size = p.x().abs().max(p.y().abs()).max(p.z().abs());
        let slack = 128.0 * f64::EPSILON * (self.a.scale(p) + self.b.scale(p) + size);
        let across = au.cross(av);
        let sin = normal.cross(across).length() / across.length();
        Some(Reading {
            p,
            sin: if sin.is_finite() { sin } else { 0.0 },
            g,
            gu: normal.dot(au),
            gv: normal.dot(av),
            normal,
            slack,
        })
    }

    /// `g` at a corner of the leaves, with the bound on its rounding.
    fn corner(&mut self, uu: u64, vv: u64) -> Option<(f64, f64)> {
        let key = (
            if self.u_periodic { uu % self.steps } else { uu },
            if self.v_periodic { vv % self.steps } else { vv },
        );
        if let Some(&known) = self.corners.get(&key) {
            return known;
        }
        let (u, v) = self.param(uu, vv);
        let value = self.read(u, v).map(|r| (r.g, r.slack));
        self.corners.insert(key, value);
        value
    }

    /// Where the ball of `reach` about `p` lies against the region and the
    /// second surface's patch.
    fn place(&self, p: Point3, reach: f64) -> Place {
        let ball = Aabb3 { min: p, max: p }.expanded(reach);
        let in_region = match self.region {
            None => Place::Inside,
            Some((meet, march)) => {
                if !ball.intersects(meet) {
                    Place::Outside
                } else if inside_box(ball, march) {
                    Place::Inside
                } else {
                    Place::Across
                }
            }
        };
        if in_region == Place::Outside {
            return Place::Outside;
        }
        match (in_region, on_patch(self.b, self.b_v_range, p, reach)) {
            (_, Place::Outside) => Place::Outside,
            (Place::Inside, Place::Inside) => Place::Inside,
            _ => Place::Across,
        }
    }

    #[allow(clippy::similar_names, clippy::too_many_lines)]
    fn read_leaf(&mut self, k: usize) {
        let leaf = self.leaves[k];
        let (hu, hv) = self.half_widths(&leaf);
        let (um, vm) = self.param(
            leaf.u.0 + (leaf.u.1 - leaf.u.0) / 2,
            leaf.v.0 + (leaf.v.1 - leaf.v.0) / 2,
        );
        // Each point of the leaf lies within the half widths times the
        // greatest rates at which the surface moves with each parameter over
        // it: near a ball's pole, where it moves slowly along `u`, a leaf is
        // far smaller than its parameters' spans read at the equator.
        let rates = rates(self.a, hv, vm);
        let reach = hu.madd(rates.u, hv * rates.v) * UP;
        let Some(r) = self.read(um, vm).filter(|_| reach.is_finite()) else {
            self.leaves[k].reach = reach;
            self.cut_or_stop(k, (hu * rates.u, hv * rates.v), reach);
            return;
        };
        {
            let leaf = &mut self.leaves[k];
            leaf.reach = reach;
            leaf.middle = r.p;
            leaf.gap = r.g;
            leaf.normal = r.normal;
        }
        let place = self.place(r.p, reach);
        if place == Place::Outside {
            self.leaves[k].kind = Kind::Off;
            return;
        }
        let gap = r.g.abs() - r.slack;
        if gap > reach {
            self.leaves[k].kind = Kind::Empty;
            return;
        }
        let Some(bend) = self.b.curvature_within(r.p, reach) else {
            self.cut_or_stop(k, (hu * rates.u, hv * rates.v), reach);
            return;
        };
        // Where the surfaces cross at an angle whose sine is at most `s`
        // over the leaf, a fit within the linear tolerance of both places
        // the section only to within that tolerance over `s`: a leaf a few
        // times that is below what the fit resolves, as a leaf below the
        // tolerance is where they cross square. Each normal turns by at most
        // its distance's curvature times the reach.
        let fine = RESOLVED * self.floor;
        let crossing = || {
            self.a
                .distance(r.p)
                .zip(self.a.curvature_within(r.p, reach))
                .map_or(1.0, |((_, na), bend_a)| {
                    let turn = (bend_a * reach).madd(1.0 + bend * reach, bend * reach);
                    (na.cross(r.normal).length() + turn).min(1.0)
                })
        };
        if reach <= fine || (reach * r.sin * 0.5 <= fine && reach * crossing() <= fine) {
            self.leaves[k].kind = Kind::Fine;
            return;
        }
        let luu = rates.u.madd(rates.u * bend, rates.uu) * UP;
        let luv = rates.u.madd(rates.v * bend, rates.uv) * UP;
        let lvv = rates.v.madd(rates.v * bend, rates.vv) * UP;
        let (su, sv) = (64.0 * f64::EPSILON * rates.u, 64.0 * f64::EPSILON * rates.v);
        let second = 0.5 * (luu * hu).madd(hu, (2.0 * luv * hu).madd(hv, lvv * hv * hv));
        let first = (r.gu.abs() + su).madd(hu, (r.gv.abs() + sv) * hv);
        if gap > (first + second) * UP {
            self.leaves[k].kind = Kind::Empty;
            return;
        }
        // An edge along `u` (or `v`) holds at most one zero: none where `g`
        // keeps its sign along it, one at most where its derivative along
        // the edge does.
        let single = |edge: Option<Reading>, along_u: bool| {
            let (rate, slope, half, slack) = if along_u {
                (rates.u, luu, hu, su)
            } else {
                (rates.v, lvv, hv, sv)
            };
            edge.is_some_and(|e| {
                let level = e.g.abs() - e.slack;
                let along = if along_u { e.gu.abs() } else { e.gv.abs() };
                let moves =
                    (rate * half).min((along + slack).madd(half, 0.5 * slope * half * half));
                level > moves * UP || along - slack > slope * half * UP
            })
        };
        let (low, high) = (
            self.param(leaf.u.0, leaf.v.0),
            self.param(leaf.u.1, leaf.v.1),
        );
        if r.gv.abs() - sv > luv.madd(hu, lvv * hv) * UP
            && single(self.read(um, low.1), true)
            && single(self.read(um, high.1), true)
        {
            self.simple(k, false, place);
            return;
        }
        if r.gu.abs() - su > luu.madd(hu, luv * hv) * UP
            && single(self.read(low.0, vm), false)
            && single(self.read(high.0, vm), false)
        {
            self.simple(k, true, place);
            return;
        }
        self.cut_or_stop(k, (hu * rates.u, hv * rates.v), reach);
    }

    fn simple(&mut self, k: usize, along_u: bool, place: Place) {
        let leaf = &mut self.leaves[k];
        leaf.kind = Kind::Simple { along_u };
        leaf.inside = place == Place::Inside;
    }

    /// Cuts the leaf across its longer side, or both where neither is more
    /// than twice the other, or stops where it is below the floor or cannot
    /// be cut.
    fn cut_or_stop(&mut self, k: usize, (wide_u, wide_v): (f64, f64), reach: f64) {
        let leaf = self.leaves[k];
        let can_u = leaf.u.1 - leaf.u.0 >= 2;
        let can_v = leaf.v.1 - leaf.v.0 >= 2;
        let cut_u = can_u && (!wide_u.is_finite() || 2.0 * wide_u >= wide_v);
        let cut_v = can_v && (!wide_v.is_finite() || 2.0 * wide_v >= wide_u);
        if reach <= self.floor || !(cut_u || cut_v) {
            self.leaves[k].kind = Kind::Fine;
            return;
        }
        self.leaves[k].kind = Kind::Cut;
        self.leaves[k].first = self.leaves.len();
        let halves = |r: (u64, u64), cut: bool| {
            if cut {
                let mid = r.0 + (r.1 - r.0) / 2;
                vec![(r.0, mid), (mid, r.1)]
            } else {
                vec![r]
            }
        };
        for u in halves(leaf.u, cut_u) {
            for v in halves(leaf.v, cut_v) {
                self.push(u, v);
            }
        }
        self.leaves[k].parts = self.leaves.len() - self.leaves[k].first;
    }

    /// The sign of `g` at a corner, `None` where rounding or an unread
    /// distance leaves it open.
    fn sign(&mut self, uu: u64, vv: u64) -> Option<bool> {
        let (g, slack) = self.corner(uu, vv)?;
        (g.abs() > slack).then_some(g > 0.0)
    }

    /// Whether `g` surely changes sign between two corners.
    fn changes(&mut self, p: (u64, u64), q: (u64, u64)) -> bool {
        matches!((self.sign(p.0, p.1), self.sign(q.0, q.1)), (Some(x), Some(y)) if x != y)
    }

    /// Whether a simple leaf may hold an arc: its corners' signs are not all
    /// surely alike.
    fn may_hold(&mut self, k: usize) -> bool {
        let Leaf { u, v, .. } = self.leaves[k];
        let signs = [
            self.sign(u.0, v.0),
            self.sign(u.1, v.0),
            self.sign(u.0, v.1),
            self.sign(u.1, v.1),
        ];
        !(signs.iter().all(|s| *s == Some(true)) || signs.iter().all(|s| *s == Some(false)))
    }

    /// The classes of leaves still to be seeded: simple leaves that may
    /// hold an arc, linked where they share an edge holding one point of
    /// it and lie inside both the region and the second patch. A class any
    /// of whose leaves was seeded, or holds a marched point, is left out.
    ///
    /// A fine leaf is not seeded: a part of the section that leaves it runs
    /// through a simple leaf, and one that stays in it is finer than a fit
    /// within the linear tolerance resolves.
    pub(super) fn classes(&mut self) -> Vec<Vec<usize>> {
        let mut parent: Vec<usize> = (0..self.leaves.len()).collect();
        let mut live = vec![false; self.leaves.len()];
        for k in 0..self.leaves.len() {
            live[k] = matches!(self.leaves[k].kind, Kind::Simple { .. }) && self.may_hold(k);
        }
        // Edges of linkable leaves along each line: (line, from, to, leaf,
        // whether the leaf lies before the line).
        let mut across_u: Vec<(u64, u64, u64, usize, bool)> = Vec::new();
        let mut across_v: Vec<(u64, u64, u64, usize, bool)> = Vec::new();
        let (wrap_u, wrap_v) = (self.u_periodic, self.v_periodic);
        let steps = self.steps;
        let line = |x: u64, wrap: bool| if wrap { x % steps } else { x };
        for (k, leaf) in self.leaves.iter().enumerate() {
            if live[k] && leaf.inside && matches!(leaf.kind, Kind::Simple { .. }) {
                across_u.push((line(leaf.u.0, wrap_u), leaf.v.0, leaf.v.1, k, false));
                across_u.push((line(leaf.u.1, wrap_u), leaf.v.0, leaf.v.1, k, true));
                across_v.push((line(leaf.v.0, wrap_v), leaf.u.0, leaf.u.1, k, false));
                across_v.push((line(leaf.v.1, wrap_v), leaf.u.0, leaf.u.1, k, true));
            }
        }
        let mut links = Vec::new();
        for (edges, u_line) in [(&mut across_u, true), (&mut across_v, false)] {
            edges.sort_unstable_by_key(|e| (e.0, e.4, e.1));
            let mut i = 0;
            while i < edges.len() {
                let mut j = i;
                while j < edges.len() && edges[j].0 == edges[i].0 {
                    j += 1;
                }
                let group = &edges[i..j];
                let split = group.partition_point(|e| !e.4);
                let (after, before) = group.split_at(split);
                let (mut x, mut y) = (0, 0);
                while x < before.len() && y < after.len() {
                    let (p, q) = (before[x], after[y]);
                    let (lo, hi) = (p.1.max(q.1), p.2.min(q.2));
                    if hi > lo {
                        links.push((p.0, lo, hi, p.3, q.3, u_line));
                    }
                    if p.2 <= q.2 {
                        x += 1;
                    } else {
                        y += 1;
                    }
                }
                i = j;
            }
        }
        for (at, lo, hi, p, q, u_line) in links {
            let (from, to) = if u_line {
                ((at, lo), (at, hi))
            } else {
                ((lo, at), (hi, at))
            };
            if self.changes(from, to) {
                let (rp, rq) = (find(&mut parent, p), find(&mut parent, q));
                parent[rp] = rq;
            }
        }
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for k in 0..self.leaves.len() {
            if live[k] {
                let root = find(&mut parent, k);
                groups.entry(root).or_default().push(k);
            }
        }
        let mut out: Vec<Vec<usize>> = groups
            .into_values()
            .filter(|class| class.iter().all(|&k| !self.leaves[k].done))
            .collect();
        for class in &mut out {
            class.sort_unstable();
        }
        out.sort_unstable();
        out
    }

    /// Points to seed a class from: each point where a leaf's edge holds a
    /// zero of `g`, found by halving the edge.
    pub(super) fn seeds(&mut self, class: &[usize]) -> Vec<Point3> {
        let mut out = Vec::new();
        for &k in class {
            let leaf = self.leaves[k];
            let (u, v) = (leaf.u, leaf.v);
            let corners = [(u.0, v.0), (u.1, v.0), (u.1, v.1), (u.0, v.1)];
            let before = out.len();
            for e in 0..4 {
                let (from, to) = (corners[e], corners[(e + 1) % 4]);
                if self.changes(from, to)
                    && let Some(p) = self.zero_between(from, to)
                {
                    out.push(p);
                }
            }
            // A corner whose sign rounding leaves open lies on the section
            // to within that rounding.
            for &(cu, cv) in &corners {
                if self.sign(cu, cv).is_none() {
                    let (pu, pv) = self.param(cu, cv);
                    out.push(frame(self.a, pu, pv).0);
                }
            }
            if out.len() == before {
                out.push(leaf.middle - leaf.normal * (0.5 * leaf.gap));
            }
        }
        out
    }

    /// The first surface's point where `g` is zero on the edge between two
    /// corners where it has opposite signs, by halving.
    fn zero_between(&mut self, from: (u64, u64), to: (u64, u64)) -> Option<Point3> {
        let low = self.sign(from.0, from.1)?;
        let (u, v) = self.zero_on(self.param(from.0, from.1), self.param(to.0, to.1), low)?;
        Some(frame(self.a, u, v).0)
    }

    /// The parameters where `g` is zero between `(u0, v0)`, where its sign
    /// is `low`, and `(u1, v1)`, where it has the other, by halving.
    fn zero_on(&self, (u0, v0): (f64, f64), (u1, v1): (f64, f64), low: bool) -> Option<(f64, f64)> {
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        for _ in 0..64 {
            let mid = f64::midpoint(lo, hi);
            if mid <= lo || mid >= hi {
                break;
            }
            let r = self.read((u1 - u0).madd(mid, u0), (v1 - v0).madd(mid, v0))?;
            if (r.g > 0.0) == low {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let t = f64::midpoint(lo, hi);
        Some(((u1 - u0).madd(t, u0), (v1 - v0).madd(t, v0)))
    }

    /// A point inside each leaf of a class on its arc: halfway along the
    /// parameter the arc is a graph over, between where it leaves the leaf,
    /// or else the leaf's middle moved onto the section where that lands
    /// inside the leaf, as where the arc ends on a pole.
    pub(super) fn interiors(&mut self, class: &[usize]) -> Vec<Point3> {
        let mut out = Vec::new();
        for &k in class {
            let before = out.len();
            self.interior(k, &mut out);
            if out.len() == before
                && let Some(q) =
                    super::section_fit::settle_on_section(self.a, self.b, self.leaves[k].middle)
                && self.locate(q, 0.0) == Some(k)
            {
                out.push(q);
            }
        }
        out
    }

    fn interior(&mut self, k: usize, out: &mut Vec<Point3>) {
        {
            let leaf = self.leaves[k];
            let Kind::Simple { along_u } = leaf.kind else {
                return;
            };
            let (u, v) = (leaf.u, leaf.v);
            let corners = [(u.0, v.0), (u.1, v.0), (u.1, v.1), (u.0, v.1)];
            let mut ends = Vec::new();
            for e in 0..4 {
                let (from, to) = (corners[e], corners[(e + 1) % 4]);
                if self.changes(from, to)
                    && let Some(low) = self.sign(from.0, from.1)
                    && let Some(end) =
                        self.zero_on(self.param(from.0, from.1), self.param(to.0, to.1), low)
                {
                    ends.push(end);
                }
            }
            let [first, second, ..] = ends[..] else {
                return;
            };
            let (lo, hi) = (self.param(u.0, v.0), self.param(u.1, v.1));
            let (from, to) = if along_u {
                let v = f64::midpoint(first.1, second.1);
                ((lo.0, v), (hi.0, v))
            } else {
                let u = f64::midpoint(first.0, second.0);
                ((u, lo.1), (u, hi.1))
            };
            let (Some(a), Some(b)) = (self.read(from.0, from.1), self.read(to.0, to.1)) else {
                return;
            };
            if a.g.abs() > a.slack
                && b.g.abs() > b.slack
                && (a.g > 0.0) != (b.g > 0.0)
                && let Some((pu, pv)) = self.zero_on(from, to, a.g > 0.0)
            {
                out.push(frame(self.a, pu, pv).0);
            }
        }
    }

    /// Marks every leaf of a class reached.
    pub(super) fn mark(&mut self, class: &[usize]) {
        for &k in class {
            self.leaves[k].done = true;
        }
    }

    /// Cuts a lone leaf that straddles the region's or the second patch's
    /// edge, so its parts either lie off both, or inside both where a seed
    /// reaches them, and reads them; `false` where there is none to cut, or
    /// where the parts cannot all be read within the budget: the cut is then
    /// undone, so every leaf stays read.
    pub(super) fn refine(&mut self, class: &[usize]) -> bool {
        let &[k] = class else {
            return false;
        };
        let leaf = self.leaves[k];
        let start = self.leaves.len();
        if leaf.inside || start + 4 > self.most {
            return false;
        }
        let (hu, hv) = self.half_widths(&leaf);
        let (_, vm) = self.param(leaf.u.0, leaf.v.0 + (leaf.v.1 - leaf.v.0) / 2);
        let rates = rates(self.a, hv, vm);
        self.cut_or_stop(k, (hu * rates.u, hv * rates.v), leaf.reach);
        if self.leaves[k].kind == Kind::Fine {
            self.leaves[k] = leaf;
            return false;
        }
        self.settle_from(start);
        if self.leaves[start..].iter().any(|l| l.kind == Kind::Open) {
            self.leaves.truncate(start);
            self.leaves[k] = leaf;
            return false;
        }
        true
    }

    /// Settles a class. It is reached only where a marched point lies in
    /// one of its leaves, or a fitted curve passes through one: `fitted`
    /// reads whether the curve nearest a probe does, and `seed` settles a
    /// probe and marches from it, returning the points of the march where
    /// it lies on a marched branch (none where it traced nothing), or
    /// `None` where it does not settle. `Ok(true)` where the class is
    /// reached, `Ok(false)` where a lone leaf across an edge was cut and
    /// its parts are to be read again.
    ///
    /// # Errors
    ///
    /// Any error of `seed`, and [`MathError::ConvergenceFailure`] where no
    /// seed reaches the class and it cannot be cut: its arc may lie on both
    /// patches, and it is refused rather than dropped.
    pub(super) fn resolve(
        &mut self,
        class: &[usize],
        fitted: &dyn Fn(&Self, &[usize], Point3) -> bool,
        seed: &mut dyn FnMut(Point3) -> Result<Option<Vec<Point3>>, MathError>,
    ) -> Result<bool, MathError> {
        let mut probes = self.interiors(class);
        probes.extend(self.seeds(class));
        let reached = |cover: &Self| {
            class.iter().any(|&k| cover.leaves[k].done)
                || probes.iter().any(|&z| fitted(cover, class, z))
        };
        if reached(self) {
            self.mark(class);
            return Ok(true);
        }
        let mut tries = 0;
        for &z in &probes {
            let Some(points) = seed(z)? else {
                continue;
            };
            self.mark_marched(&points);
            if reached(self) {
                self.mark(class);
                return Ok(true);
            }
            tries += 1;
            if tries == SEED_TRIES {
                break;
            }
        }
        if self.refine(class) {
            return Ok(false);
        }
        Err(MathError::ConvergenceFailure { iterations: 0 })
    }

    /// Marks the leaf each point of a marched section lies in, where it
    /// lies inside that leaf by more than the rounding of its place on the
    /// section: the leaf's arc is then that section's.
    pub(super) fn mark_marched(&mut self, points: &[Point3]) {
        for &q in points {
            if let Some(k) = self.locate(q, 0.0) {
                self.leaves[k].done = true;
            }
        }
    }

    /// Whether `q`, a point within `off` of both surfaces, lies on the arc
    /// of a leaf of `class`.
    pub(super) fn holds(&self, class: &[usize], q: Point3, off: f64) -> bool {
        self.locate(q, off)
            .is_some_and(|k| class.binary_search(&k).is_ok())
    }

    /// The simple leaf whose arc holds the point of the section nearest
    /// `q`, a point within `off` of both surfaces (and within rounding):
    /// that point lies within `off`, and its rounding, over the sine of the
    /// angle between the surfaces' normals, and this leaf must hold every
    /// point that near `q`. `None` where none is sure.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn locate(&self, q: Point3, off: f64) -> Option<usize> {
        let steps = self.steps as f64;
        let (span_u, span_v) = (
            self.u_range.1 - self.u_range.0,
            self.v_range.1 - self.v_range.0,
        );
        let (Some((_, na)), Some((_, nb))) = (self.a.distance(q), self.b.distance(q)) else {
            return None;
        };
        let sin = na.cross(nb).length();
        let (u, v) = sheet_params(self.a, q, self.u_range, self.v_range);
        let (_, au, av) = frame(self.a, u, v);
        let size = q.x().abs().max(q.y().abs()).max(q.z().abs());
        let rounding = 1024.0 * f64::EPSILON * (self.a.scale(q) + self.b.scale(q) + size);
        let reach = 2.5 * (off + rounding) / sin;
        let (mu, mv) = (
            reach / au.length() / span_u * steps,
            reach / av.length() / span_v * steps,
        );
        let (fu, fv) = (
            (u - self.u_range.0) / span_u * steps,
            (v - self.v_range.0) / span_v * steps,
        );
        let inside_patch = |f: f64, m: f64, periodic: bool| {
            f.is_finite() && (periodic || (f >= m && f <= steps - m))
        };
        if !(mu.is_finite()
            && mv.is_finite()
            && inside_patch(fu, mu, self.u_periodic)
            && inside_patch(fv, mv, self.v_periodic))
        {
            return None;
        }
        let lift = |f: f64, shift: u64, periodic: bool| {
            let f = if periodic { f.rem_euclid(steps) } else { f };
            if periodic && f < shift as f64 {
                f + steps
            } else {
                f
            }
        };
        let (fu, fv) = (
            lift(fu, self.shift.0, self.u_periodic),
            lift(fv, self.shift.1, self.v_periodic),
        );
        let (uu, vv) = (fu as u64, fv as u64);
        let (ru, rv) = (self.row(uu, true)?, self.row(vv, false)?);
        let mut k = *self.roots.get(&(ru.0, rv.0))?;
        while self.leaves[k].kind == Kind::Cut {
            let first = self.leaves[k].first;
            let last = first + self.leaves[k].parts;
            let next = self.leaves[first..last]
                .iter()
                .position(|c| c.u.0 <= uu && uu < c.u.1 && c.v.0 <= vv && vv < c.v.1)?;
            k = first + next;
        }
        let leaf = self.leaves[k];
        let inside = |x: f64, lo: u64, hi: u64, m: f64| x - lo as f64 >= m && hi as f64 - x >= m;
        (matches!(leaf.kind, Kind::Simple { .. })
            && inside(fu, leaf.u.0, leaf.u.1, mu)
            && inside(fv, leaf.v.0, leaf.v.1, mv))
        .then_some(k)
    }
}

fn find(parent: &mut [usize], mut k: usize) -> usize {
    while parent[k] != k {
        parent[k] = parent[parent[k]];
        k = parent[k];
    }
    k
}

/// The parameters of the point of `surface` nearest `q`, within the
/// patch's ranges: on a spindle torus, on whichever sheet `q` lies nearer,
/// where the plain projection reads the near tube's.
fn sheet_params(
    surface: &AnalyticSurface<'_>,
    q: Point3,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> (f64, f64) {
    match surface {
        AnalyticSurface::Torus(t) => {
            let (u, v) = t.project_point_nearest(q);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        _ => super::project_analytic(surface, q, u_range, v_range),
    }
}

/// Whether `inner` lies inside `outer`, off its faces.
fn inside_box(inner: Aabb3, outer: Aabb3) -> bool {
    inner.min.x() > outer.min.x()
        && inner.min.y() > outer.min.y()
        && inner.min.z() > outer.min.z()
        && inner.max.x() < outer.max.x()
        && inner.max.y() < outer.max.y()
        && inner.max.z() < outer.max.z()
}

/// Where the points of `surface` within `reach` of `p` lie against its
/// patch from `v0` to `v1`: its `v` there moves from `p`'s by at most
/// `reach` times the greatest rate at which `v` moves with the point.
fn on_patch(surface: &AnalyticSurface<'_>, (v0, v1): (f64, f64), p: Point3, reach: f64) -> Place {
    let (v, rate, turn) = match surface {
        AnalyticSurface::Cylinder(c) => (c.axis().dot(p - c.origin()), 1.0, None),
        AnalyticSurface::Cone(c) => {
            let sin = c.half_angle().sin();
            (c.axis().dot(p - c.apex()) / sin, 1.0 / sin, None)
        }
        AnalyticSurface::Sphere(s) => {
            let w = p - s.center();
            let r = w.length();
            if r <= reach {
                return Place::Across;
            }
            let z = (s.z_axis().dot(w) / r).clamp(-1.0, 1.0);
            (z.asin(), 1.0 / (r - reach), None)
        }
        AnalyticSurface::Torus(t) => {
            if v1 - v0 >= std::f64::consts::TAU {
                return Place::Inside;
            }
            // A spindle torus's far sheet reads its `v` about the centre
            // circle's far point, which the bound below does not follow.
            if t.minor_radius() >= t.major_radius() {
                return Place::Across;
            }
            let w = p - t.center();
            let h = t.z_axis().dot(w);
            let rho = (w - t.z_axis() * h).length();
            let q = (rho - t.major_radius()).hypot(h);
            if rho <= reach || q <= reach {
                return Place::Across;
            }
            let v = h
                .atan2(rho - t.major_radius())
                .rem_euclid(std::f64::consts::TAU);
            (v, 1.0 / (q - reach), Some(std::f64::consts::TAU))
        }
    };
    let spread = reach * rate * UP;
    let (lo, hi) = (v - spread, v + spread);
    let shifts: &[f64] = match turn {
        Some(t) => &[-t, 0.0, t],
        None => &[0.0],
    };
    if shifts.iter().any(|s| lo + s > v0 && hi + s < v1) {
        Place::Inside
    } else if shifts.iter().all(|s| hi + s < v0 || lo + s > v1) {
        Place::Outside
    } else {
        Place::Across
    }
}

/// The point of `surface` at `(u, v)` and its partial derivatives there.
fn frame(surface: &AnalyticSurface<'_>, u: f64, v: f64) -> (Point3, Vec3, Vec3) {
    let (sin_u, cos_u) = u.sin_cos();
    match surface {
        AnalyticSurface::Cylinder(c) => {
            let out = c.x_axis() * cos_u + c.y_axis() * sin_u;
            let turn = c.y_axis() * cos_u - c.x_axis() * sin_u;
            (
                c.origin() + out * c.radius() + c.axis() * v,
                turn * c.radius(),
                c.axis(),
            )
        }
        AnalyticSurface::Cone(c) => {
            let (sin_a, cos_a) = c.half_angle().sin_cos();
            let out = c.x_axis() * cos_u + c.y_axis() * sin_u;
            let turn = c.y_axis() * cos_u - c.x_axis() * sin_u;
            let ray = out * cos_a + c.axis() * sin_a;
            (c.apex() + ray * v, turn * (v * cos_a), ray)
        }
        AnalyticSurface::Sphere(s) => {
            let (sin_v, cos_v) = v.sin_cos();
            let out = s.x_axis() * cos_u + s.y_axis() * sin_u;
            let turn = s.y_axis() * cos_u - s.x_axis() * sin_u;
            let r = s.radius();
            (
                s.center() + (out * cos_v + s.z_axis() * sin_v) * r,
                turn * (r * cos_v),
                (s.z_axis() * cos_v - out * sin_v) * r,
            )
        }
        AnalyticSurface::Torus(t) => {
            let (sin_v, cos_v) = v.sin_cos();
            let out = t.x_axis() * cos_u + t.y_axis() * sin_u;
            let turn = t.y_axis() * cos_u - t.x_axis() * sin_u;
            let (big, small) = (t.major_radius(), t.minor_radius());
            let tube = small.madd(cos_v, big);
            (
                t.center() + out * tube + t.z_axis() * (small * sin_v),
                turn * tube,
                (t.z_axis() * cos_v - out * sin_v) * small,
            )
        }
    }
}

/// Bounds of the lengths of `surface`'s partial derivatives over the
/// parameters within `hv` of `v`, along `u` anywhere.
fn rates(surface: &AnalyticSurface<'_>, hv: f64, v: f64) -> Rates {
    match surface {
        AnalyticSurface::Cylinder(c) => Rates {
            u: c.radius(),
            v: 1.0,
            uu: c.radius(),
            uv: 0.0,
            vv: 0.0,
        },
        AnalyticSurface::Cone(c) => {
            let cos_a = c.half_angle().cos();
            let far = (v.abs() + hv) * cos_a;
            Rates {
                u: far,
                v: 1.0,
                uu: far,
                uv: cos_a,
                vv: 0.0,
            }
        }
        AnalyticSurface::Sphere(s) => {
            let r = s.radius();
            let (lo, hi) = (v - hv, v + hv);
            // `cos` is greatest nearest 0 and `sin` farthest from it.
            let cos = if lo <= 0.0 && hi >= 0.0 {
                1.0
            } else {
                lo.abs().min(hi.abs()).cos()
            };
            let sin = lo
                .abs()
                .max(hi.abs())
                .min(std::f64::consts::FRAC_PI_2)
                .sin();
            Rates {
                u: r * cos,
                v: r,
                uu: r * cos,
                uv: r * sin,
                vv: r,
            }
        }
        AnalyticSurface::Torus(t) => {
            let (big, small) = (t.major_radius(), t.minor_radius());
            Rates {
                u: big + small,
                v: small,
                uu: big + small,
                uv: small,
                vv: small,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{PI, TAU};

    use super::*;
    use crate::surfaces::{CylindricalSurface, ToroidalSurface};

    /// The spindle torus of radii 1 and 2 about the origin's z axis.
    fn spindle() -> ToroidalSurface {
        ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0, 2.0).unwrap()
    }

    /// A point of a spindle torus's far sheet, where `1 + 2 cos v` is
    /// negative, lies half a turn round from the near tube's point that the
    /// plain projection reads: read there, it lay 0.469 off, and its leaf
    /// could be another class's. It is read on its own sheet.
    #[test]
    fn a_spindle_torus_point_on_its_far_sheet_is_read_there() {
        let torus = spindle();
        let surface = AnalyticSurface::Torus(&torus);
        let q = torus.evaluate(0.0, 0.75 * PI);
        let (u, v) = sheet_params(&surface, q, (0.0, TAU), (0.0, TAU));
        let off = (frame(&surface, u, v).0 - q).length();
        assert!(off < 1e-12, "read {off:.3e} off");
    }

    /// The point at `v` 2.5 of a spindle torus of radii 1 and 2 lies on the
    /// patch from `v` 2.4 to 2.6, but its near tube's parameter is 1.89:
    /// the patch was read as off the ball about it, and a cell holding it
    /// was set aside.
    #[test]
    fn a_spindle_torus_patch_holds_its_far_sheet_points() {
        let torus = spindle();
        let surface = AnalyticSurface::Torus(&torus);
        let p = torus.evaluate(0.0, 2.5);
        assert!(on_patch(&surface, (2.4, 2.6), p, 1e-3) != Place::Outside);
    }

    /// The rod and ring of the two dips in one cell (see
    /// `two_loops_in_one_cell_of_each_patch_are_both_traced`).
    fn rod_and_ring() -> (CylindricalSurface, ToroidalSurface) {
        let ring = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let rod = CylindricalSurface::new(
            Point3::new(0.0, 4.0 - 0.007, 2.1 - 8e-6),
            Vec3::new(1.0, 0.0, 0.0),
            0.6,
        )
        .unwrap();
        (rod, ring)
    }

    fn every_cell(n: usize) -> Vec<(usize, usize)> {
        (0..n).flat_map(|i| (0..n).map(move |j| (i, j))).collect()
    }

    /// A class whose seeds do not settle, or whose march traces nothing, is
    /// not reached: it was marked reached all the same, and its arc was
    /// lost with no error. Only a marched point in one of its leaves
    /// reaches it.
    #[test]
    fn a_class_no_march_reaches_is_not_taken_as_reached() {
        let (rod, ring) = rod_and_ring();
        let (a, b) = (
            AnalyticSurface::Cylinder(&rod),
            AnalyticSurface::Torus(&ring),
        );
        let mut cover = Cover::new(
            &a,
            &b,
            ((0.0, TAU), (-10.0, 10.0)),
            (0.0, TAU),
            32,
            &every_cell(32),
            None,
            1e-7,
            MOST_LEAVES,
        );
        assert!(cover.complete());
        let classes = cover.classes();
        let class = classes
            .iter()
            .find(|c| c.iter().all(|&k| cover.leaves[k].inside))
            .expect("a class inside both patches")
            .clone();
        let never = |_: &Cover<'_, '_>, _: &[usize], _: Point3| false;
        let traced_nothing = cover.resolve(&class, &never, &mut |_: Point3| Ok(Some(Vec::new())));
        assert!(traced_nothing.is_err(), "{traced_nothing:?}");
        let unsettled = cover.resolve(&class, &never, &mut |_: Point3| Ok(None));
        assert!(unsettled.is_err(), "{unsettled:?}");
        assert!(cover.classes().contains(&class));
        let reached = cover.resolve(&class, &never, &mut |z: Point3| Ok(Some(vec![z])));
        assert!(
            reached.is_ok_and(|r| r),
            "a marched point on its arc reaches it"
        );
    }

    /// A lone cell across the region's edge cut past the budget left its
    /// parts unread after the cover was proved complete: the arcs in them
    /// were dropped, and the second patch's seeds no longer ran. The cut is
    /// undone, and every leaf stays read.
    #[test]
    fn a_cut_past_the_budget_is_undone() {
        let (rod, ring) = rod_and_ring();
        let (a, b) = (
            AnalyticSurface::Cylinder(&rod),
            AnalyticSurface::Torus(&ring),
        );
        // The region's edge across one loop, at a few places along it.
        let mut tried = 0;
        for edge in [0.17, 0.2, 0.23, 0.25, 0.27, 0.29] {
            let meet = Aabb3 {
                min: Point3::new(edge, 3.9, 1.4),
                max: Point3::new(1.0, 4.1, 1.6),
            };
            let mut cover = Cover::new(
                &a,
                &b,
                ((0.0, TAU), (-10.0, 10.0)),
                (0.0, TAU),
                32,
                &every_cell(32),
                Some((meet, meet.expanded(1e-4))),
                1e-7,
                MOST_LEAVES,
            );
            assert!(cover.complete());
            let classes = cover.classes();
            let Some(lone) = classes
                .iter()
                .find(|c| c.len() == 1 && !cover.leaves[c[0]].inside)
                .cloned()
            else {
                continue;
            };
            tried += 1;
            let before = cover.leaves.len();
            cover.most = before + 2;
            let cut = cover.refine(&lone);
            assert!(
                cover.complete(),
                "edge {edge}: cut {cut}, a leaf is left unread"
            );
            assert!(!cut && cover.leaves.len() == before, "edge {edge}");
        }
        assert!(tried > 0, "no cell across the region's edge");
    }
}

//! How far a wire drawn on a B-spline surface strays from its chords: a
//! bound over each chord, not a value read at points.
//!
//! A chord runs between the drawn points of two points `A` and `B` of an
//! edge, at the parameters `wa` and `wb`. Each point of the edge between them
//! lies within `d` in space of the surface along `L`, the straight run of
//! parameters from `wa` to `wb`. `d` is the sum of three distances: the
//! edge's from the line `AB`, the line's from the surface along `L`, and the
//! surface's at `wa` and `wb` from `A` and `B`.
//!
//! Take a box `R` of parameters that reaches `r` past `L` in the drawing's
//! measure, and a lower bound `λ` with `|S(w1) - S(w2)| ≥ λ |w1 - w2|` for
//! all `w1` and `w2` in `R`. An edge point with its parameters in `R` then
//! lies within `d / λ` of `L`. Where `d / λ < r`, the edge does not leave
//! `R`. It starts on `L`, and on its way out of `R` it would pass a point of
//! `R` more than `d / λ` from `L`. In the drawing's measure the drawing moves
//! no point farther than its parameters move. So the drawn edge lies within
//! `d / λ` of the drawing of `L`. A polar drawing bows `L` off the drawn
//! chord by at most an eighth of its second derivative along `L`.
//!
//! The measure is the parameters' fractions. On a polar drawing the measure
//! first stretches the fraction round by `2π ρ`, `ρ` the most radius on `R`.
//! Then the drawing moves no point farther than the measure does. Near a
//! pole the surface shrinks a step round to nothing, and the measure shrinks
//! it as much, so `λ` stays clear of nothing.
//!
//! `λ` comes from `H = (W S, W)`, the homogeneous form of the Bezier nets of
//! the knot cells that `R` meets. With `S2 = S(w2)` and `s` the surface's
//! point at the middle of `L`, `F = H.xyz - s W - (S2 - s) W` is
//! `W (S - S2)`. It is a polynomial on each cell, continuous where the
//! surface is, and nothing at `w2`. So `|S(w1) - S2| = |F(w1)| / W(w1)`, and
//! `F(w1)` is the step `w1 - w2` times the mean of `F'` along it. That holds
//! across a knot where `F'` breaks too: `F` is continuous there.
//!
//! On each cell, each column of `D(H.xyz - s W)` lies in the hull of its
//! net, so every mean lies in the hull of the matrices that pair a point of
//! one column's net with a point of the other's, over all the cells. Take a
//! reference `M` in that hull, `U` an orthonormal frame of its columns, and
//! `K = (Uᵀ M)⁻¹`. For a unit step `d` and a mean `G`,
//! `|G d| ≥ dᵀ K Uᵀ G d / ‖K‖ ≥ (1 + μ) σ`, with `σ` the least singular value
//! of `M` and `μ` the least eigenvalue of the symmetric part of
//! `K Uᵀ (G - M)`. That least eigenvalue is concave in `G`, so its least over
//! the hull lies at a pair of net points. The `(S2 - s) W` part takes away
//! at most `|S2 - s|` times the most `|DW|`.
//!
//! A knot repeated as often as the degree breaks `F'`, and the two sides'
//! nets then both count in the hull. A reference between them still reads
//! the step's mean across the break. A knot repeated more often may break
//! the surface itself, and leaves the chord without a bound.
//!
//! Near a pole of a polar drawing, the edge's distance from the pole bounds
//! its drawn radius instead. See [`Bend::pole`].

use std::f64::consts::TAU;
use std::rc::Rc;

use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::{Bend, MOST_CELLS};
use crate::properties::boundary::{BoundaryEdge, edge_pieces};

/// A knot cell's Bezier net in homogeneous form `[w x, w y, w z, w]`, with
/// `u` the outer index: point `(i, j)` at `i (q + 1) + j`.
pub(super) type Net = Rc<[[f64; 4]]>;

/// How a B-spline surface's drawing stands to its parameters' fractions.
#[derive(Clone, Copy, Debug)]
pub(super) enum Frame {
    /// The fractions as they are.
    Flat,
    /// `ρ (cos 2π x, sin 2π x)`, with `x` the fraction of parameter `round`
    /// and `ρ = base + sign y`, `y` the fraction of the other.
    Polar {
        round: usize,
        base: f64,
        sign: f64,
        /// The pole at the drawing's origin, where the surface has one.
        pole: Option<Pole>,
    },
}

/// What bounds the drawn radius near a pole: a point of the surface within
/// `depth` of the pole in the fraction along has a drawn radius of at most
/// `gain |p - at| + slack`.
#[derive(Clone, Copy, Debug)]
pub(super) struct Pole {
    at: Point3,
    /// The fraction along at the pole: 0 or 1.
    y: f64,
    depth: f64,
    gain: f64,
    slack: f64,
}

/// A chord of a wire on a B-spline surface.
pub(super) struct Chord {
    /// The parameters' fractions at its start.
    pub from: (f64, f64),
    /// The run of the fractions to its end, the short way round a closed
    /// parameter.
    pub by: (f64, f64),
    /// Its drawn ends.
    pub ends: (Point2, Point2),
    /// The edge's points at its ends.
    pub ends_3d: (Point3, Point3),
    /// How far the edge strays from the line between those points.
    pub off: f64,
    /// How far the surface at the ends' parameters lies from those points.
    pub miss: f64,
}

/// The part of a knot cell in a box of parameters: its net, and the box's
/// fractions in the cell, unwrapped round a closed parameter.
struct Part {
    net: Vec<[f64; 4]>,
    lo: [f64; 2],
    hi: [f64; 2],
}

/// A knot span's run in a range of a parameter: the index of its first
/// knot, the range's fractions in it, unwrapped, and their shares of the way
/// across it.
struct Run {
    knot: usize,
    lo: f64,
    hi: f64,
    share: (f64, f64),
}

impl<'s> Bend<'s> {
    /// The bounds of a B-spline surface drawn as `frame` describes, closed
    /// in the parameters `closed`. The frame's pole is filled in later. A
    /// surface whose knots are not all finite and rising gets no bound.
    pub(super) fn new(surface: &'s NurbsSurface, closed: (bool, bool), frame: Frame) -> Self {
        let (u0, u1) = surface.domain_u();
        let (v0, v1) = surface.domain_v();
        let (pu, pv) = (u1 - u0, v1 - v0);
        let rows = surface.control_points();
        let weights = surface.weights();
        let (nu, nv) = (rows.len(), rows.first().map_or(0, Vec::len));
        let rising = |knots: &[f64]| {
            knots.iter().all(|k| k.is_finite()) && knots.windows(2).all(|w| w[0] <= w[1])
        };
        let sound = rising(surface.knots_u()) && rising(surface.knots_v());
        let fractions = |knots: &[f64], degree: usize, count: usize, start: f64, length: f64| {
            let k = knots.get(degree..=count).filter(|_| sound && length > 0.0);
            k.map_or_else(Vec::new, |k| {
                k.iter().map(|&k| (k - start) / length).collect()
            })
        };
        let column = |j: usize| -> (Vec<Point3>, Vec<f64>) {
            (
                rows.iter().filter_map(|r| r.get(j).copied()).collect(),
                weights.iter().filter_map(|r| r.get(j).copied()).collect(),
            )
        };
        // The surface runs along its first and last rows only on knots
        // clamped at both ends.
        let clamped = |knots: &[f64], degree: usize, count: usize| {
            knots.len() == count + degree + 1
                && knots.first() >= knots.get(degree)
                && knots.last() <= knots.get(count)
        };
        let seam_u = match (rows.first(), rows.last(), weights.first(), weights.last()) {
            (Some(a), Some(b), Some(wa), Some(wb))
                if closed.0 && clamped(surface.knots_u(), surface.degree_u(), nu) =>
            {
                seam(a, b, wa, wb)
            }
            _ => None,
        };
        let seam_v = if closed.1 && nv > 0 && clamped(surface.knots_v(), surface.degree_v(), nv) {
            let ((a, wa), (b, wb)) = (column(0), column(nv - 1));
            seam(&a, &b, &wa, &wb)
        } else {
            None
        };
        // One power of two for every weight brings the most to between one
        // and two: the surface stays the same, and the nets' products keep
        // their digits.
        let most = weights
            .iter()
            .flatten()
            .fold(0.0_f64, |m, w| m.max(w.abs()));
        let unit = if most.is_finite() && most > 0.0 {
            #[allow(clippy::cast_possible_truncation)]
            let e = most.log2().floor() as i32;
            2.0_f64.powi(-e)
        } else {
            1.0
        };
        Self {
            surface,
            start: (u0, v0),
            length: (pu, pv),
            closed,
            frame,
            unit,
            seams: [seam_u, seam_v],
            fractions: [
                fractions(surface.knots_u(), surface.degree_u(), nu, u0, pu),
                fractions(surface.knots_v(), surface.degree_v(), nv, v0, pv),
            ],
            nets: std::cell::RefCell::new(std::collections::HashMap::new()),
        }
    }

    /// The parameters at `p`, each the fraction of its span, and how far
    /// the surface there lies from `p`.
    pub(super) fn params(&self, p: Point3) -> ((f64, f64), f64) {
        use brepkit_math::traits::ParametricSurface;
        let (u, v) = self.surface.project_point(p);
        let miss = (self.surface.evaluate(u, v) - p).length();
        (
            (
                (u - self.start.0) / self.length.0,
                (v - self.start.1) / self.length.1,
            ),
            miss,
        )
    }

    /// A bound on how far the drawn edge between the chord's ends strays
    /// from the drawn chord; infinite where none holds. A bound at or under
    /// `target` from the pole ends the search.
    pub(super) fn sag(&self, chord: &Chord, target: f64) -> f64 {
        if self.fractions.iter().any(|f| f.len() < 2) {
            return f64::INFINITY;
        }
        let near = self.near_pole(chord).unwrap_or(f64::INFINITY);
        if near <= target {
            return near;
        }
        let Some(stray) = self.stray(chord.from, chord.by) else {
            return near;
        };
        let d = chord.off + stray + chord.miss;
        self.spread(chord.from, chord.by, d)
            .map_or(near, |far| far.min(near))
    }

    /// The degree, the knots and the count of control points along
    /// parameter `d`.
    fn along(&self, d: usize) -> (usize, &[f64], usize) {
        let s = self.surface;
        if d == 0 {
            (s.degree_u(), s.knots_u(), s.control_points().len())
        } else {
            (
                s.degree_v(),
                s.knots_v(),
                s.control_points().first().map_or(0, Vec::len),
            )
        }
    }

    const fn closes(&self, d: usize) -> bool {
        if d == 0 { self.closed.0 } else { self.closed.1 }
    }

    /// The net of the knot cell whose first knots have indices `(iu, iv)`,
    /// read once. `None` where a weight on the way is not positive.
    pub(super) fn net(&self, iu: usize, iv: usize) -> Option<Net> {
        if let Some(net) = self.nets.borrow().get(&(iu, iv)) {
            return Some(Rc::clone(net));
        }
        let net: Net = cell_net(self.surface, iu, iv, self.unit)?.into();
        #[cfg(test)]
        super::CELLS_READ.with(|read| read.borrow_mut().insert((iu, iv)));
        self.nets.borrow_mut().insert((iu, iv), Rc::clone(&net));
        Some(net)
    }

    /// The knot span along parameter `d` that holds the fraction `x`, taken
    /// from below `x` or from above it where `x` lies on a knot: the index
    /// of its first knot, `x`'s share of the way across it, and its length
    /// as a fraction. `None` off an open parameter's span.
    fn locate(&self, d: usize, x: f64, below: bool) -> Option<(usize, f64, f64)> {
        let f = &self.fractions[d];
        let (degree, ..) = self.along(d);
        let x = if self.closes(d) {
            let x = x - x.floor();
            if below && x == 0.0 { 1.0 } else { x }
        } else if (0.0..=1.0).contains(&x) {
            x
        } else {
            return None;
        };
        let last = f.len().checked_sub(2)?;
        let mut k = if below {
            f.partition_point(|&a| a < x)
        } else {
            f.partition_point(|&a| a <= x)
        }
        .saturating_sub(1)
        .min(last);
        // Step off empty spans, toward `x`'s side.
        loop {
            if f[k] < f[k + 1] {
                break;
            }
            k = if below { k.checked_sub(1)? } else { k + 1 };
            if k > last {
                return None;
            }
        }
        let width = f[k + 1] - f[k];
        Some((degree + k, ((x - f[k]) / width).clamp(0.0, 1.0), width))
    }

    /// The surface's homogeneous point at the fractions `w`, with its
    /// partials by the fractions. On a knot of parameter `d` the partials
    /// come from below it where `below[d]`, and from above it otherwise.
    fn at(&self, w: [f64; 2], below: [bool; 2]) -> Option<[[f64; 4]; 3]> {
        let (iu, a, wu) = self.locate(0, w[0], below[0])?;
        let (iv, b, wv) = self.locate(1, w[1], below[1])?;
        let net = self.net(iu, iv)?;
        let degrees = (self.surface.degree_u(), self.surface.degree_v());
        let [h, ha, hb] = net_at(&net, degrees, (a, b));
        Some([h, ha.map(|x| x / wu), hb.map(|x| x / wv)])
    }

    /// The runs of the knot spans along parameter `d` in the fractions from
    /// `lo` to `hi`. `None` past [`MOST_CELLS`] runs, or for a range that
    /// spans more than two periods.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn runs(&self, d: usize, lo: f64, hi: f64) -> Option<Vec<Run>> {
        let f = &self.fractions[d];
        let (degree, ..) = self.along(d);
        let (first, last) = if self.closes(d) {
            (lo.floor(), hi.floor())
        } else {
            (0.0, 0.0)
        };
        if !(first.is_finite() && last.is_finite()) || last - first > 2.0 {
            return None;
        }
        let mut out = Vec::new();
        for m in (first as i64)..=(last as i64) {
            let m = m as f64;
            let start = f.partition_point(|&k| k + m <= lo).saturating_sub(1);
            for k in start..f.len().saturating_sub(1) {
                let (a, b) = (f[k] + m, f[k + 1] + m);
                if a >= hi {
                    break;
                }
                let (x, y) = (a.max(lo), b.min(hi));
                if a < b && x < y {
                    out.push(Run {
                        knot: degree + k,
                        lo: x,
                        hi: y,
                        share: (
                            ((x - a) / (b - a)).clamp(0.0, 1.0),
                            ((y - a) / (b - a)).clamp(0.0, 1.0),
                        ),
                    });
                }
                if out.len() > MOST_CELLS {
                    return None;
                }
            }
        }
        Some(out)
    }

    /// The parts of the knot cells in the box of fractions from `lo` to
    /// `hi`. `None` past [`MOST_CELLS`] cells, or where a net does not
    /// read.
    fn parts(&self, lo: [f64; 2], hi: [f64; 2]) -> Option<Vec<Part>> {
        let along_u = self.runs(0, lo[0], hi[0])?;
        let along_v = self.runs(1, lo[1], hi[1])?;
        if along_u.len() * along_v.len() > MOST_CELLS {
            return None;
        }
        let degrees = (self.surface.degree_u(), self.surface.degree_v());
        let mut parts = Vec::with_capacity(along_u.len() * along_v.len());
        for ru in &along_u {
            for rv in &along_v {
                let net = self.net(ru.knot, rv.knot)?;
                parts.push(Part {
                    net: sub_net(&net, degrees, ru.share, rv.share),
                    lo: [ru.lo, rv.lo],
                    hi: [ru.hi, rv.hi],
                });
            }
        }
        Some(parts)
    }

    /// Whether an inner knot repeated more often than the degree lies
    /// strictly inside the box: the surface may break there.
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn tears(&self, lo: [f64; 2], hi: [f64; 2]) -> bool {
        for d in 0..2 {
            let f = &self.fractions[d];
            let (degree, ..) = self.along(d);
            let inner = f.get(1..f.len().saturating_sub(1)).unwrap_or(&[]);
            let (first, last) = if self.closes(d) {
                (lo[d].floor(), hi[d].floor())
            } else {
                (0.0, 0.0)
            };
            if !(first.is_finite() && last.is_finite()) || last - first > 2.0 {
                return true;
            }
            for m in (first as i64)..=(last as i64) {
                let m = m as f64;
                // The knots strictly inside the box: every copy of a knot
                // lies inside or none does.
                let start = inner.partition_point(|&x| x + m <= lo[d]);
                let end = inner.partition_point(|&x| x + m < hi[d]);
                let mut k = start;
                while k < end {
                    let x = inner[k];
                    // The knots rise, so those not past `x` are `x`.
                    let times = inner[k..end].iter().take_while(|&&y| y <= x).count().max(1);
                    if times > degree {
                        return true;
                    }
                    k += times;
                }
            }
        }
        false
    }

    /// The gap the surface may leave across the seams that the box crosses.
    /// `None` across a seam whose weights differ on its two sides.
    fn seams_crossed(&self, lo: [f64; 2], hi: [f64; 2]) -> Option<f64> {
        let mut gap = 0.0;
        for d in 0..2 {
            let crossed = hi[d].ceil() - lo[d].floor() - 1.0;
            if self.closes(d) && crossed > 0.0 {
                gap += self.seams[d]? * crossed;
            }
        }
        Some(gap)
    }

    /// The drawing's measure on the box: one for each fraction, but for the
    /// fraction round a polar drawing, `2π` times the most radius on the
    /// box. `None` where that radius is not positive.
    fn measure(&self, lo: [f64; 2], hi: [f64; 2]) -> Option<[f64; 2]> {
        match self.frame {
            Frame::Flat => Some([1.0, 1.0]),
            Frame::Polar {
                round, base, sign, ..
            } => {
                let y = 1 - round;
                let rho = sign.madd(lo[y], base).max(sign.madd(hi[y], base));
                let mut out = [1.0; 2];
                out[round] = TAU * rho;
                (rho > 0.0 && rho.is_finite()).then_some(out)
            }
        }
    }

    /// How far a polar drawing bows the run from `wa` to `wb` off the
    /// chord between its drawn ends: an eighth of the most of its second
    /// derivative, `(2π Δx)² ρ + 4π |Δx Δy|`.
    fn bow(&self, wa: [f64; 2], wb: [f64; 2]) -> f64 {
        match self.frame {
            Frame::Flat => 0.0,
            Frame::Polar {
                round, base, sign, ..
            } => {
                let y = 1 - round;
                let dx = (wb[round] - wa[round]).abs();
                let dy = (wb[y] - wa[y]).abs();
                let rho = sign.madd(wa[y], base).max(sign.madd(wb[y], base)).max(0.0);
                let turn = TAU * dx;
                (turn * turn).madd(rho, 2.0 * turn * dy) / 8.0
            }
        }
    }

    /// The bound of the argument at the head of this module, for an edge
    /// within `d` in space of the surface along the chord's parameters.
    /// `None` where no box round the chord holds the edge.
    fn spread(&self, from: (f64, f64), by: (f64, f64), d: f64) -> Option<f64> {
        let wa = [from.0, from.1];
        let wb = [from.0 + by.0, from.1 + by.1];
        let mid = [f64::midpoint(wa[0], wb[0]), f64::midpoint(wa[1], wb[1])];
        let [h, hx, hy] = self.at(mid, [false; 2])?;
        let s0 = Point3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]);
        let reference = |measure: [f64; 2]| {
            [
                moved(hx, s0) * (1.0 / measure[0]),
                moved(hy, s0) * (1.0 / measure[1]),
            ]
        };
        let lo_l = [wa[0].min(wb[0]), wa[1].min(wb[1])];
        let hi_l = [wa[0].max(wb[0]), wa[1].max(wb[1])];
        let first = reference(self.measure(lo_l, hi_l)?);
        let sigma = least_singular(first[0], first[1]) / h[3];
        if !(sigma > 0.0 && sigma.is_finite() && d.is_finite()) {
            return None;
        }
        let mut r = (2.0 * d / sigma).max(1e-12);
        // Each box that does not hold the edge gives the next one's reach:
        // where the stretch falls off past the box, a wider box finds it.
        for _ in 0..8 {
            let (mut lo, mut hi) = (lo_l, hi_l);
            let grow = |lo: &mut [f64; 2], hi: &mut [f64; 2], k: usize, by: f64| {
                lo[k] -= by;
                hi[k] += by;
                if !self.closes(k) {
                    lo[k] = lo[k].max(0.0);
                    hi[k] = hi[k].min(1.0);
                }
            };
            // The fraction along first: its reach sets the polar measure
            // round.
            match self.frame {
                Frame::Flat => {
                    grow(&mut lo, &mut hi, 0, r);
                    grow(&mut lo, &mut hi, 1, r);
                }
                Frame::Polar { round, .. } => {
                    grow(&mut lo, &mut hi, 1 - round, r);
                    let measure = self.measure(lo, hi)?;
                    grow(&mut lo, &mut hi, round, r / measure[round]);
                }
            }
            let measure = self.measure(lo, hi)?;
            let reach = d + self.seams_crossed(lo, hi)?;
            if self.tears(lo, hi) {
                return None;
            }
            let parts = self.parts(lo, hi)?;
            let stretch = self.least_stretch(&parts, s0, reference(measure), measure);
            // A stretch that is not finite bounds nothing: its sums lost
            // their digits.
            if !stretch.is_finite() || stretch <= 0.0 {
                return None;
            }
            if reach < stretch * r {
                let sag = reach / stretch + self.bow(wa, wb);
                return sag.is_finite().then_some(sag);
            }
            r = 2.0 * reach / stretch;
            if r.is_nan() || r >= 1.0 {
                return None;
            }
        }
        None
    }

    /// The least, the most weight on the parts, and the most distance of
    /// their nets' points from `s`.
    fn hull_of(parts: &[Part], s: Point3) -> (f64, f64, f64) {
        let mut out = (f64::INFINITY, 0.0_f64, 0.0_f64);
        for c in parts.iter().flat_map(|part| part.net.iter()) {
            out.0 = out.0.min(c[3]);
            out.1 = out.1.max(c[3]);
            if c[3] > 0.0 {
                let p = Point3::new(c[0] / c[3], c[1] / c[3], c[2] / c[3]);
                out.2 = out.2.max((p - s).length());
            }
        }
        out
    }

    /// `λ` over the parts, by the argument at the head of this module, with
    /// `at` the columns of `D(H.xyz - s W)` at the middle of the chord. Of
    /// that reference and the mean of the nets' points, the better one
    /// counts. Nothing or less where neither bounds anything.
    fn least_stretch(&self, parts: &[Part], s: Point3, at: [Vec3; 2], measure: [f64; 2]) -> f64 {
        let (least, most, spread) = Self::hull_of(parts, s);
        let (p, q) = (self.surface.degree_u(), self.surface.degree_v());
        if parts.is_empty() || p == 0 || q == 0 || least.is_nan() || least <= 0.0 {
            return 0.0;
        }
        let sv = Vec3::new(s.x(), s.y(), s.z());
        let mut columns: Vec<[Vec<Vec3>; 2]> = Vec::with_capacity(parts.len());
        let (mut sum, mut count) = ([Vec3::new(0.0, 0.0, 0.0); 2], [0.0_f64; 2]);
        let mut slope = [0.0_f64; 2];
        for part in parts {
            #[allow(clippy::cast_precision_loss)]
            let factor = [
                p as f64 / ((part.hi[0] - part.lo[0]) * measure[0]),
                q as f64 / ((part.hi[1] - part.lo[1]) * measure[1]),
            ];
            let mut nets = [Vec::new(), Vec::new()];
            for i in 0..=p {
                for j in 0..=q {
                    let c = part.net[i * (q + 1) + j];
                    for (k, next) in [(0, i < p), (1, j < q)] {
                        if !next {
                            continue;
                        }
                        let n = if k == 0 {
                            part.net[(i + 1) * (q + 1) + j]
                        } else {
                            part.net[i * (q + 1) + j + 1]
                        };
                        let dw = n[3] - c[3];
                        let dp = (Vec3::new(n[0] - c[0], n[1] - c[1], n[2] - c[2]) - sv * dw)
                            * factor[k];
                        slope[k] = slope[k].max((dw * factor[k]).abs());
                        sum[k] += dp;
                        count[k] += 1.0;
                        nets[k].push(dp);
                    }
                }
            }
            columns.push(nets);
        }
        let mean = [sum[0] * (1.0 / count[0]), sum[1] * (1.0 / count[1])];
        let bound = tangent_bound(at, &columns).max(tangent_bound(mean, &columns));
        (bound - spread * slope[0].hypot(slope[1])) / most
    }

    /// The bound near the pole at the drawing's origin, for a chord whose
    /// ends lie in the knot cells next to the pole. The edge's distance from
    /// the pole gives the most radius `ρ` for each of its points in those
    /// cells. Where that `ρ` stays inside the cells, the edge does not leave
    /// them: on its way out it would pass a point of the cells farther out.
    /// So the drawn edge lies within `ρ` of the origin, and the drawn chord
    /// passes within its own distance from the origin.
    fn near_pole(&self, chord: &Chord) -> Option<f64> {
        let Frame::Polar {
            round,
            pole: Some(pole),
            ..
        } = self.frame
        else {
            return None;
        };
        let y = 1 - round;
        let ya = [chord.from.0, chord.from.1][y];
        let yb = ya + [chord.by.0, chord.by.1][y];
        if (ya - pole.y).abs() > pole.depth || (yb - pole.y).abs() > pole.depth {
            return None;
        }
        let (a, b) = chord.ends_3d;
        let far = (a - pole.at).length().max((b - pole.at).length()) + chord.off;
        let most = pole.gain.madd(far, pole.slack);
        (most < pole.depth).then(|| most + to_segment(Point2::new(0.0, 0.0), chord.ends))
    }

    /// The bound near the pole at the end of the fraction along that
    /// `at_end` gives, on a surface closed round parameter `round`.
    ///
    /// On each knot cell next to the pole, take `t` as the share of the way
    /// out of the cell from the pole, and `N_k` as the rows of `H - P W` out
    /// from the pole, `P` the pole. Then `H - P W = B_0(t) N_0 + t U`, with
    /// `U` the patch whose rows are `(q / k) N_k`, `k` from 1 to `q`, since
    /// `B_k^q(t) = t (q / k) B_(k-1)^(q-1)(t)`. `N_0` is nothing but for the
    /// pole's rounding. So with `|U| ≥ m` from the hull of its net,
    /// `|S - P| ≥ (t m - |N_0|) / W`, and `t` is at most
    /// `(W |S - P| + |N_0|) / m`. `None` where a hull leaves `U` at nothing.
    pub(super) fn pole(&self, round: usize, at_end: bool) -> Option<Pole> {
        let y = 1 - round;
        let f = &self.fractions[y];
        let (degree_y, ..) = self.along(y);
        let (degree_r, ..) = self.along(round);
        let spans = 0..f.len().saturating_sub(1);
        let k = if at_end {
            spans.rev().find(|&k| f[k] < f[k + 1])?
        } else {
            spans.into_iter().find(|&k| f[k] < f[k + 1])?
        };
        let depth = f[k + 1] - f[k];
        let rows = self.surface.control_points();
        let side: Vec<Point3> = match (y, at_end) {
            (1, false) => rows.iter().filter_map(|r| r.first().copied()).collect(),
            (1, true) => rows.iter().filter_map(|r| r.last().copied()).collect(),
            (_, false) => rows.first()?.clone(),
            (_, true) => rows.last()?.clone(),
        };
        #[allow(clippy::cast_precision_loss)]
        let share = 1.0 / side.len() as f64;
        let at = side.iter().fold(Point3::new(0.0, 0.0, 0.0), |sum, &p| {
            sum + Vec3::new(p.x(), p.y(), p.z()) * share
        });
        let fr = &self.fractions[round];
        let q = self.surface.degree_v();
        let (mut least, mut most, mut slack) = (f64::INFINITY, 0.0_f64, 0.0_f64);
        for kr in 0..fr.len().saturating_sub(1) {
            if fr[kr] >= fr[kr + 1] {
                continue;
            }
            let (iu, iv) = if round == 0 {
                (degree_r + kr, degree_y + k)
            } else {
                (degree_y + k, degree_r + kr)
            };
            let net = self.net(iu, iv)?;
            // Point `i` round and `l` out from the pole, as `H - P W`.
            let point = |i: usize, l: usize| {
                let l = if at_end { degree_y - l } else { l };
                let c = if round == 0 {
                    net[i * (q + 1) + l]
                } else {
                    net[l * (q + 1) + i]
                };
                (
                    [
                        at.x().madd(-c[3], c[0]),
                        at.y().madd(-c[3], c[1]),
                        at.z().madd(-c[3], c[2]),
                    ],
                    c[3],
                )
            };
            let mut patch: Vec<Vec<[f64; 3]>> = Vec::with_capacity(degree_y);
            for l in 0..=degree_y {
                let mut row = Vec::with_capacity(degree_r + 1);
                for i in 0..=degree_r {
                    let (n, w) = point(i, l);
                    if w.is_nan() || w <= 0.0 {
                        return None;
                    }
                    most = most.max(w);
                    if l == 0 {
                        slack = slack.max(n[0].hypot(n[1]).hypot(n[2]));
                    } else {
                        #[allow(clippy::cast_precision_loss)]
                        let share = degree_y as f64 / l as f64;
                        row.push(n.map(|x| x * share));
                    }
                }
                if l > 0 {
                    patch.push(row);
                }
            }
            least = least.min(clear_of_origin(&patch, 6));
        }
        (least > 0.0 && least.is_finite()).then_some(Pole {
            at,
            y: if at_end { 1.0 } else { 0.0 },
            depth,
            gain: depth * most / least,
            slack: depth * slack / least,
        })
    }
}

/// The gap across a closed parameter's seam, between its first and last
/// rows of control points. `None` where their weights differ: the two rows
/// then give the surface in two forms that do not meet in `H`.
fn seam(a: &[Point3], b: &[Point3], wa: &[f64], wb: &[f64]) -> Option<f64> {
    let same = wa.len() == wb.len()
        && wa
            .iter()
            .zip(wb)
            .all(|(x, y)| (x - y).abs() <= 1e-12 * x.abs().max(y.abs()));
    same.then(|| {
        a.iter()
            .zip(b)
            .map(|(p, q)| (*p - *q).length())
            .fold(0.0, f64::max)
    })
}

/// The Bezier net of the knot cell whose first knots have indices
/// `(iu, iv)`, with each weight times `unit`. `None` where a weight on the
/// way is not positive.
fn cell_net(s: &NurbsSurface, iu: usize, iv: usize, unit: f64) -> Option<Vec<[f64; 4]>> {
    let (p, q) = (s.degree_u(), s.degree_v());
    let (ku, kv) = (s.knots_u(), s.knots_v());
    let (rows, weights) = (s.control_points(), s.weights());
    let (u0, u1) = (*ku.get(iu)?, *ku.get(iu + 1)?);
    let (v0, v1) = (*kv.get(iv)?, *kv.get(iv + 1)?);
    // Along u, the Bezier points of each row the cell reads; then along v,
    // those of each column of them.
    let mut along_u: Vec<Vec<[f64; 4]>> = Vec::with_capacity(q + 1);
    for j in iv.checked_sub(q)?..=iv {
        let (mut points, mut w) = (Vec::with_capacity(p + 1), Vec::with_capacity(p + 1));
        for i in iu.checked_sub(p)?..=iu {
            points.push(*rows.get(i)?.get(j)?);
            w.push(*weights.get(i)?.get(j)? * unit);
        }
        let knots = ku.get(iu - p..=iu + p + 1)?.to_vec();
        let row = NurbsCurve::new(p, knots, points, w).ok()?;
        along_u.push(curve_bezier_between(&row, u0, u1).ok()?);
    }
    let mut net = vec![[0.0; 4]; (p + 1) * (q + 1)];
    for k in 0..=p {
        let (mut points, mut w) = (Vec::with_capacity(q + 1), Vec::with_capacity(q + 1));
        for row in &along_u {
            let [x, y, z, h] = *row.get(k)?;
            if h <= 0.0 {
                return None;
            }
            points.push(Point3::new(x / h, y / h, z / h));
            w.push(h);
        }
        let knots = kv.get(iv - q..=iv + q + 1)?.to_vec();
        let column = NurbsCurve::new(q, knots, points, w).ok()?;
        for (l, c) in curve_bezier_between(&column, v0, v1)
            .ok()?
            .into_iter()
            .enumerate()
        {
            *net.get_mut(k * (q + 1) + l)? = c;
        }
    }
    Some(net)
}

/// The Bezier points of the piece from `s` to `t` of the Bezier curve with
/// points `c`: point `k` is the curve's blossom at `s` taken `n - k` times
/// and at `t` taken `k` times. A `t` below `s` gives the piece in reverse.
pub(super) fn piece<const N: usize>(c: &[[f64; N]], s: f64, t: f64) -> Vec<[f64; N]> {
    let n = c.len().saturating_sub(1);
    let mut work = Vec::with_capacity(c.len());
    (0..c.len())
        .map(|k| {
            work.clear();
            work.extend_from_slice(c);
            for level in 1..=n {
                let x = if level <= k { t } else { s };
                for j in 0..=n - level {
                    let next = work[j + 1];
                    for (a, b) in work[j].iter_mut().zip(next) {
                        *a = (b - *a).madd(x, *a);
                    }
                }
            }
            work[0]
        })
        .collect()
}

/// The net of the part of a cell between the shares `u` and `v` of the
/// way across it.
pub(super) fn sub_net(
    net: &[[f64; 4]],
    (p, q): (usize, usize),
    u: (f64, f64),
    v: (f64, f64),
) -> Vec<[f64; 4]> {
    let mut out = net.to_vec();
    if u != (0.0, 1.0) {
        for j in 0..=q {
            let column: Vec<[f64; 4]> = (0..=p).map(|i| out[i * (q + 1) + j]).collect();
            for (i, c) in piece(&column, u.0, u.1).into_iter().enumerate() {
                out[i * (q + 1) + j] = c;
            }
        }
    }
    if v != (0.0, 1.0) {
        for row in out.chunks_mut(q + 1) {
            let cut = piece(row, v.0, v.1);
            row.copy_from_slice(&cut);
        }
    }
    out
}

/// A Bezier curve's point at `t` and its derivative, by de Casteljau.
fn curve_at(c: &[[f64; 4]], t: f64) -> ([f64; 4], [f64; 4]) {
    let n = c.len().saturating_sub(1);
    let mut d = c.to_vec();
    if n == 0 {
        return (d.first().copied().unwrap_or([0.0; 4]), [0.0; 4]);
    }
    for level in 1..n {
        for j in 0..=n - level {
            let next = d[j + 1];
            for (a, b) in d[j].iter_mut().zip(next) {
                *a = (b - *a).madd(t, *a);
            }
        }
    }
    #[allow(clippy::cast_precision_loss)]
    let degree = n as f64;
    let mut value = d[0];
    let mut slope = [0.0; 4];
    for k in 0..4 {
        value[k] = (d[1][k] - d[0][k]).madd(t, d[0][k]);
        slope[k] = degree * (d[1][k] - d[0][k]);
    }
    (value, slope)
}

/// A net's point at the shares `(a, b)` of its cell, with its partials by
/// them.
fn net_at(net: &[[f64; 4]], (p, q): (usize, usize), (a, b): (f64, f64)) -> [[f64; 4]; 3] {
    let (mut values, mut slopes) = (Vec::with_capacity(p + 1), Vec::with_capacity(p + 1));
    for row in net.chunks(q + 1) {
        let (value, slope) = curve_at(row, b);
        values.push(value);
        slopes.push(slope);
    }
    let (h, h_a) = curve_at(&values, a);
    let (h_b, _) = curve_at(&slopes, a);
    [h, h_a, h_b]
}

/// The partial `c` of `H` less `s` times that of `W`.
fn moved(c: [f64; 4], s: Point3) -> Vec3 {
    Vec3::new(
        s.x().madd(-c[3], c[0]),
        s.y().madd(-c[3], c[1]),
        s.z().madd(-c[3], c[2]),
    )
}

/// The least singular value of the matrix with columns `x` and `y`.
fn least_singular(x: Vec3, y: Vec3) -> f64 {
    let (a, b, c) = (x.dot(x), y.dot(y), x.dot(y));
    let most = 0.5f64.madd(a + b, (0.25 * (a - b) * (a - b)).madd(1.0, c * c).sqrt());
    if most > 0.0 {
        (x.cross(y).length_squared() / most).sqrt()
    } else {
        0.0
    }
}

/// A lower bound on `|G d|` over the unit steps `d` and the matrices `G` in
/// the hull of the pairs of a point of one column's net and a point of the
/// other's, part by part: `(1 + μ) σ` from the reference `m`, as the module's
/// head describes. Nothing for a reference whose columns do not span a
/// plane.
fn tangent_bound(m: [Vec3; 2], columns: &[[Vec<Vec3>; 2]]) -> f64 {
    let Ok(e1) = m[0].normalize() else {
        return 0.0;
    };
    let Ok(e2) = (m[1] - e1 * e1.dot(m[1])).normalize() else {
        return 0.0;
    };
    // `Uᵀ M` is upper triangular: `[[a, b], [0, c]]`, and `K` its inverse.
    let (a, b, c) = (e1.dot(m[0]), e1.dot(m[1]), e2.dot(m[1]));
    if !(a > 0.0 && c > 0.0) {
        return 0.0;
    }
    let k = |x: f64, y: f64| [x / a - b * y / (a * c), y / c];
    let mut mu = f64::INFINITY;
    for [xs, ys] in columns {
        let kx: Vec<[f64; 2]> = xs.iter().map(|v| k(e1.dot(*v) - a, e2.dot(*v))).collect();
        let ky: Vec<[f64; 2]> = ys
            .iter()
            .map(|v| k(e1.dot(*v) - b, e2.dot(*v) - c))
            .collect();
        for x in &kx {
            for y in &ky {
                let mid = 0.5 * (x[0] + y[1]);
                let half = 0.5 * (x[0] - y[1]);
                let off = 0.5 * (y[0] + x[1]);
                mu = mu.min(mid - half.hypot(off));
            }
        }
    }
    (1.0 + mu) * least_singular(m[0], m[1])
}

/// The distance from `q` to the segment between `ends`.
fn to_segment(q: Point2, (a, b): (Point2, Point2)) -> f64 {
    let d = b - a;
    let len2 = d.dot(d);
    let f = if len2 > 0.0 {
        ((q - a).dot(d) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (q - (a + d * f)).length()
}

/// A lower bound on `|U|` over a polynomial patch with the net `rows`, each
/// row a Bezier curve: the least reach of the net's points along the unit
/// vector of their sum. Where that is not positive, the patch is halved
/// along the rows, up to `depth` times.
fn clear_of_origin(rows: &[Vec<[f64; 3]>], depth: u32) -> f64 {
    let sum = rows
        .iter()
        .flatten()
        .fold([0.0; 3], |s, c| [s[0] + c[0], s[1] + c[1], s[2] + c[2]]);
    let Ok(n) = Vec3::new(sum[0], sum[1], sum[2]).normalize() else {
        return 0.0;
    };
    let least = rows
        .iter()
        .flatten()
        .map(|c| n.dot(Vec3::new(c[0], c[1], c[2])))
        .fold(f64::INFINITY, f64::min);
    if least > 0.0 || depth == 0 {
        return least.max(0.0);
    }
    [(0.0, 0.5), (0.5, 1.0)]
        .iter()
        .map(|&(a, b)| {
            let half: Vec<Vec<[f64; 3]>> = rows.iter().map(|r| piece(r, a, b)).collect();
            clear_of_origin(&half, depth - 1)
        })
        .fold(f64::INFINITY, f64::min)
}

/// A lower bound on how far a rational Bezier curve with homogeneous points
/// `c`, taken from the point it is read from, keeps from that point: the
/// least reach of its points along the unit vector of their sum. Where that
/// is not positive, the curve is halved, up to `depth` times.
fn hull_clear(c: &[[f64; 4]], depth: u32) -> f64 {
    let points: Vec<Vec3> = c
        .iter()
        .map(|h| Vec3::new(h[0] / h[3], h[1] / h[3], h[2] / h[3]))
        .collect();
    let sum = points.iter().fold(Vec3::new(0.0, 0.0, 0.0), |s, &p| s + p);
    let Ok(n) = sum.normalize() else {
        return 0.0;
    };
    let least = points
        .iter()
        .map(|p| n.dot(*p))
        .fold(f64::INFINITY, f64::min);
    if least > 0.0 || depth == 0 {
        return least.max(0.0);
    }
    hull_clear(&piece(c, 0.0, 0.5), depth - 1).min(hull_clear(&piece(c, 0.5, 1.0), depth - 1))
}

/// A lower bound on how near the edges come to `q`: exact for a line, over
/// the whole curve for a circle, over the ring between its half-axes for an
/// ellipse, and over halved hulls of its control points for a B-spline.
/// `None` for a B-spline edge with a weight that is not positive.
pub(super) fn clearance(edges: &[&BoundaryEdge<'_>], q: Point3) -> Option<f64> {
    let mut least = f64::INFINITY;
    for e in edges {
        let near = match e.curve {
            EdgeCurve::Line => {
                let a = e.curve.evaluate_with_endpoints(e.from, e.start, e.end);
                let b = e.curve.evaluate_with_endpoints(e.to, e.start, e.end);
                let d = b - a;
                let len2 = d.dot(d);
                let f = if len2 > 0.0 {
                    ((q - a).dot(d) / len2).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                (q - (a + d * f)).length()
            }
            EdgeCurve::Circle(c) => {
                let d = q - c.center();
                let h = d.dot(c.normal());
                let flat = (d - c.normal() * h).length();
                (flat - c.radius()).hypot(h)
            }
            EdgeCurve::Ellipse(c) => {
                let d = q - c.center();
                let h = d.dot(c.normal());
                let flat = (d - c.normal() * h).length();
                let gap = (flat - c.semi_major()).max(c.semi_minor() - flat).max(0.0);
                gap.hypot(h)
            }
            EdgeCurve::NurbsCurve(n) => {
                let mut near = f64::INFINITY;
                for (ta, tb) in edge_pieces(e) {
                    let hull = curve_bezier_between(n, ta.min(tb), ta.max(tb)).ok()?;
                    if hull.iter().any(|c| c[3] <= 0.0) {
                        return None;
                    }
                    let from_q: Vec<[f64; 4]> = hull
                        .iter()
                        .map(|c| {
                            [
                                q.x().madd(-c[3], c[0]),
                                q.y().madd(-c[3], c[1]),
                                q.z().madd(-c[3], c[2]),
                                c[3],
                            ]
                        })
                        .collect();
                    near = near.min(hull_clear(&from_q, 8));
                }
                near
            }
        };
        least = least.min(near);
    }
    Some(least)
}

/// How far a conic piece from `ta` to `tb` strays from its chord: the
/// sagitta of a circle's arc, and an eighth of the most second derivative
/// times the square of the run for an ellipse. Nothing for a line, and for
/// a B-spline, whose hull gives it.
pub(super) fn conic_off(curve: &EdgeCurve, ta: f64, tb: f64) -> f64 {
    let run = (tb - ta).abs();
    match curve {
        EdgeCurve::Circle(c) if run <= std::f64::consts::PI => {
            let s = (0.25 * run).sin();
            2.0 * c.radius() * s * s
        }
        EdgeCurve::Circle(c) => 2.0 * c.radius(),
        EdgeCurve::Ellipse(c) => (0.125 * run * run).min(2.0) * c.semi_major(),
        EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => 0.0,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::face_loops::FaceLoops;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A flat patch 15 by 20, of degree one, with the knots `knots_u`.
    fn patch(knots_u: Vec<f64>) -> NurbsSurface {
        let control = (0..4)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(5.0 * f64::from(i), 20.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        NurbsSurface::new(
            1,
            1,
            knots_u,
            vec![0.0, 0.0, 1.0, 1.0],
            control,
            vec![vec![1.0; 2]; 4],
        )
        .unwrap()
    }

    /// A knot that is not a number, or one that falls back, leaves the
    /// surface without a bound: a chord on it reads none, and a face on it
    /// reads no wire.
    #[test]
    fn a_surface_with_a_broken_knot_bounds_no_chord() {
        for knots in [
            vec![0.0, 0.0, f64::NAN, 0.5, 1.0, 1.0],
            vec![0.0, 0.0, 0.6, 0.4, 1.0, 1.0],
        ] {
            let surface = patch(knots);
            let bend = Bend::new(&surface, (false, false), Frame::Flat);
            let chord = Chord {
                from: (0.2, 0.3),
                by: (0.4, 0.2),
                ends: (Point2::new(0.2, 0.3), Point2::new(0.6, 0.5)),
                ends_3d: (Point3::new(3.0, 6.0, 0.0), Point3::new(9.0, 10.0, 0.0)),
                off: 0.0,
                miss: 0.0,
            };
            assert!(bend.sag(&chord, 1e-4).is_infinite());
            let mut topo = Topology::default();
            let corners = [[2.0, 2.0], [12.0, 2.0], [12.0, 18.0], [2.0, 18.0]]
                .map(|c| topo.add_vertex(Vertex::new(Point3::new(c[0], c[1], 0.0), 1e-7)));
            let edges = (0..4)
                .map(|k| {
                    let e =
                        topo.add_edge(Edge::new(corners[k], corners[(k + 1) % 4], EdgeCurve::Line));
                    OrientedEdge::new(e, true)
                })
                .collect();
            let outer = topo.add_wire(Wire::new(edges, true).unwrap());
            let face = topo.add_face(Face::new(outer, vec![], FaceSurface::Nurbs(surface)));
            let loops = FaceLoops::of(&topo, face, 1e-7).unwrap();
            assert!(loops.is_none_or(|l| l.loops.iter().all(Option::is_none)));
        }
    }

    /// A flat patch 10 by 10 of degree one reads the same bound with all
    /// its weights 1e100 as with all of them 1: weights of one size give the
    /// same surface. Without the power of two that brings the weights down,
    /// the sums overflow, and the chord reads no bound rather than nothing.
    #[test]
    fn a_patch_whose_weights_are_all_huge_reads_the_bound_of_weights_of_one() {
        let control = (0..2)
            .map(|i| {
                (0..2)
                    .map(|j| Point3::new(10.0 * f64::from(i), 10.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect::<Vec<Vec<Point3>>>();
        let sags: Vec<f64> = [(1.0, None), (1e100, None), (1e100, Some(1.0))]
            .iter()
            .map(|&(w, unit)| {
                let surface = NurbsSurface::new(
                    1,
                    1,
                    vec![0.0, 0.0, 1.0, 1.0],
                    vec![0.0, 0.0, 1.0, 1.0],
                    control.clone(),
                    vec![vec![w; 2]; 2],
                )
                .unwrap();
                let mut bend = Bend::new(&surface, (false, false), Frame::Flat);
                if let Some(unit) = unit {
                    bend.unit = unit;
                }
                let chord = Chord {
                    from: (0.2, 0.3),
                    by: (0.4, 0.2),
                    ends: (Point2::new(0.2, 0.3), Point2::new(0.6, 0.5)),
                    ends_3d: (Point3::new(2.0, 3.0, 0.0), Point3::new(6.0, 5.0, 0.0)),
                    off: 0.01,
                    miss: 0.0,
                };
                bend.sag(&chord, 1e-4)
            })
            .collect();
        // The edge may lie 0.01 off the run, which the patch stretches by
        // 10 in each fraction.
        assert!((sags[0] - 1e-3).abs() < 1e-12, "{sags:?}");
        assert!((sags[1] - sags[0]).abs() < 1e-12, "{sags:?}");
        assert!(sags[2].is_infinite(), "{sags:?}");
    }

    /// Whether a box holds a knot repeated more often than the degree, read
    /// against every inner knot.
    fn tears_by_scan(bend: &Bend<'_>, lo: [f64; 2], hi: [f64; 2]) -> bool {
        (0..2).any(|d| {
            let f = &bend.fractions[d];
            let (degree, ..) = bend.along(d);
            let inner = &f[1..f.len() - 1];
            let shifts: Vec<f64> = if bend.closes(d) {
                (-2..=3).map(f64::from).collect()
            } else {
                vec![0.0]
            };
            shifts.iter().any(|&m| {
                inner.iter().any(|&x| {
                    let times = inner.iter().filter(|&&y| y.total_cmp(&x).is_eq()).count();
                    times > degree && lo[d] < x + m && x + m < hi[d]
                })
            })
        })
    }

    /// On a surface of 120 knot spans, closed in `u`, with knots taken twice
    /// and three times, the knots read inside each of many boxes are those
    /// that a scan of every knot finds.
    #[test]
    fn a_box_finds_its_torn_knots_among_many_spans() {
        let mut knots_u = vec![0.0; 3];
        for k in 1..120 {
            let x = f64::from(k) / 120.0;
            let times = if k % 13 == 0 {
                3
            } else if k % 7 == 0 {
                2
            } else {
                1
            };
            knots_u.extend(std::iter::repeat_n(x, times));
        }
        knots_u.extend([1.0; 3]);
        let rows = knots_u.len() - 3;
        let control = (0..rows)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let x = i as f64;
                (0..2)
                    .map(|j| Point3::new(x, 10.0 * f64::from(j), 0.0))
                    .collect()
            })
            .collect();
        let surface = NurbsSurface::new(
            2,
            1,
            knots_u,
            vec![0.0, 0.0, 1.0, 1.0],
            control,
            vec![vec![1.0; 2]; rows],
        )
        .unwrap();
        let mut found = 0;
        for closed in [false, true] {
            let bend = Bend::new(&surface, (closed, false), Frame::Flat);
            let mut seed = 0x9e37_79b9_u32;
            let mut next = || {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                f64::from(seed) / f64::from(u32::MAX)
            };
            for _ in 0..2000 {
                let (a, b) = if closed {
                    (2.0f64.madd(next(), -0.5), 0.1 * next())
                } else {
                    (next(), 0.1 * next())
                };
                let (lo, hi) = ([a, 0.2], [a + b, 0.4]);
                let tears = bend.tears(lo, hi);
                assert_eq!(tears, tears_by_scan(&bend, lo, hi), "{lo:?} {hi:?}");
                found += usize::from(tears);
            }
        }
        assert!(found > 100, "{found}");
    }

    /// The count of a knot's copies moves on past a knot that is not a
    /// number.
    #[test]
    fn a_knot_that_is_not_a_number_does_not_stall_the_count_of_its_copies() {
        let surface = patch(vec![0.0, 0.0, 0.3, 0.6, 1.0, 1.0]);
        let mut bend = Bend::new(&surface, (false, false), Frame::Flat);
        bend.fractions[0] = vec![0.0, f64::NAN, f64::NAN, 0.5, 1.0];
        assert!(!bend.tears([0.0, 0.0], [1.0, 1.0]));
    }
}

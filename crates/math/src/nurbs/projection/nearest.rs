//! The proof that a point of a NURBS surface is the nearest to a point.
//!
//! On one knot-span patch in Bezier form, with homogeneous control points
//! `q = w (P - p)` taken from the point `p`, the squared distance is
//! `|q|² / w²`. Both `|q|²` and `w²` are polynomials of twice the degree, and
//! their Bernstein coefficients come from products of the control points.
//! With every weight positive, each coefficient of `w²` is positive, so the
//! squared distance on the patch is no less than the least ratio of the two
//! coefficients. That bound closes on the true least value as the patch is
//! halved, as the square of the part's size.
//!
//! Near a nearest point that bound closes too slowly, so each part also gets
//! a bound from its curvature. With `d` the nearest distance found so far and
//! `c = (d - slack)²`, `g = |q|² - c w²` is a polynomial too, and the
//! coefficients of its second partials bound the Hessian of `g` on the part.
//! From the lowest point of `g` on the part, its value, its gradient and that
//! bound give a quadratic that `g` never goes below on the part. Where the
//! quadratic stays at or above zero, no point of the part is nearer than
//! `d - slack`.

use std::collections::BinaryHeap;

use crate::aabb::Aabb3;
use crate::nurbs::span_index::SpanIndex;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

use super::{SurfaceProjection, surface_descend};

/// The most halvings of parts that one proof makes. A point with a whole
/// curve of nearest points, or with two nearest points that the slack cannot
/// tell apart, can need more; past this many, the nearest point found
/// returns.
const MOST_HALVINGS: usize = 512;

/// The share of the largest coordinate of the point and the surface below
/// which the slack of the proof does not go: rounding moves the sums of the
/// proof by about this much, so a smaller slack proves nothing.
const ROUNDING: f64 = 1e-12;

/// A Newton step on a part shorter than this share of it settles the
/// lowest point: what is left moves the bound by the square of it.
const SETTLED: f64 = 1e-10;

/// `C(p, i) C(p, j) / C(2p, i + j)` at `i * (p + 1) + j`: the share of the
/// product of Bernstein terms `i` and `j` of degree `p` in term `i + j` of
/// degree `2p`.
fn product_shares(p: usize) -> Vec<f64> {
    let choose = |n: usize, k: usize| -> f64 {
        (0..k).fold(1.0, |acc, m| acc * (n - m) as f64 / (m + 1) as f64)
    };
    let mut shares = Vec::with_capacity((p + 1) * (p + 1));
    for i in 0..=p {
        for j in 0..=p {
            shares.push(choose(p, i) * choose(p, j) / choose(2 * p, i + j));
        }
    }
    shares
}

/// A part of a patch: its control points taken from the point, its
/// parameters, and the coefficients of `|q|²` and `w²` on it.
struct Cell {
    points: Vec<[f64; 4]>,
    u: (f64, f64),
    v: (f64, f64),
    squares: Vec<f64>,
    weights: Vec<f64>,
    /// No point of the part comes nearer than the square root of this: the
    /// least ratio of the coefficients, or the squared distance to the box
    /// of the control points where that is more.
    floor: f64,
}

impl Cell {
    /// The coefficients of `g = |q|² - c w²` on the part.
    fn gaps(&self, c: f64) -> Vec<f64> {
        self.squares
            .iter()
            .zip(&self.weights)
            .map(|(s, w)| s - c * w)
            .collect()
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == std::cmp::Ordering::Equal
    }
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    /// The part with the lowest floor is the greatest, for a max-heap.
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.floor.total_cmp(&self.floor)
    }
}

/// What one proof reads throughout.
struct Search {
    tolerance: f64,
    rows: usize,
    cols: usize,
    shares_u: Vec<f64>,
    shares_v: Vec<f64>,
    domain: [f64; 4],
    closed: (bool, bool),
    scratch: std::cell::RefCell<Scratch>,
}

/// Room the evaluation of `g` works in, kept between evaluations.
#[derive(Default)]
struct Scratch {
    work: Vec<[f64; 4]>,
    rows: Vec<[[f64; 4]; 3]>,
    column: Vec<[f64; 4]>,
}

/// The nearest point of `surface` to `point`, from `start`: no point of the
/// surface is nearer than the result by more than `tolerance`, or than a
/// share of the size of the model that rounding can move a distance by.
/// Each part of a patch whose bounds do not prove that is halved, up to
/// [`MOST_HALVINGS`] times.
pub(super) fn nearest(
    surface: &NurbsSurface,
    index: &SpanIndex,
    point: Point3,
    tolerance: f64,
    start: SurfaceProjection,
) -> SurfaceProjection {
    let hull = surface.aabb();
    let size = [hull.min, hull.max, point]
        .iter()
        .flat_map(|c| [c.x(), c.y(), c.z()])
        .fold(0.0, |m: f64, c| m.max(c.abs()));
    let slack = tolerance.max(ROUNDING * size);
    let mut best = start;
    if best.distance <= slack {
        return best;
    }
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let (rows, cols) = (surface.degree_u() + 1, surface.degree_v() + 1);
    let search = Search {
        tolerance,
        rows,
        cols,
        shares_u: product_shares(rows - 1),
        shares_v: product_shares(cols - 1),
        domain: [u_min, u_max, v_min, v_max],
        closed: (surface.is_periodic_u(), surface.is_periodic_v()),
        scratch: std::cell::RefCell::default(),
    };
    let below = |best: &SurfaceProjection| (best.distance - slack).powi(2);

    // Each span lies in the box of the control points that shape it, so a
    // span whose box is no nearer than the bound holds no nearer point.
    let mut heap = BinaryHeap::new();
    let bound = below(&best);
    for (i, _) in index.spans_within(point, bound.sqrt()) {
        let Some((net, u, v)) = index.bezier_net(surface, i) else {
            continue;
        };
        if !net.iter().flatten().all(|c| c.is_finite()) {
            return best;
        }
        // A common factor of the weights leaves the surface as it is; taken
        // out, it cannot carry the products out of the range of a float.
        let scale = net.iter().fold(0.0_f64, |m, q| m.max(q[3].abs()));
        if scale <= 0.0 {
            return best;
        }
        let points = net
            .iter()
            .map(|q| {
                let w = q[3] / scale;
                [
                    q[0] / scale - w * point.x(),
                    q[1] / scale - w * point.y(),
                    q[2] / scale - w * point.z(),
                    w,
                ]
            })
            .collect();
        heap.extend(search.cell(points, u, v, bound));
    }

    let mut halvings = 0;
    while let Some(cell) = heap.pop() {
        let bound = below(&best);
        if cell.floor >= bound {
            break;
        }
        let gaps = cell.gaps(bound);
        let (at, gap, slope) = search.lowest(&cell, bound, &best, &gaps);
        if gap < 0.0 {
            // Along a direction of degree zero the surface steps at each
            // knot, so the seed keeps to the middle of the part's span.
            let seed = (
                if search.rows == 1 {
                    f64::midpoint(cell.u.0, cell.u.1)
                } else {
                    param_at(at[0], cell.u)
                },
                if search.cols == 1 {
                    f64::midpoint(cell.v.0, cell.v.1)
                } else {
                    param_at(at[1], cell.v)
                },
            );
            let found = surface_descend(
                surface,
                point,
                seed,
                search.domain,
                search.closed,
                search.tolerance,
            );
            if found.distance < best.distance {
                best = found;
                if best.distance <= slack {
                    return best;
                }
                heap.push(cell);
                continue;
            }
        }
        let partials = search.second_partials(&gaps);
        let [ss, st, tt] = &partials;
        let bends = (ss.least(), tt.least(), st.largest_size());
        if model_floor(at, gap, slope, bends) >= 0.0 || halvings >= MOST_HALVINGS {
            continue;
        }
        halvings += 1;
        heap.extend(search.halves(&cell, search.halve_along_u(&gaps, &partials), below(&best)));
    }
    best
}

/// The parameter `share` of the way across `range`, its end exactly at 1.
fn param_at(share: f64, (lo, hi): (f64, f64)) -> f64 {
    if share >= 1.0 {
        hi
    } else {
        share * (hi - lo) + lo
    }
}

impl Search {
    /// The part on `points`, or `None` when no point of it can come nearer
    /// than the square root of `bound`.
    fn cell(
        &self,
        points: Vec<[f64; 4]>,
        u: (f64, f64),
        v: (f64, f64),
        bound: f64,
    ) -> Option<Cell> {
        // The part lies in the box of its points, which the point is at the
        // origin of: the box's squared distance from the origin bounds too.
        let hull = Aabb3::from_points(
            points
                .iter()
                .map(|q| Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3])),
        )
        .distance_squared_to_point(Point3::new(0.0, 0.0, 0.0));
        if hull >= bound {
            return None;
        }
        let (rows, cols) = (self.rows, self.cols);
        let wide = 2 * cols - 1;
        let len = (2 * rows - 1) * wide;
        let polynomial = points.iter().all(|q| q[3].to_bits() == 1.0_f64.to_bits());
        let (mut squares, mut weights) = (vec![0.0; len], vec![1.0; len]);
        if !polynomial {
            weights.fill(0.0);
        }
        // Each pair of points adds to one coefficient, the same either way
        // round, so each pair is read once.
        for (n1, a) in points.iter().enumerate() {
            let (i1, j1) = (n1 / cols, n1 % cols);
            for (n2, b) in points.iter().enumerate().skip(n1) {
                let (i2, j2) = (n2 / cols, n2 % cols);
                let twice = if n2 == n1 { 1.0 } else { 2.0 };
                let share = twice * self.shares_u[i1 * rows + i2] * self.shares_v[j1 * cols + j2];
                let k = (i1 + i2) * wide + j1 + j2;
                squares[k] += share * (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
                if !polynomial {
                    weights[k] += share * (a[3] * b[3]);
                }
            }
        }
        // A ratio that is not a number bounds nothing, and leaves the floor
        // at zero.
        let floor = squares
            .iter()
            .zip(&weights)
            .map(|(s, w)| s / w)
            .fold(f64::INFINITY, |m, r| {
                if r.is_nan() { 0.0_f64.min(m) } else { m.min(r) }
            })
            .max(hull);
        (floor < bound).then_some(Cell {
            points,
            u,
            v,
            squares,
            weights,
            floor,
        })
    }

    /// The Bernstein coefficients of the second partials of `g` on the part,
    /// in its own parameters `s` and `t`: `g_ss`, `g_st` and `g_tt`.
    fn second_partials(&self, gaps: &[f64]) -> [Grid; 3] {
        let (du, dv) = (2 * (self.rows - 1), 2 * (self.cols - 1));
        let at = |k: usize, l: usize| gaps[k * (dv + 1) + l];
        let grid = |rows: usize, cols: usize, scale: usize, value: &dyn Fn(usize, usize) -> f64| {
            let mut values = Vec::with_capacity(rows * cols);
            for k in 0..rows {
                for l in 0..cols {
                    values.push(scale as f64 * value(k, l));
                }
            }
            Grid { cols, values }
        };
        let (rows_ss, rows_st) = (du.saturating_sub(1), du);
        let (cols_tt, cols_st) = (dv.saturating_sub(1), dv);
        [
            grid(rows_ss, dv + 1, du * du.saturating_sub(1), &|k, l| {
                at(k + 2, l) - 2.0 * at(k + 1, l) + at(k, l)
            }),
            grid(rows_st, cols_st, du * dv, &|k, l| {
                at(k + 1, l + 1) - at(k + 1, l) - at(k, l + 1) + at(k, l)
            }),
            grid(du + 1, cols_tt, dv * dv.saturating_sub(1), &|k, l| {
                at(k, l + 2) - 2.0 * at(k, l + 1) + at(k, l)
            }),
        ]
    }

    /// Whether to halve the part across u: the way in which its second
    /// partials spread more, since that spread is what keeps their bounds
    /// from the values at any one point. Where they do not spread, the way
    /// in which the coefficients of `g` bend more.
    fn halve_along_u(&self, gaps: &[f64], partials: &[Grid; 3]) -> bool {
        let (du, dv) = (2 * (self.rows - 1), 2 * (self.cols - 1));
        if du == 0 || dv == 0 {
            return dv == 0;
        }
        let spread_s = partials.iter().map(Grid::spread_down).fold(0.0, f64::max);
        let spread_t = partials.iter().map(Grid::spread_across).fold(0.0, f64::max);
        if spread_s > spread_t {
            return true;
        }
        if spread_t > spread_s {
            return false;
        }
        let at = |k: usize, l: usize| gaps[k * (dv + 1) + l];
        let most = |values: &mut dyn Iterator<Item = f64>| values.fold(0.0, f64::max);
        let along_s = most(
            &mut (0..du - 1)
                .flat_map(|k| (0..=dv).map(move |l| (k, l)))
                .map(|(k, l)| (at(k + 2, l) - 2.0 * at(k + 1, l) + at(k, l)).abs()),
        );
        let along_t = most(
            &mut (0..=du)
                .flat_map(|k| (0..dv - 1).map(move |l| (k, l)))
                .map(|(k, l)| (at(k, l + 2) - 2.0 * at(k, l + 1) + at(k, l)).abs()),
        );
        du as f64 * along_s >= dv as f64 * along_t
    }

    /// `g = |q|² - c w²` at `(s, t)` of the part, with its gradient and its
    /// Hessian `[g_ss, g_st, g_tt]`.
    fn gap_at(&self, cell: &Cell, c: f64, [s, t]: [f64; 2]) -> (f64, [f64; 2], [f64; 3]) {
        let mut scratch = self.scratch.borrow_mut();
        let Scratch { work, rows, column } = &mut *scratch;
        rows.clear();
        for row in cell.points.chunks(self.cols) {
            rows.push(with_derivatives(row, t, work));
        }
        let mut across = |k: usize| {
            column.clear();
            column.extend(rows.iter().map(|r| r[k]));
            with_derivatives(column, s, work)
        };
        let [q, q_s, q_ss] = across(0);
        let [q_t, q_st, _] = across(1);
        let [q_tt, _, _] = across(2);
        let dot =
            |a: [f64; 4], b: [f64; 4]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - c * a[3] * b[3];
        (
            dot(q, q),
            [2.0 * dot(q, q_s), 2.0 * dot(q, q_t)],
            [
                2.0 * (dot(q_s, q_s) + dot(q, q_ss)),
                2.0 * (dot(q_s, q_t) + dot(q, q_st)),
                2.0 * (dot(q_t, q_t) + dot(q, q_tt)),
            ],
        )
    }

    /// The lowest point of `g` on the part that Newton steps, held inside
    /// the part, find from the best of: the nearest point so far held to
    /// the part, its middle, and its lowest corner, which `gaps` gives. Its
    /// `(s, t)`, `g` there, and the gradient there.
    fn lowest(
        &self,
        cell: &Cell,
        c: f64,
        best: &SurfaceProjection,
        gaps: &[f64],
    ) -> ([f64; 2], f64, [f64; 2]) {
        let share = |x: f64, (lo, hi): (f64, f64)| {
            if hi > lo {
                ((x - lo) / (hi - lo)).clamp(0.0, 1.0)
            } else {
                0.0
            }
        };
        let (du, dv) = (2 * (self.rows - 1), 2 * (self.cols - 1));
        let corner = [
            (gaps[0], [0.0, 0.0]),
            (gaps[dv], [0.0, 1.0]),
            (gaps[du * (dv + 1)], [1.0, 0.0]),
            (gaps[du * (dv + 1) + dv], [1.0, 1.0]),
        ]
        .into_iter()
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map_or([0.0, 0.0], |(_, at)| at);
        let held = [share(best.u, cell.u), share(best.v, cell.v)];
        let inside = |a: f64| a > 0.0 && a < 1.0;
        // A part that holds the nearest point so far has it for its lowest
        // point, or as near as one Newton step finds.
        let starts: &[[f64; 2]] = if inside(held[0]) && inside(held[1]) {
            &[held]
        } else {
            &[held, [0.5, 0.5], corner]
        };
        let mut x = starts[0];
        let (mut g, mut grad, mut hess) = self.gap_at(cell, c, x);
        for start in &starts[1..] {
            let (g_start, grad_start, hess_start) = self.gap_at(cell, c, *start);
            if g_start < g {
                (x, g, grad, hess) = (*start, g_start, grad_start, hess_start);
            }
        }
        for _ in 0..8 {
            let [h_ss, h_st, h_tt] = hess;
            let det = h_ss * h_tt - h_st * h_st;
            if !(h_ss > 0.0 && det > 0.0) {
                break;
            }
            let full = [
                x[0] - (h_tt * grad[0] - h_st * grad[1]) / det,
                x[1] - (h_ss * grad[1] - h_st * grad[0]) / det,
            ];
            let within = |a: f64| (0.0..=1.0).contains(&a);
            let mut next = [full[0].clamp(0.0, 1.0), full[1].clamp(0.0, 1.0)];
            if !within(full[0]) && within(full[1]) {
                next[1] = (x[1] - (grad[1] + h_st * (next[0] - x[0])) / h_tt).clamp(0.0, 1.0);
            } else if within(full[0]) && !within(full[1]) {
                next[0] = (x[0] - (grad[0] + h_st * (next[1] - x[1])) / h_ss).clamp(0.0, 1.0);
            }
            let (g_next, grad_next, hess_next) = self.gap_at(cell, c, next);
            if g_next.is_nan() || g_next >= g {
                break;
            }
            let moved = (next[0] - x[0]).abs().max((next[1] - x[1]).abs());
            (x, g, grad, hess) = (next, g_next, grad_next, hess_next);
            if moved < SETTLED {
                break;
            }
        }
        (x, g, grad)
    }

    /// The two halves of the part, across u or across v, by de Casteljau on
    /// its control points.
    fn halves(&self, cell: &Cell, along_u: bool, bound: f64) -> impl Iterator<Item = Cell> {
        let (rows, cols) = (self.rows, self.cols);
        let (n, m) = if along_u { (rows, cols) } else { (cols, rows) };
        let at = |k: usize, l: usize| if along_u { k * cols + l } else { l * cols + k };
        let mut low = cell.points.clone();
        let mut high = cell.points.clone();
        let mut work = vec![[0.0; 4]; n];
        for l in 0..m {
            for (k, slot) in work.iter_mut().enumerate() {
                *slot = cell.points[at(k, l)];
            }
            low[at(0, l)] = work[0];
            high[at(n - 1, l)] = work[n - 1];
            for r in 1..n {
                for k in 0..n - r {
                    for c in 0..4 {
                        work[k][c] = 0.5 * (work[k][c] + work[k + 1][c]);
                    }
                }
                low[at(r, l)] = work[0];
                high[at(n - 1 - r, l)] = work[n - 1 - r];
            }
        }
        let halves = if along_u {
            let mid = f64::midpoint(cell.u.0, cell.u.1);
            [
                self.cell(low, (cell.u.0, mid), cell.v, bound),
                self.cell(high, (mid, cell.u.1), cell.v, bound),
            ]
        } else {
            let mid = f64::midpoint(cell.v.0, cell.v.1);
            [
                self.cell(low, cell.u, (cell.v.0, mid), bound),
                self.cell(high, cell.u, (mid, cell.v.1), bound),
            ]
        };
        halves.into_iter().flatten()
    }
}

/// A Bezier curve of homogeneous points at `t`, with its first and second
/// derivatives.
fn with_derivatives(points: &[[f64; 4]], t: f64, work: &mut Vec<[f64; 4]>) -> [[f64; 4]; 3] {
    let n = points.len() - 1;
    let zero = [0.0; 4];
    if n == 0 {
        return [points[0], zero, zero];
    }
    work.clear();
    work.extend_from_slice(points);
    for level in 0..n - n.min(2) {
        for k in 0..n - level {
            for c in 0..4 {
                work[k][c] = (1.0 - t) * work[k][c] + t * work[k + 1][c];
            }
        }
    }
    let lerp = |a: [f64; 4], b: [f64; 4]| std::array::from_fn(|c| (1.0 - t) * a[c] + t * b[c]);
    let step = |a: [f64; 4], b: [f64; 4], k: f64| std::array::from_fn(|c| k * (b[c] - a[c]));
    let degree = n as f64;
    if n == 1 {
        return [lerp(work[0], work[1]), step(work[0], work[1], 1.0), zero];
    }
    let (low, high) = (lerp(work[0], work[1]), lerp(work[1], work[2]));
    let second = std::array::from_fn(|c| {
        degree * (degree - 1.0) * (work[2][c] - 2.0 * work[1][c] + work[0][c])
    });
    [lerp(low, high), step(low, high, degree), second]
}

/// The least, on the part, of the quadratic that `g` never goes below:
/// `g(x + h) ≥ g + slope · h + h_s² D_s / 2 + h_t² D_t / 2` while every
/// Hessian on the part less `diag(D_s, D_t)` has no negative eigenvalue.
/// With `g_ss ≥ a`, `g_tt ≥ b` and `|g_st| ≤ e` on the part, that holds for
/// `D_s = a - e r` and `D_t = b - e / r` at each `r > 0`; the best of a few
/// returns.
fn model_floor(x: [f64; 2], gap: f64, slope: [f64; 2], (a, b, e): (f64, f64, f64)) -> f64 {
    let along = |x: f64, slope: f64, bend: f64| {
        let (lo, hi) = (-x, 1.0 - x);
        let at = |h: f64| slope * h + 0.5 * bend * h * h;
        let ends = at(lo).min(at(hi));
        if bend > 0.0 {
            ends.min(at((-slope / bend).clamp(lo, hi)))
        } else {
            ends
        }
    };
    if e <= 0.0 {
        return gap + along(x[0], slope[0], a) + along(x[1], slope[1], b);
    }
    let ratio = (a.abs().max(f64::MIN_POSITIVE) / b.abs().max(f64::MIN_POSITIVE)).sqrt();
    [0.0625, 0.25, 1.0, 4.0, 16.0]
        .iter()
        .map(|k| {
            let r = ratio * k;
            gap + along(x[0], slope[0], a - e * r) + along(x[1], slope[1], b - e / r)
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

/// The coefficients of one second partial of `g` on a part, in rows along
/// `s` of `cols` each.
struct Grid {
    cols: usize,
    values: Vec<f64>,
}

impl Grid {
    /// The least coefficient, and so the least value; 0 for a partial that
    /// is zero for want of degree.
    fn least(&self) -> f64 {
        self.values.iter().copied().reduce(f64::min).unwrap_or(0.0)
    }

    fn largest_size(&self) -> f64 {
        self.values.iter().fold(0.0, |m, v| m.max(v.abs()))
    }

    /// The most the coefficients of one column spread along `s`.
    fn spread_down(&self) -> f64 {
        (0..self.cols)
            .map(|l| {
                let column = self.values.iter().skip(l).step_by(self.cols.max(1));
                let (lo, hi) = column.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                    (lo.min(v), hi.max(v))
                });
                (hi - lo).max(0.0)
            })
            .fold(0.0, f64::max)
    }

    /// The most the coefficients of one row spread along `t`.
    fn spread_across(&self) -> f64 {
        self.values
            .chunks(self.cols.max(1))
            .map(|row| {
                let (lo, hi) = row
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                        (lo.min(v), hi.max(v))
                    });
                (hi - lo).max(0.0)
            })
            .fold(0.0, f64::max)
    }
}

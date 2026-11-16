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

use std::cell::RefCell;
use std::collections::BinaryHeap;
use std::rc::Rc;

use crate::MathError;
use crate::aabb::Aabb3;
use crate::nurbs::span_index::SpanIndex;
use crate::nurbs::surface::NurbsSurface;
use crate::vec::Point3;

use super::{SurfaceProjection, surface_descend};

/// The most halvings of parts that one proof makes. Two nearest points that
/// the slack cannot tell apart can need more, and so can a span whose
/// weights differ too much to bound; past this many, the proof fails.
const MOST_HALVINGS: usize = 512;

/// Weights of a part farther apart than this ratio leave the products of
/// its bounds without digits: the squares of the least underflow.
const WEIGHT_SPREAD: f64 = 1e100;

/// The share of the largest coordinate of the point and the surface below
/// which the slack of the proof does not go: rounding moves the sums of the
/// proof by about this much, so a smaller slack proves nothing.
const ROUNDING: f64 = 1e-12;

/// How many times finer one way than the other a part of a surface of
/// degree one is before the search cuts it across that way too.
const FINER: f64 = 16.0;

/// A Newton step on a part shorter than this share of it settles the
/// lowest point: what is left moves the bound by the square of it.
const SETTLED: f64 = 1e-10;

/// `C(p, i) C(p, j) / C(2p, i + j)` at `i * (p + 1) + j`: the share of the
/// product of Bernstein terms `i` and `j` of degree `p` in term `i + j` of
/// degree `2p`. Each degree's shares are made once on each thread.
fn product_shares(p: usize) -> Rc<[f64]> {
    thread_local! {
        static SHARES: RefCell<Vec<Option<Rc<[f64]>>>> = const { RefCell::new(Vec::new()) };
    }
    SHARES.with(|cache| {
        let mut cache = cache.borrow_mut();
        if cache.len() <= p {
            cache.resize(p + 1, None);
        }
        cache[p]
            .get_or_insert_with(|| make_product_shares(p).into())
            .clone()
    })
}

fn make_product_shares(p: usize) -> Vec<f64> {
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
    /// The weights spread too far for the coefficients to bound the part:
    /// only its box does, and the bound from the curvature is not read.
    rough: bool,
    /// Where, in the part's own parameters, the lowest point of the part
    /// it was cut from lies, held to it.
    hint: Option<[f64; 2]>,
    /// The part's size across u and across v, as shares of its span's.
    share: [f64; 2],
}

impl Cell {
    /// The coefficients of `g = |q|² - c w²` on the part, in `out`.
    fn gaps(&self, c: f64, out: &mut Vec<f64>) {
        out.clear();
        out.extend(
            self.squares
                .iter()
                .zip(&self.weights)
                .map(|(s, w)| s - c * w),
        );
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
    shares_u: Rc<[f64]>,
    shares_v: Rc<[f64]>,
    domain: [f64; 4],
    closed: (bool, bool),
    scratch: RefCell<Scratch>,
    pool: RefCell<Pool>,
}

/// Room the evaluation of `g` works in, kept between evaluations.
#[derive(Default)]
struct Scratch {
    work: Vec<[f64; 4]>,
    rows: Vec<[[f64; 4]; 3]>,
    column: Vec<[f64; 4]>,
}

/// Lists that parts are done with, for the next parts to fill: a proof
/// makes and drops many parts, each of a few short lists.
#[derive(Default)]
struct Pool {
    points: Vec<Vec<[f64; 4]>>,
    values: Vec<Vec<f64>>,
}

/// The most lists of each kind that a thread keeps between proofs.
const KEPT_LISTS: usize = 64;

thread_local! {
    /// The room and the lists of the last proof on this thread, for the
    /// next: most proofs are short, and would spend much of their time
    /// making these afresh.
    static ROOM: RefCell<Option<(Scratch, Pool)>> = const { RefCell::new(None) };
}

impl Drop for Search {
    fn drop(&mut self) {
        let scratch = std::mem::take(self.scratch.get_mut());
        let mut pool = std::mem::take(self.pool.get_mut());
        pool.points.truncate(KEPT_LISTS);
        pool.values.truncate(KEPT_LISTS);
        // A thread that is ending has no room to keep, and needs none.
        let _ = ROOM.try_with(|room| {
            if let Ok(mut room) = room.try_borrow_mut() {
                *room = Some((scratch, pool));
            }
        });
    }
}

impl Search {
    fn take_points(&self) -> Vec<[f64; 4]> {
        let mut list = self.pool.borrow_mut().points.pop().unwrap_or_default();
        list.clear();
        list
    }

    fn take_values(&self) -> Vec<f64> {
        let mut list = self.pool.borrow_mut().values.pop().unwrap_or_default();
        list.clear();
        list
    }

    fn give_values(&self, list: Vec<f64>) {
        self.pool.borrow_mut().values.push(list);
    }

    fn give(&self, cell: Cell) {
        let mut pool = self.pool.borrow_mut();
        pool.points.push(cell.points);
        pool.values.push(cell.squares);
        pool.values.push(cell.weights);
    }
}

/// The nearest point of `surface` to `point`, from `start`: no point of the
/// surface is nearer than the result by more than `tolerance`, or than a
/// share of the size of the model that rounding can move a distance by.
/// Each part of a patch whose bounds do not prove that is halved, up to
/// [`MOST_HALVINGS`] times.
///
/// # Errors
///
/// Returns [`MathError::ConvergenceFailure`] where a part is left that the
/// halvings did not settle.
pub(super) fn nearest(
    surface: &NurbsSurface,
    index: &SpanIndex,
    point: Point3,
    tolerance: f64,
    start: SurfaceProjection,
) -> Result<SurfaceProjection, MathError> {
    let hull = surface.aabb();
    let size = [hull.min, hull.max, point]
        .iter()
        .flat_map(|c| [c.x(), c.y(), c.z()])
        .fold(0.0, |m: f64, c| m.max(c.abs()));
    let slack = tolerance.max(ROUNDING * size);
    let mut best = start;
    if best.distance <= slack {
        return Ok(best);
    }
    let unsettled = Err(MathError::ConvergenceFailure {
        iterations: MOST_HALVINGS,
    });
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let (rows, cols) = (surface.degree_u() + 1, surface.degree_v() + 1);
    let (scratch, pool) = ROOM
        .try_with(|room| room.try_borrow_mut().ok().and_then(|mut room| room.take()))
        .ok()
        .flatten()
        .unwrap_or_default();
    let search = Search {
        tolerance,
        rows,
        cols,
        shares_u: product_shares(rows - 1),
        shares_v: product_shares(cols - 1),
        domain: [u_min, u_max, v_min, v_max],
        closed: (surface.is_periodic_u(), surface.is_periodic_v()),
        scratch: RefCell::new(scratch),
        pool: RefCell::new(pool),
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
        // A common factor of the weights leaves the surface as it is; taken
        // out, it cannot carry the products out of the range of a float.
        let scale = net.iter().fold(0.0_f64, |m, q| m.max(q[3].abs()));
        // A weight that is no longer a normal float once a share of the
        // largest leaves no sum to bound the span with.
        if scale <= 0.0
            || !net.iter().flatten().all(|c| c.is_finite())
            || net.iter().any(|q| !(q[3] / scale).is_normal())
        {
            return unsettled;
        }
        let mut points = search.take_points();
        points.extend(net.iter().map(|q| {
            let w = q[3] / scale;
            [
                q[0] / scale - w * point.x(),
                q[1] / scale - w * point.y(),
                q[2] / scale - w * point.z(),
                w,
            ]
        }));
        heap.extend(search.cell(points, u, v, bound, None));
    }

    let mut halvings = 0;
    while let Some(cell) = heap.pop() {
        let bound = below(&best);
        if cell.floor >= bound {
            search.give(cell);
            break;
        }
        let mut gaps = search.take_values();
        cell.gaps(bound, &mut gaps);
        let (at, gap, slope) = if cell.rough {
            // The sums for `g` lose their digits; the surface itself does
            // not, and its middle may be nearer.
            let middle = surface.evaluate(
                f64::midpoint(cell.u.0, cell.u.1),
                f64::midpoint(cell.v.0, cell.v.1),
            );
            let d = (middle - point).length_squared();
            ([0.5, 0.5], d - bound, [0.0, 0.0])
        } else {
            search.lowest(&cell, bound, &best, &gaps)
        };
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
                    return Ok(best);
                }
                search.give_values(gaps);
                heap.push(cell);
                continue;
            }
        }
        // Along a way of degree one, `g` is a quadratic, whose least on each
        // line across the part its three coefficients give exactly: such a
        // part is only ever cut along the other way.
        let (straight_u, straight_v) = (search.rows == 2, search.cols == 2);
        let (mut settled, mut lost) = (false, false);
        for (straight, across_u) in [(straight_u, true), (straight_v, false)] {
            if straight && !settled && !cell.rough {
                match search.settles_across(&gaps, across_u, at) {
                    Some(proved) => settled = proved,
                    None => lost = true,
                }
            }
        }
        // Across the way of degree one, the bound is exact, but rounding
        // in its sums is not: where `g` comes within rounding of zero, the
        // part has to shrink both ways. So that way is cut too, once the
        // part is far finer the other way.
        let mut along_u = if straight_u {
            search.cols == 1 || cell.share[1] * FINER < cell.share[0]
        } else {
            search.rows != 1 && cell.share[0] * FINER >= cell.share[1]
        };
        if !settled && (straight_u == straight_v || lost) {
            let partials = search.second_partials(&gaps);
            let [ss, st, tt] = &partials;
            let bends = (ss.least(), tt.least(), st.largest_size());
            settled = !cell.rough && model_floor(at, gap, slope, bends) >= 0.0;
            along_u = search.halve_along_u(&gaps, &partials);
            for grid in partials {
                search.give_values(grid.values);
            }
        }
        search.give_values(gaps);
        if settled {
            search.give(cell);
            continue;
        }
        if halvings >= MOST_HALVINGS {
            return unsettled;
        }
        halvings += 1;
        let pieces = search.pieces(cell, along_u, at, below(&best));
        heap.extend(pieces.into_iter().flatten());
    }
    for cell in heap {
        search.give(cell);
    }
    Ok(best)
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
        mut points: Vec<[f64; 4]>,
        u: (f64, f64),
        v: (f64, f64),
        bound: f64,
        hint: Option<[f64; 2]>,
    ) -> Option<Cell> {
        // A common factor of the weights leaves the part as it is; taken
        // out, the products below stay in the range of a float.
        let most = points.iter().fold(0.0_f64, |m, q| m.max(q[3]));
        if most > 0.0 && most.is_finite() && most.to_bits() != 1.0_f64.to_bits() {
            for q in &mut points {
                for c in q.iter_mut() {
                    *c /= most;
                }
            }
        }
        // The part lies in the box of its points, which the point is at the
        // origin of: the box's squared distance from the origin bounds too.
        let hull = Aabb3::from_points(points.iter().map(|q| {
            let w = q[3];
            Point3::new(q[0] / w, q[1] / w, q[2] / w)
        }))
        .distance_squared_to_point(Point3::new(0.0, 0.0, 0.0));
        if hull >= bound {
            self.pool.borrow_mut().points.push(points);
            return None;
        }
        let (rows, cols) = (self.rows, self.cols);
        let wide = 2 * cols - 1;
        let len = (2 * rows - 1) * wide;
        let polynomial = points.iter().all(|q| q[3].to_bits() == 1.0_f64.to_bits());
        let (mut squares, mut weights) = (self.take_values(), self.take_values());
        squares.resize(len, 0.0);
        weights.resize(len, if polynomial { 1.0 } else { 0.0 });
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
        let (least, most) = points.iter().fold((f64::INFINITY, 0.0_f64), |(lo, hi), q| {
            (lo.min(q[3]), hi.max(q[3]))
        });
        let rough = least * WEIGHT_SPREAD < most
            || !squares.iter().chain(&weights).all(|c| c.is_finite())
            || weights.iter().any(|&w| w <= 0.0);
        let floor = if rough {
            hull
        } else {
            squares
                .iter()
                .zip(&weights)
                .map(|(s, w)| s / w)
                .fold(f64::INFINITY, f64::min)
                .max(hull)
        };
        let cell = Cell {
            points,
            u,
            v,
            squares,
            weights,
            floor,
            rough,
            hint,
            share: [1.0, 1.0],
        };
        if floor < bound {
            Some(cell)
        } else {
            self.give(cell);
            None
        }
    }

    /// Whether `g` stays at or above zero on the part, where it is a
    /// quadratic across u (`across_u`) or across v. On each line across,
    /// with Bernstein coefficients `b0`, `b1` and `b2`, that holds if and
    /// only if `b0` and `b2` are not negative and `b1 ≥ -√(b0 b2)`: so where
    /// `b0`, `b2`, and `b1` or `b0 b2 - b1²`, stay at or above zero along the
    /// other way. `at` is the lowest point of `g` on the part. `None` where
    /// a product of two coefficients falls below the least float, and so
    /// can lose its sign.
    fn settles_across(&self, gaps: &[f64], across_u: bool, at: [f64; 2]) -> Option<bool> {
        let (du, dv) = (2 * (self.rows - 1), 2 * (self.cols - 1));
        let mut lines = [self.take_values(), self.take_values(), self.take_values()];
        for (k, line) in lines.iter_mut().enumerate() {
            if across_u {
                line.extend_from_slice(&gaps[k * (dv + 1)..=k * (dv + 1) + dv]);
            } else {
                line.extend((0..=du).map(|i| gaps[i * (dv + 1) + k]));
            }
        }
        // A positive factor leaves each sign as it is; taken out, the
        // products below cannot drop below the least float.
        let most = lines.iter().flatten().fold(0.0_f64, |m, c| m.max(c.abs()));
        if most > 0.0 && most.is_finite() {
            for c in lines.iter_mut().flatten() {
                *c /= most;
            }
        }
        let hint = if across_u { at[1] } else { at[0] };
        let [low, middle, high] = &lines;
        let mut work = self.take_values();
        let mut settled =
            stays_positive(low, hint, &mut work) && stays_positive(high, hint, &mut work);
        if settled && middle.iter().any(|&c| c < 0.0) {
            let n = low.len() - 1;
            let shares = product_shares(n);
            let mut cross = self.take_values();
            cross.resize(2 * n + 1, 0.0);
            let lost = |a: f64, b: f64| a != 0.0 && b != 0.0 && (a * b).abs() < f64::MIN_POSITIVE;
            let mut whole = true;
            for i in 0..=n {
                for j in 0..=n {
                    whole &= !lost(low[i], high[j]) && !lost(middle[i], middle[j]);
                    cross[i + j] +=
                        shares[i * (n + 1) + j] * (low[i] * high[j] - middle[i] * middle[j]);
                }
            }
            settled = whole && stays_positive(&cross, hint, &mut work);
            self.give_values(cross);
            if !whole {
                self.give_values(work);
                for line in lines {
                    self.give_values(line);
                }
                return None;
            }
        }
        self.give_values(work);
        for line in lines {
            self.give_values(line);
        }
        Some(settled)
    }

    /// The Bernstein coefficients of the second partials of `g` on the part,
    /// in its own parameters `s` and `t`: `g_ss`, `g_st` and `g_tt`.
    fn second_partials(&self, gaps: &[f64]) -> [Grid; 3] {
        let (du, dv) = (2 * (self.rows - 1), 2 * (self.cols - 1));
        let at = |k: usize, l: usize| gaps[k * (dv + 1) + l];
        let grid = |rows: usize, cols: usize, scale: usize, value: &dyn Fn(usize, usize) -> f64| {
            let mut values = self.take_values();
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
    /// the part, or the lowest point of the part it was cut from, its
    /// middle, and its lowest corner, which `gaps` gives. Its `(s, t)`, `g`
    /// there, and the gradient there.
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
        let inside = |a: [f64; 2]| a.iter().all(|&a| a > 0.0 && a < 1.0);
        // A part that holds the nearest point so far has it for its lowest
        // point, or as near as one Newton step finds. A part cut from
        // another starts where the lowest point of that one lies.
        let starts: &[[f64; 2]] = match cell.hint {
            _ if inside(held) => &[held],
            Some(hint) if inside(hint) => &[hint],
            Some(hint) => &[hint, corner],
            None => &[held, [0.5, 0.5], corner],
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
            let moved = (next[0] - x[0]).abs().max((next[1] - x[1]).abs());
            if moved < SETTLED {
                break;
            }
            let (g_next, grad_next, hess_next) = self.gap_at(cell, c, next);
            if g_next.is_nan() || g_next >= g {
                break;
            }
            (x, g, grad, hess) = (next, g_next, grad_next, hess_next);
        }
        (x, g, grad)
    }

    /// The pieces of the part, across u or across v, by de Casteljau on
    /// its control points. Where its lowest point `at` lies inside it, the
    /// middle piece is half the part, about that point: the bounds from the
    /// curvature hold on a piece whose bends spread little, and they spread
    /// least about the lowest point. Elsewhere the pieces are the halves.
    fn pieces(&self, cell: Cell, along_u: bool, at: [f64; 2], bound: f64) -> Vec<Option<Cell>> {
        let x = if along_u { at[0] } else { at[1] };
        let mut cuts = Vec::with_capacity(2);
        if x > 0.0 && x < 1.0 {
            // A piece narrower than this share of the part is left joined
            // to the piece beside it.
            let least = 1.0 / 16.0;
            if x - 0.25 > least {
                cuts.push(x - 0.25);
            }
            if x + 0.25 < 1.0 - least {
                cuts.push(x + 0.25);
            }
        } else {
            cuts.push(0.5);
        }
        let (lo, hi) = if along_u { cell.u } else { cell.v };
        let mut pieces = Vec::with_capacity(cuts.len() + 1);
        let mut rest = self.take_points();
        rest.extend_from_slice(&cell.points);
        let mut from = 0.0;
        let k = usize::from(!along_u);
        let shared = |piece: Option<Cell>, a: f64, b: f64| {
            piece.map(|mut piece| {
                piece.share = cell.share;
                piece.share[k] *= b - a;
                piece
            })
        };
        let hint_in = |a: f64, b: f64| {
            let mut hint = at;
            let k = usize::from(!along_u);
            hint[k] = ((at[k] - a) / (b - a)).clamp(0.0, 1.0);
            Some(hint)
        };
        for &cut in &cuts {
            let mut high = self.take_points();
            self.split(&mut rest, &mut high, along_u, (cut - from) / (1.0 - from));
            let range = (param_at(from, (lo, hi)), param_at(cut, (lo, hi)));
            let (u, v) = if along_u {
                (range, cell.v)
            } else {
                (cell.u, range)
            };
            pieces.push(shared(
                self.cell(rest, u, v, bound, hint_in(from, cut)),
                from,
                cut,
            ));
            rest = high;
            from = cut;
        }
        let range = (param_at(from, (lo, hi)), hi);
        let (u, v) = if along_u {
            (range, cell.v)
        } else {
            (cell.u, range)
        };
        pieces.push(shared(
            self.cell(rest, u, v, bound, hint_in(from, 1.0)),
            from,
            1.0,
        ));
        self.give(cell);
        pieces
    }

    /// Split the part on `low` at `tau` of the way across u or across v:
    /// `low` keeps the piece before, and `high` gets the piece after.
    fn split(&self, low: &mut [[f64; 4]], high: &mut Vec<[f64; 4]>, along_u: bool, tau: f64) {
        let (rows, cols) = (self.rows, self.cols);
        let (n, m) = if along_u { (rows, cols) } else { (cols, rows) };
        let at = |k: usize, l: usize| if along_u { k * cols + l } else { l * cols + k };
        high.clear();
        high.extend_from_slice(low);
        let mut scratch = self.scratch.borrow_mut();
        let work = &mut scratch.work;
        for l in 0..m {
            work.clear();
            work.extend((0..n).map(|k| low[at(k, l)]));
            high[at(n - 1, l)] = work[n - 1];
            for r in 1..n {
                for k in 0..n - r {
                    for c in 0..4 {
                        work[k][c] = (1.0 - tau) * work[k][c] + tau * work[k + 1][c];
                    }
                }
                low[at(r, l)] = work[0];
                high[at(n - 1 - r, l)] = work[n - 1 - r];
            }
        }
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

/// Whether the polynomial with Bernstein coefficients `coefs` stays at or
/// above zero on `[0, 1]`: its coefficients do, or the quadratic it never
/// goes below does. That quadratic is taken at its lowest point, which
/// Newton steps find from the best of `hint` and the two ends, and its bend
/// is the least second derivative, from the coefficients.
fn stays_positive(coefs: &[f64], hint: f64, work: &mut Vec<f64>) -> bool {
    if coefs.iter().all(|&c| c >= 0.0) {
        return true;
    }
    let n = coefs.len().saturating_sub(1);
    if n < 2 {
        // A line has its least at an end, which its coefficient there is.
        return false;
    }
    let bend = coefs
        .windows(3)
        .map(|w| w[2] - 2.0 * w[1] + w[0])
        .fold(f64::INFINITY, f64::min)
        * (n * (n - 1)) as f64;
    let mut x = hint.clamp(0.0, 1.0);
    let [mut value, mut slope, mut curve] = line_at(coefs, x, work);
    let degree = n as f64;
    let ends = [
        (
            0.0,
            coefs[0],
            degree * (coefs[1] - coefs[0]),
            coefs[2] - 2.0 * coefs[1] + coefs[0],
        ),
        (
            1.0,
            coefs[n],
            degree * (coefs[n] - coefs[n - 1]),
            coefs[n] - 2.0 * coefs[n - 1] + coefs[n - 2],
        ),
    ];
    for (end, v, s, c) in ends {
        if v < value {
            (x, value, slope, curve) = (end, v, s, degree * (degree - 1.0) * c);
        }
    }
    for _ in 0..8 {
        if curve <= 0.0 {
            break;
        }
        let next = (x - slope / curve).clamp(0.0, 1.0);
        if (next - x).abs() < SETTLED {
            break;
        }
        let [v, s, c] = line_at(coefs, next, work);
        if v.is_nan() || v >= value {
            break;
        }
        (x, value, slope, curve) = (next, v, s, c);
    }
    value >= 0.0 && value + quadratic_floor(x, slope, bend) >= 0.0
}

/// The polynomial with Bernstein coefficients `coefs` at `t`, with its
/// first and second derivatives, by de Casteljau in `work`.
fn line_at(coefs: &[f64], t: f64, work: &mut Vec<f64>) -> [f64; 3] {
    let n = coefs.len() - 1;
    work.clear();
    work.extend_from_slice(coefs);
    for level in 0..n.saturating_sub(2) {
        for k in 0..n - level {
            work[k] = (1.0 - t) * work[k] + t * work[k + 1];
        }
    }
    let lerp = |a: f64, b: f64| (1.0 - t) * a + t * b;
    let degree = n as f64;
    match n {
        0 => [work[0], 0.0, 0.0],
        1 => [lerp(work[0], work[1]), work[1] - work[0], 0.0],
        _ => {
            let (low, high) = (lerp(work[0], work[1]), lerp(work[1], work[2]));
            [
                lerp(low, high),
                degree * (high - low),
                degree * (degree - 1.0) * (work[2] - 2.0 * work[1] + work[0]),
            ]
        }
    }
}

/// The least of `slope h + bend h² / 2` over `h` in `[-x, 1 - x]`.
fn quadratic_floor(x: f64, slope: f64, bend: f64) -> f64 {
    let (lo, hi) = (-x, 1.0 - x);
    let at = |h: f64| slope * h + 0.5 * bend * h * h;
    let ends = at(lo).min(at(hi));
    if bend > 0.0 {
        ends.min(at((-slope / bend).clamp(lo, hi)))
    } else {
        ends
    }
}

/// The least, on the part, of the quadratic that `g` never goes below:
/// `g(x + h) ≥ g + slope · h + h_s² D_s / 2 + h_t² D_t / 2` while every
/// Hessian on the part less `diag(D_s, D_t)` has no negative eigenvalue.
/// With `g_ss ≥ a`, `g_tt ≥ b` and `|g_st| ≤ e` on the part, that holds for
/// `D_s = a - e r` and `D_t = b - e / r` at each `r > 0`; the best of a few
/// returns.
fn model_floor(x: [f64; 2], gap: f64, slope: [f64; 2], (a, b, e): (f64, f64, f64)) -> f64 {
    let along = quadratic_floor;
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

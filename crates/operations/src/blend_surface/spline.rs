//! Spline spaces: collocation at Greville sites, and fitting a family of
//! curves to a shared knot vector, refining it until the fit is in tolerance.

use std::collections::HashMap;

use brepkit_math::nurbs::basis::{basis_funs, ders_basis_funs, find_span};
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::Vec3;

use crate::OperationsError;

/// Knots closer than this are one knot.
const KNOT_MERGE: f64 = 1e-9;

/// More spans than this means the data is not smooth where a break says it
/// is, and more knots will not help.
const MAX_SPANS: usize = 2048;

/// Bisecting a span below this share of the span it started from has stopped
/// being useful.
const MIN_SPAN: f64 = 1e-5;

/// Rounding, amplified a little by a fit, can sit just above a tolerance. A
/// split that gains nothing when this close means the fit has hit that floor.
const NOISE_ROOM: f64 = 8.0;

/// A clamped spline space over one parameter interval.
#[derive(Debug, Clone)]
pub(super) struct Space {
    pub degree: usize,
    pub knots: Vec<f64>,
}

impl Space {
    /// The space of `degree` over `breaks`, each interior one repeated to its
    /// given multiplicity (clamped to the degree) and the ends clamped.
    pub fn clamped(degree: usize, breaks: &[(f64, usize)]) -> Self {
        let mut knots = Vec::new();
        let last = breaks.len().saturating_sub(1);
        for (i, &(at, multiplicity)) in breaks.iter().enumerate() {
            let m = if i == 0 || i == last {
                degree + 1
            } else {
                multiplicity.clamp(1, degree)
            };
            knots.extend(std::iter::repeat_n(at, m));
        }
        Self { degree, knots }
    }

    /// The space of `degree` over `breaks`, continuous only in value at each
    /// interior one.
    pub fn c0_over(degree: usize, breaks: &[f64]) -> Self {
        let with_multiplicity: Vec<_> = breaks.iter().map(|&b| (b, degree)).collect();
        Self::clamped(degree, &with_multiplicity)
    }

    pub fn count(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    /// The Greville abscissae, one per basis function, strictly increasing.
    pub fn sites(&self) -> Vec<f64> {
        let q = self.degree;
        (0..self.count())
            .map(|i| {
                let run = &self.knots[i + 1..=i + q];
                if run[run.len() - 1] - run[0] <= 0.0 {
                    run[0]
                } else {
                    run.iter().sum::<f64>() / q as f64
                }
            })
            .collect()
    }

    /// The first basis function not zero at `x`, and the values from there.
    pub fn basis_at(&self, x: f64) -> (usize, Vec<f64>) {
        let span = find_span(self.count(), self.degree, x, &self.knots);
        (
            span - self.degree,
            basis_funs(span, x, self.degree, &self.knots),
        )
    }

    /// Basis functions and their derivatives up to order 2 at `x`: the first
    /// index, and `ders[k][i]`.
    pub fn basis_ders_at(&self, x: f64, order: usize) -> (usize, Vec<Vec<f64>>) {
        let span = find_span(self.count(), self.degree, x, &self.knots);
        (
            span - self.degree,
            ders_basis_funs(span, x, self.degree, order, &self.knots),
        )
    }

    pub fn evaluate(&self, coefficients: &[Vec3], x: f64) -> Vec3 {
        let (first, values) = self.basis_at(x);
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        for (k, n) in values.iter().enumerate() {
            sum += coefficients[first + k] * *n;
        }
        sum
    }

    pub fn spans(&self) -> Vec<(f64, f64)> {
        self.knots
            .windows(2)
            .filter(|w| w[1] - w[0] > KNOT_MERGE)
            .map(|w| (w[0], w[1]))
            .collect()
    }

    /// This space with a simple knot at the middle of each of `spans` and of
    /// the others between the same value-only knots. A collocation amplifies
    /// noise with the ratio of neighbouring spans, so spans are kept even.
    pub fn split(&self, spans: &[(f64, f64)]) -> Self {
        let mut bounds: Vec<f64> = Vec::new();
        let mut run = 0;
        for (i, &knot) in self.knots.iter().enumerate() {
            run = if i > 0 && knot - self.knots[i - 1] <= KNOT_MERGE {
                run + 1
            } else {
                1
            };
            if run == self.degree {
                bounds.push(knot);
            }
        }
        let piece = |x: f64| bounds.partition_point(|b| *b <= x);
        let stray: Vec<usize> = spans.iter().map(|&(a, b)| piece(0.5 * (a + b))).collect();
        let mut knots = self.knots.clone();
        knots.extend(
            self.spans()
                .into_iter()
                .filter(|&(a, b)| stray.contains(&piece(0.5 * (a + b))))
                .map(|(a, b)| 0.5 * (a + b)),
        );
        knots.sort_by(f64::total_cmp);
        Self {
            degree: self.degree,
            knots,
        }
    }

    /// Whether any of `spans` has been bisected too far below the span of this
    /// space it lies in.
    fn split_too_far(&self, spans: &[(f64, f64)]) -> bool {
        let start = self.spans();
        spans.iter().any(|&(a, b)| {
            let from = start
                .iter()
                .find(|&&(lo, hi)| a >= lo && b <= hi)
                .map_or(b - a, |&(lo, hi)| hi - lo);
            b - a < MIN_SPAN * from
        })
    }

    pub fn collocation(&self) -> Result<Collocation, OperationsError> {
        let n = self.count();
        let q = self.degree;
        let mut band = vec![0.0; n * (2 * q + 1)];
        for (i, &site) in self.sites().iter().enumerate() {
            let (first, values) = self.basis_at(site);
            for (k, value) in values.iter().enumerate() {
                let j = first + k;
                if j + q >= i && j <= i + q {
                    band[i * (2 * q + 1) + j + q - i] = *value;
                }
            }
        }
        let mut factored = Collocation { n, q, band };
        factored.factor()?;
        Ok(factored)
    }
}

/// A B-spline collocation matrix at Greville sites, in band storage.
///
/// Such a matrix is totally positive, so elimination without pivoting is
/// stable and keeps the band.
pub(super) struct Collocation {
    n: usize,
    q: usize,
    band: Vec<f64>,
}

impl Collocation {
    fn at(&self, i: usize, j: usize) -> usize {
        i * (2 * self.q + 1) + j + self.q - i
    }

    fn factor(&mut self) -> Result<(), OperationsError> {
        let (n, q) = (self.n, self.q);
        for k in 0..n {
            let pivot = self.band[self.at(k, k)];
            if pivot.abs() < 1e-13 {
                return Err(OperationsError::InvalidInput {
                    reason: "a spline fit met a singular collocation matrix".into(),
                });
            }
            for i in k + 1..(k + q + 1).min(n) {
                let at_ik = self.at(i, k);
                let factor = self.band[at_ik] / pivot;
                self.band[at_ik] = factor;
                for j in k + 1..(k + q + 1).min(n) {
                    if j + q >= i && j <= i + q {
                        let from = self.band[self.at(k, j)];
                        let at_ij = self.at(i, j);
                        self.band[at_ij] -= factor * from;
                    }
                }
            }
        }
        Ok(())
    }

    pub fn solve(&self, data: &mut [Vec3]) {
        let (n, q) = (self.n, self.q);
        for i in 1..n {
            for j in i.saturating_sub(q)..i {
                data[i] -= data[j] * self.band[self.at(i, j)];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..(i + q + 1).min(n) {
                data[i] -= data[j] * self.band[self.at(i, j)];
            }
            data[i] = data[i] * (1.0 / self.band[self.at(i, i)]);
        }
    }
}

/// Fit each of `rows` curves `f(x)[row]` in one shared space, starting from
/// `start` and bisecting any span where a fit strays past `tolerance`.
///
/// # Errors
///
/// Returns an error from `f`, or if the data will not come within tolerance,
/// which means it is not smooth inside a span.
pub(super) fn fit_rows<F>(
    start: Space,
    rows: usize,
    tolerance: f64,
    what: &str,
    mut f: F,
) -> Result<(Space, Vec<Vec<Vec3>>), OperationsError>
where
    F: FnMut(f64) -> Result<Vec<Vec3>, OperationsError>,
{
    let origin = start.clone();
    let mut space = start;
    let mut cache: HashMap<u64, Vec<Vec3>> = HashMap::new();
    let points = gauss_legendre_points(3);
    let mut last_worst = f64::INFINITY;
    loop {
        let sites = space.sites();
        let mut coefficients = vec![Vec::with_capacity(sites.len()); rows];
        for &site in &sites {
            let values = cached(&mut cache, site, &mut f)?;
            for (row, value) in coefficients.iter_mut().zip(values) {
                row.push(*value);
            }
        }
        let collocation = space.collocation()?;
        for row in &mut coefficients {
            collocation.solve(row);
        }

        let mut stray = Vec::new();
        let mut worst = 0.0_f64;
        for (a, b) in space.spans() {
            let mut error = 0.0_f64;
            for point in points {
                let x = 0.5 * (a + b) + 0.5 * (b - a) * point.x;
                let values = cached(&mut cache, x, &mut f)?;
                for (row, value) in coefficients.iter().zip(values) {
                    error = error.max((space.evaluate(row, x) - *value).length());
                }
            }
            worst = worst.max(error);
            if error > tolerance {
                stray.push((a, b));
            }
        }
        let at_the_floor = worst <= NOISE_ROOM * tolerance && worst >= 0.5 * last_worst;
        if stray.is_empty() || at_the_floor {
            return Ok((space, coefficients));
        }
        last_worst = worst;
        let spans = space.spans().len();
        if spans + stray.len() > MAX_SPANS || origin.split_too_far(&stray) {
            return Err(OperationsError::InvalidInput {
                reason: format!(
                    "{what} could not be fitted within {tolerance:.1e}: {worst:.1e} remains \
                     over {spans} spans, so the data is not smooth along it"
                ),
            });
        }
        space = space.split(&stray);
    }
}

fn cached<'a, F>(
    cache: &'a mut HashMap<u64, Vec<Vec3>>,
    x: f64,
    f: &mut F,
) -> Result<&'a Vec<Vec3>, OperationsError>
where
    F: FnMut(f64) -> Result<Vec<Vec3>, OperationsError>,
{
    Ok(match cache.entry(x.to_bits()) {
        std::collections::hash_map::Entry::Occupied(hit) => hit.into_mut(),
        std::collections::hash_map::Entry::Vacant(slot) => slot.insert(f(x)?),
    })
}

/// A tensor-product net, `net[i][j]`, of data on the grid of the two spaces'
/// sites.
pub(super) fn solve_grid(
    s_space: &Space,
    t_space: &Space,
    mut data: Vec<Vec<Vec3>>,
) -> Result<Vec<Vec<Vec3>>, OperationsError> {
    let along_t = t_space.collocation()?;
    for row in &mut data {
        along_t.solve(row);
    }
    let along_s = s_space.collocation()?;
    let mut column = vec![Vec3::new(0.0, 0.0, 0.0); s_space.count()];
    for j in 0..t_space.count() {
        for (i, slot) in column.iter_mut().enumerate() {
            *slot = data[i][j];
        }
        along_s.solve(&mut column);
        for (i, value) in column.iter().enumerate() {
            data[i][j] = *value;
        }
    }
    Ok(data)
}

/// Fit `f(s, t)` in a tensor space grown from `s_start` and `t_start` until it
/// is within `tolerance` of `f` between the sites in both directions.
///
/// # Errors
///
/// Returns an error from `f`, or if the data will not come within tolerance.
pub(super) fn fit_grid<F>(
    s_start: Space,
    t_start: Space,
    tolerance: f64,
    what: &str,
    mut f: F,
) -> Result<(Space, Space, Vec<Vec<Vec3>>), OperationsError>
where
    F: FnMut(f64, f64) -> Result<Vec3, OperationsError>,
{
    let (s_origin, t_origin) = (s_start.clone(), t_start.clone());
    let (mut s_space, mut t_space) = (s_start, t_start);
    let mut cache: HashMap<(u64, u64), Vec3> = HashMap::new();
    let mut value = |s: f64, t: f64| -> Result<Vec3, OperationsError> {
        match cache.entry((s.to_bits(), t.to_bits())) {
            std::collections::hash_map::Entry::Occupied(hit) => Ok(*hit.get()),
            std::collections::hash_map::Entry::Vacant(slot) => Ok(*slot.insert(f(s, t)?)),
        }
    };
    let points = gauss_legendre_points(3);
    let mut last_worst = f64::INFINITY;
    loop {
        let s_sites = s_space.sites();
        let t_sites = t_space.sites();
        let mut grid = Vec::with_capacity(s_sites.len());
        for &s in &s_sites {
            let mut row = Vec::with_capacity(t_sites.len());
            for &t in &t_sites {
                row.push(value(s, t)?);
            }
            grid.push(row);
        }
        let net = solve_grid(&s_space, &t_space, grid)?;
        let at = |s: f64, t: f64| -> Vec3 {
            let (fs, ns) = s_space.basis_at(s);
            let (ft, nt) = t_space.basis_at(t);
            let mut sum = Vec3::new(0.0, 0.0, 0.0);
            for (a, na) in ns.iter().enumerate() {
                for (b, nb) in nt.iter().enumerate() {
                    sum += net[fs + a][ft + b] * (*na * *nb);
                }
            }
            sum
        };

        // Error that vanishes on every line through a site is invisible to
        // reads along those lines, so each cell is also read between them.
        let (s_spans, t_spans) = (s_space.spans(), t_space.spans());
        let between = |spans: &[(f64, f64)]| -> Vec<(usize, f64)> {
            spans
                .iter()
                .enumerate()
                .flat_map(|(k, &(a, b))| {
                    points
                        .iter()
                        .map(move |point| (k, 0.5 * (a + b) + 0.5 * (b - a) * point.x))
                })
                .collect()
        };
        let (s_between, t_between) = (between(&s_spans), between(&t_spans));
        let mut s_off = vec![false; s_spans.len()];
        let mut t_off = vec![false; t_spans.len()];
        let mut worst = 0.0_f64;
        let mut strays = |s: f64, t: f64| -> Result<bool, OperationsError> {
            let error = (at(s, t) - value(s, t)?).length();
            worst = worst.max(error);
            Ok(error > tolerance)
        };
        for &(i, s) in &s_between {
            for &t in &t_sites {
                s_off[i] |= strays(s, t)?;
            }
        }
        for &(j, t) in &t_between {
            for &s in &s_sites {
                t_off[j] |= strays(s, t)?;
            }
        }
        for &(i, s) in &s_between {
            for &(j, t) in &t_between {
                if strays(s, t)? {
                    s_off[i] = true;
                    t_off[j] = true;
                }
            }
        }
        let flagged = |spans: &[(f64, f64)], off: &[bool]| -> Vec<(f64, f64)> {
            spans
                .iter()
                .zip(off)
                .filter(|(_, off)| **off)
                .map(|(span, _)| *span)
                .collect()
        };
        let stray_s = flagged(&s_spans, &s_off);
        let stray_t = flagged(&t_spans, &t_off);
        let at_the_floor = worst <= NOISE_ROOM * tolerance && worst >= 0.5 * last_worst;
        if (stray_s.is_empty() && stray_t.is_empty()) || at_the_floor {
            return Ok((s_space, t_space, net));
        }
        last_worst = worst;
        let (s_count, t_count) = (s_spans.len(), t_spans.len());
        if s_count + stray_s.len() > MAX_SPANS
            || t_count + stray_t.len() > MAX_SPANS
            || s_origin.split_too_far(&stray_s)
            || t_origin.split_too_far(&stray_t)
        {
            return Err(OperationsError::InvalidInput {
                reason: format!(
                    "{what} could not be fitted within {tolerance:.1e}: {worst:.1e} remains \
                     over {s_count} by {t_count} spans"
                ),
            });
        }
        s_space = s_space.split(&stray_s);
        t_space = t_space.split(&stray_t);
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    fn grid_at(s_space: &Space, t_space: &Space, net: &[Vec<Vec3>], s: f64, t: f64) -> Vec3 {
        let (fs, ns) = s_space.basis_at(s);
        let (ft, nt) = t_space.basis_at(t);
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        for (a, na) in ns.iter().enumerate() {
            for (b, nb) in nt.iter().enumerate() {
                sum += net[fs + a][ft + b] * (*na * *nb);
            }
        }
        sum
    }

    #[test]
    fn collocation_reproduces_what_the_space_holds() {
        let space = Space::clamped(5, &[(0.0, 0), (0.3, 1), (0.5, 5), (0.7, 3), (1.0, 0)]);
        let exact: Vec<Vec3> = (0..space.count())
            .map(|i| Vec3::new(i as f64, (i as f64).sin(), (i * i) as f64 * 0.01))
            .collect();
        let mut data: Vec<Vec3> = space
            .sites()
            .iter()
            .map(|&x| space.evaluate(&exact, x))
            .collect();
        space.collocation().unwrap().solve(&mut data);
        for (got, want) in data.iter().zip(&exact) {
            assert!((*got - *want).length() < 1e-12, "{got:?} against {want:?}");
        }
    }

    #[test]
    fn a_smooth_curve_comes_within_tolerance() {
        let (space, rows) = fit_rows(Space::c0_over(5, &[0.0, 1.0]), 1, 1e-10, "test", |x| {
            Ok(vec![Vec3::new(x.cos(), x.sin(), x * x)])
        })
        .unwrap();
        for i in 0..=200 {
            let x = f64::from(i) / 200.0;
            let got = space.evaluate(&rows[0], x);
            assert!((got - Vec3::new(x.cos(), x.sin(), x * x)).length() < 1e-9);
        }
    }

    #[test]
    fn a_jump_inside_a_span_is_refused() {
        let result = fit_rows(Space::c0_over(5, &[0.0, 1.0]), 1, 1e-10, "test", |x| {
            Ok(vec![Vec3::new(if x < 0.37 { 0.0 } else { 1.0 }, 0.0, 0.0)])
        });
        assert!(matches!(result, Err(OperationsError::InvalidInput { .. })));
    }

    #[test]
    fn a_grid_fit_holds_a_smooth_surface() {
        let (s_space, t_space, net) = fit_grid(
            Space::c0_over(5, &[0.0, 1.0]),
            Space::c0_over(5, &[0.0, 1.0]),
            1e-9,
            "test",
            |s, t| Ok(Vec3::new(s, t, (s * 3.0).sin() * (t * 2.0).cos())),
        )
        .unwrap();
        for (s, t) in [(0.13, 0.91), (0.5, 0.5), (0.77, 0.21)] {
            let sum = grid_at(&s_space, &t_space, &net, s, t);
            let want = Vec3::new(s, t, (s * 3.0).sin() * (t * 2.0).cos());
            assert!((sum - want).length() < 1e-8);
        }
    }

    /// `P(s) P(t)`, with `P` zero at every site of the first spaces, is zero
    /// wherever one parameter sits on a site line, and 0.10125 high in the
    /// middle.
    #[test]
    fn a_grid_fit_sees_a_bump_between_the_site_lines() {
        let p = |x: f64| (0..=5_u32).map(|i| x - f64::from(i) / 5.0).product::<f64>();
        let f = |s: f64, t: f64| Vec3::new(s, t, 2e6 * p(s) * p(t));
        let tolerance = 1e-6;
        let (s_space, t_space, net) = fit_grid(
            Space::c0_over(5, &[0.0, 1.0]),
            Space::c0_over(5, &[0.0, 1.0]),
            tolerance,
            "test",
            |s, t| Ok(f(s, t)),
        )
        .unwrap();
        let mut worst = 0.0_f64;
        for i in 0..=40 {
            for j in 0..=40 {
                let (s, t) = (f64::from(i) / 40.0, f64::from(j) / 40.0);
                worst = worst.max((grid_at(&s_space, &t_space, &net, s, t) - f(s, t)).length());
            }
        }
        assert!(worst < 10.0 * tolerance, "{worst}");
    }
}

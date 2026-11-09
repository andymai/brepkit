//! Fairing: trading a surface's bending energy against distance from itself.
//!
//! Each free control point moves along the surface's normal at its Greville
//! point, by an amount `t`. Over those amounts, `F` minimises
//! `|F - S|^2 + lambda E(F)`, both integrals over the surface, where `E` is the
//! thin-plate energy `F_xx^2 + 2 F_xy^2 + F_yy^2` in physical lengths along the
//! net. That is one linear solve per `lambda`, and raising `lambda` only lowers
//! the energy. Moving along the normal changes the shape and not the
//! parameterisation: left free, the points would spend the tolerance sliding
//! about in the surface to even out its parameter speed.
//!
//! How far to go is the question. In the modes of the energy against the
//! mass, a shape with stiffness `mu` comes out scaled by `1 / (1 + lambda mu)`,
//! and stiffness goes as the inverse fourth power of wavelength. So `lambda` is
//! set from `mu0`, the softest shape the free points can make: a shape of the
//! surface's own size then keeps nearly all of itself, and bumps of a small
//! fraction of its size lose nearly all of theirs. Only then does the tolerance
//! come in, as a cap on `lambda`. An already fair surface barely moves, however
//! large the tolerance, since it holds nothing short enough to remove.
//!
//! With its weights fixed, a rational surface is linear in its control points
//! too, so it is faired as it stands, with its rational basis functions.

use brepkit_math::nurbs::basis::{basis_funs, find_span};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::predicates::point_in_polygon;
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::wire::WireId;

use super::chain::Chain;
use super::neighbour::Neighbour;
use super::spline::Space;
use crate::OperationsError;

/// A shape this many times stiffer than the softest loses half of itself. At
/// a quarter of the wavelength of the softest shape, stiffness is 256 times
/// higher, so such a wave loses an eighth. A sixteenth of it loses 97%. The
/// softest keeps all but 1 part in 2000.
const HALF_LOST_AT: f64 = 2000.0;

/// Moved less than this share of the tolerance, a surface has not changed.
const NO_CHANGE: f64 = 1e-9;

/// Sample intervals per knot span the distance moved is read at, and the room
/// left in the tolerance for a spline's peak between two samples.
const SAMPLES_PER_SPAN: usize = 8;
const BETWEEN_SAMPLES: f64 = 1.05;

/// A basis function smaller than this at a boundary point does not move it by
/// anything a tolerance would notice.
const REACHES_BOUNDARY: f64 = 1e-9;

/// Smooth `face`, staying within `tolerance` of it, with its boundary held.
///
/// Returns a new face over the same `(u, v)` with copies of its wires.
/// `face` does not change. The surface is held wherever it reaches the
/// boundary (and any seam or hole), and elsewhere stays within `tolerance` at
/// the same `(u, v)`. Its thin-plate energy comes out no higher than it went
/// in. Control points move along the surface's normal only.
///
/// A surface with no bumps to remove barely moves, however large the
/// tolerance: flattening it further would use the tolerance without making it
/// fairer. Any plane, cylinder, cone, sphere or torus comes back unchanged.
/// The shapes removed are those much shorter than the surface: a wave a quarter
/// as long as the longest the surface can hold loses about an eighth of itself,
/// and one a sixteenth as long nearly all. A tight tolerance stops it sooner.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if `tolerance` is not positive
/// and finite. A topology lookup can also fail.
pub fn fair_surface(
    topo: &mut Topology,
    face: FaceId,
    tolerance: f64,
) -> Result<FaceId, OperationsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!("a fairing tolerance must be positive, got {tolerance}"),
        });
    }
    let FaceSurface::Nurbs(source) = topo.face(face)?.surface() else {
        return crate::copy::copy_face(topo, face);
    };
    let source = source.clone();
    let held = held_points(topo, face, &source)?;
    let Some(faired) = fair_net(&source, &held, tolerance)? else {
        return crate::copy::copy_face(topo, face);
    };
    let copy = crate::copy::copy_face(topo, face)?;
    topo.face_mut(copy)?.set_surface(FaceSurface::Nurbs(faired));
    Ok(copy)
}

/// Which control points must not move: those that reach the face's boundary,
/// those whose influence falls outside the face, and a closed surface's seam.
fn held_points(
    topo: &Topology,
    face: FaceId,
    surface: &NurbsSurface,
) -> Result<Vec<Vec<bool>>, OperationsError> {
    let net = surface.control_points();
    let (n, m) = (net.len(), net[0].len());
    let mut held = vec![vec![false; m]; n];
    if surface.is_periodic_u() {
        held[0] = vec![true; m];
        held[n - 1] = vec![true; m];
    }
    if surface.is_periodic_v() {
        for column in &mut held {
            column[0] = true;
            column[m - 1] = true;
        }
    }

    let reader = Neighbour::of(topo, face)?;
    let data = topo.face(face)?;
    let outer = data.outer_wire();
    let inner: Vec<WireId> = data.inner_wires().to_vec();
    let (pu, pv) = (surface.degree_u(), surface.degree_v());
    let (knots_u, knots_v) = (surface.knots_u(), surface.knots_v());
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let samples = 64.max(4 * (n + m));
    let snap = |x: f64, lo: f64, hi: f64| {
        let near = 1e-9 * (hi - lo);
        if (x - lo).abs() < near {
            lo
        } else if (x - hi).abs() < near {
            hi
        } else {
            x
        }
    };
    for wire in std::iter::once(outer).chain(inner.iter().copied()) {
        for (edge, forward) in wire_walk(topo, wire)? {
            let Ok(chain) = Chain::from_walk(topo, &[(edge, forward)], false) else {
                continue;
            };
            for k in 0..=samples {
                let (u, v) = reader.locate(chain.point(k as f64 / samples as f64))?;
                let (u, v) = (snap(u, u0, u1), snap(v, v0, v1));
                let span_u = find_span(n, pu, u, knots_u);
                let span_v = find_span(m, pv, v, knots_v);
                let nu = basis_funs(span_u, u, pu, knots_u);
                let nv = basis_funs(span_v, v, pv, knots_v);
                for (a, value_u) in nu.iter().enumerate() {
                    for (b, value_v) in nv.iter().enumerate() {
                        if value_u * value_v > REACHES_BOUNDARY {
                            held[span_u - pu + a][span_v - pv + b] = true;
                        }
                    }
                }
            }
        }
    }

    hold_outside(&reader, topo, outer, &inner, surface, &mut held)?;
    Ok(held)
}

fn wire_walk(topo: &Topology, wire: WireId) -> Result<Vec<(EdgeId, bool)>, OperationsError> {
    Ok(topo
        .wire(wire)?
        .edges()
        .iter()
        .map(|oe| (oe.edge(), oe.is_forward()))
        .collect())
}

/// Hold control points whose Greville point lies outside the face: they move
/// nothing on it, and moving them only spends the tolerance.
fn hold_outside(
    reader: &Neighbour,
    topo: &Topology,
    outer: WireId,
    inner: &[WireId],
    surface: &NurbsSurface,
    held: &mut [Vec<bool>],
) -> Result<(), OperationsError> {
    let (pu, pv) = reader.periods();
    let closes = |loop_uv: &[(f64, f64)]| {
        let (Some(first), Some(last)) = (loop_uv.first(), loop_uv.last()) else {
            return false;
        };
        pu.is_none_or(|p| (last.0 - first.0).abs() < 0.5 * p)
            && pv.is_none_or(|p| (last.1 - first.1).abs() < 0.5 * p)
    };
    let polygon = |wire: WireId| -> Result<Option<Vec<Point2>>, OperationsError> {
        let loop_uv = reader.chart_loop(topo, wire)?;
        Ok(closes(&loop_uv).then(|| {
            loop_uv
                .iter()
                .map(|&(u, v)| Point2::new(u, v))
                .collect::<Vec<_>>()
        }))
    };
    let outer_polygon = polygon(outer)?;
    let hole_polygons: Vec<_> = inner
        .iter()
        .map(|&wire| polygon(wire))
        .collect::<Result<_, _>>()?;
    let sites_u = Space {
        degree: surface.degree_u(),
        knots: surface.knots_u().to_vec(),
    }
    .sites();
    let sites_v = Space {
        degree: surface.degree_v(),
        knots: surface.knots_v().to_vec(),
    }
    .sites();
    for (i, &u) in sites_u.iter().enumerate() {
        for (j, &v) in sites_v.iter().enumerate() {
            if held[i][j] {
                continue;
            }
            let at = Point2::new(u, v);
            let outside = outer_polygon
                .as_ref()
                .is_some_and(|ring| !point_in_polygon(at, ring));
            let in_hole = hole_polygons
                .iter()
                .flatten()
                .any(|ring| point_in_polygon(at, ring));
            if outside || in_hole {
                held[i][j] = true;
            }
        }
    }
    Ok(())
}

/// Every knot span of positive width inside the domain, however narrow the
/// domain: an unclamped spline's knot vector runs past it.
fn domain_spans(space: &Space) -> Vec<(f64, f64)> {
    let (lo, hi) = (
        space.knots[space.degree],
        space.knots[space.knots.len() - space.degree - 1],
    );
    space
        .knots
        .windows(2)
        .filter(|w| w[1] > w[0] && w[0] >= lo && w[1] <= hi)
        .map(|w| (w[0], w[1]))
        .collect()
}

/// A spline's basis functions and their derivatives at one Gauss point along
/// one parameter.
struct GaussLine {
    first: usize,
    /// `values[k][i]`: the `k`th derivative of the `i`th function from `first`.
    values: Vec<Vec<f64>>,
    /// The Gauss weight, as a share of the parameter's whole range.
    weight: f64,
}

/// The Gauss points of a spline's spans, with `extra` more points per span
/// than a polynomial's mass integrals need.
fn gauss_lines(degree: usize, knots: &[f64], extra: usize) -> Vec<GaussLine> {
    let space = Space {
        degree,
        knots: knots.to_vec(),
    };
    let (lo, hi) = (knots[degree], knots[knots.len() - degree - 1]);
    let points = gauss_legendre_points(degree + 1 + extra);
    let mut lines = Vec::new();
    for (a, b) in domain_spans(&space) {
        let half = 0.5 * (b - a);
        for point in points {
            let (first, mut values) =
                space.basis_ders_at(0.5 * (a + b) + half * point.x, 2.min(degree));
            values.resize(3, vec![0.0; degree + 1]);
            lines.push(GaussLine {
                first,
                values,
                weight: point.w * half / (hi - lo),
            });
        }
    }
    lines
}

/// A rational basis function's value and first and second partials.
#[derive(Clone, Copy, Default)]
struct Shape {
    r: f64,
    ru: f64,
    rv: f64,
    ruu: f64,
    ruv: f64,
    rvv: f64,
}

/// The rational basis functions active at a Gauss point, as `(i, j, shape)`.
fn shapes_at(
    weights: &[Vec<f64>],
    along_u: &GaussLine,
    along_v: &GaussLine,
    out: &mut Vec<(usize, usize, Shape)>,
) {
    out.clear();
    let mut denominator = Shape::default();
    for a in 0..along_u.values[0].len() {
        let i = along_u.first + a;
        for b in 0..along_v.values[0].len() {
            let j = along_v.first + b;
            let w = weights[i][j];
            let (du, dv) = (&along_u.values, &along_v.values);
            let h = Shape {
                r: w * du[0][a] * dv[0][b],
                ru: w * du[1][a] * dv[0][b],
                rv: w * du[0][a] * dv[1][b],
                ruu: w * du[2][a] * dv[0][b],
                ruv: w * du[1][a] * dv[1][b],
                rvv: w * du[0][a] * dv[2][b],
            };
            denominator.r += h.r;
            denominator.ru += h.ru;
            denominator.rv += h.rv;
            denominator.ruu += h.ruu;
            denominator.ruv += h.ruv;
            denominator.rvv += h.rvv;
            out.push((i, j, h));
        }
    }
    let d = denominator;
    for (_, _, shape) in out.iter_mut() {
        let h = *shape;
        let r = h.r / d.r;
        let ru = (h.ru - r * d.ru) / d.r;
        let rv = (h.rv - r * d.rv) / d.r;
        *shape = Shape {
            r,
            ru,
            rv,
            ruu: (h.ruu - 2.0 * ru * d.ru - r * d.ruu) / d.r,
            ruv: (h.ruv - ru * d.rv - rv * d.ru - r * d.ruv) / d.r,
            rvv: (h.rvv - 2.0 * rv * d.rv - r * d.rvv) / d.r,
        };
    }
}

/// A symmetric band matrix in lower band storage.
#[derive(Clone)]
struct BandMatrix {
    n: usize,
    width: usize,
    data: Vec<f64>,
}

impl BandMatrix {
    fn zeros(n: usize, width: usize) -> Self {
        Self {
            n,
            width,
            data: vec![0.0; n * (width + 1)],
        }
    }

    fn at(&self, row: usize, column: usize) -> usize {
        row * (self.width + 1) + column + self.width - row
    }

    /// `self + scale * other`, as a Cholesky factor.
    fn factor_sum(&self, other: &Self, scale: f64) -> Result<Self, OperationsError> {
        let mut sum = self.clone();
        for (slot, value) in sum.data.iter_mut().zip(&other.data) {
            *slot += scale * value;
        }
        for row in 0..sum.n {
            let lo = row.saturating_sub(sum.width);
            for column in lo..=row {
                let mut value = sum.data[sum.at(row, column)];
                for k in lo.max(column.saturating_sub(sum.width))..column {
                    value -= sum.data[sum.at(row, k)] * sum.data[sum.at(column, k)];
                }
                let here = sum.at(row, column);
                if column == row {
                    if value <= 0.0 {
                        return Err(OperationsError::InvalidInput {
                            reason: "fairing met a matrix that is not positive definite".into(),
                        });
                    }
                    sum.data[here] = value.sqrt();
                } else {
                    sum.data[here] = value / sum.data[sum.at(column, column)];
                }
            }
        }
        Ok(sum)
    }

    fn solve_factored(&self, values: &mut [f64]) {
        for row in 0..self.n {
            let mut value = values[row];
            for k in row.saturating_sub(self.width)..row {
                value -= values[k] * self.data[self.at(row, k)];
            }
            values[row] = value / self.data[self.at(row, row)];
        }
        for row in (0..self.n).rev() {
            let mut value = values[row];
            for k in row + 1..(row + self.width + 1).min(self.n) {
                value -= values[k] * self.data[self.at(k, row)];
            }
            values[row] = value / self.data[self.at(row, row)];
        }
    }

    fn times(&self, values: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.n];
        for row in 0..self.n {
            for column in row.saturating_sub(self.width)..=row {
                let entry = self.data[self.at(row, column)];
                out[row] += entry * values[column];
                if column != row {
                    out[column] += entry * values[row];
                }
            }
        }
        out
    }

    fn quadratic(&self, values: &[f64]) -> f64 {
        let mut sum = 0.0;
        for row in 0..self.n {
            for column in row.saturating_sub(self.width)..=row {
                let pair = values[row] * values[column] * self.data[self.at(row, column)];
                sum += if column == row { pair } else { 2.0 * pair };
            }
        }
        sum
    }
}

/// One point on the trade between distance and energy.
struct Step {
    moves: Vec<f64>,
    distance: f64,
    energy: f64,
}

/// Reads how far a displaced net moves its surface, on every knot span.
struct Sampler<'a> {
    weights: &'a [Vec<f64>],
    along_u: Vec<(usize, Vec<f64>)>,
    along_v: Vec<(usize, Vec<f64>)>,
}

impl<'a> Sampler<'a> {
    fn of(surface: &'a NurbsSurface) -> Self {
        let line = |degree: usize, knots: &[f64]| -> Vec<(usize, Vec<f64>)> {
            let space = Space {
                degree,
                knots: knots.to_vec(),
            };
            let mut found = Vec::new();
            for (a, b) in domain_spans(&space) {
                for k in 0..=SAMPLES_PER_SPAN {
                    found.push(space.basis_at(a + (b - a) * k as f64 / SAMPLES_PER_SPAN as f64));
                }
            }
            found
        };
        Self {
            weights: surface.weights(),
            along_u: line(surface.degree_u(), surface.knots_u()),
            along_v: line(surface.degree_v(), surface.knots_v()),
        }
    }

    fn farthest(&self, displacement: &[Vec<Vec3>]) -> f64 {
        let mut worst = 0.0_f64;
        for (first_u, values_u) in &self.along_u {
            for (first_v, values_v) in &self.along_v {
                let mut sum = Vec3::new(0.0, 0.0, 0.0);
                let mut total = 0.0;
                for (a, nu) in values_u.iter().enumerate() {
                    for (b, nv) in values_v.iter().enumerate() {
                        let (i, j) = (first_u + a, first_v + b);
                        let share = nu * nv * self.weights[i][j];
                        sum += displacement[i][j] * share;
                        total += share;
                    }
                }
                worst = worst.max(sum.length() / total);
            }
        }
        worst
    }
}

/// The linear system for the free control points of a net.
struct Fairing<'a> {
    mass: BandMatrix,
    stiffness: BandMatrix,
    /// The energy's gradient along each free point's normal, at the net as it
    /// stands.
    gradient: Vec<f64>,
    energy: f64,
    sampler: Sampler<'a>,
    free: &'a [(usize, usize)],
    normals: &'a [Vec3],
    shape: (usize, usize),
}

impl Fairing<'_> {
    fn step(&self, lambda: f64) -> Result<Step, OperationsError> {
        let factor = self.mass.factor_sum(&self.stiffness, lambda)?;
        let mut moves: Vec<f64> = self.gradient.iter().map(|g| *g * -lambda).collect();
        factor.solve_factored(&mut moves);
        let mut displacement = vec![vec![Vec3::new(0.0, 0.0, 0.0); self.shape.1]; self.shape.0];
        for (at, &(i, j)) in self.free.iter().enumerate() {
            displacement[i][j] = self.normals[at] * moves[at];
        }
        let distance = self.sampler.farthest(&displacement) * BETWEEN_SAMPLES;
        let crossed: f64 = moves.iter().zip(&self.gradient).map(|(m, g)| m * g).sum();
        let energy = self.energy + 2.0 * crossed + self.stiffness.quadratic(&moves);
        Ok(Step {
            moves,
            distance,
            energy,
        })
    }

    /// The stiffness of the softest shape the free points can make, by inverse
    /// iteration from the shape that moves them all alike.
    fn softest(&self) -> Result<f64, OperationsError> {
        let trace = |matrix: &BandMatrix| {
            (0..matrix.n)
                .map(|r| matrix.data[matrix.at(r, r)])
                .sum::<f64>()
        };
        let shift = 1e-10 * trace(&self.stiffness) / trace(&self.mass);
        let factor = self.stiffness.factor_sum(&self.mass, shift)?;
        let mut shape = vec![1.0; self.mass.n];
        let mut stiffness = 0.0;
        for _ in 0..16 {
            let mut next = self.mass.times(&shape);
            factor.solve_factored(&mut next);
            let size = next.iter().map(|x| x * x).sum::<f64>().sqrt();
            if size <= 0.0 {
                return Ok(0.0);
            }
            shape = next.iter().map(|x| x / size).collect();
            stiffness = self.stiffness.quadratic(&shape) / self.mass.quadratic(&shape);
        }
        Ok(stiffness)
    }

    /// The step that takes out what is short against the surface, cut back to
    /// fit `budget`.
    fn choose(&self, budget: f64) -> Result<Option<Step>, OperationsError> {
        let softest = self.softest()?;
        if self.energy <= 0.0 || softest <= 0.0 {
            return Ok(None);
        }
        let lambda = 1.0 / (HALF_LOST_AT * softest);
        let mut step = self.step(lambda)?;
        if step.distance > budget {
            let (mut low, mut high) = (0.0, lambda);
            let mut fits = None;
            for _ in 0..24 {
                let middle = if low == 0.0 {
                    0.25 * high
                } else {
                    (low * high).sqrt()
                };
                let inner = self.step(middle)?;
                if inner.distance <= budget {
                    low = middle;
                    fits = Some(inner);
                } else {
                    high = middle;
                }
            }
            let Some(fitted) = fits else {
                return Ok(None);
            };
            step = fitted;
        }
        Ok((step.distance > NO_CHANGE * budget && step.energy < self.energy).then_some(step))
    }
}

/// `surface` with its free control points faired, or `None` if there is
/// nothing to gain.
fn fair_net(
    surface: &NurbsSurface,
    held: &[Vec<bool>],
    budget: f64,
) -> Result<Option<NurbsSurface>, OperationsError> {
    let net = surface.control_points();
    let (n, m) = (net.len(), net[0].len());
    let (pu, pv) = (surface.degree_u(), surface.degree_v());

    // A point with no normal at its Greville point is at a pole, and stays.
    let sites_u = Space {
        degree: pu,
        knots: surface.knots_u().to_vec(),
    }
    .sites();
    let sites_v = Space {
        degree: pv,
        knots: surface.knots_v().to_vec(),
    }
    .sites();
    let normal_at = |i: usize, j: usize| surface.normal(sites_u[i], sites_v[j]).ok();

    // The band is narrowest numbered along the shorter side first.
    let along_v_first = m <= n;
    let mut index = vec![vec![usize::MAX; m]; n];
    let mut free = Vec::new();
    let mut normals = Vec::new();
    let mut consider = |i: usize, j: usize, slot: &mut usize| {
        if !held[i][j]
            && let Some(normal) = normal_at(i, j)
        {
            *slot = free.len();
            free.push((i, j));
            normals.push(normal);
        }
    };
    if along_v_first {
        for (i, column) in index.iter_mut().enumerate() {
            for (j, slot) in column.iter_mut().enumerate() {
                consider(i, j, slot);
            }
        }
    } else {
        for j in 0..m {
            for (i, column) in index.iter_mut().enumerate() {
                consider(i, j, &mut column[j]);
            }
        }
    }
    if free.is_empty() {
        return Ok(None);
    }

    let chord_mean = |lengths: &mut dyn Iterator<Item = f64>, count: usize| {
        (lengths.sum::<f64>() / count as f64).max(1e-12)
    };
    let length_u = chord_mean(
        &mut (0..m).map(|j| (1..n).map(|i| (net[i][j] - net[i - 1][j]).length()).sum()),
        m,
    );
    let length_v = chord_mean(
        &mut (0..n).map(|i| (1..m).map(|j| (net[i][j] - net[i][j - 1]).length()).sum()),
        n,
    );

    let mut width = 0;
    for &(i, j) in &free {
        let at = index[i][j];
        for k in i.saturating_sub(pu)..=(i + pu).min(n - 1) {
            for l in j.saturating_sub(pv)..=(j + pv).min(m - 1) {
                let other = index[k][l];
                if other != usize::MAX && other < at {
                    width = width.max(at - other);
                }
            }
        }
    }

    // Derivatives in physical length along the net: `S_xx = S_uu (du / L_u)^2`.
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let (to_x, to_y) = ((u1 - u0) / length_u, (v1 - v0) / length_v);
    let (kuu, kuv, kvv) = (to_x.powi(4), 2.0 * to_x * to_x * to_y * to_y, to_y.powi(4));
    let extra = if surface.is_rational() { 2 } else { 0 };
    let lines_u = gauss_lines(pu, surface.knots_u(), extra);
    let lines_v = gauss_lines(pv, surface.knots_v(), extra);
    let weights = surface.weights();

    let mut mass = BandMatrix::zeros(free.len(), width);
    let mut stiffness = BandMatrix::zeros(free.len(), width);
    let mut gradient = vec![0.0; free.len()];
    let mut energy = 0.0;
    let mut active = Vec::with_capacity((pu + 1) * (pv + 1));
    let origin = Point3::new(0.0, 0.0, 0.0);
    for along_u in &lines_u {
        for along_v in &lines_v {
            let weight = along_u.weight * along_v.weight;
            shapes_at(weights, along_u, along_v, &mut active);
            let (mut suu, mut suv, mut svv) = (
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
            );
            for &(i, j, shape) in &active {
                let p = net[i][j] - origin;
                suu += p * shape.ruu;
                suv += p * shape.ruv;
                svv += p * shape.rvv;
            }
            energy += weight * (kuu * suu.dot(suu) + kuv * suv.dot(suv) + kvv * svv.dot(svv));
            for &(i, j, a) in &active {
                let row = index[i][j];
                if row == usize::MAX {
                    continue;
                }
                let pull = suu * (kuu * a.ruu) + suv * (kuv * a.ruv) + svv * (kvv * a.rvv);
                gradient[row] += weight * pull.dot(normals[row]);
                for &(k, l, b) in &active {
                    let column = index[k][l];
                    if column == usize::MAX || column > row {
                        continue;
                    }
                    let slot = stiffness.at(row, column);
                    let aligned = weight * normals[row].dot(normals[column]);
                    stiffness.data[slot] +=
                        aligned * (kuu * a.ruu * b.ruu + kuv * a.ruv * b.ruv + kvv * a.rvv * b.rvv);
                    mass.data[slot] += aligned * a.r * b.r;
                }
            }
        }
    }
    let fairing = Fairing {
        mass,
        stiffness,
        gradient,
        energy,
        sampler: Sampler::of(surface),
        free: &free,
        normals: &normals,
        shape: (n, m),
    };
    let Some(step) = fairing.choose(budget)? else {
        return Ok(None);
    };

    let mut moved: Vec<Vec<Point3>> = net.to_vec();
    for (at, &(i, j)) in free.iter().enumerate() {
        moved[i][j] = net[i][j] + normals[at] * step.moves[at];
    }
    Ok(Some(NurbsSurface::new(
        pu,
        pv,
        surface.knots_u().to_vec(),
        surface.knots_v().to_vec(),
        moved,
        surface.weights().to_vec(),
    )?))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn a_band_factor_solves_its_system() {
        let mut mass = BandMatrix::zeros(4, 1);
        let mut other = BandMatrix::zeros(4, 1);
        for row in 0..4 {
            let slot = mass.at(row, row);
            mass.data[slot] = 2.0;
            if row > 0 {
                let slot = other.at(row, row - 1);
                other.data[slot] = -1.0;
            }
        }
        let factor = mass.factor_sum(&other, 0.5).unwrap();
        let mut values = vec![1.0; 4];
        factor.solve_factored(&mut values);
        let residual = |row: usize| {
            let left = if row > 0 { values[row - 1] } else { 0.0 };
            let right = if row < 3 { values[row + 1] } else { 0.0 };
            2.0 * values[row] - 0.5 * left - 0.5 * right - 1.0
        };
        for row in 0..4 {
            assert!(residual(row).abs() < 1e-12);
        }
    }

    #[test]
    fn rational_shapes_sum_to_one_and_their_derivatives_to_nothing() {
        let (knots_u, knots_v) = (
            vec![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        );
        let weights: Vec<Vec<f64>> = (0..4)
            .map(|i| (0..4).map(|j| 1.0 + 0.3 * f64::from(i * j)).collect())
            .collect();
        let (lines_u, lines_v) = (gauss_lines(2, &knots_u, 2), gauss_lines(3, &knots_v, 2));
        let mut active = Vec::new();
        for along_u in &lines_u {
            for along_v in &lines_v {
                shapes_at(&weights, along_u, along_v, &mut active);
                let mut sum = Shape::default();
                for (_, _, shape) in &active {
                    sum.r += shape.r;
                    sum.ru += shape.ru;
                    sum.rv += shape.rv;
                    sum.ruu += shape.ruu;
                    sum.ruv += shape.ruv;
                    sum.rvv += shape.rvv;
                }
                assert!((sum.r - 1.0).abs() < 1e-12);
                for rest in [sum.ru, sum.rv, sum.ruu, sum.ruv, sum.rvv] {
                    assert!(rest.abs() < 1e-10, "{rest}");
                }
            }
        }
    }
}

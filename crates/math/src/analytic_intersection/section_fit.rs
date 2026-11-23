//! Cubic fits of a section of two surfaces whose distance from each surface
//! is bounded along their whole length, not only at their points.
//!
//! On one knot span the fit is a cubic Bezier `C(t)`. Each surface here has
//! an implicit form `F` that is a polynomial in the point (a quadric, or a
//! quartic for a torus), so `F(C(t))` is a polynomial in `t` whose Bernstein
//! coefficients come from products of the control points. The polynomial lies
//! between its least and greatest coefficients on the span, and `F` is a
//! length no less than the distance from the surface times a factor with a
//! lower bound on the span, so the two give an upper bound of the distance.
//! The bound closes on the true distance as the span is halved.

use super::AnalyticSurface;
use crate::MathError;
use crate::madd::Madd;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::decompose::curve_bezier_between;
use crate::nurbs::fitting::{chord_length_params, interpolate};
use crate::vec::{Point3, Vec3};

/// `CHOOSE[n][k]` is the binomial coefficient, for `n` up to 12.
const CHOOSE: [[f64; 13]; 13] = pascal();

const fn pascal() -> [[f64; 13]; 13] {
    let mut table = [[0.0; 13]; 13];
    let mut n = 0;
    while n < 13 {
        table[n][0] = 1.0;
        let mut k = 1;
        while k <= n {
            table[n][k] = table[n - 1][k - 1] + table[n - 1][k];
            k += 1;
        }
        n += 1;
    }
    table
}

/// How many times a span of the fit is halved before its bound counts.
const HALVINGS: usize = 3;

/// The most passes the fit makes, and the most points it takes.
const PASSES: usize = 24;
const MOST_POINTS: usize = 8192;

/// A surface that a section lies on, as the fit reads it.
pub(super) trait Carrier {
    /// A signed distance from `p` to the surface, exact near it, with its
    /// unit gradient; `None` where the gradient is not defined.
    fn distance(&self, p: Point3) -> Option<(f64, Vec3)>;

    /// An upper bound of the distance from the surface of every point of the
    /// cubic Bezier `ctrl`, or `None` where the factor has no positive lower
    /// bound on it.
    fn bound(&self, ctrl: &[Point3; 4]) -> Option<f64>;

    /// A length that sizes the rounding of the distance near the surface.
    fn reach(&self) -> f64;
}

/// The Bernstein coefficients of the product of the polynomials with
/// Bernstein coefficients `p` and `q`, written to `out`.
fn product(p: &[f64], q: &[f64], out: &mut [f64]) {
    let (m, n) = (p.len() - 1, q.len() - 1);
    out.fill(0.0);
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            let share = CHOOSE[m][i] * CHOOSE[n][j] / CHOOSE[m + n][i + j];
            out[i + j] = (share * pi).madd(qj, out[i + j]);
        }
    }
}

/// The degree-6 Bernstein coefficients of `a(t) · b(t)` for two cubics.
fn dot(a: &[Vec3; 4], b: &[Vec3; 4]) -> [f64; 7] {
    let mut out = [0.0; 7];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            let share = CHOOSE[3][i] * CHOOSE[3][j] / CHOOSE[6][i + j];
            out[i + j] = share.madd(ai.dot(*bj), out[i + j]);
        }
    }
    out
}

fn least(coefs: &[f64]) -> f64 {
    coefs.iter().copied().fold(f64::INFINITY, f64::min)
}

fn largest(coefs: &[f64]) -> f64 {
    coefs.iter().fold(0.0, |m: f64, c| m.max(c.abs()))
}

/// Each control point less `o`, and that less its part along `axis`, with
/// the largest length of the first.
fn about_axis(ctrl: &[Point3; 4], o: Point3, axis: Vec3) -> ([Vec3; 4], [f64; 4], f64) {
    let w = ctrl.map(|p| p - o);
    let h = w.map(|v| axis.dot(v));
    let across = std::array::from_fn(|i| w[i] - axis * h[i]);
    let far = w.iter().fold(0.0, |m: f64, v| m.max(v.length()));
    (across, h, far)
}

/// Rounding in the Bezier points read off the fit, and in each point less
/// the surface's origin.
fn point_rounding(ctrl: &[Point3; 4], o: Point3) -> f64 {
    let size = ctrl.iter().fold(0.0, |m: f64, p| {
        m.max(p.x().abs()).max(p.y().abs()).max(p.z().abs())
    });
    let o_size = o.x().abs().max(o.y().abs()).max(o.z().abs());
    32.0 * f64::EPSILON * (size + o_size)
}

impl Carrier for AnalyticSurface<'_> {
    fn distance(&self, p: Point3) -> Option<(f64, Vec3)> {
        match self {
            Self::Cylinder(c) => {
                let w = p - c.origin();
                let across = w - c.axis() * c.axis().dot(w);
                let rho = across.length();
                (rho > 0.0).then(|| (rho - c.radius(), across * (1.0 / rho)))
            }
            Self::Sphere(s) => {
                let w = p - s.center();
                let r = w.length();
                (r > 0.0).then(|| (r - s.radius(), w * (1.0 / r)))
            }
            // The distance from the ruling through `p`'s side of the axis,
            // which lies on the cone and is the nearest ruling near it.
            Self::Cone(c) => {
                let (sin, cos) = c.half_angle().sin_cos();
                let w = p - c.apex();
                let h = c.axis().dot(w);
                let across = w - c.axis() * h;
                let rho = across.length();
                (rho > 0.0).then(|| {
                    let out = across * (1.0 / rho);
                    (rho.madd(sin, -h * cos), out * sin - c.axis() * cos)
                })
            }
            // The distance from the tube about the centre circle's point on
            // `p`'s side of the axis, or on the far side where that tube is
            // nearer (a spindle torus's inner sheet).
            Self::Torus(t) => {
                let w = p - t.center();
                let h = t.z_axis().dot(w);
                let across = w - t.z_axis() * h;
                let rho = across.length();
                if rho <= 0.0 {
                    return None;
                }
                let tube = |side: f64| {
                    let to = w - across * (side * t.major_radius() / rho);
                    let q = to.length();
                    (q - t.minor_radius(), to, q)
                };
                let (near, far) = (tube(1.0), tube(-1.0));
                let (f, to, q) = if far.0.abs() < near.0.abs() {
                    far
                } else {
                    near
                };
                (q > 0.0).then(|| (f, to * (1.0 / q)))
            }
        }
    }

    fn bound(&self, ctrl: &[Point3; 4]) -> Option<f64> {
        let eps = f64::EPSILON;
        match self {
            // F = ρ² − R² = (ρ − R)(ρ + R).
            Self::Cylinder(c) => {
                let o = c.origin();
                let (across, _, far) = about_axis(ctrl, o, c.axis());
                let r = c.radius();
                let rho2 = dot(&across, &across);
                let f = rho2.map(|x| r.madd(-r, x));
                let floor = r + least(&rho2).max(0.0).sqrt();
                let wide = largest(&rho2).sqrt();
                let slack = 64.0 * eps * wide.madd(wide + far, r * r);
                (floor > 0.0).then(|| (largest(&f) + slack) / floor + point_rounding(ctrl, o))
            }
            // F = |w|² − R² = (r − R)(r + R).
            Self::Sphere(s) => {
                let o = s.center();
                let w = ctrl.map(|p| p - o);
                let r = s.radius();
                let r2 = dot(&w, &w);
                let f = r2.map(|x| r.madd(-r, x));
                let floor = r + least(&r2).max(0.0).sqrt();
                let slack = 64.0 * eps * r.madd(r, largest(&r2));
                (floor > 0.0).then(|| (largest(&f) + slack) / floor + point_rounding(ctrl, o))
            }
            // F = h² cos² − ρ² sin² = (h cos − ρ sin)(h cos + ρ sin): the
            // first factor is the distance from the ruling on `p`'s side.
            Self::Cone(c) => {
                let o = c.apex();
                let (sin, cos) = c.half_angle().sin_cos();
                let (across, h, far) = about_axis(ctrl, o, c.axis());
                let rho2 = dot(&across, &across);
                let mut h2 = [0.0; 7];
                product(&h, &h, &mut h2);
                let f: [f64; 7] =
                    std::array::from_fn(|k| (cos * cos).madd(h2[k], -(sin * sin) * rho2[k]));
                let floor = cos.madd(least(&h), sin * least(&rho2).max(0.0).sqrt());
                let slack = 128.0 * eps * far * far;
                (floor > 0.0).then(|| (largest(&f) + slack) / floor + point_rounding(ctrl, o))
            }
            // F = G² − 4R²ρ² with G = |w|² + R² − r², which is
            // (q − r)(q + r)(G + 2Rρ) for q the distance from the centre
            // circle's point on `p`'s side, and (q' − r)(q' + r)(G − 2Rρ)
            // for q' the distance from the far side's: off the far tube
            // `G + 2Rρ` is positive, and inside the near one `G − 2Rρ` is
            // negative.
            Self::Torus(t) => {
                let o = t.center();
                let (big, small) = (t.major_radius(), t.minor_radius());
                let (across, h, far) = about_axis(ctrl, o, t.z_axis());
                let rho2 = dot(&across, &across);
                let mut h2 = [0.0; 7];
                product(&h, &h, &mut h2);
                let shift = big.madd(big, -small * small);
                let g: [f64; 7] = std::array::from_fn(|k| rho2[k] + h2[k] + shift);
                let mut g2 = [0.0; 13];
                product(&g, &g, &mut g2);
                let mut rho2_up = [0.0; 13];
                product(&rho2, &[1.0; 7], &mut rho2_up);
                let f: [f64; 13] =
                    std::array::from_fn(|k| (-4.0 * big * big).madd(rho2_up[k], g2[k]));
                let rho_least = least(&rho2).max(0.0).sqrt();
                let greatest = g.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let floor = small
                    * (2.0 * big)
                        .madd(rho_least, least(&g))
                        .max((2.0 * big).madd(rho_least, -greatest));
                let size = far.madd(far, big.madd(big, small * small));
                let slack = 512.0 * eps * size * size;
                (floor > 0.0).then(|| (largest(&f) + slack) / floor + point_rounding(ctrl, o))
            }
        }
    }

    fn reach(&self) -> f64 {
        let size = |p: Point3| p.x().abs().max(p.y().abs()).max(p.z().abs());
        match self {
            Self::Cylinder(c) => size(c.origin()) + c.radius(),
            Self::Sphere(s) => size(s.center()) + s.radius(),
            Self::Cone(c) => size(c.apex()),
            Self::Torus(t) => size(t.center()) + t.major_radius() + t.minor_radius(),
        }
    }
}

/// The point of the section of `a` and `b` that `p`, a point near it, moves
/// to: Newton steps on both distances, each the shortest move that clears
/// both to first order, so the point moves across the section and not along
/// it. `None` where the two surfaces meet at a tangent (their normals within
/// about 1e-6 of parallel), where a distance has no gradient, or where the
/// steps do not settle.
pub(super) fn settle_on_section(a: &impl Carrier, b: &impl Carrier, p: Point3) -> Option<Point3> {
    let size = |q: Point3| q.x().abs().max(q.y().abs()).max(q.z().abs());
    let rounding = 64.0 * f64::EPSILON * (1.0 + size(p) + a.reach() + b.reach());
    let mut x = p;
    for _ in 0..32 {
        let (fa, ga) = a.distance(x)?;
        let (fb, gb) = b.distance(x)?;
        if !(fa.is_finite() && fb.is_finite()) {
            return None;
        }
        if fa.abs() <= rounding && fb.abs() <= rounding {
            return Some(x);
        }
        let c = ga.dot(gb);
        let det = c.madd(-c, 1.0);
        if det < 1e-12 {
            return None;
        }
        let along_a = c.madd(-fb, fa) / det;
        let along_b = c.madd(-fa, fb) / det;
        x = x - ga * along_a - gb * along_b;
    }
    let (fa, _) = a.distance(x)?;
    let (fb, _) = b.distance(x)?;
    (fa.abs() <= 1e3 * rounding && fb.abs() <= 1e3 * rounding).then_some(x)
}

/// The cubic Bezier from `t0` to `t1` of `curve`, one of its knot spans, a
/// line or a parabola raised to a cubic.
fn span_bezier(curve: &NurbsCurve, t0: f64, t1: f64) -> Result<[Point3; 4], MathError> {
    let mut points = Vec::with_capacity(4);
    for q in curve_bezier_between(curve, t0, t1)? {
        if q[3] <= 0.0 {
            return Err(MathError::PoleAtInfinity { start: t0, end: t1 });
        }
        points.push(Point3::new(q[0] / q[3], q[1] / q[3], q[2] / q[3]));
    }
    let at = |p: Point3, q: Point3, s: f64| p + (q - p) * s;
    match points[..] {
        [p0, p1] => Ok([p0, at(p0, p1, 1.0 / 3.0), at(p0, p1, 2.0 / 3.0), p1]),
        [p0, p1, p2] => Ok([p0, at(p0, p1, 2.0 / 3.0), at(p2, p1, 2.0 / 3.0), p2]),
        [p0, p1, p2, p3] => Ok([p0, p1, p2, p3]),
        _ => Err(MathError::EmptyInput),
    }
}

/// The two halves of a cubic Bezier.
fn halves(c: &[Point3; 4]) -> ([Point3; 4], [Point3; 4]) {
    let mid = |p: Point3, q: Point3| p + (q - p) * 0.5;
    let (a, b, d) = (mid(c[0], c[1]), mid(c[1], c[2]), mid(c[2], c[3]));
    let (e, f) = (mid(a, b), mid(b, d));
    let g = mid(e, f);
    ([c[0], a, e, g], [g, f, d, c[3]])
}

/// The greater of the two surfaces' bounds on `ctrl`, the halves taken
/// `depth` more times while it exceeds `tol`; infinite where either has none.
fn piece_bound(
    a: &impl Carrier,
    b: &impl Carrier,
    ctrl: &[Point3; 4],
    tol: f64,
    depth: usize,
) -> f64 {
    let here = match (a.bound(ctrl), b.bound(ctrl)) {
        (Some(x), Some(y)) if x.is_finite() && y.is_finite() => x.max(y),
        _ => f64::INFINITY,
    };
    if here <= tol || depth == 0 {
        return here;
    }
    let (left, right) = halves(ctrl);
    piece_bound(a, b, &left, tol, depth - 1).max(piece_bound(a, b, &right, tol, depth - 1))
}

/// A cubic interpolant through `points`, points of the section of `a` and
/// `b` in order along it, and through more points of the section where it
/// needs them, with the points it passes through.
///
/// Each pass bounds, on each knot span, how far the cubic strays from each
/// surface. Where a bound exceeds `tol`, the points either side of the span
/// get points of the section between them, as many as the bound asks for at
/// the fourth power of their spacing, each moved onto the section from the
/// cubic.
///
/// # Errors
///
/// Returns an error if the interpolation fails, if a point the fit needs
/// does not move onto the section, or moves by more than half its spacing
/// from the cubic (the cubic left the section there), or if the fit needs
/// more than 8192 points or 24 passes.
pub(super) fn fit_section(
    a: &impl Carrier,
    b: &impl Carrier,
    points: &[Point3],
    tol: f64,
) -> Result<(NurbsCurve, Vec<Point3>), MathError> {
    let mut pts = points.to_vec();
    for pass in 1..=PASSES {
        let curve = interpolate(&pts, 3.min(pts.len().saturating_sub(1)))?;
        let params = chord_length_params(&pts);
        let mut cuts: Vec<f64> = curve.knots().to_vec();
        cuts.dedup_by(|x, y| (*x - *y).abs() <= 0.0);
        let mut pieces = vec![1_usize; pts.len() - 1];
        let mut first = 0;
        for span in cuts.windows(2) {
            let ctrl = span_bezier(&curve, span[0], span[1])?;
            let worst = piece_bound(a, b, &ctrl, tol, HALVINGS);
            if worst <= tol {
                continue;
            }
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let more = if worst.is_finite() {
                ((worst / tol).sqrt().sqrt().ceil() as usize).clamp(2, 8)
            } else {
                2
            };
            while first + 1 < params.len() && params[first + 1] <= span[0] {
                first += 1;
            }
            let mut i = first;
            while i + 1 < params.len() && params[i] < span[1] {
                pieces[i] = pieces[i].max(more);
                i += 1;
            }
        }
        if pieces.iter().all(|&k| k == 1) {
            return Ok((curve, pts));
        }
        let mut finer = Vec::with_capacity(pts.len() + pieces.iter().sum::<usize>());
        for (i, &k) in pieces.iter().enumerate() {
            finer.push(pts[i]);
            let gap = (pts[i + 1] - pts[i]).length();
            for j in 1..k {
                #[allow(clippy::cast_precision_loss)]
                let t = (params[i + 1] - params[i]).madd(j as f64 / k as f64, params[i]);
                let on_fit = curve.evaluate(t);
                let on = settle_on_section(a, b, on_fit)
                    .filter(|on| (*on - on_fit).length() < 0.5 * gap)
                    .ok_or(MathError::ConvergenceFailure { iterations: pass })?;
                finer.push(on);
            }
        }
        if let Some(&last) = pts.last() {
            finer.push(last);
        }
        if finer.len() > MOST_POINTS {
            return Err(MathError::ConvergenceFailure { iterations: pass });
        }
        pts = finer;
    }
    Err(MathError::ConvergenceFailure { iterations: PASSES })
}

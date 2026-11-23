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

/// `SHARES_3[i][j]` and `SHARES_6[i][j]`: the share of the product of
/// Bernstein terms `i` and `j` of degree 3 (or 6) in term `i + j` of twice
/// the degree, `C(n, i) C(n, j) / C(2n, i + j)`.
const SHARES_3: [[f64; 4]; 4] = shares::<4>();
const SHARES_6: [[f64; 7]; 7] = shares::<7>();

const fn shares<const N: usize>() -> [[f64; N]; N] {
    let n = N - 1;
    let mut table = [[0.0; N]; N];
    let mut i = 0;
    while i < N {
        let mut j = 0;
        while j < N {
            table[i][j] = CHOOSE[n][i] * CHOOSE[n][j] / CHOOSE[2 * n][i + j];
            j += 1;
        }
        i += 1;
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

    /// A length that sizes the rounding of the distance at `p`: the
    /// surface's own size and `p`'s distance from its frame, or for a
    /// cylinder, whose frame can sit anywhere along its axis, from its axis.
    fn scale(&self, p: Point3) -> f64;

    /// The second derivative of the signed distance of [`Self::distance`]
    /// along the unit vector `v` at `p`: on the surface, with `v` tangent
    /// to it, its curvature along `v`, positive where it bends away from
    /// the gradient. `None` where the distance has no second derivative: on
    /// an axis, at a centre, or on a cone's apex plane.
    fn bend(&self, p: Point3, v: Vec3) -> Option<f64>;

    /// Which sheet of the surface [`Self::distance`] reads at `p`: a cone's
    /// nappe or a torus's tube on `p`'s side of the axis or the far one, as
    /// 1 or -1; 0 for a surface of one sheet.
    fn sheet(&self, p: Point3) -> i8;
}

/// The degree-6 Bernstein coefficients of the product of two cubics.
fn product_3(p: &[f64; 4], q: &[f64; 4]) -> [f64; 7] {
    let mut out = [0.0; 7];
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            out[i + j] = (SHARES_3[i][j] * pi).madd(qj, out[i + j]);
        }
    }
    out
}

/// The degree-12 Bernstein coefficients of the product of two sextics.
fn product_6(p: &[f64; 7], q: &[f64; 7]) -> [f64; 13] {
    let mut out = [0.0; 13];
    for (i, &pi) in p.iter().enumerate() {
        for (j, &qj) in q.iter().enumerate() {
            out[i + j] = (SHARES_6[i][j] * pi).madd(qj, out[i + j]);
        }
    }
    out
}

/// The degree-6 Bernstein coefficients of `a(t) · b(t)` for two cubics.
fn dot(a: &[Vec3; 4], b: &[Vec3; 4]) -> [f64; 7] {
    let mut out = [0.0; 7];
    for (i, ai) in a.iter().enumerate() {
        for (j, bj) in b.iter().enumerate() {
            out[i + j] = SHARES_3[i][j].madd(ai.dot(*bj), out[i + j]);
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
            // The distance from the ruling on `p`'s side of the axis and on
            // `p`'s nappe, which lies on the cone and is the nearest ruling
            // near it.
            Self::Cone(c) => {
                let (sin, cos) = c.half_angle().sin_cos();
                let w = p - c.apex();
                let h = c.axis().dot(w);
                let across = w - c.axis() * h;
                let rho = across.length();
                let nappe = if h < 0.0 { -1.0 } else { 1.0 };
                (rho > 0.0).then(|| {
                    let out = across * (1.0 / rho);
                    (
                        rho.madd(sin, -h.abs() * cos),
                        out * sin - c.axis() * (nappe * cos),
                    )
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
            // F = h² cos² − ρ² sin² = (|h| cos − ρ sin)(|h| cos + ρ sin): the
            // first factor is the distance from the ruling on `p`'s side and
            // nappe. `|h|` is no less than the least `|h_i|` where every
            // `h_i` has one sign, and no less than 0 where the piece may
            // cross the apex's plane; there the bound rests on `ρ` alone,
            // and a piece that may also reach the axis, through the apex,
            // has none.
            Self::Cone(c) => {
                let o = c.apex();
                let (sin, cos) = c.half_angle().sin_cos();
                let (across, h, far) = about_axis(ctrl, o, c.axis());
                let rho2 = dot(&across, &across);
                let h2 = product_3(&h, &h);
                let f: [f64; 7] =
                    std::array::from_fn(|k| (cos * cos).madd(h2[k], -(sin * sin) * rho2[k]));
                let h_least = if h.iter().all(|&x| x > 0.0) {
                    least(&h)
                } else if h.iter().all(|&x| x < 0.0) {
                    least(&h.map(|x| -x))
                } else {
                    0.0
                };
                let floor = cos.madd(h_least, sin * least(&rho2).max(0.0).sqrt());
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
                let h2 = product_3(&h, &h);
                let shift = big.madd(big, -small * small);
                let g: [f64; 7] = std::array::from_fn(|k| rho2[k] + h2[k] + shift);
                let g2 = product_6(&g, &g);
                let rho2_up = product_6(&rho2, &[1.0; 7]);
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

    // The distance from an axis is `rho`, whose second derivative along `v`
    // is the square of `v`'s part across the axis and across the point's
    // direction from it, over `rho`. A sphere's is the same with its centre
    // for the axis; a cone's distance is `rho sin` less a term linear in the
    // point. A torus's is the distance `q` from a point of its centre
    // circle, which moves with the point by `R / rho` times the part across
    // the axis and across the point's direction.
    fn bend(&self, p: Point3, v: Vec3) -> Option<f64> {
        let across_of = |w: Vec3, axis: Vec3| w - axis * axis.dot(w);
        let turn = |v_across: Vec3, e: Vec3| {
            let along = v_across.dot(e);
            v_across.dot(v_across) - along * along
        };
        match self {
            Self::Cylinder(c) => {
                let across = across_of(p - c.origin(), c.axis());
                let rho = across.length();
                (rho > 0.0).then(|| turn(across_of(v, c.axis()), across * (1.0 / rho)) / rho)
            }
            Self::Sphere(s) => {
                let w = p - s.center();
                let r = w.length();
                (r > 0.0).then(|| turn(v, w * (1.0 / r)) / r)
            }
            // `|h|` has no second derivative across the apex's plane: there,
            // within the rounding of `h`, none is read.
            Self::Cone(c) => {
                let w = p - c.apex();
                let h = c.axis().dot(w);
                let across = w - c.axis() * h;
                let rho = across.length();
                (rho > 0.0 && h.abs() > 64.0 * f64::EPSILON * w.length()).then(|| {
                    c.half_angle().sin() * turn(across_of(v, c.axis()), across * (1.0 / rho)) / rho
                })
            }
            Self::Torus(t) => {
                let w = p - t.center();
                let across = across_of(w, t.z_axis());
                let rho = across.length();
                if rho <= 0.0 {
                    return None;
                }
                let tube = |side: f64| {
                    let to = w - across * (side * t.major_radius() / rho);
                    (to.length() - t.minor_radius(), to, side)
                };
                let (near, far) = (tube(1.0), tube(-1.0));
                let (_, to, side) = if far.0.abs() < near.0.abs() {
                    far
                } else {
                    near
                };
                let q = to.length();
                (q > 0.0).then(|| {
                    let centre = side * t.major_radius() / rho;
                    let spread = turn(across_of(v, t.z_axis()), across * (1.0 / rho));
                    centre.madd(-spread, turn(v, to * (1.0 / q))) / q
                })
            }
        }
    }

    fn sheet(&self, p: Point3) -> i8 {
        match self {
            Self::Cylinder(_) | Self::Sphere(_) => 0,
            Self::Cone(c) => {
                if c.axis().dot(p - c.apex()) < 0.0 {
                    -1
                } else {
                    1
                }
            }
            Self::Torus(t) => {
                let w = p - t.center();
                let across = w - t.z_axis() * t.z_axis().dot(w);
                let rho = across.length();
                if rho <= 0.0 {
                    return 1;
                }
                let tube = |side: f64| {
                    (w - across * (side * t.major_radius() / rho)).length() - t.minor_radius()
                };
                if tube(-1.0).abs() < tube(1.0).abs() {
                    -1
                } else {
                    1
                }
            }
        }
    }

    fn scale(&self, p: Point3) -> f64 {
        match self {
            Self::Cylinder(c) => {
                let w = p - c.origin();
                (w - c.axis() * c.axis().dot(w)).length() + c.radius()
            }
            Self::Sphere(s) => (p - s.center()).length() + s.radius(),
            Self::Cone(c) => (p - c.apex()).length(),
            Self::Torus(t) => (p - t.center()).length() + t.major_radius() + t.minor_radius(),
        }
    }
}

/// The point of the section of `a` and `b` that `p`, a point near it, moves
/// to: Newton steps on both distances, each the shortest move that clears
/// both to first order, so the point moves across the section and not along
/// it. `None` where the two surfaces meet at a tangent (their normals within
/// about 1e-6 of parallel), where a distance has no gradient, or where the
/// steps do not settle.
///
/// The steps settle when each part of a step along an axis is below the
/// spacing of floats in that coordinate of the point, below which no point
/// can be placed, plus the rounding of the distances, sized by the surfaces'
/// own sizes and the point's distance from their frames, over the sine of the
/// angle between the normals. A short step, not small distances, ends them:
/// where the normals lie near each other a step is the distances over that
/// sine, so distances within rounding can leave the point far off the
/// section, near a touch with no section at all. Each coordinate is held to
/// its own spacing: a far coordinate's coarse spacing does not hide a step
/// in a near one.
pub(super) fn settle_on_section(a: &impl Carrier, b: &impl Carrier, p: Point3) -> Option<Point3> {
    let mut x = p;
    for _ in 0..48 {
        let (fa, ga) = a.distance(x)?;
        let (fb, gb) = b.distance(x)?;
        if !(fa.is_finite() && fb.is_finite()) {
            return None;
        }
        let c = ga.dot(gb);
        let det = c.madd(-c, 1.0);
        if det < 1e-12 {
            return None;
        }
        let along_a = c.madd(-fb, fa) / det;
        let along_b = c.madd(-fa, fb) / det;
        let step = ga * along_a + gb * along_b;
        x = x - ga * along_a - gb * along_b;
        let surfaces = 64.0 * f64::EPSILON * (a.scale(x) + b.scale(x)) / det.sqrt();
        let settled = [(step.x(), x.x()), (step.y(), x.y()), (step.z(), x.z())]
            .into_iter()
            .all(|(part, at)| part.abs() <= f64::EPSILON.madd(at.abs(), surfaces));
        if settled {
            return Some(x);
        }
    }
    None
}

/// Whether `a` and `b` cross at `p`, a point on both: their normals there lie
/// more than about 1e-6 from parallel, as the Newton steps of
/// [`settle_on_section`] need.
pub(super) fn crosses_at(a: &impl Carrier, b: &impl Carrier, p: Point3) -> bool {
    match (a.distance(p), b.distance(p)) {
        (Some((_, ga)), Some((_, gb))) => {
            let c = ga.dot(gb);
            c.madd(-c, 1.0) >= 1e-12
        }
        _ => false,
    }
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

/// A share of the tolerance the first pass of a fit aims each span at, so
/// that its bound, which reads up to about twice the true miss, passes.
const AIM: f64 = 0.25;

/// Points of the section, from the first of `points` to their last, spaced
/// so that a cubic interpolant through them misses both surfaces by about
/// [`AIM`] of `tol` on every span. A cubic through `points` reads, a
/// quarter, half and three quarters of the way between each two, how far it
/// lies off either surface, `e`: the cubic's miss goes as the fourth power
/// of the spacing, so that stretch wants `(e / (AIM tol))^(1/4)` spans, and
/// no fewer than one, so the points are never sparser than `points`: a
/// face's boundary drawn on its surface from a curve of fewer spans read a
/// small dip's volume 4% low. The new points lie on that cubic at equal
/// steps of the spans so counted, each moved onto the section by Newton
/// steps. `None` where the cubic fails, the spacing would take more points
/// than the fit allows, or a point does not move onto the section within
/// half its spacing.
fn spaced(a: &impl Carrier, b: &impl Carrier, points: &[Point3], tol: f64) -> Option<Vec<Point3>> {
    let curve = interpolate(points, 3).ok()?;
    let params = chord_length_params(points);
    let off = |p: Point3| match (a.distance(p), b.distance(p)) {
        (Some((x, _)), Some((y, _))) => x.abs().max(y.abs()),
        _ => f64::INFINITY,
    };
    let want: Vec<f64> = params
        .windows(2)
        .map(|w| {
            let e = [0.25, 0.5, 0.75]
                .iter()
                .map(|&q| off(curve.evaluate((w[1] - w[0]).madd(q, w[0]))))
                .fold(0.0, f64::max);
            (e / (AIM * tol)).sqrt().sqrt().max(1.0)
        })
        .collect();
    let total: f64 = want.iter().sum();
    if !total.is_finite() {
        return None;
    }
    #[allow(clippy::cast_precision_loss)]
    if total.ceil() + 1.0 > MOST_POINTS as f64 {
        return None;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let n = (total.ceil() as usize).max(1);
    #[allow(clippy::cast_precision_loss)]
    let per = total / n as f64;
    let mut out = Vec::with_capacity(n + 1);
    out.push(points[0]);
    let (mut i, mut before) = (0, 0.0);
    for j in 1..n {
        #[allow(clippy::cast_precision_loss)]
        let level = per * j as f64;
        while i + 1 < want.len() && before + want[i] < level {
            before += want[i];
            i += 1;
        }
        let s = if want[i] > 0.0 {
            ((level - before) / want[i]).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let from = curve.evaluate((params[i + 1] - params[i]).madd(s, params[i]));
        let gap = (points[i + 1] - points[i]).length();
        let on = settle_on_section(a, b, from).filter(|on| (*on - from).length() < 0.5 * gap)?;
        if out.last().is_some_and(|&last| (on - last).length() > 0.0) {
            out.push(on);
        }
    }
    out.push(points[points.len() - 1]);
    Some(out)
}

/// A cubic interpolant through points of the section of `a` and `b`, from
/// the first of `points`, points of it in order along it, to their last,
/// and through more points of the section where it needs them, with the
/// points it passes through.
///
/// The first pass places the points by how far a cubic through `points`
/// lies off the surfaces between each two (see [`spaced`]), so that most
/// spans pass at once. Each pass bounds, on each knot span, how far the
/// cubic strays from each surface. Where a bound exceeds `tol`, the points
/// either side of the span get points of the section between them, as many
/// as the bound asks for at the fourth power of their spacing, each moved
/// onto the section from the cubic.
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
    let mut pts = if points.len() > 4 {
        spaced(a, b, points, tol).unwrap_or_else(|| points.to_vec())
    } else {
        points.to_vec()
    };
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
                // The cubic's point, or where it strays too far for Newton
                // steps to bring it back (a fit through points far apart
                // on a thin loop), the chord's point between the two
                // points of the section either side.
                #[allow(clippy::cast_precision_loss)]
                let chord = pts[i] + (pts[i + 1] - pts[i]) * (j as f64 / k as f64);
                let on = [curve.evaluate(t), chord]
                    .into_iter()
                    .find_map(|from| {
                        settle_on_section(a, b, from).filter(|on| (*on - from).length() < 0.5 * gap)
                    })
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::mat::Mat4;
    use crate::surfaces::{CylindricalSurface, ToroidalSurface};

    /// How many knot spans of a cubic through `points` the bound reads as
    /// more than `tol` off either surface, and how many spans there are.
    fn unproved(a: &impl Carrier, b: &impl Carrier, points: &[Point3], tol: f64) -> (usize, usize) {
        let curve = interpolate(points, 3).unwrap();
        let mut cuts: Vec<f64> = curve.knots().to_vec();
        cuts.dedup_by(|x, y| (*x - *y).abs() <= 0.0);
        let failing = cuts
            .windows(2)
            .filter(|s| {
                piece_bound(
                    a,
                    b,
                    &span_bezier(&curve, s[0], s[1]).unwrap(),
                    tol,
                    HALVINGS,
                ) > tol
            })
            .count();
        (failing, cuts.len() - 1)
    }

    /// The ring of radii 4 and 1.5 and a rod of radius 0.6 along y through
    /// (0.5, y, 1), in three poses, meet in loops that a march samples about
    /// 0.12 apart. A cubic through those samples misses the surfaces by more
    /// than 1e-7 on nearly every span, and the fit halved them pass by pass,
    /// three passes to about 250 points. The first pass now places its
    /// points by how far that cubic misses, and all but a few spans of the
    /// cubic through them pass at once; the fit still stays within 1e-7.
    #[test]
    fn a_fits_first_points_are_placed_so_most_spans_pass() {
        let tol = 1e-7;
        let poses = [
            Mat4::identity(),
            Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9),
            Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4),
        ];
        for (k, pose) in poses.iter().enumerate() {
            let o = Point3::new(0.0, 0.0, 0.0);
            let along = |v: Vec3| pose.mul_point(o + v) - pose.mul_point(o);
            let ring = ToroidalSurface::with_axis(
                pose.mul_point(o),
                4.0,
                1.5,
                along(Vec3::new(0.0, 0.0, 1.0)),
            )
            .unwrap();
            let rod = CylindricalSurface::new(
                pose.mul_point(Point3::new(0.5, 0.0, 1.0)),
                along(Vec3::new(0.0, 1.0, 0.0)),
                0.6,
            )
            .unwrap();
            let (a, b) = (
                AnalyticSurface::Cylinder(&rod),
                AnalyticSurface::Torus(&ring),
            );
            let tangent = |p: Point3, toward: Vec3| {
                let t = a.distance(p).unwrap().1.cross(b.distance(p).unwrap().1);
                let t = t * (1.0 / t.length());
                if t.dot(toward) < 0.0 { -t } else { t }
            };
            let mut p =
                settle_on_section(&a, &b, pose.mul_point(Point3::new(0.5, 3.0, 1.6))).unwrap();
            let mut t = tangent(p, along(Vec3::new(1.0, 0.0, 0.0)));
            let mut points = vec![p];
            for _ in 0..40 {
                p = settle_on_section(&a, &b, p + t * 0.12).unwrap();
                t = tangent(p, t);
                points.push(p);
            }
            let (raw, spans) = unproved(&a, &b, &points, tol);
            assert!(
                2 * raw > spans,
                "pose {k}: {raw} of {spans} spans fail on the samples"
            );
            let placed = spaced(&a, &b, &points, tol).unwrap();
            let (failing, spans) = unproved(&a, &b, &placed, tol);
            assert!(
                20 * failing <= spans,
                "pose {k}: {failing} of {spans} spans fail"
            );

            let (curve, _) = fit_section(&a, &b, &points, tol).unwrap();
            let (t0, t1) = curve.domain();
            let worst = (0..=4000)
                .map(|s| curve.evaluate((t1 - t0).madd(f64::from(s) / 4000.0, t0)))
                .map(|q| {
                    a.distance(q)
                        .unwrap()
                        .0
                        .abs()
                        .max(b.distance(q).unwrap().0.abs())
                })
                .fold(0.0, f64::max);
            assert!(worst <= tol, "pose {k}: off by {worst:.2e}");

            // At a tolerance no fit reaches, the first pass would ask for
            // millions of points: it declines before it places any, and the
            // fit then refuses within its own limit.
            assert!(spaced(&a, &b, &points, 1e-30).is_none(), "pose {k}");
            assert!(fit_section(&a, &b, &points, 1e-30).is_err(), "pose {k}");
        }
    }
}

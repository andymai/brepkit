//! Exact integrals over a face bounded on a surface of revolution, taken
//! along its wires' own curves by Green's theorem in `(u, v)`.
//!
//! With `F(u, v) = ∫₀ᵘ g(s, v) ds`, the region `R` on the wires' left gives
//! `∫∫_R g du dv = ∮ F dv`, `u` carried continuously along each wire. A wire
//! that winds the axis `k` times ends a `k`-fold turn from where it began,
//! which the seam its region is cut along makes good: `-2πk H(v₀)`, with
//! `H(v) = ∫ ḡ` from the reference latitude and `ḡ` the mean of `g` over a
//! turn, `v₀` the latitude the wire starts at. A region holding a pole adds
//! `2π H` there, taken toward growing `v`. Nothing is sampled into a polygon, so the result is
//! exact to quadrature precision however an edge bows.

use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::{NurbsRow, NurbsSurface, UBasis};
use brepkit_math::quadrature::gauss_legendre_points;
use brepkit_math::surfaces::SphericalSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{Face, FaceSurface};

use super::face_integrator::FaceContribution;
use crate::CheckError;

/// Integrands a face's properties sum at `(u, v)`: area, the volume flux
/// about `about`, the three volume moments and the three area moments.
pub(super) fn integrand<S: ParametricSurface>(
    surface: &S,
    u: f64,
    v: f64,
    about: Vec3,
) -> [f64; 8] {
    let p = surface.evaluate(u, v);
    let (su, sv) = surface.partials(u, v);
    let n = su.cross(sv);
    terms(Vec3::new(p.x(), p.y(), p.z()), n, about)
}

/// The integrands from a point and its normal. Summing about a point of its
/// own means a solid far from the origin, and only its volume is read: the
/// moments are left out there, as their coordinates' squares would swamp the
/// halving's test of whether an integral has settled, and it would never stop.
fn terms(p: Vec3, n: Vec3, about: Vec3) -> [f64; 8] {
    let n_len = n.length();
    let flux = (p - about).dot(n) / 3.0;
    if about.length_squared() > 0.0 {
        return [n_len, flux, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    }
    [
        n_len,
        flux,
        0.5 * p.x() * p.x() * n.x(),
        0.5 * p.y() * p.y() * n.y(),
        0.5 * p.z() * p.z() * n.z(),
        p.x() * n_len,
        p.y() * n_len,
        p.z() * n_len,
    ]
}

/// [`integrand`] on a NURBS surface from one evaluation of its point and
/// first partials, where the surface's own partials evaluate it once each.
fn nurbs_integrand(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    u: f64,
    v: f64,
    about: Vec3,
) -> [f64; 8] {
    let d = surface.first_derivatives(u, v);
    terms(d.0, d.1.cross(d.2), about)
}

/// [`integrand`] on a NURBS surface's line of constant `v`, alike bit for bit.
fn row_integrand(row: &NurbsRow<'_>, u: f64, about: Vec3) -> [f64; 8] {
    terms_at(row.point_and_partials(u), about)
}

/// [`integrand`] from a point and its first partials.
fn terms_at((p, su, sv): (Point3, Vec3, Vec3), about: Vec3) -> [f64; 8] {
    let n = su.cross(sv);
    terms(Vec3::new(p.x(), p.y(), p.z()), n, about)
}

/// One edge of a wire in its traversal: its curve, end points, and the
/// curve parameter it runs over from where the wire enters it.
pub struct BoundaryEdge<'a> {
    pub curve: &'a EdgeCurve,
    pub start: Point3,
    pub end: Point3,
    pub from: f64,
    pub to: f64,
}

/// The surface, how points map to its `(u, v)`, and where `H` starts.
pub(super) struct Chart<'a, S: ParametricSurface> {
    pub surface: &'a S,
    pub project: &'a dyn Fn(Point3) -> (f64, f64),
    /// The latitude `H` is taken from: a pole (or apex) the wires may run
    /// through, where `H` vanishes and so does a wire's jump across it.
    pub v_ref: f64,
    /// The poles at the low and high ends of `v`, where the surface has
    /// them (a cone's apex at one end, a sphere's at both).
    pub low: Option<f64>,
    pub high: Option<f64>,
    /// `v` is an angle too (round a torus's tube): it is carried along each
    /// wire as `u` is, and a wire winding round the tube needs nothing
    /// more, every integrand being periodic in it.
    pub v_periodic: bool,
    /// The surface's integrals in `u` in closed form.
    pub revolution: Revolution,
}

const TAU: f64 = std::f64::consts::TAU;

fn wrap_pi(x: f64) -> f64 {
    (x + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
}

fn add(acc: &mut [f64; 8], g: &[f64; 8], w: f64) {
    for (a, x) in acc.iter_mut().zip(g) {
        *a += w * x;
    }
}

/// `∫ f` over `[a, b]` by `rule`, each interval halved until its two halves
/// agree with it to 1e-12 of their size or to `floor`, or `None` past
/// `depth` halvings or where `f` is.
fn adaptive(
    f: &mut dyn FnMut(f64) -> Option<[f64; 8]>,
    a: f64,
    b: f64,
    rule: &[brepkit_math::quadrature::GaussPoint],
    depth: u32,
    floor: f64,
) -> Option<[f64; 8]> {
    let mut over = |a: f64, b: f64| -> Option<[f64; 8]> {
        let (half, mid) = (0.5 * (b - a), 0.5 * (a + b));
        let mut acc = [0.0; 8];
        for gp in rule {
            add(&mut acc, &f(half.madd(gp.x, mid))?, half * gp.w);
        }
        Some(acc)
    };
    let m = f64::midpoint(a, b);
    let whole = over(a, b)?;
    let (left, right) = (over(a, m)?, over(m, b)?);
    let mut halves = left;
    add(&mut halves, &right, 1.0);
    let size = halves.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let gap = whole
        .iter()
        .zip(&halves)
        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
    if gap <= (1e-12 * size).max(floor) {
        return Some(halves);
    }
    if depth == 0 {
        return None;
    }
    let mut sum = adaptive(f, a, m, rule, depth - 1, floor)?;
    add(&mut sum, &adaptive(f, m, b, rule, depth - 1, floor)?, 1.0);
    Some(sum)
}

/// `c₀ + Σₖ (cₖ cos ks + sₖ sin ks)` for `k` up to 3, as
/// `[c₀, c₁, s₁, c₂, s₂, c₃, s₃]`.
#[derive(Clone, Copy, Default)]
struct Trig([f64; 7]);

impl Trig {
    fn linear(c0: f64, c1: f64, s1: f64) -> Self {
        Self([c0, c1, s1, 0.0, 0.0, 0.0, 0.0])
    }

    /// The product with `d₀ + d₁ cos s + e₁ sin s`. The degree of `self`
    /// is at most 2, so the product's is at most 3.
    fn times(self, d0: f64, d1: f64, e1: f64) -> Self {
        let [c0, c1, s1, c2, s2, ..] = self.0;
        let (c, s) = ([c0, c1, c2], [0.0, s1, s2]);
        let (mut cos, mut sin) = ([0.0; 4], [0.0; 4]);
        for k in 0..3 {
            cos[k] += c[k] * d0;
            sin[k] += s[k] * d0;
            // By the product-to-sum rules each of `cos ks` and `sin ks` times
            // `cos s` or `sin s` is half a wave of `k + 1` and `k - 1`.
            let (cd, ce, sd, se) = (
                0.5 * c[k] * d1,
                0.5 * c[k] * e1,
                0.5 * s[k] * d1,
                0.5 * s[k] * e1,
            );
            cos[k + 1] += cd - se;
            sin[k + 1] += ce + sd;
            if k == 0 {
                // `cos(-s) = cos s` and `sin(-s) = -sin s`.
                cos[1] += cd + se;
                sin[1] -= sd - ce;
            } else {
                cos[k - 1] += cd + se;
                sin[k - 1] += sd - ce;
            }
        }
        Self([cos[0], cos[1], sin[1], cos[2], sin[2], cos[3], sin[3]])
    }

    fn sum(self, other: Self) -> Self {
        Self(std::array::from_fn(|i| self.0[i] + other.0[i]))
    }

    fn scaled(self, w: f64) -> Self {
        Self(self.0.map(|x| w * x))
    }

    /// `∫₀ᵘ`, with `waves` holding `sin ku` and `1 - cos ku` for `k` from 1
    /// to 3.
    fn integral(self, u: f64, waves: &[(f64, f64); 3]) -> f64 {
        let mut sum = self.0[0] * u;
        for (k, (&(sin_ku, versine), inverse)) in
            waves.iter().zip([1.0, 0.5, 1.0 / 3.0]).enumerate()
        {
            sum += inverse * self.0[2 * k + 1].madd(sin_ku, self.0[2 * k + 2] * versine);
        }
        sum
    }
}

/// How the circle at each `v` of a surface of revolution is set: its radius
/// `a` and height `b` along the axis, with their `v` derivatives.
#[derive(Clone, Copy)]
enum Profile {
    Cylinder { radius: f64 },
    Cone { cos_a: f64, sin_a: f64 },
    Sphere { radius: f64 },
    Torus { major: f64, minor: f64 },
}

impl Profile {
    /// `(a, b, a', b')` at `v`.
    fn at(self, v: f64) -> (f64, f64, f64, f64) {
        match self {
            Self::Cylinder { radius } => (radius, v, 0.0, 1.0),
            Self::Cone { cos_a, sin_a } => (v * cos_a, v * sin_a, cos_a, sin_a),
            Self::Sphere { radius } => {
                let (sin_v, cos_v) = v.sin_cos();
                (
                    radius * cos_v,
                    radius * sin_v,
                    -radius * sin_v,
                    radius * cos_v,
                )
            }
            Self::Torus { major, minor } => {
                let (sin_v, cos_v) = v.sin_cos();
                (
                    minor.madd(cos_v, major),
                    minor * sin_v,
                    -minor * sin_v,
                    minor * cos_v,
                )
            }
        }
    }
}

/// A cylinder, cone, sphere or torus as `p(u, v) = centre + a(v) (cos u x +
/// sin u y) + b(v) z`. Along a circle of constant `v` its point and normal
/// are trigonometric polynomials of degree one in `u` and its normal's length
/// is constant, so each [`integrand`] is one of degree three at most and its
/// integral in `u` has a closed form.
#[derive(Clone, Copy)]
pub(super) struct Revolution {
    centre: Vec3,
    x: Vec3,
    y: Vec3,
    z: Vec3,
    profile: Profile,
}

impl Revolution {
    pub(super) fn cylinder(s: &brepkit_math::surfaces::CylindricalSurface) -> Self {
        let o = s.origin();
        Self {
            centre: Vec3::new(o.x(), o.y(), o.z()),
            x: s.x_axis(),
            y: s.y_axis(),
            z: s.axis(),
            profile: Profile::Cylinder { radius: s.radius() },
        }
    }

    pub(super) fn cone(s: &brepkit_math::surfaces::ConicalSurface) -> Self {
        let o = s.apex();
        let (sin_a, cos_a) = s.half_angle().sin_cos();
        Self {
            centre: Vec3::new(o.x(), o.y(), o.z()),
            x: s.x_axis(),
            y: s.y_axis(),
            z: s.axis(),
            profile: Profile::Cone { cos_a, sin_a },
        }
    }

    pub(super) fn sphere(s: &SphericalSurface) -> Self {
        let o = s.center();
        Self {
            centre: Vec3::new(o.x(), o.y(), o.z()),
            x: s.x_axis(),
            y: s.y_axis(),
            z: s.z_axis(),
            profile: Profile::Sphere { radius: s.radius() },
        }
    }

    pub(super) fn torus(s: &brepkit_math::surfaces::ToroidalSurface) -> Self {
        let o = s.center();
        Self {
            centre: Vec3::new(o.x(), o.y(), o.z()),
            x: s.x_axis(),
            y: s.y_axis(),
            z: s.z_axis(),
            profile: Profile::Torus {
                major: s.major_radius(),
                minor: s.minor_radius(),
            },
        }
    }

    /// `dv/dt` of the point the chart's projection reads for a point moving
    /// at `velocity` through `p`, at `(u, v)`. The projection reads `v` from
    /// the height along the axis on a cylinder and a cone, and from the
    /// angle round the meridian's centre on a ball and a torus, so an edge
    /// off the surface (a chord of a rim) is read where its points project.
    fn dv_dt(&self, p: Point3, velocity: Vec3, u: f64, v: f64) -> f64 {
        let (sin_u, cos_u) = u.sin_cos();
        let out = self.x * cos_u + self.y * sin_u;
        let (along, across) = (self.z.dot(velocity), out.dot(velocity));
        let centre = match self.profile {
            Profile::Cylinder { .. } => return along,
            Profile::Cone { cos_a, sin_a } => {
                return if sin_a.abs() > 1e-15 {
                    along / sin_a
                } else if cos_a.abs() > 1e-15 {
                    across / cos_a
                } else {
                    0.0
                };
            }
            Profile::Sphere { .. } => self.centre,
            Profile::Torus { major, .. } => self.centre + out * major,
        };
        let reach = (Vec3::new(p.x(), p.y(), p.z()) - centre).length();
        if reach > 0.0 {
            let (sin_v, cos_v) = v.sin_cos();
            cos_v.madd(along, -sin_v * across) / reach
        } else {
            0.0
        }
    }

    /// The same surface with `u` taken from `angle`: `u - angle` in it is
    /// `u` here.
    fn turned(self, angle: f64) -> Self {
        let (sin, cos) = angle.sin_cos();
        Self {
            x: self.x * cos + self.y * sin,
            y: self.y * cos - self.x * sin,
            ..self
        }
    }

    /// Each [`integrand`] at `v` as a trigonometric polynomial in `u`.
    fn series(&self, v: f64, about: Vec3) -> [Trig; 8] {
        let (a, b, da, db) = self.profile.at(v);
        // `p - about = p0 + p1 cos u + p2 sin u`, and with the partials
        // `a (-sin u x + cos u y)` and `a' (cos u x + sin u y) + b' z` the
        // normal is `n0 + n1 cos u + n2 sin u`.
        let p0 = self.centre - about + self.z * b;
        let (p1, p2) = (self.x * a, self.y * a);
        let n0 = self.x.cross(self.y) * (-a * da);
        let (n1, n2) = (
            self.y.cross(self.z) * (a * db),
            self.z.cross(self.x) * (a * db),
        );
        let length = a.abs() * da.hypot(db);
        let coords = |w: Vec3| [w.x(), w.y(), w.z()];
        let (p0, p1, p2) = (coords(p0), coords(p1), coords(p2));
        let (n0, n1, n2) = (coords(n0), coords(n1), coords(n2));
        let p = |i: usize| Trig::linear(p0[i], p1[i], p2[i]);
        let flux = (0..3)
            .map(|i| p(i).times(n0[i], n1[i], n2[i]))
            .fold(Trig::default(), Trig::sum)
            .scaled(1.0 / 3.0);
        let area = Trig::linear(length, 0.0, 0.0);
        if about.length_squared() > 0.0 {
            return [
                area,
                flux,
                Trig::default(),
                Trig::default(),
                Trig::default(),
                Trig::default(),
                Trig::default(),
                Trig::default(),
            ];
        }
        let moment = |i: usize| {
            p(i).times(p0[i], p1[i], p2[i])
                .times(n0[i], n1[i], n2[i])
                .scaled(0.5)
        };
        let centroid = |i: usize| p(i).scaled(length);
        [
            area,
            flux,
            moment(0),
            moment(1),
            moment(2),
            centroid(0),
            centroid(1),
            centroid(2),
        ]
    }

    /// `∫₀ᵘ g(s, v) ds` for each [`integrand`] `g`.
    fn antiderivative(&self, u: f64, v: f64, about: Vec3) -> [f64; 8] {
        // `1 - cos ku` as `2 sin²(ku / 2)`: the difference would round to
        // nothing for a small `u`, where a sliver of a face is all of it.
        let (sin_half, cos_half) = (0.5 * u).sin_cos();
        let (sin_u, cos_u) = (
            2.0 * sin_half * cos_half,
            cos_half.madd(cos_half, -sin_half * sin_half),
        );
        let sin_three_halves = sin_u.madd(cos_half, cos_u * sin_half);
        let (sin_2u, cos_2u) = (2.0 * sin_u * cos_u, cos_u.madd(cos_u, -sin_u * sin_u));
        let waves = [
            (sin_u, 2.0 * sin_half * sin_half),
            (sin_2u, 2.0 * sin_u * sin_u),
            (
                sin_2u.madd(cos_u, cos_2u * sin_u),
                2.0 * sin_three_halves * sin_three_halves,
            ),
        ];
        self.series(v, about).map(|g| g.integral(u, &waves))
    }

    /// The most each [`integrand`] can be at `v`, the sum of its waves'
    /// sizes: how far [`Self::antiderivative`] moves for a step in `u`.
    fn slope_bound(&self, v: f64, about: Vec3) -> [f64; 8] {
        self.series(v, about)
            .map(|g| g.0.iter().fold(0.0, |sum, c| sum + c.abs()))
    }

    /// The mean of each [`integrand`] over a turn at `v`.
    fn mean(&self, v: f64, about: Vec3) -> [f64; 8] {
        self.series(v, about).map(|g| g.0[0])
    }
}

/// `H(v) = ∫ ḡ` from `v_ref` to `v`, `ḡ` the mean over a turn.
fn mean_integral<S: ParametricSurface>(chart: &Chart<'_, S>, v: f64, about: Vec3) -> [f64; 8] {
    let gauss = gauss_legendre_points(10);
    let span = v - chart.v_ref;
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let panels = (span.abs() / std::f64::consts::FRAC_PI_4)
        .ceil()
        .clamp(1.0, 64.0) as u32;
    let width = span / f64::from(panels);
    let mut acc = [0.0; 8];
    for i in 0..panels {
        let mid = width.madd(f64::from(i) + 0.5, chart.v_ref);
        for gp in gauss {
            let w = (0.5 * width).madd(gp.x, mid);
            add(
                &mut acc,
                &chart.revolution.mean(w, about),
                0.5 * width * gp.w,
            );
        }
    }
    acc
}

/// An edge's parameter span in traversal order, cut where the quadrature
/// wants it: a conic every eighth of a turn (its parameter is its angle), a
/// NURBS curve at its knots, a line in two.
pub fn edge_pieces(edge: &BoundaryEdge<'_>) -> Vec<(f64, f64)> {
    let span = edge.to - edge.from;
    let uniform = |n: u32| {
        (0..n)
            .map(|i| {
                (
                    span.madd(f64::from(i) / f64::from(n), edge.from),
                    span.madd(f64::from(i + 1) / f64::from(n), edge.from),
                )
            })
            .collect()
    };
    match edge.curve {
        EdgeCurve::Line => uniform(2),
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => uniform(
            (span.abs() / std::f64::consts::FRAC_PI_8)
                .ceil()
                .clamp(1.0, 16.0) as u32,
        ),
        EdgeCurve::NurbsCurve(nc) => {
            let (lo, hi) = (edge.from.min(edge.to), edge.from.max(edge.to));
            let mut cuts: Vec<f64> = nc
                .knots()
                .iter()
                .copied()
                .filter(|&k| k > lo + 1e-12 * (hi - lo) && k < hi - 1e-12 * (hi - lo))
                .collect();
            cuts.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (hi - lo));
            let mut ends = vec![lo];
            ends.extend(cuts);
            ends.push(hi);
            let mut pieces: Vec<(f64, f64)> = ends.windows(2).map(|w| (w[0], w[1])).collect();
            if edge.to < edge.from {
                pieces = pieces.into_iter().rev().map(|(a, b)| (b, a)).collect();
            }
            pieces
        }
    }
}

/// A face's integrals over the region on its wires' left, each wire's edges
/// in traversal order: area and area moments unsigned, the volume terms
/// times `sign` (the face's orientation against its surface's normal).
/// `None` when the wires do not read: consecutive edges that do not meet, a
/// piece that turns a quarter turn about the axis however finely it is cut
/// (a wire through a pole), a turn that is not whole, a wire of no area whose side the
/// sum of turns cannot tell, a pole the surface does not have, or a region
/// of no area.
#[allow(clippy::too_many_lines)]
pub(super) fn integrate_by_boundary<S: ParametricSurface>(
    chart: &Chart<'_, S>,
    wires: &[Vec<BoundaryEdge<'_>>],
    sign: f64,
    about: Vec3,
) -> Option<FaceContribution> {
    const AREA_SLACK: f64 = 1e-12;
    let gauss = gauss_legendre_points(8);
    // A point with no `u` of its own (a cone's apex) may take any.
    let singular = |u: f64, v: f64| {
        chart.surface.partial_u(u, v).length() <= 1e-6 * chart.surface.partial_v(u, v).length()
    };
    let quarter = std::f64::consts::FRAC_PI_2;
    // `F` is taken from where the first wire starts rather than from the
    // frame's `u = 0`, which may lie most of a turn away: there `F` is the
    // integral over most of a turn, and the wires' sums cancel it out to the
    // face's far smaller one, losing its digits on the way. `F(u, v) -
    // F(u₀, v)` serves as well, for the face's wires together do not wind
    // round `v`; so does a wire's `u` a whole turn on, which moves it by
    // `∮ 2π ḡ(v) dv`, nothing, or on a torus a whole turn of its tube, which
    // the count of whole surfaces below takes off.
    let u_from = wires.first().and_then(|w| w.first()).map_or(0.0, |e| {
        let walk = Walk::new(e.curve, e.start, e.end);
        (chart.project)(walk.point(walk.tau(e.from))).0
    });
    let from_start = chart.revolution.turned(u_from);
    // The straight step in `(u, v)` from where one edge's curve ends to where
    // the next one's starts. Each curve is read from its own parameters, and
    // two of them can part at a vertex by its tolerance (a long arc of a
    // huge ellipse reads its ends 1e-7 along itself). The step keeps each
    // loop closed, so its integral does not hang on where `F` is taken from.
    let bridge = |acc: &mut [f64; 8], twice_area: &mut f64, from: (f64, f64), to: (f64, f64)| {
        let (du, dv) = (to.0 - from.0, to.1 - from.1);
        if dv.abs() > 0.0 {
            for gp in gauss {
                let s = 0.5.madd(gp.x, 0.5);
                let g = from_start.antiderivative(
                    du.madd(s, from.0) - u_from,
                    dv.madd(s, from.1),
                    about,
                );
                add(acc, &g, 0.5 * gp.w * dv);
            }
        }
        *twice_area += from.0.madd(dv, -(from.1 * du));
    };
    let mut total = [0.0; 8];
    let (mut turns_sum, mut winds, mut patch, mut holes) = (0_i64, false, false, true);
    for wire in wires {
        let first = wire.first()?;
        let first_walk = Walk::new(first.curve, first.start, first.end);
        let entry = first_walk.point(first_walk.tau(first.from));
        let (u_entry, v_start) = (chart.project)(entry);
        // Each wire starts within half a turn of the first.
        let mut u = u_from + wrap_pi(u_entry - u_from);
        let u_start = u;
        let mut v = v_start;
        let v_wrap = |x: f64| if chart.v_periodic { wrap_pi(x) } else { x };
        let mut acc = [0.0; 8];
        let mut twice_area = 0.0;
        let (mut exit, mut chord) = (entry, 0.0);
        for edge in wire {
            let walk = Walk::new(edge.curve, edge.start, edge.end);
            let (from, to) = (walk.tau(edge.from), walk.tau(edge.to));
            let point = |t: f64| walk.point(t);
            let at = |t: f64| (chart.project)(point(t));
            let (enter, leave) = (point(from), point(to));
            chord = (leave - enter).length();
            if (enter - exit).length() > 1e-6 + 1e-3 * chord {
                return None;
            }
            exit = leave;
            let (ue, ve) = at(from);
            let entered = (u + wrap_pi(ue - u), v + v_wrap(ve - v));
            bridge(&mut acc, &mut twice_area, (u, v), entered);
            (u, v) = entered;
            // A piece that turns a quarter turn about the axis is halved (a
            // NURBS rim's knot span may be a third of a circle); one that
            // still does at a thousandth of its edge runs past a pole.
            let floor = 1e-3 * (to - from).abs();
            let mut pending = edge_pieces(&BoundaryEdge {
                curve: edge.curve,
                start: edge.start,
                end: edge.end,
                from,
                to,
            });
            pending.reverse();
            while let Some((ta, tb)) = pending.pop() {
                let (ua, va) = at(ta);
                let u_a = u + wrap_pi(ua - u);
                let v_a = v + v_wrap(va - v);
                let (ub, vb) = at(tb);
                let u_b = u_a + wrap_pi(ub - u_a);
                let v_b = v_a + v_wrap(vb - v_a);
                let mid = f64::midpoint(ta, tb);
                // The rule's nodes over `[a, b]`, `u` and `v` carried on from
                // the piece's start.
                let nodes_over = |a: f64, b: f64| {
                    let (half, centre) = (0.5 * (b - a), f64::midpoint(a, b));
                    let mut nodes = [(0.0, 0.0, 0.0, 0.0); 8];
                    for (node, gp) in nodes.iter_mut().zip(gauss) {
                        let t = half.madd(gp.x, centre);
                        let (un, vn) = at(t);
                        *node = (
                            t,
                            u_a + wrap_pi(un - u_a),
                            v_a + v_wrap(vn - v_a),
                            half * gp.w,
                        );
                    }
                    nodes
                };
                let nodes = nodes_over(ta, tb);
                let swings = !singular(ua, va)
                    && (nodes.iter().any(|n| (n.1 - u_a).abs() > quarter)
                        || (!singular(ub, vb) && (u_b - u_a).abs() > quarter));
                if swings {
                    if (tb - ta).abs() <= floor {
                        return None;
                    }
                    pending.push((mid, tb));
                    pending.push((ta, mid));
                    continue;
                }
                // The integrals over `[a, b]` by the rule, the area's last,
                // and the most the stencil's rounding can move any of them.
                let sums = |a: f64, b: f64, nodes: [(f64, f64, f64, f64); 8]| {
                    // A five-point stencil wide enough that rounding in
                    // far-off coordinates stays small against it.
                    let h = 1e-3 * (b - a);
                    let mut sums = [0.0; 9];
                    let mut rounding = 0.0_f64;
                    for (t, un, vn, w) in nodes {
                        let [u1, u2, u3, u4] =
                            [t + h, t - h, t + 2.0 * h, t - 2.0 * h].map(|x| at(x).0);
                        let du = 8.0f64.madd(wrap_pi(u1 - u2), -wrap_pi(u3 - u4)) / (12.0 * h);
                        // `dv/dt` of the projected edge from the edge's
                        // velocity. A stencil's differences carry the
                        // rounding of each point, which a curve far from its
                        // own centre (a long arc of a huge ellipse) has in
                        // plenty.
                        let dv = chart.revolution.dv_dt(point(t), walk.velocity(t), un, vn);
                        // Each difference rounds by an ulp of its coordinate.
                        let (nu, nv) = (
                            1.5 * f64::EPSILON * (1.0 + un.abs()) / h.abs(),
                            1.5 * f64::EPSILON * (1.0 + vn.abs()) / h.abs(),
                        );
                        // A latitude (a rim circle about the axis) adds nothing.
                        if dv.abs() > 1e-12 * (1.0 + vn.abs()) {
                            let g = from_start.antiderivative(un - u_from, vn, about);
                            // `F` rounds as `u` does, by its slope's ulps of
                            // `u`, which counts where `F` itself is near
                            // nothing: along an edge on the ruling where the
                            // wires start, every piece's integral is rounding.
                            let slopes = chart.revolution.slope_bound(vn, about);
                            let u_rounding = 4.0 * f64::EPSILON * (1.0 + un.abs() + u_from.abs());
                            for ((s, x), slope) in sums.iter_mut().zip(g).zip(slopes) {
                                *s += w * dv * x;
                                rounding = rounding
                                    .max((w * nv * x).abs())
                                    .max((w * dv * slope).abs() * u_rounding);
                            }
                        }
                        sums[8] += w * (un * dv - vn * du);
                        rounding = rounding.max((w * (un.abs() * nv + vn.abs() * nu)).abs());
                    }
                    (sums, 8.0 * rounding)
                };
                let (mut piece, _) = sums(ta, tb, nodes);
                // A NURBS edge's parameter can run unevenly along the surface
                // (a rational arc whose weights grow along it, or a polynomial
                // whose speed swings), so its piece is kept only where its
                // halves agree with it, and halved otherwise.
                if matches!(edge.curve, EdgeCurve::NurbsCurve(_)) {
                    let ((left, left_rounding), (right, right_rounding)) = (
                        sums(ta, mid, nodes_over(ta, mid)),
                        sums(mid, tb, nodes_over(mid, tb)),
                    );
                    let halves: [f64; 9] = std::array::from_fn(|k| left[k] + right[k]);
                    let rounding = left_rounding + right_rounding;
                    let size = halves.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                    let gap = piece
                        .iter()
                        .zip(&halves)
                        .fold(0.0_f64, |m, (x, y)| m.max((x - y).abs()));
                    // The gap measures the whole piece's error; the halves
                    // kept are far closer. A gap within the stencil's
                    // rounding is no error of the rule (a knot span 6e-11
                    // wide, or a stretch where the edge barely moves), and
                    // halving only multiplies that rounding.
                    if gap > 1e-9 * size && gap > 100.0 * rounding {
                        if (tb - ta).abs() <= floor {
                            return None;
                        }
                        pending.push((mid, tb));
                        pending.push((ta, mid));
                        continue;
                    }
                    piece = halves;
                }
                for (a, x) in acc.iter_mut().zip(&piece) {
                    *a += x;
                }
                twice_area += piece[8];
                u = u_b;
                v = v_b;
            }
        }
        if (entry - exit).length() > 1e-6 + 1e-3 * chord {
            return None;
        }
        let turn = u - u_start;
        let turns = (turn / TAU).round();
        if (turn - turns * TAU).abs() > 1e-6 {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)]
        let turns = turns as i64;
        let tube_turns = ((v - v_start) / TAU).round();
        if chart.v_periodic && (v - v_start - tube_turns * TAU).abs() > 1e-6 {
            return None;
        }
        #[allow(clippy::cast_precision_loss)]
        let closed = (
            (turns as f64).madd(TAU, u_start),
            if chart.v_periodic {
                tube_turns.madd(TAU, v_start)
            } else {
                v_start
            },
        );
        bridge(&mut acc, &mut twice_area, (u, v), closed);
        if chart.v_periodic && tube_turns != 0.0 {
            if turns != 0 {
                return None;
            }
            add(&mut total, &acc, 1.0);
            winds = true;
            continue;
        }
        if turns != 0 {
            let h = mean_integral(chart, v_start, about);
            #[allow(clippy::cast_precision_loss)]
            add(&mut acc, &h, -TAU * turns as f64);
        } else if twice_area.abs() <= AREA_SLACK {
            return None;
        }
        add(&mut total, &acc, 1.0);
        turns_sum += turns;
        winds |= turns != 0;
        patch |= turns == 0 && twice_area > AREA_SLACK;
        holes &= turns == 0 && twice_area < -AREA_SLACK;
    }
    // A total of one turn holds the high pole (the face on the wire's left,
    // toward growing `v`), minus one the low; with none, a band or a patch
    // holds neither and a face of clockwise holes both. A pole the surface
    // does not have (a cone's open end) holds no face.
    let (holds_low, holds_high) = match turns_sum {
        1 => (false, true),
        -1 => (true, false),
        0 if winds || patch => (false, false),
        0 if holes => (true, true),
        _ => return None,
    };
    if holds_high {
        add(&mut total, &mean_integral(chart, chart.high?, about), TAU);
    }
    if holds_low {
        add(&mut total, &mean_integral(chart, chart.low?, about), -TAU);
    }
    // On a torus, a wire round the tube carries `u` on from wherever it
    // starts, and one round the axis reads `H` at whichever turn of `v` it
    // starts in: a turn more or less moves either by the whole surface's
    // integral. A face holds more than none of the surface and less than all
    // of it.
    if chart.v_periodic && winds {
        let mut whole = mean_integral(chart, chart.v_ref + TAU, about);
        for x in &mut whole {
            *x *= TAU;
        }
        if whole[0] <= 0.0 {
            return None;
        }
        let turns = (total[0] / whole[0]).floor();
        add(&mut total, &whole, -turns);
    }
    if total[0] <= 0.0 {
        return None;
    }
    Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    })
}

/// A face's wires as [`BoundaryEdge`]s in traversal order. Each open edge
/// runs from where the wire stands (the stored flags need not chain), each
/// closed one from its vertex. `None` for a closed line.
fn boundary_wires<'t>(
    topo: &'t Topology,
    face: &Face,
) -> Result<Option<Vec<Vec<BoundaryEdge<'t>>>>, CheckError> {
    let ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();
    wires_of(topo, &ids)
}

/// [`boundary_wires`] for the wires `ids`.
pub fn wires_of<'t>(
    topo: &'t Topology,
    ids: &[brepkit_topology::wire::WireId],
) -> Result<Option<Vec<Vec<BoundaryEdge<'t>>>>, CheckError> {
    let mut wires = Vec::new();
    for &wid in ids {
        let mut edges = Vec::new();
        let mut cursor: Option<Point3> = None;
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let start = topo.vertex(edge.start())?.point();
            let end = topo.vertex(edge.end())?.point();
            let curve = edge.curve();
            if edge.start() == edge.end() {
                let spans = match curve {
                    EdgeCurve::Circle(c) => {
                        let at = c.project(start);
                        vec![(at, at + TAU)]
                    }
                    EdgeCurve::Ellipse(e) => {
                        let at = e.project(start);
                        vec![(at, at + TAU)]
                    }
                    // A closed NURBS rim runs from its vertex to its domain's
                    // end and on from its start back to the vertex.
                    EdgeCurve::NurbsCurve(nc) => {
                        let (u0, u1) = nc.domain();
                        let at = crate::util::nurbs_seam_parameter(nc, start, u0, u1);
                        vec![(at, u1), (u0, at)]
                    }
                    EdgeCurve::Line => return Ok(None),
                };
                let spans: Vec<(f64, f64)> = if oe.is_forward() {
                    spans
                } else {
                    spans.into_iter().rev().map(|(a, b)| (b, a)).collect()
                };
                for (from, to) in spans {
                    if (to - from).abs() > 0.0 {
                        edges.push(BoundaryEdge {
                            curve,
                            start,
                            end,
                            from,
                            to,
                        });
                    }
                }
                cursor = Some(start);
                continue;
            }
            if matches!(curve, EdgeCurve::Line) && (end - start).length() < 1e-12 {
                continue;
            }
            let (t0, t1) = curve.domain_with_endpoints(start, end);
            let first = curve.evaluate_with_endpoints(t0, start, end);
            let runs_start_to_end = (first - start).length() <= (first - end).length();
            let forward = cursor.map_or_else(
                || oe.is_forward(),
                |c| (start - c).length() <= (end - c).length(),
            );
            let (from, to) = if forward == runs_start_to_end {
                (t0, t1)
            } else {
                (t1, t0)
            };
            edges.push(BoundaryEdge {
                curve,
                start,
                end,
                from,
                to,
            });
            cursor = Some(if forward { end } else { start });
        }
        if !edges.is_empty() {
            wires.push(edges);
        }
    }
    Ok(Some(wires))
}

/// Points along every wire, sixty-five to an edge.
fn wire_samples(wires: &[Vec<BoundaryEdge<'_>>]) -> Vec<Point3> {
    wires
        .iter()
        .flatten()
        .flat_map(|e| {
            (0..=64).map(move |k| {
                let t = (e.to - e.from).madd(f64::from(k) / 64.0, e.from);
                e.curve.evaluate_with_endpoints(t, e.start, e.end)
            })
        })
        .collect()
}

/// A cylinder, cone or sphere face's integrals by [`integrate_by_boundary`],
/// `sign` its orientation against its surface. `None` for another surface,
/// or a face the boundary does not read (a cone face across its apex, a
/// sphere face whose wires come near every candidate axis's poles).
pub(super) fn curved_face_by_boundary(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    if !matches!(
        face.surface(),
        FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
    ) {
        return Ok(None);
    }
    if let FaceSurface::Torus(t) = face.surface()
        && t.minor_radius() < t.major_radius()
        && whole_surface(topo, face)?
    {
        return whole_torus_less_holes(topo, face, t, sign, about);
    }
    let Some(wires) = boundary_wires(topo, face)? else {
        return Ok(None);
    };
    Ok(match face.surface() {
        FaceSurface::Cylinder(s) => {
            let project = |p: Point3| s.project_point(p);
            // No pole: `H` from where the first wire starts, where one panel
            // of its integral stays short.
            let v_ref = wires.first().and_then(|w| w.first()).map_or(0.0, |e| {
                project(e.curve.evaluate_with_endpoints(e.from, e.start, e.end)).1
            });
            let chart = Chart {
                surface: s,
                project: &project,
                v_ref,
                low: None,
                high: None,
                v_periodic: false,
                revolution: Revolution::cylinder(s),
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Cone(s) => {
            let vs: Vec<f64> = wire_samples(&wires)
                .into_iter()
                .map(|p| s.project_point(p).1)
                .collect();
            let slack = 1e-9 * vs.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let (low, high) = if vs.iter().all(|&v| v >= -slack) {
                (Some(0.0), None)
            } else if vs.iter().all(|&v| v <= slack) {
                (None, Some(0.0))
            } else {
                return Ok(None);
            };
            let project = |p: Point3| s.project_point(p);
            let chart = Chart {
                surface: s,
                project: &project,
                v_ref: 0.0,
                low,
                high,
                v_periodic: false,
                revolution: Revolution::cone(s),
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Sphere(s) => {
            let samples = wire_samples(&wires);
            let reach = |axis: Vec3| {
                samples.iter().fold(0.0_f64, |m, p| {
                    let w = *p - s.center();
                    m.max((w.dot(axis) / w.length().max(1e-300)).abs())
                })
            };
            let axis = [
                s.z_axis(),
                s.x_axis(),
                s.y_axis(),
                Vec3::new(
                    0.447_213_595_499_957_9,
                    0.547_722_557_505_166_1,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
                Vec3::new(-0.5, 0.763_762_615_825_973_4, 0.408_248_290_463_863),
                Vec3::new(
                    0.597_614_304_667_196_8,
                    -0.377_964_473_009_227_2,
                    std::f64::consts::FRAC_1_SQRT_2,
                ),
            ]
            .into_iter()
            .map(|a| (reach(a), a))
            .filter(|(m, _)| *m <= 3.0_f64.to_radians().cos())
            .min_by(|a, b| a.0.total_cmp(&b.0));
            let Some((_, axis)) = axis else {
                return Ok(None);
            };
            let Ok(turned) = SphericalSurface::with_axis(s.center(), s.radius(), axis) else {
                return Ok(None);
            };
            let project = |p: Point3| turned.project_point(p);
            let chart = Chart {
                surface: &turned,
                project: &project,
                v_ref: -std::f64::consts::FRAC_PI_2,
                low: Some(-std::f64::consts::FRAC_PI_2),
                high: Some(std::f64::consts::FRAC_PI_2),
                v_periodic: false,
                revolution: Revolution::sphere(&turned),
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        // A ring torus: no pole, and its tube angle `v` wraps as `u` does.
        FaceSurface::Torus(t) if t.minor_radius() < t.major_radius() => {
            let project = |p: Point3| t.project_point(p);
            let chart = Chart {
                surface: t,
                project: &project,
                v_ref: 0.0,
                low: None,
                high: None,
                v_periodic: true,
                revolution: Revolution::torus(t),
            };
            integrate_by_boundary(&chart, &wires, sign, about)
        }
        FaceSurface::Plane { .. } | FaceSurface::Torus(_) | FaceSurface::Nurbs(_) => None,
    })
}

/// Whether a face's outer wire is its surface's seams alone, closed edges of
/// no length: the face is the whole surface, less its holes.
fn whole_surface(topo: &Topology, face: &Face) -> Result<bool, CheckError> {
    let wire = topo.wire(face.outer_wire())?;
    let mut all = true;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        all &= edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Line);
    }
    Ok(all && !wire.edges().is_empty())
}

/// A ring torus face that is the whole torus less its holes: the torus's
/// integrals over both turns, by the trapezoid rule (exact for its
/// trigonometric integrands), less each hole's, read on the hole's wire run
/// the other way round, which holds the hole on its left.
fn whole_torus_less_holes(
    topo: &Topology,
    face: &Face,
    t: &brepkit_math::surfaces::ToroidalSurface,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    const N: u32 = 32;
    let step = TAU / f64::from(N);
    let mut total = [0.0; 8];
    for i in 0..N {
        for j in 0..N {
            let g = integrand(t, step * f64::from(i), step * f64::from(j), about);
            add(&mut total, &g, step * step);
        }
    }
    let project = |p: Point3| t.project_point(p);
    let chart = Chart {
        surface: t,
        project: &project,
        v_ref: 0.0,
        low: None,
        high: None,
        v_periodic: true,
        revolution: Revolution::torus(t),
    };
    for &hole in face.inner_wires() {
        let Some(wires) = wires_of(topo, &[hole])? else {
            return Ok(None);
        };
        let around: Vec<Vec<BoundaryEdge<'_>>> = wires
            .into_iter()
            .map(|wire| {
                wire.into_iter()
                    .rev()
                    .map(|e| BoundaryEdge {
                        from: e.to,
                        to: e.from,
                        ..e
                    })
                    .collect()
            })
            .collect();
        let Some(inside) = integrate_by_boundary(&chart, &around, 1.0, about) else {
            return Ok(None);
        };
        let inside = [
            inside.area,
            inside.volume,
            inside.volume_moment_x,
            inside.volume_moment_y,
            inside.volume_moment_z,
            inside.centroid_x,
            inside.centroid_y,
            inside.centroid_z,
        ];
        add(&mut total, &inside, -1.0);
    }
    if total[0] <= 0.0 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// The most one weight of a patch read over its domain may be against
/// another. Weights far apart crowd the surface into a corner of its domain,
/// where the panels' rules miss it: a patch weighted 1e16 over half of it
/// read as half its area.
const MAX_WEIGHT_RATIO: f64 = 64.0;

/// Side `k` of `s`'s domain, `v = v0`, `u = u1`, `v = v1` and `u = u0` in
/// turn: the outer row of control points there, and their weights.
fn side_row(s: &NurbsSurface, k: usize) -> (Vec<Point3>, Vec<f64>) {
    let (cps, ws) = (s.control_points(), s.weights());
    let (nu, nv) = (cps.len(), cps.first().map_or(0, Vec::len));
    let at = |i: usize, j: usize| (cps[i][j], ws[i][j]);
    match k {
        0 => (0..nu).map(|i| at(i, 0)).unzip(),
        1 => (0..nv).map(|j| at(nu - 1, j)).unzip(),
        2 => (0..nu).map(|i| at(i, nv - 1)).unzip(),
        _ => (0..nv).map(|j| at(0, j)).unzip(),
    }
}

/// Whether `c` is side `k` of `s` (see [`side_row`]): the same curve, the
/// side's degree and knots, up to scale and direction, on the side's
/// control points within `reach` with the side's weights up to a common
/// factor, which leaves the curve as it is.
fn is_side(s: &NurbsSurface, c: &NurbsCurve, k: usize, reach: f64) -> bool {
    let (degree, knots) = if k.is_multiple_of(2) {
        (s.degree_u(), s.knots_u())
    } else {
        (s.degree_v(), s.knots_v())
    };
    let (row, side_w) = side_row(s, k);
    let unit = |ks: &[f64]| -> Vec<f64> {
        let (lo, hi) = (ks[0], ks[ks.len() - 1]);
        ks.iter().map(|&x| (x - lo) / (hi - lo)).collect()
    };
    if c.degree() != degree
        || c.control_points().len() != row.len()
        || c.knots().len() != knots.len()
    {
        return false;
    }
    let ratios = |w: Vec<f64>| -> Vec<f64> { w.iter().map(|x| x / w[0]).collect() };
    // A ratio that overflows would pass against any weight: the bound
    // scales with it.
    let same = |a: &[f64], b: &[f64]| {
        a.iter().zip(b).all(|(x, y)| {
            x.is_finite() && y.is_finite() && (x - y).abs() <= 1e-12 * x.abs().max(y.abs())
        })
    };
    let side_w = ratios(side_w);
    let weighs_forward = same(&ratios(c.weights().to_vec()), &side_w);
    let weighs_backward = same(
        &ratios(c.weights().iter().rev().copied().collect()),
        &side_w,
    );
    let (mine, theirs) = (unit(c.knots()), unit(knots));
    let close = |p: &Point3, q: &Point3| (*p - *q).length() <= reach;
    // Knots match to rounding only: knots a hair apart can bound a
    // different curve on the same control points.
    let forward = weighs_forward
        && c.control_points()
            .iter()
            .zip(&row)
            .all(|(p, q)| close(p, q))
        && mine
            .iter()
            .zip(&theirs)
            .all(|(a, b)| a.to_bits() == b.to_bits());
    let backward = weighs_backward
        && c.control_points()
            .iter()
            .rev()
            .zip(&row)
            .all(|(p, q)| close(p, q))
        && {
            // Read reversed, each span against the side's at its own
            // scale: mirroring rounds on the domain's scale, which can
            // hide a knot moved inside a narrow span.
            let back: Vec<f64> = mine.windows(2).rev().map(|w| w[1] - w[0]).collect();
            theirs
                .windows(2)
                .map(|w| w[1] - w[0])
                .zip(&back)
                .all(|(a, &b)| (a - b).abs() <= 4.0 * f64::EPSILON * a.abs().max(b.abs()))
        };
    forward || backward
}

/// A face that is its whole open NURBS surface, its integrals over the
/// surface's `(u, v)` domain directly, in panels between its knots. Green's
/// theorem along its boundary comes to the same through a projection of
/// every boundary point, which a face cut from nothing need not pay for: a
/// sweep tiles its tube with hundreds of such patches. Taken only where the
/// face is shown to be the whole surface: no holes, and its wire an edge
/// from each corner of the domain to the next, each side of the surface
/// shown to be its edge, a straight edge by the side's control points lying
/// on it, and a NURBS, circle or ellipse edge by being the side's own curve.
/// A band closed along `v`, its first and last rows across `u` one curve (a
/// blend round a closed rim), is the whole surface when its wire runs both
/// closed sides and its seam row once each way.
/// The surface has clamped knots and positive weights none more than
/// [`MAX_WEIGHT_RATIO`] times another, so its sides are the outer rows of
/// its control points, which bound them, and its integrands are smooth in
/// each panel. `None` for any other face.
pub(super) fn nurbs_face_over_domain(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    const DEPTH: u32 = 24;
    let FaceSurface::Nurbs(s) = face.surface() else {
        return Ok(None);
    };
    let clamped = |knots: &[f64], degree: usize| {
        knots.len() > 2 * degree + 1
            && knots[..=degree]
                .iter()
                .all(|&k| k.to_bits() == knots[0].to_bits())
            && knots[knots.len() - degree - 1..]
                .iter()
                .all(|&k| k.to_bits() == knots[knots.len() - 1].to_bits())
    };
    // A knot repeated more often than the degree breaks the surface apart:
    // its sides' control points no longer show it continuous.
    let continuous = |knots: &[f64], degree: usize| {
        knots[degree + 1..knots.len() - degree - 1]
            .chunk_by(|a, b| a.to_bits() == b.to_bits())
            .all(|run| run.len() <= degree)
    };
    let (w_lo, w_hi) = s
        .weights()
        .iter()
        .flatten()
        .fold((f64::INFINITY, 0.0_f64), |(lo, hi), &w| {
            (lo.min(w), hi.max(w))
        });
    // A fold by `min` and `max` passes over a NaN, so each weight is read.
    let weighted = s.weights().iter().flatten().all(|w| w.is_finite())
        && w_lo > 0.0
        && w_hi <= MAX_WEIGHT_RATIO * w_lo;
    // A band closed along `v` (round a closed rim) is the whole surface when
    // its wire runs both closed sides and its one seam row each way.
    let closed_v = !open_and_poleless(s) && closed_along_v_and_poleless(s);
    if !face.inner_wires().is_empty()
        || !weighted
        || !clamped(s.knots_u(), s.degree_u())
        || !clamped(s.knots_v(), s.degree_v())
        || !continuous(s.knots_u(), s.degree_u())
        || !continuous(s.knots_v(), s.degree_v())
        || !(closed_v || open_and_poleless(s))
    {
        return Ok(None);
    }
    let cps = s.control_points();
    let (nu, nv) = (cps.len(), cps.first().map_or(0, Vec::len));
    if nu < 2 || nv < 2 {
        return Ok(None);
    }
    // Side `k` runs from corner `k` to the next, along `v = v0`, `u = u1`,
    // `v = v1` and `u = u0` in turn, its control points the outer row there.
    let corners = [
        cps[0][0],
        cps[nu - 1][0],
        cps[nu - 1][nv - 1],
        cps[0][nv - 1],
    ];
    let side = |k: usize| -> Vec<Point3> {
        match k {
            0 => (0..nu).map(|i| cps[i][0]).collect(),
            1 => (0..nv).map(|j| cps[nu - 1][j]).collect(),
            2 => (0..nu).map(|i| cps[i][nv - 1]).collect(),
            _ => (0..nv).map(|j| cps[0][j]).collect(),
        }
    };
    let ws = s.weights();
    // Matched to the shortest side: a tolerance taken from the whole patch
    // would let a long thin one's trim pass for its side. A closed band's
    // sides along `v` start and end at one corner: its seam is the measure.
    let shortest = if closed_v {
        (corners[1] - corners[0]).length()
    } else {
        (0..4)
            .map(|k| (corners[(k + 1) % 4] - corners[k]).length())
            .fold(f64::INFINITY, f64::min)
    };
    if !shortest.is_finite() || shortest <= 0.0 {
        return Ok(None);
    }
    let reach = 1e-9 * shortest;
    let on_segment = |p: Point3, a: Point3, b: Point3| {
        let (ab, ap) = (b - a, p - a);
        let w = (ap.dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
        (ap - ab * w).length() <= reach
    };
    let corner_of = |p: Point3| corners.iter().position(|&c| (c - p).length() <= reach);
    let is_side = |c: &NurbsCurve, k: usize| is_side(s, c, k, reach);
    let mut covered = [false; 4];
    for oe in topo.wire(face.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (Some(i), Some(j)) = (corner_of(a), corner_of(b)) else {
            return Ok(None);
        };
        let k = if closed_v {
            // Corners 0 and 3 are one point, and 1 and 2: each closed side
            // starts and ends at its own, and the seam runs between them as
            // side 0 one way and side 2 the other.
            match (i, j) {
                (0, 0) => 3,
                (1, 1) => 1,
                (0, 1) | (1, 0) if !covered[0] => 0,
                (0, 1) | (1, 0) if !covered[2] => 2,
                _ => return Ok(None),
            }
        } else if (i + 1) % 4 == j {
            i
        } else if (j + 1) % 4 == i {
            j
        } else {
            return Ok(None);
        };
        // A conic is read as the NURBS over its own span, from its start.
        let shown = match edge.curve() {
            EdgeCurve::NurbsCurve(c) => is_side(c, k),
            EdgeCurve::Circle(c) => {
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                brepkit_geometry::convert::circle_to_nurbs(c, t0, t1).is_ok_and(|n| is_side(&n, k))
            }
            EdgeCurve::Ellipse(e) => {
                let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                brepkit_geometry::convert::ellipse_to_nurbs(e, t0, t1).is_ok_and(|n| is_side(&n, k))
            }
            EdgeCurve::Line => side(k).iter().all(|&p| on_segment(p, a, b)),
        };
        if !shown {
            return Ok(None);
        }
        covered[k] = true;
    }
    if !covered.iter().all(|&c| c) {
        return Ok(None);
    }
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    // Every distinct knot cuts: a span however short can hold a share of
    // the surface.
    let cuts = |knots: &[f64], lo: f64, hi: f64| {
        let mut c: Vec<f64> = std::iter::once(lo)
            .chain(knots.iter().copied().filter(|&k| k > lo && k < hi))
            .chain(std::iter::once(hi))
            .collect();
        c.dedup_by(|a, b| a.to_bits() == b.to_bits());
        c
    };
    let (u_cuts, v_cuts) = (cuts(s.knots_u(), u0, u1), cuts(s.knots_v(), v0, v1));
    // How often a panel may be halved before its halves come within a
    // thousand units in the last place of its parameters, where the rule's
    // points round onto each other and a halving reads the same panel twice.
    // A panel too narrow to halve four times is not read here.
    let halvings = |a: f64, b: f64| {
        let ulp = f64::EPSILON * a.abs().max(b.abs()).max(f64::MIN_POSITIVE);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let depth = ((b - a) / (1024.0 * ulp)).log2().floor().max(0.0) as u32;
        depth.min(DEPTH)
    };
    if u_cuts
        .windows(2)
        .chain(v_cuts.windows(2))
        .any(|w| halvings(w[0], w[1]) < 4)
    {
        return Ok(None);
    }
    let size = (corners[2] - corners[0])
        .length()
        .max((corners[1] - corners[3]).length());
    let floor = 1e-14 * size.powi(3);
    // Along a direction its weights hold constant, the volume's integrand is
    // a polynomial of a few times the surface's degree, which a rule a little
    // above the degree takes exactly. Along one they vary it is no
    // polynomial, and a wide rule settles each panel in one step: across a
    // quarter arc eight points leave 4e-10 of it, sixteen far less.
    let even_along_u = (0..nv).all(|j| ws.iter().all(|row| row[j].to_bits() == ws[0][j].to_bits()));
    let even_along_v = ws
        .iter()
        .all(|row| row.iter().all(|w| w.to_bits() == row[0].to_bits()));
    let polynomial_order = (2 + s.degree_u().max(s.degree_v())).min(8);
    let order = |even: bool| if even { polynomial_order } else { 16 };
    let (gauss_u, gauss_v) = (
        gauss_legendre_points(order(even_along_u)),
        gauss_legendre_points(order(even_along_v)),
    );
    let mut across = |v: f64| -> Option<[f64; 8]> {
        let mut acc = [0.0; 8];
        for w in u_cuts.windows(2) {
            // A sample that is no number would pass `adaptive`'s maxima.
            let mut g = |u: f64| {
                let sample = nurbs_integrand(s, u, v, about);
                sample.iter().all(|x| x.is_finite()).then_some(sample)
            };
            add(
                &mut acc,
                &adaptive(&mut g, w[0], w[1], gauss_u, halvings(w[0], w[1]), floor)?,
                1.0,
            );
        }
        Some(acc)
    };
    let mut total = [0.0; 8];
    for w in v_cuts.windows(2) {
        let Some(panel) = adaptive(
            &mut across,
            w[0],
            w[1],
            gauss_v,
            halvings(w[0], w[1]),
            floor,
        ) else {
            return Ok(None);
        };
        add(&mut total, &panel, 1.0);
    }
    if total[0] <= 1e-12 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// A face on a NURBS surface, its integrals by Green's theorem as
/// [`integrate_by_boundary`] takes them, with `F(u, v) = ∫ g(s, v) ds` from
/// the start of `u`'s domain in panels between the surface's knots, and
/// each edge cut where it crosses a knot line, so every panel's integrand
/// is smooth.
///
/// A surface that closes on itself along one direction (a wall turned in
/// B-spline form) is cut open along its seam. Each wire's coordinate that way
/// is carried on across the seam, which is a knot line too, so the wire's
/// path in `(u, v)` closes up as on an open surface. Across `u`, each period
/// on adds a whole turn's integral, `F(u + P, v) = F(u, v) + F(u₁, v)`;
/// across `v`, `F` repeats as `v` does.
///
/// `None` for a surface that closes both ways or shrinks to a point along a
/// side (its `(u, v)` would not be one-to-one), a wire that winds round the
/// closed direction, a wire a point of which does not project, or a region
/// of no area.
pub(super) fn nurbs_face_by_boundary(
    topo: &Topology,
    face: &Face,
    sign: f64,
    about: Vec3,
) -> Result<Option<FaceContribution>, CheckError> {
    use brepkit_math::nurbs::projection::{
        project_point_to_surface, project_point_to_surface_from,
    };
    const DEPTH: u32 = 24;
    let FaceSurface::Nurbs(s) = face.surface() else {
        return Ok(None);
    };
    let Some(closure) = closure(s) else {
        return Ok(None);
    };
    let Some(wires) = boundary_wires(topo, face)? else {
        return Ok(None);
    };
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    let interior = |knots: &[f64], lo: f64, hi: f64| {
        let mut k: Vec<f64> = knots
            .iter()
            .copied()
            .filter(|&k| k > lo && k < hi)
            .collect();
        k.dedup_by(|a, b| (*a - *b).abs() <= 1e-12 * (hi - lo));
        k
    };
    let (u_knots, v_knots) = (interior(s.knots_u(), u0, u1), interior(s.knots_v(), v0, v1));
    let size = (s.evaluate(u1, v1) - s.evaluate(u0, v0))
        .length()
        .max((s.evaluate(u1, v0) - s.evaluate(u0, v1)).length());
    let reach = 1e-6 * size.max(1.0);
    // Where an integrand vanishes (along an edge at the start of `u`) its
    // halves agree only to rounding.
    let floor = 1e-14 * size.powi(3);
    // Each integral is halved until it settles: a rational surface can
    // crowd its area into a sliver of its parameter.
    let gauss_u = gauss_legendre_points(10);
    // A whole knot span is integrated at the same `u` for every point of the
    // boundary, unless its integral halves past the first step: its points'
    // basis functions in `u` are worked out once, and taken where a point is
    // the very same number.
    let span_points: Vec<std::sync::OnceLock<(Vec<u64>, Vec<UBasis>)>> =
        u_knots.iter().map(|_| std::sync::OnceLock::new()).collect();
    let points_of = |i: usize, from: f64, to: f64| {
        let span_from = if i == 0 { u0 } else { u_knots[i - 1] };
        if i >= u_knots.len()
            || from.to_bits() != span_from.to_bits()
            || to.to_bits() != u_knots[i].to_bits()
        {
            return None;
        }
        Some(span_points[i].get_or_init(|| {
            let mut us = Vec::new();
            let mut record = |x: f64| {
                us.push(x);
                Some([0.0; 8])
            };
            adaptive(&mut record, from, to, gauss_u, 0, 0.0);
            let bases: Vec<UBasis> = us.iter().filter_map(|&x| s.u_basis(x)).collect();
            (bases.iter().map(|b| b.u.to_bits()).collect(), bases)
        }))
    };
    let antiderivative = |u: f64, v: f64| -> Option<[f64; 8]> {
        // Every point of the integral lies on one line of constant `v`.
        let row = s.at_v(v);
        let mut acc = [0.0; 8];
        let mut from = u0;
        for (i, to) in u_knots
            .iter()
            .copied()
            .filter(|&k| k < u)
            .chain([u])
            .enumerate()
        {
            if to > from {
                let points = points_of(i, from, to);
                let mut g = |s_u: f64| {
                    let known = points.and_then(|(bits, bases)| {
                        let at = bits.iter().position(|&b| b == s_u.to_bits())?;
                        Some(&bases[at])
                    });
                    Some(match known {
                        Some(b) => terms_at(row.point_and_partials_with(b), about),
                        None => row_integrand(&row, s_u, about),
                    })
                };
                add(
                    &mut acc,
                    &adaptive(&mut g, from, to, gauss_u, DEPTH, floor)?,
                    1.0,
                );
            }
            from = to;
        }
        Some(acc)
    };
    let gauss = gauss_legendre_points(8);
    // A point's `(u, v)` by Newton from a neighbour's where that lands on the
    // surface, else by a search of the whole surface.
    let on = |edge: &BoundaryEdge<'_>, t: f64, seed: Option<(f64, f64)>| -> Option<(f64, f64)> {
        let p = edge.curve.evaluate_with_endpoints(t, edge.start, edge.end);
        // Far from the origin a point's own rounding passes a fixed
        // tolerance: Newton could not settle, and every point of every edge
        // would take the search.
        let tolerance = 1e-12_f64.max(2e-14 * p.x().abs().max(p.y().abs()).max(p.z().abs()));
        if let Some(seed) = seed
            && let Ok(q) = project_point_to_surface_from(s, p, seed, tolerance)
            && q.distance <= reach
        {
            return Some((q.u, q.v));
        }
        project_point_to_surface(s, p, tolerance)
            .ok()
            .filter(|q| q.distance <= reach)
            .map(|q| (q.u, q.v))
    };
    // Every edge cut into pieces, with the wire it is in: each piece is
    // integrated on its own, from its own seed, so the pieces can be taken at
    // once and summed in their order after.
    // An edge that is the side `v = v0` or `v = v1` of the surface has
    // `dv = 0` all along it, so it adds nothing to `∮ F dv`: its integrand is
    // rounding alone, which the adaptive rule would halve to its depth. Sides
    // are matched as `nurbs_face_over_domain` matches them, to the shortest
    // side of the control net.
    let side_reach = {
        let cps = s.control_points();
        let (nu, nv) = (cps.len(), cps.first().map_or(0, Vec::len));
        if nu < 2 || nv < 2 {
            0.0
        } else {
            let corners = [
                cps[0][0],
                cps[nu - 1][0],
                cps[nu - 1][nv - 1],
                cps[0][nv - 1],
            ];
            // A side closed on itself starts and ends at one corner: it
            // is no measure.
            1e-9 * (0..4)
                .map(|k| (corners[(k + 1) % 4] - corners[k]).length())
                .filter(|&d| d > 0.0)
                .fold(f64::INFINITY, f64::min)
        }
    };
    let along_v_side = |curve: &EdgeCurve| {
        side_reach.is_finite()
            && side_reach > 0.0
            && matches!(curve, EdgeCurve::NurbsCurve(c)
                if is_side(s, c, 0, side_reach) || is_side(s, c, 2, side_reach))
    };
    // On a surface closed along one direction, the coordinate that way is
    // carried on along each wire across the seam: a point's copy of it a
    // whole number of periods on from the last point's.
    let closed = match closure {
        Closure::Open => None,
        Closure::AlongU => Some((0_usize, u1 - u0, u0)),
        Closure::AlongV => Some((1_usize, v1 - v0, v0)),
    };
    let coord = |q: (f64, f64), pick: usize| if pick == 0 { q.0 } else { q.1 };
    let lift = |near: f64, x: f64, period: f64| period.madd(((near - x) / period).round(), x);
    let mut pieces: Vec<Piece<'_, '_>> = Vec::new();
    for (k, wire) in wires.iter().enumerate() {
        let first = pieces.len();
        // The closed coordinate where the wire starts and where it stands.
        let (mut began, mut run): (Option<f64>, Option<f64>) = (None, None);
        for edge in wire {
            // On a closed surface a side is followed all the same: it
            // carries the closed coordinate a turn round.
            let side = along_v_side(edge.curve);
            if side && closed.is_none() {
                continue;
            }
            let on = |t: f64, seed: Option<(f64, f64)>| on(edge, t, seed);
            for (ta, tb) in edge_pieces(edge) {
                // Where the edge crosses a knot line the integrand bends:
                // cut there, found by bisection between samples astride it.
                let mut samples: Vec<(f64, (f64, f64))> = Vec::with_capacity(9);
                let mut seed = None;
                for k in 0..=8 {
                    let t = (tb - ta).madd(f64::from(k) / 8.0, ta);
                    let Some(q) = on(t, seed) else {
                        return Ok(None);
                    };
                    samples.push((t, q));
                    seed = Some(q);
                }
                let mut lifted: Vec<f64> = Vec::with_capacity(9);
                if let Some((pick, period, _)) = closed {
                    for &(_, q) in &samples {
                        let x = coord(q, pick);
                        let at = run.map_or(x, |r| lift(r, x, period));
                        // A step that long could as well run the other way.
                        if run.is_some_and(|r| (at - r).abs() > 0.25 * period) {
                            return Ok(None);
                        }
                        began.get_or_insert(at);
                        run = Some(at);
                        lifted.push(at);
                    }
                }
                if side {
                    continue;
                }
                let mut cuts = vec![ta, tb];
                // A sample on a knot line is a cut itself; the bisection
                // below only takes a knot strictly between two samples.
                let (u_span, v_span) = (u1 - u0, v1 - v0);
                for &(t, q) in &samples {
                    let on_knot = u_knots.iter().any(|&k| (q.0 - k).abs() <= 1e-12 * u_span)
                        || v_knots.iter().any(|&k| (q.1 - k).abs() <= 1e-12 * v_span);
                    if on_knot {
                        cuts.push(t);
                    }
                }
                for (i, w) in samples.windows(2).enumerate() {
                    let ((t_a, a), (t_b, b)) = (w[0], w[1]);
                    for (knots, pick) in [(&u_knots, 0_usize), (&v_knots, 1_usize)] {
                        // Along the closed direction the knot lines repeat
                        // every period, the seam among them, and are found on
                        // the coordinate carried on.
                        if let Some((_, period, seam)) = closed.filter(|c| c.0 == pick) {
                            let (la, lb) = (lifted[i], lifted[i + 1]);
                            let (low, high) = (la.min(lb), la.max(lb));
                            for &line in knots.iter().chain([&seam]) {
                                #[allow(clippy::cast_possible_truncation)]
                                let (first, last) = (
                                    ((low - line) / period).floor() as i64,
                                    ((high - line) / period).ceil() as i64,
                                );
                                for m in first..=last {
                                    #[allow(clippy::cast_precision_loss)]
                                    let x = period.madd(m as f64, line);
                                    if x <= low || x >= high {
                                        continue;
                                    }
                                    let (mut lo, mut hi) = (t_a, t_b);
                                    for _ in 0..50 {
                                        let mid = 0.5 * (lo + hi);
                                        let Some(q) = on(mid, Some(a)) else {
                                            return Ok(None);
                                        };
                                        if (lift(la, coord(q, pick), period) - x) * (la - x) > 0.0 {
                                            lo = mid;
                                        } else {
                                            hi = mid;
                                        }
                                    }
                                    cuts.push(0.5 * (lo + hi));
                                }
                            }
                            continue;
                        }
                        for &k in knots {
                            if (coord(a, pick) - k) * (coord(b, pick) - k) >= 0.0 {
                                continue;
                            }
                            let (mut lo, mut hi) = (t_a, t_b);
                            for _ in 0..50 {
                                let mid = 0.5 * (lo + hi);
                                let Some(q) = on(mid, Some(a)) else {
                                    return Ok(None);
                                };
                                if (coord(q, pick) - k) * (coord(a, pick) - k) > 0.0 {
                                    lo = mid;
                                } else {
                                    hi = mid;
                                }
                            }
                            cuts.push(0.5 * (lo + hi));
                        }
                    }
                }
                // A piece's points are read on the copy of the closed
                // coordinate nearest a sample of it: one that turns far is
                // cut at every sample, so each piece stays within a quarter
                // turn of it.
                if let Some((_, period, _)) = closed {
                    let (lo, hi) = lifted
                        .iter()
                        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &x| {
                            (lo.min(x), hi.max(x))
                        });
                    if hi - lo > 0.125 * period {
                        cuts.extend(samples[1..8].iter().map(|&(t, _)| t));
                    }
                }
                cuts.sort_by(f64::total_cmp);
                if tb < ta {
                    cuts.reverse();
                }
                cuts.dedup_by(|x, y| (*x - *y).abs() <= 1e-12 * (tb - ta).abs());
                pieces.extend(cuts.windows(2).map(|w| {
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let nearest = (8.0 * (f64::midpoint(w[0], w[1]) - ta) / (tb - ta))
                        .round()
                        .clamp(0.0, 8.0) as usize;
                    Piece {
                        wire: k,
                        edge,
                        from: w[0],
                        to: w[1],
                        start: samples[nearest].1,
                        lifted: lifted.get(nearest).copied().unwrap_or(0.0),
                    }
                }));
            }
        }
        // A wire that winds round the closed direction bounds a band with
        // another: which side of it the face lies on its area cannot tell.
        if let (Some((_, period, _)), Some(b), Some(r)) = (closed, began, run)
            && ((r - b) / period).abs() > 1e-6
        {
            return Ok(None);
        }
        // A wire that closes up may be read on any copy of the domain, a
        // whole turn on along `u` adding `∮ F(u₁, v) dv = 0`: the copy it
        // mostly lies in takes the fewest whole turns.
        if let Some((0, period, seam)) = closed {
            let wire_pieces = &mut pieces[first..];
            #[allow(clippy::cast_precision_loss)]
            let mean =
                wire_pieces.iter().map(|p| p.lifted).sum::<f64>() / wire_pieces.len().max(1) as f64;
            let shift = period * ((mean - seam) / period).floor();
            for piece in wire_pieces {
                piece.lifted -= shift;
            }
        }
    }
    // One piece failing fails the face: pieces not yet begun are passed over.
    let failed = std::sync::atomic::AtomicBool::new(false);
    let integrate = |piece: &Piece<'_, '_>| -> Option<[f64; 8]> {
        use std::sync::atomic::Ordering::Relaxed;
        if failed.load(Relaxed) {
            return None;
        }
        let edge = piece.edge;
        let mut seed = piece.start;
        let mut along = |t: f64| -> Option<[f64; 8]> {
            let c = on(edge, t, Some(seed))?;
            seed = c;
            // `dv/dt` from the curve's velocity in the surface's tangent
            // plane, by the first fundamental form.
            let velocity = edge_velocity(edge.curve, t, edge.start, edge.end);
            let d = s.first_derivatives(c.0, c.1);
            let (su, sv) = (d.1, d.2);
            let (e, f, g) = (su.dot(su), su.dot(sv), sv.dot(sv));
            let det = e.madd(g, -f * f);
            if det <= 1e-24 * (e * g).max(1e-300) {
                return None;
            }
            let dv = e.madd(sv.dot(velocity), -f * su.dot(velocity)) / det;
            let mut term = antiderivative(c.0, c.1)?;
            // Each turn on round `u` adds the integral of a whole turn.
            if let Some((0, period, _)) = closed {
                let turns = ((piece.lifted - c.0) / period).round();
                if turns != 0.0 {
                    let whole = antiderivative(u1, c.1)?;
                    for (x, w) in term.iter_mut().zip(whole) {
                        *x = turns.madd(w, *x);
                    }
                }
            }
            for x in &mut term {
                *x *= dv;
            }
            Some(term)
        };
        let integral = adaptive(&mut along, piece.from, piece.to, gauss, DEPTH, floor);
        if integral.is_none() {
            failed.store(true, Relaxed);
        }
        integral
    };
    #[cfg(not(target_arch = "wasm32"))]
    let integrals: Vec<Option<[f64; 8]>> = {
        use rayon::prelude::*;
        pieces.par_iter().map(integrate).collect()
    };
    #[cfg(target_arch = "wasm32")]
    let integrals: Vec<Option<[f64; 8]>> = pieces.iter().map(integrate).collect();
    let mut total = [0.0; 8];
    // The pieces lie in the order of their wires, so one pass meets each
    // wire's run of them in turn.
    let mut rest = pieces.iter().zip(&integrals).peekable();
    for k in 0..wires.len() {
        let mut region = [0.0; 8];
        while let Some((_, integral)) = rest.next_if(|(piece, _)| piece.wire == k) {
            let Some(integral) = integral else {
                return Ok(None);
            };
            add(&mut region, integral, 1.0);
        }
        // Each wire counts the region it bounds, whichever way it runs in
        // `(u, v)`; holes are taken away.
        let side = region[0].signum() * if k == 0 { 1.0 } else { -1.0 };
        add(&mut total, &region, side);
    }
    if total[0] <= 1e-12 {
        return Ok(None);
    }
    Ok(Some(FaceContribution {
        area: total[0],
        volume: total[1] * sign,
        volume_moment_x: total[2] * sign,
        volume_moment_y: total[3] * sign,
        volume_moment_z: total[4] * sign,
        centroid_x: total[5],
        centroid_y: total[6],
        centroid_z: total[7],
    }))
}

/// A stretch of a boundary edge between two cuts, and where its first point
/// lies in `(u, v)`.
struct Piece<'e, 'a> {
    wire: usize,
    edge: &'e BoundaryEdge<'a>,
    from: f64,
    to: f64,
    start: (f64, f64),
    /// On a surface closed along one direction, the coordinate that way
    /// carried on along the wire to the sample `start` is read at.
    lifted: f64,
}

/// How the boundary integrals walk an edge: by `τ`, its parameter less
/// `base`. A conic (`base` its parameter at its start vertex) is read as its
/// point there plus its chord from there, by the half-angle forms
/// `cos(b + τ) − cos b = −2 sin(b + τ/2) sin(τ/2)` and
/// `sin(b + τ) − sin b = 2 cos(b + τ/2) sin(τ/2)`. The conic's own form
/// rounds the angle and the centre's coordinates: a short arc of an ellipse
/// 2e8 across has its centre 1e8 away, and one rounding of its angle moves
/// its point 4e-8 along it. The chord keeps its digits, and two faces on one
/// edge walk the same points whichever way each runs it. Any other curve is
/// read at its own parameter (`base` is nothing).
pub(super) struct Walk<'a> {
    curve: &'a EdgeCurve,
    start: Point3,
    end: Point3,
    base: f64,
    at_base: Point3,
}

impl<'a> Walk<'a> {
    pub(super) fn new(curve: &'a EdgeCurve, start: Point3, end: Point3) -> Self {
        let base = match curve {
            EdgeCurve::Circle(c) => c.project(start),
            EdgeCurve::Ellipse(e) => e.project(start),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => 0.0,
        };
        let at_base = curve.evaluate_with_endpoints(base, start, end);
        Self {
            curve,
            start,
            end,
            base,
            at_base,
        }
    }

    /// `τ` at the curve's parameter `t`.
    pub(super) fn tau(&self, t: f64) -> f64 {
        t - self.base
    }

    /// The point at `τ`.
    pub(super) fn point(&self, tau: f64) -> Point3 {
        let (major, minor, x, y) = match self.curve {
            EdgeCurve::Circle(c) => (c.radius(), c.radius(), c.u_axis(), c.v_axis()),
            EdgeCurve::Ellipse(e) => (e.semi_major(), e.semi_minor(), e.u_axis(), e.v_axis()),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => {
                return self
                    .curve
                    .evaluate_with_endpoints(self.base + tau, self.start, self.end);
            }
        };
        let half = 0.5 * tau;
        let s = half.sin();
        let (sin_m, cos_m) = (self.base + half).sin_cos();
        self.at_base + x * (-2.0 * major * sin_m * s) + y * (2.0 * minor * cos_m * s)
    }

    /// The derivative in `τ` at `τ`.
    pub(super) fn velocity(&self, tau: f64) -> Vec3 {
        edge_velocity(self.curve, self.base + tau, self.start, self.end)
    }
}

/// `d/dt` of `curve` at `t` as `evaluate_with_endpoints` runs it from
/// `start` to `end`.
fn edge_velocity(curve: &EdgeCurve, t: f64, start: Point3, end: Point3) -> Vec3 {
    match curve {
        EdgeCurve::Line => end - start,
        EdgeCurve::Circle(c) => c.tangent(t) * c.radius(),
        EdgeCurve::Ellipse(e) => e.tangent(t),
        EdgeCurve::NurbsCurve(n) => n.derivatives(t, 1)[1],
    }
}

/// Which way a NURBS surface closes on itself.
#[derive(Clone, Copy)]
enum Closure {
    Open,
    AlongU,
    AlongV,
}

/// Which way a NURBS surface that shrinks to a point along no side closes
/// on itself; `None` for one that closes both ways or has a pole.
fn closure(s: &brepkit_math::nurbs::surface::NurbsSurface) -> Option<Closure> {
    let (sides, near) = side_samples(s);
    let length = |pts: &[Point3]| pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>();
    let same =
        |a: &[Point3], b: &[Point3]| a.iter().zip(b).all(|(p, q)| (*p - *q).length() <= near);
    if !sides.iter().all(|p| length(p) > near) {
        return None;
    }
    match (same(&sides[0], &sides[1]), same(&sides[2], &sides[3])) {
        (false, false) => Some(Closure::Open),
        (true, false) => Some(Closure::AlongU),
        (false, true) => Some(Closure::AlongV),
        (true, true) => None,
    }
}

/// Whether a NURBS surface neither closes on itself along `u` or `v` nor
/// shrinks to a point along a side.
fn open_and_poleless(s: &brepkit_math::nurbs::surface::NurbsSurface) -> bool {
    let (sides, near) = side_samples(s);
    let length = |pts: &[Point3]| pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>();
    let same =
        |a: &[Point3], b: &[Point3]| a.iter().zip(b).all(|(p, q)| (*p - *q).length() <= near);
    !same(&sides[0], &sides[1])
        && !same(&sides[2], &sides[3])
        && sides.iter().all(|p| length(p) > near)
}

/// Whether a NURBS surface closes on itself along `v` alone, its first and
/// last rows across `u` one curve (a band round a closed rim), and shrinks
/// to a point along no side.
fn closed_along_v_and_poleless(s: &brepkit_math::nurbs::surface::NurbsSurface) -> bool {
    let (sides, near) = side_samples(s);
    let length = |pts: &[Point3]| pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>();
    let same =
        |a: &[Point3], b: &[Point3]| a.iter().zip(b).all(|(p, q)| (*p - *q).length() <= near);
    !same(&sides[0], &sides[1])
        && same(&sides[2], &sides[3])
        && sides.iter().all(|p| length(p) > near)
}

/// Samples along each side of a NURBS surface's domain, `u = u0`, `u = u1`,
/// `v = v0` and `v = v1` in turn, and the distance within which two points
/// of it are one; an empty list and no distance for a surface of no size.
fn side_samples(s: &brepkit_math::nurbs::surface::NurbsSurface) -> ([Vec<Point3>; 4], f64) {
    const N: u32 = 8;
    let (u0, u1) = s.domain_u();
    let (v0, v1) = s.domain_v();
    let lerp = |a: f64, b: f64, k: u32| (b - a).madd(f64::from(k) / f64::from(N), a);
    let side = |along_u: bool, at: f64| -> Vec<Point3> {
        (0..=N)
            .map(|k| {
                if along_u {
                    s.evaluate(lerp(u0, u1, k), at)
                } else {
                    s.evaluate(at, lerp(v0, v1, k))
                }
            })
            .collect()
    };
    let length = |pts: &[Point3]| pts.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>();
    let sides = [
        side(false, u0),
        side(false, u1),
        side(true, v0),
        side(true, v1),
    ];
    let size = sides.iter().map(|p| length(p)).fold(0.0_f64, f64::max);
    let near = 1e-9 * size.max(1.0);
    if size > near {
        (sides, near)
    } else {
        (
            [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            f64::INFINITY,
        )
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::surfaces::{ConicalSurface, CylindricalSurface, ToroidalSurface};

    use super::*;

    /// `∫₀ᵗᵘʳⁿ g(s, v) ds` by the trapezoid rule, exact for the integrands of
    /// a surface of revolution.
    fn turn_by_trapezoid<S: ParametricSurface>(surface: &S, v: f64, about: Vec3) -> [f64; 8] {
        const N: u32 = 32;
        let mut acc = [0.0; 8];
        for k in 0..N {
            let s = TAU * f64::from(k) / f64::from(N);
            add(
                &mut acc,
                &integrand(surface, s, v, about),
                TAU / f64::from(N),
            );
        }
        acc
    }

    /// `∫₀ᵘ g(s, v) ds` by quadrature: whole turns toward zero by the
    /// trapezoid rule, the rest by Gauss-Legendre in panels of at most a
    /// quarter turn, so a small `u` of either sign reads its own sliver.
    fn by_quadrature<S: ParametricSurface>(surface: &S, u: f64, v: f64, about: Vec3) -> [f64; 8] {
        let turns = (u / TAU).trunc();
        let rest = u - turns * TAU;
        let mut acc = [0.0; 8];
        if turns != 0.0 {
            add(&mut acc, &turn_by_trapezoid(surface, v, about), turns);
        }
        let gauss = gauss_legendre_points(10);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let panels = (rest.abs() / std::f64::consts::FRAC_PI_2).ceil().max(1.0) as u32;
        let width = rest / f64::from(panels);
        for i in 0..panels {
            let mid = width * (f64::from(i) + 0.5);
            for gp in gauss {
                let s = (0.5 * width).madd(gp.x, mid);
                add(
                    &mut acc,
                    &integrand(surface, s, v, about),
                    0.5 * width * gp.w,
                );
            }
        }
        acc
    }

    /// A fixed sequence of numbers in `[0, 1)`.
    struct Draws(u64);

    impl Draws {
        fn next(&mut self) -> f64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            #[allow(clippy::cast_precision_loss)]
            let x = (self.0 >> 11) as f64 / (1_u64 << 53) as f64;
            x
        }

        fn within(&mut self, lo: f64, hi: f64) -> f64 {
            (hi - lo).madd(self.next(), lo)
        }

        fn direction(&mut self) -> Vec3 {
            loop {
                let w = Vec3::new(
                    self.within(-1.0, 1.0),
                    self.within(-1.0, 1.0),
                    self.within(-1.0, 1.0),
                );
                if (0.1..1.0).contains(&w.length()) {
                    return w;
                }
            }
        }

        fn point(&mut self, reach: f64) -> Point3 {
            let w = self.direction() * reach;
            Point3::new(w.x(), w.y(), w.z())
        }
    }

    /// The largest gap between the closed form and the quadrature at
    /// `(u, v)`, against the largest integrand over the turn times the
    /// length of `u`'s run, and the closed form's mean against the
    /// trapezoid rule's.
    fn gap<S: ParametricSurface>(
        surface: &S,
        revolution: &Revolution,
        u: f64,
        v: f64,
        about: Vec3,
    ) -> f64 {
        let size = (0..64)
            .flat_map(|k| integrand(surface, TAU * f64::from(k) / 64.0, v, about))
            .fold(f64::MIN_POSITIVE, |m, x| m.max(x.abs()));
        let closed = revolution.antiderivative(u, v, about);
        let quadrature = by_quadrature(surface, u, v, about);
        let mean = revolution.mean(v, about);
        let turn = turn_by_trapezoid(surface, v, about);
        let along = closed
            .iter()
            .zip(&quadrature)
            .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
        let over_turn = mean
            .iter()
            .zip(&turn)
            .fold(0.0_f64, |m, (a, b)| m.max((TAU * a - b).abs()));
        (along / (size * u.abs().max(1.0))).max(over_turn / (size * TAU))
    }

    /// The largest gap between the closed form and the quadrature at a small
    /// `u`, each integral against its own integrand's largest size times
    /// `|u|`: a sliver of a face is the whole of what it holds.
    fn sliver_gap<S: ParametricSurface>(
        surface: &S,
        revolution: &Revolution,
        u: f64,
        v: f64,
        about: Vec3,
    ) -> f64 {
        let mut size = [0.0_f64; 8];
        for k in 0..64 {
            let g = integrand(surface, TAU * f64::from(k) / 64.0, v, about);
            for (m, x) in size.iter_mut().zip(g) {
                *m = m.max(x.abs());
            }
        }
        let closed = revolution.antiderivative(u, v, about);
        let quadrature = by_quadrature(surface, u, v, about);
        (0..8)
            .filter(|&k| size[k] > 0.0)
            .map(|k| (closed[k] - quadrature[k]).abs() / (size[k] * u.abs()))
            .fold(0.0, f64::max)
    }

    #[test]
    fn closed_form_antiderivative_keeps_a_sliver_of_a_turn() {
        // A cylinder of radius 1e6 about z through the origin: over `u` up
        // to 1e-8, its area moment along y is `r² (1 - cos u)`, all of it of
        // second order in `u`.
        let origin = Point3::new(0.0, 0.0, 0.0);
        let radius = 1e6;
        let wall = CylindricalSurface::with_ref_dir(
            origin,
            Vec3::new(0.0, 0.0, 1.0),
            radius,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let u = 1e-8;
        let along_y =
            Revolution::cylinder(&wall).antiderivative(u, 0.5, Vec3::new(0.0, 0.0, 0.0))[6];
        let expected = radius * radius * 2.0 * (0.5 * u).sin().powi(2);
        assert!(
            (along_y - expected).abs() <= 1e-12 * expected.abs(),
            "{along_y} against {expected}"
        );

        let mut draws = Draws(0x2545_f491_4f6c_dd1d);
        let mut worst = 0.0_f64;
        for _ in 0..40 {
            let axis = draws.direction();
            let ref_dir = draws.direction();
            let reach = draws.within(0.0, 50.0);
            let centre = draws.point(reach);
            let radius = draws.within(0.1, 20.0);
            let about = draws.direction() * 30.0;
            let cylinder = CylindricalSurface::with_ref_dir(centre, axis, radius, ref_dir).unwrap();
            let cone = ConicalSurface::with_ref_dir(centre, axis, draws.within(0.05, 1.5), ref_dir)
                .unwrap();
            let sphere =
                SphericalSurface::with_axis_and_ref_dir(centre, radius, axis, ref_dir).unwrap();
            let torus = ToroidalSurface::with_axis_and_ref_dir(
                centre,
                radius,
                radius * draws.within(0.05, 0.9),
                axis,
                ref_dir,
            )
            .unwrap();
            let (along, latitude, tube) = (
                draws.within(1.0, 30.0),
                draws.within(-1.5, 1.5),
                draws.within(-TAU, TAU),
            );
            for u in [1e-12, -1e-12, 1e-8, -1e-8, 1e-5, -1e-5, 1e-3, -1e-3] {
                for about in [Vec3::new(0.0, 0.0, 0.0), about] {
                    worst = worst
                        .max(sliver_gap(
                            &cylinder,
                            &Revolution::cylinder(&cylinder),
                            u,
                            along,
                            about,
                        ))
                        .max(sliver_gap(&cone, &Revolution::cone(&cone), u, along, about))
                        .max(sliver_gap(
                            &sphere,
                            &Revolution::sphere(&sphere),
                            u,
                            latitude,
                            about,
                        ))
                        .max(sliver_gap(
                            &torus,
                            &Revolution::torus(&torus),
                            u,
                            tube,
                            about,
                        ));
                }
            }
        }
        assert!(worst < 2e-14, "worst relative gap {worst:e}");
    }

    #[test]
    fn closed_form_antiderivative_agrees_with_quadrature() {
        let mut draws = Draws(0x9e37_79b9_7f4a_7c15);
        let mut worst = 0.0_f64;
        let mut cases = 0;
        for _ in 0..40 {
            let axis = draws.direction();
            let ref_dir = draws.direction();
            let reach = draws.within(0.0, 50.0);
            let centre = draws.point(reach);
            let radius = draws.within(0.1, 20.0);
            let abouts = [Vec3::new(0.0, 0.0, 0.0), draws.direction() * 30.0];
            let cylinder = CylindricalSurface::with_ref_dir(centre, axis, radius, ref_dir).unwrap();
            let cone = ConicalSurface::with_ref_dir(centre, axis, draws.within(0.05, 1.5), ref_dir)
                .unwrap();
            let sphere =
                SphericalSurface::with_axis_and_ref_dir(centre, radius, axis, ref_dir).unwrap();
            let torus = ToroidalSurface::with_axis_and_ref_dir(
                centre,
                radius,
                radius * draws.within(0.05, 1.5),
                axis,
                ref_dir,
            )
            .unwrap();
            for _ in 0..25 {
                let u = draws.within(-3.0 * TAU, 3.0 * TAU);
                let along = draws.within(-30.0, 30.0);
                let latitude = draws.within(-1.55, 1.55);
                let tube = draws.within(-TAU, TAU);
                for about in abouts {
                    worst = worst
                        .max(gap(
                            &cylinder,
                            &Revolution::cylinder(&cylinder),
                            u,
                            along,
                            about,
                        ))
                        .max(gap(&cone, &Revolution::cone(&cone), u, along, about))
                        .max(gap(
                            &sphere,
                            &Revolution::sphere(&sphere),
                            u,
                            latitude,
                            about,
                        ))
                        .max(gap(&torus, &Revolution::torus(&torus), u, tube, about));
                    cases += 4;
                }
            }
        }
        assert_eq!(cases, 8000);
        assert!(worst < 2e-14, "worst relative gap {worst:e}");
    }
}

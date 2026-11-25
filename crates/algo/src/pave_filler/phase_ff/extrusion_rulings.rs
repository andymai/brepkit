//! The section of a cylinder and a B-spline extrusion along the cylinder's
//! axis: the lines along the axis through the points where the circle
//! crosses the extrusion's profile.
//!
//! Marched instead, each line came back as a fitted curve that strayed from
//! the straight ruling by up to 1.2e-5 where the walls cross at a shallow
//! angle, and the faces it bounded lost that sliver of volume.

use brepkit_math::aabb::Aabb3;
use brepkit_math::madd::Madd;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::decompose::curve_bezier_between;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::edge::EdgeCurve;

use super::{RawCurve, trim_t_range_to_aabb};

/// Past this sine of the angle between them, the extrusion does not run
/// along the cylinder's axis.
const PARALLEL_SIN: f64 = 1e-12;

/// A parameter interval of the profile this much narrower than its knot
/// span is not split further.
const NARROWEST: f64 = 1e-12;

/// The most intervals one section may read before it is left to the
/// marcher.
const MOST_INTERVALS: usize = 4096;

/// A crossing that rounding hides lies within a run of parameters whose
/// points spread no wider than this, or it is left to the marcher.
const SPREAD: f64 = brepkit_math::tolerance::Tolerance::new().linear;

/// A profile within rounding of the circle over more of its knot span than
/// this runs along the cylinder, and the section is left to the marcher.
const WIDEST_GRAZE: f64 = 1e-3;

/// The rulings where `cylinder` crosses the B-spline extrusion `surface`,
/// trimmed to both faces' boxes and to the extrusion's height. `None` where
/// `surface` is not an extrusion along the cylinder's axis, or where its
/// profile runs along the circle, so the section is not a set of isolated
/// lines.
///
/// A ruling where the circle only touches the profile is left out, as a
/// graze splits neither face. A crossing so shallow that rounding hides the
/// profile's side about it is placed at the middle of the hidden run, or
/// left to the marcher where that run spreads wider than [`SPREAD`].
pub(super) fn cylinder_extrusion_rulings(
    cylinder: &CylindricalSurface,
    surface: &NurbsSurface,
    bbox_a: &Aabb3,
    bbox_b: &Aabb3,
) -> Option<Vec<RawCurve>> {
    let axis = cylinder.axis().normalize().ok()?;
    let (profile, along) = extrusion(surface)?;
    let length = along.length();
    if length <= 0.0 || along.cross(axis).length() > PARALLEL_SIN * length {
        return None;
    }
    let section = CircleSection {
        profile: &profile,
        origin: cylinder.origin(),
        axis,
        radius: cylinder.radius(),
    };
    let roots = section.crossings()?;
    // Each ruling runs from the profile to its copy moved by `along`, past
    // which the extrusion has no surface: a point there projects on to the
    // patch's edge and reads as on the face.
    let rise = along.dot(axis);
    let mut lines = Vec::with_capacity(roots.len());
    for t in roots {
        let base = section.foot(t)?;
        let boxed = trim_t_range_to_aabb(base, axis, bbox_a, bbox_b);
        let t_range = (boxed.0.max(rise.min(0.0)), boxed.1.min(rise.max(0.0)));
        if t_range.1 - t_range.0 < 1e-9 {
            continue;
        }
        let (p0, p1) = (base + axis * t_range.0, base + axis * t_range.1);
        lines.push(RawCurve {
            curve: EdgeCurve::Line,
            bbox: Aabb3::from_points([p0, p1]),
            t_range,
            p_start: p0,
            p_end: p1,
        });
    }
    Some(lines)
}

/// `surface` as a profile moved along a vector: degree one with two control
/// rows (or columns) across one direction, the second row the first moved
/// by one vector, with the same weights.
fn extrusion(surface: &NurbsSurface) -> Option<(NurbsCurve, Vec3)> {
    let cps = surface.control_points();
    let weights = surface.weights();
    let (first, first_w, second, second_w, degree, knots) =
        if surface.degree_u() == 1 && cps.len() == 2 && surface.knots_u().len() == 4 {
            (
                cps[0].clone(),
                weights[0].clone(),
                cps[1].clone(),
                weights[1].clone(),
                surface.degree_v(),
                surface.knots_v().to_vec(),
            )
        } else if surface.degree_v() == 1
            && cps.iter().all(|row| row.len() == 2)
            && surface.knots_v().len() == 4
        {
            (
                cps.iter().map(|row| row[0]).collect(),
                weights.iter().map(|row| row[0]).collect(),
                cps.iter().map(|row| row[1]).collect(),
                weights.iter().map(|row| row[1]).collect(),
                surface.degree_u(),
                surface.knots_u().to_vec(),
            )
        } else {
            return None;
        };
    let (&start, &moved) = (first.first()?, second.first()?);
    let along = moved - start;
    let size = first
        .iter()
        .map(|p| (*p - start).length())
        .fold(along.length(), f64::max);
    let exact = |a: f64, b: f64, scale: f64| (a - b).abs() <= 8.0 * f64::EPSILON * scale;
    let same_move = first.iter().zip(&second).all(|(a, b)| {
        let step = *b - *a;
        exact(step.x(), along.x(), size + a.x().abs())
            && exact(step.y(), along.y(), size + a.y().abs())
            && exact(step.z(), along.z(), size + a.z().abs())
    });
    let same_weights = first_w
        .iter()
        .zip(&second_w)
        .all(|(a, b)| exact(*a, *b, a.abs()));
    if !same_move || !same_weights || first_w.iter().any(|w| *w <= 0.0) {
        return None;
    }
    let profile = NurbsCurve::new(degree, knots, first, first_w).ok()?;
    Some((profile, along))
}

/// A profile curve against the circle a cylinder cuts across its axis.
struct CircleSection<'a> {
    profile: &'a NurbsCurve,
    origin: Point3,
    axis: Vec3,
    radius: f64,
}

impl CircleSection<'_> {
    /// Where `point` lies across the axis from the cylinder's origin.
    fn across(&self, point: Point3) -> Vec3 {
        let w = point - self.origin;
        w - self.axis * w.dot(self.axis)
    }

    /// The squared distance of the profile's point at `t` from the axis,
    /// less the squared radius: negative inside the cylinder.
    fn reach(&self, t: f64) -> f64 {
        let w = self.across(self.profile.evaluate(t));
        w.dot(w) - self.radius * self.radius
    }

    /// The point on the cylinder where the profile at `t` crosses it, set
    /// on the circle across from the profile's point.
    fn foot(&self, t: f64) -> Option<Point3> {
        let point = self.profile.evaluate(t);
        let w = point - self.origin;
        let height = w.dot(self.axis);
        let out = self.across(point).normalize().ok()?;
        Some(self.origin + self.axis * height + out * self.radius)
    }

    /// The profile's parameters where it crosses the circle, in order.
    /// `None` when the crossings are not isolated within
    /// [`MOST_INTERVALS`] intervals, where the profile runs within rounding
    /// of the circle over more than [`WIDEST_GRAZE`] of a span, or where a
    /// crossing hidden by rounding is not placed within [`SPREAD`].
    ///
    /// The intervals are read in order. One with no root, or with one,
    /// proves the profile's side of the circle at its ends. The intervals
    /// where rounding hides the side make runs: a run between two sides
    /// that differ holds a crossing, read at its middle; between two sides
    /// alike, a touch. A closed profile's last run goes on into its first.
    /// An open profile's run at either end gives no ruling: the line there
    /// is the extrusion's own boundary edge, which the edge and face phase
    /// meets on the cylinder.
    fn crossings(&self) -> Option<Vec<f64>> {
        let (d0, d1) = self.profile.domain();
        let mut cuts: Vec<f64> = std::iter::once(d0)
            .chain(
                self.profile
                    .knots()
                    .iter()
                    .copied()
                    .filter(|&k| k > d0 && k < d1),
            )
            .chain(std::iter::once(d1))
            .collect();
        cuts.dedup_by(|a, b| (*a - *b).abs() <= 0.0);
        let mut roots: Vec<f64> = Vec::new();
        let mut budget = MOST_INTERVALS;
        let mut walk = Walk::default();
        for span in cuts.windows(2) {
            let width = span[1] - span[0];
            let mut pending = vec![(span[0], span[1])];
            while let Some((a, b)) = pending.pop() {
                budget = budget.checked_sub(1)?;
                let read = self.bernstein(a, b)?;
                match read.reading() {
                    Reading::Apart => {
                        walk.arrive(read.side_at_start()?, |run| self.middle(run), &mut roots)?;
                    }
                    Reading::Crossing => {
                        walk.arrive(read.side_at_start()?, |run| self.middle(run), &mut roots)?;
                        roots.push(self.bisect(a, b));
                        walk.last = Some(read.side_at_end()?);
                    }
                    Reading::Graze if b - a > WIDEST_GRAZE * width => return None,
                    Reading::Unsettled if b - a > NARROWEST * width => {
                        let m = f64::midpoint(a, b);
                        pending.push((m, b));
                        pending.push((a, m));
                    }
                    Reading::Graze | Reading::Unsettled => walk.unknown(a, b),
                }
            }
        }
        let poles = self.profile.control_points();
        let closed = (*poles.first()? - *poles.last()?).length() <= SPREAD;
        walk.finish(
            closed,
            |run| self.middle(run),
            |(from, to)| self.middle(((from, d1), (d0, to))),
            &mut roots,
        )?;
        roots.sort_by(f64::total_cmp);
        Some(roots)
    }

    /// The middle of a run of parameters, given as one or two pieces, the
    /// second on from the first across a closed profile's seam. `None` where
    /// the run's points spread wider than [`SPREAD`] apart.
    fn middle(&self, run: ((f64, f64), (f64, f64))) -> Option<f64> {
        let ((a, b), (c, d)) = run;
        let at = |t: f64| self.profile.evaluate(t);
        let spread = (at(b) - at(a)).length() + (at(d) - at(c)).length();
        if spread > SPREAD {
            return None;
        }
        let half = 0.5 * ((b - a) + (d - c));
        Some(if a + half <= b {
            a + half
        } else {
            c + (half - (b - a))
        })
    }

    /// The Bernstein coefficients over `[a, b]` of `reach` in the profile's
    /// Bezier form there, and what rounding can move each by. The control
    /// points are read across the axis from the cylinder's origin, so the
    /// coefficients grow with the profile's size about the axis; the
    /// rounding of that subtraction grows with where the part sits.
    fn bernstein(&self, a: f64, b: f64) -> Option<Bernstein> {
        let homogeneous = curve_bezier_between(self.profile, a, b).ok()?;
        let p = homogeneous.len().checked_sub(1)?;
        // Each control point across the axis, times its weight, and the
        // weight.
        let mut across = Vec::with_capacity(p + 1);
        let origin = self.origin;
        let mut place = origin.x().abs().max(origin.y().abs()).max(origin.z().abs());
        let mut reach: f64 = self.radius;
        for q in &homogeneous {
            let w = q[3];
            if w <= 0.0 {
                return None;
            }
            let point = Point3::new(q[0] / w, q[1] / w, q[2] / w);
            place = place
                .max(point.x().abs())
                .max(point.y().abs())
                .max(point.z().abs());
            let v = self.across(point);
            reach = reach.max(v.length());
            across.push((v * w, w));
        }
        let max_w = across.iter().map(|&(_, w)| w).fold(0.0, f64::max);
        let r2 = self.radius * self.radius;
        let n = 2 * p;
        let mut c = vec![0.0; n + 1];
        for (i, &(vi, wi)) in across.iter().enumerate() {
            for (j, &(vj, wj)) in across.iter().enumerate() {
                let share = binomial(p, i) * binomial(p, j) / binomial(n, i + j);
                c[i + j] += share * (vi.dot(vj) - r2 * wi * wj);
            }
        }
        let band = 64.0
            * f64::EPSILON
            * max_w
            * max_w
            * reach.madd(place, reach.madd(reach, r2))
            * (n + 1) as f64;
        Some(Bernstein { c, band })
    }
    /// The root of `reach` in `[a, b]`, where it is monotone and changes
    /// sign beyond rounding between the ends.
    fn bisect(&self, mut a: f64, mut b: f64) -> f64 {
        let below = self.reach(a) < 0.0;
        for _ in 0..200 {
            let m = f64::midpoint(a, b);
            if m <= a || m >= b {
                break;
            }
            if (self.reach(m) < 0.0) == below {
                a = m;
            } else {
                b = m;
            }
        }
        f64::midpoint(a, b)
    }
}

/// A walk along the profile: the sides of the circle it is known to lie on,
/// outside (`true`) or inside, and the runs between where rounding hides
/// the side.
#[derive(Default)]
struct Walk {
    /// The first side known, and the run before it, if any.
    first: Option<(bool, Option<(f64, f64)>)>,
    /// The side last known.
    last: Option<bool>,
    /// The run since the side last known, or since the start.
    run: Option<(f64, f64)>,
}

/// A run, as one piece and an empty second piece.
const fn piece(run: (f64, f64)) -> ((f64, f64), (f64, f64)) {
    (run, (run.1, run.1))
}

impl Walk {
    /// The side over `[a, b]` is not known.
    fn unknown(&mut self, a: f64, b: f64) {
        self.run = Some(self.run.map_or((a, b), |(from, _)| (from, b)));
    }

    /// The profile is known to lie on `side` again: add to `roots` the
    /// crossing in the run since the side last known, where that side
    /// differs. `None` where the sides differ and no run is between, or
    /// `middle` does not place the crossing.
    fn arrive(
        &mut self,
        side: bool,
        middle: impl Fn(((f64, f64), (f64, f64))) -> Option<f64>,
        roots: &mut Vec<f64>,
    ) -> Option<()> {
        let run = self.run.take();
        match self.last {
            None => self.first = Some((side, run)),
            Some(before) if before != side => roots.push(middle(piece(run?))?),
            Some(_) => {}
        }
        self.last = Some(side);
        Some(())
    }

    /// Add to `roots` the crossing in the run left at the end of a closed
    /// profile's walk, across its seam. `None` where no side is known, or
    /// where the sides differ across the seam and `middle` or `across` does
    /// not place the crossing there.
    fn finish(
        self,
        closed: bool,
        middle: impl Fn(((f64, f64), (f64, f64))) -> Option<f64>,
        across: impl Fn((f64, f64)) -> Option<f64>,
        roots: &mut Vec<f64>,
    ) -> Option<()> {
        let (first, lead) = self.first?;
        let last = self.last?;
        if closed && first != last {
            roots.push(match (self.run, lead) {
                (None, None) => return None,
                (Some(run), None) | (None, Some(run)) => middle(piece(run))?,
                (Some(trail), Some(lead)) => across((trail.0, lead.1))?,
            });
        }
        Some(())
    }
}

/// The Bernstein coefficients of `reach` over an interval, and what
/// rounding can move each by.
struct Bernstein {
    c: Vec<f64>,
    band: f64,
}

impl Bernstein {
    /// The side of the circle at the interval's start, where it is known:
    /// outside (`true`) or inside. The first coefficient is the value there.
    fn side_at_start(&self) -> Option<bool> {
        self.side(*self.c.first()?)
    }

    /// The side of the circle at the interval's end, where it is known.
    fn side_at_end(&self) -> Option<bool> {
        self.side(*self.c.last()?)
    }

    fn side(&self, value: f64) -> Option<bool> {
        (value.abs() > self.band).then_some(value > 0.0)
    }

    /// What the coefficients prove, which bound `reach` and whose
    /// differences bound its slope.
    fn reading(&self) -> Reading {
        let (c, band) = (&self.c, self.band);
        if c.iter().all(|x| x.abs() <= band) {
            return Reading::Graze;
        }
        if c.iter().all(|x| *x > band) || c.iter().all(|x| *x < -band) {
            return Reading::Apart;
        }
        let rising = c.windows(2).all(|pair| pair[1] - pair[0] > 2.0 * band);
        let falling = c.windows(2).all(|pair| pair[1] - pair[0] < -2.0 * band);
        if rising || falling {
            match (self.side_at_start(), self.side_at_end()) {
                (Some(first), Some(last)) if first != last => return Reading::Crossing,
                (Some(_), Some(_)) => return Reading::Apart,
                _ => {}
            }
        }
        Reading::Unsettled
    }
}

/// What the Bernstein coefficients over an interval prove.
enum Reading {
    /// No root.
    Apart,
    /// The profile stays within rounding of the circle: a touch, which
    /// splits neither face.
    Graze,
    /// Exactly one root, where the profile crosses the circle: the sides at
    /// the two ends differ beyond rounding, and the slope keeps one sign.
    Crossing,
    /// None of these: the interval is split.
    Unsettled,
}

fn binomial(n: usize, k: usize) -> f64 {
    let mut value = 1.0;
    for i in 0..k.min(n - k) {
        value = value * (n - i) as f64 / (i + 1) as f64;
    }
    value
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;

    use super::*;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "mirrored",
                Mat4::translation(1.0, 0.5, -2.0) * Mat4::scale(-1.0, 1.0, 1.0),
            ),
        ]
    }

    fn direction(pose: &Mat4, v: Vec3) -> Vec3 {
        pose.mul_point(Point3::new(v.x(), v.y(), v.z()))
            - pose.mul_point(Point3::new(0.0, 0.0, 0.0))
    }

    /// An ellipse of semi-axes `a` along x and `b` along y about `(cx, cy)`
    /// in `z = -1`, as a rational B-spline, moved up by `rise` (and across by
    /// `lean`), placed by `pose`.
    fn elliptic_extrusion(
        (cx, cy): (f64, f64),
        (a, b): (f64, f64),
        rise: f64,
        lean: f64,
        pose: &Mat4,
    ) -> NurbsSurface {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let unit = [
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (-1.0, 1.0),
            (-1.0, 0.0),
            (-1.0, -1.0),
            (0.0, -1.0),
            (1.0, -1.0),
            (1.0, 0.0),
        ];
        let row = |shift: Vec3| -> Vec<Point3> {
            unit.iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(cx + a * x, cy + b * y, -1.0) + shift))
                .collect()
        };
        let weights: Vec<f64> = (0..9).map(|i| if i % 2 == 0 { 1.0 } else { s }).collect();
        NurbsSurface::new(
            1,
            2,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            vec![
                row(Vec3::new(0.0, 0.0, 0.0)),
                row(Vec3::new(lean, 0.0, rise)),
            ],
            vec![weights.clone(), weights],
        )
        .unwrap()
    }

    fn rod(pose: &Mat4) -> CylindricalSurface {
        CylindricalSurface::new(
            pose.mul_point(Point3::new(5.0, 5.0, 0.0)),
            direction(pose, Vec3::new(0.0, 0.0, 1.0)),
            2.0,
        )
        .unwrap()
    }

    fn room() -> Aabb3 {
        Aabb3 {
            min: Point3::new(-100.0, -100.0, -100.0),
            max: Point3::new(100.0, 100.0, 100.0),
        }
    }

    /// The circle of radius 2 about (5, 5) and the ellipse of semi-axes 3
    /// and `sqrt(5.4)` about (4, 5) cross at (6, 5 -/+ sqrt(3)) and touch at
    /// (7, 5): two rulings, from the profile up the extrusion's height.
    #[test]
    fn a_rod_crosses_an_elliptic_extrusion_on_two_rulings_and_touches_it_on_none() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((4.0, 5.0), (3.0, 5.4_f64.sqrt()), 4.0, 0.0, &pose);
            let lines = cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 2, "{name}");
            let up = direction(&pose, Vec3::new(0.0, 0.0, 4.0));
            let mut lows: Vec<Point3> = Vec::new();
            for line in &lines {
                let (from, to) = if (line.p_end - line.p_start).dot(up) > 0.0 {
                    (line.p_start, line.p_end)
                } else {
                    (line.p_end, line.p_start)
                };
                assert!(
                    (to - (from + up)).length() < 1e-12,
                    "{name}: {from:?} {to:?}"
                );
                lows.push(from);
            }
            for y in [5.0 - 3.0_f64.sqrt(), 5.0 + 3.0_f64.sqrt()] {
                let crossing = pose.mul_point(Point3::new(6.0, y, -1.0));
                let near = lows
                    .iter()
                    .filter(|p| (**p - crossing).length() < 1e-12)
                    .count();
                assert_eq!(near, 1, "{name}: {lows:?} against {crossing:?}");
            }
        }
    }

    #[test]
    fn an_extrusion_leaning_off_the_axis_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((4.0, 5.0), (3.0, 5.4_f64.sqrt()), 4.0, 1e-6, &pose);
            assert!(
                cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_profile_running_along_the_circle_is_left_to_the_marcher() {
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((5.0, 5.0), (2.0, 2.0), 4.0, 0.0, &pose);
            assert!(
                cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).is_none(),
                "{name}"
            );
        }
    }

    #[test]
    fn a_profile_clear_of_the_circle_gives_no_ruling() {
        for (name, pose) in poses() {
            for (centre, axes) in [
                ((5.0, 5.0), (3.0, 2.5)),
                ((5.0, 5.0), (1.0, 0.5)),
                ((12.0, 5.0), (3.0, 2.0)),
            ] {
                let wall = elliptic_extrusion(centre, axes, 4.0, 0.0, &pose);
                let lines =
                    cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
                assert!(lines.is_empty(), "{name} {centre:?} {axes:?}");
            }
        }
    }

    /// A degree-1 profile through `points` in `z = -1` over `knots`, moved
    /// up by 4, placed by `pose`.
    fn polyline_extrusion(points: &[(f64, f64)], knots: Vec<f64>, pose: &Mat4) -> NurbsSurface {
        let row = |z: f64| -> Vec<Point3> {
            points
                .iter()
                .map(|&(x, y)| pose.mul_point(Point3::new(x, y, z)))
                .collect()
        };
        NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            knots,
            vec![row(-1.0), row(3.0)],
            vec![vec![1.0; points.len()]; 2],
        )
        .unwrap()
    }

    fn cylinder_about(x: f64, y: f64, radius: f64, pose: &Mat4) -> CylindricalSurface {
        CylindricalSurface::new(
            pose.mul_point(Point3::new(x, y, 0.0)),
            direction(pose, Vec3::new(0.0, 0.0, 1.0)),
            radius,
        )
        .unwrap()
    }

    /// An elliptic profile whose seam lies on the rod's circle, where it
    /// crosses it: the side there is hidden by rounding at both ends of the
    /// profile's domain, and the crossing is read across the seam.
    #[test]
    fn a_crossing_on_a_closed_profiles_seam_is_one_ruling() {
        let seam = (6.0, 5.0 + 3.0_f64.sqrt());
        for (name, pose) in poses() {
            let wall = elliptic_extrusion((seam.0 - 3.0, seam.1), (3.0, 2.0), 4.0, 0.0, &pose);
            let lines = cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 2, "{name}");
            let crossing = pose.mul_point(Point3::new(seam.0, seam.1, -1.0));
            let near = lines
                .iter()
                .filter(|l| {
                    (l.p_start - crossing).length() < 1e-9 || (l.p_end - crossing).length() < 1e-9
                })
                .count();
            assert_eq!(near, 1, "{name}");
        }
    }

    /// A quarter of the elliptic profile, open, that starts on the rod's
    /// circle where the two touch: the run at its start gives no ruling, and
    /// its one crossing gives one.
    #[test]
    fn an_open_profile_that_starts_on_a_touch_gives_only_its_crossing() {
        let b = 5.4_f64.sqrt();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        for (name, pose) in poses() {
            let row = |z: f64| -> Vec<Point3> {
                [(7.0, 5.0), (7.0, 5.0 + b), (4.0, 5.0 + b)]
                    .iter()
                    .map(|&(x, y)| pose.mul_point(Point3::new(x, y, z)))
                    .collect()
            };
            let wall = NurbsSurface::new(
                1,
                2,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![row(-1.0), row(3.0)],
                vec![vec![1.0, s, 1.0]; 2],
            )
            .unwrap();
            let lines = cylinder_extrusion_rulings(&rod(&pose), &wall, &room(), &room()).unwrap();
            assert_eq!(lines.len(), 1, "{name}");
            let crossing = pose.mul_point(Point3::new(6.0, 5.0 + 3.0_f64.sqrt(), -1.0));
            let line = &lines[0];
            assert!(
                (line.p_start - crossing).length() < 1e-9
                    || (line.p_end - crossing).length() < 1e-9,
                "{name}: {:?} {:?}",
                line.p_start,
                line.p_end
            );
        }
    }

    /// Two crossings two apart in space and 5e-11 apart in the profile's
    /// parameter, on a span 1e-10 wide: two rulings.
    #[test]
    fn crossings_close_in_parameter_and_apart_in_space_are_two_rulings() {
        for (name, pose) in poses() {
            let wall = polyline_extrusion(
                &[(-2.0, 0.0), (2.0, 0.0), (3.0, 1.0)],
                vec![0.0, 0.0, 1e-10, 1.0, 1.0],
                &pose,
            );
            let lines = cylinder_extrusion_rulings(
                &cylinder_about(0.0, 0.0, 1.0, &pose),
                &wall,
                &room(),
                &room(),
            )
            .unwrap();
            assert_eq!(
                lines.len(),
                2,
                "{name}: {:?}",
                lines
                    .iter()
                    .map(|l| (l.p_start, l.p_end))
                    .collect::<Vec<_>>()
            );
            for x in [-1.0, 1.0] {
                let crossing = pose.mul_point(Point3::new(x, 0.0, -1.0));
                assert!(
                    lines.iter().any(|l| (l.p_start - crossing).length() < 1e-9
                        || (l.p_end - crossing).length() < 1e-9),
                    "{name}: no ruling at {x}"
                );
            }
        }
    }

    /// A profile that keeps at least 1.01 from the axis of a unit cylinder,
    /// a million from the origin: no ruling.
    #[test]
    fn a_profile_clear_of_a_cylinder_far_from_the_origin_gives_no_ruling() {
        for (name, pose) in poses() {
            let wall = polyline_extrusion(
                &[
                    (1_000_002.0, 0.0),
                    (1_000_000.606, 0.808),
                    (1_000_000.0, 2.0),
                ],
                vec![0.0, 0.0, 0.5, 1.0, 1.0],
                &pose,
            );
            let lines = cylinder_extrusion_rulings(
                &cylinder_about(1_000_000.0, 0.0, 1.0, &pose),
                &wall,
                &room(),
                &room(),
            );
            assert!(lines.is_some_and(|l| l.is_empty()), "{name}");
        }
    }
}

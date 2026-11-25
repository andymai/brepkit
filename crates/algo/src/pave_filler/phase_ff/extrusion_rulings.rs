//! The section of a cylinder and a B-spline extrusion along the cylinder's
//! axis: the lines along the axis through the points where the circle
//! crosses the extrusion's profile.
//!
//! Marched instead, each line came back as a fitted curve that strayed from
//! the straight ruling by up to 1.2e-5 where the walls cross at a shallow
//! angle, and the faces it bounded lost that sliver of volume.

use brepkit_math::aabb::Aabb3;
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
/// graze splits neither face. So is a crossing so shallow that the profile
/// stays within rounding of the circle on both sides of it.
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
    /// [`MOST_INTERVALS`] intervals, or where the profile runs within
    /// rounding of the circle over more than [`WIDEST_GRAZE`] of a span.
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
        let mut roots = Vec::new();
        let mut budget = MOST_INTERVALS;
        for span in cuts.windows(2) {
            let width = span[1] - span[0];
            let mut pending = vec![(span[0], span[1])];
            while let Some((a, b)) = pending.pop() {
                budget = budget.checked_sub(1)?;
                match self.read(a, b)? {
                    Reading::Apart => {}
                    Reading::Graze if b - a > WIDEST_GRAZE * width => return None,
                    Reading::Graze => {}
                    Reading::Crossing => roots.push(self.bisect(a, b)),
                    Reading::Unsettled if b - a > NARROWEST * width => {
                        let m = f64::midpoint(a, b);
                        pending.push((m, b));
                        pending.push((a, m));
                    }
                    Reading::Unsettled => {}
                }
            }
        }
        roots.sort_by(f64::total_cmp);
        // A root on the end of an interval is found from both sides.
        roots.dedup_by(|a, b| (*a - *b).abs() <= 1e-9 * (d1 - d0));
        Some(roots)
    }

    /// What the profile's Bezier form over `[a, b]` proves about `reach`
    /// there, read off its Bernstein coefficients, which bound it and
    /// whose differences bound its slope.
    fn read(&self, a: f64, b: f64) -> Option<Reading> {
        let homogeneous = curve_bezier_between(self.profile, a, b).ok()?;
        let p = homogeneous.len().checked_sub(1)?;
        // Each control point across the axis, times its weight, and the
        // weight.
        let mut across = Vec::with_capacity(p + 1);
        let mut size: f64 = self.radius;
        for q in &homogeneous {
            let w = q[3];
            if w <= 0.0 {
                return None;
            }
            let point = Point3::new(q[0] / w, q[1] / w, q[2] / w);
            let v = self.across(point);
            size = size
                .max(v.length())
                .max((point - Point3::new(0.0, 0.0, 0.0)).length());
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
        // What rounding can move a coefficient by.
        let band = 64.0 * f64::EPSILON * (size * max_w).powi(2) * (n + 1) as f64;
        if c.iter().all(|x| x.abs() <= band) {
            return Some(Reading::Graze);
        }
        if c.iter().all(|x| *x > band) || c.iter().all(|x| *x < -band) {
            return Some(Reading::Apart);
        }
        let rising = c.windows(2).all(|pair| pair[1] - pair[0] > 2.0 * band);
        let falling = c.windows(2).all(|pair| pair[1] - pair[0] < -2.0 * band);
        if rising || falling {
            let (first, last) = (c[0], c[n]);
            let meets = first.abs() <= band || last.abs() <= band || (first > 0.0) != (last > 0.0);
            return Some(if meets {
                Reading::Crossing
            } else {
                Reading::Apart
            });
        }
        Some(Reading::Unsettled)
    }

    /// The root of `reach` in `[a, b]`, where it is monotone and has one
    /// root: on an end where both ends read on one side.
    fn bisect(&self, mut a: f64, mut b: f64) -> f64 {
        let (fa, fb) = (self.reach(a), self.reach(b));
        if (fa < 0.0) == (fb < 0.0) {
            return if fa.abs() <= fb.abs() { a } else { b };
        }
        let below = fa < 0.0;
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

/// What the Bernstein coefficients over an interval prove.
enum Reading {
    /// No root.
    Apart,
    /// The profile stays within rounding of the circle: a touch, which
    /// splits neither face.
    Graze,
    /// Exactly one root, where the profile crosses the circle.
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
}

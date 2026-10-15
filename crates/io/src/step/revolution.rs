//! STEP surfaces of revolution as brepkit surfaces.
//!
//! A `SURFACE_OF_REVOLUTION` turns a profile curve a full turn about an axis.
//! STEP parameterizes it by the turn in u and the profile in v, so its normal
//! is the direction of turning crossed with the profile's tangent. Where the
//! profile sweeps a plane, cylinder, cone, sphere or torus, brepkit keeps the
//! exact analytic surface. Any other profile becomes the rational B-spline
//! surface that carries each of the profile's control points round a full
//! rational circle, which is the same surface exactly.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, TAU};

use brepkit_math::MathError;
use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::nurbs::{NurbsCurve, NurbsSurface};
use brepkit_math::surfaces::{
    ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::face::FaceSurface;

/// The curve a surface of revolution turns.
pub(super) enum Profile {
    Line { point: Point3, direction: Vec3 },
    Circle(Circle3D),
    Spline(NurbsCurve),
}

impl Profile {
    fn point_at(&self, t: f64) -> Point3 {
        match self {
            Self::Line { point, direction } => *point + *direction * t,
            Self::Circle(c) => c.evaluate(t),
            Self::Spline(n) => n.evaluate(t),
        }
    }

    /// The profile's direction at `t`, unnormalized.
    fn tangent_at(&self, t: f64) -> Vec3 {
        match self {
            Self::Line { direction, .. } => *direction,
            Self::Circle(_) => self.point_at(t + 1e-6) - self.point_at(t - 1e-6),
            Self::Spline(n) => {
                let (d0, d1) = n.domain();
                let h = 1e-6 * (d1 - d0);
                self.point_at((t + h).min(d1)) - self.point_at((t - h).max(d0))
            }
        }
    }

    /// Parameters to test the surface's sense at, spread along the profile.
    fn sample_parameters(&self) -> Vec<f64> {
        match self {
            Self::Line { .. } => vec![0.0, 1.0, -1.0, 10.0, -10.0],
            Self::Circle(_) => (0..8).map(|k| 0.3 + f64::from(k) * FRAC_PI_4).collect(),
            Self::Spline(n) => {
                let (d0, d1) = n.domain();
                (1..8)
                    .map(|k| (d1 - d0).mul_add(f64::from(k) / 8.0, d0))
                    .collect()
            }
        }
    }
}

/// Where a face lies on its surface of revolution, read off its boundary.
///
/// A profile that crosses the axis sweeps some of its surface twice, once
/// from each side, with opposite normals; only where the face lies says
/// which side it is on.
#[derive(Default)]
pub(super) struct Boundary {
    /// Points of the face's boundary.
    pub points: Vec<Point3>,
    /// Boundary edges that run along the profile curve itself, as their
    /// ends in the curve's own direction.
    pub on_profile: Vec<(Point3, Point3)>,
}

/// The axis the profile turns about: a point on it and its unit direction.
#[derive(Clone, Copy)]
pub(super) struct Axis {
    pub origin: Point3,
    pub direction: Vec3,
}

impl Axis {
    fn turn(&self, p: Point3, angle: f64) -> Point3 {
        let v = p - self.origin;
        let k = self.direction;
        let (s, c) = angle.sin_cos();
        self.origin + v * c + k.cross(v) * s + k * (k.dot(v) * (1.0 - c))
    }

    /// The point of the axis level with `p`, and the offset from it to `p`.
    fn foot(&self, p: Point3) -> (Point3, Vec3) {
        let v = p - self.origin;
        let foot = self.origin + self.direction * v.dot(self.direction);
        (foot, p - foot)
    }
}

/// The surface the profile sweeps where `boundary` bounds a face, and
/// whether its normal runs against the STEP surface's.
///
/// # Errors
///
/// Returns an error if the profile sweeps no surface (it lies on the axis),
/// sweeps a hyperboloid (a line that misses the axis at a slant), or the
/// swept surface cannot be built.
pub(super) fn revolution(
    profile: &Profile,
    axis: Axis,
    boundary: &Boundary,
) -> Result<(FaceSurface, bool), MathError> {
    match analytic(profile, axis, boundary)? {
        Some(surface) => {
            let inverted = inverted(&surface, profile, axis, boundary)?;
            Ok((surface, inverted))
        }
        // The B-spline surface turns about the axis in u and runs along the
        // profile in v, as the STEP surface does: its Su x Sv is the STEP
        // normal.
        None => Ok((FaceSurface::Nurbs(spline(profile, axis)?), false)),
    }
}

/// Below this a length is noise in coordinates of the part's size.
fn tolerance(scale: f64) -> f64 {
    1e-6 * (1.0 + scale)
}

/// Below this sine a line runs along, or square to, the axis.
const PARALLEL: f64 = 1e-8;

/// The analytic surface the profile sweeps, if it sweeps one.
fn analytic(
    profile: &Profile,
    axis: Axis,
    boundary: &Boundary,
) -> Result<Option<FaceSurface>, MathError> {
    let k = axis.direction;
    match profile {
        Profile::Line { point, direction } => {
            let d = direction.normalize()?;
            let (foot, offset) = axis.foot(*point);
            let tol = tolerance(offset.length());
            let (rise, across) = (d.dot(k), d.cross(k).length());
            // Square to the axis, a line sweeps a plane whether or not it
            // passes through the axis.
            if rise.abs() < PARALLEL {
                let d = k.dot(Vec3::new(point.x(), point.y(), point.z()));
                return Ok(Some(FaceSurface::Plane { normal: k, d }));
            }
            if across < PARALLEL {
                if offset.length() < tol {
                    return Err(MathError::ZeroVector);
                }
                let cylinder = CylindricalSurface::new(foot, k, offset.length())?;
                return Ok(Some(FaceSurface::Cylinder(cylinder)));
            }
            // The line meets the axis when the gap between them is nothing.
            let gap = (*point - axis.origin).dot(d.cross(k)) / across;
            if gap.abs() > tol {
                return Ok(None);
            }
            // Where the line crosses the axis: along it from `point` until
            // its offset from the axis is gone.
            let out = d - k * rise;
            let t = -offset.dot(out) / out.dot(out);
            let apex = axis.foot(*point + d * t).0;
            // The line sweeps two cones, apex to apex; the face is on one.
            let toward = boundary
                .points
                .iter()
                .chain(std::iter::once(point))
                .map(|&p| p - apex)
                .find(|v| v.length() > tol)
                .unwrap_or(d);
            let opening = if toward.dot(k) >= 0.0 { k } else { -k };
            let half_angle = rise.abs().atan2(across);
            let cone = ConicalSurface::new(apex, opening, half_angle)?;
            Ok(Some(FaceSurface::Cone(cone)))
        }
        Profile::Circle(c) => {
            let (foot, offset) = axis.foot(c.center());
            let tol = tolerance(offset.length() + c.radius());
            let off_plane = c.radius() * c.normal().dot(k).abs()
                + (c.center() - axis.origin).dot(c.normal()).abs();
            if off_plane > tol {
                return Ok(None);
            }
            if offset.length() < tol {
                let reference = if c.u_axis().cross(k).length() > 1e-9 {
                    c.u_axis()
                } else {
                    c.v_axis()
                };
                let sphere =
                    SphericalSurface::with_axis_and_ref_dir(foot, c.radius(), k, reference)?;
                return Ok(Some(FaceSurface::Sphere(sphere)));
            }
            // A circle reaching the axis sweeps a torus that passes through
            // itself, which brepkit's torus does not describe.
            if c.radius() >= offset.length() {
                return Ok(None);
            }
            let torus = ToroidalSurface::with_axis_and_ref_dir(
                foot,
                offset.length(),
                c.radius(),
                k,
                offset,
            )?;
            Ok(Some(FaceSurface::Torus(torus)))
        }
        Profile::Spline(_) => Ok(None),
    }
}

/// The knots of a full rational circle in four quarter arcs.
const CIRCLE_KNOTS: [f64; 12] = [
    0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
];

/// The nine control points that carry `p` round the axis: on its circle at
/// each quarter turn, and pushed out to the quarter arcs' corners between
/// them.
fn circle_row(p: Point3, axis: Axis) -> Vec<Point3> {
    (0..9)
        .map(|k| {
            let on_circle = axis.turn(p, f64::from(k) * FRAC_PI_4);
            if k % 2 == 0 {
                on_circle
            } else {
                let (foot, offset) = axis.foot(on_circle);
                foot + offset * std::f64::consts::SQRT_2
            }
        })
        .collect()
}

fn circle_weights(w: f64) -> Vec<f64> {
    (0..9)
        .map(|k| if k % 2 == 0 { w } else { w * FRAC_1_SQRT_2 })
        .collect()
}

/// An ellipse as the rational B-spline it is: the nine-point circle
/// stretched along its axes, from its parameter origin round the way it runs.
pub(super) fn ellipse_profile(e: &Ellipse3D) -> Result<NurbsCurve, MathError> {
    let points = (0..9)
        .map(|k| {
            let (s, c) = (f64::from(k) * FRAC_PI_4).sin_cos();
            let reach = if k % 2 == 0 {
                1.0
            } else {
                std::f64::consts::SQRT_2
            };
            e.center()
                + (e.u_axis() * (e.semi_major() * c) + e.v_axis() * (e.semi_minor() * s)) * reach
        })
        .collect();
    NurbsCurve::new(2, CIRCLE_KNOTS.to_vec(), points, circle_weights(1.0))
}

/// The B-spline surface the profile sweeps: round the axis in `u`, along the
/// profile in `v`.
fn spline(profile: &Profile, axis: Axis) -> Result<NurbsSurface, MathError> {
    let curve = match profile {
        Profile::Spline(n) => n.clone(),
        Profile::Circle(c) => {
            let around = Axis {
                origin: c.center(),
                direction: c.normal(),
            };
            NurbsCurve::new(
                2,
                CIRCLE_KNOTS.to_vec(),
                circle_row(c.evaluate(0.0), around),
                circle_weights(1.0),
            )?
        }
        // A line that misses the axis sweeps a hyperboloid, which has no
        // finite span to fit until a face bounds it.
        Profile::Line { .. } => return Err(MathError::ZeroVector),
    };
    let turned: Vec<Vec<Point3>> = curve
        .control_points()
        .iter()
        .map(|&p| circle_row(p, axis))
        .collect();
    let round = circle_weights(1.0);
    NurbsSurface::new(
        2,
        curve.degree(),
        CIRCLE_KNOTS.to_vec(),
        curve.knots().to_vec(),
        (0..9)
            .map(|i| turned.iter().map(|row| row[i]).collect())
            .collect(),
        round
            .iter()
            .map(|&r| curve.weights().iter().map(|&w| w * r).collect())
            .collect(),
    )
}

/// Whether the analytic `surface`'s normal runs against the STEP surface's,
/// read where the face lies.
fn inverted(
    surface: &FaceSurface,
    profile: &Profile,
    axis: Axis,
    boundary: &Boundary,
) -> Result<bool, MathError> {
    let against_at = |t: f64| -> Option<bool> {
        let p = profile.point_at(t);
        let (_, offset) = axis.foot(p);
        if offset.length() < 1e-9 * (1.0 + (p - axis.origin).length()) {
            return None;
        }
        let turning = axis.direction.cross(offset).normalize().ok()?;
        let along = profile.tangent_at(t).normalize().ok()?;
        let step_normal = turning.cross(along);
        if step_normal.length() < 1e-9 {
            return None;
        }
        let normal = if let FaceSurface::Plane { normal, .. } = surface {
            *normal
        } else {
            let (u, v) = surface.project_point(p)?;
            surface.normal(u, v)
        };
        Some(normal.dot(step_normal) < 0.0)
    };
    let samples = profile.sample_parameters();
    let on_face = face_parameters(profile, axis, boundary);
    match surface {
        // A line through the axis sweeps two cones point to point; brepkit
        // keeps the one the face is on, so only points on it count.
        FaceSurface::Cone(cone) => {
            let on_cone = |t: &f64| (profile.point_at(*t) - cone.apex()).dot(cone.axis()) > 0.0;
            on_face
                .into_iter()
                .chain(samples)
                .filter(on_cone)
                .find_map(against_at)
                .ok_or(MathError::ZeroVector)
        }
        // The profile sweeps these from both sides of the axis, with
        // opposite normals.
        FaceSurface::Plane { .. } | FaceSurface::Sphere(_) => {
            if let Some(sense) = on_face.into_iter().find_map(against_at) {
                return Ok(sense);
            }
            log::warn!(
                "STEP surface of revolution: its profile crosses the axis and its face does not \
                 say which side it is on"
            );
            samples
                .into_iter()
                .find_map(against_at)
                .ok_or(MathError::ZeroVector)
        }
        _ => samples
            .into_iter()
            .find_map(against_at)
            .ok_or(MathError::ZeroVector),
    }
}

/// Profile parameters where the face lies: for a line, at the height of a
/// boundary point (each height is one point of a slanted line); for any
/// profile, midway along a boundary edge that runs on the profile itself.
fn face_parameters(profile: &Profile, axis: Axis, boundary: &Boundary) -> Vec<f64> {
    let k = axis.direction;
    match profile {
        Profile::Line { point, direction } => {
            let along = |q: Point3| (q - *point).dot(*direction) / direction.dot(*direction);
            let mut out: Vec<f64> = boundary
                .on_profile
                .iter()
                .map(|&(a, b)| 0.5 * (along(a) + along(b)))
                .collect();
            let rise = direction.dot(k);
            if rise.abs() > PARALLEL * direction.length() {
                out.extend(boundary.points.iter().map(|&q| (q - *point).dot(k) / rise));
            }
            out
        }
        Profile::Circle(c) => boundary
            .on_profile
            .iter()
            .filter_map(|&(start, end)| {
                let t0 = c.project(start);
                let span = (c.project(end) - t0).rem_euclid(TAU);
                (span > 1e-9).then_some(0.5f64.mul_add(span, t0))
            })
            .collect(),
        Profile::Spline(n) => boundary
            .on_profile
            .iter()
            .filter_map(|&(start, end)| {
                let at = |p| {
                    brepkit_math::nurbs::projection::project_point_to_curve(n, p, 1e-9)
                        .ok()
                        .map(|q| q.parameter)
                };
                Some(0.5 * (at(start)? + at(end)?))
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const Z: Axis = Axis {
        origin: Point3::new(0.0, 0.0, 0.0),
        direction: Vec3::new(0.0, 0.0, 1.0),
    };

    /// The STEP surface at turn `u` and profile parameter `v`, and its normal
    /// there: the derivative along the turn crossed with the one along the
    /// profile.
    fn step_normal(profile: &Profile, u: f64, v: f64) -> (Point3, Vec3) {
        let h = 1e-6;
        let at = |u: f64, v: f64| Z.turn(profile.point_at(v), u);
        let su = (at(u + h, v) - at(u - h, v)) * (0.5 / h);
        let sv = (at(u, v + h) - at(u, v - h)) * (0.5 / h);
        (at(u, v), su.cross(sv))
    }

    /// The revolution of `profile` is a `kind`, holds the profile turned to
    /// any angle, and keeps the STEP surface's side at profile parameters
    /// `on_face`, where a face bounded by `boundary` lies.
    fn assert_sense(profile: &Profile, kind: &str, boundary: &Boundary, on_face: &[f64]) {
        let (surface, inverted) = revolution(profile, Z, boundary).expect(kind);
        assert_eq!(surface.type_tag(), kind);
        if let FaceSurface::Nurbs(n) = &surface {
            // At no turn the surface is the profile itself.
            assert!(!inverted, "{kind}: a spline surface keeps the STEP sense");
            let (v0, v1) = n.domain_v();
            for k in 1..16 {
                let v = (v1 - v0).mul_add(f64::from(k) / 16.0, v0);
                let p = n.evaluate(0.0, v);
                let t = match profile {
                    Profile::Circle(c) => c.project(p),
                    _ => v,
                };
                let (q, step) = step_normal(profile, 0.0, t);
                assert!((q - p).length() < 1e-9, "{kind}: {p:?} is not the profile");
                if step.length() > 1e-6 {
                    assert!(n.normal(0.0, v).unwrap().dot(step) > 0.0, "{kind} at v {v}");
                }
            }
            return;
        }
        let mut checked = 0;
        for &t in on_face {
            for u in [0.1, 2.0, 4.0] {
                let (p, n) = step_normal(profile, u, t);
                assert!(n.length() > 1e-12, "{kind}: t {t} on the axis");
                let normal = if let FaceSurface::Plane { normal, d } = &surface {
                    let height = normal.dot(Vec3::new(p.x(), p.y(), p.z()));
                    assert!((height - d).abs() < 1e-9, "{kind}: {p:?} off the plane");
                    *normal
                } else {
                    let (a, b) = surface.project_point(p).unwrap();
                    let q = surface.evaluate(a, b).unwrap();
                    assert!((q - p).length() < 1e-9, "{kind}: {p:?} off the surface");
                    surface.normal(a, b)
                };
                assert_eq!(normal.dot(n) > 0.0, !inverted, "{kind} at t {t}, u {u}");
                checked += 1;
            }
        }
        assert!(checked > 0);
    }

    fn line(point: [f64; 3], d: [f64; 3]) -> Profile {
        Profile::Line {
            point: Point3::new(point[0], point[1], point[2]),
            direction: Vec3::new(d[0], d[1], d[2]),
        }
    }

    fn circle(center: [f64; 3], normal: [f64; 3], r: f64) -> Profile {
        Profile::Circle(
            Circle3D::new(
                Point3::new(center[0], center[1], center[2]),
                Vec3::new(normal[0], normal[1], normal[2]),
                r,
            )
            .unwrap(),
        )
    }

    fn arc(profile: &Profile, t0: f64, t1: f64) -> Boundary {
        Boundary {
            points: vec![profile.point_at(t0), profile.point_at(t1)],
            on_profile: vec![(profile.point_at(t0), profile.point_at(t1))],
        }
    }

    fn heights(profile: &Profile, ts: &[f64]) -> Boundary {
        Boundary {
            points: ts.iter().map(|&t| profile.point_at(t)).collect(),
            on_profile: Vec::new(),
        }
    }

    #[test]
    fn a_profile_off_the_axis_keeps_its_side_everywhere() {
        let none = Boundary::default();
        for s in [1.0, -1.0] {
            let cylinder = line([4.0, 0.0, 3.0], [0.0, 0.0, s]);
            assert_sense(&cylinder, "cylinder", &none, &cylinder.sample_parameters());
            let torus = circle([10.0, 0.0, 1.0], [0.0, s, 0.0], 3.0);
            assert_sense(&torus, "torus", &none, &torus.sample_parameters());
            let spline = Profile::Spline(
                NurbsCurve::new(
                    2,
                    vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                    vec![
                        Point3::new(3.0, 0.0, -2.0 * s),
                        Point3::new(6.0, 0.0, 0.0),
                        Point3::new(3.0, 0.0, 2.0 * s),
                    ],
                    vec![1.0, 1.0, 1.0],
                )
                .unwrap(),
            );
            assert_sense(&spline, "nurbs", &none, &[]);
            // A line a hair off the axis's direction sweeps a cylinder.
            let nearly = line([4.0, 0.0, 3.0], [1e-9, 0.0, s]);
            assert_sense(&nearly, "cylinder", &none, &[0.0]);
        }
    }

    #[test]
    fn a_profile_across_the_axis_takes_the_side_its_face_is_on() {
        for s in [1.0, -1.0] {
            // A line through the axis sweeps two cones; a face's height
            // says which. Steep and shallow lines both.
            for (point, slope, apex) in
                [([4.0, 0.0, 0.0], -0.4, 10.0), ([12.0, 0.0, 0.0], -2.0, 6.0)]
            {
                let cone = line(point, [slope * s, 0.0, s]);
                for (near, far) in [(apex * 0.2, apex * 0.5), (apex * 1.5, apex * 2.0)] {
                    let ts = [near * s, far * s];
                    assert_sense(&cone, "cone", &heights(&cone, &ts), &ts);
                }
            }
            // A line square to the axis sweeps the plane twice, whether it
            // crosses the axis or passes it by.
            for y in [0.0, 1.0] {
                let plane = line([4.0, y, 2.0], [s, 0.0, 0.0]);
                for (a, b) in [(-1.0, 3.0), (-9.0, -6.0)] {
                    let span = arc(&plane, a * s, b * s);
                    assert_sense(&plane, "plane", &span, &[0.5 * (a + b) * s]);
                }
            }
            // A circle about a point of the axis sweeps the sphere twice.
            let sphere = circle([0.0, 0.0, 1.0], [0.0, s, 0.0], 5.0);
            for (a, b) in [(0.3, 1.2), (3.5, 4.4)] {
                assert_sense(&sphere, "sphere", &arc(&sphere, a, b), &[0.5 * (a + b)]);
            }
            // A circle reaching over the axis sweeps a torus through itself.
            let spindle = circle([2.0, 0.0, 0.0], [0.0, s, 0.0], 3.0);
            assert_sense(&spindle, "nurbs", &Boundary::default(), &[]);
        }
    }

    #[test]
    fn small_parts_in_large_units_keep_their_side() {
        // A 10 mm ball and a 5 mm fillet ring, written in metres.
        let ball = circle([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.01);
        assert_sense(&ball, "sphere", &arc(&ball, 0.3, 1.2), &[0.75]);
        let ring = circle([0.05, 0.0, 0.0], [0.0, 1.0, 0.0], 0.005);
        assert_sense(
            &ring,
            "torus",
            &Boundary::default(),
            &ring.sample_parameters(),
        );
    }

    #[test]
    fn an_ellipse_is_its_rational_spline() {
        let e = Ellipse3D::new_with_ref(
            Point3::new(6.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            3.0,
            1.5,
            Vec3::new(1.0, 0.0, 1.0),
        )
        .unwrap();
        let n = ellipse_profile(&e).unwrap();
        for k in 0..=64 {
            let p = n.evaluate(f64::from(k) / 64.0);
            let d = p - e.center();
            let (x, y) = (d.dot(e.u_axis()) / 3.0, d.dot(e.v_axis()) / 1.5);
            assert!((x.hypot(y) - 1.0).abs() < 1e-12, "{p:?} off the ellipse");
            assert!(d.dot(e.normal()).abs() < 1e-12, "{p:?} off its plane");
        }
        // It starts where the ellipse does and runs the same way.
        assert!((n.evaluate(0.0) - e.evaluate(0.0)).length() < 1e-12);
        assert!((n.evaluate(0.01) - e.evaluate(0.0)).dot(e.tangent(0.0)) > 0.0);
        let profile = Profile::Spline(n);
        assert_sense(&profile, "nurbs", &Boundary::default(), &[]);
    }

    #[test]
    fn a_skew_line_is_refused() {
        let skew = line([4.0, 0.0, 0.0], [0.0, 1.0, 1.0]);
        assert!(revolution(&skew, Z, &Boundary::default()).is_err());
    }
}

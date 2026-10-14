//! STEP surfaces of revolution as brepkit surfaces.
//!
//! A `SURFACE_OF_REVOLUTION` turns a profile curve a full turn about an axis.
//! Where that sweeps a plane, cylinder, cone, sphere or torus, brepkit keeps
//! the exact analytic surface. Any other profile becomes the rational
//! B-spline surface that carries each of the profile's control points round
//! a full rational circle, which is the same surface exactly.

use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_4, TAU};

use brepkit_math::MathError;
use brepkit_math::curves::Circle3D;
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
/// whether its normal runs against the STEP surface's: the profile's tangent
/// crossed with the way it turns.
///
/// # Errors
///
/// Returns an error if the profile sweeps no surface (it lies on the axis)
/// or the swept surface cannot be built.
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
        // The B-spline surface runs along the profile in u and round the axis
        // in v, as the STEP surface does: its Su x Sv is the STEP normal.
        None => Ok((FaceSurface::Nurbs(spline(profile, axis)?), false)),
    }
}

/// Below this a length is noise in coordinates of the part's size.
fn tolerance(scale: f64) -> f64 {
    1e-6 * (1.0 + scale)
}

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
            let across = d.cross(k);
            if across.length() < 1e-12 {
                if offset.length() < tol {
                    return Err(MathError::ZeroVector);
                }
                let cylinder = CylindricalSurface::new(foot, k, offset.length())?;
                return Ok(Some(FaceSurface::Cylinder(cylinder)));
            }
            // The line meets the axis when the gap between them is nothing.
            let gap = (*point - axis.origin).dot(across) / across.length();
            if gap.abs() > tol {
                return Ok(None);
            }
            if d.dot(k).abs() < 1e-12 {
                let d = k.dot(Vec3::new(point.x(), point.y(), point.z()));
                return Ok(Some(FaceSurface::Plane { normal: k, d }));
            }
            // Where the line crosses the axis: along it from `point` until
            // its offset from the axis is gone.
            let out = d - k * d.dot(k);
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
            let half_angle = d.dot(k).abs().clamp(0.0, 1.0).asin();
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

/// The nine control points and weights that carry `p` round the axis: on its
/// circle at each quarter turn, and pushed out to the quarter arcs' corners
/// between them.
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

/// The B-spline surface the profile sweeps: along the profile in `u`, round
/// the axis in `v`.
fn spline(profile: &Profile, axis: Axis) -> Result<NurbsSurface, MathError> {
    let curve = match profile {
        Profile::Spline(n) => n.clone(),
        Profile::Circle(c) => {
            let around = Axis {
                origin: c.center(),
                direction: c.normal(),
            };
            let start = c.evaluate(0.0);
            NurbsCurve::new(
                2,
                CIRCLE_KNOTS.to_vec(),
                circle_row(start, around),
                circle_weights(1.0),
            )?
        }
        // A line that misses the axis sweeps a hyperboloid, which has no
        // finite span to fit until a face bounds it.
        Profile::Line { .. } => return Err(MathError::ZeroVector),
    };
    NurbsSurface::new(
        curve.degree(),
        2,
        curve.knots().to_vec(),
        CIRCLE_KNOTS.to_vec(),
        curve
            .control_points()
            .iter()
            .map(|&p| circle_row(p, axis))
            .collect(),
        curve.weights().iter().map(|&w| circle_weights(w)).collect(),
    )
}

/// Whether `surface`'s normal runs against the STEP surface's, read where
/// the profile lies off the axis: anywhere, unless the profile crosses the
/// axis, and then where `boundary` puts the face.
fn inverted(
    surface: &FaceSurface,
    profile: &Profile,
    axis: Axis,
    boundary: &Boundary,
) -> Result<bool, MathError> {
    let against_at = |t: f64| -> Option<bool> {
        let p = profile.point_at(t);
        let (_, offset) = axis.foot(p);
        let h = 1e-6;
        let along = profile.point_at(t + h) - profile.point_at(t - h);
        let turning = axis.direction.cross(offset);
        let step_normal = along.cross(turning);
        if step_normal.length() < 1e-9 * (1.0 + offset.length()).powi(2) {
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
    let senses: Vec<bool> = profile
        .sample_parameters()
        .into_iter()
        .filter_map(against_at)
        .collect();
    let Some(&first) = senses.first() else {
        return Err(MathError::ZeroVector);
    };
    if senses.iter().all(|&s| s == first) {
        return Ok(first);
    }
    if let Some(sense) = face_parameters(profile, axis, boundary)
        .into_iter()
        .find_map(against_at)
    {
        return Ok(sense);
    }
    log::warn!(
        "STEP surface of revolution: its profile crosses the axis and its face does not say which side it is on"
    );
    Ok(first)
}

/// Profile parameters where the face lies: for a line, at the height of a
/// boundary point (each height is one point of the line); for a curve,
/// midway along a boundary edge that runs on the curve itself.
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
            if rise.abs() > 1e-12 {
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

    /// The STEP normal of the revolution at profile parameter `t`, angle `v`.
    fn step_normal(profile: &Profile, t: f64, v: f64) -> (Point3, Vec3) {
        let h = 1e-6;
        let at = |t: f64, v: f64| Z.turn(profile.point_at(t), v);
        let su = (at(t + h, v) - at(t - h, v)) * (0.5 / h);
        let sv = (at(t, v + h) - at(t, v - h)) * (0.5 / h);
        (at(t, v), su.cross(sv))
    }

    /// The revolution of `profile` is a `kind`, holds the profile turned to
    /// any angle, and keeps the STEP surface's side at profile parameters
    /// `on_face`, where a face bounded by `boundary` lies.
    fn assert_sense(profile: &Profile, kind: &str, boundary: &Boundary, on_face: &[f64]) {
        let (surface, inverted) = revolution(profile, Z, boundary).expect(kind);
        assert_eq!(surface.type_tag(), kind);
        if let FaceSurface::Nurbs(n) = &surface {
            // At angle zero the surface is the profile itself.
            assert!(!inverted, "{kind}: a spline surface keeps the STEP sense");
            let (u0, u1) = n.domain_u();
            for k in 1..16 {
                let u = (u1 - u0).mul_add(f64::from(k) / 16.0, u0);
                let p = n.evaluate(u, 0.0);
                let t = match profile {
                    Profile::Circle(c) => c.project(p),
                    _ => u,
                };
                let (q, step) = step_normal(profile, t, 0.0);
                assert!((q - p).length() < 1e-9, "{kind}: {p:?} is not the profile");
                if step.length() > 1e-6 {
                    assert!(n.normal(u, 0.0).unwrap().dot(step) > 0.0, "{kind} at u {u}");
                }
            }
            return;
        }
        let mut checked = 0;
        for &t in on_face {
            for v in [0.1, 2.0, 4.0] {
                let (p, n) = step_normal(profile, t, v);
                assert!(n.length() > 1e-6, "{kind}: t {t} on the axis");
                let normal = if let FaceSurface::Plane { normal, d } = &surface {
                    let height = normal.dot(Vec3::new(p.x(), p.y(), p.z()));
                    assert!((height - d).abs() < 1e-9, "{kind}: {p:?} off the plane");
                    *normal
                } else {
                    let (u, w) = surface.project_point(p).unwrap();
                    let q = surface.evaluate(u, w).unwrap();
                    assert!((q - p).length() < 1e-9, "{kind}: {p:?} off the surface");
                    surface.normal(u, w)
                };
                assert_eq!(normal.dot(n) > 0.0, !inverted, "{kind} at t {t}, v {v}");
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
            assert_sense(&spline, "nurbs", &none, &spline.sample_parameters());
        }
    }

    #[test]
    fn a_profile_across_the_axis_takes_the_side_its_face_is_on() {
        for s in [1.0, -1.0] {
            // A line through the axis sweeps two cones; a face's height
            // says which.
            let cone = line([4.0, 0.0, 0.0], [-0.4 * s, 0.0, s]);
            for (near, far) in [(2.0, 5.0), (15.0, 20.0)] {
                let boundary = Boundary {
                    points: vec![cone.point_at(near * s), cone.point_at(far * s)],
                    on_profile: Vec::new(),
                };
                assert_sense(&cone, "cone", &boundary, &[near * s, far * s]);
            }
            // A line square to the axis sweeps the plane twice.
            let plane = line([4.0, 0.0, 2.0], [s, 0.0, 0.0]);
            for (a, b) in [(-1.0, 3.0), (-9.0, -6.0)] {
                assert_sense(
                    &plane,
                    "plane",
                    &arc(&plane, a * s, b * s),
                    &[0.5 * (a + b) * s],
                );
            }
            // A circle about a point of the axis sweeps the sphere twice.
            let sphere = circle([0.0, 0.0, 1.0], [0.0, s, 0.0], 5.0);
            for (a, b) in [(0.3, 1.2), (3.5, 4.4)] {
                assert_sense(&sphere, "sphere", &arc(&sphere, a, b), &[0.5 * (a + b)]);
            }
            // A circle reaching over the axis sweeps a torus through itself.
            let spindle = circle([2.0, 0.0, 0.0], [0.0, s, 0.0], 3.0);
            for (a, b) in [(-0.5, 0.5), (2.8, 3.4)] {
                let t = spindle.point_at(0.5 * (a + b));
                let (_, offset) = Z.foot(t);
                assert!(offset.length() > 0.5, "sample clear of the axis");
                assert_sense(&spindle, "nurbs", &arc(&spindle, a, b), &[0.5 * (a + b)]);
            }
        }
    }

    #[test]
    fn a_skew_line_is_refused() {
        let skew = line([4.0, 0.0, 0.0], [0.0, 1.0, 1.0]);
        assert!(revolution(&skew, Z, &Boundary::default()).is_err());
    }
}

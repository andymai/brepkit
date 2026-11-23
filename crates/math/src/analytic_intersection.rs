//! Closed-form and semi-analytic intersections of analytic surfaces with planes.
//!
//! Provides specialized intersection algorithms for cylinder, cone, sphere,
//! and torus surfaces with planes, as well as a general marching approach
//! for analytic-analytic surface intersections.

use crate::madd::Madd;
use std::f64::consts::{FRAC_PI_2, TAU};

use crate::MathError;
use crate::aabb::Aabb3;
use crate::curves::{Circle3D, Ellipse3D, Hyperbola3D, Line3D, Parabola3D};
use crate::frame::Frame3;
use crate::nurbs::curve::NurbsCurve;
use crate::nurbs::fitting::interpolate;
use crate::nurbs::intersection::{IntersectionCurve, IntersectionPoint};
use crate::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface};
use crate::tolerance::Tolerance;
use crate::vec::{Point3, Vec3};

mod section_fit;
use section_fit::{crosses_at, fit_section, settle_on_section};

/// Exact curve type resulting from plane-analytic surface intersection.
#[derive(Debug, Clone)]
pub enum ExactIntersectionCurve {
    /// A circle (plane perpendicular to axis of cylinder/cone/sphere).
    Circle(Circle3D),
    /// An ellipse (plane oblique to cylinder/cone axis).
    Ellipse(Ellipse3D),
    /// Fallback to sampled point chain (torus, degenerate cases).
    Points(Vec<Point3>),
}

/// A plane's section of an analytic surface, from [`exact_plane_section`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum PlaneSection {
    /// A circle.
    Circle(Circle3D),
    /// An ellipse.
    Ellipse(Ellipse3D),
    /// A straight section: a cylinder's ruling, running both ways from the
    /// line's origin, or with `ray` a cone's ruling, running from the apex at
    /// the line's origin along its direction only.
    Line {
        /// The ruling's line.
        line: Line3D,
        /// Whether the section runs one way only, from the line's origin.
        ray: bool,
    },
    /// The one point where the plane touches the surface.
    Point(Point3),
    /// A sampled chain of points along the section.
    Points(Vec<Point3>),
    /// A cone's parabola.
    Parabola(Parabola3D),
    /// The branch of a cone's hyperbola on the cone's one nappe, which the
    /// hyperbola's `u_axis` points to from its centre.
    Hyperbola(Hyperbola3D),
}

impl From<ExactIntersectionCurve> for PlaneSection {
    fn from(curve: ExactIntersectionCurve) -> Self {
        match curve {
            ExactIntersectionCurve::Circle(circle) => Self::Circle(circle),
            ExactIntersectionCurve::Ellipse(ellipse) => Self::Ellipse(ellipse),
            ExactIntersectionCurve::Points(points) => Self::Points(points),
        }
    }
}

/// A plane's exact section of an analytic surface.
///
/// It includes the sections [`exact_plane_analytic`] has no curve for: the
/// rulings a plane along a cylinder's axis or through a cone's apex cuts,
/// and the point where a plane touches a sphere or meets a cone at its apex
/// alone. A cone's parabola and hyperbola come back as curves, which
/// [`exact_plane_analytic`] samples into points, except in a band about the
/// parabola where neither form keeps its digits.
///
/// The plane is `dot(normal, p) = d`, with `normal` of unit length.
///
/// # Errors
///
/// Returns an error if a section's curve cannot be built.
pub fn exact_plane_section(
    surface: AnalyticSurface<'_>,
    normal: Vec3,
    d: f64,
) -> Result<Vec<PlaneSection>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) if normal.dot(cyl.axis()).abs() < 1e-10 => {
            return plane_cylinder_rulings(cyl, normal, d);
        }
        AnalyticSurface::Cone(cone) if plane_holds_apex(cone, normal, d) => {
            return plane_cone_apex_rulings(cone, normal);
        }
        AnalyticSurface::Cone(cone) => {
            if let Some(open) = plane_cone_open_section(cone, normal, d)? {
                return Ok(open);
            }
        }
        AnalyticSurface::Sphere(sphere) => {
            let h = dot_np(normal, sphere.center()) - d;
            let r = sphere.radius();
            if (r - h.abs()).abs() <= TOUCH * r {
                return Ok(vec![PlaneSection::Point(Point3::new(
                    h.madd(-normal.x(), sphere.center().x()),
                    h.madd(-normal.y(), sphere.center().y()),
                    h.madd(-normal.z(), sphere.center().z()),
                ))]);
            }
        }
        AnalyticSurface::Cylinder(_) | AnalyticSurface::Torus(_) => {}
    }
    Ok(exact_plane_analytic(surface, normal, d)?
        .into_iter()
        .map(PlaneSection::from)
        .collect())
}

/// How far a cone section's `e² − 1` (its eccentricity squared, less one)
/// may sit from zero for the section to be read as a parabola.
const PARABOLIC: f64 = 1e-12;

/// Where a cone's hyperbola starts, in `e² − 1`: its centre lies the
/// plane's offset over the conic coefficient away, and nearer zero its
/// vertex loses digits.
const HYPERBOLIC: f64 = 1e-6;

/// A plane's parabola or hyperbola section of a cone, on the cone's one
/// nappe, worked in the frame of [`exact_plane_cone`]. `None` for a circle
/// or an ellipse, in the band between [`PARABOLIC`] and [`HYPERBOLIC`], and
/// for a curve whose points do not hold to the cone (a cone so narrow or so
/// flat that the formulas lose their digits), all of which go to the
/// sampled section.
#[allow(clippy::many_single_char_names)]
fn plane_cone_open_section(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Option<Vec<PlaneSection>>, MathError> {
    let axis = cone.axis();
    let c = normal.dot(axis);
    let p2 = c.madd(-c, 1.0).max(0.0);
    let p = p2.sqrt();
    let rise = cone.half_angle().sin();
    let k = rise * rise;
    let a_coeff = p2 - k;
    let eccentric = a_coeff / k;
    let m = axis - normal * c;
    let m_len = m.length();
    if m_len < 1e-12 || eccentric < -PARABOLIC || (eccentric > PARABOLIC && eccentric <= HYPERBOLIC)
    {
        return Ok(None);
    }
    let e1 = m * (1.0 / m_len);
    let e2 = normal.cross(e1);
    let apex = cone.apex();
    let e = d - dot_np(normal, apex);
    let origin = apex + normal * e;
    // A point `s` along `e1` from `origin` lies `e·c + s·p` up the axis from
    // the apex, which the real nappe keeps non-negative.
    let real = |s: f64| e.madd(c, s * p) >= 0.0;
    // On the one nappe a point's height up the axis is `rise` times its
    // distance from the apex.
    let holds = |q: Point3| {
        let w = q - apex;
        let reach = w.length();
        reach.is_finite() && (w.dot(axis) - rise * reach).abs() <= 1e-9 * (1.0 + reach)
    };
    if eccentric <= PARABOLIC {
        // k t² = 2ecp s + e²(c² − k): the vertex at t = 0, opening along e1
        // with the sign of ecp, `1 / 4f = k / 2|ecp|`.
        let ecp = e * c * p;
        let vertex_s = -e * (c * c - k) / (2.0 * c * p);
        let focal = ecp.abs() / (2.0 * k);
        if !(vertex_s.is_finite() && focal > 0.0 && focal.is_finite()) {
            return Ok(None);
        }
        if !real(vertex_s) {
            return Ok(Some(Vec::new()));
        }
        let parabola = Parabola3D::with_axes(origin + e1 * vertex_s, e1 * ecp.signum(), e2, focal)?;
        if ![0.0, focal, -focal]
            .iter()
            .all(|&t| holds(parabola.evaluate(t)))
        {
            return Ok(None);
        }
        return Ok(Some(vec![PlaneSection::Parabola(parabola)]));
    }
    // A(s − s_c)² − k t² = e²k(1 − k)/A: the branches' vertices lie a real
    // semi-axis either side of the centre, one on each nappe.
    let centre_s = -e * c * p / a_coeff;
    let rhs = e * e * k * (1.0 - k) / a_coeff;
    let (real_axis, imaginary_axis) = ((rhs / a_coeff).sqrt(), (rhs / k).sqrt());
    if !(real_axis > 0.0 && imaginary_axis > 0.0 && centre_s.is_finite()) {
        return Ok(None);
    }
    let toward = if real(centre_s + real_axis) { e1 } else { -e1 };
    let hyperbola = Hyperbola3D::with_axes(
        origin + e1 * centre_s,
        real_axis,
        imaginary_axis,
        toward,
        e2,
    )?;
    if ![0.0, 1.0, -1.0]
        .iter()
        .all(|&t| holds(hyperbola.evaluate(t)))
    {
        return Ok(None);
    }
    Ok(Some(vec![PlaneSection::Hyperbola(hyperbola)]))
}

/// Rounding band, relative, within which a plane counts as touching a
/// surface or passing through a cone's apex.
const TOUCH: f64 = 4.0 * f64::EPSILON;

/// The rulings a plane along a cylinder's axis cuts from it: two lines, the
/// one it touches along, or none.
fn plane_cylinder_rulings(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<PlaneSection>, MathError> {
    // n·P(u, 0) = n·O + r·(a cos u + b sin u): the plane holds the ruling at
    // u where cos(u − φ) = (d − n·O) / (r·√(a² + b²)).
    let (a, b) = (normal.dot(cyl.x_axis()), normal.dot(cyl.y_axis()));
    let amp = a.hypot(b);
    if amp < 1e-12 {
        return Ok(vec![]);
    }
    let ratio = (d - dot_np(normal, cyl.origin())) / (cyl.radius() * amp);
    if ratio.abs() > 1.0 + TOUCH {
        return Ok(vec![]);
    }
    let phi = b.atan2(a);
    let spread = ratio.clamp(-1.0, 1.0).acos();
    let at = if ratio.abs() >= 1.0 - TOUCH {
        vec![phi + spread]
    } else {
        vec![phi + spread, phi - spread]
    };
    at.into_iter()
        .map(|u| {
            Ok(PlaneSection::Line {
                line: Line3D::new(cyl.evaluate(u, 0.0), cyl.axis())?,
                ray: false,
            })
        })
        .collect()
}

/// Whether a plane passes through a cone's apex, but for rounding.
fn plane_holds_apex(cone: &ConicalSurface, normal: Vec3, d: f64) -> bool {
    let apex = cone.apex() - Point3::new(0.0, 0.0, 0.0);
    (normal.dot(apex) - d).abs() <= TOUCH * (1.0 + apex.length() + d.abs())
}

/// The rulings a plane through a cone's apex cuts from its nappe, as rays
/// from the apex: two, the one it touches along, or none, when the plane
/// meets the cone at the apex alone.
fn plane_cone_apex_rulings(
    cone: &ConicalSurface,
    normal: Vec3,
) -> Result<Vec<PlaneSection>, MathError> {
    // g(u) = cos a·(cos u·X + sin u·Y) + sin a·axis, so n·g(u) = 0 reads
    // A cos u + B sin u = −C.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let a = cos_a * normal.dot(cone.x_axis());
    let b = cos_a * normal.dot(cone.y_axis());
    let c = sin_a * normal.dot(cone.axis());
    let amp = a.hypot(b);
    let apex = cone.apex();
    if amp < 1e-12 * c.abs().max(1.0) {
        return Ok(vec![PlaneSection::Point(apex)]);
    }
    let ratio = -c / amp;
    if ratio.abs() > 1.0 + TOUCH {
        return Ok(vec![PlaneSection::Point(apex)]);
    }
    let phi = b.atan2(a);
    let spread = ratio.clamp(-1.0, 1.0).acos();
    let at = if ratio.abs() >= 1.0 - TOUCH {
        vec![phi + spread]
    } else {
        vec![phi + spread, phi - spread]
    };
    at.into_iter()
        .map(|u| {
            Ok(PlaneSection::Line {
                line: Line3D::new(apex, cone.evaluate(u, 1.0) - apex)?,
                ray: true,
            })
        })
        .collect()
}

/// Compute exact intersection curves between a plane and an analytic surface.
///
/// Returns exact `Circle3D` or `Ellipse3D` where possible, falling back to
/// sampled points for complex cases (torus).
///
/// A plane along a cylinder's axis or through a cone's apex, whose sections
/// are straight, and a plane touching a sphere get no curve here;
/// [`exact_plane_section`] returns those sections.
///
/// The plane is defined by `dot(normal, p) = d`.
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn exact_plane_analytic(
    surface: AnalyticSurface<'_>,
    plane_normal: Vec3,
    plane_d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    exact_plane_analytic_reaching(surface, plane_normal, plane_d, 0.0)
}

/// [`exact_plane_analytic`] with a cone's sampled hyperbola or parabola
/// carried at least `reach` from the apex, so it spans whatever faces the
/// caller will trim it to.
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn exact_plane_analytic_reaching(
    surface: AnalyticSurface<'_>,
    plane_normal: Vec3,
    plane_d: f64,
    reach: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => exact_plane_cylinder(cyl, plane_normal, plane_d),
        AnalyticSurface::Sphere(sphere) => exact_plane_sphere(sphere, plane_normal, plane_d),
        AnalyticSurface::Cone(cone) if plane_holds_apex(cone, plane_normal, plane_d) => Ok(vec![]),
        AnalyticSurface::Cone(cone) => exact_plane_cone(cone, plane_normal, plane_d, reach),
        AnalyticSurface::Torus(torus) => {
            if let Some(circles) = exact_plane_torus(torus, plane_normal, plane_d)? {
                return Ok(circles);
            }
            if let Some(loops) = plane_torus_winding_loops(torus, plane_normal, plane_d, 128) {
                return Ok(loops
                    .into_iter()
                    .map(ExactIntersectionCurve::Points)
                    .collect());
            }
            // Other torus sections are degree-4 — fall back to sampling.
            let chains = sample_plane_torus(torus, plane_normal, plane_d)?;
            Ok(chains
                .into_iter()
                .map(ExactIntersectionCurve::Points)
                .collect())
        }
    }
}

/// The plane-torus sections that are circles:
///
/// - a plane across the axis at height `h` from the centre, `|h| < r`: the
///   two circles of radius `R ± sqrt(r² − h²)` about the axis;
/// - a plane through the axis: the two tube cross-sections of radius `r`,
///   `R` either side of the axis.
///
/// A plane across the axis tangent to the tube's top or bottom touches it
/// along the one circle of radius `R`, ring or spindle.
///
/// `Some` of no curves for a plane across the axis that misses the tube;
/// `None` for any other plane, and for a section across the axis whose inner
/// circle would reach the axis.
fn exact_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let len = normal.length();
    let n = normal.normalize()?;
    let d = d / len;
    let axis = torus.z_axis();
    let center = torus.center();
    let (big, small) = (torus.major_radius(), torus.minor_radius());
    let height = d - dot_np(n, center);
    let along = n.dot(axis);
    if along.abs() > 1.0 - 1e-10 {
        if height.abs() >= small - 1e-10 * small {
            if height.abs() > small + 1e-10 * small {
                return Ok(Some(Vec::new()));
            }
            // Tangent to the tube's top or bottom: the plane touches the
            // torus along the one circle of its own radius. That circle lies
            // on the torus within the plane's distance from the tube's top,
            // so it stands for the section while that distance is within the
            // linear tolerance, whatever the scale. A tilted plane's section
            // is no circle.
            if big <= 1e-10 * small
                || axis.cross(n).length() > 1e-12
                || (small - height.abs()).abs() > crate::tolerance::Tolerance::default().linear
            {
                return Ok(None);
            }
            let middle = center + n * height;
            return Ok(Some(vec![ExactIntersectionCurve::Circle(Circle3D::new(
                middle, n, big,
            )?)]));
        }
        let reach = small.madd(small, -(height * height)).sqrt();
        if big - reach <= 1e-10 * big {
            return Ok(None);
        }
        let middle = center + n * height;
        return Ok(Some(vec![
            ExactIntersectionCurve::Circle(Circle3D::new(middle, n, big + reach)?),
            ExactIntersectionCurve::Circle(Circle3D::new(middle, n, big - reach)?),
        ]));
    }
    if along.abs() < 1e-10 && height.abs() < 1e-10 * (big + small) {
        let out = axis.cross(n).normalize()?;
        return Ok(Some(vec![
            ExactIntersectionCurve::Circle(Circle3D::new(center + out * big, n, small)?),
            ExactIntersectionCurve::Circle(Circle3D::new(center - out * big, n, small)?),
        ]));
    }
    Ok(None)
}

/// Exact plane-cylinder intersection.
///
/// - Plane perpendicular to axis → `Circle3D`
/// - Plane oblique to axis → `Ellipse3D`
/// - Plane parallel to axis → `Points` fallback (0 or 2 lines)
fn exact_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    let axis = cyl.axis();
    let cos_theta = normal.dot(axis).abs();
    let r = cyl.radius();

    if cos_theta < 1e-10 {
        // A plane along the axis cuts straight rulings, which
        // `exact_plane_section` returns.
        return Ok(vec![]);
    }

    // Find where axis intersects the plane: axis_point + t*axis, dot(normal, P) = d
    // t = (d - dot(normal, origin)) / dot(normal, axis)
    let n_dot_axis = normal.dot(axis);
    let n_dot_origin = dot_np(normal, cyl.origin());
    let t = (d - n_dot_origin) / n_dot_axis;
    let center_on_axis = Point3::new(
        cyl.origin().x() + t * axis.x(),
        cyl.origin().y() + t * axis.y(),
        cyl.origin().z() + t * axis.z(),
    );

    if cos_theta > 1.0 - 1e-10 {
        // Plane perpendicular to axis → Circle
        let circle = Circle3D::new(center_on_axis, normal, r)?;
        Ok(vec![ExactIntersectionCurve::Circle(circle)])
    } else {
        // Oblique plane → Ellipse
        // Semi-minor = r (the cylinder radius, unchanged)
        // Semi-major = r / cos(θ) where θ = angle between plane normal and axis
        let semi_minor = r;
        let semi_major = r / cos_theta;

        // The major axis direction lies in the intersection of the plane
        // with the plane containing the axis and the plane normal.
        // It's the projection of the axis onto the cutting plane, normalized.
        let axis_proj = Vec3::new(
            axis.x() - n_dot_axis * normal.x(),
            axis.y() - n_dot_axis * normal.y(),
            axis.z() - n_dot_axis * normal.z(),
        );
        let u_axis = axis_proj.normalize()?;
        let v_axis = normal.cross(u_axis);

        let ellipse = Ellipse3D::with_axes(
            center_on_axis,
            normal,
            semi_major,
            semi_minor,
            u_axis,
            v_axis,
        )?;
        Ok(vec![ExactIntersectionCurve::Ellipse(ellipse)])
    }
}

/// Exact plane-sphere intersection.
///
/// Always produces a `Circle3D` (or empty if no intersection).
fn exact_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    let h = dot_np(normal, sphere.center()) - d;
    let r = sphere.radius();

    // A plane touching the sphere meets it at a point, which
    // `exact_plane_section` returns.
    if r - h.abs() <= TOUCH * r {
        return Ok(vec![]);
    }

    let circle_r = (r.madd(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.madd(-normal.x(), sphere.center().x()),
        h.madd(-normal.y(), sphere.center().y()),
        h.madd(-normal.z(), sphere.center().z()),
    );

    let circle = Circle3D::new(circle_center, normal, circle_r)?;
    Ok(vec![ExactIntersectionCurve::Circle(circle)])
}

/// Exact sphere-sphere intersection: the circle on the two spheres' radical
/// plane, its normal pointing from the first centre to the second. `None`
/// for one sphere twice (the same surface, no curve).
///
/// The circle is built from the smaller sphere, whose radius bounds it, and
/// from the centres' offset alone, so neither the operands' order nor where
/// the pair sits in space decides whether a small circle survives.
///
/// # Errors
///
/// Returns an error if the circle cannot be built.
pub fn exact_sphere_sphere(
    s1: &SphericalSurface,
    s2: &SphericalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let delta = s2.center() - s1.center();
    let d = delta.length();
    let (r1, r2) = (s1.radius(), s2.radius());
    let scale = r1.max(r2);
    if d <= 1e-12 * scale {
        // Concentric spheres of different radii never meet.
        return Ok(((r1 - r2).abs() > 1e-12 * scale).then(Vec::new));
    }
    let normal = delta * (1.0 / d);
    // From the smaller sphere's centre toward the larger's, the radical plane
    // lies `along` away.
    let (centre, toward, r_small, r_big) = if r1 <= r2 {
        (s1.center(), normal, r1, r2)
    } else {
        (s2.center(), -normal, r2, r1)
    };
    let along = d.madd(d, r_small.madd(r_small, -(r_big * r_big))) / (2.0 * d);
    if along.abs() > r_small - 1e-10 * scale.max(1.0) {
        return Ok(Some(Vec::new()));
    }
    let radius = r_small.madd(r_small, -(along * along)).sqrt();
    let circle = Circle3D::new(centre + toward * along, normal, radius)?;
    Ok(Some(vec![ExactIntersectionCurve::Circle(circle)]))
}

/// Exact plane-cone intersection.
///
/// The conic type is set by the cone's half-opening angle from the axis
/// (`γ = π/2 − half_angle`) versus the plane-axis angle `ψ`:
/// - Plane perpendicular to axis (`ψ = π/2`) → `Circle3D`
/// - `ψ > γ` (ellipse) → closed-form `Ellipse3D`
/// - `ψ ≤ γ` (parabola/hyperbola) → bounded single-branch `Points` (one chain
///   per branch — a hyperbola's two nappes never share a chain)
fn exact_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
    reach: f64,
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    let axis = cone.axis();
    let cos_theta = normal.dot(axis).abs();
    let half_angle = cone.half_angle();

    if cos_theta > 1.0 - 1e-10 {
        // Plane perpendicular to axis → Circle
        // Find where axis meets the plane
        let n_dot_axis = normal.dot(axis);
        let n_dot_apex = dot_np(normal, cone.apex());
        let t = (d - n_dot_apex) / n_dot_axis;

        // t is the signed distance from apex to plane along the axis.
        // The real cone is a single nappe; the perpendicular-plane section is a
        // circle whose radius follows from the axial offset |t|.
        // |t| ≈ 0 means the plane passes through the apex → degenerate point.
        if t.abs() < 1e-10 {
            return Ok(vec![]);
        }

        let center = Point3::new(
            cone.apex().x() + t * axis.x(),
            cone.apex().y() + t * axis.y(),
            cone.apex().z() + t * axis.z(),
        );
        // half_angle is the angle from the radial plane to the surface.
        // Axial distance t = v * sin(half_angle), so v = t / sin(half_angle).
        // Radius at v = v * cos(half_angle) = t * cos(half_angle) / sin(half_angle).
        let circle_r = t.abs() * half_angle.cos() / half_angle.sin();
        if circle_r < 1e-15 {
            return Ok(vec![]);
        }

        let circle = Circle3D::new(center, normal, circle_r)?;
        return Ok(vec![ExactIntersectionCurve::Circle(circle)]);
    }

    // Oblique plane. Classify the conic in the plane-aligned frame.
    //
    // Decompose the (unit) axis as a = c·n + p·e1, where c = n·a, e1 is the unit
    // in-plane projection of the axis, and p = |projection| = sqrt(1−c²). Write a
    // point Q on the plane as Q = apex + e·n + s·e1 + t·e2 (e = d − n·apex,
    // e2 = n×e1). The cone equation (w·a)² = cos²γ·(w·w) with k = cos²γ =
    // sin²(half_angle) reduces to (no s·t cross term, since e1/e2 align with the
    // conic axes):
    //     (p²−k)·s² + 2ecp·s + e²(c²−k) = k·t²
    // The s² coefficient A = p²−k = sin²θ − sin²(half_angle) sets the type:
    // A < 0 → ellipse, A = 0 → parabola, A > 0 → hyperbola.
    let c = normal.dot(axis);
    let p2 = (1.0 - c * c).max(0.0);
    let p = p2.sqrt();
    let k = half_angle.sin().powi(2);
    let a_coeff = p2 - k;

    // Build the plane-aligned frame e1 (in-plane axis projection), e2 = n×e1.
    let m = Vec3::new(
        axis.x() - c * normal.x(),
        axis.y() - c * normal.y(),
        axis.z() - c * normal.z(),
    );
    let m_len = m.length();
    if m_len < 1e-12 {
        // Axis parallel to normal — handled by the perpendicular branch above;
        // fall back to sampling for safety.
        let chains = sample_plane_cone(cone, normal, d, reach)?;
        return Ok(chains
            .into_iter()
            .map(ExactIntersectionCurve::Points)
            .collect());
    }
    let e1 = m * (1.0 / m_len);
    let e2 = normal.cross(e1);
    let apex = cone.apex();
    let e = d - dot_np(normal, apex);

    // Ellipse → closed form. A = p²−k < 0 with a margin to keep the
    // near-parabolic regime on the robust sampled path.
    if a_coeff < -1e-9 {
        let abs_a = -a_coeff; // = k − p² > 0
        // Real-nappe guard: in the ellipse regime n·g(u) keeps constant sign(c),
        // so v = e/(n·g) ≥ 0 only when e and c share a sign. When e·c < 0 the
        // plane is offset to the far side of the apex from the cone's opening —
        // the section lies entirely on the phantom nappe, so there is no real
        // curve (RHS below is positive regardless of sign, so it can't catch this).
        if e * c < 0.0 {
            return Ok(vec![]);
        }
        // |A|(s − s_c)² + k·t² = RHS, with s_c = ecp/|A| and
        // RHS = e²·k·(1−k)/|A| (always > 0 for a real ellipse).
        let s_c = e * c * p / abs_a;
        let rhs = e * e * k * (1.0 - k) / abs_a;
        if rhs <= 0.0 {
            return Ok(vec![]);
        }
        let semi_s = (rhs / abs_a).sqrt(); // extent along e1
        let semi_t = (rhs / k).sqrt(); // extent along e2
        if semi_s < 1e-12 || semi_t < 1e-12 {
            return Ok(vec![]);
        }
        let center = apex + normal * e + e1 * s_c;
        let (semi_major, semi_minor, u_axis, v_axis) = if semi_s >= semi_t {
            (semi_s, semi_t, e1, e2)
        } else {
            (semi_t, semi_s, e2, e1)
        };
        let ellipse = Ellipse3D::with_axes(center, normal, semi_major, semi_minor, u_axis, v_axis)?;
        return Ok(vec![ExactIntersectionCurve::Ellipse(ellipse)]);
    }

    // Parabola / hyperbola (and the near-parabolic ellipse margin): the section
    // is unbounded, so emit bounded, branch-separated sample chains.
    let chains = sample_plane_cone(cone, normal, d, reach)?;
    Ok(chains
        .into_iter()
        .map(ExactIntersectionCurve::Points)
        .collect())
}

/// The exact arc from `from` to `to` of a plane's parabola or hyperbola
/// section of a cone, both ends on one branch of it, as a rational quadratic
/// NURBS.
///
/// A hyperbola `x = a cosh φ, y = b sinh φ` (in the plane frame of
/// `exact_plane_cone`) is cut into pieces of at most one unit of `φ`, each
/// a conic Bézier: its middle point is where the end tangents meet and its
/// middle weight is the cosh of half the piece's span. A parabola is one
/// polynomial quadratic. `None` for an elliptic or circular section, a plane
/// through the apex, or ends off one branch of the section.
///
/// # Errors
///
/// Returns an error if the plane normal is zero or the curve cannot be built.
#[allow(clippy::many_single_char_names)]
pub fn plane_cone_conic_arc(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
    from: Point3,
    to: Point3,
) -> Result<Option<NurbsCurve>, MathError> {
    let len = normal.length();
    if len < 1e-15 {
        return Err(MathError::ZeroVector);
    }
    let (normal, d) = (normal * (1.0 / len), d / len);
    let axis = cone.axis();
    let c = normal.dot(axis);
    let p2 = (1.0 - c * c).max(0.0);
    let p = p2.sqrt();
    let k = cone.half_angle().sin().powi(2);
    let a_coeff = p2 - k;
    let m = Vec3::new(
        axis.x() - c * normal.x(),
        axis.y() - c * normal.y(),
        axis.z() - c * normal.z(),
    );
    let m_len = m.length();
    if m_len < 1e-12 || a_coeff < -1e-9 {
        return Ok(None);
    }
    let e1 = m * (1.0 / m_len);
    let e2 = normal.cross(e1);
    let apex = cone.apex();
    let e = d - dot_np(normal, apex);
    let origin = apex + normal * e;
    let plane_st = |q: Point3| {
        let w = q - origin;
        (w.dot(e1), w.dot(e2))
    };
    let ((s0, t0), (s1, t1)) = (plane_st(from), plane_st(to));
    let scale = s0.abs().max(t0.abs()).max(s1.abs()).max(t1.abs()).max(1.0);
    if e.abs() < 1e-9 * scale || (from - to).length() <= 1e-9 * scale {
        return Ok(None);
    }
    let point = |s: f64, t: f64| origin + e1 * s + e2 * t;
    let on_curve = |q: Point3, r: Point3| (q - r).length() <= 1e-6 * scale;
    let (control, weights) = if a_coeff.abs() <= 1e-9 {
        // (p² − k) s² vanishes: 2ecp·s + e²(c² − k) = k·t², s = α t² + β.
        let lin = 2.0 * e * c * p;
        if lin.abs() < 1e-12 * scale {
            return Ok(None);
        }
        let (alpha, beta) = (k / lin, -e * e * (c * c - k) / lin);
        if !on_curve(point(alpha * t0 * t0 + beta, t0), from)
            || !on_curve(point(alpha * t1 * t1 + beta, t1), to)
        {
            return Ok(None);
        }
        let mid = point(alpha * t0 * t1 + beta, 0.5 * (t0 + t1));
        (vec![from, mid, to], vec![1.0; 3])
    } else {
        // A (s − s_c)² − k t² = R with R = e² k (1 − k) / A.
        let s_c = -e * c * p / a_coeff;
        let r = e * e * k * (1.0 - k) / a_coeff;
        if r <= 0.0 {
            return Ok(None);
        }
        let (a, b) = ((r / a_coeff).sqrt(), (r / k).sqrt());
        let (x0, x1) = (s0 - s_c, s1 - s_c);
        if x0 * x1 <= 0.0 {
            return Ok(None);
        }
        let side = x0.signum();
        let hyperbola = |phi: f64| point(s_c + side * a * phi.cosh(), b * phi.sinh());
        let (phi0, phi1) = ((t0 / b).asinh(), (t1 / b).asinh());
        if !on_curve(hyperbola(phi0), from) || !on_curve(hyperbola(phi1), to) {
            return Ok(None);
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pieces = ((phi1 - phi0).abs().ceil() as usize).max(1);
        let mut control = vec![from];
        let mut weights = vec![1.0];
        for i in 0..pieces {
            #[allow(clippy::cast_precision_loss)]
            let (fa, fb) = (i as f64 / pieces as f64, (i + 1) as f64 / pieces as f64);
            let (pa, pb) = (phi0 + (phi1 - phi0) * fa, phi0 + (phi1 - phi0) * fb);
            let (mid, half) = (0.5 * (pa + pb), 0.5 * (pb - pa));
            let w = half.cosh();
            control.push(point(s_c + side * a * mid.cosh() / w, b * mid.sinh() / w));
            weights.push(w);
            control.push(if i + 1 == pieces { to } else { hyperbola(pb) });
            weights.push(1.0);
        }
        (control, weights)
    };
    let pieces = (control.len() - 1) / 2;
    let mut knots = vec![0.0; 3];
    for i in 1..pieces {
        #[allow(clippy::cast_precision_loss)]
        knots.extend([i as f64; 2]);
    }
    #[allow(clippy::cast_precision_loss)]
    knots.extend([pieces as f64; 3]);
    let curve = NurbsCurve::new(2, knots, control, weights)?;
    // The closed forms drop terms that vanish only on the exact conic (a
    // barely elliptic section read as a parabola), so the arc must meet the
    // cone between its ends too: a point at radius ρ and height h off the
    // apex lies |ρ sin α − |h| cos α| from it.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let off_cone = |q: Point3| {
        let w = q - apex;
        let h = w.dot(axis);
        (w - axis * h)
            .length()
            .madd(sin_a, -(h.abs() * cos_a))
            .abs()
    };
    for i in 0..pieces {
        for f in [0.25, 0.5, 0.75] {
            #[allow(clippy::cast_precision_loss)]
            if off_cone(curve.evaluate(i as f64 + f)) > 1e-9 * scale {
                return Ok(None);
            }
        }
    }
    Ok(Some(curve))
}

/// Reference to an analytic surface for intersection dispatch.
#[derive(Clone, Copy)]
pub enum AnalyticSurface<'a> {
    /// Cylindrical surface reference.
    Cylinder(&'a CylindricalSurface),
    /// Conical surface reference.
    Cone(&'a ConicalSurface),
    /// Spherical surface reference.
    Sphere(&'a SphericalSurface),
    /// Toroidal surface reference.
    Torus(&'a ToroidalSurface),
}

/// Compute `n . p` treating a `Point3` as a position vector.
fn dot_np(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// Intersect a plane with an analytic surface.
///
/// The plane is defined by `dot(normal, p) = d`.
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn intersect_plane_analytic(
    surface: AnalyticSurface<'_>,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => intersect_plane_cylinder(cyl, normal, d),
        AnalyticSurface::Cone(cone) => intersect_plane_cone(cone, normal, d),
        AnalyticSurface::Sphere(sphere) => intersect_plane_sphere(sphere, normal, d),
        AnalyticSurface::Torus(torus) => intersect_plane_torus(torus, normal, d),
    }
}

/// Sample points on the plane-analytic intersection without NURBS curve fitting.
///
/// Returns chains of ordered 3D sample points. Each chain is one connected
/// component of the intersection curve. This is much faster than
/// `intersect_plane_analytic` when only sample points are needed (e.g. for
/// boolean intersection segment generation).
///
/// # Errors
///
/// Returns an error if the intersection computation fails.
pub fn sample_plane_analytic(
    surface: AnalyticSurface<'_>,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    match surface {
        AnalyticSurface::Cylinder(cyl) => sample_plane_cylinder(cyl, normal, d),
        AnalyticSurface::Cone(cone) => sample_plane_cone(cone, normal, d, 0.0),
        AnalyticSurface::Sphere(sphere) => sample_plane_sphere(sphere, normal, d),
        AnalyticSurface::Torus(torus) => sample_plane_torus(torus, normal, d),
    }
}

/// Sample the plane-cylinder intersection as ordered 3D points.
#[allow(clippy::cast_precision_loss, clippy::unnecessary_wraps)]
fn sample_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let n_samples = 64_usize;
    let mut points = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        let base = cyl.evaluate(u, 0.0);
        let n_dot_axis = normal.dot(cyl.axis());
        let n_dot_base = dot_np(normal, base);

        if n_dot_axis.abs() < 1e-12 {
            if (n_dot_base - d).abs() < 1e-6 {
                points.push(base);
            }
        } else {
            let v = (d - n_dot_base) / n_dot_axis;
            if v.abs() <= 100.0 {
                points.push(cyl.evaluate(u, v));
            }
        }
    }

    if points.len() < 2 {
        Ok(vec![])
    } else {
        Ok(vec![points])
    }
}

/// Sample the plane-sphere intersection as ordered 3D points.
#[allow(clippy::cast_precision_loss)]
fn sample_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let h = dot_np(normal, sphere.center()) - d;
    let r = sphere.radius();

    if h.abs() > r - 1e-10 {
        return Ok(vec![]);
    }

    let circle_r = (r.madd(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.madd(-normal.x(), sphere.center().x()),
        h.madd(-normal.y(), sphere.center().y()),
        h.madd(-normal.z(), sphere.center().z()),
    );

    let basis = Frame3::from_normal(circle_center, normal)?;
    let u_dir = basis.x;
    let v_dir = basis.y;

    let n_samples = 64_usize;
    let mut points = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let theta = TAU * (i as f64) / (n_samples as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        points.push(circle_center + u_dir * (circle_r * cos_t) + v_dir * (circle_r * sin_t));
    }

    Ok(vec![points])
}

/// Sample the plane-cone intersection as ordered 3D points.
///
/// The cone is the single real nappe `v >= 0` of `P(u,v) = apex + v·g(u)`.
/// Along each generator `g(u)` the plane `n·P = d` is linear in `v`, so
/// `v = (d − n·apex) / (n·g(u))`. We keep only `v >= 0` (the phantom `v < 0`
/// nappe is geometrically absent) and `v` below a finite bound (near an
/// asymptote `n·g(u) → 0` so `v → ∞` — those points run off the surface and
/// must be excluded). The angular samples that survive form one contiguous arc
/// (ellipse) or two (parabola/hyperbola, one per branch); each contiguous run
/// is returned as a separate ordered chain so the consumer never stitches two
/// disjoint branches into one curve.
#[allow(clippy::cast_precision_loss, clippy::unnecessary_wraps)]
fn sample_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
    reach: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let apex = cone.apex();
    let n_dot_apex = dot_np(normal, apex);
    let e = d - n_dot_apex;

    // Per-generator solve: along g(u) the plane is linear in v, v = e / (n·g(u)).
    // Sample u densely; keep only the real nappe (v >= 0) and skip near-asymptote
    // generators (n·g(u) ≈ 0 → v → ∞).
    let n_samples = 512_usize;
    let mut vs: Vec<Option<f64>> = Vec::with_capacity(n_samples);
    let mut v_min = f64::INFINITY;
    for i in 0..n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        let g = cone.evaluate(u, 1.0) - apex;
        let n_dot_g = normal.dot(Vec3::new(g.x(), g.y(), g.z()));
        if n_dot_g.abs() < 1e-12 {
            vs.push(None);
            continue;
        }
        let v = e / n_dot_g;
        if v >= -1e-12 {
            let v = v.max(0.0);
            v_min = v_min.min(v);
            vs.push(Some(v));
        } else {
            vs.push(None);
        }
    }

    if !v_min.is_finite() {
        return Ok(Vec::new());
    }

    // Bound the arc around the conic vertex (closest approach to the apex, at
    // v_min). An ellipse is naturally bounded; a parabola/hyperbola is not, so
    // cap the cone radius at a generous multiple of the vertex radius. This is
    // scale-invariant and centred on where any finite cone face's overlap lies;
    // the downstream consumer trims the fitted curve to the actual face AABB, so
    // over-coverage is harmless. The floor handles a vertex at the apex (v_min≈0).
    // A caller that knows its faces asks for their reach: an open hyperbola
    // (a vertex close to the axis) crosses a rim far past eight vertex radii.
    let v_max = (8.0 * v_min).max(v_min + 4.0).max(reach);

    // Per-sample v within the cap; the raw values stay in `vs` for the
    // boundary solve below.
    let kept: Vec<Option<f64>> = vs.iter().map(|v| v.filter(|&v| v <= v_max)).collect();

    let point_at = |u: f64, v: f64| -> Point3 {
        let g = cone.evaluate(u, 1.0) - apex;
        apex + g * v
    };
    #[allow(clippy::cast_precision_loss)]
    let u_of = |i: usize| TAU * (i as f64) / (n_samples as f64);
    let n_dot_g_at = |u: f64| -> f64 {
        let g = cone.evaluate(u, 1.0) - apex;
        normal.dot(Vec3::new(g.x(), g.y(), g.z()))
    };

    if kept.iter().all(Option::is_some) {
        // Closed loop (ellipse regime): emit all points and repeat the first.
        let mut pts: Vec<Point3> = kept
            .iter()
            .enumerate()
            .filter_map(|(i, v)| v.map(|v| point_at(u_of(i), v)))
            .collect();
        if let Some(&first) = pts.first() {
            pts.push(first);
        }
        return Ok(vec![pts]);
    }

    // A hyperbola/parabola tail diverges as 1/(n·g), so between the last kept
    // sample and its dropped neighbour v can leap far past `v_max` in one
    // uniform-u pitch — and any finite face window inside that leap is lost
    // (a taper cone grazed 0.05 by a prism plane lost its entire 0.5-tall
    // section to exactly this aliasing). Extend each run end to the exact
    // `v_max` boundary: bisect u for `n·g(u) = e/v_max` inside the dropped
    // pitch (n·g is monotone there — its extrema sit at the conic vertex,
    // far from any asymptote), then fill the tail with uniform-u samples.
    let tail = |i_end: usize, forward: bool, kept: &[Option<f64>]| -> Vec<Point3> {
        let Some(v_end) = kept[i_end] else {
            return Vec::new();
        };
        let u_end = u_of(i_end);
        #[allow(clippy::cast_precision_loss)]
        let pitch = TAU / (n_samples as f64);
        let u_next = if forward {
            u_end + pitch
        } else {
            u_end - pitch
        };
        let target = e / v_max;
        let h_end = n_dot_g_at(u_end) - target;
        let h_next = n_dot_g_at(u_next) - target;
        if v_end >= v_max || h_end == 0.0 || h_end.signum() == h_next.signum() {
            return Vec::new();
        }
        let (mut lo, mut hi) = (u_end, u_next);
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            if (n_dot_g_at(mid) - target).signum() == h_end.signum() {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let u_star = f64::midpoint(lo, hi);
        let tail_n = 8_usize;
        (1..=tail_n)
            .filter_map(|k| {
                #[allow(clippy::cast_precision_loss)]
                let u = u_end + (u_star - u_end) * (k as f64) / (tail_n as f64);
                let ng = n_dot_g_at(u);
                if ng.abs() < 1e-12 {
                    return None;
                }
                let v = e / ng;
                (v >= -1e-12 && v <= v_max * (1.0 + 1e-9)).then(|| point_at(u, v.max(0.0)))
            })
            .collect()
    };

    // Split into contiguous runs of kept samples, treating the array as
    // circular (rotate past a gap) so a branch straddling u=0 stays whole.
    let gap = kept.iter().position(Option::is_none).unwrap_or(0);
    let mut chains: Vec<Vec<Point3>> = Vec::new();
    let mut run: Vec<usize> = Vec::new();
    let flush = |run: &mut Vec<usize>, chains: &mut Vec<Vec<Point3>>| {
        if run.len() >= 2 {
            let first = run[0];
            let last = run[run.len() - 1];
            let mut pts: Vec<Point3> = tail(first, false, &kept);
            pts.reverse();
            pts.extend(
                run.iter()
                    .filter_map(|&i| kept[i].map(|v| point_at(u_of(i), v))),
            );
            pts.extend(tail(last, true, &kept));
            chains.push(pts);
        }
        run.clear();
    };
    for k in 0..n_samples {
        let idx = (gap + k) % n_samples;
        if kept[idx].is_some() {
            run.push(idx);
        } else {
            flush(&mut run, &mut chains);
        }
    }
    flush(&mut run, &mut chains);
    Ok(chains.into_iter().filter(|c| c.len() >= 2).collect())
}

/// Sample the plane-torus intersection as ordered 3D points.
///
/// Uses the same closed-form loops as `intersect_plane_torus`
/// but skips NURBS curve fitting (the callers here only need the points).
#[allow(clippy::unnecessary_wraps)] // sibling match-arms and `?` callers need `Result`
fn sample_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    Ok(plane_torus_loops(torus, normal, d, 128)
        .into_iter()
        .map(|run| run.into_iter().map(|p| p.point).collect())
        .collect())
}

/// Intersect a plane with a cylindrical surface.
///
/// For each `u` in `[0, 2pi)`, the cylinder point is linear in `v`,
/// so the plane equation `dot(normal, P(u,v)) = d` is linear in `v`
/// and can be solved directly.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_cylinder(
    cyl: &CylindricalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..=n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        // P(u, v) = origin + r*(cos(u)*x + sin(u)*y) + v*axis
        // dot(normal, P) = d  =>  dot(normal, base(u)) + v * dot(normal, axis) = d
        let base = cyl.evaluate(u, 0.0);
        let n_dot_axis = normal.dot(cyl.axis());
        let n_dot_base = dot_np(normal, base);

        if n_dot_axis.abs() < 1e-12 {
            // Plane parallel to axis -- check if base is on plane.
            if (n_dot_base - d).abs() < 1e-6 {
                let pt = base;
                points_3d.push(pt);
                ipoints.push(IntersectionPoint {
                    point: pt,
                    param1: (u, 0.0),
                    param2: (0.0, 0.0),
                });
            }
        } else {
            let v = (d - n_dot_base) / n_dot_axis;
            // Only keep points within a reasonable v range.
            if v.abs() <= 100.0 {
                let pt = cyl.evaluate(u, v);
                points_3d.push(pt);
                ipoints.push(IntersectionPoint {
                    point: pt,
                    param1: (u, v),
                    param2: (0.0, 0.0),
                });
            }
        }
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a spherical surface.
///
/// The intersection of a plane with a sphere is a circle (or empty/point).
/// Computes the circle center, radius, and samples points on it.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_sphere(
    sphere: &SphericalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let h = dot_np(normal, sphere.center()) - d;
    let r = sphere.radius();

    // No intersection if plane is too far from center.
    if h.abs() > r - 1e-10 {
        return Ok(vec![]);
    }

    let circle_r = (r.madd(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.madd(-normal.x(), sphere.center().x()),
        h.madd(-normal.y(), sphere.center().y()),
        h.madd(-normal.z(), sphere.center().z()),
    );

    // Build a local frame on the plane.
    let basis = Frame3::from_normal(circle_center, normal)?;
    let u_dir = basis.x;
    let v_dir = basis.y;

    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..=n_samples {
        let theta = TAU * (i as f64) / (n_samples as f64);
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = circle_center + u_dir * (circle_r * cos_t) + v_dir * (circle_r * sin_t);
        points_3d.push(pt);
        ipoints.push(IntersectionPoint {
            point: pt,
            param1: (theta, 0.0),
            param2: (0.0, 0.0),
        });
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a conical surface.
///
/// Like a cylinder, the cone is linear along each generatrix, so the plane
/// equation is linear in `v` for each fixed `u`.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(clippy::cast_precision_loss)]
pub fn intersect_plane_cone(
    cone: &ConicalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_samples = 64_usize;
    let mut points_3d = Vec::new();
    let mut ipoints = Vec::new();

    for i in 0..n_samples {
        let u = TAU * (i as f64) / (n_samples as f64);
        // P(u, v) = apex + v * dir(u)
        // dot(normal, apex) + v * dot(normal, dir(u)) = d
        let apex = cone.apex();
        let n_dot_apex = dot_np(normal, apex);
        // dir(u) = P(u,1) - apex
        let p1 = cone.evaluate(u, 1.0);
        let dir = p1 - apex;
        let n_dot_dir = normal.dot(dir);

        if n_dot_dir.abs() < 1e-12 {
            continue;
        }

        let v = (d - n_dot_apex) / n_dot_dir;
        // Allow negative v — the cone surface extends in both directions from the apex.
        if v.abs() > 1e-10 && v.abs() < 100.0 {
            let pt = cone.evaluate(u, v);
            points_3d.push(pt);
            ipoints.push(IntersectionPoint {
                point: pt,
                param1: (u, v),
                param2: (0.0, 0.0),
            });
        }
    }

    build_curves_from_points(&points_3d, ipoints)
}

/// Intersect a plane with a toroidal surface.
///
/// The section is a degree-4 curve, but for each `v` the `u` values solve in
/// closed form (see `plane_torus_loops`), so it is sampled by a v-scan
/// and each loop is fitted to a NURBS curve.
///
/// # Errors
///
/// Never returns an error today (curve-fit failures drop the affected loop);
/// the `Result` is kept for signature parity with the other plane-analytic
/// intersectors.
#[allow(clippy::unnecessary_wraps)]
pub fn intersect_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    // The section satisfies a per-v closed form (see `plane_torus_loops`),
    // so scan v and solve u directly instead of a 2D sign-change grid with
    // Newton refinement: O(n) rather than O(n²), and every point is exact.
    let mut curves = Vec::new();
    for ipts in plane_torus_loops(torus, normal, d, 128) {
        let pts: Vec<Point3> = ipts.iter().map(|p| p.point).collect();
        if let Ok(curve) = fit_plane_torus_section(torus, normal, d, &pts) {
            curves.push(IntersectionCurve {
                curve,
                points: ipts,
            });
        }
    }

    Ok(curves)
}

/// The point of a plane's section of a torus that `p`, a point near it,
/// moves to across both surfaces.
///
/// Newton steps take the point onto the plane and onto the tube nearer `p`,
/// both to rounding. Each step is the shortest move that clears both
/// first-order distances, so the point moves across the section and not
/// along it.
///
/// `None` where the two surfaces meet at a tangent (their normals within
/// about 1e-6 of parallel), where `p` lies on the torus's axis or a tube's
/// centre circle, or where the steps do not settle.
#[must_use]
pub fn plane_torus_section_point(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    p: Point3,
) -> Option<Point3> {
    plane_torus_newton(torus, normal, d, p, 0.0)
}

/// [`plane_torus_section_point`], but where the steps so far have moved `p`
/// by less than `short`, the point they reach: the section's point lies
/// within about the square of the last step times the section's curvature
/// of it.
fn plane_torus_newton(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    p: Point3,
    short: f64,
) -> Option<Point3> {
    let scale = normal.length();
    if scale <= 0.0 || !scale.is_finite() {
        return None;
    }
    let n = normal * (1.0 / scale);
    let d = d / scale;
    let (axis, centre) = (torus.z_axis(), torus.center());
    let (big, small) = (torus.major_radius(), torus.minor_radius());
    let radial_of = |x: Point3| {
        let w = x - centre;
        w - axis * axis.dot(w)
    };
    // The tube is held to the side `p` starts on: a spindle torus's sheet
    // across the axis belongs to the far tube circle.
    let side = {
        let radial = radial_of(p);
        let rho = radial.length();
        let h = axis.dot(p - centre);
        if (rho + big).hypot(h) - small < ((rho - big).hypot(h) - small).abs() {
            -1.0
        } else {
            1.0
        }
    };
    // Each surface's distance from `x` and the torus's normal there.
    let residuals = |x: Point3| -> Option<(f64, f64, Vec3)> {
        let radial = radial_of(x);
        let rho = radial.length();
        if rho <= 0.0 {
            return None;
        }
        let to = (x - centre) - radial * (side * big / rho);
        let reach = to.length();
        if reach <= 0.0 {
            return None;
        }
        Some((dot_np(n, x) - d, reach - small, to * (1.0 / reach)))
    };
    let rounding = 16.0 * f64::EPSILON * (big + small + (p - centre).length());
    let mut x = p;
    for _ in 0..16 {
        let (f_plane, f_torus, g) = residuals(x)?;
        if f_plane.abs() <= rounding && f_torus.abs() <= rounding {
            return Some(x);
        }
        let c = n.dot(g);
        let det = c.madd(-c, 1.0);
        if det < 1e-12 {
            return None;
        }
        let along_n = c.madd(-f_torus, f_plane) / det;
        let along_g = c.madd(-f_plane, f_torus) / det;
        x = x - n * along_n - g * along_g;
        if (x - p).length() < short {
            return Some(x);
        }
    }
    let (f_plane, f_torus, _) = residuals(x)?;
    (f_plane.abs() <= 1e3 * rounding && f_torus.abs() <= 1e3 * rounding).then_some(x)
}

/// How far a fitted plane x torus section may stray from the true section
/// between its points, as a fraction of the torus's outer radius.
const PLANE_TORUS_FIT: f64 = 1e-9;

/// The most points a fitted plane x torus section takes.
const PLANE_TORUS_FIT_POINTS: usize = 4096;

/// A cubic through `points`, points of a plane's section of a torus in order
/// along it, and through more points of the section where it needs them.
///
/// The cubic meets the section only at its points, and between two of them
/// it strays by about the fourth power of their spacing, which a tight turn
/// of the section makes large: 25 points along the section of a wall 0.4
/// outside the inner equator of a torus of radii 4 and 1.5 leave the cubic
/// 8e-5 off it. Each pass checks the cubic halfway between each two points:
/// where that point moves onto the section, within half the points'
/// spacing, by more than 1e-9 of the torus's outer radius, the section's
/// point there goes between them. Where it does not move onto the section
/// (on the axis, or where the plane all but touches the torus), it must lie
/// on both the plane and the torus within that bound itself.
///
/// # Errors
///
/// Returns an error if the interpolation fails (fewer than 2 points, or
/// repeated points), if a middle neither moves onto the section nor lies
/// on both surfaces, or if the cubic needs more than 4096 points.
pub fn fit_plane_torus_section(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    points: &[Point3],
) -> Result<NurbsCurve, MathError> {
    let tol = PLANE_TORUS_FIT.madd(
        torus.major_radius() + torus.minor_radius(),
        64.0 * f64::EPSILON * (torus.center() - Point3::new(0.0, 0.0, 0.0)).length(),
    );
    // How far a point lies from the plane, or from the torus, whichever is
    // farther.
    let strays = |p: Point3| {
        let w = p - torus.center();
        let height = torus.z_axis().dot(w);
        let rho = (w - torus.z_axis() * height).length();
        let off_torus = [1.0, -1.0]
            .into_iter()
            .map(|side: f64| {
                (side.madd(-torus.major_radius(), rho).hypot(height) - torus.minor_radius()).abs()
            })
            .fold(f64::INFINITY, f64::min);
        ((dot_np(normal, p) - d) / normal.length())
            .abs()
            .max(off_torus)
    };
    let mut pts = points.to_vec();
    let mut passes = 0;
    loop {
        passes += 1;
        let curve = interpolate(&pts, 3.min(pts.len().saturating_sub(1)))?;
        let params = crate::nurbs::fitting::chord_length_params(&pts);
        let mut finer = Vec::with_capacity(2 * pts.len());
        let mut added = false;
        for (i, pair) in pts.windows(2).enumerate() {
            finer.push(pair[0]);
            let gap = (pair[1] - pair[0]).length();
            let middle = curve.evaluate(f64::midpoint(params[i], params[i + 1]));
            // A point of the section half a spacing or more away lies on
            // another stretch of it (a thin tube's other loop).
            let on = plane_torus_newton(torus, normal, d, middle, 0.25 * tol)
                .filter(|on| (*on - middle).length() < 0.5 * gap);
            match on {
                Some(on) if (on - middle).length() <= tol => {}
                Some(on) => {
                    finer.push(on);
                    added = true;
                }
                None if strays(middle) <= tol => {}
                None => return Err(MathError::ConvergenceFailure { iterations: passes }),
            }
        }
        if !added {
            return Ok(curve);
        }
        if let Some(&last) = pts.last() {
            finer.push(last);
        }
        if finer.len() > PLANE_TORUS_FIT_POINTS {
            return Err(MathError::ConvergenceFailure { iterations: passes });
        }
        pts = finer;
    }
}

/// The fewest and most steps along each branch of a loop a plane cuts from
/// a torus between its two turns.
const PLANE_TORUS_LOOP_SAMPLES: (f64, f64) = (24.0, 512.0);

/// The loops a plane cuts from a torus, each as points on the torus in
/// order, closed by repeating its first point.
///
/// In the torus's own frame let `a = n·X`, `b = n·Y`, `c = n·Z`,
/// `s = hypot(a, b)`, `phi = atan2(b, a)`. Substituting the torus
/// parameterization into `n·P = d` gives
///   `(R + r·cos v)·s·cos(u − phi) + r·c·sin v = d − n·center`,
/// so at each tube angle `v` where `|rhs(v)| <= 1`, with
/// `rhs(v) = (d − n·center − r·c·sin v) / (s·(R + r·cos v))`, the section's
/// two branches are `u = phi ± acos(rhs(v))`. Each run of `v` where the
/// section exists is one loop: out along one branch to where the run ends
/// and the branches meet, and back along the other. A section present at
/// every `v` is two loops winding the tube, one per branch. Every point
/// comes from the closed form, so the loops need no chaining.
///
/// Where the branches touch inside a run (a plane tangent to the tube's
/// inner side, at a saddle) the section crosses itself: each branch passes
/// through that node, and each loop there starts and ends on it.
///
/// When `s ≈ 0` the plane is perpendicular to the axis and the section is up
/// to two full circles at the `v` values solving `r·c·sin v = d − n·center`,
/// sampled by scanning `u`.
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
fn plane_torus_loops(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    n_v: usize,
) -> Vec<Vec<IntersectionPoint>> {
    let big_r = torus.major_radius();
    let small_r = torus.minor_radius();
    let a = normal.dot(torus.x_axis());
    let b = normal.dot(torus.y_axis());
    let c = normal.dot(torus.z_axis());
    let s = a.hypot(b);
    let phi = b.atan2(a);
    let d_local = d - dot_np(normal, torus.center());
    let point = |u: f64, v: f64| IntersectionPoint {
        point: torus.evaluate(u, v),
        param1: (u, v.rem_euclid(TAU)),
        param2: (0.0, 0.0),
    };
    let closed = |mut run: Vec<IntersectionPoint>| {
        run.push(run[0]);
        run
    };

    // Plane perpendicular to the axis: the section is up to two full circles.
    if s < 1e-12 {
        if c.abs() < 1e-12 {
            return Vec::new();
        }
        let sin_v = d_local / (small_r * c);
        if sin_v.abs() > 1.0 + 1e-9 {
            return Vec::new();
        }
        let v0 = sin_v.clamp(-1.0, 1.0).asin();
        let v1 = std::f64::consts::PI - v0;
        let mut vs = vec![v0];
        // Skip the mirror circle when the plane is tangent: at the tube's top
        // v0 == v1, and at its bottom they differ by a whole turn.
        let apart = (v1 - v0).rem_euclid(TAU);
        if apart.min(TAU - apart) > 1e-9 {
            vs.push(v1);
        }
        return vs
            .into_iter()
            .map(|v| {
                closed(
                    (0..n_v)
                        .map(|i| point(TAU * (i as f64) / (n_v as f64), v))
                        .collect(),
                )
            })
            .collect();
    }

    // R + r·cos v > 0 on a ring torus.
    let length = normal.length();
    TorusSection {
        torus,
        phi,
        num: [d_local, 0.0, -small_r * c],
        den: [s * big_r, s * small_r],
        clearance: |p: Point3| (dot_np(normal, p) - d) / length,
        slack: 64.0 * f64::EPSILON * (big_r + small_r + d_local.abs() / length),
    }
    .loops(n_v, PLANE_TORUS_LOOP_SAMPLES)
}

/// A surface's section of a ring torus that holds, at each tube angle `v`,
/// the torus's points with `cos(u − phi) = rhs(v)`: the two branches
/// `u = phi ± acos(rhs(v))` wherever `|rhs(v)| <= 1`, as for a plane
/// ([`plane_torus_loops`]) or a sphere off the axis ([`sphere_torus_section`]).
/// For both, `rhs(v) = (n0 + n1·cos v + n2·sin v) / (d0 + d1·cos v)` with a
/// positive denominator, so `rhs` meets `±1` where a sum of `cos v` and
/// `sin v` vanishes, at up to four tube angles in closed form.
struct TorusSection<'a, F> {
    torus: &'a ToroidalSurface,
    phi: f64,
    /// `n0`, `n1`, `n2`.
    num: [f64; 3],
    /// `d0`, `d1`.
    den: [f64; 2],
    /// A point's signed distance from the surface cutting the torus.
    clearance: F,
    /// The distance rounding can make or hide between the two surfaces.
    slack: f64,
}

impl<F: Fn(Point3) -> f64> TorusSection<'_, F> {
    fn rhs(&self, v: f64) -> f64 {
        let [n0, n1, n2] = self.num;
        let [d0, d1] = self.den;
        let (sin, cos) = v.sin_cos();
        n2.madd(sin, n1.madd(cos, n0)) / d1.madd(cos, d0)
    }

    fn point(&self, u: f64, v: f64) -> IntersectionPoint {
        IntersectionPoint {
            point: self.torus.evaluate(u, v),
            param1: (u.rem_euclid(TAU), v.rem_euclid(TAU)),
            param2: (0.0, 0.0),
        }
    }

    fn branch(&self, v: f64, sign: f64) -> IntersectionPoint {
        self.point(sign.madd(self.rhs(v).clamp(-1.0, 1.0).acos(), self.phi), v)
    }

    /// Where both branches meet at `v`: `acos` is 0 or π there.
    fn node(&self, v: f64) -> IntersectionPoint {
        let u = if self.rhs(v) >= 0.0 {
            self.phi
        } else {
            self.phi + std::f64::consts::PI
        };
        self.point(u, v)
    }

    /// How far the surface lies from the point where the branches would meet
    /// at `v`: zero where the section's run ends, and where it only touches.
    fn depth(&self, v: f64) -> f64 {
        (self.clearance)(self.node(v).point).abs()
    }

    /// The tube angles in `[0, 2π)` where `rhs` is `±1`, in order. Each solves
    /// `α + β·cos v + γ·sin v = 0`, so `cos(v − atan2(γ, β)) = −α / hypot(β, γ)`;
    /// a ratio past 1 by rounding alone is a double root, a touch.
    fn crossings(&self) -> Vec<f64> {
        let [n0, n1, n2] = self.num;
        let [d0, d1] = self.den;
        let mut roots = Vec::with_capacity(4);
        for c in [1.0_f64, -1.0] {
            let (alpha, beta, gamma) = (c.madd(-d0, n0), c.madd(-d1, n1), n2);
            let m = beta.hypot(gamma);
            let ratio = -alpha / m;
            if !ratio.is_finite() || ratio.abs() > 1.0 + 1e-12 {
                continue;
            }
            let theta = gamma.atan2(beta);
            let delta = ratio.clamp(-1.0, 1.0).acos();
            roots.push((theta - delta).rem_euclid(TAU));
            roots.push((theta + delta).rem_euclid(TAU));
        }
        roots.sort_by(f64::total_cmp);
        roots
    }

    /// Each run of `v` where the section lies, from `lo` to `hi` unwrapped,
    /// with the node inside it where its branches touch, and whether it
    /// takes the whole turn. Between two crossings `|rhs|` stays on one side
    /// of 1; a stretch the surface clears or cuts by no more than `slack` is
    /// a touch, a node inside a run and nothing between runs.
    fn runs(&self) -> Vec<(f64, f64, Option<f64>, bool)> {
        let roots = self.crossings();
        let n = roots.len();
        if n == 0 {
            let whole = self.rhs(0.0).abs() <= 1.0 && self.depth(0.0) > self.slack;
            return if whole {
                vec![(0.0, TAU, None, true)]
            } else {
                Vec::new()
            };
        }
        let arcs: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                (
                    roots[i],
                    if i + 1 < n {
                        roots[i + 1]
                    } else {
                        roots[0] + TAU
                    },
                )
            })
            .collect();
        let middle = |i: usize| f64::midpoint(arcs[i].0, arcs[i].1);
        let inside: Vec<bool> = (0..n).map(|i| self.rhs(middle(i)).abs() <= 1.0).collect();
        let shallow: Vec<bool> = (0..n)
            .map(|i| self.depth(middle(i)) <= self.slack)
            .collect();
        let deep_beside = |i: usize, back: bool| {
            (1..n)
                .map(|j| if back { (i + n - j) % n } else { (i + j) % n })
                .find(|&j| !shallow[j])
        };
        let kept: Vec<bool> = (0..n)
            .map(|i| {
                if !shallow[i] {
                    return inside[i];
                }
                matches!(
                    (deep_beside(i, true), deep_beside(i, false)),
                    (Some(a), Some(b)) if inside[a] && inside[b]
                )
            })
            .collect();
        let node_in = |from: usize, len: usize| {
            (0..len).find_map(|j| {
                let i = (from + j) % n;
                let wraps = if from + j >= n { TAU } else { 0.0 };
                shallow[i].then(|| middle(i) + wraps)
            })
        };
        let Some(first) = (0..n).find(|&i| kept[i] && !kept[(i + n - 1) % n]) else {
            return if kept.iter().all(|&k| k) {
                let node = node_in(0, n);
                let lo = node.unwrap_or(roots[0]);
                vec![(lo, lo + TAU, node, true)]
            } else {
                Vec::new()
            };
        };
        let mut runs = Vec::new();
        let mut j = 0;
        while j < n {
            let i = (first + j) % n;
            if !kept[i] {
                j += 1;
                continue;
            }
            let len = (0..n - j).take_while(|&k| kept[(i + k) % n]).count();
            let last = i + len - 1;
            let hi = arcs[last % n].1 + if last >= n { TAU } else { 0.0 };
            runs.push((arcs[i].0, hi, node_in(i, len), false));
            j += len;
        }
        runs
    }

    /// Each run of the section as one loop, out along one branch to where
    /// the run ends and back along the other, closed by repeating its first
    /// point; a run taking the whole turn is two loops winding the tube.
    /// Where the branches touch inside a run the section crosses itself:
    /// each branch passes through that node, and each loop there starts and
    /// ends on it.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn loops(&self, n_v: usize, (fewest, most): (f64, f64)) -> Vec<Vec<IntersectionPoint>> {
        let closed = |mut run: Vec<IntersectionPoint>| {
            run.push(run[0]);
            run
        };
        let step = TAU / (n_v as f64);
        let mut loops = Vec::new();
        for (t_lo, t_hi, at_node, whole) in self.runs() {
            if whole {
                for sign in [1.0, -1.0] {
                    loops.push(closed(match at_node {
                        Some(v_node) => std::iter::once(self.node(v_node))
                            .chain(
                                (1..n_v).map(|i| self.branch((i as f64).madd(step, v_node), sign)),
                            )
                            .collect(),
                        None => (0..n_v)
                            .map(|i| self.branch((i as f64 + 0.5).madd(step, t_lo), sign))
                            .collect(),
                    }));
                }
                continue;
            }
            // `u` moves like the square root of the distance in `v` to a turn,
            // so the loop is sampled at `v = t_lo + (t_hi - t_lo) (1 - cos θ) / 2`
            // for even steps of `θ`: `v` then moves like `θ²` at each turn and
            // the points fall at near-even steps along the curve, turns
            // included. A loop that sweeps far round the axis on a short run of
            // `v` (a plane through the centre, tilted a little) takes a step per
            // half scan step of `u` its branches sweep.
            let len = ((t_hi - t_lo) / step).ceil();
            let probes = (len as usize).max(8);
            let (u_lo, u_hi) = (0..=probes)
                .map(|k| (t_hi - t_lo).madd(k as f64 / probes as f64, t_lo))
                .map(|v| self.rhs(v).clamp(-1.0, 1.0).acos())
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), u| {
                    (lo.min(u), hi.max(u))
                });
            let m = len
                .max((n_v as f64) * (u_hi - u_lo) / std::f64::consts::PI)
                .max(fewest)
                .min(most)
                .ceil();
            let steps = m as usize;
            if let Some(v_node) = at_node {
                // Out along one branch from the node to a turn and back along
                // the other, once each side of the node.
                for (from, to) in [(v_node, t_lo), (v_node, t_hi)] {
                    let along = |k: usize| {
                        let f = 0.5 * (1.0 - (std::f64::consts::PI * k as f64 / m).cos());
                        (to - from).madd(f, from)
                    };
                    let mut pts = vec![self.node(v_node)];
                    pts.extend((1..=steps).map(|k| self.branch(along(k), 1.0)));
                    pts.extend((1..steps).rev().map(|k| self.branch(along(k), -1.0)));
                    loops.push(closed(pts));
                }
            } else {
                let at = |k: usize| {
                    let f = 0.5 * (1.0 - (std::f64::consts::PI * k as f64 / m).cos());
                    (t_hi - t_lo).madd(f, t_lo)
                };
                let mut pts: Vec<IntersectionPoint> =
                    (0..=steps).map(|k| self.branch(at(k), 1.0)).collect();
                pts.extend((1..steps).rev().map(|k| self.branch(at(k), -1.0)));
                loops.push(closed(pts));
            }
        }
        loops
    }

    /// The section's point halfway in `v` between two of its points, given
    /// by their `(u, v)` on the torus, on the branch nearer their middle `u`.
    fn between(&self, a: (f64, f64), b: (f64, f64)) -> IntersectionPoint {
        let half = |from: f64, to: f64| {
            let gap = (to - from + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
            0.5_f64.madd(gap, from)
        };
        let (u, v) = (half(a.0, b.0), half(a.1, b.1));
        let off = self.rhs(v).clamp(-1.0, 1.0).acos();
        let miss =
            |w: f64| ((w - u + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI).abs();
        let sign = if miss(self.phi + off) <= miss(self.phi - off) {
            1.0
        } else {
            -1.0
        };
        self.branch(v, sign)
    }
}

/// The two sections of a plane that crosses every tube cross-section of a
/// torus twice (one parallel to the axis within `R − r` of it, or tilted a
/// little from that): both branches `u = phi ± acos(rhs(v))` of
/// [`plane_torus_loops`] are then defined for every `v`, so each closes
/// into a loop that winds once around the tube. Sampled at `n_v` steps from
/// `v = 0`, the outer equator, so every such loop on a torus starts on one
/// latitude, as the tube cross-sections of a plane through the axis do.
/// `None` when a branch lapses somewhere or the two come close to meeting.
#[allow(clippy::cast_precision_loss)]
fn plane_torus_winding_loops(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    n_v: usize,
) -> Option<Vec<Vec<Point3>>> {
    let big_r = torus.major_radius();
    let small_r = torus.minor_radius();
    let a = normal.dot(torus.x_axis());
    let b = normal.dot(torus.y_axis());
    let c = normal.dot(torus.z_axis());
    let s = a.hypot(b);
    if s < 1e-12 * normal.length() || small_r >= big_r {
        return None;
    }
    let phi = b.atan2(a);
    let d_local = d - dot_np(normal, torus.center());
    let rhs = |v: f64| (d_local - small_r * c * v.sin()) / (s * small_r.madd(v.cos(), big_r));
    let dense = 8 * n_v;
    if (0..dense).any(|i| rhs(TAU * i as f64 / dense as f64).abs() > 1.0 - 1e-3) {
        return None;
    }
    let mut loops = [Vec::with_capacity(n_v + 1), Vec::with_capacity(n_v + 1)];
    for i in 0..n_v {
        let v = TAU * i as f64 / n_v as f64;
        let delta = rhs(v).acos();
        loops[0].push(torus.evaluate(phi + delta, v));
        loops[1].push(torus.evaluate(phi - delta, v));
    }
    Some(
        loops
            .into_iter()
            .map(|mut run| {
                run.push(run[0]);
                run
            })
            .collect(),
    )
}

/// Real intersection parameters `t` of the line `origin + t·dir` with a torus.
///
/// A line meets a torus in up to four points (degree-4). Substituting the line
/// into the torus implicit `(a² + b² + c² + R² − r²)² = 4R²(a² + b²)` — where
/// `(a, b, c)` are the line point's coordinates in the torus frame — gives a
/// quartic in `t`, solved here for its real roots, each refined by Newton steps
/// on the tube it lies nearer (on a spindle torus, the near tube or the one
/// swung across the axis). `dir` need not be unit length; `t` is in units of
/// `dir`. Returns the roots sorted ascending (0–4 of them).
///
/// Used by the boolean section trimmer to find where a plane×torus oval exits a
/// box face's straight boundary edge — the exact crossing shared by the two
/// adjacent faces, which is what makes the notch watertight.
#[must_use]
pub fn intersect_line_torus(torus: &ToroidalSurface, origin: Point3, dir: Vec3) -> Vec<f64> {
    let c = torus.center();
    let (xa, ya, za) = (torus.x_axis(), torus.y_axis(), torus.z_axis());
    let big_r = torus.major_radius();
    let small_r = torus.minor_radius();

    // Line point in torus frame: a(t)=a0+a1 t, b(t)=b0+b1 t, c(t)=c0+c1 t.
    let o = Vec3::new(origin.x() - c.x(), origin.y() - c.y(), origin.z() - c.z());
    let (a0, a1) = (xa.dot(o), xa.dot(dir));
    let (b0, b1) = (ya.dot(o), ya.dot(dir));
    let (c0, c1) = (za.dot(o), za.dot(dir));

    // G(t) = a² + b² + c² + R² − r²  (quadratic: g2 t² + g1 t + g0)
    let g2 = a1.madd(a1, b1.madd(b1, c1 * c1));
    let g1 = 2.0 * a1.madd(a0, b1.madd(b0, c1 * c0));
    let g0 = a0.madd(
        a0,
        b0.madd(b0, c0.madd(c0, big_r.madd(big_r, -small_r * small_r))),
    );

    // H(t) = 4R² (a² + b²)  (quadratic: h2 t² + h1 t + h0)
    let four_rr = 4.0 * big_r * big_r;
    let h2 = four_rr * a1.madd(a1, b1 * b1);
    let h1 = four_rr * (2.0 * a1.madd(a0, b1 * b0));
    let h0 = four_rr * a0.madd(a0, b0 * b0);

    // Quartic G² − H = 0:  e4 t⁴ + e3 t³ + e2 t² + e1 t + e0.
    let e4 = g2 * g2;
    let e3 = 2.0 * g2 * g1;
    let e2 = g1.madd(g1, 2.0 * g2 * g0) - h2;
    let e1 = 2.0f64.madd(g1 * g0, -h1);
    let e0 = g0.madd(g0, -h0);

    let mut roots = real_roots_quartic(e4, e3, e2, e1, e0);
    // The quartic also holds a spindle torus's tube swung across the axis
    // (`G = -sqrt(H)`), whose roots pair up with the tube's own a few hundredths
    // apart when `R` is small: the solver leaves them loose, and polishing one
    // against the near tube carries it beside the other root. Each root is
    // polished on the tube it lies nearer.
    let tube = |t: f64, across: bool| -> f64 {
        let p = origin + dir * t;
        let q = Vec3::new(p.x() - c.x(), p.y() - c.y(), p.z() - c.z());
        let (a, b, cc) = (xa.dot(q), ya.dot(q), za.dot(q));
        let centre = if across { -big_r } else { big_r };
        (a.hypot(b) - centre).hypot(cc) - small_r
    };
    for t in &mut roots {
        let across = big_r < small_r && tube(*t, true).abs() < tube(*t, false).abs();
        for _ in 0..8 {
            let eps = 1e-7;
            let f = tube(*t, across);
            let df = (tube(*t + eps, across) - tube(*t - eps, across)) / (2.0 * eps);
            if df.abs() <= 1e-12 {
                break;
            }
            let step = f / df;
            *t -= step;
            if step.abs() < 1e-14 * (1.0 + t.abs()) {
                break;
            }
        }
    }
    roots.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    roots.dedup_by(|a, b| (*a - *b).abs() < 1e-9 * (1.0 + b.abs()));
    roots
}

/// Real roots of `c4 x⁴ + c3 x³ + c2 x² + c1 x + c0` via Durand–Kerner, falling
/// back to the lower-degree solvers when the leading coefficients vanish.
fn real_roots_quartic(c4: f64, c3: f64, c2: f64, c1: f64, c0: f64) -> Vec<f64> {
    // Degenerate leading coefficient → lower degree.
    if c4.abs() < 1e-14 {
        return real_roots_cubic(c3, c2, c1, c0);
    }
    // Monic: x⁴ + a x³ + b x² + c x + d.
    let (a, b, c, d) = (c3 / c4, c2 / c4, c1 / c4, c0 / c4);
    let eval = |z: Complex| -> Complex {
        // Horner.
        let mut acc = Complex::new(1.0, 0.0);
        acc = acc * z + Complex::new(a, 0.0);
        acc = acc * z + Complex::new(b, 0.0);
        acc = acc * z + Complex::new(c, 0.0);
        acc * z + Complex::new(d, 0.0)
    };
    // Durand–Kerner: four roots seeded on a circle, iterated to convergence.
    let seed = Complex::new(0.4, 0.9);
    let mut r = [
        Complex::new(1.0, 0.0),
        seed,
        seed * seed,
        seed * seed * seed,
    ];
    for _ in 0..100 {
        let mut max_step = 0.0_f64;
        for i in 0..4 {
            let mut denom = Complex::new(1.0, 0.0);
            for j in 0..4 {
                if i != j {
                    denom = denom * (r[i] - r[j]);
                }
            }
            if denom.norm() < 1e-300 {
                continue;
            }
            let step = eval(r[i]) / denom;
            r[i] = r[i] - step;
            max_step = max_step.max(step.norm());
        }
        if max_step < 1e-14 {
            break;
        }
    }
    // Keep roots with negligible imaginary part AND a small REAL-polynomial
    // residual — Durand–Kerner stops after a fixed iteration cap whether or not
    // it converged, so a non-converged iterate could otherwise be returned as a
    // spurious root. Evaluate the monic quartic at each candidate (real part) and
    // keep only |p(x)| below a magnitude-scaled tolerance; de-dup near-equal
    // roots (a double root converges to two near-identical iterates).
    let p_real = |x: f64| -> f64 { (((x + a) * x + b) * x + c) * x + d };
    let mut out: Vec<f64> = Vec::new();
    for z in r {
        if z.im.abs() >= 1e-7 {
            continue;
        }
        let x = z.re;
        // Residual tolerance scales with the polynomial's coefficient magnitude
        // and |x|^4 so large-coefficient quartics are not over-rejected.
        let scale = 1.0 + a.abs() + b.abs() + c.abs() + d.abs() + x.abs().powi(4);
        if p_real(x).abs() > 1e-6 * scale {
            continue;
        }
        if out.iter().any(|&y| (y - x).abs() < 1e-9 * (1.0 + x.abs())) {
            continue;
        }
        out.push(x);
    }
    out
}

/// Real roots of `a x³ + b x² + c x + d` (Cardano), with quadratic fallback.
fn real_roots_cubic(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    if a.abs() < 1e-14 {
        return real_roots_quadratic(b, c, d);
    }
    // Depressed cubic t³ + p t + q via x = t − b/(3a).
    let (b, c, d) = (b / a, c / a, d / a);
    let p = c - b * b / 3.0;
    let q = 2.0 * b * b * b / 27.0 - b * c / 3.0 + d;
    let shift = -b / 3.0;
    let disc = q * q / 4.0 + p * p * p / 27.0;
    if disc > 1e-14 {
        let sq = disc.sqrt();
        let u = (-q / 2.0 + sq).cbrt();
        let v = (-q / 2.0 - sq).cbrt();
        vec![u + v + shift]
    } else if disc < -1e-14 {
        // Three real roots (trigonometric).
        let m = 2.0 * (-p / 3.0).sqrt();
        let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
        (0..3)
            .map(|k| {
                m.madd(
                    (theta - 2.0 * std::f64::consts::PI * f64::from(k) / 3.0).cos(),
                    shift,
                )
            })
            .collect()
    } else {
        // Repeated roots.
        let u = (-q / 2.0).cbrt();
        vec![2.0 * u + shift, -u + shift]
    }
}

/// Real roots of `a x² + b x + c`, with linear fallback.
fn real_roots_quadratic(a: f64, b: f64, c: f64) -> Vec<f64> {
    if a.abs() < 1e-14 {
        if b.abs() < 1e-14 {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        Vec::new()
    } else {
        let sq = disc.sqrt();
        vec![(-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a)]
    }
}

/// Minimal complex number for the quartic root finder.
#[derive(Clone, Copy)]
struct Complex {
    re: f64,
    im: f64,
}

impl Complex {
    const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(
            self.re.madd(o.re, -(self.im * o.im)),
            self.re.madd(o.im, self.im * o.re),
        )
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let den = o.re.madd(o.re, o.im * o.im);
        Self::new(
            self.re.madd(o.re, self.im * o.im) / den,
            self.im.madd(o.re, -(self.re * o.im)) / den,
        )
    }
}

/// Build intersection curves from a collection of ordered 3D points.
///
/// If there are enough points, fits a NURBS curve through them.
fn build_curves_from_points(
    points_3d: &[Point3],
    ipoints: Vec<IntersectionPoint>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    if points_3d.len() < 2 {
        return Ok(vec![]);
    }

    let degree = 3.min(points_3d.len() - 1);
    let curve = interpolate(points_3d, degree)?;
    Ok(vec![IntersectionCurve {
        curve,
        points: ipoints,
    }])
}

// -- Analytic-Analytic Intersection -------------------------------------------

/// Intersect two analytic surfaces using a general marching approach.
///
/// Seeds intersection points by sampling both parameter spaces on a grid,
/// then marches along the intersection curve using the cross product of
/// the two surface normals as the tangent direction.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::unnecessary_wraps,
    clippy::type_complexity
)]
pub fn intersect_analytic_analytic(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
) -> Result<Vec<IntersectionCurve>, MathError> {
    intersect_analytic_analytic_bounded(a, b, grid_res, None, None)
}

/// Intersect two analytic surfaces with optional v-range overrides.
///
/// When `v_range_hint_a` or `v_range_hint_b` is `Some((min, max))`, the
/// marching algorithm searches that v-range instead of the hardcoded default.
/// This is essential for cylinders and cones whose default v-range is small
/// (-1..1 or 0.01..2) but whose actual face may extend much further.
///
/// # Errors
///
/// Returns `MathError` if algebraic intersection fails or marching diverges.
pub fn intersect_analytic_analytic_bounded(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
    v_range_hint_a: Option<(f64, f64)>,
    v_range_hint_b: Option<(f64, f64)>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    intersect_analytic_analytic_impl(a, b, grid_res, v_range_hint_a, v_range_hint_b, None)
}

/// [`intersect_analytic_analytic_bounded`] with its general marcher confined
/// to `region`.
///
/// The marcher starts only from seeds that converge into the region (with a
/// margin of about two grid cells) and stops where a curve leaves it. Pairs
/// solved in closed form are returned whole, as by the unconfined call. Two
/// bounded faces meet only inside both their boxes, so their overlap is a
/// sound region for a face pair.
///
/// # Errors
///
/// Returns `MathError` if algebraic intersection fails or marching diverges.
pub fn intersect_analytic_analytic_in_region(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
    v_range_hint_a: Option<(f64, f64)>,
    v_range_hint_b: Option<(f64, f64)>,
    region: Aabb3,
) -> Result<Vec<IntersectionCurve>, MathError> {
    intersect_analytic_analytic_impl(a, b, grid_res, v_range_hint_a, v_range_hint_b, Some(region))
}

fn intersect_analytic_analytic_impl(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
    v_range_hint_a: Option<(f64, f64)>,
    v_range_hint_b: Option<(f64, f64)>,
    region: Option<Aabb3>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    // Try algebraic specialization for known surface pairs before falling
    // back to the general marching approach.
    if let Some(result) = try_algebraic_intersection(&a, &b, v_range_hint_a, v_range_hint_b)? {
        return Ok(result);
    }
    marched_sections(a, b, grid_res, v_range_hint_a, v_range_hint_b, region)
}

/// `x + y t` with one rounding, at the end: Dekker's split product and
/// Knuth's two-sum keep the rounding of `y t` and of the sum, and the last
/// addition rounds their total once.
fn add_product(x: f64, y: f64, t: f64) -> f64 {
    let split = |v: f64| {
        let c = 134_217_729.0 * v;
        let hi = c - (c - v);
        (hi, v - hi)
    };
    let p = y * t;
    let ((yh, yl), (th, tl)) = (split(y), split(t));
    let p_err = yl.madd(tl, yl.madd(th, yh.madd(tl, yh.madd(th, -p))));
    let s = x + p;
    let back = s - x;
    let s_err = (x - (s - back)) + (p - back);
    s + (s_err + p_err)
}

/// An analytic surface moved by a vector, held while the marcher works on it.
enum Moved {
    Cylinder(CylindricalSurface),
    Cone(ConicalSurface),
    Sphere(SphericalSurface),
    Torus(ToroidalSurface),
}

impl Moved {
    /// `surface` moved so that `anchor` lies at the origin, how far its `v`
    /// moved, and a bound on how far the move's rounding shifted it.
    ///
    /// A cylinder's frame, which can sit anywhere along its axis, goes to
    /// the point of the axis nearest `anchor`, so the surface reads alike
    /// wherever its frame sat, and its `v` drops by as much. That point is
    /// the old frame plus a multiple of the axis, each coordinate summed with
    /// one rounding at the end ([`add_product`]), so it lies within a float
    /// step of the axis however far the old frame sat: moved by plain sums,
    /// a frame 1e12 along a tilted axis landed 4.7e-6 off it. Each other
    /// surface's frame point (an apex or a centre) is fixed by its shape and
    /// moves by one subtraction, rounding each coordinate by up to half its
    /// float spacing.
    fn new(surface: &AnalyticSurface<'_>, anchor: Point3) -> (Self, f64, f64) {
        let size = |p: Point3| p.x().abs().max(p.y().abs()).max(p.z().abs());
        let shift = Point3::new(0.0, 0.0, 0.0) - anchor;
        match surface {
            AnalyticSurface::Cylinder(c) => {
                let (o, a) = (c.origin(), c.axis());
                let along = a.dot(anchor - o);
                let foot = Point3::new(
                    add_product(o.x(), a.x(), along),
                    add_product(o.y(), a.y(), along),
                    add_product(o.z(), a.z(), along),
                );
                let near = foot + shift;
                let error = 4.0 * f64::EPSILON * (size(foot) + size(near));
                (Self::Cylinder(c.framed_at(near)), along, error)
            }
            AnalyticSurface::Cone(c) => {
                let near = c.translated(shift);
                let error = 2.0 * f64::EPSILON * size(near.apex());
                (Self::Cone(near), 0.0, error)
            }
            AnalyticSurface::Sphere(s) => {
                let near = s.translated(shift);
                let error = 2.0 * f64::EPSILON * size(near.center());
                (Self::Sphere(near), 0.0, error)
            }
            AnalyticSurface::Torus(t) => {
                let near = t.translated(shift);
                let error = 2.0 * f64::EPSILON * size(near.center());
                (Self::Torus(near), 0.0, error)
            }
        }
    }

    const fn surface(&self) -> AnalyticSurface<'_> {
        match self {
            Self::Cylinder(c) => AnalyticSurface::Cylinder(c),
            Self::Cone(c) => AnalyticSurface::Cone(c),
            Self::Sphere(s) => AnalyticSurface::Sphere(s),
            Self::Torus(t) => AnalyticSurface::Torus(t),
        }
    }
}

/// The sections of `a` and `b` that the general marcher traces from seeds
/// on a grid over `a`, confined to `region` when one is given.
///
/// The marcher works with both surfaces moved by the same vector to the
/// middle of the region, or of `a`'s patch, each cylinder's frame moved
/// along its axis to there, and moves its curves back. Far
/// from the origin the spacing of floats in one coordinate would otherwise
/// decide whether a point lies on both surfaces and whether they cross
/// there, and a pair moved there would not read as it does near the origin:
/// a touch as a crossing, a shallow crossing as a touch. Moving a curve back
/// rounds each coordinate by up to half its spacing there, and moving each
/// surface shifts it by a bound [`Moved::new`] gives, so each fit is held
/// that much within the linear tolerance; where that leaves less than half
/// of it, a traced section is refused before its fit.
fn marched_sections(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
    v_range_hint_a: Option<(f64, f64)>,
    v_range_hint_b: Option<(f64, f64)>,
    region: Option<Aabb3>,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let size = |p: Point3| p.x().abs().max(p.y().abs()).max(p.z().abs());
    let (anchor, reach) = if let Some(r) = region {
        (r.min + (r.max - r.min) * 0.5, (r.max - r.min).length())
    } else {
        let (surf_a, _, (u0, u1), default_v_a) = surface_closures(&a);
        let (v0, v1) = v_range_hint_a.unwrap_or(default_v_a);
        let (first, last) = (surf_a(u0, v0), surf_a(u1, v1));
        let reach = patch_diagonal(surf_a.as_ref(), (u0, u1), (v0, v1));
        (first + (last - first) * 0.5, reach)
    };
    let shift = anchor - Point3::new(0.0, 0.0, 0.0);
    let ((near_a, dv_a, moved_a), (near_b, dv_b, moved_b)) =
        (Moved::new(&a, anchor), Moved::new(&b, anchor));
    let v_range = |surface: &AnalyticSurface<'_>, hint: Option<(f64, f64)>, dv: f64| {
        let (lo, hi) = hint.unwrap_or_else(|| surface_closures(surface).3);
        Some((lo - dv, hi - dv))
    };
    let linear = Tolerance::new().linear;
    let fit_tol =
        ((-f64::EPSILON).madd(size(anchor) + reach, linear) - moved_a - moved_b).min(linear);
    let near_region = region.map(|r| Aabb3 {
        min: r.min - shift,
        max: r.max - shift,
    });
    let curves = marched_sections_near(
        near_a.surface(),
        near_b.surface(),
        grid_res,
        v_range(&a, v_range_hint_a, dv_a),
        v_range(&b, v_range_hint_b, dv_b),
        near_region,
        fit_tol,
    )?;
    curves
        .into_iter()
        .map(|c| {
            let ctrl = c
                .curve
                .control_points()
                .iter()
                .map(|&p| p + shift)
                .collect();
            let curve = NurbsCurve::new(
                c.curve.degree(),
                c.curve.knots().to_vec(),
                ctrl,
                c.curve.weights().to_vec(),
            )?;
            let points = c
                .points
                .into_iter()
                .map(|ip| IntersectionPoint {
                    point: ip.point + shift,
                    ..ip
                })
                .collect();
            Ok(IntersectionCurve { curve, points })
        })
        .collect()
}

/// [`marched_sections`] on surfaces near the origin, each fit held within
/// `fit_tol` of both surfaces. A traced section is refused before its fit
/// where `fit_tol` is less than half the linear tolerance.
#[allow(clippy::too_many_lines, clippy::similar_names)]
fn marched_sections_near(
    a: AnalyticSurface<'_>,
    b: AnalyticSurface<'_>,
    grid_res: usize,
    v_range_hint_a: Option<(f64, f64)>,
    v_range_hint_b: Option<(f64, f64)>,
    region: Option<Aabb3>,
    fit_tol: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let (surf_a, norm_a, u_range_a, default_v_a) = surface_closures(&a);
    let (surf_b, norm_b, u_range_b, default_v_b) = surface_closures(&b);
    let v_range_a = v_range_hint_a.unwrap_or(default_v_a);
    let v_range_b = v_range_hint_b.unwrap_or(default_v_b);

    // Compute characteristic surface dimensions for adaptive parameters.
    let diag_a = patch_diagonal(surf_a.as_ref(), u_range_a, v_range_a);
    let diag_b = patch_diagonal(surf_b.as_ref(), u_range_b, v_range_b);
    let char_size = diag_a.min(diag_b).max(0.1);

    let march_step = (char_size * 0.02).clamp(0.005, 0.5);
    #[allow(clippy::cast_precision_loss)]
    let region = region.map(|r| r.expanded(2.0 * char_size / grid_res as f64));

    // Seeds: the middle of each cell of a grid over a patch whose distance
    // from the other patch is within the cell's reach from its middle, a
    // quarter over its farthest corner, with that distance. Every point of
    // the section lies in a cell of each patch, so each cell it crosses
    // gives a seed. Also the least distance of any cell's middle.
    let cells = |surf: &dyn Fn(f64, f64) -> Point3,
                 (u_range, v_range): ((f64, f64), (f64, f64)),
                 other: &AnalyticSurface<'_>,
                 (other_u, other_v): ((f64, f64), (f64, f64)),
                 other_surf: &dyn Fn(f64, f64) -> Point3| {
        let n = grid_res;
        #[allow(clippy::cast_precision_loss)]
        let at = |i: usize, j: usize, half: f64| {
            let u = (u_range.1 - u_range.0).madd((i as f64 + half) / n as f64, u_range.0);
            let v = (v_range.1 - v_range.0).madd((j as f64 + half) / n as f64, v_range.0);
            surf(u, v)
        };
        let corners: Vec<Point3> = (0..=n)
            .flat_map(|i| (0..=n).map(move |j| (i, j)))
            .map(|(i, j)| at(i, j, 0.0))
            .collect();
        let mut found = Vec::new();
        let mut least = f64::INFINITY;
        for i in 0..n {
            for j in 0..n {
                let middle = at(i, j, 0.5);
                let (u, v) = project_analytic(other, middle, other_u, other_v);
                let foot = other_surf(u, v);
                let dist = (middle - foot).length();
                least = least.min(dist);
                let reach = 1.25
                    * [(0, 0), (1, 0), (0, 1), (1, 1)]
                        .iter()
                        .map(|&(di, dj)| (corners[(i + di) * (n + 1) + j + dj] - middle).length())
                        .fold(0.0, f64::max);
                if dist <= reach && region.is_none_or(|r| r.expanded(reach).contains_point(middle))
                {
                    found.push((dist, middle + (foot - middle) * 0.5));
                }
            }
        }
        (found, least)
    };
    let (mut candidates, min_dist) = cells(
        surf_a.as_ref(),
        (u_range_a, v_range_a),
        &b,
        (u_range_b, v_range_b),
        surf_b.as_ref(),
    );

    // Cheap rejection: the grid samples surface A; the closest sample's
    // distance to B lower-bounds how near the two bounded patches come. A
    // transversal crossing puts a sample within ~one grid cell of it
    // (distance on the order of a cell), so if even the nearest sample is
    // several cells away the patches cannot cross — skip the expensive
    // marching and return empty. Result-preserving: non-crossing pairs
    // already march to nothing, just slowly (this is the gridfinity lip's
    // ~80 inner-wall × outer-wall pairs that dominate pavefiller time).
    #[allow(clippy::cast_precision_loss)]
    let reject_dist = (char_size / grid_res as f64) * 3.0;
    if min_dist > reject_dist {
        return Ok(vec![]);
    }
    // Two loops in one cell of one patch can lie in two cells of the other.
    candidates.extend(
        cells(
            surf_b.as_ref(),
            (u_range_b, v_range_b),
            &a,
            (u_range_a, v_range_a),
            surf_a.as_ref(),
        )
        .0,
    );
    candidates.sort_by(|p, q| p.0.total_cmp(&q.0));

    // A seed is only near the intersection; settle it onto the curve before
    // asking whether it lies in the region, so a coarse grid point beside a
    // short in-region span still starts its march, and march from there, so
    // the march does not begin outside the region and stop at once. The
    // projections from one surface to the other also settle where the two
    // pass closest without meeting, with both normals along the gap: a
    // seed there lies on no curve, and a march from it never closes on it.
    // Nor does a seed where the two surfaces touch without crossing (their
    // normals within about 1e-6 of parallel): the section there can be a
    // lone point, and no direction along it is defined.
    let settle = |mid: Point3| {
        let on = settle_on_section(&a, &b, mid).or_else(|| {
            let mut p = mid;
            for _ in 0..8 {
                let (ua, va) = project_analytic(&a, p, u_range_a, v_range_a);
                let pa = surf_a(ua, va);
                let (ub, vb) = project_analytic(&b, pa, u_range_b, v_range_b);
                let pb = surf_b(ub, vb);
                p = Point3::new(
                    (pa.x() + pb.x()) * 0.5,
                    (pa.y() + pb.y()) * 0.5,
                    (pa.z() + pb.z()) * 0.5,
                );
                if (pa - pb).length() < 1e-9 {
                    break;
                }
            }
            settle_on_section(&a, &b, p)
        });
        // The steps settle on the section of the whole surfaces, which can
        // lie on a part neither patch holds (a cone's other nappe).
        let near_patches = |q: Point3| {
            let (ua, va) = project_analytic(&a, q, u_range_a, v_range_a);
            let (ub, vb) = project_analytic(&b, q, u_range_b, v_range_b);
            (q - surf_a(ua, va)).length() <= march_step
                && (q - surf_b(ub, vb)).length() <= march_step
        };
        on.filter(|&on| {
            crosses_at(&a, &b, on)
                && near_patches(on)
                && region.is_none_or(|r| r.contains_point(on))
        })
        .and_then(|on| station(&a, &b, on, Vec3::new(0.0, 0.0, 0.0)))
    };
    let off_patch = |q: Point3| {
        let (ua, va) = project_analytic(&a, q, u_range_a, v_range_a);
        let (ub, vb) = project_analytic(&b, q, u_range_b, v_range_b);
        (!is_u_periodic(&a) && (ua <= u_range_a.0 || ua >= u_range_a.1))
            || va <= v_range_a.0
            || va >= v_range_a.1
            || (!is_u_periodic(&b) && (ub <= u_range_b.0 || ub >= u_range_b.1))
            || vb <= v_range_b.0
            || vb >= v_range_b.1
    };

    // March from each seed, nearest first, unless it settles on a curve
    // already marched. Seeds on both patches march first: one just past a
    // patch's edge marches along a curve that one inside may already have
    // marched, which its middle then lies on.
    let mut curves = Vec::new();
    let mut marched: Vec<Vec<Point3>> = Vec::new();
    let mut past_edges = Vec::new();
    let mut k = 0;
    while k < candidates.len() + past_edges.len() {
        let seed = if k < candidates.len() {
            let mid = candidates[k].1;
            k += 1;
            let Some(seed) = settle(mid) else {
                continue;
            };
            if off_patch(seed.p) {
                past_edges.push(seed);
                continue;
            }
            seed
        } else {
            k += 1;
            past_edges[k - 1 - candidates.len()]
        };
        if marched.iter().any(|m| on_march(seed.p, seed.gap, m)) {
            continue;
        }

        let march_result = march_analytic_intersection(
            &a,
            &b,
            surf_a.as_ref(),
            norm_a.as_ref(),
            surf_b.as_ref(),
            norm_b.as_ref(),
            seed.p,
            u_range_a,
            v_range_a,
            u_range_b,
            v_range_b,
            march_step,
            is_u_periodic(&a),
            is_u_periodic(&b),
            region,
        )
        .ok_or(MathError::ConvergenceFailure {
            iterations: MARCH_STEPS,
        })?;

        if march_result.len() >= 2 {
            let middle = march_result[march_result.len() / 2];
            let gap = station(&a, &b, middle, Vec3::new(0.0, 0.0, 0.0)).map_or(seed.gap, |s| s.gap);
            if marched.iter().any(|m| on_march(middle, gap, m)) {
                continue;
            }
            marched.push(march_result.clone());

            // Two points far closer than a step only crowd the fit.
            let crowd = 1e-3 * march_step;
            let mut spaced: Vec<Point3> = Vec::with_capacity(march_result.len());
            for (k, &p) in march_result.iter().enumerate() {
                let last = k + 1 == march_result.len();
                match spaced.last() {
                    Some(&q) if (p - q).length() < crowd => {
                        if last && spaced.len() > 1 {
                            spaced.pop();
                            spaced.push(p);
                        }
                    }
                    _ => spaced.push(p),
                }
            }
            if spaced.len() < 2 {
                continue;
            }
            if fit_tol < 0.5 * Tolerance::new().linear {
                return Err(MathError::ConvergenceFailure { iterations: 0 });
            }
            let (curve, points) = fit_section(&a, &b, &spaced, fit_tol)?;
            let ipts: Vec<IntersectionPoint> = points
                .iter()
                .map(|&pt| IntersectionPoint {
                    point: pt,
                    param1: (0.0, 0.0),
                    param2: (0.0, 0.0),
                })
                .collect();
            curves.push(IntersectionCurve {
                curve,
                points: ipts,
            });
        }
    }

    Ok(curves)
}

/// Whether `p`, a point of the section whose next branch lies `gap` away,
/// lies on the curve a march traced through `points`: a chord of the march
/// passes nearer it than a step's sag, and nearer than the next branch. A
/// point on another branch, however near, does not.
fn on_march(p: Point3, gap: f64, points: &[Point3]) -> bool {
    points.windows(2).any(|w| {
        let chord = w[1] - w[0];
        let len2 = chord.dot(chord);
        let s = if len2 > 0.0 {
            ((p - w[0]).dot(chord) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let off = (p - (w[0] + chord * s)).length();
        off <= MARCH_BASIN * gap && off <= MARCH_TURN * len2.sqrt()
    })
}

/// The diagonal of the patch `surf` spans over `u_range` by `v_range`: the
/// chord between its first and last corners, or, for a patch closed both
/// ways (a whole torus, whose last corner is its first), the chord from its
/// first corner to its middle.
fn patch_diagonal(
    surf: &dyn Fn(f64, f64) -> Point3,
    (u0, u1): (f64, f64),
    (v0, v1): (f64, f64),
) -> f64 {
    let corner = (surf(u0, v0) - surf(u1, v1)).length();
    let middle = (surf(u0, v0) - surf(f64::midpoint(u0, u1), f64::midpoint(v0, v1))).length();
    if corner > 1e-9 * middle {
        corner
    } else {
        middle
    }
}

/// Try algebraic (closed-form or semi-algebraic) intersection for known
/// surface pairs before falling back to general marching.
///
/// Returns `Some(curves)` if a specialized method exists, `None` otherwise.
///
/// Currently handles:
/// - **Sphere-sphere**: intersection is a circle (plane through the two centers)
/// - **Coaxial cylinders**: same axis → circle(s) or empty
/// - **Sphere-cylinder**: reduce to quadratic in one parameter
/// - **Cone-cylinder**: parallel axes in the cone's own `v`, other axes
///   along the cylinder's rulings
/// - **Cone-sphere**: off the cone's axis, along the cone's generators
/// - **Torus-cylinder**: along the cylinder's rulings
#[allow(clippy::too_many_lines)]
fn try_algebraic_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
    v_range_a: Option<(f64, f64)>,
    v_range_b: Option<(f64, f64)>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    match (a, b) {
        (AnalyticSurface::Cone(cone), AnalyticSurface::Cylinder(cyl)) => Ok(
            algebraic_parallel_cone_cylinder(cone, cyl, v_range_a, v_range_b)?
                .or_else(|| ruling_cone_cylinder(cone, cyl, true)),
        ),
        (AnalyticSurface::Cylinder(cyl), AnalyticSurface::Cone(cone)) => Ok(
            algebraic_parallel_cone_cylinder(cone, cyl, v_range_b, v_range_a)?
                .or_else(|| ruling_cone_cylinder(cone, cyl, false)),
        ),
        (AnalyticSurface::Sphere(s1), AnalyticSurface::Sphere(s2)) => {
            algebraic_sphere_sphere(s1, s2).map(Some)
        }
        (AnalyticSurface::Cylinder(c1), AnalyticSurface::Cylinder(c2)) => {
            let axis_dot = c1.axis().dot(c2.axis()).abs();
            if axis_dot > 1.0 - 1e-10 {
                // Axes are parallel — check if they're the same line.
                let delta = c2.origin() - c1.origin();
                let delta_vec = Vec3::new(delta.x(), delta.y(), delta.z());
                let along = delta_vec.dot(c1.axis());
                let perp = (delta_vec - c1.axis() * along).length();
                if perp < 1e-8 {
                    // Coaxial: same axis, different radii → no intersection
                    // (unless equal radius → degenerate overlap, skip)
                    if (c1.radius() - c2.radius()).abs() < 1e-8 {
                        return Ok(None); // Overlapping — let marcher handle
                    }
                    return Ok(Some(vec![])); // Coaxial, different radii
                }
                if let Some(rulings) = parallel_cylinders(c1, c2, v_range_a, v_range_b)? {
                    return Ok(Some(rulings));
                }
            }
            // Non-coaxial: algebraic quadratic in v.
            algebraic_cylinder_cylinder(c1, c2)
        }
        // Sphere-cylinder (both orderings).
        (AnalyticSurface::Sphere(s), AnalyticSurface::Cylinder(c)) => {
            algebraic_sphere_cylinder(s, c, true)
        }
        (AnalyticSurface::Cylinder(c), AnalyticSurface::Sphere(s)) => {
            algebraic_sphere_cylinder(s, c, false)
        }
        (AnalyticSurface::Cone(c1), AnalyticSurface::Cone(c2)) => {
            if let Some(rulings) = shared_apex_cones(c1, c2, v_range_a, v_range_b)? {
                return Ok(Some(rulings));
            }
            if let Some(curves) = algebraic_cone_cone(c1, c2)? {
                return Ok(Some(curves));
            }
            Ok(algebraic_parallel_cone_cone(c1, c2, v_range_a, v_range_b))
        }
        (AnalyticSurface::Cone(cone), AnalyticSurface::Sphere(sphere)) => {
            Ok(ruling_cone_sphere(cone, sphere, true))
        }
        (AnalyticSurface::Sphere(sphere), AnalyticSurface::Cone(cone)) => {
            Ok(ruling_cone_sphere(cone, sphere, false))
        }
        (AnalyticSurface::Torus(t), AnalyticSurface::Cylinder(c)) => {
            if let Some(exacts) = exact_cylinder_torus(c, t)? {
                return exact_circle_curves(exacts, |p| (t.project_point(p), c.project_point(p)))
                    .map(Some);
            }
            Ok(parallel_axis_torus_cylinder(t, c, true)
                .or_else(|| ruling_torus_cylinder(t, c, true)))
        }
        (AnalyticSurface::Cylinder(c), AnalyticSurface::Torus(t)) => {
            if let Some(exacts) = exact_cylinder_torus(c, t)? {
                return exact_circle_curves(exacts, |p| (c.project_point(p), t.project_point(p)))
                    .map(Some);
            }
            Ok(parallel_axis_torus_cylinder(t, c, false)
                .or_else(|| ruling_torus_cylinder(t, c, false)))
        }
        (AnalyticSurface::Sphere(s), AnalyticSurface::Torus(t)) => {
            match exact_sphere_torus(s, t)? {
                Some(exacts) => {
                    exact_circle_curves(exacts, |p| (s.project_point(p), t.project_point(p)))
                        .map(Some)
                }
                None => Ok(sphere_torus_curves(s, t, true)),
            }
        }
        (AnalyticSurface::Torus(t), AnalyticSurface::Sphere(s)) => {
            match exact_sphere_torus(s, t)? {
                Some(exacts) => {
                    exact_circle_curves(exacts, |p| (t.project_point(p), s.project_point(p)))
                        .map(Some)
                }
                None => Ok(sphere_torus_curves(s, t, false)),
            }
        }
        (AnalyticSurface::Torus(t1), AnalyticSurface::Torus(t2)) => exact_torus_torus(t1, t2)?
            .map(|exacts| {
                exact_circle_curves(exacts, |p| (t1.project_point(p), t2.project_point(p)))
            })
            .transpose(),
        (AnalyticSurface::Cone(c), AnalyticSurface::Torus(t)) => exact_cone_torus(c, t)?
            .map(|exacts| exact_circle_curves(exacts, |p| (c.project_point(p), t.project_point(p))))
            .transpose(),
        (AnalyticSurface::Torus(t), AnalyticSurface::Cone(c)) => exact_cone_torus(c, t)?
            .map(|exacts| exact_circle_curves(exacts, |p| (t.project_point(p), c.project_point(p))))
            .transpose(),
    }
}

/// A torus and a cylinder whose axes are parallel but distinct (a drill
/// through a ring parallel to its axis), traced along the cylinder's
/// rulings. A ruling stays at one distance `ρ` from the torus axis, so it
/// meets the tube where `(ρ − R)² + z² = r²`: a quadratic in its axial
/// parameter. `None` for any other pair (tilted or coaxial axes) and when
/// the sampling misses a window narrower than itself.
fn parallel_axis_torus_cylinder(
    torus: &ToroidalSurface,
    cyl: &CylindricalSurface,
    torus_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let axis = torus.z_axis();
    let along = cyl.axis().dot(axis);
    if along.abs() < 1.0 - 1e-10 {
        return None;
    }
    let offset = cyl.origin() - torus.center();
    if (offset - axis * offset.dot(axis)).length() < Tolerance::new().linear {
        return None;
    }
    let (major, minor) = (torus.major_radius(), torus.minor_radius());
    let roots = |u: f64| {
        let q = cyl.evaluate(u, 0.0) - torus.center();
        let height = q.dot(axis);
        let rho = (q - axis * height).length();
        let reach = minor * minor - (rho - major) * (rho - major);
        ruling_quadratic(1.0, 2.0 * along.signum() * height, height * height - reach)
    };
    let samples = ruling_samples(cyl, &roots);
    let loops = if samples.iter().all(Option::is_some) {
        closed_ruling_loops(cyl, &roots, &samples)
    } else {
        partial_ruling_loops(cyl, &roots, &samples)
    };
    if loops.is_empty() {
        return None;
    }
    Some(fit_ruling_loops(&loops, |p| {
        in_order(torus.project_point(p), cyl.project_point(p), torus_first)
    }))
}

/// Where two circles in a half-plane through an axis cross, as `(rho, z)`
/// pairs (distance from the axis, height along it): each sweeps a circle
/// about the axis. Two crossings closer than the linear tolerance are the
/// one point where the circles touch, side by side or one inside the
/// other. `None` (defer to the marcher) when the circles coincide or a
/// crossing lands on or past the axis; `Some` of none when they miss.
fn meridian_crossings(
    first: (f64, f64, f64),
    second: (f64, f64, f64),
    scale: f64,
) -> Option<Vec<(f64, f64)>> {
    let ((x1, z1, r1), (x2, z2, r2)) = (first, second);
    let (dx, dz) = (x2 - x1, z2 - z1);
    let dist = dx.hypot(dz);
    let slack = 1e-9 * scale;
    if dist < slack {
        return None;
    }
    let rounding = 8.0 * f64::EPSILON * (x1.abs() + x2.abs() + z1.abs() + z2.abs() + r1 + r2);
    if dist > r1 + r2 + rounding || dist < (r1 - r2).abs() - rounding {
        return Some(Vec::new());
    }
    let along = r2.madd(-r2, r1.madd(r1, dist * dist)) / (2.0 * dist);
    let across = r1.madd(r1, -(along * along)).max(0.0).sqrt();
    let (ux, uz) = (dx / dist, dz / dist);
    let sides: &[f64] = if 2.0 * across <= Tolerance::new().linear {
        &[0.0]
    } else {
        &[1.0, -1.0]
    };
    let mut crossings = Vec::with_capacity(sides.len());
    for side in sides {
        let rho = x1 + along * ux - side * across * uz;
        if rho <= slack {
            return None;
        }
        crossings.push((rho, z1 + along * uz + side * across * ux));
    }
    Some(crossings)
}

/// Exact circles as intersection curves, each sampled at 33 points that
/// carry both surfaces' parameters, and a lone touching point as a curve
/// that stays on it.
fn exact_circle_curves(
    exacts: Vec<ExactIntersectionCurve>,
    params: impl Fn(Point3) -> ((f64, f64), (f64, f64)),
) -> Result<Vec<IntersectionCurve>, MathError> {
    let mut curves = Vec::with_capacity(exacts.len());
    for exact in exacts {
        match exact {
            ExactIntersectionCurve::Circle(circle) => {
                let points = (0..=32)
                    .map(|i| {
                        let point = circle.evaluate(TAU * f64::from(i) / 32.0);
                        let (param1, param2) = params(point);
                        IntersectionPoint {
                            point,
                            param1,
                            param2,
                        }
                    })
                    .collect();
                let r = circle.radius();
                let curve = conic_loop(circle.center(), circle.u_axis(), circle.v_axis(), r, r)?;
                curves.push(IntersectionCurve { curve, points });
            }
            // A lone point is where the surfaces touch.
            ExactIntersectionCurve::Points(points) if points.len() == 1 => {
                let point = points[0];
                let (param1, param2) = params(point);
                curves.push(IntersectionCurve {
                    curve: interpolate(&[point, point], 1)?,
                    points: vec![IntersectionPoint {
                        point,
                        param1,
                        param2,
                    }],
                });
            }
            ExactIntersectionCurve::Ellipse(_) | ExactIntersectionCurve::Points(_) => {}
        }
    }
    Ok(curves)
}

/// Circles about an axis through `base`, at the given `(rho, z)` crossings.
fn circles_about_axis(
    base: Point3,
    axis: Vec3,
    crossings: &[(f64, f64)],
) -> Result<Vec<ExactIntersectionCurve>, MathError> {
    crossings
        .iter()
        .map(|&(rho, z)| {
            Circle3D::new(base + axis * z, axis, rho).map(ExactIntersectionCurve::Circle)
        })
        .collect()
}

/// Exact intersection of two tori sharing an axis: their tube cross-sections
/// in a half-plane through the axis cross in up to two points, and each sweeps
/// a circle about the axis.
///
/// Cross-sections that touch meet at one point, sweeping one circle. `None`
/// (defer to the marcher) unless the axes lie on one line, or when the
/// cross-sections coincide.
///
/// # Errors
///
/// Returns an error if a section circle cannot be built.
pub fn exact_torus_torus(
    first: &ToroidalSurface,
    second: &ToroidalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = first.z_axis();
    let scale = first.major_radius() + second.major_radius();
    let offset = second.center() - first.center();
    // A spindle torus's tube also crosses the far side of the axis.
    if first.minor_radius() >= first.major_radius()
        || second.minor_radius() >= second.major_radius()
        || axis.cross(second.z_axis()).length() > 1e-9
        || offset.cross(axis).length() > 1e-9 * scale
    {
        return Ok(None);
    }
    let Some(crossings) = meridian_crossings(
        (first.major_radius(), 0.0, first.minor_radius()),
        (
            second.major_radius(),
            offset.dot(axis),
            second.minor_radius(),
        ),
        scale,
    ) else {
        return Ok(None);
    };
    circles_about_axis(first.center(), axis, &crossings).map(Some)
}

/// Exact intersection of a torus with a cylinder sharing its axis.
///
/// The wall line and the tube's cross-section in a half-plane through the
/// axis cross in up to two points, each sweeping a circle about the axis. A
/// wall touching the tube's outer equator, ring or spindle, or a ring torus's
/// inner equator meets it along the one circle at the torus's centre.
///
/// `None` (defer to the marcher) unless the axes lie on one line, or when
/// the wall reaches the part of a spindle torus's tube across the axis.
///
/// # Errors
///
/// Returns an error if a section circle cannot be built.
pub fn exact_cylinder_torus(
    cylinder: &CylindricalSurface,
    torus: &ToroidalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = torus.z_axis();
    let scale = torus.major_radius() + cylinder.radius();
    let offset = cylinder.origin() - torus.center();
    let linear = crate::tolerance::Tolerance::default().linear;
    if axis.cross(cylinder.axis()).length() > 1e-9
        || offset.cross(axis).length() > 1e-9 * scale
        || cylinder.radius() + torus.major_radius() <= torus.minor_radius() + linear
    {
        return Ok(None);
    }
    let gap = cylinder.radius() - torus.major_radius();
    let small = torus.minor_radius();
    // The equator circle lies on the torus within the wall's distance from
    // the tube, so it stands for the section while that distance is within
    // the linear tolerance, whatever the scale.
    if (gap.abs() - small).abs() <= linear {
        return circles_about_axis(torus.center(), axis, &[(cylinder.radius(), 0.0)]).map(Some);
    }
    if gap.abs() > small {
        return Ok(Some(Vec::new()));
    }
    let height = small.madd(small, -(gap * gap)).sqrt();
    circles_about_axis(
        torus.center(),
        axis,
        &[(cylinder.radius(), height), (cylinder.radius(), -height)],
    )
    .map(Some)
}

/// Exact intersection of a ring torus with a cone sharing its axis.
///
/// In a half-plane through the axis, a generator of the cone runs from the
/// apex at the half angle `a` from the radial direction, and the tube's
/// cross-section is a circle of the minor radius `r` about `(R, h)`: `R` the
/// major radius, `h` the torus centre's height above the apex along the
/// cone's axis. The generator meets that circle at `v = t ± √(r² − p²)`
/// along it, `t = R cos a + h sin a` the foot of the perpendicular from the
/// centre and `p = |R sin a − h cos a|` its length. Each root ahead of the
/// apex sweeps a circle about the axis. A generator within the linear
/// tolerance of tangent to the cross-section touches the tube along the one
/// circle at its foot.
///
/// `None` (defer to the marcher) unless the axes lie on one line, and for a
/// horn or spindle torus, whose tube reaches the axis.
///
/// # Errors
///
/// Returns an error if a section circle cannot be built.
pub fn exact_cone_torus(
    cone: &ConicalSurface,
    torus: &ToroidalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = cone.axis();
    let (major, minor) = (torus.major_radius(), torus.minor_radius());
    let offset = torus.center() - cone.apex();
    // The cone's axis must pass the torus's centre within the torus's own
    // scale: an apex far away does not widen the band.
    if minor >= major
        || axis.cross(torus.z_axis()).length() > 1e-9
        || offset.cross(axis).length() > 1e-9 * (major + minor)
    {
        return Ok(None);
    }
    let linear = Tolerance::new().linear;
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let height = offset.dot(axis);
    let foot = major.madd(cos_a, height * sin_a);
    let reach = major.madd(sin_a, -(height * cos_a)).abs();
    let roots = if (reach - minor).abs() <= linear {
        vec![foot]
    } else if reach > minor {
        Vec::new()
    } else {
        let half = ((minor - reach) * (minor + reach)).sqrt();
        vec![foot - half, foot + half]
    };
    let crossings: Vec<(f64, f64)> = roots
        .into_iter()
        .filter(|&v| v * cos_a > linear)
        .map(|v| (v * cos_a, v * sin_a))
        .collect();
    circles_about_axis(cone.apex(), axis, &crossings).map(Some)
}

/// Exact intersection of a torus with a sphere centred on its axis.
///
/// The sphere's great circle and the tube's cross-section in a half-plane
/// through the axis cross in up to two points, and each sweeps a circle
/// about the axis; where they touch (a ball sitting in the ring's hole or
/// on its rim) they meet along one circle. A sphere off the axis that
/// touches a ring torus at one point gives that point.
///
/// `None` (defer to the marcher) for a spindle torus, and for a sphere off
/// the axis that does not touch the torus at one point.
///
/// # Errors
///
/// Returns an error if a section circle cannot be built.
pub fn exact_sphere_torus(
    sphere: &SphericalSurface,
    torus: &ToroidalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = torus.z_axis();
    let scale = torus.major_radius() + sphere.radius();
    let offset = sphere.center() - torus.center();
    // A spindle torus's tube also crosses the far side of the axis.
    if torus.minor_radius() >= torus.major_radius() {
        return Ok(None);
    }
    if offset.cross(axis).length() > 1e-9 * scale {
        // A sphere centred on the core circle with the tube's own radius
        // holds the meridian circle through its centre, and the tube round
        // it: they meet along that circle alone. The circle strays from the
        // tube by the centre's distance from the core circle and the radii's
        // difference, which must both be rounding and within tolerance.
        let radial = offset - axis * offset.dot(axis);
        let slack = 64.0
            * f64::EPSILON
            * (torus.major_radius() + torus.minor_radius() + sphere.radius() + offset.length());
        let stray = (radial.length() - torus.major_radius()).hypot(offset.dot(axis))
            + (sphere.radius() - torus.minor_radius()).abs();
        if stray <= slack.min(Tolerance::new().linear) {
            let along = axis.cross(radial).normalize()?;
            return Ok(Some(vec![ExactIntersectionCurve::Circle(Circle3D::new(
                sphere.center(),
                along,
                sphere.radius(),
            )?)]));
        }
        return Ok(
            off_axis_touch(sphere, torus).map(|p| vec![ExactIntersectionCurve::Points(vec![p])])
        );
    }
    let Some(crossings) = meridian_crossings(
        (0.0, offset.dot(axis), sphere.radius()),
        (torus.major_radius(), 0.0, torus.minor_radius()),
        scale,
    ) else {
        return Ok(None);
    };
    circles_about_axis(torus.center(), axis, &crossings).map(Some)
}

/// The fewest and most steps along each branch of a loop a sphere cuts from
/// a torus: its curve is fitted through them alone, so a small loop takes
/// enough to hold the fit to the section.
const SPHERE_TORUS_LOOP_SAMPLES: (f64, f64) = (128.0, 1024.0);

/// The section a sphere off a ring torus's axis cuts from it. In the
/// torus's frame let the sphere's centre sit at `(a, b, h)`, with
/// `s = hypot(a, b)` and `phi = atan2(b, a)`. The torus's point at `(u, v)`,
/// with `ρ = R + r·cos v`, lies on the sphere of radius `q` when
/// `ρ² − 2ρs·cos(u − phi) + s² + (r·sin v − h)² = q²`, so at each `v`
/// `cos(u − phi) = (R² + r² + h² + s² − q² + 2Rr·cos v − 2hr·sin v) / (2s·ρ)`.
/// `None` for a spindle torus, whose `ρ` reaches zero, or a sphere on the
/// axis, whose sections are circles about it.
fn sphere_torus_section<'a>(
    sphere: &SphericalSurface,
    torus: &'a ToroidalSurface,
) -> Option<TorusSection<'a, impl Fn(Point3) -> f64>> {
    let (big, small, radius) = (torus.major_radius(), torus.minor_radius(), sphere.radius());
    let offset = sphere.center() - torus.center();
    let (a, b, h) = (
        offset.dot(torus.x_axis()),
        offset.dot(torus.y_axis()),
        offset.dot(torus.z_axis()),
    );
    let s = a.hypot(b);
    if small >= big || s <= 1e-9 * (big + radius) {
        return None;
    }
    let centre = sphere.center();
    Some(TorusSection {
        torus,
        phi: b.atan2(a),
        num: [
            big.madd(
                big,
                small.madd(small, h.madd(h, s.madd(s, -radius * radius))),
            ),
            2.0 * big * small,
            -2.0 * h * small,
        ],
        den: [2.0 * s * big, 2.0 * s * small],
        clearance: move |p: Point3| (p - centre).length() - radius,
        slack: 64.0 * f64::EPSILON * (big + small + radius + offset.length()),
    })
}

/// How many times a loop's fitted curve may be refined where it strays.
const SPHERE_TORUS_FIT_ROUNDS: usize = 6;

/// The curves a sphere off a ring torus's axis meets it in, each fitted
/// through the closed-form points of [`sphere_torus_section`] and every point
/// given its parameters on both surfaces, the sphere's first when
/// `sphere_first`. Halfway between each pair of points the curve must lie
/// within a quarter of the linear tolerance of both surfaces; where it
/// strays the section's own point there joins the fit. `None`, deferring to the marcher, for a
/// curve that still strays, or whose points do.
fn sphere_torus_curves(
    sphere: &SphericalSurface,
    torus: &ToroidalSurface,
    sphere_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let section = sphere_torus_section(sphere, torus)?;
    let tol = Tolerance::new().linear;
    let (big, small) = (torus.major_radius(), torus.minor_radius());
    // The centre's least and greatest distance to the torus: the ball meets
    // it whenever its radius lies between them. A section that comes out
    // empty for such a ball has lost its loops to rounding, so the marcher
    // takes it rather than the pair being called apart.
    let offset = sphere.center() - torus.center();
    let rise = offset.dot(torus.z_axis());
    let reach = (offset - torus.z_axis() * rise).length();
    let (nearest, farthest) = (
        ((reach - big).hypot(rise) - small).abs(),
        (reach + big).hypot(rise) + small,
    );
    let margin = 8.0 * f64::EPSILON * (big + small + offset.length() + sphere.radius());
    let crosses = sphere.radius() > nearest + margin && sphere.radius() < farthest - margin;
    let off = |p: Point3| {
        let local = p - torus.center();
        let (x, y, z) = (
            local.dot(torus.x_axis()),
            local.dot(torus.y_axis()),
            local.dot(torus.z_axis()),
        );
        let on_torus = (x.hypot(y) - big).hypot(z) - small;
        on_torus.abs().max((section.clearance)(p).abs())
    };
    // `acos` of a ratio near 1 loses digits a small ball on a large torus
    // needs, so each point steps onto the sphere across the section, along
    // the torus, the shortest way in its own measure.
    let centre = sphere.center();
    let polish = |q: IntersectionPoint| {
        let (mut u, mut v) = q.param1;
        let mut best = q;
        for _ in 0..4 {
            let gap = (section.clearance)(best.point);
            let (sin_u, cos_u) = u.sin_cos();
            let (sin_v, cos_v) = v.sin_cos();
            let rho = small.madd(cos_v, big);
            let outward = torus.x_axis() * cos_u + torus.y_axis() * sin_u;
            let p_u = (torus.y_axis() * cos_u - torus.x_axis() * sin_u) * rho;
            let p_v = outward * (-small * sin_v) + torus.z_axis() * (small * cos_v);
            let away = best.point - centre;
            let normal = away * (1.0 / away.length());
            let (g_u, g_v) = (normal.dot(p_u), normal.dot(p_v));
            let (w_u, w_v) = (g_u / (rho * rho), g_v / (small * small));
            let reach = g_u.madd(w_u, g_v * w_v);
            if reach <= f64::MIN_POSITIVE {
                break;
            }
            let next = section.point(gap.madd(-w_u / reach, u), gap.madd(-w_v / reach, v));
            if (section.clearance)(next.point).abs() >= gap.abs() {
                break;
            }
            (u, v) = next.param1;
            best = next;
        }
        best
    };
    let loops = section.loops(256, SPHERE_TORUS_LOOP_SAMPLES);
    if loops.is_empty() && crosses {
        return None;
    }
    let mut curves = Vec::new();
    for points in loops {
        let mut points: Vec<IntersectionPoint> = points.into_iter().map(polish).collect();
        if points.iter().any(|q| off(q.point) > tol) {
            return None;
        }
        // Samples past a turn that `acos` clamped onto the same `u` all step
        // to the turn itself.
        points.dedup_by(|b, a| (b.point - a.point).length() <= 1e-3 * tol);
        let mut rounds = 0;
        let curve = loop {
            let positions: Vec<Point3> = points.iter().map(|q| q.point).collect();
            let curve = interpolate(&positions, 3.min(positions.len() - 1)).ok()?;
            let params = crate::nurbs::fitting::chord_length_params(&positions);
            let strays: Vec<usize> = (0..positions.len() - 1)
                .filter(|&k| {
                    off(curve.evaluate(f64::midpoint(params[k], params[k + 1]))) > 0.25 * tol
                })
                .collect();
            if strays.is_empty() {
                break curve;
            }
            if rounds == SPHERE_TORUS_FIT_ROUNDS {
                return None;
            }
            rounds += 1;
            for &k in strays.iter().rev() {
                let mid = polish(section.between(points[k].param1, points[k + 1].param1));
                if off(mid.point) > tol {
                    return None;
                }
                points.insert(k + 1, mid);
            }
        };
        for q in &mut points {
            let (on_sphere, on_torus) = (sphere.project_point(q.point), q.param1);
            (q.param1, q.param2) = if sphere_first {
                (on_sphere, on_torus)
            } else {
                (on_torus, on_sphere)
            };
        }
        curves.push(IntersectionCurve { curve, points });
    }
    Some(curves)
}

/// Where a sphere off a ring torus's axis touches it at one point: from
/// outside the tube, from inside it, or holding the whole ring. The touch
/// sits over the core circle's point nearest the sphere's centre, at
/// `d_near`, or farthest from it, at `d_far`, and the sphere's radius is
/// then `d_near − r`, `r − d_near` or `d_far + r`. Each is the centre's
/// least or greatest distance to the torus, so the point is all the two
/// share. `None` unless the radius matches one to rounding.
fn off_axis_touch(sphere: &SphericalSurface, torus: &ToroidalSurface) -> Option<Point3> {
    let axis = torus.z_axis();
    let (big, small, radius) = (torus.major_radius(), torus.minor_radius(), sphere.radius());
    let offset = sphere.center() - torus.center();
    let radial = offset - axis * offset.dot(axis);
    let out = radial * (1.0 / radial.length());
    let near = torus.center() + out * big;
    let far = torus.center() - out * big;
    let (to_near, to_far) = (sphere.center() - near, sphere.center() - far);
    let (d_near, d_far) = (to_near.length(), to_far.length());
    let slack = 64.0 * f64::EPSILON * (big + small + radius + offset.length());
    // A ball centred on the core circle meets the tube along a whole
    // meridian, not at a point.
    if d_near > slack
        && ((d_near - small - radius).abs() <= slack || (small - d_near - radius).abs() <= slack)
    {
        return Some(near + to_near * (small / d_near));
    }
    ((d_far + small - radius).abs() <= slack).then(|| far - to_far * (small / d_far))
}

/// Exact coaxial cone-cone intersection: returns the shared circle.
///
/// Two cones that share an axis are concentric circles at every axial
/// station, so they meet only where their radii are equal. Each cone's
/// radius is linear in the axial coordinate `t` (measured along the shared
/// axis from cone 1's apex): `r1 = m1·t` and `r2 = m2·σ·(t − d2)`, where
/// `m_i = cot(half_angle_i)`, `σ = sign(axis2·axis1)`, and `d2` is cone 2's
/// apex position in that coordinate. Equating gives a single crossing `t*`
/// → one circle (the shared rim). The general marcher mishandles this case:
/// at the radii-crossing the surfaces are nearly tangent, so a grid-seeded
/// march fragments the clean circle into dozens of degenerate micro-curves.
///
/// Returns `Some(vec![circle])` for a genuine crossing, `Some(vec![])` when
/// the cones do not meet (parallel radius lines or a crossing on the wrong
/// nappe), and `None` for the identical-cone overlap or a degenerate
/// (near-flat) cone — both of which fall through to the general path.
/// Parallel-but-offset axes with equal half-angle tangents reduce to a
/// radical-plane conic (`offset_parallel_cone_cone`); other offset
/// configurations defer to the marcher with `None`.
///
/// # Errors
///
/// Returns [`MathError`] if the shared-rim `Circle3D` cannot be constructed
/// (e.g. a non-finite center or radius from a malformed cone).
pub fn exact_cone_cone(
    c1: &ConicalSurface,
    c2: &ConicalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = c1.axis();
    let axis2 = c2.axis();

    // Coaxial check: parallel axes and the second apex lies on the first axis.
    if axis.dot(axis2).abs() < 1.0 - 1e-10 {
        return Ok(None); // Non-coaxial: quartic curve, let the marcher handle.
    }
    let apex1 = c1.apex();
    let apex2 = c2.apex();
    let delta = apex2 - apex1;
    let delta_v = Vec3::new(delta.x(), delta.y(), delta.z());
    let along = delta_v.dot(axis);
    if (delta_v - axis * along).length() > 1e-8 {
        return offset_parallel_cone_cone(c1, c2);
    }

    let (s1, s2) = (c1.half_angle().sin(), c2.half_angle().sin());
    if s1.abs() < 1e-12 || s2.abs() < 1e-12 {
        return Ok(None); // Degenerate (near-flat) cone.
    }
    let m1 = c1.half_angle().cos() / s1;
    let m2 = c2.half_angle().cos() / s2;
    let sigma = if axis.dot(axis2) >= 0.0 { 1.0 } else { -1.0 };
    let d2 = along; // apex2 position along `axis`, measured from apex1.

    let denom = m1 - m2 * sigma;
    if denom.abs() < 1e-12 {
        // Parallel radius lines: identical cones (coincident apex, same opening)
        // overlap — defer to the general/same-domain path; otherwise no meeting.
        if sigma > 0.0 && d2.abs() < 1e-9 {
            return Ok(None);
        }
        return Ok(Some(vec![]));
    }

    let t_star = (-m2 * sigma * d2) / denom;
    let radius = m1 * t_star;
    if radius < 1e-12 {
        return Ok(Some(vec![])); // Crossing on the wrong nappe / no real circle.
    }

    let center = Point3::new(
        apex1.x() + axis.x() * t_star,
        apex1.y() + axis.y() * t_star,
        apex1.z() + axis.z() * t_star,
    );
    let circle = Circle3D::new(center, axis, radius)?;
    Ok(Some(vec![ExactIntersectionCurve::Circle(circle)]))
}

/// Parallel-axis (or anti-parallel), offset-apex cones with equal half-angle
/// tangents: subtracting the two quadric equations cancels both the radial
/// and the axial quadratic terms (their coefficients depend only on
/// `tan²(half_angle)`), so every intersection point lies on a plane — the
/// degenerate member of the quadric pencil — and plane ∩ cone is an exact
/// conic. The gridfinity spacer lip fuse hits this exactly: opposed 45°
/// corner cones offset 0.25mm, which the marcher shreds into ~64 closed
/// micro-loops per pair (#1570). Unequal angles keep a genuine quadratic
/// term, and an unbounded section (hyperbola/parabola) has no closed-form
/// win over the marcher — both defer with `None`.
fn offset_parallel_cone_cone(
    c1: &ConicalSurface,
    c2: &ConicalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    if c1.half_angle().sin().abs() < 1e-12 || c2.half_angle().sin().abs() < 1e-12 {
        return Ok(None); // Degenerate (near-flat) cone, as in the coaxial path.
    }
    let t1 = c1.half_angle().tan();
    let t2 = c2.half_angle().tan();
    if !t1.is_finite() || !t2.is_finite() {
        return Ok(None);
    }
    if (t1 - t2).abs() > 1e-9 * (1.0 + t1.abs().max(t2.abs())) {
        return Ok(None);
    }

    let w = c1.axis();
    let apex1 = c1.apex();
    let apex2 = c2.apex();
    let delta = apex2 - apex1;
    let delta_v = Vec3::new(delta.x(), delta.y(), delta.z());
    let s = delta_v.dot(w);
    let tm = 0.5 * (t1 + t2);
    let k = 1.0 + tm * tm;

    // In the apex1 frame each cone is |P|² − k(P·w)² = 0 (shifted by δ for
    // cone 2; the axis SIGN drops out since only (P·w)² appears). Their
    // difference: P·(2δ − 2ksw) = |δ|² − ks².
    let n = (delta_v - w * (k * s)) * 2.0;
    let n_len = n.length();
    if n_len < 1e-12 {
        return Ok(None);
    }
    let n_hat = n * (1.0 / n_len);
    let d = (dot_np(n, apex1) + delta_v.dot(delta_v) - k * s * s) / n_len;

    // `exact_plane_cone` already rejects sections on cone 1's phantom nappe;
    // cone 2's nappe must be checked here. A conic on the shared quadric
    // pencil cannot cross between nappes except exactly through apex 2, so
    // sampled quarter-points either all pass or all fail; a mixed verdict
    // means an apex-touching degeneracy — defer to the marcher.
    let axis2 = c2.axis();
    let scale = 1.0 + delta_v.length();
    let mut out = Vec::new();
    for curve in exact_plane_cone(c1, n_hat, d, 0.0)? {
        let samples: Vec<Point3> = match &curve {
            ExactIntersectionCurve::Circle(c) => (0..4)
                .map(|i| crate::traits::ParametricCurve::evaluate(c, TAU * f64::from(i) / 4.0))
                .collect(),
            ExactIntersectionCurve::Ellipse(e) => (0..4)
                .map(|i| crate::traits::ParametricCurve::evaluate(e, TAU * f64::from(i) / 4.0))
                .collect(),
            ExactIntersectionCurve::Points(_) => return Ok(None),
        };
        let on_real_nappe = |p: &Point3| {
            let rel = *p - apex2;
            Vec3::new(rel.x(), rel.y(), rel.z()).dot(axis2) >= -1e-9 * scale
        };
        let hits = samples.iter().filter(|p| on_real_nappe(p)).count();
        match hits {
            0 => {}
            4 => out.push(curve),
            _ => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Exact coaxial cone-cylinder intersection: returns the shared circle.
///
/// A cone and a cylinder sharing an axis are concentric circles at every
/// axial station, so they meet only where the cone's radius equals the
/// cylinder's. The cone radius is linear in the axial coordinate `t` from its
/// apex (`r = m·t`, `m = cot(half_angle)`), the cylinder radius is the
/// constant `R`, so `m·t = R` gives a single crossing `t*` → one circle. This
/// is the gridfinity lip's top knife edge (inner tapered corner = cone, outer
/// corner = cylinder, concentric, radii matching at `Z_PEAK`); the general
/// marcher fragments that near-tangent contact into dozens of degenerate
/// micro-curves.
///
/// Returns `Some(vec![circle])` for a genuine crossing, `Some(vec![])` when
/// the crossing degenerates to the apex, and `None` (defer to the marcher)
/// when the surfaces are not coaxial or the cone is near-flat / near-axial.
///
/// # Errors
///
/// Returns [`MathError`] if the shared `Circle3D` cannot be constructed.
pub fn exact_cone_cylinder(
    cone: &ConicalSurface,
    cyl: &CylindricalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let axis = cone.axis();
    let cyl_axis = cyl.axis();

    // Coaxial check: parallel axes and the cone apex on the cylinder's axis.
    if axis.dot(cyl_axis).abs() < 1.0 - 1e-10 {
        return Ok(None);
    }
    let apex = cone.apex();
    let delta = apex - cyl.origin();
    let delta_v = Vec3::new(delta.x(), delta.y(), delta.z());
    let along = delta_v.dot(cyl_axis);
    if (delta_v - cyl_axis * along).length() > 1e-8 {
        return Ok(None);
    }

    let s = cone.half_angle().sin();
    if s.abs() < 1e-12 {
        return Ok(None); // near-flat cone.
    }
    let m = cone.half_angle().cos() / s; // dr/dt along the cone axis.
    if m.abs() < 1e-12 {
        return Ok(None); // near-axial cone: radius ~constant.
    }

    let t_star = cyl.radius() / m; // where the cone radius m·t equals R.
    if t_star.abs() < 1e-12 {
        return Ok(Some(vec![])); // crossing at the apex — no real circle.
    }
    let center = Point3::new(
        apex.x() + axis.x() * t_star,
        apex.y() + axis.y() * t_star,
        apex.z() + axis.z() * t_star,
    );
    let circle = Circle3D::new(center, axis, cyl.radius())?;
    Ok(Some(vec![ExactIntersectionCurve::Circle(circle)]))
}

/// The exact rational quadratic NURBS of the full ellipse `centre +
/// a·cos(t)·u + b·sin(t)·v` (a circle when `a == b`), starting at
/// `centre + a·u`: four quarter arcs, each with its corner weighted by
/// cos 45°.
fn conic_loop(centre: Point3, u: Vec3, v: Vec3, a: f64, b: f64) -> Result<NurbsCurve, MathError> {
    let w = std::f64::consts::FRAC_1_SQRT_2;
    let corners: [(f64, f64); 9] = [
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
    let points = corners
        .iter()
        .map(|&(x, y)| centre + u * (a * x) + v * (b * y))
        .collect();
    let weights = vec![1.0, w, 1.0, w, 1.0, w, 1.0, w, 1.0];
    let knots = vec![
        0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
    ];
    NurbsCurve::new(2, knots, points, weights)
}

/// Two cones sharing an apex meet only along rulings through it. A
/// ruling's direction `d` makes each cone's angle with its axis, `d·aᵢ =
/// ±sin(half_angleᵢ)` (the half-angle runs from the radial plane), a line
/// that crosses the unit sphere twice, touches it where the cones are
/// tangent along one ruling, or misses it. Along `apex + t·d` the first
/// cone's `v` is `t` and the second's `±t`, so each ruling runs where both
/// faces' `v` ranges hold it, as a straight segment cut at the apex. Two
/// rulings closer than the linear tolerance across the faces are one.
///
/// `None` unless both ranges are given, the axes are not parallel (the
/// coaxial path's case), and the apexes coincide to rounding.
fn shared_apex_cones(
    c1: &ConicalSurface,
    c2: &ConicalSurface,
    v1: Option<(f64, f64)>,
    v2: Option<(f64, f64)>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let (Some(v1), Some(v2)) = (v1, v2) else {
        return Ok(None);
    };
    let (a1, a2, apex) = (c1.axis(), c2.axis(), c1.apex());
    let across = a1.cross(a2);
    let sin = across.length();
    if sin < 1e-9 {
        return Ok(None);
    }
    let reach = v1.0.abs().max(v1.1.abs()).max(v2.0.abs()).max(v2.1.abs());
    let scale = reach.max((apex - Point3::new(0.0, 0.0, 0.0)).length());
    if (c2.apex() - apex).length() > 64.0 * f64::EPSILON * scale {
        return Ok(None);
    }
    let normal = across * (1.0 / sin);
    let tol = Tolerance::new().linear;
    // A ruling's part across the first axis is `(t2 − s1·cos) / sin`, whose
    // rounding grows as the axes close up; past a tenth of the tolerance
    // over the faces, the pair is left to the marcher.
    if f64::EPSILON * reach > 0.1 * tol * sin {
        return Ok(None);
    }
    // The second axis's part across the first, a unit vector.
    let toward = normal.cross(a1);
    let cos = a1.dot(a2);
    let (s1, s2) = (c1.half_angle().sin(), c2.half_angle().sin());
    // `evaluate` turns its radial direction round for negative `v`.
    let params = |cone: &ConicalSurface, p: Point3| {
        let (u, v) = cone.project_point(p);
        if v < 0.0 {
            ((u + std::f64::consts::PI).rem_euclid(TAU), v)
        } else {
            (u, v)
        }
    };
    let mut curves = Vec::new();
    for nappe in [1.0, -1.0] {
        let t2 = nappe * s2;
        let side = cos.madd(-s1, t2) / sin;
        let foot = a1 * s1 + toward * side;
        let lift_sq = s1.madd(-s1, side.madd(-side, 1.0));
        let rounding = 4.0 * f64::EPSILON;
        let lifts = if lift_sq.abs() <= rounding {
            // Within rounding of touching: the rulings that rounding could
            // hide part by up to `√(2·rounding)` radians, taken as one while
            // that stays within ten tolerances over the faces, and beyond
            // it not settled here.
            if (2.0 * rounding).sqrt() * reach > 10.0 * tol {
                return Ok(None);
            }
            vec![0.0]
        } else if lift_sq < 0.0 {
            Vec::new()
        } else if 2.0 * lift_sq.sqrt() * reach <= tol {
            vec![0.0]
        } else {
            vec![lift_sq.sqrt(), -lift_sq.sqrt()]
        };
        let along2 = if nappe > 0.0 { v2 } else { (-v2.1, -v2.0) };
        let (lo, hi) = (v1.0.max(along2.0), v1.1.min(along2.1));
        if hi - lo <= tol {
            continue;
        }
        for lift in lifts {
            let d = (foot + normal * lift).normalize()?;
            let pieces = if lo < 0.0 && hi > 0.0 {
                vec![(lo, 0.0), (0.0, hi)]
            } else {
                vec![(lo, hi)]
            };
            for (from, to) in pieces {
                curves.push(ruling_segment(apex + d * from, apex + d * to, |p| {
                    (params(c1, p), params(c2, p))
                })?);
            }
        }
    }
    Ok(Some(curves))
}

/// Two cylinders of one radius whose axes cross meet in two ellipses, one
/// in each plane through the crossing that bisects the axes: a point lies
/// as far from both axes where its offsets along them are equal or
/// opposite. Both ellipses have their minor axis, of the radius, along the
/// axes' common normal, and meet at its two ends. They come back as two
/// closed loops, as the first cylinder's rulings pair them: the halves of
/// both ellipses on one side of the crossing along its axis, then on the
/// other, each exact, rational, and cornered where its halves meet. One
/// loop starts at each end of the common normal.
///
/// `None` unless the radii match and the axes cross to rounding: axes that
/// miss by `δ` part the curves at the crossings by about `√(δ·r)`.
fn crossing_equal_cylinders(
    c1: &CylindricalSurface,
    c2: &CylindricalSurface,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let (d1, d2, r) = (c1.axis(), c2.axis(), c1.radius());
    if (c2.radius() - r).abs() > 8.0 * f64::EPSILON * r {
        return Ok(None);
    }
    // An ellipse's semi-major axis is `2r / |d1 ∓ d2|`. Rounding the axes
    // turns its major axis by about `ε / |d1 ∓ d2|`, which moves its far
    // vertices by `2rε / |d1 ∓ d2|²`; axes nearer parallel than keeps that
    // well inside the tolerance are left to the rulings.
    let bisectors = [d1 - d2, d1 + d2];
    let tol = Tolerance::new().linear;
    if bisectors
        .iter()
        .any(|b| 2.0 * r * f64::EPSILON > 1e-3 * tol * b.dot(*b))
    {
        return Ok(None);
    }
    let across = d1.cross(d2);
    let sin = across.length();
    let normal = across * (1.0 / sin);
    let w = c1.origin() - c2.origin();
    let cos = d1.dot(d2);
    let along = cos.madd(d2.dot(w), -d1.dot(w)) / (sin * sin);
    let centre = c1.origin() + d1 * along;
    // The miss is read at the crossing's own scale: origins far along
    // their axes leave it more rounding than that, and are left to the
    // rulings too.
    let scale = r.max((centre - Point3::new(0.0, 0.0, 0.0)).length());
    if w.dot(normal).abs() > 64.0 * f64::EPSILON * scale {
        return Ok(None);
    }
    // Each ellipse's major axis, turned to lean along the first axis, and
    // its semi-major axis.
    let majors = bisectors.map(|bisector| {
        let length = bisector.length();
        let major = (bisector * (1.0 / length)).cross(normal);
        let major = if major.dot(d1) < 0.0 { -major } else { major };
        (major, 2.0 * r / length)
    });
    let corner = std::f64::consts::FRAC_1_SQRT_2;
    let half = [(1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)];
    let mut curves = Vec::with_capacity(2);
    for side in [1.0, -1.0] {
        let [(m1, a1), (m2, a2)] = majors;
        let out = half
            .iter()
            .map(|&(x, y)| centre + normal * (side * r * x) + m1 * (side * a1 * y));
        let back = half[1..]
            .iter()
            .map(|&(x, y)| centre - normal * (side * r * x) + m2 * (side * a2 * y));
        let curve = NurbsCurve::new(
            2,
            vec![
                0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0,
            ],
            out.chain(back).collect(),
            vec![1.0, corner, 1.0, corner, 1.0, corner, 1.0, corner, 1.0],
        )?;
        #[allow(clippy::cast_precision_loss)]
        let points = (0..=32)
            .map(|i| {
                let point = curve.evaluate(i as f64 / 32.0);
                IntersectionPoint {
                    point,
                    param1: c1.project_point(point),
                    param2: c2.project_point(point),
                }
            })
            .collect();
        curves.push(IntersectionCurve { curve, points });
    }
    Ok(Some(curves))
}

/// Algebraic cone-cone intersection (NURBS form for the general bounded
/// path). Delegates to [`exact_cone_cone`] and returns each exact conic
/// (coaxial circle or offset-parallel radical-plane ellipse) as an exact
/// rational NURBS `IntersectionCurve` with sample points along it, mirroring
/// the sphere-cylinder algebraic path. phase FF prefers the exact circle form
/// directly (so the section edge links to the coincident boundary), but a
/// caller of `intersect_analytic_analytic_bounded` still gets one clean
/// curve instead of the marcher's fragments.
fn algebraic_cone_cone(
    c1: &ConicalSurface,
    c2: &ConicalSurface,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let Some(exacts) = exact_cone_cone(c1, c2)? else {
        return Ok(None);
    };
    let mut curves = Vec::new();
    for exact in exacts {
        let n_samples = 33;
        let mut points = Vec::with_capacity(n_samples);
        #[allow(clippy::cast_precision_loss)]
        for i in 0..n_samples {
            let theta = TAU * i as f64 / (n_samples - 1) as f64;
            let pt = match &exact {
                ExactIntersectionCurve::Circle(circle) => {
                    crate::traits::ParametricCurve::evaluate(circle, theta)
                }
                ExactIntersectionCurve::Ellipse(ellipse) => {
                    crate::traits::ParametricCurve::evaluate(ellipse, theta)
                }
                ExactIntersectionCurve::Points(_) => break,
            };
            points.push(IntersectionPoint {
                point: pt,
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            });
        }
        let curve = match &exact {
            ExactIntersectionCurve::Circle(c) => {
                conic_loop(c.center(), c.u_axis(), c.v_axis(), c.radius(), c.radius())?
            }
            ExactIntersectionCurve::Ellipse(e) => conic_loop(
                e.center(),
                e.u_axis(),
                e.v_axis(),
                e.semi_major(),
                e.semi_minor(),
            )?,
            ExactIntersectionCurve::Points(_) => continue,
        };
        curves.push(IntersectionCurve { curve, points });
    }
    Ok(Some(curves))
}

/// Exact coaxial sphere-cylinder intersection: returns the shared circle(s).
///
/// A sphere of radius `R` centered at `C` and a cylinder of radius `r` whose
/// axis passes through `C` meet in concentric circles of radius `r` at the
/// axial stations where `sqrt(R² − z²) = r`, i.e. `z = ±sqrt(R² − r²)`
/// measured from `C` along the axis. A proper crossing yields two circles; a
/// tangent contact (`r = R`) yields one; a cylinder wider than the sphere, or
/// a non-coaxial configuration (quartic curve), yields none/defers.
///
/// Mirrors [`exact_cone_cylinder`] so phase FF can emit the section as an
/// exact `Circle3D` (which the closed-circle split + seam adoption recognise)
/// rather than the marcher's NURBS fragments.
///
/// Returns `Some(vec![..])` (0, 1, or 2 circles) for the coaxial case, and
/// `None` (defer to the general marcher) when the axes are not coaxial.
///
/// # Errors
///
/// Returns [`MathError`] if a shared `Circle3D` cannot be constructed.
pub fn exact_sphere_cylinder(
    sphere: &SphericalSurface,
    cyl: &CylindricalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let sc = sphere.center();
    let r_sphere = sphere.radius();
    let co = cyl.origin();
    let axis = cyl.axis();
    let r_cyl = cyl.radius();

    // Project sphere center onto the cylinder axis.
    let delta = sc - co;
    let delta_vec = Vec3::new(delta.x(), delta.y(), delta.z());
    let along = delta_vec.dot(axis);
    let perp_vec = delta_vec - axis * along;
    let d_perp = perp_vec.length();

    // Non-coaxial sphere-cylinder intersections produce quartic curves;
    // defer those to the general marcher.
    if d_perp > 1e-7 {
        return Ok(None);
    }

    // Coaxial: the sphere center lies on the cylinder axis. No real circle
    // when the cylinder is wider than the sphere or they are tangent-internal.
    if r_cyl > r_sphere + 1e-10 {
        return Ok(Some(vec![]));
    }
    let z_sq = r_sphere * r_sphere - r_cyl * r_cyl;
    if z_sq < 0.0 {
        return Ok(Some(vec![]));
    }
    let z = z_sq.sqrt();

    // The sphere center projected onto the axis is the midpoint of the two
    // section circles, each offset by ±z along the axis with radius `r_cyl`.
    let center_axis_pt = Point3::new(
        co.x() + axis.x() * along,
        co.y() + axis.y() * along,
        co.z() + axis.z() * along,
    );

    let mut circles = Vec::new();
    let offsets: &[f64] = if z < 1e-10 { &[0.0] } else { &[z, -z] };
    for &z_offset in offsets {
        let center = Point3::new(
            center_axis_pt.x() + axis.x() * z_offset,
            center_axis_pt.y() + axis.y() * z_offset,
            center_axis_pt.z() + axis.z() * z_offset,
        );
        let circle = Circle3D::new(center, axis, r_cyl)?;
        circles.push(ExactIntersectionCurve::Circle(circle));
    }
    Ok(Some(circles))
}

/// Exact coaxial cone-sphere intersection: the shared circles.
///
/// When the cone's axis runs through the sphere's centre `C`, every generator
/// `apex + v g` (`g` a unit direction, `v` the distance from the apex) has
/// the same `h = g·(apex − C)`, so all of them meet the sphere at the same
/// roots of `v² + 2hv + |apex − C|² − R² = 0`. Each root ahead of the apex is
/// a circle of radius `v cos a` at `v sin a` along the axis: two for a ball
/// the cone passes through, one for a ball that swallows the apex, passes
/// through it or touches the wall, none for a ball the cone misses or that
/// sits inside it clear of the wall.
///
/// Returns `Some(vec![..])` for a coaxial pair and `None` (the ruling trace
/// or the marcher) otherwise.
///
/// # Errors
///
/// Returns [`MathError`] if a shared `Circle3D` cannot be constructed.
pub fn exact_cone_sphere(
    cone: &ConicalSurface,
    sphere: &SphericalSurface,
) -> Result<Option<Vec<ExactIntersectionCurve>>, MathError> {
    let offset = cone.apex() - sphere.center();
    let along = offset.dot(cone.axis());
    if (offset - cone.axis() * along).length() > 1e-7 {
        return Ok(None);
    }
    let lin_tol = Tolerance::new().linear;
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let (far_sq, radius_sq) = (offset.dot(offset), sphere.radius() * sphere.radius());
    let b = 2.0 * sin_a * along;
    let (disc, far, near) = ruling_quadratic(1.0, b, far_sq - radius_sq);
    // At a tangent ball `b² − 4c` cancels to a few ulps of its terms, which
    // can land below zero: that is a touch, not a miss.
    let noise = 16.0 * f64::EPSILON * 4.0f64.madd(far_sq + radius_sq, b * b);
    if disc < -noise {
        return Ok(Some(vec![]));
    }
    let roots: &[f64] = if far - near < lin_tol {
        &[far]
    } else {
        &[near, far]
    };
    let mut circles = Vec::new();
    for &v in roots {
        if v * cos_a > lin_tol {
            let centre = cone.apex() + cone.axis() * (v * sin_a);
            let circle = Circle3D::new(centre, cone.axis(), v * cos_a)?;
            circles.push(ExactIntersectionCurve::Circle(circle));
        }
    }
    Ok(Some(circles))
}

/// Algebraic sphere-cylinder intersection (NURBS form for the general bounded
/// path). A coaxial pair delegates to [`exact_sphere_cylinder`] and returns
/// each exact circle as an exact rational NURBS `IntersectionCurve`. phase FF
/// prefers the exact circle form directly (so the section edge links to the
/// coincident boundary and the closed-circle splitter can carve the spherical
/// band), but a caller of `intersect_analytic_analytic_bounded` still gets
/// clean curves instead of the marcher's fragments. Any other pair is traced
/// along the cylinder's rulings ([`off_axis_sphere_cylinder`]).
fn algebraic_sphere_cylinder(
    sphere: &SphericalSurface,
    cyl: &CylindricalSurface,
    sphere_first: bool,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let Some(exacts) = exact_sphere_cylinder(sphere, cyl)? else {
        return Ok(off_axis_sphere_cylinder(sphere, cyl, sphere_first));
    };

    let mut curves = Vec::new();
    for exact in exacts {
        let ExactIntersectionCurve::Circle(circle) = exact else {
            continue;
        };
        let n_samples = 33;
        let mut points = Vec::with_capacity(n_samples);
        #[allow(clippy::cast_precision_loss)]
        for i in 0..n_samples {
            let theta = TAU * i as f64 / (n_samples - 1) as f64;
            let pt = crate::traits::ParametricCurve::evaluate(&circle, theta);
            let (param1, param2) = in_order(
                sphere.project_point(pt),
                cyl.project_point(pt),
                sphere_first,
            );
            points.push(IntersectionPoint {
                point: pt,
                param1,
                param2,
            });
        }
        let r = circle.radius();
        let curve = conic_loop(circle.center(), circle.u_axis(), circle.v_axis(), r, r)?;
        curves.push(IntersectionCurve { curve, points });
    }

    Ok(Some(curves))
}

/// A sphere and a cylinder whose axis misses the sphere's centre (a drill
/// entering a ball off its axis), traced along the cylinder's rulings: the
/// ruling `c(u) + v·a` meets the sphere where `v² + 2(q·a)·v + |q|² − R² = 0`,
/// with `q = c(u) − C`. When every ruling meets the sphere (the cylinder
/// passes wholly through it) the roots trace an entry and an exit loop;
/// otherwise each window of meeting rulings carries one loop. `Some(empty)`
/// when the two cannot meet, `None` when the sampling misses a window
/// narrower than itself.
fn off_axis_sphere_cylinder(
    sphere: &SphericalSurface,
    cyl: &CylindricalSurface,
    sphere_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let (centre, radius) = (sphere.center(), sphere.radius());
    let axis = cyl.axis();
    let offset = centre - cyl.origin();
    let axis_distance = (offset - axis * offset.dot(axis)).length();
    let lin_tol = Tolerance::new().linear;
    if axis_distance > radius + cyl.radius() + lin_tol
        || axis_distance + radius < cyl.radius() - lin_tol
    {
        return Some(Vec::new());
    }
    let roots = |u: f64| {
        let q = cyl.evaluate(u, 0.0) - centre;
        ruling_quadratic(1.0, 2.0 * q.dot(axis), q.dot(q) - radius * radius)
    };
    let samples = ruling_samples(cyl, &roots);
    let loops = if samples.iter().all(Option::is_some) {
        closed_ruling_loops(cyl, &roots, &samples)
    } else {
        partial_ruling_loops(cyl, &roots, &samples)
    };
    if loops.is_empty() {
        return None;
    }
    Some(fit_ruling_loops(&loops, |p| {
        in_order(sphere.project_point(p), cyl.project_point(p), sphere_first)
    }))
}

/// Parameters on the pair's first and second surfaces, from those on `a`
/// and `b` and whether `a` came first.
const fn in_order(a: (f64, f64), b: (f64, f64), a_first: bool) -> ((f64, f64), (f64, f64)) {
    if a_first { (a, b) } else { (b, a) }
}

/// Two cylinders whose axes are parallel but apart meet along rulings:
/// every plane across the axes cuts them in two circles `d` apart, which
/// cross in two points, touch in one, or miss, alike in every plane. Each
/// crossing sweeps a ruling, cut to the stretch of axis both faces' `v`
/// ranges share. Two crossings closer than the linear tolerance are the one
/// where the circles touch, side by side or one inside the other. `None`
/// (the next solver's case) when neither range is known, since a ruling has
/// no ends of its own, or when a ruling's end strays off the second
/// cylinder, its axis leaning off the first's by more than the span allows.
fn parallel_cylinders(
    c1: &CylindricalSurface,
    c2: &CylindricalSurface,
    v_range_1: Option<(f64, f64)>,
    v_range_2: Option<(f64, f64)>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let axis = c1.axis();
    let delta = c2.origin() - c1.origin();
    let along = delta.dot(axis);
    let across = delta - axis * along;
    let d = across.length();
    let (r1, r2) = (c1.radius(), c2.radius());
    // The second face's range, measured along the first one's axis.
    let flip = c2.axis().dot(axis);
    let ranges = [
        v_range_1,
        v_range_2.map(|(a, b)| (along + a * flip, along + b * flip)),
    ];
    let (lo, hi) = ranges
        .into_iter()
        .flatten()
        .fold((f64::NEG_INFINITY, f64::INFINITY), |(lo, hi), (a, b)| {
            (lo.max(a.min(b)), hi.min(a.max(b)))
        });
    if !lo.is_finite() || !hi.is_finite() {
        return Ok(None);
    }
    let rounding = 8.0 * f64::EPSILON * (r1 + r2 + d);
    if d > r1 + r2 + rounding || d < (r1 - r2).abs() - rounding || hi <= lo {
        return Ok(Some(Vec::new()));
    }
    let toward = across * (1.0 / d);
    let side = axis.cross(toward);
    let x = r2.madd(-r2, d.madd(d, r1 * r1)) / (2.0 * d);
    let y = r1.madd(r1, -(x * x)).max(0.0).sqrt();
    let linear = Tolerance::new().linear;
    let feet = if 2.0 * y <= linear {
        vec![toward * x]
    } else {
        vec![toward * x + side * y, toward * x - side * y]
    };
    let on_second = |p: Point3| {
        let w = p - c2.origin();
        ((w - c2.axis() * w.dot(c2.axis())).length() - r2).abs() <= linear
    };
    let mut rulings = Vec::with_capacity(feet.len());
    for foot in feet {
        let base = c1.origin() + foot;
        let (from, to) = (base + axis * lo, base + axis * hi);
        if !on_second(from) || !on_second(to) {
            return Ok(None);
        }
        rulings.push(ruling_segment(from, to, |p| {
            (c1.project_point(p), c2.project_point(p))
        })?);
    }
    Ok(Some(rulings))
}

/// A straight intersection curve from `from` to `to`, carrying its ends and
/// middle as points.
fn ruling_segment(
    from: Point3,
    to: Point3,
    params: impl Fn(Point3) -> ((f64, f64), (f64, f64)),
) -> Result<IntersectionCurve, MathError> {
    let curve = NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![from, to], vec![1.0, 1.0])?;
    let points = [from, from + (to - from) * 0.5, to]
        .into_iter()
        .map(|point| {
            let (param1, param2) = params(point);
            IntersectionPoint {
                point,
                param1,
                param2,
            }
        })
        .collect();
    Ok(IntersectionCurve { curve, points })
}

/// Algebraic cylinder-cylinder intersection for non-coaxial cylinders.
///
/// For two cylinders with axes that are NOT parallel, the intersection
/// consists of up to two closed space curves. These are found by
/// parameterizing one cylinder's angular coordinate `u ∈ [0, 2π]` and
/// solving a quadratic in the axial parameter `v` to find where each
/// "ring" of cylinder A sits on cylinder B.
///
/// The quadratic is:
///   `v²·(1 - α²) + 2v·(q·a₁ - α·q·a₂) + (|q|² - (q·a₂)² - r₂²) = 0`
/// where `α = a₁·a₂`, `q(u)` is the radial point on cylinder 1 minus
/// cylinder 2's origin, `a₁`/`a₂` are the cylinder axes, and `r₂` is
/// cylinder 2's radius.
#[allow(clippy::too_many_lines, clippy::unnecessary_wraps)]
fn algebraic_cylinder_cylinder(
    c1: &CylindricalSurface,
    c2: &CylindricalSurface,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    if let Some(halves) = crossing_equal_cylinders(c1, c2)? {
        return Ok(Some(halves));
    }
    let alpha = c1.axis().dot(c2.axis());
    let a_coeff = 1.0 - alpha * alpha;

    // Should only be called for non-parallel axes.
    if a_coeff.abs() < 1e-12 {
        return Ok(None);
    }

    let r1 = c1.radius();
    let r2 = c2.radius();
    let o1 = c1.origin();
    let o2 = c2.origin();
    let a1 = c1.axis();
    let a2 = c2.axis();

    // Separation check: distance between axes vs sum of radii.
    // Closest approach of two skew lines:
    let delta = Vec3::new(o1.x() - o2.x(), o1.y() - o2.y(), o1.z() - o2.z());
    let cross = a1.cross(a2);
    let cross_len = cross.length();
    if cross_len > 1e-12 {
        let axis_dist = delta.dot(cross).abs() / cross_len;
        if axis_dist > r1 + r2 + Tolerance::new().linear {
            return Ok(Some(vec![])); // No intersection
        }
    }

    // Solve every ruling of one cylinder against the other. When EVERY
    // ruling of the swept cylinder meets the other (the thinner of two
    // crossing tubes), the two roots trace the curve's two closed loops.
    // Swept the other way only a window of rulings meets, and each root
    // traces an open arc of one loop.
    let roots = |sweep: &CylindricalSurface, other: &CylindricalSurface| {
        let (o, a, radius) = (other.origin(), other.axis(), other.radius());
        let alpha = sweep.axis().dot(a);
        let quad = 1.0 - alpha * alpha;
        let (axis, sweep) = (sweep.axis(), sweep.clone());
        move |u: f64| {
            let q = sweep.evaluate(u, 0.0) - o;
            let (q_a1, q_a2) = (q.dot(axis), q.dot(a));
            let b = 2.0 * (q_a1 - alpha * q_a2);
            let c = q.dot(q) - q_a2 * q_a2 - radius * radius;
            ruling_quadratic(quad, b, c)
        }
    };
    let (roots1, roots2) = (roots(c1, c2), roots(c2, c1));
    let samples1 = ruling_samples(c1, &roots1);
    let loops = if samples1.iter().all(Option::is_some) {
        closed_ruling_loops(c1, &roots1, &samples1)
    } else {
        let samples2 = ruling_samples(c2, &roots2);
        if samples2.iter().all(Option::is_some) {
            closed_ruling_loops(c2, &roots2, &samples2)
        } else if samples1.iter().any(Option::is_some) {
            partial_ruling_loops(c1, &roots1, &samples1)
        } else {
            partial_ruling_loops(c2, &roots2, &samples2)
        }
    };
    if loops.is_empty() {
        return Ok(None);
    }
    Ok(Some(fit_ruling_loops(&loops, |p| {
        (c1.project_point(p), c2.project_point(p))
    })))
}

/// A cone and a cylinder whose axes are not parallel, traced along the
/// cylinder's rulings. A ruling `q + t w` meets the cone's double quadric
/// `|p - apex|^2 = h^2 / sin^2(half_angle)`, with `h` the offset along the
/// cone's axis, where a quadratic in `t` vanishes. `None` (the marcher's
/// case) when a ruling meets the far nappe, where no cone face lies, when
/// the rulings run along the cone's generators, or when no ruling meets it.
fn ruling_cone_cylinder(
    cone: &ConicalSurface,
    cyl: &CylindricalSurface,
    cone_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let (sin_t, cos_t) = cone.half_angle().sin_cos();
    if sin_t < 1e-12 || cos_t < 1e-12 {
        return None;
    }
    let (apex, d, w) = (cone.apex(), cone.axis(), cyl.axis());
    let s = 1.0 / (sin_t * sin_t);
    let alpha = w.dot(d);
    let quad = 1.0 - s * alpha * alpha;
    if quad.abs() < 1e-9 {
        return None;
    }
    let roots = |u: f64| {
        let delta = cyl.evaluate(u, 0.0) - apex;
        let (dd, dw) = (delta.dot(d), delta.dot(w));
        let b = 2.0 * (dw - s * dd * alpha);
        let c = delta.dot(delta) - s * dd * dd;
        ruling_quadratic(quad, b, c)
    };
    let lin_tol = Tolerance::new().linear;
    let far_nappe = (0..WINDOW_SCAN * RULING_SAMPLES).any(|k| {
        #[allow(clippy::cast_precision_loss)]
        let u = TAU * (k as f64 + 0.5) / (WINDOW_SCAN * RULING_SAMPLES) as f64;
        let (disc, vp, vm) = roots(u);
        disc >= -lin_tol
            && [vp, vm]
                .iter()
                .any(|&t| (cyl.evaluate(u, t) - apex).dot(d) < -lin_tol)
    });
    if far_nappe {
        return None;
    }
    let samples = ruling_samples(cyl, &roots);
    // The samples that truly meet the cone. One that the tolerance alone
    // admits lies past its window's end: the end bisected from it, and a
    // loop traced from it, lie off the cone.
    let meeting: Vec<Option<(Point3, Point3)>> = samples
        .iter()
        .enumerate()
        .map(|(i, &pair)| pair.filter(|_| roots(ruling_u(i)).0 >= 0.0))
        .collect();
    // A window of meeting rulings that no sample truly meets would vanish
    // (a cone's tip just through the wall) while the others still made
    // loops; a finer scan finds every window, and one with no such sample
    // goes to the marcher. A window with one is sampled afresh between its
    // ends, however few samples it holds.
    let scan = WINDOW_SCAN * RULING_SAMPLES;
    #[allow(clippy::cast_precision_loss)]
    let scan_u = |k: usize| TAU * (k as f64 + 0.5) / scan as f64;
    let discs: Vec<f64> = (0..scan).map(|k| roots(scan_u(k)).0).collect();
    // A window narrower than the scan's step lies between two scan points
    // that miss, where the discriminant peaks: each such peak is found, and
    // one that reaches the tolerance goes to the marcher.
    for k in 0..scan {
        let (before, here, after) = (
            discs[(k + scan - 1) % scan],
            discs[k],
            discs[(k + 1) % scan],
        );
        if here >= -lin_tol || here < before || here < after {
            continue;
        }
        let (mut lo, mut hi) = (scan_u(k) - TAU / scan as f64, scan_u(k) + TAU / scan as f64);
        for _ in 0..100 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if roots(a).0 < roots(b).0 {
                lo = a;
            } else {
                hi = b;
            }
        }
        if roots(f64::midpoint(lo, hi)).0 >= -lin_tol {
            return None;
        }
    }
    // Ruling sample `i` lies midway between scan points
    // `WINDOW_SCAN i + 7` and `WINDOW_SCAN i + 8`.
    let meets = |k: usize| discs[k % scan] >= -lin_tol;
    if let Some(start) = (0..scan).find(|&k| !meets(k)) {
        let mut k = start;
        while k < start + scan {
            if !meets(k) {
                k += 1;
                continue;
            }
            let first = k;
            while k < start + scan && meets(k) {
                k += 1;
            }
            let sampled = (first..k).any(|j| {
                j % WINDOW_SCAN == WINDOW_SCAN / 2 - 1
                    && meeting[(j / WINDOW_SCAN) % RULING_SAMPLES].is_some()
            });
            if !sampled {
                return None;
            }
        }
    }
    let loops = if samples.iter().all(Option::is_some) {
        closed_ruling_loops(cyl, &roots, &samples)
    } else {
        partial_ruling_loops(cyl, &roots, &meeting)
    };
    if loops.is_empty() {
        return None;
    }
    Some(fit_ruling_loops(&loops, |p| {
        in_order(cone.project_point(p), cyl.project_point(p), cone_first)
    }))
}

/// A torus and a cylinder whose axes are not parallel, where every ruling
/// of the cylinder meets the torus the same nonzero even number of times (a
/// rod through the ring's tube): a ruling meets the torus where a quartic in
/// its parameter vanishes, and each of its roots, taken in order, sweeps one
/// closed loop around the cylinder. `None` (the marcher's case) when the
/// count varies between rulings, where the curve turns back between them,
/// checked on a scan finer than the sampling so a narrow window of missing
/// rulings is not stepped over, and for a spindle torus, whose quartic also
/// holds its inner lemon.
fn ruling_torus_cylinder(
    torus: &ToroidalSurface,
    cyl: &CylindricalSurface,
    torus_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    if cyl.axis().dot(torus.z_axis()).abs() > 1.0 - 1e-9
        || torus.minor_radius() >= torus.major_radius()
    {
        return None;
    }
    let roots = |u: f64| intersect_line_torus(torus, cyl.evaluate(u, 0.0), cyl.axis());
    let rows: Vec<Vec<f64>> = (0..RULING_SAMPLES).map(|i| roots(ruling_u(i))).collect();
    let count = rows[0].len();
    let scan = WINDOW_SCAN * RULING_SAMPLES;
    #[allow(clippy::cast_precision_loss)]
    if count == 0
        || count % 2 == 1
        || (0..scan).any(|k| roots(TAU * (k as f64 + 0.5) / scan as f64).len() != count)
    {
        return None;
    }
    let loops: Vec<Vec<Point3>> = (0..count)
        .map(|j| {
            let mut pts: Vec<Point3> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| cyl.evaluate(ruling_u(i), r[j]))
                .collect();
            pts.push(pts[0]);
            pts
        })
        .collect();
    Some(fit_ruling_loops(&loops, |p| {
        in_order(torus.project_point(p), cyl.project_point(p), torus_first)
    }))
}

/// A cone and a sphere whose centre is off the cone's axis, where every
/// generator of the cone crosses the sphere twice on the cone's nappe (a pin
/// through a ball): along a generator `apex + v g` the sphere is a quadratic
/// in `v`, and each of its two roots, taken in order, sweeps one closed loop
/// around the cone. A ball holding the apex is left once by every generator,
/// one loop. `None` when the centre lies on the axis (phase FF takes
/// [`exact_cone_sphere`]'s circles there), or the apex on the sphere. A
/// sphere only some generators cross goes to [`window_cone_sphere`].
fn ruling_cone_sphere(
    cone: &ConicalSurface,
    sphere: &SphericalSurface,
    cone_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let (apex, centre, radius) = (cone.apex(), sphere.center(), sphere.radius());
    let offset = apex - centre;
    let lin_tol = Tolerance::new().linear;
    let along = offset.dot(cone.axis());
    let across = (offset - cone.axis() * along).length();
    if across < lin_tol {
        return None;
    }
    // With the apex inside the ball, `v² + 2hv + K` has roots of opposite
    // signs (`K = |offset|² − R² < 0`): every generator leaves the ball once
    // ahead of the apex, at `v = √(h² − K) − h`, one loop round the cone.
    // Where `h > 0` the root is taken as `−K / (h + √(h² − K))`, free of the
    // cancellation.
    let k = offset.dot(offset) - radius * radius;
    if radius - offset.length() > lin_tol {
        let exit = |u: f64| {
            let h = (cone.evaluate(u, 1.0) - apex).dot(offset);
            let root = h.madd(h, -k).sqrt();
            cone.evaluate(u, if h > 0.0 { -k / (h + root) } else { root - h })
        };
        // A ball that barely holds the apex turns the loop sharply where it
        // passes the apex, at the scale of `√−K`: a step whose midpoint
        // strays from its chord by a hundredth of the chord (a turn of about
        // 0.08 radians) is halved until the turn is resolved.
        let mut samples: Vec<(f64, Point3)> = (0..=RULING_SAMPLES)
            .map(|i| (ruling_u(i), exit(ruling_u(i))))
            .collect();
        for _ in 0..10 {
            let mut refined = Vec::with_capacity(2 * samples.len());
            for pair in samples.windows(2) {
                let ((u0, p0), (u1, p1)) = (pair[0], pair[1]);
                refined.push(pair[0]);
                let um = 0.5 * (u0 + u1);
                let pm = exit(um);
                let chord = (p1 - p0).length();
                if chord > lin_tol && (pm - (p0 + (p1 - p0) * 0.5)).length() > 0.01 * chord {
                    refined.push((um, pm));
                }
            }
            refined.extend(samples.last().copied());
            if refined.len() == samples.len() {
                break;
            }
            samples = refined;
        }
        let mut pts: Vec<Point3> = samples.iter().map(|&(_, p)| p).collect();
        if let Some(last) = pts.last_mut() {
            *last = samples[0].1;
        }
        return Some(fit_ruling_loops(&[pts], |p| {
            in_order(cone.project_point(p), sphere.project_point(p), cone_first)
        }));
    }
    // Both roots along a generator of unit direction `g` (`v` the distance
    // from the apex), nearer first, from `h = g·offset`, when it crosses the
    // sphere twice ahead of the apex.
    let crossing = |h: f64| {
        let (disc, vp, vm) = ruling_quadratic(1.0, 2.0 * h, k);
        (disc > lin_tol && vm >= lin_tol).then_some((vm, vp))
    };
    // Around the cone `h` runs between these bounds. Where both roots lie
    // ahead of the apex, raising `h` shrinks the discriminant and moves the
    // nearer root out, so every generator crosses when the two extreme ones
    // do.
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    if crossing(sin_a.madd(along, cos_a * across)).is_none()
        || crossing(sin_a.madd(along, -cos_a * across)).is_none()
    {
        return window_cone_sphere(cone, sphere, cone_first);
    }
    let rows: Vec<(f64, f64)> = (0..RULING_SAMPLES)
        .map(|i| crossing((cone.evaluate(ruling_u(i), 1.0) - apex).dot(offset)))
        .collect::<Option<_>>()?;
    let loops: Vec<Vec<Point3>> = [0, 1]
        .iter()
        .map(|&j| {
            let mut pts: Vec<Point3> = rows
                .iter()
                .enumerate()
                .map(|(i, &(near, far))| {
                    cone.evaluate(ruling_u(i), if j == 0 { near } else { far })
                })
                .collect();
            pts.push(pts[0]);
            pts
        })
        .collect();
    Some(fit_ruling_loops(&loops, |p| {
        in_order(cone.project_point(p), sphere.project_point(p), cone_first)
    }))
}

/// A cone and a sphere off its axis that only some generators cross (a ball
/// against the cone's side), with the apex outside the sphere: along the
/// generator at `u`, `h(u) = g·(apex − C)` is `c + A cos(u − φ)`, and the
/// generator crosses twice ahead of the apex where `h ≤ −√K`, with
/// `K = |apex − C|² − R²`. That is one arc of `u`, whose ends are where the
/// generator touches the sphere; the section is one loop, the nearer roots
/// out along the arc and the farther ones back. Sampled at
/// `u = mid − half·cos θ`, the roots' split `±√(h² − K)` changes sign with
/// `sin θ` and the loop stays smooth through both touching generators.
/// `Some(empty)` when no generator crosses ahead of the apex. `None`, the
/// marcher's case, when the apex lies within the sphere, when the centre is
/// too near the axis to place the arc, and when every generator reaches the
/// sphere (the extremes' test then failed on a touching generator or a root
/// at the apex).
fn window_cone_sphere(
    cone: &ConicalSurface,
    sphere: &SphericalSurface,
    cone_first: bool,
) -> Option<Vec<IntersectionCurve>> {
    let offset = cone.apex() - sphere.center();
    let lin_tol = Tolerance::new().linear;
    if offset.length() - sphere.radius() <= lin_tol {
        return None;
    }
    let k = offset.dot(offset) - sphere.radius() * sphere.radius();
    let (sin_a, cos_a) = cone.half_angle().sin_cos();
    let (ox, oy) = (offset.dot(cone.x_axis()), offset.dot(cone.y_axis()));
    let (c, a) = (sin_a * offset.dot(cone.axis()), cos_a * ox.hypot(oy));
    if a < lin_tol {
        return None;
    }
    let reach = (-k.sqrt() - c) / a;
    if reach <= -1.0 {
        return Some(Vec::new());
    }
    if reach >= 1.0 {
        return None;
    }
    let (mid, half) = (oy.atan2(ox) + std::f64::consts::PI, reach.acos());
    let half = std::f64::consts::PI - half;
    let n = RULING_SAMPLES;
    let mut pts: Vec<Point3> = (0..n)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let theta = TAU * i as f64 / n as f64;
            let u = half.madd(-theta.cos(), mid);
            let h = a.madd((u - mid + std::f64::consts::PI).cos(), c);
            let split = h.madd(h, -k).max(0.0).sqrt();
            cone.evaluate(u, -h - split.copysign(theta.sin()))
        })
        .collect();
    pts.push(pts[0]);
    Some(fit_ruling_loops(&[pts], |p| {
        in_order(cone.project_point(p), sphere.project_point(p), cone_first)
    }))
}

/// Scan points per ruling sample when looking for windows of meeting
/// rulings.
const WINDOW_SCAN: usize = 16;

/// Rulings sampled around a swept cylinder, half a step off u = 0 so the
/// branches of a self-touching curve (equal crossing cylinders) do not share
/// a sample.
const RULING_SAMPLES: usize = 128;

#[allow(clippy::cast_precision_loss)]
fn ruling_u(i: usize) -> f64 {
    TAU * (i as f64 + 0.5) / RULING_SAMPLES as f64
}

/// The discriminant and roots of `quad·v² + b·v + c = 0`.
fn ruling_quadratic(quad: f64, b: f64, c: f64) -> (f64, f64, f64) {
    let disc = b * b - 4.0 * quad * c;
    let root = disc.max(0.0).sqrt();
    (disc, (-b + root) / (2.0 * quad), (-b - root) / (2.0 * quad))
}

/// The two points where each sampled ruling of `sweep` meets the other
/// surface, from `roots(u)` (the discriminant and the two axial parameters),
/// or `None` for a ruling that misses it.
fn ruling_samples(
    sweep: &CylindricalSurface,
    roots: &impl Fn(f64) -> (f64, f64, f64),
) -> Vec<Option<(Point3, Point3)>> {
    let lin_tol = Tolerance::new().linear;
    (0..RULING_SAMPLES)
        .map(|i| {
            let u = ruling_u(i);
            let (disc, vp, vm) = roots(u);
            (disc >= -lin_tol).then(|| (sweep.evaluate(u, vp), sweep.evaluate(u, vm)))
        })
        .collect()
}

/// Every ruling meets the other surface: each root traces a closed loop.
/// Where the loops pass closest at one ruling they are sampled from it,
/// closer together near it, so the fit follows the narrow neck between
/// them; where they touch there, they are the two halves of one curve
/// crossing itself, each with a corner at that point, which then falls at
/// their shared start and end, where the fit keeps it.
fn closed_ruling_loops(
    sweep: &CylindricalSurface,
    roots: &impl Fn(f64) -> (f64, f64, f64),
    samples: &[Option<(Point3, Point3)>],
) -> Vec<Vec<Point3>> {
    let (mut plus, mut minus): (Vec<Point3>, Vec<Point3>) =
        if let Some((neck, touching)) = narrowest_ruling(roots) {
            let count = 2 * RULING_SAMPLES;
            #[allow(clippy::cast_precision_loss)]
            (0..count)
                .map(|i| {
                    let t = i as f64 / count as f64;
                    let u = TAU.madd(t - 0.9 * (TAU * t).sin() / TAU, neck);
                    let (_, vp, vm) = roots(u);
                    if i == 0 && touching {
                        let at = sweep.evaluate(u, 0.5 * (vp + vm));
                        (at, at)
                    } else {
                        (sweep.evaluate(u, vp), sweep.evaluate(u, vm))
                    }
                })
                .unzip()
        } else {
            samples.iter().flatten().copied().unzip()
        };
    plus.push(plus[0]);
    minus.push(minus[0]);
    vec![plus, minus]
}

/// The ruling where the two roots pass closest, when exactly one ruling
/// stands out: a local minimum of their gap, scanned finer than the
/// sampling and refined, that either closes to within tolerance (the
/// roots touch there; the tolerance scales with the widest gap, the loops'
/// own size, so the decision does not move with the model) or narrows
/// below a quarter of the widest gap while no other minimum does. Equal
/// crossing cylinders touch at two rulings and get `None`.
fn narrowest_ruling(roots: &impl Fn(f64) -> (f64, f64, f64)) -> Option<(f64, bool)> {
    let scan = WINDOW_SCAN * RULING_SAMPLES;
    #[allow(clippy::cast_precision_loss)]
    let step = TAU / scan as f64;
    let gap = |u: f64| {
        let (_, vp, vm) = roots(u);
        (vp - vm).abs()
    };
    #[allow(clippy::cast_precision_loss)]
    let gaps: Vec<f64> = (0..scan).map(|k| gap(step * k as f64)).collect();
    let widest = gaps.iter().copied().fold(0.0, f64::max);
    let tol = Tolerance::new().linear * (1.0 + widest);
    let mut necks = Vec::new();
    for k in 0..scan {
        let (before, here, after) = (gaps[(k + scan - 1) % scan], gaps[k], gaps[(k + 1) % scan]);
        if here > before || here >= after || here >= 0.25 * widest {
            continue;
        }
        #[allow(clippy::cast_precision_loss)]
        let (mut lo, mut hi) = (step * (k as f64 - 1.0), step * (k as f64 + 1.0));
        for _ in 0..100 {
            let (a, b) = (lo + (hi - lo) / 3.0, hi - (hi - lo) / 3.0);
            if gap(a) < gap(b) {
                hi = b;
            } else {
                lo = a;
            }
        }
        let u = 0.5 * (lo + hi);
        necks.push((u, gap(u) <= tol));
    }
    match necks[..] {
        [neck] => Some(neck),
        _ => None,
    }
}

/// Only windows of rulings meet the other surface: each cyclic window
/// carries one loop, out along one root and back along the other, the two
/// joined where the discriminant vanishes. Empty when no sample meets it (a
/// window narrower than the sampling).
///
/// Each window is sampled afresh between its two ends, closer together near
/// them: a window may hold only a few of the rulings sampled around the whole
/// turn, and the roots swing fastest where they join, so a fit through those
/// alone strays from the curve by far more than the tolerance (a ball beside a
/// wide ring's wall).
fn partial_ruling_loops(
    sweep: &CylindricalSurface,
    roots: &impl Fn(f64) -> (f64, f64, f64),
    samples: &[Option<(Point3, Point3)>],
) -> Vec<Vec<Point3>> {
    let lin_tol = Tolerance::new().linear;
    // Where the discriminant vanishes between a meeting ruling and a missing
    // one, approached from the meeting side.
    let branch_u = |inside: usize, outside: usize| -> f64 {
        let (mut lo, mut hi) = (ruling_u(inside), ruling_u(outside));
        if (hi - lo).abs() > std::f64::consts::PI {
            hi += if hi < lo { TAU } else { -TAU };
        }
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if roots(mid).0 >= 0.0 {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let joint = |u: f64| {
        let (_, vp, vm) = roots(u);
        sweep.evaluate(u, 0.5 * (vp + vm))
    };
    let Some(first_gap) = samples.iter().position(Option::is_none) else {
        return Vec::new();
    };
    let mut loops = Vec::new();
    let mut k = 0;
    while k < RULING_SAMPLES {
        let i = (first_gap + k) % RULING_SAMPLES;
        if samples[i].is_none() {
            k += 1;
            continue;
        }
        let start = i;
        let mut run = Vec::new();
        while k < RULING_SAMPLES {
            let j = (first_gap + k) % RULING_SAMPLES;
            let Some(pair) = samples[j] else { break };
            run.push(pair);
            k += 1;
        }
        let end = (start + run.len() - 1) % RULING_SAMPLES;
        let u_head = branch_u(start, (start + RULING_SAMPLES - 1) % RULING_SAMPLES);
        let u_tail = branch_u(end, (end + 1) % RULING_SAMPLES);
        let span = (u_tail - u_head).rem_euclid(TAU);
        #[allow(clippy::cast_precision_loss)]
        let fresh: Option<Vec<(Point3, Point3)>> = (1..RULING_SAMPLES)
            .map(|n| {
                let t = n as f64 / RULING_SAMPLES as f64;
                let u = span.madd(0.5 * (1.0 - (std::f64::consts::PI * t).cos()), u_head);
                let (disc, vp, vm) = roots(u);
                (disc >= -lin_tol).then(|| (sweep.evaluate(u, vp), sweep.evaluate(u, vm)))
            })
            .collect();
        // A gap inside the window that the sampling stepped over: keep the
        // sampled rulings, which stay on one side of it.
        let run = fresh.unwrap_or(run);
        let (head, tail) = (joint(u_head), joint(u_head + span));
        let mut pts = vec![head];
        pts.extend(run.iter().map(|p| p.0));
        pts.push(tail);
        pts.extend(run.iter().rev().map(|p| p.1));
        pts.push(head);
        loops.push(pts);
    }
    loops
}

/// Cubic interpolants through the swept loops, with `params(p)` giving each
/// point's parameters on the two surfaces.
fn fit_ruling_loops(
    loops: &[Vec<Point3>],
    params: impl Fn(Point3) -> ((f64, f64), (f64, f64)),
) -> Vec<IntersectionCurve> {
    let mut curves = Vec::new();
    for pts in loops {
        if pts.len() < 4 {
            continue;
        }
        let ipts: Vec<IntersectionPoint> = pts
            .iter()
            .map(|&p| {
                let (param1, param2) = params(p);
                IntersectionPoint {
                    point: p,
                    param1,
                    param2,
                }
            })
            .collect();
        let degree = 3.min(pts.len() - 1);
        if let Ok(curve) = interpolate(pts, degree) {
            curves.push(IntersectionCurve {
                curve,
                points: ipts,
            });
        }
    }
    curves
}

/// Algebraic cone-cylinder intersection for PARALLEL (or antiparallel) axes.
///
/// When the axes are parallel, every plane perpendicular to them cuts the cone
/// in a circle of radius `rho = v * cos(half_angle)` about a FIXED centre and
/// the cylinder in a circle of radius `R` about a second FIXED centre, so the
/// axis separation `d` is constant in `v`. Two coplanar circles meet at
/// `u = phi0 +/- acos((d^2 + rho^2 - R^2) / (2*d*rho))`, giving two branches
/// parameterised exactly by the cone's own `v`. The branches exist only where
/// `rho` lies in `[|d - R|, d + R]`, which bounds the curve naturally.
///
/// This replaces the general grid-seeded marcher for the configuration, which
/// mis-handles it badly: seeds are accepted anywhere within half the surface
/// diagonal of the partner, the march-result dedup only consumes seeds the
/// traced polyline passes near, and the survivors are dozens of overlapping
/// partial traces of the same curve. Those fragments carry no usable in-face
/// span, so a cone corner-round crossed by a boss cylinder never splits (a
/// counterbore/countersink meeting a pad — the gridfinity lightweight base).
///
/// Returns `None` (defer to the caller's other paths) when the axes are not
/// parallel, or when they are coaxial — a coaxial pair degenerates to shared
/// circles, which [`exact_cone_cylinder`] emits exactly and phase FF calls
/// directly. Note that `intersect_analytic_analytic_bounded` does NOT consult
/// `exact_cone_cylinder`, so a coaxial pair reaching this path through that
/// caller falls through to the marcher; only the FF path gets the exact circles.
// Result-wrapped to match the other `try_algebraic_intersection` arms' shape.
#[allow(clippy::unnecessary_wraps)]
fn algebraic_parallel_cone_cylinder(
    cone: &ConicalSurface,
    cyl: &CylindricalSurface,
    v_range_cone: Option<(f64, f64)>,
    v_range_cyl: Option<(f64, f64)>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    let axis = cone.axis();
    if axis.cross(cyl.axis()).length() > Tolerance::new().angular {
        return Ok(None); // Skew or oblique: `ruling_cone_cylinder` traces it.
    }

    let apex = cone.apex();
    let delta = cyl.origin() - apex;
    let along = delta.dot(axis);
    let perp = delta - axis * along;
    let d = perp.length();
    if d < 1e-9 {
        return Ok(None); // Coaxial — `exact_cone_cylinder` owns this.
    }

    let (e1, e2) = (cone.x_axis(), cone.y_axis());
    let phi0 = perp.dot(e2).atan2(perp.dot(e1));

    let (sin_t, cos_t) = cone.half_angle().sin_cos();
    if cos_t < 1e-12 || sin_t < 1e-12 {
        return Ok(None);
    }
    let r = cyl.radius();

    // Branch existence: |d - R| <= rho <= d + R, with rho = v * cos(half_angle).
    let mut v_min = (d - r).abs() / cos_t;
    let mut v_max = (d + r) / cos_t;
    if v_max <= v_min {
        return Ok(Some(vec![]));
    }

    // Narrow the sampled span to the faces' own extents so the fixed sample
    // budget resolves the in-face part of the curve rather than spreading over
    // a loop that mostly lies off both patches. A face's crossing can be a
    // fraction of a degree of the cone's sweep (the corner-round case above),
    // and an unnarrowed sampling puts fewer than one sample across it.
    let mut lo = v_min;
    let mut hi = v_max;
    // Clip EXACTLY to the hints, not to a padded window: an endpoint that lands
    // exactly on the face's own v-limit lies ON that boundary rim, so the
    // downstream pave machinery anchors it to the rim edge instead of leaving
    // the section dangling just past the face.
    if let Some((a, b)) = v_range_cone {
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        lo = lo.max(a);
        hi = hi.min(b);
    }
    if let Some((a, b)) = v_range_cyl {
        // The cylinder's v is a signed distance along its axis from its origin;
        // convert both ends to the cone's v via the shared axial direction.
        let flip = cyl.axis().dot(axis);
        let to_cone_v = |cv: f64| (along + cv * flip) / sin_t;
        let (a, b) = (to_cone_v(a), to_cone_v(b));
        let (a, b) = if a <= b { (a, b) } else { (b, a) };
        lo = lo.max(a);
        hi = hi.min(b);
    }
    let (turn_lo, turn_hi) = (v_min, v_max);
    v_min = lo.max(v_min);
    v_max = hi.min(v_max);
    if v_max - v_min <= 1e-12 {
        return Ok(Some(vec![]));
    }
    // A cylinder beside the axis, with both turning points inside both faces,
    // meets the cone in a whole loop: traced along the cylinder's rulings,
    // each of which meets the nappe once, it comes back as one closed curve,
    // which the band split of the cylinder it winds needs. Around the axis
    // the loop winds the cone too and stays in two branches, and so does a
    // wall passing so near the axis that the loop's bend there, about
    // `(d - r) / sqrt(d r)` of a turn wide, spans fewer than a few rulings.
    let slack = Tolerance::new().linear;
    #[allow(clippy::cast_precision_loss)]
    let resolved = d - r > 3.0 * (d * r).sqrt() * TAU / RULING_SAMPLES as f64;
    if v_min <= turn_lo + slack && v_max >= turn_hi - slack && resolved {
        let mut pts: Vec<Point3> = (0..RULING_SAMPLES)
            .map(|i| {
                let (sin_u, cos_u) = ruling_u(i).sin_cos();
                let foot = cyl.origin() + (cyl.x_axis() * cos_u + cyl.y_axis() * sin_u) * r;
                let off = foot - apex;
                let across = off - axis * off.dot(axis);
                apex + across + axis * (across.length() * sin_t / cos_t)
            })
            .collect();
        pts.push(pts[0]);
        return Ok(Some(fit_ruling_loops(&[pts], |p| {
            (cone.project_point(p), cyl.project_point(p))
        })));
    }

    let n_samples = 128;
    let mut plus: Vec<Point3> = Vec::with_capacity(n_samples + 1);
    let mut minus: Vec<Point3> = Vec::with_capacity(n_samples + 1);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..=n_samples {
        let v = v_min + (v_max - v_min) * (i as f64) / (n_samples as f64);
        let rho = v * cos_t;
        if rho < 1e-12 {
            // The apex. `cos_alpha` has rho in its denominator, so it is only
            // meaningful in the limit: it tends to 0 (alpha -> pi/2) when the
            // cylinder passes exactly through the apex (d == R), and diverges
            // otherwise — where the clamp would manufacture a spurious alpha of
            // 0 or pi. So keep the apex only in the d == R case, where it is a
            // genuine point of the intersection and the shared endpoint at
            // which the two branches meet.
            if (d - r).abs() < 1e-12 {
                let apex = cone.evaluate(phi0, v);
                plus.push(apex);
                minus.push(apex);
            }
            continue;
        }
        let cos_alpha = ((d * d + rho * rho - r * r) / (2.0 * d * rho)).clamp(-1.0, 1.0);
        let alpha = cos_alpha.acos();
        plus.push(cone.evaluate(phi0 + alpha, v));
        minus.push(cone.evaluate(phi0 - alpha, v));
    }

    let mut curves = Vec::new();
    for pts in [&plus, &minus] {
        // Fewer than four samples in range means this branch does not cross the
        // bounded region at all (the other branch may still).
        if pts.len() < 4 {
            continue;
        }
        let ipts: Vec<IntersectionPoint> = pts
            .iter()
            .map(|&p| IntersectionPoint {
                point: p,
                param1: cone.project_point(p),
                param2: cyl.project_point(p),
            })
            .collect();
        let degree = 3.min(pts.len() - 1);
        match interpolate(pts, degree) {
            Ok(curve) => curves.push(IntersectionCurve {
                curve,
                points: ipts,
            }),
            // Emitting only the branch that happened to fit would starve the
            // section chain of exactly the piece this path exists to supply —
            // the same silent half-answer the marcher's fragments produced.
            // Defer the whole pair to the caller's other paths instead.
            Err(_) => return Ok(None),
        }
    }

    Ok(Some(curves))
}

/// How many times a parallel cone-cone branch's fitted curve may be refined
/// where it strays.
const PARALLEL_CONES_FIT_ROUNDS: usize = 8;

/// The distance from `p` to the nappe of `cone` nearest it, for a point near
/// the cone's wall: the distance in `p`'s meridian plane to the generator.
fn cone_wall_distance(cone: &ConicalSurface, p: Point3) -> f64 {
    let rel = p - cone.apex();
    let along = rel.dot(cone.axis());
    let across = (rel - cone.axis() * along).length();
    let (sin, cos) = cone.half_angle().sin_cos();
    across.madd(sin, -along * cos)
}

/// Cone-cone intersection for PARALLEL (or antiparallel) axes that do not
/// coincide, with openings of unequal slope (equal slopes give the exact
/// conic of [`offset_parallel_cone_cone`]).
///
/// Every plane across the axes cuts each cone in a circle about a fixed
/// centre, the two centres `d` apart, and both radii change linearly with the
/// height. Two coplanar circles meet at
/// `u = phi0 +/- acos((d^2 + rho1^2 - rho2^2) / (2*d*rho1))` on the first
/// cone, which gives two branches parameterised exactly by its own `v`. They
/// exist where `|rho1 - rho2| <= d <= rho1 + rho2` on both nappes, conditions
/// linear in `v`, and join where one of the first holds with equality, where
/// the circles touch. There `u` changes like the square root of the distance
/// in `v`, so the samples close in on both ends of the span, and the branches
/// end at the one point where the circles touch.
///
/// Halfway between each pair of points the fitted curve must lie within a
/// quarter of the linear tolerance of both cones; where it strays, the
/// section's own point there joins the fit.
///
/// `None` (the marcher's case) when the axes are not parallel to the angular
/// tolerance or coincide, when a cone is flat or a line, when a range reaches
/// a cone's negative nappe, when nothing bounds the span, or when a fit still
/// strays.
fn algebraic_parallel_cone_cone(
    c1: &ConicalSurface,
    c2: &ConicalSurface,
    v_range_1: Option<(f64, f64)>,
    v_range_2: Option<(f64, f64)>,
) -> Option<Vec<IntersectionCurve>> {
    let tol = Tolerance::new();
    // The span below runs on both cones' positive nappes only.
    if [v_range_1, v_range_2]
        .into_iter()
        .flatten()
        .any(|(p, q)| p.min(q) < -tol.linear)
    {
        return None;
    }
    let axis = c1.axis();
    if axis.cross(c2.axis()).length() > tol.angular {
        return None;
    }
    let sigma = axis.dot(c2.axis()).signum();
    let delta = c2.apex() - c1.apex();
    let along = delta.dot(axis);
    let perp = delta - axis * along;
    let d = perp.length();
    if d < 1e-9 {
        return None;
    }
    let phi0 = perp.dot(c1.y_axis()).atan2(perp.dot(c1.x_axis()));
    let (sin1, cos1) = c1.half_angle().sin_cos();
    let (sin2, cos2) = c2.half_angle().sin_cos();
    if [sin1, cos1, sin2, cos2].iter().any(|&x| x < 1e-12) {
        return None;
    }
    // The second cone's radius at the first one's `v`, `a + b v`, from the
    // height `v sin1` above the first apex.
    let slope2 = cos2 / sin2;
    let (a, b) = (-sigma * along * slope2, sigma * sin1 * slope2);
    // The span, and whether each end is where the circles touch.
    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    let (mut lo_touch, mut hi_touch) = (false, false);
    // Each condition `k v + c >= 0`: the second nappe, then the circles
    // meeting (`rho1 + rho2 >= d`, `rho2 - rho1 <= d`, `rho1 - rho2 <= d`).
    for (k, c, touch) in [
        (b, a, false),
        (cos1 + b, a - d, true),
        (b - cos1, a + d, true),
        (cos1 - b, d - a, true),
    ] {
        // However small, a slope moves the bound: slopes nearly equal to
        // each other hold their condition only far out, past `-c / k`.
        if k > 0.0 {
            if -c / k > lo {
                (lo, lo_touch) = (-c / k, touch);
            }
        } else if k < 0.0 {
            if -c / k < hi {
                (hi, hi_touch) = (-c / k, touch);
            }
        } else if c < 0.0 {
            return Some(Vec::new());
        }
    }
    let mut clip = |p: f64, q: f64| {
        if p.min(q) > lo {
            (lo, lo_touch) = (p.min(q), false);
        }
        if p.max(q) < hi {
            (hi, hi_touch) = (p.max(q), false);
        }
    };
    if let Some((p, q)) = v_range_1 {
        clip(p, q);
    }
    if let Some((p, q)) = v_range_2 {
        let to_v1 = |v2: f64| sigma.madd(v2 * sin2, along) / sin1;
        clip(to_v1(p), to_v1(q));
    }
    if !hi.is_finite() {
        return None;
    }
    if hi - lo <= 1e-12 {
        return Some(Vec::new());
    }

    // The two points at `v`, or `None` at the first apex off the second cone.
    let at = |v: f64, touch: bool| -> Option<(Point3, Point3)> {
        let rho1 = v * cos1;
        let rho2 = b.madd(v, a);
        if rho1 < 1e-12 {
            // `u` means nothing at the apex, a point of the curve only where
            // it lies on the second cone.
            return ((rho2 - d).abs() < 1e-12).then(|| (c1.apex(), c1.apex()));
        }
        let cos_alpha = (d.madd(d, rho1 * rho1) - rho2 * rho2) / (2.0 * d * rho1);
        if touch {
            // Rounding leaves `cos_alpha` off +/-1 there, and `acos` turns
            // that into a gap between the branches' ends.
            let p = c1.evaluate(
                if cos_alpha >= 0.0 {
                    phi0
                } else {
                    phi0 + std::f64::consts::PI
                },
                v,
            );
            return Some((p, p));
        }
        let alpha = cos_alpha.clamp(-1.0, 1.0).acos();
        Some((c1.evaluate(phi0 + alpha, v), c1.evaluate(phi0 - alpha, v)))
    };
    // Samples are spaced evenly in `t`, which runs `v` like a square from
    // each end, where `u` runs like the square root of `v`.
    let v_at = |t: f64| (hi - lo).madd(0.5 * (1.0 - (std::f64::consts::PI * t).cos()), lo);
    let n_samples = 128;
    let mut plus: Vec<(f64, Point3)> = Vec::with_capacity(n_samples + 1);
    let mut minus: Vec<(f64, Point3)> = Vec::with_capacity(n_samples + 1);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..=n_samples {
        let t = i as f64 / n_samples as f64;
        let touch = (i == 0 && lo_touch) || (i == n_samples && hi_touch);
        if let Some((p, m)) = at(v_at(t), touch) {
            plus.push((t, p));
            minus.push((t, m));
        }
    }

    let off = |p: Point3| {
        cone_wall_distance(c1, p)
            .abs()
            .max(cone_wall_distance(c2, p).abs())
    };
    let mut curves = Vec::new();
    for (branch, side) in [(&mut plus, 0), (&mut minus, 1)] {
        if branch.len() < 4 {
            continue;
        }
        let mut rounds = 0;
        // One branch alone would leave the section chain short of the piece
        // this path exists to supply: a branch that fails defers the pair.
        let curve = loop {
            let positions: Vec<Point3> = branch.iter().map(|&(_, p)| p).collect();
            let curve = interpolate(&positions, 3).ok()?;
            let params = crate::nurbs::fitting::chord_length_params(&positions);
            let errors: Vec<f64> = (0..positions.len() - 1)
                .map(|k| off(curve.evaluate(f64::midpoint(params[k], params[k + 1]))))
                .collect();
            if errors.iter().all(|&e| e <= 0.25 * tol.linear) {
                break curve;
            }
            if rounds == PARALLEL_CONES_FIT_ROUNDS {
                return None;
            }
            rounds += 1;
            // Every span within a factor of four of the limit is split, and
            // a stray's neighbours too: the fit through new points moves the
            // error onto the spans beside them.
            let mut split: Vec<usize> = (0..errors.len())
                .filter(|&k| errors[k] > 0.0625 * tol.linear)
                .flat_map(|k| {
                    let stray = errors[k] > 0.25 * tol.linear;
                    let (first, last) = if stray {
                        (k.saturating_sub(1), k + 1)
                    } else {
                        (k, k)
                    };
                    first..=last.min(errors.len() - 1)
                })
                .collect();
            split.sort_unstable();
            split.dedup();
            for &k in split.iter().rev() {
                let t = f64::midpoint(branch[k].0, branch[k + 1].0);
                let (p, m) = at(v_at(t), false)?;
                branch.insert(k + 1, (t, if side == 0 { p } else { m }));
            }
        };
        let points = branch
            .iter()
            .map(|&(_, p)| IntersectionPoint {
                point: p,
                param1: c1.project_point(p),
                param2: c2.project_point(p),
            })
            .collect();
        curves.push(IntersectionCurve { curve, points });
    }
    Some(curves)
}

/// Algebraic sphere-sphere intersection.
///
/// Two spheres intersect in a circle lying in the radical plane.
/// The radical plane is perpendicular to the line connecting the centers,
/// at a distance d1 from center1 where:
///   d1 = (D² + R1² - R2²) / (2D)
/// and D is the distance between centers.
fn algebraic_sphere_sphere(
    s1: &SphericalSurface,
    s2: &SphericalSurface,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let c1 = s1.center();
    let c2 = s2.center();
    let r1 = s1.radius();
    let r2 = s2.radius();

    let delta = c2 - c1;
    let d_sq = delta.x() * delta.x() + delta.y() * delta.y() + delta.z() * delta.z();
    let d = d_sq.sqrt();

    if d < 1e-12 {
        // Concentric spheres: no intersection (unless same radius → degenerate).
        return Ok(vec![]);
    }

    // Check separation conditions.
    if d > r1 + r2 + 1e-10 {
        return Ok(vec![]); // Too far apart
    }
    if d + r2.min(r1) + 1e-10 < r1.max(r2) {
        return Ok(vec![]); // One inside the other
    }

    // Distance from c1 to the radical plane along the center line.
    let d1 = (d_sq + r1 * r1 - r2 * r2) / (2.0 * d);

    // Radius of the intersection circle.
    let r_circle_sq = r1 * r1 - d1 * d1;
    if r_circle_sq < 0.0 {
        // Tangent or no intersection (numerical noise).
        if r_circle_sq > -1e-10 {
            // Tangent: single point.
            let axis = Vec3::new(delta.x() / d, delta.y() / d, delta.z() / d);
            let tangent_pt = Point3::new(
                c1.x() + axis.x() * d1,
                c1.y() + axis.y() * d1,
                c1.z() + axis.z() * d1,
            );
            let ipt = IntersectionPoint {
                point: tangent_pt,
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            };
            // Single-point "curve" — not very useful but correct.
            return Ok(vec![IntersectionCurve {
                curve: interpolate(&[tangent_pt, tangent_pt], 1)?,
                points: vec![ipt],
            }]);
        }
        return Ok(vec![]);
    }

    let r_circle = r_circle_sq.sqrt();
    let axis = Vec3::new(delta.x() / d, delta.y() / d, delta.z() / d);
    let center = Point3::new(
        c1.x() + axis.x() * d1,
        c1.y() + axis.y() * d1,
        c1.z() + axis.z() * d1,
    );

    // Build a reference frame for the circle.
    let basis = Frame3::from_normal(center, axis)?;
    let u_dir = basis.x;
    let v_dir = basis.y;

    // Sample points along the circle for the IntersectionCurve.
    let n_samples = 33; // Odd for symmetry
    let mut points = Vec::with_capacity(n_samples);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n_samples {
        let theta = TAU * i as f64 / (n_samples - 1) as f64;
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = Point3::new(
            center.x() + (u_dir.x() * cos_t + v_dir.x() * sin_t) * r_circle,
            center.y() + (u_dir.y() * cos_t + v_dir.y() * sin_t) * r_circle,
            center.z() + (u_dir.z() * cos_t + v_dir.z() * sin_t) * r_circle,
        );
        points.push(IntersectionPoint {
            point: pt,
            param1: (0.0, 0.0),
            param2: (0.0, 0.0),
        });
    }

    let curve = conic_loop(center, u_dir, v_dir, r_circle, r_circle)?;

    Ok(vec![IntersectionCurve { curve, points }])
}

/// Newton correction: project a point back onto the intersection curve
/// of two analytic surfaces. Solves the 3×3 system:
///   δ · na = -da  (eliminate distance to surface A)
///   δ · nb = -db  (eliminate distance to surface B)
///   δ · t  = 0    (minimal correction, perpendicular to tangent)
#[allow(clippy::too_many_arguments)]
fn correct_to_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
    surf_a: &dyn Fn(f64, f64) -> Point3,
    norm_a: &dyn Fn(f64, f64) -> Vec3,
    surf_b: &dyn Fn(f64, f64) -> Point3,
    norm_b: &dyn Fn(f64, f64) -> Vec3,
    point: Point3,
    u_range_a: (f64, f64),
    v_range_a: (f64, f64),
    u_range_b: (f64, f64),
    v_range_b: (f64, f64),
    max_iters: usize,
) -> Point3 {
    let mut p = point;
    for _ in 0..max_iters {
        let (ua, va) = project_analytic(a, p, u_range_a, v_range_a);
        let (ub, vb) = project_analytic(b, p, u_range_b, v_range_b);
        let pa = surf_a(ua, va);
        let pb = surf_b(ub, vb);
        let na = norm_a(ua, va);
        let nb = norm_b(ub, vb);
        let pv = Vec3::new(p.x(), p.y(), p.z());

        let da = (pv - Vec3::new(pa.x(), pa.y(), pa.z())).dot(na);
        let db = (pv - Vec3::new(pb.x(), pb.y(), pb.z())).dot(nb);

        if da.abs() < 1e-7 && db.abs() < 1e-7 {
            break;
        }

        let t = na.cross(nb);
        let t_len = t.length();
        if t_len < 1e-10 {
            // Surfaces are tangent — fall back to midpoint.
            return Point3::new(
                (pa.x() + pb.x()) * 0.5,
                (pa.y() + pb.y()) * 0.5,
                (pa.z() + pb.z()) * 0.5,
            );
        }
        let t_hat = t * (1.0 / t_len);

        // Solve [na; nb; t_hat] · δ = [-da, -db, 0] via Cramer's rule.
        let det = na.x() * (nb.y() * t_hat.z() - nb.z() * t_hat.y())
            - na.y() * (nb.x() * t_hat.z() - nb.z() * t_hat.x())
            + na.z() * (nb.x() * t_hat.y() - nb.y() * t_hat.x());
        if det.abs() < 1e-15 {
            return Point3::new(
                (pa.x() + pb.x()) * 0.5,
                (pa.y() + pb.y()) * 0.5,
                (pa.z() + pb.z()) * 0.5,
            );
        }
        let inv = 1.0 / det;
        // Cramer's rule: replace each column of A with rhs = (-da, -db, 0).
        let dx = inv
            * (-da * (nb.y() * t_hat.z() - nb.z() * t_hat.y())
                + db * (na.y() * t_hat.z() - na.z() * t_hat.y()));
        let dy = inv
            * (da * (nb.x() * t_hat.z() - nb.z() * t_hat.x())
                - db * (na.x() * t_hat.z() - na.z() * t_hat.x()));
        let dz = inv
            * (-da * (nb.x() * t_hat.y() - nb.y() * t_hat.x())
                + db * (na.x() * t_hat.y() - na.y() * t_hat.x()));
        let candidate = Point3::new(p.x() + dx, p.y() + dy, p.z() + dz);

        // Divergence guard: if the correction moves farther from both
        // surfaces, abandon Newton and return the best point so far.
        let (uc, vc) = project_analytic(a, candidate, u_range_a, v_range_a);
        let (ud, vd) = project_analytic(b, candidate, u_range_b, v_range_b);
        let pc_a = surf_a(uc, vc);
        let pc_b = surf_b(ud, vd);
        let cv = Vec3::new(candidate.x(), candidate.y(), candidate.z());
        let da_new = (cv - Vec3::new(pc_a.x(), pc_a.y(), pc_a.z()))
            .dot(norm_a(uc, vc))
            .abs();
        let db_new = (cv - Vec3::new(pc_b.x(), pc_b.y(), pc_b.z()))
            .dot(norm_b(ud, vd))
            .abs();
        if da_new > da.abs() && db_new > db.abs() {
            return p;
        }

        p = candidate;
    }
    p
}

/// The most a march step turns the section's tangent, in radians.
const MARCH_TURN: f64 = 0.2;

/// The share of the distance to the section's next branch that a march
/// step's guess may miss the section by.
const MARCH_BASIN: f64 = 0.25;

/// The most steps a march takes each way.
const MARCH_STEPS: usize = 4096;

/// A point of the section of two surfaces, with what a march step from it
/// needs.
#[derive(Clone, Copy)]
struct Station {
    p: Point3,
    /// The unit tangent of the section, along the march.
    t: Vec3,
    /// The curvature of the section.
    bend: f64,
    /// The distance, across the section, at which the two surfaces' second
    /// order models through `p` cross again: a lower estimate of how far
    /// the section's next branch lies.
    gap: f64,
    /// The sine of the angle between the two surfaces' normals.
    sin: f64,
}

/// The station at `p`, a point of the section of `a` and `b`, its tangent
/// along `toward`; `None` where the two surfaces touch there (their
/// normals within about 1e-6 of parallel) or a distance has no gradient.
///
/// The section's curvature vector `k` lies across it, and the curve's
/// normal curvature on each surface, the second derivative of its distance
/// along the tangent, is `-k . n` for that surface's normal `n`. Across the
/// section, each surface's trace is a curve of the curvature its distance
/// gives along the trace; two circles that cross at angle `θ`, of
/// curvatures `κa` and `κb`, cross again no nearer than `2 sin θ / (κa + κb)`.
fn station(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
    p: Point3,
    toward: Vec3,
) -> Option<Station> {
    use section_fit::Carrier;
    let (_, na) = a.distance(p)?;
    let (_, nb) = b.distance(p)?;
    let cross = na.cross(nb);
    let sin = cross.length();
    if sin < 1e-6 {
        return None;
    }
    let mut t = cross * (1.0 / sin);
    if t.dot(toward) < 0.0 {
        t = -t;
    }
    let (ka, kb) = (a.bend(p, t)?, b.bend(p, t)?);
    let c = na.dot(nb);
    let det = sin * sin;
    let along_a = c.madd(kb, -ka) / det;
    let along_b = c.madd(ka, -kb) / det;
    let bend = (2.0 * along_a * along_b)
        .madd(c, along_a.madd(along_a, along_b * along_b))
        .max(0.0)
        .sqrt();
    let (qa, qb) = (a.bend(p, t.cross(na))?, b.bend(p, t.cross(nb))?);
    let spread = qa.abs() + qb.abs();
    let gap = if spread > 0.0 {
        2.0 * sin / spread
    } else {
        f64::INFINITY
    };
    bend.is_finite().then_some(Station {
        p,
        t,
        bend,
        gap,
        sin,
    })
}

impl Station {
    /// The longest step from here: one that turns the tangent by no more
    /// than [`MARCH_TURN`], and whose guess along the tangent misses the
    /// section by no more than [`MARCH_BASIN`] of the gap.
    fn reach(&self, longest: f64) -> f64 {
        if self.bend > 0.0 {
            longest
                .min(MARCH_TURN / self.bend)
                .min((2.0 * MARCH_BASIN * self.gap / self.bend).sqrt())
        } else {
            longest
        }
    }
}

/// March along the section of two surfaces from `seed`, a point on it, each
/// way, until the march comes back to the seed, leaves a patch or the
/// region, or reaches a point where the section's branches meet or the
/// surfaces touch; `None` where it cannot follow the section or runs out
/// of steps.
///
/// Each step goes along the tangent, no farther than [`Station::reach`],
/// and moves onto the section by Newton steps. It is taken again at half the
/// length unless the move onto the section stays within [`MARCH_BASIN`] of
/// the gap at both ends (so the step stays on its branch), the tangent turns
/// by less than twice [`MARCH_TURN`], and the step goes forward by at least
/// half its length (so it does not pass a turn of the section). A march
/// closes where the seed lies within its next step, ahead along the
/// tangent and near it, and the section there runs the way it ran from the
/// seed: the march comes back to the seed along the curve, from the seed's
/// own side.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn march_analytic_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
    surf_a: &dyn Fn(f64, f64) -> Point3,
    norm_a: &dyn Fn(f64, f64) -> Vec3,
    surf_b: &dyn Fn(f64, f64) -> Point3,
    norm_b: &dyn Fn(f64, f64) -> Vec3,
    seed: Point3,
    u_range_a: (f64, f64),
    v_range_a: (f64, f64),
    u_range_b: (f64, f64),
    v_range_b: (f64, f64),
    initial_step: f64,
    u_periodic_a: bool,
    u_periodic_b: bool,
    region: Option<Aabb3>,
) -> Option<Vec<Point3>> {
    let h_max = initial_step * 4.0;
    let h_min = initial_step * 1e-6;

    // Whether `q` lies off either patch, its feet clamped to a patch's edge.
    let off_patch = |q: Point3| {
        let (ua, va) = project_analytic(a, q, u_range_a, v_range_a);
        let (ub, vb) = project_analytic(b, q, u_range_b, v_range_b);
        (!u_periodic_a && (ua <= u_range_a.0 || ua >= u_range_a.1))
            || va <= v_range_a.0
            || va >= v_range_a.1
            || (!u_periodic_b && (ub <= u_range_b.0 || ub >= u_range_b.1))
            || vb <= v_range_b.0
            || vb >= v_range_b.1
    };
    // The point on the intersection that the step from `from` reaches.
    let on_curve = |from: Point3, step: Vec3| {
        correct_to_intersection(
            a,
            b,
            surf_a,
            norm_a,
            surf_b,
            norm_b,
            from + step,
            u_range_a,
            v_range_a,
            u_range_b,
            v_range_b,
            5,
        )
    };
    // The last point on the intersection still on both patches along the
    // step from `from`, whose end lies off a patch: on the patch's edge the
    // step crosses, bisected; `None` when the step leaves at once.
    let last_on_patch = |from: Point3, step: Vec3| {
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..48 {
            let s = f64::midpoint(lo, hi);
            if off_patch(on_curve(from, step * s)) {
                hi = s;
            } else {
                lo = s;
            }
        }
        (lo > 0.0).then(|| on_curve(from, step * lo))
    };

    let start = station(a, b, seed, Vec3::new(0.0, 0.0, 0.0))?;

    // March forward from seed, collecting points.
    let mut forward = Vec::new();
    // March backward from seed, collecting points (reversed at end).
    let mut backward = Vec::new();

    // A loop the forward march closes is whole: marched back as well, it
    // would come round a second time.
    let mut looped = false;
    for (direction, points) in [(1.0_f64, &mut forward), (-1.0_f64, &mut backward)] {
        if looped {
            break;
        }
        let home = start.t * direction;
        let mut here = Station { t: home, ..start };
        let mut h = here.reach(h_max.min(initial_step));
        let mut ended = false;

        for _ in 0..MARCH_STEPS {
            h = h.min(here.reach(h_max));
            if !points.is_empty() {
                let to_seed = seed - here.p;
                let ahead = to_seed.dot(here.t);
                let aside = (to_seed - here.t * ahead).length();
                if ahead > 0.0
                    && ahead <= h
                    && aside <= MARCH_TURN * ahead
                    && aside <= MARCH_BASIN * here.gap.min(start.gap)
                    && home.dot(here.t) >= (2.0 * MARCH_TURN).cos()
                {
                    points.push(seed);
                    looped = true;
                    ended = true;
                    break;
                }
            }

            let next = loop {
                if h < h_min {
                    break None;
                }
                let guess = Point3::new(
                    h.madd(here.t.x(), here.p.x()),
                    h.madd(here.t.y(), here.p.y()),
                    h.madd(here.t.z(), here.p.z()),
                );
                let taken = settle_on_section(a, b, guess)
                    .and_then(|q| station(a, b, q, here.t))
                    .filter(|next| {
                        let missed = (next.p - guess).length();
                        missed <= MARCH_BASIN * here.gap.min(next.gap)
                            && missed <= 0.5 * h
                            && next.t.dot(here.t) >= (2.0 * MARCH_TURN).cos()
                            && (next.p - here.p).dot(here.t) >= 0.5 * h
                    });
                match taken {
                    Some(next) => break Some(next),
                    None => h *= 0.5,
                }
            };
            // A step a millionth of the first that still leaves its branch,
            // turns back, or finds no crossing, where the normals lie within
            // 1e-4 of parallel, meets a point where the section's branches
            // meet or the surfaces touch: the curve ends there, as at a
            // patch's edge. Anywhere else the march cannot follow the
            // section, and the section is refused.
            let Some(next) = next else {
                if here.sin < 1e-4 {
                    ended = true;
                    break;
                }
                return None;
            };

            if off_patch(next.p) {
                // The step crosses a patch's edge. End the curve on that
                // edge, neither a step short of it nor past it, where the
                // faces' other sections meet it.
                points.extend(last_on_patch(here.p, next.p - here.p));
                ended = true;
                break;
            }
            if region.is_some_and(|r| !r.contains_point(next.p)) {
                points.push(next.p);
                ended = true;
                break;
            }
            points.push(next.p);
            here = next;
            h *= 2.0;
        }
        if !ended {
            return None;
        }
    }

    // Assemble result: backward (reversed) + seed + forward
    backward.reverse();
    let mut result = backward;
    result.push(seed);
    result.append(&mut forward);

    // A seed settles onto the intersection with its feet clamped to the
    // patches, so one can lie just past a patch's edge and end the curve
    // there: move that end back onto the edge.
    if let [first, second, ..] = result[..]
        && off_patch(first)
        && !off_patch(second)
        && let Some(edge) = last_on_patch(second, first - second)
    {
        result[0] = edge;
    }
    if let [.., second, last] = result[..]
        && off_patch(last)
        && !off_patch(second)
        && let Some(edge) = last_on_patch(second, last - second)
        && let Some(end) = result.last_mut()
    {
        *end = edge;
    }

    // Settle each point onto the intersection curve, to rounding where
    // Newton steps on both surfaces' distances converge.
    for pt in &mut result {
        let from = *pt;
        *pt = settle_on_section(a, b, from)
            .filter(|on| (*on - from).length() <= initial_step)
            .unwrap_or_else(|| {
                correct_to_intersection(
                    a, b, surf_a, norm_a, surf_b, norm_b, *pt, u_range_a, v_range_a, u_range_b,
                    v_range_b, 5,
                )
            });
    }

    Some(result)
}

/// Project a 3D point onto an analytic surface using the surface's
/// analytical projection method. Falls back to grid search for surface
/// types without analytical projection.
fn project_analytic(
    surface: &AnalyticSurface<'_>,
    point: Point3,
    u_range: (f64, f64),
    v_range: (f64, f64),
) -> (f64, f64) {
    match surface {
        AnalyticSurface::Cylinder(cyl) => {
            let (u, v) = cyl.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        AnalyticSurface::Sphere(sphere) => {
            let (u, v) = sphere.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
        // A foot on the negative nappe lies along `-radial(u)` from the apex:
        // its `u` is half a turn from the point's own side of the axis.
        AnalyticSurface::Cone(cone) => {
            let (u, v) = cone.project_point(point);
            let v = v.clamp(v_range.0, v_range.1);
            let u = if v < 0.0 {
                (u + std::f64::consts::PI).rem_euclid(TAU)
            } else {
                u
            };
            (u.clamp(u_range.0, u_range.1), v)
        }
        AnalyticSurface::Torus(torus) => {
            let (u, v) = torus.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
        }
    }
}

/// Returns `true` if the surface's u-parameter is periodic (wraps around 2π).
/// All current `AnalyticSurface` variants have periodic u — this is trivially
/// true today but exists as a guard for future non-periodic analytic types.
fn is_u_periodic(surface: &AnalyticSurface<'_>) -> bool {
    matches!(
        surface,
        AnalyticSurface::Cylinder(_)
            | AnalyticSurface::Cone(_)
            | AnalyticSurface::Sphere(_)
            | AnalyticSurface::Torus(_)
    )
}

/// Extract closures and parameter ranges for an analytic surface.
#[allow(clippy::type_complexity)]
fn surface_closures<'a>(
    surface: &'a AnalyticSurface<'a>,
) -> (
    Box<dyn Fn(f64, f64) -> Point3 + 'a>,
    Box<dyn Fn(f64, f64) -> Vec3 + 'a>,
    (f64, f64),
    (f64, f64),
) {
    match surface {
        AnalyticSurface::Cylinder(cyl) => (
            Box::new(|u, v| cyl.evaluate(u, v)),
            Box::new(|u, v| cyl.normal(u, v)),
            (0.0, TAU),
            (-1.0, 1.0),
        ),
        AnalyticSurface::Cone(cone) => (
            Box::new(|u, v| cone.evaluate(u, v)),
            Box::new(|u, v| cone.normal(u, v)),
            (0.0, TAU),
            (0.01, 2.0),
        ),
        AnalyticSurface::Sphere(sphere) => (
            Box::new(|u, v| sphere.evaluate(u, v)),
            Box::new(|u, v| sphere.normal(u, v)),
            (0.0, TAU),
            (-FRAC_PI_2, FRAC_PI_2),
        ),
        AnalyticSurface::Torus(torus) => (
            Box::new(|u, v| torus.evaluate(u, v)),
            Box::new(|u, v| torus.normal(u, v)),
            (0.0, TAU),
            (0.0, TAU),
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::tolerance::Tolerance;

    /// Arcs of a cone's hyperbola (a plane parallel to the axis) and parabola
    /// (a plane parallel to a ruling) between two of their sampled points
    /// stay on both the plane and the cone everywhere, not just at samples.
    #[test]
    fn plane_cone_conic_arcs_lie_on_both_surfaces() {
        let half_angle = 1.1_f64;
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            half_angle,
        )
        .unwrap();
        let ruling = Vec3::new(half_angle.sin(), 0.0, half_angle.cos());
        for (normal, d) in [(Vec3::new(1.0, 0.0, 0.0), 0.5), (ruling, 1.0)] {
            let chains =
                exact_plane_analytic_reaching(AnalyticSurface::Cone(&cone), normal, d, 10.0)
                    .unwrap();
            let chain = chains
                .iter()
                .find_map(|c| match c {
                    ExactIntersectionCurve::Points(chain) => Some(chain),
                    _ => None,
                })
                .expect("a parabola or hyperbola section is sampled");
            let (from, to) = (chain[2], chain[chain.len() - 3]);
            let arc = plane_cone_conic_arc(&cone, normal, d, from, to)
                .unwrap()
                .expect("an exact arc");
            let (t0, t1) = arc.domain();
            assert!((arc.evaluate(t0) - from).length() < 1e-12);
            assert!((arc.evaluate(t1) - to).length() < 1e-12);
            for i in 0..=200 {
                let q = arc.evaluate(t0 + (t1 - t0) * f64::from(i) / 200.0);
                let w = q - Point3::new(0.0, 0.0, 0.0);
                let off_plane = (normal.dot(w) - d).abs();
                let off_cone = (w.z() - w.length() * half_angle.sin()).abs();
                assert!(off_plane < 1e-9, "off the plane by {off_plane}");
                assert!(off_cone < 1e-9, "off the cone by {off_cone}");
            }
        }
    }

    /// A plane a few 1e-10 short of parallel to a ruling cuts a vast ellipse
    /// that the parabola's closed form only approximates: the arc is either
    /// declined or on the cone, and an arc with coincident ends is declined.
    #[test]
    fn plane_cone_conic_arc_declines_a_near_parabolic_ellipse() {
        let half_angle = 1.1_f64;
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            half_angle,
        )
        .unwrap();
        for shortfall in [1e-10, 3e-10, 8e-10] {
            let tilt = half_angle - shortfall / (2.0 * half_angle).sin();
            let normal = Vec3::new(tilt.sin(), 0.0, tilt.cos());
            let chains =
                exact_plane_analytic_reaching(AnalyticSurface::Cone(&cone), normal, 1.0, 10.0)
                    .unwrap();
            let Some(chain) = chains.iter().find_map(|c| match c {
                ExactIntersectionCurve::Points(chain) => Some(chain),
                _ => None,
            }) else {
                continue;
            };
            let (from, to) = (chain[2], chain[chain.len() - 3]);
            assert!(
                plane_cone_conic_arc(&cone, normal, 1.0, from, from)
                    .unwrap()
                    .is_none(),
                "coincident ends"
            );
            let Some(arc) = plane_cone_conic_arc(&cone, normal, 1.0, from, to).unwrap() else {
                continue;
            };
            let (t0, t1) = arc.domain();
            for i in 0..=200 {
                let w = arc.evaluate(t0 + (t1 - t0) * f64::from(i) / 200.0)
                    - Point3::new(0.0, 0.0, 0.0);
                let off_cone = (w.z() - w.length() * half_angle.sin()).abs();
                assert!(off_cone < 1e-8, "{shortfall}: off the cone by {off_cone}");
            }
        }
    }

    #[test]
    fn plane_cylinder_perpendicular() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();

        // Horizontal plane at z=3 -- produces a circle at height 3.
        let curves = intersect_plane_cylinder(&cyl, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        assert!(!curves.is_empty(), "should find intersection curve");
        assert!(
            curves[0].points.len() > 10,
            "should have many sample points"
        );

        let tol = Tolerance::loose();
        for pt in &curves[0].points {
            assert!(
                tol.approx_eq(pt.point.z(), 3.0),
                "z should be ~3.0, got {}",
                pt.point.z()
            );
            let r = pt.point.x().hypot(pt.point.y());
            assert!(tol.approx_eq(r, 2.0), "radius should be ~2.0, got {r}");
        }
    }

    #[test]
    fn plane_sphere_equator() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();

        let curves = intersect_plane_sphere(&sphere, Vec3::new(0.0, 0.0, 1.0), 0.0).unwrap();
        assert!(!curves.is_empty());

        let tol = Tolerance::loose();
        for pt in &curves[0].points {
            assert!(
                tol.approx_eq(pt.point.z(), 0.0),
                "z should be ~0, got {}",
                pt.point.z()
            );
            let r = pt.point.x().hypot(pt.point.y());
            assert!(tol.approx_eq(r, 3.0), "radius should be ~3.0, got {r}");
        }
    }

    #[test]
    fn plane_sphere_no_intersection() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();

        let curves = intersect_plane_sphere(&sphere, Vec3::new(0.0, 0.0, 1.0), 5.0).unwrap();
        assert!(curves.is_empty());
    }

    #[test]
    fn plane_cone_cross_section() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();

        let curves = intersect_plane_cone(&cone, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        assert!(!curves.is_empty(), "should find intersection with cone");
    }

    /// The 1u gridfinity spacer lip fuse corner (#1570): the body's lip
    /// recess cone (45 deg, opening downward) meets the tool's lip cone
    /// (45 deg, opening upward) with axes offset 0.25mm in x and y. Equal
    /// half-angle tangents put the whole intersection on the radical plane,
    /// so the section is one exact ellipse; the marcher shredded this into
    /// ~64 closed micro-loops per pair.
    #[test]
    fn offset_parallel_equal_angle_cones_give_one_exact_ellipse() {
        let c1 = ConicalSurface::new(
            Point3::new(
                -16.999_999_999_999_975,
                -16.999_999_999_999_975,
                5.849_999_999_999_951,
            ),
            Vec3::new(0.0, 0.0, -1.0),
            0.785_398_163_397_433_5,
        )
        .unwrap();
        let c2 = ConicalSurface::new(
            Point3::new(
                -16.750_000_000_000_036,
                -16.750_000_000_000_018,
                0.749_999_999_999_881,
            ),
            Vec3::new(0.0, 0.0, 1.0),
            0.785_398_163_397_467_6,
        )
        .unwrap();

        let curves = exact_cone_cone(&c1, &c2)
            .unwrap()
            .expect("offset parallel equal-angle cones must take the radical-plane path");
        assert_eq!(curves.len(), 1, "expected exactly one section conic");
        assert!(
            matches!(curves[0], ExactIntersectionCurve::Ellipse(_)),
            "expected an ellipse section, got {:?}",
            curves[0]
        );
        let ExactIntersectionCurve::Ellipse(ellipse) = &curves[0] else {
            return;
        };

        // Every sample must lie on BOTH cones: distance to the axis equals
        // tan(half_angle) times the axial distance from the apex, on the
        // real nappe of each.
        for i in 0..16 {
            let p = crate::traits::ParametricCurve::evaluate(ellipse, TAU * f64::from(i) / 16.0);
            for (cone, label) in [(&c1, "c1"), (&c2, "c2")] {
                let rel = p - cone.apex();
                let rel_v = Vec3::new(rel.x(), rel.y(), rel.z());
                let axial = rel_v.dot(cone.axis());
                let radial = (rel_v - cone.axis() * axial).length();
                assert!(
                    axial > 0.0,
                    "{label}: sample on phantom nappe (axial {axial})"
                );
                let expect = cone.half_angle().tan() * axial;
                assert!(
                    (radial - expect).abs() < 1e-9,
                    "{label}: sample off surface by {}",
                    (radial - expect).abs()
                );
            }
        }
    }

    /// Opposed cones whose real nappes occupy disjoint half-spaces share a
    /// radical-plane conic only on the phantom nappe — the exact path must
    /// report a definitive empty intersection, not defer to the marcher.
    #[test]
    fn offset_parallel_cones_opening_apart_have_no_real_intersection() {
        let c1 = ConicalSurface::new(
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let c2 = ConicalSurface::new(
            Point3::new(0.25, 0.25, 20.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let curves = exact_cone_cone(&c1, &c2)
            .unwrap()
            .expect("radical-plane path");
        assert!(curves.is_empty(), "disjoint nappes must yield no curves");
    }

    /// Unequal half-angles keep a quadratic term in the pencil — no plane
    /// reduction exists, so the exact path must defer to the marcher.
    #[test]
    fn offset_parallel_cones_with_unequal_angles_defer() {
        let c1 = ConicalSurface::new(
            Point3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let c2 = ConicalSurface::new(Point3::new(0.25, 0.25, 0.5), Vec3::new(0.0, 0.0, 1.0), 0.6)
            .unwrap();
        assert!(exact_cone_cone(&c1, &c2).unwrap().is_none());
    }

    /// A 45 degree cone and a radius 0.1 tube tilted 40 degrees that pierces
    /// both nappes, so the ruling path declines and the general marcher runs:
    /// the tube's axis meets the near nappe at (0.1, -1.368, 1.370).
    fn cone_and_tilted_tube() -> (ConicalSurface, CylindricalSurface) {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let (s, c) = 40.0_f64.to_radians().sin_cos();
        let tube =
            CylindricalSurface::new(Point3::new(0.1, 0.0, 3.0), Vec3::new(0.0, s, c), 0.1).unwrap();
        (cone, tube)
    }

    #[test]
    fn marcher_keeps_to_its_region() {
        let (cone, tube) = cone_and_tilted_tube();
        let run = |region: Option<Aabb3>| {
            let (a, b) = (
                AnalyticSurface::Cone(&cone),
                AnalyticSurface::Cylinder(&tube),
            );
            let (va, vb) = (Some((0.5, 4.0)), Some((-5.0, 5.0)));
            match region {
                Some(r) => intersect_analytic_analytic_in_region(a, b, 32, va, vb, r),
                None => intersect_analytic_analytic_bounded(a, b, 32, va, vb),
            }
            .unwrap()
        };
        assert!(!run(None).is_empty());
        let near = Aabb3 {
            min: Point3::new(-0.5, -2.0, 0.8),
            max: Point3::new(0.7, -0.7, 2.0),
        };
        let curves = run(Some(near));
        assert!(!curves.is_empty(), "the loop through the region is kept");
        // The region's margin (two grid cells) and one marching step past it.
        let reach = near.expanded(0.6);
        for curve in &curves {
            assert!(curve.points.iter().all(|p| reach.contains_point(p.point)));
        }
        let away = Aabb3 {
            min: Point3::new(5.0, 5.0, 5.0),
            max: Point3::new(6.0, 6.0, 6.0),
        };
        assert!(run(Some(away)).is_empty(), "nothing is marched outside it");
    }

    #[test]
    fn coaxial_cones_cross_at_single_circle() {
        // Two coaxial truncated cones (outer base r10->top r8, inner r9->r8
        // over height 10) cross where their radii match: z=10, r=8. The
        // intersection must be ONE clean circle, not the dozens of degenerate
        // micro-curves the general marcher produces at near-tangency.
        let outer = ConicalSurface::new(
            Point3::new(0.0, 0.0, 50.0),
            Vec3::new(0.0, 0.0, -1.0),
            5.0_f64.atan(),
        )
        .unwrap();
        let inner = ConicalSurface::new(
            Point3::new(0.0, 0.0, 90.0),
            Vec3::new(0.0, 0.0, -1.0),
            10.0_f64.atan(),
        )
        .unwrap();

        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Cone(&outer),
            AnalyticSurface::Cone(&inner),
            32,
            None,
            None,
        )
        .unwrap();

        assert_eq!(
            curves.len(),
            1,
            "coaxial cones crossing at one circle must yield exactly one curve, got {}",
            curves.len()
        );
        for p in &curves[0].points {
            let r = p.point.x().hypot(p.point.y());
            assert!(
                (p.point.z() - 10.0).abs() < 1e-6 && (r - 8.0).abs() < 1e-6,
                "intersection point off the expected z=10,r=8 circle: {:?}",
                p.point
            );
        }
    }

    #[test]
    fn plane_torus_cross_section() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 5.0, 1.0).unwrap();

        let curves = intersect_plane_torus(&torus, Vec3::new(0.0, 0.0, 1.0), 0.0).unwrap();
        assert!(
            !curves.is_empty(),
            "should find intersection curves with torus"
        );
    }

    /// A plane resting on a torus's tube touches it along one circle of the
    /// torus's own radius, ring or spindle, at the tube's top or bottom.
    #[test]
    fn plane_tangent_to_a_tube_touches_it_along_one_circle() {
        for (major, minor) in [(5.0, 1.0), (0.1, 2.45)] {
            let torus = ToroidalSurface::new(Point3::new(1.0, 2.0, 3.0), major, minor).unwrap();
            for z in [3.0 - minor, 3.0 + minor] {
                let exact = exact_plane_analytic(
                    AnalyticSurface::Torus(&torus),
                    Vec3::new(0.0, 0.0, 1.0),
                    z,
                )
                .unwrap();
                assert_eq!(exact.len(), 1, "R={major} r={minor} z={z}");
                let circle = match &exact[0] {
                    ExactIntersectionCurve::Circle(c) => Some(c),
                    _ => None,
                };
                let c = circle.expect("a circle");
                assert!((c.radius() - major).abs() < 1e-12);
                assert!((c.center() - Point3::new(1.0, 2.0, z)).length() < 1e-12);

                let sampled = intersect_plane_torus(&torus, Vec3::new(0.0, 0.0, 1.0), z).unwrap();
                assert_eq!(sampled.len(), 1, "R={major} r={minor} z={z}");
            }
            // A tilted plane cuts no single circle of radius R.
            let single = |normal: Vec3, d: f64| {
                matches!(
                    exact_plane_analytic(AnalyticSurface::Torus(&torus), normal, d)
                        .unwrap()
                        .as_slice(),
                    [ExactIntersectionCurve::Circle(_)]
                )
            };
            assert!(!single(Vec3::new(1e-6, 0.0, 1.0), 3.0 + minor));
        }
    }

    /// Tangency is read from the plane's distance to the tube's top, so it
    /// survives rounding on a large torus, and a plane further inside than
    /// the linear tolerance still cuts its two circles.
    #[test]
    fn plane_tangent_to_a_large_tube_is_read_through_rounding() {
        let torus = ToroidalSurface::new(Point3::new(0.3, -0.7, 3.0), 200.0, 100.1).unwrap();
        let level = Vec3::new(0.0, 0.0, 1.0);
        let curves =
            |d: f64| exact_plane_analytic(AnalyticSurface::Torus(&torus), level, d).unwrap();
        assert!(matches!(
            curves(3.0 + 100.1).as_slice(),
            [ExactIntersectionCurve::Circle(_)]
        ));
        let inside = curves(3.0 + 100.1 - 4e-7);
        assert!(!matches!(
            inside.as_slice(),
            [ExactIntersectionCurve::Circle(_)]
        ));
    }

    /// Signed distance of a point to a z-axis torus centred at the origin:
    /// `sqrt((sqrt(x^2+y^2) - R)^2 + z^2) - r`.
    fn torus_implicit(p: Point3, major: f64, minor: f64) -> f64 {
        let rho = p.x().hypot(p.y());
        ((rho - major).hypot(p.z())) - minor
    }

    /// The gridfinity lightweight base's failing corner, reduced: a cavity
    /// corner-round cone (apex below the floor, 45 deg, axis +z) crossed by a
    /// parallel-axis boss cylinder. The general marcher returned ~49 overlapping
    /// partial traces of one curve here; the algebraic path must return exactly
    /// the two branches, each ON both surfaces and inside the cone's v-hint.
    #[test]
    fn oblique_cone_cylinder_traces_curves_on_both() {
        use crate::traits::ParametricCurve;
        // A pointed cone opening down from (0, 0, 3), radius half the depth,
        // and a rod along y through (x0, ., 1): one loop through the wall
        // when the rod pokes out, two when every ruling meets the cone.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0_f64.atan(),
        )
        .unwrap();
        for (x0, loops) in [(0.5, 1), (0.0, 2)] {
            let cyl =
                CylindricalSurface::new(Point3::new(x0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 0.6)
                    .unwrap();
            for cone_first in [true, false] {
                let (a, b) = if cone_first {
                    (
                        AnalyticSurface::Cone(&cone),
                        AnalyticSurface::Cylinder(&cyl),
                    )
                } else {
                    (
                        AnalyticSurface::Cylinder(&cyl),
                        AnalyticSurface::Cone(&cone),
                    )
                };
                let curves = intersect_analytic_analytic(a, b, 32).unwrap();
                assert_eq!(curves.len(), loops, "x0 {x0}: loops");
                for c in &curves {
                    let (t0, t1) = c.curve.domain();
                    for k in 0..=64 {
                        let t = (t1 - t0).madd(f64::from(k) / 64.0, t0);
                        let p = ParametricCurve::evaluate(&c.curve, t);
                        // A cubic through the ruling samples, bent most at the
                        // loop's branch points.
                        let rod = (p.x() - x0).hypot(p.z() - 1.0);
                        assert!(
                            (rod - 0.6).abs() < 1e-4,
                            "x0 {x0}: off the rod by {}",
                            rod - 0.6
                        );
                        let cone_r = p.x().hypot(p.y());
                        assert!(
                            (cone_r - 0.5 * (3.0 - p.z())).abs() < 1e-4,
                            "x0 {x0}: off the cone at {p:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_rod_through_a_rings_tube_traces_four_loops() {
        use crate::traits::ParametricCurve;
        let ring = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        // Along y through (0.5, ., 0.3): every ruling enters and leaves the
        // tube on either side of the hole, four roots, four loops.
        let rod =
            CylindricalSurface::new(Point3::new(0.5, 0.0, 0.3), Vec3::new(0.0, 1.0, 0.0), 0.6)
                .unwrap();
        let curves = ruling_torus_cylinder(&ring, &rod, true).unwrap();
        assert_eq!(curves.len(), 4);
        for c in &curves {
            let (t0, t1) = c.curve.domain();
            for k in 0..=64 {
                let p =
                    ParametricCurve::evaluate(&c.curve, (t1 - t0).madd(f64::from(k) / 64.0, t0));
                let on_rod = (p.x() - 0.5).hypot(p.z() - 0.3) - 0.6;
                let on_ring = (p.x().hypot(p.y()) - 4.0).hypot(p.z()) - 1.5;
                assert!(
                    on_rod.abs() < 1e-4 && on_ring.abs() < 1e-4,
                    "off by {on_rod}, {on_ring}"
                );
            }
        }
        // Higher, the rod's top rulings pass over the tube: the count varies.
        let high =
            CylindricalSurface::new(Point3::new(0.5, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 0.6)
                .unwrap();
        assert!(ruling_torus_cylinder(&ring, &high, true).is_none());
        // Its top clears the tube only between two sampled rulings.
        let grazing =
            CylindricalSurface::new(Point3::new(0.5, 0.0, 0.9001), Vec3::new(0.0, 1.0, 0.0), 0.6)
                .unwrap();
        assert!(ruling_torus_cylinder(&ring, &grazing, true).is_none());
        // A spindle torus's quartic also holds its inner lemon.
        let spindle = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0, 2.0).unwrap();
        let thin =
            CylindricalSurface::new(Point3::new(0.3, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), 0.2)
                .unwrap();
        assert!(ruling_torus_cylinder(&spindle, &thin, true).is_none());
    }

    #[test]
    fn a_pin_through_a_ball_traces_two_loops() {
        use crate::traits::ParametricCurve;
        let ball = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        // A pin tapering from radius 1.2 at (1, 0.5, -5) to 0.4 at 10 up:
        // radial 0.08 per unit of height.
        let half = 0.08_f64.atan();
        let apex = Point3::new(1.0, 0.5, -5.0 + 1.2 / 0.08);
        let pin = ConicalSurface::new(apex, Vec3::new(0.0, 0.0, -1.0), FRAC_PI_2 - half).unwrap();
        for cone_first in [true, false] {
            let (a, b) = if cone_first {
                (AnalyticSurface::Cone(&pin), AnalyticSurface::Sphere(&ball))
            } else {
                (AnalyticSurface::Sphere(&ball), AnalyticSurface::Cone(&pin))
            };
            let curves = intersect_analytic_analytic(a, b, 32).unwrap();
            assert_eq!(curves.len(), 2, "entry and exit loops");
            for c in &curves {
                let (t0, t1) = c.curve.domain();
                for k in 0..=64 {
                    let p = ParametricCurve::evaluate(
                        &c.curve,
                        (t1 - t0).madd(f64::from(k) / 64.0, t0),
                    );
                    let on_ball = (p - Point3::new(0.0, 0.0, 0.0)).length() - 3.0;
                    let axial = apex.z() - p.z();
                    let on_pin = (p.x() - 1.0).hypot(p.y() - 0.5) - axial * half.tan();
                    assert!(
                        on_ball.abs() < 1e-4 && on_pin.abs() < 1e-4,
                        "off by {on_ball}, {on_pin}"
                    );
                }
            }
        }
        // On the ball's axis the loops are circles, which defer. A pin whose
        // apex is inside the ball leaves it once along every generator, and
        // one only partly through the ball meets it over the generators that
        // reach it: one loop each. One that opens away meets it not at all.
        let coaxial =
            ConicalSurface::new(Point3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0), 1.4)
                .unwrap();
        assert!(ruling_cone_sphere(&coaxial, &ball, true).is_none());
        let aside = ConicalSurface::new(
            Point3::new(2.8, 0.0, 10.0),
            Vec3::new(0.0, 0.0, -1.0),
            FRAC_PI_2 - half,
        )
        .unwrap();
        assert_eq!(ruling_cone_sphere(&aside, &ball, true).unwrap().len(), 1);
        let holding = ConicalSurface::new(
            Point3::new(1.0, 0.5, 1.0),
            Vec3::new(0.0, 0.0, -1.0),
            FRAC_PI_2 - half,
        )
        .unwrap();
        assert_eq!(ruling_cone_sphere(&holding, &ball, true).unwrap().len(), 1);
        let away = ConicalSurface::new(
            Point3::new(1.0, 0.5, 10.0),
            Vec3::new(0.0, 0.0, 1.0),
            FRAC_PI_2 - half,
        )
        .unwrap();
        assert!(ruling_cone_sphere(&away, &ball, true).unwrap().is_empty());
        // Grazing: the generators that miss span about 0.002 radians,
        // narrower than a 2048-angle scan's step, and the near and far
        // roots join into one loop across them.
        let step = TAU / 2048.0;
        let grazed =
            SphericalSurface::new(Point3::new(step.cos(), step.sin(), 10.0), 9.255_250_971_8)
                .unwrap();
        let wide =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5).unwrap();
        assert_eq!(ruling_cone_sphere(&wide, &grazed, true).unwrap().len(), 1);
    }

    #[test]
    fn a_ball_beside_a_cone_meets_it_in_one_loop() {
        use crate::traits::ParametricCurve;
        // Apex 3 up, opening downward, the radius half the depth below it.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0_f64.atan(),
        )
        .unwrap();
        for (centre, radius) in [
            (Point3::new(1.0, 0.8, 1.2), 1.1),
            (Point3::new(1.5, 0.0, 0.0), 0.8),
        ] {
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let curves = ruling_cone_sphere(&cone, &ball, true).unwrap();
            assert_eq!(curves.len(), 1, "one loop for the ball at {centre:?}");
            let (t0, t1) = curves[0].curve.domain();
            for k in 0..=64 {
                let p = ParametricCurve::evaluate(
                    &curves[0].curve,
                    (t1 - t0).madd(f64::from(k) / 64.0, t0),
                );
                let on_ball = (p - centre).length() - radius;
                let on_cone = p.x().hypot(p.y()) - (3.0 - p.z()) / 2.0;
                assert!(
                    on_ball.abs() < 1e-5 && on_cone.abs() < 1e-5,
                    "ball at {centre:?}: off by {on_ball}, {on_cone}"
                );
            }
        }
        // A ball the nappe's generators miss, and one on the apex.
        let clear = SphericalSurface::new(Point3::new(4.0, 0.0, 0.0), 0.5).unwrap();
        assert!(ruling_cone_sphere(&cone, &clear, true).unwrap().is_empty());
        let on_apex = SphericalSurface::new(Point3::new(0.6, 0.0, 3.8), 1.0).unwrap();
        assert!(ruling_cone_sphere(&cone, &on_apex, true).is_none());
    }

    #[test]
    fn a_ball_holding_a_cones_apex_meets_it_in_one_loop() {
        use crate::traits::ParametricCurve;
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0_f64.atan(),
        )
        .unwrap();
        // Off the axis, and barely holding the apex: the generators pointing
        // away from the centre leave it just past the apex.
        for (centre, radius) in [
            (Point3::new(0.5, 0.0, 2.5), 2.0),
            (Point3::new(-0.4, 0.3, 2.0), 1.5),
            (Point3::new(0.0, 0.8, 3.0), 0.8001),
            (Point3::new(0.0, 0.8, 3.0), 0.800_001),
        ] {
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let curves = ruling_cone_sphere(&cone, &ball, true).unwrap();
            assert_eq!(curves.len(), 1, "one loop for the ball at {centre:?}");
            let (t0, t1) = curves[0].curve.domain();
            for k in 0..=4096 {
                let p = ParametricCurve::evaluate(
                    &curves[0].curve,
                    (t1 - t0).madd(f64::from(k) / 4096.0, t0),
                );
                let on_ball = (p - centre).length() - radius;
                let on_cone = p.x().hypot(p.y()) - (3.0 - p.z()) / 2.0;
                assert!(
                    on_ball.abs() < 1e-5 && on_cone.abs() < 1e-5 && p.z() < 3.0,
                    "ball at {centre:?}: off by {on_ball}, {on_cone} at {p:?}"
                );
            }
        }
    }

    /// A pin whose tip just pierces a rod's wall meets the rulings within
    /// 2.2e-4 of the wall's top, between two of the 128 samples. The
    /// sample nearest them misses the pin, its discriminant -5.9e-8 within
    /// the tolerance alone: a loop traced from it would lie 3e-5 off the
    /// pin. The window lies between two scan points too, and the rulings
    /// leave the pin to the marcher.
    #[test]
    fn a_pin_tip_between_two_samples_is_not_traced_from_a_miss() {
        use std::f64::consts::{FRAC_PI_2, PI};
        let slope = 0.04375_f64;
        let pin = ConicalSurface::new(
            Point3::new(0.0, 5.005, 10.0),
            Vec3::new(0.0, -1.0, 0.0),
            (1.0 / slope).atan(),
        )
        .unwrap();
        let turn = FRAC_PI_2 + 5e-5 - PI / 128.0;
        let rod = CylindricalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            Vec3::new(turn.cos(), turn.sin(), 0.0),
        )
        .unwrap();
        assert!(ruling_cone_cylinder(&pin, &rod, true).is_none());
    }

    /// A taper pin through a rod, the rod turned about its axis: the pin's
    /// narrow end meets fewer than 8 of the rod's 128 sampled rulings in
    /// some turns, and the rulings still trace both of its loops there.
    #[test]
    fn a_taper_pin_through_a_turned_rod_traces_both_loops() {
        use crate::traits::ParametricCurve;
        let slope = 0.7_f64 / 16.0;
        let half = (1.0 / slope).atan();
        for flip in [false, true] {
            let (apex_y, axis_y) = if flip {
                (-8.0 - 0.8 / slope, 1.0)
            } else {
                (-8.0 + 1.5 / slope, -1.0)
            };
            let pin = ConicalSurface::new(
                Point3::new(0.0, apex_y, 10.0),
                Vec3::new(0.0, axis_y, 0.0),
                half,
            )
            .unwrap();
            for spin in [0.0_f64, 0.3, 0.9, 1.4, 2.0, 2.6, 3.0, 3.6] {
                let rod = CylindricalSurface::with_ref_dir(
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, 1.0),
                    5.0,
                    Vec3::new(spin.cos(), spin.sin(), 0.0),
                )
                .unwrap();
                let curves = ruling_cone_cylinder(&pin, &rod, true);
                assert!(curves.is_some(), "spin {spin} flip {flip}: no trace");
                let curves = curves.unwrap();
                assert_eq!(curves.len(), 2, "spin {spin} flip {flip}: loops");
                for c in &curves {
                    let (t0, t1) = c.curve.domain();
                    for k in 0..=1000 {
                        let p = ParametricCurve::evaluate(
                            &c.curve,
                            (t1 - t0).madd(f64::from(k) / 1000.0, t0),
                        );
                        let on_rod = p.x().hypot(p.y()) - 5.0;
                        let on_pin = p.x().hypot(p.z() - 10.0) - slope * (p.y() - apex_y).abs();
                        assert!(
                            on_rod.abs() < 1e-7 && on_pin.abs() < 1e-7,
                            "spin {spin} flip {flip}: off by {on_rod:.2e}, {on_pin:.2e}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn oblique_cone_cylinder_defers_where_rulings_cannot_trace_it() {
        let t = 2.0_f64.atan();
        let cone =
            ConicalSurface::new(Point3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 0.0, -1.0), t).unwrap();
        // A rod through the apex meets the far nappe.
        let through_apex =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 3.0), Vec3::new(0.0, 1.0, 0.0), 0.6)
                .unwrap();
        assert!(ruling_cone_cylinder(&cone, &through_apex, true).is_none());
        // A rod along a generator meets each ruling once.
        let generator = Vec3::new(t.cos(), 0.0, -t.sin());
        let along = CylindricalSurface::new(Point3::new(0.0, 0.3, 0.0), generator, 0.2).unwrap();
        assert!(ruling_cone_cylinder(&cone, &along, true).is_none());
        // A pin's tip just through a tube's wall: the tube's rulings that
        // meet it span a window narrower than the sampling.
        let pin =
            ConicalSurface::new(Point3::new(20.5, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0), t).unwrap();
        let tube =
            CylindricalSurface::new(Point3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, 1.0), 20.0)
                .unwrap();
        assert!(ruling_cone_cylinder(&pin, &tube, true).is_none());
    }

    #[test]
    fn parallel_cone_cylinder_gives_two_exact_branches() {
        use crate::traits::ParametricCurve;
        let cone = ConicalSurface::new(
            Point3::new(-5.45, -36.55, -4.85),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let cyl = CylindricalSurface::new(
            Point3::new(-8.0, -34.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
            4.45,
        )
        .unwrap();
        // The cone face spans z in [-3.8, -3.0]; v = (z - apex_z) / sin(45 deg).
        let v_hint = (1.484_924_240_492_058, 2.616_295_090_390_43);
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&cyl),
            32,
            Some(v_hint),
            Some((0.0, 2.5)),
        )
        .unwrap();

        assert_eq!(curves.len(), 2, "expected exactly the two branches");
        for c in &curves {
            let (t0, t1) = c.curve.domain();
            for k in 0..=32 {
                let t = (t1 - t0).madd(f64::from(k) / 32.0, t0);
                let p = ParametricCurve::evaluate(&c.curve, t);
                // On the cylinder: radial distance from its axis is the radius.
                let radial = ((p.x() + 8.0).powi(2) + (p.y() + 34.0).powi(2)).sqrt();
                assert!((radial - 4.45).abs() < 1e-6, "off cylinder: {radial}");
                // On the cone: radial distance from its axis is z - apex_z.
                let cone_r = ((p.x() + 5.45).powi(2) + (p.y() + 36.55).powi(2)).sqrt();
                assert!((cone_r - (p.z() + 4.85)).abs() < 1e-6, "off cone at {p:?}");
                // Inside the cone face's own v-window (the hint is respected).
                assert!(p.z() >= -3.8 - 1e-9 && p.z() <= -3.0 + 1e-9, "z={}", p.z());
            }
        }
    }

    #[test]
    fn parallel_rod_through_a_cones_wall_closes_one_loop() {
        use crate::traits::ParametricCurve;
        // Apex 3 up, opening downward, the radius half the depth below it.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0_f64.atan(),
        )
        .unwrap();
        // Beside the axis: each ruling meets the nappe once.
        for (x, y) in [(0.0, 1.3), (1.2, 0.5)] {
            let rod =
                CylindricalSurface::new(Point3::new(x, y, -10.0), Vec3::new(0.0, 0.0, 1.0), 0.6)
                    .unwrap();
            let curves = algebraic_parallel_cone_cylinder(&cone, &rod, None, None)
                .unwrap()
                .unwrap();
            assert_eq!(curves.len(), 1, "one closed loop at ({x}, {y})");
            let (t0, t1) = curves[0].curve.domain();
            let (first, last) = (
                ParametricCurve::evaluate(&curves[0].curve, t0),
                ParametricCurve::evaluate(&curves[0].curve, t1),
            );
            assert!((first - last).length() < 1e-9, "open at ({x}, {y})");
            for k in 0..=64 {
                let p = ParametricCurve::evaluate(
                    &curves[0].curve,
                    (t1 - t0).madd(f64::from(k) / 64.0, t0),
                );
                let on_rod = (p.x() - x).hypot(p.y() - y) - 0.6;
                let on_cone = p.x().hypot(p.y()) - (3.0 - p.z()) / 2.0;
                assert!(
                    on_rod.abs() < 1e-5 && on_cone.abs() < 1e-5,
                    "({x}, {y}): off by {on_rod}, {on_cone}"
                );
            }
        }
        // Around the axis, and beside it by less than the bend the rulings
        // resolve: two branches.
        for (x, y) in [(0.3, 0.2), (0.65, 0.0)] {
            let rod =
                CylindricalSurface::new(Point3::new(x, y, -10.0), Vec3::new(0.0, 0.0, 1.0), 0.6)
                    .unwrap();
            let curves = algebraic_parallel_cone_cylinder(&cone, &rod, None, None)
                .unwrap()
                .unwrap();
            assert_eq!(curves.len(), 2, "two branches at ({x}, {y})");
        }
    }

    /// A frustum (radius 72 to 50 over 84) and a smaller one standing beside
    /// its axis on the same base (radius 10 to 5 over 44, axis 70 away): the
    /// smaller one notches the base rim, and the section is an arch from the
    /// rim up to where the two circles at one height touch, in two branches
    /// that meet there. Each lies on both cones, between its samples too.
    #[test]
    fn parallel_cones_give_two_exact_branches() {
        use crate::traits::ParametricCurve;
        let frustum = |x: f64, r_big: f64, r_small: f64, h: f64| {
            let apex = h + r_small * h / (r_big - r_small);
            let cone = ConicalSurface::new(
                Point3::new(x, 0.0, apex),
                Vec3::new(0.0, 0.0, -1.0),
                apex.atan2(r_big),
            )
            .unwrap();
            // `v` at height `z`, from the apex down the wall.
            let v = move |z: f64| (apex - z) / apex.atan2(r_big).sin();
            (cone, v(h), v(0.0))
        };
        let (big, big_top, big_base) = frustum(0.0, 72.0, 50.0, 84.0);
        for (x, flip) in [(-70.0_f64, false), (70.0, true)] {
            let (small, small_top, small_base) = frustum(x, 10.0, 5.0, 44.0);
            let (first, second) = if flip {
                (AnalyticSurface::Cone(&small), AnalyticSurface::Cone(&big))
            } else {
                (AnalyticSurface::Cone(&big), AnalyticSurface::Cone(&small))
            };
            let (hint_first, hint_second) = if flip {
                ((small_top, small_base), (big_top, big_base))
            } else {
                ((big_top, big_base), (small_top, small_base))
            };
            let curves = intersect_analytic_analytic_bounded(
                first,
                second,
                32,
                Some(hint_first),
                Some(hint_second),
            )
            .unwrap();
            assert_eq!(curves.len(), 2, "x = {x}: the two branches");
            // The circles touch where 72 - 22 z / 84 + 10 - 5 z / 44 = 70.
            let touch = 12.0 / (22.0 / 84.0 + 5.0 / 44.0);
            let mut tops = Vec::new();
            for c in &curves {
                let (t0, t1) = c.curve.domain();
                let ends = [
                    ParametricCurve::evaluate(&c.curve, t0),
                    ParametricCurve::evaluate(&c.curve, t1),
                ];
                let (low, high) = if ends[0].z() < ends[1].z() {
                    (ends[0], ends[1])
                } else {
                    (ends[1], ends[0])
                };
                assert!(low.z().abs() < 1e-9, "x = {x}: starts at z = {}", low.z());
                assert!((high.z() - touch).abs() < 1e-9, "x = {x}: tops at {high:?}");
                tops.push(high);
                for k in 0..=256 {
                    let p = ParametricCurve::evaluate(
                        &c.curve,
                        (t1 - t0).madd(f64::from(k) / 256.0, t0),
                    );
                    let on_big = p.x().hypot(p.y()) - (72.0 - 22.0 * p.z() / 84.0);
                    let on_small = (p.x() - x).hypot(p.y()) - (10.0 - 5.0 * p.z() / 44.0);
                    assert!(
                        on_big.abs() < 1e-6 && on_small.abs() < 1e-6,
                        "x = {x}: {p:?} off by {on_big}, {on_small}"
                    );
                }
            }
            assert!((tops[0] - tops[1]).length() < 1e-9, "x = {x}: {tops:?}");
        }
    }

    /// Apices 1 apart, radii `z` and `1.001 z` at the height `z`: the circles
    /// meet from `z = 1 / 2.001` up to `z = 1000`, and the two branches, fitted
    /// over that long span, lie on both cones between their points and meet
    /// at both ends.
    #[test]
    fn parallel_cones_of_nearly_equal_slope_trace_on_both() {
        use crate::traits::ParametricCurve;
        let up = Vec3::new(0.0, 0.0, 1.0);
        let first =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, std::f64::consts::FRAC_PI_4)
                .unwrap();
        let second =
            ConicalSurface::new(Point3::new(1.0, 0.0, 0.0), up, (1.0 / 1.001_f64).atan()).unwrap();
        let curves = algebraic_parallel_cone_cone(&first, &second, None, None).unwrap();
        assert_eq!(curves.len(), 2, "the two branches");
        let mut ends = Vec::new();
        for c in &curves {
            let (t0, t1) = c.curve.domain();
            ends.push(ParametricCurve::evaluate(&c.curve, t0));
            ends.push(ParametricCurve::evaluate(&c.curve, t1));
            for k in 0..=4096 {
                let p =
                    ParametricCurve::evaluate(&c.curve, (t1 - t0).madd(f64::from(k) / 4096.0, t0));
                let on_first = p.x().hypot(p.y()) - p.z();
                let on_second = (p.x() - 1.0).hypot(p.y()) - 1.001 * p.z();
                assert!(
                    on_first.abs() < 1e-7 && on_second.abs() < 1e-7,
                    "{p:?} off by {on_first}, {on_second}"
                );
            }
        }
        let (low, high) = (1.0 / 2.001, 1000.0);
        for (k, z) in [(0, low), (1, high)] {
            let (a, b) = (ends[k], ends[2 + k]);
            assert!(
                (a.z() - z).abs() < 1e-6 * z,
                "branch end at {a:?}, not z = {z}"
            );
            assert!((a - b).length() < 1e-9, "branches part at {a:?} and {b:?}");
        }
    }

    /// Two cones of one axis direction whose slopes differ by about `5e-13`:
    /// the narrower circle starts inside the wider one, a unit clear of its
    /// rim, and reaches it only near `v = 2e12`. Within a span out to `4e12`
    /// the cones meet, so the path does not call them apart.
    #[test]
    fn nearly_equal_slopes_meet_far_out() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let wide = ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, std::f64::consts::FRAC_PI_4)
            .unwrap();
        let (sin1, cos1) = std::f64::consts::FRAC_PI_4.sin_cos();
        // `rho2 - rho1 = (b - cos1) v + a`, with `b - cos1 = 5e-13` and
        // `a = -(d + 1)` for axes `d = 1` apart.
        let slope = (cos1 + 5e-13) / sin1;
        let along = 2.0 / slope;
        let narrow =
            ConicalSurface::new(Point3::new(1.0, 0.0, along), up, (1.0 / slope).atan()).unwrap();
        let curves = algebraic_parallel_cone_cone(&wide, &narrow, Some((0.0, 4e12)), None);
        assert!(
            !curves.as_ref().is_some_and(Vec::is_empty),
            "called apart: {:?}",
            curves.map(|c| c.len())
        );
    }

    /// Two cones of one axis direction, a unit apart, bounded on their
    /// negative nappes: both hold `(0.125, sqrt(15)/8, -0.5)`, at `v` about
    /// -0.707 on the first and -1.118 on the second. The path traces positive
    /// nappes only, so it leaves the pair to the marcher and does not call the
    /// cones apart.
    #[test]
    fn parallel_cones_bounded_on_their_negative_nappes_go_to_the_marcher() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let wide = ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, std::f64::consts::FRAC_PI_4)
            .unwrap();
        let narrow = ConicalSurface::new(Point3::new(1.0, 0.0, 0.0), up, 0.5_f64.atan()).unwrap();
        let shared = Point3::new(0.125, 15.0_f64.sqrt() / 8.0, -0.5);
        for cone in [&wide, &narrow] {
            assert!(off_cone(cone, shared).0 < 1e-12);
        }
        let (r1, r2) = (std::f64::consts::SQRT_2, 5.0_f64.sqrt());
        let curves = algebraic_parallel_cone_cone(
            &wide,
            &narrow,
            Some((-r1, -r1 / 3.0)),
            Some((-r2, -r2 / 3.0)),
        );
        assert!(curves.is_none(), "{:?} curves", curves.map(|c| c.len()));
    }

    /// Axes `1e-5` radians apart, which a dot product within `1e-10` of one
    /// read as parallel: the parallel cone-cone and cone-cylinder paths both
    /// decline them.
    #[test]
    fn nearly_parallel_axes_take_neither_parallel_path() {
        let tilt = 1e-5_f64;
        let down = Vec3::new(0.0, 0.0, -1.0);
        let tilted = Vec3::new(tilt.sin(), 0.0, -tilt.cos());
        let big = ConicalSurface::new(
            Point3::new(0.0, 0.0, 274.909_090_909),
            down,
            274.909_090_909_f64.atan2(72.0),
        )
        .unwrap();
        let small =
            ConicalSurface::new(Point3::new(-70.0, 0.0, 88.0), tilted, 88.0_f64.atan2(10.0))
                .unwrap();
        assert!(algebraic_parallel_cone_cone(&big, &small, None, None).is_none());
        let cone = ConicalSurface::new(Point3::new(0.0, 0.0, 3.0), down, 2.0_f64.atan()).unwrap();
        let rod = CylindricalSurface::new(Point3::new(1.2, 0.5, -10.0), tilted, 0.6).unwrap();
        assert!(
            algebraic_parallel_cone_cylinder(&cone, &rod, None, None)
                .unwrap()
                .is_none()
        );
    }

    /// A coaxial pair has no radical line; the algebraic path must defer rather
    /// than divide by a zero axis separation.
    #[test]
    fn coaxial_cone_cylinder_defers_to_other_paths() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0)
                .unwrap();
        assert!(
            algebraic_parallel_cone_cylinder(&cone, &cyl, None, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn oblique_cone_cylinder_defers_to_other_paths() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(3.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 0.0), 1.0)
                .unwrap();
        assert!(
            algebraic_parallel_cone_cylinder(&cone, &cyl, None, None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn plane_torus_lobe_closes_and_stays_on_surface() {
        use crate::traits::ParametricCurve;
        let (major, minor) = (10.0, 3.0);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();

        // The census cutting planes (y=-4, x=6) each cut the +x and -x tube lobes
        // in a CLOSED oval. The greedy marcher stops one grid step short of
        // closing; the wrap-close must make every fitted lobe close exactly.
        for (n, d) in [
            (Vec3::new(0.0, -1.0, 0.0), 4.0),  // y = -4
            (Vec3::new(-1.0, 0.0, 0.0), -6.0), // x = 6
            (Vec3::new(0.0, 0.0, 1.0), 0.0),   // z = 0 -> two concentric circles
        ] {
            let curves = intersect_plane_torus(&torus, n, d).unwrap();
            assert!(!curves.is_empty(), "plane n={n:?} d={d} found no curves");
            for c in &curves {
                let p0 = ParametricCurve::evaluate(&c.curve, 0.0);
                let p1 = ParametricCurve::evaluate(&c.curve, 1.0);
                assert!(
                    (p0 - p1).length() < 1e-7,
                    "lobe not closed: gap={} (n={n:?} d={d})",
                    (p0 - p1).length()
                );
                // Every fitted sample stays on the torus (shape-preserving).
                for k in 0..=64 {
                    let t = f64::from(k) / 64.0;
                    let p = ParametricCurve::evaluate(&c.curve, t);
                    assert!(
                        torus_implicit(p, major, minor).abs() < 1e-2,
                        "off-surface point {p:?} implicit={}",
                        torus_implicit(p, major, minor)
                    );
                }
            }
        }
    }

    /// A wall 0.4 outside the inner equator turns its loops tightly there.
    /// Between its points, each fitted loop stays on the torus and on the
    /// plane within the fit's bound.
    #[test]
    fn plane_torus_fits_stay_on_the_section_where_it_turns_tightly() {
        use crate::traits::ParametricCurve;
        let (major, minor) = (4.0, 1.5);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
        for (n, d) in [
            (Vec3::new(0.0, 1.0, 0.0), 2.1),
            (Vec3::new(1.0, 0.0, 0.0), -1.9),
            (Vec3::new(0.0, 0.6, 0.8), 3.5),
        ] {
            let curves = intersect_plane_torus(&torus, n, d).unwrap();
            assert!(!curves.is_empty(), "plane n={n:?} d={d} found no curves");
            for c in &curves {
                for k in 0..=4000 {
                    let p = ParametricCurve::evaluate(&c.curve, f64::from(k) / 4000.0);
                    let off = ((p.x().hypot(p.y()) - major).hypot(p.z()) - minor).abs();
                    assert!(off <= 1e-8, "{off} off the torus at {p:?} (n={n:?} d={d})");
                    let off_plane = (n.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs();
                    assert!(off_plane <= 1e-12, "{off_plane} off the plane at {p:?}");
                }
            }
        }
    }

    /// The plane `z = 0`'s section of the torus of radii 4 and 1.5 read
    /// from two of its points across the axis: the line between them strays
    /// 2.5 off the torus at its middle, on the axis, where no point moves onto
    /// the section. The fit cannot check that span, and is refused.
    #[test]
    fn a_plane_torus_fit_it_cannot_check_is_refused() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let across = [Point3::new(5.5, 0.0, 0.0), Point3::new(-5.5, 0.0, 0.0)];
        assert!(fit_plane_torus_section(&torus, Vec3::new(0.0, 0.0, 1.0), 0.0, &across).is_err());
    }

    /// The fit's check stops its Newton steps once they have moved a middle
    /// by less than a quarter of the bound. A middle it then reads as within
    /// the bound is within it, and a middle farther off moves onto the same
    /// point of the section as before, so the fit keeps the same points.
    #[test]
    fn a_short_newton_check_reads_each_middle_as_the_full_one_does() {
        let torus = ToroidalSurface::new(Point3::new(1.0, -2.0, 0.5), 4.0, 1.5).unwrap();
        let (n, d) = (Vec3::new(0.0, 2.0, 0.0), 2.0 * (2.1 - 2.0));
        let tol = PLANE_TORUS_FIT * 5.5;
        for ipts in plane_torus_loops(&torus, n, d, 128) {
            for (ip, power) in ipts.iter().zip((0..10).cycle()) {
                let off = Vec3::new(0.3, -0.7, 0.2) * (1e-13 * 10f64.powi(power));
                let p = ip.point + off;
                let full = plane_torus_section_point(&torus, n, d, p).unwrap();
                let short = plane_torus_newton(&torus, n, d, p, 0.25 * tol).unwrap();
                if (short - p).length() < 0.25 * tol {
                    assert!((full - p).length() <= tol, "{:.2e}", (full - p).length());
                    assert!((full - short).length() < 1e-15);
                } else {
                    assert_eq!(full, short);
                }
            }
        }
    }

    /// Points near a plane's section of a torus move onto it, on the plane
    /// and the tube to rounding.
    #[test]
    fn plane_torus_section_points_lie_on_both_surfaces() {
        let torus = ToroidalSurface::new(Point3::new(1.0, -2.0, 0.5), 4.0, 1.5).unwrap();
        let (n, d) = (Vec3::new(0.0, 2.0, 0.0), 2.0 * (2.1 - 2.0));
        for ipts in plane_torus_loops(&torus, n, d, 128) {
            for ip in ipts.iter().step_by(5) {
                let near = ip.point + Vec3::new(1e-4, -2e-4, 3e-4);
                let on = plane_torus_section_point(&torus, n, d, near).unwrap();
                let w = on - torus.center();
                let off = ((w.x().hypot(w.y()) - 4.0).hypot(w.z()) - 1.5).abs();
                assert!(off <= 1e-14, "{off} off the torus");
                assert!(
                    (on.y() - 0.1).abs() <= 1e-14,
                    "{} off the plane",
                    on.y() - 0.1
                );
                assert!((on - near).length() < 1e-3);
            }
        }
    }

    /// A plane tangent to the inner equator (x = major - minor = 7) cuts a
    /// figure-eight that crosses itself on that equator. It comes out as its
    /// two lobes, each a loop that starts and ends on the node, one either
    /// side of the plane y = 0, never as one loop sealed across the node.
    #[test]
    fn plane_torus_inner_tangent_figure_eight_is_two_loops_through_its_node() {
        let (major, minor) = (10.0, 3.0);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
        let curves =
            intersect_plane_torus(&torus, Vec3::new(-1.0, 0.0, 0.0), -(major - minor)).unwrap();
        assert_eq!(curves.len(), 2, "one loop per lobe");
        let node = Point3::new(major - minor, 0.0, 0.0);
        let mut sides = Vec::new();
        for c in &curves {
            let (first, last) = (c.points[0].point, c.points[c.points.len() - 1].point);
            assert!((first - node).length() < 1e-6, "starts at {first:?}");
            assert_eq!(first, last, "closes on the node");
            for p in c.points.iter().map(|p| p.point) {
                assert!((p.x() - node.x()).abs() < 1e-9, "off the plane at {p:?}");
                assert!(
                    torus_implicit(p, major, minor).abs() < 1e-6,
                    "off the torus at {p:?}"
                );
            }
            let y: f64 = c.points.iter().map(|p| p.point.y()).sum();
            sides.push(y > 0.0);
            assert!(
                c.points.iter().all(|p| p.point.y() * y >= -1e-9),
                "a lobe crosses y = 0"
            );
        }
        assert_ne!(sides[0], sides[1], "the lobes lie either side of y = 0");
    }

    /// A wall just off the inner tangent has no node. Just past it the
    /// section has a gap 0.028 wide in tube angle about the inner equator,
    /// between two samples inside it, and is one loop turning either side of
    /// the gap; just short of it the section winds the tube twice. Every
    /// point lies on the wall.
    #[test]
    fn a_wall_just_off_the_inner_tangent_cuts_no_node() {
        let (major, minor) = (3.0, 1.0);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
        for (x, loops) in [(2.0001, 1), (1.9999, 2)] {
            let curves = intersect_plane_torus(&torus, Vec3::new(1.0, 0.0, 0.0), x).unwrap();
            assert_eq!(curves.len(), loops, "x = {x}");
            for c in &curves {
                let (first, last) = (c.points[0].point, c.points[c.points.len() - 1].point);
                assert_eq!(first, last, "x = {x}: the loop closes");
                for p in c.points.iter().map(|p| p.point) {
                    assert!((p.x() - x).abs() < 1e-9, "x = {x}: off the wall at {p:?}");
                    assert!(
                        torus_implicit(p, major, minor).abs() < 1e-6,
                        "x = {x}: off the torus at {p:?}"
                    );
                }
            }
        }
        // The loop past the tangent turns either side of the gap, 0.028 apart.
        let curves = intersect_plane_torus(&torus, Vec3::new(1.0, 0.0, 0.0), 2.0001).unwrap();
        let low = curves[0]
            .points
            .iter()
            .map(|p| p.point)
            .filter(|p| p.y().abs() < 1e-6)
            .map(Point3::z)
            .collect::<Vec<_>>();
        let gap = (2.0001_f64 - 2.0)
            .madd(2.0, -(2.0001_f64 - 2.0).powi(2))
            .sqrt();
        assert!(
            low.iter().any(|z| (z - gap).abs() < 1e-6)
                && low.iter().any(|z| (z + gap).abs() < 1e-6),
            "turns at {low:?}, not at z = ±{gap}"
        );
    }

    /// A plane tangent to the tube's inner side off its equator, at a saddle,
    /// cuts a figure-eight that crosses itself where it touches. Its lobes
    /// come out as two loops, each starting and ending on that node.
    #[test]
    fn a_tilted_plane_tangent_inside_the_tube_cuts_two_loops_through_its_node() {
        let (major, minor) = (3.0, 1.0);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
        for v0 in [1.7_f64, 2.0, 2.5, 4.0] {
            let normal = Vec3::new(v0.cos(), 0.0, v0.sin());
            let node = torus.evaluate(0.0, v0);
            let d = normal.dot(Vec3::new(node.x(), node.y(), node.z()));
            let curves = intersect_plane_torus(&torus, normal, d).unwrap();
            assert_eq!(curves.len(), 2, "v0 {v0}: one loop per lobe");
            for c in &curves {
                let (first, last) = (c.points[0].point, c.points[c.points.len() - 1].point);
                assert!(
                    (first - node).length() < 1e-6,
                    "v0 {v0}: starts at {first:?}, not {node:?}"
                );
                assert_eq!(first, last, "v0 {v0}: closes on the node");
                for p in c.points.iter().map(|p| p.point) {
                    let off = normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d;
                    assert!(off.abs() < 1e-9, "v0 {v0}: off the plane at {p:?}");
                    assert!(
                        torus_implicit(p, major, minor).abs() < 1e-6,
                        "v0 {v0}: off the torus at {p:?}"
                    );
                }
            }
        }
    }

    /// A wall parallel to the axis cuts one loop from the ring past the
    /// inner equator (its two turns where the branches meet), and two loops
    /// winding the tube inside it; each closes, at any scale, however near
    /// the wall runs to an equator.
    #[test]
    fn plane_torus_wall_sections_close_into_their_loops() {
        for (major, minor) in [(4.0, 1.5), (100.0, 30.0), (0.05, 0.01)] {
            let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
            for k in 1..200 {
                let (d, want) = match k.cmp(&100) {
                    // Within the inner equator: two loops winding the tube.
                    std::cmp::Ordering::Less => ((major - minor) * f64::from(k) / 100.0, 2),
                    // Past it, short of the outer equator: one loop.
                    std::cmp::Ordering::Greater => (
                        2.0f64.madd(minor * f64::from(k - 100) / 100.0, major - minor),
                        1,
                    ),
                    std::cmp::Ordering::Equal => continue,
                };
                let loops = plane_torus_loops(&torus, Vec3::new(1.0, 0.0, 0.0), d, 128);
                let closed = loops
                    .iter()
                    .filter(|l| (l[0].point - l[l.len() - 1].point).length() < 1e-12)
                    .count();
                assert_eq!(
                    (loops.len(), closed),
                    (want, want),
                    "R {major} r {minor}, wall at {d}"
                );
            }
        }
    }

    /// A plane through the centre tilted a little from the equator cuts two
    /// loops that each run all the way round the axis on a short run of `v`;
    /// their fitted curves stay on the torus.
    #[test]
    fn plane_torus_sections_round_the_axis_stay_on_the_torus() {
        let (major, minor) = (4.0, 1.5);
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), major, minor).unwrap();
        for tilt in [0.03_f64, 0.08, 0.2] {
            let normal = Vec3::new(tilt.sin(), 0.0, tilt.cos());
            let curves = intersect_plane_torus(&torus, normal, 0.0).unwrap();
            assert_eq!(curves.len(), 2, "tilt {tilt}");
            for c in &curves {
                let (t0, t1) = c.curve.domain();
                let off = (0..=400)
                    .map(|k| {
                        let p = c.curve.evaluate((t1 - t0).madd(f64::from(k) / 400.0, t0));
                        (p.x().hypot(p.y()) - major).hypot(p.z()) - minor
                    })
                    .fold(0.0_f64, |m, e| m.max(e.abs()));
                assert!(
                    off < 1e-6,
                    "tilt {tilt}: fitted section {off} off the torus"
                );
            }
        }
    }

    #[test]
    fn line_torus_box_edge_crossing_is_exact() {
        // The census box edge x=6, y=-4 (z varying) crosses the torus (R=10,r=3)
        // at z = ±sqrt(r² − (rho−R)²), rho = hypot(6,4) ≈ 7.2111 → z ≈ ±1.1055.
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
        let ts = intersect_line_torus(
            &torus,
            Point3::new(6.0, -4.0, -5.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        // Vertical line through (6,-4) meets the tube twice.
        assert_eq!(ts.len(), 2, "expected 2 crossings, got {ts:?}");
        let zs: Vec<f64> = ts.iter().map(|t| -5.0 + t).collect();
        let rho = 6.0_f64.hypot(4.0);
        let z_exp = (9.0 - (rho - 10.0).powi(2)).sqrt();
        assert!(
            (zs[0] - (-z_exp)).abs() < 1e-9,
            "z0={} exp={}",
            zs[0],
            -z_exp
        );
        assert!((zs[1] - z_exp).abs() < 1e-9, "z1={} exp={}", zs[1], z_exp);
        // Each crossing lies on the torus.
        for &t in &ts {
            let p = Point3::new(6.0, -4.0, -5.0 + t);
            let rho = p.x().hypot(p.y());
            let impl_v = (rho - 10.0).hypot(p.z()) - 3.0;
            assert!(impl_v.abs() < 1e-9, "off-torus impl={impl_v}");
        }
    }

    /// A ray out of a pocket corner's spindle torus (major 0.1, minor 2.45)
    /// meets the tube swung across the axis at t 0.124 and its own tube at
    /// 0.398: each root lies on one of the two tubes, not polished off the
    /// first onto a point beside the second.
    #[test]
    fn line_spindle_torus_roots_lie_on_their_tubes() {
        let torus = ToroidalSurface::new(Point3::new(41.0, 17.0, 4.7), 0.1, 2.45).unwrap();
        let origin = Point3::new(42.36, 18.31, 3.4);
        let dir = Vec3::new(
            0.573_576_436_351_046,
            0.740_535_693_464_567_5,
            0.350_889_803_483_932_2,
        );
        let ts = intersect_line_torus(&torus, origin, dir);
        let tube = |t: f64, across: f64| {
            let q = origin + dir * t;
            ((q.x() - 41.0).hypot(q.y() - 17.0) - across * 0.1).hypot(q.z() - 4.7) - 2.45
        };
        for &t in &ts {
            assert!(
                tube(t, 1.0).abs().min(tube(t, -1.0).abs()) < 1e-9,
                "root {t} off both tubes in {ts:?}"
            );
        }
        for want in [0.124, 0.398] {
            assert!(
                ts.iter().any(|&t| (t - want).abs() < 1e-3),
                "no root near {want} in {ts:?}"
            );
        }
    }

    #[test]
    fn line_torus_miss_and_tangent() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 10.0, 3.0).unwrap();
        // A vertical line at rho beyond the outer rim (x=20) misses entirely.
        let miss = intersect_line_torus(
            &torus,
            Point3::new(20.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        );
        assert!(miss.is_empty(), "expected no crossings, got {miss:?}");
        // The z-axis (rho=0) passes through the hole — no intersection.
        let axis =
            intersect_line_torus(&torus, Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
        assert!(axis.is_empty(), "z-axis should miss the tube, got {axis:?}");
    }

    #[test]
    fn dispatch_via_analytic_surface() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let curves = intersect_plane_analytic(
            AnalyticSurface::Cylinder(&cyl),
            Vec3::new(0.0, 0.0, 1.0),
            0.0,
        )
        .unwrap();
        assert!(!curves.is_empty());
    }

    #[test]
    fn perpendicular_cylinders_intersect() {
        let cyl_z =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let cyl_x =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Cylinder(&cyl_z),
            AnalyticSurface::Cylinder(&cyl_x),
            16,
        )
        .unwrap();

        assert!(
            !curves.is_empty(),
            "perpendicular cylinders should intersect"
        );

        for c in &curves {
            assert!(
                c.points.len() >= 2,
                "intersection curve should have >= 2 points, got {}",
                c.points.len()
            );
        }
    }

    /// Neither cylinder's rulings all meet the other: the curve is one loop
    /// joined at its two branch points.
    #[test]
    fn partially_overlapping_cylinders_meet_in_one_closed_loop() {
        let cyl_z =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let cyl_x =
            CylindricalSurface::new(Point3::new(0.0, 1.2, 0.0), Vec3::new(1.0, 0.0, 0.0), 1.0)
                .unwrap();
        let curves = algebraic_cylinder_cylinder(&cyl_z, &cyl_x)
            .unwrap()
            .unwrap();
        assert_eq!(curves.len(), 1);
        let curve = &curves[0].curve;
        let (t0, t1) = curve.domain();
        assert!((curve.evaluate(t0) - curve.evaluate(t1)).length() < 1e-9);
        let off = |p: Point3| {
            let on_z = (p.x().hypot(p.y()) - 1.0).abs();
            let on_x = ((p.y() - 1.2).hypot(p.z()) - 1.0).abs();
            on_z.max(on_x)
        };
        let worst = (0..=400)
            .map(|k| off(curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 400.0)))
            .fold(0.0, f64::max);
        assert!(worst < 2e-4, "curve leaves the cylinders by {worst}");
    }

    /// Near tangency the thick cylinder's window of rulings (0.02 either side
    /// of a quarter turn) falls between its samples; the thin one's sweep
    /// finds the loop.
    #[test]
    fn near_tangent_cylinders_find_their_loop_on_the_thinner_sweep() {
        let cyl_z =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let cyl_x =
            CylindricalSurface::new(Point3::new(0.0, 1.1998, 0.0), Vec3::new(1.0, 0.0, 0.0), 0.2)
                .unwrap();
        let curves = algebraic_cylinder_cylinder(&cyl_z, &cyl_x)
            .unwrap()
            .expect("the thin cylinder's sweep finds the loop");
        assert_eq!(curves.len(), 1);
    }

    /// Cylinders of radius 1.5 whose axes cross at 60° meet in two ellipses
    /// crossing at the ends of the axes' common normal: two exact loops,
    /// one from each end, each the halves of both ellipses on one side of
    /// the crossing along the first axis, on both cylinders to rounding.
    #[test]
    fn equal_cylinders_with_crossing_axes_meet_in_two_cornered_loops() {
        let crossing = Point3::new(1.0, 2.0, 3.0);
        let (s, c) = 60.0_f64.to_radians().sin_cos();
        let (d1, d2) = (Vec3::new(0.0, 0.0, 1.0), Vec3::new(s, 0.0, c));
        let c1 = CylindricalSurface::new(crossing - d1 * 2.0, d1, 1.5).unwrap();
        let c2 = CylindricalSurface::new(crossing + d2 * 0.7, d2, 1.5).unwrap();
        let curves = algebraic_cylinder_cylinder(&c1, &c2).unwrap().unwrap();
        assert_eq!(curves.len(), 2);
        let off = |p: Point3, cyl: &CylindricalSurface| {
            let q = p - cyl.origin();
            ((q - cyl.axis() * q.dot(cyl.axis())).length() - 1.5).abs()
        };
        let ends = [
            crossing + Vec3::new(0.0, 1.5, 0.0),
            crossing - Vec3::new(0.0, 1.5, 0.0),
        ];
        let mut vertices: Vec<Point3> = Vec::new();
        for (ic, (end, side)) in curves.iter().zip([(ends[0], 1.0), (ends[1], -1.0)]) {
            let (t0, t1) = ic.curve.domain();
            for k in 0..=400 {
                let p = ic.curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 400.0);
                assert!(off(p, &c1).max(off(p, &c2)) < 1e-13);
                assert!(side * (p - crossing).dot(d1) > -1e-13);
            }
            assert!((ic.curve.evaluate(t0) - end).length() < 1e-14);
            assert!((ic.curve.evaluate(t1) - end).length() < 1e-14);
            let other = ends[usize::from(side > 0.0)];
            assert!((ic.curve.evaluate(0.5) - other).length() < 1e-14);
            for t in [0.25, 0.75] {
                let vertex = ic.curve.evaluate(t);
                assert!(vertices.iter().all(|&v| (v - vertex).length() > 1.0));
                vertices.push(vertex);
            }
        }
    }

    /// Axes that miss each other by 1e-9, or radii that differ by as much,
    /// part the curves where the ellipses would cross; neither is taken for
    /// a pair of ellipses, nor a miss of 1e-6 read between origins 1e8
    /// apart along an axis.
    #[test]
    fn nearly_crossing_equal_cylinders_are_not_ellipses() {
        let z = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .unwrap();
        let x = |(x0, y): (f64, f64), r: f64| {
            CylindricalSurface::new(Point3::new(x0, y, 0.0), Vec3::new(1.0, 0.0, 0.0), r).unwrap()
        };
        let taken = |c: &CylindricalSurface| crossing_equal_cylinders(&z, c).unwrap().is_some();
        assert!(taken(&x((-2.0, 0.0), 1.0)));
        assert!(!taken(&x((-2.0, 1e-9), 1.0)));
        assert!(!taken(&x((-2.0, 0.0), 1.0 + 1e-9)));
        assert!(!taken(&x((-1e8, 1e-6), 1.0)));
    }

    /// Axes 1e-8 and 1.5e-5 apart in angle cross in ellipses whose far
    /// vertices rounding alone would move past the tolerance; they are
    /// left to the rulings.
    #[test]
    fn nearly_parallel_equal_cylinders_are_not_ellipses() {
        let z = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 10.0)
            .unwrap();
        for slope in [1e-8, 1.5e-5] {
            let tilted = CylindricalSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(slope, 0.0, 1.0),
                10.0,
            )
            .unwrap();
            assert!(crossing_equal_cylinders(&z, &tilted).unwrap().is_none());
        }
    }

    #[test]
    fn sphere_cylinder_intersect() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Sphere(&sphere),
            AnalyticSurface::Cylinder(&cyl),
            16,
        )
        .unwrap();

        // A sphere of radius 2 and a cylinder of radius 1, both centered
        // at the origin, should intersect (the cylinder passes through
        // the sphere).
        assert!(!curves.is_empty(), "sphere and cylinder should intersect");
    }

    #[test]
    fn exact_sphere_cylinder_coaxial_two_circles() {
        // Sphere r=6 at origin, coaxial cylinder r=3 along z: two latitude
        // circles at z = ±sqrt(36-9) = ±sqrt(27), each of radius 3.
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        let circles = exact_sphere_cylinder(&sphere, &cyl)
            .unwrap()
            .expect("coaxial case returns Some");
        assert_eq!(circles.len(), 2, "through-bore meets the sphere twice");
        let mut zs: Vec<f64> = circles
            .iter()
            .filter_map(|c| match c {
                ExactIntersectionCurve::Circle(circle) => {
                    assert!(
                        (circle.radius() - 3.0).abs() < 1e-9,
                        "rim radius == cyl radius"
                    );
                    Some(circle.center().z())
                }
                _ => None,
            })
            .collect();
        assert_eq!(zs.len(), 2, "both sections must be exact circles");
        zs.sort_by(f64::total_cmp);
        let z = 27.0_f64.sqrt();
        assert!((zs[0] + z).abs() < 1e-9 && (zs[1] - z).abs() < 1e-9);
    }

    #[test]
    fn exact_sphere_cylinder_non_coaxial_defers() {
        // Cylinder axis offset from the sphere center → quartic curve, deferred.
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 6.0).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 3.0)
                .unwrap();
        assert!(
            exact_sphere_cylinder(&sphere, &cyl).unwrap().is_none(),
            "non-coaxial sphere/cylinder defers to the marcher"
        );
    }

    #[test]
    fn a_ball_on_a_cones_axis_meets_it_in_circles() {
        // Apex 3 up, opening downward, the radius half the depth below it.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 3.0),
            Vec3::new(0.0, 0.0, -1.0),
            2.0_f64.atan(),
        )
        .unwrap();
        for (height, radius, count) in [
            (0.0, 2.0, 2), // through the cone's middle
            (2.5, 1.3, 1), // swallowing the apex
            (2.5, 0.5, 1), // through the apex: the apex root is no circle
            (0.0, 1.0, 0), // inside, clear of the wall
            (5.0, 1.0, 0), // on the other nappe's side
        ] {
            let centre = Point3::new(0.0, 0.0, height);
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let curves = exact_cone_sphere(&cone, &ball).unwrap().unwrap();
            let circles = circles_of(&curves);
            assert_eq!(circles.len(), count, "ball at {height}, radius {radius}");
            for circle in circles {
                for k in 0..16 {
                    let p = circle.evaluate(TAU * f64::from(k) / 16.0);
                    let on_ball = (p - centre).length() - radius;
                    let on_cone = p.x().hypot(p.y()) - (3.0 - p.z()) / 2.0;
                    assert!(
                        on_ball.abs() < 1e-9 && on_cone.abs() < 1e-9,
                        "ball at {height}: off by {on_ball}, {on_cone}"
                    );
                }
            }
        }
        let aside = SphericalSurface::new(Point3::new(0.5, 0.0, 0.0), 2.0).unwrap();
        assert!(exact_cone_sphere(&cone, &aside).unwrap().is_none());
        // A ball touching a wide cone a million units out, where the
        // discriminant cancels to a few ulps below zero.
        let wide =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.3).unwrap();
        let far = SphericalSurface::new(Point3::new(0.0, 0.0, 1e6), 1e6 * 0.3_f64.cos()).unwrap();
        let curves = exact_cone_sphere(&wide, &far).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(circles.len(), 1, "the touch");
        let touch = 1e6 * 0.3_f64.sin() * 0.3_f64.cos();
        assert!(
            (circles[0].radius() - touch).abs() < 1e-3,
            "{}",
            circles[0].radius()
        );
    }

    /// The circles among exact section curves.
    fn circles_of(curves: &[ExactIntersectionCurve]) -> Vec<&Circle3D> {
        curves
            .iter()
            .filter_map(|c| match c {
                ExactIntersectionCurve::Circle(circle) => Some(circle),
                _ => None,
            })
            .collect()
    }

    /// Worst distance of a circle's points from a torus and from a second
    /// surface given by its own distance function.
    fn worst_off(
        circles: &[&Circle3D],
        torus: &ToroidalSurface,
        other: impl Fn(Point3) -> f64,
    ) -> f64 {
        let mut worst = 0.0_f64;
        for circle in circles {
            for k in 0..16 {
                let p = circle.evaluate(TAU * f64::from(k) / 16.0);
                let q = p - torus.center();
                let along = q.dot(torus.z_axis());
                let rho = (q - torus.z_axis() * along).length();
                let off = ((rho - torus.major_radius()).hypot(along) - torus.minor_radius()).abs();
                worst = worst.max(off).max(other(p).abs());
            }
        }
        worst
    }

    #[test]
    fn exact_sphere_torus_meets_a_ball_on_the_axis_in_circles() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        for height in [0.0, 1.0] {
            let centre = Point3::new(0.0, 0.0, height);
            let sphere = SphericalSurface::new(centre, 3.0).unwrap();
            let curves = exact_sphere_torus(&sphere, &torus).unwrap().unwrap();
            let circles = circles_of(&curves);
            assert_eq!((curves.len(), circles.len()), (2, 2), "height {height}");
            let worst = worst_off(&circles, &torus, |p| (p - centre).length() - 3.0);
            assert!(worst < 1e-9, "height {height}: {worst}");
        }
    }

    /// Balls off a ring torus's axis, one poking out of the tube's top, one
    /// wider than the tube on its core circle, one anywhere, and one just past
    /// the tube's outside, in a loop narrower than the scan's step, meet the
    /// torus in curves whose points lie on both surfaces, given either way
    /// round, where the marcher found nothing with the torus first.
    #[test]
    fn a_ball_off_a_torus_axis_meets_it_in_its_closed_form_loops() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let on_torus = |p: Point3| (p.x().hypot(p.y()) - 4.0).hypot(p.z()) - 1.5;
        for (centre, radius, loops) in [
            (Point3::new(4.0, 0.0, 1.0), 1.0, 1),
            (Point3::new(4.0, 0.0, 0.0), 2.0, 2),
            (Point3::new(2.0, 1.0, 0.5), 1.2, 1),
            (Point3::new(6.4999, 0.0, 0.0), 1.0, 1),
        ] {
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let on_ball = |p: Point3| (p - centre).length() - radius;
            for ball_first in [true, false] {
                let (a, b) = if ball_first {
                    (
                        AnalyticSurface::Sphere(&ball),
                        AnalyticSurface::Torus(&torus),
                    )
                } else {
                    (
                        AnalyticSurface::Torus(&torus),
                        AnalyticSurface::Sphere(&ball),
                    )
                };
                let curves = intersect_analytic_analytic(a, b, 32).unwrap();
                let case = format!("ball {radius} at {centre:?}, ball first {ball_first}");
                assert_eq!(curves.len(), loops, "{case}");
                for ic in &curves {
                    for q in &ic.points {
                        let off = on_ball(q.point).abs().max(on_torus(q.point).abs());
                        assert!(off < 1e-9, "{case}: a point lies {off:.2e} off");
                        let (on_ball_at, on_torus_at) = if ball_first {
                            (q.param1, q.param2)
                        } else {
                            (q.param2, q.param1)
                        };
                        let (pb, pt) = (
                            ball.evaluate(on_ball_at.0, on_ball_at.1),
                            torus.evaluate(on_torus_at.0, on_torus_at.1),
                        );
                        assert!(
                            (pb - q.point).length() < 1e-9 && (pt - q.point).length() < 1e-9,
                            "{case}: a point's parameters miss it"
                        );
                    }
                    let (t0, t1) = ic.curve.domain();
                    for k in 0..=400 {
                        let p = ic.curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 400.0);
                        let off = on_ball(p).abs().max(on_torus(p).abs());
                        assert!(off < 1e-7, "{case}: the curve strays {off:.2e}");
                    }
                }
            }
        }
    }

    /// Balls whose loops a fixed scan of the tube misses or blurs: one far
    /// smaller than its torus, one near the axis whose two loops share a
    /// scan step, one a hair larger than the tube it sits in, whose two
    /// loops lie close together all the way round, and one whose winding
    /// loops swing round the axis within a few scan steps. Each loop is met
    /// either way round, and its curve holds to both surfaces.
    #[test]
    fn balls_meet_tori_in_loops_a_scan_would_miss() {
        for (big, small, centre, radius, loops) in [
            (4000.0, 1500.0, Point3::new(5500.0, 0.0, 0.0), 0.001, 1),
            (4.0, 1.5, Point3::new(1e-6, 0.0, 0.0), 2.500_002, 2),
            (1000.0, 1.0, Point3::new(1000.0, 0.0, 0.0), 1.000_000_5, 2),
            (4.0, 1.5, Point3::new(1.500_001, 0.0, 0.0), 4.0, 2),
        ] {
            let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), big, small).unwrap();
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let on_torus = |p: Point3| (p.x().hypot(p.y()) - big).hypot(p.z()) - small;
            let on_ball = |p: Point3| (p - centre).length() - radius;
            for ball_first in [true, false] {
                let (a, b) = if ball_first {
                    (
                        AnalyticSurface::Sphere(&ball),
                        AnalyticSurface::Torus(&torus),
                    )
                } else {
                    (
                        AnalyticSurface::Torus(&torus),
                        AnalyticSurface::Sphere(&ball),
                    )
                };
                let curves = intersect_analytic_analytic(a, b, 32).unwrap();
                let case = format!("ball {radius} at {centre:?} on torus ({big}, {small})");
                assert_eq!(curves.len(), loops, "{case}");
                for ic in &curves {
                    for q in &ic.points {
                        let off = on_ball(q.point).abs().max(on_torus(q.point).abs());
                        assert!(off < 1e-9, "{case}: a point lies {off:.2e} off");
                    }
                    let (t0, t1) = ic.curve.domain();
                    for k in 0..=2000 {
                        let p = ic.curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 2000.0);
                        let off = on_ball(p).abs().max(on_torus(p).abs());
                        assert!(off < 1e-7, "{case}: the curve strays {off:.2e}");
                    }
                }
            }
        }
    }

    /// Sections a closed form can lose to scale or rounding: a plane whose
    /// equation is scaled by 1e14 keeps the loop of the plane it describes; a
    /// ball 1e-5 across poking out of a large torus, and a ball a micron wider
    /// than the tube of a ring 1e8 across, defer to the marcher instead of
    /// reading as apart, and the last is not taken for the meridian circle.
    #[test]
    fn scaled_and_ill_conditioned_sections_are_kept_or_deferred() {
        let origin = Point3::new(0.0, 0.0, 0.0);
        let torus = ToroidalSurface::new(origin, 4.0, 1.5).unwrap();
        let unit = plane_torus_loops(&torus, Vec3::new(1.0, 0.0, 0.0), 4.0, 128).len();
        let scaled = plane_torus_loops(&torus, Vec3::new(1e14, 0.0, 0.0), 4e14, 128).len();
        assert_eq!((unit, scaled), (1, 1));

        let large = ToroidalSurface::new(origin, 4000.0, 1500.0).unwrap();
        let speck = SphericalSurface::new(Point3::new(5_500.000_001, 0.0, 0.0), 1e-5).unwrap();
        assert!(sphere_torus_curves(&speck, &large, true).is_none());

        let ring = ToroidalSurface::new(origin, 1e8, 1.0).unwrap();
        let ball = SphericalSurface::new(Point3::new(1e8, 0.0, 0.0), 1.000_001).unwrap();
        assert!(exact_sphere_torus(&ball, &ring).unwrap().is_none());
        assert!(sphere_torus_curves(&ball, &ring, true).is_none());
    }

    /// A ball centred on a ring torus's core circle with the tube's own
    /// radius meets it along the meridian circle through its centre: one
    /// circle, given either way round.
    #[test]
    fn a_ball_filling_a_torus_tube_meets_it_in_one_meridian_circle() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let centre = Point3::new(4.0, 0.0, 0.0);
        let ball = SphericalSurface::new(centre, 1.5).unwrap();
        for (a, b) in [
            (
                AnalyticSurface::Sphere(&ball),
                AnalyticSurface::Torus(&torus),
            ),
            (
                AnalyticSurface::Torus(&torus),
                AnalyticSurface::Sphere(&ball),
            ),
        ] {
            let curves = intersect_analytic_analytic(a, b, 32).unwrap();
            assert_eq!(curves.len(), 1);
            let (t0, t1) = curves[0].curve.domain();
            for k in 0..=100 {
                let p = curves[0]
                    .curve
                    .evaluate(t0 + (t1 - t0) * f64::from(k) / 100.0);
                let off = ((p - centre).length() - 1.5).abs().max(p.y().abs());
                assert!(off < 1e-9, "the circle strays {off:.2e}");
            }
        }
    }

    #[test]
    fn exact_sphere_torus_misses_touches_and_defers() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let ball = |x: f64, r: f64| SphericalSurface::new(Point3::new(x, 0.0, 0.0), r).unwrap();
        assert!(
            exact_sphere_torus(&ball(0.0, 1.0), &torus)
                .unwrap()
                .unwrap()
                .is_empty(),
            "a small ball in the hole misses"
        );
        let touching = exact_sphere_torus(&ball(0.0, 2.5), &torus)
            .unwrap()
            .unwrap();
        let circles = circles_of(&touching);
        assert_eq!(
            circles.len(),
            1,
            "a ball in the hole touches the inner equator"
        );
        assert!((circles[0].radius() - 2.5).abs() < 1e-12);
        assert!(circles[0].center().z().abs() < 1e-12);
        assert!(
            exact_sphere_torus(&ball(1.0, 3.0), &torus)
                .unwrap()
                .is_none(),
            "a ball off the axis defers"
        );
        let spindle = ToroidalSurface::with_axis_and_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            2.0,
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        assert!(
            exact_sphere_torus(&ball(0.0, 2.5), &spindle)
                .unwrap()
                .is_none()
        );
    }

    /// A rod of radius 0.6 along `y` through the ring (4, 1.5) at `x = 0.5`,
    /// `z = 1` leaves the tube through its top, and meets it in one loop on
    /// each side of the ring. The march back to a loop's seed can be several
    /// steps long; each loop's curve lies on both surfaces all along it.
    #[test]
    fn a_marched_loop_closes_on_both_surfaces() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let rod =
            CylindricalSurface::new(Point3::new(0.5, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0), 0.6)
                .unwrap();
        let region = Aabb3 {
            min: Point3::new(-0.1, -5.5, 0.4),
            max: Point3::new(1.1, 5.5, 1.5),
        };
        let curves = intersect_analytic_analytic_in_region(
            AnalyticSurface::Cylinder(&rod),
            AnalyticSurface::Torus(&torus),
            32,
            Some((-10.0, 10.0)),
            None,
            region,
        )
        .unwrap();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            let (t0, t1) = c.curve.domain();
            let worst = (0..=4000)
                .map(|k| {
                    let p = c.curve.evaluate((t1 - t0).madd(f64::from(k) / 4000.0, t0));
                    let on_torus = ((p.x().hypot(p.y()) - 4.0).hypot(p.z()) - 1.5).abs();
                    let on_rod = ((p.x() - 0.5).hypot(p.z() - 1.0) - 0.6).abs();
                    on_torus.max(on_rod)
                })
                .fold(0.0, f64::max);
            assert!(worst <= 1e-7, "off the surfaces by {worst}");
        }
    }

    /// A funnel's outer wall, a 45 degree cone, crosses the tube of a lip
    /// that leans a thousandth of a radian off the cone's axis in one loop
    /// round the axis. The marcher closes the loop on its seed, and the loop
    /// runs round the axis once.
    #[test]
    fn a_marched_loop_round_an_axis_runs_round_it_once() {
        let skin = 1.6 * std::f64::consts::SQRT_2;
        let tilt = 0.001_f64;
        let torus = ToroidalSurface::with_axis(
            Point3::new(0.0, 0.0, 70.5),
            80.8 + skin / 2.0,
            2.0,
            Vec3::new(0.0, -tilt.sin(), tilt.cos()),
        )
        .unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -9.5 - skin),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let region = Aabb3 {
            min: Point3::new(-84.0, -84.0, 68.0),
            max: Point3::new(84.0, 84.0, 73.0),
        };
        let curves = intersect_analytic_analytic_in_region(
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Torus(&torus),
            32,
            Some((16.6, 116.3)),
            None,
            region,
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        let (t0, t1) = curves[0].curve.domain();
        let (start, end) = (curves[0].curve.evaluate(t0), curves[0].curve.evaluate(t1));
        assert!((start - end).length() < 1e-9, "{start:?} to {end:?}");
        let n = 4000;
        let mut turn = 0.0;
        let mut last = start.y().atan2(start.x());
        for k in 1..=n {
            let p = curves[0]
                .curve
                .evaluate((t1 - t0).madd(f64::from(k) / f64::from(n), t0));
            let at = p.y().atan2(p.x());
            turn += (at - last + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
            last = at;
        }
        assert!((turn.abs() - TAU).abs() < 1e-6, "turns {}", turn / TAU);
    }

    /// A 45 degree cone about the axis of the ring (4, 1.5), its apex at
    /// `z = apex` below the ring: its generator `ρ = z − apex` stands
    /// `|4 + apex| / √2` from the tube's centre in the half-plane.
    #[test]
    fn exact_cone_torus_meets_a_coaxial_cone_in_circles() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let quarter = std::f64::consts::FRAC_PI_4;
        let cone = |apex: f64, axis: Vec3| {
            ConicalSurface::new(Point3::new(0.0, 0.0, apex), axis, quarter).unwrap()
        };
        let on_cone = |apex: f64| move |p: Point3| p.x().hypot(p.y()) - (p.z() - apex);
        let curves = exact_cone_torus(&cone(-5.0, z), &torus).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(
            (curves.len(), circles.len()),
            (2, 2),
            "a wall through the tube"
        );
        let worst = worst_off(&circles, &torus, on_cone(-5.0));
        assert!(worst < 1e-12, "{worst}");
        assert!(
            exact_cone_torus(&cone(-8.0, z), &torus)
                .unwrap()
                .unwrap()
                .is_empty(),
            "a wall clear of the tube misses it"
        );
        let touch = -4.0 - 1.5 * std::f64::consts::SQRT_2;
        let curves = exact_cone_torus(&cone(touch, z), &torus).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(circles.len(), 1, "a wall touching the tube");
        let worst = worst_off(&circles, &torus, on_cone(touch));
        assert!(worst < 1e-12, "{worst}");
        assert_eq!(
            exact_cone_torus(&cone(touch + 1e-6, z), &torus)
                .unwrap()
                .unwrap()
                .len(),
            2,
            "a wall into the tube by more than the tolerance crosses it twice"
        );
        // The cone's axis may run either way along the torus's.
        let below = ConicalSurface::new(Point3::new(0.0, 0.0, 5.0), -z, quarter).unwrap();
        let curves = exact_cone_torus(&below, &torus).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(circles.len(), 2);
        let worst = worst_off(&circles, &torus, |p| p.x().hypot(p.y()) - (5.0 - p.z()));
        assert!(worst < 1e-12, "{worst}");
        let tilted = cone(-5.0, Vec3::new(0.0, 0.1, 1.0));
        let offset = ConicalSurface::new(Point3::new(0.5, 0.0, -5.0), z, quarter).unwrap();
        assert!(exact_cone_torus(&tilted, &torus).unwrap().is_none());
        assert!(exact_cone_torus(&offset, &torus).unwrap().is_none());
        // A near cylinder whose apex lies a million below, its axis 0.0005
        // beside the torus's: radius 4.75 at the torus, off its axis.
        let far = ConicalSurface::new(Point3::new(0.0005, 0.0, -1e6), z, (1e6_f64 / 4.75).atan())
            .unwrap();
        assert!(
            exact_cone_torus(&far, &torus).unwrap().is_none(),
            "a cone off the torus's axis by 0.0005 is not coaxial, however far its apex"
        );
        let spindle = ToroidalSurface::with_axis_and_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            2.0,
            z,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        assert!(
            exact_cone_torus(&cone(-5.0, z), &spindle)
                .unwrap()
                .is_none(),
            "a spindle torus's tube reaches the axis"
        );
    }

    #[test]
    fn exact_cylinder_torus_meets_a_coaxial_rod_in_circles() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let rod = |r: f64| CylindricalSurface::new(Point3::new(0.0, 0.0, -5.0), z, r).unwrap();
        let curves = exact_cylinder_torus(&rod(4.2), &torus).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!((curves.len(), circles.len()), (2, 2));
        let worst = worst_off(&circles, &torus, |p| p.x().hypot(p.y()) - 4.2);
        assert!(worst < 1e-9, "{worst}");
        assert!(
            exact_cylinder_torus(&rod(2.0), &torus)
                .unwrap()
                .unwrap()
                .is_empty(),
            "a rod clear in the hole misses"
        );
        for (wall, label) in [(5.5, "outer"), (2.5, "inner")] {
            let curves = exact_cylinder_torus(&rod(wall), &torus).unwrap().unwrap();
            let circles = circles_of(&curves);
            assert_eq!(circles.len(), 1, "a wall touching the {label} equator");
            assert!(circles[0].center().z().abs() < 1e-12, "{label}");
            let worst = worst_off(&circles, &torus, |p| p.x().hypot(p.y()) - wall);
            assert!(worst < 1e-9, "{label}: {worst}");
        }
        assert!(
            exact_cylinder_torus(&rod(5.5 + 1e-6), &torus)
                .unwrap()
                .unwrap()
                .is_empty(),
            "a wall clear of the tube by more than the tolerance misses it"
        );
        assert_eq!(
            exact_cylinder_torus(&rod(5.5 - 1e-6), &torus)
                .unwrap()
                .unwrap()
                .len(),
            2,
            "a wall into the tube by more than the tolerance crosses it twice"
        );
        // Within the linear tolerance of the tube the wall reads as touching
        // it: the two crossings lie under 1e-3 apart on a band the wall
        // never leaves by more than that tolerance.
        for wall in [5.5 - 5e-8, 5.5 + 5e-8] {
            let curves = exact_cylinder_torus(&rod(wall), &torus).unwrap().unwrap();
            assert_eq!(circles_of(&curves).len(), 1, "a wall at {wall}");
        }
        let tilted =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.1, 1.0), 4.2)
                .unwrap();
        let offset = CylindricalSurface::new(Point3::new(0.5, 0.0, 0.0), z, 4.2).unwrap();
        assert!(exact_cylinder_torus(&tilted, &torus).unwrap().is_none());
        assert!(exact_cylinder_torus(&offset, &torus).unwrap().is_none());
        let spindle = ToroidalSurface::with_axis_and_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            1.0,
            2.0,
            z,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        assert!(
            exact_cylinder_torus(&rod(0.5), &spindle).unwrap().is_none(),
            "a spindle torus's inner lemon also meets the rod"
        );
        // The rounded air wall of a fillet whose radius nearly fills its
        // corner: the spindle's outer equator rests on the wall.
        let spindle = ToroidalSurface::with_axis_and_ref_dir(
            Point3::new(0.0, 0.0, 4.7),
            0.1,
            2.45,
            z,
            Vec3::new(1.0, 0.0, 0.0),
        )
        .unwrap();
        let curves = exact_cylinder_torus(&rod(2.55), &spindle).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(circles.len(), 1);
        assert!((circles[0].center().z() - 4.7).abs() < 1e-12);
        let worst = worst_off(&circles, &spindle, |p| p.x().hypot(p.y()) - 2.55);
        assert!(worst < 1e-9, "{worst}");
        let curves = exact_cylinder_torus(&rod(2.4), &spindle).unwrap().unwrap();
        let circles = circles_of(&curves);
        assert_eq!(circles.len(), 2, "a wall into the spindle's outer tube");
        let worst = worst_off(&circles, &spindle, |p| p.x().hypot(p.y()) - 2.4);
        assert!(worst < 1e-9, "{worst}");
    }

    /// Loops of an off-axis sphere-cylinder pair: `(count, worst distance
    /// from either surface)`.
    fn off_axis_loops(cylinder_origin: Point3, cylinder_radius: f64) -> (usize, f64) {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let cyl =
            CylindricalSurface::new(cylinder_origin, Vec3::new(0.0, 0.0, 1.0), cylinder_radius)
                .unwrap();
        let curves = algebraic_sphere_cylinder(&sphere, &cyl, true)
            .unwrap()
            .unwrap();
        let mut worst: f64 = 0.0;
        for c in &curves {
            for ip in &c.points {
                let on_sphere = sphere.evaluate(ip.param1.0, ip.param1.1);
                let on_cylinder = cyl.evaluate(ip.param2.0, ip.param2.1);
                worst = worst
                    .max((on_sphere - ip.point).length())
                    .max((on_cylinder - ip.point).length());
            }
            let (t0, t1) = c.curve.domain();
            assert!((c.curve.evaluate(t0) - c.curve.evaluate(t1)).length() < 1e-9);
            for k in 0..=400 {
                let p = c.curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 400.0);
                let on_sphere = ((p - Point3::new(0.0, 0.0, 0.0)).length() - 2.0).abs();
                let on_cylinder = ((p.x() - cylinder_origin.x())
                    .hypot(p.y() - cylinder_origin.y())
                    - cylinder_radius)
                    .abs();
                worst = worst.max(on_sphere).max(on_cylinder);
            }
        }
        (curves.len(), worst)
    }

    /// A drill off the ball's axis passes through it: an entry and an exit
    /// loop.
    #[test]
    fn off_axis_drill_through_a_sphere_meets_it_in_two_loops() {
        let (count, worst) = off_axis_loops(Point3::new(0.5, 0.0, 0.0), 0.2);
        assert_eq!(count, 2);
        assert!(worst < 1e-5, "loops leave the surfaces by {worst}");
    }

    /// A cylinder over the ball's side: one loop joined at its branch points.
    #[test]
    fn cylinder_over_a_spheres_side_meets_it_in_one_loop() {
        let (count, worst) = off_axis_loops(Point3::new(1.8, 0.0, 0.0), 0.5);
        assert_eq!(count, 1);
        assert!(worst < 5e-4, "loop leaves the surfaces by {worst}");
    }

    /// A ball through the wall of a ring far wider than it: the rulings that
    /// meet the ball span a twelfth of the turn, and the loop through them
    /// stays on both surfaces.
    #[test]
    fn a_ball_through_a_wide_wall_meets_it_in_one_close_loop() {
        let centre = Point3::new(7.35, 0.0, 0.0);
        let sphere = SphericalSurface::new(centre, 2.4145).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 9.15)
                .unwrap();
        let curves = algebraic_sphere_cylinder(&sphere, &cyl, true)
            .unwrap()
            .unwrap();
        assert_eq!(curves.len(), 1);
        let (t0, t1) = curves[0].curve.domain();
        let mut worst: f64 = 0.0;
        for k in 0..=2000 {
            let p = curves[0]
                .curve
                .evaluate(t0 + (t1 - t0) * f64::from(k) / 2000.0);
            let on_sphere = ((p - centre).length() - 2.4145).abs();
            let on_cylinder = (p.x().hypot(p.y()) - 9.15).abs();
            worst = worst.max(on_sphere).max(on_cylinder);
        }
        assert!(worst < 1e-6, "the loop leaves the surfaces by {worst}");
    }

    /// A whole torus ends where it starts both ways round; its size is read
    /// from its first corner to its middle, across the ring.
    #[test]
    fn a_whole_torus_patch_spans_its_ring() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 50.0, 10.0).unwrap();
        let size = patch_diagonal(&|u, v| torus.evaluate(u, v), (0.0, TAU), (0.0, TAU));
        assert!((size - 100.0).abs() < 1e-9, "size {size}");
    }

    /// A cone's face, between two of its circles, and a cylinder's face
    /// across it: each marched curve ends on the edge of one face's band,
    /// where the other faces' sections meet it, not a march step short.
    #[test]
    fn marched_cone_cylinder_curves_end_on_the_band_edges() {
        let half_angle = 14.0_f64.atan2(10.0);
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 273.2),
            Vec3::new(0.0, 0.0, -1.0),
            half_angle,
        )
        .unwrap();
        let cyl = CylindricalSurface::new(
            Point3::new(0.0, -50.0, 215.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
        )
        .unwrap();
        // The cone from z = 192 down to z = 178, the cylinder from y = -40
        // to y = 40.
        let v_cone = (81.2 / half_angle.sin(), 95.2 / half_angle.sin());
        let region = Aabb3 {
            min: Point3::new(-59.98, -40.0, 178.0),
            max: Point3::new(-47.23, 40.0, 192.0),
        };
        let curves = intersect_analytic_analytic_in_region(
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&cyl),
            32,
            Some(v_cone),
            Some((10.0, 90.0)),
            region,
        )
        .unwrap();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            let (t0, t1) = c.curve.domain();
            for p in [c.curve.evaluate(t0), c.curve.evaluate(t1)] {
                let on_edge = (p.z() - 192.0).abs() < 1e-6 || (p.y().abs() - 40.0).abs() < 1e-6;
                assert!(on_edge, "a curve ends at {p:?}");
            }
        }
    }

    /// Long steps along the same cone and cylinder: a step whose predictor
    /// leaves a band while the curve stays on it carries the march on, and
    /// each end still lies on a band's edge.
    #[test]
    fn long_march_steps_end_on_the_band_edges() {
        let half_angle = 14.0_f64.atan2(10.0);
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 273.2),
            Vec3::new(0.0, 0.0, -1.0),
            half_angle,
        )
        .unwrap();
        let cyl = CylindricalSurface::new(
            Point3::new(0.0, -50.0, 215.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
        )
        .unwrap();
        let (a, b) = (
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&cyl),
        );
        let (surf_a, norm_a, u_a, _) = surface_closures(&a);
        let (surf_b, norm_b, u_b, _) = surface_closures(&b);
        let v_a = (81.2 / half_angle.sin(), 95.2 / half_angle.sin());
        let v_b = (10.0, 90.0);
        let seed = correct_to_intersection(
            &a,
            &b,
            surf_a.as_ref(),
            norm_a.as_ref(),
            surf_b.as_ref(),
            norm_b.as_ref(),
            Point3::new(-53.0, -30.0, 187.0),
            u_a,
            v_a,
            u_b,
            v_b,
            20,
        );
        for step in [0.5, 1.0, 2.0, 3.0] {
            let points = march_analytic_intersection(
                &a,
                &b,
                surf_a.as_ref(),
                norm_a.as_ref(),
                surf_b.as_ref(),
                norm_b.as_ref(),
                seed,
                u_a,
                v_a,
                u_b,
                v_b,
                step,
                true,
                true,
                None,
            )
            .unwrap();
            for p in [points[0], points[points.len() - 1]] {
                let on_edge = (p.z() - 192.0).abs() < 1e-6 || (p.y().abs() - 40.0).abs() < 1e-6;
                assert!(on_edge, "step {step}: the march ends at {p:?}");
            }
        }
    }

    #[test]
    fn disjoint_cylinders_no_intersection() {
        let cyl_a =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5)
                .unwrap();
        let cyl_b =
            CylindricalSurface::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5)
                .unwrap();

        let curves = intersect_analytic_analytic(
            AnalyticSurface::Cylinder(&cyl_a),
            AnalyticSurface::Cylinder(&cyl_b),
            16,
        )
        .unwrap();

        assert!(curves.is_empty(), "disjoint cylinders should not intersect");
    }

    // ── Oblique plane × cone conic (ellipse / parabola / hyperbola) ──────

    /// Collect 3D points from a returned exact curve, sampling analytic forms.
    fn collect_points(curve: &ExactIntersectionCurve) -> Vec<Point3> {
        use crate::traits::ParametricCurve;
        match curve {
            ExactIntersectionCurve::Circle(c) => (0..=64)
                .map(|i| ParametricCurve::evaluate(c, TAU * f64::from(i) / 64.0))
                .collect(),
            ExactIntersectionCurve::Ellipse(e) => (0..=64)
                .map(|i| ParametricCurve::evaluate(e, TAU * f64::from(i) / 64.0))
                .collect(),
            ExactIntersectionCurve::Points(pts) => pts.clone(),
        }
    }

    /// Assert every returned point lies on the plane and the cone surface, on
    /// the real (`v >= 0`) nappe, and within a sane axial bound.
    fn assert_on_plane_and_cone(
        curves: &[ExactIntersectionCurve],
        cone: &ConicalSurface,
        n: Vec3,
        d: f64,
        z_bound: (f64, f64),
    ) {
        assert!(!curves.is_empty(), "expected at least one section curve");
        let mut total = 0;
        for curve in curves {
            for p in collect_points(curve) {
                total += 1;
                let plane_err = (n.x() * p.x() + n.y() * p.y() + n.z() * p.z() - d).abs();
                assert!(
                    plane_err < 1e-9,
                    "point off plane by {plane_err:.2e}: {p:?}"
                );
                let (u, v) = cone.project_point(p);
                let q = cone.evaluate(u, v);
                let cone_err =
                    ((p.x() - q.x()).powi(2) + (p.y() - q.y()).powi(2) + (p.z() - q.z()).powi(2))
                        .sqrt();
                assert!(cone_err < 1e-7, "point off cone by {cone_err:.2e}: {p:?}");
                assert!(v >= -1e-9, "point on phantom nappe (v={v:.4}): {p:?}");
                assert!(
                    p.z() >= z_bound.0 - 1e-6 && p.z() <= z_bound.1 + 1e-6,
                    "point z={:.4} outside sane bound {z_bound:?}: {p:?}",
                    p.z()
                );
            }
        }
        assert!(total >= 8, "too few section points ({total})");
    }

    #[test]
    fn oblique_plane_cone_ellipse_is_exact_and_on_both() {
        // 45°-half-angle cone (axis +z). A plane tilted only ~16.7° off horizontal
        // has plane-axis angle ≈ 73° > 45° (the cone's half-opening from axis) →
        // ellipse. Must come back as an exact Ellipse, fully on both surfaces.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let n = Vec3::new(0.3, 0.0, 1.0).normalize().unwrap();
        // Plane through (0,0,5): d = n·(0,0,5).
        let d = n.z() * 5.0;
        let curves = exact_plane_cone(&cone, n, d, 0.0).unwrap();
        assert!(
            curves
                .iter()
                .any(|c| matches!(c, ExactIntersectionCurve::Ellipse(_))),
            "oblique steep plane × cone must yield an exact Ellipse"
        );
        // The ellipse straddles z=5; with the 0.3 tilt the z-extent stays modest.
        assert_on_plane_and_cone(&curves, &cone, n, d, (0.0, 12.0));
    }

    #[test]
    fn oblique_plane_cone_wrong_nappe_is_empty() {
        // Same ellipse-regime plane as above, but offset to the FAR side of the
        // apex (z=-5). The +z cone's real (v≥0) nappe is not met — only the
        // phantom v<0 nappe — so the result must be EMPTY, not a phantom ellipse.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let n = Vec3::new(0.3, 0.0, 1.0).normalize().unwrap();
        let d = n.z() * -5.0;
        let curves = exact_plane_cone(&cone, n, d, 0.0).unwrap();
        assert!(
            curves.is_empty(),
            "plane on the phantom-nappe side must yield no real curve, got {}",
            curves.len()
        );
    }

    #[test]
    fn oblique_plane_cone_parabola_on_both_single_branch() {
        // Plane normal at exactly 45° to the axis (= the cone half-opening) → the
        // plane is parallel to a generator → parabola. One unbounded branch.
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let n = Vec3::new(1.0, 0.0, 1.0).normalize().unwrap();
        let d = n.x() * 3.0 + n.z() * 3.0; // through (3,0,3)
        let curves = exact_plane_cone(&cone, n, d, 0.0).unwrap();
        assert_eq!(
            curves.len(),
            1,
            "a parabola is a single branch, got {}",
            curves.len()
        );
        // Bounded by r_max = 32·|e|; |e| here is O(few), so allow a wide z window.
        assert_on_plane_and_cone(&curves, &cone, n, d, (0.0, 400.0));
    }

    #[test]
    fn oblique_plane_cone_hyperbola_real_nappe_only() {
        // Faithful scooplabel lip-foot geometry: a 45° cone with axis −z and
        // apex at (−59,−59,15.85) (a bin corner), cut by the upper ramp tread
        // plane n=(0,0.99518,0.09802), d=−58.36056. The plane is nearly parallel
        // to the axis (cos≈0.098) → plane-axis angle ≈ 5.6° < 45° → hyperbola.
        // The downward real nappe is hit by exactly one branch; the phantom
        // upward nappe (and the asymptote runaway) must NOT appear, and the arc
        // must stay near the apex (the plane is ~1.2 mm from it).
        let cone = ConicalSurface::new(
            Point3::new(-59.0, -59.0, 15.85),
            Vec3::new(0.0, 0.0, -1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let n = Vec3::new(0.0, 0.995_18, 0.098_02).normalize().unwrap();
        let d = -58.360_56;
        let cos_theta = n.dot(cone.axis()).abs();
        assert!(cos_theta < 0.2, "expected a shallow (hyperbola) plane");
        let curves = exact_plane_cone(&cone, n, d, 0.0).unwrap();
        // Real downward nappe only: never above the apex (z=15.85). The vertex is
        // ~1.2 mm from the apex, so the bounded arc stays within a few mm of it.
        assert_on_plane_and_cone(&curves, &cone, n, d, (5.0, 15.85));
        // Every returned curve is sampled Points (no false Circle/Ellipse).
        for c in &curves {
            assert!(
                matches!(c, ExactIntersectionCurve::Points(_)),
                "hyperbola must be sampled Points, not a closed conic"
            );
        }
    }

    /// Two spheres meet in one circle on both of them; spheres apart, one
    /// inside the other, or concentric of different radii meet in nothing,
    /// and one sphere twice has no curve.
    #[test]
    fn sphere_sphere_meets_in_a_circle_on_both() {
        let ball = |x: f64, y: f64, z: f64, r: f64| {
            SphericalSurface::new(Point3::new(x, y, z), r).unwrap()
        };
        let (a, b) = (ball(0.0, 0.0, 0.0, 2.0), ball(1.6, 0.3, 0.2, 1.5));
        let curves = exact_sphere_sphere(&a, &b).unwrap().unwrap();
        assert_eq!(curves.len(), 1);
        let circle = match &curves[0] {
            ExactIntersectionCurve::Circle(circle) => circle,
            other => unreachable!("expected a circle, got {other:?}"),
        };
        for k in 0..16 {
            let p = circle.evaluate(std::f64::consts::TAU * f64::from(k) / 16.0);
            assert!(((p - a.center()).length() - 2.0).abs() < 1e-12);
            assert!(((p - b.center()).length() - 1.5).abs() < 1e-12);
        }
        assert!(
            exact_sphere_sphere(&a, &ball(4.0, 0.0, 0.0, 1.5))
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(
            exact_sphere_sphere(&a, &ball(0.2, 0.0, 0.0, 1.0))
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(
            exact_sphere_sphere(&a, &ball(0.0, 0.0, 0.0, 1.0))
                .unwrap()
                .unwrap()
                .is_empty()
        );
        assert!(
            exact_sphere_sphere(&a, &ball(0.0, 0.0, 0.0, 2.0))
                .unwrap()
                .is_none()
        );

        // A tiny sphere nearly bisected by a large one, in either order, and a
        // pair far from the origin: each meets in its small circle.
        let radius = |p: &SphericalSurface, q: &SphericalSurface| -> f64 {
            match exact_sphere_sphere(p, q).unwrap().unwrap().as_slice() {
                [ExactIntersectionCurve::Circle(c)] => c.radius(),
                other => unreachable!("expected one circle, got {other:?}"),
            }
        };
        let tiny = ball(2.0, 0.0, 0.0, 1e-5);
        assert!((radius(&a, &tiny) - 1e-5).abs() < 1e-9);
        assert!((radius(&tiny, &a) - 1e-5).abs() < 1e-9);
        let near = radius(&ball(0.0, 0.0, 0.0, 2.0), &ball(3.5, 0.0, 0.0, 1.500_001));
        let far = radius(
            &ball(1e10, 0.0, 0.0, 2.0),
            &ball(1e10 + 3.5, 0.0, 0.0, 1.500_001),
        );
        assert!(near > 1e-3);
        assert!((far - near).abs() < 1e-9 * near.max(1.0));
    }

    /// Each line among `sections`, and whether it is a ray; any other
    /// section fails the test.
    #[allow(clippy::panic)]
    fn lines_of(sections: &[PlaneSection]) -> Vec<(Line3D, bool)> {
        sections
            .iter()
            .map(|s| match s {
                PlaneSection::Line { line, ray } => (line.clone(), *ray),
                other => panic!("expected a line, got {other:?}"),
            })
            .collect()
    }

    /// A plane along a cylinder's axis cuts its rulings: two through the
    /// axis, two 2e-6 apart where it sinks a hair into the cylinder, the one
    /// it touches along, and none past it. `exact_plane_analytic` gives no
    /// curve for any of them.
    #[test]
    fn a_plane_along_a_cylinder_cuts_its_rulings() {
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let y = Vec3::new(0.0, 1.0, 0.0);
        let surface = AnalyticSurface::Cylinder(&cyl);
        let on_both = |p: Point3, d: f64| {
            assert!((p.y() - d).abs() < 1e-12, "{p:?} off the plane y = {d}");
            assert!(
                (p.x().hypot(p.y()) - 1.0).abs() < 1e-12,
                "{p:?} off the cylinder"
            );
        };
        for (d, count) in [(0.0, 2), (1e-6_f64.cos(), 2), (1.0, 1), (1.5, 0)] {
            assert!(exact_plane_analytic(surface, y, d).unwrap().is_empty());
            let lines = lines_of(&exact_plane_section(surface, y, d).unwrap());
            assert_eq!(lines.len(), count, "plane y = {d}");
            let mut xs = Vec::new();
            for (line, ray) in lines {
                assert!(!ray);
                assert!(line.direction().cross(cyl.axis()).length() < 1e-12);
                for t in [-3.0, 0.0, 3.0] {
                    on_both(line.evaluate(t), d);
                }
                xs.push(line.evaluate(0.0).x());
            }
            if count == 2 {
                let apart = (xs[0] - xs[1]).abs();
                let want = 2.0 * (1.0 - d * d).sqrt();
                assert!(
                    (apart - want).abs() < 1e-9 * want.max(1e-6),
                    "{apart} apart"
                );
            }
        }
    }

    /// A plane through a cone's apex cuts rays from it: two for a plane
    /// through the axis, one for a plane touching the cone along a ruling,
    /// and the apex alone for a plane missing the nappe. A plane 5e-8 off the
    /// apex still cuts its conic.
    #[test]
    #[allow(clippy::panic)]
    fn a_plane_through_a_cone_apex_cuts_its_rulings() {
        use std::f64::consts::FRAC_PI_4;

        let cone = ConicalSurface::new(
            Point3::new(1.0, 2.0, 3.0),
            Vec3::new(0.0, 0.0, 1.0),
            FRAC_PI_4,
        )
        .unwrap();
        let surface = AnalyticSurface::Cone(&cone);
        let apex = cone.apex();
        let through = |n: Vec3| {
            let n = n.normalize().unwrap();
            let d = n.dot(apex - Point3::new(0.0, 0.0, 0.0));
            assert!(exact_plane_analytic(surface, n, d).unwrap().is_empty());
            (n, d, exact_plane_section(surface, n, d).unwrap())
        };
        for (n, count) in [
            (Vec3::new(1.0, 0.0, 0.0), 2),
            (Vec3::new(1.0, 0.0, -1.0), 1),
        ] {
            let (n, d, sections) = through(n);
            let rays = lines_of(&sections);
            assert_eq!(rays.len(), count, "plane normal {n:?}");
            for (line, ray) in rays {
                assert!(ray);
                assert!((line.evaluate(0.0) - apex).length() < 1e-12);
                let p = line.evaluate(2.0);
                assert!((n.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs() < 1e-12);
                let rel = p - apex;
                assert!(rel.z() > 0.0, "{p:?} on the phantom nappe");
                assert!(
                    (rel.x().hypot(rel.y()) - rel.z()).abs() < 1e-12,
                    "{p:?} off the cone"
                );
            }
        }
        for n in [Vec3::new(0.1, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0)] {
            let (_, _, sections) = through(n);
            assert!(
                matches!(sections[..], [PlaneSection::Point(p)] if (p - apex).length() < 1e-12),
                "{sections:?}"
            );
        }

        let upright = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            FRAC_PI_4,
        )
        .unwrap();
        let n = Vec3::new((0.5 - 1e-8_f64).sqrt(), 0.0, (0.5 + 1e-8_f64).sqrt());
        let sections = exact_plane_section(AnalyticSurface::Cone(&upright), n, 5e-8).unwrap();
        let [PlaneSection::Ellipse(ellipse)] = &sections[..] else {
            panic!("{sections:?}");
        };
        assert!(
            ellipse.semi_major() > 1.0,
            "semi-major {}",
            ellipse.semi_major()
        );
    }

    /// A plane touching a sphere meets it at the point of contact, and one a
    /// hair inside cuts the small circle there, however small.
    #[test]
    #[allow(clippy::panic)]
    fn a_plane_touching_a_sphere_meets_it_at_a_point() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let surface = AnalyticSurface::Sphere(&sphere);
        let z = Vec3::new(0.0, 0.0, 1.0);
        assert!(exact_plane_analytic(surface, z, 0.0).unwrap().is_empty());
        let sections = exact_plane_section(surface, z, 0.0).unwrap();
        assert!(
            matches!(sections[..], [PlaneSection::Point(p)] if p.z().abs() < 1e-15 && p.x() == 0.0),
            "{sections:?}"
        );
        assert!(exact_plane_section(surface, z, -1e-9).unwrap().is_empty());
        for height in [5e-15, 1e-12, 1e-8] {
            let sections = exact_plane_section(surface, z, height).unwrap();
            let [PlaneSection::Circle(circle)] = &sections[..] else {
                panic!("{sections:?}");
            };
            let want = (height * (2.0 - height)).sqrt();
            assert!(
                (circle.radius() - want).abs() < 1e-3 * want,
                "circle at z = {height}: radius {}, want {want}",
                circle.radius()
            );
            assert!((circle.center().z() - height).abs() < 1e-15);
        }
    }

    /// The farthest any of `curves` strays from `off`, sampled densely.
    fn farthest(curves: &[IntersectionCurve], off: impl Fn(Point3) -> f64) -> f64 {
        let mut far = 0.0_f64;
        for ic in curves {
            let (t0, t1) = ic.curve.domain();
            for i in 0..=2000 {
                let t = t0 + (t1 - t0) * f64::from(i) / 2000.0;
                far = far.max(off(ic.curve.evaluate(t)));
            }
        }
        far
    }

    /// A ball sitting in a bore of its own radius touches it along its
    /// equator, which comes back as the exact circle.
    #[test]
    fn a_ball_in_its_bore_meets_it_on_an_exact_circle() {
        let ball = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let bore =
            CylindricalSurface::new(Point3::new(0.0, 0.0, -2.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Sphere(&ball),
            AnalyticSurface::Cylinder(&bore),
            64,
            None,
            Some((0.0, 4.0)),
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        let far = farthest(&curves, |p| {
            let on_ball = ((p - Point3::new(0.0, 0.0, 0.0)).length() - 1.0).abs();
            on_ball
                .max((p.x().hypot(p.y()) - 1.0).abs())
                .max(p.z().abs())
        });
        assert!(far < 1e-14, "circle strays {far:.2e}");
    }

    /// Two unit balls a unit apart meet in a circle of radius √3/2, which
    /// comes back exact.
    #[test]
    fn overlapping_balls_meet_on_an_exact_circle() {
        let a = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let b = SphericalSurface::new(Point3::new(1.0, 0.0, 0.0), 1.0).unwrap();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Sphere(&a),
            AnalyticSurface::Sphere(&b),
            64,
            None,
            None,
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        let far = farthest(&curves, |p| {
            let on_a = ((p - Point3::new(0.0, 0.0, 0.0)).length() - 1.0).abs();
            on_a.max(((p - Point3::new(1.0, 0.0, 0.0)).length() - 1.0).abs())
        });
        assert!(far < 1e-14, "circle strays {far:.2e}");
    }

    /// A loop with semi-axes 3 and 1.5 in a tilted frame lies on its ellipse
    /// all the way round and starts at the end of its major axis.
    #[test]
    fn a_conic_loop_is_its_ellipse() {
        let centre = Point3::new(1.0, -2.0, 0.5);
        let frame = Frame3::from_normal(centre, Vec3::new(0.3, -0.4, 1.0)).unwrap();
        let curve = conic_loop(centre, frame.x, frame.y, 3.0, 1.5).unwrap();
        let (t0, t1) = curve.domain();
        assert!((curve.evaluate(t0) - (centre + frame.x * 3.0)).length() < 1e-15);
        for i in 0..=2000 {
            let p = curve.evaluate(t0 + (t1 - t0) * f64::from(i) / 2000.0) - centre;
            let (x, y) = (p.dot(frame.x) / 3.0, p.dot(frame.y) / 1.5);
            assert!(
                (x.hypot(y) - 1.0).abs() < 1e-14,
                "off its ellipse by {}",
                x.hypot(y) - 1.0
            );
            assert!(p.dot(frame.z).abs() < 1e-14);
        }
    }

    /// A ball sitting in a ring's hole, and a ring resting on a wider one,
    /// meet them along the one circle where they touch, and the bounded
    /// call returns it exactly.
    #[test]
    fn touching_coaxial_rings_meet_along_one_circle() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let x = Vec3::new(1.0, 0.0, 0.0);
        let ring =
            ToroidalSurface::with_axis_and_ref_dir(Point3::new(0.0, 0.0, 0.0), 3.0, 1.0, z, x)
                .unwrap();
        let ball = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 2.0).unwrap();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Sphere(&ball),
            AnalyticSurface::Torus(&ring),
            64,
            None,
            None,
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        let far = farthest(&curves, |p| {
            let q = p - Point3::new(0.0, 0.0, 0.0);
            let rho = q.x().hypot(q.y());
            let on_ring = ((rho - 3.0).hypot(q.z()) - 1.0).abs();
            on_ring.max((q.length() - 2.0).abs())
        });
        assert!(far < 1e-14, "circle strays {far:.2e}");

        // Tubes of radius 1 and 0.5 on rings of radius 3 and 3.9, the second
        // raised 1.2: their centres lie 1.5 apart, so they touch a third of
        // the way along, at height 0.8 on a circle of radius 3.6.
        let wider =
            ToroidalSurface::with_axis_and_ref_dir(Point3::new(0.0, 0.0, 1.2), 3.9, 0.5, z, x)
                .unwrap();
        let touching = exact_torus_torus(&ring, &wider).unwrap().unwrap();
        let circles = circles_of(&touching);
        assert_eq!(circles.len(), 1);
        assert!(
            (circles[0].radius() - 3.6).abs() < 1e-12,
            "{}",
            circles[0].radius()
        );
        assert!((circles[0].center().z() - 0.8).abs() < 1e-12);
    }

    /// Unit tubes on rings of radius 1e6 and 1e6 + 1.999 overlap by 0.001
    /// and cross in two circles 0.063 apart; on rings 2.001 apart they miss.
    /// Neither is read as a touch, at a scale where a millionth of it is
    /// two thousandths.
    #[test]
    fn huge_coaxial_rings_cross_or_miss_rather_than_touch() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let x = Vec3::new(1.0, 0.0, 0.0);
        let ring = |big: f64| {
            ToroidalSurface::with_axis_and_ref_dir(Point3::new(0.0, 0.0, 0.0), big, 1.0, z, x)
                .unwrap()
        };
        let crossing = exact_torus_torus(&ring(1e6), &ring(1_000_001.999))
            .unwrap()
            .unwrap();
        let circles = circles_of(&crossing);
        assert_eq!(circles.len(), 2);
        for circle in circles {
            let height = (1.0 - 0.9995_f64 * 0.9995).sqrt();
            assert!(
                (circle.center().z().abs() - height).abs() < 1e-9,
                "{}",
                circle.center().z()
            );
        }
        let apart = exact_torus_torus(&ring(1e6), &ring(1_000_002.001))
            .unwrap()
            .unwrap();
        assert!(apart.is_empty());
    }

    /// A plane parallel to a cone's axis cuts the branch of a hyperbola on
    /// the cone's one nappe; planes parallel to a generator cut a parabola
    /// on one side of the apex and miss the cone on the other.
    #[test]
    fn a_plane_cuts_a_cones_open_conics() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let on_cone = |p: Point3, slope: f64| (p.z() - slope * p.x().hypot(p.y())).abs();
        let steep =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, 60_f64.to_radians()).unwrap();
        let slope = 60_f64.to_radians().tan();
        let along =
            exact_plane_section(AnalyticSurface::Cone(&steep), Vec3::new(1.0, 0.0, 0.0), 1.0)
                .unwrap();
        let [PlaneSection::Hyperbola(h)] = along.as_slice() else {
            unreachable!("{along:?}")
        };
        for i in -20..=20 {
            let p = h.evaluate(f64::from(i) / 5.0);
            assert!((p.x() - 1.0).abs() < 1e-12 && p.z() > 0.0, "{p:?}");
            assert!(on_cone(p, slope) < 1e-12 * p.z().max(1.0), "{p:?}");
        }
        let even =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, 45_f64.to_radians()).unwrap();
        let tilt = Vec3::new(-1.0, 0.0, 1.0) * std::f64::consts::FRAC_1_SQRT_2;
        let above = exact_plane_section(AnalyticSurface::Cone(&even), tilt, 1.0).unwrap();
        let [PlaneSection::Parabola(parabola)] = above.as_slice() else {
            unreachable!("{above:?}")
        };
        for i in -20..=20 {
            let p = parabola.evaluate(f64::from(i) / 2.0);
            assert!(
                (tilt.dot(p - Point3::new(0.0, 0.0, 0.0)) - 1.0).abs() < 1e-12,
                "{p:?}"
            );
            assert!(
                p.z() > 0.0 && on_cone(p, 1.0) < 1e-12 * p.z().max(1.0),
                "{p:?}"
            );
        }
        let below = exact_plane_section(AnalyticSurface::Cone(&even), tilt, -1.0).unwrap();
        assert!(below.is_empty(), "{below:?}");
    }

    /// A cone so flat it rises 1e-7 from its radial plane, cut by a plane
    /// tilted 7e-7 off it, and a cone so narrow it opens 5e-7 off its axis,
    /// cut by a plane parallel to the axis, give sections that lie on the
    /// cone: a hyperbola for the flat one, read from its eccentricity, and
    /// sampled points where the formulas lose their digits.
    #[test]
    fn extreme_cones_give_sections_on_the_cone() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let apex = Point3::new(0.0, 0.0, 0.0);
        let holds = |cone: &ConicalSurface, q: Point3| {
            let w = q - apex;
            (w.dot(up) - cone.half_angle().sin() * w.length()).abs() <= 1e-6 * (1.0 + w.length())
        };
        let flat = ConicalSurface::new(apex, up, 1e-7).unwrap();
        let tilt = Vec3::new(5e-13_f64.sqrt(), 0.0, (1.0 - 5e-13_f64).sqrt());
        let needle = ConicalSurface::new(apex, up, FRAC_PI_2 - 5e-7).unwrap();
        for (cone, normal, d) in [(&flat, tilt, 1.0), (&needle, Vec3::new(1.0, 0.0, 0.0), 1.0)] {
            let sections = exact_plane_section(AnalyticSurface::Cone(cone), normal, d).unwrap();
            assert!(!sections.is_empty());
            for section in &sections {
                let points: Vec<Point3> = match section {
                    PlaneSection::Hyperbola(h) => {
                        (-4..=4).map(|i| h.evaluate(f64::from(i) / 4.0)).collect()
                    }
                    PlaneSection::Parabola(q) => {
                        (-4..=4).map(|i| q.evaluate(f64::from(i))).collect()
                    }
                    PlaneSection::Points(points) => points.clone(),
                    other => unreachable!("{other:?}"),
                };
                for q in points {
                    assert!(holds(cone, q), "{q:?} off the cone");
                }
            }
        }
        let flat_section = exact_plane_section(AnalyticSurface::Cone(&flat), tilt, 1.0).unwrap();
        assert!(matches!(
            flat_section.as_slice(),
            [PlaneSection::Hyperbola(_)]
        ));
    }

    /// The rulings two parallel cylinders meet along, from the bounded call
    /// with each face's `v` range: each ruling's ends, sorted by height.
    fn parallel_rulings(
        (c1, r1, v1): (Point3, f64, (f64, f64)),
        (c2, r2, axis2, v2): (Point3, f64, Vec3, (f64, f64)),
    ) -> Vec<(Point3, Point3)> {
        rulings_within((c1, r1, v1), (c2, r2, axis2, v2), 1e-12)
    }

    /// [`parallel_rulings`] with each curve held to `tolerance` of both
    /// cylinders.
    fn rulings_within(
        (c1, r1, v1): (Point3, f64, (f64, f64)),
        (c2, r2, axis2, v2): (Point3, f64, Vec3, (f64, f64)),
        tolerance: f64,
    ) -> Vec<(Point3, Point3)> {
        let a = CylindricalSurface::new(c1, Vec3::new(0.0, 0.0, 1.0), r1).unwrap();
        let b = CylindricalSurface::new(c2, axis2, r2).unwrap();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Cylinder(&a),
            AnalyticSurface::Cylinder(&b),
            32,
            Some(v1),
            Some(v2),
        )
        .unwrap();
        curves
            .iter()
            .map(|ic| {
                let (t0, t1) = ic.curve.domain();
                let (p, q) = (ic.curve.evaluate(t0), ic.curve.evaluate(t1));
                for i in 0..=100 {
                    let x = ic.curve.evaluate(t0 + (t1 - t0) * f64::from(i) / 100.0);
                    let on_a = ((x - c1) - Vec3::new(0.0, 0.0, (x - c1).z())).length() - r1;
                    let w = x - c2;
                    let on_b = (w - axis2 * w.dot(axis2)).length() - r2;
                    assert!(
                        on_a.abs() < tolerance && on_b.abs() < tolerance,
                        "{x:?} off a cylinder by {on_a:.1e}, {on_b:.1e}"
                    );
                }
                if p.z() <= q.z() { (p, q) } else { (q, p) }
            })
            .collect()
    }

    /// Rods of radius 1 with axes 1.2 apart overlap along two rulings, at
    /// x 0.6 and y ±0.8, over the height both faces share; the second rod's
    /// axis runs down, so its range counts down from its origin.
    #[test]
    fn overlapping_parallel_rods_meet_along_two_rulings() {
        let mut rulings = parallel_rulings(
            (Point3::new(0.0, 0.0, 0.0), 1.0, (0.0, 4.0)),
            (
                Point3::new(1.2, 0.0, 3.0),
                1.0,
                Vec3::new(0.0, 0.0, -1.0),
                (0.0, 2.0),
            ),
        );
        rulings.sort_by(|a, b| a.0.y().total_cmp(&b.0.y()));
        assert_eq!(rulings.len(), 2);
        for ((lo, hi), y) in rulings.into_iter().zip([-0.8, 0.8]) {
            assert!((lo - Point3::new(0.6, y, 1.0)).length() < 1e-12, "{lo:?}");
            assert!((hi - Point3::new(0.6, y, 3.0)).length() < 1e-12, "{hi:?}");
        }
    }

    /// Rods touching side by side, or one inside another against its wall,
    /// meet along the one ruling where they touch; rods apart do not meet.
    #[test]
    fn touching_parallel_rods_meet_along_one_ruling() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let beside = parallel_rulings(
            (Point3::new(0.0, 0.0, 0.0), 1.0, (-2.0, 2.0)),
            (Point3::new(2.0, 0.0, -2.0), 1.0, up, (0.0, 4.0)),
        );
        assert_eq!(beside.len(), 1);
        assert!((beside[0].0 - Point3::new(1.0, 0.0, -2.0)).length() < 1e-12);
        assert!((beside[0].1 - Point3::new(1.0, 0.0, 2.0)).length() < 1e-12);
        let inside = parallel_rulings(
            (Point3::new(0.0, 0.0, 0.0), 2.0, (0.0, 1.0)),
            (Point3::new(-1.0, 0.0, 0.0), 1.0, up, (0.0, 1.0)),
        );
        assert_eq!(inside.len(), 1);
        assert!((inside[0].0 - Point3::new(-2.0, 0.0, 0.0)).length() < 1e-12);
        let apart = parallel_rulings(
            (Point3::new(0.0, 0.0, 0.0), 1.0, (0.0, 1.0)),
            (Point3::new(2.5, 0.0, 0.0), 1.0, up, (0.0, 1.0)),
        );
        assert!(apart.is_empty());
    }

    /// Rods whose axes lean 1e-5 apart drift 1e-3 over a span of 100, so
    /// the rulings defer to the next solver there; over a span of 0.001
    /// they drift 1e-8 and keep their two rulings.
    #[test]
    fn rods_leaning_apart_keep_rulings_only_over_a_short_span() {
        let up = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
            .unwrap();
        let lean = Vec3::new(1e-5_f64.sin(), 0.0, 1e-5_f64.cos());
        let leaning = CylindricalSurface::new(Point3::new(1.2, 0.0, 0.0), lean, 1.0).unwrap();
        let span = |h: f64| parallel_cylinders(&up, &leaning, Some((0.0, h)), Some((0.0, h)));
        assert!(span(100.0).unwrap().is_none());
        assert_eq!(span(0.001).unwrap().map(|r| r.len()), Some(2));
    }

    /// Huge rods a short stretch long still meet along both rulings, and
    /// rods of radius 100 whose axes lie 3e-7 short of touching meet along
    /// two rulings 0.011 apart, not the one where they would touch.
    #[test]
    fn parallel_rods_keep_short_and_close_rulings() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let huge = rulings_within(
            (Point3::new(0.0, 0.0, 0.0), 1e6, (0.0, 0.001)),
            (Point3::new(1.2e6, 0.0, 0.0), 1e6, up, (0.0, 0.001)),
            1e-9,
        );
        assert_eq!(huge.len(), 2);
        assert!(
            huge.iter()
                .all(|(lo, hi)| ((*hi - *lo).length() - 0.001).abs() < 1e-12)
        );
        let mut close = rulings_within(
            (Point3::new(0.0, 0.0, 0.0), 100.0, (0.0, 1.0)),
            (Point3::new(199.999_999_7, 0.0, 0.0), 100.0, up, (0.0, 1.0)),
            1e-9,
        );
        assert_eq!(close.len(), 2);
        close.sort_by(|a, b| a.0.y().total_cmp(&b.0.y()));
        let apart = close[1].0.y() - close[0].0.y();
        assert!((apart - 0.010_954).abs() < 1e-5, "{apart}");
    }

    /// How far `p` lies off `cone` (either nappe), and its `v` there.
    fn off_cone(cone: &ConicalSurface, p: Point3) -> (f64, f64) {
        let q = p - cone.apex();
        let (along, s) = (q.dot(cone.axis()), cone.half_angle().sin());
        ((along.abs() - s * q.length()).abs(), along / s)
    }

    /// Cones sharing an apex, their axes 40° apart and each opening 20°
    /// from its axis, touch along the one ruling between the axes, which
    /// runs where both faces hold it.
    #[test]
    fn cones_sharing_an_apex_touch_along_one_ruling() {
        let apex = Point3::new(1.0, 2.0, 3.0);
        let a = 20.0_f64.to_radians();
        let half = std::f64::consts::FRAC_PI_2 - a;
        let c1 = ConicalSurface::new(apex, Vec3::new(a.cos(), 0.0, a.sin()), half).unwrap();
        let c2 = ConicalSurface::new(apex, Vec3::new(a.cos(), 0.0, -a.sin()), half).unwrap();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Cone(&c1),
            AnalyticSurface::Cone(&c2),
            64,
            Some((0.2, 3.0)),
            Some((0.5, 4.0)),
        )
        .unwrap();
        assert_eq!(curves.len(), 1);
        let curve = &curves[0].curve;
        let (t0, t1) = curve.domain();
        let x = Vec3::new(1.0, 0.0, 0.0);
        assert!((curve.evaluate(t0) - (apex + x * 0.5)).length() < 1e-12);
        assert!((curve.evaluate(t1) - (apex + x * 3.0)).length() < 1e-12);
    }

    /// Cones sharing an apex, their axes 20° apart and each opening 30°,
    /// cross along two rulings; faces reaching past the apex hold each on
    /// both sides of it, in two segments that end there.
    #[test]
    fn cones_sharing_an_apex_cross_along_two_rulings() {
        let apex = Point3::new(-1.0, 0.5, 2.0);
        let b = 20.0_f64.to_radians();
        let half = 60.0_f64.to_radians();
        let c1 = ConicalSurface::new(apex, Vec3::new(0.0, 0.0, 1.0), half).unwrap();
        let c2 = ConicalSurface::new(apex, Vec3::new(b.sin(), 0.0, b.cos()), half).unwrap();
        for (range, pieces) in [((0.5, 2.0), 2), ((-2.0, 2.0), 4)] {
            let curves = intersect_analytic_analytic_bounded(
                AnalyticSurface::Cone(&c1),
                AnalyticSurface::Cone(&c2),
                64,
                Some(range),
                Some(range),
            )
            .unwrap();
            assert_eq!(curves.len(), pieces, "{range:?}");
            for ic in &curves {
                let (t0, t1) = ic.curve.domain();
                for k in 0..=20 {
                    let p = ic.curve.evaluate(t0 + (t1 - t0) * f64::from(k) / 20.0);
                    for cone in [&c1, &c2] {
                        let (off, v) = off_cone(cone, p);
                        assert!(off < 1e-12, "{range:?}: off a cone by {off:.2e}");
                        assert!(v > range.0 - 1e-12 && v < range.1 + 1e-12);
                    }
                }
                if pieces == 4 {
                    let ends = [ic.curve.evaluate(t0), ic.curve.evaluate(t1)];
                    assert!(ends.iter().any(|&e| (e - apex).length() < 1e-12));
                }
                for point in &ic.points {
                    let (on1, on2) = (point.param1, point.param2);
                    assert!((c1.evaluate(on1.0, on1.1) - point.point).length() < 1e-9);
                    assert!((c2.evaluate(on2.0, on2.1) - point.point).length() < 1e-9);
                }
            }
        }
    }

    /// Axes 1e-8 apart in angle leave a ruling's part across them to
    /// rounding, and cones touching within rounding along a ruling leave
    /// how far apart their rulings may be 1e6 out to it: neither is solved
    /// here.
    #[test]
    fn cones_sharing_an_apex_defer_what_rounding_cannot_settle() {
        let apex = Point3::new(0.0, 0.0, 0.0);
        let half = 60.0_f64.to_radians();
        let c1 = ConicalSurface::new(apex, Vec3::new(0.0, 0.0, 1.0), half).unwrap();
        let c2 = ConicalSurface::new(apex, Vec3::new(1e-8_f64.sin(), 0.0, 1e-8_f64.cos()), half)
            .unwrap();
        let near = Some((0.5, 2.0));
        assert!(shared_apex_cones(&c1, &c2, near, near).unwrap().is_none());
        let a = 20.0_f64.to_radians();
        let half = std::f64::consts::FRAC_PI_2 - a;
        let c1 = ConicalSurface::new(apex, Vec3::new(a.cos(), 0.0, a.sin()), half).unwrap();
        let c2 = ConicalSurface::new(apex, Vec3::new(a.cos(), 0.0, -a.sin()), half).unwrap();
        let far = Some((0.5, 1e6));
        assert!(shared_apex_cones(&c1, &c2, far, far).unwrap().is_none());
    }

    /// Perpendicular cones opening 1e-15 short of touching cross along two
    /// rulings, which part by about 0.09 a million out: two segments.
    #[test]
    fn cones_sharing_an_apex_just_short_of_touching_cross_in_two_rulings() {
        let apex = Point3::new(0.0, 0.0, 0.0);
        let half = std::f64::consts::FRAC_PI_4 - 1e-15;
        let c1 = ConicalSurface::new(apex, Vec3::new(0.0, 0.0, 1.0), half).unwrap();
        let c2 = ConicalSurface::new(apex, Vec3::new(1.0, 0.0, 0.0), half).unwrap();
        let range = Some((0.5, 1e6));
        let curves = shared_apex_cones(&c1, &c2, range, range).unwrap().unwrap();
        assert_eq!(curves.len(), 2);
        let ends: Vec<Point3> = curves
            .iter()
            .map(|ic| ic.curve.evaluate(ic.curve.domain().1))
            .collect();
        assert!((ends[0] - ends[1]).length() > 0.05);
    }

    /// A ball off a ring's axis touches it at one point from outside the
    /// tube, from inside it, or holding the whole ring, and that point is
    /// all they share. A ball reaching 1e-6 further cuts the tube in a small
    /// loop, which is left to the section's closed form, and one centred on
    /// the core circle with the tube's radius meets it along the meridian
    /// circle through its centre.
    #[test]
    fn a_ball_off_a_rings_axis_touches_it_at_one_point() {
        let o = Point3::new(1.0, 2.0, 3.0);
        let ring = ToroidalSurface::new(o, 3.0, 1.0).unwrap();
        let (s, c) = 30.0_f64.to_radians().sin_cos();
        let (out, up) = (Vec3::new(c, s, 0.0), Vec3::new(0.0, 0.0, 1.0));
        let on_ring = |p: Point3| {
            let q = p - o;
            ((q.x().hypot(q.y()) - 3.0).hypot(q.z()) - 1.0).abs()
        };
        let balls = [
            (o + out * 5.0, 1.0),
            (o + out * 3.2 + up * 0.1, 1.0 - 0.05_f64.sqrt()),
            (o + out + up * 0.5, 1.0 + 16.25_f64.sqrt()),
        ];
        let mut touches = Vec::new();
        for (centre, radius) in balls {
            let ball = SphericalSurface::new(centre, radius).unwrap();
            let exacts = exact_sphere_torus(&ball, &ring).unwrap().unwrap();
            let p = match exacts.as_slice() {
                [ExactIntersectionCurve::Points(points)] => points[0],
                _ => unreachable!("one touching point"),
            };
            assert!(on_ring(p) < 1e-12 && ((p - centre).length() - radius).abs() < 1e-12);
            let curves = intersect_analytic_analytic(
                AnalyticSurface::Sphere(&ball),
                AnalyticSurface::Torus(&ring),
                16,
            )
            .unwrap();
            assert_eq!(curves.len(), 1);
            let (t0, t1) = curves[0].curve.domain();
            assert!((curves[0].curve.evaluate(t0) - p).length() < 1e-15);
            assert!((curves[0].curve.evaluate(t1) - p).length() < 1e-15);
            touches.push(p);
        }
        assert!((touches[0] - (o + out * 4.0)).length() < 1e-12);
        let past = SphericalSurface::new(o + out * 5.0, 1.0 + 1e-6).unwrap();
        assert!(exact_sphere_torus(&past, &ring).unwrap().is_none());
        let on_core = SphericalSurface::new(o + out * 3.0, 1.0).unwrap();
        let meridian = exact_sphere_torus(&on_core, &ring).unwrap().unwrap();
        let [ExactIntersectionCurve::Circle(circle)] = meridian.as_slice() else {
            unreachable!("one meridian circle");
        };
        assert!((circle.center() - (o + out * 3.0)).length() < 1e-12);
        assert!((circle.radius() - 1.0).abs() < 1e-12);
    }

    /// A taper pin through a rod turned `spin` about its axis, as
    /// `a_taper_pin_through_a_turned_rod_traces_both_loops` builds it: the pin,
    /// its faces' `v` range, and its apex's `y`.
    fn taper_pin_and_rod(
        flip: bool,
        spin: f64,
    ) -> (ConicalSurface, (f64, f64), f64, CylindricalSurface) {
        let slope = 0.7_f64 / 16.0;
        let half = (1.0 / slope).atan();
        let (apex_y, axis_y) = if flip {
            (-8.0 - 0.8 / slope, 1.0)
        } else {
            (-8.0 + 1.5 / slope, -1.0)
        };
        let pin = ConicalSurface::new(
            Point3::new(0.0, apex_y, 10.0),
            Vec3::new(0.0, axis_y, 0.0),
            half,
        )
        .unwrap();
        let (h0, h1) = ((-8.0 - apex_y).abs(), (8.0 - apex_y).abs());
        let v_pin = (h0.min(h1) / half.sin(), h0.max(h1) / half.sin());
        let rod = CylindricalSurface::with_ref_dir(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            Vec3::new(spin.cos(), spin.sin(), 0.0),
        )
        .unwrap();
        (pin, v_pin, apex_y, rod)
    }

    /// The overlap of the taper pin's and the rod's faces, a little wide.
    const TAPER_PIN_REGION: Aabb3 = Aabb3 {
        min: Point3::new(-1.6, -5.1, 8.4),
        max: Point3::new(1.6, 5.1, 11.6),
    };

    /// The farthest that `n` points of `curve` lie off a surface, by `off`.
    fn farthest_along(curve: &NurbsCurve, n: u32, off: impl Fn(Point3) -> f64) -> f64 {
        let (t0, t1) = curve.domain();
        (0..=n)
            .map(|k| off(curve.evaluate((t1 - t0).madd(f64::from(k) / f64::from(n), t0))))
            .fold(0.0, f64::max)
    }

    /// The general marcher traces a taper pin's two loops on a rod (which the
    /// rulings trace, so the marcher is called on its own here), a cone and a
    /// tilted tube, and a cone's band across a wide cylinder. Its cubic fits
    /// stay within the linear tolerance of both surfaces between their
    /// points, not only at them: fits through the marched points alone lay
    /// 2e-5 to 1.1e-4 off both, with the points 0.08 to 0.31 apart.
    #[test]
    fn marched_cone_cylinder_sections_stay_on_both_between_their_points() {
        let lin = Tolerance::new().linear;
        for flip in [false, true] {
            for spin in [0.0_f64, 0.3, 0.9, 1.4, 2.0, 2.6, 3.0, 3.6] {
                let (pin, v_pin, apex_y, rod) = taper_pin_and_rod(flip, spin);
                let half = pin.half_angle();
                let off_pin = |p: Point3| {
                    let rho = p.x().hypot(p.z() - 10.0);
                    let h = (p.y() - apex_y).abs();
                    rho.madd(half.sin(), -h * half.cos()).abs()
                };
                let off_rod = |p: Point3| (p.x().hypot(p.y()) - 5.0).abs();
                for rod_first in [true, false] {
                    let (a, b, va, vb) = if rod_first {
                        (
                            AnalyticSurface::Cylinder(&rod),
                            AnalyticSurface::Cone(&pin),
                            Some((0.0, 20.0)),
                            Some(v_pin),
                        )
                    } else {
                        (
                            AnalyticSurface::Cone(&pin),
                            AnalyticSurface::Cylinder(&rod),
                            Some(v_pin),
                            Some((0.0, 20.0)),
                        )
                    };
                    let curves =
                        marched_sections(a, b, 32, va, vb, Some(TAPER_PIN_REGION)).unwrap();
                    assert_eq!(curves.len(), 2, "spin {spin} flip {flip}: loops");
                    for c in &curves {
                        let worst = farthest_along(&c.curve, 4000, |p| off_pin(p).max(off_rod(p)));
                        assert!(worst <= lin, "spin {spin} flip {flip}: off by {worst:.2e}");
                    }
                }
            }
        }

        let (cone, tube) = cone_and_tilted_tube();
        let curves = intersect_analytic_analytic_bounded(
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&tube),
            32,
            Some((0.5, 4.0)),
            Some((-5.0, 5.0)),
        )
        .unwrap();
        assert!(!curves.is_empty());
        let (sin, cos) = cone.half_angle().sin_cos();
        let axis = tube.axis();
        for c in &curves {
            let worst = farthest_along(&c.curve, 4000, |p| {
                let on_cone = p.x().hypot(p.y()).madd(sin, -p.z().abs() * cos).abs();
                let w = p - tube.origin();
                let on_tube = ((w - axis * axis.dot(w)).length() - 0.1).abs();
                on_cone.max(on_tube)
            });
            assert!(worst <= lin, "tilted tube: off by {worst:.2e}");
        }

        let half_angle = 14.0_f64.atan2(10.0);
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 273.2),
            Vec3::new(0.0, 0.0, -1.0),
            half_angle,
        )
        .unwrap();
        let cyl = CylindricalSurface::new(
            Point3::new(0.0, -50.0, 215.0),
            Vec3::new(0.0, 1.0, 0.0),
            60.0,
        )
        .unwrap();
        let curves = intersect_analytic_analytic_in_region(
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&cyl),
            32,
            Some((81.2 / half_angle.sin(), 95.2 / half_angle.sin())),
            Some((10.0, 90.0)),
            Aabb3 {
                min: Point3::new(-59.98, -40.0, 178.0),
                max: Point3::new(-47.23, 40.0, 192.0),
            },
        )
        .unwrap();
        assert_eq!(curves.len(), 2);
        let (sin, cos) = half_angle.sin_cos();
        for c in &curves {
            let worst = farthest_along(&c.curve, 4000, |p| {
                let on_cone = p.x().hypot(p.y()).madd(sin, -(273.2 - p.z()) * cos).abs();
                let on_cyl = (p.x().hypot(p.z() - 215.0) - 60.0).abs();
                on_cone.max(on_cyl)
            });
            assert!(worst <= lin, "band: off by {worst:.2e}");
        }
    }

    /// Between the taper pin's two holes, the pin's side and the rod's wall
    /// pass closest without meeting. Projections from one to the other
    /// settle there, both normals along the gap: with the rod turned 0.9, at
    /// (-2.50, -3.35, 9.88), 0.82 off the rod. The march from that seed never
    /// came back to it: it ran 1001 points and ended 2.6 from where it began.
    /// The seed now moves onto both surfaces by Newton steps before the march,
    /// or is dropped where they cannot take it there, and every loop closes
    /// on itself.
    #[test]
    fn a_seed_where_two_walls_pass_closest_starts_no_march() {
        let (pin, v_pin, _, rod) = taper_pin_and_rod(false, 0.9);
        let (a, b) = (AnalyticSurface::Cylinder(&rod), AnalyticSurface::Cone(&pin));
        let stalled = Point3::new(
            -2.503_835_657_485_280_5,
            -3.349_109_255_073_426,
            9.881_016_348_226_973,
        );
        let off_rod = |p: Point3| section_fit::Carrier::distance(&a, p).unwrap().0;
        let off_pin = |p: Point3| section_fit::Carrier::distance(&b, p).unwrap().0;
        assert!(off_rod(stalled).abs() > 0.8);
        let on = settle_on_section(&a, &b, stalled).unwrap();
        let (d_rod, d_pin) = (off_rod(on), off_pin(on));
        assert!(
            d_rod.abs() < 1e-12 && d_pin.abs() < 1e-12,
            "{d_rod:.1e}, {d_pin:.1e}"
        );

        let curves = marched_sections(
            a,
            b,
            32,
            Some((0.0, 20.0)),
            Some(v_pin),
            Some(TAPER_PIN_REGION),
        )
        .unwrap();
        assert_eq!(curves.len(), 2);
        for c in &curves {
            let (first, last) = (c.points[0].point, c.points[c.points.len() - 1].point);
            assert!((first - last).length() < 1e-9, "{first:?} to {last:?}");
            assert!(c.points.len() < 400, "{} points", c.points.len());
        }

        // Where both normals lie along the gap, no Newton step crosses it.
        let ball = SphericalSurface::new(Point3::new(3.0, 0.0, 0.0), 1.0).unwrap();
        let post =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let gap = Point3::new(1.5, 0.0, 0.0);
        assert!(
            settle_on_section(
                &AnalyticSurface::Sphere(&ball),
                &AnalyticSurface::Cylinder(&post),
                gap
            )
            .is_none()
        );
    }

    /// Two rods whose axes cross square, as far apart as their radii add up
    /// to, touch at one point and cross nowhere. A seed settles onto that
    /// point, where no direction along a section is defined: the march from
    /// it wandered up to 1.3e-5 off both rods, and its fit was refused. The
    /// seed is now dropped, and no section is traced.
    #[test]
    fn rods_that_only_touch_trace_no_section() {
        let wall =
            CylindricalSurface::new(Point3::new(1.1, 1.1, 1.2), Vec3::new(0.0, 0.0, 1.0), 2.55)
                .unwrap();
        let air =
            CylindricalSurface::new(Point3::new(-3.9, 1.1, 4.7), Vec3::new(0.0, 1.0, 0.0), 2.45)
                .unwrap();
        let (a, b) = (
            AnalyticSurface::Cylinder(&wall),
            AnalyticSurface::Cylinder(&air),
        );
        let touch = Point3::new(-1.45, 1.1, 4.7);
        assert!(!crosses_at(&a, &b, touch));
        let region = Aabb3 {
            min: Point3::new(-2.1625, 0.3875, 1.5375),
            max: Point3::new(-0.7375, 1.8125, 5.4125),
        };
        let curves = marched_sections(
            a,
            b,
            32,
            Some((1.05, 12.45)),
            Some((0.0, 30.9)),
            Some(region),
        )
        .unwrap();
        assert!(curves.is_empty(), "{} sections", curves.len());
    }

    /// The distance from the double cone `cone`, by the ruling on `p`'s side
    /// of the axis on `p`'s nappe.
    fn off_double_cone(cone: &ConicalSurface, p: Point3) -> f64 {
        let w = p - cone.apex();
        let h = cone.axis().dot(w);
        let rho = (w - cone.axis() * h).length();
        let (sin, cos) = cone.half_angle().sin_cos();
        rho.madd(sin, -h.abs() * cos).abs()
    }

    /// Two cones of one axis direction, a unit apart, bounded on their
    /// negative nappes between `z = -1` and `z = -1/3`, meet in one loop
    /// there (through `(0.125, sqrt(15)/8, -0.5)`). The marcher traces it
    /// either way round, and the fit stays on both nappes between its
    /// points. Read by the ruling of the positive nappe, the shared point lay
    /// 0.707 off the first cone, and no seed settled on the loop.
    #[test]
    fn negative_nappe_sections_are_marched_on_both_cones() {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let wide = ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), up, std::f64::consts::FRAC_PI_4)
            .unwrap();
        let narrow = ConicalSurface::new(Point3::new(1.0, 0.0, 0.0), up, 0.5_f64.atan()).unwrap();
        let shared = Point3::new(0.125, 15.0_f64.sqrt() / 8.0, -0.5);
        for cone in [&wide, &narrow] {
            let (off, _) =
                section_fit::Carrier::distance(&AnalyticSurface::Cone(cone), shared).unwrap();
            assert!(off.abs() < 1e-12, "{off}");
        }
        let (r1, r2) = (std::f64::consts::SQRT_2, 5.0_f64.sqrt());
        let (v_wide, v_narrow) = ((-r1, -r1 / 3.0), (-r2, -r2 / 3.0));
        for wide_first in [true, false] {
            let (a, b, va, vb) = if wide_first {
                (
                    AnalyticSurface::Cone(&wide),
                    AnalyticSurface::Cone(&narrow),
                    v_wide,
                    v_narrow,
                )
            } else {
                (
                    AnalyticSurface::Cone(&narrow),
                    AnalyticSurface::Cone(&wide),
                    v_narrow,
                    v_wide,
                )
            };
            let curves = marched_sections(a, b, 32, Some(va), Some(vb), None).unwrap();
            assert!(!curves.is_empty(), "wide first {wide_first}: no section");
            for c in &curves {
                let worst = farthest_along(&c.curve, 4000, |p| {
                    off_double_cone(&wide, p).max(off_double_cone(&narrow, p))
                });
                assert!(worst <= 1e-7, "wide first {wide_first}: off by {worst:.2e}");
                let below = farthest_along(&c.curve, 4000, |p| (p.z() + 1.0 / 3.0).max(0.0));
                assert!(
                    below < 1e-9,
                    "wide first {wide_first}: above z = -1/3 by {below:.2e}"
                );
            }
        }
    }

    /// A 45-degree cone holds its negative ruling from (1, 0, -1) to
    /// (4, 0, -4), and so does a rod of radius 1 along it. The bound on the
    /// cone reads that ruling as on it, and the fit through its points is
    /// the ruling; read by the positive nappe, its factor was negative and
    /// no span could be bounded. A span through the apex, where the factor
    /// vanishes, has no bound.
    #[test]
    fn a_fit_along_a_negative_ruling_is_bounded() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let along = Vec3::new(1.0, 0.0, -1.0) * std::f64::consts::FRAC_1_SQRT_2;
        let rod = CylindricalSurface::new(Point3::new(0.0, 1.0, 0.0), along, 1.0).unwrap();
        let (a, b) = (
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Cylinder(&rod),
        );
        let ruling: Vec<Point3> = (1..=4)
            .map(|k| Point3::new(f64::from(k), 0.0, -f64::from(k)))
            .collect();
        let ctrl = [ruling[0], ruling[1], ruling[2], ruling[3]];
        let bound = section_fit::Carrier::bound(&a, &ctrl).unwrap();
        assert!(bound < 1e-12, "{bound:.2e}");
        let (curve, points) = fit_section(&a, &b, &ruling, 1e-7).unwrap();
        assert_eq!(points.len(), 4);
        let worst = farthest_along(&curve, 1000, |p| p.y().abs().max((p.x() + p.z()).abs()));
        assert!(worst < 1e-12, "{worst:.2e}");
        let through_apex = [
            Point3::new(-1.0, 0.0, 1.0),
            Point3::new(-1.0 / 3.0, 0.0, 1.0 / 3.0),
            Point3::new(1.0 / 3.0, 0.0, -1.0 / 3.0),
            Point3::new(1.0, 0.0, -1.0),
        ];
        assert!(section_fit::Carrier::bound(&a, &through_apex).is_none());
    }

    /// Two rods, `x^2 + y^2 = 1` and `(x - 3)^2 + z^2 = 4`, touch at
    /// (1, 0, 0) alone, where a seed near it settles on no crossing. Moved
    /// 1e6 along x, a seed settled 3.1e-4 from the touch, where both
    /// distances lay below the rounding of the coordinates and the normals
    /// 1.5e-4 from parallel, and a march started there; moved 1e12, the
    /// spacing of one coordinate let a point one float step from the touch
    /// read as a crossing. The marcher now works near the origin, so the
    /// rods moved along each axis trace no section, as they do unmoved.
    #[test]
    fn a_touch_far_from_the_origin_settles_as_one_near_it_does() {
        for o in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e6, 0.0, 0.0),
            Vec3::new(1e12, 0.0, 0.0),
            Vec3::new(0.0, 1e12, 0.0),
            Vec3::new(0.0, 0.0, 1e12),
        ] {
            let post = CylindricalSurface::new(
                Point3::new(0.0, 0.0, 0.0) + o,
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
            )
            .unwrap();
            let beam = CylindricalSurface::new(
                Point3::new(3.0, 0.0, 0.0) + o,
                Vec3::new(0.0, 1.0, 0.0),
                2.0,
            )
            .unwrap();
            let (a, b) = (
                AnalyticSurface::Cylinder(&post),
                AnalyticSurface::Cylinder(&beam),
            );
            if o.length() == 0.0 {
                let settled = settle_on_section(&a, &b, Point3::new(1.0, 0.01, 0.01));
                assert!(
                    settled.is_none_or(|p| !crosses_at(&a, &b, p)),
                    "settles on a crossing at {settled:?}"
                );
            }
            let region = Aabb3 {
                min: Point3::new(0.5, -0.5, -0.5) + o,
                max: Point3::new(1.5, 0.5, 0.5) + o,
            };
            let curves =
                marched_sections(a, b, 32, Some((-1.0, 1.0)), Some((-1.0, 1.0)), Some(region))
                    .unwrap();
            assert!(curves.is_empty(), "moved {o:?}: {} sections", curves.len());
        }
    }

    /// A cone opening 0.01 or 0.001 either side of its axis, or lying that
    /// far off its radial plane, meets a rod on its axis in one circle. The
    /// fit through points of that circle is bounded on both surfaces within
    /// the linear tolerance: the cone's factor, `|h| cos + rho sin`, keeps a
    /// lower bound however slender or flat the cone.
    #[test]
    fn fits_on_slender_and_flat_cones_are_bounded() {
        use std::f64::consts::FRAC_PI_2;
        for half in [FRAC_PI_2 - 0.01, FRAC_PI_2 - 0.001, 0.01, 0.001] {
            let cone =
                ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), half)
                    .unwrap();
            let v = 10.0;
            let (rho, h) = (v * half.cos(), v * half.sin());
            let rod =
                CylindricalSurface::new(Point3::new(0.0, 0.0, -1.0), Vec3::new(0.0, 0.0, 1.0), rho)
                    .unwrap();
            let (a, b) = (
                AnalyticSurface::Cone(&cone),
                AnalyticSurface::Cylinder(&rod),
            );
            let circle: Vec<Point3> = (0..=12)
                .map(|k| {
                    let t = TAU * f64::from(k) / 12.0;
                    Point3::new(rho * t.cos(), rho * t.sin(), h)
                })
                .collect();
            let (curve, points) = fit_section(&a, &b, &circle, 1e-7).unwrap();
            let worst = farthest_along(&curve, 4000, |p| {
                off_double_cone(&cone, p).max((p.x().hypot(p.y()) - rho).abs())
            });
            assert!(worst <= 1e-7, "half {half}: off by {worst:.2e}");
            assert!(points.len() < 2000, "half {half}: {} points", points.len());
        }
    }

    /// The ring of radii 4 and 1.5 and a rod of radius 0.6 along y through
    /// (0.5, y, 1), moved 1e6 along each axis, meet in the two loops they
    /// meet in unmoved, each within the linear tolerance of both surfaces.
    /// Moved 1e9, where the floats' spacing (1.2e-7) leaves no room for the
    /// tolerance once a curve is moved back, the traced section is refused
    /// before any fit, not dropped; its fit ran 7 passes over 1.4 s first.
    #[test]
    fn a_section_far_from_the_origin_is_traced_as_near_it() {
        let ring = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let run = |o: Vec3| {
            let torus = ring.translated(o);
            let rod = CylindricalSurface::new(
                Point3::new(0.5, 0.0, 1.0) + o,
                Vec3::new(0.0, 1.0, 0.0),
                0.6,
            )
            .unwrap();
            let region = Aabb3 {
                min: Point3::new(-0.1, -5.5, 0.4) + o,
                max: Point3::new(1.1, 5.5, 1.5) + o,
            };
            intersect_analytic_analytic_in_region(
                AnalyticSurface::Cylinder(&rod),
                AnalyticSurface::Torus(&torus),
                32,
                Some((-10.0, 10.0)),
                None,
                region,
            )
        };
        for o in [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1e6, 0.0, 0.0),
            Vec3::new(0.0, 1e6, 0.0),
            Vec3::new(0.0, 0.0, 1e6),
        ] {
            let curves = run(o).unwrap();
            assert_eq!(curves.len(), 2, "moved {o:?}");
            for c in &curves {
                let worst = farthest_along(&c.curve, 4000, |p| {
                    let q = p - o;
                    let on_torus = ((q.x().hypot(q.y()) - 4.0).hypot(q.z()) - 1.5).abs();
                    let on_rod = ((q.x() - 0.5).hypot(q.z() - 1.0) - 0.6).abs();
                    on_torus.max(on_rod)
                });
                assert!(worst <= 1e-7, "moved {o:?}: off by {worst:.2e}");
            }
        }
        for o in [
            Vec3::new(1e9, 0.0, 0.0),
            Vec3::new(0.0, 1e9, 0.0),
            Vec3::new(0.0, 0.0, 1e9),
        ] {
            assert!(
                matches!(run(o), Err(MathError::ConvergenceFailure { iterations: 0 })),
                "moved {o:?}: not refused before its fit"
            );
        }
    }

    /// A cylinder's frame can sit anywhere along its axis. The rods
    /// `x^2 + y^2 = 1` and `(x - 3)^2 + z^2 = 4`, with their frames 1e6 along
    /// their axes, touch at (1, 0, 0) alone, as with their frames beside it:
    /// a seed near the touch settled 1.55e-4 from it and read as a crossing,
    /// and the marcher refused the pair. Now the seed settles on no crossing
    /// and no section is traced. The ring and rod of the truth audit, the
    /// rod's frame 1e6 along its axis, meet in the loops they meet in with
    /// the frame beside them.
    #[test]
    fn frames_far_along_a_rods_axis_read_as_near_ones() {
        let post =
            CylindricalSurface::new(Point3::new(0.0, 0.0, -1e6), Vec3::new(0.0, 0.0, 1.0), 1.0)
                .unwrap();
        let beam =
            CylindricalSurface::new(Point3::new(3.0, -1e6, 0.0), Vec3::new(0.0, 1.0, 0.0), 2.0)
                .unwrap();
        let (a, b) = (
            AnalyticSurface::Cylinder(&post),
            AnalyticSurface::Cylinder(&beam),
        );
        let settled = settle_on_section(&a, &b, Point3::new(1.0, 0.01, 0.01));
        assert!(
            settled.is_none_or(|p| !crosses_at(&a, &b, p)),
            "settles on a crossing at {settled:?}"
        );
        let region = Aabb3 {
            min: Point3::new(0.5, -0.5, -0.5),
            max: Point3::new(1.5, 0.5, 0.5),
        };
        let far = Some((1e6 - 1.0, 1e6 + 1.0));
        let curves = marched_sections(a, b, 32, far, far, Some(region)).unwrap();
        assert!(curves.is_empty(), "{} sections", curves.len());

        let ring = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let boxes = |frame: f64| {
            let rod = CylindricalSurface::new(
                Point3::new(0.5, -frame, 1.0),
                Vec3::new(0.0, 1.0, 0.0),
                0.6,
            )
            .unwrap();
            let region = Aabb3 {
                min: Point3::new(-0.1, -5.5, 0.4),
                max: Point3::new(1.1, 5.5, 1.5),
            };
            let curves = intersect_analytic_analytic_in_region(
                AnalyticSurface::Cylinder(&rod),
                AnalyticSurface::Torus(&ring),
                32,
                Some((frame - 10.0, frame + 10.0)),
                None,
                region,
            )
            .unwrap();
            let mut boxes: Vec<Aabb3> = curves
                .iter()
                .map(|c| {
                    let worst = farthest_along(&c.curve, 4000, |p| {
                        let on_torus = ((p.x().hypot(p.y()) - 4.0).hypot(p.z()) - 1.5).abs();
                        let on_rod = ((p.x() - 0.5).hypot(p.z() - 1.0) - 0.6).abs();
                        on_torus.max(on_rod)
                    });
                    assert!(worst <= 1e-7, "frame {frame}: off by {worst:.2e}");
                    Aabb3::try_from_points(c.points.iter().map(|ip| ip.point)).unwrap()
                })
                .collect();
            boxes.sort_by(|p, q| p.min.y().total_cmp(&q.min.y()));
            boxes
        };
        let (near, far) = (boxes(0.0), boxes(1e6));
        assert_eq!(near.len(), 2);
        assert_eq!(far.len(), near.len());
        for (n, f) in near.iter().zip(&far) {
            assert!((n.min - f.min).length() < 1e-6 && (n.max - f.max).length() < 1e-6);
        }
    }

    /// `a + b` as a float and its rounding error.
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let back = s - a;
        (s, (a - (s - back)) + (b - back))
    }

    /// `a b` as a float and its rounding error, by Dekker's split.
    fn two_prod(a: f64, b: f64) -> (f64, f64) {
        let split = |v: f64| {
            let c = 134_217_729.0 * v;
            let hi = c - (c - v);
            (hi, v - hi)
        };
        let p = a * b;
        let ((ah, al), (bh, bl)) = (split(a), split(b));
        (p, ((ah * bh - p) + ah * bl + al * bh) + al * bl)
    }

    /// How far `p` lies from the line through `o` along `a`, in
    /// double-length arithmetic, so a far `o` costs no digits of the answer
    /// (nor does `a` being a float step from unit length).
    fn off_line(o: Point3, a: Vec3, p: Point3) -> f64 {
        let w = [
            two_sum(p.x(), -o.x()),
            two_sum(p.y(), -o.y()),
            two_sum(p.z(), -o.z()),
        ];
        let a = [a.x(), a.y(), a.z()];
        // A sum of products `a_i y_i` with each `y_i` double-length.
        let dot = |y: &[(f64, f64); 3]| {
            let (mut hi, mut lo) = (0.0, 0.0);
            for (ai, (yh, yl)) in a.iter().zip(y) {
                let (prod, prod_err) = two_prod(*ai, *yh);
                let (sum, sum_err) = two_sum(hi, prod);
                hi = sum;
                lo += sum_err + prod_err + ai * yl;
            }
            two_sum(hi, lo)
        };
        let (aw, aw_err) = dot(&w);
        let (aa, aa_err) = dot(&a.map(|x| (x, 0.0)));
        // `s = aw / aa` to double length: its float, then the remainder
        // over `aa`.
        let s = aw / aa;
        let (sa, sa_err) = two_prod(s, aa);
        let rest = ((aw - sa) - sa_err) + aw_err - s * aa_err;
        let s_err = rest / aa;
        let across = a.iter().zip(w).map(|(ai, (hi, lo))| {
            let (prod, prod_err) = two_prod(*ai, s);
            let (diff, diff_err) = two_sum(hi, -prod);
            diff + (diff_err + lo - prod_err - ai * s_err)
        });
        across.map(|c| c * c).sum::<f64>().sqrt()
    }

    /// A rod of radius 0.6 on an axis along (0.3, 0.4, 0.5) through the
    /// centre circle of the ring of radii 4 and 1.5 at (4, 0, 0), its frame
    /// 1e12 along the axis. Moved to the region by plain sums, such a frame
    /// landed 2.1e-5 off the axis, and a fit certified against the moved rod
    /// need not lie within 1e-7 of the input one. The frame now lands within
    /// 4e-15 of the axis, and both loops lie within 1e-7 of the input rod
    /// (measured in double-length arithmetic) and of the ring.
    #[test]
    fn a_tilted_rod_framed_far_along_its_axis_is_traced_on_it() {
        let along = Vec3::new(0.3, 0.4, 0.5);
        let through = Point3::new(4.0, 0.0, 0.0);
        let ring = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.5).unwrap();
        let near_rod = CylindricalSurface::new(through, along, 0.6).unwrap();
        let axis = near_rod.axis();
        let far = through + axis * 1e12;
        let rod = CylindricalSurface::new(far, along, 0.6).unwrap();
        let anchor = Point3::new(4.0, 0.0, 0.0);
        let (moved, dv, error) = Moved::new(&AnalyticSurface::Cylinder(&rod), anchor);
        let AnalyticSurface::Cylinder(moved) = moved.surface() else {
            unreachable!("a rod moves to a rod");
        };
        let foot = moved.origin() + (anchor - Point3::new(0.0, 0.0, 0.0));
        let off = off_line(rod.origin(), rod.axis(), foot);
        assert!(
            off <= error && error < 1e-14,
            "the frame lies {off:.2e} off the axis"
        );
        assert!((dv + 1e12).abs() < 1.0, "v moved by {dv}");

        let region = Aabb3 {
            min: Point3::new(2.0, -2.0, -2.0),
            max: Point3::new(6.0, 2.0, 2.0),
        };
        let sweep = (1e12 - 10.0, 1e12 + 10.0);
        let result = marched_sections(
            AnalyticSurface::Cylinder(&rod),
            AnalyticSurface::Torus(&ring),
            32,
            Some((-sweep.1, -sweep.0)),
            None,
            Some(region),
        );
        let curves = result.unwrap();
        assert_eq!(curves.len(), 2);
        {
            for c in &curves {
                let worst = farthest_along(&c.curve, 4000, |p| {
                    let on_rod = (off_line(rod.origin(), rod.axis(), p) - 0.6).abs();
                    let on_ring = ((p.x().hypot(p.y()) - 4.0).hypot(p.z()) - 1.5).abs();
                    on_rod.max(on_ring)
                });
                assert!(worst <= 1e-7, "off by {worst:.2e}");
            }
        }
    }

    /// Each surface's closed-form second derivative of its distance along a
    /// direction matches the distance's central second difference, at points
    /// on and near the surface, along and across it.
    #[test]
    fn each_surface_bends_as_its_distance_does() {
        use section_fit::Carrier;
        let cyl =
            CylindricalSurface::new(Point3::new(0.3, -0.2, 0.5), Vec3::new(0.2, 0.3, 1.0), 0.7)
                .unwrap();
        let ball = SphericalSurface::new(Point3::new(-0.4, 0.1, 0.2), 1.3).unwrap();
        let cone = ConicalSurface::new(Point3::new(0.1, 0.2, -0.3), Vec3::new(-0.3, 0.1, 1.0), 0.6)
            .unwrap();
        let ring = ToroidalSurface::new(Point3::new(0.2, -0.1, 0.1), 2.0, 0.6).unwrap();
        let surfaces = [
            AnalyticSurface::Cylinder(&cyl),
            AnalyticSurface::Sphere(&ball),
            AnalyticSurface::Cone(&cone),
            AnalyticSurface::Torus(&ring),
        ];
        let directions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.6, -0.48, 0.64),
        ];
        let (_, _, _, cone_v) = surface_closures(&AnalyticSurface::Cone(&cone));
        for (k, s) in surfaces.iter().enumerate() {
            let (surf, norm, _, (v0, v1)) = surface_closures(s);
            let (v0, v1) = if matches!(s, AnalyticSurface::Cone(_)) {
                (cone_v.0.max(0.5), cone_v.1)
            } else {
                (v0, v1)
            };
            for (u, v) in [(0.3, 0.25), (1.9, 0.5), (4.4, 0.75)] {
                let v = (v1 - v0).madd(v, v0);
                for off in [0.0, 0.05, -0.05] {
                    let p = surf(u, v) + norm(u, v) * off;
                    let f = |q: Point3| s.distance(q).unwrap().0;
                    for d in directions {
                        let d = d * (1.0 / d.length());
                        let h = 1e-4;
                        let second = (f(p + d * h) - 2.0 * f(p) + f(p - d * h)) / (h * h);
                        let exact = s.bend(p, d).unwrap();
                        assert!(
                            (second - exact).abs() <= 1e-5 * (1.0 + exact.abs()),
                            "surface {k} at {p:?} along {d:?}: {exact} against {second}"
                        );
                    }
                }
            }
        }
    }

    /// Upright, turned and moved, and mirrored through a slanted plane.
    fn three_poses() -> [crate::mat::Mat4; 3] {
        use crate::mat::Mat4;
        [
            Mat4::identity(),
            Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9),
            Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4),
        ]
    }

    /// A cone with its apex at the origin and its axis along z, and a rod of
    /// radius `r` along x through `(0, 0, z0)`, both moved by `pose`; and
    /// the points of their section where the rod's angle about its axis is
    /// `k / n` of a turn, for `k` below `n`, each moved by `pose`.
    fn posed_cone_and_rod(
        pose: &crate::mat::Mat4,
        half: f64,
        z0: f64,
        r: f64,
        n: u32,
    ) -> (ConicalSurface, CylindricalSurface, Vec<Point3>) {
        let o = Point3::new(0.0, 0.0, 0.0);
        let along = |v: Vec3| pose.mul_point(o + v) - pose.mul_point(o);
        let cone =
            ConicalSurface::new(pose.mul_point(o), along(Vec3::new(0.0, 0.0, 1.0)), half).unwrap();
        let rod = CylindricalSurface::new(
            pose.mul_point(Point3::new(0.0, 0.0, z0)),
            along(Vec3::new(1.0, 0.0, 0.0)),
            r,
        )
        .unwrap();
        let mut truth = Vec::new();
        for k in 0..n {
            let phi = TAU * f64::from(k) / f64::from(n);
            let (y, z) = (r * phi.sin(), r.madd(-phi.cos(), z0));
            let rho = z.abs() / half.tan();
            let x2 = rho.madd(rho, -y * y);
            if x2 >= 0.0 {
                for x in [x2.sqrt(), -x2.sqrt()] {
                    truth.push(pose.mul_point(Point3::new(x, y, z)));
                }
            }
        }
        (cone, rod, truth)
    }

    /// A cone lying 0.01 off its radial plane meets a rod of radius 0.05 at
    /// height 0.1 in two loops 10 long and 0.1 across, which turn at radius
    /// 5e-4 at their ends. A cone opening 0.01 about its axis meets rods
    /// across it in loops 0.2 to 0.6 long. The march's steps of up to 1.5
    /// passed the turns and left points 2.4e-2 off the rod, and its closure,
    /// five steps from the seed, joined a loop's far side: each fit was
    /// refused, but for the rod of radius 0.2, whose second loop was lost
    /// with no error. The march now follows each loop round once, in every
    /// pose and either way round: each fit lies within the linear tolerance
    /// of both surfaces, closes, and passes within 1e-5 of every point of
    /// the section.
    #[test]
    fn thin_loops_on_flat_and_slender_cones_are_marched_round() {
        use std::f64::consts::FRAC_PI_2;
        let lin = Tolerance::new().linear;
        let cases = [
            (0.01, 0.1, 0.05),
            (FRAC_PI_2 - 0.01, 3.0, 0.05),
            (FRAC_PI_2 - 0.01, 10.0, 0.05),
            (FRAC_PI_2 - 0.01, 10.0, 0.2),
        ];
        for (half, z0, r) in cases {
            for (p, pose) in three_poses().iter().enumerate() {
                let (cone, rod, truth) = posed_cone_and_rod(pose, half, z0, r, 360);
                for cone_first in [true, false] {
                    let label =
                        format!("half {half} z0 {z0} r {r} pose {p} cone first {cone_first}");
                    let (a, b, va, vb) = if cone_first {
                        (
                            AnalyticSurface::Cone(&cone),
                            AnalyticSurface::Cylinder(&rod),
                            (1.0, 20.0),
                            (-20.0, 20.0),
                        )
                    } else {
                        (
                            AnalyticSurface::Cylinder(&rod),
                            AnalyticSurface::Cone(&cone),
                            (-20.0, 20.0),
                            (1.0, 20.0),
                        )
                    };
                    let result = marched_sections(a, b, 32, Some(va), Some(vb), None);
                    assert!(result.is_ok(), "{label}: {:?}", result.as_ref().err());
                    let curves = result.unwrap();
                    assert_eq!(curves.len(), 2, "{label}: loops");
                    let off_rod = |p: Point3| {
                        let (w, ax) = (p - rod.origin(), rod.axis());
                        ((w - ax * ax.dot(w)).length() - r).abs()
                    };
                    for c in &curves {
                        let worst = farthest_along(&c.curve, 4000, |p| {
                            off_double_cone(&cone, p).max(off_rod(p))
                        });
                        assert!(worst <= lin, "{label}: off by {worst:.2e}");
                        let (first, last) = (c.points[0].point, c.points[c.points.len() - 1].point);
                        assert!((first - last).length() < 1e-9, "{label}: open");
                    }
                    for &t in &truth {
                        let near = curves
                            .iter()
                            .map(|c| {
                                crate::nurbs::projection::project_point_to_curve(&c.curve, t, 1e-12)
                                    .unwrap()
                                    .distance
                            })
                            .fold(f64::INFINITY, f64::min);
                        assert!(near < 1e-5, "{label}: {t:?} lies {near:.1e} off");
                    }
                }
            }
        }
    }
}

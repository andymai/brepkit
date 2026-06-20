//! Closed-form and semi-analytic intersections of analytic surfaces with planes.
//!
//! Provides specialized intersection algorithms for cylinder, cone, sphere,
//! and torus surfaces with planes, as well as a general marching approach
//! for analytic-analytic surface intersections.

use std::f64::consts::{FRAC_PI_2, TAU};

use crate::MathError;
use crate::curves::{Circle3D, Ellipse3D};
use crate::frame::Frame3;
use crate::nurbs::fitting::interpolate;
use crate::nurbs::intersection::{IntersectionCurve, IntersectionPoint};
use crate::surfaces::{ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface};
use crate::tolerance::Tolerance;
use crate::vec::{Point3, Vec3};

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

/// Compute exact intersection curves between a plane and an analytic surface.
///
/// Returns exact `Circle3D` or `Ellipse3D` where possible, falling back to
/// sampled points for complex cases (torus).
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
    match surface {
        AnalyticSurface::Cylinder(cyl) => exact_plane_cylinder(cyl, plane_normal, plane_d),
        AnalyticSurface::Sphere(sphere) => exact_plane_sphere(sphere, plane_normal, plane_d),
        AnalyticSurface::Cone(cone) => exact_plane_cone(cone, plane_normal, plane_d),
        AnalyticSurface::Torus(torus) => {
            // Torus intersections are degree-4 — fall back to sampling.
            let chains = sample_plane_torus(torus, plane_normal, plane_d)?;
            Ok(chains
                .into_iter()
                .map(ExactIntersectionCurve::Points)
                .collect())
        }
    }
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
        // Plane parallel to cylinder axis → 0 or 2 line segments.
        // Fall back to sampled points.
        let chains = sample_plane_cylinder(cyl, normal, d)?;
        return Ok(chains
            .into_iter()
            .map(ExactIntersectionCurve::Points)
            .collect());
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

    if h.abs() > r - 1e-10 {
        return Ok(vec![]);
    }

    let circle_r = (r.mul_add(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.mul_add(-normal.x(), sphere.center().x()),
        h.mul_add(-normal.y(), sphere.center().y()),
        h.mul_add(-normal.z(), sphere.center().z()),
    );

    let circle = Circle3D::new(circle_center, normal, circle_r)?;
    Ok(vec![ExactIntersectionCurve::Circle(circle)])
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
        let chains = sample_plane_cone(cone, normal, d)?;
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
    let chains = sample_plane_cone(cone, normal, d)?;
    Ok(chains
        .into_iter()
        .map(ExactIntersectionCurve::Points)
        .collect())
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
        AnalyticSurface::Cone(cone) => sample_plane_cone(cone, normal, d),
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

    let circle_r = (r.mul_add(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.mul_add(-normal.x(), sphere.center().x()),
        h.mul_add(-normal.y(), sphere.center().y()),
        h.mul_add(-normal.z(), sphere.center().z()),
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
    let v_max = (8.0 * v_min).max(v_min + 4.0);

    let valid: Vec<Option<Point3>> = vs
        .into_iter()
        .enumerate()
        .map(|(i, v)| {
            let v = v?;
            if v <= v_max {
                let u = TAU * (i as f64) / (n_samples as f64);
                let g = cone.evaluate(u, 1.0) - apex;
                Some(apex + g * v)
            } else {
                None
            }
        })
        .collect();

    // Split into contiguous runs of `Some`, treating the array as circular so a
    // branch straddling u=0 stays in one chain. A fully-valid sweep (ellipse)
    // becomes a single closed chain.
    let chains = contiguous_chains(&valid);
    Ok(chains.into_iter().filter(|c| c.len() >= 2).collect())
}

/// Split a circular array of optional samples into contiguous `Some` runs.
///
/// If every entry is `Some` the whole array is one chain, closed by repeating
/// the first point. Otherwise each maximal run of `Some` between `None` gaps is
/// one chain; a run wrapping past index 0 is rejoined into a single chain.
fn contiguous_chains(valid: &[Option<Point3>]) -> Vec<Vec<Point3>> {
    let n = valid.len();
    if n == 0 {
        return Vec::new();
    }
    if valid.iter().all(Option::is_some) {
        // Closed loop (ellipse): emit all points and repeat the first to close.
        let mut pts: Vec<Point3> = valid.iter().filter_map(|p| *p).collect();
        if let Some(&first) = pts.first() {
            pts.push(first);
        }
        return vec![pts];
    }
    // Rotate the start to just after a gap so no run wraps the array boundary.
    let gap = valid.iter().position(Option::is_none).unwrap_or(0);
    let mut chains: Vec<Vec<Point3>> = Vec::new();
    let mut current: Vec<Point3> = Vec::new();
    for k in 0..n {
        let idx = (gap + k) % n;
        match valid[idx] {
            Some(p) => current.push(p),
            None => {
                if current.len() >= 2 {
                    chains.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
        }
    }
    if current.len() >= 2 {
        chains.push(current);
    }
    chains
}

/// Sample the plane-torus intersection as ordered 3D points.
///
/// Uses the same Newton-refined sampling grid as `intersect_plane_torus`
/// but skips NURBS curve fitting.
#[allow(clippy::cast_precision_loss)]
fn sample_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<Vec<Point3>>, MathError> {
    let curves = intersect_plane_torus(torus, normal, d)?;
    Ok(curves
        .into_iter()
        .map(|c| c.points.into_iter().map(|p| p.point).collect())
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

    let circle_r = (r.mul_add(r, -(h * h))).sqrt();
    let circle_center = Point3::new(
        h.mul_add(-normal.x(), sphere.center().x()),
        h.mul_add(-normal.y(), sphere.center().y()),
        h.mul_add(-normal.z(), sphere.center().z()),
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
/// The torus-plane intersection is a degree-4 curve with no simple closed form.
/// Uses grid sampling with sign-change detection and Newton refinement.
///
/// # Errors
///
/// Returns an error if curve fitting fails.
#[allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]
pub fn intersect_plane_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
) -> Result<Vec<IntersectionCurve>, MathError> {
    let n_grid = 128_usize;

    // Signed distance to plane for a torus point.
    let sdf = |u: f64, v: f64| -> f64 { dot_np(normal, torus.evaluate(u, v)) - d };

    // Collect zero-crossing points by scanning edges of a (u,v) grid.
    let mut crossing_pts: Vec<(f64, f64, Point3)> = Vec::new();

    let du = TAU / (n_grid as f64);
    let dv = TAU / (n_grid as f64);

    // Offset grid by half a cell to avoid landing exactly on zero crossings
    // (e.g. sin(0) = 0.0 exactly in IEEE 754, which defeats sign-change detection).
    let u_off = du * 0.5;
    let v_off = dv * 0.5;

    for iu in 0..n_grid {
        for iv in 0..n_grid {
            let u0 = (iu as f64).mul_add(du, u_off);
            let v0 = (iv as f64).mul_add(dv, v_off);
            let u1 = u0 + du;
            let v1 = v0 + dv;

            let f00 = sdf(u0, v0);
            let f10 = sdf(u1, v0);
            let f01 = sdf(u0, v1);

            // Check horizontal edge (u0,v0)-(u1,v0).
            if f00 * f10 < 0.0 {
                let t = f00 / (f00 - f10);
                let u = t.mul_add(u1 - u0, u0);
                let (u_r, v_r) = newton_refine_torus(torus, normal, d, u, v0);
                crossing_pts.push((u_r, v_r, torus.evaluate(u_r, v_r)));
            }

            // Check vertical edge (u0,v0)-(u0,v1).
            if f00 * f01 < 0.0 {
                let t = f00 / (f00 - f01);
                let v = t.mul_add(v1 - v0, v0);
                let (u_r, v_r) = newton_refine_torus(torus, normal, d, u0, v);
                crossing_pts.push((u_r, v_r, torus.evaluate(u_r, v_r)));
            }
        }
    }

    if crossing_pts.is_empty() {
        return Ok(vec![]);
    }

    // Group nearby points into connected curves via greedy chaining.
    let mut used = vec![false; crossing_pts.len()];
    let mut curves = Vec::new();

    for start in 0..crossing_pts.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain = vec![start];

        loop {
            let last = chain[chain.len() - 1];
            let last_pt = crossing_pts[last].2;
            let mut best_idx = None;
            let mut best_dist = 1.0_f64;

            for (j, &is_used) in used.iter().enumerate() {
                if is_used {
                    continue;
                }
                let dist = (crossing_pts[j].2 - last_pt).length();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = Some(j);
                }
            }

            if let Some(j) = best_idx {
                used[j] = true;
                chain.push(j);
            } else {
                break;
            }
        }

        if chain.len() >= 4 {
            let pts: Vec<Point3> = chain.iter().map(|&i| crossing_pts[i].2).collect();
            let ipts: Vec<IntersectionPoint> = chain
                .iter()
                .map(|&i| IntersectionPoint {
                    point: crossing_pts[i].2,
                    param1: (crossing_pts[i].0, crossing_pts[i].1),
                    param2: (0.0, 0.0),
                })
                .collect();

            if let Ok(curve) = interpolate(&pts, 3.min(pts.len() - 1)) {
                curves.push(IntersectionCurve {
                    curve,
                    points: ipts,
                });
            }
        }
    }

    Ok(curves)
}

/// Newton-refine a torus parameter to lie on the cutting plane.
fn newton_refine_torus(
    torus: &ToroidalSurface,
    normal: Vec3,
    d: f64,
    mut u: f64,
    mut v: f64,
) -> (f64, f64) {
    let eps = 1e-6;
    for _ in 0..10 {
        let f = dot_np(normal, torus.evaluate(u, v)) - d;
        if f.abs() < 1e-12 {
            break;
        }
        // Numerical gradient via central differences.
        let fu = (dot_np(normal, torus.evaluate(u + eps, v))
            - dot_np(normal, torus.evaluate(u - eps, v)))
            / (2.0 * eps);
        let fv = (dot_np(normal, torus.evaluate(u, v + eps))
            - dot_np(normal, torus.evaluate(u, v - eps)))
            / (2.0 * eps);

        let grad_sq = fu.mul_add(fu, fv * fv);
        if grad_sq < 1e-20 {
            break;
        }
        let step = f / grad_sq;
        u -= step * fu;
        v -= step * fv;
    }
    (u, v)
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
    // Try algebraic specialization for known surface pairs before falling
    // back to the general marching approach.
    if let Some(result) = try_algebraic_intersection(&a, &b)? {
        return Ok(result);
    }

    let (surf_a, norm_a, u_range_a, default_v_a) = surface_closures(&a);
    let (surf_b, norm_b, u_range_b, default_v_b) = surface_closures(&b);
    let v_range_a = v_range_hint_a.unwrap_or(default_v_a);
    let v_range_b = v_range_hint_b.unwrap_or(default_v_b);

    // Compute characteristic surface dimensions for adaptive parameters.
    let diag_a = {
        let p00 = surf_a(u_range_a.0, v_range_a.0);
        let p11 = surf_a(u_range_a.1, v_range_a.1);
        (p00 - p11).length()
    };
    let diag_b = {
        let p00 = surf_b(u_range_b.0, v_range_b.0);
        let p11 = surf_b(u_range_b.1, v_range_b.1);
        (p00 - p11).length()
    };
    let char_size = diag_a.min(diag_b).max(0.1);

    // Sample surface A on a grid. For each grid point, project it
    // analytically onto surface B to find the closest point, then check
    // if the distance is below threshold (indicating near-intersection).
    #[allow(clippy::type_complexity)]
    let mut seeds: Vec<(Point3, (f64, f64), (f64, f64))> = Vec::new();
    // Coarse threshold scales with the surface size — the distance from
    // a grid point on A to its projection on B can be large even near
    // the intersection (e.g., sphere R=2 and cylinder R=1 → gap ≈ 1).
    let seed_threshold = diag_a.max(diag_b).max(1.0) * 0.5;
    let mut min_dist = f64::INFINITY;

    #[allow(clippy::cast_precision_loss)]
    for ia in 0..grid_res {
        for ja in 0..grid_res {
            let ua =
                u_range_a.0 + (u_range_a.1 - u_range_a.0) * (ia as f64 + 0.5) / (grid_res as f64);
            let va =
                v_range_a.0 + (v_range_a.1 - v_range_a.0) * (ja as f64 + 0.5) / (grid_res as f64);

            let pa = surf_a(ua, va);

            // Analytically project onto surface B.
            let (ub, vb) = project_analytic(&b, pa, u_range_b, v_range_b);
            let pb = surf_b(ub, vb);
            let dist = (pa - pb).length();
            min_dist = min_dist.min(dist);

            if dist < seed_threshold {
                // Use the coarse seed directly. The marching algorithm
                // corrects positions at each step via projection, so seeds
                // don't need to be on the exact intersection — they just
                // need to be close enough for the marcher to converge.
                let mid = Point3::new(
                    (pa.x() + pb.x()) * 0.5,
                    (pa.y() + pb.y()) * 0.5,
                    (pa.z() + pb.z()) * 0.5,
                );
                seeds.push((mid, (ua, va), (ub, vb)));
            }
        }
    }

    // Cheap rejection: the grid samples surface A; the closest sample's
    // distance to B lower-bounds how near the two bounded patches come. A
    // transversal crossing puts a sample within ~one grid cell of it
    // (distance on the order of a cell), so if even the nearest sample is
    // several cells away the patches cannot cross — skip the expensive
    // marching and return empty. Result-preserving: non-crossing pairs
    // already march to nothing, just slowly (this is the gridfinity lip's
    // ~80 inner-wall × outer-wall pairs that dominate pavefiller time).
    let reject_dist = (char_size / grid_res as f64) * 3.0;
    if min_dist > reject_dist {
        return Ok(vec![]);
    }

    if seeds.is_empty() {
        return Ok(vec![]);
    }

    // Aggressively deduplicate seeds — we only need 1-2 per intersection
    // branch. Scale dedup radius to ~2% of characteristic surface size
    // (at least 10× the march step size) to avoid redundant marches.
    let march_step = (char_size * 0.02).clamp(0.005, 0.5);
    let dedup_radius = march_step * 10.0;
    let mut unique_seeds = Vec::new();
    for seed in &seeds {
        let dominated = unique_seeds
            .iter()
            .any(|s: &(Point3, (f64, f64), (f64, f64))| (s.0 - seed.0).length() < dedup_radius);
        if !dominated {
            unique_seeds.push(*seed);
        }
    }

    // March from each seed.
    let mut curves = Vec::new();
    let mut used_seeds = vec![false; unique_seeds.len()];

    for si in 0..unique_seeds.len() {
        if used_seeds[si] {
            continue;
        }
        used_seeds[si] = true;

        let march_result = march_analytic_intersection(
            &a,
            &b,
            surf_a.as_ref(),
            norm_a.as_ref(),
            surf_b.as_ref(),
            norm_b.as_ref(),
            unique_seeds[si].0,
            u_range_a,
            v_range_a,
            u_range_b,
            v_range_b,
            march_step,
            is_u_periodic(&a),
            is_u_periodic(&b),
        );

        if march_result.len() >= 2 {
            for (sj, other) in unique_seeds.iter().enumerate() {
                if !used_seeds[sj]
                    && march_result
                        .iter()
                        .any(|p| (*p - other.0).length() < dedup_radius)
                {
                    used_seeds[sj] = true;
                }
            }

            let ipts: Vec<IntersectionPoint> = march_result
                .iter()
                .map(|&pt| IntersectionPoint {
                    point: pt,
                    param1: (0.0, 0.0),
                    param2: (0.0, 0.0),
                })
                .collect();

            let degree = 3.min(march_result.len() - 1);
            if let Ok(curve) = interpolate(&march_result, degree) {
                curves.push(IntersectionCurve {
                    curve,
                    points: ipts,
                });
            }
        }
    }

    Ok(curves)
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
#[allow(clippy::too_many_lines)]
fn try_algebraic_intersection(
    a: &AnalyticSurface<'_>,
    b: &AnalyticSurface<'_>,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
    match (a, b) {
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
            }
            // Non-coaxial: algebraic quadratic in v.
            algebraic_cylinder_cylinder(c1, c2)
        }
        // Sphere-cylinder (both orderings).
        (AnalyticSurface::Sphere(s), AnalyticSurface::Cylinder(c))
        | (AnalyticSurface::Cylinder(c), AnalyticSurface::Sphere(s)) => {
            algebraic_sphere_cylinder(s, c)
        }
        (AnalyticSurface::Cone(c1), AnalyticSurface::Cone(c2)) => algebraic_cone_cone(c1, c2),
        _ => Ok(None),
    }
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
        return Ok(None); // Parallel but offset axes — not coaxial.
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

/// Algebraic coaxial cone-cone intersection (NURBS form for the general
/// bounded path). Delegates to [`exact_cone_cone`] and samples each exact
/// circle into an interpolated NURBS `IntersectionCurve`, mirroring the
/// sphere-cylinder algebraic path. phase FF prefers the exact circle form
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
        let ExactIntersectionCurve::Circle(circle) = exact else {
            continue;
        };
        let n_samples = 33;
        let mut positions = Vec::with_capacity(n_samples);
        let mut points = Vec::with_capacity(n_samples);
        #[allow(clippy::cast_precision_loss)]
        for i in 0..n_samples {
            let theta = TAU * i as f64 / (n_samples - 1) as f64;
            let pt = crate::traits::ParametricCurve::evaluate(&circle, theta);
            positions.push(pt);
            points.push(IntersectionPoint {
                point: pt,
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            });
        }
        let degree = 3.min(positions.len() - 1);
        let curve = interpolate(&positions, degree)?;
        curves.push(IntersectionCurve { curve, points });
    }
    Ok(Some(curves))
}

/// Algebraic sphere-cylinder intersection.
///
/// For a sphere of radius R centered at C and a cylinder of radius r with
/// axis through O in direction A, the intersection is found by:
///
/// 1. Project the sphere center onto the cylinder axis.
/// 2. Compute the perpendicular distance `d_perp` from center to axis.
/// 3. If `d_perp + r > R` or `d_perp + R < r`: no intersection.
/// 4. Otherwise, the intersection lies at axial positions where
///    `sqrt(R² - z²) = r` for the coaxial case (d_perp = 0), giving
///    two circles at `z = ±sqrt(R² - r²)`.
/// 5. For the general case, solve a quartic for the axial position.
///    Currently only handles the coaxial case analytically.
fn algebraic_sphere_cylinder(
    sphere: &SphericalSurface,
    cyl: &CylindricalSurface,
) -> Result<Option<Vec<IntersectionCurve>>, MathError> {
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

    // Separation check.
    if d_perp > r_sphere + r_cyl + 1e-10 {
        return Ok(Some(vec![])); // Too far apart
    }

    // Currently only handle the coaxial/near-coaxial case.
    // Non-coaxial sphere-cylinder intersections produce quartic curves;
    // fall back to the general marching approach for those.
    if d_perp > 1e-7 {
        return Ok(None); // Fall through to marching
    }

    // Coaxial case: sphere center is on the cylinder axis.
    // The intersection is at z = ±sqrt(R² - r²) relative to sphere center.
    if r_cyl > r_sphere + 1e-10 {
        return Ok(Some(vec![])); // Cylinder larger than sphere
    }

    let z_sq = r_sphere * r_sphere - r_cyl * r_cyl;
    if z_sq < 0.0 {
        return Ok(Some(vec![])); // No real intersection
    }

    let z = z_sq.sqrt();

    // The intersection circles are centered on the axis at height ±z
    // from the sphere center, with radius = r_cyl.
    let center_axis_pt = Point3::new(
        co.x() + axis.x() * along,
        co.y() + axis.y() * along,
        co.z() + axis.z() * along,
    );

    let mut curves = Vec::new();

    // Build reference frame perpendicular to axis.
    let basis = Frame3::from_normal(center_axis_pt, axis)?;
    let u_dir = basis.x;
    let v_dir = basis.y;

    for &z_offset in &[z, -z] {
        let center = Point3::new(
            center_axis_pt.x() + axis.x() * z_offset,
            center_axis_pt.y() + axis.y() * z_offset,
            center_axis_pt.z() + axis.z() * z_offset,
        );

        let n_samples = 33;
        let mut points = Vec::with_capacity(n_samples);
        let mut positions = Vec::with_capacity(n_samples);
        #[allow(clippy::cast_precision_loss)]
        for i in 0..n_samples {
            let theta = TAU * i as f64 / (n_samples - 1) as f64;
            let (sin_t, cos_t) = theta.sin_cos();
            let pt = Point3::new(
                center.x() + (u_dir.x() * cos_t + v_dir.x() * sin_t) * r_cyl,
                center.y() + (u_dir.y() * cos_t + v_dir.y() * sin_t) * r_cyl,
                center.z() + (u_dir.z() * cos_t + v_dir.z() * sin_t) * r_cyl,
            );
            positions.push(pt);
            points.push(IntersectionPoint {
                point: pt,
                param1: (0.0, 0.0),
                param2: (0.0, 0.0),
            });
        }

        let degree = 3.min(positions.len() - 1);
        let curve = interpolate(&positions, degree)?;
        curves.push(IntersectionCurve { curve, points });
    }

    // If z is effectively 0, the two circles coincide (tangent case).
    if z < 1e-10 {
        curves.pop(); // Remove duplicate
    }

    Ok(Some(curves))
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
    let x1 = c1.x_axis();
    let y1 = c1.y_axis();

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

    // Sample u from 0 to 2π on cylinder 1. Offset by half a step to avoid
    // landing exactly on crossing points where disc=0 and both curves coincide.
    // This ensures the two algebraic branches have distinct sample endpoints,
    // so the face splitter's wire builder doesn't face 4-way junction ambiguity.
    let n_samples = 128;
    let mut curve_plus: Vec<Point3> = Vec::with_capacity(n_samples + 1);
    let mut curve_minus: Vec<Point3> = Vec::with_capacity(n_samples + 1);
    let u_offset = TAU / (n_samples as f64 * 2.0); // half a step

    // Sample n_samples DISTINCT points (no duplicate at closure).
    // After the loop, explicitly close each curve by copying the first point.
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n_samples {
        let u = u_offset + TAU * i as f64 / n_samples as f64;
        let (sin_u, cos_u) = u.sin_cos();

        // Radial point on c1 at angle u, height v=0:
        // q = c1.origin + r1*(cos(u)*x1 + sin(u)*y1) - c2.origin
        let qx = o1.x() + r1 * (cos_u * x1.x() + sin_u * y1.x()) - o2.x();
        let qy = o1.y() + r1 * (cos_u * x1.y() + sin_u * y1.y()) - o2.y();
        let qz = o1.z() + r1 * (cos_u * x1.z() + sin_u * y1.z()) - o2.z();

        let q_dot_a1 = qx * a1.x() + qy * a1.y() + qz * a1.z();
        let q_dot_a2 = qx * a2.x() + qy * a2.y() + qz * a2.z();
        let q_sq = qx * qx + qy * qy + qz * qz;

        let b_coeff = 2.0 * (q_dot_a1 - alpha * q_dot_a2);
        let c_coeff = q_sq - q_dot_a2 * q_dot_a2 - r2 * r2;

        let disc = b_coeff * b_coeff - 4.0 * a_coeff * c_coeff;
        // Clamp tiny negative discriminant (floating-point noise near tangent
        // crossing points where disc → 0) to avoid gaps in the sample set.
        if disc < -Tolerance::new().linear {
            continue;
        }

        let sqrt_disc = disc.max(0.0).sqrt();
        let v_plus = (-b_coeff + sqrt_disc) / (2.0 * a_coeff);
        let v_minus = (-b_coeff - sqrt_disc) / (2.0 * a_coeff);

        curve_plus.push(c1.evaluate(u, v_plus));
        curve_minus.push(c1.evaluate(u, v_minus));
    }

    // Explicitly close each curve by copying the first point (exact match
    // avoids near-zero chord length in NURBS interpolation).
    if !curve_plus.is_empty() {
        curve_plus.push(curve_plus[0]);
    }
    if !curve_minus.is_empty() {
        curve_minus.push(curve_minus[0]);
    }

    let mut curves = Vec::new();

    for pts in [&curve_plus, &curve_minus] {
        if pts.len() < 4 {
            continue;
        }

        let ipts: Vec<IntersectionPoint> = pts
            .iter()
            .map(|&p| {
                let (u1, v1) = c1.project_point(p);
                let (u2, v2) = c2.project_point(p);
                IntersectionPoint {
                    point: p,
                    param1: (u1, v1),
                    param2: (u2, v2),
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

    Ok(Some(curves))
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

    // Sample the circle for the IntersectionCurve representation.
    let n_samples = 33; // Odd for symmetry
    let mut points = Vec::with_capacity(n_samples);
    let mut positions = Vec::with_capacity(n_samples);
    #[allow(clippy::cast_precision_loss)]
    for i in 0..n_samples {
        let theta = TAU * i as f64 / (n_samples - 1) as f64;
        let (sin_t, cos_t) = theta.sin_cos();
        let pt = Point3::new(
            center.x() + (u_dir.x() * cos_t + v_dir.x() * sin_t) * r_circle,
            center.y() + (u_dir.y() * cos_t + v_dir.y() * sin_t) * r_circle,
            center.z() + (u_dir.z() * cos_t + v_dir.z() * sin_t) * r_circle,
        );
        positions.push(pt);
        points.push(IntersectionPoint {
            point: pt,
            param1: (0.0, 0.0),
            param2: (0.0, 0.0),
        });
    }

    let degree = 3.min(positions.len() - 1);
    let curve = interpolate(&positions, degree)?;

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

/// March along the intersection of two surfaces from a seed point.
///
/// Uses the cross product of surface normals as the tangent direction
/// and projects back onto both surfaces using analytical projection
/// (for cylinders/spheres) or grid search (fallback).
#[allow(clippy::too_many_arguments)]
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
) -> Vec<Point3> {
    let max_steps = 500;
    let h_min = 1e-6;
    let h_max = initial_step * 4.0;
    // Fixed closure threshold: the adaptive step `h` varies with curvature
    // and can shrink below the actual miss distance at the seed re-approach.
    // Use `initial_step * 5` to robustly detect closure on the first pass.
    let closure_dist = initial_step * 5.0;
    // Angular thresholds for curvature-adaptive stepping.
    let max_angle = 10.0_f64.to_radians();
    let min_angle = 2.0_f64.to_radians();

    // March forward from seed, collecting points.
    let mut forward = Vec::new();
    // March backward from seed, collecting points (reversed at end).
    let mut backward = Vec::new();

    for (direction, points) in [(1.0_f64, &mut forward), (-1.0_f64, &mut backward)] {
        let mut current = seed;
        let mut h = initial_step;
        let mut prev_tangent: Option<Vec3> = None;

        for _ in 0..max_steps {
            let (ua, va) = project_analytic(a, current, u_range_a, v_range_a);
            let (ub, vb) = project_analytic(b, current, u_range_b, v_range_b);

            let na = norm_a(ua, va);
            let nb = norm_b(ub, vb);

            let tangent = na.cross(nb);
            let t_len = tangent.length();
            if t_len < 1e-10 {
                break;
            }
            let t_dir = tangent * (direction / t_len);

            // Curvature-adaptive step: check angular deviation from previous tangent.
            if let Some(prev_t) = prev_tangent {
                let cos_angle = prev_t.dot(t_dir).clamp(-1.0, 1.0);
                let angle = cos_angle.acos();
                if angle > max_angle && h > h_min {
                    h = (h * 0.5).max(h_min);
                } else if angle < min_angle {
                    h = (h * 2.0).min(h_max);
                }
            }
            prev_tangent = Some(t_dir);

            let next = Point3::new(
                h.mul_add(t_dir.x(), current.x()),
                h.mul_add(t_dir.y(), current.y()),
                h.mul_add(t_dir.z(), current.z()),
            );

            let (ua2, va2) = project_analytic(a, next, u_range_a, v_range_a);
            let (ub2, vb2) = project_analytic(b, next, u_range_b, v_range_b);

            let pa = surf_a(ua2, va2);
            let pb = surf_b(ub2, vb2);
            let mid = Point3::new(
                (pa.x() + pb.x()) * 0.5,
                (pa.y() + pb.y()) * 0.5,
                (pa.z() + pb.z()) * 0.5,
            );
            let out_a = (!u_periodic_a && (ua2 <= u_range_a.0 || ua2 >= u_range_a.1))
                || va2 <= v_range_a.0
                || va2 >= v_range_a.1;
            let out_b = (!u_periodic_b && (ub2 <= u_range_b.0 || ub2 >= u_range_b.1))
                || vb2 <= v_range_b.0
                || vb2 >= v_range_b.1;

            if out_a || out_b {
                break;
            }

            // Check for loop closure — if we've collected enough points and
            // the current point is close to the seed, the curve is closed.
            // Require ≥10 steps to avoid premature closure near the seed.
            let dist_to_seed = (mid - seed).length();
            if points.len() > 10 && dist_to_seed < closure_dist {
                points.push(seed);
                break;
            }

            points.push(mid);
            current = mid;
        }
    }

    // Assemble result: backward (reversed) + seed + forward
    backward.reverse();
    let mut result = backward;
    result.push(seed);
    result.append(&mut forward);

    // Refine all points onto the intersection curve via Newton correction.
    for pt in &mut result {
        *pt = correct_to_intersection(
            a, b, surf_a, norm_a, surf_b, norm_b, *pt, u_range_a, v_range_a, u_range_b, v_range_b,
            5,
        );
    }

    result
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
        AnalyticSurface::Cone(cone) => {
            let (u, v) = cone.project_point(point);
            (u.clamp(u_range.0, u_range.1), v.clamp(v_range.0, v_range.1))
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
        let curves = exact_plane_cone(&cone, n, d).unwrap();
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
        let curves = exact_plane_cone(&cone, n, d).unwrap();
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
        let curves = exact_plane_cone(&cone, n, d).unwrap();
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
        let curves = exact_plane_cone(&cone, n, d).unwrap();
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
}

// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Analytic fast paths for common surface pairs.
//!
//! Closed-form fillet and chamfer solutions for surface pairs that admit
//! axisymmetric or planar blend geometry. These bypass the walking engine
//! entirely, producing exact geometry 10–100× faster than Newton-Raphson
//! marching.
//!
//! # Coverage matrix
//!
//! Each pair below has fast paths for both fillet and chamfer (unless
//! noted), handling all four convex/concave combinations via per-face
//! `signed_offset_i ∈ {+1, −1}`:
//!
//! | Pair                           | Blend surface           | Axis-alignment requirement      |
//! |--------------------------------|-------------------------|---------------------------------|
//! | Plane × Plane                  | Cylinder (fillet) / Plane (chamfer) | none — dihedral edge       |
//! | Plane × {Cylinder, Cone, Sphere} | Torus (fillet) / Cone (chamfer) | other surface's axis ⟂ plate     |
//! | Sphere × {Cylinder, Cone}      | Torus (fillet) / Cone (chamfer) | sphere centre on cyl/cone axis line |
//! | Sphere × Sphere                | Torus (fillet) / Cone (chamfer) | none — axis is line C1→C2 by construction |
//! | Cylinder × Cylinder (parallel axes) | Cylinder (fillet) / Plane (chamfer) | parallel cyl axes, intersecting   |
//! | Cone × Cone (coaxial)          | Torus (fillet) / Cone (chamfer) | shared axis line, β1 ≠ β2         |
//!
//! # Fallthrough configurations
//!
//! Each helper returns `Ok(None)` when its preconditions don't hold,
//! so the walker takes over. Common reasons to fall through:
//!   - Required axis alignment fails (e.g. sphere centre off the
//!     cylinder axis, cone axes not coincident, cyl axes not parallel)
//!   - Pair has no closed-form blend at all (Cyl × Cone in any
//!     orientation, perpendicular cyl × cyl, non-coaxial cone × cone)
//!   - Helper-specific guards (degenerate spine, spindle torus,
//!     `r ≥ R_s` for concave-sphere, etc. — see each helper's docs)
//!   - Surface variant not yet wired analytically (Torus, NURBS)
//!
//! Roughly 80% of real-world fillets fit one of the analytic pairs, so
//! these fast paths are high-impact optimizations.

use brepkit_math::curves2d::{Curve2D, Line2D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::BlendError;
use crate::section::CircSection;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};

/// Linear tolerance for "essentially zero" guards in analytic helpers.
///
/// 1e-9 is tighter than `Tolerance::default().linear` (= 1e-7, the
/// vertex-tolerance default) — appropriate here because we're checking
/// whether floating-point intermediates have collapsed to zero (e.g.
/// `r_p_sq <= ANALYTIC_TOL_LIN²` flags a degenerate spine), not whether
/// two coordinates are "geometrically equal" up to vertex tol.
const ANALYTIC_TOL_LIN: f64 = 1e-9;

/// Dimensionless tolerance for parallelism / perpendicularity checks in
/// analytic helpers, used in `1 − |cos θ|` form (dot products of unit
/// vectors) and `|sin θ|` form (cross-product magnitudes).
///
/// The naming parallels `ANALYTIC_TOL_LIN` for grep-ability, but the
/// unit is dimensionless, NOT radians. The effective angular gate
/// depends on which form the check uses:
///   - `dot.abs() < 1.0 − ANALYTIC_TOL_ANG` (most common): for unit
///     vectors with `cos θ ≈ 1 − θ²/2`, the threshold corresponds to
///     `θ ≈ √(2 · ANALYTIC_TOL_ANG) ≈ 4.5e-5 rad ≈ 9 arcseconds`.
///   - `cross(a, b).length() > ANALYTIC_TOL_ANG`: this is `|sin θ|`,
///     so the threshold is `θ ≈ ANALYTIC_TOL_ANG ≈ 5.7e-8°` directly.
///
/// 1e-9 was chosen to match the `ANALYTIC_TOL_LIN` floating-point-noise
/// scale in the same helpers, NOT to mirror `Tolerance::default().angular
/// = 1e-12`. A blend pair beyond either gate is no longer axis-aligned
/// in a way that admits a closed-form solution.
const ANALYTIC_TOL_ANG: f64 = 1e-9;

/// Result of an analytic fillet/chamfer computation.
///
/// Contains the blend surface and contact geometry, but not yet
/// integrated into topology (no new edges created at this stage).
pub struct AnalyticResult {
    /// The blend surface (cylinder for plane-plane fillet, plane for chamfer).
    pub surface: FaceSurface,
    /// 3D contact curve on face 1.
    pub contact1: NurbsCurve,
    /// 3D contact curve on face 2.
    pub contact2: NurbsCurve,
    /// PCurve on face 1 (UV-space).
    pub pcurve1: Curve2D,
    /// PCurve on face 2 (UV-space).
    pub pcurve2: Curve2D,
    /// Cross-sections at spine start and end.
    pub sections: Vec<CircSection>,
}

/// Try to compute a fillet analytically for two surfaces.
///
/// Returns `Some(StripeResult)` if the surface pair has a closed-form solution,
/// `None` otherwise (caller should fall back to the walking engine).
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_arguments)]
pub fn try_analytic_fillet(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    match (surf1, surf2) {
        (FaceSurface::Plane { normal: n1, d: _d1 }, FaceSurface::Plane { normal: n2, d: _d2 }) => {
            let result = plane_plane_fillet(spine, topo, *n1, *n2, radius, face1, face2)?;
            Ok(Some(result))
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Cylinder(cyl)) => {
            plane_cylinder_fillet(*normal, *d, cyl, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Cylinder(cyl), FaceSurface::Plane { normal, d }) => {
            // Plane is on side 2; swap to keep the analytic helper's argument
            // order canonical (plane first), and remember to swap the
            // resulting Stripe's face/pcurve/contact assignments.
            let mut result =
                plane_cylinder_fillet(*normal, *d, cyl, spine, topo, radius, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Cone(cone)) => {
            plane_cone_fillet(*normal, *d, cone, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Cone(cone), FaceSurface::Plane { normal, d }) => {
            let mut result =
                plane_cone_fillet(*normal, *d, cone, spine, topo, radius, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Sphere(sph)) => {
            plane_sphere_fillet(*normal, *d, sph, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Plane { normal, d }) => {
            let mut result =
                plane_sphere_fillet(*normal, *d, sph, spine, topo, radius, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Sphere(s1), FaceSurface::Sphere(s2)) => {
            sphere_sphere_fillet(s1, s2, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Cylinder(cyl), FaceSurface::Sphere(sph)) => {
            let mut result = sphere_cylinder_fillet(sph, cyl, spine, topo, radius, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Cylinder(cyl)) => {
            sphere_cylinder_fillet(sph, cyl, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Cone(cone)) => {
            sphere_cone_fillet(sph, cone, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Cone(cone), FaceSurface::Sphere(sph)) => {
            let mut result = sphere_cone_fillet(sph, cone, spine, topo, radius, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Cylinder(c1), FaceSurface::Cylinder(c2)) => {
            cylinder_cylinder_fillet(c1, c2, spine, topo, radius, face1, face2)
        }
        (FaceSurface::Cone(co1), FaceSurface::Cone(co2)) => {
            cone_cone_coaxial_fillet(co1, co2, spine, topo, radius, face1, face2)
        }
        // Pairs without an analytic path → walker fallback. Enumerated
        // exhaustively (matching `try_analytic_chamfer`) so adding a new
        // `FaceSurface` variant produces a compile error at this site
        // rather than silently routing through the walker.
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
        )
        | (
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
        )
        | (
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_),
        ) => Ok(None),
    }
}

/// Swap face1↔face2, pcurve1↔pcurve2, contact1↔contact2, and section.p1↔p2,
/// uv1↔uv2 in a `StripeResult`. Used when the analytic helper is called with
/// the canonical "plane first" ordering but the dispatcher saw the pair
/// reversed; the caller-facing `face1`/`face2` must reflect the original
/// ordering, not the helper's internal one.
fn swap_stripe_sides(r: &mut StripeResult) {
    std::mem::swap(&mut r.stripe.face1, &mut r.stripe.face2);
    std::mem::swap(&mut r.stripe.pcurve1, &mut r.stripe.pcurve2);
    std::mem::swap(&mut r.stripe.contact1, &mut r.stripe.contact2);
    for s in &mut r.stripe.sections {
        std::mem::swap(&mut s.p1, &mut s.p2);
        std::mem::swap(&mut s.uv1, &mut s.uv2);
    }
}

/// Try to compute a chamfer analytically for two surfaces.
///
/// Returns `Some(StripeResult)` if the surface pair has a closed-form solution,
/// `None` otherwise (caller should fall back to the walking engine).
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_arguments)]
pub fn try_analytic_chamfer(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    match (surf1, surf2) {
        (
            FaceSurface::Plane {
                normal: n1,
                d: _dd1,
            },
            FaceSurface::Plane {
                normal: n2,
                d: _dd2,
            },
        ) => {
            let result = plane_plane_chamfer(spine, topo, *n1, *n2, d1, d2, face1, face2)?;
            Ok(Some(result))
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Cylinder(cyl)) => {
            plane_cylinder_chamfer(*normal, *d, cyl, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Cylinder(cyl), FaceSurface::Plane { normal, d }) => {
            // Plane on side 2: swap d1↔d2 (so they refer to the right surface
            // in the canonical helper ordering) and swap result sides back.
            let mut result =
                plane_cylinder_chamfer(*normal, *d, cyl, spine, topo, d2, d1, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Cone(cone)) => {
            plane_cone_chamfer(*normal, *d, cone, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Cone(cone), FaceSurface::Plane { normal, d }) => {
            let mut result =
                plane_cone_chamfer(*normal, *d, cone, spine, topo, d2, d1, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Plane { normal, d }, FaceSurface::Sphere(sph)) => {
            plane_sphere_chamfer(*normal, *d, sph, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Plane { normal, d }) => {
            let mut result =
                plane_sphere_chamfer(*normal, *d, sph, spine, topo, d2, d1, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Sphere(s1), FaceSurface::Sphere(s2)) => {
            sphere_sphere_chamfer(s1, s2, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Cylinder(cyl)) => {
            sphere_cylinder_chamfer(sph, cyl, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Cylinder(cyl), FaceSurface::Sphere(sph)) => {
            let mut result = sphere_cylinder_chamfer(sph, cyl, spine, topo, d2, d1, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Sphere(sph), FaceSurface::Cone(cone)) => {
            sphere_cone_chamfer(sph, cone, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Cone(cone), FaceSurface::Sphere(sph)) => {
            let mut result = sphere_cone_chamfer(sph, cone, spine, topo, d2, d1, face2, face1)?;
            if let Some(ref mut r) = result {
                swap_stripe_sides(r);
            }
            Ok(result)
        }
        (FaceSurface::Cylinder(c1), FaceSurface::Cylinder(c2)) => {
            cylinder_cylinder_chamfer(c1, c2, spine, topo, d1, d2, face1, face2)
        }
        (FaceSurface::Cone(co1), FaceSurface::Cone(co2)) => {
            cone_cone_coaxial_chamfer(co1, co2, spine, topo, d1, d2, face1, face2)
        }
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
        )
        | (
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
        )
        | (
            FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_),
        ) => Ok(None),
    }
}

/// Make a degree-1 NURBS line between two 3D points.
fn nurbs_line(p0: Point3, p1: Point3) -> Result<NurbsCurve, BlendError> {
    let curve = NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![p0, p1], vec![1.0, 1.0])?;
    Ok(curve)
}

/// Compute the dihedral half-angle between two plane normals.
///
/// Returns the half-angle in radians. The angle is between 0 and pi/2
/// for convex edges and pi/2 to pi for concave edges.
fn dihedral_half_angle(n1: Vec3, n2: Vec3) -> f64 {
    let cos_angle = n1.dot(n2).clamp(-1.0, 1.0);
    cos_angle.acos() / 2.0
}

/// Compute the section plane basis from two plane normals and spine tangent.
///
/// Returns `(bisector, cross_dir)` where bisector points from edge toward
/// fillet center and `cross_dir` is perpendicular to both in the section plane.
fn section_basis(n1: Vec3, n2: Vec3, spine_tangent: Vec3) -> (Vec3, Vec3) {
    // Bisector of the two normals — points from the edge toward the fillet center
    let bisector_raw = n1 + n2;
    let bisector = bisector_raw.normalize().unwrap_or_else(|_| {
        // Normals are antiparallel (180 deg) — use cross product with tangent
        spine_tangent.cross(n1)
    });

    // In the section plane, the direction perpendicular to the bisector
    // that lies in the plane of the two normals.
    let cross_dir_raw = spine_tangent.cross(bisector);
    let cross_dir = cross_dir_raw
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    (bisector, cross_dir)
}

/// Compute the direction from edge toward contact point on a plane.
///
/// This is the component of the bisector projected onto the plane surface,
/// pointing away from the edge toward where the fillet touches the plane.
fn compute_contact_direction(normal: Vec3, bisector: Vec3) -> Vec3 {
    // Project bisector onto the plane (remove component along normal)
    let proj = bisector - normal * bisector.dot(normal);
    proj.normalize().unwrap_or(bisector)
}

/// Compute the midpoint of two 3D points.
fn midpoint_3d(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        f64::midpoint(a.x(), b.x()),
        f64::midpoint(a.y(), b.y()),
        f64::midpoint(a.z(), b.z()),
    )
}

/// Fillet between two planes: the result is a cylindrical surface.
///
/// # Geometry
///
/// Given two planes meeting at a straight edge:
/// - The fillet surface is a cylinder whose axis is parallel to the edge
/// - The cylinder radius equals the fillet radius
/// - The center is offset from the edge along the angle bisector
/// - Contact lines are straight lines on each plane
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn plane_plane_fillet(
    spine: &Spine,
    topo: &Topology,
    n1: Vec3,
    n2: Vec3,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<StripeResult, BlendError> {
    // Spine endpoints and tangent
    let p_start = spine.evaluate(topo, 0.0)?;
    let p_end = spine.evaluate(topo, spine.length())?;
    let tangent = spine.tangent(topo, 0.0)?;

    // Dihedral geometry
    let half_angle = dihedral_half_angle(n1, n2);
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();

    // Guard against degenerate cases (parallel or antiparallel normals)
    if sin_half.abs() < 1e-10 {
        return Err(BlendError::Math(brepkit_math::MathError::ZeroVector));
    }

    let (bisector, _cross_dir) = section_basis(n1, n2, tangent);

    // Center offset from the edge along the bisector
    let center_offset = radius / sin_half;

    // Cylinder origin: on the center line at the spine start
    let cyl_origin = p_start + bisector * center_offset;
    let cyl_axis = tangent;

    // Create the cylindrical surface
    let cylinder = CylindricalSurface::new(cyl_origin, cyl_axis, radius)?;

    // Contact point offsets from the edge
    // The contact point on each plane is at distance R/tan(half_angle) from the edge.
    let contact_offset = radius * cos_half / sin_half; // = R / tan(half_angle)

    // Direction from edge toward contact on each plane
    let contact_dir1 = compute_contact_direction(n1, bisector);
    let contact_dir2 = compute_contact_direction(n2, bisector);

    // Contact lines (straight lines on each plane)
    let c1_start = p_start + contact_dir1 * contact_offset;
    let c1_end = p_end + contact_dir1 * contact_offset;
    let c2_start = p_start + contact_dir2 * contact_offset;
    let c2_end = p_end + contact_dir2 * contact_offset;

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // PCurves: project 3D contact endpoints onto each face surface to get UV
    let pcurve1 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n1, 0.0);
        let (u0, v0) = adapter.project_point(c1_start);
        let (u1, v1) = adapter.project_point(c1_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };
    let pcurve2 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n2, 0.0);
        let (u0, v0) = adapter.project_point(c2_start);
        let (u1, v1) = adapter.project_point(c2_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };

    // Cross-sections at start and end
    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: cyl_origin,
        radius,
        uv1: (0.0, 0.0),
        uv2: (0.0, 0.0),
        t: 0.0,
    };
    let cyl_end = p_end + bisector * center_offset;
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: cyl_end,
        radius,
        uv1: (1.0, 0.0),
        uv2: (1.0, 0.0),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cylinder(cylinder),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };

    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// Chamfer between two planes: the result is a flat ruled surface (plane).
///
/// # Geometry
///
/// Given two planes meeting at an edge with chamfer distances d1, d2:
/// - The chamfer surface is a plane connecting two lines
/// - Line 1 is at distance d1 from the edge on plane 1
/// - Line 2 is at distance d2 from the edge on plane 2
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn plane_plane_chamfer(
    spine: &Spine,
    topo: &Topology,
    n1: Vec3,
    n2: Vec3,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<StripeResult, BlendError> {
    // Spine endpoints and tangent
    let p_start = spine.evaluate(topo, 0.0)?;
    let p_end = spine.evaluate(topo, spine.length())?;
    let tangent = spine.tangent(topo, 0.0)?;

    let (bisector, _cross_dir) = section_basis(n1, n2, tangent);

    // Contact directions on each plane
    let contact_dir1 = compute_contact_direction(n1, bisector);
    let contact_dir2 = compute_contact_direction(n2, bisector);

    // Contact lines at specified distances
    let c1_start = p_start + contact_dir1 * d1;
    let c1_end = p_end + contact_dir1 * d1;
    let c2_start = p_start + contact_dir2 * d2;
    let c2_end = p_end + contact_dir2 * d2;

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // The chamfer surface is a plane through the two contact lines.
    // Its normal is perpendicular to both the spine tangent and the line
    // connecting corresponding contact points.
    let chamfer_span = c2_start - c1_start;
    let chamfer_normal_raw = tangent.cross(chamfer_span);
    let chamfer_normal = chamfer_normal_raw
        .normalize()
        .map_err(|_| BlendError::Math(brepkit_math::MathError::ZeroVector))?;

    // Signed distance from origin
    let chamfer_d = chamfer_normal.dot(Vec3::new(c1_start.x(), c1_start.y(), c1_start.z()));

    // PCurves: project 3D contact endpoints onto each face surface to get UV
    let pcurve1 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n1, 0.0);
        let (u0, v0) = adapter.project_point(c1_start);
        let (u1, v1) = adapter.project_point(c1_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };
    let pcurve2 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n2, 0.0);
        let (u0, v0) = adapter.project_point(c2_start);
        let (u1, v1) = adapter.project_point(c2_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };

    // Sections at start and end
    let midpoint_start = midpoint_3d(c1_start, c2_start);
    let midpoint_end = midpoint_3d(c1_end, c2_end);
    let chamfer_radius = (c1_start - c2_start).length() / 2.0;

    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: midpoint_start,
        radius: chamfer_radius,
        uv1: (0.0, 0.0),
        uv2: (0.0, 0.0),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: midpoint_end,
        radius: chamfer_radius,
        uv1: (1.0, 0.0),
        uv2: (1.0, 0.0),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Plane {
            normal: chamfer_normal,
            d: chamfer_d,
        },
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };

    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// Fillet between a plane and a cylinder whose axis is parallel to the
/// plane normal.
///
/// Returns `Some(StripeResult)` with an exact toroidal blend surface for
/// both the convex "post on plate" case (cylinder face not reversed) and
/// the concave "hole through plate" case (cylinder face reversed). The
/// formulas differ only in the torus major radius:
///
///   - convex: `major = r_c + r`, plate-side contact at radial `r_c + r`
///     (outside the spine on the plate);
///   - concave: `major = r_c - r`, plate-side contact at radial `r_c - r`
///     (inside the spine on the plate).
///
/// In both cases the torus center sits one fillet radius "above" the
/// plane along `-n_p_inward` (the empty-wedge direction), the cylinder-
/// side contact circle has radius `r_c`, and the torus axis is the
/// cylinder axis. The active tube portion is a quarter of the small
/// circle in either case — `[π/2, π]` for convex and `[3π/2, 2π]` for
/// concave (mirror images about the equatorial plane).
///
/// Returns `None` (walker fallback) for cases the analytic path
/// doesn't cover:
///   - the cylinder axis isn't parallel to the plane normal,
///   - the spine geometry is too short or degenerate,
///   - the fillet radius exceeds the cylinder radius (would invert
///     `r_c - r` for the concave case or geometrically nest the convex
///     fillet inside the cylinder).
///
/// # Errors
///
/// Returns `BlendError` if topology lookups fail.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_cylinder_fillet(
    n_p_inward: Vec3,
    d_plane: f64,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face_plane: FaceId,
    face_cyl: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;

    let tol_ang = ANALYTIC_TOL_ANG;
    let tol_lin = ANALYTIC_TOL_LIN;

    // 1) Cylinder axis must be parallel (up to sign) to the inward plane
    //    normal — this is the perpendicular plane-cylinder case.
    let axis_c = cyl.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot.abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // 2) Detect concave ("hole through plate") vs convex ("post on plate").
    //    The cylinder face's `reversed` flag tells us which side of the
    //    cylinder lateral the surrounding material lives on:
    //      * not reversed: material is on the cylinder's *inward* side
    //        (a solid post). Convex external corner.
    //      * reversed:     material is on the cylinder's *outward* side
    //        (a hole through a slab). Concave internal corner.
    let concave = topo.face(face_cyl)?.is_reversed();

    // 3) Radius bound depends on the case:
    //    - Convex: major = `r_c + r`, always > minor = `r`, so the only
    //      regime to reject is `r ≥ r_c` (rolling ball would encircle the
    //      cylinder axis).
    //    - Concave: major = `r_c - r`; needs `r < r_c` to keep major
    //      positive *and* `r ≤ r_c/2` to keep major ≥ minor. Past `r_c/2`
    //      the construction becomes a spindle (self-intersecting) torus
    //      which is invalid as a fillet surface.
    let r_c = cyl.radius();
    let max_radius = if concave { r_c * 0.5 } else { r_c };
    if radius <= tol_lin || radius >= max_radius {
        return Ok(None);
    }

    // 4) Project the cylinder origin onto the plane along axis_c. The
    //    projection lands on the spine plane and on the cylinder axis line.
    //    For axis_c ∥ n_p_inward, this is just `o_c ± n_p_inward * step`
    //    where `step` solves `(o_c + n_p_inward * step) · n_p_inward = d`.
    let o_c = cyl.origin();
    let step = d_plane - n_p_inward.dot(Vec3::new(o_c.x(), o_c.y(), o_c.z()));
    let p_axis_on_plane = o_c + n_p_inward * step;

    // 5) The torus center sits one fillet radius "above" the spine plane
    //    along the side opposite the plane material (`-n_p_inward`) for
    //    both convex and concave — the empty wedge is on the same side of
    //    the plate in either case. Major radius differs: `r_c + r` for
    //    convex (plate contact OUTSIDE the spine), `r_c - r` for concave
    //    (plate contact INSIDE the spine).
    let torus_center = p_axis_on_plane - n_p_inward * radius;
    let major_radius = if concave { r_c - radius } else { r_c + radius };
    let minor_radius = radius;

    // 6) Spine must span an arc to be useful. `Spine::from_single_edge`
    //    measures CHORD length, which is zero for a closed-circle spine
    //    (start vertex == end vertex), so we detect closure by walking the
    //    spine's first edge directly.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 7) Build the torus. Use the `with_axis_and_ref_dir` constructor so the
    //    torus inherits a u=0 reference direction matching the cylinder
    //    frame — this lets us derive (u, v) parameters for sections cleanly.
    let cyl_x = cyl.x_axis();
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        axis_c,
        cyl_x,
    )?;

    // 8) Spine endpoints in 3D and corresponding cylinder u-parameters.
    //    For a closed spine (full revolution) we span [u_start, u_start + 2π]
    //    regardless of where the projection lands. Otherwise we disambiguate
    //    the seam so u_end > u_start when the spine sweeps CCW.
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = ParametricSurface::project_point(cyl, p_spine_start).0;
    let u_end = if is_closed_spine {
        u_start + 2.0 * std::f64::consts::PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = ParametricSurface::project_point(cyl, p_spine_end).0;
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * std::f64::consts::PI
        }
    };

    // 9) 3D contact curves.
    //    - On the plane: a circle of radius `R = r_c + r` around the cylinder
    //      axis on the spine plane.
    //    - On the cylinder: a circle of radius `r_c` at axial offset `r`
    //      (the height of the ball trajectory above the plane).
    let z_axis_dir = -n_p_inward; // Direction "out of the plate" along the cylinder axis side.
    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        axis_c,
        major_radius,
        cyl_x,
        cyl.y_axis(),
    )?;
    let contact_cyl_center = p_axis_on_plane + z_axis_dir * radius;
    let contact_cyl_circle = brepkit_math::curves::Circle3D::with_axes(
        contact_cyl_center,
        axis_c,
        r_c,
        cyl_x,
        cyl.y_axis(),
    )?;

    // Convert to arc NURBS over [u_start, u_end].
    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_cyl = circle_arc_to_nurbs(&contact_cyl_circle, u_start, u_end)?;

    // 10) PCurves.
    //     - On the plane (face_plane): use PlaneAdapter so UV matches what
    //       the rest of the fillet pipeline expects for plane faces.
    //     - On the cylinder (face_cyl): contact runs at constant axial
    //       offset (v = `radius`, i.e. the height above the plane along the
    //       cylinder's v parameter). PCurve is a horizontal Line2D in (u, v)
    //       UV-space spanning u_start → u_end at v = `r`.
    let v_cyl = cyl_v_at_point(cyl, contact_cyl_center);
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);

    // The plane contact is always an arc on a circle (the rolling-ball
    // trajectory at z=0), so represent the pcurve as `Curve2D::Circle` in
    // the plane's local frame. A line-segment pcurve would zero out for the
    // closed-spine case (start and end project to the same point).
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            major_radius,
        )?)
    };
    let pcurve_cyl = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_cyl),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 11) Cross-sections at the spine endpoints. `uv1` is the plane contact
    //     in the PlaneAdapter local frame; `uv2` is the cylinder contact in
    //     `(u, v)` cylinder UV. The plane has no native UV — we use the same
    //     adapter as `pcurve_plane` so any downstream consumer gets a
    //     consistent local-frame pair instead of zeros or cylinder coords.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_cyl_at = |u: f64| contact_cyl_circle.evaluate(u);
    let center_at = |u: f64| {
        // Ball trajectory: same circle as `contact_plane_circle` but lifted
        // to the height of the cylinder contact (axial offset `r`).
        contact_plane_circle.evaluate(u) + z_axis_dir * radius
    };
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_start = CircSection {
        p1: p_plane_at(u_start),
        p2: p_cyl_at(u_start),
        center: center_at(u_start),
        radius,
        uv1: plane_uv_at(u_start),
        uv2: (u_start, v_cyl),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: p_plane_at(u_end),
        p2: p_cyl_at(u_end),
        center: center_at(u_end),
        radius,
        uv1: plane_uv_at(u_end),
        uv2: (u_end, v_cyl),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_cyl,
        contact1: contact_plane,
        contact2: contact_cyl,
        face1: face_plane,
        face2: face_cyl,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Recover the cylinder's axial v-parameter for a 3D point known to lie on
/// the cylinder lateral.
fn cyl_v_at_point(cyl: &brepkit_math::surfaces::CylindricalSurface, p: Point3) -> f64 {
    let axis = cyl.axis();
    let to_p = p - cyl.origin();
    axis.dot(to_p)
}

/// Chamfer between a plane and a cylinder whose axis is parallel to the
/// plane normal, for the convex bottom-rim case.
///
/// `d1` is the chamfer distance on the plane (radially inward from the
/// spine on the plate face); `d2` is the distance on the cylinder lateral
/// (axially into the material from the spine).
///
/// # Geometry
///
/// The chamfer surface is a frustum of a cone:
///   - axis = cylinder axis (the cone's `+axis_c` direction points toward
///     the cylinder material; `v` grows from apex into the material);
///   - half-angle `α = atan2(d1, d2)` (45° for symmetric `d1 = d2`),
///   - apex on the cylinder axis, axial offset `(r_c - d1)·d2/d1` away
///     from the cylinder material (in the empty-wedge half-space — i.e.
///     opposite to the cylinder body relative to the plate);
///   - contact 1 on the plate: circle at radial `r_c - d1`, on the plate;
///   - contact 2 on the cylinder lateral: circle at radial `r_c`, axially
///     offset `+d2` into the material.
///
/// Both contacts are circles around the cylinder axis. The chamfer face
/// connects them with a flat cone (ruled surface).
///
/// Returns `None` (walker fallback) when:
///   - the cylinder axis isn't parallel to the plane normal,
///   - the cylinder face is reversed (concave / hole),
///   - either chamfer distance is non-positive or `d1 >= r_c` (would
///     pass through the cylinder axis), or
///   - the spine is too short.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_cylinder_chamfer(
    n_p_inward: Vec3,
    d_plane: f64,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face_plane: FaceId,
    face_cyl: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_ang = ANALYTIC_TOL_ANG;
    let tol_lin = ANALYTIC_TOL_LIN;

    // 1) Cylinder axis must be parallel (up to sign) to the inward plane
    //    normal — perpendicular plane-cylinder configuration.
    let axis_c = cyl.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot.abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // 2) Detect convex ("post on plate") vs concave ("hole through plate")
    //    via the cylinder face's `reversed` flag, mirroring the fillet
    //    path. The geometry differs only in a single signed factor.
    let concave = topo.face(face_cyl)?.is_reversed();
    let signed_offset: f64 = if concave { -1.0 } else { 1.0 };

    // 3) Both distances must be positive. The convex case additionally
    //    requires `d1 < r_c` (so the plate-side contact at radial
    //    `r_c − d1` doesn't pass through the cylinder axis); the concave
    //    case has no upper bound from the cylinder geometry since plate
    //    contact lives at `r_c + d1` (always outside the spine).
    let r_c = cyl.radius();
    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    if !concave && d1 >= r_c {
        return Ok(None);
    }

    // 4) Project the cylinder origin onto the plate.
    let o_c = cyl.origin();
    let step = d_plane - n_p_inward.dot(Vec3::new(o_c.x(), o_c.y(), o_c.z()));
    let p_axis_on_plane = o_c + n_p_inward * step;

    // 5) The chamfer dispatcher does NOT apply `orient_plane_surface`, so
    //    `n_p_inward` here is the face's raw geometric outward normal.
    //    Material lives on `-n_p_inward` for both convex AND concave cases,
    //    so the cylinder-side contact (at axial offset `d2 along
    //    -n_p_inward`) is built identically. The chamfer cone's apex
    //    sits in the `−ẑ` direction in absolute coords for *both* cases,
    //    but expressed relative to `n_p_inward` it differs:
    //      * Convex (`s = +1`): apex direction = `+n_p_inward` (the
    //        empty-wedge side, where the rolling-ball-equivalent lives).
    //      * Concave (`s = -1`): apex direction = `-n_p_inward` (the
    //        material side; the cone *opens* upward through the plate
    //        toward the empty wedge inside the hole).
    //    We bake this into a single `apex_dir = s · n_p_inward` factor.
    let axis_toward_material = -n_p_inward;

    // 6) Spine: detect closed-circle case so we can spin a full 2π.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 7) Build the chamfer cone. brepkit's `ConicalSurface` measures
    //    `half_angle` from the AXIS to the generator, so the radial
    //    component per unit v is `cos(β)` and the axial is `sin(β)`.
    //    Generator slope `dr/dz = cos β / sin β = cot β`, matching our
    //    generator's `d1/d2` ratio (same in both cases — the sign of
    //    Δr and Δz both flip together going from convex to concave).
    //    So `β = atan2(d2, d1)` for either case.
    let half_angle = d2.atan2(d1);
    // Plate-side contact radius:
    //   - convex (s = +1): r_c − d1 (inside the spine, into post material)
    //   - concave (s = −1): r_c + d1 (outside the spine, into surrounding
    //     plate material around the hole)
    let plate_contact_radius = r_c - signed_offset * d1;
    // Apex magnitude (always positive): plate_contact_radius · d2 / d1.
    // The factor (r_c − s·d1) is exactly `plate_contact_radius`, so the
    // formula is uniform across cases.
    let apex_offset = plate_contact_radius * d2 / d1;
    let apex_dir = n_p_inward * signed_offset;
    let apex_pos = p_axis_on_plane + apex_dir * apex_offset;
    // Cone opens in the opposite direction from the apex so v grows from
    // apex through the plate toward (in convex) or past (in concave) the
    // cylinder material side.
    let cone_axis = -apex_dir;
    let cyl_x = cyl.x_axis();
    let cone = ConicalSurface::with_ref_dir(apex_pos, cone_axis, half_angle, cyl_x)?;

    // 8) 3D contact curves: both are circles around the cylinder axis.
    let cone_y = cyl.y_axis();
    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        axis_c,
        plate_contact_radius,
        cyl_x,
        cone_y,
    )?;
    let cyl_contact_center = p_axis_on_plane + axis_toward_material * d2;
    let contact_cyl_circle =
        brepkit_math::curves::Circle3D::with_axes(cyl_contact_center, axis_c, r_c, cyl_x, cone_y)?;

    // 9) Spine angular range, derived from the cylinder's u-parameter
    //    projection of the endpoints.
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = ParametricSurface::project_point(cyl, p_spine_start).0;
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = ParametricSurface::project_point(cyl, p_spine_end).0;
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_cyl = circle_arc_to_nurbs(&contact_cyl_circle, u_start, u_end)?;

    // 10) PCurves.
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            r_c - d1,
        )?)
    };
    let v_cyl = cyl_v_at_point(cyl, cyl_contact_center);
    let pcurve_cyl = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_cyl),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 11) Cross-sections at the spine endpoints. The chamfer "section"
    //     is the straight segment between the two contacts (no rolling
    //     ball). Use the segment midpoint as the section center and the
    //     half-length as the section radius — `CircSection` is shaped for
    //     fillets but the field semantics still describe the chord.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_cyl_at = |u: f64| contact_cyl_circle.evaluate(u);
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_at = |u: f64, t: f64| {
        let p1 = p_plane_at(u);
        let p2 = p_cyl_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: plane_uv_at(u),
            uv2: (u, v_cyl),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(cone),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_cyl,
        contact1: contact_plane,
        contact2: contact_cyl,
        face1: face_plane,
        face2: face_cyl,
        sections: vec![section_start, section_end],
    };

    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between a plane and a cone whose axis is parallel to the plane
/// normal, for the convex "regular frustum bottom rim" geometry.
///
/// Returns `Some(StripeResult)` with an exact toroidal blend when the cone
/// opens *toward* the plate (cone axis anti-parallel to the inward plane
/// normal — this is the configuration where filleting the bottom rim of a
/// frustum makes the corner convex from outside). Returns `None` for any
/// other configuration so the walker handles it.
///
/// # Geometry
///
/// At the spine point, the dihedral between outward surface normals is
/// `π - α` (where α is the cone half-angle), so the fillet wedge half-angle
/// is `α/2` and the rolling-ball center sits at distance `r/sin(α/2)`
/// along the outward bisector `cos(α/2)·radial - sin(α/2)·n_p_inward`
/// (convex) or `-cos(α/2)·radial - sin(α/2)·n_p_inward` (concave).
///
/// Convex / concave is detected via `face_cone.is_reversed()`. The two
/// cases share torus center placement (one fillet radius "below" the
/// plate along `-n_p_inward`), minor radius (`r`), and the cone axis
/// direction. They differ only in the major radius:
///   - Convex (face_cone not reversed): `major = r_p + r·cot(α/2)`,
///     plate contact at radial `r_p + r·cot(α/2) - r·sin α` outside the
///     spine. Geometric "post-on-plate" frustum bottom rim.
///   - Concave (face_cone reversed): `major = r_p − r·cot(α/2)`,
///     plate contact INSIDE the spine. Geometric "tapered hole through
///     plate" — the rolling ball lives inside the hole and above the
///     plate material.
///
/// At α = π/2 (degenerate "cone" approaching a cylinder), `cot(π/4) = 1`
/// so the formulas collapse to `major = r_p ± r`, matching
/// `plane_cylinder_fillet`'s convex/concave branches.
///
/// Returns `None` when:
///   - the cone axis isn't parallel to the plane normal,
///   - `axis_c · n_p_inward > -1 + tol_ang` (cone opens *away* from the
///     plate — inverted-frustum or cup geometry; the major-radius formula
///     differs and is left to the walker),
///   - the half-angle α is too close to 0 or π/2 (degenerate),
///   - the spine is too short,
///   - the apex is on the plate-material side, or
///   - the radius produces a degenerate or self-intersecting torus
///     (concave: `r·cot(α/2) ≥ r_p` makes major non-positive, and
///     `r·(cot(α/2) + 1) ≥ r_p` produces a spindle torus; convex always
///     non-spindle since `r·cot(α/2) ≥ 0`).
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_cone_fillet(
    n_p_inward: Vec3,
    d_plane: f64,
    cone: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face_plane: FaceId,
    face_cone: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_ang = ANALYTIC_TOL_ANG;
    let tol_lin = ANALYTIC_TOL_LIN;

    // 1) Cone axis must be parallel (up to sign) to the inward plane
    //    normal — both cases boil down to "axis points along the plate
    //    normal." The two valid configurations differ in sign:
    //       - Convex (post on plate): apex sits on the same side of the
    //         plate as the cone material, so `axis_c · n_p_inward = -1`.
    //       - Concave (tapered hole): apex sits on the empty-wedge side
    //         (across the plate from the cone material), so
    //         `axis_c · n_p_inward = +1`.
    //    Either way `|n_dot| ≈ 1` must hold; the sign distinguishes the
    //    two cases and is cross-checked against `face_cone.is_reversed()`
    //    below.
    let axis_c = cone.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot.abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // 2) Detect concave ("tapered hole through plate") vs convex ("post on
    //    plate") via the cone face's `reversed` flag. Both cases share
    //    torus-center placement and tube structure; they differ only in
    //    the sign of the `r·cot(α/2)` major-radius term.
    let concave = topo.face(face_cone)?.is_reversed();

    // 3) Reject degenerate half-angles. Too close to 0 → flat disk; too
    //    close to π/2 → cylinder limit (callers should hit
    //    `plane_cylinder_fillet` instead since the surface tag would be
    //    `Cylinder`, not `Cone`, for that case).
    let alpha = cone.half_angle();
    if alpha <= 1e-3 || alpha >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }
    let half_alpha = alpha * 0.5;
    let cot_half = half_alpha.tan().recip();

    // 4) Apex projection onto the plate. `step` is the signed distance
    //    you move along `n_p_inward` from the apex to land on the plate.
    //    The valid sign depends on the case:
    //       - Convex: apex on the material side ⇒ `step < 0` (you must
    //         move along `+n_p_inward` to reach the plate, but `step` is
    //         the projection sign which lands negative under
    //         `d_plane − n_p_inward·apex`).
    //       - Concave: apex on the empty-wedge side ⇒ `step > 0`.
    //    Reject `step ≈ 0` (apex on the plate ⇒ degenerate `r_p = 0`).
    let apex = cone.apex();
    let step = d_plane - n_p_inward.dot(Vec3::new(apex.x(), apex.y(), apex.z()));
    if step.abs() <= tol_lin {
        return Ok(None);
    }
    // Cross-check the case against the apex-side: convex requires `step < 0`
    // and concave requires `step > 0`. If they disagree the topology is
    // not the regular-frustum geometry the formulas below assume.
    if (concave && step <= 0.0) || (!concave && step >= 0.0) {
        return Ok(None);
    }
    let apex_height = step.abs();
    let p_axis_on_plane = apex + n_p_inward * step;

    // 5) Spine radius `r_p = apex_height · cot(α)` (geometric: the cone-plate
    //    intersection circle has this radius).
    let r_p = apex_height * (alpha.cos() / alpha.sin());

    // 6) Major / minor radii and torus center. Convex adds `r·cot(α/2)`
    //    to the spine radius; concave subtracts it. Concave additionally
    //    needs `r·(cot(α/2) + 1) ≤ r_p` to keep `major ≥ minor`
    //    (otherwise the construction becomes a spindle torus, which is
    //    invalid as a fillet surface). The convex case is always
    //    non-spindle since `r·cot(α/2) ≥ 0`.
    let signed_offset = if concave { -1.0 } else { 1.0 };
    let major_radius = r_p + signed_offset * radius * cot_half;
    let minor_radius = radius;
    if major_radius <= tol_lin {
        return Ok(None);
    }
    // `major - minor < tol` rejects both the spindle regime AND the
    // horn-torus boundary (`major == minor`, where the tube touches the
    // axis at a degenerate point). Tolerance lets us catch the boundary
    // even when floating-point rounding leaves the difference at +ε.
    if concave && major_radius - minor_radius < tol_lin {
        return Ok(None);
    }
    // Torus center sits one fillet radius below the plate (in the
    // -n_p_inward direction, where the empty wedge is).
    let torus_center = p_axis_on_plane - n_p_inward * radius;
    // Torus axis = -n_p_inward (= +axis_c for the regular-frustum case
    // where axis_c · n_p_inward = -1). With this convention sin(v) points
    // away from the plate, so plate contact is at v = 3π/2 (sin v = -1
    // pulls the tube point back toward +n_p_inward) and cone contact is
    // at v = atan2(cos α, -sin α).
    let axis_dir = -n_p_inward;

    // 7) Spine: detect closed-circle case so we can spin a full 2π without
    //    relying on `Spine::length()` (which measures chord length and is
    //    zero for closed-loop edges).
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 8) Build the torus. The torus's ref direction is the cone's x_axis so
    //    its angular u parameter aligns with the cone's u parameter.
    let cone_x = cone.x_axis();
    let cone_y = cone.y_axis();
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        axis_dir,
        cone_x,
    )?;

    // 9) Spine angular range. Project endpoints into the (cone_x, cone_y)
    //    plane to recover their u parameter.
    let u_at = |p: Point3| {
        let v = p - p_axis_on_plane;
        cone_y.dot(v).atan2(cone_x.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // 10) 3D contact curves.
    //     Plate contact: circle of radius `major_radius` around the cone
    //       axis, on the plate.
    //     Cone contact: circle on the analytical cone surface; for
    //       convex it lands BELOW the plate at axial `-r·(1 + cos α)`
    //       (on the cone's analytical extension below the frustum), and
    //       for concave ABOVE the plate at `+r·(1 + cos α)` (between
    //       apex and plate). The axial direction toward both is the
    //       empty-wedge direction `-n_p_inward`. The radial offset from
    //       `major_radius` to the cone-side contact also flips sign:
    //       `-r·sin α` for convex (contact tucks INSIDE the spine on
    //       the cone-extension side) and `+r·sin α` for concave (contact
    //       hangs OUTSIDE the inner-hole spine on the cone above).
    let contact_plane_radius = major_radius;
    let contact_cone_radius = (major_radius - signed_offset * radius * alpha.sin()).max(tol_lin);
    let contact_cone_axial_magnitude = radius * (1.0 + alpha.cos());
    let cone_contact_center = p_axis_on_plane + (-n_p_inward) * contact_cone_axial_magnitude;

    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        axis_dir,
        contact_plane_radius,
        cone_x,
        cone_y,
    )?;
    let contact_cone_circle = brepkit_math::curves::Circle3D::with_axes(
        cone_contact_center,
        axis_dir,
        contact_cone_radius,
        cone_x,
        cone_y,
    )?;

    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_cone = circle_arc_to_nurbs(&contact_cone_circle, u_start, u_end)?;

    // 11) PCurves.
    //     Plane contact is a `Curve2D::Circle` in the PlaneAdapter local
    //     frame (a Line2D would zero out for the closed-spine case).
    //     Cone contact runs at constant `v_cone` in the cone's UV; v_cone
    //     is recovered by projecting the cone-contact center onto the cone.
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            major_radius,
        )?)
    };
    let v_cone = ParametricSurface::project_point(cone, cone_contact_center).1;
    let pcurve_cone = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_cone),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 12) Cross-sections at the spine endpoints.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_cone_at = |u: f64| contact_cone_circle.evaluate(u);
    let center_at = |u: f64| {
        // Ball trajectory: same circle as `contact_plane_circle` but lifted
        // by `-r·n_p_inward` (one fillet radius into the empty wedge).
        contact_plane_circle.evaluate(u) + (-n_p_inward) * radius
    };
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_start = CircSection {
        p1: p_plane_at(u_start),
        p2: p_cone_at(u_start),
        center: center_at(u_start),
        radius,
        uv1: plane_uv_at(u_start),
        uv2: (u_start, v_cone),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: p_plane_at(u_end),
        p2: p_cone_at(u_end),
        center: center_at(u_end),
        radius,
        uv1: plane_uv_at(u_end),
        uv2: (u_end, v_cone),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_cone,
        contact1: contact_plane,
        contact2: contact_cone,
        face1: face_plane,
        face2: face_cone,
        sections: vec![section_start, section_end],
    };

    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between a plane and a cone whose axis is parallel to the plane
/// normal, for the convex regular-frustum bottom-rim case.
///
/// `d1` is the chamfer distance on the plate (radially inward from the
/// spine on the plate face); `d2` is the distance along the cone's
/// generator (going from the spine toward the apex into the cylinder
/// material).
///
/// # Geometry
///
/// At a frustum bottom rim with cone half-angle `α`, the plate-side
/// contact is a circle at radial `r_p - d1` on the plate, while the
/// cone-side contact lands at radial `r_p - d2·cos α` and axial offset
/// `+d2·sin α` into the cylinder material. Connecting these two
/// concentric circles with a flat ruled surface gives a cone:
///
///   - chamfer half-angle `β = atan2(d2·sin α, d1 - d2·cos α)`
///     (collapses to `β = π/2 - α/2` for symmetric `d1 = d2`, and to
///     `β = π/4` in the cylinder limit `α → π/2` — matching
///     `plane_cylinder_chamfer`);
///   - apex on the cone axis, axial offset
///     `(r_p - d1)·d2·sin α / (d1 - d2·cos α)` *out* of the cylinder
///     material (in the empty-wedge half-space);
///   - axis parallel to the cone's axis, oriented so `+axis_c` points
///     into the cylinder material (cone evaluation walks from apex,
///     across the plate, into the material as `v` grows).
///
/// Returns `None` (walker fallback) for any case the analytic path
/// doesn't yet cover:
///   - cone axis not anti-parallel to the inward plane normal,
///   - cone face reversed (concave / "tapered hole"),
///   - half-angle α too close to 0 or π/2 (degenerate),
///   - apex on the plate-material side (`step >= 0`),
///   - either chamfer distance non-positive,
///   - `d1 >= r_p` (would pass through cone axis),
///   - `d1 - d2·cos α <= 0` (chamfer "flares outward" on the cone — apex
///     would land above the plate, distinct geometric configuration).
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_cone_chamfer(
    n_p_inward: Vec3,
    d_plane: f64,
    cone: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face_plane: FaceId,
    face_cone: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_ang = ANALYTIC_TOL_ANG;
    let tol_lin = ANALYTIC_TOL_LIN;

    // 1) Cone axis must be (anti)parallel to the raw plate normal. The
    //    chamfer dispatcher does NOT apply `orient_plane_surface`, so
    //    `n_p_inward` here is actually the face's raw geometric (outward)
    //    normal. Two configurations are accepted:
    //      • convex (frustum-as-post-on-plate): axis_c ∥ n_p_inward
    //        (`axis_c · n_p_inward ≈ +1`); cone primitive's apex sits on
    //        the +n_p_inward side of the plate.
    //      • concave (frustum-as-hole-tool, top rim): axis_c ⫯ n_p_inward
    //        (`axis_c · n_p_inward ≈ -1`); cone primitive's apex sits on
    //        the -n_p_inward side of the plate.
    //    The sign of `n_dot` distinguishes these and drives the
    //    `signed_offset = ±1` factor that flips contact-radius and
    //    apex-direction signs throughout.
    let axis_c = cone.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot.abs() < 1.0 - tol_ang {
        return Ok(None);
    }
    let signed_offset: f64 = if n_dot > 0.0 { 1.0 } else { -1.0 };
    let concave = signed_offset < 0.0;
    // Cross-check geometric sign against the topological flag — if a
    // caller hands us a non-reversed face whose cone axis happens to be
    // antiparallel to the plate normal (or vice versa), the geometry-only
    // detection above would silently apply the wrong formula. Hard-bail
    // (release + debug) on disagreement so callers fall back to the
    // walker rather than getting a malformed analytic stripe.
    if concave != topo.face(face_cone)?.is_reversed() {
        return Ok(None);
    }

    // 2) Validate half-angle and chamfer distances.
    let alpha = cone.half_angle();
    if alpha <= 1e-3 || alpha >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }
    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }

    // 3) Apex projection onto the plate. With raw normals, `step` is
    //    positive for convex (apex on +n_p_inward side) and negative for
    //    concave; we require the magnitude exceed the linear tol and
    //    cross-check that its sign agrees with `signed_offset`.
    let apex = cone.apex();
    let step = d_plane - n_p_inward.dot(Vec3::new(apex.x(), apex.y(), apex.z()));
    if step.abs() <= tol_lin {
        return Ok(None);
    }
    if step * signed_offset <= 0.0 {
        return Ok(None);
    }
    let apex_height = step.abs();
    let p_axis_on_plane = apex + n_p_inward * step;

    // 4) Spine radius from cone-plate intersection.
    let r_p = apex_height * (alpha.cos() / alpha.sin());
    // For convex: contact_plane sits at radius `r_p - d1` so we need
    //   `d1 < r_p`. For concave: contact_plane is at `r_p + d1`, no upper
    //   bound from cone geometry (plate extends radially).
    if !concave && d1 >= r_p {
        return Ok(None);
    }

    // 6) Compute chamfer cone parameters via 2D (radial, axial) generator
    //    direction connecting the two contact points.
    let (sin_a, cos_a) = alpha.sin_cos();
    let dr = d1 - d2 * cos_a;
    let dz = d2 * sin_a;
    if dz <= tol_lin {
        return Ok(None);
    }
    // V1 only handles `dr > 0` (chamfer "tilts inward" on the cone side
    // — apex below the plate). The `dr <= 0` case ("outward-flaring"
    // chamfer) needs a different apex placement.
    if dr <= tol_lin {
        return Ok(None);
    }
    // brepkit's `ConicalSurface` measures the half-angle from the AXIS to
    // the generator (so the radial component of `position(0, v)` per unit v
    // is `cos(β)`, the axial component is `sin(β)`, and the generator slope
    // in (r, z) is `cot β = cos β / sin β`). Matching that to our generator
    // slope `dr/dz`: cot β = dr/dz ⇒ tan β = dz/dr ⇒ β = atan2(dz, dr).
    // For symmetric `d1 = d2` and frustum half-angle α this collapses to
    // `β = π/2 − α/2`, and to `β = π/4` in the α → π/2 cylinder limit.
    let chamfer_half_angle = dz.atan2(dr);
    if chamfer_half_angle <= 1e-3 || chamfer_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }

    // 7) Apex of the chamfer cone — extrapolate the generator from the
    //    plate-side contact backward (or forward, in the concave case) to
    //    the axis.
    //
    //    Plate-contact radius = `r_p − signed_offset · d1`, generator slope
    //    `dr/dz = (d1 − d2·cos α)/(d2·sin α)` (same magnitude in both
    //    cases). The chamfer-cone apex lands on the same side of the plate
    //    as the cone primitive's "open" side (i.e. on the +n_p_inward side
    //    for convex and the −n_p_inward side for concave); both reduce to
    //    `axis_toward_apex = n_p_inward · signed_offset`. Note that
    //    `axis_toward_apex` and `axis_toward_material` only coincide in
    //    sign for the convex case — for concave the chamfer apex sits on
    //    the OPPOSITE side from the plate material it tucks into, so we
    //    track them as independent directions:
    //      • `axis_toward_apex` — apex placement (z = ∓mag below/above)
    //      • `chamfer_axis` — direction the chamfer cone opens (always
    //        −axis_toward_apex, so it grows from apex through the plate
    //        into the empty wedge)
    //      • `axis_into_material` — direction from spine into plate
    //        material; always equal to `−n_p_inward` (NOT
    //        `−axis_toward_apex`, which differs from this in concave).
    let plate_contact_radius = r_p - signed_offset * d1;
    let chamfer_apex_offset = plate_contact_radius * dz / dr;
    let axis_toward_apex = n_p_inward * signed_offset;
    let chamfer_apex_pos = p_axis_on_plane + axis_toward_apex * chamfer_apex_offset;
    let chamfer_axis = -axis_toward_apex;
    let axis_into_material = -n_p_inward;

    // 8) Spine: detect closed-circle case so we can spin a full 2π without
    //    relying on `Spine::length()` (chord-based, zero for closed loops).
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 9) Build the chamfer cone.
    let cone_x = cone.x_axis();
    let cone_y = cone.y_axis();
    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, chamfer_axis, chamfer_half_angle, cone_x)?;

    // 10) 3D contact circles. Both lie around the cone axis.
    //     Concave flips the cone-side contact circle's radius: instead of
    //     `r_p − d2·cos α` (post case, contact moves inward toward axis)
    //     it becomes `r_p + d2·cos α` (hole case, contact moves outward
    //     into surrounding plate material). The axial offset direction
    //     `axis_into_material` is always `−n_p_inward` regardless of
    //     convex/concave.
    let cone_contact_radius = r_p - signed_offset * d2 * cos_a;
    let cone_contact_axial_offset = d2 * sin_a;
    let cone_contact_center = p_axis_on_plane + axis_into_material * cone_contact_axial_offset;

    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        axis_c,
        plate_contact_radius,
        cone_x,
        cone_y,
    )?;
    let contact_cone_circle = brepkit_math::curves::Circle3D::with_axes(
        cone_contact_center,
        axis_c,
        cone_contact_radius,
        cone_x,
        cone_y,
    )?;

    // 11) Spine angular range, derived from the cone's u parameter
    //     projection of the endpoints.
    let u_at = |p: Point3| {
        let v = p - p_axis_on_plane;
        cone_y.dot(v).atan2(cone_x.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_cone = circle_arc_to_nurbs(&contact_cone_circle, u_start, u_end)?;

    // 12) PCurves.
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            plate_contact_radius,
        )?)
    };
    let v_cone = ParametricSurface::project_point(cone, cone_contact_center).1;
    let pcurve_cone = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_cone),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 13) Cross-sections at the spine endpoints.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_cone_at = |u: f64| contact_cone_circle.evaluate(u);
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_at = |u: f64, t: f64| {
        let p1 = p_plane_at(u);
        let p2 = p_cone_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: plane_uv_at(u),
            uv2: (u, v_cone),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_cone,
        contact1: contact_plane,
        contact2: contact_cone,
        face1: face_plane,
        face2: face_cone,
        sections: vec![section_start, section_end],
    };

    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between a plane and a sphere whose center sits along the plate
/// normal. Handles all four sub-configurations of plane × sphere fillet
/// via a unified `signed_offset = ±1` factor:
///
///   1. Convex post-on-slab — sphere face NOT reversed, sphere center on
///      the empty-wedge side (`h_signed < 0`, e.g. a hemisphere on a plate
///      slab). Rolling ball **externally** tangent to sphere (`R + r`).
///   2. Convex sphere-buried — sphere face NOT reversed, sphere center on
///      the plate-material side (`h_signed > 0`, e.g. half-buried sphere).
///   3. Concave spherical pocket — sphere face REVERSED, sphere center on
///      plate-material side (`h_signed > 0`). Rolling ball **internally**
///      tangent to sphere (`R − r`); ball is INSIDE the pocket air.
///   4. Concave spherical hole-through-plate — sphere face REVERSED,
///      sphere center on empty-wedge side (`h_signed < 0`). Rolling ball
///      internally tangent, INSIDE the hole.
///
/// All four collapse to a single closed-form torus blend; the formulas
/// differ only in the sign of one term.
///
/// # Geometry
///
/// Let `h_signed = (sphere_center − p_axis_on_plane) · n_p_inward` (signed)
/// and `R = sphere.radius()`. With `signed_offset = +1` for the convex
/// (face not reversed) configuration and `signed_offset = −1` for concave
/// (reversed):
///
///   - spine radius `r_p = √(R² − h_signed²)`;
///   - rolling-ball axial offset along `n_p_inward`: `−signed_offset · r`
///     (convex puts the ball on the −n_p_inward side / empty wedge,
///     concave on the +n_p_inward side / inside the cavity);
///   - **major radius** `R_t² = r_p² + signed_offset · 2r·(R − h_signed)`;
///   - fillet surface: torus with axis ⊥ plate, major `R_t`, minor `r`;
///   - plate-side contact: circle of radius `R_t` on the spine plane;
///   - sphere-side contact: circle at radial `R · R_t / (R + signed_offset·r)`,
///     axially offset `signed_offset · r · (h_signed − R) / (R + signed_offset·r)`
///     along `n_p_inward`.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere doesn't intersect the plate (`|h_signed| ≥ R`),
///   - the spindle bound is exceeded (`major < minor` — torus
///     self-intersects), or
///   - `radius` is non-positive, or the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_sphere_fillet(
    n_p_inward: Vec3,
    d_plane: f64,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face_plane: FaceId,
    face_sphere: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    // 1) Convex (face not reversed) vs concave (face reversed) drive a
    //    `signed_offset = ±1` factor that flips the rolling-ball axial
    //    side and the sphere tangency type (external `R+r` for convex,
    //    internal `R−r` for concave).
    if radius <= tol_lin {
        return Ok(None);
    }
    let concave = topo.face(face_sphere)?.is_reversed();
    let signed_offset: f64 = if concave { -1.0 } else { 1.0 };

    // The pcurve_sphere construction below assumes the contact circle is
    // a constant-v latitude on the sphere — which only holds when the
    // sphere's parametric axis is (anti)parallel to the plate normal. If
    // the sphere has an oblique frame, fall back to the walker.
    if sphere.z_axis().dot(n_p_inward).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    let big_r = sphere.radius();
    let center = sphere.center();
    let center_v = Vec3::new(center.x(), center.y(), center.z());

    // 2) Project sphere center onto the plate to get the spine-circle
    //    center. By construction `p_axis_on_plane − center` is along
    //    `n_p_inward`, so the spine is automatically axisymmetric about
    //    the plate normal — the only valid configuration for the analytic
    //    formula.
    let step = d_plane - n_p_inward.dot(center_v);
    let p_axis_on_plane = center + n_p_inward * step;

    // 3) Signed distance from plate to sphere center along n_p_inward:
    //    `h_signed = (sphere_center − p_axis_on_plane) · n_p_inward`.
    //    Negative means sphere center is on the side OPPOSITE the plate
    //    material (the typical "sphere post on plate slab"); positive
    //    means same side as plate material (sphere buried with cap
    //    emerging). The R_t formula and contact_sphere placement use
    //    `h_signed` directly — both convex configurations share a
    //    unified expression `R_t² = r_p² + 2r(R − h_signed)` once the
    //    sign is preserved.
    let h_signed = -step;
    let h_abs = h_signed.abs();

    // 4) Sphere must intersect the plate to give a spine. `|h_signed| < R`
    //    ⇒ spine exists.
    if h_abs >= big_r - tol_lin {
        return Ok(None);
    }

    let r_p_sq = big_r * big_r - h_abs * h_abs;
    if r_p_sq <= tol_lin * tol_lin {
        return Ok(None);
    }

    // 5) Major radius via rolling-ball constraint, unified across convex
    //    (external tangency `R + r`) and concave (internal tangency
    //    `R − r`). Ball axial offset is `−signed_offset · r` along
    //    n_p_inward. Solving
    //      R_t² + (signed_offset·r + h_signed)² = (R + signed_offset·r)²
    //    expands to
    //      R_t² = r_p² + signed_offset · 2r·(R − h_signed).
    //    For convex this is `r_p² + 2r(R − h_signed)` (always ≥ r_p²
    //    since h_signed ≤ R); for concave it's `r_p² − 2r(R − h_signed)`,
    //    which can shrink below r_p² and even below r² (spindle).
    let major_radius_sq = r_p_sq + signed_offset * 2.0 * radius * (big_r - h_signed);
    if major_radius_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let major_radius = major_radius_sq.sqrt();
    let minor_radius = radius;
    // Spindle check: a torus with major < minor self-intersects and is
    // invalid as a fillet surface. Tightest in the concave case (and
    // also in the convex buried-sphere sub-case where h_signed near R).
    if major_radius < minor_radius - tol_lin {
        return Ok(None);
    }

    // 6) Spine span — same closed-circle handling as plane-cylinder.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 7) Construct the torus. Axis: parallel to n_p_inward (always — the
    //    torus is symmetric about the line from sphere center perpendicular
    //    to the plate, which IS the n_p_inward axis). Reference direction:
    //    inherit the sphere's u=0 frame so contact-circle parameterization
    //    aligns with the sphere's u-coord. For brepkit's SphericalSurface
    //    `Frame3::from_normal(axis)` produces (x_axis, y_axis) consistent
    //    with the spine's u parameter when projected.
    let torus_axis = n_p_inward;
    let sphere_x = sphere.x_axis();
    let sphere_y = sphere.y_axis();
    // `with_axis_and_ref_dir` requires the ref dir non-parallel to the
    // axis. sphere_x ⊥ sphere_z and torus_axis ∥ ±sphere_z (alignment
    // guard above), so they're perpendicular — but guard with sphere_y
    // as a backup against floating-point drift.
    let ref_dir = if sphere_x.cross(torus_axis).length() > tol_ang {
        sphere_x
    } else {
        sphere_y
    };

    // Torus center on the rolling-ball side: convex puts it on the
    // −n_p_inward side (empty wedge above plate), concave on the
    // +n_p_inward side (inside the cavity). `−signed_offset` unifies.
    let torus_center = p_axis_on_plane - n_p_inward * (signed_offset * radius);
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        torus_axis,
        ref_dir,
    )?;

    // 8) Spine endpoints in 3D and corresponding u-parameters. We
    //    parameterize the contact circles with the sphere's frame so the
    //    pcurve on the sphere is a horizontal Line2D in (u, v).
    let u_at = |p: Point3| {
        let v = p - p_axis_on_plane;
        sphere_y.dot(v).atan2(sphere_x.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // 9) Contact circles.
    //
    //    Plate-side: circle of radius `major_radius` at z = plate (where
    //    the torus tube touches the plate face).
    //
    //    Sphere-side contact, unified across convex/concave:
    //    `contact = sphere_center + R · (ball − sphere_center) / |ball − sc|`
    //    with `|ball − sc| = R + signed_offset·r`. Decomposed:
    //      contact_radial = R_t · R / (R + signed_offset·r),
    //      contact_axial_along_n_p_inward
    //          = signed_offset · r · (h_signed − R) / (R + signed_offset·r).
    //    For convex this is the negative of `r·(R − h_signed)/(R+r)` (i.e.
    //    a positive axial when h_signed < 0, meaning contact lies on the
    //    upper hemisphere); for concave the sign flips so the contact
    //    lies on the lower hemisphere where the actual sphere face lives.
    let denom = big_r + signed_offset * radius;
    let contact_sphere_radial = major_radius * big_r / denom;
    let contact_sphere_axial = signed_offset * radius * (h_signed - big_r) / denom;
    let contact_sphere_center = p_axis_on_plane + n_p_inward * contact_sphere_axial;

    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        torus_axis,
        major_radius,
        sphere_x,
        sphere_y,
    )?;
    let contact_sphere_circle = brepkit_math::curves::Circle3D::with_axes(
        contact_sphere_center,
        torus_axis,
        contact_sphere_radial,
        sphere_x,
        sphere_y,
    )?;
    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_sphere = circle_arc_to_nurbs(&contact_sphere_circle, u_start, u_end)?;

    // 10) PCurves.
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            major_radius,
        )?)
    };
    // Sphere pcurve at constant v (latitude on sphere). Use
    // ParametricSurface::project_point to get the correct (u, v) for one
    // point on the contact circle, then sweep u. For brepkit's
    // SphericalSurface the v-parameter is co-latitude from the +axis.
    let sample_p = contact_sphere_circle.evaluate(u_start);
    let v_sphere = ParametricSurface::project_point(sphere, sample_p).1;
    let pcurve_sphere = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_sphere),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 11) Cross-sections at spine endpoints.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_sphere_at = |u: f64| contact_sphere_circle.evaluate(u);
    let center_at =
        |u: f64| contact_plane_circle.evaluate(u) - n_p_inward * (signed_offset * radius);
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_at = |u: f64, t: f64| CircSection {
        p1: p_plane_at(u),
        p2: p_sphere_at(u),
        center: center_at(u),
        radius,
        uv1: plane_uv_at(u),
        uv2: (u, v_sphere),
        t,
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_sphere,
        contact1: contact_plane,
        contact2: contact_sphere,
        face1: face_plane,
        face2: face_sphere,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between a plane and a sphere whose center sits along the plate
/// normal. Handles convex (sphere-on-plate post) and concave (spherical
/// pocket / hole through plate) via a unified `signed_offset = ±1` factor.
///
/// `d1` is the chamfer distance on the plate (radially outward into plate
/// material from the spine); `d2` is the geodesic distance on the sphere
/// surface (arc length along the meridian from the spine, going INTO the
/// sphere FACE — toward the apex on the upper cap for convex, toward the
/// south pole on the lower cap for concave).
///
/// # Geometry
///
/// In a local frame where `n_p_inward` is +z and `p_axis_on_plane` is
/// the origin (spine center), with sphere center at `(0, 0, h_signed)`
/// and `δ = d2 / R`. With `signed_offset = +1` (face NOT reversed,
/// convex) or `−1` (face reversed, concave):
///
///   - plate contact:  `(r_p + d1, 0, 0)`,
///   - sphere contact: `(r_p cos δ + signed_offset · h_signed sin δ, 0,
///                       h_signed(1 − cos δ) + signed_offset · r_p sin δ)`.
///
/// Convex flips the meridian arm (going toward apex), concave goes toward
/// the south pole. The chamfer surface is the cone generated by rotating
/// the line through these contacts around the plate normal:
///
///   Δr = sphere_radial − (r_p + d1)
///   Δz = sphere_axial
///   z_apex = −(r_p + d1) · Δz / Δr
///   cone half-angle β = atan(|z_apex| / (r_p + d1))
///   chamfer_axis     = −sign(z_apex) · n_p_inward  (apex above ⇒ axis −,
///                                                   apex below ⇒ axis +)
///
/// For symmetric `d1 = d2` and small `δ`, `tan β ≈ r_p / (R − h_signed)`
/// in the convex case (and a similar expression with `signed_offset` for
/// concave).
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere doesn't intersect the plate (`|h_signed| ≥ R`),
///   - sphere axis isn't aligned with plate normal (oblique frame),
///   - `d1` or `d2` non-positive,
///   - `Δr ≥ 0` — sphere contact lands at or beyond the plate
///     contact's radius (e.g. asymmetric `d2 ≫ d1` in concave, where
///     sphere_radial bulges past `r_p + d1`); the resulting outward-
///     flaring cone is the wrong geometric configuration, or
///   - `Δz ≈ 0` (degenerate flat-disk chamfer), or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn plane_sphere_chamfer(
    n_p_inward: Vec3,
    d_plane: f64,
    sphere: &brepkit_math::surfaces::SphericalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face_plane: FaceId,
    face_sphere: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    // 1) Convex (face not reversed) vs concave (face reversed) drive the
    //    `signed_offset = ±1` factor. Convex sends the sphere-side
    //    contact toward the apex; concave sends it toward the south
    //    pole, which flips the sign of the `r_p sin δ` and
    //    `h_signed sin δ` terms in the contact formulas.
    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let concave = topo.face(face_sphere)?.is_reversed();
    let signed_offset: f64 = if concave { -1.0 } else { 1.0 };

    // The pcurve_sphere construction below assumes the contact circle is
    // a constant-v latitude on the sphere — which only holds when the
    // sphere's parametric axis is (anti)parallel to the plate normal. If
    // the sphere has an oblique frame, fall back to the walker.
    if sphere.z_axis().dot(n_p_inward).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    let big_r = sphere.radius();
    let center = sphere.center();
    let center_v = Vec3::new(center.x(), center.y(), center.z());

    // 2) Project sphere center onto the plate to get spine-circle center.
    let step = d_plane - n_p_inward.dot(center_v);
    let p_axis_on_plane = center + n_p_inward * step;
    let h_signed = -step;
    let h_abs = h_signed.abs();
    if h_abs >= big_r - tol_lin {
        return Ok(None);
    }
    let r_p_sq = big_r * big_r - h_abs * h_abs;
    if r_p_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let r_p = r_p_sq.sqrt();

    // 3) Sphere-side contact along the meridian "into sphere face"
    //    direction. Convex (face NOT reversed) goes -ψ from the spine
    //    (toward apex on the upper cap); concave (face reversed) goes
    //    +ψ (toward south pole on the lower cap). With `signed_offset
    //    = +1 (convex) / −1 (concave)`, both reduce to the same
    //    formulas after the meridian-arm flip:
    //      sphere_radial = r_p cos δ + signed_offset · h_signed · sin δ
    //      sphere_axial  = h_signed (1 − cos δ) + signed_offset · r_p · sin δ
    //    where sphere_axial is the offset along +n_p_inward from
    //    p_axis_on_plane. For convex post-on-slab (h_signed < 0,
    //    signed_offset = +1) sphere_axial > 0 (contact above plate);
    //    for concave pocket (h_signed > 0, signed_offset = −1)
    //    sphere_axial < 0 (contact below plate, on the lower cap where
    //    the face actually exists).
    let delta = d2 / big_r;
    let (sin_d, cos_d) = delta.sin_cos();
    let sphere_radial = r_p * cos_d + signed_offset * h_signed * sin_d;
    let sphere_axial = h_signed * (1.0 - cos_d) + signed_offset * r_p * sin_d;
    // For very large `d2` the meridian can sweep past the pole and
    // sphere_radial collapses to ≤ 0 — at that point Circle3D would
    // reject the construction. Bail to the walker first.
    if sphere_radial <= tol_lin {
        return Ok(None);
    }

    // 4) Chamfer line between plate contact (r_p+d1, 0) and sphere contact.
    //    Require `Δr < 0` (sphere contact closer to axis than plate
    //    contact) — the natural chamfer geometry for both convex post
    //    and concave pocket. `Δr > 0` would flip the apex side and
    //    produce an outward-flaring cone, which is a different
    //    geometric configuration (not what the user requested). This
    //    can arise in concave pockets with asymmetric `d2 >> d1` where
    //    sphere_radial bulges past `r_p + d1`; bail to walker.
    //    `Δz ≈ 0` means a flat disk, also degenerate.
    let delta_r = sphere_radial - (r_p + d1);
    let delta_z = sphere_axial;
    if delta_r >= -tol_lin {
        return Ok(None);
    }
    if delta_z.abs() <= tol_lin {
        return Ok(None);
    }

    // 5) Cone apex on the n_p_inward axis, half-angle β from the radial
    //    plane. At apex r = 0, so the line through (r_p+d1, 0) and
    //    (sphere_radial, sphere_axial) hits axial position
    //      z_apex = −(r_p + d1) · Δz / Δr.
    //    The sign of z_apex tells us which side of the plate the apex
    //    sits on (convex post → apex above; concave pocket → apex below
    //    in our local frame). The cone axis points AWAY from the apex
    //    toward the contacts, i.e. `−sign(z_apex) · n_p_inward`.
    //    `tan β = |z_apex| / (r_p + d1)`.
    let z_apex = -(r_p + d1) * delta_z / delta_r;
    let cone_half_angle = (z_apex.abs() / (r_p + d1)).atan();
    if cone_half_angle <= 1e-3 || cone_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }
    let chamfer_apex_pos = p_axis_on_plane + n_p_inward * z_apex;
    let chamfer_axis = if z_apex > 0.0 {
        -n_p_inward
    } else {
        n_p_inward
    };

    // 6) Spine span (closed-circle aware).
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // 7) Build the chamfer cone using the sphere's u=0 frame as ref dir.
    //    `with_ref_dir` requires the ref dir to be NON-parallel to the
    //    cone axis; sphere_x ⊥ sphere_z and chamfer_axis ∥ ±sphere_z (by
    //    the alignment guard above), so sphere_x ⊥ chamfer_axis — but
    //    guard with sphere_y as a backup against floating-point drift.
    let sphere_x = sphere.x_axis();
    let sphere_y = sphere.y_axis();
    let ref_dir = if sphere_x.cross(chamfer_axis).length() > tol_ang {
        sphere_x
    } else {
        sphere_y
    };
    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, chamfer_axis, cone_half_angle, ref_dir)?;

    // 8) Spine endpoints in 3D and corresponding u-parameters around the
    //    n_p_inward axis.
    let u_at = |p: Point3| {
        let v = p - p_axis_on_plane;
        sphere_y.dot(v).atan2(sphere_x.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // 9) 3D contact circles around the n_p_inward axis.
    let plate_axis = n_p_inward;
    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        plate_axis,
        r_p + d1,
        sphere_x,
        sphere_y,
    )?;
    let contact_sphere_center = p_axis_on_plane + n_p_inward * sphere_axial;
    let contact_sphere_circle = brepkit_math::curves::Circle3D::with_axes(
        contact_sphere_center,
        plate_axis,
        sphere_radial,
        sphere_x,
        sphere_y,
    )?;
    let contact_plane = circle_arc_to_nurbs(&contact_plane_circle, u_start, u_end)?;
    let contact_sphere = circle_arc_to_nurbs(&contact_sphere_circle, u_start, u_end)?;

    // 10) PCurves.
    let plane_adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
    let pcurve_plane = {
        let (cu, cv) = plane_adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            r_p + d1,
        )?)
    };
    let sample_p = contact_sphere_circle.evaluate(u_start);
    let v_sphere = ParametricSurface::project_point(sphere, sample_p).1;
    let pcurve_sphere = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_sphere),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 11) Cross-sections at spine endpoints.
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_sphere_at = |u: f64| contact_sphere_circle.evaluate(u);
    let plane_uv_at = |u: f64| plane_adapter.project_point(p_plane_at(u));
    let section_at = |u: f64, t: f64| {
        let p1 = p_plane_at(u);
        let p2 = p_sphere_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: plane_uv_at(u),
            uv2: (u, v_sphere),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1: pcurve_plane,
        pcurve2: pcurve_sphere,
        contact1: contact_plane,
        contact2: contact_sphere,
        face1: face_plane,
        face2: face_sphere,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between two intersecting spheres — the rolling-ball blend is
/// an exact torus around the line connecting the sphere centers.
///
/// Handles all four convex/concave combinations via per-sphere
/// `signed_offset_i = ±1`:
///   - face NOT reversed (`+1`): rolling ball externally tangent
///     (`|ball − Ci| = Ri + r`)
///   - face REVERSED (`−1`): rolling ball internally tangent
///     (`|ball − Ci| = Ri − r`)
///
/// # Geometry
///
/// Place the C1→C2 line as the symmetry axis (length `D = |C2 − C1|`).
/// With effective radii `Q1 = R1 + s1·r`, `Q2 = R2 + s2·r`:
///   `a_ball = (Q1² − Q2² + D²) / (2D)`
///   `R_t² = Q1² − a_ball²`
///   torus axis     = (C2 − C1) / D
///   torus center   = C1 + axis · a_ball
///   minor radius   = r
///
/// The spine circle itself depends ONLY on the original sphere radii
/// and center distance — `a₀ = (R1² − R2² + D²)/(2D)` with radius
/// `r_p = √(R1² − a₀²)` — and is independent of `r` and the convexity
/// flags. The `Q`-substitution flips the rolling-ball trajectory
/// (`a_ball`, `R_t`) and per-sphere contact circles, but not the spine.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - the spheres don't intersect properly (`D ≤ |R1−R2|` or
///     `D ≥ R1+R2`),
///   - a concave-side effective radius collapses (`Qi ≤ tol`, e.g.
///     fillet radius ≥ a concave sphere's radius),
///   - the resulting major < minor (spindle), or
///   - the spine is degenerate, or sphere axes don't align with
///     C1→C2 (oblique frames).
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_sphere_fillet(
    s1: &brepkit_math::surfaces::SphericalSurface,
    s2: &brepkit_math::surfaces::SphericalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;

    if radius <= tol_lin {
        return Ok(None);
    }
    let s1_signed: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2_signed: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r1 = s1.radius();
    let big_r2 = s2.radius();
    let c1 = s1.center();
    let c2 = s2.center();
    let c1_to_c2 = c2 - c1;
    let big_d = c1_to_c2.length();
    if big_d <= tol_lin {
        return Ok(None);
    }
    // Spheres must form a real intersection circle: |R1−R2| < D < R1+R2.
    if big_d <= (big_r1 - big_r2).abs() + tol_lin || big_d >= big_r1 + big_r2 - tol_lin {
        return Ok(None);
    }

    let axis = (c1_to_c2 * (1.0 / big_d)).normalize()?;

    // Spine geometry along the C1→C2 axis.
    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    if r_p_sq <= tol_lin * tol_lin {
        return Ok(None);
    }

    // Effective radii pick up the per-sphere tangency direction. For
    // concave (face reversed, ball internally tangent) Q_i shrinks
    // below R_i; if the fillet radius approaches R_i, Q_i collapses to
    // 0 (rolling ball would coincide with sphere center) — bail.
    let q1 = big_r1 + s1_signed * radius;
    let q2 = big_r2 + s2_signed * radius;
    if q1 <= tol_lin || q2 <= tol_lin {
        return Ok(None);
    }

    // Rolling-ball axial position and major radius. With Q-substitution
    // the formulas mirror the convex case exactly.
    let a_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let major_radius_sq = q1 * q1 - a_ball * a_ball;
    if major_radius_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let major_radius = major_radius_sq.sqrt();
    let minor_radius = radius;
    if major_radius < minor_radius - tol_lin {
        return Ok(None);
    }

    // Spine span (closed-circle aware).
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // Axisymmetry guards: each sphere's parametric z-axis must align
    // with the C1→C2 axis so the contact circles are constant-v
    // latitudes on the respective spheres (otherwise the pcurves we
    // build below as constant-v Line2Ds are wrong).
    let tol_ang = ANALYTIC_TOL_ANG;
    if s1.z_axis().dot(axis).abs() < 1.0 - tol_ang || s2.z_axis().dot(axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // Pick a reference direction perpendicular to the axis. Inherit
    // sphere1's frame (well-defined when its z-axis is aligned with
    // `axis`); fall back to sphere1.y_axis if x_axis happens to coincide
    // with axis under floating-point drift.
    let s1_x = s1.x_axis();
    let s1_y = s1.y_axis();
    let ref_dir = if s1_x.cross(axis).length() > tol_ang {
        s1_x
    } else {
        s1_y
    };

    let torus_center = c1 + axis * a_ball;
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        axis,
        ref_dir,
    )?;

    // Spine plane center (where the spine circle lies).
    let spine_plane_center = c1 + axis * a0;

    // u-parameter for a point on a contact circle around the axis.
    // Use the in-axis-perpendicular component of (point − spine_center)
    // projected onto (ref_dir, axis × ref_dir) to recover u.
    let perp_y = axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // 3D contact circles. Each is a small circle on its sphere in a
    // plane perpendicular to the axis. The contact = sphere_center +
    // R_i · (ball − sphere_center) / |ball − sphere_center|, with
    // |ball − Ci| = Qi (the effective tangency distance).
    //   axial component (from Ci toward ball-center direction)
    //     = R_i · (ball_axial_from_Ci) / Qi
    //   radial component
    //     = R_i · R_t / Qi
    let s1_contact_axial = big_r1 * a_ball / q1;
    let s1_contact_radial = big_r1 * major_radius / q1;
    let s1_contact_center = c1 + axis * s1_contact_axial;
    let contact1_circle = brepkit_math::curves::Circle3D::with_axes(
        s1_contact_center,
        axis,
        s1_contact_radial,
        ref_dir,
        perp_y,
    )?;

    let s2_contact_axial_from_c2 = big_r2 * (a_ball - big_d) / q2;
    let s2_contact_radial = big_r2 * major_radius / q2;
    let s2_contact_center = c2 + axis * s2_contact_axial_from_c2;
    let contact2_circle = brepkit_math::curves::Circle3D::with_axes(
        s2_contact_center,
        axis,
        s2_contact_radial,
        ref_dir,
        perp_y,
    )?;

    let contact1 = circle_arc_to_nurbs(&contact1_circle, u_start, u_end)?;
    let contact2 = circle_arc_to_nurbs(&contact2_circle, u_start, u_end)?;

    // PCurves on each sphere — constant-v latitude lines at the
    // contact's v-parameter (constant by axisymmetry guard above).
    let sample1 = contact1_circle.evaluate(u_start);
    let v1 = ParametricSurface::project_point(s1, sample1).1;
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v1),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample2 = contact2_circle.evaluate(u_start);
    let v2 = ParametricSurface::project_point(s2, sample2).1;
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v2),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections at spine endpoints.
    let p1_at = |u: f64| contact1_circle.evaluate(u);
    let p2_at = |u: f64| contact2_circle.evaluate(u);
    let center_at = |u: f64| {
        let on_torus_eq = spine_plane_center
            + ref_dir * (major_radius * u.cos())
            + perp_y * (major_radius * u.sin());
        on_torus_eq + axis * (a_ball - a0)
    };
    let section_at = |u: f64, t: f64| CircSection {
        p1: p1_at(u),
        p2: p2_at(u),
        center: center_at(u),
        radius,
        uv1: (u, v1),
        uv2: (u, v2),
        t,
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between a sphere and a cylinder whose axis passes through the
/// sphere center — the rolling-ball blend is an exact torus around the
/// cylinder axis.
///
/// Spine exists only when the cylinder axis-line passes through the
/// sphere center: the sphere–cylinder intersection is then a pair of
/// circles at axial offsets `±h_s = ±√(R_s² − r_c²)` from the sphere
/// center along the cylinder axis (each of radius `r_c`). The user
/// passes ONE of these as the spine.
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1` (face NOT reversed = +1 = external tangency
/// `Q_i = R_i + r`; face REVERSED = −1 = internal tangency `Q_i = R_i − r`).
///
/// # Geometry
///
/// Place sphere center at the origin, cylinder axis = +z. Define
///   `Q_s = R_s + s_s · r`,
///   `Q_c = r_c + s_c · r`.
/// Tangency constraints `|ball − C_s| = Q_s`, `|ball − cyl_axis| = Q_c`
/// give:
///   torus axis    = cyl axis (same direction as the spine's signed
///                   axial offset from sphere center)
///   torus center  = sphere_center + axis · a_ball
///   major         = R_t = Q_c
///   a_ball        = sign(spine_axial) · √(Q_s² − Q_c²)
///   minor         = r
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere center isn't on the cylinder axis,
///   - sphere doesn't enclose cylinder (`r_c ≥ R_s`),
///   - effective radii collapse (e.g. `Q_c ≤ 0` for very large `r` in
///     concave-cylinder),
///   - resulting major < minor (spindle: `Q_c < r` ⇒ `r > r_c/2` for
///     concave cylinder), or
///   - `Q_s ≤ Q_c` (the rolling ball can't reach axially), or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_cylinder_fillet(
    sph: &brepkit_math::surfaces::SphericalSurface,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face_sphere: FaceId,
    face_cyl: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if radius <= tol_lin {
        return Ok(None);
    }
    let s_sphere: f64 = if topo.face(face_sphere)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s_cyl: f64 = if topo.face(face_cyl)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r_s = sph.radius();
    let r_c = cyl.radius();
    let c_s = sph.center();
    let cyl_origin = cyl.origin();
    let cyl_axis = cyl.axis();

    // Sphere center must lie on the cylinder axis line.
    let to_sphere = c_s - cyl_origin;
    let to_sphere_v = Vec3::new(to_sphere.x(), to_sphere.y(), to_sphere.z());
    let along = to_sphere_v.dot(cyl_axis);
    let perp = to_sphere_v - cyl_axis * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }

    // Sphere's parametric z_axis must be (anti)parallel to the cylinder
    // axis so the contact circle on the sphere is a constant-v latitude.
    if sph.z_axis().dot(cyl_axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // Sphere must enclose cylinder: r_c < R_s for spine to exist.
    if r_c >= big_r_s - tol_lin {
        return Ok(None);
    }
    let h_s_sq = big_r_s * big_r_s - r_c * r_c;
    if h_s_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let h_s = h_s_sq.sqrt();

    // Determine which spine the user passed (z = +h_s or z = −h_s
    // along cyl_axis from sphere center). Project a spine sample.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - c_s;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(cyl_axis);
    let sample_radial_v = to_sample_v - cyl_axis * sample_axial;
    let sample_radial = sample_radial_v.length();
    // Spine must lie on one of the two intersection circles: at axial
    // ±h_s and radial r_c from the cylinder axis. Otherwise it's an
    // oblique slice the helper can't handle.
    if (sample_axial.abs() - h_s).abs() > tol_lin || (sample_radial - r_c).abs() > tol_lin {
        return Ok(None);
    }
    let spine_sign = if sample_axial >= 0.0 { 1.0 } else { -1.0 };

    // Effective radii.
    let q_s = big_r_s + s_sphere * radius;
    let q_c = r_c + s_cyl * radius;
    if q_s <= tol_lin || q_c <= tol_lin {
        return Ok(None);
    }

    // Rolling-ball position. Q_s² − Q_c² must be ≥ 0 (else ball can't
    // reach the spine axially).
    let a_ball_sq = q_s * q_s - q_c * q_c;
    if a_ball_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let a_ball = spine_sign * a_ball_sq.sqrt();

    let major_radius = q_c;
    let minor_radius = radius;
    if major_radius < minor_radius - tol_lin {
        return Ok(None);
    }

    // Build the torus.
    let cyl_x = cyl.x_axis();
    let cyl_y = cyl.y_axis();
    let ref_dir = if cyl_x.cross(cyl_axis).length() > tol_ang {
        cyl_x
    } else {
        cyl_y
    };
    let torus_center = c_s + cyl_axis * a_ball;
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        cyl_axis,
        ref_dir,
    )?;

    // Spine plane center on cylinder axis at axial = sample_axial.
    let spine_plane_center = c_s + cyl_axis * sample_axial;
    let perp_y = cyl_axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Sphere contact: sphere_center + R_s · (ball − sphere_center) / Q_s.
    let sph_contact_axial = big_r_s * a_ball / q_s;
    let sph_contact_radial = big_r_s * major_radius / q_s;
    let sph_contact_center = c_s + cyl_axis * sph_contact_axial;
    let contact_sph_circle = brepkit_math::curves::Circle3D::with_axes(
        sph_contact_center,
        cyl_axis,
        sph_contact_radial,
        ref_dir,
        perp_y,
    )?;

    // Cylinder contact: at axial = a_ball (along cyl axis from sphere
    // center; convert to cyl-origin frame), radial = r_c.
    let cyl_contact_axial_world = c_s + cyl_axis * a_ball; // same as torus_center
    let contact_cyl_circle = brepkit_math::curves::Circle3D::with_axes(
        cyl_contact_axial_world,
        cyl_axis,
        r_c,
        ref_dir,
        perp_y,
    )?;

    let contact_sph = circle_arc_to_nurbs(&contact_sph_circle, u_start, u_end)?;
    let contact_cyl = circle_arc_to_nurbs(&contact_cyl_circle, u_start, u_end)?;

    // PCurves on each surface. The sphere's u parameter is measured in
    // its OWN frame (sph.x_axis / sph.y_axis) — which can differ from
    // the cylinder frame's `ref_dir` even when their z-axes align.
    // Project the start sample to recover the sphere's u, then sweep
    // by the same angular delta `u_end − u_start` (the angular change
    // is frame-independent on a circle around the shared axis).
    let sample_sph = contact_sph_circle.evaluate(u_start);
    let (u_sph_start, v_sph) = ParametricSurface::project_point(sph, sample_sph);
    let pcurve_sph = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_sph_start, v_sph),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    // Cylinder pcurve. Same frame-independence reasoning: derive the
    // cylinder's own u for the start sample.
    let sample_cyl = contact_cyl_circle.evaluate(u_start);
    let u_cyl_start = ParametricSurface::project_point(cyl, sample_cyl).0;
    let v_cyl = cyl_v_at_point(cyl, sample_cyl);
    let pcurve_cyl = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_cyl_start, v_cyl),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections. Section uv1/uv2 must use each surface's own u
    // parameter (matching the pcurves above), not the cylinder-frame
    // `u` we used for the contact-circle parameterization.
    let p_sph_at = |u: f64| contact_sph_circle.evaluate(u);
    let p_cyl_at = |u: f64| contact_cyl_circle.evaluate(u);
    let section_at = |u: f64, t: f64| CircSection {
        p1: p_sph_at(u),
        p2: p_cyl_at(u),
        center: torus_center
            + ref_dir * (major_radius * u.cos())
            + perp_y * (major_radius * u.sin()),
        radius,
        uv1: (u_sph_start + (u - u_start), v_sph),
        uv2: (u_cyl_start + (u - u_start), v_cyl),
        t,
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1: pcurve_sph,
        pcurve2: pcurve_cyl,
        contact1: contact_sph,
        contact2: contact_cyl,
        face1: face_sphere,
        face2: face_cyl,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between a sphere and a cone whose axis passes through the
/// sphere center — the rolling-ball blend is an exact torus around the
/// cone axis.
///
/// Spine exists when the cone surface intersects the sphere; the
/// intersection is a pair of circles (where they exist), and the user
/// passes ONE of them as the spine.
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1` (face NOT reversed = +1, face REVERSED = −1).
/// `s_sph` flips sphere tangency type (external `R+r` ↔ internal
/// `R−r`); `s_cone` flips cone tangency direction (ball outside cone
/// ↔ inside).
///
/// # Geometry
///
/// Place sphere center at origin, cone axis = +z, cone apex at
/// `(0, 0, a_apex)`, half-angle β (radial plane to generator,
/// brepkit convention). With `h = 0 − a_apex`, `Q_s = R_s + s_sph · r`,
/// `A = s_cone · r + h · cos β`. Tangency constraints
///   R_t · sin β − (z_b + h) · cos β = s_cone · r       (cone)
///   R_t² + z_b² = Q_s²                                  (sphere)
/// yield a quadratic in `c = z_b`:
///
///   `(c + A · cos β)² = sin²β · (Q_s² − A²)`
///
/// Both roots correspond to the two spine candidates; the user-supplied
/// spine selects the closer one. Once `c = z_b` is known,
/// `R_t = (s_cone · r + (c + h)·cos β) / sin β`.
///
/// At `β → π/2` the formulas collapse to plane-sphere; at `β → 0`
/// (degenerate cone = cylinder) they collapse to sphere-cylinder.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere center isn't on the cone axis line,
///   - sphere parametric z-axis isn't aligned with cone axis,
///   - `Q_s ≤ tol` (concave-sphere with `r ≥ R_s` — degenerate),
///   - `Q_s² < A²` (no valid rolling-ball position),
///   - the spine isn't at the predicted axial position (within tol),
///   - the resulting major < minor (spindle), or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_cone_fillet(
    sph: &brepkit_math::surfaces::SphericalSurface,
    cone: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face_sphere: FaceId,
    face_cone: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if radius <= tol_lin {
        return Ok(None);
    }
    let s_sph: f64 = if topo.face(face_sphere)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s_cone: f64 = if topo.face(face_cone)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r_s = sph.radius();
    let c_s = sph.center();
    let cone_apex = cone.apex();
    let cone_axis = cone.axis();
    let beta = cone.half_angle();

    // Sphere center must lie on cone axis line.
    let to_sphere = c_s - cone_apex;
    let to_sphere_v = Vec3::new(to_sphere.x(), to_sphere.y(), to_sphere.z());
    let along = to_sphere_v.dot(cone_axis);
    let perp = to_sphere_v - cone_axis * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }

    // Sphere z-axis aligned with cone axis.
    if sph.z_axis().dot(cone_axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    let (sin_b, cos_b) = beta.sin_cos();
    if sin_b <= tol_lin || cos_b <= tol_lin {
        return Ok(None);
    }

    // h = sphere_center_axial − apex_axial along cone_axis. Positive
    // when sphere center is on the +cone_axis side of the apex.
    let h_signed = along; // along = cone_axis · (c_s − apex)

    // Quadratic for c = z_b (ball axial position relative to sphere center
    // along cone_axis), generalized for all 4 convex/concave combinations
    // via per-face signed_offset:
    //   (c + A·cos β)² = sin²β · (Q_s² − A²)
    // where:
    //   Q_s = R_s + s_sph · r       (effective sphere tangency radius)
    //   A   = s_cone · r + h_signed · cos β   (effective cone offset)
    // For convex-convex (s_sph = s_cone = +1) this matches the original
    // formula. For concave-sphere s_sph = −1 ⇒ Q_s = R_s − r (internal
    // tangency to sphere). For concave-cone s_cone = −1 ⇒ A flips sign
    // on the radius term (ball inside cone region instead of outside).
    let q_s = big_r_s + s_sph * radius;
    if q_s <= tol_lin {
        return Ok(None);
    }
    let big_a = s_cone * radius + h_signed * cos_b;
    let disc = q_s * q_s - big_a * big_a;
    if disc <= tol_lin * tol_lin {
        return Ok(None);
    }
    let disc_sqrt = disc.sqrt();
    let c_root_a = -big_a * cos_b + sin_b * disc_sqrt;
    let c_root_b = -big_a * cos_b - sin_b * disc_sqrt;

    // Spine validation + root selection.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - c_s;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(cone_axis);
    let sample_radial_v = to_sample_v - cone_axis * sample_axial;
    let sample_radial = sample_radial_v.length();
    // Find the spine candidate axial — the spine axial is determined
    // by sphere ∩ cone, NOT by `c` directly. Solve sphere ∩ cone:
    //   r_spine² + z_spine² = R_s²,
    //   r_spine = (z_spine + h_signed)·cot β.
    // ⇒ z_spine²·(1 + cot²β) + 2 z_spine·h_signed·cot²β + h_signed²·cot²β = R_s²
    // ⇒ z_spine²/sin²β + 2 z_spine·h_signed·cot²β + (h_signed²·cot²β − R_s²) = 0.
    let cot_b = cos_b / sin_b;
    let qa = 1.0 / (sin_b * sin_b);
    let qb = 2.0 * h_signed * cot_b * cot_b;
    let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb * qb - 4.0 * qa * qc;
    if q_disc <= tol_lin * tol_lin {
        return Ok(None);
    }
    let q_disc_sqrt = q_disc.sqrt();
    let z_spine_root_a = (-qb + q_disc_sqrt) / (2.0 * qa);
    let z_spine_root_b = (-qb - q_disc_sqrt) / (2.0 * qa);
    // Pick the spine root that matches the sample (axial within tol).
    let spine_match_tol = tol_lin * 1e3;
    let spine_z = if (sample_axial - z_spine_root_a).abs() < spine_match_tol {
        z_spine_root_a
    } else if (sample_axial - z_spine_root_b).abs() < spine_match_tol {
        z_spine_root_b
    } else {
        return Ok(None);
    };
    // Verify radial.
    let r_spine = (spine_z + h_signed) * cot_b;
    if r_spine <= tol_lin || (sample_radial - r_spine).abs() > spine_match_tol {
        return Ok(None);
    }

    // Pick rolling-ball root. Each `c` root corresponds to a torus
    // around ONE of the two spines; the rolling-ball axial position is
    // close to (but not equal to) the spine axial — the small offset
    // is the ball's perpendicular shift away from the spine into the
    // empty wedge. Pick the root whose distance to `spine_z` is
    // smallest. This is robust even for small half-angles where both
    // c roots could share a sign.
    let z_b = if (c_root_a - spine_z).abs() <= (c_root_b - spine_z).abs() {
        c_root_a
    } else {
        c_root_b
    };
    // R_t from cone tangency: s_cone · r + (z_b + h_signed) · cos β = R_t · sin β.
    let r_t = (s_cone * radius + (z_b + h_signed) * cos_b) / sin_b;
    if r_t <= tol_lin {
        return Ok(None);
    }

    let major_radius = r_t;
    let minor_radius = radius;
    if major_radius < minor_radius - tol_lin {
        return Ok(None);
    }

    // Build the torus.
    let cone_x = cone.x_axis();
    let ref_dir = cone_x;

    let torus_center = c_s + cone_axis * z_b;
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        cone_axis,
        ref_dir,
    )?;

    // Spine plane center.
    let spine_plane_center = c_s + cone_axis * spine_z;
    let perp_y = cone_axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Sphere contact: sphere_center + R_s · (ball − sphere_center) / |ball − sphere_center|.
    // |ball − sphere_center| = Q_s = R_s + s_sph · r, so
    //   sphere_contact_axial = R_s · z_b / Q_s,
    //   sphere_contact_radial = R_s · R_t / Q_s.
    let sph_contact_axial = big_r_s * z_b / q_s;
    let sph_contact_radial = big_r_s * major_radius / q_s;
    let sph_contact_center = c_s + cone_axis * sph_contact_axial;
    let contact_sph_circle = brepkit_math::curves::Circle3D::with_axes(
        sph_contact_center,
        cone_axis,
        sph_contact_radial,
        ref_dir,
        perp_y,
    )?;

    // Cone contact: closest point on cone to ball center, along the
    // outward cone normal. The cone surface at the contact has
    // axial offset (from apex) = z_b − a_apex_offset_from_sphere_center
    // − r·sin β (since the ball is offset r from the surface along the
    // outward normal direction, which has axis-component −cos β).
    //
    // Equivalently, contact_cone is on the cone surface line
    //   r = (z + h_signed) · cot β
    // closest to (R_t, z_b). Cone line in (axial, radial) form:
    // cos β · (z + h_signed) − sin β · r = 0 with unit normal
    // (cos β, −sin β). The cone tangency constraint chose R_t such
    // that R_t · sin β − (z_b + h_signed) · cos β = s_cone · r,
    // i.e. signed distance from (R_t, z_b) to the line = −s_cone · r.
    // Foot of perpendicular = (R_t, z_b) − (cos β, −sin β) · (−s_cone · r):
    //   cone_contact_axial = z_b + s_cone · r · cos β
    //   cone_contact_radial = R_t − s_cone · r · sin β
    let cone_contact_axial = z_b + s_cone * radius * cos_b;
    let cone_contact_radial = major_radius - s_cone * radius * sin_b;
    if cone_contact_radial <= tol_lin {
        return Ok(None);
    }
    let cone_contact_center = c_s + cone_axis * cone_contact_axial;
    let contact_cone_circle = brepkit_math::curves::Circle3D::with_axes(
        cone_contact_center,
        cone_axis,
        cone_contact_radial,
        ref_dir,
        perp_y,
    )?;

    let contact_sph = circle_arc_to_nurbs(&contact_sph_circle, u_start, u_end)?;
    let contact_cone = circle_arc_to_nurbs(&contact_cone_circle, u_start, u_end)?;

    // PCurves on each surface (constant-v Line2D).
    let sample_sph = contact_sph_circle.evaluate(u_start);
    let (u_sph_start, v_sph) = ParametricSurface::project_point(sph, sample_sph);
    let pcurve_sph = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_sph_start, v_sph),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample_cone = contact_cone_circle.evaluate(u_start);
    let (u_cone_start, v_cone) = ParametricSurface::project_point(cone, sample_cone);
    let pcurve_cone = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_cone_start, v_cone),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p_sph_at = |u: f64| contact_sph_circle.evaluate(u);
    let p_cone_at = |u: f64| contact_cone_circle.evaluate(u);
    let section_at = |u: f64, t: f64| CircSection {
        p1: p_sph_at(u),
        p2: p_cone_at(u),
        center: torus_center
            + ref_dir * (major_radius * u.cos())
            + perp_y * (major_radius * u.sin()),
        radius,
        uv1: (u_sph_start + (u - u_start), v_sph),
        uv2: (u_cone_start + (u - u_start), v_cone),
        t,
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1: pcurve_sph,
        pcurve2: pcurve_cone,
        contact1: contact_sph,
        contact2: contact_cone,
        face1: face_sphere,
        face2: face_cone,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between two intersecting spheres — the chamfer surface is an
/// axisymmetric cone connecting the two sphere-side contact circles.
///
/// `d1` is the geodesic distance on sphere1 (arc length along the
/// meridian from the spine, going INTO sphere1's face); `d2` likewise
/// on sphere2. Each sphere's "into face" direction is determined by
/// its convexity (face NOT reversed = convex, going AWAY from the
/// other sphere's center; face REVERSED = concave, going TOWARD).
///
/// All four convex/concave combinations are unified via per-sphere
/// `signed_offset_i = ±1` flipping the meridian arm.
///
/// # Geometry
///
/// Place the C1→C2 line as the symmetry axis. Spine at axial position
/// `a₀ = (R1² − R2² + D²)/(2D)`, radius `r_p = √(R1² − a₀²)`. With
/// `δi = di / Ri`, contact_i in cylindrical (r, axial) coordinates
/// (axial measured along C1→C2 from C1):
///   contact1.r = r_p cos δ1 + s1 · a₀ · sin δ1
///   contact1.z = a₀ cos δ1 − s1 · r_p · sin δ1
///   contact2.r = r_p cos δ2 + s2 · (D − a₀) · sin δ2
///   contact2.z = D − (D − a₀) cos δ2 + s2 · r_p · sin δ2
///
/// (At s1 = +1 / s2 = +1 the contacts go to the OUTSIDE caps — the
/// usual convex-convex case where each sphere's face is the cap
/// further from the other sphere's center.)
///
/// The chamfer surface is the cone obtained by rotating the line from
/// contact1 to contact2 around the C1→C2 axis. Apex on that axis at
/// `z_apex` where the line P1P2 hits `r = 0`; cone half-angle from
/// the radial plane is determined by the line's slope.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - spheres don't intersect properly (`D ≤ |R1−R2|` or
///     `D ≥ R1+R2`),
///   - contact line is degenerate (r-axial parallel: a flat disk; or
///     constant-r: a cylinder rather than a cone),
///   - sphere axes don't align with C1→C2,
///   - `d1` or `d2` non-positive, or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_sphere_chamfer(
    s1: &brepkit_math::surfaces::SphericalSurface,
    s2: &brepkit_math::surfaces::SphericalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let s1_signed: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2_signed: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r1 = s1.radius();
    let big_r2 = s2.radius();
    let c1 = s1.center();
    let c2 = s2.center();
    let c1_to_c2 = c2 - c1;
    let big_d = c1_to_c2.length();
    if big_d <= tol_lin {
        return Ok(None);
    }
    if big_d <= (big_r1 - big_r2).abs() + tol_lin || big_d >= big_r1 + big_r2 - tol_lin {
        return Ok(None);
    }

    let axis = (c1_to_c2 * (1.0 / big_d)).normalize()?;

    // Axisymmetry guards.
    if s1.z_axis().dot(axis).abs() < 1.0 - tol_ang || s2.z_axis().dot(axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    // Spine geometry.
    let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
    let r_p_sq = big_r1 * big_r1 - a0 * a0;
    if r_p_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let r_p = r_p_sq.sqrt();

    // Contact 1 on sphere 1 in (radial, axial-from-C1) coords.
    let delta1 = d1 / big_r1;
    let (sin1, cos1) = delta1.sin_cos();
    let p1_r = r_p * cos1 + s1_signed * a0 * sin1;
    let p1_z_from_c1 = a0 * cos1 - s1_signed * r_p * sin1;

    // Contact 2 on sphere 2 in (radial, axial-from-C1) coords. Sphere 2
    // is centered at C2 (axial offset = D from C1), so its formulas use
    // `(D − a0)` (axial distance from C2 to spine) where sphere 1 used
    // `a0`. That structural asymmetry — not `s2_signed` — is what
    // encodes the convex-convex case's "spheres extend away from each
    // other" geometry: in the convex-convex case both contacts lie on
    // their respective FAR caps. `s2_signed` only flips when sphere 2's
    // FACE is reversed (concave), redirecting its meridian arm just
    // like `s1_signed` does for sphere 1.
    let delta2 = d2 / big_r2;
    let (sin2, cos2) = delta2.sin_cos();
    let p2_r = r_p * cos2 + s2_signed * (big_d - a0) * sin2;
    let p2_z_from_c1 = big_d - (big_d - a0) * cos2 + s2_signed * r_p * sin2;

    // Both contacts have positive radial (must be on the sphere
    // surfaces — not past the pole/equator on the wrong side).
    if p1_r <= tol_lin || p2_r <= tol_lin {
        return Ok(None);
    }

    // Chamfer line P1→P2 in 2D (r, z). For an axisymmetric CONE we
    // need:
    //   - p1_r ≠ p2_r (else line is constant-r ⇒ cylinder, degenerate)
    //   - p1_z ≠ p2_z (else line is constant-z ⇒ flat disk)
    let dr = p2_r - p1_r;
    let dz = p2_z_from_c1 - p1_z_from_c1;
    if dr.abs() <= tol_lin || dz.abs() <= tol_lin {
        return Ok(None);
    }

    // Apex position: line P1→P2 extrapolated to r = 0.
    // r(t) = p1_r + t·dr = 0 ⇒ t = -p1_r/dr.
    // z(t) = p1_z + t·dz = p1_z - p1_r·dz/dr.
    let z_apex_from_c1 = p1_z_from_c1 - p1_r * dz / dr;

    // Cone axis: pointing AWAY from apex toward the contacts. The
    // contacts are at z_from_c1 = p1_z, p2_z; if z_apex < min(p1_z,
    // p2_z) the contacts are above apex and axis = +c1_to_c2; if
    // z_apex > max(p1_z, p2_z) the contacts are below and axis =
    // -c1_to_c2. The mid-contact direction sign tells us which:
    let mid_z_from_c1 = 0.5 * (p1_z_from_c1 + p2_z_from_c1);
    let cone_axis = if mid_z_from_c1 > z_apex_from_c1 {
        axis
    } else {
        -axis
    };

    // Cone half-angle from radial plane: generator from apex to a
    // contact has slope `tan β = |Δz_from_apex| / r_at_contact`.
    let dz_from_apex = mid_z_from_c1 - z_apex_from_c1;
    let r_avg = 0.5 * (p1_r + p2_r);
    let cone_half_angle = (dz_from_apex.abs() / r_avg).atan();
    if cone_half_angle <= 1e-3 || cone_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }

    let chamfer_apex_pos = c1 + axis * z_apex_from_c1;

    // Spine span (closed-circle aware).
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }

    // Reference direction perpendicular to the axis. Inherit sphere1's
    // frame (well-defined when its z-axis is aligned with `axis`).
    let s1_x = s1.x_axis();
    let s1_y = s1.y_axis();
    let ref_dir = if s1_x.cross(axis).length() > tol_ang {
        s1_x
    } else {
        s1_y
    };

    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, cone_axis, cone_half_angle, ref_dir)?;

    // Spine plane center.
    let spine_plane_center = c1 + axis * a0;
    let perp_y = axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let p_spine_start = spine.evaluate(topo, 0.0)?;
    let u_start = u_at(p_spine_start);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // 3D contact circles.
    let contact1_center = c1 + axis * p1_z_from_c1;
    let contact1_circle =
        brepkit_math::curves::Circle3D::with_axes(contact1_center, axis, p1_r, ref_dir, perp_y)?;
    let contact2_center = c1 + axis * p2_z_from_c1;
    let contact2_circle =
        brepkit_math::curves::Circle3D::with_axes(contact2_center, axis, p2_r, ref_dir, perp_y)?;
    let contact1 = circle_arc_to_nurbs(&contact1_circle, u_start, u_end)?;
    let contact2 = circle_arc_to_nurbs(&contact2_circle, u_start, u_end)?;

    // PCurves on each sphere — constant-v Line2D (axisymmetry guard).
    let sample1 = contact1_circle.evaluate(u_start);
    let v1 = ParametricSurface::project_point(s1, sample1).1;
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v1),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample2 = contact2_circle.evaluate(u_start);
    let v2 = ParametricSurface::project_point(s2, sample2).1;
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v2),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p1_at = |u: f64| contact1_circle.evaluate(u);
    let p2_at = |u: f64| contact2_circle.evaluate(u);
    let section_at = |u: f64, t: f64| {
        let p1 = p1_at(u);
        let p2 = p2_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: (u, v1),
            uv2: (u, v2),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between a sphere and a cylinder whose axis passes through
/// the sphere center — the chamfer surface is an axisymmetric cone
/// connecting the sphere-side and cylinder-side contact circles.
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1` (face NOT reversed = +1, face REVERSED = −1).
/// `signed_offset_sphere` flips the sphere meridian arm; `signed_offset_cyl`
/// flips the cylinder's "into face" axial direction.
///
/// `d1` is the geodesic distance on the sphere; `d2` is the axial
/// distance along the cylinder lateral.
///
/// # Geometry
///
/// Place sphere center at origin, cylinder axis = +z. Spine at axial
/// `a_spine = ±h_s = ±√(R_s² − r_c²)`, radial `r_c`. With `δ = d1 / R_s`,
/// `s_sph` = sphere signed_offset, `s_cyl` = cylinder signed_offset:
///
///   sphere_arm_sign = −spine_sign · s_sph
///   r_sph = r_c · cos δ + sphere_arm_sign · a_spine · sin δ
///         = r_c · cos δ − s_sph · h_s · sin δ      (using a_spine = spine_sign·h_s)
///   z_sph = a_spine · cos δ + s_sph · spine_sign · r_c · sin δ
///
///   z_cyl = a_spine − spine_sign · s_cyl · d2
///   r_cyl = r_c
///
/// For convex-convex (s_sph = s_cyl = +1) sphere goes toward the cap
/// AWAY from cylinder; cylinder goes AWAY from sphere along axis. For
/// concave (s = −1) the corresponding face flips its meridian arm.
///
/// The chamfer surface is the cone generated by rotating the line from
/// (r_sph, z_sph) to (r_cyl, z_cyl) around the cylinder axis. Apex on
/// the axis at `z_apex = z_sph − r_sph · (z_cyl − z_sph)/(r_cyl − r_sph)`.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere center isn't on the cylinder axis line,
///   - sphere parametric z-axis isn't aligned with cyl axis,
///   - sphere doesn't enclose cylinder (`r_c ≥ R_s`),
///   - the spine isn't at one of the two intersection circles
///     (axial = ±h_s, radial = r_c),
///   - chamfer line is degenerate (Δr ≈ 0 or Δz ≈ 0), or
///   - `d1` or `d2` non-positive.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_cylinder_chamfer(
    sph: &brepkit_math::surfaces::SphericalSurface,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face_sphere: FaceId,
    face_cyl: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let s_sph: f64 = if topo.face(face_sphere)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s_cyl: f64 = if topo.face(face_cyl)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r_s = sph.radius();
    let r_c = cyl.radius();
    let c_s = sph.center();
    let cyl_origin = cyl.origin();
    let cyl_axis = cyl.axis();

    // Sphere center on cylinder axis line.
    let to_sphere = c_s - cyl_origin;
    let to_sphere_v = Vec3::new(to_sphere.x(), to_sphere.y(), to_sphere.z());
    let along = to_sphere_v.dot(cyl_axis);
    let perp = to_sphere_v - cyl_axis * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }

    // Sphere parametric axis aligned with cyl axis.
    if sph.z_axis().dot(cyl_axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    if r_c >= big_r_s - tol_lin {
        return Ok(None);
    }
    let h_s_sq = big_r_s * big_r_s - r_c * r_c;
    let h_s = h_s_sq.sqrt();

    // Spine validation.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - c_s;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(cyl_axis);
    let sample_radial_v = to_sample_v - cyl_axis * sample_axial;
    let sample_radial = sample_radial_v.length();
    if (sample_axial.abs() - h_s).abs() > tol_lin || (sample_radial - r_c).abs() > tol_lin {
        return Ok(None);
    }
    let spine_sign = if sample_axial >= 0.0 { 1.0 } else { -1.0 };
    let a_spine = spine_sign * h_s;

    // Sphere contact along meridian going INTO sphere face. For
    // convex (s_sph=+1) this is the cap AWAY from cylinder; for
    // concave (s_sph=−1) it's the cap TOWARD cylinder. The
    // sphere_arm_sign (= −spine_sign · s_sph) selects the meridian
    // direction.
    let delta = d1 / big_r_s;
    let (sin_d, cos_d) = delta.sin_cos();
    let r_sph = r_c * cos_d - s_sph * h_s * sin_d;
    let z_sph = a_spine * cos_d + s_sph * r_c * spine_sign * sin_d;
    if r_sph <= tol_lin {
        // Sphere meridian swept past the pole — d1 too large.
        // (Only reachable for s_sph = +1, where r_sph = r_c·cos δ −
        // h_s·sin δ shrinks with δ. For s_sph = −1, r_sph =
        // r_c·cos δ + h_s·sin δ grows monotonically and never reaches
        // zero.)
        return Ok(None);
    }

    // Cylinder contact going INTO cylinder material from spine. For
    // convex (s_cyl=+1) we go AWAY from sphere along axis; for concave
    // (s_cyl=−1, cylinder = hole tool) we go TOWARD sphere.
    let r_cyl = r_c;
    let z_cyl = a_spine - spine_sign * s_cyl * d2;

    // Chamfer line P_sph → P_cyl in (r, axial) coords.
    let dr = r_cyl - r_sph;
    let dz = z_cyl - z_sph;
    if dr.abs() <= tol_lin || dz.abs() <= tol_lin {
        return Ok(None);
    }

    // Apex on cyl axis at the line's r=0 intersection.
    let z_apex = z_sph - r_sph * dz / dr;
    let mid_z = 0.5 * (z_sph + z_cyl);
    let chamfer_axis = if mid_z > z_apex { cyl_axis } else { -cyl_axis };
    let r_avg = 0.5 * (r_sph + r_cyl);
    let cone_half_angle = ((mid_z - z_apex).abs() / r_avg).atan();
    if cone_half_angle <= 1e-3 || cone_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }

    let chamfer_apex_pos = c_s + cyl_axis * z_apex;

    // Build chamfer cone using the cyl frame as ref dir.
    // `cyl.x_axis()` is always perpendicular to `cyl_axis` by
    // construction, so we don't need a fallback to `cyl_y`.
    let ref_dir = cyl.x_axis();
    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, chamfer_axis, cone_half_angle, ref_dir)?;

    // Spine plane center.
    let spine_plane_center = c_s + cyl_axis * a_spine;
    let perp_y = cyl_axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Contact circles.
    let sph_contact_center = c_s + cyl_axis * z_sph;
    let contact_sph_circle = brepkit_math::curves::Circle3D::with_axes(
        sph_contact_center,
        cyl_axis,
        r_sph,
        ref_dir,
        perp_y,
    )?;
    let cyl_contact_center = c_s + cyl_axis * z_cyl;
    let contact_cyl_circle = brepkit_math::curves::Circle3D::with_axes(
        cyl_contact_center,
        cyl_axis,
        r_cyl,
        ref_dir,
        perp_y,
    )?;
    let contact_sph = circle_arc_to_nurbs(&contact_sph_circle, u_start, u_end)?;
    let contact_cyl = circle_arc_to_nurbs(&contact_cyl_circle, u_start, u_end)?;

    // PCurves on each surface, derived in the surface's own u frame.
    let sample_sph = contact_sph_circle.evaluate(u_start);
    let (u_sph_start, v_sph) = ParametricSurface::project_point(sph, sample_sph);
    let pcurve_sph = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_sph_start, v_sph),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample_cyl = contact_cyl_circle.evaluate(u_start);
    let u_cyl_start = ParametricSurface::project_point(cyl, sample_cyl).0;
    let v_cyl = cyl_v_at_point(cyl, sample_cyl);
    let pcurve_cyl = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_cyl_start, v_cyl),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p_sph_at = |u: f64| contact_sph_circle.evaluate(u);
    let p_cyl_at = |u: f64| contact_cyl_circle.evaluate(u);
    let section_at = |u: f64, t: f64| {
        let p1 = p_sph_at(u);
        let p2 = p_cyl_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: (u_sph_start + (u - u_start), v_sph),
            uv2: (u_cyl_start + (u - u_start), v_cyl),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1: pcurve_sph,
        pcurve2: pcurve_cyl,
        contact1: contact_sph,
        contact2: contact_cyl,
        face1: face_sphere,
        face2: face_cyl,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between a sphere and a cone whose axis passes through the
/// sphere center — the chamfer surface is an axisymmetric cone
/// connecting the sphere-side and cone-side contact circles.
///
/// `d1` is the geodesic distance on the sphere (arc length along the
/// meridian from the spine, going INTO sphere face); `d2` is the
/// linear distance along the cone's generator (going INTO cone
/// material from the spine, toward the apex).
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1`. `s_sph` flips the sphere meridian arm;
/// `s_cone` flips the cone-generator direction (toward vs away from
/// apex).
///
/// # Geometry
///
/// Place sphere center at origin, cone axis = +z, cone apex at
/// `(0, 0, −h_signed)`. Spine on sphere ∩ cone: `r² + z² = R_s²` AND
/// `r = (z + h_signed) · cot β`.
///
/// With `δ1 = d1/R_s` and `sphere_arm_sign = −spine_sign · s_sph`:
///   r_sph = r_spine · cos δ1 + sphere_arm_sign · spine_z · sin δ1
///   z_sph = spine_z · cos δ1 − sphere_arm_sign · r_spine · sin δ1
///
/// Cone contact along the generator. For convex (s_cone=+1) the
/// "into face" direction is TOWARD apex; for concave (s_cone=−1, cone
/// is a hole tool) it's AWAY from apex:
///   r_cone = r_spine − s_cone · d2 · cos β
///   z_cone = spine_z − s_cone · d2 · sin β
///
/// The chamfer surface is the cone obtained by rotating the line
/// P_sph → P_cone around the cone axis. Apex on axis at the line's
/// r=0 intersection.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - sphere center isn't on the cone axis line,
///   - sphere parametric z-axis isn't aligned with cone axis,
///   - β is degenerate (≤ tol or ≥ π/2 − tol),
///   - the spine isn't at a valid sphere ∩ cone intersection circle,
///   - chamfer line is degenerate (Δr ≈ 0 or Δz ≈ 0), or
///   - `d1` or `d2` non-positive.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sphere_cone_chamfer(
    sph: &brepkit_math::surfaces::SphericalSurface,
    cone: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face_sphere: FaceId,
    face_cone: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let s_sph: f64 = if topo.face(face_sphere)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s_cone: f64 = if topo.face(face_cone)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let big_r_s = sph.radius();
    let c_s = sph.center();
    let cone_apex = cone.apex();
    let cone_axis = cone.axis();
    let beta = cone.half_angle();

    // Sphere center on cone axis line.
    let to_sphere = c_s - cone_apex;
    let to_sphere_v = Vec3::new(to_sphere.x(), to_sphere.y(), to_sphere.z());
    let along = to_sphere_v.dot(cone_axis);
    let perp = to_sphere_v - cone_axis * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }

    // Sphere z-axis aligned with cone axis.
    if sph.z_axis().dot(cone_axis).abs() < 1.0 - tol_ang {
        return Ok(None);
    }

    let (sin_b, cos_b) = beta.sin_cos();
    if sin_b <= tol_lin || cos_b <= tol_lin {
        return Ok(None);
    }
    let cot_b = cos_b / sin_b;

    let h_signed = along; // axial offset of sphere center from apex along cone_axis

    // Spine validation: solve sphere ∩ cone for the two candidate spine
    // axials (in sphere-centered coords).
    let qa = 1.0 / (sin_b * sin_b);
    let qb = 2.0 * h_signed * cot_b * cot_b;
    let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
    let q_disc = qb * qb - 4.0 * qa * qc;
    if q_disc <= tol_lin * tol_lin {
        return Ok(None);
    }
    let q_disc_sqrt = q_disc.sqrt();
    let z_spine_root_a = (-qb + q_disc_sqrt) / (2.0 * qa);
    let z_spine_root_b = (-qb - q_disc_sqrt) / (2.0 * qa);

    // Match the spine sample.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - c_s;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(cone_axis);
    let sample_radial_v = to_sample_v - cone_axis * sample_axial;
    let sample_radial = sample_radial_v.length();
    let spine_match_tol = tol_lin * 1e3;
    let spine_z = if (sample_axial - z_spine_root_a).abs() < spine_match_tol {
        z_spine_root_a
    } else if (sample_axial - z_spine_root_b).abs() < spine_match_tol {
        z_spine_root_b
    } else {
        return Ok(None);
    };
    let r_spine = (spine_z + h_signed) * cot_b;
    if r_spine <= tol_lin || (sample_radial - r_spine).abs() > spine_match_tol {
        return Ok(None);
    }
    // Spine must be above apex along cone_axis (spine_z + h_signed > 0
    // for r_spine > 0 with cot β > 0).
    if spine_z + h_signed <= tol_lin {
        return Ok(None);
    }

    // Sphere-side contact going INTO sphere face. For convex (s_sph=+1)
    // this is the cap AWAY from cone (`sphere_arm_sign = -spine_sign`);
    // for concave (s_sph=-1) it's the cap TOWARD cone (sign flipped).
    let spine_sign = if spine_z >= 0.0 { 1.0 } else { -1.0 };
    let sphere_arm_sign = -spine_sign * s_sph;
    let delta1 = d1 / big_r_s;
    let (sin_d1, cos_d1) = delta1.sin_cos();
    let r_sph = r_spine * cos_d1 + sphere_arm_sign * spine_z * sin_d1;
    let z_sph = spine_z * cos_d1 - sphere_arm_sign * r_spine * sin_d1;
    if r_sph <= tol_lin {
        // Sphere meridian swept past the pole — d1 too large.
        // (Reachable in both convex and concave depending on direction;
        // the sign of `sphere_arm_sign · spine_z` decides whether r_sph
        // shrinks or grows with d1.)
        return Ok(None);
    }

    // Cone-side contact along the generator. For convex (s_cone=+1) we
    // go TOWARD apex (away from spine on the apex side). For concave
    // (s_cone=−1, cone is a hole tool) we go AWAY from apex along the
    // generator. The cone-arm sign is `s_cone` directly (since the
    // generator unit vector going TOWARD apex is `-(cos β, sin β)` in
    // r-z, and `−s_cone` flips the sign for concave).
    let r_cone = r_spine - s_cone * d2 * cos_b;
    let z_cone = spine_z - s_cone * d2 * sin_b;
    if r_cone <= tol_lin {
        // Cone overshoot toward the apex — only reachable for the convex
        // case (s_cone = +1). For concave (s_cone = −1), `r_cone =
        // r_spine + d2·cos β` strictly grows with d2 and never reaches
        // tol; this branch is dead in concave but harmless.
        return Ok(None);
    }

    // Chamfer line from sphere-contact to cone-contact in (r, z).
    let dr = r_cone - r_sph;
    let dz = z_cone - z_sph;
    if dr.abs() <= tol_lin || dz.abs() <= tol_lin {
        return Ok(None);
    }

    // Apex of chamfer cone on axis at line P_sph→P_cone extrapolated to r=0.
    let z_apex_chamfer = z_sph - r_sph * dz / dr;
    let mid_z = 0.5 * (z_sph + z_cone);
    let chamfer_axis = if mid_z > z_apex_chamfer {
        cone_axis
    } else {
        -cone_axis
    };
    let r_avg = 0.5 * (r_sph + r_cone);
    let cone_half_angle = ((mid_z - z_apex_chamfer).abs() / r_avg).atan();
    if cone_half_angle <= 1e-3 || cone_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }

    let chamfer_apex_pos = c_s + cone_axis * z_apex_chamfer;

    // Build chamfer cone using cone's frame as ref dir (cone.x_axis()
    // is always perpendicular to cone_axis).
    let ref_dir = cone.x_axis();
    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, chamfer_axis, cone_half_angle, ref_dir)?;

    // Spine plane center.
    let spine_plane_center = c_s + cone_axis * spine_z;
    let perp_y = cone_axis.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Contact circles.
    let sph_contact_center = c_s + cone_axis * z_sph;
    let contact_sph_circle = brepkit_math::curves::Circle3D::with_axes(
        sph_contact_center,
        cone_axis,
        r_sph,
        ref_dir,
        perp_y,
    )?;
    let cone_contact_center = c_s + cone_axis * z_cone;
    let contact_cone_circle = brepkit_math::curves::Circle3D::with_axes(
        cone_contact_center,
        cone_axis,
        r_cone,
        ref_dir,
        perp_y,
    )?;
    let contact_sph = circle_arc_to_nurbs(&contact_sph_circle, u_start, u_end)?;
    let contact_cone = circle_arc_to_nurbs(&contact_cone_circle, u_start, u_end)?;

    // PCurves on each surface — derive u in surface's own frame.
    let sample_sph = contact_sph_circle.evaluate(u_start);
    let (u_sph_start, v_sph) = ParametricSurface::project_point(sph, sample_sph);
    let pcurve_sph = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_sph_start, v_sph),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample_cone = contact_cone_circle.evaluate(u_start);
    let (u_cone_start, v_cone) = ParametricSurface::project_point(cone, sample_cone);
    let pcurve_cone = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_cone_start, v_cone),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p_sph_at = |u: f64| contact_sph_circle.evaluate(u);
    let p_cone_at = |u: f64| contact_cone_circle.evaluate(u);
    let section_at = |u: f64, t: f64| {
        let p1 = p_sph_at(u);
        let p2 = p_cone_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: (u_sph_start + (u - u_start), v_sph),
            uv2: (u_cone_start + (u - u_start), v_cone),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1: pcurve_sph,
        pcurve2: pcurve_cone,
        contact1: contact_sph,
        contact2: contact_cone,
        face1: face_sphere,
        face2: face_cone,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between two cylinders with **parallel axes**, intersecting in
/// a pair of straight lines (not circles). The rolling-ball blend is an
/// exact cylinder around an axis parallel to the original cylinder axes.
///
/// This is the only cylinder × cylinder configuration with a clean
/// closed-form blend — perpendicular or oblique-axis cylinders intersect
/// in a 4th-degree curve and require the walker.
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1`.
///
/// # Geometry
///
/// Place cyl1 axis = +z through origin (axis-aligned in cyl1's frame).
/// cyl2 axis is parallel; project its origin offset onto the perpendicular
/// plane to get displacement vector `d_perp` of length `D`. The two cyls
/// intersect when `|r1 − r2| < D < r1 + r2`; their intersection consists
/// of two straight lines parallel to the cyl axes at perpendicular
/// position
///   x_spine = (r1² − r2² + D²) / (2D)   along d_perp from cyl1 axis,
///   y_spine = ±√(r1² − x_spine²)        perpendicular to both.
///
/// With `Q1 = r1 + s1·r`, `Q2 = r2 + s2·r`, the rolling-ball position
/// follows the same algebra:
///   x_ball = (Q1² − Q2² + D²) / (2D),
///   y_ball = sign(spine_y) · √(Q1² − x_ball²).
///
/// The fillet surface is the cylinder of radius `r` around the ball
/// trajectory line `(x_ball, y_ball, z)`. Cyl1-side contact line at
/// `(R1·x_ball/Q1, R1·y_ball/Q1, z)`, cyl2-side at `(D + R2·(x_ball−D)/Q2,
/// R2·y_ball/Q2, z)`.
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - cylinder axes aren't parallel (general cyl-cyl is non-analytic),
///   - cylinders don't intersect (`D ≤ |r1−r2|` or `D ≥ r1+r2`),
///   - effective radii collapse (`Q_i ≤ tol`),
///   - the spine isn't on one of the two intersection lines, or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn cylinder_cylinder_fillet(
    cyl1: &brepkit_math::surfaces::CylindricalSurface,
    cyl2: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::CylindricalSurface;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if radius <= tol_lin {
        return Ok(None);
    }
    let s1: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let r1 = cyl1.radius();
    let r2 = cyl2.radius();
    let a1 = cyl1.axis();
    let a2 = cyl2.axis();

    // Cylinder axes must be parallel (or anti-parallel).
    if a1.dot(a2).abs() < 1.0 - tol_ang {
        return Ok(None);
    }
    // Use cyl1's axis as the canonical direction; flip cyl2's if needed.
    let a_cyl = a1; // shared direction

    // Perpendicular displacement from cyl1 axis to cyl2 axis.
    let o1 = cyl1.origin();
    let o2 = cyl2.origin();
    let d_axes = o2 - o1;
    let d_axes_v = Vec3::new(d_axes.x(), d_axes.y(), d_axes.z());
    let along = d_axes_v.dot(a_cyl);
    let d_perp = d_axes_v - a_cyl * along;
    let big_d = d_perp.length();
    if big_d <= tol_lin {
        // Coaxial cylinders — no intersection (parallel surfaces).
        return Ok(None);
    }

    // Intersection requires |r1 − r2| < D < r1 + r2.
    if big_d <= (r1 - r2).abs() + tol_lin || big_d >= r1 + r2 - tol_lin {
        return Ok(None);
    }

    // Build local frame in the perpendicular plane: x̂ along d_perp,
    // ŷ perpendicular (in the perpendicular plane).
    let x_hat = d_perp * (1.0 / big_d);
    let y_hat = a_cyl.cross(x_hat).normalize()?;

    // Spine geometry (sphere-cylinder pattern, but with two LINEAR spines).
    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine_sq = r1 * r1 - x_spine * x_spine;
    if y_spine_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let y_spine_abs = y_spine_sq.sqrt();

    // Spine validation: parallel-axis cyl-cyl spines are LINEAR (parallel
    // to the cyl axes), so a closed spine signals degenerate or erroneous
    // input — straight lines can't form loops. Bail to walker.
    let edges = spine.edges();
    if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        if e.start() == e.end() {
            return Ok(None);
        }
    }
    let spine_len = spine.length();
    if spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - o1;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_x = to_sample_v.dot(x_hat);
    let sample_y = to_sample_v.dot(y_hat);
    let spine_match_tol = tol_lin * 1e3;
    if (sample_x - x_spine).abs() > spine_match_tol {
        return Ok(None);
    }
    let y_spine = if (sample_y - y_spine_abs).abs() < spine_match_tol {
        y_spine_abs
    } else if (sample_y + y_spine_abs).abs() < spine_match_tol {
        -y_spine_abs
    } else {
        return Ok(None);
    };
    let y_sign = if y_spine >= 0.0 { 1.0 } else { -1.0 };

    // Effective radii.
    let q1 = r1 + s1 * radius;
    let q2 = r2 + s2 * radius;
    if q1 <= tol_lin || q2 <= tol_lin {
        return Ok(None);
    }

    // Rolling-ball center in (x, y) of the perpendicular plane.
    let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
    let y_ball_sq = q1 * q1 - x_ball * x_ball;
    if y_ball_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let y_ball = y_sign * y_ball_sq.sqrt();

    // Spine endpoints in 3D.
    let p_spine_start = p_spine_sample;
    let spine_tangent = spine.tangent(topo, 0.0)?;
    // Confirm spine direction is parallel to the cyl axis (linear spine
    // must be along the parallel axis).
    if spine_tangent.dot(a_cyl).abs() < 1.0 - tol_ang {
        return Ok(None);
    }
    let p_spine_end = spine.evaluate(topo, spine_len)?;

    // Project spine endpoints axially to (x_spine, y_spine, z).
    // The spine line has fixed (x, y) in cyl1's frame; only z varies.
    // Get z extent from spine endpoint axials relative to o1.
    let to_start = p_spine_start - o1;
    let to_start_v = Vec3::new(to_start.x(), to_start.y(), to_start.z());
    let z_start = to_start_v.dot(a_cyl);
    let to_end = p_spine_end - o1;
    let to_end_v = Vec3::new(to_end.x(), to_end.y(), to_end.z());
    let z_end = to_end_v.dot(a_cyl);

    // Fillet cylinder: axis parallel to a_cyl, origin at the ball line
    // at the spine_start z.
    let ball_line_origin = o1 + x_hat * x_ball + y_hat * y_ball + a_cyl * z_start;
    let fillet_cyl = CylindricalSurface::new(ball_line_origin, a_cyl, radius)?;

    // Cyl1 contact line in 3D: (r1·x_ball/q1, r1·y_ball/q1, z) in cyl1's frame.
    let c1_x = r1 * x_ball / q1;
    let c1_y = r1 * y_ball / q1;
    let c1_start = o1 + x_hat * c1_x + y_hat * c1_y + a_cyl * z_start;
    let c1_end = o1 + x_hat * c1_x + y_hat * c1_y + a_cyl * z_end;

    // Cyl2 contact line in 3D: ((D + R2·(x_ball−D)/q2), R2·y_ball/q2, z)
    // in cyl1's frame.
    let c2_x = big_d + r2 * (x_ball - big_d) / q2;
    let c2_y = r2 * y_ball / q2;
    let c2_start = o1 + x_hat * c2_x + y_hat * c2_y + a_cyl * z_start;
    let c2_end = o1 + x_hat * c2_x + y_hat * c2_y + a_cyl * z_end;

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // PCurves: each contact line lies at constant cyl-radial direction
    // on its respective cylinder — so it's a constant-u line with v
    // ranging over [z_start, z_end] (cylinder's v parameter is axial).
    let u1 = ParametricSurface::project_point(cyl1, c1_start).0;
    let v1_start = cyl_v_at_point(cyl1, c1_start);
    let v1_end = cyl_v_at_point(cyl1, c1_end);
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u1, v1_start),
        brepkit_math::vec::Vec2::new(0.0, v1_end - v1_start),
    )?);
    let u2 = ParametricSurface::project_point(cyl2, c2_start).0;
    let v2_start = cyl_v_at_point(cyl2, c2_start);
    let v2_end = cyl_v_at_point(cyl2, c2_end);
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u2, v2_start),
        brepkit_math::vec::Vec2::new(0.0, v2_end - v2_start),
    )?);

    // Cross-sections at spine endpoints.
    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: ball_line_origin,
        radius,
        uv1: (u1, v1_start),
        uv2: (u2, v2_start),
        t: 0.0,
    };
    let ball_end = ball_line_origin + a_cyl * (z_end - z_start);
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: ball_end,
        radius,
        uv1: (u1, v1_end),
        uv2: (u2, v2_end),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cylinder(fillet_cyl),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Fillet between two **coaxial** cones with different half-angles —
/// the rolling-ball blend is an exact torus around the shared axis.
///
/// The two cones must share the SAME axis line (axes coincident) AND
/// have different half-angles; otherwise the cones either don't
/// intersect or coincide identically. When both conditions hold,
/// they intersect in a single circle on the shared axis.
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1`.
///
/// # Geometry
///
/// Place the shared axis = +z, cone1 apex at z = 0 (β1), cone2 apex at
/// z = h_2 (β2). At axial z, cone_i radius = (z − z_apex_i) · cot β_i.
/// Setting equal yields the spine z:
///   z_spine = h_2 · cos β2 · sin β1 / sin(β1 − β2)
/// and r_spine = z_spine · cot β1 (must be > 0; both cones must exist
/// at z_spine).
///
/// For the rolling ball, two linear cone-tangency constraints
/// (one per cone) solve uniquely for `(R_t, z_b)`:
///   z_b = [h_2 · cos β2 · sin β1 + r · (s1 · sin β2 − s2 · sin β1)] / sin(β1 − β2)
///   R_t = [z_b · (cos β1 − cos β2) + h_2 · cos β2 + (s1 − s2) · r] / (sin β1 − sin β2)
/// Note: unlike sphere-cone, there's no quadratic — the rolling ball
/// has a unique position because the cone-cone intersection is a
/// SINGLE circle (not a pair).
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - cones aren't coaxial (axis lines don't coincide),
///   - half-angles equal (sin(β1−β2) ≈ 0; cones identical or shifted),
///   - resulting r_spine ≤ tol or major < minor (spindle), or
///   - the spine isn't at the predicted (axial, radial) position, or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn cone_cone_coaxial_fillet(
    cone1: &brepkit_math::surfaces::ConicalSurface,
    cone2: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ToroidalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if radius <= tol_lin {
        return Ok(None);
    }
    let s1: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let beta1 = cone1.half_angle();
    let beta2 = cone2.half_angle();
    let apex1 = cone1.apex();
    let apex2 = cone2.apex();
    let axis1 = cone1.axis();
    let axis2 = cone2.axis();

    // Axes must point in the SAME direction. Cone axis direction is
    // geometrically significant: flipping the axis selects the opposite
    // nappe (v ≥ 0 from apex). Anti-parallel axes would feed the
    // formulas the wrong cone, producing incorrect z_spine and
    // potentially silently bypassing the spine-validation gate. Reject.
    if axis1.dot(axis2) < 1.0 - tol_ang {
        return Ok(None);
    }
    let a_cone = axis1; // shared axis direction

    // Apex line: A2 must lie on the line through A1 along a_cone.
    let to_apex2 = apex2 - apex1;
    let to_apex2_v = Vec3::new(to_apex2.x(), to_apex2.y(), to_apex2.z());
    let along = to_apex2_v.dot(a_cone);
    let perp = to_apex2_v - a_cone * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }
    let h_2 = along; // axial offset of cone2 apex from cone1 apex.

    // Different half-angles required. With both half-angles in
    // (0, π/2) (per `ConicalSurface::new`), `sin` is strictly monotone,
    // so `sin(β1 − β2) ≈ 0 ⇔ β1 ≈ β2 ⇔ sin β1 ≈ sin β2`. A single
    // sin-minus check therefore suffices; sin_diff has the same zero set.
    let (sin_b1, cos_b1) = beta1.sin_cos();
    let (sin_b2, cos_b2) = beta2.sin_cos();
    let sin_diff = sin_b1 - sin_b2;
    let sin_minus = (beta1 - beta2).sin();
    if sin_minus.abs() <= tol_ang {
        return Ok(None);
    }

    // Solve linear system for (R_t, z_b).
    let z_b = (h_2 * cos_b2 * sin_b1 + radius * (s1 * sin_b2 - s2 * sin_b1)) / sin_minus;
    let r_t = (z_b * (cos_b1 - cos_b2) + h_2 * cos_b2 + (s1 - s2) * radius) / sin_diff;
    if r_t <= tol_lin {
        return Ok(None);
    }

    let major_radius = r_t;
    let minor_radius = radius;
    if major_radius < minor_radius - tol_lin {
        return Ok(None);
    }

    // Spine z (r=0 case) and r_spine.
    let z_spine = h_2 * cos_b2 * sin_b1 / sin_minus;
    let cot_b1 = cos_b1 / sin_b1;
    let r_spine = z_spine * cot_b1;
    if r_spine <= tol_lin {
        return Ok(None);
    }

    // Spine validation.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - apex1;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(a_cone);
    let sample_radial_v = to_sample_v - a_cone * sample_axial;
    let sample_radial = sample_radial_v.length();
    let spine_match_tol = tol_lin * 1e3;
    if (sample_axial - z_spine).abs() > spine_match_tol
        || (sample_radial - r_spine).abs() > spine_match_tol
    {
        return Ok(None);
    }

    // Build the torus.
    let ref_dir = cone1.x_axis();
    let torus_center = apex1 + a_cone * z_b;
    let torus = ToroidalSurface::with_axis_and_ref_dir(
        torus_center,
        major_radius,
        minor_radius,
        a_cone,
        ref_dir,
    )?;

    // Spine plane center at apex1 + z_spine.
    let spine_plane_center = apex1 + a_cone * z_spine;
    let perp_y = a_cone.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Cone i contact: foot of perpendicular from ball (R_t, z_b) onto
    // cone_i's meridian line. Same formula as sphere-cone:
    //   contact_axial_from_apex_i = (z_b − z_apex_i) + s_i · r · cos β_i
    //   contact_radial            = R_t − s_i · r · sin β_i.
    // Express axials in apex1-relative coords.
    let cone1_contact_axial = z_b + s1 * radius * cos_b1;
    let cone1_contact_radial = major_radius - s1 * radius * sin_b1;
    let cone2_contact_axial = z_b + s2 * radius * cos_b2;
    let cone2_contact_radial = major_radius - s2 * radius * sin_b2;
    if cone1_contact_radial <= tol_lin || cone2_contact_radial <= tol_lin {
        return Ok(None);
    }

    let cone1_contact_center = apex1 + a_cone * cone1_contact_axial;
    let contact1_circle = brepkit_math::curves::Circle3D::with_axes(
        cone1_contact_center,
        a_cone,
        cone1_contact_radial,
        ref_dir,
        perp_y,
    )?;
    let cone2_contact_center = apex1 + a_cone * cone2_contact_axial;
    let contact2_circle = brepkit_math::curves::Circle3D::with_axes(
        cone2_contact_center,
        a_cone,
        cone2_contact_radial,
        ref_dir,
        perp_y,
    )?;

    let contact1 = circle_arc_to_nurbs(&contact1_circle, u_start, u_end)?;
    let contact2 = circle_arc_to_nurbs(&contact2_circle, u_start, u_end)?;

    // PCurves on each cone (constant-v Line2D).
    let sample_c1 = contact1_circle.evaluate(u_start);
    let (u_c1_start, v_c1) = ParametricSurface::project_point(cone1, sample_c1);
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_c1_start, v_c1),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample_c2 = contact2_circle.evaluate(u_start);
    let (u_c2_start, v_c2) = ParametricSurface::project_point(cone2, sample_c2);
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_c2_start, v_c2),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p1_at = |u: f64| contact1_circle.evaluate(u);
    let p2_at = |u: f64| contact2_circle.evaluate(u);
    let section_at = |u: f64, t: f64| CircSection {
        p1: p1_at(u),
        p2: p2_at(u),
        center: torus_center
            + ref_dir * (major_radius * u.cos())
            + perp_y * (major_radius * u.sin()),
        radius,
        uv1: (u_c1_start + (u - u_start), v_c1),
        uv2: (u_c2_start + (u - u_start), v_c2),
        t,
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Torus(torus),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between two **coaxial** cones with different half-angles —
/// the chamfer surface is an axisymmetric cone connecting the two
/// cone-generator contact circles.
///
/// `d1` is the linear distance along cone1's generator from the spine
/// (going INTO cone1's face); `d2` likewise for cone2. The convex
/// "into face" direction on each cone is opposite: cone1 goes TOWARD
/// apex1 and cone2 goes AWAY from apex2 (since they extend from
/// opposite sides of the spine in the typical β1 > β2 setup).
///
/// Handles all four convex/concave combinations via per-face
/// `signed_offset_i = ±1`.
///
/// # Geometry
///
/// Place shared axis = +z, cone1 apex at z=0, cone2 apex at z=h_2.
/// Spine at axial z_spine, radial r_spine (from `cone_cone_coaxial_fillet`).
///
/// Generator unit direction on cone i (away from apex): `(cos β_i, sin β_i)`
/// in (r, z). Contact along generator going INTO face material:
///   contact1 = (r_spine − s1·d1·cos β1, z_spine − s1·d1·sin β1)
///   contact2 = (r_spine + s2·d2·cos β2, z_spine + s2·d2·sin β2)
///
/// (For convex s1 = s2 = +1: cone1 retreats toward apex1 (down-left),
/// cone2 extends away from apex2 (up-right). The chord goes
/// up-and-out from contact1 to contact2.)
///
/// The chamfer cone is the surface of revolution of the line from
/// contact1 to contact2 around the shared axis. Apex on axis at the
/// line's r=0 intersection; cone axis points from apex toward the
/// contacts (whichever side they're on).
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - cones aren't coaxial,
///   - half-angles equal (sin(β1−β2) ≈ 0),
///   - chamfer line is degenerate (Δr ≈ 0 or Δz ≈ 0),
///   - the spine isn't at the predicted (axial, radial),
///   - either contact lands at non-positive radial, or
///   - the spine is degenerate.
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn cone_cone_coaxial_chamfer(
    cone1: &brepkit_math::surfaces::ConicalSurface,
    cone2: &brepkit_math::surfaces::ConicalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    use brepkit_math::surfaces::ConicalSurface;
    use std::f64::consts::PI;

    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let s1: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let beta1 = cone1.half_angle();
    let beta2 = cone2.half_angle();
    let apex1 = cone1.apex();
    let apex2 = cone2.apex();
    let axis1 = cone1.axis();
    let axis2 = cone2.axis();

    if axis1.dot(axis2) < 1.0 - tol_ang {
        return Ok(None);
    }
    let a_cone = axis1;

    let to_apex2 = apex2 - apex1;
    let to_apex2_v = Vec3::new(to_apex2.x(), to_apex2.y(), to_apex2.z());
    let along = to_apex2_v.dot(a_cone);
    let perp = to_apex2_v - a_cone * along;
    if perp.length() > tol_lin {
        return Ok(None);
    }
    let h_2 = along;

    let (sin_b1, cos_b1) = beta1.sin_cos();
    let (sin_b2, cos_b2) = beta2.sin_cos();
    let sin_minus = (beta1 - beta2).sin();
    if sin_minus.abs() <= tol_ang {
        return Ok(None);
    }

    // Spine geometry (shared with fillet).
    let z_spine = h_2 * cos_b2 * sin_b1 / sin_minus;
    let cot_b1 = cos_b1 / sin_b1;
    let r_spine = z_spine * cot_b1;
    if r_spine <= tol_lin {
        return Ok(None);
    }

    // Spine validation.
    let edges = spine.edges();
    let is_closed_spine = if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        e.start() == e.end()
    } else {
        false
    };
    let spine_len = spine.length();
    if !is_closed_spine && spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - apex1;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_axial = to_sample_v.dot(a_cone);
    let sample_radial_v = to_sample_v - a_cone * sample_axial;
    let sample_radial = sample_radial_v.length();
    let spine_match_tol = tol_lin * 1e3;
    if (sample_axial - z_spine).abs() > spine_match_tol
        || (sample_radial - r_spine).abs() > spine_match_tol
    {
        return Ok(None);
    }

    // Per-cone contacts along generators.
    let r_c1 = r_spine - s1 * d1 * cos_b1;
    let z_c1 = z_spine - s1 * d1 * sin_b1;
    let r_c2 = r_spine + s2 * d2 * cos_b2;
    let z_c2 = z_spine + s2 * d2 * sin_b2;
    if r_c1 <= tol_lin || r_c2 <= tol_lin {
        return Ok(None);
    }

    // Chamfer line P1 → P2 in (r, z). The Δr guard avoids `r_c1·dz/dr`
    // blowing up to ±∞ when the line is vertical (Δr = 0); the
    // `dz ≈ 0` case (horizontal line ⇒ flat-disk chamfer) is caught
    // downstream by the half-angle ≤ 1e-3 check, so we don't need a
    // separate guard for it.
    let dr = r_c2 - r_c1;
    let dz = z_c2 - z_c1;
    if dr.abs() <= tol_lin {
        return Ok(None);
    }

    // Apex on axis at line r=0.
    let z_apex_chamfer = z_c1 - r_c1 * dz / dr;
    let mid_z = 0.5 * (z_c1 + z_c2);
    let chamfer_axis = if mid_z > z_apex_chamfer {
        a_cone
    } else {
        -a_cone
    };
    let r_avg = 0.5 * (r_c1 + r_c2);
    let cone_half_angle = ((mid_z - z_apex_chamfer).abs() / r_avg).atan();
    // Reject near-degenerate cone (close to flat disk or needle).
    // brepkit's `ConicalSurface::new` rejects β ≤ 0 or β ≥ π/2; the
    // 1e-3 rad ≈ 0.057° margin is a project-wide convention used by all
    // analytic chamfer helpers (plane-cone, sphere-cone, cyl-cyl-fillet)
    // for the same purpose — see `plane_cone_chamfer` for context.
    if cone_half_angle <= 1e-3 || cone_half_angle >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }

    let chamfer_apex_pos = apex1 + a_cone * z_apex_chamfer;
    let ref_dir = cone1.x_axis();
    let chamfer_cone =
        ConicalSurface::with_ref_dir(chamfer_apex_pos, chamfer_axis, cone_half_angle, ref_dir)?;

    // Spine plane center.
    let spine_plane_center = apex1 + a_cone * z_spine;
    let perp_y = a_cone.cross(ref_dir).normalize()?;
    let u_at = |p: Point3| {
        let v = p - spine_plane_center;
        perp_y.dot(v).atan2(ref_dir.dot(v))
    };
    let u_start = u_at(p_spine_sample);
    let u_end = if is_closed_spine {
        u_start + 2.0 * PI
    } else {
        let p_spine_end = spine.evaluate(topo, spine_len)?;
        let u_end_raw = u_at(p_spine_end);
        if u_end_raw > u_start {
            u_end_raw
        } else {
            u_end_raw + 2.0 * PI
        }
    };

    // Contact circles.
    let c1_center = apex1 + a_cone * z_c1;
    let contact1_circle =
        brepkit_math::curves::Circle3D::with_axes(c1_center, a_cone, r_c1, ref_dir, perp_y)?;
    let c2_center = apex1 + a_cone * z_c2;
    let contact2_circle =
        brepkit_math::curves::Circle3D::with_axes(c2_center, a_cone, r_c2, ref_dir, perp_y)?;
    let contact1 = circle_arc_to_nurbs(&contact1_circle, u_start, u_end)?;
    let contact2 = circle_arc_to_nurbs(&contact2_circle, u_start, u_end)?;

    // PCurves on each cone (constant-v Line2D).
    let sample_c1 = contact1_circle.evaluate(u_start);
    let (u_c1_start, v_c1) = ParametricSurface::project_point(cone1, sample_c1);
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_c1_start, v_c1),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);
    let sample_c2 = contact2_circle.evaluate(u_start);
    let (u_c2_start, v_c2) = ParametricSurface::project_point(cone2, sample_c2);
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_c2_start, v_c2),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // Cross-sections.
    let p1_at = |u: f64| contact1_circle.evaluate(u);
    let p2_at = |u: f64| contact2_circle.evaluate(u);
    let section_at = |u: f64, t: f64| {
        let p1 = p1_at(u);
        let p2 = p2_at(u);
        let mid = midpoint_3d(p1, p2);
        CircSection {
            p1,
            p2,
            center: mid,
            radius: (p1 - p2).length() * 0.5,
            uv1: (u_c1_start + (u - u_start), v_c1),
            uv2: (u_c2_start + (u - u_start), v_c2),
            t,
        }
    };
    let section_start = section_at(u_start, 0.0);
    let section_end = section_at(u_end, 1.0);

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cone(chamfer_cone),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Chamfer between two cylinders with **parallel axes**, intersecting in
/// a pair of straight lines. The chamfer surface is a **plane** that
/// contains the two contact lines (each parallel to the cyl axes).
///
/// Convex/concave configurations are unified via per-face
/// `signed_offset_i = ±1`: the angular displacement on each cylinder
/// from the spine flips toward or away from the OTHER cylinder. For
/// convex-convex, both contacts move AWAY from the other cyl;
/// concave-concave moves both TOWARD; mixed configurations swap one.
///
/// # Geometry
///
/// In the perpendicular plane (cyl1's frame: x along d_perp, y
/// perpendicular, z along axis), with the spine at `(x_spine, y_spine, *)`:
///
///   cyl1 contact angular displacement: Δθ_1 = sign(y_spine) · s1 · d1 / r1
///   cyl1 contact (x, y) =
///     (x_spine·cos Δθ_1 − y_spine·sin Δθ_1,
///      y_spine·cos Δθ_1 + x_spine·sin Δθ_1)
///
///   cyl2 contact (in cyl2's local frame, then translated by D along x):
///     Δθ_2 = −sign(y_spine) · s2 · d2 / r2 (note the negation: cyl2's
///     "AWAY from cyl1" direction is opposite cyl1's "AWAY from cyl2")
///     contact_in_cyl2 =
///       ((x_spine−D)·cos Δθ_2 − y_spine·sin Δθ_2,
///        y_spine·cos Δθ_2 + (x_spine−D)·sin Δθ_2)
///     contact_global = (D + that.x, that.y)
///
/// The chamfer surface is a plane whose normal is `ẑ × (c2 − c1)`
/// (perpendicular to both the chord between contacts and the shared
/// axis direction; defined up to sign).
///
/// # Returns
///
/// `Ok(None)` (walker fallback) when:
///   - cylinder axes aren't parallel,
///   - cylinders don't intersect (`D ≤ |r1−r2|` or `D ≥ r1+r2`),
///   - the spine isn't on one of the two intersection lines, or
///   - chamfer line is degenerate (both contacts coincide).
///
/// # Errors
///
/// Returns `BlendError` if topology lookups or NURBS construction fails.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn cylinder_cylinder_chamfer(
    cyl1: &brepkit_math::surfaces::CylindricalSurface,
    cyl2: &brepkit_math::surfaces::CylindricalSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    let tol_lin = ANALYTIC_TOL_LIN;
    let tol_ang = ANALYTIC_TOL_ANG;

    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    let s1: f64 = if topo.face(face1)?.is_reversed() {
        -1.0
    } else {
        1.0
    };
    let s2: f64 = if topo.face(face2)?.is_reversed() {
        -1.0
    } else {
        1.0
    };

    let r1 = cyl1.radius();
    let r2 = cyl2.radius();
    let a1 = cyl1.axis();
    let a2 = cyl2.axis();

    // Cylinder axes must be parallel.
    if a1.dot(a2).abs() < 1.0 - tol_ang {
        return Ok(None);
    }
    let a_cyl = a1;

    // Perpendicular displacement between cyl axes.
    let o1 = cyl1.origin();
    let o2 = cyl2.origin();
    let d_axes = o2 - o1;
    let d_axes_v = Vec3::new(d_axes.x(), d_axes.y(), d_axes.z());
    let perp = d_axes_v - a_cyl * d_axes_v.dot(a_cyl);
    let big_d = perp.length();
    if big_d <= tol_lin {
        return Ok(None);
    }
    if big_d <= (r1 - r2).abs() + tol_lin || big_d >= r1 + r2 - tol_lin {
        return Ok(None);
    }

    let x_hat = perp * (1.0 / big_d);
    let y_hat = a_cyl.cross(x_hat).normalize()?;

    // Spine intersection lines.
    let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
    let y_spine_sq = r1 * r1 - x_spine * x_spine;
    if y_spine_sq <= tol_lin * tol_lin {
        return Ok(None);
    }
    let y_spine_abs = y_spine_sq.sqrt();

    // Spine-line validation.
    let edges = spine.edges();
    if edges.len() == 1 {
        let e = topo.edge(edges[0])?;
        if e.start() == e.end() {
            return Ok(None);
        }
    }
    let spine_len = spine.length();
    if spine_len < tol_lin {
        return Ok(None);
    }
    let p_spine_sample = spine.evaluate(topo, 0.0)?;
    let to_sample = p_spine_sample - o1;
    let to_sample_v = Vec3::new(to_sample.x(), to_sample.y(), to_sample.z());
    let sample_x = to_sample_v.dot(x_hat);
    let sample_y = to_sample_v.dot(y_hat);
    let spine_match_tol = tol_lin * 1e3;
    if (sample_x - x_spine).abs() > spine_match_tol {
        return Ok(None);
    }
    let y_spine = if (sample_y - y_spine_abs).abs() < spine_match_tol {
        y_spine_abs
    } else if (sample_y + y_spine_abs).abs() < spine_match_tol {
        -y_spine_abs
    } else {
        return Ok(None);
    };
    let y_sign = if y_spine >= 0.0 { 1.0 } else { -1.0 };

    let p_spine_start = p_spine_sample;
    let p_spine_end = spine.evaluate(topo, spine_len)?;
    let spine_tangent = spine.tangent(topo, 0.0)?;
    if spine_tangent.dot(a_cyl).abs() < 1.0 - tol_ang {
        return Ok(None);
    }
    let to_start = p_spine_start - o1;
    let z_start = Vec3::new(to_start.x(), to_start.y(), to_start.z()).dot(a_cyl);
    let to_end = p_spine_end - o1;
    let z_end = Vec3::new(to_end.x(), to_end.y(), to_end.z()).dot(a_cyl);

    // Angular displacements per the unified formula.
    let dtheta1 = y_sign * s1 * d1 / r1;
    let dtheta2 = -y_sign * s2 * d2 / r2;
    let (sin1, cos1) = dtheta1.sin_cos();
    let (sin2, cos2) = dtheta2.sin_cos();

    // Cyl1 contact in cyl1's frame.
    let c1_x = x_spine * cos1 - y_spine * sin1;
    let c1_y = y_spine * cos1 + x_spine * sin1;
    // Cyl2 contact: in cyl2's frame use spine_local = (x_spine − D, y_spine).
    let c2_local_x = (x_spine - big_d) * cos2 - y_spine * sin2;
    let c2_local_y = y_spine * cos2 + (x_spine - big_d) * sin2;
    // Translate cyl2 contact back to global frame (cyl2 origin is at +D along x_hat).
    let c2_x = c2_local_x + big_d;
    let c2_y = c2_local_y;

    // Contact lines in 3D (parallel to a_cyl).
    let c1_start = o1 + x_hat * c1_x + y_hat * c1_y + a_cyl * z_start;
    let c1_end = o1 + x_hat * c1_x + y_hat * c1_y + a_cyl * z_end;
    let c2_start = o1 + x_hat * c2_x + y_hat * c2_y + a_cyl * z_start;
    let c2_end = o1 + x_hat * c2_x + y_hat * c2_y + a_cyl * z_end;

    let chamfer_span_v = c2_start - c1_start;
    if chamfer_span_v.length() <= tol_lin {
        return Ok(None);
    }
    let chamfer_normal_raw = a_cyl.cross(chamfer_span_v);
    let chamfer_normal = chamfer_normal_raw
        .normalize()
        .map_err(|_| BlendError::Math(brepkit_math::MathError::ZeroVector))?;
    let chamfer_d = chamfer_normal.dot(Vec3::new(c1_start.x(), c1_start.y(), c1_start.z()));

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // PCurves: each contact line is at constant u (angular) and varying
    // v (axial) on its respective cylinder.
    let u1 = ParametricSurface::project_point(cyl1, c1_start).0;
    let v1_start = cyl_v_at_point(cyl1, c1_start);
    let v1_end = cyl_v_at_point(cyl1, c1_end);
    let pcurve1 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u1, v1_start),
        brepkit_math::vec::Vec2::new(0.0, v1_end - v1_start),
    )?);
    let u2 = ParametricSurface::project_point(cyl2, c2_start).0;
    let v2_start = cyl_v_at_point(cyl2, c2_start);
    let v2_end = cyl_v_at_point(cyl2, c2_end);
    let pcurve2 = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u2, v2_start),
        brepkit_math::vec::Vec2::new(0.0, v2_end - v2_start),
    )?);

    // Cross-sections at spine endpoints.
    let chamfer_radius = (c1_start - c2_start).length() * 0.5;
    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: midpoint_3d(c1_start, c2_start),
        radius: chamfer_radius,
        uv1: (u1, v1_start),
        uv2: (u2, v2_start),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: midpoint_3d(c1_end, c2_end),
        radius: chamfer_radius,
        uv1: (u1, v1_end),
        uv2: (u2, v2_end),
        t: 1.0,
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Plane {
            normal: chamfer_normal,
            d: chamfer_d,
        },
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };
    Ok(Some(StripeResult {
        stripe,
        new_edges: Vec::new(),
    }))
}

/// Build a rational quadratic NURBS for an arc on a `Circle3D` from
/// `t_start` to `t_end` (radians).
///
/// Decomposes the arc span into quarter-pi pieces; each piece becomes one
/// rational quadratic Bezier with weight `cos(half_angle)` on the off-curve
/// middle control point. The result is geometrically exact for circles
/// (unlike the chord-only approximation used by `nurbs_line` for short
/// arcs).
///
/// Inlined here to keep `crates/blend` from picking up a dependency on
/// `brepkit-geometry`. The geometry crate has the same algorithm in
/// `convert::curve_to_nurbs::circle_to_nurbs`; consolidating both into the
/// math layer is a follow-up.
fn circle_arc_to_nurbs(
    circle: &brepkit_math::curves::Circle3D,
    t_start: f64,
    t_end: f64,
) -> Result<brepkit_math::nurbs::curve::NurbsCurve, BlendError> {
    use std::f64::consts::FRAC_PI_2;

    let span = t_end - t_start;
    if span.abs() < 1e-15 {
        return Err(BlendError::Math(brepkit_math::MathError::ZeroVector));
    }

    let n_arcs = ((span.abs() / FRAC_PI_2).ceil() as usize).max(1);
    #[allow(clippy::cast_precision_loss)]
    let delta = span / n_arcs as f64;

    let n_cps = 2 * n_arcs + 1;
    let mut cps: Vec<Point3> = Vec::with_capacity(n_cps);
    let mut weights: Vec<f64> = Vec::with_capacity(n_cps);

    let mut knots: Vec<f64> = Vec::with_capacity(2 * n_arcs + 5);
    knots.push(0.0);
    knots.push(0.0);
    knots.push(0.0);
    for i in 1..n_arcs {
        #[allow(clippy::cast_precision_loss)]
        let knot = i as f64 / n_arcs as f64;
        knots.push(knot);
        knots.push(knot);
    }
    knots.push(1.0);
    knots.push(1.0);
    knots.push(1.0);

    for arc_idx in 0..n_arcs {
        #[allow(clippy::cast_precision_loss)]
        let t0 = t_start + arc_idx as f64 * delta;
        let t1 = t0 + delta;
        let half_angle = delta * 0.5;
        let r = circle.radius();

        let p0 = circle.evaluate(t0);
        let p1 = circle.evaluate(t1);
        // Tangent at endpoints, scaled by `r` so the segment-segment
        // intersection lands at the off-curve control point.
        let tan0 = circle.tangent(t0) * r;
        let tan1 = circle.tangent(t1) * r;
        let p_mid = tangent_intersection(p0, tan0, p1, tan1);

        let w_mid = half_angle.abs().cos();
        if arc_idx == 0 {
            cps.push(p0);
            weights.push(1.0);
        }
        cps.push(p_mid);
        weights.push(w_mid);
        cps.push(p1);
        weights.push(1.0);
    }

    Ok(brepkit_math::nurbs::curve::NurbsCurve::new(
        2, knots, cps, weights,
    )?)
}

/// Intersect two parametric rays `p0 + s·d0`, `p1 + t·d1` and return the
/// 3D point. Falls back to the midpoint when the rays are near-parallel,
/// which matches the round-trip behavior of `circle_arc_to_nurbs` for
/// degenerate inputs.
fn tangent_intersection(p0: Point3, d0: Vec3, p1: Point3, d1: Vec3) -> Point3 {
    let rhs = p1 - p0;
    let cross = d0.cross(d1);
    let cx = cross.x().abs();
    let cy = cross.y().abs();
    let cz = cross.z().abs();
    let (a00, a01, b0, a10, a11, b1) = if cz >= cx && cz >= cy {
        (d0.x(), -d1.x(), rhs.x(), d0.y(), -d1.y(), rhs.y())
    } else if cy >= cx {
        (d0.x(), -d1.x(), rhs.x(), d0.z(), -d1.z(), rhs.z())
    } else {
        (d0.y(), -d1.y(), rhs.y(), d0.z(), -d1.z(), rhs.z())
    };
    let det = a00 * a11 - a01 * a10;
    if det.abs() < 1e-30 {
        return Point3::new(
            (p0.x() + p1.x()) * 0.5,
            (p0.y() + p1.y()) * 0.5,
            (p0.z() + p1.z()) * 0.5,
        );
    }
    let s = (b0 * a11 - b1 * a01) / det;
    p0 + d0 * s
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Create a spine along a single edge from `a` to `b`, plus two dummy faces.
    fn make_spine(topo: &mut Topology, a: Point3, b: Point3) -> (Spine, FaceId, FaceId) {
        let v0 = topo.add_vertex(Vertex::new(a, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(b, 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        let oe = OrientedEdge::new(eid, true);
        let w1 = topo.add_wire(Wire::new(vec![oe], false).unwrap());
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let f1 = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let f2 = topo.add_face(Face::new(
            w2,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));

        let spine = Spine::from_single_edge(topo, eid).unwrap();
        (spine, f1, f2)
    }

    #[test]
    fn plane_plane_90_degree_fillet() {
        let mut topo = Topology::new();

        // Two perpendicular planes meeting along X axis at origin
        let n1 = Vec3::new(0.0, 0.0, 1.0); // XY plane (top)
        let n2 = Vec3::new(0.0, 1.0, 0.0); // XZ plane (front)
        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );

        let radius = 2.0;
        let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

        // The result surface should be a cylinder
        match &result.stripe.surface {
            FaceSurface::Cylinder(cyl) => {
                assert!(
                    (cyl.radius() - radius).abs() < 1e-10,
                    "Expected radius {radius}, got {}",
                    cyl.radius()
                );

                let axis = cyl.axis();
                assert!(
                    axis.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
                    "Cylinder axis should be along X, got {axis:?}"
                );
            }
            other => panic!("Expected Cylinder surface, got {other:?}"),
        }

        // Contact curves should be lines parallel to X axis
        let c1_start = result.stripe.contact1.evaluate(0.0);
        let c1_end = result.stripe.contact1.evaluate(1.0);
        let c1_dir = (c1_end - c1_start).normalize().unwrap();
        assert!(
            c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
            "Contact 1 should be along X"
        );

        // Half-angle for 90 deg is pi/4, so offset = R/sin(pi/4) = R*sqrt(2)
        let half_angle = std::f64::consts::FRAC_PI_4;
        let expected_offset = radius / half_angle.sin();
        let sections = &result.stripe.sections;
        assert_eq!(sections.len(), 2);
        assert!((sections[0].radius - radius).abs() < 1e-10);

        // Center should be offset from origin by R/sin(45deg) along bisector
        let bisector = (n1 + n2).normalize().unwrap();
        let expected_center = Point3::new(0.0, 0.0, 0.0) + bisector * expected_offset;
        let actual_center = sections[0].center;
        assert!(
            (actual_center - expected_center).length() < 1e-10,
            "Expected center at {expected_center:?}, got {actual_center:?}"
        );
    }

    #[test]
    fn plane_plane_60_degree_fillet() {
        let mut topo = Topology::new();

        let n1 = Vec3::new(0.0, 0.0, 1.0);
        // Normal at 60 deg from n1
        let angle = std::f64::consts::FRAC_PI_3;
        let n2 = Vec3::new(0.0, angle.sin(), angle.cos());
        let n2 = n2.normalize().unwrap();

        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
        );

        let radius = 1.5;
        let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

        match &result.stripe.surface {
            FaceSurface::Cylinder(cyl) => {
                assert!(
                    (cyl.radius() - radius).abs() < 1e-10,
                    "Expected radius {radius}, got {}",
                    cyl.radius()
                );
            }
            other => panic!("Expected Cylinder surface, got {other:?}"),
        }

        // Verify center offset matches expected geometry
        let cos_angle = n1.dot(n2);
        let half = cos_angle.acos() / 2.0;
        let expected_offset = radius / half.sin();

        let center = result.stripe.sections[0].center;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let actual_offset = (center - origin).length();
        assert!(
            (actual_offset - expected_offset).abs() < 1e-10,
            "Expected offset {expected_offset}, got {actual_offset}"
        );
    }

    #[test]
    fn plane_plane_chamfer_is_flat() {
        let mut topo = Topology::new();

        let n1 = Vec3::new(0.0, 0.0, 1.0);
        let n2 = Vec3::new(0.0, 1.0, 0.0);
        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );

        let d1 = 3.0;
        let d2 = 2.0;
        let result = plane_plane_chamfer(&spine, &topo, n1, n2, d1, d2, f1, f2).unwrap();

        // The result should be a plane
        match &result.stripe.surface {
            FaceSurface::Plane { normal, d } => {
                // Normal should be perpendicular to the spine direction (X axis)
                let spine_dir = Vec3::new(1.0, 0.0, 0.0);
                assert!(
                    normal.dot(spine_dir).abs() < 1e-10,
                    "Chamfer normal should be perpendicular to spine, dot={:.6}",
                    normal.dot(spine_dir)
                );
                assert!(
                    (normal.length() - 1.0).abs() < 1e-10,
                    "Normal should be unit length"
                );
                assert!(d.is_finite(), "d should be finite");
            }
            other => panic!("Expected Plane surface for chamfer, got {other:?}"),
        }

        // Contact curves should be lines parallel to X
        let c1_start = result.stripe.contact1.evaluate(0.0);
        let c1_end = result.stripe.contact1.evaluate(1.0);
        let c1_dir = (c1_end - c1_start).normalize().unwrap();
        assert!(
            c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
            "Contact 1 should be along X"
        );
    }

    #[test]
    fn non_analytic_returns_none() {
        let mut topo = Topology::new();

        // One NURBS surface — should return None
        let nurbs_surf = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                    vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
                ],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            )
            .unwrap(),
        );
        let plane_surf = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };

        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );

        let result =
            try_analytic_fillet(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, f1, f2).unwrap();
        assert!(result.is_none(), "NURBS-Plane pair should return None");

        let result =
            try_analytic_chamfer(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, 1.0, f1, f2)
                .unwrap();
        assert!(result.is_none(), "NURBS-Plane chamfer should return None");
    }

    /// Concave plane-cylinder fillet — verifies the analytic helper's
    /// concave branch (cylinder face reversed = "hole through plate")
    /// emits a torus with `major = r_c − r` (NOT `r_c + r`), positioned
    /// to touch the plate inside the spine and the cylinder lateral
    /// inside the hole.
    ///
    /// Built via direct topology synthesis since `boolean(Cut)` would
    /// tessellate the cylinder lateral and yield a polygonal hole.
    #[test]
    fn plane_cylinder_fillet_concave_emits_torus_with_smaller_major() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r_c: f64 = 2.0;
        let r_fillet = 0.3;

        // Spine: a closed Circle3D edge of radius r_c around the +z axis,
        // sharing a single vertex (start == end) since the spine wraps.
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Plate face — non-reversed, normal = +z (top of plate).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Cylinder face — REVERSED, marking it as the wall of a hole
        // (topological outward = -radial, opposite to the geometric +radial
        // returned by ParametricSurface::normal).
        let cyl_surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl_surface.clone()),
        ));

        // The fillet dispatcher applies `orient_plane_surface` so the helper
        // sees the inward plane normal. For a plate top with outward = +z
        // and material below, inward = -z.
        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

        let result = plane_cylinder_fillet(
            n_p_inward,
            0.0,
            &cyl_surface,
            &spine,
            &topo,
            r_fillet,
            face_plate,
            face_cyl,
        )
        .unwrap()
        .expect("concave plane-cylinder fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Concave: major = r_c − r_fillet = 1.7, minor = 0.3.
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-9,
            "torus minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );
        assert!(
            (torus.major_radius() - (r_c - r_fillet)).abs() < 1e-9,
            "torus major should be r_c − r_fillet = {} for concave, got {}",
            r_c - r_fillet,
            torus.major_radius()
        );

        // The torus center sits at `+r` ABOVE the plate (in the empty
        // wedge direction = -n_p_inward = +z), distinguishing the concave
        // case from the convex one (which would have center at `-r`).
        let center = torus.center();
        assert!(
            (center.x()).abs() < 1e-9 && (center.y()).abs() < 1e-9,
            "torus center should be on the cylinder axis"
        );
        assert!(
            (center.z() - r_fillet).abs() < 1e-9,
            "concave torus center should sit at z = +r ({r_fillet}), got {}",
            center.z()
        );
    }

    /// Convex plane-cylinder fillet sanity check — the existing
    /// `fillet_cylinder_base_circle_produces_torus` integration test
    /// covers the convex path through `fillet_v2`, but adding a direct
    /// helper-level convex test alongside the concave one above guards
    /// against regression in the shared `major_radius` branch.
    #[test]
    fn plane_cylinder_fillet_convex_emits_torus_with_larger_major() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r_c: f64 = 2.0;
        let r_fillet = 0.3;

        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
        ));

        // NOT reversed — typical post-on-plate cylinder face.
        let cyl_surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl_surface.clone()),
        ));

        // For the cylinder primitive bottom rim the dispatcher gives
        // n_p_inward = +z (after flipping the bottom-cap outward = -z).
        let n_p_inward = Vec3::new(0.0, 0.0, 1.0);

        let result = plane_cylinder_fillet(
            n_p_inward,
            0.0,
            &cyl_surface,
            &spine,
            &topo,
            r_fillet,
            face_plate,
            face_cyl,
        )
        .unwrap()
        .expect("convex plane-cylinder fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        assert!(
            (torus.major_radius() - (r_c + r_fillet)).abs() < 1e-9,
            "torus major should be r_c + r_fillet = {} for convex, got {}",
            r_c + r_fillet,
            torus.major_radius()
        );
        // Convex case: torus center at z = -r (below plate, in empty wedge).
        let center = torus.center();
        assert!(
            (center.z() - (-r_fillet)).abs() < 1e-9,
            "convex torus center should sit at z = -r ({}), got {}",
            -r_fillet,
            center.z()
        );
    }

    /// Concave plane-cylinder fillet rejects radii ≥ r_c/2 — past that
    /// threshold `major = r_c - r ≤ minor = r` and the construction
    /// becomes a self-intersecting spindle torus, which is invalid as a
    /// fillet surface. Convex must still accept radii up to `r_c`.
    #[test]
    fn plane_cylinder_fillet_concave_rejects_spindle_radius() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let r_c: f64 = 2.0;

        let setup = |topo: &mut Topology, reversed: bool| {
            let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
            let circle =
                Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
            let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let spine = Spine::from_single_edge(topo, eid).unwrap();
            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
            let face_plate = topo.add_face(Face::new(
                w1,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, -1.0),
                    d: 0.0,
                },
            ));
            let cyl =
                CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                    .unwrap();
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
            let face_cyl = if reversed {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Cylinder(cyl.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())))
            };
            (spine, cyl, face_plate, face_cyl)
        };

        // Concave: r > r_c/2 ⇒ would form spindle torus ⇒ reject.
        let mut topo_concave = Topology::new();
        let (spine_concave, cyl_concave, fp_concave, fc_concave) = setup(&mut topo_concave, true);
        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
        let result = plane_cylinder_fillet(
            n_p_inward,
            0.0,
            &cyl_concave,
            &spine_concave,
            &topo_concave,
            // Just above r_c/2 = 1.0 — would produce major = 0.9 < minor = 1.1.
            1.1,
            fp_concave,
            fc_concave,
        )
        .unwrap();
        assert!(
            result.is_none(),
            "concave fillet must reject r > r_c/2 (spindle-torus regime)"
        );

        // Concave at r exactly r_c/2 is also a degenerate equality (major
        // = minor); rejected.
        let result_eq = plane_cylinder_fillet(
            n_p_inward,
            0.0,
            &cyl_concave,
            &spine_concave,
            &topo_concave,
            1.0,
            fp_concave,
            fc_concave,
        )
        .unwrap();
        assert!(
            result_eq.is_none(),
            "concave fillet must reject r = r_c/2 (degenerate major = minor)"
        );

        // Convex at r_c/2 < r < r_c is still fine — major = r_c + r > minor.
        let mut topo_convex = Topology::new();
        let (spine_convex, cyl_convex, fp_convex, fc_convex) = setup(&mut topo_convex, false);
        let n_p_inward_convex = Vec3::new(0.0, 0.0, 1.0);
        let result_convex = plane_cylinder_fillet(
            n_p_inward_convex,
            0.0,
            &cyl_convex,
            &spine_convex,
            &topo_convex,
            1.5,
            fp_convex,
            fc_convex,
        )
        .unwrap();
        assert!(
            result_convex.is_some(),
            "convex fillet should accept r in (r_c/2, r_c)"
        );
    }

    /// Concave plane-cone fillet ("tapered hole through plate") emits a
    /// torus with `major = r_p − r·cot(α/2)` (the convex case is
    /// `r_p + r·cot(α/2)`). Direct helper test mirroring the
    /// plane-cylinder concave coverage.
    #[test]
    fn plane_cone_fillet_concave_emits_torus_with_smaller_major() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        // Apex 6 units above the plate, half-angle α = atan2(6, 3) so the
        // cone-plate intersection (the spine) lands at radius r_p = 3.
        let alpha = 6.0_f64.atan2(3.0);
        let r_p = 3.0;
        let r_fillet = 0.3;

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Plate face: outward = +z (top of plate, plate material below).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Cone face REVERSED — wall of a tapered hole, with topological
        // outward pointing into the empty hole.
        let cone_surface =
            ConicalSurface::new(Point3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0), alpha)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone_surface.clone()),
        ));

        // Plate top has outward = +z; after `orient_plane_surface` the
        // helper sees inward = -z.
        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

        let result = plane_cone_fillet(
            n_p_inward,
            0.0,
            &cone_surface,
            &spine,
            &topo,
            r_fillet,
            face_plate,
            face_cone,
        )
        .unwrap()
        .expect("concave plane-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Expected: major = r_p - r·cot(α/2). For α ≈ 1.107 (atan2(6,3)),
        // cot(α/2) ≈ 1.618. So major ≈ 3 - 0.485 = 2.515.
        let expected_major = r_p - r_fillet * (alpha * 0.5).tan().recip();
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-9,
            "torus minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );
        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "concave torus major should be r_p − r·cot(α/2) ≈ {expected_major:.6}, got {}",
            torus.major_radius()
        );

        // Center sits at +r ABOVE the plate (in the empty wedge direction
        // = -n_p_inward = +z), distinguishing concave from convex
        // (which has center at -r below the plate).
        let center = torus.center();
        assert!(
            (center.z() - r_fillet).abs() < 1e-9,
            "concave torus center should sit at z = +r ({r_fillet}), got {}",
            center.z()
        );
    }

    /// Concave plane-cone fillet rejects radii that would produce a
    /// spindle torus (i.e. when `r·(cot(α/2) + 1) ≥ r_p` so
    /// `major ≤ minor`). At the cylinder limit α = π/2 this collapses to
    /// `r ≥ r_p/2`, matching the plane-cylinder bound.
    #[test]
    fn plane_cone_fillet_concave_rejects_spindle_radius() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        // Same cone setup as the previous test.
        let alpha = 6.0_f64.atan2(3.0);
        let r_p = 3.0;
        let cot_half = (alpha * 0.5).tan().recip();
        // Max valid concave radius: r_p / (cot(α/2) + 1).
        let r_max = r_p / (cot_half + 1.0);

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let cone_surface =
            ConicalSurface::new(Point3::new(0.0, 0.0, 6.0), Vec3::new(0.0, 0.0, -1.0), alpha)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone_surface.clone()),
        ));
        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

        // Just above r_max → spindle regime → reject.
        let result_spindle = plane_cone_fillet(
            n_p_inward,
            0.0,
            &cone_surface,
            &spine,
            &topo,
            r_max * 1.01,
            face_plate,
            face_cone,
        )
        .unwrap();
        assert!(
            result_spindle.is_none(),
            "concave fillet must reject r > r_p / (cot(α/2)+1) (spindle-torus regime)"
        );

        // Exactly at r_max → horn-torus boundary (major = minor), where the
        // tube touches the axis at a degenerate point — also rejected.
        let result_horn = plane_cone_fillet(
            n_p_inward,
            0.0,
            &cone_surface,
            &spine,
            &topo,
            r_max,
            face_plate,
            face_cone,
        )
        .unwrap();
        assert!(
            result_horn.is_none(),
            "concave fillet must reject r = r_p / (cot(α/2)+1) (horn-torus boundary)"
        );

        // Below r_max — should succeed.
        let result_ok = plane_cone_fillet(
            n_p_inward,
            0.0,
            &cone_surface,
            &spine,
            &topo,
            r_max * 0.5,
            face_plate,
            face_cone,
        )
        .unwrap();
        assert!(
            result_ok.is_some(),
            "concave fillet should accept r below the spindle threshold"
        );
    }

    /// Concave plane-cylinder chamfer (chamfer at the top rim of a hole
    /// through a plate). The chamfer face is a cone whose plate-side
    /// contact lands at radial `r_c + d1` (outside the spine, in the
    /// surrounding plate material), and whose cylinder-side contact lands
    /// at axial `−d2` along the hole wall going into the plate.
    #[test]
    fn plane_cylinder_chamfer_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r_c: f64 = 2.0;
        let d = 0.4;

        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Plate top face: outward = +z (raw — chamfer dispatcher passes
        // unflipped; it's the plate top of a plate with a hole).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Cylinder face REVERSED — the hole wall, with topological outward
        // pointing into the empty hole.
        let cyl_surface =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl_surface.clone()),
        ));

        let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
        let result = plane_cylinder_chamfer(
            n_p_inward,
            0.0,
            &cyl_surface,
            &spine,
            &topo,
            d,
            d,
            face_plate,
            face_cyl,
        )
        .unwrap()
        .expect("concave plane-cylinder chamfer should produce a stripe");

        let cone_surf = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Half-angle for symmetric chamfer is π/4 in either case.
        assert!(
            (cone_surf.half_angle() - std::f64::consts::FRAC_PI_4).abs() < 1e-12,
            "chamfer cone half-angle should be π/4 for symmetric d, got {}",
            cone_surf.half_angle()
        );

        // Plate-side contact at `r_c + d` (in the surrounding plate
        // material at z=0), cylinder-side contact at `-d` (going down
        // into the hole wall).
        // Frame3::from_normal(+z) gives x_axis = (0, 1, 0); chamfer cone's
        // axis is -z (= -apex_dir = -(s·n_p_inward) = -(-z) = +z, but
        // wait — let me re-derive: s = -1 for concave; apex_dir =
        // s · n_p_inward = -n_p_inward = -z; cone_axis = -apex_dir = +z).
        // So cone axis is +z, frame x_axis = (0, 1, 0).
        let want_plate = Point3::new(0.0, r_c + d, 0.0);
        let want_cyl = Point3::new(0.0, r_c, -d);
        let mut closest_plate = f64::INFINITY;
        let mut closest_cyl = f64::INFINITY;
        for i in 0..1440 {
            let v = (f64::from(i) / 1440.0) * std::f64::consts::TAU;
            // Try u=0 across a range of v.
            let p = ParametricSurface::evaluate(&cone_surf, 0.0, v);
            closest_plate = closest_plate.min((p - want_plate).length());
            closest_cyl = closest_cyl.min((p - want_cyl).length());
        }
        // The cone surface at SOME (u=0, v) should pass close to both
        // contact points; sampling along v with sufficient density.
        assert!(
            closest_plate < 1e-3,
            "concave chamfer cone should pass near plate contact at {want_plate:?}; closest = {closest_plate:.6}"
        );
        assert!(
            closest_cyl < 1e-3,
            "concave chamfer cone should pass near cyl contact at {want_cyl:?}; closest = {closest_cyl:.6}"
        );
    }

    /// Convex plane-sphere fillet: a sphere intersecting a plate from above
    /// (post-on-slab configuration). The fillet rounds the convex spine
    /// circle and produces a Toroidal blend surface.
    ///
    /// Scenario:
    ///   - Plate top face at z=0 with raw outward = +z.
    ///   - After orient_plane_surface (which the dispatcher applies for
    ///     fillet), n_p_inward = -z (into plate material below).
    ///   - Sphere center at (0,0,h)=( 0,0,1) above plate, radius R=2,
    ///     so spine circle is z=0 with r_p = √(R²−h²) = √3.
    ///   - Fillet radius r=0.3.
    ///
    /// Predicted analytics (using h_signed = -h = -1):
    ///   - R_t² = r_p² + 2r(R − h_signed) = 3 + 2·0.3·(2−(−1)) = 4.8
    ///   - Torus center: p_axis − n_p_inward · r = (0,0,0) − (−z)·0.3 = (0,0,+0.3)
    ///   - Plate contact at radial R_t, z=0
    ///   - Sphere contact at radial R·R_t/(R+r) ≈ 1.905,
    ///     z = +r(R − h_signed)/(R + r) ≈ +0.391 (above plate)
    #[test]
    fn plane_sphere_fillet_convex_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let h_real: f64 = 1.0;
        let r_fillet: f64 = 0.3;
        let r_p_sq = big_r * big_r - h_real * h_real;
        let r_p = r_p_sq.sqrt();

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Plate top face at z=0 with raw outward = +z (away from plate
        // material at z<0). After orient_plane_surface, the dispatcher
        // would pass n_p_inward = -z; we mirror that here.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Sphere centered above plate, NOT reversed (convex post).
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(sphere.clone())));

        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
        let result = plane_sphere_fillet(
            n_p_inward,
            0.0,
            &sphere,
            &spine,
            &topo,
            r_fillet,
            face_plate,
            face_sphere,
        )
        .unwrap()
        .expect("convex plane-sphere fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let expected_major_sq = r_p_sq + 2.0 * r_fillet * (big_r + h_real);
        let expected_major = expected_major_sq.sqrt();
        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "torus major should be √(r_p² + 2r(R+h)) = {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "torus minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center at z = +r (above plate, in empty wedge between
        // plate top and upper hemisphere).
        let center = torus.center();
        assert!(
            (center.x()).abs() < 1e-12 && (center.y()).abs() < 1e-12,
            "torus center should be on z-axis, got {center:?}"
        );
        assert!(
            (center.z() - r_fillet).abs() < 1e-12,
            "torus center z should be +r_fillet = {r_fillet}, got {}",
            center.z()
        );

        // Both contacts must lie ON the torus surface — verify via
        // project_point (frame-orientation-agnostic).
        let want_plate = Point3::new(expected_major, 0.0, 0.0);
        let r_plus_r = big_r + r_fillet;
        let want_sphere = Point3::new(
            expected_major * big_r / r_plus_r,
            0.0,
            r_fillet * (big_r + h_real) / r_plus_r,
        );
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_plate);
        let on_torus_plate = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_s, v_s) = ParametricSurface::project_point(&torus, want_sphere);
        let on_torus_sphere = ParametricSurface::evaluate(&torus, u_s, v_s);
        assert!(
            (on_torus_plate - want_plate).length() < 1e-9,
            "plate contact must lie on torus: project→eval gave {on_torus_plate:?}, want {want_plate:?}"
        );
        assert!(
            (on_torus_sphere - want_sphere).length() < 1e-9,
            "sphere contact must lie on torus: project→eval gave {on_torus_sphere:?}, want {want_sphere:?}"
        );

        // Sanity-check: sphere contact point should also lie on the
        // sphere surface itself (distance R from center).
        let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, h_real)).length();
        assert!(
            (sphere_dist - big_r).abs() < 1e-9,
            "sphere contact must lie on sphere: distance from center = {sphere_dist}, want {big_r}"
        );
    }

    /// Concave plane-sphere fillet: a spherical pocket carved out of a plate
    /// top — fillet rounds the rim where plate top meets pocket wall. Sphere
    /// face is REVERSED (its topological outward points INTO the pocket air,
    /// away from plate material).
    ///
    /// Geometry differs from the convex post-on-slab case in two ways:
    /// the rolling ball lives INSIDE the pocket (axially on the +n_p_inward
    /// side) and is INTERNALLY tangent to the sphere (`R − r` instead of
    /// `R + r`). The unified `signed_offset = −1` factor flips both.
    ///
    /// For sphere center at (0,0,−h)=(0,0,−1), R=2, plate top at z=0 with
    /// raw outward +z, n_p_inward = −z, h_signed = +1, r=0.3:
    ///   - R_t² = r_p² − 2r(R − h) = 3 − 0.6 = 2.4
    ///   - Torus center at z = −r (below plate, in pocket)
    ///   - Plate contact at radial R_t < r_p (closer to axis than the spine)
    ///   - Sphere contact at z = -0.176 (below plate, on the LOWER
    ///     hemisphere where the pocket face actually exists — confirms
    ///     internal tangency lands on the right portion of sphere)
    #[test]
    fn plane_sphere_fillet_concave_emits_torus_with_smaller_major() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let h_real: f64 = 1.0;
        let r_fillet: f64 = 0.3;
        let r_p_sq = big_r * big_r - h_real * h_real;
        let r_p = r_p_sq.sqrt();

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Sphere centered BELOW plate (pocket); face REVERSED.
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(sphere.clone()),
        ));

        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);
        let result = plane_sphere_fillet(
            n_p_inward,
            0.0,
            &sphere,
            &spine,
            &topo,
            r_fillet,
            face_plate,
            face_sphere,
        )
        .unwrap()
        .expect("concave plane-sphere fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let expected_major_sq = r_p_sq - 2.0 * r_fillet * (big_r - h_real);
        let expected_major = expected_major_sq.sqrt();
        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "concave torus major should be √(r_p² − 2r(R−h)) = {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            torus.major_radius() < r_p,
            "concave torus major must be smaller than spine radius (plate contact moves INWARD), got {} vs r_p={r_p}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "torus minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center at z = -r (below plate, inside pocket air).
        let center = torus.center();
        assert!(
            (center.z() - (-r_fillet)).abs() < 1e-12,
            "concave torus center z should be −r_fillet = {}, got {}",
            -r_fillet,
            center.z()
        );

        // Sphere contact must land on the LOWER hemisphere (z<0) — the
        // actual pocket face. If we'd applied the convex external-tangency
        // formula by mistake, contact would end up on the upper hemisphere
        // (z>0) where there's no face.
        let denom = big_r - r_fillet;
        let want_sphere = Point3::new(
            expected_major * big_r / denom,
            0.0,
            -r_fillet * (big_r - h_real) / denom,
        );
        assert!(
            want_sphere.z() < 0.0,
            "concave sphere contact must be on lower hemisphere (z<0), got z={}",
            want_sphere.z()
        );
        let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, -h_real)).length();
        assert!(
            (sphere_dist - big_r).abs() < 1e-9,
            "sphere contact must lie on sphere: distance from center = {sphere_dist}, want {big_r}"
        );

        // Verify both contacts land on the torus surface.
        let want_plate = Point3::new(expected_major, 0.0, 0.0);
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_plate);
        let on_torus_plate = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_s, v_s) = ParametricSurface::project_point(&torus, want_sphere);
        let on_torus_sphere = ParametricSurface::evaluate(&torus, u_s, v_s);
        assert!(
            (on_torus_plate - want_plate).length() < 1e-9,
            "plate contact must lie on torus: project→eval gave {on_torus_plate:?}, want {want_plate:?}"
        );
        assert!(
            (on_torus_sphere - want_sphere).length() < 1e-9,
            "sphere contact must lie on torus: project→eval gave {on_torus_sphere:?}, want {want_sphere:?}"
        );
    }

    /// Concave plane-sphere fillet rejects radii past the spindle bound,
    /// where `major² = r_p² − 2r(R−h)` would shrink below `r²`. Convex
    /// must still accept those same radii (its `+2r(R−h)` term grows).
    #[test]
    fn plane_sphere_fillet_concave_rejects_spindle_radius() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let h_real: f64 = 1.0;
        let r_p_sq = big_r * big_r - h_real * h_real;
        let r_p = r_p_sq.sqrt();

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(sphere.clone()),
        ));

        let n_p_inward = Vec3::new(0.0, 0.0, -1.0);

        // Concave spindle threshold: solving r² + 2r(R−h) > r_p² for the
        // positive root gives r > √((R−h)² + r_p²) − (R−h).
        // For R=2, r_p²=3, R−h=1: r > √(1+3)−1 = 1. So r=1.5 must reject.
        let too_big = 1.5;
        let result = plane_sphere_fillet(
            n_p_inward,
            0.0,
            &sphere,
            &spine,
            &topo,
            too_big,
            face_plate,
            face_sphere,
        )
        .unwrap();
        assert!(
            result.is_none(),
            "concave fillet at r={too_big} should reject (spindle / R_t < minor)"
        );

        // But convex at the same r is still fine: R_t² = r_p² + 2r·3 = 3 + 9 = 12, R_t ≈ 3.46 > r.
        // Build a mirror topology with face NOT reversed.
        let mut topo2 = Topology::new();
        let v2 = topo2.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle2 =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid2 = topo2.add_edge(Edge::new(v2, v2, EdgeCurve::Circle(circle2)));
        let spine2 = Spine::from_single_edge(&topo2, eid2).unwrap();
        let w1b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, true)], true).unwrap());
        let face_plate2 = topo2.add_face(Face::new(
            w1b,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let sphere2 = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
        let w2b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, false)], true).unwrap());
        let face_sphere2 =
            topo2.add_face(Face::new(w2b, vec![], FaceSurface::Sphere(sphere2.clone())));
        let result_convex = plane_sphere_fillet(
            n_p_inward,
            0.0,
            &sphere2,
            &spine2,
            &topo2,
            too_big,
            face_plate2,
            face_sphere2,
        )
        .unwrap();
        assert!(
            result_convex.is_some(),
            "convex fillet at the same r={too_big} should still succeed"
        );
    }

    /// Convex plane-sphere chamfer: a sphere intersecting a plate from
    /// above (post-on-slab). The chamfer cuts the rim with a flat conical
    /// slice tangent to plate at radial `r_p+d1` and to the sphere at
    /// arc-length `d2` along the meridian toward the apex.
    ///
    /// For sphere center at (0, 0, h)=(0, 0, 1), R=2, plate top at z=0
    /// with raw outward +z (chamfer dispatcher uses raw, no orient), and
    /// symmetric d1=d2=0.3:
    ///   - δ = d2/R = 0.15
    ///   - Sphere contact at radial r_p·cos δ + h·sin δ ≈ √3·0.989 + 1·0.149 ≈ 1.862
    ///     and z = h(1−cos δ) + r_p·sin δ ≈ 0.011 + √3·0.149 ≈ 0.270
    ///   - Plate contact at radial r_p+d1 = √3+0.3 ≈ 2.032 (z=0)
    ///   - Cone half-angle β where tan β = −Δz/Δr = 0.270/0.170 ≈ 1.589
    ///     (β ≈ 57.8°; not the small-δ Taylor limit which would give
    ///     atan(r_p/(R−h)) = atan(√3/1) = 60°)
    ///   - Cone apex at z = (r_p+d1)·tan β ≈ 3.230
    #[test]
    fn plane_sphere_chamfer_convex_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let h_real: f64 = 1.0;
        let d: f64 = 0.3;
        let r_p_sq = big_r * big_r - h_real * h_real;
        let r_p = r_p_sq.sqrt();

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        // Chamfer convention: dispatcher passes the surface's RAW normal
        // (no orient). Plate slab top has raw outward = +z.
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, h_real), big_r).unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(sphere.clone())));

        let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
        let result = plane_sphere_chamfer(
            n_p_inward,
            0.0,
            &sphere,
            &spine,
            &topo,
            d,
            d,
            face_plate,
            face_sphere,
        )
        .unwrap()
        .expect("convex plane-sphere chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts.
        let delta = d / big_r;
        let (sin_d, cos_d) = delta.sin_cos();
        let sphere_radial_pred = r_p * cos_d + h_real * sin_d;
        let sphere_axial_pred = h_real * (1.0 - cos_d) + r_p * sin_d;

        // Predicted cone half-angle: tan β = −Δz/Δr.
        let delta_r = sphere_radial_pred - (r_p + d);
        let delta_z = sphere_axial_pred;
        assert!(delta_r < 0.0, "Δr should be negative (cone narrows up)");
        let expected_beta = (-delta_z / delta_r).atan();
        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
            "chamfer cone half-angle should be atan(−Δz/Δr) = {expected_beta}, got {}",
            chamfer_cone.half_angle()
        );

        // Apex position: on +z axis at z = (r_p+d) · tan β.
        let expected_apex_z = (r_p + d) * expected_beta.tan();
        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );

        // Cone axis points down (-z) — generator opens away from apex
        // toward the chamfer line.
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
            "chamfer cone axis should be -z, got {axis:?}"
        );

        // Both contact points must lie on the chamfer cone surface.
        let want_plate = Point3::new(r_p + d, 0.0, 0.0);
        let want_sphere = Point3::new(sphere_radial_pred, 0.0, sphere_axial_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
        let on_cone_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_s, v_s) = ParametricSurface::project_point(&chamfer_cone, want_sphere);
        let on_cone_sphere = ParametricSurface::evaluate(&chamfer_cone, u_s, v_s);
        assert!(
            (on_cone_plate - want_plate).length() < 1e-9,
            "plate contact must lie on chamfer cone: project→eval gave {on_cone_plate:?}, want {want_plate:?}"
        );
        assert!(
            (on_cone_sphere - want_sphere).length() < 1e-9,
            "sphere contact must lie on chamfer cone: project→eval gave {on_cone_sphere:?}, want {want_sphere:?}"
        );

        // Sanity: sphere contact lies on the actual sphere (distance R).
        let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, h_real)).length();
        assert!(
            (sphere_dist - big_r).abs() < 1e-9,
            "sphere contact must lie on sphere: distance = {sphere_dist}, want {big_r}"
        );
    }

    /// Concave plane-sphere chamfer: a spherical pocket carved out of a
    /// plate top — chamfer rounds the rim where plate meets pocket wall.
    /// Sphere face is REVERSED. The chamfer surface is a cone with apex
    /// BELOW the plate (in pocket air, axis pointing upward through the
    /// chamfer back to the plate level).
    ///
    /// For sphere at (0, 0, −h)=(0, 0, −1) (below plate), R=2, plate top
    /// raw outward +z, n_p_inward=+z (chamfer convention), face reversed,
    /// d=0.3:
    ///   - δ = d/R = 0.15
    ///   - sphere_radial = r_p·cos δ + (−1)(−1)·sin δ ≈ 1.862 (same as convex)
    ///   - sphere_axial  = (−1)(1−cos δ) + (−1)·r_p·sin δ ≈ −0.270 (BELOW plate)
    ///   - Δr = −0.170, Δz = −0.270
    ///   - z_apex = −(r_p+d)·Δz/Δr ≈ −3.227 (apex BELOW plate)
    ///   - chamfer axis = +z (opens upward toward contacts)
    ///   - cone β = atan(|z_apex|/(r_p+d)) ≈ 57.8° (same magnitude as convex)
    #[test]
    fn plane_sphere_chamfer_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let h_real: f64 = 1.0;
        let d: f64 = 0.3;
        let r_p_sq = big_r * big_r - h_real * h_real;
        let r_p = r_p_sq.sqrt();

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Sphere centered BELOW plate (pocket); face REVERSED.
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, -h_real), big_r).unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(sphere.clone()),
        ));

        let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
        let result = plane_sphere_chamfer(
            n_p_inward,
            0.0,
            &sphere,
            &spine,
            &topo,
            d,
            d,
            face_plate,
            face_sphere,
        )
        .unwrap()
        .expect("concave plane-sphere chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts (concave: signed_offset = -1, h_signed = -1).
        let h_signed = -h_real;
        let signed_offset = -1.0_f64;
        let delta = d / big_r;
        let (sin_d, cos_d) = delta.sin_cos();
        let sphere_radial_pred = r_p * cos_d + signed_offset * h_signed * sin_d;
        let sphere_axial_pred = h_signed * (1.0 - cos_d) + signed_offset * r_p * sin_d;

        let delta_r = sphere_radial_pred - (r_p + d);
        let delta_z = sphere_axial_pred;
        let expected_z_apex = -(r_p + d) * delta_z / delta_r;
        let expected_beta = (expected_z_apex.abs() / (r_p + d)).atan();
        assert!(
            expected_z_apex < 0.0,
            "concave: z_apex should be negative (apex below plate), got {expected_z_apex}"
        );
        assert!(
            sphere_axial_pred < 0.0,
            "concave: sphere contact must be below plate (z<0), got {sphere_axial_pred}"
        );

        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
            "chamfer cone half-angle should be {expected_beta}, got {}",
            chamfer_cone.half_angle()
        );

        // Apex BELOW plate.
        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_z_apex).abs() < 1e-9,
            "apex z = {}, expected {expected_z_apex}",
            apex.z()
        );

        // Cone axis points UP (+z) for concave (toward the contacts above
        // the apex) — opposite to convex.
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "concave chamfer cone axis should be +z, got {axis:?}"
        );

        // Both contacts lie on the chamfer cone.
        let want_plate = Point3::new(r_p + d, 0.0, 0.0);
        let want_sphere = Point3::new(sphere_radial_pred, 0.0, sphere_axial_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
        let on_cone_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_s, v_s) = ParametricSurface::project_point(&chamfer_cone, want_sphere);
        let on_cone_sphere = ParametricSurface::evaluate(&chamfer_cone, u_s, v_s);
        assert!(
            (on_cone_plate - want_plate).length() < 1e-9,
            "plate contact must lie on chamfer cone: gave {on_cone_plate:?}, want {want_plate:?}"
        );
        assert!(
            (on_cone_sphere - want_sphere).length() < 1e-9,
            "sphere contact must lie on chamfer cone: gave {on_cone_sphere:?}, want {want_sphere:?}"
        );

        // Sphere contact must lie on the actual sphere face — i.e. on the
        // LOWER hemisphere where the pocket boundary lives. Convex would
        // have placed it on the upper hemisphere, but with signed_offset
        // = -1 the contact lands at z<0 here.
        let sphere_dist = (want_sphere - Point3::new(0.0, 0.0, -h_real)).length();
        assert!(
            (sphere_dist - big_r).abs() < 1e-9,
            "sphere contact must lie on sphere: distance = {sphere_dist}, want {big_r}"
        );
    }

    /// Sphere-sphere convex fillet: two intersecting spheres meeting along
    /// a circular spine, rolling-ball blend traces a torus around the
    /// line connecting their centers.
    ///
    /// For sphere1 at origin (R=2), sphere2 at (3, 0, 0) (R=2.5),
    /// D=3, both faces NOT reversed:
    ///   a₀ = (4 − 6.25 + 9) / 6 = 6.75/6 = 1.125
    ///   r_p² = 4 − 1.265625 = 2.734375, r_p ≈ 1.654
    ///   For r=0.4:
    ///     δ = (2 − 2.5)/3 = −1/6
    ///     a_ball = 1.125 + 0.4·(−1/6) ≈ 1.0583
    ///     R_t² = (2.4)² − (1.0583)² = 5.76 − 1.12 = 4.64
    ///     R_t ≈ 2.154
    #[test]
    fn sphere_sphere_fillet_convex_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;

        // Place spheres along +z (brepkit's `SphericalSurface::new` uses
        // Frame3::from_normal with default z-axis = +z, which our
        // axisymmetry guard requires to be aligned with the C1→C2 line).
        // Sphere 1 at origin, sphere 2 at (0, 0, D); spine in the
        // z = a0 plane with axis +z.
        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r1 * big_r1 - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2.clone())));

        let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("convex sphere-sphere fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted torus parameters.
        let big_delta = (big_r1 - big_r2) / big_d;
        let a_ball = a0 + r_fillet * big_delta;
        let expected_major = ((big_r1 + r_fillet) * (big_r1 + r_fillet) - a_ball * a_ball).sqrt();
        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "major should be √((R1+r)²−a_ball²)={expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center sits on the C1-C2 axis at the rolling-ball axial
        // position. C1=origin, axis=+z, so center=(0, 0, a_ball).
        let center = torus.center();
        assert!(
            center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
            "torus center should be on +z axis, got {center:?}"
        );
        assert!(
            (center.z() - a_ball).abs() < 1e-12,
            "torus center z should be a_ball={a_ball}, got {}",
            center.z()
        );

        // Predict and verify both 3D contact points lie ON the torus.
        // sphere1 contact axial from C1 = R1·a_ball/(R1+r).
        let s1_axial = big_r1 * a_ball / (big_r1 + r_fillet);
        let s1_radial = big_r1 * expected_major / (big_r1 + r_fillet);
        let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
        // sphere2 contact axial from C2 = R2·(a_ball − D)/(R2+r).
        // World z = D + that.
        let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / (big_r2 + r_fillet);
        let s2_radial = big_r2 * expected_major / (big_r2 + r_fillet);
        let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
        let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
        let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_s1 - want_s1).length() < 1e-9,
            "sphere1 contact must lie on torus: gave {on_torus_s1:?}, want {want_s1:?}"
        );
        assert!(
            (on_torus_s2 - want_s2).length() < 1e-9,
            "sphere2 contact must lie on torus: gave {on_torus_s2:?}, want {want_s2:?}"
        );

        // And both contact points lie on their respective spheres.
        let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
        let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
        );
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
        );
    }

    /// Sphere-sphere both-concave fillet: two intersecting spherical
    /// cavities (e.g. two overlapping ball-shaped voids carved into a
    /// solid). Both faces REVERSED ⇒ rolling ball internally tangent to
    /// both spheres; effective radii Q1=R1−r, Q2=R2−r.
    ///
    /// For sphere1 at origin (R=2), sphere2 at (0,0,3) (R=2.5), D=3,
    /// both faces REVERSED, r=0.4:
    ///   Q1 = 1.6, Q2 = 2.1
    ///   a_ball = (Q1²−Q2²+D²)/(2D) = 7.15/6 ≈ 1.192
    ///   R_t²   = Q1²−a_ball² = 2.56−1.421 ≈ 1.139
    ///   R_t    ≈ 1.067 (smaller than convex case where R_t ≈ 2.154,
    ///                    confirming the internal-tangency reduction)
    #[test]
    fn sphere_sphere_fillet_both_concave_emits_smaller_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;

        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r1 * big_r1 - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(s1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(s2.clone()),
        ));

        let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("both-concave sphere-sphere fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let q1 = big_r1 - r_fillet;
        let q2 = big_r2 - r_fillet;
        let a_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
        let expected_major = (q1 * q1 - a_ball * a_ball).sqrt();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "major should be √(Q1²−a_ball²)={expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should equal fillet radius {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Crucial check: concave torus is SMALLER than the convex
        // counterpart at the same r — internal vs external tangency.
        // Compute the convex major for reference.
        let q1_conv = big_r1 + r_fillet;
        let q2_conv = big_r2 + r_fillet;
        let a_ball_conv = (q1_conv * q1_conv - q2_conv * q2_conv + big_d * big_d) / (2.0 * big_d);
        let convex_major = (q1_conv * q1_conv - a_ball_conv * a_ball_conv).sqrt();
        assert!(
            torus.major_radius() < convex_major,
            "concave major ({}) must be smaller than convex major ({convex_major}) at same r",
            torus.major_radius()
        );

        // Verify each contact lies on its respective sphere.
        let s1_axial = big_r1 * a_ball / q1;
        let s1_radial = big_r1 * expected_major / q1;
        let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
        let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / q2;
        let s2_radial = big_r2 * expected_major / q2;
        let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

        let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
        let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
        );
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
        );

        // And both lie on the torus.
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
        let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
        let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_s1 - want_s1).length() < 1e-9,
            "sphere1 contact must lie on torus: {on_torus_s1:?} vs {want_s1:?}"
        );
        assert!(
            (on_torus_s2 - want_s2).length() < 1e-9,
            "sphere2 contact must lie on torus: {on_torus_s2:?} vs {want_s2:?}"
        );
    }

    /// Sphere-sphere fillet rejects radii that collapse `Qi = Ri − r` to
    /// zero in the concave case (rolling ball would coincide with sphere
    /// center). Convex at the same r is still valid.
    #[test]
    fn sphere_sphere_fillet_concave_rejects_collapsing_q() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r: f64 = 2.0;
        let big_d: f64 = 3.0;
        // r ≥ R1 ⇒ Q1 ≤ 0 in the concave case.
        let r_too_big = 2.1;

        let a0 = (big_r * big_r - big_r * big_r + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r * big_r - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(s1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(s2.clone()),
        ));

        let result =
            sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_too_big, face1, face2).unwrap();
        assert!(
            result.is_none(),
            "concave fillet at r≥R should reject (Qi collapses to ≤ 0)"
        );

        // Convex at the same r is still fine.
        let mut topo2 = Topology::new();
        let v2 = topo2.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let circle2 =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid2 = topo2.add_edge(Edge::new(v2, v2, EdgeCurve::Circle(circle2)));
        let spine2 = Spine::from_single_edge(&topo2, eid2).unwrap();
        let w1b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, true)], true).unwrap());
        let face1b = topo2.add_face(Face::new(w1b, vec![], FaceSurface::Sphere(s1.clone())));
        let w2b = topo2.add_wire(Wire::new(vec![OrientedEdge::new(eid2, false)], true).unwrap());
        let face2b = topo2.add_face(Face::new(w2b, vec![], FaceSurface::Sphere(s2.clone())));
        let result_convex =
            sphere_sphere_fillet(&s1, &s2, &spine2, &topo2, r_too_big, face1b, face2b).unwrap();
        assert!(
            result_convex.is_some(),
            "convex fillet at the same r={r_too_big} should still succeed"
        );
    }

    /// Sphere-sphere mixed-convexity fillet: sphere1 face NOT reversed
    /// (convex; ball externally tangent, `Q1 = R1 + r`); sphere2 face
    /// REVERSED (concave; ball internally tangent, `Q2 = R2 − r`).
    /// Geometrically this is the "post emerging through a spherical
    /// cavity" configuration — uncommon but the Q-substitution handles
    /// it just like the symmetric cases.
    ///
    /// For R1=2, R2=2.5, D=3, sphere1 NOT reversed, sphere2 REVERSED,
    /// r=0.4:
    ///   Q1 = 2.4, Q2 = 2.1
    ///   a_ball = (5.76 − 4.41 + 9)/6 = 10.35/6 = 1.725
    ///   R_t² = 5.76 − 2.976 = 2.784, R_t ≈ 1.668
    /// (Sandwiched between the convex-convex R_t ≈ 2.154 and the
    /// concave-concave R_t ≈ 1.067, which is what we'd expect.)
    #[test]
    fn sphere_sphere_fillet_mixed_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;

        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r1 * big_r1 - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        // Sphere 1: NOT reversed (convex, external tangency).
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        // Sphere 2: REVERSED (concave, internal tangency).
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(s2.clone()),
        ));

        let result = sphere_sphere_fillet(&s1, &s2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("mixed sphere-sphere fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let q1 = big_r1 + r_fillet; // sphere1 convex
        let q2 = big_r2 - r_fillet; // sphere2 concave
        let a_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
        let expected_major = (q1 * q1 - a_ball * a_ball).sqrt();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "mixed major should be √(Q1²−a_ball²)={expected_major}, got {}",
            torus.major_radius()
        );

        // Check ordering: mixed major must sit BETWEEN convex/convex and
        // concave/concave at the same r — confirms the Q-substitution
        // produces the right interpolation.
        let q1_cc = big_r1 + r_fillet;
        let q2_cc = big_r2 + r_fillet;
        let a_ball_cc = (q1_cc * q1_cc - q2_cc * q2_cc + big_d * big_d) / (2.0 * big_d);
        let convex_convex_major = (q1_cc * q1_cc - a_ball_cc * a_ball_cc).sqrt();
        let q1_kk = big_r1 - r_fillet;
        let q2_kk = big_r2 - r_fillet;
        let a_ball_kk = (q1_kk * q1_kk - q2_kk * q2_kk + big_d * big_d) / (2.0 * big_d);
        let concave_concave_major = (q1_kk * q1_kk - a_ball_kk * a_ball_kk).sqrt();
        assert!(
            torus.major_radius() < convex_convex_major
                && torus.major_radius() > concave_concave_major,
            "mixed major ({}) should sit between concave-concave ({concave_concave_major}) and convex-convex ({convex_convex_major})",
            torus.major_radius()
        );

        // Both contacts on respective spheres.
        let s1_axial = big_r1 * a_ball / q1;
        let s1_radial = big_r1 * expected_major / q1;
        let want_s1 = Point3::new(s1_radial, 0.0, s1_axial);
        let s2_axial_from_c2 = big_r2 * (a_ball - big_d) / q2;
        let s2_radial = big_r2 * expected_major / q2;
        let want_s2 = Point3::new(s2_radial, 0.0, big_d + s2_axial_from_c2);

        let dist_s1 = (want_s1 - Point3::new(0.0, 0.0, 0.0)).length();
        let dist_s2 = (want_s2 - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "sphere1 contact must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
        );
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "sphere2 contact must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
        );

        // Both on torus.
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_s1);
        let on_torus_s1 = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_s2);
        let on_torus_s2 = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_s1 - want_s1).length() < 1e-9,
            "sphere1 contact must lie on torus: {on_torus_s1:?} vs {want_s1:?}"
        );
        assert!(
            (on_torus_s2 - want_s2).length() < 1e-9,
            "sphere2 contact must lie on torus: {on_torus_s2:?} vs {want_s2:?}"
        );
    }

    /// Sphere-sphere convex chamfer: two intersecting spheres meeting
    /// along a circular spine; chamfer surface is an axisymmetric cone
    /// connecting both sphere-side contact circles.
    ///
    /// For sphere1 at origin (R=2), sphere2 at (0, 0, 3) (R=2.5), D=3,
    /// both faces NOT reversed, symmetric d=0.4:
    ///   - δ1 = 0.2, δ2 = 0.16
    ///   - contact1 at radial r_p·cos δ1 + a₀·sin δ1 ≈ 1.844,
    ///     z ≈ a₀·cos δ1 − r_p·sin δ1 ≈ 0.774 (z<a₀: below spine)
    ///   - contact2 at radial r_p·cos δ2 + (D−a₀)·sin δ2 ≈ 1.932,
    ///     z ≈ D − (D−a₀)·cos δ2 + r_p·sin δ2 ≈ 1.413 (z>a₀: above spine)
    ///   - Cone apex below both contacts on the +z axis (~z=−12.6)
    ///   - Cone axis = +z (opens upward toward both contacts)
    #[test]
    fn sphere_sphere_chamfer_convex_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;

        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r1 * big_r1 - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2.clone())));

        let result = sphere_sphere_chamfer(&s1, &s2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("convex sphere-sphere chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts.
        let delta1 = d / big_r1;
        let delta2 = d / big_r2;
        let (sin1, cos1) = delta1.sin_cos();
        let (sin2, cos2) = delta2.sin_cos();
        let p1_r = r_p * cos1 + a0 * sin1;
        let p1_z = a0 * cos1 - r_p * sin1;
        let p2_r = r_p * cos2 + (big_d - a0) * sin2;
        let p2_z = big_d - (big_d - a0) * cos2 + r_p * sin2;

        // Contact1 below spine, contact2 above spine — characteristic
        // of the convex-convex case (faces extend AWAY from each other).
        assert!(p1_z < a0, "convex contact1 should be below spine z=a0");
        assert!(p2_z > a0, "convex contact2 should be above spine z=a0");

        // Predicted apex from line P1-P2 extrapolation to r=0.
        let dr = p2_r - p1_r;
        let dz = p2_z - p1_z;
        let expected_apex_z = p1_z - p1_r * dz / dr;

        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );

        // Cone axis: contacts are above apex (mid_z > apex_z), so axis = +z.
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "convex chamfer cone axis should be +z, got {axis:?}"
        );

        // Both contacts must lie on the chamfer cone.
        let want_p1 = Point3::new(p1_r, 0.0, p1_z);
        let want_p2 = Point3::new(p2_r, 0.0, p2_z);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_p1);
        let on_cone_p1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_p2);
        let on_cone_p2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_p1 - want_p1).length() < 1e-9,
            "contact1 must lie on chamfer cone: {on_cone_p1:?} vs {want_p1:?}"
        );
        assert!(
            (on_cone_p2 - want_p2).length() < 1e-9,
            "contact2 must lie on chamfer cone: {on_cone_p2:?} vs {want_p2:?}"
        );

        // Both contacts also lie on their respective spheres.
        let dist_s1 = (want_p1 - Point3::new(0.0, 0.0, 0.0)).length();
        let dist_s2 = (want_p2 - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "contact1 must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
        );
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "contact2 must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
        );
    }

    /// Sphere-cylinder convex fillet: a sphere primitive fused to a
    /// cylinder primitive along their shared axis. The intersection is
    /// a pair of circles at axial offset ±h_s = ±√(R_s²−r_c²) from the
    /// sphere center; we fillet the +h_s spine.
    ///
    /// For sphere at origin (R=3), cylinder axis +z through origin
    /// (r_c=2), both faces NOT reversed, r=0.4:
    ///   - h_s = √(9−4) = √5 ≈ 2.236
    ///   - Q_s = 3.4, Q_c = 2.4
    ///   - a_ball = √(Q_s² − Q_c²) = √(11.56 − 5.76) = √5.8 ≈ 2.408
    ///   - major = Q_c = 2.4
    #[test]
    fn sphere_cylinder_fillet_convex_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let r_fillet: f64 = 0.4;
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();

        // Spine: circle at z = +h_s, radius r_c, axis +z.
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())));

        let result =
            sphere_cylinder_fillet(&sph, &cyl, &spine, &topo, r_fillet, face_sphere, face_cyl)
                .unwrap()
                .expect("convex sphere-cylinder fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let q_s = big_r_s + r_fillet;
        let q_c = r_c + r_fillet;
        let expected_major = q_c;
        let expected_a_ball = (q_s * q_s - q_c * q_c).sqrt();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "major should be Q_c = {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should be r = {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center on +z axis at z = a_ball (positive since spine
        // is at +h_s).
        let center = torus.center();
        assert!(
            center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
            "torus center should be on z-axis, got {center:?}"
        );
        assert!(
            (center.z() - expected_a_ball).abs() < 1e-12,
            "torus center z should be a_ball = {expected_a_ball}, got {}",
            center.z()
        );

        // Sphere contact radial = R_s · Q_c / Q_s, axial = R_s · a_ball / Q_s.
        let sph_axial = big_r_s * expected_a_ball / q_s;
        let sph_radial = big_r_s * q_c / q_s;
        let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
        // Cylinder contact at radial r_c, axial a_ball.
        let want_cyl = Point3::new(r_c, 0.0, expected_a_ball);

        // Both lie on torus.
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
        let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cyl);
        let on_torus_cyl = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_torus_cyl - want_cyl).length() < 1e-9,
            "cylinder contact must lie on torus: {on_torus_cyl:?} vs {want_cyl:?}"
        );

        // Both lie on their respective surfaces.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance = {dist_sph}, want R_s = {big_r_s}"
        );
        let dist_cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
        assert!(
            (dist_cyl_radial - r_c).abs() < 1e-9,
            "cylinder contact must lie on cylinder: radial = {dist_cyl_radial}, want r_c = {r_c}"
        );
    }

    /// Sphere-cylinder convex chamfer: a sphere primitive fused to a
    /// cylinder primitive along their shared axis. The chamfer surface
    /// is an axisymmetric cone connecting the sphere-side and
    /// cylinder-side contacts.
    ///
    /// For sphere at origin (R=3), cylinder axis +z (r_c=2), spine at
    /// z=+h_s=+√5≈2.236, both faces NOT reversed, symmetric d=0.4:
    ///   - δ = 0.4/3 ≈ 0.1333
    ///   - sphere contact at radial r_c·cos δ − h_s·sin δ ≈ 1.685,
    ///     z = h_s·cos δ + r_c·sin δ ≈ 2.482
    ///   - cyl contact at radial r_c=2.0, z = h_s − d ≈ 1.836
    ///   - Δr ≈ 0.315, Δz ≈ −0.646
    ///   - apex z = z_sph − r_sph·Δz/Δr ≈ 5.94 (above contacts)
    #[test]
    fn sphere_cylinder_chamfer_convex_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let d: f64 = 0.4;
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();

        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl.clone())));

        let result =
            sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
                .unwrap()
                .expect("convex sphere-cylinder chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts.
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let r_sph_pred = r_c * cos_d - h_s * sin_d;
        let z_sph_pred = h_s * cos_d + r_c * sin_d;
        let r_cyl_pred = r_c;
        let z_cyl_pred = h_s - d;

        // Predicted apex from line extrapolation.
        let dr = r_cyl_pred - r_sph_pred;
        let dz = z_cyl_pred - z_sph_pred;
        let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;

        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );
        assert!(
            expected_apex_z > z_sph_pred,
            "apex should be above the contacts (mid z < apex z), got apex_z={expected_apex_z}"
        );

        // Cone axis: contacts are below apex, so axis points -z.
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
            "convex chamfer cone axis should be -z (apex above), got {axis:?}"
        );

        // Both contacts on the chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cyl = Point3::new(r_cyl_pred, 0.0, z_cyl_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
        let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cyl - want_cyl).length() < 1e-9,
            "cylinder contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
        );

        // Sphere contact lies on sphere; cylinder contact has correct radius.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
        );
        let dist_cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
        assert!(
            (dist_cyl_radial - r_c).abs() < 1e-9,
            "cylinder contact must have radial r_c: got {dist_cyl_radial}, want {r_c}"
        );
    }

    /// Sphere-sphere both-concave chamfer: two intersecting spherical
    /// cavities. Both `s1 = s2 = −1` flip the meridian arms; each sphere
    /// goes TOWARD the other (instead of AWAY in the convex case).
    ///
    /// The implementation already handled per-sphere signed_offsets; this
    /// test confirms it works for the symmetric concave case.
    ///
    /// For R1=2, R2=2.5, D=3, both faces REVERSED, symmetric d=0.4:
    ///   - δ1 = 0.2, δ2 = 0.16, a₀ ≈ 1.125, r_p ≈ 1.654
    ///   - p1_r = r_p·cos δ1 − a₀·sin δ1 ≈ 1.398
    ///     (less than convex r_p+a₀_term ≈ 1.844 — going toward axis)
    ///   - p1_z = a₀·cos δ1 + r_p·sin δ1 ≈ 1.432
    ///     (ABOVE spine z = a₀, opposite convex which had below)
    ///   - p2_z = D − (D−a₀)·cos δ2 − r_p·sin δ2 ≈ 0.886
    ///     (BELOW spine, opposite convex)
    #[test]
    fn sphere_sphere_chamfer_both_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;

        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p_sq = big_r1 * big_r1 - a0 * a0;
        let r_p = r_p_sq.sqrt();

        let s1 = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
        let s2 = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(s1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Sphere(s2.clone()),
        ));

        let result = sphere_sphere_chamfer(&s1, &s2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("both-concave sphere-sphere chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts with s1 = s2 = -1.
        let delta1 = d / big_r1;
        let delta2 = d / big_r2;
        let (sin1, cos1) = delta1.sin_cos();
        let (sin2, cos2) = delta2.sin_cos();
        let p1_r = r_p * cos1 - a0 * sin1; // s1 = -1 ⇒ -a0
        let p1_z = a0 * cos1 + r_p * sin1; // s1 = -1 ⇒ +r_p
        let p2_r = r_p * cos2 - (big_d - a0) * sin2; // s2 = -1
        let p2_z = big_d - (big_d - a0) * cos2 - r_p * sin2; // s2 = -1 ⇒ -r_p

        // Concave-concave: contact1 ABOVE spine (z > a₀), contact2 BELOW
        // spine (z < a₀) — opposite the convex-convex pattern.
        assert!(
            p1_z > a0,
            "concave contact1 should be above spine (z > a0): got {p1_z}"
        );
        assert!(
            p2_z < a0,
            "concave contact2 should be below spine (z < a0): got {p2_z}"
        );

        // Both contacts on chamfer cone.
        let want_p1 = Point3::new(p1_r, 0.0, p1_z);
        let want_p2 = Point3::new(p2_r, 0.0, p2_z);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_p1);
        let on_cone_p1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_p2);
        let on_cone_p2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_p1 - want_p1).length() < 1e-9,
            "concave contact1 must lie on chamfer cone: {on_cone_p1:?} vs {want_p1:?}"
        );
        assert!(
            (on_cone_p2 - want_p2).length() < 1e-9,
            "concave contact2 must lie on chamfer cone: {on_cone_p2:?} vs {want_p2:?}"
        );

        // Both contacts on respective spheres.
        let dist_s1 = (want_p1 - Point3::new(0.0, 0.0, 0.0)).length();
        let dist_s2 = (want_p2 - Point3::new(0.0, 0.0, big_d)).length();
        assert!(
            (dist_s1 - big_r1).abs() < 1e-9,
            "contact1 must lie on sphere1: distance={dist_s1}, want R1={big_r1}"
        );
        assert!(
            (dist_s2 - big_r2).abs() < 1e-9,
            "contact2 must lie on sphere2: distance={dist_s2}, want R2={big_r2}"
        );
    }

    /// Sphere-sphere mixed-convexity chamfer: covers BOTH (s1=+1, s2=−1)
    /// and (s1=−1, s2=+1). Each is geometrically distinct from the
    /// symmetric cases (and from each other), with contacts on different
    /// cap arms.
    ///
    /// Verifies the chamfer via project_point on the EMITTED
    /// `result.stripe.contact{1,2}` curves (not on test-computed
    /// formulas, which would be tautological for the sphere-distance
    /// check since `r² + (z − Cz)² = R²` holds algebraically for any
    /// sign).
    #[test]
    fn sphere_sphere_chamfer_mixed_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::SphericalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let big_r1: f64 = 2.0;
        let big_r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;
        let a0 = (big_r1 * big_r1 - big_r2 * big_r2 + big_d * big_d) / (2.0 * big_d);
        let r_p = (big_r1 * big_r1 - a0 * a0).sqrt();

        // Run both mixed configurations (s1, s2) ∈ {(+,−), (−,+)} via a
        // closure parameterized by which face to reverse. For each:
        //   - emit the chamfer
        //   - extract the actual 3D contact point from the impl's
        //     `contact1`/`contact2` NURBS curves
        //   - assert that the EMITTED contact lies on the corresponding
        //     sphere (this DOES test the implementation, unlike
        //     sampling the test's own formula — see Greptile feedback
        //     on PR #604: r² + (z − Cz)² = R² holds algebraically for
        //     any sign of `s_i`, so a formula-derived contact would
        //     pass tautologically).
        let run_case = |reverse_s1: bool, reverse_s2: bool| {
            let mut topo = Topology::new();
            let s1_surf = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r1).unwrap();
            let s2_surf = SphericalSurface::new(Point3::new(0.0, 0.0, big_d), big_r2).unwrap();
            let spine_circle =
                Circle3D::new(Point3::new(0.0, 0.0, a0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
            let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, a0), 1e-7));
            let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
            let spine = Spine::from_single_edge(&topo, eid).unwrap();

            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
            let face1 = if reverse_s1 {
                topo.add_face(Face::new_reversed(
                    w1,
                    vec![],
                    FaceSurface::Sphere(s1_surf.clone()),
                ))
            } else {
                topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(s1_surf.clone())))
            };
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
            let face2 = if reverse_s2 {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Sphere(s2_surf.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Sphere(s2_surf.clone())))
            };

            let result =
                sphere_sphere_chamfer(&s1_surf, &s2_surf, &spine, &topo, d, d, face1, face2)
                    .unwrap()
                    .expect("mixed sphere-sphere chamfer should produce a stripe");

            // Verify emitted contact endpoints lie on their respective
            // spheres. Sample the EMITTED curve via
            // `evaluate(t_start)` rather than reading control points —
            // rational NURBS arcs have intermediate control points
            // OFF the curve, and even endpoint coverage couples the
            // test to construction details (degree, knot vector).
            let (t1_start, _) = result.stripe.contact1.domain();
            let c1_point = result.stripe.contact1.evaluate(t1_start);
            let dist_s1 = (c1_point - Point3::new(0.0, 0.0, 0.0)).length();
            assert!(
                (dist_s1 - big_r1).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): emitted contact1 must lie on sphere1: \
                 distance = {dist_s1}, want R1 = {big_r1}"
            );

            let (t2_start, _) = result.stripe.contact2.domain();
            let c2_point = result.stripe.contact2.evaluate(t2_start);
            let dist_s2 = (c2_point - Point3::new(0.0, 0.0, big_d)).length();
            assert!(
                (dist_s2 - big_r2).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): emitted contact2 must lie on sphere2: \
                 distance = {dist_s2}, want R2 = {big_r2}"
            );

            // Emitted surface is a Cone (the chamfer is a cone of
            // revolution about the C1-C2 axis).
            assert!(
                matches!(result.stripe.surface, FaceSurface::Cone(_)),
                "({reverse_s1}, {reverse_s2}): expected Cone, got {}",
                result.stripe.surface.type_tag()
            );

            // Both emitted contact points lie on the chamfer cone via
            // project_point round-trip.
            if let FaceSurface::Cone(ref cone) = result.stripe.surface {
                let (u_p, v_p) = ParametricSurface::project_point(cone, c1_point);
                let on_cone_p1 = ParametricSurface::evaluate(cone, u_p, v_p);
                assert!(
                    (on_cone_p1 - c1_point).length() < 1e-9,
                    "({reverse_s1}, {reverse_s2}): emitted contact1 must lie on chamfer cone"
                );
                let (u_q, v_q) = ParametricSurface::project_point(cone, c2_point);
                let on_cone_p2 = ParametricSurface::evaluate(cone, u_q, v_q);
                assert!(
                    (on_cone_p2 - c2_point).length() < 1e-9,
                    "({reverse_s1}, {reverse_s2}): emitted contact2 must lie on chamfer cone"
                );
            }
        };

        run_case(false, true); // (s1=+1, s2=-1)
        run_case(true, false); // (s1=-1, s2=+1)
    }

    /// Sphere-cylinder both-concave chamfer: spherical cavity + cylindrical
    /// hole-tool. Both `s_sph = s_cyl = −1` flip the meridian arms relative
    /// to convex.
    ///
    /// For R_s=3, r_c=2, +z spine, both faces REVERSED, d=0.4:
    ///   - r_sph = 2·cos δ + h_s·sin δ ≈ 2.280 (sphere goes AWAY from
    ///     spine on the +radial side, opposite the convex case)
    ///   - z_sph ≈ 1.950 (BELOW spine, opposite z_sph_convex ≈ 2.482)
    ///   - z_cyl = a_spine + d ≈ 2.636 (ABOVE spine, opposite convex)
    ///   - Apex z ≈ 7.54 (still above contacts, but at different position)
    #[test]
    fn sphere_cylinder_chamfer_both_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let d: f64 = 0.4;
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(sph.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl.clone()),
        ));

        let result =
            sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
                .unwrap()
                .expect("both-concave sphere-cylinder chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts with s_sph = s_cyl = -1.
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let r_sph_pred = r_c * cos_d + h_s * sin_d; // s_sph=-1 flips sphere arm
        let z_sph_pred = h_s * cos_d - r_c * sin_d; // s_sph=-1 flips axial offset
        let z_cyl_pred = h_s + d; // a_spine + d (s_cyl=-1 reverses cyl direction)

        // Concave sphere contact moved to OPPOSITE arm: now r > r_c (above spine
        // radially) AND z < spine_z (below spine, toward cyl side).
        assert!(
            r_sph_pred > r_c,
            "concave sphere contact should have r > r_c (opposite convex case): got {r_sph_pred}"
        );
        assert!(
            z_sph_pred < h_s,
            "concave sphere contact should have z < h_s (toward cyl side): got {z_sph_pred} vs h_s={h_s}"
        );
        assert!(
            z_cyl_pred > h_s,
            "concave cyl contact should have z > h_s (opposite the convex direction): got {z_cyl_pred} vs h_s={h_s}"
        );

        // Both contacts on chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
        let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "concave sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cyl - want_cyl).length() < 1e-9,
            "concave cyl contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
        );

        // Sphere contact at distance R_s from sphere center.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
        );
    }

    /// Sphere-cylinder mixed chamfer: sphere convex (s_sph=+1) + cyl
    /// concave (s_cyl=−1). Sphere contact lies on the AWAY-from-cyl
    /// cap (like convex-convex), cyl contact moves OPPOSITE direction
    /// from spine (like both-concave). Apex ends up BELOW the contacts
    /// (axis = +z), a distinct topological configuration from the
    /// symmetric cases where apex is always above.
    #[test]
    fn sphere_cylinder_chamfer_mixed_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let d: f64 = 0.4;
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Sphere convex (NOT reversed), cyl concave (REVERSED).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl.clone()),
        ));

        let result =
            sphere_cylinder_chamfer(&sph, &cyl, &spine, &topo, d, d, face_sphere, face_cyl)
                .unwrap()
                .expect("mixed sphere-cylinder chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts (s_sph=+1, s_cyl=-1).
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let r_sph_pred = r_c * cos_d - h_s * sin_d; // sphere convex arm
        let z_sph_pred = h_s * cos_d + r_c * sin_d;
        let z_cyl_pred = h_s + d; // cyl concave goes opposite

        let dr = r_c - r_sph_pred;
        let dz = z_cyl_pred - z_sph_pred;
        let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;

        // Mixed apex BELOW both contacts ⇒ axis = +z.
        assert!(
            expected_apex_z < z_sph_pred && expected_apex_z < z_cyl_pred,
            "mixed apex should be below both contacts, got apex_z={expected_apex_z}, \
             z_sph={z_sph_pred}, z_cyl={z_cyl_pred}"
        );
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "mixed chamfer cone axis should be +z (apex below contacts), got {axis:?}"
        );

        // Apex on z-axis at predicted position.
        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );

        // Both contacts on the chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
        let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "mixed sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cyl - want_cyl).length() < 1e-9,
            "mixed cyl contact must lie on chamfer cone: {on_cone_cyl:?} vs {want_cyl:?}"
        );
    }

    /// `try_analytic_chamfer` with (Cylinder, Sphere) ordering: the
    /// dispatcher must swap d1/d2 + face1/face2 then `swap_stripe_sides`
    /// so the caller-facing Stripe is consistent with the original
    /// surface ordering. Confirms the swap path is wired correctly.
    #[test]
    fn try_analytic_chamfer_cylinder_sphere_dispatch_swaps_correctly() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let d1_outer: f64 = 0.5; // distance on cylinder (the FIRST surface here)
        let d2_outer: f64 = 0.4; // distance on sphere (the SECOND surface here)
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();

        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Note the ordering: cylinder is FIRST, sphere is SECOND.
        let w_cyl = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1_cyl = topo.add_face(Face::new(w_cyl, vec![], FaceSurface::Cylinder(cyl.clone())));
        let w_sph = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2_sph = topo.add_face(Face::new(w_sph, vec![], FaceSurface::Sphere(sph.clone())));

        // Surface order: surf1 = cylinder, surf2 = sphere.
        let surf1 = FaceSurface::Cylinder(cyl);
        let surf2 = FaceSurface::Sphere(sph);
        let result = try_analytic_chamfer(
            &surf1, &surf2, &spine, &topo, d1_outer, d2_outer, face1_cyl, face2_sph,
        )
        .unwrap()
        .expect("dispatcher should produce a stripe for (cyl, sphere) chamfer");

        // After swap, the stripe's face1 should be the cylinder face (the
        // dispatcher's face1) and face2 should be the sphere face. The
        // direct (Sphere, Cylinder) call would have face1 = sphere; the
        // swap_stripe_sides flip restores the original ordering.
        assert_eq!(
            result.stripe.face1, face1_cyl,
            "stripe.face1 should match the dispatcher's first face (cylinder), \
             confirming swap_stripe_sides restored the caller-facing ordering"
        );
        assert_eq!(
            result.stripe.face2, face2_sph,
            "stripe.face2 should match the dispatcher's second face (sphere)"
        );

        // The d1/d2 swap means the cylinder gets `d1_outer` and the
        // sphere gets `d2_outer`. We can verify by predicting the
        // contacts and confirming they match: cyl axial offset from
        // spine = d1_outer (going INTO cyl).
        let z_cyl_pred = h_s - d1_outer;
        // sphere geodesic δ = d2_outer / R_s
        let delta = d2_outer / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let r_sph_pred = r_c * cos_d - h_s * sin_d;
        let z_sph_pred = h_s * cos_d + r_c * sin_d;

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };
        let want_cyl = Point3::new(r_c, 0.0, z_cyl_pred);
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_cyl);
        let on_cone_cyl = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_cyl - want_cyl).length() < 1e-9,
            "cylinder contact (using dispatcher's d1) must lie on cone: {on_cone_cyl:?} vs {want_cyl:?}"
        );
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "sphere contact (using dispatcher's d2) must lie on cone: {on_cone_sph:?} vs {want_sph:?}"
        );
    }

    /// Sphere-cone convex fillet: a sphere centered on the cone axis,
    /// fillet around one of the two sphere-cone intersection circles.
    ///
    /// For sphere at origin (R_s=3), cone apex at (0,0,−2) with axis +z
    /// and half-angle π/3, both faces NOT reversed, r=0.3:
    ///   - h_signed = +2 (sphere center is 2 units above apex along axis)
    ///   - β = π/3, cos β = 0.5, sin β = √3/2
    ///   - Spine z (from sphere center) = (−4 + √384)/8 ≈ 1.949 (the +z spine)
    ///   - Spine radial (z+h)·cot β ≈ 2.279
    ///   - A = r + h·cos β = 0.3 + 1.0 = 1.3
    ///   - c_root = −A·cos β + sin β·√((R_s+r)²−A²) ≈ 1.977 (matches +z spine sign)
    ///   - R_t = (r + (c+h)·cos β)/sin β ≈ 2.642
    #[test]
    fn sphere_cone_fillet_convex_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0; // apex 2 units below sphere center
        let beta: f64 = std::f64::consts::PI / 3.0;
        let r_fillet: f64 = 0.3;

        // Spine z (from sphere center) on the +z side.
        let cot_b = beta.cos() / beta.sin();
        let qa = 1.0 / (beta.sin() * beta.sin());
        let qb = 2.0 * h_signed * cot_b * cot_b;
        let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb * qb - 4.0 * qa * qc;
        let z_spine = (-qb + q_disc.sqrt()) / (2.0 * qa);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone.clone())));

        let result =
            sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
                .unwrap()
                .expect("convex sphere-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted torus parameters.
        let big_a = r_fillet + h_signed * beta.cos();
        let disc = (big_r_s + r_fillet) * (big_r_s + r_fillet) - big_a * big_a;
        let expected_z_b = -big_a * beta.cos() + beta.sin() * disc.sqrt();
        let expected_major = (r_fillet + (expected_z_b + h_signed) * beta.cos()) / beta.sin();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "major should be (r + (c+h)·cos β)/sin β = {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should be r = {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center on +z axis at z = expected_z_b.
        let center = torus.center();
        assert!(
            center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
            "torus center should be on z-axis, got {center:?}"
        );
        assert!(
            (center.z() - expected_z_b).abs() < 1e-9,
            "torus center z should be c_root = {expected_z_b}, got {}",
            center.z()
        );

        // Predicted contacts.
        let sph_axial = big_r_s * expected_z_b / (big_r_s + r_fillet);
        let sph_radial = big_r_s * expected_major / (big_r_s + r_fillet);
        let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
        let cone_axial = expected_z_b + r_fillet * beta.cos();
        let cone_radial = expected_major - r_fillet * beta.sin();
        let want_cone = Point3::new(cone_radial, 0.0, cone_axial);

        // Verify both contacts lie on the torus.
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
        let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cone);
        let on_torus_cone = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_torus_cone - want_cone).length() < 1e-9,
            "cone contact must lie on torus: {on_torus_cone:?} vs {want_cone:?}"
        );

        // Sphere contact on sphere.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
        );

        // Cone contact on cone: r = (z + h_signed) · cot β.
        let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
        assert!(
            (cone_radial - predicted_cone_radial).abs() < 1e-9,
            "cone contact must lie on cone surface: predicted radial {predicted_cone_radial}, got {cone_radial}"
        );
    }

    /// Sphere-cone fillet with cone face REVERSED (sphere convex, cone
    /// concave) — sphere fitting into a conical cavity. With s_sph=+1,
    /// s_cone=−1 the geometry uses internal cone tangency.
    ///
    /// For R_s=3, h_signed=+2, β=π/3, sphere face NOT reversed, cone
    /// face REVERSED, r=0.3:
    ///   - Q_s = 3.3, A = s_cone·r + h·cos β = −0.3 + 1.0 = 0.7
    ///   - disc = Q_s² − A² = 10.4, sqrt ≈ 3.225
    ///   - c_root_a = −A·cos β + sin β·sqrt ≈ 2.443 (closest to +z spine)
    ///   - R_t = (s_cone·r + (c+h)·cos β)/sin β ≈ 2.219
    ///   - Compare convex case: R_t was ≈ 2.642, so concave-cone gives
    ///     SMALLER major (consistent with the rolling ball being inside
    ///     the cone region instead of outside).
    #[test]
    fn sphere_cone_fillet_concave_cone_emits_smaller_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0;
        let beta: f64 = std::f64::consts::PI / 3.0;
        let r_fillet: f64 = 0.3;

        let cot_b = beta.cos() / beta.sin();
        let qa_q = 1.0 / (beta.sin() * beta.sin());
        let qb_q = 2.0 * h_signed * cot_b * cot_b;
        let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
        let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Sphere face NOT reversed (convex post); cone face REVERSED
        // (cone-shaped cavity).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone.clone()),
        ));

        let result =
            sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
                .unwrap()
                .expect("mixed sphere-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted torus parameters with s_sph=+1, s_cone=-1.
        let q_s = big_r_s + r_fillet; // s_sph = +1
        let big_a_pred = -r_fillet + h_signed * beta.cos(); // s_cone = -1
        let disc = q_s * q_s - big_a_pred * big_a_pred;
        let c_root_a = -big_a_pred * beta.cos() + beta.sin() * disc.sqrt();
        let c_root_b = -big_a_pred * beta.cos() - beta.sin() * disc.sqrt();
        let z_b = if (c_root_a - z_spine).abs() <= (c_root_b - z_spine).abs() {
            c_root_a
        } else {
            c_root_b
        };
        let expected_major = (-r_fillet + (z_b + h_signed) * beta.cos()) / beta.sin();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "concave-cone major should be {expected_major}, got {}",
            torus.major_radius()
        );

        // Sanity: concave-cone major < convex-cone major at same r.
        let big_a_convex = r_fillet + h_signed * beta.cos();
        let disc_convex = (big_r_s + r_fillet) * (big_r_s + r_fillet) - big_a_convex * big_a_convex;
        let c_convex = -big_a_convex * beta.cos() + beta.sin() * disc_convex.sqrt();
        let convex_major = (r_fillet + (c_convex + h_signed) * beta.cos()) / beta.sin();
        assert!(
            torus.major_radius() < convex_major,
            "concave-cone major ({}) should be smaller than convex ({convex_major})",
            torus.major_radius()
        );

        // Sphere contact at distance R_s from sphere center.
        let sph_axial = big_r_s * z_b / q_s;
        let sph_radial = big_r_s * expected_major / q_s;
        let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
        );

        // Cone contact at the predicted (axial, radial) — should lie on cone.
        // s_cone = -1 ⇒ axial offset NEGATIVE, radial offset POSITIVE.
        let cone_axial = z_b - r_fillet * beta.cos();
        let cone_radial = expected_major + r_fillet * beta.sin();
        let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
        assert!(
            (cone_radial - predicted_cone_radial).abs() < 1e-9,
            "cone contact must lie on cone: predicted radial {predicted_cone_radial}, got {cone_radial}"
        );

        // Both contacts on the torus.
        let want_cone = Point3::new(cone_radial, 0.0, cone_axial);
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
        let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cone);
        let on_torus_cone = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on torus: {on_torus_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_torus_cone - want_cone).length() < 1e-9,
            "cone contact must lie on torus: {on_torus_cone:?} vs {want_cone:?}"
        );
    }

    /// Sphere-cone fillet with BOTH faces reversed (sphere cavity inside
    /// a cone cavity). s_sph = s_cone = −1 so Q_s = R_s − r AND
    /// A = −r + h·cos β; both flips are independent and the test below
    /// pins down the (concave-sphere, concave-cone) branch that's
    /// distinct from the previously-tested (convex, concave-cone) case.
    ///
    /// For R_s=3, h=2, β=π/3, both faces REVERSED, r=0.3:
    ///   - Q_s = 2.7, A = 0.7
    ///   - disc = Q_s² − A² = 6.80, sqrt ≈ 2.608
    ///   - c_root closer to +z spine (z_spine ≈ 1.949) ≈ 1.908
    ///   - R_t = (s_cone·r + (c+h)·cos β)/sin β ≈ 1.910
    ///     (smaller than convex-convex 2.642 AND smaller than
    ///     concave-cone-only 2.219 — confirms BOTH flips compose)
    #[test]
    fn sphere_cone_fillet_both_concave_emits_smaller_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0;
        let beta: f64 = std::f64::consts::PI / 3.0;
        let r_fillet: f64 = 0.3;

        let cot_b = beta.cos() / beta.sin();
        let qa_q = 1.0 / (beta.sin() * beta.sin());
        let qb_q = 2.0 * h_signed * cot_b * cot_b;
        let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
        let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED (sphere cavity meets cone cavity).
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(sph.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone.clone()),
        ));

        let result =
            sphere_cone_fillet(&sph, &cone, &spine, &topo, r_fillet, face_sphere, face_cone)
                .unwrap()
                .expect("both-concave sphere-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted torus parameters with s_sph = s_cone = -1.
        let q_s = big_r_s - r_fillet;
        let big_a = -r_fillet + h_signed * beta.cos();
        let disc = q_s * q_s - big_a * big_a;
        let c_root_a = -big_a * beta.cos() + beta.sin() * disc.sqrt();
        let c_root_b = -big_a * beta.cos() - beta.sin() * disc.sqrt();
        let z_b = if (c_root_a - z_spine).abs() <= (c_root_b - z_spine).abs() {
            c_root_a
        } else {
            c_root_b
        };
        let expected_major = (-r_fillet + (z_b + h_signed) * beta.cos()) / beta.sin();

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "both-concave major should be {expected_major}, got {}",
            torus.major_radius()
        );

        // Sanity: both-concave major < convex-convex major < ALSO <
        // concave-cone-only — i.e. the two flips compose to give the
        // smallest torus.
        let convex_a = r_fillet + h_signed * beta.cos();
        let convex_disc = (big_r_s + r_fillet) * (big_r_s + r_fillet) - convex_a * convex_a;
        let convex_c = -convex_a * beta.cos() + beta.sin() * convex_disc.sqrt();
        let convex_major = (r_fillet + (convex_c + h_signed) * beta.cos()) / beta.sin();
        let concave_cone_a = -r_fillet + h_signed * beta.cos();
        let concave_cone_disc =
            (big_r_s + r_fillet) * (big_r_s + r_fillet) - concave_cone_a * concave_cone_a;
        let concave_cone_c = -concave_cone_a * beta.cos() + beta.sin() * concave_cone_disc.sqrt();
        let concave_cone_major =
            (-r_fillet + (concave_cone_c + h_signed) * beta.cos()) / beta.sin();
        assert!(
            torus.major_radius() < concave_cone_major,
            "both-concave major ({}) should be smaller than concave-cone-only ({concave_cone_major})",
            torus.major_radius()
        );
        assert!(
            concave_cone_major < convex_major,
            "concave-cone-only major ({concave_cone_major}) should be smaller than convex ({convex_major})"
        );

        // Sphere contact at distance R_s.
        let sph_axial = big_r_s * z_b / q_s;
        let sph_radial = big_r_s * expected_major / q_s;
        let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
        );

        // Cone contact on cone surface.
        let cone_axial = z_b - r_fillet * beta.cos();
        let cone_radial = expected_major + r_fillet * beta.sin();
        let predicted_cone_radial = (cone_axial + h_signed) * cot_b;
        assert!(
            (cone_radial - predicted_cone_radial).abs() < 1e-9,
            "cone contact must lie on cone: predicted {predicted_cone_radial}, got {cone_radial}"
        );
    }

    /// Sphere-cone convex chamfer: a sphere centered on the cone axis,
    /// chamfer rounding the corner where they meet.
    ///
    /// For sphere at origin (R_s=3), cone apex at (0,0,−2) with axis +z
    /// and half-angle π/3, both faces NOT reversed, symmetric d=0.3:
    ///   - h_signed = +2, β = π/3, cot β = 1/√3
    ///   - Spine z (from sphere center) ≈ +1.949 (the +z spine)
    ///   - Spine radial r_spine = (z+h)·cot β ≈ 2.279
    ///   - Sphere contact: δ=0.1, sphere_arm_sign=−1,
    ///     r_sph = r_spine·cos δ − spine_z·sin δ ≈ 2.073,
    ///     z_sph = spine_z·cos δ + r_spine·sin δ ≈ 2.167
    ///   - Cone contact (toward apex): r=r_spine−d·cos β ≈ 2.129,
    ///     z=spine_z−d·sin β ≈ 1.689
    ///   - Δr ≈ +0.056, Δz ≈ −0.478 ⇒ apex z ≈ 19.86 (well above contacts)
    #[test]
    fn sphere_cone_chamfer_convex_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0;
        let beta: f64 = std::f64::consts::PI / 3.0;
        let d: f64 = 0.3;

        // Solve for spine z (on +z side).
        let cot_b = beta.cos() / beta.sin();
        let qa = 1.0 / (beta.sin() * beta.sin());
        let qb = 2.0 * h_signed * cot_b * cot_b;
        let qc = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb * qb - 4.0 * qa * qc;
        let z_spine = (-qb + q_disc.sqrt()) / (2.0 * qa);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone.clone())));

        let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
            .unwrap()
            .expect("convex sphere-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts.
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let sphere_arm_sign = -1.0_f64; // spine_z > 0
        let r_sph_pred = r_spine * cos_d + sphere_arm_sign * z_spine * sin_d;
        let z_sph_pred = z_spine * cos_d - sphere_arm_sign * r_spine * sin_d;
        let r_cone_pred = r_spine - d * beta.cos();
        let z_cone_pred = z_spine - d * beta.sin();

        // Predicted apex.
        let dr = r_cone_pred - r_sph_pred;
        let dz = z_cone_pred - z_sph_pred;
        let expected_apex_z = z_sph_pred - r_sph_pred * dz / dr;
        let mid_z = 0.5 * (z_sph_pred + z_cone_pred);
        let r_avg = 0.5 * (r_sph_pred + r_cone_pred);
        let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

        assert!(
            expected_apex_z > z_sph_pred && expected_apex_z > z_cone_pred,
            "apex should be above both contacts, got apex_z={expected_apex_z}"
        );
        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
            "chamfer half-angle should be atan(|z_apex - mid_z| / r_avg) = {expected_beta}, got {}",
            chamfer_cone.half_angle()
        );

        // Apex on +z axis.
        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );

        // Cone axis = -z (apex above contacts, opens downward).
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
            "convex chamfer cone axis should be -z, got {axis:?}"
        );

        // Both contacts on chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
        let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cone - want_cone).length() < 1e-9,
            "cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
        );

        // Both contacts on their respective surfaces.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: distance={dist_sph}, want R_s={big_r_s}"
        );
        let cone_predicted_radial = (z_cone_pred + h_signed) * cot_b;
        assert!(
            (r_cone_pred - cone_predicted_radial).abs() < 1e-9,
            "cone contact must lie on cone surface: predicted radial {cone_predicted_radial}, got {r_cone_pred}"
        );
    }

    /// Sphere-cone chamfer with BOTH faces reversed (sphere cavity meets
    /// cone cavity at a concave corner). s_sph = s_cone = −1 flip both
    /// meridian arms.
    ///
    /// For R_s=3, h=2, β=π/3, both faces REVERSED, d=0.3:
    ///   - Sphere arm flips: contact moves to OPPOSITE cap (toward cone
    ///     side) ⇒ z_sph DECREASES below spine_z
    ///   - Cone arm flips: contact moves AWAY from apex along generator
    ///     ⇒ r_cone INCREASES, z_cone INCREASES (away from apex)
    ///   - These flips compose to give a different chamfer cone than
    ///     the convex case
    #[test]
    fn sphere_cone_chamfer_both_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0;
        let beta: f64 = std::f64::consts::PI / 3.0;
        let d: f64 = 0.3;

        let cot_b = beta.cos() / beta.sin();
        let qa_q = 1.0 / (beta.sin() * beta.sin());
        let qb_q = 2.0 * h_signed * cot_b * cot_b;
        let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
        let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(sph.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone.clone()),
        ));

        let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
            .unwrap()
            .expect("both-concave sphere-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts with s_sph = s_cone = -1.
        // sphere_arm_sign = -spine_sign · s_sph = -1·-1 = +1 (flipped from convex).
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        let sphere_arm_sign = 1.0_f64; // s_sph = -1, spine_sign = +1
        let r_sph_pred = r_spine * cos_d + sphere_arm_sign * z_spine * sin_d;
        let z_sph_pred = z_spine * cos_d - sphere_arm_sign * r_spine * sin_d;
        // Cone arm: s_cone = -1, so go AWAY from apex.
        let r_cone_pred = r_spine + d * beta.cos();
        let z_cone_pred = z_spine + d * beta.sin();

        // Sphere contact moved toward cone side (z DECREASED below spine).
        assert!(
            z_sph_pred < z_spine,
            "concave sphere contact should be below spine z (toward cone): got {z_sph_pred} vs spine {z_spine}"
        );
        // Cone contact moved AWAY from apex (z and r INCREASED).
        assert!(
            r_cone_pred > r_spine && z_cone_pred > z_spine,
            "concave cone contact should be away from apex: got ({r_cone_pred}, {z_cone_pred})"
        );

        // Both contacts on chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
        let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "concave sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cone - want_cone).length() < 1e-9,
            "concave cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
        );

        // Sphere contact at distance R_s.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
        );

        // Cone contact on cone surface.
        let predicted_cone_radial = (z_cone_pred + h_signed) * cot_b;
        assert!(
            (r_cone_pred - predicted_cone_radial).abs() < 1e-9,
            "cone contact must lie on cone surface: predicted radial {predicted_cone_radial}, got {r_cone_pred}"
        );
    }

    /// Sphere-cone mixed chamfer: sphere convex (s_sph=+1) + cone
    /// concave (s_cone=−1). Sphere contact lies on the AWAY-from-cone
    /// cap (like convex-convex), but cone contact moves AWAY from
    /// apex along generator instead of toward it (like both-concave
    /// for the cone arm).
    ///
    /// For R_s=3, h=2, β=π/3, sphere NOT reversed, cone REVERSED, d=0.3:
    ///   - sphere_arm_sign = -1·+1 = -1 (convex)
    ///   - r_sph = r_spine·cos δ − spine_z·sin δ ≈ 2.073 (same as convex)
    ///   - z_sph = spine_z·cos δ + r_spine·sin δ ≈ 2.167 (above spine)
    ///   - cone goes AWAY from apex: r_cone = r_spine + d·cos β ≈ 2.429,
    ///     z_cone = spine_z + d·sin β ≈ 2.209 (above spine, toward sphere)
    #[test]
    fn sphere_cone_chamfer_mixed_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{ConicalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let h_signed: f64 = 2.0;
        let beta: f64 = std::f64::consts::PI / 3.0;
        let d: f64 = 0.3;

        let cot_b = beta.cos() / beta.sin();
        let qa_q = 1.0 / (beta.sin() * beta.sin());
        let qb_q = 2.0 * h_signed * cot_b * cot_b;
        let qc_q = h_signed * h_signed * cot_b * cot_b - big_r_s * big_r_s;
        let q_disc = qb_q * qb_q - 4.0 * qa_q * qc_q;
        let z_spine = (-qb_q + q_disc.sqrt()) / (2.0 * qa_q);
        let r_spine = (z_spine + h_signed) * cot_b;

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, -h_signed),
            Vec3::new(0.0, 0.0, 1.0),
            beta,
        )
        .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Sphere NOT reversed, cone REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new(w1, vec![], FaceSurface::Sphere(sph.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone.clone()),
        ));

        let result = sphere_cone_chamfer(&sph, &cone, &spine, &topo, d, d, face_sphere, face_cone)
            .unwrap()
            .expect("mixed sphere-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts (s_sph=+1, s_cone=-1).
        let delta = d / big_r_s;
        let (sin_d, cos_d) = delta.sin_cos();
        // sphere_arm_sign = -spine_sign · s_sph = -1 · +1 = -1 (convex sphere).
        let r_sph_pred = r_spine * cos_d - z_spine * sin_d;
        let z_sph_pred = z_spine * cos_d + r_spine * sin_d;
        // s_cone = -1 ⇒ cone goes AWAY from apex.
        let r_cone_pred = r_spine + d * beta.cos();
        let z_cone_pred = z_spine + d * beta.sin();

        // Sphere contact on natural convex arm (above spine).
        assert!(
            z_sph_pred > z_spine,
            "convex sphere contact should be above spine: got {z_sph_pred}"
        );
        // Cone contact moves AWAY from apex (away from convex direction).
        assert!(
            r_cone_pred > r_spine && z_cone_pred > z_spine,
            "concave cone contact should be away from apex: got ({r_cone_pred}, {z_cone_pred})"
        );

        // Both contacts on chamfer cone.
        let want_sph = Point3::new(r_sph_pred, 0.0, z_sph_pred);
        let want_cone = Point3::new(r_cone_pred, 0.0, z_cone_pred);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_sph);
        let on_cone_sph = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_cone);
        let on_cone_cone = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_sph - want_sph).length() < 1e-9,
            "sphere contact must lie on chamfer cone: {on_cone_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_cone_cone - want_cone).length() < 1e-9,
            "cone contact must lie on chamfer cone: {on_cone_cone:?} vs {want_cone:?}"
        );

        // Sphere contact at distance R_s.
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: {dist_sph} vs R_s={big_r_s}"
        );

        // Cone contact on cone surface.
        let predicted_cone_radial = (z_cone_pred + h_signed) * cot_b;
        assert!(
            (r_cone_pred - predicted_cone_radial).abs() < 1e-9,
            "cone contact must lie on cone: predicted {predicted_cone_radial}, got {r_cone_pred}"
        );
    }

    /// Cylinder-cylinder convex fillet for two intersecting cylinders
    /// with PARALLEL axes. The intersection is two straight lines parallel
    /// to the cyl axes, and the rolling-ball blend is an exact cylinder
    /// around an axis parallel to those.
    ///
    /// For cyl1 axis = +z through origin (r=2), cyl2 axis = +z at (3, 0, *)
    /// (r=2.5), D=3, both faces NOT reversed, r=0.4:
    ///   - x_spine = (4 − 6.25 + 9)/6 = 1.125
    ///   - y_spine = ±√(4 − 1.265625) = ±1.654
    ///   - For r=0.4: Q1=2.4, Q2=2.9
    ///     x_ball = (Q1²−Q2²+D²)/(2D) = (5.76−8.41+9)/6 ≈ 1.058
    ///     y_ball = sign·√(Q1²−x_ball²) = √(5.76−1.119) ≈ 2.154
    ///   - Fillet cylinder axis +z at (1.058, 2.154, *), radius 0.4
    #[test]
    fn cylinder_cylinder_fillet_parallel_axes_emits_cylinder() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;

        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();

        // Spine: line at (x_spine, +y_spine, z) for z ∈ [0, 4]; segment
        // direction = +z.
        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())));

        let result = cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("parallel-axis cyl-cyl fillet should produce a stripe");

        let fillet_cyl = match result.stripe.surface {
            FaceSurface::Cylinder(c) => c,
            other => panic!("expected Cylinder, got {}", other.type_tag()),
        };

        // Predicted ball position.
        let q1 = r1 + r_fillet;
        let q2 = r2 + r_fillet;
        let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
        let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

        assert!(
            (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
            "fillet cylinder radius should equal r = {r_fillet}, got {}",
            fillet_cyl.radius()
        );
        // Axis = +z (parallel to original cyls).
        let axis = fillet_cyl.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "fillet cylinder axis should be +z, got {axis:?}"
        );
        // Origin at (x_ball, y_ball, z_lo).
        let origin = fillet_cyl.origin();
        assert!(
            (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
            "fillet cylinder origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
        );

        // Cyl1 contact line at (r1·x_ball/q1, r1·y_ball/q1, z).
        let want_c1 = Point3::new(r1 * x_ball / q1, r1 * y_ball / q1, z_lo);
        let dist_c1_axis = (want_c1.x().powi(2) + want_c1.y().powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "cyl1 contact must lie on cyl1 (radial = r1): got {dist_c1_axis}, want {r1}"
        );
        // Cyl2 contact line at (D + r2·(x_ball−D)/q2, r2·y_ball/q2, z).
        let want_c2 = Point3::new(big_d + r2 * (x_ball - big_d) / q2, r2 * y_ball / q2, z_lo);
        let dist_c2_axis = ((want_c2.x() - big_d).powi(2) + want_c2.y().powi(2)).sqrt();
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "cyl2 contact must lie on cyl2 (radial from cyl2 axis = r2): got {dist_c2_axis}, want {r2}"
        );

        // Both contacts on the fillet cylinder surface (distance r from
        // ball-line in xy).
        let dist_c1_to_ball =
            ((want_c1.x() - x_ball).powi(2) + (want_c1.y() - y_ball).powi(2)).sqrt();
        let dist_c2_to_ball =
            ((want_c2.x() - x_ball).powi(2) + (want_c2.y() - y_ball).powi(2)).sqrt();
        assert!(
            (dist_c1_to_ball - r_fillet).abs() < 1e-9,
            "cyl1 contact must lie on fillet cylinder: distance from ball-line = {dist_c1_to_ball}, want r = {r_fillet}"
        );
        assert!(
            (dist_c2_to_ball - r_fillet).abs() < 1e-9,
            "cyl2 contact must lie on fillet cylinder: distance from ball-line = {dist_c2_to_ball}, want r = {r_fillet}"
        );
    }

    /// Cylinder-cylinder both-concave fillet: two intersecting cylindrical
    /// holes with parallel axes. Both s_i = −1 ⇒ Q_i = r_i − r (internal
    /// tangency), so the rolling ball is INSIDE both cylinders.
    ///
    /// For r1=2, r2=2.5, D=3, both faces REVERSED, r=0.4:
    ///   - Q1 = 1.6, Q2 = 2.1
    ///   - x_ball = (Q1²−Q2²+D²)/(2D) = (2.56−4.41+9)/6 ≈ 1.192
    ///   - y_ball = sign·√(Q1²−x_ball²) = √(2.56−1.421) ≈ 1.067
    ///   - Both contacts internal: cyl1 contact at radial r1·x_ball/Q1 from
    ///     cyl1 axis (different from convex case)
    #[test]
    fn cylinder_cylinder_fillet_both_concave_emits_cylinder() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;

        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();

        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Cylinder(cyl1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl2.clone()),
        ));

        let result = cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
            .unwrap()
            .expect("both-concave cyl-cyl fillet should produce a stripe");

        let fillet_cyl = match result.stripe.surface {
            FaceSurface::Cylinder(c) => c,
            other => panic!("expected Cylinder, got {}", other.type_tag()),
        };

        let q1 = r1 - r_fillet;
        let q2 = r2 - r_fillet;
        let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
        let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

        assert!(
            (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
            "fillet radius should be r = {r_fillet}, got {}",
            fillet_cyl.radius()
        );
        let origin = fillet_cyl.origin();
        assert!(
            (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
            "concave fillet origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
        );

        // Axis parallel to original cyl axes (+z).
        let axis = fillet_cyl.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "concave fillet axis should be +z (parallel to original cyls), got {axis:?}"
        );

        // Verify ball is INSIDE both cyls (internal tangency) — read
        // from the EMITTED cylinder origin, not from our own computed
        // x_ball/y_ball (which would be tautologically Q_i < r_i by
        // construction).
        let actual_dist_to_cyl1_axis = (origin.x().powi(2) + origin.y().powi(2)).sqrt();
        let actual_dist_to_cyl2_axis = ((origin.x() - big_d).powi(2) + origin.y().powi(2)).sqrt();
        assert!(
            actual_dist_to_cyl1_axis < r1 - 1e-9,
            "concave: emitted fillet origin must be INSIDE cyl1 (distance {actual_dist_to_cyl1_axis} < r1 = {r1})"
        );
        assert!(
            actual_dist_to_cyl2_axis < r2 - 1e-9,
            "concave: emitted fillet origin must be INSIDE cyl2 (distance {actual_dist_to_cyl2_axis} < r2 = {r2})"
        );

        // Cyl1 and cyl2 contacts lie on their respective cylinder surfaces.
        let want_c1 = Point3::new(r1 * x_ball / q1, r1 * y_ball / q1, z_lo);
        let dist_c1_axis = (want_c1.x().powi(2) + want_c1.y().powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "cyl1 contact must lie on cyl1: got {dist_c1_axis}, want {r1}"
        );
        let want_c2 = Point3::new(big_d + r2 * (x_ball - big_d) / q2, r2 * y_ball / q2, z_lo);
        let dist_c2_axis = ((want_c2.x() - big_d).powi(2) + want_c2.y().powi(2)).sqrt();
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "cyl2 contact must lie on cyl2: got {dist_c2_axis}, want {r2}"
        );

        // Tangency to the EMITTED fillet cylinder: each contact must be
        // at distance `r_fillet` from the fillet-cyl axis (the ball
        // line) in the perpendicular plane. This catches axis/origin
        // bugs that the previous assertions wouldn't see.
        let dist_c1_to_ball =
            ((want_c1.x() - origin.x()).powi(2) + (want_c1.y() - origin.y()).powi(2)).sqrt();
        let dist_c2_to_ball =
            ((want_c2.x() - origin.x()).powi(2) + (want_c2.y() - origin.y()).powi(2)).sqrt();
        assert!(
            (dist_c1_to_ball - r_fillet).abs() < 1e-9,
            "cyl1 contact must be at distance r from fillet ball-line: got {dist_c1_to_ball}, want {r_fillet}"
        );
        assert!(
            (dist_c2_to_ball - r_fillet).abs() < 1e-9,
            "cyl2 contact must be at distance r from fillet ball-line: got {dist_c2_to_ball}, want {r_fillet}"
        );
    }

    /// Cylinder-cylinder mixed-convexity fillet: covers BOTH (s1=+1,
    /// s2=−1) and (s1=−1, s2=+1) via a parameterized closure.
    ///
    /// For mixed configs, one cylinder is internally tangent to the
    /// rolling ball (`Q_i = r_i − r`) and the other externally tangent
    /// (`Q_i = r_i + r`). The resulting fillet cylinder has its origin
    /// at a position determined by the asymmetric `(Q1², Q2²)` pair —
    /// distinct from both convex (both `+ r`) and both-concave
    /// (both `− r`) cases.
    #[test]
    fn cylinder_cylinder_fillet_mixed_emits_cylinder() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let r_fillet: f64 = 0.4;
        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();

        let run_case = |reverse_s1: bool, reverse_s2: bool| {
            let mut topo = Topology::new();
            let cyl1 =
                CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                    .unwrap();
            let cyl2 =
                CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                    .unwrap();
            let z_lo = 0.0_f64;
            let z_hi = 4.0_f64;
            let p_start = Point3::new(x_spine, y_spine, z_lo);
            let p_end = Point3::new(x_spine, y_spine, z_hi);
            let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
            let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
            let line = brepkit_math::nurbs::curve::NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![p_start, p_end],
                vec![1.0, 1.0],
            )
            .unwrap();
            let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
            let spine = Spine::from_single_edge(&topo, eid).unwrap();

            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
            let face1 = if reverse_s1 {
                topo.add_face(Face::new_reversed(
                    w1,
                    vec![],
                    FaceSurface::Cylinder(cyl1.clone()),
                ))
            } else {
                topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())))
            };
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
            let face2 = if reverse_s2 {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Cylinder(cyl2.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())))
            };

            let result =
                cylinder_cylinder_fillet(&cyl1, &cyl2, &spine, &topo, r_fillet, face1, face2)
                    .unwrap()
                    .expect("mixed cyl-cyl fillet should produce a stripe");

            let fillet_cyl = match result.stripe.surface {
                FaceSurface::Cylinder(c) => c,
                other => panic!(
                    "({reverse_s1}, {reverse_s2}): expected Cylinder, got {}",
                    other.type_tag()
                ),
            };

            // Predicted parameters with per-face Q-substitution.
            let s1_signed = if reverse_s1 { -1.0_f64 } else { 1.0_f64 };
            let s2_signed = if reverse_s2 { -1.0_f64 } else { 1.0_f64 };
            let q1 = r1 + s1_signed * r_fillet;
            let q2 = r2 + s2_signed * r_fillet;
            let x_ball = (q1 * q1 - q2 * q2 + big_d * big_d) / (2.0 * big_d);
            let y_ball = (q1 * q1 - x_ball * x_ball).sqrt();

            assert!(
                (fillet_cyl.radius() - r_fillet).abs() < 1e-12,
                "({reverse_s1}, {reverse_s2}): fillet radius should be r = {r_fillet}, got {}",
                fillet_cyl.radius()
            );
            let origin = fillet_cyl.origin();
            assert!(
                (origin.x() - x_ball).abs() < 1e-12 && (origin.y() - y_ball).abs() < 1e-12,
                "({reverse_s1}, {reverse_s2}): fillet origin should be ({x_ball}, {y_ball}, *), got {origin:?}"
            );

            // Axis parallel to original cyls.
            let axis = fillet_cyl.axis();
            assert!(
                axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
                "({reverse_s1}, {reverse_s2}): fillet axis should be +z, got {axis:?}"
            );

            // Read EMITTED contact endpoints. Each lies on its
            // respective cylinder (radial r_i from cyl_i axis) AND at
            // distance r from the fillet ball-line.
            let (t1_start, _) = result.stripe.contact1.domain();
            let c1_point = result.stripe.contact1.evaluate(t1_start);
            let (t2_start, _) = result.stripe.contact2.domain();
            let c2_point = result.stripe.contact2.evaluate(t2_start);

            let dist_c1_axis = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
            let dist_c2_axis = ((c2_point.x() - big_d).powi(2) + c2_point.y().powi(2)).sqrt();
            assert!(
                (dist_c1_axis - r1).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c1 must lie on cyl1: {dist_c1_axis} vs r1 = {r1}"
            );
            assert!(
                (dist_c2_axis - r2).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c2 must lie on cyl2: {dist_c2_axis} vs r2 = {r2}"
            );

            let dist_c1_to_ball =
                ((c1_point.x() - x_ball).powi(2) + (c1_point.y() - y_ball).powi(2)).sqrt();
            let dist_c2_to_ball =
                ((c2_point.x() - x_ball).powi(2) + (c2_point.y() - y_ball).powi(2)).sqrt();
            assert!(
                (dist_c1_to_ball - r_fillet).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c1 must be at r from fillet ball-line: \
                 {dist_c1_to_ball} vs r = {r_fillet}"
            );
            assert!(
                (dist_c2_to_ball - r_fillet).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c2 must be at r from fillet ball-line: \
                 {dist_c2_to_ball} vs r = {r_fillet}"
            );
        };

        run_case(false, true); // (s1=+1, s2=-1)
        run_case(true, false); // (s1=-1, s2=+1)
    }

    /// Cone-cone coaxial convex fillet: two cones sharing the same axis
    /// line with different half-angles. Their intersection is a single
    /// circle, and the rolling-ball blend is a torus.
    ///
    /// For cone1 apex at origin, β1=π/3 (60°); cone2 apex at (0,0,2),
    /// β2=π/4 (45°); both axes +z, both faces NOT reversed, r=0.3:
    ///   - sin(β1−β2) = sin(π/12) ≈ 0.2588
    ///   - z_spine = h_2·cos β2·sin β1/sin(β1−β2) ≈ 4.732
    ///   - r_spine = z_spine·cot β1 ≈ 2.732
    ///   - z_b ≈ 4.548 (slightly less than z_spine for convex case)
    ///   - R_t ≈ 2.974 (slightly larger than r_spine — fillet outside both cones)
    #[test]
    fn cone_cone_coaxial_fillet_convex_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let r_fillet: f64 = 0.3;

        // Predicted spine.
        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())));

        let result =
            cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
                .unwrap()
                .expect("convex coaxial cone-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted z_b and R_t (convex-convex: s1=s2=+1).
        let expected_z_b =
            (h_2 * beta2.cos() * beta1.sin() + r_fillet * (beta2.sin() - beta1.sin())) / sin_minus;
        let expected_major = (expected_z_b * (beta1.cos() - beta2.cos()) + h_2 * beta2.cos())
            / (beta1.sin() - beta2.sin());

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "torus major should be {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should equal r = {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Major > spine radius (convex fillet outside both cones).
        assert!(
            torus.major_radius() > r_spine,
            "convex fillet major ({}) should be > r_spine ({r_spine})",
            torus.major_radius()
        );

        // Torus center on axis at z = z_b.
        let center = torus.center();
        assert!(
            center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
            "torus center on z-axis, got {center:?}"
        );
        assert!(
            (center.z() - expected_z_b).abs() < 1e-9,
            "torus center z should be {expected_z_b}, got {}",
            center.z()
        );

        // Verify rolling-ball external tangency to BOTH cones:
        //   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = r.
        let tang1 = expected_major * beta1.sin() - expected_z_b * beta1.cos();
        let tang2 = expected_major * beta2.sin() - (expected_z_b - h_2) * beta2.cos();
        assert!(
            (tang1 - r_fillet).abs() < 1e-9,
            "cone1 tangency: {tang1} should equal r = {r_fillet}"
        );
        assert!(
            (tang2 - r_fillet).abs() < 1e-9,
            "cone2 tangency: {tang2} should equal r = {r_fillet}"
        );

        // Both contacts on the torus + on their respective cones.
        let cot_b1 = beta1.cos() / beta1.sin();
        let cot_b2 = beta2.cos() / beta2.sin();
        let c1_axial = expected_z_b + r_fillet * beta1.cos();
        let c1_radial = expected_major - r_fillet * beta1.sin();
        let c2_axial = expected_z_b + r_fillet * beta2.cos();
        let c2_radial = expected_major - r_fillet * beta2.sin();
        let want_c1 = Point3::new(c1_radial, 0.0, c1_axial);
        let want_c2 = Point3::new(c2_radial, 0.0, c2_axial);

        // Cone1: r = (z − z_apex_1) · cot β1 = c1_axial · cot β1.
        let pred_c1_radial = c1_axial * cot_b1;
        assert!(
            (c1_radial - pred_c1_radial).abs() < 1e-9,
            "cone1 contact must lie on cone1 surface: predicted radial {pred_c1_radial}, got {c1_radial}"
        );
        // Cone2: r = (z − h_2) · cot β2.
        let pred_c2_radial = (c2_axial - h_2) * cot_b2;
        assert!(
            (c2_radial - pred_c2_radial).abs() < 1e-9,
            "cone2 contact must lie on cone2 surface: predicted radial {pred_c2_radial}, got {c2_radial}"
        );

        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_c1);
        let on_torus_c1 = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_c2);
        let on_torus_c2 = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_c1 - want_c1).length() < 1e-9,
            "cone1 contact must lie on torus: {on_torus_c1:?} vs {want_c1:?}"
        );
        assert!(
            (on_torus_c2 - want_c2).length() < 1e-9,
            "cone2 contact must lie on torus: {on_torus_c2:?} vs {want_c2:?}"
        );
    }

    /// Cone-cone coaxial both-concave fillet: two concave conical
    /// cavities sharing an axis. Both s_i = −1 ⇒ rolling ball is
    /// internally tangent to both cones.
    ///
    /// The same setup as the convex test (cone1 apex at origin β=π/3;
    /// cone2 apex at z=2 β=π/4; shared axis +z; r=0.3) but with both
    /// faces REVERSED — exercises the `s_i = −1` branches in both
    /// `z_b` and `R_t` formulas, plus the `(s1 − s2) · r` term that
    /// vanishes when both signs match.
    #[test]
    fn cone_cone_coaxial_fillet_both_concave_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let r_fillet: f64 = 0.3;

        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Cone(cone1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone2.clone()),
        ));

        let result =
            cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
                .unwrap()
                .expect("both-concave coaxial cone-cone fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        // Predicted z_b, R_t (s1=s2=-1).
        let s1 = -1.0_f64;
        let s2 = -1.0_f64;
        let expected_z_b = (h_2 * beta2.cos() * beta1.sin()
            + r_fillet * (s1 * beta2.sin() - s2 * beta1.sin()))
            / sin_minus;
        let expected_major =
            (expected_z_b * (beta1.cos() - beta2.cos()) + h_2 * beta2.cos() + (s1 - s2) * r_fillet)
                / (beta1.sin() - beta2.sin());

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-9,
            "concave torus major should be {expected_major}, got {}",
            torus.major_radius()
        );

        // Concave fillet sits INSIDE both cones (major < r_spine).
        assert!(
            torus.major_radius() < r_spine,
            "concave fillet major ({}) should be < r_spine ({r_spine})",
            torus.major_radius()
        );

        // Internal tangency to both cones:
        //   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = s_i · r = −r.
        let tang1 = expected_major * beta1.sin() - expected_z_b * beta1.cos();
        let tang2 = expected_major * beta2.sin() - (expected_z_b - h_2) * beta2.cos();
        assert!(
            (tang1 + r_fillet).abs() < 1e-9,
            "cone1 internal tangency: {tang1} should equal -r = {}",
            -r_fillet
        );
        assert!(
            (tang2 + r_fillet).abs() < 1e-9,
            "cone2 internal tangency: {tang2} should equal -r = {}",
            -r_fillet
        );

        // Contacts on respective cone surfaces.
        let cot_b1 = beta1.cos() / beta1.sin();
        let cot_b2 = beta2.cos() / beta2.sin();
        let c1_axial = expected_z_b + s1 * r_fillet * beta1.cos();
        let c1_radial = expected_major - s1 * r_fillet * beta1.sin();
        let c2_axial = expected_z_b + s2 * r_fillet * beta2.cos();
        let c2_radial = expected_major - s2 * r_fillet * beta2.sin();
        let pred_c1_radial = c1_axial * cot_b1;
        assert!(
            (c1_radial - pred_c1_radial).abs() < 1e-9,
            "cone1 contact must lie on cone1: predicted {pred_c1_radial}, got {c1_radial}"
        );
        let pred_c2_radial = (c2_axial - h_2) * cot_b2;
        assert!(
            (c2_radial - pred_c2_radial).abs() < 1e-9,
            "cone2 contact must lie on cone2: predicted {pred_c2_radial}, got {c2_radial}"
        );

        // Both contacts on torus.
        let want_c1 = Point3::new(c1_radial, 0.0, c1_axial);
        let want_c2 = Point3::new(c2_radial, 0.0, c2_axial);
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_c1);
        let on_torus_c1 = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_c2);
        let on_torus_c2 = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_c1 - want_c1).length() < 1e-9,
            "cone1 contact must lie on torus: {on_torus_c1:?} vs {want_c1:?}"
        );
        assert!(
            (on_torus_c2 - want_c2).length() < 1e-9,
            "cone2 contact must lie on torus: {on_torus_c2:?} vs {want_c2:?}"
        );
    }

    /// Cone-cone coaxial mixed-convexity fillet: covers BOTH
    /// (s1=+1, s2=−1) and (s1=−1, s2=+1) sign combinations.
    ///
    /// In mixed configs the linear system
    ///   R_t · sin β_i − (z_b − z_apex_i) · cos β_i = s_i · r
    /// solves with `s1 ≠ s2`, so the (s1−s2) term in the major-radius
    /// formula contributes ±2r and the (s1·sin β2 − s2·sin β1) term
    /// adds (sin β1 + sin β2) — both opposite signs from the symmetric
    /// cases. The result places the fillet ball *outside* the spine
    /// ring (one external + one internal contact) rather than aligned
    /// with it.
    ///
    /// Verifies emitted torus major satisfies the closed-form formula
    /// and that contacts read from emitted curves via `evaluate(t_start)`
    /// satisfy each cone's tangency equation `R_t · sin β_i − (z_b − z_apex_i) · cos β_i = s_i · r`
    /// with the correct sign of `r`.
    #[test]
    fn cone_cone_coaxial_fillet_mixed_emits_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let r_fillet: f64 = 0.3;
        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let run_case = |reverse_s1: bool, reverse_s2: bool| {
            let mut topo = Topology::new();
            let cone1 =
                ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                    .unwrap();
            let cone2 =
                ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                    .unwrap();
            let spine_circle = Circle3D::new(
                Point3::new(0.0, 0.0, z_spine),
                Vec3::new(0.0, 0.0, 1.0),
                r_spine,
            )
            .unwrap();
            let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
            let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
            let spine = Spine::from_single_edge(&topo, eid).unwrap();

            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
            let face1 = if reverse_s1 {
                topo.add_face(Face::new_reversed(
                    w1,
                    vec![],
                    FaceSurface::Cone(cone1.clone()),
                ))
            } else {
                topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())))
            };
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
            let face2 = if reverse_s2 {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Cone(cone2.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())))
            };

            let result =
                cone_cone_coaxial_fillet(&cone1, &cone2, &spine, &topo, r_fillet, face1, face2)
                    .unwrap()
                    .expect("mixed coaxial cone-cone fillet should produce a stripe");

            let torus = match result.stripe.surface {
                FaceSurface::Torus(ref t) => t.clone(),
                ref other => panic!(
                    "({reverse_s1}, {reverse_s2}): expected Torus, got {}",
                    other.type_tag()
                ),
            };

            let s1 = if reverse_s1 { -1.0_f64 } else { 1.0 };
            let s2 = if reverse_s2 { -1.0_f64 } else { 1.0 };
            let expected_z_b = (h_2 * beta2.cos() * beta1.sin()
                + r_fillet * (s1 * beta2.sin() - s2 * beta1.sin()))
                / sin_minus;
            let expected_major = (expected_z_b * (beta1.cos() - beta2.cos())
                + h_2 * beta2.cos()
                + (s1 - s2) * r_fillet)
                / (beta1.sin() - beta2.sin());

            assert!(
                (torus.major_radius() - expected_major).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): mixed torus major should be {expected_major}, got {}",
                torus.major_radius()
            );
            assert!(
                (torus.minor_radius() - r_fillet).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): minor should be {r_fillet}, got {}",
                torus.minor_radius()
            );

            // Tangency check using closed form.
            let tang1 = expected_major * beta1.sin() - expected_z_b * beta1.cos();
            let tang2 = expected_major * beta2.sin() - (expected_z_b - h_2) * beta2.cos();
            assert!(
                (tang1 - s1 * r_fillet).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): cone1 tangency: {tang1} should equal s1·r = {}",
                s1 * r_fillet
            );
            assert!(
                (tang2 - s2 * r_fillet).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): cone2 tangency: {tang2} should equal s2·r = {}",
                s2 * r_fillet
            );

            // Read EMITTED contact endpoints — c_i is on cone_i.
            let (t1_start, _) = result.stripe.contact1.domain();
            let c1_point = result.stripe.contact1.evaluate(t1_start);
            let (t2_start, _) = result.stripe.contact2.domain();
            let c2_point = result.stripe.contact2.evaluate(t2_start);

            let cot_b1 = beta1.cos() / beta1.sin();
            let cot_b2 = beta2.cos() / beta2.sin();
            let c1_radial = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
            let c2_radial = (c2_point.x().powi(2) + c2_point.y().powi(2)).sqrt();
            let pred_c1_radial = c1_point.z() * cot_b1;
            let pred_c2_radial = (c2_point.z() - h_2) * cot_b2;
            assert!(
                (c1_radial - pred_c1_radial).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact1 must lie on cone1: predicted r={pred_c1_radial}, got {c1_radial}"
            );
            assert!(
                (c2_radial - pred_c2_radial).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact2 must lie on cone2: predicted r={pred_c2_radial}, got {c2_radial}"
            );
        };

        run_case(false, true); // (s1=+1, s2=-1)
        run_case(true, false); // (s1=-1, s2=+1)
    }

    /// Cylinder-cylinder convex chamfer: parallel-axis cyls with the
    /// chamfer surface a planar bevel containing both contact lines
    /// (each parallel to the cyl axes).
    ///
    /// For cyl1 r=2 at origin axis +z, cyl2 r=2.5 at (3,0,*) axis +z,
    /// D=3, both faces NOT reversed, +y spine, d=0.4 each:
    ///   - Spine at (1.125, √3, *)
    ///   - Δθ_1 = +1·d/r1 = 0.2 (CCW on cyl1, AWAY from cyl2)
    ///   - Δθ_2 = −1·d/r2 = −0.16 (CW on cyl2, AWAY from cyl1)
    ///   - Chamfer plane through both contact lines
    #[test]
    fn cylinder_cylinder_chamfer_convex_emits_plane() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;

        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();

        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())));

        let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("convex parallel-axis cyl-cyl chamfer should produce a stripe");

        let (chamfer_normal, chamfer_d) = match result.stripe.surface {
            FaceSurface::Plane { normal, d } => (normal, d),
            other => panic!("expected Plane, got {}", other.type_tag()),
        };

        // Predicted contacts.
        let dtheta1 = d / r1; // y_sign=+1, s1=+1
        let dtheta2 = -d / r2; // y_sign=+1, s2=+1, negation
        let (sin1, cos1) = dtheta1.sin_cos();
        let (sin2, cos2) = dtheta2.sin_cos();
        let c1_x = x_spine * cos1 - y_spine * sin1;
        let c1_y = y_spine * cos1 + x_spine * sin1;
        let c2_local_x = (x_spine - big_d) * cos2 - y_spine * sin2;
        let c2_local_y = y_spine * cos2 + (x_spine - big_d) * sin2;
        let c2_x = c2_local_x + big_d;
        let c2_y = c2_local_y;

        // Contact1 must lie on cyl1 (radial = r1 from cyl1 axis).
        let dist_c1_axis = (c1_x.powi(2) + c1_y.powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "cyl1 contact must lie on cyl1: distance = {dist_c1_axis}, want r1 = {r1}"
        );
        // Contact2 must lie on cyl2.
        let dist_c2_axis = ((c2_x - big_d).powi(2) + c2_y.powi(2)).sqrt();
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "cyl2 contact must lie on cyl2: distance = {dist_c2_axis}, want r2 = {r2}"
        );

        // Both contact LINES (z varies) must lie on the chamfer plane.
        // For a plane (normal · p = d) and contact at (c_x, c_y, z) for
        // any z, the plane equation must hold ∀z. Since the plane normal
        // is computed as a_cyl.cross(span) and a_cyl = +z, the normal
        // has zero z-component. ⇒ checking with z = z_lo suffices.
        let p1 = Point3::new(c1_x, c1_y, z_lo);
        let p2 = Point3::new(c2_x, c2_y, z_lo);
        let on_plane_1 = chamfer_normal.dot(Vec3::new(p1.x(), p1.y(), p1.z())) - chamfer_d;
        let on_plane_2 = chamfer_normal.dot(Vec3::new(p2.x(), p2.y(), p2.z())) - chamfer_d;
        assert!(
            on_plane_1.abs() < 1e-9,
            "cyl1 contact must lie on chamfer plane: residual {on_plane_1}"
        );
        assert!(
            on_plane_2.abs() < 1e-9,
            "cyl2 contact must lie on chamfer plane: residual {on_plane_2}"
        );

        // Chamfer plane normal must be perpendicular to the cyl axis +z
        // (the contact lines are along +z, so the plane contains them).
        assert!(
            chamfer_normal.z().abs() < 1e-12,
            "chamfer plane normal should be perpendicular to z axis, got {chamfer_normal:?}"
        );
    }

    /// Cylinder-cylinder both-concave chamfer at the −y spine (exercises
    /// BOTH the `s_i = −1` branches AND the `y_sign = −1` branch in the
    /// `dtheta_i = y_sign · s_i · d_i / r_i` formulas).
    ///
    /// Setup: same cylinders as the convex test (r1=2, r2=2.5, D=3,
    /// d=0.4) but spine at the −y intersection line and both faces
    /// REVERSED. This means:
    ///   - dtheta_1 = (−1)·(−1)·d/r1 = +d/r1 (still CCW on cyl1, but
    ///     for a different geometric reason — concave going TOWARD
    ///     cyl2 from the −y spine = +θ direction)
    ///   - dtheta_2 = −(−1)·(−1)·d/r2 = −d/r2 (CW on cyl2)
    ///
    /// The signs happen to match the convex +y-spine case numerically,
    /// but they reach there via a DIFFERENT path through the formulas,
    /// exercising both the y_sign and s_i flip branches.
    #[test]
    fn cylinder_cylinder_chamfer_both_concave_negative_y_emits_plane() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;

        let cyl1 =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                .unwrap();
        let cyl2 =
            CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                .unwrap();

        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = -((r1 * r1 - x_spine * x_spine).sqrt()); // NEGATIVE-y spine
        let z_lo = 0.0_f64;
        let z_hi = 4.0_f64;
        let p_start = Point3::new(x_spine, y_spine, z_lo);
        let p_end = Point3::new(x_spine, y_spine, z_hi);
        let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
        let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
        let line = brepkit_math::nurbs::curve::NurbsCurve::new(
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![p_start, p_end],
            vec![1.0, 1.0],
        )
        .unwrap();
        let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Cylinder(cyl1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl2.clone()),
        ));

        let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("both-concave −y-spine cyl-cyl chamfer should produce a stripe");

        let (chamfer_normal, chamfer_d) = match result.stripe.surface {
            FaceSurface::Plane { normal, d } => (normal, d),
            other => panic!("expected Plane, got {}", other.type_tag()),
        };

        // Predicted contacts. y_sign = -1, s1 = s2 = -1.
        // dtheta_1 = (-1)·(-1)·d/r1 = +d/r1
        // dtheta_2 = -(-1)·(-1)·d/r2 = -d/r2
        let dtheta1 = d / r1;
        let dtheta2 = -d / r2;
        let (sin1, cos1) = dtheta1.sin_cos();
        let (sin2, cos2) = dtheta2.sin_cos();
        let c1_x = x_spine * cos1 - y_spine * sin1;
        let c1_y = y_spine * cos1 + x_spine * sin1;
        let c2_local_x = (x_spine - big_d) * cos2 - y_spine * sin2;
        let c2_local_y = y_spine * cos2 + (x_spine - big_d) * sin2;
        let c2_x = c2_local_x + big_d;
        let c2_y = c2_local_y;

        // Contacts on respective cylinder surfaces.
        let dist_c1_axis = (c1_x.powi(2) + c1_y.powi(2)).sqrt();
        assert!(
            (dist_c1_axis - r1).abs() < 1e-9,
            "cyl1 contact must lie on cyl1: distance = {dist_c1_axis}, want r1 = {r1}"
        );
        let dist_c2_axis = ((c2_x - big_d).powi(2) + c2_y.powi(2)).sqrt();
        assert!(
            (dist_c2_axis - r2).abs() < 1e-9,
            "cyl2 contact must lie on cyl2: distance = {dist_c2_axis}, want r2 = {r2}"
        );

        // Both contacts on the chamfer plane.
        let p1 = Point3::new(c1_x, c1_y, z_lo);
        let p2 = Point3::new(c2_x, c2_y, z_lo);
        let on_plane_1 = chamfer_normal.dot(Vec3::new(p1.x(), p1.y(), p1.z())) - chamfer_d;
        let on_plane_2 = chamfer_normal.dot(Vec3::new(p2.x(), p2.y(), p2.z())) - chamfer_d;
        assert!(
            on_plane_1.abs() < 1e-9,
            "cyl1 contact must lie on chamfer plane: residual {on_plane_1}"
        );
        assert!(
            on_plane_2.abs() < 1e-9,
            "cyl2 contact must lie on chamfer plane: residual {on_plane_2}"
        );

        // Concave-going-TOWARD geometry: contacts now pull TOWARD the
        // other cyl (rather than away). For y_sign=-1 with concave, both
        // contacts have negative y components less negative than the
        // spine (toward y=0).
        assert!(
            c1_y > y_spine,
            "concave cyl1 contact should pull TOWARD y=0: got {c1_y} vs spine {y_spine}"
        );
        assert!(
            c2_y > y_spine,
            "concave cyl2 contact should pull TOWARD y=0: got {c2_y} vs spine {y_spine}"
        );

        // Chamfer plane perpendicular to z axis.
        assert!(
            chamfer_normal.z().abs() < 1e-12,
            "chamfer plane normal must be perpendicular to z, got {chamfer_normal:?}"
        );
    }

    /// Cylinder-cylinder mixed-convexity chamfer: covers BOTH (s1=+1,
    /// s2=−1) and (s1=−1, s2=+1) at the +y spine.
    ///
    /// In mixed configs the per-cyl angular displacement signs combine
    /// asymmetrically:
    ///   Δθ_1 = y_sign · s1 · d/r1
    ///   Δθ_2 = −y_sign · s2 · d/r2
    /// The product `s1 · s2` differs from the symmetric cases, giving
    /// a chord between contacts with a different slope and hence a
    /// distinct chamfer plane.
    ///
    /// Verifies emitted contact endpoints via `evaluate(t_start)` and
    /// asserts each lies on its cylinder surface AND on the chamfer
    /// plane (residual against `n · p − d` < tol).
    #[test]
    fn cylinder_cylinder_chamfer_mixed_emits_plane() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let r1: f64 = 2.0;
        let r2: f64 = 2.5;
        let big_d: f64 = 3.0;
        let d: f64 = 0.4;
        let x_spine = (r1 * r1 - r2 * r2 + big_d * big_d) / (2.0 * big_d);
        let y_spine = (r1 * r1 - x_spine * x_spine).sqrt();

        let run_case = |reverse_s1: bool, reverse_s2: bool| {
            let mut topo = Topology::new();
            let cyl1 =
                CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r1)
                    .unwrap();
            let cyl2 =
                CylindricalSurface::new(Point3::new(big_d, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r2)
                    .unwrap();
            let z_lo = 0.0_f64;
            let z_hi = 4.0_f64;
            let p_start = Point3::new(x_spine, y_spine, z_lo);
            let p_end = Point3::new(x_spine, y_spine, z_hi);
            let v_start = topo.add_vertex(Vertex::new(p_start, 1e-7));
            let v_end = topo.add_vertex(Vertex::new(p_end, 1e-7));
            let line = brepkit_math::nurbs::curve::NurbsCurve::new(
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![p_start, p_end],
                vec![1.0, 1.0],
            )
            .unwrap();
            let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(line)));
            let spine = Spine::from_single_edge(&topo, eid).unwrap();

            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
            let face1 = if reverse_s1 {
                topo.add_face(Face::new_reversed(
                    w1,
                    vec![],
                    FaceSurface::Cylinder(cyl1.clone()),
                ))
            } else {
                topo.add_face(Face::new(w1, vec![], FaceSurface::Cylinder(cyl1.clone())))
            };
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], false).unwrap());
            let face2 = if reverse_s2 {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Cylinder(cyl2.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Cylinder(cyl2.clone())))
            };

            let result = cylinder_cylinder_chamfer(&cyl1, &cyl2, &spine, &topo, d, d, face1, face2)
                .unwrap()
                .expect("mixed cyl-cyl chamfer should produce a stripe");

            let (chamfer_normal, chamfer_d) = match result.stripe.surface {
                FaceSurface::Plane { normal, d } => (normal, d),
                ref other => panic!(
                    "({reverse_s1}, {reverse_s2}): expected Plane, got {}",
                    other.type_tag()
                ),
            };

            // Read EMITTED contact endpoints.
            let (t1_start, _) = result.stripe.contact1.domain();
            let c1_point = result.stripe.contact1.evaluate(t1_start);
            let (t2_start, _) = result.stripe.contact2.domain();
            let c2_point = result.stripe.contact2.evaluate(t2_start);

            // Each lies on its cylinder surface.
            let dist_c1_axis = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
            let dist_c2_axis = ((c2_point.x() - big_d).powi(2) + c2_point.y().powi(2)).sqrt();
            assert!(
                (dist_c1_axis - r1).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c1 must lie on cyl1: \
                 distance = {dist_c1_axis}, want r1 = {r1}"
            );
            assert!(
                (dist_c2_axis - r2).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c2 must lie on cyl2: \
                 distance = {dist_c2_axis}, want r2 = {r2}"
            );

            // Both contacts on the chamfer plane.
            let on_plane_1 =
                chamfer_normal.dot(Vec3::new(c1_point.x(), c1_point.y(), c1_point.z())) - chamfer_d;
            let on_plane_2 =
                chamfer_normal.dot(Vec3::new(c2_point.x(), c2_point.y(), c2_point.z())) - chamfer_d;
            assert!(
                on_plane_1.abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c1 on chamfer plane: residual {on_plane_1}"
            );
            assert!(
                on_plane_2.abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): c2 on chamfer plane: residual {on_plane_2}"
            );

            // Plane normal perpendicular to z (cyl axes).
            assert!(
                chamfer_normal.z().abs() < 1e-12,
                "({reverse_s1}, {reverse_s2}): plane normal should be perpendicular to z, got {chamfer_normal:?}"
            );
        };

        run_case(false, true); // (s1=+1, s2=-1)
        run_case(true, false); // (s1=-1, s2=+1)
    }

    /// Cone-cone coaxial convex chamfer: two cones sharing an axis with
    /// different half-angles. Chamfer surface is an axisymmetric cone
    /// connecting both cone-generator contact circles.
    ///
    /// Same setup as the cone-cone fillet test (cone1 apex at origin
    /// β=π/3; cone2 apex at z=2 β=π/4; faces NOT reversed; d=0.3):
    ///   - z_spine ≈ 4.732, r_spine ≈ 2.732
    ///   - cone1 contact retreats toward apex1: (r_spine − d·cos β1,
    ///     z_spine − d·sin β1) ≈ (2.582, 4.472)
    ///   - cone2 contact extends away from apex2: (r_spine + d·cos β2,
    ///     z_spine + d·sin β2) ≈ (2.944, 4.944)
    ///   - chamfer cone apex on axis at line P1−P2 extrapolated to r=0
    #[test]
    fn cone_cone_coaxial_chamfer_convex_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let d: f64 = 0.3;

        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())));

        let result = cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("convex coaxial cone-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts (s1=s2=+1).
        let r_c1 = r_spine - d * beta1.cos();
        let z_c1 = z_spine - d * beta1.sin();
        let r_c2 = r_spine + d * beta2.cos();
        let z_c2 = z_spine + d * beta2.sin();

        // Cone1 contact retreats toward apex1 (r and z BOTH decrease).
        assert!(
            r_c1 < r_spine && z_c1 < z_spine,
            "cone1 contact should retreat toward apex1: got ({r_c1}, {z_c1}) vs spine ({r_spine}, {z_spine})"
        );
        // Cone2 contact extends away from apex2 (r and z BOTH increase).
        assert!(
            r_c2 > r_spine && z_c2 > z_spine,
            "cone2 contact should extend away from apex2: got ({r_c2}, {z_c2}) vs spine"
        );

        // Predicted chamfer apex (line extrapolated to r=0).
        let dr = r_c2 - r_c1;
        let dz = z_c2 - z_c1;
        let expected_apex_z = z_c1 - r_c1 * dz / dr;
        let mid_z = 0.5 * (z_c1 + z_c2);
        let r_avg = 0.5 * (r_c1 + r_c2);
        let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {expected_apex_z}",
            apex.z()
        );
        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
            "chamfer half-angle should be {expected_beta}, got {}",
            chamfer_cone.half_angle()
        );

        // Cone axis: contacts above apex (apex z ≈ 1.105 < contacts z ≈ 4.5–5).
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "convex chamfer cone axis should be +z (contacts above apex), got {axis:?}"
        );

        // Both contacts on chamfer cone.
        let want_c1 = Point3::new(r_c1, 0.0, z_c1);
        let want_c2 = Point3::new(r_c2, 0.0, z_c2);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_c1);
        let on_cone_c1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_c2);
        let on_cone_c2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_c1 - want_c1).length() < 1e-9,
            "cone1 contact must lie on chamfer cone: {on_cone_c1:?} vs {want_c1:?}"
        );
        assert!(
            (on_cone_c2 - want_c2).length() < 1e-9,
            "cone2 contact must lie on chamfer cone: {on_cone_c2:?} vs {want_c2:?}"
        );

        // Both contacts on respective cones.
        let cot_b1 = beta1.cos() / beta1.sin();
        let cot_b2 = beta2.cos() / beta2.sin();
        let pred_r_c1 = z_c1 * cot_b1;
        let pred_r_c2 = (z_c2 - h_2) * cot_b2;
        assert!(
            (r_c1 - pred_r_c1).abs() < 1e-9,
            "cone1 contact must lie on cone1: predicted radial {pred_r_c1}, got {r_c1}"
        );
        assert!(
            (r_c2 - pred_r_c2).abs() < 1e-9,
            "cone2 contact must lie on cone2: predicted radial {pred_r_c2}, got {r_c2}"
        );
    }

    /// Cone-cone coaxial both-concave chamfer: two cone-shaped cavities
    /// sharing an axis. Both s_i = −1 ⇒ each contact moves to the
    /// OPPOSITE side along its generator from the convex case.
    ///
    /// Same setup as `cone_cone_coaxial_chamfer_convex_emits_cone` but
    /// with both faces REVERSED. Per-face orientation flips compose:
    /// cone1 contact extends AWAY from apex1 (instead of retreating
    /// toward); cone2 contact retreats TOWARD apex2 (instead of
    /// extending away).
    #[test]
    fn cone_cone_coaxial_chamfer_both_concave_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let d: f64 = 0.3;

        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let cone1 =
            ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                .unwrap();
        let cone2 =
            ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                .unwrap();

        let spine_circle = Circle3D::new(
            Point3::new(0.0, 0.0, z_spine),
            Vec3::new(0.0, 0.0, 1.0),
            r_spine,
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face1 = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Cone(cone1.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face2 = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone2.clone()),
        ));

        let result = cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
            .unwrap()
            .expect("both-concave coaxial cone-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Predicted contacts (s1=s2=-1 ⇒ both signs flip).
        let r_c1 = r_spine + d * beta1.cos();
        let z_c1 = z_spine + d * beta1.sin();
        let r_c2 = r_spine - d * beta2.cos();
        let z_c2 = z_spine - d * beta2.sin();

        // Cone1 contact extends AWAY from apex1 (opposite convex).
        assert!(
            r_c1 > r_spine && z_c1 > z_spine,
            "concave cone1 contact should extend away from apex1: got ({r_c1}, {z_c1}) vs spine ({r_spine}, {z_spine})"
        );
        // Cone2 contact retreats TOWARD apex2 (opposite convex).
        assert!(
            r_c2 < r_spine && z_c2 < z_spine,
            "concave cone2 contact should retreat toward apex2: got ({r_c2}, {z_c2}) vs spine"
        );

        // Predicted chamfer apex (line P1-P2 extrapolated to r=0) and
        // half-angle. Two non-coaxial points on a cone determine apex z
        // and half-angle, but axis direction needs an explicit check.
        let dr = r_c2 - r_c1;
        let dz = z_c2 - z_c1;
        let expected_apex_z = z_c1 - r_c1 * dz / dr;
        let mid_z = 0.5 * (z_c1 + z_c2);
        let r_avg = 0.5 * (r_c1 + r_c2);
        let expected_beta = ((mid_z - expected_apex_z).abs() / r_avg).atan();

        let apex = chamfer_cone.apex();
        assert!(
            apex.x().abs() < 1e-12 && apex.y().abs() < 1e-12,
            "concave apex should be on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "concave apex z = {}, expected {expected_apex_z}",
            apex.z()
        );
        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-9,
            "concave chamfer half-angle should be {expected_beta}, got {}",
            chamfer_cone.half_angle()
        );

        // Cone axis direction (apex below contacts ⇒ axis = +z, opens
        // upward). project_point would not catch a flipped axis with a
        // mirrored apex, so check the axis direction explicitly.
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "concave chamfer cone axis should be +z (apex below contacts), got {axis:?}"
        );

        // Both contacts on chamfer cone.
        let want_c1 = Point3::new(r_c1, 0.0, z_c1);
        let want_c2 = Point3::new(r_c2, 0.0, z_c2);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_c1);
        let on_cone_c1 = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&chamfer_cone, want_c2);
        let on_cone_c2 = ParametricSurface::evaluate(&chamfer_cone, u_q, v_q);
        assert!(
            (on_cone_c1 - want_c1).length() < 1e-9,
            "cone1 contact must lie on chamfer cone: {on_cone_c1:?} vs {want_c1:?}"
        );
        assert!(
            (on_cone_c2 - want_c2).length() < 1e-9,
            "cone2 contact must lie on chamfer cone: {on_cone_c2:?} vs {want_c2:?}"
        );

        // Both contacts on respective cone surfaces.
        let cot_b1 = beta1.cos() / beta1.sin();
        let cot_b2 = beta2.cos() / beta2.sin();
        let pred_r_c1 = z_c1 * cot_b1;
        let pred_r_c2 = (z_c2 - h_2) * cot_b2;
        assert!(
            (r_c1 - pred_r_c1).abs() < 1e-9,
            "cone1 contact must lie on cone1: predicted radial {pred_r_c1}, got {r_c1}"
        );
        assert!(
            (r_c2 - pred_r_c2).abs() < 1e-9,
            "cone2 contact must lie on cone2: predicted radial {pred_r_c2}, got {r_c2}"
        );
    }

    /// Cone-cone coaxial mixed-convexity chamfer: covers BOTH (s1=+1,
    /// s2=−1) and (s1=−1, s2=+1). For each, contacts are on different
    /// generator arms relative to the symmetric cases.
    ///
    /// Reads emitted contact endpoints via `evaluate(t_start)`. Asserts
    /// each emitted contact matches the analytic prediction
    /// `(r_spine ∓ s_i·d·cos β_i, z_spine ∓ s_i·d·sin β_i)` (sharper
    /// than just "lies on cone surface", which a degenerate impl
    /// returning `(r_spine, z_spine)` for both contacts would
    /// trivially satisfy via `r = (z − z_apex)·cot β`).
    #[test]
    fn cone_cone_coaxial_chamfer_mixed_emits_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let beta1: f64 = std::f64::consts::PI / 3.0;
        let beta2: f64 = std::f64::consts::PI / 4.0;
        let h_2: f64 = 2.0;
        let d: f64 = 0.3;
        let sin_minus = (beta1 - beta2).sin();
        let z_spine = h_2 * beta2.cos() * beta1.sin() / sin_minus;
        let r_spine = z_spine * (beta1.cos() / beta1.sin());

        let run_case = |reverse_s1: bool, reverse_s2: bool| {
            let mut topo = Topology::new();
            let cone1 =
                ConicalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), beta1)
                    .unwrap();
            let cone2 =
                ConicalSurface::new(Point3::new(0.0, 0.0, h_2), Vec3::new(0.0, 0.0, 1.0), beta2)
                    .unwrap();
            let spine_circle = Circle3D::new(
                Point3::new(0.0, 0.0, z_spine),
                Vec3::new(0.0, 0.0, 1.0),
                r_spine,
            )
            .unwrap();
            let v = topo.add_vertex(Vertex::new(Point3::new(r_spine, 0.0, z_spine), 1e-7));
            let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
            let spine = Spine::from_single_edge(&topo, eid).unwrap();

            let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
            let face1 = if reverse_s1 {
                topo.add_face(Face::new_reversed(
                    w1,
                    vec![],
                    FaceSurface::Cone(cone1.clone()),
                ))
            } else {
                topo.add_face(Face::new(w1, vec![], FaceSurface::Cone(cone1.clone())))
            };
            let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
            let face2 = if reverse_s2 {
                topo.add_face(Face::new_reversed(
                    w2,
                    vec![],
                    FaceSurface::Cone(cone2.clone()),
                ))
            } else {
                topo.add_face(Face::new(w2, vec![], FaceSurface::Cone(cone2.clone())))
            };

            let result =
                cone_cone_coaxial_chamfer(&cone1, &cone2, &spine, &topo, d, d, face1, face2)
                    .unwrap()
                    .expect("mixed coaxial cone-cone chamfer should produce a stripe");

            // Sample EMITTED contact curves at start of domain.
            let (t1_start, _) = result.stripe.contact1.domain();
            let c1_point = result.stripe.contact1.evaluate(t1_start);
            let (t2_start, _) = result.stripe.contact2.domain();
            let c2_point = result.stripe.contact2.evaluate(t2_start);

            // Compute analytic predictions: contact_i moves along
            // generator_i from spine by s_i·d, retreating toward apex_i
            // when s_i=+1 and extending away when s_i=−1.
            let s1_signed = if reverse_s1 { -1.0_f64 } else { 1.0_f64 };
            let s2_signed = if reverse_s2 { -1.0_f64 } else { 1.0_f64 };
            let pred_c1_r = r_spine - s1_signed * d * beta1.cos();
            let pred_c1_z = z_spine - s1_signed * d * beta1.sin();
            let pred_c2_r = r_spine + s2_signed * d * beta2.cos();
            let pred_c2_z = z_spine + s2_signed * d * beta2.sin();

            let c1_radial = (c1_point.x().powi(2) + c1_point.y().powi(2)).sqrt();
            let c2_radial = (c2_point.x().powi(2) + c2_point.y().powi(2)).sqrt();
            assert!(
                (c1_radial - pred_c1_r).abs() < 1e-9 && (c1_point.z() - pred_c1_z).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact1 should be at (r={pred_c1_r}, z={pred_c1_z}); \
                 got (r={c1_radial}, z={})",
                c1_point.z()
            );
            assert!(
                (c2_radial - pred_c2_r).abs() < 1e-9 && (c2_point.z() - pred_c2_z).abs() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact2 should be at (r={pred_c2_r}, z={pred_c2_z}); \
                 got (r={c2_radial}, z={})",
                c2_point.z()
            );

            // Emitted surface is a Cone with axis = −z (apex ABOVE
            // contacts). Distinct from the symmetric cases which have
            // axis = +z: in mixed configs both contacts retreat
            // /extend along generators with the SAME radial sign
            // (one s_i flips), so the chord between them slopes the
            // OPPOSITE way and the line P1−P2 extrapolates to r=0
            // ABOVE the contacts rather than below.
            let chamfer_cone = match result.stripe.surface {
                FaceSurface::Cone(ref c) => c,
                ref other => panic!(
                    "({reverse_s1}, {reverse_s2}): expected Cone, got {}",
                    other.type_tag()
                ),
            };
            let axis = chamfer_cone.axis();
            assert!(
                axis.dot(Vec3::new(0.0, 0.0, 1.0)) < -1.0 + 1e-12,
                "({reverse_s1}, {reverse_s2}): chamfer cone axis should be −z (mixed = apex above), got {axis:?}"
            );

            // Both contacts on the chamfer cone via project_point
            // round-trip (the impl chose them on its own surface, so
            // this is a regression-check only).
            let (u_p, v_p) = ParametricSurface::project_point(chamfer_cone, c1_point);
            let on_cone_p1 = ParametricSurface::evaluate(chamfer_cone, u_p, v_p);
            assert!(
                (on_cone_p1 - c1_point).length() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact1 must lie on chamfer cone"
            );
            let (u_q, v_q) = ParametricSurface::project_point(chamfer_cone, c2_point);
            let on_cone_p2 = ParametricSurface::evaluate(chamfer_cone, u_q, v_q);
            assert!(
                (on_cone_p2 - c2_point).length() < 1e-9,
                "({reverse_s1}, {reverse_s2}): contact2 must lie on chamfer cone"
            );
        };

        run_case(false, true); // (s1=+1, s2=-1)
        run_case(true, false); // (s1=-1, s2=+1)
    }

    /// Sphere-cylinder both-concave fillet: spherical cavity intersecting
    /// a cylindrical hole. Both faces REVERSED ⇒ Q_sph = R_s − r,
    /// Q_cyl = r_c − r (internal tangency on both surfaces).
    ///
    /// For R_s=3, r_c=2 (sphere encloses cyl), both faces REVERSED,
    /// r=0.4, +z spine:
    ///   - Q_sph = 2.6, Q_cyl = 1.6
    ///   - a_ball = √(Q_sph² − Q_cyl²) = √4.2 ≈ 2.049
    ///   - major = Q_cyl = 1.6 (concave fillet sits INSIDE cyl axially)
    ///   - Major < r_c (vs convex which has major = r_c + r > r_c) —
    ///     confirms internal-tangency reduction.
    #[test]
    fn sphere_cylinder_fillet_both_concave_emits_smaller_torus() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::{CylindricalSurface, SphericalSurface};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let big_r_s: f64 = 3.0;
        let r_c: f64 = 2.0;
        let r_fillet: f64 = 0.4;
        let h_s = (big_r_s * big_r_s - r_c * r_c).sqrt();

        let sph = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), big_r_s).unwrap();
        let cyl =
            CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_c)
                .unwrap();
        let spine_circle =
            Circle3D::new(Point3::new(0.0, 0.0, h_s), Vec3::new(0.0, 0.0, 1.0), r_c).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(r_c, 0.0, h_s), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(spine_circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        // Both faces REVERSED.
        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_sphere = topo.add_face(Face::new_reversed(
            w1,
            vec![],
            FaceSurface::Sphere(sph.clone()),
        ));
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cyl = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cylinder(cyl.clone()),
        ));

        let result =
            sphere_cylinder_fillet(&sph, &cyl, &spine, &topo, r_fillet, face_sphere, face_cyl)
                .unwrap()
                .expect("both-concave sphere-cylinder fillet should produce a stripe");

        let torus = match result.stripe.surface {
            FaceSurface::Torus(t) => t,
            other => panic!("expected Torus, got {}", other.type_tag()),
        };

        let q_sph = big_r_s - r_fillet;
        let q_cyl = r_c - r_fillet;
        let expected_a_ball = (q_sph * q_sph - q_cyl * q_cyl).sqrt();
        let expected_major = q_cyl;

        assert!(
            (torus.major_radius() - expected_major).abs() < 1e-12,
            "concave major should be Q_cyl = {expected_major}, got {}",
            torus.major_radius()
        );
        assert!(
            torus.major_radius() < r_c,
            "concave major ({}) must be < r_c = {r_c} (vs convex which would be > r_c)",
            torus.major_radius()
        );
        assert!(
            (torus.minor_radius() - r_fillet).abs() < 1e-12,
            "minor should equal r_fillet = {r_fillet}, got {}",
            torus.minor_radius()
        );

        // Torus center on +z axis at z = a_ball (positive since +z spine).
        let center = torus.center();
        assert!(
            center.x().abs() < 1e-12 && center.y().abs() < 1e-12,
            "torus center on z-axis, got {center:?}"
        );
        assert!(
            (center.z() - expected_a_ball).abs() < 1e-12,
            "torus center z should be a_ball = {expected_a_ball}, got {}",
            center.z()
        );

        // Verify ball is INSIDE both surfaces (internal tangency).
        let actual_dist_to_sphere_center =
            (center.x().powi(2) + center.y().powi(2) + center.z().powi(2)).sqrt();
        assert!(
            actual_dist_to_sphere_center < big_r_s - 1e-9,
            "concave: ball must be INSIDE sphere (distance {actual_dist_to_sphere_center} < R_s = {big_r_s})"
        );
        let actual_dist_to_cyl_axis = (center.x().powi(2) + center.y().powi(2)).sqrt();
        assert!(
            actual_dist_to_cyl_axis < r_c - 1e-9,
            "concave: ball must be INSIDE cyl (distance {actual_dist_to_cyl_axis} < r_c = {r_c})"
        );

        // Sphere contact at distance R_s from sphere center.
        let sph_axial = big_r_s * expected_a_ball / q_sph;
        let sph_radial = big_r_s * expected_major / q_sph;
        let want_sph = Point3::new(sph_radial, 0.0, sph_axial);
        let dist_sph = (want_sph - Point3::new(0.0, 0.0, 0.0)).length();
        assert!(
            (dist_sph - big_r_s).abs() < 1e-9,
            "sphere contact must lie on sphere: {dist_sph} vs R_s = {big_r_s}"
        );

        // Cylinder contact at radial r_c.
        let want_cyl = Point3::new(r_c, 0.0, expected_a_ball);
        let cyl_radial = (want_cyl.x().powi(2) + want_cyl.y().powi(2)).sqrt();
        assert!(
            (cyl_radial - r_c).abs() < 1e-9,
            "cyl contact must have radial r_c: got {cyl_radial}, want {r_c}"
        );

        // Both contacts lie on the torus.
        let (u_p, v_p) = ParametricSurface::project_point(&torus, want_sph);
        let on_torus_sph = ParametricSurface::evaluate(&torus, u_p, v_p);
        let (u_q, v_q) = ParametricSurface::project_point(&torus, want_cyl);
        let on_torus_cyl = ParametricSurface::evaluate(&torus, u_q, v_q);
        assert!(
            (on_torus_sph - want_sph).length() < 1e-9,
            "sphere contact on torus: {on_torus_sph:?} vs {want_sph:?}"
        );
        assert!(
            (on_torus_cyl - want_cyl).length() < 1e-9,
            "cyl contact on torus: {on_torus_cyl:?} vs {want_cyl:?}"
        );
    }

    /// Concave plane-cone chamfer: chamfering the top rim of a tapered hole.
    ///
    /// Geometry: cone primitive (apex above plate at z=h, axis −z,
    /// half-angle α) used as a hole tool through a plate at z=0. The
    /// resulting solid has plate material at z<0 and the hole wall is the
    /// (reversed) cone primitive's lateral face. We chamfer the spine where
    /// hole wall meets plate top.
    ///
    /// Detection: `axis_c · n_p_inward = (−z)·(+z) = −1` (antiparallel),
    /// so `signed_offset = −1`. The chamfer cone's apex sits BELOW the
    /// plate (z<0), its axis is +z, opening upward through the plate into
    /// the empty wedge above the chamfer.
    ///
    /// For α = π/4, h = 4 (so r_p = 4), and symmetric d1 = d2 = 1:
    ///   chamfer half-angle β = π/2 − α/2 = 3π/8 (independent of concave/convex)
    ///   plate-side contact at radius r_p + d1 = 5, z = 0
    ///   cone-side contact at radius r_p + d2·cos α ≈ 4.707, z = −d2·sin α ≈ −0.707
    #[test]
    fn plane_cone_chamfer_concave_emits_chamfer_cone() {
        use brepkit_math::curves::Circle3D;
        use brepkit_math::surfaces::ConicalSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let alpha: f64 = std::f64::consts::FRAC_PI_4;
        let h: f64 = 4.0;
        let r_p: f64 = h * (alpha.cos() / alpha.sin());
        let d: f64 = 1.0;

        let v = topo.add_vertex(Vertex::new(Point3::new(r_p, 0.0, 0.0), 1e-7));
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r_p).unwrap();
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let spine = Spine::from_single_edge(&topo, eid).unwrap();

        let w1 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face_plate = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        // Cone primitive: apex at (0,0,h), axis = −z, half-angle α. Used
        // here as the wall of a hole, so the FACE is reversed (topological
        // outward points into the empty hole).
        let cone_surf =
            ConicalSurface::new(Point3::new(0.0, 0.0, h), Vec3::new(0.0, 0.0, -1.0), alpha)
                .unwrap();
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, false)], true).unwrap());
        let face_cone = topo.add_face(Face::new_reversed(
            w2,
            vec![],
            FaceSurface::Cone(cone_surf.clone()),
        ));

        let n_p_inward = Vec3::new(0.0, 0.0, 1.0);
        let result = plane_cone_chamfer(
            n_p_inward, 0.0, &cone_surf, &spine, &topo, d, d, face_plate, face_cone,
        )
        .unwrap()
        .expect("concave plane-cone chamfer should produce a stripe");

        let chamfer_cone = match result.stripe.surface {
            FaceSurface::Cone(c) => c,
            other => panic!("expected Cone, got {}", other.type_tag()),
        };

        // Symmetric chamfer half-angle: β = π/2 − α/2 = 3π/8.
        let expected_beta = std::f64::consts::FRAC_PI_2 - alpha * 0.5;
        assert!(
            (chamfer_cone.half_angle() - expected_beta).abs() < 1e-12,
            "chamfer cone half-angle should be 3π/8 for α=π/4, d1=d2; got {}",
            chamfer_cone.half_angle()
        );

        // Apex BELOW the plate at z = −(r_p+d) · dz/dr where dz = sin α,
        // dr = 1 − cos α (with d=1). For α=π/4: dz=√2/2, dr=1−√2/2,
        // ratio ≈ 2.414, mag ≈ 5·2.414 ≈ 12.07.
        let apex = chamfer_cone.apex();
        let dz = alpha.sin();
        let dr = 1.0 - alpha.cos();
        let expected_apex_z = -(r_p + d) * dz / dr;
        assert!(
            (apex.x()).abs() < 1e-12 && (apex.y()).abs() < 1e-12,
            "apex should lie on z-axis, got {apex:?}"
        );
        assert!(
            (apex.z() - expected_apex_z).abs() < 1e-9,
            "apex z = {}, expected {}",
            apex.z(),
            expected_apex_z
        );

        // Chamfer cone axis = +z (opens upward through the plate).
        let axis = chamfer_cone.axis();
        assert!(
            axis.dot(Vec3::new(0.0, 0.0, 1.0)) > 1.0 - 1e-12,
            "chamfer cone axis should be +z, got {axis:?}"
        );

        // Both contact points must lie on the chamfer cone surface. We
        // project each onto the chamfer cone and verify the resulting
        // surface point matches to high precision; this avoids depending
        // on the exact frame orientation chosen by `Frame3::from_normal`.
        let want_plate = Point3::new(r_p + d, 0.0, 0.0);
        let cone_contact_axial = -d * alpha.sin();
        let cone_contact_radial = r_p + d * alpha.cos();
        let want_cone = Point3::new(cone_contact_radial, 0.0, cone_contact_axial);
        let (u_p, v_p) = ParametricSurface::project_point(&chamfer_cone, want_plate);
        let on_surf_plate = ParametricSurface::evaluate(&chamfer_cone, u_p, v_p);
        let (u_c, v_c) = ParametricSurface::project_point(&chamfer_cone, want_cone);
        let on_surf_cone = ParametricSurface::evaluate(&chamfer_cone, u_c, v_c);
        assert!(
            (on_surf_plate - want_plate).length() < 1e-9,
            "plate contact must lie on chamfer cone: project→eval gave {on_surf_plate:?}, want {want_plate:?}"
        );
        assert!(
            (on_surf_cone - want_cone).length() < 1e-9,
            "cone-side contact must lie on chamfer cone: project→eval gave {on_surf_cone:?}, want {want_cone:?}"
        );
    }
}

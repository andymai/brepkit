// Walking engine infrastructure — used progressively as more blend paths are wired up.
#![allow(dead_code)]
//! Analytic fast paths for common surface pairs.
//!
//! Closed-form fillet and chamfer solutions for plane-plane and plane-cylinder
//! surface pairs. These bypass the walking engine entirely, producing exact
//! geometry that is 10-100x faster than Newton-Raphson marching.
//!
//! About 80% of real-world fillets are between plane-plane or plane-cylinder
//! pairs, making these fast paths high-impact optimizations.

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
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_),
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
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_),
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

    let tol_ang = 1e-9;
    let tol_lin = 1e-9;

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

    let tol_ang = 1e-9;
    let tol_lin = 1e-9;

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

    let tol_ang = 1e-9;
    let tol_lin = 1e-9;

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

    let tol_ang = 1e-9;
    let tol_lin = 1e-9;

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

    let tol_lin = 1e-9;
    let tol_ang = 1e-9;

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

    let tol_lin = 1e-9;
    let tol_ang = 1e-9;

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

/// Fillet between two cylinders.
///
/// Not yet implemented. Returns `None` so the caller falls back to the walking
/// engine.
#[allow(unused_variables)]
#[must_use]
pub fn cylinder_cylinder_fillet(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
) -> Option<StripeResult> {
    // TODO: implement cylinder-cylinder fillet
    None
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

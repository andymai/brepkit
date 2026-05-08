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
                std::mem::swap(&mut r.stripe.face1, &mut r.stripe.face2);
                std::mem::swap(&mut r.stripe.pcurve1, &mut r.stripe.pcurve2);
                std::mem::swap(&mut r.stripe.contact1, &mut r.stripe.contact2);
                for s in &mut r.stripe.sections {
                    std::mem::swap(&mut s.p1, &mut s.p2);
                    std::mem::swap(&mut s.uv1, &mut s.uv2);
                }
            }
            Ok(result)
        }
        // Plane-Cylinder/Cylinder-Plane is handled above; the catch-all
        // covers everything else (no analytic path yet → walker fallback).
        _ => Ok(None),
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
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
        )
        | (
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            FaceSurface::Plane { .. },
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

/// Fillet between a plane and a cylinder whose axis is parallel to the plane
/// normal.
///
/// Returns `Some(StripeResult)` with an exact toroidal blend surface for the
/// convex perpendicular case (typical "post on a plate" geometry). Returns
/// `None` for any case the analytic path doesn't yet cover; the caller then
/// falls back to the walking engine. Specifically, `None` is returned when:
///   - the cylinder axis is not parallel to the plane normal,
///   - the cylinder face is reversed in the topology (concave / "hole through
///     plate" geometry — the fillet rolls inside the hole, requiring a
///     different torus major radius and tube quadrant),
///   - the spine geometry is too short or degenerate,
///   - or the fillet radius exceeds the cylinder radius (would invert R).
///
/// The torus has axis parallel to the cylinder axis, major radius `r_c + r`,
/// and minor radius `r`. The active tube portion is `v ∈ [π, 3π/2]` — the
/// quarter that touches the plane below and the cylinder lateral inward.
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

    // 2) Concave (hole-through-plate) case requires a different torus major
    //    radius (`r_c - r`) and tube quadrant. Defer those to the walker.
    if topo.face(face_cyl)?.is_reversed() {
        return Ok(None);
    }

    // 3) The radius must shrink the cylinder rather than invert it. Larger
    //    fillet radii than the cylinder can't form a clean torus.
    let r_c = cyl.radius();
    if radius <= tol_lin || radius >= r_c {
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
    //    along the side opposite the plane material — i.e. `-n_p_inward`.
    //    Its axis is the cylinder axis (kept as-is for parametric clarity).
    let torus_center = p_axis_on_plane - n_p_inward * radius;
    let major_radius = r_c + radius;
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
    // The plane contact is always an arc on a circle (the rolling-ball
    // trajectory at z=0), so represent the pcurve as `Curve2D::Circle` in
    // the plane's local frame. A line-segment pcurve would zero out for the
    // closed-spine case (start and end project to the same point).
    let pcurve_plane = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n_p_inward, d_plane);
        let (cu, cv) = adapter.project_point(p_axis_on_plane);
        Curve2D::Circle(brepkit_math::curves2d::Circle2D::new(
            brepkit_math::vec::Point2::new(cu, cv),
            major_radius,
        )?)
    };
    let pcurve_cyl = Curve2D::Line(Line2D::new(
        brepkit_math::vec::Point2::new(u_start, v_cyl),
        brepkit_math::vec::Vec2::new(u_end - u_start, 0.0),
    )?);

    // 11) Cross-sections at the spine endpoints. Each section's two contact
    //     points and the rolling-ball center sit on a small circle of the
    //     torus tube — they share the section's `u` and span the active `v`
    //     quadrant from plane (v=3π/2) to cylinder (v=π).
    let p_plane_at = |u: f64| contact_plane_circle.evaluate(u);
    let p_cyl_at = |u: f64| contact_cyl_circle.evaluate(u);
    let center_at = |u: f64| {
        // Ball trajectory: same circle as `contact_plane_circle` but lifted
        // to the height of the cylinder contact (axial offset `r`).
        contact_plane_circle.evaluate(u) + z_axis_dir * radius
    };
    let section_start = CircSection {
        p1: p_plane_at(u_start),
        p2: p_cyl_at(u_start),
        center: center_at(u_start),
        radius,
        uv1: (u_start, v_cyl), // unused for plane
        uv2: (u_start, v_cyl),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: p_plane_at(u_end),
        p2: p_cyl_at(u_end),
        center: center_at(u_end),
        radius,
        uv1: (u_end, v_cyl),
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

/// Fillet between a plane and a cone.
///
/// Not yet implemented. Returns `None` so the caller falls back to the walking
/// engine.
#[allow(unused_variables)]
#[must_use]
pub fn plane_cone_fillet(
    surf_plane: &FaceSurface,
    surf_cone: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
) -> Option<StripeResult> {
    // TODO: implement plane-cone fillet
    None
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
}

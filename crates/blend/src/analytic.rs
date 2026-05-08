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
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
        )
        | (
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
        )
        | (
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
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
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
        )
        | (
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
            FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
        )
        | (
            FaceSurface::Sphere(_) | FaceSurface::Torus(_) | FaceSurface::Nurbs(_),
            FaceSurface::Plane { .. } | FaceSurface::Cylinder(_) | FaceSurface::Cone(_),
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

    // 2) Concave (cylinder face reversed = hole through plate) needs a
    //    different chamfer geometry; defer.
    if topo.face(face_cyl)?.is_reversed() {
        return Ok(None);
    }

    // 3) Both distances must be positive, and d1 must leave a non-negative
    //    radius on the plate (otherwise the contact circle would have to
    //    pass through the cylinder axis or beyond).
    let r_c = cyl.radius();
    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }
    if d1 >= r_c {
        return Ok(None);
    }

    // 4) Project the cylinder origin onto the plate.
    let o_c = cyl.origin();
    let step = d_plane - n_p_inward.dot(Vec3::new(o_c.x(), o_c.y(), o_c.z()));
    let p_axis_on_plane = o_c + n_p_inward * step;

    // 5) Establish a "spine-to-empty-wedge" axial direction. The chamfer
    //    dispatcher (unlike the fillet dispatcher) does NOT apply
    //    `orient_plane_surface`, so `n_p_inward` here is actually the
    //    face's raw geometric normal — for a non-reversed bottom-cap face
    //    of an upright cylinder it points OUTWARD (-z), away from the
    //    cylinder material. Two distinct directions matter below:
    //      * `axis_toward_apex = +n_p_inward` — toward the chamfer cone's
    //        virtual apex, which lives in the empty-wedge half-space;
    //      * `axis_toward_material = -n_p_inward` — toward the cylinder
    //        body, where the cylinder-side contact circle lives.
    //    The cone surface is parameterized so that its apex is on the
    //    empty-wedge side and its `+axis_c` direction points toward the
    //    plate, so the cone evaluation walks INTO the cylinder material as
    //    `v` increases.
    let axis_toward_apex = n_p_inward;
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

    // 7) Build the chamfer cone.
    //    The cone opens in `axis_toward_material` (= -n_p_inward), with its
    //    apex on the empty-wedge side of the plate. As `v` grows from 0 at
    //    the apex, the cone first sweeps to the plate at `v = (r_c - d1) /
    //    cos(α)`, then to the cylinder lateral at `v = r_c / cos(α)`.
    // brepkit's `ConicalSurface` measures `half_angle` from the AXIS to
    // the generator, so the axial component per unit v is `sin(β)` and the
    // radial component is `cos(β)`. Generator slope `dr/dz = cos β / sin β
    // = cot β`, and our generator goes from `(r_c - d1, 0)` to `(r_c, +d2)`
    // with slope `d1 / d2`, giving `β = atan2(d2, d1)`. (For symmetric
    // `d1 = d2` either ordering gives π/4, but the asymmetric case needs
    // this convention to match the cone surface.)
    let half_angle = d2.atan2(d1);
    // Apex axial offset from the plate: derived from similar triangles —
    // the cone's generator slopes (d1 : d2) and the apex sits where the
    // generator extension hits r=0. Apex is on the empty-wedge side so
    // the cone opens "through" the plate into the cylinder material as v
    // increases.
    let apex_offset = (r_c - d1) * d2 / d1;
    let apex_pos = p_axis_on_plane + axis_toward_apex * apex_offset;
    let cone_axis = axis_toward_material;
    let cyl_x = cyl.x_axis();
    let cone = ConicalSurface::with_ref_dir(apex_pos, cone_axis, half_angle, cyl_x)?;

    // 8) 3D contact curves: both are circles around the cylinder axis.
    let cone_y = cyl.y_axis();
    let contact_plane_circle = brepkit_math::curves::Circle3D::with_axes(
        p_axis_on_plane,
        axis_c,
        r_c - d1,
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
/// along the outward bisector `cos(α/2)·radial - sin(α/2)·n_p_inward`.
/// The ball center lands at radial `r_p + r·cot(α/2)` and offset
/// `r·(-n_p_inward)` from the plate (one fillet radius in the
/// empty-wedge direction).
///
/// The fillet surface is a torus:
///   - axis = `-n_p_inward` (= `+axis_c` for the regular-frustum case
///     where `axis_c · n_p_inward = -1`); with this convention `sin v`
///     points away from the plate, so `sin v = -1` is what pulls the
///     tube point back toward the plate. Plate contact lands at
///     `v = 3π/2`; cone contact at `v = atan2(cos α, -sin α)`.
///   - center at the cone-axis projection onto the plate, offset by
///     `-r·n_p_inward`;
///   - major radius `r_p + r·cot(α/2)`,
///   - minor radius `r`;
///   - active tube parameter `v ∈ [atan2(cos α, -sin α), 3π/2]`,
///     width `π - α`.
///
/// At α = π/2 (degenerate "cone" approaching a cylinder), `cot(π/4) = 1`
/// so major reduces to `r_p + r` and the active range becomes
/// `[π, 3π/2]` — exactly the plane-cylinder result.
///
/// Returns `None` when:
///   - the cone axis isn't parallel to the plane normal,
///   - `axis_c · n_p_inward > -1 + tol_ang` (cone opens *away* from the
///     plate — inverted-frustum or cup geometry; the major-radius formula
///     differs and is left to the walker),
///   - the cone face is reversed (concave / "tapered hole" geometry),
///   - the half-angle α is too close to 0 or π/2 (degenerate),
///   - the spine is too short, or
///   - the apex is on the plate-material side.
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

    // 1) Cone axis must be antiparallel to the inward plane normal — this
    //    means the cone opens TOWARD the plate (regular frustum bottom-rim
    //    geometry). Inverted frustums (axes parallel) need a different
    //    formula and are deferred to the walker.
    let axis_c = cone.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot > -1.0 + tol_ang {
        return Ok(None);
    }

    // 2) Concave (cone face reversed = "tapered hole through plate") needs a
    //    different torus quadrant and major-radius formula; defer.
    if topo.face(face_cone)?.is_reversed() {
        return Ok(None);
    }

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

    // 4) Apex projection onto the plate. `step` is the signed distance you
    //    move along `n_p_inward` from the apex to land on the plate. For a
    //    regular-frustum bottom-rim geometry, the apex lies on the
    //    `+n_p_inward` side (above the plate material in the test setup),
    //    so `step` is negative. Reject anything else (apex on the plate or
    //    on the inverted side — the formulas below assume the
    //    bottom-rim configuration).
    let apex = cone.apex();
    let step = d_plane - n_p_inward.dot(Vec3::new(apex.x(), apex.y(), apex.z()));
    if step >= -tol_lin {
        return Ok(None);
    }
    let apex_height = -step;
    let p_axis_on_plane = apex + n_p_inward * step;

    // 5) Spine radius `r_p = apex_height · cot(α)` (geometric: the cone-plate
    //    intersection circle has this radius).
    let r_p = apex_height * (alpha.cos() / alpha.sin());

    // 6) Major / minor radii and torus center.
    let major_radius = r_p + radius * cot_half;
    let minor_radius = radius;
    if major_radius <= tol_lin {
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
    //     Cone contact: circle of radius `r_p + r·cot(α/2) - r·sin(α)`
    //       at axial offset `-r·(1 + cos(α))` (on the analytical cone
    //       surface, extended below the frustum's base).
    let contact_plane_radius = major_radius;
    let contact_cone_radius = (major_radius - radius * alpha.sin()).max(tol_lin);
    let contact_cone_axial_offset = -radius * (1.0 + alpha.cos());
    let cone_contact_center = p_axis_on_plane + (-n_p_inward) * (-contact_cone_axial_offset);
    // Equivalently: p_axis_on_plane + n_p_inward * (radius * (1.0 + cos(α)))
    // — i.e. *into* plate material. Written via `-n_p_inward * |offset|`
    // for symmetry with the torus_center formula above.

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

    // 1) Cone axis must be parallel to the raw plate normal — for a
    //    regular frustum bottom-rim the cone axis (= "direction the cone
    //    opens", -z for `make_cone(big, small, h)`) is parallel to the
    //    bottom cap's outward normal (also -z). Note: the chamfer
    //    dispatcher does NOT apply `orient_plane_surface`, so `n_p_inward`
    //    here is actually the face's raw geometric (outward) normal —
    //    detection therefore tests for parallel rather than the
    //    fillet-convention antiparallel.
    let axis_c = cone.axis();
    let n_dot = axis_c.dot(n_p_inward);
    if n_dot < 1.0 - tol_ang {
        return Ok(None);
    }

    // 2) Concave (cone face reversed) needs a different formulation.
    if topo.face(face_cone)?.is_reversed() {
        return Ok(None);
    }

    // 3) Validate half-angle and chamfer distances.
    let alpha = cone.half_angle();
    if alpha <= 1e-3 || alpha >= std::f64::consts::FRAC_PI_2 - 1e-3 {
        return Ok(None);
    }
    if d1 <= tol_lin || d2 <= tol_lin {
        return Ok(None);
    }

    // 4) Apex projection onto the plate. For frustum bottom-rim geometry
    //    with the chamfer's raw-normal convention `step` is positive
    //    (apex sits on the +n_p_inward side of the plate, since
    //    `n_p_inward` here is the outward normal pointing away from the
    //    cylinder material).
    let apex = cone.apex();
    let step = d_plane - n_p_inward.dot(Vec3::new(apex.x(), apex.y(), apex.z()));
    if step <= tol_lin {
        return Ok(None);
    }
    let apex_height = step;
    let p_axis_on_plane = apex + n_p_inward * step;

    // 5) Spine radius from cone-plate intersection.
    let r_p = apex_height * (alpha.cos() / alpha.sin());
    if d1 >= r_p {
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
    //    plate-side contact backward to the axis.
    let chamfer_apex_offset = (r_p - d1) * dz / dr;
    let axis_toward_apex = n_p_inward;
    let axis_toward_material = -n_p_inward;
    let chamfer_apex_pos = p_axis_on_plane + axis_toward_apex * chamfer_apex_offset;
    let chamfer_axis = axis_toward_material;

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
    let plate_contact_radius = r_p - d1;
    let cone_contact_radius = r_p - d2 * cos_a;
    let cone_contact_axial_offset = d2 * sin_a;
    let cone_contact_center = p_axis_on_plane + axis_toward_material * cone_contact_axial_offset;

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

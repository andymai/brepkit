//! Surface analysis — singularity, closure, degeneracy, equivalence.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::face::FaceSurface;

use crate::status::Status;

/// Result of analyzing a face surface.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct SurfaceAnalysis {
    /// Whether the surface is closed (periodic) in U.
    pub is_closed_u: bool,
    /// Whether the surface is closed (periodic) in V.
    pub is_closed_v: bool,
    /// Whether the surface has a singularity at the U-min boundary.
    pub has_singularity_u_min: bool,
    /// Whether the surface has a singularity at the U-max boundary.
    pub has_singularity_u_max: bool,
    /// Whether the surface has a singularity at the V-min boundary.
    pub has_singularity_v_min: bool,
    /// Whether the surface has a singularity at the V-max boundary.
    pub has_singularity_v_max: bool,
    /// Whether the surface is degenerate (zero area).
    pub is_degenerate: bool,
    /// Outcome status flags.
    pub status: Status,
}

/// Analyze a face surface for singularities, closure, and degeneracy.
///
/// Analytic surfaces have known characteristics:
/// - Cylinder/Cone/Sphere/Torus are closed in U (periodic 2pi).
/// - Sphere has singularities at V = -pi/2 (south pole) and V = pi/2
///   (north pole).
/// - Cone has a singularity at V = 0 (apex).
/// - Torus is closed in both U and V.
///
/// For NURBS surfaces, closure is checked by comparing control points
/// at opposite boundaries.
#[must_use]
pub fn analyze_surface(surface: &FaceSurface, tolerance: &Tolerance) -> SurfaceAnalysis {
    match surface {
        FaceSurface::Plane { .. } => SurfaceAnalysis {
            is_closed_u: false,
            is_closed_v: false,
            has_singularity_u_min: false,
            has_singularity_u_max: false,
            has_singularity_v_min: false,
            has_singularity_v_max: false,
            is_degenerate: false,
            status: Status::OK,
        },
        FaceSurface::Cylinder(_) => SurfaceAnalysis {
            is_closed_u: true,
            is_closed_v: false,
            has_singularity_u_min: false,
            has_singularity_u_max: false,
            has_singularity_v_min: false,
            has_singularity_v_max: false,
            is_degenerate: false,
            status: Status::OK,
        },
        FaceSurface::Cone(_) => SurfaceAnalysis {
            is_closed_u: true,
            is_closed_v: false,
            has_singularity_u_min: false,
            has_singularity_u_max: false,
            // Cone apex is at v=0.
            has_singularity_v_min: true,
            has_singularity_v_max: false,
            is_degenerate: false,
            status: Status::DONE1,
        },
        FaceSurface::Sphere(_) => SurfaceAnalysis {
            is_closed_u: true,
            is_closed_v: false,
            has_singularity_u_min: false,
            has_singularity_u_max: false,
            // Sphere poles at v = -pi/2 and v = pi/2.
            has_singularity_v_min: true,
            has_singularity_v_max: true,
            is_degenerate: false,
            status: Status::DONE1,
        },
        FaceSurface::Torus(_) => SurfaceAnalysis {
            is_closed_u: true,
            is_closed_v: true,
            has_singularity_u_min: false,
            has_singularity_u_max: false,
            has_singularity_v_min: false,
            has_singularity_v_max: false,
            is_degenerate: false,
            status: Status::OK,
        },
        FaceSurface::Nurbs(n) => analyze_nurbs_surface(n, tolerance),
    }
}

/// Analyze a NURBS surface for closure and degeneracy.
fn analyze_nurbs_surface(
    n: &brepkit_math::nurbs::surface::NurbsSurface,
    tolerance: &Tolerance,
) -> SurfaceAnalysis {
    let cps = n.control_points();
    let n_rows = cps.len();
    let n_cols = if n_rows > 0 { cps[0].len() } else { 0 };

    let tol = tolerance.linear;

    // Check U-closure: first row vs last row.
    let is_closed_u = n_rows >= 2
        && n_cols > 0
        && cps[0]
            .iter()
            .zip(cps[n_rows - 1].iter())
            .all(|(a, b)| (*a - *b).length() < tol);

    // Check V-closure: first column vs last column.
    let is_closed_v = n_cols >= 2
        && cps
            .iter()
            .all(|row| (row[0] - row[n_cols - 1]).length() < tol);

    // Check for singularities: all control points in a row/column collapse
    // to a single point.
    let has_singularity_v_min = n_cols > 0 && n_rows > 0 && {
        let p0 = cps[0][0];
        cps[0].iter().all(|p| (*p - p0).length() < tol)
    };

    let has_singularity_v_max = n_cols > 0 && n_rows > 0 && {
        let p0 = cps[n_rows - 1][0];
        cps[n_rows - 1].iter().all(|p| (*p - p0).length() < tol)
    };

    let has_singularity_u_min = n_rows > 0 && n_cols > 0 && {
        let p0 = cps[0][0];
        cps.iter().all(|row| (row[0] - p0).length() < tol)
    };

    let has_singularity_u_max = n_rows > 0 && n_cols > 0 && {
        let p0 = cps[0][n_cols - 1];
        cps.iter().all(|row| (row[n_cols - 1] - p0).length() < tol)
    };

    // Degenerate if all control points collapse to a single point.
    let is_degenerate = n_rows > 0 && n_cols > 0 && {
        let p0 = cps[0][0];
        cps.iter()
            .flat_map(|row| row.iter())
            .all(|p| (*p - p0).length() < tol)
    };

    let mut status = Status::OK;
    if has_singularity_u_min
        || has_singularity_u_max
        || has_singularity_v_min
        || has_singularity_v_max
    {
        status = status.merge(Status::DONE1);
    }
    if is_degenerate {
        status = status.merge(Status::DONE2);
    }

    SurfaceAnalysis {
        is_closed_u,
        is_closed_v,
        has_singularity_u_min,
        has_singularity_u_max,
        has_singularity_v_min,
        has_singularity_v_max,
        is_degenerate,
        status,
    }
}

/// Check if two face surfaces are geometrically equivalent.
///
/// Two surfaces are equivalent if they represent the same infinite surface
/// (e.g., same plane, same cylinder axis/radius). NURBS surfaces are
/// conservatively treated as never equivalent.
#[must_use]
pub fn surfaces_equivalent(a: &FaceSurface, b: &FaceSurface, tolerance: &Tolerance) -> bool {
    let lin = tolerance.linear;
    let ang = tolerance.angular;

    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            // Relaxed tolerance for plane comparison (mesh-derived coplanar faces).
            let plane_ang = 1e-4_f64;
            let plane_lin = 1e-3_f64;
            let dot = na.dot(*nb);
            (dot.abs() - 1.0).abs() < plane_ang && (da - db * dot.signum()).abs() < plane_lin
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            (ca.radius() - cb.radius()).abs() < lin
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.origin() - ca.origin();
                    d.cross(ca.axis()).length_squared() < lin * lin
                }
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            (ca.half_angle() - cb.half_angle()).abs() < ang
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.apex() - ca.apex();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            (sa.radius() - sb.radius()).abs() < lin && {
                let d = sb.center() - sa.center();
                d.dot(d) < lin * lin
            }
        }
        (FaceSurface::Torus(ta), FaceSurface::Torus(tb)) => {
            (ta.major_radius() - tb.major_radius()).abs() < lin
                && (ta.minor_radius() - tb.minor_radius()).abs() < lin
                && ta.z_axis().dot(tb.z_axis()).abs() > 1.0 - ang
                && {
                    let d = tb.center() - ta.center();
                    d.dot(d) < lin * lin
                }
        }
        // Different surface types or NURBS are never equivalent.
        (
            FaceSurface::Plane { .. }
            | FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_)
            | FaceSurface::Nurbs(_),
            _,
        ) => false,
    }
}

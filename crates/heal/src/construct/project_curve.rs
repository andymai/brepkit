//! Curve projection onto surfaces.
//!
//! Projects 3D edge curves into a face's (u, v) parameter space to produce
//! PCurves. For plane surfaces, constructs a local orthonormal frame and
//! projects via dot products. For analytic and NURBS surfaces, uses
//! `FaceSurface::project_point()` with periodic unwrapping.

use std::f64::consts::TAU;

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::HealError;

/// Default number of sample points for PCurve fitting.
const DEFAULT_SAMPLES: usize = 16;

/// Project a sequence of 3D points onto a surface to get UV coordinates.
///
/// Uses `FaceSurface::project_point()` for each point, with periodic
/// unwrapping to prevent jumps > half-period.
///
/// For `Plane` surfaces, constructs a local orthonormal frame from the
/// normal vector and projects via dot products.
///
/// # Errors
///
/// Returns [`HealError::FixFailed`] if the surface cannot project points
/// (e.g. empty input).
pub fn project_points_to_surface(
    points: &[Point3],
    surface: &FaceSurface,
    _tolerance: &Tolerance,
) -> Result<Vec<Point2>, HealError> {
    if points.is_empty() {
        return Err(HealError::FixFailed(
            "cannot project empty point sequence onto surface".to_string(),
        ));
    }

    let mut uv_pts: Vec<Point2> = if let FaceSurface::Plane { normal, d } = surface {
        // Fixed UV origin: closest point on plane to global origin.
        let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);
        let frame = brepkit_math::frame::Frame3::from_normal(origin, *normal)
            .map_err(|e| HealError::AnalysisFailed(format!("plane frame: {e}")))?;
        points
            .iter()
            .map(|&p| {
                let delta = p - frame.origin;
                Point2::new(delta.dot(frame.x), delta.dot(frame.y))
            })
            .collect()
    } else {
        // Analytic / NURBS: use surface.project_point().
        let mut uv = Vec::with_capacity(points.len());
        for &p in points {
            let (u, v) = surface.project_point(p).ok_or_else(|| {
                HealError::FixFailed(format!("surface failed to project 3D point {p:?}"))
            })?;
            uv.push(Point2::new(u, v));
        }
        uv
    };

    // Unwrap periodicity to prevent seam jumps.
    let (u_period, v_period) = surface_periods(surface);
    unwrap_periodic_params(&mut uv_pts, u_period, v_period);

    Ok(uv_pts)
}

/// Project a 3D edge curve onto a face surface to produce a 2D PCurve.
///
/// Samples the edge at `num_samples` points, projects each to UV via
/// [`project_points_to_surface`], then fits a NURBS curve through the
/// UV points using `brepkit_math::nurbs::fitting::interpolate()`.
///
/// The returned `NurbsCurve` lives in 3D space with z=0, representing
/// (u, v, 0) coordinates suitable for constructing a `NurbsCurve2D`.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail, projection fails, or
/// the NURBS fit fails.
pub fn project_edge_to_pcurve(
    topo: &Topology,
    edge_id: EdgeId,
    face_id: FaceId,
    num_samples: usize,
    tolerance: &Tolerance,
) -> Result<NurbsCurve, HealError> {
    let samples = if num_samples < 2 {
        DEFAULT_SAMPLES
    } else {
        num_samples
    };

    // Snapshot edge data.
    let edge = topo.edge(edge_id)?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let curve = edge.curve();

    // Sample 3D points along the edge.
    let (t0, t1) = curve.domain_with_endpoints(start_pos, end_pos);
    let mut pts_3d = Vec::with_capacity(samples + 1);
    for i in 0..=samples {
        #[allow(clippy::cast_precision_loss)]
        let frac = i as f64 / samples as f64;
        let t = t0 + (t1 - t0) * frac;
        let p = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        pts_3d.push(p);
    }

    // Get the face surface.
    let face = topo.face(face_id)?;
    let surface = face.surface();

    // Project to UV.
    let uv_pts = project_points_to_surface(&pts_3d, surface, tolerance)?;

    // Lift UV points to 3D (z=0) for NURBS interpolation.
    let pts_3d_uv: Vec<Point3> = uv_pts
        .iter()
        .map(|p| Point3::new(p.x(), p.y(), 0.0))
        .collect();

    // Fit: use degree 3 (or less if too few points).
    let degree = 3.min(pts_3d_uv.len() - 1);
    let nurbs = brepkit_math::nurbs::fitting::interpolate(&pts_3d_uv, degree)?;

    Ok(nurbs)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Returns `(u_period, v_period)` for a surface -- `Some(TAU)` if periodic.
fn surface_periods(surface: &FaceSurface) -> (Option<f64>, Option<f64>) {
    match surface {
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => (None, None),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => (Some(TAU), None),
        FaceSurface::Sphere(_) => (Some(TAU), None),
        FaceSurface::Torus(_) => (Some(TAU), Some(TAU)),
    }
}

/// Unwrap periodic UV parameters to remove seam jumps.
///
/// Detects jumps > half-period in consecutive points and adjusts subsequent
/// points by +/- period to maintain continuity.
fn unwrap_periodic_params(pts: &mut [Point2], u_period: Option<f64>, v_period: Option<f64>) {
    if pts.len() < 2 {
        return;
    }

    for i in 1..pts.len() {
        if let Some(period) = u_period {
            let half = period * 0.5;
            let du = pts[i].x() - pts[i - 1].x();
            if du > half {
                pts[i] = Point2::new(pts[i].x() - period, pts[i].y());
            } else if du < -half {
                pts[i] = Point2::new(pts[i].x() + period, pts[i].y());
            }
        }
        if let Some(period) = v_period {
            let half = period * 0.5;
            let dv = pts[i].y() - pts[i - 1].y();
            if dv > half {
                pts[i] = Point2::new(pts[i].x(), pts[i].y() - period);
            } else if dv < -half {
                pts[i] = Point2::new(pts[i].x(), pts[i].y() + period);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use brepkit_math::vec::Vec3;

    use super::*;

    #[test]
    fn project_points_on_xy_plane() {
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let points = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
        ];
        let tol = Tolerance::new();
        let uv = project_points_to_surface(&points, &surface, &tol).unwrap();

        // First point is origin -> (0, 0).
        assert!(uv[0].x().abs() < 1e-10);
        assert!(uv[0].y().abs() < 1e-10);
        // The 2D distance from origin to second point should be 1.0.
        let dist = (uv[1].x().powi(2) + uv[1].y().powi(2)).sqrt();
        assert!((dist - 1.0).abs() < 1e-10);
    }

    #[test]
    fn project_points_on_cylinder() {
        let cyl = brepkit_math::surfaces::CylindricalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
        )
        .unwrap();
        let surface = FaceSurface::Cylinder(cyl);

        // Points along a ruling line on the cylinder.
        let points = vec![
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 1.0),
            Point3::new(1.0, 0.0, 2.0),
        ];
        let tol = Tolerance::new();
        let uv = project_points_to_surface(&points, &surface, &tol).unwrap();

        // All u values should be the same (constant azimuthal angle).
        let u0 = uv[0].x();
        for pt in &uv[1..] {
            assert!(
                (pt.x() - u0).abs() < 1e-6,
                "u values should be constant: u0={u0}, got {}",
                pt.x()
            );
        }
        // v values should be 0, 1, 2 (axial distance).
        assert!((uv[0].y() - 0.0).abs() < 1e-6);
        assert!((uv[1].y() - 1.0).abs() < 1e-6);
        assert!((uv[2].y() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn project_empty_points_fails() {
        let surface = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let tol = Tolerance::new();
        assert!(project_points_to_surface(&[], &surface, &tol).is_err());
    }

    #[test]
    fn periodic_unwrap_removes_seam_jump() {
        let mut pts = vec![
            Point2::new(6.0, 0.0),
            Point2::new(6.2, 0.0),
            Point2::new(0.1, 0.0), // Jump from ~6.2 to ~0.1 (crossed 2pi)
            Point2::new(0.3, 0.0),
        ];
        unwrap_periodic_params(&mut pts, Some(TAU), None);
        // After unwrapping, point[2] should be ~6.383 (0.1 + TAU).
        assert!(
            (pts[2].x() - (0.1 + TAU)).abs() < 0.01,
            "got {}",
            pts[2].x()
        );
    }
}

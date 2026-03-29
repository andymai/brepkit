//! Coordinate/type conversions between 3D, UV, and topology types.

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;

use super::super::pcurve_compute::{
    compute_pcurve_on_surface, project_point_on_surface, sample_edge_to_uv,
};
use super::super::plane_frame::PlaneFrame;
use super::super::split_types::OrientedPCurveEdge;

/// Collect 3D vertex positions from a wire's edges.
pub fn collect_wire_points(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Vec<Point3> {
    let wire = match topo.wire(wire_id) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };
    let mut pts = Vec::new();
    for oe in wire.edges() {
        if let Ok(edge) = topo.edge(oe.edge()) {
            if let Ok(v) = topo.vertex(edge.start()) {
                pts.push(v.point());
            }
        }
    }
    pts
}

/// Extract the plane normal from a `FaceSurface`, defaulting to +Z.
pub(super) fn extract_plane_normal(surface: &FaceSurface) -> Vec3 {
    if let FaceSurface::Plane { normal, .. } = surface {
        *normal
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

/// Convert a wire's edges to `OrientedPCurveEdge`s on a surface.
pub(super) fn boundary_edges_to_pcurve(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &FaceSurface,
    wire_pts: &[Point3],
    frame: Option<&PlaneFrame>,
) -> Vec<OrientedPCurveEdge> {
    let wire = match topo.wire(wire_id) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let mut result = Vec::new();
    for oe in wire.edges() {
        let edge = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let start_v = match topo.vertex(if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        }) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let end_v = match topo.vertex(if oe.is_forward() {
            edge.end()
        } else {
            edge.start()
        }) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let start_3d = start_v.point();
        let end_3d = end_v.point();

        let pcurve =
            compute_pcurve_on_surface(edge.curve(), start_3d, end_3d, surface, wire_pts, frame);

        // For closed edges (start_3d approx end_3d, e.g. full circle), projecting
        // start and end to UV gives the same point. Use pcurve sampling to
        // get distinct UV endpoints spanning the full curve.
        let is_closed = (start_3d - end_3d).length() < 1e-10;
        let (start_uv, end_uv) = if is_closed && !matches!(surface, FaceSurface::Plane { .. }) {
            let uv_samples = sample_edge_to_uv(edge.curve(), start_3d, end_3d, surface);
            let su = uv_samples
                .first()
                .copied()
                .unwrap_or_else(|| project_point_on_surface(start_3d, surface, wire_pts, frame));
            let eu = uv_samples
                .last()
                .copied()
                .unwrap_or_else(|| project_point_on_surface(end_3d, surface, wire_pts, frame));
            (su, eu)
        } else {
            (
                project_point_on_surface(start_3d, surface, wire_pts, frame),
                project_point_on_surface(end_3d, surface, wire_pts, frame),
            )
        };

        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve().clone(),
            pcurve,
            start_uv,
            end_uv,
            start_3d,
            end_3d,
            forward: oe.is_forward(),
            source_edge_idx: None,
            pave_block_id: None,
        });
    }
    result
}

/// Check if a 3D point lies on any boundary edge in UV space.
///
/// Projects the point to UV (trying periodic shifts for seam-adjacent
/// points), then checks if the projected UV is within tolerance of any
/// boundary edge's UV segment.
pub(super) fn is_point_on_boundary_uv(
    point: Point3,
    surface: &FaceSurface,
    boundary: &[OrientedPCurveEdge],
    tol: f64,
) -> bool {
    let Some((pu, pv)) = surface.project_point(point) else {
        return false;
    };

    // For periodic surfaces, try the original u and u +/- 2pi.
    let u_period = match surface {
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
        _ => None,
    };
    let u_candidates: Vec<f64> = if let Some(period) = u_period {
        vec![pu, pu - period, pu + period]
    } else {
        vec![pu]
    };

    for &u in &u_candidates {
        let pt_uv = Point2::new(u, pv);
        for edge in boundary {
            let su = edge.start_uv;
            let eu = edge.end_uv;
            let dx = eu.x() - su.x();
            let dy = eu.y() - su.y();
            let seg_len_sq = dx * dx + dy * dy;

            if seg_len_sq < 1e-20 {
                // Closed edge (circle) -- check v-distance only.
                if (pv - su.y()).abs() < tol {
                    return true;
                }
            } else {
                let t = ((pt_uv.x() - su.x()) * dx + (pt_uv.y() - su.y()) * dy) / seg_len_sq;
                let t = t.clamp(0.0, 1.0);
                let cx = su.x() + t * dx;
                let cy = su.y() + t * dy;
                let dist = ((pt_uv.x() - cx).powi(2) + (pt_uv.y() - cy).powi(2)).sqrt();
                if dist < tol {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract UV endpoints from a pcurve's evaluation rather than independent
/// surface projection. This ensures consistency -- e.g. a pcurve that goes
/// from (pi, v) to (2pi, v) won't have its end snapped to (0, v) by the
/// surface's `project_point` which normalizes u into `[0, 2pi)`.
pub(super) fn uv_endpoints_from_pcurve(
    pcurve: &brepkit_math::curves2d::Curve2D,
    start_3d: Point3,
    end_3d: Point3,
    surface: &FaceSurface,
    wire_pts: &[Point3],
) -> (Point2, Point2) {
    use brepkit_math::curves2d::Curve2D;

    match pcurve {
        Curve2D::Line(line) => {
            // Line2D: start is at t=0. End is estimated by projecting the
            // 3D endpoint and computing the 2D distance along the line.
            let su = line.evaluate(0.0);
            let eu_proj = project_point_on_surface(end_3d, surface, wire_pts, None);
            let du = eu_proj.x() - su.x();
            let dv = eu_proj.y() - su.y();
            let len_2d = (du * du + dv * dv).sqrt();
            let eu = line.evaluate(len_2d);
            // Sanity: if the Line2D evaluation diverges from the projected
            // endpoint by more than pi (half a period), the line direction
            // is wrong -- fall back to direct projection.
            if (eu.x() - eu_proj.x()).abs() > std::f64::consts::PI
                || (eu.y() - eu_proj.y()).abs() > std::f64::consts::PI
            {
                (su, eu_proj)
            } else {
                (su, eu)
            }
        }
        Curve2D::Nurbs(nurbs) => {
            let knots = nurbs.knots();
            if knots.len() >= 2 {
                let t0 = knots[0];
                let tn = knots[knots.len() - 1];
                (nurbs.evaluate(t0), nurbs.evaluate(tn))
            } else {
                (
                    project_point_on_surface(start_3d, surface, wire_pts, None),
                    project_point_on_surface(end_3d, surface, wire_pts, None),
                )
            }
        }
        _ => (
            project_point_on_surface(start_3d, surface, wire_pts, None),
            project_point_on_surface(end_3d, surface, wire_pts, None),
        ),
    }
}

//! Phase EF: Edge-face intersection detection.
//!
//! For each (edge, face) pair across solids, finds points where the
//! edge crosses or touches the face surface. Records EF interferences
//! and adds extra paves to the edge for later splitting.

use std::collections::HashSet;

use crate::ds::{GfaArena, Interference, Pave};
use crate::error::AlgoError;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::Vertex;

use super::helpers::{add_pave_to_edge, find_nearby_pave_vertex as find_nearby_vertex};

/// Number of samples along each edge for sign-change detection.
const N_SAMPLES: usize = 64;

/// Detect edge-face intersections between the two solids.
///
/// Checks edges of A against faces of B, and edges of B against
/// faces of A. When an edge crosses a face surface (within tolerance),
/// an EF interference is recorded and an extra pave is added to the
/// edge's pave block.
///
/// # Errors
///
/// Returns [`AlgoError`] if any topology lookup fails.
pub fn perform(
    topo: &mut Topology,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    // AABB pre-filter: skip if solids are disjoint
    let bbox_a = crate::classifier::compute_solid_bbox(topo, solid_a)?;
    let bbox_b = crate::classifier::compute_solid_bbox(topo, solid_b)?;
    if !bbox_a
        .expanded(tol.linear)
        .intersects(bbox_b.expanded(tol.linear))
    {
        log::debug!("EF: solids are disjoint, skipping");
        return Ok(());
    }

    let edges_a = brepkit_topology::explorer::solid_edges(topo, solid_a)?;
    let edges_b = brepkit_topology::explorer::solid_edges(topo, solid_b)?;
    let faces_a = brepkit_topology::explorer::solid_faces(topo, solid_a)?;
    let faces_b = brepkit_topology::explorer::solid_faces(topo, solid_b)?;

    // Collect face boundary edge sets to skip edges that are already
    // on the face boundary.
    let face_boundary_edges_b = collect_face_boundary_edges(topo, &faces_b)?;
    let face_boundary_edges_a = collect_face_boundary_edges(topo, &faces_a)?;

    // Edges of A against faces of B
    check_edge_face_pairs(topo, &edges_a, &faces_b, &face_boundary_edges_b, tol, arena)?;

    // Edges of B against faces of A
    check_edge_face_pairs(topo, &edges_b, &faces_a, &face_boundary_edges_a, tol, arena)?;

    Ok(())
}

/// Collect the set of boundary edge IDs for each face.
fn collect_face_boundary_edges(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<Vec<HashSet<EdgeId>>, AlgoError> {
    let mut result = Vec::with_capacity(faces.len());
    for &fid in faces {
        let edges = brepkit_topology::explorer::face_edges(topo, fid)?;
        result.push(edges.into_iter().collect());
    }
    Ok(result)
}

/// Check each edge against each face.
#[allow(clippy::too_many_lines)]
fn check_edge_face_pairs(
    topo: &mut Topology,
    edges: &[EdgeId],
    faces: &[FaceId],
    face_boundary_edges: &[HashSet<EdgeId>],
    tol: Tolerance,
    arena: &mut GfaArena,
) -> Result<(), AlgoError> {
    for &eid in edges {
        // Snapshot edge data to avoid holding immutable borrow across add_vertex
        let (curve, start_pos, end_pos, t0, t1) = {
            let edge = topo.edge(eid)?;
            let sp = topo.vertex(edge.start())?.point();
            let ep = topo.vertex(edge.end())?.point();
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            (edge.curve().clone(), sp, ep, t0, t1)
        };

        for (face_idx, &fid) in faces.iter().enumerate() {
            // Skip if edge is already a boundary edge of this face
            if face_boundary_edges[face_idx].contains(&eid) {
                continue;
            }

            let face = topo.face(fid)?;
            let surface = face.surface();

            let (is_coplanar, crossings) = match surface {
                FaceSurface::Plane { normal, d } => {
                    let c = find_edge_plane_crossings(
                        &curve, start_pos, end_pos, t0, t1, *normal, *d, tol,
                    );
                    if c.is_empty() {
                        // Check if edge lies ON the plane (coplanar)
                        let dist_start = start_pos.x() * normal.x()
                            + start_pos.y() * normal.y()
                            + start_pos.z() * normal.z()
                            - d;
                        let dist_end = end_pos.x() * normal.x()
                            + end_pos.y() * normal.y()
                            + end_pos.z() * normal.z()
                            - d;
                        if dist_start.abs() < tol.linear * 10.0
                            && dist_end.abs() < tol.linear * 10.0
                        {
                            // Edge lies ON the face. Check if it's inside the face
                            // boundary (not just touching it from outside).
                            // Test: is the midpoint inside the face polygon?
                            let mid = Point3::new(
                                (start_pos.x() + end_pos.x()) * 0.5,
                                (start_pos.y() + end_pos.y()) * 0.5,
                                (start_pos.z() + end_pos.z()) * 0.5,
                            );
                            if point_inside_face_boundary(topo, fid, mid, *normal) {
                                // The entire edge is a coplanar section edge.
                                // Add paves at both endpoints (they're on the boundary).
                                let mut results = Vec::new();
                                // Only add pave at interior parameter values
                                // (endpoints at t0/t1 are already boundary vertices).
                                // Add a midpoint pave to register the edge in the arena.
                                let t_mid = (t0 + t1) * 0.5;
                                results.push((t_mid, mid));
                                (true, results)
                            } else {
                                // Midpoint outside face → edge is external
                                let bc = find_coplanar_edge_boundary_crossings(
                                    topo, fid, &curve, start_pos, end_pos, t0, t1, *normal, tol,
                                );
                                let coplanar = !bc.is_empty();
                                (coplanar, bc)
                            }
                        } else {
                            (false, Vec::new())
                        }
                    } else {
                        (false, c)
                    }
                }
                _ => (
                    false,
                    find_edge_surface_crossings(&curve, start_pos, end_pos, t0, t1, surface, tol),
                ),
            };

            for (t, pt) in crossings {
                // Check if an existing vertex is at this point
                let existing = find_nearby_vertex(topo, arena, pt, tol);

                let vertex_id = if let Some(vid) = existing {
                    vid
                } else {
                    // No existing vertex near this point — create one.
                    topo.add_vertex(Vertex::new(pt, tol.linear))
                };

                // Add extra pave to the edge
                let pave = Pave::new(vertex_id, t);
                add_pave_to_edge(arena, eid, pave);

                arena.interference.ef.push(Interference::EF {
                    edge: eid,
                    face: fid,
                    new_vertex: Some(vertex_id),
                    parameter: Some(t),
                    coplanar: Some(is_coplanar),
                });

                // Add vertex to face info
                arena.face_info_mut(fid).vertices_in.insert(vertex_id);

                log::debug!("EF: edge {eid:?} crosses face {fid:?} at t={t:.6}",);
            }
        }
    }

    Ok(())
}

/// Find edge-plane crossings using algebraic ray-plane intersection.
#[allow(clippy::too_many_arguments)]
fn find_edge_plane_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    if matches!(curve, EdgeCurve::Line) {
        // Algebraic: line-plane intersection
        let dir = end_pos - start_pos;
        let denom = dir.dot(normal);

        // 1e-15 checks for mathematical degeneracy (line parallel to
        // plane), not geometric tolerance.
        if denom.abs() < 1e-15 {
            // Line parallel to plane — no single crossing
            return Vec::new();
        }

        let origin_dot =
            start_pos.x() * normal.x() + start_pos.y() * normal.y() + start_pos.z() * normal.z();
        let s = (d - origin_dot) / denom;

        // s is in [0, 1] parameterization of start..end
        if !(-1e-7..=1.0 + 1e-7).contains(&s) {
            return Vec::new();
        }

        let s_clamped = s.clamp(0.0, 1.0);
        let pt = start_pos + dir * s_clamped;
        let t = s_clamped.mul_add(t1 - t0, t0);
        vec![(t, pt)]
    } else {
        // General case: sample and find sign changes
        find_crossings_by_sampling(
            curve,
            start_pos,
            end_pos,
            t0,
            t1,
            &|pt: Point3| pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d,
            tol.linear,
        )
    }
}

/// Find edge-surface crossings by sampling signed distance and refining.
fn find_edge_surface_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    surface: &FaceSurface,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    // Sample and find places where distance to surface is minimal
    let n = N_SAMPLES;
    let mut crossings = Vec::new();
    let mut prev_dist = f64::MAX;
    let mut prev_t = t0;

    for i in 0..=n {
        let t = t0 + (t1 - t0) * (i as f64 / n as f64);
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let dist = distance_to_surface(pt, surface);

        // Check if we've found a close approach or sign change in
        // the signed distance proxy
        if i > 0 && dist < tol.linear {
            // Already within tolerance — record
            let is_dup = crossings
                .iter()
                .any(|&(ct, _): &(f64, Point3)| (t - ct).abs() < (t1 - t0) / (n as f64) * 2.0);
            if !is_dup {
                // Refine with bisection
                let refined = refine_crossing(curve, start_pos, end_pos, prev_t, t, surface, tol);
                crossings.push(refined);
            }
        } else if i > 0 && prev_dist > tol.linear && dist > tol.linear {
            // Check for a minimum between these two samples
            let mid_t = f64::midpoint(prev_t, t);
            let mid_pt = curve.evaluate_with_endpoints(mid_t, start_pos, end_pos);
            let mid_dist = distance_to_surface(mid_pt, surface);
            if mid_dist < prev_dist.min(dist) && mid_dist < tol.linear * 2.0 {
                let refined = refine_crossing(curve, start_pos, end_pos, prev_t, t, surface, tol);
                if distance_to_surface(refined.1, surface) < tol.linear {
                    crossings.push(refined);
                }
            }

            // Tangent contact: near-surface sample triggers golden section minimum search
            if prev_dist < 4.0 * tol.linear || dist < 4.0 * tol.linear {
                let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
                let mut lo = prev_t;
                let mut hi = t;
                for _ in 0..30 {
                    let m1 = hi - phi * (hi - lo);
                    let m2 = lo + phi * (hi - lo);
                    let d1 = distance_to_surface(
                        curve.evaluate_with_endpoints(m1, start_pos, end_pos),
                        surface,
                    );
                    let d2 = distance_to_surface(
                        curve.evaluate_with_endpoints(m2, start_pos, end_pos),
                        surface,
                    );
                    if d1 < d2 {
                        hi = m2;
                    } else {
                        lo = m1;
                    }
                }
                let t_min = f64::midpoint(lo, hi);
                let pt_min = curve.evaluate_with_endpoints(t_min, start_pos, end_pos);
                if distance_to_surface(pt_min, surface) < tol.linear {
                    let is_dup = crossings.iter().any(|&(ct, _): &(f64, Point3)| {
                        (t_min - ct).abs() < (t1 - t0) / (n as f64) * 2.0
                    });
                    if !is_dup {
                        let refined =
                            refine_crossing(curve, start_pos, end_pos, lo, hi, surface, tol);
                        crossings.push(refined);
                    }
                }
            }
        }

        prev_dist = dist;
        prev_t = t;
    }

    crossings
}

/// Find crossings by sampling a signed distance function and detecting sign changes.
fn find_crossings_by_sampling(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    signed_dist: &dyn Fn(Point3) -> f64,
    tol_linear: f64,
) -> Vec<(f64, Point3)> {
    let n = N_SAMPLES;
    let mut crossings = Vec::new();

    let mut samples: Vec<(f64, f64)> = Vec::with_capacity(n + 1);
    for i in 0..=n {
        let t = t0 + (t1 - t0) * (i as f64 / n as f64);
        let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
        let sd = signed_dist(pt);
        samples.push((t, sd));
    }

    for i in 0..n {
        let (t_a, sd_a) = samples[i];
        let (t_b, sd_b) = samples[i + 1];

        // Sign change indicates a crossing
        if sd_a * sd_b < 0.0 {
            // Bisect to find exact crossing
            let mut lo = t_a;
            let mut hi = t_b;
            let mut sd_lo = sd_a;

            for _ in 0..30 {
                let mid = f64::midpoint(lo, hi);
                let pt_mid = curve.evaluate_with_endpoints(mid, start_pos, end_pos);
                let sd_mid = signed_dist(pt_mid);

                if sd_mid * sd_lo < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                    sd_lo = sd_mid;
                }
            }

            let t = f64::midpoint(lo, hi);
            let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
            crossings.push((t, pt));
        }
        // Tangent contact: minimum approaches zero without sign change
        else if sd_a.abs() < 4.0 * tol_linear || sd_b.abs() < 4.0 * tol_linear {
            let phi = 0.5 * (5.0_f64.sqrt() - 1.0);
            let mut lo = t_a;
            let mut hi = t_b;
            for _ in 0..30 {
                let m1 = hi - phi * (hi - lo);
                let m2 = lo + phi * (hi - lo);
                let d1 = signed_dist(curve.evaluate_with_endpoints(m1, start_pos, end_pos)).abs();
                let d2 = signed_dist(curve.evaluate_with_endpoints(m2, start_pos, end_pos)).abs();
                if d1 < d2 {
                    hi = m2;
                } else {
                    lo = m1;
                }
            }
            let t_min = f64::midpoint(lo, hi);
            let pt_min = curve.evaluate_with_endpoints(t_min, start_pos, end_pos);
            let d_min = signed_dist(pt_min).abs();
            if d_min < tol_linear {
                let is_dup = crossings.iter().any(|&(ct, _): &(f64, Point3)| {
                    (t_min - ct).abs() < (t1 - t0) / (n as f64) * 2.0
                });
                if !is_dup {
                    crossings.push((t_min, pt_min));
                }
            }
        }
    }

    crossings
}

/// Compute distance from point to surface.
fn distance_to_surface(pt: Point3, surface: &FaceSurface) -> f64 {
    if let FaceSurface::Plane { normal, d } = surface {
        (pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d).abs()
    } else if let Some((u, v)) = surface.project_point(pt) {
        if let Some(surf_pt) = surface.evaluate(u, v) {
            (pt - surf_pt).length()
        } else {
            f64::MAX
        }
    } else {
        f64::MAX
    }
}

/// Refine a crossing between two parameter values using ternary search.
fn refine_crossing(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t_lo: f64,
    t_hi: f64,
    surface: &FaceSurface,
    _tol: Tolerance,
) -> (f64, Point3) {
    let mut lo = t_lo;
    let mut hi = t_hi;

    for _ in 0..30 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 = distance_to_surface(
            curve.evaluate_with_endpoints(m1, start_pos, end_pos),
            surface,
        );
        let d2 = distance_to_surface(
            curve.evaluate_with_endpoints(m2, start_pos, end_pos),
            surface,
        );
        if d1 < d2 {
            hi = m2;
        } else {
            lo = m1;
        }
    }

    let t = f64::midpoint(lo, hi);
    let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);
    (t, pt)
}

/// Check if a 3D point lies inside a face's boundary polygon (2D test).
fn point_inside_face_boundary(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
    normal: Vec3,
) -> bool {
    use brepkit_math::vec::Point2;

    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return false,
    };

    // Build 2D frame
    let ref_axis = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = ref_axis.cross(normal);
    let u_len = u_axis.length();
    if u_len < 1e-12 {
        return false;
    }
    let u_axis = u_axis * (1.0 / u_len);
    let v_axis = normal.cross(u_axis);

    let project = |p: Point3| -> Point2 {
        Point2::new(
            p.x() * u_axis.x() + p.y() * u_axis.y() + p.z() * u_axis.z(),
            p.x() * v_axis.x() + p.y() * v_axis.y() + p.z() * v_axis.z(),
        )
    };

    let pt_2d = project(point);
    let mut poly = Vec::new();
    for oe in wire.edges() {
        let e = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = match topo.vertex(oe.oriented_start(e)) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        poly.push(project(p));
    }
    if poly.len() < 3 {
        return false;
    }

    // Ray-casting point-in-polygon
    let mut inside = false;
    let px = pt_2d.x();
    let py = pt_2d.y();
    let n = poly.len();
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly[i].y();
        let yj = poly[j].y();
        let xi = poly[i].x();
        let xj = poly[j].x();
        if ((yi > py) != (yj > py)) && (px < (xj - xi) * (py - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Find where a coplanar edge crosses a face's boundary polygon.
///
/// When an edge lies on a plane face's surface (distance ≈ 0), this
/// function projects both to 2D and finds where the edge line intersects
/// the face's boundary polygon edges. Returns the 3D crossing points
/// with their parameters on the edge.
#[allow(clippy::too_many_arguments)]
fn find_coplanar_edge_boundary_crossings(
    topo: &Topology,
    face_id: FaceId,
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    normal: Vec3,
    tol: Tolerance,
) -> Vec<(f64, Point3)> {
    use brepkit_math::vec::Point2;

    // Build 2D frame from plane normal
    let ref_axis = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_axis = ref_axis.cross(normal);
    let u_len = u_axis.length();
    if u_len < 1e-12 {
        return Vec::new();
    }
    let u_axis = u_axis * (1.0 / u_len);
    let v_axis = normal.cross(u_axis);

    let project = |p: Point3| -> Point2 {
        Point2::new(
            p.x() * u_axis.x() + p.y() * u_axis.y() + p.z() * u_axis.z(),
            p.x() * v_axis.x() + p.y() * v_axis.y() + p.z() * v_axis.z(),
        )
    };

    // Get face boundary polygon in 2D
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let wire = match topo.wire(face.outer_wire()) {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };
    let mut boundary_2d = Vec::new();
    for oe in wire.edges() {
        let e = match topo.edge(oe.edge()) {
            Ok(e) => e,
            Err(_) => continue,
        };
        let p = match topo.vertex(oe.oriented_start(e)) {
            Ok(v) => v.point(),
            Err(_) => continue,
        };
        boundary_2d.push(project(p));
    }
    if boundary_2d.len() < 3 {
        return Vec::new();
    }

    // Project edge endpoints to 2D
    let e0_2d = project(start_pos);
    let e1_2d = project(end_pos);
    let edge_dx = e1_2d.x() - e0_2d.x();
    let edge_dy = e1_2d.y() - e0_2d.y();

    let mut crossings = Vec::new();
    let n = boundary_2d.len();

    for i in 0..n {
        let b0 = &boundary_2d[i];
        let b1 = &boundary_2d[(i + 1) % n];

        let bnd_dx = b1.x() - b0.x();
        let bnd_dy = b1.y() - b0.y();

        let det = edge_dx * bnd_dy - edge_dy * bnd_dx;
        if det.abs() < 1e-15 {
            continue; // Parallel
        }

        let dx = b0.x() - e0_2d.x();
        let dy = b0.y() - e0_2d.y();
        let s = (dx * bnd_dy - dy * bnd_dx) / det; // Parameter on edge [0,1]
        let u = (dx * edge_dy - dy * edge_dx) / det; // Parameter on boundary edge [0,1]

        // Strictly interior crossings (avoid endpoint coincidence)
        let eps = tol.linear * 10.0;
        if s > eps && s < 1.0 - eps && u > -eps && u < 1.0 + eps {
            // Convert s from [0,1] parameterization to [t0,t1]
            let t = s.mul_add(t1 - t0, t0);
            let pt = curve.evaluate_with_endpoints(t, start_pos, end_pos);

            // Dedup
            let is_dup = crossings
                .iter()
                .any(|&(ct, _): &(f64, Point3)| (t - ct).abs() < tol.linear);
            if !is_dup {
                crossings.push((t, pt));
            }
        }
    }

    crossings
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::EdgeCurve;

    #[test]
    fn sampling_detects_tangent_touch() {
        // Signed distance: parabola touching zero at t=0.5 (exact tangent)
        let curve = EdgeCurve::Line;
        let start = Point3::new(0.0, 0.0, 0.0);
        let end = Point3::new(1.0, 0.0, 0.0);
        let signed_dist = |pt: Point3| -> f64 {
            let t = pt.x();
            (t - 0.5) * (t - 0.5) // minimum = 0 at t=0.5
        };

        let crossings =
            find_crossings_by_sampling(&curve, start, end, 0.0, 1.0, &signed_dist, 1e-7);

        assert!(
            !crossings.is_empty(),
            "tangent touch (minimum=0) should be detected"
        );
        let (t, _) = crossings[0];
        assert!(
            (t - 0.5).abs() < 0.02,
            "tangent point should be near t=0.5, got {t}"
        );
    }
}

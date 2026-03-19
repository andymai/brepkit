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

            let crossings = match surface {
                FaceSurface::Plane { normal, d } => {
                    find_edge_plane_crossings(&curve, start_pos, end_pos, t0, t1, *normal, *d)
                }
                _ => find_edge_surface_crossings(&curve, start_pos, end_pos, t0, t1, surface, tol),
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
fn find_edge_plane_crossings(
    curve: &EdgeCurve,
    start_pos: Point3,
    end_pos: Point3,
    t0: f64,
    t1: f64,
    normal: Vec3,
    d: f64,
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
        find_crossings_by_sampling(curve, start_pos, end_pos, t0, t1, &|pt: Point3| {
            pt.x() * normal.x() + pt.y() * normal.y() + pt.z() * normal.z() - d
        })
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
        // TODO: tangent contact detection — golden section search when
        // sample distance < 4*tol to catch near-tangent edge-face touches
        // that don't produce a sign change.
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

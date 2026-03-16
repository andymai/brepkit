//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SubFace`]s.

#![allow(dead_code)] // Used by later boolean_v2 pipeline stages.

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;

use super::classify_2d::{sample_interior_point, signed_area_2d};
use super::pcurve_compute::{compute_pcurve_on_surface, project_point_on_surface};
use super::pipeline::{OrientedPCurveEdge, SectionEdge, SubFace};
use super::types::Source;
use super::wire_builder::build_wire_loops;

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// # Arguments
/// - `topo` — the topology arena (immutable read)
/// - `face_id` — the face to split
/// - `sections` — intersection curves that cut this face (already trimmed)
/// - `source` — which solid this face belongs to (A or B)
/// - `tol` — UV-space tolerance for vertex deduplication
#[allow(clippy::too_many_lines)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    source: Source,
    tol: f64,
) -> Vec<SubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let surface = face.surface().clone();
    let reversed = face.is_reversed();

    // Collect the outer wire's vertex positions for the PlaneFrame origin.
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let _normal = extract_plane_normal(&surface);

    // Convert boundary edges to OrientedPCurveEdge.
    let boundary_edges = boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts);

    // If no section edges, the face is unsplit — return as-is.
    if sections.is_empty() {
        return vec![SubFace {
            surface,
            outer_wire: boundary_edges,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            source,
        }];
    }

    // Stage 2: Split boundary edges at section edge endpoints.
    // Section edges start/end on boundary edges. We need to add vertices
    // at these points so the wire builder can connect them.
    let split_pts: Vec<Point2> = sections
        .iter()
        .flat_map(|s| {
            let start_uv = project_point_on_surface(s.start, &surface, &wire_pts);
            let end_uv = project_point_on_surface(s.end, &surface, &wire_pts);
            vec![start_uv, end_uv]
        })
        .collect();
    let boundary_edges = split_boundary_edges_at_points(boundary_edges, &split_pts, tol);

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    for section in sections {
        let pcurve_on_this_face = match source {
            Source::A => &section.pcurve_a,
            Source::B => &section.pcurve_b,
        };
        let start_uv = project_point_on_surface(section.start, &surface, &wire_pts);
        let end_uv = project_point_on_surface(section.end, &surface, &wire_pts);

        // Forward direction.
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
        });
        // Reverse direction (for the adjacent sub-face).
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv: end_uv,
            end_uv: start_uv,
            start_3d: section.end,
            end_3d: section.start,
            forward: false,
        });
    }

    // Build wire loops via angular-sorting traversal.
    let loops = build_wire_loops(&all_edges, tol, false, false);

    // Classify each loop as outer (positive area) or hole (negative).
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    for wire_loop in loops {
        let pts: Vec<Point2> = wire_loop.iter().map(|e| e.start_uv).collect();
        let area = signed_area_2d(&pts);
        if area > 0.0 {
            outers.push((wire_loop, area));
        } else {
            holes.push(wire_loop);
        }
    }

    // If all loops are CW (negative area), the winding is reversed.
    // In that case, treat the largest-magnitude-area loop as outer.
    if outers.is_empty() && !holes.is_empty() {
        // Reverse all loops — they're actually CCW in a reversed face.
        for hole in &mut holes {
            hole.reverse();
            for edge in hole.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }
        let pts: Vec<Point2> = holes[0].iter().map(|e| e.start_uv).collect();
        let area = signed_area_2d(&pts);
        outers.push((holes.remove(0), area));
    }

    // Match holes to containing outer wires.
    // For now (plane-only, simple cases), just assign all holes to the
    // largest outer wire.
    let mut sub_faces = Vec::new();
    for (outer_wire, _area) in outers {
        sub_faces.push(SubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(), // Holes matched below.
            reversed,
            parent: face_id,
            source,
        });
    }

    // Simple hole matching: each hole goes to the outer that contains its
    // first vertex (via 2D point-in-polygon).
    for hole in holes {
        if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
            let mut assigned = false;
            for sf in &mut sub_faces {
                let outer_pts: Vec<Point2> = sf.outer_wire.iter().map(|e| e.start_uv).collect();
                if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                    sf.inner_wires.push(hole.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                // Fallback: assign to the first outer.
                if let Some(sf) = sub_faces.first_mut() {
                    sf.inner_wires.push(hole);
                }
            }
        }
    }

    sub_faces
}

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// then evaluate it to 3D via the surface.
pub fn interior_point_3d(sub_face: &SubFace, wire_pts: &[Point3]) -> Point3 {
    let pts_2d: Vec<Point2> = sub_face.outer_wire.iter().map(|e| e.start_uv).collect();
    let interior_uv = sample_interior_point(&pts_2d);

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return p;
    }

    // For plane faces, evaluate via PlaneFrame.
    if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = &sub_face.surface {
        let frame = super::plane_frame::PlaneFrame::from_plane_face(*normal, wire_pts);
        return frame.evaluate(interior_uv.x(), interior_uv.y());
    }

    // Last resort: average of 3D endpoints.
    let sum: Point3 = sub_face
        .outer_wire
        .iter()
        .fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
            acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
        });
    let n = sub_face.outer_wire.len() as f64;
    Point3::new(sum.x() / n, sum.y() / n, sum.z() / n)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split boundary edges at points where section edges start/end.
///
/// For each split point that lies on a boundary edge (within tolerance),
/// replace that edge with two sub-edges: start→split and split→end.
fn split_boundary_edges_at_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts: &[Point2],
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    let mut result = Vec::new();
    for edge in edges {
        // Find all split points that lie on this edge.
        let mut splits_on_edge: Vec<(f64, Point2)> = Vec::new();
        for &sp in split_pts {
            if let Some(t) = point_on_edge_parameter(&edge, sp, tol) {
                if t > tol && t < 1.0 - tol {
                    splits_on_edge.push((t, sp));
                }
            }
        }

        if splits_on_edge.is_empty() {
            result.push(edge);
            continue;
        }

        // Sort by parameter.
        splits_on_edge.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        // Dedup close splits.
        splits_on_edge.dedup_by(|a, b| (a.0 - b.0).abs() < tol);

        // Split edge into segments.
        let mut prev_uv = edge.start_uv;
        let mut prev_3d = edge.start_3d;
        for (t, split_uv) in &splits_on_edge {
            // Interpolate 3D position.
            let split_3d = Point3::new(
                edge.start_3d.x() + (edge.end_3d.x() - edge.start_3d.x()) * t,
                edge.start_3d.y() + (edge.end_3d.y() - edge.start_3d.y()) * t,
                edge.start_3d.z() + (edge.end_3d.z() - edge.start_3d.z()) * t,
            );
            let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
                &edge.curve_3d,
                prev_3d,
                split_3d,
                &brepkit_topology::face::FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 0.0,
                },
                &[prev_3d],
            );
            result.push(OrientedPCurveEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve,
                start_uv: prev_uv,
                end_uv: *split_uv,
                start_3d: prev_3d,
                end_3d: split_3d,
                forward: edge.forward,
            });
            prev_uv = *split_uv;
            prev_3d = split_3d;
        }
        // Final segment.
        let pcurve = super::pcurve_compute::compute_pcurve_on_surface(
            &edge.curve_3d,
            prev_3d,
            edge.end_3d,
            &brepkit_topology::face::FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
            &[prev_3d],
        );
        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve_3d.clone(),
            pcurve,
            start_uv: prev_uv,
            end_uv: edge.end_uv,
            start_3d: prev_3d,
            end_3d: edge.end_3d,
            forward: edge.forward,
        });
    }
    result
}

/// Check if a 2D point lies on an edge and return the parameter t ∈ [0,1].
fn point_on_edge_parameter(edge: &OrientedPCurveEdge, point: Point2, tol: f64) -> Option<f64> {
    let dx = edge.end_uv.x() - edge.start_uv.x();
    let dy = edge.end_uv.y() - edge.start_uv.y();
    let len_sq = dx * dx + dy * dy;
    if len_sq < tol * tol {
        return None; // Degenerate edge.
    }
    let px = point.x() - edge.start_uv.x();
    let py = point.y() - edge.start_uv.y();

    // Parameter along the edge.
    let t = (px * dx + py * dy) / len_sq;
    if t < -tol || t > 1.0 + tol {
        return None;
    }

    // Distance from point to the edge line.
    let dist = (px * dy - py * dx).abs() / len_sq.sqrt();
    if dist > tol * 100.0 {
        // Allow wider tolerance because UV coordinates come from different
        // projections (boundary vs. section) and may have rounding.
        return None;
    }

    Some(t.clamp(0.0, 1.0))
}

fn collect_wire_points(topo: &Topology, wire_id: brepkit_topology::wire::WireId) -> Vec<Point3> {
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

fn extract_plane_normal(surface: &brepkit_topology::face::FaceSurface) -> Vec3 {
    if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = surface {
        *normal
    } else {
        Vec3::new(0.0, 0.0, 1.0) // Fallback for non-plane (shouldn't happen in Step 1).
    }
}

fn boundary_edges_to_pcurve(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &brepkit_topology::face::FaceSurface,
    wire_pts: &[Point3],
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

        let pcurve = compute_pcurve_on_surface(edge.curve(), start_3d, end_3d, surface, wire_pts);
        let start_uv = project_point_on_surface(start_3d, surface, wire_pts);
        let end_uv = project_point_on_surface(end_3d, surface, wire_pts);

        result.push(OrientedPCurveEdge {
            curve_3d: edge.curve().clone(),
            pcurve,
            start_uv,
            end_uv,
            start_3d,
            end_3d,
            forward: oe.is_forward(),
        });
    }
    result
}

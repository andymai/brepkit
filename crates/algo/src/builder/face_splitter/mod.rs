//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SplitSubFace`]s.

mod containment;
mod conversion;
mod edge_splitting;
mod sampling;
mod special_cases;

pub use conversion::collect_wire_points;

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::classify_2d::{sample_interior_point, signed_area_2d};
use super::plane_frame::PlaneFrame;
use super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace, SurfaceInfo};
use super::wire_builder::build_wire_loops;
use crate::ds::Rank;

use containment::{find_point_outside_holes, is_inside_any_hole};
use conversion::{
    boundary_edges_to_pcurve, extract_plane_normal, is_point_on_boundary_uv,
    uv_endpoints_from_pcurve,
};
use edge_splitting::split_boundary_edges_at_3d_points;
use sampling::{sample_wire_loop_uv, sample_wire_loop_uv_periodic};
use special_cases::{
    split_face_with_internal_loops, split_noseam_face_direct, try_split_crossing_plane_face,
};

/// Split a face by its section edges, producing sub-faces.
///
/// If there are no section edges, returns a single sub-face covering
/// the entire face (pass-through).
///
/// # Arguments
/// - `topo` -- the topology arena (immutable read)
/// - `face_id` -- the face to split
/// - `sections` -- intersection curves that cut this face (already trimmed)
/// - `rank` -- which solid this face belongs to (A or B)
/// - `tol` -- tolerance (`.linear` for 3D matching, UV tol derived internally)
/// - `frame` -- cached `PlaneFrame` for this face (avoids origin mismatch)
/// - `info` -- cached `SurfaceInfo` for periodicity flags
#[allow(clippy::too_many_lines)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    rank: Rank,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
) -> Vec<SplitSubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let is_plane = matches!(surface, FaceSurface::Plane { .. });

    // Use provided frame or build one from wire points (plane faces only).
    let wire_pts = collect_wire_points(topo, face.outer_wire());
    let owned_frame;
    let frame = if let Some(f) = frame {
        f
    } else if is_plane {
        let normal = extract_plane_normal(&surface);
        owned_frame = PlaneFrame::from_plane_face(normal, &wire_pts);
        &owned_frame
    } else {
        // For non-plane faces, PlaneFrame is not used -- set a dummy.
        // All UV projection goes through surface.project_point().
        owned_frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &[]);
        &owned_frame
    };

    // Extract periodicity from SurfaceInfo.
    // Periodic quantization is needed for boundary wire connectivity (circle
    // end at u=2pi connects to seam start at u=0). Keep it enabled.
    let (u_periodic, v_periodic) = info.map_or((false, false), SurfaceInfo::periodicity);

    // Convert boundary edges to OrientedPCurveEdge.
    let mut boundary_edges = if is_plane {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, Some(frame))
    } else {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, None)
    };

    // Convert original inner wires (holes) to OrientedPCurveEdge.
    let original_inner_wires: Vec<Vec<OrientedPCurveEdge>> = face
        .inner_wires()
        .iter()
        .filter_map(|&iw_id| {
            let iw_pts = collect_wire_points(topo, iw_id);
            if iw_pts.len() < 3 {
                return None;
            }
            let edges = if is_plane {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, Some(frame))
            } else {
                boundary_edges_to_pcurve(topo, iw_id, &surface, &iw_pts, None)
            };
            if edges.is_empty() { None } else { Some(edges) }
        })
        .collect();

    // If no section edges, the face is unsplit -- return as-is with original holes.
    if sections.is_empty() {
        return vec![SplitSubFace {
            surface,
            outer_wire: boundary_edges,
            inner_wires: original_inner_wires,
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        }];
    }

    // No-seam face shortcut: faces whose boundary is entirely Line edges
    // (no seam edges) can't be split by the wire builder (it needs vertical
    // seam connections to form rectangular bands). Construct cap + band
    // sub-faces directly instead. Applies to sphere hemispheres and any
    // other face topology without seam edges.
    let all_boundary_line = boundary_edges.iter().all(|e| {
        matches!(e.curve_3d, EdgeCurve::Line)
            // Exclude degenerate seam edges (start approx end) -- those are periodic
            // seam connections (e.g., torus), not true line boundaries.
            && (e.start_3d - e.end_3d).length() > tol.linear
    });
    if all_boundary_line && !is_plane {
        return split_noseam_face_direct(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            &wire_pts,
        );
    }

    // Internal section edge shortcut: when section edges form closed loops
    // entirely within the face (not connecting to boundary edges), the wire
    // builder struggles with periodic UV and 4-way junctions. Instead, group
    // the section edges into closed loops and construct sub-faces directly.
    //
    // Detection: check if ALL section endpoints are far from the face
    // boundary in UV space. Project each section endpoint to UV and test
    // if it lies on any boundary edge's UV segment (within tolerance).
    // This is surface-type agnostic and handles curved boundary edges.
    let all_sections_internal = if sections.is_empty() {
        false
    } else if is_plane {
        // Only for plane faces with exactly 1 closed section curve.
        // Multiple circles on the same plane face need the wire builder
        // for correct loop formation.
        sections.len() == 1
            && sections.iter().all(|s| {
                (s.start - s.end).length() < tol.linear // closed curve
            })
    } else {
        // Non-plane faces: check if all section endpoints are off the
        // boundary in UV space.
        let uv_tol = 0.01; // ~0.6 deg in angular coordinates
        sections.iter().all(|s| {
            let start_on_boundary =
                is_point_on_boundary_uv(s.start, &surface, &boundary_edges, uv_tol);
            let end_on_boundary = is_point_on_boundary_uv(s.end, &surface, &boundary_edges, uv_tol);
            !start_on_boundary && !end_on_boundary
        })
    };

    if all_sections_internal {
        return split_face_with_internal_loops(
            &surface,
            &boundary_edges,
            sections,
            rank,
            reversed,
            face_id,
            &wire_pts,
        );
    }

    // Stage 2: Split boundary edges at section edge endpoints (3D matching).
    let mut split_pts_3d: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();

    // For periodic faces, align closed boundary edge UV with seam edge UV.
    // The same 3D vertex projects to u=0 (from circle unwrapping) and u=seam
    // (from Line edge projection). Shift the circle UV so it starts at seam_u.
    if u_periodic {
        let seam_u_opt = boundary_edges.iter().find_map(|e| {
            if matches!(e.curve_3d, EdgeCurve::Line) {
                surface.project_point(e.start_3d).map(|(u, _)| u)
            } else {
                None
            }
        });
        if let Some(seam_u) = seam_u_opt {
            for edge in &mut boundary_edges {
                if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                    // Closed edge: shift UV so start_uv.x() == seam_u.
                    let shift = seam_u - edge.start_uv.x();
                    if shift.abs() > 0.01 {
                        edge.start_uv = Point2::new(edge.start_uv.x() + shift, edge.start_uv.y());
                        edge.end_uv = Point2::new(edge.end_uv.x() + shift, edge.end_uv.y());
                    }
                }
            }
        }
    }

    // For periodic faces with section edges, split closed boundary edges
    // (full circles) at the point diametrically opposite the seam vertex
    // in the surface's UV parameterization (u = seam_u + pi).
    //
    // The seam vertex (where the boundary circle starts/ends) is shared
    // with the seam Line edge. Splitting the circle at the UV-antipodal
    // point creates half-arcs whose endpoints match the seam edge vertices,
    // enabling the wire builder to form proper rectangular bands.
    if u_periodic && !sections.is_empty() {
        // Find the seam Line edge's vertex UV to determine seam_u.
        let seam_u = {
            let mut su = 0.0_f64;
            for edge in &boundary_edges {
                if matches!(edge.curve_3d, EdgeCurve::Line) {
                    if let Some((u, _)) = surface.project_point(edge.start_3d) {
                        su = u;
                        break;
                    }
                }
            }
            su
        };
        let anti_u = (seam_u + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU);

        for edge in &boundary_edges {
            if (edge.start_3d - edge.end_3d).length() < 1e-10 {
                // Closed edge: find the 3D point at u = seam_u + pi on the surface.
                // Project the boundary vertex to get v, then evaluate surface at (anti_u, v).
                if let Some((_, v)) = surface.project_point(edge.start_3d) {
                    if let Some(anti_pt) = surface.evaluate(anti_u, v) {
                        split_pts_3d.push(anti_pt);
                    }
                }
            }
        }
    }

    let boundary_edges = split_boundary_edges_at_3d_points(
        boundary_edges,
        &split_pts_3d,
        if is_plane { Some(frame) } else { None },
        &surface,
        tol.linear,
    );

    // Reorder boundary edges: Line (seam) edges first, then curved (circle)
    // edges. This ensures the wire builder starts loops from seam edges,
    // forming rectangular bands before circle arcs can self-close.
    let boundary_edges = if u_periodic && !sections.is_empty() {
        let (mut lines, curves): (Vec<_>, Vec<_>) = boundary_edges
            .into_iter()
            .partition(|e| matches!(e.curve_3d, EdgeCurve::Line));
        lines.extend(curves);
        lines
    } else {
        boundary_edges
    };

    let boundary_edges_backup = if is_plane && sections.len() >= 2 {
        Some(boundary_edges.clone())
    } else {
        None
    };

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    for section in sections {
        let pcurve_on_this_face = match rank {
            Rank::A => &section.pcurve_a,
            Rank::B => &section.pcurve_b,
        };

        // Skip full-circle section edges on plane faces -- they have
        // start approx end in 3D and would produce degenerate UV edges.
        // The half-arc section edges handle the plane face correctly.
        let is_closed_edge = (section.start - section.end).length() < 1e-10;
        if is_closed_edge && is_plane {
            continue;
        }

        // Project section endpoints to UV.
        // Use pre-computed UV endpoints when available (e.g. seam-split half-arcs
        // where the unwrapped UV was computed from the arc samples). Otherwise,
        // for non-plane faces, use the pcurve's endpoint evaluations instead
        // of independent surface projection -- this ensures UV endpoints are
        // consistent with the pcurve's unwrapped parameterization (e.g. arc
        // ending at u=2pi rather than u=0 after periodic unwrapping).
        let (start_uv, end_uv) = match rank {
            Rank::A => {
                if let (Some(su), Some(eu)) = (section.start_uv_a, section.end_uv_a) {
                    (su, eu)
                } else if is_plane {
                    (frame.project(section.start), frame.project(section.end))
                } else {
                    uv_endpoints_from_pcurve(
                        pcurve_on_this_face,
                        section.start,
                        section.end,
                        &surface,
                        &wire_pts,
                    )
                }
            }
            Rank::B => {
                if let (Some(su), Some(eu)) = (section.start_uv_b, section.end_uv_b) {
                    (su, eu)
                } else if is_plane {
                    (frame.project(section.start), frame.project(section.end))
                } else {
                    uv_endpoints_from_pcurve(
                        pcurve_on_this_face,
                        section.start,
                        section.end,
                        &surface,
                        &wire_pts,
                    )
                }
            }
        };

        // Forward direction. Both forward and reverse share the same
        // source_edge_idx so build_topology_face creates one shared edge.
        let section_idx = all_edges.len();
        let pb_id = section.pave_block_id;
        all_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
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
            source_edge_idx: Some(section_idx),
            pave_block_id: pb_id,
        });
    }

    // Build wire loops via angular-sorting traversal.
    let loops = build_wire_loops(&all_edges, tol.linear, u_periodic, v_periodic);

    // Fallback: wire builder produced only 1 loop despite having 2+ section
    // edges that cross in the face interior. Use direct geometric quadrant
    // construction. The wire builder struggles with 4-way junctions when
    // boundary edges have inconsistent winding.
    if loops.len() <= 1 && sections.len() >= 2 && is_plane {
        if let Some(ref boundary) = boundary_edges_backup {
            if let Some(result) = try_split_crossing_plane_face(
                &surface, boundary, sections, rank, reversed, face_id, frame, tol,
            ) {
                return result;
            }
        }
    }

    // Classify each loop as outer (positive area) or hole (negative).
    // For loops with curved edges, sample intermediate UV points to get
    // an accurate area -- using only start_uv gives degenerate polygons
    // for 2-edge circles.
    let mut outers: Vec<(Vec<OrientedPCurveEdge>, f64)> = Vec::new();
    let mut holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    let u_per_opt = if u_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };
    let v_per_opt = if v_periodic {
        Some(std::f64::consts::TAU)
    } else {
        None
    };

    // For periodic faces with section edges, use structural classification
    // instead of signed area. Band loops (containing seam + section edges)
    // are outer wires. Circle-only self-loops are holes. Signed area on
    // periodic surfaces is unreliable because UV wraps around the period.
    let use_structural_classification = u_periodic && !sections.is_empty();

    for wire_loop in loops {
        if use_structural_classification {
            // Structural: a loop containing both Line edges (seam) and
            // non-Line edges (section arcs / circles) is a band = outer.
            let has_line = wire_loop
                .iter()
                .any(|e| matches!(e.curve_3d, EdgeCurve::Line));
            let has_nonline = wire_loop
                .iter()
                .any(|e| !matches!(e.curve_3d, EdgeCurve::Line));
            if has_line && has_nonline {
                outers.push((wire_loop, 1.0)); // area placeholder
            } else {
                holes.push(wire_loop);
            }
        } else {
            let pts = sample_wire_loop_uv_periodic(&wire_loop, u_per_opt, v_per_opt);
            let area = signed_area_2d(&pts);
            if area > 0.0 {
                outers.push((wire_loop, area));
            } else {
                holes.push(wire_loop);
            }
        }
    }

    // If all loops are CW (negative area), the winding is reversed.
    if !use_structural_classification && outers.is_empty() && !holes.is_empty() {
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
    let mut sub_faces = Vec::new();
    for (outer_wire, _area) in outers {
        sub_faces.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        });
    }

    // Simple hole matching: each hole goes to the outer that contains its
    // first vertex (via 2D point-in-polygon). Uses sampled UV points for
    // accurate containment with curved outer wires.
    for hole in holes {
        if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
            let mut assigned = false;
            for sf in &mut sub_faces {
                let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                    sf.inner_wires.push(hole.clone());
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                if let Some(sf) = sub_faces.first_mut() {
                    sf.inner_wires.push(hole);
                }
            }
        }
    }

    // Distribute original inner wires (holes from the source face) to sub-faces.
    // Each hole is assigned to the sub-face whose outer wire contains it.
    if !original_inner_wires.is_empty() {
        for hole in &original_inner_wires {
            if let Some(first_pt) = hole.first().map(|e| e.start_uv) {
                let mut assigned = false;
                for sf in &mut sub_faces {
                    let outer_pts = sample_wire_loop_uv(&sf.outer_wire);
                    if super::classify_2d::point_in_polygon_2d(first_pt, &outer_pts) {
                        sf.inner_wires.push(hole.clone());
                        assigned = true;
                        break;
                    }
                }
                if !assigned {
                    log::warn!(
                        "face_splitter: hole with {} edges could not be assigned to any sub-face",
                        hole.len()
                    );
                }
            }
        }
    }

    sub_faces
}

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// not inside any inner wire (hole), then evaluate it to 3D via the surface.
#[allow(clippy::too_many_lines)]
pub fn interior_point_3d(sub_face: &SplitSubFace, frame: Option<&PlaneFrame>) -> Point3 {
    let pts_2d = sample_wire_loop_uv(&sub_face.outer_wire);
    let mut interior_uv = sample_interior_point(&pts_2d);

    // Sphere cap fix: sphere sub-faces with degenerate UV boundaries (thin
    // strip at constant v) need the interior UV offset toward the pole.
    // The outer wire of a sphere cap maps to a horizontal line in UV,
    // producing a near-zero-area polygon whose centroid lies on the boundary.
    if let FaceSurface::Sphere(_) = &sub_face.surface {
        if !pts_2d.is_empty() {
            let v_min = pts_2d.iter().map(|p| p.y()).fold(f64::INFINITY, f64::min);
            let v_max = pts_2d
                .iter()
                .map(|p| p.y())
                .fold(f64::NEG_INFINITY, f64::max);
            if (v_max - v_min) < 0.1 {
                let v_boundary = (v_min + v_max) * 0.5;
                let v_pole = if v_boundary >= 0.0 {
                    std::f64::consts::FRAC_PI_2
                } else {
                    -std::f64::consts::FRAC_PI_2
                };
                let u_center = pts_2d.iter().map(|p| p.x()).sum::<f64>() / pts_2d.len() as f64;
                interior_uv = Point2::new(u_center, (v_boundary + v_pole) * 0.5);
            }
        }
    }

    // If the point falls inside a hole, find a point between the outer wire
    // and the nearest hole boundary.
    if is_inside_any_hole(&interior_uv, &sub_face.inner_wires) {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires);
    }

    // Secondary hole check: sample_wire_loop_uv for curved hole wires may
    // produce an under-sampled polygon that misses containment. Cross-check
    // using the hole's 3D boundary: if the interior 3D point is close to
    // the centroid of any hole, it's likely inside and needs displacement.
    if !sub_face.inner_wires.is_empty() {
        let eval_3d = |uv: Point2| -> Option<Point3> {
            if let Some(p) = sub_face.surface.evaluate(uv.x(), uv.y()) {
                return Some(p);
            }
            if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
                if let Some(f) = frame {
                    return Some(f.evaluate(uv.x(), uv.y()));
                }
                let wire_pts: Vec<Point3> =
                    sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
                let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
                return Some(f.evaluate(uv.x(), uv.y()));
            }
            None
        };

        if let Some(test_3d) = eval_3d(interior_uv) {
            for hole in &sub_face.inner_wires {
                // Compute hole centroid in 3D.
                if hole.is_empty() {
                    continue;
                }
                let hc: Point3 = {
                    let sum = hole.iter().fold(Point3::new(0.0, 0.0, 0.0), |acc, e| {
                        acc + (e.start_3d - Point3::new(0.0, 0.0, 0.0))
                    });
                    let n = hole.len() as f64;
                    Point3::new(sum.x() / n, sum.y() / n, sum.z() / n)
                };
                // Compute hole boundary radius from centroid.
                let max_r = hole
                    .iter()
                    .map(|e| (e.start_3d - hc).length())
                    .fold(0.0_f64, f64::max);

                if (test_3d - hc).length() < max_r * 0.95 {
                    // Interior point is inside the hole in 3D. Try outer wire
                    // vertex that's farthest from the hole centroid.
                    let best = sub_face
                        .outer_wire
                        .iter()
                        .max_by(|a, b| {
                            let da = (a.start_3d - hc).length();
                            let db = (b.start_3d - hc).length();
                            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|e| e.start_uv);
                    if let Some(uv) = best {
                        // Nudge slightly toward the centroid so the point
                        // is strictly interior, not on the boundary vertex.
                        interior_uv = Point2::new(
                            uv.x() * 0.95 + interior_uv.x() * 0.05,
                            uv.y() * 0.95 + interior_uv.y() * 0.05,
                        );
                    }
                    break;
                }
            }
        }
    }

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return p;
    }

    // For plane faces, evaluate via PlaneFrame.
    if let FaceSurface::Plane { normal, .. } = &sub_face.surface {
        if let Some(f) = frame {
            return f.evaluate(interior_uv.x(), interior_uv.y());
        }
        let wire_pts: Vec<Point3> = sub_face.outer_wire.iter().map(|e| e.start_3d).collect();
        let f = PlaneFrame::from_plane_face(*normal, &wire_pts);
        return f.evaluate(interior_uv.x(), interior_uv.y());
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

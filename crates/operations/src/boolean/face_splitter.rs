//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SubFace`]s.

#![allow(dead_code)] // Used by later boolean_v2 pipeline stages.

use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::classify_2d::{sample_interior_point, signed_area_2d};
use super::pcurve_compute::{
    compute_pcurve_on_surface, evaluate_edge_at_t, project_point_on_surface, sample_edge_to_uv,
};
use super::pipeline::{OrientedPCurveEdge, SectionEdge, SubFace, SurfaceInfo};
use super::plane_frame::PlaneFrame;
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
/// - `tol` — tolerance (`.linear` for 3D matching, UV tol derived internally)
/// - `frame` — cached `PlaneFrame` for this face (avoids origin mismatch)
/// - `info` — cached `SurfaceInfo` for periodicity flags
#[allow(clippy::too_many_lines)]
pub fn split_face_2d(
    topo: &Topology,
    face_id: FaceId,
    sections: &[SectionEdge],
    source: Source,
    tol: &brepkit_math::tolerance::Tolerance,
    frame: Option<&PlaneFrame>,
    info: Option<&SurfaceInfo>,
) -> Vec<SubFace> {
    let face = match topo.face(face_id) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let surface = face.surface().clone();
    let reversed = face.is_reversed();
    let is_plane = matches!(surface, brepkit_topology::face::FaceSurface::Plane { .. });

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
        // For non-plane faces, PlaneFrame is not used — set a dummy.
        // All UV projection goes through surface.project_point().
        owned_frame = PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &[]);
        &owned_frame
    };

    // Extract periodicity from SurfaceInfo.
    // Periodic quantization is needed for boundary wire connectivity (circle
    // end at u=2π connects to seam start at u=0). Keep it enabled.
    let (u_periodic, v_periodic) = info.map_or((false, false), SurfaceInfo::periodicity);

    // Convert boundary edges to OrientedPCurveEdge.
    let mut boundary_edges = if is_plane {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, Some(frame))
    } else {
        boundary_edges_to_pcurve(topo, face.outer_wire(), &surface, &wire_pts, None)
    };

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

    // Sphere cap shortcut: sphere hemisphere faces have no seam edges (only
    // equatorial Line edges), so the wire builder can't form proper bands.
    // Construct cap + band sub-faces directly instead.
    // Guard: only apply when ALL boundary edges are equatorial Lines (no seam edges).
    let all_boundary_line = boundary_edges
        .iter()
        .all(|e| matches!(e.curve_3d, EdgeCurve::Line));
    if matches!(surface, brepkit_topology::face::FaceSurface::Sphere(_)) && all_boundary_line {
        return split_sphere_face_direct(
            &surface,
            &boundary_edges,
            sections,
            source,
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
    // Heuristic: NURBS section edges on non-planar faces come from
    // analytic-analytic intersection (e.g., cylinder-cylinder) and form
    // internal loops. Circle/Ellipse/Line sections come from plane-analytic
    // intersection and connect to boundary edges.
    let all_sections_internal = if !is_plane && !sections.is_empty() {
        sections
            .iter()
            .all(|s| matches!(s.curve_3d, EdgeCurve::NurbsCurve(_)))
    } else {
        false
    };

    if all_sections_internal {
        return split_face_with_internal_loops(
            &surface,
            &boundary_edges,
            sections,
            source,
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
    // in the surface's UV parameterization (u = seam_u + π).
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
                // Closed edge: find the 3D point at u = seam_u + π on the surface.
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

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    for section in sections {
        let pcurve_on_this_face = match source {
            Source::A => &section.pcurve_a,
            Source::B => &section.pcurve_b,
        };

        // Skip full-circle section edges on plane faces — they have
        // start ≈ end in 3D and would produce degenerate UV edges.
        // The half-arc section edges handle the plane face correctly.
        let is_closed_edge = (section.start - section.end).length() < 1e-10;
        if is_closed_edge && is_plane {
            continue;
        }

        // Project section endpoints to UV.
        // Use pre-computed UV endpoints when available (e.g. seam-split half-arcs
        // where the unwrapped UV was computed from the arc samples). Otherwise,
        // for non-plane faces, use the pcurve's endpoint evaluations instead
        // of independent surface projection — this ensures UV endpoints are
        // consistent with the pcurve's unwrapped parameterization (e.g. arc
        // ending at u=2π rather than u=0 after periodic unwrapping).
        let (start_uv, end_uv) = match source {
            Source::A => {
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
            Source::B => {
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
    let loops = build_wire_loops(&all_edges, tol.linear, u_periodic, v_periodic);

    // Classify each loop as outer (positive area) or hole (negative).
    // For loops with curved edges, sample intermediate UV points to get
    // an accurate area — using only start_uv gives degenerate polygons
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
        sub_faces.push(SubFace {
            surface: surface.clone(),
            outer_wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            source,
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

    sub_faces
}

/// Get a point guaranteed inside a sub-face's outer wire (in UV space),
/// not inside any inner wire (hole), then evaluate it to 3D via the surface.
#[allow(clippy::too_many_lines)]
pub fn interior_point_3d(sub_face: &SubFace, frame: Option<&PlaneFrame>) -> Point3 {
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

/// Sample UV points along a wire loop, interpolating along curved edges.
///
/// For line edges, uses only the start point. For curved edges (Circle,
/// Ellipse, NurbsCurve), samples N intermediate points to approximate the
/// true curve shape in UV. This is critical for signed area computation
/// and point-in-polygon tests on loops with curved edges.
fn sample_wire_loop_uv(wire: &[OrientedPCurveEdge]) -> Vec<Point2> {
    sample_wire_loop_uv_periodic(wire, None, None)
}

/// Sample UV points along a wire loop with optional periodic unwrapping.
///
/// When `u_period`/`v_period` is set, unwraps consecutive points so the
/// UV path is continuous (no jumps of ~2π between edges connected via
/// periodic quantization).
fn sample_wire_loop_uv_periodic(
    wire: &[OrientedPCurveEdge],
    u_period: Option<f64>,
    v_period: Option<f64>,
) -> Vec<Point2> {
    use brepkit_math::curves2d::Curve2D;
    const CURVE_SAMPLES: usize = 8;

    let mut pts = Vec::new();
    let has_period = u_period.is_some() || v_period.is_some();
    for edge in wire {
        match &edge.pcurve {
            Curve2D::Line(_) => {
                // For periodic surfaces, push both start and end to enable
                // proper unwrapping across periodic jumps at seam vertices.
                pts.push(edge.start_uv);
                if has_period {
                    pts.push(edge.end_uv);
                }
            }
            Curve2D::Nurbs(nurbs) => {
                let knots = nurbs.knots();
                if knots.len() >= 2 {
                    let t0 = knots[0];
                    let tn = knots[knots.len() - 1];
                    // For reverse edges, the pcurve was computed for the forward
                    // direction. Evaluate from tn→t0 to trace the reverse path.
                    #[allow(clippy::cast_precision_loss)]
                    for i in 0..CURVE_SAMPLES {
                        let frac = i as f64 / CURVE_SAMPLES as f64;
                        let t = if edge.forward {
                            t0 + (tn - t0) * frac
                        } else {
                            tn - (tn - t0) * frac
                        };
                        pts.push(nurbs.evaluate(t));
                    }
                } else {
                    pts.push(edge.start_uv);
                }
            }
            Curve2D::Circle(_) | Curve2D::Ellipse(_) => {
                // Circle2D/Ellipse2D pcurves: interpolate between start_uv
                // and end_uv. This is approximate (chord, not arc) but these
                // pcurve types are rare in the boolean_v2 pipeline — section
                // edges use NURBS and boundary edges use Line2D.
                #[allow(clippy::cast_precision_loss)]
                for i in 0..CURVE_SAMPLES {
                    let t = i as f64 / CURVE_SAMPLES as f64;
                    pts.push(Point2::new(
                        edge.start_uv.x() + (edge.end_uv.x() - edge.start_uv.x()) * t,
                        edge.start_uv.y() + (edge.end_uv.y() - edge.start_uv.y()) * t,
                    ));
                }
            }
        }
    }

    // Unwrap periodic UV jumps between consecutive points.
    if pts.len() >= 2 {
        super::pcurve_compute::unwrap_periodic_params_pub(&mut pts, u_period, v_period);
    }

    pts
}

/// Check if a UV point is inside any of the inner wire (hole) polygons.
fn is_inside_any_hole(pt: &Point2, inner_wires: &[Vec<OrientedPCurveEdge>]) -> bool {
    for hole in inner_wires {
        let hole_pts = sample_wire_loop_uv(hole);
        if hole_pts.len() >= 3 && super::classify_2d::point_in_polygon_2d(*pt, &hole_pts) {
            return true;
        }
    }
    false
}

/// Find a UV point inside the outer wire but outside all holes.
///
/// Tries midpoints between outer wire vertices and the centroid of the first
/// hole. Falls back to midpoints of outer wire edges nudged outward from holes.
fn find_point_outside_holes(
    outer_pts: &[Point2],
    inner_wires: &[Vec<OrientedPCurveEdge>],
) -> Point2 {
    // Strategy: take midpoints between outer wire edge midpoints and the outer
    // boundary — these are likely in the ring region between outer and inner.
    let centroid_x = outer_pts.iter().map(|p| p.x()).sum::<f64>() / outer_pts.len() as f64;
    let centroid_y = outer_pts.iter().map(|p| p.y()).sum::<f64>() / outer_pts.len() as f64;
    for i in 0..outer_pts.len() {
        let j = (i + 1) % outer_pts.len();
        let edge_mid = Point2::new(
            (outer_pts[i].x() + outer_pts[j].x()) * 0.5,
            (outer_pts[i].y() + outer_pts[j].y()) * 0.5,
        );
        // Nudge the edge midpoint slightly toward the centroid.
        let candidate = Point2::new(
            edge_mid.x() * 0.9 + centroid_x * 0.1,
            edge_mid.y() * 0.9 + centroid_y * 0.1,
        );
        if super::classify_2d::point_in_polygon_2d(candidate, outer_pts)
            && !is_inside_any_hole(&candidate, inner_wires)
        {
            return candidate;
        }
    }

    // Fallback: try vertex midpoints between consecutive outer wire vertices.
    if outer_pts.len() >= 2 {
        let mid = Point2::new(
            (outer_pts[0].x() + outer_pts[1].x()) * 0.5,
            (outer_pts[0].y() + outer_pts[1].y()) * 0.5,
        );
        return mid;
    }

    // Ultimate fallback: centroid (even though it may be in a hole).
    let n = outer_pts.len() as f64;
    Point2::new(
        outer_pts.iter().map(|p| p.x()).sum::<f64>() / n,
        outer_pts.iter().map(|p| p.y()).sum::<f64>() / n,
    )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Split boundary edges at 3D points where section edges start/end.
///
/// Handles Line, Circle, and Ellipse edges. For curved edges, projects
/// split points onto the curve via `Circle3D::project` / `Ellipse3D::project`
/// and checks distance from the curve. Creates sub-arc edges with pcurves
/// computed via sampling.
#[allow(clippy::too_many_lines)]
fn split_boundary_edges_at_3d_points(
    edges: Vec<OrientedPCurveEdge>,
    split_pts_3d: &[Point3],
    frame: Option<&PlaneFrame>,
    surface: &brepkit_topology::face::FaceSurface,
    tol: f64,
) -> Vec<OrientedPCurveEdge> {
    let mut result = Vec::new();
    for edge in edges {
        let splits = match &edge.curve_3d {
            EdgeCurve::Circle(circle) => find_splits_on_circle(circle, &edge, split_pts_3d, tol),
            EdgeCurve::Ellipse(ellipse) => {
                find_splits_on_ellipse(ellipse, &edge, split_pts_3d, tol)
            }
            _ => find_splits_on_line(&edge, split_pts_3d, tol),
        };

        if splits.is_empty() {
            result.push(edge);
            continue;
        }

        // Split edge into segments.
        let mut prev_uv = edge.start_uv;
        let mut prev_3d = edge.start_3d;
        for &(t, _) in &splits {
            let split_3d = evaluate_edge_at_t(&edge.curve_3d, edge.start_3d, edge.end_3d, t);
            let split_uv = if let Some(f) = frame {
                f.project(split_3d)
            } else {
                project_point_on_surface(split_3d, surface, &[], None)
            };
            let pcurve =
                compute_pcurve_on_surface(&edge.curve_3d, prev_3d, split_3d, surface, &[], frame);
            result.push(OrientedPCurveEdge {
                curve_3d: edge.curve_3d.clone(),
                pcurve,
                start_uv: prev_uv,
                end_uv: split_uv,
                start_3d: prev_3d,
                end_3d: split_3d,
                forward: edge.forward,
            });
            prev_uv = split_uv;
            prev_3d = split_3d;
        }
        // Final segment.
        let pcurve =
            compute_pcurve_on_surface(&edge.curve_3d, prev_3d, edge.end_3d, surface, &[], frame);
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

/// Find split parameters on a line edge. Returns `(t, split_3d)` sorted by `t`.
fn find_splits_on_line(
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let edge_dir = edge.end_3d - edge.start_3d;
    let edge_len_sq = edge_dir.dot(edge_dir);
    if edge_len_sq < tol * tol {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        let to_pt = sp - edge.start_3d;
        let t = to_pt.dot(edge_dir) / edge_len_sq;
        if t <= tol || t >= 1.0 - tol {
            continue;
        }
        let closest = edge.start_3d + edge_dir * t;
        let dist = (sp - closest).length();
        if dist < tol {
            splits.push((t, sp));
        }
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Extract UV endpoints from a pcurve's evaluation rather than independent
/// surface projection. This ensures consistency — e.g. a pcurve that goes
/// from (π, v) to (2π, v) won't have its end snapped to (0, v) by the
/// surface's `project_point` which normalizes u into `[0, 2π)`.
fn uv_endpoints_from_pcurve(
    pcurve: &brepkit_math::curves2d::Curve2D,
    start_3d: Point3,
    end_3d: Point3,
    surface: &brepkit_topology::face::FaceSurface,
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
            // endpoint by more than π (half a period), the line direction
            // is wrong — fall back to direct projection.
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

/// Find split parameters on a circle edge. Uses `Circle3D::project` for angular
/// projection, then normalizes into the edge's `[0, 1]` parameter range.
///
/// Note: `domain_with_endpoints` for full circles (start ≈ end) returns the
/// full `(-π, π]` domain. For true arcs, it uses endpoint projection — this
/// is correct for the boundary edges produced by `make_cylinder`/`make_cone`.
fn find_splits_on_circle(
    circle: &brepkit_math::curves::Circle3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let (t0, t1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
    let span = t1 - t0;
    if span.abs() < 1e-14 {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        let angle = circle.project(sp);
        let closest = circle.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = normalize_angle_in_span(angle, t0, span);
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Find split parameters on an ellipse edge.
fn find_splits_on_ellipse(
    ellipse: &brepkit_math::curves::Ellipse3D,
    edge: &OrientedPCurveEdge,
    split_pts_3d: &[Point3],
    tol: f64,
) -> Vec<(f64, Point3)> {
    let (t0, t1) = edge
        .curve_3d
        .domain_with_endpoints(edge.start_3d, edge.end_3d);
    let span = t1 - t0;
    if span.abs() < 1e-14 {
        return Vec::new();
    }
    let mut splits = Vec::new();
    for &sp in split_pts_3d {
        let angle = ellipse.project(sp);
        let closest = ellipse.evaluate(angle);
        if (sp - closest).length() > tol {
            continue;
        }
        let t_norm = normalize_angle_in_span(angle, t0, span);
        if t_norm <= tol || t_norm >= 1.0 - tol {
            continue;
        }
        splits.push((t_norm, sp));
    }
    splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    splits.dedup_by(|a, b| (a.0 - b.0).abs() < tol);
    splits
}

/// Normalize an angle into the `[0, 1]` parameter range of an edge span.
///
/// `t0` is the start angle, `span = t1 - t0` is the signed angular range.
/// Returns `(angle - t0) / span`, wrapping by 2π to stay within the arc.
fn normalize_angle_in_span(angle: f64, t0: f64, span: f64) -> f64 {
    use std::f64::consts::TAU;
    let mut delta = angle - t0;
    if span > 0.0 {
        // CCW arc: delta should be in [0, span].
        // At most 2 wraps needed (angle is in (-π, π]).
        for _ in 0..3 {
            if delta >= -1e-10 {
                break;
            }
            delta += TAU;
        }
        for _ in 0..3 {
            if delta <= span + 1e-10 {
                break;
            }
            delta -= TAU;
        }
    } else {
        // CW arc: delta should be in [span, 0].
        for _ in 0..3 {
            if delta <= 1e-10 {
                break;
            }
            delta -= TAU;
        }
        for _ in 0..3 {
            if delta >= span - 1e-10 {
                break;
            }
            delta += TAU;
        }
    }
    delta / span
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
        Vec3::new(0.0, 0.0, 1.0)
    }
}

fn boundary_edges_to_pcurve(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &brepkit_topology::face::FaceSurface,
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

        // For closed edges (start_3d ≈ end_3d, e.g. full circle), projecting
        // start and end to UV gives the same point. Use pcurve sampling to
        // get distinct UV endpoints spanning the full curve.
        let is_closed = (start_3d - end_3d).length() < 1e-10;
        let (start_uv, end_uv) =
            if is_closed && !matches!(surface, brepkit_topology::face::FaceSurface::Plane { .. }) {
                let uv_samples = sample_edge_to_uv(edge.curve(), start_3d, end_3d, surface);
                let su = uv_samples.first().copied().unwrap_or_else(|| {
                    project_point_on_surface(start_3d, surface, wire_pts, frame)
                });
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
        });
    }
    result
}

// ---------------------------------------------------------------------------
// Sphere-specific face splitting
// ---------------------------------------------------------------------------

/// Split a sphere hemisphere face directly into cap + band sub-faces.
///
/// Sphere hemispheres have no seam edges — their boundary is entirely equatorial
/// Line edges. The generic wire builder can't form proper rectangular bands
/// because there are no vertical connections between the equator (v≈0) and the
/// section circle (at some other v). This function bypasses the wire builder
/// and constructs the sub-faces geometrically:
///
/// - **Cap**: bounded by the section circle (2 half-arcs), covers the pole side.
/// - **Band**: bounded by the equatorial boundary, with the section circle as hole.
#[allow(clippy::too_many_arguments)]
fn split_sphere_face_direct(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    source: Source,
    reversed: bool,
    face_id: FaceId,
    wire_pts: &[Point3],
) -> Vec<SubFace> {
    // Helper: return the face unsplit (used in fallback paths).
    let unsplit = || {
        vec![SubFace {
            surface: surface.clone(),
            outer_wire: boundary_edges.to_vec(),
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            source,
        }]
    };

    // Collect section forward/reverse edges on this face.
    let mut cap_edges = Vec::new();
    let mut hole_edges = Vec::new();

    for section in sections {
        let pcurve_on_this_face = match source {
            Source::A => &section.pcurve_a,
            Source::B => &section.pcurve_b,
        };

        // Skip full-circle section edges (start ≈ end in 3D) — only use
        // the half-arcs produced by build_seam_split_sections.
        if (section.start - section.end).length() < brepkit_math::tolerance::Tolerance::new().linear
        {
            continue;
        }

        let precomputed_uv = match source {
            Source::A => section.start_uv_a.zip(section.end_uv_a),
            Source::B => section.start_uv_b.zip(section.end_uv_b),
        };
        let (start_uv, end_uv) = precomputed_uv.unwrap_or_else(|| {
            uv_endpoints_from_pcurve(
                pcurve_on_this_face,
                section.start,
                section.end,
                surface,
                wire_pts,
            )
        });

        // Forward: for the cap outer wire.
        cap_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv,
            end_uv,
            start_3d: section.start,
            end_3d: section.end,
            forward: true,
        });

        // Reverse: for the band's inner wire (hole).
        hole_edges.push(OrientedPCurveEdge {
            curve_3d: section.curve_3d.clone(),
            pcurve: pcurve_on_this_face.clone(),
            start_uv: end_uv,
            end_uv: start_uv,
            start_3d: section.end,
            end_3d: section.start,
            forward: false,
        });
    }

    if cap_edges.is_empty() {
        // No valid section edges — return the face unsplit.
        return unsplit();
    }

    // Validate: cap edges must form a single closed loop (last end ≈ first start).
    // If the topology is unexpected (multiple loops, open chain), fall back to unsplit.
    let loop_gap = (cap_edges
        .last()
        .map_or(Point3::new(0.0, 0.0, 0.0), |e| e.end_3d)
        - cap_edges
            .first()
            .map_or(Point3::new(0.0, 0.0, 0.0), |e| e.start_3d))
    .length();
    if loop_gap > brepkit_math::tolerance::Tolerance::new().linear * 100.0 {
        return unsplit();
    }

    // Cap sub-face: outer wire = section forward half-arcs.
    // The half-arcs connect end-to-end, forming a closed loop (the section circle).
    // Band sub-face: outer wire = equatorial boundary, inner wire = section reversed.
    vec![
        SubFace {
            surface: surface.clone(),
            outer_wire: cap_edges,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            source,
        },
        SubFace {
            surface: surface.clone(),
            outer_wire: boundary_edges.to_vec(),
            inner_wires: vec![hole_edges],
            reversed,
            parent: face_id,
            source,
        },
    ]
}

// ---------------------------------------------------------------------------
// Internal-loop face splitting
// ---------------------------------------------------------------------------

/// Split a face when ALL section edges are interior (don't touch the boundary).
///
/// Groups section edges into closed loops by chaining shared 3D endpoints.
/// Each closed loop produces:
/// - An "inside" sub-face with the loop as outer wire
/// - A reversed copy added as an inner wire (hole) of the "outside" sub-face
///
/// The "outside" sub-face has the original boundary as outer wire with all
/// loops as holes.
#[allow(clippy::too_many_arguments)]
fn split_face_with_internal_loops(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    source: Source,
    reversed: bool,
    face_id: FaceId,
    _wire_pts: &[Point3],
) -> Vec<SubFace> {
    let tol_3d = brepkit_math::tolerance::Tolerance::new().linear;

    // Convert each section edge to a polyline of Line edges.
    // Using NURBS edges on analytic surfaces causes tessellation failures
    // (the tessellator uses a rectangular UV grid that ignores NURBS boundaries).
    // Polyline approximation with ~32 Line segments per half-arc preserves
    // the cylindrical surface while giving correct boundary polygons.
    let mut forward_edges: Vec<OrientedPCurveEdge> = Vec::new();
    let n_seg = 32_usize;

    for section in sections {
        // Sample the NURBS curve at n_seg+1 points.
        let sample_pts: Vec<Point3> = match &section.curve_3d {
            EdgeCurve::NurbsCurve(nc) => {
                let domain = nc.domain();
                (0..=n_seg)
                    .map(|i| {
                        #[allow(clippy::cast_precision_loss)]
                        let t = domain.0 + (domain.1 - domain.0) * (i as f64 / n_seg as f64);
                        nc.evaluate(t)
                    })
                    .collect()
            }
            _ => {
                // For non-NURBS curves, just use start and end.
                vec![section.start, section.end]
            }
        };

        // Project sample points to UV, unwrapping u across the seam.
        let u_period = match surface {
            FaceSurface::Cylinder(_)
            | FaceSurface::Cone(_)
            | FaceSurface::Sphere(_)
            | FaceSurface::Torus(_) => Some(std::f64::consts::TAU),
            _ => None,
        };

        let mut uv_samples: Vec<(f64, f64)> = Vec::with_capacity(sample_pts.len());
        for &pt in &sample_pts {
            let (mut u, v) = surface.project_point(pt).unwrap_or((0.0, 0.0));
            // Unwrap u to be continuous with the previous sample.
            if let (Some(period), Some(&(prev_u, _))) = (u_period, uv_samples.last()) {
                let diff = u - prev_u;
                let shifts = (diff / period + 0.5).floor();
                u -= shifts * period;
            }
            uv_samples.push((u, v));
        }

        // Create Line edges between consecutive samples.
        for i in 0..sample_pts.len() - 1 {
            let p0 = sample_pts[i];
            let p1 = sample_pts[i + 1];
            let uv0 = uv_samples[i];
            let uv1 = uv_samples[i + 1];

            let dir = Vec2::new(uv1.0 - uv0.0, uv1.1 - uv0.1);
            let pcurve = if let Ok(line) =
                brepkit_math::curves2d::Line2D::new(Point2::new(uv0.0, uv0.1), dir)
            {
                brepkit_math::curves2d::Curve2D::Line(line)
            } else {
                // Degenerate — zero-length edge, skip.
                continue;
            };

            forward_edges.push(OrientedPCurveEdge {
                curve_3d: EdgeCurve::Line,
                pcurve,
                start_uv: Point2::new(uv0.0, uv0.1),
                end_uv: Point2::new(uv1.0, uv1.1),
                start_3d: p0,
                end_3d: p1,
                forward: true,
            });
        }
    }

    // Group edges into closed loops by chaining: edge.end_3d ≈ next.start_3d.
    let mut used = vec![false; forward_edges.len()];
    let mut loops: Vec<Vec<OrientedPCurveEdge>> = Vec::new();

    for start_idx in 0..forward_edges.len() {
        if used[start_idx] {
            continue;
        }
        used[start_idx] = true;
        let mut chain = vec![forward_edges[start_idx].clone()];
        let loop_start_3d = chain[0].start_3d;

        // Follow the chain until we close the loop.
        loop {
            let last_end = chain.last().map_or(loop_start_3d, |e| e.end_3d);

            // Check if the loop is closed.
            if chain.len() > 1 && (last_end - loop_start_3d).length() < tol_3d * 100.0 {
                break;
            }

            // Find the next unused edge whose start matches last_end.
            let next = forward_edges
                .iter()
                .enumerate()
                .find(|(i, e)| !used[*i] && (e.start_3d - last_end).length() < tol_3d * 100.0);

            if let Some((idx, _)) = next {
                used[idx] = true;
                chain.push(forward_edges[idx].clone());
            } else {
                break; // Can't continue — open chain.
            }
        }

        if chain.len() >= 2 {
            loops.push(chain);
        }
    }

    // Build sub-faces.
    let mut result = Vec::new();

    // For each closed loop: create an "inside" sub-face.
    // The loop winding determines which region of the face is enclosed.
    // We want the SMALLER region (the Steinmetz lobe), so check signed area
    // in UV and reverse if the loop encloses the larger region.
    let mut all_holes: Vec<Vec<OrientedPCurveEdge>> = Vec::new();
    for loop_edges in &mut loops {
        // Compute signed area in UV.
        let mut signed_area = 0.0;
        for edge in loop_edges.iter() {
            signed_area +=
                (edge.end_uv.x() - edge.start_uv.x()) * (edge.end_uv.y() + edge.start_uv.y());
        }
        // If signed area is positive (CW in standard UV), the loop encloses
        // the "right" region. If negative (CCW), it encloses the complement.
        // For interior loops on analytic faces, the Steinmetz lobe is the
        // SMALLER region. Check by comparing signed area magnitude to the
        // face's total UV area estimate. If > half, reverse.
        // Heuristic: use signed_area sign directly. The section edges are
        // ordered so that CCW in UV = exterior. Reverse to get interior.
        if signed_area < 0.0 {
            // CCW → enclosing exterior. Reverse to CW → interior.
            loop_edges.reverse();
            for edge in loop_edges.iter_mut() {
                std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                edge.forward = !edge.forward;
            }
        }

        // The loop as outer wire of the inside sub-face.
        result.push(SubFace {
            surface: surface.clone(),
            outer_wire: loop_edges.clone(),
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            source,
        });

        // Build reversed loop for the outside sub-face's hole.
        let hole: Vec<OrientedPCurveEdge> = loop_edges
            .iter()
            .rev()
            .map(|e| OrientedPCurveEdge {
                curve_3d: e.curve_3d.clone(),
                pcurve: e.pcurve.clone(),
                start_uv: e.end_uv,
                end_uv: e.start_uv,
                start_3d: e.end_3d,
                end_3d: e.start_3d,
                forward: !e.forward,
            })
            .collect();
        // Verify hole is closed.
        if let (Some(first), Some(last)) = (hole.first(), hole.last()) {
            if (last.end_3d - first.start_3d).length() < tol_3d * 100.0 {
                all_holes.push(hole);
            }
        }
    }

    // The "outside" sub-face: original boundary with all loops as holes.
    result.push(SubFace {
        surface: surface.clone(),
        outer_wire: boundary_edges.to_vec(),
        inner_wires: all_holes,
        reversed,
        parent: face_id,
        source,
    });

    result
}

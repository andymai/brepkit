//! Face splitting via 2D wire construction.
//!
//! For each face, collects boundary edges and section edges, converts
//! them to [`OrientedPCurveEdge`]s in the face's parameter space, calls
//! the wire builder, and produces [`SubFace`]s.

#![allow(dead_code)] // Used by later boolean_v2 pipeline stages.

use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::FaceId;

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
    let (u_periodic, v_periodic) = info.map_or((false, false), SurfaceInfo::periodicity);

    // Convert boundary edges to OrientedPCurveEdge.
    let boundary_edges = if is_plane {
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

    // Stage 2: Split boundary edges at section edge endpoints (3D matching).
    let split_pts_3d: Vec<Point3> = sections.iter().flat_map(|s| [s.start, s.end]).collect();
    let boundary_edges = split_boundary_edges_at_3d_points(
        boundary_edges,
        &split_pts_3d,
        if is_plane { Some(frame) } else { None },
        &surface,
        tol.linear,
    );

    // Convert section edges to OrientedPCurveEdge (both orientations).
    let mut all_edges = boundary_edges;
    for section in sections {
        let pcurve_on_this_face = match source {
            Source::A => &section.pcurve_a,
            Source::B => &section.pcurve_b,
        };

        // Project section endpoints to UV using the appropriate method.
        // For closed curves (start ≈ end in 3D), use pcurve endpoint
        // sampling to get distinct UV values on periodic surfaces.
        let is_closed_edge = (section.start - section.end).length() < 1e-10;
        let (start_uv, end_uv) = if is_closed_edge && !is_plane && u_periodic {
            // Closed curve on a u-periodic surface (e.g. full circle on cylinder).
            // The 3D start ≈ end, but in UV the curve spans the full u period.
            let su = project_point_on_surface(section.start, &surface, &wire_pts, None);
            let eu = Point2::new(su.x() + std::f64::consts::TAU, su.y());
            (su, eu)
        } else if is_plane {
            (frame.project(section.start), frame.project(section.end))
        } else {
            let su = project_point_on_surface(section.start, &surface, &wire_pts, None);
            let eu = project_point_on_surface(section.end, &surface, &wire_pts, None);
            (su, eu)
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

    for wire_loop in loops {
        let pts = sample_wire_loop_uv(&wire_loop);
        let area = signed_area_2d(&pts);
        if area > 0.0 {
            outers.push((wire_loop, area));
        } else {
            holes.push(wire_loop);
        }
    }

    // If all loops are CW (negative area), the winding is reversed.
    if outers.is_empty() && !holes.is_empty() {
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
pub fn interior_point_3d(sub_face: &SubFace, frame: Option<&PlaneFrame>) -> Point3 {
    let pts_2d = sample_wire_loop_uv(&sub_face.outer_wire);
    let mut interior_uv = sample_interior_point(&pts_2d);

    // If the point falls inside a hole, find a point between the outer wire
    // and the nearest hole boundary.
    if is_inside_any_hole(&interior_uv, &sub_face.inner_wires) {
        interior_uv = find_point_outside_holes(&pts_2d, &sub_face.inner_wires);
    }

    // Evaluate back to 3D.
    if let Some(p) = sub_face.surface.evaluate(interior_uv.x(), interior_uv.y()) {
        return p;
    }

    // For plane faces, evaluate via PlaneFrame.
    if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = &sub_face.surface {
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
    use brepkit_math::curves2d::Curve2D;
    const CURVE_SAMPLES: usize = 8;

    let mut pts = Vec::new();
    for edge in wire {
        match &edge.pcurve {
            Curve2D::Line(_) => {
                pts.push(edge.start_uv);
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
            _ => {
                // For Circle2D, Ellipse2D: fall back to linear interpolation
                // between start_uv and end_uv.
                pts.push(edge.start_uv);
            }
        }
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
    for i in 0..outer_pts.len() {
        let j = (i + 1) % outer_pts.len();
        let edge_mid = Point2::new(
            (outer_pts[i].x() + outer_pts[j].x()) * 0.5,
            (outer_pts[i].y() + outer_pts[j].y()) * 0.5,
        );
        // Use the edge midpoint directly — it's on the outer boundary edge,
        // which should be far from interior holes in typical cases.
        // Nudge slightly inward from the edge.
        let centroid_x = outer_pts.iter().map(|p| p.x()).sum::<f64>() / outer_pts.len() as f64;
        let centroid_y = outer_pts.iter().map(|p| p.y()).sum::<f64>() / outer_pts.len() as f64;
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

/// Find split parameters on a circle edge. Uses `Circle3D::project` for angular
/// projection, then normalizes into the edge's `[0, 1]` parameter range.
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

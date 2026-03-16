//! Boolean v2: OCCT-style parameter-space pipeline.
//!
//! Operates entirely in 2D parameter space (pcurves on surfaces) for face
//! splitting. Surface-type agnostic — same code for plane, cylinder, sphere,
//! torus, NURBS. Step 1 implements plane-only support.

#![allow(dead_code, clippy::too_many_lines, clippy::missing_errors_doc)]

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec2, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use super::assembly::{quantize_point, vertex_merge_resolution};
use super::classify::try_build_analytic_classifier;
use super::classify_2d::point_in_polygon_2d;
use super::face_splitter::{interior_point_3d, split_face_2d};
use super::pipeline::{BooleanPipeline, SectionEdge};
use super::plane_frame::PlaneFrame;
use super::types::{AnalyticClassifier, BooleanOp, FaceClass, Source, select_fragment};
use crate::OperationsError;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Perform a boolean operation using the v2 parameter-space pipeline.
///
/// Currently supports solids composed entirely of `FaceSurface::Plane` faces.
/// Returns `Err` for solids containing non-plane faces.
pub fn boolean_v2(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
) -> Result<SolidId, OperationsError> {
    let tol = Tolerance::new();

    // Guard: all faces must be plane (step 1 limitation).
    if !all_faces_plane(topo, a)? || !all_faces_plane(topo, b)? {
        return Err(OperationsError::InvalidInput {
            reason: "boolean_v2 step 1: only plane faces supported".into(),
        });
    }

    let mut pipeline = BooleanPipeline {
        solid_a: Some(a),
        solid_b: Some(b),
        ..BooleanPipeline::default()
    };

    // Cache PlaneFrame per face (consistent UV origin across all stages).
    init_plane_frames(topo, a, b, &mut pipeline)?;

    // Build analytic classifiers for O(1) point-in-solid tests.
    let classifier_a = try_build_analytic_classifier(topo, a);
    let classifier_b = try_build_analytic_classifier(topo, b);

    // Stage 1: Intersect all face pairs.
    intersect_plane_faces(topo, a, b, &mut pipeline, &tol)?;

    // Disjoint shortcut (with containment detection).
    if pipeline.intersections.is_empty() {
        return handle_disjoint_v2(
            topo,
            op,
            a,
            b,
            classifier_a.as_ref(),
            classifier_b.as_ref(),
            &tol,
        );
    }

    // Stage 2: Split edges at intersection vertices.
    // For plane-plane, the boundary edges are pre-split during face splitting
    // (the wire builder handles the split edges naturally via section edges).
    // No explicit edge splitting needed in Step 1.

    // Stage 3: Split faces via wire builder.
    split_all_faces(topo, a, b, &mut pipeline, &tol)?;

    // Stage 4: Classify sub-faces against opposing solid.
    classify_sub_faces(
        topo,
        &mut pipeline,
        op,
        classifier_a.as_ref(),
        classifier_b.as_ref(),
    )?;

    // Stage 5: Assemble result.
    let result = assemble_v2(topo, &pipeline, &tol)?;

    // Post-processing: healing.
    crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    crate::heal::unify_faces(topo, result)?;

    Ok(result)
}

// ---------------------------------------------------------------------------
// PlaneFrame cache
// ---------------------------------------------------------------------------

fn init_plane_frames(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
) -> Result<(), OperationsError> {
    for solid in [a, b] {
        let faces = collect_solid_faces(topo, solid)?;
        for fid in faces {
            let face = topo.face(fid)?;
            if let FaceSurface::Plane { normal, .. } = face.surface() {
                let poly = collect_face_polygon(topo, fid)?;
                let frame = PlaneFrame::from_plane_face(*normal, &poly);
                pipeline.plane_frames.insert(fid, frame);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 1: Intersect plane faces
// ---------------------------------------------------------------------------

fn intersect_plane_faces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
    _tol: &Tolerance,
) -> Result<(), OperationsError> {
    let faces_a = collect_solid_faces(topo, a)?;
    let faces_b = collect_solid_faces(topo, b)?;

    for &fa in &faces_a {
        for &fb in &faces_b {
            let sections = intersect_two_plane_faces(topo, fa, fb, &pipeline.plane_frames)?;
            if !sections.is_empty() {
                pipeline.intersections.insert((fa, fb), sections);
            }
        }
    }
    Ok(())
}

/// Compute the intersection of two plane faces.
///
/// Returns trimmed section edges (finite line segments where the faces overlap).
fn intersect_two_plane_faces(
    topo: &Topology,
    fa: FaceId,
    fb: FaceId,
    plane_frames: &HashMap<FaceId, PlaneFrame>,
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;

    // Guarded by `all_faces_plane()` at the entry point of `boolean_v2()`.
    let Some((na, da)) = (match face_a.surface() {
        FaceSurface::Plane { normal, d } => Some((*normal, *d)),
        _ => None,
    }) else {
        debug_assert!(false, "non-plane face reached intersect_two_plane_faces");
        return Ok(Vec::new());
    };
    let Some((nb, db)) = (match face_b.surface() {
        FaceSurface::Plane { normal, d } => Some((*normal, *d)),
        _ => None,
    }) else {
        debug_assert!(false, "non-plane face reached intersect_two_plane_faces");
        return Ok(Vec::new());
    };

    // Plane-plane intersection: line direction = na × nb.
    let line_dir = na.cross(nb);
    let line_len = line_dir.length();
    if line_len < 1e-10 {
        // Parallel or coincident planes — no intersection curve.
        return Ok(Vec::new());
    }
    let line_dir_n = Vec3::new(
        line_dir.x() / line_len,
        line_dir.y() / line_len,
        line_dir.z() / line_len,
    );

    // Find a point on the intersection line.
    // Solve: na·p = da, nb·p = db, line_dir·p = 0
    let line_origin = solve_two_planes_origin(na, da, nb, db, line_dir_n)?;

    // Collect boundary polygons for both faces.
    let poly_a = collect_face_polygon(topo, fa)?;
    let poly_b = collect_face_polygon(topo, fb)?;

    // Use cached frames for consistent UV projections.
    let owned_frame_a;
    let frame_a = if let Some(f) = plane_frames.get(&fa) {
        f
    } else {
        owned_frame_a = plane_frame_for_polygon(na, &poly_a);
        &owned_frame_a
    };
    let owned_frame_b;
    let frame_b = if let Some(f) = plane_frames.get(&fb) {
        f
    } else {
        owned_frame_b = plane_frame_for_polygon(nb, &poly_b);
        &owned_frame_b
    };

    // Trim the infinite line to both face boundaries independently.
    // Both use the same line_origin so t-values are compatible.
    let segments_a = trim_line_to_polygon_3d(&line_origin, &line_dir_n, &poly_a, frame_a);
    let segments_b = trim_line_to_polygon_3d(&line_origin, &line_dir_n, &poly_b, frame_b);

    // Intersect the two interval sets to find the overlap.
    let mut result = Vec::new();
    for &(a0, a1) in &segments_a {
        for &(b0, b1) in &segments_b {
            let t0 = a0.max(b0);
            let t1 = a1.min(b1);
            if t1 - t0 < 1e-10 {
                continue; // No overlap or degenerate.
            }
            let start = line_origin + line_dir_n * t0;
            let end = line_origin + line_dir_n * t1;
            let pcurve_a = compute_line_pcurve(frame_a, start, end);
            let pcurve_b = compute_line_pcurve(frame_b, start, end);
            result.push(SectionEdge {
                curve_3d: EdgeCurve::Line,
                pcurve_a,
                pcurve_b,
                start,
                end,
            });
        }
    }
    Ok(result)
}

// ---------------------------------------------------------------------------
// Stage 3: Split faces
// ---------------------------------------------------------------------------

fn split_all_faces(
    topo: &Topology,
    a: SolidId,
    b: SolidId,
    pipeline: &mut BooleanPipeline,
    _tol: &Tolerance,
) -> Result<(), OperationsError> {
    let faces_a = collect_solid_faces(topo, a)?;
    let faces_b = collect_solid_faces(topo, b)?;

    let uv_tol = 1e-7;

    // Collect section edges per face.
    let mut sections_for_face: HashMap<FaceId, Vec<SectionEdge>> = HashMap::new();
    for ((fa, fb), sections) in &pipeline.intersections {
        for s in sections {
            sections_for_face.entry(*fa).or_default().push(s.clone());
            sections_for_face.entry(*fb).or_default().push(s.clone());
        }
    }

    // Split faces from solid A.
    for &fid in &faces_a {
        let sections = sections_for_face
            .get(&fid)
            .map_or(&[][..], |v| v.as_slice());
        let frame = pipeline.plane_frames.get(&fid);
        let sub = split_face_2d(topo, fid, sections, Source::A, uv_tol, frame);
        pipeline.sub_faces.extend(sub);
    }

    // Split faces from solid B.
    for &fid in &faces_b {
        let sections = sections_for_face
            .get(&fid)
            .map_or(&[][..], |v| v.as_slice());
        let frame = pipeline.plane_frames.get(&fid);
        let sub = split_face_2d(topo, fid, sections, Source::B, uv_tol, frame);
        pipeline.sub_faces.extend(sub);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Stage 4: Classify sub-faces
// ---------------------------------------------------------------------------

fn classify_sub_faces(
    topo: &Topology,
    pipeline: &mut BooleanPipeline,
    op: BooleanOp,
    classifier_a: Option<&AnalyticClassifier>,
    classifier_b: Option<&AnalyticClassifier>,
) -> Result<(), OperationsError> {
    let solid_a = pipeline
        .solid_a
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "no solid A".into(),
        })?;
    let solid_b = pipeline
        .solid_b
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "no solid B".into(),
        })?;

    // Collect polygons for both solids (for ray-cast fallback).
    let polys_a = collect_solid_face_polygons(topo, solid_a)?;
    let polys_b = collect_solid_face_polygons(topo, solid_b)?;
    let tol = Tolerance::new();

    // Classify each sub-face and mark for selection.
    let mut selected = Vec::new();
    for sub_face in &pipeline.sub_faces {
        // Get test point inside this sub-face.
        let frame = pipeline.plane_frames.get(&sub_face.parent);
        let test_pt = interior_point_3d(sub_face, frame);

        // Classify against opposing solid: try analytic first, fall back to ray-cast.
        // If the interior point is on the boundary (classifier returns None),
        // try wire vertices for a definitive answer before falling back to
        // the ray-cast (which can also be degenerate at corners/edges).
        let class = match sub_face.source {
            Source::A => {
                classify_with_fallback(test_pt, &sub_face.outer_wire, classifier_b, &polys_b, tol)
            }
            Source::B => {
                classify_with_fallback(test_pt, &sub_face.outer_wire, classifier_a, &polys_a, tol)
            }
        };

        // Use the boolean truth table to decide keep/discard/flip.
        if let Some(flip) = select_fragment(sub_face.source, class, op) {
            selected.push((sub_face.clone(), flip));
        }
    }

    // Apply flip flag: when flip=true, reverse the sub-face orientation.
    pipeline.sub_faces = selected
        .into_iter()
        .map(|(mut sf, flip)| {
            if flip {
                sf.reversed = !sf.reversed;
                // Reverse outer wire winding.
                sf.outer_wire.reverse();
                for edge in &mut sf.outer_wire {
                    std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                    std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                    edge.forward = !edge.forward;
                }
                // Same for inner wires.
                for inner in &mut sf.inner_wires {
                    inner.reverse();
                    for edge in inner.iter_mut() {
                        std::mem::swap(&mut edge.start_uv, &mut edge.end_uv);
                        std::mem::swap(&mut edge.start_3d, &mut edge.end_3d);
                        edge.forward = !edge.forward;
                    }
                }
            }
            sf
        })
        .collect();
    Ok(())
}

/// Classify a sub-face against the opposing solid with robust boundary handling.
///
/// 1. Try the analytic classifier on the interior point.
/// 2. If on-boundary (None), try each wire vertex until one gives a definitive answer.
/// 3. Fall back to ray-casting if no analytic result.
fn classify_with_fallback(
    test_pt: Point3,
    wire: &[super::pipeline::OrientedPCurveEdge],
    classifier: Option<&AnalyticClassifier>,
    face_polys: &[(Vec<Point3>, Vec3, f64)],
    tol: Tolerance,
) -> FaceClass {
    if let Some(c) = classifier {
        // Try the interior point first.
        if let Some(class) = c.classify(test_pt, tol) {
            return class;
        }
        // Interior point is on boundary — try wire vertices.
        for edge in wire {
            if let Some(class) = c.classify(edge.start_3d, tol) {
                return class;
            }
        }
    }
    // Fall back to ray-casting.
    classify_point_against_solid(test_pt, face_polys)
}

/// Classify a 3D point as inside/outside/on a solid using face polygon ray-casting.
fn classify_point_against_solid(
    point: Point3,
    face_polys: &[(Vec<Point3>, Vec3, f64)],
) -> FaceClass {
    // Cast a ray along +Z and count crossings using even-odd parity.
    // Odd crossing count = inside, even = outside.
    let ray_dir = Vec3::new(0.0, 0.0, 1.0);
    let mut crossings = 0u32;

    for (verts, normal, d) in face_polys {
        // Check if the ray from `point` in `ray_dir` crosses this face polygon.
        let denom = normal.dot(ray_dir);
        if denom.abs() < 1e-15 {
            continue; // Ray parallel to face.
        }
        let t = (*d - normal.dot(Vec3::new(point.x(), point.y(), point.z()))) / denom;
        if t < 1e-10 {
            continue; // Intersection behind ray origin.
        }
        let hit = point + ray_dir * t;
        // Check if hit point is inside the face polygon using 2D projection.
        // Project onto the face's dominant plane.
        if point_in_face_polygon_3d(hit, verts, normal) {
            crossings += 1;
        }
    }

    if crossings != 0 && crossings % 2 != 0 {
        FaceClass::Inside
    } else {
        FaceClass::Outside
    }
}

fn point_in_face_polygon_3d(point: Point3, verts: &[Point3], normal: &Vec3) -> bool {
    if verts.len() < 3 {
        return false;
    }
    // Project onto the dominant axis plane.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();
    let (proj_pt, proj_verts): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        // Drop Z.
        (
            Point2::new(point.x(), point.y()),
            verts.iter().map(|v| Point2::new(v.x(), v.y())).collect(),
        )
    } else if ay >= ax {
        // Drop Y.
        (
            Point2::new(point.x(), point.z()),
            verts.iter().map(|v| Point2::new(v.x(), v.z())).collect(),
        )
    } else {
        // Drop X.
        (
            Point2::new(point.y(), point.z()),
            verts.iter().map(|v| Point2::new(v.y(), v.z())).collect(),
        )
    };
    point_in_polygon_2d(proj_pt, &proj_verts)
}

// ---------------------------------------------------------------------------
// Stage 5: Assemble
// ---------------------------------------------------------------------------

fn assemble_v2(
    topo: &mut Topology,
    pipeline: &BooleanPipeline,
    tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    // Collect all 3D points for vertex merge resolution.
    let all_pts = pipeline.sub_faces.iter().flat_map(|sf| {
        sf.outer_wire
            .iter()
            .flat_map(|e| [e.start_3d, e.end_3d])
            .chain(
                sf.inner_wires
                    .iter()
                    .flat_map(|w| w.iter().flat_map(|e| [e.start_3d, e.end_3d])),
            )
    });
    let resolution = vertex_merge_resolution(all_pts, *tol);
    let mut vertex_map: HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId> =
        HashMap::new();

    let mut face_ids = Vec::new();

    for sub_face in &pipeline.sub_faces {
        // Create vertices + edges + wire for the outer boundary.
        let wire_id =
            create_wire_from_edges_dedup(topo, &sub_face.outer_wire, &mut vertex_map, resolution)?;
        let inner_wires: Vec<_> = sub_face
            .inner_wires
            .iter()
            .filter_map(|inner| {
                create_wire_from_edges_dedup(topo, inner, &mut vertex_map, resolution).ok()
            })
            .collect();

        let face = if sub_face.reversed {
            Face::new_reversed(wire_id, inner_wires, sub_face.surface.clone())
        } else {
            Face::new(wire_id, inner_wires, sub_face.surface.clone())
        };
        let fid = topo.add_face(face);
        face_ids.push(fid);
    }

    if face_ids.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "boolean_v2: no faces in result".into(),
        });
    }

    let shell = Shell::new(face_ids).map_err(OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    let solid = Solid::new(shell_id, Vec::new());
    Ok(topo.add_solid(solid))
}

fn create_wire_from_edges_dedup(
    topo: &mut Topology,
    edges: &[super::pipeline::OrientedPCurveEdge],
    vertex_map: &mut HashMap<(i64, i64, i64), brepkit_topology::vertex::VertexId>,
    resolution: f64,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    let mut oriented_edges = Vec::new();
    for pe in edges {
        // When pe.forward=false, the wire traverses the edge backward
        // (edge.end → edge.start). To make the backward traversal go
        // pe.start_3d → pe.end_3d, swap the vertex roles so
        // edge.end = pe.start_3d (traversal start) and
        // edge.start = pe.end_3d (traversal end).
        let (natural_start, natural_end) = if pe.forward {
            (pe.start_3d, pe.end_3d)
        } else {
            (pe.end_3d, pe.start_3d)
        };
        let key_start = quantize_point(natural_start, resolution);
        let v_start = *vertex_map
            .entry(key_start)
            .or_insert_with(|| topo.add_vertex(Vertex::new(natural_start, 0.0)));
        let key_end = quantize_point(natural_end, resolution);
        let v_end = *vertex_map
            .entry(key_end)
            .or_insert_with(|| topo.add_vertex(Vertex::new(natural_end, 0.0)));
        let edge = Edge::new(v_start, v_end, pe.curve_3d.clone());
        let eid = topo.add_edge(edge);
        oriented_edges.push(OrientedEdge::new(eid, pe.forward));
    }
    let wire = Wire::new(oriented_edges, true).map_err(OperationsError::Topology)?;
    Ok(topo.add_wire(wire))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn all_faces_plane(topo: &Topology, solid: SolidId) -> Result<bool, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    for fid in faces {
        if !matches!(topo.face(fid)?.surface(), FaceSurface::Plane { .. }) {
            return Ok(false);
        }
    }
    Ok(true)
}

fn collect_solid_faces(topo: &Topology, solid: SolidId) -> Result<Vec<FaceId>, OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    Ok(shell.faces().to_vec())
}

fn collect_face_polygon(topo: &Topology, face_id: FaceId) -> Result<Vec<Point3>, OperationsError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut pts = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        pts.push(topo.vertex(vid)?.point());
    }
    Ok(pts)
}

fn collect_solid_face_polygons(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(Vec<Point3>, Vec3, f64)>, OperationsError> {
    let faces = collect_solid_faces(topo, solid)?;
    let mut result = Vec::new();
    for fid in faces {
        let face = topo.face(fid)?;
        if let FaceSurface::Plane { normal, d } = face.surface() {
            let poly = collect_face_polygon(topo, fid)?;
            let effective_normal = if face.is_reversed() {
                -*normal
            } else {
                *normal
            };
            let effective_d = if face.is_reversed() { -*d } else { *d };
            result.push((poly, effective_normal, effective_d));
        }
    }
    Ok(result)
}

fn plane_frame_for_polygon(normal: Vec3, poly: &[Point3]) -> PlaneFrame {
    PlaneFrame::from_plane_face(normal, poly)
}

fn compute_line_pcurve(
    frame: &PlaneFrame,
    start: Point3,
    end: Point3,
) -> brepkit_math::curves2d::Curve2D {
    use brepkit_math::curves2d::Curve2D;
    let p0 = frame.project(start);
    let p1 = frame.project(end);
    let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
    // Build a Line2D from the projected direction. If degenerate
    // (zero-length), fall back to unit x-axis via make_line2d_safe which
    // centralizes the single #[allow(clippy::unwrap_used)].
    Curve2D::Line(super::pcurve_compute::make_line2d_safe(p0, dir))
}

// ---------------------------------------------------------------------------
// Intersection helpers
// ---------------------------------------------------------------------------

/// Find a point on the intersection line of two planes.
fn solve_two_planes_origin(
    na: Vec3,
    da: f64,
    nb: Vec3,
    db: f64,
    line_dir: Vec3,
) -> Result<Point3, OperationsError> {
    // Use the formula: p = (da*(nb×line) + db*(line×na)) / (line·(na×nb))
    let na_cross_nb = na.cross(nb);
    let denom = line_dir.dot(na_cross_nb);
    if denom.abs() < 1e-15 {
        return Err(OperationsError::InvalidInput {
            reason: "degenerate plane-plane intersection".into(),
        });
    }
    let nb_cross_l = nb.cross(line_dir);
    let l_cross_na = line_dir.cross(na);
    Ok(Point3::new(
        (da * nb_cross_l.x() + db * l_cross_na.x()) / denom,
        (da * nb_cross_l.y() + db * l_cross_na.y()) / denom,
        (da * nb_cross_l.z() + db * l_cross_na.z()) / denom,
    ))
}

/// Trim an infinite 3D line to a face's polygon boundary.
///
/// Projects the line and polygon into the face's 2D frame, computes
/// line-polygon intersection, and returns parameter pairs on the 3D line.
fn trim_line_to_polygon_3d(
    line_origin: &Point3,
    line_dir: &Vec3,
    polygon: &[Point3],
    frame: &PlaneFrame,
) -> Vec<(f64, f64)> {
    // Project the line into 2D.
    let origin_2d = frame.project(*line_origin);
    let dir_2d = Vec2::new(line_dir.dot(frame.u_axis()), line_dir.dot(frame.v_axis()));

    // Project polygon to 2D.
    let poly_2d: Vec<Point2> = polygon.iter().map(|p| frame.project(*p)).collect();

    trim_line_to_polygon_2d(origin_2d, dir_2d, &poly_2d)
}

/// Trim an infinite 2D line to a 2D polygon boundary.
///
/// Returns sorted parameter pairs (t_enter, t_exit) for segments inside.
fn trim_line_to_polygon_2d(origin: Point2, dir: Vec2, polygon: &[Point2]) -> Vec<(f64, f64)> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    let dir_len_sq = dir.x() * dir.x() + dir.y() * dir.y();
    if dir_len_sq < 1e-30 {
        return Vec::new();
    }

    // Compute intersection parameter for the line with each polygon edge.
    let mut hits: Vec<f64> = Vec::new();
    for i in 0..n {
        let j = (i + 1) % n;
        let pi = polygon[i];
        let pj = polygon[j];
        let edge_dir = Vec2::new(pj.x() - pi.x(), pj.y() - pi.y());
        let denom = dir.x() * edge_dir.y() - dir.y() * edge_dir.x();
        if denom.abs() < 1e-15 {
            continue; // Parallel.
        }
        let dp = Vec2::new(pi.x() - origin.x(), pi.y() - origin.y());
        let t = (dp.x() * edge_dir.y() - dp.y() * edge_dir.x()) / denom;
        let u = (dp.x() * dir.y() - dp.y() * dir.x()) / denom;
        // u must be in [0, 1] for the edge segment.
        if (-1e-10..=1.0 + 1e-10).contains(&u) {
            hits.push(t);
        }
    }

    if hits.is_empty() {
        return Vec::new();
    }

    // Sort and pair up as (enter, exit).
    hits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // Dedup close hits.
    hits.dedup_by(|a, b| (*a - *b).abs() < 1e-10);

    let mut segments = Vec::new();
    let mut i = 0;
    while i + 1 < hits.len() {
        // Check if midpoint of this pair is inside the polygon.
        let t_mid = (hits[i] + hits[i + 1]) * 0.5;
        let mid = Point2::new(origin.x() + dir.x() * t_mid, origin.y() + dir.y() * t_mid);
        if point_in_polygon_2d(mid, polygon) {
            segments.push((hits[i], hits[i + 1]));
        }
        i += 2;
    }
    segments
}

// ---------------------------------------------------------------------------
// Disjoint handling (with containment detection)
// ---------------------------------------------------------------------------

fn handle_disjoint_v2(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    b: SolidId,
    classifier_a: Option<&AnalyticClassifier>,
    classifier_b: Option<&AnalyticClassifier>,
    tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    // Check if B is inside A or A is inside B.
    let sample_b = sample_solid_vertex(topo, b)?;
    let sample_a = sample_solid_vertex(topo, a)?;

    let b_in_a = classifier_a
        .and_then(|c| c.classify(sample_b, *tol))
        .is_some_and(|c| c == FaceClass::Inside);
    let a_in_b = classifier_b
        .and_then(|c| c.classify(sample_a, *tol))
        .is_some_and(|c| c == FaceClass::Inside);

    if b_in_a {
        // B is entirely inside A.
        return match op {
            BooleanOp::Fuse => Ok(crate::copy::copy_solid(topo, a)?),
            BooleanOp::Cut => Err(OperationsError::InvalidInput {
                reason:
                    "boolean_v2: B is inside A — cut would create void (not supported in step 1)"
                        .into(),
            }),
            BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, b)?),
        };
    }

    if a_in_b {
        // A is entirely inside B.
        return match op {
            BooleanOp::Fuse => Ok(crate::copy::copy_solid(topo, b)?),
            BooleanOp::Cut => Err(OperationsError::InvalidInput {
                reason: "boolean_v2: A is inside B — cut result is empty".into(),
            }),
            BooleanOp::Intersect => Ok(crate::copy::copy_solid(topo, a)?),
        };
    }

    // Truly disjoint.
    match op {
        BooleanOp::Fuse => {
            // For disjoint fuse, return a copy of A.
            // TODO: Step 1 limitation — this discards solid B. A full
            // implementation should copy both solids into a compound or
            // merged shell so the result contains all geometry.
            Ok(crate::copy::copy_solid(topo, a)?)
        }
        BooleanOp::Cut => {
            // A - B where B doesn't touch A → result is A.
            Ok(crate::copy::copy_solid(topo, a)?)
        }
        BooleanOp::Intersect => {
            // A ∩ B where they don't touch → empty.
            Err(OperationsError::InvalidInput {
                reason: "boolean_v2: intersection of disjoint solids is empty".into(),
            })
        }
    }
}

/// Sample one vertex position from a solid (for containment testing).
fn sample_solid_vertex(topo: &Topology, solid: SolidId) -> Result<Point3, OperationsError> {
    let s = topo.solid(solid)?;
    let shell = topo.shell(s.outer_shell())?;
    let fid = *shell
        .faces()
        .first()
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "solid has no faces".into(),
        })?;
    let face = topo.face(fid)?;
    let wire = topo.wire(face.outer_wire())?;
    let oe = wire
        .edges()
        .first()
        .ok_or_else(|| OperationsError::InvalidInput {
            reason: "face has no edges".into(),
        })?;
    let edge = topo.edge(oe.edge())?;
    let v = topo.vertex(edge.start())?;
    Ok(v.point())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::primitives::make_box;
    use brepkit_math::mat::Mat4;

    /// Create two partially-overlapping boxes with NO shared faces.
    /// a: [0,10]³. b: [3,8]×[2,12]×[1,11] (5×10×10 box translated).
    fn make_overlapping_boxes(topo: &mut Topology) -> (SolidId, SolidId) {
        let a = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(topo, 5.0, 10.0, 10.0).unwrap();
        // Translate b by (3, 2, 1) so no faces are coplanar with a.
        let mat = Mat4::translation(3.0, 2.0, 1.0);
        crate::transform::transform_solid(topo, b, &mat).unwrap();
        (a, b)
    }

    #[test]
    fn trim_line_to_square_polygon_2d() {
        // Square [0,10]×[0,10] in 2D. Vertical line through x=5.
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        // Line: origin=(5, -5), direction=(0, 1).
        let origin = Point2::new(5.0, -5.0);
        let dir = Vec2::new(0.0, 1.0);
        let segments = trim_line_to_polygon_2d(origin, dir, &polygon);

        assert_eq!(segments.len(), 1, "expected 1 segment, got {segments:?}");
        let (t0, t1) = segments[0];
        // At t=5, we reach y=0 (bottom edge). At t=15, we reach y=10 (top edge).
        assert!((t0 - 5.0).abs() < 1e-6, "t0 = {t0}, expected ~5.0");
        assert!((t1 - 15.0).abs() < 1e-6, "t1 = {t1}, expected ~15.0");
    }

    #[test]
    fn trim_line_missing_polygon_2d() {
        let polygon = vec![
            Point2::new(0.0, 0.0),
            Point2::new(10.0, 0.0),
            Point2::new(10.0, 10.0),
            Point2::new(0.0, 10.0),
        ];
        // Line that misses the polygon entirely: x=15.
        let origin = Point2::new(15.0, -5.0);
        let dir = Vec2::new(0.0, 1.0);
        let segments = trim_line_to_polygon_2d(origin, dir, &polygon);
        assert!(
            segments.is_empty(),
            "expected no segments for miss, got {segments:?}"
        );
    }

    #[test]
    fn solve_two_planes_gives_point_on_both() {
        // Plane A: z=5 → normal=(0,0,1), d=5
        // Plane B: x=3 → normal=(1,0,0), d=3
        let na = Vec3::new(0.0, 0.0, 1.0);
        let nb = Vec3::new(1.0, 0.0, 0.0);
        let line_dir = na.cross(nb).normalize().unwrap();
        let origin = solve_two_planes_origin(na, 5.0, nb, 3.0, line_dir).unwrap();

        // Origin should satisfy both planes: z=5 and x=3.
        assert!(
            (origin.z() - 5.0).abs() < 1e-10,
            "origin.z = {}, expected 5.0",
            origin.z()
        );
        assert!(
            (origin.x() - 3.0).abs() < 1e-10,
            "origin.x = {}, expected 3.0",
            origin.x()
        );
    }

    #[test]
    fn trim_3d_line_to_box_face() {
        // Face A is z=0 plane of a 10×10×10 box: polygon at z=0.
        // Intersection line: x=5, z=0, y=variable.
        let polygon = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
        ];
        let normal = Vec3::new(0.0, 0.0, 1.0);
        let frame = PlaneFrame::from_plane_face(normal, &polygon);

        let line_origin = Point3::new(5.0, 0.0, 0.0);
        let line_dir = Vec3::new(0.0, 1.0, 0.0);

        let segments = trim_line_to_polygon_3d(&line_origin, &line_dir, &polygon, &frame);
        assert_eq!(segments.len(), 1, "expected 1 segment, got {segments:?}");

        let (t0, t1) = segments[0];
        // Should trim to y=[0,10].
        let start = line_origin + line_dir * t0;
        let end = line_origin + line_dir * t1;
        assert!(
            start.y().abs() < 0.1 && (end.y() - 10.0).abs() < 0.1,
            "segment ({:.2},{:.2},{:.2})→({:.2},{:.2},{:.2}), expected y=[0,10]",
            start.x(),
            start.y(),
            start.z(),
            end.x(),
            end.y(),
            end.z()
        );
    }

    #[test]
    fn intersect_two_actual_box_faces() {
        // Create a box and inspect its face geometry.
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let faces_a = collect_solid_faces(&topo, a).unwrap();
        let faces_b = collect_solid_faces(&topo, b).unwrap();

        // Try intersecting specific face pairs and check the result.
        let frames = HashMap::new();
        let mut total_sections = 0;
        for &fa in &faces_a {
            for &fb in &faces_b {
                let sections = intersect_two_plane_faces(&topo, fa, fb, &frames).unwrap();
                if !sections.is_empty() {
                    for s in &sections {
                        // Verify section edge is within bounds of both boxes.
                        assert!(
                            s.start.x() >= -0.1 && s.start.x() <= 10.1,
                            "section start x={:.2} out of bounds",
                            s.start.x()
                        );
                        assert!(
                            s.start.y() >= -0.1 && s.start.y() <= 12.1,
                            "section start y={:.2} out of bounds",
                            s.start.y()
                        );
                        assert!(
                            s.start.z() >= -0.1 && s.start.z() <= 11.1,
                            "section start z={:.2} out of bounds",
                            s.start.z()
                        );
                    }
                    total_sections += sections.len();
                }
            }
        }
        assert!(total_sections > 0, "no sections found");
    }

    #[test]
    fn stage1_intersection_produces_section_edges() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        intersect_plane_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        // Two boxes with 6 faces each = 36 face pairs.
        // Many are parallel (no intersection). Non-parallel pairs produce
        // trimmed section edges where the faces actually overlap.
        let total_sections: usize = pipeline.intersections.values().map(Vec::len).sum();
        assert!(
            total_sections > 0,
            "expected section edges, got 0. intersections map has {} entries",
            pipeline.intersections.len()
        );
    }

    #[test]
    fn stage1_sections_match_face_ids() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        intersect_plane_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        let faces_a = collect_solid_faces(&topo, a).unwrap();
        let faces_b = collect_solid_faces(&topo, b).unwrap();

        // Check which faces have sections.
        let mut sections_for_face: HashMap<FaceId, Vec<SectionEdge>> = HashMap::new();
        for ((fa, fb), sections) in &pipeline.intersections {
            for s in sections {
                sections_for_face.entry(*fa).or_default().push(s.clone());
                sections_for_face.entry(*fb).or_default().push(s.clone());
            }
        }
        let a_with_sections = faces_a
            .iter()
            .filter(|f| sections_for_face.contains_key(f))
            .count();
        let b_with_sections = faces_b
            .iter()
            .filter(|f| sections_for_face.contains_key(f))
            .count();
        eprintln!("A faces with sections: {a_with_sections}/{}", faces_a.len());
        eprintln!("B faces with sections: {b_with_sections}/{}", faces_b.len());
    }

    #[test]
    fn stage3_split_produces_sub_faces() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        let tol = Tolerance::new();
        let mut pipeline = BooleanPipeline {
            solid_a: Some(a),
            solid_b: Some(b),
            ..BooleanPipeline::default()
        };
        init_plane_frames(&topo, a, b, &mut pipeline).unwrap();
        intersect_plane_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        split_all_faces(&topo, a, b, &mut pipeline, &tol).unwrap();
        eprintln!(
            "Stage 3: {} sub-faces total (A: {}, B: {})",
            pipeline.sub_faces.len(),
            pipeline
                .sub_faces
                .iter()
                .filter(|sf| sf.source == Source::A)
                .count(),
            pipeline
                .sub_faces
                .iter()
                .filter(|sf| sf.source == Source::B)
                .count(),
        );
        // Both boxes have 6 faces. Some get split. We expect > 12 sub-faces.
        assert!(
            pipeline.sub_faces.len() >= 12,
            "expected >= 12 sub-faces, got {}",
            pipeline.sub_faces.len()
        );
    }

    #[test]
    fn boolean_v2_box_cut_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // a: [0,10]³ = 1000. b: [3,8]×[2,12]×[1,11].
        // Overlap: [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        // Cut = a - overlap = 1000 - 360 = 640.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 640.0).abs() < 70.0,
            "Cut volume {vol} should be ~640"
        );
    }

    #[test]
    fn boolean_v2_box_fuse_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Fuse = a + b - overlap = 1000 + 500 - 360 = 1140.
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1140.0).abs() < 120.0,
            "Fuse volume {vol} should be ~1140"
        );
    }

    #[test]
    fn boolean_v2_box_intersect_box() {
        let mut topo = Topology::new();
        let (a, b) = make_overlapping_boxes(&mut topo);
        // Intersect = overlap = [3,8]×[2,10]×[1,10] = 5×8×9 = 360.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 360.0).abs() < 40.0,
            "Intersect volume {vol} should be ~360"
        );
    }

    // --- Edge-case tests ---

    #[test]
    fn boolean_v2_3d_offset_overlap() {
        // Boxes offset on all 3 axes (no shared plane).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Overlap: [5,10]³ = 125.
        // Intersect is simpler to verify: overlap = 125.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 20.0,
            "3D-offset intersect volume {vol} should be ~125"
        );
    }

    #[test]
    fn boolean_v2_3d_offset_fuse() {
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let mat = Mat4::translation(5.0, 5.0, 5.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // a=[0,10]³, b=[5,15]³. Fuse = 1000 + 1000 - 125 = 1875.
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1875.0).abs() < 200.0,
            "3D-offset fuse volume {vol} should be ~1875"
        );
    }

    #[test]
    fn boolean_v2_b_inside_a_fuse() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // B is at [3,7]×[3,7]×[3,7], entirely inside A=[0,10]³.
        // Fuse = A (since B adds no volume).
        let result = boolean_v2(&mut topo, BooleanOp::Fuse, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 1000.0).abs() < 10.0,
            "B-inside-A fuse volume {vol} should be ~1000"
        );
    }

    #[test]
    fn boolean_v2_b_inside_a_intersect() {
        // Small box entirely inside large box.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Intersect = B.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "B-inside-A intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn boolean_v2_a_inside_b_intersect() {
        // Large box inside even larger box (A inside B).
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
        let mat_a = Mat4::translation(3.0, 3.0, 3.0);
        crate::transform::transform_solid(&mut topo, a, &mat_a).unwrap();
        let b = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        // A is at [3,7]³, B is at [0,10]³. A inside B.
        // Intersect = A.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 64.0).abs() < 10.0,
            "A-inside-B intersect volume {vol} should be ~64"
        );
    }

    #[test]
    fn boolean_v2_disjoint_cut() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Cut = A.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 125.0).abs() < 10.0,
            "Disjoint cut volume {vol} should be ~125"
        );
    }

    #[test]
    fn boolean_v2_disjoint_intersect_is_error() {
        // Two completely separate boxes.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let b = make_box(&mut topo, 5.0, 5.0, 5.0).unwrap();
        let mat = Mat4::translation(20.0, 0.0, 0.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // Disjoint: Intersect = error.
        let result = boolean_v2(&mut topo, BooleanOp::Intersect, a, b);
        assert!(result.is_err(), "disjoint intersect should return Err");
    }

    #[test]
    fn boolean_v2_asymmetric_cut() {
        // Non-symmetric overlap verifying correct face selection.
        let mut topo = Topology::new();
        let a = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let b = make_box(&mut topo, 6.0, 4.0, 3.0).unwrap();
        let mat = Mat4::translation(7.0, 3.0, 2.0);
        crate::transform::transform_solid(&mut topo, b, &mat).unwrap();
        // b: [7,13]×[3,7]×[2,5]. Overlap with a: [7,10]×[3,7]×[2,5] = 3×4×3 = 36.
        // Cut = 1000 - 36 = 964.
        let result = boolean_v2(&mut topo, BooleanOp::Cut, a, b).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            (vol - 964.0).abs() < 100.0,
            "Asymmetric cut volume {vol} should be ~964"
        );
    }
}

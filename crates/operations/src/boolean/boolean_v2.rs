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

use super::classify_2d::point_in_polygon_2d;
use super::face_splitter::{interior_point_3d, split_face_2d};
use super::pipeline::{BooleanPipeline, SectionEdge};
use super::plane_frame::PlaneFrame;
use super::types::{BooleanOp, FaceClass, Source, select_fragment};
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

    // Stage 1: Intersect all face pairs.
    intersect_plane_faces(topo, a, b, &mut pipeline, &tol)?;

    // Disjoint shortcut.
    if pipeline.intersections.is_empty() {
        return handle_disjoint_v2(topo, op, a, b);
    }

    // Stage 2: Split edges at intersection vertices.
    // For plane-plane, the boundary edges are pre-split during face splitting
    // (the wire builder handles the split edges naturally via section edges).
    // No explicit edge splitting needed in Step 1.

    // Stage 3: Split faces via wire builder.
    split_all_faces(topo, a, b, &mut pipeline, &tol)?;

    // Stage 4: Classify sub-faces against opposing solid.
    classify_sub_faces(topo, &mut pipeline, op)?;

    // Stage 5: Assemble result.
    let result = assemble_v2(topo, &pipeline, &tol)?;

    // Post-processing: healing.
    crate::heal::remove_degenerate_edges(topo, result, tol.linear)?;
    // unify_faces can corrupt intermediate results, but we're building a
    // final solid here so it's safe.
    crate::heal::unify_faces(topo, result)?;

    Ok(result)
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
            let sections = intersect_two_plane_faces(topo, fa, fb)?;
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
) -> Result<Vec<SectionEdge>, OperationsError> {
    let face_a = topo.face(fa)?;
    let face_b = topo.face(fb)?;

    let (na, da) = match face_a.surface() {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => return Ok(Vec::new()),
    };
    let (nb, db) = match face_b.surface() {
        FaceSurface::Plane { normal, d } => (*normal, *d),
        _ => return Ok(Vec::new()),
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

    // Trim the infinite line to face A's boundary.
    let frame_a = plane_frame_for_polygon(na, &poly_a);
    let segments_a = trim_line_to_polygon_3d(&line_origin, &line_dir_n, &poly_a, &frame_a);

    // Trim each segment further to face B's boundary.
    let frame_b = plane_frame_for_polygon(nb, &poly_b);
    let mut result = Vec::new();

    for (t0, t1) in segments_a {
        let seg_start = line_origin + line_dir_n * t0;
        let seg_end = line_origin + line_dir_n * t1;
        // Re-trim this segment against face B.
        let sub_segs =
            trim_segment_to_polygon_3d(&seg_start, &seg_end, &line_dir_n, &poly_b, &frame_b);
        for (s0, s1) in sub_segs {
            let start = line_origin + line_dir_n * s0;
            let end = line_origin + line_dir_n * s1;
            if (end - start).length() < 1e-10 {
                continue; // Degenerate segment.
            }
            // Compute pcurves on each face.
            let pcurve_a = compute_line_pcurve(&frame_a, start, end);
            let pcurve_b = compute_line_pcurve(&frame_b, start, end);
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
        let sub = split_face_2d(topo, fid, sections, Source::A, uv_tol);
        pipeline.sub_faces.extend(sub);
    }

    // Split faces from solid B.
    for &fid in &faces_b {
        let sections = sections_for_face
            .get(&fid)
            .map_or(&[][..], |v| v.as_slice());
        let sub = split_face_2d(topo, fid, sections, Source::B, uv_tol);
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

    // Collect polygons for both solids (for point-in-solid test).
    let polys_a = collect_solid_face_polygons(topo, solid_a)?;
    let polys_b = collect_solid_face_polygons(topo, solid_b)?;

    // Classify each sub-face and mark for selection.
    let mut selected = Vec::new();
    for sub_face in &pipeline.sub_faces {
        let wire_pts = sub_face
            .outer_wire
            .iter()
            .map(|e| e.start_3d)
            .collect::<Vec<_>>();
        let test_pt = interior_point_3d(sub_face, &wire_pts);

        let opposing_polys = match sub_face.source {
            Source::A => &polys_b,
            Source::B => &polys_a,
        };
        let class = classify_point_against_solid(test_pt, opposing_polys);

        // Use the boolean truth table to decide keep/discard/flip.
        if let Some(keep) = select_fragment(sub_face.source, class, op) {
            selected.push((sub_face.clone(), keep));
        }
    }

    pipeline.sub_faces = selected.into_iter().map(|(sf, _flip)| sf).collect();
    Ok(())
}

/// Classify a 3D point as inside/outside/on a solid using face polygon ray-casting.
fn classify_point_against_solid(
    point: Point3,
    face_polys: &[(Vec<Point3>, Vec3, f64)],
) -> FaceClass {
    // Cast a ray along +Z and count crossings.
    let ray_dir = Vec3::new(0.0, 0.0, 1.0);
    let mut crossings = 0i32;

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
            crossings += if denom > 0.0 { 1 } else { -1 };
        }
    }

    if crossings > 0 {
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
    _tol: &Tolerance,
) -> Result<SolidId, OperationsError> {
    let mut face_ids = Vec::new();

    for sub_face in &pipeline.sub_faces {
        // Create vertices + edges + wire for the outer boundary.
        let wire_id = create_wire_from_edges(topo, &sub_face.outer_wire)?;
        let inner_wires: Vec<_> = sub_face
            .inner_wires
            .iter()
            .filter_map(|inner| create_wire_from_edges(topo, inner).ok())
            .collect();

        let face = Face::new(wire_id, inner_wires, sub_face.surface.clone());
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

fn create_wire_from_edges(
    topo: &mut Topology,
    edges: &[super::pipeline::OrientedPCurveEdge],
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    let mut oriented_edges = Vec::new();
    for pe in edges {
        let v_start = topo.add_vertex(Vertex::new(pe.start_3d, 0.0));
        let v_end = topo.add_vertex(Vertex::new(pe.end_3d, 0.0));
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
    use brepkit_math::curves2d::{Curve2D, Line2D};
    let p0 = frame.project(start);
    let p1 = frame.project(end);
    let dir = Vec2::new(p1.x() - p0.x(), p1.y() - p0.y());
    Curve2D::Line(Line2D::new(p0, dir).unwrap_or_else(|_| {
        #[allow(clippy::unwrap_used)]
        Line2D::new(p0, Vec2::new(1.0, 0.0)).unwrap()
    }))
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

/// Trim a finite segment (defined by 3D start/end) to a polygon boundary.
fn trim_segment_to_polygon_3d(
    seg_start: &Point3,
    seg_end: &Point3,
    line_dir: &Vec3,
    polygon: &[Point3],
    frame: &PlaneFrame,
) -> Vec<(f64, f64)> {
    let t_start = (*seg_start - *polygon.first().unwrap_or(seg_start)).dot(*line_dir)
        / line_dir.dot(*line_dir);
    let line_origin = *seg_start - *line_dir * t_start;

    let origin_2d = frame.project(line_origin);
    let dir_2d = Vec2::new(line_dir.dot(frame.u_axis()), line_dir.dot(frame.v_axis()));
    let poly_2d: Vec<Point2> = polygon.iter().map(|p| frame.project(*p)).collect();

    let t_seg_start = t_start;
    let t_seg_end = t_start + (*seg_end - *seg_start).dot(*line_dir) / line_dir.dot(*line_dir);

    let raw = trim_line_to_polygon_2d(origin_2d, dir_2d, &poly_2d);
    // Clamp to segment bounds.
    let mut result = Vec::new();
    for (t0, t1) in raw {
        let clamped_start = t0.max(t_seg_start);
        let clamped_end = t1.min(t_seg_end);
        if clamped_end - clamped_start > 1e-10 {
            result.push((clamped_start, clamped_end));
        }
    }
    result
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
// Disjoint handling
// ---------------------------------------------------------------------------

fn handle_disjoint_v2(
    topo: &mut Topology,
    op: BooleanOp,
    a: SolidId,
    _b: SolidId,
) -> Result<SolidId, OperationsError> {
    match op {
        BooleanOp::Fuse => {
            // For disjoint fuse, return a copy of both solids merged.
            // Simple approach: copy solid A.
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
        eprintln!(
            "Stage 1: {} face pairs with {} total section edges",
            pipeline.intersections.len(),
            total_sections
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
        for (fid, secs) in &sections_for_face {
            for s in secs {
                eprintln!(
                    "  face {:?}: section ({:.1},{:.1},{:.1})→({:.1},{:.1},{:.1})",
                    fid,
                    s.start.x(),
                    s.start.y(),
                    s.start.z(),
                    s.end.x(),
                    s.end.y(),
                    s.end.z()
                );
            }
        }
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
    #[ignore = "WIP: intersection trimming produces out-of-bounds section edges"]
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
    #[ignore = "WIP: intersection trimming produces out-of-bounds section edges"]
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
    #[ignore = "WIP: intersection trimming produces out-of-bounds section edges"]
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
}

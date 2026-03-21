//! Wire loop construction from trimmed intersection edges.
//!
//! After earlier phases compute intersection curves between adjacent offset
//! faces and create preliminary edges, this phase trims those edges to their
//! mutual intersections and assembles them into closed wire loops for each
//! offset face.

use std::collections::{HashMap, HashSet};

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::data::{OffsetData, OffsetStatus, find_or_create_vertex};
use crate::error::OffsetError;

/// Build closed wire loops for each offset face from the trimmed
/// intersection curves and split edges.
///
/// For each non-excluded offset face, collects intersection edges that
/// touch the face, computes their pairwise intersections to find corner
/// vertices, creates trimmed edges between those corners, and assembles
/// them into closed wire loops.
///
/// # Errors
///
/// Returns [`OffsetError`] if a wire loop cannot be closed or topology
/// lookups fail.
pub fn build_wire_loops(topo: &mut Topology, data: &mut OffsetData) -> Result<(), OffsetError> {
    let active_faces: Vec<FaceId> = data
        .offset_faces
        .iter()
        .filter(|(_, of)| of.status == OffsetStatus::Done)
        .map(|(&fid, _)| fid)
        .collect();

    for face_id in active_faces {
        let wires = build_loops_for_face(topo, data, face_id)?;
        if !wires.is_empty() {
            data.face_wires.insert(face_id, wires);
        }
    }

    Ok(())
}

/// A line segment in 3D representing an intersection edge's geometry.
struct LineSeg {
    /// Start point of the intersection line.
    p0: Point3,
    /// End point of the intersection line.
    p1: Point3,
}

/// Build wire loops for a single face.
///
/// Tries three strategies in order:
/// 1. **Circle/seam pattern** — if the face has Circle edges (closed curves),
///    build wires using circle + seam topology (cylinder/cone lateral faces,
///    or single-circle cap faces).
/// 2. **Direct chain** — if intersection edges already share vertices and
///    form closed loops, chain them directly (sphere polygon faces).
/// 3. **Line intersection** — find corners via pairwise line-line intersection,
///    create trimmed edges, walk loops (box faces).
#[allow(clippy::too_many_lines)]
fn build_loops_for_face(
    topo: &mut Topology,
    data: &OffsetData,
    face_id: FaceId,
) -> Result<Vec<WireId>, OffsetError> {
    // Collect all intersection edges for this face.
    let mut face_edges: Vec<EdgeId> = Vec::new();
    for intersection in &data.intersections {
        if intersection.face_a != face_id && intersection.face_b != face_id {
            continue;
        }
        face_edges.extend_from_slice(&intersection.new_edges);
    }

    // Include boundary edges (thick-solid mode).
    if let Some(boundary) = data.boundary_edges.get(&face_id) {
        face_edges.extend_from_slice(boundary);
    }

    if face_edges.is_empty() {
        return Ok(Vec::new());
    }

    // Strategy 1: Circle/seam pattern.
    if let Some(wires) = try_circle_seam_wire(topo, &face_edges)? {
        return Ok(wires);
    }

    // Strategy 2: Direct chain (edges already share vertices).
    if let Some(wires) = try_direct_chain(topo, &face_edges)? {
        return Ok(wires);
    }

    // Strategy 3: Line intersection fallback (box-like faces).
    build_loops_via_line_intersection(topo, data, face_id)
}

/// Build wire from Circle edges and seam edges.
///
/// Handles two patterns:
/// - **Single closed circle**: one Circle edge (start == end) → wire = [circle].
/// - **Two circles + seam** (cylinder lateral): two Circle edges at different
///   positions → create a seam Line edge connecting their vertices, then
///   build wire = [circle_a, seam_fwd, circle_b_rev, seam_rev].
fn try_circle_seam_wire(
    topo: &mut Topology,
    edges: &[EdgeId],
) -> Result<Option<Vec<WireId>>, OffsetError> {
    // Separate circle edges (closed) from non-circle edges.
    let mut circles: Vec<EdgeId> = Vec::new();
    let mut others: Vec<EdgeId> = Vec::new();
    for &eid in edges {
        let edge = topo.edge(eid)?;
        if edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::Circle(_)) {
            circles.push(eid);
        } else {
            others.push(eid);
        }
    }

    if circles.is_empty() {
        return Ok(None);
    }

    // Single circle: cap face.
    if circles.len() == 1 && others.is_empty() {
        let wire = Wire::new(vec![OrientedEdge::new(circles[0], true)], true)?;
        return Ok(Some(vec![topo.add_wire(wire)]));
    }

    // Two circles: cylinder/cone lateral face.
    if circles.len() == 2 && others.is_empty() {
        let va = topo.edge(circles[0])?.start();
        let vb = topo.edge(circles[1])?.start();

        if va == vb {
            // Degenerate: same vertex — shouldn't happen, but handle gracefully.
            return Ok(None);
        }

        // Create seam edge connecting the two circle vertices.
        let seam = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));

        // Wire: circle_a(fwd) → seam(fwd) → circle_b(rev) → seam(rev)
        let wire = Wire::new(
            vec![
                OrientedEdge::new(circles[0], true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(circles[1], false),
                OrientedEdge::new(seam, false),
            ],
            true,
        )?;
        return Ok(Some(vec![topo.add_wire(wire)]));
    }

    // Mixed circle + non-circle: not handled by this strategy.
    Ok(None)
}

/// Try to chain edges into closed loops using vertex adjacency.
///
/// Works when edges already share vertices (e.g., projected polygon edges
/// for sphere faces). Returns `None` if edges can't form closed loops.
fn try_direct_chain(
    topo: &mut Topology,
    edges: &[EdgeId],
) -> Result<Option<Vec<WireId>>, OffsetError> {
    // Snapshot edge data.
    let edge_info: Vec<(EdgeId, VertexId, VertexId)> = edges
        .iter()
        .map(|&eid| {
            let edge = topo.edge(eid)?;
            Ok((eid, edge.start(), edge.end()))
        })
        .collect::<Result<Vec<_>, OffsetError>>()?;

    // Build vertex adjacency for non-closed edges.
    let mut adjacency: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (list_idx, &(_, start, end)) in edge_info.iter().enumerate() {
        if start == end {
            continue;
        }
        adjacency
            .entry(start.index())
            .or_default()
            .push((end.index(), list_idx));
        adjacency
            .entry(end.index())
            .or_default()
            .push((start.index(), list_idx));
    }

    // Every vertex must have exactly 2 incident edges for simple closed loops.
    for neighbors in adjacency.values() {
        if neighbors.len() != 2 {
            return Ok(None);
        }
    }

    if adjacency.is_empty() {
        return Ok(None);
    }

    // Walk adjacency to form closed loops.
    let mut visited: HashSet<usize> = HashSet::new();
    let mut all_loops: Vec<Vec<OrientedEdge>> = Vec::new();

    for &(_, start, end) in &edge_info {
        if start == end {
            continue;
        }
        let start_idx = edge_info
            .iter()
            .position(|&(_, s, e)| s == start && e == end && !visited.contains(&0))
            .unwrap_or(usize::MAX);
        if start_idx == usize::MAX || visited.contains(&start_idx) {
            continue;
        }

        let start_vertex = start.index();
        let mut current = start_vertex;
        let mut loop_edges: Vec<OrientedEdge> = Vec::new();

        loop {
            let neighbors = match adjacency.get(&current) {
                Some(n) => n,
                None => return Ok(None),
            };

            let next = neighbors.iter().find(|(_, idx)| !visited.contains(idx));
            let Some(&(next_vertex, list_idx)) = next else {
                return Ok(None);
            };

            visited.insert(list_idx);

            let (eid, si, _) = edge_info[list_idx];
            let is_forward = si.index() == current;
            loop_edges.push(OrientedEdge::new(eid, is_forward));

            current = next_vertex;
            if current == start_vertex {
                break;
            }
        }

        all_loops.push(loop_edges);
    }

    // All non-closed edges must be consumed.
    let non_closed = edge_info.iter().filter(|(_, s, e)| s != e).count();
    if visited.len() != non_closed {
        return Ok(None);
    }

    let mut wire_ids = Vec::new();
    for loop_edges in all_loops {
        let wire = Wire::new(loop_edges, true)?;
        wire_ids.push(topo.add_wire(wire));
    }

    if wire_ids.is_empty() {
        Ok(None)
    } else {
        Ok(Some(wire_ids))
    }
}

/// Build wire loops using the original line-intersection approach.
///
/// Collects intersection line segments, finds corners via pairwise
/// line-line intersection, creates trimmed edges, and walks loops.
#[allow(clippy::too_many_lines)]
fn build_loops_via_line_intersection(
    topo: &mut Topology,
    data: &OffsetData,
    face_id: FaceId,
) -> Result<Vec<WireId>, OffsetError> {
    let mut line_segs: Vec<LineSeg> = Vec::new();

    for intersection in &data.intersections {
        if intersection.face_a != face_id && intersection.face_b != face_id {
            continue;
        }
        for &eid in &intersection.new_edges {
            let edge = topo.edge(eid)?;
            let p0 = topo.vertex(edge.start())?.point();
            let p1 = topo.vertex(edge.end())?.point();
            line_segs.push(LineSeg { p0, p1 });
        }
    }

    if let Some(boundary) = data.boundary_edges.get(&face_id) {
        if let Some(offset_face) = data.offset_faces.get(&face_id) {
            for &eid in boundary {
                let edge = topo.edge(eid)?;
                let orig_p0 = topo.vertex(edge.start())?.point();
                let orig_p1 = topo.vertex(edge.end())?.point();
                let (p0, p1) = project_boundary_edge(orig_p0, orig_p1, &offset_face.surface);
                line_segs.push(LineSeg { p0, p1 });
            }
        }
    }

    if line_segs.is_empty() {
        return Ok(Vec::new());
    }

    let tol = data.options.tolerance.linear;
    let mut corner_cache: Vec<(Point3, VertexId)> = Vec::new();
    let mut corners_on_line: Vec<Vec<(VertexId, f64)>> = vec![Vec::new(); line_segs.len()];

    for i in 0..line_segs.len() {
        for j in (i + 1)..line_segs.len() {
            if let Some((pt, ti, tj)) = line_line_closest_point(&line_segs[i], &line_segs[j], tol) {
                let vid = find_or_create_vertex(topo, &mut corner_cache, pt, tol);
                corners_on_line[i].push((vid, ti));
                corners_on_line[j].push((vid, tj));
            }
        }
    }

    let mut trimmed_edges: Vec<EdgeId> = Vec::new();
    for corners in &mut corners_on_line {
        if corners.len() < 2 {
            continue;
        }
        corners.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for pair in corners.windows(2) {
            let v_start = pair[0].0;
            let v_end = pair[1].0;
            if v_start == v_end {
                continue;
            }
            let eid = topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::Line));
            trimmed_edges.push(eid);
        }
    }

    if trimmed_edges.is_empty() {
        return Ok(Vec::new());
    }

    let edge_info: Vec<(EdgeId, usize, usize)> = trimmed_edges
        .iter()
        .map(|&eid| {
            let edge = topo.edge(eid)?;
            Ok((eid, edge.start().index(), edge.end().index()))
        })
        .collect::<Result<Vec<_>, OffsetError>>()?;

    let mut adjacency: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (list_idx, &(_, si, ei)) in edge_info.iter().enumerate() {
        adjacency.entry(si).or_default().push((ei, list_idx));
        adjacency.entry(ei).or_default().push((si, list_idx));
    }

    let mut visited: HashSet<usize> = HashSet::new();
    let mut all_loops: Vec<Vec<OrientedEdge>> = Vec::new();

    for (start_idx, &(_, start_si, _)) in edge_info.iter().enumerate() {
        if visited.contains(&start_idx) {
            continue;
        }

        let start_vertex = start_si;
        let mut current = start_vertex;
        let mut loop_edges: Vec<OrientedEdge> = Vec::new();

        loop {
            let neighbors = adjacency
                .get(&current)
                .ok_or_else(|| OffsetError::AssemblyFailed {
                    reason: format!("wire loop walk: vertex index {current} not in adjacency"),
                })?;

            let next = neighbors.iter().find(|(_, idx)| !visited.contains(idx));
            let Some(&(next_vertex, list_idx)) = next else {
                return Err(OffsetError::AssemblyFailed {
                    reason: format!(
                        "wire loop walk: no unvisited edge from vertex {current} \
                         ({} visited, {} in loop)",
                        visited.len(),
                        loop_edges.len()
                    ),
                });
            };

            visited.insert(list_idx);

            let (eid, si, _ei) = edge_info[list_idx];
            let is_forward = si == current;
            loop_edges.push(OrientedEdge::new(eid, is_forward));

            current = next_vertex;
            if current == start_vertex {
                break;
            }
        }

        all_loops.push(loop_edges);
    }

    let mut wire_ids = Vec::with_capacity(all_loops.len());
    for loop_edges in all_loops {
        let wire = Wire::new(loop_edges, true)?;
        wire_ids.push(topo.add_wire(wire));
    }

    Ok(wire_ids)
}

/// Compute the closest-approach point of two infinite lines, each defined
/// by a `LineSeg`'s endpoints.
///
/// Returns `Some((point, t_a, t_b))` if the lines are not parallel and their
/// closest-approach distance is below a threshold. `t_a` and `t_b` are
/// parameters along each line (`0.0` = `p0`, `1.0` = `p1`).
fn line_line_closest_point(a: &LineSeg, b: &LineSeg, tol: f64) -> Option<(Point3, f64, f64)> {
    let da = pt_sub(a.p1, a.p0);
    let db = pt_sub(b.p1, b.p0);
    let w0 = pt_sub(a.p0, b.p0);

    let aa = dot3(da, da);
    let bb = dot3(db, db);
    let ab = dot3(da, db);
    let aw = dot3(da, w0);
    let bw = dot3(db, w0);

    let denom = aa * bb - ab * ab;

    // Parallel lines — no unique intersection.
    // Parallel lines — cross product denominator is near-zero.
    if denom.abs() < 1e-20 {
        return None;
    }

    let t = (ab * bw - bb * aw) / denom;
    let s = (aa * bw - ab * aw) / denom;

    // Points on each line at closest approach.
    let pa = Point3::new(
        a.p0.x() + t * da.0,
        a.p0.y() + t * da.1,
        a.p0.z() + t * da.2,
    );
    let pb = Point3::new(
        b.p0.x() + s * db.0,
        b.p0.y() + s * db.1,
        b.p0.z() + s * db.2,
    );

    // Check that the lines actually intersect (distance below threshold).
    let dx = pa.x() - pb.x();
    let dy = pa.y() - pb.y();
    let dz = pa.z() - pb.z();
    let dist_sq = dx * dx + dy * dy + dz * dz;

    if dist_sq > tol * tol {
        return None;
    }

    let mid = Point3::new(
        (pa.x() + pb.x()) * 0.5,
        (pa.y() + pb.y()) * 0.5,
        (pa.z() + pb.z()) * 0.5,
    );

    Some((mid, t, s))
}

/// Subtract two points, returning a direction tuple.
fn pt_sub(a: Point3, b: Point3) -> (f64, f64, f64) {
    (a.x() - b.x(), a.y() - b.y(), a.z() - b.z())
}

/// Dot product of two 3-tuples.
fn dot3(a: (f64, f64, f64), b: (f64, f64, f64)) -> f64 {
    a.0 * b.0 + a.1 * b.1 + a.2 * b.2
}

/// Project a boundary edge's endpoints onto an offset surface.
///
/// For planar surfaces, this projects the point onto the plane (translates
/// along the normal). For other surfaces, it returns the original points
/// (approximation — proper projection requires parametric solvers).
fn project_boundary_edge(
    p0: Point3,
    p1: Point3,
    surface: &brepkit_topology::face::FaceSurface,
) -> (Point3, Point3) {
    match surface {
        brepkit_topology::face::FaceSurface::Plane { normal, d } => {
            // Project each point onto the plane: p' = p + (d - n·p) * n
            let project = |p: Point3| -> Point3 {
                let n_dot_p = normal.x() * p.x() + normal.y() * p.y() + normal.z() * p.z();
                let dist = d - n_dot_p;
                Point3::new(
                    p.x() + dist * normal.x(),
                    p.y() + dist * normal.y(),
                    p.z() + dist * normal.z(),
                )
            };
            (project(p0), project(p1))
        }
        _ => {
            // Non-planar: return original positions as approximation.
            (p0, p1)
        }
    }
}

// Uses crate::data::find_or_create_vertex (shared helper).

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    use crate::data::{OffsetData, OffsetOptions};

    fn run_phases_1_to_7(topo: &mut Topology, solid: SolidId, distance: f64) -> OffsetData {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        crate::inter3d::intersect_faces_3d(topo, solid, &mut data).unwrap();
        crate::inter2d::intersect_pcurves_2d(topo, solid, &mut data).unwrap();
        build_wire_loops(topo, &mut data).unwrap();
        data
    }

    #[test]
    fn box_each_face_has_one_wire() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        assert_eq!(data.face_wires.len(), 6, "each face should have wire loops");
        for wires in data.face_wires.values() {
            assert_eq!(
                wires.len(),
                1,
                "each box face should have exactly 1 wire loop"
            );
        }
    }

    #[test]
    fn box_wires_have_4_edges() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        for (&face_id, wires) in &data.face_wires {
            for &wire_id in wires {
                let wire = topo.wire(wire_id).unwrap();
                assert_eq!(
                    wire.edges().len(),
                    4,
                    "box face {face_id:?} wire should have 4 edges, got {}",
                    wire.edges().len()
                );
            }
        }
    }

    #[test]
    fn box_wires_are_closed() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        for wires in data.face_wires.values() {
            for &wire_id in wires {
                let wire = topo.wire(wire_id).unwrap();
                assert!(wire.is_closed(), "wire should be closed");
            }
        }
    }

    #[test]
    fn box_wire_edges_chain_correctly() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        for wires in data.face_wires.values() {
            for &wire_id in wires {
                let wire = topo.wire(wire_id).unwrap();
                let edges = wire.edges();
                for i in 0..edges.len() {
                    let curr = &edges[i];
                    let next = &edges[(i + 1) % edges.len()];
                    let curr_edge = topo.edge(curr.edge()).unwrap();
                    let next_edge = topo.edge(next.edge()).unwrap();
                    let curr_end = curr.oriented_end(curr_edge);
                    let next_start = next.oriented_start(next_edge);
                    assert_eq!(curr_end, next_start, "wire edge chain broken at index {i}");
                }
            }
        }
    }

    #[test]
    fn cylinder_each_face_has_one_wire() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_cylinder(&mut topo, 2.0, 5.0).unwrap();
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        assert_eq!(
            data.face_wires.len(),
            3,
            "cylinder has 3 faces, each should get a wire loop"
        );
    }

    #[test]
    fn sphere_each_face_has_one_wire() {
        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_sphere(&mut topo, 3.0, 16).unwrap();
        let data = run_phases_1_to_7(&mut topo, solid, 0.5);
        assert_eq!(
            data.face_wires.len(),
            2,
            "sphere has 2 faces, each should get a wire loop"
        );
    }
}

//! Wire loop construction from trimmed intersection edges.
//!
//! After earlier phases compute intersection curves between adjacent offset
//! faces and create preliminary edges, this phase trims those edges to their
//! mutual intersections and assembles them into closed wire loops for each
//! offset face.

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::data::{OffsetData, OffsetStatus};
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
/// 1. Collect intersection edges for this face with their line geometry.
/// 2. Compute pairwise line-line intersections to find corner vertices.
/// 3. For each intersection line, sort its corners by parameter and create
///    trimmed edges between consecutive corners.
/// 4. Walk the vertex adjacency graph to form closed loops.
/// 5. Create `Wire` entities from the loops.
#[allow(clippy::too_many_lines)]
fn build_loops_for_face(
    topo: &mut Topology,
    data: &OffsetData,
    face_id: FaceId,
) -> Result<Vec<WireId>, OffsetError> {
    // Step 1: collect intersection line segments for this face.
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

    if line_segs.is_empty() {
        return Ok(Vec::new());
    }

    let tol = data.options.tolerance.linear;

    // Step 2: compute pairwise line-line intersections to find corner vertices.
    let mut corner_cache: Vec<(Point3, VertexId)> = Vec::new();
    // corners_on_line[i] = [(vertex_id, parameter_t), ...]
    let mut corners_on_line: Vec<Vec<(VertexId, f64)>> = vec![Vec::new(); line_segs.len()];

    for i in 0..line_segs.len() {
        for j in (i + 1)..line_segs.len() {
            if let Some((pt, ti, tj)) = line_line_closest_point(&line_segs[i], &line_segs[j]) {
                let vid = find_or_create_vertex(topo, &mut corner_cache, pt, tol);
                corners_on_line[i].push((vid, ti));
                corners_on_line[j].push((vid, tj));
            }
        }
    }

    // Step 3: create trimmed edges between consecutive corner vertices on each
    // intersection line.
    let mut trimmed_edges: Vec<EdgeId> = Vec::new();

    for corners in &mut corners_on_line {
        if corners.len() < 2 {
            continue;
        }

        // Sort by parameter along the line.
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

    // Step 4: build vertex adjacency graph and walk to form closed loops.
    // Snapshot all edge endpoints before mutating topology (adding wires).
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

    // Step 5: create Wire entities.
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
fn line_line_closest_point(a: &LineSeg, b: &LineSeg) -> Option<(Point3, f64, f64)> {
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

    if dist_sq > 1e-6 {
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

/// Find an existing vertex within tolerance, or create a new one.
fn find_or_create_vertex(
    topo: &mut Topology,
    cache: &mut Vec<(Point3, VertexId)>,
    point: Point3,
    tol: f64,
) -> VertexId {
    for &(cached_pt, vid) in cache.iter() {
        let dx = point.x() - cached_pt.x();
        let dy = point.y() - cached_pt.y();
        let dz = point.z() - cached_pt.z();
        if dx * dx + dy * dy + dz * dz <= tol * tol {
            return vid;
        }
    }
    let vid = topo.add_vertex(Vertex::new(point, Tolerance::default().linear));
    cache.push((point, vid));
    vid
}

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
}

//! Edge chamfering (cutting edges at an angle).
//!
//! Chamfer replaces each target edge with a flat bevel face. The algorithm
//! works by rebuilding face polygons with offset vertices and inserting new
//! quadrilateral chamfer faces, then assembling the result using the same
//! spatial-hash dedup pattern as [`crate::boolean`].

use std::collections::{HashMap, HashSet};

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

use crate::boolean::assemble_solid;
use crate::dot_normal_point;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Chamfer one or more edges of a solid.
///
/// Each target edge is replaced by a flat bevel face. The `distance`
/// parameter controls how far from each vertex the bevel is placed
/// along the adjacent edges.
///
/// # Errors
///
/// Returns an error if:
/// - `distance` is zero or negative
/// - any edge is not shared by exactly two faces in the solid
/// - any face is a NURBS surface (only planar faces are supported)
/// - the result cannot be assembled into a valid solid
#[allow(clippy::too_many_lines)]
pub fn chamfer(
    topo: &mut Topology,
    solid: SolidId,
    edges: &[EdgeId],
    distance: f64,
) -> Result<SolidId, crate::OperationsError> {
    // -- Validation --
    if distance <= 0.0 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("chamfer distance must be positive, got {distance}"),
        });
    }
    if edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no edges specified for chamfer".into(),
        });
    }

    let tol = Tolerance::new();

    // -- Phase 1: Collect face data --
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let shell_face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Build edge→faces map and collect per-face polygon data.
    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();
    let mut face_polygons: HashMap<usize, FacePolygon> = HashMap::new();

    for &face_id in &shell_face_ids {
        let face = topo.face(face_id)?;
        let (normal, _d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "chamfer on NURBS faces is not supported".into(),
                });
            }
        };

        let wire = topo.wire(face.outer_wire())?;
        let mut vertex_ids = Vec::with_capacity(wire.edges().len());
        let mut positions = Vec::with_capacity(wire.edges().len());
        let mut wire_edge_ids = Vec::with_capacity(wire.edges().len());

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            vertex_ids.push(vid);
            positions.push(topo.vertex(vid)?.point());
            wire_edge_ids.push(oe.edge());

            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(face_id);
        }

        face_polygons.insert(
            face_id.index(),
            FacePolygon {
                vertex_ids,
                positions,
                wire_edge_ids,
                normal,
            },
        );
    }

    // -- Phase 2: Validate target edges --
    let target_set: HashSet<usize> = edges.iter().map(|e| e.index()).collect();

    for &edge_id in edges {
        let faces = edge_to_faces.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!("edge {} is not part of the solid", edge_id.index()),
            }
        })?;
        if faces.len() != 2 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!(
                    "edge {} is shared by {} faces, expected exactly 2",
                    edge_id.index(),
                    faces.len()
                ),
            });
        }
    }

    // -- Phase 3: Build modified polygons + collect chamfer face data --

    // For each target edge, we collect the chamfer points from both faces.
    let mut chamfer_data: HashMap<usize, ChamferEdgeData> = HashMap::new();
    let mut result_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for &face_id in &shell_face_ids {
        let poly = &face_polygons[&face_id.index()];
        let n = poly.positions.len();
        let mut new_verts: Vec<Point3> = Vec::with_capacity(n + target_set.len());

        for i in 0..n {
            let prev_i = if i == 0 { n - 1 } else { i - 1 };
            let next_i = (i + 1) % n;

            // Edge before vertex i: wire_edge_ids[prev_i] connects V[prev_i]→V[i]
            // Edge after vertex i:  wire_edge_ids[i]      connects V[i]→V[next_i]
            let before_chamfered = target_set.contains(&poly.wire_edge_ids[prev_i].index());
            let after_chamfered = target_set.contains(&poly.wire_edge_ids[i].index());

            let pos = poly.positions[i];
            let prev_pos = poly.positions[prev_i];
            let next_pos = poly.positions[next_i];

            match (before_chamfered, after_chamfered) {
                (false, false) => {
                    // No chamfer at this vertex — keep as-is.
                    new_verts.push(pos);
                }
                (true, false) => {
                    // Only the edge before is chamfered. Offset toward V[next].
                    let dir = (next_pos - pos).normalize()?;
                    let c = pos + dir * distance;
                    new_verts.push(c);

                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (false, true) => {
                    // Only the edge after is chamfered. Offset toward V[prev].
                    let dir = (prev_pos - pos).normalize()?;
                    let c = pos + dir * distance;
                    new_verts.push(c);

                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c,
                    );
                }
                (true, true) => {
                    // Both adjacent edges are chamfered. Emit two points:
                    // 1st: from the "after" edge chamfer, offset toward V[prev]
                    let dir_prev = (prev_pos - pos).normalize()?;
                    let c_after = pos + dir_prev * distance;
                    new_verts.push(c_after);

                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_after,
                    );

                    // 2nd: from the "before" edge chamfer, offset toward V[next]
                    let dir_next = (next_pos - pos).normalize()?;
                    let c_before = pos + dir_next * distance;
                    new_verts.push(c_before);

                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        c_before,
                    );
                }
            }
        }

        // Recompute plane d from the (possibly shifted) polygon.
        // Normal stays the same since vertices only moved within the face plane.
        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        result_faces.push((new_verts, poly.normal, new_d));
    }

    // -- Phase 4: Build chamfer faces --
    for &edge_id in edges {
        let data = chamfer_data.get(&edge_id.index()).ok_or_else(|| {
            crate::OperationsError::InvalidInput {
                reason: format!(
                    "failed to compute chamfer data for edge {}",
                    edge_id.index()
                ),
            }
        })?;

        let edge = topo.edge(edge_id)?;
        let v_start = edge.start();
        let v_end = edge.end();

        let face_list = &edge_to_faces[&edge_id.index()];
        let f1 = face_list[0];
        let f2 = face_list[1];

        // Retrieve chamfer points: (face, vertex) → Point3
        let c1_start = data.get_point(f1, v_start)?;
        let c1_end = data.get_point(f1, v_end)?;
        let c2_start = data.get_point(f2, v_start)?;
        let c2_end = data.get_point(f2, v_end)?;

        // Build the chamfer quad. Check orientation against the average
        // of the two adjacent face normals to ensure outward winding.
        let n1 = face_polygons[&f1.index()].normal;
        let n2 = face_polygons[&f2.index()].normal;
        let avg_normal = n1 + n2;

        let edge_a = c2_start - c1_start;
        let edge_b = c1_end - c1_start;
        let raw_normal = edge_a.cross(edge_b);

        let (quad, normal) = if raw_normal.dot(avg_normal) >= 0.0 {
            (
                vec![c1_start, c2_start, c2_end, c1_end],
                raw_normal.normalize()?,
            )
        } else {
            // Reverse winding.
            let flipped = edge_b.cross(edge_a);
            (
                vec![c1_start, c1_end, c2_end, c2_start],
                flipped.normalize()?,
            )
        };

        let d = dot_normal_point(normal, quad[0]);
        result_faces.push((quad, normal, d));
    }

    // -- Phase 5: Assemble result solid --
    assemble_solid(topo, &result_faces, tol)
}

// ---------------------------------------------------------------------------
// Internal data structures
// ---------------------------------------------------------------------------

/// Per-face polygon data collected from the solid.
struct FacePolygon {
    vertex_ids: Vec<VertexId>,
    positions: Vec<Point3>,
    /// The `EdgeId` for each wire edge: `wire_edge_ids[i]` connects
    /// `vertex_ids[i]` to `vertex_ids[(i+1) % n]`.
    wire_edge_ids: Vec<EdgeId>,
    normal: Vec3,
}

/// Chamfer point data collected during polygon rebuilding.
///
/// Maps `(face_index, vertex_index)` → chamfer point position.
struct ChamferEdgeData {
    points: HashMap<(usize, usize), Point3>,
}

impl ChamferEdgeData {
    fn new() -> Self {
        Self {
            points: HashMap::new(),
        }
    }

    fn insert(&mut self, face_id: FaceId, vertex_id: VertexId, point: Point3) {
        self.points
            .insert((face_id.index(), vertex_id.index()), point);
    }

    fn get_point(
        &self,
        face_id: FaceId,
        vertex_id: VertexId,
    ) -> Result<Point3, crate::OperationsError> {
        self.points
            .get(&(face_id.index(), vertex_id.index()))
            .copied()
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!(
                    "missing chamfer point for face {} vertex {}",
                    face_id.index(),
                    vertex_id.index()
                ),
            })
    }
}

/// Record a chamfer point for a target edge at a specific face and vertex.
fn record_chamfer_point(
    data: &mut HashMap<usize, ChamferEdgeData>,
    edge_index: usize,
    vertex_id: VertexId,
    face_id: FaceId,
    point: Point3,
) {
    data.entry(edge_index)
        .or_insert_with(ChamferEdgeData::new)
        .insert(face_id, vertex_id, point);
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used)]
mod tests {
    use brepkit_topology::test_utils::make_unit_cube_manifold;
    use brepkit_topology::validation::validate_shell_manifold;

    use super::*;

    /// Helper: get all unique edge IDs from a solid's outer shell.
    fn solid_edge_ids(topo: &Topology, solid_id: SolidId) -> Vec<EdgeId> {
        let solid = topo.solid(solid_id).expect("test solid");
        let shell = topo.shell(solid.outer_shell()).expect("test shell");
        let mut seen = HashSet::new();
        let mut edges = Vec::new();
        for &fid in shell.faces() {
            let face = topo.face(fid).expect("test face");
            let wire = topo.wire(face.outer_wire()).expect("test wire");
            for oe in wire.edges() {
                if seen.insert(oe.edge().index()) {
                    edges.push(oe.edge());
                }
            }
        }
        edges
    }

    #[test]
    fn chamfer_single_edge() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let edges = solid_edge_ids(&topo, cube);
        let target = edges[0];

        let result = chamfer(&mut topo, cube, &[target], 0.2).expect("chamfer should succeed");

        // Original cube has 6 faces. Chamfering one edge modifies 2 faces
        // (they keep same vertex count, just shifted) and adds 1 chamfer face → 7.
        let result_solid = topo.solid(result).expect("result solid");
        let result_shell = topo.shell(result_solid.outer_shell()).expect("shell");
        assert_eq!(
            result_shell.faces().len(),
            7,
            "expected 7 faces after single-edge chamfer"
        );
    }

    #[test]
    fn chamfer_zero_distance_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = chamfer(&mut topo, cube, &[edges[0]], 0.0);
        assert!(result.is_err(), "zero distance should fail");
    }

    #[test]
    fn chamfer_negative_distance_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = chamfer(&mut topo, cube, &[edges[0]], -0.5);
        assert!(result.is_err(), "negative distance should fail");
    }

    #[test]
    fn chamfer_invalid_edge_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Create a stray edge not part of the cube.
        let v0 = topo.vertices.alloc(brepkit_topology::vertex::Vertex::new(
            Point3::new(99.0, 99.0, 99.0),
            1e-7,
        ));
        let v1 = topo.vertices.alloc(brepkit_topology::vertex::Vertex::new(
            Point3::new(100.0, 100.0, 100.0),
            1e-7,
        ));
        let stray = topo.edges.alloc(brepkit_topology::edge::Edge::new(
            v0,
            v1,
            brepkit_topology::edge::EdgeCurve::Line,
        ));

        let result = chamfer(&mut topo, cube, &[stray], 0.2);
        assert!(result.is_err(), "invalid edge should fail");
    }

    #[test]
    fn chamfer_result_is_manifold() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = chamfer(&mut topo, cube, &[edges[0]], 0.2).expect("chamfer should succeed");

        let result_solid = topo.solid(result).expect("result solid");
        let result_shell = topo.shell(result_solid.outer_shell()).expect("shell");
        validate_shell_manifold(result_shell, &topo.faces, &topo.wires)
            .expect("result should be manifold");
    }

    #[test]
    fn chamfer_parallel_edges() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        // Find two edges that don't share a vertex (parallel on a cube).
        let mut pair = None;
        'outer: for (i, &ea) in edges.iter().enumerate() {
            let data_a = topo.edge(ea).expect("edge");
            let va = [data_a.start().index(), data_a.end().index()];
            for &eb in &edges[i + 1..] {
                let data_b = topo.edge(eb).expect("edge");
                let vb = [data_b.start().index(), data_b.end().index()];
                if !va.iter().any(|v| vb.contains(v)) {
                    pair = Some([ea, eb]);
                    break 'outer;
                }
            }
        }
        let targets = pair.expect("should find non-adjacent edges on a cube");

        let result =
            chamfer(&mut topo, cube, &targets, 0.2).expect("parallel chamfer should succeed");

        // 2 chamfered edges → 2 new chamfer faces.
        // Non-adjacent edges on a cube: 6 original + 2 chamfer = 8 faces.
        let result_solid = topo.solid(result).expect("result solid");
        let result_shell = topo.shell(result_solid.outer_shell()).expect("shell");
        assert_eq!(
            result_shell.faces().len(),
            8,
            "expected 8 faces after 2 non-adjacent chamfers"
        );

        validate_shell_manifold(result_shell, &topo.faces, &topo.wires)
            .expect("result should be manifold");
    }
}

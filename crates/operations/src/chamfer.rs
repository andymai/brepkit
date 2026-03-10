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

use crate::boolean::{FaceSpec, assemble_solid_mixed};
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

        // Include inner wire edges in adjacency map so hole-boundary
        // edges are correctly counted as shared by 2 faces.
        for &inner_wire_id in face.inner_wires() {
            let inner_wire = topo.wire(inner_wire_id)?;
            for oe in inner_wire.edges() {
                edge_to_faces
                    .entry(oe.edge().index())
                    .or_default()
                    .push(face_id);
            }
        }

        // Only build polygon data for planar faces. Non-planar faces
        // will be passed through unchanged if they don't contain target edges.
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue, // Skip non-planar faces for polygon data
        };

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

    // -- Phase 2: Filter to manifold edges, validate --
    // Like fillet, silently skip non-manifold edges (shared by != 2 faces)
    // which commonly occur in boolean operation output.
    let filtered_edges: Vec<EdgeId> = edges
        .iter()
        .copied()
        .filter(|edge_id| {
            edge_to_faces
                .get(&edge_id.index())
                .is_some_and(|faces| faces.len() == 2)
        })
        .collect();

    if filtered_edges.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "no manifold edges to chamfer (all edges are boundary or missing)".into(),
        });
    }

    let target_set: HashSet<usize> = filtered_edges.iter().map(|e| e.index()).collect();

    // Vertices at endpoints of chamfered edges (used to detect side-face corners).
    let mut vertex_chamfer_endpoints: HashSet<usize> = HashSet::new();
    for &edge_id in &filtered_edges {
        let edge = topo.edge(edge_id)?;
        vertex_chamfer_endpoints.insert(edge.start().index());
        vertex_chamfer_endpoints.insert(edge.end().index());
    }

    // -- Phase 3: Build modified polygons + collect chamfer face data --

    // For each target edge, we collect the chamfer points from both faces.
    let mut chamfer_data: HashMap<usize, ChamferEdgeData> = HashMap::new();
    let mut result_specs: Vec<FaceSpec> = Vec::new();

    // Track corner vertices where all adjacent edges are chamfered.
    // Maps vertex_id → (original_position, Vec<(face_id, intersection_point)>).
    let mut corner_data: HashMap<usize, (Point3, Vec<(FaceId, Point3)>)> = HashMap::new();

    // Count how many faces reference each vertex (to detect full-corner chamfer).
    let mut vertex_face_count: HashMap<usize, usize> = HashMap::new();
    for poly in face_polygons.values() {
        for vid in &poly.vertex_ids {
            *vertex_face_count.entry(vid.index()).or_default() += 1;
        }
    }

    for &face_id in &shell_face_ids {
        // Non-planar faces pass through unchanged.
        let Some(poly) = face_polygons.get(&face_id.index()) else {
            let face = topo.face(face_id)?;
            let verts = crate::boolean::face_polygon(topo, face_id)?;
            result_specs.push(FaceSpec::Surface {
                vertices: verts,
                surface: face.surface().clone(),
            });
            continue;
        };
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

            let at_chamfer_endpoint =
                vertex_chamfer_endpoints.contains(&poly.vertex_ids[i].index());

            match (before_chamfered, after_chamfered, at_chamfer_endpoint) {
                (false, false, false) => {
                    // No chamfer at this vertex — keep as-is.
                    new_verts.push(pos);
                }
                (false, false, true) => {
                    // Side face corner: vertex is at a chamfered edge endpoint
                    // but neither adjacent edge of THIS face is chamfered.
                    // Split into two offset points along the face's own edges.
                    let dir_prev = (prev_pos - pos).normalize()?;
                    new_verts.push(pos + dir_prev * distance);

                    let dir_next = (next_pos - pos).normalize()?;
                    new_verts.push(pos + dir_next * distance);
                }
                (true, false, _) => {
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
                (false, true, _) => {
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
                (true, true, _) => {
                    // Both adjacent edges are chamfered. Compute a single
                    // intersection point where the two trim planes meet on
                    // this face, rather than two separate offset points.
                    let d1 = (next_pos - pos).normalize()?;
                    let d2_dir = (pos - prev_pos).normalize()?;

                    // Inward perpendiculars within the face plane.
                    // For CCW winding (matching outward normal), inward = n × d.
                    let p1 = poly.normal.cross(d1);
                    let p2 = poly.normal.cross(d2_dir);

                    let cos_angle = p1.dot(p2);
                    let denom = 1.0 + cos_angle;

                    let intersection = if denom.abs() < 1e-12 {
                        // Nearly antiparallel — fall back to midpoint.
                        let mid = Point3::new(
                            (prev_pos.x() + next_pos.x()) * 0.5,
                            (prev_pos.y() + next_pos.y()) * 0.5,
                            (prev_pos.z() + next_pos.z()) * 0.5,
                        );
                        let dir = (mid - pos).normalize()?;
                        pos + dir * distance
                    } else {
                        let scale = distance / denom;
                        pos + (p1 + p2) * scale
                    };

                    new_verts.push(intersection);

                    // Record this point for both adjacent chamfered edges.
                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        intersection,
                    );
                    record_chamfer_point(
                        &mut chamfer_data,
                        poly.wire_edge_ids[prev_i].index(),
                        poly.vertex_ids[i],
                        face_id,
                        intersection,
                    );

                    // Track for corner triangle generation.
                    corner_data
                        .entry(poly.vertex_ids[i].index())
                        .or_insert_with(|| (pos, Vec::new()))
                        .1
                        .push((face_id, intersection));
                }
            }
        }

        // Recompute plane d from the (possibly shifted) polygon.
        // Normal stays the same since vertices only moved within the face plane.
        let new_d = dot_normal_point(poly.normal, new_verts[0]);
        result_specs.push(FaceSpec::Planar {
            vertices: new_verts,
            normal: poly.normal,
            d: new_d,
        });
    }

    // -- Phase 4: Build chamfer faces --
    for &edge_id in &filtered_edges {
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
        result_specs.push(FaceSpec::Planar {
            vertices: quad,
            normal,
            d,
        });
    }

    // -- Phase 4.5: Corner faces --
    // At each original vertex where ALL adjacent edges are chamfered,
    // the trim-plane intersections from each face create a polygonal gap
    // (triangle for box vertices, k-gon for degree-k vertices).
    // Compute an approximate solid center for outward normal orientation.
    let solid_center = {
        let mut cx = 0.0;
        let mut cy = 0.0;
        let mut cz = 0.0;
        let mut count = 0.0;
        for poly in face_polygons.values() {
            for p in &poly.positions {
                cx += p.x();
                cy += p.y();
                cz += p.z();
                count += 1.0;
            }
        }
        Point3::new(cx / count, cy / count, cz / count)
    };

    for (vid, (orig_pos, entries)) in corner_data {
        // Only create a corner face if ALL faces at this vertex contributed
        // (i.e. all edges at this vertex are chamfered).
        let expected = vertex_face_count.get(&vid).copied().unwrap_or(0);
        if entries.len() != expected || entries.len() < 3 {
            continue;
        }

        // Order the corner vertices for consistent winding.
        // Use the original vertex position to compute outward direction.
        let outward = (orig_pos - solid_center)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

        // For a triangle (3 entries), compute the normal and ensure it
        // points outward (away from solid center).
        let pts: Vec<Point3> = entries.iter().map(|(_, p)| *p).collect();
        let e1 = pts[1] - pts[0];
        let e2 = pts[2] - pts[0];
        let tri_normal = e1.cross(e2);

        let mut corner_verts: Vec<Point3> = if tri_normal.dot(outward) >= 0.0 {
            pts
        } else {
            let mut rev = pts;
            rev.reverse();
            rev
        };

        // For k > 3 corner faces, sort by angle around the outward axis.
        if corner_verts.len() > 3 {
            let center = Point3::new(
                corner_verts.iter().map(|p| p.x()).sum::<f64>() / corner_verts.len() as f64,
                corner_verts.iter().map(|p| p.y()).sum::<f64>() / corner_verts.len() as f64,
                corner_verts.iter().map(|p| p.z()).sum::<f64>() / corner_verts.len() as f64,
            );
            // Pick a reference direction in the corner face plane.
            let ref_dir = (corner_verts[0] - center)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let binormal = outward.cross(ref_dir);
            corner_verts.sort_by(|a, b| {
                let da = *a - center;
                let db = *b - center;
                let angle_a = da.dot(binormal).atan2(da.dot(ref_dir));
                let angle_b = db.dot(binormal).atan2(db.dot(ref_dir));
                angle_a
                    .partial_cmp(&angle_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Re-check winding after sort.
            let se1 = corner_verts[1] - corner_verts[0];
            let se2 = corner_verts[2] - corner_verts[0];
            if se1.cross(se2).dot(outward) < 0.0 {
                corner_verts.reverse();
            }
        }

        let cn = {
            let ce1 = corner_verts[1] - corner_verts[0];
            let ce2 = corner_verts[2] - corner_verts[0];
            ce1.cross(ce2)
                .normalize()
                .unwrap_or(Vec3::new(0.0, 0.0, 1.0))
        };
        let cd = dot_normal_point(cn, corner_verts[0]);
        result_specs.push(FaceSpec::Planar {
            vertices: corner_verts,
            normal: cn,
            d: cd,
        });
    }

    // -- Phase 5: Assemble result solid --
    assemble_solid_mixed(topo, &result_specs, tol)
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

    /// Chamfer all 12 edges of a 10³ box with d=1.0.
    ///
    /// Volume derivation for 10³ box chamfered at d=1:
    ///   Each edge removes a right-triangular prism with legs d, length L-2d=8:
    ///     12 × (d²/2) × (L-2d) = 12 × 0.5 × 8 = 48
    ///   Each corner removes a tetrahedron with volume d³/6:
    ///     8 × (1/6) ≈ 1.333
    ///   Total removed ≈ 49.333, expected ≈ 950.7
    ///
    /// Use 5% tolerance to account for implementation variations in corner
    /// treatment (the actual value depends on whether edge prisms use full
    /// edge length L=10 or trimmed L-2d=8).
    #[test]
    fn chamfer_all_edges_volume() {
        let mut topo = Topology::new();
        let cube = crate::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        let edges = solid_edge_ids(&topo, cube);

        assert_eq!(edges.len(), 12, "box should have 12 edges");
        let result = chamfer(&mut topo, cube, &edges, 1.0).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 6 trimmed faces + 12 chamfer strips + 8 corner triangles = 26 faces.
        assert_eq!(sh.faces().len(), 26, "chamfered box should have 26 faces");

        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        // Expected ≈ 950.7 (see doc comment). 5% tolerance window
        // covers the range [903, 998], catching gross errors while
        // allowing for implementation variations in corner treatment.
        let expected = 950.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "chamfered 10³ box with d=1 should have volume ~{expected}, got {vol} \
             (rel_err={rel_err:.2e}). Was previously 800-1000 tolerance."
        );
    }

    /// Single-edge chamfer on a unit cube: d=0.2.
    ///
    /// Removes a right-triangular prism: legs = 0.2, length = 1.0.
    /// V_removed = (0.2²/2) × 1.0 = 0.02.
    /// V_expected = 1.0 - 0.02 = 0.98.
    #[test]
    fn chamfer_single_edge_volume() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let edges = solid_edge_ids(&topo, cube);

        let result = chamfer(&mut topo, cube, &[edges[0]], 0.2).unwrap();

        let vol = crate::measure::solid_volume(&topo, result, 0.01).unwrap();
        // V = 1.0 - (0.2²/2 × 1.0) = 1.0 - 0.02 = 0.98
        let expected = 0.98;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-4,
            "single-edge chamfer d=0.2 on unit cube: expected {expected}, got {vol} \
             (rel_err={rel_err:.2e})"
        );
    }
}

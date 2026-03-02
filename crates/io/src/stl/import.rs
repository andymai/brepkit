//! STL import: convert triangle meshes into B-Rep topology.
//!
//! Takes a [`TriangleMesh`] (from [`read_stl`](super::reader::read_stl))
//! and builds topology entities: one planar face per triangle, assembled
//! into a shell and solid.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::IoError;

/// Import a [`TriangleMesh`] into topology as a single solid.
///
/// Each triangle becomes a planar face. Vertices at the same position
/// (within `tolerance`) are merged. The resulting faces are assembled
/// into a closed shell and solid.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - The mesh has no triangles
/// - Wire or shell construction fails
pub fn import_mesh(
    topo: &mut Topology,
    mesh: &TriangleMesh,
    tolerance: f64,
) -> Result<SolidId, IoError> {
    if mesh.indices.len() < 3 {
        return Err(IoError::InvalidTopology {
            reason: "mesh has no triangles".to_string(),
        });
    }

    // Build a vertex map: merge coincident positions.
    let vertex_ids = build_vertex_map(topo, &mesh.positions, tolerance);

    // Build one face per triangle.
    let mut face_ids = Vec::new();
    for tri in mesh.indices.chunks_exact(3) {
        let i0 = tri[0] as usize;
        let i1 = tri[1] as usize;
        let i2 = tri[2] as usize;

        let v0 = vertex_ids[i0];
        let v1 = vertex_ids[i1];
        let v2 = vertex_ids[i2];

        // Skip degenerate triangles (two or more coincident vertices).
        if v0 == v1 || v1 == v2 || v0 == v2 {
            continue;
        }

        let face_id = build_triangle_face(topo, v0, v1, v2)?;
        face_ids.push(face_id);
    }

    if face_ids.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no valid triangles in mesh".to_string(),
        });
    }

    let shell = Shell::new(face_ids).map_err(|e| IoError::ParseError {
        reason: format!("failed to build shell from mesh: {e}"),
    })?;
    let shell_id = topo.shells.alloc(shell);
    let solid_id = topo.solids.alloc(Solid::new(shell_id, Vec::new()));

    Ok(solid_id)
}

/// Build vertex IDs, merging coincident positions.
fn build_vertex_map(
    topo: &mut Topology,
    positions: &[Point3],
    tolerance: f64,
) -> Vec<brepkit_topology::vertex::VertexId> {
    let tol_sq = tolerance * tolerance;
    let mut unique_verts: Vec<(Point3, brepkit_topology::vertex::VertexId)> = Vec::new();
    let mut map = Vec::with_capacity(positions.len());

    for &pos in positions {
        // Check if a nearby vertex already exists.
        let existing = unique_verts.iter().find(|(p, _)| {
            let dx = p.x() - pos.x();
            let dy = p.y() - pos.y();
            let dz = p.z() - pos.z();
            dx.mul_add(dx, dy.mul_add(dy, dz * dz)) < tol_sq
        });

        if let Some(&(_, vid)) = existing {
            map.push(vid);
        } else {
            let vid = topo.vertices.alloc(Vertex::new(pos, tolerance));
            unique_verts.push((pos, vid));
            map.push(vid);
        }
    }

    map
}

/// Build a single triangular planar face from three vertex IDs.
fn build_triangle_face(
    topo: &mut Topology,
    v0: brepkit_topology::vertex::VertexId,
    v1: brepkit_topology::vertex::VertexId,
    v2: brepkit_topology::vertex::VertexId,
) -> Result<brepkit_topology::face::FaceId, IoError> {
    // Create edges.
    let e01 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
    let e12 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
    let e20 = topo.edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

    // Build wire.
    let oriented = vec![
        OrientedEdge::new(e01, true),
        OrientedEdge::new(e12, true),
        OrientedEdge::new(e20, true),
    ];
    let wire = Wire::new(oriented, true).map_err(|e| IoError::ParseError {
        reason: format!("failed to build triangle wire: {e}"),
    })?;
    let wire_id = topo.wires.alloc(wire);

    // Compute face normal.
    let p0 = topo.vertex(v0).map_err(topo_err)?.point();
    let p1 = topo.vertex(v1).map_err(topo_err)?.point();
    let p2 = topo.vertex(v2).map_err(topo_err)?.point();

    let edge1 = p1 - p0;
    let edge2 = p2 - p0;
    let normal = edge1
        .cross(edge2)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let d = normal.dot(Vec3::new(p0.x(), p0.y(), p0.z()));

    let surface = FaceSurface::Plane { normal, d };
    let face_id = topo.faces.alloc(Face::new(wire_id, Vec::new(), surface));

    Ok(face_id)
}

/// Convert a [`TopologyError`] into an [`IoError`].
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube;

    use super::*;
    use crate::stl::reader::read_stl;
    use crate::stl::writer::{self, StlFormat};

    #[test]
    fn import_single_triangle() {
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            indices: vec![0, 1, 2],
        };

        let mut topo = Topology::new();
        let solid_id = import_mesh(&mut topo, &mesh, 1e-7).unwrap();

        let solid = topo.solid(solid_id).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();
        assert_eq!(shell.faces().len(), 1);
    }

    #[test]
    fn import_two_triangles() {
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let solid_id = import_mesh(&mut topo, &mesh, 1e-7).unwrap();

        let solid = topo.solid(solid_id).unwrap();
        let shell = topo.shell(solid.outer_shell()).unwrap();
        assert_eq!(shell.faces().len(), 2);
    }

    #[test]
    fn import_stl_roundtrip_unit_cube() {
        // Write a unit cube to STL, read it back, import to topology.
        let mut write_topo = Topology::new();
        let solid = make_unit_cube(&mut write_topo);

        let stl_bytes = writer::write_stl(&write_topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&stl_bytes).unwrap();

        let mut read_topo = Topology::new();
        let imported = import_mesh(&mut read_topo, &mesh, 1e-4).unwrap();

        let read_solid = read_topo.solid(imported).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        // Unit cube: 12 triangles.
        assert_eq!(shell.faces().len(), 12);
    }

    #[test]
    fn vertex_merging() {
        // Two triangles sharing an edge — should merge 2 vertices.
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0), // Same as [1]
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0), // Same as [2]
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let _solid = import_mesh(&mut topo, &mesh, 1e-6).unwrap();

        // Should have 4 unique vertices, not 6.
        assert_eq!(topo.vertices.len(), 4);
    }

    #[test]
    fn empty_mesh_error() {
        let mesh = TriangleMesh::default();
        let mut topo = Topology::new();
        let result = import_mesh(&mut topo, &mesh, 1e-7);
        assert!(result.is_err());
    }

    #[test]
    fn degenerate_triangles_skipped() {
        // Triangle with two coincident vertices should be skipped.
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 0.0), // Same as [0]
                Point3::new(1.0, 1.0, 0.0),
                // Valid triangle
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let solid = import_mesh(&mut topo, &mesh, 1e-6).unwrap();

        let s = topo.solid(solid).unwrap();
        let shell = topo.shell(s.outer_shell()).unwrap();
        // Only the valid triangle should remain.
        assert_eq!(shell.faces().len(), 1);
    }
}

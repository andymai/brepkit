//! Tessellation: convert B-Rep faces to triangle meshes.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::face::FaceId;

/// A triangle mesh produced by tessellation.
#[derive(Debug, Clone)]
pub struct TriangleMesh {
    /// Vertex positions.
    pub positions: Vec<Point3>,
    /// Per-vertex normals.
    pub normals: Vec<Vec3>,
    /// Triangle indices (groups of 3).
    pub indices: Vec<u32>,
}

/// Tessellate a face into a triangle mesh.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate(_face: FaceId, _deflection: f64) -> Result<TriangleMesh, crate::OperationsError> {
    todo!("tessellate not yet implemented")
}

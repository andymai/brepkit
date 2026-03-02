//! Tessellation: convert B-Rep faces to triangle meshes.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

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
/// For planar faces, this performs fan triangulation from the first vertex,
/// which produces correct results for convex polygons.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated (e.g. NURBS surfaces).
pub fn tessellate(
    topo: &Topology,
    face: FaceId,
    _deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    let face_data = topo.face(face)?;

    let normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Nurbs(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "tessellation of NURBS faces is not yet supported".into(),
            });
        }
    };

    // Collect wire vertex positions in traversal order.
    let wire = topo.wire(face_data.outer_wire())?;
    let mut positions = Vec::new();

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        positions.push(topo.vertex(vid)?.point());
    }

    let n = positions.len();
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("face has only {n} vertices, need at least 3"),
        });
    }

    // Per-vertex normals (all the same for a planar face).
    let normals = vec![normal; n];

    // Fan triangulation: triangle (0, i, i+1) for i in 1..n-1.
    let num_triangles = n - 2;
    let mut indices = Vec::with_capacity(num_triangles * 3);
    for i in 1..n - 1 {
        indices.push(0);
        #[allow(clippy::cast_possible_truncation)]
        indices.push(i as u32);
        #[allow(clippy::cast_possible_truncation)]
        indices.push((i + 1) as u32);
    }

    Ok(TriangleMesh {
        positions,
        normals,
        indices,
    })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};

    use super::*;

    #[test]
    fn tessellate_square() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 4);
        assert_eq!(mesh.normals.len(), 4);
        // 4 vertices → 2 triangles → 6 indices
        assert_eq!(mesh.indices.len(), 6);
    }

    #[test]
    fn tessellate_triangle() {
        let mut topo = Topology::new();
        let face = make_unit_triangle_face(&mut topo);

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 3);
        assert_eq!(mesh.normals.len(), 3);
        // 3 vertices → 1 triangle → 3 indices
        assert_eq!(mesh.indices.len(), 3);
    }
}

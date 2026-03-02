//! Tessellation: convert B-Rep faces to triangle meshes.

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

/// A triangle mesh produced by tessellation.
#[derive(Debug, Clone, Default)]
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
/// For NURBS faces, the surface is sampled on a uniform (u, v) grid whose
/// density is derived from `deflection` — smaller values produce finer meshes.
///
/// # Errors
///
/// Returns an error if the face geometry cannot be tessellated.
pub fn tessellate(
    topo: &Topology,
    face: FaceId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    let face_data = topo.face(face)?;

    match face_data.surface() {
        FaceSurface::Plane { normal, .. } => tessellate_planar(topo, face_data, *normal),
        FaceSurface::Nurbs(surface) => Ok(tessellate_nurbs(surface, deflection)),
    }
}

/// Tessellate a planar face via fan triangulation.
fn tessellate_planar(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    normal: Vec3,
) -> Result<TriangleMesh, crate::OperationsError> {
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

    let normals = vec![normal; n];

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

/// Tessellate a NURBS surface via uniform parameter sampling.
///
/// Samples on a `steps_u × steps_v` grid in [0,1]×[0,1] parameter space,
/// then splits each quad cell into two triangles.
fn tessellate_nurbs(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    deflection: f64,
) -> TriangleMesh {
    // Derive grid density from deflection (uniform in both directions).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let steps = ((1.0 / deflection).round() as usize).max(4);
    let verts = steps + 1;

    let mut positions = Vec::with_capacity(verts * verts);
    let mut normals = Vec::with_capacity(verts * verts);

    #[allow(clippy::cast_precision_loss)]
    let inv_steps = 1.0 / (steps as f64);
    for i in 0..verts {
        #[allow(clippy::cast_precision_loss)]
        let u = (i as f64) * inv_steps;
        for j in 0..verts {
            #[allow(clippy::cast_precision_loss)]
            let v = (j as f64) * inv_steps;

            positions.push(surface.evaluate(u, v));
            let n = surface.normal(u, v).unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            normals.push(n);
        }
    }

    // Triangulate: each quad cell becomes 2 triangles.
    let mut indices = Vec::with_capacity(steps * steps * 6);

    for i in 0..steps {
        for j in 0..steps {
            #[allow(clippy::cast_possible_truncation)]
            let base = (i * verts + j) as u32;
            #[allow(clippy::cast_possible_truncation)]
            let stride = verts as u32;

            // Triangle 1: (i,j) → (i+1,j) → (i+1,j+1)
            indices.push(base);
            indices.push(base + stride);
            indices.push(base + stride + 1);

            // Triangle 2: (i,j) → (i+1,j+1) → (i,j+1)
            indices.push(base);
            indices.push(base + stride + 1);
            indices.push(base + 1);
        }
    }

    TriangleMesh {
        positions,
        normals,
        indices,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

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

    /// Tessellate a simple bilinear NURBS surface (a flat quad as NURBS).
    #[test]
    fn tessellate_nurbs_surface() {
        let mut topo = Topology::new();

        // Create a simple degree-1×1 NURBS surface (bilinear patch) representing
        // a flat quad from (0,0,0) to (1,1,0).
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![
                vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
            ],
            vec![vec![1.0, 1.0], vec![1.0, 1.0]],
        )
        .unwrap();

        // Create a wire around the surface boundary (not strictly needed for
        // NURBS tessellation, but required for a valid Face).
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);

        let face = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

        let mesh = tessellate(&topo, face, 0.25).unwrap();

        // deflection=0.25 → steps = round(1/0.25) = 4
        // Grid: 5×5 = 25 vertices
        assert_eq!(mesh.positions.len(), 25);
        assert_eq!(mesh.normals.len(), 25);
        // 4×4 quads × 2 triangles × 3 indices = 96
        assert_eq!(mesh.indices.len(), 96);

        // All positions should be in [0,1] range since it's a flat quad.
        for pos in &mesh.positions {
            assert!(pos.x() >= -1e-10 && pos.x() <= 1.0 + 1e-10);
            assert!(pos.y() >= -1e-10 && pos.y() <= 1.0 + 1e-10);
            assert!((pos.z()).abs() < 1e-10);
        }
    }
}

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

/// Tessellate a planar face via ear-clipping triangulation.
///
/// Works for both convex and non-convex (simple) polygons by
/// projecting to 2D and using the ear-clipping algorithm.
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

    let normals_out = vec![normal; n];
    let indices = ear_clip_triangulate(&positions, normal);

    Ok(TriangleMesh {
        positions,
        normals: normals_out,
        indices,
    })
}

/// Ear-clipping triangulation for a simple polygon in 3D.
///
/// Projects the polygon to 2D (dropping the coordinate corresponding to
/// the dominant normal component), then applies the ear-clipping algorithm.
fn ear_clip_triangulate(positions: &[Point3], normal: Vec3) -> Vec<u32> {
    let n = positions.len();
    if n < 3 {
        return vec![];
    }
    if n == 3 {
        return vec![0, 1, 2];
    }

    // Project to 2D by dropping the dominant normal axis.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let project = |p: Point3| -> (f64, f64) {
        if az >= ax && az >= ay {
            (p.x(), p.y())
        } else if ay >= ax {
            (p.x(), p.z())
        } else {
            (p.y(), p.z())
        }
    };

    let pts2d: Vec<(f64, f64)> = positions.iter().map(|&p| project(p)).collect();

    // Ensure CCW winding in 2D.
    let signed_area = polygon_signed_area_2d(&pts2d);
    let ccw = signed_area > 0.0;

    // Active vertex list (indices into the original positions array).
    let mut active: Vec<usize> = if ccw {
        (0..n).collect()
    } else {
        (0..n).rev().collect()
    };

    let mut indices = Vec::with_capacity((n - 2) * 3);
    let mut safety = n * n; // prevent infinite loop on degenerate input

    while active.len() > 3 && safety > 0 {
        safety -= 1;
        let len = active.len();
        let mut found_ear = false;

        for i in 0..len {
            let prev = active[(i + len - 1) % len];
            let curr = active[i];
            let next = active[(i + 1) % len];

            // Check if this vertex forms a convex (left-turn) ear.
            let (ax2, ay2) = pts2d[prev];
            let (bx, by) = pts2d[curr];
            let (cx, cy) = pts2d[next];

            let cross = (bx - ax2).mul_add(cy - ay2, -(by - ay2) * (cx - ax2));
            if cross <= 0.0 {
                continue; // reflex vertex, not an ear
            }

            // Check that no other active vertex lies inside this triangle.
            let mut contains_point = false;
            for j in 0..len {
                if j == (i + len - 1) % len || j == i || j == (i + 1) % len {
                    continue;
                }
                let (px, py) = pts2d[active[j]];
                if point_in_triangle_2d(px, py, ax2, ay2, bx, by, cx, cy) {
                    contains_point = true;
                    break;
                }
            }

            if !contains_point {
                // This is an ear — emit the triangle.
                #[allow(clippy::cast_possible_truncation)]
                {
                    indices.push(prev as u32);
                    indices.push(curr as u32);
                    indices.push(next as u32);
                }
                active.remove(i);
                found_ear = true;
                break;
            }
        }

        if !found_ear {
            // Fallback: no ear found (degenerate polygon).
            // Use fan triangulation as best-effort.
            break;
        }
    }

    // Handle remaining triangle.
    if active.len() == 3 {
        #[allow(clippy::cast_possible_truncation)]
        {
            indices.push(active[0] as u32);
            indices.push(active[1] as u32);
            indices.push(active[2] as u32);
        }
    } else if active.len() > 3 {
        // Fallback fan triangulation for degenerate cases.
        for i in 1..active.len() - 1 {
            #[allow(clippy::cast_possible_truncation)]
            {
                indices.push(active[0] as u32);
                indices.push(active[i] as u32);
                indices.push(active[i + 1] as u32);
            }
        }
    }

    indices
}

/// Signed area of a 2D polygon (positive = CCW).
fn polygon_signed_area_2d(pts: &[(f64, f64)]) -> f64 {
    let n = pts.len();
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1;
        area -= pts[j].0 * pts[i].1;
    }
    area / 2.0
}

/// Test if point (px,py) is inside triangle (ax,ay)-(bx,by)-(cx,cy).
#[allow(clippy::too_many_arguments)]
fn point_in_triangle_2d(
    px: f64,
    py: f64,
    ax: f64,
    ay: f64,
    bx: f64,
    by: f64,
    cx: f64,
    cy: f64,
) -> bool {
    let d1 = (px - bx).mul_add(ay - by, -(ax - bx) * (py - by));
    let d2 = (px - cx).mul_add(by - cy, -(bx - cx) * (py - cy));
    let d3 = (px - ax).mul_add(cy - ay, -(cx - ax) * (py - ay));

    let has_neg = (d1 < 0.0) || (d2 < 0.0) || (d3 < 0.0);
    let has_pos = (d1 > 0.0) || (d2 > 0.0) || (d3 > 0.0);

    !(has_neg && has_pos)
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

    /// Test tessellation of an L-shaped (non-convex) polygon.
    #[test]
    fn tessellate_l_shape_nonconvex() {
        let mut topo = Topology::new();

        // L-shaped polygon on XY plane:
        //  (0,0) → (2,0) → (2,1) → (1,1) → (1,2) → (0,2) → (0,0)
        let points = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(2.0, 1.0, 0.0),
            Point3::new(1.0, 1.0, 0.0),
            Point3::new(1.0, 2.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
        ];

        let verts: Vec<_> = points
            .iter()
            .map(|&p| topo.vertices.alloc(Vertex::new(p, 1e-7)))
            .collect();

        let n = verts.len();
        let edges: Vec<_> = (0..n)
            .map(|i| {
                let next = (i + 1) % n;
                topo.edges
                    .alloc(Edge::new(verts[i], verts[next], EdgeCurve::Line))
            })
            .collect();

        let wire = Wire::new(
            edges.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);

        let face = topo.faces.alloc(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));

        let mesh = tessellate(&topo, face, 0.1).unwrap();

        assert_eq!(mesh.positions.len(), 6, "should have 6 vertices");
        // 6-gon → 4 triangles
        assert_eq!(
            mesh.indices.len(),
            12,
            "L-shape should have 4 triangles (12 indices)"
        );

        // Verify area: L-shape is 2×2 minus 1×1 = 3.0
        let mut total_area = 0.0;
        for t in 0..mesh.indices.len() / 3 {
            let i0 = mesh.indices[t * 3] as usize;
            let i1 = mesh.indices[t * 3 + 1] as usize;
            let i2 = mesh.indices[t * 3 + 2] as usize;
            let a = mesh.positions[i1] - mesh.positions[i0];
            let b = mesh.positions[i2] - mesh.positions[i0];
            total_area += 0.5 * a.cross(b).length();
        }
        assert!(
            (total_area - 3.0).abs() < 0.01,
            "L-shape area should be ~3.0, got {total_area}"
        );
    }
}

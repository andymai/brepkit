//! Linear extrusion of faces along a direction vector.
//!
//! Supports both planar and NURBS profile faces. For NURBS faces, the
//! extrusion translates all control points, preserving the exact surface
//! representation for both caps.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;

/// Extrude a planar face along a direction to produce a solid.
///
/// The extrusion creates a prism-like solid from the face. A reversed copy of
/// the original face becomes the bottom (outward normal pointing opposite to
/// the extrusion direction), an offset copy becomes the top, and rectangular
/// side faces connect them.
///
/// # Errors
///
/// Returns an error if the direction is zero-length, the face is not found,
/// or the face surface is not a plane.
#[allow(clippy::too_many_lines)]
pub fn extrude(
    topo: &mut Topology,
    face: FaceId,
    direction: Vec3,
    distance: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    // Validate direction is non-zero.
    if tol.approx_eq(direction.length_squared(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion direction is zero-length".into(),
        });
    }

    // Validate distance is non-zero.
    if tol.approx_eq(distance, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "extrusion distance is zero".into(),
        });
    }

    // Read the input face's data.
    let face_data = topo.face(face)?;
    let input_surface = face_data.surface().clone();
    let input_wire_id = face_data.outer_wire();

    // Read input wire oriented edges.
    let input_wire = topo.wire(input_wire_id)?;
    let input_oriented: Vec<_> = input_wire.edges().to_vec();

    // Collect start vertices of each oriented edge (traversal order).
    let mut input_verts: Vec<VertexId> = Vec::with_capacity(input_oriented.len());
    for oe in &input_oriented {
        let edge = topo.edge(oe.edge())?;
        let vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        input_verts.push(vid);
    }

    let n = input_verts.len();

    // Read input vertex positions.
    let input_positions: Vec<Point3> = input_verts
        .iter()
        .map(|&vid| {
            topo.vertex(vid)
                .map(brepkit_topology::vertex::Vertex::point)
        })
        .collect::<Result<_, _>>()?;

    // Compute offset vector.
    let offset = Vec3::new(
        direction.x() * distance,
        direction.y() * distance,
        direction.z() * distance,
    );

    // Create top vertices at offset positions.
    let top_verts: Vec<VertexId> = input_positions
        .iter()
        .map(|p| {
            let top_point = *p + offset;
            topo.vertices.alloc(Vertex::new(top_point, tol.linear))
        })
        .collect();

    // Original edge IDs from the input wire (used by side faces).
    let input_edge_ids: Vec<_> = input_oriented
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .collect();

    // Create top edges mirroring input edges (same winding as input).
    let mut top_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let next = (i + 1) % n;
        let top_edge = topo
            .edges
            .alloc(Edge::new(top_verts[i], top_verts[next], EdgeCurve::Line));
        top_edge_ids.push(top_edge);
    }

    // Create vertical edges: input_vert[i] → top_vert[i].
    let mut vertical_edge_ids = Vec::with_capacity(n);
    for i in 0..n {
        let vert_edge = topo
            .edges
            .alloc(Edge::new(input_verts[i], top_verts[i], EdgeCurve::Line));
        vertical_edge_ids.push(vert_edge);
    }

    let mut all_faces = Vec::with_capacity(n + 2);

    // --- Bottom face: reversed copy of input face ---
    // The bottom face normal points opposite to the extrusion direction.
    let reversed_bottom_edges: Vec<OrientedEdge> = input_oriented
        .iter()
        .rev()
        .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
        .collect();
    let bottom_wire =
        Wire::new(reversed_bottom_edges, true).map_err(crate::OperationsError::Topology)?;
    let bottom_wire_id = topo.wires.alloc(bottom_wire);

    let bottom_surface = match &input_surface {
        FaceSurface::Plane { normal, .. } => {
            let bottom_normal = Vec3::new(-normal.x(), -normal.y(), -normal.z());
            let bottom_d = dot_normal_point(bottom_normal, input_positions[0]);
            FaceSurface::Plane {
                normal: bottom_normal,
                d: bottom_d,
            }
        }
        FaceSurface::Nurbs(nurbs) => {
            // For NURBS bottom, reverse the surface orientation.
            // We keep the same surface (the bottom IS the original surface).
            FaceSurface::Nurbs(nurbs.clone())
        }
        other => other.clone(),
    };
    let bottom_face = topo
        .faces
        .alloc(Face::new(bottom_wire_id, vec![], bottom_surface));
    all_faces.push(bottom_face);

    // --- Side faces ---
    for i in 0..n {
        let next = (i + 1) % n;

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(input_edge_ids[i], input_oriented[i].is_forward()),
                OrientedEdge::new(vertical_edge_ids[next], true),
                OrientedEdge::new(top_edge_ids[i], false),
                OrientedEdge::new(vertical_edge_ids[i], false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.wires.alloc(side_wire);

        // Side normal = cross(edge_direction, extrusion_offset), normalized.
        let p0 = input_positions[i];
        let p1 = input_positions[next];
        let edge_dir = p1 - p0;
        let side_normal = edge_dir
            .cross(offset)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let side_d = dot_normal_point(side_normal, p0);

        let side_face = topo.faces.alloc(Face::new(
            side_wire_id,
            vec![],
            FaceSurface::Plane {
                normal: side_normal,
                d: side_d,
            },
        ));
        all_faces.push(side_face);
    }

    // --- Top face ---
    let top_wire = Wire::new(
        top_edge_ids
            .iter()
            .map(|&eid| OrientedEdge::new(eid, true))
            .collect(),
        true,
    )
    .map_err(crate::OperationsError::Topology)?;
    let top_wire_id = topo.wires.alloc(top_wire);

    let top_surface = match &input_surface {
        FaceSurface::Plane { normal, .. } => {
            let top_d = dot_normal_point(*normal, input_positions[0] + offset);
            FaceSurface::Plane {
                normal: *normal,
                d: top_d,
            }
        }
        FaceSurface::Nurbs(nurbs) => {
            // Translate all NURBS control points by the offset vector.
            let translated_cps: Vec<Vec<Point3>> = nurbs
                .control_points()
                .iter()
                .map(|row| row.iter().map(|&p| p + offset).collect())
                .collect();
            let translated_surface = brepkit_math::nurbs::surface::NurbsSurface::new(
                nurbs.degree_u(),
                nurbs.degree_v(),
                nurbs.knots_u().to_vec(),
                nurbs.knots_v().to_vec(),
                translated_cps,
                nurbs.weights().to_vec(),
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Nurbs(translated_surface)
        }
        other => other.clone(),
    };
    let top_face = topo
        .faces
        .alloc(Face::new(top_wire_id, vec![], top_surface));
    all_faces.push(top_face);

    // Assemble shell and solid.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.shells.alloc(shell);
    let solid = topo.solids.alloc(Solid::new(shell_id, vec![]));

    Ok(solid)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use std::collections::HashMap;

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::{make_unit_square_face, make_unit_triangle_face};

    use super::*;

    #[test]
    fn extrude_square_creates_box() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 4 sides + top + bottom = 6 faces
        assert_eq!(shell.faces().len(), 6);
        // 4 input + 4 top + 4 vertical = 12 edges (original input edges are reused)
        assert_eq!(topo.edges.len(), 12);
        // 4 input + 4 top = 8 vertices
        assert_eq!(topo.vertices.len(), 8);
    }

    #[test]
    fn extrude_triangle_creates_prism() {
        let mut topo = Topology::new();
        let face = make_unit_triangle_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        // 3 sides + top + bottom = 5 faces
        assert_eq!(shell.faces().len(), 5);
        assert_eq!(topo.edges.len(), 9);
        assert_eq!(topo.vertices.len(), 6);
    }

    #[test]
    fn extrude_zero_direction_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 0.0), 1.0);
        assert!(result.is_err());
    }

    #[test]
    fn extrude_zero_distance_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let result = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 0.0);
        assert!(result.is_err());
    }

    /// Verify that extruding a +Z face upward produces a solid where:
    /// - The bottom face normal points -Z (outward-downward)
    /// - The top face normal points +Z (outward-upward)
    /// - All edges are shared by exactly 2 faces (manifold)
    #[test]
    fn extrude_orientation_correct() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();

        let tol = Tolerance::new();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();

        let mut found_bottom = false;
        let mut found_top = false;
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                // Bottom: normal ≈ (0, 0, -1)
                if tol.approx_eq(normal.z(), -1.0)
                    && tol.approx_eq(normal.x(), 0.0)
                    && tol.approx_eq(normal.y(), 0.0)
                {
                    found_bottom = true;
                }
                // Top: normal ≈ (0, 0, 1)
                if tol.approx_eq(normal.z(), 1.0)
                    && tol.approx_eq(normal.x(), 0.0)
                    && tol.approx_eq(normal.y(), 0.0)
                {
                    found_top = true;
                }
            }
        }
        assert!(found_bottom, "bottom face should have -Z normal");
        assert!(found_top, "top face should have +Z normal");

        // Verify manifold: every edge used by exactly 2 faces.
        let mut edge_counts: HashMap<usize, usize> = HashMap::new();
        for &fid in shell.faces() {
            let f = topo.face(fid).unwrap();
            let wire = topo.wire(f.outer_wire()).unwrap();
            for oe in wire.edges() {
                *edge_counts.entry(oe.edge().index()).or_insert(0) += 1;
            }
        }
        for (&edge_idx, &count) in &edge_counts {
            assert_eq!(
                count, 2,
                "edge {edge_idx} shared by {count} faces, expected 2"
            );
        }
    }

    // ── NURBS face extrusion tests ──────────────────────

    /// Build a NURBS face: a curved surface on the XY plane.
    fn make_nurbs_face(topo: &mut Topology) -> FaceId {
        use brepkit_math::nurbs::surface::NurbsSurface;

        // Bicubic surface with some curvature.
        let cps = vec![
            vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
            vec![Point3::new(0.0, 1.0, 0.5), Point3::new(1.0, 1.0, 0.5)],
        ];
        let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let surface = NurbsSurface::new(1, 1, knots.clone(), knots, cps, weights).unwrap();

        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.5), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.5), tol));

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

        topo.faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)))
    }

    #[test]
    fn extrude_nurbs_face_creates_solid() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 4 sides + top + bottom = 6 faces
        assert_eq!(
            sh.faces().len(),
            6,
            "extruded NURBS face should have 6 faces"
        );
    }

    #[test]
    fn extrude_nurbs_face_top_is_nurbs() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // At least 2 faces should be NURBS (top and bottom caps).
        let nurbs_count = sh
            .faces()
            .iter()
            .filter(|&&fid| matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();

        assert!(
            nurbs_count >= 2,
            "extruded NURBS face should have at least 2 NURBS caps, got {nurbs_count}"
        );
    }

    #[test]
    fn extrude_nurbs_face_top_translated() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // Find the top NURBS face (the one at higher z).
        let mut top_z = f64::MIN;
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Nurbs(surface) = f.surface() {
                let pt = surface.evaluate(0.0, 0.0);
                if pt.z() > top_z {
                    top_z = pt.z();
                }
            }
        }

        // The top surface should be translated by distance 3.0 along Z.
        // Original (0,0) point is at z=0, so top should be at z=3.0.
        assert!(
            (top_z - 3.0).abs() < 0.1,
            "top NURBS surface should be at z≈3.0, got z={top_z}"
        );
    }

    #[test]
    fn extrude_nurbs_face_positive_volume() {
        let mut topo = Topology::new();
        let face = make_nurbs_face(&mut topo);

        let solid = extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "extruded NURBS solid should have positive volume, got {vol}"
        );
    }
}

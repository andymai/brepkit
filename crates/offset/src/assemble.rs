//! Final shell and solid assembly from offset faces and wire loops.

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::data::{OffsetData, OffsetStatus};
use crate::error::OffsetError;

/// Assemble the final offset solid from trimmed offset faces, joint
/// faces, and wire loops.
///
/// For each non-excluded offset face that has reconstructed wire loops,
/// a new [`Face`] is created with the offset surface and wires. All
/// new faces (including any joint faces from Phase 6) are collected
/// into a [`Shell`], which is then wrapped in a [`Solid`].
///
/// # Errors
///
/// Returns [`OffsetError::AssemblyFailed`] if no faces could be
/// assembled or the shell construction fails.
pub fn assemble_solid(topo: &mut Topology, data: &OffsetData) -> Result<SolidId, OffsetError> {
    let mut new_faces = Vec::new();

    // Create a face for each successfully offset face that has wire loops.
    for (face_id, offset_face) in &data.offset_faces {
        if offset_face.status == OffsetStatus::Excluded {
            continue;
        }

        let Some(wires) = data.face_wires.get(face_id) else {
            // No wires built for this face — skip with a warning.
            // In a production build this might log; for now we silently skip.
            continue;
        };

        if wires.is_empty() {
            continue;
        }

        let outer_wire = wires[0];
        let inner_wires = wires[1..].to_vec();

        let face = Face::new(outer_wire, inner_wires, offset_face.surface.clone());
        let face_id = topo.add_face(face);
        new_faces.push(face_id);
    }

    // Generate wall faces for thick-solid mode (excluded faces).
    if !data.excluded_faces.is_empty() {
        let wall_faces = build_wall_faces(topo, data)?;
        new_faces.extend(wall_faces);
    }

    // Include any joint faces created during Phase 6 (arc joints).
    for &joint_face in &data.joint_faces {
        new_faces.push(joint_face);
    }

    if new_faces.is_empty() {
        return Err(OffsetError::AssemblyFailed {
            reason: "no faces could be assembled for the offset solid".to_string(),
        });
    }

    let shell = Shell::new(new_faces)?;
    let shell_id = topo.add_shell(shell);

    let solid = Solid::new(shell_id, vec![]);
    let solid_id = topo.add_solid(solid);

    Ok(solid_id)
}

/// Build wall faces connecting excluded face edges to offset vertices.
///
/// For each edge of an excluded face, we compute where the adjacent
/// non-excluded face's offset surface is at the original edge vertices,
/// create new vertices there, and build a quad wall face connecting
/// the original edge to the offset positions.
fn build_wall_faces(topo: &mut Topology, data: &OffsetData) -> Result<Vec<FaceId>, OffsetError> {
    use brepkit_math::vec::Point3;

    let mut wall_faces = Vec::new();
    let tol = data.options.tolerance.linear;

    for &excluded_face_id in &data.excluded_faces {
        let outer_wire_id = topo.face(excluded_face_id)?.outer_wire();
        let wire_edges: Vec<_> = topo.wire(outer_wire_id)?.edges().to_vec();

        for oriented_edge in &wire_edges {
            let edge_id = oriented_edge.edge();

            // Find the adjacent non-excluded face for this edge.
            // Look through intersections OR use offset face data to compute
            // the offset position at the original vertices.
            //
            // Simple approach: offset each vertex along the face normal by
            // the offset distance. This works for planar faces.
            let edge = topo.edge(edge_id)?;
            let p0_id = if oriented_edge.is_forward() {
                edge.start()
            } else {
                edge.end()
            };
            let p1_id = if oriented_edge.is_forward() {
                edge.end()
            } else {
                edge.start()
            };

            let p0 = topo.vertex(p0_id)?.point();
            let p1 = topo.vertex(p1_id)?.point();

            // Get the excluded face's outward normal to determine offset direction.
            let excl_face = topo.face(excluded_face_id)?;
            let excl_normal = match excl_face.surface() {
                FaceSurface::Plane { normal, .. } => {
                    if excl_face.is_reversed() {
                        Vec3::new(-normal.x(), -normal.y(), -normal.z())
                    } else {
                        *normal
                    }
                }
                FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_)
                | FaceSurface::Nurbs(_) => {
                    // Non-planar excluded faces: wall generation not yet implemented.
                    continue;
                }
            };

            // The wall goes perpendicular to the excluded face normal,
            // inward by the offset distance. The offset vertices are the
            // original vertices displaced along the excluded face's inward
            // normal by |distance|.
            let disp = excl_normal * (-data.distance);
            let q0 = Point3::new(p0.x() + disp.x(), p0.y() + disp.y(), p0.z() + disp.z());
            let q1 = Point3::new(p1.x() + disp.x(), p1.y() + disp.y(), p1.z() + disp.z());

            // Create offset vertices.
            let q0_id = topo.add_vertex(Vertex::new(q0, tol));
            let q1_id = topo.add_vertex(Vertex::new(q1, tol));

            if let Some(face_id) = make_wall_quad(topo, p0_id, p1_id, q1_id, q0_id, p0, p1, q1, q0)?
            {
                wall_faces.push(face_id);
            }
        }
    }

    Ok(wall_faces)
}

/// Create a planar quad wall face from 4 vertices (p0→p1→p2→p3).
///
/// Returns `None` if the quad is degenerate (zero area).
#[allow(clippy::too_many_arguments)]
fn make_wall_quad(
    topo: &mut Topology,
    p0_id: VertexId,
    p1_id: VertexId,
    p2_id: VertexId,
    p3_id: VertexId,
    p0: brepkit_math::vec::Point3,
    p1: brepkit_math::vec::Point3,
    _p2: brepkit_math::vec::Point3,
    p3: brepkit_math::vec::Point3,
) -> Result<Option<FaceId>, OffsetError> {
    let edge_a = p1 - p0;
    let edge_b = p3 - p0;
    let cross = edge_a.cross(edge_b);
    let len = cross.length();
    // Degenerate quad (zero area cross product).
    if len < 1e-15 {
        return Ok(None);
    }
    let normal = cross * (1.0 / len);
    let d = normal.dot(Vec3::new(p0.x(), p0.y(), p0.z()));

    // Create 4 line edges.
    let e01 = topo.add_edge(Edge::new(p0_id, p1_id, EdgeCurve::Line));
    let e12 = topo.add_edge(Edge::new(p1_id, p2_id, EdgeCurve::Line));
    let e23 = topo.add_edge(Edge::new(p2_id, p3_id, EdgeCurve::Line));
    let e30 = topo.add_edge(Edge::new(p3_id, p0_id, EdgeCurve::Line));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e01, true),
            OrientedEdge::new(e12, true),
            OrientedEdge::new(e23, true),
            OrientedEdge::new(e30, true),
        ],
        true,
    )?;
    let wire_id = topo.add_wire(wire);
    let face = Face::new(wire_id, vec![], FaceSurface::Plane { normal, d });
    Ok(Some(topo.add_face(face)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions};
    use brepkit_topology::Topology;

    fn run_full_pipeline(topo: &mut Topology, solid: SolidId, distance: f64) -> SolidId {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        crate::inter3d::intersect_faces_3d(topo, solid, &mut data).unwrap();
        crate::inter2d::intersect_pcurves_2d(topo, solid, &mut data).unwrap();
        crate::loops::build_wire_loops(topo, &mut data).unwrap();
        assemble_solid(topo, &data).unwrap()
    }

    #[test]
    fn box_offset_produces_valid_solid() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = run_full_pipeline(&mut topo, solid, 0.5);

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6, "offset box should have 6 faces");
    }

    #[test]
    fn box_offset_faces_have_wires() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = run_full_pipeline(&mut topo, solid, 0.5);

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        for &fid in shell.faces() {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            assert_eq!(wire.edges().len(), 4, "each face should have 4 edges");
        }
    }

    #[test]
    fn box_offset_end_to_end() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let result = crate::offset_solid(
            &mut topo,
            solid,
            0.5,
            OffsetOptions {
                remove_self_intersections: false,
                ..Default::default()
            },
        )
        .unwrap();

        let shell_id = topo.solid(result).unwrap().outer_shell();
        let shell = topo.shell(shell_id).unwrap();
        assert_eq!(shell.faces().len(), 6);
    }

    fn run_thick_pipeline(
        topo: &mut Topology,
        solid: SolidId,
        distance: f64,
        exclude: Vec<FaceId>,
    ) -> SolidId {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), exclude);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        crate::inter3d::intersect_faces_3d(topo, solid, &mut data).unwrap();
        crate::inter2d::intersect_pcurves_2d(topo, solid, &mut data).unwrap();
        crate::loops::build_wire_loops(topo, &mut data).unwrap();
        assemble_solid(topo, &data).unwrap()
    }

    /// Thick solid with excluded face.
    ///
    /// Currently the wire loop builder doesn't handle faces adjacent to
    /// excluded faces (their loop is incomplete because the shared edge
    /// has no intersection). This test documents the current limitation.
    #[test]
    fn thick_solid_with_excluded_face() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let shell_id = topo.solid(solid).unwrap().outer_shell();
        let faces: Vec<_> = topo.shell(shell_id).unwrap().faces().to_vec();

        let exclude = vec![faces[0]];
        let result = run_thick_pipeline(&mut topo, solid, -0.1, exclude);

        let result_shell = topo
            .shell(topo.solid(result).unwrap().outer_shell())
            .unwrap();
        assert!(
            result_shell.faces().len() >= 9,
            "thick solid should have at least 9 faces (5 offset + 4 walls), got {}",
            result_shell.faces().len()
        );
    }
}

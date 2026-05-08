//! Shape contents analysis — entity counts and type breakdown.

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Summary of the topological contents of a solid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct ContentsAnalysis {
    /// Number of faces in the solid (outer shell + inner cavity shells).
    pub face_count: usize,
    /// Number of distinct edges across all faces.
    pub edge_count: usize,
    /// Number of distinct vertices across all faces.
    pub vertex_count: usize,
    /// Number of inner wires (holes) across all faces.
    pub hole_count: usize,
}

/// Analyze the contents (entity counts) of a solid.
///
/// Counts faces, edges, vertices, and inner wires across the outer
/// shell *and* any inner (cavity) shells. Hollow solids produced by
/// `shell_op` or boolean cuts hold their cavity faces in
/// `Solid::inner_shells()`; an outer-shell-only count would
/// under-report by the cavity face count.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn analyze_contents(topo: &Topology, solid_id: SolidId) -> Result<ContentsAnalysis, HealError> {
    let face_ids = solid_faces(topo, solid_id)?;

    let mut edge_set = HashSet::new();
    let mut vertex_set = HashSet::new();
    let mut hole_count = 0usize;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        hole_count += face.inner_wires().len();

        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                edge_set.insert(eid.index());
                let edge = topo.edge(eid)?;
                vertex_set.insert(edge.start().index());
                vertex_set.insert(edge.end().index());
            }
        }
    }

    Ok(ContentsAnalysis {
        face_count: face_ids.len(),
        edge_count: edge_set.len(),
        vertex_count: vertex_set.len(),
        hole_count,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn add_triangle_face(
        topo: &mut Topology,
        a: Point3,
        b: Point3,
        c: Point3,
    ) -> brepkit_topology::face::FaceId {
        let va = topo.add_vertex(Vertex::new(a, 1e-7));
        let vb = topo.add_vertex(Vertex::new(b, 1e-7));
        let vc = topo.add_vertex(Vertex::new(c, 1e-7));
        let eab = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
        let ebc = topo.add_edge(Edge::new(vb, vc, EdgeCurve::Line));
        let eca = topo.add_edge(Edge::new(vc, va, EdgeCurve::Line));
        let wire = Wire::new(
            vec![
                OrientedEdge::new(eab, true),
                OrientedEdge::new(ebc, true),
                OrientedEdge::new(eca, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    #[test]
    fn analyze_contents_counts_inner_shell_faces() {
        // A solid with 1 outer-shell face and 1 inner-shell face should
        // report 2 faces, not 1. The previous outer-shell-only
        // implementation under-counted hollow solids.
        let mut topo = Topology::new();

        let outer_face = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
        );
        let inner_face = add_triangle_face(
            &mut topo,
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(1.0, 0.0, 5.0),
            Point3::new(0.0, 1.0, 5.0),
        );

        let outer_shell = topo.add_shell(Shell::new(vec![outer_face]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![inner_face]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let analysis = analyze_contents(&topo, solid_id).unwrap();
        assert_eq!(
            analysis.face_count, 2,
            "should count both outer and inner faces"
        );
        assert_eq!(analysis.edge_count, 6, "3 edges per triangle × 2 triangles");
        assert_eq!(
            analysis.vertex_count, 6,
            "3 vertices per triangle × 2 triangles"
        );
        assert_eq!(analysis.hole_count, 0, "no inner wires on either face");
    }
}

//! Remove internal (hole) wires from faces.

use brepkit_topology::Topology;
use brepkit_topology::explorer::solid_faces;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Remove all inner (hole) wires from faces in a solid.
///
/// Walks the outer shell *and* any inner (cavity) shells so that hollow
/// solids' cavity-face inner wires are also removed — boolean cuts
/// and `shell_op` can produce cavity faces with their own internal
/// loops, which the prior outer-shell-only implementation silently
/// preserved.
///
/// Returns the total number of wires removed.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn remove_internal_wires(topo: &mut Topology, solid_id: SolidId) -> Result<usize, HealError> {
    let face_ids = solid_faces(topo, solid_id)?;

    let mut removed = 0;

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let n_inner = face.inner_wires().len();
        if n_inner > 0 {
            let face_mut = topo.face_mut(fid)?;
            face_mut.inner_wires_mut().clear();
            removed += n_inner;
        }
    }

    Ok(removed)
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

    fn add_face_with_n_inner_wires(
        topo: &mut Topology,
        anchor: Point3,
        n_inner: usize,
    ) -> brepkit_topology::face::FaceId {
        let make_triangle = |topo: &mut Topology, base: Point3| -> brepkit_topology::wire::WireId {
            let va = topo.add_vertex(Vertex::new(base, 1e-7));
            let vb = topo.add_vertex(Vertex::new(
                Point3::new(base.x() + 1.0, base.y(), base.z()),
                1e-7,
            ));
            let vc = topo.add_vertex(Vertex::new(
                Point3::new(base.x(), base.y() + 1.0, base.z()),
                1e-7,
            ));
            let eab = topo.add_edge(Edge::new(va, vb, EdgeCurve::Line));
            let ebc = topo.add_edge(Edge::new(vb, vc, EdgeCurve::Line));
            let eca = topo.add_edge(Edge::new(vc, va, EdgeCurve::Line));
            topo.add_wire(
                Wire::new(
                    vec![
                        OrientedEdge::new(eab, true),
                        OrientedEdge::new(ebc, true),
                        OrientedEdge::new(eca, true),
                    ],
                    true,
                )
                .unwrap(),
            )
        };

        let outer_wid = make_triangle(topo, anchor);
        let inner_wires: Vec<_> = (0..n_inner)
            .map(|i| {
                make_triangle(
                    topo,
                    Point3::new(
                        anchor.x() + 0.1 * (i as f64 + 1.0),
                        anchor.y() + 0.1,
                        anchor.z(),
                    ),
                )
            })
            .collect();
        topo.add_face(Face::new(
            outer_wid,
            inner_wires,
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ))
    }

    #[test]
    fn remove_internal_wires_walks_inner_shells() {
        // Outer shell: 1 face with 2 inner wires.
        // Inner (cavity) shell: 1 face with 3 inner wires.
        // Total expected removals: 2 + 3 = 5.
        let mut topo = Topology::new();

        let outer_face = add_face_with_n_inner_wires(&mut topo, Point3::new(0.0, 0.0, 0.0), 2);
        let inner_face = add_face_with_n_inner_wires(&mut topo, Point3::new(0.0, 0.0, 5.0), 3);

        let outer_shell = topo.add_shell(Shell::new(vec![outer_face]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![inner_face]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let removed = remove_internal_wires(&mut topo, solid_id).unwrap();
        assert_eq!(
            removed, 5,
            "expected 5 inner-wire removals (2 outer + 3 inner), got {removed}"
        );

        // Both faces should now have zero inner wires.
        assert_eq!(topo.face(outer_face).unwrap().inner_wires().len(), 0);
        assert_eq!(topo.face(inner_face).unwrap().inner_wires().len(), 0);
    }
}

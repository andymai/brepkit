//! The faults a walking fillet is refused for are read on the faces it made
//! or trimmed, not on faces it left as they were. A box whose bottom face
//! carries a hole across its outline is not a valid solid, but rounding a
//! top edge leaves the bottom alone, so the round is taken.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::primitives::make_box;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    #[test]
    fn a_fault_the_fillet_left_alone_does_not_refuse_it() {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
        // A hole about the bottom's corner at the origin, across its outline.
        let bottom = solid_faces(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                    if normal.z().abs() > 0.5 && d.abs() < 1e-9)
            })
            .unwrap();
        let z = Vec3::new(0.0, 0.0, 1.0);
        let circle =
            Circle3D::new_with_ref(Point3::new(0.0, 0.0, 0.0), z, 2.0, Vec3::new(1.0, 0.0, 0.0))
                .unwrap();
        let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
        let hole = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        topo.face_mut(bottom).unwrap().inner_wires_mut().push(hole);

        let top = solid_edges(&topo, block)
            .unwrap()
            .into_iter()
            .find(|&edge| {
                let data = topo.edge(edge).unwrap();
                [data.start(), data.end()].iter().all(|&v| {
                    let p = topo.vertex(v).unwrap().point();
                    (p.z() - 10.0).abs() < 1e-9 && (p.y() - 10.0).abs() < 1e-9
                })
            })
            .unwrap();
        let result = fillet_v2(&mut topo, block, &[top], 1.0).unwrap();
        assert!(result.failed.is_empty(), "refused: {:?}", result.failed);
        assert_ne!(result.solid, block);
    }
}

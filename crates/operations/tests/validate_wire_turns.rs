//! A wire that runs round its face's surface twice bounds no region of it.
//!
//! Where a chamfer of a rim on a closed B-spline wall uses each of its two
//! contact edges the wrong way in every face round them (the wall, the
//! chamfer band and the cap), the faces still use each shared edge once each
//! way, so the shell's checks pass. The cap and the band have both their
//! rims turned, which turns each of them as a whole. On the wall, the rim
//! left uncut and the contact run the same way round, so the wall's wire
//! runs twice round it. That wall meshes wrong: the solid measures 39.684588
//! for the cylinder's bottom rim against 112.181038.
//!
//! The solid here is the chamfer that the builder makes, with its contacts'
//! uses turned.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::collections::HashSet;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::blend_ops::chamfer_v2;
    use brepkit_operations::primitives::{make_cone, make_cylinder};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.7)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// A cylinder (3, 4) or a frustum (3 to 1, 4 high) with its wall in its
    /// B-spline form, placed by `pose`, and its bottom or top rim.
    fn b_spline_walled(
        topo: &mut Topology,
        frustum: bool,
        top: bool,
        pose: &Mat4,
    ) -> (SolidId, EdgeId) {
        let solid = if frustum {
            make_cone(topo, 3.0, 1.0, 4.0).unwrap()
        } else {
            make_cylinder(topo, 3.0, 4.0).unwrap()
        };
        let height = if top { 4.0 } else { 0.0 };
        let rim = solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if (c.center().z() - height).abs() < 1e-9)
            })
            .unwrap();
        for face in solid_faces(topo, solid).unwrap() {
            if topo.face(face).unwrap().surface().is_planar() {
                continue;
            }
            let wall = brepkit_heal::custom::convert_to_bspline::face_bspline_surface(topo, face)
                .unwrap()
                .unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(FaceSurface::Nurbs(wall));
        }
        transform_solid(topo, solid, pose).unwrap();
        (solid, rim)
    }

    /// Each use of the chamfer's two contacts in `result`, the closed
    /// B-spline edges it made, turned the other way.
    fn turn_contacts(topo: &mut Topology, result: SolidId) {
        let contacts: HashSet<EdgeId> = solid_edges(topo, result)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                edge.start() == edge.end() && matches!(edge.curve(), EdgeCurve::NurbsCurve(_))
            })
            .collect();
        assert_eq!(contacts.len(), 2);
        for face in solid_faces(topo, result).unwrap() {
            let outer = topo.face(face).unwrap().outer_wire();
            let uses: Vec<OrientedEdge> = topo.wire(outer).unwrap().edges().to_vec();
            if !uses.iter().any(|oe| contacts.contains(&oe.edge())) {
                continue;
            }
            let turned = uses
                .iter()
                .map(|oe| {
                    OrientedEdge::new(oe.edge(), oe.is_forward() != contacts.contains(&oe.edge()))
                })
                .collect();
            let wire = topo.add_wire(Wire::new(turned, true).unwrap());
            topo.face_mut(face).unwrap().set_outer_wire(wire);
        }
    }

    /// `validate_solid` reports the wall's wire of the chamfer with its
    /// contacts turned as running round the wall twice. The chamfer as built
    /// passes.
    #[test]
    fn a_chamfer_with_its_contacts_turned_is_reported() {
        for (frustum, top) in [(false, false), (false, true), (true, false)] {
            for (name, pose) in poses() {
                let what = format!("frustum {frustum} top {top} {name}");
                let mut topo = Topology::new();
                let (solid, rim) = b_spline_walled(&mut topo, frustum, top, &pose);
                let built = chamfer_v2(&mut topo, solid, &[rim], 0.4, 0.25).unwrap();
                assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
                let report = validate_solid(&topo, built.solid).unwrap();
                assert!(report.is_valid(), "{what}: {report:?}");

                turn_contacts(&mut topo, built.solid);
                let report = validate_solid(&topo, built.solid).unwrap();
                assert!(
                    report
                        .issues
                        .iter()
                        .any(|i| i.description.contains("runs round the surface 2 times")),
                    "{what}: {report:?}"
                );
            }
        }
    }
}

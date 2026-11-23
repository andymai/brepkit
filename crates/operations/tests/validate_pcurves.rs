//! A pcurve that does not follow its edge is a fault of the solid.
//!
//! A walking chamfer once gave the contact on a cylinder's B-spline wall a
//! pcurve that ran over u from 1 to 2, a whole turn past the wall's knot
//! domain from 0 to 1. Read through the wall, it lay 5.98 off its edge, and
//! the solid's mesh was open, yet the operations `validate_solid` passed it:
//! it did not read pcurves. The solid here is the chamfer that the builder
//! makes, with its contact's pcurve on the wall moved one turn along u.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::collections::HashSet;

    use brepkit_math::curves2d::{Curve2D, Line2D, NurbsCurve2D};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Vec2;
    use brepkit_operations::blend_ops::chamfer_v2;
    use brepkit_operations::primitives::{make_cone, make_cylinder};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::{FaceId, FaceSurface};
    use brepkit_topology::pcurve::PCurve;
    use brepkit_topology::solid::SolidId;

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

    /// `curve` moved `du` along u.
    fn moved(curve: &Curve2D, du: f64) -> Curve2D {
        let by = Vec2::new(du, 0.0);
        match curve {
            Curve2D::Line(l) => Curve2D::Line(Line2D::new(l.origin() + by, l.direction()).unwrap()),
            Curve2D::Nurbs(n) => Curve2D::Nurbs(
                NurbsCurve2D::new(
                    n.degree(),
                    n.knots().to_vec(),
                    n.control_points().iter().map(|&p| p + by).collect(),
                    n.weights().to_vec(),
                )
                .unwrap(),
            ),
            other => panic!("a contact's pcurve is a line or a B-spline: {other:?}"),
        }
    }

    /// Each pcurve on the B-spline wall of an edge that `old` does not
    /// hold, with the length of the wall's span in u.
    fn contact_pcurves(
        topo: &Topology,
        result: SolidId,
        old: &HashSet<EdgeId>,
    ) -> Vec<(EdgeId, FaceId, f64)> {
        let mut found = Vec::new();
        for face in solid_faces(topo, result).unwrap() {
            let f = topo.face(face).unwrap();
            let FaceSurface::Nurbs(wall) = f.surface() else {
                continue;
            };
            let (u0, u1) = wall.domain_u();
            for oe in topo.wire(f.outer_wire()).unwrap().edges() {
                if !old.contains(&oe.edge()) && topo.pcurves().get(oe.edge(), face).is_some() {
                    found.push((oe.edge(), face, u1 - u0));
                }
            }
        }
        found
    }

    /// The chamfer of a rim of a B-spline wall passes `validate_solid` as
    /// built. With its contact's pcurve on the wall moved one turn along u,
    /// off the wall's knot domain, `validate_solid` reports that pcurve.
    #[test]
    fn a_chamfers_contact_pcurve_a_turn_off_its_wall_is_reported() {
        for (frustum, top) in [(false, false), (false, true), (true, false)] {
            for (name, pose) in poses() {
                let what = format!("frustum {frustum} top {top} {name}");
                let mut topo = Topology::new();
                let (solid, rim) = b_spline_walled(&mut topo, frustum, top, &pose);
                let old: HashSet<EdgeId> = solid_edges(&topo, solid).unwrap().into_iter().collect();
                let built = chamfer_v2(&mut topo, solid, &[rim], 0.4, 0.25).unwrap();
                assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
                let report = validate_solid(&topo, built.solid).unwrap();
                assert!(report.is_valid(), "{what}: {report:?}");

                let contacts = contact_pcurves(&topo, built.solid, &old);
                assert!(
                    !contacts.is_empty(),
                    "{what}: no contact pcurve on the wall"
                );
                for (edge, face, turn) in contacts {
                    let pcurve = topo.pcurves().get(edge, face).unwrap().clone();
                    let off = PCurve::new(
                        moved(pcurve.curve(), turn),
                        pcurve.t_start(),
                        pcurve.t_end(),
                    );
                    topo.pcurves_mut().set(edge, face, off);
                }
                let report = validate_solid(&topo, built.solid).unwrap();
                assert!(
                    report
                        .issues
                        .iter()
                        .any(|i| i.description.contains("pcurve of edge")),
                    "{what}: {report:?}"
                );
            }
        }
    }
}

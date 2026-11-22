//! Two solid cylinders of one radius whose axes cross at right angles, on
//! z and on y, through their middles. They share a bicylinder of volume
//! `16 r^3 / 3`, so no boolean of the two may come back as its operands.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;

    const LENGTH: f64 = 60.0;

    fn poses() -> Vec<(&'static str, Mat4)> {
        let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
        vec![
            ("upright", Mat4::identity()),
            ("turned", turn),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
            ("turned and mirrored", Mat4::scale(1.0, -1.0, 1.0) * turn),
            ("rolled about z", Mat4::rotation_z(2.0)),
        ]
    }

    /// Each op of the two cylinders, with no mesh fallback, in each pose:
    /// an exact solid of cylinder and plane faces, valid, meshed closed, at
    /// its closed-form volume. A fuse that kept both walls whole read the
    /// sum of the two cylinders, its intersect nothing, and its cut the
    /// first cylinder.
    #[test]
    fn crossing_cylinders_of_one_radius_share_their_bicylinder() {
        for r in [8.25, 3.0] {
            let cylinder = PI * r * r * LENGTH;
            let shared = 16.0 * r.powi(3) / 3.0;
            for (op, truth) in [
                (BooleanOp::Fuse, 2.0 * cylinder - shared),
                (BooleanOp::Cut, cylinder - shared),
                (BooleanOp::Intersect, shared),
            ] {
                for (name, pose) in poses() {
                    let label = format!("r {r} {op:?} {name}");
                    let mut topo = Topology::new();
                    let a = make_cylinder(&mut topo, r, LENGTH).unwrap();
                    let up = Mat4::translation(0.0, 0.0, -LENGTH / 2.0);
                    transform_solid(&mut topo, a, &(pose * up)).unwrap();
                    let b = make_cylinder(&mut topo, r, LENGTH).unwrap();
                    let across = Mat4::rotation_x(PI / 2.0) * up;
                    transform_solid(&mut topo, b, &(pose * across)).unwrap();

                    let options = BooleanOptions {
                        mesh_fallback: false,
                        ..BooleanOptions::default()
                    };
                    let result = boolean_with_options(&mut topo, op, a, b, options).unwrap();

                    for face in solid_faces(&topo, result).unwrap() {
                        let surface = topo.face(face).unwrap().surface();
                        assert!(
                            matches!(
                                surface,
                                FaceSurface::Cylinder(_) | FaceSurface::Plane { .. }
                            ),
                            "{label}: a {} face",
                            surface.type_tag()
                        );
                    }
                    let report = validate_solid(&topo, result).unwrap();
                    assert!(report.is_valid(), "{label}: {:?}", report.issues);
                    let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
                    assert_eq!(boundary_edge_count(&mesh), 0, "{label}: open mesh");
                    let volume = solid_volume(&topo, result, 0.001).unwrap();
                    assert!(
                        ((volume - truth) / truth).abs() < 1e-6,
                        "{label}: volume {volume} against {truth}"
                    );
                }
            }
        }
    }
}

//! A wire on a closed surface that runs up a seam and back down it uses the
//! seam once each way. Its two uses lie along each other, and the wire
//! passes the seam's vertices twice, but it does not cross itself.
//!
//! The check crate's `validate_solid` read every pair of a wire's edges that
//! do not follow each other in 3D, and took any two that came within its
//! tolerance for a crossing. It reported the seam of a cylinder, a frustum
//! and a torus that way. It now reads a wire as the operations validator
//! reads a plane face's wire: where two edges meet, the wire is followed a
//! short way each side, and a crossing is read in the surface's tangent
//! plane there.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_check::validate::{CheckId, ValidateOptions, validate_solid as check_solid};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{
        make_box, make_cone, make_cylinder, make_sphere, make_torus,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
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

    const PRIMITIVES: [&str; 5] = ["cylinder", "cone", "frustum", "ball", "ring"];

    fn primitive(topo: &mut Topology, name: &str) -> SolidId {
        match name {
            "cylinder" => make_cylinder(topo, 3.0, 4.0).unwrap(),
            "cone" => make_cone(topo, 3.0, 0.0, 4.0).unwrap(),
            "frustum" => make_cone(topo, 3.0, 1.0, 4.0).unwrap(),
            "ball" => make_sphere(topo, 3.0, 16).unwrap(),
            _ => make_torus(topo, 4.0, 1.5, 16).unwrap(),
        }
    }

    /// Each curved face of `solid` given its B-spline form.
    fn b_spline_walled(topo: &mut Topology, solid: SolidId) {
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
    }

    fn crossings(topo: &Topology, solid: SolidId) -> Vec<String> {
        check_solid(topo, solid, &ValidateOptions::default())
            .unwrap()
            .issues
            .into_iter()
            .filter(|issue| issue.check == CheckId::WireSelfIntersection)
            .map(|issue| issue.description)
            .collect()
    }

    /// A cylinder, a pointed cone, a frustum, a ball and a ring, each with
    /// its curved faces as they are built and in their B-spline form, in
    /// four poses: the check crate's `validate_solid` finds no error.
    #[test]
    fn plain_primitives_pass_the_check_crates_validate_solid() {
        for name in PRIMITIVES {
            for spline in [false, true] {
                for (pose, place) in poses() {
                    let what = format!("{name} spline {spline} {pose}");
                    let mut topo = Topology::new();
                    let solid = primitive(&mut topo, name);
                    if spline {
                        b_spline_walled(&mut topo, solid);
                    }
                    transform_solid(&mut topo, solid, &place).unwrap();
                    let report = check_solid(&topo, solid, &ValidateOptions::default()).unwrap();
                    assert!(report.is_valid(), "{what}: {:?}", report.issues);
                }
            }
        }
    }

    /// A slab across a frustum and across a ring leaves bands whose rims
    /// are whole circles that start away from the seam's vertex. Each band's
    /// wire runs round a rim from that vertex and up the seam: it does not
    /// cross itself. The operations validator, which does not read curved
    /// faces' wires for crossings, accepts each result.
    #[test]
    fn bands_a_slab_leaves_on_a_frustum_and_a_ring_do_not_cross_themselves() {
        let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(0.3);
        let (at, normal) = (Point3::new(0.3, 0.0, 0.0), Vec3::new(1.0, 0.2, 0.1));
        for name in ["frustum", "ring"] {
            for op in [BooleanOp::Cut, BooleanOp::Intersect] {
                for pose in ["upright", "turned", "mirrored"] {
                    let what = format!("{name} {op:?} {pose}");
                    let mut topo = Topology::new();
                    let mut body = if name == "frustum" {
                        let c = make_cone(&mut topo, 3.0, 1.5, 6.0).unwrap();
                        transform_solid(&mut topo, c, &Mat4::translation(0.0, 0.0, -3.0)).unwrap();
                        c
                    } else {
                        make_torus(&mut topo, 4.0, 1.5, 32).unwrap()
                    };
                    let mut slab = make_box(&mut topo, 20.0, 20.0, 1.0).unwrap();
                    transform_solid(&mut topo, slab, &Mat4::translation(-10.0, -10.0, 1.0))
                        .unwrap();
                    if pose == "turned" {
                        transform_solid(&mut topo, body, &turn).unwrap();
                        transform_solid(&mut topo, slab, &turn).unwrap();
                    } else if pose == "mirrored" {
                        body = mirror(&mut topo, body, at, normal).unwrap();
                        slab = mirror(&mut topo, slab, at, normal).unwrap();
                    }
                    let result = boolean(&mut topo, op, body, slab).unwrap();
                    let report = validate_solid(&topo, result).unwrap();
                    assert!(report.is_valid(), "{what}: {report:?}");
                    let found = crossings(&topo, result);
                    assert!(found.is_empty(), "{what}: {found:?}");
                }
            }
        }
    }
}

//! A cylinder of radius 1 and height 2 cut by a slab whose lower face, tilted
//! 0.5 about y, passes through the point (1, 0, 0) of the bottom rim where
//! its seam starts. The section is an ellipse closed at the rim's own vertex,
//! so the wall that stays runs between two loops through one vertex: a step
//! that takes two closed edges at one vertex for one edge welds the rim and
//! the ellipse together. The result keeps its bottom, its wall and its
//! slanted cap, and holds pi tan 0.5. Every case runs in a turned pose.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;

    const TILT: f64 = 0.5;

    fn poses() -> [(&'static str, Mat4); 5] {
        [
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "tipped",
                Mat4::translation(-1.0, 4.0, 2.0) * Mat4::rotation_y(0.9) * Mat4::rotation_x(-0.3),
            ),
            (
                "mirrored",
                Mat4::translation(1.0, 0.5, -2.0)
                    * Mat4::scale(-1.0, 1.0, 1.0)
                    * Mat4::rotation_x(0.4)
                    * Mat4::rotation_z(0.6),
            ),
            ("leaned", Mat4::rotation_x(0.4)),
            (
                "spun",
                Mat4::translation(2.0, 2.0, 2.0) * Mat4::rotation_z(1.3) * Mat4::rotation_y(0.2),
            ),
        ]
    }

    #[test]
    fn a_slanted_cut_through_the_rim_seam_keeps_the_rim_and_the_section_apart() {
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let cylinder = make_cylinder(&mut topo, 1.0, 2.0).unwrap();
            transform_solid(&mut topo, cylinder, &pose).unwrap();
            let slab = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            let place = pose
                * Mat4::translation(1.0, 0.0, 0.0)
                * Mat4::rotation_y(TILT)
                * Mat4::translation(-5.0, -5.0, 0.0);
            transform_solid(&mut topo, slab, &place).unwrap();
            let result =
                boolean_with_options(&mut topo, BooleanOp::Cut, cylinder, slab, options).unwrap();
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let faces = solid_faces(&topo, result).unwrap().len();
            assert_eq!(faces, 3, "{name}: {faces} faces");
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert!(is_watertight(&mesh), "{name}: open mesh");
            let volume = solid_volume(&topo, result, 0.001).unwrap();
            let want = std::f64::consts::PI * TILT.tan();
            assert!(
                (volume - want).abs() < 1e-9,
                "{name}: volume {volume}, want {want}"
            );
            // The cap lies at height tan(0.5) (1 - x), 0.546 over the axis.
            for (p, inside) in [
                ([0.0, 0.0, 0.3], true),
                ([0.0, 0.0, 0.8], false),
                ([-0.9, 0.0, 0.9], true),
                ([-0.9, 0.0, 1.2], false),
                ([0.9, 0.0, 0.02], true),
            ] {
                let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
                let got = classify_point(&topo, result, q, 0.01, 1e-7).unwrap();
                assert_eq!(
                    matches!(got, PointClassification::Inside),
                    inside,
                    "{name}: at {p:?}"
                );
            }
        }
    }
}

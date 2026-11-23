//! A short rod standing in a drum, parallel to its axis, that pokes out of
//! the drum's wall over part of its turn. Every vertex of the rod lies in
//! the drum, and turned, its box lies in the drum's box, so the rod is
//! taken for one inside the drum unless a point of its rims is found out
//! of the wall.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    /// The rod's axis, in the drum's frame, and its radius and ends.
    const ROD_AT: (f64, f64) = (0.804_222_615_208_922_9, -0.368_228_326_007_635_55);
    const ROD_RADIUS: f64 = 0.433_646_389_596_180_9;
    const ROD_ENDS: (f64, f64) = (2.923_517_338_973_687_7, 3.061_756_322_428_605_5);

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

    /// The area two discs of radii `a` and `b`, `d` apart, share.
    fn lens(a: f64, b: f64, d: f64) -> f64 {
        let p = a
            * a
            * ((d * d + a * a - b * b) / (2.0 * d * a))
                .clamp(-1.0, 1.0)
                .acos();
        let q = b
            * b
            * ((d * d + b * b - a * a) / (2.0 * d * b))
                .clamp(-1.0, 1.0)
                .acos();
        let r = 0.5
            * ((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b))
                .max(0.0)
                .sqrt();
        p + q - r
    }

    /// The volume of the rod in the drum.
    fn shared() -> f64 {
        (ROD_ENDS.1 - ROD_ENDS.0) * lens(1.0, ROD_RADIUS, ROD_AT.0.hypot(ROD_AT.1))
    }

    fn rod_volume() -> f64 {
        (ROD_ENDS.1 - ROD_ENDS.0) * PI * ROD_RADIUS * ROD_RADIUS
    }

    fn drum_volume() -> f64 {
        4.0 * PI
    }

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

    /// The drum (radius 1, from z = 0 to 4) and the rod, in `pose`.
    fn build(topo: &mut Topology, pose: &Mat4) -> (SolidId, SolidId) {
        let drum = make_cylinder(topo, 1.0, 4.0).unwrap();
        transform_solid(
            topo,
            drum,
            &(*pose * Mat4::rotation_z(5.253_812_970_944_041)),
        )
        .unwrap();
        let rod = make_cylinder(topo, ROD_RADIUS, ROD_ENDS.1 - ROD_ENDS.0).unwrap();
        let place = Mat4::translation(ROD_AT.0, ROD_AT.1, ROD_ENDS.0)
            * Mat4::rotation_z(4.283_357_772_576_871);
        transform_solid(topo, rod, &(*pose * place)).unwrap();
        (drum, rod)
    }

    /// Points of the rod out of the wall and in it, before the pose: on the
    /// line from the drum's axis through the rod's, 1.1 and 0.7 out.
    fn out_and_in() -> [Point3; 2] {
        let d = ROD_AT.0.hypot(ROD_AT.1);
        let z = 0.5 * (ROD_ENDS.0 + ROD_ENDS.1);
        [1.1, 0.7].map(|s| Point3::new(ROD_AT.0 * s / d, ROD_AT.1 * s / d, z))
    }

    fn check(op: BooleanOp, truth: f64, want: [PointClassification; 2]) {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (drum, rod) = build(&mut topo, &pose);
            let result = boolean_with_options(&mut topo, op, drum, rod, exact()).unwrap();
            assert!(
                topo.solid(result).unwrap().inner_shells().is_empty(),
                "{name}: {op:?} has a cavity"
            );
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name}: {op:?} {:?}", report.issues);
            let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{name}: {op:?} open mesh");
            let volume = solid_volume(&topo, result, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 1e-5,
                "{name}: {op:?} volume {volume} against {truth}"
            );
            for (p, want) in out_and_in().into_iter().zip(want) {
                let at = pose.mul_point(p);
                let got = classify_point(&topo, result, at, &ClassifyOptions::default()).unwrap();
                assert_eq!(got, want, "{name}: {op:?} at {p:?}");
            }
        }
    }

    #[test]
    fn fuse_keeps_the_part_out_of_the_wall() {
        check(
            BooleanOp::Fuse,
            drum_volume() + rod_volume() - shared(),
            [PointClassification::Inside, PointClassification::Inside],
        );
    }

    #[test]
    fn cut_notches_the_wall() {
        check(
            BooleanOp::Cut,
            drum_volume() - shared(),
            [PointClassification::Outside, PointClassification::Outside],
        );
    }

    #[test]
    fn intersect_keeps_the_part_in_the_drum() {
        check(
            BooleanOp::Intersect,
            shared(),
            [PointClassification::Outside, PointClassification::Inside],
        );
    }
}

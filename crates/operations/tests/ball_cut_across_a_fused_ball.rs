//! A ball fused with a small ball on its top, then cut by a third ball
//! that swallows the joint between the two: the small ball's cap is left
//! standing clear of the cratered ball.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_sphere;
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    /// Centre, radius and turn about x, y and z of each ball.
    const BALLS: [([f64; 3], f64, [f64; 3]); 3] = [
        (
            [0.0, 0.0, 0.0],
            1.0,
            [
                2.279_591_927_054_571,
                0.324_558_142_417_568_6,
                4.938_624_814_893_679_5,
            ],
        ),
        (
            [
                0.010_294_717_767_945_251,
                0.038_993_825_251_738_126,
                1.144_734_792_356_348_7,
            ],
            0.192_935_220_745_243,
            [0.0, 0.0, 0.0],
        ),
        (
            [
                0.046_322_836_985_853_884,
                0.020_986_724_242_115_944,
                0.651_674_045_112_836_7,
            ],
            0.465_674_090_318_948_35,
            [
                -0.011_170_437_513_698_839,
                0.238_513_292_667_748_07,
                0.572_685_478_397_596_4,
            ],
        ),
    ];

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

    fn point(c: [f64; 3]) -> Point3 {
        Point3::new(c[0], c[1], c[2])
    }

    /// The volume two balls of radii `a` and `b`, `d` apart, share.
    fn shared(a: f64, b: f64, d: f64) -> f64 {
        PI * (a + b - d).powi(2) * (d * d + 2.0 * d * (a + b) - 3.0 * (a - b).powi(2)) / (12.0 * d)
    }

    /// The fused balls less the third. The lens the first two share lies
    /// inside the third (checked on the circle where their spheres meet, the
    /// lens's farthest points from the third's centre), so the third takes
    /// what it shares with each of the first two, and the lens is counted
    /// in neither.
    fn truth() -> f64 {
        let [(c0, r0, _), (c1, r1, _), (c2, r2, _)] = BALLS;
        let (c0, c1, c2) = (point(c0), point(c1), point(c2));
        let axis = (c1 - c0).normalize().unwrap();
        let d01 = (c1 - c0).length();
        let along = (d01 * d01 + r0 * r0 - r1 * r1) / (2.0 * d01);
        let rim = (r0 * r0 - along * along).sqrt();
        let across = axis.cross(brepkit_math::vec::Vec3::new(1.0, 0.0, 0.0));
        let across = across.normalize().unwrap();
        let other = axis.cross(across);
        for k in 0..64 {
            let a = 2.0 * PI * f64::from(k) / 64.0;
            let p = c0 + axis * along + across * (rim * a.cos()) + other * (rim * a.sin());
            assert!(
                (p - c2).length() < r2,
                "the joint reaches past the third ball"
            );
        }
        let ball = |r: f64| 4.0 / 3.0 * PI * r.powi(3);
        ball(r0) + ball(r1)
            - shared(r0, r2, (c2 - c0).length())
            - shared(r1, r2, (c2 - c1).length())
    }

    fn build(topo: &mut Topology, pose: &Mat4) -> Vec<SolidId> {
        BALLS
            .iter()
            .map(|&(c, r, spin)| {
                let s = make_sphere(topo, r, 32).unwrap();
                let place = Mat4::translation(c[0], c[1], c[2])
                    * Mat4::rotation_z(spin[2])
                    * Mat4::rotation_y(spin[1])
                    * Mat4::rotation_x(spin[0]);
                transform_solid(topo, s, &(*pose * place)).unwrap();
                s
            })
            .collect()
    }

    /// The fuse and then the cut, in each pose: an exact solid of sphere
    /// faces whose holes lie where they belong, valid, meshed closed, at its
    /// volume. The cut's loop on the big ball circled the hole the small
    /// ball had left there, and the hole stayed on the part outside the
    /// loop, inside the loop's own hole; the same on the small ball. The
    /// result measured 7.6e-3 too large and its mesh had 56 to 72 open
    /// edges, then the boolean was refused.
    #[test]
    fn a_ball_cut_across_a_fused_balls_joint_is_exact() {
        let truth = truth();
        let exact = BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let balls = build(&mut topo, &pose);
            let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, balls[0], balls[1], exact)
                .unwrap();
            let cut =
                boolean_with_options(&mut topo, BooleanOp::Cut, fused, balls[2], exact).unwrap();

            for face in solid_faces(&topo, cut).unwrap() {
                let surface = topo.face(face).unwrap().surface();
                assert!(
                    matches!(surface, FaceSurface::Sphere(_)),
                    "{name}: a {} face",
                    surface.type_tag()
                );
            }
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, cut, 0.01).unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
            let volume = solid_volume(&topo, cut, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 1e-5,
                "{name}: volume {volume} against {truth}"
            );
        }
    }
}

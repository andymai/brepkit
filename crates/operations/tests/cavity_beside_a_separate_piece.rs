//! Cuts of a solid in two separate pieces, a ball and a box beside it, where
//! the ball has or gets a cavity. The cavity stays an inner shell of the
//! result in every pose.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    /// The box beside the unit ball: `[1.5, 2.5] x [-0.5, 0.5] x [-0.5, 0.5]`.
    const BOX_VOLUME: f64 = 1.0;

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

    fn ball_volume(r: f64) -> f64 {
        4.0 / 3.0 * PI * r.powi(3)
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

    fn ball(topo: &mut Topology, centre: [f64; 3], r: f64, spin: [f64; 3]) -> SolidId {
        let s = make_sphere(topo, r, 32).unwrap();
        let place = Mat4::translation(centre[0], centre[1], centre[2])
            * Mat4::rotation_z(spin[2])
            * Mat4::rotation_y(spin[1])
            * Mat4::rotation_x(spin[0]);
        transform_solid(topo, s, &place).unwrap();
        s
    }

    /// The cavity off the ball's centre in a hollow ball.
    const CAVITY: ([f64; 3], f64) = ([0.5, 0.1, 0.0], 0.25);

    /// The unit ball at the origin, less [`CAVITY`] when `hollow`, fused
    /// with the box beside it into one solid of two separate pieces.
    fn ball_and_box(topo: &mut Topology, pose: &Mat4, hollow: bool) -> SolidId {
        let mut b = ball(topo, [0.0; 3], 1.0, [0.3, 0.2, 1.1]);
        transform_solid(topo, b, pose).unwrap();
        if hollow {
            let cavity = ball(topo, CAVITY.0, CAVITY.1, [1.2, 0.4, 2.0]);
            transform_solid(topo, cavity, pose).unwrap();
            b = boolean_with_options(topo, BooleanOp::Cut, b, cavity, exact()).unwrap();
        }
        let bx = make_box(topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(topo, bx, &(*pose * Mat4::translation(1.5, -0.5, -0.5))).unwrap();
        boolean_with_options(topo, BooleanOp::Fuse, b, bx, exact()).unwrap()
    }

    /// The result has `cavities` inner shells, is valid, meshes closed at
    /// two deflections, holds `truth` within 1e-5, and classifies each of
    /// `probes` (given before the pose) as stated.
    fn check(
        topo: &Topology,
        name: &str,
        solid: SolidId,
        cavities: usize,
        truth: f64,
        pose: &Mat4,
        probes: &[([f64; 3], PointClassification)],
    ) {
        assert_eq!(
            topo.solid(solid).unwrap().inner_shells().len(),
            cavities,
            "{name}: inner shells"
        );
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        for deflection in [0.01, 0.002] {
            let mesh = tessellate_solid(topo, solid, deflection).unwrap();
            assert_eq!(
                boundary_edge_count(&mesh),
                0,
                "{name}: open mesh at {deflection}"
            );
        }
        let volume = solid_volume(topo, solid, 0.0005).unwrap();
        assert!(
            ((volume - truth) / truth).abs() < 1e-5,
            "{name}: volume {volume} against {truth}"
        );
        for &(p, want) in probes {
            let at = pose.mul_point(Point3::new(p[0], p[1], p[2]));
            let got = classify_point(topo, solid, at, &ClassifyOptions::default()).unwrap();
            assert_eq!(got, want, "{name}: point {p:?}");
        }
    }

    /// A small ball inside the ball, off its centre, cut out: the engine's
    /// result keeps the small ball as a cavity. The gate that takes a result
    /// in separate pieces counted the cavity's shell as a third piece and
    /// refused the result, and the cut per piece that came after left the
    /// solid as it was.
    #[test]
    fn a_ball_cut_inside_a_ball_beside_a_box_leaves_a_cavity() {
        let (c, r) = ([0.4, 0.1, -0.2], 0.3);
        let truth = ball_volume(1.0) - ball_volume(r) + BOX_VOLUME;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let both = ball_and_box(&mut topo, &pose, false);
            let small = ball(&mut topo, c, r, [1.2, 0.4, 2.0]);
            transform_solid(&mut topo, small, &pose).unwrap();
            let cut =
                boolean_with_options(&mut topo, BooleanOp::Cut, both, small, exact()).unwrap();
            check(
                &topo,
                name,
                cut,
                1,
                truth,
                &pose,
                &[
                    (c, PointClassification::Outside),
                    ([-0.6, 0.0, 0.0], PointClassification::Inside),
                    ([2.0, 0.0, 0.0], PointClassification::Inside),
                ],
            );
        }
    }

    /// A rod along z through the centre of the hollow ball, which misses the
    /// cavity. Turned, the fuse of the hollow ball and the box goes to the
    /// engine, and the gate that takes a result in separate pieces counted
    /// the cavity's shell as a piece and refused the fuse.
    #[test]
    fn a_rod_through_a_hollow_ball_beside_a_box_keeps_its_cavity() {
        let (c, r) = CAVITY;
        let rod_r: f64 = 0.15;
        let truth =
            ball_volume(1.0) * (1.0 - rod_r * rod_r).powf(1.5) - ball_volume(r) + BOX_VOLUME;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let both = ball_and_box(&mut topo, &pose, true);
            let rod = make_cylinder(&mut topo, rod_r, 4.0).unwrap();
            let place = Mat4::translation(0.0, 0.0, -2.0) * Mat4::rotation_z(0.4);
            transform_solid(&mut topo, rod, &(pose * place)).unwrap();
            let cut = boolean_with_options(&mut topo, BooleanOp::Cut, both, rod, exact()).unwrap();
            check(
                &topo,
                name,
                cut,
                1,
                truth,
                &pose,
                &[
                    (c, PointClassification::Outside),
                    ([0.0, 0.0, -0.5], PointClassification::Outside),
                    ([-0.6, 0.0, 0.0], PointClassification::Inside),
                    ([2.0, 0.0, 0.0], PointClassification::Inside),
                ],
            );
        }
    }

    /// A small ball at the centre of the hollow ball cut out: the result
    /// has both cavities. The gate counted each cavity's shell as a piece
    /// and refused the engine's result. The cut per piece that came after
    /// built each piece from its outer shell and kept only the outer shells
    /// of its results, so both cavities were lost.
    #[test]
    fn a_ball_cut_at_the_centre_of_a_hollow_ball_beside_a_box_keeps_both_cavities() {
        let (c, r) = CAVITY;
        let r2 = 0.2;
        let truth = ball_volume(1.0) - ball_volume(r) - ball_volume(r2) + BOX_VOLUME;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let both = ball_and_box(&mut topo, &pose, true);
            let small = ball(&mut topo, [0.0; 3], r2, [0.0; 3]);
            transform_solid(&mut topo, small, &pose).unwrap();
            let cut =
                boolean_with_options(&mut topo, BooleanOp::Cut, both, small, exact()).unwrap();
            check(
                &topo,
                name,
                cut,
                2,
                truth,
                &pose,
                &[
                    (c, PointClassification::Outside),
                    ([0.0, 0.0, 0.0], PointClassification::Outside),
                    ([-0.6, 0.0, 0.0], PointClassification::Inside),
                    ([2.0, 0.0, 0.0], PointClassification::Inside),
                ],
            );
        }
    }
}

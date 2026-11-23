//! A drum with a tunnel across it, from a rod along y, and a second rod
//! along y across it above the tunnel. The second rod's footprint on the
//! drum's wall is a loop across the wall's seam on one side and a loop clear
//! of it on the other, on a wall that already has the tunnel's two holes.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

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

    /// A rod along y from -2 to 2: its axis's `(z, x)`, its radius and its
    /// turn about its axis.
    struct Rod {
        z: f64,
        x: f64,
        r: f64,
        spin: f64,
    }

    const TUNNEL: Rod = Rod {
        z: 0.955_340_685_995_104_6,
        x: 0.631_787_283_442_663_7,
        r: 0.317_809_499_103_046_3,
        spin: 1.704_558_601_996_555_7,
    };
    const ROD: Rod = Rod {
        z: 1.975_204_494_939_679_4,
        x: -0.065_813_566_239_207_42,
        r: 0.322_649_353_487_013_35,
        spin: 4.647_729_150_167_483,
    };
    const DRUM_SPIN: f64 = 4.900_426_979_214_649;

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

    /// The volume a rod shares with the drum: over the rod's section, the
    /// drum's chord along y, `2 sqrt(1 - x^2)`, with `x = x0 + r sin t` to
    /// keep the integrand smooth, by Simpson's rule.
    fn shared(rod: &Rod) -> f64 {
        let f = |t: f64| {
            let x = rod.x + rod.r * t.sin();
            4.0 * rod.r * rod.r * t.cos().powi(2) * (1.0 - x * x).max(0.0).sqrt()
        };
        let n = 20_000;
        let h = PI / f64::from(n);
        let mut s = f(-PI / 2.0) + f(PI / 2.0);
        for i in 1..n {
            s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(-PI / 2.0 + h * f64::from(i));
        }
        s * h / 3.0
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

    fn rod(topo: &mut Topology, rod: &Rod, pose: &Mat4) -> SolidId {
        let s = make_cylinder(topo, rod.r, 4.0).unwrap();
        let along_z = Mat4::translation(rod.z, rod.x, -2.0) * Mat4::rotation_z(rod.spin);
        // (z, x, y) of the rod's frame to (x, y, z).
        let along_y = Mat4::rotation_x(-PI / 2.0) * Mat4::rotation_z(-PI / 2.0) * along_z;
        transform_solid(topo, s, &(*pose * along_y)).unwrap();
        s
    }

    /// The drum (radius 1, from z = 0 to 4) less the tunnel, and the rod.
    fn build(topo: &mut Topology, pose: &Mat4) -> (SolidId, SolidId) {
        let drum = make_cylinder(topo, 1.0, 4.0).unwrap();
        transform_solid(topo, drum, &(*pose * Mat4::rotation_z(DRUM_SPIN))).unwrap();
        let tunnel = rod(topo, &TUNNEL, pose);
        let tunnelled = boolean_with_options(topo, BooleanOp::Cut, drum, tunnel, exact()).unwrap();
        (tunnelled, rod(topo, &ROD, pose))
    }

    /// Points before the pose: in the rod on either side of the drum's axis,
    /// in the rod out of the drum, in the drum away from both rods, and in
    /// the tunnel.
    fn probes() -> [Point3; 5] {
        [
            Point3::new(ROD.x, 0.5, ROD.z),
            Point3::new(ROD.x, -0.5, ROD.z),
            Point3::new(ROD.x, 1.5, ROD.z),
            Point3::new(0.0, 0.0, 3.5),
            Point3::new(TUNNEL.x, 0.0, TUNNEL.z),
        ]
    }

    fn check(op: BooleanOp, truth: f64, want: [PointClassification; 5]) {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (drum, rod) = build(&mut topo, &pose);
            let result = boolean_with_options(&mut topo, op, drum, rod, exact())
                .unwrap_or_else(|e| panic!("{name}: {op:?} {e}"));
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name}: {op:?} {:?}", report.issues);
            for deflection in [0.01, 0.002] {
                let mesh = tessellate_solid(&topo, result, deflection).unwrap();
                assert_eq!(
                    boundary_edge_count(&mesh),
                    0,
                    "{name}: {op:?} open mesh at {deflection}"
                );
            }
            let volume = solid_volume(&topo, result, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 1e-5,
                "{name}: {op:?} volume {volume} against {truth}"
            );
            for (p, want) in probes().into_iter().zip(want) {
                let at = pose.mul_point(p);
                let got = classify_point(&topo, result, at, &ClassifyOptions::default()).unwrap();
                assert_eq!(got, want, "{name}: {op:?} at {p:?}");
            }
        }
    }

    #[test]
    fn intersect_keeps_the_rod_in_the_drum() {
        use PointClassification::{Inside, Outside};
        check(
            BooleanOp::Intersect,
            shared(&ROD),
            [Inside, Inside, Outside, Outside, Outside],
        );
    }

    #[test]
    fn fuse_adds_the_rod() {
        use PointClassification::{Inside, Outside};
        let truth = 4.0 * PI - shared(&TUNNEL) + 4.0 * PI * ROD.r * ROD.r - shared(&ROD);
        check(
            BooleanOp::Fuse,
            truth,
            [Inside, Inside, Inside, Inside, Outside],
        );
    }

    #[test]
    fn cut_bores_the_rod_out() {
        use PointClassification::{Inside, Outside};
        let truth = 4.0 * PI - shared(&TUNNEL) - shared(&ROD);
        check(
            BooleanOp::Cut,
            truth,
            [Outside, Outside, Outside, Inside, Outside],
        );
    }
}

//! A rod along y clipped to a drum, so that each end of its wall is one
//! closed curve on the drum's wall, then fused with a rod along x that
//! crosses it, then cut down to a rod along z.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    /// The drum: radius 1 along z through the origin, from 0 to 4.
    const DRUM_SPIN: f64 = 3.696_424_691_539_894;
    /// The rod along y, through `(z, x)`, from -2 to 2.
    const Y_ROD_AT: (f64, f64) = (1.052_181_116_004_248_7, 0.159_076_885_934_773_1);
    const Y_ROD_RADIUS: f64 = 0.365_526_227_064_571_3;
    const Y_ROD_SPIN: f64 = 2.834_264_982_558_358_3;
    /// The rod along x, through `(y, z)`, from -2 to 2.
    const X_ROD_AT: (f64, f64) = (0.102_455_396_630_358_08, 0.762_986_182_608_410_1);
    const X_ROD_RADIUS: f64 = 0.427_801_976_685_517_34;
    const X_ROD_SPIN: f64 = 2.013_860_515_190_202_7;
    /// The rod along z, through `(x, y)`, from `Z_ROD_SPAN.0` to `Z_ROD_SPAN.1`.
    const Z_ROD_AT: (f64, f64) = (-0.006_831_574_043_620_225_5, -0.724_335_322_575_022_5);
    const Z_ROD_RADIUS: f64 = 0.274_937_909_837_767_95;
    const Z_ROD_SPAN: (f64, f64) = (0.711_163_794_593_038_9, 4.647_069_702_395_88);
    const Z_ROD_SPIN: f64 = 4.765_762_149_560_931;

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
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

    /// A cylinder of radius `r` along z through `(p, q)` from `t0` to `t1`,
    /// turned by `spin` about its axis, then laid along `axis`.
    fn rod(
        topo: &mut Topology,
        axis: usize,
        (p, q): (f64, f64),
        r: f64,
        (t0, t1): (f64, f64),
        spin: f64,
    ) -> SolidId {
        let s = make_cylinder(topo, r, t1 - t0).unwrap();
        let along_z = Mat4::translation(p, q, t0) * Mat4::rotation_z(spin);
        let place = match axis {
            0 => Mat4::rotation_z(PI / 2.0) * Mat4::rotation_x(PI / 2.0) * along_z,
            1 => Mat4::rotation_x(-PI / 2.0) * Mat4::rotation_z(-PI / 2.0) * along_z,
            _ => along_z,
        };
        transform_solid(topo, s, &place).unwrap();
        s
    }

    fn build(topo: &mut Topology, pose: &Mat4) -> [SolidId; 4] {
        let solids = [
            rod(topo, 2, (0.0, 0.0), 1.0, (0.0, 4.0), DRUM_SPIN),
            rod(topo, 1, Y_ROD_AT, Y_ROD_RADIUS, (-2.0, 2.0), Y_ROD_SPIN),
            rod(topo, 0, X_ROD_AT, X_ROD_RADIUS, (-2.0, 2.0), X_ROD_SPIN),
            rod(topo, 2, Z_ROD_AT, Z_ROD_RADIUS, Z_ROD_SPAN, Z_ROD_SPIN),
        ];
        for s in solids {
            transform_solid(topo, s, pose).unwrap();
        }
        solids
    }

    /// The interval in y of the drum and the three rods, before the pose,
    /// on the line through `(x, z)`.
    fn spans(x: f64, z: f64) -> [Option<(f64, f64)>; 4] {
        let drum = {
            let q = 1.0 - x * x;
            (z > 0.0 && z < 4.0 && q > 0.0).then(|| (-q.sqrt(), q.sqrt()))
        };
        let y_rod = ((z - Y_ROD_AT.0).powi(2) + (x - Y_ROD_AT.1).powi(2) < Y_ROD_RADIUS.powi(2))
            .then_some((-2.0, 2.0));
        let x_rod = {
            let q = X_ROD_RADIUS.powi(2) - (z - X_ROD_AT.1).powi(2);
            (x > -2.0 && x < 2.0 && q > 0.0).then(|| (X_ROD_AT.0 - q.sqrt(), X_ROD_AT.0 + q.sqrt()))
        };
        let z_rod = {
            let q = Z_ROD_RADIUS.powi(2) - (x - Z_ROD_AT.0).powi(2);
            (z > Z_ROD_SPAN.0 && z < Z_ROD_SPAN.1 && q > 0.0)
                .then(|| (Z_ROD_AT.1 - q.sqrt(), Z_ROD_AT.1 + q.sqrt()))
        };
        [drum, y_rod, x_rod, z_rod]
    }

    /// Whether `y` on the line through `(x, z)` lies in the drum's part of
    /// the rod along y, fused with the rod along x, and, with `cut_down`,
    /// cut down to the rod along z.
    fn holds(s: &[Option<(f64, f64)>; 4], y: f64, cut_down: bool) -> bool {
        let inside = |k: usize| s[k].is_some_and(|(a, b)| y > a && y < b);
        let fused = (inside(0) && inside(1)) || inside(2);
        fused && (!cut_down || inside(3))
    }

    /// The volume by the midpoint rule over cells in x and z across a box
    /// round the result, with the exact intervals in y on each line. The
    /// rods' flat ends across x and z lie on the cells' borders.
    fn truth(cut_down: bool) -> f64 {
        let cells = |lo: f64, hi: f64, borders: &[f64], n: usize| -> Vec<(f64, f64)> {
            let mut br = vec![lo, hi];
            br.extend(borders.iter().copied().filter(|&b| b > lo && b < hi));
            br.sort_by(f64::total_cmp);
            let mut out = Vec::new();
            for w in br.windows(2) {
                let k = (((w[1] - w[0]) / (hi - lo) * n as f64).round() as usize).max(1);
                let h = (w[1] - w[0]) / k as f64;
                out.extend((0..k).map(|i| (w[0] + (i as f64 + 0.5) * h, h)));
            }
            out
        };
        let ((x0, x1), (z0, z1)) = if cut_down {
            (
                (Z_ROD_AT.0 - Z_ROD_RADIUS, Z_ROD_AT.0 + Z_ROD_RADIUS),
                (Z_ROD_SPAN.0, Y_ROD_AT.0 + Y_ROD_RADIUS),
            )
        } else {
            (
                (-2.0, 2.0),
                (X_ROD_AT.1 - X_ROD_RADIUS, Y_ROD_AT.0 + Y_ROD_RADIUS),
            )
        };
        let xs = cells(x0, x1, &[], 2000);
        let zs = cells(z0, z1, &[], 2000);
        let mut total = 0.0;
        let mut ends = Vec::with_capacity(8);
        for &(z, dz) in &zs {
            for &(x, dx) in &xs {
                let s = spans(x, z);
                ends.clear();
                for &(a, b) in s.iter().flatten() {
                    ends.push(a);
                    ends.push(b);
                }
                ends.sort_by(f64::total_cmp);
                for w in ends.windows(2) {
                    if w[1] > w[0] && holds(&s, 0.5 * (w[0] + w[1]), cut_down) {
                        total += (w[1] - w[0]) * dx * dz;
                    }
                }
            }
        }
        total
    }

    fn check(topo: &Topology, solid: SolidId, truth: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        for deflection in [0.01, 0.002] {
            let mesh = tessellate_solid(topo, solid, deflection).unwrap();
            assert_eq!(
                boundary_edge_count(&mesh),
                0,
                "{what}: open mesh at {deflection}"
            );
        }
        let volume = solid_volume(topo, solid, 0.0005).unwrap();
        assert!(
            ((volume - truth) / truth).abs() < 2e-5,
            "{what}: volume {volume} against {truth}"
        );
    }

    /// Each end of the clipped rod's wall is one closed curve whose pcurve
    /// lies a period from the ends its wire gives it. The face trace read
    /// the top end from its chord, which a period folds to nothing, and
    /// took the end's twin into the wall's region. The fuse came back
    /// invalid, 5.6% to 6.7% short with 106 open mesh edges, and the cut
    /// down to the rod along z hid it. The fuse and the cut down now come
    /// back valid, mesh closed, hold the volume of an integral over the
    /// four solids, and classify points as they do.
    #[test]
    fn a_clipped_rod_takes_a_crossing_rod_exactly_in_every_pose() {
        let (fused_truth, cut_truth) = (truth(false), truth(true));
        let probes = [
            Point3::new(1.5, 0.1, 0.76),
            Point3::new(0.159, 0.8, 1.052),
            Point3::new(0.159, -0.9, 1.052),
            Point3::new(0.6, 0.6, 1.3),
            Point3::new(-0.5, 0.1, 1.2),
            Point3::new(0.0, -0.72, 0.9),
            Point3::new(0.0, 0.0, 0.2),
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let [drum, y_rod, x_rod, z_rod] = build(&mut topo, &pose);
            let clipped =
                boolean_with_options(&mut topo, BooleanOp::Intersect, drum, y_rod, exact())
                    .unwrap();
            let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, clipped, x_rod, exact())
                .map_err(|e| format!("{name}: fuse: {e}"))
                .unwrap();
            check(&topo, fused, fused_truth, &format!("{name}: fuse"));
            for p in probes {
                let expected = if holds(&spans(p.x(), p.z()), p.y(), false) {
                    PointClassification::Inside
                } else {
                    PointClassification::Outside
                };
                let got = classify_point(&topo, fused, pose.mul_point(p), 0.01, 1e-7).unwrap();
                assert_eq!(got, expected, "{name}: {p:?}");
            }
            let cut = boolean_with_options(&mut topo, BooleanOp::Intersect, fused, z_rod, exact())
                .map_err(|e| format!("{name}: cut down: {e}"))
                .unwrap();
            check(&topo, cut, cut_truth, &format!("{name}: cut down"));
        }
    }
}

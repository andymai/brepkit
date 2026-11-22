//! A ball with a small ball fused on near its top, a rod along z cut up
//! through it, and a thin slab fused across the rod's hole and out through
//! the ball's lower side. The slab's window on the hole's wall holds the
//! wall's seam, and the wall's top rim is one closed curve.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    const BALL_SPIN: [f64; 3] = [
        3.325_665_722_837_124,
        2.516_757_695_461_895,
        4.284_855_211_168_162,
    ];
    const BUMP_CENTRE: [f64; 3] = [
        -0.165_825_142_038_512_84,
        -0.161_845_503_033_171_02,
        0.732_404_014_823_267_3,
    ];
    const BUMP_RADIUS: f64 = 0.241_369_621_185_969_27;
    const BUMP_SPIN: [f64; 3] = [
        -0.178_091_002_994_844_88,
        -0.113_974_007_953_889_47,
        4.897_152_891_298_79,
    ];
    /// The rod runs along z through `(x, y)`, from `ROD_FROM` to 2.
    const ROD_AT: (f64, f64) = (0.029_355_704_820_890_56, 0.098_204_048_902_626_68);
    const ROD_RADIUS: f64 = 0.260_195_695_106_176_57;
    const ROD_FROM: f64 = -0.265_690_976_734_235_25;
    const ROD_SPIN: f64 = 6.081_966_453_157_022;
    const SLAB_LO: [f64; 3] = [
        -0.104_546_067_327_844_44,
        0.026_889_487_238_644_26,
        -0.921_364_354_295_329_7,
    ];
    const SLAB_HI: [f64; 3] = [
        0.707_864_877_447_787_9,
        0.127_331_908_776_046_78,
        0.205_835_350_870_651_97,
    ];

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

    fn spun(s: [f64; 3]) -> Mat4 {
        Mat4::rotation_z(s[2]) * Mat4::rotation_y(s[1]) * Mat4::rotation_x(s[0])
    }

    fn build(topo: &mut Topology, pose: &Mat4) -> [SolidId; 4] {
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        transform_solid(topo, ball, &spun(BALL_SPIN)).unwrap();
        let bump = make_sphere(topo, BUMP_RADIUS, 32).unwrap();
        let [cx, cy, cz] = BUMP_CENTRE;
        transform_solid(
            topo,
            bump,
            &(Mat4::translation(cx, cy, cz) * spun(BUMP_SPIN)),
        )
        .unwrap();
        let rod = make_cylinder(topo, ROD_RADIUS, 2.0 - ROD_FROM).unwrap();
        let place = Mat4::translation(ROD_AT.0, ROD_AT.1, ROD_FROM) * Mat4::rotation_z(ROD_SPIN);
        transform_solid(topo, rod, &place).unwrap();
        let [lx, ly, lz] = SLAB_LO;
        let [hx, hy, hz] = SLAB_HI;
        let slab = make_box(topo, hx - lx, hy - ly, hz - lz).unwrap();
        transform_solid(topo, slab, &Mat4::translation(lx, ly, lz)).unwrap();
        for s in [ball, bump, rod, slab] {
            transform_solid(topo, s, pose).unwrap();
        }
        [ball, bump, rod, slab]
    }

    /// The interval in y of the ball, the small ball, the rod and the slab,
    /// before the pose, on the line through `(x, z)`.
    fn spans(x: f64, z: f64) -> [Option<(f64, f64)>; 4] {
        let ball = {
            let q = 1.0 - x * x - z * z;
            (q > 0.0).then(|| (-q.sqrt(), q.sqrt()))
        };
        let bump = {
            let [cx, cy, cz] = BUMP_CENTRE;
            let q = BUMP_RADIUS * BUMP_RADIUS - (x - cx).powi(2) - (z - cz).powi(2);
            (q > 0.0).then(|| (cy - q.sqrt(), cy + q.sqrt()))
        };
        let rod = {
            let q = ROD_RADIUS * ROD_RADIUS - (x - ROD_AT.0).powi(2);
            (z > ROD_FROM && z < 2.0 && q > 0.0).then(|| (ROD_AT.1 - q.sqrt(), ROD_AT.1 + q.sqrt()))
        };
        let slab = (x > SLAB_LO[0] && x < SLAB_HI[0] && z > SLAB_LO[2] && z < SLAB_HI[2])
            .then_some((SLAB_LO[1], SLAB_HI[1]));
        [ball, bump, rod, slab]
    }

    /// Whether `y` on the line through `(x, z)` lies in the ball fused with
    /// the small ball, less the rod, fused with the slab.
    fn holds(s: &[Option<(f64, f64)>; 4], y: f64) -> bool {
        let inside = |k: usize| s[k].is_some_and(|(a, b)| y > a && y < b);
        ((inside(0) || inside(1)) && !inside(2)) || inside(3)
    }

    /// The volume of the result, by the midpoint rule over cells in x and z
    /// with the exact intervals in y on each line. The flat faces across x
    /// and z lie on the cells' borders.
    fn truth() -> f64 {
        let cells = |borders: &[f64], n: usize| -> Vec<(f64, f64)> {
            let mut br = vec![-1.0, 1.0];
            br.extend(borders.iter().copied());
            br.sort_by(f64::total_cmp);
            let mut out = Vec::new();
            for w in br.windows(2) {
                let k = (((w[1] - w[0]) / 2.0 * n as f64).round() as usize).max(1);
                let h = (w[1] - w[0]) / k as f64;
                out.extend((0..k).map(|i| (w[0] + (i as f64 + 0.5) * h, h)));
            }
            out
        };
        let xs = cells(&[SLAB_LO[0], SLAB_HI[0]], 2000);
        let zs = cells(&[SLAB_LO[2], SLAB_HI[2], ROD_FROM], 2000);
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
                    if w[1] > w[0] && holds(&s, 0.5 * (w[0] + w[1])) {
                        total += (w[1] - w[0]) * dx * dz;
                    }
                }
            }
        }
        total
    }

    /// Fuse, cut, fuse. The slab's fuse came back invalid and 9.7e-4 short
    /// in three poses, with 24 to 26 open mesh edges, and the engine refused
    /// it in the other two. On the hole's wall, the twin of the top rim, one
    /// closed curve, made a loop of its own that the face trace kept as a
    /// region, and the wall's remainder took it as a hole. Each result is
    /// now valid, meshes closed, holds the volume of an integral over the
    /// four solids, and classifies points as they do.
    #[test]
    fn a_slab_fused_across_a_rod_hole_in_a_ball_is_exact_in_every_pose() {
        let truth = truth();
        let probes = [
            Point3::new(0.0, 0.08, 0.0),
            Point3::new(0.0, 0.25, 0.0),
            Point3::new(0.0, 0.08, 0.5),
            Point3::new(0.6, 0.08, -0.85),
            Point3::new(0.75, 0.08, -0.85),
            Point3::new(-0.2167, -0.2115, 0.9572),
            Point3::new(0.6, 0.08, 0.5),
            Point3::new(0.0, 0.08, -0.5),
            Point3::new(0.0, 0.08, 0.98),
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let [ball, bump, rod, slab] = build(&mut topo, &pose);
            let mut result = ball;
            for (op, tool) in [
                (BooleanOp::Fuse, bump),
                (BooleanOp::Cut, rod),
                (BooleanOp::Fuse, slab),
            ] {
                result = boolean_with_options(&mut topo, op, result, tool, exact())
                    .map_err(|e| format!("{name}: {op:?}: {e}"))
                    .unwrap();
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{name}: {op:?}: {:?}", report.issues);
            }

            let faces = solid_faces(&topo, result).unwrap();
            let count = |f: fn(&FaceSurface) -> bool| {
                faces
                    .iter()
                    .filter(|&&id| f(topo.face(id).unwrap().surface()))
                    .count()
            };
            assert!(
                count(|s| matches!(s, FaceSurface::Sphere(_))) >= 2
                    && count(|s| matches!(s, FaceSurface::Cylinder(_))) >= 1,
                "{name}: the balls and the rod's hole keep their surfaces"
            );
            for deflection in [0.01, 0.002] {
                let mesh = tessellate_solid(&topo, result, deflection).unwrap();
                assert_eq!(
                    boundary_edge_count(&mesh),
                    0,
                    "{name}: open mesh at {deflection}"
                );
            }
            let volume = solid_volume(&topo, result, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 2e-5,
                "{name}: volume {volume} against {truth}"
            );
            for p in probes {
                let expected = if holds(&spans(p.x(), p.z()), p.y()) {
                    PointClassification::Inside
                } else {
                    PointClassification::Outside
                };
                let got = classify_point(&topo, result, pose.mul_point(p), 0.01, 1e-7).unwrap();
                assert_eq!(got, expected, "{name}: {p:?}");
            }
        }
    }
}

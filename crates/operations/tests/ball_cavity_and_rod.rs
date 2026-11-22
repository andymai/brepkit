//! A ball with a cavity off its centre, fused with a rod through it below
//! the cavity: the cavity stays an inner shell, so the ball keeps its two
//! halves and the result meshes closed in every pose.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    const CAVITY_CENTRE: [f64; 3] = [
        -0.056_582_760_471_055_45,
        -0.013_934_050_800_119_542,
        0.694_305_750_023_576_5,
    ];
    const CAVITY_RADIUS: f64 = 0.261_141_980_270_943_4;
    const CAVITY_SPIN: [f64; 3] = [
        2.545_251_168_334_399_2,
        2.638_460_818_868_343_6,
        0.841_436_693_133_441_6,
    ];
    /// The rod runs along x through `(y, z)`.
    const ROD_AT: (f64, f64) = (0.030_134_429_278_411_312, -0.139_861_523_089_576_94);
    const ROD_RADIUS: f64 = 0.415_149_829_202_368_9;
    const ROD_SPIN: f64 = 2.971_097_052_031_638_4;

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

    /// The area two discs of radii `a` and `b`, `d` apart, share.
    fn lens(a: f64, b: f64, d: f64) -> f64 {
        if d >= a + b {
            return 0.0;
        }
        if d <= (a - b).abs() {
            return PI * a.min(b).powi(2);
        }
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

    /// The unit ball's volume less the cavity, with the rod's added: the
    /// rod's part in the ball is the discs it shares with each section of
    /// the ball across x, summed by Simpson's rule.
    fn truth() -> f64 {
        let d = ROD_AT.0.hypot(ROD_AT.1);
        let n = 20_000;
        let h = 2.0 / f64::from(n);
        let f = |x: f64| lens((1.0 - x * x).max(0.0).sqrt(), ROD_RADIUS, d);
        let mut shared = f(-1.0) + f(1.0);
        for i in 1..n {
            shared += if i % 2 == 1 { 4.0 } else { 2.0 } * f(-1.0 + h * f64::from(i));
        }
        shared *= h / 3.0;
        let ball = |r: f64| 4.0 / 3.0 * PI * r.powi(3);
        ball(1.0) - ball(CAVITY_RADIUS) + PI * ROD_RADIUS * ROD_RADIUS * 4.0 - shared
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

    fn build(topo: &mut Topology, pose: &Mat4) -> (SolidId, SolidId, SolidId) {
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let cavity = make_sphere(topo, CAVITY_RADIUS, 32).unwrap();
        let [cx, cy, cz] = CAVITY_CENTRE;
        let [sx, sy, sz] = CAVITY_SPIN;
        let place = Mat4::translation(cx, cy, cz)
            * Mat4::rotation_z(sz)
            * Mat4::rotation_y(sy)
            * Mat4::rotation_x(sx);
        transform_solid(topo, cavity, &place).unwrap();
        let rod = make_cylinder(topo, ROD_RADIUS, 4.0).unwrap();
        let along_z = Mat4::translation(ROD_AT.0, ROD_AT.1, -2.0) * Mat4::rotation_z(ROD_SPIN);
        let along_x = Mat4::rotation_z(PI / 2.0) * Mat4::rotation_x(PI / 2.0) * along_z;
        transform_solid(topo, rod, &along_x).unwrap();
        for s in [ball, cavity, rod] {
            transform_solid(topo, s, pose).unwrap();
        }
        (ball, cavity, rod)
    }

    /// The fuse keeps the cavity as an inner shell of two sphere faces and
    /// the ball's two halves, each holding one end of the rod's hole. It is
    /// valid, meshes with no open edge, and holds the volume of the ball
    /// less the cavity with the rod added. The cavity's shell had read as a
    /// lump of its own: the corners of each of its halves centre on the
    /// cavity's centre, where no normal can be read, so its corner fan held
    /// twice a cone's volume in place of none. The ball's halves were then
    /// merged into one face, which meshed open.
    #[test]
    fn a_ball_with_a_cavity_takes_a_rod_below_it_exactly() {
        let truth = truth();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (ball, cavity, rod) = build(&mut topo, &pose);
            let hollow =
                boolean_with_options(&mut topo, BooleanOp::Cut, ball, cavity, exact()).unwrap();
            let fused =
                boolean_with_options(&mut topo, BooleanOp::Fuse, hollow, rod, exact()).unwrap();

            let solid = topo.solid(fused).unwrap();
            assert_eq!(
                solid.inner_shells().len(),
                1,
                "{name}: the cavity is not an inner shell"
            );
            let spheres = solid_faces(&topo, fused)
                .unwrap()
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                .count();
            assert_eq!(spheres, 4, "{name}: sphere faces");

            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            for deflection in [0.01, 0.002] {
                let mesh = tessellate_solid(&topo, fused, deflection).unwrap();
                assert_eq!(
                    boundary_edge_count(&mesh),
                    0,
                    "{name}: open mesh at {deflection}"
                );
            }
            let volume = solid_volume(&topo, fused, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 1e-5,
                "{name}: volume {volume} against {truth}"
            );
        }
    }
}

//! `unify_faces` on a ball fused with a rod through it, off its centre. The
//! ball's two halves share the edges between the rod's two mouths, but one
//! face of the whole ball would hold both poles of its sphere with a loop at
//! each mouth, which the sphere's chart cannot read: its mesh came out open,
//! and a later cut of its top changed nothing. The halves stay apart.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::heal::unify_faces;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    /// The rod runs along x through `(y, z)`.
    const ROD_AT: (f64, f64) = (0.030_134_429_278_411_312, -0.139_861_523_089_576_94);
    const ROD_RADIUS: f64 = 0.415_149_829_202_368_9;
    const ROD_SPIN: f64 = 2.971_097_052_031_638_4;
    /// The later cut takes the ball's cap above this height, clear of the rod.
    const CAP_FROM: f64 = 0.6;

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

    /// The unit ball's volume with the rod's added: the rod's part in the
    /// ball is the discs it shares with each section of the ball across x,
    /// summed by Simpson's rule.
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
        4.0 / 3.0 * PI + PI * ROD_RADIUS * ROD_RADIUS * 4.0 - shared
    }

    /// The unit ball's cap above `CAP_FROM`.
    fn cap() -> f64 {
        let h = 1.0 - CAP_FROM;
        PI * h * h * (3.0 - h) / 3.0
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

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let measured = solid_volume(topo, solid, 0.0005).unwrap();
        assert!(
            (measured - volume).abs() < 1e-6 * volume,
            "{what}: volume {measured}, expected {volume}"
        );
        for deflection in [0.01, 0.05] {
            let mesh = tessellate_solid(topo, solid, deflection).unwrap();
            assert_eq!(
                boundary_edge_count(&mesh),
                0,
                "{what}: open mesh at {deflection}"
            );
        }
    }

    #[test]
    fn unify_keeps_a_bored_balls_halves_apart() {
        let (truth, cap) = (truth(), cap());
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, 1.0, 32).unwrap();
            let rod = make_cylinder(&mut topo, ROD_RADIUS, 4.0).unwrap();
            let along_z = Mat4::translation(ROD_AT.0, ROD_AT.1, -2.0) * Mat4::rotation_z(ROD_SPIN);
            let along_x = Mat4::rotation_z(PI / 2.0) * Mat4::rotation_x(PI / 2.0) * along_z;
            transform_solid(&mut topo, rod, &along_x).unwrap();
            let top = make_box(&mut topo, 3.0, 3.0, 1.0).unwrap();
            transform_solid(&mut topo, top, &Mat4::translation(-1.5, -1.5, CAP_FROM)).unwrap();
            for s in [ball, rod, top] {
                transform_solid(&mut topo, s, &pose).unwrap();
            }
            let fused =
                boolean_with_options(&mut topo, BooleanOp::Fuse, ball, rod, exact()).unwrap();
            unify_faces(&mut topo, fused).unwrap();

            let spheres = solid_faces(&topo, fused)
                .unwrap()
                .into_iter()
                .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                .count();
            assert_eq!(spheres, 2, "{name}: sphere faces");
            assert_sound(&topo, fused, truth, name);

            let capped =
                boolean_with_options(&mut topo, BooleanOp::Cut, fused, top, exact()).unwrap();
            assert_sound(&topo, capped, truth - cap, &format!("{name}, capped"));
        }
    }
}

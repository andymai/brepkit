//! A ball whose centre lies on a rod's wall and whose top touches the rod's
//! top rim at one point, after a slot cut the wall's seam away. The ball's
//! face then has one loop with no seam, through both of its poles or round
//! neither, and the wall's hole meets the rim at the point.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{
        TriangleMesh, boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;

    const BALL: f64 = 0.4;

    /// The area two discs of radii `r1` and `r2` with centres `d` apart
    /// share.
    fn lens(r1: f64, r2: f64, d: f64) -> f64 {
        if d >= r1 + r2 {
            return 0.0;
        }
        if d <= (r1 - r2).abs() {
            return PI * r1.min(r2).powi(2);
        }
        let a1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).acos();
        let a2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).acos();
        r1 * r1 * (a1 - a1.sin() * a1.cos()) + r2 * r2 * (a2 - a2.sin() * a2.cos())
    }

    /// The rod of radius 1 and height 4 less the slot `0.8 < x`, `|y| < 0.2`,
    /// and the part of the ball within the rod, by slices along z.
    fn volumes() -> (f64, f64) {
        let half = 0.2_f64;
        let slot = half * (1.0 - half * half).sqrt() + half.asin() - 1.6 * half;
        let rod = 4.0 * (PI - slot);
        let n = 40_000;
        let mut within = 0.0;
        for i in 0..n {
            let z = (2.0 * f64::from(i) + 1.0) / (2.0 * f64::from(n)) * 2.0 * BALL - BALL;
            let r = (BALL * BALL - z * z).max(0.0).sqrt();
            within += lens(1.0, r, 1.0) * 2.0 * BALL / f64::from(n);
        }
        let ball = 4.0 / 3.0 * PI * BALL.powi(3);
        (rod - within + ball, rod - within)
    }

    fn mesh_volume(mesh: &TriangleMesh) -> f64 {
        let origin = Point3::new(0.0, 0.0, 0.0);
        mesh.indices
            .chunks_exact(3)
            .map(|t| {
                let [a, b, c] = [0, 1, 2].map(|k| mesh.positions[t[k] as usize] - origin);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// The slotted rod and the ball turned `spin` about x through its centre,
    /// then `pose` on both, combined by `op`.
    fn check(op: BooleanOp, spin: f64, pose: &Mat4, name: &str) {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
        let slot = make_box(&mut topo, 1.2, 0.4, 6.0).unwrap();
        transform_solid(&mut topo, slot, &Mat4::translation(0.8, -0.2, -1.0)).unwrap();
        let ball = make_sphere(&mut topo, BALL, 32).unwrap();
        transform_solid(
            &mut topo,
            ball,
            &(Mat4::translation(-1.0, 0.0, 4.0 - BALL) * Mat4::rotation_x(spin)),
        )
        .unwrap();
        for s in [rod, slot, ball] {
            transform_solid(&mut topo, s, pose).unwrap();
        }
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let slotted =
            boolean::boolean_with_options(&mut topo, BooleanOp::Cut, rod, slot, opts).unwrap();
        let result = boolean::boolean_with_options(&mut topo, op, slotted, ball, opts)
            .map_err(|e| format!("{name}: {e}"))
            .unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let spheres = solid_faces(&topo, result)
            .unwrap()
            .into_iter()
            .filter(|&f| topo.face(f).unwrap().surface().type_tag() == "sphere")
            .count();
        assert!(spheres > 0, "{name}: no sphere face");
        let (fused, cut) = volumes();
        let expected = if op == BooleanOp::Fuse { fused } else { cut };
        let exact = solid_volume(&topo, result, 0.0005).unwrap();
        assert!(
            (exact - expected).abs() < 1e-6 * expected,
            "{name}: volume {exact} against {expected}"
        );
        let mesh = tessellate_solid(&topo, result, 0.005).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
        assert_eq!(
            non_manifold_edge_count(&mesh),
            0,
            "{name}: non-manifold mesh"
        );
        let meshed = mesh_volume(&mesh);
        assert!(
            (meshed - expected).abs() < 5e-3 * expected,
            "{name}: mesh volume {meshed} against {expected}"
        );
    }

    #[test]
    fn a_ball_touching_a_seamless_walls_rim_fuses_and_cuts_exactly() {
        let turned = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
        for (pose_name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", turned),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0) * turned),
            ("turned 2 about z", Mat4::rotation_z(2.0)),
        ] {
            // Unturned the ball's loop runs through both of its poles; turned
            // 0.3 or a quarter turn it runs round neither, and the face holds
            // both.
            for spin in [0.0, 0.3, FRAC_PI_2] {
                for op in [BooleanOp::Fuse, BooleanOp::Cut] {
                    check(
                        op,
                        spin,
                        &pose,
                        &format!("{pose_name}, spin {spin}, {op:?}"),
                    );
                }
            }
        }
    }
}

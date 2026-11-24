//! A ball less the cap above `z = 0.95`, and a box through its south pole.
//! The ball's face that the box cuts holds a pole of every axis, since the
//! removed cap is too small to hold both poles of one. Turned a quarter turn
//! about `y`, the cap and the box lie round the ball's own equator, and the
//! face holds both poles of the ball's own chart.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_sphere};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const CAP_PLANE: f64 = 0.95;
    const HALF_WIDTH: f64 = 0.1;
    const BOX_Z: (f64, f64) = (-1.2, -0.9);

    /// The unit ball's volume between the box's top and the ball's bottom,
    /// over the box's square: the square lies inside the disc where the ball
    /// reaches below the top.
    fn ball_in_box() -> f64 {
        let gauss = brepkit_math::quadrature::gauss_legendre_points(20);
        let mut total = 0.0;
        for gx in gauss {
            for gy in gauss {
                let (x, y) = (HALF_WIDTH * gx.x, HALF_WIDTH * gy.x);
                let depth = (1.0 - x * x - y * y).sqrt() + BOX_Z.1;
                total += HALF_WIDTH * HALF_WIDTH * gx.w * gy.w * depth;
            }
        }
        total
    }

    fn capped_ball() -> f64 {
        let h = 1.0 - CAP_PLANE;
        4.0 / 3.0 * std::f64::consts::PI - std::f64::consts::PI * h * h * (3.0 - h) / 3.0
    }

    fn expected(op: BooleanOp) -> f64 {
        let cube = (2.0 * HALF_WIDTH).powi(2) * (BOX_Z.1 - BOX_Z.0);
        match op {
            BooleanOp::Cut => capped_ball() - ball_in_box(),
            BooleanOp::Fuse => capped_ball() + cube - ball_in_box(),
            BooleanOp::Intersect => ball_in_box(),
        }
    }

    /// Points (in the upright pose) inside and outside each result.
    fn probes(op: BooleanOp) -> (Vec<[f64; 3]>, Vec<[f64; 3]>) {
        let in_box = [0.0, 0.0, -0.95];
        let box_tip = [0.05, -0.05, -1.1];
        let ball = [0.3, 0.2, -0.5];
        let near_pole = [0.15, 0.0, -0.95];
        let cap = [0.0, 0.1, 0.97];
        let below = [0.0, 0.0, -1.25];
        match op {
            BooleanOp::Cut => (
                vec![ball, near_pole, [0.0, 0.0, -0.85]],
                vec![in_box, box_tip, cap, below],
            ),
            BooleanOp::Fuse => (
                vec![ball, near_pole, in_box, box_tip],
                vec![cap, below, [0.15, 0.0, -1.1]],
            ),
            BooleanOp::Intersect => (
                vec![in_box, [-0.08, 0.08, -0.97]],
                vec![ball, near_pole, box_tip, cap, below],
            ),
        }
    }

    /// The capped ball and the box, the cap and the box turned by `around`
    /// about the ball's centre, then all of it placed by `pose`.
    fn scene(topo: &mut Topology, around: &Mat4, pose: &Mat4) -> (SolidId, SolidId) {
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let lid = make_box(topo, 4.0, 4.0, 2.0).unwrap();
        transform_solid(topo, lid, &Mat4::translation(-2.0, -2.0, CAP_PLANE)).unwrap();
        let w = 2.0 * HALF_WIDTH;
        let block = make_box(topo, w, w, BOX_Z.1 - BOX_Z.0).unwrap();
        transform_solid(
            topo,
            block,
            &Mat4::translation(-HALF_WIDTH, -HALF_WIDTH, BOX_Z.0),
        )
        .unwrap();
        for s in [lid, block] {
            transform_solid(topo, s, around).unwrap();
        }
        for s in [ball, lid, block] {
            transform_solid(topo, s, pose).unwrap();
        }
        let capped = boolean_with_options(topo, BooleanOp::Cut, ball, lid, opts).unwrap();
        (capped, block)
    }

    /// Why `solid` is not the exact result of `op`, its points placed by
    /// `pose`.
    fn fault(topo: &Topology, solid: SolidId, op: BooleanOp, pose: &Mat4) -> Option<String> {
        let report = validate_solid(topo, solid).unwrap();
        if !report.is_valid() {
            return Some(format!("invalid: {:?}", report.issues));
        }
        let mut census: BTreeMap<&str, usize> = BTreeMap::new();
        for f in solid_faces(topo, solid).unwrap() {
            *census
                .entry(topo.face(f).unwrap().surface().type_tag())
                .or_default() += 1;
        }
        if !census.contains_key("sphere") || census.values().sum::<usize>() > 12 {
            return Some(format!("faces {census:?}"));
        }
        let mesh = tessellate_solid(topo, solid, 0.002).unwrap();
        if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
            return Some("mesh not closed".into());
        }
        let volume = solid_volume(topo, solid, 0.0005).unwrap();
        let truth = expected(op);
        if (volume - truth).abs() >= 1e-6 * truth {
            return Some(format!("volume {volume} should be {truth}"));
        }
        let (inside, outside) = probes(op);
        for (points, want) in [
            (inside, PointClassification::Inside),
            (outside, PointClassification::Outside),
        ] {
            for p in points {
                let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
                let got = classify_point(topo, solid, q, 0.001, 1e-7).unwrap();
                if got != want {
                    return Some(format!("{p:?} reads {got:?}, not {want:?}"));
                }
            }
        }
        None
    }

    /// The three operations in three poses, the cap and the box turned by
    /// `around` about the ball's centre.
    fn faults(around: &Mat4) -> Vec<String> {
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let mut faults = Vec::new();
        for (name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.37) * Mat4::rotation_x(0.21)),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ] {
            for op in [BooleanOp::Cut, BooleanOp::Fuse, BooleanOp::Intersect] {
                let mut topo = Topology::new();
                let (capped, block) = scene(&mut topo, around, &pose);
                let why = match boolean_with_options(&mut topo, op, capped, block, opts) {
                    Ok(solid) => fault(&topo, solid, op, &(pose * *around)),
                    Err(e) => Some(format!("error {e}")),
                };
                if let Some(why) = why {
                    faults.push(format!("{name} {op:?}: {why}"));
                }
            }
        }
        faults
    }

    #[test]
    fn a_box_through_the_south_pole_of_a_ball_less_its_north_cap_is_exact() {
        let faults = faults(&Mat4::identity());
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }

    #[test]
    fn a_box_through_a_ball_less_a_cap_round_its_equator_is_exact() {
        let faults = faults(&Mat4::rotation_y(std::f64::consts::FRAC_PI_2));
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

//! A torus of radii 4 and 1.5 touching a 45 degree frustum's wall along one
//! circle, both on one axis: resting on the wall from outside, and sitting
//! in the frustum against its wall. Each operand lies whole on one side of
//! the other, so each boolean keeps or drops whole operands.
//!
//! The circle is the one section, and on the whole ring a lone loop round
//! the ring parts nothing. The ring was split along it into a disc and a
//! ring with a hole round it, and then read against samples of which a
//! quarter lie on the contact circle, on neither side: every boolean of the
//! torus resting on the wall fell back to a mesh.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::f64::consts::{PI, SQRT_2};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::OperationsError;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_cone, make_torus};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;

    const MAJOR: f64 = 4.0;
    const MINOR: f64 = 1.5;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9),
            ),
            (
                "mirrored",
                Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4),
            ),
        ]
    }

    /// The frustum about `z` from `z0` to `z1` whose wall rises at 45
    /// degrees from its apex at `apex`, with the ring inside it or not.
    fn check(apex: f64, (z0, z1): (f64, f64), ring_inside: bool, label: &str) {
        let (r0, r1, h) = (z0 - apex, z1 - apex, z1 - z0);
        let cone_volume = PI * h * r1.mul_add(r1, r0.mul_add(r0, r0 * r1)) / 3.0;
        let ring_volume = 2.0 * PI * PI * MAJOR * MINOR * MINOR;
        let in_cone = Point3::new(0.0, 0.0, 2.5);
        let in_ring = Point3::new(MAJOR + 0.5, 0.0, 0.0);
        let away = Point3::new(0.0, 0.0, -10.0);
        for (name, swap, op, want) in [
            (
                "fuse",
                false,
                BooleanOp::Fuse,
                if ring_inside {
                    cone_volume
                } else {
                    cone_volume + ring_volume
                },
            ),
            (
                "frustum less ring",
                false,
                BooleanOp::Cut,
                if ring_inside {
                    cone_volume - ring_volume
                } else {
                    cone_volume
                },
            ),
            (
                "ring less frustum",
                true,
                BooleanOp::Cut,
                if ring_inside { 0.0 } else { ring_volume },
            ),
            (
                "intersect",
                false,
                BooleanOp::Intersect,
                if ring_inside { ring_volume } else { 0.0 },
            ),
        ] {
            for (pose_name, pose) in poses() {
                let label = format!("{label}, {name}, {pose_name}");
                let mut topo = Topology::new();
                let cone = make_cone(&mut topo, r0, r1, h).unwrap();
                transform_solid(&mut topo, cone, &Mat4::translation(0.0, 0.0, z0)).unwrap();
                let ring = make_torus(&mut topo, MAJOR, MINOR, 32).unwrap();
                for solid in [cone, ring] {
                    transform_solid(&mut topo, solid, &pose).unwrap();
                }
                let exact = BooleanOptions {
                    mesh_fallback: false,
                    ..BooleanOptions::default()
                };
                let (a, b) = if swap { (ring, cone) } else { (cone, ring) };
                let piece = match boolean_with_options(&mut topo, op, a, b, exact) {
                    Ok(piece) => piece,
                    Err(OperationsError::EmptyResult { .. }) if want == 0.0 => continue,
                    Err(e) => panic!("{label}: {e}"),
                };
                let volume = solid_volume(&topo, piece, 0.001).unwrap();
                assert!(
                    (volume - want).abs() <= 1e-9 * want.max(1.0),
                    "{label}: volume {volume}, want {want}"
                );
                if want == 0.0 {
                    continue;
                }
                let report = validate_solid(&topo, piece).unwrap();
                assert!(report.is_valid(), "{label}: invalid: {report:?}");
                assert!(
                    is_watertight(&tessellate_solid(&topo, piece, 0.01).unwrap()),
                    "{label}: mesh not watertight"
                );
                let tol = brepkit_math::tolerance::Tolerance::new();
                let holds_cone = !swap && op != BooleanOp::Intersect;
                let holds_ring = match op {
                    BooleanOp::Fuse => true,
                    BooleanOp::Cut => swap,
                    BooleanOp::Intersect => ring_inside,
                };
                for (p, inside) in [(in_cone, holds_cone), (in_ring, holds_ring), (away, false)] {
                    let class =
                        classify_point(&topo, piece, pose.mul_point(p), 0.01, tol.linear).unwrap();
                    let want = if inside {
                        PointClassification::Inside
                    } else {
                        PointClassification::Outside
                    };
                    assert_eq!(class, want, "{label}: point {p:?}");
                }
            }
        }
    }

    #[test]
    fn a_ring_resting_on_a_frustums_wall() {
        check(
            MINOR.mul_add(SQRT_2, -MAJOR),
            (-1.5, 3.0),
            false,
            "resting on the wall",
        );
    }

    #[test]
    fn a_ring_sitting_in_a_frustum_against_its_wall() {
        check(
            -MINOR.mul_add(SQRT_2, MAJOR),
            (-3.0, 3.0),
            true,
            "sitting in the frustum",
        );
    }
}

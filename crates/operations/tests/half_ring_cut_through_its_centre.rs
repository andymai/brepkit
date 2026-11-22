//! Half of a thin ring (radii 4 and 0.1, the half `x > 0`) cut and
//! intersected by a box whose floor is the plane through the ring's centre
//! across its axis. The floor meets the half ring's torus face along its
//! inner equator and its outer equator, which is the face's seam, and
//! crosses each of its two end discs along a diameter.
//!
//! The half ring's torus face holds the whole tube over half a turn about
//! the axis. Its outline was laid half a turn of `u` from the floor's arcs on
//! it, so they joined none of its rims; a point of its lower half read the
//! tube angle of the upper one; and the floor's arc along the outer equator
//! read a tube angle of a whole turn by rounding when turned, past the face's
//! sampled window. Each case fell back to a mesh.
//!
//! A half turn about the axis `x` maps the half ring onto itself and the two
//! sides of the floor onto each other, so each side holds a quarter of the
//! ring, `pi^2 R r^2 / 2`.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_torus};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;

    const MAJOR: f64 = 4.0;
    const MINOR: f64 = 0.1;

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

    fn check(op: BooleanOp, name: &str, pose: &Mat4) {
        let label = format!("{op:?}, {name}");
        let mut topo = Topology::new();
        let ring = make_torus(&mut topo, MAJOR, MINOR, 32).unwrap();
        let other_half = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(
            &mut topo,
            other_half,
            &Mat4::translation(-20.0, -10.0, -10.0),
        )
        .unwrap();
        let lid = make_box(&mut topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(&mut topo, lid, &Mat4::translation(-10.0, -10.0, 0.0)).unwrap();
        for solid in [ring, other_half, lid] {
            transform_solid(&mut topo, solid, pose).unwrap();
        }
        let exact = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let half = boolean_with_options(&mut topo, BooleanOp::Cut, ring, other_half, exact)
            .unwrap_or_else(|e| panic!("{label}: the half ring: {e}"));
        let piece = boolean_with_options(&mut topo, op, half, lid, exact)
            .unwrap_or_else(|e| panic!("{label}: {e}"));

        let report = validate_solid(&topo, piece).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {report:?}");
        assert!(
            is_watertight(&tessellate_solid(&topo, piece, 0.001).unwrap()),
            "{label}: mesh not watertight"
        );
        let want = 0.5 * PI * PI * MAJOR * MINOR * MINOR;
        let volume = solid_volume(&topo, piece, 0.0001).unwrap();
        assert!(
            (volume - want).abs() <= 1e-9 * want,
            "{label}: volume {volume}, want {want}"
        );

        // Points in the tube above and below the floor, on both sides of the
        // ring's middle meridian, and one in the half the cut takes off.
        let tol = brepkit_math::tolerance::Tolerance::new();
        let kept_below = op == BooleanOp::Cut;
        for (p, below) in [
            (Point3::new(4.0, 0.0, -0.05), true),
            (Point3::new(0.4, 3.95, -0.05), true),
            (Point3::new(0.4, -4.05, -0.05), true),
            (Point3::new(4.0, 0.0, 0.05), false),
            (Point3::new(0.4, 3.95, 0.05), false),
            (Point3::new(0.4, -4.05, 0.05), false),
        ] {
            let class = classify_point(&topo, piece, pose.mul_point(p), 0.01, tol.linear).unwrap();
            let want = if below == kept_below {
                PointClassification::Inside
            } else {
                PointClassification::Outside
            };
            assert_eq!(class, want, "{label}: point {p:?}");
        }
        let off = Point3::new(-4.0, 0.0, 0.05 * if kept_below { -1.0 } else { 1.0 });
        let class = classify_point(&topo, piece, pose.mul_point(off), 0.01, tol.linear).unwrap();
        assert_eq!(
            class,
            PointClassification::Outside,
            "{label}: point {off:?}"
        );
    }

    #[test]
    fn the_part_under_a_floor_through_the_centre_is_a_quarter_ring() {
        for (name, pose) in poses() {
            check(BooleanOp::Cut, name, &pose);
        }
    }

    #[test]
    fn the_part_over_a_floor_through_the_centre_is_a_quarter_ring() {
        for (name, pose) in poses() {
            check(BooleanOp::Intersect, name, &pose);
        }
    }
}

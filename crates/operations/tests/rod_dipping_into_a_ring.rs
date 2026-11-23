//! A rod of radius 0.6 along y whose underside dips into the top of the
//! ring of radii 4 and 1.5 by 7e-4 or 2.7e-5, where the two surfaces cross
//! at about 0.03 to 0.05 or 0.006 to 0.01 radians, in two loops 0.02 to 0.04
//! across. Near the origin and moved 1e6 or 1e9 along each axis, the cut
//! returned the whole ring and the intersect nothing: the dips were missed
//! with no error. The boolean now carves each dip or refuses; it never
//! misses a section without saying so.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_cylinder, make_torus};
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;

    /// Tanh-sinh quadrature over `[a, b]`, exact to rounding for a smooth
    /// integrand that vanishes like a power at the ends.
    fn tanh_sinh(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let h = 1.0 / 32.0;
        let half = 0.5 * (b - a);
        let mut sum = 0.0;
        for k in -120..=120 {
            let t = f64::from(k) * h;
            let s = 0.5 * PI * t.sinh();
            let weight = 0.5 * PI * t.cosh() / (s.cosh() * s.cosh());
            let gap = half / (s.abs().exp() * s.cosh());
            let x = if s >= 0.0 { b - gap } else { a + gap };
            if x > a && x < b {
                sum += weight * f(x);
            }
        }
        sum * h * half
    }

    /// The part of the ring under one dip: at each `x` across the rod, the
    /// tube's top above the rod's underside, integrated over `y` between the
    /// two distances from the axis where they meet.
    fn dip_volume(depth: f64) -> f64 {
        let underside = |x: f64| (2.1 - depth) - x.mul_add(-x, 0.36).sqrt();
        let reach = depth.mul_add(-depth, 1.2 * depth).sqrt();
        let across = |x: f64| {
            let b = underside(x);
            let w = b.mul_add(-b, 2.25).max(0.0).sqrt();
            let (y0, y1) = (
                (4.0 - w).mul_add(4.0 - w, -x * x).sqrt(),
                (4.0 + w).mul_add(4.0 + w, -x * x).sqrt(),
            );
            let height = |y: f64| {
                let off = x.hypot(y) - 4.0;
                off.mul_add(-off, 2.25).max(0.0).sqrt() - b
            };
            tanh_sinh(&height, y0, y1)
        };
        tanh_sinh(&across, -reach, reach)
    }

    #[test]
    fn a_rod_dipping_into_a_ring_is_carved_or_refused_in_every_pose() {
        let poses = [
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],
            [0.0, 1e6, 0.0],
            [0.0, 0.0, 1e6],
            [1e9, 0.0, 0.0],
            [0.0, 1e9, 0.0],
            [0.0, 0.0, 1e9],
        ];
        let options = ClassifyOptions::default();
        for depth in [7e-4, 2.7e-5] {
            let dips = 2.0 * dip_volume(depth);
            for op in [BooleanOp::Cut, BooleanOp::Intersect] {
                for s in poses {
                    let mut topo = Topology::new();
                    let ring = make_torus(&mut topo, 4.0, 1.5, 32).unwrap();
                    let rod = make_cylinder(&mut topo, 0.6, 20.0).unwrap();
                    let lay = Mat4::translation(0.0, 10.0, 1.5 + 0.6 - depth)
                        * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
                    transform_solid(&mut topo, rod, &lay).unwrap();
                    for solid in [ring, rod] {
                        transform_solid(&mut topo, solid, &Mat4::translation(s[0], s[1], s[2]))
                            .unwrap();
                    }
                    let exact = BooleanOptions {
                        mesh_fallback: false,
                        ..BooleanOptions::default()
                    };
                    let Ok(piece) = boolean_with_options(&mut topo, op, ring, rod, exact) else {
                        continue;
                    };
                    let label = format!("depth {depth} {op:?} moved {s:?}");
                    let at = |x: f64, y: f64, z: f64| {
                        classify_point(
                            &topo,
                            piece,
                            Point3::new(x + s[0], y + s[1], z + s[2]),
                            &options,
                        )
                        .unwrap()
                    };
                    let (inside, outside) =
                        (PointClassification::Inside, PointClassification::Outside);
                    let (in_dip, in_ring, in_rod) = match op {
                        BooleanOp::Cut => (outside, inside, outside),
                        _ => (inside, outside, outside),
                    };
                    for y in [4.0, -4.0] {
                        assert_eq!(
                            at(0.0, y, 1.5 - depth / 2.0),
                            in_dip,
                            "{label}: dip at y {y}"
                        );
                    }
                    assert_eq!(
                        at(4.0, 0.0, 0.0),
                        in_ring,
                        "{label}: the tube away from the dips"
                    );
                    assert_eq!(
                        at(0.0, 0.0, 2.1 - depth),
                        in_rod,
                        "{label}: the rod over the hole"
                    );
                    let want = match op {
                        BooleanOp::Cut => (18.0 * PI * PI) - dips,
                        _ => dips,
                    };
                    let volume = solid_volume(&topo, piece, 0.001).unwrap();
                    assert!(
                        (volume - want).abs() <= 1e-2 * dips,
                        "{label}: volume {volume}, want {want} (dips {dips})"
                    );
                }
            }
        }
    }
}

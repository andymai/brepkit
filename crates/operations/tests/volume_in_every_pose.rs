//! Exact solids measured upright, turned and mirrored against their closed
//! forms.
//!
//! - A block 24 x 24 x 1000 less a D-shaped hole: a bore of radius 10 cut
//!   off by a flat that leans 5e-8 off the bore's axis, which meets the bore
//!   along two arcs of an ellipse 2e8 across, centred 1e8 away.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::quadrature::gauss_legendre_points;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::solid::SolidId;

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            ("turned", Mat4::rotation_z(0.7)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// `f` integrated from `a` to `b` in `panels` equal panels.
    fn integral(a: f64, b: f64, panels: u32, f: impl Fn(f64) -> f64) -> f64 {
        let h = (b - a) / f64::from(panels);
        (0..panels)
            .map(|k| {
                let lo = h.mul_add(f64::from(k), a);
                gauss_legendre_points(20)
                    .iter()
                    .map(|g| 0.5 * h * g.w * f((0.5 * h).mul_add(g.x + 1.0, lo)))
                    .sum::<f64>()
            })
            .sum()
    }

    const BORE: f64 = 10.0;
    const FLAT: f64 = 5.0;
    const LEAN: f64 = 5e-8;
    const LENGTH: f64 = 1000.0;

    /// The block less the D-shaped hole: the bore on the `z` axis and the
    /// side of the flat `y cos θ + z sin θ = 5` toward it, `θ = atan(5e-8)`.
    fn d_holed_block(topo: &mut Topology) -> SolidId {
        let block = make_box(topo, 24.0, 24.0, LENGTH).unwrap();
        transform_solid(topo, block, &Mat4::translation(-12.0, -12.0, 0.0)).unwrap();
        let rod = make_cylinder(topo, BORE, LENGTH + 20.0).unwrap();
        transform_solid(topo, rod, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
        let keep = make_box(topo, 2.0 * BORE, 2.0 * BORE, LENGTH + 40.0).unwrap();
        let place =
            Mat4::rotation_x(LEAN.atan()) * Mat4::translation(-BORE, FLAT - 2.0 * BORE, -20.0);
        transform_solid(topo, keep, &place).unwrap();
        let d = boolean(topo, BooleanOp::Intersect, rod, keep).unwrap();
        boolean(topo, BooleanOp::Cut, block, d).unwrap()
    }

    /// The area of the disc of radius 10 on the side of `y = y0` toward its
    /// centre.
    fn d_area(y0: f64) -> f64 {
        let r = BORE;
        PI * r * r - (r * r * (y0 / r).acos() - y0 * (r * r - y0 * y0).sqrt())
    }

    #[test]
    fn a_block_with_a_long_d_shaped_hole_measures_alike_in_every_pose() {
        let theta = LEAN.atan();
        let hole = integral(0.0, LENGTH, 4, |z| {
            d_area(z.mul_add(-theta.sin(), FLAT) / theta.cos())
        });
        let expected = 24.0 * 24.0 * LENGTH - hole;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = d_holed_block(&mut topo);
            transform_solid(&mut topo, solid, &pose).unwrap();
            let volume = solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }
}

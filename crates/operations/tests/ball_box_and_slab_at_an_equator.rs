//! A ball, a box whose top lies on the ball's equator, and a slab across the
//! equator's arc on that top, fused one solid at a time. The ball and the box
//! fuse first: the ball's faces join into one face round its north pole,
//! whose outline runs once round the ball, and the slab's sections end on
//! that outline.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
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

    /// An axis box, by its low and high corners.
    type Cuboid = ([f64; 3], [f64; 3]);

    const BOX: Cuboid = ([-0.5, -2.0, -2.0], [0.5, 2.0, 0.0]);
    const SLABS: [Cuboid; 3] = [
        ([0.1, 0.5, -0.5], [0.3, 1.5, 0.5]),
        ([-0.3, 0.6, -0.4], [-0.15, 1.6, 0.4]),
        ([0.05, 0.7, -0.3], [0.15, 1.3, 0.3]),
    ];

    fn meet(a: Cuboid, b: Cuboid) -> Option<Cuboid> {
        let lo = [0, 1, 2].map(|k| a.0[k].max(b.0[k]));
        let hi = [0, 1, 2].map(|k| a.1[k].min(b.1[k]));
        (0..3).all(|k| lo[k] < hi[k]).then_some((lo, hi))
    }

    fn cuboid_volume(c: Cuboid) -> f64 {
        (0..3).map(|k| c.1[k] - c.0[k]).product()
    }

    /// The area of the disc of radius `r` about the origin inside the
    /// rectangle `[y0, y1] x [z0, z1]`, in closed form between the `y` where
    /// the circle crosses `z = z0` or `z = z1`.
    fn disc_in_rectangle(r: f64, (y0, y1): (f64, f64), (z0, z1): (f64, f64)) -> f64 {
        let half = |y: f64| (r * r - y * y).max(0.0).sqrt();
        // The integral of `half` from 0 to `y`.
        let under = |y: f64| {
            let y = y.clamp(-r, r);
            0.5 * (y * half(y) + r * r * (y / r).asin())
        };
        let mut cuts = vec![y0.max(-r), y1.min(r)];
        if cuts[0] >= cuts[1] {
            return 0.0;
        }
        for z in [z0, z1] {
            if z.abs() < r {
                let y = half(z);
                cuts.extend([-y, y]);
            }
        }
        cuts.retain(|&y| y >= y0.max(-r) && y <= y1.min(r));
        cuts.sort_by(f64::total_cmp);
        let mut area = 0.0;
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b - a <= 0.0 {
                continue;
            }
            let s = half(0.5 * (a + b));
            let top = if z1 < s { None } else { Some(()) };
            let bottom = if z0 > -s { None } else { Some(()) };
            // Between two cuts the chord's span in z is one of these.
            let length = match (top, bottom) {
                (Some(()), Some(())) => 2.0 * (under(b) - under(a)),
                (Some(()), None) => (under(b) - under(a)) - z0 * (b - a),
                (None, Some(())) => z1 * (b - a) + (under(b) - under(a)),
                (None, None) => (z1 - z0) * (b - a),
            };
            area += length.max(0.0);
        }
        area
    }

    /// The volume of the unit ball inside an axis box: the disc of each
    /// slice `x` inside the box's rectangle, integrated over `x`.
    fn ball_in_cuboid(c: Cuboid) -> f64 {
        const PANELS: usize = 4000;
        let (x0, x1) = (c.0[0].max(-1.0), c.1[0].min(1.0));
        if x0 >= x1 {
            return 0.0;
        }
        let gauss = brepkit_math::quadrature::gauss_legendre_points(8);
        let width = (x1 - x0) / PANELS as f64;
        let mut total = 0.0;
        for p in 0..PANELS {
            let mid = x0 + width * (p as f64 + 0.5);
            for g in gauss {
                let x = mid + 0.5 * width * g.x;
                let r = (1.0 - x * x).max(0.0).sqrt();
                total +=
                    0.5 * width * g.w * disc_in_rectangle(r, (c.0[1], c.1[1]), (c.0[2], c.1[2]));
            }
        }
        total
    }

    fn expected_volume(slab: Cuboid) -> f64 {
        let ball = 4.0 / 3.0 * std::f64::consts::PI;
        let both = meet(BOX, slab);
        ball + cuboid_volume(BOX) + cuboid_volume(slab)
            - ball_in_cuboid(BOX)
            - ball_in_cuboid(slab)
            - both.map_or(0.0, cuboid_volume)
            + both.map_or(0.0, ball_in_cuboid)
    }

    fn scene(topo: &mut Topology, slab: Cuboid, pose: &Mat4) -> [SolidId; 3] {
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        transform_solid(topo, ball, pose).unwrap();
        let cuboid = |topo: &mut Topology, (lo, hi): Cuboid| {
            let s = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
            transform_solid(topo, s, &(*pose * Mat4::translation(lo[0], lo[1], lo[2]))).unwrap();
            s
        };
        [ball, cuboid(topo, BOX), cuboid(topo, slab)]
    }

    fn fault(topo: &Topology, fused: SolidId, expected: f64) -> Option<String> {
        let report = validate_solid(topo, fused).unwrap();
        if !report.is_valid() {
            return Some(format!("invalid: {:?}", report.issues));
        }
        let mut census: BTreeMap<&str, usize> = BTreeMap::new();
        for f in solid_faces(topo, fused).unwrap() {
            *census
                .entry(topo.face(f).unwrap().surface().type_tag())
                .or_default() += 1;
        }
        if !census.contains_key("sphere") || census.values().sum::<usize>() > 40 {
            return Some(format!("faces {census:?}"));
        }
        let mesh = tessellate_solid(topo, fused, 0.002).unwrap();
        if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
            return Some("mesh not closed".into());
        }
        let volume = solid_volume(topo, fused, 0.0005).unwrap();
        ((volume - expected).abs() >= 1e-6 * expected)
            .then(|| format!("volume {volume} should be {expected}"))
    }

    #[test]
    fn a_ball_and_a_box_on_its_equator_then_a_slab_fuse_one_at_a_time() {
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let mut faults = Vec::new();
        for slab in SLABS {
            let expected = expected_volume(slab);
            for (name, pose) in [
                ("upright", Mat4::identity()),
                ("turned", Mat4::rotation_z(0.37) * Mat4::rotation_x(0.21)),
                ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
            ] {
                for order in [[0, 1, 2], [1, 0, 2]] {
                    let mut topo = Topology::new();
                    let solids = scene(&mut topo, slab, &pose);
                    let [a, b, c] = order.map(|i| solids[i]);
                    let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, options)
                        .and_then(|ab| {
                            boolean_with_options(&mut topo, BooleanOp::Fuse, ab, c, options)
                        });
                    let why = match fused {
                        Ok(fused) => fault(&topo, fused, expected),
                        Err(e) => Some(format!("error {e}")),
                    };
                    if let Some(why) = why {
                        faults.push(format!("{slab:?} {name} {order:?}: {why}"));
                    }
                }
            }
        }
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

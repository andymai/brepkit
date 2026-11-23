//! Two round posts through a plate whose top circles cross: the plate's top
//! keeps one hole, the outline of both discs, in an N-way fuse, in a fuse of
//! the compound and one solid at a time.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;
    use std::f64::consts::PI;

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::compound_ops::fuse_all;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::compound::Compound;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const RADIUS: f64 = 1.0;
    const ABOVE: f64 = 2.0;

    /// The plate `6 x 6 x 1` with its top at `z = 0`, and two posts of
    /// radius 1 from `z = -0.5` to `z = 2` with their axes `apart` from each
    /// other on `y = 0`, all placed by `pose`.
    fn scene(topo: &mut Topology, apart: f64, pose: &Mat4) -> [SolidId; 3] {
        let plate = make_box(topo, 6.0, 6.0, 1.0).unwrap();
        transform_solid(topo, plate, &(*pose * Mat4::translation(-3.0, -3.0, -1.0))).unwrap();
        let mut post = |x: f64| {
            let post = make_cylinder(topo, RADIUS, ABOVE + 0.5).unwrap();
            transform_solid(topo, post, &(*pose * Mat4::translation(x, 0.0, -0.5))).unwrap();
            post
        };
        [plate, post(-0.5 * apart), post(0.5 * apart)]
    }

    /// The plate and the part of both posts above it: two discs of radius 1
    /// whose centres stand `apart`, less the lens they share.
    fn expected_volume(apart: f64) -> f64 {
        let lens = 2.0 * RADIUS * RADIUS * (0.5 * apart / RADIUS).acos()
            - 0.5 * apart * (4.0 * RADIUS * RADIUS - apart * apart).sqrt();
        36.0 + (2.0 * PI * RADIUS * RADIUS - lens) * ABOVE
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(4.0, -3.0, 1.5) * Mat4::rotation_x(0.43) * Mat4::rotation_z(1.1),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    /// What is wrong with `fused`, if anything.
    fn fault(topo: &Topology, fused: SolidId, apart: f64, pose: &Mat4) -> Option<String> {
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
        if census.len() != 2
            || census.get("cylinder").is_none_or(|&n| n < 2)
            || census.values().sum::<usize>() >= 20
        {
            return Some(format!("faces {census:?}"));
        }
        let mesh = tessellate_solid(topo, fused, 0.005).unwrap();
        if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
            return Some(format!(
                "mesh: {} open, {} not manifold",
                boundary_edge_count(&mesh),
                non_manifold_edge_count(&mesh)
            ));
        }
        let expected = expected_volume(apart);
        let volume = solid_volume(topo, fused, 0.0005).unwrap();
        if (volume - expected).abs() >= 1e-6 * expected {
            return Some(format!("volume {volume} should be {expected}"));
        }
        // Just above the plate: on the lens both posts share, in each post
        // alone, and beside the posts; just below the plate's top, under a
        // post and beside them.
        let half = 0.5 * apart;
        for (x, y, z, inside) in [
            (0.0, 0.0, 0.01, true),
            (-half - 0.9, 0.0, 0.01, true),
            (half + 0.9, 0.0, 0.01, true),
            (
                0.0,
                (RADIUS * RADIUS - half * half).sqrt() + 0.05,
                0.01,
                false,
            ),
            (-half - 1.1, 0.0, 0.01, false),
            (0.0, 0.0, -0.01, true),
            (2.5, 2.5, -0.01, true),
            (2.5, 2.5, 0.01, false),
        ] {
            let p = pose.mul_point(Point3::new(x, y, z));
            let class = classify_point(topo, fused, p, &ClassifyOptions::default()).unwrap();
            let want = if inside {
                PointClassification::Inside
            } else {
                PointClassification::Outside
            };
            if class != want {
                return Some(format!("({x}, {y}, {z}) reads {class:?}"));
            }
        }
        None
    }

    #[test]
    fn two_posts_whose_circles_cross_on_a_plate_fuse_exactly() {
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let mut faults = Vec::new();
        let mut note = |label: String, result: Option<String>| {
            if let Some(why) = result {
                faults.push(format!("{label}: {why}"));
            }
        };
        for apart in [1.2, 0.5, 1.9] {
            for (name, pose) in poses() {
                let label = format!("{name}, {apart} apart");

                let mut topo = Topology::new();
                let solids = scene(&mut topo, apart, &pose);
                note(
                    format!("{label}, fuse_n"),
                    match brepkit_algo::gfa::fuse_n(&mut topo, &solids) {
                        Ok(fused) => fault(&topo, fused, apart, &pose),
                        Err(e) => Some(format!("error {e}")),
                    },
                );

                let mut topo = Topology::new();
                let solids = scene(&mut topo, apart, &pose);
                let compound = topo.add_compound(Compound::new(solids.to_vec()));
                note(
                    format!("{label}, fuse_all"),
                    match fuse_all(&mut topo, compound) {
                        Ok(fused) => fault(&topo, fused, apart, &pose),
                        Err(e) => Some(format!("error {e}")),
                    },
                );

                for order in [[0, 1, 2], [1, 2, 0], [1, 0, 2]] {
                    let mut topo = Topology::new();
                    let solids = scene(&mut topo, apart, &pose);
                    let [a, b, c] = order.map(|i| solids[i]);
                    let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, options)
                        .and_then(|ab| {
                            boolean_with_options(&mut topo, BooleanOp::Fuse, ab, c, options)
                        });
                    note(
                        format!("{label}, pairwise {order:?}"),
                        match fused {
                            Ok(fused) => fault(&topo, fused, apart, &pose),
                            Err(e) => Some(format!("error {e}")),
                        },
                    );
                }
            }
        }
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

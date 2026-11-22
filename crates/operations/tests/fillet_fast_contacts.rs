//! A round whose contacts run far faster than its spine: a box's unit edge
//! between two square faces, rounded with a radius that grows linearly from
//! 1 to `end` along it. Each section is the quarter circle of its radius
//! square to the edge, so the contacts move about `end - 1` times as fast as
//! the spine. The round takes off `(1 - π/4) r²` per unit length, so
//! `(1 - π/4) (end³ - 1) / (3 (end - 1))` in all.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_blend::radius_law::RadiusLaw;
    use brepkit_math::mat::Mat4;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_edges;

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    #[test]
    fn a_round_whose_radius_grows_two_hundred_fold_along_a_unit_edge_is_walked() {
        for end in [51.0, 201.0] {
            for (name, pose) in poses() {
                let what = format!("1 to {end}, {name}");
                let mut topo = Topology::new();
                let block = make_box(&mut topo, 1.0, 300.0, 300.0).unwrap();
                let edge = solid_edges(&topo, block)
                    .unwrap()
                    .into_iter()
                    .find(|&e| {
                        let edge = topo.edge(e).unwrap();
                        [edge.start(), edge.end()].iter().all(|&v| {
                            let p = topo.vertex(v).unwrap().point();
                            p.y().abs() < 1e-9 && (p.z() - 300.0).abs() < 1e-9
                        })
                    })
                    .unwrap();
                transform_solid(&mut topo, block, &pose).unwrap();
                let before = solid_volume(&topo, block, 1e-3).unwrap();
                let mut builder = FilletBuilder::new(&mut topo, block);
                builder.add_edges_with_law(&[edge], RadiusLaw::Linear { start: 1.0, end });
                let built = builder.build().unwrap();
                assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
                assert!(
                    validate_solid(&topo, built.solid).unwrap().is_valid(),
                    "{what}: not valid"
                );
                let expected = (1.0 - PI / 4.0) * (end.powi(3) - 1.0) / (3.0 * (end - 1.0));
                let removed = before - solid_volume(&topo, built.solid, 1e-3).unwrap();
                assert!(
                    (removed - expected).abs() < 1e-5 * expected,
                    "{what}: took off {removed}, expected {expected}"
                );
                let mesh = tessellate_solid(&topo, built.solid, 0.05).unwrap();
                assert!(is_watertight(&mesh), "{what}: open mesh");
            }
        }
    }
}

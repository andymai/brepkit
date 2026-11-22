//! The rim of a box's sloped top, its upright corner edge rounded first.
//! The top's edges along the slope rise, so a round's section square to
//! one of them leans: at the corner where it meets the round of a level
//! edge, and where it runs on to the rounded corner, it stands past the
//! side wall's end. The corner solver builds neither corner, and the rounds
//! it would join there leave faces whose outlines cross themselves: the
//! fillet is refused, and the box comes back as it was.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;

    const SLOPE: f64 = 0.05;

    fn edges_where(
        topo: &Topology,
        solid: SolidId,
        keep: impl Fn(Point3, Point3) -> bool,
    ) -> Vec<EdgeId> {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                keep(a, b)
            })
            .collect()
    }

    /// A 20 x 20 x 10 box, its upright edge at x = y = 20 rounded by
    /// `corner`, cut by the plane z = 8 + 0.05 y; and the five edges of
    /// its top.
    fn sloped_box(topo: &mut Topology, corner: f64) -> (SolidId, Vec<EdgeId>) {
        let block = make_box(topo, 20.0, 20.0, 10.0).unwrap();
        let upright = edges_where(topo, block, |a, b| {
            [a, b]
                .iter()
                .all(|p| (p.x() - 20.0).abs() < 1e-9 && (p.y() - 20.0).abs() < 1e-9)
        });
        let rounded = fillet_v2(topo, block, &upright, corner).unwrap();
        assert!(rounded.failed.is_empty());
        let cutter = make_box(topo, 100.0, 100.0, 100.0).unwrap();
        let place = Mat4::translation(0.0, 0.0, 8.0)
            * Mat4::rotation_x(SLOPE.atan())
            * Mat4::translation(-50.0, -50.0, 0.0);
        transform_solid(topo, cutter, &place).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, rounded.solid, cutter).unwrap();
        let on_top = |p: Point3| (p.z() - 8.0 - SLOPE * p.y()).abs() < 1e-7;
        let rim = edges_where(topo, cut, |a, b| on_top(a) && on_top(b));
        assert_eq!(rim.len(), 5);
        (cut, rim)
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(4.0, -9.5, 21.0) * Mat4::rotation_x(0.7) * Mat4::rotation_z(1.9),
            ),
            ("mirrored", Mat4::scale(1.0, 1.0, -1.0)),
        ]
    }

    #[test]
    fn the_sloped_rim_is_refused() {
        for (corner, r) in [(2.0, 2.0), (2.0, 1.0), (3.0, 1.5)] {
            for (name, pose) in poses() {
                let label = format!("{name}, corner {corner}, r {r}");
                let mut topo = Topology::new();
                let (solid, rim) = sloped_box(&mut topo, corner);
                transform_solid(&mut topo, solid, &pose).unwrap();
                let before = solid_volume(&topo, solid, 0.01).unwrap();
                // A refusal before the build is an error; one after it
                // names each edge and returns the box.
                if let Ok(result) = fillet_v2(&mut topo, solid, &rim, r) {
                    assert_eq!(result.solid, solid, "{label}: a result came back");
                    assert!(result.succeeded.is_empty(), "{label}");
                    let mut failed: Vec<usize> =
                        result.failed.iter().map(|(e, _)| e.index()).collect();
                    failed.sort_unstable();
                    let mut picked: Vec<usize> = rim.iter().map(|e| e.index()).collect();
                    picked.sort_unstable();
                    assert_eq!(failed, picked, "{label}: the failed edges");
                }
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label}: the box is spoiled: {report:?}");
                let after = solid_volume(&topo, solid, 0.01).unwrap();
                assert!((after - before).abs() <= 1e-9 * before, "{label}");
            }
        }
    }
}

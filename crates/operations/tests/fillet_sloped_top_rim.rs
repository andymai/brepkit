//! The rim of a box's sloped top. The top's edges along the slope rise, so
//! a round's section square to one of them leans and stands past the side
//! wall's end.
//!
//! Where a rising edge's round meets a level edge's round beside a sharp
//! upright edge, the corner is capped: on a plain box the rim rounds to its
//! sections. With the box's upright corner edge rounded first, a rising
//! edge runs on to that round's arc, and the two rounds' end sections there
//! do not meet: the round along the line ends past the line where the side
//! wall meets the corner's cylinder, and the round along the arc starts
//! past it. The corner solver does not build that joint, and the patch it
//! puts there has an outline that crosses itself: the fillet is refused,
//! and the box comes back as it was.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::blend_ops::{all_blended, fillet_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{solid_edges, solid_faces};
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

    /// `solid` cut by the plane z = 8 + 0.05 y, and the `edges` edges of
    /// its top.
    fn cut_by_the_slope(
        topo: &mut Topology,
        solid: SolidId,
        edges: usize,
    ) -> (SolidId, Vec<EdgeId>) {
        let cutter = make_box(topo, 100.0, 100.0, 100.0).unwrap();
        let place = Mat4::translation(0.0, 0.0, 8.0)
            * Mat4::rotation_x(SLOPE.atan())
            * Mat4::translation(-50.0, -50.0, 0.0);
        transform_solid(topo, cutter, &place).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, solid, cutter).unwrap();
        let on_top = |p: Point3| (p.z() - 8.0 - SLOPE * p.y()).abs() < 1e-7;
        let rim = edges_where(topo, cut, |a, b| on_top(a) && on_top(b));
        assert_eq!(rim.len(), edges);
        (cut, rim)
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
        cut_by_the_slope(topo, rounded.solid, 5)
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

    /// `∫ sqrt(a^2 - u^2) du` from 0 to `u`.
    fn quarter(a: f64, u: f64) -> f64 {
        let u = u.clamp(-a, a);
        0.5 * (u * (a * a - u * u).sqrt() + a * a * (u / a).asin())
    }

    /// The area cut from a top edge's corner, where the top meets a wall at
    /// `alpha` inside the solid, by the round's circle of radius `r`, at
    /// depths below the top from `w` down: at depth `d` the cut runs from
    /// the circle, `r cot(alpha / 2) - sqrt(r^2 - (r - d)^2)` from the
    /// edge, to the wall, `d cot alpha` back toward it, down to where the
    /// circle touches the wall, `r (1 + cos alpha)` below the top.
    fn corner_below(alpha: f64, r: f64, w: f64) -> f64 {
        let deepest = r * (1.0 + alpha.cos());
        if w >= deepest {
            return 0.0;
        }
        let cut =
            |d: f64| r * d / (0.5 * alpha).tan() - 0.5 * d * d / alpha.tan() - quarter(r, d - r);
        cut(deepest) - cut(w)
    }

    /// Gauss-Legendre nodes and weights on [-1, 1].
    const NODES: [(f64, f64); 8] = [
        (-0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
        (-0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (-0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (-0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.183_434_642_495_649_8, 0.362_683_783_378_362),
        (0.525_532_409_916_329, 0.313_706_645_877_887_3),
        (0.796_666_477_413_626_7, 0.222_381_034_453_374_47),
        (0.960_289_856_497_536_2, 0.101_228_536_290_376_26),
    ];

    /// The volume of the plain sloped box with the four edges of its top
    /// rounded by `r`, the upright edges sharp. Across the box, at `t` in
    /// from an end, the section is the box's with its top lowered by the end
    /// round's inset `w = r - sqrt(r^2 - (r - t)^2)` (none past `r`) and
    /// each top corner cut by its round's circle: the rounds along the
    /// level edges stand square to the ends, and the ends to the top.
    fn sloped_rim_volume(r: f64) -> f64 {
        let rise = SLOPE.hypot(1.0);
        // The top meets the wall y = 0 and the wall y = 20 at these angles.
        let low = std::f64::consts::PI - (SLOPE / rise).acos();
        let high = std::f64::consts::PI - (-SLOPE / rise).acos();
        let area = |w: f64| {
            20.0 * (8.0 + 10.0 * SLOPE - w * rise)
                - corner_below(low, r, w)
                - corner_below(high, r, w)
        };
        // `t = r (1 - sin phi)`, so `w = r (1 - cos phi)`.
        let panels = 64;
        let width = std::f64::consts::FRAC_PI_2 / f64::from(panels);
        let mut end = 0.0;
        for panel in 0..panels {
            let mid = width * (f64::from(panel) + 0.5);
            for (x, weight) in NODES {
                let phi = mid + 0.5 * width * x;
                end += 0.5
                    * width
                    * weight
                    * (area(r * (1.0 - phi.cos())) - area(0.0))
                    * r
                    * phi.cos();
            }
        }
        20.0 * area(0.0) + 2.0 * end
    }

    /// The box with no rounded upright edge: at the two corners where the
    /// top's rising edges meet the high level edge, the top meets the wall
    /// at less than a right angle, and each rising edge's round runs on past
    /// the level edge's round's end. The rim rounds to its sections, valid
    /// and watertight, in each pose.
    #[test]
    fn the_rim_of_a_plain_sloped_box_rounds_to_its_sections() {
        for r in [1.0, 2.0] {
            let want = sloped_rim_volume(r);
            for (name, pose) in poses() {
                let label = format!("{name}, r {r}");
                let mut topo = Topology::new();
                let block = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
                let (solid, rim) = cut_by_the_slope(&mut topo, block, 4);
                transform_solid(&mut topo, solid, &pose).unwrap();
                let result = fillet_v2(&mut topo, solid, &rim, r)
                    .and_then(all_blended)
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{label}: {report:?}");
                let mesh = tessellate_solid(&topo, result, 0.02).unwrap();
                assert!(is_watertight(&mesh), "{label}: the mesh is open");
                assert_eq!(solid_faces(&topo, result).unwrap().len(), 10, "{label}");
                let got = solid_volume(&topo, result, 0.001).unwrap();
                assert!(
                    (got - want).abs() <= 1e-7 * want,
                    "{label}: volume {got} against {want}"
                );
            }
        }
    }
}

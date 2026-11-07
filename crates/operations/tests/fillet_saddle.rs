//! A fillet round the saddle where a rod stands on a bar across its axis. The
//! edge is one closed curve on two cylinders, and neither face's normal lies
//! square to the edge away from its symmetry planes, so the ball's section
//! must lean with its own path to stay tangent to both.
//!
//! The ball's centre has a closed form in the two symmetry planes. In `y = 0`
//! the bar's top and the rod's side meet square, so the centre stands `r` off
//! each, at `(10 + r, 0, 15 + r)`. In `x = 0` it lies `r` off the rod's side and
//! `15 + r` from the bar's axis, at `(0, 10 + r, sqrt((15 + r)^2 - (10 + r)^2))`.
//! Between the ball and the corner the fillet fills; past the ball it is air.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::FRAC_PI_2;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;

    /// A bar of radius 15 along x and a rod of radius 10 standing on it, and
    /// the edge between them.
    fn saddle(topo: &mut Topology) -> (SolidId, EdgeId) {
        let bar = make_cylinder(topo, 15.0, 80.0).unwrap();
        let along_x = Mat4::translation(-40.0, 0.0, 0.0) * Mat4::rotation_y(FRAC_PI_2);
        transform_solid(topo, bar, &along_x).unwrap();
        let rod = make_cylinder(topo, 10.0, 40.0).unwrap();
        let solid = boolean(topo, BooleanOp::Fuse, bar, rod).unwrap();
        let on_both: Vec<EdgeId> = solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let (p, q) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                let (t0, t1) = edge.curve().domain_with_endpoints(p, q);
                let m = edge.curve().evaluate_with_endpoints(0.5 * (t0 + t1), p, q);
                (m.y().hypot(m.z()) - 15.0).abs() < 1e-3 && (m.x().hypot(m.y()) - 10.0).abs() < 1e-3
            })
            .collect();
        assert_eq!(on_both.len(), 1);
        (solid, on_both[0])
    }

    /// Points in the corner, and clear of the ball's reach, in `y = 0` and `x = 0`.
    fn expectations(r: f64) -> Vec<(Point3, PointClassification)> {
        let mut points = Vec::new();
        let mut probe = |center: Point3, corner: Point3| {
            let toward = (corner - center).normalize().unwrap();
            let reach = (corner - center).length();
            // Past the ball's surface, short of the corner: the fillet's material.
            points.push((
                center + toward * (r + 0.5 * (reach - r)),
                PointClassification::Inside,
            ));
            // Within the ball: air.
            points.push((center + toward * (0.8 * r), PointClassification::Outside));
        };
        probe(
            Point3::new(10.0 + r, 0.0, 15.0 + r),
            Point3::new(10.0, 0.0, 15.0),
        );
        probe(
            Point3::new(
                0.0,
                10.0 + r,
                ((15.0 + r).powi(2) - (10.0 + r).powi(2)).sqrt(),
            ),
            Point3::new(0.0, 10.0, (225.0_f64 - 100.0).sqrt()),
        );
        points
    }

    #[test]
    fn a_saddle_fillet_rolls_a_true_ball() {
        for r in [1.0, 3.0] {
            let mut topo = Topology::new();
            let (solid, edge) = saddle(&mut topo);
            let before = solid_volume(&topo, solid, 0.001).unwrap();
            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
            assert_eq!(result.succeeded, vec![edge], "r = {r}");
            let solid = result.solid;

            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "r = {r}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, solid, 0.01).unwrap();
            assert!(is_watertight(&mesh), "r = {r}: open mesh");
            let volume = solid_volume(&topo, solid, 0.001).unwrap();
            let oriented = oriented_solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - oriented).abs() < 1e-3 * volume,
                "r = {r}: {volume} and {oriented}"
            );
            assert!(
                volume > before,
                "r = {r}: a fillet on a concave edge adds material"
            );

            for (p, expected) in expectations(r) {
                let read = classify_point(&topo, solid, p, 0.01, 1e-7).unwrap();
                assert_eq!(read, expected, "r = {r}: {p:?}");
            }
        }
    }

    /// The fill scales near the radius squared: a test against the edge's
    /// length and a corner area of `(1 - pi/4) r^2` at a right angle would hold
    /// only where the two faces meet square, which they do only in `y = 0`.
    #[test]
    fn a_saddle_fillet_fills_more_with_a_larger_ball() {
        let mut fills = Vec::new();
        for r in [0.5, 1.0, 2.0] {
            let mut topo = Topology::new();
            let (solid, edge) = saddle(&mut topo);
            let before = solid_volume(&topo, solid, 0.001).unwrap();
            let result = fillet_v2(&mut topo, solid, &[edge], r).unwrap();
            assert!(result.failed.is_empty(), "r = {r}: {:?}", result.failed);
            fills.push(solid_volume(&topo, result.solid, 0.001).unwrap() - before);
        }
        for pair in fills.windows(2) {
            let ratio = pair[1] / pair[0];
            assert!((3.0..5.5).contains(&ratio), "{fills:?}");
        }
    }
}

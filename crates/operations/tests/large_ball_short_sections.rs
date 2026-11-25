//! A square bar 20 wide through a ball of radius 1e4, slanted to the ball's
//! axis and off its centre. Each of the bar's walls cuts the ball's face in
//! an arc about 20 long, on a circle of radius near 1e4, at a latitude near
//! 55 degrees. The arc's samples lie within 1e-6 of a line in the ball's
//! (u, v), but that line strays thousandths off the arc in space, so the
//! arc's pcurve on the ball is a fit through its samples.
#![allow(missing_docs)]

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::primitives::{make_box, make_sphere};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;

    const RADIUS: f64 = 1e4;
    /// The bar's axis in its own frame, before the slant: `(AXIS_X, AXIS_Y, z)`.
    const AXIS_X: f64 = 37.0;
    const AXIS_Y: f64 = -23.0;

    fn slant() -> Mat4 {
        Mat4::rotation_z(0.3) * Mat4::rotation_y(0.6)
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        let turned = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
        [
            ("upright", Mat4::identity()),
            ("turned", turned),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0) * turned),
        ]
    }

    /// The point of the ball at `depth` below its face (above it, if
    /// negative), on the line along the bar's axis moved `(dx, dy)` across
    /// it, on the side the axis's `z` grows to.
    fn probe(pose: &Mat4, dx: f64, dy: f64, depth: f64) -> Point3 {
        let (x, y) = (AXIS_X + dx, AXIS_Y + dy);
        let z = ((RADIUS - depth).powi(2) - x * x - y * y).sqrt();
        pose.mul_point(slant().mul_point(Point3::new(x, y, z)))
    }

    #[test]
    fn a_slanted_bar_through_a_large_ball_cuts_a_clean_hole() {
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let bar = make_box(&mut topo, 20.0, 20.0, 3.0 * RADIUS).unwrap();
            transform_solid(
                &mut topo,
                bar,
                &(slant() * Mat4::translation(AXIS_X - 10.0, AXIS_Y - 10.0, -1.5 * RADIUS)),
            )
            .unwrap();
            for s in [ball, bar] {
                transform_solid(&mut topo, s, &pose).unwrap();
            }
            let cut = boolean_with_options(&mut topo, BooleanOp::Cut, ball, bar, opts).expect(name);
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let faces = solid_faces(&topo, cut).unwrap();
            let curved = faces
                .iter()
                .filter(|&&f| !topo.face(f).unwrap().surface().is_planar())
                .count();
            assert_eq!(
                (faces.len(), curved),
                (5, 1),
                "{name}: the ball's face and the bar's four walls"
            );
            let mesh = tessellate_solid(&topo, cut, 1.0).unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
            assert_eq!(
                non_manifold_edge_count(&mesh),
                0,
                "{name}: non-manifold mesh"
            );

            let at =
                |p: Point3| classify_point(&topo, cut, p, &ClassifyOptions::default()).unwrap();
            for (dx, dy, depth, want) in [
                (0.0, 0.0, 5.0, PointClassification::Outside),
                (8.0, -8.0, 0.5, PointClassification::Outside),
                (13.0, 0.0, 5.0, PointClassification::Inside),
                (0.0, -13.0, 5.0, PointClassification::Inside),
                (-11.0, 11.0, 0.5, PointClassification::Inside),
                (13.0, 0.0, -5.0, PointClassification::Outside),
            ] {
                let p = probe(&pose, dx, dy, depth);
                assert_eq!(at(p), want, "{name}: ({dx}, {dy}) at depth {depth}");
            }
        }
    }
}

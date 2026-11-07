//! A boolean's `unify_faces` merges a primitive sphere's two halves into one
//! face where every section leaves their shared equator alone: a cube's six
//! caps come off the ball, and the sphere is one face with six holes. The
//! merged face meshes closed, measures the same, and classifies points as the
//! two halves did.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::primitives::{make_box, make_sphere};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    const RADIUS: f64 = 25.0;
    /// A ball of radius 25 less the six caps past `|x|, |y|, |z| = 20`.
    fn cube_in_ball_volume() -> f64 {
        let cap = PI * (RADIUS - 20.0) * (RADIUS - 20.0) * (3.0 * RADIUS - (RADIUS - 20.0)) / 3.0;
        4.0 / 3.0 * PI * RADIUS * RADIUS * RADIUS - 6.0 * cap
    }

    fn sphere_faces(topo: &Topology, solid: SolidId) -> usize {
        solid_faces(topo, solid)
            .unwrap()
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
            .count()
    }

    fn simplified() -> BooleanOptions {
        BooleanOptions {
            unify_faces: true,
            ..BooleanOptions::default()
        }
    }

    fn as_built() -> BooleanOptions {
        BooleanOptions {
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - volume).abs() < 1e-6 * volume,
            "{what}: volume {exact}, expected {volume}"
        );
        let mesh = tessellate_solid(topo, solid, 0.05).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let meshed = oriented_solid_volume(topo, solid, 0.05).unwrap();
        assert!(
            (meshed - volume).abs() < 2e-3 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    /// The issue's cube within a ball: seven faces, in any pose.
    #[test]
    fn a_cube_within_a_ball_keeps_one_sphere_face() {
        for (name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_x(0.7) * Mat4::rotation_y(0.3)),
        ] {
            let mut topo = Topology::new();
            let cube = make_box(&mut topo, 40.0, 40.0, 40.0).unwrap();
            let centred = pose * Mat4::translation(-20.0, -20.0, -20.0);
            transform_solid(&mut topo, cube, &centred).unwrap();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let result =
                boolean_with_options(&mut topo, BooleanOp::Intersect, cube, ball, simplified())
                    .unwrap();

            assert_eq!(solid_faces(&topo, result).unwrap().len(), 7, "{name}");
            assert_eq!(sphere_faces(&topo, result), 1, "{name}");
            assert_sound(&topo, result, cube_in_ball_volume(), name);

            // In the ball's bulge past a face, and in the cap the cube took off.
            for (p, expected) in [
                (Point3::new(21.0, 5.0, 5.0), PointClassification::Outside),
                (Point3::new(18.0, 12.0, 8.0), PointClassification::Inside),
                (Point3::new(0.0, 0.0, 0.0), PointClassification::Inside),
                (Point3::new(16.0, 16.0, 16.0), PointClassification::Outside),
            ] {
                let at = pose.mul_point(p);
                let read = classify_point(&topo, result, at, 0.01, 1e-7).unwrap();
                assert_eq!(read, expected, "{name}: {p:?}");
            }
        }
    }

    /// A ball with one bite, two bites side by side or three: whatever holes it
    /// keeps, none opposite another, the merged face takes both of the axis's
    /// poles only where the holes leave them.
    #[test]
    fn a_ball_keeps_one_face_round_any_bites_clear_of_its_equator() {
        let bite = |topo: &mut Topology, at: (f64, f64, f64)| {
            let b = make_box(topo, 2.0, 2.0, 2.0).unwrap();
            transform_solid(topo, b, &Mat4::translation(at.0, at.1, at.2)).unwrap();
            b
        };
        for bites in [
            vec![(1.0, 2.0, 8.0)],
            vec![(1.0, 2.0, 8.0), (1.0, 2.0, -10.0)],
            vec![(1.0, 2.0, 8.0), (-5.0, -5.0, 5.0)],
            vec![(1.0, 2.0, 8.0), (-5.0, -5.0, 5.0), (-4.0, 6.0, -7.5)],
        ] {
            let mut topo = Topology::new();
            let mut solid = make_sphere(&mut topo, 10.0, 32).unwrap();
            for (i, &at) in bites.iter().enumerate() {
                let tool = bite(&mut topo, at);
                // Only the last cut simplifies, so its merge meets every hole.
                let options = if i + 1 == bites.len() {
                    simplified()
                } else {
                    as_built()
                };
                solid =
                    boolean_with_options(&mut topo, BooleanOp::Cut, solid, tool, options).unwrap();
            }
            assert_eq!(sphere_faces(&topo, solid), 1, "{bites:?}");
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{bites:?}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, solid, 0.02).unwrap();
            assert!(is_watertight(&mesh), "{bites:?}: open mesh");
            let exact = solid_volume(&topo, solid, 0.001).unwrap();
            let meshed = oriented_solid_volume(&topo, solid, 0.02).unwrap();
            assert!(
                (exact - meshed).abs() < 2e-3 * exact,
                "{bites:?}: {exact} exact, {meshed} meshed"
            );
        }
    }

    /// The merged face goes on to later booleans as the two halves do.
    #[test]
    fn a_later_cut_reads_the_merged_face_as_it_reads_the_halves() {
        let cut_after = |options: BooleanOptions| {
            let mut topo = Topology::new();
            let cube = make_box(&mut topo, 40.0, 40.0, 40.0).unwrap();
            transform_solid(&mut topo, cube, &Mat4::translation(-20.0, -20.0, -20.0)).unwrap();
            let ball = make_sphere(&mut topo, RADIUS, 32).unwrap();
            let solid =
                boolean_with_options(&mut topo, BooleanOp::Intersect, cube, ball, options).unwrap();
            let half = make_box(&mut topo, 100.0, 100.0, 100.0).unwrap();
            transform_solid(&mut topo, half, &Mat4::translation(5.0, -50.0, -50.0)).unwrap();
            let cut =
                boolean_with_options(&mut topo, BooleanOp::Cut, solid, half, as_built()).unwrap();
            solid_volume(&topo, cut, 0.01).unwrap()
        };
        let (merged, halves) = (cut_after(simplified()), cut_after(as_built()));
        assert!(
            (merged - halves).abs() < 1e-3 * halves,
            "merged {merged}, halves {halves}"
        );
    }

    /// A hole whose samples average outside it reads the axis wrong, so a U
    /// bitten from the ball's top leaves the two halves as they are, whatever
    /// the mesh then has to read.
    #[test]
    fn a_ball_with_a_u_shaped_bite_stays_sound() {
        let mut topo = Topology::new();
        let ball = make_sphere(&mut topo, 10.0, 32).unwrap();
        let block = make_box(&mut topo, 6.0, 6.0, 4.0).unwrap();
        transform_solid(&mut topo, block, &Mat4::translation(-3.0, -3.0, 7.0)).unwrap();
        let notch = make_box(&mut topo, 2.0, 5.0, 6.0).unwrap();
        transform_solid(&mut topo, notch, &Mat4::translation(-1.0, -1.5, 6.0)).unwrap();
        let tool =
            boolean_with_options(&mut topo, BooleanOp::Cut, block, notch, as_built()).unwrap();
        let solid =
            boolean_with_options(&mut topo, BooleanOp::Cut, ball, tool, simplified()).unwrap();
        assert_eq!(sphere_faces(&topo, solid), 2);
        let report = validate_solid(&topo, solid).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let mesh = tessellate_solid(&topo, solid, 0.02).unwrap();
        assert!(is_watertight(&mesh), "open mesh");
        let exact = solid_volume(&topo, solid, 0.001).unwrap();
        let meshed = oriented_solid_volume(&topo, solid, 0.02).unwrap();
        assert!(
            (exact - meshed).abs() < 2e-3 * exact,
            "{exact} exact, {meshed} meshed"
        );
    }
}

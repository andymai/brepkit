//! A whole torus inside a solid of several faces, where the classifier reads it
//! by casting rays: a tool of several tori, and a torus left as a cavity. A lone
//! torus takes the analytic classifier, which hid a ray reader that saw no
//! crossing on the face at all.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_algo::classifier::{RayCastGeoms, ray_parity_cached};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean, compound_cut};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_torus};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const MAJOR: f64 = 0.6;
    const MINOR: f64 = 0.2;

    fn torus_at(t: &mut Topology, x: f64, y: f64, z: f64) -> SolidId {
        let s = make_torus(t, MAJOR, MINOR, 16).unwrap();
        transform_solid(t, s, &Mat4::translation(x, y, z)).unwrap();
        s
    }

    /// Every ray from a point agrees on which side of a lone torus it lies. These
    /// read the rays themselves, not the analytic classifier that answers first.
    #[test]
    fn a_ray_crosses_a_whole_torus() {
        let mut t = Topology::default();
        let ring = torus_at(&mut t, 1.5, 1.5, 3.0);
        let geoms = RayCastGeoms::new(&t, ring).unwrap();
        let dirs = [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.48, 0.6, 0.64),
        ];
        for (name, point, inside) in [
            ("in the tube", Point3::new(1.5, 2.1, 2.9), true),
            (
                "in the tube, off centre",
                Point3::new(1.47, 2.05, 3.05),
                true,
            ),
            ("in the ring's hole", Point3::new(1.55, 1.45, 3.03), false),
            ("past the ring", Point3::new(3.0, 1.1, 3.07), false),
        ] {
            for dir in dirs {
                let (odd, _) = ray_parity_cached(&geoms, point, dir);
                assert_eq!(odd, inside, "{name} along {dir:?}");
            }
        }
    }

    /// Half of each tube, sunk in a plate's top face, every ring clear of the
    /// plate's sides.
    #[test]
    fn a_grid_of_tori_takes_every_cut() {
        for n in [2_u32, 3, 5] {
            let mut t = Topology::default();
            let plate = make_box(&mut t, 12.0, 12.0, 3.0).unwrap();
            let mut tools = Vec::new();
            for i in 0..n {
                for j in 0..n {
                    let (x, y) = (1.5 + 2.0 * f64::from(i), 1.5 + 2.0 * f64::from(j));
                    tools.push(torus_at(&mut t, x, y, 3.0));
                }
            }
            let result = compound_cut(&mut t, plate, &tools, BooleanOptions::default()).unwrap();

            let half = PI * MINOR * MINOR / 2.0 * (2.0 * PI * MAJOR);
            let cuts = n * n;
            let volume = solid_volume(&t, result, 0.001).unwrap();
            let truth = 432.0 - half * f64::from(cuts);
            assert!(
                (volume - truth).abs() < 1e-6,
                "{n} a side: volume {volume}, expected {truth}"
            );
            // The plate's 5 other faces and its top, and a band and a disc per ring.
            let faces = solid_faces(&t, result).unwrap().len();
            assert_eq!(faces as u32, 6 + 2 * cuts, "{n} a side");
            let report = validate_solid(&t, result).unwrap();
            assert!(report.is_valid(), "{n} a side: {:?}", report.issues);
            assert!(is_watertight(&tessellate_solid(&t, result, 0.02).unwrap()));
        }
    }

    /// A tool in a torus's cavity lies in the air, and takes nothing off.
    #[test]
    fn a_tool_in_a_torus_cavity_cuts_nothing() {
        let mut t = Topology::default();
        let block = make_box(&mut t, 4.0, 4.0, 4.0).unwrap();
        let ring = torus_at(&mut t, 2.0, 2.0, 2.0);
        let hollow = boolean(&mut t, BooleanOp::Cut, block, ring).unwrap();
        let before = solid_volume(&t, hollow, 0.001).unwrap();
        let truth = 64.0 - 2.0 * PI * PI * MAJOR * MINOR * MINOR;
        assert!((before - truth).abs() < 1e-6, "cavity: volume {before}");

        // A cube at the tube's centreline, its corners within the tube.
        let side = 0.1;
        let tool = make_box(&mut t, side, side, side).unwrap();
        let place = Mat4::translation(2.6 - side / 2.0, 2.0 - side / 2.0, 2.0 - side / 2.0);
        transform_solid(&mut t, tool, &place).unwrap();
        let result = boolean(&mut t, BooleanOp::Cut, hollow, tool).unwrap();

        let volume = solid_volume(&t, result, 0.001).unwrap();
        assert!(
            (volume - before).abs() < 1e-6,
            "volume {volume}, expected {before}"
        );
        assert_eq!(
            classify_point(&t, result, Point3::new(2.6, 2.0, 2.0), 0.01, 1e-7).unwrap(),
            PointClassification::Outside
        );
    }
}

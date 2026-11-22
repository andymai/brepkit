//! A four-pole stator stack, built as the brepjs playground example
//! `axial-fan` builds it: a ring from radius 4.6 to 7, four arms (boxes 7 x
//! 4 from x = 6.5 to 13.5) and four pole shoes (70 degree sectors of a ring
//! from radius 13 to 14.6), all from z = 7 to 14. The arms and the shoes are
//! turned 0, 90, 180 and 270 degrees about z, each shoe first turned -35
//! degrees so that it sits across its arm.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::compound_ops::fuse_all;
    use brepkit_operations::copy::copy_solid;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
    use brepkit_topology::compound::Compound;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const POSES: [&str; 3] = ["upright", "turned", "mirrored"];
    const QUARTERS: [f64; 4] = [0.0, 90.0, 180.0, 270.0];

    fn turn() -> Mat4 {
        Mat4::translation(5.0, -4.0, 3.0) * Mat4::rotation_x(0.43) * Mat4::rotation_z(0.71)
    }

    fn mirror_plane() -> (Point3, Vec3) {
        (Point3::new(-10.0, 3.0, 1.0), Vec3::new(0.2, 1.0, 0.4))
    }

    fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
        match pose {
            "turned" => {
                transform_solid(topo, solid, &turn()).unwrap();
                solid
            }
            "mirrored" => {
                let (at, normal) = mirror_plane();
                mirror(topo, solid, at, normal).unwrap()
            }
            _ => solid,
        }
    }

    fn placed(p: Point3, pose: &str) -> Point3 {
        match pose {
            "turned" => turn().mul_point(p),
            "mirrored" => {
                let (at, normal) = mirror_plane();
                let unit = normal.normalize().unwrap();
                p - unit * (2.0 * (p - at).dot(unit))
            }
            _ => p,
        }
    }

    /// A ring from radius `r0` to `r1` and z = 7 to 14, swept `sweep`
    /// degrees about z from the x axis.
    fn ring(topo: &mut Topology, r0: f64, r1: f64, sweep: f64) -> SolidId {
        let corners: Vec<Point3> = [(r0, 7.0), (r1, 7.0), (r1, 14.0), (r0, 14.0)]
            .iter()
            .map(|&(x, z)| Point3::new(x, 0.0, z))
            .collect();
        let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
        let profile = make_planar_face_from_wire(topo, wire).unwrap();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        revolve(topo, profile, origin, z, sweep.to_radians()).unwrap()
    }

    fn turned_copy(topo: &mut Topology, solid: SolidId, degrees: f64) -> SolidId {
        let copy = copy_solid(topo, solid).unwrap();
        transform_solid(topo, copy, &Mat4::rotation_z(degrees.to_radians())).unwrap();
        copy
    }

    /// The hub ring, then the four arms, then the four shoes.
    fn parts(topo: &mut Topology) -> Vec<SolidId> {
        let hub = ring(topo, 4.6, 7.0, 360.0);
        let arm = make_box(topo, 7.0, 4.0, 7.0).unwrap();
        transform_solid(topo, arm, &Mat4::translation(6.5, -2.0, 7.0)).unwrap();
        let sector = ring(topo, 13.0, 14.6, 70.0);
        let shoe = turned_copy(topo, sector, -35.0);
        let mut all = vec![hub];
        all.extend(QUARTERS.iter().map(|&a| turned_copy(topo, arm, a)));
        all.extend(QUARTERS.iter().map(|&a| turned_copy(topo, shoe, a)));
        all
    }

    /// The area under a circle of radius `r` over |y| <= `a`.
    fn under_circle(r: f64, a: f64) -> f64 {
        a * (r * r - a * a).sqrt() + r * r * (a / r).asin()
    }

    fn holds(topo: &Topology, solid: SolidId, p: Point3) -> bool {
        matches!(
            classify_point(topo, solid, p, 0.01, 1e-7).unwrap(),
            PointClassification::Inside
        )
    }

    fn check(topo: &Topology, solid: SolidId, faces: usize, want: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: invalid: {report:?}");
        let n = solid_faces(topo, solid).unwrap().len();
        assert_eq!(n, faces, "{what}: {n} faces");
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let got = solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (got - want).abs() < 1e-6 * want,
            "{what}: volume {got}, want {want}"
        );
    }

    /// Each arm fuses exactly with its shoe in every pose: 12 faces, valid
    /// and watertight, with the closed-form volume, at all four quarters.
    #[test]
    fn each_arm_fuses_with_its_pole_shoe() {
        let arm = 7.0 * 4.0 * 7.0;
        let shoe = 70.0 / 360.0 * PI * (14.6 * 14.6 - 13.0 * 13.0) * 7.0;
        let overlap = 7.0 * (13.5 * 4.0 - under_circle(13.0, 2.0));
        let exact = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for pose in POSES {
            let mut topo = Topology::new();
            let all = parts(&mut topo);
            for (k, deg) in QUARTERS.iter().enumerate() {
                let a = posed(&mut topo, all[1 + k], pose);
                let b = posed(&mut topo, all[5 + k], pose);
                let what = format!("{pose} {deg}");
                let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, a, b, exact)
                    .map_err(|e| format!("{what}: {e}"))
                    .unwrap();
                check(&topo, fused, 12, arm + shoe - overlap, &what);
            }
        }
    }

    /// The nine parts fuse in one `fuse_all` with no mesh fallback in every
    /// pose. The result is valid and watertight, has 67 faces and the
    /// closed-form volume, and holds the hub, each arm and each shoe but
    /// not the gaps between the arms.
    #[test]
    fn the_stator_stack_fuses_exactly() {
        let hub = PI * (7.0 * 7.0 - 4.6 * 4.6) * 7.0;
        let arm = 7.0 * 4.0 * 7.0;
        let shoe = 70.0 / 360.0 * PI * (14.6 * 14.6 - 13.0 * 13.0) * 7.0;
        let arm_in_hub = 7.0 * (under_circle(7.0, 2.0) - 6.5 * 4.0);
        let arm_in_shoe = 7.0 * (13.5 * 4.0 - under_circle(13.0, 2.0));
        let want = hub + 4.0 * (arm + shoe - arm_in_hub - arm_in_shoe);
        for pose in POSES {
            let mut topo = Topology::new();
            let all: Vec<SolidId> = parts(&mut topo)
                .into_iter()
                .map(|s| posed(&mut topo, s, pose))
                .collect();
            let compound = topo.add_compound(Compound::new(all));
            let fused = fuse_all(&mut topo, compound)
                .map_err(|e| format!("{pose}: {e}"))
                .unwrap();
            check(&topo, fused, 67, want, pose);
            for deg in QUARTERS {
                let (c, s) = (deg.to_radians().cos(), deg.to_radians().sin());
                let at =
                    |x: f64, y: f64| placed(Point3::new(x * c - y * s, x * s + y * c, 10.5), pose);
                let what = format!("{pose} {deg}");
                assert!(holds(&topo, fused, at(5.8, 0.0)), "{what}: hub");
                assert!(holds(&topo, fused, at(10.0, 1.5)), "{what}: arm");
                assert!(holds(&topo, fused, at(13.2, 0.0)), "{what}: arm in shoe");
                assert!(holds(&topo, fused, at(13.7, 4.0)), "{what}: shoe");
                assert!(
                    !holds(&topo, fused, at(10.0, 2.5)),
                    "{what}: beside the arm"
                );
                assert!(
                    !holds(&topo, fused, at(7.07, 7.07)),
                    "{what}: between the arms"
                );
            }
        }
    }
}

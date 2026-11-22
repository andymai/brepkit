//! A rod with a slanted top, intersected with a half-space whose plane holds
//! the rod's axis. The rod is `make_cylinder(2, 5)` less a box whose bottom
//! face slants 0.3 rad through `(0, 0, 4)`, about x or about y, either way.
//! The half-space keeps `x >= 0`, `x <= 0`, `y >= 0` or `y <= 0`; the planes
//! `y = 0` hold the seam of the rod's wall. The slanted top's rim is one
//! closed ellipse: mirrored, the top face runs it against its curve, and
//! turned, the wall's seam and the half-space's ruling sit on different
//! copies of the wall's `u`.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

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

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    /// The rod less a box over the plane through `(0, 0, 4)` turned `angle`
    /// about x or y. The top is `z = 4 + g . (x, y)`, and `g` comes back too.
    fn slanted_rod(topo: &mut Topology, about_x: bool, angle: f64) -> (SolidId, (f64, f64)) {
        let rod = make_cylinder(topo, 2.0, 5.0).unwrap();
        let cutter = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let tilt = if about_x {
            Mat4::rotation_x(angle)
        } else {
            Mat4::rotation_y(angle)
        };
        let slant = Mat4::translation(0.0, 0.0, 4.0) * tilt * Mat4::translation(-5.0, -5.0, 0.0);
        transform_solid(topo, cutter, &slant).unwrap();
        let cut = boolean_with_options(topo, BooleanOp::Cut, rod, cutter, exact()).unwrap();
        assert_eq!(solid_faces(topo, cut).unwrap().len(), 3);
        let slope = if about_x {
            (0.0, angle.tan())
        } else {
            (-angle.tan(), 0.0)
        };
        (cut, slope)
    }

    /// A box 10 on a side with one face on the plane through the axis
    /// square to `d`, on the side `d` points to.
    fn half(topo: &mut Topology, d: (f64, f64)) -> SolidId {
        let h = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let at = match d {
            (x, _) if x > 0.5 => (0.0, -5.0),
            (x, _) if x < -0.5 => (-10.0, -5.0),
            (_, y) if y > 0.5 => (-5.0, 0.0),
            _ => (-5.0, -10.0),
        };
        transform_solid(topo, h, &Mat4::translation(at.0, at.1, -2.0)).unwrap();
        h
    }

    fn holds(topo: &Topology, solid: SolidId, p: Point3) -> bool {
        matches!(
            classify_point(topo, solid, p, 0.01, 1e-7).unwrap(),
            PointClassification::Inside
        )
    }

    /// Each half of the rod, for each slant, in every pose: exact, valid
    /// and watertight, four faces, and the volume of the half disc under the
    /// slanted top, `8 pi + (g . d) 16 / 3`. Points under the top on the
    /// kept side are in it; points over the top or on the other side are not.
    #[test]
    fn each_half_of_a_slanted_rod_is_exact() {
        let slants = [(false, -0.3), (false, 0.3), (true, -0.3), (true, 0.3)];
        let sides = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)];
        for (about_x, angle) in slants {
            for d in sides {
                for pose in POSES {
                    let what = format!(
                        "about {} {angle}, side {d:?}, {pose}",
                        if about_x { "x" } else { "y" }
                    );
                    let mut topo = Topology::new();
                    let (rod, g) = slanted_rod(&mut topo, about_x, angle);
                    let h = half(&mut topo, d);
                    let (a, b) = (posed(&mut topo, rod, pose), posed(&mut topo, h, pose));
                    let common =
                        boolean_with_options(&mut topo, BooleanOp::Intersect, a, b, exact())
                            .map_err(|e| format!("{what}: {e}"))
                            .unwrap();
                    let report = validate_solid(&topo, common).unwrap();
                    assert!(report.is_valid(), "{what}: invalid: {report:?}");
                    let faces = solid_faces(&topo, common).unwrap().len();
                    assert_eq!(faces, 4, "{what}: {faces} faces");
                    let mesh = tessellate_solid(&topo, common, 0.01).unwrap();
                    assert!(is_watertight(&mesh), "{what}: open mesh");
                    let rise = g.0 * d.0 + g.1 * d.1;
                    let want = 8.0 * PI + rise * 16.0 / 3.0;
                    let got = solid_volume(&topo, common, 0.01).unwrap();
                    assert!(
                        (got - want).abs() < 1e-6 * want,
                        "{what}: volume {got}, want {want}"
                    );
                    let at = |s: f64, z: f64| placed(Point3::new(s * d.0, s * d.1, z), pose);
                    let top = 4.0 + rise;
                    assert!(
                        holds(&topo, common, at(1.0, 1.0)),
                        "{what}: low on the kept side"
                    );
                    assert!(
                        holds(&topo, common, at(1.0, top - 0.3)),
                        "{what}: under the top"
                    );
                    assert!(
                        !holds(&topo, common, at(1.0, top + 0.3)),
                        "{what}: over the top"
                    );
                    assert!(
                        !holds(&topo, common, at(-1.0, 1.0)),
                        "{what}: on the other side"
                    );
                }
            }
        }
    }
}

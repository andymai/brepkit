//! A window cut through a tube's wall by a ring sector that holds the
//! whole wall over its sweep. The tube runs from radius 38 to 42 and from
//! z = 1 to 23; the window runs from z = 3 to 4.5. The bore's piece inside
//! the window is bounded by one loop of sections, and it must be read on
//! the bore: the loop's 3D centroid lies near the axis, outside the ring,
//! and a window over half a turn wide then kept that piece as an open
//! growth shell.

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
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::{make_planar_face_from_wire, make_polygon_wire};
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

    /// The tube, its bore's seam turned `bore_turn` about z.
    fn tube(topo: &mut Topology, bore_turn: f64) -> SolidId {
        let outer = make_cylinder(topo, 42.0, 22.0).unwrap();
        transform_solid(topo, outer, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
        let bore = make_cylinder(topo, 38.0, 24.0).unwrap();
        transform_solid(topo, bore, &Mat4::rotation_z(bore_turn)).unwrap();
        boolean_with_options(topo, BooleanOp::Cut, outer, bore, exact()).unwrap()
    }

    /// A ring sector from radius `r0` to `r1` and z = 3 to 4.5, swept
    /// `sweep` degrees about z from `start` degrees.
    fn sector(topo: &mut Topology, r0: f64, r1: f64, sweep: f64, start: f64) -> SolidId {
        let corners: Vec<Point3> = [(r0, 3.0), (r1, 3.0), (r1, 4.5), (r0, 4.5)]
            .iter()
            .map(|&(x, z)| Point3::new(x, 0.0, z))
            .collect();
        let wire = make_polygon_wire(topo, &corners, 1e-7).unwrap();
        let profile = make_planar_face_from_wire(topo, wire).unwrap();
        let origin = Point3::new(0.0, 0.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);
        let ring = revolve(topo, profile, origin, z, sweep.to_radians()).unwrap();
        transform_solid(topo, ring, &Mat4::rotation_z(start.to_radians())).unwrap();
        ring
    }

    fn holds(topo: &Topology, solid: SolidId, p: Point3) -> bool {
        matches!(
            classify_point(topo, solid, p, 0.01, 1e-7).unwrap(),
            PointClassification::Inside
        )
    }

    /// Cut, intersect and fuse the tube with the sector in every pose. Each
    /// result is exact, valid and watertight, has `faces[op]` faces, and holds
    /// the closed-form volume. Points at mid-height of the window classify
    /// against the sector's sweep.
    fn window_is_exact(
        bore_turn: f64,
        r0: f64,
        r1: f64,
        sweep: f64,
        start: f64,
        faces: [usize; 3],
    ) {
        let annulus = |a: f64, b: f64| PI * (b * b - a * a);
        let tube_volume = annulus(38.0, 42.0) * 22.0;
        let common = sweep / 360.0 * annulus(r0.max(38.0), r1.min(42.0)) * 1.5;
        let sector_volume = sweep / 360.0 * annulus(r0, r1) * 1.5;
        let ops = [
            (BooleanOp::Cut, tube_volume - common),
            (BooleanOp::Intersect, common),
            (BooleanOp::Fuse, tube_volume + sector_volume - common),
        ];
        for pose in POSES {
            for (k, (op, want)) in ops.into_iter().enumerate() {
                let mut topo = Topology::new();
                let t = tube(&mut topo, bore_turn);
                let s = sector(&mut topo, r0, r1, sweep, start);
                let (a, b) = (posed(&mut topo, t, pose), posed(&mut topo, s, pose));
                let result = boolean_with_options(&mut topo, op, a, b, exact())
                    .map_err(|e| format!("{pose} {op:?}: {e}"))
                    .unwrap();
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{pose} {op:?}: invalid: {report:?}");
                let n = solid_faces(&topo, result).unwrap().len();
                assert_eq!(n, faces[k], "{pose} {op:?}: {n} faces");
                let mesh = tessellate_solid(&topo, result, 0.05).unwrap();
                assert!(is_watertight(&mesh), "{pose} {op:?}: open mesh");
                let got = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (got - want).abs() < 1e-6 * want,
                    "{pose} {op:?}: volume {got}, want {want}"
                );
                // At mid-height of the window, round the bore's radius 40.
                for step in 0..36 {
                    let deg = f64::from(step) * 10.0 + 5.0;
                    let rad = deg.to_radians();
                    let p = placed(Point3::new(40.0 * rad.cos(), 40.0 * rad.sin(), 3.75), pose);
                    let in_window = (deg - start).rem_euclid(360.0) < sweep;
                    let want_in = match op {
                        BooleanOp::Cut => !in_window || r1 < 40.0,
                        BooleanOp::Intersect => in_window && r1 > 40.0,
                        BooleanOp::Fuse => true,
                    };
                    assert_eq!(
                        holds(&topo, result, p),
                        want_in,
                        "{pose} {op:?}: at {deg} degrees"
                    );
                }
            }
        }
    }

    /// A window 200 degrees wide from -154 degrees, with the bore's seam
    /// outside it.
    #[test]
    fn a_window_over_half_a_turn_wide_cuts_exactly() {
        window_is_exact(PI, 36.0, 43.0, 200.0, -154.0, [8, 6, 14]);
    }

    /// A window 190 degrees wide from -150 degrees.
    #[test]
    fn a_window_190_degrees_wide_cuts_exactly() {
        window_is_exact(PI, 36.0, 43.0, 190.0, -150.0, [8, 6, 14]);
    }

    /// A groove 200 degrees wide that stops short of the outer wall: the
    /// sector runs from radius 37 to 41.
    #[test]
    fn a_groove_over_half_a_turn_wide_cuts_exactly() {
        window_is_exact(PI, 37.0, 41.0, 200.0, -154.0, [9, 6, 9]);
    }

    /// A window 120 degrees wide, and the 200 degree window with the bore's
    /// seam inside it.
    #[test]
    fn narrower_windows_and_a_window_over_the_seam_cut_exactly() {
        window_is_exact(PI, 36.0, 43.0, 120.0, -60.0, [8, 6, 14]);
        window_is_exact(0.0, 36.0, 43.0, 200.0, -154.0, [8, 6, 14]);
    }
}

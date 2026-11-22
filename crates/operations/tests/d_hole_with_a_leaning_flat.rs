//! A 24 x 24 block less a D-shaped hole: the part of a rod of radius 10
//! about the `z` axis on the axis's side of a flat `y = 5 - 5e-8 z`. The
//! flat meets the rod's wall in an ellipse of semi-major axis 2e8, centred
//! 1e8 along the axis. The hole uses two arcs of it, each under a millionth
//! of a radian long.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    const WALL: f64 = 10.0;
    const FLAT: f64 = 5.0;
    const TILT: f64 = 5e-8;

    /// The area of the rod's section on the axis's side of `y = y0`.
    fn d_area(y0: f64) -> f64 {
        let segment = (WALL * WALL).mul_add(
            (y0 / WALL).acos(),
            -y0 * WALL.mul_add(WALL, -y0 * y0).sqrt(),
        );
        PI.mul_add(WALL * WALL, -segment)
    }

    /// The block `length` long less the hole, its flat at `y = 5 - tilt z`.
    fn block_volume(length: f64, tilt: f64) -> f64 {
        const STEPS: u32 = 50;
        let h = length / f64::from(STEPS);
        let hole: f64 = (0..STEPS)
            .map(|k| d_area((-tilt).mul_add(h * (f64::from(k) + 0.5), FLAT)) * h)
            .sum();
        24.0f64.mul_add(24.0 * length, -hole)
    }

    /// The block, the rod 20 longer, and the box that keeps the axis's side
    /// of the flat, all placed by `pose`.
    fn parts(topo: &mut Topology, length: f64, tilt: f64, pose: &Mat4) -> [SolidId; 3] {
        let block = make_box(topo, 24.0, 24.0, length).unwrap();
        transform_solid(topo, block, &Mat4::translation(-12.0, -12.0, 0.0)).unwrap();
        let rod = make_cylinder(topo, WALL, length + 20.0).unwrap();
        transform_solid(topo, rod, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
        let keep = make_box(topo, 2.0 * WALL, 2.0 * WALL, length + 40.0).unwrap();
        let place =
            Mat4::rotation_x(tilt.atan()) * Mat4::translation(-WALL, FLAT - 2.0 * WALL, -20.0);
        transform_solid(topo, keep, &place).unwrap();
        for s in [block, rod, keep] {
            transform_solid(topo, s, pose).unwrap();
        }
        [block, rod, keep]
    }

    fn check(length: f64, name: &str, pose: &Mat4) {
        let what = format!("length {length}, {name}");
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let mut topo = Topology::new();
        let [block, rod, keep] = parts(&mut topo, length, TILT, pose);
        let d = boolean::boolean_with_options(&mut topo, BooleanOp::Intersect, rod, keep, opts)
            .map_err(|e| format!("{what}: intersect {e}"))
            .unwrap();
        let report = validate_solid(&topo, d).unwrap();
        assert!(report.is_valid(), "{what}: D rod {:?}", report.issues);
        let faces = solid_faces(&topo, d).unwrap();
        assert_eq!(faces.len(), 4, "{what}: D rod faces");
        assert!(
            faces
                .iter()
                .any(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_))),
            "{what}: D rod has no wall"
        );

        let holed = boolean::boolean_with_options(&mut topo, BooleanOp::Cut, block, d, opts)
            .map_err(|e| format!("{what}: cut {e}"))
            .unwrap();
        let report = validate_solid(&topo, holed).unwrap();
        assert!(report.is_valid(), "{what}: block {:?}", report.issues);
        assert_eq!(solid_faces(&topo, holed).unwrap().len(), 8, "{what}: faces");
        // The measure reads this block up to 3e-7 of the whole off; the
        // mesh boolean's block reads 1.1e-3 off.
        let expected = block_volume(length, TILT);
        let v = solid_volume(&topo, holed, 1e-3).unwrap();
        assert!(
            (v - expected).abs() < 1e-6 * expected,
            "{what}: volume {v}, expected {expected}"
        );
        let mesh = tessellate_solid(&topo, holed, 0.05).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{what}: open mesh");
        assert_eq!(non_manifold_edge_count(&mesh), 0, "{what}: non-manifold");
        let z = 0.5 * length;
        for (p, want) in [
            (Point3::new(0.0, 0.0, z), PointClassification::Outside),
            (Point3::new(-6.0, -6.0, z), PointClassification::Outside),
            (Point3::new(0.0, 8.0, z), PointClassification::Inside),
            (Point3::new(11.0, 0.0, z), PointClassification::Inside),
            (Point3::new(0.0, -11.0, z), PointClassification::Inside),
        ] {
            let got = classify_point(&topo, holed, pose.mul_point(p), 0.01, 1e-7).unwrap();
            assert_eq!(got, want, "{what} at {p:?}");
        }
    }

    #[test]
    fn a_short_d_hole_whose_flat_leans_off_the_bore_cuts_exactly() {
        for length in [60.0, 100.0] {
            for (name, pose) in [
                ("upright", Mat4::identity()),
                ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
                ("spun about the bore", Mat4::rotation_z(0.7)),
                (
                    "moved",
                    Mat4::translation(3.0, -2.0, 7.0) * Mat4::rotation_z(2.0),
                ),
                ("upside down", Mat4::rotation_x(PI)),
                ("on its side", Mat4::rotation_y(FRAC_PI_2)),
            ] {
                check(length, name, &pose);
            }
        }
    }

    /// Turned obliquely, `Ellipse3D::project` takes the angle of a point's
    /// image on the ellipse's major circle, not the angle of its nearest
    /// point. With the rounding of the centre 1e8 away, the crossings of
    /// the flat's edges with the wall then read up to 0.09 off the curve,
    /// they are dropped, and the flat is not split.
    #[test]
    #[ignore = "Ellipse3D::project reads a turned long thin ellipse's crossings off the curve"]
    fn a_short_d_hole_cuts_exactly_turned_obliquely() {
        let pose = Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3);
        for length in [100.0, 200.0] {
            check(length, "turned", &pose);
        }
    }
}

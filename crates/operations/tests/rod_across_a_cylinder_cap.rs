//! A cylinder fused with a rod across it that stands above its top cap:
//! the rod's wall crosses the cap along two chords of the rod's section.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;

    const WALL_SPIN: f64 = 2.231_596_762_531_308_5;
    /// The rod runs along x through `(y, z)`.
    const ROD_AT: (f64, f64) = (0.444_106_566_867_297_2, 3.856_028_824_387_512);
    const ROD_RADIUS: f64 = 0.237_645_286_841_210_7;
    const ROD_SPIN: f64 = 0.472_572_851_806_060_2;

    fn poses() -> Vec<(&'static str, Mat4)> {
        let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1);
        vec![
            ("upright", Mat4::identity()),
            ("turned", turn),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
            ("turned and mirrored", Mat4::scale(1.0, -1.0, 1.0) * turn),
            ("rolled about z", Mat4::rotation_z(2.0)),
        ]
    }

    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: u32) -> f64 {
        let h = (b - a) / f64::from(n);
        let mut s = f(a) + f(b);
        for i in 1..n {
            s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(a + h * f64::from(i));
        }
        s * h / 3.0
    }

    /// The unit cylinder from `z = 0` to 4 with the rod, 4 long, added. The
    /// two share, at each `x`, the part of the rod's section under the cap
    /// and inside the chord `|y| < sqrt(1 - x^2)`: read over `z` as the
    /// rod's chords at each height, cut there.
    fn truth() -> f64 {
        let (yc, zc) = ROD_AT;
        let r = ROD_RADIUS;
        let shared_at = |x: f64| {
            let w = (1.0 - x * x).max(0.0).sqrt();
            simpson(
                |z| {
                    let h = (r * r - (z - zc) * (z - zc)).max(0.0).sqrt();
                    ((yc + h).min(w) - (yc - h).max(-w)).max(0.0)
                },
                zc - r,
                (zc + r).min(4.0),
                4000,
            )
        };
        let reach = (1.0 - (yc - r) * (yc - r)).sqrt();
        let shared = simpson(shared_at, -reach, reach, 4000);
        4.0 * PI + PI * r * r * 4.0 - shared
    }

    /// The fuse, in each pose: an exact solid of cylinder and plane faces
    /// whose holes lie where they belong, valid, meshed closed, at its
    /// volume. The cap's hole, the rod's section between the two chords,
    /// had crossed the cap's outline, and the mesh had 6 to 24 open edges.
    #[test]
    fn a_rod_across_a_cylinder_cap_fuses_exactly() {
        let truth = truth();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let wall = make_cylinder(&mut topo, 1.0, 4.0).unwrap();
            transform_solid(&mut topo, wall, &(pose * Mat4::rotation_z(WALL_SPIN))).unwrap();
            let rod = make_cylinder(&mut topo, ROD_RADIUS, 4.0).unwrap();
            let along_z = Mat4::translation(ROD_AT.0, ROD_AT.1, -2.0) * Mat4::rotation_z(ROD_SPIN);
            let along_x = Mat4::rotation_z(PI / 2.0) * Mat4::rotation_x(PI / 2.0) * along_z;
            transform_solid(&mut topo, rod, &(pose * along_x)).unwrap();

            let options = BooleanOptions {
                mesh_fallback: false,
                unify_faces: false,
                ..BooleanOptions::default()
            };
            let fused =
                boolean_with_options(&mut topo, BooleanOp::Fuse, wall, rod, options).unwrap();

            for face in solid_faces(&topo, fused).unwrap() {
                let surface = topo.face(face).unwrap().surface();
                assert!(
                    matches!(
                        surface,
                        FaceSurface::Cylinder(_) | FaceSurface::Plane { .. }
                    ),
                    "{name}: a {} face",
                    surface.type_tag()
                );
            }
            let report = validate_solid(&topo, fused).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
            assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
            let volume = solid_volume(&topo, fused, 0.0005).unwrap();
            assert!(
                ((volume - truth) / truth).abs() < 1e-5,
                "{name}: volume {volume} against {truth}"
            );
        }
    }
}

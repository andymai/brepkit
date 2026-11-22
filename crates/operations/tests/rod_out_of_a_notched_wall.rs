//! A cylinder with a tunnel across its foot, which notches its wall's lower
//! rim, fused with a thin rod that stands out of the wall beside its seam.

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
    use brepkit_topology::solid::SolidId;

    const WALL_SPIN: f64 = 0.016_424_844_383_926_298;
    /// The tunnel runs along y through `(z, x)`.
    const TUNNEL_AT: (f64, f64) = (0.395_505_750_462_132_7, -0.084_286_726_986_464_89);
    const TUNNEL_RADIUS: f64 = 0.467_896_913_653_425_9;
    const TUNNEL_SPIN: f64 = 5.836_856_302_813_882_5;
    /// The rod runs along z through `(x, y)`, from `ROD_FROM` to `ROD_TO`.
    const ROD_AT: (f64, f64) = (0.957_434_371_991_607_6, -0.060_758_778_833_065_24);
    const ROD_RADIUS: f64 = 0.082_590_283_998_385_83;
    const ROD_FROM: f64 = 2.996_688_302_672_595_2;
    const ROD_TO: f64 = 3.254_711_200_931_275_5;
    const ROD_SPIN: f64 = 0.378_006_659_164_070_6;

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        }
    }

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

    fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let n = 20_000;
        let h = (b - a) / f64::from(n);
        let mut s = f(a) + f(b);
        for i in 1..n {
            s += if i % 2 == 1 { 4.0 } else { 2.0 } * f(a + h * f64::from(i));
        }
        s * h / 3.0
    }

    /// The area two discs of radii `a` and `b`, `d` apart, share.
    fn lens(a: f64, b: f64, d: f64) -> f64 {
        let p = a
            * a
            * ((d * d + a * a - b * b) / (2.0 * d * a))
                .clamp(-1.0, 1.0)
                .acos();
        let q = b
            * b
            * ((d * d + b * b - a * a) / (2.0 * d * b))
                .clamp(-1.0, 1.0)
                .acos();
        let r = 0.5 * ((-d + a + b) * (d + a - b) * (d - a + b) * (d + a + b)).sqrt();
        p + q - r
    }

    /// The unit cylinder from `z = 0` to 4, less the tunnel when `tunnel`,
    /// with the rod's part outside the wall added. The tunnel takes, at
    /// each `x` across it, its section's height above `z = 0` times the
    /// chord `2 sqrt(1 - x^2)` of the unit disc.
    fn truth(tunnel: bool) -> f64 {
        let (zc, xc) = TUNNEL_AT;
        let taken = simpson(
            |x| {
                let half = (TUNNEL_RADIUS * TUNNEL_RADIUS - (x - xc) * (x - xc))
                    .max(0.0)
                    .sqrt();
                let height = ((zc + half) - (zc - half).max(0.0)).max(0.0);
                height * 2.0 * (1.0 - x * x).max(0.0).sqrt()
            },
            xc - TUNNEL_RADIUS,
            xc + TUNNEL_RADIUS,
        );
        let height = ROD_TO - ROD_FROM;
        let d = ROD_AT.0.hypot(ROD_AT.1);
        let outside = (PI * ROD_RADIUS * ROD_RADIUS - lens(1.0, ROD_RADIUS, d)) * height;
        4.0 * PI - if tunnel { taken } else { 0.0 } + outside
    }

    fn build(topo: &mut Topology, pose: &Mat4) -> (SolidId, SolidId, SolidId) {
        let wall = make_cylinder(topo, 1.0, 4.0).unwrap();
        transform_solid(topo, wall, &Mat4::rotation_z(WALL_SPIN)).unwrap();
        let tunnel = make_cylinder(topo, TUNNEL_RADIUS, 4.0).unwrap();
        let along_z =
            Mat4::translation(TUNNEL_AT.0, TUNNEL_AT.1, -2.0) * Mat4::rotation_z(TUNNEL_SPIN);
        let along_y = Mat4::rotation_x(-PI / 2.0) * Mat4::rotation_z(-PI / 2.0) * along_z;
        transform_solid(topo, tunnel, &along_y).unwrap();
        let rod = make_cylinder(topo, ROD_RADIUS, ROD_TO - ROD_FROM).unwrap();
        let place = Mat4::translation(ROD_AT.0, ROD_AT.1, ROD_FROM) * Mat4::rotation_z(ROD_SPIN);
        transform_solid(topo, rod, &place).unwrap();
        for s in [wall, tunnel, rod] {
            transform_solid(topo, s, pose).unwrap();
        }
        (wall, tunnel, rod)
    }

    /// The rod fused on, with and without the tunnel, in each pose: an
    /// exact solid of cylinder and plane faces, valid, meshed closed, at its
    /// volume. The rod's sides cross the wall along two rulings beside the
    /// wall's seam. With the notched rim, the wall's outline is no band, and
    /// the clip read the rulings against chords of the rim arcs, which do
    /// not meet them, and dropped them. The wall's outline in `(u, v)` was
    /// unwrapped about a mean of its samples, which put the seam's side of
    /// the wall outside it. The wall kept no hole, and the fuse was refused.
    /// On an earlier engine it returned a solid 2.2e-3 too large, its faces
    /// running the same way along 6 edges.
    #[test]
    fn a_rod_stands_out_of_a_notched_wall_exactly() {
        for tunnel in [true, false] {
            let truth = truth(tunnel);
            for (name, pose) in poses() {
                let label = format!("tunnel {tunnel}, {name}");
                let mut topo = Topology::new();
                let (wall, hole, rod) = build(&mut topo, &pose);
                let base = if tunnel {
                    boolean_with_options(&mut topo, BooleanOp::Cut, wall, hole, exact()).unwrap()
                } else {
                    wall
                };
                let fused =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, base, rod, exact()).unwrap();

                for face in solid_faces(&topo, fused).unwrap() {
                    let surface = topo.face(face).unwrap().surface();
                    assert!(
                        matches!(
                            surface,
                            FaceSurface::Cylinder(_) | FaceSurface::Plane { .. }
                        ),
                        "{label}: a {} face",
                        surface.type_tag()
                    );
                }
                let report = validate_solid(&topo, fused).unwrap();
                assert!(report.is_valid(), "{label}: {:?}", report.issues);
                let mesh = tessellate_solid(&topo, fused, 0.01).unwrap();
                assert_eq!(boundary_edge_count(&mesh), 0, "{label}: open mesh");
                let volume = solid_volume(&topo, fused, 0.0005).unwrap();
                assert!(
                    ((volume - truth) / truth).abs() < 1e-5,
                    "{label}: volume {volume} against {truth}"
                );
            }
        }
    }
}

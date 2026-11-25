//! A 10 x 10 x 2 plate, a rod of radius 2 about (5, 5), and an elliptic
//! prism of semi-axes 3 along x and `sqrt(5.4)` along y about (4, 5). The rod
//! and the prism run from z = -1 to 3. Their walls cross along two lines, at
//! (6, 5 -/+ sqrt(3)), and touch along a third, at (7, 5), where the ellipse
//! lies inside the circle: between the walls, beside the touch, two thin
//! lenses lie inside the rod and outside the prism.
//!
//! Each boolean runs upright, turned and mirrored, with the seams of both
//! walls on the touching line and away from it.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Ellipse3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::compound_ops::fuse_all;
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::compound::Compound;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "mirrored",
                Mat4::translation(1.0, 0.5, -2.0) * Mat4::scale(-1.0, 1.0, 1.0),
            ),
        ]
    }

    /// Where the walls' seams lie: the rod's turned about its axis from +x,
    /// and the prism's at that parameter of its ellipse. Both at zero lie on
    /// the touching line.
    const SEAMS: [(&str, f64); 2] = [("seams on the touch", 0.0), ("seams away", 2.0)];

    /// The area inside the circle or the ellipse: the ellipse, and the
    /// circle's segment past the chord x = 6 less the ellipse's.
    fn union_area() -> f64 {
        let b = 5.4_f64.sqrt();
        let s0 = (2.0_f64 / 3.0).acos();
        let circle_segment = 2.0 * (2.0 * PI / 3.0 - (2.0 * PI / 3.0).sin());
        let ellipse_segment = 0.5 * 3.0 * b * (2.0 * s0 - (2.0 * s0).sin());
        PI * 3.0 * b + circle_segment - ellipse_segment
    }

    /// The plate, the rod and the prism, placed by `pose`.
    fn parts(topo: &mut Topology, seam: f64, pose: &Mat4) -> [SolidId; 3] {
        let plate = make_box(topo, 10.0, 10.0, 2.0).unwrap();
        let rod = make_cylinder(topo, 2.0, 4.0).unwrap();
        transform_solid(
            topo,
            rod,
            &(Mat4::translation(5.0, 5.0, -1.0) * Mat4::rotation_z(seam)),
        )
        .unwrap();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let ellipse = Ellipse3D::with_axes(
            Point3::new(4.0, 5.0, -1.0),
            up,
            3.0,
            5.4_f64.sqrt(),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(ellipse.evaluate(seam), 1e-7));
        let rim = topo.add_edge(Edge::new(v, v, EdgeCurve::Ellipse(ellipse)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let base = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: up,
                d: -1.0,
            },
        ));
        let prism = extrude(topo, base, up, 4.0).unwrap();
        for solid in [plate, rod, prism] {
            transform_solid(topo, solid, pose).unwrap();
        }
        [plate, rod, prism]
    }

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    /// That `solid` is valid and closed, within 1e-9 of `volume`, and holds
    /// each of `inside` and none of `outside`, placed by `pose`.
    fn assert_exact(
        topo: &Topology,
        solid: SolidId,
        volume: f64,
        (inside, outside): (&[Point3], &[Point3]),
        pose: &Mat4,
        what: &str,
    ) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let measured = solid_volume(topo, solid, 1e-4).unwrap();
        assert!(
            (measured - volume).abs() <= 1e-9 * volume,
            "{what}: volume {measured}, expected {volume}"
        );
        assert!(
            is_watertight(&tessellate_solid(topo, solid, 0.01).unwrap()),
            "{what}: open mesh"
        );
        for (points, want) in [
            (inside, PointClassification::Inside),
            (outside, PointClassification::Outside),
        ] {
            for &p in points {
                let read = classify_point(topo, solid, pose.mul_point(p), 0.01, 1e-7).unwrap();
                assert_eq!(read, want, "{what}: at {p:?}");
            }
        }
    }

    /// In a lens between the walls beside the touch: inside the rod, outside
    /// the prism. At y = 6 the ellipse reaches x = 6.708 and the circle 6.732.
    const LENS: Point3 = Point3::new(6.72, 6.0, 1.0);
    /// Just past the circle there.
    const PAST_LENS: Point3 = Point3::new(6.74, 6.0, 1.0);
    /// Inside both walls, by the touch.
    const BY_TOUCH: Point3 = Point3::new(6.9, 5.0, 1.0);

    #[test]
    fn a_rod_and_a_prism_touching_along_a_line_fuse_exactly() {
        let volume = 4.0 * union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let [_, rod, prism] = parts(&mut topo, seam, &pose);
                let what = format!("{seams}, {name}");
                let fused =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, rod, prism, exact()).unwrap();
                assert_exact(
                    &topo,
                    fused,
                    volume,
                    (
                        &[LENS, BY_TOUCH, Point3::new(1.5, 5.0, 2.0)],
                        &[
                            PAST_LENS,
                            Point3::new(7.1, 5.0, 1.0),
                            Point3::new(6.72, 6.0, 3.5),
                        ],
                    ),
                    &pose,
                    &what,
                );
                // The two caps, the rod's wall between the crossings, and the
                // prism's wall outside the rod in the pieces of its quarters.
                assert_eq!(solid_faces(&topo, fused).unwrap().len(), 7, "{what}");
            }
        }
    }

    #[test]
    fn a_plate_less_a_rod_and_a_prism_fused_is_exact() {
        let volume = 200.0 - 2.0 * union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let [plate, rod, prism] = parts(&mut topo, seam, &pose);
                let what = format!("{seams}, {name}");
                let tool =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, rod, prism, exact()).unwrap();
                let cut =
                    boolean_with_options(&mut topo, BooleanOp::Cut, plate, tool, exact()).unwrap();
                assert_exact(
                    &topo,
                    cut,
                    volume,
                    (&[PAST_LENS, Point3::new(9.0, 9.0, 1.0)], &[LENS, BY_TOUCH]),
                    &pose,
                    &what,
                );
            }
        }
    }

    /// The fuse of all three and the plate less the rod and then the prism
    /// give the exact solid or refuse, never a wrong one.
    #[test]
    fn the_three_way_fuse_and_the_two_cuts_are_exact_or_refused() {
        let u = union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let what = format!("{seams}, {name}");
                let mut topo = Topology::new();
                let [plate, rod, prism] = parts(&mut topo, seam, &pose);
                let compound = topo.add_compound(Compound::new(vec![plate, rod, prism]));
                if let Ok(fused) = fuse_all(&mut topo, compound) {
                    assert_exact(
                        &topo,
                        fused,
                        200.0 + 2.0 * u,
                        (
                            &[
                                Point3::new(6.72, 6.0, 2.5),
                                BY_TOUCH,
                                Point3::new(9.0, 9.0, 1.0),
                            ],
                            &[Point3::new(6.74, 6.0, 2.5)],
                        ),
                        &pose,
                        &format!("fuse, {what}"),
                    );
                }
                let mut topo = Topology::new();
                let [plate, rod, prism] = parts(&mut topo, seam, &pose);
                let cut = boolean_with_options(&mut topo, BooleanOp::Cut, plate, rod, exact())
                    .and_then(|c| {
                        boolean_with_options(&mut topo, BooleanOp::Cut, c, prism, exact())
                    });
                if let Ok(cut) = cut {
                    assert_exact(
                        &topo,
                        cut,
                        200.0 - 2.0 * u,
                        (&[PAST_LENS], &[LENS, BY_TOUCH]),
                        &pose,
                        &format!("cuts, {what}"),
                    );
                }
            }
        }
    }

    /// The GFA's own fuse of all three, with no layer above it to check its
    /// result: the exact solid or an error, never an invalid solid.
    #[test]
    fn the_gfa_fuse_of_all_three_is_exact_or_an_error() {
        let u = union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let what = format!("{seams}, {name}");
                let mut topo = Topology::new();
                let parts = parts(&mut topo, seam, &pose);
                if let Ok(fused) = brepkit_algo::gfa::fuse_n(&mut topo, &parts) {
                    assert_exact(
                        &topo,
                        fused,
                        200.0 + 2.0 * u,
                        (
                            &[
                                Point3::new(6.72, 6.0, 2.5),
                                BY_TOUCH,
                                Point3::new(9.0, 9.0, 1.0),
                            ],
                            &[Point3::new(6.74, 6.0, 2.5)],
                        ),
                        &pose,
                        &format!("fuse_n, {what}"),
                    );
                }
            }
        }
    }
}

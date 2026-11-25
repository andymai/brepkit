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
        parts_with(topo, 5.4_f64.sqrt(), seam, pose)
    }

    /// The plate, the rod and a prism of semi-axes 3 along x and `minor`
    /// along y about (4, 5), placed by `pose`.
    fn parts_with(topo: &mut Topology, minor: f64, seam: f64, pose: &Mat4) -> [SolidId; 3] {
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
            minor,
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

    /// That each edge of `solid` between a face on a cylinder and a face on
    /// a B-spline surface lies within the linear tolerance of both, at its
    /// two ends and its middle: off the cylinder by its distance from the
    /// axis less the radius, off the B-spline by the distance to a point of
    /// it that a projection finds, which no nearest point is farther than.
    fn assert_sections_on_both(topo: &Topology, solid: SolidId, what: &str) {
        let tol = 1e-7;
        let mut owners: std::collections::HashMap<_, Vec<FaceSurface>> =
            std::collections::HashMap::new();
        for f in solid_faces(topo, solid).unwrap() {
            let face = topo.face(f).unwrap();
            for w in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
                for oe in topo.wire(w).unwrap().edges() {
                    owners
                        .entry(oe.edge())
                        .or_default()
                        .push(face.surface().clone());
                }
            }
        }
        for (edge_id, surfaces) in owners {
            let cylinder = surfaces.iter().find_map(|s| match s {
                FaceSurface::Cylinder(c) => Some(c.clone()),
                _ => None,
            });
            let wall = surfaces.iter().find_map(|s| match s {
                FaceSurface::Nurbs(n) => Some(n.clone()),
                _ => None,
            });
            let (Some(cylinder), Some(wall)) = (cylinder, wall) else {
                continue;
            };
            let edge = topo.edge(edge_id).unwrap();
            let (start, end) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let axis = cylinder.axis().normalize().unwrap();
            for t in [t0, 0.5 * (t0 + t1), t1] {
                let p = edge.curve().evaluate_with_endpoints(t, start, end);
                let w = p - cylinder.origin();
                let off_cylinder = ((w - axis * w.dot(axis)).length() - cylinder.radius()).abs();
                let off_wall =
                    brepkit_math::nurbs::projection::project_point_to_surface(&wall, p, 1e-10)
                        .unwrap()
                        .distance;
                assert!(
                    off_cylinder <= tol && off_wall <= tol,
                    "{what}: section edge at {p:?} {off_cylinder:e} off the cylinder, \
                     {off_wall:e} off the wall"
                );
            }
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
        assert_sections_on_both(topo, solid, what);
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

    /// A rod of radius 2 about (5, 5) from z = -2 to 4, placed by `pose`
    /// with its seam turned by `seam`.
    fn tall_rod(topo: &mut Topology, seam: f64, pose: &Mat4) -> SolidId {
        let rod = make_cylinder(topo, 2.0, 6.0).unwrap();
        transform_solid(
            topo,
            rod,
            &(*pose * Mat4::translation(5.0, 5.0, -2.0) * Mat4::rotation_z(seam)),
        )
        .unwrap();
        rod
    }

    /// A prism of semi-axes 3 and `sqrt(3)` about (4, 5), from z = -1 to 3,
    /// crosses a rod from z = -2 to 4 at (4, 5 -/+ sqrt(3)) and touches it at
    /// (7, 5). With the seams on the touch, the prism's wall comes as quarter
    /// faces that end on the crossings, where rounding hides each quarter's
    /// side of the circle: the section there is the quarter's own boundary
    /// edge, and it must bound the window cut in the rod's wall.
    #[test]
    fn a_prism_whose_quarters_end_on_the_crossings_fuses_exactly() {
        let root3 = 3.0_f64.sqrt();
        // The left half of the ellipse, and the circle past the chord x = 4.
        let union = 1.5 * root3 * PI + 8.0 * PI / 3.0 + root3;
        // The rod, and the prism outside it over its height.
        let volume = 24.0 * PI + 4.0 * (union - 4.0 * PI);
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let [_, _, prism] = parts_with(&mut topo, root3, seam, &pose);
                let rod = tall_rod(&mut topo, seam, &pose);
                let what = format!("{seams}, {name}");
                let fused =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, rod, prism, exact()).unwrap();
                assert_exact(
                    &topo,
                    fused,
                    volume,
                    (
                        &[
                            Point3::new(6.9, 5.0, 1.0),
                            Point3::new(5.0, 6.9, 3.5),
                            Point3::new(1.5, 5.0, 2.0),
                            Point3::new(4.0, 5.0 + root3 - 0.05, 1.0),
                        ],
                        &[
                            Point3::new(7.1, 5.0, 1.0),
                            Point3::new(4.0, 5.0 + root3 + 0.05, 1.0),
                            Point3::new(4.0, 5.0 - root3 - 0.05, 1.0),
                            Point3::new(1.5, 5.0, 3.5),
                        ],
                    ),
                    &pose,
                    &what,
                );
                // The rod's wall holds an edge along each crossing, over the
                // prism's height.
                for y in [5.0 - root3, 5.0 + root3] {
                    let (low, high) = (
                        pose.mul_point(Point3::new(4.0, y, -1.0)),
                        pose.mul_point(Point3::new(4.0, y, 3.0)),
                    );
                    let bounded = solid_faces(&topo, fused).unwrap().into_iter().any(|f| {
                        let face = topo.face(f).unwrap();
                        matches!(face.surface(), FaceSurface::Cylinder(_))
                            && std::iter::once(face.outer_wire())
                                .chain(face.inner_wires().iter().copied())
                                .flat_map(|w| topo.wire(w).unwrap().edges().to_vec())
                                .any(|oe| {
                                    let edge = topo.edge(oe.edge()).unwrap();
                                    let ends = [
                                        topo.vertex(edge.start()).unwrap().point(),
                                        topo.vertex(edge.end()).unwrap().point(),
                                    ];
                                    ends.iter().any(|p| (*p - low).length() < 1e-9)
                                        && ends.iter().any(|p| (*p - high).length() < 1e-9)
                                })
                    });
                    assert!(bounded, "{what}: no wall edge at y = {y}");
                }
            }
        }
    }

    /// A rod of radius 2 about (6.5, 5) from z = 0 to 2, and the prism from
    /// z = -1 to 3: their walls cross at x = 10.25 - sqrt(19.9375), along
    /// rulings that run only over the rod's height, and the fuse is exact.
    #[test]
    fn a_shorter_rod_across_the_prism_fuses_exactly() {
        let b = 5.4_f64.sqrt();
        let crossing = 10.25 - 19.9375_f64.sqrt();
        // The area under half a circle of radius r from its centre out to u.
        let under = |r: f64, u: f64| 0.5 * (u * (r * r - u * u).sqrt() + r * r * (u / r).asin());
        // Inside both: the circle left of the crossing, the ellipse right of it.
        let circle_part = 2.0 * (under(2.0, crossing - 6.5) - under(2.0, -2.0));
        let ellipse_part = 2.0 * b / 3.0 * (under(3.0, 3.0) - under(3.0, crossing - 4.0));
        let both = circle_part + ellipse_part;
        let ellipse = PI * 3.0 * b;
        let volume = 4.0 * ellipse + 2.0 * (4.0 * PI - both);
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let [_, _, prism] = parts(&mut topo, seam, &pose);
                let rod = make_cylinder(&mut topo, 2.0, 2.0).unwrap();
                transform_solid(
                    &mut topo,
                    rod,
                    &(pose * Mat4::translation(6.5, 5.0, 0.0) * Mat4::rotation_z(seam)),
                )
                .unwrap();
                let what = format!("{seams}, {name}");
                let fused =
                    boolean_with_options(&mut topo, BooleanOp::Fuse, rod, prism, exact()).unwrap();
                assert_exact(
                    &topo,
                    fused,
                    volume,
                    (
                        &[Point3::new(8.0, 5.0, 1.0), Point3::new(1.5, 5.0, 2.5)],
                        &[Point3::new(8.0, 5.0, 2.5), Point3::new(8.0, 5.0, -0.5)],
                    ),
                    &pose,
                    &what,
                );
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

    /// The fuse of all three is exact upright and mirrored. Turned, it is
    /// exact or refused (#2878), and so are the plate less the rod and then
    /// the prism in every pose: never a wrong solid.
    #[test]
    fn the_three_way_fuse_and_the_two_cuts_are_exact_or_refused() {
        let u = union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let what = format!("{seams}, {name}");
                let mut topo = Topology::new();
                let [plate, rod, prism] = parts(&mut topo, seam, &pose);
                let compound = topo.add_compound(Compound::new(vec![plate, rod, prism]));
                let fused = fuse_all(&mut topo, compound);
                assert!(
                    fused.is_ok() || name == "turned",
                    "fuse, {what}: {:?}",
                    fused.as_ref().err()
                );
                if let Ok(fused) = fused {
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
    /// result: the exact solid upright and mirrored, and turned the exact
    /// solid or an error (#2878), never an invalid solid.
    #[test]
    fn the_gfa_fuse_of_all_three_is_exact_or_an_error() {
        let u = union_area();
        for (seams, seam) in SEAMS {
            for (name, pose) in poses() {
                let what = format!("{seams}, {name}");
                let mut topo = Topology::new();
                let parts = parts(&mut topo, seam, &pose);
                let fused = brepkit_algo::gfa::fuse_n(&mut topo, &parts);
                assert!(
                    fused.is_ok() || name == "turned",
                    "fuse_n, {what}: {:?}",
                    fused.as_ref().err()
                );
                if let Ok(fused) = fused {
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

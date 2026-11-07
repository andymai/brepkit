//! Corner options on a fillet: how far back along each edge its round stops, and
//! the patch closing the hole that leaves.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::OperationsError;
    use brepkit_operations::blend_ops::{CornerKind, FilletCorner, fillet_v2_with_corners};
    use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_planar_face;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{solid_edges, solid_faces, solid_vertices};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::VertexId;

    const RADIUS: f64 = 2.0;

    fn poses() -> [(&'static str, Mat4); 4] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "far away",
                Mat4::translation(370.0, 4830.0, -91.0)
                    * Mat4::rotation_y(1.1)
                    * Mat4::rotation_x(0.3),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    fn cube(topo: &mut Topology, pose: Mat4) -> SolidId {
        let body = make_box(topo, 20.0, 20.0, 20.0).unwrap();
        transform_solid(topo, body, &pose).unwrap();
        body
    }

    fn vertex_at(topo: &Topology, solid: SolidId, at: Point3) -> VertexId {
        solid_vertices(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&v| (topo.vertex(v).unwrap().point() - at).length() < 1e-6)
            .unwrap()
    }

    fn edges_at(topo: &Topology, solid: SolidId, vertex: VertexId) -> Vec<EdgeId> {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                edge.start() == vertex || edge.end() == vertex
            })
            .collect()
    }

    fn set_back(edges: &[EdgeId], distances: [f64; 3]) -> Vec<(EdgeId, f64)> {
        edges.iter().copied().zip(distances).collect()
    }

    fn origin_corner(topo: &mut Topology, pose: Mat4) -> (SolidId, VertexId, Vec<EdgeId>) {
        let body = cube(topo, pose);
        let corner = vertex_at(topo, body, pose.mul_point(Point3::new(0.0, 0.0, 0.0)));
        let edges = edges_at(topo, body, corner);
        (body, corner, edges)
    }

    /// Valid and watertight, its faces' integral agreeing with its fine mesh.
    fn assert_sound(topo: &Topology, solid: SolidId, case: &str) -> f64 {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{case}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, solid, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{case}: the mesh leaks");
        solid_volume(topo, solid, 0.01).unwrap()
    }

    /// A corner patch's parameters along the side its `k`th wire edge runs.
    fn along_side(k: usize, t: f64) -> (f64, f64) {
        match k {
            0 => (t, 0.0),
            1 => (1.0, t),
            2 => (1.0 - t, 1.0),
            _ => (0.0, 1.0 - t),
        }
    }

    /// The largest angle between a corner patch and the face beside it, read at
    /// points along each of its sides.
    fn largest_crease(topo: &Topology, solid: SolidId) -> f64 {
        let faces = solid_faces(topo, solid).unwrap();
        let wire_of = |face| {
            topo.wire(topo.face(face).unwrap().outer_wire())
                .unwrap()
                .edges()
                .to_vec()
        };
        let mut worst = 0.0_f64;
        for &face in &faces {
            let FaceSurface::Nurbs(patch) = topo.face(face).unwrap().surface() else {
                continue;
            };
            for (k, oriented) in wire_of(face).iter().enumerate() {
                let other = faces
                    .iter()
                    .copied()
                    .find(|&f| f != face && wire_of(f).iter().any(|o| o.edge() == oriented.edge()))
                    .unwrap();
                let other_data = topo.face(other).unwrap();
                for step in 1..16 {
                    let t = f64::from(step) / 16.0;
                    let (u, v) = along_side(k, t);
                    let at = patch.evaluate(u, v);
                    let here = patch.normal(u, v).unwrap();
                    let there = match other_data.surface() {
                        FaceSurface::Plane { normal, .. } => *normal,
                        FaceSurface::Nurbs(neighbour) => {
                            let j = wire_of(other)
                                .iter()
                                .position(|o| o.edge() == oriented.edge())
                                .unwrap();
                            let (a, b) = (along_side(j, t), along_side(j, 1.0 - t));
                            let (p, q) =
                                (neighbour.evaluate(a.0, a.1), neighbour.evaluate(b.0, b.1));
                            let (x, y) = if (p - at).length() <= (q - at).length() {
                                a
                            } else {
                                b
                            };
                            neighbour.normal(x, y).unwrap()
                        }
                        surface => {
                            let (x, y) = surface.project_point(at).unwrap();
                            surface.normal(x, y)
                        }
                    };
                    let there = if other_data.is_reversed() {
                        -there
                    } else {
                        there
                    };
                    assert!(here.dot(there) > 0.0, "the faces at an edge face apart");
                    worst = worst.max(here.cross(there).length().asin());
                }
            }
        }
        worst
    }

    #[test]
    fn a_set_back_corner_is_sound_and_smooth_in_every_pose() {
        for (name, pose) in poses() {
            for distances in [[3.0, 3.0, 3.0], [2.5, 4.0, 6.0]] {
                let case = format!("{name} {distances:?}");
                let mut topo = Topology::new();
                let (body, corner, edges) = origin_corner(&mut topo, pose);
                let rolled = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &[]).unwrap();
                let rolled_volume = assert_sound(&topo, rolled, &case);

                let corners = [FilletCorner {
                    vertex: corner,
                    setbacks: set_back(&edges, distances),
                    kind: CornerKind::Rolling,
                }];
                let result =
                    fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
                let volume = assert_sound(&topo, result, &case);
                assert_eq!(solid_faces(&topo, result).unwrap().len(), 11, "{case}");
                assert!(
                    volume < rolled_volume && volume > rolled_volume - 30.0,
                    "{case}: holds {volume}, the rolled corner {rolled_volume}"
                );
                let crease = largest_crease(&topo, result);
                assert!(crease < 1e-5, "{case}: a crease of {crease}");
            }
        }
    }

    #[test]
    fn a_longer_set_back_takes_more_off() {
        let mut volumes = Vec::new();
        for past in [0.5, 1.0, 2.0, 4.0] {
            let mut topo = Topology::new();
            let (body, corner, edges) = origin_corner(&mut topo, Mat4::identity());
            let d = RADIUS + past;
            let corners = [FilletCorner {
                vertex: corner,
                setbacks: set_back(&edges, [d, d, d]),
                kind: CornerKind::G2,
            }];
            let result = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
            volumes.push(assert_sound(&topo, result, "longer"));
        }
        assert!(volumes.windows(2).all(|w| w[1] < w[0]), "{volumes:?}");
    }

    #[test]
    fn a_g2_corner_with_no_set_back_stops_one_radius_past_the_ball() {
        let mut topo = Topology::new();
        let (body, corner, edges) = origin_corner(&mut topo, Mat4::identity());
        let g2 = [FilletCorner {
            vertex: corner,
            setbacks: Vec::new(),
            kind: CornerKind::G2,
        }];
        let by_default = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &g2).unwrap();
        let explicit = [FilletCorner {
            vertex: corner,
            setbacks: set_back(&edges, [4.0, 4.0, 4.0]),
            kind: CornerKind::G2,
        }];
        let by_hand = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &explicit).unwrap();
        let (a, b) = (
            assert_sound(&topo, by_default, "default"),
            assert_sound(&topo, by_hand, "by hand"),
        );
        assert!((a - b).abs() < 1e-9, "{a} against {b}");
    }

    #[test]
    fn a_rolling_corner_with_no_set_back_stays_a_ball() {
        let mut topo = Topology::new();
        let (body, corner, edges) = origin_corner(&mut topo, Mat4::identity());
        let corners = [FilletCorner {
            vertex: corner,
            setbacks: Vec::new(),
            kind: CornerKind::Rolling,
        }];
        let result = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
        let balls = solid_faces(&topo, result)
            .unwrap()
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
            .count();
        assert_eq!(balls, 1);
        assert_sound(&topo, result, "ball");
    }

    #[test]
    fn every_corner_of_a_box_sets_back() {
        for (name, pose) in poses().into_iter().filter(|(name, _)| *name != "far away") {
            let mut topo = Topology::new();
            let body = cube(&mut topo, pose);
            let edges = solid_edges(&topo, body).unwrap();
            let corners: Vec<FilletCorner> = solid_vertices(&topo, body)
                .unwrap()
                .into_iter()
                .map(|vertex| FilletCorner {
                    vertex,
                    setbacks: set_back(&edges_at(&topo, body, vertex), [3.0, 4.0, 5.0]),
                    kind: CornerKind::G2,
                })
                .collect();
            let result = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
            assert_sound(&topo, result, name);
            assert_eq!(
                solid_faces(&topo, result).unwrap().len(),
                6 + 12 + 16,
                "{name}"
            );
            let crease = largest_crease(&topo, result);
            assert!(crease < 1e-5, "{name}: a crease of {crease}");
        }
    }

    /// A corner of a slanted prism, whose flat faces do not meet square.
    #[test]
    fn a_corner_off_square_sets_back() {
        for (name, pose) in poses().into_iter().filter(|(name, _)| *name != "far away") {
            let mut topo = Topology::new();
            let outline = [
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(30.0, 0.0, 0.0),
                Point3::new(35.0, 25.0, 0.0),
                Point3::new(8.0, 22.0, 0.0),
            ];
            let bottom = make_planar_face(&mut topo, &outline, 1e-7).unwrap();
            let prism = extrude(&mut topo, bottom, Vec3::new(0.0, 0.0, 1.0), 20.0).unwrap();
            transform_solid(&mut topo, prism, &pose).unwrap();
            let corner = vertex_at(&topo, prism, pose.mul_point(Point3::new(0.0, 0.0, 0.0)));
            let edges = edges_at(&topo, prism, corner);
            let rolled = fillet_v2_with_corners(&mut topo, prism, &edges, RADIUS, &[]).unwrap();
            let rolled_volume = assert_sound(&topo, rolled, name);

            let corners = [FilletCorner {
                vertex: corner,
                setbacks: set_back(&edges, [5.0, 6.0, 7.0]),
                kind: CornerKind::G2,
            }];
            let result =
                fillet_v2_with_corners(&mut topo, prism, &edges, RADIUS, &corners).unwrap();
            let volume = assert_sound(&topo, result, name);
            assert!(
                volume < rolled_volume,
                "{name}: {volume} from {rolled_volume}"
            );
            let crease = largest_crease(&topo, result);
            assert!(crease < 1e-5, "{name}: a crease of {crease}");
        }
    }

    /// A pocket's inner corner, whose rounds add material and whose ball sits in
    /// the air.
    #[test]
    fn a_pockets_inner_corner_sets_back() {
        for (name, pose) in poses().into_iter().filter(|(name, _)| *name != "far away") {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 40.0, 40.0, 40.0).unwrap();
            let pocket = make_box(&mut topo, 20.0, 20.0, 40.0).unwrap();
            transform_solid(&mut topo, pocket, &Mat4::translation(10.0, 10.0, 10.0)).unwrap();
            let cut = boolean(&mut topo, BooleanOp::Cut, block, pocket).unwrap();
            transform_solid(&mut topo, cut, &pose).unwrap();
            let corner = vertex_at(&topo, cut, pose.mul_point(Point3::new(10.0, 10.0, 10.0)));
            let meeting = edges_at(&topo, cut, corner);
            assert_eq!(meeting.len(), 3, "{name}");
            let rolled = fillet_v2_with_corners(&mut topo, cut, &meeting, RADIUS, &[]).unwrap();
            let rolled_volume = assert_sound(&topo, rolled, name);

            let corners = [FilletCorner {
                vertex: corner,
                setbacks: set_back(&meeting, [4.0, 5.0, 6.0]),
                kind: CornerKind::G2,
            }];
            let result =
                fillet_v2_with_corners(&mut topo, cut, &meeting, RADIUS, &corners).unwrap();
            let volume = assert_sound(&topo, result, name);
            assert!(
                volume > rolled_volume && volume < rolled_volume + 30.0,
                "{name}: {volume} from {rolled_volume}"
            );
            let crease = largest_crease(&topo, result);
            assert!(crease < 1e-5, "{name}: a crease of {crease}");
        }
    }

    /// A corner mixing convex and concave edges has no ball, so asking for one is
    /// refused rather than answered with whatever closes it.
    #[test]
    fn a_rolling_corner_with_no_ball_is_refused() {
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 40.0, 40.0, 40.0).unwrap();
        let step = make_box(&mut topo, 20.0, 40.0, 20.0).unwrap();
        transform_solid(&mut topo, step, &Mat4::translation(20.0, 0.0, 20.0)).unwrap();
        let stepped = boolean(&mut topo, BooleanOp::Cut, block, step).unwrap();
        let at = Point3::new(20.0, 0.0, 20.0);
        let corner = vertex_at(&topo, stepped, at);
        let edges = edges_at(&topo, stepped, corner);
        assert_eq!(edges.len(), 3);
        let plain = fillet_v2_with_corners(&mut topo, stepped, &edges, RADIUS, &[]);
        assert!(plain.is_ok(), "the plain fillet closes it: {plain:?}");
        let corners = [FilletCorner {
            vertex: corner,
            setbacks: Vec::new(),
            kind: CornerKind::Rolling,
        }];
        let refused = fillet_v2_with_corners(&mut topo, stepped, &edges, RADIUS, &corners);
        assert!(
            refused.is_err_and(|e| e.to_string().contains("ball corner")),
            "a mixed corner has no ball"
        );
    }

    /// A cavity the rolled corner clears but the set-back's patches could reach
    /// is refused, and one far off is kept.
    #[test]
    fn a_cavity_the_patches_could_reach_is_refused() {
        let hollow = |topo: &mut Topology, at: f64| {
            let block = make_box(topo, 40.0, 40.0, 40.0).unwrap();
            let void = make_box(topo, 1.0, 1.0, 1.0).unwrap();
            transform_solid(topo, void, &Mat4::translation(at, at, at)).unwrap();
            boolean(topo, BooleanOp::Cut, block, void).unwrap()
        };
        let corner_of = |topo: &Topology, solid: SolidId| {
            let corner = vertex_at(topo, solid, Point3::new(0.0, 0.0, 0.0));
            let edges = edges_at(topo, solid, corner);
            (corner, edges)
        };
        let mut topo = Topology::new();
        let near = hollow(&mut topo, 9.0);
        let (corner, edges) = corner_of(&topo, near);
        let rolled = fillet_v2_with_corners(&mut topo, near, &edges, RADIUS, &[]);
        assert!(
            rolled.is_ok(),
            "the rolled corner clears the cavity: {rolled:?}"
        );
        let corners = [FilletCorner {
            vertex: corner,
            setbacks: set_back(&edges, [14.0, 14.0, 14.0]),
            kind: CornerKind::G2,
        }];
        let refused = fillet_v2_with_corners(&mut topo, near, &edges, RADIUS, &corners);
        assert!(refused.is_err(), "the patches could reach the cavity");

        let far = hollow(&mut topo, 20.0);
        let (corner, edges) = corner_of(&topo, far);
        let corners = [FilletCorner {
            vertex: corner,
            setbacks: set_back(&edges, [8.0, 8.0, 8.0]),
            kind: CornerKind::G2,
        }];
        let kept = fillet_v2_with_corners(&mut topo, far, &edges, RADIUS, &corners).unwrap();
        assert_eq!(topo.solid(kept).unwrap().inner_shells().len(), 1);
        assert_sound(&topo, kept, "a cavity far off");
    }

    /// The patches do not depend on which way a wire happens to run, so a mirror
    /// image holds what the part did, and so does one turned and moved far off.
    #[test]
    fn a_corners_volume_does_not_depend_on_its_pose() {
        let mut volumes = Vec::new();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (body, corner, edges) = origin_corner(&mut topo, pose);
            let corners = [FilletCorner {
                vertex: corner,
                setbacks: set_back(&edges, [3.0, 4.0, 5.0]),
                kind: CornerKind::G2,
            }];
            let result = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
            volumes.push((name, solid_volume(&topo, result, 0.01).unwrap()));
        }
        for &(name, volume) in &volumes {
            assert!((volume - volumes[0].1).abs() < 1e-6, "{volumes:?}: {name}");
        }
    }

    #[test]
    fn a_set_back_corner_goes_on_to_a_later_boolean() {
        let mut topo = Topology::new();
        let (body, corner, edges) = origin_corner(&mut topo, Mat4::identity());
        let corners = [FilletCorner {
            vertex: corner,
            setbacks: set_back(&edges, [3.0, 4.0, 5.0]),
            kind: CornerKind::G2,
        }];
        let result = fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &corners).unwrap();
        let held = assert_sound(&topo, result, "before");
        let tool = make_box(&mut topo, 5.0, 5.0, 40.0).unwrap();
        transform_solid(&mut topo, tool, &Mat4::translation(12.0, 12.0, -10.0)).unwrap();
        let fallbacks = mesh_fallback_count();
        let cut = boolean(&mut topo, BooleanOp::Cut, result, tool).unwrap();
        assert_eq!(
            mesh_fallback_count(),
            fallbacks,
            "the cut fell back to a mesh"
        );
        let left = assert_sound(&topo, cut, "after");
        assert!(
            (held - 500.0 - left).abs() < 1e-3,
            "{held} less 500 is not {left}"
        );
    }

    #[test]
    fn bad_corners_are_refused() {
        let refused = |result: Result<SolidId, OperationsError>, what: &str| {
            assert!(result.is_err(), "{what}");
        };
        let mut topo = Topology::new();
        let (body, corner, edges) = origin_corner(&mut topo, Mat4::identity());
        let with = |setbacks, kind| {
            [FilletCorner {
                vertex: corner,
                setbacks,
                kind,
            }]
        };

        // At or before where the ball stops the round.
        let short = with(set_back(&edges, [2.0, 3.0, 3.0]), CornerKind::G2);
        refused(
            fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &short),
            "at the ball",
        );

        // Past the round's far end: the far corner is rolled too.
        let all = solid_edges(&topo, body).unwrap();
        let long = with(set_back(&edges, [3.0, 3.0, 19.0]), CornerKind::G2);
        refused(
            fillet_v2_with_corners(&mut topo, body, &all, RADIUS, &long),
            "past the end",
        );

        // Two given of three, and only two rounds meeting.
        let two = with(set_back(&edges[..2], [3.0, 3.0, 3.0]), CornerKind::G2);
        refused(
            fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &two),
            "two given",
        );
        let g2 = with(Vec::new(), CornerKind::G2);
        refused(
            fillet_v2_with_corners(&mut topo, body, &edges[..2], RADIUS, &g2),
            "two rounds",
        );

        // An edge from somewhere else.
        let far = all.iter().copied().find(|e| !edges.contains(e)).unwrap();
        let elsewhere = with(
            vec![(edges[0], 3.0), (edges[1], 3.0), (far, 3.0)],
            CornerKind::G2,
        );
        refused(
            fillet_v2_with_corners(&mut topo, body, &edges, RADIUS, &elsewhere),
            "elsewhere",
        );

        // The body comes through all of it unharmed.
        assert_sound(&topo, body, "body");
    }
}

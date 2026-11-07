//! Marking a solid with a face or an edge, and cutting it in pieces with faces.
//!
//! The tool is the face itself, not its plane: one that dips into a body marks
//! what it touches and leaves the rest, and a body is only separated where a
//! tool closes a region off. Every case runs upright, turned off the axes and
//! mirrored, and each result is held to validity, watertightness, volume and
//! face count.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::nurbs::surface::NurbsSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::imprint::{ImprintTool, imprint_faces, split_by_faces};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::{
        make_box, make_cone, make_cylinder, make_sphere, make_torus,
    };
    use brepkit_operations::sheet::extrude_wire;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::{
        make_circle_edge, make_line_edge, make_nurbs_face, make_planar_face,
    };
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const PLATE: f64 = 4000.0;
    const DIP: [[f64; 3]; 4] = [
        [-15.0, 0.0, 9.0],
        [15.0, 0.0, 9.0],
        [15.0, 0.0, 11.0],
        [-15.0, 0.0, 11.0],
    ];
    const LYING: [[f64; 3]; 4] = [
        [-15.0, 0.0, 10.0],
        [15.0, 0.0, 10.0],
        [15.0, 15.0, 10.0],
        [-15.0, 15.0, 10.0],
    ];
    const ACROSS: [[f64; 3]; 4] = [
        [-15.0, 0.0, -5.0],
        [15.0, 0.0, -5.0],
        [15.0, 0.0, 15.0],
        [-15.0, 0.0, 15.0],
    ];
    const ACROSS_Y: [[f64; 3]; 4] = [
        [0.0, -15.0, -5.0],
        [0.0, 15.0, -5.0],
        [0.0, 15.0, 15.0],
        [0.0, -15.0, 15.0],
    ];

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

    fn at(pose: Mat4, p: [f64; 3]) -> Point3 {
        pose.mul_point(Point3::new(p[0], p[1], p[2]))
    }

    /// A box with x and y from -10 to 10, z from 0 to 10.
    fn plate(topo: &mut Topology, pose: Mat4) -> SolidId {
        let plate = make_box(topo, 20.0, 20.0, 10.0).unwrap();
        transform_solid(topo, plate, &(pose * Mat4::translation(-10.0, -10.0, 0.0))).unwrap();
        plate
    }

    /// The plate less a bore of radius 3 down the z axis.
    fn holed_plate(topo: &mut Topology, pose: Mat4) -> SolidId {
        let plate = make_box(topo, 20.0, 20.0, 10.0).unwrap();
        transform_solid(topo, plate, &Mat4::translation(-10.0, -10.0, 0.0)).unwrap();
        let bore = make_cylinder(topo, 3.0, 20.0).unwrap();
        transform_solid(topo, bore, &Mat4::translation(0.0, 0.0, -5.0)).unwrap();
        let holed = boolean(topo, BooleanOp::Cut, plate, bore).unwrap();
        transform_solid(topo, holed, &pose).unwrap();
        holed
    }

    fn cylinder(topo: &mut Topology, pose: Mat4) -> SolidId {
        let body = make_cylinder(topo, 10.0, 10.0).unwrap();
        transform_solid(topo, body, &pose).unwrap();
        body
    }

    fn rectangle(topo: &mut Topology, pose: Mat4, corners: [[f64; 3]; 4]) -> FaceId {
        let corners: Vec<Point3> = corners.iter().map(|&c| at(pose, c)).collect();
        make_planar_face(topo, &corners, 1e-7).unwrap()
    }

    fn segment(topo: &mut Topology, pose: Mat4, a: [f64; 3], b: [f64; 3]) -> ImprintTool {
        ImprintTool::Edge(make_line_edge(topo, at(pose, a), at(pose, b), 1e-7).unwrap())
    }

    fn circle(topo: &mut Topology, pose: Mat4, z: f64, radius: f64) -> EdgeId {
        let center = at(pose, [0.0, 0.0, z]);
        let axis = (at(pose, [0.0, 0.0, z + 1.0]) - center)
            .normalize()
            .unwrap();
        make_circle_edge(topo, center, axis, radius, 1e-7).unwrap()
    }

    /// A flat ring around the z axis at height `z`.
    fn ring(topo: &mut Topology, pose: Mat4, z: f64, inner: f64, outer: f64) -> FaceId {
        let center = at(pose, [0.0, 0.0, z]);
        let normal = (at(pose, [0.0, 0.0, z + 1.0]) - center)
            .normalize()
            .unwrap();
        let rim = circle(topo, pose, z, outer);
        let hole = circle(topo, pose, z, inner);
        let rim = Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap();
        let hole = Wire::new(vec![OrientedEdge::new(hole, false)], true).unwrap();
        let (rim, hole) = (topo.add_wire(rim), topo.add_wire(hole));
        let d = normal.dot(Vec3::new(center.x(), center.y(), center.z()));
        topo.add_face(Face::new(rim, vec![hole], FaceSurface::Plane { normal, d }))
    }

    /// Valid, watertight, with no face running an edge out and back, holding
    /// `volume` to `margin` in `faces` faces.
    fn check(topo: &Topology, piece: SolidId, volume: f64, margin: f64, faces: usize, case: &str) {
        let report = validate_solid(topo, piece).unwrap();
        assert!(report.is_valid(), "{case}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, piece, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{case}: the mesh leaks");
        let measured = solid_volume(topo, piece, 0.005).unwrap();
        assert!(
            (measured - volume).abs() <= margin,
            "{case}: holds {measured}, should hold {volume}"
        );
        let found = solid_faces(topo, piece).unwrap();
        assert_eq!(found.len(), faces, "{case}: face count");
        for face in found {
            let data = topo.face(face).unwrap();
            if !matches!(data.surface(), FaceSurface::Plane { .. }) {
                continue;
            }
            let wire = topo.wire(data.outer_wire()).unwrap();
            for oe in wire.edges() {
                let uses = wire
                    .edges()
                    .iter()
                    .filter(|o| o.edge() == oe.edge())
                    .count();
                assert_eq!(uses, 1, "{case}: a plane face runs an edge twice");
            }
        }
    }

    /// The areas of the faces in the plane through `p` square to `normal`,
    /// smallest first.
    fn areas_in_plane(
        topo: &Topology,
        piece: SolidId,
        pose: Mat4,
        p: [f64; 3],
        normal: [f64; 3],
    ) -> Vec<f64> {
        let origin = at(pose, p);
        let tip = at(pose, [p[0] + normal[0], p[1] + normal[1], p[2] + normal[2]]);
        let normal = (tip - origin).normalize().unwrap();
        let on_plane = normal.dot(Vec3::new(origin.x(), origin.y(), origin.z()));
        let mut areas: Vec<f64> = solid_faces(topo, piece)
            .unwrap()
            .into_iter()
            .filter(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal: n, d }
                    if n.dot(normal).abs() > 1.0 - 1e-9
                        && (d - on_plane * n.dot(normal).signum()).abs() < 1e-7)
            })
            .map(|f| face_area(topo, f, 0.005).unwrap())
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    fn curved_faces_of(topo: &Topology, solid: SolidId) -> Vec<FaceId> {
        solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&f| !topo.face(f).unwrap().surface().is_planar())
            .collect()
    }

    fn by_volume(topo: &Topology, pieces: &mut [SolidId]) {
        pieces.sort_by(|&a, &b| {
            let (a, b) = (
                solid_volume(topo, a, 0.01).unwrap(),
                solid_volume(topo, b, 0.01).unwrap(),
            );
            a.total_cmp(&b)
        });
    }

    fn assert_halved_top(topo: &Topology, piece: SolidId, pose: Mat4, case: &str) {
        let areas = areas_in_plane(topo, piece, pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
        assert_eq!(areas.len(), 2, "{case}: the top's pieces");
        for area in areas {
            assert!((area - 200.0).abs() < 1e-6, "{case}: a top piece of {area}");
        }
    }

    #[test]
    fn a_face_dipping_into_the_top_marks_only_the_top() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, DIP);

            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, marked, pose, name);
            // Its short verticals on the side walls stop inside them.
            for wall in [[10.0, 0.0, 5.0], [-10.0, 0.0, 5.0]] {
                let areas = areas_in_plane(&topo, marked, pose, wall, [1.0, 0.0, 0.0]);
                assert_eq!(areas.len(), 1, "{name}: a side wall came apart");
                assert!((areas[0] - 200.0).abs() < 1e-6, "{name}: wall {}", areas[0]);
            }

            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: a dip separates nothing");
            check(&topo, pieces[0], PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, pieces[0], pose, name);
        }
    }

    #[test]
    fn a_face_lying_on_the_top_marks_it_at_its_edge() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, LYING);

            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, marked, pose, name);

            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: lying on a face separates nothing");
            check(&topo, pieces[0], PLATE, 1e-6, 7, name);
        }
    }

    #[test]
    fn an_edge_on_the_top_marks_it() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);

            let rim_to_rim = segment(&mut topo, pose, [-10.0, 0.0, 10.0], [10.0, 0.0, 10.0]);
            let marked = imprint_faces(&mut topo, body, &[rim_to_rim]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, marked, pose, name);

            let past_the_rims = segment(&mut topo, pose, [-15.0, 0.0, 10.0], [15.0, 0.0, 10.0]);
            let marked = imprint_faces(&mut topo, body, &[past_the_rims]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, marked, pose, name);

            let stops_inside = segment(&mut topo, pose, [-10.0, 0.0, 10.0], [4.0, 0.0, 10.0]);
            let marked = imprint_faces(&mut topo, body, &[stops_inside]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);

            let floating = segment(&mut topo, pose, [-5.0, 0.0, 10.0], [5.0, 0.0, 10.0]);
            let marked = imprint_faces(&mut topo, body, &[floating]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);

            let through_the_air = segment(&mut topo, pose, [-5.0, 0.0, 12.0], [5.0, 0.0, 12.0]);
            let marked = imprint_faces(&mut topo, body, &[through_the_air]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);
        }
    }

    #[test]
    fn edges_closing_a_loop_mark_what_none_marks_alone() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let corners = [[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]];
            let sides: Vec<ImprintTool> = (0..4)
                .map(|k| {
                    let (a, b) = (corners[k], corners[(k + 1) % 4]);
                    segment(&mut topo, pose, [a[0], a[1], 10.0], [b[0], b[1], 10.0])
                })
                .collect();

            let open = imprint_faces(&mut topo, body, &sides[..3]).unwrap();
            check(&topo, open, PLATE, 1e-6, 6, name);

            let closed = imprint_faces(&mut topo, body, &sides).unwrap();
            check(&topo, closed, PLATE, 1e-6, 7, name);
            let areas = areas_in_plane(&topo, closed, pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
            assert_eq!(areas.len(), 2, "{name}: island and frame");
            assert!(
                (areas[0] - 100.0).abs() < 1e-6,
                "{name}: island {}",
                areas[0]
            );
            assert!(
                (areas[1] - 300.0).abs() < 1e-6,
                "{name}: frame {}",
                areas[1]
            );
        }
    }

    #[test]
    fn a_circle_edge_marks_a_top_and_a_wall() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let on_top = ImprintTool::Edge(circle(&mut topo, pose, 10.0, 4.0));
            let marked = imprint_faces(&mut topo, body, &[on_top]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
            let areas = areas_in_plane(&topo, marked, pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
            assert_eq!(areas.len(), 2, "{name}: disc and frame");
            assert!(
                (areas[0] - 16.0 * PI).abs() < 1e-3,
                "{name}: disc {}",
                areas[0]
            );

            let body = cylinder(&mut topo, pose);
            let round_the_wall = ImprintTool::Edge(circle(&mut topo, pose, 5.0, 10.0));
            let marked = imprint_faces(&mut topo, body, &[round_the_wall]).unwrap();
            check(&topo, marked, 1000.0 * PI, 0.5, 4, name);
        }
    }

    #[test]
    fn a_ruling_edge_marks_a_cylinder_wall() {
        // A ruling and the wall's seam part the wall in two, wherever the
        // ruling stands against the seam and against the rims' own origin;
        // along the seam, there is nothing to part.
        for (name, pose) in poses() {
            for (angle, faces) in [
                (FRAC_PI_2, 4),
                (1.0, 4),
                (2.5, 4),
                (PI, 4),
                (4.0, 4),
                (0.0, 3),
            ] {
                let mut topo = Topology::new();
                let body = cylinder(&mut topo, pose);
                let (x, y) = (10.0 * f64::cos(angle), 10.0 * f64::sin(angle));
                let rule = segment(&mut topo, pose, [x, y, 0.0], [x, y, 10.0]);
                let marked = imprint_faces(&mut topo, body, &[rule]).unwrap();
                let case = format!("{name} at {angle}");
                check(&topo, marked, 1000.0 * PI, 0.5, faces, &case);
            }
        }
    }

    #[test]
    fn a_face_across_the_whole_body_separates_it() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, ACROSS);

            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: two sides");
            for &piece in &pieces {
                check(&topo, piece, 2000.0, 1e-6, 6, name);
            }
            let holds = |topo: &Topology, piece: SolidId, y: f64| {
                let p = at(pose, [1.0, y, 5.0]);
                classify_point(topo, piece, p, 0.0, 1e-7).unwrap() == PointClassification::Inside
            };
            let (a, b) = (pieces[0], pieces[1]);
            assert!(
                (holds(&topo, a, 5.0) && holds(&topo, b, -5.0))
                    || (holds(&topo, a, -5.0) && holds(&topo, b, 5.0)),
                "{name}: one piece on each side"
            );

            // Only marked, its section runs all the way round.
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 10, name);
        }
    }

    #[test]
    fn a_tool_missing_the_body_leaves_it_unchanged() {
        let beside = [
            [-15.0, 0.0, 50.0],
            [15.0, 0.0, 50.0],
            [15.0, 0.0, 60.0],
            [-15.0, 0.0, 60.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, beside);

            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            assert_ne!(marked, body, "{name}: a copy, not the input");
            check(&topo, marked, PLATE, 1e-6, 6, name);
            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}");
            check(&topo, pieces[0], PLATE, 1e-6, 6, name);

            let nothing = imprint_faces(&mut topo, body, &[]).unwrap();
            check(&topo, nothing, PLATE, 1e-6, 6, name);
            let nothing = split_by_faces(&mut topo, body, &[]).unwrap();
            assert_eq!(nothing.len(), 1, "{name}");
            check(&topo, nothing[0], PLATE, 1e-6, 6, name);
        }
    }

    #[test]
    fn a_face_marks_a_cylinders_cap_and_wall() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = cylinder(&mut topo, pose);
            let volume = 1000.0 * PI;

            let chord = rectangle(&mut topo, pose, DIP);
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(chord)]).unwrap();
            check(&topo, marked, volume, 0.5, 4, name);
            let cap = areas_in_plane(&topo, marked, pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
            assert_eq!(cap.len(), 2, "{name}: the cap's halves");
            for area in cap {
                assert!((area - 50.0 * PI).abs() < 1e-2, "{name}: half a cap {area}");
            }

            let band = ring(&mut topo, pose, 4.0, 8.0, 12.0);
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(band)]).unwrap();
            check(&topo, marked, volume, 0.5, 4, name);

            // The ring stops short of the axis, so it marks and does not cut.
            let pieces = split_by_faces(&mut topo, body, &[band]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: the ring has a hole");
            check(&topo, pieces[0], volume, 0.5, 4, name);

            // A whole disc past the wall does cut.
            let rim = circle(&mut topo, pose, 4.0, 12.0);
            let rim = Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap();
            let rim = topo.add_wire(rim);
            let (center, tip) = (at(pose, [0.0, 0.0, 4.0]), at(pose, [0.0, 0.0, 5.0]));
            let normal = (tip - center).normalize().unwrap();
            let d = normal.dot(Vec3::new(center.x(), center.y(), center.z()));
            let disc = topo.add_face(Face::new(rim, vec![], FaceSurface::Plane { normal, d }));
            let mut pieces = split_by_faces(&mut topo, body, &[disc]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: a whole disc");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 400.0 * PI, 0.5, 3, name);
            check(&topo, pieces[1], 600.0 * PI, 0.5, 3, name);
        }
    }

    #[test]
    fn a_plane_separates_a_cylinder_off_its_axis_and_on_a_tilt() {
        let off_axis = [
            [-15.0, 4.0, -5.0],
            [15.0, 4.0, -5.0],
            [15.0, 4.0, 15.0],
            [-15.0, 4.0, 15.0],
        ];
        let tilted = [
            [-15.0, -15.0, 2.0],
            [15.0, -15.0, 2.0],
            [15.0, 15.0, 8.0],
            [-15.0, 15.0, 8.0],
        ];
        // The segment of the disc past the chord at y = 4.
        let segment_area = 100.0 * (0.4_f64).acos() - 4.0 * (100.0_f64 - 16.0).sqrt();
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = cylinder(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, off_axis);
            let mut pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 10.0 * segment_area, 0.5, 4, name);
            let rest = 1000.0 * PI - 10.0 * segment_area;
            check(&topo, pieces[1], rest, 0.5, 5, name);

            let tool = rectangle(&mut topo, pose, tilted);
            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: tilted");
            for piece in pieces {
                check(&topo, piece, 500.0 * PI, 0.5, 3, name);
            }
        }
    }

    #[test]
    fn a_tool_across_a_hole() {
        let volume = PLATE - PI * 90.0;
        let one_web = [
            [-15.0, 0.0, -5.0],
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 15.0],
            [-15.0, 0.0, 15.0],
        ];
        let other_web = [
            [0.0, 0.0, -5.0],
            [15.0, 0.0, -5.0],
            [15.0, 0.0, 15.0],
            [0.0, 0.0, 15.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = holed_plate(&mut topo, pose);

            let dip = rectangle(&mut topo, pose, DIP);
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(dip)]).unwrap();
            check(&topo, marked, volume, 0.5, 8, name);
            let top = areas_in_plane(&topo, marked, pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
            assert_eq!(top.len(), 2, "{name}: the holed top's halves");

            let across = rectangle(&mut topo, pose, ACROSS);
            let pieces = split_by_faces(&mut topo, body, &[across]).unwrap();
            assert_eq!(
                pieces.len(),
                2,
                "{name}: the hole does not hold them together"
            );
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 0.5, 8, name);
            }

            // One web alone leaves the plate joined round the hole's far side:
            // a section from the rim to the hole leaves the top in one piece,
            // while the wall and the bore each come apart at theirs.
            let web = rectangle(&mut topo, pose, one_web);
            let pieces = split_by_faces(&mut topo, body, &[web]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: one web");
            check(&topo, pieces[0], volume, 0.5, 9, name);
            let top = areas_in_plane(&topo, pieces[0], pose, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0]);
            assert_eq!(top.len(), 1, "{name}: a rim-to-hole section splits nothing");

            // In two pieces end to end it is still one cut, rim to hole.
            let chain = [
                segment(&mut topo, pose, [-10.0, 0.0, 10.0], [-6.0, 0.0, 10.0]),
                segment(&mut topo, pose, [-6.0, 0.0, 10.0], [-3.0, 0.0, 10.0]),
            ];
            let marked = imprint_faces(&mut topo, body, &chain).unwrap();
            check(&topo, marked, volume, 0.5, 7, name);

            // Out to the hole on both sides, the two cuts do part the top.
            let both = [
                segment(&mut topo, pose, [-10.0, 0.0, 10.0], [-3.0, 0.0, 10.0]),
                segment(&mut topo, pose, [3.0, 0.0, 10.0], [10.0, 0.0, 10.0]),
            ];
            let marked = imprint_faces(&mut topo, body, &both).unwrap();
            check(&topo, marked, volume, 0.5, 8, name);

            let far = rectangle(&mut topo, pose, other_web);
            let pieces = split_by_faces(&mut topo, body, &[web, far]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: both webs");
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 0.5, 8, name);
            }
        }
    }

    #[test]
    fn a_plane_square_to_a_bore_leaves_it_a_hole_in_the_wall() {
        let level = [
            [-15.0, -15.0, 4.0],
            [15.0, -15.0, 4.0],
            [15.0, 15.0, 4.0],
            [-15.0, 15.0, 4.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = holed_plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, level);
            let mut pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 1600.0 - PI * 36.0, 0.5, 7, name);
            check(&topo, pieces[1], 2400.0 - PI * 54.0, 0.5, 7, name);

            // Crossed by a second tool down the bore's axis, into four.
            let across = rectangle(&mut topo, pose, ACROSS);
            let pieces = split_by_faces(&mut topo, body, &[tool, across]).unwrap();
            assert_eq!(pieces.len(), 4, "{name}: crossed");
            let total: f64 = pieces
                .iter()
                .map(|&piece| solid_volume(&topo, piece, 0.01).unwrap())
                .sum();
            assert!(
                (total - (PLATE - PI * 90.0)).abs() < 0.5,
                "{name}: the pieces hold {total}"
            );
            for piece in pieces {
                assert!(validate_solid(&topo, piece).unwrap().is_valid(), "{name}");
            }
        }
    }

    /// A plane running over a notch's floor and on through the material: it
    /// lies on the body along one stretch and is a wall along the other, and
    /// the line between them is a section of the wall's outline.
    #[test]
    fn a_tool_lying_on_a_step_and_cutting_past_it() {
        let level = [
            [-15.0, -15.0, 6.0],
            [15.0, -15.0, 6.0],
            [15.0, 15.0, 6.0],
            [-15.0, 15.0, 6.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
            transform_solid(&mut topo, block, &Mat4::translation(-10.0, -10.0, 0.0)).unwrap();
            let notch = make_box(&mut topo, 15.0, 30.0, 10.0).unwrap();
            transform_solid(&mut topo, notch, &Mat4::translation(0.0, -15.0, 6.0)).unwrap();
            let stepped = boolean(&mut topo, BooleanOp::Cut, block, notch).unwrap();
            transform_solid(&mut topo, stepped, &pose).unwrap();

            let tool = rectangle(&mut topo, pose, level);
            let mut pieces = split_by_faces(&mut topo, stepped, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 800.0, 1e-6, 6, name);
            check(&topo, pieces[1], 2400.0, 1e-6, 7, name);
        }
    }

    #[test]
    fn tools_crossing_each_other_make_four_pieces() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let a = rectangle(&mut topo, pose, ACROSS);
            let b = rectangle(&mut topo, pose, ACROSS_Y);
            let pieces = split_by_faces(&mut topo, body, &[a, b]).unwrap();
            assert_eq!(pieces.len(), 4, "{name}");
            for piece in pieces {
                check(&topo, piece, 1000.0, 1e-6, 6, name);
            }

            // Two dips cross on the top, which they quarter, and cut nothing.
            let c = rectangle(&mut topo, pose, DIP);
            let d = rectangle(
                &mut topo,
                pose,
                [
                    [0.0, -15.0, 9.0],
                    [0.0, 15.0, 9.0],
                    [0.0, 15.0, 11.0],
                    [0.0, -15.0, 11.0],
                ],
            );
            let pieces = split_by_faces(&mut topo, body, &[c, d]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: dips");
            check(&topo, pieces[0], PLATE, 1e-6, 9, name);
            let tools = [ImprintTool::Face(c), ImprintTool::Face(d)];
            let marked = imprint_faces(&mut topo, body, &tools).unwrap();
            check(&topo, marked, PLATE, 1e-6, 9, name);
        }
    }

    #[test]
    fn tools_separating_only_together() {
        let half_from_the_left = [
            [-15.0, 0.0, -5.0],
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 15.0],
            [-15.0, 0.0, 15.0],
        ];
        let half_from_the_right = [
            [0.0, 0.0, -5.0],
            [15.0, 0.0, -5.0],
            [15.0, 0.0, 15.0],
            [0.0, 0.0, 15.0],
        ];
        let turning_a_corner = [
            [0.0, 0.0, -5.0],
            [0.0, 15.0, -5.0],
            [0.0, 15.0, 15.0],
            [0.0, 0.0, 15.0],
        ];
        // Each of these stops at the other's middle, past which it dangles.
        let long_arm = [
            [-15.0, 0.0, -5.0],
            [5.0, 0.0, -5.0],
            [5.0, 0.0, 15.0],
            [-15.0, 0.0, 15.0],
        ];
        let other_long_arm = [
            [0.0, -5.0, -5.0],
            [0.0, 15.0, -5.0],
            [0.0, 15.0, 15.0],
            [0.0, -5.0, 15.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let left = rectangle(&mut topo, pose, half_from_the_left);
            let right = rectangle(&mut topo, pose, half_from_the_right);
            let corner = rectangle(&mut topo, pose, turning_a_corner);

            let alone = split_by_faces(&mut topo, body, &[left]).unwrap();
            assert_eq!(alone.len(), 1, "{name}: half a wall");
            check(&topo, alone[0], PLATE, 1e-6, 7, name);

            let pieces = split_by_faces(&mut topo, body, &[left, right]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: both halves");
            for piece in pieces {
                check(&topo, piece, 2000.0, 1e-6, 7, name);
            }

            let mut sheet = split_by_faces(&mut topo, body, &[left, corner]).unwrap();
            assert_eq!(sheet.len(), 2, "{name}: an L of two faces");
            by_volume(&topo, &mut sheet);
            check(&topo, sheet[0], 1000.0, 1e-6, 6, name);
            check(&topo, sheet[1], 3000.0, 1e-6, 8, name);

            let (a, b) = (
                rectangle(&mut topo, pose, long_arm),
                rectangle(&mut topo, pose, other_long_arm),
            );
            let mut quadrant = split_by_faces(&mut topo, body, &[a, b]).unwrap();
            assert_eq!(quadrant.len(), 2, "{name}: arms crossing past their corner");
            by_volume(&topo, &mut quadrant);
            assert!(
                (solid_volume(&topo, quadrant[0], 0.01).unwrap() - 1000.0).abs() < 1e-6,
                "{name}: the walled-off quadrant"
            );
        }
    }

    #[test]
    fn a_closed_set_of_faces_inside_cuts_out_what_it_holds() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let inner = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();
            transform_solid(
                &mut topo,
                inner,
                &(pose * Mat4::translation(-1.0, -1.0, 4.0)),
            )
            .unwrap();
            let sheet = solid_faces(&topo, inner).unwrap();

            let mut pieces = split_by_faces(&mut topo, body, &sheet).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 8.0, 1e-6, 6, name);
            check(&topo, pieces[1], PLATE - 8.0, 1e-6, 12, name);
        }
    }

    #[test]
    fn a_tool_across_one_prong_of_a_fork() {
        // A plane through both prongs would take the far prong's tip as well.
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
            let notch = make_box(&mut topo, 8.0, 15.0, 20.0).unwrap();
            transform_solid(&mut topo, notch, &Mat4::translation(6.0, 5.0, -5.0)).unwrap();
            let fork = boolean(&mut topo, BooleanOp::Cut, body, notch).unwrap();
            transform_solid(&mut topo, fork, &pose).unwrap();

            let tool = rectangle(
                &mut topo,
                pose,
                [
                    [-5.0, 12.0, -5.0],
                    [10.0, 12.0, -5.0],
                    [10.0, 12.0, 15.0],
                    [-5.0, 12.0, 15.0],
                ],
            );
            let mut pieces = split_by_faces(&mut topo, fork, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 480.0, 1e-6, 6, name);
            check(&topo, pieces[1], 2320.0, 1e-6, 10, name);
        }
    }

    #[test]
    fn a_cavity_goes_with_the_piece_around_it() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let plate_id = make_box(&mut topo, 20.0, 20.0, 10.0).unwrap();
            transform_solid(&mut topo, plate_id, &Mat4::translation(-10.0, -10.0, 0.0)).unwrap();
            let void = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
            transform_solid(&mut topo, void, &Mat4::translation(-2.0, -2.0, 3.0)).unwrap();
            let hollow = boolean(&mut topo, BooleanOp::Cut, plate_id, void).unwrap();
            transform_solid(&mut topo, hollow, &pose).unwrap();

            let through = rectangle(&mut topo, pose, ACROSS);
            let pieces = split_by_faces(&mut topo, hollow, &[through]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: through the cavity");
            for piece in pieces {
                check(&topo, piece, (PLATE - 64.0) / 2.0, 1e-6, 11, name);
            }

            let beside = rectangle(
                &mut topo,
                pose,
                [
                    [5.0, -15.0, -5.0],
                    [5.0, 15.0, -5.0],
                    [5.0, 15.0, 15.0],
                    [5.0, -15.0, 15.0],
                ],
            );
            let mut pieces = split_by_faces(&mut topo, hollow, &[beside]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: beside the cavity");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 1000.0, 1e-6, 6, name);
            check(&topo, pieces[1], 3000.0 - 64.0, 1e-6, 12, name);
        }
    }

    #[test]
    fn a_curved_tool_separates_and_marks() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tube = make_cylinder(&mut topo, 4.0, 20.0).unwrap();
            transform_solid(&mut topo, tube, &(pose * Mat4::translation(0.0, 0.0, -5.0))).unwrap();
            let wall = curved_faces_of(&topo, tube);

            let core = 160.0 * PI;
            let mut pieces = split_by_faces(&mut topo, body, &wall).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], core, 0.5, 3, name);
            check(&topo, pieces[1], PLATE - core, 0.5, 7, name);

            let tools: Vec<ImprintTool> = wall.iter().map(|&f| ImprintTool::Face(f)).collect();
            let marked = imprint_faces(&mut topo, body, &tools).unwrap();
            check(&topo, marked, PLATE, 1e-6, 8, name);
        }
    }

    #[test]
    fn a_tube_crossed_by_a_plane_makes_four_pieces() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tube = make_cylinder(&mut topo, 4.0, 20.0).unwrap();
            transform_solid(&mut topo, tube, &(pose * Mat4::translation(0.0, 0.0, -5.0))).unwrap();
            let mut tools = curved_faces_of(&topo, tube);
            tools.push(rectangle(&mut topo, pose, ACROSS));

            let mut pieces = split_by_faces(&mut topo, body, &tools).unwrap();
            assert_eq!(pieces.len(), 4, "{name}");
            by_volume(&topo, &mut pieces);
            let half_core = 80.0 * PI;
            for &piece in &pieces[..2] {
                check(&topo, piece, half_core, 0.5, 4, name);
            }
            for &piece in &pieces[2..] {
                check(&topo, piece, 2000.0 - half_core, 0.5, 8, name);
            }
        }
    }

    #[test]
    fn a_balls_faces_cut_out_what_they_close() {
        let ball_volume = 36.0 * PI;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let inside = make_sphere(&mut topo, 3.0, 32).unwrap();
            transform_solid(
                &mut topo,
                inside,
                &(pose * Mat4::translation(0.0, 0.0, 5.0)),
            )
            .unwrap();
            let faces = solid_faces(&topo, inside).unwrap();

            // Neither hemisphere closes anything alone.
            let alone = split_by_faces(&mut topo, body, &faces[..1]).unwrap();
            assert_eq!(alone.len(), 1, "{name}: one hemisphere");
            check(&topo, alone[0], PLATE, 1e-6, 6, name);

            let mut pieces = split_by_faces(&mut topo, body, &faces).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the whole ball");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], ball_volume, 0.5, 2, name);
            check(&topo, pieces[1], PLATE - ball_volume, 0.5, 8, name);

            // Poking through the top, it takes out all but the cap above z = 10,
            // of height 2, and leaves a disc on the top.
            let through = make_sphere(&mut topo, 3.0, 32).unwrap();
            transform_solid(
                &mut topo,
                through,
                &(pose * Mat4::translation(0.0, 0.0, 9.0)),
            )
            .unwrap();
            let faces = solid_faces(&topo, through).unwrap();
            let cap = PI * 4.0 * 7.0 / 3.0;
            let mut pieces = split_by_faces(&mut topo, body, &faces).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: poking through");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], ball_volume - cap, 0.5, 3, name);
            check(&topo, pieces[1], PLATE - ball_volume + cap, 0.5, 8, name);

            let tools: Vec<ImprintTool> = faces.iter().map(|&f| ImprintTool::Face(f)).collect();
            let marked = imprint_faces(&mut topo, body, &tools).unwrap();
            check(&topo, marked, PLATE, 1e-6, 7, name);
        }
    }

    #[test]
    fn a_nurbs_sheet_and_a_cone_wall_separate() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            // The bulge y = 4 u (1 - u) along x = 30 u - 15, swept over z.
            let rows = [-15.0, 0.0, 15.0]
                .iter()
                .zip([0.0, 2.0, 0.0])
                .map(|(&x, y)| vec![at(pose, [x, y, -5.0]), at(pose, [x, y, 15.0])])
                .collect();
            let bulge = NurbsSurface::new(
                2,
                1,
                vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                rows,
                vec![vec![1.0; 2]; 3],
            )
            .unwrap();
            let sheet = make_nurbs_face(&mut topo, bulge, 1e-7).unwrap();
            let held_over = 170.370_370_370_370_37;
            let mut pieces = split_by_faces(&mut topo, body, &[sheet]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the sheet");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], 2000.0 - held_over, 0.5, 6, name);
            check(&topo, pieces[1], 2000.0 + held_over, 0.5, 6, name);

            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(sheet)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 10, name);

            let cone = make_cone(&mut topo, 6.0, 2.0, 20.0).unwrap();
            transform_solid(&mut topo, cone, &(pose * Mat4::translation(0.0, 0.0, -5.0))).unwrap();
            let wall = curved_faces_of(&topo, cone);
            let core = PI * 10.0 / 3.0 * (25.0 + 15.0 + 9.0);
            let mut pieces = split_by_faces(&mut topo, body, &wall).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the cone's wall");
            by_volume(&topo, &mut pieces);
            check(&topo, pieces[0], core, 0.5, 3, name);
            check(&topo, pieces[1], PLATE - core, 0.5, 7, name);
        }
    }

    #[test]
    fn a_plane_cuts_a_cone_and_halves_a_ball() {
        let across = [
            [-15.0, 0.0, -15.0],
            [15.0, 0.0, -15.0],
            [15.0, 0.0, 15.0],
            [-15.0, 0.0, 15.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let frustum = make_cone(&mut topo, 10.0, 5.0, 10.0).unwrap();
            transform_solid(&mut topo, frustum, &pose).unwrap();
            let volume = PI * 10.0 / 3.0 * (100.0 + 50.0 + 25.0);
            let tool = rectangle(&mut topo, pose, across);
            let pieces = split_by_faces(&mut topo, frustum, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the frustum");
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 0.5, 4, name);
            }

            let ball = make_sphere(&mut topo, 10.0, 32).unwrap();
            transform_solid(&mut topo, ball, &pose).unwrap();
            let volume = 4.0 / 3.0 * PI * 1000.0;
            let tool = rectangle(&mut topo, pose, across);
            let pieces = split_by_faces(&mut topo, ball, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the ball");
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 1.0, 3, name);
            }

            // Along its equator the wall has the outline of both hemispheres.
            let equator = rectangle(
                &mut topo,
                pose,
                [
                    [-15.0, -15.0, 0.0],
                    [15.0, -15.0, 0.0],
                    [15.0, 15.0, 0.0],
                    [-15.0, 15.0, 0.0],
                ],
            );
            let pieces = split_by_faces(&mut topo, ball, &[equator]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the ball's equator");
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 1.0, 2, name);
            }
        }
    }

    #[test]
    fn a_ring_is_cut_by_two_meridian_discs_and_not_by_one() {
        let meridian = [
            [0.0, 0.0, -5.0],
            [15.0, 0.0, -5.0],
            [15.0, 0.0, 5.0],
            [0.0, 0.0, 5.0],
        ];
        let opposite = [
            [-15.0, 0.0, -5.0],
            [0.0, 0.0, -5.0],
            [0.0, 0.0, 5.0],
            [-15.0, 0.0, 5.0],
        ];
        let volume = 2.0 * PI * PI * 10.0 * 9.0;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let ring_body = make_torus(&mut topo, 10.0, 3.0, 32).unwrap();
            transform_solid(&mut topo, ring_body, &pose).unwrap();

            let one = rectangle(&mut topo, pose, meridian);
            let pieces = split_by_faces(&mut topo, ring_body, &[one]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}: one disc holds nothing apart");
            check(&topo, pieces[0], volume, 1.0, 1, name);
            let marked = imprint_faces(&mut topo, ring_body, &[ImprintTool::Face(one)]).unwrap();
            check(&topo, marked, volume, 1.0, 1, name);

            let other = rectangle(&mut topo, pose, opposite);
            let pieces = split_by_faces(&mut topo, ring_body, &[one, other]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: two discs");
            for piece in pieces {
                check(&topo, piece, volume / 2.0, 1.0, 3, name);
            }
        }
    }

    /// A unit block whose top is the Bezier lid `z = 100 u (1 - u)^4 v (1 - v)`,
    /// flat along its rim, holding 1 + 5/9.
    fn block_under_a_nurbs_lid(topo: &mut Topology) -> SolidId {
        let block = make_box(topo, 1.0, 1.0, 1.0).unwrap();
        transform_solid(topo, block, &Mat4::translation(0.0, 0.0, -1.0)).unwrap();
        let top = solid_faces(topo, block)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, d }
                    if normal.z() > 0.5 && d.abs() < 1e-9)
            })
            .unwrap();
        let rows = (0..6)
            .map(|i| {
                (0..3)
                    .map(|j| {
                        let z = if (i, j) == (1, 1) { 10.0 } else { 0.0 };
                        Point3::new(f64::from(i) / 5.0, f64::from(j) / 2.0, z)
                    })
                    .collect()
            })
            .collect();
        let lid = NurbsSurface::new(
            5,
            2,
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            rows,
            vec![vec![1.0; 3]; 6],
        )
        .unwrap();
        topo.face_mut(top)
            .unwrap()
            .set_surface(FaceSurface::Nurbs(lid));
        block
    }

    #[test]
    fn a_face_across_a_nurbs_lid_marks_and_separates() {
        let volume = 1.0 + 5.0 / 9.0;
        let mut topo = Topology::new();
        let body = block_under_a_nurbs_lid(&mut topo);
        let dip = rectangle(
            &mut topo,
            Mat4::identity(),
            [
                [-1.0, 0.5, -0.5],
                [2.0, 0.5, -0.5],
                [2.0, 0.5, 5.0],
                [-1.0, 0.5, 5.0],
            ],
        );
        let pieces = split_by_faces(&mut topo, body, &[dip]).unwrap();
        assert_eq!(pieces.len(), 1, "stopped at z = -0.5, it cuts nothing");
        let held = solid_volume(&topo, pieces[0], 1e-3).unwrap();
        assert!((held - volume).abs() < 1e-2 * volume, "{held}");
        assert!(validate_solid(&topo, pieces[0]).unwrap().is_valid());
        let lid_pieces = solid_faces(&topo, pieces[0])
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Nurbs(_)))
            .count();
        assert_eq!(lid_pieces, 2, "the lid comes apart along the section");

        let across = rectangle(
            &mut topo,
            Mat4::identity(),
            [
                [-1.0, 0.5, -2.0],
                [2.0, 0.5, -2.0],
                [2.0, 0.5, 5.0],
                [-1.0, 0.5, 5.0],
            ],
        );
        let pieces = split_by_faces(&mut topo, body, &[across]).unwrap();
        assert_eq!(pieces.len(), 2, "down to below the block");
        let total: f64 = pieces
            .iter()
            .map(|&piece| {
                assert!(validate_solid(&topo, piece).unwrap().is_valid());
                solid_volume(&topo, piece, 1e-3).unwrap()
            })
            .sum();
        assert!((total - volume).abs() < 1e-2 * volume, "{total}");
    }

    #[test]
    fn a_tool_through_two_edges_splits_along_them() {
        let diagonal = [
            [-15.0, -15.0, -5.0],
            [15.0, 15.0, -5.0],
            [15.0, 15.0, 15.0],
            [-15.0, -15.0, 15.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, diagonal);
            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            assert_eq!(pieces.len(), 2, "{name}");
            for piece in pieces {
                check(&topo, piece, 2000.0, 1e-6, 5, name);
            }
        }
    }

    /// A sheet swept from an open wire steps across the box: only its three
    /// faces together close one side off from the other.
    #[test]
    fn a_stepped_sheet_separates_what_its_faces_do_not_alone() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let steps = [
                [-15.0, 2.0, -5.0],
                [0.0, 2.0, -5.0],
                [0.0, -2.0, -5.0],
                [15.0, -2.0, -5.0],
            ];
            let edges: Vec<OrientedEdge> = steps
                .windows(2)
                .map(|w| {
                    let edge = make_line_edge(&mut topo, at(pose, w[0]), at(pose, w[1]), 1e-7);
                    OrientedEdge::new(edge.unwrap(), true)
                })
                .collect();
            let wire = topo.add_wire(Wire::new(edges, false).unwrap());
            let lift = (at(pose, [0.0, 0.0, 1.0]) - at(pose, [0.0, 0.0, 0.0]))
                .normalize()
                .unwrap();
            let sheet = extrude_wire(&mut topo, wire, lift, 20.0).unwrap();
            let faces = topo.shell(sheet).unwrap().faces().to_vec();
            assert_eq!(faces.len(), 3, "{name}: three steps");

            let alone = split_by_faces(&mut topo, body, &faces[..1]).unwrap();
            assert_eq!(alone.len(), 1, "{name}: one step");

            let pieces = split_by_faces(&mut topo, body, &faces).unwrap();
            assert_eq!(pieces.len(), 2, "{name}: the whole sheet");
            for piece in pieces {
                check(&topo, piece, 2000.0, 1e-6, 8, name);
            }
        }
    }

    #[test]
    fn a_faces_own_face_as_a_tool_changes_nothing() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let own = solid_faces(&topo, body).unwrap()[0];
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(own)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);
            let pieces = split_by_faces(&mut topo, body, &[own]).unwrap();
            assert_eq!(pieces.len(), 1, "{name}");
            check(&topo, pieces[0], PLATE, 1e-6, 6, name);
        }
    }

    #[test]
    fn a_tool_on_a_faces_own_outline_changes_nothing() {
        let over_the_top = [
            [-15.0, -15.0, 10.0],
            [15.0, -15.0, 10.0],
            [15.0, 15.0, 10.0],
            [-15.0, 15.0, 10.0],
        ];
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, over_the_top);
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);
            let along_a_rim = segment(&mut topo, pose, [-10.0, -10.0, 10.0], [10.0, -10.0, 10.0]);
            let marked = imprint_faces(&mut topo, body, &[along_a_rim]).unwrap();
            check(&topo, marked, PLATE, 1e-6, 6, name);
        }
    }

    #[test]
    fn imprinting_twice_is_imprinting_once() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, DIP);
            let once = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();
            let twice = imprint_faces(&mut topo, once, &[ImprintTool::Face(tool)]).unwrap();
            check(&topo, twice, PLATE, 1e-6, 7, name);
            assert_halved_top(&topo, twice, pose, name);
        }
    }

    #[test]
    fn an_imprinted_solid_goes_on_to_a_later_boolean() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, DIP);
            let marked = imprint_faces(&mut topo, body, &[ImprintTool::Face(tool)]).unwrap();

            // A pocket across the mark, its walls crossing the mark's line.
            let pocket = make_box(&mut topo, 8.0, 8.0, 4.0).unwrap();
            transform_solid(
                &mut topo,
                pocket,
                &(pose * Mat4::translation(-4.0, -4.0, 7.0)),
            )
            .unwrap();
            let before = mesh_fallback_count();
            let cut = boolean(&mut topo, BooleanOp::Cut, marked, pocket).unwrap();
            assert_eq!(mesh_fallback_count(), before, "{name}: stayed exact");
            // Two pieces of the top around the pocket, its floor and walls
            // (each cut by the mark's plane only where the top was), and the
            // plate's five other sides.
            let report = validate_solid(&topo, cut).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, cut, 0.02).unwrap();
            assert!(is_watertight(&mesh), "{name}: the cut leaks");
            let volume = solid_volume(&topo, cut, 0.005).unwrap();
            assert!((volume - (PLATE - 192.0)).abs() < 1e-6, "{name}: {volume}");

            // And a block fused over the mark.
            let block = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();
            transform_solid(
                &mut topo,
                block,
                &(pose * Mat4::translation(6.0, -2.0, 8.0)),
            )
            .unwrap();
            let before = mesh_fallback_count();
            let fused = boolean(&mut topo, BooleanOp::Fuse, marked, block).unwrap();
            assert_eq!(mesh_fallback_count(), before, "{name}: fuse stayed exact");
            let volume = solid_volume(&topo, fused, 0.005).unwrap();
            assert!((volume - (PLATE + 32.0)).abs() < 1e-6, "{name}: {volume}");
            assert!(validate_solid(&topo, fused).unwrap().is_valid(), "{name}");
        }
    }

    #[test]
    fn a_piece_of_a_split_goes_on_to_a_later_boolean() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let body = holed_plate(&mut topo, pose);
            let tool = rectangle(&mut topo, pose, ACROSS);
            let pieces = split_by_faces(&mut topo, body, &[tool]).unwrap();
            let refused = boolean(&mut topo, BooleanOp::Fuse, pieces[0], pieces[1]).unwrap();
            let volume = solid_volume(&topo, refused, 0.005).unwrap();
            assert!(
                (volume - (PLATE - PI * 90.0)).abs() < 0.5,
                "{name}: the halves fuse back to {volume}"
            );
        }
    }

    #[test]
    fn a_handle_from_elsewhere_is_refused() {
        let mut topo = Topology::new();
        let body = plate(&mut topo, Mat4::identity());
        let mut other = Topology::new();
        for _ in 0..50 {
            make_box(&mut other, 1.0, 1.0, 1.0).unwrap();
        }
        let last = make_box(&mut other, 1.0, 1.0, 1.0).unwrap();
        let foreign_face = solid_faces(&other, last).unwrap().pop().unwrap();
        assert!(imprint_faces(&mut topo, body, &[ImprintTool::Face(foreign_face)]).is_err());
        assert!(split_by_faces(&mut topo, body, &[foreign_face]).is_err());

        let empty = topo.add_empty_solid();
        assert!(imprint_faces(&mut topo, empty, &[]).is_err());
        assert!(split_by_faces(&mut topo, empty, &[]).is_err());
    }
}

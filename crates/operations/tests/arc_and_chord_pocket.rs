//! A plane face that holds an arc and the line between its ends: the segment
//! between them is a face of its own.
//!
//! Edges imprinted on a plate's top (an arc about (5, 4) of radius root 2 from
//! (6, 5) to (4, 5), its chord, and lines from the chord's ends to the rim)
//! split off the segment, and a D pocket under the arc and its chord, and a
//! round pocket whose floor a box cuts along the same chord, keep their
//! closed-form volumes. Every case runs upright, turned and mirrored.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::imprint::{ImprintTool, imprint_faces};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_line_edge;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// The area of the segment between the arc and its chord.
    const SEGMENT: f64 = FRAC_PI_2 - 1.0;

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

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    /// The circle about (5, 4) of radius root 2 in the plane at height `z`,
    /// posed, so that it runs from (6, 5) to (4, 5) the short way round.
    fn circle(pose: Mat4, z: f64) -> Circle3D {
        let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, z));
        let (u, v) = (at(1.0, 0.0) - at(0.0, 0.0), at(0.0, 1.0) - at(0.0, 0.0));
        Circle3D::with_axes(at(5.0, 4.0), u.cross(v), SQRT_2, u, v).unwrap()
    }

    /// The posed plate, 10 by 10 by 2, its top imprinted with the arc, its
    /// chord, and lines from (6, 5) to `far.0` and from `far.1` to (4, 5).
    fn imprinted(
        pose: Mat4,
        far: ((f64, f64), (f64, f64)),
        chord_first: bool,
    ) -> (Topology, SolidId) {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
        transform_solid(&mut topo, plate, &pose).unwrap();
        let at = |(x, y): (f64, f64)| pose.mul_point(Point3::new(x, y, 2.0));
        let a = topo.add_vertex(Vertex::new(at((6.0, 5.0)), 1e-7));
        let b = topo.add_vertex(Vertex::new(at((4.0, 5.0)), 1e-7));
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle(pose, 2.0))));
        let chord = make_line_edge(&mut topo, at((4.0, 5.0)), at((6.0, 5.0)), 1e-7).unwrap();
        let to = make_line_edge(&mut topo, at((6.0, 5.0)), at(far.0), 1e-7).unwrap();
        let from = make_line_edge(&mut topo, at(far.1), at((4.0, 5.0)), 1e-7).unwrap();
        let mut tools = vec![ImprintTool::Edge(arc), ImprintTool::Edge(chord)];
        if chord_first {
            tools.swap(0, 1);
        }
        tools.extend([ImprintTool::Edge(to), ImprintTool::Edge(from)]);
        let result = imprint_faces(&mut topo, plate, &tools).unwrap();
        (topo, result)
    }

    /// The areas of the solid's faces in the posed plate's top, sorted.
    fn top_areas(topo: &Topology, solid: SolidId, pose: Mat4) -> Vec<f64> {
        let origin = pose.mul_point(Point3::new(0.0, 0.0, 0.0));
        let top = pose.mul_point(Point3::new(0.0, 0.0, 2.0));
        let up = (top - origin).normalize().unwrap();
        let mut areas: Vec<f64> = solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&f| {
                let face = topo.face(f).unwrap();
                matches!(face.surface(), FaceSurface::Plane { .. })
                    && topo
                        .wire(face.outer_wire())
                        .unwrap()
                        .edges()
                        .iter()
                        .all(|oe| {
                            let start = topo.edge(oe.edge()).unwrap().start();
                            let p = topo.vertex(start).unwrap().point();
                            (p - top).dot(up).abs() < 1e-9
                        })
            })
            .map(|f| face_area(topo, f, 0.01).unwrap())
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    /// The arc and its chord on a line across the top, and the arc, its chord
    /// and two lines up to (5, 10) on the rim, imprinted in either order: the
    /// segment is a face of the top, and the solid stays valid, closed and of
    /// the plate's volume.
    #[test]
    fn imprinting_an_arc_and_its_chord_splits_off_their_segment() {
        let cases = [
            (((10.0, 5.0), (0.0, 5.0)), [SEGMENT, 50.0 - SEGMENT, 50.0]),
            (((5.0, 10.0), (5.0, 10.0)), [SEGMENT, 5.0 - SEGMENT, 95.0]),
        ];
        for (far, want) in cases {
            for (name, pose) in poses() {
                for chord_first in [false, true] {
                    let label = format!("{far:?} {name}, chord first {chord_first}");
                    let (topo, solid) = imprinted(pose, far, chord_first);
                    let areas = top_areas(&topo, solid, pose);
                    assert!(
                        areas.len() == want.len()
                            && areas.iter().zip(want).all(|(a, w)| (a - w).abs() < 1e-9),
                        "{label}: top faces of area {areas:?}, expected {want:?}"
                    );
                    let report = validate_solid(&topo, solid).unwrap();
                    assert!(report.is_valid(), "{label}: {:?}", report.issues);
                    let mesh = tessellate_solid(&topo, solid, 0.02).unwrap();
                    assert!(is_watertight(&mesh), "{label}: open mesh");
                    let volume = solid_volume(&topo, solid, 0.005).unwrap();
                    assert!((volume - 200.0).abs() < 1e-9, "{label}: volume {volume}");
                }
            }
        }
    }

    /// The D under the arc and its chord at height 1, extruded 2 up, posed.
    fn d_prism(topo: &mut Topology, pose: Mat4) -> SolidId {
        let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 1.0));
        let a = topo.add_vertex(Vertex::new(at(6.0, 5.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(at(4.0, 5.0), 1e-7));
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle(pose, 1.0))));
        let chord = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
        let wire = Wire::new(
            vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)],
            true,
        )
        .unwrap();
        let wire = topo.add_wire(wire);
        let up = at(0.0, 0.0) - pose.mul_point(Point3::new(0.0, 0.0, 0.0));
        let up = up.normalize().unwrap();
        let normal = (at(1.0, 0.0) - at(0.0, 0.0)).cross(at(0.0, 1.0) - at(0.0, 0.0));
        let normal = normal.normalize().unwrap();
        let d = normal.dot(at(0.0, 0.0) - Point3::new(0.0, 0.0, 0.0));
        let face = topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }));
        extrude(topo, face, up, 2.0).unwrap()
    }

    /// The result is exact, valid and closed, holds the volume `want`, and
    /// reads each probe (a point of the plate before the pose) as it says.
    fn check(
        topo: &Topology,
        result: SolidId,
        pose: Mat4,
        faces: usize,
        want: f64,
        probes: &[([f64; 3], bool)],
        label: &str,
    ) {
        let report = validate_solid(topo, result).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let n = solid_faces(topo, result).unwrap().len();
        assert_eq!(n, faces, "{label}: {n} faces");
        let mesh = tessellate_solid(topo, result, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{label}: open mesh");
        let volume = solid_volume(topo, result, 0.005).unwrap();
        assert!(
            (volume - want).abs() < 1e-9,
            "{label}: volume {volume}, want {want}"
        );
        for &(p, inside) in probes {
            let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
            let got = classify_point(topo, result, q, 0.01, 1e-7).unwrap();
            assert_eq!(
                matches!(got, PointClassification::Inside),
                inside,
                "{label}: at {p:?}"
            );
        }
    }

    /// A D pocket 1 deep under the arc and its chord.
    #[test]
    fn a_d_pocket_under_an_arc_and_its_chord_keeps_its_volume() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
            transform_solid(&mut topo, plate, &pose).unwrap();
            let tool = d_prism(&mut topo, pose);
            let result =
                boolean_with_options(&mut topo, BooleanOp::Cut, plate, tool, exact()).unwrap();
            let probes = [
                ([5.0, 5.2, 1.5], false),
                ([5.0, 4.8, 1.5], true),
                ([5.0, 5.5, 1.5], true),
                ([5.0, 5.2, 0.5], true),
            ];
            check(&topo, result, pose, 9, 200.0 - SEGMENT, &probes, name);
        }
    }

    /// A round pocket 1 deep about (5, 4) and a box 1.5 deep over y from 5
    /// to 9 and x from 2 to 8, whose side y = 5 cuts the pocket's floor along
    /// the chord from (4, 5) to (6, 5): one after the other in both orders,
    /// and as one fused tool.
    #[test]
    fn a_round_pocket_whose_floor_a_box_cuts_along_a_chord_keeps_its_volume() {
        let want = 200.0 - 2.0 * PI - 36.0 + SEGMENT;
        let probes = [
            ([5.0, 4.0, 0.9], true),
            ([5.0, 4.0, 1.1], false),
            ([5.0, 5.2, 0.9], false),
            ([5.0, 5.2, 0.4], true),
            ([5.0, 3.0, 1.5], false),
            ([1.0, 1.0, 1.5], true),
        ];
        for (name, pose) in poses() {
            for order in ["round first", "box first", "fused"] {
                let mut topo = Topology::new();
                let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
                let round = make_cylinder(&mut topo, SQRT_2, 2.0).unwrap();
                transform_solid(&mut topo, round, &Mat4::translation(5.0, 4.0, 1.0)).unwrap();
                let bx = make_box(&mut topo, 6.0, 4.0, 2.5).unwrap();
                transform_solid(&mut topo, bx, &Mat4::translation(2.0, 5.0, 0.5)).unwrap();
                for solid in [plate, round, bx] {
                    transform_solid(&mut topo, solid, &pose).unwrap();
                }
                let cut = |topo: &mut Topology, a: SolidId, b: SolidId| {
                    boolean_with_options(topo, BooleanOp::Cut, a, b, exact()).unwrap()
                };
                let result = match order {
                    "round first" => {
                        let once = cut(&mut topo, plate, round);
                        cut(&mut topo, once, bx)
                    }
                    "box first" => {
                        let once = cut(&mut topo, plate, bx);
                        cut(&mut topo, once, round)
                    }
                    _ => {
                        let tool =
                            boolean_with_options(&mut topo, BooleanOp::Fuse, round, bx, exact())
                                .unwrap();
                        cut(&mut topo, plate, tool)
                    }
                };
                check(
                    &topo,
                    result,
                    pose,
                    13,
                    want,
                    &probes,
                    &format!("{name}, {order}"),
                );
            }
        }
    }
}

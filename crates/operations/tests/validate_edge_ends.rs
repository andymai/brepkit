//! An edge's curve passes through its vertices. Read at the ends of the
//! edge's span, as every reader of the edge reads it, the curve gives the
//! vertices' points. A chamfer once left an arc whose end vertex lay 7 off
//! its circle, and the operations `validate_solid` reported it only as a
//! wire that crossed itself.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::nurbs::NurbsCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::primitives::{
        make_box, make_cone, make_cylinder, make_sphere, make_torus,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.7)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// What `validate_solid` reports of curves that miss their vertices.
    fn misses(topo: &Topology, solid: SolidId) -> Vec<String> {
        validate_solid(topo, solid)
            .unwrap()
            .issues
            .into_iter()
            .map(|issue| issue.description)
            .filter(|text| text.contains("its curve misses"))
            .collect()
    }

    /// The edge of `solid` from `a` to `b`.
    fn edge_from(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let p = topo.vertex(edge.start()).unwrap().point();
                let q = topo.vertex(edge.end()).unwrap().point();
                (p - a).length() < 1e-12 && (q - b).length() < 1e-12
            })
            .unwrap()
    }

    /// A 2 x 3 x 4 box whose edge from the origin to `(2, 0, 0)` runs on a
    /// straight B-spline through `from` and `to`, with a tolerance of its
    /// own if `tolerance` gives one, placed by `pose`.
    fn box_on_spline(
        topo: &mut Topology,
        from: Point3,
        to: Point3,
        tolerance: Option<f64>,
        pose: &Mat4,
    ) -> SolidId {
        let solid = make_box(topo, 2.0, 3.0, 4.0).unwrap();
        let edge = edge_from(
            topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 0.0),
        );
        let line =
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![from, to], vec![1.0; 2]).unwrap();
        let edge = topo.edge_mut(edge).unwrap();
        edge.set_curve(EdgeCurve::NurbsCurve(line));
        edge.set_tolerance(tolerance);
        transform_solid(topo, solid, pose).unwrap();
        solid
    }

    /// A D: half a disc of radius 5 about the origin, closed by a straight
    /// back wall on `x = 0`, 4 high. Its bottom arc runs on a circle of
    /// `radius` about the origin, placed by `pose`.
    fn d_prism(topo: &mut Topology, radius: f64, pose: &Mat4) -> SolidId {
        let a = topo.add_vertex(Vertex::new(Point3::new(0.0, -5.0, 0.0), 1e-7));
        let b = topo.add_vertex(Vertex::new(Point3::new(0.0, 5.0, 0.0), 1e-7));
        let circle = |r: f64| {
            EdgeCurve::Circle(
                Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), r).unwrap(),
            )
        };
        let arc = topo.add_edge(Edge::new(a, b, circle(5.0)));
        let edges = vec![
            OrientedEdge::new(arc, true),
            OrientedEdge::new(topo.add_edge(Edge::new(b, a, EdgeCurve::Line)), true),
        ];
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let face = topo.add_face(Face::new(wire, Vec::new(), plane));
        let solid = extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), 4.0).unwrap();
        let bottom = solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let p = topo.vertex(edge.start()).unwrap().point();
                matches!(edge.curve(), EdgeCurve::Circle(_)) && p.z().abs() < 1e-12
            })
            .unwrap();
        topo.edge_mut(bottom).unwrap().set_curve(circle(radius));
        transform_solid(topo, solid, pose).unwrap();
        solid
    }

    #[test]
    fn primitives_pass_through_their_vertices() {
        for (pose_name, pose) in poses() {
            let mut topo = Topology::new();
            let solids = [
                make_box(&mut topo, 2.0, 3.0, 4.0).unwrap(),
                make_cylinder(&mut topo, 3.0, 4.0).unwrap(),
                make_cone(&mut topo, 3.0, 1.0, 4.0).unwrap(),
                make_cone(&mut topo, 3.0, 0.0, 4.0).unwrap(),
                make_sphere(&mut topo, 2.5, 16).unwrap(),
                make_torus(&mut topo, 4.0, 1.0, 16).unwrap(),
                d_prism(&mut topo, 5.0, &Mat4::identity()),
            ];
            for solid in solids {
                transform_solid(&mut topo, solid, &pose).unwrap();
                assert_eq!(misses(&topo, solid), Vec::<String>::new(), "{pose_name}");
            }
        }
    }

    /// A B-spline may run either way along its edge and past its ends: read
    /// over the span its vertices give, it passes through them. It may end
    /// off a vertex by less than the band in which its readers take the
    /// vertex to lie on it, or by less than its own tolerance.
    #[test]
    fn a_spline_edge_through_its_vertices_passes() {
        let o = Point3::new(0.0, 0.0, 0.0);
        let x = Point3::new(2.0, 0.0, 0.0);
        for (pose_name, pose) in poses() {
            for (from, to, tolerance) in [
                (o, x, None),
                (x, o, None),
                (
                    Point3::new(-1.0, 0.0, 0.0),
                    Point3::new(3.0, 0.0, 0.0),
                    None,
                ),
                (o, Point3::new(2.0, 0.0, 1e-9), None),
                (o, Point3::new(2.0, 0.0, 5e-6), None),
                (o, Point3::new(2.0, 0.0, 5e-4), Some(1e-3)),
            ] {
                let mut topo = Topology::new();
                let solid = box_on_spline(&mut topo, from, to, tolerance, &pose);
                let report = validate_solid(&topo, solid).unwrap();
                assert!(
                    report.is_valid(),
                    "{pose_name} {from:?} {to:?}: {:?}",
                    report.issues
                );
            }
        }
    }

    /// A B-spline's two ends, and each vertex it misses with the reading.
    type Case = (Point3, Point3, &'static [(&'static str, f64)]);

    /// A B-spline that ends off its vertex, past the band in which its
    /// readers take the vertex to lie on it, misses it. A B-spline that runs
    /// against its edge is then read the way it runs, so each end misses its
    /// vertex by the edge's length.
    #[test]
    fn a_spline_edge_that_misses_its_vertex_is_reported() {
        let o = Point3::new(0.0, 0.0, 0.0);
        let x = Point3::new(2.0, 0.0, 0.0);
        let cases: [Case; 4] = [
            (o, Point3::new(2.0, 0.0, 1e-3), &[("end", 1e-3)]),
            (o, Point3::new(2.0, 0.0, 2e-5), &[("end", 2e-5)]),
            (Point3::new(0.0, 0.0, 1e-3), x, &[("start", 1e-3)]),
            (
                x,
                Point3::new(0.0, 0.0, 2e-5),
                &[("start", 2.0), ("end", 2.0)],
            ),
        ];
        for (pose_name, pose) in poses() {
            for (from, to, expected) in cases {
                let mut topo = Topology::new();
                let solid = box_on_spline(&mut topo, from, to, None, &pose);
                let found = misses(&topo, solid);
                let what = format!("{pose_name} {from:?} {to:?}");
                assert_eq!(found.len(), expected.len(), "{what}: {found:?}");
                for &(vertex, gap) in expected {
                    let reading = found
                        .iter()
                        .find(|text| text.contains(&format!("misses its {vertex} vertex")))
                        .and_then(|text| text.split("deviates ").nth(1))
                        .and_then(|rest| rest.split(' ').next())
                        .and_then(|number| number.parse::<f64>().ok());
                    assert!(
                        reading.is_some_and(|r| (r - gap).abs() <= 0.01 * gap),
                        "{what}: {found:?}"
                    );
                }
            }
        }
    }

    /// A circle that misses its vertices: an arc's two ends, and a closed
    /// rim's one vertex.
    #[test]
    fn a_circle_that_misses_its_vertices_is_reported() {
        for (pose_name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = d_prism(&mut topo, 5.5, &pose);
            let found = misses(&topo, solid);
            assert_eq!(found.len(), 2, "{pose_name}: {found:?}");
            for vertex in ["start", "end"] {
                assert!(
                    found
                        .iter()
                        .any(|text| text.contains(&format!("misses its {vertex} vertex"))
                            && text.contains("deviates 5.00e-1")),
                    "{pose_name}: {found:?}"
                );
            }

            let mut topo = Topology::new();
            let solid = make_cylinder(&mut topo, 3.0, 4.0).unwrap();
            let rim = solid_edges(&topo, solid)
                .unwrap()
                .into_iter()
                .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
                .unwrap();
            let EdgeCurve::Circle(circle) = topo.edge(rim).unwrap().curve().clone() else {
                unreachable!()
            };
            let wider = Circle3D::new(circle.center(), circle.normal(), 3.25).unwrap();
            topo.edge_mut(rim)
                .unwrap()
                .set_curve(EdgeCurve::Circle(wider));
            transform_solid(&mut topo, solid, &pose).unwrap();
            let found = misses(&topo, solid);
            assert_eq!(found.len(), 1, "{pose_name}: {found:?}");
            assert!(
                found[0].contains("misses its one vertex") && found[0].contains("2.50e-1"),
                "{pose_name}: {found:?}"
            );
        }
    }
}

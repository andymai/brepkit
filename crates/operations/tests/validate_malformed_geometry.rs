//! A solid built in memory can hold a B-spline curve or surface that no
//! evaluation can walk: a degree past its control points, knots that fall,
//! a weight of zero or below the normal floats, a span whose reciprocal
//! overflows, or weights whose product with the coordinates overflows. The
//! validators report it, read nothing of it, and never panic. A vertex id
//! that is not in the topology is an error, not a pass.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_check::validate::{
        CheckId, ValidateOptions, check_vertex_on_curve, validate_shell,
        validate_solid as check_solid,
    };
    use brepkit_math::nurbs::{NurbsCurve, NurbsSurface};
    use brepkit_math::vec::Point3;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::validate::{validate_solid, validate_solid_relaxed};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::{Vertex, VertexId};

    /// A 2 x 3 x 4 box and its edge from the origin to `(2, 0, 0)`.
    fn boxed() -> (Topology, SolidId, EdgeId) {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let edge = solid_edges(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                a.x().abs() + a.y().abs() + a.z().abs() < 1e-12
                    && (b - Point3::new(2.0, 0.0, 0.0)).length() < 1e-12
            })
            .unwrap();
        (topo, solid, edge)
    }

    /// The edges' curves and the faces' surfaces no evaluation can walk.
    fn malformed() -> (Vec<NurbsCurve>, Vec<NurbsSurface>) {
        let p = |x: f64, y: f64| Point3::new(x, y, 0.0);
        let line = |degree: usize, knots: Vec<f64>, weights: Vec<f64>| {
            NurbsCurve::new(degree, knots, vec![p(0.0, 0.0), p(2.0, 0.0)], weights).unwrap()
        };
        let unit = vec![0.0, 0.0, 1.0, 1.0];
        let curves = vec![
            line(2, vec![0.0, 0.0, 0.0, 1.0, 1.0], vec![1.0; 2]),
            line(1, vec![1.0, 1.0, 0.0, 0.0], vec![1.0; 2]),
            line(1, unit.clone(), vec![1.0, 0.0]),
            line(1, unit.clone(), vec![1.0, 1e-310]),
            line(1, vec![0.0, 0.0, 1e-310, 1e-310], vec![1.0; 2]),
            line(1, unit.clone(), vec![1e308; 2]),
        ];
        let patch = |degree_u: usize, knots_u: Vec<f64>, weights: Vec<Vec<f64>>| {
            NurbsSurface::new(
                degree_u,
                1,
                knots_u,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![
                    vec![p(0.0, 0.0), p(0.0, 3.0)],
                    vec![p(2.0, 0.0), p(2.0, 3.0)],
                ],
                weights,
            )
            .unwrap()
        };
        let ones = vec![vec![1.0; 2]; 2];
        let surfaces = vec![
            patch(2, vec![0.0, 0.0, 0.0, 1.0, 1.0], ones.clone()),
            patch(1, vec![1.0, 1.0, 0.0, 0.0], ones.clone()),
            patch(1, unit.clone(), vec![vec![1.0, 0.0], vec![1.0; 2]]),
            patch(1, unit.clone(), vec![vec![1.0, 1e-310], vec![1.0; 2]]),
            patch(1, vec![0.0, 0.0, 1e-310, 1e-310], ones),
            patch(1, unit, vec![vec![1e308; 2]; 2]),
        ];
        (curves, surfaces)
    }

    /// Each validator's errors on `solid`, which must not panic.
    fn errors(topo: &Topology, solid: SolidId) -> [Vec<String>; 4] {
        use brepkit_check::validate::Severity as CheckSeverity;
        use brepkit_operations::validate::Severity;
        let ops = |report: brepkit_operations::validate::ValidationReport| {
            report
                .issues
                .into_iter()
                .filter(|i| i.severity == Severity::Error)
                .map(|i| i.description)
                .collect::<Vec<_>>()
        };
        let check = |report: brepkit_check::validate::ValidationReport| {
            report
                .issues
                .into_iter()
                .filter(|i| i.severity == CheckSeverity::Error)
                .map(|i| format!("{:?}: {}", i.check, i.description))
                .collect::<Vec<_>>()
        };
        let options = ValidateOptions::default();
        let shell = topo.solid(solid).unwrap().outer_shell();
        [
            ops(validate_solid(topo, solid).unwrap()),
            ops(validate_solid_relaxed(topo, solid).unwrap()),
            check(check_solid(topo, solid, &options).unwrap()),
            check(validate_shell(topo, shell, &options).unwrap()),
        ]
    }

    #[test]
    fn a_malformed_curve_or_surface_is_reported_and_not_read() {
        let (topo, solid, _) = boxed();
        for found in errors(&topo, solid) {
            assert!(found.is_empty(), "{found:?}");
        }
        let (curves, surfaces) = malformed();
        for curve in curves {
            let (mut topo, solid, edge) = boxed();
            topo.edge_mut(edge)
                .unwrap()
                .set_curve(EdgeCurve::NurbsCurve(curve.clone()));
            for found in errors(&topo, solid) {
                assert!(
                    found.len() == 1 && found[0].contains("edge's curve is malformed"),
                    "{curve:?}: {found:?}"
                );
            }
        }
        for surface in surfaces {
            let (mut topo, solid, _) = boxed();
            let bottom = solid_faces(&topo, solid)
                .unwrap()
                .into_iter()
                .find(|&f| {
                    matches!(topo.face(f).unwrap().surface(),
                        FaceSurface::Plane { normal, d } if normal.z() < -0.5 && d.abs() < 1e-12)
                })
                .unwrap();
            topo.face_mut(bottom)
                .unwrap()
                .set_surface(FaceSurface::Nurbs(surface.clone()));
            for found in errors(&topo, solid) {
                assert!(
                    found.len() == 1 && found[0].contains("face's surface is malformed"),
                    "{surface:?}: {found:?}"
                );
            }
        }
    }

    /// A vertex id from another topology, past this one's vertices.
    fn dangling() -> VertexId {
        let mut other = Topology::new();
        (0..1000)
            .map(|_| other.add_vertex(Vertex::new(Point3::new(9.0, 9.0, 9.0), 1e-7)))
            .last()
            .unwrap()
    }

    #[test]
    fn a_dangling_vertex_is_an_error() {
        let options = ValidateOptions::default();
        for curve in [
            EdgeCurve::Line,
            EdgeCurve::NurbsCurve(
                NurbsCurve::new(
                    1,
                    vec![0.0, 0.0, 1.0, 1.0],
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(2.0, 0.0, 0.0)],
                    vec![1.0; 2],
                )
                .unwrap(),
            ),
        ] {
            let (topo, _, edge) = boxed();
            assert!(check_vertex_on_curve(&topo, dangling(), edge, 1e-7).is_err());
            for end in [false, true] {
                let (mut topo, solid, edge) = boxed();
                let held = topo.edge(edge).unwrap().start();
                let data = topo.edge_mut(edge).unwrap();
                data.set_curve(curve.clone());
                if end {
                    data.set_end(dangling());
                } else {
                    data.set_start(dangling());
                }
                assert!(check_vertex_on_curve(&topo, held, edge, 1e-7).is_err());
                assert!(validate_solid(&topo, solid).is_err());
                assert!(validate_solid_relaxed(&topo, solid).is_err());
                assert!(check_solid(&topo, solid, &options).is_err());
            }
        }
    }

    /// A line's ends are its vertices; another vertex is measured against
    /// the segment between them.
    #[test]
    fn a_vertex_off_a_line_is_measured_against_it() {
        let (mut topo, _, edge) = boxed();
        let on = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let off = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.5, 0.0), 1e-7));
        assert!(
            check_vertex_on_curve(&topo, on, edge, 1e-7)
                .unwrap()
                .is_empty()
        );
        let found = check_vertex_on_curve(&topo, off, edge, 1e-7).unwrap();
        assert!(
            found.len() == 1 && found[0].check == CheckId::VertexOnCurve,
            "{found:?}"
        );
    }
}

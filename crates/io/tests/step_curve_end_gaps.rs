//! Parts read from STEP files whose B-spline edges end off their vertices.
//!
//! A file gives each vertex as a point and each curve as its control
//! points, and has no tolerance for a vertex. In these three parts, written
//! by brepkit's STEP writer, curves end up to 1.06e-3 off their vertices.
//! Each such vertex takes the gap as its tolerance, so the part reads as a
//! valid solid, and the vertices with no gap keep the file's tolerance.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::HashMap;
    use std::path::Path;

    use brepkit_io::step::reader::read_step;
    use brepkit_math::mat::Mat4;
    use brepkit_math::nurbs::projection::project_point_to_curve;
    use brepkit_math::vec::Point3;
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::{solid_edges, solid_vertices};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::VertexId;

    fn load(topo: &mut Topology, name: &str) -> SolidId {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        read_step(&std::fs::read_to_string(path).unwrap(), topo).unwrap()[0]
    }

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(30.0, -20.0, 50.0)
                    * Mat4::rotation_x(0.4)
                    * Mat4::rotation_y(1.1)
                    * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// For each vertex of `solid`, the largest distance from it to the end
    /// of one of its edges' curves, read at the end of the edge's span, or
    /// to the curve of a closed edge.
    fn gaps(topo: &Topology, solid: SolidId) -> HashMap<VertexId, f64> {
        let mut gaps = HashMap::new();
        for e in solid_edges(topo, solid).unwrap() {
            let edge = topo.edge(e).unwrap();
            if edge.start() == edge.end() {
                let p = topo.vertex(edge.start()).unwrap().point();
                let gap = match edge.curve() {
                    EdgeCurve::NurbsCurve(curve) => {
                        project_point_to_curve(curve, p, 1e-10).unwrap().distance
                    }
                    EdgeCurve::Circle(circle) => (circle.evaluate(circle.project(p)) - p).length(),
                    EdgeCurve::Ellipse(ellipse) => {
                        (ellipse.evaluate(ellipse.project(p)) - p).length()
                    }
                    EdgeCurve::Line => 0.0,
                };
                let entry = gaps.entry(edge.start()).or_insert(0.0_f64);
                *entry = entry.max(gap);
                continue;
            }
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for (v, p, t) in [(edge.start(), a, t0), (edge.end(), b, t1)] {
                let gap = (edge.curve().evaluate_with_endpoints(t, a, b) - p).length();
                let entry = gaps.entry(v).or_insert(0.0_f64);
                *entry = entry.max(gap);
            }
        }
        gaps
    }

    #[test]
    fn parts_whose_curves_end_off_their_vertices_read_as_valid_solids() {
        for (name, worst) in [
            ("leg_ledge_tips.step", 1.062e-3),
            ("leg_ledge_wall_corner.step", 1.046e-3),
            ("wing_rib_ends.step", 9.649e-4),
        ] {
            for (pose, at) in poses() {
                let mut topo = Topology::new();
                let solid = load(&mut topo, name);
                transform_solid(&mut topo, solid, &at).unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{name} {pose}: {:?}", report.issues);
                let gaps = gaps(&topo, solid);
                let largest = gaps.values().copied().fold(0.0, f64::max);
                assert!(
                    (largest - worst).abs() < 1e-6,
                    "{name} {pose}: largest gap {largest:e}"
                );
                for v in solid_vertices(&topo, solid).unwrap() {
                    let tolerance = topo.vertex(v).unwrap().tolerance();
                    let gap = gaps.get(&v).copied().unwrap_or(0.0);
                    if gap > 1e-7 {
                        // The gap plus the linear tolerance, as read before
                        // the part was placed.
                        assert!(
                            tolerance >= gap && tolerance <= gap + 1e-7 + 1e-9,
                            "{name} {pose}: gap {gap:e}, tolerance {tolerance:e}"
                        );
                    } else {
                        assert!(
                            (tolerance - 1e-7).abs() < 1e-20,
                            "{name} {pose}: gap {gap:e}, tolerance {tolerance:e}"
                        );
                    }
                }
            }
        }
    }

    /// The largest gap of the ledge part, read off the file's numbers: the
    /// last control point of the B-spline `#767` and the vertex point `#13`
    /// that ends its edge `#768`.
    #[test]
    fn a_vertex_takes_the_gap_between_its_point_and_its_curve_s_end() {
        let mut topo = Topology::new();
        let solid = load(&mut topo, "leg_ledge_tips.step");
        let vertex = Point3::new(-1.737_416_331_625e2, -8.984_146_101_733e1, -7e1);
        let curve_end = Point3::new(-1.737_412_458_731e2, -8.984_244_966_744e1, -7e1);
        let gap = (vertex - curve_end).length();
        let v = solid_vertices(&topo, solid)
            .unwrap()
            .into_iter()
            .find(|&v| (topo.vertex(v).unwrap().point() - vertex).length() < 1e-9)
            .unwrap();
        let tolerance = topo.vertex(v).unwrap().tolerance();
        assert!(
            (tolerance - (gap + 1e-7)).abs() < 1e-12,
            "gap {gap:e}, tolerance {tolerance:e}"
        );
    }
}

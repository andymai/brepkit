//! Parts read from STEP files whose B-spline edges end off their vertices.
//!
//! A file gives each vertex as a point and each curve as its control
//! points, and has no tolerance for a vertex. In these three parts, written
//! by brepkit's STEP writer, curves end up to 1.06e-3 off their vertices.
//! Each such vertex takes the gap as its tolerance, in the millimetres the
//! part is placed in, so the part reads as a valid solid, and the vertices
//! with no gap keep the linear tolerance. A scale of the part scales the
//! tolerances with it.

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

    /// Each pose with the factor by which it scales a length.
    fn poses() -> Vec<(&'static str, Mat4, f64)> {
        let turn = Mat4::translation(30.0, -20.0, 50.0)
            * Mat4::rotation_x(0.4)
            * Mat4::rotation_y(1.1)
            * Mat4::rotation_z(0.3);
        vec![
            ("upright", Mat4::identity(), 1.0),
            ("turned", turn, 1.0),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0), 1.0),
            ("scaled", turn * Mat4::scale(25.4, 25.4, 25.4), 25.4),
        ]
    }

    /// That `solid` is valid, that its largest gap is `worst`, and that
    /// each vertex with a gap has a tolerance of the gap and at most
    /// `slack` more, and each other vertex `base`.
    fn assert_gaps_covered(
        topo: &Topology,
        solid: SolidId,
        worst: f64,
        slack: f64,
        base: f64,
        what: &str,
    ) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let gaps = gaps(topo, solid);
        let largest = gaps.values().copied().fold(0.0, f64::max);
        assert!(
            (largest - worst).abs() < 1e-3 * worst,
            "{what}: largest gap {largest:e}"
        );
        for v in solid_vertices(topo, solid).unwrap() {
            let tolerance = topo.vertex(v).unwrap().tolerance();
            let gap = gaps.get(&v).copied().unwrap_or(0.0);
            if gap > base {
                assert!(
                    tolerance >= gap && tolerance <= gap + slack,
                    "{what}: gap {gap:e}, tolerance {tolerance:e}"
                );
            } else {
                assert!(
                    (tolerance - base).abs() <= 1e-12 * base,
                    "{what}: gap {gap:e}, tolerance {tolerance:e}"
                );
            }
        }
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
            for (pose, at, scale) in poses() {
                let mut topo = Topology::new();
                let solid = load(&mut topo, name);
                transform_solid(&mut topo, solid, &at).unwrap();
                // The gap plus the linear tolerance, as read, scaled with
                // the part. A gap read where a vertex projects on to its
                // curve is found to 1e-9, so it moves a little with the
                // part.
                assert_gaps_covered(
                    &topo,
                    solid,
                    worst * scale,
                    (1e-7 + 1e-9) * scale,
                    1e-7 * scale,
                    &format!("{name} {pose}"),
                );
            }
        }
    }

    /// The ledge part written in metres and in centimetres: its gaps are a
    /// thousand and ten times as long in millimetres, and each vertex
    /// takes its gap in millimetres plus the linear tolerance.
    #[test]
    fn a_part_in_another_unit_takes_its_gaps_in_millimetres() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/leg_ledge_tips.step");
        let text = std::fs::read_to_string(path).unwrap();
        let unit = "SI_UNIT(.MILLI.,.METRE.)";
        assert_eq!(text.matches(unit).count(), 1);
        for (prefix, millimetres) in [("$", 1000.0), (".CENTI.", 10.0)] {
            let mut topo = Topology::new();
            let file = text.replace(unit, &format!("SI_UNIT({prefix},.METRE.)"));
            let solid = read_step(&file, &mut topo).unwrap()[0];
            assert_gaps_covered(
                &topo,
                solid,
                1.062e-3 * millimetres,
                1e-7 + 1e-9,
                1e-7,
                &format!("leg_ledge_tips.step in {prefix}"),
            );
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

//! Two halves of ellipses about the plate's (5, 5) from (8, 5) to (2, 5),
//! of semi-axes 3 along x and 1 and 2, with their chord and two lines from
//! the chord at x = 4 and 6 down to the plate's edge, imprinted on a plate's
//! top. The two arcs share both ends, so a step that tells edges apart by
//! their ends alone reads them as one. The top splits into the inner half
//! ellipse, the band between the two, the strip between the lines, and the
//! rest. Every case runs upright, turned and mirrored.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Ellipse3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::imprint::{ImprintTool, imprint_faces};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_line_edge;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;

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
            .map(|f| face_area(topo, f, 0.001).unwrap())
            .collect();
        areas.sort_by(f64::total_cmp);
        areas
    }

    #[test]
    fn two_half_ellipses_with_one_pair_of_ends_imprint_their_own_faces() {
        let half = |b: f64| 0.5 * PI * 3.0 * b;
        let mut want = [
            half(1.0),
            half(2.0) - half(1.0),
            10.0,
            100.0 - half(2.0) - 10.0,
        ];
        want.sort_by(f64::total_cmp);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
            transform_solid(&mut topo, plate, &pose).unwrap();
            let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 2.0));
            let (u, v) = (at(1.0, 0.0) - at(0.0, 0.0), at(0.0, 1.0) - at(0.0, 0.0));
            let n = u.cross(v);
            let (p, q) = (at(8.0, 5.0), at(2.0, 5.0));
            let vp = topo.add_vertex(Vertex::new(p, 1e-7));
            let vq = topo.add_vertex(Vertex::new(q, 1e-7));
            let mut tools = Vec::new();
            for b in [1.0, 2.0] {
                let ellipse = Ellipse3D::with_axes(at(5.0, 5.0), n, 3.0, b, u, v).unwrap();
                tools.push(ImprintTool::Edge(topo.add_edge(Edge::new(
                    vp,
                    vq,
                    EdgeCurve::Ellipse(ellipse),
                ))));
            }
            tools.push(ImprintTool::Edge(
                make_line_edge(&mut topo, q, p, 1e-7).unwrap(),
            ));
            for x in [4.0, 6.0] {
                tools.push(ImprintTool::Edge(
                    make_line_edge(&mut topo, at(x, 5.0), at(x, 0.0), 1e-7).unwrap(),
                ));
            }
            let result = imprint_faces(&mut topo, plate, &tools).unwrap();
            let areas = top_areas(&topo, result, pose);
            assert!(
                areas.len() == want.len()
                    && areas.iter().zip(&want).all(|(a, w)| (a - w).abs() < 1e-9),
                "{name}: top faces of area {areas:?}, want {want:?}"
            );
            let report = validate_solid(&topo, result).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            assert!(
                is_watertight(&tessellate_solid(&topo, result, 0.02).unwrap()),
                "{name}: open mesh"
            );
            let volume = solid_volume(&topo, result, 0.005).unwrap();
            assert!((volume - 200.0).abs() < 1e-9, "{name}: volume {volume}");
        }
    }
}

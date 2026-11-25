//! An arc of a sixth of a turn and the rest of its circle: two curves with
//! one pair of ends.
//!
//! Imprinted on a plate's top as two edges, they split it into the disc and
//! the rest. A rod whose wall is two faces, the arc's and the rest's, cuts,
//! fuses and meets a plate to the closed-form volumes. Every case runs
//! upright, turned and mirrored.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
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

    /// The result is valid and closed, and holds the volume `want`.
    fn check_solid(topo: &Topology, solid: SolidId, want: f64, label: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{label}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, solid, 0.02).unwrap();
        assert!(is_watertight(&mesh), "{label}: open mesh");
        let volume = solid_volume(topo, solid, 0.005).unwrap();
        assert!(
            (volume - want).abs() < 1e-9,
            "{label}: volume {volume}, want {want}"
        );
    }

    /// The plate, and a rod of radius 2 about (5, 5) from z = -1 to 3 with
    /// its seam at angle 0.2 and a line imprinted on its wall at 0.2 + pi /
    /// 3, posed: its wall is two faces, a sixth of a turn and the rest.
    fn seamed_scene(topo: &mut Topology, pose: Mat4) -> [SolidId; 2] {
        let plate = make_box(topo, 10.0, 10.0, 2.0).unwrap();
        transform_solid(topo, plate, &pose).unwrap();
        let rod = make_cylinder(topo, 2.0, 4.0).unwrap();
        transform_solid(
            topo,
            rod,
            &(Mat4::translation(5.0, 5.0, -1.0) * Mat4::rotation_z(0.2)),
        )
        .unwrap();
        let (sin, cos) = (0.2 + PI / 3.0).sin_cos();
        let (x, y) = (5.0 + 2.0 * cos, 5.0 + 2.0 * sin);
        let line =
            make_line_edge(topo, Point3::new(x, y, -1.0), Point3::new(x, y, 3.0), 1e-7).unwrap();
        let rod = imprint_faces(topo, rod, &[ImprintTool::Edge(line)]).unwrap();
        let walls: Vec<f64> = solid_faces(topo, rod)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .map(|f| face_area(topo, f, 0.005).unwrap())
            .collect();
        assert_eq!(walls.len(), 2, "rod walls of area {walls:?}");
        transform_solid(topo, rod, &pose).unwrap();
        [plate, rod]
    }

    /// An arc of a sixth of a turn and the rest of its circle, two edges
    /// with one pair of ends, imprinted on the plate's top: the top splits
    /// into the disc and the rest.
    #[test]
    fn an_arc_and_the_rest_of_its_circle_imprint_the_disc_on_a_plate() {
        let disc = 4.0 * PI;
        for back in [false, true] {
            for (name, pose) in poses() {
                let label = format!("{name}, rest about -z: {back}");
                let mut topo = Topology::new();
                let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
                transform_solid(&mut topo, plate, &pose).unwrap();
                let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 2.0));
                let (u, v) = (at(1.0, 0.0) - at(0.0, 0.0), at(0.0, 1.0) - at(0.0, 0.0));
                let normal = u.cross(v);
                let up = Circle3D::with_axes(at(5.0, 5.0), normal, 2.0, u, v).unwrap();
                let down = Circle3D::with_axes(at(5.0, 5.0), -normal, 2.0, u, -v).unwrap();
                let p = topo.add_vertex(Vertex::new(up.evaluate(0.2), 1e-7));
                let q = topo.add_vertex(Vertex::new(up.evaluate(0.2 + PI / 3.0), 1e-7));
                let arc = topo.add_edge(Edge::new(p, q, EdgeCurve::Circle(up.clone())));
                let rest = if back {
                    topo.add_edge(Edge::new(p, q, EdgeCurve::Circle(down)))
                } else {
                    topo.add_edge(Edge::new(q, p, EdgeCurve::Circle(up)))
                };
                let result = imprint_faces(
                    &mut topo,
                    plate,
                    &[ImprintTool::Edge(arc), ImprintTool::Edge(rest)],
                )
                .unwrap();
                let areas = top_areas(&topo, result, pose);
                let want = [disc, 100.0 - disc];
                assert!(
                    areas.len() == want.len()
                        && areas.iter().zip(&want).all(|(a, w)| (a - w).abs() < 1e-9),
                    "{label}: top faces of area {areas:?}, expected {want:?}"
                );
                check_solid(&topo, result, 200.0, &label);
            }
        }
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

    /// The plate less the seamed rod, the two fused, and their common part.
    /// The rest's rim on each plate face runs between two corners of the
    /// rest's wall, so it gets a vertex at its middle, and no two sections
    /// there share their ends.
    #[test]
    fn a_rod_whose_wall_is_an_arc_and_the_rest_of_its_circle_cuts_fuses_and_meets_a_plate() {
        let disc = 4.0 * PI;
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            for (op, want) in [
                (BooleanOp::Cut, 200.0 - 2.0 * disc),
                (BooleanOp::Fuse, 200.0 + 2.0 * disc),
                (BooleanOp::Intersect, 2.0 * disc),
            ] {
                let label = format!("{op:?}, {name}");
                let mut topo = Topology::new();
                let [plate, rod] = seamed_scene(&mut topo, pose);
                let result =
                    boolean_with_options(&mut topo, op, plate, rod, options).expect(&label);
                check_solid(&topo, result, want, &label);
            }
        }
    }
}

//! A whole circle on a plate's top that another cut meets.
//!
//! A circle edge about (5, 4) of radius root 2 imprinted on the top of a
//! 10 x 10 x 2 plate with a line across the top through (4, 5) and (6, 5),
//! or with the chord between those points, splits the top into its true
//! regions. A plate fused with a rod through it and a box whose side meets
//! the rod's rim at those points keeps its closed-form volume. Every case
//! runs upright, turned and mirrored.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI, SQRT_2};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::compound_ops::fuse_all;
    use brepkit_operations::imprint::{ImprintTool, imprint_faces};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_line_edge;
    use brepkit_topology::compound::Compound;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;

    /// The area of the segment the line cuts off the circle's disc.
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

    /// The circle edge and a line from `from` to `to` imprinted on the top
    /// split it into faces of the areas `want`, and the plate stays valid,
    /// closed and of its volume.
    #[test]
    fn a_circle_and_a_line_through_two_of_its_points_imprint_their_regions() {
        let disc = 2.0 * PI;
        let cases = [
            (
                (0.0, 5.0),
                (10.0, 5.0),
                vec![
                    SEGMENT,
                    disc - SEGMENT,
                    50.0 - disc + SEGMENT,
                    50.0 - SEGMENT,
                ],
            ),
            (
                (4.0, 5.0),
                (6.0, 5.0),
                vec![SEGMENT, disc - SEGMENT, 100.0 - disc],
            ),
        ];
        for (from, to, mut want) in cases {
            want.sort_by(f64::total_cmp);
            for (name, pose) in poses() {
                let label = format!("line {from:?} to {to:?}, {name}");
                let mut topo = Topology::new();
                let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
                transform_solid(&mut topo, plate, &pose).unwrap();
                let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 2.0));
                let (u, v) = (at(1.0, 0.0) - at(0.0, 0.0), at(0.0, 1.0) - at(0.0, 0.0));
                let circle = Circle3D::with_axes(at(5.0, 4.0), u.cross(v), SQRT_2, u, v).unwrap();
                let seam = topo.add_vertex(Vertex::new(circle.evaluate(-FRAC_PI_2), 1e-7));
                let ring = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
                let line =
                    make_line_edge(&mut topo, at(from.0, from.1), at(to.0, to.1), 1e-7).unwrap();
                let result = imprint_faces(
                    &mut topo,
                    plate,
                    &[ImprintTool::Edge(ring), ImprintTool::Edge(line)],
                )
                .unwrap();
                let areas = top_areas(&topo, result, pose);
                assert!(
                    areas.len() == want.len()
                        && areas.iter().zip(&want).all(|(a, w)| (a - w).abs() < 1e-9),
                    "{label}: top faces of area {areas:?}, expected {want:?}"
                );
                check_solid(&topo, result, 200.0, &label);
            }
        }
    }

    /// The plate, a rod of radius root 2 about (5, 4) from z = -1 to 3, and
    /// a box over x 0 to 10, y 5 to 7, z 1 to 3, whose side y = 5 meets the
    /// rod's rim on the plate's top at (4, 5) and (6, 5), posed.
    fn scene(topo: &mut Topology, pose: Mat4) -> [SolidId; 3] {
        let plate = make_box(topo, 10.0, 10.0, 2.0).unwrap();
        transform_solid(topo, plate, &pose).unwrap();
        let rod = make_cylinder(topo, SQRT_2, 4.0).unwrap();
        transform_solid(topo, rod, &(pose * Mat4::translation(5.0, 4.0, -1.0))).unwrap();
        let slot = make_box(topo, 10.0, 2.0, 2.0).unwrap();
        transform_solid(topo, slot, &(pose * Mat4::translation(0.0, 5.0, 1.0))).unwrap();
        [plate, rod, slot]
    }

    /// The plate, the rod and the box fused in one pass, as a compound, and
    /// one at a time: the plate, the rod's 4 pi outside it, and the box's 20
    /// over the plate less the segment of the rod's disc it shares, 1 high.
    #[test]
    fn a_plate_a_rod_through_it_and_a_box_on_its_rim_fuse_to_their_volume() {
        let want = 220.0 + 4.0 * PI - SEGMENT;
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solids = scene(&mut topo, pose);
            let fused = brepkit_algo::gfa::fuse_n(&mut topo, &solids).unwrap();
            check_solid(&topo, fused, want, &format!("{name}, one pass"));

            let mut topo = Topology::new();
            let solids = scene(&mut topo, pose);
            let compound = topo.add_compound(Compound::new(solids.to_vec()));
            let fused = fuse_all(&mut topo, compound).unwrap();
            check_solid(&topo, fused, want, &format!("{name}, compound"));

            let mut topo = Topology::new();
            let [plate, rod, slot] = scene(&mut topo, pose);
            let fused = boolean_with_options(&mut topo, BooleanOp::Fuse, plate, rod, options)
                .and_then(|a| boolean_with_options(&mut topo, BooleanOp::Fuse, a, slot, options))
                .unwrap();
            check_solid(&topo, fused, want, &format!("{name}, one at a time"));
        }
    }
}

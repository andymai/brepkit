//! A tool sitting flush in the corner of a plate, with a rounded side: its end
//! face's plane crosses the plate's floor, and the floor keeps a hole wherever
//! that hole is. The section of that end face on the floor stops where the tool
//! does, and the floor's split pieces keep the holes no section touches.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::FRAC_PI_4;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const RADIUS: f64 = 0.5;
    const TOOL_START: f64 = 8.0;
    const TOOL_LENGTH: f64 = 32.0;

    /// The corner section at `x = 8` swept from `y = 8` to the plate's far side:
    /// flat on the floor and the wall, its third side an arc tangent to both.
    fn rounded_corner_tool(topo: &mut Topology) -> SolidId {
        let r = RADIUS;
        let at = |x: f64, z: f64| Point3::new(x, TOOL_START, z);
        let center = at(8.0 - r, r);
        let (a, corner, b) = (at(8.0, r), at(8.0, 0.0), at(8.0 - r, 0.0));
        let va = topo.add_vertex(Vertex::new(a, 1e-7));
        let vc = topo.add_vertex(Vertex::new(corner, 1e-7));
        let vb = topo.add_vertex(Vertex::new(b, 1e-7));
        let normal = (b - center).cross(a - center).normalize().unwrap();
        let circle = Circle3D::new(center, normal, r).unwrap();
        let edges = [
            (va, vc, EdgeCurve::Line),
            (vc, vb, EdgeCurve::Line),
            (vb, va, EdgeCurve::Circle(circle)),
        ]
        .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true));
        let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
        let along = Vec3::new(0.0, 1.0, 0.0);
        let face = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: -along,
                d: -TOOL_START,
            },
        ));
        extrude(topo, face, along, TOOL_LENGTH).unwrap()
    }

    /// An 8 x 40 x 20 plate with a hole of radius 0.25 up from its floor, `deep`
    /// into it. Deep enough to pass the top, it is a through hole.
    fn drilled_plate(topo: &mut Topology, x: f64, y: f64, deep: f64) -> SolidId {
        let plate = make_box(topo, 8.0, 40.0, 20.0).unwrap();
        let pin = make_cylinder(topo, 0.25, deep + 1.0).unwrap();
        transform_solid(topo, pin, &Mat4::translation(x, y, -1.0)).unwrap();
        boolean(topo, BooleanOp::Cut, plate, pin).unwrap()
    }

    fn removed() -> f64 {
        (1.0 - FRAC_PI_4) * RADIUS * RADIUS * TOOL_LENGTH
    }

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - volume).abs() < 1e-6,
            "{what}: volume {exact}, expected {volume}"
        );
        let meshed = oriented_solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (meshed - volume).abs() < 1e-3 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    /// Blind and through, in each part of the floor the end face's plane and the
    /// tool's rim divide it into, and on that plane.
    #[test]
    fn a_rounded_tool_leaves_a_hole_anywhere_on_the_floor() {
        for (name, x, y) in [
            ("beyond the tool's side", 2.0, 20.0),
            ("short of the tool's end", 2.0, 4.0),
            ("on the end face's plane", 2.0, 8.0),
            ("before the tool, in line with its strip", 7.6, 4.0),
            ("just beside the tool", 6.9, 20.0),
        ] {
            for (kind, deep) in [("blind", 4.0), ("through", 29.0)] {
                let what = format!("{kind} hole {name}");
                let mut topo = Topology::new();
                let plate = drilled_plate(&mut topo, x, y, deep);
                let before = solid_volume(&topo, plate, 0.001).unwrap();
                let tool = rounded_corner_tool(&mut topo);
                let result = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
                assert_sound(&topo, result, before - removed(), &what);

                let curved = solid_faces(&topo, result)
                    .unwrap()
                    .iter()
                    .filter(|&&f| {
                        !matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. })
                    })
                    .count();
                assert_eq!(curved, 2, "{what}: fell back to a mesh, or lost a wall");
                let in_hole = Point3::new(x, y, 1.0);
                assert_eq!(
                    classify_point(&topo, result, in_hole, 0.01, 1e-7).unwrap(),
                    PointClassification::Outside,
                    "{what}: the hole is filled"
                );
            }
        }
    }

    /// The same cut with everything turned off the axes, or mirrored.
    #[test]
    fn the_hole_stays_in_any_pose() {
        for (name, pose) in [
            (
                "turned",
                Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_x(0.6) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ] {
            for (x, y) in [(2.0, 20.0), (2.0, 8.0)] {
                let what = format!("{name} hole at ({x}, {y})");
                let mut topo = Topology::new();
                let plate = drilled_plate(&mut topo, x, y, 4.0);
                let tool = rounded_corner_tool(&mut topo);
                let before = solid_volume(&topo, plate, 0.001).unwrap();
                transform_solid(&mut topo, plate, &pose).unwrap();
                transform_solid(&mut topo, tool, &pose).unwrap();
                let result = boolean(&mut topo, BooleanOp::Cut, plate, tool).unwrap();
                assert_sound(&topo, result, before - removed(), &what);
                let in_hole = pose.mul_point(Point3::new(x, y, 1.0));
                assert_eq!(
                    classify_point(&topo, result, in_hole, 0.01, 1e-7).unwrap(),
                    PointClassification::Outside,
                    "{what}: the hole is filled"
                );
            }
        }
    }
}

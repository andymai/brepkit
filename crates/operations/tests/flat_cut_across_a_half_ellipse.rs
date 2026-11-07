//! A prism on a half ellipse, one flat wall and one curved. A plane across it
//! meets the flat wall in a chord and the curved wall in an arc, and the two
//! share both ends, so nothing but their middles tells them apart.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Ellipse3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::primitives::make_box;
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

    const AREA: f64 = PI * 5.0 * 3.0 / 2.0;
    const HEIGHT: f64 = 10.0;

    /// A half ellipse of semi-axes 5 and 3 in `z = 0`, up by `HEIGHT`. Its flat
    /// wall lies in `x = 0`, its curved wall towards negative `x`.
    fn half_ellipse_prism(topo: &mut Topology) -> SolidId {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let ellipse = Ellipse3D::new(Point3::new(0.0, 0.0, 0.0), up, 5.0, 3.0).unwrap();
        let (p, q) = (ellipse.evaluate(0.0), ellipse.evaluate(PI));
        let a = topo.add_vertex(Vertex::new(p, 1e-7));
        let b = topo.add_vertex(Vertex::new(q, 1e-7));
        let arc = topo.add_edge(Edge::new(a, b, EdgeCurve::Ellipse(ellipse)));
        let chord = topo.add_edge(Edge::new(b, a, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(arc, true), OrientedEdge::new(chord, true)],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane { normal: up, d: 0.0 },
        ));
        extrude(topo, face, up, HEIGHT).unwrap()
    }

    /// A wide slab from `z0` up by `height`.
    fn slab(topo: &mut Topology, z0: f64, height: f64) -> SolidId {
        let s = make_box(topo, 60.0, 60.0, height).unwrap();
        transform_solid(topo, s, &Mat4::translation(-30.0, -30.0, z0)).unwrap();
        s
    }

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, faces: usize, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - volume).abs() < 1e-6 * volume,
            "{what}: volume {exact}, expected {volume}"
        );
        let count = solid_faces(topo, solid).unwrap().len();
        assert_eq!(count, faces, "{what}: faces");
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let meshed = oriented_solid_volume(topo, solid, 0.005).unwrap();
        assert!(
            (meshed - volume).abs() < 2e-3 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    #[test]
    fn a_plane_across_it_keeps_the_arc_and_the_chord_apart() {
        // Off its base it keeps its top, its two walls and a new cap.
        let mut topo = Topology::new();
        let prism = half_ellipse_prism(&mut topo);
        let tool = slab(&mut topo, -5.0, 8.0);
        let top = boolean(&mut topo, BooleanOp::Cut, prism, tool).unwrap();
        assert_sound(&topo, top, AREA * 7.0, 4, "off its base");

        // A slab through it leaves two pieces of the prism's section.
        let mut topo = Topology::new();
        let prism = half_ellipse_prism(&mut topo);
        let tool = slab(&mut topo, 3.0, 2.0);
        let apart = boolean(&mut topo, BooleanOp::Cut, prism, tool).unwrap();
        assert_sound(&topo, apart, AREA * 8.0, 8, "a slab through it");
        for (z, expected) in [
            (2.0, PointClassification::Inside),
            (4.0, PointClassification::Outside),
            (4.9, PointClassification::Outside),
            (6.0, PointClassification::Inside),
        ] {
            // Near the ellipse's rim, where the chord's face would miss it.
            let read = classify_point(&topo, apart, Point3::new(-2.0, 1.0, z), 0.01, 1e-7).unwrap();
            assert_eq!(read, expected, "at height {z}");
        }

        // What the slab takes is the same section, closed by its two caps.
        let mut topo = Topology::new();
        let prism = half_ellipse_prism(&mut topo);
        let tool = slab(&mut topo, 3.0, 2.0);
        let common = boolean(&mut topo, BooleanOp::Intersect, prism, tool).unwrap();
        assert_sound(&topo, common, AREA * 2.0, 4, "the slab's share");
    }

    /// The same cut with everything turned off the axes, or mirrored.
    #[test]
    fn it_holds_in_any_pose() {
        for (name, pose) in [
            (
                "turned",
                Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_x(0.6) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ] {
            let mut topo = Topology::new();
            let prism = half_ellipse_prism(&mut topo);
            let tool = slab(&mut topo, -5.0, 8.0);
            transform_solid(&mut topo, prism, &pose).unwrap();
            transform_solid(&mut topo, tool, &pose).unwrap();
            let top = boolean(&mut topo, BooleanOp::Cut, prism, tool).unwrap();
            assert_sound(&topo, top, AREA * 7.0, 4, name);
        }
    }

    /// A plane not square to the walls meets the curved one in an ellipse arc of
    /// its own. The piece below it is the prism's section times its mean height.
    #[test]
    fn a_tilted_plane_keeps_them_apart_too() {
        let mut topo = Topology::new();
        let prism = half_ellipse_prism(&mut topo);
        let tool = slab(&mut topo, 5.0, 30.0);
        // Tilted about the flat wall's own direction, through (0, 0, 5).
        let tilt = Mat4::translation(0.0, 0.0, 5.0)
            * Mat4::rotation_y(0.2)
            * Mat4::translation(0.0, 0.0, -5.0);
        transform_solid(&mut topo, tool, &tilt).unwrap();
        let below = boolean(&mut topo, BooleanOp::Cut, prism, tool).unwrap();
        // The plane is z = 5 - x tan 0.2; x runs to -3, centroid x = -4 * 3 / (3 pi).
        let centroid = -4.0 * 3.0 / (3.0 * PI);
        let truth = AREA * (5.0 - centroid * 0.2_f64.tan());
        assert_sound(&topo, below, truth, 4, "tilted");
    }
}

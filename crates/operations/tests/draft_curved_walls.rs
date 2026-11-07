//! Drafting a cylinder wall that runs along the pull turns it into the cone
//! through its circle on the neutral plane, and the planes tangent to it turn
//! with it, so the two stay tangent and the solid keeps its topology.
//!
//! Each horizontal section of a drafted obround is the obround offset by the
//! radius change, so its volume integrates in closed form: `60 r + pi r^2` over
//! a half-length of 30, with `r` running linearly in height.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::draft::draft;
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::{solid_faces, solid_vertices};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const RADIUS: f64 = 10.0;

    /// An obround prism: a 30-long straight run at `y = 30 +- 10` between half
    /// circles about `(25, 30)` and `(55, 30)`, from `z0` up by `height`.
    fn obround_prism(topo: &mut Topology, z0: f64, height: f64) -> SolidId {
        let at = |t: &mut Topology, x, y| t.add_vertex(Vertex::new(Point3::new(x, y, z0), 1e-7));
        let a = at(topo, 25.0, 20.0);
        let b = at(topo, 55.0, 20.0);
        let c = at(topo, 55.0, 40.0);
        let d = at(topo, 25.0, 40.0);
        let circle = |x: f64| {
            let center = Point3::new(x, 30.0, z0);
            EdgeCurve::Circle(Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), RADIUS).unwrap())
        };
        let edges: Vec<_> = [
            (a, b, EdgeCurve::Line),
            (b, c, circle(55.0)),
            (c, d, EdgeCurve::Line),
            (d, a, circle(25.0)),
        ]
        .into_iter()
        .map(|(s, e, curve)| OrientedEdge::new(topo.add_edge(Edge::new(s, e, curve)), true))
        .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: z0,
        };
        let face = topo.add_face(Face::new(wire, Vec::new(), plane));
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap()
    }

    /// The integral over height of an obround's section area, the radius
    /// running from `r0` to `r1` over `height`.
    fn obround_volume(r0: f64, r1: f64, height: f64) -> f64 {
        let antiderivative = |r: f64| (30.0 * r).mul_add(r, PI * r * r * r / 3.0);
        if (r1 - r0).abs() < 1e-12 {
            return (60.0 * r0 + PI * r0 * r0) * height;
        }
        (antiderivative(r1) - antiderivative(r0)) * height / (r1 - r0)
    }

    /// The faces of `solid` standing along `pull`, all of whose vertices lie
    /// at or above `from` along it.
    fn walls_above(topo: &Topology, solid: SolidId, pull: Vec3, from: f64) -> Vec<FaceId> {
        solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&f| {
                let face = topo.face(f).unwrap();
                let upright = match face.surface() {
                    FaceSurface::Plane { normal, .. } => normal.dot(pull).abs() < 1e-6,
                    FaceSurface::Cylinder(_) => true,
                    _ => false,
                };
                let wire = topo.wire(face.outer_wire()).unwrap();
                upright
                    && wire.edges().iter().all(|oe| {
                        let edge = topo.edge(oe.edge()).unwrap();
                        [edge.start(), edge.end()].iter().all(|&v| {
                            let p = topo.vertex(v).unwrap().point();
                            pull.dot(p - Point3::new(0.0, 0.0, 0.0)) >= from - 1e-6
                        })
                    })
            })
            .collect()
    }

    fn assert_sound(topo: &Topology, solid: SolidId, truth: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let oriented = oriented_solid_volume(topo, solid, 0.001).unwrap();
        assert!(oriented > 0.0, "{what}: inverted, {oriented}");
        let volume = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (volume - truth).abs() < 1e-6 * truth,
            "{what}: volume {volume}, expected {truth}"
        );
    }

    fn poses() -> Vec<(&'static str, Mat4, bool)> {
        let turn =
            Mat4::translation(3.0, -7.0, 11.0) * Mat4::rotation_x(0.6) * Mat4::rotation_z(0.4);
        vec![
            ("upright", Mat4::identity(), false),
            ("turned", turn, false),
            ("mirrored", Mat4::identity(), true),
        ]
    }

    /// A direction and a point carried through a pose.
    fn posed(pose: &Mat4, mirrored: bool, p: Point3, v: Vec3) -> (Point3, Vec3) {
        if mirrored {
            return (
                Point3::new(-p.x(), p.y(), p.z()),
                Vec3::new(-v.x(), v.y(), v.z()),
            );
        }
        let moved = pose.mul_point(p);
        let tip = pose.mul_point(p + v);
        (moved, tip - moved)
    }

    fn place(topo: &mut Topology, solid: SolidId, pose: &Mat4, mirrored: bool) -> SolidId {
        if mirrored {
            return mirror(
                topo,
                solid,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
            )
            .unwrap();
        }
        transform_solid(topo, solid, pose).unwrap();
        solid
    }

    /// The issue's pocket: 10 deep in the top of an 80 x 60 x 25 block, its
    /// walls drafted about the rim.
    #[test]
    fn an_obround_pockets_walls_lean_together() {
        let angle = 5.0_f64.to_radians();
        for (name, pose, mirrored) in poses() {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 80.0, 60.0, 25.0).unwrap();
            let tool = obround_prism(&mut topo, 15.0, 10.0);
            let pocket = boolean(&mut topo, BooleanOp::Cut, block, tool).unwrap();
            let pocket = place(&mut topo, pocket, &pose, mirrored);
            let (rim, pull) = posed(
                &pose,
                mirrored,
                Point3::new(0.0, 0.0, 25.0),
                Vec3::new(0.0, 0.0, 1.0),
            );
            let (floor, _) = posed(&pose, mirrored, Point3::new(0.0, 0.0, 15.0), pull);
            let from = pull.dot(floor - Point3::new(0.0, 0.0, 0.0));
            let walls = walls_above(&topo, pocket, pull, from);
            assert_eq!(walls.len(), 4, "{name}");
            let before = solid_faces(&topo, pocket).unwrap().len();

            let drafted = draft(&mut topo, pocket, &walls, pull, rim, angle).unwrap();

            assert_eq!(solid_faces(&topo, drafted).unwrap().len(), before, "{name}");
            let cones = solid_faces(&topo, drafted)
                .unwrap()
                .iter()
                .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
                .count();
            assert_eq!(cones, 2, "{name}");
            // Toward the floor the pocket narrows, so the part releases upward.
            let floor_radius = 10.0_f64.mul_add(-angle.tan(), RADIUS);
            let truth = 80.0 * 60.0 * 25.0 - obround_volume(floor_radius, RADIUS, 10.0);
            assert_sound(&topo, drafted, truth, name);
        }
    }

    /// An obround boss standing on nothing: a prism whose wall leans in above
    /// the neutral plane, or out with the angle turned round.
    #[test]
    fn an_obround_prisms_walls_lean_together() {
        for (name, angle) in [("in", 5.0_f64), ("out", -5.0_f64)] {
            let angle = angle.to_radians();
            let mut topo = Topology::new();
            let prism = obround_prism(&mut topo, 0.0, 12.0);
            let pull = Vec3::new(0.0, 0.0, 1.0);
            let walls = walls_above(&topo, prism, pull, 0.0);
            assert_eq!(walls.len(), 4, "{name}");
            let neutral = Point3::new(0.0, 0.0, 4.0);
            let drafted = draft(&mut topo, prism, &walls, pull, neutral, angle).unwrap();
            let t = angle.tan();
            let truth = obround_volume(
                4.0_f64.mul_add(t, RADIUS),
                8.0_f64.mul_add(-t, RADIUS),
                12.0,
            );
            assert_sound(&topo, drafted, truth, name);
        }
    }

    /// A whole cylinder's wall, its seam and closed rims, turned into a frustum.
    #[test]
    fn a_whole_cylinder_turns_into_a_frustum() {
        let angle = 3.0_f64.to_radians();
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let walls = walls_above(&topo, rod, pull, 0.0);
        assert_eq!(walls.len(), 1);
        let drafted = draft(
            &mut topo,
            rod,
            &walls,
            pull,
            Point3::new(0.0, 0.0, 0.0),
            angle,
        )
        .unwrap();
        let top = 20.0_f64.mul_add(-angle.tan(), 5.0);
        let truth = PI * 20.0 / 3.0 * (5.0_f64.mul_add(top, 25.0) + top * top);
        assert_sound(&topo, drafted, truth, "rod");
    }

    /// A round through hole in a plate: the bore narrows toward the bottom, its
    /// plate faces left where they stand.
    #[test]
    fn a_bore_turns_into_a_tapered_hole() {
        let angle = 4.0_f64.to_radians();
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 30.0, 30.0, 10.0).unwrap();
        let drill = make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        transform_solid(&mut topo, drill, &Mat4::translation(15.0, 15.0, -5.0)).unwrap();
        let drilled = boolean(&mut topo, BooleanOp::Cut, plate, drill).unwrap();
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let walls: Vec<FaceId> = solid_faces(&topo, drilled)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .collect();
        assert_eq!(walls.len(), 1);
        let neutral = Point3::new(0.0, 0.0, 10.0);
        let drafted = draft(&mut topo, drilled, &walls, pull, neutral, angle).unwrap();
        let low = 10.0_f64.mul_add(-angle.tan(), 5.0);
        let frustum = PI * 10.0 / 3.0 * (5.0_f64.mul_add(low, 25.0) + low * low);
        assert_sound(&topo, drafted, 30.0 * 30.0 * 10.0 - frustum, "bore");
    }

    /// A wall needs its neighbours to lean with it: left standing, a plane
    /// tangent to it no longer meets its ruling.
    #[test]
    fn a_tangent_neighbour_left_standing_is_refused() {
        let mut topo = Topology::new();
        let prism = obround_prism(&mut topo, 0.0, 12.0);
        let curved: Vec<FaceId> = solid_faces(&topo, prism)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .collect();
        assert_eq!(curved.len(), 2);
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let result = draft(
            &mut topo,
            prism,
            &curved,
            pull,
            Point3::new(0.0, 0.0, 0.0),
            0.05,
        );
        assert!(result.is_err());
    }

    /// A plane meeting a wall at a sharp edge leans into the cone along a
    /// hyperbola, not along a ruling.
    #[test]
    fn a_sharp_neighbour_is_refused() {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 10.0).unwrap();
        let cutter = make_box(&mut topo, 20.0, 20.0, 30.0).unwrap();
        transform_solid(&mut topo, cutter, &Mat4::translation(-10.0, 2.0, -10.0)).unwrap();
        let flat = boolean(&mut topo, BooleanOp::Cut, rod, cutter).unwrap();
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let walls = walls_above(&topo, flat, pull, 0.0);
        assert!(walls.len() >= 2);
        let result = draft(
            &mut topo,
            flat,
            &walls,
            pull,
            Point3::new(0.0, 0.0, 0.0),
            0.05,
        );
        assert!(result.is_err());
    }

    /// A wall whose radius runs out before its far rim would pass through the
    /// cone's apex, and a cylinder lying across the pull has no such cone.
    #[test]
    fn walls_it_cannot_lean_are_refused() {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 1.0, 10.0).unwrap();
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let walls = walls_above(&topo, rod, pull, 0.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        assert!(draft(&mut topo, rod, &walls, pull, origin, 0.5).is_err());
        assert!(
            draft(
                &mut topo,
                rod,
                &walls,
                Vec3::new(1.0, 0.0, 0.0),
                origin,
                0.1
            )
            .is_err()
        );
    }

    /// The source stays as it was.
    #[test]
    fn the_source_is_left_alone() {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        let before: Vec<Point3> = solid_vertices(&topo, rod)
            .unwrap()
            .iter()
            .map(|&v| topo.vertex(v).unwrap().point())
            .collect();
        let pull = Vec3::new(0.0, 0.0, 1.0);
        let walls = walls_above(&topo, rod, pull, 0.0);
        draft(
            &mut topo,
            rod,
            &walls,
            pull,
            Point3::new(0.0, 0.0, 0.0),
            0.05,
        )
        .unwrap();
        let after: Vec<Point3> = solid_vertices(&topo, rod)
            .unwrap()
            .iter()
            .map(|&v| topo.vertex(v).unwrap().point())
            .collect();
        assert_eq!(before.len(), after.len());
        for (a, b) in before.iter().zip(&after) {
            assert!((*a - *b).length() < 1e-12);
        }
        assert!(matches!(
            topo.face(walls[0]).unwrap().surface(),
            FaceSurface::Cylinder(_)
        ));
    }
}

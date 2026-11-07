//! A round along a straight edge that ends at a concave corner of the solid.
//! At that end the material goes on past the third face's plane under the
//! round, which stops against a flat tab in that plane, so the third face grows
//! where a convex end's would be notched. What comes off is the corner's section
//! swept along the edge, `(tan(t/2) - t/2) r^2` per unit length for a turn of `t`
//! between the two faces' normals.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::FRAC_PI_4;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::fillet::concave_end::fillet_concave_ends;
    use brepkit_operations::measure::{oriented_solid_volume, solid_volume};
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::make_cylinder;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A prism on `outline` in `z = 0`, up by `height`.
    fn prism(topo: &mut Topology, outline: &[(f64, f64)], height: f64) -> SolidId {
        let vertices: Vec<_> = outline
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
            .collect();
        let edges: Vec<_> = (0..vertices.len())
            .map(|i| {
                let next = vertices[(i + 1) % vertices.len()];
                OrientedEdge::new(
                    topo.add_edge(Edge::new(vertices[i], next, EdgeCurve::Line)),
                    true,
                )
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        extrude(topo, face, Vec3::new(0.0, 0.0, 1.0), height).unwrap()
    }

    /// The straight edge from `a` to `b`, either way round.
    fn edge_between(topo: &Topology, solid: SolidId, a: Point3, b: Point3) -> EdgeId {
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let (p, q) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                let near = |u: Point3, v: Point3| (u - v).length() < 1e-9;
                (near(p, a) && near(q, b)) || (near(p, b) && near(q, a))
            })
            .unwrap()
    }

    /// A square corner's section.
    fn square_section(r: f64) -> f64 {
        (1.0 - FRAC_PI_4) * r * r
    }

    fn assert_sound(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - volume).abs() < 1e-6 * volume,
            "{what}: volume {exact}, expected {volume}"
        );
        let meshed = oriented_solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (meshed - volume).abs() < 1e-3 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    const PEN_L: [(f64, f64); 6] = [
        (0.0, 0.0),
        (60.0, 0.0),
        (60.0, 8.0),
        (8.0, 8.0),
        (8.0, 40.0),
        (0.0, 40.0),
    ];
    const PEN_L_VOLUME: f64 = (60.0 * 8.0 + 8.0 * 32.0) * 20.0;

    /// The issue's edge: `(8, 8, 0)` to `(8, 40, 0)`, in any pose.
    #[test]
    fn an_edge_from_a_concave_corner_stops_on_its_plane() {
        for (name, r) in [("upright", 0.5), ("turned", 2.0), ("mirrored", 1.0)] {
            let mut topo = Topology::new();
            let solid = prism(&mut topo, &PEN_L, 20.0);
            let carry = |p: Point3| -> Point3 {
                match name {
                    "turned" => {
                        (Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_x(0.6)).mul_point(p)
                    }
                    "mirrored" => Point3::new(-p.x(), p.y(), p.z()),
                    _ => p,
                }
            };
            let solid = match name {
                "turned" => {
                    let m = Mat4::translation(3.0, -1.0, 2.0) * Mat4::rotation_x(0.6);
                    transform_solid(&mut topo, solid, &m).unwrap();
                    solid
                }
                "mirrored" => mirror(
                    &mut topo,
                    solid,
                    Point3::new(0.0, 0.0, 0.0),
                    Vec3::new(1.0, 0.0, 0.0),
                )
                .unwrap(),
                _ => solid,
            };
            let edge = edge_between(
                &topo,
                solid,
                carry(Point3::new(8.0, 8.0, 0.0)),
                carry(Point3::new(8.0, 40.0, 0.0)),
            );
            let result = fillet_concave_ends(&mut topo, solid, &[edge], r)
                .unwrap()
                .unwrap();
            let truth = PEN_L_VOLUME - square_section(r) * 32.0;
            assert_sound(&topo, result, truth, name);

            // The corner past the arc is gone, along the edge and right up to
            // the third plane; under the tab, and under the arc, the part stays.
            let t = r * 0.1;
            for (p, expected) in [
                (Point3::new(8.0 - t, 20.0, t), PointClassification::Outside),
                (
                    Point3::new(8.0 - t, 8.0 + t, t),
                    PointClassification::Outside,
                ),
                (
                    Point3::new(8.0 - t, 8.0 - t, t),
                    PointClassification::Inside,
                ),
                (Point3::new(8.0 - r, 20.0, r), PointClassification::Inside),
            ] {
                let read = classify_point(&topo, result, carry(p), 0.01, 1e-7).unwrap();
                assert_eq!(read, expected, "{name}: {p:?}");
            }
        }
    }

    /// A U's inner floor runs between two concave corners, and its arms keep
    /// their walls.
    #[test]
    fn an_edge_between_two_concave_corners_stops_on_both() {
        let outline = [
            (0.0, 0.0),
            (20.0, 0.0),
            (20.0, 20.0),
            (15.0, 20.0),
            (15.0, 5.0),
            (5.0, 5.0),
            (5.0, 20.0),
            (0.0, 20.0),
        ];
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &outline, 10.0);
        let edge = edge_between(
            &topo,
            solid,
            Point3::new(5.0, 5.0, 10.0),
            Point3::new(15.0, 5.0, 10.0),
        );
        let result = fillet_concave_ends(&mut topo, solid, &[edge], 1.5)
            .unwrap()
            .unwrap();
        let volume = (20.0 * 5.0 + 2.0 * 5.0 * 15.0) * 10.0 - square_section(1.5) * 10.0;
        assert_sound(&topo, result, volume, "U floor");
    }

    /// Two edges apart take their cuts in turn.
    #[test]
    fn both_rims_of_a_wall_take_their_cuts() {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &PEN_L, 20.0);
        let edges = [
            edge_between(
                &topo,
                solid,
                Point3::new(8.0, 8.0, 0.0),
                Point3::new(8.0, 40.0, 0.0),
            ),
            edge_between(
                &topo,
                solid,
                Point3::new(8.0, 8.0, 20.0),
                Point3::new(8.0, 40.0, 20.0),
            ),
        ];
        let result = fillet_concave_ends(&mut topo, solid, &edges, 1.0)
            .unwrap()
            .unwrap();
        let volume = PEN_L_VOLUME - 2.0 * square_section(1.0) * 32.0;
        assert_sound(&topo, result, volume, "both rims");
    }

    /// The long leg's edge ends on the same corner from the other side.
    #[test]
    fn the_long_legs_edge_stops_on_the_short_legs_plane() {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &PEN_L, 20.0);
        let edge = edge_between(
            &topo,
            solid,
            Point3::new(60.0, 8.0, 0.0),
            Point3::new(8.0, 8.0, 0.0),
        );
        let result = fillet_concave_ends(&mut topo, solid, &[edge], 1.0)
            .unwrap()
            .unwrap();
        let volume = PEN_L_VOLUME - square_section(1.0) * 52.0;
        assert_sound(&topo, result, volume, "the long leg's edge");
    }

    /// An L swept up at a slant: the short leg's wall leans over the floor, so
    /// the edge between them turns other than a right angle, and the section
    /// is the general kite less its sector.
    #[test]
    fn a_leaning_wall_takes_its_own_section() {
        let outline: Vec<(f64, f64)> = PEN_L.to_vec();
        let mut topo = Topology::new();
        let vertices: Vec<_> = outline
            .iter()
            .map(|&(x, y)| topo.add_vertex(Vertex::new(Point3::new(x, y, 0.0), 1e-7)))
            .collect();
        let edges: Vec<_> = (0..vertices.len())
            .map(|i| {
                let next = vertices[(i + 1) % vertices.len()];
                OrientedEdge::new(
                    topo.add_edge(Edge::new(vertices[i], next, EdgeCurve::Line)),
                    true,
                )
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let lean = Vec3::new(0.3, 0.0, 1.0);
        let solid = extrude(&mut topo, face, lean, 20.0).unwrap();
        let before = solid_volume(&topo, solid, 0.001).unwrap();
        let edge = edge_between(
            &topo,
            solid,
            Point3::new(8.0, 8.0, 0.0),
            Point3::new(8.0, 40.0, 0.0),
        );

        let r = 1.0;
        let result = fillet_concave_ends(&mut topo, solid, &[edge], r)
            .unwrap()
            .unwrap();

        // The floor's normal against the leaning wall's, which runs along `y`
        // and `lean`.
        let wall = Vec3::new(0.0, 1.0, 0.0).cross(lean).normalize().unwrap();
        let turn = wall.dot(Vec3::new(0.0, 0.0, -1.0)).abs().acos();
        assert!((turn - std::f64::consts::FRAC_PI_2).abs() > 0.1);
        let section = r * r * ((turn / 2.0).tan() - turn / 2.0);
        assert_sound(&topo, result, before - section * 32.0, "a leaning wall");
    }

    /// A contact line past its face's far side reaches the opposite wall. Just
    /// over the wall's width the cut would remove nearly the section's volume,
    /// and a tall part hides a wider ball's shortfall in its own size. A ball
    /// exactly the wall's width lands its contact line on the far rim and fits.
    #[test]
    fn a_ball_wider_than_its_wall_is_refused_however_slightly() {
        for (r, height) in [(8.1, 20.0), (9.0, 1000.0)] {
            let mut topo = Topology::new();
            let solid = prism(&mut topo, &PEN_L, height);
            let edge = edge_between(
                &topo,
                solid,
                Point3::new(8.0, 8.0, 0.0),
                Point3::new(8.0, 40.0, 0.0),
            );
            assert!(
                fillet_concave_ends(&mut topo, solid, &[edge], r)
                    .unwrap()
                    .is_none(),
                "radius {r} at height {height}"
            );
        }

        let mut topo = Topology::new();
        let solid = prism(&mut topo, &PEN_L, 20.0);
        let edge = edge_between(
            &topo,
            solid,
            Point3::new(8.0, 8.0, 0.0),
            Point3::new(8.0, 40.0, 0.0),
        );
        let result = fillet_concave_ends(&mut topo, solid, &[edge], 8.0)
            .unwrap()
            .unwrap();
        let truth = PEN_L_VOLUME - square_section(8.0) * 32.0;
        assert_sound(&topo, result, truth, "a ball the wall's width");
    }

    /// A hole in the strip is boundary the round would have to run across.
    #[test]
    fn a_hole_in_the_strip_is_refused() {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &PEN_L, 20.0);
        let pin = make_cylinder(&mut topo, 0.25, 5.0).unwrap();
        let place = Mat4::translation(7.0, 20.0, -1.0);
        transform_solid(&mut topo, pin, &place).unwrap();
        let drilled = boolean(&mut topo, BooleanOp::Cut, solid, pin).unwrap();
        let edge = edge_between(
            &topo,
            drilled,
            Point3::new(8.0, 8.0, 0.0),
            Point3::new(8.0, 40.0, 0.0),
        );
        assert!(
            fillet_concave_ends(&mut topo, drilled, &[edge], 2.0)
                .unwrap()
                .is_none()
        );
    }

    /// Whatever this path cannot build, it hands back untouched: a convex end
    /// at both ends is the walking engine's, a ball wider than its wall takes
    /// off less than the section, and two edges sharing a corner need a corner.
    #[test]
    fn it_declines_what_it_cannot_build() {
        let mut topo = Topology::new();
        let solid = prism(&mut topo, &PEN_L, 20.0);
        let both_convex = edge_between(
            &topo,
            solid,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(60.0, 0.0, 0.0),
        );
        assert!(
            fillet_concave_ends(&mut topo, solid, &[both_convex], 1.0)
                .unwrap()
                .is_none()
        );
        let concave = edge_between(
            &topo,
            solid,
            Point3::new(8.0, 8.0, 0.0),
            Point3::new(8.0, 40.0, 0.0),
        );
        assert!(
            fillet_concave_ends(&mut topo, solid, &[concave], 9.0)
                .unwrap()
                .is_none()
        );
        let sharing = edge_between(
            &topo,
            solid,
            Point3::new(60.0, 8.0, 0.0),
            Point3::new(8.0, 8.0, 0.0),
        );
        assert!(
            fillet_concave_ends(&mut topo, solid, &[concave, sharing], 1.0)
                .unwrap()
                .is_none()
        );
        assert_eq!(solid_faces(&topo, solid).unwrap().len(), 8);
    }
}

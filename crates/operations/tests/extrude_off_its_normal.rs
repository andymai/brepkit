//! A profile with circular edges extruded along a direction off its plane's
//! normal. Each circle then sweeps an oblique cylinder, and the solid is a prism
//! whatever its slant: the profile's area times the height along the normal.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::Circle3D;
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
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

    const UP: Vec3 = Vec3::new(0.0, 0.0, 1.0);

    /// Builds a profile face and reports its area.
    type Profile = fn(&mut Topology) -> (FaceId, f64);

    fn at(x: f64, y: f64) -> Point3 {
        Point3::new(x, y, 0.0)
    }

    /// Edges round `corners`, each a line or, given its centre, an arc, sharing
    /// a vertex with the next.
    fn outline(topo: &mut Topology, corners: &[(Point3, Option<(Point3, f64)>)]) -> Vec<EdgeId> {
        let vertices: Vec<_> = corners
            .iter()
            .map(|&(p, _)| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        (0..corners.len())
            .map(|i| {
                let curve = match corners[i].1 {
                    Some((center, r)) => EdgeCurve::Circle(Circle3D::new(center, UP, r).unwrap()),
                    None => EdgeCurve::Line,
                };
                topo.add_edge(Edge::new(
                    vertices[i],
                    vertices[(i + 1) % corners.len()],
                    curve,
                ))
            })
            .collect()
    }

    fn loop_of(topo: &mut Topology, edges: Vec<EdgeId>) -> WireId {
        let oriented = edges
            .into_iter()
            .map(|e| OrientedEdge::new(e, true))
            .collect();
        topo.add_wire(Wire::new(oriented, true).unwrap())
    }

    fn plane_face(topo: &mut Topology, outer: WireId, holes: Vec<WireId>) -> FaceId {
        topo.add_face(Face::new(
            outer,
            holes,
            FaceSurface::Plane { normal: UP, d: 0.0 },
        ))
    }

    fn closed_circle(topo: &mut Topology, center: Point3, r: f64) -> EdgeId {
        let start = center + Vec3::new(r, 0.0, 0.0);
        let v = topo.add_vertex(Vertex::new(start, 1e-7));
        let circle = Circle3D::new(center, UP, r).unwrap();
        topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)))
    }

    /// A half disc of radius 5: an arc over its diameter.
    fn half_disc(topo: &mut Topology) -> (FaceId, f64) {
        let edges = outline(
            topo,
            &[
                (at(5.0, 0.0), Some((at(0.0, 0.0), 5.0))),
                (at(-5.0, 0.0), None),
            ],
        );
        let wire = loop_of(topo, edges);
        (plane_face(topo, wire, Vec::new()), PI * 25.0 / 2.0)
    }

    /// A whole disc of radius 5, on one closed edge.
    fn disc(topo: &mut Topology) -> (FaceId, f64) {
        let rim = closed_circle(topo, at(0.0, 0.0), 5.0);
        let wire = loop_of(topo, vec![rim]);
        (plane_face(topo, wire, Vec::new()), PI * 25.0)
    }

    /// A disc of radius 5 with a round hole of radius 1.5 off its centre.
    fn washer(topo: &mut Topology) -> (FaceId, f64) {
        let rim = closed_circle(topo, at(0.0, 0.0), 5.0);
        let wire = loop_of(topo, vec![rim]);
        let bore = closed_circle(topo, at(1.0, 0.5), 1.5);
        let hole = loop_of(topo, vec![bore]);
        (plane_face(topo, wire, vec![hole]), PI * (25.0 - 2.25))
    }

    /// A slot: two half circles of radius 2 joined by lines 6 long.
    fn slot(topo: &mut Topology) -> (FaceId, f64) {
        let edges = outline(
            topo,
            &[
                (at(3.0, -2.0), Some((at(3.0, 0.0), 2.0))),
                (at(3.0, 2.0), None),
                (at(-3.0, 2.0), Some((at(-3.0, 0.0), 2.0))),
                (at(-3.0, -2.0), None),
            ],
        );
        let wire = loop_of(topo, edges);
        (plane_face(topo, wire, Vec::new()), PI * 4.0 + 24.0)
    }

    fn assert_prism(topo: &Topology, solid: SolidId, volume: f64, what: &str) {
        let report = validate_solid(topo, solid).unwrap();
        assert!(report.is_valid(), "{what}: {:?}", report.issues);
        let exact = solid_volume(topo, solid, 0.001).unwrap();
        assert!(
            (exact - volume).abs() < 1e-6 * volume,
            "{what}: volume {exact}, expected {volume}"
        );
        let mesh = tessellate_solid(topo, solid, 0.01).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        let meshed = oriented_solid_volume(topo, solid, 0.01).unwrap();
        assert!(
            (meshed - volume).abs() < 5e-3 * volume,
            "{what}: meshed volume {meshed}, expected {volume}"
        );
    }

    #[test]
    fn a_slanted_profile_holds_its_area_times_its_height() {
        let profiles: [(&str, Profile); 4] = [
            ("half disc", half_disc),
            ("disc", disc),
            ("washer", washer),
            ("slot", slot),
        ];
        for (name, profile) in profiles {
            for lean in [
                Vec3::new(1.0, 0.0, 1.0),
                Vec3::new(0.2, 0.1, 1.0),
                Vec3::new(-0.7, 1.2, 1.0),
                Vec3::new(1e-4, 0.0, 1.0),
            ] {
                let along = lean.normalize().unwrap();
                let what = format!("{name} along {lean:?}");
                let mut topo = Topology::new();
                let (face, area) = profile(&mut topo);
                let solid = extrude(&mut topo, face, along, 10.0).unwrap();
                assert_prism(&topo, solid, area * along.z() * 10.0, &what);
            }
        }
    }

    /// Off the normal no side face reads as a `Cylinder`, and a point inside the
    /// slanted body but outside a square one reads inside.
    #[test]
    fn a_slanted_circle_sweeps_an_oblique_wall() {
        let along = Vec3::new(1.0, 0.0, 1.0).normalize().unwrap();
        let mut topo = Topology::new();
        let (face, _) = half_disc(&mut topo);
        let solid = extrude(&mut topo, face, along, 10.0).unwrap();
        let cylinders = solid_faces(&topo, solid)
            .unwrap()
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cylinders, 0);
        // At height 7 the profile has moved 7 along x.
        let height = 10.0 * along.z();
        let shift = 10.0 * along.x() * 7.0 / height;
        for (point, expected) in [
            (Point3::new(shift, 2.0, 7.0), PointClassification::Inside),
            (Point3::new(shift, 4.9, 7.0), PointClassification::Inside),
            (Point3::new(shift, 5.1, 7.0), PointClassification::Outside),
            (Point3::new(0.0, 2.0, 7.0), PointClassification::Outside),
            (Point3::new(shift, -1.0, 7.0), PointClassification::Outside),
        ] {
            let read = classify_point(&topo, solid, point, 0.01, 1e-7).unwrap();
            assert_eq!(read, expected, "{point:?}");
        }
    }

    /// Along the normal an arc still sweeps a true cylinder, in any pose.
    #[test]
    fn along_the_normal_an_arc_stays_a_cylinder() {
        let mut topo = Topology::new();
        let (face, area) = half_disc(&mut topo);
        let solid = extrude(&mut topo, face, UP, 10.0).unwrap();
        let turned = Mat4::translation(1.0, 2.0, 3.0) * Mat4::rotation_x(0.7);
        transform_solid(&mut topo, solid, &turned).unwrap();
        let cylinders = solid_faces(&topo, solid)
            .unwrap()
            .iter()
            .filter(|&&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .count();
        assert_eq!(cylinders, 1);
        assert_prism(&topo, solid, area * 10.0, "along the normal");
    }

    /// A slanted body goes on to booleans: a box across it takes off what the
    /// body holds in that height.
    #[test]
    fn a_slanted_body_takes_a_cut() {
        let along = Vec3::new(0.5, 0.0, 1.0).normalize().unwrap();
        let height = 10.0 * along.z();
        let mut topo = Topology::new();
        let (face, area) = disc(&mut topo);
        let solid = extrude(&mut topo, face, along, 10.0).unwrap();
        let slab = make_box(&mut topo, 60.0, 60.0, 2.0).unwrap();
        let place = Mat4::translation(-30.0, -30.0, 3.0);
        transform_solid(&mut topo, slab, &place).unwrap();
        let result = boolean(&mut topo, BooleanOp::Cut, solid, slab).unwrap();
        assert_prism(&topo, result, area * (height - 2.0), "cut across");
    }
}

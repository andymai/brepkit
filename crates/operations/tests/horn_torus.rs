//! A horn torus: major and minor radius equal, its tube closing on a pinch at
//! the centre, where the axis touches it. A point at radius `rho` from the axis
//! and height `z` lies inside when `z^2 < rho (2 R - rho)`, which lets a box's
//! share of it integrate across x and y.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean, mesh_fallback_count};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_torus};
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const R: f64 = 5.0;

    /// A horn torus's share of a box, by the height it holds over each cell of
    /// a grid across x and y.
    fn share_of_box(lo: (f64, f64, f64), hi: (f64, f64, f64)) -> f64 {
        const CELLS: u32 = 1200;
        let (dx, dy) = (
            (hi.0 - lo.0) / f64::from(CELLS),
            (hi.1 - lo.1) / f64::from(CELLS),
        );
        let mut volume = 0.0;
        for i in 0..CELLS {
            let x = (f64::from(i) + 0.5).mul_add(dx, lo.0);
            for j in 0..CELLS {
                let y = (f64::from(j) + 0.5).mul_add(dy, lo.1);
                let rho = x.hypot(y);
                let reach = (rho * (2.0 * R - rho)).max(0.0).sqrt();
                volume += (hi.2.min(reach) - lo.2.max(-reach)).max(0.0) * dx * dy;
            }
        }
        volume
    }

    fn placed_box(topo: &mut Topology, lo: (f64, f64, f64), hi: (f64, f64, f64)) -> SolidId {
        let b = make_box(topo, hi.0 - lo.0, hi.1 - lo.1, hi.2 - lo.2).unwrap();
        transform_solid(topo, b, &Mat4::translation(lo.0, lo.1, lo.2)).unwrap();
        b
    }

    fn posed(topo: &mut Topology, s: SolidId, pose: &str) -> SolidId {
        match pose {
            "turned" => {
                let m = Mat4::translation(1.0, -2.0, 0.5)
                    * Mat4::rotation_x(0.6)
                    * Mat4::rotation_z(0.4);
                transform_solid(topo, s, &m).unwrap();
                s
            }
            "mirrored" => mirror(
                topo,
                s,
                Point3::new(0.3, 0.0, 0.0),
                Vec3::new(1.0, 0.3, 0.2),
            )
            .unwrap(),
            _ => s,
        }
    }

    fn whole_volume() -> f64 {
        2.0 * PI * PI * R * R * R
    }

    #[test]
    fn a_horn_torus_stands_alone() {
        let mut topo = Topology::new();
        let horn = make_torus(&mut topo, R, R, 16).unwrap();
        let report = validate_solid(&topo, horn).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let mesh = tessellate_solid(&topo, horn, 0.01).unwrap();
        assert!(is_watertight(&mesh));
        let volume = solid_volume(&topo, horn, 0.001).unwrap();
        assert!(
            (volume - whole_volume()).abs() < 1e-9 * whole_volume(),
            "{volume}"
        );

        for (p, expected) in [
            (Point3::new(5.0, 0.0, 0.0), PointClassification::Inside),
            (Point3::new(5.0, 0.0, 4.9), PointClassification::Inside),
            (Point3::new(0.0, 0.0, 1.0), PointClassification::Outside),
            (Point3::new(0.0, 0.0, -3.0), PointClassification::Outside),
            (Point3::new(11.0, 0.0, 0.0), PointClassification::Outside),
            (Point3::new(0.0, 0.0, 0.0), PointClassification::OnBoundary),
        ] {
            let read = classify_point(&topo, horn, p, 0.01, 1e-7).unwrap();
            assert_eq!(read, expected, "{p:?}");
        }
    }

    /// Radii equal within the tolerance make one horn torus, with no tube left
    /// to cross the axis: its surface holds them exactly equal.
    #[test]
    fn radii_a_rounding_apart_are_equal() {
        for minor in [R + 5e-8, R - 5e-8] {
            let mut topo = Topology::new();
            let solid = make_torus(&mut topo, R, minor, 16).unwrap();
            let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
            let FaceSurface::Torus(torus) = topo.face(faces[0]).unwrap().surface() else {
                unreachable!("a torus face");
            };
            assert_eq!(
                torus.major_radius().to_bits(),
                torus.minor_radius().to_bits(),
                "minor {minor}"
            );
        }
    }

    /// Past the pinch the tube would pass through itself, which no solid does.
    #[test]
    fn a_spindle_past_the_horn_is_refused() {
        let mut topo = Topology::new();
        assert!(make_torus(&mut topo, R, R + 0.1, 16).is_err());
    }

    /// A disc of radius 5 turned whole about a line in its plane tangent to its
    /// rim is the same solid, and a quarter turn keeps the general path.
    #[test]
    fn a_disc_turned_about_its_tangent_is_a_horn_torus() {
        let disc = |topo: &mut Topology| {
            let normal = Vec3::new(0.0, -1.0, 0.0);
            let circle = Circle3D::new(Point3::new(R, 0.0, 0.0), normal, R).unwrap();
            let v = topo.add_vertex(Vertex::new(circle.evaluate(0.0), 1e-7));
            let rim = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));
            let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
            topo.add_face(Face::new(
                wire,
                Vec::new(),
                FaceSurface::Plane { normal, d: 0.0 },
            ))
        };
        let origin = Point3::new(0.0, 0.0, 0.0);
        let z = Vec3::new(0.0, 0.0, 1.0);

        let mut topo = Topology::new();
        let face = disc(&mut topo);
        let horn = revolve(&mut topo, face, origin, z, TAU).unwrap();
        assert!(validate_solid(&topo, horn).unwrap().is_valid());
        let mesh = tessellate_solid(&topo, horn, 0.01).unwrap();
        assert!(is_watertight(&mesh));
        let volume = solid_volume(&topo, horn, 0.001).unwrap();
        assert!(
            (volume - whole_volume()).abs() < 1e-9 * whole_volume(),
            "{volume}"
        );

        let mut topo = Topology::new();
        let face = disc(&mut topo);
        let quarter = revolve(&mut topo, face, origin, z, FRAC_PI_2).unwrap();
        let volume = solid_volume(&topo, quarter, 0.001).unwrap();
        assert!(
            (volume - whole_volume() / 4.0).abs() < 1e-3 * whole_volume(),
            "{volume}"
        );
    }

    /// Boxes round the pinch, clear of it, through it, and across the hole,
    /// whose two ops must each match the integrated share and make up the whole.
    #[test]
    fn booleans_through_the_pinch_hold_their_shares() {
        let boxes = [
            ((-20.0, -20.0, 1.0), (20.0, 20.0, 2.0)),
            ((2.0, -20.0, -20.0), (20.0, 20.0, 20.0)),
            ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
            ((-2.0, -2.0, 2.5), (2.0, 2.0, 3.5)),
            ((-1.0, -1.0, 0.5), (1.0, 1.0, 1.5)),
            ((-0.5, -0.5, -3.0), (0.5, 0.5, 3.0)),
            ((-2.0, -2.0, -1.0), (2.0, 2.0, 1.0)),
            ((-2.0, -2.0, -1.0), (6.0, 2.0, 1.0)),
        ];
        for pose in ["upright", "turned", "mirrored"] {
            for (lo, hi) in boxes {
                let share = share_of_box(lo, hi);
                let mut sum = 0.0;
                for op in [BooleanOp::Intersect, BooleanOp::Cut] {
                    let case = format!("{lo:?}..{hi:?} {pose} {op:?}");
                    let mut topo = Topology::new();
                    let horn = make_torus(&mut topo, R, R, 16).unwrap();
                    let tool = placed_box(&mut topo, lo, hi);
                    let horn = posed(&mut topo, horn, pose);
                    let tool = posed(&mut topo, tool, pose);
                    let before = mesh_fallback_count();
                    let result = boolean(&mut topo, op, horn, tool).unwrap();
                    assert_eq!(mesh_fallback_count(), before, "{case}: mesh fallback");
                    let report = validate_solid(&topo, result).unwrap();
                    assert!(report.is_valid(), "{case}: {:?}", report.issues);
                    for deflection in [0.02, 0.01] {
                        let mesh = tessellate_solid(&topo, result, deflection).unwrap();
                        assert!(is_watertight(&mesh), "{case}: open mesh at {deflection}");
                    }
                    let volume = solid_volume(&topo, result, 0.001).unwrap();
                    if op == BooleanOp::Intersect {
                        assert!(
                            (volume - share).abs() < 1e-3 * share.max(1.0),
                            "{case}: volume {volume}, its share {share}"
                        );
                    }
                    sum += volume;
                }
                assert!(
                    (sum - whole_volume()).abs() < 1e-6 * whole_volume(),
                    "{lo:?}..{hi:?} {pose}: the pieces sum to {sum}"
                );
            }
        }
    }
}

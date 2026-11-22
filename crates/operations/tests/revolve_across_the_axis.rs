//! A profile that reaches across the revolution axis. Each side of the axis
//! sweeps the space round it that the other side sweeps, the other way, so
//! the faces the two sides sweep overlap and no closed boundary holds the
//! result. Revolve refuses such a profile. The same arc drawn on one side of
//! the axis revolves into a closed spherical zone.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_3, PI, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::OperationsError;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceId, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const R: f64 = 100.0;

    /// The axis runs through `origin` along `axis`; `out` points from it
    /// into the profile's plane, and `side` is that plane's normal.
    struct Frame {
        origin: Point3,
        axis: Vec3,
        out: Vec3,
        side: Vec3,
    }

    impl Frame {
        fn at(&self, out: f64, up: f64) -> Point3 {
            self.origin + self.out * out + self.axis * up
        }
    }

    fn frames() -> [(&'static str, Frame); 3] {
        let axis = Vec3::new(0.3, -0.5, 0.8).normalize().unwrap();
        let out = axis.cross(Vec3::new(1.0, 0.0, 0.0)).normalize().unwrap();
        let upright = |out: Vec3| Frame {
            origin: Point3::new(0.0, 0.0, 0.0),
            axis: Vec3::new(0.0, 0.0, 1.0),
            out,
            side: Vec3::new(0.0, 0.0, 1.0).cross(out),
        };
        [
            ("upright", upright(Vec3::new(1.0, 0.0, 0.0))),
            (
                "tilted",
                Frame {
                    origin: Point3::new(37.0, -12.5, 410.0),
                    axis,
                    out,
                    side: axis.cross(out),
                },
            ),
            ("mirrored", upright(Vec3::new(-1.0, 0.0, 0.0))),
        ]
    }

    /// An arc of radius `R` about a point of the axis, from 60 degrees below
    /// the plane square to the axis at its centre to 60 degrees above it,
    /// closed by lines to the axis and along it. The short way (120 degrees) keeps
    /// to the `out` side; the long way (240 degrees) runs over both poles,
    /// across the axis.
    fn zone_profile(topo: &mut Topology, f: &Frame, long_way: bool) -> FaceId {
        let h = R * FRAC_PI_3.sin();
        let (a, b) = (f.at(0.5 * R, -h), f.at(0.5 * R, h));
        let (top, bottom) = (f.at(0.0, h), f.at(0.0, -h));
        let [va, vb, vt, vbot] = [a, b, top, bottom].map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        // About `side`, the circle turns from `out` toward the axis's down
        // direction: from `a` to `b` that is the long way round.
        let normal = if long_way { f.side } else { -f.side };
        let circle = Circle3D::new(f.origin, normal, R).unwrap();
        let edges = [
            Edge::new(va, vb, EdgeCurve::Circle(circle)),
            Edge::new(vb, vt, EdgeCurve::Line),
            Edge::new(vt, vbot, EdgeCurve::Line),
            Edge::new(vbot, va, EdgeCurve::Line),
        ]
        .map(|e| OrientedEdge::new(topo.add_edge(e), true));
        let wire = topo.add_wire(Wire::new(edges.to_vec(), true).unwrap());
        let d = f
            .side
            .dot(Vec3::new(f.origin.x(), f.origin.y(), f.origin.z()));
        topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: f.side, d },
        ))
    }

    /// The long way round, a full turn or a half turn: refused.
    #[test]
    fn an_arc_over_both_poles_is_refused() {
        for (name, f) in frames() {
            // The arc's middle lies across the axis.
            let mut topo = Topology::new();
            let face = zone_profile(&mut topo, &f, true);
            let wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
            let arc = topo.edge(wire.edges()[0].edge()).unwrap();
            let (a, b) = (
                topo.vertex(arc.start()).unwrap().point(),
                topo.vertex(arc.end()).unwrap().point(),
            );
            let (t0, t1) = arc.curve().domain_with_endpoints(a, b);
            assert!((t1 - t0 - 4.0 * FRAC_PI_3).abs() < 1e-9, "{name}");
            let middle = arc.curve().evaluate_with_endpoints(0.5 * (t0 + t1), a, b);
            assert!((middle - f.at(-R, 0.0)).length() < 1e-9, "{name}");

            for turn in [TAU, PI] {
                let mut topo = Topology::new();
                let face = zone_profile(&mut topo, &f, true);
                let result = revolve(&mut topo, face, f.origin, f.axis, turn);
                assert!(
                    matches!(result, Err(OperationsError::InvalidInput { .. })),
                    "{name}, turn {turn}: {result:?}"
                );
            }
        }
    }

    /// A rectangle across the axis: refused.
    #[test]
    fn a_rectangle_across_the_axis_is_refused() {
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let corners = [
                f.at(-1.0, 0.0),
                f.at(3.0, 0.0),
                f.at(3.0, 2.0),
                f.at(-1.0, 2.0),
            ];
            let vertices = corners.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
            let edges: Vec<OrientedEdge> = (0..4)
                .map(|i| {
                    let e = Edge::new(vertices[i], vertices[(i + 1) % 4], EdgeCurve::Line);
                    OrientedEdge::new(topo.add_edge(e), true)
                })
                .collect();
            let wire = topo.add_wire(Wire::new(edges, true).unwrap());
            let d = f
                .side
                .dot(Vec3::new(f.origin.x(), f.origin.y(), f.origin.z()));
            let face = topo.add_face(Face::new(
                wire,
                vec![],
                FaceSurface::Plane { normal: f.side, d },
            ));
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(
                matches!(result, Err(OperationsError::InvalidInput { .. })),
                "{name}: {result:?}"
            );
        }
    }

    /// The short way round: a valid zone whose mesh closes, its volume the
    /// zone's, `2 pi h (R^2 - h^2 / 3)`.
    #[test]
    fn the_arc_on_one_side_revolves_into_a_closed_zone() {
        let h = R * FRAC_PI_3.sin();
        let zone = TAU * h * (R * R - h * h / 3.0);
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let face = zone_profile(&mut topo, &f, false);
            let solid = revolve(&mut topo, face, f.origin, f.axis, TAU).unwrap();
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let volume = solid_volume(&topo, solid, 0.01).unwrap();
            assert!(
                (volume - zone).abs() < 1e-6 * zone,
                "{name}: volume {volume}, zone {zone}"
            );
            for deflection in [1.0, 0.05] {
                let mesh = tessellate_solid(&topo, solid, deflection).unwrap();
                assert!(is_watertight(&mesh), "{name}: open mesh at {deflection}");
            }
        }
    }
}

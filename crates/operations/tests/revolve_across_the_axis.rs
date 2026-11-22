//! A profile that reaches across the revolution axis. Each side of the axis
//! sweeps the space round it that the other side sweeps, the other way, so
//! the faces the two sides sweep overlap and no closed boundary holds the
//! result. Revolve refuses such a profile. The same arc drawn on one side of
//! the axis revolves into a closed spherical zone.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_3, PI, TAU};

    use brepkit_geometry::convert::circle_to_nurbs;
    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::curve::NurbsCurve;
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

    /// A face in the profile's plane bounded by `curve` from `a` to `b` and
    /// lines through `corners` back to `a`.
    fn profile(
        topo: &mut Topology,
        f: &Frame,
        (a, b): (Point3, Point3),
        curve: EdgeCurve,
        corners: &[Point3],
    ) -> FaceId {
        let points: Vec<Point3> = [a, b].into_iter().chain(corners.iter().copied()).collect();
        let vertices: Vec<_> = points
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let n = vertices.len();
        let mut edges = vec![OrientedEdge::new(
            topo.add_edge(Edge::new(vertices[0], vertices[1], curve)),
            true,
        )];
        for i in 1..n {
            let line = Edge::new(vertices[i], vertices[(i + 1) % n], EdgeCurve::Line);
            edges.push(OrientedEdge::new(topo.add_edge(line), true));
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let d = f
            .side
            .dot(Vec3::new(f.origin.x(), f.origin.y(), f.origin.z()));
        topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane { normal: f.side, d },
        ))
    }

    /// A half circle of radius `R` about the point `centre` out from the
    /// axis, from the angle `from` (read from `out` towards the axis's
    /// direction) half a turn on, closed by its chord, as a circle or as its
    /// rational B-spline.
    fn half_disc(topo: &mut Topology, f: &Frame, centre: f64, from: f64, spline: bool) -> FaceId {
        let weigh = |_: usize, w: f64| w;
        if spline {
            spline_half_disc(topo, f, centre, from, weigh)
        } else {
            let circle = Circle3D::new_with_ref(f.at(centre, 0.0), -f.side, R, f.out).unwrap();
            let ends = (circle.evaluate(from), circle.evaluate(from + PI));
            profile(topo, f, ends, EdgeCurve::Circle(circle), &[])
        }
    }

    /// [`half_disc`] as a rational B-spline whose `k`-th weight `w` is made
    /// `weigh(k, w)`.
    fn spline_half_disc(
        topo: &mut Topology,
        f: &Frame,
        centre: f64,
        from: f64,
        weigh: impl Fn(usize, f64) -> f64,
    ) -> FaceId {
        let circle = Circle3D::new_with_ref(f.at(centre, 0.0), -f.side, R, f.out).unwrap();
        let arc = circle_to_nurbs(&circle, from, from + PI).unwrap();
        let weights = arc
            .weights()
            .iter()
            .enumerate()
            .map(|(k, &w)| weigh(k, w))
            .collect();
        let arc = NurbsCurve::new(
            arc.degree(),
            arc.knots().to_vec(),
            arc.control_points().to_vec(),
            weights,
        )
        .unwrap();
        let ends = (circle.evaluate(from), circle.evaluate(from + PI));
        profile(topo, f, ends, EdgeCurve::NurbsCurve(arc), &[])
    }

    /// Whether `result` is revolve's refusal of a profile it reads against
    /// its axis.
    fn refused_at_the_axis<T: std::fmt::Debug>(result: &Result<T, OperationsError>) -> bool {
        matches!(result, Err(OperationsError::InvalidInput { reason }) if reason.contains("axis"))
    }

    /// A half circle whose middle reaches just across the axis, between
    /// where samples of it evenly spaced along its parameter would read it:
    /// refused, drawn as a circle or as a B-spline.
    #[test]
    fn an_arc_across_the_axis_between_samples_is_refused() {
        for (name, f) in frames() {
            for (centre, spline) in [(R - 0.01, false), (R - 0.001, true)] {
                let mut topo = Topology::new();
                let face = half_disc(&mut topo, &f, centre, 0.5 * PI + PI / 128.0, spline);
                let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
                assert!(
                    matches!(result, Err(OperationsError::InvalidInput { .. })),
                    "{name}, spline {spline}: {result:?}"
                );
            }
            // Kept clear of the axis by as much, it revolves.
            for (centre, spline) in [(R + 0.01, false), (R + 0.001, true)] {
                let mut topo = Topology::new();
                let face = half_disc(&mut topo, &f, centre, 0.5 * PI + PI / 128.0, spline);
                let solid = revolve(&mut topo, face, f.origin, f.axis, TAU).unwrap();
                let report = validate_solid(&topo, solid).unwrap();
                assert!(
                    report.is_valid(),
                    "{name}, spline {spline}: {:?}",
                    report.issues
                );
            }
        }
    }

    /// A quadratic B-spline whose first span has a zero weight, and whose
    /// span of width 1e-5 dips from `x = 1` to `x = -0.5`, across the axis:
    /// samples evenly spaced along it all read `x = 1`. No hull bounds the
    /// span with the zero weight, so the profile is refused.
    #[test]
    fn a_spline_with_a_zero_weight_and_a_narrow_dip_across_the_axis_is_refused() {
        let (a, b) = (0.500_117_070_312_5, 0.500_127_070_312_5);
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let points: Vec<Point3> = [1.0, 1.0, 1.0, -2.0, 1.0, 1.0, 1.0]
                .into_iter()
                .zip(0..)
                .map(|(x, z)| f.at(x, f64::from(z)))
                .collect();
            let spline = NurbsCurve::new(
                2,
                vec![0.0, 0.0, 0.0, a, a, b, b, 1.0, 1.0, 1.0],
                points.clone(),
                vec![1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            )
            .unwrap();
            let face = profile(
                &mut topo,
                &f,
                (points[0], points[6]),
                EdgeCurve::NurbsCurve(spline),
                &[f.at(3.0, 6.0), f.at(3.0, 0.0)],
            );
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(refused_at_the_axis(&result), "{name}: {result:?}");
        }
    }

    /// The half circle as a B-spline with every weight negative, the same
    /// curve: its hull bounds it. Across the axis between samples, it is
    /// refused for reaching across; clear of the axis, the test passes it.
    /// With one weight of the other sign, no hull bounds it, and it is
    /// refused.
    #[test]
    fn a_spline_with_negative_weights_is_read_by_its_hull() {
        let from = 0.5 * PI + PI / 128.0;
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let face = spline_half_disc(&mut topo, &f, R - 0.001, from, |_, w| -w);
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(
                matches!(&result, Err(OperationsError::InvalidInput { reason }) if reason.contains("reaches across")),
                "{name}: {result:?}"
            );
            let mut topo = Topology::new();
            let face = spline_half_disc(&mut topo, &f, R + 0.001, from, |_, w| -w);
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(!refused_at_the_axis(&result), "{name}: {result:?}");
            let mut topo = Topology::new();
            let face =
                spline_half_disc(
                    &mut topo,
                    &f,
                    R + 0.001,
                    from,
                    |k, w| {
                        if k == 1 { -w } else { w }
                    },
                );
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(
                matches!(&result, Err(OperationsError::InvalidInput { reason }) if reason.contains("rational span")),
                "{name}: {result:?}"
            );
        }
    }

    /// A quadratic B-spline of 801 spans along the axis. Each span but the
    /// first comes down to touch the axis once, at a third of the way along
    /// it or two thirds, between the points that halving it reads: its hull
    /// reaches across until it is halved several times. The first span dips
    /// 0.01 across the axis about two thirds of the way along it, where
    /// neither its ends nor its middle reach. Each span is halved within a
    /// budget of its own, so the many spans that only touch the axis leave
    /// the dip's span the halvings that find it, and the profile is refused.
    #[test]
    fn a_narrow_dip_after_many_spans_that_touch_the_axis_is_refused() {
        const SPANS: u32 = 801;
        // A span `x = A (t - 1/3)^2` from `x = A / 9` to `x = 4 A / 9`, or
        // the same run backward, whose middle control point is `-2 A / 9`.
        let a = 1000.0;
        let (low, high, middle) = (a / 9.0, 4.0 * a / 9.0, -2.0 * a / 9.0);
        // Lowering the dip span's middle control point by `d` lowers the
        // curve by `2 d t (1 - t)`, by 0.01 at t = 2 / 3.
        let d = 0.0225;
        for (name, f) in frames() {
            let mut points = Vec::new();
            let mut knots = vec![0.0; 3];
            for span in 0..SPANS {
                let z = f64::from(span);
                // The spans run high to low, then low to high, by turns.
                let start = if span.is_multiple_of(2) { high } else { low };
                let dip = if span == 0 { d } else { 0.0 };
                points.push(f.at(start, z));
                points.push(f.at(middle - dip, z + 0.5));
                if span > 0 {
                    knots.extend([z, z]);
                }
            }
            let end = if SPANS.is_multiple_of(2) { high } else { low };
            points.push(f.at(end, f64::from(SPANS)));
            knots.extend([f64::from(SPANS); 3]);
            let weights = vec![1.0; points.len()];
            let spline = NurbsCurve::new(2, knots, points.clone(), weights).unwrap();
            let top = f64::from(SPANS);
            let mut topo = Topology::new();
            let face = profile(
                &mut topo,
                &f,
                (points[0], points[points.len() - 1]),
                EdgeCurve::NurbsCurve(spline),
                &[f.at(500.0, top), f.at(500.0, 0.0)],
            );
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(
                matches!(&result, Err(OperationsError::InvalidInput { reason }) if reason.contains("reaches across")),
                "{name}: {result:?}"
            );
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

    /// A quadrilateral whose corners are `corners` in the frame's
    /// `(out, side, axis)` coordinates, on `surface`: its plane, or any
    /// surface the corners need not lie on, as revolve sweeps a curved
    /// profile's boundary alone.
    fn quad(
        topo: &mut Topology,
        f: &Frame,
        corners: [[f64; 3]; 4],
        surface: Option<FaceSurface>,
    ) -> FaceId {
        let points = corners.map(|[o, s, a]| f.origin + f.out * o + f.side * s + f.axis * a);
        let vertices = points.map(|p| topo.add_vertex(Vertex::new(p, 1e-7)));
        let edges: Vec<OrientedEdge> = (0..4)
            .map(|i| {
                let line = Edge::new(vertices[i], vertices[(i + 1) % 4], EdgeCurve::Line);
                OrientedEdge::new(topo.add_edge(line), true)
            })
            .collect();
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let surface = surface.unwrap_or_else(|| {
            let normal = (points[1] - points[0])
                .cross(points[2] - points[0])
                .normalize()
                .unwrap();
            let d = normal.dot(Vec3::new(points[0].x(), points[0].y(), points[0].z()));
            FaceSurface::Plane { normal, d }
        });
        topo.add_face(Face::new(wire, vec![], surface))
    }

    /// A square two wide in a plane that leans `lean` from the axis, `s`
    /// across the axis's shadow on that plane from `from` to `from + 2`,
    /// and from 1 to 3 along the shadow. The plane meets the axis at the
    /// frame's origin, or holds no point of it for a lean of nothing, where
    /// it stands `R / 10` from the axis.
    fn leaning_square(topo: &mut Topology, f: &Frame, lean: f64, from: f64) -> FaceId {
        let (sin, cos) = lean.sin_cos();
        let off = if lean == 0.0 { 0.1 * R } else { 0.0 };
        let corner = |s: f64, w: f64| [w * sin + off, s, w * cos];
        quad(
            topo,
            f,
            [
                corner(from, 1.0),
                corner(from + 2.0, 1.0),
                corner(from + 2.0, 3.0),
                corner(from, 3.0),
            ],
            None,
        )
    }

    /// A square in a plane that does not hold the axis, across the axis's
    /// shadow on it: refused for any turn, the plane leaning off the axis
    /// or standing beside it.
    #[test]
    fn a_square_across_the_shadow_of_the_axis_is_refused() {
        for (name, f) in frames() {
            for lean in [0.0, 0.3, 1.2] {
                for turn in [TAU, PI, 0.2] {
                    let mut topo = Topology::new();
                    let face = leaning_square(&mut topo, &f, lean, -0.5);
                    let result = revolve(&mut topo, face, f.origin, f.axis, turn);
                    assert!(
                        refused_at_the_axis(&result),
                        "{name}, lean {lean}, turn {turn}: {result:?}"
                    );
                }
            }
        }
    }

    /// The square on one side of the shadow revolves. A point `s` across the
    /// shadow and `w` along it lies `sqrt(s^2 + (w sin lean + off)^2)` from
    /// the axis, at height `w cos lean`: the drawing's area element is
    /// `|s| cos lean / r` of the square's, and the volume a turn `θ` sweeps
    /// is `θ cos lean` times the square's moment about the shadow, `4 * 2`.
    #[test]
    fn a_square_beside_the_shadow_of_the_axis_revolves() {
        for (name, f) in frames() {
            for lean in [0.0, 0.3, 1.2] {
                for turn in [TAU, PI] {
                    let mut topo = Topology::new();
                    let face = leaning_square(&mut topo, &f, lean, 1.0);
                    let solid = revolve(&mut topo, face, f.origin, f.axis, turn).unwrap();
                    let report = validate_solid(&topo, solid).unwrap();
                    assert!(
                        report.is_valid(),
                        "{name}, lean {lean}: {:?}",
                        report.issues
                    );
                    let expected = turn * lean.cos() * 8.0;
                    let volume = solid_volume(&topo, solid, 1e-3).unwrap();
                    assert!(
                        (volume - expected).abs() < 1e-4 * expected,
                        "{name}, lean {lean}, turn {turn}: volume {volume}, expected {expected}"
                    );
                }
            }
        }
    }

    /// A surface the boundaries below need not lie on.
    fn wall() -> FaceSurface {
        FaceSurface::Cylinder(
            brepkit_math::surfaces::CylindricalSurface::new(
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
            )
            .unwrap(),
        )
    }

    /// A curved profile's boundary in no one plane that winds round the
    /// axis: any surface it bounds meets the axis, and a full turn is
    /// refused.
    #[test]
    fn a_boundary_round_the_axis_is_refused() {
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let corners = [
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 1.0],
                [-2.0, 0.0, 0.0],
                [0.0, -2.0, 1.0],
            ];
            let face = quad(&mut topo, &f, corners, Some(wall()));
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(refused_at_the_axis(&result), "{name}: {result:?}");
        }
    }

    /// A curved profile's boundary in no one plane whose drawing in distance
    /// from the axis and height along it crosses itself: two of its points
    /// lie on one circle about the axis, and a full turn is refused. Its
    /// corners draw at (2, 0), (4, 2), (4, 0) and (2, 2).
    #[test]
    fn a_boundary_whose_drawing_crosses_itself_is_refused() {
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let corners = [
                [2.0, 0.0, 0.0],
                [0.0, 4.0, 2.0],
                [4.0, 0.0, 0.0],
                [0.0, -2.0, 2.0],
            ];
            let face = quad(&mut topo, &f, corners, Some(wall()));
            let result = revolve(&mut topo, face, f.origin, f.axis, TAU);
            assert!(refused_at_the_axis(&result), "{name}: {result:?}");
        }
    }

    /// A curved profile's boundary in no one plane whose drawing is a simple
    /// loop clear of the axis revolves into a valid solid.
    #[test]
    fn a_boundary_drawn_as_a_simple_loop_revolves() {
        for (name, f) in frames() {
            let mut topo = Topology::new();
            let corners = [
                [2.0, 0.0, 0.0],
                [4.0, 0.6, 0.0],
                [4.0, 0.0, 3.0],
                [2.0, 0.6, 3.0],
            ];
            let face = quad(&mut topo, &f, corners, Some(wall()));
            let solid = revolve(&mut topo, face, f.origin, f.axis, TAU).unwrap();
            let report = validate_solid(&topo, solid).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
        }
    }
}

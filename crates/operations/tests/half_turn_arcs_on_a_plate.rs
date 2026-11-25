//! A half turn of a circle or an ellipse and its chord, imprinted on a
//! plate's top, alone and with a line from the arc's middle to the chord's.
//!
//! The circle has radius 2 and the ellipse semi-axes 3 along x and 1.5,
//! both about (5, 5). Each arc runs counter-clockwise over half a turn from
//! one of several angles, so its chord passes the centre. Alone, the top
//! splits into the half and the rest; with the line, into two quarters and
//! the rest; with a small disc inside the circle's half, into the disc, the
//! half holding it, and the rest. Each arc edge of the top must run on the arc's own half, which
//! a reading from its ends the shorter way round could tip to the other.
//! Every case runs upright, turned and mirrored.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::curves::{Circle3D, Ellipse3D};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::imprint::{ImprintTool, imprint_faces};
    use brepkit_operations::measure::{face_area, solid_volume};
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_line_edge;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{FaceId, FaceSurface};
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

    /// The faces of the solid in the posed plate's top.
    fn top_faces(topo: &Topology, solid: SolidId, pose: Mat4) -> Vec<FaceId> {
        let origin = pose.mul_point(Point3::new(0.0, 0.0, 0.0));
        let top = pose.mul_point(Point3::new(0.0, 0.0, 2.0));
        let up = (top - origin).normalize().unwrap();
        solid_faces(topo, solid)
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
            .collect()
    }

    #[test]
    fn half_turn_arcs_and_their_chords_imprint_their_own_halves() {
        for (name, pose) in poses() {
            let at = |x: f64, y: f64| pose.mul_point(Point3::new(x, y, 2.0));
            let (u, v) = (at(1.0, 0.0) - at(0.0, 0.0), at(0.0, 1.0) - at(0.0, 0.0));
            let n = u.cross(v);
            let centre = at(5.0, 5.0);
            let curves = [
                (
                    EdgeCurve::Circle(Circle3D::with_axes(centre, n, 2.0, u, v).unwrap()),
                    4.0,
                ),
                (
                    EdgeCurve::Ellipse(Ellipse3D::with_axes(centre, n, 3.0, 1.5, u, v).unwrap()),
                    4.5,
                ),
            ];
            for (curve, ab) in &curves {
                let point = |t: f64| curve.evaluate_with_endpoints(t, centre, centre);
                // The angle about the centre, in the curve's own frame,
                // counter-clockwise from its first axis.
                let angle = |p: Point3| match curve {
                    EdgeCurve::Circle(c) => c.project(p),
                    EdgeCurve::Ellipse(e) => e.project(p),
                    _ => f64::NAN,
                };
                for start in [0.0, 0.3, 0.5 * PI, PI, 1.7] {
                    // An ellipse's segment holding a small disc is refused
                    // whatever the arc's sweep, so only the circle's is read.
                    let variants: &[(bool, bool)] = if matches!(curve, EdgeCurve::Circle(_)) {
                        &[(false, false), (true, false), (false, true)]
                    } else {
                        &[(false, false), (true, false)]
                    };
                    for &(with_line, with_disc) in variants {
                        let what = format!(
                            "{name}, {}, from {start:.3}, line {with_line}, disc {with_disc}",
                            curve.type_tag()
                        );
                        let mut topo = Topology::new();
                        let plate = make_box(&mut topo, 10.0, 10.0, 2.0).unwrap();
                        transform_solid(&mut topo, plate, &pose).unwrap();
                        let (p, q) = (point(start), point(start + PI));
                        let vp = topo.add_vertex(Vertex::new(p, 1e-7));
                        let vq = topo.add_vertex(Vertex::new(q, 1e-7));
                        let arc = topo.add_edge(Edge::new(vp, vq, curve.clone()));
                        let chord = make_line_edge(&mut topo, q, p, 1e-7).unwrap();
                        let mut tools = vec![ImprintTool::Edge(arc), ImprintTool::Edge(chord)];
                        if with_line {
                            let middle = point(start + 0.5 * PI);
                            tools.push(ImprintTool::Edge(
                                make_line_edge(&mut topo, middle, centre, 1e-7).unwrap(),
                            ));
                        }
                        // A small disc inside the half, which its face must
                        // hold as a hole.
                        let small = PI * 0.2 * 0.2;
                        if with_disc {
                            let inside = centre + (point(start + 0.5 * PI) - centre) * 0.5;
                            let disc = Circle3D::with_axes(inside, n, 0.2, u, v).unwrap();
                            let seam = topo.add_vertex(Vertex::new(disc.evaluate(0.0), 1e-7));
                            tools.push(ImprintTool::Edge(topo.add_edge(Edge::new(
                                seam,
                                seam,
                                EdgeCurve::Circle(disc),
                            ))));
                        }
                        let result = imprint_faces(&mut topo, plate, &tools).unwrap();
                        let half = 0.5 * PI * ab;
                        let mut want = if with_line {
                            vec![0.5 * half, 0.5 * half, 100.0 - half]
                        } else if with_disc {
                            vec![small, half - small, 100.0 - half]
                        } else {
                            vec![half, 100.0 - half]
                        };
                        want.sort_by(f64::total_cmp);
                        let faces = top_faces(&topo, result, pose);
                        let mut areas: Vec<f64> = faces
                            .iter()
                            .map(|&f| face_area(&topo, f, 0.001).unwrap())
                            .collect();
                        areas.sort_by(f64::total_cmp);
                        assert!(
                            areas.len() == want.len()
                                && areas.iter().zip(&want).all(|(a, w)| (a - w).abs() < 1e-9),
                            "{what}: top faces of area {areas:?}, want {want:?}"
                        );
                        // Each arc edge's middle, read as the edge reads its
                        // range from its vertices, lies on the arc's half.
                        for &f in &faces {
                            let face = topo.face(f).unwrap();
                            let wires = std::iter::once(face.outer_wire())
                                .chain(face.inner_wires().iter().copied());
                            for w in wires {
                                for oe in topo.wire(w).unwrap().edges() {
                                    let edge = topo.edge(oe.edge()).unwrap();
                                    if matches!(edge.curve(), EdgeCurve::Line) {
                                        continue;
                                    }
                                    let (s, e) = (
                                        topo.vertex(edge.start()).unwrap().point(),
                                        topo.vertex(edge.end()).unwrap().point(),
                                    );
                                    let (t0, t1) = edge.curve().domain_with_endpoints(s, e);
                                    let mid =
                                        edge.curve().evaluate_with_endpoints(0.5 * (t0 + t1), s, e);
                                    if (mid - centre).length() < 1.0 {
                                        continue;
                                    }
                                    let off = (angle(mid) - start).rem_euclid(2.0 * PI);
                                    assert!(
                                        off > 1e-6 && off < PI - 1e-6,
                                        "{what}: an arc edge from {s:?} to {e:?} runs through \
                                         {mid:?}, off the arc's half by angle {off}"
                                    );
                                }
                            }
                        }
                        let report = validate_solid(&topo, result).unwrap();
                        assert!(report.is_valid(), "{what}: {:?}", report.issues);
                        assert!(
                            is_watertight(&tessellate_solid(&topo, result, 0.02).unwrap()),
                            "{what}: open mesh"
                        );
                        let volume = solid_volume(&topo, result, 0.005).unwrap();
                        assert!((volume - 200.0).abs() < 1e-9, "{what}: volume {volume}");
                    }
                }
            }
        }
    }
}

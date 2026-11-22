//! A circular arc whose centre lies a hair off the revolution axis sweeps a
//! sphere band centred on the axis. The vertices and the edges of the band
//! lie on that sphere too.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_face_from_wire;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// A frame for the profile: the axis runs through `origin` along `axis`,
    /// and the profile lies in the half plane towards `out`.
    struct Frame {
        origin: Point3,
        axis: Vec3,
        out: Vec3,
    }

    impl Frame {
        fn at(&self, out: f64, up: f64) -> Point3 {
            self.origin + self.out * out + self.axis * up
        }
    }

    fn frames() -> [(&'static str, Frame); 3] {
        let axis = Vec3::new(0.3, -0.5, 0.8).normalize().unwrap();
        let out = axis.cross(Vec3::new(1.0, 0.0, 0.0)).normalize().unwrap();
        [
            (
                "upright",
                Frame {
                    origin: Point3::new(0.0, 0.0, 0.0),
                    axis: Vec3::new(0.0, 0.0, 1.0),
                    out: Vec3::new(1.0, 0.0, 0.0),
                },
            ),
            (
                "tilted",
                Frame {
                    origin: Point3::new(37.0, -12.5, 410.0),
                    axis,
                    out,
                },
            ),
            (
                "mirrored",
                Frame {
                    origin: Point3::new(0.0, 0.0, 0.0),
                    axis: Vec3::new(0.0, 0.0, 1.0),
                    out: Vec3::new(-1.0, 0.0, 0.0),
                },
            ),
        ]
    }

    /// The profile of a band of a sphere of radius `r`, from 60 degrees
    /// below its equator to 60 degrees above, closed by the axis. The arc's
    /// centre lies `off` out from the axis.
    fn band_profile(
        topo: &mut Topology,
        f: &Frame,
        r: f64,
        off: f64,
    ) -> brepkit_topology::face::FaceId {
        let h = r * FRAC_PI_3.sin();
        let w = r * FRAC_PI_3.cos();
        let pts = [
            f.at(off + w, -h),
            f.at(off + w, h),
            f.at(0.0, h),
            f.at(0.0, -h),
        ];
        let vs: Vec<_> = pts
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let centre = f.at(off, 0.0);
        let normal = f.out.cross(f.axis);
        let arc = Circle3D::new_with_ref(centre, -normal, r, pts[0] - centre).unwrap();
        let mut edge = |a: usize, b: usize, curve: EdgeCurve| {
            OrientedEdge::new(topo.add_edge(Edge::new(vs[a], vs[b], curve)), true)
        };
        let edges = vec![
            edge(0, 1, EdgeCurve::Circle(arc)),
            edge(1, 2, EdgeCurve::Line),
            edge(2, 3, EdgeCurve::Line),
            edge(3, 0, EdgeCurve::Line),
        ];
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        make_face_from_wire(topo, wire).unwrap()
    }

    /// The largest distance from a sphere face of the solid to a vertex or
    /// to a sample of an edge that bounds it, and the count of sphere faces.
    fn largest_gap_to_spheres(topo: &Topology, solid: SolidId) -> (f64, usize) {
        let mut gap = 0.0_f64;
        let mut spheres = 0;
        for face in solid_faces(topo, solid).unwrap() {
            let face = topo.face(face).unwrap();
            let FaceSurface::Sphere(sphere) = face.surface() else {
                continue;
            };
            spheres += 1;
            let off = |p: Point3| ((p - sphere.center()).length() - sphere.radius()).abs();
            let wires =
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
            for wire in wires {
                for oe in topo.wire(wire).unwrap().edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let a = topo.vertex(edge.start()).unwrap().point();
                    let b = topo.vertex(edge.end()).unwrap().point();
                    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                    gap = gap.max(off(a)).max(off(b));
                    for k in 0..=16 {
                        let t = t0 + (t1 - t0) * f64::from(k) / 16.0;
                        gap = gap.max(off(edge.curve().evaluate_with_endpoints(t, a, b)));
                    }
                }
            }
        }
        (gap, spheres)
    }

    #[test]
    fn a_band_swept_by_an_arc_a_hair_off_the_axis_lies_on_its_sphere() {
        let (r, off) = (100.0, 5e-7);
        let h = r * FRAC_PI_3.sin();
        // The arc swept about its own centre, with the band of width `off`
        // that the centre's place adds.
        let whole = std::f64::consts::PI * (2.0 * r * r * h - 2.0 * h * h * h / 3.0);
        for (label, f) in frames() {
            for angle in [TAU, FRAC_PI_2] {
                let mut topo = Topology::new();
                let face = band_profile(&mut topo, &f, r, off);
                let solid = revolve(&mut topo, face, f.origin, f.axis, angle).unwrap();
                let (gap, spheres) = largest_gap_to_spheres(&topo, solid);
                assert!(spheres > 0, "{label} {angle}: no sphere face");
                assert!(
                    gap <= 1e-7,
                    "{label} {angle}: an edge or vertex {gap:e} off its sphere"
                );
                let report = validate_solid(&topo, solid).unwrap();
                assert!(report.is_valid(), "{label} {angle}: {:?}", report.issues);
                let exact = whole * angle / TAU;
                let volume = solid_volume(&topo, solid, 0.01).unwrap();
                assert!(
                    (volume - exact).abs() < 1e-7 * exact,
                    "{label} {angle}: {volume} against {exact}"
                );
            }
        }
    }
}

//! A round of a tilted rim on a wall that is two faces of one surface: a
//! rod's wall extruded from a circle in two arcs, that wall drafted to a
//! frustum, or a ring revolved from a circle in two arcs. The rim crosses from one face to the other, and the
//! round's contact crosses between them away from where the rim does. The
//! round must be that of the same wall in one face: the same volume, valid,
//! with a closed mesh.
//!
//! The rod is the unit rod `0 <= z <= 4` less the half space above the plane
//! through `(0, 0, 3)` tilted `t` about the x axis. A level cut (`t = 0`)
//! rounded by `r` takes off the corner of area `(1 - π/4) r²` turned about
//! the axis at its centroid, `r (10 - 3π) / (3 (4 - π))` in from the wall.
//! A tilted cut has no closed form: the rounded rod is the union of the
//! balls of radius `r` in the cut rod, and numeric integration of that
//! union gives 9.3663916 for `t = 0.3`, `r = 0.2`, within 2e-6.
//!
//! The ring rounds the same way. The cut ring's volume is integrated in
//! cylindrical coordinates about its axis, and the material the round takes
//! off in tube coordinates about the curve of the balls' centres (each ball
//! on both the plane and the tube): out from that curve between the two
//! contact normals, from the ball's radius to where the line leaves the cut
//! ring. A ring cut square to `x` at `x = 1.9` and rounded by 0.1 is then
//! 0.957122072, and one cut at `x = 2.3` tilted 0.3 about `y` and rounded
//! by 0.08 is 0.0895633436, each within 1e-9.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_4, PI, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::draft::draft;
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_torus};
    use brepkit_operations::revolve::revolve;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(1.5, -2.0, 0.75) * Mat4::rotation_y(1.1) * Mat4::rotation_x(0.4),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// A plane face bounded by the circle about `center` with `normal` and
    /// `radius`, in two arcs that meet at `center ± radius * from`.
    fn disc_in_two_arcs(
        topo: &mut Topology,
        center: Point3,
        normal: Vec3,
        from: Vec3,
        radius: f64,
    ) -> brepkit_topology::face::FaceId {
        let a = topo.add_vertex(Vertex::new(center + from * radius, 1e-7));
        let b = topo.add_vertex(Vertex::new(center - from * radius, 1e-7));
        let circle = EdgeCurve::Circle(Circle3D::new(center, normal, radius).unwrap());
        let e1 = topo.add_edge(Edge::new(a, b, circle.clone()));
        let e2 = topo.add_edge(Edge::new(b, a, circle));
        let wire = Wire::new(
            vec![OrientedEdge::new(e1, true), OrientedEdge::new(e2, true)],
            true,
        )
        .unwrap();
        let wire = topo.add_wire(wire);
        let d = normal.dot(Vec3::new(center.x(), center.y(), center.z()));
        topo.add_face(Face::new(wire, vec![], FaceSurface::Plane { normal, d }))
    }

    /// The unit rod `0 <= z <= 4`, its wall in one face, or in two halves
    /// that meet along the lines at `seam` radians and opposite.
    fn rod(topo: &mut Topology, seam: Option<f64>) -> SolidId {
        let Some(seam) = seam else {
            return make_cylinder(topo, 1.0, 4.0).unwrap();
        };
        let up = Vec3::new(0.0, 0.0, 1.0);
        let from = Vec3::new(seam.cos(), seam.sin(), 0.0);
        let base = disc_in_two_arcs(topo, Point3::new(0.0, 0.0, 0.0), up, from, 1.0);
        extrude(topo, base, up, 4.0).unwrap()
    }

    /// The ring about the z axis of major radius 2 and minor radius 0.5,
    /// raised to `z = 3`, its
    /// tube in one face, or in two halves, inside and outside, that meet
    /// along its top and bottom circles.
    fn ring(topo: &mut Topology, halves: bool) -> SolidId {
        let ring = if halves {
            let profile = disc_in_two_arcs(
                topo,
                Point3::new(2.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                0.5,
            );
            let (origin, axis) = (Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0));
            revolve(topo, profile, origin, axis, TAU).unwrap()
        } else {
            make_torus(topo, 2.0, 0.5, 32).unwrap()
        };
        transform_solid(topo, ring, &Mat4::translation(0.0, 0.0, 3.0)).unwrap();
        ring
    }

    /// `solid` less the box `[0, 8]³` placed by `slab`, then placed by
    /// `pose`, and the edges of its plane faces with a vertex above
    /// `z = 2` before the pose.
    fn cut_and_rims(
        topo: &mut Topology,
        solid: SolidId,
        slab: &Mat4,
        pose: &Mat4,
    ) -> (SolidId, Vec<EdgeId>) {
        let cutter = make_box(topo, 8.0, 8.0, 8.0).unwrap();
        transform_solid(topo, cutter, slab).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, solid, cutter).unwrap();
        let mut rims = Vec::new();
        for face in solid_faces(topo, cut).unwrap() {
            let data = topo.face(face).unwrap();
            if !matches!(data.surface(), FaceSurface::Plane { .. }) {
                continue;
            }
            let wires: Vec<_> = std::iter::once(data.outer_wire())
                .chain(data.inner_wires().iter().copied())
                .collect();
            let mut edges = Vec::new();
            for wire in wires {
                edges.extend(
                    topo.wire(wire)
                        .unwrap()
                        .edges()
                        .iter()
                        .map(OrientedEdge::edge),
                );
            }
            let high = edges.iter().any(|&e| {
                let start = topo.edge(e).unwrap().start();
                topo.vertex(start).unwrap().point().z() > 2.0
            });
            if high {
                rims.extend(edges);
            }
        }
        transform_solid(topo, cut, pose).unwrap();
        (cut, rims)
    }

    /// The round of `rims` of `solid` by `r`, checked valid with a closed
    /// mesh. Returns its volume.
    fn round(label: &str, topo: &mut Topology, solid: SolidId, rims: &[EdgeId], r: f64) -> f64 {
        let round = fillet_v2(topo, solid, rims, r).unwrap();
        assert!(round.failed.is_empty(), "{label}: {:?}", round.failed);
        assert_ne!(round.solid, solid, "{label}: nothing rounded");
        let report = validate_solid(topo, round.solid).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {report:?}");
        let mesh = tessellate_solid(topo, round.solid, 0.005).unwrap();
        assert!(is_watertight(&mesh), "{label}: open mesh");
        solid_volume(topo, round.solid, 1e-4).unwrap()
    }

    /// The slab above the plane through `(0, 0, 3)` tilted `t` about x.
    fn above_the_top(t: f64) -> Mat4 {
        Mat4::translation(0.0, 0.0, 3.0) * Mat4::rotation_x(t) * Mat4::translation(-4.0, -4.0, 0.0)
    }

    /// The rim of a rod's top, cut level, rounds to the closed form whether
    /// its wall is one face or two.
    #[test]
    fn a_level_rim_on_a_split_rod_rounds_to_the_closed_form() {
        let r = 0.2;
        let area = r * r * (1.0 - FRAC_PI_4);
        let inset = r * (10.0 - 3.0 * PI) / (3.0 * (4.0 - PI));
        let expected = 3.0 * PI - TAU * (1.0 - inset) * area;
        for seam in [None, Some(0.0), Some(1.2)] {
            for (name, pose) in poses() {
                let label = format!("seam {seam:?}, {name}");
                let mut topo = Topology::new();
                let solid = rod(&mut topo, seam);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &above_the_top(0.0), &pose);
                let volume = round(&label, &mut topo, cut, &rims, r);
                assert!(
                    (volume - expected).abs() < 1e-9 * expected,
                    "{label}: volume {volume}, expected {expected}"
                );
            }
        }
    }

    /// The tilted rim of a rod whose wall is two faces rounds as the rod
    /// whose wall is one face, and both to the integrated volume.
    #[test]
    fn a_tilted_rim_on_a_split_rod_rounds_as_on_a_whole_rod() {
        let (t, r) = (0.3, 0.2);
        let integrated = 9.366_391_6;
        for (name, pose) in poses() {
            let mut volumes = Vec::new();
            for seam in [None, Some(0.0), Some(1.2)] {
                let label = format!("seam {seam:?}, {name}");
                let mut topo = Topology::new();
                let solid = rod(&mut topo, seam);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &above_the_top(t), &pose);
                let volume = round(&label, &mut topo, cut, &rims, r);
                assert!(
                    (volume - integrated).abs() < 1e-6 * integrated,
                    "{label}: volume {volume}, integrated {integrated}"
                );
                volumes.push(volume);
            }
            for volume in &volumes[1..] {
                assert!(
                    (volume - volumes[0]).abs() < 1e-9 * volumes[0],
                    "{name}: the split rods {volumes:?}"
                );
            }
        }
    }

    /// The rod of [`rod`] with its wall drafted 0.1 radians in toward `+z`
    /// about `z = 0`: a frustum, its wall in one cone face or two.
    fn frustum(topo: &mut Topology, seam: Option<f64>) -> SolidId {
        let solid = rod(topo, seam);
        let walls: Vec<_> = solid_faces(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_)))
            .collect();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let drafted = draft(topo, solid, &walls, up, Point3::new(0.0, 0.0, 0.0), 0.1).unwrap();
        let cones = solid_faces(topo, drafted)
            .unwrap()
            .into_iter()
            .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Cone(_)))
            .count();
        assert_eq!(cones, walls.len(), "the drafted walls");
        drafted
    }

    /// The tilted rim of a frustum whose wall is two cone faces rounds as
    /// the rim of the frustum whose wall is one.
    #[test]
    fn a_tilted_rim_on_a_split_frustum_rounds_as_on_a_whole_frustum() {
        let (t, r) = (0.3, 0.15);
        for (name, pose) in poses() {
            let mut volumes = Vec::new();
            for seam in [None, Some(0.0), Some(1.2)] {
                let label = format!("seam {seam:?}, {name}");
                let mut topo = Topology::new();
                let solid = frustum(&mut topo, seam);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &above_the_top(t), &pose);
                volumes.push(round(&label, &mut topo, cut, &rims, r));
            }
            for volume in &volumes[1..] {
                assert!(
                    (volume - volumes[0]).abs() < 1e-9 * volumes[0],
                    "{name}: the whole frustum and the split ones {volumes:?}"
                );
            }
        }
    }

    /// The slab beyond the plane through `(x, 0, 3)` turned `t` about `y`,
    /// on the side toward the ring's axis.
    fn beside_the_axis(x: f64, t: f64) -> Mat4 {
        Mat4::translation(x, 0.0, 3.0) * Mat4::rotation_y(t) * Mat4::translation(-8.0, -4.0, -4.0)
    }

    /// The rim of a ring cut square to `x` across its tube, where the tube
    /// is two faces that the rim crosses between four times and a seam cuts
    /// twice, rounds as the rim of the ring in one face, and both to the
    /// integrated volume. The round's contact runs straight over the top of
    /// the tube, where its face curves across it.
    #[test]
    fn a_level_rim_on_a_split_ring_rounds_to_the_integrated_volume() {
        let r = 0.1;
        let integrated = 0.957_122_072;
        for (name, pose) in poses() {
            let mut volumes = Vec::new();
            for halves in [false, true] {
                let label = format!("halves {halves}, {name}");
                let mut topo = Topology::new();
                let solid = ring(&mut topo, halves);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &beside_the_axis(1.9, 0.0), &pose);
                let volume = round(&label, &mut topo, cut, &rims, r);
                assert!(
                    (volume - integrated).abs() < 2e-7,
                    "{label}: volume {volume}, integrated {integrated}"
                );
                volumes.push(volume);
            }
            assert!(
                (volumes[1] - volumes[0]).abs() < 1e-7 * volumes[0],
                "{name}: the whole ring and the split one {volumes:?}"
            );
        }
    }

    /// The rim of a ring cut close to the outside of its tube, at a tilt,
    /// rounds to the integrated volume, with the tube in one face or two.
    /// The rim turns tighter at its narrow ends than the balls' centres lie
    /// from it, so planes square to the rim fold over the curve of the
    /// centres.
    #[test]
    fn a_rim_cut_close_to_a_rings_outside_rounds_to_the_integrated_volume() {
        let r = 0.08;
        let integrated = 0.089_563_343_6;
        for (name, pose) in poses() {
            for halves in [false, true] {
                let label = format!("halves {halves}, {name}");
                let mut topo = Topology::new();
                let solid = ring(&mut topo, halves);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &beside_the_axis(2.3, 0.3), &pose);
                let volume = round(&label, &mut topo, cut, &rims, r);
                assert!(
                    (volume - integrated).abs() < 2e-7,
                    "{label}: volume {volume}, integrated {integrated}"
                );
            }
        }
    }

    /// The rim of a ring cut by a plane tilted across its tube, where the
    /// tube is two faces that the rim crosses between four times, rounds as
    /// the rim of the ring in one face, within the fit of the walked round.
    #[test]
    fn a_tilted_rim_on_a_split_ring_rounds_as_on_a_whole_ring() {
        let r = 0.08;
        let slab = Mat4::translation(1.9, 0.0, 3.0)
            * Mat4::rotation_y(0.2)
            * Mat4::translation(-8.0, -4.0, -4.0);
        for (name, pose) in poses() {
            let mut volumes = Vec::new();
            for halves in [false, true] {
                let label = format!("halves {halves}, {name}");
                let mut topo = Topology::new();
                let solid = ring(&mut topo, halves);
                let (cut, rims) = cut_and_rims(&mut topo, solid, &slab, &pose);
                volumes.push(round(&label, &mut topo, cut, &rims, r));
            }
            assert!(
                (volumes[1] - volumes[0]).abs() < 1e-6 * volumes[0],
                "{name}: the whole ring and the split one {volumes:?}"
            );
        }
    }
}

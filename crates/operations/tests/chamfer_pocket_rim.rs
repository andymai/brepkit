//! A 200 x 80 x 20 block with its top at z = 0, less a square pocket w wide
//! centred on the top, one straight rim edge chamfered. The rim lies on the
//! top face's hole and ends at two of the pocket's corners. The chamfer
//! planes end on the pocket's side walls, which each take a triangle, and
//! the chamfer takes off a prism of d1 d2 / 2 along the rim's length.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::chamfer::{chamfer, chamfer_two_on};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::{FaceId, FaceSurface};
    use brepkit_topology::solid::SolidId;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(13.0, -7.5, 2.25) * Mat4::rotation_z(0.4) * Mat4::rotation_x(1.1),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    fn pocketed(topo: &mut Topology, w: f64, depth: f64, pose: &Mat4) -> SolidId {
        let block = make_box(topo, 200.0, 80.0, 20.0).unwrap();
        transform_solid(topo, block, &Mat4::translation(-100.0, -40.0, -20.0)).unwrap();
        let pocket = make_box(topo, w, w, depth + 10.0).unwrap();
        transform_solid(topo, pocket, &Mat4::translation(-w / 2.0, -w / 2.0, -depth)).unwrap();
        let solid = boolean(topo, BooleanOp::Cut, block, pocket).unwrap();
        transform_solid(topo, solid, pose).unwrap();
        solid
    }

    /// The edge of `solid` from `a` to `b` (either way round), in the
    /// block's own frame before `pose`.
    fn edge_between(topo: &Topology, solid: SolidId, pose: &Mat4, a: Point3, b: Point3) -> EdgeId {
        let (a, b) = (pose.mul_point(a), pose.mul_point(b));
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .find(|&e| {
                let edge = topo.edge(e).unwrap();
                let p = topo.vertex(edge.start()).unwrap().point();
                let q = topo.vertex(edge.end()).unwrap().point();
                ((p - a).length() < 1e-9 && (q - b).length() < 1e-9)
                    || ((p - b).length() < 1e-9 && (q - a).length() < 1e-9)
            })
            .unwrap()
    }

    /// The plane face along `edge` that holds the block-frame point `p`.
    fn plane_through(
        topo: &Topology,
        solid: SolidId,
        edge: EdgeId,
        pose: &Mat4,
        p: Point3,
    ) -> FaceId {
        let p = pose.mul_point(p);
        let adjacency = topo.build_adjacency(solid).unwrap();
        *adjacency
            .faces_for_edge(edge)
            .iter()
            .find(|&&f| {
                let FaceSurface::Plane { normal, d } = topo.face(f).unwrap().surface() else {
                    return false;
                };
                (normal
                    .x()
                    .mul_add(p.x(), normal.y().mul_add(p.y(), normal.z() * p.z()))
                    - d)
                    .abs()
                    < 1e-9
            })
            .unwrap()
    }

    /// Whether `solid` has a vertex at the block-frame point `p`.
    fn has_vertex(topo: &Topology, solid: SolidId, pose: &Mat4, p: Point3) -> bool {
        let p = pose.mul_point(p);
        solid_edges(topo, solid).unwrap().into_iter().any(|e| {
            let edge = topo.edge(e).unwrap();
            [edge.start(), edge.end()]
                .iter()
                .any(|&v| (topo.vertex(v).unwrap().point() - p).length() < 1e-7)
        })
    }

    fn check(
        label: &str,
        topo: &Topology,
        before: SolidId,
        result: SolidId,
        chamfers: usize,
        want: f64,
    ) {
        let report = validate_solid(topo, result).unwrap();
        assert!(report.is_valid(), "{label}: invalid: {:?}", report.issues);
        let mesh = tessellate_solid(topo, result, 0.05).unwrap();
        assert!(is_watertight(&mesh), "{label}: mesh not watertight");
        assert_eq!(
            solid_faces(topo, result).unwrap().len(),
            solid_faces(topo, before).unwrap().len() + chamfers,
            "{label}: faces"
        );
        let removed =
            solid_volume(topo, before, 0.01).unwrap() - solid_volume(topo, result, 0.01).unwrap();
        assert!(
            (removed - want).abs() <= 1e-9 * 320_000.0,
            "{label}: removed {removed}, want {want}"
        );
    }

    #[test]
    fn a_pocket_rim_takes_two_distances_from_either_face() {
        for (w, depth, d1, d2) in [
            (20.0, 5.0, 4.0, 2.0),
            (20.0, 5.0, 2.0, 1.0),
            (20.0, 5.0, 1.0, 1.0),
            (20.0, 20.0, 2.0, 1.0),
        ] {
            let h = w / 2.0;
            for (name, pose) in poses() {
                for on_top in [true, false] {
                    let mut topo = Topology::new();
                    let solid = pocketed(&mut topo, w, depth, &pose);
                    let rim = edge_between(
                        &topo,
                        solid,
                        &pose,
                        Point3::new(-h, -h, 0.0),
                        Point3::new(h, -h, 0.0),
                    );
                    let reference = if on_top {
                        plane_through(&topo, solid, rim, &pose, Point3::new(-50.0, -30.0, 0.0))
                    } else {
                        plane_through(&topo, solid, rim, &pose, Point3::new(0.0, -h, -depth))
                    };
                    let label = format!("{w}x{depth} {d1},{d2} {name} on top {on_top}");
                    let result = chamfer_two_on(&mut topo, solid, &[(rim, reference)], d1, d2)
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    check(&label, &topo, solid, result, 1, w * d1 * d2 / 2.0);
                    let (top, wall) = if on_top { (d1, d2) } else { (d2, d1) };
                    for x in [-h, h] {
                        assert!(
                            has_vertex(&topo, result, &pose, Point3::new(x, -h - top, 0.0)),
                            "{label}: the top is set back by {top}"
                        );
                        assert!(
                            has_vertex(&topo, result, &pose, Point3::new(x, -h, -wall)),
                            "{label}: the wall is set back by {wall}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_pocket_rim_takes_one_distance() {
        for (w, depth, d) in [(20.0, 5.0, 4.0), (20.0, 5.0, 1.0), (20.0, 20.0, 2.0)] {
            let h = w / 2.0;
            for (name, pose) in poses() {
                let mut topo = Topology::new();
                let solid = pocketed(&mut topo, w, depth, &pose);
                let rim = edge_between(
                    &topo,
                    solid,
                    &pose,
                    Point3::new(-h, -h, 0.0),
                    Point3::new(h, -h, 0.0),
                );
                let label = format!("{w}x{depth} {d} {name}");
                let result =
                    chamfer(&mut topo, solid, &[rim], d).unwrap_or_else(|e| panic!("{label}: {e}"));
                check(&label, &topo, solid, result, 1, w * d * d / 2.0);
            }
        }
    }

    /// A diamond pocket beside the rim's end: one of its edges, from
    /// (9.8, -12.8) to (11.3, -11.3), spans the contact's end in both
    /// directions of the contact's frame but passes 0.42 from it.
    #[test]
    fn a_hole_beside_the_contacts_end_leaves_the_chamfer_alone() {
        let (w, depth, d) = (20.0, 5.0, 2.0);
        let h = w / 2.0;
        let side = 1.5 * std::f64::consts::SQRT_2;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let block = pocketed(&mut topo, w, depth, &Mat4::identity());
            let diamond = make_box(&mut topo, side, side, depth + 10.0).unwrap();
            let place = Mat4::translation(11.3, -14.3, -depth)
                * Mat4::rotation_z(std::f64::consts::FRAC_PI_4);
            transform_solid(&mut topo, diamond, &place).unwrap();
            let solid = boolean(&mut topo, BooleanOp::Cut, block, diamond).unwrap();
            transform_solid(&mut topo, solid, &pose).unwrap();
            let rim = edge_between(
                &topo,
                solid,
                &pose,
                Point3::new(-h, -h, 0.0),
                Point3::new(h, -h, 0.0),
            );
            let result =
                chamfer(&mut topo, solid, &[rim], d).unwrap_or_else(|e| panic!("{name}: {e}"));
            check(name, &topo, solid, result, 1, w * d * d / 2.0);
        }
    }

    /// Two opposite rims share no corner: each chamfer ends on the side
    /// walls alone.
    #[test]
    fn two_opposite_pocket_rims_chamfer_together() {
        let (w, depth, d) = (20.0, 5.0, 2.0);
        let h = w / 2.0;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = pocketed(&mut topo, w, depth, &pose);
            let rims = [-h, h].map(|y| {
                edge_between(
                    &topo,
                    solid,
                    &pose,
                    Point3::new(-h, y, 0.0),
                    Point3::new(h, y, 0.0),
                )
            });
            let result =
                chamfer(&mut topo, solid, &rims, d).unwrap_or_else(|e| panic!("{name}: {e}"));
            check(name, &topo, solid, result, 2, 2.0 * w * d * d / 2.0);
        }
    }
}

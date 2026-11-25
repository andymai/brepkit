//! A ball or a ring that pokes a hundredth past a face of a solid it
//! otherwise lies in: the plane cap of a prism with B-spline walls, or the
//! cylinder band of a rod, alone or fused with a block. The fuse holds the
//! poke and the common part leaves it out, or the boolean is refused: the
//! solid is never read as inside the other.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    use brepkit_math::curves::Ellipse3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere, make_torus};
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::solid::SolidId;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

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

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    /// A prism of semi-axes 3 along x and 2 along y about (4, 5), from
    /// z = -1 to 3, its wall a B-spline extrusion of the ellipse.
    fn prism(topo: &mut Topology) -> SolidId {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let ellipse = Ellipse3D::with_axes(
            Point3::new(4.0, 5.0, -1.0),
            up,
            3.0,
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        )
        .unwrap();
        let v = topo.add_vertex(Vertex::new(ellipse.evaluate(0.0), 1e-7));
        let rim = topo.add_edge(Edge::new(v, v, EdgeCurve::Ellipse(ellipse)));
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(rim, true)], true).unwrap());
        let base = topo.add_face(Face::new(
            wire,
            Vec::new(),
            FaceSurface::Plane {
                normal: up,
                d: -1.0,
            },
        ));
        extrude(topo, base, up, 4.0).unwrap()
    }

    /// A rod of radius 3 about the z axis from z = 0 to 2, alone or fused
    /// with a block on its far side from the poke.
    fn rod(topo: &mut Topology, with_block: bool) -> SolidId {
        let rod = make_cylinder(topo, 3.0, 2.0).unwrap();
        if !with_block {
            return rod;
        }
        let block = make_box(topo, 3.0, 1.0, 1.0).unwrap();
        transform_solid(topo, block, &Mat4::translation(-4.0, -0.5, 0.5)).unwrap();
        boolean_with_options(topo, BooleanOp::Fuse, rod, block, exact()).unwrap()
    }

    /// The direction from the rod's axis to where the ring pokes out.
    fn out() -> Vec3 {
        Vec3::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2, 0.0)
    }

    /// A ring of radii 1 and 0.3 about a centre 1.71 along `out` at z = 1,
    /// its axis tilted by `tilt` about the line across `out`, and turned
    /// by `turn` about its own axis first: it pokes 0.01 past the rod's
    /// band only where the tilt is zero. With a tilt, its centre moves out
    /// so that it pokes 0.01 again.
    fn ring(topo: &mut Topology, turn: f64, tilt: f64) -> (SolidId, Point3) {
        let ring = make_torus(topo, 1.0, 0.3, 16).unwrap();
        // Tilted about the line across `out`, the ring reaches out along
        // `out` by its major radius times the cosine plus its tube.
        let reach = 1.0 * tilt.cos() + 0.3;
        let centre = 3.01 - reach;
        let place = Mat4::translation(centre * FRAC_1_SQRT_2, centre * FRAC_1_SQRT_2, 1.0)
            * Mat4::rotation_z(PI / 4.0)
            * Mat4::rotation_y(tilt)
            * Mat4::rotation_z(turn);
        transform_solid(topo, ring, &place).unwrap();
        // Half way out of the rod from the poke's deepest point, which lies
        // below the centre by the major radius times the sine of the tilt.
        let poke = Point3::new(0.0, 0.0, 1.0 - tilt.sin()) + out() * 3.005;
        (ring, poke)
    }

    /// A ball of radius 0.5 with `pose_ball` placed about `centre`.
    fn ball(topo: &mut Topology, centre: Point3, pose_ball: &Mat4) -> SolidId {
        let ball = make_sphere(topo, 0.5, 16).unwrap();
        transform_solid(
            topo,
            ball,
            &(Mat4::translation(centre.x(), centre.y(), centre.z()) * *pose_ball),
        )
        .unwrap();
        ball
    }

    /// Each boolean of `outer` and `inner` that misreads `poke`: the fuse
    /// must hold it and the common part leave it out, unless refused.
    fn poke_misread(
        topo: &mut Topology,
        (outer, inner): (SolidId, SolidId),
        poke: Point3,
        pose: &Mat4,
        what: &str,
    ) -> Vec<String> {
        let poke = pose.mul_point(poke);
        for solid in [outer, inner] {
            transform_solid(topo, solid, pose).unwrap();
        }
        assert_eq!(
            classify_point(topo, outer, poke, 1e-3, 1e-7).unwrap(),
            PointClassification::Outside,
            "{what}: the poke lies outside the container"
        );
        assert_eq!(
            classify_point(topo, inner, poke, 1e-3, 1e-7).unwrap(),
            PointClassification::Inside,
            "{what}: the poke lies inside the solid"
        );
        let mut wrong = Vec::new();
        for (op, inside) in [
            (BooleanOp::Fuse, PointClassification::Inside),
            (BooleanOp::Intersect, PointClassification::Outside),
        ] {
            let mut copy = topo.clone();
            if let Ok(result) = boolean_with_options(&mut copy, op, outer, inner, exact()) {
                let read = classify_point(&copy, result, poke, 1e-3, 1e-7).unwrap();
                if read != inside {
                    wrong.push(format!("{what} {op:?}: {read:?}"));
                }
            }
        }
        wrong
    }

    #[test]
    fn a_ball_poking_through_a_prisms_cap_is_not_held_in_it() {
        let mut wrong = Vec::new();
        for (name, pose) in poses() {
            for (lean, pose_ball) in [
                ("upright ball", Mat4::identity()),
                ("ball on its side", Mat4::rotation_x(PI / 2.0)),
            ] {
                let mut topo = Topology::new();
                let outer = prism(&mut topo);
                let inner = ball(&mut topo, Point3::new(4.0, 5.0, 2.51), &pose_ball);
                let poke = Point3::new(4.0, 5.0, 3.005);
                wrong.extend(poke_misread(
                    &mut topo,
                    (outer, inner),
                    poke,
                    &pose,
                    &format!("{name}, {lean}"),
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn a_ball_poking_through_a_rods_band_is_not_held_in_it() {
        let mut wrong = Vec::new();
        for (name, pose) in poses() {
            for with_block in [false, true] {
                let mut topo = Topology::new();
                let outer = rod(&mut topo, with_block);
                let centre = Point3::new(0.0, 0.0, 1.0) + out() * 2.51;
                let inner = ball(&mut topo, centre, &Mat4::rotation_x(PI / 2.0));
                let poke = Point3::new(0.0, 0.0, 1.0) + out() * 3.005;
                wrong.extend(poke_misread(
                    &mut topo,
                    (outer, inner),
                    poke,
                    &pose,
                    &format!("{name}, block {with_block}"),
                ));
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }

    #[test]
    fn a_ring_poking_through_a_rods_band_is_not_held_in_it() {
        let mut wrong = Vec::new();
        for (name, pose) in poses() {
            for with_block in [false, true] {
                for (turn, tilt) in [(0.0, 0.0), (PI, 0.0), (2.0, 0.0), (0.0, 0.3), (2.5, 0.3)] {
                    let mut topo = Topology::new();
                    let outer = rod(&mut topo, with_block);
                    let (inner, poke) = ring(&mut topo, turn, tilt);
                    wrong.extend(poke_misread(
                        &mut topo,
                        (outer, inner),
                        poke,
                        &pose,
                        &format!("{name}, block {with_block}, turn {turn}, tilt {tilt}"),
                    ));
                }
            }
        }
        assert!(wrong.is_empty(), "{wrong:#?}");
    }
}

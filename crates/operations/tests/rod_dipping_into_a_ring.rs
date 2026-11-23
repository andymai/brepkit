//! A rod of radius 0.6 along y whose underside dips into the top of the
//! ring of radii 4 and 1.5 by 7e-4 or 2.7e-5, where the two surfaces cross
//! at about 0.03 to 0.05 or 0.006 to 0.01 radians, in two loops 0.02 to 0.04
//! across. Near the origin and moved 1e6 or 1e9 along each axis, the cut
//! returned the whole ring and the intersect nothing: the dips were missed
//! with no error. The boolean now carves each dip or refuses; it never
//! misses a section without saying so.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::mat::Mat4;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::primitives::{make_cylinder, make_torus};
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;

    #[test]
    fn a_rod_dipping_into_a_ring_is_carved_or_refused_in_every_pose() {
        let poses = [
            [0.0, 0.0, 0.0],
            [1e6, 0.0, 0.0],
            [0.0, 1e6, 0.0],
            [0.0, 0.0, 1e6],
            [1e9, 0.0, 0.0],
            [0.0, 1e9, 0.0],
            [0.0, 0.0, 1e9],
        ];
        for depth in [7e-4, 2.7e-5] {
            for (op, untouched) in [(BooleanOp::Cut, 1), (BooleanOp::Intersect, 0)] {
                for s in poses {
                    let mut topo = Topology::new();
                    let ring = make_torus(&mut topo, 4.0, 1.5, 32).unwrap();
                    let rod = make_cylinder(&mut topo, 0.6, 20.0).unwrap();
                    let lay = Mat4::translation(0.0, 10.0, 1.5 + 0.6 - depth)
                        * Mat4::rotation_x(std::f64::consts::FRAC_PI_2);
                    transform_solid(&mut topo, rod, &lay).unwrap();
                    for solid in [ring, rod] {
                        transform_solid(&mut topo, solid, &Mat4::translation(s[0], s[1], s[2]))
                            .unwrap();
                    }
                    let exact = BooleanOptions {
                        mesh_fallback: false,
                        ..BooleanOptions::default()
                    };
                    if let Ok(piece) = boolean_with_options(&mut topo, op, ring, rod, exact) {
                        let faces = solid_faces(&topo, piece).unwrap().len();
                        assert!(
                            faces > untouched,
                            "depth {depth} {op:?} moved {s:?}: the dips are missed ({faces} faces)"
                        );
                    }
                }
            }
        }
    }
}

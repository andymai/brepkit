//! A round of the rim where a plane `z = c` cuts a unit ball, the part above
//! the plane kept. The ball's two faces meet at its equator, so below it the
//! cap's rim lies on a narrow band of sphere between the rim and the
//! equator, and a round wide enough runs its contact on to the equator or
//! past it, on to the face above.
//!
//! A ball of radius `r` rolls inside the sphere on the plane: its centre
//! stands at `z_c = c + r`, `ρ_c = sqrt((1 - r)² - z_c²)` from the axis.
//! The round takes off the corner between the plane, the sphere and the
//! ball's arc, turned about the axis: by Green's theorem in the `(ρ, z)`
//! half plane, `2π ∮ ρ²/2 dz` round that corner.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    use brepkit_math::mat::Mat4;
    use brepkit_math::quadrature::gauss_legendre_points;
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_sphere};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    fn integral(a: f64, b: f64, f: impl Fn(f64) -> f64) -> f64 {
        let panels = 64;
        let h = (b - a) / f64::from(panels);
        (0..panels)
            .map(|k| {
                let lo = h.mul_add(f64::from(k), a);
                gauss_legendre_points(20)
                    .iter()
                    .map(|g| 0.5 * h * g.w * f((0.5 * h).mul_add(g.x, lo + 0.5 * h)))
                    .sum::<f64>()
            })
            .sum()
    }

    /// The volume the round of `r` takes off the cap above `z = c`.
    fn removed(c: f64, r: f64) -> f64 {
        let zc = c + r;
        let rc = ((1.0 - r).powi(2) - zc * zc).sqrt();
        let sphere = integral(c.asin(), zc.atan2(rc), |phi| 0.5 * phi.cos().powi(3));
        let arc = integral(zc.atan2(rc), -FRAC_PI_2, |a| {
            0.5 * r.mul_add(a.cos(), rc).powi(2) * r * a.cos()
        });
        TAU * (sphere + arc)
    }

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// The unit ball less the box below `z = c`, placed by `pose`, and the
    /// circles of its rim.
    fn cap(topo: &mut Topology, c: f64, pose: &Mat4) -> (SolidId, Vec<EdgeId>) {
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let slab = make_box(topo, 4.0, 4.0, 4.0).unwrap();
        transform_solid(topo, slab, &Mat4::translation(-2.0, -2.0, -4.0 + c)).unwrap();
        let cap = boolean(topo, BooleanOp::Cut, ball, slab).unwrap();
        transform_solid(topo, cap, pose).unwrap();
        let rims = solid_edges(topo, cap)
            .unwrap()
            .into_iter()
            .filter(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Circle(_)))
            .collect();
        (cap, rims)
    }

    /// The unit ball less the box below a plane `tilt` off level at `c` from
    /// the centre, placed by `pose`, and the edges of its flat face. The cut
    /// is the cap above `z = c` turned, but the ball's equator stays level,
    /// so the rim crosses it: half the rim lies on one face of the ball and
    /// half on the other.
    fn tilted_cap(topo: &mut Topology, c: f64, tilt: f64, pose: &Mat4) -> (SolidId, Vec<EdgeId>) {
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let slab = make_box(topo, 4.0, 4.0, 4.0).unwrap();
        let place = Mat4::rotation_x(tilt) * Mat4::translation(-2.0, -2.0, -4.0 + c);
        transform_solid(topo, slab, &place).unwrap();
        let cap = boolean(topo, BooleanOp::Cut, ball, slab).unwrap();
        transform_solid(topo, cap, pose).unwrap();
        let flat = solid_faces(topo, cap)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { .. }))
            .unwrap();
        let rims = topo
            .wire(topo.face(flat).unwrap().outer_wire())
            .unwrap()
            .edges()
            .iter()
            .map(brepkit_topology::wire::OrientedEdge::edge)
            .collect();
        (cap, rims)
    }

    /// The round of a rim that crosses the ball's equator is the round of
    /// the same cut level, turned: the same volume, valid, with a closed
    /// mesh. The cap is left as it was.
    #[test]
    fn a_round_of_a_tilted_caps_rim_across_the_equator_is_the_level_round_turned() {
        for (c, tilt, r) in [(-0.1, 0.3, 0.2), (-0.15, 0.2, 0.1), (0.1, 0.25, 0.15)] {
            for (name, pose) in poses() {
                let what = format!("c = {c}, tilt = {tilt}, r = {r}, {name}");
                let mut topo = Topology::new();
                let (solid, rims) = tilted_cap(&mut topo, c, tilt, &pose);
                let faces_before = solid_faces(&topo, solid).unwrap();
                let volume_before = solid_volume(&topo, solid, 1e-4).unwrap();
                let round = fillet_v2(&mut topo, solid, &rims, r).unwrap();
                assert!(round.failed.is_empty(), "{what}: {:?}", round.failed);
                assert!(
                    validate_solid(&topo, round.solid).unwrap().is_valid(),
                    "{what}: not valid"
                );
                let h = 1.0 - c;
                let expected = PI * h * h * (3.0 - h) / 3.0 - removed(c, r);
                let volume = solid_volume(&topo, round.solid, 1e-4).unwrap();
                assert!(
                    (volume - expected).abs() < 1e-6 * expected,
                    "{what}: volume {volume}, expected {expected}"
                );
                for deflection in [0.01, 0.002] {
                    let mesh = tessellate_solid(&topo, round.solid, deflection).unwrap();
                    assert!(is_watertight(&mesh), "{what}: open mesh at {deflection}");
                }
                assert_eq!(
                    solid_faces(&topo, solid).unwrap(),
                    faces_before,
                    "{what}: the cap's faces"
                );
                let after = solid_volume(&topo, solid, 1e-4).unwrap();
                assert!(
                    (after - volume_before).abs() < 1e-9 * volume_before,
                    "{what}: the cap's volume {after}, before {volume_before}"
                );
            }
        }
    }

    /// Short of the equator (`c = -0.5`), on it (`r = -c`) and past it.
    #[test]
    fn a_round_of_a_ball_caps_rim_runs_on_to_the_face_past_the_equator() {
        for (c, r) in [(-0.5, 0.2), (-0.2, 0.2), (-0.2, 0.25), (-0.1, 0.15)] {
            for (name, pose) in poses() {
                let what = format!("c = {c}, r = {r}, {name}");
                let mut topo = Topology::new();
                let (solid, rims) = cap(&mut topo, c, &pose);
                let round = fillet_v2(&mut topo, solid, &rims, r).unwrap();
                assert!(round.failed.is_empty(), "{what}: {:?}", round.failed);
                assert!(
                    validate_solid(&topo, round.solid).unwrap().is_valid(),
                    "{what}: not valid"
                );
                let h = 1.0 - c;
                let expected = PI * h * h * (3.0 - h) / 3.0 - removed(c, r);
                let volume = solid_volume(&topo, round.solid, 1e-4).unwrap();
                assert!(
                    (volume - expected).abs() < 1e-5 * expected,
                    "{what}: volume {volume}, expected {expected}"
                );
                let mesh = tessellate_solid(&topo, round.solid, 0.002).unwrap();
                assert!(is_watertight(&mesh), "{what}: open mesh");
            }
        }
    }
}

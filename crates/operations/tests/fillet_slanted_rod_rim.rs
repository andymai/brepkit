//! A round of the ellipse where a slanted plane cuts a rod.
//!
//! The rod has radius 2 and stands 5 high; the plane through `(0, 0, 4)`
//! leans 0.3 about `x`, so the rod keeps `z <= 4 + y tan 0.3`. No closed form
//! rounds the ellipse, so the band is walked. The round takes off what a
//! ball of radius `r` rolling inside the rod cannot reach: a point is taken
//! when it lies further than `r` from every centre the ball can have, the
//! points at least `r` inside both the wall and the cap. That set is convex,
//! so the distance to it is read from the nearest point on the wall's or the
//! cap's offset or on the ellipse where they meet, and the volume taken off
//! is integrated over the rod's rim, in cylindrical coordinates about its
//! axis.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::TAU;

    use brepkit_math::mat::Mat4;
    use brepkit_math::quadrature::gauss_legendre_points;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::fillet_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{EdgeCurve, EdgeId};
    use brepkit_topology::explorer::solid_edges;
    use brepkit_topology::solid::SolidId;

    /// The rod's radius, and the cap plane's lean about `x`.
    const ROD: f64 = 2.0;
    const TILT: f64 = 0.3;

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(-17.0, 230.0, 8.0)
                    * Mat4::rotation_y(1.1)
                    * Mat4::rotation_z(0.4),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    /// The rod cut by the slanted plane, moved by `pose`, and its rim.
    fn slanted_rod(topo: &mut Topology, pose: &Mat4) -> (SolidId, EdgeId) {
        let rod = make_cylinder(topo, ROD, 5.0).unwrap();
        let cutter = make_box(topo, 10.0, 10.0, 10.0).unwrap();
        let tilt = Mat4::translation(0.0, 0.0, 4.0)
            * Mat4::rotation_x(TILT)
            * Mat4::translation(-5.0, -5.0, 0.0);
        transform_solid(topo, cutter, &tilt).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, rod, cutter).unwrap();
        transform_solid(topo, cut, pose).unwrap();
        let rim = solid_edges(topo, cut)
            .unwrap()
            .into_iter()
            .find(|&e| matches!(topo.edge(e).unwrap().curve(), EdgeCurve::Ellipse(_)))
            .unwrap();
        (cut, rim)
    }

    /// How far `p` lies from the centres a ball of radius `r` can have in
    /// the rod: within `ROD - r` of the axis, and `r` under the cap.
    fn from_centres(p: Point3, r: f64) -> f64 {
        let a = ROD - r;
        let n = Vec3::new(0.0, -TILT.sin(), TILT.cos());
        let level = n.dot(Vec3::new(0.0, 0.0, 4.0)) - r;
        let under = |q: Point3| n.dot(Vec3::new(q.x(), q.y(), q.z())) <= level + 1e-12;
        let radial = p.x().hypot(p.y());
        let inside = radial <= a;
        if inside && under(p) {
            return 0.0;
        }
        let mut best = f64::INFINITY;
        let on_wall = if inside {
            p
        } else {
            Point3::new(p.x() * a / radial, p.y() * a / radial, p.z())
        };
        if under(on_wall) {
            best = best.min((p - on_wall).length());
        }
        let off = n.dot(Vec3::new(p.x(), p.y(), p.z())) - level;
        let on_cap = p - n * off.max(0.0);
        if on_cap.x().hypot(on_cap.y()) <= a + 1e-12 {
            best = best.min((p - on_cap).length());
        }
        // The ellipse where the two offsets meet.
        let at = |t: f64| {
            let (x, y) = (a * t.cos(), a * t.sin());
            Point3::new(x, y, (level + TILT.sin() * y) / TILT.cos())
        };
        let gap = |t: f64| (p - at(t)).length();
        let steps = 96;
        let mut t_best = 0.0;
        let mut d_best = f64::INFINITY;
        for k in 0..steps {
            let t = TAU * f64::from(k) / f64::from(steps);
            if gap(t) < d_best {
                d_best = gap(t);
                t_best = t;
            }
        }
        let (mut lo, mut hi) = (
            t_best - TAU / f64::from(steps),
            t_best + TAU / f64::from(steps),
        );
        for _ in 0..50 {
            let m1 = lo + (hi - lo) / 3.0;
            let m2 = hi - (hi - lo) / 3.0;
            if gap(m1) < gap(m2) {
                hi = m2;
            } else {
                lo = m1;
            }
        }
        best.min(gap(0.5 * (lo + hi)))
    }

    /// The point at radius `rho` and angle `phi` about the axis, at height `z`.
    fn cylindrical(rho: f64, phi: f64, z: f64) -> Point3 {
        Point3::new(rho * phi.cos(), rho * phi.sin(), z)
    }

    /// The cap's height over the axis at radius `rho` and angle `phi`.
    fn cap(rho: f64, phi: f64) -> f64 {
        4.0 + rho * phi.sin() * TILT.tan()
    }

    /// How deep under the cap the round takes material, at radius `rho` and
    /// angle `phi`.
    fn taken(rho: f64, phi: f64, r: f64) -> f64 {
        let top = cap(rho, phi);
        if from_centres(cylindrical(rho, phi, top), r) <= r {
            return 0.0;
        }
        let (mut lo, mut hi) = (top - 3.0 * r, top);
        for _ in 0..50 {
            let mid = 0.5 * (lo + hi);
            if from_centres(cylindrical(rho, phi, mid), r) > r {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        top - 0.5 * (lo + hi)
    }

    /// The volume the round of radius `r` takes off.
    fn removed(r: f64) -> f64 {
        let turns = 128;
        let mut total = 0.0;
        for k in 0..turns {
            let phi = TAU * f64::from(k) / f64::from(turns);
            // The radius where the round leaves the cap.
            let (mut lo, mut hi) = (ROD - 3.0 * r, ROD);
            for _ in 0..50 {
                let mid = 0.5 * (lo + hi);
                if from_centres(cylindrical(mid, phi, cap(mid, phi)), r) > r {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let width = ROD - 0.5 * (lo + hi);
            // `rho = ROD - width u^2` takes the square root the band leaves
            // the wall with out of the integrand.
            let ring: f64 = gauss_legendre_points(20)
                .iter()
                .map(|g| {
                    let u = 0.5 * (g.x + 1.0);
                    let rho = ROD - width * u * u;
                    0.5 * g.w * taken(rho, phi, r) * rho * 2.0 * width * u
                })
                .sum();
            total += ring * TAU / f64::from(turns);
        }
        total
    }

    #[test]
    fn a_round_of_a_slanted_rods_rim_is_sound_in_every_pose() {
        for r in [0.3, 0.5] {
            let want = removed(r);
            for (name, pose) in poses() {
                let case = format!("{name}, r {r}");
                let mut topo = Topology::new();
                let (solid, rim) = slanted_rod(&mut topo, &pose);
                let before = solid_volume(&topo, solid, 1e-4).unwrap();
                let result = fillet_v2(&mut topo, solid, &[rim], r).unwrap();
                assert!(result.failed.is_empty(), "{case}: {:?}", result.failed);
                let report = validate_solid(&topo, result.solid).unwrap();
                assert!(report.is_valid(), "{case}: {:?}", report.issues);
                for deflection in [0.1, 0.01] {
                    let mesh = tessellate_solid(&topo, result.solid, deflection).unwrap();
                    assert!(is_watertight(&mesh), "{case}: open mesh at {deflection}");
                }
                // The band is walked, so it holds the ball's envelope to its
                // fit: it takes off 4e-5 less than the integral at r = 0.3.
                let took = before - solid_volume(&topo, result.solid, 1e-4).unwrap();
                assert!(
                    (took - want).abs() <= 1e-4 * want,
                    "{case}: took off {took}, want {want}"
                );
            }
        }
    }
}

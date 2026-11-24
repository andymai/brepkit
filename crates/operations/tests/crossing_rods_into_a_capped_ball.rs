//! A ball less the cap above `z = 0.95`, and two rods into it, their axes
//! through its centre and 0.3 rad apart, fused at once. Each rod cuts the
//! ball's face in a small circle clear of both poles, and the two circles
//! cross twice. With the cap and a rod turned a quarter turn about `y`,
//! the cut lies across `x`; at `x = 0.95` the ball's one face holds both
//! poles of its own chart, and at `x = 0` it runs from pole to pole.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder, make_sphere};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    /// Where the cap's plane cuts the ball, before the scene is turned.
    const CAP_PLANE: f64 = 0.95;
    const ROD_RADIUS: f64 = 0.2;
    /// The angle between the rods' axes.
    const APART: f64 = 0.3;
    /// Each rod runs along its axis from this far from the ball's centre...
    const ROD_START: f64 = 0.5;
    /// ...to this far.
    const ROD_END: f64 = 1.5;

    /// How each rod lies, before the scene is turned: its axis turned from
    /// `-z` about `y`, half the rods' angle either side of the direction
    /// halfway between `+x` and `-z`.
    fn rod_place(first: bool) -> Mat4 {
        let half = if first { 0.5 * APART } else { -0.5 * APART };
        Mat4::rotation_y(-FRAC_PI_4 - half)
            * Mat4::rotation_x(PI)
            * Mat4::translation(0.0, 0.0, ROD_START)
    }

    fn rod_axis(first: bool) -> Vec3 {
        let on_axis = rod_place(first).mul_point(Point3::new(0.0, 0.0, 1.0));
        (on_axis - Point3::new(0.0, 0.0, 0.0)).normalize().unwrap()
    }

    /// The volume of a rod outside the unit ball.
    fn rod_outside() -> f64 {
        let r2 = ROD_RADIUS * ROD_RADIUS;
        PI * r2 * ROD_END - 2.0 * PI / 3.0 * (1.0 - (1.0 - r2).powf(1.5))
    }

    /// The volume of both rods outside the unit ball. Each line along the
    /// first rod's axis is cut exactly by the second rod and the ball; the
    /// lengths are summed by the midpoint rule over the first rod's disc.
    fn both_outside() -> f64 {
        let (a, b) = (rod_axis(true), rod_axis(false));
        let e1 = a.cross(Vec3::new(0.0, 1.0, 0.0)).normalize().unwrap();
        let e2 = a.cross(e1);
        let c = a.dot(b);
        let n = 1500;
        let (dr, dphi) = (ROD_RADIUS / f64::from(n), 2.0 * PI / f64::from(n));
        let mut sum = 0.0;
        for i in 0..n {
            let rho = (f64::from(i) + 0.5) * dr;
            let out_of_ball = (1.0 - rho * rho).sqrt();
            for j in 0..n {
                let phi = (f64::from(j) + 0.5) * dphi;
                let q = e1 * (rho * phi.cos()) + e2 * (rho * phi.sin());
                let qb = q.dot(b);
                // Inside the second rod: (1 - c^2) s^2 - 2 qb c s + |q|^2 -
                // qb^2 - r^2 <= 0, and its own run along b.
                let (k2, k1) = (1.0 - c * c, -2.0 * qb * c);
                let k0 = rho * rho - qb * qb - ROD_RADIUS * ROD_RADIUS;
                let disc = k1 * k1 - 4.0 * k2 * k0;
                if disc <= 0.0 {
                    continue;
                }
                let root = disc.sqrt();
                let lo = ((-k1 - root) / (2.0 * k2))
                    .max((ROD_START - qb) / c)
                    .max(out_of_ball);
                let hi = ((-k1 + root) / (2.0 * k2))
                    .min((ROD_END - qb) / c)
                    .min(ROD_END);
                if hi > lo {
                    sum += (hi - lo) * rho;
                }
            }
        }
        sum * dr * dphi
    }

    /// The volume of the capped ball and its first `rods` rods.
    fn expected(cap: f64, rods: usize) -> f64 {
        let h = 1.0 - cap;
        let capped = 4.0 / 3.0 * PI - PI * h * h * (3.0 - h) / 3.0;
        match rods {
            1 => capped + rod_outside(),
            _ => capped + 2.0 * rod_outside() - both_outside(),
        }
    }

    /// The ball less the cap above `z = cap` and its first `rods` rods, the
    /// cap and the rods turned by `around` about the ball's centre, then all
    /// of it placed by `pose`.
    fn scene(
        topo: &mut Topology,
        cap: f64,
        rods: usize,
        around: &Mat4,
        pose: &Mat4,
    ) -> Vec<SolidId> {
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let lid = make_box(topo, 4.0, 4.0, 2.0).unwrap();
        transform_solid(topo, lid, &Mat4::translation(-2.0, -2.0, cap)).unwrap();
        let rods: Vec<SolidId> = [true, false][..rods]
            .iter()
            .map(|&first| {
                let rod = make_cylinder(topo, ROD_RADIUS, ROD_END - ROD_START).unwrap();
                transform_solid(topo, rod, &rod_place(first)).unwrap();
                rod
            })
            .collect();
        for &s in std::iter::once(&lid).chain(&rods) {
            transform_solid(topo, s, around).unwrap();
        }
        for &s in [ball, lid].iter().chain(&rods) {
            transform_solid(topo, s, pose).unwrap();
        }
        let capped = boolean_with_options(topo, BooleanOp::Cut, ball, lid, opts).unwrap();
        std::iter::once(capped).chain(rods).collect()
    }

    /// Why `solid` is not the exact fuse of the ball less the cap above `z
    /// = cap` and its first `rods` rods, its points placed by `around` then
    /// `pose`.
    fn fault(
        topo: &Topology,
        solid: SolidId,
        (cap, rods): (f64, usize),
        around: &Mat4,
        pose: &Mat4,
    ) -> Option<String> {
        let report = validate_solid(topo, solid).unwrap();
        if !report.is_valid() {
            return Some(format!("invalid: {:?}", report.issues));
        }
        let mut census: BTreeMap<&str, usize> = BTreeMap::new();
        for f in solid_faces(topo, solid).unwrap() {
            *census
                .entry(topo.face(f).unwrap().surface().type_tag())
                .or_default() += 1;
        }
        if !census.contains_key("sphere") || census.values().sum::<usize>() > 20 {
            return Some(format!("faces {census:?}"));
        }
        let mesh = tessellate_solid(topo, solid, 0.002).unwrap();
        if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
            return Some("mesh not closed".into());
        }
        let volume = solid_volume(topo, solid, 0.0005).unwrap();
        let truth = expected(cap, rods);
        if (volume - truth).abs() >= 1e-6 * truth {
            return Some(format!("volume {volume} should be {truth}"));
        }
        let origin = Point3::new(0.0, 0.0, 0.0);
        let (a, b) = (rod_axis(true), rod_axis(false));
        let middle = (a + b).normalize().unwrap();
        // In the ball and in the first rod; above the cap and past the first
        // rod's end. With both rods: in the second rod, and in both rods
        // outside the ball; between the rods. With one rod: where the second
        // would be.
        let mut inside = vec![Point3::new(0.0, 0.0, cap - 0.3), origin + a * 1.3];
        let mut outside = vec![Point3::new(0.0, 0.1, 0.5 * (1.0 + cap)), origin + a * 1.6];
        if rods == 2 {
            inside.extend([origin + b * 1.3, origin + middle * 1.05]);
            outside.push(origin + middle * 1.45);
        } else {
            outside.push(origin + b * 1.3);
        }
        for (points, want) in [
            (inside, PointClassification::Inside),
            (outside, PointClassification::Outside),
        ] {
            for p in points {
                let q = pose.mul_point(around.mul_point(p));
                let got = classify_point(topo, solid, q, 0.001, 1e-7).unwrap();
                if got != want {
                    return Some(format!("{p:?} reads {got:?}, not {want:?}"));
                }
            }
        }
        None
    }

    fn faults(cap: f64, rods: usize, around: &Mat4) -> Vec<String> {
        let mut faults = Vec::new();
        for (name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.37) * Mat4::rotation_x(0.21)),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ] {
            let mut topo = Topology::new();
            let solids = scene(&mut topo, cap, rods, around, &pose);
            let why = match brepkit_algo::gfa::fuse_n(&mut topo, &solids) {
                Ok(solid) => fault(&topo, solid, (cap, rods), around, &pose),
                Err(e) => Some(format!("error {e}")),
            };
            if let Some(why) = why {
                faults.push(format!("{name}: {why}"));
            }
        }
        faults
    }

    #[test]
    fn two_rods_whose_circles_cross_on_a_capped_ball_fuse_at_once() {
        let faults = faults(CAP_PLANE, 2, &Mat4::identity());
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }

    /// With the cap and the rod turned a quarter turn about `y`, the cut
    /// lies at `x = 0.95`, and the ball's one face holds both poles of its
    /// own chart.
    #[test]
    #[ignore = "open: the ball's face takes no section from the rod, and the fuse needs the \
                mesh fallback; with the cap above, the same rod fuses exactly"]
    fn a_rod_into_a_ball_less_a_cap_turned_to_x_fuses() {
        let faults = faults(CAP_PLANE, 1, &Mat4::rotation_y(FRAC_PI_2));
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }

    /// The cut turned to `x = 0`: the ball's half face runs from pole to
    /// pole of its own chart.
    #[test]
    fn a_rod_into_a_ball_halved_at_x_0_fuses() {
        let faults = faults(0.0, 1, &Mat4::rotation_y(FRAC_PI_2));
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

//! A box across the tube of a torus of radii 4 and 1.5, intersected with
//! it. Each box wall meets the tube in a loop that no circle or conic
//! traces, so the loop is a cubic fit through points of the true section.
//! A fixed sampling left such a fit up to 8e-5 off the torus where the loop
//! turns tightly (a wall near the inner equator), and a box wall's arc was
//! refitted through points of that fit, not of the section. The results were
//! exact, valid and watertight, but their volumes missed an integral by
//! 1.3e-6 to 1.5e-5 of it, at every deflection.
//!
//! Each box has its centre at `c` and the half-size `h`. Its part of the
//! torus is integrated over the distance from the axis: at each distance the
//! turn of the circle inside the box's outline times the tube's height
//! inside the box's floor and top.
#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use std::f64::consts::{PI, TAU};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_torus};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::face::FaceSurface;

    const MAJOR: f64 = 4.0;
    const MINOR: f64 = 1.5;

    /// Tanh-sinh quadrature over `[a, b]`, exact to rounding for a smooth
    /// integrand with square-root ends.
    fn tanh_sinh(f: &dyn Fn(f64) -> f64, a: f64, b: f64) -> f64 {
        let h = 1.0 / 64.0;
        let half = 0.5 * (b - a);
        let mut sum = 0.0;
        for k in -240..=240 {
            let t = f64::from(k) * h;
            let s = 0.5 * PI * t.sinh();
            let weight = 0.5 * PI * t.cosh() / (s.cosh() * s.cosh());
            // The distance from the nearer end, without cancellation.
            let gap = half / (s.abs().exp() * s.cosh());
            let x = if s >= 0.0 { b - gap } else { a + gap };
            if x > a && x < b {
                sum += weight * f(x);
            }
        }
        sum * h * half
    }

    /// The turn of the circle of radius `rho` about the axis inside the
    /// rectangle from `lo` to `hi`.
    fn turn_in_box(rho: f64, lo: [f64; 3], hi: [f64; 3]) -> f64 {
        let mut angles = vec![0.0, TAU];
        for x in [lo[0], hi[0]] {
            if x.abs() < rho {
                let a = (x / rho).acos();
                angles.extend([a, TAU - a]);
            }
        }
        for y in [lo[1], hi[1]] {
            if y.abs() < rho {
                let a = (y / rho).asin();
                angles.extend([a.rem_euclid(TAU), (PI - a).rem_euclid(TAU)]);
            }
        }
        angles.sort_by(f64::total_cmp);
        angles
            .windows(2)
            .filter(|w| {
                let a = f64::midpoint(w[0], w[1]);
                let (x, y) = (rho * a.cos(), rho * a.sin());
                x > lo[0] && x < hi[0] && y > lo[1] && y < hi[1]
            })
            .map(|w| w[1] - w[0])
            .sum()
    }

    /// The torus's part inside the box, integrated over the distance from
    /// the axis between the distances where the integrand has a kink.
    fn volume_in_box(c: [f64; 3], h: f64) -> f64 {
        let lo = [c[0] - h, c[1] - h, c[2] - h];
        let hi = [c[0] + h, c[1] + h, c[2] + h];
        let mut breaks = vec![MAJOR - MINOR, MAJOR + MINOR];
        for x in [lo[0], hi[0]] {
            breaks.push(x.abs());
            for y in [lo[1], hi[1]] {
                breaks.extend([y.abs(), x.hypot(y)]);
            }
        }
        for z in [lo[2], hi[2]] {
            if z.abs() < MINOR {
                let w = MINOR.mul_add(MINOR, -z * z).sqrt();
                breaks.extend([MAJOR - w, MAJOR + w]);
            }
        }
        breaks.retain(|r| (MAJOR - MINOR..=MAJOR + MINOR).contains(r));
        breaks.sort_by(f64::total_cmp);
        breaks.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
        let slice = |rho: f64| {
            let off = rho - MAJOR;
            let reach = MINOR.mul_add(MINOR, -off * off);
            if reach <= 0.0 {
                return 0.0;
            }
            let half = reach.sqrt();
            let height = (hi[2].min(half) - lo[2].max(-half)).max(0.0);
            rho * turn_in_box(rho, lo, hi) * height
        };
        breaks
            .windows(2)
            .map(|w| tanh_sinh(&slice, w[0], w[1]))
            .sum()
    }

    fn poses() -> [(&'static str, Mat4); 3] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(-3.7, 5.5, 14.0) * Mat4::rotation_z(0.7) * Mat4::rotation_x(-0.9),
            ),
            (
                "mirrored",
                Mat4::scale(-1.0, 1.0, 1.0) * Mat4::rotation_y(0.4),
            ),
        ]
    }

    /// How far a point lies from the torus, in its own frame.
    fn off_torus(p: Point3) -> f64 {
        ((p.x().hypot(p.y()) - MAJOR).hypot(p.z()) - MINOR).abs()
    }

    fn check(c: [f64; 3], h: f64, rel: f64) {
        let want = volume_in_box(c, h);
        for (name, pose) in poses() {
            let label = format!("box at {c:?}, half-size {h}, {name}");
            let mut topo = Topology::new();
            let ring = make_torus(&mut topo, MAJOR, MINOR, 32).unwrap();
            let block = make_box(&mut topo, 2.0 * h, 2.0 * h, 2.0 * h).unwrap();
            transform_solid(
                &mut topo,
                block,
                &Mat4::translation(c[0] - h, c[1] - h, c[2] - h),
            )
            .unwrap();
            for solid in [ring, block] {
                transform_solid(&mut topo, solid, &pose).unwrap();
            }
            let exact = BooleanOptions {
                mesh_fallback: false,
                ..BooleanOptions::default()
            };
            let piece = boolean_with_options(&mut topo, BooleanOp::Intersect, ring, block, exact)
                .unwrap_or_else(|e| panic!("{label}: {e}"));

            let report = validate_solid(&topo, piece).unwrap();
            assert!(report.is_valid(), "{label}: invalid: {report:?}");
            assert!(
                is_watertight(&tessellate_solid(&topo, piece, 0.01).unwrap()),
                "{label}: mesh not watertight"
            );

            // The torus face's edges lie on the torus.
            let back = pose.inverse().unwrap();
            for face in solid_faces(&topo, piece).unwrap() {
                let face = topo.face(face).unwrap();
                if !matches!(face.surface(), FaceSurface::Torus(_)) {
                    continue;
                }
                for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().to_vec()) {
                    for used in topo.wire(wire).unwrap().edges() {
                        let edge = topo.edge(used.edge()).unwrap();
                        if !matches!(edge.curve(), EdgeCurve::NurbsCurve(_)) {
                            continue;
                        }
                        let a = topo.vertex(edge.start()).unwrap().point();
                        let b = topo.vertex(edge.end()).unwrap().point();
                        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                        for k in 0..=400 {
                            let t = (t1 - t0).mul_add(f64::from(k) / 400.0, t0);
                            let p = back.mul_point(edge.curve().evaluate_with_endpoints(t, a, b));
                            assert!(
                                off_torus(p) <= 1e-8,
                                "{label}: an edge strays {} off the torus at {p:?}",
                                off_torus(p)
                            );
                        }
                    }
                }
            }

            let volume = solid_volume(&topo, piece, 0.001).unwrap();
            assert!(
                (volume - want).abs() <= rel * want,
                "{label}: volume {volume}, want {want}, off by {:.2e} of it",
                (volume - want) / want
            );
        }
    }

    /// The box whose walls cut the ring's top: 1.5e-5 off when turned.
    #[test]
    fn a_box_over_the_tubes_top() {
        check([3.0, 5.2, 1.1], 0.6, 3e-8);
    }

    /// A box through the tube's middle: 6.3e-6 off when turned.
    #[test]
    fn a_small_box_through_the_tube() {
        check([3.0, 5.2, 0.37], 0.6, 3e-8);
    }

    /// A box whose wall `y = 2.1` runs 0.4 outside the inner equator, where
    /// its loop turns tightly: 1.3e-6 off when turned.
    #[test]
    fn a_wall_beside_the_inner_equator() {
        check([3.0, 4.1, -0.4], 2.0, 1e-9);
    }

    /// A box across the ring's hole whose walls `x = +-1.9` and `y = 1.05`
    /// loop round the tube: 2.0e-6 off when turned.
    #[test]
    fn walls_round_the_tube_across_the_hole() {
        check([0.1, 3.05, -0.4], 2.0, 1e-9);
    }
}

//! A taper pin's hole through a rod: a cone along y cut from a rod along z,
//! its two loops on the rod's wall half a turn apart. Read in one window of
//! `u` about one loop, the other loop spanned the window and the first read
//! as inside it.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI};

    use brepkit_check::classify::{ClassifyOptions, PointClassification, classify_point};
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{self, BooleanOp, BooleanOptions};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_cone, make_cylinder};
    use brepkit_operations::tessellate::{boundary_edge_count, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;

    /// The pin's radius at `y`, its wide end at `y = 8` when `flip`.
    fn pin_radius(y: f64, flip: bool) -> f64 {
        let along = if flip { 8.0 - y } else { y + 8.0 };
        (-0.7 / 16.0f64).mul_add(along, 1.5)
    }

    /// The volume of the pin inside the rod: at each `y` the pin's disc of
    /// radius `r` about the line `x = 0, z = 10`, less the parts past the
    /// rod's wall at `|x| = w = sqrt(25 - y^2)`, two circular segments. The
    /// segments start where `w = r`, at each hole's branch points, so the
    /// integral is split there; with `y = 5 sin a` its ends are smooth.
    fn hole_volume(flip: bool) -> f64 {
        let area = |y: f64| {
            let r = pin_radius(y, flip);
            let w = (y.mul_add(-y, 25.0)).max(0.0).sqrt();
            if w >= r {
                PI * r * r
            } else {
                let segment = (r * r).mul_add((w / r).acos(), -w * r.mul_add(r, -w * w).sqrt());
                (PI * r).mul_add(r, -2.0 * segment)
            }
        };
        let gap = |y: f64| (y.mul_add(-y, 25.0)).sqrt() - pin_radius(y, flip);
        let branch = |(mut lo, mut hi): (f64, f64)| {
            for _ in 0..200 {
                let mid = 0.5 * (lo + hi);
                if (gap(mid) > 0.0) == (gap(lo) > 0.0) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (0.5 * (lo + hi) / 5.0).asin()
        };
        let cuts = [
            -FRAC_PI_2,
            branch((-4.0, -5.0)),
            branch((4.0, 5.0)),
            FRAC_PI_2,
        ];
        // Five-point Gauss-Legendre on each of many panels.
        let nodes = [
            (0.0, 128.0 / 225.0),
            (0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
            (-0.538_469_310_105_683_1, 0.478_628_670_499_366_5),
            (0.906_179_845_938_664, 0.236_926_885_056_189_1),
            (-0.906_179_845_938_664, 0.236_926_885_056_189_1),
        ];
        let panels = 2000;
        cuts.windows(2)
            .map(|span| {
                let h = (span[1] - span[0]) / f64::from(panels);
                (0..panels)
                    .map(|i| {
                        let mid = h.mul_add(f64::from(i) + 0.5, span[0]);
                        nodes
                            .iter()
                            .map(|&(x, w)| {
                                let a = (0.5 * h).mul_add(x, mid);
                                w * area(5.0 * a.sin()) * 5.0 * a.cos()
                            })
                            .sum::<f64>()
                            * 0.5
                            * h
                    })
                    .sum::<f64>()
            })
            .sum()
    }

    /// A rod of radius 5 and height 20 turned `spin` about its axis, less a
    /// cone of radii 1.5 and 0.8 along y through it at z = 10, its wide end
    /// at y = 8 when `flip`, else at y = -8: exact, 4 faces, closed, the
    /// hole's points outside, and its volume the rod's less the pin's part
    /// inside it.
    fn cut(spin: f64, flip: bool) {
        let mut topo = Topology::new();
        let rod = make_cylinder(&mut topo, 5.0, 20.0).unwrap();
        transform_solid(&mut topo, rod, &Mat4::rotation_z(spin)).unwrap();
        let pin = make_cone(&mut topo, 1.5, 0.8, 16.0).unwrap();
        let (lay, start) = if flip {
            (Mat4::rotation_x(PI / 2.0), 8.0)
        } else {
            (Mat4::rotation_x(-PI / 2.0), -8.0)
        };
        transform_solid(&mut topo, pin, &(Mat4::translation(0.0, start, 10.0) * lay)).unwrap();
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let name = format!("spin {spin} flip {flip}");
        let result = boolean::boolean_with_options(&mut topo, BooleanOp::Cut, rod, pin, opts);
        assert!(result.is_ok(), "{name}: {result:?}");
        let result = result.unwrap();
        let report = validate_solid(&topo, result).unwrap();
        assert!(report.is_valid(), "{name}: {:?}", report.issues);
        let faces = brepkit_topology::explorer::solid_faces(&topo, result).unwrap();
        assert_eq!(faces.len(), 4, "{name}: faces");
        let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
        assert_eq!(boundary_edge_count(&mesh), 0, "{name}: open mesh");
        let expected = (PI * 25.0).mul_add(20.0, -hole_volume(flip));
        let v = solid_volume(&topo, result, 0.0005).unwrap();
        assert!(
            (v - expected).abs() < 2e-11 * expected,
            "{name}: volume {v}, expected {expected}"
        );
        let o = ClassifyOptions::default();
        for p in [
            Point3::new(0.0, 4.9, 10.0),
            Point3::new(0.0, -4.9, 10.0),
            Point3::new(0.0, 0.0, 10.0),
        ] {
            assert_eq!(
                classify_point(&topo, result, p, &o).unwrap(),
                PointClassification::Outside,
                "{name}: {p:?} lies in the hole"
            );
        }
        for p in [Point3::new(3.0, 0.0, 10.0), Point3::new(0.0, 4.0, 12.0)] {
            assert_eq!(
                classify_point(&topo, result, p, &o).unwrap(),
                PointClassification::Inside,
                "{name}: {p:?} lies in the rod"
            );
        }
    }

    /// In some turns the pin's narrow end meets fewer than 8 of the 128
    /// rulings sampled round the rod, and the cone's seam crosses a loop
    /// exactly at one of the samples read along it.
    #[test]
    fn a_taper_pin_cuts_both_of_its_holes_in_a_rod() {
        for spin in [0.0, 0.3, 0.9, 1.4, 2.0, 2.6, 3.0, 3.6] {
            for flip in [false, true] {
                cut(spin, flip);
            }
        }
    }
}

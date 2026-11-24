//! A ball less the cap above `z = 0.95`, and two boxes under it, the second
//! turned an eighth of a turn about the first's upright middle, fused at
//! once. Each box's sides cut the ball's face in a loop of four arcs clear
//! of both poles, and the two loops cross at eight points.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::collections::BTreeMap;
    use std::f64::consts::FRAC_PI_4;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::Point3;
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_sphere};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const CAP_PLANE: f64 = 0.95;
    /// The boxes' upright middle, and half their width.
    const MIDDLE: (f64, f64) = (0.35, 0.0);
    const HALF_WIDTH: f64 = 0.1;
    /// Each box's bottom and top.
    const FIRST_Z: (f64, f64) = (-1.2, -0.85);
    const SECOND_Z: (f64, f64) = (-1.25, -0.84);

    type Polygon = Vec<(f64, f64)>;

    /// Each box's square seen from above, counter-clockwise.
    fn squares() -> [Polygon; 2] {
        let (cx, cy) = MIDDLE;
        let w = HALF_WIDTH;
        let d = w * std::f64::consts::SQRT_2;
        [
            vec![
                (cx - w, cy - w),
                (cx + w, cy - w),
                (cx + w, cy + w),
                (cx - w, cy + w),
            ],
            vec![(cx + d, cy), (cx, cy + d), (cx - d, cy), (cx, cy - d)],
        ]
    }

    /// The part of the convex polygon `subject` inside the convex polygon
    /// `clip`, both counter-clockwise.
    fn clip(subject: &[(f64, f64)], clip: &[(f64, f64)]) -> Polygon {
        let mut out: Polygon = subject.to_vec();
        for k in 0..clip.len() {
            let (a, b) = (clip[k], clip[(k + 1) % clip.len()]);
            let side = |p: (f64, f64)| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
            let input = std::mem::take(&mut out);
            for j in 0..input.len() {
                let (p, q) = (input[j], input[(j + 1) % input.len()]);
                let (sp, sq) = (side(p), side(q));
                if sp >= 0.0 {
                    out.push(p);
                }
                if (sp >= 0.0) != (sq >= 0.0) {
                    let t = sp / (sp - sq);
                    out.push((p.0 + t * (q.0 - p.0), p.1 + t * (q.1 - p.1)));
                }
            }
        }
        out
    }

    /// The integral of `f` over a convex polygon: a fan of triangles from
    /// its first corner, each by a tensor Gauss rule collapsed onto it.
    fn integrate(poly: &[(f64, f64)], f: &dyn Fn(f64, f64) -> f64) -> f64 {
        let gauss = brepkit_math::quadrature::gauss_legendre_points(20);
        let a = poly[0];
        let mut total = 0.0;
        for k in 1..poly.len() - 1 {
            let (b, c) = (poly[k], poly[k + 1]);
            let twice = ((b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)).abs();
            for gu in gauss {
                let u = 0.5 * (gu.x + 1.0);
                for gv in gauss {
                    let v = 0.5 * (gv.x + 1.0);
                    let x = a.0 + u * (b.0 - a.0) + u * v * (c.0 - b.0);
                    let y = a.1 + u * (b.1 - a.1) + u * v * (c.1 - b.1);
                    total += 0.25 * gu.w * gv.w * u * twice * f(x, y);
                }
            }
        }
        total
    }

    fn expected() -> f64 {
        let [first, second] = squares();
        let both = clip(&second, &first);
        let area = |p: &[(f64, f64)]| integrate(p, &|_, _| 1.0);
        let below = |top: f64| move |x: f64, y: f64| (1.0 - x * x - y * y).sqrt() + top;
        // The ball under each box's top, over the box's square; where the
        // squares overlap, under the higher top, the second's.
        let in_ball = integrate(&first, &below(FIRST_Z.1)) - integrate(&both, &below(FIRST_Z.1))
            + integrate(&second, &below(SECOND_Z.1));
        let boxes = area(&first) * (FIRST_Z.1 - FIRST_Z.0)
            + area(&second) * (SECOND_Z.1 - SECOND_Z.0)
            - area(&both) * (FIRST_Z.1 - FIRST_Z.0);
        let h = 1.0 - CAP_PLANE;
        let capped =
            4.0 / 3.0 * std::f64::consts::PI - std::f64::consts::PI * h * h * (3.0 - h) / 3.0;
        capped + boxes - in_ball
    }

    fn scene(topo: &mut Topology, pose: &Mat4) -> [SolidId; 3] {
        let opts = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let ball = make_sphere(topo, 1.0, 32).unwrap();
        let lid = make_box(topo, 4.0, 4.0, 2.0).unwrap();
        transform_solid(topo, lid, &Mat4::translation(-2.0, -2.0, CAP_PLANE)).unwrap();
        let (cx, cy) = MIDDLE;
        let w = 2.0 * HALF_WIDTH;
        let first = make_box(topo, w, w, FIRST_Z.1 - FIRST_Z.0).unwrap();
        transform_solid(
            topo,
            first,
            &Mat4::translation(cx - HALF_WIDTH, cy - HALF_WIDTH, FIRST_Z.0),
        )
        .unwrap();
        let second = make_box(topo, w, w, SECOND_Z.1 - SECOND_Z.0).unwrap();
        transform_solid(
            topo,
            second,
            &(Mat4::translation(cx, cy, SECOND_Z.0)
                * Mat4::rotation_z(FRAC_PI_4)
                * Mat4::translation(-HALF_WIDTH, -HALF_WIDTH, 0.0)),
        )
        .unwrap();
        for s in [ball, lid, first, second] {
            transform_solid(topo, s, pose).unwrap();
        }
        let capped = boolean_with_options(topo, BooleanOp::Cut, ball, lid, opts).unwrap();
        [capped, first, second]
    }

    fn fault(topo: &Topology, solid: SolidId, pose: &Mat4) -> Option<String> {
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
        if !census.contains_key("sphere") || census.values().sum::<usize>() > 30 {
            return Some(format!("faces {census:?}"));
        }
        let mesh = tessellate_solid(topo, solid, 0.002).unwrap();
        if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
            return Some("mesh not closed".into());
        }
        let volume = solid_volume(topo, solid, 0.0005).unwrap();
        let truth = expected();
        if (volume - truth).abs() >= 1e-6 * truth {
            return Some(format!("volume {volume} should be {truth}"));
        }
        // In the ball, in the first box only and in the second only below
        // the ball; above the cap, and beside both boxes below the ball.
        let inside = [[0.0, 0.0, 0.0], [0.35, 0.0, -1.0], [0.46, 0.0, -1.22]];
        let outside = [[0.0, 0.1, 0.97], [0.26, 0.09, -1.22], [0.6, 0.0, -1.0]];
        for (points, want) in [
            (inside, PointClassification::Inside),
            (outside, PointClassification::Outside),
        ] {
            for p in points {
                let q = pose.mul_point(Point3::new(p[0], p[1], p[2]));
                let got = classify_point(topo, solid, q, 0.001, 1e-7).unwrap();
                if got != want {
                    return Some(format!("{p:?} reads {got:?}, not {want:?}"));
                }
            }
        }
        None
    }

    #[test]
    #[ignore = "open: the ball's face splits in ten, read about its pole, but holes cut in \
                the second box's sides where the first box crosses them have 12 free edges; \
                volume 4.195254 against 4.195609; fusing the boxes first is exact"]
    fn two_crossing_boxes_under_a_capped_ball_fuse_at_once() {
        let mut faults = Vec::new();
        for (name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.37) * Mat4::rotation_x(0.21)),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ] {
            let mut topo = Topology::new();
            let solids = scene(&mut topo, &pose);
            let why = match brepkit_algo::gfa::fuse_n(&mut topo, &solids) {
                Ok(solid) => fault(&topo, solid, &pose),
                Err(e) => Some(format!("error {e}")),
            };
            if let Some(why) = why {
                faults.push(format!("{name}: {why}"));
            }
        }
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

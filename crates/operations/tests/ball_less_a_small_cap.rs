//! A ball less a small cap: its spherical face is bounded by the cap's small
//! circle alone, and holds nearly the whole sphere. Rods and edges that cross
//! that face far from the cap must find it there.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, PI, TAU};

    use brepkit_math::curves::Circle3D;
    use brepkit_math::madd::Madd;
    use brepkit_math::mat::Mat4;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::{Solid, SolidId};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// The height of the plane that cuts the cap off a unit ball.
    const CAP: f64 = 0.95;

    /// A unit ball at the origin less the cap above `z = CAP`: a spherical
    /// face bounded by the cap's circle alone, and the flat disc of the cap.
    fn ball_less_a_cap(topo: &mut Topology) -> SolidId {
        let up = Vec3::new(0.0, 0.0, 1.0);
        let radius = CAP.madd(-CAP, 1.0).sqrt();
        let centre = Point3::new(0.0, 0.0, CAP);
        let rim = Circle3D::new_with_ref(centre, up, radius, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(radius, 0.0, CAP), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(rim)));
        // Each face runs the rim counterclockwise about its own outward
        // normal: the disc along the circle, the ball's face against it.
        let disc_wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let ball_wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, false)], true).unwrap());
        let disc = topo.add_face(Face::new(
            disc_wire,
            vec![],
            FaceSurface::Plane { normal: up, d: CAP },
        ));
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 1.0).unwrap();
        let ball = topo.add_face(Face::new(ball_wire, vec![], FaceSurface::Sphere(sphere)));
        let shell = topo.add_shell(Shell::new(vec![ball, disc]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    fn whole_volume() -> f64 {
        let h = 1.0 - CAP;
        4.0 / 3.0 * PI - PI * h * h * (3.0 - h) / 3.0
    }

    /// The volume of a unit ball inside a rod along y of radius `rho` round
    /// the line through `(x0, ., z0)`, the rod's section wholly inside the
    /// ball's: over the section, the chord of the ball along y, summed by
    /// Gauss points across the radius and evenly round the section.
    fn ball_held_by_a_rod(rho: f64, x0: f64, z0: f64) -> f64 {
        const ROUND: u32 = 512;
        let step = TAU / f64::from(ROUND);
        let mut sum = 0.0;
        for gp in brepkit_math::quadrature::gauss_legendre_points(20) {
            let r = 0.5 * rho * (gp.x + 1.0);
            let weight = 0.5 * rho * gp.w * r * step;
            for k in 0..ROUND {
                let a = step * f64::from(k);
                let (x, z) = (r.madd(a.cos(), x0), r.madd(a.sin(), z0));
                sum += weight * 2.0 * (1.0 - x * x - z * z).sqrt();
            }
        }
        sum
    }

    /// The poses each case is read in: as built, turned so the cap leans,
    /// and mirrored through a slanted plane.
    #[derive(Clone, Copy, Debug)]
    enum Pose {
        Upright,
        Turned,
        Mirrored,
    }

    fn place(topo: &mut Topology, solid: SolidId, pose: Pose) -> SolidId {
        match pose {
            Pose::Upright => solid,
            Pose::Turned => {
                let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(1.1) * Mat4::rotation_y(0.3);
                transform_solid(topo, solid, &turn).unwrap();
                solid
            }
            Pose::Mirrored => mirror(
                topo,
                solid,
                Point3::new(0.3, 0.0, 0.0),
                Vec3::new(1.0, 0.2, 0.1),
            )
            .unwrap(),
        }
    }

    fn map(pose: Pose, p: Point3) -> Point3 {
        match pose {
            Pose::Upright => p,
            Pose::Turned => {
                let turn = Mat4::rotation_z(0.7) * Mat4::rotation_x(1.1) * Mat4::rotation_y(0.3);
                turn.mul_point(p)
            }
            Pose::Mirrored => {
                let (at, n) = (
                    Point3::new(0.3, 0.0, 0.0),
                    Vec3::new(1.0, 0.2, 0.1).normalize().unwrap(),
                );
                p - n * (2.0 * (p - at).dot(n))
            }
        }
    }

    /// A rod along y of radius `rho` round the line through `(x0, ., z0)`,
    /// from `y = -2` to `y = 2`, through the ball far below its cap.
    fn rod(topo: &mut Topology, rho: f64, x0: f64, z0: f64) -> SolidId {
        let rod = make_cylinder(topo, rho, 4.0).unwrap();
        let place = Mat4::translation(x0, -2.0, z0) * Mat4::rotation_x(-FRAC_PI_2);
        transform_solid(topo, rod, &place).unwrap();
        rod
    }

    /// The box from `lo` to `hi`.
    fn block(topo: &mut Topology, lo: [f64; 3], hi: [f64; 3]) -> SolidId {
        let b = make_box(topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
        transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
        b
    }

    /// The area of the rectangle from the origin to `(a, b)`, signed by the
    /// signs of `a` and `b`, inside the disc of radius `rho` at the origin.
    fn quadrant_area(a: f64, b: f64, rho: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        let (x, y) = (a.abs(), b.abs());
        // The area under the circle from 0 to `s`.
        let under = |s: f64| {
            0.5 * s.madd(
                (rho * rho - s * s).max(0.0).sqrt(),
                rho * rho * (s / rho).clamp(-1.0, 1.0).asin(),
            )
        };
        let flat = (rho * rho - y * y).max(0.0).sqrt().min(x);
        let round = x.min(rho);
        a.signum() * b.signum() * (y * flat + under(round) - under(flat.min(round)))
    }

    /// The volume of a unit ball at the origin inside the box from `lo` to
    /// `hi`: the area of the box's section inside the ball's section, summed
    /// over the box's heights by Gauss points between the heights where the
    /// ball's section reaches a side or a corner of the box's.
    fn ball_held_by_a_box(lo: [f64; 3], hi: [f64; 3]) -> f64 {
        let area = |z: f64| {
            let rho = z.madd(-z, 1.0).max(0.0).sqrt();
            quadrant_area(hi[0], hi[1], rho)
                - quadrant_area(lo[0], hi[1], rho)
                - quadrant_area(hi[0], lo[1], rho)
                + quadrant_area(lo[0], lo[1], rho)
        };
        let mut reach = Vec::new();
        for x in [lo[0], hi[0]] {
            for y in [lo[1], hi[1]] {
                reach.extend([x.abs(), y.abs(), x.hypot(y)]);
            }
        }
        let mut breaks = vec![lo[2], hi[2]];
        for d in reach.into_iter().filter(|&d| d < 1.0) {
            let z = d.madd(-d, 1.0).sqrt();
            breaks.extend([z, -z]);
        }
        breaks.retain(|&z| z >= lo[2] && z <= hi[2]);
        breaks.sort_by(f64::total_cmp);
        let points = brepkit_math::quadrature::gauss_legendre_points(20);
        breaks
            .windows(2)
            .map(|w| {
                let (mid, half) = (0.5 * (w[0] + w[1]), 0.5 * (w[1] - w[0]));
                points
                    .iter()
                    .map(|gp| half * gp.w * area(half.madd(gp.x, mid)))
                    .sum::<f64>()
            })
            .sum()
    }

    /// Each operation of the ball less a cap and `tool` in each pose: exact,
    /// valid, closed, of the volume `held` (the ball's part inside the tool,
    /// of volume `tool_volume`) gives, and each probe (a point, whether it
    /// lies in the ball and in the tool) read as the operation keeps it.
    fn check_each_operation(
        what: &str,
        tool: &dyn Fn(&mut Topology) -> SolidId,
        held: f64,
        tool_volume: f64,
        probes: &[(Point3, bool, bool)],
    ) -> Vec<String> {
        let opts = BooleanOptions {
            mesh_fallback: false,
            unify_faces: false,
            ..BooleanOptions::default()
        };
        let mut failures = Vec::new();
        for (op, want) in [
            (BooleanOp::Cut, whole_volume() - held),
            (BooleanOp::Intersect, held),
            (BooleanOp::Fuse, whole_volume() - held + tool_volume),
        ] {
            for pose in [Pose::Upright, Pose::Turned, Pose::Mirrored] {
                let what = format!("{what} {op:?} {pose:?}");
                let mut topo = Topology::new();
                let ball = ball_less_a_cap(&mut topo);
                let tool = tool(&mut topo);
                let (ball, tool) = (place(&mut topo, ball, pose), place(&mut topo, tool, pose));
                let result = match boolean_with_options(&mut topo, op, ball, tool, opts) {
                    Ok(r) => r,
                    Err(e) => {
                        failures.push(format!("{what}: {e}"));
                        continue;
                    }
                };
                let report = validate_solid(&topo, result).unwrap();
                if !report.is_valid() {
                    failures.push(format!("{what}: invalid {:?}", report.issues));
                }
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
                    failures.push(format!("{what}: mesh not closed"));
                }
                let volume = solid_volume(&topo, result, 0.0005).unwrap();
                if ((volume - want) / want).abs() > 1e-7 {
                    failures.push(format!("{what}: volume {volume} against {want}"));
                }
                for &(at, in_ball, in_tool) in probes {
                    let kept = match op {
                        BooleanOp::Cut => in_ball && !in_tool,
                        BooleanOp::Intersect => in_ball && in_tool,
                        BooleanOp::Fuse => in_ball || in_tool,
                    };
                    let class = classify_point(&topo, result, map(pose, at), 0.0, 1e-7).unwrap();
                    let want = if kept {
                        PointClassification::Inside
                    } else {
                        PointClassification::Outside
                    };
                    if class != want {
                        failures.push(format!("{what}: {at:?} reads {class:?}"));
                    }
                }
            }
        }
        failures
    }

    /// Rods across a ball less a small cap, far below the cap, in three
    /// poses: each cut, intersect and fuse is exact, valid, meshes closed,
    /// holds the volume a quadrature over the rod's section gives, and a ray
    /// cast reads points in and beside the rod as the operation keeps them.
    #[test]
    fn rods_across_a_ball_less_a_small_cap_are_exact() {
        let mut failures = Vec::new();
        for (rho, x0, z0) in [(0.3_f64, 0.0, -0.5), (0.2, 0.35, -0.3), (0.25, -0.1, 0.2)] {
            // Points in the ball inside the rod, in the ball beside the rod,
            // and in the rod outside the ball.
            let probes = [
                (Point3::new(x0, 0.0, z0), true, true),
                (Point3::new(x0, 0.0, z0 + rho + 0.05), true, false),
                (Point3::new(x0, 1.5, z0), false, true),
            ];
            failures.extend(check_each_operation(
                &format!("rod {rho} at ({x0}, {z0})"),
                &|topo| rod(topo, rho, x0, z0),
                ball_held_by_a_rod(rho, x0, z0),
                PI * rho * rho * 4.0,
                &probes,
            ));
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }

    /// Small boxes through the ball's face between its equator and its cap,
    /// beside the cap, and near the bottom off its pole, in three poses: the
    /// face-face pass keeps the arcs the boxes' sides cut on the ball there,
    /// so each cut, intersect and fuse is exact, valid, meshes closed, holds
    /// the volume the box's sections give, and a ray cast reads points in the
    /// box and the ball, in the ball beside the box, and in the box outside
    /// the ball as the operation keeps them.
    #[test]
    fn small_boxes_through_a_ball_less_a_small_cap_are_exact() {
        let mut failures = Vec::new();
        for (lo, hi, probes) in [
            (
                [0.7, -0.1, 0.3],
                [1.2, 0.1, 0.5],
                [
                    (Point3::new(0.75, 0.0, 0.4), true, true),
                    (Point3::new(0.5, 0.0, 0.4), true, false),
                    (Point3::new(1.15, 0.0, 0.4), false, true),
                ],
            ),
            (
                [0.85, -0.15, 0.4],
                [1.15, 0.15, 0.7],
                [
                    (Point3::new(0.87, 0.0, 0.45), true, true),
                    (Point3::new(0.6, 0.0, 0.5), true, false),
                    (Point3::new(1.1, 0.0, 0.6), false, true),
                ],
            ),
            (
                [0.3, -0.1, -1.2],
                [0.5, 0.1, -0.8],
                [
                    (Point3::new(0.4, 0.0, -0.85), true, true),
                    (Point3::new(0.4, 0.0, -0.5), true, false),
                    (Point3::new(0.4, 0.0, -1.1), false, true),
                ],
            ),
        ] {
            let tool_volume = (hi[0] - lo[0]) * (hi[1] - lo[1]) * (hi[2] - lo[2]);
            failures.extend(check_each_operation(
                &format!("box {lo:?} to {hi:?}"),
                &|topo| block(topo, lo, hi),
                ball_held_by_a_box(lo, hi),
                tool_volume,
                &probes,
            ));
        }
        assert!(failures.is_empty(), "{}", failures.join("\n"));
    }
}

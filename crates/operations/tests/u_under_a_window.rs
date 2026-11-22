//! A solid that meets a duct's bore in a U under a window through its
//! wall, built as in the brepjs playground example `axial-fan`: a wall from
//! radius 38 to 42 and z = 1 to 23, less a slot 3.2 wide from z = 3 to 4.5,
//! both turned -45 degrees about z. A rod along the slot whose axis lies at
//! or under the slot's floor, or a box under the slot between its side
//! planes, meets the bore below the window's bottom edge: its footprint on
//! the bore runs from that edge down and back, and neither end reaches the
//! bore's rims.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::classify::{PointClassification, classify_point};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::mirror::mirror;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::explorer::solid_faces;
    use brepkit_topology::solid::SolidId;

    const POSES: [&str; 3] = ["upright", "turned", "mirrored"];

    fn turn() -> Mat4 {
        Mat4::translation(5.0, -4.0, 3.0) * Mat4::rotation_x(0.43) * Mat4::rotation_z(0.71)
    }

    fn mirror_plane() -> (Point3, Vec3) {
        (Point3::new(-10.0, 3.0, 1.0), Vec3::new(0.2, 1.0, 0.4))
    }

    fn posed(topo: &mut Topology, solid: SolidId, pose: &str) -> SolidId {
        match pose {
            "turned" => {
                transform_solid(topo, solid, &turn()).unwrap();
                solid
            }
            "mirrored" => {
                let (at, normal) = mirror_plane();
                mirror(topo, solid, at, normal).unwrap()
            }
            _ => solid,
        }
    }

    fn placed(p: Point3, pose: &str) -> Point3 {
        match pose {
            "turned" => turn().mul_point(p),
            "mirrored" => {
                let (at, normal) = mirror_plane();
                let unit = normal.normalize().unwrap();
                p - unit * (2.0 * (p - at).dot(unit))
            }
            _ => p,
        }
    }

    fn exact() -> BooleanOptions {
        BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        }
    }

    /// A box of `size` centred on `at`, turned -45 degrees about z.
    fn turned_box(topo: &mut Topology, size: [f64; 3], at: [f64; 3]) -> SolidId {
        let b = make_box(topo, size[0], size[1], size[2]).unwrap();
        let place = Mat4::rotation_z(-FRAC_PI_4)
            * Mat4::translation(
                at[0] - size[0] / 2.0,
                at[1] - size[1] / 2.0,
                at[2] - size[2] / 2.0,
            );
        transform_solid(topo, b, &place).unwrap();
        b
    }

    /// The duct wall less the slot: 3.2 wide, from z = 3 to 4.5, through the
    /// wall at y = -6 before the turn.
    fn wall(topo: &mut Topology) -> SolidId {
        let outer = make_cylinder(topo, 42.0, 22.0).unwrap();
        transform_solid(topo, outer, &Mat4::translation(0.0, 0.0, 1.0)).unwrap();
        let bore = make_cylinder(topo, 38.0, 24.0).unwrap();
        let tube = boolean_with_options(topo, BooleanOp::Cut, outer, bore, exact()).unwrap();
        let slot = turned_box(topo, [30.0, 3.2, 1.5], [51.0, -6.0, 3.75]);
        boolean_with_options(topo, BooleanOp::Cut, tube, slot, exact()).unwrap()
    }

    /// A rod of radius 1.2 along x from 30 to 40 at y = -6, its axis at
    /// height `z`, turned -45 degrees about z.
    fn rod(topo: &mut Topology, z: f64) -> SolidId {
        let r = make_cylinder(topo, 1.2, 10.0).unwrap();
        let place = Mat4::rotation_z(-FRAC_PI_4)
            * Mat4::translation(30.0, -6.0, z)
            * Mat4::rotation_y(FRAC_PI_2);
        transform_solid(topo, r, &place).unwrap();
        r
    }

    /// The area under a circle of radius `r` from y = `a` to `b`.
    fn under_circle(r: f64, a: f64, b: f64) -> f64 {
        let f = |y: f64| 0.5 * (y * (r * r - y * y).sqrt() + r * r * (y / r).asin());
        f(b) - f(a)
    }

    /// The common part of the wall and a tool whose section across y, from
    /// z = `lo(y)` to `hi(y)` over y in `[y0, y1]`, runs along x from the
    /// bore to x = 40. With `y = -6 + 1.2 sin t` over the rod's span, the
    /// rod's section height has no root at the ends.
    fn common(lo: impl Fn(f64) -> f64, hi: impl Fn(f64) -> f64, rod_span: bool) -> f64 {
        let n = 20_000;
        let length = |y: f64| 40.0 - (38.0f64 * 38.0 - y * y).sqrt();
        let height = |y: f64| (hi(y).min(3.0) - lo(y)).max(0.0);
        let (a, b) = if rod_span {
            (-FRAC_PI_2, FRAC_PI_2)
        } else {
            (-7.6, -4.4)
        };
        let step = (b - a) / f64::from(n);
        (0..n)
            .map(|k| {
                let t = a + (f64::from(k) + 0.5) * step;
                if rod_span {
                    let y = 1.2 * t.sin() - 6.0;
                    height(y) * length(y) * 1.2 * t.cos() * step
                } else {
                    height(t) * length(t) * step
                }
            })
            .sum()
    }

    /// The shortest edge of `solid`, measured along 16 chords.
    fn shortest_edge(topo: &Topology, solid: SolidId) -> f64 {
        let mut shortest = f64::INFINITY;
        for face in solid_faces(topo, solid).unwrap() {
            let face = topo.face(face).unwrap();
            for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                for oe in topo.wire(wire).unwrap().edges() {
                    let edge = topo.edge(oe.edge()).unwrap();
                    let a = topo.vertex(edge.start()).unwrap().point();
                    let b = topo.vertex(edge.end()).unwrap().point();
                    let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                    let at = |k: u32| {
                        let t = t0 + (t1 - t0) * f64::from(k) / 16.0;
                        edge.curve().evaluate_with_endpoints(t, a, b)
                    };
                    let length: f64 = (0..16).map(|k| (at(k + 1) - at(k)).length()).sum();
                    shortest = shortest.min(length);
                }
            }
        }
        shortest
    }

    fn holds(topo: &Topology, solid: SolidId, p: Point3) -> Option<bool> {
        match classify_point(topo, solid, p, 0.01, 1e-7).unwrap() {
            PointClassification::Inside => Some(true),
            PointClassification::Outside => Some(false),
            PointClassification::OnBoundary => None,
        }
    }

    /// Fuse, cut and intersect the wall with `tool` in every pose. Each
    /// result is exact, valid and watertight, has no edge shorter than 0.1,
    /// holds the closed-form volume (`common` is the tool's part in the
    /// wall), and over a grid round the U holds a point as its operands say.
    fn u_is_exact(tool: impl Fn(&mut Topology) -> SolidId, tool_volume: f64, common: f64) {
        let window = 1.5 * (under_circle(42.0, -7.6, -4.4) - under_circle(38.0, -7.6, -4.4));
        let wall_volume = PI * (42.0f64 * 42.0 - 38.0 * 38.0) * 22.0 - window;
        let ops = [
            (BooleanOp::Fuse, wall_volume + tool_volume - common),
            (BooleanOp::Cut, wall_volume - common),
            (BooleanOp::Intersect, common),
        ];
        for pose in POSES {
            for (op, want) in ops {
                let mut topo = Topology::new();
                let w = wall(&mut topo);
                let t = tool(&mut topo);
                let (a, b) = (posed(&mut topo, w, pose), posed(&mut topo, t, pose));
                let what = format!("{pose} {op:?}");
                let result = boolean_with_options(&mut topo, op, a, b, exact())
                    .map_err(|e| format!("{what}: {e}"))
                    .unwrap();
                let report = validate_solid(&topo, result).unwrap();
                assert!(report.is_valid(), "{what}: invalid: {report:?}");
                let mesh = tessellate_solid(&topo, result, 0.01).unwrap();
                assert!(is_watertight(&mesh), "{what}: open or non-manifold mesh");
                let shortest = shortest_edge(&topo, result);
                assert!(shortest > 0.1, "{what}: an edge {shortest} long");
                let got = solid_volume(&topo, result, 0.01).unwrap();
                assert!(
                    (got - want).abs() < 1e-6 * want,
                    "{what}: volume {got}, want {want}"
                );
                let centre = Mat4::rotation_z(-FRAC_PI_4).mul_point(Point3::new(38.5, -6.0, 3.0));
                let n = 14;
                let mut read = 0;
                #[allow(clippy::cast_precision_loss)]
                let at = |s: usize| (s as f64 + 0.5) / n as f64 - 0.5;
                for i in 0..n {
                    for j in 0..n {
                        for k in 0..n {
                            let p = placed(
                                Point3::new(
                                    centre.x() + 5.0 * at(i),
                                    centre.y() + 5.0 * at(j),
                                    centre.z() + 4.0 * at(k),
                                ),
                                pose,
                            );
                            let (Some(r), Some(ia), Some(ib)) = (
                                holds(&topo, result, p),
                                holds(&topo, a, p),
                                holds(&topo, b, p),
                            ) else {
                                continue;
                            };
                            let want_in = match op {
                                BooleanOp::Fuse => ia || ib,
                                BooleanOp::Cut => ia && !ib,
                                BooleanOp::Intersect => ia && ib,
                            };
                            assert_eq!(r, want_in, "{what}: at {p:?}");
                            read += 1;
                        }
                    }
                }
                assert!(read > n * n * n / 2, "{what}: only {read} points read");
            }
        }
    }

    fn rod_half_height(y: f64) -> f64 {
        (1.44 - (y + 6.0) * (y + 6.0)).max(0.0).sqrt()
    }

    /// A rod whose axis lies in the slot's floor: its lower half meets the
    /// bore in a U from the window's bottom edge.
    #[test]
    fn a_rod_along_the_slot_floor_meets_the_bore_exactly() {
        let common = common(|y| 3.0 - rod_half_height(y), |_| 3.0, true);
        u_is_exact(|topo| rod(topo, 3.0), PI * 1.44 * 10.0, common);
    }

    /// A rod whose axis lies 0.4 under the slot's floor.
    #[test]
    fn a_rod_under_the_slot_floor_meets_the_bore_exactly() {
        let common = common(
            |y| 2.6 - rod_half_height(y),
            |y| 2.6 + rod_half_height(y),
            true,
        );
        u_is_exact(|topo| rod(topo, 2.6), PI * 1.44 * 10.0, common);
    }

    /// A box 3.2 wide from z = 2 to 3 under the slot, its sides in the
    /// slot's side planes and its top in the slot's floor.
    #[test]
    fn a_box_under_the_slot_meets_the_bore_exactly() {
        let common = common(|_| 2.0, |_| 3.0, false);
        u_is_exact(
            |topo| turned_box(topo, [10.0, 3.2, 1.0], [35.0, -6.0, 2.5]),
            10.0 * 3.2,
            common,
        );
    }
}

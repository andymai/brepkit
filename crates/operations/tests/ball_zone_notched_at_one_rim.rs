//! A ball between two latitudes, flat at both ends: its one sphere face is a
//! zone bounded by a meridian seam and two rims of three arcs each. A box
//! notches the upper rim, so its sections reach that rim alone. The zone
//! holds no pole, and spans a whole turn round the axis: a cut or a fuse with
//! the box is exact or refused, never a solid split along a rim alone.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::curves::Circle3D;
    use brepkit_math::mat::Mat4;
    use brepkit_math::surfaces::SphericalSurface;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, BooleanOptions, boolean_with_options};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{
        boundary_edge_count, non_manifold_edge_count, tessellate_solid,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::{Solid, SolidId};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    const LOW: f64 = -0.4;
    const HIGH: f64 = 0.6;
    /// The notch: `x` from 0.3 on, `|y| < 0.2`, `z` from 0.2 below the top.
    const NOTCH: ([f64; 3], [f64; 3]) = ([0.3, -0.2, HIGH - 0.2], [2.0, 0.2, HIGH + 1.2]);

    /// The unit ball between `z = LOW` and `z = HIGH`.
    fn zone_ball(topo: &mut Topology) -> SolidId {
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let rim = |topo: &mut Topology, h: f64| -> Vec<EdgeId> {
            let w = (1.0 - h * h).sqrt();
            let circle = Circle3D::new(Point3::new(0.0, 0.0, h), axis, w).unwrap();
            let vs: Vec<_> = (0..3)
                .map(|k| {
                    let a = std::f64::consts::TAU * f64::from(k) / 3.0;
                    topo.add_vertex(Vertex::new(Point3::new(w * a.cos(), w * a.sin(), h), 1e-7))
                })
                .collect();
            (0..3)
                .map(|k| {
                    topo.add_edge(Edge::new(
                        vs[k],
                        vs[(k + 1) % 3],
                        EdgeCurve::Circle(circle.clone()),
                    ))
                })
                .collect()
        };
        let low = rim(topo, LOW);
        let high = rim(topo, HIGH);
        let (from, to) = (
            topo.edge(low[0]).unwrap().start(),
            topo.edge(high[0]).unwrap().start(),
        );
        let centre = Point3::new(0.0, 0.0, 0.0);
        let at = topo.vertex(from).unwrap().point();
        let meridian =
            Circle3D::new_with_ref(centre, Vec3::new(0.0, -1.0, 0.0), 1.0, at - centre).unwrap();
        let seam = topo.add_edge(Edge::new(from, to, EdgeCurve::Circle(meridian)));
        let mut zone: Vec<OrientedEdge> = low.iter().map(|&e| OrientedEdge::new(e, true)).collect();
        zone.push(OrientedEdge::new(seam, true));
        zone.extend(high.iter().rev().map(|&e| OrientedEdge::new(e, false)));
        zone.push(OrientedEdge::new(seam, false));
        let zone = topo.add_wire(Wire::new(zone, true).unwrap());
        let ball = FaceSurface::Sphere(SphericalSurface::new(centre, 1.0).unwrap());
        let zone = topo.add_face(Face::new(zone, vec![], ball));
        let bottom = topo.add_wire(
            Wire::new(
                low.iter()
                    .rev()
                    .map(|&e| OrientedEdge::new(e, false))
                    .collect(),
                true,
            )
            .unwrap(),
        );
        let bottom = topo.add_face(Face::new(
            bottom,
            vec![],
            FaceSurface::Plane {
                normal: -axis,
                d: -LOW,
            },
        ));
        let top = topo.add_wire(
            Wire::new(
                high.iter().map(|&e| OrientedEdge::new(e, true)).collect(),
                true,
            )
            .unwrap(),
        );
        let top = topo.add_face(Face::new(
            top,
            vec![],
            FaceSurface::Plane {
                normal: axis,
                d: HIGH,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![zone, bottom, top]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    /// The area of the disc of radius `r` about the origin inside the
    /// rectangle `[y0, y1] x [z0, z1]`, in closed form between the `y` where
    /// the circle crosses `z = z0` or `z = z1`.
    fn disc_in_rectangle(r: f64, (y0, y1): (f64, f64), (z0, z1): (f64, f64)) -> f64 {
        let half = |y: f64| (r * r - y * y).max(0.0).sqrt();
        // The integral of `half` from 0 to `y`.
        let under = |y: f64| {
            let y = y.clamp(-r, r);
            0.5 * (y * half(y) + r * r * (y / r).asin())
        };
        let mut cuts = vec![y0.max(-r), y1.min(r)];
        if cuts[0] >= cuts[1] {
            return 0.0;
        }
        for z in [z0, z1] {
            if z.abs() < r {
                let y = half(z);
                cuts.extend([-y, y]);
            }
        }
        cuts.retain(|&y| y >= y0.max(-r) && y <= y1.min(r));
        cuts.sort_by(f64::total_cmp);
        let mut area = 0.0;
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            if b - a <= 0.0 {
                continue;
            }
            let s = half(0.5 * (a + b));
            let top = if z1 < s { None } else { Some(()) };
            let bottom = if z0 > -s { None } else { Some(()) };
            // Between two cuts the chord's span in z is one of these.
            let length = match (top, bottom) {
                (Some(()), Some(())) => 2.0 * (under(b) - under(a)),
                (Some(()), None) => (under(b) - under(a)) - z0 * (b - a),
                (None, Some(())) => z1 * (b - a) + (under(b) - under(a)),
                (None, None) => (z1 - z0) * (b - a),
            };
            area += length.max(0.0);
        }
        area
    }

    /// The ball between the two latitudes inside the notch: each slice's
    /// disc inside the notch's rectangle, integrated over `z`.
    fn zone_in_notch() -> f64 {
        const PANELS: u32 = 400;
        let (z0, z1) = (NOTCH.0[2], HIGH);
        let gauss = brepkit_math::quadrature::gauss_legendre_points(8);
        let width = (z1 - z0) / f64::from(PANELS);
        let mut total = 0.0;
        for p in 0..PANELS {
            let mid = (f64::from(p) + 0.5).mul_add(width, z0);
            for g in gauss {
                let z = (0.5 * width).mul_add(g.x, mid);
                let r = (1.0 - z * z).max(0.0).sqrt();
                let area = disc_in_rectangle(r, (NOTCH.0[0], NOTCH.1[0]), (NOTCH.0[1], NOTCH.1[1]));
                total += 0.5 * width * g.w * area;
            }
        }
        total
    }

    #[test]
    fn a_ball_zone_notched_at_one_rim_cuts_and_fuses_exactly_or_refuses() {
        let options = BooleanOptions {
            mesh_fallback: false,
            ..BooleanOptions::default()
        };
        let zone = std::f64::consts::PI * ((HIGH - LOW) - (HIGH.powi(3) - LOW.powi(3)) / 3.0);
        let notch: f64 = (0..3).map(|k| NOTCH.1[k] - NOTCH.0[k]).product();
        let shared = zone_in_notch();
        let mut faults = Vec::new();
        for (name, pose) in [
            ("upright", Mat4::identity()),
            ("turned", Mat4::rotation_z(0.37) * Mat4::rotation_x(0.21)),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ] {
            for (op, expected) in [
                (BooleanOp::Cut, zone - shared),
                (BooleanOp::Fuse, zone + notch - shared),
            ] {
                let mut topo = Topology::new();
                let ball = zone_ball(&mut topo);
                transform_solid(&mut topo, ball, &pose).unwrap();
                let (lo, hi) = NOTCH;
                let tool =
                    make_box(&mut topo, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]).unwrap();
                let place = pose * Mat4::translation(lo[0], lo[1], lo[2]);
                transform_solid(&mut topo, tool, &place).unwrap();
                let Ok(result) = boolean_with_options(&mut topo, op, ball, tool, options) else {
                    continue;
                };
                let label = format!("{name} {op:?}");
                let report = validate_solid(&topo, result).unwrap();
                if !report.is_valid() {
                    faults.push(format!("{label}: invalid {:?}", report.issues));
                    continue;
                }
                let mesh = tessellate_solid(&topo, result, 0.002).unwrap();
                if boundary_edge_count(&mesh) != 0 || non_manifold_edge_count(&mesh) != 0 {
                    faults.push(format!("{label}: mesh not closed"));
                }
                let volume = solid_volume(&topo, result, 0.0005).unwrap();
                if (volume - expected).abs() >= 1e-6 * expected {
                    faults.push(format!("{label}: volume {volume} should be {expected}"));
                }
            }
        }
        assert!(faults.is_empty(), "{}", faults.join("\n"));
    }
}

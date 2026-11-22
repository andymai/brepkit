//! Exact solids measured upright, turned and mirrored against their closed
//! forms.
//!
//! - A block 24 x 24 x 1000 less a D-shaped hole: a bore of radius 10 cut
//!   off by a flat that leans 5e-8 off the bore's axis, which meets the bore
//!   along two arcs of an ellipse 2e8 across, centred 1e8 away.
//! - A frustum whose wall is put in its B-spline form, which closes round on
//!   itself, with its bottom rim chamfered: the wall is trimmed along the
//!   chamfer's contact, which its seam crosses.
//! - The corner of a cube that reaches into a torus: three flats and a patch
//!   of the torus a sixth of a turn from where its frame's angle starts, or
//!   most of a turn the other way once mirrored.
//! - A ball less a corner: the ball's halves meet along a polygon of chords
//!   of its equator, which lie inside the ball.
//! - A rod with a cone on each end, its two rims polygons of chords: each
//!   face reads a chord where the chord's points project onto it, on the
//!   circle of the rim's height.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_heal::custom::convert_to_bspline::face_bspline_surface;
    use brepkit_math::mat::Mat4;
    use brepkit_math::quadrature::gauss_legendre_points;
    use brepkit_math::surfaces::{ConicalSurface, CylindricalSurface};
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::chamfer_v2;
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{
        make_box, make_cone, make_cylinder, make_sphere, make_torus,
    };
    use brepkit_operations::transform::transform_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::{Solid, SolidId};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    fn poses() -> Vec<(&'static str, Mat4)> {
        vec![
            ("upright", Mat4::translation(0.0, 0.0, 0.0)),
            ("turned", Mat4::rotation_z(0.7)),
            (
                "tilted",
                Mat4::rotation_x(0.4) * Mat4::rotation_y(1.1) * Mat4::rotation_z(0.3),
            ),
            ("mirrored", Mat4::scale(-1.0, 1.0, 1.0)),
        ]
    }

    /// `f` integrated from `a` to `b` in `panels` equal panels.
    fn integral(a: f64, b: f64, panels: u32, f: impl Fn(f64) -> f64) -> f64 {
        let h = (b - a) / f64::from(panels);
        (0..panels)
            .map(|k| {
                let lo = h.mul_add(f64::from(k), a);
                gauss_legendre_points(20)
                    .iter()
                    .map(|g| 0.5 * h * g.w * f((0.5 * h).mul_add(g.x + 1.0, lo)))
                    .sum::<f64>()
            })
            .sum()
    }

    const BORE: f64 = 10.0;
    const FLAT: f64 = 5.0;
    const LEAN: f64 = 5e-8;
    const LENGTH: f64 = 1000.0;

    /// The block less the D-shaped hole: the bore on the `z` axis and the
    /// side of the flat `y cos θ + z sin θ = 5` toward it, `θ = atan(5e-8)`.
    fn d_holed_block(topo: &mut Topology) -> SolidId {
        let block = make_box(topo, 24.0, 24.0, LENGTH).unwrap();
        transform_solid(topo, block, &Mat4::translation(-12.0, -12.0, 0.0)).unwrap();
        let rod = make_cylinder(topo, BORE, LENGTH + 20.0).unwrap();
        transform_solid(topo, rod, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
        let keep = make_box(topo, 2.0 * BORE, 2.0 * BORE, LENGTH + 40.0).unwrap();
        let place =
            Mat4::rotation_x(LEAN.atan()) * Mat4::translation(-BORE, FLAT - 2.0 * BORE, -20.0);
        transform_solid(topo, keep, &place).unwrap();
        let d = boolean(topo, BooleanOp::Intersect, rod, keep).unwrap();
        boolean(topo, BooleanOp::Cut, block, d).unwrap()
    }

    /// The area of the disc of radius 10 on the side of `y = y0` toward its
    /// centre.
    fn d_area(y0: f64) -> f64 {
        let r = BORE;
        PI * r * r - (r * r * (y0 / r).acos() - y0 * (r * r - y0 * y0).sqrt())
    }

    #[test]
    fn a_block_with_a_long_d_shaped_hole_measures_alike_in_every_pose() {
        let theta = LEAN.atan();
        let hole = integral(0.0, LENGTH, 4, |z| {
            d_area(z.mul_add(-theta.sin(), FLAT) / theta.cos())
        });
        let expected = 24.0 * 24.0 * LENGTH - hole;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = d_holed_block(&mut topo);
            transform_solid(&mut topo, solid, &pose).unwrap();
            let volume = solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }

    /// The frustum 3 to 1 across and 4 high, its wall in B-spline form,
    /// placed by `pose`, with its bottom rim chamfered 0.25 into the base and
    /// 0.4 up the wall.
    fn chamfered_b_spline_frustum(topo: &mut Topology, pose: &Mat4) -> SolidId {
        let frustum = make_cone(topo, 3.0, 1.0, 4.0).unwrap();
        let rim = solid_edges(topo, frustum)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
            })
            .unwrap();
        for face in solid_faces(topo, frustum).unwrap() {
            if topo.face(face).unwrap().surface().is_planar() {
                continue;
            }
            let wall = face_bspline_surface(topo, face).unwrap().unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(FaceSurface::Nurbs(wall));
        }
        transform_solid(topo, frustum, pose).unwrap();
        let built = chamfer_v2(topo, frustum, &[rim], 0.4, 0.25).unwrap();
        assert!(built.failed.is_empty(), "{:?}", built.failed);
        built.solid
    }

    #[test]
    fn a_chamfered_b_spline_frustum_measures_exactly_at_a_coarse_deflection() {
        // The section the chamfer takes off, 0.25 along the base and 0.4 up
        // the wall, which rises 4 as it closes in by 2, swept round the axis
        // at its centroid's radius.
        let (sin, cos) = 2.0_f64.atan2(1.0).sin_cos();
        let section = 0.5 * 0.25 * 0.4 * sin;
        let removed = 2.0 * PI * (3.0 - 0.4f64.mul_add(cos, 0.25) / 3.0) * section;
        let expected = PI * 4.0 / 3.0 * (9.0 + 3.0 + 1.0) - removed;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = chamfered_b_spline_frustum(&mut topo, &pose);
            let volume = solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }

    /// A cone 3 across at its base and 4 high, its wall in B-spline form,
    /// placed by `pose`, with its base rim chamfered 0.25 into the base and
    /// 0.4 up the wall. The wall runs to a point, where the exact integral
    /// declines it: it alone is read over its mesh.
    fn chamfered_b_spline_cone(topo: &mut Topology, pose: &Mat4) -> SolidId {
        let cone = make_cone(topo, 3.0, 0.0, 4.0).unwrap();
        let rim = solid_edges(topo, cone)
            .unwrap()
            .into_iter()
            .find(|&e| {
                matches!(topo.edge(e).unwrap().curve(),
                    EdgeCurve::Circle(c) if c.center().z().abs() < 1e-9)
            })
            .unwrap();
        for face in solid_faces(topo, cone).unwrap() {
            if topo.face(face).unwrap().surface().is_planar() {
                continue;
            }
            let wall = face_bspline_surface(topo, face).unwrap().unwrap();
            topo.face_mut(face)
                .unwrap()
                .set_surface(FaceSurface::Nurbs(wall));
        }
        transform_solid(topo, cone, pose).unwrap();
        let built = chamfer_v2(topo, cone, &[rim], 0.4, 0.25).unwrap();
        assert!(built.failed.is_empty(), "{:?}", built.failed);
        built.solid
    }

    #[test]
    fn a_chamfered_b_spline_cone_measures_exactly() {
        // The chamfer's section swept round the axis at its centroid's
        // radius, the wall rising 4 as it closes in by 3.
        let (sin, cos) = 4.0_f64.atan2(3.0).sin_cos();
        let section = 0.5 * 0.25 * 0.4 * sin;
        let removed = 2.0 * PI * (3.0 - 0.4f64.mul_add(cos, 0.25) / 3.0) * section;
        let expected = PI * 9.0 * 4.0 / 3.0 - removed;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = chamfered_b_spline_cone(&mut topo, &pose);
            let volume = solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }

    const MAJOR: f64 = 4.0;
    const MINOR: f64 = 1.5;
    /// The cube's corner that reaches into the torus, `(x, y, z)` least.
    const CORNER: [f64; 3] = [2.4, 4.6, 0.5];

    /// The area of the disc of radius `rho` about the `z` axis beyond the
    /// lines `x = x0` and `y = y0`, where the circle stays inside a cube's
    /// other sides.
    fn corner_cap(rho: f64, (x0, y0): (f64, f64)) -> f64 {
        if rho * rho <= x0.mul_add(x0, y0 * y0) {
            return 0.0;
        }
        let x1 = rho.mul_add(rho, -y0 * y0).sqrt();
        let below = |x: f64| {
            0.5 * x.mul_add(
                rho.mul_add(rho, -x * x).sqrt(),
                rho * rho * (x / rho).asin(),
            )
        };
        below(x1) - below(x0) - y0 * (x1 - x0)
    }

    #[test]
    fn a_cube_corner_in_a_torus_measures_alike_in_every_pose() {
        // At height `z` the torus holds the disc out to `R + sqrt(r² - z²)`,
        // which reaches the corner's edge below `top`.
        let [x0, y0, z0] = CORNER;
        let reach = x0.hypot(y0) - MAJOR;
        let top = MINOR.mul_add(MINOR, -reach * reach).sqrt();
        let expected = integral(z0, top, 8, |z| {
            corner_cap(MAJOR + MINOR.mul_add(MINOR, -z * z).sqrt(), (x0, y0))
        });
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let ring = make_torus(&mut topo, MAJOR, MINOR, 32).unwrap();
            let cube = make_box(&mut topo, 1.2, 1.2, 1.2).unwrap();
            transform_solid(&mut topo, cube, &Mat4::translation(x0, y0, z0)).unwrap();
            let piece = boolean(&mut topo, BooleanOp::Intersect, ring, cube).unwrap();
            transform_solid(&mut topo, piece, &pose).unwrap();
            let volume = solid_volume(&topo, piece, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }

    #[test]
    fn a_ball_less_a_corner_measures_alike_in_every_pose() {
        // The corner `x > 1, y > 1.2, z > 0.8` takes the disc of the ball's
        // section beyond its two lines, up to where the section's rim passes
        // the corner's edge.
        let (x0, y0, z0) = (1.0_f64, 1.2_f64, 0.8);
        let top = (9.0 - x0.mul_add(x0, y0 * y0)).sqrt();
        let corner = integral(z0, top, 8, |z| {
            corner_cap(z.mul_add(-z, 9.0).sqrt(), (x0, y0))
        });
        let expected = 36.0 * PI - corner;
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let ball = make_sphere(&mut topo, 3.0, 32).unwrap();
            let cube = make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();
            transform_solid(&mut topo, cube, &Mat4::translation(x0, y0, z0)).unwrap();
            let rest = boolean(&mut topo, BooleanOp::Cut, ball, cube).unwrap();
            transform_solid(&mut topo, rest, &pose).unwrap();
            let volume = solid_volume(&topo, rest, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }

    /// The rod's radius and length, each cone's height, and the chords of
    /// each rim.
    const ROD: (f64, f64, f64, u32) = (2.0, 3.0, 1.5, 24);

    /// A polygon of chords round the rim of radius `ROD.0` at `z`.
    fn chord_rim(topo: &mut Topology, z: f64) -> Vec<EdgeId> {
        let (radius, _, _, chords) = ROD;
        let corners: Vec<_> = (0..chords)
            .map(|k| {
                let angle = 2.0 * PI * f64::from(k) / f64::from(chords);
                let at = Point3::new(radius * angle.cos(), radius * angle.sin(), z);
                topo.add_vertex(Vertex::new(at, 1e-7))
            })
            .collect();
        (0..corners.len())
            .map(|k| {
                let next = corners[(k + 1) % corners.len()];
                topo.add_edge(Edge::new(corners[k], next, EdgeCurve::Line))
            })
            .collect()
    }

    /// The rim's chords round the `z` axis, counter-clockwise seen from
    /// above or the other way.
    fn rim_wire(topo: &mut Topology, rim: &[EdgeId], up: bool) -> brepkit_topology::wire::WireId {
        let edges = if up {
            rim.iter().map(|&e| OrientedEdge::new(e, true)).collect()
        } else {
            rim.iter()
                .rev()
                .map(|&e| OrientedEdge::new(e, false))
                .collect()
        };
        topo.add_wire(Wire::new(edges, true).unwrap())
    }

    /// The rod from `z = 0` to its length, with a cone of `ROD.2` on each
    /// end, its rims polygons of chords shared by the rod and the cones.
    fn rod_with_chord_rims(topo: &mut Topology) -> SolidId {
        let (radius, length, height, _) = ROD;
        let lean = height.atan2(radius);
        let (bottom, top) = (chord_rim(topo, 0.0), chord_rim(topo, length));
        let up = Vec3::new(0.0, 0.0, 1.0);
        let wall = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), up, radius).unwrap();
        let below = ConicalSurface::new(Point3::new(0.0, 0.0, -height), up, lean).unwrap();
        let above = ConicalSurface::new(Point3::new(0.0, 0.0, length + height), -up, lean).unwrap();
        let (wall_bottom, wall_top) = (rim_wire(topo, &bottom, true), rim_wire(topo, &top, false));
        let wall = topo.add_face(Face::new(
            wall_bottom,
            vec![wall_top],
            FaceSurface::Cylinder(wall),
        ));
        let cone_below = rim_wire(topo, &bottom, false);
        let cone_above = rim_wire(topo, &top, true);
        let below = topo.add_face(Face::new(cone_below, vec![], FaceSurface::Cone(below)));
        let above = topo.add_face(Face::new(cone_above, vec![], FaceSurface::Cone(above)));
        let shell = topo.add_shell(Shell::new(vec![below, wall, above]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn a_rod_and_cones_whose_rims_are_chords_measure_alike_in_every_pose() {
        let (radius, length, height, _) = ROD;
        let expected = PI * radius * radius * (length + 2.0 * height / 3.0);
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let solid = rod_with_chord_rims(&mut topo);
            transform_solid(&mut topo, solid, &pose).unwrap();
            let volume = solid_volume(&topo, solid, 1e-3).unwrap();
            assert!(
                (volume - expected).abs() < 1e-9 * expected,
                "{name}: {volume}, expected {expected}, off {:e}",
                (volume - expected) / expected
            );
        }
    }
}

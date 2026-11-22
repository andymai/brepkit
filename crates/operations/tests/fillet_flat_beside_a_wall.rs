//! Rounds where a flat meets a cylinder wall along a line: `y = 5` beside a
//! wall of radius 10 about the `z` axis, exactly parallel to the axis or
//! leaning off it, `y + 5e-8 z = 5`.
//!
//! - A D-shaped hole through a block, both its corners rounded at 1: the
//!   flat and the bore are both reversed faces, whose wires wind about
//!   their surfaces' normals. Exactly parallel, each corner's band is an
//!   exact cylinder; leaning, the walker takes it.
//! - A D-shaped rod 10000 long, one corner rounded at 1: a band laid
//!   straight along the corner is off the wall by 3.6e-5 at its far end.
//!
//! In the section square to the axis, a round of the corner where the
//! flat `y = y0` meets the circle takes off (or fills) the region between
//! the flat, the circle and the ball's arc: the ball's centre is 1 off the
//! flat and 9 from the axis. By Green's theorem round that region, its area
//! `A(y0)`, integrated along `z` as `y0` runs from 5 to `5 - 5e-8 z`.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::FRAC_PI_2;

    use brepkit_blend::fillet_builder::FilletBuilder;
    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::{make_box, make_cylinder};
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::adjacency::AdjacencyIndex;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{edge_to_face_map, solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    const WALL: f64 = 10.0;
    const FLAT: f64 = 5.0;
    const TILT: f64 = 5e-8;
    const R: f64 = 1.0;

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

    /// The area a round of the corner where the flat `y = y0` meets the
    /// circle takes off.
    fn corner_area(y0: f64) -> f64 {
        const N: u32 = 4000;
        let yc = y0 - R;
        let xc = ((WALL - R).powi(2) - yc * yc).sqrt();
        let xe = (WALL * WALL - y0 * y0).sqrt();
        let mut points = vec![(xc, y0), (xe, y0)];
        let (a0, a1) = (y0.atan2(xe), yc.atan2(xc));
        for i in 1..=N {
            let a = (a1 - a0).mul_add(f64::from(i) / f64::from(N), a0);
            points.push((WALL * a.cos(), WALL * a.sin()));
        }
        let b0 = yc.atan2(xc);
        for i in 1..=N {
            let b = (FRAC_PI_2 - b0).mul_add(f64::from(i) / f64::from(N), b0);
            points.push((R.mul_add(b.cos(), xc), R.mul_add(b.sin(), yc)));
        }
        let mut twice = 0.0;
        for (k, &(x0, y0)) in points.iter().enumerate() {
            let (x1, y1) = points[(k + 1) % points.len()];
            twice += x0 * y1 - x1 * y0;
        }
        0.5 * twice.abs()
    }

    /// `corner_area` integrated over `z` from 0 to `length`, the flat at
    /// `y = 5 - tilt z`.
    fn corner_volume(length: f64, tilt: f64) -> f64 {
        const STEPS: u32 = 50;
        let h = length / f64::from(STEPS);
        (0..STEPS)
            .map(|k| corner_area((-tilt).mul_add(h * (f64::from(k) + 0.5), FLAT)) * h)
            .sum()
    }

    /// The edges of `solid` where a plane meets a cylinder, longer than
    /// `long` along `z` before `pose`.
    fn corners(topo: &Topology, solid: SolidId, long: f64) -> Vec<EdgeId> {
        let adjacency = AdjacencyIndex::build(topo, solid).unwrap();
        solid_edges(topo, solid)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let faces = adjacency.faces_for_edge(e);
                let edge = topo.edge(e).unwrap();
                let a = topo.vertex(edge.start()).unwrap().point();
                let b = topo.vertex(edge.end()).unwrap().point();
                faces.len() == 2
                    && faces
                        .iter()
                        .any(|&f| topo.face(f).unwrap().surface().is_planar())
                    && faces.iter().any(|&f| {
                        matches!(topo.face(f).unwrap().surface(), FaceSurface::Cylinder(_))
                    })
                    && (a.z() - b.z()).abs() > long
            })
            .collect()
    }

    /// How far the farthest of 40 samples along any edge of `solid` lies
    /// off a face that uses it.
    fn worst_edge_off_its_faces(topo: &Topology, solid: SolidId) -> f64 {
        let faces_of = edge_to_face_map(topo, solid).unwrap();
        let mut worst: f64 = 0.0;
        for edge_id in solid_edges(topo, solid).unwrap() {
            let edge = topo.edge(edge_id).unwrap();
            let (a, b) = (
                topo.vertex(edge.start()).unwrap().point(),
                topo.vertex(edge.end()).unwrap().point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
            for face_id in &faces_of[&edge_id.index()] {
                let surface = topo.face(*face_id).unwrap().surface();
                for k in 1..40 {
                    let p = edge.curve().evaluate_with_endpoints(
                        (t1 - t0).mul_add(f64::from(k) / 40.0, t0),
                        a,
                        b,
                    );
                    let off = if let FaceSurface::Plane { normal, d } = surface {
                        (normal.dot(p - Point3::new(0.0, 0.0, 0.0)) - d).abs()
                    } else {
                        let (u, v) = surface.project_point(p).unwrap();
                        (surface.evaluate(u, v).unwrap() - p).length()
                    };
                    worst = worst.max(off);
                }
            }
        }
        worst
    }

    /// Rounds `edges` of `solid` placed by `pose`: the volume the round
    /// adds, read with the result turned back upright (a long part's volume
    /// read in a turned pose moves by about 2e-5 of the whole), and the
    /// result as it then stands.
    fn round(
        topo: &mut Topology,
        solid: SolidId,
        edges: &[EdgeId],
        pose: &Mat4,
        what: &str,
    ) -> (f64, SolidId) {
        let before = solid_volume(topo, solid, 1e-3).unwrap();
        transform_solid(topo, solid, pose).unwrap();
        let mut builder = FilletBuilder::new(topo, solid);
        builder.add_edges(edges, R);
        let built = builder.build().unwrap();
        assert!(built.failed.is_empty(), "{what}: {:?}", built.failed);
        let report = validate_solid(topo, built.solid).unwrap();
        assert!(report.is_valid(), "{what}: not valid {:?}", report.issues);
        let off = worst_edge_off_its_faces(topo, built.solid);
        assert!(off < 1e-5, "{what}: an edge stands {off} off its face");
        let mesh = tessellate_solid(topo, built.solid, 0.05).unwrap();
        assert!(is_watertight(&mesh), "{what}: open mesh");
        transform_solid(topo, built.solid, &pose.inverse().unwrap()).unwrap();
        (
            solid_volume(topo, built.solid, 1e-3).unwrap() - before,
            built.solid,
        )
    }

    /// The flat `y = 5 - tilt z` and the side of it toward the axis, as a
    /// box turned about `x` through the origin.
    fn toward_axis(topo: &mut Topology, length: f64, tilt: f64) -> SolidId {
        let keep = make_box(topo, 2.0 * WALL, 2.0 * WALL, length + 40.0).unwrap();
        let place =
            Mat4::rotation_x(tilt.atan()) * Mat4::translation(-WALL, FLAT - 2.0 * WALL, -20.0);
        transform_solid(topo, keep, &place).unwrap();
        keep
    }

    /// A block `length` long less the D-shaped hole, its flat leaning
    /// `tilt` off the bore's axis, and the hole's two corners.
    fn d_hole(topo: &mut Topology, length: f64, tilt: f64) -> (SolidId, Vec<EdgeId>) {
        let block = make_box(topo, 24.0, 24.0, length).unwrap();
        transform_solid(topo, block, &Mat4::translation(-12.0, -12.0, 0.0)).unwrap();
        let rod = make_cylinder(topo, WALL, length + 20.0).unwrap();
        transform_solid(topo, rod, &Mat4::translation(0.0, 0.0, -10.0)).unwrap();
        let keep = toward_axis(topo, length, tilt);
        let d = boolean(topo, BooleanOp::Intersect, rod, keep).unwrap();
        let holed = boolean(topo, BooleanOp::Cut, block, d).unwrap();
        let edges = corners(topo, holed, 0.5 * length);
        (holed, edges)
    }

    /// The flat exactly parallel to the bore: each corner takes an exact
    /// cylinder band of radius 1 in the hole, its axis 1 off the flat and 9
    /// from the bore's axis, on the corner's side of the hole.
    #[test]
    fn a_d_holes_corners_round_exactly_where_its_flat_is_parallel_to_the_bore() {
        let length = 100.0;
        let expected = 2.0 * corner_volume(length, 0.0);
        let x_axis = ((WALL - R).powi(2) - (FLAT - R).powi(2)).sqrt();
        for (name, pose) in poses() {
            let what = format!("parallel D hole, {name}");
            let mut topo = Topology::new();
            let (holed, edges) = d_hole(&mut topo, length, 0.0);
            assert_eq!(edges.len(), 2, "{what}");
            let (filled, result) = round(&mut topo, holed, &edges, &pose, &what);
            assert!(
                (filled - expected).abs() < 1e-6 * expected,
                "{what}: filled {filled}, expected {expected}"
            );
            let mut axes: Vec<f64> = Vec::new();
            for face in solid_faces(&topo, result).unwrap() {
                let FaceSurface::Cylinder(band) = topo.face(face).unwrap().surface() else {
                    continue;
                };
                if (band.radius() - WALL).abs() < 1e-9 {
                    continue;
                }
                assert!(
                    (band.radius() - R).abs() < 1e-9,
                    "{what}: a band of {}",
                    band.radius()
                );
                assert!(
                    band.axis().cross(Vec3::new(0.0, 0.0, 1.0)).length() < 1e-9,
                    "{what}: a band not along the bore"
                );
                let o = band.origin();
                assert!(
                    (o.y() - (FLAT - R)).abs() < 1e-7,
                    "{what}: a band's axis at y = {}",
                    o.y()
                );
                axes.push(o.x());
            }
            axes.sort_by(f64::total_cmp);
            assert_eq!(axes.len(), 2, "{what}: bands {axes:?}");
            assert!(
                (axes[0] + x_axis).abs() < 1e-7 && (axes[1] - x_axis).abs() < 1e-7,
                "{what}: bands' axes at x = {axes:?}, expected +-{x_axis}"
            );
        }
    }

    #[test]
    fn a_d_holes_corners_round_where_its_flat_leans_off_the_bore() {
        let length = 1000.0;
        let expected = 2.0 * corner_volume(length, TILT);
        for (name, pose) in poses() {
            let what = format!("D hole, {name}");
            let mut topo = Topology::new();
            let (holed, edges) = d_hole(&mut topo, length, TILT);
            assert_eq!(edges.len(), 2, "{what}");
            let (filled, _) = round(&mut topo, holed, &edges, &pose, &what);
            assert!(
                (filled - expected).abs() < 1e-3 * expected,
                "{what}: filled {filled}, expected {expected}"
            );
        }
    }

    #[test]
    fn a_long_d_rods_corner_rounds_where_its_flat_leans_off_the_wall() {
        let length = 10_000.0;
        let expected = corner_volume(length, TILT);
        for (name, pose) in poses() {
            let what = format!("D rod, {name}");
            let mut topo = Topology::new();
            let rod = make_cylinder(&mut topo, WALL, length).unwrap();
            let keep = toward_axis(&mut topo, length, TILT);
            let d = boolean(&mut topo, BooleanOp::Intersect, rod, keep).unwrap();
            let edges = corners(&topo, d, 0.5 * length);
            assert_eq!(edges.len(), 2, "{what}");
            let removed = -round(&mut topo, d, &edges[..1], &pose, &what).0;
            assert!(
                (removed - expected).abs() < 1e-5 * expected,
                "{what}: took off {removed}, expected {expected}"
            );
        }
    }
}

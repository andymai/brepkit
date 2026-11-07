//! A pocket's inner corner, where three concave rounds meet: a ball in the air
//! tangent to the three walls, closing what the rounds leave.

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use std::f64::consts::PI;

    use brepkit_math::mat::Mat4;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_operations::blend_ops::{all_blended, fillet_v2};
    use brepkit_operations::boolean::{BooleanOp, boolean};
    use brepkit_operations::extrude::extrude;
    use brepkit_operations::measure::solid_volume;
    use brepkit_operations::primitives::make_box;
    use brepkit_operations::tessellate::{is_watertight, tessellate_solid};
    use brepkit_operations::transform::transform_solid;
    use brepkit_operations::validate::validate_solid;
    use brepkit_topology::Topology;
    use brepkit_topology::builder::make_planar_face;
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::explorer::{solid_edges, solid_faces};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::solid::SolidId;

    fn poses() -> [(&'static str, Mat4); 4] {
        [
            ("upright", Mat4::identity()),
            (
                "turned",
                Mat4::translation(3.0, -2.0, 1.0) * Mat4::rotation_x(0.4) * Mat4::rotation_z(0.6),
            ),
            (
                "far away",
                Mat4::translation(370.0, 4830.0, -91.0)
                    * Mat4::rotation_y(1.1)
                    * Mat4::rotation_x(0.3),
            ),
            ("mirrored", Mat4::scale(1.0, -1.0, 1.0)),
        ]
    }

    /// A 40 mm block with a pocket `width` square, its floor at z = 10, open at
    /// the top, and the eight edges inside the pocket.
    fn pocket(topo: &mut Topology, pose: Mat4, width: f64) -> (SolidId, Vec<EdgeId>) {
        let block = make_box(topo, 40.0, 40.0, 40.0).unwrap();
        let tool = make_box(topo, width, width, 40.0).unwrap();
        transform_solid(topo, tool, &Mat4::translation(10.0, 10.0, 10.0)).unwrap();
        let cut = boolean(topo, BooleanOp::Cut, block, tool).unwrap();
        transform_solid(topo, cut, &pose).unwrap();
        let (low, high) = (10.0, 10.0 + width);
        let mid = 10.0 + width / 2.0;
        let middles = [
            [mid, low, low],
            [mid, high, low],
            [low, mid, low],
            [high, mid, low],
            [low, low, 25.0],
            [high, low, 25.0],
            [low, high, 25.0],
            [high, high, 25.0],
        ]
        .map(|[x, y, z]| pose.mul_point(Point3::new(x, y, z)));
        let edges = solid_edges(topo, cut)
            .unwrap()
            .into_iter()
            .filter(|&e| {
                let edge = topo.edge(e).unwrap();
                let (a, b) = (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                );
                let middle = a + (b - a) * 0.5;
                middles.iter().any(|&m| (m - middle).length() < 1e-6)
            })
            .collect();
        (cut, edges)
    }

    /// What a polygon pocket of this `area`, `perimeter` and sum of half-turn
    /// tangents `k`, `depth` deep, takes out once rounded. Within `r` of the floor
    /// each section is the polygon inset by what the ball leaves and rounded by
    /// its radius there; above that, one section rounded by `r`.
    fn rounded_pocket_air(area: f64, perimeter: f64, k: f64, depth: f64, r: f64) -> f64 {
        let floor_layer = (area - perimeter * r + k * r * r) * r
            + (perimeter - 2.0 * k * r) * PI * r * r / 4.0
            + PI * 2.0 * r * r * r / 3.0;
        let section = area - r * r * (k - PI);
        floor_layer + section * (depth - r)
    }

    fn rounded_pocket_volume(width: f64, r: f64) -> f64 {
        64000.0 - rounded_pocket_air(width * width, 4.0 * width, 4.0, 30.0, r)
    }

    /// The area, perimeter and sum of half-turn tangents of a counter-clockwise
    /// polygon.
    fn polygon_measures(outline: &[(f64, f64)]) -> (f64, f64, f64) {
        let n = outline.len();
        let (mut area, mut perimeter, mut k) = (0.0, 0.0, 0.0);
        for i in 0..n {
            let (a, b, c) = (outline[i], outline[(i + 1) % n], outline[(i + 2) % n]);
            area += (a.0 * b.1 - b.0 * a.1) / 2.0;
            perimeter += (b.0 - a.0).hypot(b.1 - a.1);
            let (u, v) = ((b.0 - a.0, b.1 - a.1), (c.0 - b.0, c.1 - b.1));
            k += ((u.0 * v.1 - u.1 * v.0).atan2(u.0 * v.0 + u.1 * v.1) / 2.0).tan();
        }
        (area, perimeter, k)
    }

    /// A pocket whose walls are not square.
    #[test]
    fn a_pocket_off_square_rounds_and_closes() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 60.0, 60.0, 40.0).unwrap();
            let outline = [
                Point3::new(10.0, 10.0, 10.0),
                Point3::new(45.0, 8.0, 10.0),
                Point3::new(50.0, 40.0, 10.0),
                Point3::new(15.0, 48.0, 10.0),
            ];
            let floor = make_planar_face(&mut topo, &outline, 1e-7).unwrap();
            let tool = extrude(&mut topo, floor, Vec3::new(0.0, 0.0, 1.0), 50.0).unwrap();
            let cut = boolean(&mut topo, BooleanOp::Cut, block, tool).unwrap();
            transform_solid(&mut topo, cut, &pose).unwrap();
            let sharp = solid_volume(&topo, cut, 0.01).unwrap();
            let corners = outline.map(|p| pose.mul_point(p));
            let edges: Vec<EdgeId> = solid_edges(&topo, cut)
                .unwrap()
                .into_iter()
                .filter(|&e| {
                    let edge = topo.edge(e).unwrap();
                    let (a, b) = (
                        topo.vertex(edge.start()).unwrap().point(),
                        topo.vertex(edge.end()).unwrap().point(),
                    );
                    corners
                        .iter()
                        .any(|&c| (c - a).length() < 1e-6 || (c - b).length() < 1e-6)
                })
                .collect();
            assert_eq!(edges.len(), 8, "{name}");
            let rounded = all_blended(fillet_v2(&mut topo, cut, &edges, 2.0).unwrap()).unwrap();
            let report = validate_solid(&topo, rounded).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, rounded, 0.02).unwrap();
            assert!(is_watertight(&mesh), "{name}: the mesh leaks");
            let held = solid_volume(&topo, rounded, 0.01).unwrap();
            let (area, perimeter, k) =
                polygon_measures(&[(10.0, 10.0), (45.0, 8.0), (50.0, 40.0), (15.0, 48.0)]);
            let want = 144_000.0 - rounded_pocket_air(area, perimeter, k, 30.0, 2.0);
            assert!(
                (held - want).abs() < 1e-3,
                "{name}: holds {held}, not {want}"
            );
            assert!(held > sharp, "{name}: {held} from {sharp}");
        }
    }

    /// Only the three rounds meeting at one corner, their far ends running out at
    /// the walls and floor there.
    #[test]
    fn three_rounds_at_one_pocket_corner() {
        for (name, pose) in poses() {
            let mut topo = Topology::new();
            let (cut, edges) = pocket(&mut topo, pose, 20.0);
            let corner = pose.mul_point(Point3::new(10.0, 10.0, 10.0));
            let meeting: Vec<EdgeId> = edges
                .into_iter()
                .filter(|&e| {
                    let edge = topo.edge(e).unwrap();
                    (topo.vertex(edge.start()).unwrap().point() - corner).length() < 1e-6
                        || (topo.vertex(edge.end()).unwrap().point() - corner).length() < 1e-6
                })
                .collect();
            assert_eq!(meeting.len(), 3, "{name}");
            let rounded = all_blended(fillet_v2(&mut topo, cut, &meeting, 2.0).unwrap()).unwrap();
            let report = validate_solid(&topo, rounded).unwrap();
            assert!(report.is_valid(), "{name}: {:?}", report.issues);
            let mesh = tessellate_solid(&topo, rounded, 0.02).unwrap();
            assert!(is_watertight(&mesh), "{name}: the mesh leaks");
            let held = solid_volume(&topo, rounded, 0.01).unwrap();
            assert!(held > 52000.0, "{name}: holds {held}");
        }
    }

    #[test]
    fn a_rounded_pocket_holds_its_closed_form_and_closes() {
        for (name, pose) in poses() {
            for (width, r) in [(20.0, 2.0), (20.0, 5.0), (12.0, 1.0)] {
                let case = format!("{name} {width} {r}");
                let mut topo = Topology::new();
                let (cut, edges) = pocket(&mut topo, pose, width);
                assert_eq!(edges.len(), 8, "{case}");
                let rounded = all_blended(fillet_v2(&mut topo, cut, &edges, r).unwrap()).unwrap();
                let report = validate_solid(&topo, rounded).unwrap();
                assert!(report.is_valid(), "{case}: {:?}", report.issues);
                let held = solid_volume(&topo, rounded, 0.01).unwrap();
                let want = rounded_pocket_volume(width, r);
                assert!(
                    (held - want).abs() < 1e-3,
                    "{case}: holds {held}, not {want}"
                );
                let mesh = tessellate_solid(&topo, rounded, 0.02).unwrap();
                assert!(is_watertight(&mesh), "{case}: the mesh leaks");

                let balls = solid_faces(&topo, rounded)
                    .unwrap()
                    .into_iter()
                    .filter(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Sphere(_)))
                    .count();
                assert_eq!(balls, 4, "{case}: one ball at each floor corner");
            }
        }
    }
}

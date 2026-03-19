//! Ray-cast point-in-solid classification.
//!
//! Shoots rays from a sample point and counts boundary crossings
//! to determine inside/outside status. Ported from
//! `operations/boolean/classify.rs`.

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Classify a point by ray casting against the solid's faces.
///
/// Shoots 3 rays (+Z, +X, +Y) and uses majority vote. A point is
/// inside if 2+ rays report an odd crossing count.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate after multiple ray directions.
pub fn classify_ray_cast(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    let face_data = collect_face_polygons(topo, solid)?;

    if face_data.is_empty() {
        return Err(AlgoError::ClassificationFailed(
            "no face polygons collected for ray-cast".into(),
        ));
    }

    let tol = Tolerance::new();
    let ray_dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];

    let mut inside_votes = 0u8;
    for ray_dir in &ray_dirs {
        let mut crossings = 0i32;
        for &(ref verts, normal, d) in &face_data {
            crossings += ray_face_crossing(point, *ray_dir, verts, normal, d, tol);
        }
        if crossings % 2 != 0 {
            inside_votes += 1;
        }
    }

    if inside_votes >= 2 {
        Ok(FaceClass::Inside)
    } else {
        Ok(FaceClass::Outside)
    }
}

/// Collect face polygons from a solid for ray-cast testing.
///
/// For each face, samples boundary edges to get polygon vertices. For
/// curved edges, samples at the midpoint for better coverage.
fn collect_face_polygons(
    topo: &Topology,
    solid: SolidId,
) -> Result<Vec<(Vec<Point3>, Vec3, f64)>, AlgoError> {
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let mut result = Vec::with_capacity(faces.len());

    for fid in faces {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;

        let mut verts = Vec::new();
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            verts.push(start_pos);

            // For curved edges, add a midpoint sample.
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
                let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
                let mid = edge
                    .curve()
                    .evaluate_with_endpoints(t_mid, start_pos, end_pos);
                verts.push(mid);
            }
        }

        if verts.len() < 3 {
            continue;
        }

        // Compute face normal.
        let raw_normal =
            if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
                *normal
            } else {
                newell_normal(&verts)
            };
        let normal = if face.is_reversed() {
            -raw_normal
        } else {
            raw_normal
        };

        let d = dot_normal_point(normal, verts[0]);
        result.push((verts, normal, d));
    }

    Ok(result)
}

/// Test a single face polygon against a ray for crossing parity.
///
/// Returns +1 for a crossing, 0 for no intersection.
#[inline]
fn ray_face_crossing(
    origin: Point3,
    ray_dir: Vec3,
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> i32 {
    let denom = normal.dot(ray_dir);
    if denom.abs() < tol.angular {
        return 0;
    }
    let numer = d - dot_normal_point(normal, origin);
    let t = numer / denom;
    if t <= tol.linear {
        return 0;
    }
    let hit = Point3::new(
        origin.x() + ray_dir.x() * t,
        origin.y() + ray_dir.y() * t,
        origin.z() + ray_dir.z() * t,
    );
    if point_in_face_3d(hit, verts, &normal) {
        1
    } else {
        0
    }
}

/// Test if a 3D point lies inside a planar face polygon by projecting to 2D.
fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (project_point, project_polygon): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        (
            Point2::new(point.x(), point.y()),
            polygon.iter().map(|p| Point2::new(p.x(), p.y())).collect(),
        )
    } else if ay >= ax {
        (
            Point2::new(point.x(), point.z()),
            polygon.iter().map(|p| Point2::new(p.x(), p.z())).collect(),
        )
    } else {
        (
            Point2::new(point.y(), point.z()),
            polygon.iter().map(|p| Point2::new(p.y(), p.z())).collect(),
        )
    };

    point_in_polygon(project_point, &project_polygon)
}

/// Compute `n . p` treating a `Point3` as a direction vector.
fn dot_normal_point(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// Compute the solid-level AABB from boundary vertices.
pub fn compute_solid_bbox(
    topo: &Topology,
    solid: SolidId,
) -> Result<brepkit_math::aabb::Aabb3, AlgoError> {
    let mut points = Vec::new();
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    for fid in faces {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            points.push(start_pos);
            points.push(end_pos);
            // Curved edges can bulge beyond their endpoints
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
                let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
                let mid = edge
                    .curve()
                    .evaluate_with_endpoints(t_mid, start_pos, end_pos);
                points.push(mid);
            }
        }
    }
    brepkit_math::aabb::Aabb3::try_from_points(points)
        .ok_or_else(|| AlgoError::ClassificationFailed("solid has no boundary vertices".into()))
}

/// Compute polygon normal via Newell's method.
fn newell_normal(verts: &[Point3]) -> Vec3 {
    let n = verts.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let curr = verts[i];
        let next = verts[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-15 {
        Vec3::new(nx / len, ny / len, nz / len)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Build a degenerate solid where all faces have < 3 vertices
    /// (single-edge faces). This tests the empty polygon fallback.
    fn make_degenerate_solid(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
        // Create a "solid" with a single face that has only 2 vertices
        // (a degenerate line edge). This will produce < 3 polygon vertices.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e10 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e01, true), OrientedEdge::new(e10, true)],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn empty_face_polygons_returns_error() {
        let mut topo = Topology::default();
        let solid = make_degenerate_solid(&mut topo);

        let result = classify_ray_cast(&topo, solid, Point3::new(0.5, 0.5, 0.5));
        assert!(
            result.is_err(),
            "ray-cast with no valid face polygons should return Err, got {result:?}"
        );
    }

    /// Build a unit box for classification tests.
    fn make_box(
        topo: &mut Topology,
        min: [f64; 3],
        max: [f64; 3],
    ) -> brepkit_topology::solid::SolidId {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let v = [
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z1), 1e-7)),
        ];
        let mut edge = |a: usize, b: usize| -> brepkit_topology::edge::EdgeId {
            topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line))
        };
        let e01 = edge(0, 1);
        let e12 = edge(1, 2);
        let e23 = edge(2, 3);
        let e30 = edge(3, 0);
        let e45 = edge(4, 5);
        let e56 = edge(5, 6);
        let e67 = edge(6, 7);
        let e74 = edge(7, 4);
        let e04 = edge(0, 4);
        let e15 = edge(1, 5);
        let e26 = edge(2, 6);
        let e37 = edge(3, 7);

        let fwd = |eid| OrientedEdge::new(eid, true);
        let w_bot =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e12), fwd(e23), fwd(e30)], true).unwrap());
        let w_top =
            topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
        let w_front =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), fwd(e45), fwd(e04)], true).unwrap());
        let w_back =
            topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), fwd(e67), fwd(e26)], true).unwrap());
        let w_left =
            topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), fwd(e74), fwd(e37)], true).unwrap());
        let w_right =
            topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), fwd(e56), fwd(e15)], true).unwrap());

        let mk_face =
            |w, n: Vec3, d: f64| Face::new(w, vec![], FaceSurface::Plane { normal: n, d });
        let faces = vec![
            topo.add_face(mk_face(w_bot, Vec3::new(0.0, 0.0, -1.0), -z0)),
            topo.add_face(mk_face(w_top, Vec3::new(0.0, 0.0, 1.0), z1)),
            topo.add_face(mk_face(w_front, Vec3::new(0.0, -1.0, 0.0), -y0)),
            topo.add_face(mk_face(w_back, Vec3::new(0.0, 1.0, 0.0), y1)),
            topo.add_face(mk_face(w_left, Vec3::new(-1.0, 0.0, 0.0), -x0)),
            topo.add_face(mk_face(w_right, Vec3::new(1.0, 0.0, 0.0), x1)),
        ];
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn ray_cast_classifies_inside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(1.0, 1.0, 1.0)).unwrap();
        assert_eq!(result, FaceClass::Inside, "center of box should be Inside");
    }

    #[test]
    fn ray_cast_classifies_outside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(5.0, 5.0, 5.0)).unwrap();
        assert_eq!(
            result,
            FaceClass::Outside,
            "point far from box should be Outside"
        );
    }
}

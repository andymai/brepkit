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
        return Ok(FaceClass::Outside);
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

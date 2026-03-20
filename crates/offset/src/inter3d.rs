//! 3D intersection of adjacent offset faces.

use brepkit_math::analytic_intersection::{
    AnalyticSurface, intersect_analytic_analytic, intersect_plane_analytic,
};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::data::{FaceIntersection, OffsetData};
use crate::error::OffsetError;

/// Intersect pairs of adjacent offset faces in 3D to find new edge curves.
///
/// For each manifold edge shared by two offset faces, compute the 3D
/// intersection curve of their offset surfaces and store sampled points
/// in [`OffsetData::intersections`].
///
/// # Errors
///
/// Returns [`OffsetError::IntersectionFailed`] if a face pair cannot be intersected.
#[allow(clippy::too_many_lines)]
pub fn intersect_faces_3d(
    topo: &Topology,
    solid: SolidId,
    data: &mut OffsetData,
) -> Result<(), OffsetError> {
    let edge_face_map = brepkit_topology::explorer::edge_to_face_map(topo, solid)?;

    for (&edge_idx, face_ids) in &edge_face_map {
        // Only process manifold edges (shared by exactly 2 faces).
        if face_ids.len() != 2 {
            continue;
        }

        let face_a = face_ids[0];
        let face_b = face_ids[1];

        // Both faces must have offset surfaces.
        let (Some(off_a), Some(off_b)) = (
            data.offset_faces.get(&face_a),
            data.offset_faces.get(&face_b),
        ) else {
            continue;
        };

        // Skip excluded faces.
        if data.excluded_faces.contains(&face_a) || data.excluded_faces.contains(&face_b) {
            continue;
        }

        let edge_id =
            topo.edge_id_from_index(edge_idx)
                .ok_or_else(|| OffsetError::InvalidInput {
                    reason: format!("edge index {edge_idx} not found in arena"),
                })?;

        let surf_a = &off_a.surface;
        let surf_b = &off_b.surface;

        let curve_points = intersect_surface_pair(topo, face_a, face_b, surf_a, surf_b)?;

        data.intersections.push(FaceIntersection {
            original_edge: edge_id,
            face_a,
            face_b,
            curve_points,
            new_edges: Vec::new(),
        });
    }

    Ok(())
}

/// Grid resolution for analytic-analytic intersection marching.
const ANALYTIC_GRID_RES: usize = 32;

/// Dispatch intersection based on surface types.
#[allow(clippy::too_many_lines)]
fn intersect_surface_pair(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
) -> Result<Vec<Point3>, OffsetError> {
    // Plane-Plane: exact line intersection.
    if let (FaceSurface::Plane { normal: n1, d: d1 }, FaceSurface::Plane { normal: n2, d: d2 }) =
        (surf_a, surf_b)
    {
        return intersect_plane_plane(topo, face_a, face_b, *n1, *d1, *n2, *d2);
    }

    // Plane-Analytic or Analytic-Plane.
    if let Some(pts) = try_plane_analytic(surf_a, surf_b)? {
        return Ok(pts);
    }
    if let Some(pts) = try_plane_analytic(surf_b, surf_a)? {
        return Ok(pts);
    }

    // Analytic-Analytic.
    if let (Some(a), Some(b)) = (to_analytic(surf_a), to_analytic(surf_b)) {
        let curves = intersect_analytic_analytic(a, b, ANALYTIC_GRID_RES).map_err(|e| {
            OffsetError::IntersectionFailed {
                face_a,
                face_b,
                reason: format!("analytic-analytic intersection: {e}"),
            }
        })?;
        return Ok(extract_points(&curves));
    }

    // NURBS fallback: convert both to NURBS and intersect.
    // TODO: implement NURBS-NURBS intersection path.
    log::warn!(
        "inter3d: NURBS intersection not yet implemented for faces {:?}/{:?}",
        face_a,
        face_b
    );
    Ok(Vec::new())
}

/// Try Plane-Analytic intersection. Returns Some if surf_a is a Plane
/// and surf_b is an analytic (non-plane, non-NURBS) surface.
fn try_plane_analytic(
    surf_a: &FaceSurface,
    surf_b: &FaceSurface,
) -> Result<Option<Vec<Point3>>, OffsetError> {
    let FaceSurface::Plane { normal, d } = surf_a else {
        return Ok(None);
    };
    let Some(analytic) = to_analytic(surf_b) else {
        return Ok(None);
    };
    let curves =
        intersect_plane_analytic(analytic, *normal, *d).map_err(|e| OffsetError::InvalidInput {
            reason: format!("plane-analytic intersection: {e}"),
        })?;
    Ok(Some(extract_points(&curves)))
}

/// Convert a `FaceSurface` to an `AnalyticSurface` if applicable.
fn to_analytic(surf: &FaceSurface) -> Option<AnalyticSurface<'_>> {
    match surf {
        FaceSurface::Cylinder(c) => Some(AnalyticSurface::Cylinder(c)),
        FaceSurface::Cone(c) => Some(AnalyticSurface::Cone(c)),
        FaceSurface::Sphere(s) => Some(AnalyticSurface::Sphere(s)),
        FaceSurface::Torus(t) => Some(AnalyticSurface::Torus(t)),
        _ => None,
    }
}

/// Extract 3D points from intersection curve results.
fn extract_points(curves: &[brepkit_math::nurbs::intersection::IntersectionCurve]) -> Vec<Point3> {
    curves
        .iter()
        .flat_map(|c| c.points.iter().map(|p| p.point))
        .collect()
}

/// Intersect two planes and sample the intersection line within the bounding
/// region of the two faces.
///
/// Given planes `n1 · x = d1` and `n2 · x = d2`, the intersection line
/// direction is `dir = n1 × n2`. A point on the line is found by solving the
/// 2-equation system with one coordinate fixed to 0 (choosing the axis where
/// `dir` has the largest component for numerical stability).
#[allow(clippy::similar_names)]
fn intersect_plane_plane(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    n1: Vec3,
    d1: f64,
    n2: Vec3,
    d2: f64,
) -> Result<Vec<Point3>, OffsetError> {
    let dir = n1.cross(n2);
    let dir_len = dir.length();

    // Parallel or near-parallel planes — no intersection.
    if dir_len < 1e-10 {
        return Ok(Vec::new());
    }

    let dir_norm = Vec3::new(dir.x() / dir_len, dir.y() / dir_len, dir.z() / dir_len);

    // Find a point on the intersection line by setting the coordinate
    // corresponding to the largest component of `dir` to zero and solving
    // the remaining 2×2 system.
    let origin = find_point_on_line(n1, d1, n2, d2, &dir);

    // Compute the bounding parameter range by projecting all face vertices
    // onto the intersection line.
    let (t_min, t_max) = face_vertex_range(topo, face_a, face_b, &origin, &dir_norm)?;

    // Sample the intersection line.
    let num_samples: usize = 10;
    let mut points = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        #[allow(clippy::cast_precision_loss)]
        let t = t_min + (t_max - t_min) * (i as f64) / ((num_samples - 1) as f64);
        points.push(Point3::new(
            origin.x() + t * dir_norm.x(),
            origin.y() + t * dir_norm.y(),
            origin.z() + t * dir_norm.z(),
        ));
    }

    Ok(points)
}

/// Find a point on the intersection of two planes by solving a 2×2 system.
///
/// We set the coordinate where `dir = n1 × n2` is largest to zero and solve
/// the remaining two equations.
fn find_point_on_line(n1: Vec3, d1: f64, n2: Vec3, d2: f64, dir: &Vec3) -> Point3 {
    let ax = dir.x().abs();
    let ay = dir.y().abs();
    let az = dir.z().abs();

    // Choose the axis with the largest |dir| component — set it to 0,
    // solve for the other two.
    if az >= ax && az >= ay {
        let det = n1.x() * n2.y() - n1.y() * n2.x();
        let x = (d1 * n2.y() - d2 * n1.y()) / det;
        let y = (n1.x() * d2 - n2.x() * d1) / det;
        Point3::new(x, y, 0.0)
    } else if ay >= ax {
        let det = n1.x() * n2.z() - n1.z() * n2.x();
        let x = (d1 * n2.z() - d2 * n1.z()) / det;
        let z = (n1.x() * d2 - n2.x() * d1) / det;
        Point3::new(x, 0.0, z)
    } else {
        let det = n1.y() * n2.z() - n1.z() * n2.y();
        let y = (d1 * n2.z() - d2 * n1.z()) / det;
        let z = (n1.y() * d2 - n2.y() * d1) / det;
        Point3::new(0.0, y, z)
    }
}

/// Compute the parameter range along the intersection line by projecting
/// all vertices of both faces onto the line.
fn face_vertex_range(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    origin: &Point3,
    dir: &Vec3,
) -> Result<(f64, f64), OffsetError> {
    let mut t_min = f64::INFINITY;
    let mut t_max = f64::NEG_INFINITY;

    for &face_id in &[face_a, face_b] {
        let face = topo.face(face_id)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let p = topo.vertex(edge.start())?.point();
            let t = project_onto_line(origin, dir, &p);
            if t < t_min {
                t_min = t;
            }
            if t > t_max {
                t_max = t;
            }
        }
    }

    // Add a small margin to avoid missing endpoints.
    let margin = (t_max - t_min) * 0.01;
    Ok((t_min - margin, t_max + margin))
}

/// Project a point onto a line defined by `origin + t * dir`, returning `t`.
fn project_onto_line(origin: &Point3, dir: &Vec3, point: &Point3) -> f64 {
    let dx = point.x() - origin.x();
    let dy = point.y() - origin.y();
    let dz = point.z() - origin.z();
    dx * dir.x() + dy * dir.y() + dz * dir.z()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::data::{OffsetData, OffsetOptions};
    use brepkit_topology::Topology;

    fn run_phases_1_2_3(topo: &Topology, solid: SolidId, distance: f64) -> OffsetData {
        let mut data = OffsetData::new(distance, OffsetOptions::default(), vec![]);
        crate::analyse::analyse_edges(topo, solid, &mut data).unwrap();
        crate::offset::build_offset_faces(topo, solid, &mut data).unwrap();
        intersect_faces_3d(topo, solid, &mut data).unwrap();
        data
    }

    #[test]
    fn box_offset_produces_12_intersections() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_2_3(&topo, solid, 0.5);
        assert_eq!(
            data.intersections.len(),
            12,
            "box offset should produce 12 face-face intersections (one per edge)"
        );
    }

    #[test]
    fn box_intersection_curves_are_nonempty() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let data = run_phases_1_2_3(&topo, solid, 0.5);
        for fi in &data.intersections {
            assert!(
                !fi.curve_points.is_empty(),
                "intersection for edge {:?} should have points",
                fi.original_edge
            );
        }
    }
}

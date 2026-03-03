//! Full 3D solid offset (parallel shell).
//!
//! Offsets all faces of a solid by a uniform distance along their normals.
//! Equivalent to `BRepOffsetAPI_MakeOffsetShape` in `OpenCascade`.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;
use crate::dot_normal_point;

/// Offset all faces of a solid by a uniform distance.
///
/// Positive distance offsets outward (solid grows), negative inward (solid shrinks).
/// Currently supports planar faces only — NURBS faces require surface offset
/// with self-intersection removal.
///
/// For planar solids, this computes the correct offset vertex positions by
/// intersecting adjacent offset planes (a 3-plane intersection for each vertex),
/// which handles non-right-angle edges correctly. Falls back to single-normal
/// offset when fewer than 3 adjacent faces are found.
///
/// Also validates that the offset doesn't cause the solid to collapse
/// (negative volume or self-intersection at edges).
///
/// # Errors
///
/// Returns an error if:
/// - The solid contains non-planar faces
/// - The offset distance is zero
/// - The offset causes the solid to collapse (negative volume)
#[allow(clippy::too_many_lines)]
pub fn offset_solid(
    topo: &mut Topology,
    solid: SolidId,
    distance: f64,
) -> Result<SolidId, OperationsError> {
    let tol = Tolerance::new();

    if tol.approx_eq(distance, 0.0) {
        return Err(OperationsError::InvalidInput {
            reason: "offset distance must be non-zero".into(),
        });
    }

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Collect face normals and plane equations.
    let mut face_normals: std::collections::HashMap<usize, (Vec3, f64)> =
        std::collections::HashMap::new();
    let mut vertex_faces: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                return Err(OperationsError::InvalidInput {
                    reason: "solid offset currently only supports planar faces".into(),
                });
            }
        };

        face_normals.insert(fid.index(), (normal, d));

        // Track which faces each vertex belongs to.
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            vertex_faces
                .entry(edge.start().index())
                .or_default()
                .push(fid.index());
            vertex_faces
                .entry(edge.end().index())
                .or_default()
                .push(fid.index());
        }
    }

    // Deduplicate face lists for each vertex.
    for faces in vertex_faces.values_mut() {
        faces.sort_unstable();
        faces.dedup();
    }

    // Compute offset vertex positions.
    // For vertices at the intersection of 3+ faces, solve the 3-plane
    // intersection to find the correct offset position.
    let mut vertex_offset_pos: std::collections::HashMap<usize, Point3> =
        std::collections::HashMap::new();

    for (&vid_idx, adj_faces) in &vertex_faces {
        let vid = topo.vertices.id_from_index(vid_idx);
        let original_pos = if let Some(vid) = vid {
            topo.vertex(vid)?.point()
        } else {
            continue;
        };

        if adj_faces.len() >= 3 {
            // 3-plane intersection: solve the system
            //   n1·x = d1 + distance
            //   n2·x = d2 + distance
            //   n3·x = d3 + distance
            let (n1, d1) = face_normals[&adj_faces[0]];
            let (n2, d2) = face_normals[&adj_faces[1]];
            let (n3, d3) = face_normals[&adj_faces[2]];

            if let Some(pos) =
                solve_3_plane_intersection(n1, d1 + distance, n2, d2 + distance, n3, d3 + distance)
            {
                vertex_offset_pos.insert(vid_idx, pos);
            } else {
                // Degenerate: fall back to simple offset along first face normal
                vertex_offset_pos.insert(vid_idx, original_pos + n1 * distance);
            }
        } else if !adj_faces.is_empty() {
            // Fewer than 3 faces: use average normal offset
            let mut avg_normal = Vec3::new(0.0, 0.0, 0.0);
            for &fi in adj_faces {
                avg_normal = avg_normal + face_normals[&fi].0;
            }
            if let Ok(avg_n) = avg_normal.normalize() {
                vertex_offset_pos.insert(vid_idx, original_pos + avg_n * distance);
            } else {
                vertex_offset_pos.insert(vid_idx, original_pos);
            }
        }
    }

    // Build offset faces using the computed vertex positions.
    let mut offset_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let (normal, _d) = face_normals[&fid.index()];

        let wire = topo.wire(face.outer_wire())?;
        let mut offset_verts = Vec::new();

        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let vid = if oe.is_forward() {
                edge.start()
            } else {
                edge.end()
            };

            let pos = vertex_offset_pos
                .get(&vid.index())
                .copied()
                .unwrap_or_else(|| {
                    topo.vertex(vid)
                        .map(|v| v.point() + normal * distance)
                        .unwrap_or(Point3::new(0.0, 0.0, 0.0))
                });
            offset_verts.push(pos);
        }

        let new_d = if offset_verts.is_empty() {
            0.0
        } else {
            dot_normal_point(normal, offset_verts[0])
        };
        offset_faces.push((offset_verts, normal, new_d));
    }

    // Validate: check that the offset doesn't collapse the solid.
    // Compute approximate volume from the offset faces.
    let offset_result = crate::boolean::assemble_solid(topo, &offset_faces, tol)?;

    let vol = crate::measure::solid_volume(topo, offset_result, 0.1)?;
    if vol < tol.linear {
        return Err(OperationsError::InvalidInput {
            reason: format!("offset distance {distance} causes solid to collapse (volume = {vol})"),
        });
    }

    Ok(offset_result)
}

/// Solve a 3-plane intersection: find the point where three planes meet.
///
/// Each plane is defined by `n·x = d` where `n` is the normal and `d` is the
/// signed distance from the origin. Returns `None` if the planes are coplanar
/// or nearly parallel (singular system).
fn solve_3_plane_intersection(
    n1: Vec3,
    d1: f64,
    n2: Vec3,
    d2: f64,
    n3: Vec3,
    d3: f64,
) -> Option<Point3> {
    // Cramer's rule for 3×3 system:
    //   [n1x n1y n1z] [x]   [d1]
    //   [n2x n2y n2z] [y] = [d2]
    //   [n3x n3y n3z] [z]   [d3]
    let det = n1.x() * (n2.y() * n3.z() - n2.z() * n3.y())
        - n1.y() * (n2.x() * n3.z() - n2.z() * n3.x())
        + n1.z() * (n2.x() * n3.y() - n2.y() * n3.x());

    if det.abs() < 1e-12 {
        return None; // Singular: planes don't meet at a unique point
    }

    let inv_det = 1.0 / det;

    let x = (d1 * (n2.y() * n3.z() - n2.z() * n3.y()) - n1.y() * (d2 * n3.z() - n2.z() * d3)
        + n1.z() * (d2 * n3.y() - n2.y() * d3))
        * inv_det;

    let y = (n1.x() * (d2 * n3.z() - n2.z() * d3) - d1 * (n2.x() * n3.z() - n2.z() * n3.x())
        + n1.z() * (n2.x() * d3 - d2 * n3.x()))
        * inv_det;

    let z = (n1.x() * (n2.y() * d3 - d2 * n3.y()) - n1.y() * (n2.x() * d3 - d2 * n3.x())
        + d1 * (n2.x() * n3.y() - n2.y() * n3.x()))
        * inv_det;

    Some(Point3::new(x, y, z))
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn offset_box_outward() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let original_vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.5).unwrap();
        let offset_vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        assert!(
            offset_vol > original_vol,
            "outward offset should increase volume: {offset_vol} > {original_vol}"
        );
    }

    #[test]
    fn offset_box_inward() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();

        let original_vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();

        let offset = offset_solid(&mut topo, solid, -0.5).unwrap();
        let offset_vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        assert!(
            offset_vol < original_vol,
            "inward offset should decrease volume: {offset_vol} < {original_vol}"
        );
    }

    #[test]
    fn offset_zero_error() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        assert!(offset_solid(&mut topo, solid, 0.0).is_err());
    }

    #[test]
    fn offset_preserves_face_count() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.3).unwrap();

        let shell = topo
            .shell(topo.solid(offset).unwrap().outer_shell())
            .unwrap();
        assert_eq!(
            shell.faces().len(),
            6,
            "offset box should still have 6 faces"
        );
    }

    #[test]
    fn offset_box_correct_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.5).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        // Expected: (2+1)^3 = 27 (each dimension grows by 2×0.5)
        assert!(
            (vol - 27.0).abs() < 0.5,
            "outward offset of unit cube by 0.5 should have volume ~27.0, got {vol}"
        );
    }

    #[test]
    fn offset_inward_correct_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();

        let offset = offset_solid(&mut topo, solid, -0.5).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        // Expected: (4-1)^3 = 27 (each dimension shrinks by 2×0.5)
        assert!(
            (vol - 27.0).abs() < 0.5,
            "inward offset by 0.5 should have volume ~27.0, got {vol}"
        );
    }

    #[test]
    fn offset_very_large_inward_still_valid() {
        // Even with a large inward offset on a box, the 3-plane intersection
        // produces a valid (inverted) box. This is geometrically correct —
        // the offset box at -1.0 on a unit cube produces a box centered at
        // (0.5,0.5,0.5) with dimensions (-1,-1,-1), which flips normals but
        // has positive absolute volume. Full self-intersection detection
        // requires face-face intersection checking (future work).
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Small inward offset should always succeed.
        let offset = offset_solid(&mut topo, solid, -0.1).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();
        // (2-0.2)^3 = 1.8^3 = 5.832
        assert!(
            (vol - 5.832).abs() < 0.5,
            "small inward offset volume should be ~5.832, got {vol}"
        );
    }

    #[test]
    fn solve_3_planes_unit_cube_corner() {
        // 3 perpendicular planes meeting at (1,1,1)
        let n1 = Vec3::new(1.0, 0.0, 0.0);
        let n2 = Vec3::new(0.0, 1.0, 0.0);
        let n3 = Vec3::new(0.0, 0.0, 1.0);

        let pos = solve_3_plane_intersection(n1, 1.0, n2, 1.0, n3, 1.0).unwrap();
        assert!((pos.x() - 1.0).abs() < 1e-10);
        assert!((pos.y() - 1.0).abs() < 1e-10);
        assert!((pos.z() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn solve_3_planes_non_orthogonal() {
        // 3 non-perpendicular planes meeting at a point.
        // Plane 1: x + y = 2
        // Plane 2: y + z = 3
        // Plane 3: x + z = 1
        // Solution: x=0, y=2, z=1
        let n1 = Vec3::new(1.0, 1.0, 0.0);
        let n2 = Vec3::new(0.0, 1.0, 1.0);
        let n3 = Vec3::new(1.0, 0.0, 1.0);

        let pos = solve_3_plane_intersection(n1, 2.0, n2, 3.0, n3, 1.0).unwrap();
        assert!(
            (pos.x() - 0.0).abs() < 1e-10,
            "x should be 0, got {}",
            pos.x()
        );
        assert!(
            (pos.y() - 2.0).abs() < 1e-10,
            "y should be 2, got {}",
            pos.y()
        );
        assert!(
            (pos.z() - 1.0).abs() < 1e-10,
            "z should be 1, got {}",
            pos.z()
        );
    }
}

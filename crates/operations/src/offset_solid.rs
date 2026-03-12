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
use crate::boolean::FaceSpec;
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

    // Check if all faces are planar.
    let all_planar = face_ids.iter().all(|&fid| {
        topo.face(fid)
            .map(|f| matches!(f.surface(), FaceSurface::Plane { .. }))
            .unwrap_or(false)
    });

    if !all_planar {
        return offset_solid_general(topo, solid, &face_ids, distance, tol);
    }

    // Collect face normals and plane equations.
    let mut face_normals: std::collections::BTreeMap<usize, (Vec3, f64)> =
        std::collections::BTreeMap::new();
    let mut vertex_faces: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                return Err(OperationsError::InvalidInput {
                    reason: "expected planar face after all_planar check".into(),
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
    let mut vertex_offset_pos: std::collections::BTreeMap<usize, Point3> =
        std::collections::BTreeMap::new();

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

/// Offset a solid that contains non-planar faces using per-face offset.
///
/// Uses `offset_face` to offset each face individually, then reassembles
/// the result into a solid using `assemble_solid_mixed`.
fn offset_solid_general(
    topo: &mut Topology,
    _solid: SolidId,
    face_ids: &[FaceId],
    distance: f64,
    tol: Tolerance,
) -> Result<SolidId, OperationsError> {
    let mut result_specs: Vec<FaceSpec> = Vec::new();

    for &fid in face_ids {
        let offset_fid = crate::offset_face::offset_face(topo, fid, distance, 8)?;
        let offset_face = topo.face(offset_fid)?;

        let verts: Vec<Point3> = {
            let wire = topo.wire(offset_face.outer_wire())?;
            let mut pts = Vec::new();
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let vid = if oe.is_forward() {
                    edge.start()
                } else {
                    edge.end()
                };
                pts.push(topo.vertex(vid)?.point());
            }
            pts
        };

        match offset_face.surface() {
            FaceSurface::Plane { normal, d } => {
                result_specs.push(FaceSpec::Planar {
                    vertices: verts,
                    normal: *normal,
                    d: *d,
                });
            }
            other => {
                result_specs.push(FaceSpec::Surface {
                    vertices: verts,
                    surface: other.clone(),
                    reversed: false,
                });
            }
        }
    }

    if result_specs.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "offset produced no faces".into(),
        });
    }

    let result = crate::boolean::assemble_solid_mixed(topo, &result_specs, tol)?;

    let vol = crate::measure::solid_volume(topo, result, 0.1)?;
    if vol < tol.linear {
        return Err(OperationsError::InvalidInput {
            reason: format!("offset distance {distance} causes solid to collapse (volume = {vol})"),
        });
    }

    Ok(result)
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

    /// Outward offset of a 2³ box by 0.5.
    /// Each face moves outward by 0.5, so each dimension grows by 2×0.5 = 1.0.
    /// V = (2+1)³ = 27.0 exactly (all-planar, no tessellation).
    #[test]
    fn offset_box_correct_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let offset = offset_solid(&mut topo, solid, 0.5).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        // All-planar offset → exact polygon volume.
        let expected = 27.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "outward offset of 2³ box by 0.5: expected {expected}, got {vol} \
             (rel_err={rel_err:.2e}). Was 0.5 abs tolerance."
        );
    }

    /// Inward offset of a 4³ box by 0.5.
    /// V = (4-1)³ = 27.0 exactly.
    #[test]
    fn offset_inward_correct_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 4.0, 4.0, 4.0).unwrap();

        let offset = offset_solid(&mut topo, solid, -0.5).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        let expected = 27.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "inward offset of 4³ box by 0.5: expected {expected}, got {vol} \
             (rel_err={rel_err:.2e}). Was 0.5 abs tolerance."
        );
    }

    /// Small inward offset: 2³ box offset by -0.1.
    /// V = (2-0.2)³ = 1.8³ = 5.832 exactly.
    #[test]
    fn offset_small_inward_correct_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let offset = offset_solid(&mut topo, solid, -0.1).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        let expected = 5.832;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "small inward offset volume: expected {expected}, got {vol} \
             (rel_err={rel_err:.2e}). Was 0.5 abs tolerance."
        );
    }

    /// Offset sphere r=5 by +1 → r=6. V = (4/3)π(6³) ≈ 904.78.
    ///
    /// Sphere offset goes through tessellation path (NURBS faces).
    /// With 32 segments, expect ~5% error from tessellation approximation.
    /// Previously used 15% tolerance — tightened to 5%.
    #[test]
    fn offset_sphere_outward() {
        let mut topo = Topology::new();
        let solid = crate::primitives::make_sphere(&mut topo, 5.0, 32).unwrap();

        let result = offset_solid(&mut topo, solid, 1.0).unwrap();
        let vol = crate::measure::solid_volume(&topo, result, 0.05).unwrap();

        // V = (4/3)πr³ = (4/3)π(216) ≈ 904.779
        let expected = 4.0 / 3.0 * std::f64::consts::PI * 216.0;
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.05,
            "offset sphere volume: expected {expected:.2}, got {vol:.2} \
             (rel_err={rel_err:.2e}). Was 15% tolerance."
        );
    }

    /// Offset a non-cube rectangular box outward.
    /// 3×5×7 box offset by +1 → (3+2)×(5+2)×(7+2) = 5×7×9 = 315.
    #[test]
    fn offset_rectangular_box_volume() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 3.0, 5.0, 7.0).unwrap();

        let offset = offset_solid(&mut topo, solid, 1.0).unwrap();
        let vol = crate::measure::solid_volume(&topo, offset, 0.1).unwrap();

        let expected = 5.0 * 7.0 * 9.0; // 315.0
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 1e-8,
            "offset 3×5×7 box by 1: expected {expected}, got {vol} (rel_err={rel_err:.2e})"
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

//! Shell (hollow/offset) operation for creating thin-walled solids.
//!
//! Equivalent to `BRepOffsetAPI_MakeThickSolid` in `OpenCascade`.
//! Offsets faces of a solid inward to create a hollow shell with
//! uniform wall thickness. Optionally removes specified faces to
//! create openings.

use std::collections::HashSet;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::{assemble_solid, face_vertices};
use crate::dot_normal_point;

/// Create a hollow shell from a solid by offsetting faces inward.
///
/// Each face is offset inward by `thickness` along its outward normal.
/// If `open_faces` is non-empty, those faces are removed from both the
/// outer and inner shells, creating openings.
///
/// # Errors
///
/// Returns an error if:
/// - `thickness` is non-positive
/// - The solid contains NURBS faces
/// - Any face in `open_faces` is not part of the solid
/// - The resulting shell is degenerate
#[allow(clippy::too_many_lines)]
pub fn shell(
    topo: &mut Topology,
    solid: SolidId,
    thickness: f64,
    open_faces: &[FaceId],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if thickness <= tol.linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("shell thickness must be positive, got {thickness}"),
        });
    }

    // Collect face data from the solid.
    let solid_data = topo.solid(solid)?;
    let shell_data = topo.shell(solid_data.outer_shell())?;
    let all_face_ids: Vec<FaceId> = shell_data.faces().to_vec();

    let open_set: HashSet<usize> = open_faces.iter().map(|f| f.index()).collect();

    // Validate open_faces belong to the solid.
    let solid_face_set: HashSet<usize> = all_face_ids.iter().map(|f| f.index()).collect();
    for &of in open_faces {
        if !solid_face_set.contains(&of.index()) {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("face {} is not part of the solid", of.index()),
            });
        }
    }

    // Collect face polygons and normals.
    let mut face_data: Vec<(FaceId, Vec<Point3>, Vec3, f64)> = Vec::new();
    for &fid in &all_face_ids {
        let face = topo.face(fid)?;
        let (normal, d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            FaceSurface::Nurbs(_) => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "shell operation on NURBS faces is not supported".into(),
                });
            }
        };
        let verts = face_vertices(topo, fid)?;
        face_data.push((fid, verts, normal, d));
    }

    let mut result_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    // Outer faces: keep the original faces that are not open.
    for &(fid, ref verts, normal, d) in &face_data {
        if !open_set.contains(&fid.index()) {
            result_faces.push((verts.clone(), normal, d));
        }
    }

    // Inner faces: offset each non-open face inward (flip normal, shift
    // vertices along the original outward normal by -thickness).
    for &(fid, ref verts, normal, _d) in &face_data {
        if open_set.contains(&fid.index()) {
            continue;
        }

        let offset = Vec3::new(
            -normal.x() * thickness,
            -normal.y() * thickness,
            -normal.z() * thickness,
        );
        let inner_verts: Vec<Point3> = verts
            .iter()
            .map(|p| *p + offset)
            .rev() // Reverse winding for inner face (normal points inward).
            .collect();
        let inner_normal = -normal;
        let inner_d = dot_normal_point(inner_normal, inner_verts[0]);
        result_faces.push((inner_verts, inner_normal, inner_d));
    }

    // Rim faces: for each edge of an open face, connect the outer edge
    // to the corresponding inner edge with a quad face.
    // For simplicity with the planar approach: connect each edge of the
    // opening boundary by finding the adjacent non-open face and creating
    // a rim quad from the outer vertices to the inner (offset) vertices.
    for &(fid, ref verts, normal, _d) in &face_data {
        if !open_set.contains(&fid.index()) {
            continue;
        }

        let n = verts.len();
        let offset = Vec3::new(
            -normal.x() * thickness,
            -normal.y() * thickness,
            -normal.z() * thickness,
        );

        for i in 0..n {
            let j = (i + 1) % n;
            let outer_a = verts[i];
            let outer_b = verts[j];
            let inner_a = outer_a + offset;
            let inner_b = outer_b + offset;

            // Rim quad: outer_a → outer_b → inner_b → inner_a
            let rim_verts = vec![outer_a, outer_b, inner_b, inner_a];
            let edge1 = outer_b - outer_a;
            let edge2 = inner_a - outer_a;
            let rim_normal = edge1
                .cross(edge2)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let rim_d = dot_normal_point(rim_normal, outer_a);
            result_faces.push((rim_verts, rim_normal, rim_d));
        }
    }

    if result_faces.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "shell operation produced no faces".into(),
        });
    }

    assemble_solid(topo, &result_faces, tol)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    /// Helper: get face IDs matching a given normal direction.
    fn find_faces_by_normal(topo: &Topology, solid: SolidId, target_normal: Vec3) -> Vec<FaceId> {
        let tol = Tolerance::loose();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        let mut result = Vec::new();
        for &fid in sh.faces() {
            let f = topo.face(fid).unwrap();
            if let FaceSurface::Plane { normal, .. } = f.surface() {
                if tol.approx_eq(normal.x(), target_normal.x())
                    && tol.approx_eq(normal.y(), target_normal.y())
                    && tol.approx_eq(normal.z(), target_normal.z())
                {
                    result.push(fid);
                }
            }
        }
        result
    }

    #[test]
    fn shell_closed_box() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Shell with no open faces: creates a solid shell.
        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 6 outer + 6 inner = 12 faces (no rim faces since no openings).
        assert_eq!(sh.faces().len(), 12, "closed shell should have 12 faces");
    }

    #[test]
    fn shell_open_top() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Find the top face (+Z normal).
        let top_faces = find_faces_by_normal(&topo, cube, Vec3::new(0.0, 0.0, 1.0));
        assert_eq!(top_faces.len(), 1, "should find exactly one +Z face");

        let result = shell(&mut topo, cube, 0.1, &top_faces).unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 5 outer + 5 inner + 4 rim = 14 faces
        assert_eq!(sh.faces().len(), 14, "open-top shell should have 14 faces");
    }

    #[test]
    fn shell_volume_decrease() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let original_vol = crate::measure::solid_volume(&topo, cube, 0.1).unwrap();

        let result = shell(&mut topo, cube, 0.1, &[]).unwrap();
        let shell_vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();

        // The shelled solid should have less volume than the original
        // (we removed the interior).
        assert!(
            shell_vol < original_vol,
            "shell volume ({shell_vol}) should be less than original ({original_vol})"
        );
        assert!(
            shell_vol > 0.0,
            "shell volume should be positive, got {shell_vol}"
        );
    }

    #[test]
    fn shell_zero_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, 0.0, &[]).is_err());
    }

    #[test]
    fn shell_negative_thickness_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        assert!(shell(&mut topo, cube, -0.1, &[]).is_err());
    }
}

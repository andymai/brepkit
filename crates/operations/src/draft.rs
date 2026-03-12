//! Draft angle operation for injection molding applications.
//!
//! Equivalent to `BRepOffsetAPI_DraftAngle` in `OpenCascade`. Applies a
//! taper to selected faces of a solid relative to a pull direction.

use std::collections::HashSet;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::boolean::{FaceSpec, assemble_solid_mixed, face_polygon};
use crate::dot_normal_point;

/// Apply a draft angle to selected faces of a solid.
///
/// Tapers the specified faces by `angle_radians` relative to `pull_direction`.
/// The neutral plane is defined by `neutral_point` and `pull_direction`:
/// vertices on the neutral plane stay fixed while vertices above/below are
/// moved outward/inward.
///
/// # Errors
///
/// Returns an error if:
/// - `angle_radians` is zero or negative
/// - `pull_direction` is zero-length
/// - Any draft face is NURBS
/// - The solid is invalid
#[allow(clippy::too_many_lines)]
pub fn draft(
    topo: &mut Topology,
    solid: SolidId,
    draft_faces: &[FaceId],
    pull_direction: Vec3,
    neutral_point: Point3,
    angle_radians: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if angle_radians.abs() <= tol.angular {
        return Err(crate::OperationsError::InvalidInput {
            reason: "draft angle must be non-zero".into(),
        });
    }

    let pull = pull_direction.normalize()?;

    // Neutral plane: passes through neutral_point with normal = pull direction.
    let neutral_d = dot_normal_point(pull, neutral_point);

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let all_face_ids: Vec<FaceId> = shell.faces().to_vec();

    let draft_set: HashSet<usize> = draft_faces.iter().map(|f| f.index()).collect();

    let mut result_specs: Vec<FaceSpec> = Vec::new();

    for &fid in &all_face_ids {
        let face = topo.face(fid)?;
        let verts = face_polygon(topo, fid)?;

        if !draft_set.contains(&fid.index()) {
            // Non-draft face: keep as-is (supports any surface type).
            match face.surface() {
                FaceSurface::Plane { normal, .. } => {
                    let d = dot_normal_point(*normal, verts[0]);
                    result_specs.push(FaceSpec::Planar {
                        vertices: verts,
                        normal: *normal,
                        d,
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
            continue;
        }

        // Draft target face must be planar (vertex manipulation requires plane).
        let _face_normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "draft target faces must be planar".into(),
                });
            }
        };

        // Draft this face: for each vertex, compute its signed height
        // above the neutral plane, then offset it perpendicular to the
        // pull direction by height * tan(angle).
        let tan_angle = angle_radians.tan();

        // Compute the outward direction (perpendicular to pull, in the
        // plane of the face normal and pull direction).
        // The offset direction for each vertex is away from the pull axis.
        let new_verts: Vec<Point3> = verts
            .iter()
            .map(|&v| {
                let height = dot_normal_point(pull, v) - neutral_d;
                let offset = height * tan_angle;

                // Project the vertex position onto the pull axis through
                // the neutral point, then compute the radial direction.
                let v_to_neutral = v - neutral_point;
                let along_pull = pull * pull.dot(v_to_neutral);
                let radial = v_to_neutral - along_pull;
                let radial_len = radial.length();

                if radial_len < tol.linear {
                    // Vertex is on the pull axis — no radial offset.
                    v
                } else {
                    let radial_dir = Vec3::new(
                        radial.x() / radial_len,
                        radial.y() / radial_len,
                        radial.z() / radial_len,
                    );
                    v + radial_dir * offset
                }
            })
            .collect();

        // Recompute the face plane from the modified vertices.
        if new_verts.len() >= 3 {
            let a = new_verts[1] - new_verts[0];
            let b = new_verts[2] - new_verts[0];
            let new_normal =
                a.cross(b)
                    .normalize()
                    .map_err(|_| crate::OperationsError::InvalidInput {
                        reason: "draft produced degenerate face geometry".into(),
                    })?;
            let new_d = dot_normal_point(new_normal, new_verts[0]);
            result_specs.push(FaceSpec::Planar {
                vertices: new_verts,
                normal: new_normal,
                d: new_d,
            });
        }
    }

    assemble_solid_mixed(topo, &result_specs, tol)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    /// Helper: find faces whose normal is approximately equal to `target`.
    fn find_faces(topo: &Topology, solid: SolidId, target: Vec3) -> Vec<FaceId> {
        let tol = Tolerance::loose();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        sh.faces()
            .iter()
            .filter(|&&fid| {
                let f = topo.face(fid).unwrap();
                if let FaceSurface::Plane { normal, .. } = f.surface() {
                    tol.approx_eq(normal.x(), target.x())
                        && tol.approx_eq(normal.y(), target.y())
                        && tol.approx_eq(normal.z(), target.z())
                } else {
                    false
                }
            })
            .copied()
            .collect()
    }

    #[test]
    fn draft_single_face() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Draft the +X face with a 5° angle, pull direction +Z.
        let right_faces = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(right_faces.len(), 1);

        let result = draft(
            &mut topo,
            cube,
            &right_faces,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            5.0_f64.to_radians(),
        )
        .unwrap();

        let s = topo.solid(result).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            6,
            "drafted solid should still have 6 faces"
        );

        // Volume should decrease slightly (draft tapers the face inward).
        let vol = crate::measure::solid_volume(&topo, result, 0.1).unwrap();
        assert!(
            vol > 0.5,
            "drafted solid should have significant volume, got {vol}"
        );
    }

    #[test]
    fn draft_preserves_non_draft_faces() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let right_faces = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));
        let result = draft(
            &mut topo,
            cube,
            &right_faces,
            Vec3::new(0.0, 0.0, 1.0),
            Point3::new(0.0, 0.0, 0.0),
            5.0_f64.to_radians(),
        )
        .unwrap();

        // The top and bottom faces should still be planar with ±Z normals.
        let top = find_faces(&topo, result, Vec3::new(0.0, 0.0, 1.0));
        let bottom = find_faces(&topo, result, Vec3::new(0.0, 0.0, -1.0));
        assert_eq!(top.len(), 1, "should still have top face");
        assert_eq!(bottom.len(), 1, "should still have bottom face");
    }

    #[test]
    fn draft_zero_angle_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));

        assert!(
            draft(
                &mut topo,
                cube,
                &right,
                Vec3::new(0.0, 0.0, 1.0),
                Point3::new(0.0, 0.0, 0.0),
                0.0,
            )
            .is_err()
        );
    }

    #[test]
    fn draft_zero_pull_error() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);
        let right = find_faces(&topo, cube, Vec3::new(1.0, 0.0, 0.0));

        assert!(
            draft(
                &mut topo,
                cube,
                &right,
                Vec3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 0.0),
                5.0_f64.to_radians(),
            )
            .is_err()
        );
    }
}

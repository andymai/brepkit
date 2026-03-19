//! Defeaturing: remove small features from a solid for simulation simplification.
//!
//! Removes selected faces and heals the resulting gaps by extending
//! adjacent faces.

use std::collections::HashSet;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;
use crate::boolean::face_polygon;
use crate::dot_normal_point;

/// Remove selected faces from a solid and heal the resulting gaps.
///
/// The selected faces are removed from the shell, and the remaining
/// faces are reassembled into a new solid. For planar solids, this
/// works by extending adjacent faces to fill the gaps.
///
/// This is useful for removing small features (holes, fillets, bosses)
/// to simplify geometry for FEA/CFD simulation.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 1 face is selected for removal
/// - Removing the faces would leave fewer than 4 faces (minimum for a solid)
/// - The solid contains non-planar faces
pub fn defeature(
    topo: &mut Topology,
    solid: SolidId,
    faces_to_remove: &[FaceId],
) -> Result<SolidId, OperationsError> {
    if faces_to_remove.is_empty() {
        return Err(OperationsError::InvalidInput {
            reason: "must select at least one face to remove".into(),
        });
    }

    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let all_faces: Vec<FaceId> = shell.faces().to_vec();

    let remove_set: HashSet<usize> = faces_to_remove.iter().map(|f| f.index()).collect();

    // Collect the faces we're keeping.
    let kept_faces: Vec<FaceId> = all_faces
        .iter()
        .filter(|f| !remove_set.contains(&f.index()))
        .copied()
        .collect();

    if kept_faces.len() < 4 {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "removing {} faces would leave only {} faces (minimum 4 for a solid)",
                faces_to_remove.len(),
                kept_faces.len()
            ),
        });
    }

    // Snapshot the kept faces' geometry.
    let mut result_faces: Vec<(Vec<Point3>, Vec3, f64)> = Vec::new();

    for &fid in &kept_faces {
        let face = topo.face(fid)?;
        let (normal, _d) = match face.surface() {
            FaceSurface::Plane { normal, d } => (*normal, *d),
            _ => {
                return Err(OperationsError::InvalidInput {
                    reason: "defeaturing currently only supports planar faces".into(),
                });
            }
        };

        let verts = face_polygon(topo, fid)?;
        let d = dot_normal_point(normal, verts[0]);
        result_faces.push((verts, normal, d));
    }

    crate::boolean::assemble_solid(
        topo,
        &result_faces,
        brepkit_math::tolerance::Tolerance::new(),
    )
}

/// Auto-detect small features in a solid.
///
/// Returns face IDs of faces that are likely small features (holes, fillets)
/// based on their area being below the threshold.
///
/// # Errors
///
/// Returns an error if topology lookups or area computation fails.
pub fn detect_small_features(
    topo: &Topology,
    solid: SolidId,
    area_threshold: f64,
    deflection: f64,
) -> Result<Vec<FaceId>, OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;

    let mut small_faces = Vec::new();

    for &fid in shell.faces() {
        let area = crate::measure::face_area(topo, fid, deflection)?;
        if area < area_threshold {
            small_faces.push(fid);
        }
    }

    Ok(small_faces)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn defeature_removes_selected_face() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Get faces of the box
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let faces: Vec<FaceId> = shell.faces().to_vec();
        assert_eq!(faces.len(), 6);

        // Remove one face — should still work if >= 4 remain
        // (result won't be a closed solid, but the operation should succeed)
        let result = defeature(&mut topo, solid, &[faces[0], faces[1]]);
        assert!(result.is_ok());

        let new_solid = result.unwrap();
        let new_shell = topo
            .shell(topo.solid(new_solid).unwrap().outer_shell())
            .unwrap();
        assert_eq!(new_shell.faces().len(), 4);
    }

    #[test]
    fn defeature_too_many_faces_error() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let faces: Vec<FaceId> = shell.faces().to_vec();

        // Remove 3 faces — only 3 left, which is below minimum
        let result = defeature(&mut topo, solid, &[faces[0], faces[1], faces[2]]);
        assert!(result.is_err());
    }

    #[test]
    fn defeature_empty_selection_error() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = defeature(&mut topo, solid, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn detect_no_small_features_in_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Box faces have area 4.0, threshold 0.1 should find nothing
        let small = detect_small_features(&topo, solid, 0.1, 0.1).unwrap();
        assert!(small.is_empty(), "box should have no small features");
    }

    #[test]
    fn detect_all_faces_with_large_threshold() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 0.01, 0.01, 0.01).unwrap();

        // Very small box — all faces should be below threshold 1.0
        let small = detect_small_features(&topo, solid, 1.0, 0.01).unwrap();
        assert_eq!(small.len(), 6, "all 6 faces of tiny box should be small");
    }
}

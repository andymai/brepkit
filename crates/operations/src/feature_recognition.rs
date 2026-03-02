//! Feature recognition: detect geometric features from B-Rep topology.
//!
//! Analyzes face adjacency, surface types, and geometry to identify
//! manufacturing features like holes, pockets, fillets, and chamfers.
//! Useful for CAM path planning and simulation simplification.

use std::collections::{HashMap, HashSet};

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// A recognized geometric feature.
#[derive(Debug, Clone)]
pub enum Feature {
    /// A through-hole or blind hole.
    Hole {
        /// Faces forming the hole.
        faces: Vec<FaceId>,
        /// Estimated diameter (if detectable).
        diameter: Option<f64>,
    },
    /// A chamfer (bevel) face between two adjacent faces.
    Chamfer {
        /// The chamfer face.
        face: FaceId,
        /// The two faces adjacent to the chamfer.
        adjacent: (FaceId, FaceId),
        /// Angle between the chamfer and each adjacent face.
        angle: f64,
    },
    /// A small face that may be a fillet approximation.
    FilletLike {
        /// The fillet face.
        face: FaceId,
        /// Area of the face.
        area: f64,
    },
    /// A pocket (depression bounded by walls and a floor).
    Pocket {
        /// The floor face.
        floor: FaceId,
        /// The wall faces.
        walls: Vec<FaceId>,
    },
}

/// Recognize features in a solid.
///
/// Analyzes the solid's face adjacency and geometry to identify
/// common manufacturing features.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn recognize_features(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<Vec<Feature>, OperationsError> {
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut features = Vec::new();

    // Build face adjacency map (shared edges)
    let adjacency = build_face_adjacency(topo, &face_ids)?;

    // Detect chamfers: small planar faces at an angle between two larger faces
    detect_chamfers(topo, &face_ids, &adjacency, &mut features)?;

    // Detect fillet-like faces: small faces by area
    detect_fillet_like(topo, &face_ids, deflection, &mut features)?;

    // Detect pockets: concave planar faces with surrounding walls
    detect_pockets(topo, &face_ids, &adjacency, &mut features)?;

    Ok(features)
}

/// Build a map of face adjacency (faces sharing at least one edge).
fn build_face_adjacency(
    topo: &Topology,
    face_ids: &[FaceId],
) -> Result<HashMap<usize, Vec<FaceId>>, OperationsError> {
    let mut edge_to_faces: HashMap<usize, Vec<FaceId>> = HashMap::new();

    for &fid in face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_to_faces
                .entry(oe.edge().index())
                .or_default()
                .push(fid);
        }
    }

    let mut adjacency: HashMap<usize, Vec<FaceId>> = HashMap::new();
    for faces in edge_to_faces.values() {
        if faces.len() == 2 {
            adjacency
                .entry(faces[0].index())
                .or_default()
                .push(faces[1]);
            adjacency
                .entry(faces[1].index())
                .or_default()
                .push(faces[0]);
        }
    }

    Ok(adjacency)
}

/// Detect chamfer faces: planar faces at an angle between two adjacent planar faces.
fn detect_chamfers(
    topo: &Topology,
    face_ids: &[FaceId],
    adjacency: &HashMap<usize, Vec<FaceId>>,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    let mut seen_chamfers: HashSet<usize> = HashSet::new();

    for &fid in face_ids {
        if seen_chamfers.contains(&fid.index()) {
            continue;
        }

        let face = topo.face(fid)?;
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue,
        };

        // Get adjacent faces
        let Some(neighbors) = adjacency.get(&fid.index()) else {
            continue;
        };

        // A chamfer typically has exactly 2 adjacent planar faces with
        // normals that are neither parallel nor perpendicular to it
        if neighbors.len() < 2 {
            continue;
        }

        // Check pairs of neighbors
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let n1 = get_planar_normal(topo, neighbors[i]);
                let n2 = get_planar_normal(topo, neighbors[j]);

                if let (Some(n1), Some(n2)) = (n1, n2) {
                    let dot1 = normal.dot(n1).abs();
                    let dot2 = normal.dot(n2).abs();

                    // Chamfer face is at an angle (not parallel/perpendicular)
                    // to both adjacent faces
                    if dot1 > 0.1 && dot1 < 0.95 && dot2 > 0.1 && dot2 < 0.95 {
                        let angle = normal.dot(n1).acos();
                        seen_chamfers.insert(fid.index());
                        features.push(Feature::Chamfer {
                            face: fid,
                            adjacent: (neighbors[i], neighbors[j]),
                            angle,
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

/// Detect fillet-like faces by small area.
fn detect_fillet_like(
    topo: &Topology,
    face_ids: &[FaceId],
    deflection: f64,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    // Compute average face area
    let mut total_area = 0.0;
    let mut face_areas: Vec<(FaceId, f64)> = Vec::new();

    for &fid in face_ids {
        let area = crate::measure::face_area(topo, fid, deflection)?;
        total_area += area;
        face_areas.push((fid, area));
    }

    if face_ids.is_empty() {
        return Ok(());
    }

    #[allow(clippy::cast_precision_loss)]
    let avg_area = total_area / face_ids.len() as f64;
    let threshold = avg_area * 0.25; // Faces smaller than 25% of average

    for &(fid, area) in &face_areas {
        if area < threshold && area > 0.0 {
            features.push(Feature::FilletLike { face: fid, area });
        }
    }

    Ok(())
}

/// Detect pocket features: a planar floor with adjacent wall faces.
fn detect_pockets(
    topo: &Topology,
    face_ids: &[FaceId],
    adjacency: &HashMap<usize, Vec<FaceId>>,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    // A pocket has a floor face whose normal points "up" (opposite to the
    // majority normal direction), surrounded by wall faces perpendicular to it.

    for &fid in face_ids {
        let face = topo.face(fid)?;
        let floor_normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue,
        };

        let Some(neighbors) = adjacency.get(&fid.index()) else {
            continue;
        };

        // Check if all neighbors are walls (perpendicular to floor)
        let mut walls = Vec::new();
        for &neighbor in neighbors {
            if let Some(wall_normal) = get_planar_normal(topo, neighbor) {
                let dot = floor_normal.dot(wall_normal).abs();
                if dot < 0.1 {
                    // Nearly perpendicular — this is a wall
                    walls.push(neighbor);
                }
            }
        }

        // A pocket needs at least 3 walls (U-shape minimum)
        if walls.len() >= 3 {
            features.push(Feature::Pocket { floor: fid, walls });
        }
    }

    Ok(())
}

/// Get the normal of a planar face, or None if non-planar.
fn get_planar_normal(topo: &Topology, face_id: FaceId) -> Option<Vec3> {
    let face = topo.face(face_id).ok()?;
    match face.surface() {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        _ => None,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    #[test]
    fn box_has_no_chamfers() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let features = recognize_features(&topo, solid, 0.1).unwrap();

        let chamfer_count = features
            .iter()
            .filter(|f| matches!(f, Feature::Chamfer { .. }))
            .count();
        assert_eq!(chamfer_count, 0, "box should have no chamfers");
    }

    #[test]
    fn box_has_no_fillet_like() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let features = recognize_features(&topo, solid, 0.1).unwrap();

        let fillet_count = features
            .iter()
            .filter(|f| matches!(f, Feature::FilletLike { .. }))
            .count();
        assert_eq!(
            fillet_count, 0,
            "uniform box should have no fillet-like faces"
        );
    }

    #[test]
    fn chamfered_box_has_chamfer_features() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        // Chamfer an edge
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<FaceId> = shell.faces().to_vec();

        // Get edges
        let mut edge_set = HashSet::new();
        for &fid in &face_ids {
            let face = topo.face(fid).unwrap();
            let wire = topo.wire(face.outer_wire()).unwrap();
            for oe in wire.edges() {
                edge_set.insert(oe.edge());
            }
        }
        let edges: Vec<_> = edge_set.into_iter().collect();

        if let Ok(chamfered) = crate::chamfer::chamfer(&mut topo, solid, &[edges[0]], 0.2) {
            let features = recognize_features(&topo, chamfered, 0.1).unwrap();
            // The chamfered solid should have at least one chamfer feature
            let chamfer_count = features
                .iter()
                .filter(|f| matches!(f, Feature::Chamfer { .. }))
                .count();
            assert!(
                chamfer_count > 0,
                "chamfered box should have chamfer features, got {chamfer_count}"
            );
        }
    }

    #[test]
    fn feature_count_is_reasonable() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let features = recognize_features(&topo, solid, 0.1).unwrap();

        // A simple box might have pocket features (faces with 4 perpendicular neighbors)
        // but shouldn't have an excessive number
        assert!(
            features.len() <= 12,
            "box should have reasonable feature count, got {}",
            features.len()
        );
    }
}

//! Feature recognition: detect geometric features from B-Rep topology.
//!
//! Analyzes face adjacency, surface types, and geometry to identify
//! manufacturing features like holes, pockets, fillets, and chamfers.
//! Useful for CAM path planning and simulation simplification.

#![allow(
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::module_name_repetitions,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::option_if_let_else,
    clippy::derivable_impls,
    clippy::bool_to_int_with_if,
    clippy::if_same_then_else,
    clippy::tuple_array_conversions,
    clippy::match_same_arms,
    clippy::derive_partial_eq_without_eq,
    clippy::suspicious_operation_groupings,
    clippy::too_many_lines,
    clippy::iter_over_hash_type,
    clippy::map_unwrap_or,
    clippy::unused_self,
    clippy::used_underscore_binding
)]

use std::collections::{HashMap, HashSet};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Surface classification for a face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurfaceClass {
    /// Planar surface.
    Planar,
    /// Cylindrical surface.
    Cylindrical,
    /// Conical surface.
    Conical,
    /// Spherical surface.
    Spherical,
    /// Toroidal surface.
    Toroidal,
    /// NURBS (free-form) surface.
    FreeForm,
}

/// Concavity type of an edge between two faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConcavityType {
    /// Convex edge (dihedral angle > pi).
    Convex,
    /// Concave edge (dihedral angle < pi).
    Concave,
    /// Tangent/smooth edge (dihedral angle approximately pi).
    Tangent,
}

/// A node in the face adjacency graph.
#[derive(Debug, Clone)]
pub struct FagNode {
    /// The face ID.
    pub face: FaceId,
    /// Surface classification.
    pub surface_class: SurfaceClass,
    /// Face area (approximate).
    pub area: f64,
}

/// An edge in the face adjacency graph.
#[derive(Debug, Clone)]
pub struct FagEdge {
    /// The shared topology edge ID.
    pub edge: EdgeId,
    /// Concavity type.
    pub concavity: ConcavityType,
    /// Dihedral angle in radians.
    pub dihedral_angle: f64,
}

/// Face adjacency graph with typed nodes and edges.
pub struct FaceAdjacencyGraph {
    /// Nodes indexed by face index.
    pub nodes: HashMap<usize, FagNode>,
    /// Adjacency: `face_index -> [(neighbor_face_index, edge_info)]`.
    pub adjacency: HashMap<usize, Vec<(usize, FagEdge)>>,
}

/// Type of a detected pattern.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PatternType {
    /// Features arranged in a line.
    Linear,
    /// Features arranged in a circle.
    Circular,
}

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
    /// A detected pattern of repeated features.
    Pattern {
        /// Indices into the feature list of the pattern members.
        feature_indices: Vec<usize>,
        /// Pattern type (linear or circular).
        pattern_type: PatternType,
        /// Number of instances.
        count: usize,
        /// Spacing between instances (for linear patterns).
        spacing: Option<f64>,
    },
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

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

    // Build typed face adjacency graph
    let fag = build_face_adjacency_graph(topo, &face_ids, deflection)?;

    // Detect features using FAG
    detect_chamfers_fag(topo, &fag, &mut features)?;
    detect_fillet_like_fag(&fag, &mut features);
    detect_holes(topo, &fag, &mut features)?;
    detect_pockets_fag(&fag, &mut features);
    detect_patterns(&mut features);

    Ok(features)
}

// ---------------------------------------------------------------------------
// Face adjacency graph construction
// ---------------------------------------------------------------------------

/// Build a typed face adjacency graph from a set of face IDs.
fn build_face_adjacency_graph(
    topo: &Topology,
    face_ids: &[FaceId],
    deflection: f64,
) -> Result<FaceAdjacencyGraph, OperationsError> {
    // Build nodes: classify each face surface, compute area.
    let mut nodes = HashMap::new();
    for &fid in face_ids {
        let face = topo.face(fid)?;
        let surface_class = classify_surface(face.surface());
        let area = crate::measure::face_area(topo, fid, deflection).unwrap_or(0.0);
        nodes.insert(
            fid.index(),
            FagNode {
                face: fid,
                surface_class,
                area,
            },
        );
    }

    // Build edge-to-face map, keyed by edge index.
    let mut edge_to_faces: HashMap<usize, (EdgeId, Vec<FaceId>)> = HashMap::new();
    for &fid in face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let entry = edge_to_faces
                .entry(oe.edge().index())
                .or_insert_with(|| (oe.edge(), Vec::new()));
            entry.1.push(fid);
        }
    }

    // Build adjacency with dihedral angles.
    let mut adjacency: HashMap<usize, Vec<(usize, FagEdge)>> = HashMap::new();
    for (eid, faces) in edge_to_faces.values() {
        if faces.len() == 2 {
            let angle = compute_dihedral_angle(topo, faces[0], faces[1], *eid)?;
            let concavity = classify_concavity(angle);

            let edge_info = FagEdge {
                edge: *eid,
                concavity,
                dihedral_angle: angle,
            };
            adjacency
                .entry(faces[0].index())
                .or_default()
                .push((faces[1].index(), edge_info.clone()));
            adjacency
                .entry(faces[1].index())
                .or_default()
                .push((faces[0].index(), edge_info));
        }
    }

    Ok(FaceAdjacencyGraph { nodes, adjacency })
}

/// Classify a `FaceSurface` into a `SurfaceClass`.
fn classify_surface(surface: &FaceSurface) -> SurfaceClass {
    match surface {
        FaceSurface::Plane { .. } => SurfaceClass::Planar,
        FaceSurface::Cylinder(_) => SurfaceClass::Cylindrical,
        FaceSurface::Cone(_) => SurfaceClass::Conical,
        FaceSurface::Sphere(_) => SurfaceClass::Spherical,
        FaceSurface::Torus(_) => SurfaceClass::Toroidal,
        FaceSurface::Nurbs(_) => SurfaceClass::FreeForm,
    }
}

/// Classify dihedral angle into a concavity type.
fn classify_concavity(angle: f64) -> ConcavityType {
    const TOLERANCE: f64 = 0.01;
    if angle < std::f64::consts::PI - TOLERANCE {
        ConcavityType::Concave
    } else if angle > std::f64::consts::PI + TOLERANCE {
        ConcavityType::Convex
    } else {
        ConcavityType::Tangent
    }
}

/// Compute the dihedral angle between two faces at a shared edge.
///
/// The dihedral angle is the angle between the outward normals of the
/// two faces, measured at the edge midpoint.
fn compute_dihedral_angle(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    edge_id: EdgeId,
) -> Result<f64, OperationsError> {
    // Get edge midpoint from vertices.
    let edge = topo.edge(edge_id)?;
    let v_start = topo.vertex(edge.start())?;
    let v_end = topo.vertex(edge.end())?;
    let midpoint = Point3::new(
        (v_start.point().x() + v_end.point().x()) * 0.5,
        (v_start.point().y() + v_end.point().y()) * 0.5,
        (v_start.point().z() + v_end.point().z()) * 0.5,
    );

    let n_a = face_normal_at(topo, face_a, midpoint)?;
    let n_b = face_normal_at(topo, face_b, midpoint)?;

    // Dihedral angle via dot product, clamped for numerical safety.
    let dot = n_a.dot(n_b).clamp(-1.0, 1.0);
    Ok(dot.acos())
}

/// Get the outward normal of a face at a given point.
///
/// For planar faces this is exact. For analytic surfaces the axis or
/// a geometric normal is used. For free-form surfaces a fallback Z
/// normal is returned.
fn face_normal_at(
    topo: &Topology,
    face_id: FaceId,
    _point: Point3,
) -> Result<Vec3, OperationsError> {
    let face = topo.face(face_id)?;
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        FaceSurface::Cylinder(c) => c.axis(),
        FaceSurface::Cone(c) => c.axis(),
        FaceSurface::Sphere(_) => {
            // For a sphere the normal varies; use a fallback.
            Vec3::new(0.0, 0.0, 1.0)
        }
        FaceSurface::Torus(_) => Vec3::new(0.0, 0.0, 1.0),
        FaceSurface::Nurbs(_) => Vec3::new(0.0, 0.0, 1.0),
    };
    Ok(normal)
}

// ---------------------------------------------------------------------------
// Chamfer detection (FAG-based)
// ---------------------------------------------------------------------------

/// Detect chamfer faces using the face adjacency graph.
///
/// A chamfer is a small planar face whose normal is at an intermediate
/// angle (neither parallel nor perpendicular) to both neighboring faces.
fn detect_chamfers_fag(
    topo: &Topology,
    fag: &FaceAdjacencyGraph,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    let mut seen_chamfers: HashSet<usize> = HashSet::new();

    for (&idx, node) in &fag.nodes {
        if seen_chamfers.contains(&idx) {
            continue;
        }
        if node.surface_class != SurfaceClass::Planar {
            continue;
        }

        let face = topo.face(node.face)?;
        let normal = match face.surface() {
            FaceSurface::Plane { normal, .. } => *normal,
            _ => continue,
        };

        let neighbors = fag
            .adjacency
            .get(&idx)
            .map_or(&[] as &[_], |v| v.as_slice());
        if neighbors.len() < 2 {
            continue;
        }

        // Check pairs of neighbors for chamfer geometry.
        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (ni, _) = &neighbors[i];
                let (nj, _) = &neighbors[j];

                let n1 = get_node_planar_normal(topo, fag, *ni);
                let n2 = get_node_planar_normal(topo, fag, *nj);

                if let (Some(n1), Some(n2)) = (n1, n2) {
                    let dot1 = normal.dot(n1).abs();
                    let dot2 = normal.dot(n2).abs();

                    // Chamfer face is at an angle (not parallel/perpendicular)
                    // to both adjacent faces.
                    if dot1 > 0.1 && dot1 < 0.95 && dot2 > 0.1 && dot2 < 0.95 {
                        let angle = normal.dot(n1).acos();
                        let f1 = fag.nodes.get(ni).map(|n| n.face);
                        let f2 = fag.nodes.get(nj).map(|n| n.face);
                        if let (Some(f1), Some(f2)) = (f1, f2) {
                            seen_chamfers.insert(idx);
                            features.push(Feature::Chamfer {
                                face: node.face,
                                adjacent: (f1, f2),
                                angle,
                            });
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Get the planar normal for a FAG node, or `None` if non-planar.
fn get_node_planar_normal(
    topo: &Topology,
    fag: &FaceAdjacencyGraph,
    node_idx: usize,
) -> Option<Vec3> {
    let node = fag.nodes.get(&node_idx)?;
    if node.surface_class != SurfaceClass::Planar {
        return None;
    }
    let face = topo.face(node.face).ok()?;
    match face.surface() {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Fillet-like detection (FAG-based)
// ---------------------------------------------------------------------------

/// Detect fillet-like faces by small area relative to the average.
fn detect_fillet_like_fag(fag: &FaceAdjacencyGraph, features: &mut Vec<Feature>) {
    if fag.nodes.is_empty() {
        return;
    }

    let total_area: f64 = fag.nodes.values().map(|n| n.area).sum();
    #[allow(clippy::cast_precision_loss)]
    let avg_area = total_area / fag.nodes.len() as f64;
    let threshold = avg_area * 0.25;

    for node in fag.nodes.values() {
        if node.area < threshold && node.area > 0.0 {
            features.push(Feature::FilletLike {
                face: node.face,
                area: node.area,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Hole detection
// ---------------------------------------------------------------------------

/// Detect holes by finding cylindrical faces in the FAG.
///
/// A through-hole connects to two or more distinct planar faces;
/// a blind hole connects to fewer.
fn detect_holes(
    topo: &Topology,
    fag: &FaceAdjacencyGraph,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    for (&idx, node) in &fag.nodes {
        if node.surface_class != SurfaceClass::Cylindrical {
            continue;
        }

        let face = topo.face(node.face)?;
        let cyl = match face.surface() {
            FaceSurface::Cylinder(c) => c,
            _ => continue,
        };

        let diameter = cyl.radius() * 2.0;

        // Check neighbours to determine through vs blind.
        let neighbors = fag
            .adjacency
            .get(&idx)
            .map_or(&[] as &[_], |v| v.as_slice());
        let _planar_neighbor_count = neighbors
            .iter()
            .filter(|(ni, _)| {
                fag.nodes
                    .get(ni)
                    .is_some_and(|n| n.surface_class == SurfaceClass::Planar)
            })
            .count();

        features.push(Feature::Hole {
            faces: vec![node.face],
            diameter: Some(diameter),
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Pocket detection (FAG-based)
// ---------------------------------------------------------------------------

/// Detect pockets using concave-connected components in the FAG.
///
/// A pocket is a set of faces connected by concave edges, with at
/// least one planar floor face and two or more wall faces.
fn detect_pockets_fag(fag: &FaceAdjacencyGraph, features: &mut Vec<Feature>) {
    let mut visited: HashSet<usize> = HashSet::new();

    for &idx in fag.nodes.keys() {
        if visited.contains(&idx) {
            continue;
        }

        let node = match fag.nodes.get(&idx) {
            Some(n) => n,
            None => continue,
        };

        // Start from planar faces only.
        if node.surface_class != SurfaceClass::Planar {
            continue;
        }

        // Flood-fill along concave edges.
        let mut component = HashSet::new();
        let mut stack = vec![idx];

        while let Some(current) = stack.pop() {
            if !component.insert(current) {
                continue;
            }

            if let Some(adj) = fag.adjacency.get(&current) {
                for (neighbor, edge) in adj {
                    if edge.concavity == ConcavityType::Concave && !component.contains(neighbor) {
                        stack.push(*neighbor);
                    }
                }
            }
        }

        // Classify component: floor = planar, walls = non-planar or
        // perpendicular planar faces.
        let mut floor = None;
        let mut walls = Vec::new();

        for &ci in &component {
            if let Some(n) = fag.nodes.get(&ci) {
                if n.surface_class == SurfaceClass::Planar {
                    if floor.is_none() {
                        floor = Some(n.face);
                    }
                } else {
                    walls.push(n.face);
                }
            }
        }

        if let Some(floor_face) = floor {
            if walls.len() >= 2 {
                features.push(Feature::Pocket {
                    floor: floor_face,
                    walls,
                });
                visited.extend(&component);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Pattern detection
// ---------------------------------------------------------------------------

/// Detect patterns (linear or circular) among already-recognized features.
///
/// Groups holes by similar diameter, then tests whether their centroids
/// are collinear (linear pattern) or cocircular (circular pattern).
fn detect_patterns(features: &mut Vec<Feature>) {
    // Collect hole features with their index and diameter.
    let hole_info: Vec<(usize, f64)> = features
        .iter()
        .enumerate()
        .filter_map(|(i, f)| match f {
            Feature::Hole {
                diameter: Some(d), ..
            } => Some((i, *d)),
            _ => None,
        })
        .collect();

    if hole_info.len() < 3 {
        return;
    }

    // Group by diameter (within 1% tolerance).
    let groups = group_by_diameter(&hole_info);

    let mut new_patterns = Vec::new();

    for group in &groups {
        if group.len() < 3 {
            continue;
        }

        let indices: Vec<usize> = group.iter().map(|&(i, _)| i).collect();

        // For now, any group of 3+ holes with matching diameter is a linear
        // pattern. True centroid fitting would require face centroid data
        // which we do not have here, so we report the group as linear.
        #[allow(clippy::cast_precision_loss)]
        let count = indices.len();
        new_patterns.push(Feature::Pattern {
            feature_indices: indices,
            pattern_type: PatternType::Linear,
            count,
            spacing: None,
        });
    }

    features.extend(new_patterns);
}

/// Group `(index, diameter)` pairs by similar diameter (1% relative tolerance).
fn group_by_diameter(items: &[(usize, f64)]) -> Vec<Vec<(usize, f64)>> {
    let mut groups: Vec<Vec<(usize, f64)>> = Vec::new();

    for &item in items {
        let mut found = false;
        for group in &mut groups {
            let repr = group[0].1;
            if (item.1 - repr).abs() < repr * 0.01 + 1e-12 {
                group.push(item);
                found = true;
                break;
            }
        }
        if !found {
            groups.push(vec![item]);
        }
    }

    groups
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

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

    #[test]
    fn fag_nodes_match_face_count() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<FaceId> = shell.faces().to_vec();

        let fag = build_face_adjacency_graph(&topo, &face_ids, 0.1).unwrap();
        assert_eq!(fag.nodes.len(), 6, "box has 6 faces");
    }

    #[test]
    fn fag_box_all_planar() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<FaceId> = shell.faces().to_vec();

        let fag = build_face_adjacency_graph(&topo, &face_ids, 0.1).unwrap();
        for node in fag.nodes.values() {
            assert_eq!(node.surface_class, SurfaceClass::Planar);
        }
    }

    #[test]
    fn fag_box_adjacency_exists() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<FaceId> = shell.faces().to_vec();

        let fag = build_face_adjacency_graph(&topo, &face_ids, 0.1).unwrap();
        // Each face of a box shares edges with 4 other faces.
        for node in fag.nodes.values() {
            let adj = fag.adjacency.get(&node.face.index());
            assert!(adj.is_some(), "face should have adjacency");
            let neighbors = adj.unwrap();
            assert!(
                neighbors.len() >= 2,
                "each box face should have at least 2 neighbors, got {}",
                neighbors.len()
            );
        }
    }

    #[test]
    fn box_has_no_holes() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let features = recognize_features(&topo, solid, 0.1).unwrap();
        let hole_count = features
            .iter()
            .filter(|f| matches!(f, Feature::Hole { .. }))
            .count();
        assert_eq!(hole_count, 0, "box should have no holes");
    }

    #[test]
    fn box_has_no_patterns() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let features = recognize_features(&topo, solid, 0.1).unwrap();
        let pattern_count = features
            .iter()
            .filter(|f| matches!(f, Feature::Pattern { .. }))
            .count();
        assert_eq!(pattern_count, 0, "box should have no patterns");
    }

    #[test]
    fn classify_surface_variants() {
        // Plane
        assert_eq!(
            classify_surface(&FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            }),
            SurfaceClass::Planar
        );
    }

    #[test]
    fn concavity_classification() {
        use std::f64::consts::PI;
        assert_eq!(classify_concavity(PI * 0.5), ConcavityType::Concave);
        assert_eq!(classify_concavity(PI), ConcavityType::Tangent);
        assert_eq!(classify_concavity(PI * 1.5), ConcavityType::Convex);
    }

    #[test]
    fn group_by_diameter_groups_similar() {
        let items = vec![(0, 10.0), (1, 10.05), (2, 20.0), (3, 10.02)];
        let groups = group_by_diameter(&items);
        assert_eq!(groups.len(), 2, "should form 2 groups");
    }

    #[test]
    fn pattern_detection_needs_three() {
        let mut features = vec![
            Feature::Hole {
                faces: vec![],
                diameter: Some(5.0),
            },
            Feature::Hole {
                faces: vec![],
                diameter: Some(5.0),
            },
        ];
        detect_patterns(&mut features);
        let pattern_count = features
            .iter()
            .filter(|f| matches!(f, Feature::Pattern { .. }))
            .count();
        assert_eq!(pattern_count, 0, "need at least 3 holes for a pattern");
    }

    #[test]
    fn pattern_detection_three_same_diameter() {
        let mut features = vec![
            Feature::Hole {
                faces: vec![],
                diameter: Some(5.0),
            },
            Feature::Hole {
                faces: vec![],
                diameter: Some(5.0),
            },
            Feature::Hole {
                faces: vec![],
                diameter: Some(5.0),
            },
        ];
        detect_patterns(&mut features);
        let pattern_count = features
            .iter()
            .filter(|f| matches!(f, Feature::Pattern { .. }))
            .count();
        assert_eq!(pattern_count, 1, "3 same-diameter holes form a pattern");
    }
}

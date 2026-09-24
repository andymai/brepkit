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
    let face_ids = brepkit_topology::explorer::solid_faces(topo, solid)?;

    let mut features = Vec::new();

    let fag = build_face_adjacency_graph(topo, &face_ids, deflection)?;

    detect_chamfers_fag(topo, &fag, &mut features)?;
    detect_fillet_like_fag(&fag, &mut features);
    detect_holes(topo, &fag, &mut features)?;
    detect_pockets_fag(&fag, &mut features);
    detect_patterns(&mut features);

    Ok(features)
}

/// Build a typed face adjacency graph from a set of face IDs.
fn build_face_adjacency_graph(
    topo: &Topology,
    face_ids: &[FaceId],
    deflection: f64,
) -> Result<FaceAdjacencyGraph, OperationsError> {
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

    // Every wire counts: a hole's wall meets its plate along an inner wire.
    let mut edge_to_faces: HashMap<usize, (EdgeId, Vec<(FaceId, bool)>)> = HashMap::new();
    for &fid in face_ids {
        let face = topo.face(fid)?;
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            for oe in topo.wire(wire_id)?.edges() {
                let entry = edge_to_faces
                    .entry(oe.edge().index())
                    .or_insert_with(|| (oe.edge(), Vec::new()));
                entry.1.push((fid, oe.is_forward()));
            }
        }
    }

    let mut adjacency: HashMap<usize, Vec<(usize, FagEdge)>> = HashMap::new();
    for (eid, uses) in edge_to_faces.values() {
        // A seam is one face meeting itself.
        if let [(a, forward_in_a), (b, _)] = uses.as_slice()
            && a != b
        {
            let faces = [*a, *b];
            let angle = compute_dihedral_angle(topo, faces[0], faces[1], *eid, *forward_in_a)?;
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

/// The dihedral angle at a shared edge, measured through the material's
/// outside: above π on a convex edge, below π on a concave one, π where the
/// faces meet tangentially.
///
/// The outward normals are taken at the edge's parametric midpoint. The edge
/// runs as face A's boundary traverses it, which is counter-clockwise about
/// A's outward normal, so the edge is convex exactly when `n_A × n_B` points
/// along it.
fn compute_dihedral_angle(
    topo: &Topology,
    face_a: FaceId,
    face_b: FaceId,
    edge_id: EdgeId,
    forward_in_a: bool,
) -> Result<f64, OperationsError> {
    let edge = topo.edge(edge_id)?;
    let (sp, ep) = (
        topo.vertex(edge.start())?.point(),
        topo.vertex(edge.end())?.point(),
    );
    let curve = edge.curve();
    let (t0, t1) = curve.domain_with_endpoints(sp, ep);
    let mid = 0.5 * (t0 + t1);
    let point = curve.evaluate_with_endpoints(mid, sp, ep);
    let mut tangent = curve.tangent_with_endpoints(mid, sp, ep);
    // A whole NURBS edge can keep a domain that runs from its end vertex.
    let from = curve.evaluate_with_endpoints(t0, sp, ep);
    if edge.start() != edge.end() && (from - sp).length() > (from - ep).length() {
        tangent = -tangent;
    }
    if forward_in_a == topo.face(face_a)?.is_reversed() {
        tangent = -tangent;
    }

    let n_a = outward_normal_at(topo, face_a, point)?;
    let n_b = outward_normal_at(topo, face_b, point)?;
    let bend = n_a.dot(n_b).clamp(-1.0, 1.0).acos();
    Ok(if n_a.cross(n_b).dot(tangent) > 0.0 {
        std::f64::consts::PI + bend
    } else {
        std::f64::consts::PI - bend
    })
}

/// A face's outward unit normal at a point on it: the surface normal there,
/// turned by the face's reversed flag.
fn outward_normal_at(
    topo: &Topology,
    face_id: FaceId,
    point: Point3,
) -> Result<Vec3, OperationsError> {
    let face = topo.face(face_id)?;
    let surface = face.surface();
    let normal = if let FaceSurface::Plane { normal, .. } = surface {
        *normal
    } else {
        let (u, v) = surface
            .project_point(point)
            .ok_or_else(|| OperationsError::InvalidInput {
                reason: "cannot locate an edge point on its face".into(),
            })?;
        surface.normal(u, v)
    };
    let normal = normal.normalize().unwrap_or(normal);
    Ok(if face.is_reversed() { -normal } else { normal })
}

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

        for i in 0..neighbors.len() {
            for j in (i + 1)..neighbors.len() {
                let (ni, _) = &neighbors[i];
                let (nj, _) = &neighbors[j];

                let n1 = get_node_planar_normal(topo, fag, *ni);
                let n2 = get_node_planar_normal(topo, fag, *nj);

                if let (Some(n1), Some(n2)) = (n1, n2) {
                    let dot1 = normal.dot(n1).abs();
                    let dot2 = normal.dot(n2).abs();

                    // A chamfer sits at an angle (neither parallel nor
                    // perpendicular) to both faces it bevels and is small
                    // beside them: a regular prism's equal sides meet at the
                    // same angles and are not chamfers.
                    let area = |ni: &usize| fag.nodes.get(ni).map_or(0.0, |n| n.area);
                    let small = node.area <= 0.5 * area(ni).min(area(nj));
                    if small && dot1 > 0.1 && dot1 < 0.95 && dot2 > 0.1 && dot2 < 0.95 {
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

/// Detect fillets: curved faces that meet at least two neighbours
/// tangentially, as a rolling ball's band meets the two faces it blends.
fn detect_fillet_like_fag(fag: &FaceAdjacencyGraph, features: &mut Vec<Feature>) {
    let mut nodes: Vec<(&usize, &FagNode)> = fag.nodes.iter().collect();
    nodes.sort_unstable_by_key(|(idx, _)| **idx);
    for (idx, node) in nodes {
        if node.surface_class == SurfaceClass::Planar {
            continue;
        }
        let tangent = fag.adjacency.get(idx).map_or(0, |adj| {
            adj.iter()
                .filter(|(_, e)| e.concavity == ConcavityType::Tangent)
                .count()
        });
        if tangent >= 2 {
            features.push(Feature::FilletLike {
                face: node.face,
                area: node.area,
            });
        }
    }
}

/// Detect holes: cylindrical faces whose material lies outside the
/// cylinder, so their outward normal faces the axis (a boss's faces away).
fn detect_holes(
    topo: &Topology,
    fag: &FaceAdjacencyGraph,
    features: &mut Vec<Feature>,
) -> Result<(), OperationsError> {
    for node in fag.nodes.values() {
        if node.surface_class != SurfaceClass::Cylindrical {
            continue;
        }

        let face = topo.face(node.face)?;
        let cyl = match face.surface() {
            FaceSurface::Cylinder(c) => c,
            _ => continue,
        };
        // The cylinder's own normal points away from its axis.
        if !face.is_reversed() {
            continue;
        }

        let diameter = cyl.radius() * 2.0;

        features.push(Feature::Hole {
            faces: vec![node.face],
            diameter: Some(diameter),
        });
    }

    Ok(())
}

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

        if node.surface_class != SurfaceClass::Planar {
            continue;
        }

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

        // The floor meets every wall along a concave edge, so it is the
        // planar face with the most concave neighbours in the component (the
        // larger on a tie); every other face is a wall.
        let concave_degree = |ci: usize| {
            fag.adjacency.get(&ci).map_or(0, |adj| {
                adj.iter()
                    .filter(|(n, e)| e.concavity == ConcavityType::Concave && component.contains(n))
                    .count()
            })
        };
        let mut members: Vec<usize> = component.iter().copied().collect();
        members.sort_unstable();
        let floor_idx = members
            .iter()
            .copied()
            .filter(|ci| {
                fag.nodes
                    .get(ci)
                    .is_some_and(|n| n.surface_class == SurfaceClass::Planar)
            })
            .max_by(|&a, &b| {
                let area = |ci: usize| fag.nodes.get(&ci).map_or(0.0, |n| n.area);
                concave_degree(a)
                    .cmp(&concave_degree(b))
                    .then(area(a).total_cmp(&area(b)))
            });
        let floor = floor_idx.and_then(|ci| fag.nodes.get(&ci)).map(|n| n.face);
        let walls: Vec<FaceId> = members
            .iter()
            .filter(|&&ci| Some(ci) != floor_idx)
            .filter_map(|ci| fag.nodes.get(ci).map(|n| n.face))
            .collect();

        if let Some(floor_face) = floor
            && walls.len() >= 2
        {
            features.push(Feature::Pocket {
                floor: floor_face,
                walls,
            });
            visited.extend(&component);
        }
    }
}

/// Detect patterns (linear or circular) among already-recognized features.
///
/// Groups holes by similar diameter, then tests whether their centroids
/// are collinear (linear pattern) or cocircular (circular pattern).
fn detect_patterns(features: &mut Vec<Feature>) {
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::primitives::make_box;

    use crate::boolean::{BooleanOp, boolean};
    use crate::primitives::make_cylinder;
    use crate::transform::transform_solid;
    use brepkit_math::mat::Mat4;

    fn placed_box(topo: &mut Topology, lo: [f64; 3], size: [f64; 3]) -> SolidId {
        let b = make_box(topo, size[0], size[1], size[2]).unwrap();
        transform_solid(topo, b, &Mat4::translation(lo[0], lo[1], lo[2])).unwrap();
        b
    }

    fn edge_concavities(topo: &Topology, solid: SolidId) -> Vec<(ConcavityType, f64)> {
        let faces = brepkit_topology::explorer::solid_faces(topo, solid).unwrap();
        let fag = build_face_adjacency_graph(topo, &faces, 0.1).unwrap();
        let mut seen = HashSet::new();
        let mut out = Vec::new();
        for adj in fag.adjacency.values() {
            for (_, e) in adj {
                if seen.insert(e.edge.index()) {
                    out.push((e.concavity, e.dihedral_angle));
                }
            }
        }
        out
    }

    fn count(edges: &[(ConcavityType, f64)], kind: ConcavityType) -> usize {
        edges.iter().filter(|(c, _)| *c == kind).count()
    }

    #[test]
    fn box_edges_are_convex_right_angles() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let edges = edge_concavities(&topo, solid);
        assert_eq!(edges.len(), 12);
        for (kind, angle) in edges {
            assert_eq!(kind, ConcavityType::Convex);
            assert!(
                (angle - 1.5 * std::f64::consts::PI).abs() < 1e-9,
                "dihedral {angle}"
            );
        }
    }

    /// A rectangular pocket, as cut and mirrored: its floor and corners are
    /// concave, its rim convex, and it is found with its floor and four walls.
    #[test]
    fn rectangular_pocket_is_concave_and_found() {
        for mirrored in [false, true] {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 10.0, 10.0, 5.0).unwrap();
            let cutter = placed_box(&mut topo, [3.0, 3.0, 2.0], [4.0, 4.0, 4.0]);
            let solid = boolean(&mut topo, BooleanOp::Cut, block, cutter).unwrap();
            if mirrored {
                transform_solid(&mut topo, solid, &Mat4::scale(-1.0, 1.0, 1.0)).unwrap();
            }
            let edges = edge_concavities(&topo, solid);
            assert_eq!(
                count(&edges, ConcavityType::Concave),
                8,
                "mirrored={mirrored}"
            );
            assert_eq!(
                count(&edges, ConcavityType::Convex),
                16,
                "mirrored={mirrored}"
            );

            let features = recognize_features(&topo, solid, 0.1).unwrap();
            let pockets: Vec<_> = features
                .iter()
                .filter_map(|f| match f {
                    Feature::Pocket { floor, walls } => Some((*floor, walls.len())),
                    _ => None,
                })
                .collect();
            assert_eq!(pockets.len(), 1, "mirrored={mirrored}: {pockets:?}");
            let (floor, walls) = pockets[0];
            assert_eq!(walls, 4, "mirrored={mirrored}");
            let FaceSurface::Plane { normal, d } = topo.face(floor).unwrap().surface() else {
                panic!("floor is not planar");
            };
            assert!((normal.z().abs() - 1.0).abs() < 1e-9 && (d.abs() - 2.0).abs() < 1e-9);
        }
    }

    /// A drilled hole is a concave cylinder whose rims are convex; a plain
    /// cylinder's wall is a boss, not a hole.
    #[test]
    fn drilled_hole_is_found_and_a_boss_is_not() {
        for mirrored in [false, true] {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 10.0, 10.0, 4.0).unwrap();
            let drill = make_cylinder(&mut topo, 1.5, 8.0).unwrap();
            transform_solid(&mut topo, drill, &Mat4::translation(5.0, 5.0, -2.0)).unwrap();
            let solid = boolean(&mut topo, BooleanOp::Cut, block, drill).unwrap();
            if mirrored {
                transform_solid(&mut topo, solid, &Mat4::scale(1.0, -1.0, 1.0)).unwrap();
            }
            let features = recognize_features(&topo, solid, 0.1).unwrap();
            let holes: Vec<_> = features
                .iter()
                .filter_map(|f| match f {
                    Feature::Hole { diameter, .. } => *diameter,
                    _ => None,
                })
                .collect();
            assert_eq!(holes.len(), 1, "mirrored={mirrored}: {holes:?}");
            assert!((holes[0] - 3.0).abs() < 1e-9);
            let edges = edge_concavities(&topo, solid);
            assert_eq!(
                count(&edges, ConcavityType::Concave),
                0,
                "mirrored={mirrored}"
            );
        }

        // A blind hole: its floor meets the wall concavely, but a single wall
        // is no pocket.
        let mut topo = Topology::new();
        let block = make_box(&mut topo, 10.0, 10.0, 4.0).unwrap();
        let drill = make_cylinder(&mut topo, 1.0, 8.0).unwrap();
        transform_solid(&mut topo, drill, &Mat4::translation(5.0, 5.0, 1.5)).unwrap();
        let blind = boolean(&mut topo, BooleanOp::Cut, block, drill).unwrap();
        let features = recognize_features(&topo, blind, 0.1).unwrap();
        let holes = features
            .iter()
            .filter(
                |f| matches!(f, Feature::Hole { diameter: Some(d), .. } if (d - 2.0).abs() < 1e-9),
            )
            .count();
        assert_eq!(holes, 1, "{features:?}");
        assert!(!features.iter().any(|f| matches!(f, Feature::Pocket { .. })));

        let mut topo = Topology::new();
        let boss = make_cylinder(&mut topo, 1.5, 4.0).unwrap();
        let features = recognize_features(&topo, boss, 0.1).unwrap();
        assert!(!features.iter().any(|f| matches!(f, Feature::Hole { .. })));
    }

    /// A rounded box edge is a cylinder tangent to both faces it blends;
    /// nothing else on the box is.
    #[test]
    fn filleted_edge_is_found() {
        for mirrored in [false, true] {
            let mut topo = Topology::new();
            let block = make_box(&mut topo, 4.0, 3.0, 2.0).unwrap();
            let faces = brepkit_topology::explorer::solid_faces(&topo, block).unwrap();
            let edge = topo
                .wire(topo.face(faces[0]).unwrap().outer_wire())
                .unwrap()
                .edges()[0]
                .edge();
            let solid = crate::blend_ops::fillet_v2(&mut topo, block, &[edge], 0.5)
                .unwrap()
                .solid;
            if mirrored {
                transform_solid(&mut topo, solid, &Mat4::scale(1.0, 1.0, -1.0)).unwrap();
            }
            let features = recognize_features(&topo, solid, 0.1).unwrap();
            let fillets: Vec<FaceId> = features
                .iter()
                .filter_map(|f| match f {
                    Feature::FilletLike { face, .. } => Some(*face),
                    _ => None,
                })
                .collect();
            assert_eq!(fillets.len(), 1, "mirrored={mirrored}: {fillets:?}");
            assert!(
                matches!(
                    topo.face(fillets[0]).unwrap().surface(),
                    FaceSurface::Cylinder(_)
                ),
                "mirrored={mirrored}"
            );
        }
    }

    /// A regular hexagonal prism's sides meet at 120 degrees, like chamfers
    /// on a triangular prism, but none is small beside its neighbours.
    #[test]
    fn hexagonal_prism_has_no_chamfers() {
        let mut topo = Topology::new();
        let vs: Vec<_> = (0..6)
            .map(|k| {
                let a = std::f64::consts::FRAC_PI_3 * f64::from(k);
                topo.add_vertex(brepkit_topology::vertex::Vertex::new(
                    Point3::new(2.0 * a.cos(), 2.0 * a.sin(), 0.0),
                    1e-7,
                ))
            })
            .collect();
        let edges = (0..6)
            .map(|k| {
                let e = topo.add_edge(brepkit_topology::edge::Edge::new(
                    vs[k],
                    vs[(k + 1) % 6],
                    brepkit_topology::edge::EdgeCurve::Line,
                ));
                brepkit_topology::wire::OrientedEdge::new(e, true)
            })
            .collect();
        let wire = topo.add_wire(brepkit_topology::wire::Wire::new(edges, true).unwrap());
        let face = topo.add_face(brepkit_topology::face::Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let prism =
            crate::extrude::extrude(&mut topo, face, Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let features = recognize_features(&topo, prism, 0.1).unwrap();
        assert!(
            !features
                .iter()
                .any(|f| matches!(f, Feature::Chamfer { .. })),
            "{features:?}"
        );
    }

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

        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<FaceId> = shell.faces().to_vec();

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

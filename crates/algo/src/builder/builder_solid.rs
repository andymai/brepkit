//! BuilderSolid — OCCT-style 4-phase shell assembly.
//!
//! Takes BOP-selected faces and assembles them into manifold shells,
//! classifies shells as Growth/Hole, and nests holes inside growth shells.
//!
//! # Phases
//!
//! 1. **`perform_shapes_to_avoid`** — iterative free-edge removal
//! 2. **`perform_loops`** — connectivity flood-fill into shells
//! 3. **`perform_areas`** — Growth vs Hole classification
//! 4. **Assemble** — build final Solid from shells

use std::collections::{HashMap, HashSet, VecDeque};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};

use crate::bop::SelectedFace;
use crate::error::AlgoError;

/// Edge key for adjacency: canonical `(min, max)` vertex-index pair.
type VPair = (usize, usize);

/// Build a solid from BOP-selected faces using the 4-phase algorithm.
///
/// # Errors
///
/// Returns [`AlgoError`] if assembly produces no valid shells or
/// topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn build_solid(topo: &mut Topology, selected: &[SelectedFace]) -> Result<SolidId, AlgoError> {
    if selected.is_empty() {
        return Err(AlgoError::AssemblyFailed("no faces selected".into()));
    }

    // Step 0: Create reversed copies for Cut B-faces
    let mut face_ids: Vec<FaceId> = Vec::with_capacity(selected.len());
    for sf in selected {
        if sf.reversed {
            let face = topo.face(sf.face_id)?;
            let surface = face.surface().clone();
            let outer_wire = face.outer_wire();
            let inner_wires = face.inner_wires().to_vec();
            let reversed_face = Face::new_reversed(outer_wire, inner_wires, surface);
            face_ids.push(topo.add_face(reversed_face));
        } else {
            face_ids.push(sf.face_id);
        }
    }

    // Step 0b: Merge duplicate edges across selected faces.
    // Faces from different input solids may have separate edge entities for the
    // same geometric boundary. Merge them by quantized endpoint position so that
    // the BuilderSolid's connectivity flood-fill sees shared edges.
    // This is operation-safe: only operates on BOP-selected faces.
    merge_duplicate_edges(topo, &mut face_ids)?;

    if face_ids.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "all faces avoided (all have free edges)".into(),
        ));
    }

    // Phase 2: Build shells via connectivity flood-fill
    let shells = perform_loops(topo, &face_ids)?;

    if shells.is_empty() {
        return Err(AlgoError::AssemblyFailed("no shells formed".into()));
    }

    // Phase 3: Classify Growth vs Hole
    let (growth, holes) = perform_areas(topo, &shells);

    if growth.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "no outer shell found (all shells classified as holes)".into(),
        ));
    }

    // Phase 4: Assemble
    assemble(topo, growth, holes)
}

// ── Phase 1 ──────────────────────────────────────────────────────────

/// Iteratively remove faces with free (single-face) edges.
///
/// Only removes a face when ALL its edges are free (shared by ≤1 face).
/// This avoids stripping valid faces from multi-region boolean results.
#[allow(dead_code)] // Disabled pending full edge-identity sharing via CommonBlocks
fn perform_shapes_to_avoid(
    topo: &Topology,
    faces: &mut Vec<FaceId>,
) -> Result<Vec<FaceId>, AlgoError> {
    let mut avoided = Vec::new();

    loop {
        let edge_map = build_edge_face_map(topo, faces)?;
        let mut to_remove: HashSet<FaceId> = HashSet::new();

        // Only remove faces where EVERY edge is free (≤1 face).
        // This is less aggressive than OCCT's approach (which removes any
        // face with any free edge) to avoid stripping valid multi-region faces.
        for &fid in faces.iter() {
            let face_keys = face_edge_keys(topo, fid)?;
            if face_keys.is_empty() {
                continue;
            }
            let all_free = face_keys.iter().all(|key| {
                edge_map
                    .get(key)
                    .is_none_or(|faces_for_edge| faces_for_edge.len() <= 1)
            });
            if all_free {
                to_remove.insert(fid);
            }
        }

        if to_remove.is_empty() {
            break;
        }

        avoided.extend(to_remove.iter());
        faces.retain(|f| !to_remove.contains(f));
    }

    if !avoided.is_empty() {
        log::debug!(
            "BuilderSolid: avoided {} faces with free edges",
            avoided.len()
        );
    }

    Ok(avoided)
}

// ── Phase 2 ──────────────────────────────────────────────────────────

/// Group faces into connected shells via edge connectivity.
///
/// Uses flood-fill with dihedral angle selection at non-manifold edges.
#[allow(clippy::too_many_lines)]
fn perform_loops(topo: &Topology, faces: &[FaceId]) -> Result<Vec<Vec<FaceId>>, AlgoError> {
    let edge_map = build_edge_face_map(topo, faces)?;
    let edge_positions = build_edge_positions(topo, faces)?;

    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut shells: Vec<Vec<FaceId>> = Vec::new();

    // Pre-compute face → edge keys for neighbor lookup
    let face_edges: HashMap<FaceId, Vec<VPair>> = faces
        .iter()
        .filter_map(|&fid| Some((fid, face_edge_keys(topo, fid).ok()?)))
        .collect();

    for &start_face in faces {
        if visited.contains(&start_face) {
            continue;
        }

        let mut shell = Vec::new();
        let mut queue = VecDeque::new();

        // Track edges already filled (2 faces) in this shell
        let mut shell_edge_count: HashMap<VPair, u32> = HashMap::new();

        visited.insert(start_face);
        shell.push(start_face);
        queue.push_back(start_face);

        // Count edges of start face
        if let Some(keys) = face_edges.get(&start_face) {
            for key in keys {
                *shell_edge_count.entry(*key).or_default() += 1;
            }
        }

        while let Some(current) = queue.pop_front() {
            let Some(keys) = face_edges.get(&current) else {
                continue;
            };

            for key in keys {
                // Skip edges already manifold in this shell
                if shell_edge_count.get(key).copied().unwrap_or(0) >= 2 {
                    continue;
                }

                let Some(candidates) = edge_map.get(key) else {
                    continue;
                };

                // Filter to unvisited faces
                let unvisited: Vec<FaceId> = candidates
                    .iter()
                    .filter(|&&f| f != current && !visited.contains(&f))
                    .copied()
                    .collect();

                if unvisited.is_empty() {
                    continue;
                }

                // Select best face
                let selected = if unvisited.len() == 1 {
                    unvisited[0]
                } else if let Some((start, end)) = edge_positions.get(key) {
                    // Non-manifold: dihedral angle selection
                    get_face_off(topo, *start, *end, current, &unvisited).unwrap_or(unvisited[0])
                } else {
                    unvisited[0]
                };

                visited.insert(selected);
                shell.push(selected);
                queue.push_back(selected);

                // Update edge counts
                if let Some(sel_keys) = face_edges.get(&selected) {
                    for k in sel_keys {
                        *shell_edge_count.entry(*k).or_default() += 1;
                    }
                }
            }
        }

        shells.push(shell);
    }

    log::debug!(
        "BuilderSolid: {} shells (sizes: {:?})",
        shells.len(),
        shells.iter().map(Vec::len).collect::<Vec<_>>()
    );

    Ok(shells)
}

/// Dihedral angle selection at a non-manifold edge.
///
/// At an edge shared by 3+ faces, selects the face with the smallest
/// positive dihedral angle relative to the current face. This implements
/// clockwise face traversal around the edge.
///
/// Reference: OCCT `BOPTools_AlgoTools::GetFaceOff` + `AngleWithRef`.
pub fn get_face_off(
    topo: &Topology,
    edge_start: Point3,
    edge_end: Point3,
    current_face: FaceId,
    candidates: &[FaceId],
) -> Option<FaceId> {
    let edge_dir = edge_end - edge_start;
    let edge_len = edge_dir.length();
    if edge_len < 1e-12 {
        return candidates.first().copied();
    }
    let t = edge_dir * (1.0 / edge_len); // unit tangent

    let mid = Point3::new(
        (edge_start.x() + edge_end.x()) * 0.5,
        (edge_start.y() + edge_end.y()) * 0.5,
        (edge_start.z() + edge_end.z()) * 0.5,
    );

    // Compute bi-normal for current face: b = t × n (outward from face)
    let n_current = face_normal_at(topo, current_face, mid)?;
    let b_current = t.cross(n_current);
    let b_current_len = b_current.length();
    if b_current_len < 1e-12 {
        return candidates.first().copied();
    }
    let b_current = b_current * (1.0 / b_current_len);

    // Reference direction: the edge tangent itself. The dihedral angle is
    // measured around the edge, so the signed angle reference must be along t.
    // (n × b ≈ t for planar faces, but diverges for curved surfaces.)
    let d_ref = t;

    let mut best_face = None;
    let mut best_angle = f64::MAX;

    for &cand in candidates {
        let Some(n_cand) = face_normal_at(topo, cand, mid) else {
            continue;
        };
        let b_cand = t.cross(n_cand);
        let b_cand_len = b_cand.length();
        if b_cand_len < 1e-12 {
            continue;
        }
        let b_cand = b_cand * (1.0 / b_cand_len);

        // Signed angle from b_current to b_cand using d_ref as reference
        let mut angle = angle_with_ref(b_current, b_cand, d_ref);

        // Coplanar same-direction: small angle → natural neighbor (keep as-is)
        // Coplanar opposite-direction: angle ≈ π (keep as-is)
        // Only adjust truly zero angles (identical faces — shouldn't happen
        // since candidates exclude current_face)
        if angle.abs() < 1e-10 {
            angle = std::f64::consts::TAU; // deprioritize identical geometry
        }

        // Normalize to positive
        if angle < 0.0 {
            angle += std::f64::consts::TAU;
        }

        if angle < best_angle {
            best_angle = angle;
            best_face = Some(cand);
        }
    }

    best_face
}

/// Signed angle between two direction vectors using a reference axis.
///
/// Returns the angle from `d1` to `d2` measured around `d_ref`.
/// Port of OCCT's `AngleWithRef`.
fn angle_with_ref(d1: Vec3, d2: Vec3, d_ref: Vec3) -> f64 {
    let cross = d1.cross(d2);
    let sin_val = cross.length();
    let cos_val = d1.dot(d2);

    let mut angle = sin_val.atan2(cos_val);

    // Determine sign from reference direction
    if cross.dot(d_ref) < 0.0 {
        angle = -angle;
    }

    angle
}

/// Get face normal at a given 3D point (projects point to surface).
fn face_normal_at(topo: &Topology, face_id: FaceId, point: Point3) -> Option<Vec3> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface();

    if let FaceSurface::Plane { normal, .. } = surface {
        let n = if face.is_reversed() {
            -*normal
        } else {
            *normal
        };
        Some(n)
    } else {
        let (u, v) = surface.project_point(point)?;
        let mut n = surface.normal(u, v);
        if face.is_reversed() {
            n = -n;
        }
        Some(n)
    }
}

// ── Phase 3 ──────────────────────────────────────────────────────────

/// Classify shells as Growth (outer) or Hole (inner).
///
/// Uses signed volume: positive → outward normals (growth),
/// negative → inward normals (hole).
fn perform_areas(topo: &Topology, shells: &[Vec<FaceId>]) -> (Vec<Vec<FaceId>>, Vec<Vec<FaceId>>) {
    let mut growth = Vec::new();
    let mut holes = Vec::new();

    for shell in shells {
        if shell.is_empty() {
            continue;
        }

        let signed_vol = signed_volume_of_shell(topo, shell);

        if signed_vol >= 0.0 {
            growth.push(shell.clone());
        } else {
            holes.push(shell.clone());
        }
    }

    log::debug!(
        "BuilderSolid: {} growth shells, {} hole shells",
        growth.len(),
        holes.len()
    );

    (growth, holes)
}

/// Compute a signed volume estimate for a shell using the divergence theorem.
///
/// Positive = outward-oriented normals (growth shell).
/// Negative = inward-oriented normals (hole shell).
fn signed_volume_of_shell(topo: &Topology, faces: &[FaceId]) -> f64 {
    let mut volume = 0.0;

    for &fid in faces {
        let Ok(face) = topo.face(fid) else { continue };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            continue;
        };

        // Collect wire vertices
        let mut verts = Vec::new();
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let vid = oe.oriented_start(edge);
            if let Ok(v) = topo.vertex(vid) {
                verts.push(v.point());
            }
        }

        if verts.len() < 3 {
            continue;
        }

        // Fan triangulation from first vertex
        let v0 = verts[0];
        let sign = if face.is_reversed() { -1.0 } else { 1.0 };
        for i in 1..verts.len() - 1 {
            let v1 = verts[i];
            let v2 = verts[i + 1];
            // Signed volume of tetrahedron with origin
            volume += sign
                * (v0.x() * (v1.y() * v2.z() - v2.y() * v1.z())
                    + v1.x() * (v2.y() * v0.z() - v0.y() * v2.z())
                    + v2.x() * (v0.y() * v1.z() - v1.y() * v0.z()));
        }
    }

    volume / 6.0
}

// ── Phase 4 ──────────────────────────────────────────────────────────

/// Final assembly: build Solid from growth + hole shells.
fn assemble(
    topo: &mut Topology,
    growth_shells: Vec<Vec<FaceId>>,
    hole_shells: Vec<Vec<FaceId>>,
) -> Result<SolidId, AlgoError> {
    // Use the largest growth shell as outer
    let outer_idx = growth_shells
        .iter()
        .enumerate()
        .max_by_key(|(_, s)| s.len())
        .map(|(i, _)| i)
        .unwrap_or(0);

    let outer_shell = Shell::new(growth_shells[outer_idx].clone())
        .map_err(|e| AlgoError::AssemblyFailed(format!("outer shell: {e}")))?;
    let outer_id = topo.add_shell(outer_shell);

    // All hole shells become inner shells of this solid
    let mut inner_ids = Vec::new();
    for hole in &hole_shells {
        if let Ok(inner_shell) = Shell::new(hole.clone()) {
            inner_ids.push(topo.add_shell(inner_shell));
        }
    }

    // Additional growth shells (disjoint outward-oriented regions) are added
    // as extra shells. For boolean results this typically doesn't happen (single
    // connected result), but disjoint fuse can produce multiple growth shells.
    // TODO: use Compound for true multi-region results.
    for (i, gs) in growth_shells.iter().enumerate() {
        if i != outer_idx {
            if let Ok(extra_shell) = Shell::new(gs.clone()) {
                inner_ids.push(topo.add_shell(extra_shell));
            }
        }
    }

    let solid = Solid::new(outer_id, inner_ids);
    let solid_id = topo.add_solid(solid);

    log::debug!(
        "BuilderSolid: assembled solid {solid_id:?} with {} faces",
        growth_shells
            .iter()
            .chain(hole_shells.iter())
            .map(Vec::len)
            .sum::<usize>()
    );

    Ok(solid_id)
}

// ── Edge Merging ─────────────────────────────────────────────────────

/// Quantized 3D position key for edge endpoint matching.
type QPos = (i64, i64, i64);

/// Quantized position pair (canonical order: min first).
type QPosEdge = (QPos, QPos);

/// Quantize a 3D point to integer coordinates at tolerance resolution.
fn quantize_point(p: Point3, tol: f64) -> QPos {
    let scale = 1.0 / tol;
    (
        (p.x() * scale).round() as i64,
        (p.y() * scale).round() as i64,
        (p.z() * scale).round() as i64,
    )
}

/// Edge data for duplicate detection.
struct EdgeEntry {
    edge_id: brepkit_topology::edge::EdgeId,
    face_idx: usize,
    qpair: QPosEdge,
}

/// Merge duplicate edges across selected faces by quantized endpoint position.
///
/// For each group of edges with the same quantized start/end positions,
/// picks one canonical edge and rebuilds the other faces' wires to reference it.
/// Handles reversed edge direction: if the canonical edge's vertex order is
/// opposite to the duplicate, flips the `is_forward` flag to preserve wire
/// traversal direction.
#[allow(clippy::too_many_lines)]
fn merge_duplicate_edges(topo: &mut Topology, face_ids: &mut [FaceId]) -> Result<(), AlgoError> {
    use brepkit_topology::edge::EdgeId;

    let tol = 1e-7; // linear tolerance

    let mut entries: Vec<EdgeEntry> = Vec::new();

    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let qs = quantize_point(sp, tol);
                let qe = quantize_point(ep, tol);
                let qpair = if qs <= qe { (qs, qe) } else { (qe, qs) };
                entries.push(EdgeEntry {
                    edge_id: oe.edge(),
                    face_idx: fi,
                    qpair,
                });
            }
        }
    }

    // Step 2: Group edges by quantized position pair.
    // Find groups where multiple DIFFERENT EdgeIds share the same qpair.
    let mut groups: HashMap<QPosEdge, Vec<EdgeId>> = HashMap::new();
    for entry in &entries {
        groups.entry(entry.qpair).or_default().push(entry.edge_id);
    }

    // Build edge replacement map: duplicate EdgeId → (canonical EdgeId, flip_forward).
    // flip_forward is true when the duplicate's vertex order is reversed vs canonical,
    // requiring the OrientedEdge's is_forward flag to be flipped.
    let mut replacements: HashMap<EdgeId, (EdgeId, bool)> = HashMap::new();
    for edge_ids in groups.values() {
        let mut unique: Vec<EdgeId> = edge_ids.clone();
        unique.sort_by_key(|e| e.index());
        unique.dedup();

        if unique.len() < 2 {
            continue;
        }

        let canonical = unique[0];
        // Get canonical edge's vertex positions for direction comparison
        let (canon_start, canon_end) = if let Ok(e) = topo.edge(canonical) {
            let sp = topo
                .vertex(e.start())
                .map(brepkit_topology::vertex::Vertex::point)
                .ok();
            let ep = topo
                .vertex(e.end())
                .map(brepkit_topology::vertex::Vertex::point)
                .ok();
            (sp, ep)
        } else {
            (None, None)
        };

        for &dup in &unique[1..] {
            // Check if duplicate has reversed vertex order
            let flip = if let (Some(cs), Some(ce)) = (canon_start, canon_end) {
                if let Ok(de) = topo.edge(dup) {
                    let ds = topo
                        .vertex(de.start())
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    let de_end = topo
                        .vertex(de.end())
                        .map(brepkit_topology::vertex::Vertex::point)
                        .ok();
                    // Reversed if dup's start matches canonical's end (within tolerance)
                    match (ds, de_end) {
                        (Some(ds), Some(de)) => {
                            (ds - ce).length() < tol && (de - cs).length() < tol
                        }
                        _ => false,
                    }
                } else {
                    false
                }
            } else {
                false
            };
            replacements.insert(dup, (canonical, flip));
        }
    }

    if replacements.is_empty() {
        return Ok(());
    }

    let merge_count = replacements.len();

    // Step 3: Rebuild faces that have replaced edges.
    // Snapshot face data, then create new faces.
    let faces_to_rebuild: HashSet<usize> = entries
        .iter()
        .filter(|e| replacements.contains_key(&e.edge_id))
        .map(|e| e.face_idx)
        .collect();

    for &fi in &faces_to_rebuild {
        let fid = face_ids[fi];

        // Snapshot
        let (surface, is_reversed, outer_oes, inner_oes_list) = {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();

            let outer_wire = topo.wire(face.outer_wire())?;
            let outer_oes: Vec<(EdgeId, bool)> = outer_wire
                .edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect();

            let inner_wids = face.inner_wires().to_vec();
            let mut inner_oes_list = Vec::new();
            for &iw in &inner_wids {
                let w = topo.wire(iw)?;
                inner_oes_list.push(
                    w.edges()
                        .iter()
                        .map(|oe| (oe.edge(), oe.is_forward()))
                        .collect::<Vec<_>>(),
                );
            }

            (surface, is_reversed, outer_oes, inner_oes_list)
        };

        // Rebuild outer wire with replaced edges (handling direction flips)
        let new_outer_oes: Vec<_> = outer_oes
            .iter()
            .map(|(eid, fwd)| {
                if let Some(&(canonical, flip)) = replacements.get(eid) {
                    let new_fwd = if flip { !*fwd } else { *fwd };
                    brepkit_topology::wire::OrientedEdge::new(canonical, new_fwd)
                } else {
                    brepkit_topology::wire::OrientedEdge::new(*eid, *fwd)
                }
            })
            .collect();
        let Ok(new_outer) = brepkit_topology::wire::Wire::new(new_outer_oes, true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer);

        // Rebuild inner wires
        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            let new_oes: Vec<_> = inner_oes
                .iter()
                .map(|(eid, fwd)| {
                    if let Some(&(canonical, flip)) = replacements.get(eid) {
                        let new_fwd = if flip { !*fwd } else { *fwd };
                        brepkit_topology::wire::OrientedEdge::new(canonical, new_fwd)
                    } else {
                        brepkit_topology::wire::OrientedEdge::new(*eid, *fwd)
                    }
                })
                .collect();
            if let Ok(w) = brepkit_topology::wire::Wire::new(new_oes, true) {
                new_inner_ids.push(topo.add_wire(w));
            }
        }

        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        face_ids[fi] = topo.add_face(new_face);
    }

    log::debug!(
        "merge_duplicate_edges: merged {merge_count} duplicate edges across {} faces",
        faces_to_rebuild.len()
    );

    Ok(())
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Build edge→face adjacency map using vertex-pair as key.
fn build_edge_face_map(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, Vec<FaceId>>, AlgoError> {
    let mut map: HashMap<VPair, Vec<FaceId>> = HashMap::new();

    for &fid in faces {
        for key in face_edge_keys(topo, fid)? {
            map.entry(key).or_default().push(fid);
        }
    }

    Ok(map)
}

/// Get all edge keys (vertex-pair) for a face's wires.
fn face_edge_keys(topo: &Topology, fid: FaceId) -> Result<Vec<VPair>, AlgoError> {
    let face = topo.face(fid)?;
    let mut keys = Vec::new();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let s = edge.start().index();
            let e = edge.end().index();
            keys.push(if s <= e { (s, e) } else { (e, s) });
        }
    }
    Ok(keys)
}

/// Build vertex-pair → 3D positions map for `get_face_off`.
fn build_edge_positions(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, (Point3, Point3)>, AlgoError> {
    let mut map: HashMap<VPair, (Point3, Point3)> = HashMap::new();

    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let s = edge.start().index();
                let e = edge.end().index();
                let key = if s <= e { (s, e) } else { (e, s) };
                if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
                    let sp = topo.vertex(edge.start())?.point();
                    let ep = topo.vertex(edge.end())?.point();
                    entry.insert((sp, ep));
                }
            }
        }
    }

    Ok(map)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn angle_with_ref_perpendicular() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(0.0, 1.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "90° between X and Y around Z: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_opposite() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(-1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle.abs() - std::f64::consts::PI).abs() < 1e-10,
            "180° between X and -X: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_negative() {
        let d1 = Vec3::new(0.0, 1.0, 0.0);
        let d2 = Vec3::new(1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle + std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "-90° between Y and X around Z: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_coplanar_same_direction() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(angle.abs() < 1e-10, "0° between X and X: got {angle}");
    }
}

//! Operations on compound entities.
//!
//! Provides utilities for working with compounds of solids:
//! extracting individual solids, fusing all solids in a compound,
//! and computing compound-level measurements.

use brepkit_math::aabb::Aabb3;
use brepkit_topology::Topology;
use brepkit_topology::compound::CompoundId;
use brepkit_topology::solid::SolidId;

use crate::boolean::{BooleanOp, boolean};

/// Extract all solid IDs from a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn explode(
    topo: &Topology,
    compound: CompoundId,
) -> Result<Vec<SolidId>, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().to_vec())
}

/// Fuse (union) all solids in a compound into a single solid.
///
/// Performs iterative boolean union on all solids. Requires at least
/// one solid in the compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or a boolean operation fails.
pub fn fuse_all(
    topo: &mut Topology,
    compound: CompoundId,
) -> Result<SolidId, crate::OperationsError> {
    let solids = {
        let comp = topo.compound(compound)?;
        comp.solids().to_vec()
    };

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound has no solids to fuse".into(),
        });
    }

    // Partition solids into overlapping groups. Disjoint solids can be merged
    // directly (no boolean needed), while overlapping groups use boolean fuse.
    let bboxes: Vec<Aabb3> = solids
        .iter()
        .map(|&sid| crate::measure::solid_bounding_box(topo, sid))
        .collect::<Result<_, _>>()?;

    // Build disjoint groups: solids whose AABBs overlap go in the same group.
    let groups = partition_overlapping(&bboxes);

    // Process each group: boolean fuse within the group, then merge shells.
    let mut group_results: Vec<SolidId> = Vec::new();
    for group in &groups {
        let group_solids: Vec<SolidId> = group.iter().map(|&i| solids[i]).collect();
        if group_solids.len() == 1 {
            group_results.push(group_solids[0]);
            continue;
        }
        // Pairwise balanced reduction within the overlapping group.
        let mut current = group_solids;
        while current.len() > 1 {
            let mut next = Vec::with_capacity(current.len().div_ceil(2));
            let mut i = 0;
            while i + 1 < current.len() {
                next.push(boolean(topo, BooleanOp::Fuse, current[i], current[i + 1])?);
                i += 2;
            }
            if i < current.len() {
                next.push(current[i]);
            }
            current = next;
        }
        group_results.push(current[0]);
    }

    if group_results.len() == 1 {
        return Ok(group_results[0]);
    }

    // Merge disjoint groups by collecting all faces into a single solid.
    merge_disjoint_solids(topo, &group_results)
}

/// Count the total number of solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound ID is invalid.
pub fn solid_count(topo: &Topology, compound: CompoundId) -> Result<usize, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    Ok(comp.solids().len())
}

/// Compute the combined bounding box of all solids in a compound.
///
/// # Errors
///
/// Returns an error if the compound is empty or measurement fails.
pub fn compound_bounding_box(
    topo: &Topology,
    compound: CompoundId,
) -> Result<brepkit_math::aabb::Aabb3, crate::OperationsError> {
    let comp = topo.compound(compound)?;
    let solids = comp.solids();

    if solids.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound is empty".into(),
        });
    }

    let mut combined = crate::measure::solid_bounding_box(topo, solids[0])?;
    for &sid in &solids[1..] {
        let bb = crate::measure::solid_bounding_box(topo, sid)?;
        combined = combined.union(bb);
    }

    Ok(combined)
}

/// Union-find path-compressed lookup.
fn uf_find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

/// Partition indices into groups where AABBs overlap (union-find).
fn partition_overlapping(bboxes: &[Aabb3]) -> Vec<Vec<usize>> {
    let n = bboxes.len();
    let mut parent: Vec<usize> = (0..n).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            if bboxes[i].intersects(bboxes[j]) {
                let ri = uf_find(&mut parent, i);
                let rj = uf_find(&mut parent, j);
                if ri != rj {
                    parent[ri] = rj;
                }
            }
        }
    }

    let mut groups: std::collections::HashMap<usize, Vec<usize>> = std::collections::HashMap::new();
    for i in 0..n {
        groups.entry(uf_find(&mut parent, i)).or_default().push(i);
    }
    groups.into_values().collect()
}

/// Merge disjoint solids into a single solid by combining all faces.
///
/// Note: the resulting outer shell contains disconnected face groups,
/// which technically violates the connected-shell invariant. This is
/// acceptable for volume measurement and tessellation (which iterate
/// faces independently), but algorithms that assume shell connectivity
/// should be aware. A future improvement would return a `Compound`.
fn merge_disjoint_solids(
    topo: &mut Topology,
    solids: &[SolidId],
) -> Result<SolidId, crate::OperationsError> {
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;

    let mut all_faces = Vec::new();
    let mut inner_shell_ids = Vec::new();

    // Snapshot phase: collect all face IDs and inner shell face sets.
    let mut inner_face_sets: Vec<Vec<brepkit_topology::face::FaceId>> = Vec::new();
    for &sid in solids {
        let solid_data = topo.solid(sid)?;
        let outer_shell = topo.shell(solid_data.outer_shell())?;
        all_faces.extend_from_slice(outer_shell.faces());

        let inner_ids: Vec<_> = solid_data.inner_shells().to_vec();
        for inner_id in inner_ids {
            let inner_shell = topo.shell(inner_id)?;
            inner_face_sets.push(inner_shell.faces().to_vec());
        }
    }

    // Allocate phase: create inner shells.
    for faces in inner_face_sets {
        let inner = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
        inner_shell_ids.push(topo.add_shell(inner));
    }

    let outer = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let outer_id = topo.add_shell(outer);
    Ok(topo.add_solid(Solid::new(outer_id, inner_shell_ids)))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::compound::Compound;

    use super::*;

    #[test]
    fn explode_returns_solids() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.add_compound(Compound::new(vec![s1, s2]));

        let solids = explode(&topo, cid).unwrap();
        assert_eq!(solids.len(), 2);
    }

    #[test]
    fn solid_count_works() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let cid = topo.add_compound(Compound::new(vec![s1]));

        assert_eq!(solid_count(&topo, cid).unwrap(), 1);
    }

    #[test]
    fn compound_bbox() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Move s2 to (5, 0, 0).
        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(5.0, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.add_compound(Compound::new(vec![s1, s2]));
        let bb = compound_bounding_box(&topo, cid).unwrap();

        let tol = Tolerance::loose();
        // s1 is [0,1], s2 translated by 5 is [5,6]
        assert!(tol.approx_eq(bb.min.x(), 0.0));
        assert!(tol.approx_eq(bb.max.x(), 6.0));
    }

    #[test]
    fn fuse_all_two_overlapping_boxes() {
        let mut topo = Topology::new();
        let s1 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();
        let s2 = crate::primitives::make_box(&mut topo, 1.0, 1.0, 1.0).unwrap();

        // Offset s2 slightly — overlapping boxes.
        crate::transform::transform_solid(
            &mut topo,
            s2,
            &brepkit_math::mat::Mat4::translation(0.5, 0.0, 0.0),
        )
        .unwrap();

        let cid = topo.add_compound(Compound::new(vec![s1, s2]));
        let fused = fuse_all(&mut topo, cid).unwrap();

        let vol = crate::measure::solid_volume(&topo, fused, 0.1).unwrap();
        // Two overlapping unit cubes: total should be less than 2.0.
        assert!(
            vol > 1.0 && vol < 2.0,
            "fused volume should be between 1 and 2, got {vol}"
        );
    }
}

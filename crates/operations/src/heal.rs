//! Topology healing: repair common defects in B-Rep models.
//!
//! Equivalent to `ShapeFix` in `OpenCascade`. Provides repair
//! operations for common geometry issues.

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;

/// Merge near-coincident vertices in a solid.
///
/// Finds vertex pairs that are within `tolerance` of each other and
/// merges them by updating all edge references to point to a single
/// canonical vertex. This fixes small gaps caused by floating-point
/// imprecision during modeling operations.
///
/// Returns the number of vertices merged.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn merge_coincident_vertices(
    topo: &mut Topology,
    solid: SolidId,
    tolerance: f64,
) -> Result<usize, crate::OperationsError> {
    let tol = if tolerance > 0.0 {
        tolerance
    } else {
        Tolerance::new().linear
    };
    let tol_sq = tol * tol;

    // Collect all vertex positions.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<_> = shell.faces().to_vec();

    // Gather all unique vertex IDs and their positions.
    let mut vertex_ids: Vec<VertexId> = Vec::new();
    let mut positions: Vec<Point3> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for &vid in &[edge.start(), edge.end()] {
                if seen.insert(vid.index()) {
                    let point = topo.vertex(vid)?.point();
                    vertex_ids.push(vid);
                    positions.push(point);
                }
            }
        }
    }

    // Build merge map: for each vertex, find the canonical (lowest-index)
    // vertex it should merge into.
    let n = vertex_ids.len();
    let mut merge_to: HashMap<usize, VertexId> = HashMap::new();
    let mut merged_count = 0;

    for i in 0..n {
        if merge_to.contains_key(&vertex_ids[i].index()) {
            continue;
        }
        for j in (i + 1)..n {
            if merge_to.contains_key(&vertex_ids[j].index()) {
                continue;
            }
            let dist_sq = (positions[i] - positions[j]).length_squared();
            if dist_sq < tol_sq {
                merge_to.insert(vertex_ids[j].index(), vertex_ids[i]);
                merged_count += 1;
            }
        }
    }

    if merged_count == 0 {
        return Ok(0);
    }

    // Apply merges: update edge start/end vertices.
    // Collect edge IDs first to avoid borrow conflicts.
    let mut edge_ids = Vec::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            edge_ids.push(oe.edge());
        }
    }
    edge_ids.sort_by_key(|e| e.index());
    edge_ids.dedup_by_key(|e| e.index());

    // Read edge data, then write updated edges.
    let updates: Vec<_> = edge_ids
        .iter()
        .filter_map(|&eid| {
            let edge = topo.edge(eid).ok()?;
            let new_start = merge_to
                .get(&edge.start().index())
                .copied()
                .unwrap_or_else(|| edge.start());
            let new_end = merge_to
                .get(&edge.end().index())
                .copied()
                .unwrap_or_else(|| edge.end());
            if new_start != edge.start() || new_end != edge.end() {
                Some((eid, new_start, new_end))
            } else {
                None
            }
        })
        .collect();

    for (eid, new_start, new_end) in updates {
        let edge = topo.edge_mut(eid)?;
        *edge = brepkit_topology::edge::Edge::new(new_start, new_end, edge.curve().clone());
    }

    Ok(merged_count)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_manifold;

    use super::*;

    #[test]
    fn no_merge_needed_on_clean_solid() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        let count = merge_coincident_vertices(&mut topo, cube, 1e-7).unwrap();
        assert_eq!(count, 0, "clean cube should have no coincident vertices");
    }

    #[test]
    fn merge_with_large_tolerance_merges_vertices() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Use a tolerance larger than the cube's edge length (1.0) — should
        // merge adjacent vertices that are within 1.1 of each other.
        let count = merge_coincident_vertices(&mut topo, cube, 1.1).unwrap();
        assert!(count > 0, "large tolerance should merge some vertices");
    }

    #[test]
    fn merge_with_zero_tolerance_uses_default() {
        let mut topo = Topology::new();
        let cube = make_unit_cube_manifold(&mut topo);

        // Zero tolerance should use default (1e-7).
        let count = merge_coincident_vertices(&mut topo, cube, 0.0).unwrap();
        assert_eq!(count, 0, "default tolerance should not merge cube vertices");
    }
}

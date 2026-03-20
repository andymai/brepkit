//! Shape contents analysis — entity counts and type breakdown.

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Summary of the topological contents of a solid.
#[allow(clippy::struct_field_names)]
#[derive(Debug, Clone)]
pub struct ContentsAnalysis {
    /// Number of faces in the outer shell.
    pub face_count: usize,
    /// Number of distinct edges across all faces.
    pub edge_count: usize,
    /// Number of distinct vertices across all faces.
    pub vertex_count: usize,
    /// Number of inner wires (holes) across all faces.
    pub hole_count: usize,
}

/// Analyze the contents (entity counts) of a solid.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn analyze_contents(topo: &Topology, solid_id: SolidId) -> Result<ContentsAnalysis, HealError> {
    let solid = topo.solid(solid_id)?;
    let shell = topo.shell(solid.outer_shell())?;

    let mut edge_set = HashSet::new();
    let mut vertex_set = HashSet::new();
    let mut hole_count = 0usize;

    for &fid in shell.faces() {
        let face = topo.face(fid)?;
        hole_count += face.inner_wires().len();

        let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();

        for wid in wire_ids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                edge_set.insert(eid.index());
                let edge = topo.edge(eid)?;
                vertex_set.insert(edge.start().index());
                vertex_set.insert(edge.end().index());
            }
        }
    }

    Ok(ContentsAnalysis {
        face_count: shell.faces().len(),
        edge_count: edge_set.len(),
        vertex_count: vertex_set.len(),
        hole_count,
    })
}

//! Shape rebuilding via entity replacement tracking.
//!
//! [`ReShape`] records vertex/edge/wire/face/shell replacements and
//! removals during fixing.  After all fixes are recorded, call
//! [`ReShape::apply`] to rebuild the solid with all substitutions
//! applied atomically.
//!
//! This is the Rust equivalent of OCCT's `BRepTools_ReShape`.

use std::collections::HashMap;

use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};
use brepkit_topology::face::FaceId;
use brepkit_topology::shell::ShellId;
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, WireId};

use crate::HealError;

/// Action to perform on a vertex during reshape.
#[derive(Debug, Clone)]
pub enum VertexAction {
    /// Replace with another vertex.
    Replace(VertexId),
    /// Remove the vertex entirely.
    Remove,
}

/// Action to perform on an edge during reshape.
#[derive(Debug, Clone)]
pub enum EdgeAction {
    /// Replace with a single edge.
    Replace(EdgeId),
    /// Replace with multiple edges (split).
    Split(Vec<EdgeId>),
    /// Remove the edge entirely.
    Remove,
}

/// Action to perform on a wire during reshape.
#[derive(Debug, Clone)]
pub enum WireAction {
    /// Replace with another wire.
    Replace(WireId),
    /// Remove the wire entirely.
    Remove,
}

/// Action to perform on a face during reshape.
#[derive(Debug, Clone)]
pub enum FaceAction {
    /// Replace with another face.
    Replace(FaceId),
    /// Replace with multiple faces (split).
    Split(Vec<FaceId>),
    /// Remove the face entirely.
    Remove,
}

/// Action to perform on a shell during reshape.
#[derive(Debug, Clone)]
pub enum ShellAction {
    /// Replace with another shell.
    Replace(ShellId),
    /// Remove the shell entirely.
    Remove,
}

/// Tracks entity replacements and removals for atomic shape rebuilding.
#[derive(Debug, Default, Clone)]
pub struct ReShape {
    vertices: HashMap<VertexId, VertexAction>,
    edges: HashMap<EdgeId, EdgeAction>,
    wires: HashMap<WireId, WireAction>,
    faces: HashMap<FaceId, FaceAction>,
    shells: HashMap<ShellId, ShellAction>,
}

impl ReShape {
    /// Create an empty reshape tracker.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    // ── Vertex operations ───────────────────────────────────────────

    /// Record that `from` should be replaced by `to`.
    pub fn replace_vertex(&mut self, from: VertexId, to: VertexId) {
        self.vertices.insert(from, VertexAction::Replace(to));
    }

    /// Record that a vertex should be removed.
    pub fn remove_vertex(&mut self, id: VertexId) {
        self.vertices.insert(id, VertexAction::Remove);
    }

    /// Resolve a vertex through the replacement chain.
    #[must_use]
    pub fn resolve_vertex(&self, mut id: VertexId) -> VertexId {
        let mut depth = 0;
        while let Some(VertexAction::Replace(target)) = self.vertices.get(&id) {
            id = *target;
            depth += 1;
            if depth > 100 {
                break; // prevent infinite loops
            }
        }
        id
    }

    /// Check if a vertex is marked for removal.
    #[must_use]
    pub fn is_vertex_removed(&self, id: VertexId) -> bool {
        matches!(self.vertices.get(&id), Some(VertexAction::Remove))
    }

    // ── Edge operations ─────────────────────────────────────────────

    /// Record that `from` should be replaced by `to`.
    pub fn replace_edge(&mut self, from: EdgeId, to: EdgeId) {
        self.edges.insert(from, EdgeAction::Replace(to));
    }

    /// Record that an edge was split into multiple edges.
    pub fn split_edge(&mut self, from: EdgeId, into: Vec<EdgeId>) {
        self.edges.insert(from, EdgeAction::Split(into));
    }

    /// Record that an edge should be removed.
    pub fn remove_edge(&mut self, id: EdgeId) {
        self.edges.insert(id, EdgeAction::Remove);
    }

    /// Resolve an edge through the replacement chain.
    #[must_use]
    pub fn resolve_edge(&self, mut id: EdgeId) -> Option<EdgeId> {
        let mut depth = 0;
        loop {
            match self.edges.get(&id) {
                Some(EdgeAction::Replace(target)) => {
                    id = *target;
                    depth += 1;
                    if depth > 100 {
                        return Some(id);
                    }
                }
                Some(EdgeAction::Remove) => return None,
                Some(EdgeAction::Split(_)) | None => return Some(id),
            }
        }
    }

    /// Check if an edge is marked for removal.
    #[must_use]
    pub fn is_edge_removed(&self, id: EdgeId) -> bool {
        matches!(self.edges.get(&id), Some(EdgeAction::Remove))
    }

    /// Get edge action if any.
    #[must_use]
    pub fn edge_action(&self, id: EdgeId) -> Option<&EdgeAction> {
        self.edges.get(&id)
    }

    // ── Wire operations ─────────────────────────────────────────────

    /// Record that `from` should be replaced by `to`.
    pub fn replace_wire(&mut self, from: WireId, to: WireId) {
        self.wires.insert(from, WireAction::Replace(to));
    }

    /// Record that a wire should be removed.
    pub fn remove_wire(&mut self, id: WireId) {
        self.wires.insert(id, WireAction::Remove);
    }

    // ── Face operations ─────────────────────────────────────────────

    /// Record that `from` should be replaced by `to`.
    pub fn replace_face(&mut self, from: FaceId, to: FaceId) {
        self.faces.insert(from, FaceAction::Replace(to));
    }

    /// Record that a face was split into multiple faces.
    pub fn split_face(&mut self, from: FaceId, into: Vec<FaceId>) {
        self.faces.insert(from, FaceAction::Split(into));
    }

    /// Record that a face should be removed.
    pub fn remove_face(&mut self, id: FaceId) {
        self.faces.insert(id, FaceAction::Remove);
    }

    /// Check if a face is marked for removal.
    #[must_use]
    pub fn is_face_removed(&self, id: FaceId) -> bool {
        matches!(self.faces.get(&id), Some(FaceAction::Remove))
    }

    // ── Shell operations ────────────────────────────────────────────

    /// Record that `from` should be replaced by `to`.
    pub fn replace_shell(&mut self, from: ShellId, to: ShellId) {
        self.shells.insert(from, ShellAction::Replace(to));
    }

    /// Record that a shell should be removed.
    pub fn remove_shell(&mut self, id: ShellId) {
        self.shells.insert(id, ShellAction::Remove);
    }

    // ── Apply ───────────────────────────────────────────────────────

    /// Whether any replacements or removals have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
            && self.edges.is_empty()
            && self.wires.is_empty()
            && self.faces.is_empty()
            && self.shells.is_empty()
    }

    /// Apply all recorded replacements to a solid, rebuilding the shape tree.
    ///
    /// Returns the (possibly new) solid ID after all substitutions.
    ///
    /// # Errors
    ///
    /// Returns [`HealError`] if entity lookups fail during rebuilding.
    pub fn apply(&self, topo: &mut Topology, solid_id: SolidId) -> Result<SolidId, HealError> {
        if self.is_empty() {
            return Ok(solid_id);
        }

        // 1. Apply vertex replacements to all edges.
        if !self.vertices.is_empty() {
            self.apply_vertex_replacements(topo, solid_id)?;
        }

        // 2. Rebuild wires (remove/split edges, rebuild edge lists).
        if !self.edges.is_empty() {
            self.apply_edge_replacements(topo, solid_id)?;
        }

        // 3. Rebuild faces (remove/replace wires, remove/replace faces).
        if !self.faces.is_empty() || !self.wires.is_empty() {
            self.apply_face_replacements(topo, solid_id)?;
        }

        // 4. Rebuild shell (remove/replace faces).
        self.apply_shell_replacements(topo, solid_id)?;

        Ok(solid_id)
    }

    /// Apply vertex replacements: update edge start/end vertices.
    fn apply_vertex_replacements(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(), HealError> {
        let solid_data = topo.solid(solid_id)?;
        let shell = topo.shell(solid_data.outer_shell())?;
        let face_ids: Vec<_> = shell.faces().to_vec();

        // Collect all unique edge IDs.
        let mut edge_ids = Vec::new();
        for &fid in &face_ids {
            let face = topo.face(fid)?;
            let wire = topo.wire(face.outer_wire())?;
            for oe in wire.edges() {
                edge_ids.push(oe.edge());
            }
            for &iw_id in face.inner_wires() {
                let iw = topo.wire(iw_id)?;
                for oe in iw.edges() {
                    edge_ids.push(oe.edge());
                }
            }
        }
        edge_ids.sort_by_key(|e| e.index());
        edge_ids.dedup_by_key(|e| e.index());

        // Snapshot updates.
        let updates: Vec<_> = edge_ids
            .iter()
            .filter_map(|&eid| {
                let edge = topo.edge(eid).ok()?;
                let new_start = self.resolve_vertex(edge.start());
                let new_end = self.resolve_vertex(edge.end());
                if new_start != edge.start() || new_end != edge.end() {
                    Some((eid, new_start, new_end, edge.curve().clone()))
                } else {
                    None
                }
            })
            .collect();

        // Apply.
        for (eid, new_start, new_end, curve) in updates {
            let edge = topo.edge_mut(eid)?;
            *edge = Edge::new(new_start, new_end, curve);
        }

        Ok(())
    }

    /// Apply edge replacements: rebuild wires with new edge lists.
    fn apply_edge_replacements(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(), HealError> {
        let solid_data = topo.solid(solid_id)?;
        let shell = topo.shell(solid_data.outer_shell())?;
        let face_ids: Vec<_> = shell.faces().to_vec();

        for &fid in &face_ids {
            let face = topo.face(fid)?;
            let all_wires: Vec<_> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();

            for wire_id in all_wires {
                let wire = topo.wire(wire_id)?;
                let old_edges: Vec<OrientedEdge> = wire.edges().to_vec();
                let is_closed = wire.is_closed();
                let mut new_edges = Vec::new();
                let mut any_changed = false;

                for oe in &old_edges {
                    match self.edges.get(&oe.edge()) {
                        Some(EdgeAction::Remove) => {
                            any_changed = true;
                        }
                        Some(EdgeAction::Replace(new_eid)) => {
                            new_edges.push(OrientedEdge::new(*new_eid, oe.is_forward()));
                            any_changed = true;
                        }
                        Some(EdgeAction::Split(new_eids)) => {
                            for &ne in new_eids {
                                new_edges.push(OrientedEdge::new(ne, oe.is_forward()));
                            }
                            any_changed = true;
                        }
                        None => {
                            new_edges.push(*oe);
                        }
                    }
                }

                if any_changed && !new_edges.is_empty() {
                    let new_wire = brepkit_topology::wire::Wire::new(new_edges, is_closed)?;
                    let new_wire_id = topo.add_wire(new_wire);

                    let face_mut = topo.face_mut(fid)?;
                    if face_mut.outer_wire() == wire_id {
                        face_mut.set_outer_wire(new_wire_id);
                    } else {
                        let iw = face_mut.inner_wires_mut();
                        for w in iw.iter_mut() {
                            if *w == wire_id {
                                *w = new_wire_id;
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Apply face replacements: update shell face list.
    fn apply_face_replacements(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(), HealError> {
        let solid_data = topo.solid(solid_id)?;
        let shell_id = solid_data.outer_shell();
        let shell = topo.shell(shell_id)?;
        let old_faces: Vec<_> = shell.faces().to_vec();

        let mut new_faces = Vec::new();
        let mut any_changed = false;

        for &fid in &old_faces {
            match self.faces.get(&fid) {
                Some(FaceAction::Remove) => {
                    any_changed = true;
                }
                Some(FaceAction::Replace(new_fid)) => {
                    new_faces.push(*new_fid);
                    any_changed = true;
                }
                Some(FaceAction::Split(new_fids)) => {
                    new_faces.extend(new_fids);
                    any_changed = true;
                }
                None => {
                    new_faces.push(fid);
                }
            }
        }

        if any_changed && !new_faces.is_empty() {
            let new_shell = brepkit_topology::shell::Shell::new(new_faces)?;
            let shell_mut = topo.shell_mut(shell_id)?;
            *shell_mut = new_shell;
        }

        Ok(())
    }

    /// Apply shell replacements (if any shells themselves were replaced).
    fn apply_shell_replacements(
        &self,
        topo: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(), HealError> {
        if self.shells.is_empty() {
            return Ok(());
        }

        let solid_data = topo.solid(solid_id)?;
        let shell_id = solid_data.outer_shell();

        if let Some(ShellAction::Replace(new_shell)) = self.shells.get(&shell_id) {
            let solid_mut = topo.solid_mut(solid_id)?;
            solid_mut.set_outer_shell(*new_shell);
        }

        Ok(())
    }
}

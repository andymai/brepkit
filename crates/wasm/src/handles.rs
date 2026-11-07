//! Entity handle resolution and ID conversion helpers.
//!
//! The `resolve_*` methods convert raw `u32` handles from JavaScript
//! into typed arena IDs, returning [`WasmError::InvalidHandle`] on failure.
//! The `*_id_to_u32` functions do the reverse for returning handles to JS.

#![allow(clippy::missing_errors_doc)]

use crate::error::WasmError;
use crate::kernel::BrepKernel;

// ── resolve_* methods ─────────────────────────────────────────────

impl BrepKernel {
    /// Resolve a `u32` face handle to a typed `FaceId`.
    pub fn resolve_face(&self, handle: u32) -> Result<brepkit_topology::face::FaceId, WasmError> {
        let index = handle as usize;
        self.topo
            .face_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "face",
                index,
            })
    }

    /// Read `[{"vertex": h, "setbacks": [{"edge": h, "distance": d}], "corner": "g2"}]`,
    /// `setbacks` and `corner` being optional and `corner` defaulting to rolling.
    pub(crate) fn parse_fillet_corners(
        &self,
        corners: &serde_json::Value,
    ) -> Result<Vec<brepkit_operations::blend_ops::FilletCorner>, String> {
        use brepkit_operations::blend_ops::{CornerKind, FilletCorner};

        let handle = |value: &serde_json::Value, what: &str| {
            value
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| format!("a corner's {what} is not a handle"))
        };
        corners
            .as_array()
            .ok_or_else(|| "corners must be an array".to_string())?
            .iter()
            .map(|corner| {
                let vertex = self
                    .resolve_vertex(handle(&corner["vertex"], "vertex")?)
                    .map_err(|e| e.to_string())?;
                let kind = match corner["corner"].as_str() {
                    None | Some("rolling") => CornerKind::Rolling,
                    Some("g2") => CornerKind::G2,
                    Some(other) => return Err(format!("unknown corner kind '{other}'")),
                };
                let mut setbacks = Vec::new();
                if !corner["setbacks"].is_null() {
                    let given = corner["setbacks"]
                        .as_array()
                        .ok_or_else(|| "a corner's setbacks must be an array".to_string())?;
                    for setback in given {
                        let edge = self
                            .resolve_edge(handle(&setback["edge"], "set-back edge")?)
                            .map_err(|e| e.to_string())?;
                        let distance = setback["distance"]
                            .as_f64()
                            .filter(|d| d.is_finite() && *d > 0.0)
                            .ok_or_else(|| "a set-back's distance must be positive".to_string())?;
                        setbacks.push((edge, distance));
                    }
                }
                Ok(FilletCorner {
                    vertex,
                    setbacks,
                    kind,
                })
            })
            .collect()
    }

    /// Resolve a `u32` vertex handle to a typed `VertexId`.
    pub fn resolve_vertex(
        &self,
        handle: u32,
    ) -> Result<brepkit_topology::vertex::VertexId, WasmError> {
        let index = handle as usize;
        self.topo
            .vertex_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "vertex",
                index,
            })
    }

    /// Resolve a `u32` edge handle to a typed `EdgeId`.
    pub fn resolve_edge(&self, handle: u32) -> Result<brepkit_topology::edge::EdgeId, WasmError> {
        let index = handle as usize;
        self.topo
            .edge_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "edge",
                index,
            })
    }

    /// Resolve a `u32` solid handle to a typed `SolidId`.
    pub fn resolve_solid(
        &self,
        handle: u32,
    ) -> Result<brepkit_topology::solid::SolidId, WasmError> {
        let index = handle as usize;
        self.topo
            .solid_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "solid",
                index,
            })
    }

    /// Resolve a `u32` wire handle to a typed `WireId`.
    pub fn resolve_wire(&self, handle: u32) -> Result<brepkit_topology::wire::WireId, WasmError> {
        let index = handle as usize;
        self.topo
            .wire_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "wire",
                index,
            })
    }

    /// Resolve a `u32` shell handle to a typed `ShellId`.
    pub fn resolve_shell(
        &self,
        handle: u32,
    ) -> Result<brepkit_topology::shell::ShellId, WasmError> {
        let index = handle as usize;
        self.topo
            .shell_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "shell",
                index,
            })
    }

    /// Resolve a `u32` compound handle to a typed `CompoundId`.
    pub fn resolve_compound(
        &self,
        handle: u32,
    ) -> Result<brepkit_topology::compound::CompoundId, WasmError> {
        let index = handle as usize;
        self.topo
            .compound_id_from_index(index)
            .ok_or(WasmError::InvalidHandle {
                entity: "compound",
                index,
            })
    }
}

// ── ID-to-u32 converters ──────────────────────────────────────────

/// Convert a `FaceId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn face_id_to_u32(id: brepkit_topology::face::FaceId) -> u32 {
    id.index() as u32
}

/// Convert a `SolidId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn solid_id_to_u32(id: brepkit_topology::solid::SolidId) -> u32 {
    id.index() as u32
}

/// Convert a `VertexId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn vertex_id_to_u32(id: brepkit_topology::vertex::VertexId) -> u32 {
    id.index() as u32
}

/// Convert an `EdgeId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn edge_id_to_u32(id: brepkit_topology::edge::EdgeId) -> u32 {
    id.index() as u32
}

/// Convert a `WireId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn wire_id_to_u32(id: brepkit_topology::wire::WireId) -> u32 {
    id.index() as u32
}

/// Convert a `ShellId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn shell_id_to_u32(id: brepkit_topology::shell::ShellId) -> u32 {
    id.index() as u32
}

/// Convert a `CompoundId` to a `u32` handle for JavaScript.
#[allow(clippy::cast_possible_truncation)]
pub const fn compound_id_to_u32(id: brepkit_topology::compound::CompoundId) -> u32 {
    id.index() as u32
}

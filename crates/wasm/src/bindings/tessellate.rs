//! Tessellation and wireframe bindings.

#![allow(clippy::missing_errors_doc)]

use brepkit_operations::tessellate;
use wasm_bindgen::prelude::*;

use crate::error::validate_positive;
use crate::kernel::BrepKernel;
use crate::shapes::JsMesh;
use crate::types::{GroupedMeshResult, UvMeshResult};

#[wasm_bindgen]
impl BrepKernel {
    // ── Tessellation ───────────────────────────────────────────────

    /// Tessellate a single face into a triangle mesh.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateFace")]
    pub fn tessellate_face(&self, face: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        let mesh = tessellate::tessellate(&self.topo, face_id, deflection)?;
        Ok(mesh.into())
    }

    /// Tessellate all faces of a solid into a single merged triangle mesh.
    ///
    /// Includes both the outer shell and any inner shells (voids).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateSolid")]
    pub fn tessellate_solid(&self, solid: u32, deflection: f64) -> Result<JsMesh, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;

        // Use watertight tessellation that shares edge vertices between
        // adjacent faces, eliminating cracks at face boundaries.
        let merged = tessellate::tessellate_solid(&self.topo, solid_id, deflection)?;

        Ok(merged.into())
    }

    /// Tessellate a solid with per-face triangle grouping.
    ///
    /// Returns a JSON string containing `{ positions, normals, indices, faceOffsets }`.
    /// `faceOffsets` is an array where `faceOffsets[i]` is the start index into
    /// `indices` for face `i`, and the last element is `indices.length`.
    #[wasm_bindgen(js_name = "tessellateSolidGrouped")]
    pub fn tessellate_solid_grouped(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;

        let mut all_positions: Vec<f64> = Vec::new();
        let mut all_normals: Vec<f64> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut face_offsets: Vec<u32> = Vec::new();

        for &face_id in &faces {
            #[allow(clippy::cast_possible_truncation)]
            let idx_offset = (all_positions.len() / 3) as u32;
            face_offsets.push(all_indices.len() as u32);

            if let Ok(mesh) = tessellate::tessellate(&self.topo, face_id, deflection) {
                for p in &mesh.positions {
                    all_positions.extend_from_slice(&[p.x(), p.y(), p.z()]);
                }
                for n in &mesh.normals {
                    all_normals.extend_from_slice(&[n.x(), n.y(), n.z()]);
                }
                for &idx in &mesh.indices {
                    all_indices.push(idx + idx_offset);
                }
            }
        }
        #[allow(clippy::cast_possible_truncation)]
        face_offsets.push(all_indices.len() as u32);

        let result = GroupedMeshResult {
            positions: all_positions,
            normals: all_normals,
            indices: all_indices,
            face_offsets,
        };
        Ok(serde_json::to_string(&result)
            .map_err(|e| JsError::new(&e.to_string()))?
            .into())
    }

    /// Tessellate a solid and include per-vertex UV coordinates.
    ///
    /// Returns a JSON string containing `{ positions, normals, indices, uvs }`.
    /// `uvs` is a flat array of `[u0, v0, u1, v1, ...]` values, two per vertex.
    /// For analytic and NURBS surfaces, these are the parametric (u, v) values.
    /// For planar faces, UVs are computed by projection onto the face plane.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "tessellateSolidUV")]
    pub fn tessellate_solid_uv(&self, solid: u32, deflection: f64) -> Result<JsValue, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_topology::explorer::solid_faces(&self.topo, solid_id)?;

        let mut all_positions: Vec<f64> = Vec::new();
        let mut all_normals: Vec<f64> = Vec::new();
        let mut all_uvs: Vec<f64> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for &face_id in &faces {
            #[allow(clippy::cast_possible_truncation)]
            let idx_offset = (all_positions.len() / 3) as u32;

            let mesh_uv = match tessellate::tessellate_with_uvs(&self.topo, face_id, deflection) {
                Ok(m) => m,
                Err(e) => {
                    log::warn!("skipping UV face during tessellation: {e}");
                    continue;
                }
            };
            for p in &mesh_uv.mesh.positions {
                all_positions.extend_from_slice(&[p.x(), p.y(), p.z()]);
            }
            for n in &mesh_uv.mesh.normals {
                all_normals.extend_from_slice(&[n.x(), n.y(), n.z()]);
            }
            for uv in &mesh_uv.uvs {
                all_uvs.extend_from_slice(uv);
            }
            for &idx in &mesh_uv.mesh.indices {
                all_indices.push(idx + idx_offset);
            }
        }

        let result = UvMeshResult {
            positions: all_positions,
            normals: all_normals,
            indices: all_indices,
            uvs: all_uvs,
        };
        Ok(serde_json::to_string(&result)
            .map_err(|e| JsError::new(&e.to_string()))?
            .into())
    }

    // ── Edge wireframe ────────────────────────────────────────────

    /// Sample edges of a solid into polylines for wireframe rendering.
    ///
    /// Returns a `JsEdgeLines` containing flattened positions and per-edge
    /// offset indices. The `deflection` parameter controls sampling density.
    ///
    /// Smooth edges (between faces on the same underlying surface) are
    /// automatically filtered out to reduce wireframe clutter. These edges
    /// arise from boolean face-splitting and don't represent visible creases.
    #[wasm_bindgen(js_name = "meshEdges")]
    pub fn mesh_edges(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<crate::shapes::JsEdgeLines, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_lines =
            tessellate::sample_solid_edges_filtered(&self.topo, solid_id, deflection, true)?;
        Ok(edge_lines.into())
    }

    /// Sample ALL edges of a solid (no smooth-edge filtering).
    ///
    /// Same as `meshEdges` but includes edges between co-surface faces.
    /// Useful for debugging topology.
    #[wasm_bindgen(js_name = "meshEdgesAll")]
    pub fn mesh_edges_all(
        &self,
        solid: u32,
        deflection: f64,
    ) -> Result<crate::shapes::JsEdgeLines, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let edge_lines =
            tessellate::sample_solid_edges_filtered(&self.topo, solid_id, deflection, false)?;
        Ok(edge_lines.into())
    }
}

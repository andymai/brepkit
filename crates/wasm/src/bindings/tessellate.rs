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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::kernel::BrepKernel;

    /// Create a kernel containing a 2×3×4 box and return (kernel, solid_handle).
    fn kernel_with_box() -> (BrepKernel, u32) {
        let mut k = BrepKernel::new();
        let solid = k.make_box_solid(2.0, 3.0, 4.0).unwrap();
        (k, solid)
    }

    // ── tessellate_solid ──────────────────────────────────────────

    #[test]
    fn tessellate_solid_box_produces_nonempty_mesh() {
        let (k, solid) = kernel_with_box();
        let mesh = k.tessellate_solid(solid, 0.1).unwrap();
        assert!(mesh.vertex_count() > 0, "expected vertices, got 0");
        assert!(mesh.triangle_count() > 0, "expected triangles, got 0");
    }

    #[test]
    fn tessellate_solid_positions_and_normals_lengths_match() {
        let (k, solid) = kernel_with_box();
        let mesh = k.tessellate_solid(solid, 0.1).unwrap();
        let positions = mesh.positions();
        let normals = mesh.normals();
        // Both must be flat [x, y, z, …] arrays — same length
        assert_eq!(
            positions.len(),
            normals.len(),
            "positions.len()={} normals.len()={}",
            positions.len(),
            normals.len()
        );
        // Divisible by 3 (complete xyz triples)
        assert_eq!(positions.len() % 3, 0);
    }

    #[test]
    fn tessellate_solid_indices_are_valid_vertex_refs() {
        let (k, solid) = kernel_with_box();
        let mesh = k.tessellate_solid(solid, 0.1).unwrap();
        let vertex_count = mesh.vertex_count();
        let indices = mesh.indices();
        for &idx in &indices {
            assert!(
                idx < vertex_count,
                "index {idx} out of bounds (vertex_count={vertex_count})"
            );
        }
    }

    #[test]
    fn tessellate_solid_coarser_deflection_has_fewer_triangles() {
        let (k, solid) = kernel_with_box();
        let fine = k.tessellate_solid(solid, 0.01).unwrap();
        let coarse = k.tessellate_solid(solid, 1.0).unwrap();
        assert!(
            fine.triangle_count() >= coarse.triangle_count(),
            "fine={} coarse={}",
            fine.triangle_count(),
            coarse.triangle_count()
        );
    }

    // ── tessellate_solid_grouped ──────────────────────────────────
    // tessellate_solid_grouped returns a JsValue (via JsValue::from_str),
    // which panics on non-wasm targets. Test the underlying logic via the
    // operations layer instead.

    #[test]
    fn tessellate_solid_grouped_via_operations() {
        let mut topo = brepkit_topology::topology::Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 2.0, 3.0, 4.0).unwrap();
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();

        let mut all_positions = 0usize;
        let mut all_indices = 0usize;
        let mut face_offsets = Vec::new();

        for &face_id in &faces {
            face_offsets.push(all_indices);
            if let Ok(mesh) = brepkit_operations::tessellate::tessellate(&topo, face_id, 0.1) {
                all_positions += mesh.positions.len();
                all_indices += mesh.indices.len();
            }
        }
        face_offsets.push(all_indices);

        assert!(all_positions > 0, "expected vertices");
        assert!(all_indices > 0, "expected indices");
        // Box has 6 faces, so faceOffsets has 7 entries (6 starts + 1 sentinel).
        assert_eq!(face_offsets.len(), 7, "expected 7 face offsets for a box");
        assert_eq!(*face_offsets.last().unwrap(), all_indices);
    }

    // ── mesh_edges_all ────────────────────────────────────────────

    #[test]
    fn mesh_edges_all_box_produces_nonempty_edge_lines() {
        let (k, solid) = kernel_with_box();
        let edge_lines = k.mesh_edges_all(solid, 0.1).unwrap();
        assert!(edge_lines.edge_count() > 0, "expected edges, got 0");
        assert!(
            !edge_lines.positions().is_empty(),
            "positions must be non-empty"
        );
    }

    #[test]
    fn mesh_edges_all_box_has_twelve_edges() {
        // A box has exactly 12 edges.
        let (k, solid) = kernel_with_box();
        let edge_lines = k.mesh_edges_all(solid, 0.1).unwrap();
        assert_eq!(
            edge_lines.edge_count(),
            12,
            "expected 12 box edges, got {}",
            edge_lines.edge_count()
        );
    }

    // ── Invalid handle ────────────────────────────────────────────
    // Error-path tests use internal operations to avoid JsError panics.

    #[test]
    fn tessellate_solid_invalid_handle_returns_error() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "tessellateSolid", "args": {"solid": 9999, "deflection": 0.1}}]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    #[test]
    fn mesh_edges_all_invalid_handle_returns_error() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "meshEdgesAll", "args": {"solid": 9999, "deflection": 0.1}}]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(parsed[0]["error"].is_string());
    }

    // ── Zero / non-positive deflection ────────────────────────────
    // validate_positive is a pure function that returns WasmError (not JsError),
    // so we test the validation logic directly.

    #[test]
    fn tessellate_solid_zero_deflection_is_invalid() {
        use crate::error::validate_positive;
        let result = validate_positive(0.0, "deflection");
        assert!(result.is_err(), "zero deflection must be rejected");
    }

    #[test]
    fn mesh_edges_all_zero_deflection_is_invalid() {
        use crate::error::validate_positive;
        let result = validate_positive(0.0, "deflection");
        assert!(result.is_err(), "zero deflection must be rejected");
    }

    #[test]
    fn negative_deflection_is_invalid() {
        use crate::error::validate_positive;
        let result = validate_positive(-1.0, "deflection");
        assert!(result.is_err(), "negative deflection must be rejected");
    }
}

//! File I/O (import/export) bindings.

#![cfg(feature = "io")]
#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_positive};
use crate::handles::{shell_id_to_u32, solid_id_to_u32};
use crate::helpers::TOL;
use crate::kernel::BrepKernel;
use crate::types::MeshImportResult;
use brepkit_operations::tessellate::TriangleMesh;

/// Import a mesh, one handle for each of its bodies.
fn mesh_parts(
    topo: &mut brepkit_topology::Topology,
    mesh: &TriangleMesh,
) -> Result<MeshImportResult, brepkit_io::IoError> {
    let parts = brepkit_io::stl::import::import_mesh_parts(topo, mesh, TOL)?;
    Ok(MeshImportResult {
        solids: parts.solids.into_iter().map(solid_id_to_u32).collect(),
        shells: parts.shells.into_iter().map(shell_id_to_u32).collect(),
    })
}

fn to_json<T: serde::Serialize>(value: &T) -> Result<String, JsError> {
    serde_json::to_string(value).map_err(|e| JsError::new(&e.to_string()))
}

/// Build a mesh with no normals from flat position and index arrays.
fn indexed_mesh(positions: &[f64], indices: &[u32]) -> Result<TriangleMesh, WasmError> {
    use brepkit_math::vec::Point3;

    if !positions.len().is_multiple_of(3) {
        return Err(WasmError::InvalidInput {
            reason: format!(
                "positions length {} is not a multiple of 3",
                positions.len()
            ),
        });
    }
    if !indices.len().is_multiple_of(3) {
        return Err(WasmError::InvalidInput {
            reason: format!("indices length {} is not a multiple of 3", indices.len()),
        });
    }
    Ok(TriangleMesh {
        positions: positions
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect(),
        normals: Vec::new(),
        indices: indices.to_vec(),
    })
}

#[wasm_bindgen]
impl BrepKernel {
    // ── Export ─────────────────────────────────────────────────────

    /// Export a solid to 3MF format (ZIP archive as bytes).
    ///
    /// Returns a `Uint8Array` in JavaScript containing the `.3mf` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "export3mf")]
    pub fn export_3mf(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::threemf::write_threemf(&self.topo, &[solid_id], deflection)?;
        Ok(bytes)
    }

    /// Export a solid to binary STL format.
    ///
    /// Returns a `Uint8Array` containing the `.stl` file.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportStl")]
    pub fn export_stl(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::stl::writer::write_stl(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::stl::writer::StlFormat::Binary,
        )?;
        Ok(bytes)
    }

    /// Export a solid to STL ASCII format.
    ///
    /// Returns the ASCII STL as UTF-8 bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportStlAscii")]
    pub fn export_stl_ascii(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::stl::writer::write_stl(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::stl::writer::StlFormat::Ascii,
        )?;
        Ok(bytes)
    }

    /// Export a solid to OBJ format (UTF-8 string as bytes).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportObj")]
    pub fn export_obj(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let obj_str = brepkit_io::obj::write_obj(&self.topo, &[solid_id], deflection)?;
        Ok(obj_str.into_bytes())
    }

    /// Export a solid to glTF binary (.glb) format.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportGlb")]
    pub fn export_glb(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let glb = brepkit_io::gltf::write_glb(&self.topo, &[solid_id], deflection)?;
        Ok(glb)
    }

    /// Export a solid to PLY format (binary little-endian).
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "exportPly")]
    pub fn export_ply(&self, solid: u32, deflection: f64) -> Result<Vec<u8>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let ply = brepkit_io::ply::write_ply(
            &self.topo,
            &[solid_id],
            deflection,
            brepkit_io::ply::writer::PlyFormat::BinaryLittleEndian,
        )?;
        Ok(ply)
    }

    // ── Import ──────────────────────────────────────────────────────

    /// Import an OBJ file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[wasm_bindgen(js_name = "importObj")]
    pub fn import_obj(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let text = std::str::from_utf8(data).map_err(|e| WasmError::InvalidInput {
            reason: format!("OBJ must be valid UTF-8: {e}"),
        })?;
        let mesh = brepkit_io::obj::read_obj(text)?;
        let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import a GLB (glTF binary) file and return a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or mesh import fails.
    #[wasm_bindgen(js_name = "importGlb")]
    pub fn import_glb(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::gltf::read_glb(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), &mesh, 1e-7)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(solid_id.index() as u32)
    }

    /// Import an STL file (binary or ASCII) and return a solid handle.
    ///
    /// The mesh triangles are converted to planar B-Rep faces with
    /// vertex merging, and triangles that meet at an edge share it. Two or
    /// more closed bodies come back as the pieces of one solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the STL data is malformed or empty, two closed
    /// bodies of the mesh cross, a part of the mesh does not close, or a
    /// body lies in the material of another. `importStlParts` returns each
    /// part of such a mesh.
    #[wasm_bindgen(js_name = "importStl")]
    pub fn import_stl(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let mesh = brepkit_io::stl::reader::read_stl(data)?;
        let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Import a 3MF file and return solid handles.
    ///
    /// Returns handles for each object found in the 3MF archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3MF data is malformed.
    #[wasm_bindgen(js_name = "import3mf")]
    pub fn import_3mf(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let meshes = brepkit_io::threemf::reader::read_threemf(data)?;
        let mut handles = Vec::new();
        for mesh in &meshes {
            let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), mesh, TOL)?;
            handles.push(solid_id_to_u32(solid_id));
        }
        Ok(handles)
    }

    /// Import a triangle mesh from flat vertex/index arrays.
    ///
    /// `positions` is a flat `[x0,y0,z0, x1,y1,z1, ...]` array.
    /// `indices` is a flat `[i0,i1,i2, i3,i4,i5, ...]` array of triangle
    /// vertex indices. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the arrays are malformed, two closed bodies of
    /// the mesh cross, a part of the mesh does not close, or a body lies in
    /// the material of another. `importIndexedMeshParts` returns each part
    /// of such a mesh.
    #[wasm_bindgen(js_name = "importIndexedMesh")]
    pub fn import_indexed_mesh(
        &mut self,
        positions: &[f64],
        indices: &[u32],
    ) -> Result<u32, JsError> {
        let mesh = indexed_mesh(positions, indices)?;
        let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Import an STL file (binary or ASCII), one handle for each body.
    ///
    /// Returns a JSON string `{ solids: number[], shells: number[] }`. A
    /// closed body is a solid, with each inward closed body in it as a
    /// cavity. Two closed bodies that share an edge are two solids. A body
    /// that does not close, and an inward closed body that crosses another
    /// body, is a shell.
    ///
    /// # Errors
    ///
    /// Returns an error if the STL data is malformed or empty.
    #[wasm_bindgen(js_name = "importStlParts")]
    pub fn import_stl_parts(&mut self, data: &[u8]) -> Result<String, JsError> {
        let mesh = brepkit_io::stl::reader::read_stl(data)?;
        to_json(&mesh_parts(self.topo_mut(), &mesh)?)
    }

    /// Import an OBJ file, one handle for each body.
    ///
    /// Returns the same JSON string as `importStlParts`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or has no triangle.
    #[wasm_bindgen(js_name = "importObjParts")]
    pub fn import_obj_parts(&mut self, data: &[u8]) -> Result<String, JsError> {
        let text = std::str::from_utf8(data).map_err(|e| WasmError::InvalidInput {
            reason: format!("OBJ must be valid UTF-8: {e}"),
        })?;
        let mesh = brepkit_io::obj::read_obj(text)?;
        to_json(&mesh_parts(self.topo_mut(), &mesh)?)
    }

    /// Import a GLB (glTF binary) file, one handle for each body.
    ///
    /// Returns the same JSON string as `importStlParts`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file is malformed or has no triangle.
    #[wasm_bindgen(js_name = "importGlbParts")]
    pub fn import_glb_parts(&mut self, data: &[u8]) -> Result<String, JsError> {
        let mesh = brepkit_io::gltf::read_glb(data)?;
        to_json(&mesh_parts(self.topo_mut(), &mesh)?)
    }

    /// Import a 3MF file, one entry for each object and one handle for each
    /// body of the object.
    ///
    /// Returns a JSON string: an array of `{ solids, shells }`, in the
    /// order of the objects in the archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the 3MF data is malformed.
    #[wasm_bindgen(js_name = "import3mfParts")]
    pub fn import_3mf_parts(&mut self, data: &[u8]) -> Result<String, JsError> {
        let meshes = brepkit_io::threemf::reader::read_threemf(data)?;
        let mut objects = Vec::with_capacity(meshes.len());
        for mesh in &meshes {
            objects.push(mesh_parts(self.topo_mut(), mesh)?);
        }
        to_json(&objects)
    }

    /// Import a triangle mesh from flat vertex/index arrays, one handle for
    /// each body.
    ///
    /// The arrays are those of `importIndexedMesh`. Returns the same JSON
    /// string as `importStlParts`.
    ///
    /// # Errors
    ///
    /// Returns an error if the arrays are malformed or have no triangle.
    #[wasm_bindgen(js_name = "importIndexedMeshParts")]
    pub fn import_indexed_mesh_parts(
        &mut self,
        positions: &[f64],
        indices: &[u32],
    ) -> Result<String, JsError> {
        let mesh = indexed_mesh(positions, indices)?;
        to_json(&mesh_parts(self.topo_mut(), &mesh)?)
    }

    /// Export a solid to STEP AP203 format.
    ///
    /// Returns the STEP file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportStep")]
    pub fn export_step(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let step_str = brepkit_io::step::writer::write_step(&self.topo, &[solid_id])?;
        Ok(step_str.into_bytes())
    }

    /// Import a STEP file and return solid handles.
    ///
    /// Returns handles for each solid found in the STEP file.
    ///
    /// # Errors
    ///
    /// Returns an error if the STEP data is malformed.
    #[wasm_bindgen(js_name = "importStep")]
    pub fn import_step(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("STEP data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::step::reader::read_step(text, self.topo_mut())?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    // ── IGES Import/Export ────────────────────────────────────────

    /// Export a solid to IGES format.
    ///
    /// Returns the IGES file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportIges")]
    pub fn export_iges(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let iges_str = brepkit_io::iges::writer::write_iges(&self.topo, &[solid_id])?;
        Ok(iges_str.into_bytes())
    }

    /// Export several solids to one IGES document, each of its faces a trimmed
    /// surface that refers to its other entities by number, so documents written
    /// one solid at a time cannot simply be joined. A handle given twice is
    /// written once.
    ///
    /// Returns the IGES file as a UTF-8 encoded byte vector.
    ///
    /// # Errors
    ///
    /// Returns an error if a solid handle is invalid or export fails.
    #[wasm_bindgen(js_name = "exportIgesSolids")]
    pub fn export_iges_solids(&self, solids: Vec<u32>) -> Result<Vec<u8>, JsError> {
        let solid_ids = solids
            .into_iter()
            .map(|solid| self.resolve_solid(solid))
            .collect::<Result<Vec<_>, _>>()?;
        let iges_str = brepkit_io::iges::writer::write_iges(&self.topo, &solid_ids)?;
        Ok(iges_str.into_bytes())
    }

    /// Import an IGES file and return solid handles.
    ///
    /// # Errors
    ///
    /// Returns an error if the IGES data is malformed.
    #[wasm_bindgen(js_name = "importIges")]
    pub fn import_iges(&mut self, data: &[u8]) -> Result<Vec<u32>, JsError> {
        let text = std::str::from_utf8(data)
            .map_err(|e| JsError::new(&format!("IGES data is not valid UTF-8: {e}")))?;
        let solid_ids = brepkit_io::iges::reader::read_iges(text, self.topo_mut())?;
        Ok(solid_ids.iter().map(|id| solid_id_to_u32(*id)).collect())
    }

    // ── Arena debug serialization ─────────────────────────────────

    /// Serialize a solid's complete in-memory topology sub-arena to bytes.
    ///
    /// Captures every vertex, edge, wire, face, shell reachable from the
    /// solid with byte-exact f64 values (no geometry re-derivation or
    /// tolerance normalization). Unlike STEP/IGES export, this preserves the
    /// kernel's exact in-memory state — intended for capturing live operands
    /// and replaying them in a native Rust harness to reproduce
    /// sub-ULP-sensitive boolean behavior.
    ///
    /// Returns a `Uint8Array` consumable by `brepkit_io::arena_io::deserialize_solid`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or serialization fails.
    #[wasm_bindgen(js_name = "serializeSolid")]
    pub fn serialize_solid(&self, solid: u32) -> Result<Vec<u8>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let bytes = brepkit_io::arena_io::serialize_solid(&self.topo, solid_id)?;
        Ok(bytes)
    }

    /// Reconstruct a solid from a buffer produced by [`Self::serialize_solid`].
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is malformed or reconstruction fails.
    #[wasm_bindgen(js_name = "deserializeSolid")]
    pub fn deserialize_solid(&mut self, data: &[u8]) -> Result<u32, JsError> {
        let solid_id = brepkit_io::arena_io::deserialize_solid(data, self.topo_mut())?;
        Ok(solid_id_to_u32(solid_id))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    const CUBE: [u32; 36] = [
        0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 3, 7, 6, 3, 6, 2, 0, 4, 7, 0, 7, 3,
        1, 2, 6, 1, 6, 5,
    ];

    fn cube_positions(at: f64) -> Vec<f64> {
        let mut out = Vec::new();
        for (x, y, z) in [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ] {
            out.extend([at + x * 10.0, y * 10.0, z * 10.0]);
        }
        out
    }

    #[test]
    fn an_indexed_cube_is_one_solid_in_one_piece() {
        let mut k = BrepKernel::new();
        let solid = k.import_indexed_mesh(&cube_positions(0.0), &CUBE).unwrap();
        assert_eq!(k.get_solid_shells(solid).unwrap().len(), 1);
        assert_eq!(k.get_solid_pieces(solid).unwrap(), vec![solid]);
    }

    #[test]
    fn an_stl_box_is_one_solid_in_one_piece() {
        let mut k = BrepKernel::new();
        let source = k.make_box_solid(10.0, 10.0, 10.0).unwrap();
        let stl = k.export_stl(source, 0.1).unwrap();
        let solid = k.import_stl(&stl).unwrap();
        assert_eq!(k.get_solid_shells(solid).unwrap().len(), 1);
        assert_eq!(k.get_solid_pieces(solid).unwrap(), vec![solid]);
    }

    #[test]
    fn two_indexed_cubes_and_a_sheet_are_two_solids_and_a_shell() {
        let mut positions = cube_positions(0.0);
        positions.extend(cube_positions(20.0));
        let mut indices = CUBE.to_vec();
        indices.extend(CUBE.iter().map(|i| i + 8));
        positions.extend([0.0, 0.0, 50.0, 1.0, 0.0, 50.0, 1.0, 1.0, 50.0]);
        indices.extend([16, 17, 18]);

        let mut k = BrepKernel::new();
        let mesh = indexed_mesh(&positions, &indices).unwrap();
        let parts = mesh_parts(k.topo_mut(), &mesh).unwrap();
        assert_eq!((parts.solids.len(), parts.shells.len()), (2, 1));
        assert_eq!(
            serde_json::to_string(&parts).unwrap(),
            format!(
                "{{\"solids\":[{},{}],\"shells\":[{}]}}",
                parts.solids[0], parts.solids[1], parts.shells[0]
            )
        );
        assert_eq!(k.get_shell_faces(parts.shells[0]).unwrap().len(), 1);
    }

    #[test]
    fn two_indexed_cubes_that_share_an_edge_are_two_solids() {
        let mut positions = cube_positions(0.0);
        positions.extend(
            cube_positions(10.0)
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1] + 10.0, p[2]]),
        );
        let mut indices = CUBE.to_vec();
        indices.extend(CUBE.iter().map(|i| i + 8));

        let mut k = BrepKernel::new();
        let solid = k.import_indexed_mesh(&positions, &indices).unwrap();
        assert_eq!(k.get_solid_pieces(solid).unwrap().len(), 2);
        let mesh = indexed_mesh(&positions, &indices).unwrap();
        let parts = mesh_parts(k.topo_mut(), &mesh).unwrap();
        assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    }

    #[test]
    fn two_indexed_cubes_that_cross_are_not_one_solid() {
        let mut positions = cube_positions(0.0);
        positions.extend(
            cube_positions(5.0)
                .chunks_exact(3)
                .flat_map(|p| [p[0], p[1] + 5.0, p[2] + 5.0]),
        );
        let mut indices = CUBE.to_vec();
        indices.extend(CUBE.iter().map(|i| i + 8));

        let mut k = BrepKernel::new();
        let mesh = indexed_mesh(&positions, &indices).unwrap();
        assert!(brepkit_io::stl::import::import_mesh(k.topo_mut(), &mesh, TOL).is_err());
        let parts = mesh_parts(k.topo_mut(), &mesh).unwrap();
        assert_eq!((parts.solids.len(), parts.shells.len()), (2, 0));
    }

    #[test]
    fn an_open_indexed_mesh_is_no_solid() {
        let mut k = BrepKernel::new();
        let mesh =
            indexed_mesh(&[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0], &[0, 1, 2]).unwrap();
        assert!(brepkit_io::stl::import::import_mesh(k.topo_mut(), &mesh, TOL).is_err());
        assert!(indexed_mesh(&[0.0, 0.0], &[0, 1, 2]).is_err());
    }
}

//! File I/O (import/export) bindings.

#![cfg(feature = "io")]
#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::{WasmError, validate_positive};
use crate::handles::solid_id_to_u32;
use crate::helpers::TOL;
use crate::kernel::BrepKernel;

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
    /// vertex merging.
    ///
    /// # Errors
    ///
    /// Returns an error if the STL data is malformed or empty.
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
    /// Returns an error if the arrays are malformed or mesh import fails.
    #[wasm_bindgen(js_name = "importIndexedMesh")]
    pub fn import_indexed_mesh(
        &mut self,
        positions: &[f64],
        indices: &[u32],
    ) -> Result<u32, JsError> {
        use brepkit_math::vec::Point3;

        if positions.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "positions length {} is not a multiple of 3",
                    positions.len()
                ),
            }
            .into());
        }
        if indices.len() % 3 != 0 {
            return Err(WasmError::InvalidInput {
                reason: format!("indices length {} is not a multiple of 3", indices.len()),
            }
            .into());
        }

        let verts: Vec<Point3> = positions
            .chunks_exact(3)
            .map(|c| Point3::new(c[0], c[1], c[2]))
            .collect();
        let normals = Vec::new();

        let mesh = brepkit_operations::tessellate::TriangleMesh {
            positions: verts,
            normals,
            indices: indices.to_vec(),
        };

        let solid_id = brepkit_io::stl::import::import_mesh(self.topo_mut(), &mesh, TOL)?;
        Ok(solid_id_to_u32(solid_id))
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
}

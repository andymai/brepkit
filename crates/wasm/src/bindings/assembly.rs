//! Assembly management bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use crate::error::WasmError;
use crate::handles::solid_id_to_u32;
use crate::helpers::{mat4_to_array, parse_mat4};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    /// Create a new empty assembly. Returns an assembly index.
    #[wasm_bindgen(js_name = "assemblyNew")]
    pub fn assembly_new(&mut self, name: &str) -> u32 {
        self.assemblies
            .push(brepkit_operations::assembly::Assembly::new(name));
        #[allow(clippy::cast_possible_truncation)]
        let idx = (self.assemblies.len() - 1) as u32;
        idx
    }

    /// Add a root component to an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddRoot")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_root(
        &mut self,
        assembly: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_root_component(name, solid_id, mat);
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Add a child component to a parent in an assembly.
    ///
    /// Returns the component ID.
    #[wasm_bindgen(js_name = "assemblyAddChild")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn assembly_add_child(
        &mut self,
        assembly: u32,
        parent: u32,
        name: &str,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let mat = parse_mat4(&matrix)?;
        let asm = self
            .assemblies
            .get_mut(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let cid = asm.add_child_component(parent as usize, name, solid_id, mat)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(cid as u32)
    }

    /// Flatten an assembly into `[(solid, matrix), ...]`.
    ///
    /// Returns a JSON string: `[{"solid": u32, "matrix": [16 floats]}, ...]`.
    #[wasm_bindgen(js_name = "assemblyFlatten")]
    pub fn assembly_flatten(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let flat = asm.flatten();
        let entries: Vec<serde_json::Value> = flat
            .iter()
            .map(|(solid_id, mat)| {
                serde_json::json!({
                    "solid": solid_id_to_u32(*solid_id),
                    "matrix": mat4_to_array(mat),
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }

    /// Get the bill of materials for an assembly.
    ///
    /// Returns a JSON string: `[{"name": "...", "solidIndex": n, "instanceCount": n}, ...]`.
    #[wasm_bindgen(js_name = "assemblyBom")]
    pub fn assembly_bom(&self, assembly: u32) -> Result<String, JsError> {
        let asm = self
            .assemblies
            .get(assembly as usize)
            .ok_or(WasmError::InvalidHandle {
                entity: "assembly",
                index: assembly as usize,
            })?;
        let bom = asm.bill_of_materials();
        let entries: Vec<serde_json::Value> = bom
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "name": entry.name,
                    "solidIndex": entry.solid_index,
                    "instanceCount": entry.instance_count,
                })
            })
            .collect();
        Ok(serde_json::Value::Array(entries).to_string())
    }
}

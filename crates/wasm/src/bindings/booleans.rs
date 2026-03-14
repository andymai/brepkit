//! Boolean operation bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use brepkit_operations::boolean::{BooleanOp, boolean};

use crate::handles::solid_id_to_u32;
use crate::helpers::{build_triangle_mesh, parse_boolean_op, triangle_mesh_to_js};
use crate::kernel::BrepKernel;
use crate::shapes::JsMesh;

#[wasm_bindgen]
impl BrepKernel {
    // ── Boolean operations ──────────────────────────────────────────

    /// Fuse (union) two solids into one.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "fuse")]
    pub fn fuse(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Fuse, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Cut (subtract) solid `b` from solid `a`.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "cut")]
    pub fn cut(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Cut, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Intersect two solids, keeping only their common volume.
    ///
    /// Returns a new solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty result.
    #[wasm_bindgen(js_name = "intersect")]
    pub fn intersect_solids(&mut self, a: u32, b: u32) -> Result<u32, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = boolean(&mut self.topo, BooleanOp::Intersect, a_id, b_id)?;
        Ok(solid_id_to_u32(result))
    }

    /// Cut a target solid by multiple tool solids in a single pass.
    ///
    /// This is more efficient than sequential `cut()` calls when many tools
    /// are applied to the same target — it avoids re-processing unchanged
    /// faces at each step.
    ///
    /// `tool_ids` is a JS `Uint32Array` or array of solid handles.
    ///
    /// # Errors
    ///
    /// Returns an error if any handle is invalid or the operation fails.
    #[wasm_bindgen(js_name = "compoundCut")]
    pub fn compound_cut(&mut self, target: u32, tool_ids: &[u32]) -> Result<u32, JsError> {
        let target_id = self.resolve_solid(target)?;
        let tools: Vec<brepkit_topology::solid::SolidId> = tool_ids
            .iter()
            .map(|&h| self.resolve_solid(h))
            .collect::<Result<Vec<_>, _>>()?;
        let result = brepkit_operations::boolean::compound_cut(
            &mut self.topo,
            target_id,
            &tools,
            brepkit_operations::boolean::BooleanOptions::default(),
        )?;
        Ok(solid_id_to_u32(result))
    }

    // ── Boolean operations with evolution tracking ─────────────────

    /// Fuse (union) two solids and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "fuseWithEvolution")]
    pub fn fuse_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Fuse,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    /// Cut (subtract) solid `b` from solid `a` and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty or non-manifold result.
    #[wasm_bindgen(js_name = "cutWithEvolution")]
    pub fn cut_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Cut,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    /// Intersect two solids and return evolution tracking data.
    ///
    /// Returns a JSON string: `{"solid": <u32>, "evolution": {...}}`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid or the operation
    /// produces an empty result.
    #[wasm_bindgen(js_name = "intersectWithEvolution")]
    pub fn intersect_with_evolution(&mut self, a: u32, b: u32) -> Result<JsValue, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let (result, evo) = brepkit_operations::boolean::boolean_with_evolution(
            &mut self.topo,
            BooleanOp::Intersect,
            a_id,
            b_id,
        )?;
        let json = format!(
            "{{\"solid\":{},\"evolution\":{}}}",
            solid_id_to_u32(result),
            evo.to_json()
        );
        Ok(JsValue::from_str(&json))
    }

    /// Perform a mesh boolean on raw triangle data.
    ///
    /// Returns a `JsMesh` with the result.
    #[wasm_bindgen(js_name = "meshBoolean")]
    #[allow(
        clippy::needless_pass_by_value,
        clippy::too_many_arguments,
        clippy::unused_self
    )]
    pub fn mesh_boolean(
        &self,
        positions_a: Vec<f64>,
        indices_a: Vec<u32>,
        positions_b: Vec<f64>,
        indices_b: Vec<u32>,
        op: &str,
        tolerance: f64,
    ) -> Result<JsMesh, JsError> {
        let mesh_a = build_triangle_mesh(&positions_a, &indices_a)?;
        let mesh_b = build_triangle_mesh(&positions_b, &indices_b)?;
        let bool_op = parse_boolean_op(op)?;
        let result =
            brepkit_operations::mesh_boolean::mesh_boolean(&mesh_a, &mesh_b, bool_op, tolerance)?;
        Ok(triangle_mesh_to_js(&result.mesh))
    }
}

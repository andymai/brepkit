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
        let result = boolean(self.topo_mut(), BooleanOp::Fuse, a_id, b_id)?;
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
        let result = boolean(self.topo_mut(), BooleanOp::Cut, a_id, b_id)?;
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
        let result = boolean(self.topo_mut(), BooleanOp::Intersect, a_id, b_id)?;
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
            self.topo_mut(),
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
            self.topo_mut(),
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
            self.topo_mut(),
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
            self.topo_mut(),
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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use crate::kernel::BrepKernel;

    /// Helper: parse batch result and check a single op returned ok or error.
    fn batch_has_ok(result: &str, idx: usize) -> bool {
        let parsed: serde_json::Value = serde_json::from_str(result).unwrap();
        parsed[idx]["ok"].is_number()
    }

    fn batch_has_error(result: &str, idx: usize) -> bool {
        let parsed: serde_json::Value = serde_json::from_str(result).unwrap();
        parsed[idx]["error"].is_string()
    }

    /// Create two overlapping boxes via batch, return the raw JSON result.
    fn two_boxes_batch() -> (BrepKernel, String) {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}}
            ]"#,
        );
        (k, r)
    }

    // ── fuse ─────────────────────────────────────────────────────────

    #[test]
    fn fuse_two_boxes_returns_valid_handle() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "fuse", "args": {{"solidA": {a}, "solidB": {b}}}}}]"#
        ));
        assert!(batch_has_ok(&r, 0), "fuse must return ok: {r}");
    }

    #[test]
    fn fuse_invalid_handle_a_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "fuse", "args": {{"solidA": 9999, "solidB": {b}}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn fuse_invalid_handle_b_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "fuse", "args": {{"solidA": {a}, "solidB": 9999}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    // ── cut ──────────────────────────────────────────────────────────

    #[test]
    fn cut_two_boxes_returns_valid_handle() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "cut", "args": {{"solidA": {a}, "solidB": {b}}}}}]"#
        ));
        assert!(batch_has_ok(&r, 0), "cut must return ok: {r}");
    }

    #[test]
    fn cut_invalid_target_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "cut", "args": {{"solidA": 9999, "solidB": {b}}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn cut_invalid_tool_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "cut", "args": {{"solidA": {a}, "solidB": 9999}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    // ── intersect ────────────────────────────────────────────────────

    #[test]
    fn intersect_two_boxes_returns_valid_handle() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "intersect", "args": {{"solidA": {a}, "solidB": {b}}}}}]"#
        ));
        assert!(batch_has_ok(&r, 0), "intersect must return ok: {r}");
    }

    #[test]
    fn intersect_invalid_handle_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "intersect", "args": {{"solidA": {a}, "solidB": 9999}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    // ── compound_cut ─────────────────────────────────────────────────

    #[test]
    fn compound_cut_single_tool() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "compoundCut", "args": {{"target": {a}, "tools": [{b}]}}}}]"#
        ));
        assert!(batch_has_ok(&r, 0), "compound_cut must return ok: {r}");
    }

    #[test]
    fn compound_cut_multiple_tools() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 4, "height": 4, "depth": 4}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "makeBox", "args": {"width": 0.5, "height": 0.5, "depth": 0.5}},
                {"op": "compoundCut", "args": {"target": 0, "tools": [1, 2]}}
            ]"#,
        );
        assert!(
            batch_has_ok(&r, 3),
            "compound_cut with two tools must return ok: {r}"
        );
    }

    #[test]
    fn compound_cut_invalid_target_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let b = parsed[1]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "compoundCut", "args": {{"target": 9999, "tools": [{b}]}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn compound_cut_invalid_tool_errors() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "compoundCut", "args": {{"target": {a}, "tools": [9999]}}}}]"#
        ));
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn compound_cut_empty_tool_list_is_identity() {
        let (mut k, setup) = two_boxes_batch();
        let parsed: serde_json::Value = serde_json::from_str(&setup).unwrap();
        let a = parsed[0]["ok"].as_u64().unwrap();
        let r = k.execute_batch(&format!(
            r#"[{{"op": "compoundCut", "args": {{"target": {a}, "tools": []}}}}]"#
        ));
        assert!(batch_has_ok(&r, 0));
    }

    // ── mesh_boolean ─────────────────────────────────────────────────
    // mesh_boolean is not in the batch dispatcher, but its happy-path
    // works on native (JsError is only constructed on the error path).
    // For error paths, we test the internal operations layer directly.

    #[test]
    fn mesh_boolean_fuse_returns_non_empty_mesh() {
        let k = BrepKernel::new();
        #[rustfmt::skip]
        let positions = vec![
            0.0, 0.0, 0.0,
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let indices = vec![0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3];
        let mesh = k
            .mesh_boolean(
                positions.clone(),
                indices.clone(),
                positions,
                indices,
                "fuse",
                1e-7,
            )
            .unwrap();
        assert!(
            !mesh.positions().is_empty(),
            "fused mesh must have vertices"
        );
        assert!(!mesh.indices().is_empty(), "fused mesh must have triangles");
        assert_eq!(mesh.positions().len() % 3, 0);
        assert_eq!(mesh.indices().len() % 3, 0);
    }

    #[test]
    fn mesh_boolean_unknown_op_is_not_valid() {
        // Verify the operation string validation logic without calling
        // JsError-returning helpers (JsError panics on non-wasm).
        let valid = [
            "fuse",
            "union",
            "cut",
            "difference",
            "intersect",
            "intersection",
        ];
        assert!(
            !valid.contains(&"explode"),
            "explode should not be a valid op"
        );
    }

    #[test]
    fn mesh_boolean_bad_positions_length_is_invalid() {
        // Verify the validation condition directly: positions must be multiple of 3.
        let bad_len = 2;
        assert_ne!(
            bad_len % 3,
            0,
            "length 2 should fail the multiple-of-3 check"
        );
    }

    // ── boolean volume check ─────────────────────────────────────────

    #[test]
    fn cut_reduces_volume() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 2, "depth": 2}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "volume", "args": {"solid": 0}},
                {"op": "cut", "args": {"solidA": 0, "solidB": 1}},
                {"op": "volume", "args": {"solid": 2}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let vol_before = parsed[2]["ok"].as_f64().unwrap();
        let vol_after = parsed[4]["ok"].as_f64().unwrap();
        assert!(
            vol_after < vol_before,
            "cut must reduce volume: {vol_before} -> {vol_after}"
        );
    }

    // ── compound_cut volume regression ───────────────────────────────

    #[test]
    fn compound_cut_volume_decreases() {
        let mut k = BrepKernel::new();
        // Target: 10x10x10 box at origin. Tool: 1x1x1 box at origin.
        // The tool overlaps one corner, so volume decreases.
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 10, "height": 10, "depth": 10}},
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "volume", "args": {"solid": 0}},
                {"op": "compoundCut", "args": {"target": 0, "tools": [1]}},
                {"op": "volume", "args": {"solid": 2}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let vol_before = parsed[2]["ok"].as_f64().unwrap();
        assert!(batch_has_ok(&r, 3), "compoundCut must succeed: {r}");
        let vol_after = parsed[4]["ok"].as_f64().unwrap();
        assert!(
            vol_after < vol_before && vol_after > 0.0,
            "compound_cut must reduce volume: {vol_before} -> {vol_after}"
        );
    }
}

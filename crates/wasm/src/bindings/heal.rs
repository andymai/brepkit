//! Shape healing, validation, and feature recognition bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use brepkit_topology::face::Face;

use crate::handles::{face_id_to_u32, solid_id_to_u32};
use crate::helpers::{TOL, serialize_feature};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // -- Sewing ----------------------------------------------------------------

    /// Sew loose faces into a connected solid.
    ///
    /// `face_handles` is an array of face handles. Returns a solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if fewer than 2 faces or sewing fails.
    #[wasm_bindgen(js_name = "sewFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn sew_faces(&mut self, face_handles: Vec<u32>, tolerance: f64) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let solid = brepkit_operations::sew::sew_faces(self.topo_mut(), &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    /// Create a solid from a set of faces by sewing them together.
    ///
    /// Alias for `sewFaces` with a default tolerance. This is the equivalent
    /// of sewing faces into a closed shell and building a solid.
    #[wasm_bindgen(js_name = "makeSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn make_solid_from_faces(&mut self, face_handles: Vec<u32>) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let tolerance = brepkit_math::tolerance::Tolerance::new().linear;
        let solid = brepkit_operations::sew::sew_faces(self.topo_mut(), &face_ids, tolerance)?;
        Ok(solid_id_to_u32(solid))
    }

    /// Remove all holes from a face, returning a new face with only the outer wire.
    #[wasm_bindgen(js_name = "removeHolesFromFace")]
    pub fn remove_holes_from_face(&mut self, face: u32) -> Result<u32, JsError> {
        let face_id = self.resolve_face(face)?;
        let face_data = self.topo.face(face_id)?;
        let outer_wire = face_data.outer_wire();
        let surface = face_data.surface().clone();
        let new_face = Face::new(outer_wire, vec![], surface);
        let fid = self.topo_mut().add_face(new_face);
        Ok(face_id_to_u32(fid))
    }

    /// Weld shells and faces into a single solid by sewing.
    ///
    /// Accepts an array of face handles from potentially different shells.
    /// Sews all faces together into a single solid.
    #[wasm_bindgen(js_name = "weldShellsAndFaces")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn weld_shells_and_faces(
        &mut self,
        face_handles: Vec<u32>,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let face_ids: Vec<brepkit_topology::face::FaceId> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<_, _>>()?;
        let tol = if tolerance > 0.0 {
            tolerance
        } else {
            brepkit_math::tolerance::Tolerance::new().linear
        };
        let solid = brepkit_operations::sew::sew_faces(self.topo_mut(), &face_ids, tol)?;
        Ok(solid_id_to_u32(solid))
    }

    // -- Healing ---------------------------------------------------------------

    /// Unify adjacent faces that lie on the same geometric surface.
    ///
    /// Merges co-surface face fragments (produced by boolean operations)
    /// back into single faces, reducing face count and improving topology.
    /// Returns the number of faces removed.
    #[wasm_bindgen(js_name = "unifyFaces")]
    pub fn unify_faces(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let removed = brepkit_operations::heal::unify_faces(self.topo_mut(), solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(removed as u32)
    }

    /// Convert all analytic geometry in a solid to NURBS representation.
    ///
    /// Replaces planes, cylinders, cones, spheres, tori with NURBS surfaces and
    /// lines, circles, ellipses with NURBS curves. NURBS surfaces and curves
    /// already in the model are left untouched. Returns the number of entities
    /// converted.
    ///
    /// Equivalent to OCCT's `BRepBuilderAPI_NurbsConvert`. Stored pcurves are
    /// dropped during conversion — callers that depend on pcurves should
    /// recompute them afterwards.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or conversion fails.
    #[wasm_bindgen(js_name = "convertToBspline")]
    pub fn convert_to_bspline(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::convert_to_bspline(self.topo_mut(), solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Heal a solid topology.
    ///
    /// Returns the number of issues fixed.
    #[wasm_bindgen(js_name = "healSolid")]
    pub fn heal_solid(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::heal::heal_solid(self.topo_mut(), solid_id, TOL)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok((report.vertices_merged
            + report.degenerate_edges_removed
            + report.orientations_fixed
            + report.wire_gaps_closed
            + report.small_faces_removed
            + report.duplicate_faces_removed) as u32)
    }

    /// Validate, heal, and re-validate a solid in one pass.
    ///
    /// Returns the number of remaining validation errors after repair.
    /// A return value of 0 means the solid is valid after repair.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "repairSolid")]
    pub fn repair_solid(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::heal::repair_solid(self.topo_mut(), solid_id, TOL)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.after.error_count() as u32)
    }

    /// Remove degenerate (zero-length) edges from a solid.
    ///
    /// Returns the number of edges removed.
    #[wasm_bindgen(js_name = "removeDegenerateEdges")]
    pub fn remove_degenerate_edges(&mut self, solid: u32, tolerance: f64) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::remove_degenerate_edges(
            self.topo_mut(),
            solid_id,
            tolerance,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    /// Fix face orientations to ensure consistent outward normals.
    ///
    /// Returns the number of faces fixed.
    #[wasm_bindgen(js_name = "fixFaceOrientations")]
    pub fn fix_face_orientations(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::fix_face_orientations(self.topo_mut(), solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }

    // -- Defeaturing & Feature Recognition -------------------------------------

    /// Remove specified faces from a solid (defeaturing).
    ///
    /// `face_handles` is an array of face handles to remove.
    /// Returns a new solid handle.
    #[wasm_bindgen(js_name = "defeature")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn defeature(&mut self, solid: u32, face_handles: Vec<u32>) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let face_ids: Vec<_> = face_handles
            .iter()
            .map(|&h| self.resolve_face(h))
            .collect::<Result<Vec<_>, _>>()?;
        let result =
            brepkit_operations::defeature::defeature(self.topo_mut(), solid_id, &face_ids)?;
        Ok(solid_id_to_u32(result))
    }

    /// Detect small features (faces below an area threshold).
    ///
    /// Returns an array of face handles.
    #[wasm_bindgen(js_name = "detectSmallFeatures")]
    pub fn detect_small_features(
        &self,
        solid: u32,
        area_threshold: f64,
        deflection: f64,
    ) -> Result<Vec<u32>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let faces = brepkit_operations::defeature::detect_small_features(
            &self.topo,
            solid_id,
            area_threshold,
            deflection,
        )?;
        Ok(faces.iter().map(|f| face_id_to_u32(*f)).collect())
    }

    /// Recognize geometric features in a solid.
    ///
    /// Returns a JSON string describing the recognized features.
    #[wasm_bindgen(js_name = "recognizeFeatures")]
    pub fn recognize_features(&self, solid: u32, deflection: f64) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let features = brepkit_operations::feature_recognition::recognize_features(
            &self.topo, solid_id, deflection,
        )?;
        let json_features: Vec<serde_json::Value> =
            features.iter().map(serialize_feature).collect();
        Ok(serde_json::Value::Array(json_features).to_string())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use crate::kernel::BrepKernel;

    #[test]
    fn convert_to_bspline_returns_count_and_solid() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeCylinder", "args": {"radius": 1, "height": 2}},
                {"op": "convertToBspline", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let ok = parsed[1]["ok"].as_object().expect("expected ok object");
        assert!(ok.get("solid").is_some(), "missing 'solid' field");
        let converted = ok["converted"].as_u64().expect("expected 'converted' u64");
        // Cylinder has 3 faces (lateral + 2 caps) and 3 edges (2 circles + 1 seam)
        // → 6 conversions on first run.
        assert!(converted >= 5, "expected >=5 conversions, got {converted}");
    }

    #[test]
    fn convert_to_bspline_invalid_handle_errors() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "convertToBspline", "args": {"solid": 999}}]"#);
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert!(
            parsed[0]["error"].is_string(),
            "expected error for invalid handle, got: {}",
            parsed[0]
        );
    }

    #[test]
    fn convert_to_bspline_idempotent_second_call_is_zero() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "convertToBspline", "args": {"solid": 0}},
                {"op": "convertToBspline", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let first = parsed[1]["ok"]["converted"].as_u64().unwrap();
        let second = parsed[2]["ok"]["converted"].as_u64().unwrap();
        assert!(first > 0);
        assert_eq!(second, 0, "second pass should convert nothing");
    }
}

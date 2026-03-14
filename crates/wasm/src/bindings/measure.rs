//! Measurement, validation, and distance query bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use brepkit_math::vec::Point3;
use brepkit_operations::measure;

use crate::error::validate_positive;
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    // ── Measurement ───────────────────────────────────────────────

    /// Compute the axis-aligned bounding box of a solid.
    ///
    /// Returns `[min_x, min_y, min_z, max_x, max_y, max_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or has no vertices.
    #[wasm_bindgen(js_name = "boundingBox")]
    pub fn bounding_box(&self, solid: u32) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let aabb = measure::solid_bounding_box(&self.topo, solid_id)?;
        Ok(vec![
            aabb.min.x(),
            aabb.min.y(),
            aabb.min.z(),
            aabb.max.x(),
            aabb.max.y(),
            aabb.max.z(),
        ])
    }

    /// Compute the volume of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "volume")]
    pub fn volume(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_volume(&self.topo, solid_id, deflection)?)
    }

    /// Compute the total surface area of a solid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "surfaceArea")]
    pub fn surface_area(&self, solid: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        Ok(measure::solid_surface_area(
            &self.topo, solid_id, deflection,
        )?)
    }

    /// Compute the area of a single face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid or tessellation fails.
    #[wasm_bindgen(js_name = "faceArea")]
    pub fn face_area(&self, face: u32, deflection: f64) -> Result<f64, JsError> {
        validate_positive(deflection, "deflection")?;
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_area(&self.topo, face_id, deflection)?)
    }

    /// Compute the center of mass of a solid (uniform density).
    ///
    /// Returns `[x, y, z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid has zero volume or tessellation fails.
    #[wasm_bindgen(js_name = "centerOfMass")]
    pub fn center_of_mass(&self, solid: u32, deflection: f64) -> Result<Vec<f64>, JsError> {
        validate_positive(deflection, "deflection")?;
        let solid_id = self.resolve_solid(solid)?;
        let com = measure::solid_center_of_mass(&self.topo, solid_id, deflection)?;
        Ok(vec![com.x(), com.y(), com.z()])
    }

    /// Classify a point relative to a solid: inside, outside, or on boundary.
    ///
    /// Returns `"inside"`, `"outside"`, or `"boundary"`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "classifyPoint")]
    pub fn classify_point(
        &self,
        solid: u32,
        x: f64,
        y: f64,
        z: f64,
        tolerance: f64,
    ) -> Result<String, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let point = brepkit_math::vec::Point3::new(x, y, z);
        let result = brepkit_operations::classify::classify_point(
            &self.topo, solid_id, point, 0.1, tolerance,
        )?;
        Ok(match result {
            brepkit_operations::classify::PointClassification::Inside => "inside".into(),
            brepkit_operations::classify::PointClassification::Outside => "outside".into(),
            brepkit_operations::classify::PointClassification::OnBoundary => "boundary".into(),
        })
    }

    /// Compute the length of an edge.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "edgeLength")]
    pub fn edge_length(&self, edge: u32) -> Result<f64, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        Ok(measure::edge_length(&self.topo, edge_id)?)
    }

    /// Compute the perimeter of a face.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "facePerimeter")]
    pub fn face_perimeter(&self, face: u32) -> Result<f64, JsError> {
        let face_id = self.resolve_face(face)?;
        Ok(measure::face_perimeter(&self.topo, face_id)?)
    }

    /// Validate a solid, returning the number of errors found.
    ///
    /// Returns 0 if the solid is valid.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "validateSolid")]
    pub fn validate_solid(&self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::validate::validate_solid(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.error_count() as u32)
    }

    /// Validate a solid with relaxed checks suitable for assembled geometry.
    ///
    /// Operations like boolean, fillet, and shell produce geometrically
    /// correct shapes that may not have fully manifold topology (faces
    /// from different operations may not share edges). This validation
    /// skips Euler characteristic, boundary edge, non-manifold edge, and
    /// shell connectivity checks.
    ///
    /// Returns 0 if the solid passes all structural checks.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "validateSolidRelaxed")]
    pub fn validate_solid_relaxed(&self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let report = brepkit_operations::validate::validate_solid_relaxed(&self.topo, solid_id)?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(report.error_count() as u32)
    }

    // ── Distance ──────────────────────────────────────────────────

    /// Compute minimum distance from a point to a solid.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "pointToSolidDistance")]
    pub fn point_to_solid_distance(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        solid: u32,
    ) -> Result<Vec<f64>, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::distance::point_to_solid_distance(
            &self.topo,
            Point3::new(px, py, pz),
            solid_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    /// Compute minimum distance between two solids.
    ///
    /// Returns `[distance]`.
    ///
    /// # Errors
    ///
    /// Returns an error if either solid handle is invalid.
    #[wasm_bindgen(js_name = "solidToSolidDistance")]
    pub fn solid_to_solid_distance(&self, a: u32, b: u32) -> Result<f64, JsError> {
        let a_id = self.resolve_solid(a)?;
        let b_id = self.resolve_solid(b)?;
        let result = brepkit_operations::distance::solid_to_solid_distance(&self.topo, a_id, b_id)?;
        Ok(result.distance)
    }

    /// Compute minimum distance from a point to a face.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the face handle is invalid.
    #[wasm_bindgen(js_name = "pointToFaceDistance")]
    pub fn point_to_face_distance(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        face: u32,
    ) -> Result<Vec<f64>, JsError> {
        let face_id = self.resolve_face(face)?;
        let result = brepkit_operations::distance::point_to_face(
            &self.topo,
            Point3::new(px, py, pz),
            face_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }

    /// Compute minimum distance from a point to an edge.
    ///
    /// Returns `[distance, closest_x, closest_y, closest_z]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the edge handle is invalid.
    #[wasm_bindgen(js_name = "pointToEdgeDistance")]
    pub fn point_to_edge_distance(
        &self,
        px: f64,
        py: f64,
        pz: f64,
        edge: u32,
    ) -> Result<Vec<f64>, JsError> {
        let edge_id = self.resolve_edge(edge)?;
        let result = brepkit_operations::distance::point_to_edge(
            &self.topo,
            Point3::new(px, py, pz),
            edge_id,
        )?;
        Ok(vec![
            result.distance,
            result.point_b.x(),
            result.point_b.y(),
            result.point_b.z(),
        ])
    }
}

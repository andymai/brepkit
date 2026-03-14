//! Primitive solid creation bindings.

#![allow(clippy::missing_errors_doc)]

use wasm_bindgen::prelude::*;

use brepkit_operations::transform::transform_solid;

use crate::error::{validate_finite, validate_positive};
use crate::handles::solid_id_to_u32;
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    /// Create a box solid with the given dimensions, centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if any dimension is non-positive or non-finite.
    #[wasm_bindgen(js_name = "makeBox")]
    pub fn make_box_solid(&mut self, dx: f64, dy: f64, dz: f64) -> Result<u32, JsError> {
        validate_positive(dx, "dx")?;
        validate_positive(dy, "dy")?;
        validate_positive(dz, "dz")?;
        let solid_id = brepkit_operations::primitives::make_box(&mut self.topo, dx, dy, dz)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cylinder solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius or height is non-positive.
    #[wasm_bindgen(js_name = "makeCylinder")]
    pub fn make_cylinder_solid(&mut self, radius: f64, height: f64) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        validate_positive(height, "height")?;
        let solid_id =
            brepkit_operations::primitives::make_cylinder(&mut self.topo, radius, height)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a sphere solid centered at the origin.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radius is non-positive or segments < 4.
    #[wasm_bindgen(js_name = "makeSphere")]
    pub fn make_sphere_solid(&mut self, radius: f64, segments: u32) -> Result<u32, JsError> {
        validate_positive(radius, "radius")?;
        let solid_id =
            brepkit_operations::primitives::make_sphere(&mut self.topo, radius, segments as usize)?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a cone or frustum solid centered at the origin, axis along +Z.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if height is non-positive or both radii are zero.
    #[wasm_bindgen(js_name = "makeCone")]
    pub fn make_cone_solid(
        &mut self,
        bottom_radius: f64,
        top_radius: f64,
        height: f64,
    ) -> Result<u32, JsError> {
        validate_finite(bottom_radius, "bottom_radius")?;
        validate_finite(top_radius, "top_radius")?;
        validate_positive(height, "height")?;
        let solid_id = brepkit_operations::primitives::make_cone(
            &mut self.topo,
            bottom_radius,
            top_radius,
            height,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create a torus solid centered at the origin in the XY plane.
    ///
    /// Returns a solid handle (`u32`).
    ///
    /// # Errors
    ///
    /// Returns an error if radii are non-positive or minor >= major.
    #[wasm_bindgen(js_name = "makeTorus")]
    pub fn make_torus_solid(
        &mut self,
        major_radius: f64,
        minor_radius: f64,
        segments: u32,
    ) -> Result<u32, JsError> {
        validate_positive(major_radius, "major_radius")?;
        validate_positive(minor_radius, "minor_radius")?;
        let solid_id = brepkit_operations::primitives::make_torus(
            &mut self.topo,
            major_radius,
            minor_radius,
            segments as usize,
        )?;
        Ok(solid_id_to_u32(solid_id))
    }

    /// Create an ellipsoid solid centered at the origin.
    ///
    /// Built by creating a unit sphere and scaling it by `(rx, ry, rz)`.
    ///
    /// # Errors
    ///
    /// Returns an error if any radius is non-positive.
    #[wasm_bindgen(js_name = "makeEllipsoid")]
    pub fn make_ellipsoid_solid(&mut self, rx: f64, ry: f64, rz: f64) -> Result<u32, JsError> {
        validate_positive(rx, "rx")?;
        validate_positive(ry, "ry")?;
        validate_positive(rz, "rz")?;
        // Create a unit sphere, then scale it non-uniformly.
        let solid_id = brepkit_operations::primitives::make_sphere(&mut self.topo, 1.0, 16)?;
        let mat = brepkit_math::mat::Mat4::scale(rx, ry, rz);
        transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(solid_id_to_u32(solid_id))
    }
}

//! Transform, copy, mirror, and pattern bindings.

#![allow(clippy::missing_errors_doc, clippy::too_many_arguments)]

use wasm_bindgen::prelude::*;

use brepkit_math::mat::Mat4;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::transform::transform_solid;

use crate::error::{WasmError, validate_finite, validate_positive};
use crate::handles::{compound_id_to_u32, solid_id_to_u32, wire_id_to_u32};
use crate::kernel::BrepKernel;

#[wasm_bindgen]
impl BrepKernel {
    /// Apply a 4×4 affine transform to a solid (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "transformSolid")]
    #[allow(clippy::needless_pass_by_value)] // wasm-bindgen requires owned Vec
    pub fn transform_solid_binding(&mut self, solid: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(())
    }

    /// Compose (multiply) two 4x4 transformation matrices.
    ///
    /// Returns the composed matrix as a flat 16-element array (row-major).
    /// This computes `a * b`, meaning `b` is applied first, then `a`.
    ///
    /// # Errors
    ///
    /// Returns an error if either matrix doesn't have 16 elements.
    #[wasm_bindgen(js_name = "composeTransforms")]
    #[allow(clippy::needless_pass_by_value, clippy::unused_self)]
    pub fn compose_transforms(
        &self,
        matrix_a: Vec<f64>,
        matrix_b: Vec<f64>,
    ) -> Result<Vec<f64>, JsError> {
        if matrix_a.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix A must have 16 elements, got {}", matrix_a.len()),
            }
            .into());
        }
        if matrix_b.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix B must have 16 elements, got {}", matrix_b.len()),
            }
            .into());
        }
        let rows_a = std::array::from_fn(|i| std::array::from_fn(|j| matrix_a[i * 4 + j]));
        let rows_b = std::array::from_fn(|i| std::array::from_fn(|j| matrix_b[i * 4 + j]));
        let result = Mat4(rows_a) * Mat4(rows_b);
        let mut out = Vec::with_capacity(16);
        for row in &result.0 {
            out.extend_from_slice(row);
        }
        Ok(out)
    }

    // ── Copy / Mirror / Pattern ───────────────────────────────────

    /// Deep copy a solid, returning a new independent solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid.
    #[wasm_bindgen(js_name = "copySolid")]
    pub fn copy_solid(&mut self, solid: u32) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let copy = brepkit_operations::copy::copy_solid(&mut self.topo, solid_id)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Deep copy a wire, returning a new independent wire handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid.
    #[wasm_bindgen(js_name = "copyWire")]
    pub fn copy_wire(&mut self, wire: u32) -> Result<u32, JsError> {
        let wire_id = self.resolve_wire(wire)?;
        let copy = brepkit_operations::copy::copy_wire(&mut self.topo, wire_id)?;
        Ok(wire_id_to_u32(copy))
    }

    /// Apply a 4×4 affine transform to a wire (in place).
    ///
    /// The `matrix` must contain exactly 16 values in row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error if the wire handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "transformWire")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn transform_wire(&mut self, wire: u32, matrix: Vec<f64>) -> Result<(), JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let wire_id = self.resolve_wire(wire)?;
        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);
        brepkit_operations::transform::transform_wire(&mut self.topo, wire_id, &mat)?;
        Ok(())
    }

    /// Copy a solid and apply a 4×4 row-major affine transform in one pass.
    ///
    /// Equivalent to `copySolid` + `transformSolid` but performs both in a
    /// single topology traversal, avoiding redundant NURBS clones.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid, the matrix doesn't
    /// have 16 elements, or the matrix is singular.
    #[wasm_bindgen(js_name = "copyAndTransformSolid")]
    #[allow(clippy::needless_pass_by_value)]
    pub fn copy_and_transform_solid(
        &mut self,
        solid: u32,
        matrix: Vec<f64>,
    ) -> Result<u32, JsError> {
        if matrix.len() != 16 {
            return Err(WasmError::InvalidInput {
                reason: format!(
                    "transform matrix must have 16 elements, got {}",
                    matrix.len()
                ),
            }
            .into());
        }

        if let Some(pos) = matrix.iter().position(|v| !v.is_finite()) {
            return Err(WasmError::InvalidInput {
                reason: format!("matrix element at index {pos} is not finite"),
            }
            .into());
        }

        let solid_id = self.resolve_solid(solid)?;

        let rows = std::array::from_fn(|i| std::array::from_fn(|j| matrix[i * 4 + j]));
        let mat = Mat4(rows);

        let copy =
            brepkit_operations::copy::copy_and_transform_solid(&mut self.topo, solid_id, &mat)?;
        Ok(solid_id_to_u32(copy))
    }

    /// Mirror a solid across a plane.
    ///
    /// Returns a new solid handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the solid handle is invalid or the normal is zero.
    #[wasm_bindgen(js_name = "mirror")]
    #[allow(clippy::too_many_arguments)]
    pub fn mirror_solid(
        &mut self,
        solid: u32,
        px: f64,
        py: f64,
        pz: f64,
        nx: f64,
        ny: f64,
        nz: f64,
    ) -> Result<u32, JsError> {
        validate_finite(px, "px")?;
        validate_finite(py, "py")?;
        validate_finite(pz, "pz")?;
        validate_finite(nx, "nx")?;
        validate_finite(ny, "ny")?;
        validate_finite(nz, "nz")?;
        let solid_id = self.resolve_solid(solid)?;
        let result = brepkit_operations::mirror::mirror(
            &mut self.topo,
            solid_id,
            Point3::new(px, py, pz),
            Vec3::new(nx, ny, nz),
        )?;
        Ok(solid_id_to_u32(result))
    }

    /// Create a linear pattern of a solid.
    ///
    /// Returns a compound handle containing all copies.
    ///
    /// # Errors
    ///
    /// Returns an error if inputs are invalid.
    #[wasm_bindgen(js_name = "linearPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn linear_pattern(
        &mut self,
        solid: u32,
        dx: f64,
        dy: f64,
        dz: f64,
        spacing: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        validate_finite(dx, "dx")?;
        validate_finite(dy, "dy")?;
        validate_finite(dz, "dz")?;
        validate_positive(spacing, "spacing")?;
        let solid_id = self.resolve_solid(solid)?;
        let compound = brepkit_operations::pattern::linear_pattern(
            &mut self.topo,
            solid_id,
            Vec3::new(dx, dy, dz),
            spacing,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    // ── Grid Pattern ──────────────────────────────────────────────

    /// Create a 2D grid pattern of a solid.
    ///
    /// Produces `count_x × count_y` copies arranged in a rectangular grid.
    #[wasm_bindgen(js_name = "gridPattern")]
    #[allow(clippy::too_many_arguments)]
    pub fn grid_pattern(
        &mut self,
        solid: u32,
        dir_x_x: f64,
        dir_x_y: f64,
        dir_x_z: f64,
        dir_y_x: f64,
        dir_y_y: f64,
        dir_y_z: f64,
        spacing_x: f64,
        spacing_y: f64,
        count_x: u32,
        count_y: u32,
    ) -> Result<u32, JsError> {
        validate_finite(dir_x_x, "dir_x_x")?;
        validate_finite(dir_x_y, "dir_x_y")?;
        validate_finite(dir_x_z, "dir_x_z")?;
        validate_finite(dir_y_x, "dir_y_x")?;
        validate_finite(dir_y_y, "dir_y_y")?;
        validate_finite(dir_y_z, "dir_y_z")?;
        validate_positive(spacing_x, "spacing_x")?;
        validate_positive(spacing_y, "spacing_y")?;
        let solid_id = self.resolve_solid(solid)?;
        let compound = brepkit_operations::pattern::grid_pattern(
            &mut self.topo,
            solid_id,
            Vec3::new(dir_x_x, dir_x_y, dir_x_z),
            Vec3::new(dir_y_x, dir_y_y, dir_y_z),
            spacing_x,
            spacing_y,
            count_x as usize,
            count_y as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    /// Create a circular pattern of a solid around an axis.
    ///
    /// Returns a compound handle.
    #[wasm_bindgen(js_name = "circularPattern")]
    pub fn circular_pattern(
        &mut self,
        solid: u32,
        ax: f64,
        ay: f64,
        az: f64,
        count: u32,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let axis = Vec3::new(ax, ay, az);
        let compound = brepkit_operations::pattern::circular_pattern(
            &mut self.topo,
            solid_id,
            axis,
            count as usize,
        )?;
        Ok(compound_id_to_u32(compound))
    }

    /// Merge coincident vertices in a solid.
    ///
    /// Returns the number of vertices merged.
    #[wasm_bindgen(js_name = "mergeCoincidentVertices")]
    pub fn merge_coincident_vertices(
        &mut self,
        solid: u32,
        tolerance: f64,
    ) -> Result<u32, JsError> {
        let solid_id = self.resolve_solid(solid)?;
        let count = brepkit_operations::heal::merge_coincident_vertices(
            &mut self.topo,
            solid_id,
            tolerance,
        )?;
        #[allow(clippy::cast_possible_truncation)]
        Ok(count as u32)
    }
}

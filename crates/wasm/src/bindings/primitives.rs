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
        let solid_id = brepkit_operations::primitives::make_box(self.topo_mut(), dx, dy, dz)?;
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
            brepkit_operations::primitives::make_cylinder(self.topo_mut(), radius, height)?;
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
        let solid_id = brepkit_operations::primitives::make_sphere(
            self.topo_mut(),
            radius,
            segments as usize,
        )?;
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
            self.topo_mut(),
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
            self.topo_mut(),
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
        let solid_id = brepkit_operations::primitives::make_sphere(self.topo_mut(), 1.0, 16)?;
        let mat = brepkit_math::mat::Mat4::scale(rx, ry, rz);
        transform_solid(self.topo_mut(), solid_id, &mat)?;
        Ok(solid_id_to_u32(solid_id))
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

    // ── make_box ────────────────────────────────────────────────────

    #[test]
    fn make_box_happy_path() {
        let mut k = BrepKernel::new();
        let r = k
            .execute_batch(r#"[{"op": "makeBox", "args": {"width": 1, "height": 2, "depth": 3}}]"#);
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_box_negative_width() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeBox", "args": {"width": -1, "height": 2, "depth": 3}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_box_zero_height() {
        let mut k = BrepKernel::new();
        let r = k
            .execute_batch(r#"[{"op": "makeBox", "args": {"width": 1, "height": 0, "depth": 3}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_box_volume() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 2, "height": 3, "depth": 4}},
                {"op": "volume", "args": {"solid": 0}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let vol = parsed[1]["ok"].as_f64().unwrap();
        assert!((vol - 24.0).abs() < 0.1, "expected ~24.0, got {vol}");
    }

    // ── make_cylinder ───────────────────────────────────────────────

    #[test]
    fn make_cylinder_happy_path() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "makeCylinder", "args": {"radius": 1, "height": 2}}]"#);
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_cylinder_negative_radius() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "makeCylinder", "args": {"radius": -1, "height": 2}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_cylinder_zero_height() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "makeCylinder", "args": {"radius": 1, "height": 0}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    // ── make_sphere ─────────────────────────────────────────────────

    #[test]
    fn make_sphere_happy_path() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "makeSphere", "args": {"radius": 1}}]"#);
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_sphere_negative_radius() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "makeSphere", "args": {"radius": -1}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    // ── make_cone ───────────────────────────────────────────────────

    #[test]
    fn make_cone_happy_path_full_cone() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeCone", "args": {"bottomRadius": 2, "topRadius": 0, "height": 3}}]"#,
        );
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_cone_happy_path_frustum() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeCone", "args": {"bottomRadius": 2, "topRadius": 1, "height": 3}}]"#,
        );
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_cone_both_radii_zero() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeCone", "args": {"bottomRadius": 0, "topRadius": 0, "height": 3}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_cone_negative_radius() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeCone", "args": {"bottomRadius": -1, "topRadius": 1, "height": 3}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_cone_zero_height() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeCone", "args": {"bottomRadius": 1, "topRadius": 0.5, "height": 0}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    // ── make_torus ──────────────────────────────────────────────────

    #[test]
    fn make_torus_happy_path() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeTorus", "args": {"majorRadius": 3, "minorRadius": 1}}]"#,
        );
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_torus_minor_equals_major() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeTorus", "args": {"majorRadius": 2, "minorRadius": 2}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_torus_minor_greater_than_major() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeTorus", "args": {"majorRadius": 1, "minorRadius": 2}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_torus_zero_major_radius() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[{"op": "makeTorus", "args": {"majorRadius": 0, "minorRadius": 1}}]"#,
        );
        assert!(batch_has_error(&r, 0));
    }

    // ── make_ellipsoid ──────────────────────────────────────────────

    #[test]
    fn make_ellipsoid_happy_path() {
        let mut k = BrepKernel::new();
        let r =
            k.execute_batch(r#"[{"op": "makeEllipsoid", "args": {"rx": 1, "ry": 2, "rz": 3}}]"#);
        assert!(batch_has_ok(&r, 0));
    }

    #[test]
    fn make_ellipsoid_negative_rx() {
        let mut k = BrepKernel::new();
        let r =
            k.execute_batch(r#"[{"op": "makeEllipsoid", "args": {"rx": -1, "ry": 2, "rz": 3}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    #[test]
    fn make_ellipsoid_zero_ry() {
        let mut k = BrepKernel::new();
        let r =
            k.execute_batch(r#"[{"op": "makeEllipsoid", "args": {"rx": 1, "ry": 0, "rz": 3}}]"#);
        assert!(batch_has_error(&r, 0));
    }

    // ── multiple primitives in one kernel ─────────────────────────────

    #[test]
    fn multiple_primitives_independent_handles() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 1, "depth": 1}},
                {"op": "makeCylinder", "args": {"radius": 0.5, "height": 2}},
                {"op": "makeSphere", "args": {"radius": 0.5}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        let h0 = parsed[0]["ok"].as_u64().unwrap();
        let h1 = parsed[1]["ok"].as_u64().unwrap();
        let h2 = parsed[2]["ok"].as_u64().unwrap();
        assert_ne!(h0, h1);
        assert_ne!(h1, h2);
        assert_ne!(h0, h2);
    }

    #[test]
    fn all_six_primitives_in_one_kernel() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(
            r#"[
                {"op": "makeBox", "args": {"width": 1, "height": 2, "depth": 3}},
                {"op": "makeCylinder", "args": {"radius": 1, "height": 2}},
                {"op": "makeSphere", "args": {"radius": 1}},
                {"op": "makeCone", "args": {"bottomRadius": 1, "topRadius": 0.5, "height": 2}},
                {"op": "makeTorus", "args": {"majorRadius": 3, "minorRadius": 1}},
                {"op": "makeEllipsoid", "args": {"rx": 1, "ry": 2, "rz": 3}}
            ]"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&r).unwrap();
        for i in 0..6 {
            assert!(
                parsed[i]["ok"].is_number(),
                "primitive {i} failed: {}",
                parsed[i]
            );
        }
    }

    #[test]
    fn invalid_handle_returns_error() {
        let mut k = BrepKernel::new();
        let r = k.execute_batch(r#"[{"op": "volume", "args": {"solid": 9999}}]"#);
        assert!(batch_has_error(&r, 0));
    }
}

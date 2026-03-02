//! JS-facing shape types via `wasm-bindgen`.

use wasm_bindgen::prelude::*;

/// A 3D point exposed to JavaScript.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct JsPoint3 {
    /// X coordinate.
    pub x: f64,
    /// Y coordinate.
    pub y: f64,
    /// Z coordinate.
    pub z: f64,
}

#[wasm_bindgen]
impl JsPoint3 {
    /// Create a new 3D point.
    #[wasm_bindgen(constructor)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }
}

/// A 3D vector exposed to JavaScript.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy)]
pub struct JsVec3 {
    /// X component.
    pub x: f64,
    /// Y component.
    pub y: f64,
    /// Z component.
    pub z: f64,
}

#[wasm_bindgen]
impl JsVec3 {
    /// Create a new 3D vector.
    #[wasm_bindgen(constructor)]
    #[must_use]
    #[allow(clippy::missing_const_for_fn)]
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Compute the length of this vector.
    #[must_use]
    pub fn length(&self) -> f64 {
        self.x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
            .sqrt()
    }
}

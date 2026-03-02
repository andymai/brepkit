//! JS-facing shape types via `wasm-bindgen`.

use brepkit_operations::tessellate::TriangleMesh;
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

/// A triangle mesh exposed to JavaScript.
///
/// Positions and normals are flattened to `[x, y, z, x, y, z, ...]` format
/// for efficient WASM transfer and direct use as GPU vertex buffers.
#[wasm_bindgen]
pub struct JsMesh {
    positions: Vec<f64>,
    normals: Vec<f64>,
    indices: Vec<u32>,
}

#[wasm_bindgen]
impl JsMesh {
    /// Flattened vertex positions as `[x, y, z, ...]`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn positions(&self) -> Vec<f64> {
        self.positions.clone()
    }

    /// Flattened per-vertex normals as `[nx, ny, nz, ...]`.
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn normals(&self) -> Vec<f64> {
        self.normals.clone()
    }

    /// Triangle indices (groups of 3).
    #[wasm_bindgen(getter)]
    #[must_use]
    pub fn indices(&self) -> Vec<u32> {
        self.indices.clone()
    }

    /// Number of vertices in the mesh.
    #[wasm_bindgen(getter, js_name = "vertexCount")]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn vertex_count(&self) -> u32 {
        (self.positions.len() / 3) as u32
    }

    /// Number of triangles in the mesh.
    #[wasm_bindgen(getter, js_name = "triangleCount")]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn triangle_count(&self) -> u32 {
        (self.indices.len() / 3) as u32
    }
}

impl From<TriangleMesh> for JsMesh {
    fn from(mesh: TriangleMesh) -> Self {
        let positions = mesh
            .positions
            .iter()
            .flat_map(|p| [p.x(), p.y(), p.z()])
            .collect();

        let normals = mesh
            .normals
            .iter()
            .flat_map(|n| [n.x(), n.y(), n.z()])
            .collect();

        Self {
            positions,
            normals,
            indices: mesh.indices,
        }
    }
}

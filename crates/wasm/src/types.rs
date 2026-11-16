//! Typed result structs for structured WASM returns.
//!
//! Types annotated with `Tsify` generate the TypeScript definitions that
//! describe these shapes. They cross the boundary as JSON strings, not over
//! the wasm ABI, so they deliberately carry no `into_wasm_abi` impl.

use tsify::Tsify;

/// Typed result for `tessellateSolidGrouped`.
#[derive(serde::Serialize, Tsify)]
#[serde(rename_all = "camelCase")]
pub struct GroupedMeshResult {
    pub positions: Vec<f64>,
    pub normals: Vec<f64>,
    pub indices: Vec<u32>,
    pub face_offsets: Vec<u32>,
}

/// Typed result for the `import*Parts` mesh imports.
#[derive(Debug, serde::Serialize, Tsify)]
pub struct MeshImportResult {
    /// A solid handle for each closed body, cavities included.
    pub solids: Vec<u32>,
    /// A shell handle for each body that does not bound a solid: a body that
    /// does not close, and an inward closed body that crosses another body.
    pub shells: Vec<u32>,
}

/// Typed result for `tessellateSolidUV`.
#[derive(serde::Serialize, Tsify)]
pub struct UvMeshResult {
    pub positions: Vec<f64>,
    pub normals: Vec<f64>,
    pub indices: Vec<u32>,
    pub uvs: Vec<f64>,
}

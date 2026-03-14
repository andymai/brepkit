//! Typed result structs for structured WASM returns.

/// Typed result for `tessellateSolidGrouped` — avoids `json!()` intermediate `Value`.
#[derive(serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroupedMeshResult {
    pub positions: Vec<f64>,
    pub normals: Vec<f64>,
    pub indices: Vec<u32>,
    pub face_offsets: Vec<u32>,
}

/// Typed result for `tessellateSolidUV` — avoids `json!()` intermediate `Value`.
#[derive(serde::Serialize)]
pub struct UvMeshResult {
    pub positions: Vec<f64>,
    pub normals: Vec<f64>,
    pub indices: Vec<u32>,
    pub uvs: Vec<f64>,
}

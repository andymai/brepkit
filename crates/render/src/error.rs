//! Error type for the offscreen renderer.

/// Errors that can occur while rendering a solid offscreen.
#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    /// No wgpu adapter could be obtained (neither a real GPU nor a software
    /// fallback). The crate is still usable on machines that do provide one.
    #[error("no wgpu adapter available (tried real GPU then software fallback): {0}")]
    NoAdapter(String),

    /// The adapter could not provide a device/queue matching the request.
    #[error("failed to request wgpu device: {0}")]
    DeviceRequest(String),

    /// A GPU buffer could not be mapped for readback.
    #[error("failed to map GPU buffer for readback: {0}")]
    BufferMap(String),

    /// Polling the device for readback completion failed.
    #[error("failed to poll wgpu device: {0}")]
    Poll(String),

    /// The requested render dimensions were invalid (zero width or height).
    #[error("invalid render size: width and height must be non-zero, got {width}x{height}")]
    InvalidSize {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
    },

    /// The requested render dimensions exceed the adapter's maximum 2D texture
    /// size, so a render would fail GPU validation.
    #[error(
        "render size {width}x{height} exceeds the device limit of {max}x{max} (max 2D texture dimension)"
    )]
    SizeTooLarge {
        /// Requested width in pixels.
        width: u32,
        /// Requested height in pixels.
        height: u32,
        /// The adapter's `max_texture_dimension_2d`.
        max: u32,
    },

    /// The tessellation produced a mesh that violates a renderer invariant
    /// (e.g. an index buffer length not divisible by 3, an out-of-range vertex
    /// index, or grouped face offsets that do not cover every triangle).
    #[error("malformed tessellation mesh: {0}")]
    MeshData(String),

    /// The windowing event loop could not be created or run (viewer only).
    #[error("windowing event loop error: {0}")]
    EventLoop(String),

    /// A window surface could not be created or configured (viewer only).
    #[error("failed to create or configure window surface: {0}")]
    SurfaceConfig(String),

    /// Tessellation of the input solid failed.
    #[error(transparent)]
    Operations(#[from] brepkit_operations::OperationsError),

    /// Topology traversal of the input solid failed.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),
}

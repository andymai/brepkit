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

    /// Tessellation of the input solid failed.
    #[error(transparent)]
    Operations(#[from] brepkit_operations::OperationsError),

    /// Topology traversal of the input solid failed.
    #[error(transparent)]
    Topology(#[from] brepkit_topology::TopologyError),
}

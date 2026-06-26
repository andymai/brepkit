//! Offscreen GPU renderer for brepkit B-Rep solids.
//!
//! [`render_solid_offscreen`] tessellates a [`Solid`](brepkit_topology::solid)
//! and rasterizes it to images with [`wgpu`], entirely off-screen (render to
//! texture + read back). No window or display is required, so it runs in
//! headless CI given any wgpu adapter — a real GPU or a software fallback
//! (e.g. Mesa lavapipe via the Vulkan backend).
//!
//! # Outputs
//!
//! Each render produces a shaded color image plus a parallel face-id buffer:
//! every pixel carries the [`FaceId`](brepkit_topology::face::FaceId) of the
//! face drawn there (`0` = background). Use [`RenderOutput::face_id_at`] for
//! pixel picking.
//!
//! # Precision (render-relative-to-center)
//!
//! Geometry is tessellated in f64. To keep f32 GPU coordinates accurate even
//! for models far from the origin, vertex positions are uploaded relative to
//! the model's AABB center; the f64 center is folded into the camera's view
//! matrix on the CPU.
//!
//! # Example
//!
//! ```no_run
//! use brepkit_render::{Camera, RenderOpts, render_solid_offscreen};
//! use brepkit_math::vec::{Point3, Vec3};
//! use brepkit_topology::Topology;
//!
//! let mut topo = Topology::new();
//! let solid = brepkit_operations::primitives::make_box(&mut topo, 20.0, 20.0, 20.0)?;
//! let cam = Camera {
//!     eye: Point3::new(60.0, 50.0, 70.0),
//!     target: Point3::new(10.0, 10.0, 10.0),
//!     up: Vec3::new(0.0, 0.0, 1.0),
//!     fov_y: 45.0_f64.to_radians(),
//!     aspect: 1.0,
//!     near: 0.1,
//!     far: 1000.0,
//! };
//! let opts = RenderOpts::new(512, 512);
//! let out = render_solid_offscreen(&topo, solid, &cam, &opts)?;
//! out.color.save("box.png")?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod camera;
mod error;
mod mesh;
mod pipeline;

pub use camera::Camera;
pub use error::RenderError;
pub use pipeline::probe_adapter;

use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

/// Default linear chord tolerance used when [`RenderOpts`] does not override it.
pub const DEFAULT_DEFLECTION: f64 = 0.05;

/// Options controlling an offscreen render.
#[derive(Debug, Clone, Copy)]
pub struct RenderOpts {
    /// Output width in pixels (must be non-zero).
    pub width: u32,
    /// Output height in pixels (must be non-zero).
    pub height: u32,
    /// Draw topological edges as crisp dark lines over the shaded mesh.
    pub edges: bool,
    /// Background clear color as linear RGBA in `[0, 1]`.
    pub background: [f32; 4],
    /// Ambient light fraction in `[0, 1]` added to the Lambert headlight term.
    pub ambient: f32,
    /// Linear chord tolerance for tessellation (smaller = finer mesh).
    pub deflection: f64,
}

impl RenderOpts {
    /// Create options for a `width` x `height` render with sensible defaults
    /// (edges on, light-gray background, modest ambient term).
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            edges: true,
            background: [0.11, 0.12, 0.14, 1.0],
            ambient: 0.25,
            deflection: DEFAULT_DEFLECTION,
        }
    }
}

/// The result of an offscreen render.
pub struct RenderOutput {
    /// Shaded color image (sRGB, RGBA8).
    pub color: image::RgbaImage,
    /// Per-pixel face id, row-major (`width * height` entries). `0` is
    /// background; otherwise the value is `FaceId.index() + 1`.
    pub id_buffer: Vec<u32>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
}

impl RenderOutput {
    /// Face id at pixel `(x, y)`, or `None` for background or out-of-bounds.
    ///
    /// The returned value is `FaceId.index() + 1` (the same encoding stored in
    /// [`RenderOutput::id_buffer`]); `0`/background maps to `None`.
    #[must_use]
    pub fn face_id_at(&self, x: u32, y: u32) -> Option<u32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) as usize;
        match self.id_buffer.get(idx).copied() {
            Some(0) | None => None,
            Some(v) => Some(v),
        }
    }
}

/// Render a solid offscreen to a shaded color image and a face-id buffer.
///
/// Tessellates `solid`, sets up a wgpu device (trying a real GPU first, then a
/// software fallback), rasterizes the mesh (and edges, if
/// [`RenderOpts::edges`]) into off-screen targets, and reads them back.
///
/// # Errors
///
/// - [`RenderError::InvalidSize`] if `opts.width` or `opts.height` is zero.
/// - [`RenderError::NoAdapter`] if no wgpu adapter (GPU or software) exists.
/// - [`RenderError::DeviceRequest`] / [`RenderError::BufferMap`] /
///   [`RenderError::Poll`] on GPU setup or readback failure.
/// - [`RenderError::Operations`] if tessellating the solid fails.
pub fn render_solid_offscreen(
    topo: &Topology,
    solid: SolidId,
    cam: &Camera,
    opts: &RenderOpts,
) -> Result<RenderOutput, RenderError> {
    if opts.width == 0 || opts.height == 0 {
        return Err(RenderError::InvalidSize {
            width: opts.width,
            height: opts.height,
        });
    }
    let render_mesh = mesh::RenderMesh::build(topo, solid, opts.deflection)?;
    pipeline::render(&render_mesh, cam, opts)
}

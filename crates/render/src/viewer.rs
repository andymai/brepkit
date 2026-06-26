//! Interactive viewer window (winit surface + orbit camera + click-to-pick).
//!
//! [`view_solid`] opens a window that renders a solid with the same Lambert
//! mesh + crisp-edge pipeline as the offscreen path ([`crate::pipeline`]), but
//! to a swapchain surface instead of a texture. It adds:
//!
//! - **Orbit camera:** left-drag orbits (azimuth/elevation), the scroll wheel
//!   dollies in/out, and right-drag (or shift + left-drag) pans the target.
//! - **Click-to-pick:** a left click that does not drag reads the per-pixel
//!   face-id target under the cursor and highlights the picked
//!   [`FaceId`](brepkit_topology::face::FaceId) by tinting it in the shader.
//!
//! The face-id target is the same `R32Uint` buffer the offscreen renderer
//! produces, so a click yields the kernel `FaceId` directly — the CAD payoff.
//!
//! # Running
//!
//! This needs a display server and the `window` feature:
//!
//! ```text
//! cargo run -p brepkit-render --example viewer --features window
//! ```
//!
//! It cannot run headlessly (no surface without a display); the offscreen path
//! ([`crate::render_solid_offscreen`]) covers headless rendering.
//!
//! # Precision
//!
//! Like the offscreen path, geometry is uploaded relative to the model's AABB
//! center (RTC) and the f64 center is folded into the view matrix, so models
//! far from the origin stay crisp.

use std::sync::Arc;

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;
use winit::window::{Window, WindowId};

use crate::DEFAULT_DEFLECTION;
use crate::camera::Camera;
use crate::error::RenderError;
use crate::mesh::RenderMesh;
use crate::pipeline::{self, DEPTH_FORMAT, GeometryBuffers, GlobalsBinding, ID_FORMAT, Pipelines};

/// Options controlling an interactive viewer window.
#[derive(Debug, Clone)]
pub struct ViewOpts {
    /// Window title.
    pub title: String,
    /// Initial window width in physical pixels.
    pub width: u32,
    /// Initial window height in physical pixels.
    pub height: u32,
    /// Background clear color as linear RGBA in `[0, 1]`.
    pub background: [f32; 4],
    /// Ambient light fraction in `[0, 1]`.
    pub ambient: f32,
    /// Draw topological edges as crisp dark lines.
    pub edges: bool,
    /// Linear chord tolerance for tessellation (smaller = finer mesh).
    pub deflection: f64,
}

impl Default for ViewOpts {
    fn default() -> Self {
        Self {
            title: "brepkit viewer".to_string(),
            width: 1024,
            height: 768,
            background: [0.11, 0.12, 0.14, 1.0],
            ambient: 0.25,
            edges: true,
            deflection: DEFAULT_DEFLECTION,
        }
    }
}

impl ViewOpts {
    /// Default options with a custom title.
    #[must_use]
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            ..Self::default()
        }
    }
}

/// Open an interactive window showing `solid`, blocking until it is closed.
///
/// Left-drag orbits, scroll zooms, right-drag (or shift + left-drag) pans, and a
/// left click highlights the picked face. Tessellation happens once up front;
/// only the camera and selection change per frame.
///
/// # Errors
///
/// - [`RenderError::Operations`] / [`RenderError::Topology`] if the solid
///   cannot be tessellated.
/// - [`RenderError::EventLoop`] if the windowing event loop fails to start.
/// - [`RenderError::NoAdapter`] / [`RenderError::DeviceRequest`] /
///   [`RenderError::SurfaceConfig`] on GPU/surface setup failure (surfaced from
///   inside the loop).
pub fn view_solid(topo: &Topology, solid: SolidId, opts: &ViewOpts) -> Result<(), RenderError> {
    let mesh = RenderMesh::build(topo, solid, opts.deflection)?;

    let event_loop = EventLoop::new().map_err(|e| RenderError::EventLoop(e.to_string()))?;
    // Wait for events rather than busy-looping; we explicitly request redraws on
    // interaction and resize.
    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = ViewerApp::new(mesh, opts.clone());
    event_loop
        .run_app(&mut app)
        .map_err(|e| RenderError::EventLoop(e.to_string()))?;
    app.into_result()
}

/// An orbit camera parameterized by spherical coordinates around a target.
#[derive(Debug, Clone, Copy)]
struct OrbitCamera {
    target: Point3,
    /// Horizontal angle (radians) about the world +Z axis.
    azimuth: f64,
    /// Vertical angle (radians) from the XY plane, clamped away from the poles.
    elevation: f64,
    /// Eye distance from the target.
    distance: f64,
    /// Model bounding-sphere radius; sets the scale for the clip planes so they
    /// track `distance` as the user zooms (see [`OrbitCamera::clip_planes`]).
    radius: f64,
    /// Vertical field of view (radians).
    fov_y: f64,
}

/// Keep elevation a hair below the poles so the up vector never degenerates.
const ELEVATION_LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 0.01;

impl OrbitCamera {
    /// Frame a model of bounding-sphere `radius` centered at `target` from an
    /// isometric-ish direction, matching the offscreen test's framing.
    fn framing(target: Point3, radius: f64) -> Self {
        let fov_y = 40.0_f64.to_radians();
        let radius = radius.max(1e-6);
        let distance = radius / (fov_y * 0.5).sin() * 2.0;
        Self {
            target,
            azimuth: 45.0_f64.to_radians(),
            elevation: 30.0_f64.to_radians(),
            distance,
            radius,
            fov_y,
        }
    }

    /// Unit direction from the target toward the eye.
    fn eye_dir(&self) -> Vec3 {
        let ce = self.elevation.cos();
        Vec3::new(
            ce * self.azimuth.cos(),
            ce * self.azimuth.sin(),
            self.elevation.sin(),
        )
    }

    /// Near/far clip planes for the current eye distance.
    ///
    /// Derived from `distance` (not cached) so the model stays inside the
    /// frustum across the whole zoom range: the model spans roughly
    /// `[distance - radius, distance + radius]` from the eye, so near sits just
    /// inside the near surface and far just beyond the far surface. Both are
    /// kept strictly positive with `near < far`.
    fn clip_planes(&self) -> (f64, f64) {
        let near = (self.distance - self.radius).max(self.distance * 0.01);
        let near = near.max(1e-4);
        let far = (self.distance + self.radius * 4.0).max(near * 10.0);
        (near, far)
    }

    /// Build the f64 [`Camera`] for the current orbit state at `aspect`.
    fn camera(&self, aspect: f64) -> Camera {
        let eye = self.target + self.eye_dir() * self.distance;
        let (near, far) = self.clip_planes();
        Camera {
            eye,
            target: self.target,
            up: Vec3::new(0.0, 0.0, 1.0),
            fov_y: self.fov_y,
            aspect,
            near,
            far,
        }
    }

    /// Orbit by mouse deltas (pixels): horizontal drag changes azimuth,
    /// vertical drag changes elevation (clamped away from the poles).
    fn orbit(&mut self, dx: f64, dy: f64) {
        const SPEED: f64 = 0.005;
        self.azimuth -= dx * SPEED;
        self.elevation = (self.elevation + dy * SPEED).clamp(-ELEVATION_LIMIT, ELEVATION_LIMIT);
    }

    /// Dolly toward/away from the target by a scroll amount (multiplicative so
    /// zoom feels uniform regardless of current distance).
    ///
    /// The clip planes are recomputed from `distance` in [`OrbitCamera::camera`],
    /// so zooming never leaves the model behind a stale near/far plane. Distance
    /// is floored to a small fraction of the model radius so it can't collapse
    /// to zero.
    fn dolly(&mut self, amount: f64) {
        let factor = (1.0 - amount * 0.1).clamp(0.2, 5.0);
        self.distance = (self.distance * factor).max(self.radius * 0.05 + 1e-6);
    }

    /// Pan the target in the camera's view plane by mouse deltas (pixels),
    /// scaled by distance so panning tracks the cursor at any zoom.
    fn pan(&mut self, dx: f64, dy: f64) {
        // eye_dir points target -> eye, so forward (eye -> target) is its negation.
        let forward = -self.eye_dir();
        let up = Vec3::new(0.0, 0.0, 1.0);
        let right = forward
            .cross(up)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let cam_up = right.cross(forward);
        let scale = self.distance * 0.0015;
        // Drag right -> scene moves right -> target moves left; drag down ->
        // target moves up. Hence the sign choices below.
        self.target = self.target + right * (-dx * scale) + cam_up * (dy * scale);
    }
}

/// What a left-drag currently does, decided at press time by the modifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DragMode {
    None,
    Orbit,
    Pan,
}

/// Size-dependent render targets: the depth buffer, the `R32Uint` id target
/// (kept as a texture so it can be copied back for picking), and a scratch
/// color target used by the on-demand id render during a pick.
struct Targets {
    depth_view: wgpu::TextureView,
    id_texture: wgpu::Texture,
    id_view: wgpu::TextureView,
    pick_color_view: wgpu::TextureView,
}

impl Targets {
    fn new(device: &wgpu::Device, format: wgpu::TextureFormat, width: u32, height: u32) -> Self {
        let extent = wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        };
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let id_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer id target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: ID_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let pick_color = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewer pick scratch color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        Self {
            depth_view: depth_tex.create_view(&wgpu::TextureViewDescriptor::default()),
            id_view: id_texture.create_view(&wgpu::TextureViewDescriptor::default()),
            id_texture,
            pick_color_view: pick_color.create_view(&wgpu::TextureViewDescriptor::default()),
        }
    }
}

/// GPU + window state, created lazily in `resumed` once a window exists.
struct GpuState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipelines: Pipelines,
    globals: GlobalsBinding,
    geometry: GeometryBuffers,
    targets: Targets,
}

impl GpuState {
    /// Reconfigure the surface and rebuild size-dependent targets.
    ///
    /// The size is clamped to the device's max 2D texture dimension so an
    /// oversized window never trips GPU validation on strict drivers.
    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        let max = self.device.limits().max_texture_dimension_2d;
        self.config.width = size.width.min(max);
        self.config.height = size.height.min(max);
        self.surface.configure(&self.device, &self.config);
        self.targets = Targets::new(
            &self.device,
            self.config.format,
            self.config.width,
            self.config.height,
        );
    }

    /// Aspect ratio (width / height) of the current surface.
    fn aspect(&self) -> f64 {
        aspect_of(self.config.width, self.config.height)
    }
}

/// Aspect ratio (width / height), guarding against a zero dimension.
fn aspect_of(width: u32, height: u32) -> f64 {
    f64::from(width.max(1)) / f64::from(height.max(1))
}

/// The winit application: owns the prepared geometry, options, orbit state, and
/// (once resumed) the GPU/window state. Any setup error is stashed in `error`
/// and the loop exits so `view_solid` can surface it.
struct ViewerApp {
    mesh: RenderMesh,
    opts: ViewOpts,
    orbit: OrbitCamera,
    gpu: Option<GpuState>,
    error: Option<RenderError>,

    // Input state.
    modifiers: ModifiersState,
    cursor: PhysicalPosition<f64>,
    drag: DragMode,
    right_drag: bool,
    /// Cursor position when the left button went down, to distinguish a click
    /// (pick) from a drag (orbit/pan).
    press_pos: Option<PhysicalPosition<f64>>,
    moved_while_pressed: bool,
    /// Encoded id (`FaceId.index() + 1`) of the highlighted face, or 0 for none.
    selected_id: u32,
}

/// Pixel-distance threshold below which a press+release counts as a click, not
/// a drag.
const CLICK_SLOP: f64 = 4.0;

impl ViewerApp {
    fn new(mesh: RenderMesh, opts: ViewOpts) -> Self {
        // Frame the model from its tessellated AABB (the mesh positions are in
        // world space; center is the RTC origin).
        let (min, max) = mesh_world_aabb(&mesh);
        let center = Point3::new(
            (min.x() + max.x()) * 0.5,
            (min.y() + max.y()) * 0.5,
            (min.z() + max.z()) * 0.5,
        );
        let radius = ((max.x() - min.x()).powi(2)
            + (max.y() - min.y()).powi(2)
            + (max.z() - min.z()).powi(2))
        .sqrt()
            * 0.5;
        let orbit = OrbitCamera::framing(center, radius);

        Self {
            mesh,
            opts,
            orbit,
            gpu: None,
            error: None,
            modifiers: ModifiersState::empty(),
            cursor: PhysicalPosition::new(0.0, 0.0),
            drag: DragMode::None,
            right_drag: false,
            press_pos: None,
            moved_while_pressed: false,
            selected_id: 0,
        }
    }

    /// Consume the app and return any deferred setup error.
    fn into_result(self) -> Result<(), RenderError> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Ask the window to redraw, if it exists yet.
    fn request_redraw(&self) {
        if let Some(gpu) = self.gpu.as_ref() {
            gpu.window.request_redraw();
        }
    }

    /// Record a setup error and ask the loop to exit.
    fn fail(&mut self, event_loop: &ActiveEventLoop, err: RenderError) {
        if self.error.is_none() {
            self.error = Some(err);
        }
        event_loop.exit();
    }

    /// Build the GPU state for a freshly created window.
    fn init_gpu(&self, window: Arc<Window>) -> Result<GpuState, RenderError> {
        let instance = wgpu::Instance::default();
        // An Arc<Window> is 'static and implements winit's window/display handle
        // traits, which is exactly what wgpu 29's create_surface accepts.
        let surface = instance
            .create_surface(window.clone())
            .map_err(|e| RenderError::SurfaceConfig(e.to_string()))?;

        let ctx = pipeline::GpuContext::with_instance(instance, Some(&surface))?;

        // Clamp the surface/target size to the device's max 2D texture
        // dimension so an oversized window never trips GPU validation.
        let max = ctx.device.limits().max_texture_dimension_2d;
        let size = window.inner_size();
        let width = size.width.clamp(1, max);
        let height = size.height.clamp(1, max);

        let caps = surface.get_capabilities(&ctx.adapter);
        let format = choose_surface_format(&caps);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            // Fifo (vsync) is guaranteed supported by the WebGPU spec; prefer it
            // explicitly rather than trusting the first advertised mode.
            present_mode: caps
                .present_modes
                .iter()
                .copied()
                .find(|m| *m == wgpu::PresentMode::Fifo)
                .or_else(|| caps.present_modes.first().copied())
                .unwrap_or(wgpu::PresentMode::Fifo),
            desired_maximum_frame_latency: 2,
            alpha_mode: caps
                .alpha_modes
                .first()
                .copied()
                .unwrap_or(wgpu::CompositeAlphaMode::Auto),
            view_formats: vec![],
        };
        surface.configure(&ctx.device, &config);

        let cam = self.orbit.camera(aspect_of(width, height));
        let globals = pipeline::build_globals(&cam, self.mesh.center, self.opts.ambient);
        let globals = GlobalsBinding::new(&ctx.device, &globals);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("viewer pipeline layout"),
                bind_group_layouts: &[Some(&globals.layout)],
                immediate_size: 0,
            });
        let with_edges = self.opts.edges && !self.mesh.edge_vertices.is_empty();
        let pipelines = Pipelines::new(&ctx.device, &pipeline_layout, format, with_edges);
        let geometry = GeometryBuffers::new(&ctx.device, &self.mesh);
        let targets = Targets::new(&ctx.device, format, width, height);

        Ok(GpuState {
            window,
            surface,
            device: ctx.device,
            queue: ctx.queue,
            config,
            pipelines,
            globals,
            geometry,
            targets,
        })
    }

    /// Upload the current camera + selection and draw one frame to the surface.
    fn redraw(&mut self) {
        let Some(gpu) = self.gpu.as_mut() else {
            return;
        };

        let frame = match gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(f)
            | wgpu::CurrentSurfaceTexture::Suboptimal(f) => f,
            // Surface needs reconfiguring; do it and skip this frame (a fresh
            // redraw is requested below).
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let size = PhysicalSize::new(gpu.config.width, gpu.config.height);
                gpu.resize(size);
                gpu.window.request_redraw();
                return;
            }
            // Transient: skip and try again next frame.
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return,
        };

        let cam = self.orbit.camera(gpu.aspect());
        let mut globals = pipeline::build_globals(&cam, self.mesh.center, self.opts.ambient);
        globals.selected_id = self.selected_id;
        gpu.globals.upload(&gpu.queue, &globals);

        let color_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("viewer encoder"),
            });
        pipeline::encode_scene(
            &mut encoder,
            &gpu.pipelines,
            &gpu.globals,
            &gpu.geometry,
            &pipeline::PassTargets {
                color: &color_view,
                id: &gpu.targets.id_view,
                depth: &gpu.targets.depth_view,
                background: self.opts.background,
            },
        );
        gpu.queue.submit(Some(encoder.finish()));
        gpu.window.pre_present_notify();
        frame.present();
    }

    /// Render the id buffer for the current camera and read the face id under
    /// the cursor, updating the highlight. Returns whether the selection
    /// changed (so the caller can request a redraw).
    ///
    /// The id pass is rendered fresh here rather than reusing the last frame's
    /// id target, so a pick is correct even immediately after a resize (before
    /// the next on-screen redraw).
    fn pick(&mut self) -> bool {
        let Some(gpu) = self.gpu.as_ref() else {
            return false;
        };
        if gpu.config.width == 0 || gpu.config.height == 0 {
            return false;
        }
        // Reject positions outside the viewport, then floor to a pixel index and
        // clamp to the last in-bounds pixel so an in-bounds cursor (which can sit
        // at exactly `width`/`height` at the far edge) never reads out of range.
        let (w, h) = (gpu.config.width, gpu.config.height);
        let cx = self.cursor.x;
        let cy = self.cursor.y;
        if cx < 0.0 || cy < 0.0 || cx >= f64::from(w) || cy >= f64::from(h) {
            return false;
        }
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let px = (cx.floor() as u32).min(w - 1);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let py = (cy.floor() as u32).min(h - 1);

        // Re-render the id target for the current camera (selection irrelevant
        // to face ids), then read the one pixel.
        let cam = self.orbit.camera(gpu.aspect());
        let globals = pipeline::build_globals(&cam, self.mesh.center, self.opts.ambient);
        gpu.globals.upload(&gpu.queue, &globals);

        let picked = match render_and_read_id(gpu, &self.opts.background, px, py) {
            Ok(id) => id,
            Err(e) => {
                // Non-fatal: a failed id readback (e.g. device lost) must not
                // crash the viewer, but surface it so it isn't a silent no-op.
                log::warn!("brepkit-render: face pick readback failed: {e}");
                return false;
            }
        };
        // Clicking the same face again clears the highlight.
        let next = if picked == self.selected_id {
            0
        } else {
            picked
        };
        if next == self.selected_id {
            return false;
        }
        self.selected_id = next;
        true
    }
}

impl ApplicationHandler for ViewerApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title(self.opts.title.clone())
            .with_inner_size(PhysicalSize::new(self.opts.width, self.opts.height));
        let window = match event_loop.create_window(attrs) {
            Ok(w) => Arc::new(w),
            Err(e) => {
                self.fail(event_loop, RenderError::EventLoop(e.to_string()));
                return;
            }
        };
        match self.init_gpu(window) {
            Ok(gpu) => {
                gpu.window.request_redraw();
                self.gpu = Some(gpu);
            }
            Err(e) => self.fail(event_loop, e),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),

            WindowEvent::Resized(size) => {
                if let Some(gpu) = self.gpu.as_mut() {
                    gpu.resize(size);
                    gpu.window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
            }

            WindowEvent::MouseInput { state, button, .. } => match (button, state) {
                (MouseButton::Left, ElementState::Pressed) => {
                    self.press_pos = Some(self.cursor);
                    self.moved_while_pressed = false;
                    self.drag = if self.modifiers.shift_key() {
                        DragMode::Pan
                    } else {
                        DragMode::Orbit
                    };
                }
                (MouseButton::Left, ElementState::Released) => {
                    // A press that never left the click slop (neither at release
                    // nor at any point during the press) is a pick; anything that
                    // dragged past the slop already moved the camera.
                    let near_press = self.press_pos.is_some_and(|p| {
                        (p.x - self.cursor.x).abs() <= CLICK_SLOP
                            && (p.y - self.cursor.y).abs() <= CLICK_SLOP
                    });
                    let was_click = near_press && !self.moved_while_pressed;
                    self.drag = DragMode::None;
                    self.press_pos = None;
                    if was_click && self.pick() {
                        self.request_redraw();
                    }
                }
                (MouseButton::Right, ElementState::Pressed) => self.right_drag = true,
                (MouseButton::Right, ElementState::Released) => self.right_drag = false,
                _ => {}
            },

            WindowEvent::CursorMoved { position, .. } => {
                let dx = position.x - self.cursor.x;
                let dy = position.y - self.cursor.y;
                self.cursor = position;

                // Right-drag always pans; otherwise the left-drag mode decides.
                let changed = if self.right_drag {
                    self.orbit.pan(dx, dy);
                    true
                } else {
                    match self.drag {
                        DragMode::Orbit => {
                            self.orbit.orbit(dx, dy);
                            true
                        }
                        DragMode::Pan => {
                            self.orbit.pan(dx, dy);
                            true
                        }
                        DragMode::None => false,
                    }
                };
                if changed {
                    // Only count as a drag (suppressing a pick) once the cursor
                    // has moved beyond the click slop from where it was pressed;
                    // sub-slop jitter must still register as a click.
                    self.moved_while_pressed |= self.press_pos.is_some_and(|p| {
                        (p.x - self.cursor.x).abs() > CLICK_SLOP
                            || (p.y - self.cursor.y).abs() > CLICK_SLOP
                    });
                    self.request_redraw();
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => f64::from(y),
                    MouseScrollDelta::PixelDelta(p) => p.y / 50.0,
                };
                self.orbit.dolly(amount);
                self.request_redraw();
            }

            WindowEvent::RedrawRequested => self.redraw(),

            _ => {}
        }
    }
}

/// Prefer an sRGB surface format (so shading matches the offscreen
/// `Rgba8UnormSrgb` path); fall back to the surface's preferred format.
fn choose_surface_format(caps: &wgpu::SurfaceCapabilities) -> wgpu::TextureFormat {
    caps.formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .or_else(|| caps.formats.first().copied())
        .unwrap_or(wgpu::TextureFormat::Bgra8UnormSrgb)
}

/// Render the id target for the current (already-uploaded) globals, then copy
/// the single row containing `(px, py)` back and decode that pixel.
///
/// The id target is `R32Uint`; the value is `FaceId.index() + 1` (or 0 for
/// background). Rendering the id pass here makes the pick independent of the
/// last on-screen frame. The scratch color target absorbs the color writes the
/// shared pass also produces (only the id is read back).
fn render_and_read_id(
    gpu: &GpuState,
    background: &[f32; 4],
    px: u32,
    py: u32,
) -> Result<u32, RenderError> {
    let padded_bpr = pipeline::padded_bytes_per_row(gpu.config.width, 4);

    // Read back only the one row that holds the picked pixel.
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("id pick readback"),
        size: u64::from(padded_bpr),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("id pick encoder"),
        });
    pipeline::encode_scene(
        &mut encoder,
        &gpu.pipelines,
        &gpu.globals,
        &gpu.geometry,
        &pipeline::PassTargets {
            color: &gpu.targets.pick_color_view,
            id: &gpu.targets.id_view,
            depth: &gpu.targets.depth_view,
            background: *background,
        },
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &gpu.targets.id_texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 0, y: py, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bpr),
                rows_per_image: Some(1),
            },
        },
        wgpu::Extent3d {
            width: gpu.config.width,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));

    let bytes = pipeline::map_and_read(&gpu.device, &readback)?;
    let off = (px * 4) as usize;
    let id = bytes
        .get(off..off + 4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .unwrap_or(0);
    Ok(id)
}

/// Axis-aligned bounding box of the mesh's world-space positions, recovered by
/// adding the RTC center back to the uploaded center-relative vertices.
fn mesh_world_aabb(mesh: &RenderMesh) -> (Point3, Point3) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for v in &mesh.vertices {
        let p = [
            f64::from(v.position[0]) + mesh.center.x(),
            f64::from(v.position[1]) + mesh.center.y(),
            f64::from(v.position[2]) + mesh.center.z(),
        ];
        for i in 0..3 {
            if p[i] < min[i] {
                min[i] = p[i];
            }
            if p[i] > max[i] {
                max[i] = p[i];
            }
        }
    }
    if !min[0].is_finite() {
        return (Point3::new(-1.0, -1.0, -1.0), Point3::new(1.0, 1.0, 1.0));
    }
    (
        Point3::new(min[0], min[1], min[2]),
        Point3::new(max[0], max[1], max[2]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model spans `[distance - radius, distance + radius]` along the view
    /// ray. Some of it must be visible (not entirely clipped) and the back must
    /// not be clipped away, with a valid frustum throughout. (The near plane may
    /// legitimately sit ahead of the model front when the eye is *inside* the
    /// bounding sphere — that geometry is correctly behind the camera.)
    fn model_visible(cam: &OrbitCamera) -> bool {
        let (near, far) = cam.clip_planes();
        let model_far = cam.distance + cam.radius;
        near > 0.0 && near < far && near < model_far && far >= model_far
    }

    /// When the eye is outside the bounding sphere (the normal framing regime),
    /// the near plane must also not clip the *front* of the model.
    fn whole_model_visible(cam: &OrbitCamera) -> bool {
        let (near, _far) = cam.clip_planes();
        let model_near = cam.distance - cam.radius;
        model_visible(cam) && near <= model_near + 1e-9
    }

    #[test]
    fn clip_planes_valid_across_full_zoom_range() {
        let mut cam = OrbitCamera::framing(Point3::new(10.0, 20.0, 30.0), 50.0);
        // Initial framing sits well outside the model, so the whole model fits.
        assert!(
            whole_model_visible(&cam),
            "initial framing must show the whole model"
        );

        // Zoom all the way in: repeated dolly-in must never produce an invalid
        // frustum or clip the model entirely out of view.
        for _ in 0..200 {
            cam.dolly(1.0);
            let (near, far) = cam.clip_planes();
            assert!(near > 0.0, "near must stay positive (near={near})");
            assert!(near < far, "near < far must hold (near={near} far={far})");
            assert!(model_visible(&cam), "model must stay visible zooming in");
        }

        // Zoom all the way out: the model must remain fully framed (eye stays
        // outside the bounding sphere), so this is the regime the P1 bug hit.
        let mut cam = OrbitCamera::framing(Point3::new(0.0, 0.0, 0.0), 2.0);
        for _ in 0..200 {
            cam.dolly(-1.0);
            assert!(
                whole_model_visible(&cam),
                "whole model must stay framed zooming out (distance={} near/far stale?)",
                cam.distance
            );
        }
    }

    #[test]
    fn dolly_floors_distance_above_zero() {
        let mut cam = OrbitCamera::framing(Point3::new(0.0, 0.0, 0.0), 10.0);
        for _ in 0..1000 {
            cam.dolly(1.0);
        }
        assert!(
            cam.distance > 0.0,
            "distance must not collapse to zero (distance={})",
            cam.distance
        );
    }

    #[test]
    fn tiny_radius_still_yields_valid_planes() {
        let cam = OrbitCamera::framing(Point3::new(0.0, 0.0, 0.0), 1e-9);
        let (near, far) = cam.clip_planes();
        assert!(near > 0.0 && near < far, "near={near} far={far}");
    }
}

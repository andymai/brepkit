//! The wgpu pipeline: adapter/device setup, render passes, readback.
//!
//! The pieces here are shared by two render targets: the offscreen path
//! ([`render`], which draws to textures and reads them back) and the
//! interactive viewer ([`crate::viewer`], which draws to a window surface).
//! [`GpuContext`], [`Pipelines`], [`GeometryBuffers`], and [`encode_scene`] are
//! the reusable building blocks; the offscreen path also owns its readback.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::error::RenderError;
use crate::mesh::{EdgeVertex, RenderMesh, Vertex};
use crate::{RenderOpts, RenderOutput};

/// Uniform block shared by both shaders (must match the WGSL `Globals` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Globals {
    /// Combined view-projection matrix (column-major), RTC-folded.
    pub view_proj: [f32; 16],
    /// World-space view direction (xyz; w padding).
    pub view_dir: [f32; 4],
    /// Ambient light fraction.
    pub ambient: f32,
    /// Encoded `FaceId` (`index + 1`) to highlight, or `0` for none.
    pub selected_id: u32,
    /// Padding to a 16-byte boundary.
    pub _pad: [f32; 2],
}

pub const COLOR_FORMAT_OFFSCREEN: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Probe whether any wgpu adapter (real GPU first, then software fallback) can
/// be obtained on this machine.
///
/// Renders never run when this returns `false`; useful for gating tests in
/// headless environments. Returns the adapter backend/name on success.
#[must_use]
pub fn probe_adapter() -> Option<String> {
    let instance = wgpu::Instance::default();
    candidate_adapters(&instance, None).first().map(|adapter| {
        let info = adapter.get_info();
        format!(
            "{:?} / {} ({:?})",
            info.backend, info.name, info.device_type
        )
    })
}

/// Request the preferred (real GPU) adapter, then the software fallback.
///
/// Returns adapters in priority order (real first, fallback second); either may
/// be absent. Both are returned so device creation can fall back if the first
/// adapter fails to produce a device. When `surface` is `Some`, each adapter is
/// required to be compatible with it (a hard requirement for window presentation).
fn candidate_adapters(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Vec<wgpu::Adapter> {
    let mut out = Vec::new();
    if let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: surface,
        }))
    {
        out.push(adapter);
    }
    if let Ok(adapter) =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            force_fallback_adapter: true,
            compatible_surface: surface,
        }))
    {
        out.push(adapter);
    }
    out
}

/// Acquire an adapter + device + queue, trying each candidate adapter in
/// priority order.
///
/// A real adapter that exists but cannot create a device falls back to the
/// software adapter rather than failing outright. The chosen adapter is
/// returned alongside the device so callers (the viewer) can query surface
/// capabilities. When `surface` is `Some`, only surface-compatible adapters are
/// considered.
pub fn acquire_device(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface<'_>>,
) -> Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue), RenderError> {
    let adapters = candidate_adapters(instance, surface);
    if adapters.is_empty() {
        return Err(RenderError::NoAdapter(
            "request_adapter returned no adapter".into(),
        ));
    }
    let mut last_err = String::new();
    for adapter in adapters {
        match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("brepkit-render device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            ..Default::default()
        })) {
            Ok((device, queue)) => return Ok((adapter, device, queue)),
            Err(e) => last_err = e.to_string(),
        }
    }
    Err(RenderError::DeviceRequest(last_err))
}

/// A wgpu adapter/device/queue, set up once and shared across frames.
///
/// The instance used to obtain these is dropped after device creation — wgpu's
/// adapter/device/surface keep their own handles to the backend.
pub struct GpuContext {
    // Read by the viewer (surface capabilities) but not by the offscreen path.
    #[cfg_attr(not(feature = "window"), allow(dead_code))]
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Create a surfaceless context for the offscreen path.
    ///
    /// Window callers must use [`GpuContext::with_instance`] instead, passing a
    /// surface created from the *same* instance — a surface created from a
    /// different instance than the adapter/device is invalid on strict backends,
    /// so this constructor deliberately does not accept one.
    ///
    /// # Errors
    ///
    /// [`RenderError::NoAdapter`] if no adapter exists, or
    /// [`RenderError::DeviceRequest`] if the device cannot be created.
    pub fn new() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::default();
        Self::with_instance(instance, None)
    }

    /// Build a context from an existing instance, optionally constraining the
    /// adapter to be compatible with `surface`.
    ///
    /// The surface (when `Some`) must have been created from `instance`, so the
    /// viewer builds the instance first, creates the surface from it, then hands
    /// both here — guaranteeing the adapter/device and surface share an instance.
    ///
    /// # Errors
    ///
    /// See [`GpuContext::new`].
    pub fn with_instance(
        instance: wgpu::Instance,
        surface: Option<&wgpu::Surface<'_>>,
    ) -> Result<Self, RenderError> {
        let (adapter, device, queue) = acquire_device(&instance, surface)?;
        Ok(Self {
            adapter,
            device,
            queue,
        })
    }
}

/// The globals uniform buffer plus its bind group and layout.
///
/// The buffer is `COPY_DST` so the viewer can re-upload [`Globals`] every frame
/// (camera orbit, selection change) without rebuilding the bind group.
pub struct GlobalsBinding {
    // Retained to keep the uniform buffer alive (it backs `bind_group`); also
    // re-uploaded each frame by the viewer via `upload`.
    #[cfg_attr(not(feature = "window"), allow(dead_code))]
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub layout: wgpu::BindGroupLayout,
}

impl GlobalsBinding {
    pub fn new(device: &wgpu::Device, globals: &Globals) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("globals"),
            contents: bytemuck::bytes_of(globals),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals bind group"),
            layout: &layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            buffer,
            bind_group,
            layout,
        }
    }

    /// Re-upload the uniform block (call before encoding a frame).
    #[cfg_attr(not(feature = "window"), allow(dead_code))]
    pub fn upload(&self, queue: &wgpu::Queue, globals: &Globals) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(globals));
    }
}

/// The mesh pipeline plus the optional edge pipeline, built for one color
/// format. Both passes target `[color, id]` so the id buffer is always written
/// alongside the shaded image (the edge pass masks out id writes).
pub struct Pipelines {
    pub mesh: wgpu::RenderPipeline,
    pub edge: Option<wgpu::RenderPipeline>,
}

impl Pipelines {
    /// Build the pipelines for `color_format` (offscreen uses
    /// `Rgba8UnormSrgb`; the viewer uses the surface's preferred sRGB format).
    /// `with_edges` controls whether the edge pipeline is built.
    pub fn new(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        with_edges: bool,
    ) -> Self {
        let mesh_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mesh.wgsl"));
        let color_targets = [
            Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: ID_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let mesh = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("mesh pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &mesh_shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 24,
                            shader_location: 2,
                        },
                    ],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &mesh_shader,
                entry_point: Some("fs_main"),
                targets: &color_targets,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let edge = if with_edges {
            let edge_shader =
                device.create_shader_module(wgpu::include_wgsl!("../shaders/edge.wgsl"));
            // The id target is still bound during the edge pass, so give it a
            // target with writes masked off (edges only recolor; the underlying
            // face id must survive for picking).
            let edge_color_targets = [
                Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
                Some(wgpu::ColorTargetState {
                    format: ID_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::empty(),
                }),
            ];
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("edge pipeline"),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &edge_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<EdgeVertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        }],
                    }],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::LineList,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::LessEqual),
                    stencil: wgpu::StencilState::default(),
                    bias: wgpu::DepthBiasState::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &edge_shader,
                    entry_point: Some("fs_main"),
                    targets: &edge_color_targets,
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                multiview_mask: None,
                cache: None,
            });
            Some(pipeline)
        } else {
            None
        };

        Self { mesh, edge }
    }
}

/// GPU vertex/index/edge buffers built once from a [`RenderMesh`].
pub struct GeometryBuffers {
    pub vertex: wgpu::Buffer,
    pub index: wgpu::Buffer,
    pub index_count: u32,
    pub edge: Option<wgpu::Buffer>,
    pub edge_count: u32,
}

impl GeometryBuffers {
    pub fn new(device: &wgpu::Device, mesh: &RenderMesh) -> Self {
        let vertex = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        #[allow(clippy::cast_possible_truncation)]
        let index_count = mesh.indices.len() as u32;

        let (edge, edge_count) = if mesh.edge_vertices.is_empty() {
            (None, 0)
        } else {
            let buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("edge vertices"),
                contents: bytemuck::cast_slice(&mesh.edge_vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
            #[allow(clippy::cast_possible_truncation)]
            let count = mesh.edge_vertices.len() as u32;
            (Some(buf), count)
        };

        Self {
            vertex,
            index,
            index_count,
            edge,
            edge_count,
        }
    }
}

/// Views and clear color for one scene-pass encode.
pub struct PassTargets<'a> {
    pub color: &'a wgpu::TextureView,
    pub id: &'a wgpu::TextureView,
    pub depth: &'a wgpu::TextureView,
    pub background: [f32; 4],
}

/// Encode the mesh pass (and the edge pass, if both the pipeline and the
/// geometry have edges) into `encoder`, drawing to `targets`.
///
/// Shared verbatim by the offscreen path and the viewer so the two never drift.
pub fn encode_scene(
    encoder: &mut wgpu::CommandEncoder,
    pipelines: &Pipelines,
    globals: &GlobalsBinding,
    geometry: &GeometryBuffers,
    targets: &PassTargets<'_>,
) {
    let bg = targets.background;
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("mesh + edge pass"),
        color_attachments: &[
            Some(wgpu::RenderPassColorAttachment {
                view: targets.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: f64::from(bg[0]),
                        g: f64::from(bg[1]),
                        b: f64::from(bg[2]),
                        a: f64::from(bg[3]),
                    }),
                    store: wgpu::StoreOp::Store,
                },
            }),
            Some(wgpu::RenderPassColorAttachment {
                view: targets.id,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // 0 = background sentinel.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            }),
        ],
        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
            view: targets.depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });

    pass.set_bind_group(0, &globals.bind_group, &[]);
    pass.set_pipeline(&pipelines.mesh);
    pass.set_vertex_buffer(0, geometry.vertex.slice(..));
    pass.set_index_buffer(geometry.index.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..geometry.index_count, 0, 0..1);

    if let (Some(edge_pipeline), Some(edge_buf)) = (pipelines.edge.as_ref(), geometry.edge.as_ref())
    {
        pass.set_pipeline(edge_pipeline);
        pass.set_vertex_buffer(0, edge_buf.slice(..));
        pass.draw(0..geometry.edge_count, 0..1);
    }
}

/// Build the [`Globals`] block for a frame from the camera, RTC center, and the
/// rendering options (with no face selected).
pub fn build_globals(cam: &Camera, center: brepkit_math::vec::Point3, ambient: f32) -> Globals {
    let view_proj = crate::camera::view_proj_rtc(cam, center);
    let view_dir = cam.view_direction();
    #[allow(clippy::cast_possible_truncation)]
    Globals {
        view_proj,
        view_dir: [
            view_dir.x() as f32,
            view_dir.y() as f32,
            view_dir.z() as f32,
            0.0,
        ],
        ambient,
        selected_id: 0,
        _pad: [0.0; 2],
    }
}

/// Render a solid's prepared geometry offscreen and read back color + ids.
///
/// `mesh` is the center-relative geometry; `cam` and `opts` control the view
/// and targets. This performs all GPU work synchronously (blocking on async
/// via `pollster`).
///
/// # Errors
///
/// See [`crate::render_solid_offscreen`].
#[allow(clippy::too_many_lines)]
pub fn render(
    mesh: &RenderMesh,
    cam: &Camera,
    opts: &RenderOpts,
) -> Result<RenderOutput, RenderError> {
    if opts.width == 0 || opts.height == 0 {
        return Err(RenderError::InvalidSize {
            width: opts.width,
            height: opts.height,
        });
    }

    let ctx = GpuContext::new()?;
    let device = &ctx.device;
    let queue = &ctx.queue;

    let (width, height) = (opts.width, opts.height);
    // Reject oversized targets with a clean error rather than tripping wgpu's
    // internal validation (which surfaces as a device-error/panic path).
    let max = device.limits().max_texture_dimension_2d;
    if width > max || height > max {
        return Err(RenderError::SizeTooLarge { width, height, max });
    }
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    // --- Targets -----------------------------------------------------------
    let color_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("color target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: COLOR_FORMAT_OFFSCREEN,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let id_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("id target"),
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: ID_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
    let id_view = id_tex.create_view(&wgpu::TextureViewDescriptor::default());

    // --- Shared GPU objects ------------------------------------------------
    let globals = build_globals(cam, mesh.center, opts.ambient);
    let globals_binding = GlobalsBinding::new(device, &globals);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[Some(&globals_binding.layout)],
        immediate_size: 0,
    });
    let with_edges = opts.edges && !mesh.edge_vertices.is_empty();
    let pipelines = Pipelines::new(device, &pipeline_layout, COLOR_FORMAT_OFFSCREEN, with_edges);
    let geometry = GeometryBuffers::new(device, mesh);

    // --- Readback buffers --------------------------------------------------
    // Bytes per row must be a multiple of COPY_BYTES_PER_ROW_ALIGNMENT (256).
    let color_bpp = 4_u32; // Rgba8
    let id_bpp = 4_u32; // R32Uint
    let color_padded_bpr = padded_bytes_per_row(width, color_bpp);
    let id_padded_bpr = padded_bytes_per_row(width, id_bpp);

    let color_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("color readback"),
        size: u64::from(color_padded_bpr) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let id_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("id readback"),
        size: u64::from(id_padded_bpr) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // --- Encode ------------------------------------------------------------
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    encode_scene(
        &mut encoder,
        &pipelines,
        &globals_binding,
        &geometry,
        &PassTargets {
            color: &color_view,
            id: &id_view,
            depth: &depth_view,
            background: opts.background,
        },
    );

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &color_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(color_padded_bpr),
                rows_per_image: Some(height),
            },
        },
        extent,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &id_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &id_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(id_padded_bpr),
                rows_per_image: Some(height),
            },
        },
        extent,
    );

    queue.submit(Some(encoder.finish()));

    // --- Map + read --------------------------------------------------------
    let color_bytes = map_and_read(device, &color_readback)?;
    let id_bytes = map_and_read(device, &id_readback)?;

    let color = unpad_to_rgba(&color_bytes, width, height, color_padded_bpr);
    let id_buffer = unpad_to_u32(&id_bytes, width, height, id_padded_bpr);

    Ok(RenderOutput {
        color,
        id_buffer,
        width,
        height,
    })
}

/// Round `width * bpp` up to the next multiple of the row-copy alignment.
pub fn padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

/// Map a readback buffer (blocking) and copy its bytes out.
pub fn map_and_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, RenderError> {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |res| {
        let _ = tx.send(res);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| RenderError::Poll(e.to_string()))?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return Err(RenderError::BufferMap(e.to_string())),
        Err(e) => return Err(RenderError::BufferMap(e.to_string())),
    }
    let data = buffer.slice(..).get_mapped_range().to_vec();
    buffer.unmap();
    Ok(data)
}

/// Strip per-row copy padding and build an RGBA image (rows are tightly packed).
pub fn unpad_to_rgba(bytes: &[u8], width: u32, height: u32, padded_bpr: u32) -> image::RgbaImage {
    let row_len = (width * 4) as usize;
    let mut packed = Vec::with_capacity(row_len * height as usize);
    for row in 0..height as usize {
        let start = row * padded_bpr as usize;
        if let Some(slice) = bytes.get(start..start + row_len) {
            packed.extend_from_slice(slice);
        } else {
            packed.resize(packed.len() + row_len, 0);
        }
    }
    image::RgbaImage::from_raw(width, height, packed)
        .unwrap_or_else(|| image::RgbaImage::new(width, height))
}

/// Strip per-row copy padding and decode the R32Uint id target to `Vec<u32>`.
pub fn unpad_to_u32(bytes: &[u8], width: u32, height: u32, padded_bpr: u32) -> Vec<u32> {
    let mut out = Vec::with_capacity((width * height) as usize);
    for row in 0..height as usize {
        let row_start = row * padded_bpr as usize;
        for col in 0..width as usize {
            let off = row_start + col * 4;
            let v = bytes
                .get(off..off + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .unwrap_or(0);
            out.push(v);
        }
    }
    out
}

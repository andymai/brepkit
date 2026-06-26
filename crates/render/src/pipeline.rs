//! The offscreen wgpu pipeline: adapter/device setup, render passes, readback.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::error::RenderError;
use crate::mesh::{EdgeVertex, RenderMesh, Vertex};
use crate::{RenderOpts, RenderOutput};

/// Uniform block shared by both shaders (must match the WGSL `Globals` layout).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [f32; 16],
    view_dir: [f32; 4],
    ambient: f32,
    _pad: [f32; 3],
}

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const ID_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;

/// Probe whether any wgpu adapter (real GPU first, then software fallback) can
/// be obtained on this machine.
///
/// Renders never run when this returns `false`; useful for gating tests in
/// headless environments. Returns the adapter backend/name on success.
#[must_use]
pub fn probe_adapter() -> Option<String> {
    let instance = wgpu::Instance::default();
    request_any_adapter(&instance).map(|adapter| {
        let info = adapter.get_info();
        format!(
            "{:?} / {} ({:?})",
            info.backend, info.name, info.device_type
        )
    })
}

/// Request a real adapter, falling back to a software adapter.
fn request_any_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    let real = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
    }));
    if let Ok(adapter) = real {
        return Some(adapter);
    }
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::LowPower,
        force_fallback_adapter: true,
        compatible_surface: None,
    }))
    .ok()
}

/// Render a solid's prepared geometry offscreen and read back color + ids.
///
/// `mesh` is the center-relative geometry; `cam` and `opts` control the view
/// and targets. This performs all GPU work synchronously (blocking on async
/// via `pollster`).
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

    let instance = wgpu::Instance::default();
    let adapter = request_any_adapter(&instance)
        .ok_or_else(|| RenderError::NoAdapter("request_adapter returned no adapter".into()))?;

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("brepkit-render device"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults(),
        ..Default::default()
    }))
    .map_err(|e| RenderError::DeviceRequest(e.to_string()))?;

    let (width, height) = (opts.width, opts.height);
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
        format: COLOR_FORMAT,
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

    // --- Uniforms ----------------------------------------------------------
    let view_proj = crate::camera::view_proj_rtc(cam, mesh.center);
    let view_dir = cam.view_direction();
    #[allow(clippy::cast_possible_truncation)]
    let globals = Globals {
        view_proj,
        view_dir: [
            view_dir.x() as f32,
            view_dir.y() as f32,
            view_dir.z() as f32,
            0.0,
        ],
        ambient: opts.ambient,
        _pad: [0.0; 3],
    };
    let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("globals"),
        contents: bytemuck::bytes_of(&globals),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buf.as_entire_binding(),
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("pipeline layout"),
        bind_group_layouts: &[Some(&bind_group_layout)],
        immediate_size: 0,
    });

    // --- Geometry buffers --------------------------------------------------
    let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh vertices"),
        contents: bytemuck::cast_slice(&mesh.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("mesh indices"),
        contents: bytemuck::cast_slice(&mesh.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // --- Mesh pipeline -----------------------------------------------------
    let mesh_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mesh.wgsl"));
    let color_targets = [
        Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: ID_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    let mesh_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("mesh pipeline"),
        layout: Some(&pipeline_layout),
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

    // --- Edge pipeline (optional) -----------------------------------------
    let edge_resources = if opts.edges && !mesh.edge_vertices.is_empty() {
        let edge_shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/edge.wgsl"));
        let edge_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("edge vertices"),
            contents: bytemuck::cast_slice(&mesh.edge_vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        // The id target is still bound during the edge pass, so give it a
        // write target (edges keep the underlying face id; they only recolor).
        let edge_color_targets = [
            Some(wgpu::ColorTargetState {
                format: COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: ID_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::empty(),
            }),
        ];
        let edge_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("edge pipeline"),
            layout: Some(&pipeline_layout),
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
        #[allow(clippy::cast_possible_truncation)]
        let edge_count = mesh.edge_vertices.len() as u32;
        Some((edge_pipeline, edge_buf, edge_count))
    } else {
        None
    };

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
    let bg = opts.background;
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("encoder"),
    });
    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("mesh + edge pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
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
                    view: &id_view,
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
                view: &depth_view,
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

        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&mesh_pipeline);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        #[allow(clippy::cast_possible_truncation)]
        let index_count = mesh.indices.len() as u32;
        pass.draw_indexed(0..index_count, 0, 0..1);

        if let Some((edge_pipeline, edge_buf, edge_count)) = edge_resources.as_ref() {
            pass.set_pipeline(edge_pipeline);
            pass.set_vertex_buffer(0, edge_buf.slice(..));
            pass.draw(0..*edge_count, 0..1);
        }
    }

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
    let color_bytes = map_and_read(&device, &color_readback)?;
    let id_bytes = map_and_read(&device, &id_readback)?;

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
fn padded_bytes_per_row(width: u32, bytes_per_pixel: u32) -> u32 {
    let unpadded = width * bytes_per_pixel;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    unpadded.div_ceil(align) * align
}

/// Map a readback buffer (blocking) and copy its bytes out.
fn map_and_read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Result<Vec<u8>, RenderError> {
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
fn unpad_to_rgba(bytes: &[u8], width: u32, height: u32, padded_bpr: u32) -> image::RgbaImage {
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
fn unpad_to_u32(bytes: &[u8], width: u32, height: u32, padded_bpr: u32) -> Vec<u32> {
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

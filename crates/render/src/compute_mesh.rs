//! GPU compute-shader mesher for analytic quadric surfaces.
//!
//! brepkit emits exact analytic surfaces (cylinder, cone, sphere, torus).
//! Rather than CPU-tessellating one into thousands of triangles and uploading
//! them, this path uploads the surface's *parameters* and lets a WGSL compute
//! shader evaluate the parametric surface into a vertex grid at a caller-chosen
//! tessellation factor (the LOD knob). The compute output then feeds the same
//! offscreen mesh draw pass as the solid path (`shaders/mesh.wgsl`).
//!
//! WebGPU/wgpu have no tessellation or mesh shaders, so the per-vertex
//! evaluation runs in a compute pass. This module currently meshes a cylinder;
//! the descriptor + shader generalize to the other quadrics (see the crate
//! docs for the extension plan).
//!
//! # Precision
//!
//! As with the solid path, positions are emitted relative to the model center
//! (RTC) and the f64 center is folded into the camera matrix on the CPU, so the
//! GPU never sees large absolute coordinates.

use std::f64::consts::TAU;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::camera::Camera;
use crate::error::RenderError;
use crate::pipeline;
use crate::{RenderOpts, RenderOutput};

/// Tessellation factor (level of detail) for the compute mesher.
///
/// `n_u` angular steps around the surface and `n_v` steps along it. Higher
/// values produce more triangles and a rounder silhouette at the cost of a
/// larger vertex buffer; for a cylinder the chord error of the circular cross
/// section falls off as `1 - cos(π / n_u)`.
#[derive(Debug, Clone, Copy)]
pub struct TessFactor {
    /// Angular subdivisions around the surface (must be >= 3).
    pub n_u: u32,
    /// Axial subdivisions along the surface (must be >= 1).
    pub n_v: u32,
}

impl TessFactor {
    /// Create a tessellation factor, clamping to the minimum that yields a
    /// non-degenerate closed mesh (`n_u >= 3`, `n_v >= 1`).
    #[must_use]
    pub fn new(n_u: u32, n_v: u32) -> Self {
        Self {
            n_u: n_u.max(3),
            n_v: n_v.max(1),
        }
    }

    // TODO: view-dependent (screen-space) LOD — pick (n_u, n_v) per frame from a
    // screen-space chord-error metric (projected radius / pixel budget) instead
    // of a caller-supplied factor. Next milestone.
}

/// A cylinder surface packed for GPU evaluation.
///
/// Extracted from a [`FaceSurface::Cylinder`] face via
/// [`extract_cylinder_descriptor`]: the axis frame and radius come from the
/// [`CylindricalSurface`]; the parametric trim range (`v0..v1` axial,
/// `u0..u1` angular) comes from the face's boundary.
#[derive(Debug, Clone, Copy)]
pub struct CylinderDescriptor {
    /// RTC origin — positions are emitted relative to this point. Defaults to
    /// the descriptor's own AABB center via [`extract_cylinder_descriptor`].
    pub center: Point3,
    /// Cylinder axis origin (a point on the axis at `v = 0`).
    pub axis_origin: Point3,
    /// Cylinder axis direction (unit, points along increasing `v`).
    pub axis: Vec3,
    /// First radial reference direction (unit; `u = 0` points here).
    pub x_ref: Vec3,
    /// Second radial reference direction (unit; `axis × x_ref`).
    pub y_ref: Vec3,
    /// Cylinder radius.
    pub radius: f64,
    /// Axial parameter at the lower trim boundary.
    pub v0: f64,
    /// Axial parameter at the upper trim boundary.
    pub v1: f64,
    /// Angular parameter at the start of the trim (radians).
    pub u0: f64,
    /// Angular parameter at the end of the trim (radians); `u1 - u0 == 2π` for
    /// a full cylinder.
    pub u1: f64,
}

impl CylinderDescriptor {
    /// World-space point on the surface at parameters `(u, v)`, the same
    /// parameterization the GPU shader evaluates:
    /// `pos(u, v) = axis_origin + radius·(cos u · x_ref + sin u · y_ref) + v·axis`.
    ///
    /// Note this returns an absolute world point; the GPU emits it minus
    /// [`center`](Self::center) (RTC). Used to compute the descriptor's AABB and
    /// available for CPU-side geometric checks.
    #[must_use]
    pub fn evaluate(&self, u: f64, v: f64) -> Point3 {
        let radial = self.x_ref * (self.radius * u.cos()) + self.y_ref * (self.radius * u.sin());
        self.axis_origin + radial + self.axis * v
    }

    /// Axis-aligned bounding box of the trimmed cylinder, sampled around the
    /// angular range and across the two axial caps.
    fn aabb(&self) -> (Point3, Point3) {
        let mut min = [f64::INFINITY; 3];
        let mut max = [f64::NEG_INFINITY; 3];
        let samples = 64;
        for k in 0..=samples {
            let t = f64::from(k) / f64::from(samples);
            let u = self.u0 + (self.u1 - self.u0) * t;
            for &v in &[self.v0, self.v1] {
                let p = self.evaluate(u, v);
                let c = [p.x(), p.y(), p.z()];
                for axis in 0..3 {
                    min[axis] = min[axis].min(c[axis]);
                    max[axis] = max[axis].max(c[axis]);
                }
            }
        }
        (
            Point3::new(min[0], min[1], min[2]),
            Point3::new(max[0], max[1], max[2]),
        )
    }

    /// Number of triangles a given tessellation factor produces (`2·n_u·n_v`).
    #[must_use]
    pub fn triangle_count(tess: TessFactor) -> usize {
        2 * tess.n_u as usize * tess.n_v as usize
    }
}

/// Extract a [`CylinderDescriptor`] from a cylindrical face.
///
/// Reads the [`CylindricalSurface`] frame and radius, then derives the axial
/// trim range `v0..v1` by projecting the face's outer-wire vertices onto the
/// axis. The angular range is taken as a full revolution (`0..2π`) — the M2
/// scope is a full cylinder (e.g. [`make_cylinder`](brepkit_operations::primitives::make_cylinder)),
/// whose lateral face wraps the entire circle via a degenerate seam wire.
///
/// `center` is set to the descriptor's own AABB center, so the returned
/// descriptor renders correctly on its own.
///
/// # Errors
///
/// - [`RenderError::Operations`] if `face` is not a cylindrical face.
/// - [`RenderError::Topology`] if the face's wire/edge/vertex lookups fail.
pub fn extract_cylinder_descriptor(
    topo: &Topology,
    face: FaceId,
) -> Result<CylinderDescriptor, RenderError> {
    let face_data = topo.face(face)?;
    let FaceSurface::Cylinder(cyl) = face_data.surface() else {
        return Err(RenderError::Operations(
            brepkit_operations::OperationsError::InvalidInput {
                reason: "extract_cylinder_descriptor: face is not a cylindrical surface".into(),
            },
        ));
    };

    let (v0, v1) = axial_range(topo, face, cyl)?;

    let mut desc = CylinderDescriptor {
        center: Point3::new(0.0, 0.0, 0.0),
        axis_origin: cyl.origin(),
        axis: cyl.axis(),
        x_ref: cyl.x_axis(),
        y_ref: cyl.y_axis(),
        radius: cyl.radius(),
        v0,
        v1,
        u0: 0.0,
        u1: TAU,
    };
    let (min, max) = desc.aabb();
    desc.center = Point3::new(
        (min.x() + max.x()) * 0.5,
        (min.y() + max.y()) * 0.5,
        (min.z() + max.z()) * 0.5,
    );
    Ok(desc)
}

/// Project every vertex of the face's outer wire onto the cylinder axis to find
/// the axial parameter span `[v0, v1]`.
fn axial_range(
    topo: &Topology,
    face: FaceId,
    cyl: &CylindricalSurface,
) -> Result<(f64, f64), RenderError> {
    let face_data = topo.face(face)?;
    let wire = topo.wire(face_data.outer_wire())?;
    let axis = cyl.axis();
    let origin = cyl.origin();

    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        for vid in [edge.start(), edge.end()] {
            let p = topo.vertex(vid)?.point();
            let v = axis.dot(p - origin);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
    }
    if !(min_v.is_finite() && max_v.is_finite()) || (max_v - min_v).abs() < f64::EPSILON {
        return Err(RenderError::Operations(
            brepkit_operations::OperationsError::InvalidInput {
                reason: "extract_cylinder_descriptor: degenerate axial range on cylinder face"
                    .into(),
            },
        ));
    }
    Ok((min_v, max_v))
}

/// GPU descriptor uniform. Field order and padding match the WGSL `Descriptor`
/// struct in `quadric_mesh.wgsl` (vec3 fields are 16-byte aligned, with the
/// trailing scalar packed into the 4th word of each 16-byte slot; the final
/// group pads to 16 bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GpuDescriptor {
    center: [f32; 3],
    radius: f32,
    axis_origin: [f32; 3],
    v0: f32,
    axis: [f32; 3],
    v1: f32,
    x_ref: [f32; 3],
    u0: f32,
    y_ref: [f32; 3],
    u1: f32,
    n_u: u32,
    n_v: u32,
    face_id: u32,
    _pad: u32,
}

/// Render a compute-meshed cylinder offscreen to a shaded color image + face-id
/// buffer.
///
/// The cylinder is meshed entirely on the GPU from `desc` at the `tess` LOD: a
/// compute pass evaluates the parametric surface into vertex + index storage
/// buffers, which are then drawn by the same offscreen mesh pass as the solid
/// path. Every emitted vertex carries `face_id` (use `1` if you have no real
/// face).
///
/// # Errors
///
/// - [`RenderError::InvalidSize`] if `opts.width` or `opts.height` is zero.
/// - [`RenderError::NoAdapter`] / [`RenderError::DeviceRequest`] on GPU setup.
/// - [`RenderError::BufferMap`] / [`RenderError::Poll`] on readback failure.
#[allow(clippy::too_many_lines)]
pub fn render_cylinder_compute_offscreen(
    desc: &CylinderDescriptor,
    tess: TessFactor,
    face_id: u32,
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
    let (_adapter, device, queue) = pipeline::acquire_device(&instance, None)?;

    // Reject oversized targets with a clean error rather than tripping wgpu's
    // internal validation (mirrors the solid path).
    let max = device.limits().max_texture_dimension_2d;
    if opts.width > max || opts.height > max {
        return Err(RenderError::SizeTooLarge {
            width: opts.width,
            height: opts.height,
            max,
        });
    }

    // --- Grid sizing -------------------------------------------------------
    let full = (desc.u1 - desc.u0 - TAU).abs() < 1.0e-6;
    let cols = if full { tess.n_u } else { tess.n_u + 1 };
    let rows = tess.n_v + 1;
    let vertex_count = u64::from(cols) * u64::from(rows);
    let index_count = u64::from(tess.n_u) * u64::from(tess.n_v) * 6;
    let vert_bytes = vertex_count * 7 * 4; // 7 u32 words per vertex
    let index_bytes = index_count * 4;

    // --- Descriptor uniform -----------------------------------------------
    #[allow(clippy::cast_possible_truncation)]
    let gpu_desc = GpuDescriptor {
        center: pt_f32(desc.center),
        radius: desc.radius as f32,
        axis_origin: pt_f32(desc.axis_origin),
        v0: desc.v0 as f32,
        axis: vec_f32(desc.axis),
        v1: desc.v1 as f32,
        x_ref: vec_f32(desc.x_ref),
        u0: desc.u0 as f32,
        y_ref: vec_f32(desc.y_ref),
        u1: desc.u1 as f32,
        n_u: tess.n_u,
        n_v: tess.n_v,
        face_id,
        _pad: 0,
    };
    let desc_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("cylinder descriptor"),
        contents: bytemuck::bytes_of(&gpu_desc),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // --- Compute output buffers (also used directly as draw inputs) --------
    let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compute vertices"),
        size: vert_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    });
    let index_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compute indices"),
        size: index_bytes,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    });

    // --- Compute pipeline --------------------------------------------------
    let compute_shader =
        device.create_shader_module(wgpu::include_wgsl!("../shaders/quadric_mesh.wgsl"));
    let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("compute mesher layout"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_entry(1),
            storage_entry(2),
        ],
    });
    let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("compute mesher bind group"),
        layout: &compute_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: desc_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: index_buf.as_entire_binding(),
            },
        ],
    });
    let compute_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("compute pipeline layout"),
        bind_group_layouts: &[Some(&compute_bgl)],
        immediate_size: 0,
    });
    let vertex_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cylinder vertex mesher"),
        layout: Some(&compute_layout),
        module: &compute_shader,
        entry_point: Some("cs_vertices"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });
    let index_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("cylinder index mesher"),
        layout: Some(&compute_layout),
        module: &compute_shader,
        entry_point: Some("cs_indices"),
        compilation_options: wgpu::PipelineCompilationOptions::default(),
        cache: None,
    });

    // --- Draw resources (shared mesh shader) -------------------------------
    let draw = build_draw_resources(&device, desc, cam, opts);

    // --- Targets -----------------------------------------------------------
    let (width, height) = (opts.width, opts.height);
    let targets = RenderTargets::new(&device, width, height);

    // --- Encode ------------------------------------------------------------
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("compute + draw encoder"),
    });
    {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("cylinder mesher"),
            timestamp_writes: None,
        });
        cpass.set_bind_group(0, &compute_bind_group, &[]);
        let groups_x = cols.div_ceil(8).max(1);
        let groups_y = rows.div_ceil(8).max(1);
        cpass.set_pipeline(&vertex_pipeline);
        cpass.dispatch_workgroups(groups_x, groups_y, 1);
        let igx = tess.n_u.div_ceil(8).max(1);
        let igy = tess.n_v.div_ceil(8).max(1);
        cpass.set_pipeline(&index_pipeline);
        cpass.dispatch_workgroups(igx, igy, 1);
    }

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("compute mesh pass"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::from(opts.background[0]),
                            g: f64::from(opts.background[1]),
                            b: f64::from(opts.background[2]),
                            a: f64::from(opts.background[3]),
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &targets.id_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth_view,
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
        pass.set_bind_group(0, &draw.bind_group, &[]);
        pass.set_pipeline(&draw.pipeline);
        pass.set_vertex_buffer(0, vertex_buf.slice(..));
        pass.set_index_buffer(index_buf.slice(..), wgpu::IndexFormat::Uint32);
        #[allow(clippy::cast_possible_truncation)]
        pass.draw_indexed(0..index_count as u32, 0, 0..1);
    }

    let color_bpr = pipeline::padded_bytes_per_row(width, 4);
    let id_bpr = pipeline::padded_bytes_per_row(width, 4);
    let color_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("color readback"),
        size: u64::from(color_bpr) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let id_readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("id readback"),
        size: u64::from(id_bpr) * u64::from(height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let extent = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.color_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &color_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(color_bpr),
                rows_per_image: Some(height),
            },
        },
        extent,
    );
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &targets.id_tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &id_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(id_bpr),
                rows_per_image: Some(height),
            },
        },
        extent,
    );

    queue.submit(Some(encoder.finish()));

    let color_bytes = pipeline::map_and_read(&device, &color_readback)?;
    let id_bytes = pipeline::map_and_read(&device, &id_readback)?;
    let color = pipeline::unpad_to_rgba(&color_bytes, width, height, color_bpr);
    let id_buffer = pipeline::unpad_to_u32(&id_bytes, width, height, id_bpr);

    Ok(RenderOutput {
        color,
        id_buffer,
        width,
        height,
    })
}

/// A read-write storage-buffer bind-group-layout entry visible to compute.
fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Mesh-draw pipeline + bind group reusing the solid path's `mesh.wgsl`.
struct DrawResources {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
}

/// Build the globals uniform, bind group, and mesh-draw pipeline for the
/// compute-generated vertex buffer (same vertex layout + shader as the solid
/// path).
fn build_draw_resources(
    device: &wgpu::Device,
    desc: &CylinderDescriptor,
    cam: &Camera,
    opts: &RenderOpts,
) -> DrawResources {
    let view_proj = crate::camera::view_proj_rtc(cam, desc.center);
    let view_dir = cam.view_direction();
    #[allow(clippy::cast_possible_truncation)]
    let globals = pipeline::Globals {
        view_proj,
        view_dir: [
            view_dir.x() as f32,
            view_dir.y() as f32,
            view_dir.z() as f32,
            0.0,
        ],
        ambient: opts.ambient,
        selected_id: 0,
        _pad: [0.0; 2],
    };
    let globals_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("globals"),
        contents: bytemuck::bytes_of(&globals),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
        layout: &bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: globals_buf.as_entire_binding(),
        }],
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("draw pipeline layout"),
        bind_group_layouts: &[Some(&bgl)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/mesh.wgsl"));
    let color_targets = [
        Some(wgpu::ColorTargetState {
            format: pipeline::COLOR_FORMAT_OFFSCREEN,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
        Some(wgpu::ColorTargetState {
            format: pipeline::ID_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        }),
    ];
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("compute mesh draw pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 28, // 7 words: pos(3) + normal(3) + face_id(1)
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
            format: pipeline::DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &color_targets,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        multiview_mask: None,
        cache: None,
    });
    DrawResources {
        pipeline,
        bind_group,
    }
}

/// The offscreen color/depth/id targets and their views.
struct RenderTargets {
    color_tex: wgpu::Texture,
    id_tex: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    id_view: wgpu::TextureView,
}

impl RenderTargets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let color_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("color target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pipeline::COLOR_FORMAT_OFFSCREEN,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pipeline::DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let id_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("id target"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: pipeline::ID_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let id_view = id_tex.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            color_tex,
            id_tex,
            color_view,
            depth_view,
            id_view,
        }
    }
}

#[allow(clippy::cast_possible_truncation)]
fn vec_f32(v: Vec3) -> [f32; 3] {
    [v.x() as f32, v.y() as f32, v.z() as f32]
}

#[allow(clippy::cast_possible_truncation)]
fn pt_f32(p: Point3) -> [f32; 3] {
    [p.x() as f32, p.y() as f32, p.z() as f32]
}

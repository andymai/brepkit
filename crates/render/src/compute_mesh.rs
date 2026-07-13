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

/// Upper bound on each tessellation dimension.
///
/// Caps `n_u`/`n_v` so the derived vertex and index counts stay far below
/// `u32::MAX` (the worst case `MAX_TESS² · 6 ≈ 1.6e9` indices) and so a single
/// quadric can never request an absurd buffer. Already well past any sane LOD
/// for one surface (a 16384-gon cross section is sub-pixel at any zoom).
const MAX_TESS: u32 = 16_384;

/// Words per emitted vertex in the flat `out_verts` storage buffer. Must match
/// `WORDS_PER_VERT` in `quadric_mesh.wgsl` and the 28-byte draw `Vertex` stride:
/// pos(3) + normal(3) + face_id(1).
const WORDS_PER_VERT: u64 = 7;

/// Tessellation factor (level of detail) for the compute mesher.
///
/// `n_u` angular steps around the surface and `n_v` steps along it. Higher
/// values produce more triangles and a rounder silhouette at the cost of a
/// larger vertex buffer; for a cylinder the chord error of the circular cross
/// section falls off as `1 - cos(π / n_u)`.
#[derive(Debug, Clone, Copy)]
pub struct TessFactor {
    /// Angular subdivisions around the surface (clamped to `[3, MAX_TESS]`).
    pub n_u: u32,
    /// Axial subdivisions along the surface (clamped to `[1, MAX_TESS]`).
    pub n_v: u32,
}

impl TessFactor {
    /// Create a tessellation factor, clamping each dimension into the range
    /// that yields a non-degenerate closed mesh without overflowing the GPU
    /// buffer index math: `n_u ∈ [3, MAX_TESS]`, `n_v ∈ [1, MAX_TESS]`.
    #[must_use]
    pub fn new(n_u: u32, n_v: u32) -> Self {
        Self {
            n_u: n_u.clamp(3, MAX_TESS),
            n_v: n_v.clamp(1, MAX_TESS),
        }
    }
}

/// Default screen-space chord-error budget, in pixels.
///
/// Sub-pixel: the faceting of a cylinder tessellated to this bound is invisible
/// at the rendered resolution.
pub const DEFAULT_TARGET_PX: f64 = 0.5;

/// Derive a [`TessFactor`] from the cylinder's *projected screen size* so the
/// silhouette's chord error stays within `target_px` pixels at the given view.
///
/// A zoomed-in cylinder (large projected radius) gets a fine mesh; a distant one
/// (small projected radius) gets a coarse mesh — view-dependent LOD, the payoff
/// of meshing analytic surfaces on the GPU from their parameters.
///
/// # Math
///
/// The chord error of an `n_u`-gon inscribed in a circle of radius `r` is
/// `ε = r·(1 − cos(π/n_u))`. Projecting `r` to pixels under perspective,
/// `r_px = r · (H/2) / (d · tan(fov_y/2))` where `H` is the viewport height and
/// `d` is the center's *view-space depth* (its projection onto the view
/// direction, `view_dir · (center − eye)`) — not the Euclidean eye distance, so
/// an off-axis cylinder at the same depth is not under-tessellated. Bounding the
/// *screen-space* error `r_px·(1 − cos(π/n_u)) ≤ target_px` and solving:
/// `n_u = ceil(π / acos(1 − clamp(target_px / r_px, 0, 2)))`. A sub-pixel
/// cylinder (`r_px ≤ target_px`) floors to the [`TessFactor`] minimum; a cylinder
/// engulfing the camera (`r_px → ∞`) requests the maximum.
///
/// `n_v` is fixed at 1: a cylinder's lateral face is *ruled* (straight and of
/// constant normal along the axis), so one axial division is geometrically and
/// shading-exact. Sphere/torus surfaces will later need `n_v` adaptivity too,
/// since they curve in both parametric directions.
///
/// The result always passes through [`TessFactor::new`], so the
/// `[3, MAX_TESS]` clamp and the buffer-overflow guard still apply.
#[must_use]
pub fn screen_space_tess_factor(
    desc: &CylinderDescriptor,
    cam: &Camera,
    viewport: (u32, u32),
    target_px: f64,
) -> TessFactor {
    let n_u = angular_subdivisions_for_screen_error(desc, cam, viewport, target_px);
    TessFactor::new(n_u, 1)
}

/// Angular subdivisions needed to keep the projected chord error within
/// `target_px`. Returns a raw count (the caller clamps via [`TessFactor::new`]).
///
/// Edge cases collapse so the clamp lands on a valid factor: an unbounded
/// projection (`r_px → ∞`, the camera engulfed by the surface) → the maximum;
/// a sub-pixel projection, a center behind the camera, or a non-finite/≤0 budget
/// → the minimum.
fn angular_subdivisions_for_screen_error(
    desc: &CylinderDescriptor,
    cam: &Camera,
    viewport: (u32, u32),
    target_px: f64,
) -> u32 {
    let (_, height) = viewport;

    // Clamp the FOV into the valid open interval `(0, π)` so `tan(fov/2)` is
    // always finite-positive (matching a sane render); an out-of-range fov must
    // not poison the projection.
    let fov_y = cam.fov_y.clamp(1.0e-4, std::f64::consts::PI - 1.0e-4);
    let half_fov_tan = (fov_y * 0.5).tan();

    // Perspective scale is set by the *view-space depth* of the center (its
    // projection onto the view axis), not the Euclidean eye distance: an
    // off-axis cylinder at the same depth must not be under-tessellated.
    let depth = cam.view_direction().dot(desc.center - cam.eye);

    // A non-finite or non-positive budget can't bound anything → max detail.
    if !(target_px.is_finite() && target_px > 0.0) {
        return MAX_TESS;
    }
    let r_px = desc.radius * (f64::from(height) * 0.5) / (depth * half_fov_tan);
    // Classify the projected radius:
    //   +∞  → the surface engulfs/fills the screen (depth → 0): finest mesh.
    //   ≤ 0 or NaN → behind the camera or degenerate: won't render → coarsest.
    //   finite > 0 → the normal screen-size formula below.
    if r_px.is_infinite() && r_px > 0.0 {
        return MAX_TESS;
    }
    if !(r_px.is_finite() && r_px > 0.0) {
        return 3;
    }

    // ratio ∈ [0, 2] keeps the acos argument (1 − ratio) in [−1, 1].
    let ratio = (target_px / r_px).clamp(0.0, 2.0);
    let theta = (1.0 - ratio).acos(); // half the per-facet angle bound
    if !(theta.is_finite() && theta > 0.0) {
        // r_px ≤ target_px (sub-pixel facets already): minimum tessellation.
        return 3;
    }
    let n = (std::f64::consts::PI / theta).ceil();
    // n is finite and ≥ 1 here; clamp into u32 range before TessFactor re-clamps.
    if n >= f64::from(MAX_TESS) {
        MAX_TESS
    } else {
        // Safe: 1 ≤ n < MAX_TESS ≤ u32::MAX, and n is finite.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let v = n as u32;
        v
    }
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
/// group fills its 16 bytes exactly).
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
    /// `1` for a full revolution (seam columns shared), `0` for a partial arc.
    /// Computed once on the CPU so the two compute entry points never re-derive
    /// the `(span ≈ 2π)` test in f32 (which could disagree with this f64 one).
    full: u32,
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

    // Normalize the factor at the boundary: the public `TessFactor::new` clamps
    // to `[3, MAX_TESS]` / `[1, MAX_TESS]`, but the fields are `pub`, so a
    // struct-literal could bypass it. Re-clamping here makes every downstream
    // count (and the u32 index cast) provably within range.
    let tess = TessFactor::new(tess.n_u, tess.n_v);

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
    // `full` is the single source of truth for the seam decision: it is uploaded
    // to the shader (see GpuDescriptor::full) so the two compute entry points
    // never recompute the `(span ≈ 2π)` test in f32. A full revolution shares
    // the u = 0 / u = 2π columns, so it emits only `n_u` columns (the wrap quad
    // reuses column 0); a partial arc emits `n_u + 1`.
    let full = (desc.u1 - desc.u0 - TAU).abs() < 1.0e-6;
    let cols = if full { tess.n_u } else { tess.n_u + 1 };
    let rows = tess.n_v + 1;
    // With both dims clamped to MAX_TESS, the worst case is MAX_TESS²·6 ≈ 1.6e9,
    // comfortably inside u32, so the index cast for `draw_indexed` cannot wrap.
    let vertex_count = u64::from(cols) * u64::from(rows);
    let index_count = u64::from(tess.n_u) * u64::from(tess.n_v) * 6;
    // The MAX_TESS clamp guarantees this fits in u32 (worst case ≈ 1.6e9); the
    // checked `try_from` keeps the draw count from ever silently wrapping even if
    // that invariant is later weakened.
    let index_count_u32 = u32::try_from(index_count).unwrap_or(u32::MAX);
    let vert_bytes = vertex_count * WORDS_PER_VERT * 4; // 4 bytes per u32 word
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
        full: u32::from(full),
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
        pass.draw_indexed(0..index_count_u32, 0, 0..1);
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

/// Render a compute-meshed cylinder with the tessellation chosen automatically
/// from its projected screen size (view-dependent LOD).
///
/// Computes a [`screen_space_tess_factor`] from `cam` and the render dimensions
/// in `opts` (bounding the silhouette chord error to `target_px` pixels — pass
/// [`DEFAULT_TARGET_PX`] for the sub-pixel default), then meshes and renders
/// exactly as [`render_cylinder_compute_offscreen`]. A near view yields a fine
/// mesh, a far view a coarse one, both staying within the pixel budget.
///
/// # Errors
///
/// Same as [`render_cylinder_compute_offscreen`].
pub fn render_cylinder_compute_screen_lod(
    desc: &CylinderDescriptor,
    face_id: u32,
    cam: &Camera,
    opts: &RenderOpts,
    target_px: f64,
) -> Result<RenderOutput, RenderError> {
    let tess = screen_space_tess_factor(desc, cam, (opts.width, opts.height), target_px);
    render_cylinder_compute_offscreen(desc, tess, face_id, cam, opts)
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
            buffers: &[Some(wgpu::VertexBufferLayout {
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
            })],
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tess_factor_clamps_below_minimum() {
        let t = TessFactor::new(0, 0);
        assert_eq!(t.n_u, 3, "n_u floors at 3 (degenerate below)");
        assert_eq!(t.n_v, 1, "n_v floors at 1");
    }

    #[test]
    fn tess_factor_clamps_above_maximum() {
        let t = TessFactor::new(u32::MAX, u32::MAX);
        assert_eq!(t.n_u, MAX_TESS, "n_u caps at MAX_TESS");
        assert_eq!(t.n_v, MAX_TESS, "n_v caps at MAX_TESS");
    }

    #[test]
    fn tess_factor_passes_through_valid_range() {
        let t = TessFactor::new(48, 4);
        assert_eq!((t.n_u, t.n_v), (48, 4));
    }

    #[test]
    fn max_tess_keeps_index_and_vertex_counts_within_u32() {
        // The buffer index math (vertex `slot`, index `quad`) runs in u32 on the
        // GPU and the draw count is u32; the clamp must keep every derived count
        // strictly inside u32 so nothing wraps. Worst case: full grid at MAX_TESS.
        let n = u64::from(MAX_TESS);
        let cols = n + 1; // partial-arc column count (the larger of the two)
        let rows = n + 1;
        let vertex_count = cols * rows;
        let index_count = n * n * 6;
        assert!(
            u32::try_from(vertex_count).is_ok(),
            "vertex_count {vertex_count} exceeds u32"
        );
        assert!(
            u32::try_from(index_count).is_ok(),
            "index_count {index_count} exceeds u32"
        );
        // The vertex word stream (7 words/vertex) also must not overflow u32
        // element indexing in the shader (`slot * WORDS_PER_VERT`).
        assert!(
            u32::try_from(vertex_count * WORDS_PER_VERT).is_ok(),
            "vertex word count exceeds u32"
        );
    }

    /// A unit cylinder of `radius` centered at the origin, axis +Z.
    fn unit_cylinder(radius: f64) -> CylinderDescriptor {
        CylinderDescriptor {
            center: Point3::new(0.0, 0.0, 0.0),
            axis_origin: Point3::new(0.0, 0.0, -1.0),
            axis: Vec3::new(0.0, 0.0, 1.0),
            x_ref: Vec3::new(1.0, 0.0, 0.0),
            y_ref: Vec3::new(0.0, 1.0, 0.0),
            radius,
            v0: 0.0,
            v1: 2.0,
            u0: 0.0,
            u1: TAU,
        }
    }

    /// A camera at distance `dist` along +X looking back at the origin.
    fn camera_at(dist: f64) -> Camera {
        Camera {
            eye: Point3::new(dist, 0.0, 0.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
            fov_y: 45.0_f64.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: dist * 10.0,
        }
    }

    #[test]
    fn screen_lod_increases_when_closer() {
        let desc = unit_cylinder(5.0);
        let viewport = (512, 512);
        let near = screen_space_tess_factor(&desc, &camera_at(20.0), viewport, 0.5);
        let far = screen_space_tess_factor(&desc, &camera_at(200.0), viewport, 0.5);
        assert!(
            near.n_u > far.n_u,
            "closer camera should subdivide more: near {} far {}",
            near.n_u,
            far.n_u
        );
        assert_eq!(near.n_v, 1, "ruled axial direction stays at 1");
        assert_eq!(far.n_v, 1);
    }

    #[test]
    fn screen_lod_increases_with_radius() {
        let viewport = (512, 512);
        let cam = camera_at(50.0);
        let small = screen_space_tess_factor(&unit_cylinder(2.0), &cam, viewport, 0.5);
        let large = screen_space_tess_factor(&unit_cylinder(40.0), &cam, viewport, 0.5);
        assert!(
            large.n_u > small.n_u,
            "larger projected radius should subdivide more: small {} large {}",
            small.n_u,
            large.n_u
        );
    }

    #[test]
    fn screen_lod_floors_at_minimum_when_subpixel() {
        // A tiny cylinder very far away projects to under a pixel: the coarsest
        // mesh (the TessFactor minimum) already satisfies any sane budget.
        let desc = unit_cylinder(0.01);
        let t = screen_space_tess_factor(&desc, &camera_at(5_000.0), (256, 256), 0.5);
        assert_eq!(t.n_u, 3, "sub-pixel cylinder floors at the minimum");
    }

    #[test]
    fn screen_lod_tighter_budget_subdivides_more() {
        let desc = unit_cylinder(5.0);
        let cam = camera_at(40.0);
        let coarse = screen_space_tess_factor(&desc, &cam, (512, 512), 2.0);
        let fine = screen_space_tess_factor(&desc, &cam, (512, 512), 0.25);
        assert!(
            fine.n_u > coarse.n_u,
            "a tighter pixel budget should subdivide more: coarse {} fine {}",
            coarse.n_u,
            fine.n_u
        );
    }

    #[test]
    fn screen_lod_handles_degenerate_inputs() {
        let desc = unit_cylinder(5.0);
        let viewport = (512, 512);

        // Zero / non-finite target budget can't bound anything → max detail.
        let t0 = screen_space_tess_factor(&desc, &camera_at(40.0), viewport, 0.0);
        assert_eq!(
            t0.n_u, MAX_TESS,
            "zero pixel budget requests the maximum LOD"
        );
        let t_nan = screen_space_tess_factor(&desc, &camera_at(40.0), viewport, f64::NAN);
        assert_eq!(t_nan.n_u, MAX_TESS, "NaN budget falls back to maximum");
    }

    #[test]
    fn screen_lod_engulfing_camera_requests_maximum() {
        // A camera engulfed by / sitting on the cylinder has zero view-space
        // depth, so the projected radius is unbounded (r_px → +∞): it must
        // tessellate FINELY (max), not coarsely (the bug this guards against —
        // the prior code treated +∞ as sub-pixel and returned the minimum).
        let desc = unit_cylinder(5.0);
        let viewport = (512, 512);
        let mut on_center = camera_at(40.0);
        on_center.eye = desc.center;
        assert_eq!(
            screen_space_tess_factor(&desc, &on_center, viewport, 0.5).n_u,
            MAX_TESS,
            "a camera engulfed by the cylinder (depth 0) must request the maximum LOD"
        );
    }

    #[test]
    fn screen_lod_clamps_extreme_fov_to_bounded_high_lod() {
        // A near-zero FOV (extreme telephoto zoom) is clamped to a valid minimum
        // rather than poisoning the projection: it yields a high but *bounded*
        // tessellation, not a degenerate one.
        let desc = unit_cylinder(5.0);
        let viewport = (512, 512);
        let mut tiny_fov = camera_at(40.0);
        tiny_fov.fov_y = 1.0e-12;
        let normal = screen_space_tess_factor(&desc, &camera_at(40.0), viewport, 0.5);
        let zoomed = screen_space_tess_factor(&desc, &tiny_fov, viewport, 0.5);
        assert!(
            zoomed.n_u > normal.n_u,
            "extreme zoom should subdivide far more than the normal fov: zoomed {} normal {}",
            zoomed.n_u,
            normal.n_u
        );
    }

    #[test]
    fn screen_lod_behind_camera_floors_at_minimum() {
        // The cylinder behind the camera (negative view-space depth) does not
        // render meaningfully → the coarsest mesh. `camera_at` looks down −X at
        // the origin, so a center placed further down +X is behind the eye.
        let viewport = (512, 512);
        let mut desc = unit_cylinder(5.0);
        let cam = camera_at(40.0); // eye at (40,0,0) looking toward −X
        desc.center = Point3::new(80.0, 0.0, 0.0); // behind the camera
        desc.axis_origin = Point3::new(80.0, 0.0, -1.0);
        let depth_is_negative = cam.view_direction().dot(desc.center - cam.eye) < 0.0;
        assert!(
            depth_is_negative,
            "test setup: center should be behind the camera"
        );
        assert_eq!(
            screen_space_tess_factor(&desc, &cam, viewport, 0.5).n_u,
            3,
            "a cylinder behind the camera floors at the minimum LOD"
        );
    }
}

// GPU compute mesher for analytic quadric surfaces.
//
// Instead of CPU-tessellating an exact analytic surface and uploading the
// triangles, the surface's *parameters* are uploaded and this shader evaluates
// the parametric surface into a vertex grid at a caller-chosen tessellation
// factor (the LOD knob). The output buffers feed the regular mesh draw pass
// unchanged (see `mesh.wgsl`).
//
// Currently meshes a cylinder. The vertex storage buffer is a flat `array<u32>`
// so the byte layout exactly matches the CPU `Vertex` struct consumed by the
// draw pass: 7 words per vertex = [px, py, pz, nx, ny, nz, face_id], a 28-byte
// stride. Positions are written relative to `desc.center` (RTC) so the f64
// model center can be folded into the camera matrix on the CPU, matching the
// precision scheme of the solid path.

struct Descriptor {
    // RTC origin: positions are emitted relative to this point.
    center: vec3<f32>,
    radius: f32,
    // A point on the cylinder axis at v = 0 (the axis need not pass through the
    // world origin).
    axis_origin: vec3<f32>,
    v0: f32,
    // Cylinder axis (unit, points along increasing v).
    axis: vec3<f32>,
    v1: f32,
    // Orthonormal radial reference frame; x_ref is u = 0.
    x_ref: vec3<f32>,
    u0: f32,
    y_ref: vec3<f32>,
    // Angular trim [u0, u1] in radians (u1 - u0 == 2π for a full cylinder).
    u1: f32,
    // Grid resolution: n_u angular steps, n_v axial steps.
    n_u: u32,
    n_v: u32,
    // FaceId.index() + 1, written into every emitted vertex (0 = background).
    face_id: u32,
    _pad: u32,
};

@group(0) @binding(0)
var<uniform> desc: Descriptor;

// Flat word streams (see the file header for the vertex layout).
@group(0) @binding(1)
var<storage, read_write> out_verts: array<u32>;

@group(0) @binding(2)
var<storage, read_write> out_indices: array<u32>;

const WORDS_PER_VERT: u32 = 7u;

// Number of unique angular columns. A full revolution shares the u = 0 and
// u = 2π columns (identical positions), so only n_u columns are emitted and the
// wrap-around quad reuses column 0 — the seam is watertight by construction.
// A partial arc (u1 - u0 < 2π) emits n_u + 1 distinct columns.
fn col_count(full: bool) -> u32 {
    if full {
        return desc.n_u;
    }
    return desc.n_u + 1u;
}

fn write_vertex(slot: u32, pos: vec3<f32>, normal: vec3<f32>) {
    let base = slot * WORDS_PER_VERT;
    out_verts[base + 0u] = bitcast<u32>(pos.x);
    out_verts[base + 1u] = bitcast<u32>(pos.y);
    out_verts[base + 2u] = bitcast<u32>(pos.z);
    out_verts[base + 3u] = bitcast<u32>(normal.x);
    out_verts[base + 4u] = bitcast<u32>(normal.y);
    out_verts[base + 5u] = bitcast<u32>(normal.z);
    out_verts[base + 6u] = desc.face_id;
}

// One invocation per emitted grid vertex. Valid columns are [0, cols) and
// valid rows are [0, n_v]. For a full revolution `cols == n_u`, so the wrap
// column (i == n_u) is never emitted — it reuses column 0 in the index buffer,
// making the seam watertight. For a partial arc `cols == n_u + 1`.
@compute @workgroup_size(8, 8, 1)
fn cs_vertices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x; // angular index
    let j = gid.y; // axial index
    let span = desc.u1 - desc.u0;
    let full = abs(span - 6.28318530717958647692) < 1.0e-6;
    let cols = col_count(full);
    if i >= cols || j > desc.n_v {
        return;
    }

    let fu = f32(i) / f32(desc.n_u);
    let u = desc.u0 + span * fu;
    let fv = f32(j) / f32(desc.n_v);
    let v = desc.v0 + (desc.v1 - desc.v0) * fv;

    let radial = cos(u) * desc.x_ref + sin(u) * desc.y_ref;
    let world = desc.axis_origin + radial * desc.radius + desc.axis * v;
    // RTC: subtract the model center so GPU coordinates stay small.
    let pos = world - desc.center;
    // Outward radial normal (direction only, unaffected by the center shift).
    let normal = normalize(radial);

    let row_stride = desc.n_v + 1u;
    let slot = i * row_stride + j;
    write_vertex(slot, pos, normal);
}

// One invocation per grid quad (i in [0, n_u), j in [0, n_v)). Each quad emits
// two triangles (6 indices). For a full revolution the last column (i ==
// n_u - 1) wraps its "next" column back to column 0 so the seam is shared.
@compute @workgroup_size(8, 8, 1)
fn cs_indices(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    if i >= desc.n_u || j >= desc.n_v {
        return;
    }
    let span = desc.u1 - desc.u0;
    let full = abs(span - 6.28318530717958647692) < 1.0e-6;

    let row_stride = desc.n_v + 1u;
    var i_next = i + 1u;
    if full && i_next == desc.n_u {
        i_next = 0u; // wrap the seam back onto column 0
    }

    let v00 = i * row_stride + j;
    let v01 = i * row_stride + (j + 1u);
    let v10 = i_next * row_stride + j;
    let v11 = i_next * row_stride + (j + 1u);

    let quad = i * desc.n_v + j;
    let base = quad * 6u;
    // CCW winding when viewed from outside (consistent outward-normal faces).
    out_indices[base + 0u] = v00;
    out_indices[base + 1u] = v10;
    out_indices[base + 2u] = v11;
    out_indices[base + 3u] = v00;
    out_indices[base + 4u] = v11;
    out_indices[base + 5u] = v01;
}

// Mesh pass: MVP transform + Lambert headlight shading.
// Writes shaded color to the color target and the triangle's FaceId to the
// id target. Positions arrive relative to the model center (RTC); the f64
// center is already folded into `view_proj` on the CPU.

struct Globals {
    view_proj: mat4x4<f32>,
    // World-space direction the camera looks along (the headlight points
    // opposite this). xyz used; w is padding for 16-byte alignment.
    view_dir: vec4<f32>,
    ambient: f32,
    // Encoded FaceId (index + 1) to highlight, or 0 for "no selection". Matches
    // the encoding written to the id target so a picked id round-trips directly.
    selected_id: u32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_id: u32,
};

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) @interpolate(flat) face_id: u32,
};

struct FsOut {
    @location(0) color: vec4<f32>,
    @location(1) face_id: u32,
};

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.clip_position = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.normal = in.normal;
    out.face_id = in.face_id;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> FsOut {
    let n = normalize(in.normal);
    // Headlight points along -view_dir (toward the camera). Two-sided so
    // back-facing triangles still receive light rather than going black.
    let light_dir = -normalize(globals.view_dir.xyz);
    let diffuse = abs(dot(n, light_dir));
    let intensity = clamp(globals.ambient + (1.0 - globals.ambient) * diffuse, 0.0, 1.0);

    // Tint the selected face warm orange; everything else uses the neutral base.
    var base = vec3<f32>(0.72, 0.74, 0.78);
    if (globals.selected_id != 0u && in.face_id == globals.selected_id) {
        base = vec3<f32>(0.95, 0.55, 0.18);
    }
    var out: FsOut;
    out.color = vec4<f32>(base * intensity, 1.0);
    out.face_id = in.face_id;
    return out;
}

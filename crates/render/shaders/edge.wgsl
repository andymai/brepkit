// Edge pass: draw topological edge polylines as dark lines over the mesh.
// A small depth bias (applied to clip-space z) lifts the lines toward the
// camera so they sit on top of the shaded surface instead of z-fighting it.

struct Globals {
    view_proj: mat4x4<f32>,
    view_dir: vec4<f32>,
    ambient: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0)
var<uniform> globals: Globals;

struct VsOut {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VsOut {
    var out: VsOut;
    var clip = globals.view_proj * vec4<f32>(position, 1.0);
    // Pull edges toward the near plane by a fraction of w (perspective-correct
    // bias). Without this the lines coincide with surface fragments and flicker.
    clip.z = clip.z - 0.0002 * clip.w;
    out.clip_position = clip;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.05, 0.05, 0.06, 1.0);
}

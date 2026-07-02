# render-verify reference

Deep catalog for `brepkit-render`. Everything here was verified against the repo; when in doubt, re-check with the `rg` patterns given. Line numbers are deliberately absent.

## 1. Crate map

`crates/render/` is an L4 leaf crate. Allowed deps: `brepkit-math`, `brepkit-topology`, `brepkit-operations`. `scripts/check-boundaries.sh` also rejects any crate that depends on render (see the `layer-boundaries` skill). Stack: wgpu + pollster; winit only behind the optional `window` feature (`window = ["dep:winit", "dep:log"]` in `crates/render/Cargo.toml`; the `viewer` example declares `required-features = ["window"]`).

| File | Provides |
|---|---|
| `src/lib.rs` | `render_solid_offscreen`, `RenderOpts`, `RenderOutput`, `DEFAULT_DEFLECTION`, re-exports |
| `src/pipeline.rs` | `probe_adapter`, `acquire_device`, `GpuContext`, `Pipelines`, `GeometryBuffers`, `encode_scene`, texture formats |
| `src/mesh.rs` | `RenderMesh::build`: CPU tessellation to GPU buffers, face-id per vertex, RTC positions |
| `src/camera.rs` | `Camera`, `view_proj_rtc(cam, center)` (folds the f64 model center into the view matrix) |
| `src/viewer.rs` | `view_solid`, `ViewOpts` (feature `window` only) |
| `src/compute_mesh.rs` | GPU compute mesher: `CylinderDescriptor`, `extract_cylinder_descriptor`, `TessFactor`, `screen_space_tess_factor`, `render_cylinder_compute_offscreen`, `render_cylinder_compute_screen_lod`, `DEFAULT_TARGET_PX` |
| `shaders/mesh.wgsl`, `shaders/edge.wgsl` | Draw passes (shaded mesh + id target, edge overlay) |
| `shaders/quadric_mesh.wgsl` | Compute mesher: `cs_vertices` / `cs_indices` entry points, `@workgroup_size(8, 8, 1)` |
| `examples/viewer.rs` | Demo scene: 40x40x20 box fused with an r=10 h=35 cylinder via `boolean(.., BooleanOp::Fuse, ..)` |
| `tests/offscreen_render.rs` | Offscreen smoke + size-validation tests, image-assertion helpers |
| `tests/compute_mesh_render.rs` | Compute mesher vs CPU silhouette, seam, off-origin tests |
| `tests/compute_mesh_lod.rs` | Screen-space LOD behavior tests |

## 2. Offscreen API

```rust
pub fn render_solid_offscreen(
    topo: &Topology, solid: SolidId, cam: &Camera, opts: &RenderOpts,
) -> Result<RenderOutput, RenderError>
```

- `RenderOpts::new(w, h)` defaults: `edges: true`, `background: [0.11, 0.12, 0.14, 1.0]` (linear RGBA), `ambient: 0.25`, `deflection: DEFAULT_DEFLECTION` (0.05). Zero or oversized dimensions are rejected (tests `invalid_size_is_rejected`, `oversized_render_is_rejected`).
- `RenderOutput { color: image::RgbaImage, id_buffer: Vec<u32>, width, height }`. `id_buffer` is row-major, `0` = background, else `FaceId.index() + 1`. `face_id_at(x, y) -> Option<u32>` returns the same encoding, `None` for background or out-of-bounds.
- Formats (`src/pipeline.rs`): color `Rgba8UnormSrgb` (`COLOR_FORMAT_OFFSCREEN`), depth `Depth32Float` (`DEPTH_FORMAT`), id `R32Uint` (`ID_FORMAT`).
- Mesh path: `RenderMesh::build` calls `brepkit_operations::tessellate::tessellate_solid_grouped_with_tolerance` (grouped mesh with `face_offsets`, one range per face plus a sentinel) and `sample_solid_edges` for the edge overlay. Positions are uploaded as f32 relative to the model's f64 AABB center (RTC); `view_proj_rtc` folds the center into the matrix so far-from-origin models stay precise.

### Adapter gating

`probe_adapter() -> Option<String>` returns `"{backend:?} / {name} ({device_type:?})"`. Device acquisition tries a real GPU first, then a software fallback (Mesa lavapipe via Vulkan), so offscreen tests pass on machines without hardware GPUs. Every render test starts with:

```rust
let Some(_) = probe_adapter() else {
    return;
};
```

### Image-assertion helpers (copy from tests, not exported)

Defined in `crates/render/tests/*.rs`:

- `non_background_pixels(&out, bg)`: count of color pixels differing from the background.
- `id_silhouette_bbox(&out)`: bounding box of nonzero `id_buffer` pixels, `None` if blank.
- `horizontal_profile(&out)`: per-row `(min_x, max_x)` of the id silhouette; compare profiles between two renders to assert silhouette equivalence (used to compare compute mesh vs CPU mesh).
- `interior_holes(&out)`: background pixels strictly inside the silhouette; the seam-watertightness image check asserts this is 0.

Prefer id-buffer assertions over color assertions: ids are invariant to lighting and shading defaults.

## 3. Compute mesher (M2) and screen-space LOD (M2.1)

Scope: cylinder lateral faces only, full `0..2π` angular range. `extract_cylinder_descriptor(topo, face)` errors with `RenderError::Operations` if the face is not `FaceSurface::Cylinder`; the axial trim `v0..v1` comes from projecting the outer-wire vertices onto the axis. Cone, sphere, torus, quadric ray-casting, and a wasm render path are roadmap, not shipped.

- `TessFactor::new(n_u, n_v)` clamps `n_u` to `[3, 16384]`, `n_v` to `[1, 16384]` (`MAX_TESS = 16_384`).
- The WGSL writes a flat `array<u32>` at `WORDS_PER_VERT = 7` (pos 3 + normal 3 + face_id 1, the 28-byte draw `Vertex` stride) so one buffer binds as STORAGE for the compute pass and VERTEX for the draw pass, reusing `shaders/mesh.wgsl` unchanged.
- Entry points: `render_cylinder_compute_offscreen(desc, tess, face_id, cam, opts)` and `render_cylinder_compute_screen_lod(desc, face_id, cam, opts, target_px)` (the latter derives `n_u` per view; `DEFAULT_TARGET_PX = 0.5` px).

LOD math (`screen_space_tess_factor`): projected pixel radius `r_px = r * (H/2) / (d * tan(fov_y/2))` where `d` is view-space depth (`view_dir . (center - eye)`). Chord error bound `r_px * (1 - cos(pi/n_u)) <= target_px` gives `n_u = ceil(pi / acos(1 - clamp(target_px / r_px, 0, 2)))`. Edge cases: camera engulfed (`r_px` infinite) or a non-finite/non-positive `target_px` budget yields `MAX_TESS` (a budget that cannot bound anything falls back to max detail); behind camera or NaN `r_px` yields the minimum (3); fov is clamped into `(0, pi)`. `n_v = 1` because a cylinder's lateral face is ruled.

Verify silhouette smoothness scaling with the pattern in `tests/compute_mesh_lod.rs`: render at two distances, assert the closer render used more angular subdivisions and a smoother `horizontal_profile`.

## 4. Viewer

```rust
pub fn view_solid(topo: &Topology, solid: SolidId, opts: &ViewOpts) -> Result<(), RenderError>
```

Blocks until the window closes. `ViewOpts::new(title)`, default 1024x768. Needs a surface, so it cannot run headlessly; anything assertable belongs in the offscreen path. Face picking reads the same `R32Uint` id target under the cursor and maps back to a kernel `FaceId`.

Run: `cargo run -p brepkit-render --example viewer --features window`.

## 5. Sandbox environment fact sheet (proven here, not code)

- `DISPLAY=:0` is a live X display visible on the user's desktop; `WAYLAND_DISPLAY` is also set. The machine has a discrete NVIDIA GPU (`nvidia-smi` works).
- winit 0.30 prefers Wayland when `WAYLAND_DISPLAY` is set; `WINIT_UNIX_BACKEND` is ignored (removed in winit 0.30). A native Wayland window cannot be captured here (`grim` is absent). Fix: `env -u WAYLAND_DISPLAY DISPLAY=:0 <cmd>` forces X11/XWayland, which is both visible and capturable.
- Capture tools present: `import` and `xwininfo` (ImageMagick / X11 utils). Absent: `grim`.
- Foreground `sleep` is blocked in the agent shell (dies with exit 144, even inside a backgrounded compound command). Run the viewer with the Bash tool's `run_in_background`, then poll for the window in later tool calls.
- `pkill -f <pattern>` matches your own command line; if the pattern appears in your own invocation you kill yourself (exit 144). Match the binary path (`pkill -f 'target/debug/examples/viewer'`), never a bare word like `viewer` from a command containing it.

## 6. Symptom-to-cause table

| Symptom | Likely cause | Check |
|---|---|---|
| Test skipped / `probe_adapter()` is `None` | No GPU and no lavapipe in this environment | expected in bare CI; not a code bug |
| Offscreen image is all background | Camera not looking at the solid, or solid empty/degenerate | print the solid AABB; check `id_silhouette_bbox` is `None`; verify the solid with the `solid-verification` skill |
| Render OK but `face_id_at` gives unexpected ids | Off-by-one: buffer stores `FaceId.index() + 1` | subtract 1 before comparing to `FaceId.index()` |
| Pixel gaps along a cylinder seam in compute renders | Seam vertices not bitwise-identical at u=0 and u=2pi | `interior_holes` > 0; see `compute_mesh_seam_is_watertight` in `tests/compute_mesh_render.rs` |
| Compute render differs from CPU render | Compare silhouettes, not pixels: shading may differ legitimately | `horizontal_profile` diff, pattern in `compute_mesh_matches_cpu_silhouette` |
| `extract_cylinder_descriptor` errors | Face is not `FaceSurface::Cylinder` (may have degraded to NURBS upstream) | inspect the face surface type; see the `analytic-preservation` skill |
| Far-from-origin model shimmers or distorts | RTC path bypassed (raw f32 absolute positions) | positions must be center-relative; `view_proj_rtc` in `src/camera.rs` |
| Viewer window never appears in `xwininfo` | Still compiling; or it opened as native Wayland | wait for the build; re-launch with `env -u WAYLAND_DISPLAY` |
| `import` captures a black or wrong image | Wrong window id, or window unmapped | re-run `xwininfo -root -tree`, grep for `brepkit viewer` |
| Viewer process refuses to die | Self-matching `pkill` killed your shell instead | kill by `target/debug/examples/viewer` path |
| Render looks perfect but downstream consumers see holes | Rendering does not prove watertightness | `is_watertight` / `boundary_edge_count` in `crates/operations/src/tessellate/mesh_ops.rs` |

## 7. Glossary

- **face-id buffer**: per-pixel `R32Uint` target storing `FaceId.index() + 1` (0 = background); maps pixels back to kernel topology for picking and tests.
- **RTC**: render-relative-to-center; f32 vertices relative to the f64 AABB center, center folded into the view matrix on CPU.
- **quadric**: second-degree surface (cylinder, cone, sphere; torus grouped loosely); the compute mesher evaluates these parametrically on GPU.
- **deflection**: linear chord tolerance for CPU tessellation; smaller = finer; render default 0.05.
- **seam**: the u=0 / u=2pi closure line of a periodic surface; both sides must emit identical vertices or pixels leak.
- **lavapipe**: Mesa's software Vulkan implementation; the adapter fallback that lets offscreen tests run without hardware.
- **watertight / 2-manifold**: every mesh edge shared by exactly two triangles; orthogonal to "renders correctly".

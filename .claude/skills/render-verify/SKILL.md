---
name: render-verify
description: Use when working with brepkit-render (offscreen GPU render, face-id buffer, interactive viewer, compute mesher, screen-space LOD) or when visually verifying a solid in this sandbox, including capturing a live viewer window headlessly, writing image-based render tests, or deciding between render checks and mesh watertightness checks.
---

# render-verify: GPU rendering and visual verification

## When to use

- You changed geometry, tessellation, booleans, or the render crate and need to SEE the result.
- You need an automated image assertion (test that a render is non-blank, silhouette shape, seam holes).
- You need to verify interactive behavior (orbit, zoom LOD, click-to-pick) in the live viewer.
- You are tempted to conclude "renders fine, so the mesh is correct" (do not; see anti-patterns).

For mesh manifoldness itself see the `tessellation` and `solid-verification` skills. For why analytic surface types survive operations, see `analytic-preservation`.

## Quick reference

| Goal | Tool | How |
|---|---|---|
| Automated image assertion | `render_solid_offscreen` | headless, no window; assert on `id_buffer` |
| Compute-mesher / LOD assertion | `render_cylinder_compute_offscreen`, `render_cylinder_compute_screen_lod` | same, cylinder faces only |
| Eyeball a solid | offscreen render, save PNG, Read it | fastest, no window needed |
| Interactive behavior (pick, orbit, live LOD) | viewer example + X11 capture recipe | see Procedure B |
| Watertightness | `is_watertight`, `boundary_edge_count`, `non_manifold_edge_count` in `crates/operations/src/tessellate/mesh_ops.rs` | NOT a render question |

```bash
cargo test -p brepkit-render                                    # offscreen render tests
cargo run -p brepkit-render --example viewer --features window  # live viewer (needs display)
```

Crate map, full API signatures, LOD math, and the environment fact sheet: [reference.md](reference.md).

## Procedure A: offscreen verification (default path)

1. Gate on an adapter. `brepkit_render::probe_adapter()` returns `Some("Vulkan / NVIDIA ... (DiscreteGpu)")` style strings, or a software fallback (lavapipe) when no real GPU exists, or `None`. Tests must early-return on `None`, never fail. Copy the `let Some(_) = probe_adapter() else { return; }` pattern from `crates/render/tests/offscreen_render.rs`.
2. Build the scene, render:
   ```rust
   let out = render_solid_offscreen(&topo, solid, &camera, &RenderOpts::new(800, 600))?;
   out.color.save("/path/render.png")?;
   ```
3. Checkpoint: Read the PNG. Expect a shaded solid on a dark gray background (default `[0.11, 0.12, 0.14, 1.0]`) with dark edge lines. All-background image: see reference.md, symptom table.
4. Assert on the face-id buffer, not color bytes. `out.face_id_at(x, y)` returns `None` for background or out-of-bounds; the stored `id_buffer` value is `FaceId.index() + 1`, `0` means background. Test helpers to copy from `crates/render/tests/`: `non_background_pixels`, `id_silhouette_bbox`, `horizontal_profile`, `interior_holes`.

## Procedure B: live viewer verification (proven sandbox recipe)

This sandbox has a real GPU and a live display `:0`. These steps are environment facts proven here, not brepkit code behavior.

1. Launch, forcing X11. winit 0.30 prefers Wayland when `WAYLAND_DISPLAY` is set and a native Wayland window is not capturable here (`grim` is absent, `WINIT_UNIX_BACKEND` is ignored by winit 0.30). Run via the Bash tool with `run_in_background: true` (foreground `sleep` is blocked in the agent shell, so never `sleep && capture` in one command):
   ```bash
   env -u WAYLAND_DISPLAY DISPLAY=:0 cargo run -p brepkit-render --example viewer --features window
   ```
2. Checkpoint: find the window (poll across tool calls until it appears; first launch includes compile time):
   ```bash
   DISPLAY=:0 xwininfo -root -tree | grep -i 'brepkit viewer'
   ```
   Expect a line like `0x1c00007 "brepkit viewer ...": ... 1024x768+...`. No match after the build finished: see reference.md, symptom table.
3. Capture with ImageMagick and Read the PNG. Use the window id from step 2, and write into your session scratchpad directory if the system prompt lists one (`${TMPDIR:-/tmp}` works as a fallback):
   ```bash
   DISPLAY=:0 import -window 0x1c00007 "${TMPDIR:-/tmp}/viewer.png"
   ```
4. Kill it by binary path, never by a word that appears in your own command line (`pkill -f` matches your own cmdline and self-kills with exit 144):
   ```bash
   pkill -f 'target/debug/examples/viewer'
   ```

Viewer controls: left-drag orbit, right-drag or shift+left-drag pan, scroll zoom, plain left click picks a face (tint via the id buffer, click again to clear).

## The architectural bet

brepkit preserves exact analytic surfaces (plane, cylinder, cone, sphere, torus) through operations. The render bet: ship the surface parameters to the GPU and evaluate the mesh there in a compute pass at a view-dependent LOD, instead of CPU-tessellating and uploading triangles. WebGPU has no tessellation or mesh shaders, hence the compute pass. This is why analytic preservation matters to rendering: a face degraded to NURBS cannot take this path. Currently shipped for cylinder faces only; module doc of `crates/render/src/compute_mesh.rs` is the source of truth.

## Anti-patterns: what NOT to conclude

- "It renders correctly, so the mesh is watertight." False. The rasterizer hides hairline gaps; a mesh with boundary edges can look perfect. Use `mesh_ops` counters.
- "The mesh is watertight, so the render is correct." False. Watertight meshes can render with bad normals or wrong face ids. Check the image and `id_buffer` too.
- "No adapter in CI means the render code is broken." No. Gate on `probe_adapter()` and skip.
- "The compute mesher renders wrong, so tessellation is broken." They are separate paths: `render_solid_offscreen` uses CPU tessellation; the compute path is `render_cylinder_compute_*` only.
- "The viewer would prove this" when an offscreen render answers the question. Offscreen first; the window is only for interaction and live LOD.
- Do not document or rely on cone/sphere/torus compute meshing, quadric ray-casting, or a wasm render path. Roadmap, not shipped.
- Do not assert on color bytes when the id buffer answers the question; shading changes with lighting defaults, ids do not.
- Do not compare render performance against other kernels here; that lives in the `parity-benchmarking` skill via the brepjs harness.

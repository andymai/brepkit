# brepkit-render

[![crates.io](https://img.shields.io/crates/v/brepkit-render)](https://crates.io/crates/brepkit-render) [![docs.rs](https://img.shields.io/docsrs/brepkit-render)](https://docs.rs/brepkit-render)

Offscreen GPU rendering of B-Rep solids to an image plus a face-id buffer.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L4.** A leaf: nothing in the workspace depends on it.

Full API reference on [docs.rs](https://docs.rs/brepkit-render).

Renders a solid with wgpu to a color image and a parallel buffer holding
the face id at every pixel, which makes click-to-pick and headless visual
verification possible without a windowing system.

Includes a GPU compute mesher for analytic quadrics and, behind the `window`
feature, an interactive viewer with an orbit camera.

No core operation requires this crate. It exists for tests and headless
verification.

## Stability

Experimental. The API is real and tested, but its shape is still open and can
change without a deprecation cycle. See
[STABILITY.md](https://github.com/andymai/brepkit/blob/main/STABILITY.md).

## License

[AGPL-3.0-only](https://github.com/andymai/brepkit/blob/main/LICENSE), or a
[commercial license](https://github.com/andymai/brepkit/blob/main/COMMERCIAL-LICENSE.md)
for proprietary use.

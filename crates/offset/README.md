# brepkit-offset

[![crates.io](https://img.shields.io/crates/v/brepkit-offset)](https://crates.io/crates/brepkit-offset) [![docs.rs](https://img.shields.io/docsrs/brepkit-offset)](https://docs.rs/brepkit-offset)

Solid offset and thickening via global face-face intersection.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Depends on `brepkit-math`, `brepkit-topology`, and `brepkit-geometry`.

Full API reference on [docs.rs](https://docs.rs/brepkit-offset).

Offsets every face of a solid, intersects the offset faces against each
other in 3D, splits edges at the intersections, rebuilds the wire loops, and
fills convex corners with arc joints (pipes along edges, spherical caps at
vertices). Self-intersections in the result are removed with a boolean pass.

## Stability

Internal. This crate is published so that `brepkit-operations` resolves from
crates.io, not because its API is meant to be called directly. Treat it as
private and expect breakage on any release. Depend on
[`brepkit-operations`](https://crates.io/crates/brepkit-operations) instead.
See [STABILITY.md](https://github.com/andymai/brepkit/blob/main/STABILITY.md).

## License

[AGPL-3.0-only](https://github.com/andymai/brepkit/blob/main/LICENSE), or a
[commercial license](https://github.com/andymai/brepkit/blob/main/COMMERCIAL-LICENSE.md)
for proprietary use.

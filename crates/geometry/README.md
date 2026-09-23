# brepkit-geometry

[![crates.io](https://img.shields.io/crates/v/brepkit-geometry)](https://crates.io/crates/brepkit-geometry) [![docs.rs](https://img.shields.io/docsrs/brepkit-geometry)](https://docs.rs/brepkit-geometry)

Curve and surface sampling, extrema, and analytic/NURBS conversion.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L1.** Depends only on `brepkit-math`.

Full API reference on [docs.rs](https://docs.rs/brepkit-geometry), including how to choose a sampler.

Three subsystems:

- **sampling**: uniform, deflection-adaptive, arc-length-uniform, and curvature-adaptive curve sampling, plus surface grids.
- **extrema**: point-to-curve projection, curve-to-curve and point-to-surface distance, segment-segment distance, and a Lipschitz global optimizer.
- **convert**: analytic geometry to NURBS, and recognition of NURBS back into analytic curves and surfaces.

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

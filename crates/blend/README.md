# brepkit-blend

[![crates.io](https://img.shields.io/crates/v/brepkit-blend)](https://crates.io/crates/brepkit-blend) [![docs.rs](https://img.shields.io/docsrs/brepkit-blend)](https://docs.rs/brepkit-blend)

Walking-based fillet and chamfer engine with constant, variable, and custom radius laws.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Depends on `brepkit-math` and `brepkit-topology`.

Full API reference on [docs.rs](https://docs.rs/brepkit-blend).

Builds a blend by walking a rolling-ball cross-section along an edge chain
with Newton-Raphson, solving for contact points at each station, then trimming
the adjacent faces along the resulting contact curves.

Analytic fast paths (plane-plane, plane-cylinder, and others) skip the walk and
emit an exact cylindrical or conical stripe, so a filleted straight edge between
two planes keeps an exact surface rather than a NURBS approximation of one.

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

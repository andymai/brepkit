# brepkit-algo

[![crates.io](https://img.shields.io/crates/v/brepkit-algo)](https://crates.io/crates/brepkit-algo) [![docs.rs](https://img.shields.io/docsrs/brepkit-algo)](https://docs.rs/brepkit-algo)

The General Fuse boolean engine: pave filler, face classification, solid assembly.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Depends on `brepkit-math` and `brepkit-topology`.

Full API reference on [docs.rs](https://docs.rs/brepkit-algo).

The exact boolean path. Intersects two shapes into a shared set of split
faces and edges (the pave filler, running VV/VE/EE/VF/EF/FF phases), classifies
each fragment against the other solid, and reassembles the selected fragments
into a result shell.

Analytic surfaces survive the operation: a cylinder cut by a plane stays a
cylinder, which is what keeps face counts flat across chained booleans.

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

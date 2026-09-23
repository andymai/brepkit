# brepkit-check

[![crates.io](https://img.shields.io/crates/v/brepkit-check)](https://crates.io/crates/brepkit-check) [![docs.rs](https://img.shields.io/docsrs/brepkit-check)](https://docs.rs/brepkit-check)

Point classification, validation, mass properties, and distance queries.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Depends on `brepkit-math`, `brepkit-topology`, and `brepkit-geometry`.

Full API reference on [docs.rs](https://docs.rs/brepkit-check).

Interrogation rather than modification:

- **classify**: point-in-solid by ray casting or generalized winding number.
- **validate**: wire, shell, solid, vertex, edge, and face checks with per-check severity.
- **properties**: volume, surface area, center of mass, and bounding box, using closed-form formulas for analytic primitives and Gauss integration otherwise.
- **distance**: point-to-surface, edge-to-edge, point-to-solid, and solid-to-solid.

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

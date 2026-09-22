# brepkit-heal

[![crates.io](https://img.shields.io/crates/v/brepkit-heal)](https://crates.io/crates/brepkit-heal) [![docs.rs](https://img.shields.io/docsrs/brepkit-heal)](https://docs.rs/brepkit-heal)

Shape healing: analysis, fixing, upgrading, and sewing for imported B-Rep models.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Depends on `brepkit-math`, `brepkit-topology`, and `brepkit-geometry`.

Full API reference on [docs.rs](https://docs.rs/brepkit-heal).

Repairs the defects that real CAD files arrive with: wires out of order or
with gaps, edges whose 3D curve and PCurve disagree, faces with inconsistent
orientation, shells with free edges.

Analysis and fixing are separate. Analysis reports what is wrong without
touching the model; fixing applies a configurable set of `FixMode` decisions
through a reshape log. The `upgrade` module goes further, merging same-domain
faces, splitting curves at continuity breaks, and sewing shells.

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

# brepkit-topology

[![crates.io](https://img.shields.io/crates/v/brepkit-topology)](https://crates.io/crates/brepkit-topology) [![docs.rs](https://img.shields.io/docsrs/brepkit-topology)](https://docs.rs/brepkit-topology)

Arena-allocated B-Rep data structures: vertex, edge, wire, face, shell, solid.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L1.** Depends only on `brepkit-math`.

Full API reference on [docs.rs](https://docs.rs/brepkit-topology), including the arena borrow pattern, the surface enums, and the inner-shell trap.

Owns the `Topology` arena that every brepkit operation takes as `&mut`.
Entities are stored in typed arenas and referenced by `Id<T>` handles rather
than pointers, which keeps traversal cache-friendly, avoids reference counting,
and makes ownership unambiguous.

Also here: the edge-to-face adjacency index, the shape explorer, the PCurve
registry, and topological validation.

## Example

```rust
use brepkit_topology::Topology;
use brepkit_topology::vertex::Vertex;
use brepkit_math::vec::Point3;

let mut topo = Topology::new();
let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 2.0, 3.0), 1e-7));

// Entities are owned by the arena and reached through their handle.
assert_eq!(topo.vertex(v)?.point().x(), 1.0);
```

## Stability

Part of brepkit's consumer surface: this crate is meant to be named directly in
a downstream `Cargo.toml`. Breaking changes are possible but not routine, and
land with a CHANGELOG entry. See
[STABILITY.md](https://github.com/andymai/brepkit/blob/main/STABILITY.md).

## License

[AGPL-3.0-only](https://github.com/andymai/brepkit/blob/main/LICENSE), or a
[commercial license](https://github.com/andymai/brepkit/blob/main/COMMERCIAL-LICENSE.md)
for proprietary use.

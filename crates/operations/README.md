# brepkit-operations

[![crates.io](https://img.shields.io/crates/v/brepkit-operations)](https://crates.io/crates/brepkit-operations) [![docs.rs](https://img.shields.io/docsrs/brepkit-operations)](https://docs.rs/brepkit-operations)

CAD modeling operations: booleans, fillets, sweeps, offsets, measurement, tessellation.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L3.** The entry point for Rust consumers.

Full API reference on [docs.rs](https://docs.rs/brepkit-operations), including the exactness contract, fallback detection, and how to verify a result.

The crate most projects want. It pulls in the geometry, topology, and
engine crates it needs and presents them as operations:

| Family | What it covers |
|--------|----------------|
| Creation | box, cylinder, cone, sphere, torus, convex hull, Minkowski sum |
| Sweeps | extrude, revolve, sweep, loft, pipe, helix |
| Booleans | union, cut, intersect, compound cut, batch fuse |
| Blends | fillet and chamfer, constant and variable radius |
| Surface | shell, draft, thicken, offset, fill, sew, untrim, section, split |
| Transform | transform, copy, mirror, linear and circular pattern |
| Analysis | measure, distance, classify, validate, query, feature recognition |
| Output | tessellation to triangle meshes |

Modeling operations take `&mut Topology` and return a typed handle.
Interrogation (measure, distance, classify, validate, query, tessellate)
borrows it as `&` and returns a value. Fallible work returns a `Result`
rather than panicking.

## Example

```rust
use brepkit_topology::Topology;
use brepkit_operations::primitives::{make_box, make_cylinder};
use brepkit_operations::boolean::{boolean, BooleanOp};
use brepkit_operations::measure::solid_volume;

let mut topo = Topology::new();

// Primitives are anchored at the origin, so this cylinder rounds off the
// block's corner. Use `transform_solid` to place it somewhere else.
let block = make_box(&mut topo, 30.0, 20.0, 10.0)?;
let cutter = make_cylinder(&mut topo, 5.0, 15.0)?;
let notched = boolean(&mut topo, BooleanOp::Cut, block, cutter)?;

// The cut removes a quarter-cylinder of height 10 from the corner.
let expected = 30.0 * 20.0 * 10.0 - 0.25 * std::f64::consts::PI * 25.0 * 10.0;
let volume = solid_volume(&topo, notched, 0.01)?;
assert!((volume - expected).abs() / expected < 1e-3);
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

# brepkit-io

[![crates.io](https://img.shields.io/crates/v/brepkit-io)](https://crates.io/crates/brepkit-io) [![docs.rs](https://img.shields.io/docsrs/brepkit-io)](https://docs.rs/brepkit-io)

Data exchange: STEP, IGES, STL, 3MF, OBJ, PLY, and glTF import and export.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L3.** Depends on `brepkit-math`, `brepkit-topology`, and `brepkit-operations`.

Full API reference on [docs.rs](https://docs.rs/brepkit-io), including healing on import, tolerance guidance, and the round-trip rules.

| Format | Type | Import | Export |
|--------|------|--------|--------|
| STEP | B-Rep | yes | yes |
| STL | Mesh | yes | yes |
| 3MF | Mesh | yes | yes |
| OBJ | Mesh | yes | yes |
| PLY | Mesh | yes | yes |
| glTF (`.glb`) | Mesh | yes | yes |
| IGES | B-Rep | preview | lossy |

STEP is the lossless path. Analytic surfaces (plane, cylinder, cone, sphere,
torus) are written as native STEP surface entities rather than tessellated, and
read back as the same surface types. NURBS surfaces and line, circle, ellipse,
and NURBS edges are preserved too.

Mesh formats export tessellated triangles. IGES is experimental: export skips
analytic surfaces and approximates circular edges as polylines, and import
reconstructs planar placeholder faces only. Use STEP for B-Rep exchange.

## Example

```rust
use brepkit_topology::Topology;
use brepkit_operations::primitives::make_cylinder;
use brepkit_io::step::{read_step, write_step};

let mut topo = Topology::new();
let cylinder = make_cylinder(&mut topo, 5.0, 20.0)?;

let step = write_step(&topo, &[cylinder])?;

// Round-trips as a cylinder, not as a tessellation of one.
let mut reloaded = Topology::new();
let solids = read_step(&step, &mut reloaded)?;
assert_eq!(solids.len(), 1);
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

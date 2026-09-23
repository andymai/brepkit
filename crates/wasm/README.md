# brepkit-wasm

[![crates.io](https://img.shields.io/crates/v/brepkit-wasm)](https://crates.io/crates/brepkit-wasm) [![docs.rs](https://img.shields.io/docsrs/brepkit-wasm)](https://docs.rs/brepkit-wasm) [![npm](https://img.shields.io/npm/v/brepkit-wasm)](https://www.npmjs.com/package/brepkit-wasm)

WebAssembly bindings for brepkit: browser-native B-Rep solid modeling.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L4.** Depends on `brepkit-operations`, `brepkit-io` (optional),
`brepkit-algo`, `brepkit-check`, `brepkit-geometry`, `brepkit-heal`,
`brepkit-math`, and `brepkit-topology`. It does not depend on
`brepkit-render`, which is a separate L4 leaf.

Full API reference on [docs.rs](https://docs.rs/brepkit-wasm).

Exposes the kernel to JavaScript through wasm-bindgen as a `BrepKernel`
object, with batch execution and checkpoint/restore.

Most JavaScript users want the npm package rather than this crate:

```bash
npm install brepkit-wasm
```

```js
import { BrepKernel } from 'brepkit-wasm';

const kernel = new BrepKernel();
const block = kernel.makeBox(30, 20, 10);
const cutter = kernel.makeCylinder(5, 15);
const notched = kernel.cut(block, cutter);
```

For a higher-level TypeScript API built on this package, see
[brepjs](https://github.com/andymai/brepjs) and its documentation at
[brepjs.dev](https://brepjs.dev).

The `io` feature (on by default) pulls in `brepkit-io`. Build with
`--no-default-features` for a smaller binary without import and export.

## Stability

Part of brepkit's consumer surface: this crate is meant to be named directly in
a downstream `Cargo.toml`. Breaking changes are possible but not routine, and
land with a CHANGELOG entry. See
[STABILITY.md](https://github.com/andymai/brepkit/blob/main/STABILITY.md).

## License

[AGPL-3.0-only](https://github.com/andymai/brepkit/blob/main/LICENSE), or a
[commercial license](https://github.com/andymai/brepkit/blob/main/COMMERCIAL-LICENSE.md)
for proprietary use.

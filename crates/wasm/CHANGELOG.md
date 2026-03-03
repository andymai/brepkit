# Changelog

## [0.3.0](https://github.com/andymai/brepkit/compare/v0.2.0...v0.3.0) (2026-03-03)


### Features

* **fillet:** rolling-ball fillet with G1-continuous NURBS blend surfaces ([#11](https://github.com/andymai/brepkit/issues/11)) ([098966c](https://github.com/andymai/brepkit/commit/098966cd868d203b1131ea33897da9c198339e70))
* **operations:** exact analytic booleans preserving surface types ([e9e4a40](https://github.com/andymai/brepkit/commit/e9e4a40eeabb5f997455079212b186d61fe42705))
* **operations:** exact analytic booleans preserving surface types ([b110646](https://github.com/andymai/brepkit/commit/b11064666fcdf2fbc81aecdb2e563d27de1acafe))
* **sweep,wasm:** smooth NURBS sweep + WASM bindings for loftSmooth/sweepSmooth ([#15](https://github.com/andymai/brepkit/issues/15)) ([9741de3](https://github.com/andymai/brepkit/commit/9741de3023b12c1a5075fc373aa0672e4f50d8a6))
* **tessellate:** watertight solid tessellation with shared edge vertices ([#9](https://github.com/andymai/brepkit/issues/9)) ([25e2a17](https://github.com/andymai/brepkit/commit/25e2a176978b0f3fc8c50c6713b39a18ad244859))

## [0.2.0](https://github.com/andymai/brepkit/compare/v0.1.0...v0.2.0) (2026-03-03)


### Features

* add Phase 1 foundation for OCCT feature parity ([41aca1d](https://github.com/andymai/brepkit/commit/41aca1df884e4940ab1b64cbfc20dc7142a1f69f))
* initialize brepkit workspace ([e516477](https://github.com/andymai/brepkit/commit/e516477b9823748262e681c4679cbc72a9b2ff73))
* **io,wasm:** add STL mesh import and WASM bindings for IO ([347fb69](https://github.com/andymai/brepkit/commit/347fb6901aa49dbfcef7de2b77552367eacc6ca5))
* **io,wasm:** implement 3MF export with tessellation pipeline ([0557961](https://github.com/andymai/brepkit/commit/0557961288ee4451e813c7b5a139e612311ed826))
* **io:** add glTF 2.0 binary (.glb) writer ([e292970](https://github.com/andymai/brepkit/commit/e292970411a5c095f21138065121d4870aa4e501))
* **io:** add glTF binary (.glb) reader ([e1c029e](https://github.com/andymai/brepkit/commit/e1c029ec717b430bbbaf0d757dfa51e3740c87ed))
* **io:** add OBJ (Wavefront) reader and writer ([f944629](https://github.com/andymai/brepkit/commit/f944629745d5a47ba81b8d773163374c22ebca9c))
* **io:** add PLY reader and writer (ASCII + binary) ([4c96f6a](https://github.com/andymai/brepkit/commit/4c96f6aa85a92e97a608badc1291bc4b858e9bfa))
* **operations,wasm:** add edge/wire/face length measurement ([f858e83](https://github.com/andymai/brepkit/commit/f858e8336a13a8a25984cde9200eda3c0f540c84))
* **operations,wasm:** implement chamfer and expose boolean bindings ([469e437](https://github.com/andymai/brepkit/commit/469e4371e4793359c7cfffc082cc7d3e21c64b3b))
* **operations,wasm:** implement revolve operation with NURBS tessellation ([a34bb1c](https://github.com/andymai/brepkit/commit/a34bb1c5ffc1776207390a505132f03b03c87d67))
* **operations,wasm:** implement sweep operation along NURBS paths ([f5c9417](https://github.com/andymai/brepkit/commit/f5c9417fec5a94006cdd340b25ebe8b2659d4642))
* **operations:** add evolution tracking for boolean operations ([#4](https://github.com/andymai/brepkit/issues/4)) ([3c2ced9](https://github.com/andymai/brepkit/commit/3c2ced9e59ebc80bff4e275b28e159041a66d7e3))
* **operations:** add point-in-solid classification ([ef08826](https://github.com/andymai/brepkit/commit/ef08826ff83f9e69d026894cdf8d4cfe0a470a4b))
* performance optimizations — packed mesh transfer, fused copy+transform, analytic boolean fast path ([fd1ff7b](https://github.com/andymai/brepkit/commit/fd1ff7b554e1f48da0d97ea486630bbdb7fafe4f))
* **wasm:** add BrepKernel WASM bindings for JS API ([b399c02](https://github.com/andymai/brepkit/commit/b399c027662b02c05751abb870b4d95df917e3c1))
* **wasm:** add distance, sewing WASM bindings ([4f6ba5f](https://github.com/andymai/brepkit/commit/4f6ba5f471977fa113edfed3a393541d756e9a41))
* **wasm:** add semantic APIs for shape orientation and reversal ([#5](https://github.com/andymai/brepkit/issues/5)) ([d6561da](https://github.com/andymai/brepkit/commit/d6561dad4c6c95fc2db136f2815fba0379a30895))
* **wasm:** add split, draft, and pipe WASM bindings ([7a36e1b](https://github.com/andymai/brepkit/commit/7a36e1b986c5675ca3d3666d07c66b311fb40341))
* **wasm:** add STL export, copy, mirror, and pattern bindings ([7c1e43d](https://github.com/andymai/brepkit/commit/7c1e43df4bdaeb38d997f7ab9ef6dbe6fdb88442))
* **wasm:** add topology query bindings; fix review issues ([d05f03e](https://github.com/andymai/brepkit/commit/d05f03e3bb66bc7397784b01391a1b76eaa0fcdd))
* **wasm:** expose primitives, section, loft, shell, chamfer, fillet bindings ([51101f5](https://github.com/andymai/brepkit/commit/51101f5b2330055e314ac76dee4a940562659b2f))
* **wasm:** topology traversal exports for compounds, shells, wires ([#1](https://github.com/andymai/brepkit/issues/1)) ([ed38d5d](https://github.com/andymai/brepkit/commit/ed38d5d1955fd936c9cded9f03cc7596461fa4b5))


### Bug Fixes

* address code review issues; add WASM bindings for IGES/helix ([2be8ba0](https://github.com/andymai/brepkit/commit/2be8ba0932123b841946f034ebb74fa879eff5a5))

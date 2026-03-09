# Changelog

## [0.8.1](https://github.com/andymai/brepkit/compare/v0.8.0...v0.8.1) (2026-03-09)


### Performance Improvements

* fix algorithmic bottlenecks — test suite 370s → 9s ([#125](https://github.com/andymai/brepkit/issues/125)) ([27ae79f](https://github.com/andymai/brepkit/commit/27ae79f2eb9ac5bec2d36f36bdce85ecd04bc774))

## [0.8.0](https://github.com/andymai/brepkit/compare/v0.7.13...v0.8.0) (2026-03-09)


### Features

* add relative tolerance for scale-aware comparisons ([#122](https://github.com/andymai/brepkit/issues/122)) ([6c748cc](https://github.com/andymai/brepkit/commit/6c748cc48cab5a3542793c24c97afb7a59b31e38))

## [0.7.13](https://github.com/andymai/brepkit/compare/v0.7.12...v0.7.13) (2026-03-09)


### Bug Fixes

* SSI branch detection and offset self-intersection trimming ([#120](https://github.com/andymai/brepkit/issues/120)) ([e287fd0](https://github.com/andymai/brepkit/commit/e287fd08eafad9da23ebf4b8e1bf47f2a0458e88))

## [0.7.12](https://github.com/andymai/brepkit/compare/v0.7.11...v0.7.12) (2026-03-09)


### Bug Fixes

* architecture improvements — curved fillets, NURBS boolean, SoS predicates ([#114](https://github.com/andymai/brepkit/issues/114)) ([5fdcd58](https://github.com/andymai/brepkit/commit/5fdcd58be0f1809fcb2d54430fc3aae7bb073927))

## [0.7.11](https://github.com/andymai/brepkit/compare/v0.7.10...v0.7.11) (2026-03-09)


### Bug Fixes

* Sprint 8 — SSI perf, adaptive offsets, G1 fillets, algebraic SSI ([#115](https://github.com/andymai/brepkit/issues/115)) ([20b9943](https://github.com/andymai/brepkit/commit/20b99435f5735426291dbc8145af5ececd1e40f5))

## [0.7.10](https://github.com/andymai/brepkit/compare/v0.7.9...v0.7.10) (2026-03-09)


### Bug Fixes

* deep robustness — polygon clipping, Newton singularity, fat line signs, CSI ([#113](https://github.com/andymai/brepkit/issues/113)) ([2337aab](https://github.com/andymai/brepkit/commit/2337aab2e2c87e782dae02dc58f1c5632d6d8b6e))

## [0.7.9](https://github.com/andymai/brepkit/compare/v0.7.8...v0.7.9) (2026-03-09)


### Bug Fixes

* boolean robustness — multi-ray classification, coplanar handling, exact predicates ([#108](https://github.com/andymai/brepkit/issues/108)) ([82d45c8](https://github.com/andymai/brepkit/commit/82d45c81773cd0a0b232713a83c4fc111a595f31))
* fillet robustness — edge curves, rational arcs, validation, spherical blends ([#112](https://github.com/andymai/brepkit/issues/112)) ([d69391e](https://github.com/andymai/brepkit/commit/d69391efa5804c0a1fbfec7c8f344b9fc790facb))
* NURBS intersection foundation — periodic surfaces, 4D Newton, overlap detection ([#109](https://github.com/andymai/brepkit/issues/109)) ([82c3b95](https://github.com/andymai/brepkit/commit/82c3b95d3e57a7193875334dd895989e1d07ccad))
* tessellation correctness — concave holes, analytic u_range, CDT, PCurves ([#110](https://github.com/andymai/brepkit/issues/110)) ([5ecd91e](https://github.com/andymai/brepkit/commit/5ecd91e2a22a33635abf40d1a64dc2c912866806))

## [0.7.8](https://github.com/andymai/brepkit/compare/v0.7.7...v0.7.8) (2026-03-09)


### Bug Fixes

* consolidate boolean edges and prevent fillet panic corruption ([#106](https://github.com/andymai/brepkit/issues/106)) ([7c5588a](https://github.com/andymai/brepkit/commit/7c5588a2660d938ca4a347c3114f6d146faa3f0b))

## [0.7.7](https://github.com/andymai/brepkit/compare/v0.7.6...v0.7.7) (2026-03-09)


### Bug Fixes

* Tier 1 critical fixes — SSI domains, STEP I/O, extrude surfaces ([#104](https://github.com/andymai/brepkit/issues/104)) ([14069fd](https://github.com/andymai/brepkit/commit/14069fdd69cff3d272c8fb68abc24dd0ffe6f911))

## [0.7.6](https://github.com/andymai/brepkit/compare/v0.7.5...v0.7.6) (2026-03-09)


### Performance Improvements

* algorithmic optimizations for booleans, CDT, and tessellation ([#102](https://github.com/andymai/brepkit/issues/102)) ([a7383e8](https://github.com/andymai/brepkit/commit/a7383e82b3553c989e0c4c1fef118b10d36a031c))

## [0.7.5](https://github.com/andymai/brepkit/compare/v0.7.4...v0.7.5) (2026-03-09)


### Performance Improvements

* preserve analytic surfaces through sequential booleans ([#98](https://github.com/andymai/brepkit/issues/98)) ([7923932](https://github.com/andymai/brepkit/commit/7923932149a29acd58536cdd82000d35dd0c8d08))

## [0.7.4](https://github.com/andymai/brepkit/compare/v0.7.3...v0.7.4) (2026-03-08)


### Bug Fixes

* fillet tolerates non-manifold edges from boolean results ([#96](https://github.com/andymai/brepkit/issues/96)) ([b64caa8](https://github.com/andymai/brepkit/commit/b64caa81b93e023a3121f59a10682c6fef73ca78))

## [0.7.3](https://github.com/andymai/brepkit/compare/v0.7.2...v0.7.3) (2026-03-08)


### Bug Fixes

* address outstanding PR review comments ([#94](https://github.com/andymai/brepkit/issues/94)) ([483d990](https://github.com/andymai/brepkit/commit/483d990537c5be9ec0c0138976538c5731f1ba47))

## [0.7.2](https://github.com/andymai/brepkit/compare/v0.7.1...v0.7.2) (2026-03-08)


### Bug Fixes

* compute cylinder band normal from surface point, not centroid ([#92](https://github.com/andymai/brepkit/issues/92)) ([24f52ee](https://github.com/andymai/brepkit/commit/24f52ee6703582fda742c00825d7f4ec621b48a1))

## [0.7.1](https://github.com/andymai/brepkit/compare/v0.7.0...v0.7.1) (2026-03-08)


### Bug Fixes

* deduplicate edges in analytic boolean for proper adjacency ([9a09ff7](https://github.com/andymai/brepkit/commit/9a09ff70bf7f94fe63c4bbb1846197c6f389b2f9))

## [0.7.0](https://github.com/andymai/brepkit/compare/v0.6.0...v0.7.0) (2026-03-08)


### Features

* analytic sphere boolean with O(1) classification ([#89](https://github.com/andymai/brepkit/issues/89)) ([327d0f2](https://github.com/andymai/brepkit/commit/327d0f25227e6464ff086be236d1e253feb71d8a))

## [0.6.0](https://github.com/andymai/brepkit/compare/v0.5.3...v0.6.0) (2026-03-08)


### Features

* xtask WASM build pipeline with validation and smoke test ([#81](https://github.com/andymai/brepkit/issues/81)) ([9595615](https://github.com/andymai/brepkit/commit/95956155fd14f3200c9b230a9fa2ef7bbe970ba6))

## [0.5.3](https://github.com/andymai/brepkit/compare/v0.5.2...v0.5.3) (2026-03-08)


### Bug Fixes

* brepjs compatibility fixes across geometry and operations ([#76](https://github.com/andymai/brepkit/issues/76)) ([f17f392](https://github.com/andymai/brepkit/commit/f17f3929b7182ad2a4d689c6b815d9e6225aecf2))

## [0.5.2](https://github.com/andymai/brepkit/compare/v0.5.1...v0.5.2) (2026-03-06)


### Bug Fixes

* address 110 brepjs-wasm test failures across 12 categories ([#74](https://github.com/andymai/brepkit/issues/74)) ([df31ae4](https://github.com/andymai/brepkit/commit/df31ae4f6c1ef4e3346a24804836bc463345ce9d))

## [0.5.1](https://github.com/andymai/brepkit/compare/v0.5.0...v0.5.1) (2026-03-06)


### Bug Fixes

* **wasm:** align edge curve types, fix section, add wire ops ([#71](https://github.com/andymai/brepkit/issues/71)) ([3186285](https://github.com/andymai/brepkit/commit/3186285c8ec880387350894d35f248554e545371))

## [0.5.0](https://github.com/andymai/brepkit/compare/v0.4.3...v0.5.0) (2026-03-05)


### Features

* **operations:** support closed-path sweep ([#68](https://github.com/andymai/brepkit/issues/68)) ([b965c60](https://github.com/andymai/brepkit/commit/b965c60f72135df4ff0ce6e76b270e83f52a8549))

## [0.4.3](https://github.com/andymai/brepkit/compare/v0.4.2...v0.4.3) (2026-03-05)


### Bug Fixes

* **operations:** analytic boolean for contained curves ([#65](https://github.com/andymai/brepkit/issues/65)) ([49a7568](https://github.com/andymai/brepkit/commit/49a7568236ef8e621e2aa495e29250478eaa0e8c))

## [0.4.2](https://github.com/andymai/brepkit/compare/v0.4.1...v0.4.2) (2026-03-05)


### Bug Fixes

* **measure:** analytic volume for sphere, cylinder, cone, torus ([#62](https://github.com/andymai/brepkit/issues/62)) ([368ec48](https://github.com/andymai/brepkit/commit/368ec4873c09285e6973d0070482781275533127))

## [0.4.1](https://github.com/andymai/brepkit/compare/v0.4.0...v0.4.1) (2026-03-04)


### Bug Fixes

* **ci:** use GitHub App token for release-please ([#58](https://github.com/andymai/brepkit/issues/58)) ([462d6c4](https://github.com/andymai/brepkit/commit/462d6c434721f5e4fe8150112a1d00f2e6e53d5f))
* exclude non-code paths from release-please version bumps ([#54](https://github.com/andymai/brepkit/issues/54)) ([bac08ce](https://github.com/andymai/brepkit/commit/bac08ce3a9076ccf98a7a3ec2a0f97c2036a8970))
* **operations:** fix intersect(box, sphere) 3400× perf regression ([#55](https://github.com/andymai/brepkit/issues/55)) ([5fd0fcc](https://github.com/andymai/brepkit/commit/5fd0fcc119be1c6f38d1e8196503799e51428bbd))
* release-please and npm publish configuration ([#52](https://github.com/andymai/brepkit/issues/52)) ([f6726f1](https://github.com/andymai/brepkit/commit/f6726f1beedbef3ab417912535aff09788146742))
* sphere topology + CDT-constrained NURBS tessellation ([#50](https://github.com/andymai/brepkit/issues/50)) ([6c9b953](https://github.com/andymai/brepkit/commit/6c9b953011d73963f244403094753d3ab19c27f4))
* use simple release type for cargo workspace compatibility ([#56](https://github.com/andymai/brepkit/issues/56)) ([3672800](https://github.com/andymai/brepkit/commit/3672800f5e9b61ee28acbc2566e241d9af31fd42))
* **wasm:** use npm-expected repository URL format in Cargo.toml ([#51](https://github.com/andymai/brepkit/issues/51)) ([97ea812](https://github.com/andymai/brepkit/commit/97ea812893b0a0fadd6d388a04f3d6a48203eeb3))

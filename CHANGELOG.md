# Changelog

## [0.5.2](https://github.com/andymai/brepkit/compare/v0.5.1...v0.5.2) (2026-03-06)

### Bug Fixes

* **operations:** migrate from `face_vertices` to `face_polygon` across 10 files — correctly samples curved edges (circles, ellipses) into 32-point polygons ([#74](https://github.com/andymai/brepkit/issues/74))
* **measure:** analytic area formulas for cylinder (angular sweep × r × h) and sphere (spherical zone formula) instead of tessellation approximation
* **measure:** subtract hole (inner wire) areas from planar face area
* **measure:** use `face_polygon` for boundary sampling in analytic area computations
* **fillet:** guard against empty-wire faces to prevent panic on degenerate topology
* **fillet:** support single-edge wires and non-planar face passthrough
* **boolean:** improve evolution tracking with relaxed thresholds and generated-face attribution
* **topology:** support single-edge closed wires; use Newell's method for wire normal computation
* **wasm:** fix corrupted doc comment on `create_apex_face` / `detect_nurbs_curve_type`
* **wasm:** add input validation to `makeEllipsoid` batch dispatch
* **wasm:** propagate tessellation errors in `tessellateSolidUV` instead of silently dropping faces

### Features

* **wasm:** add `tessellateSolidUV` — mesh with per-vertex UV coordinates
* **wasm:** add `toBREP` — serialize solid topology to JSON
* **wasm:** add `composeTransforms` — 4×4 matrix multiplication
* **wasm:** add `loftWithOptions` with start/end apex points and ruled/smooth mode
* **wasm:** add `thicken`, `makeEllipsoid`, `makeSolid`, `weldShellsAndFaces`
* **wasm:** add curvature queries: `measureCurvatureAtEdge`, `measureCurvatureAtSurface`
* **wasm:** add wire queries: `isWireClosed`, `wireLength`, `removeHolesFromFace`
* **wasm:** add 2D blueprint ops: `pointInPolygon2d`, `polygonsIntersect2d`, `intersectPolygons2d`, `commonSegment2d`, `fillet2d`, `chamfer2d`
* **wasm:** detect analytic types (line, circle, ellipse, plane, cylinder, etc.) from NURBS curves and surfaces
* **wasm:** expand batch dispatch with 9 new operations: `offsetFace`, `offsetSolid`, `section`, `split`, `sewFaces`, `draft`, `thicken`, `pipe`, `linearPattern`

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

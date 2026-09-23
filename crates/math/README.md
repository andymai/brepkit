# brepkit-math

[![crates.io](https://img.shields.io/crates/v/brepkit-math)](https://crates.io/crates/brepkit-math) [![docs.rs](https://img.shields.io/docsrs/brepkit-math)](https://docs.rs/brepkit-math)

Vectors, matrices, NURBS, analytic curves and surfaces, and exact geometric predicates.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L0.** The foundation, with no workspace dependencies.

Full API reference on [docs.rs](https://docs.rs/brepkit-math), including the tolerance model, the exactness rules, and when tolerance bites.

Everything the rest of the kernel computes with:

- **Linear algebra**: `Point3`/`Vec3` as distinct newtypes, `Mat4` affine transforms, planes, orthonormal frames.
- **NURBS**: curve and surface evaluation, derivatives, knot insertion and removal, Bezier decomposition, fitting, projection, self-intersection.
- **Analytic geometry**: `Line3D`, `Circle3D`, `Ellipse3D`, `Parabola3D`, `Hyperbola3D`, and cylinder, cone, sphere, torus surfaces, with closed-form intersections where a pair admits one.
- **Robustness**: filtered exact predicates (`orient2d`, `orient3d`), and the `Tolerance` model every comparison goes through.
- **Spatial structures**: AABB, OBB, BVH, constrained Delaunay triangulation, convex hull, ray-triangle intersection.

## Example

```rust
use brepkit_math::curves::Circle3D;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};

let center = Point3::new(0.0, 0.0, 0.0);
let circle = Circle3D::new(center, Vec3::new(0.0, 0.0, 1.0), 2.0)?;

let tol = Tolerance::new();
let quarter = circle.evaluate(std::f64::consts::FRAC_PI_2);

assert!(tol.approx_eq((quarter - center).length(), 2.0));
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

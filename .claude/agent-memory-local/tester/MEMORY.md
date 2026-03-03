# Tester Memory — brepkit

## Test Framework
- Framework: `cargo test` (standard Rust unit tests), `proptest` for property-based, `criterion` for benchmarks
- Test modules: `#[cfg(test)] mod tests` at the bottom of each source file
- Required allow attributes in test modules: `#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]`

## Key Patterns

### Building Topology in Tests
- `brepkit_topology::test_utils` (behind `test-utils` feature) provides `make_unit_square_face`, `make_unit_cube`, `make_unit_cube_manifold`, `make_unit_triangle_face`
- For a minimal solid with a custom face surface: build Wire → Face → Shell → Solid manually
- Shell::new errors if empty; Wire::new takes `Vec<OrientedEdge>` and a validity bool

### NURBS Face Construction
- Degree-1, 2×2 bilinear patch: knots `[0.0, 0.0, 1.0, 1.0]` for both u and v
- `NurbsSurface::new(degree_u, degree_v, knots_u, knots_v, ctrl_pts, weights)`
- Wrap in `FaceSurface::Nurbs(nurbs)` on a `Face::new(wire_id, vec![], surface)`

### Analytic Surfaces
- `CylindricalSurface::new(origin, axis, radius)` — errors if radius <= 0 or axis is zero
- `ConicalSurface::new(apex, axis, half_angle)` — half_angle must be in (0, π/2)
- `SphericalSurface::new(center, radius)` — errors if radius <= 0
- `ToroidalSurface::new(center, major_radius, minor_radius)` — errors if either radius <= 0
- All live in `brepkit_math::surfaces`; FaceSurface variants: `Cylinder`, `Cone`, `Sphere`, `Torus`

### Testing Private Functions
- Private helpers in the same module can be called as `super::fn_name()` from the `tests` submodule

### Tolerance
- `Tolerance::new()` has linear=1e-7, angular=1e-12
- `Tolerance::loose()` is available for approximate rotation checks
- Use `tol.approx_eq(a, b)` for float comparison; never `==`

## Key File Paths
- Test utils: `crates/topology/src/test_utils.rs`
- Analytic surfaces: `crates/math/src/surfaces.rs`
- NurbsSurface: `crates/math/src/nurbs/surface.rs`
- offset_face tests: `crates/operations/src/offset_face.rs`
- transform tests: `crates/operations/src/transform.rs`

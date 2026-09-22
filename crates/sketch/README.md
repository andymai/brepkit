# brepkit-sketch

[![crates.io](https://img.shields.io/crates/v/brepkit-sketch)](https://crates.io/crates/brepkit-sketch) [![docs.rs](https://img.shields.io/docsrs/brepkit-sketch)](https://docs.rs/brepkit-sketch)

2D parametric geometric constraint solver (GCS) for sketch-mode design.

Part of [brepkit](https://github.com/andymai/brepkit), a solid modeling kernel for Rust and WebAssembly.

**Layer L2.** Standalone: no workspace dependencies, so it is usable on its own.

Full API reference on [docs.rs](https://docs.rs/brepkit-sketch), including degrees of freedom and why the solver uses DogLeg.

A constraint system over points, lines, circles, and arcs, with ten
constraint types carrying analytic Jacobians. Solved with a DogLeg
trust-region method, which is globally convergent rather than dependent on a
good initial guess. QR factorization gives rank detection, so the system can
report degrees of freedom and identify redundant or conflicting constraints.

This crate has no B-Rep dependency. It solves 2D constraint systems and nothing
else, so it can be used independently of the rest of brepkit.

## Example

```rust
use brepkit_sketch::{Constraint, GcsSystem, PointData};

let mut sys = GcsSystem::new();
let anchor = sys.add_point(PointData { x: 0.0, y: 0.0, fixed: true });
let free = sys.add_point(PointData { x: 5.0, y: 1.0, fixed: false });

sys.add_constraint(Constraint::Distance(anchor, free, 3.0))?;
let result = sys.solve(100, 1e-10)?;

assert!(result.converged);
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

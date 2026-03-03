# Reviewer Memory

## Project: brepkit
- B-Rep CAD kernel in Rust, strict layered workspace
- L0 math crate at `crates/math/src/`
- NURBS algorithms follow Piegl & Tiller "The NURBS Book"
- Strict lints: no unsafe, no unwrap, no panic
- Uses `proptest` for property-based testing

## Reviewed Files (2026-03-01)
- `nurbs/basis.rs` - find_span (A2.1), basis_funs (A2.2), ders_basis_funs (A2.3) -- correct
- `nurbs/curve.rs` - evaluate (De Boor), derivatives (A3.2+A4.2) -- correct
- `nurbs/surface.rs` - evaluate (A3.5), derivatives (A3.6+A4.4) -- correct
- `nurbs/knot_ops.rs` - knot insertion (A5.1), split, refine -- correct
- `nurbs/decompose.rs` - Bezier decomposition, degree elevation (A5.9) -- correct
- `bvh.rs` - SAH-based flat BVH -- correct (O(n^2) right_aabb recomputation is perf concern only)
- `mat.rs` - 4x4 inverse via adjugate -- correct
- `predicates.rs` - thin wrappers over `robust` crate -- correct

## Patterns
- `find_span` takes `n` = number of control points (not n-1 as in some formulations)
- Knot insertion uses iterative single-insertion approach (simpler than batch A5.1)
- Homogeneous coordinates used throughout for rational operations
- Binomial coefficient uses iterative multiply-then-divide pattern (safe for small n)

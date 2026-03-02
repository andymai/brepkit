# Architecture

brepkit is the computational engine behind brepjs. While brepjs provides the
TypeScript API that developers interact with, brepkit handles the underlying
B-Rep modeling: geometry evaluation, boolean operations, tessellation, and
data exchange.

brepkit uses a strict layered architecture. Each layer may only depend on
layers below it, never above or sideways.

```
┌─────────────────────────────────────┐
│  L3: brepkit-wasm                   │  WASM bindings (JS API)
├─────────────────┬───────────────────┤
│  L2: operations │  L2: io           │  Modeling ops / Data exchange
├─────────────────┴───────────────────┤
│  L1: topology                       │  B-Rep data structures
├─────────────────────────────────────┤
│  L0: math                           │  Vectors, NURBS, predicates
└─────────────────────────────────────┘
```

## Layer Rules

| Crate | Layer | Allowed Dependencies |
|-------|-------|---------------------|
| `brepkit-math` | L0 | External crates only |
| `brepkit-topology` | L1 | `brepkit-math` |
| `brepkit-operations` | L2 | `brepkit-math`, `brepkit-topology` |
| `brepkit-io` | L2 | `brepkit-math`, `brepkit-topology` |
| `brepkit-wasm` | L3 | All workspace crates |

These rules are enforced by `scripts/check-boundaries.sh`.

## Arena-Based Topology

All topological entities (vertices, edges, faces, etc.) are stored in a
central `Arena` and referenced by typed index handles. This approach:

- Avoids reference counting overhead (`Rc`/`Arc`)
- Enables cache-friendly traversal (data locality)
- Makes ownership clear (the arena owns everything)
- Provides O(1) entity lookup

## NURBS-Native Geometry

Geometric entities (curves, surfaces) use NURBS as the native
representation. This means:

- Exact representation of conics (circles, ellipses) via rational NURBS
- Uniform algorithms for evaluation, subdivision, and intersection
- No special-casing for different curve/surface types

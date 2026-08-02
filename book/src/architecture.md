# Architecture

brepkit is the computational engine behind brepjs. While brepjs provides the
TypeScript API that developers interact with, brepkit handles the underlying
B-Rep modeling: geometry evaluation, boolean operations, tessellation, and
data exchange.

brepkit uses a strict layered architecture. Each layer may only depend on
layers below it, never above or sideways.

```
┌──────────────────────────────────────────────────────┐
│  L4: brepkit-wasm          brepkit-render            │  JS API / offscreen GPU
├──────────────────────────────────────────────────────┤
│  L3: brepkit-operations    brepkit-io                │  Modeling ops / exchange
├──────────────────────────────────────────────────────┤
│  L2: algo  blend  check  heal  offset  sketch        │  Engines
├──────────────────────────────────────────────────────┤
│  L1: brepkit-topology      brepkit-geometry          │  B-Rep structures
├──────────────────────────────────────────────────────┤
│  L0: brepkit-math                                    │  Vectors, NURBS, predicates
└──────────────────────────────────────────────────────┘
```

## Layer Rules

| Crate | Layer | Allowed Dependencies |
|-------|-------|---------------------|
| `brepkit-math` | L0 | External crates only |
| `brepkit-geometry` | L1 | `brepkit-math` |
| `brepkit-topology` | L1 | `brepkit-math` |
| `brepkit-algo` | L2 | `math`, `topology` |
| `brepkit-blend` | L2 | `math`, `topology` |
| `brepkit-check` | L2 | `math`, `topology`, `geometry` |
| `brepkit-heal` | L2 | `math`, `topology`, `geometry` |
| `brepkit-offset` | L2 | `math`, `topology`, `geometry` |
| `brepkit-sketch` | L2 | External crates only |
| `brepkit-operations` | L3 | All L0 to L2 crates |
| `brepkit-io` | L3 | `math`, `topology`, `operations` |
| `brepkit-wasm` | L4 | All workspace crates |
| `brepkit-render` | L4 | `math`, `topology`, `operations` |

These rules are enforced by `scripts/check-boundaries.sh`, which runs in CI.
`brepkit-render` is a leaf: nothing may depend on it.

## Arena-Based Topology

All topological entities (vertices, edges, faces, etc.) are stored in a
central `Arena` and referenced by typed index handles. This approach:

- Avoids reference counting overhead (`Rc`/`Arc`)
- Enables cache-friendly traversal (data locality)
- Makes ownership clear (the arena owns everything)
- Provides O(1) entity lookup

## Analytic Geometry, with NURBS as the General Case

Curves and surfaces are enums, not a single universal representation.
`FaceSurface` is one of `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, or
`Nurbs`; `EdgeCurve` is one of `Line`, `Circle`, `Ellipse`, or `NurbsCurve`.

Analytic types are deliberately special-cased rather than collapsed into
NURBS:

- Operations preserve them. A cylinder cut by a plane stays a `Cylinder`,
  so face counts stay low across chained booleans instead of growing with
  every step.
- Intersections take exact closed-form paths where a pair allows one
  (see `math/src/analytic_intersection.rs`), falling back to NURBS
  marching only when no analytic solution exists.
- STEP export writes them as native surface entities, so a round-trip is
  lossless rather than an approximation.

NURBS is the general representation that everything can convert into, and
the one used for free-form geometry. It is the fallback, not the default.

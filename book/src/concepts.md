# Concepts

## Boundary Representation (B-Rep)

A B-Rep solid is defined by its boundary — the surfaces, edges, and
vertices that form its outer skin. This is in contrast to:

- **CSG** (Constructive Solid Geometry): solids defined by boolean
  combinations of primitives
- **Mesh**: solids approximated by triangle meshes

B-Rep is the standard representation used in professional CAD systems
because it preserves exact geometry and supports precise operations.

## Topology vs Geometry

brepkit separates **topology** (how things are connected) from
**geometry** (where things are in space):

- **Topology**: Vertex → Edge → Wire → Face → Shell → Solid
- **Geometry**: Points, curves (NURBS), surfaces (NURBS)

A `Face` knows which `Wire` forms its boundary (topology) and which
`Surface` defines its shape (geometry). This separation is key to
robust boolean operations.

## Tolerance Model

Floating-point arithmetic introduces rounding errors. brepkit uses a
tolerance model to handle this:

- **Linear tolerance**: distance below which two points are "the same"
  (default: 1e-7)
- **Angular tolerance**: angle below which two directions are "parallel"
  (default: 1e-12 radians)

Tolerances can be configured globally or per-operation.

## NURBS

Non-Uniform Rational B-Splines (NURBS) are the mathematical foundation
for curves and surfaces in brepkit. A NURBS curve is defined by:

- **Degree**: polynomial degree (typically 1–5)
- **Control points**: points that influence the curve shape
- **Knots**: parameter values controlling basis function spans
- **Weights**: rational weights (1.0 for polynomial B-splines)

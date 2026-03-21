# check Crate Review Rules

## Layer Boundary

Flag any `use brepkit_algo::*`, `use brepkit_blend::*`, `use brepkit_heal::*`,
`use brepkit_operations::*`, or `use brepkit_io::*` import. `brepkit-check` is
L1.5 and may only depend on `brepkit-math`, `brepkit-topology`, and `brepkit-geometry`.

## Solver Positive Filtering

Flag `solve_quadratic` or `solve_cubic` used inside `solve_quartic` or other
intermediate algebraic solvers. These functions filter to positive roots only
(for ray parameters). Use `solve_quadratic_all` / `solve_cubic_all` for
intermediate results where negative values are valid before a final shift.

## Oriented Edge Traversal

Flag `edge.start()` or `edge.end()` used when iterating a wire's edges. Wire
traversal must use `oe.oriented_start(edge)` / `oe.oriented_end(edge)` to
respect edge reversal within the wire. Compare with `util.rs::face_polygon`
for the correct pattern.

## Face Trimming in Distance

Flag `point_to_face` branches for analytic or NURBS surfaces that return a
closest point without checking `is_point_in_face_boundary`. The closest point
on the infinite surface may lie outside the trimmed face — must fall back to
wire-edge distance.

## Periodic UV Coordinates

Flag UV coordinate computation on cylinders, cones, spheres, or tori that
does not unwrap angular coordinates via `unwrap_angle`. Naive `min/max` on
raw `project_point` u-values produces wrong ranges when a face straddles
the 0/2π seam.

## Inner Wire Validation

Flag validation code that iterates only `face.outer_wire()` without also
checking `face.inner_wires()`. Hole boundaries must be validated alongside
outer boundaries.

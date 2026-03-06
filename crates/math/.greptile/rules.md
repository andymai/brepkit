# math Crate Review Rules

## No Workspace Dependencies

Flag any `use brepkit_*` import or workspace dependency. `brepkit-math` is L0 and
must have zero workspace deps.

## NURBS Invariants

Flag code that constructs NURBS curves/surfaces without ensuring:
- Knot vectors are non-decreasing
- Control point count matches `n_knots - degree - 1`
- Rational weights are positive

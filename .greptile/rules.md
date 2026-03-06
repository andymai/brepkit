# brepkit Review Rules

Refer to `CLAUDE.md` for full architecture docs, module map, and coding conventions.
These rules tell Greptile what to **flag during review**.

## Layer Violations

Flag any `use brepkit_*` import or `[dependencies]` entry that breaks the layer
hierarchy defined in `CLAUDE.md` "Architecture". This is also enforced by
`scripts/check-boundaries.sh` in CI.

## Unsafe Patterns in New Code

- Flag new `unwrap()`, `expect()`, or `panic!()` outside `#[cfg(test)]` modules
- Flag `==` comparisons on floating-point values — require `tolerance.approx_eq()`
- Flag simultaneous borrow + allocate on `Topology` — require snapshot-then-allocate

## Enum Variant Changes

When a PR adds variants to `EdgeCurve` or `FaceSurface`, verify all match sites
listed in `CLAUDE.md` "Ripple-Effect Checklists" are updated — especially files
with `_ =>` wildcards that won't trigger compiler errors.

## WASM Interface Changes

When a PR modifies public `#[wasm_bindgen]` methods, check that `andymai/brepjs`
consumers won't break (parameter types, return types, `js_name` renames).

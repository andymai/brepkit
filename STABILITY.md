# Stability Policy

brepkit publishes its crates to crates.io at a shared `2.x` version. That number
comes from the `brepkit-wasm` npm release line, which the crates were unified
onto so there is one version across the workspace. **It is not a claim that the
Rust APIs have settled.** This document says what the version actually promises.

## What the shared version means

Every crate ships at the same version from the same commit, so pinning one minor
line across all of them is always consistent. Use `cargo add`, which resolves the
current version rather than one written down here.

The consequence of lockstep: a breaking change in *any* crate bumps *all* of
them, including `brepkit-wasm` and the npm package. There is no way to break one
crate's API without moving the whole line.

## Which crates are meant to be depended on

Stability follows the layer architecture rather than a per-crate list, so this
does not drift as crates change. The layer table in
[CLAUDE.md](./CLAUDE.md#layer-dependency-rules) is the source of truth for which
crate sits where, and `scripts/check-boundaries.sh` enforces it in CI.

**Consumer surface.** `brepkit-operations`, `brepkit-io`, `brepkit-topology`,
`brepkit-math`, `brepkit-sketch`, and `brepkit-wasm`. These are what a
downstream project should name in its `Cargo.toml`. Breaking changes are
possible but not routine, and land with a CHANGELOG entry.

**Internal, published only because the dependency graph requires it.** Every
other L1/L2 crate. They are published so that `brepkit-operations` resolves from
crates.io, not because their APIs are meant to be called directly. Depend on
`brepkit-operations` instead; treat these as private and expect breakage on any
release.

**Experimental.** Anything the [README status table](./README.md#status) marks
Experimental, currently including `brepkit-render`. The API is real and tested
but its shape is still open. That table is the single place feature maturity is
recorded; this document does not repeat it.

## The known upcoming break

`FaceSurface` and `EdgeCurve` (both in `brepkit-topology`) are exhaustive enums
with no `_ =>` wildcards, so **any code that matches on them exhaustively will
fail to compile when a variant is added**. Adding variants is on the roadmap:
surface-of-revolution and surface-of-linear-extrusion are needed for STEP files
that today fail to import. That will be a semver-major change for every
downstream matcher.

If you match on these enums, prefer the delegate methods on each type rather
than matching variants directly. Code that goes through a delegate keeps
compiling when a variant is added. The delegates are defined in
`crates/math/src/traits.rs`; the ripple-effect checklists in
[CLAUDE.md](./CLAUDE.md#ripple-effect-checklists) explain the pattern and how to
enumerate the current match sites.

## Semver enforcement

CI runs `cargo semver-checks` on every PR as an **advisory** check, scoped to
the consumer-surface crates above (the internal crates promise no stability, so
checking them only reports breaks this policy already permits). It reports API
breakage against the published baseline but does not block the merge, and it is
not part of the required `CI Pass` gate.

That is deliberate. The kernel is under active development and the enum change
above is planned, so a blocking gate would force major bumps on routine work
before there is a deprecation process to justify them. The check exists to make
breakage visible and deliberate rather than accidental.

When it reports a break, the options are: rework the change to be additive,
accept it and note it in the PR, or take a major bump. Promoting this check to
blocking is the right move once `brepkit-topology` stops moving.

## What is not covered

- Behavioral changes that keep the same signature. Geometry algorithms are
  improved continuously; face counts, tessellation output, and numerical results
  can change between patch versions. Pin an exact version if you keep golden
  files.
- `#[doc(hidden)]` items, and anything reachable only through an internal crate.
- The `test-utils` feature on `brepkit-topology`, which exists for this
  workspace's own tests.

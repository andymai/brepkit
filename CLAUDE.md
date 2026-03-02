# brepkit — Project Guidelines

## Architecture

Strict layered Cargo workspace. Each layer depends only on layers below it.

```
L3: brepkit-wasm        → JS bindings (wasm-bindgen)
L2: brepkit-operations  → Booleans, fillets, extrusions, tessellation
L2: brepkit-io          → STEP, 3MF import/export
L1: brepkit-topology    → B-Rep data structures (arena-based)
L0: brepkit-math        → Vectors, matrices, NURBS, predicates
```

**Layer rules** — enforced by `scripts/check-boundaries.sh`:
- L0 (math): no workspace deps
- L1 (topology): math only
- L2 (operations, io): math + topology
- L3 (wasm): all crates

## Commands

```bash
cargo build --workspace                    # Build all
cargo test --workspace                     # Test all
cargo clippy --all-targets -- -D warnings  # Lint
cargo fmt --all                            # Format
cargo build -p brepkit-wasm --target wasm32-unknown-unknown  # WASM
./scripts/check-boundaries.sh              # Verify layer deps
```

## Key Patterns

### Error handling
- `thiserror` for typed error enums per crate (`MathError`, `TopologyError`, `OperationsError`, `IoError`)
- Never `unwrap()`, `expect()`, or `panic!()` — return `Result`
- Use `#[error(transparent)]` for error propagation across crate boundaries

### Topology
- Arena-based allocation with typed `Id<T>` handles
- All entities owned by the arena, referenced by ID
- Half-edge / winged-edge adjacency via `graph` module

### Tolerance
- `Tolerance` struct with `linear` (1e-7) and `angular` (1e-12) defaults
- Compare floats via `tolerance.approx_eq(a, b)`, never `==`

### Types
- `Point3` (position) vs `Vec3` (direction) — separate newtypes
- `Mat4` for affine transforms
- NURBS curves/surfaces as the native geometry representation

## Lints

Workspace-level strict lints:
- `unsafe_code = "deny"` — no unsafe
- `unwrap_used = "deny"` — no unwrap
- `panic = "deny"` — no panic
- `clippy::pedantic`, `clippy::nursery` = warn
- `missing_docs` = warn

## Testing

- Unit tests in each module
- `proptest` for property-based testing
- Golden file tests in `tests/golden/`
- Integration tests in `tests/integration/`
- `criterion` benchmarks in `benches/`

## Git Conventions

- Conventional commits enforced by commitlint
- Pre-commit: fmt + clippy (parallel) → test
- Pre-push: full test + cargo-deny
- Branch: `main` is the primary branch

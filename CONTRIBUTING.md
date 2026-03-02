# Contributing to brepkit

## Getting Started

1. Fork and clone the repository
2. Install Rust via [rustup](https://rustup.rs/) (the `rust-toolchain.toml` will pin the version)
3. Install Node.js 20+ for commit hooks
4. Run `npm install` to set up Husky hooks and commitlint
5. Run `cargo build --workspace` to verify your setup

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings`
4. Run `cargo test --workspace`
5. Commit with a [conventional commit](https://www.conventionalcommits.org/) message
6. Open a pull request

## Commit Messages

We use conventional commits, enforced by commitlint:

```
feat: add circle primitive to math crate
fix: correct NURBS knot validation edge case
docs: update architecture diagram
test: add proptest for boolean union
refactor: simplify arena allocation
```

## Code Style

- Rust: `rustfmt` defaults (enforced by CI)
- TypeScript: Prettier + ESLint (see configs)
- No `unsafe`, no `unwrap()`, no `panic!()` in library code
- All public items must have doc comments

## Architecture

See `CLAUDE.md` for the layer system and dependency rules. The key rule:
**never add a dependency from a lower layer to a higher layer.**

## Testing

- Write unit tests alongside the code
- Use `proptest` for property-based tests where applicable
- Add golden files for STEP/3MF round-trip tests
- Run the full suite with `cargo test --workspace`

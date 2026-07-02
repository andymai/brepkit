# pr-workflow reference

Deep catalog behind SKILL.md. Everything here was verified against the repo; when it disagrees with prose elsewhere (including CLAUDE.md's "Git Conventions"), the workflow files (`.husky/*`, `.github/workflows/*.yml`) are authoritative.

## CI jobs (`.github/workflows/ci.yml`)

All jobs except `wasm-size` fan into `ci-pass` (display name `CI Pass`), the only check required by branch protection.

| Job | Display name | What it runs |
|-----|--------------|--------------|
| `fmt` | Format | `cargo fmt --all -- --check` |
| `clippy` | Clippy | `cargo clippy --all-targets --all-features -- -D warnings` |
| `test` | Test | `cargo nextest run --workspace`, doc tests, complexity guards |
| `coverage` | Coverage | llvm-cov coverage report |
| `msrv` | MSRV (1.88) | build on the minimum supported Rust |
| `wasm` | WASM Build & Validate | `cargo xtask wasm-build --skip-opt` (wasm-pack build + validation for `wasm32-unknown-unknown`) |
| `wasm-size` | WASM Size Report | PR-only size delta comment (informational, not gated by `ci-pass`) |
| `boundaries` | Layer Boundaries | `./scripts/check-boundaries.sh` |
| `deny` | Cargo Deny | license/advisory/ban checks |
| `audit` | Security Audit | `cargo audit` |
| `docs` | Rustdoc | doc build with warnings denied |
| `machete` | Unused Dependencies | `cargo machete` |
| `taplo` | TOML Format | `taplo fmt --check` |

Local pre-commit covers only fmt, clippy, taplo, machete, and the last two only when the binaries are installed (the hook skips them silently otherwise). Everything else (tests, boundaries, deny, docs) first runs in CI unless you run it yourself. Before pushing, run at minimum the tests for touched crates and, on any `Cargo.toml` change, `./scripts/check-boundaries.sh`.

## Repo merge settings (verified via `gh api repos/andymai/brepkit`)

- `allow_squash_merge: true`; merge commits and rebase merges disabled.
- `allow_auto_merge: true`; `delete_branch_on_merge: true`.
- Branch protection on `main`: linear history, no force pushes, required check = `CI Pass` only, `required_approving_review_count: 0`.
- Squash commit titles on `main` look like `type(scope): subject (#N)`.

## AI reviewers

| Actor | Surface |
|-------|---------|
| `greptile-apps` | Inline findings with P0/P1/P2 severity; posts the `Greptile Review` status check (not branch-protection required) |
| `copilot-pull-request-reviewer` | Inline review comments |
| `cubic-dev-ai` | Review plus a `cubic · AI code reviewer` check |

Reading findings:

```bash
gh api repos/andymai/brepkit/pulls/<N>/comments   # inline (diff-anchored) comments
gh pr view <N> --comments                          # issue-level comments
```

Triage: P0/P1 findings get a fix commit before auto-merge is set. P2 and style findings get a short reply (agree-and-fix, or explain why not). A finding being wrong is fine; ignoring it silently is not.

## Sandbox push details

- `origin` is `git@github.com:andymai/brepkit.git`; SSH to github.com:22 is blocked, so plain `git push` hangs until timeout.
- Global rewrite trap: `git config --get-regexp 'url\..*insteadof'` shows `url.git@github.com:.insteadof https://github.com/`. This silently converts even an explicit `git push https://github.com/...` back to SSH. The token-embedded URL avoids the rewrite because it does not match the prefix:

```bash
git push "https://x-access-token:$(gh auth token)@github.com/andymai/brepkit.git" <branch> \
  2>&1 | sed 's/x-access-token:[^@]*@/x-access-token:***@/g'
```

- Explicit-URL pushes never update local `origin/<branch>` tracking refs. Verify what the remote actually has:

```bash
gh pr view <N> --json headRefOid --jq .headRefOid
gh api repos/andymai/brepkit/commits/<branch> --jq .sha
```

Do not conclude "push failed" or "remote is behind" from `git rev-parse origin/<branch>`; that ref is stale by construction here.

- All `gh` operations (create, view, merge, api) go over HTTPS with the CLI token and work normally.

## Release-please (`.github/workflows/publish.yml`)

- Runs on every push to `main` using `googleapis/release-please-action` v5 with a bot app token.
- Config: `release-please-config.json`; current version manifest: `.release-please-manifest.json`. Single package rooted at `.`, component `brepkit-wasm`; the version is also bumped in `crates/wasm/Cargo.toml`.
- Flow: merging a `feat`/`fix`/`perf` PR creates or updates the pending release PR (`chore(main): release X.Y.Z`, head branch `release-please--branches--main--components--brepkit-wasm`). Merging that release PR creates the tag and GitHub release and publishes to npm.
- Version-neutral changes: `docs` and `chore` commits are changelog-hidden; changes only under excluded paths (`.github`, `book`, `scripts`, `benches`, `bench-results`, `examples`, `bindings`) do not bump.
- Manual escape hatch: `workflow_dispatch` on the Publish workflow with a `publish_version` input skips release-please.
- Cross-repo: brepjs (`~/Git/brepjs`) consumes the published wasm package; see the release-flow skill for the two-repo runbook.

## CI failures you did not cause

Supply-chain and toolchain jobs can fail on a PR that never touched dependencies. Root cause: `Cargo.lock` is gitignored (see `.gitignore`), so every CI run resolves dependencies fresh. The `audit` job even runs `cargo generate-lockfile` explicitly. A new advisory or a new dep release changes the verdict with zero diff on your branch.

### cargo-deny / audit / OSV advisories

The `deny` job runs cargo-deny-action against `deny.toml`; the `audit` job runs rustsec/audit-check on a freshly generated lockfile. Both gate PRs through `CI Pass`. OSV lives in its own workflow (`.github/workflows/osv-scan.yml`): on PRs it is report-only and does not gate; pushes to `main` and the Monday 06:00 UTC schedule fail closed.

Policy: do NOT widen the `deny.toml` license allowlist and do NOT add blanket ignore entries to get green. Current state to preserve: `licenses.allow` has eight permissive entries, the four non-obvious ones (CC0-1.0, ISC, BSD-2-Clause, BSD-3-Clause) carrying a provenance comment that names the dependency tree pulling them in, and there is no `[advisories] ignore` list at all (`[advisories]` sets only `unmaintained` and `yanked`).

Triage order when an advisory lands on your PR:

1. Check whether a patched version exists (`cargo audit` output or the advisory page names it). Semver-compatible patches are picked up automatically by the fresh resolution, so a persistent failure usually means the fix is in a new major or minor. Raise the version requirement in the relevant `Cargo.toml` in a separate commit, not mixed into your feature diff.
2. If no patched version exists, add a narrowly scoped `[advisories] ignore` entry to `deny.toml`: the specific advisory id, a comment explaining why it does not apply or cannot be fixed yet, and a concrete re-check trigger (e.g. "remove when <crate> X.Y ships"). State the ignore and its rationale in the PR body.
3. If unsure whether the advisory is exploitable or how to scope it, stop and report per the blocked-state rule. Silencing a security check is never a way to unblock a PR.

### MSRV job

Workspace `Cargo.toml` declares `rust-version = "1.88"`; the CI job `MSRV (1.88)` runs `cargo check --workspace --all-features` on toolchain 1.88.0. Because resolution is fresh, a dependency releasing a version that needs newer Rust breaks this job on unrelated PRs. The errors are confusing: syntax, edition, or feature errors deep inside the dependency, never the word MSRV. Fix: constrain that dependency in `Cargo.toml` to the last version that builds on 1.88. Do not bump `rust-version` casually; it is a public contract, and raising it is its own PR with its own justification.

### wasm-bindgen pin

The workspace `Cargo.toml` pins `wasm-bindgen = "=0.2.125"` and says why in an inline comment: the crate version must match the wasm-bindgen-cli tooling. The coupling is enforced in `xtask/src/wasm.rs` via the `WASM_BINDGEN_VERSION` constant; `cargo xtask wasm-build` bails when a locally installed wasm-bindgen-cli differs. The two locations must move together, and the xtask constant has lagged the Cargo.toml pin before, so verify both:

```bash
rg -n 'wasm-bindgen' Cargo.toml xtask/src/wasm.rs
```

Bumping wasm-bindgen is its own change with its own PR. Never bump it as a drive-by to fix an unrelated failure, and never let a general `cargo update`-style version bump move it silently.

### Scheduled workflows

- `.github/workflows/mutants.yml` (Mutation Testing): Sundays 02:00 UTC plus manual dispatch. Runs cargo-mutants on `brepkit-math` and `brepkit-algo` and uploads a report artifact. It never runs on PRs and never gates them; a red scheduled run is a signal to improve tests, not a merge blocker.
- `.github/workflows/osv-scan.yml`: Mondays 06:00 UTC plus main pushes, fail-closed there; report-only on PRs (see above).
- `benchmark.yml` runs on pushes and PRs, not on a schedule, and is not part of `CI Pass`.

## Symptom table

| Symptom | Cause | Action |
|---------|-------|--------|
| `git push` hangs, no output | SSH origin plus blocked port 22; or the `insteadOf` rewrite converted your HTTPS URL | Use the token-embedded HTTPS URL above |
| `origin/<branch>` does not match what you pushed | Explicit-URL pushes never update tracking refs | Verify with `gh pr view <N> --json headRefOid` |
| commitlint prints `✖ found N problems` yet the commit succeeded | The commit-msg hook swallows commitlint failures and exits 0 by design of its fallback | Treat as a rejection: `git commit --amend` to `type(scope): subject` form |
| pre-commit fails on clippy warnings you did not write | Pre-existing breakage on the branch base | Stop and report; never `--no-verify` |
| `⚠️ commitlint not available` warning on commit | `node_modules` missing, but only if no `✖` lines print above it; the same warning also follows a real lint failure (see the `✖` row) because the hook's fallback fires on any nonzero commitlint exit | If `✖` lines precede it, fix the message; otherwise `npm install` at repo root. Either way the commit went through unchecked, re-verify the message manually |
| PR shows mergeable but you have not read reviews | `Greptile Review` is not branch-protection required | Wait for the check to complete, read findings, only then `--auto` |
| `Greptile Review` check absent minutes after PR creation | Reviewers take roughly 5 to 7 minutes to post | Keep polling `gh pr checks <N>`; do other work meanwhile |
| CI `boundaries` job fails | A crate dependency violates the layer rules | Run `./scripts/check-boundaries.sh` locally; see the layer-boundaries skill |
| CI `taplo` or `machete` fails but pre-commit passed | Tool not installed locally; the hook skips it silently | `cargo install taplo-cli cargo-machete`, fix, re-commit |
| Compliance grep hits in a file you touched | You introduced a banned reference-kernel name, or you touched a grandfathered file | Remove new occurrences; leave grandfathered ones as-is |
| Release PR did not update after merge | Commit type was `docs`/`chore`, or all changes fell under excluded paths | Expected; only `feat`/`fix`/`perf` in versioned paths bump |
| `deny` or `audit` fails on a PR that never touched deps | `Cargo.lock` is gitignored; CI resolved a newly-advisoried or newly-released dep | Follow the triage order in "CI failures you did not cause"; never blanket-ignore |
| MSRV job fails with syntax or feature errors inside a dependency | A dep released a version requiring Rust newer than 1.88 | Constrain that dep in `Cargo.toml`; do not bump `rust-version` |
| `cargo xtask wasm-build` bails with a wasm-bindgen-cli version mismatch | Local CLI differs from the pin; the crate pin and `xtask/src/wasm.rs` constant must match | Install the pinned CLI version; bump the pin only as its own PR |
| Push rejected on `main` | Branch protection; direct pushes to main are not allowed | Branch and open a PR |

## Anti-patterns (what NOT to conclude)

- "CI is green so the PR is done": review findings do not block CI. The gate is the review check completing plus findings addressed.
- "The pre-push hook printed one line and passed, so the change is validated": the hook intentionally runs nothing. Validation is CI plus the local tests you ran yourself.
- "CLAUDE.md says pre-push runs tests and cargo-deny": stale. The hook file delegates to CI; do not re-add local suites to it and do not cite the stale description.
- "High-risk change, better wait for a human": no human gate exists. Address findings, then auto-merge.
- "gh pr view showed no comments, so there are no findings": inline findings live on `pulls/<N>/comments` (the API), check both surfaces.
- "The plan doc helps reviewers, commit it": working plans and specs never get committed.
- "The commit went through, so the message passed commitlint": the commit-msg hook never blocks. Check the hook output for `✖` lines and amend if any appeared.

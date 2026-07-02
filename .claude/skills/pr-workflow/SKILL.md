---
name: pr-workflow
description: Use when committing, pushing, opening, reviewing, or merging a pull request in brepkit, or when a git hook fails, a push hangs, commitlint flags a message, a PR sits waiting on AI review, or parallel work needs a worktree. Covers hooks, conventional commits, the AI-review merge gate, the sandbox HTTPS push, and release-please.
---

# PR Workflow

End-to-end change flow for this repo: branch, commit, push, PR, AI review gate, squash-merge, release. Main is protected; every change lands as a squash-merged PR. There is no human approval gate: the AI review check is the gate.

## Quick reference

| Task | Command |
|------|---------|
| Branch | `git checkout -b <type>/<kebab-description>` (e.g. `feat/render-lod`, `fix/ci-crates-io-flake`) |
| Local gate before push | `cargo nextest run -p <touched-crate>` and, if any `Cargo.toml` changed, `./scripts/check-boundaries.sh` |
| Compliance grep | See "Banned-name compliance" below |
| Push (sandbox) | `git push "https://x-access-token:$(gh auth token)@github.com/andymai/brepkit.git" <branch>` |
| Verify remote head | `gh pr view <N> --json headRefOid` (never `git rev-parse origin/<branch>`) |
| Review-gate poll | `gh pr checks <N>` until the `Greptile Review` check is completed |
| Read findings | `gh api repos/andymai/brepkit/pulls/<N>/comments` and `gh pr view <N> --comments` |
| Merge | `gh pr merge <N> --squash --auto` (only after findings are addressed) |
| Post-merge | `git checkout main && git pull --ff-only` |
| Worktree | `git worktree add .worktrees/<branch-name> <branch>` |

## Hooks: what actually runs

Hooks live in `.husky/`. Read the hook files themselves when in doubt; the "Git Conventions" section of CLAUDE.md describes an older pre-push behavior and the hook file is authoritative.

- `pre-commit`: fmt, clippy, taplo, and cargo-machete run in parallel. No tests. Expect `✅ Pre-commit checks passed.` If it fails, fix and re-commit. Caveat: the hook silently skips taplo and cargo-machete when the binaries are not installed (`command -v` guards in `.husky/pre-commit`), so a passing hook does not prove TOML formatting or unused-dep cleanliness. Install both with `cargo install taplo-cli cargo-machete`.
- `commit-msg`: runs commitlint (`@commitlint/config-conventional`, `commitlint.config.js`) but always exits 0: its not-installed fallback also swallows real lint failures, so violations print `✖` lines without blocking the commit. Treat any `✖` output as a hard failure and `git commit --amend` the message. Shape: `type(scope): subject`, e.g. `feat(render): screen-space adaptive LOD`. Nothing in CI lints messages either, and release-please parses the squash-commit title (the PR title), so a malformed title silently skips the version bump.
- `pre-push`: prints one info line and exits 0. Validation is deliberately delegated to CI (`.github/workflows/ci.yml`). Do not re-add local test runs to this hook, and do not treat its emptiness as a reason to skip local testing: run touched-crate tests yourself before pushing.

Hard rules:
- Never `--no-verify`, never `HUSKY=0`, never edit a hook to get past it. If a hook fails on pre-existing breakage you did not cause, stop and report; do not bypass.
- CI is the real gate: fmt, clippy, test (nextest workspace + doc tests), coverage, MSRV, wasm, boundaries, deny, audit, docs, machete, taplo, all fanned into one required check named `CI Pass`. Job catalog: see [reference.md](reference.md), "CI jobs".

## Procedure: land a change

1. **Branch** off `main` as `<type>/<kebab-description>`. Never commit to `main` directly.
2. **Develop and test locally.** `cargo nextest run -p <crate>` for touched crates; `./scripts/check-boundaries.sh` if you touched any `Cargo.toml` (this check only runs in CI, catch it early).
3. **Commit.** Conventional message. Never commit plan/spec working documents: if `git status` shows untracked planning docs (ad-hoc `*-plan.md` or `*-spec.md` working documents), leave them untracked.
4. **Compliance grep** (below). Expect zero output.
5. **Push.** Plain `git push` hangs: `origin` is SSH and SSH is blocked in this sandbox, and a global `insteadOf` rewrite converts bare `https://github.com/...` URLs back to SSH. Use the token-embedded URL from the quick reference. Checkpoint: `gh pr view <N> --json headRefOid` matches `git rev-parse HEAD`. Local `origin/<branch>` refs do not update after explicit-URL pushes, so never trust them.
6. **Create the PR** with `gh pr create`. Do NOT set auto-merge yet.
7. **Review gate** (next section).
8. **After merge:** `git checkout main && git pull --ff-only`. The remote branch is deleted automatically.

## The review gate

Branch protection requires only `CI Pass`. The AI review check (`Greptile Review`) is NOT required by branch protection, so the PR can show mergeable while unread findings sit on it. Policy, not GitHub, enforces the gate:

1. After `gh pr create`, work on the next independent task. Reviewers (Greptile, Copilot, cubic) comment within roughly 5 to 7 minutes.
2. Poll until the review check completes:
   ```bash
   gh pr view <N> --json statusCheckRollup \
     --jq '.statusCheckRollup[] | select(.name=="Greptile Review") | .status'
   ```
   Expect `COMPLETED`. A background watcher may poll for this, but it must hand control back for the next step, never merge on its own.
3. Read every inline finding: `gh api repos/andymai/brepkit/pulls/<N>/comments`. Fix P0/P1 findings (push a follow-up commit, which restarts CI). Reply to lower-severity findings with a reasoned response.
4. Only then: `gh pr merge <N> --squash --auto`. Auto-merge fires once `CI Pass` is green.
5. This applies to every PR including high-risk core changes (GFA boolean engine, public WASM API). No human review step exists; review-check completion plus addressed findings is the whole gate.

Anti-patterns:
- Do NOT merge because CI is green or `mergeStateStatus` is `CLEAN`. That state says nothing about review findings (see PRs #828/#830 in history, fixed in #832).
- Do NOT set `--auto` at PR-creation time.
- Do NOT conclude "no findings" from an empty `gh pr view --comments`; inline review comments live on the pulls comments API, check both.

## Banned-name compliance

The names of the reference kernel (the incumbent C++ CAD kernel) must not appear in changed files, commit messages, or PR titles/bodies. Run before pushing:

```bash
# pattern split so this file itself stays clean
banned='oc''ct|open''cascade'
git diff main... --name-only | xargs -r rg -n -i "$banned"
git log main.. --format='%s%n%b' | rg -n -i "$banned"
```

Pass condition: no output (rg exits 1). Grandfathered files that legitimately contain the names: `README.md`, `CHANGELOG.md`, `crates/wasm/CHANGELOG.md`, `scripts/bench-compare.sh`, `scripts/bench-report.ts`, `scripts/parity-loop.sh`. Do not add new occurrences anywhere, and do not "clean up" the grandfathered ones. Reading the reference kernel's source locally to study an approach is fine; naming it in committed text is not. For benchmark instructions, point at the brepjs harness scripts by path instead (see the parity-benchmarking skill).

## Worktrees

Parallel work lives inside the repo under `.worktrees/` (gitignored):

```bash
git worktree add .worktrees/<branch-name> <branch>
```

Ignore the older `../feat-branch` sibling-directory form in CLAUDE.md; in-repo `.worktrees/` is the rule. Each worktree pushes and PRs independently with the same procedure above.

## Release flow

release-please (`.github/workflows/publish.yml`) maintains a pending `chore(main): release X.Y.Z` PR. Merging a `feat`/`fix`/`perf` PR updates it; merging the release PR itself tags, creates the GitHub release, and publishes the wasm package to npm. `docs`/`chore` commits and changes under excluded paths (`.github`, `scripts`, `benches`, `examples`, and similar) do not bump the version. Cross-repo consumption by brepjs: see the release-flow skill. Details and manual escape hatch: [reference.md](reference.md), "Release-please".

## CI failures you did not cause

`Cargo.lock` is gitignored, so deny, audit, and MSRV re-resolve dependencies on every CI run; a new advisory or dep release can fail an unrelated PR with zero diff. Never widen `deny.toml` to get green. Triage order, the MSRV and wasm-bindgen pins, and scheduled workflows: see [reference.md](reference.md), "CI failures you did not cause".

## Symptoms

Symptom-to-cause table (push hangs, commitlint rejects, review check missing, release PR not updating): see [reference.md](reference.md), "Symptom table".

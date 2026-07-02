---
name: release-flow
description: Shipping a brepkit change all the way into brepjs. Use when a feature PR has merged and the change must reach npm and the brepjs adapter, when bumping the brepkit-wasm pin in brepjs, when running or debugging the type sync (sync:brepkit-types, tsc failures on Brepkit* types), when a dep-bump branch conflicts with a release-please commit, or when a push fails or hangs in this sandbox.
---

# release-flow: brepkit merge to brepjs activation

## When to use

A brepkit change is not "shipped" when its PR merges. It is shipped when brepjs
installs the new kernel version, the type sync regenerates cleanly, and the
adapter can call the new method. This skill covers that 4-hop chain and the
trap at each hop. For getting the feature PR itself merged, see the
`pr-workflow` skill. For benchmark comparisons after a release, see
`parity-benchmarking`. To test an unreleased kernel build inside brepjs
without waiting for any of this chain, see the `wasm-bindings` skill.

## Quick reference: the 4-hop chain

Each hop is gated on the previous one. Never skip a gate.

| Hop | Action | Gate before next hop |
|-----|--------|---------------------|
| 1 | Feature PR squash-merges to brepkit main | Merge lands on main |
| 2 | release-please opens `chore(main): release X.Y.Z` PR; it auto-merges, tags `vX.Y.Z`, and the release event triggers the publish workflow | GitHub release exists |
| 3 | Publish workflow builds wasm and runs `npm publish` | `npm view brepkit-wasm versions --json` lists X.Y.Z AND your commit is an ancestor of the tag |
| 4 | In brepjs: bump the devDependency pin, `npm install`, `npm run sync:brepkit-types` | tsc and knip pass, diff drops no method signatures |

Key commands:

```bash
npm view brepkit-wasm versions --json
cd ~/Git/brepkit && git fetch --tags
git merge-base --is-ancestor <feature-sha> vX.Y.Z && echo in-release
cd ~/Git/brepjs && npm install && npm run sync:brepkit-types
```

## Procedure

### Hop 1 to 2: brepkit release

1. After the feature PR merges, the push to main runs
   `.github/workflows/publish.yml`. Its `release-please` job opens or updates
   the release PR, and its `auto-merge` job enables squash auto-merge on that
   PR. You do not merge it by hand; it lands when checks pass.
2. Checkpoint: `gh pr list --search 'chore(main): release'` shows the PR, then
   it disappears and `gh release list` shows the new version. If the release
   PR is stuck, check its CI, not the workflow.
3. The tag is plain `vX.Y.Z` (no package prefix; see
   `release-please-config.json`, `include-component-in-tag: false`).

### Hop 3: poll npm, do not trust the tag

The GitHub tag and release appear minutes before npm has the version, because
the publish job is still building wasm. Gate on both of these:

```bash
npm view brepkit-wasm versions --json          # array must end with "X.Y.Z"
git merge-base --is-ancestor <feature-sha> vX.Y.Z && echo in-release
```

The second check matters when releases are frequent: your commit may have
missed the cut and be waiting for the NEXT release. Do not bump brepjs to a
version that does not contain your change.

If the release exists but npm never gets the version, the publish job failed;
`workflow_dispatch` on publish.yml with a `publish_version` input is the
manual escape hatch.

### Hop 4: brepjs dependency bump and type sync

In `~/Git/brepjs` (use a worktree if the checkout is busy):

1. Edit `package.json`: set the **devDependencies** entry
   `"brepkit-wasm": "X.Y.Z"` (exact pin). Leave the **peerDependencies**
   range alone; it already covers `^2.0.0`. The pin is what brepjs CI tests
   against; the range is what consumers may bring.
2. `npm install`. Then verify the lockfile kept the three `@emnapi` entries
   (see Pitfalls below).
3. `npm run sync:brepkit-types`. This regenerates
   `src/kernel/brepkit/brepkitWasmTypes.ts` from the installed
   `node_modules/brepkit-wasm/brepkit_wasm.d.ts`. That file is generated;
   never hand-edit it.
4. Checkpoint: `npm run validate && npm run knip`; pass means both exit 0.
   `validate` (`scripts/validate-change.sh`) runs typecheck, lint, boundary
   check, format check, and tests. knip is a separate gate not covered by
   `validate`; do not skip it. If tsc fails on an undefined
   `Js*` or `Brepkit*` type, you hit the type-sync trap: see
   [reference.md](reference.md), "Type-sync internals".
5. Checkpoint: the diff of `brepkitWasmTypes.ts` may legitimately shrink
   (comment compaction), but it must drop zero real method signatures:

   ```bash
   git diff src/kernel/brepkit/brepkitWasmTypes.ts | grep '^-' | grep -E '\w+\(' 
   ```

   Every removed line here must reappear as a `+` line or be a comment.

## Pitfalls: symptom to cause

| Symptom | Cause | Fix |
|---------|-------|-----|
| Tag `vX.Y.Z` exists but `npm view` lacks X.Y.Z | Publish job still running or failed | Wait and re-poll; if failed, `workflow_dispatch` publish |
| tsc: cannot find name `JsFoo` after sync | New return type has no `mapReturnType` case and no emitted interface block | reference.md, "Type-sync internals": both edits are required |
| New method missing from generated types | Method not listed in `METHOD_SECTIONS` in `scripts/sync-brepkit-types.ts` | Add it to the right section |
| Dep-bump PR: "merge commit cannot be cleanly created" | release-please bumped package.json/lock on brepjs main right after a merge | reference.md, "Lockfile rebase": rebase, take main's lock, `npm install`, never hand-merge |
| `gh pr view` says CONFLICTING right after force-push | GitHub computes mergeability async; the value is stale | Re-query after a few seconds before acting |
| knip flags a src export as unused, blocks push | knip cannot trace usage from `tests/` | `@testOnly` JSDoc tag on the export; `knip.config.ts` already has `tags: ['-testOnly']` |
| CI `npm ci` fails: `Missing: @emnapi/core ... from lock file` | Old npm (< 11.11) dropped the three `@emnapi` lock entries | reference.md, "@emnapi lockfile" |
| `git push` hangs or times out | SSH port 22 blocked, and a global insteadOf rewrite converts even explicit https URLs back to SSH | reference.md, "Pushing over HTTPS": token-embedded URL |
| `git rev-parse origin/<branch>` disagrees with GitHub | Explicit-URL pushes do not update remote-tracking refs | Confirm head via `gh pr view <n> --json headRefOid` |

## Anti-patterns: what NOT to conclude

- Tag exists, so npm has it: false, poll npm.
- Tag exists, so my commit is in it: false, check ancestry.
- Types now declare the method, so the adapter's `typeof` guard is dead code:
  false. Consumers bring any kernel in the peer range, which is wider than
  the pin. Keep the guard (see reference.md, "Adapter feature-detection").
- Lockfile conflict, so resolve hunks by hand: never. Take main's lockfile
  and regenerate with `npm install`.
- Peer range needs bumping with the pin: no, only the devDependency pin moves
  per release.
- `brepkitWasmTypes.ts` needs a small fix, so edit it directly: no, it is
  regenerated; fix the sync script and re-run.

## Before any push, in either repo

Grep changed files, commit messages, and the PR body for the banned
reference-kernel names. See reference.md, "Reference-kernel hygiene" for the
exact commands (the pattern is split there so these skill files pass their
own check).

# release-flow reference

Deep detail for the hops in SKILL.md. Paths are absolute; brepjs lives at
`~/Git/brepjs`.

## Publish workflow anatomy (brepkit)

File: `~/Git/brepkit/.github/workflows/publish.yml` ("Release & Publish").

- `release-please` job: runs on push to main, opens or updates the
  `chore(main): release X.Y.Z` PR, exposes a `release_created` output.
- `auto-merge` job: runs `gh pr merge <release-pr> --auto --squash`. The
  release PR merges itself once checks pass. Do not merge it manually unless
  auto-merge visibly failed.
- `publish` job: triggers on the GitHub `release` event, on the
  `release_created` output, or on `workflow_dispatch` with a
  `publish_version` input (the manual escape hatch). It builds the wasm
  package for both targets:

  ```bash
  wasm-pack build crates/wasm --target bundler --release --out-dir pkg
  wasm-pack build crates/wasm --target nodejs --release --out-dir pkg-node
  ```

  merges them into one package, verifies `package.json` version equals the
  tag, and runs `npm publish --provenance`. Publish is idempotent: it skips
  if the version is already on npm, so a re-run is safe.
- Versioning config: `release-please-config.json` sets
  `"component": "brepkit-wasm"` but `"include-component-in-tag": false`, so
  tags are plain `vX.Y.Z`. It also bumps `crates/wasm/Cargo.toml` via
  `extra-files`.

## Type-sync internals (brepjs)

Script: `~/Git/brepjs/scripts/sync-brepkit-types.ts`, run via
`npm run sync:brepkit-types`.

What it does:

1. Parses the `BrepKernel` class body out of
   `node_modules/brepkit-wasm/brepkit_wasm.d.ts` with a regex (the regex
   cannot handle nested-paren parameter types; a method like that will parse
   wrong, check the output).
2. Emits `src/kernel/brepkit/brepkitWasmTypes.ts`. The header records
   `Synced against brepkit-wasm@<version>`. The file is AUTO-GENERATED;
   hand edits are clobbered on the next sync.
3. Only methods listed in `const METHOD_SECTIONS: Section[]` appear in the
   output, grouped by category label. Find it:
   `rg -n 'const METHOD_SECTIONS' scripts/sync-brepkit-types.ts`.
4. Methods not referenced from the adapter layer
   (`src/kernel/brepkit/*.ts`, excluding the generated file) are tagged
   `/** @unwired */`.

### The new-return-type trap

`function mapReturnType(rt, methodName)` maps known wasm return types to
local interface names: `JsMesh` to `BrepkitMesh`, `JsEdgeLines` to
`BrepkitEdgeLines`, `JsGroupedMesh` to `BrepkitGroupedMesh`. A return type of
`any` maps to `ANY_RETURN_OVERRIDES[methodName] ?? 'string'` (example
override: `getEdgeNurbsData: 'string | null'`).

An unmapped `JsFoo` passes through verbatim into the generated file, where no
`BrepkitFoo` interface exists, so tsc fails. Adding the method to
`METHOD_SECTIONS` alone is NOT enough. A new `Js*` return type requires BOTH:

1. A case in `mapReturnType` returning `'BrepkitFoo'`.
2. A hardcoded interface block emitted by the script. The existing ones are
   pushed as lines, e.g. `lines.push('export interface BrepkitMesh {')`.
   Find them: `rg -n "export interface Brepkit" scripts/sync-brepkit-types.ts`.
   Add a matching block for the new type, mirroring the fields of the Rust
   struct in `crates/wasm/src/shapes.rs` or `types.rs`.

Then re-run the sync and re-check tsc.

Notes:

- wasm-bindgen `Vec<f32>` / `Vec<u32>` getters return owned copies (the JS
  glue slices and frees the Rust buffer), so the interface fields are plain
  `Float32Array` / `Uint32Array` and no defensive copy is needed JS-side.
- Interim state before a dep bump lands: a hand-added OPTIONAL forward
  declaration for the new method lets the adapter's `typeof` guard typecheck
  against the old kernel; the next sync replaces it canonically.

### Verifying the sync diff

The regenerated file may shrink for legitimate reasons (JSDoc and formatting
compaction). What it must never do is silently drop a method. Check:

```bash
git diff src/kernel/brepkit/brepkitWasmTypes.ts | grep '^-' | grep -E '\w+\('
```

Eyeball each hit: it must either reappear as an added line or be a comment.
A genuinely dropped signature means a method fell out of `METHOD_SECTIONS`
or the upstream `.d.ts` regressed; stop and find out which.

## Lockfile rebase (the release-please collision)

brepjs also uses release-please. Right after any brepjs PR merges, the bot
pushes a commit to main bumping `package.json` version, `package-lock.json`,
and `CHANGELOG.md`. A dep-bump branch that also touches package.json and the
lock then fails to merge ("merge commit cannot be cleanly created").

Resolution, always the same shape:

```bash
git fetch origin
git rebase origin/main
git checkout --ours package-lock.json
npm install
git add package-lock.json
git rebase --continue
expected=$(gh pr view <n> --json headRefOid -q .headRefOid)
git push --force-with-lease=<branch>:"$expected" \
  "https://x-access-token:$(gh auth token)@github.com/andymai/brepjs.git" <branch>
```

Notes:

- The lease MUST carry an explicit expected value. Explicit-URL pushes never
  update remote-tracking refs (see "Pushing over HTTPS"), so a bare
  `--force-with-lease` has no expected value to check against and is always
  rejected with `! [rejected] ... (stale info)`. Take the expected old head
  from `gh pr view` immediately before the push, as above.
- In a REBASE, `--ours` is the side you are rebasing ONTO, i.e. main. That
  is intentional: discard your branch's lockfile, then let `npm install`
  regenerate it against the merged `package.json`.
- `package.json` usually auto-merges cleanly (the version field and your dep
  line are far apart); only the lockfile truly conflicts.
- Never resolve lockfile hunks by hand. The file is generated; hand-merged
  lockfiles are how CI gets `npm ci` integrity failures.
- After the force-push, GitHub's mergeable state is computed asynchronously.
  `gh pr view <n> --json mergeable,mergeStateStatus` returning `CONFLICTING`
  or `DIRTY` immediately after the push is usually stale. Re-query after a
  few seconds; it flips to `MERGEABLE` or `BLOCKED` (blocked = waiting on
  checks, which is fine).

## brepjs push gates

### knip and @testOnly

knip (the unused-export linter, a pre-push gate) cannot trace usage from
`tests/` (separate tsconfig and import alias), so a src export exercised only
by tests gets flagged as unused and blocks the push. The sanctioned escape
hatch, already wired in `~/Git/brepjs/knip.config.ts` via
`tags: ['-testOnly']`:

```ts
/** @testOnly Exercised by tests/vectorOperations.test.ts. */
export function myHelper() { ... }
```

Live examples: `src/utils/vec2d.ts`, `src/measurement/measureCache.ts`,
`src/core/kernelBoundary.ts`. Do not delete the export or inline it into the
test to appease knip; tag it.

### @emnapi lockfile entries

`brepkit-wasm` has three transitive optional deps whose top-level lockfile
entries npm older than 11.11 silently dropped on `npm install`, making CI's
`npm ci` fail with `Missing: @emnapi/core@... from lock file`. Current local
npm is past 11.11, so plain `npm install` is safe. Keep the verification
step regardless, after any install that touches the lock:

```bash
grep -c '"node_modules/@emnapi/core"' package-lock.json
grep -c '"node_modules/@emnapi/runtime"' package-lock.json
grep -c '"node_modules/@emnapi/wasi-threads"' package-lock.json
```

Each must print `1`. If any prints `0` you are on an old npm; upgrade npm
rather than hand-editing the lock.

### Adapter feature-detection

The adapter layer (`src/kernel/brepkit/*Ops.ts`) guards newer kernel methods
and keeps a fallback path:

```ts
if (typeof bk.tessellateSolidGroupedBinary === 'function') { ... }
```

Live guard sites: `meshOps.ts` (`tessellateSolidGroupedBinary`),
`modifierOps.ts` (`chamferAsymmetric`, `offsetWire2DWithJoin`),
`ioOps.ts` (`fromBREP`), `repairOps.ts` (`validateSolidDetails`).

Keep the guard even after the re-synced types declare the method as present.
The devDependency pin is exact, but consumers satisfy the wide
peerDependencies range (`^0.10.1 || ^1.0.0 || ^2.0.0`) and may run an older
kernel where the method does not exist. eslint does not flag the guard as
redundant; leave it.

## Reference-kernel hygiene

The name of the C++ kernel brepkit replaces is banned from commits, PR
titles and bodies, and code in both repos. Call it "the reference kernel".
The brepkit-side compliance grep and the grandfathered file list (README,
the two changelogs, three bench scripts) live in the pr-workflow skill.
In brepjs, package and adapter identifiers that contain the name are
legitimate code; the ban targets prose, commit messages, and PR text.

For brepjs, the pattern below is assembled from split shell strings so this
file passes its own check; the shell concatenates them into the real pattern.

```bash
pat="$(printf 'o%s|open%s' 'cct' 'cascade')"
git diff main --name-only --diff-filter=d | xargs -r rg -in "$pat" --
git log main.. --format='%B' | rg -in "$pat"
gh pr view <n> --json title,body -q '.title + .body' | rg -in "$pat"
```

Run all three before every push and before merging, in BOTH repos. Expected
output: nothing (rg exits nonzero on no match, which is the pass state).
Any hit in prose must be rewritten before pushing. A hit that is a literal
package identifier in brepjs code, or inside a brepkit grandfathered file,
is acceptable.

## Pushing over HTTPS (SSH blocked)

Both repos' `origin` is an SSH URL, port 22 is blocked, and a global
`insteadOf` rewrite converts even explicit `https://github.com/...` URLs
back to SSH. Full mechanics: see the pr-workflow skill, "Sandbox push
details". The same token-embedded form works for brepjs:

```bash
git push "https://x-access-token:$(gh auth token)@github.com/andymai/brepjs.git" <branch> \
  2>&1 | sed 's/x-access-token:[^@]*@/x-access-token:***@/g'
```

Consequences:

- Explicit-URL pushes do NOT update `origin/<branch>` remote-tracking refs.
  Never trust `git rev-parse origin/<branch>` for the remote head; use
  `gh pr view <n> --json headRefOid` or
  `gh api repos/andymai/<repo>/commits/<branch>`.
- `gh` itself (pr create, view, merge, api) talks HTTPS and is unaffected.

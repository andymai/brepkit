# brepkit Skill Library

Distilled working knowledge for building, debugging, and shipping the brepkit B-Rep kernel:
each skill captures the method and traps for one recurring class of task. Written for
engineers and agents working in this repo (plus `~/Git/brepjs` for cross-repo work), with
only this repo and CLAUDE.md as context.

## Index

| Skill | Reach for it when |
|-------|-------------------|
| [roadmap](roadmap/SKILL.md) | Picking work in an autonomous session: what is open, what is terminal, the chase filters, and the acceptance bar. |
| [debugging-doctrine](debugging-doctrine/SKILL.md) | Any hard geometry bug where the first diagnosis did not hold, or before a multi-pass investigation. |
| [solid-verification](solid-verification/SKILL.md) | Deciding whether a solid is actually correct: watertight, manifold, right volume, and whether a "passing" check proves anything. |
| [boolean-debugging](boolean-debugging/SKILL.md) | A fuse, cut, or intersect mesh-falls-back, loses faces, gives wrong volume, fails validation, or varies across runs. |
| [fillet-blend](fillet-blend/SKILL.md) | A fillet or chamfer leaves free edges, opens the shell, silently changes nothing, or you must decide whether the deprecated v1 fillet code is safe to touch. |
| [analytic-preservation](analytic-preservation/SKILL.md) | An operation degrades exact analytic geometry to NURBS or a mesh, face counts explode, or you are triaging the approx census. |
| [numerical-robustness](numerical-robustness/SKILL.md) | Results flip under tiny input nudges, seams and periodic geometry misbehave, or code compares, hashes, or buckets floats. |
| [tessellation](tessellation/SKILL.md) | Mesh cracks at face boundaries, boundary edges or non-manifold edges appear, or you are adding a face mesher in `crates/operations/src/tessellate/`. |
| [add-operation](add-operation/SKILL.md) | Adding or extending a modeling operation in `crates/operations`, end to end through tests and wasm exposure. |
| [layer-boundaries](layer-boundaries/SKILL.md) | Adding a workspace dep, placing new code in a crate, adding an `EdgeCurve` or `FaceSurface` variant, or fixing a boundaries CI failure. |
| [wasm-bindings](wasm-bindings/SKILL.md) | Adding `BrepKernel` methods, wiring `executeBatch`, building the wasm package, or debugging wasm-only failures. |
| [io-formats](io-formats/SKILL.md) | A STEP or other file imports wrong (dropped solids, all-NURBS, hard entity errors), verifying writer round-trips, capturing a faithful fixture, or adding a format in `crates/io`. |
| [render-verify](render-verify/SKILL.md) | Working on `brepkit-render` or visually verifying a solid, including headless capture of a live viewer window. |
| [testing](testing/SKILL.md) | Writing or placing tests, building a faithful regression fixture, handling golden mismatches, or ending a session with unverified work. |
| [profiling](profiling/SKILL.md) | An operation or benchmark is slow, a criterion bench misbehaves, or a PR needs before/after perf numbers. |
| [parity-benchmarking](parity-benchmarking/SKILL.md) | Proving brepkit matches or beats the reference kernel, overlaying a local build into the gridfinity tool, or quoting any perf or parity claim. |
| [pr-workflow](pr-workflow/SKILL.md) | Committing, pushing, opening, or merging a PR; hook failures, commitlint, the AI-review merge gate, worktrees. |
| [release-flow](release-flow/SKILL.md) | Landing a merged brepkit change in brepjs: npm release, wasm pin bump, type sync, adapter update. |

## Suggested reading order for a new engineer

1. Doctrine and verification: `roadmap`, `debugging-doctrine`, `solid-verification`, `numerical-robustness`, `testing`.
2. The engine: `boolean-debugging`, `fillet-blend`, `analytic-preservation`, `tessellation`.
3. Building: `layer-boundaries`, `add-operation`, `wasm-bindings`, `io-formats`, `render-verify`.
4. Shipping: `pr-workflow`, `profiling`, `parity-benchmarking`, `release-flow`.

## Glossary

- **GFA**: the General Fuse Algorithm, the boolean engine in `crates/algo` (`gfa.rs` is the orchestrator).
- **PaveFiller**: the GFA intersection phase (`crates/algo/src/pave_filler/`). It finds all pairwise interferences between the operands before any faces are split.
- **FF/EE/VF phases**: PaveFiller sub-phases, one per entity pair type: vertex-vertex, vertex-edge, edge-edge, vertex-face, edge-face, face-face. See `pave_filler/phase_*.rs`.
- **Pave**: an intersection point on an edge, stored with its curve parameter.
- **Pave block**: the edge segment between two consecutive paves; the unit the builder splits edges into.
- **SD (same-domain)**: two faces (or edges) from different operands that occupy the same geometry. They must be detected and merged to one representative or booleans produce duplicates and open shells.
- **PCurve**: the 2D curve in a face's UV parameter space that traces a 3D edge on that surface. Face splitting and classification run in UV space, so a wrong pcurve breaks them.
- **Analytic vs NURBS**: analytic means an exact typed surface or curve (plane, cylinder, cone, sphere, torus, line, circle, ellipse). NURBS is the free-form spline representation. Keeping results analytic is brepkit's differentiator: exact downstream math, compact data, GPU meshing from parameters.
- **Mesh fallback**: when a boolean cannot assemble a valid B-Rep, it degrades to a triangle-mesh boolean and re-imports the triangles as many small planar faces. Always a defect to investigate, never an acceptable result.
- **Watertight**: the tessellated mesh has zero boundary edges; every triangle edge is shared by exactly two triangles.
- **Manifold**: at the B-Rep level, every edge is used by exactly two faces with consistent orientation.
- **Euler check**: the V - E + F consistency check in solid validation; a cheap topological invariant that catches missing or duplicated entities.
- **Seam**: the edge where a closed surface's parameterization wraps around (u=0 meets u=2*pi on a cylinder). Seam edges appear twice in a face's UV boundary.
- **Periodic surface**: a surface closed in one or both parameter directions (cylinder, cone, sphere, torus). Periodic wrap-around is a standing source of seam and interval bugs.
- **Deflection**: the maximum allowed chord deviation between a mesh and the true surface; the knob that controls tessellation density.
- **The reference kernel**: the established C++ CAD kernel brepkit benchmarks against through the brepjs harness. Parity with it, then beating it, is the project's acceptance bar; see `parity-benchmarking` for how to run the head-to-head.

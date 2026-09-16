# Isolated accepted/adjoining fillet repair (N339)

Base: upstream v3.4.0, `f3b0eb113db42dfc997996c7f4fc6523661c6923`.
The consumer is OttoCAD's explicit `fillet_rolling_ball` route. This change does
not reroute brepkit's production/WASM v2 API or alter any validator or tolerance.

## Reproduction and cause

Extrude the rectangle `(0,0)–(40,0)–(40,25)–(0,25)` by 30 mm along +Z.
Fillet `(0,0,0)–(40,0,0)` at radius 1 mm. Accept that result and resolve
`(0,0,1)–(0,0,30)` on it; fillet at radius 1 mm. The original rolling-ball
builder leaves four free boundary edges: it passes the pre-existing cylinder
through unchanged, although the new strip terminates at its tangent vertex.
The missing corner is not coincident geometry that sewing can recover.

The v2 sequential control refuses at the boundary-incidence audit. Its grouped
control is structurally closed but is not a geometry-qualified substitute.
The explicit ignored v2 regression remains a failing negative qualification
probe when invoked; it is not fixed or counted as a successful case here.

## Bounded construction

Recognize a single equal-radius convex quarter-cylinder, with perpendicular
planar supports and planar axial caps, beside one requested sharp edge.
Recover the original sharp supports on copied topology; extend only collinear
straight boundary edges. Require rectangular corner supports and more than
twice the radius of clearance on the three sharp corner edges. Rebuild the old
and new strips together, with an exact rational biquintic Hermite-Coons corner.
Its four cross-derivative constraints give tangent joins to both cylinders and
both planar supports; its retained sharp-edge endpoint is a boundary singularity.
The two planar cubic boundary curves are installed on their shared edges.

Strict native output validation and positive finite volume are required before
returning this route's result. Original and accepted source geometry are not
mutated. Unrecognized cases retain their existing engine behavior, not a claim
of new support. No general unequal-radius, multi-corner, non-rectangular, cavity,
or arbitrary grouped-adjoining capability is introduced.

A positional Coons candidate was rejected despite closing (about 23-degree
cylinder seam creases). A horn-torus candidate was rejected despite closing
(opposite seam normals and a folded transition). The retained candidate's
four seam normal dots are 1 within floating-point rounding. Its local test
checks 99 samples per boundary and 9801 interior normals, bounds and material
containment; the integration checks 36 radius/end/support/rigid-pose combinations
and rigid covariance of sampled patch and boundary geometry.

## STEP prerequisite

The existing writer discarded rational surface weights, changing the corner
even though import preserved closed topology. The surface writer now emits the
complex base/knots/weights entity form. The reader assembles those components
and refuses missing, malformed, wrongly shaped, nonpositive or nonfinite weight
grids instead of substituting unit weights. Non-rational surface output and
rational-curve handling are unchanged.

At the original radii the first volume is 29991.309264420124 mm³; the second is
29984.72052947068 mm³, measured at 0.01 mm deflection. Before the STEP fix the
second imported as 29984.728233340913 mm³ with maximum sampled surface error
0.03620572526177567 mm. After the fix the error is 7.021752650337982e-16 mm and
volume is unchanged at each of 0.01, 0.001 and 0.0001 mm deflection. These are
tessellated mass measurements, not a claim of analytic volume or pose-invariant
tessellation. The degree, knots and weights also round-trip.

## Verification

- `cargo test --workspace`: passes, with existing ignored tests and the explicitly
  ignored v2/comparison probes still visible.
- `cargo clippy --workspace --all-targets -- -D warnings`, workspace build,
  formatting, boundary/doc-path/version checks: pass.
- `regress_fillet_accepted_adjoining`: five acceptance/control tests pass,
  including independent grouped/sequential edges and source preservation on
  clearance/excessive-radius refusal; two probes explicitly ignored.
- `regress_adjoining_fillet_step`: geometry, weights, strict topology and the
  consumer's unchanged volume bound pass. All 44 STEP unit tests pass.
- OttoCAD candidate override: 100 adapter unit tests, 399 core, 601 UI, 164 app,
  and three REF-PART fixture tests pass. The worker previews and accepts twice,
  reopens, and validates three fresh replays. The three reduced tessellation
  fingerprints agree for each stage. The full gauntlet intentionally fails its
  immutable-pin assertion under a local override; historical capability failures
  remain (notably connected curvature-limited fillet and hybrid STEP refusal).

This is a bounded upstream repair candidate, not closure of OttoCAD's missing
literal original user package or permission to bypass its separate review gates.

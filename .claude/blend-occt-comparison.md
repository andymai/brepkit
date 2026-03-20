# brepkit-blend vs OCCT Fillet/Chamfer Feature Comparison

Generated: 2026-03-19

## Source files compared

| Area | brepkit-blend | OCCT |
|------|--------------|------|
| Fillet builder | `fillet_builder.rs` | `ChFi3d_FilBuilder.cxx` |
| Chamfer builder | `chamfer_builder.rs` | `ChFi3d_ChBuilder.cxx` |
| Walking engine | `walker.rs` | `BRepBlend_Walking.cxx` |
| Blend constraint | `blend_func.rs` | `BlendFunc_ConstRad.cxx` |
| Analytic fast paths | `analytic.rs` | `ChFiKPart_ComputeData.cxx` |
| Vertex/corner | `corner.rs` | `ChFi3d_Builder_6.cxx`, `ChFi3d_Builder.cxx` |
| Spine | `spine.rs` | `ChFiDS_Spine.hxx` |
| Stripe/section | `stripe.rs`, `section.rs` | `ChFiDS_Stripe`, `ChFiDS_SurfData` |
| Radius law | `radius_law.rs` | `ChFiDS_FilSpine` + `Law_Function` |
| Face trimming | `trimmer.rs` | Integrated into `ChFi3d_Builder` assembly |

---

## Feature Comparison Matrix

### 1. Fillet Builder (FilletBuilder vs ChFi3d_FilBuilder)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Constant radius fillet | Full | Full (plane-plane only) | Only analytic plane-plane path works; walker fallback not integrated into builder | P0 |
| Variable radius fillet (Law_Function) | Full | Partial | `EvolRadBlend` exists but builder `add_edges_with_law` snapshots to constant at t=0 (line 88: `law.evaluate(0.0)`) | P0 |
| Per-edge radius | Full | Full | Both support different radii per edge set | -- |
| Per-vertex radius | Full (`SetRadius(R, IC, Vertex)`) | None | OCCT can set radius at individual vertices for smooth transitions | P1 |
| Radius at parameter (UandR pairs) | Full (`SetRadius(gp_XY, IC, IinC)`) | None | OCCT supports radius specified at arbitrary spine parameters | P1 |
| G1 edge chain propagation | Full (`PerformElement` traverses tangent edges) | None | brepkit builds single-edge spines only (comment: "no G1 chain propagation in v1") | P0 |
| Fillet shape control | Full (Rational / QuasiAngular / Polynomial) | None | OCCT supports 3 section shapes via `ChFi3d_FilletShape`; brepkit uses rational quadratic only | P2 |
| Surface-surface walking | Full (all surface types) | Partial | Walker engine exists and works, but builder only dispatches to analytic; returns `UnsupportedSurface` error for non-plane pairs | P0 |
| Surf-Rst (surface-restriction) walking | Full (`BRepBlend_SurfRstLineBuilder`) | None | OCCT handles fillet rolling off one surface onto a face boundary edge | P1 |
| Rst-Rst (restriction-restriction) walking | Full (`BRepBlend_RstRstLineBuilder`) | None | OCCT handles fillet between two boundary edges (e.g., fillet meeting another fillet) | P1 |
| Singular point handling | Full (`ChFi3d_SearchSing`, jalon points) | None | OCCT adds singular points as "jalons" the walker must pass through | P1 |
| Partial failure recovery | Full (per-stripe error handling) | Full | Both record per-edge failures and return partial results | -- |
| Reverse walking (bidirectional) | Full (`Continu` walks backward from midpoint) | None | brepkit walks forward only from `s_start` to `s_end` | P1 |

### 2. Chamfer Builder (ChamferBuilder vs ChFi3d_ChBuilder)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Symmetric chamfer (equal distances) | Full | Full (plane-plane only) | Same limitation as fillet: analytic only | P0 |
| Asymmetric chamfer (two distances) | Full | Full (plane-plane only) | Analytic path works; walker not integrated | P0 |
| Distance-angle chamfer | Full | Full | `ChamferAngleBlend` + `add_edges_distance_angle` | -- |
| Constant throat chamfer | Full (`ChFiDS_ConstThroatChamfer`) | None | OCCT has a mode where the throat (height of isosceles triangle in section) is constant | P2 |
| Constant throat with penetration | Full (`ChFiDS_ConstThroatWithPenetrationChamfer`) | None | OCCT has a mode with right-angled triangle section and offset surface | P2 |
| Chamfer mode enum | Full (3 modes: Classic, ConstThroat, ConstThroatWithPenetration) | None | brepkit has no concept of chamfer modes | P2 |
| Spine extension at vertex | Full (`ExtentSpineOnCommonFace`) | None | OCCT extends two chamfer spines where they meet at a vertex | P1 |

### 3. Walking Engine (Walker vs BRepBlend_Walking)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Newton-Raphson correction | Full (`math_FunctionSetRoot`, up to 30 iters) | Full (custom `newton_solve`, 20 iters) | Equivalent core approach | -- |
| Adaptive step control | Full (halve on failure, grow on success) | Full (halve on failure, 1.5x on success) | Equivalent | -- |
| Linear predictor | Partial (uses `evalpinit` for extrapolation) | Full (linear extrapolation from previous two solutions) | brepkit's predictor is cleaner | -- |
| Guide deflection control | Full (angle check via `CosRef3D`, halves step if guide tangent turns too fast) | None | OCCT checks the angle between consecutive spine tangents and rejects steps where the 3D deflection is too large | P0 |
| Domain classification | Full (`TopolTool::Classify` checks if UV solution is inside face domain) | None | OCCT classifies each solution point as IN/OUT/ON relative to the face boundaries; brepkit does no domain check | P0 |
| Boundary recadrage | Full (`Recadre` via `Blend_FuncInv` finds where blend curve exits face domain) | None | When the solution falls outside a face, OCCT computes the exact exit point on the boundary restriction | P0 |
| Extremity correction | Full (`CorrectExtremityOnOneRst`) | None | OCCT corrects endpoint positioning on restrictions | P1 |
| Twist detection | Full (`twistflag1/2` members, test during walking) | None (error variant exists but no detection logic) | OCCT tracks twist state per surface during walking | P1 |
| Singular point passing | Full (jalon-based: `AddSingularPoint`, walker ensures it passes through them) | None | OCCT inserts mandatory waypoints for the walker to handle singularities | P1 |
| Multiple solution branches | Full (8 `choix` values for ConstRad: combinations of ray1/ray2 sign) | None (single sign convention) | OCCT supports 8 solution branches; brepkit always uses positive radius from both surfaces | P1 |
| 4x4 solver | OCCT: `math_FunctionSetRoot` (general NxN, Gauss with backtracking) | brepkit: custom Gaussian elimination with partial pivoting | Both adequate; OCCT's is more general | -- |
| Bi-directional walking | Full (`Perform` + `Complete` walk both directions from start) | None (single direction only) | OCCT can start from a midpoint and walk both directions | P1 |
| Step floor check | Full (`abs(stepw) < tolgui` terminates with current extremity) | Full (`step < min_step` returns error) | OCCT gracefully terminates; brepkit returns error | P1 |

### 4. Blend Constraint Function (ConstRadBlend vs BlendFunc_ConstRad)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Core equation: planarity + equidistance | Full | Full | Same mathematical formulation: `E(1) = nplan.(mid - guide)`, `E(2..4) = (P1 + R*npn1) - (P2 + R*npn2)` | -- |
| Normal projection to section plane | Full (`ncrossns / |ncrossns|`) | Full (`project_normal_to_section`) | Same double-cross-product approach: `npn = (nplan x nsurf) x nplan / |nplan x nsurf|` | -- |
| Analytic Jacobian | Full (exact derivatives using D2 surface data: `dns1u1`, `dns1v1`, etc.) | Partial (finite differences for dnpn/du terms via `finite_diff_npn`) | OCCT computes exact `d(npn)/du` from surface second derivatives; brepkit uses central finite differences with h=1e-7 | P1 |
| Degenerate surface normal handling | Full (`BlendFunc::ComputeNormal` fallback for zero-length normals) | None (returns zero vector, which will cause Newton failure) | OCCT handles surface singularities (poles, seams) where `du x dv = 0` | P1 |
| Second-order derivatives (D2E) | Full (`D2EDX2`, `D2EDXDT`, `D2EDT2`) | None | OCCT computes full Hessian for the constraint system (used by approximation, not by walking) | P2 |
| Solution choix (8 branches) | Full (ray1/ray2 independently signed based on `choix` 1-8) | None | OCCT handles all 4 combinations of rolling ball position relative to each surface (convex/concave) | P1 |
| Cache/memoization | Full (checks if X and T unchanged before recomputing) | None | OCCT avoids redundant surface evaluations | P2 |
| Section shape (Rational/QuasiAngular/Polynomial) | Full | None (rational only) | OCCT can generate sections using different approximation strategies | P2 |
| DEDT (derivative w.r.t. spine parameter) | Full | None | OCCT computes how the constraints change as the spine parameter varies (needed for Continu/Complete methods) | P1 |

### 5. Analytic Fast Paths (analytic.rs vs ChFiKPart_ComputeData)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| **Fillet: Plane-Plane** | Full | Full | Both produce cylindrical surface | -- |
| **Fillet: Plane-Cylinder** (line spine) | Full | Stub (returns `None`) | OCCT: `ChFiKPart_ComputeData_FilPlnCyl` produces a torus section | P0 |
| **Fillet: Plane-Cylinder** (circle spine) | Full | None | OCCT handles circular edges between plane and cylinder | P0 |
| **Fillet: Plane-Cone** | Full | Stub (returns `None`) | OCCT: `ChFiKPart_ComputeData_FilPlnCon` | P1 |
| **Fillet: Cylinder-Plane** | Full (swaps operands) | None | OCCT handles symmetry internally | P0 |
| **Fillet: Cone-Plane** | Full (swaps operands) | None | OCCT handles symmetry internally | P1 |
| **Chamfer: Plane-Plane** | Full | Full | Both produce planar chamfer surface | -- |
| **Chamfer: Plane-Cylinder** (sym/asym) | Full | None | OCCT: `ChFiKPart_ComputeData_ChPlnCyl` | P0 |
| **Chamfer: Plane-Cone** | Full | None | OCCT: `ChFiKPart_ComputeData_ChPlnCon` | P1 |
| **Chamfer: Asymmetric Plane-Plane** | Full | Full (via `try_analytic_chamfer` with d1, d2) | -- | -- |
| **Chamfer: Asymmetric Plane-Cylinder** | Full (`ChAsymPlnCyl`) | None | OCCT has specific asymmetric variants for cylinder | P1 |
| **Chamfer: Asymmetric Plane-Cone** | Full (`ChAsymPlnCon`) | None | OCCT has specific asymmetric variants for cone | P2 |
| **Rotule (knuckle)** | Full (`ChFiKPart_ComputeData_Rotule`) | None | Specialized surface for rolling-ball contact | P2 |
| **Sphere** | Full (`ChFiKPart_ComputeData_Sphere`) | None | Analytic sphere cap at 3-edge vertex | P2 |
| PCurve computation | Full (exact 2D curves on both faces and blend surface) | Placeholder (hardcoded `Line2D` at origin) | brepkit PCurves are non-functional placeholders | P0 |
| Orientation handling | Full (considers face orientations `OrFace1/OrFace2` and `Or1/Or2`) | None | OCCT correctly handles reversed faces and different normal orientations | P0 |

Total analytic cases: OCCT handles **10+ surface pair combinations** (with line and circle spine variants) for fillets, plus **8+ combinations** for chamfers. brepkit handles **1 fillet case** (plane-plane) and **1 chamfer case** (plane-plane).

### 6. Vertex Blend / Corner (corner.rs vs ChFi3d_Builder_6.cxx + ChFi3d_Builder.cxx)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Corner classification | Full (1/2/3/N stripes per vertex) | Full (None/TwoEdge/SphereCap/CoonsPatch) | Equivalent classification | -- |
| 1-stripe corner (`PerformOneCorner`) | Full (~800 LOC, complex boundary adjustment) | None (classified as "None") | OCCT adjusts the blend surface endpoint to meet the vertex constraint | P0 |
| 2-stripe corner (`PerformTwoCorner`) | Full (intersection-based: `PerformTwoCornerbyInter`) | Partial (classified as `TwoEdge` but no implementation shown) | OCCT intersects two blend surfaces and trims them to meet; brepkit classifies but does not build | P0 |
| 3-stripe corner (`PerformThreeCorner`) | Full (virtual, implemented per builder) | Partial (SphereCap + CoonsPatch fallback) | brepkit's SphereCap works for orthogonal equal-radius case; OCCT's is more general | P1 |
| N-stripe corner (`PerformMoreThreeCorner`) | Full | Partial (CoonsPatch truncates to 3 points) | brepkit ignores points beyond 3 for N>4; OCCT handles arbitrary valence | P1 |
| Corner surface quality | Full (G1-continuous patches with proper tangent matching) | G0 only (bilinear/degenerate NURBS patches) | brepkit uses degree 1 interpolation; OCCT produces smooth patches | P1 |
| Edge-face contact reconstruction | Full (recomputes blend points at vertex via `CompBlendPoint`) | None (uses section endpoints) | OCCT reconstructs the blend contact at the vertex from UV coordinates; brepkit uses proximity fallback | P1 |
| Vertex topology (obstacle detection) | Full (`IsObst`, `IsVois` graph traversal) | None | OCCT walks the vertex-edge adjacency graph to classify obstacles | P2 |
| Line/parameter update at corner | Full (`UpdateLine`, `SearchIndex`) | None | OCCT updates the blend line parameter sequence when corners modify it | P1 |

### 7. Spine (Spine vs ChFiDS_Spine)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Edge chain storage | Full | Full | Both store ordered edge sequences with cumulative arc lengths | -- |
| Arc-length parameterization | Full (via `Absc`, `Parameter`, `Length`) | Full (`locate`, `evaluate`, `params`) | Equivalent | -- |
| Closed spine detection | Full (`IsClosed`, `IsPeriodic`, `Period`) | Full (`is_closed`) | brepkit lacks periodicity support | P1 |
| D0/D1/D2 evaluation | Full (up to second derivatives) | Partial (D0 via linear interpolation, D1 via finite difference between endpoints) | brepkit uses chord-length approximation for curves; OCCT evaluates the actual edge curve | P0 |
| Curved edge evaluation | Full (via `BRepAdaptor_Curve`, handles Line, Circle, BSpline, etc.) | None (linear interpolation between vertices) | brepkit treats all edges as straight lines regardless of `EdgeCurve` type | P0 |
| First/Last status (free/closed/tangent) | Full (`ChFiDS_State`: `FreeBoundary`, `ClosedSect`, `OnSame`, `Closed`, etc.) | None | OCCT tracks what happens at spine endpoints (free boundary, intersection, closed loop) | P0 |
| Tangent management at endpoints | Full (`SetFirstTgt`, `SetLastTgt`, `HasFirstTgt`, `HasLastTgt`) | None | OCCT stores tangent constraints at spine endpoints for boundary conditions | P1 |
| Reference parameter | Full (`SetReference`, used for approximation) | None | OCCT uses reference parameters for B-spline approximation quality | P2 |
| ElSpine (smooth C2 representation) | Full (`ChFiDS_ElSpine`, a smooth BSpline through the edge chain) | None | OCCT builds a smooth C2 spline over the piecewise edge chain to avoid tangent discontinuities | P0 |
| Offset spine | Full (`SetOffsetEdges`, `OffsetEdges`) | None | OCCT supports offset spines for certain chamfer modes (ConstThroatWithPenetration) | P2 |
| Error status tracking | Full (`SetErrorStatus`, `ErrorStatus`) | None (errors propagate via `BlendError`) | OCCT has structured error status on the spine itself | P2 |
| Concavity type | Full (`ChFiDS_TypeOfConcavity`) | None | OCCT tracks whether the blend is convex or concave | P1 |

### 8. Face Trimming (trimmer.rs vs ChFi3d_Builder assembly)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Planar face trimming | Full | Full | brepkit's 2D segment intersection works for convex planar faces | -- |
| Non-planar face trimming | Full (UV-space PCurve-based splitting) | None (returns original face untrimmed) | OCCT trims faces using PCurves in UV space, supporting any surface type | P0 |
| Same-edge hit handling | Full | None (returns error if both hits on same edge) | OCCT handles the case where the contact line enters and exits through the same boundary edge | P1 |
| Multiple intersection handling | Full (handles complex face boundaries with holes) | None (requires exactly 2 hits) | brepkit only works for convex faces with exactly 2 boundary crossings | P1 |
| Edge sharing (topology merge) | Full (shared edges between trimmed face and blend face) | None (creates duplicate vertices/edges) | brepkit creates new vertices at trim points instead of sharing with blend face vertices | P0 |
| Inner wire (hole) handling | Full | None | OCCT correctly handles faces with holes during trimming | P2 |

### 9. NURBS Surface Approximation

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Section-to-surface fitting | Full (`BRepBlend_AppSurf`, `BRepBlend_AppSurface`) | Partial (`approximate_blend_surface` in walker.rs) | brepkit builds a simple interpolated surface from circular sections; OCCT uses full B-spline approximation with error control | P1 |
| Rational quadratic arcs | Full (via `GeomFill`, multiple section shapes) | Full (degree-2 rational Bezier per section) | Both produce rational arcs; OCCT has more flexibility | -- |
| Approximation error control | Full (tolerance-based refinement) | None (uniform knot vector, no error check) | OCCT iteratively refines until approximation tolerance is met | P1 |
| V-direction fitting | Full (B-spline approximation through sections) | Partial (uniform clamped knots, degree min(n-1, 3)) | brepkit's V-direction is a simple interpolation without error control | P1 |

### 10. Radius Law (RadiusLaw vs Law_Function + ChFiDS_FilSpine)

| Feature | OCCT | brepkit | Gap | Priority |
|---------|------|---------|-----|----------|
| Constant radius | Full | Full | Both support constant radius | -- |
| Linear interpolation | Full (via `Law_Linear`) | Full (`RadiusLaw::Linear`) | Equivalent | -- |
| S-curve (smooth transition) | Full (via `Law_S`) | Full (`RadiusLaw::SCurve`) | brepkit uses 3t^2 - 2t^3; OCCT's S-law is similar | -- |
| Custom law function | Full (any `Law_Function` subclass) | Full (`RadiusLaw::Custom`) | Both support arbitrary functions | -- |
| Composite law | Full (`Law_Composite` chains multiple laws) | None | OCCT can compose different laws along different spine segments | P1 |
| Per-edge constant radius | Full (different R per edge in chain) | None | OCCT supports different constant radii on different edges of the same spine | P1 |
| Law evaluation at spine parameter | Full (evaluates law in spine curvilinear abscissa space) | Partial (evaluates in [0,1] normalized parameter) | brepkit normalizes to [0,1]; OCCT uses actual arc-length parameters | P1 |

---

## Summary Statistics

| Category | OCCT Features | brepkit Full | brepkit Partial | brepkit None | Coverage |
|----------|--------------|-------------|-----------------|-------------|----------|
| Fillet Builder | 12 | 2 | 2 | 8 | 17% |
| Chamfer Builder | 7 | 3 | 0 | 4 | 43% |
| Walking Engine | 14 | 3 | 1 | 10 | 21% |
| Blend Function | 8 | 2 | 1 | 5 | 25% |
| Analytic Fast Paths | 14 | 3 | 0 | 11 | 21% |
| Corner/Vertex | 8 | 1 | 2 | 5 | 13% |
| Spine | 12 | 3 | 1 | 8 | 25% |
| Trimming | 6 | 1 | 0 | 5 | 17% |
| **Total** | **81** | **18** | **7** | **56** | **22%** |

---

## Priority Distribution

| Priority | Count | Description |
|----------|-------|-------------|
| **P0** | 19 | Must-have for basic functionality beyond plane-plane boxes |
| **P1** | 26 | Important for production quality (convergence, robustness, multi-edge) |
| **P2** | 12 | Nice-to-have (advanced chamfer modes, polynomial sections, caching) |

---

## Top P0 Gaps (Recommended Implementation Order)

1. **Integrate walker into fillet/chamfer builders** -- The walking engine exists and works (verified by tests) but the builders return `UnsupportedSurface` for any non-plane pair. Wire up `ConstRadBlend` + `Walker` as fallback when `try_analytic_fillet` returns `None`.

2. **Spine curve evaluation** -- Currently all edges treated as straight lines (`p0 + (p1 - p0) * t`). For circular or NURBS edges, must evaluate the actual `EdgeCurve`. This is prerequisite for plane-cylinder fillets.

3. **Plane-cylinder analytic fillet** -- Covers ~30% of real-world fillets (box edges meeting cylindrical holes/bosses). The stub exists. Produces torus section.

4. **ElSpine (smooth spine)** -- OCCT's `ChFiDS_ElSpine` builds a C2 B-spline through the edge chain, avoiding tangent discontinuities at vertices. Without this, multi-edge spines will have kinks.

5. **G1 chain propagation** -- Walk tangent edges to build multi-edge spines. Required for filleting along edge chains (e.g., box lip).

6. **Domain classification in walker** -- Check if UV solutions are inside face boundaries. Without this, the walker produces solutions that may be outside the face domain.

7. **PCurve computation** -- Current placeholders prevent proper face trimming for non-planar faces. Need real UV-space curves on the support surfaces.

8. **Orientation handling** -- Current analytic paths ignore face orientation, producing incorrect geometry for reversed faces.

9. **Spine endpoint status** -- OCCT's `ChFiDS_State` (FreeBoundary, ClosedSect, etc.) determines how the blend terminates. Without this, blends at free edges or closed loops will fail.

10. **Edge sharing in trimming** -- Trimmed faces create duplicate vertices instead of sharing edges with the blend face, producing non-manifold topology.

---

## Architectural Observations

### What brepkit does well
- Clean trait-based `BlendFunction` abstraction (vs OCCT's class hierarchy)
- Predictor-corrector in walker is well-structured (linear extrapolation + Newton correction)
- `RadiusLaw` enum is simpler and more ergonomic than OCCT's `Law_Function` hierarchy
- Error handling via `BlendError` enum is idiomatic Rust (vs OCCT's exception throwing)
- Builder pattern with partial failure recording is user-friendly
- Test coverage is solid for the implemented scope

### What OCCT does that brepkit fundamentally lacks
- **Restriction (boundary) handling**: OCCT's entire `SurfRst` and `RstRst` subsystem handles blends that touch face boundaries. This is critical for adjacent fillets, fillets meeting chamfers, and fillets that run off a face edge.
- **C2 smooth spines**: Without `ElSpine`, multi-edge fillets will have visible seams at edge chain joints.
- **Full Jacobian**: OCCT computes exact analytic derivatives of the projected normal, while brepkit uses finite differences. This affects convergence speed and robustness on high-curvature surfaces.
- **8-branch solution selection**: OCCT's `choix` system selects which of 8 possible ball positions to track. brepkit always uses one configuration, which will produce wrong results for concave fillets on some face orientations.

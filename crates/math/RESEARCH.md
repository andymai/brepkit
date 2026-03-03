# Math Foundation Layer Research for brepkit

Temporary research file. Gathered 2026-03-01.

## Current State of brepkit-math (L0)

Already implemented:
- `Vec2`, `Vec3`, `Point2`, `Point3` newtypes with basic arithmetic
- `Mat3`, `Mat4` with transforms (translate, scale, rotate), determinant
- `NurbsCurve` with De Boor evaluation and `find_span` (binary search)
- `NurbsSurface` data structure (evaluation is `todo!()`)
- `predicates`: `orient2d`, `in_circle` via the `robust` crate (Shewchuk)
- `Tolerance` struct with linear/angular thresholds

---

## 1. Essential NURBS Algorithms

### 1.1 Basis Function Evaluation (The NURBS Book Ch. 2)

These are the innermost hot-path routines. Everything else builds on them.

| Algorithm | NURBS Book | Purpose |
|-----------|-----------|---------|
| FindSpan | A2.1 | Binary search for knot span index. **Already implemented** in `curve.rs`. |
| BasisFuns | A2.2 | Non-vanishing basis functions N_{i,p}(u) via Cox-de Boor recurrence. Currently embedded in De Boor eval; should be a standalone function for reuse. |
| DersBasisFuns | A2.3 | Basis function derivatives up to k-th order. Needed for tangents, normals, curvature. |
| OneBasisFun | A2.4 | Single basis function value (useful for knot refinement checks). |
| DersOneBasisFun | A2.5 | Derivatives of a single basis function. |

Sources:
- NURBS-Python implements A2.1-A2.5: https://nurbs-python.readthedocs.io/en/5.x/module_utilities.html
- LNLib (C++ reference impl): https://github.com/BIMCoderLiang/LNLib

### 1.2 Curve Evaluation and Derivatives (Ch. 3-4)

| Algorithm | NURBS Book | Purpose |
|-----------|-----------|---------|
| CurvePoint | A3.1 | Evaluate B-spline curve point. **Already implemented** as `NurbsCurve::evaluate()`. |
| CurveDerivsAlg1 | A3.2 | Curve derivatives directly from basis function derivatives. Needed for tangent vectors. |
| CurveDerivCpts | A3.3 | Control points of derivative curves. Used by Alg2. |
| CurveDerivsAlg2 | A3.4 | Curve derivatives via derivative control points. More efficient for repeated evaluation. |
| RatCurveDerivs | A4.2 | Derivatives of rational (NURBS) curves. Applies quotient rule to homogeneous derivatives. |

**Why needed:** Tangent vectors are required for edge direction, trimming curve orientation, fillet radius computation, and sweep frame calculation. Second derivatives give curvature for adaptive tessellation and G2 continuity checks.

### 1.3 Surface Evaluation and Derivatives (Ch. 3-4)

| Algorithm | NURBS Book | Purpose |
|-----------|-----------|---------|
| SurfacePoint | A3.5 | Evaluate B-spline surface point. Currently `todo!()` in `surface.rs`. |
| SurfaceDerivsAlg1 | A3.6 | Surface partial derivatives (du, dv, mixed). |
| SurfaceDerivCpts | A3.7 | Control points of partial derivative surfaces. |
| RatSurfaceDerivs | A4.4 | Derivatives of rational (NURBS) surfaces. |

**Why needed:** Surface normal = cross(du, dv) is required for face orientation, Boolean classification, rendering, and STEP export. Second derivatives needed for curvature analysis and adaptive tessellation.

### 1.4 Knot Operations (Ch. 5)

| Algorithm | NURBS Book | Purpose |
|-----------|-----------|---------|
| CurveKnotIns | A5.1 | Insert a single knot into a curve (Boehm's algorithm). |
| SurfaceKnotIns | A5.3 | Insert knot into surface in u or v direction. |
| RefineKnotVectCurve | A5.4 | Insert multiple knots simultaneously (knot refinement). |
| RefineKnotVectSurface | A5.5 | Surface knot refinement. |
| RemoveCurveKnot | A5.8 | Remove a knot from a curve (with tolerance). |
| DegreeElevateCurve | A5.9 | Raise curve degree by 1. |
| DegreeReduceCurve | (Eq. 5.36+) | Lower curve degree (approximate, with error bound). |

**Why needed:**
- **Knot insertion:** Required for curve/surface splitting (split at parameter = insert knot to full multiplicity). Splitting is needed by Boolean operations to divide faces along intersection curves.
- **Knot refinement:** Needed for Bezier decomposition (insert all interior knots to full multiplicity). Bezier decomposition is the gateway to Bezier clipping intersection algorithms.
- **Degree elevation:** Needed for STEP export compatibility (matching degree requirements) and for merging curves of different degrees.
- **Knot removal:** Used for simplification/compression of NURBS data, especially after Boolean operations produce unnecessarily complex geometry.

### 1.5 Point Inversion and Projection (Ch. 6)

| Algorithm | NURBS Book | Purpose |
|-----------|-----------|---------|
| CurvePointInversion | A6.1 | Given a point known to be on the curve, find parameter u. Newton iteration. |
| SurfacePointInversion | A6.2 | Given a point on the surface, find (u, v). Newton iteration. |
| CurvePointProjection | A6.3-A6.4 | Find closest point on curve to arbitrary point. Subdivision + Newton. |
| SurfacePointProjection | A6.5-A6.6 | Find closest point on surface to arbitrary point. |

**Why needed:** Point projection is one of the most-called operations in a B-Rep kernel. Used for:
- Classifying points as inside/outside/on geometry
- Snapping operations
- Distance computation between entities
- Tessellation (projecting sample points back to surface)
- Trimmed surface evaluation (classifying parameter-space points)

The standard approach: (1) subdivide into Bezier segments, (2) use control polygon/net as initial guess, (3) refine with Newton-Raphson iteration.

Sources:
- Ma & Hewitt, "Point inversion and projection for NURBS curve and surface: Control polygon approach": https://www.sciencedirect.com/science/article/abs/pii/S0167839603000219

### 1.6 Curve and Surface Creation (Ch. 7-8)

| Algorithm | Purpose |
|-----------|---------|
| CircleToNURBS | Exact rational NURBS representation of circular arcs (weight = cos(theta/2)). |
| ConicToNURBS | Ellipses, parabolas, hyperbolas as rational quadratic NURBS. |
| LineToNURBS | Degree-1 NURBS (trivial but needed for uniform representation). |
| BilinearSurface | Four-corner interpolating surface. |
| RuledSurface | Linear interpolation between two boundary curves. |
| ExtrudedSurface | Translational sweep of a curve. |
| RevolvedSurface | Surface of revolution from a profile curve. |
| SweptSurface | General sweep (curve along a rail with frame). |
| LoftSurface | Surface through a sequence of cross-section curves. |
| GordonSurface | Surface from a network of curves. |
| CoonsPatch | Boundary-interpolating surface from 4 boundary curves. |

**Why needed:** These are the constructors that `brepkit-operations` (L2) will call. Every extrude, revolve, loft, and fillet operation creates NURBS surfaces using these algorithms.

### 1.7 Intersection Algorithms

This is the most complex area. Boolean operations depend entirely on robust intersection.

#### Curve-Curve Intersection

| Method | Description |
|--------|-------------|
| Bezier clipping | Convert to Bezier, iteratively clip parameter domain. Quadratically convergent for transversal intersections. |
| Subdivision + Newton | Subdivide both curves, test bounding box overlap, refine with Newton-Raphson on the distance function. |
| Implicitization (low degree) | For line-curve and conic-curve, convert one to implicit form and substitute. |

#### Curve-Surface Intersection

| Method | Description |
|--------|-------------|
| Bezier clipping (ray-surface) | Project to 2D distance function, clip in parameter space. Used by Sederberg & Nishita (1990). |
| Newton-Raphson | 3 equations (surface point = curve point) in 3 unknowns (u_curve, u_surf, v_surf). |
| Subdivision | Recursive subdivision of both entities, bounding volume tests for early rejection. |

#### Surface-Surface Intersection (SSI)

| Method | Description |
|--------|-------------|
| Marching method | Start from seed points, trace the intersection curve by stepping along it. Standard approach in OCCT. |
| Lattice method | Tessellate both surfaces to polyhedra, intersect the meshes, then fit NURBS to the intersection polyline. |
| Subdivision + marching | Subdivide into Bezier patches, test bounding box overlap, find seed points, march. |
| Algebraic (low degree) | Resultant-based methods for low-degree surfaces. |

**Starting point computation** (critical for marching): OpenCascade uses the IntPolyh package which meshes both surfaces, intersects the meshes, and uses intersection points as starting seeds.

Sources:
- Sederberg & Nishita, "Curve intersection using Bezier clipping" (1990): https://www.sciencedirect.com/science/article/abs/pii/001044859090039F
- Bezier clipping convergence proof: https://www.sciencedirect.com/science/article/abs/pii/S0167839607001434
- Beer et al., "Algorithms for geometrical operations with NURBS surfaces": https://arxiv.org/pdf/2210.13160
- OpenCascade modeling algorithms: https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_algos.html

---

## 2. Geometric Predicates

### 2.1 Currently Implemented (via `robust` crate)

- `orient2d(a, b, c)` -- 2D orientation test (Shewchuk)
- `in_circle(a, b, c, d)` -- 2D incircle test (Shewchuk)

### 2.2 Needed Additions

| Predicate | Purpose |
|-----------|---------|
| **orient3d(a, b, c, d)** | Determines if d is above/below/on the plane of (a,b,c). Essential for 3D point classification in Boolean operations. The `robust` crate does not ship this -- would need a separate implementation or the `robust-predicates` approach. |
| **insphere(a, b, c, d, e)** | 3D Delaunay criterion. Needed if doing 3D Delaunay tessellation. |
| **point_in_polygon_2d** | Winding number or ray-casting test in parameter space. Critical for trimmed surface evaluation -- determines if a (u,v) point is inside trim loops. |
| **point_on_segment** | Collinearity + range check. Used in edge classification. |
| **segment_segment_2d** | 2D segment intersection with classification (crossing, touching, overlapping, disjoint). Needed for trim curve processing. |
| **point_vs_plane** | Signed distance from point to plane. Used for BSP classification in Boolean ops. |
| **edge_classification** | Classify an edge as IN/OUT/ON relative to a solid. Core Boolean operation primitive. |

### 2.3 Robustness Strategy

Shewchuk's adaptive precision approach is the gold standard. It uses a multi-level filter:
1. Fast floating-point evaluation (exact for well-separated cases)
2. Error bound check -- if result magnitude > error bound, the sign is correct
3. Exact arithmetic fallback (expansion arithmetic) only when needed

This gives exact results with near-floating-point speed for non-degenerate cases.

Sources:
- Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (1997): https://www.cs.cmu.edu/~quake/robust.html
- Paper: https://people.eecs.berkeley.edu/~jrs/papers/robust-predicates.pdf
- Rust `robust` crate (already a dependency): https://github.com/georust/robust

---

## 3. Bounding Volumes

### 3.1 Types Needed

| Type | Description | Use Case |
|------|-------------|----------|
| **AABB (Axis-Aligned Bounding Box)** | Min/max corners in world coordinates. Cheapest to compute and test. | First-pass broad-phase rejection for all intersection queries. NURBS control polygon convex hull is bounded by the AABB of control points. |
| **OBB (Oriented Bounding Box)** | Tighter fit via principal component analysis or min-volume box. More expensive to compute. | Tighter culling for elongated geometry (e.g., thin surface patches). Worth it when AABB produces too many false positives. |
| **Bounding Sphere** | Center + radius. Trivial overlap test (distance < sum of radii). | Quick pre-filter. Particularly useful for rotation-invariant tests. |
| **k-DOP (Discrete Oriented Polytope)** | Generalization of AABB to k oriented slabs. 6-DOP = AABB. 14-DOP or 26-DOP common. | Better fit than AABB, cheaper than OBB. Good middle ground for BVH nodes. |

### 3.2 Bounding Volume Hierarchy (BVH)

A BVH tree over surface patches is needed for:
- Surface-surface intersection: recursively test child bounding volumes, prune non-overlapping pairs
- Point projection: prune patches that are far from the query point
- Ray casting for visualization/selection
- Collision detection between solids

**Recommended approach:** AABB tree (cheapest to build and query) as the default, with OBB as an option for pathological cases. AABB of NURBS patches can be conservatively computed from control point extrema (convex hull property).

### 3.3 NURBS-Specific Bounding Properties

The **convex hull property** of B-splines means:
- A NURBS curve lies within the convex hull of its control polygon
- A NURBS surface lies within the convex hull of its control net
- After knot insertion, the control polygon/net converges to the curve/surface

This means the AABB of control points is a valid (conservative) bounding volume, and it tightens with each subdivision.

Sources:
- BVH overview: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy
- Bounding volume types: https://en.wikipedia.org/wiki/Bounding_volume

---

## 4. Interval Arithmetic and Root-Finding

### 4.1 Bezier Clipping (Primary Method)

The most important intersection root-finding method for a NURBS kernel.

**How it works:**
1. Convert NURBS segment to Bezier form (via knot insertion to full multiplicity)
2. Express the distance/intersection function in Bernstein basis
3. Use the convex hull of the Bernstein control polygon to clip away parameter regions that cannot contain roots
4. Recurse until the parameter interval is smaller than tolerance

**Convergence:** Quadratically convergent for simple (transversal) roots. Linear for tangential intersections.

**Key advantage over Newton-Raphson:** Guaranteed to find all roots in the interval. Newton can miss roots or diverge.

Source:
- Sederberg & Nishita (1990): https://www.sciencedirect.com/science/article/abs/pii/001044859090039F
- Convergence proof: https://www.sciencedirect.com/science/article/abs/pii/S0167839607001434

### 4.2 Newton-Raphson Iteration

Used as a refinement step after getting a good initial guess from subdivision or Bezier clipping.

**For point inversion:** Minimize ||S(u,v) - P||^2. Gradient is zero at the solution. Use Gauss-Newton (linearize the surface, solve 2x2 system each step).

**For curve-curve intersection:** Minimize ||C1(u) - C2(v)||^2. Newton on 2 variables.

**For surface-surface marching:** At each step, solve for the next point on the intersection curve using Newton on the constraint that the point lies on both surfaces.

### 4.3 Interval Arithmetic

Used to get guaranteed bounds on function values over parameter intervals.

| Concept | Purpose |
|---------|---------|
| **Interval evaluation** | Given [u_lo, u_hi], compute [f_lo, f_hi] that bounds f(u) over the interval. For Bernstein polynomials, the convex hull property gives this for free. |
| **Interval Newton** | Newton's method with interval arithmetic. Guaranteed enclosure of all roots. Bezier clipping is essentially a geometric version of interval Newton. |
| **Bernstein sign test** | If all Bernstein coefficients have the same sign, the polynomial has no root in [0,1]. |
| **de Casteljau subdivision** | Split a Bezier curve at a parameter, producing two sub-curves. The control polygons of the sub-curves converge to the curve, tightening bounds. |

### 4.4 Additional Root-Finding Needs

| Algorithm | Purpose |
|-----------|---------|
| **Brent's method** | Robust scalar root-finding. Good fallback for 1D problems where Newton oscillates. |
| **Projected polyhedron** | For SSI starting points: approximate both surfaces with polyhedra, intersect them. |
| **Eigenvalue method** | For finding all roots of univariate Bernstein polynomials via companion matrix. Exact count guaranteed. |

---

## 5. Tessellation Support

### 5.1 Adaptive Surface Tessellation

The operations crate (`brepkit-operations`) has a `tessellate` module, but the math layer needs to provide:

| Capability | Description |
|------------|-------------|
| **Surface evaluation at (u,v)** | The core evaluation. Must be fast since tessellation calls it thousands of times. |
| **Surface normal at (u,v)** | Needed for: flat-facet quality assessment, lighting, and STL/3MF export. |
| **Curvature estimation** | First and second fundamental forms. Used to decide where to subdivide. |
| **Flatness criterion** | Given a surface patch, estimate max deviation from a planar approximation. If deviation < tolerance, the patch is flat enough. |
| **Bezier decomposition** | Decompose NURBS surface into Bezier patches for independent tessellation. |

### 5.2 Trim Curve Tessellation

For trimmed NURBS surfaces:
1. Tessellate trim curves in 2D parameter space
2. Triangulate the trimmed parameter domain (constrained Delaunay triangulation)
3. Map parameter-space triangles to 3D via surface evaluation

The math layer needs point-in-polygon tests (see Predicates section) and the 2D curve evaluation support for this.

### 5.3 Flatness Criteria

| Method | Description |
|--------|-------------|
| **Control polygon deviation** | Max distance from interior control points to the line/plane connecting endpoints. Cheap, conservative. |
| **Mid-point test** | Evaluate surface at patch center, compare to bilinear interpolation of corners. |
| **Normal deviation** | Max angle between normals at patch corners. If small, the patch is approximately flat. |

---

## 6. Summary: Implementation Priority

Based on what's already in brepkit-math and what the operations layer needs, here is a
suggested implementation order.

### Phase 1: Core NURBS (unblocks surface evaluation and basic operations)

1. **Standalone BasisFuns** (A2.2) -- extract from De Boor, make reusable
2. **DersBasisFuns** (A2.3) -- basis function derivatives
3. **SurfacePoint** (A3.5) -- surface evaluation (currently `todo!()`)
4. **CurveDerivsAlg1** (A3.2) -- curve tangent vectors
5. **SurfaceDerivsAlg1** (A3.6) -- surface partials and normals
6. **RatCurveDerivs / RatSurfaceDerivs** (A4.2, A4.4) -- rational derivatives

### Phase 2: Manipulation (unblocks Boolean operations)

7. **CurveKnotIns** (A5.1) -- single knot insertion
8. **SurfaceKnotIns** (A5.3) -- surface knot insertion
9. **Bezier decomposition** -- via knot refinement to full multiplicity
10. **Curve splitting** -- insert knot to full multiplicity, split
11. **DegreeElevateCurve** (A5.9) -- degree elevation
12. **AABB computation** -- from control point extrema

### Phase 3: Intersection and Projection (unblocks Boolean classification)

13. **Point projection to curve** (A6.3-A6.4)
14. **Point projection to surface** (A6.5-A6.6)
15. **orient3d predicate** -- 3D point classification
16. **point_in_polygon_2d** -- trim curve classification
17. **Bezier clipping** -- curve-curve intersection
18. **Curve-surface intersection** -- Newton + subdivision
19. **BVH for surface patches** -- AABB tree

### Phase 4: Advanced (unblocks SSI, fillets, full Booleans)

20. **Surface-surface intersection** (marching + lattice seed points)
21. **Knot removal** (A5.8) -- simplification
22. **Surface creation** (ruled, revolved, swept, loft)
23. **Conic/circle to NURBS** -- exact representation
24. **Curvature computation** -- fundamental forms

---

## 7. Key References

### Books
- Piegl & Tiller, "The NURBS Book" 2nd ed., Springer 1997. The primary algorithm reference. All A-numbered algorithms come from here.
  https://link.springer.com/book/10.1007/978-3-642-59223-2

### Papers
- Shewchuk, "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates" (1997)
  https://people.eecs.berkeley.edu/~jrs/papers/robust-predicates.pdf
- Sederberg & Nishita, "Curve intersection using Bezier clipping" (1990)
  https://www.sciencedirect.com/science/article/abs/pii/001044859090039F
- Ma & Hewitt, "Point inversion and projection for NURBS curve and surface" (2003)
  https://www.sciencedirect.com/science/article/abs/pii/S0167839603000219
- Beer et al., "Algorithms for geometrical operations with NURBS surfaces" (2022)
  https://arxiv.org/pdf/2210.13160

### Open Source References
- LNLib (C++, matches NURBS Book algorithms): https://github.com/BIMCoderLiang/LNLib
- NURBS-Python (Python, implements A2.1-A5.x): https://github.com/jckchow/NURBS-Python
- OpenCASCADE modeling algorithms: https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_algos.html
- Rust `robust` crate (Shewchuk predicates): https://github.com/georust/robust

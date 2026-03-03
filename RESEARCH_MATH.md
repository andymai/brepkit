# Math Layer Research: Academic Papers and Numerical Methods

Research compiled 2026-03-02 for brepkit's L0 math layer.

This document surveys the academic literature and industrial publications relevant
to a B-Rep CAD modeling engine's math foundation. It is organized by topic, with
each section identifying key papers, their contributions, and concrete applicability
to brepkit's current and planned implementation.

---

## Table of Contents

1. [Exact and Adaptive Arithmetic](#1-exact-and-adaptive-arithmetic)
2. [Robust Geometric Predicates](#2-robust-geometric-predicates)
3. [Numerical Methods for Curve/Surface Intersection](#3-numerical-methods-for-curvesurface-intersection)
4. [Geometric Constraint Solving](#4-geometric-constraint-solving)
5. [Distance Computations](#5-distance-computations)
6. [Convex Hull, Voronoi, and Delaunay](#6-convex-hull-voronoi-and-delaunay)
7. [Floating-Point Geometry](#7-floating-point-geometry)
8. [Reference Libraries](#8-reference-libraries)
9. [Recommendations for brepkit](#9-recommendations-for-brepkit)

---

## 1. Exact and Adaptive Arithmetic

### 1.1 Shewchuk's Expansion Arithmetic

**Paper:** Jonathan Richard Shewchuk. "Adaptive Precision Floating-Point Arithmetic
and Fast Robust Geometric Predicates." *Discrete & Computational Geometry*,
18(3):305-363, 1997.

- Full paper: <https://people.eecs.berkeley.edu/~jrs/papers/robust-predicates.pdf>
- Homepage: <https://www.cs.cmu.edu/~quake/robust.html>
- Springer: <https://link.springer.com/article/10.1007/PL00009321>

**Key contributions:**
- Defines *expansion arithmetic*: represents exact values as non-overlapping sums
  of IEEE 754 doubles. Addition and multiplication are O(n) and O(nm) in expansion
  length.
- Introduces *adaptive precision*: a multi-stage pipeline where each stage uses
  more precision than the last, returning early when the error bound proves the
  sign is correct. Most geometric queries resolve in the fast first stage (plain
  doubles).
- Provides provably correct implementations of `orient2d`, `orient3d`, `incircle`,
  and `insphere` predicates.
- Requires IEEE 754 radix-2 arithmetic with exact rounding (standard on all modern
  hardware).

**Applicability to brepkit:** The codebase already uses the `robust` crate (a Rust
port of Shewchuk's predicates.c) for `orient2d` and `in_circle`. The expansion
arithmetic technique could be extended to build additional exact predicates beyond
the standard four (e.g., point-on-segment, edge classification, sign-of-determinant
for general matrices).

### 1.2 Exact Geometric Computation (EGC) Paradigm

**Key authors:** Chee K. Yap, Chen Li, Sylvain Pion

**Papers:**
- Chen Li, Sylvain Pion, Chee K. Yap. "Recent Progress in Exact Geometric
  Computation." *Journal of Logic and Algebraic Programming*, 64(1):85-111, 2005.
  <https://www.sciencedirect.com/science/article/pii/S1567832604000773>
- Chee K. Yap. "Robust Geometric Computation." Chapter 45, *Handbook of Discrete
  and Computational Geometry*, 3rd ed. <http://www.csun.edu/~ctoth/Handbook/chap45.pdf>
- CGAL's EGC overview: <https://www.cgal.org/exact.html>

**Key contributions:**
- Formalizes the principle that geometric algorithms should outsource all numerical
  decisions to a small set of *predicates* (orientation, incircle, etc.) and
  *constructions* (intersection points, circumcenters), each implemented to return
  provably correct results.
- Three pillars of EGC: (1) constructive zero bounds (knowing how small a nonzero
  result can be), (2) approximate evaluation with error tracking, (3) numerical
  filters that short-circuit exact computation when unnecessary.
- The "exact computation" is only applied to *sign decisions* -- the actual
  coordinates of constructed points can still be floating-point approximations as
  long as the topological decisions are correct.

**Applicability to brepkit:** The EGC paradigm suggests a clean architectural
separation: all topological decisions (is a point inside/outside a solid, which side
of a plane is a vertex on, does an edge cross a face) should be routed through
provably correct predicates. Coordinate construction (intersection points, projected
points) can remain floating-point. This is exactly the approach brepkit should take --
exact predicates for classification, tolerance-based approximation for geometry.

### 1.3 Cascaded/Staged Arithmetic Filters

**Papers:**
- Christoph Burnikel, Stefan Funke, Michael Seel. "Exact Geometric Predicates
  Using Cascaded Computation." *International Journal of Computational Geometry and
  Applications*, 2001.
  <https://www.semanticscholar.org/paper/Exact-geometric-predicates-using-cascaded-Burnikel-Funke/1406f605157c801532e0f971264bccdfb38291f0>
- Herve Bronnimann, Christoph Burnikel, Sylvain Pion. "Interval Arithmetic Yields
  Efficient Dynamic Filters for Computational Geometry." *Discrete Applied
  Mathematics*, 109(1-2):25-47, 2001.
- Stefan Funke. "Of What Use Is Floating-Point Arithmetic in Computational
  Geometry?" CSAIL MIT, 2009.
  <https://link.springer.com/chapter/10.1007/978-3-642-03456-5_23>

**Key contributions:**
- Demonstrates that interval arithmetic can serve as an efficient *filter* for
  geometric predicates: evaluate the predicate using interval arithmetic, and if the
  result interval does not contain zero, the sign is determined without exact
  computation.
- Cascaded computation chains multiple filter stages of increasing precision before
  falling back to full exact arithmetic.
- In practice, 95-99% of predicate evaluations are resolved by the fast filter
  stage.

**Applicability to brepkit:** If brepkit needs custom predicates beyond the standard
four (orient2d/3d, incircle/insphere), interval arithmetic filters provide a
practical implementation path. Rust's type system is well-suited to encoding the
filter stages as a generic trait with specializations.

---

## 2. Robust Geometric Predicates

### 2.1 Core Predicates

**Paper:** Shewchuk (1997), as above.

The four fundamental predicates are:

| Predicate | Dimension | Decides | Matrix Size |
|-----------|-----------|---------|-------------|
| orient2d  | 2D | Is point C left/right/on line AB? | 3x3 |
| orient3d  | 3D | Is point D above/below/on plane ABC? | 4x4 |
| incircle  | 2D | Is point D inside/outside/on circle(ABC)? | 4x4 |
| insphere  | 3D | Is point E inside/outside/on sphere(ABCD)? | 5x5 |

**Current state in brepkit:** `orient2d` and `incircle` are available via the
`robust` crate. `orient3d` and `insphere` are missing.

**Gap:** The `robust` crate (georust/robust) only provides 2D predicates. The
`robust-predicates` crate (<https://crates.io/crates/robust-predicates>) provides
all four, and the `geometry-predicates` crate
(<https://github.com/elrnv/geometry-predicates-rs>) provides all four plus the
underlying expansion arithmetic primitives for building custom predicates.

### 2.2 Simulation of Simplicity (Symbolic Perturbation)

**Paper:** Herbert Edelsbrunner, Ernst Peter Mucke. "Simulation of Simplicity: A
Technique to Cope with Degenerate Cases in Geometric Algorithms." *ACM Transactions
on Graphics*, 9(1):66-104, 1990.

- Paper: <https://dl.acm.org/doi/10.1145/77635.77639>
- PDF: <https://arxiv.org/pdf/math/9410209>

**Key contributions:**
- Provides a general technique to handle geometric degeneracies (collinear points,
  coplanar points, cocircular points) without special-case code.
- Adds infinitesimal symbolic perturbations to input coordinates, ensuring that no
  predicate ever evaluates to exactly zero. The perturbations are never actually
  applied; instead, their effect is computed analytically by examining lower-order
  terms of the predicate's polynomial expansion.
- Reduces the code complexity of geometric algorithms significantly -- no need for
  separate handling of degenerate configurations.

**Applicability to brepkit:** The current codebase handles degeneracies via tolerance
checks (`tolerance.approx_eq`). SoS would provide a more principled alternative for
predicates used in topological decisions (Boolean operations, point classification).
Geogram's PCK (see Section 8.2) implements SoS and could serve as a reference.

### 2.3 Additional Predicates Needed for B-Rep

Beyond the standard four, a B-Rep kernel needs:

| Predicate | Description | Used By |
|-----------|-------------|---------|
| point_in_polygon_2d | Winding-number or ray-casting in parameter space | Trimmed surface evaluation |
| segment_segment_2d | Classify 2D segment intersection (cross/touch/overlap/disjoint) | Trim curve processing |
| point_vs_plane | Signed distance, point-to-plane classification | BSP trees, Boolean ops |
| edge_classification | Classify edge as IN/OUT/ON relative to a solid | Boolean operations |

These are documented in the existing `crates/math/RESEARCH.md` file.

---

## 3. Numerical Methods for Curve/Surface Intersection

### 3.1 Bezier Clipping

**Paper:** Thomas W. Sederberg, Tomoyuki Nishita. "Curve Intersection Using Bezier
Clipping." *Computer-Aided Design*, 22(9):538-549, 1990.

- Paper: <https://www.sciencedirect.com/science/article/abs/pii/001044859090039F>
- Semantic Scholar: <https://www.semanticscholar.org/paper/Curve-intersection-using-B%C3%A9zier-clipping-Sederberg-Nishita/e5cdd6b4a1d0cc0b49264e66a13f635545fe78ed>

**Key contributions:**
- Formulates curve-curve intersection as root finding of a distance function
  expressed in Bernstein basis.
- Uses the convex hull property of Bezier curves to clip parameter intervals that
  cannot contain roots.
- Convergence rate: quadratic for transversal (simple) intersections, linear for
  tangential intersections.
- Guarantees finding *all* roots in the interval (unlike Newton-Raphson, which can
  miss roots or diverge).

**Follow-up -- Hybrid Clipping:**
- Liu, Ma, et al. "Curve Intersection Using Hybrid Clipping." *Shape Modeling
  International (SMI)*, 2012.
  <http://staff.ustc.edu.cn/~lgliu/Publications/Publications/2012_SMI_hybridclipping.pdf>
- Uses cubic fat arcs instead of linear envelopes for tighter clipping, achieving
  faster convergence.

**Applicability to brepkit:** The codebase currently uses Newton-Raphson for
intersections. Bezier clipping should replace it as the primary method for
curve-curve intersection because it is globally convergent and finds all roots.
Newton-Raphson should be retained as a final refinement step after Bezier clipping
narrows the interval.

### 3.2 Projected Polyhedron (Interval Projected Polyhedron)

**Authors:** Nicholas M. Patrikalakis, Takashi Maekawa, Wonjoon Cho

**Reference:** *Shape Interrogation for Computer Aided Design and Manufacturing*,
MIT hyperbook.

- Projected Polyhedron algorithm: <https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node42.html>
- Curve/surface intersection: <https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node89.html>

**Key contributions:**
- An n-dimensional root-finding algorithm based on the convex hull property of
  Bernstein polynomials.
- Recasts continuous intersection problems into discrete convex hull computations:
  project the Bernstein control polygon onto coordinate axes, compute convex hull
  intersections, and use these to clip parameter domains.
- Handles overconstrained problems naturally.
- Foundation for robust surface-surface intersection: used to find seed points,
  followed by marching methods to trace intersection curves.

**Applicability to brepkit:** The current SSI implementation uses grid sampling
plus Newton refinement. The Projected Polyhedron method would provide more reliable
seed-point computation by exploiting the convex hull property of NURBS, reducing
the chance of missing intersection branches.

### 3.3 Surface-Surface Intersection (SSI) via Marching

**Paper:** Shankar Krishnan, Dinesh Manocha. "An Efficient Surface Intersection
Algorithm Based on Lower-Dimensional Formulation." *ACM Transactions on Graphics*,
16(1):74-106, 1997.
<https://dl.acm.org/doi/10.1145/237748.237751>

**Related:**
- Laurent Busé, et al. "Surface-to-Surface Intersections."
  *Computer-Aided Design Journal*, 1(1-4):449-457, 2004.
  <https://www.cad-journal.net/files/vol_1/CAD_1(1-4)_2004_449-457.pdf>

**Key contributions:**
- Lower-dimensional formulation reduces SSI to a set of 1D root-finding problems,
  avoiding the instability of full 3D marching methods.
- Marching methods step along the intersection curve by solving an ODE:
  the tangent direction is the cross product of the two surface normals,
  and the step is constrained to lie on both surfaces via Newton correction.
- Seed points are the critical input; the quality of the SSI result depends
  entirely on finding all connected components of the intersection curve.

**Applicability to brepkit:** The current `intersection.rs` uses alternating
projection plus marching. The lower-dimensional formulation and explicit ODE-based
marching could improve robustness, especially near tangential intersections where
the cross product of normals approaches zero.

### 3.4 Subdivision Methods

**Reference:** Piegl and Tiller. *The NURBS Book*, 2nd ed. Springer, 1997.
<https://link.springer.com/book/10.1007/978-3-642-59223-2>

Subdivision is the backbone of all NURBS intersection methods:

1. Decompose NURBS into Bezier segments/patches (knot insertion to full multiplicity).
2. Recursively subdivide, testing bounding volume overlap at each level.
3. When patches are small enough, apply Newton-Raphson or Bezier clipping for
   final refinement.

The convex hull property guarantees that bounding volumes tighten with each
subdivision, so the process converges.

---

## 4. Geometric Constraint Solving

### 4.1 Survey of Approaches

**Paper:** Christoph M. Hoffmann. "Geometric Constraint Solving in Parametric
Computer-Aided Design." *Journal of Computing and Information Science in
Engineering*, 5(2):69-74, 2005.
<https://www.researchgate.net/publication/270766236_Geometric_Constraint_Solving_in_Parametric_Computer-Aided_Design>

**Paper:** Christophe Jermann, et al. "Decomposition of Geometric Constraint
Systems: A Survey." *International Journal of Computational Geometry and
Applications*, 16(5-6):379-414, 2006.
<https://hal.science/hal-00481267/document>

**Paper:** Hoffmann. "Summary of Basic 2D Constraint Solving." *International
Journal of Product Lifecycle Management*, 2006.
<https://www.cs.purdue.edu/cgvlab/www/resources/papers/Hoffmann-IJPLM-2006-Summary_of_Basis_2D_Constaint_Solving.pdf>

**Key approaches:**

| Method | Description | Strengths | Weaknesses |
|--------|-------------|-----------|------------|
| Numerical (Newton-Raphson) | Solve F(x)=0 system directly | Simple to implement | May diverge, misses solutions, no decomposition |
| Graph-based decomposition | Map constraints to a graph, decompose into rigid subclusters, solve independently | Scales to industrial problems, exploits structure | Complex implementation, some configurations not decomposable |
| Geometric reasoning | Apply geometric rules (e.g., "two distances from fixed points = circle intersection") | Intuitive, teaches well | Hard to generalize, limited expressiveness |
| Algebraic (Grobner bases) | Symbolic elimination of variables | Complete for algebraic constraints | Exponential complexity for large systems |

### 4.2 Graph-Based Decomposition (Recommended Direction)

**Paper:** Christophe Jermann, Gilles Trombettoni, Bertrand Neveu. "Geometric
Constraint Solving via C-tree Decomposition." *Proceedings ACM Solid Modeling*,
2003.
<https://dl.acm.org/doi/10.1145/781606.781617>

**Key contributions:**
- Translates the constraint problem into a graph where vertices are geometric
  elements and edges are constraints.
- Decomposes the graph into a tree of rigid subclusters, each small enough to solve
  independently (often 2-3 elements).
- Recomposes solved subclusters to obtain the global solution.
- Dramatically reduces the effective problem size for industrial-scale sketches.

**Applicability to brepkit:** The current sketch solver (`sketch.rs`) uses pure
Newton-Raphson on the full system of 9 constraint types. For small sketches this
works, but it will not scale to complex parametric models. Graph-based decomposition
would allow solving each rigid subcluster independently and is the approach used by
production CAD systems (SolveSpace, FreeCAD's Sketcher/PlaneGCS). This is the
single most impactful improvement path for the constraint solver.

### 4.3 References for Implementation

- SolveSpace constraint solver (open source, C++):
  <https://github.com/solvespace/solvespace>
- FreeCAD PlaneGCS (open source, C++):
  <https://github.com/FreeCAD/FreeCAD/tree/main/src/Mod/Sketcher/App/planegcs>
- Both use graph decomposition + numerical solving of subclusters.

---

## 5. Distance Computations

### 5.1 Minimum Distance Between Bezier Curves/Surfaces

**Paper:** Young-Taek Chang, Hyun-Chul Kang, Myung-Soo Kim. "Computation of the
Minimum Distance Between Two Bezier Curves/Surfaces." *Computers & Graphics*,
35(3):677-684, 2011.
<https://www.sciencedirect.com/science/article/abs/pii/S0097849311000641>

**Paper:** Eunhye Chang, Myung-Soo Kim. "Computing the Minimum Distance Between
Two Bezier Curves." *Journal of Computational and Applied Mathematics*,
229(1):294-301, 2009.
<https://www.sciencedirect.com/science/article/pii/S0377042708005785>

**Key contributions:**
- Culling-based approach: uses bounding volumes (sweeping spheres, cone-spheres)
  to prune parameter regions that cannot contain the closest point pair.
- Avoids the problem of Newton-Raphson getting trapped in local minima of the
  distance function.
- Global convergence guaranteed by the subdivision-and-cull strategy.

**Applicability to brepkit:** The current `measure` module computes bounding box,
area, volume, and center of mass. Minimum distance between entities is a natural
next addition. The culling approach is directly applicable to NURBS after Bezier
decomposition.

### 5.2 Point Projection onto NURBS

**Paper:** Y. Ma, W.T. Hewitt. "Point Inversion and Projection for NURBS Curve and
Surface: Control Polygon Approach." *Computer Aided Geometric Design*,
20(2):79-99, 2003.
<https://www.sciencedirect.com/science/article/abs/pii/S0167839603000219>

**Paper:** S. Hu, J. Wallner. "A Geometric Strategy Algorithm for Orthogonal
Projection onto a Parametric Surface." *Journal of Computer Science and Technology*,
2019.
<https://link.springer.com/article/10.1007/s11390-019-1967-z>

**Key contributions (Ma & Hewitt):**
1. Decompose NURBS surface into Bezier patches.
2. Use control polygon geometry to extract candidate patches and approximate
   closest points.
3. Refine with Newton-Raphson.
4. Handles multiple local minima correctly (the global minimum is among the
   candidates from step 2).

**Applicability to brepkit:** Point projection is the most-called operation in a
B-Rep kernel (used by classification, snapping, tessellation, distance queries).
The control-polygon approach provides reliable initial guesses for Newton iteration,
which is critical since the distance function often has multiple local minima.

### 5.3 Hausdorff Distance

**Paper:** M. Kim, Y. Oh, et al. "Efficient Hausdorff Distance Computation for
Freeform Geometric Models in Close Proximity." *Computer-Aided Design*,
45(2):251-262, 2013.
<https://www.sciencedirect.com/science/article/abs/pii/S0010448512002151>

**Paper:** S. Krishnamurthy, et al. "GPU-Accelerated Hausdorff Distance Computation
Between Dynamic Deformable NURBS Surfaces." *Computer-Aided Design*,
44(2):140-152, 2012.
<https://www.sciencedirect.com/science/article/abs/pii/S0010448511002211>

**Paper:** H. Elber, et al. "Hausdorff and Minimal Distances Between Parametric
Freeforms in R^2 and R^3." Springer, 2008.
<https://link.springer.com/chapter/10.1007/978-3-540-79246-8_15>

**Key contributions:**
- BVH-based culling with two-sided pruning: simultaneously eliminate bounding-box
  pairs from both surfaces that cannot contribute to the Hausdorff distance.
- Coons patch approximation of NURBS patches for cheaper intermediate distance
  bounds.
- Polyline approximation method for curves: approximate curves as polylines, compute
  exact Hausdorff distance between polylines as an approximation with bounded error.

**Applicability to brepkit:** Hausdorff distance is essential for validating
geometric operations (e.g., verifying that a simplified/approximated surface is
within tolerance of the original). It would be a valuable addition to the `measure`
module.

---

## 6. Convex Hull, Voronoi, and Delaunay

### 6.1 Convex Hull

**Paper:** Timothy M. Chan. "Optimal Output-Sensitive Convex Hull Algorithms in Two
and Three Dimensions." *Discrete & Computational Geometry*, 16(4):361-368, 1996.
<https://link.springer.com/article/10.1007/BF02712873>

**Key contributions:**
- O(n log h) output-sensitive algorithm for 2D and 3D convex hulls, where h is the
  number of hull vertices.
- Simpler than the earlier Kirkpatrick-Seidel algorithm.
- Combines Graham scan (O(n log n) for small groups) with Jarvis march (output-
  sensitive for combining groups).

**Applicability to brepkit:** Convex hull is needed for: (1) NURBS bounding volume
computation (convex hull of control points), (2) Bezier clipping (convex hull of
control polygon projected onto an axis), (3) general collision/proximity queries.
For NURBS bounding, the simpler gift-wrapping or quickhull algorithms suffice since
control polygon sizes are small.

### 6.2 Delaunay Triangulation and Mesh Generation

**Paper:** Jonathan Richard Shewchuk. "Triangle: Engineering a 2D Quality Mesh
Generator and Delaunay Triangulator." In *Applied Computational Geometry: Towards
Geometric Engineering*, Springer LNCS 1148, pp.203-222, 1996.
<https://link.springer.com/chapter/10.1007/BFb0014497>

**Paper (Bowyer-Watson):** Paul L. George, Houman Borouchaki. "Efficient
Unstructured Mesh Generation by Means of Delaunay Triangulation and Bowyer-Watson
Algorithm." *Journal of Computational Physics*, 106(1):125-138, 1993.
<https://www.sciencedirect.com/science/article/abs/pii/S0021999183710971>

**Key contributions:**
- **Triangle** is the reference implementation for 2D constrained Delaunay
  triangulation (CDT) with Ruppert's refinement for quality mesh generation.
- Uses Shewchuk's own exact predicates for robustness.
- The **Bowyer-Watson** algorithm provides incremental Delaunay triangulation:
  insert points one at a time, remove invalidated triangles, re-triangulate the
  cavity. O(n log n) expected time.
- CDT forces specified edges into the triangulation while maintaining the Delaunay
  property elsewhere -- essential for triangulating trimmed NURBS domains.

**Applicability to brepkit:** CDT is needed for trimmed surface tessellation:
trim curves define constrained edges in 2D parameter space, and CDT produces a
quality triangulation of the trimmed domain. The current tessellation approach
could be improved by using a proper CDT rather than simple subdivision.

### 6.3 Voronoi Diagrams for Offset Computation

**Reference:** Martin Held. VRONI project: Voronoi diagrams of points, segments,
and circular arcs.
<https://www.cosy.sbg.ac.at/~held/projects/vroni/vroni.html>

**Paper:** Martin Held. "An Experimental Evaluation of Offset Computation for
Polygons." *Computer-Aided Design Journal*, 21(5):807-818, 2024.
<https://cad-journal.net/files/vol_21/CAD_21(5)_2024_807-818.pdf>

**Key contributions:**
- Voronoi diagrams provide exact offset curves for polygonal inputs: the offset
  at distance d consists of arcs centered on Voronoi edges/vertices at distance d
  from the input boundary.
- More robust than the naive approach of offsetting each edge and computing
  intersections.

**Applicability to brepkit:** The current `offset_face.rs` and `offset_solid.rs`
use planar (exact) and NURBS (sampling + refit) approaches. Voronoi-based offsetting
would provide an alternative for polygonal/polyline inputs with guaranteed
correctness.

---

## 7. Floating-Point Geometry

### 7.1 The Robustness Problem

**Paper:** Christoph M. Hoffmann, John E. Hopcroft, Michael S. Karasick. "Towards
Implementing Robust Geometric Computations." *Proceedings of the Fourth Annual
Symposium on Computational Geometry*, pp.106-117, 1988.
<https://dl.acm.org/doi/10.1145/73393.73405>

**Paper:** Christoph M. Hoffmann, John E. Hopcroft, Michael S. Karasick. "Robust
Set Operations on Polyhedral Solids." *IEEE Computer Graphics and Applications*,
9(6):50-59, 1989.

**Paper:** Hoffmann. "A Paradigm for Robust Geometric Algorithms." *Algorithmica*,
1993.
<https://link.springer.com/article/10.1007/BF01758769>

**Key contributions:**
- Formalizes the distinction between *model* (abstract geometric object) and
  *representation* (finite-precision approximation in computer memory).
- An algorithm is *robust* if for every input representation, there exists some
  valid input model such that the output representation corresponds to the correct
  output model. This is the "backward error" notion of robustness.
- Key tool: perturbing embedding geometry in ways consistent with topology. That is,
  if the topology says two edges intersect, ensure the geometry is adjusted so that
  they actually do, even if floating-point error would otherwise make them miss.

**Applicability to brepkit:** This is the foundational theoretical framework for
B-Rep robustness. The principle "topology drives geometry" (rather than the reverse)
should guide brepkit's Boolean and intersection implementations. When floating-point
intersection computations produce results that are topologically inconsistent,
the geometry should be *snapped* to match the topological decision, not the other
way around.

### 7.2 Shewchuk's Robustness Lecture Notes

**Reference:** Jonathan Richard Shewchuk. "Lecture Notes on Geometric Robustness."
<https://perso.uclouvain.be/jean-francois.remacle/LMECA2170/robnotes.pdf>

An accessible overview of:
- How floating-point arithmetic fails for geometric predicates.
- The hierarchy of approaches: exact arithmetic, adaptive arithmetic, epsilon
  tweaking (and why epsilon tweaking is fragile).
- Why "just add a tolerance" is not a reliable solution for topological decisions.

**Applicability to brepkit:** The current tolerance-based approach
(`tolerance.approx_eq`) is appropriate for geometric computations (point positions,
distances) but should *not* be relied upon for topological decisions (inside/outside
classification, edge intersection existence). Those require exact or adaptive
predicates.

### 7.3 Watertight Ray-Triangle Intersection

**Paper:** Sven Woop, Carsten Benthin, Ingo Wald. "Watertight Ray/Triangle
Intersection." *Journal of Computer Graphics Techniques (JCGT)*, 2(1):65-82, 2013.
<https://jcgt.org/published/0002/01/05/paper.pdf>

**Key contributions:**
- A ray-triangle intersection algorithm that is *watertight*: for any ray that
  passes through an edge or vertex shared by adjacent triangles, the algorithm
  reports intersection with exactly one triangle (or both, in the edge case).
- Achieves watertightness by transforming the problem so that the ray direction
  is axis-aligned, then using a consistent 2D edge test that avoids ambiguity at
  shared edges.
- Same performance as non-watertight methods.

**Applicability to brepkit:** The current implementation uses Moller-Trumbore, which
is not watertight. For point-in-solid classification via ray casting (the
`classify.rs` module), watertight intersection is important: a ray passing through
a mesh edge should not be double-counted or missed. Switching to Woop-Benthin-Wald
would improve robustness of the classifier.

### 7.4 Moller-Trumbore (Current Implementation)

**Paper:** Tomas Moller, Ben Trumbore. "Fast, Minimum Storage Ray/Triangle
Intersection." *Journal of Graphics Tools*, 2(1):21-28, 1997.
<https://www.tandfonline.com/doi/abs/10.1080/10867651.1997.10487468>

- The standard fast ray-triangle intersection algorithm.
- No precomputation required (computes plane equation on the fly).
- Uses Cramer's rule to solve the ray-triangle system.
- Not watertight at edges/vertices.

**Current state:** Already implemented in brepkit for the `classify_point` ray
casting classifier.

---

## 8. Reference Libraries

### 8.1 CGAL (Computational Geometry Algorithms Library)

**Paper:** Andreas Fabri, et al. "On the Design of CGAL, a Computational Geometry
Algorithms Library." *Software: Practice and Experience*, 30(11):1167-1202, 2000.
<https://onlinelibrary.wiley.com/doi/pdf/10.1002/1097-024X(200009)30:11%3C1167::AID-SPE337%3E3.0.CO;2-B>

- Homepage: <https://www.cgal.org/>
- GitHub: <https://github.com/CGAL/cgal>
- EGC philosophy: <https://www.cgal.org/exact.html>

**Algorithmic foundations:** CGAL implements the EGC paradigm throughout. All
geometric predicates are exact (using LEDA or GMP for exact arithmetic, with
interval arithmetic filters). The *kernel* concept separates geometric types from
algorithms, allowing different number types (double, exact, interval) to be plugged
in.

**Relevant algorithms:** Triangulations (2D/3D Delaunay, constrained, conforming),
Voronoi diagrams, convex hulls (2D/3D), Boolean operations on polyhedra (Nef
polyhedra), AABB trees, point set processing, surface mesh generation.

**Lessons for brepkit:** CGAL's architecture demonstrates that exact predicates and
approximate constructions can coexist cleanly. The kernel abstraction (separating
number type from algorithm) is a powerful pattern that Rust's generics could express
well.

### 8.2 Geogram

**Paper:** Bruno Levy. "Robustness and Efficiency of Geometric Programs: The
Predicate Construction Kit (PCK)." *Computer-Aided Design*, 72:3-12, 2016.
<https://www.sciencedirect.com/science/article/abs/pii/S0010448515001578>

- GitHub: <https://github.com/BrunoLevy/geogram>
- Publications: <https://github.com/BrunoLevy/geogram/wiki/Publications>

**Key features:**
- PCK (Predicate Construction Kit): auto-generates robust predicate code from
  formulas, including arithmetic filters + expansion arithmetic + SoS perturbation.
- State-of-the-art mesh Boolean operations using exact predicates throughout.
- Constrained Delaunay triangulation that handles intersecting constraints in exact
  precision.
- Rust bindings exist: `geogram_predicates` crate
  (<https://github.com/glennDittmann/geogram_predicates>).

**Lessons for brepkit:** Geogram demonstrates that a *complete* robustness solution
requires three components working together: (1) arithmetic filters for speed,
(2) expansion arithmetic for correctness, (3) symbolic perturbation for degeneracy
handling. The Rust bindings provide a practical path to integrating Geogram's
predicates if needed.

### 8.3 libigl

**Paper:** Alec Jacobson, et al. "libigl: Prototyping Geometry Processing Research
in C++." *SIGGRAPH Asia Courses*, 2017.
<https://dl.acm.org/doi/10.1145/3134472.3134497>

- Homepage: <https://libigl.github.io/>
- GitHub: <https://github.com/libigl/libigl>
- Tutorial: <https://libigl.github.io/tutorial/>

**Relevant algorithms:** Mesh Booleans (using exact predicates from Shewchuk or
Geogram), cotangent Laplacian and other discrete differential geometry operators,
parameterization, remeshing, winding number computation for inside/outside tests.

**Lessons for brepkit:** libigl's winding number approach for solid classification
is an alternative to ray casting: compute the generalized winding number of a point
with respect to a mesh, where values near 1 are inside and near 0 are outside. This
is more robust than ray casting for non-manifold or open meshes.

---

## 9. Recommendations for brepkit

Based on the research above, here are prioritized recommendations for improving
brepkit's math layer, organized by impact and implementation effort.

### High Priority (directly addresses current weaknesses)

#### R1. Switch to `robust-predicates` or `geometry-predicates` crate

**Why:** The current `robust` crate only provides 2D predicates. `orient3d` is
critical for 3D point classification in Boolean operations and `insphere` is needed
for 3D Delaunay. The `geometry-predicates` crate additionally exposes expansion
arithmetic primitives for building custom predicates.

- `robust-predicates`: <https://crates.io/crates/robust-predicates>
- `geometry-predicates`: <https://crates.io/crates/geometry-predicates>

**Effort:** Low (crate swap + add orient3d/insphere to predicates module).

#### R2. Implement Bezier clipping for curve-curve intersection

**Why:** Newton-Raphson is the current method and can miss roots or diverge. Bezier
clipping is globally convergent and finds all roots. This directly improves Boolean
operation reliability.

**Reference:** Sederberg & Nishita (1990).

**Effort:** Medium. Requires Bezier decomposition (already in `decompose.rs`) and
convex hull of 2D control polygon.

#### R3. Separate exact predicates from tolerance-based geometry

**Why:** The codebase currently uses `tolerance.approx_eq` for both geometric
computations and topological decisions. Topological decisions (inside/outside, does
edge cross face) need exact predicates to avoid inconsistent results. Geometric
computations (intersection point coordinates) can remain tolerance-based.

**Effort:** Low-Medium (architectural refactor of call sites, not new algorithms).

#### R4. Implement watertight ray-triangle intersection (Woop-Benthin-Wald)

**Why:** The current Moller-Trumbore implementation in `classify.rs` is not
watertight. For point-in-solid classification, a ray passing through a mesh edge can
produce incorrect results.

**Reference:** Woop, Benthin, Wald (2013).

**Effort:** Low (replace the core intersection routine, ~100 lines of code).

### Medium Priority (enables significant new capabilities)

#### R5. Implement Projected Polyhedron for SSI seed finding

**Why:** The current SSI uses grid sampling for seed points, which can miss
intersection branches. The IPP method exploits NURBS convex hull properties for
more reliable seed detection.

**Reference:** Patrikalakis, Maekawa, Cho. MIT hyperbook.

**Effort:** Medium-High. Requires 2D convex hull computation and integration with
the marching module.

#### R6. Add graph-based decomposition to the constraint solver

**Why:** The current Newton-Raphson solver will not scale to complex sketches. Graph
decomposition is the approach used by all production CAD constraint solvers.

**References:** Jermann et al. (2003), SolveSpace, FreeCAD PlaneGCS.

**Effort:** High. Requires building a constraint graph, implementing cluster
detection (e.g., Sitharam's algorithm or simple rigid-body detection), and solving
subclusters independently.

#### R7. Add minimum distance computation (culling-based)

**Why:** Entity-to-entity minimum distance is a fundamental CAD query used by
proximity detection, clearance checks, and interference analysis. The current
codebase lacks it.

**Reference:** Chang et al. (2011), Ma & Hewitt (2003) for point projection.

**Effort:** Medium. Build on existing BVH infrastructure plus Bezier decomposition.

#### R8. Implement constrained Delaunay triangulation for trimmed surfaces

**Why:** Proper tessellation of trimmed NURBS surfaces requires CDT in parameter
space. The current approach may produce poor-quality triangulations near trim
boundaries.

**Reference:** Shewchuk's Triangle (1996), Bowyer-Watson algorithm.

**Effort:** Medium-High. A robust CDT implementation is nontrivial, though Rust
crates like `cdt` or `spade` could be evaluated.

### Lower Priority (polish and advanced features)

#### R9. Add Simulation of Simplicity for degenerate handling

**Why:** Provides principled handling of geometric degeneracies without special-case
code. Reduces code complexity and improves reliability.

**Reference:** Edelsbrunner & Mucke (1990), Levy PCK (2016).

**Effort:** High. Requires modifying predicate implementations.

#### R10. Implement Hausdorff distance computation

**Why:** Valuable for validating geometric operations (comparing original vs.
approximated surfaces) and quality metrics.

**References:** Kim et al. (2013), Krishnamurthy et al. (2012).

**Effort:** Medium. Build on existing BVH and point projection.

#### R11. Add interval arithmetic number type

**Why:** Enables building custom robust predicates and provides guaranteed bounds
on function evaluations.

**References:** Bronnimann et al. (2001), Boost.Interval.

**Effort:** Medium. Implement an `Interval` type wrapping `[f64; 2]` with correct
rounding modes.

---

## Appendix: Existing Implementations Already in brepkit

For reference, these are the methods already implemented that are covered by the
literature above:

| Method | File | Reference |
|--------|------|-----------|
| Moller-Trumbore ray-triangle | `classify.rs` | Moller & Trumbore (1997) |
| Signed tetrahedra volume | `measure` module | Divergence theorem |
| Newton-Raphson root finding | `intersection.rs`, `sketch.rs` | Standard |
| orient2d, incircle | `predicates.rs` (via `robust` crate) | Shewchuk (1997) |
| NURBS evaluation (De Boor) | `nurbs/curve.rs` | Piegl & Tiller, A3.1 |
| BVH (AABB tree) | `bvh.rs` | Standard |
| AABB computation | `aabb.rs` | Standard |
| Bezier decomposition | `nurbs/decompose.rs` | Piegl & Tiller, Ch. 5 |
| Point projection (NURBS) | `nurbs/projection.rs` | Ma & Hewitt (2003) |
| SSI (alternating projection + marching) | `nurbs/intersection.rs` | Patrikalakis & Maekawa |
| Surface fitting | `nurbs/surface_fitting.rs` | Piegl & Tiller, Ch. 9 |

---

## Appendix: Complete Bibliography

### Books

1. **Piegl, L. and Tiller, W.** *The NURBS Book*, 2nd ed. Springer, 1997.
   <https://link.springer.com/book/10.1007/978-3-642-59223-2>

2. **Patrikalakis, N.M., Maekawa, T., and Cho, W.** *Shape Interrogation for
   Computer Aided Design and Manufacturing*. MIT, 2002.
   <https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node1.html>

3. **Sederberg, T.W.** *Computer Aided Geometric Design* (course notes). BYU, 2012.
   <https://scholarsarchive.byu.edu/facpub/1/>

### Papers -- Arithmetic and Predicates

4. **Shewchuk, J.R.** "Adaptive Precision Floating-Point Arithmetic and Fast Robust
   Geometric Predicates." *DCG*, 18(3):305-363, 1997.
   <https://link.springer.com/article/10.1007/PL00009321>

5. **Edelsbrunner, H. and Mucke, E.P.** "Simulation of Simplicity." *ACM TOG*,
   9(1):66-104, 1990.
   <https://dl.acm.org/doi/10.1145/77635.77639>

6. **Levy, B.** "Robustness and Efficiency of Geometric Programs: The Predicate
   Construction Kit (PCK)." *CAD*, 72:3-12, 2016.
   <https://www.sciencedirect.com/science/article/abs/pii/S0010448515001578>

7. **Li, C., Pion, S., and Yap, C.K.** "Recent Progress in Exact Geometric
   Computation." *JLAP*, 64(1):85-111, 2005.
   <https://www.sciencedirect.com/science/article/pii/S1567832604000773>

8. **Burnikel, C., Funke, S., and Seel, M.** "Exact Geometric Predicates Using
   Cascaded Computation." *IJCGA*, 2001.

### Papers -- Intersection and Root Finding

9. **Sederberg, T.W. and Nishita, T.** "Curve Intersection Using Bezier Clipping."
   *CAD*, 22(9):538-549, 1990.
   <https://www.sciencedirect.com/science/article/abs/pii/001044859090039F>

10. **Krishnan, S. and Manocha, D.** "An Efficient Surface Intersection Algorithm
    Based on Lower-Dimensional Formulation." *ACM TOG*, 16(1):74-106, 1997.
    <https://dl.acm.org/doi/10.1145/237748.237751>

11. **Liu, L., et al.** "Curve Intersection Using Hybrid Clipping." *SMI*, 2012.
    <http://staff.ustc.edu.cn/~lgliu/Publications/Publications/2012_SMI_hybridclipping.pdf>

### Papers -- Distance and Projection

12. **Ma, Y. and Hewitt, W.T.** "Point Inversion and Projection for NURBS Curve
    and Surface: Control Polygon Approach." *CAGD*, 20(2):79-99, 2003.
    <https://www.sciencedirect.com/science/article/abs/pii/S0167839603000219>

13. **Chang, Y.-T., et al.** "Computation of the Minimum Distance Between Two
    Bezier Curves/Surfaces." *C&G*, 35(3):677-684, 2011.
    <https://www.sciencedirect.com/science/article/abs/pii/S0097849311000641>

14. **Kim, M., et al.** "Efficient Hausdorff Distance Computation for Freeform
    Geometric Models in Close Proximity." *CAD*, 45(2):251-262, 2013.
    <https://www.sciencedirect.com/science/article/abs/pii/S0010448512002151>

### Papers -- Robustness

15. **Hoffmann, C.M., Hopcroft, J.E., and Karasick, M.S.** "Towards Implementing
    Robust Geometric Computations." *SCG*, pp.106-117, 1988.
    <https://dl.acm.org/doi/10.1145/73393.73405>

16. **Hoffmann, C.M., Hopcroft, J.E., and Karasick, M.S.** "A Paradigm for Robust
    Geometric Algorithms." *Algorithmica*, 1993.
    <https://link.springer.com/article/10.1007/BF01758769>

### Papers -- Ray Intersection

17. **Moller, T. and Trumbore, B.** "Fast, Minimum Storage Ray/Triangle
    Intersection." *JGT*, 2(1):21-28, 1997.
    <https://www.tandfonline.com/doi/abs/10.1080/10867651.1997.10487468>

18. **Woop, S., Benthin, C., and Wald, I.** "Watertight Ray/Triangle Intersection."
    *JCGT*, 2(1):65-82, 2013.
    <https://jcgt.org/published/0002/01/05/paper.pdf>

### Papers -- Constraint Solving

19. **Jermann, C., Trombettoni, G., and Neveu, B.** "Geometric Constraint Solving
    via C-tree Decomposition." *ACM Solid Modeling*, 2003.
    <https://dl.acm.org/doi/10.1145/781606.781617>

20. **Jermann, C., et al.** "Decomposition of Geometric Constraint Systems: A
    Survey." *IJCGA*, 16(5-6):379-414, 2006.
    <https://hal.science/hal-00481267/document>

21. **Hoffmann, C.M.** "Geometric Constraint Solving in Parametric CAD." *JCISE*,
    5(2):69-74, 2005.

### Papers -- Triangulation and Meshing

22. **Shewchuk, J.R.** "Triangle: Engineering a 2D Quality Mesh Generator and
    Delaunay Triangulator." *LNCS 1148*, pp.203-222, 1996.
    <https://link.springer.com/chapter/10.1007/BFb0014497>

23. **Chan, T.M.** "Optimal Output-Sensitive Convex Hull Algorithms in Two and
    Three Dimensions." *DCG*, 16(4):361-368, 1996.
    <https://link.springer.com/article/10.1007/BF02712873>

### Rust Crates

24. `robust` -- 2D Shewchuk predicates: <https://github.com/georust/robust>
25. `robust-predicates` -- All 4 Shewchuk predicates: <https://crates.io/crates/robust-predicates>
26. `geometry-predicates` -- Predicates + expansion arithmetic: <https://github.com/elrnv/geometry-predicates-rs>
27. `geogram_predicates` -- Geogram predicates via FFI: <https://github.com/glennDittmann/geogram_predicates>

### Open Source Libraries

28. **CGAL**: <https://www.cgal.org/>
29. **Geogram**: <https://github.com/BrunoLevy/geogram>
30. **libigl**: <https://libigl.github.io/>
31. **SolveSpace** (constraint solver): <https://github.com/solvespace/solvespace>
32. **FreeCAD PlaneGCS**: <https://github.com/FreeCAD/FreeCAD/tree/main/src/Mod/Sketcher/App/planegcs>

# NURBS Algorithms Research for brepkit

Comprehensive survey of academic papers, books, and open-source implementations
relevant to improving brepkit's NURBS capabilities. Organized by topic with
specific recommendations for the codebase.

**Date**: 2026-03-02
**Scope**: NURBS evaluation, SSI, trimming, booleans, fitting, advanced representations

---

## Table of Contents

1. [Foundational References](#1-foundational-references)
2. [NURBS Evaluation and Fundamental Algorithms](#2-nurbs-evaluation-and-fundamental-algorithms)
3. [Surface-Surface Intersection (SSI)](#3-surface-surface-intersection-ssi)
4. [NURBS Trimming and Splitting](#4-nurbs-trimming-and-splitting)
5. [NURBS Boolean Operations](#5-nurbs-boolean-operations)
6. [NURBS Fitting and Approximation](#6-nurbs-fitting-and-approximation)
7. [Advanced Representations: T-Splines and Hierarchical B-Splines](#7-advanced-representations-t-splines-and-hierarchical-b-splines)
8. [Open-Source Implementations](#8-open-source-implementations)
9. [Recommendations for brepkit](#9-recommendations-for-brepkit)

---

## 1. Foundational References

### The NURBS Book
- **Authors**: Les Piegl, Wayne Tiller
- **Publisher**: Springer-Verlag, 2nd Edition, 1997 (646 pages, 578 illustrations)
- **URL**: https://cse.usf.edu/~lespiegl/nurbs.htm
- **Relevance**: The definitive implementation reference. brepkit already references
  algorithm numbers from this book (A2.1, A2.2, A5.1) in `basis.rs` and `knot_ops.rs`.

Key chapters and their algorithms:
| Chapter | Topic | Key Algorithms |
|---------|-------|---------------|
| Ch 2 | B-spline basis functions | A2.1 (find span), A2.2 (basis funs), A2.3 (derivatives) |
| Ch 3 | B-spline curves/surfaces | A3.1-A3.8 (evaluation, derivatives) |
| Ch 4 | Rational curves/surfaces | A4.1-A4.5 (NURBS evaluation) |
| Ch 5 | Fundamental algorithms | A5.1 (knot insert), A5.2 (refine), A5.3 (decompose), A5.4 (remove), A5.5-A5.9 (degree elevate/reduce) |
| Ch 6 | Surface construction | Lofting, sweeping, skinning |
| Ch 9 | Shape modification | Weight manipulation, control point repositioning |

**brepkit status**: Implements A2.1, A2.2, A5.1 (knot insertion), A5.3 (Bezier decomposition),
degree elevation. Missing: A5.4 (knot removal with error bounds), A2.3 (basis function
derivatives as standalone), degree reduction.

### Shape Interrogation for Computer Aided Design and Manufacturing
- **Authors**: Nicholas M. Patrikalakis, Takashi Maekawa, Wonjoon Cho
- **Publisher**: Springer (also available as MIT Hyperbook)
- **URL**: https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/
- **Relevance**: Comprehensive treatment of intersection algorithms, distance functions,
  offset curves/surfaces, geodesics. Freely available online.

Key contributions:
- Unifying framework: all shape interrogation problems recast as nonlinear system solving
- Detailed SSI marching method with starting point strategies (Section 5.8.2.3)
- Collinear normal detection for closed intersection loops
- Treatment of singular/degenerate intersection cases
- Offset surface self-intersection analysis

**brepkit status**: The intersection module uses grid sampling + Newton refinement, which
is a simplified version of the lattice/subdivision approach described here. The marching
method in the book is more rigorous with proper ODE integration.

---

## 2. NURBS Evaluation and Fundamental Algorithms

### De Boor's Algorithm
- **Author**: Carl de Boor
- **Reference**: "A Practical Guide to Splines" (Springer, 1978, revised 2001)
- **URL**: https://en.wikipedia.org/wiki/De_Boor's_algorithm
- **Complexity**: O(p^2) for degree p curve evaluation

The standard algorithm for B-spline evaluation. For NURBS, the approach is:
1. Multiply each control point by its weight (project to 4D homogeneous space)
2. Apply de Boor's algorithm in 4D
3. Project back by dividing by the weight component

**brepkit status**: Implemented in `basis.rs` via `find_span` (A2.1) and `basis_funs` (A2.2).
Surface evaluation uses tensor-product structure.

### Time-Efficient NURBS Curve Evaluation Algorithms
- **Authors**: Krishnamurthy, McMains
- **URL**: https://www.researchgate.net/publication/228411721
- **Key finding**: The Inverted Triangular Scheme (ITS) outperforms Cox-de Boor recursion
  for higher-degree splines, especially degree >= 4.

**Recommendation**: Consider ITS for high-degree NURBS if performance profiling shows
evaluation as a bottleneck.

### Fast Degree Elevation and Knot Insertion for B-spline Curves
- **Authors**: Huang, Shi, Wang (Tsinghua University, 2005)
- **URL**: https://cg.cs.tsinghua.edu.cn/~shimin/pdf/cagd_2005_degree.pdf
- **Key contribution**: O(np) algorithm for degree elevation (vs O(np^2) for Piegl-Tiller),
  and O(n + m) knot insertion where m is the number of new knots.

**brepkit status**: `decompose.rs` implements degree elevation via Bezier decomposition
(Piegl-Tiller approach). The Huang et al. method could provide a speedup for
batch operations.

### Knot Removal Algorithms for NURBS Curves and Surfaces
- **Authors**: Wayne Tiller (1992, Computer-Aided Design)
- **Foundational work**: Lyche and Morken, "Knot removal for parametric B-spline curves
  and surfaces" (CAGD, Vol 4, Issue 3, pp 217-230, 1987)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/001044859290012Y
- **Key contribution**: Error-bounded knot removal with C-language pseudocode.
  Single call can remove all removable knots within a given tolerance.

**brepkit status**: Not implemented. Knot removal is essential for:
- Simplifying curves/surfaces after boolean operations (SSI curves often have too many knots)
- STEP export (reducing file size)
- Maintaining clean geometry after successive operations

### Vectorizing NURBS Surface Evaluation with Basis Functions in Power Basis
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0010448515001591
- **Key contribution**: Converts B-spline basis to power basis for SIMD-friendly evaluation.
  Enables vectorized evaluation of multiple parameter values simultaneously.

**Recommendation**: Relevant if brepkit needs GPU-like throughput for tessellation or
intersection sampling. Works well with Rust's SIMD intrinsics.

---

## 3. Surface-Surface Intersection (SSI)

### Topology Guaranteed B-Spline Surface/Surface Intersection
- **Authors**: Jieyin Yang, Xiaohong Jia, Dong-Ming Yan
- **Venue**: ACM SIGGRAPH Asia 2023 (ACM Trans. on Graphics, Vol 42, No 6)
- **URL**: https://dl.acm.org/doi/10.1145/3618349
- **Confidence**: HIGH -- peer-reviewed, compared against SISL, OCCT, and ACIS

Key contributions:
- **Topology guarantee**: Correctly identifies all intersection branches including
  singular points, tangential contacts, boundary intersections, and closed loops
- Uses algebraic conditions to detect and classify singular intersection configurations
- Handles high-order contacts along curves (not just isolated points)
- Benchmarked against SISL (SINTEF), OCCT, and ACIS -- finds topology errors in all three

**brepkit status**: The current SSI implementation in `intersection.rs` uses alternating
projection + marching, which can miss closed loops and has no topology guarantee.
This paper represents the state of the art for correctness.

**Priority**: HIGH -- this is the single most impactful paper for improving brepkit's
boolean operations, since SSI correctness is the foundation.

### Fast Determination and Computation of Self-intersections for NURBS Surfaces
- **Authors**: Kai Li, Xiaohong Jia, Falai Chen
- **Venue**: ACM SIGGRAPH 2025 (ACM Trans. on Graphics)
- **URL**: https://dl.acm.org/doi/10.1145/3727620
- **Confidence**: HIGH -- accepted at SIGGRAPH 2025

Key contributions:
- Algebraic signature whose non-negativity is *sufficient* for excluding self-intersections
- Recursive subdivision with early termination using the signature
- Computes self-intersection locus when detected
- Validated against OCCT and ACIS kernels -- finds cases they miss
- Critical for offset surface operations (where self-intersections are common)

**brepkit status**: No self-intersection detection. The `offset_face.rs` module samples
and refits, which can produce self-intersecting surfaces without detecting the problem.

**Priority**: HIGH for offset operations.

### An Efficient Surface Intersection Algorithm Based on Lower-Dimensional Formulation
- **Authors**: Patrikalakis (1993)
- **Venue**: ACM Transactions on Graphics
- **URL**: https://dl.acm.org/doi/10.1145/237748.237751
- **Key contribution**: Reduces SSI to a lower-dimensional algebraic problem, improving
  robustness for algebraic/parametric surface combinations.

### Topology Guaranteed Curve Tracing for Parametric Surface-Surface Intersection
- **Year**: 2025
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0167839625000214
- **Key contribution**: Follow-up to the SIGGRAPH Asia 2023 paper, focusing on robust
  curve tracing with topology guarantees during the marching phase.

### An Efficient and Robust Tracing Method Based on Matrix Representation for SSI
- **Year**: 2025
- **URL**: https://link.springer.com/chapter/10.1007/978-981-96-5812-1_4
- **Key contribution**: Uses matrix representations of surfaces for more robust
  intersection curve tracing, avoiding some numerical issues of parametric marching.

### A Line/Trimmed NURBS Surface Intersection Algorithm Using Matrix Representations
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0167839616300875
- **Key contribution**: Line-surface intersection using algebraic matrix representations
  rather than Newton iteration. More robust for near-tangent configurations.

### Marching Methods for SSI (Classical References)
- **Barnhill, Kersey**: "A marching method for parametric surface/surface intersection"
  (Computer Aided Geometric Design, 1990)
  URL: https://www.sciencedirect.com/science/article/abs/pii/016783969090035P
- **Key algorithmic elements**:
  - Marching direction = cross product of surface normals (tangent to intersection curve)
  - Step size control via adaptive Runge-Kutta integration of the ODE system
  - Termination: curve returns to start (closed loop) or exits parameter domain boundary
  - Branch point detection via collinear normal test (Sederberg et al.)

**brepkit status**: The current marching in `intersection.rs` uses fixed-step marching
with Newton correction. Adding adaptive step size (RKF45) and proper loop detection
would significantly improve robustness.

### An Explicit and Rapid Intersection of Plane and NURBS Surface
- **Year**: 2025 (Scientific Reports)
- **URL**: https://www.nature.com/articles/s41598-025-25765-z
- **Key contribution**: Optimized plane-NURBS intersection for CNC machining,
  with explicit formulations avoiding iterative methods.

---

## 4. NURBS Trimming and Splitting

### Untrimming: Precise Conversion of Trimmed Surfaces to Tensor-Product Surfaces
- **Authors**: Fady Massarwi, Ben van Sosin, Gershon Elber
- **Venue**: Computers and Graphics, Vol 70, 2018, pp 80-91
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0097849317301383
- **Confidence**: HIGH -- from the Technion Geometric Modeling group

Key contributions:
- Converts trimmed NURBS surfaces into sets of *untrimmed* tensor-product B-spline surfaces
- Two algorithms for parametric domain decomposition into 4-sided quadrilaterals
- Eliminates gap/overlap problems inherent in trimmed surface representation
- Directly addresses the core challenge in NURBS booleans

**brepkit status**: The current approach in `nurbs_boolean.rs` uses parameter-space polygon
splitting, which is a simplified version of the decomposition problem this paper solves
rigorously. Implementing Massarwi-Elber untrimming would be a major accuracy improvement.

**Priority**: HIGH -- this is the key to moving beyond tessellation-based booleans.

### Volumetric Untrimming: Precise Decomposition of Trimmed Trivariates
- **Authors**: Massarwi, Antolin, Elber (2019)
- **Venue**: Computer Aided Geometric Design, Vol 71
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0167839619300196
- **Key contribution**: Extends untrimming to 3D trivariates (subdivide at all internal
  knots to get trimmed Bezier trivariates, then decompose each).

### A Review of Trimming in Isogeometric Analysis
- **Authors**: Marussig, Hughes (2018)
- **Venue**: Archives of Computational Methods in Engineering
- **URL**: https://link.springer.com/article/10.1007/s11831-017-9220-9
- **Key contribution**: Comprehensive survey of trimming approaches, challenges, and
  data exchange issues. Good overview of the state of the field.

Trimming challenges identified:
- Computing pre-images of trim curves in parametric domain is expensive and error-prone
- Topological inconsistencies (cracks, gaps) at trim boundaries
- Most CAD kernels use trimmed representation internally, causing accumulating errors
- The untrimmed approach (converting to tensor-product patches) avoids these issues

### Exact and Approximate Representations of Trimmed Surfaces with NURBS and Bezier Surfaces
- **Venue**: IEEE (2009)
- **URL**: https://ieeexplore.ieee.org/document/5246888/
- **Key contribution**: Patches far from trimming curves keep original surface; patches
  near trimming curves use high-degree Bezier (exact) or bicubic B-spline (approximate).

### Reconstruction of Trimmed NURBS Surfaces for Gap-Free Intersections
- **Venue**: ASME J. Computing and Information Science in Engineering (2020)
- **URL**: https://asmedigitalcollection.asme.org/computingengineering/article-abstract/20/5/051008/1084390/
- **Key contribution**: Post-processing trimmed surfaces to close gaps at intersection
  boundaries. Practical for CAD data exchange.

### Conversion of Trimmed NURBS Surfaces to Catmull-Clark Subdivision Surfaces
- **Venue**: CAGD (2014)
- **URL**: https://dl.acm.org/doi/10.1016/j.cagd.2014.06.004
- **Key contribution**: Cross-field decomposition in parameter space to create
  subdivision base mesh fitting the trimmed surface to tolerance.

### Watertight Trimmed NURBS
- **Venue**: ACM Transactions on Graphics (2008)
- **URL**: https://dl.acm.org/doi/10.1145/1360612.1360678
- **Key contribution**: Interval arithmetic approach to guarantee watertightness
  of trimmed NURBS during rendering and boolean operations.

---

## 5. NURBS Boolean Operations

### Boolean Operation for CAD Models Using a Hybrid Representation
- **Venue**: ACM Transactions on Graphics, July 2025
- **URL**: https://dl.acm.org/doi/10.1145/3730908
- **Confidence**: HIGH -- ACM TOG publication, 2025

Key contributions:
- Establishes bijective mapping between B-Rep models and triangle meshes with
  controllable approximation error
- Maps B-Rep boolean operations to mesh boolean operations
- Conservative intersection detection on mesh to locate SSI curves
- Results are consistently watertight and correct
- Handles degeneration and topology errors gracefully

**brepkit status**: This approach is conceptually similar to brepkit's current
tessellate-then-clip strategy but with a rigorous bijective mapping that allows
recovering exact B-Rep geometry from the mesh result. This is the key difference --
brepkit currently loses the NURBS geometry during tessellation.

**Priority**: MEDIUM-HIGH -- provides a practical path to robust booleans while
maintaining B-Rep compatibility. Could be an intermediate step before full exact
NURBS booleans.

### Watertight Boolean Operations: A Framework for Creating CAD-Compatible Gap-Free Editable Solid Models
- **Authors**: Urick, Marussig, Cohen, Crawford, Hughes, Riesenfeld
- **Venue**: Computer-Aided Design, Vol 115, 2019
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0010448519302106
- **Confidence**: HIGH -- peer-reviewed, journal publication

Key contributions:
- Three-stage process: (1) parametric space analysis, (2) reparameterization,
  (3) model space update
- Results are un-trimmed surface patches with explicit continuity
- Accurate to the same model tolerance as existing CAD systems
- Uses information computed during conventional boolean operations
- Can be integrated into existing CAD frameworks

**brepkit status**: This paper directly addresses the gap between brepkit's current
tessellation approach and exact NURBS booleans. The three-stage framework could be
implemented on top of the existing SSI infrastructure.

### Signed Algebraic Level Sets on NURBS Surfaces and Implicit Boolean Compositions
- **Authors**: Massarwi, Elber (2016)
- **Venue**: Computer-Aided Design
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0010448516301129
- **Key contribution**: Algebraic level sets ensure geometric exactness while eliminating
  iterative numerical computations for distance estimation, point projection, and
  point containment. Enables direct analysis without volume-conforming mesh.

### Implementation of Geometrical Boolean Functions Between Bodies Defined by NURBS
- **Venue**: IEEE (2011)
- **URL**: https://ieeexplore.ieee.org/document/6058700
- **Key contribution**: Practical implementation notes for NURBS boolean operations,
  addressing trimmed surface accuracy vs flat-facet approximation.

### BRL-CAD NURBS Booleans (Open Source)
- **URL**: https://brlcad.org/wiki/NURBS_Booleans
- **Status**: Active development, implementing ray-based NURBS booleans
- **Approach**: Consolidating Owens-Reeves ray intersection with topology healing
  (`rt_heal()` to tighten trimming curves and edge/vertex pairings)

### EMBER: Exact Mesh Booleans via Efficient and Robust Local Arrangements
- **Venue**: ACM SIGGRAPH 2022 (ACM Trans. on Graphics, Vol 41, No 4)
- **URL**: https://dl.acm.org/doi/10.1145/3528223.3530181
- **Key contribution**: Exact mesh booleans using local arrangements, avoiding
  global remeshing. Relevant for the mesh-based component of hybrid approaches.

---

## 6. NURBS Fitting and Approximation

### Least-Squares Progressive-Iterative Approximation (LSPIA)
- **Original**: Deng and Lin (2014)
- **Full-LSPIA**: Computer-Aided Design, 2023
  URL: https://www.sciencedirect.com/science/article/abs/pii/S0010448523002051
- **Confidence**: HIGH -- well-established method with convergence proofs

Key contributions:
- Iterative fitting that avoids assembling and solving global linear systems
- Full-LSPIA jointly optimizes weights, knots, and control points
- Convergent even when collocation matrix is not full column rank
- Local modifications don't require re-solving the entire system
- Natural fit for progressive/adaptive refinement workflows

**brepkit status**: The current fitting in `fitting.rs` and `surface_fitting.rs` likely
uses direct least-squares. LSPIA would be better for large point clouds and
interactive/progressive workflows.

### NURBS-Diff: A Differentiable Programming Module for NURBS
- **Venue**: Computer-Aided Design, 2022
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0010448522000045
- **Key contribution**: Mathematically defines derivatives of NURBS with respect to
  all input parameters (control points, weights, knots). Enables gradient-based
  optimization and integration with deep learning.

### NeuroNURBS: Learning Efficient Surface Representations for 3D Solids
- **Year**: 2024 (arXiv)
- **URL**: https://arxiv.org/html/2411.10848v1
- **Key contribution**: Neural network that directly encodes NURBS surface parameters.
  86.7% reduction in GPU consumption vs prior methods. Relevant for
  reverse engineering / scan-to-CAD workflows.

### Optimization of a NURBS Representation
- **Venue**: Computer-Aided Design, 1993
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/001044859390011C
- **Key contribution**: Classic paper on optimizing knot placement and control point
  distribution for best fit with minimum control points.

### Adaptive Dominant Point Selection for NURBS Fitting
- **Key idea**: Select fewer control points in flat regions, more in complex regions.
  Reduces control point count while maintaining accuracy.

**brepkit relevance**: After SSI operations produce intersection curves as sampled points,
these need to be fit to NURBS curves. Better fitting = cleaner geometry downstream.
The current interpolation-based approach (`interpolate` in `fitting.rs`) may produce
curves with too many control points.

---

## 7. Advanced Representations: T-Splines and Hierarchical B-Splines

### T-Splines and T-NURCCs
- **Authors**: Thomas W. Sederberg, Jianmin Zheng, Almaz Bakenov, Ahmad Nasri
- **Venue**: ACM SIGGRAPH 2003
- **URL**: https://dl.acm.org/doi/10.1145/1201775.882295
- **PDF**: https://archive.ymsc.tsinghua.edu.cn/pacm_download/53/505-jZ-SIGGRAPH03.pdf
- **Confidence**: HIGH -- foundational paper, 2000+ citations

Key contributions:
- Generalize NURBS by allowing T-junctions in control grid
- Lines of control points need not traverse entire grid
- Enable local refinement without inserting entire rows/columns
- Can merge multiple B-spline surfaces into single gap-free model
- Reduce control point count by 50-80% vs equivalent NURBS

### T-Spline Simplification and Local Refinement
- **Authors**: Sederberg, Cardon, Finnigan, North, Zheng, Lyche
- **Venue**: ACM SIGGRAPH 2004
- **URL**: https://dl.acm.org/doi/10.1145/1015706.1015715
- **Key contribution**: Algorithms for local refinement of T-splines and conversion
  from T-spline to B-spline (while keeping surface unchanged).

### Hierarchical B-Spline Refinement
- **Author**: Forsey, Bartels
- **Venue**: ACM SIGGRAPH Computer Graphics (1988)
- **URL**: https://dl.acm.org/doi/10.1145/378456.378512
- **Key contribution**: Overlay coarse and fine B-spline patches for local detail.
  Simpler than T-splines but less general.

### Truncated Hierarchical B-Splines (THB-Splines)
- **URL**: https://link.springer.com/chapter/10.1007/978-3-642-54382-1_18
- **Key contribution**: Preserve partition of unity (THB basis functions sum to 1),
  enabling adaptive local refinement while maintaining numerical stability.
  Important for isogeometric analysis integration.

### LR B-Splines (Locally Refined)
- **URL**: https://link.springer.com/chapter/10.1007/978-3-030-92313-6_10
- **Key contribution**: Alternative to T-splines for local refinement. Used by SINTEF
  in scattered data approximation. Simpler analysis-suitable properties.

### Adaptive Refinement of Hierarchical T-Splines
- **Venue**: Computer Methods in Applied Mechanics and Engineering (2018)
- **URL**: https://www.sciencedirect.com/science/article/abs/pii/S0045782518301567
- **Key contribution**: Combines T-splines with hierarchical refinement for
  adaptivity in both design and analysis.

**brepkit status**: Currently uses only standard NURBS (tensor-product B-splines).
T-splines or hierarchical approaches would reduce control point explosion after
boolean operations but represent a significant architectural change.

**Priority**: LOW for near-term, HIGH for long-term competitiveness. T-spline support
is a differentiating feature (Autodesk acquired T-Splines Inc. for Fusion 360).

---

## 8. Open-Source Implementations

### SINTEF GoTools
- **URL**: https://github.com/SINTEF-Geometry/GoTools
- **Language**: C++ (depends on SISL, the SINTEF Spline Library)
- **License**: Open source
- **Key algorithms**: SSI (recursive + marching), approximative implicitization,
  parametrization, topology
- **History**: SISL has been under continuous development since 1989; GoTools builds on it
- **Documentation**: https://www.sintef.no/en/software/gotools/

Relevant modules:
- `intersections/`: SSI algorithms (the benchmarked "SISL" in Yang et al. 2023)
- `gotools-core/geometry/`: Spline creation, evaluation, interrogation
- `compositemodel/`: Topology management
- `trivariatemodel/`: Trivariate spline operations

**Value for brepkit**: Reference implementation for SSI algorithms. The SISL intersection
code is mature (35+ years) but has known topology issues (documented in Yang et al. 2023).

### OpenCASCADE Technology (OCCT)
- **URL**: https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_algos.html
- **Language**: C++
- **License**: LGPL

Relevant intersection approach:
- `IntPolyh`: Starting point computation via polyhedra for biparametric surfaces
- Low-level: `Geom_Surface`/`Geom_Curve` intersection
- High-level: `BRepAlgoAPI` for topological boolean operations
- `BRepBuilderAPI_NurbsConvert`: Convert all geometry to NURBS
- General Fuse algorithm with fuzzy tolerance for robustness

**Value for brepkit**: Architecture reference for layered geometry/topology intersection.
Known to have SSI topology issues in edge cases (documented in Yang et al. 2023, Li et al. 2025).

### OpenNURBS (Rhino)
- **URL**: https://github.com/mcneel/opennurbs
- **Language**: C++
- **License**: Open source (read/write 3DM files)
- **Key feature**: Production-quality NURBS data structures and I/O

### verb (Cross-platform NURBS)
- **URL**: http://verbnurbs.com/
- **Language**: Haxe (compiles to JS, Python, C#)
- **Key feature**: Clean, readable NURBS implementation with good documentation.
  Useful as a reference for algorithm clarity.

### Rust NURBS Libraries
- **nurbs crate**: https://github.com/topics/nurbs (187 stars, updated Jan 2026)
- **Rust OpenNURBS forum discussion**: https://users.rust-lang.org/t/opennurbs-library-for-rust/102241
- **Status**: No mature Rust NURBS library exists -- brepkit could become the reference.

---

## 9. Recommendations for brepkit

### Immediate Priority (High Impact, Moderate Effort)

#### 1. Implement Knot Removal (Piegl-Tiller A5.4)
- **Why**: After SSI and boolean operations, curves/surfaces accumulate unnecessary knots.
  Knot removal with error bounds is essential for clean geometry.
- **Reference**: Tiller 1992, Lyche & Morken 1987
- **Location**: Add to `crates/math/src/nurbs/knot_ops.rs`
- **Effort**: ~200 lines, well-defined algorithm from The NURBS Book

#### 2. Adaptive Step Size in SSI Marching
- **Why**: Current fixed-step marching in `intersection.rs` can miss features or
  produce unnecessarily dense output.
- **Reference**: Barnhill-Kersey 1990 (RKF45 adaptive stepping)
- **Location**: Modify `intersect_nurbs_nurbs` in `crates/math/src/nurbs/intersection.rs`
- **Effort**: Moderate -- replace fixed step with Runge-Kutta-Fehlberg

#### 3. Closed Loop Detection in SSI
- **Why**: Current implementation can miss closed intersection loops entirely.
- **Reference**: Patrikalakis-Maekawa (collinear normal test), Yang et al. 2023
- **Location**: `crates/math/src/nurbs/intersection.rs`
- **Effort**: Moderate -- add collinear normal seed point generation

#### 4. Self-Intersection Detection for Offset Surfaces
- **Why**: `offset_face.rs` can produce self-intersecting surfaces without warning.
- **Reference**: Li, Jia, Chen (SIGGRAPH 2025) -- algebraic signature approach
- **Location**: New module or addition to `crates/math/src/nurbs/`
- **Effort**: Significant but well-defined algorithm

### Medium-Term (High Impact, Significant Effort)

#### 5. Topology-Guaranteed SSI (Yang et al. 2023)
- **Why**: The single most impactful improvement for boolean operation correctness.
  Current alternating projection approach has no topology guarantee.
- **Reference**: Yang, Jia, Yan (SIGGRAPH Asia 2023)
- **Location**: Major rewrite of `crates/math/src/nurbs/intersection.rs`
- **Effort**: Large -- requires algebraic singular point classification

#### 6. Untrimming (Massarwi-Elber 2018)
- **Why**: Converts trimmed surfaces to untrimmed tensor-product patches, eliminating
  gap/crack problems. Key enabler for exact NURBS booleans.
- **Reference**: Massarwi, van Sosin, Elber 2018
- **Location**: New module in `crates/math/src/nurbs/` or `crates/operations/src/`
- **Effort**: Large -- parametric domain quadrilateral decomposition

#### 7. Hybrid Boolean Operations (ACM TOG 2025)
- **Why**: Practical path to robust booleans with B-Rep recovery. More achievable
  than full exact NURBS booleans.
- **Reference**: Boolean Operation for CAD Models Using a Hybrid Representation (2025)
- **Location**: Enhancement to `crates/operations/src/nurbs_boolean.rs`
- **Effort**: Large -- bijective mesh-BRep mapping

#### 8. LSPIA Fitting
- **Why**: Better fitting for SSI output curves and scan-to-CAD workflows.
  Avoids global linear system solve.
- **Reference**: Full-LSPIA (CAD 2023)
- **Location**: Addition to `crates/math/src/nurbs/fitting.rs`
- **Effort**: Moderate

### Long-Term (Strategic, Large Effort)

#### 9. Watertight Boolean Framework (Urick et al. 2019)
- **Why**: The theoretical ideal -- gap-free, editable, un-trimmed results from
  boolean operations. Industry-leading capability.
- **Reference**: Urick, Marussig, Cohen et al. (CAD 2019)
- **Effort**: Very large -- requires robust SSI + untrimming + reparameterization

#### 10. T-Spline Support
- **Why**: Eliminates control point explosion, enables local refinement,
  merges multiple surfaces into single model. Competitive differentiator.
- **Reference**: Sederberg et al. 2003, 2004
- **Location**: New representation type alongside `NurbsSurface`
- **Effort**: Very large -- architectural change to geometry representation
- **Note**: T-spline patents (held by Autodesk) expired or are expiring; verify status

#### 11. Basis Function Derivatives (Standalone)
- **Why**: Needed for proper curvature analysis, surface normal computation in
  intersection marching, and sensitivity analysis.
- **Reference**: Piegl-Tiller A2.3
- **Location**: Add to `crates/math/src/nurbs/basis.rs`
- **Effort**: Small -- straightforward extension of existing basis_funs

---

## Summary of Key Papers by Relevance to brepkit

| Priority | Paper | Year | Impact Area |
|----------|-------|------|-------------|
| **CRITICAL** | Yang et al. "Topology Guaranteed SSI" | 2023 | SSI correctness |
| **CRITICAL** | Massarwi et al. "Untrimming" | 2018 | Trimming/booleans |
| **HIGH** | Li et al. "Self-intersection Detection" | 2025 | Offset surfaces |
| **HIGH** | ACM TOG "Hybrid Boolean" | 2025 | Boolean operations |
| **HIGH** | Urick et al. "Watertight Booleans" | 2019 | Boolean operations |
| **HIGH** | Tiller/Lyche-Morken "Knot Removal" | 1992/1987 | Geometry cleanup |
| **MEDIUM** | Full-LSPIA | 2023 | Fitting quality |
| **MEDIUM** | Barnhill-Kersey "Marching" | 1990 | SSI robustness |
| **MEDIUM** | Piegl-Tiller "The NURBS Book" Ch 5 | 1997 | All fundamentals |
| **LOW** | Sederberg "T-Splines" | 2003 | Future architecture |
| **LOW** | NeuroNURBS | 2024 | ML-based fitting |

---

## Sources

- [The NURBS Book (Piegl & Tiller)](https://cse.usf.edu/~lespiegl/nurbs.htm)
- [Shape Interrogation (Patrikalakis, Maekawa, Cho)](https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/)
- [Topology Guaranteed SSI (Yang et al. 2023)](https://dl.acm.org/doi/10.1145/3618349)
- [Self-intersection Detection (Li et al. 2025)](https://dl.acm.org/doi/10.1145/3727620)
- [Untrimming (Massarwi et al. 2018)](https://www.sciencedirect.com/science/article/abs/pii/S0097849317301383)
- [Hybrid Boolean (ACM TOG 2025)](https://dl.acm.org/doi/10.1145/3730908)
- [Watertight Booleans (Urick et al. 2019)](https://www.sciencedirect.com/science/article/abs/pii/S0010448519302106)
- [T-Splines (Sederberg et al. 2003)](https://dl.acm.org/doi/10.1145/1201775.882295)
- [T-Spline Refinement (Sederberg et al. 2004)](https://dl.acm.org/doi/10.1145/1015706.1015715)
- [Full-LSPIA (2023)](https://www.sciencedirect.com/science/article/abs/pii/S0010448523002051)
- [NURBS-Diff (2022)](https://www.sciencedirect.com/science/article/abs/pii/S0010448522000045)
- [SINTEF GoTools](https://github.com/SINTEF-Geometry/GoTools)
- [SISL (SINTEF Spline Library)](https://www.sintef.no/en/software/sisl/)
- [OpenCASCADE Modeling Algorithms](https://dev.opencascade.org/doc/overview/html/occt_user_guides__modeling_algos.html)
- [OpenNURBS (Rhino)](https://github.com/mcneel/opennurbs)
- [BRL-CAD NURBS Booleans](https://brlcad.org/wiki/NURBS_Booleans)
- [verb NURBS](http://verbnurbs.com/)
- [De Boor's Algorithm](https://en.wikipedia.org/wiki/De_Boor's_algorithm)
- [Knot Removal (Tiller 1992)](https://www.sciencedirect.com/science/article/abs/pii/001044859290012Y)
- [Barnhill-Kersey Marching (1990)](https://www.sciencedirect.com/science/article/abs/pii/016783969090035P)
- [GPU NURBS (Krishnamurthy et al. 2009)](https://ieeexplore.ieee.org/document/4782957/)
- [NeuroNURBS (2024)](https://arxiv.org/html/2411.10848v1)
- [Hierarchical B-Splines (Forsey-Bartels 1988)](https://dl.acm.org/doi/10.1145/378456.378512)
- [THB-Splines](https://link.springer.com/chapter/10.1007/978-3-642-54382-1_18)
- [Trimming Review (Marussig-Hughes 2018)](https://link.springer.com/article/10.1007/s11831-017-9220-9)
- [Plane-NURBS Intersection (2025)](https://www.nature.com/articles/s41598-025-25765-z)
- [Topology Guaranteed Curve Tracing (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0167839625000214)
- [EMBER Exact Mesh Booleans (2022)](https://dl.acm.org/doi/10.1145/3528223.3530181)
- [Fast Degree Elevation (Huang et al. 2005)](https://cg.cs.tsinghua.edu.cn/~shimin/pdf/cagd_2005_degree.pdf)

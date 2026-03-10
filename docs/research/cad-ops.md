# CAD Geometric Operations: Academic Research Survey

Research survey for brepkit -- a B-Rep modeling engine. Papers organized by topic
with relevance to current codebase capabilities and gaps.

Date: 2026-03-02

---

## Table of Contents

1. [Robust Boolean Operations](#1-robust-boolean-operations)
2. [Fillet and Blend Algorithms](#2-fillet-and-blend-algorithms)
3. [Offset Surface Algorithms](#3-offset-surface-algorithms)
4. [Sweep, Loft, and Skinning](#4-sweep-loft-and-skinning)
5. [Feature Recognition](#5-feature-recognition)
6. [Shape Healing and Repair](#6-shape-healing-and-repair)
7. [Foundational Works](#7-foundational-works)
8. [Recommendations for brepkit](#8-recommendations-for-brepkit)

---

## 1. Robust Boolean Operations

The current brepkit boolean implementation uses a tessellate-then-clip approach for
NURBS faces. The academic state of the art has moved significantly toward exact
arithmetic and hybrid representations. This is the area with the largest gap between
brepkit's current approach and published methods.

### 1.1. Exact Predicates and Constructions

**"Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates"**
- Authors: Jonathan Richard Shewchuk
- Published: Discrete & Computational Geometry, 1997
- Key contribution: Adaptive-precision arithmetic for orient2d, orient3d, incircle,
  and insphere predicates. The algorithms do only as much work as necessary -- if the
  sign of a determinant can be resolved with floating-point, no exact arithmetic is
  invoked. This is the foundation that nearly all robust geometry libraries build on.
- Relevance to brepkit: The math crate's tolerance-based comparisons could be
  augmented with Shewchuk-style adaptive predicates for critical geometric decisions
  (point-on-plane, orientation tests). A Rust port would provide guaranteed-correct
  results without the performance penalty of always using exact arithmetic.
- URL: https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf
- Code: https://www.cs.cmu.edu/~quake/robust.html

**"Indirect Predicates for Geometric Constructions"**
- Authors: Marco Attene
- Published: Computer-Aided Design, 2020
- Key contribution: Rather than computing intermediate geometric constructions
  (e.g., intersection points) and then evaluating predicates on them, indirect
  predicates evaluate predicates on the *input primitives* that define the
  construction. This preserves floating-point filter efficiency while achieving
  exact results. Demonstrated for intersection points of lines and planes.
- Relevance to brepkit: The surface-surface intersection module computes
  intersection points via Newton refinement. Indirect predicates would allow
  exact orientation tests on these computed points without materializing them
  in exact arithmetic, improving robustness of the NURBS boolean pipeline.
- URL: https://arxiv.org/abs/2105.09772
- Code: https://github.com/MarcoAttene/Indirect_Predicates

**"Simulation of Simplicity: A Technique to Cope with Degenerate Cases in
Geometric Algorithms"**
- Authors: Herbert Edelsbrunner, Ernst Peter Mucke
- Published: ACM Transactions on Graphics, 1990
- Key contribution: A general technique for handling geometric degeneracies by
  symbolically perturbing input data using infinitesimals. Each coordinate is
  perturbed by a polynomial in epsilon, ensuring no four points are coplanar,
  no three collinear, etc. The perturbation is never applied numerically -- it
  only affects the sign of predicates when the unperturbed result is zero.
- Relevance to brepkit: The boolean and section operations encounter degenerate
  configurations (coplanar faces, edges through vertices). SoS would provide a
  principled way to handle these without ad-hoc tolerance-based branching.
- URL: https://dl.acm.org/doi/10.1145/77635.77639
- PDF: https://www.sandia.gov/files/samitch/unm_math_579/p66_edelsbrunner_simulation_of_simplicity.pdf

### 1.2. Mesh Boolean Algorithms

**"Exact Predicates, Exact Constructions and Combinatorics for Mesh CSG"**
- Authors: Bruno Levy
- Published: ACM Transactions on Graphics, 2025 (arXiv preprint May 2024)
- Key contribution: First algorithm to compute an exact Weiler model (3D
  arrangement). The pipeline is:
  1. Co-refinement: compute triangle-triangle intersections with exact
     representation of intersection points
  2. Remesh intersected triangles using constrained Delaunay triangulation
     (with symbolic perturbation for co-cyclic points)
  3. Construct the Weiler model by radial sorting facets around non-manifold
     intersection edges using exact predicates
  4. Classify facets based on the boolean expression
  Two geometric kernels evaluated: arithmetic expansions and multi-precision
  floating-point. Handles 30M+ facet inputs (200 bunnies union with sphere).
- Relevance to brepkit: This represents the current gold standard for mesh
  booleans. The Weiler model approach could replace the tessellate-then-clip
  pipeline, providing exact results. The co-refinement + CDT approach is
  particularly relevant since brepkit already tessellates NURBS faces.
- URL: https://arxiv.org/abs/2405.12949
- Published version: https://dl.acm.org/doi/10.1145/3744642

**"Interactive and Robust Mesh Booleans"**
- Authors: Gianmarco Cherchi, Fabio Pellacini, Marco Attene, Marco Livesu
- Published: ACM Transactions on Graphics (SIGGRAPH Asia), 2022
- Key contribution: First boolean algorithm with robustness guarantees that
  operates at interactive frame rates (up to 200K triangles). Uses indirect
  predicates (from Attene 2020) to avoid materializing exact intersection
  coordinates. Floating-point filters handle the common case; exact arithmetic
  is a fallback.
- Relevance to brepkit: The interactive performance target and the filtered
  predicate strategy are directly applicable. The code is open source.
- URL: https://dl.acm.org/doi/10.1145/3550454.3555460
- Code: https://github.com/gcherchi/InteractiveAndRobustMeshBooleans

**"EMBER: Exact Mesh Booleans via Efficient & Robust Local Arrangements"**
- Authors: Philip Trettner, Julius Nehring-Wirxel, Leif Kobbelt
- Published: ACM Transactions on Graphics (SIGGRAPH), 2022
- Key contribution: Exact boolean operations using plane-based representation
  (P-rep) with homogeneous integer coordinates. Avoids global acceleration
  structures by using adaptive recursive subdivision. Uses generalized winding
  numbers for inside/outside classification.
- Relevance to brepkit: The plane-based representation is attractive for
  planar faces (which brepkit handles well already). The adaptive subdivision
  approach avoids BVH construction overhead.
- URL: https://dl.acm.org/doi/10.1145/3528223.3530181
- PDF: https://www.graphics.rwth-aachen.de/media/papers/339/ember_exact_mesh_booleans_via_efficient_and_robust_local_arrangements.pdf

**"Boolean Operation for CAD Models Using a Hybrid Representation"**
- Authors: Jia-Peng Guo, Xiao-Ming Fu
- Published: ACM Transactions on Graphics, 2025
- Key contribution: Hybrid V-rep/P-rep approach that combines the efficiency
  of vertex-based representations with the robustness of plane-based
  representations. Three stages: intersection detection (encoding intersections
  as plane sets), face tessellation (exact via P-rep), and face classification
  (local BSP trees compatible with P-reps). Variadic -- handles multi-operand
  booleans without decomposition.
- Relevance to brepkit: The hybrid approach is practical for a CAD kernel
  that must handle both planar and NURBS faces. The variadic capability would
  benefit multi-body boolean operations.
- URL: https://dl.acm.org/doi/10.1145/3730908

**"Fast and Robust Mesh Arrangements using Floating-point Arithmetic"**
- Authors: Gianmarco Cherchi, Marco Livesu, Riccardo Scateni, Marco Attene
- Published: ACM Transactions on Graphics (SIGGRAPH Asia), 2020
- Key contribution: Computes mesh arrangements (intersection + classification)
  using floating-point arithmetic with exact predicates only where needed.
  Precursor to the 2022 Interactive Mesh Booleans work.
- URL: https://dl.acm.org/doi/10.1145/3414685.3417818
- Code: https://github.com/gcherchi/FastAndRobustMeshArrangements

### 1.3. Winding Numbers and Classification

**"Robust Inside-Outside Segmentation using Generalized Winding Numbers"**
- Authors: Alec Jacobson, Ladislav Kavan, Olga Sorkine-Hornung
- Published: ACM Transactions on Graphics (SIGGRAPH), 2013
- Key contribution: Generalizes the winding number to arbitrary triangle meshes.
  Unlike ray casting, the generalized winding number is robust to gaps, self-
  intersections, and non-manifold geometry. Received SIGGRAPH Test-of-Time
  Award in 2024.
- Relevance to brepkit: The current point-in-solid classifier uses Moller-
  Trumbore ray casting, which can fail on non-manifold or imperfect meshes.
  Generalized winding numbers would provide a more robust alternative,
  especially for boolean classification of tessellated NURBS solids.
- URL: https://dl.acm.org/doi/10.1145/2461912.2461916
- Project page: https://igl.ethz.ch/projects/winding-number/

### 1.4. Snap Rounding

**"Iterated Snap Rounding"**
- Authors: Dan Halperin, Eli Packer
- Published: Computational Geometry, 2002
- Key contribution: Extends snap rounding (mapping arbitrary-precision
  coordinates to integer grid points) with iteration to guarantee minimum
  vertex-edge separation. Provides a practical approach to achieving robust
  geometric operations with bounded-precision coordinates.
- Relevance to brepkit: Could be used as a post-processing step after boolean
  operations to ensure output meshes have clean, grid-aligned coordinates.
  CGAL 6.1 (2025) integrates snap rounding with autorefinement for mesh
  self-intersection repair.
- URL: https://www.cgl.cs.tau.ac.il/projects/iterated-snap-rounding/
- CGAL integration: https://www.cgal.org/2025/06/13/autorefine-and-snap/

### 1.5. Watertight Booleans for CAD

**"Watertight Boolean Operations: A Framework for Creating CAD-Compatible
Gap-Free Editable Solid Models"**
- Authors: Benjamin Y. Urick, Benjamin Marussig, Elaine Cohen, Richard H.
  Crawford, Thomas J.R. Hughes, Richard F. Riesenfeld
- Published: Computer-Aided Design, 2019
- Key contribution: Three-stage framework (parametric space analysis,
  reparameterization, model space update) that produces boolean results
  consisting of un-trimmed surface patches with explicit continuity.
  The resulting models are gap-free and directly editable. Supports feature-
  based imprinting and partial boolean operations.
- Relevance to brepkit: This addresses a fundamental limitation of
  tessellate-then-clip -- loss of parametric surface information. The
  approach could enable brepkit to produce boolean results that maintain
  NURBS surface descriptions rather than falling back to meshes.
- URL: https://dl.acm.org/doi/10.1016/j.cad.2019.05.034

### 1.6. Nef Polyhedra

**"Boolean Operations on 3D Selective Nef Complexes: Data Structure,
Algorithms, Optimized Implementation and Experiments"**
- Authors: Peter Hachenberger, Lutz Kettner, Kurt Mehlhorn
- Published: Computational Geometry, 2007
- Key contribution: Nef polyhedra provide a mathematically closed
  representation under boolean operations (unlike B-Rep, which can produce
  non-manifold results). CGAL implements this as the foundation for its
  3D boolean operations.
- Relevance to brepkit: Understanding Nef polyhedra helps identify what
  the B-Rep topology system needs to represent (non-manifold intermediate
  results, dangling edges/faces).
- URL: https://doc.cgal.org/latest/Nef_3/index.html

---

## 2. Fillet and Blend Algorithms

The brepkit fillet implementation supports variable radius with constant, linear,
and S-curve laws. The literature covers additional approaches.

### 2.1. Rolling Ball Methods

**"Fillet and Surface Intersections Defined by Rolling Balls"**
- Authors: R. Klass, B. Kuhn
- Published: Computer Aided Geometric Design, 1992
- Key contribution: Formalized the rolling ball model for fillet computation.
  The fillet surface is the envelope of a sphere rolling along the edge
  between two surfaces, maintaining tangential contact with both.
- URL: https://www.sciencedirect.com/science/article/abs/pii/016783969290016I

**"Modelling Rolling Ball Blends for Computer Aided Geometric Design"**
- Authors: M.A. Sanglikar, P. Koparkar, V.N. Joshi
- Published: Computer Aided Geometric Design, 1990
- Key contribution: Algorithm for computing rolling ball blends on arbitrary
  parametric surfaces.

**"Variable-Radius Blending of Parametric Surfaces"**
- Authors: J.H. Chuang, C.H. Lin, W.C. Hwang
- Published: The Visual Computer, 1995
- Key contribution: Given a pair of parametric surfaces, a reference curve,
  and a radius function, presents an exact representation for the variable-
  radius spine curve and a marching procedure for computing the blend surface.
- Relevance to brepkit: The marching procedure for spine curve computation
  could improve the variable fillet implementation, which currently uses
  discrete sampling approaches.
- URL: https://link.springer.com/article/10.1007/BF02434038

### 2.2. Relative Blending

**"Relative Blending"**
- Authors: Brian Whited, Jarek Rossignac
- Published: Georgia Tech, 2009
- Key contribution: Blending operations that are defined relative to a
  reference shape rather than in absolute terms. Circular blends (popular
  in manufacturing) are the subset of a canal surface swept by a ball of
  constant or varying radius rolling on the solid while maintaining two
  tangential contacts.
- URL: https://faculty.cc.gatech.edu/~jarek/papers/relativerounding.pdf

### 2.3. Fillet Detection and Removal (Inverse Problem)

**"DeFillet: Detection and Removal of Fillet Regions in Polygonal CAD Models"**
- Authors: (multiple -- SIGGRAPH 2025)
- Published: ACM Transactions on Graphics (SIGGRAPH), 2025
- Key contribution: Solves the inverse fillet problem on polygonal meshes.
  Uses the observation that rolling-ball centers define osculating spheres
  and that Voronoi diagrams of surface samples provide rolling-ball center
  candidates. Fillet detection via Voronoi vertex analysis; sharp feature
  reconstruction via quadratic optimization. Validated on 100+ Fusion 360
  Gallery models.
- Relevance to brepkit: The feature recognition module detects fillet-like
  faces. DeFillet's Voronoi-based approach is more principled and could
  improve detection accuracy, especially for variable-radius fillets.
- URL: https://dl.acm.org/doi/10.1145/3731166
- Code: https://github.com/xiaowuga/DeFillet

---

## 3. Offset Surface Algorithms

brepkit implements planar offset (exact) and NURBS offset (sampling + refit).
The literature has extensive work on exact offset computation and self-intersection
handling.

### 3.1. Foundational Work

**"Offsetting Operations in Solid Modelling"**
- Authors: Jarek Rossignac, Aristides A.G. Requicha
- Published: Computer Aided Geometric Design, 1986
- Key contribution: Introduced solid offsetting (s-offsetting) as a family
  of operations mapping solids to solids. Defined CSGO -- an extension of
  CSG incorporating offset operations as non-terminal tree nodes. Discussed
  properties including idempotence and composition of offsets.
- Relevance to brepkit: The theoretical framework for understanding offset
  operations in the context of a solid modeling kernel. The CSGO tree
  concept could inform how brepkit composes offset with boolean operations.
- URL: https://faculty.cc.gatech.edu/~jarek/papers/Offsets.pdf
- DOI: https://dl.acm.org/doi/10.1016/0167-8396(86)90017-8

### 3.2. NURBS Offset Self-Intersection

**"Computing Non-Self-Intersecting Offsets of NURBS Surfaces"**
- Published: Computer-Aided Design, 2001
- Key contribution: Method for computing offset surfaces that avoids self-
  intersections: (1) sample offset surface based on second derivatives,
  (2) eliminate sample points that produce self-intersections, (3) surface
  fitting through remaining points, (4) removal of removable knots.
- Relevance to brepkit: The current NURBS offset uses sampling + refit
  but does not explicitly detect and remove self-intersections. This
  four-step approach would improve robustness for high-curvature regions.
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0010448501000811

**"Fast Determination and Computation of Self-intersections for NURBS Surfaces"**
- Published: ACM Transactions on Graphics, 2025
- Key contribution: Constructs an algebraic signature whose non-negativity
  is sufficient for excluding self-intersections globally. Provides an
  efficient algorithm using this signature recursively to determine and
  compute self-intersection loci.
- Relevance to brepkit: Could be used as a validation step after offset
  surface computation to detect problematic regions.
- URL: https://dl.acm.org/doi/10.1145/3727620

### 3.3. Self-Intersection Detection

**"Self-Intersection Detection and Elimination in Freeform Curves and Surfaces"**
- Published: Computer-Aided Geometric Design, 2001
- Key contribution: Regional representation scheme for detecting and
  classifying self-intersection curves in freeform surfaces. Identifies
  "miter points" where self-intersection curves meet.
- Relevance to brepkit: Self-intersection detection is critical for offset,
  fillet, and sweep operations. This provides the theoretical basis for
  detecting when these operations produce invalid geometry.
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0167839621000248

### 3.4. Untrimming (Offset-Related)

**"Untrimming: Precise Conversion of Trimmed-Surfaces to Tensor-Product
Surfaces"**
- Authors: Fady Massarwi, Gershon Elber
- Published: Computers and Graphics, 2018
- Key contribution: Converts trimmed NURBS surfaces to sets of un-trimmed
  tensor-product B-spline surfaces. Three steps: (1) divide trimmed
  parametric domain into quadrilaterals with freeform boundaries preserving
  trim curves, (2) parameterize quadrilaterals into planar patches,
  (3) lift to Euclidean space via symbolic surface-surface composition.
  Handles complex industrial models with thousands of trimmed surfaces.
- Relevance to brepkit: After boolean or offset operations produce trimmed
  surfaces, untrimming could convert results back to clean tensor-product
  patches for downstream operations.
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0097849317301383

**"Volumetric Untrimming: Precise Decomposition of Trimmed Trivariates into
Tensor Products"**
- Authors: Fady Massarwi, Boris van Sosin, Gershon Elber
- Published: Computer Aided Geometric Design, 2019
- Key contribution: Extends untrimming to trivariate B-splines for
  isogeometric analysis applications.
- URL: https://dl.acm.org/doi/10.1016/j.cagd.2019.04.005

---

## 4. Sweep, Loft, and Skinning

brepkit implements extrude, revolve, sweep (with Fixed, ConstantNormal, RMF
contact modes), loft, and Coons patch face filling.

### 4.1. Rotation Minimizing Frames

**"Computation of Rotation Minimizing Frames"**
- Authors: Wenping Wang, Bert Juttler, Dayue Zheng, Yang Liu
- Published: ACM Transactions on Graphics, 2008
- Key contribution: The "double reflection method" -- uses two reflections
  to compute each frame from its preceding one, yielding a sequence of
  frames that approximate an exact RMF with fourth-order global approximation
  error. Much more accurate than the projection method (Klok) or rotation
  method (Bloomenthal). Due to minimal twist, RMF is preferred for sweep
  surface modeling, motion design, and tool path planning.
- Relevance to brepkit: The sweep operation's RMF contact mode should use
  the double reflection method if it does not already. This is the standard
  algorithm for computing RMFs on discrete curve samples.
- URL: https://dl.acm.org/doi/10.1145/1330511.1330513

**"Rotation-Minimizing Frames on Space Curves -- Theory, Algorithms,
Applications"**
- Authors: Rida T. Farouki
- Published: Multiple papers, survey ~2008-2010
- Key contribution: Theoretical foundation for RMFs and their relationship
  to Pythagorean-hodograph (PH) curves, which admit exact rational RMFs.
  Covers rational approximation schemes for general curves.
- URL: https://faculty.engineering.ucdavis.edu/farouki/wp-content/uploads/sites/51/2021/07/Rational-rotation-minimizing-frames.pdf

### 4.2. Coons and Gordon Surfaces

**"Expressing Coons-Gordon Surfaces as NURBS"**
- Authors: F. Lin, William T. Hewitt
- Published: Computer Aided Design, 1994
- Key contribution: Mathematical relationship between Coons-Gordon surfaces
  and NURBS representation. Two approaches: global (network of curves as a
  whole) and local (Coons surface on each sub-rectangle). Provides the
  mathematical basis for converting between Coons patches and NURBS.
- Relevance to brepkit: The fill_face.rs module implements Coons patches
  for 4-sided boundaries. This paper provides the mathematical framework
  for ensuring NURBS compatibility.
- URL: https://www.sciencedirect.com/science/article/abs/pii/0010448594900353

### 4.3. NURBS Skinning and Lofting

**"The NURBS Book"**
- Authors: Les Piegl, Wayne Tiller
- Published: Springer, 1995 (2nd edition 1997)
- Key contribution: The standard reference for NURBS algorithms. Chapters
  on surface skinning (lofting through cross-sections), surface
  interpolation, and compatibility (degree elevation + knot insertion to
  make cross-section curves compatible before skinning).
- Relevance to brepkit: The loft operation should follow the Piegl-Tiller
  algorithm: (1) create B-spline curves for each cross-section, (2) make
  compatible via degree elevation and knot merging, (3) interpolate in
  the cross-section direction. Known issue: progressive knot vector
  merging can produce excessive control points.

**"Algorithm for Approximate NURBS Surface Skinning and Its Application"**
- Key contribution: Addresses the control point explosion problem in
  classical skinning by using approximate methods that balance accuracy
  with control point count.

**"An Approach to Sweeping NURBS"**
- Key contribution: Derives a general NURBS approximation for sweep
  surfaces expressed in terms of offset curves of the axis curve. Methods
  for approximating sweep surfaces as tensor product NURBS, where the
  sweep is generated by deforming a NURBS cross-section along a NURBS
  axis curve.
- URL: https://www.researchgate.net/publication/289028391_An_approach_to_sweeping_NURBS

---

## 5. Feature Recognition

brepkit implements rule-based feature recognition for chamfers, fillets, pockets,
and holes. The state of the art has shifted toward graph neural networks operating
on B-Rep face-edge graphs.

### 5.1. Graph Neural Network Approaches

**"AAGNet: A Graph Neural Network Towards Multi-Task Machining Feature
Recognition"**
- Authors: Wu, Lei, et al.
- Published: Robotics and Computer-Integrated Manufacturing, 2024
- Key contribution: Uses a geometric Attributed Adjacency Graph (gAAG)
  that preserves topological, geometric, and extended attributes from
  B-Rep models. Multi-task network performs semantic segmentation, instance
  segmentation, and bottom face segmentation simultaneously. Outperforms
  prior methods on MFCAD, MFCAD++, and the new MFInstSeg dataset.
- Relevance to brepkit: The gAAG representation maps naturally to
  brepkit's arena-based topology. The face/edge attributes could be
  extracted from the existing topology data structures for use in a
  learned feature recognizer.
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0736584523001369
- Code: https://github.com/whjdark/AAGNet

**"BRepGAT: Graph Neural Network to Segment Machining Feature Faces in a
B-rep Model"**
- Published: Journal of Computational Design and Engineering, 2023
- Key contribution: Graph attention network that operates directly on
  B-Rep face-edge graphs. Defines descriptors for faces and edges,
  transforms them into homogeneous graph data. Achieves 99.1% accuracy
  on MFCAD18++ dataset.
- URL: https://academic.oup.com/jcde/article/10/6/2384/7453688

**"BrepMFR: Enhancing Machining Feature Recognition in B-rep Models Through
Deep Learning and Domain Adaptation"**
- Published: Computer Aided Geometric Design, 2024
- Key contribution: Graph neural network based on Transformer architecture
  with graph attention mechanism. Two-step transfer learning framework for
  adapting from synthetic to real-world CAD models. Handles complex STEP
  CAD models directly.
- Relevance to brepkit: The domain adaptation approach is important --
  training on synthetic data and adapting to real models addresses the data
  scarcity problem.
- URL: https://dl.acm.org/doi/10.1016/j.cagd.2024.102318
- Code: https://github.com/zhangshuming0668/BrepMFR

**"PocketFinderGNN: Manufacturing Feature Recognition Software Based on GNNs"**
- Published: SoftwareX, 2023
- Key contribution: Converts CAD files to graph representation and uses a
  Graph Convolutional Network to predict close pocket features. 95%
  accuracy on 576 3D models. Open-source reference implementation.
- URL: https://www.sciencedirect.com/science/article/pii/S2352711023001620

### 5.2. Classical Approaches

**"An Efficient Algorithm for Recognizing and Suppressing Blend Features"**
- Published: CAD Journal, 2004
- Key contribution: Rule-based blend (fillet) recognition and suppression
  for mesh simplification. Identifies blends by geometric properties
  (constant cross-section radius, tangent continuity with adjacent faces).
- URL: https://www.cad-journal.net/files/vol_1/CAD_1(1-4)_2004_421-428.pdf

---

## 6. Shape Healing and Repair

brepkit implements heal_solid, remove_degenerate_edges, and fix_face_orientations.
The literature covers more comprehensive approaches.

### 6.1. Structure-Preserving Repair

**"Structure Preserving CAD Model Repair"**
- Authors: S. Bischoff, L. Kobbelt
- Published: Computer Graphics Forum (Eurographics), 2005
- Key contribution: Hybrid approach combining surface-oriented and volumetric
  repair methods. Uses the topological simplicity of a voxel grid to
  reconstruct cleaned surfaces near intersections and cracks, while
  preserving the input tessellation in clean regions. Produces guaranteed
  manifold, closed triangle meshes.
- Relevance to brepkit: The hybrid approach is practical -- brepkit could
  use local volumetric repair near problem areas while preserving the
  B-Rep structure elsewhere.
- URL: https://www.graphics.rwth-aachen.de/media/papers/nurbs_repair1.pdf

### 6.2. Topology Repair

**"Topology Repair of Solid Models Using Skeletons"**
- Authors: Tao Ju, et al.
- Published: IEEE Transactions on Visualization and Computer Graphics, 2007
- Key contribution: Uses skeleton representation to identify and measure
  topological handles (small surface handles that are artifacts of
  reconstruction). Handle removal is guaranteed not to introduce invalid
  geometry or additional handles. Uses adaptive grid for efficiency.
- URL: https://www.cs.wustl.edu/~taoju/research/topo_paper_tvcg_final.pdf

### 6.3. B-Rep Boolean Repair

**"B-Rep Boolean Resulting Model Repair by Correcting"**
- Published: arXiv preprint, 2023
- Key contribution: Addresses repair of models that result from failed or
  imprecise boolean operations. Corrects topological inconsistencies in
  the output B-Rep model.
- URL: https://arxiv.org/pdf/2310.10351

### 6.4. Gap Healing

**"Polygon Mesh Repairing: An Application Perspective"**
- Authors: Marco Attene, Marcel Campen, Leif Kobbelt
- Published: ACM Computing Surveys, 2013
- Key contribution: Comprehensive survey of mesh repair techniques
  including gap closure, self-intersection removal, orientation fixing,
  and degeneracy handling. Classifies approaches by the type of defect
  they address.
- URL: https://dl.acm.org/doi/10.1145/2431211.2431214

---

## 7. Foundational Works

### 7.1. Solid Modeling Theory

**"Geometric and Solid Modeling"**
- Authors: Christoph M. Hoffmann
- Published: Morgan Kaufmann, 1989
- Key contribution: Foundational textbook on solid modeling theory,
  representations (B-Rep, CSG), and algorithms. Covers boolean
  operations, boundary evaluation, and robustness.
- URL: https://www.semanticscholar.org/paper/Geometric-and-Solid-Modeling-Hoffmann/99158e6e89349671b06656270120b8f5c9990b10

**"Solid Modeling and Beyond"**
- Authors: Aristides A.G. Requicha, Jarek Rossignac
- Published: IEEE Computer Graphics and Applications, 1992
- Key contribution: Survey of solid modeling field -- mathematical
  foundations, representations, algorithms, applications. Covers CSG,
  B-Rep, regularized boolean operations.
- URL: https://www.semanticscholar.org/paper/Solid-modeling-and-beyond-Requicha-Rossignac/3824d22c710858e512ece06ba0d53abe75ad9f1a

**"Robustness in Geometric Computations"**
- Authors: Christoph M. Hoffmann
- Published: Journal of Computing and Information Science in Engineering, 2001
- Key contribution: Analysis of robustness challenges in geometric
  computation. Covers floating-point issues, exact arithmetic, perturbation
  techniques, and practical strategies for building robust geometric systems.
- URL: https://www.researchgate.net/publication/220517547_Robustness_in_Geometric_Computations

---

## 8. Recommendations for brepkit

Based on this research survey, here are prioritized recommendations for
improving brepkit's geometric operations, ordered by impact and feasibility.

### High Priority (Large Impact, Well-Understood Algorithms)

1. **Adopt Shewchuk-style adaptive predicates in the math crate.**
   Currently brepkit uses tolerance-based float comparison. Adaptive
   predicates (orient2d, orient3d, incircle) provide *exact* results
   with minimal overhead in the common case. A Rust implementation would
   benefit every geometric operation.
   - Papers: Shewchuk 1997
   - Effort: Medium (port existing C code or use a Rust crate like
     `robust` or `geo-predicates`)

2. **Replace ray casting with generalized winding numbers for
   point-in-solid classification.**
   The current Moller-Trumbore ray casting is fragile on non-manifold
   meshes and near edges/vertices. Generalized winding numbers (Jacobson
   2013) are inherently robust to mesh defects.
   - Papers: Jacobson et al. 2013
   - Effort: Medium (straightforward summation over mesh triangles)

3. **Implement co-refinement-based mesh booleans.**
   The tessellate-then-clip approach loses geometric information and is
   not robust to degeneracies. A co-refinement approach (Levy 2024/2025
   or Cherchi 2022) computes exact intersections and preserves the input
   mesh structure.
   - Papers: Levy 2025, Cherchi et al. 2022, EMBER 2022
   - Effort: High (significant algorithm, but open-source reference
     implementations exist)

4. **Add self-intersection detection to offset surface computation.**
   The current NURBS sampling + refit approach can produce self-
   intersecting offsets for high-curvature regions. The four-step
   approach from the literature (sample, eliminate self-intersecting
   points, refit, simplify) would improve robustness.
   - Papers: "Computing Non-Self-Intersecting Offsets of NURBS Surfaces"
   - Effort: Medium

### Medium Priority (Significant Improvement, More Complex)

5. **Implement Simulation of Simplicity for degenerate handling.**
   Ad-hoc tolerance branching for coplanar faces, collinear edges, etc.
   is error-prone and incomplete. SoS provides a principled, general
   solution.
   - Papers: Edelsbrunner & Mucke 1990
   - Effort: Medium (requires careful integration with predicate system)

6. **Improve RMF computation with the double reflection method.**
   Verify the sweep operation's RMF implementation uses Wang et al.'s
   double reflection method for fourth-order accuracy.
   - Papers: Wang et al. 2008
   - Effort: Low (well-documented algorithm)

7. **Add DeFillet-style Voronoi-based fillet detection.**
   The current rule-based fillet detection could be augmented with the
   Voronoi vertex approach for more robust detection of variable-radius
   fillets on tessellated models.
   - Papers: DeFillet (SIGGRAPH 2025)
   - Effort: Medium

### Lower Priority (Advanced, Research-Level)

8. **Explore untrimming for boolean output cleanup.**
   After booleans produce trimmed surfaces, Massarwi-Elber untrimming
   could convert results back to clean tensor-product patches.
   - Papers: Massarwi & Elber 2018
   - Effort: High

9. **Investigate GNN-based feature recognition.**
   The rule-based feature recognition could be supplemented with a
   learned approach (AAGNet, BRepGAT). This requires training data
   (MFCAD dataset) and inference infrastructure, but provides much
   better generalization to complex intersecting features.
   - Papers: AAGNet 2024, BRepGAT 2023, BrepMFR 2024
   - Effort: High (ML infrastructure needed)

10. **Watertight boolean framework for NURBS.**
    Long-term goal: boolean operations that produce un-trimmed,
    gap-free NURBS patches (Urick et al. 2019).
    - Papers: Urick et al. 2019
    - Effort: Very High (research-level)

---

## Confidence Assessment

- **High confidence**: The paper citations, authors, publication venues, and
  key contributions are verified through multiple search results and cross-
  referenced across sources.
- **Medium confidence**: Some specific algorithmic details are summarized from
  abstracts rather than full paper text. Performance claims should be verified
  against the original papers.
- **Gap**: I was unable to access full paper PDFs for detailed algorithm
  descriptions. The summaries are based on abstracts, search result snippets,
  and project pages.

## Key Journals and Venues

- ACM Transactions on Graphics (SIGGRAPH / SIGGRAPH Asia)
- Computer-Aided Design (Elsevier)
- Computer Aided Geometric Design (Elsevier)
- Computer Graphics Forum (Eurographics)
- Computational Geometry: Theory and Applications
- Journal of Computational Design and Engineering (Oxford)
- Discrete & Computational Geometry (Springer)

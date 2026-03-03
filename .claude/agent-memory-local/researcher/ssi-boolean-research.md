# SSI and Boolean Operations Research (2026-03-03)

## Topic 1: NURBS SSI Marching Algorithms

### ODE Formulation (Patrikalakis-Maekawa)
- The intersection curve satisfies a first-order ODE: dX/ds = t where t = N1 x N2 / |N1 x N2|
- Source: MIT hyperbook https://web.mit.edu/hyperbook/Patrikalakis-Maekawa-Cho/node118.html
- For tangential intersection (N1 || N2), second-order analysis of curvature difference tensor applies
- brepkit already implements this in `singular_tangent_direction()` via `second_order_tangent()`

### Adaptive Step Size Formula
- Standard formula: h = sqrt(8 * epsilon / kappa) where epsilon = chord-height tolerance, kappa = curvature of intersection curve
- Derivation: from osculating circle, chord height = h^2 * kappa / 8; solving for h gives the formula
- Source: Barnhill-Kersey 1990 CAGD and confirmed in multiple references
- brepkit current state: fixed step (march_step parameter), NO curvature adaptation

### Predictor-Corrector Structure
1. Predictor: Euler step along tangent t = N1 x N2 (or RK4 for better accuracy)
2. Corrector: Newton iteration in (u1,v1,u2,v2) to snap back to intersection
3. Step acceptance: compare predicted vs corrected 3D position; if error > tolerance, halve step
- brepkit uses this pattern in `march_direction()` but with fixed step (no error-controlled resizing)

### 2023/2025 State of Art
- Yang, Jia, Yan (SIGGRAPH Asia 2023): "Topology Guaranteed B-Spline SSI"
  URL: https://dl.acm.org/doi/10.1145/3618349
  Hybrid: interval algebraic topology analysis → subdivision for seeds → forward marching
  Classifies 4 fundamental topological cases (transversal, tangent point, tangent curve, cusp)
- "Topology guaranteed and error controlled curve tracing" (CAGD 2025, ScienceDirect pii/S0167839625000214)
  SRS-BFS: Dixon matrix + breadth-first search for branch points
- Mukundan et al. 2004: validated ODE (interval arithmetic) eliminates straying/looping
  URL: https://diglib.eg.org/handle/10.2312/sm20041397

## Topic 2: Boolean Face Splitting and Trim Reconstruction

### OCCT Algorithm (BOPAlgo)
Source: https://old.opencascade.com/doc/occt-7.4.0/overview/html/occt_user_guides__boolean_operations.html

Pipeline:
1. Intersection Part: compute Face/Face interferences → curves stored as pave blocks
2. BOPDS_PaveBlock: (edge, pave1, pave2, shrunkData, commonBlock)
3. BOPDS_InterfFF: (tolR3D, tolR2D, curves[], points[])
4. BOPAlgo_BuilderFace:
   a. Collect all split edges for face Fi (ESPij) + section edges SEk
   b. BOPAlgo_WireSplitter: trace wires using min-clockwise-angle walk in 2D parameter space
   c. Build split face topology from closed wires
5. Classification via FaceInfo: IN, ON, SC states
6. Result extraction: union=IN+ON, cut=IN_A+ON_A, intersect=IN+ON

### Correct Trim Wire Reconstruction Algorithm
The key is the min-clockwise-angle walk in 2D (parameter space):
- Project all edges to 2D parameter space of parent face
- At each vertex, pick the outgoing edge that makes the smallest clockwise turn
- This naturally finds the minimal (innermost) loop at each step
- Handles multiple wires including holes

### brepkit gap
- brepkit CDT splits produce triangles but don't reassemble into oriented trim loops
- Need: wire reconstruction from intersection edges using the OCCT WireSplitter approach

## Topic 3: Robust Face Classification

### Single-ray (current brepkit approach)
- Cast ray from face centroid, count intersections with opposing solid
- Problem: ray tangency, numerical near-misses near edges

### Multi-ray consensus
- Cast N rays in different directions, majority vote
- Handles individual ray failures but O(N) cost

### Generalized Winding Numbers (GWN) - recommended upgrade path

#### For triangle meshes (Jacobson 2013, Barill 2018)
- w(p) = (1/4pi) * sum of signed solid angles of each triangle
- BVH with multipole expansion for far-field (O(log n) per query)
- Barill et al. 2018: https://www.dgp.toronto.edu/projects/fast-winding-numbers/
- SideFX open source: https://github.com/sideeffects/WindingNumber

#### For trimmed NURBS directly (Spainhour et al. 2025)
- Reduces surface integral to boundary integral via Stokes' theorem
- Uses adaptive Gauss-Legendre quadrature on boundary curves
- 3 spatial regimes: far-field (direct), near-field (intersection test), edge-case (disk extraction)
- Performance: ~0.15-1.3 ms per query depending on model complexity
- Handles non-watertight models robustly
- arxiv: https://arxiv.org/abs/2504.11435

#### For parametric curves in 2D (Spainhour 2024)
- GWN without quadrature for rational parametric curves
- arxiv: https://arxiv.org/html/2403.17371v1

### Face state (ON vs IN vs OUT) vs winding number
- w ~ 1.0: inside solid
- w ~ 0.0: outside solid
- w ~ 0.5: on boundary (use smaller epsilon)
- Non-integer values indicate non-watertight geometry

## Topic 4: NURBS Bounding Volume Computation

### Convex Hull Property
- NURBS surface is contained within convex hull of control points (this is exact)
- AABB of control points is a valid (loose) bound
- Source: standard NURBS theory, confirmed by Piegl-Tiller "The NURBS Book"

### Tight AABB via Subdivision
Algorithm:
1. Decompose NURBS into Bezier patches (Bezier extraction)
2. For each Bezier patch, AABB of control polygon is a valid bound
3. Recursively subdivide patch until AABB of control polygon is "tight enough"
   - Termination: max(patch_diagonal) < tolerance OR flatness criterion
4. Union of all leaf AABBs is tight bound for whole surface

### OBB via PCA
- Apply PCA to control points to find principal axes
- Build OBB in those axes
- Tighter than AABB but more expensive
- OCCT provides Bnd_OBB for shapes: https://dev.opencascade.org/doc/refman/html/class_bnd___o_b_b.html
- CGAL 5.2+ has optimal OBB: https://doc.cgal.org/5.2.4/Optimal_bounding_box/index.html

### Refinement via Curvature
- Evaluate surface on a regular grid and expand AABB based on surface curvature
- GPU-accelerated: Pabst et al. "Ray Casting of Trimmed NURBS Surfaces on the GPU"
  URL: https://www.uni-weimar.de/fileadmin/user/fak/medien/professuren/Virtual_Reality/documents/publications/Ray_Casting_of_Trimmed_NURBS_Surfaces_on_the_GPU.pdf

### Self-intersection detection (2025 state-of-art)
- Li, Jia, Chen (ACM TOG 2025): "Fast Determination and Computation of Self-intersections for NURBS Surfaces"
  URL: https://dl.acm.org/doi/10.1145/3727620
  Key: algebraic signature whose non-negativity proves absence of self-intersections
  Recursive subdivision using this signature for fast determination

## Topic 5: Curvature-Adaptive Marching

### Standard Formula (widely cited since Barnhill-Kersey 1990)
h_new = sqrt(8 * epsilon_chord / kappa)
where:
  epsilon_chord = allowed chord-height deviation (user tolerance, e.g. 1e-3)
  kappa = curvature of intersection curve at current point

### Computing kappa from surface data
At intersection point with params (u1,v1,u2,v2):
1. Get surface normals N1, N2
2. tangent t = N1 x N2 (normalized)
3. Curvature of intersection curve:
   kappa = |t' / |t|| where t' is derivative of tangent w.r.t. arc length
4. Approximate from second-order data:
   kappa ≈ |dt/ds| computed from second-order SSI ODE (uses second derivatives of surfaces)

### Practical implementation
- Compute kappa from the osculating circle fit to 3 consecutive marching points
- Update h: h_new = min(h_max, max(h_min, sqrt(8*epsilon/kappa)))
- Typical: h_min = 1e-5, h_max = 0.1, epsilon = linear_tolerance / 10

### Step rejection
- Compare predicted point (Euler) vs corrected point (Newton)
- err = |predicted_3D - corrected_3D|
- If err > tolerance: reject, halve h, retry
- If err < tolerance/4: accept, double h for next step (up to h_max)
- This is the embedded RKF45 pattern adapted for constrained manifold tracing

### brepkit current state
- `march_intersection()` uses fixed step_size parameter
- No curvature computation
- No step rejection/adaptation
- Improvement opportunity: add curvature estimation from 2nd derivatives and adapt h per step

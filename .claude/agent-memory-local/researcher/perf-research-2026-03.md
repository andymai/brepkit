# High-Performance CAD Kernel Research Survey (2026-03-08)

Comprehensive survey of algorithms and techniques for boolean operations, surface-surface
intersection, spatial acceleration, parallelism, and tessellation in B-Rep NURBS kernels.

---

## 1. Boolean Operations

### 1.1 Production Kernel Approaches

**Parasolid** (Siemens -- powers NX, SolidWorks, Onshape):
- Symmetric imprinting: intersection curves projected onto both faces simultaneously
- Sliver elision via area/angle thresholds cooperating with tolerance
- Intersection curves stored as implicit definitions, not NURBS approximations
- Local booleans: classify boundary regions as locally-inside or locally-outside near imprinted edges
- Source: [Parasolid Boolean Operations](http://www.q-solid.com/Parasolid_Docs/chapters/fd_chap.10.html)

**ACIS** (Spatial -- powers Fusion 360, CATIA internals):
- Stores NURBS approximations alongside implicit definitions
- Dual representation enables fast queries with exact fallback
- Source: [Parasolid vs ACIS data model](https://opencascade.blogspot.com/2010/10/data-model-highlights-parasolid-acis.html)

**OCCT General Fuse** (open source):
- Foundation for all boolean ops (union, cut, intersect, section)
- Accepts arbitrary number of arguments in single pass
- Pipeline: V/V -> V/E -> E/E -> V/F -> E/F -> F/F interference cascade
- Options: parallel mode, fuzzy tolerance, gluing mode, OBB acceleration, history
- Cells Builder: computes all splits once, then extracts any boolean combination
- Source: [OCCT Boolean Operations](https://dev.opencascade.org/doc/overview/html/specification__boolean_operations.html)
- Relevance to brepkit: **high** -- the Cells Builder pattern is the gold standard for multi-body fuse

### 1.2 Mesh Boolean Algorithms (State of Art 2022-2025)

#### Exact Mesh Arrangements

**Levy 2024: "Exact Predicates, Exact Constructions and Combinatorics for Mesh CSG"**
- Published: ACM TOG 2024 / arXiv:2405.12949
- URL: [arxiv](https://arxiv.org/abs/2405.12949), [ACM](https://dl.acm.org/doi/10.1145/3744642)
- Algorithm: co-refinement -> exact Weiler model (radial sort around intersection edges) -> volumetric decomposition
- Remeshes intersection triangles using CDT with symbolic perturbation
- Two exact kernels: expansion arithmetic vs GMP multi-precision
- Performance: 30M facets (200 bunnies + sphere) in ~10 minutes each for union/difference
- First algorithm to compute exact Weiler model
- Complexity to implement: **hard** (exact arithmetic, CDT, radial sort)
- Applicability to brepkit: **medium** -- brepkit is B-Rep not mesh-first, but the Weiler model concept applies to face splitting

**Guo & Fu 2024: "Exact and Efficient Intersection Resolution for Mesh Arrangements"**
- Published: ACM TOG 43(6), Article 165, 2024
- URL: [ACM](https://dl.acm.org/doi/10.1145/3687925), [project page](https://mangoleaves.github.io/projects/mesh-arrangements/)
- Key innovation: "indirect offset predicates" -- new formulation reducing numerical errors
- Localization + dimension reduction for sorting/deduplicating intersection points
- **One order of magnitude faster** than prior state-of-art (Cherchi et al.)
- Enables parallelism through localization
- Complexity: **hard**
- Applicability: **medium** -- relevant if brepkit ever needs mesh-level boolean fallback

**EMBER (Trettner et al. 2022): "Exact Mesh Booleans via Efficient & Robust Local Arrangements"**
- Published: ACM TOG 41(4), SIGGRAPH 2022
- URL: [ACM](https://dl.acm.org/doi/10.1145/3528223.3530181), [paper PDF](https://www.graphics.rwth-aachen.de/media/papers/339/ember_exact_mesh_booleans_via_efficient_and_robust_local_arrangements.pdf)
- Plane-based representation + homogeneous integer coordinates for exactness
- Adaptive recursive subdivision of bounding box (no pre-built acceleration structure)
- Early-out termination for non-contributing regions
- Generalized winding numbers for classification
- Complexity: **medium-hard**
- Applicability: **medium** -- the adaptive subdivision without pre-built BVH is interesting for brepkit

**Cherchi et al. 2022: "Interactive and Robust Mesh Booleans"**
- Published: ACM TOG (SIGGRAPH Asia 2022)
- URL: [ACM](https://dl.acm.org/doi/10.1145/3550454.3555460), [GitHub](https://github.com/gcherchi/InteractiveAndRobustMeshBooleans)
- Uses indirect predicates (Attene 2020) -- exact predicates without explicit rational construction
- Interactive frame rates on meshes up to 200K triangles
- Outperforms prior robust methods by at least one order of magnitude
- Complexity: **medium**
- Applicability: **high** -- indirect predicates could replace brepkit's current exact predicate setup

#### Hybrid B-Rep/Mesh Boolean

**Yang et al. 2025: "Boolean Operation for CAD Models Using a Hybrid Representation"**
- Published: ACM TOG 44(4), SIGGRAPH 2025
- URL: [ACM](https://dl.acm.org/doi/10.1145/3730908)
- Establishes bijective mapping between B-Rep and triangle mesh with controllable approximation error
- Maps B-Rep booleans to mesh booleans, then maps results back
- Conservative intersection detection on mesh to locate all SSI curves
- Handles degeneration and topology errors for watertight results
- Complexity: **hard**
- Applicability: **very high** -- directly relevant to brepkit's B-Rep architecture. The bijective mapping approach could be a robust fallback when direct NURBS SSI fails.

### 1.3 CGAL Approaches

**Nef Polyhedra:**
- Closed under all boolean operations + complement + regularization
- Exact but extremely slow -- uses exact number types (rationals or algebraic)
- Bounded variant improves memory and runtime
- Source: [CGAL Nef_3](https://doc.cgal.org/latest/Nef_3/index.html)

**Corefinement (Polygon Mesh Processing):**
- ~10x faster than Nef polyhedra on average (Loriot benchmark on Thingi10k)
- Fails on non-manifold inputs or shared edges/vertices
- Source: [CGAL PMP Corefinement](https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__corefinement__grp.html)
- Hybrid approach (OpenSCAD): try corefinement first, fall back to Nef
- Source: [OpenSCAD fast CSG](https://ochafik.com/jekyll/update/2022/02/09/openscad-fast-csg-contibution.html)

**BSP-tree Booleans (Bernstein & Fussell 2009):**
- 16-28x faster than CGAL Nef polyhedra
- Only 4 geometric predicates needed
- Source: [paper](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-8659.2009.01504.x)

### 1.4 Face Classification Approaches (ranked by robustness)

| Approach | Speed | Robustness | Complexity |
|----------|-------|------------|------------|
| Single-ray parity (brepkit current) | O(n) per face | Fragile near edges | Easy |
| Multi-ray consensus | O(kn) per face | Better | Easy |
| GWN with BVH (Barill 2018) | O(log n) per query | Very robust | Medium |
| GWN on trimmed NURBS (Spainhour 2025) | ~0.15-1.3ms/query | Robust, no tessellation | Medium |
| Label consistency propagation (Yang 2025) | O(1) amortized | Robust with good SSI | Medium |

**Recommendation for brepkit:** Implement label consistency propagation (classify one fragment, flood-fill across non-intersection edges). Use GWN as fallback for seed classification.

---

## 2. Surface-Surface Intersection (SSI)

### 2.1 Marching Methods

**Standard predictor-corrector (current brepkit approach):**
- Predictor: Euler step along tangent t = N1 x N2
- Corrector: Newton iteration in (u1,v1,u2,v2)
- brepkit gap: uses fixed step size, no curvature adaptation

**Curvature-adaptive step size:**
- Formula: h = sqrt(8 * epsilon / kappa) where kappa = intersection curve curvature
- Practical: estimate kappa from osculating circle through 3 consecutive points
- Step rejection: if |predicted - corrected| > tolerance, halve step; if < tolerance/4, double step
- Expected improvement: **2-5x fewer points** for smooth curves, **no missed features** on high-curvature regions
- Complexity: **easy** -- just add curvature estimation and step control to existing march loop
- Source: Barnhill-Kersey 1990 CAGD

**Tradeoffs:**
- Efficient for non-degenerate cases
- Struggles with: tangential intersections, near-overlapping surfaces, small loops, branch points
- Cannot guarantee finding all branches (seed-finding problem)

### 2.2 Subdivision Methods

**Bezier clipping for SSI:**
- Decompose NURBS patches to Bezier; recursively clip non-intersecting parameter regions
- Guaranteed to find all intersections (no seed problem)
- Convergence: superlinear for transversal crossings
- Slow for tangential/near-tangential cases (linear convergence)
- Complexity: **medium** -- brepkit already has bezier_clip for curve-curve

**Interval arithmetic:**
- Replace floating-point evaluation with interval evaluation
- Validates that intersection curves stay within tolerance bands
- Eliminates straying/looping (Mukundan et al. 2004)
- Source: [EG proceedings](https://diglib.eg.org/handle/10.2312/sm20041397)
- Complexity: **medium** -- need interval NURBS evaluation primitives

### 2.3 Hybrid Methods (Current State of Art)

**Yang, Jia, Yan (SIGGRAPH Asia 2023): "Topology Guaranteed B-Spline SSI"**
- URL: [ACM](https://dl.acm.org/doi/10.1145/3618349)
- Three-phase hybrid: algebraic topology analysis -> subdivision for seeds -> forward marching
- Classifies 4 fundamental cases: transversal, tangent point, tangent curve, cusp
- Topology guarantee: all branches found, correct connectivity
- Complexity: **hard** (Dixon resultant, algebraic analysis)
- Applicability: **very high** -- this is the algorithm brepkit should aim toward

**Topology-guaranteed curve tracing (CAGD 2025):**
- SRS-BFS: Dixon matrix + breadth-first search for branch points
- Source: ScienceDirect pii/S0167839625000214

### 2.4 Self-Intersection Detection

**Li, Jia, Chen (ACM TOG 2025): "Fast Determination and Computation of Self-intersections for NURBS Surfaces"**
- URL: [ACM](https://dl.acm.org/doi/10.1145/3727620)
- Algebraic signature whose non-negativity proves absence of self-intersections
- Recursive subdivision using signature for fast determination
- Complexity: **medium**
- Applicability: **high** -- needed for validation and healing

### 2.5 Recommendations for brepkit SSI

Priority order:
1. **Curvature-adaptive step size** in march_intersection (easy, immediate win)
2. **Step rejection/acceptance** with error control (easy, prevents bad geometry)
3. **Bezier subdivision for seed finding** before marching (medium, fixes branch-missing)
4. **Interval arithmetic validation** during marching (medium, eliminates straying)
5. **Topology-guaranteed algorithm** (hard, long-term goal)

---

## 3. Spatial Acceleration

### 3.1 BVH Construction Strategies

**Surface Area Heuristic (SAH):**
- Cost function: C(split) = C_trav + P(left) * C(left) + P(right) * C(right)
- P(child) = SA(child) / SA(parent) (surface area ratio)
- Optimal for query-heavy workloads (many intersection tests per build)
- Construction: O(n log n), query: O(log n) expected
- Source: [PBR Book](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)

**Binned SAH:**
- Limits split candidates to K bins per axis (typically K=8 or K=12)
- O(n) per node instead of O(n log n) for full sweep
- Quality: within 5% of full SAH in practice
- Source: [Wald 2007](https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf)
- Complexity: **easy** -- straightforward to implement
- Recommendation: **brepkit should switch from median-split to binned SAH**

**Linear BVH (LBVH):**
- Sort primitives along Morton/Hilbert curve, partition by bit prefix
- O(n) construction, GPU-friendly
- Ray tracing quality: ~50% of SAH (too loose for production)
- Source: [Karras 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)
- Applicability: **low for CAD** -- brepkit doesn't need sub-millisecond BVH builds

**Hybrid LBVH + SAH:**
- LBVH for upper levels (coarse structure), SAH for lower levels (quality splits)
- Good balance of construction speed and tree quality
- Complexity: **medium**

### 3.2 R-tree vs BVH for CAD

| Criterion | BVH | R-tree |
|-----------|-----|--------|
| Dynamic inserts/removes | Fair (refit or partial rebuild) | Good (designed for it) |
| Batch queries | Excellent | Good |
| Memory layout | Cache-friendly (flat array) | Pointer-heavy |
| Build quality | SAH gives optimal | R*-tree comparable |
| CAD suitability | Better for static geometry (boolean inputs) | Better for scene management |

**Recommendation:** BVH with binned SAH for boolean broadphase and SSI pair-finding. brepkit's current AABB BVH is fine; upgrade to binned SAH for better query performance.

### 3.3 Spatial Hashing

- Hash function: h(x,y,z) = (x*p1 xor y*p2 xor z*p3) mod table_size
- Cell size: slightly larger than largest primitive diameter
- O(1) expected lookup, O(n) construction
- Best for: roughly uniform distributions of similarly-sized primitives
- Source: [Teschner et al. 2003](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
- Applicability to brepkit: **low-medium** -- CAD geometry varies wildly in scale, BVH handles this better. Spatial hashing is useful for uniform tessellation grids.

### 3.4 Oriented Bounding Boxes (OBB)

- PCA on control points for principal axes
- Much tighter fit for rotated geometry (typical in CAD assemblies)
- OCCT supports OBB via SetUseOBB() in boolean ops
- Complexity: **easy** (PCA + project)
- Expected improvement: **2-5x fewer candidate pairs** for rotated geometry
- Recommendation: implement OBB for BVH broadphase in booleans

---

## 4. Parallelism in CAD Kernels

### 4.1 Which Operations Parallelize Well?

| Operation | Parallelism Type | Difficulty | Expected Speedup |
|-----------|-----------------|------------|------------------|
| Tessellation (per face) | Embarrassingly parallel | Easy | ~linear with cores |
| SSI broadphase (pair filtering) | Data parallel | Easy | ~linear |
| SSI narrowphase (per pair) | Task parallel | Medium | ~linear with pairs |
| Face classification | Data parallel | Easy | ~linear |
| Boolean split (per face) | Task parallel (if independent) | Medium | Good |
| CDT triangulation | Per-face parallel | Easy | ~linear |
| STEP export (per entity) | Embarrassingly parallel | Easy | ~linear |
| Wire reconstruction | Sequential (topological dependency) | N/A | None |
| Edge-edge intersection | Data parallel | Easy | ~linear |

**OCCT parallelism:** Each face is processed independently during tessellation with collision-free data structures. General Fuse supports parallel mode.
Source: [OCCT Mesh](https://dev.opencascade.org/doc/overview/html/occt_user_guides__mesh.html)

### 4.2 Rayon Integration for brepkit

**Pattern for parallel tessellation:**
```
faces.par_iter().map(|face| tessellate_single_face(topo, face)).collect()
```

**Key constraint:** brepkit's arena (`Topology`) requires `&mut` for allocation. Solutions:
1. **Read-only parallel pass:** tessellation only reads topology, writes triangles to thread-local buffers. Merge after. No arena mutation needed.
2. **Per-thread arenas with bumpalo-herd:** Each thread gets its own bump arena for scratch allocations. Merge results into main arena sequentially.
3. **map_init pattern:** `par_iter().map_init(|| local_state, |state, item| process(state, item))`

Source: [bumpalo-herd](https://docs.rs/bumpalo-herd), [rayon map_init](https://docs.rs/rayon/latest/rayon/)

**Recommended first target:** Parallel face tessellation. Each face produces an independent triangle mesh. No topology mutation needed. Expected speedup: ~4-8x on typical 8-core machine.

Complexity: **easy** -- just wrap the face iteration in par_iter, collect results.

### 4.3 GPU-Accelerated Booleans

**Zoo/KittyCAD approach:**
- Entire geometry engine designed for GPU (Vulkan/Nvidia)
- SSI formulated as parallelizable root-finding problem
- Claims sub-second boolean ops and order-of-magnitude rendering performance
- Source: [Zoo CAD Engine Overview](https://zoo.dev/research/zoo-cad-engine-overview)

**GPU Boolean Papers:**
- GPU-accelerated progressive boolean operations (JCAD 2015): rasterization-based approach
- Source: [JCAD](https://www.jcad.cn/en/article/id/5525a108-e880-4abe-8a32-84df0082a8f3)
- Performance bottleneck: traversing B-Rep databases on GPU is inherently unfriendly
- Source: [Iowa State GPU-CAD](https://web.me.iastate.edu/idealab/r-gpu.html)

**Assessment for brepkit:** GPU acceleration is a long-term goal. The immediate wins are CPU parallelism with rayon. GPU SSI (root-finding on parameter space) could be explored later via wgpu.

### 4.4 Lock-Free Arena Allocation

**rarena-allocator:**
- Lock-free concurrent-safe arena with two-stage allocation (main memory + freelist)
- Source: [crates.io](https://crates.io/crates/rarena-allocator)

**sync-arena:**
- Simple thread-safe arena allowing concurrent inserts
- Source: [crates.io](https://crates.io/crates/sync-arena)

**Assessment for brepkit:** brepkit's topology arena doesn't need concurrent mutation for most parallel operations. The read-only-then-merge pattern is simpler and sufficient. Lock-free arenas add complexity without matching benefit.

---

## 5. Tessellation

### 5.1 Adaptive Tessellation Strategies

**Chord-height deviation:**
- Standard approach in CAD: tessellate until max distance from triangle to surface < tolerance
- OCCT uses linear + angular deflection controls
- Formula: subdivide patch until chord height < tolerance AND normal angle < angular_tol

**Curvature-adaptive:**
- Compute surface curvature at each point, use more triangles in high-curvature regions
- Fewer triangles overall for same visual quality
- Expected improvement: **2-10x fewer triangles** for mixed-curvature models

**Screen-space adaptive (ETER/WATER):**
- Measure error in screen pixels, not world units
- For interactive rendering, not CAD export
- ETER (SIGGRAPH 2023): pixel-accurate, crack-free, real-time, 3.7M patches at 30FPS
  Source: [ACM](https://dl.acm.org/doi/10.1145/3592419)
- WATER (SIGGRAPH Asia 2025): watertight, 2-3x faster than hw tessellation for bi-3 patches
  Source: [ACM](https://doi.org/10.1145/3763317)

### 5.2 Parallel Face Tessellation

**Pattern:** Each face tessellates independently -> collect all triangles -> build mesh index
- OCCT does exactly this with OMP parallel for
- No synchronization needed if each face writes to its own buffer
- Complexity: **easy** with rayon in Rust

**Implementation sketch for brepkit:**
1. Collect all face IDs
2. `par_iter()` over faces
3. For each face: sample boundary wires -> CDT in parameter space -> evaluate surface at CDT nodes
4. Collect results into Vec<(vertices, triangles, normals)>
5. Merge sequentially (assign global vertex indices)

### 5.3 Direct Analytic Surface Tessellation

**Concept:** For cylinders, spheres, cones, tori -- generate triangles directly from analytic formulas instead of going through NURBS evaluation.

**Benefits:**
- Exact normals from analytic formula (no numerical differentiation)
- Faster evaluation (trig functions vs NURBS basis evaluation)
- Fewer triangles needed (known curvature -> optimal subdivision)
- brepkit already preserves analytic surfaces through booleans (commit 7923932)

**Algorithm for cylinder:**
1. Compute number of angular subdivisions from angular tolerance: n_theta = ceil(2*pi / acos(1 - tol/r))
2. Compute number of height subdivisions from linear tolerance (usually just 1 for straight cylinder)
3. Generate quad strip, split to triangles
4. Normals: n = (cos(theta), sin(theta), 0) in local frame

**Algorithm for sphere:**
1. Latitude subdivision from angular tolerance
2. Longitude subdivision varies per latitude band (cos-weighted)
3. Use triangle fan at poles, quad strips elsewhere

**Complexity:** **easy** -- formulaic, no NURBS overhead
**Expected speedup over NURBS tessellation:** **5-20x** for these surface types (dominates in mechanical CAD)
**Recommendation:** brepkit should implement analytic tessellation paths for all 4 analytic surface types

### 5.4 CDT for Trimmed Faces

**Standard approach:**
1. Project trimming curves to parameter space (u,v)
2. Insert as constrained edges into 2D CDT
3. Classify triangles as inside/outside trim region
4. Evaluate 3D position and normal at each CDT vertex
- Source: [robust CAD tessellation](https://www.sciencedirect.com/science/article/abs/pii/S001044850600131X)

**brepkit status:** Already has CDT implementation in `crates/math/src/cdt.rs`. Main gap is robust trim curve projection and triangle classification.

### 5.5 Watertight Tessellation (Gap-Free)

**Problem:** Adjacent faces tessellated independently produce T-junctions at shared edges.

**Solutions:**
1. **Shared edge tessellation:** Tessellate edges first, faces use shared edge vertex lists
2. **Stitching pass:** Post-process to merge vertices within tolerance at face boundaries
3. **Connectivity-aware:** Use B-Rep adjacency to ensure matching edge discretization

**brepkit approach:** Edge tessellation is already shared via `tessellate_edge()`. This is correct but needs verification that adjacent faces use exactly the same edge points.

---

## 6. Summary: Priority Recommendations for brepkit

### Immediate wins (easy, high impact)

1. **Parallel face tessellation with rayon** -- wrap face loop in par_iter, ~4-8x speedup
   - Effort: ~50 lines of code
   - Risk: none (embarrassingly parallel)

2. **Curvature-adaptive SSI step size** -- h = sqrt(8*eps/kappa)
   - Effort: ~30 lines in march_intersection
   - Benefit: 2-5x fewer intersection points, no missed features

3. **Direct analytic surface tessellation** -- bypass NURBS for cylinder/sphere/cone/torus
   - Effort: ~200 lines per surface type
   - Benefit: 5-20x faster tessellation for these types, exact normals

4. **Binned SAH for BVH** -- replace median split with 8-12 bin SAH
   - Effort: ~100 lines in bvh.rs
   - Benefit: ~30% fewer BVH traversal steps

### Medium-term improvements (medium effort, high impact)

5. **OBB broadphase** for boolean candidate pair filtering
6. **Label consistency propagation** for face classification (avoid per-face ray casting)
7. **Bezier subdivision seed-finding** before SSI marching
8. **Step rejection/acceptance** with error control in SSI
9. **Parallel SSI pair evaluation** with rayon

### Long-term goals (hard, transformative)

10. **Topology-guaranteed SSI** (Yang et al. 2023 algorithm)
11. **Hybrid B-Rep/mesh boolean** (Yang et al. 2025 approach)
12. **Multi-body fuse / Cells Builder** pattern (OCCT-style)
13. **GWN-based point classification** (Spainhour 2025 for trimmed NURBS)
14. **Indirect predicates** (Attene 2020) for exact geometric tests

---

## 7. Key References (by topic)

### Boolean Operations
- Yang et al. 2025 "Hybrid Boolean": [ACM](https://dl.acm.org/doi/10.1145/3730908)
- Levy 2024 "Exact Mesh CSG": [arxiv](https://arxiv.org/abs/2405.12949)
- Guo & Fu 2024 "Mesh Arrangements": [ACM](https://dl.acm.org/doi/10.1145/3687925)
- EMBER 2022: [ACM](https://dl.acm.org/doi/10.1145/3528223.3530181)
- Cherchi et al. 2022 "Interactive Booleans": [ACM](https://dl.acm.org/doi/10.1145/3550454.3555460), [GitHub](https://github.com/gcherchi/InteractiveAndRobustMeshBooleans)
- OCCT General Fuse: [docs](https://dev.opencascade.org/doc/overview/html/specification__boolean_operations.html)
- CGAL Corefinement: [docs](https://doc.cgal.org/latest/Polygon_mesh_processing/group__PMP__corefinement__grp.html)
- OpenSCAD hybrid approach: [blog](https://ochafik.com/jekyll/update/2022/02/09/openscad-fast-csg-contibution.html)

### Surface-Surface Intersection
- Yang, Jia, Yan 2023 "Topology Guaranteed SSI": [ACM](https://dl.acm.org/doi/10.1145/3618349)
- Li, Jia, Chen 2025 "Self-intersections": [ACM](https://dl.acm.org/doi/10.1145/3727620)
- Spainhour et al. 2025 "GWN for trimmed NURBS": [arxiv](https://arxiv.org/abs/2504.11435)

### Spatial Acceleration
- PBR Book BVH chapter: [link](https://pbr-book.org/3ed-2018/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)
- Wald 2007 parallel BVH build: [PDF](https://www.sci.utah.edu/~wald/Publications/2007/ParallelBVHBuild/fastbuild.pdf)
- Karras 2013 LBVH: [PDF](https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf)
- Teschner et al. 2003 spatial hashing: [PDF](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)

### Tessellation
- ETER 2023: [ACM](https://dl.acm.org/doi/10.1145/3592419)
- WATER 2025: [ACM](https://doi.org/10.1145/3763317)
- Zoo CAD Engine: [overview](https://zoo.dev/research/zoo-cad-engine-overview)

### Parallelism / Allocation
- Rayon: [GitHub](https://github.com/rayon-rs/rayon)
- bumpalo-herd: [docs](https://docs.rs/bumpalo-herd)
- rarena-allocator: [crates.io](https://crates.io/crates/rarena-allocator)
- Attene 2020 indirect predicates: [ResearchGate](https://www.researchgate.net/publication/341017841_Indirect_Predicates_for_Geometric_Constructions)

# Compound Boolean / Multi-Body Fuse Research (2026-03-08)

## Core Problem
Sequential booleans on brepkit tessellate analytic surfaces into triangles, causing
face-count explosion (each boolean multiplies face count). Production kernels avoid
this by preserving surface types and splitting only trim loops.

## OCCT Architecture

### General Fuse Algorithm (BOPAlgo_Builder)
- Foundation for all boolean ops (fuse, cut, common, section)
- Accepts arbitrary number of arguments
- Pipeline: Interference detection (V/V, V/E, E/E, V/F, E/F, F/F) -> Build splits -> Same-domain merge
- Result: compound where each sub-shape corresponds to argument, sharing sub-shapes at intersections
- OBB acceleration available via SetUseOBB()
- Gluing mode for coincident faces speeds up intersection
- Fuzzy tolerance for near-coincident geometry

### Cells Builder (BOPAlgo_CellsBuilder)
- Extension of General Fuse for multi-body / compound boolean expressions
- KEY ADVANTAGE: "All splits are built only once and then reused"
- Compute all intersections in single pass -> extract any boolean combination from cells
- Material IDs control internal boundary removal
- Avoids re-splitting on each sequential operation
- Sources: https://dev.opencascade.org/content/boolean-expressions-shapes
           https://dev.opencascade.org/doc/occt-7.3.0/refman/html/class_b_o_p_algo___cells_builder.html

## Production Kernel Surface Preservation
- Parasolid: stores intersection curves as implicit definitions, not NURBS approximations
- ACIS: stores NURBS approximations alongside implicit definitions
- OCCT: preserves FaceSurface type; splits modify trim wires only
- All three: cylinder split by plane -> 2 cylindrical faces (not triangles)
- Source: https://opencascade.blogspot.com/2010/10/data-model-highlights-parasolid-acis.html

## State-of-Art Papers

### Yang et al. 2025 "Boolean Operation for CAD Models Using a Hybrid Representation"
- ACM TOG (SIGGRAPH 2025): https://dl.acm.org/doi/10.1145/3730908
- P-rep (plane coefficients) for exact predicates + V-rep (vertices/BVH) for fast queries
- Label consistency: classify one fragment, propagate to neighbors across non-intersection edges
- Handles complex industrial models where OCCT fails
- RWTH publication: https://publications.rwth-aachen.de/record/1016540

### Mei et al. 2018 "Accelerated Robust Boolean Operations Based on Hybrid Representations"
- CAGD: https://www.sciencedirect.com/science/article/abs/pii/S0167839618300359
- Introduced V-rep/P-rep hybrid concept
- Deferred tessellation via "Tess-Graph": tessellate only after all intersections detected
- Label consistency for accelerated face classification

### Zhou et al. 2016 "Mesh Arrangements for Solid Geometry"
- SIGGRAPH 2016: https://www.cs.columbia.edu/cg/mesh-arrangements/
- Winding-number vector labels each cell; booleans = cell selection
- Uses CGAL exact kernel for robustness
- Foundation for modern mesh boolean approaches

### Bernstein & Fussell 2009 "Fast, Exact, Linear Booleans"
- BSP-tree boolean, 16-28x faster than CGAL Nef polyhedra
- Only 4 geometric predicates needed
- https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1467-8659.2009.01504.x

## Face Classification Approaches (ranked by robustness)
1. Single-ray parity (brepkit current) - fragile near edges/tangencies
2. Multi-ray consensus - better but O(kn)
3. Generalized Winding Number with BVH - O(log n), robust (Barill 2018)
4. GWN on trimmed NURBS directly - no tessellation needed (Spainhour 2025)
5. Label consistency propagation - O(1) amortized per fragment (Yang 2025, Mei 2018)

## Avoiding O(n^2) Face Pairs
- BVH/OBB broadphase (brepkit already does AABB BVH, threshold 32 faces)
- Spatial hashing (THSH: each primitive -> 1 hash slot, linear time)
- Dimensional cascade (OCCT: V/V -> V/E -> E/E -> V/F -> E/F -> F/F, prune using lower-dim results)
- OCCT PR #514: helper functions IsPlaneFF, IsClosedFF to skip trivially non-intersecting face pairs

## Recommendations for brepkit (priority order)
1. STOP tessellating analytic surfaces - split trim loops, preserve FaceSurface type
2. Implement compound/multi-body fuse (Cells Builder pattern)
3. Label-consistency face classification
4. GWN-based point classification for robustness
5. OBB broadphase for rotated geometry

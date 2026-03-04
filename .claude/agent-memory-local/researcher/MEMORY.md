# Researcher Agent Memory

## Research Completed
- OpenCascade feature parity analysis (2026-03-02) - see `occt-features.md`
- CAD geometric operations literature survey (2026-03-02) - see `/var/home/andy/Git/brepkit/RESEARCH_CAD_OPS.md`
- NURBS algorithms survey (2026-03-02) - see `/var/home/andy/Git/brepkit/RESEARCH_NURBS.md`
- Math layer academic papers survey (2026-03-02) - see `/var/home/andy/Git/brepkit/RESEARCH_MATH.md`
- Performance optimization research (2026-03-02) - see `/var/home/andy/Git/brepkit/RESEARCH_PERFORMANCE.md`

## Key Findings
- OCCT has no built-in 2D constraint solver in open-source version
- OCCT boolean ops use General Fuse as foundation algorithm, with fuzzy tolerance
- Shape healing is a major differentiator - comprehensive fix/analyze/upgrade pipeline
- OCCT meshing is Delaunay-based with linear+angular deflection controls
- STEP support: AP203, AP214, AP242 with colors/layers/GD&T via XDE
- NURBS SSI state-of-art: Yang et al. 2023 (SIGGRAPH Asia) topology-guaranteed algorithm
- Key gap in brepkit: no knot removal, no closed-loop detection in SSI, no self-intersection detection
- Massarwi-Elber "untrimming" (2018) is the key paper for exact NURBS booleans
- No mature Rust NURBS library exists - brepkit could be the reference
- Rust exact predicates: `geometry-predicates` crate exposes expansion arithmetic primitives
- `robust` crate only has 2D; `robust-predicates` has all 4 Shewchuk predicates
- Watertight ray-triangle (Woop-Benthin-Wald 2013) should replace Moller-Trumbore for classification
- Graph-based decomposition is the standard for production constraint solvers (SolveSpace, PlaneGCS)

## SSI / Boolean Deep Dive (2026-03-03) - see `ssi-boolean-research.md`
- Canonical step size formula: h = sqrt(8*epsilon/kappa) from osculating circle deviation
- GWN for trimmed NURBS: Spainhour et al. 2025 (arxiv 2504.11435), Stokes' theorem boundary integral
- OCCT face splitting uses BOPAlgo_WireSplitter: min clockwise angle walk in 2D param space
- brepkit march_intersection: fixed step + Newton corrector, no adaptive curvature step yet
- Self-intersection NURBS: Li, Jia, Chen 2025 ACM TOG - algebraic signature approach
- Topology-guaranteed SSI: Yang, Jia, Yan SIGGRAPH Asia 2023 (dl.acm.org/doi/10.1145/3618349)
- Fast winding numbers (meshes): Barill et al. SIGGRAPH 2018 - BVH with multipole expansion

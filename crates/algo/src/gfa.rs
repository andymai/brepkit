//! Top-level GFA orchestrator.
//!
//! Runs the complete General Fuse Algorithm pipeline:
//! PaveFiller -> Builder -> BOP -> assemble.

use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::bop::BooleanOp;
use crate::builder::Builder;
use crate::ds::{GfaArena, GfaImprintStore, GfaShapeStoreN};
use crate::error::AlgoError;
use crate::pave_filler;

/// Result-face provenance in caller-topology face indices:
/// `(result face index, Some(input face index) | None)`.
pub type FaceOriginIndices = Vec<(usize, Option<usize>)>;

/// Run the complete GFA boolean operation with default tolerance.
///
/// This is the single entry point for boolean operations via the GFA.
///
/// # Errors
///
/// Returns [`AlgoError`] if any stage fails.
pub fn boolean(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<SolidId, AlgoError> {
    // Identical-solid fast path: avoid the full GFA pipeline when both
    // operands are the same topology entity.
    if solid_a == solid_b {
        return match op {
            BooleanOp::Fuse | BooleanOp::Intersect => {
                // A ∪ A = A, A ∩ A = A — return the original solid.
                // The caller (operations crate) copies if needed.
                Ok(solid_a)
            }
            BooleanOp::Cut => {
                // A \ A = empty
                Err(AlgoError::AssemblyFailed(
                    "Cut of identical solids produces empty result".into(),
                ))
            }
        };
    }
    boolean_with_tolerance(topo, op, solid_a, solid_b, Tolerance::default())
}

/// Run the complete GFA boolean operation with custom tolerance.
///
/// Stages:
/// 1. **PaveFiller** — intersect shapes, build pave blocks
/// 2. **Builder** — split faces, classify sub-faces
/// 3. **BOP + assemble** — select faces, build result solid
///
/// # Errors
///
/// Returns [`AlgoError`] if any stage fails.
pub fn boolean_with_tolerance(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
    tol: Tolerance,
) -> Result<SolidId, AlgoError> {
    // Create an isolated shape store with deep-copied input solids.
    // The GFA pipeline operates entirely within the store, avoiding
    // vertex/edge identity conflicts with the caller's topology.
    let mut store = crate::ds::GfaShapeStore::new(topo, solid_a, solid_b)?;

    // Stage 1: PaveFiller — intersection + pave block construction
    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler(
        &mut store.topo,
        store.solid_a,
        store.solid_b,
        tol,
        &mut arena,
    )?;

    // Stage 2: Builder — face splitting + classification
    let mut builder = Builder::with_tolerance(
        std::mem::take(&mut store.topo),
        arena,
        store.solid_a,
        store.solid_b,
        tol,
    );
    builder.perform()?;

    // Stage 3: BOP selection + assembly
    let (store_topo, store_result) = builder.build_result(op)?;
    store.topo = store_topo;
    if store.topo.is_empty_solid(store_result) {
        return Ok(topo.add_empty_solid());
    }

    // Export result solid back to the caller's topology
    let result = store.export_solid(topo, store_result)?;

    Ok(result)
}

/// Fuse **N** solids into one via a single GFA arrangement.
///
/// One pass over all operands instead of the sequential pairwise fuse's
/// re-processing of a growing accumulator (O(n²)). Runs the N-way pave filler
/// (pairwise intersection into one arena, spatially pruned) then the N-way
/// builder (split once, classify each sub-face against all other sources, keep
/// the union boundary, resolve coincident faces).
///
/// Fuse-only, and currently limited to the planar-coincidence cases the N-way
/// builder handles — it returns an error (for the caller to fall back to the
/// sequential fuse) on a non-planar coincident contact or an unresolved
/// coincidence. `sources` must be non-empty; a single source returns a copy.
///
/// # Errors
///
/// Returns [`AlgoError`] if `sources` is empty or any GFA stage fails.
pub fn fuse_n(topo: &mut Topology, sources: &[SolidId]) -> Result<SolidId, AlgoError> {
    match sources {
        [] => Err(AlgoError::AssemblyFailed(
            "fuse_n needs at least one source solid".into(),
        )),
        [only] => {
            // A copy so the caller owns a distinct result, mirroring the
            // store round-trip the multi-source path performs.
            let store = GfaShapeStoreN::new(topo, &[*only])?;
            store.export_solid(topo, store.sources[0])
        }
        _ => {
            let tol = Tolerance::default();
            let mut store = GfaShapeStoreN::new(topo, sources)?;

            let mut arena = GfaArena::new();
            pave_filler::run_pave_filler_n(&mut store.topo, &store.sources, tol, &mut arena)?;

            let (store_topo, store_result) = crate::builder::build_fuse_n(
                std::mem::take(&mut store.topo),
                arena,
                &store.sources,
                &store.face_source,
                tol,
            )?;
            store.topo = store_topo;

            store.export_solid(topo, store_result)
        }
    }
}

/// A copy of `solid` with its faces split along where the tools meet them.
///
/// A tool face is intersected as an open shell: the pave filler reads its
/// faces and asks nothing of closure, which only a boolean's classification
/// would need. Every piece of the solid's faces is kept. A section that
/// separates no two regions of a face (it ends inside the face, or joins two
/// of its boundaries only once) leaves the face whole.
///
/// # Errors
///
/// Returns [`AlgoError`] if a face cannot be split along its sections or the
/// pieces do not assemble.
pub fn imprint(
    topo: &mut Topology,
    solid: SolidId,
    tool_faces: &[FaceId],
    tool_edges: &[EdgeId],
) -> Result<SolidId, AlgoError> {
    let tol = Tolerance::default();
    let groups: Vec<Vec<FaceId>> = if tool_faces.is_empty() {
        Vec::new()
    } else {
        vec![tool_faces.to_vec()]
    };
    let mut store = GfaImprintStore::new(topo, solid, &groups, tool_edges)?;
    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler_with_edges(
        &mut store.topo,
        &store.sources,
        &store.tool_edges,
        tol,
        &mut arena,
    )?;
    let (store_topo, result) = crate::builder::build_imprint(
        std::mem::take(&mut store.topo),
        &arena,
        store.sources[0],
        tol,
    )?;
    store.topo = store_topo;
    store.export_solid(topo, result)
}

/// `solid` in the pieces `tool_faces` separate it into.
///
/// A piece of a tool strictly inside the solid and bounded all around, by the
/// solid's faces or by other such pieces, is a wall between two regions, and
/// each region becomes a solid. A tool that divides nothing imprints instead;
/// with no walls at all the result is one solid.
///
/// Each tool cuts the pieces the ones before it left, so tools that cross each
/// other never meet on a face. Those that separated nothing alone then go
/// together, for the walls only they close between them, meeting along shared
/// sections as sources of their own.
///
/// # Errors
///
/// Returns [`AlgoError`] if a face cannot be split along its sections or the
/// pieces do not assemble.
pub fn split_by_faces(
    topo: &mut Topology,
    solid: SolidId,
    tool_faces: &[FaceId],
) -> Result<Vec<SolidId>, AlgoError> {
    let tol = Tolerance::default();
    let groups: Vec<Vec<FaceId>> = tool_faces.iter().map(|&f| vec![f]).collect();
    let mut scratch = GfaImprintStore::new(topo, solid, &groups, &[])?;
    let (body, tool_sources) = scratch
        .sources
        .split_first()
        .map(|(&body, tools)| (body, tools.to_vec()))
        .ok_or_else(|| AlgoError::AssemblyFailed("a split by faces needs a solid".into()))?;
    let mut tools = Vec::with_capacity(tool_sources.len());
    for source in tool_sources {
        tools.extend(brepkit_topology::explorer::solid_faces(
            &scratch.topo,
            source,
        )?);
    }

    // Whether a tool separates depends on the piece: one that cuts one piece can
    // stop short in another, and one that stops short can reach across what a
    // later tool leaves. So each piece keeps the tools it has not been cut by,
    // tries them one at a time, and takes what is left together, which marks it
    // and closes off what only those tools between them can.
    let mut pieces = Vec::new();
    let mut open = vec![(body, tools)];
    while let Some((piece, mut tools)) = open.pop() {
        let mut cut = None;
        for (i, &tool) in tools.iter().enumerate() {
            let parts = split_piece(&mut scratch.topo, piece, &[tool], tol)?;
            if parts.len() > 1 {
                cut = Some((i, parts));
                break;
            }
        }
        if let Some((i, parts)) = cut {
            tools.remove(i);
            open.extend(parts.into_iter().map(|part| (part, tools.clone())));
        } else if tools.is_empty() {
            pieces.push(piece);
        } else {
            pieces.extend(split_piece(&mut scratch.topo, piece, &tools, tol)?);
        }
    }
    pieces
        .into_iter()
        .map(|piece| scratch.export_solid(topo, piece))
        .collect()
}

/// Split a piece by all of `tools` at once.
fn split_piece(
    topo: &mut Topology,
    piece: SolidId,
    tools: &[FaceId],
    tol: Tolerance,
) -> Result<Vec<SolidId>, AlgoError> {
    let groups: Vec<Vec<FaceId>> = tools.iter().map(|&f| vec![f]).collect();
    let mut store = GfaImprintStore::new(topo, piece, &groups, &[])?;
    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler_with_edges(&mut store.topo, &store.sources, &[], tol, &mut arena)?;
    let (store_topo, parts) = crate::builder::build_split_by_faces(
        std::mem::take(&mut store.topo),
        &arena,
        &store.sources,
        tol,
    )?;
    store.topo = store_topo;
    parts
        .into_iter()
        .map(|part| store.export_solid(topo, part))
        .collect()
}

/// Run a GFA boolean, also returning each result face's provenance.
///
/// Provenance is `(result face index, Some(input face index) | None)` in the
/// **caller's** topology — the basis for faithful shape-evolution tracking.
/// Indices are `FaceId::index()` values (matching the convention the evolution
/// map uses). `None` marks a synthesised face with no input origin. Unlike
/// [`boolean`], there is no identical-operand fast path; callers handle `A == B`.
///
/// # Errors
///
/// Returns [`AlgoError`] if any GFA stage fails.
pub fn boolean_with_face_origins(
    topo: &mut Topology,
    op: BooleanOp,
    solid_a: SolidId,
    solid_b: SolidId,
) -> Result<(SolidId, FaceOriginIndices), AlgoError> {
    let tol = Tolerance::default();
    let mut store = crate::ds::GfaShapeStore::new(topo, solid_a, solid_b)?;

    let mut arena = GfaArena::new();
    pave_filler::run_pave_filler(
        &mut store.topo,
        store.solid_a,
        store.solid_b,
        tol,
        &mut arena,
    )?;

    let mut builder = Builder::with_tolerance(
        std::mem::take(&mut store.topo),
        arena,
        store.solid_a,
        store.solid_b,
        tol,
    );
    builder.perform()?;

    let (store_topo, store_result, store_origins) = builder.build_result_with_origins(op)?;
    store.topo = store_topo;
    if store.topo.is_empty_solid(store_result) {
        return Ok((topo.add_empty_solid(), Vec::new()));
    }

    let (result, export_map) = store.export_solid_with_face_map(topo, store_result)?;

    // Translate store-space provenance to caller-space face indices. The export
    // map is total over result faces, so a miss is a real provenance desync and
    // is surfaced as an error rather than silently dropped (which would mark a
    // real input as `deleted`). A missing input source is `None` by design — a
    // synthesised face with no input origin.
    let mut origins = Vec::with_capacity(store_origins.len());
    for (store_out, store_src) in store_origins {
        let caller_out = export_map
            .get(&store_out.index())
            .ok_or_else(|| {
                AlgoError::AssemblyFailed(
                    "result face missing from export map (provenance desync)".into(),
                )
            })?
            .index();
        let caller_src =
            store_src.and_then(|s| store.input_face_to_caller.get(&s.index()).copied());
        origins.push((caller_out, caller_src));
    }

    Ok((result, origins))
}

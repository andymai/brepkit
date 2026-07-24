//! Builder — splits faces and classifies sub-faces for boolean assembly.
//!
//! Takes the PaveFiller's output ([`GfaArena`] with pave blocks, face info,
//! and intersection curves) and produces classified sub-faces ready for
//! boolean operation selection.
//!
//! # Flow
//!
//! 1. **`fill_images`** — map original edges to their split images
//! 2. **`fill_images_faces`** — build sub-faces from face info
//! 3. **`same_domain`** — detect coplanar face pairs
//! 4. **`classify_sub_faces`** — classify each sub-face as IN/OUT
//!
//! [`GfaArena`]: crate::ds::GfaArena

pub mod assemble;
pub mod builder_solid;
pub mod classify_2d;
pub mod face_class;
pub mod face_splitter;
pub mod fill_images;
pub mod fill_images_faces;
pub mod pcurve_compute;
pub mod plane_frame;
pub mod same_domain;
pub mod split_types;
pub mod wire_builder;

pub use face_class::FaceClass;

use std::collections::HashMap;

use brepkit_math::tolerance::Tolerance;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceId;
use brepkit_topology::solid::SolidId;

use crate::bop::{self, BooleanOp};
use crate::classifier;
use crate::ds::{GfaArena, Rank};
use crate::error::AlgoError;

/// Provenance of result faces: for each face, the input source face it was
/// derived from, or `None` for a synthesised face. Face IDs are store-local.
pub type FaceProvenance = Vec<(FaceId, Option<FaceId>)>;

/// A sub-face produced by the Builder after splitting.
#[derive(Debug, Clone)]
pub struct SubFace {
    /// The face entity in topology (same as parent if no split occurred).
    pub face_id: FaceId,
    /// The original input face this sub-face was derived from. Provenance for
    /// shape evolution — every sub-face traces back to one input argument face.
    pub source_face: FaceId,
    /// Classification relative to the opposing solid.
    pub classification: FaceClass,
    /// Which boolean argument this face came from.
    pub rank: Rank,
    /// Pre-computed interior sample point for classification.
    /// When `Some`, the classifier uses this instead of sampling from face geometry.
    /// Set by the face splitter for split sub-faces.
    pub interior_point: Option<Point3>,
}

/// Builder — orchestrates face splitting and classification.
///
/// Owns both the `Topology` and `GfaArena`, mutating them as needed.
/// After `perform()`, call `build_result()` to extract the results.
pub struct Builder {
    /// The topology containing both solids (owned, mutable).
    topo: Topology,
    /// GFA transient state from the PaveFiller (owned).
    arena: GfaArena,
    /// First boolean argument.
    solid_a: SolidId,
    /// Second boolean argument.
    solid_b: SolidId,
    /// Geometric tolerance.
    tol: Tolerance,
    /// Sub-faces produced by splitting.
    sub_faces: Vec<SubFace>,
    /// Map from face ID to its argument rank.
    face_ranks: HashMap<FaceId, Rank>,
    /// Same-domain face pairs detected by `same_domain`.
    sd_pairs: Vec<same_domain::SameDomainPair>,
    /// Within-rank SD duplicates (boolean residue accumulated across
    /// sequential operations — issue #696). Excluded before classification.
    sd_within_rank_dups: Vec<same_domain::WithinRankDuplicate>,
}

impl Builder {
    /// Create a Builder with custom tolerance.
    #[must_use]
    pub fn with_tolerance(
        topo: Topology,
        arena: GfaArena,
        solid_a: SolidId,
        solid_b: SolidId,
        tol: Tolerance,
    ) -> Self {
        Self {
            topo,
            arena,
            solid_a,
            solid_b,
            tol,
            sub_faces: Vec::new(),
            face_ranks: HashMap::new(),
            sd_pairs: Vec::new(),
            sd_within_rank_dups: Vec::new(),
        }
    }

    /// Run the Builder pipeline: fill images, split faces, classify.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if topology lookups or classification fails.
    pub fn perform(&mut self) -> Result<(), AlgoError> {
        self.build_face_ranks()?;
        self.fill_images();
        self.classify_sub_faces()?;
        Ok(())
    }

    /// Select faces for the given boolean operation and assemble them
    /// into a solid.
    ///
    /// Consumes the Builder, returning the (potentially modified) topology
    /// and the result solid ID.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if face selection produces no faces or
    /// assembly fails.
    pub fn build_result(mut self, op: BooleanOp) -> Result<(Topology, SolidId), AlgoError> {
        let selected = bop::select_faces(
            &self.sub_faces,
            op,
            &self.sd_pairs,
            &self.sd_within_rank_dups,
        );
        let cap_planes = self.partial_overlap_cap_planes(&selected);
        let solid_id = assemble::assemble_solid(&mut self.topo, &selected, &cap_planes)?;
        Ok((self.topo, solid_id))
    }

    /// Like [`Builder::build_result`], but also returns the provenance of each
    /// result face: `(result face, Some(input source face) | None)`. The source
    /// face IDs are in this builder's (store-local) topology — callers crossing
    /// the GFA shape-store boundary must translate them to caller IDs.
    pub fn build_result_with_origins(
        mut self,
        op: BooleanOp,
    ) -> Result<(Topology, SolidId, FaceProvenance), AlgoError> {
        let selected = bop::select_faces(
            &self.sub_faces,
            op,
            &self.sd_pairs,
            &self.sd_within_rank_dups,
        );
        let cap_planes = self.partial_overlap_cap_planes(&selected);
        let (solid_id, origins) =
            assemble::assemble_solid_with_origins(&mut self.topo, &selected, &cap_planes)?;
        Ok((self.topo, solid_id, origins))
    }

    /// Candidate cap planes for partial coplanar same-domain overlaps.
    ///
    /// For each planar `geometric_overlap` SD pair whose two faces were BOTH
    /// discarded by `select_faces` (the partial-overlap signature: their
    /// coincident contact is interior, but the larger face's overhang remainder
    /// is exterior and left as free edges), record the larger face's plane so
    /// the assembler can synthesise the missing remainder cap. Coextensive pairs
    /// (one face kept) and non-planar pairs are excluded.
    fn partial_overlap_cap_planes(
        &self,
        selected: &[bop::SelectedFace],
    ) -> Vec<assemble::CapPlane> {
        use brepkit_topology::face::FaceSurface;
        let kept: std::collections::HashSet<FaceId> = selected.iter().map(|s| s.face_id).collect();
        let mut planes = Vec::new();
        for pair in &self.sd_pairs {
            if !pair.geometric_overlap {
                continue;
            }
            let fa = self.sub_faces[pair.idx_a].face_id;
            let fb = self.sub_faces[pair.idx_b].face_id;
            // Only when both faces were discarded — a kept face already covers
            // the plane and capping would double it.
            if kept.contains(&fa) || kept.contains(&fb) {
                continue;
            }
            let larger = self.sub_faces[pair.representative].face_id;
            let Ok(face) = self.topo.face(larger) else {
                continue;
            };
            let FaceSurface::Plane { normal, d } = *face.surface() else {
                continue;
            };
            let Some(out_normal) = face.effective_plane_normal() else {
                continue;
            };
            planes.push(assemble::CapPlane {
                normal,
                d,
                out_normal,
            });
        }
        planes
    }

    /// Get the sub-faces, SD pairs, and topology for testing.
    #[cfg(test)]
    pub(crate) fn debug_info(&self) -> (&[SubFace], &[same_domain::SameDomainPair], &Topology) {
        (&self.sub_faces, &self.sd_pairs, &self.topo)
    }

    /// Build the face-to-rank mapping from both solids.
    fn build_face_ranks(&mut self) -> Result<(), AlgoError> {
        let faces_a = brepkit_topology::explorer::solid_faces(&self.topo, self.solid_a)?;
        for fid in faces_a {
            self.face_ranks.insert(fid, Rank::A);
        }

        let faces_b = brepkit_topology::explorer::solid_faces(&self.topo, self.solid_b)?;
        for fid in faces_b {
            self.face_ranks.insert(fid, Rank::B);
        }

        Ok(())
    }

    /// Phase 1: map edges to split images and build sub-faces.
    fn fill_images(&mut self) {
        let edge_images = fill_images::fill_edge_images(&self.arena);
        log::debug!(
            "Builder: {} original edges mapped to split images",
            edge_images.len()
        );

        self.sub_faces = fill_images_faces::fill_images_faces(
            &mut self.topo,
            &self.arena,
            &edge_images,
            &self.face_ranks,
            self.tol,
        );
        log::debug!("Builder: {} sub-faces created", self.sub_faces.len());

        // Step 3: same-domain detection (records pairs, does NOT set FaceClass)
        let sd_result = same_domain::detect_same_domain(
            &self.topo,
            &self.arena,
            &self.sub_faces,
            &self.face_ranks,
            self.tol,
        );
        self.sd_pairs = sd_result.pairs;
        self.sd_within_rank_dups = sd_result.within_rank_dups;

        // Note: SD representative replacement (replacing B's face_id with
        // A's face_id) was attempted but produces degenerate 2-edge faces
        // because both sub-face entries then point to the same face entity,
        // and the BOP selector can't distinguish them. The correct approach
        // is to let BOP keep A's face and discard B's (which it already does),
        // then fix edge sharing at the BuilderSolid level via
        // merge_duplicate_edges.
    }

    /// Phase 2: classify each sub-face as inside/outside the opposing solid.
    #[allow(clippy::too_many_lines)]
    fn classify_sub_faces(&mut self) -> Result<(), AlgoError> {
        // SD faces are excluded from non-SD BOP selection, so their
        // classification doesn't affect the result. But the ray-cast
        // classifier is non-deterministic at coplanar boundaries,
        // which can produce non-manifold results for near-tangent
        // geometries. Mark SD faces deterministically to skip ray-cast.
        //
        // Skip SD index construction entirely when no SD pairs exist
        // (common case for non-overlapping solids).
        // Only the cross-rank SD pair indices and the within-rank duplicates
        // (NOT their representatives) should bypass ray-cast classification.
        // The representative still needs normal IN/OUT classification because
        // `select_faces` routes it through the standard truth table — adding
        // it to `sd_indices` would force it to "On" with no matching pair
        // record, so `apply_sd_selection` would never pick it up and the
        // face would silently drop out.
        let sd_indices: std::collections::HashSet<usize> =
            if self.sd_pairs.is_empty() && self.sd_within_rank_dups.is_empty() {
                std::collections::HashSet::new()
            } else {
                let cross = self.sd_pairs.iter().flat_map(|p| [p.idx_a, p.idx_b]);
                let within = self.sd_within_rank_dups.iter().map(|d| d.duplicate);
                cross.chain(within).collect()
            };

        // Collect ray-cast geometry for each argument solid ONCE. Each sub-face
        // is classified against the opposing solid; rebuilding the opposing
        // solid's face geometry per sub-face was O(faces) × O(sub-faces). A
        // collection failure leaves `None`, which falls back to the per-call
        // path (identical behaviour, just not memoised).
        let geoms_a = classifier::RayCastGeoms::new(&self.topo, self.solid_a).ok();
        let geoms_b = classifier::RayCastGeoms::new(&self.topo, self.solid_b).ok();

        for (idx, sf) in self.sub_faces.iter_mut().enumerate() {
            if !sd_indices.is_empty() && sd_indices.contains(&idx) {
                // Same-domain faces are coincident by construction; the
                // ray-cast classifier is unstable at a coplanar boundary
                // (the interior sample sits on the opposing solid's face).
                // Force them "On" so `apply_sd_selection` keeps exactly one
                // representative per cross-rank pair. This includes disc
                // sub-faces (single closed-curve loops): when a disc is part
                // of an SD pair it is a flush, coincident cap on the result's
                // exterior — ray-casting it offsets the sample into the
                // opposing solid and wrongly drops the pair, leaving a hole in
                // the coincident face (e.g. a cylinder resting flush on a box
                // floor). Within-rank duplicates are dropped by `select_faces`
                // regardless of classification, so "On" is safe for them too.
                sf.classification = FaceClass::On;
                continue;
            }

            let opposing_solid = match sf.rank {
                Rank::A => self.solid_b,
                Rank::B => self.solid_a,
            };
            let opposing_geoms = match sf.rank {
                Rank::A => geoms_b.as_ref(),
                Rank::B => geoms_a.as_ref(),
            };

            let sample = if let Some(pt) = sf.interior_point {
                Ok(pt)
            } else if face_splitter::face_has_curved_lens_holes(&self.topo, sf.face_id) {
                // A curved-lens-hole wall (cylinder/cone with closed
                // Circle/Ellipse/NURBS holes) whose contained-interior search
                // failed: every generic interior sample risks landing inside the
                // removed lens, which would misclassify and drop the wall. Abort
                // the analytic split so the boolean falls back to mesh (correct)
                // rather than build a wrong B-rep.
                return Err(AlgoError::ClassificationFailed(format!(
                    "no contained interior for curved-lens wall {:?}; aborting analytic split",
                    sf.face_id
                )));
            } else {
                sample_face_interior(&self.topo, sf.face_id, self.tol)
            };

            match sample {
                Ok(point) => {
                    // Coincident-coplanar fast path: a planar sub-face lying in
                    // a plane coincident with an opposing-solid face cannot be
                    // classified by ray-cast — its interior point sits on the
                    // opposing plane and cardinal rays graze the coincident cap
                    // (voting wrongly Inside). Classify by 2D containment in the
                    // opposing face's region instead.
                    let coincident =
                        if let brepkit_topology::face::FaceSurface::Plane { normal, d } =
                            self.topo.face(sf.face_id)?.surface()
                        {
                            classifier::classify_coincident_coplanar(
                                &self.topo,
                                opposing_solid,
                                opposing_geoms,
                                sf.face_id,
                                *normal,
                                *d,
                                self.tol,
                            )?
                        } else {
                            None
                        };
                    sf.classification = match coincident {
                        Some(class) => class,
                        None => classifier::classify_point_cached(
                            &self.topo,
                            opposing_solid,
                            opposing_geoms,
                            point,
                        )?,
                    };
                    log::trace!(
                        "classify_sub_faces: idx={idx} face={:?} rank={:?} pt={point:?} class={:?}",
                        sf.face_id,
                        sf.rank,
                        sf.classification
                    );
                }
                Err(e) => {
                    return Err(AlgoError::ClassificationFailed(format!(
                        "could not sample interior of face {:?}: {e}",
                        sf.face_id
                    )));
                }
            }
        }

        let unknown_count = self
            .sub_faces
            .iter()
            .filter(|sf| sf.classification == FaceClass::Unknown)
            .count();
        let total = self.sub_faces.len();
        log::debug!(
            "Builder: {}/{total} sub-faces classified",
            total - unknown_count
        );

        if unknown_count > 0 {
            return Err(AlgoError::ClassificationFailed(format!(
                "{unknown_count} sub-faces could not be classified"
            )));
        }

        Ok(())
    }
}

/// Build an N-way FUSE result from the shared arena of an N-way pave filler.
///
/// Reuses the two-solid Builder machinery, generalized to N sources:
///
/// - **Splitting** — sections are stored face-relative (`pcurve_a == pcurve_b`),
///   so the face splitter is rank-invariant; every face is split with a
///   constant `Rank::A`. A sub-face's *global* source is tracked separately via
///   `face_source` (original input face → source index).
/// - **Classification** — each sub-face is kept iff its interior sample is
///   OUTSIDE every OTHER source (the union boundary). One `RayCastGeoms` is
///   built per source and reused across all sub-faces of that source's rivals.
/// - **Assembly** — the kept faces feed the standard solid assembler.
///
/// This slice handles the interpenetrating case (no coincident faces). If a
/// sub-face classifies `On` against another source — the signature of a
/// coincident/flush contact that needs cross-source same-domain resolution —
/// the fuse bails with an error so the caller can fall back to the sequential
/// path. Coincident-face support is the next increment.
///
/// # Errors
///
/// Returns [`AlgoError`] if a coincident contact is detected, a sub-face cannot
/// be sampled or classified, or assembly fails.
#[allow(dead_code)] // Wired into a fuse_n entry point in the next increment.
pub fn build_fuse_n<S: std::hash::BuildHasher>(
    mut topo: Topology,
    arena: GfaArena,
    sources: &[SolidId],
    face_source: &HashMap<FaceId, usize, S>,
    tol: Tolerance,
) -> Result<(Topology, SolidId), AlgoError> {
    // Split every source face. Sections are face-relative, so a constant rank
    // is correct for all of them (see the doc comment).
    let edge_images = fill_images::fill_edge_images(&arena);
    let all_a_ranks: HashMap<FaceId, Rank> = face_source.keys().map(|&f| (f, Rank::A)).collect();
    let sub_faces =
        fill_images_faces::fill_images_faces(&mut topo, &arena, &edge_images, &all_a_ranks, tol);

    // The global source of each sub-face is its parent input face's source.
    let sub_source: Vec<usize> = sub_faces
        .iter()
        .map(|sf| {
            face_source.get(&sf.source_face).copied().ok_or_else(|| {
                AlgoError::AssemblyFailed(format!(
                    "sub-face parent {:?} has no source index",
                    sf.source_face
                ))
            })
        })
        .collect::<Result<_, _>>()?;

    // Resolve coincident faces across sources first: opposite-oriented groups
    // are interior interfaces (all dropped); same-oriented groups keep one
    // representative (still verified against sources outside the group). The
    // remaining sub-faces are classified by inside/outside as usual.
    let sd = same_domain::detect_same_domain_fuse_n(&topo, &arena, &sub_faces, &sub_source, tol)?;
    let keep_reprs: HashMap<usize, std::collections::HashSet<usize>> =
        sd.keep_reprs.into_iter().collect();

    // One ray-cast geometry per source, reused across every rival sub-face.
    let geoms: Vec<Option<classifier::RayCastGeoms>> = sources
        .iter()
        .map(|&s| classifier::RayCastGeoms::new(&topo, s).ok())
        .collect();

    // Keep a sub-face iff its interior sample is outside every source in
    // `others`. A coincident `On` against a non-coincident source means the
    // same-domain pass missed a coincidence — bail to the sequential fallback.
    let keep_if_outside = |topo: &Topology,
                           sample: Point3,
                           own: usize,
                           others: &dyn Fn(usize) -> bool|
     -> Result<bool, AlgoError> {
        for (j, &other) in sources.iter().enumerate() {
            if j == own || !others(j) {
                continue;
            }
            match classifier::classify_point_cached(topo, other, geoms[j].as_ref(), sample)? {
                FaceClass::Inside => return Ok(false),
                FaceClass::On | FaceClass::CoplanarSame | FaceClass::CoplanarOpposite => {
                    return Err(AlgoError::AssemblyFailed(
                        "N-way fuse: unresolved coincident face; sequential fallback".into(),
                    ));
                }
                FaceClass::Outside | FaceClass::Unknown => {}
            }
        }
        Ok(true)
    };

    let mut selected = Vec::new();
    for (idx, sf) in sub_faces.iter().enumerate() {
        let own = sub_source[idx];
        let sample = match sf.interior_point {
            Some(pt) => pt,
            None => sample_face_interior(&topo, sf.face_id, tol)?,
        };

        let keep = if sd.grouped.contains(&idx) {
            // A coincident face: kept only if it is this group's same-oriented
            // representative AND outside every source not in the group.
            match keep_reprs.get(&idx) {
                Some(group_sources) => {
                    keep_if_outside(&topo, sample, own, &|j| !group_sources.contains(&j))?
                }
                None => false,
            }
        } else {
            // A normal sub-face: kept iff outside every other source.
            keep_if_outside(&topo, sample, own, &|_| true)?
        };

        if keep {
            selected.push(bop::SelectedFace {
                face_id: sf.face_id,
                source_face: sf.source_face,
                reversed: false,
            });
        }
    }

    let solid_id = assemble::assemble_solid(&mut topo, &selected, &[])?;
    Ok((topo, solid_id))
}

/// Sample a point in the interior of a face.
///
/// Uses the midpoint of the first boundary edge, then offsets slightly
/// inward along (edge_tangent x face_normal) to get a point that is
/// reliably inside the face — unlike a vertex centroid, which can fall
/// outside non-convex faces.
///
/// The offset distance is scaled relative to the face's bounding box
/// diagonal to handle both very small and very large faces correctly.
fn sample_face_interior(
    topo: &Topology,
    face_id: FaceId,
    tol: Tolerance,
) -> Result<Point3, AlgoError> {
    use brepkit_math::vec::Vec3;

    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let edges = wire.edges();

    if edges.is_empty() {
        return Err(AlgoError::FaceSplitFailed(format!(
            "face {face_id:?} has empty outer wire"
        )));
    }

    // Periodic faces bounded by closed curves (e.g. an unsplit cylinder
    // lateral wall between two full boundary circles): the closed-edge
    // midpoint lies on a v-extreme of the face, and the tangent-cross-normal
    // offset direction is unreliable there. Sample at the closed edge's u
    // and the midpoint of the face's v-range instead — interior in v by
    // construction, interior in u because the boundary curve spans the
    // full period.
    if !face.surface().is_planar() {
        let mut closed_mid: Option<Point3> = None;
        let mut v_min = f64::MAX;
        let mut v_max = f64::MIN;
        for oe in edges {
            let e = topo.edge(oe.edge())?;
            let sp = topo.vertex(e.start())?.point();
            let ep = topo.vertex(e.end())?.point();
            let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
            let mid = e
                .curve()
                .evaluate_with_endpoints(0.5_f64.mul_add(t1 - t0, t0), sp, ep);
            if e.start() == e.end()
                && !matches!(e.curve(), brepkit_topology::edge::EdgeCurve::Line)
                && closed_mid.is_none()
            {
                closed_mid = Some(mid);
            }
            for p in [sp, ep, mid] {
                if let Some((_, v)) = face.surface().project_point(p) {
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                }
            }
        }
        if let Some(mid) = closed_mid
            && v_max - v_min > tol.linear
            && let Some((u, _)) = face.surface().project_point(mid)
            && let Some(pt) = face.surface().evaluate(u, 0.5 * (v_min + v_max))
        {
            return Ok(pt);
        }
    }

    let mut min_pt = Point3::new(f64::MAX, f64::MAX, f64::MAX);
    let mut max_pt = Point3::new(f64::MIN, f64::MIN, f64::MIN);
    let mut point_count = 0_usize;
    for oe in edges {
        let e = topo.edge(oe.edge())?;
        let sp = topo.vertex(e.start())?.point();
        let ep = topo.vertex(e.end())?.point();
        for p in [sp, ep] {
            min_pt = Point3::new(
                min_pt.x().min(p.x()),
                min_pt.y().min(p.y()),
                min_pt.z().min(p.z()),
            );
            max_pt = Point3::new(
                max_pt.x().max(p.x()),
                max_pt.y().max(p.y()),
                max_pt.z().max(p.z()),
            );
            point_count += 1;
        }
    }
    if point_count == 0 {
        return Err(AlgoError::FaceSplitFailed(format!(
            "face {face_id:?}: could not compute bounding box (no valid edge vertices)"
        )));
    }
    let diag = (max_pt - min_pt).length();
    let offset_scale = (diag * 1e-4).max(tol.linear);

    // Take the longest boundary edge and evaluate at its midpoint. The
    // longest edge gives the most room for the inward offset, and its
    // midpoint is least likely to sit on a shared junction plane where
    // the axis-aligned classification rays graze adjacent faces.
    let mut first_oe = &edges[0];
    let mut best_len = 0.0_f64;
    for oe in edges {
        let e = topo.edge(oe.edge())?;
        let sp = topo.vertex(e.start())?.point();
        let ep = topo.vertex(e.end())?.point();
        let len = (ep - sp).length();
        if len > best_len {
            best_len = len;
            first_oe = oe;
        }
    }
    let edge = topo.edge(first_oe.edge())?;
    let start_pos = topo.vertex(edge.start())?.point();
    let end_pos = topo.vertex(edge.end())?.point();
    let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
    let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
    let mid_pt = edge
        .curve()
        .evaluate_with_endpoints(t_mid, start_pos, end_pos);

    let tangent = edge
        .curve()
        .tangent_with_endpoints(t_mid, start_pos, end_pos);
    let surface = face.surface();

    let face_normal = if let Some((u, v)) = surface.project_point(mid_pt) {
        surface.normal(u, v)
    } else {
        // Plane: normal is constant
        match surface {
            brepkit_topology::face::FaceSurface::Plane { normal, .. } => *normal,
            _ => Vec3::new(0.0, 0.0, 1.0),
        }
    };

    // Inward direction: tangent x face_normal points into the face interior
    // (assuming CCW winding when viewed from the face normal direction)
    let inward = tangent.cross(face_normal);
    let inward_len = inward.length();

    let base_offset = if inward_len > 1e-12 {
        inward * (offset_scale / inward_len)
    } else {
        // Degenerate — use a tiny offset along the face normal instead
        face_normal * offset_scale
    };

    // For a planar face, verify the sample lands strictly inside the
    // (possibly concave) boundary polygon. The tangent×normal sign is
    // unreliable on inner/notch edges and reversed winding, and a
    // boundary-vertex centroid can fall in a concavity — e.g. the notch of an
    // L-shaped face left by a corner cut — so a centroid-based flip points the
    // sample OUTSIDE the face and into the opposing solid, misclassifying a
    // thin sliver. Project the boundary, then pick the offset sign (shrinking
    // the magnitude for strips thinner than the offset) that lands inside.
    if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = surface {
        let mut poly = Vec::with_capacity(edges.len());
        for oe in edges {
            let e = topo.edge(oe.edge())?;
            poly.push(topo.vertex(oe.oriented_start(e))?.point());
        }
        // A boundary with >= 3 vertices forms a real polygon to test against.
        // A planar face bounded by a single closed curve (one circle/ellipse
        // edge → <3 vertices) has no polygon; its centroid is the disc center
        // (interior), so it falls through to the centroid heuristic below.
        if inward_len > 1e-12 && poly.len() >= 3 {
            let frame = plane_frame::PlaneFrame::from_plane_face(*normal, &poly);
            let poly2d: Vec<_> = poly.iter().map(|p| frame.project(*p)).collect();
            let eps = classify_2d::boundary_eps(&poly2d);
            // Try LARGE offsets first, then shrink. The base offset is
            // diag·1e-4 — a sample that close to the boundary edge can hug a
            // coincident interface plane of the opposing solid (a frustum
            // wall's longest edge lies ON the coincident bottom cap), where
            // the ray-cast classifier turns unstable and mirror-image walls
            // classify differently. A deeper sample is strictly better
            // whenever the polygon containment check admits it; thin slivers
            // reject the large candidates and fall through to the fine
            // scales. 28 halvings from 64 reach scale ~2.4e-7 (min offset
            // ~diag·2.4e-11), below any physically meaningful strip width,
            // so the loop only exits to the fallback for a near-zero-area
            // (degenerate) face.
            let mut scale = 64.0_f64;
            for _ in 0..28 {
                for sign in [1.0_f64, -1.0] {
                    let cand = mid_pt + base_offset * (sign * scale);
                    let c2 = frame.project(cand);
                    if classify_2d::point_in_polygon_2d(c2, &poly2d)
                        && classify_2d::distance_to_polygon_boundary(c2, &poly2d) > eps
                    {
                        return Ok(cand);
                    }
                }
                scale *= 0.5;
            }
            // Near-zero-area face: try a robust interior point of the projected
            // boundary. Verify it before use — its last-resort path returns the
            // vertex centroid, which can fall outside a concave boundary. If
            // even that is exterior, fall back to the edge midpoint (on the
            // boundary, never exterior) rather than a known-bad sample.
            let ip = classify_2d::sample_interior_point(&poly2d);
            if classify_2d::point_in_polygon_2d(ip, &poly2d) {
                return Ok(frame.evaluate(ip.x(), ip.y()));
            }
            return Ok(mid_pt);
        }
    }

    // Non-planar surfaces: the tangent×normal direction assumes CCW winding;
    // reversed or CW-wound faces flip it, sending the sample outside the
    // face. Use the boundary vertex centroid to pick the side that points
    // into the face.
    let mut offset = base_offset;
    let centroid = {
        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        let mut n = 0_usize;
        for oe in edges {
            let e = topo.edge(oe.edge())?;
            for vid in [e.start(), e.end()] {
                let p = topo.vertex(vid)?.point();
                sum += Vec3::new(p.x(), p.y(), p.z());
                n += 1;
            }
        }
        #[allow(clippy::cast_precision_loss)]
        Point3::new(sum.x() / n as f64, sum.y() / n as f64, sum.z() / n as f64)
    };
    if offset.dot(centroid - mid_pt) < 0.0 {
        offset = offset * -1.0;
    }

    let interior_pt = mid_pt + offset;

    // Project back onto the surface to ensure the point is on-surface
    if let Some((u, v)) = surface.project_point(interior_pt)
        && let Some(on_surface) = surface.evaluate(u, v)
    {
        return Ok(on_surface);
    }

    // Planes have no UV projection, but the inward offset is already in-plane,
    // so the offset point itself is the on-surface sample. This reaches a
    // planar face only when its boundary has < 3 vertices (a single closed
    // circle/ellipse edge); the centroid above is the disc center, so the
    // flipped offset points into the disc.
    if matches!(surface, brepkit_topology::face::FaceSurface::Plane { .. }) && inward_len > 1e-12 {
        return Ok(interior_pt);
    }

    // Fallback: use the midpoint itself (it's on the boundary, not ideal
    // but better than a centroid that may be outside the face)
    Ok(mid_pt)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use brepkit_math::vec::Vec3;
    use brepkit_topology::builder::{make_face_from_wire, make_polygon_wire};

    #[test]
    fn sample_face_interior_thin_l_frame_lands_in_strip() {
        // L-frame: a side-1.0001 square with a side-1.0 corner notch removed at
        // the origin, leaving a 0.0001-thin strip. The boundary-vertex centroid
        // (~0.667, ~0.667) falls inside the removed notch, so the old
        // centroid-based flip placed the sample outside the face. The sample
        // must instead land in the strip (one coordinate >= 1.0).
        let mut topo = Topology::new();
        let s = 1.0001;
        let n = 1.0;
        let pts = vec![
            Point3::new(n, 0.0, 0.0),
            Point3::new(s, 0.0, 0.0),
            Point3::new(s, s, 0.0),
            Point3::new(0.0, s, 0.0),
            Point3::new(0.0, n, 0.0),
            Point3::new(n, n, 0.0),
        ];
        let wire = make_polygon_wire(&mut topo, &pts, 1e-7).unwrap();
        let face = make_face_from_wire(&mut topo, wire).unwrap();

        let pt = sample_face_interior(&topo, face, Tolerance::default()).unwrap();
        // Strip check (not in the notch)...
        assert!(
            pt.x() >= n - 1e-9 || pt.y() >= n - 1e-9,
            "sample {pt:?} fell in the notch instead of the L-frame strip"
        );
        // ...and a direct interior-membership proof against the L-polygon.
        let frame = plane_frame::PlaneFrame::from_plane_face(Vec3::new(0.0, 0.0, 1.0), &pts);
        let poly2d: Vec<_> = pts.iter().map(|p| frame.project(*p)).collect();
        assert!(
            classify_2d::point_in_polygon_2d(frame.project(pt), &poly2d),
            "sample {pt:?} is not inside the L-frame polygon"
        );
    }

    #[test]
    fn sample_face_interior_planar_disc_lands_inside() {
        // A planar disc bounded by a single closed circle edge has < 3 boundary
        // vertices, so the point-in-polygon path can't apply. The sample must
        // still land inside the disc, not on the bounding circle (the
        // degenerate-polygon failure mode).
        use brepkit_topology::builder::make_circle_edge;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();
        let edge = make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            1.0,
            1e-7,
        )
        .unwrap();
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap());
        let face = make_face_from_wire(&mut topo, wire).unwrap();

        let pt = sample_face_interior(&topo, face, Tolerance::default()).unwrap();
        let r = pt.x().hypot(pt.y());
        // Strictly inside the unit disc. The degenerate-polygon failure mode
        // (point_in_polygon on a single projected vertex → boundary vertex)
        // would return a point on the circle (r == 1.0).
        assert!(
            r < 1.0 - 1e-9,
            "disc sample (r={r}) should be interior, not on the bounding circle"
        );
    }
}

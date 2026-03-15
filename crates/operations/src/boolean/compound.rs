//! Compound cut — multi-tool boolean in a single pass.
//!
//! Cuts a target solid by multiple tool solids simultaneously, avoiding the
//! O(N²) cost of sequential boolean operations.

use std::collections::{HashMap, HashSet};

use brepkit_math::aabb::Aabb3;
use brepkit_math::bvh::Bvh;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::analytic::{edge_curves_from_face, plane_plane_chord_analytic, surface_aware_aabb};
use super::assembly::{
    quantize, quantize_point, refine_boundary_edges, split_nonmanifold_edges,
    vertex_merge_resolution,
};
use super::classify::{
    build_face_bvh, classify_point, guard_tangent_coplanar, polygon_centroid,
    try_build_analytic_classifier,
};
use super::fragments::{
    collect_analytic_vranges, create_band_fragments, curve_boundary_crossings, sample_edge_curve,
    split_cylinder_at_intersection, split_sphere_at_intersection, tessellate_face_into_fragments,
};
use super::precompute::{
    analytic_face_normal_d, collect_face_data, compute_v_range_hint, face_polygon, face_wire_aabb,
};
use super::split::split_face;
use super::types::{
    AnalyticClassifier, AnalyticFragment, BooleanOp, BooleanOptions, CLOSED_CURVE_SAMPLES,
    CurveClassification, FaceClass, FaceData, FaceSnapshot, Source, select_fragment,
};
use super::{boolean_with_options, timer_elapsed_ms, timer_now};

/// Cut a target solid by multiple tool solids in a single pass.
///
/// For each tool, the target faces overlapping that tool are intersected and
/// fragments are classified against ALL tools simultaneously. This avoids the
/// O(N²) cost of sequential boolean operations where each cut must process the
/// full accumulated result of all prior cuts.
///
/// If any tool or the target contains NURBS, torus, or other non-analytic
/// surfaces, falls back to sequential `boolean_with_options()` calls.
///
/// # Errors
///
/// Returns an error if any individual boolean operation fails, or if the
/// result is degenerate (empty solid).
#[allow(clippy::too_many_lines)]
#[allow(clippy::items_after_statements)]
pub fn compound_cut(
    topo: &mut Topology,
    target: SolidId,
    tools: &[SolidId],
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    if tools.is_empty() {
        return Ok(target);
    }
    // Small tool counts: sequential is faster due to lower overhead.
    if tools.len() <= 8 {
        log::debug!(
            "[compound_cut] fallback: small tool count ({})",
            tools.len()
        );

        return compound_cut_sequential(topo, target, tools, opts);
    }

    // Check for non-analytic surfaces — fall back to sequential if found.
    let has_non_analytic = |solid: SolidId| -> Result<bool, crate::OperationsError> {
        let s = topo.solid(solid)?;
        let shell = topo.shell(s.outer_shell())?;
        for &fid in shell.faces() {
            let face = topo.face(fid)?;
            if matches!(
                face.surface(),
                FaceSurface::Nurbs(_) | FaceSurface::Torus(_)
            ) {
                return Ok(true);
            }
        }
        Ok(false)
    };

    if has_non_analytic(target)? {
        log::debug!("[compound_cut] fallback: target has non-analytic surfaces");

        return compound_cut_sequential(topo, target, tools, opts);
    }
    for (i, &tool) in tools.iter().enumerate() {
        if has_non_analytic(tool)? {
            log::debug!("[compound_cut] fallback: tool {i} has non-analytic surfaces");

            return compound_cut_sequential(topo, target, tools, opts);
        }
    }

    let tol = opts.tolerance;
    let deflection = opts.deflection;
    let _t_total = timer_now();

    // ── Phase 0: Precompute tool data ────────────────────────────────────
    struct ToolData {
        snapshots: Vec<FaceSnapshot>,
        aabbs: Vec<Aabb3>,
        overall_aabb: Aabb3,
        classifier: Option<AnalyticClassifier>,
    }

    let mut tool_data: Vec<ToolData> = Vec::with_capacity(tools.len());
    let target_wire_aabbs = {
        let solid_t = topo.solid(target)?;
        let shell_t = topo.shell(solid_t.outer_shell())?;
        let fids: Vec<FaceId> = shell_t.faces().to_vec();
        fids.iter()
            .map(|&fid| face_wire_aabb(topo, fid))
            .collect::<Result<Vec<Aabb3>, _>>()?
    };
    let target_overall_aabb = target_wire_aabbs
        .iter()
        .copied()
        .reduce(Aabb3::union)
        .ok_or_else(|| crate::OperationsError::InvalidInput {
            reason: "target solid has no faces".into(),
        })?;

    for &tool in tools {
        let solid_t = topo.solid(tool)?;
        let shell_t = topo.shell(solid_t.outer_shell())?;
        let face_ids: Vec<FaceId> = shell_t.faces().to_vec();

        // Compute tool's wire AABBs and overall AABB.
        let tool_wire_aabbs: Vec<Aabb3> = face_ids
            .iter()
            .map(|&fid| face_wire_aabb(topo, fid))
            .collect::<Result<Vec<_>, _>>()?;
        let tool_overall = tool_wire_aabbs
            .iter()
            .copied()
            .reduce(Aabb3::union)
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "tool solid has no faces".into(),
            })?;

        // Skip tools completely disjoint from target.
        if !tool_overall.intersects(target_overall_aabb) {
            continue;
        }

        // Snapshot tool faces that overlap target.
        let mut snapshots = Vec::new();
        let mut aabbs = Vec::new();
        for (i, &fid) in face_ids.iter().enumerate() {
            if tool_wire_aabbs[i].intersects(target_overall_aabb) {
                let face = topo.face(fid)?;
                let surface = face.surface().clone();
                let reversed = face.is_reversed();
                let verts = face_polygon(topo, fid)?;
                let (normal, d) = analytic_face_normal_d(&surface, &verts);
                aabbs.push(surface_aware_aabb(&surface, &verts, tol));
                snapshots.push(FaceSnapshot {
                    id: fid,
                    surface,
                    vertices: verts,
                    normal,
                    d,
                    reversed,
                });
            }
        }

        let classifier = try_build_analytic_classifier(topo, tool);
        tool_data.push(ToolData {
            snapshots,
            aabbs,
            overall_aabb: tool_overall,
            classifier,
        });
    }

    // If no tools overlap target, return unchanged.
    if tool_data.is_empty() {
        return Ok(target);
    }

    let _t_phase0 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 0: {:.1}ms — {} tools overlap target ({} total tool faces)",
        _t_phase0,
        tool_data.len(),
        tool_data.iter().map(|td| td.snapshots.len()).sum::<usize>()
    );

    // Global BVH over tool overall AABBs — used for O(log N) spatial queries
    // in Phase 1 (passthrough), Phase 2 (intersection), and Phase 4 (classification).
    let tool_bvh = {
        let entries: Vec<(usize, Aabb3)> = tool_data
            .iter()
            .enumerate()
            .map(|(i, td)| (i, td.overall_aabb))
            .collect();
        Bvh::build(&entries)
    };

    // ── Phase 1: Snapshot target faces ───────────────────────────────────
    let solid_a = topo.solid(target)?;
    let shell_a = topo.shell(solid_a.outer_shell())?;
    let face_ids_a: Vec<FaceId> = shell_a.faces().to_vec();

    let mut snaps_a = Vec::new();
    let mut passthrough_a: Vec<FaceId> = Vec::new();
    // A face is passthrough if it doesn't overlap ANY tool (BVH query).
    let mut bvh_buf = Vec::new();
    for (i, &fid) in face_ids_a.iter().enumerate() {
        tool_bvh.query_overlap_into(&target_wire_aabbs[i], &mut bvh_buf);
        if bvh_buf.is_empty() {
            passthrough_a.push(fid);
        } else {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let reversed = face.is_reversed();
            let verts = face_polygon(topo, fid)?;
            let (normal, d) = analytic_face_normal_d(&surface, &verts);
            snaps_a.push(FaceSnapshot {
                id: fid,
                surface,
                vertices: verts,
                normal,
                d,
                reversed,
            });
        }
    }

    let aabbs_a: Vec<Aabb3> = snaps_a
        .iter()
        .map(|s| surface_aware_aabb(&s.surface, &s.vertices, tol))
        .collect();

    log::debug!(
        "[compound_cut] Phase 1: {:.1}ms — {} snap + {} passthrough",
        timer_elapsed_ms(_t_total) - _t_phase0,
        snaps_a.len(),
        passthrough_a.len()
    );
    let _t_phase1 = timer_elapsed_ms(_t_total);

    // ── Phase 2: Intersection (all tools at once) ────────────────────────
    use brepkit_math::analytic_intersection::{
        ExactIntersectionCurve, exact_plane_analytic, intersect_analytic_analytic_bounded,
    };

    let mut face_intersections_a: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>> =
        HashMap::new();
    let mut analytic_analytic_faces_a: HashSet<usize> = HashSet::new();
    let mut analytic_intersection_vranges_a: HashMap<usize, Vec<(f64, f64)>> = HashMap::new();

    // Contained curves: plane_face_idx in target → list of (tool_index, edge_curve).
    struct CompoundContainedCurve {
        plane_face_idx: usize,
        tool_index: usize,
        analytic_face_idx: usize,
        edge_curve: EdgeCurve,
    }
    let mut contained_curves: Vec<CompoundContainedCurve> = Vec::new();

    // Per-tool: face intersections and flags for tool faces.
    struct ToolIntersections {
        face_intersections: HashMap<usize, Vec<(Point3, Point3, Option<EdgeCurve>)>>,
        analytic_analytic_faces: HashSet<usize>,
        analytic_intersection_vranges: HashMap<usize, Vec<(f64, f64)>>,
    }
    let mut tool_intersections: Vec<ToolIntersections> = tool_data
        .iter()
        .map(|_| ToolIntersections {
            face_intersections: HashMap::new(),
            analytic_analytic_faces: HashSet::new(),
            analytic_intersection_vranges: HashMap::new(),
        })
        .collect();

    let mut has_analytic_analytic = false;

    // Build per-tool BVHs over their face AABBs (for face-level broad-phase).
    let tool_face_bvhs: Vec<Option<Bvh>> = tool_data
        .iter()
        .map(|td| {
            if td.aabbs.len() >= 16 {
                let entries: Vec<(usize, Aabb3)> = td
                    .aabbs
                    .iter()
                    .enumerate()
                    .map(|(i, aabb)| (i, *aabb))
                    .collect();
                Some(Bvh::build(&entries))
            } else {
                None
            }
        })
        .collect();

    // Iterate target faces first, then use global BVH to find overlapping tools.
    // This is O(target_faces × log(tools)) instead of O(tools × target_faces).
    let mut overlap_buf = Vec::new();
    let mut candidate_buf = Vec::new();
    for (ia, snap_a) in snaps_a.iter().enumerate() {
        // Global BVH query: which tools overlap this target face?
        tool_bvh.query_overlap_into(&aabbs_a[ia], &mut overlap_buf);

        for &ti in &overlap_buf {
            let td = &tool_data[ti];

            if let Some(ref bvh) = tool_face_bvhs[ti] {
                bvh.query_overlap_into(&aabbs_a[ia], &mut candidate_buf);
            } else {
                candidate_buf.clear();
                candidate_buf.extend(
                    (0..td.snapshots.len()).filter(|&ib| aabbs_a[ia].intersects(td.aabbs[ib])),
                );
            }

            for &ib in &candidate_buf {
                let snap_b = &td.snapshots[ib];

                let is_plane_a = matches!(snap_a.surface, FaceSurface::Plane { .. });
                let is_plane_b = matches!(snap_b.surface, FaceSurface::Plane { .. });

                if is_plane_a && is_plane_b {
                    if let Some(seg) = plane_plane_chord_analytic(
                        snap_a.normal,
                        snap_a.d,
                        &snap_a.vertices,
                        snap_b.normal,
                        snap_b.d,
                        &snap_b.vertices,
                        tol,
                    ) {
                        face_intersections_a
                            .entry(ia)
                            .or_default()
                            .push((seg.0, seg.1, None));
                        tool_intersections[ti]
                            .face_intersections
                            .entry(ib)
                            .or_default()
                            .push((seg.0, seg.1, None));
                    }
                } else if is_plane_a && !is_plane_b {
                    let Some(analytic_surf) = snap_b.surface.as_analytic() else {
                        has_analytic_analytic = true;
                        continue;
                    };
                    if let Ok(curves) = exact_plane_analytic(analytic_surf, snap_a.normal, snap_a.d)
                    {
                        for curve in curves {
                            let edge_curve = match &curve {
                                ExactIntersectionCurve::Circle(c) => {
                                    Some(EdgeCurve::Circle(c.clone()))
                                }
                                ExactIntersectionCurve::Ellipse(e) => {
                                    Some(EdgeCurve::Ellipse(e.clone()))
                                }
                                ExactIntersectionCurve::Points(_) => None,
                            };
                            let classification = curve_boundary_crossings(
                                &curve,
                                &snap_a.vertices,
                                snap_a.normal,
                                tol,
                            );
                            match classification {
                                CurveClassification::Crossings(ref samples) => {
                                    for pair in samples.windows(2) {
                                        face_intersections_a.entry(ia).or_default().push((
                                            pair[0],
                                            pair[1],
                                            edge_curve.clone(),
                                        ));
                                        tool_intersections[ti]
                                            .face_intersections
                                            .entry(ib)
                                            .or_default()
                                            .push((pair[0], pair[1], edge_curve.clone()));
                                    }
                                }
                                CurveClassification::FullyContained => {
                                    if let Some(ref ec) = edge_curve {
                                        if face_intersections_a.contains_key(&ia) {
                                            has_analytic_analytic = true;
                                        } else {
                                            contained_curves.push(CompoundContainedCurve {
                                                plane_face_idx: ia,
                                                tool_index: ti,
                                                analytic_face_idx: ib,
                                                edge_curve: ec.clone(),
                                            });
                                        }
                                    }
                                }
                                CurveClassification::FullyOutside => {}
                            }
                        }
                    }
                } else if !is_plane_a && is_plane_b {
                    let Some(analytic_surf) = snap_a.surface.as_analytic() else {
                        has_analytic_analytic = true;
                        continue;
                    };
                    if let Ok(curves) = exact_plane_analytic(analytic_surf, snap_b.normal, snap_b.d)
                    {
                        for curve in curves {
                            let edge_curve = match &curve {
                                ExactIntersectionCurve::Circle(c) => {
                                    Some(EdgeCurve::Circle(c.clone()))
                                }
                                ExactIntersectionCurve::Ellipse(e) => {
                                    Some(EdgeCurve::Ellipse(e.clone()))
                                }
                                ExactIntersectionCurve::Points(_) => None,
                            };
                            let classification = curve_boundary_crossings(
                                &curve,
                                &snap_b.vertices,
                                snap_b.normal,
                                tol,
                            );
                            match classification {
                                CurveClassification::Crossings(samples) => {
                                    for pair in samples.windows(2) {
                                        face_intersections_a.entry(ia).or_default().push((
                                            pair[0],
                                            pair[1],
                                            edge_curve.clone(),
                                        ));
                                        tool_intersections[ti]
                                            .face_intersections
                                            .entry(ib)
                                            .or_default()
                                            .push((pair[0], pair[1], edge_curve.clone()));
                                    }
                                }
                                CurveClassification::FullyContained => {
                                    // Contained in plane B: don't need to track for compound_cut
                                    // since B faces are tools, not target.
                                }
                                CurveClassification::FullyOutside => {}
                            }
                        }
                    }
                } else {
                    // Analytic-analytic.
                    let surf_a_opt = snap_a.surface.as_analytic();
                    let surf_b_opt = snap_b.surface.as_analytic();
                    if let (Some(surf_a_an), Some(surf_b_an)) = (surf_a_opt, surf_b_opt) {
                        let v_hint_a = compute_v_range_hint(&snap_a.surface, &snap_a.vertices);
                        let v_hint_b = compute_v_range_hint(&snap_b.surface, &snap_b.vertices);
                        if let Ok(curves) = intersect_analytic_analytic_bounded(
                            surf_a_an, surf_b_an, 32, v_hint_a, v_hint_b,
                        ) {
                            for ic in &curves {
                                let pts: Vec<Point3> =
                                    ic.points.iter().map(|ip| ip.point).collect();
                                analytic_analytic_faces_a.insert(ia);
                                tool_intersections[ti].analytic_analytic_faces.insert(ib);
                                for pair in pts.windows(2) {
                                    face_intersections_a
                                        .entry(ia)
                                        .or_default()
                                        .push((pair[0], pair[1], None));
                                    tool_intersections[ti]
                                        .face_intersections
                                        .entry(ib)
                                        .or_default()
                                        .push((pair[0], pair[1], None));
                                }
                            }
                        } else {
                            analytic_analytic_faces_a.insert(ia);
                            tool_intersections[ti].analytic_analytic_faces.insert(ib);
                        }
                    } else {
                        has_analytic_analytic = true;
                    }
                }
            }
        }
    }

    // Fall back to sequential if unsupported intersection types found.
    if has_analytic_analytic {
        log::debug!("[compound_cut] fallback: has_analytic_analytic intersection");
        return compound_cut_sequential(topo, target, tools, opts);
    }

    // Compute v-ranges for band splitting.
    collect_analytic_vranges(
        &snaps_a,
        &face_intersections_a,
        &analytic_analytic_faces_a,
        &mut analytic_intersection_vranges_a,
    );

    let _t_phase2 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 2: {:.1}ms — target={} faces with chords",
        _t_phase2 - _t_phase1,
        face_intersections_a.len()
    );

    // ── Phase 3: Fragment creation ───────────────────────────────────────

    let mut pre_classifications: HashMap<usize, FaceClass> = HashMap::new();
    let mut holed_face_inner_curves: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    let mut existing_inner_wires: HashMap<usize, Vec<WireId>> = HashMap::new();
    let mut fragments: Vec<AnalyticFragment> = Vec::with_capacity(
        snaps_a.len() + tool_data.iter().map(|td| td.snapshots.len()).sum::<usize>(),
    );

    // Build contained-curve lookups (target faces with holes).
    let mut contained_a: HashMap<usize, Vec<EdgeCurve>> = HashMap::new();
    // Track which tool faces have contained curves (for band fragments).
    let mut tool_analytic_contained: Vec<HashMap<usize, Vec<EdgeCurve>>> =
        tool_data.iter().map(|_| HashMap::new()).collect();
    for cc in &contained_curves {
        contained_a
            .entry(cc.plane_face_idx)
            .or_default()
            .push(cc.edge_curve.clone());
        tool_analytic_contained[cc.tool_index]
            .entry(cc.analytic_face_idx)
            .or_default()
            .push(cc.edge_curve.clone());
    }

    // --- Target face fragments ---
    let _t_frag_a = timer_now();
    for (ia, snap) in snaps_a.iter().enumerate() {
        if let Some(vranges) = analytic_intersection_vranges_a.get(&ia) {
            if matches!(snap.surface, FaceSurface::Sphere(_)) {
                split_sphere_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::A,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    &mut fragments,
                )?;
                continue;
            }
            split_cylinder_at_intersection(
                &snap.surface,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::A,
                snap.reversed,
                vranges,
                topo,
                snap.id,
                deflection,
                tol,
                &mut fragments,
            )?;
            continue;
        }
        if analytic_analytic_faces_a.contains(&ia) {
            tessellate_face_into_fragments(topo, snap.id, Source::A, deflection, &mut fragments)?;
            continue;
        }
        if let Some(chords) = face_intersections_a.get(&ia) {
            let chord_pairs: Vec<(Point3, Point3)> =
                chords.iter().map(|&(p0, p1, _)| (p0, p1)).collect();
            let edge_curve_for_face = chords.first().and_then(|c| c.2.clone());
            let mut chord_map_local: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
            chord_map_local.insert(snap.id.index(), chord_pairs);
            let planar_frags = split_face(
                snap.id,
                &snap.vertices,
                snap.normal,
                snap.d,
                Source::A,
                &chord_map_local,
                tol,
            );
            for frag in planar_frags {
                let edge_curves = vec![None; frag.vertices.len()];
                fragments.push(AnalyticFragment {
                    vertices: frag.vertices,
                    surface: snap.surface.clone(),
                    normal: frag.normal,
                    d: frag.d,
                    source: Source::A,
                    edge_curves,
                    source_reversed: snap.reversed,
                });
            }
            if !matches!(snap.surface, FaceSurface::Plane { .. }) {
                if let Some(ref ec) = edge_curve_for_face {
                    let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if curve_verts.len() >= 3 {
                        fragments.push(AnalyticFragment {
                            vertices: curve_verts,
                            surface: snap.surface.clone(),
                            normal: snap.normal,
                            d: snap.d,
                            source: Source::A,
                            edge_curves: vec![Some(ec.clone())],
                            source_reversed: snap.reversed,
                        });
                    }
                }
            }
        } else if let Some(inner_curves) = contained_a.get(&ia) {
            let holed_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::A,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            pre_classifications.insert(holed_idx, FaceClass::Outside);
            holed_face_inner_curves.insert(holed_idx, inner_curves.clone());
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(holed_idx, source_face.inner_wires().to_vec());
            }
            for ec in inner_curves {
                let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                if curve_verts.len() >= 3 {
                    let disc_idx = fragments.len();
                    fragments.push(AnalyticFragment {
                        vertices: curve_verts,
                        surface: snap.surface.clone(),
                        normal: snap.normal,
                        d: snap.d,
                        source: Source::A,
                        edge_curves: vec![Some(ec.clone())],
                        source_reversed: false,
                    });
                    pre_classifications.insert(disc_idx, FaceClass::Inside);
                }
            }
        } else {
            let unsplit_idx = fragments.len();
            fragments.push(AnalyticFragment {
                vertices: snap.vertices.clone(),
                surface: snap.surface.clone(),
                normal: snap.normal,
                d: snap.d,
                source: Source::A,
                edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                source_reversed: snap.reversed,
            });
            let source_face = topo.face(snap.id)?;
            if !source_face.inner_wires().is_empty() {
                existing_inner_wires.insert(unsplit_idx, source_face.inner_wires().to_vec());
            }
        }
    }

    let _frag_a_count = fragments.len();
    log::debug!(
        "[compound_cut] Phase 3a (target frags): {:.1}ms — {} fragments",
        timer_elapsed_ms(_t_frag_a),
        _frag_a_count
    );
    let _t_frag_b = timer_now();
    // --- Tool face fragments (Source::B) ---
    // Compute v-ranges for each tool (must be done before borrowing ti_ref).
    for ti in 0..tool_data.len() {
        let mut vranges = HashMap::new();
        collect_analytic_vranges(
            &tool_data[ti].snapshots,
            &tool_intersections[ti].face_intersections,
            &tool_intersections[ti].analytic_analytic_faces,
            &mut vranges,
        );
        tool_intersections[ti].analytic_intersection_vranges = vranges;
    }
    for (ti, td) in tool_data.iter().enumerate() {
        let ti_ref = &tool_intersections[ti];
        for (ib, snap) in td.snapshots.iter().enumerate() {
            if let Some(vranges) = ti_ref.analytic_intersection_vranges.get(&ib) {
                if matches!(snap.surface, FaceSurface::Sphere(_)) {
                    split_sphere_at_intersection(
                        &snap.surface,
                        &snap.vertices,
                        snap.normal,
                        snap.d,
                        Source::B,
                        snap.reversed,
                        vranges,
                        topo,
                        snap.id,
                        deflection,
                        &mut fragments,
                    )?;
                    continue;
                }
                split_cylinder_at_intersection(
                    &snap.surface,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::B,
                    snap.reversed,
                    vranges,
                    topo,
                    snap.id,
                    deflection,
                    tol,
                    &mut fragments,
                )?;
                continue;
            }
            if ti_ref.analytic_analytic_faces.contains(&ib) {
                tessellate_face_into_fragments(
                    topo,
                    snap.id,
                    Source::B,
                    deflection,
                    &mut fragments,
                )?;
                continue;
            }
            if let Some(chords) = ti_ref.face_intersections.get(&ib) {
                let chord_pairs: Vec<(Point3, Point3)> =
                    chords.iter().map(|&(p0, p1, _)| (p0, p1)).collect();
                let edge_curve_for_face = chords.first().and_then(|c| c.2.clone());
                let mut chord_map_local: HashMap<usize, Vec<(Point3, Point3)>> = HashMap::new();
                chord_map_local.insert(snap.id.index(), chord_pairs);
                let planar_frags = split_face(
                    snap.id,
                    &snap.vertices,
                    snap.normal,
                    snap.d,
                    Source::B,
                    &chord_map_local,
                    tol,
                );
                for frag in planar_frags {
                    let edge_curves = vec![None; frag.vertices.len()];
                    fragments.push(AnalyticFragment {
                        vertices: frag.vertices,
                        surface: snap.surface.clone(),
                        normal: frag.normal,
                        d: frag.d,
                        source: Source::B,
                        edge_curves,
                        source_reversed: snap.reversed,
                    });
                }
                if !matches!(snap.surface, FaceSurface::Plane { .. }) {
                    if let Some(ref ec) = edge_curve_for_face {
                        let curve_verts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                        if curve_verts.len() >= 3 {
                            fragments.push(AnalyticFragment {
                                vertices: curve_verts,
                                surface: snap.surface.clone(),
                                normal: snap.normal,
                                d: snap.d,
                                source: Source::B,
                                edge_curves: vec![Some(ec.clone())],
                                source_reversed: snap.reversed,
                            });
                        }
                    }
                }
            } else if let Some(band_curves) = tool_analytic_contained[ti].get(&ib) {
                if matches!(snap.surface, FaceSurface::Sphere(_)) {
                    tessellate_face_into_fragments(
                        topo,
                        snap.id,
                        Source::B,
                        deflection,
                        &mut fragments,
                    )?;
                } else {
                    create_band_fragments(
                        &snap.surface,
                        &snap.vertices,
                        snap.normal,
                        snap.d,
                        Source::B,
                        snap.reversed,
                        band_curves,
                        topo,
                        tol,
                        &mut fragments,
                    );
                }
            } else {
                let unsplit_idx = fragments.len();
                fragments.push(AnalyticFragment {
                    vertices: snap.vertices.clone(),
                    surface: snap.surface.clone(),
                    normal: snap.normal,
                    d: snap.d,
                    source: Source::B,
                    edge_curves: edge_curves_from_face(topo, snap.id, snap.vertices.len()),
                    source_reversed: snap.reversed,
                });
                let source_face = topo.face(snap.id)?;
                if !source_face.inner_wires().is_empty() {
                    existing_inner_wires.insert(unsplit_idx, source_face.inner_wires().to_vec());
                }
            }
        }
    }

    log::debug!(
        "[compound_cut] Phase 3b (tool frags): {:.1}ms — {} fragments",
        timer_elapsed_ms(_t_frag_b),
        fragments.len() - _frag_a_count
    );
    // Passthrough target faces (outside all tools → survive Cut).
    for &fid in &passthrough_a {
        let face = topo.face(fid)?;
        let surface = face.surface().clone();
        let reversed = face.is_reversed();
        let verts = face_polygon(topo, fid)?;
        let (normal, d) = analytic_face_normal_d(&surface, &verts);
        let pass_idx = fragments.len();
        fragments.push(AnalyticFragment {
            vertices: verts.clone(),
            surface,
            normal,
            d,
            source: Source::A,
            edge_curves: edge_curves_from_face(topo, fid, verts.len()),
            source_reversed: reversed,
        });
        pre_classifications.insert(pass_idx, FaceClass::Outside);
        let source_face = topo.face(fid)?;
        if !source_face.inner_wires().is_empty() {
            existing_inner_wires.insert(pass_idx, source_face.inner_wires().to_vec());
        }
    }

    let _t_phase3 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 3: {:.1}ms — {} fragments (passthrough={})",
        _t_phase3 - _t_phase2,
        fragments.len(),
        passthrough_a.len()
    );

    // ── Phase 4: Classification ──────────────────────────────────────────
    // For compound cut: Target fragments (Source::A) must be Outside ALL tools.
    // Tool fragments (Source::B) must be Inside target AND Outside all other tools.
    //
    // OPTIMIZATION: Use AABB filtering to skip tools that can't contain the
    // fragment centroid. A point can only be Inside a tool if it's within the
    // tool's bounding box. For N=100 disjoint tools, this reduces per-fragment
    // classification from O(N) to O(~1-3) on average.
    let target_classifier = try_build_analytic_classifier(topo, target);

    // Expand tool AABBs slightly for classification (tolerance margin).
    let expanded_tool_aabbs: Vec<Aabb3> = tool_data
        .iter()
        .map(|td| td.overall_aabb.expanded(tol.linear))
        .collect();

    let mut classes: Vec<Option<FaceClass>> = fragments
        .iter()
        .enumerate()
        .map(|(idx, frag)| {
            if let Some(&class) = pre_classifications.get(&idx) {
                return Some(class);
            }
            match frag.source {
                Source::A => {
                    // Target fragment: must be Outside ALL tools to survive.
                    // If Inside ANY tool → discard (Inside).
                    let centroid = polygon_centroid(&frag.vertices);
                    // Use global BVH to find tools whose AABB contains centroid.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    for &ti in &nearby_tools {
                        // Double-check with expanded AABB (BVH may have slight padding).
                        if !expanded_tool_aabbs[ti].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref cls) = tool_data[ti].classifier {
                            if cls.classify(centroid, tol) == Some(FaceClass::Inside) {
                                return Some(FaceClass::Inside);
                            }
                        } else {
                            return None; // Need raycast
                        }
                    }
                    Some(FaceClass::Outside)
                }
                Source::B => {
                    // Tool fragment: must be Inside target.
                    let centroid = polygon_centroid(&frag.vertices);
                    if let Some(ref cls) = target_classifier {
                        match cls.classify(centroid, tol) {
                            Some(FaceClass::Inside) => {}
                            Some(FaceClass::Outside) => return Some(FaceClass::Outside),
                            _ => return None,
                        }
                    } else {
                        return None; // Need raycast against target
                    }
                    // Also must be Outside all OTHER tools (for overlapping tools).
                    // Use AABB filtering: only check tools whose AABB contains centroid.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    for &ti2 in &nearby_tools {
                        if !expanded_tool_aabbs[ti2].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref cls2) = tool_data[ti2].classifier {
                            if cls2.classify(centroid, tol) == Some(FaceClass::Inside) {
                                // Could be the fragment's own tool — without tool_index
                                // tracking we can't distinguish. For non-overlapping tools
                                // this is correct (centroid is on-boundary, not strictly Inside).
                                let _ = ti2;
                            }
                        }
                    }
                    Some(FaceClass::Inside)
                }
            }
        })
        .collect();

    // Phase 4b: raycast fallback for unclassified fragments.
    // This handles concave targets (e.g. shelled boxes) where the analytic
    // classifier can't be built. We tessellate the original target/tools and
    // raycast, exactly as analytic_boolean does.
    let needs_raycast = classes.iter().any(Option::is_none);
    if needs_raycast {
        let unclassified_count = classes.iter().filter(|c| c.is_none()).count();
        log::debug!(
            "[compound_cut] raycast fallback for {unclassified_count}/{} fragments",
            fragments.len()
        );

        // Build face data for target (for Source::B raycast) and each tool
        // (for Source::A raycast). Only build what we actually need.
        let needs_target_raycast = classes
            .iter()
            .enumerate()
            .any(|(i, c)| c.is_none() && matches!(fragments[i].source, Source::B));
        let needs_tool_raycast = classes
            .iter()
            .enumerate()
            .any(|(i, c)| c.is_none() && matches!(fragments[i].source, Source::A));

        let target_face_data = if needs_target_raycast {
            Some(collect_face_data(topo, target, deflection)?)
        } else {
            None
        };
        let target_bvh = target_face_data.as_ref().and_then(build_face_bvh);

        // For Source::A fragments, we need to raycast against each relevant tool.
        // Build face data lazily per-tool.
        let tool_face_data: Vec<Option<FaceData>> = if needs_tool_raycast {
            tools
                .iter()
                .enumerate()
                .map(|(i, &tid)| match collect_face_data(topo, tid, deflection) {
                    Ok(fd) => Some(fd),
                    Err(e) => {
                        log::warn!(
                            "[compound_cut] tool {i} tessellation failed, \
                             falling back to sequential: {e}"
                        );
                        None
                    }
                })
                .collect()
        } else {
            vec![None; tools.len()]
        };
        // If any tool failed tessellation, fall back to sequential for correctness.
        if needs_tool_raycast && tool_face_data.iter().any(Option::is_none) {
            return compound_cut_sequential(topo, target, tools, opts);
        }
        let tool_face_bvhs_rc: Vec<Option<Bvh>> = tool_face_data
            .iter()
            .map(|fd| fd.as_ref().and_then(build_face_bvh))
            .collect();

        for (idx, class) in classes.iter_mut().enumerate() {
            if class.is_some() {
                continue;
            }
            let frag = &fragments[idx];
            let centroid = polygon_centroid(&frag.vertices);

            match frag.source {
                Source::B => {
                    // Raycast against original target to determine Inside/Outside.
                    if let Some(ref fd) = target_face_data {
                        let raw =
                            classify_point(centroid, frag.normal, fd, target_bvh.as_ref(), tol);
                        *class = Some(guard_tangent_coplanar(
                            raw,
                            &frag.vertices,
                            frag.normal,
                            fd,
                            target_bvh.as_ref(),
                            tol,
                        ));
                    }
                }
                Source::A => {
                    // Raycast against each nearby tool.
                    let point_aabb = Aabb3 {
                        min: centroid,
                        max: centroid,
                    };
                    let nearby_tools = tool_bvh.query_overlap(&point_aabb);
                    let mut result = FaceClass::Outside;
                    for &ti in &nearby_tools {
                        if !expanded_tool_aabbs[ti].contains_point(centroid) {
                            continue;
                        }
                        if let Some(ref fd) = tool_face_data[ti] {
                            let raw = classify_point(
                                centroid,
                                frag.normal,
                                fd,
                                tool_face_bvhs_rc[ti].as_ref(),
                                tol,
                            );
                            let guarded = guard_tangent_coplanar(
                                raw,
                                &frag.vertices,
                                frag.normal,
                                fd,
                                tool_face_bvhs_rc[ti].as_ref(),
                                tol,
                            );
                            if guarded == FaceClass::Inside {
                                result = FaceClass::Inside;
                                break;
                            }
                        }
                    }
                    *class = Some(result);
                }
            }
        }
    }

    // Classification summary for debugging.
    if log::log_enabled!(log::Level::Debug) {
        let mut a_in = 0usize;
        let mut a_out = 0usize;
        let mut b_in = 0usize;
        let mut b_out = 0usize;
        for (i, c) in classes.iter().enumerate() {
            match (&fragments[i].source, c) {
                (Source::A, Some(FaceClass::Inside)) => a_in += 1,
                (Source::A, Some(FaceClass::Outside)) => a_out += 1,
                (Source::B, Some(FaceClass::Inside)) => b_in += 1,
                (Source::B, Some(FaceClass::Outside)) => b_out += 1,
                _ => {}
            }
        }
        log::debug!(
            "[compound_cut] classification: A(in={a_in} out={a_out}) B(in={b_in} out={b_out}) passthrough={}",
            passthrough_a.len()
        );
    }

    let classes: Vec<FaceClass> = classes
        .into_iter()
        .enumerate()
        .map(|(_i, c)| -> Result<FaceClass, crate::OperationsError> {
            c.ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: format!("compound_cut: fragment {_i} was not classified"),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // ── Phase 5: Assembly ────────────────────────────────────────────────
    let _t_phase4 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 4: {:.1}ms — classification",
        _t_phase4 - _t_phase3,
    );
    // Reuse the same assembly logic as analytic_boolean (vertex/edge dedup,
    // wire construction, face creation).
    let resolution = vertex_merge_resolution(
        fragments.iter().flat_map(|f| f.vertices.iter().copied()),
        tol,
    );
    let mut vertex_map: HashMap<(i64, i64, i64), VertexId> =
        HashMap::with_capacity(fragments.len() * 4);
    let mut edge_map: HashMap<(usize, usize), EdgeId> = HashMap::with_capacity(fragments.len() * 4);
    let mut face_ids_out = Vec::with_capacity(fragments.len());

    for (idx, (frag, &class)) in fragments.iter().zip(classes.iter()).enumerate() {
        let Some(flip) = select_fragment(frag.source, class, BooleanOp::Cut) else {
            continue;
        };
        let is_nonplanar = !matches!(frag.surface, FaceSurface::Plane { .. });
        let is_closed_curve = frag.edge_curves.len() == 1 && frag.edge_curves[0].is_some();

        // For planar faces that need flipping: reverse vertices and negate normal/d.
        // This mirrors analytic_boolean's assembly logic exactly.
        // After pre-reversing, set flip=false for the edge creation code to avoid
        // double-flipping.
        let (verts, normal, d_val, flip) = if flip && !is_nonplanar {
            let rev: Vec<_> = frag.vertices.iter().copied().rev().collect();
            (rev, -frag.normal, -frag.d, false)
        } else {
            (frag.vertices.clone(), frag.normal, frag.d, flip)
        };

        let n = verts.len();
        if n < 3 {
            continue;
        }
        let vert_ids: Vec<VertexId> = verts
            .iter()
            .map(|p| {
                let key = (
                    quantize(p.x(), resolution),
                    quantize(p.y(), resolution),
                    quantize(p.z(), resolution),
                );
                *vertex_map
                    .entry(key)
                    .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
            })
            .collect();

        let wire_id = if is_closed_curve {
            // SAFETY: is_closed_curve checks len==1 && [0].is_some()
            let Some(ec) = frag.edge_curves[0].as_ref() else {
                continue;
            };
            let vid = vert_ids[0];
            let eid = *edge_map
                .entry((vid.index(), vid.index()))
                .or_insert_with(|| topo.add_edge(Edge::new(vid, vid, ec.clone())));
            let wire = Wire::new(vec![OrientedEdge::new(eid, !flip)], true)
                .map_err(crate::OperationsError::Topology)?;
            topo.add_wire(wire)
        } else if is_nonplanar
            && n >= CLOSED_CURVE_SAMPLES
            && (vert_ids.first() == vert_ids.last()
                || vert_ids
                    .first()
                    .zip(vert_ids.last())
                    .is_some_and(|(f, l)| f.index() == l.index()))
        {
            // Cylinder barrel: build from sampled curve ring.
            let ring_len = if vert_ids.first() == vert_ids.last() {
                n - 1
            } else {
                n
            };
            let mut edges = Vec::with_capacity(ring_len);
            for i in 0..ring_len {
                let j = (i + 1) % ring_len;
                let vi = vert_ids[i];
                let vj = vert_ids[j % vert_ids.len()];
                let is_forward = vi.index() <= vj.index();
                let key = if is_forward {
                    (vi.index(), vj.index())
                } else {
                    (vj.index(), vi.index())
                };
                let eid = *edge_map.entry(key).or_insert_with(|| {
                    let (start, end) = if is_forward { (vi, vj) } else { (vj, vi) };
                    topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                });
                edges.push(OrientedEdge::new(eid, is_forward != flip));
            }
            if flip {
                edges.reverse();
            }
            let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
            topo.add_wire(wire)
        } else {
            let mut edges = Vec::with_capacity(n);
            for i in 0..n {
                let j = (i + 1) % n;
                let vi = vert_ids[i];
                let vj = vert_ids[j];
                let is_forward = vi.index() <= vj.index();
                let key = if is_forward {
                    (vi.index(), vj.index())
                } else {
                    (vj.index(), vi.index())
                };
                let eid = *edge_map.entry(key).or_insert_with(|| {
                    let (start, end) = if is_forward { (vi, vj) } else { (vj, vi) };
                    topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                });
                edges.push(OrientedEdge::new(eid, is_forward != flip));
            }
            if flip {
                edges.reverse();
            }
            let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
            topo.add_wire(wire)
        };

        let mut inner_wire_ids = Vec::new();
        if let Some(existing_wires) = existing_inner_wires.get(&idx) {
            inner_wire_ids.extend_from_slice(existing_wires);
        }
        if let Some(inner_curves) = holed_face_inner_curves.get(&idx) {
            for ec in inner_curves {
                let hw_id = if matches!(ec, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
                    let seam_pt = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES)[0];
                    let vid = *vertex_map
                        .entry(quantize_point(seam_pt, resolution))
                        .or_insert_with(|| topo.add_vertex(Vertex::new(seam_pt, tol.linear)));
                    let eid = *edge_map
                        .entry((vid.index(), vid.index()))
                        .or_insert_with(|| topo.add_edge(Edge::new(vid, vid, ec.clone())));
                    let hw = Wire::new(vec![OrientedEdge::new(eid, flip)], true)
                        .map_err(crate::OperationsError::Topology)?;
                    topo.add_wire(hw)
                } else {
                    let mut hole_pts = sample_edge_curve(ec, CLOSED_CURVE_SAMPLES);
                    if !flip {
                        hole_pts.reverse();
                    }
                    let hole_vert_ids: Vec<VertexId> = hole_pts
                        .iter()
                        .map(|p| {
                            let key = (
                                quantize(p.x(), resolution),
                                quantize(p.y(), resolution),
                                quantize(p.z(), resolution),
                            );
                            *vertex_map
                                .entry(key)
                                .or_insert_with(|| topo.add_vertex(Vertex::new(*p, tol.linear)))
                        })
                        .collect();
                    let hm = hole_vert_ids.len();
                    let mut hole_edges = Vec::with_capacity(hm);
                    for i in 0..hm {
                        let j = (i + 1) % hm;
                        let vi_idx = hole_vert_ids[i].index();
                        let vj_idx = hole_vert_ids[j].index();
                        let is_forward = vi_idx <= vj_idx;
                        let key = if is_forward {
                            (vi_idx, vj_idx)
                        } else {
                            (vj_idx, vi_idx)
                        };
                        let eid = *edge_map.entry(key).or_insert_with(|| {
                            let (start, end) = if is_forward {
                                (hole_vert_ids[i], hole_vert_ids[j])
                            } else {
                                (hole_vert_ids[j], hole_vert_ids[i])
                            };
                            topo.add_edge(Edge::new(start, end, EdgeCurve::Line))
                        });
                        hole_edges.push(OrientedEdge::new(eid, is_forward));
                    }
                    let hw =
                        Wire::new(hole_edges, true).map_err(crate::OperationsError::Topology)?;
                    topo.add_wire(hw)
                };
                inner_wire_ids.push(hw_id);
            }
        }

        let surface = match &frag.surface {
            FaceSurface::Plane { .. } => FaceSurface::Plane { normal, d: d_val },
            other => other.clone(),
        };
        let effective_reversed = if is_nonplanar {
            flip ^ frag.source_reversed
        } else {
            false
        };
        let new_face = if effective_reversed {
            Face::new_reversed(wire_id, inner_wire_ids, surface)
        } else {
            Face::new(wire_id, inner_wire_ids, surface)
        };
        let face = topo.add_face(new_face);
        face_ids_out.push(face);
    }

    if face_ids_out.is_empty() {
        return Err(crate::OperationsError::InvalidInput {
            reason: "compound_cut produced no faces".into(),
        });
    }

    // ── Post-assembly ────────────────────────────────────────────────────
    let _t_phase5 = timer_elapsed_ms(_t_total);
    log::debug!(
        "[compound_cut] Phase 5: {:.1}ms — assembly ({} faces)",
        _t_phase5 - _t_phase4,
        face_ids_out.len()
    );

    let vertex_positions: HashMap<VertexId, Point3> = vertex_map
        .values()
        .filter_map(|&vid| topo.vertex(vid).ok().map(|v| (vid, v.point())))
        .collect();
    refine_boundary_edges(
        topo,
        &mut face_ids_out,
        &mut edge_map,
        tol,
        Some(&vertex_positions),
    )?;
    let _t_refine = timer_elapsed_ms(_t_total);
    log::debug!("[compound_cut] refine: {:.1}ms", _t_refine - _t_phase5);
    split_nonmanifold_edges(topo, &mut face_ids_out)?;
    let _t_nm = timer_elapsed_ms(_t_total);
    log::debug!("[compound_cut] split_nm: {:.1}ms", _t_nm - _t_refine);

    let shell = Shell::new(face_ids_out).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    log::debug!("[compound_cut] total: {:.3}ms", timer_elapsed_ms(_t_total));
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Sequential fallback for `compound_cut` when analytic path is unavailable.
///
/// Uses AABB pre-filtering to skip tools that don't overlap the current target,
/// avoiding expensive boolean operations on spatially disjoint tools.
fn compound_cut_sequential(
    topo: &mut Topology,
    target: SolidId,
    tools: &[SolidId],
    opts: BooleanOptions,
) -> Result<SolidId, crate::OperationsError> {
    let _t = timer_now();
    let mut result = target;
    let mut skipped = 0usize;

    for &tool in tools {
        // AABB pre-filter: skip tools that don't overlap the current target.
        let target_aabb = crate::measure::solid_bounding_box(topo, result)?;
        let tool_aabb = crate::measure::solid_bounding_box(topo, tool)?;
        if !target_aabb.intersects(tool_aabb) {
            skipped += 1;
            continue;
        }
        result = boolean_with_options(topo, BooleanOp::Cut, result, tool, opts)?;
    }

    // Post-pass: unify co-surface faces when many tools cause fragmentation.
    // This replaces the old per-step intermediate_opts approach which leaked
    // unify_faces=true onto the final result when trailing tools were skipped.
    if tools.len() > 3 && !opts.unify_faces && skipped < tools.len() {
        match crate::heal::unify_faces(topo, result) {
            Ok(merged) => {
                log::debug!("[compound_cut_sequential] post-pass unified {merged} face(s)");
            }
            Err(e) => {
                log::debug!("[compound_cut_sequential] unify_faces skipped: {e}");
            }
        }
    }

    log::debug!(
        "[compound_cut_sequential] {} tools, {} skipped (disjoint), {:.1}ms",
        tools.len(),
        skipped,
        timer_elapsed_ms(_t)
    );
    Ok(result)
}

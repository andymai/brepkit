//! BuilderSolid — 4-phase shell assembly.
//!
//! Takes BOP-selected faces and assembles them into manifold shells,
//! classifies shells as Growth/Hole, and nests holes inside growth shells.
//!
//! # Phases
//!
//! 1. **`perform_shapes_to_avoid`** — iterative free-edge removal
//! 2. **`perform_loops`** — connectivity flood-fill into shells
//! 3. **`perform_areas`** — Growth vs Hole classification
//! 4. **Assemble** — build final Solid from shells

use std::collections::{HashMap, HashSet, VecDeque};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::wire::{OrientedEdge, WireId};

use super::FaceProvenance;

use crate::bop::SelectedFace;
use crate::error::AlgoError;

/// Edge key for adjacency: canonical `(min, max)` quantized 3D position pair.
///
/// Using quantized positions instead of vertex indices ensures that edges at
/// the same geometric location are recognized as shared, even when faces from
/// different input solids have separate vertex entities at the same position.
type VPair = (QPos, QPos);

/// Build a solid from BOP-selected faces using the 4-phase algorithm.
///
/// # Errors
///
/// Returns [`AlgoError`] if assembly produces no valid shells or
/// topology lookups fail.
#[allow(clippy::too_many_lines)]
pub fn build_solid(
    topo: &mut Topology,
    selected: &[SelectedFace],
    cap_planes: &[CapPlane],
) -> Result<SolidId, AlgoError> {
    Ok(build_solid_with_origins(topo, selected, cap_planes)?.0)
}

/// Like [`build_solid`], but also returns each result face's provenance:
/// `(result face, Some(input source face) | None for a synthesised face)`.
///
/// `face_origins` is maintained as a vector parallel to `face_ids` —
/// the in-place edge/vertex rebuilds (`&mut [FaceId]`) replace faces by
/// position so the parallel entries stay aligned; only the length-changing
/// steps (sliver-retain, doubled-face removal, cap synthesis) sync it
/// explicitly. This is purely additive: it never changes `face_ids`, so the
/// assembled geometry is identical to [`build_solid`].
pub fn build_solid_with_origins(
    topo: &mut Topology,
    selected: &[SelectedFace],
    cap_planes: &[CapPlane],
) -> Result<(SolidId, FaceProvenance), AlgoError> {
    if selected.is_empty() {
        return Err(AlgoError::AssemblyFailed("no faces selected".into()));
    }
    log::debug!("BuilderSolid: {} faces selected", selected.len());

    // Step 0: Create reversed copies for Cut B-faces
    let mut face_ids: Vec<FaceId> = Vec::with_capacity(selected.len());
    // Provenance parallel to `face_ids`: the input face each entry derives from.
    let mut sources: Vec<Option<FaceId>> = Vec::with_capacity(selected.len());
    for sf in selected {
        if sf.reversed {
            let face = topo.face(sf.face_id)?;
            let surface = face.surface().clone();
            let outer_wire = face.outer_wire();
            let inner_wires = face.inner_wires().to_vec();
            // `SelectedFace::reversed` is a flip request relative to the
            // face's CURRENT orientation. A tool face already stored
            // reversed (e.g. a flattened planar-NURBS extrusion wall)
            // flips back to forward; constructing `new_reversed`
            // unconditionally would no-op for it and leave the face
            // pointing into the result material.
            let flipped_face = if face.is_reversed() {
                Face::new(outer_wire, inner_wires, surface)
            } else {
                Face::new_reversed(outer_wire, inner_wires, surface)
            };
            face_ids.push(topo.add_face(flipped_face));
        } else {
            face_ids.push(sf.face_id);
        }
        sources.push(Some(sf.source_face));
    }

    // Step 0a-pre: Drop degenerate sliver faces — all-Line outer wires with
    // fewer than 3 distinct vertex positions enclose zero area (e.g. a loft
    // band built over a duplicated profile point, giving [e, e-reversed]).
    // Keeping them turns their edges non-manifold in an otherwise valid
    // result. Faces with curved edges are exempt (two half-circles bound a
    // real disc with only 2 vertices).
    retain_aligned(&mut face_ids, &mut sources, |&fid| {
        !is_degenerate_line_sliver(topo, fid)
    });
    if face_ids.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "all faces degenerate slivers".into(),
        ));
    }

    // Step 0a-pre2: Strip zero-length Line edges from wires (duplicated
    // input vertices produce them; their twin lives only on the degenerate
    // slivers removed above, so they would survive as free edges).
    remove_zero_length_edges(topo, &mut face_ids)?;

    // Step 0a-pre3: Weld vertices that are coincident within snap tolerance.
    // Intersection in the pavefiller can place a vertex a few ULPs short of an
    // exact pre-existing vertex (e.g. a coincident-arc tangent point landing at
    // -11.999999 vs the body's -12.0). Such near-duplicates quantize to
    // different cells at MERGE_TOL, so the duplicate-edge merge never unifies
    // the two faces' partitions and the shared boundary stays open. Snapping
    // them to one canonical vertex (and dropping the resulting zero-length
    // slivers) lets the merge below see identical partitions.
    weld_coincident_vertices(topo, &mut face_ids)?;

    // Step 0a: Split Line edges at intermediate collinear vertices.
    // Adjacent faces can partition the same geometric boundary differently
    // (one whole edge vs several sub-edges split at paves); refining every
    // Line edge against the global vertex set gives both sides identical
    // partitions so the merge below can unify them.
    split_edges_at_collinear_vertices(topo, &mut face_ids)?;

    // Step 0a2: The same refinement for curved (Circle/Ellipse) rims. A
    // coincident rounded corner can arrive split at a seam vertex on one
    // operand but whole on the other; splitting each arc at the global vertex
    // set lets the merge below unify the shared rim.
    split_arc_edges_at_collinear_vertices(topo, &mut face_ids)?;

    // Step 0b: Merge duplicate edges across selected faces.
    // Faces from different input solids may have separate edge entities for the
    // same geometric boundary. Merge them by quantized endpoint position so that
    // the BuilderSolid's connectivity flood-fill sees shared edges.
    // This is operation-safe: only operates on BOP-selected faces.
    merge_duplicate_edges(topo, &mut face_ids)?;

    // Step 0b2: Drop doubled faces — two (or more) selected faces whose outer
    // wires reference the identical set of (merged) edge entities. Such faces
    // are geometrically coincident copies bounding zero volume between them
    // (e.g. the baseplate dovetail groove cut: the slanted slab wall and the
    // groove flank each emit the same corner triangle, and after edge-merging
    // both reference the same three edges). Keeping them makes every shared edge
    // incident to 3+ faces (non-manifold). Removing the whole group is sound:
    // coincident faces with one identical boundary cancel.
    remove_doubled_faces(topo, &mut face_ids, &mut sources);

    // Step 0b3: Excise out-and-back spurs — the same edge traversed forward
    // then immediately backward is a zero-width excursion contributing two
    // uses of one edge to a single face (a coincident-ring re-trace woven
    // through a wall's wire at a concave corner). Removing the pair never
    // changes the enclosed region. A face whose outer wire collapses below
    // 3 edges was ONLY the excursion (a slit) and is dropped entirely.
    excise_out_and_back_spurs(topo, &mut face_ids, &mut sources);

    if face_ids.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "all faces avoided (all have free edges)".into(),
        ));
    }

    // Step 0c: Synthesise the floor/ceiling cap of a partial coplanar
    // same-domain overlap (e.g. a body whose rounded corner overhangs a
    // chamfered socket — gridfinity compartmented bin). The BOP selector
    // discarded both faces of such an overlap, leaving a closed planar loop of
    // free edges where the larger face's overhang remainder should be. Cap each
    // such loop with a planar face that reuses the existing edges.
    cap_partial_overlap_free_loops(topo, &mut face_ids, &mut sources, cap_planes)?;

    // Snapshot provenance now that face_ids / sources are final and parallel:
    // map each result-bound face to the input face it derives from (None for a
    // synthesised cap). Queried after assembly against the actual result faces.
    let face_source: std::collections::HashMap<FaceId, Option<FaceId>> = face_ids
        .iter()
        .copied()
        .zip(sources.iter().copied())
        .collect();

    // Phase 2: Build shells via connectivity flood-fill
    let shells = perform_loops(topo, &face_ids)?;

    if shells.is_empty() {
        return Err(AlgoError::AssemblyFailed("no shells formed".into()));
    }

    // Phase 3: Classify Growth vs Hole
    let (growth, holes) = perform_areas(topo, &shells);

    if growth.is_empty() {
        return Err(AlgoError::AssemblyFailed(
            "no outer shell found (all shells classified as holes)".into(),
        ));
    }

    // Phase 4: Assemble
    let solid_id = assemble(topo, growth, holes)?;
    let origins = brepkit_topology::explorer::solid_faces(topo, solid_id)?
        .into_iter()
        .map(|f| (f, face_source.get(&f).copied().flatten()))
        .collect();
    Ok((solid_id, origins))
}

/// Retain entries of `items` for which `keep` is true, dropping the same
/// positions from the parallel `aligned` vector. The predicate runs exactly
/// once per element (a precomputed mask), so side effects are not duplicated.
fn retain_aligned<T, U>(
    items: &mut Vec<T>,
    aligned: &mut Vec<U>,
    mut keep: impl FnMut(&T) -> bool,
) {
    debug_assert_eq!(items.len(), aligned.len());
    let mask: Vec<bool> = items.iter().map(&mut keep).collect();
    let mut idx = 0;
    items.retain(|_| {
        let k = mask.get(idx).copied().unwrap_or(true);
        idx += 1;
        k
    });
    let mut idx = 0;
    aligned.retain(|_| {
        let k = mask.get(idx).copied().unwrap_or(true);
        idx += 1;
        k
    });
}

// ── Phase 1 ──────────────────────────────────────────────────────────

/// Iteratively remove faces with free (single-face) edges.
///
/// Only removes a face when ALL its edges are free (shared by ≤1 face).
/// This avoids stripping valid faces from multi-region boolean results.
#[allow(dead_code)] // Disabled pending full edge-identity sharing via CommonBlocks
fn perform_shapes_to_avoid(
    topo: &Topology,
    faces: &mut Vec<FaceId>,
) -> Result<Vec<FaceId>, AlgoError> {
    let mut avoided = Vec::new();

    loop {
        let edge_map = build_edge_face_map(topo, faces)?;
        let mut to_remove: HashSet<FaceId> = HashSet::new();

        // Only remove faces where EVERY edge is free (≤1 face). Removing a
        // face with *any* free edge would strip valid multi-region faces.
        for &fid in faces.iter() {
            let face_keys = face_edge_keys(topo, fid)?;
            if face_keys.is_empty() {
                continue;
            }
            let all_free = face_keys.iter().all(|key| {
                edge_map
                    .get(key)
                    .is_none_or(|faces_for_edge| faces_for_edge.len() <= 1)
            });
            if all_free {
                to_remove.insert(fid);
            }
        }

        if to_remove.is_empty() {
            break;
        }

        avoided.extend(to_remove.iter());
        faces.retain(|f| !to_remove.contains(f));
    }

    if !avoided.is_empty() {
        log::debug!(
            "BuilderSolid: avoided {} faces with free edges",
            avoided.len()
        );
    }

    Ok(avoided)
}

// ── Phase 2 ──────────────────────────────────────────────────────────

/// Group faces into connected shells via edge connectivity.
///
/// Uses flood-fill with dihedral angle selection at non-manifold edges.
#[allow(clippy::too_many_lines)]
fn perform_loops(topo: &Topology, faces: &[FaceId]) -> Result<Vec<Vec<FaceId>>, AlgoError> {
    let edge_map = build_edge_face_map(topo, faces)?;
    let edge_positions = build_edge_positions(topo, faces)?;

    let mut visited: HashSet<FaceId> = HashSet::new();
    let mut shells: Vec<Vec<FaceId>> = Vec::new();

    // Pre-compute face → edge keys for neighbor lookup
    let face_edges: HashMap<FaceId, Vec<VPair>> = faces
        .iter()
        .filter_map(|&fid| Some((fid, face_edge_keys(topo, fid).ok()?)))
        .collect();

    for &start_face in faces {
        if visited.contains(&start_face) {
            continue;
        }

        let mut shell = Vec::new();
        let mut queue = VecDeque::new();

        // Track edges already filled (2 faces) in this shell
        let mut shell_edge_count: HashMap<VPair, u32> = HashMap::new();

        visited.insert(start_face);
        shell.push(start_face);
        queue.push_back(start_face);

        if let Some(keys) = face_edges.get(&start_face) {
            for key in keys {
                *shell_edge_count.entry(*key).or_default() += 1;
            }
        }

        while let Some(current) = queue.pop_front() {
            let Some(keys) = face_edges.get(&current) else {
                continue;
            };

            for key in keys {
                // Skip edges already manifold in this shell
                if shell_edge_count.get(key).copied().unwrap_or(0) >= 2 {
                    continue;
                }

                let Some(candidates) = edge_map.get(key) else {
                    continue;
                };

                let unvisited: Vec<FaceId> = candidates
                    .iter()
                    .filter(|&&f| f != current && !visited.contains(&f))
                    .copied()
                    .collect();

                if unvisited.is_empty() {
                    continue;
                }

                let selected = if unvisited.len() == 1 {
                    unvisited[0]
                } else if let Some((start, end)) = edge_positions.get(key) {
                    // Non-manifold: dihedral angle selection
                    get_face_off(topo, *start, *end, current, &unvisited).unwrap_or(unvisited[0])
                } else {
                    unvisited[0]
                };

                visited.insert(selected);
                shell.push(selected);
                queue.push_back(selected);

                if let Some(sel_keys) = face_edges.get(&selected) {
                    for k in sel_keys {
                        *shell_edge_count.entry(*k).or_default() += 1;
                    }
                }
            }
        }

        shells.push(shell);
    }

    log::debug!(
        "BuilderSolid: {} shells (sizes: {:?})",
        shells.len(),
        shells.iter().map(Vec::len).collect::<Vec<_>>()
    );

    Ok(shells)
}

/// Dihedral angle selection at a non-manifold edge.
///
/// At an edge shared by 3+ faces, selects the face with the smallest
/// positive dihedral angle relative to the current face. This implements
/// clockwise face traversal around the edge.
pub fn get_face_off(
    topo: &Topology,
    edge_start: Point3,
    edge_end: Point3,
    current_face: FaceId,
    candidates: &[FaceId],
) -> Option<FaceId> {
    let edge_dir = edge_end - edge_start;
    let edge_len = edge_dir.length();
    if edge_len < 1e-12 {
        return candidates.first().copied();
    }
    let t = edge_dir * (1.0 / edge_len);

    let mid = Point3::new(
        (edge_start.x() + edge_end.x()) * 0.5,
        (edge_start.y() + edge_end.y()) * 0.5,
        (edge_start.z() + edge_end.z()) * 0.5,
    );

    // Compute bi-normal for current face: b = t × n (outward from face)
    let n_current = face_normal_at(topo, current_face, mid)?;
    let b_current = t.cross(n_current);
    let b_current_len = b_current.length();
    if b_current_len < 1e-12 {
        return candidates.first().copied();
    }
    let b_current = b_current * (1.0 / b_current_len);

    // Reference direction: the edge tangent itself. The dihedral angle is
    // measured around the edge, so the signed angle reference must be along t.
    // (n × b ≈ t for planar faces, but diverges for curved surfaces.)
    let d_ref = t;

    let mut best_face = None;
    let mut best_angle = f64::MAX;

    for &cand in candidates {
        let Some(n_cand) = face_normal_at(topo, cand, mid) else {
            continue;
        };
        let b_cand = t.cross(n_cand);
        let b_cand_len = b_cand.length();
        if b_cand_len < 1e-12 {
            continue;
        }
        let b_cand = b_cand * (1.0 / b_cand_len);

        // Signed angle from b_current to b_cand using d_ref as reference
        let mut angle = angle_with_ref(b_current, b_cand, d_ref);

        // Coplanar same-direction: small angle → natural neighbor (keep as-is)
        // Coplanar opposite-direction: angle ≈ π (keep as-is)
        // Only adjust truly zero angles (identical faces — shouldn't happen
        // since candidates exclude current_face)
        if angle.abs() < 1e-10 {
            angle = std::f64::consts::TAU; // deprioritize identical geometry
        }

        if angle < 0.0 {
            angle += std::f64::consts::TAU;
        }

        if angle < best_angle {
            best_angle = angle;
            best_face = Some(cand);
        }
    }

    best_face
}

/// Signed angle between two direction vectors using a reference axis.
///
/// Returns the angle from `d1` to `d2` measured around `d_ref`.
fn angle_with_ref(d1: Vec3, d2: Vec3, d_ref: Vec3) -> f64 {
    let cross = d1.cross(d2);
    let sin_val = cross.length();
    let cos_val = d1.dot(d2);

    let mut angle = sin_val.atan2(cos_val);

    if cross.dot(d_ref) < 0.0 {
        angle = -angle;
    }

    angle
}

/// Get face normal at a given 3D point (projects point to surface).
fn face_normal_at(topo: &Topology, face_id: FaceId, point: Point3) -> Option<Vec3> {
    let face = topo.face(face_id).ok()?;
    let surface = face.surface();

    if let FaceSurface::Plane { normal, .. } = surface {
        let n = if face.is_reversed() {
            -*normal
        } else {
            *normal
        };
        Some(n)
    } else {
        let (u, v) = surface.project_point(point)?;
        let mut n = surface.normal(u, v);
        if face.is_reversed() {
            n = -n;
        }
        Some(n)
    }
}

// ── Phase 3 ──────────────────────────────────────────────────────────

/// Robust outward-orientation test for a closed shell, independent of face
/// curvature and wire winding.
///
/// `signed_volume_of_shell` signs each face by its surface normal but integrates
/// the magnitude over the outer-wire CORNER vertices only — a fan that
/// under-samples (and can sign-flip) a doubly-curved band whose corners barely
/// bound the wrapped surface (the torus−box notch band reads negative despite
/// being outward). This computes the divergence-theorem flux `∮ P · n dA` per
/// face with a curvature-aware quadrature — a planar face from its corner fan
/// (exact), a curved face from a `(u, v)` GRID over its boundary's parameter box
/// with the local area element `|∂P/∂u × ∂P/∂v|` — so a wrapped band integrates
/// correctly. A closed outward shell yields a positive total (≈ 3·Volume); a
/// genuinely inward-oriented lone shell (a Cut leaving only a cavity component)
/// yields negative and is still rejected. Returns `None` when no face yields a
/// usable contribution so the caller falls back to the volume sign.
fn shell_is_outward_oriented(topo: &Topology, faces: &[FaceId]) -> Option<bool> {
    let mut flux = 0.0_f64;
    let mut any = false;
    for &fid in faces {
        let Ok(face) = topo.face(fid) else { continue };
        let surface = face.surface();
        if let FaceSurface::Plane { .. } = surface {
            // Planar: corner fan from sampled boundary points is exact.
            let Ok(wire) = topo.wire(face.outer_wire()) else {
                continue;
            };
            let mut pts: Vec<Point3> = Vec::new();
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (sp, ep) = (sv.point(), ev.point());
                for k in 0..4 {
                    let f = f64::from(k) / 4.0;
                    let f = if oe.is_forward() { f } else { 1.0 - f };
                    pts.push(edge.curve().evaluate_with_endpoints(f, sp, ep));
                }
            }
            if pts.len() < 3 {
                continue;
            }
            let centroid = {
                let n = pts.len() as f64;
                let mut c = Vec3::new(0.0, 0.0, 0.0);
                for p in &pts {
                    c += Vec3::new(p.x(), p.y(), p.z());
                }
                Point3::new(c.x() / n, c.y() / n, c.z() / n)
            };
            let Some(normal) = face_normal_at(topo, fid, centroid) else {
                continue;
            };
            let area = newell_normal(&pts).length() * 0.5;
            flux += area * Vec3::new(centroid.x(), centroid.y(), centroid.z()).dot(normal);
            any = true;
        } else {
            // Curved: integrate over the boundary's (u, v) parameter box.
            let Ok(wire) = topo.wire(face.outer_wire()) else {
                continue;
            };
            let mut uvs: Vec<(f64, f64)> = Vec::new();
            for oe in wire.edges() {
                let Ok(edge) = topo.edge(oe.edge()) else {
                    continue;
                };
                let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end())) else {
                    continue;
                };
                let (sp, ep) = (sv.point(), ev.point());
                for k in 0..=8 {
                    let f = f64::from(k) / 8.0;
                    let p = edge.curve().evaluate_with_endpoints(f, sp, ep);
                    if let Some((u, v)) = surface.project_point(p) {
                        uvs.push((u, v));
                    }
                }
            }
            if uvs.len() < 3 {
                continue;
            }
            let (u_lo, u_hi, v_lo, v_hi) = uvs.iter().fold(
                (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
                |(ul, uh, vl, vh), &(u, v)| (ul.min(u), uh.max(u), vl.min(v), vh.max(v)),
            );
            let reversed = face.is_reversed();
            let (n_u, n_v) = (24usize, 24usize);
            let du = (u_hi - u_lo) / n_u as f64;
            let dv = (v_hi - v_lo) / n_v as f64;
            if du.abs() < 1e-12 || dv.abs() < 1e-12 {
                continue;
            }
            let eps_u = du * 1e-3;
            let eps_v = dv * 1e-3;
            for iu in 0..n_u {
                for iv in 0..n_v {
                    let u = u_lo + (iu as f64 + 0.5) * du;
                    let v = v_lo + (iv as f64 + 0.5) * dv;
                    let (Some(p), Some(pu1), Some(pu0), Some(pv1), Some(pv0)) = (
                        surface.evaluate(u, v),
                        surface.evaluate(u + eps_u, v),
                        surface.evaluate(u - eps_u, v),
                        surface.evaluate(u, v + eps_v),
                        surface.evaluate(u, v - eps_v),
                    ) else {
                        continue;
                    };
                    let dp_du = (pu1 - pu0) * (1.0 / (2.0 * eps_u));
                    let dp_dv = (pv1 - pv0) * (1.0 / (2.0 * eps_v));
                    let cross = dp_du.cross(dp_dv);
                    let da = cross.length() * du.abs() * dv.abs();
                    if da < 1e-20 {
                        continue;
                    }
                    let mut n = surface.normal(u, v);
                    if reversed {
                        n = -n;
                    }
                    flux += da * Vec3::new(p.x(), p.y(), p.z()).dot(n);
                    any = true;
                }
            }
        }
    }
    if !any || flux.abs() < 1e-9 {
        return None;
    }
    Some(flux > 0.0)
}

/// Classify shells as Growth (outer) or Hole (inner).
///
/// Uses signed volume: positive → outward normals (growth),
/// negative → inward normals (hole).
fn perform_areas(topo: &Topology, shells: &[Vec<FaceId>]) -> (Vec<Vec<FaceId>>, Vec<Vec<FaceId>>) {
    let mut growth = Vec::new();
    let mut holes = Vec::new();

    for shell in shells {
        if shell.is_empty() {
            continue;
        }

        let signed_vol = signed_volume_of_shell(topo, shell);

        // `signed_volume_of_shell` integrates over the outer-wire CORNER vertices
        // only, so it can sign-flip a doubly-curved band whose corners barely
        // bound the wrapped surface (the torus−box notch band reads negative
        // despite being outward). For a LONE shell — the result's outer boundary,
        // with no enclosing shell to be a cavity of — disambiguate with the
        // curvature-robust `shell_is_outward_oriented` (surface-normal divergence
        // flux): keep it as growth only when genuinely outward, so a Cut that
        // leaves only an INWARD cavity component is still rejected. Multi-shell
        // results keep the volume-sign split (a Cut can leave the tool's interior
        // as a separate negative-volume cavity shell).
        let is_growth = if signed_vol >= 0.0 {
            // Positive corner-fan volume already reads outward — keep the
            // historical behaviour for every solid that integrates cleanly
            // (planar, and curved shells whose constant-v boundaries the fan
            // captures, e.g. the sphere − through-cylinder band). The robust
            // test is consulted ONLY below, never overriding a positive volume.
            true
        } else if shells.len() == 1 {
            // A LONE shell read NEGATIVE: either it is genuinely inward (a Cut
            // leaving only a cavity component — must be rejected) or its
            // corner-fan volume sign-flipped on a doubly-curved band whose
            // corners barely bound the wrapped surface (the torus−box notch
            // band). Disambiguate with the curvature-robust surface-normal flux;
            // fall back to the (negative) volume sign if it is inconclusive.
            shell_is_outward_oriented(topo, shell).unwrap_or(false)
        } else {
            // Multi-shell: a negative shell is the tool's interior cavity (hole).
            false
        };
        if is_growth {
            growth.push(shell.clone());
        } else {
            holes.push(shell.clone());
        }
    }

    log::debug!(
        "BuilderSolid: {} growth shells, {} hole shells",
        growth.len(),
        holes.len()
    );

    (growth, holes)
}

/// Whether a shell is closed: every quantized boundary edge is shared by an
/// even number of the shell's own faces (a watertight, manifold lump).
fn shell_is_closed(topo: &Topology, faces: &[FaceId]) -> bool {
    let mut edge_counts: HashMap<VPair, u32> = HashMap::new();
    for &fid in faces {
        let Ok(keys) = face_edge_keys(topo, fid) else {
            return false;
        };
        for key in keys {
            *edge_counts.entry(key).or_default() += 1;
        }
    }
    !edge_counts.is_empty() && edge_counts.values().all(|&c| c % 2 == 0)
}

/// Newell's method normal for a polygon (unnormalized; magnitude = 2·area).
/// Robust to non-planar / non-convex loops.
fn newell_normal(verts: &[Point3]) -> Vec3 {
    let n = verts.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let a = verts[i];
        let b = verts[(i + 1) % n];
        nx += (a.y() - b.y()) * (a.z() + b.z());
        ny += (a.z() - b.z()) * (a.x() + b.x());
        nz += (a.x() - b.x()) * (a.y() + b.y());
    }
    Vec3::new(nx, ny, nz)
}

/// Compute a signed volume estimate for a shell using the divergence theorem.
///
/// Positive = outward-oriented normals (growth shell).
/// Negative = inward-oriented normals (hole shell).
///
/// Each face's fan-triangulation contribution is oriented by the face's actual
/// geometric surface normal (which already accounts for `is_reversed`), not by
/// the raw outer-wire winding. The two agree for solids built with a
/// CCW-against-the-outward-normal convention (e.g. `make_box`), but diverge for
/// equally valid solids whose wires were wound the other way (e.g. a profile
/// extruded *opposite* its face normal). Trusting the wire winding alone made
/// such a solid read as negative volume, so every shell of a fuse that
/// consumed it got misclassified as a hole and assembly failed. Anchoring the
/// sign to the surface normal makes the classifier construction-independent.
fn signed_volume_of_shell(topo: &Topology, faces: &[FaceId]) -> f64 {
    let mut volume = 0.0;

    for &fid in faces {
        let Ok(face) = topo.face(fid) else { continue };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            continue;
        };

        let mut verts = Vec::new();
        for oe in wire.edges() {
            let Ok(edge) = topo.edge(oe.edge()) else {
                continue;
            };
            let vid = oe.oriented_start(edge);
            if let Ok(v) = topo.vertex(vid) {
                verts.push(v.point());
            }
        }

        if verts.len() < 3 {
            continue;
        }

        // Sign the contribution by the face's outward geometric normal rather
        // than the wire winding. Use the wire's centroid as the projection
        // point so curved-surface normals are evaluated near the face interior.
        let centroid = {
            let mut c = Vec3::new(0.0, 0.0, 0.0);
            for v in &verts {
                c = Vec3::new(c.x() + v.x(), c.y() + v.y(), c.z() + v.z());
            }
            let inv = 1.0 / verts.len() as f64;
            Point3::new(c.x() * inv, c.y() * inv, c.z() * inv)
        };
        let wound_normal = newell_normal(&verts);
        let sign = match face_normal_at(topo, fid, centroid) {
            // Flip when the wire winds opposite the outward normal so the
            // fan tets are consistent with the divergence-theorem convention.
            Some(outward) if wound_normal.dot(outward) < 0.0 => -1.0,
            Some(_) => 1.0,
            // No geometric normal available (degenerate face): fall back to the
            // legacy is_reversed sign.
            None => {
                if face.is_reversed() {
                    -1.0
                } else {
                    1.0
                }
            }
        };
        let v0 = verts[0];
        for i in 1..verts.len() - 1 {
            let v1 = verts[i];
            let v2 = verts[i + 1];
            // Signed volume of tetrahedron with the origin as apex.
            volume += sign
                * (v0.x() * (v1.y() * v2.z() - v2.y() * v1.z())
                    + v1.x() * (v2.y() * v0.z() - v0.y() * v2.z())
                    + v2.x() * (v0.y() * v1.z() - v1.y() * v0.z()));
        }
    }

    volume / 6.0
}

// ── Phase 4 ──────────────────────────────────────────────────────────

/// Quantized traversal-order endpoint positions for each oriented edge.
fn oriented_edge_endpoints(topo: &Topology, oes: &[OrientedEdge]) -> Option<Vec<(QPos, QPos)>> {
    let mut ends = Vec::with_capacity(oes.len());
    for oe in oes {
        let edge = topo.edge(oe.edge()).ok()?;
        let sp = topo.vertex(oe.oriented_start(edge)).ok()?.point();
        let ep = topo.vertex(oe.oriented_end(edge)).ok()?.point();
        ends.push((quantize_point(sp, MERGE_TOL), quantize_point(ep, MERGE_TOL)));
    }
    Some(ends)
}

/// Whether a list of oriented edges forms a closed loop in quantized-position
/// space: every endpoint chains to the next and the last closes back to the
/// first. Used to derive a wire's `closed` flag after normalization rather
/// than asserting it unconditionally.
fn oriented_edges_form_closed_loop(topo: &Topology, oes: &[OrientedEdge]) -> bool {
    let Some(ends) = oriented_edge_endpoints(topo, oes) else {
        return false;
    };
    let n = ends.len();
    if n == 0 {
        return false;
    }
    (0..n).all(|i| ends[i].1 == ends[(i + 1) % n].0)
}

/// Whether any oriented edge (same `EdgeId` and direction) appears more than
/// once in the list. Such a wire cannot be a simple loop: the repeat encloses
/// zero area and marks degenerate hole debris from coplanar section splitting.
fn has_repeated_oriented_edge(oes: &[OrientedEdge]) -> bool {
    let mut seen: HashSet<(usize, bool)> = HashSet::with_capacity(oes.len());
    for oe in oes {
        if !seen.insert((oe.edge().index(), oe.is_forward())) {
            return true;
        }
    }
    false
}

/// Remove cyclically-adjacent (edge, +dir)/(edge, -dir) pairs from every
/// face wire; drop faces whose outer wire collapses below 3 edges and inner
/// wires that collapse entirely.
fn excise_out_and_back_spurs(
    topo: &mut Topology,
    face_ids: &mut Vec<FaceId>,
    sources: &mut Vec<Option<FaceId>>,
) {
    let excise = |oes: &mut Vec<OrientedEdge>| -> bool {
        let mut changed = false;
        loop {
            let n = oes.len();
            if n < 2 {
                return changed;
            }
            let mut removed = false;
            for i in 0..n {
                let j = (i + 1) % n;
                if i != j
                    && oes[i].edge() == oes[j].edge()
                    && oes[i].is_forward() != oes[j].is_forward()
                {
                    let (hi, lo) = if i > j { (i, j) } else { (j, i) };
                    oes.remove(hi);
                    oes.remove(lo);
                    removed = true;
                    changed = true;
                    break;
                }
            }
            if !removed {
                return changed;
            }
        }
    };

    let mut drop: Vec<usize> = Vec::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let Ok(face) = topo.face(fid) else { continue };
        let outer_wid = face.outer_wire();
        let inner_wids: Vec<WireId> = face.inner_wires().to_vec();

        let mut outer = match topo.wire(outer_wid) {
            Ok(w) => w.edges().to_vec(),
            Err(_) => continue,
        };
        if excise(&mut outer) {
            if outer.len() < 3 {
                drop.push(fi);
                continue;
            }
            let closed = oriented_edges_form_closed_loop(topo, &outer);
            if let (Ok(new_wire), Ok(slot)) = (
                brepkit_topology::wire::Wire::new(outer, closed),
                topo.wire_mut(outer_wid),
            ) {
                *slot = new_wire;
            }
        }
        let mut kept_inners: Vec<WireId> = Vec::with_capacity(inner_wids.len());
        let mut inners_changed = false;
        for wid in inner_wids {
            let Ok(inner_wire) = topo.wire(wid) else {
                kept_inners.push(wid);
                continue;
            };
            let mut inner = inner_wire.edges().to_vec();
            if excise(&mut inner) {
                if inner.len() < 3 {
                    // The hole WAS the excursion (a zero-width slit): dropping
                    // it entirely is the only consistent outcome — writing the
                    // shrunken (or original) wire back would leave the spur.
                    inners_changed = true;
                    continue;
                }
                let closed = oriented_edges_form_closed_loop(topo, &inner);
                if let (Ok(new_wire), Ok(slot)) = (
                    brepkit_topology::wire::Wire::new(inner, closed),
                    topo.wire_mut(wid),
                ) {
                    *slot = new_wire;
                }
            }
            kept_inners.push(wid);
        }
        if inners_changed && let Ok(f) = topo.face_mut(fid) {
            *f.inner_wires_mut() = kept_inners;
        }
    }
    for &fi in drop.iter().rev() {
        log::debug!(
            "excise_out_and_back_spurs: dropping slit face {:?}",
            face_ids[fi]
        );
        face_ids.remove(fi);
        sources.remove(fi);
    }
}

/// Iteratively remove edges that cannot belong to any closed loop: in a
/// closed wire every endpoint position has even degree >= 2, so an edge with
/// a degree-1 endpoint is dangling debris (e.g. a stray section edge left in
/// a face wire by coplanar splitting). Returns `true` if any edge was removed.
fn prune_dangling_edges(topo: &Topology, oes: &mut Vec<OrientedEdge>) -> bool {
    let mut changed = false;
    loop {
        let Some(ends) = oriented_edge_endpoints(topo, oes) else {
            return changed;
        };
        let mut degree: HashMap<QPos, usize> = HashMap::new();
        for (s, e) in &ends {
            *degree.entry(*s).or_insert(0) += 1;
            *degree.entry(*e).or_insert(0) += 1;
        }
        let keep: Vec<bool> = ends
            .iter()
            .map(|(s, e)| {
                degree.get(s).copied().unwrap_or(0) >= 2 && degree.get(e).copied().unwrap_or(0) >= 2
            })
            .collect();
        if keep.iter().all(|&k| k) {
            return changed;
        }
        let mut idx = 0;
        oes.retain(|_| {
            let k = keep[idx];
            idx += 1;
            k
        });
        changed = true;
        if oes.is_empty() {
            return changed;
        }
    }
}

/// Reorder oriented edges into sequential traversal order by quantized
/// endpoint position. Wires assembled from section edges can carry a
/// geometrically closed loop whose edge list is permuted (each edge
/// correctly oriented but stored out of chain order); downstream
/// wire-closure validation and polygon walks assume sequential order.
/// Lists that are not a single unambiguous closed chain are left untouched.
/// Returns `true` if the order changed.
fn order_edges_sequential(topo: &Topology, oes: &mut Vec<OrientedEdge>) -> bool {
    let n = oes.len();
    if n < 3 {
        return false;
    }
    let Some(ends) = oriented_edge_endpoints(topo, oes) else {
        return false;
    };
    if (0..n).all(|i| ends[i].1 == ends[(i + 1) % n].0) {
        return false;
    }
    let mut by_start: HashMap<QPos, usize> = HashMap::with_capacity(n);
    for (i, (s, _)) in ends.iter().enumerate() {
        if by_start.insert(*s, i).is_some() {
            return false;
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut used = vec![false; n];
    let mut cur = 0usize;
    loop {
        order.push(cur);
        used[cur] = true;
        if order.len() == n {
            break;
        }
        let Some(&j) = by_start.get(&ends[cur].1) else {
            return false;
        };
        if used[j] {
            return false;
        }
        cur = j;
    }
    if ends[cur].1 != ends[order[0]].0 {
        return false;
    }
    *oes = order.iter().map(|&i| oes[i]).collect();
    true
}

/// Normalize a face's wires before final assembly: prune dangling debris
/// edges, drop inner wires that cannot form a loop, and restore sequential
/// edge order. The outer wire is never emptied — if pruning would remove
/// all of its edges the face is left untouched.
fn normalize_face_wires(topo: &mut Topology, fid: FaceId) {
    let Ok(face) = topo.face(fid) else { return };
    let outer_wid = face.outer_wire();
    let inner_wids: Vec<WireId> = face.inner_wires().to_vec();

    let load = |topo: &Topology, wid: WireId| -> Option<Vec<OrientedEdge>> {
        topo.wire(wid).ok().map(|w| w.edges().to_vec())
    };

    let Some(mut outer_oes) = load(topo, outer_wid) else {
        return;
    };
    let orig_outer = outer_oes.clone();
    let pruned = prune_dangling_edges(topo, &mut outer_oes);
    let outer_pruned = if outer_oes.is_empty() {
        outer_oes = orig_outer;
        false
    } else {
        pruned
    };
    let outer_changed = outer_pruned | order_edges_sequential(topo, &mut outer_oes);

    // Normalize each inner wire. A wire whose edges all prune away is dropped
    // (it could never form a loop). A wire that fails to load is kept as-is
    // rather than silently discarded, so a transient lookup error never
    // deletes hole geometry. Surviving wires reuse their original WireId by
    // overwriting in place, which avoids orphaning entries in the append-only
    // arena.
    //
    // An inner wire that lists the same oriented edge more than once is
    // degenerate hole debris (e.g. coplanar band-splitting can emit a single
    // section edge twice in the same direction, enclosing zero area). It
    // carries no real hole, so it is dropped. This is deliberately narrow:
    // valid hole wires never repeat an oriented edge, so genuine holes — even
    // ones whose edge order is permuted — are preserved. Outer wires are never
    // dropped this way; a malformed outer wire must survive to the acceptance
    // gate, which can fall the whole result back to mesh.
    let mut inners_changed = false;
    let mut kept_inner_wids: Vec<WireId> = Vec::with_capacity(inner_wids.len());
    let mut normalized_inners: Vec<(WireId, Vec<OrientedEdge>)> = Vec::new();
    for wid in &inner_wids {
        let Some(mut oes) = load(topo, *wid) else {
            kept_inner_wids.push(*wid);
            continue;
        };
        if has_repeated_oriented_edge(&oes) {
            inners_changed = true;
            continue;
        }
        let changed = prune_dangling_edges(topo, &mut oes);
        if oes.is_empty() {
            inners_changed = true;
            continue;
        }
        if changed | order_edges_sequential(topo, &mut oes) {
            inners_changed = true;
            normalized_inners.push((*wid, oes));
        }
        kept_inner_wids.push(*wid);
    }

    if !outer_changed && !inners_changed {
        return;
    }

    // Overwrite the outer wire in place (reuse its WireId) so the face's wire
    // references stay valid and no arena entry is orphaned.
    if outer_changed {
        let closed = oriented_edges_form_closed_loop(topo, &outer_oes);
        if let (Ok(new_outer), Ok(slot)) = (
            brepkit_topology::wire::Wire::new(outer_oes, closed),
            topo.wire_mut(outer_wid),
        ) {
            *slot = new_outer;
        }
    }

    for (wid, oes) in normalized_inners {
        let closed = oriented_edges_form_closed_loop(topo, &oes);
        if let (Ok(new_inner), Ok(slot)) = (
            brepkit_topology::wire::Wire::new(oes, closed),
            topo.wire_mut(wid),
        ) {
            *slot = new_inner;
        }
    }

    // Only the inner-wire *list* changes when empties were dropped; the face
    // already points at the (in-place updated) outer and surviving wires.
    if kept_inner_wids.len() != inner_wids.len()
        && let Ok(f) = topo.face_mut(fid)
    {
        *f.inner_wires_mut() = kept_inner_wids;
    }
}

/// Final assembly: build Solid from growth + hole shells.
fn assemble(
    topo: &mut Topology,
    growth_shells: Vec<Vec<FaceId>>,
    hole_shells: Vec<Vec<FaceId>>,
) -> Result<SolidId, AlgoError> {
    let all_faces: Vec<FaceId> = growth_shells
        .iter()
        .chain(hole_shells.iter())
        .flatten()
        .copied()
        .collect();
    for fid in all_faces {
        normalize_face_wires(topo, fid);
    }

    // The outer shell bounds the largest enclosed region. Selecting by face
    // count instead lets a heavily fragmented but small growth shell (e.g. an
    // overlap region split into many tiny faces) win over the shell that
    // actually carries the bulk of the volume, demoting that bulk shell to an
    // inner shell and collapsing the measured volume.
    let outer_idx = growth_shells
        .iter()
        .enumerate()
        .map(|(i, s)| (i, signed_volume_of_shell(topo, s)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap_or(0);

    // Additional growth shells (disjoint outward-oriented regions, e.g. a cut
    // that severs the solid into pieces, or a fuse that adds an interpenetrating
    // lump) join the same outer shell so their positive volume adds correctly —
    // inner shells are reserved for cavities (hole shells), and downstream
    // multi-region handling walks only the outer shell. A non-outer growth shell
    // joins only when it is closed in itself (watertight): a watertight,
    // outward-oriented shell is a genuine solid lump regardless of whether its
    // bounding box overlaps the outer shell's. A residual fragmentation sliver is
    // open (its boundary edges are not all paired), so it fails this test and is
    // dropped rather than polluting the assembled volume.
    // TODO: use a `Compound` for true multi-region results.
    let mut outer_faces = growth_shells[outer_idx].clone();
    for (i, gs) in growth_shells.iter().enumerate() {
        if i == outer_idx {
            continue;
        }
        if shell_is_closed(topo, gs) {
            outer_faces.extend_from_slice(gs);
        } else if gs.len() >= 4 {
            // An OPEN growth shell of real size is not a fragmentation
            // sliver — it is a genuine solid lump whose selection left
            // unpaired junction edges. Silently discarding it deletes its
            // whole volume from a watertight-looking result (the lite
            // fused-foot: 72 faces, ~2700 units^3, invisible to every
            // edge-pairing gate). Fail the analytic assembly instead so the
            // boolean falls back to the volume-correct mesh path. Note the
            // OUTER shell is never subjected to this test, so which lump
            // dodges it historically depended on volume-ordering luck.
            return Err(AlgoError::AssemblyFailed(format!(
                "open growth shell with {} faces would be dropped; aborting analytic assembly",
                gs.len()
            )));
        } else {
            log::debug!(
                "BuilderSolid: dropping open {}-face growth sliver",
                gs.len()
            );
        }
    }
    let outer_shell = Shell::new(outer_faces)
        .map_err(|e| AlgoError::AssemblyFailed(format!("outer shell: {e}")))?;
    let outer_id = topo.add_shell(outer_shell);

    // Genuine hole shells (negative signed volume) become inner shells. Same
    // closed-shell requirement as non-outer growth shells above: a cavity
    // boundary must be watertight in itself. A residual selection fragment
    // (e.g. a lone coincident-wall duplicate face) is open, and keeping it
    // as an "inner shell" over-shares its edges against the outer shell.
    let mut inner_ids = Vec::new();
    for hole in &hole_shells {
        if !shell_is_closed(topo, hole) {
            if hole.len() >= 4 {
                // Same fail-safe as the growth side: a sizeable open "hole"
                // is a mis-signed or mis-selected LUMP, not a cavity sliver —
                // silently discarding it deletes real material or cavity
                // boundary from a watertight-looking result.
                return Err(AlgoError::AssemblyFailed(format!(
                    "open hole shell with {} faces would be dropped; aborting analytic assembly",
                    hole.len()
                )));
            }
            log::debug!(
                "BuilderSolid: dropping open {}-face hole-shell fragment",
                hole.len()
            );
            continue;
        }
        if let Ok(inner_shell) = Shell::new(hole.clone()) {
            inner_ids.push(topo.add_shell(inner_shell));
        }
    }

    let solid = Solid::new(outer_id, inner_ids);
    let solid_id = topo.add_solid(solid);

    log::debug!(
        "BuilderSolid: assembled solid {solid_id:?} with {} faces",
        growth_shells
            .iter()
            .chain(hole_shells.iter())
            .map(Vec::len)
            .sum::<usize>()
    );

    Ok(solid_id)
}

// ── Edge Merging ─────────────────────────────────────────────────────

/// Quantized 3D position key for edge endpoint matching.
type QPos = (i64, i64, i64);

/// Quantized position pair (canonical order: min first). Alias for [`VPair`].
type QPosEdge = VPair;

/// Quantize a 3D point to integer coordinates at tolerance resolution.
fn quantize_point(p: Point3, tol: f64) -> QPos {
    let scale = 1.0 / tol;
    (
        (p.x() * scale).round() as i64,
        (p.y() * scale).round() as i64,
        (p.z() * scale).round() as i64,
    )
}

/// Edge data for duplicate detection.
struct EdgeEntry {
    edge_id: brepkit_topology::edge::EdgeId,
    face_idx: usize,
    qpair: QPosEdge,
}

/// Uniform spatial hash over a set of points, for broad-phase "which points
/// lie near this segment" queries.
///
/// [`split_edges_at_collinear_vertices`] otherwise tests every vertex against
/// every Line edge — O(V·E), which on a body grown by many sequential booleans
/// (a honeycomb wall) dominates the boolean. A point can only be an interior
/// cut of a segment if it lies within `snap` of that segment, so bucketing
/// points by cell and probing only the cells a segment's AABB spans yields the
/// identical candidate set with O(V + Σ cells-per-segment) work.
struct PointGrid {
    inv_cell: f64,
    buckets: HashMap<(i64, i64, i64), Vec<usize>>,
}

impl PointGrid {
    /// Build a grid over `points`, choosing a cell size so the total bucket
    /// count stays ~O(N): the cube root of the AABB volume per point, but never
    /// smaller than `min_cell` (the query band) so a segment never has to walk
    /// an unbounded number of cells.
    fn new(points: &[Point3], min_cell: f64) -> Self {
        let cell = Self::choose_cell(points, min_cell);
        let inv_cell = 1.0 / cell;
        let mut buckets: HashMap<(i64, i64, i64), Vec<usize>> = HashMap::new();
        for (i, p) in points.iter().enumerate() {
            buckets
                .entry(Self::cell_of(*p, inv_cell))
                .or_default()
                .push(i);
        }
        Self { inv_cell, buckets }
    }

    fn choose_cell(points: &[Point3], min_cell: f64) -> f64 {
        let Some(bb) = brepkit_math::aabb::Aabb3::try_from_points(points.iter().copied()) else {
            return min_cell.max(1.0);
        };
        let ext = bb.max - bb.min;
        let (dx, dy, dz) = (ext.x().abs(), ext.y().abs(), ext.z().abs());
        // Largest non-degenerate extent sets the scale; aim for ~N cells along
        // it so the grid is roughly N cells total across the populated region.
        let span = dx.max(dy).max(dz);
        #[allow(clippy::cast_precision_loss)]
        let n = points.len().max(1) as f64;
        let target = span / n.cbrt().max(1.0);
        target.max(min_cell).max(f64::MIN_POSITIVE)
    }

    fn cell_of(p: Point3, inv_cell: f64) -> (i64, i64, i64) {
        (
            (p.x() * inv_cell).floor() as i64,
            (p.y() * inv_cell).floor() as i64,
            (p.z() * inv_cell).floor() as i64,
        )
    }

    /// Indices of points whose cell lies within the segment `[a, b]`'s AABB,
    /// inflated by `band` (so every point within `band` of the segment is
    /// included). Conservative: returns a superset of the truly-near points;
    /// the caller still applies the exact distance test.
    fn segment_candidates(&self, a: Point3, b: Point3, band: f64) -> Vec<usize> {
        let lo = Point3::new(a.x().min(b.x()), a.y().min(b.y()), a.z().min(b.z()));
        let hi = Point3::new(a.x().max(b.x()), a.y().max(b.y()), a.z().max(b.z()));
        self.box_candidates(lo, hi, band)
    }

    /// Indices of points whose cell lies within the AABB `[lo, hi]` inflated by
    /// `band`. The geometric primitive behind [`Self::segment_candidates`]; a
    /// caller with a curved edge passes the edge's own sampled AABB so the
    /// query covers the arc's bulge, not just its chord. Returns a superset of
    /// the truly-near points (exact test still applies downstream).
    fn box_candidates(&self, lo: Point3, hi: Point3, band: f64) -> Vec<usize> {
        let lo = Point3::new(lo.x() - band, lo.y() - band, lo.z() - band);
        let hi = Point3::new(hi.x() + band, hi.y() + band, hi.z() + band);
        let (clo, chi) = (
            Self::cell_of(lo, self.inv_cell),
            Self::cell_of(hi, self.inv_cell),
        );
        // Guard against a pathological cell range (a tiny cell size paired with
        // a long edge): iterating every empty cell would defeat the speedup.
        // Iterating the populated buckets directly is still a superset and
        // bounded by the point count, so correctness is preserved.
        let cells = chi
            .0
            .saturating_sub(clo.0)
            .saturating_add(1)
            .saturating_mul(chi.1.saturating_sub(clo.1).saturating_add(1))
            .saturating_mul(chi.2.saturating_sub(clo.2).saturating_add(1));
        let bucket_budget = i64::try_from(self.buckets.len())
            .unwrap_or(i64::MAX)
            .saturating_mul(4);
        let mut out = Vec::new();
        if cells > bucket_budget {
            for (&(cx, cy, cz), list) in &self.buckets {
                if cx >= clo.0
                    && cx <= chi.0
                    && cy >= clo.1
                    && cy <= chi.1
                    && cz >= clo.2
                    && cz <= chi.2
                {
                    out.extend_from_slice(list);
                }
            }
            return out;
        }
        for cx in clo.0..=chi.0 {
            for cy in clo.1..=chi.1 {
                for cz in clo.2..=chi.2 {
                    if let Some(list) = self.buckets.get(&(cx, cy, cz)) {
                        out.extend_from_slice(list);
                    }
                }
            }
        }
        out
    }
}

/// Rebuild faces whose wires contain zero-length Line edges (quantized
/// start == end), dropping those edges. Closed curved edges (full circles)
/// legitimately have coincident endpoints and are kept.
fn remove_zero_length_edges(topo: &mut Topology, face_ids: &mut [FaceId]) -> Result<(), AlgoError> {
    use brepkit_topology::edge::{EdgeCurve, EdgeId};

    for fid in face_ids.iter_mut() {
        let (surface, is_reversed, outer_oes, inner_oes_list, has_zero) = {
            let face = topo.face(*fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();
            let collect = |wid| -> Result<Vec<(EdgeId, bool, bool)>, AlgoError> {
                let mut out = Vec::new();
                let wire = topo.wire(wid)?;
                for oe in wire.edges() {
                    let edge = topo.edge(oe.edge())?;
                    let zero = matches!(edge.curve(), EdgeCurve::Line) && {
                        let sp = topo.vertex(edge.start())?.point();
                        let ep = topo.vertex(edge.end())?.point();
                        quantize_point(sp, MERGE_TOL) == quantize_point(ep, MERGE_TOL)
                    };
                    out.push((oe.edge(), oe.is_forward(), zero));
                }
                Ok(out)
            };
            let outer_oes = collect(face.outer_wire())?;
            let inner_wids = face.inner_wires().to_vec();
            let mut inner_oes_list = Vec::new();
            for iw in inner_wids {
                inner_oes_list.push(collect(iw)?);
            }
            let has_zero = outer_oes
                .iter()
                .chain(inner_oes_list.iter().flatten())
                .any(|&(_, _, z)| z);
            (surface, is_reversed, outer_oes, inner_oes_list, has_zero)
        };
        if !has_zero {
            continue;
        }

        let strip = |oes: &[(EdgeId, bool, bool)]| -> Vec<OrientedEdge> {
            oes.iter()
                .filter(|&&(_, _, z)| !z)
                .map(|&(eid, fwd, _)| OrientedEdge::new(eid, fwd))
                .collect()
        };
        let new_outer = strip(&outer_oes);
        if !is_rebuildable_loop(topo, &new_outer) {
            continue;
        }
        let Ok(new_outer_wire) = brepkit_topology::wire::Wire::new(new_outer, true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer_wire);
        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            let kept = strip(inner_oes);
            if is_rebuildable_loop(topo, &kept)
                && let Ok(w) = brepkit_topology::wire::Wire::new(kept, true)
            {
                new_inner_ids.push(topo.add_wire(w));
            }
        }
        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        *fid = topo.add_face(new_face);
    }
    Ok(())
}

/// Whether a stripped edge list can form a valid closed wire loop. Two or
/// more edges always qualify. A single edge qualifies only when it is itself
/// closed (e.g. a circular hole is one closed-circle edge) — start vertex
/// equals end vertex, or the curve is inherently closed (Circle/Ellipse).
/// Genuinely degenerate single-Line leftovers are rejected.
fn is_rebuildable_loop(topo: &Topology, oes: &[OrientedEdge]) -> bool {
    use brepkit_topology::edge::EdgeCurve;

    match oes {
        [] => false,
        [single] => {
            let Ok(edge) = topo.edge(single.edge()) else {
                return false;
            };
            edge.is_closed() || matches!(edge.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
        }
        _ => true,
    }
}

/// Whether a face's outer wire is an all-Line loop with fewer than 3
/// distinct vertex positions (zero enclosed area).
fn is_degenerate_line_sliver(topo: &Topology, fid: FaceId) -> bool {
    use brepkit_topology::edge::EdgeCurve;

    let Ok(face) = topo.face(fid) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let mut positions: HashSet<QPos> = HashSet::new();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            return false;
        };
        if !matches!(edge.curve(), EdgeCurve::Line) {
            return false;
        }
        for vid in [edge.start(), edge.end()] {
            let Ok(v) = topo.vertex(vid) else {
                return false;
            };
            positions.insert(quantize_point(v.point(), MERGE_TOL));
            if positions.len() >= 3 {
                return false;
            }
        }
    }
    true
}

/// Weld vertices on the selected faces that are coincident within the snap
/// tolerance onto a single canonical vertex, then rebuild any touched wire.
///
/// Quantization-based merging (`merge_duplicate_edges`) keys on `MERGE_TOL`
/// cells, so two vertices a few ULPs apart but within `snap` (10·`MERGE_TOL`)
/// land in different cells and are never recognized as the same point. This
/// pass clusters by actual distance (a coarse spatial hash bounds the
/// neighbour search) so coincident-but-displaced intersection vertices share
/// one entity. An edge whose once-distinct endpoints weld together is dropped
/// — a zero-length line, or an arc that must not be re-created with
/// `start == end` (the kernel reads that as a full circle); a genuinely closed
/// input arc is preserved. Clustering is deterministic: vertices are processed
/// in `VertexId` order and each non-canonical vertex maps to the lowest-index
/// canonical vertex within `snap`. The pass is O(V log V) and runs on every
/// `build_solid`, but returns early without rebuilding when nothing welds.
fn weld_coincident_vertices(topo: &mut Topology, face_ids: &mut [FaceId]) -> Result<(), AlgoError> {
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::vertex::VertexId;

    let snap = MERGE_TOL * 10.0;

    // Collect distinct vertices (id + position) referenced by the faces.
    let mut seen: HashSet<VertexId> = HashSet::new();
    let mut verts: Vec<(VertexId, Point3)> = Vec::new();
    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                for vid in [edge.start(), edge.end()] {
                    if seen.insert(vid) {
                        verts.push((vid, topo.vertex(vid)?.point()));
                    }
                }
            }
        }
    }
    // Deterministic clustering order.
    verts.sort_by_key(|(vid, _)| vid.index());

    // Coarse spatial hash at snap resolution maps a cell to canonical
    // vertices already chosen there; a candidate only needs to probe its own
    // and the 26 neighbouring cells to find a canonical within `snap`.
    let cell = |p: Point3| -> (i64, i64, i64) {
        let s = 1.0 / snap;
        (
            (p.x() * s).floor() as i64,
            (p.y() * s).floor() as i64,
            (p.z() * s).floor() as i64,
        )
    };
    let mut buckets: HashMap<(i64, i64, i64), Vec<(VertexId, Point3)>> = HashMap::new();
    let mut weld: HashMap<VertexId, VertexId> = HashMap::new();
    for &(vid, p) in &verts {
        let c = cell(p);
        // Pick the lowest-index canonical within `snap` across the 27 cells,
        // not merely the first one probed, so the choice is independent of cell
        // iteration order. Canonicals are added in `VertexId` order, so any
        // match already has a lower index than `vid`.
        let mut canonical: Option<VertexId> = None;
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let nc = (c.0 + dx, c.1 + dy, c.2 + dz);
                    if let Some(list) = buckets.get(&nc) {
                        for &(cid, cp) in list {
                            if (cp - p).length() <= snap
                                && canonical.is_none_or(|b| cid.index() < b.index())
                            {
                                canonical = Some(cid);
                            }
                        }
                    }
                }
            }
        }
        match canonical {
            Some(cid) => {
                weld.insert(vid, cid);
            }
            None => {
                buckets.entry(c).or_default().push((vid, p));
            }
        }
    }

    if weld.is_empty() {
        return Ok(());
    }
    let resolve = |vid: VertexId| -> VertexId { weld.get(&vid).copied().unwrap_or(vid) };

    // Cache rewritten edges so a shared EdgeId is rebuilt once and stays shared.
    let mut edge_remap: HashMap<EdgeId, Option<EdgeId>> = HashMap::new();
    for fid in face_ids.iter_mut() {
        let (surface, is_reversed, outer_oes, inner_oes_list) = {
            let face = topo.face(*fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();
            let collect = |wid| -> Result<Vec<(EdgeId, bool)>, AlgoError> {
                Ok(topo
                    .wire(wid)?
                    .edges()
                    .iter()
                    .map(|oe| (oe.edge(), oe.is_forward()))
                    .collect())
            };
            let outer_oes = collect(face.outer_wire())?;
            let mut inner_oes_list = Vec::new();
            for &iw in face.inner_wires() {
                inner_oes_list.push(collect(iw)?);
            }
            (surface, is_reversed, outer_oes, inner_oes_list)
        };

        let touched = outer_oes
            .iter()
            .chain(inner_oes_list.iter().flatten())
            .any(|(eid, _)| {
                topo.edge(*eid)
                    .is_ok_and(|e| weld.contains_key(&e.start()) || weld.contains_key(&e.end()))
            });
        if !touched {
            continue;
        }

        // Rebuild one edge under welding: returns the (possibly cached) new
        // EdgeId, or None when the edge collapses to a point.
        let mut rebuild_edge =
            |topo: &mut Topology, eid: EdgeId| -> Result<Option<EdgeId>, AlgoError> {
                if let Some(&cached) = edge_remap.get(&eid) {
                    return Ok(cached);
                }
                let edge = topo.edge(eid)?;
                let curve = edge.curve().clone();
                let (ov0, ov1) = (edge.start(), edge.end());
                let nv0 = resolve(ov0);
                let nv1 = resolve(ov1);
                // Drop an edge that welding collapsed to a point: a zero-length
                // line, or a once-distinct arc whose endpoints merged (it must
                // NOT be re-created with start == end, which this kernel reads
                // as a full circle). A genuinely closed input arc (ov0 == ov1,
                // e.g. a full circle) is preserved; a zero-length line is always
                // dropped.
                let collapsed = nv0 == nv1 && (ov0 != ov1 || matches!(curve, EdgeCurve::Line));
                let result = if collapsed {
                    None
                } else if nv0 == ov0 && nv1 == ov1 {
                    Some(eid)
                } else {
                    Some(topo.add_edge(Edge::new(nv0, nv1, curve)))
                };
                edge_remap.insert(eid, result);
                Ok(result)
            };

        let mut rebuild_wire =
            |topo: &mut Topology, oes: &[(EdgeId, bool)]| -> Result<Vec<OrientedEdge>, AlgoError> {
                let mut out = Vec::with_capacity(oes.len());
                for &(eid, fwd) in oes {
                    if let Some(new_eid) = rebuild_edge(topo, eid)? {
                        out.push(OrientedEdge::new(new_eid, fwd));
                    }
                }
                Ok(out)
            };

        let new_outer = rebuild_wire(topo, &outer_oes)?;
        if !is_rebuildable_loop(topo, &new_outer) {
            continue;
        }
        let Ok(new_outer_wire) = brepkit_topology::wire::Wire::new(new_outer, true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer_wire);
        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            let kept = rebuild_wire(topo, inner_oes)?;
            if is_rebuildable_loop(topo, &kept)
                && let Ok(w) = brepkit_topology::wire::Wire::new(kept, true)
            {
                new_inner_ids.push(topo.add_wire(w));
            }
        }
        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        *fid = topo.add_face(new_face);
    }

    Ok(())
}

/// Split Line edges at intermediate collinear vertices from the global
/// vertex set of the selected faces.
///
/// Splitting is driven purely by vertex position, so geometrically
/// coincident edges on different faces always receive identical
/// partitions; the sub-edge entities are created once per `EdgeId`, so
/// faces sharing an edge keep sharing its sub-edges.
#[allow(clippy::too_many_lines)]
fn split_edges_at_collinear_vertices(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
) -> Result<(), AlgoError> {
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::vertex::VertexId;

    let tol = MERGE_TOL;
    let snap = tol * 10.0;

    // Canonical vertex per quantized position, and unique Line edges.
    let mut vert_at: HashMap<QPos, (VertexId, Point3)> = HashMap::new();
    let mut line_edges: Vec<(EdgeId, VertexId, VertexId, Point3, Point3)> = Vec::new();
    let mut seen_edges: HashSet<EdgeId> = HashSet::new();

    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        let wids: Vec<WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let (sv, ev) = (edge.start(), edge.end());
                let is_line = matches!(edge.curve(), EdgeCurve::Line);
                let sp = topo.vertex(sv)?.point();
                let ep = topo.vertex(ev)?.point();
                vert_at.entry(quantize_point(sp, tol)).or_insert((sv, sp));
                vert_at.entry(quantize_point(ep, tol)).or_insert((ev, ep));
                if is_line && seen_edges.insert(oe.edge()) {
                    line_edges.push((oe.edge(), sv, ev, sp, ep));
                }
            }
        }
    }

    // Deterministic order for sub-edge allocation.
    line_edges.sort_by_key(|(eid, ..)| eid.index());

    // Index the candidate vertices and bucket them spatially. The cut test
    // below only accepts a vertex within `snap` of the segment, so probing
    // just the grid cells the segment's AABB spans yields the same candidate
    // set as the former full scan of `vert_at`, but in O(near) per edge.
    let verts: Vec<(VertexId, Point3)> = vert_at.values().copied().collect();
    let positions: Vec<Point3> = verts.iter().map(|&(_, p)| p).collect();
    let grid = PointGrid::new(&positions, snap);

    let mut replacements: HashMap<EdgeId, Vec<OrientedEdge>> = HashMap::new();
    for (eid, sv, ev, sp, ep) in line_edges {
        let dir = ep - sp;
        let len2 = dir.dot(dir);
        if len2 < snap * snap {
            continue;
        }
        let mut cuts: Vec<(f64, VertexId)> = Vec::new();
        for ci in grid.segment_candidates(sp, ep, snap) {
            let (vid, p) = verts[ci];
            if (p - sp).length() < snap || (p - ep).length() < snap {
                continue;
            }
            let t = (p - sp).dot(dir) / len2;
            if !(0.0..=1.0).contains(&t) {
                continue;
            }
            let foot = sp + dir * t;
            if (p - foot).length() > snap {
                continue;
            }
            cuts.push((t, vid));
        }
        if cuts.is_empty() {
            continue;
        }
        // `cuts` is gathered by iterating grid buckets (a HashMap), so the `vid`
        // tiebreak makes this a total order — without it, cuts at equal `t` keep
        // nondeterministic hash order and sub-edge IDs drift.
        cuts.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.index().cmp(&b.1.index()))
        });

        let mut chain: Vec<VertexId> = Vec::with_capacity(cuts.len() + 2);
        chain.push(sv);
        chain.extend(cuts.iter().map(|&(_, vid)| vid));
        chain.push(ev);
        let mut subs = Vec::with_capacity(chain.len() - 1);
        for w in chain.windows(2) {
            let sub_eid = topo.add_edge(Edge::new(w[0], w[1], EdgeCurve::Line));
            subs.push(OrientedEdge::new(sub_eid, true));
        }
        replacements.insert(eid, subs);
    }

    if replacements.is_empty() {
        return Ok(());
    }
    let split_count = replacements.len();

    // Rebuild faces whose wires reference a split edge.
    for fid in face_ids.iter_mut() {
        let (surface, is_reversed, outer_oes, inner_oes_list) = {
            let face = topo.face(*fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();
            let outer_oes: Vec<(EdgeId, bool)> = topo
                .wire(face.outer_wire())?
                .edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect();
            let mut inner_oes_list = Vec::new();
            for &iw in face.inner_wires() {
                inner_oes_list.push(
                    topo.wire(iw)?
                        .edges()
                        .iter()
                        .map(|oe| (oe.edge(), oe.is_forward()))
                        .collect::<Vec<(EdgeId, bool)>>(),
                );
            }
            (surface, is_reversed, outer_oes, inner_oes_list)
        };

        let touched = outer_oes
            .iter()
            .chain(inner_oes_list.iter().flatten())
            .any(|(eid, _)| replacements.contains_key(eid));
        if !touched {
            continue;
        }

        let expand = |oes: &[(EdgeId, bool)]| -> Vec<OrientedEdge> {
            let mut out = Vec::with_capacity(oes.len());
            for &(eid, fwd) in oes {
                if let Some(subs) = replacements.get(&eid) {
                    if fwd {
                        out.extend(subs.iter().copied());
                    } else {
                        out.extend(
                            subs.iter()
                                .rev()
                                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward())),
                        );
                    }
                } else {
                    out.push(OrientedEdge::new(eid, fwd));
                }
            }
            out
        };

        let Ok(new_outer) = brepkit_topology::wire::Wire::new(expand(&outer_oes), true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer);
        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            if let Ok(w) = brepkit_topology::wire::Wire::new(expand(inner_oes), true) {
                new_inner_ids.push(topo.add_wire(w));
            }
        }

        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        *fid = topo.add_face(new_face);
    }

    log::debug!("split_edges_at_collinear_vertices: split {split_count} edges");

    Ok(())
}

/// Split Circle/Ellipse arc edges at interior vertices that lie ON the arc.
///
/// The arc analogue of [`split_edges_at_collinear_vertices`]. Two operands can
/// partition the same coincident curved rim differently: one solid's rounded
/// corner arrives as a single quarter-arc, the other's as two eighth-arcs
/// meeting at a 45° seam vertex (the gridfinity 3×3 stacking-lip fuse, where
/// the body corner is split at the diagonal seam but the lip corner is whole).
/// Refining each arc against the global vertex set so both sides carry the same
/// intermediate vertices lets [`merge_duplicate_edges`] unify the shared rim and
/// closes the otherwise-free corner boundary.
///
/// A child arc reuses its parent's `EdgeCurve::Circle`/`Ellipse` geometry with
/// the new endpoints; the edge's trimmed span is derived from its endpoints
/// (see [`brepkit_topology::edge::EdgeCurve::domain_with_endpoints`]), so no
/// geometry needs re-fitting. Full (closed) circles are skipped — they have no
/// interior to split and re-anchoring them is the section builder's job.
fn split_arc_edges_at_collinear_vertices(
    topo: &mut Topology,
    face_ids: &mut [FaceId],
) -> Result<(), AlgoError> {
    use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
    use brepkit_topology::vertex::VertexId;

    let tol = MERGE_TOL;
    let snap = tol * 10.0;

    // Canonical vertex per quantized position, and unique arc edges.
    let mut vert_at: HashMap<QPos, (VertexId, Point3)> = HashMap::new();
    // (edge, start_v, end_v, start_p, end_p, curve)
    let mut arc_edges: Vec<(EdgeId, VertexId, VertexId, Point3, Point3, EdgeCurve)> = Vec::new();
    let mut seen_edges: HashSet<EdgeId> = HashSet::new();

    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        let wids: Vec<WireId> = std::iter::once(face.outer_wire())
            .chain(face.inner_wires().iter().copied())
            .collect();
        for wid in wids {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let (sv, ev) = (edge.start(), edge.end());
                let sp = topo.vertex(sv)?.point();
                let ep = topo.vertex(ev)?.point();
                vert_at.entry(quantize_point(sp, tol)).or_insert((sv, sp));
                vert_at.entry(quantize_point(ep, tol)).or_insert((ev, ep));
                let is_arc = matches!(edge.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_));
                if is_arc && seen_edges.insert(oe.edge()) {
                    arc_edges.push((oe.edge(), sv, ev, sp, ep, edge.curve().clone()));
                }
            }
        }
    }

    // Deterministic order for sub-edge allocation.
    arc_edges.sort_by_key(|(eid, ..)| eid.index());

    // Spatially index the candidate vertices: a vertex can only split an arc if
    // it lies within `snap` of it, so probing the grid cells the arc's AABB
    // spans yields the same candidate set as scanning all of `vert_at` — but in
    // O(near) per arc rather than O(V·E). The arc's AABB is bounded by its
    // endpoints inflated by its sagitta (it can bulge past the chord by up to
    // the radius), so the band is the radius scale; the grid query is
    // conservative and the exact on-arc test below still runs per candidate.
    let verts: Vec<(VertexId, Point3)> = vert_at.values().copied().collect();
    let positions: Vec<Point3> = verts.iter().map(|&(_, p)| p).collect();
    let grid = PointGrid::new(&positions, snap);

    let mut replacements: HashMap<EdgeId, Vec<OrientedEdge>> = HashMap::new();
    for (eid, sv, ev, sp, ep, curve) in arc_edges {
        // A closed (full) circle/ellipse CAN be refined — at global vertices
        // that lie on it — but only into 3+ sub-arcs. With a single interior
        // cut the two halves share BOTH endpoints, and the endpoint-keyed
        // duplicate merge below would conflate them into one arc. The mate
        // side of a full rim is typically already split at the crossing
        // vertices (e.g. a pad bore poking through inset walls), so matching
        // its partition here is what lets the flood-fill pair the shells.
        let is_closed = (ep - sp).length() < snap;
        // `snap` is a LINEAR tolerance (model units); the span / branch tests
        // below are in the curve's ANGULAR domain (radians). Convert via the
        // curve's radius scale (arc length ≈ radius·angle) so the angular guard
        // is metrically equivalent to `snap`. A degenerate near-zero radius has
        // no meaningful interior to split, so leave the edge whole.
        //
        // Only arcs reach here (the collection loop filters on `is_arc`); the
        // non-arc arms are unreachable but kept explicit per the exhaustive-
        // match convention so a future curve variant can't be silently skipped.
        let radius_scale = match &curve {
            EdgeCurve::Circle(c) => c.radius(),
            EdgeCurve::Ellipse(e) => e.semi_major(),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => continue,
        };
        if radius_scale < snap {
            continue;
        }
        let angular_eps = snap / radius_scale;
        // The arc's CCW angular span [a0, a1] with a1 > a0. For a CLOSED edge
        // the endpoint-derived domain is the curve's intrinsic 0..TAU, but the
        // replacement chain below starts at the edge's seam vertex — anchor
        // the span at the seam's angle instead, or cuts sort in the intrinsic
        // frame and the sub-arcs overlap past one revolution (seam at pi with
        // cuts at pi/2 and 3pi/2 would sweep 4pi total).
        let (a0, a1) = if is_closed {
            let seam = project_angle_on_curve(&curve, sp);
            (seam, seam + std::f64::consts::TAU)
        } else {
            curve.domain_with_endpoints(sp, ep)
        };
        let span = a1 - a0;
        if span < angular_eps {
            continue;
        }

        // The arc's true 3D AABB (it bulges past the chord), sampled along the
        // span, so the spatial query covers every vertex that could lie on it.
        let mut amin = sp;
        let mut amax = sp;
        let arc_samples = ARC_AABB_SAMPLES;
        for k in 0..=arc_samples {
            let f = f64::from(k) / f64::from(arc_samples);
            let a = a0 + (a1 - a0) * f;
            let q = curve.evaluate_with_endpoints(a, sp, ep);
            amin = Point3::new(
                amin.x().min(q.x()),
                amin.y().min(q.y()),
                amin.z().min(q.z()),
            );
            amax = Point3::new(
                amax.x().max(q.x()),
                amax.y().max(q.y()),
                amax.z().max(q.z()),
            );
        }
        // A sampled min/max can under-cover the arc's bulge between samples by up
        // to the sagitta of one angular step; inflate the query band by that bound
        // so the broad phase stays conservative (never prunes a real collinear cut).
        let step = (a1 - a0) / f64::from(arc_samples);
        let sagitta = radius_scale * (1.0 - (step * 0.5).cos());
        let band = snap + sagitta;

        let mut cuts: Vec<(f64, VertexId)> = Vec::new();
        for ci in grid.box_candidates(amin, amax, band) {
            let (vid, p) = verts[ci];
            // Skip the arc's own endpoints.
            if (p - sp).length() < snap || (p - ep).length() < snap {
                continue;
            }
            // The vertex must lie ON the arc: evaluating the curve at the
            // vertex's projected angle must reproduce the vertex position.
            // `evaluate_with_endpoints` takes the angle directly for arcs.
            let a = project_angle_on_curve(&curve, p);
            let on = curve.evaluate_with_endpoints(a, sp, ep);
            if (on - p).length() > snap {
                continue;
            }
            // Bring the angle strictly inside the trimmed span [a0, a1]. The
            // margin is angular (radians), so use `angular_eps`, not the linear
            // `snap`.
            let a_branch = a0 + (a - a0).rem_euclid(std::f64::consts::TAU);
            if !(a0 + angular_eps..=a1 - angular_eps).contains(&a_branch) {
                continue;
            }
            cuts.push((a_branch, vid));
        }
        if cuts.is_empty() || (is_closed && cuts.len() < 2) {
            continue;
        }
        // Total order: angle then vertex index (the `vert_at` HashMap iteration
        // is nondeterministic without the tiebreak).
        cuts.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then_with(|| a.1.index().cmp(&b.1.index()))
        });
        cuts.dedup_by_key(|(_, vid)| *vid);

        let mut chain: Vec<VertexId> = Vec::with_capacity(cuts.len() + 2);
        chain.push(sv);
        chain.extend(cuts.iter().map(|&(_, vid)| vid));
        chain.push(ev);
        let mut subs = Vec::with_capacity(chain.len() - 1);
        for w in chain.windows(2) {
            // Children share the parent arc's geometry; their endpoints define
            // the sub-arc span.
            let sub_eid = topo.add_edge(Edge::new(w[0], w[1], curve.clone()));
            subs.push(OrientedEdge::new(sub_eid, true));
        }
        replacements.insert(eid, subs);
    }

    if replacements.is_empty() {
        return Ok(());
    }
    let split_count = replacements.len();

    // Rebuild faces whose wires reference a split arc.
    for fid in face_ids.iter_mut() {
        let (surface, is_reversed, outer_oes, inner_oes_list) = {
            let face = topo.face(*fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();
            let outer_oes: Vec<(EdgeId, bool)> = topo
                .wire(face.outer_wire())?
                .edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect();
            let mut inner_oes_list = Vec::new();
            for &iw in face.inner_wires() {
                inner_oes_list.push(
                    topo.wire(iw)?
                        .edges()
                        .iter()
                        .map(|oe| (oe.edge(), oe.is_forward()))
                        .collect::<Vec<(EdgeId, bool)>>(),
                );
            }
            (surface, is_reversed, outer_oes, inner_oes_list)
        };

        let touched = outer_oes
            .iter()
            .chain(inner_oes_list.iter().flatten())
            .any(|(eid, _)| replacements.contains_key(eid));
        if !touched {
            continue;
        }

        let expand = |oes: &[(EdgeId, bool)]| -> Vec<OrientedEdge> {
            let mut out = Vec::with_capacity(oes.len());
            for &(eid, fwd) in oes {
                if let Some(subs) = replacements.get(&eid) {
                    if fwd {
                        out.extend(subs.iter().copied());
                    } else {
                        out.extend(
                            subs.iter()
                                .rev()
                                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward())),
                        );
                    }
                } else {
                    out.push(OrientedEdge::new(eid, fwd));
                }
            }
            out
        };

        let Ok(new_outer) = brepkit_topology::wire::Wire::new(expand(&outer_oes), true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer);
        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            if let Ok(w) = brepkit_topology::wire::Wire::new(expand(inner_oes), true) {
                new_inner_ids.push(topo.add_wire(w));
            }
        }

        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        *fid = topo.add_face(new_face);
    }

    log::debug!("split_arc_edges_at_collinear_vertices: split {split_count} arcs");

    Ok(())
}

/// Project a point onto a Circle/Ellipse `EdgeCurve`, returning the angle
/// parameter; returns `0.0` for non-arc curves (never called on them).
fn project_angle_on_curve(curve: &brepkit_topology::edge::EdgeCurve, p: Point3) -> f64 {
    use brepkit_topology::edge::EdgeCurve;
    match curve {
        EdgeCurve::Circle(c) => c.project(p),
        EdgeCurve::Ellipse(e) => e.project(p),
        _ => 0.0,
    }
}

/// Merge duplicate edges across selected faces by quantized endpoint position.
///
/// For each group of edges with the same quantized start/end positions,
/// picks one canonical edge and rebuilds the other faces' wires to reference it.
/// Uses snapshot-then-allocate to satisfy the borrow checker.
#[allow(clippy::too_many_lines)]
fn merge_duplicate_edges(topo: &mut Topology, face_ids: &mut [FaceId]) -> Result<(), AlgoError> {
    use brepkit_topology::edge::EdgeId;

    let tol = MERGE_TOL;

    let mut entries: Vec<EdgeEntry> = Vec::new();

    for (fi, &fid) in face_ids.iter().enumerate() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let qs = quantize_point(sp, tol);
                let qe = quantize_point(ep, tol);
                let qpair = if qs <= qe { (qs, qe) } else { (qe, qs) };
                entries.push(EdgeEntry {
                    edge_id: oe.edge(),
                    face_idx: fi,
                    qpair,
                });
            }
        }
    }

    // Find groups where multiple DIFFERENT EdgeIds share the same qpair.
    let mut groups: HashMap<QPosEdge, Vec<EdgeId>> = HashMap::new();
    for entry in &entries {
        groups.entry(entry.qpair).or_default().push(entry.edge_id);
    }

    // Build edge replacement map: duplicate EdgeId → (canonical EdgeId, needs_flip).
    // needs_flip is true when the duplicate's vertex order is reversed vs canonical,
    // requiring the OrientedEdge's forward flag to be flipped during wire rebuilding.
    let mut replacements: HashMap<EdgeId, (EdgeId, bool)> = HashMap::new();
    for edge_ids in groups.values() {
        // Deduplicate edge IDs (same edge may appear multiple times from different faces)
        let mut unique: Vec<EdgeId> = edge_ids.clone();
        unique.sort_by_key(|e| e.index());
        unique.dedup();

        if unique.len() < 2 {
            continue; // Only one unique edge — no merge needed
        }

        // Pick the first (lowest index) as canonical
        let canonical = unique[0];
        let canon_start = topo.edge(canonical)?.start();
        let canon_end = topo.edge(canonical)?.end();
        let canon_qs = quantize_point(topo.vertex(canon_start)?.point(), tol);
        let canon_qe = quantize_point(topo.vertex(canon_end)?.point(), tol);

        for &dup in &unique[1..] {
            let dup_edge = topo.edge(dup)?;
            let dup_qs = quantize_point(topo.vertex(dup_edge.start())?.point(), tol);
            let dup_qe = quantize_point(topo.vertex(dup_edge.end())?.point(), tol);
            // Detect reversed vertex order. For closed edges (start == end),
            // qs == qe for both canonical and duplicate, so the flip condition
            // would be trivially true. Never flip closed edges.
            let is_closed = canon_qs == canon_qe;
            let needs_flip = !is_closed && dup_qs == canon_qe && dup_qe == canon_qs;
            replacements.insert(dup, (canonical, needs_flip));
        }
    }

    if replacements.is_empty() {
        return Ok(());
    }

    let merge_count = replacements.len();

    // Sort the face indices before iterating so that `topo.add_wire` and
    // `topo.add_face` are called in a deterministic order. Iterating the
    // HashSet directly picks up a random per-process iteration order,
    // which assigns different underlying WireId/FaceId values to
    // structurally identical wires across runs. Downstream flood-fill in
    // `perform_loops` can be sensitive to those ID orderings at
    // near-degenerate geometry, so fix the order here.
    let faces_to_rebuild: HashSet<usize> = entries
        .iter()
        .filter(|e| replacements.contains_key(&e.edge_id))
        .map(|e| e.face_idx)
        .collect();
    let mut faces_to_rebuild_sorted: Vec<usize> = faces_to_rebuild.into_iter().collect();
    faces_to_rebuild_sorted.sort_unstable();

    for &fi in &faces_to_rebuild_sorted {
        let fid = face_ids[fi];

        let (surface, is_reversed, outer_oes, inner_oes_list) = {
            let face = topo.face(fid)?;
            let surface = face.surface().clone();
            let is_reversed = face.is_reversed();

            let outer_wire = topo.wire(face.outer_wire())?;
            let outer_oes: Vec<(EdgeId, bool)> = outer_wire
                .edges()
                .iter()
                .map(|oe| (oe.edge(), oe.is_forward()))
                .collect();

            let inner_wids = face.inner_wires().to_vec();
            let mut inner_oes_list = Vec::new();
            for &iw in &inner_wids {
                let w = topo.wire(iw)?;
                inner_oes_list.push(
                    w.edges()
                        .iter()
                        .map(|oe| (oe.edge(), oe.is_forward()))
                        .collect::<Vec<_>>(),
                );
            }

            (surface, is_reversed, outer_oes, inner_oes_list)
        };

        let new_outer_oes: Vec<_> = outer_oes
            .iter()
            .map(|(eid, fwd)| {
                if let Some(&(new_eid, flip)) = replacements.get(eid) {
                    let new_fwd = if flip { !*fwd } else { *fwd };
                    brepkit_topology::wire::OrientedEdge::new(new_eid, new_fwd)
                } else {
                    brepkit_topology::wire::OrientedEdge::new(*eid, *fwd)
                }
            })
            .collect();
        let Ok(new_outer) = brepkit_topology::wire::Wire::new(new_outer_oes, true) else {
            continue;
        };
        let new_outer_id = topo.add_wire(new_outer);

        let mut new_inner_ids = Vec::new();
        for inner_oes in &inner_oes_list {
            let new_oes: Vec<_> = inner_oes
                .iter()
                .map(|(eid, fwd)| {
                    if let Some(&(new_eid, flip)) = replacements.get(eid) {
                        let new_fwd = if flip { !*fwd } else { *fwd };
                        brepkit_topology::wire::OrientedEdge::new(new_eid, new_fwd)
                    } else {
                        brepkit_topology::wire::OrientedEdge::new(*eid, *fwd)
                    }
                })
                .collect();
            if let Ok(w) = brepkit_topology::wire::Wire::new(new_oes, true) {
                new_inner_ids.push(topo.add_wire(w));
            }
        }

        let mut new_face = Face::new(new_outer_id, new_inner_ids, surface);
        if is_reversed {
            new_face.set_reversed(true);
        }
        face_ids[fi] = topo.add_face(new_face);
    }

    log::debug!(
        "merge_duplicate_edges: merged {merge_count} duplicate edges across {} faces",
        faces_to_rebuild_sorted.len()
    );

    Ok(())
}

/// Remove doubled faces: two or more selected faces whose outer wires reference
/// the identical multiset of edge entities.
///
/// After [`merge_duplicate_edges`] has unified shared edges, two geometrically
/// coincident sub-faces (the same boundary traced twice) reference the exact
/// same edge IDs. Such faces bound zero volume between them and make every one
/// of their shared edges incident to 3+ faces (non-manifold). This arises when
/// the planar-arrangement splitter, fed a foreign (off-plane) section, emits a
/// sliver region that duplicates the true owner face on the adjacent surface —
/// the baseplate dovetail groove cut, where the slanted slab wall and the groove
/// flank each produce the same corner triangle.
///
/// Keying on the merged-edge-ID multiset is exact (no tolerance): only faces
/// that literally share every boundary edge group together, and a coincident
/// pair with one identical boundary always cancels, so dropping the whole group
/// is sound. Inner wires are ignored — a doubled hole boundary is not a
/// manifold defect on its own and removing the holed face would be unsafe.
fn remove_doubled_faces(
    topo: &Topology,
    face_ids: &mut Vec<FaceId>,
    sources: &mut Vec<Option<FaceId>>,
) {
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::wire::OrientedEdge;

    // Key = sorted outer- AND inner-wire edge-ID multiset, so a face only
    // matches a TRULY identical one — a holed face never collides with a
    // coincident spurious non-holed copy (which would otherwise drop it).
    let mut groups: HashMap<Vec<EdgeId>, Vec<usize>> = HashMap::new();
    for (fi, &fid) in face_ids.iter().enumerate() {
        let Ok(face) = topo.face(fid) else { continue };
        let Ok(wire) = topo.wire(face.outer_wire()) else {
            continue;
        };
        let mut key: Vec<EdgeId> = wire.edges().iter().map(OrientedEdge::edge).collect();
        for &iw in face.inner_wires() {
            if let Ok(inner) = topo.wire(iw) {
                key.extend(inner.edges().iter().map(OrientedEdge::edge));
            }
        }
        key.sort_by_key(|e| e.index());
        groups.entry(key).or_default().push(fi);
    }

    let mut drop_idx: HashSet<usize> = HashSet::new();
    for members in groups.values() {
        if members.len() >= 2 {
            for &m in members {
                drop_idx.insert(m);
            }
        }
    }

    if drop_idx.is_empty() {
        return;
    }
    log::debug!(
        "remove_doubled_faces: dropped {} doubled faces",
        drop_idx.len()
    );
    let mut keep = Vec::with_capacity(face_ids.len() - drop_idx.len());
    let mut keep_sources = Vec::with_capacity(keep.capacity());
    for (fi, &fid) in face_ids.iter().enumerate() {
        if !drop_idx.contains(&fi) {
            keep.push(fid);
            // `sources` is parallel to `face_ids`, so `fi` is always in bounds;
            // index directly (panics loudly in debug if the invariant breaks).
            keep_sources.push(sources[fi]);
        }
    }
    *face_ids = keep;
    *sources = keep_sources;
}

// ── Helpers ──────────────────────────────────────────────────────────

/// Build edge→face adjacency map using vertex-pair as key.
fn build_edge_face_map(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, Vec<FaceId>>, AlgoError> {
    let mut map: HashMap<VPair, Vec<FaceId>> = HashMap::new();

    for &fid in faces {
        for key in face_edge_keys(topo, fid)? {
            map.entry(key).or_default().push(fid);
        }
    }

    Ok(map)
}

/// Tolerance for position quantization (matches system linear tolerance).
const MERGE_TOL: f64 = 1e-7;

/// Samples per arc when building its broad-phase AABB for the collinear-split
/// query (its bulge is covered by inflating the query band with the per-step
/// sagitta — see `split_arc_edges_at_collinear_vertices`).
const ARC_AABB_SAMPLES: u32 = 12;

/// Get all edge keys (quantized position-pair) for a face's wires.
fn face_edge_keys(topo: &Topology, fid: FaceId) -> Result<Vec<VPair>, AlgoError> {
    let face = topo.face(fid)?;
    let mut keys = Vec::new();
    for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let sp = topo.vertex(edge.start())?.point();
            let ep = topo.vertex(edge.end())?.point();
            let qs = quantize_point(sp, MERGE_TOL);
            let qe = quantize_point(ep, MERGE_TOL);
            keys.push(if qs <= qe { (qs, qe) } else { (qe, qs) });
        }
    }
    Ok(keys)
}

/// Build edge position-pair → 3D positions map for `get_face_off`.
fn build_edge_positions(
    topo: &Topology,
    faces: &[FaceId],
) -> Result<HashMap<VPair, (Point3, Point3)>, AlgoError> {
    let mut map: HashMap<VPair, (Point3, Point3)> = HashMap::new();

    for &fid in faces {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let qs = quantize_point(sp, MERGE_TOL);
                let qe = quantize_point(ep, MERGE_TOL);
                // Store points in the same canonical order as the key so
                // get_face_off sees a consistent tangent direction.
                let (key, ordered) = if qs <= qe {
                    ((qs, qe), (sp, ep))
                } else {
                    ((qe, qs), (ep, sp))
                };
                if let std::collections::hash_map::Entry::Vacant(entry) = map.entry(key) {
                    entry.insert(ordered);
                }
            }
        }
    }

    Ok(map)
}

/// A candidate cap plane derived from a partial-overlap same-domain pair.
///
/// `normal`/`d` describe the shared plane (`normal · p = d`); `out_normal` is
/// the **effective** outward normal of the larger discarded face, used to
/// orient any synthesised cap face so it contributes outward in the result.
#[derive(Debug, Clone, Copy)]
pub struct CapPlane {
    /// Plane normal (unit).
    pub normal: Vec3,
    /// Plane offset: `normal · p = d` for points on the plane.
    pub d: f64,
    /// Effective outward normal of the larger discarded face.
    pub out_normal: Vec3,
}

/// Synthesise the missing floor/ceiling cap face(s) of a partial coplanar
/// same-domain overlap.
///
/// When two opposing-solid faces share a plane but only *partially* overlap
/// (e.g. a body whose rounded corner overhangs a socket whose corner is
/// chamfered — gridfinity compartmented bin), the BOP selector discards both
/// (their contact is interior to the union) but the larger face's *overhang
/// remainder* is exterior and must remain. Discarding it leaves a closed planar
/// loop of free edges where that remainder face should be, so the shell never
/// closes and the result falls back to mesh.
///
/// This pass finds closed planar loops of free (single-incidence) edges that
/// lie in one of the `cap_planes` and builds a planar face for each, reusing the
/// existing edge entities (so the new face shares them exactly and the loop
/// becomes manifold). It only fires on loops coplanar with a partial-overlap SD
/// plane, so it cannot cap a legitimately-open boundary elsewhere.
fn cap_partial_overlap_free_loops(
    topo: &mut Topology,
    face_ids: &mut Vec<FaceId>,
    sources: &mut Vec<Option<FaceId>>,
    cap_planes: &[CapPlane],
) -> Result<(), AlgoError> {
    use brepkit_topology::edge::EdgeId;
    use brepkit_topology::wire::Wire;

    if cap_planes.is_empty() {
        return Ok(());
    }

    // Collect free edges: those whose quantized vertex-pair key is incident to
    // exactly one selected face.
    let edge_map = build_edge_face_map(topo, face_ids)?;
    let free_keys: HashSet<VPair> = edge_map
        .iter()
        .filter(|(_, faces)| faces.len() == 1)
        .map(|(k, _)| *k)
        .collect();
    if free_keys.is_empty() {
        return Ok(());
    }

    // Gather the actual EdgeIds whose endpoints match a free key (one canonical
    // edge per key — duplicates were merged earlier). Record each edge's
    // endpoints so we can walk loops by quantized position.
    let mut free_edges: Vec<(EdgeId, QPos, QPos)> = Vec::new();
    let mut seen_keys: HashSet<VPair> = HashSet::new();
    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                let qs = quantize_point(sp, MERGE_TOL);
                let qe = quantize_point(ep, MERGE_TOL);
                let key = if qs <= qe { (qs, qe) } else { (qe, qs) };
                if free_keys.contains(&key) && seen_keys.insert(key) {
                    free_edges.push((oe.edge(), qs, qe));
                }
            }
        }
    }

    // Build an undirected adjacency over quantized vertices, vertex -> list of
    // (edge index, other-endpoint). A closed loop walks vertices of degree 2.
    let mut adj: HashMap<QPos, Vec<(usize, QPos)>> = HashMap::new();
    for (i, &(_, qs, qe)) in free_edges.iter().enumerate() {
        adj.entry(qs).or_default().push((i, qe));
        adj.entry(qe).or_default().push((i, qs));
    }
    // Only loops whose every vertex has degree exactly 2 are unambiguous closed
    // cycles. A vertex of higher degree means the free edges branch (a T or a
    // pinch); capping those is ambiguous, so skip such components.
    if adj.values().any(|v| v.len() != 2) {
        return Ok(());
    }

    let mut used_edge: Vec<bool> = vec![false; free_edges.len()];
    let mut pos3d: HashMap<QPos, Point3> = HashMap::new();
    for &fid in face_ids.iter() {
        let face = topo.face(fid)?;
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                let sp = topo.vertex(edge.start())?.point();
                let ep = topo.vertex(edge.end())?.point();
                pos3d.entry(quantize_point(sp, MERGE_TOL)).or_insert(sp);
                pos3d.entry(quantize_point(ep, MERGE_TOL)).or_insert(ep);
            }
        }
    }

    let mut new_faces: Vec<Face> = Vec::new();

    for start in 0..free_edges.len() {
        if used_edge[start] {
            continue;
        }
        // Walk the cycle containing `start`.
        let (e0, a0, b0) = free_edges[start];
        let mut loop_edges: Vec<EdgeId> = vec![e0];
        let mut loop_verts: Vec<QPos> = vec![a0, b0];
        used_edge[start] = true;
        let mut cur = b0;
        let mut ok = true;
        loop {
            if cur == a0 {
                break; // closed back to the loop start
            }
            let Some(neigh) = adj.get(&cur) else {
                ok = false;
                break;
            };
            // Degree is exactly 2; pick the edge that isn't where we came from.
            let next = neigh.iter().find(|&&(ei, _)| !used_edge[ei]).copied();
            let Some((ei, other)) = next else {
                ok = false;
                break;
            };
            used_edge[ei] = true;
            loop_edges.push(free_edges[ei].0);
            loop_verts.push(other);
            cur = other;
            if loop_edges.len() > free_edges.len() {
                ok = false;
                break;
            }
        }
        if !ok || loop_edges.len() < 3 {
            continue;
        }

        // Coplanarity: every loop vertex must lie in one cap plane (within tol).
        let verts3d: Vec<Point3> = loop_verts
            .iter()
            .filter_map(|q| pos3d.get(q).copied())
            .collect();
        if verts3d.len() != loop_verts.len() {
            continue;
        }
        let origin = Point3::new(0.0, 0.0, 0.0);
        let Some(cap) = cap_planes.iter().copied().find(|cp| {
            verts3d
                .iter()
                .all(|p| (cp.normal.dot(*p - origin) - cp.d).abs() <= MERGE_TOL * 10.0)
        }) else {
            continue;
        };

        // Build the outer wire from the existing edges in walk order. Each
        // OrientedEdge's natural direction is recovered by matching the edge's
        // stored start vertex against the walk's incoming vertex.
        let mut oriented: Vec<OrientedEdge> = Vec::with_capacity(loop_edges.len());
        let mut walk_from = loop_verts[0];
        let mut build_ok = true;
        for &eid in &loop_edges {
            let edge = topo.edge(eid)?;
            let es = quantize_point(topo.vertex(edge.start())?.point(), MERGE_TOL);
            let ee = quantize_point(topo.vertex(edge.end())?.point(), MERGE_TOL);
            let forward = if es == walk_from {
                walk_from = ee;
                true
            } else if ee == walk_from {
                walk_from = es;
                false
            } else {
                build_ok = false;
                break;
            };
            oriented.push(OrientedEdge::new(eid, forward));
        }
        if !build_ok {
            continue;
        }
        // The walk order alone yields an arbitrary winding; orient the wire CCW
        // around the cap's outward normal so the face winding matches its
        // surface normal (otherwise a manifold-but-inside-out cap could corrupt
        // a downstream boolean). Newell's method gives the loop's normal.
        let (mut nx, mut ny, mut nz) = (0.0_f64, 0.0_f64, 0.0_f64);
        let nverts = verts3d.len();
        for i in 0..nverts {
            let c = verts3d[i];
            let np = verts3d[(i + 1) % nverts];
            nx += (c.y() - np.y()) * (c.z() + np.z());
            ny += (c.z() - np.z()) * (c.x() + np.x());
            nz += (c.x() - np.x()) * (c.y() + np.y());
        }
        if Vec3::new(nx, ny, nz).dot(cap.out_normal) < 0.0 {
            oriented = oriented
                .into_iter()
                .rev()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
        }
        let Ok(wire) = Wire::new(oriented, true) else {
            continue;
        };
        let wid = topo.add_wire(wire);
        new_faces.push(Face::new(
            wid,
            Vec::new(),
            FaceSurface::Plane {
                normal: cap.out_normal,
                d: cap.out_normal.dot(verts3d[0] - origin),
            },
        ));
    }

    for f in new_faces {
        let fid = topo.add_face(f);
        face_ids.push(fid);
        // A synthesised cap has no input source — it is a generated face.
        sources.push(None);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn shell_outward_orientation_outward_cube_is_growth() {
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        assert_eq!(
            shell_is_outward_oriented(&topo, &faces),
            Some(true),
            "a standard outward cube shell must read outward (growth)"
        );
    }

    #[test]
    fn closed_circle_splits_at_matching_mate_vertices() {
        // A full-circle rim on one face vs the SAME circle split into three
        // arcs on the mate face (a pad bore whose rim crosses other faces).
        // The refinement must split the full circle at the mate's vertices so
        // the duplicate merge can pair the two partitions.
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::wire::Wire;

        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };

        // Face A: disc bounded by the full circle (one closed edge).
        let seam = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(0.0),
            1e-7,
        ));
        let full = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle.clone())));
        let wire_a = topo.add_wire(Wire::new(vec![OrientedEdge::new(full, true)], true).unwrap());
        let face_a = topo.add_face(Face::new(wire_a, vec![], plane.clone()));

        // Face B: the same circle as three arcs split at 0, 2π/3, 4π/3.
        let thirds = std::f64::consts::TAU / 3.0;
        let v0 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(0.0),
            1e-7,
        ));
        let v1 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(thirds),
            1e-7,
        ));
        let v2 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(2.0 * thirds),
            1e-7,
        ));
        let mut oes = Vec::new();
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
            let e = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle.clone())));
            oes.push(OrientedEdge::new(e, true));
        }
        let wire_b = topo.add_wire(Wire::new(oes, true).unwrap());
        let face_b = topo.add_face(Face::new(wire_b, vec![], plane));

        let mut face_ids = vec![face_a, face_b];
        split_arc_edges_at_collinear_vertices(&mut topo, &mut face_ids).unwrap();
        merge_duplicate_edges(&mut topo, &mut face_ids).unwrap();

        let keys_a = face_edge_keys(&topo, face_ids[0]).unwrap();
        let keys_b = face_edge_keys(&topo, face_ids[1]).unwrap();
        assert_eq!(keys_a.len(), 3, "full circle must split into 3 arcs");
        let mut ka = keys_a;
        let mut kb = keys_b;
        ka.sort_unstable();
        kb.sort_unstable();
        assert_eq!(ka, kb, "the two partitions must pair edge-for-edge");
    }

    #[test]
    fn closed_circle_with_single_cut_vertex_stays_whole() {
        // One interior cut would produce two arcs sharing BOTH endpoints —
        // the endpoint-keyed merge would conflate them. The refinement must
        // leave the circle whole in that case.
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::wire::Wire;

        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let seam = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(0.0),
            1e-7,
        ));
        let full = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle.clone())));
        let wire_a = topo.add_wire(Wire::new(vec![OrientedEdge::new(full, true)], true).unwrap());
        let face_a = topo.add_face(Face::new(wire_a, vec![], plane.clone()));

        // A lone vertex on the circle, referenced by an unrelated line edge.
        let v_on = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(std::f64::consts::PI),
            1e-7,
        ));
        let v_off = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            Point3::new(5.0, 5.0, 0.0),
            1e-7,
        ));
        let line = topo.add_edge(Edge::new(v_on, v_off, EdgeCurve::Line));
        let back = topo.add_edge(Edge::new(v_off, v_on, EdgeCurve::Line));
        let wire_b = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(line, true), OrientedEdge::new(back, true)],
                true,
            )
            .unwrap(),
        );
        let face_b = topo.add_face(Face::new(wire_b, vec![], plane));

        let mut face_ids = vec![face_a, face_b];
        split_arc_edges_at_collinear_vertices(&mut topo, &mut face_ids).unwrap();

        let keys_a = face_edge_keys(&topo, face_ids[0]).unwrap();
        assert_eq!(keys_a.len(), 1, "single-cut circle must stay whole");
    }

    #[test]
    fn closed_circle_with_offset_seam_splits_into_single_revolution() {
        // Seam at pi, cuts at pi/2 and 3pi/2: the sub-arc spans must be
        // anchored at the seam, or they overlap past one revolution. The
        // three arcs must partition the rim exactly (total sweep TAU) and
        // pair with the mate's partition, which shares the seam vertex.
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::wire::Wire;

        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.0).unwrap();
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let pi = std::f64::consts::PI;
        let seam = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(pi),
            1e-7,
        ));
        let full = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle.clone())));
        let wire_a = topo.add_wire(Wire::new(vec![OrientedEdge::new(full, true)], true).unwrap());
        let face_a = topo.add_face(Face::new(wire_a, vec![], plane.clone()));

        let vs = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(pi),
            1e-7,
        ));
        let v1 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(pi / 2.0),
            1e-7,
        ));
        let v2 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            circle.evaluate(3.0 * pi / 2.0),
            1e-7,
        ));
        let mut oes = Vec::new();
        for (a, b) in [(vs, v2), (v2, v1), (v1, vs)] {
            let e = topo.add_edge(Edge::new(a, b, EdgeCurve::Circle(circle.clone())));
            oes.push(OrientedEdge::new(e, true));
        }
        let wire_b = topo.add_wire(Wire::new(oes, true).unwrap());
        let face_b = topo.add_face(Face::new(wire_b, vec![], plane));

        let mut face_ids = vec![face_a, face_b];
        split_arc_edges_at_collinear_vertices(&mut topo, &mut face_ids).unwrap();
        merge_duplicate_edges(&mut topo, &mut face_ids).unwrap();

        let mut ka = face_edge_keys(&topo, face_ids[0]).unwrap();
        let mut kb = face_edge_keys(&topo, face_ids[1]).unwrap();
        assert_eq!(ka.len(), 3, "offset-seam circle must split into 3 arcs");
        ka.sort_unstable();
        kb.sort_unstable();
        assert_eq!(ka, kb, "seam-anchored partitions must pair edge-for-edge");
    }

    #[test]
    fn closed_ellipse_splits_at_matching_mate_vertices() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::wire::Wire;

        let mut topo = Topology::new();
        let ellipse = Ellipse3D::new(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            1.5,
        )
        .unwrap();
        let plane = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };
        let seam = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            ellipse.evaluate(0.0),
            1e-7,
        ));
        let full = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Ellipse(ellipse.clone())));
        let wire_a = topo.add_wire(Wire::new(vec![OrientedEdge::new(full, true)], true).unwrap());
        let face_a = topo.add_face(Face::new(wire_a, vec![], plane.clone()));

        let thirds = std::f64::consts::TAU / 3.0;
        let v0 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            ellipse.evaluate(0.0),
            1e-7,
        ));
        let v1 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            ellipse.evaluate(thirds),
            1e-7,
        ));
        let v2 = topo.add_vertex(brepkit_topology::vertex::Vertex::new(
            ellipse.evaluate(2.0 * thirds),
            1e-7,
        ));
        let mut oes = Vec::new();
        for (a, b) in [(v0, v1), (v1, v2), (v2, v0)] {
            let e = topo.add_edge(Edge::new(a, b, EdgeCurve::Ellipse(ellipse.clone())));
            oes.push(OrientedEdge::new(e, true));
        }
        let wire_b = topo.add_wire(Wire::new(oes, true).unwrap());
        let face_b = topo.add_face(Face::new(wire_b, vec![], plane));

        let mut face_ids = vec![face_a, face_b];
        split_arc_edges_at_collinear_vertices(&mut topo, &mut face_ids).unwrap();
        merge_duplicate_edges(&mut topo, &mut face_ids).unwrap();

        let mut ka = face_edge_keys(&topo, face_ids[0]).unwrap();
        let mut kb = face_edge_keys(&topo, face_ids[1]).unwrap();
        assert_eq!(ka.len(), 3, "full ellipse must split into 3 arcs");
        ka.sort_unstable();
        kb.sort_unstable();
        assert_eq!(ka, kb, "ellipse partitions must pair edge-for-edge");
    }

    #[test]
    fn shell_outward_orientation_inward_cube_is_rejected() {
        // A single INWARD shell (every face reversed, normals point in — the
        // "Cut leaving only a cavity" case Greptile flagged) must NOT read as
        // growth, or `perform_areas` would invert it into a solid.
        let mut topo = Topology::new();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
        let faces = brepkit_topology::explorer::solid_faces(&topo, solid).unwrap();
        for &fid in &faces {
            let reversed = topo.face(fid).unwrap().is_reversed();
            if let Ok(f) = topo.face_mut(fid) {
                f.set_reversed(!reversed);
            }
        }
        assert_eq!(
            shell_is_outward_oriented(&topo, &faces),
            Some(false),
            "an all-faces-reversed (inward) cube shell must read inward (rejected)"
        );
    }

    #[test]
    fn angle_with_ref_perpendicular() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(0.0, 1.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "90° between X and Y around Z: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_opposite() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(-1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle.abs() - std::f64::consts::PI).abs() < 1e-10,
            "180° between X and -X: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_negative() {
        let d1 = Vec3::new(0.0, 1.0, 0.0);
        let d2 = Vec3::new(1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(
            (angle + std::f64::consts::FRAC_PI_2).abs() < 1e-10,
            "-90° between Y and X around Z: got {angle}"
        );
    }

    #[test]
    fn angle_with_ref_coplanar_same_direction() {
        let d1 = Vec3::new(1.0, 0.0, 0.0);
        let d2 = Vec3::new(1.0, 0.0, 0.0);
        let d_ref = Vec3::new(0.0, 0.0, 1.0);

        let angle = angle_with_ref(d1, d2, d_ref);
        assert!(angle.abs() < 1e-10, "0° between X and X: got {angle}");
    }

    #[test]
    fn open_growth_shell_aborts_assembly_instead_of_vanishing() {
        // Outer lump: a closed unit cube. Second lump: a translated cube with
        // one face REMOVED — a 5-face OPEN shell. Depending on which face is
        // omitted, the corner-fan volume signs it growth or hole; BOTH silent
        // discard paths deleted its material from a result that still read
        // watertight to every edge-pairing gate (the lite fused-foot). The
        // assembler must abort in every open >=4-face case so the boolean
        // falls back to the mesh path. Exercise each omission to cover both
        // branches.
        for omit in 0..6 {
            let mut topo = Topology::new();
            let outer = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
            let second = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);
            let ids: Vec<_> = brepkit_topology::explorer::solid_faces(&topo, second).unwrap();
            let mut moved: std::collections::HashSet<brepkit_topology::vertex::VertexId> =
                std::collections::HashSet::new();
            for &fid in &ids {
                let face = topo.face(fid).unwrap();
                let wires: Vec<_> = std::iter::once(face.outer_wire())
                    .chain(face.inner_wires().iter().copied())
                    .collect();
                for wid in wires {
                    let oes: Vec<_> = topo.wire(wid).unwrap().edges().to_vec();
                    for oe in oes {
                        let e = topo.edge(oe.edge()).unwrap();
                        for vid in [e.start(), e.end()] {
                            if moved.insert(vid) {
                                let pnt = topo.vertex(vid).unwrap().point();
                                topo.vertex_mut(vid).unwrap().set_point(Point3::new(
                                    pnt.x() + 10.0,
                                    pnt.y(),
                                    pnt.z(),
                                ));
                            }
                        }
                    }
                }
            }
            let mut selected: Vec<SelectedFace> = Vec::new();
            for &fid in &brepkit_topology::explorer::solid_faces(&topo, outer).unwrap() {
                selected.push(SelectedFace {
                    face_id: fid,
                    source_face: fid,
                    reversed: false,
                });
            }
            for (i, &fid) in ids.iter().enumerate() {
                if i == omit {
                    continue;
                }
                selected.push(SelectedFace {
                    face_id: fid,
                    source_face: fid,
                    reversed: false,
                });
            }
            // An open shell whose fan volume WINS the outer selection is the
            // sanctioned-lenient open-outer case (Cut/Fuse keep "best
            // available" open results); the fail-safe covers NON-outer lumps
            // only. Skip omissions where the open group becomes the outer.
            let face_ids: Vec<_> = selected.iter().map(|sf| sf.face_id).collect();
            let shells = perform_loops(&topo, &face_ids).unwrap();
            let (growth, _) = perform_areas(&topo, &shells);
            let outer_is_open = growth
                .iter()
                .map(|g| (signed_volume_of_shell(&topo, g), g))
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .is_some_and(|(_, g)| !shell_is_closed(&topo, g));
            if outer_is_open {
                continue;
            }
            let err = build_solid(&mut topo, &selected, &[]);
            assert!(
                matches!(err, Err(AlgoError::AssemblyFailed(_))),
                "omit={omit}: a 5-face open shell must abort assembly, got {err:?}"
            );
        }
    }
}

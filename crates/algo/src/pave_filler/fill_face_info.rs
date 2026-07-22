//! Populate [`FaceInfo`] with classified pave blocks.
//!
//! For each face involved in the boolean, collects:
//! - `pave_blocks_on`: boundary edges that were split (from the face's wires)
//! - `pave_blocks_sc`: section edges from FF intersection curves
//! - `pave_blocks_in`: edges from the opposing solid that lie inside this face
//!
//! [`FaceInfo`]: crate::ds::FaceInfo

use std::collections::HashSet;

use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::FaceId;
use brepkit_topology::vertex::VertexId;

use crate::ds::{GfaArena, Interference, PaveBlockId};
use crate::error::AlgoError;

/// Relative slack on a leaf pave block's parameter span when matching the
/// EF crossing parameter. The crossing `t` and the block endpoints are
/// computed by independent paths, so a few ULPs of rounding can push `t`
/// just outside an adjacent block; this widens each interval by that much.
const LEAF_PARAM_REL_EPS: f64 = 1e-9;

/// Weld-band multiple of the arena tolerance for the EF-IN on-surface test
/// (marched/fitted geometry sits up to ~100x linear tolerance off exact).
const ON_SURFACE_BAND_FACTOR: f64 = 100.0;

/// Maximum surface deviation of an EF-IN leaf as a fraction of its chord —
/// a dimensionless crossing-angle gate. A grazing (near-tangential) contact
/// hugs the surface (socket-loft corner arcs on box walls: 2-18% measured);
/// a transversal crossing's adjacent leaves swing away linearly (corner-pad
/// cap rims on the walls: 24-58% measured). Scale-free by design: absolute
/// bands cannot separate the two classes.
const IN_FACE_MAX_DEVIATION_RATIO: f64 = 0.2;

/// Populate [`FaceInfo`] for all faces with their classified pave blocks.
///
/// - `pave_blocks_on`: split boundary edges of each face
/// - `pave_blocks_sc`: section edges from FF intersections
/// - `pave_blocks_in`: edges from the other solid inside this face
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn perform(topo: &Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    fill_boundary_on(topo, arena)?;
    fill_section_sc(arena);
    fill_ef_in(topo, arena);
    Ok(())
}

/// For each face, find its boundary edges and map their leaf pave blocks
/// into `pave_blocks_on`.
fn fill_boundary_on(topo: &Topology, arena: &mut GfaArena) -> Result<(), AlgoError> {
    let mut all_faces: HashSet<FaceId> = arena.face_info.keys().copied().collect();
    for interf in &arena.interference.ff {
        if let Interference::FF { f1, f2, .. } = interf {
            all_faces.insert(*f1);
            all_faces.insert(*f2);
        }
    }

    for fid in all_faces {
        let face = topo.face(fid)?;

        let mut boundary_edges: HashSet<EdgeId> = HashSet::new();

        let outer_wire = topo.wire(face.outer_wire())?;
        for oe in outer_wire.edges() {
            boundary_edges.insert(oe.edge());
        }
        for &inner_wid in face.inner_wires() {
            if let Ok(inner_wire) = topo.wire(inner_wid) {
                for oe in inner_wire.edges() {
                    boundary_edges.insert(oe.edge());
                }
            }
        }

        // Snapshot pave block data first to avoid aliasing arena borrows.
        let mut on_entries: Vec<(PaveBlockId, VertexId, VertexId)> = Vec::new();
        for eid in boundary_edges {
            if let Some(pb_ids) = arena.edge_pave_blocks.get(&eid).cloned() {
                let leaves = arena.collect_leaf_pave_blocks(&pb_ids);
                for leaf_id in leaves {
                    if let Some(pb) = arena.pave_blocks.get(leaf_id) {
                        let sv = arena.resolve_vertex(pb.start.vertex);
                        let ev = arena.resolve_vertex(pb.end.vertex);
                        on_entries.push((leaf_id, sv, ev));
                    }
                }
            }
        }
        let fi = arena.face_info_mut(fid);
        for (leaf_id, sv, ev) in on_entries {
            fi.pave_blocks_on.insert(leaf_id);
            fi.vertices_on.insert(sv);
            fi.vertices_on.insert(ev);
        }
    }

    Ok(())
}

/// Section edges from FF intersection curves go into `pave_blocks_sc`.
fn fill_section_sc(arena: &mut GfaArena) {
    // Snapshot curve data to avoid aliasing
    let curve_data: Vec<_> = arena
        .curves
        .iter()
        .map(|c| (c.face_a, c.face_b, c.pave_blocks.clone()))
        .collect();

    for (face_a, face_b, pb_ids) in curve_data {
        for &pb_id in &pb_ids {
            // Snapshot vertex IDs before borrowing face_info mutably
            let Some(pb) = arena.pave_blocks.get(pb_id) else {
                continue;
            };
            let sv = arena.resolve_vertex(pb.start.vertex);
            let ev = arena.resolve_vertex(pb.end.vertex);

            let fi_a = arena.face_info_mut(face_a);
            fi_a.pave_blocks_sc.insert(pb_id);
            fi_a.vertices_sc.insert(sv);
            fi_a.vertices_sc.insert(ev);

            let fi_b = arena.face_info_mut(face_b);
            fi_b.pave_blocks_sc.insert(pb_id);
            fi_b.vertices_sc.insert(sv);
            fi_b.vertices_sc.insert(ev);
        }
    }
}

/// Distance from `p` to the face's underlying surface (infinite extent).
///
/// Returns `None` when the measurement is not trustworthy: NURBS
/// projection can silently fall back to a domain-midpoint guess on
/// convergence failure, which would read as a huge distance and falsely
/// reject a leaf that genuinely hugs the surface.
fn dist_to_surface(
    surface: &brepkit_topology::face::FaceSurface,
    p: brepkit_math::vec::Point3,
) -> Option<f64> {
    use brepkit_topology::face::FaceSurface;
    match surface {
        FaceSurface::Plane { normal, d } => {
            Some((normal.dot(brepkit_math::vec::Vec3::new(p.x(), p.y(), p.z())) - d).abs())
        }
        FaceSurface::Nurbs(_) => None,
        other => other
            .project_point(p)
            .and_then(|(u, v)| other.evaluate(u, v))
            .map(|q| (q - p).length()),
    }
}

/// Edges from EF interference go into the face's `pave_blocks_in`.
///
/// Only the leaf pave blocks adjacent to the crossing parameter are
/// inserted — the rest of the edge lies outside the face and would feed
/// out-of-face fragments into the splitter as degenerate inner wires.
fn fill_ef_in(topo: &Topology, arena: &mut GfaArena) {
    // Snapshot EF data
    let ef_data: Vec<_> = arena
        .interference
        .ef
        .iter()
        .filter_map(|interf| {
            if let Interference::EF {
                edge,
                face,
                parameter,
                ..
            } = interf
            {
                Some((*edge, *face, *parameter))
            } else {
                None
            }
        })
        .collect();

    for (edge_id, face_id, parameter) in ef_data {
        if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id).cloned() {
            let leaves = arena.collect_leaf_pave_blocks(&pb_ids);
            let selected: Vec<PaveBlockId> = match parameter {
                Some(t) => {
                    let filtered: Vec<PaveBlockId> = leaves
                        .iter()
                        .copied()
                        .filter(|&leaf_id| {
                            arena.pave_blocks.get(leaf_id).is_some_and(|pb| {
                                let (a, b) = pb.parameter_range();
                                let lo = a.min(b);
                                let hi = a.max(b);
                                let eps = (hi - lo).abs().max(1.0) * LEAF_PARAM_REL_EPS;
                                (lo - eps..=hi + eps).contains(&t)
                            })
                        })
                        .collect();
                    // If rounding pushed `t` outside every leaf interval,
                    // keep all leaves rather than silently dropping the
                    // interference (pre-PR behavior).
                    if filtered.is_empty() {
                        leaves
                    } else {
                        filtered
                    }
                }
                None => leaves,
            };
            // A leaf adjacent to a TRANSVERSAL crossing only touches the
            // face at the crossing point — it does not "lie in" the face,
            // and feeding it to the splitter plants off-surface section
            // edges whose pcurves collapse onto the boundary (the corner-
            // poking pad's cap-rim arcs on the wall planes). Keep a leaf
            // only when it actually hugs the face's surface: the pave
            // endpoints and interior curve samples must all sit within
            // max(weld band, deviation-ratio × chord).
            let on_band = ON_SURFACE_BAND_FACTOR * brepkit_math::tolerance::Tolerance::new().linear;
            let surface = match topo.face(face_id) {
                Ok(f) => f.surface().clone(),
                Err(_) => continue,
            };
            let fi_selected: Vec<PaveBlockId> = selected
                .into_iter()
                .filter(|&leaf_id| {
                    let Some(pb) = arena.pave_blocks.get(leaf_id) else {
                        return false;
                    };
                    let Ok(edge) = topo.edge(pb.original_edge) else {
                        return false;
                    };
                    let (Ok(sv), Ok(ev)) = (topo.vertex(edge.start()), topo.vertex(edge.end()))
                    else {
                        return false;
                    };
                    let (osp, oep) = (sv.point(), ev.point());
                    let (Ok(psv), Ok(pev)) =
                        (topo.vertex(pb.start.vertex), topo.vertex(pb.end.vertex))
                    else {
                        return false;
                    };
                    let (t0, t1) = pb.parameter_range();
                    let span = t1 - t0;
                    let interior = [0.25_f64, 0.5, 0.75].map(|f| {
                        edge.curve()
                            .evaluate_with_endpoints(f.mul_add(span, t0), osp, oep)
                    });
                    let mut dev: f64 = 0.0;
                    for p in std::iter::once(psv.point())
                        .chain(std::iter::once(pev.point()))
                        .chain(interior)
                    {
                        match dist_to_surface(&surface, p) {
                            Some(d) => dev = dev.max(d),
                            // Untrustworthy measurement (NURBS projection can
                            // silently return a wrong foot): keep the leaf —
                            // the pre-gate behavior — rather than risk a
                            // false drop.
                            None => return true,
                        }
                    }
                    let chord = (pev.point() - psv.point()).length();
                    dev <= on_band.max(IN_FACE_MAX_DEVIATION_RATIO * chord)
                })
                .collect();
            let fi = arena.face_info_mut(face_id);
            for leaf_id in fi_selected {
                fi.pave_blocks_in.insert(leaf_id);
            }
        }
    }
}

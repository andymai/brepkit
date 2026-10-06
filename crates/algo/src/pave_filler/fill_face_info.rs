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

/// Absolute ceiling on the ratio band. A LONG leaf crossing at a shallow
/// angle keeps a small deviation/chord ratio while sitting a macroscopic
/// distance off the face — the kumiko band's slope-bottom edge crosses a
/// wall plane 0.05 away at 2.7% of its 1.8 chord, and admitting it feeds
/// the wall's splitter an off-plane section that warps the partition. A
/// genuinely grazing contact hugs the surface in absolute terms as well;
/// beyond this the leaf is transversal regardless of ratio.
const IN_FACE_MAX_DEVIATION_ABS: f64 = 1e-2;

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
    let mut curve_data: Vec<_> = arena
        .curves
        .iter()
        .enumerate()
        .map(|(idx, c)| {
            // A face the curve does not section takes the other's place.
            let skipped = |f| {
                arena
                    .curve_skip_faces
                    .get(&idx)
                    .is_some_and(|faces| faces.contains(&f))
            };
            let face_a = if skipped(c.face_a) {
                c.face_b
            } else {
                c.face_a
            };
            let face_b = if skipped(c.face_b) {
                c.face_a
            } else {
                c.face_b
            };
            (face_a, face_b, c.pave_blocks.clone())
        })
        .collect();
    // A face a curve also sections takes its blocks as a section too (listed
    // as its own pair; the face sets absorb the repeat).
    curve_data.extend(arena.curve_extra_faces.iter().flat_map(|(&idx, faces)| {
        let pbs = arena.curves.get(idx).map(|c| c.pave_blocks.clone());
        faces
            .iter()
            .filter_map(move |&f| pbs.clone().map(|pbs| (f, f, pbs)))
    }));

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

/// A planar face's boundary in its own plane, read on its lines and arcs.
struct PlaneRegion {
    origin: brepkit_math::vec::Point3,
    x: brepkit_math::vec::Vec3,
    y: brepkit_math::vec::Vec3,
    pieces: Vec<brepkit_math::region2d::Boundary2>,
}

impl PlaneRegion {
    /// `None` for a face that is not a plane or has a NURBS edge.
    fn of(topo: &Topology, face_id: FaceId) -> Option<Self> {
        let face = topo.face(face_id).ok()?;
        let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() else {
            return None;
        };
        let first = *topo.wire(face.outer_wire()).ok()?.edges().first()?;
        let origin = topo
            .vertex(topo.edge(first.edge()).ok()?.start())
            .ok()?
            .point();
        let frame = brepkit_math::frame::Frame3::from_normal(origin, *normal).ok()?;
        let pieces =
            brepkit_topology::planar::face_boundary_2d(topo, face_id, origin, frame.x, frame.y)
                .ok()??;
        Some(Self {
            origin,
            x: frame.x,
            y: frame.y,
            pieces,
        })
    }

    /// Whether `p` lies within `tol` of the face's boundary.
    fn on_boundary(&self, p: brepkit_math::vec::Point3, tol: f64) -> bool {
        let d = p - self.origin;
        let flat = brepkit_math::vec::Point2::new(d.dot(self.x), d.dot(self.y));
        brepkit_math::region2d::on_boundary(&self.pieces, flat, tol)
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
                run,
                ..
            } = interf
            {
                Some((*edge, *face, *parameter, *run))
            } else {
                None
            }
        })
        .collect();

    let mut regions: std::collections::HashMap<FaceId, Option<PlaneRegion>> =
        std::collections::HashMap::new();
    for (edge_id, face_id, parameter, run) in ef_data {
        let region = regions
            .entry(face_id)
            .or_insert_with(|| PlaneRegion::of(topo, face_id))
            .as_ref();
        if let Some(pb_ids) = arena.edge_pave_blocks.get(&edge_id).cloned() {
            let leaves = arena.collect_leaf_pave_blocks(&pb_ids);
            let periodic = topo.edge(edge_id).is_ok_and(|e| {
                matches!(
                    e.curve(),
                    brepkit_topology::edge::EdgeCurve::Circle(_)
                        | brepkit_topology::edge::EdgeCurve::Ellipse(_)
                )
            });
            let selected: Vec<PaveBlockId> = match (run, parameter) {
                // An edge lying in part of the face: every leaf within the
                // span, also those cut out of it after the span was found.
                (Some((lo, hi)), _) => leaves
                    .iter()
                    .copied()
                    .filter(|&leaf_id| {
                        arena.pave_blocks.get(leaf_id).is_some_and(|pb| {
                            let (a, b) = pb.parameter_range();
                            let (a, b) = (a.min(b), a.max(b));
                            let eps = (hi - lo).abs().max(1.0) * LEAF_PARAM_REL_EPS;
                            let shift = if periodic {
                                ((a - lo + eps) / std::f64::consts::TAU).floor()
                                    * std::f64::consts::TAU
                            } else {
                                0.0
                            };
                            a - shift >= lo - eps && b - shift <= hi + eps
                        })
                    })
                    .collect(),
                (None, Some(t)) => {
                    // A closed rim's leaves are seam-anchored (see
                    // `make_blocks`); bring the crossing angle into the
                    // same turn before testing containment.
                    let filtered: Vec<PaveBlockId> = leaves
                        .iter()
                        .copied()
                        .filter(|&leaf_id| {
                            arena.pave_blocks.get(leaf_id).is_some_and(|pb| {
                                let (a, b) = pb.parameter_range();
                                let lo = a.min(b);
                                let hi = a.max(b);
                                let eps = (hi - lo).abs().max(1.0) * LEAF_PARAM_REL_EPS;
                                let t = if periodic {
                                    let turn = (t - lo).rem_euclid(std::f64::consts::TAU);
                                    if turn > std::f64::consts::TAU - eps {
                                        lo
                                    } else {
                                        lo + turn
                                    }
                                } else {
                                    t
                                };
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
                (None, None) => leaves,
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
                    let mut ends: f64 = 0.0;
                    let mut middle: f64 = 0.0;
                    for (k, p) in [psv.point(), pev.point()]
                        .into_iter()
                        .chain(interior)
                        .enumerate()
                    {
                        match dist_to_surface(&surface, p) {
                            Some(d) if k < 2 => ends = ends.max(d),
                            Some(d) => middle = middle.max(d),
                            // Untrustworthy measurement (NURBS projection can
                            // silently return a wrong foot): keep the leaf —
                            // the pre-gate behavior — rather than risk a
                            // false drop.
                            None => return true,
                        }
                    }
                    // Both ends on the plane and the middle off it: the leaf
                    // crosses the plane at its ends. Ending on the face's
                    // boundary, it stands in for a chord of that boundary (a
                    // box's corner arc against a faceted loft's facet), which
                    // the face needs; ending elsewhere it lies in no face (a
                    // rod's cap arc between two shallow crossings of a wall,
                    // one on either side of the wall's split).
                    if ends <= on_band
                        && middle > on_band
                        && region.is_some_and(|r| {
                            !r.on_boundary(psv.point(), on_band)
                                || !r.on_boundary(pev.point(), on_band)
                        })
                    {
                        return false;
                    }
                    let dev = ends.max(middle);
                    let chord = (pev.point() - psv.point()).length();
                    // The absolute ceiling applies to STRAIGHT leaves only: a
                    // line's deviation from a crossed plane grows linearly, so
                    // ratio-small + absolute-large means a long transversal
                    // crossing, never a graze. A curved contact (the
                    // calibrated socket-loft corner arcs) hugs via curvature
                    // and legitimately reaches larger absolute deviations at
                    // its span ends; it keeps the pure ratio gate.
                    let band = if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                        (IN_FACE_MAX_DEVIATION_RATIO * chord).min(IN_FACE_MAX_DEVIATION_ABS)
                    } else {
                        IN_FACE_MAX_DEVIATION_RATIO * chord
                    };
                    dev <= on_band.max(band)
                })
                .collect();
            let fi = arena.face_info_mut(face_id);
            for leaf_id in fi_selected {
                fi.pave_blocks_in.insert(leaf_id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use crate::ds::{Pave, PaveBlock};
    use brepkit_math::vec::Point3;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::vertex::Vertex;

    /// An edge lying in part of a face, cut into pieces within that part
    /// after its run was found: every piece of the run lies in the face, and
    /// none outside it.
    #[test]
    fn every_piece_of_a_run_lies_in_the_face() {
        let mut topo = Topology::new();
        let cube =
            brepkit_topology::test_utils::make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
        let bottom = brepkit_topology::explorer::solid_faces(&topo, cube)
            .unwrap()
            .into_iter()
            .find(|&f| {
                matches!(topo.face(f).unwrap().surface(), FaceSurface::Plane { normal, .. } if normal.z() < -0.5)
            })
            .unwrap();
        // x = -1 + 3t along y = 0.5 on z = 0: in the face for t in [1/3, 2/3].
        let at = |t: f64| Point3::new(3.0_f64.mul_add(t, -1.0), 0.5, 0.0);
        let ts = [0.0, 1.0 / 3.0, 0.4, 0.5, 0.6, 2.0 / 3.0, 1.0];
        let verts: Vec<_> = ts
            .iter()
            .map(|&t| topo.add_vertex(Vertex::new(at(t), 1e-7)))
            .collect();
        let edge = topo.add_edge(Edge::new(verts[0], verts[6], EdgeCurve::Line));
        let mut arena = GfaArena::new();
        let leaves: Vec<PaveBlockId> = (0..6)
            .map(|k| {
                arena.pave_blocks.alloc(PaveBlock::new(
                    edge,
                    Pave::new(verts[k], ts[k]),
                    Pave::new(verts[k + 1], ts[k + 1]),
                ))
            })
            .collect();
        arena.edge_pave_blocks.insert(edge, leaves.clone());
        arena.interference.ef.push(Interference::EF {
            edge,
            face: bottom,
            new_vertex: None,
            parameter: Some(0.5),
            run: Some((1.0 / 3.0, 2.0 / 3.0)),
        });

        fill_ef_in(&topo, &mut arena);

        let mut got: Vec<_> = arena
            .face_info(bottom)
            .unwrap()
            .pave_blocks_in
            .iter()
            .copied()
            .collect();
        got.sort_by_key(|pb| pb.index());
        assert_eq!(got, leaves[1..5].to_vec());
    }
}

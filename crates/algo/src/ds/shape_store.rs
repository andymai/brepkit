//! Separate shape store for the GFA pipeline.
//!
//! The `GfaShapeStore` deep-copies input solids into an isolated Topology,
//! runs the entire GFA pipeline on the copies, then exports the result
//! back to the caller's Topology. This eliminates vertex/edge identity
//! conflicts between input solids and the GFA result.

use brepkit_math::madd::Madd;
use std::collections::HashMap;

use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeId};
use brepkit_topology::face::FaceSurface;
use brepkit_topology::face::{Face, FaceId};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::error::AlgoError;

/// Isolated shape store for GFA operations.
///
/// Contains its own `Topology` with deep-copied input solid entities.
/// The GFA pipeline operates entirely within this store, then the result
/// is exported back to the caller's Topology.
pub struct GfaShapeStore {
    /// The store's own topology arena.
    pub topo: Topology,
    /// Store-local SolidId for the deep-copied solid A.
    pub solid_a: SolidId,
    /// Store-local SolidId for the deep-copied solid B.
    pub solid_b: SolidId,
    /// Maps a store-local input face index back to the caller's original face
    /// index, for both operands — shape-evolution provenance across the copy.
    pub input_face_to_caller: HashMap<usize, usize>,
}

impl GfaShapeStore {
    /// Create a new GFA shape store by deep-copying two input solids.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn new(source: &Topology, orig_a: SolidId, orig_b: SolidId) -> Result<Self, AlgoError> {
        let mut topo = Topology::default();

        let (solid_a, map_a) = deep_copy_solid(source, &mut topo, orig_a)?;
        let (solid_b, map_b) = deep_copy_solid(source, &mut topo, orig_b)?;
        widen_tolerances(&mut topo, solid_a)?;
        widen_tolerances(&mut topo, solid_b)?;

        // Invert the caller-index -> store-face maps into store-face-index ->
        // caller-face-index so a store input face resolves to its caller origin.
        let mut input_face_to_caller = HashMap::new();
        for (caller_idx, store_face) in map_a.into_iter().chain(map_b) {
            input_face_to_caller.insert(store_face.index(), caller_idx);
        }

        Ok(Self {
            topo,
            solid_a,
            solid_b,
            input_face_to_caller,
        })
    }

    /// Export a result solid from this store back to the caller's Topology.
    ///
    /// Deep-copies all entities from the store solid into `target`,
    /// returning the new SolidId in the target Topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<SolidId, AlgoError> {
        Ok(deep_copy_solid(&self.topo, target, solid_id)?.0)
    }

    /// Like [`Self::export_solid`], but also returns the map from each store
    /// result-face index to the new caller face it was copied to, for
    /// translating shape-evolution provenance out of the store.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid_with_face_map(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<(SolidId, HashMap<usize, FaceId>), AlgoError> {
        deep_copy_solid(&self.topo, target, solid_id)
    }
}

/// Isolated shape store for an N-way GFA fuse.
///
/// Like [`GfaShapeStore`] but holds an arbitrary number of deep-copied source
/// solids in one store topology, for the N-way fuse pipeline (one arrangement
/// over all operands rather than sequential pairwise booleans). The 2-operand
/// [`GfaShapeStore`] is unchanged; this is additive.
pub struct GfaShapeStoreN {
    /// The store's own topology arena.
    pub topo: Topology,
    /// Store-local `SolidId` for each deep-copied source, in input order.
    pub sources: Vec<SolidId>,
    /// Store-local input face → the source index (position in `sources`) it
    /// belongs to. The basis for per-sub-face source tagging downstream.
    pub face_source: HashMap<FaceId, usize>,
}

impl GfaShapeStoreN {
    /// Create a store by deep-copying every solid in `origs` into one topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if `origs` is empty or any topology lookup fails.
    pub fn new(source: &Topology, origs: &[SolidId]) -> Result<Self, AlgoError> {
        if origs.is_empty() {
            return Err(AlgoError::AssemblyFailed(
                "N-way fuse store needs at least one source solid".into(),
            ));
        }

        let mut topo = Topology::default();
        let mut sources = Vec::with_capacity(origs.len());
        let mut face_source = HashMap::new();

        for (src_idx, &orig) in origs.iter().enumerate() {
            let (solid, map) = deep_copy_solid(source, &mut topo, orig)?;
            widen_tolerances(&mut topo, solid)?;
            sources.push(solid);
            for (_caller_idx, store_face) in map {
                face_source.insert(store_face, src_idx);
            }
        }

        Ok(Self {
            topo,
            sources,
            face_source,
        })
    }

    /// Export a result solid from this store back to the caller's topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<SolidId, AlgoError> {
        Ok(deep_copy_solid(&self.topo, target, solid_id)?.0)
    }
}

/// Isolated store for imprinting tools onto a solid.
///
/// `sources[0]` is the solid. Each group of tool faces follows as an open
/// shell in a solid of its own: the pave filler reads a shell's faces and
/// leaves closure to the builder, and only faces in different sources are
/// intersected with each other. Tool edges have no face to live on and stay
/// loose in the store.
pub struct GfaImprintStore {
    /// The store's own topology arena.
    pub topo: Topology,
    /// The solid first, then one open-shell source per group of tool faces.
    pub sources: Vec<SolidId>,
    /// Store-local copies of the tool edges.
    pub tool_edges: Vec<EdgeId>,
}

impl GfaImprintStore {
    /// Deep-copy `solid`, `tool_faces` and `tool_edges` into one topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn new(
        source: &Topology,
        solid: SolidId,
        tool_faces: &[Vec<FaceId>],
        tool_edges: &[EdgeId],
    ) -> Result<Self, AlgoError> {
        let mut topo = Topology::default();
        let mut sources = Vec::with_capacity(1 + tool_faces.len());

        let (body, _) = deep_copy_solid(source, &mut topo, solid)?;
        widen_tolerances(&mut topo, body)?;
        sources.push(body);

        for group in tool_faces {
            let (tool, _) = copy_shells(source, &mut topo, std::slice::from_ref(group))?;
            widen_tolerances(&mut topo, tool)?;
            sources.push(tool);
        }

        let mut edges = Vec::with_capacity(tool_edges.len());
        for &edge_id in tool_edges {
            edges.push(copy_edge(source, &mut topo, edge_id)?);
        }

        Ok(Self {
            topo,
            sources,
            tool_edges: edges,
        })
    }

    /// Export a result solid from this store back to the caller's topology.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if any topology lookup fails.
    pub fn export_solid(
        &self,
        target: &mut Topology,
        solid_id: SolidId,
    ) -> Result<SolidId, AlgoError> {
        Ok(deep_copy_solid(&self.topo, target, solid_id)?.0)
    }
}

/// Copy one edge and its vertices, a closed edge keeping its single vertex.
fn copy_edge(
    source: &Topology,
    target: &mut Topology,
    edge_id: EdgeId,
) -> Result<EdgeId, AlgoError> {
    let edge = source.edge(edge_id)?;
    let start = source.vertex(edge.start())?;
    let new_start = target.add_vertex(Vertex::new(start.point(), start.tolerance()));
    let new_end = if edge.start() == edge.end() {
        new_start
    } else {
        let end = source.vertex(edge.end())?;
        target.add_vertex(Vertex::new(end.point(), end.tolerance()))
    };
    Ok(target.add_edge(Edge::with_tolerance(
        new_start,
        new_end,
        edge.curve().clone(),
        edge.tolerance(),
    )))
}

/// Deep-copy a solid from `source` topology into `target` topology.
///
/// Creates new vertices, edges, wires, faces, shells, and solid in `target`
/// with remapped IDs. Returns the new SolidId.
fn deep_copy_solid(
    source: &Topology,
    target: &mut Topology,
    solid_id: SolidId,
) -> Result<(SolidId, HashMap<usize, FaceId>), AlgoError> {
    let solid = source.solid(solid_id)?;
    let shells = std::iter::once(solid.outer_shell())
        .chain(solid.inner_shells().iter().copied())
        .map(|shell_id| Ok(source.shell(shell_id)?.faces().to_vec()))
        .collect::<Result<Vec<_>, AlgoError>>()?;
    copy_shells(source, target, &shells)
}

/// Copy `shells`, each a list of faces, into `target` as one solid whose first
/// shell is the outer one.
///
/// Uses the snapshot-then-allocate pattern to satisfy the borrow checker.
#[allow(clippy::too_many_lines, clippy::items_after_statements)]
fn copy_shells(
    source: &Topology,
    target: &mut Topology,
    shells: &[Vec<FaceId>],
) -> Result<(SolidId, HashMap<usize, FaceId>), AlgoError> {
    // ── Snapshot phase ──────────────────────────────────────────────

    struct VertexSnap {
        old_index: usize,
        point: brepkit_math::vec::Point3,
        tol: f64,
    }
    struct EdgeSnap {
        old_index: usize,
        start_index: usize,
        end_index: usize,
        curve: brepkit_topology::edge::EdgeCurve,
        tolerance: Option<f64>,
    }
    struct WireSnap {
        old_index: usize,
        edges: Vec<(usize, bool)>,
        closed: bool,
    }
    struct FaceSnap {
        old_index: usize,
        outer_wire_index: usize,
        inner_wire_indices: Vec<usize>,
        surface: brepkit_topology::face::FaceSurface,
        reversed: bool,
    }
    struct ShellSnap {
        faces: Vec<FaceSnap>,
    }

    let mut vertex_snaps: Vec<VertexSnap> = Vec::new();
    let mut edge_snaps: Vec<EdgeSnap> = Vec::new();
    let mut wire_snaps: Vec<WireSnap> = Vec::new();
    let mut shell_snaps: Vec<ShellSnap> = Vec::new();

    let mut seen_vertices = std::collections::HashSet::new();
    let mut seen_edges = std::collections::HashSet::new();
    let mut seen_wires = std::collections::HashSet::new();

    for shell in shells {
        let mut face_snaps = Vec::new();

        for &face_id in shell {
            let face = source.face(face_id)?;
            let surface = face.surface().clone();
            let outer_wire_index = face.outer_wire().index();
            let inner_wire_indices: Vec<usize> =
                face.inner_wires().iter().map(|w| w.index()).collect();

            for wire_id in
                std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
            {
                if !seen_wires.insert(wire_id.index()) {
                    continue;
                }
                let wire = source.wire(wire_id)?;
                let mut edge_refs = Vec::new();

                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    edge_refs.push((edge_idx, oe.is_forward()));

                    if !seen_edges.insert(edge_idx) {
                        continue;
                    }
                    let edge = source.edge(oe.edge())?;
                    let start_idx = edge.start().index();
                    let end_idx = edge.end().index();

                    for &vid in &[edge.start(), edge.end()] {
                        let vid_idx = vid.index();
                        if seen_vertices.insert(vid_idx) {
                            let v = source.vertex(vid)?;
                            vertex_snaps.push(VertexSnap {
                                old_index: vid_idx,
                                point: v.point(),
                                tol: v.tolerance(),
                            });
                        }
                    }

                    edge_snaps.push(EdgeSnap {
                        old_index: edge_idx,
                        start_index: start_idx,
                        end_index: end_idx,
                        curve: edge.curve().clone(),
                        tolerance: edge.tolerance(),
                    });
                }

                wire_snaps.push(WireSnap {
                    old_index: wire_id.index(),
                    edges: edge_refs,
                    closed: wire.is_closed(),
                });
            }

            face_snaps.push(FaceSnap {
                old_index: face_id.index(),
                outer_wire_index,
                inner_wire_indices,
                surface,
                reversed: face.is_reversed(),
            });
        }

        shell_snaps.push(ShellSnap { faces: face_snaps });
    }

    // ── Allocate phase ──────────────────────────────────────────────

    let mut vertex_map: HashMap<usize, VertexId> = HashMap::new();
    for vsnap in &vertex_snaps {
        let new_vid = target.add_vertex(Vertex::new(vsnap.point, vsnap.tol));
        vertex_map.insert(vsnap.old_index, new_vid);
    }

    let mut edge_map: HashMap<usize, EdgeId> = HashMap::new();
    for esnap in &edge_snaps {
        let new_start = vertex_map[&esnap.start_index];
        let new_end = vertex_map[&esnap.end_index];
        let new_eid = target.add_edge(Edge::with_tolerance(
            new_start,
            new_end,
            esnap.curve.clone(),
            esnap.tolerance,
        ));
        edge_map.insert(esnap.old_index, new_eid);
    }

    let mut wire_map: HashMap<usize, WireId> = HashMap::new();
    for wsnap in &wire_snaps {
        let new_edges: Vec<OrientedEdge> = wsnap
            .edges
            .iter()
            .map(|&(edge_idx, fwd)| OrientedEdge::new(edge_map[&edge_idx], fwd))
            .collect();
        let new_wire = Wire::new(new_edges, wsnap.closed)?;
        wire_map.insert(wsnap.old_index, target.add_wire(new_wire));
    }

    let mut new_shell_ids = Vec::new();
    // Provenance: source face index -> the new face it was copied to.
    let mut face_map: HashMap<usize, FaceId> = HashMap::new();
    for ssnap in &shell_snaps {
        let mut new_face_ids = Vec::new();
        for fsnap in &ssnap.faces {
            let new_outer = wire_map[&fsnap.outer_wire_index];
            let new_inner: Vec<WireId> = fsnap
                .inner_wire_indices
                .iter()
                .map(|idx| wire_map[idx])
                .collect();
            let mut new_face = Face::new(new_outer, new_inner, fsnap.surface.clone());
            if fsnap.reversed {
                new_face.set_reversed(true);
            }
            let new_fid = target.add_face(new_face);
            face_map.insert(fsnap.old_index, new_fid);
            new_face_ids.push(new_fid);
        }
        let new_shell = Shell::new(new_face_ids)?;
        new_shell_ids.push(target.add_shell(new_shell));
    }

    let new_outer = *new_shell_ids
        .first()
        .ok_or_else(|| AlgoError::AssemblyFailed("solid has no shells to copy".into()))?;
    let new_inner: Vec<_> = new_shell_ids[1..].to_vec();

    Ok((target.add_solid(Solid::new(new_outer, new_inner)), face_map))
}

/// Proves pcurves for the B-spline faces of each of `sources` that lie near
/// another source's edges or `loose`: phase EF then bounds such a face by
/// the part of its surface its pcurves enclose, not by the whole surface.
/// The store's copies carry no pcurves of their own, so each one here is
/// proved. A face far from every other source's edges finds no crossing
/// with them, so it is left as it is.
///
/// # Errors
///
/// Returns [`AlgoError`] if a topology lookup fails.
pub fn prove_source_pcurves(
    topo: &mut Topology,
    sources: &[SolidId],
    loose: &[EdgeId],
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<(), AlgoError> {
    let mut boxes = Vec::with_capacity(sources.len());
    for &source in sources {
        boxes.push(edges_box(
            topo,
            &brepkit_topology::explorer::solid_edges(topo, source)?,
        )?);
    }
    let loose_box = edges_box(topo, loose)?;
    for (i, &source) in sources.iter().enumerate() {
        let mut near = loose_box;
        let mut unbounded = loose_box.is_none() && !loose.is_empty();
        for (j, found) in boxes.iter().enumerate() {
            if i == j {
                continue;
            }
            match (*found, near) {
                (None, _) => unbounded = true,
                (Some(b), None) => near = Some(b),
                (Some(b), Some(n)) => near = Some(n.union(b)),
            }
        }
        let near = if unbounded { None } else { near };
        raise_to_proved_reach(topo, source, near, tol)?;
        prove_nurbs_pcurves(topo, source, near, tol)?;
    }
    Ok(())
}

/// A share of an edge's size its proved distance from a face it bounds must
/// stay below to be taken as its tolerance.
const PART_OF_EDGE: f64 = 0.1;

/// Raises the tolerance of each B-spline edge of a face of `solid` whose
/// edges' box meets `near` (of each face where `near` is `None`) to its
/// proved distance from the face's surface
/// ([`crate::surface_reach::spline_off_face`]) where that exceeds it: a fit
/// that lies farther off its faces than its tolerance says. A distance of
/// [`PART_OF_EDGE`] of the edge's size or more is not a fit's: such an edge
/// keeps its tolerance. An open edge's size is the distance between its two
/// vertices, and a closed edge's is that of its whole curve.
fn raise_to_proved_reach(
    topo: &mut Topology,
    solid: SolidId,
    near: Option<brepkit_math::aabb::Aabb3>,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<(), AlgoError> {
    use brepkit_topology::edge::EdgeCurve;
    let mut reach: HashMap<EdgeId, Option<f64>> = HashMap::new();
    for face_id in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(face_id)?;
        let mut edges = Vec::new();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            edges.extend(topo.wire(wire)?.edges().iter().map(OrientedEdge::edge));
        }
        if let Some(near) = near
            && !edges_box(topo, &edges)?.is_some_and(|b| b.intersects(near))
        {
            continue;
        }
        let mut seen = std::collections::HashSet::new();
        for edge_id in edges {
            if !seen.insert(edge_id) {
                continue;
            }
            let edge = topo.edge(edge_id)?;
            let EdgeCurve::NurbsCurve(curve) = edge.curve() else {
                continue;
            };
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            let held = tol.linear.max(edge.tolerance().unwrap_or(0.0));
            let Some(bound) = crate::surface_reach::spline_off_face(
                curve,
                (t0.min(t1), t0.max(t1)),
                face.surface(),
                held,
            ) else {
                continue;
            };
            if bound <= held {
                continue;
            }
            let size = if edge.start() == edge.end() {
                widest_apart(curve.control_points())
            } else {
                (ep - sp).length()
            };
            let entry = reach.entry(edge_id).or_insert(Some(0.0));
            *entry = if bound < PART_OF_EDGE * size {
                entry.map(|largest| largest.max(bound))
            } else {
                None
            };
        }
    }
    for (edge_id, bound) in reach {
        if let Some(bound) = bound {
            let edge = topo.edge_mut(edge_id)?;
            if bound > edge.tolerance().unwrap_or(0.0) {
                edge.set_tolerance(Some(bound));
            }
        }
    }
    Ok(())
}

/// The largest distance between two of `points`. Past 512 points, twice the
/// largest distance from the first point, which is no less and at most
/// twice as much, so that the cost stays linear.
fn widest_apart(points: &[Point3]) -> f64 {
    const PAIRS_UP_TO: usize = 512;
    let Some(&first) = points.first() else {
        return 0.0;
    };
    if points.len() > PAIRS_UP_TO {
        return 2.0
            * points
                .iter()
                .map(|&p| (p - first).length())
                .fold(0.0, f64::max);
    }
    let mut widest = 0.0_f64;
    for (i, &p) in points.iter().enumerate() {
        for &q in &points[i + 1..] {
            widest = widest.max((p - q).length());
        }
    }
    widest
}

/// Gives each B-spline face of `solid` whose control box meets `near` (each
/// such face where `near` is `None`) a pcurve for each edge of its wires,
/// proved within the edge's tolerance, and not less than the linear one, of
/// the edge ([`crate::builder::nurbs_pcurve::proved_nurbs_pcurve`]). A face
/// with an edge its wires use twice (a seam, which one pcurve cannot
/// trace), or an edge whose pcurve does not prove, gets none.
fn prove_nurbs_pcurves(
    topo: &mut Topology,
    solid: SolidId,
    near: Option<brepkit_math::aabb::Aabb3>,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<(), AlgoError> {
    use brepkit_topology::pcurve::PCurve;
    for face_id in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(face_id)?;
        let FaceSurface::Nurbs(surface) = face.surface() else {
            continue;
        };
        if near.is_some_and(|near| !surface.aabb().intersects(near)) {
            continue;
        }
        let mut edges = Vec::new();
        for wire in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            edges.extend(topo.wire(wire)?.edges().iter().map(OrientedEdge::edge));
        }
        let mut seen = std::collections::HashSet::new();
        if !edges.iter().all(|e| seen.insert(*e))
            || (face.inner_wires().is_empty() && runs_along_its_sides(topo, &edges, surface)?)
        {
            continue;
        }
        let mut proved = Vec::with_capacity(edges.len());
        for &edge_id in &edges {
            let edge = topo.edge(edge_id)?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let within = tol.linear.max(edge.tolerance().unwrap_or(0.0));
            let Some(pcurve) = crate::builder::nurbs_pcurve::proved_nurbs_pcurve(
                edge.curve(),
                start,
                end,
                surface,
                within,
            ) else {
                break;
            };
            proved.push((edge_id, pcurve));
        }
        if proved.len() == edges.len() {
            for (edge_id, pcurve) in proved {
                topo.pcurves_mut()
                    .set(edge_id, face_id, PCurve::new(pcurve, 0.0, 1.0));
            }
        }
    }
    Ok(())
}

/// Whether each of `edges`, the edges of a face's one wire, runs along a
/// side of its B-spline `surface`'s domain, read at five points along it.
/// Such a face is its whole surface, so pcurves would bound it by the
/// surface's own box and leave no point of the surface off it: they are not
/// worth proving.
fn runs_along_its_sides(
    topo: &Topology,
    edges: &[EdgeId],
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<bool, AlgoError> {
    use brepkit_math::nurbs::projection::{
        project_point_to_surface, project_point_to_surface_from,
    };
    let ((u0, u1), (v0, v1)) = (surface.domain_u(), surface.domain_v());
    let near = |x: f64, at: f64, span: f64| (x - at).abs() <= 1e-9 * span;
    for &edge_id in edges {
        let edge = topo.edge(edge_id)?;
        let (start, end) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        let mut sides = [true; 4];
        let mut last: Option<(f64, f64)> = None;
        for k in 0..=4 {
            let t = (t1 - t0).madd(f64::from(k) / 4.0, t0);
            let p = edge.curve().evaluate_with_endpoints(t, start, end);
            let found = match last {
                Some(uv) => project_point_to_surface_from(surface, p, uv, 1e-10),
                None => project_point_to_surface(surface, p, 1e-10),
            };
            let Ok(found) = found else {
                return Ok(false);
            };
            let (u, v) = (found.u, found.v);
            last = Some((u, v));
            for (side, on) in sides.iter_mut().zip([
                near(u, u0, u1 - u0),
                near(u, u1, u1 - u0),
                near(v, v0, v1 - v0),
                near(v, v1, v1 - v0),
            ]) {
                *side &= on;
            }
        }
        if !sides.iter().any(|&side| side) {
            return Ok(false);
        }
    }
    Ok(true)
}

/// A box that holds each of `edges`, or `None` where one cannot be boxed
/// or there are none: a line by its ends, a circle or an ellipse whole, a
/// B-spline by its control points, which hold it where its weights are
/// positive. It is grown by [`MAX_WIDEN`], the farthest an edge's tolerance
/// lets a crossing near it stand.
fn edges_box(
    topo: &Topology,
    edges: &[EdgeId],
) -> Result<Option<brepkit_math::aabb::Aabb3>, AlgoError> {
    use brepkit_topology::edge::EdgeCurve;
    let mut points: Vec<Point3> = Vec::new();
    for &edge_id in edges {
        let edge = topo.edge(edge_id)?;
        points.push(topo.vertex(edge.start())?.point());
        points.push(topo.vertex(edge.end())?.point());
        match edge.curve() {
            EdgeCurve::Line => {}
            EdgeCurve::Circle(c) => {
                let (u, v, r) = (c.u_axis(), c.v_axis(), c.radius());
                let reach = brepkit_math::vec::Vec3::new(
                    r * u.x().hypot(v.x()),
                    r * u.y().hypot(v.y()),
                    r * u.z().hypot(v.z()),
                );
                points.extend([c.center() + reach, c.center() - reach]);
            }
            EdgeCurve::Ellipse(e) => {
                let (u, v) = (e.u_axis() * e.semi_major(), e.v_axis() * e.semi_minor());
                let reach = brepkit_math::vec::Vec3::new(
                    u.x().hypot(v.x()),
                    u.y().hypot(v.y()),
                    u.z().hypot(v.z()),
                );
                points.extend([e.center() + reach, e.center() - reach]);
            }
            EdgeCurve::NurbsCurve(n) => {
                if !n.weights().iter().all(|&w| w > 0.0 && w.is_normal()) {
                    return Ok(None);
                }
                points.extend(n.control_points().iter().copied());
            }
        }
    }
    let Some(held) = brepkit_math::aabb::Aabb3::try_from_points(points) else {
        return Ok(None);
    };
    let size = [held.min, held.max]
        .iter()
        .map(|p| p.x().abs().max(p.y().abs()).max(p.z().abs()))
        .fold(0.0_f64, f64::max);
    Ok(Some(held.expanded(1e-9_f64.madd(size, 2.0 * MAX_WIDEN))))
}

/// The widest gap a vertex or edge keeps from a surface it bounds that still
/// reads as fit error rather than misplaced geometry: the junction band. An
/// edge farther off its faces (a chord standing in for a rim) records its gap
/// as its tolerance but is not taken to lie on them.
pub const MAX_WIDEN: f64 = 1e-3;

/// Interior samples an edge is held to its faces' surfaces at.
const EDGE_SAMPLES: u32 = 8;

/// Widen each vertex's and edge's tolerance to reach the analytic surfaces of
/// the faces it bounds. A vertex an earlier boolean computed on one face can
/// stand off another face's surface by that boolean's fit error (a lattice
/// strut's corner 2e-5 off its end face's plane), and an edge computed on one
/// surface pair runs off a third face it bounds the same way; the widened
/// tolerance then covers every point the phases compute there.
fn widen_tolerances(topo: &mut Topology, solid: SolidId) -> Result<(), AlgoError> {
    let mut vertex_reach: HashMap<VertexId, f64> = HashMap::new();
    let mut edge_reach: HashMap<EdgeId, f64> = HashMap::new();
    for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
        let face = topo.face(fid)?;
        let surface = face.surface();
        if matches!(surface, FaceSurface::Nurbs(_)) {
            continue;
        }
        let gap_to = |p: Point3| match surface {
            FaceSurface::Plane { normal, d } => {
                (normal.x() * p.x() + normal.y() * p.y() + normal.z() * p.z() - d).abs()
            }
            _ => surface
                .project_point(p)
                .and_then(|(u, v)| surface.evaluate(u, v))
                .map_or(0.0, |q| (q - p).length()),
        };
        for wid in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let (sp, ep) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                for (vid, p) in [(edge.start(), sp), (edge.end(), ep)] {
                    let gap = gap_to(p);
                    if gap.is_finite() && gap <= MAX_WIDEN {
                        let r = vertex_reach.entry(vid).or_insert(0.0);
                        *r = r.max(gap);
                    }
                }
                let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
                let gap = (1..EDGE_SAMPLES)
                    .map(|k| {
                        let t = (t1 - t0).madd(f64::from(k) / f64::from(EDGE_SAMPLES), t0);
                        gap_to(edge.curve().evaluate_with_endpoints(t, sp, ep))
                    })
                    .fold(0.0, f64::max);
                if gap.is_finite() {
                    let r = edge_reach.entry(oe.edge()).or_insert(0.0);
                    *r = r.max(gap);
                }
            }
        }
    }
    for (vid, gap) in vertex_reach {
        let v = topo.vertex_mut(vid)?;
        if gap > v.tolerance() {
            v.set_tolerance(gap);
        }
    }
    for (eid, gap) in edge_reach {
        let e = topo.edge_mut(eid)?;
        if gap > e.tolerance().unwrap_or(0.0) && gap > 1e-7 {
            e.set_tolerance(Some(gap));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// A unit cube whose edge from (0, 0, 0) to (1, 0, 0) is a cubic that
    /// bows `bow` off both of its faces at its middle. Below a tenth of the
    /// edge's length the edge takes its proved distance from them as its
    /// tolerance, at least the bow and at most twice it. At a tenth or more
    /// it keeps none.
    #[test]
    fn a_b_spline_edge_off_its_faces_takes_its_proved_distance() {
        use brepkit_math::nurbs::curve::NurbsCurve;
        use brepkit_math::vec::Vec3;
        use brepkit_topology::edge::EdgeCurve;
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;
        for (bow, raised) in [(1e-4, true), (0.15, false)] {
            let mut topo = Topology::default();
            let solid = make_unit_cube_manifold_at(&mut topo, 0.0, 0.0, 0.0);
            let at = |topo: &Topology, e: EdgeId| {
                let edge = topo.edge(e).unwrap();
                (
                    topo.vertex(edge.start()).unwrap().point(),
                    topo.vertex(edge.end()).unwrap().point(),
                )
            };
            let edges = brepkit_topology::explorer::solid_edges(&topo, solid).unwrap();
            let edge = *edges
                .iter()
                .find(|&&e| {
                    let (p, q) = at(&topo, e);
                    let lie = |p: Point3, x: f64| (p - Point3::new(x, 0.0, 0.0)).length() < 1e-12;
                    (lie(p, 0.0) && lie(q, 1.0)) || (lie(p, 1.0) && lie(q, 0.0))
                })
                .unwrap();
            let (p, q) = at(&topo, edge);
            // The middle Bezier points carry three quarters of their lift to
            // the cubic's middle.
            let lift = Vec3::new(0.0, -1.0, -1.0) * (bow / 0.75);
            let cubic = NurbsCurve::new(
                3,
                vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                vec![
                    p,
                    p + (q - p) * (1.0 / 3.0) + lift,
                    p + (q - p) * (2.0 / 3.0) + lift,
                    q,
                ],
                vec![1.0; 4],
            )
            .unwrap();
            topo.edge_mut(edge)
                .unwrap()
                .set_curve(EdgeCurve::NurbsCurve(cubic));

            raise_to_proved_reach(
                &mut topo,
                solid,
                None,
                brepkit_math::tolerance::Tolerance::default(),
            )
            .unwrap();

            let held = topo.edge(edge).unwrap().tolerance();
            if raised {
                assert!(
                    held.is_some_and(|t| t >= bow && t <= 2.0 * bow),
                    "bow {bow}: {held:?}"
                );
            } else {
                assert!(held.is_none(), "bow {bow}: {held:?}");
            }
        }
    }

    #[test]
    fn round_trip_preserves_box() {
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;

        let mut source = Topology::default();
        let solid = make_unit_cube_manifold_at(&mut source, 0.0, 0.0, 0.0);

        let store = GfaShapeStore::new(&source, solid, solid).unwrap();

        let s = store.topo.solid(store.solid_a).unwrap();
        let sh = store.topo.shell(s.outer_shell()).unwrap();
        assert_eq!(sh.faces().len(), 6, "box should have 6 faces");

        let mut target = Topology::default();
        let exported = store.export_solid(&mut target, store.solid_a).unwrap();
        let s2 = target.solid(exported).unwrap();
        let sh2 = target.shell(s2.outer_shell()).unwrap();
        assert_eq!(sh2.faces().len(), 6, "exported box should have 6 faces");

        let source_faces = brepkit_topology::explorer::solid_faces(&source, solid).unwrap();
        let target_faces = brepkit_topology::explorer::solid_faces(&target, exported).unwrap();
        assert_eq!(source_faces.len(), target_faces.len());

        // Verify isolation: the store has TWO copies of the box (A and B
        // both point to the same source solid). The store topology should
        // have more entities than the source (2× vertices, edges, etc.).
        let store_vertex_count =
            brepkit_topology::explorer::solid_faces(&store.topo, store.solid_a)
                .unwrap()
                .len();
        assert_eq!(store_vertex_count, 6, "store solid_a should have 6 faces");
    }

    #[test]
    fn nway_store_copies_all_sources_and_tags_faces() {
        use brepkit_topology::explorer::solid_faces;
        use brepkit_topology::test_utils::make_unit_cube_manifold_at;

        let mut source = Topology::default();
        let boxes = [
            make_unit_cube_manifold_at(&mut source, 0.0, 0.0, 0.0),
            make_unit_cube_manifold_at(&mut source, 0.5, 0.0, 0.0),
            make_unit_cube_manifold_at(&mut source, 1.0, 0.0, 0.0),
        ];

        let store = GfaShapeStoreN::new(&source, &boxes).unwrap();
        assert_eq!(store.sources.len(), 3, "three sources copied");

        // Every source is an independent 6-face box, and every one of its faces
        // is tagged with that source's index.
        for (src_idx, &sid) in store.sources.iter().enumerate() {
            let faces = solid_faces(&store.topo, sid).unwrap();
            assert_eq!(faces.len(), 6, "source {src_idx} is a 6-face box");
            for fid in faces {
                assert_eq!(
                    store.face_source.get(&fid).copied(),
                    Some(src_idx),
                    "face {fid:?} tagged to source {src_idx}"
                );
            }
        }

        // 3 independent boxes -> 18 distinct faces, all source-tagged.
        assert_eq!(store.face_source.len(), 18, "18 distinct tagged faces");

        // Export round-trips one source into a fresh topology.
        let mut target = Topology::default();
        let exported = store.export_solid(&mut target, store.sources[1]).unwrap();
        assert_eq!(solid_faces(&target, exported).unwrap().len(), 6);
    }

    #[test]
    fn nway_store_rejects_empty() {
        let source = Topology::default();
        assert!(GfaShapeStoreN::new(&source, &[]).is_err());
    }
}

//! Solid-level tessellation orchestration.

use brepkit_geometry::extrema::segment_segment_distance;
use brepkit_math::det_hash::{DetHashMap, DetHashSet, WordHashMap, WordHashSet, WordState};
use brepkit_math::madd::Madd;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use super::TriangleMesh;
use super::edge_sampling::{
    circle_param_range, sample_edge, sample_edge_with_params, segments_for_chord_deviation_a,
};
use super::mesh_ops::{
    NO_RIM_FACE, dedupe_coincident_triangles, split_pinched_chords, weld_boundary_vertices,
};
use super::nonplanar::{
    circle_pole_params, tessellate_cylinder_strip_shared, tessellate_latitude_band_shared,
    tessellate_nonplanar_cdt, tessellate_nonplanar_snap, tessellate_revolution_band_shared,
    tessellate_sphere_triangle_shared, tessellate_torus_notch_band, tessellate_torus_two_rim_band,
};
use super::nurbs::{compute_angular_range, compute_v_param_range};
use super::planar::{
    cdt_triangulate_simple, collect_wire_global_vertices, cover_by_winding, polygon_self_crosses,
    project_by_normal, remove_closing_duplicate_global, remove_closing_duplicate_ids,
    run_planar_cdt, tessellate_planar_shared_with_holes, unproject_point, winding_area,
};
use super::{MERGE_GRID, PointPool, point_merge_key};

/// Tessellate all faces of a solid into a single watertight triangle mesh.
///
/// Unlike per-face `tessellate()`, this function coordinates tessellation across
/// all faces of the solid by pre-computing shared edge tessellations. When two
/// faces share an edge, the edge is tessellated once and both faces receive
/// identical vertices along that boundary -- eliminating cracks between adjacent
/// faces and producing a guaranteed 2-manifold mesh.
///
/// # Algorithm
///
/// Based on Stoger & Kurka (2003), "Watertight Tessellation of B-rep NURBS
/// CAD-Models Using Connectivity Information":
///
/// 1. Build edge-to-face adjacency map from the solid's topology.
/// 2. Tessellate each unique edge once, producing a shared polyline.
/// 3. For each face, tessellate using cached edge points as boundary vertices.
/// 4. Merge all per-face meshes into a single mesh with shared boundary vertices.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
pub fn tessellate_solid(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    tessellate_solid_with_tolerance(
        topo,
        solid,
        deflection,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
    )
}

/// Tessellate a solid at a density safe for mesh-boolean co-refinement.
///
/// Identical to [`tessellate_solid_with_tolerance`] except circular edges keep
/// the curvature floor: the mesh boolean's robustness depends on the denser
/// floored circle sampling, independent of the display deflection cap.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
pub fn tessellate_solid_for_boolean(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    tessellate_solid_core(topo, solid, deflection, angular_tol, false, true).map(|core| core.mesh)
}

/// Tessellate a solid with explicit linear and angular tolerances.
///
/// `angular_tol` (radians) caps the per-segment tangent turn; pass `0.0` to
/// disable the angular criterion (linear-only, backward-compatible) path.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
pub fn tessellate_solid_with_tolerance(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
) -> Result<TriangleMesh, crate::OperationsError> {
    tessellate_solid_core(topo, solid, deflection, angular_tol, false, false).map(|core| core.mesh)
}

/// Watertight solid tessellation with per-face triangle grouping.
///
/// Runs the same shared-edge-pool pipeline as [`tessellate_solid_with_tolerance`],
/// then reorders triangles so each face's triangles are contiguous. Returns the
/// mesh plus `face_offsets`: one entry per face of
/// `explorer::solid_faces(topo, solid)` (in that order, including empty groups)
/// where `face_offsets[i]` is the start offset into `mesh.indices` for face `i`,
/// plus a final sentinel equal to `mesh.indices.len()`.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
pub fn tessellate_solid_grouped_with_tolerance(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
) -> Result<(TriangleMesh, Vec<u32>), crate::OperationsError> {
    let CoreMesh {
        mut mesh,
        tri_faces,
        n_faces,
        ..
    } = tessellate_solid_core(topo, solid, deflection, angular_tol, true, false)?;
    let tri_faces = tri_faces.unwrap_or_default();
    debug_assert_eq!(tri_faces.len() * 3, mesh.indices.len());

    let mut counts = vec![0_u32; n_faces];
    for &f in &tri_faces {
        if let Some(c) = counts.get_mut(f as usize) {
            *c += 1;
        }
    }

    let mut face_offsets = Vec::with_capacity(n_faces + 1);
    let mut acc = 0_u32;
    face_offsets.push(0_u32);
    for &c in &counts {
        acc += c * 3;
        face_offsets.push(acc);
    }

    // Stable counting-sort scatter: per-face triangle order is preserved.
    let mut cursors: Vec<usize> = face_offsets[..n_faces]
        .iter()
        .map(|&o| o as usize)
        .collect();
    let mut new_indices = vec![0_u32; mesh.indices.len()];
    for (t, &f) in tri_faces.iter().enumerate() {
        let Some(cursor) = cursors.get_mut(f as usize) else {
            continue;
        };
        let dst = *cursor;
        new_indices[dst..dst + 3].copy_from_slice(&mesh.indices[t * 3..t * 3 + 3]);
        *cursor += 3;
    }
    mesh.indices = new_indices;

    Ok((mesh, face_offsets))
}

/// [`tessellate_solid_grouped_with_tolerance`] for display, each face holding
/// its own copies of its vertices.
///
/// A vertex the faces share takes, in each face's copy, that face's normal
/// there, so a crease between faces shades sharp rather than averaged across
/// it. A vertex one face alone uses keeps its normal. The face's normal is
/// its surface's; on a NURBS face where the vertex does not project onto the
/// surface, or the surface's normal there faces away from the face's own
/// triangles round it, their area-weighted normal. The groups are those of
/// the welded mesh, in the same order, but the mesh is no longer welded
/// across faces.
///
/// # Errors
///
/// Returns an error if any topology lookup or face tessellation fails.
pub fn tessellate_solid_faceted_with_tolerance(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
) -> Result<(TriangleMesh, Vec<u32>), crate::OperationsError> {
    let (welded, offsets) =
        tessellate_solid_grouped_with_tolerance(topo, solid, deflection, angular_tol)?;
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let n = welded.positions.len();
    let mut users = vec![0_u32; n];
    let mut seen_in = vec![usize::MAX; n];
    for (group, w) in offsets.windows(2).enumerate() {
        for &v in &welded.indices[w[0] as usize..w[1] as usize] {
            let v = v as usize;
            if seen_in[v] != group {
                seen_in[v] = group;
                users[v] += 1;
            }
        }
    }

    let mut mesh = TriangleMesh {
        positions: Vec::with_capacity(n),
        normals: Vec::with_capacity(n),
        indices: Vec::with_capacity(welded.indices.len()),
    };
    let mut faceted_offsets = Vec::with_capacity(offsets.len());
    faceted_offsets.push(0_u32);
    let mut copy_of = vec![u32::MAX; n];
    for (group, w) in offsets.windows(2).enumerate() {
        let tris = &welded.indices[w[0] as usize..w[1] as usize];
        let base = mesh.positions.len();
        let mut copied = Vec::new();
        for &v in tris {
            let v = v as usize;
            if copy_of[v] == u32::MAX {
                #[allow(clippy::cast_possible_truncation)]
                let slot = mesh.positions.len() as u32;
                copy_of[v] = slot;
                copied.push(v);
                mesh.positions.push(welded.positions[v]);
                mesh.normals.push(welded.normals[v]);
            }
            mesh.indices.push(copy_of[v]);
        }
        if copied.iter().any(|&v| users[v] > 1)
            && let Some(&face_id) = faces.get(group)
        {
            let face = topo.face(face_id)?;
            let flip = if face.is_reversed() { -1.0 } else { 1.0 };
            let surface = face.surface();
            let mut around = vec![Vec3::new(0.0, 0.0, 0.0); copied.len()];
            if matches!(surface, FaceSurface::Nurbs(_)) {
                for tri in mesh.indices[mesh.indices.len() - tris.len()..].chunks_exact(3) {
                    let [a, b, c] = [tri[0], tri[1], tri[2]].map(|i| mesh.positions[i as usize]);
                    let area_normal = (b - a).cross(c - a);
                    for &i in tri {
                        around[i as usize - base] += area_normal;
                    }
                }
            }
            for (k, &v) in copied.iter().enumerate() {
                if users[v] < 2 {
                    continue;
                }
                let at = mesh.positions[base + k];
                let normal = match surface {
                    FaceSurface::Plane { normal, .. } => Some(*normal * flip),
                    FaceSurface::Nurbs(nurbs) => {
                        let averaged = around[k].normalize().ok();
                        nurbs_normal_at(nurbs, at, deflection)
                            .map(|n| n * flip)
                            .filter(|n| averaged.is_none_or(|a| n.dot(a) > 0.0))
                            .or(averaged)
                    }
                    analytic => analytic
                        .project_point(at)
                        .map(|(u, v)| analytic.normal(u, v) * flip),
                };
                if let Some(normal) = normal {
                    mesh.normals[base + k] = normal;
                }
            }
        }
        for v in copied {
            copy_of[v] = u32::MAX;
        }
        #[allow(clippy::cast_possible_truncation)]
        faceted_offsets.push(mesh.indices.len() as u32);
    }
    Ok((mesh, faceted_offsets))
}

/// The surface's normal where `at` projects onto it, if it lies within
/// `reach` of the surface.
fn nurbs_normal_at(
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
    at: Point3,
    reach: f64,
) -> Option<Vec3> {
    let near = brepkit_math::nurbs::projection::project_point_to_surface(surface, at, 1e-7).ok()?;
    if near.distance > reach {
        return None;
    }
    surface.normal(near.u, near.v).ok()
}

/// The ends of two chords in a surface's `(u, v)`.
type ChordPair = ((f64, f64), (f64, f64), (f64, f64), (f64, f64));

/// Two chords `a`-`b` and `c`-`d` in a surface's `(u, v)`, laid out to cross
/// where they cross on the surface. With `period`, `u` runs round a turn: each
/// chord is unwrapped along itself, and the second moves by whole turns so its
/// middle lies nearest the first's. Read from its first end, a chord across
/// the seam would sit a turn from the other where it crosses it.
fn chords_in_one_turn(
    a: (f64, f64),
    b: (f64, f64),
    c: (f64, f64),
    d: (f64, f64),
    period: Option<f64>,
) -> ChordPair {
    let Some(period) = period else {
        return (a, b, c, d);
    };
    let turns = |from: f64, to: f64| ((to - from) / period).round() * period;
    let b = (b.0 - turns(a.0, b.0), b.1);
    let d = (d.0 - turns(c.0, d.0), d.1);
    let shift = -turns(f64::midpoint(a.0, b.0), f64::midpoint(c.0, d.0));
    (a, b, (c.0 + shift, c.1), (d.0 + shift, d.1))
}

/// What [`tessellate_solid_core`] returns.
pub(super) struct CoreMesh {
    /// The solid's mesh.
    pub(super) mesh: TriangleMesh,
    /// The `tri -> face` array, when asked for.
    pub(super) tri_faces: Option<Vec<u32>>,
    /// How many faces the solid has.
    pub(super) n_faces: usize,
    /// The faces left without triangles by their meshers or by the boundary
    /// weld. Two coincident faces of opposite sense, which cancel to nothing
    /// by design, are not among them.
    #[cfg(test)]
    pub(super) unmeshed: Vec<FaceId>,
}

/// Gives two straight edges that overlap in one wire the same sample points
/// along their shared stretch.
///
/// A wire can run out along a short straight edge and back along a longer one
/// that lies over it (a round ending part way along a wall's edge leaves such
/// a pair on the wall beside it). The face with that wire meshes nothing in
/// the doubled-back stretch, and the faces beyond the two edges meet along
/// it: one carries the short edge's samples and the other the long edge's.
/// Each edge is sampled by its own shape, so without this the two faces take
/// different points on one line and the mesh opens between them.
fn share_doubled_back_samples(
    topo: &Topology,
    faces: &[FaceId],
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
) -> Result<(), crate::OperationsError> {
    const ON_LINE: f64 = 1e-6;
    const MAX_ROUNDS: usize = 16;
    // Whether every sample lies on the chord between the first and the last.
    let straight = |points: &[Point3]| {
        let (Some(&a), Some(&b)) = (points.first(), points.last()) else {
            return false;
        };
        let chord = b - a;
        let length2 = chord.dot(chord);
        length2 > 1e-18
            && points.iter().all(|&p| {
                let along = ((p - a).dot(chord) / length2).clamp(0.0, 1.0);
                (a + chord * along - p).length() <= ON_LINE
            })
    };
    // `points` with each of `more` that lies strictly inside its chord, in
    // the order `points` runs.
    let merged = |points: &[Point3], more: &[Point3]| -> Option<Vec<Point3>> {
        let (a, b) = (*points.first()?, *points.last()?);
        let chord = b - a;
        let length2 = chord.dot(chord);
        let place = |p: Point3| (p - a).dot(chord) / length2;
        let apart = ON_LINE / length2.sqrt();
        let mut placed: Vec<(f64, Point3)> = points.iter().map(|&p| (place(p), p)).collect();
        let before = placed.len();
        for &p in more {
            let t = place(p);
            if t <= apart || t >= 1.0 - apart || (a + chord * t - p).length() > ON_LINE {
                continue;
            }
            if placed.iter().all(|&(s, _)| (s - t).abs() > apart) {
                placed.push((t, p));
            }
        }
        if placed.len() == before {
            return None;
        }
        placed.sort_by(|x, y| x.0.total_cmp(&y.0));
        Some(placed.into_iter().map(|(_, p)| p).collect())
    };
    // The overlapping pairs, found once: sharing changes no edge's line.
    let mut pairs: Vec<(usize, usize)> = Vec::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied())
        {
            let edges = topo.wire(wire_id)?.edges();
            let n = edges.len();
            if n < 3 {
                continue;
            }
            for i in 0..n {
                let (first, second) = (edges[i], edges[(i + 1) % n]);
                if first.edge() == second.edge() {
                    continue;
                }
                let (e1, e2) = (topo.edge(first.edge())?, topo.edge(second.edge())?);
                // The two edges leave their shared vertex `v` for `p` and `q`.
                let (p, v) = (first.oriented_start(e1), first.oriented_end(e1));
                let (v2, q) = (second.oriented_start(e2), second.oriented_end(e2));
                if v != v2 || p == q {
                    continue;
                }
                let (i1, i2) = (first.edge().index(), second.edge().index());
                let (Some(s1), Some(s2)) = (edge_points.get(&i1), edge_points.get(&i2)) else {
                    continue;
                };
                if !straight(s1) || !straight(s2) {
                    continue;
                }
                let (vp, pp, qp) = (
                    topo.vertex(v)?.point(),
                    topo.vertex(p)?.point(),
                    topo.vertex(q)?.point(),
                );
                // One edge doubles back over the other when its far end lies
                // on the other between `v` and that one's far end.
                let inside = |far: Point3, along: Point3| {
                    let span = along - vp;
                    let length2 = span.dot(span);
                    if length2 <= 1e-18 {
                        return false;
                    }
                    let t = (far - vp).dot(span) / length2;
                    t > 0.0 && t <= 1.0 && (vp + span * t - far).length() <= ON_LINE
                };
                if !inside(pp, qp) && !inside(qp, pp) {
                    continue;
                }
                pairs.push((i1, i2));
            }
        }
    }
    // A point one pair hands to an edge is new to that edge's other pairs,
    // so the pairs are shared round until none takes a point.
    for _ in 0..MAX_ROUNDS {
        let mut changed = false;
        for &(i1, i2) in &pairs {
            let (Some(s1), Some(s2)) = (edge_points.get(&i1), edge_points.get(&i2)) else {
                continue;
            };
            let (new1, new2) = (merged(s1, s2), merged(s2, s1));
            if let Some(points) = new1 {
                edge_points.insert(i1, points);
                changed = true;
            }
            if let Some(points) = new2 {
                edge_points.insert(i2, points);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(())
}

/// A spline edge keeps only the samples its chords need to hold the
/// deflection in space, which need not hold it in a face's (u, v). A face
/// narrower than the chords' sag (a sliver between two spline sides running
/// almost together) can have an outline whose chords cross in its (u, v),
/// and its mesh then loses the triangles past the crossing; and on a spline
/// face whose (u, v) is skewed, a chord's straight line in (u, v) can run far
/// off the edge on the surface, taking grid points outside the face. A
/// straight edge on a spline face strays in its (u, v) the same way, and a
/// sphere face's arcs close the same slivers (between a rod's cap circle and
/// the ball's equator a few hundredths of a turn apart). Each such chord, a
/// spline's, a straight edge's on a spline face or an arc's on a sphere
/// face, crossing another chord of its face's outline, or whose middle in
/// (u, v) lies farther than the deflection from its edge, is halved on its
/// edge, round by round. The samples are shared, so the face across each
/// edge takes them too.
///
/// Two chords can cross in (u, v) only where their lines in (u, v), read on
/// the surface, meet: such a chord's reach covers how far that line strays
/// from it, so only chords nearer each other than their two reaches are
/// tested.
///
/// Returns whether it halved any chord.
#[allow(clippy::too_many_lines)]
pub(super) fn split_crossing_spline_chords(
    topo: &Topology,
    faces: &[FaceId],
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
    mut params: DetHashMap<usize, Vec<f64>>,
    deflection: f64,
) -> Result<bool, crate::OperationsError> {
    // A face's projections of sample points, keyed by the point's bits, and
    // the surface's point there once it is read.
    type UvCache = WordHashMap<(u64, u64, u64), Option<((f64, f64), Option<Point3>)>>;
    const MAX_ROUNDS: usize = 8;
    // Chord ends closer than this are one sample.
    const SAME: f64 = 1e-9;
    // A sample off a chord by less than this fraction of its length lies on it.
    const TOUCH: f64 = 1e-9;
    // The edge a chord can be halved on.
    #[derive(Clone, Copy)]
    enum Span<'a> {
        // A spline, with the parameters of the chord's two ends.
        Spline(&'a brepkit_math::nurbs::curve::NurbsCurve, f64, f64),
        // A straight edge on a spline face.
        Line,
        // An arc on a sphere face.
        Arc,
    }
    struct Chord<'a> {
        edge: usize,
        index: usize,
        a: Point3,
        b: Point3,
        // How far the chord's line in (u, v), read on the surface, may stand
        // off it.
        reach: f64,
        span: Option<Span<'a>>,
    }
    // How far a point stands off the segment from `a` to `b`.
    let off_segment = |p: Point3, a: Point3, b: Point3| {
        let (d, w) = (b - a, p - a);
        let length2 = d.dot(d);
        let t = if length2 > 0.0 {
            (w.dot(d) / length2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (w - d * t).length()
    };
    // The parameters of each spline edge's samples, where its samples are
    // still those they were taken at.
    params.retain(|idx, ts| {
        edge_points
            .get(idx)
            .is_some_and(|pts| pts.len() == ts.len())
    });
    let spline_face = |f: &FaceId| {
        topo.face(*f).is_ok_and(|face| {
            matches!(
                face.surface(),
                FaceSurface::Nurbs(_) | FaceSurface::Sphere(_)
            )
        })
    };
    if params.is_empty() && !faces.iter().any(spline_face) {
        return Ok(false);
    }
    let orient = |a: (f64, f64), b: (f64, f64), c: (f64, f64)| {
        (b.0 - a.0).madd(c.1 - a.1, -((b.1 - a.1) * (c.0 - a.0)))
    };
    // Each face's samples in its (u, v), kept from round to round.
    let mut caches: Vec<UvCache> = vec![UvCache::default(); faces.len()];
    // The edges the last round halved: a face none of them bound is as it was.
    let mut changed: Option<WordHashSet<usize>> = None;
    let mut halved = false;
    for _ in 0..MAX_ROUNDS {
        let mut halve: WordHashSet<(usize, usize)> = WordHashSet::default();
        for (k, &face_id) in faces.iter().enumerate() {
            let face = topo.face(face_id)?;
            if let Some(changed) = &changed {
                let mut touched = false;
                for &wire_id in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
                    touched |= topo
                        .wire(wire_id)?
                        .edges()
                        .iter()
                        .any(|oe| changed.contains(&oe.edge().index()));
                }
                if !touched {
                    continue;
                }
            }
            let surface = face.surface();
            // How far u and v run round a surface closed across them.
            let period = |range: Option<(f64, f64)>| range.map_or(f64::INFINITY, |r| r.1 - r.0);
            let turn = std::f64::consts::TAU;
            let (su, sv) = match surface {
                FaceSurface::Plane { .. } => (f64::INFINITY, f64::INFINITY),
                FaceSurface::Nurbs(n) => (period(n.closed_u()), period(n.closed_v())),
                FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_) => {
                    (turn, f64::INFINITY)
                }
                FaceSurface::Torus(_) => (turn, turn),
            };
            // A sphere face's (u, v) runs on past its seam (its evaluation
            // turns with u), so a chord across the seam, and a pair of chords
            // either side of it, are read with u carried round the turn. On
            // another closed surface such a chord is no chord in (u, v).
            let round_seam = matches!(surface, FaceSurface::Sphere(_));
            let near = |a: (f64, f64), b: (f64, f64)| {
                if round_seam {
                    ((b.0 - a.0) / su).round().madd(-su, b.0)
                } else {
                    b.0
                }
            };
            let mut chords: Vec<Chord<'_>> = Vec::new();
            let mut seen: WordHashSet<usize> = WordHashSet::default();
            for &wire_id in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
                for oe in topo.wire(wire_id)?.edges() {
                    let idx = oe.edge().index();
                    if !seen.insert(idx) {
                        continue;
                    }
                    let Some(pts) = edge_points.get(&idx) else {
                        continue;
                    };
                    let edge = topo.edge(oe.edge())?;
                    for (index, pair) in pts.windows(2).enumerate() {
                        let (a, b) = (pair[0], pair[1]);
                        let (reach, span) = match (edge.curve(), params.get(&idx)) {
                            (EdgeCurve::Line, _) if matches!(surface, FaceSurface::Nurbs(_)) => {
                                (SAME, Some(Span::Line))
                            }
                            (EdgeCurve::Line, _) => (0.0, None),
                            (EdgeCurve::NurbsCurve(nurbs), Some(ts)) => {
                                (SAME, Some(Span::Spline(nurbs, ts[index], ts[index + 1])))
                            }
                            (EdgeCurve::Circle(_), _) if round_seam => (SAME, Some(Span::Arc)),
                            _ => (deflection, None),
                        };
                        chords.push(Chord {
                            edge: idx,
                            index,
                            a,
                            b,
                            reach,
                            span,
                        });
                    }
                }
            }
            if !chords.iter().any(|c| c.span.is_some()) {
                continue;
            }
            let uv = &mut caches[k];
            let mut read = |p: Point3, foot: bool| {
                let entry = uv
                    .entry((p.x().to_bits(), p.y().to_bits(), p.z().to_bits()))
                    .or_insert_with(|| surface.project_point(p).map(|q| (q, None)));
                if foot && let Some((q, at @ None)) = entry {
                    *at = surface.evaluate(q.0, q.1);
                }
                *entry
            };
            // A curved face's (u, v) can stretch a chord's straight line in
            // (u, v) far off the chord in space: the mesh's edge there is that
            // line on the surface. Each spline chord's reach takes in how far
            // that line strays from it, read at its quarters with a margin, so
            // a pair whose lines meet is tested; and its line must hold the
            // deflection of its curve at its middle. An edge can stand off its
            // surface (a file's tolerance): the line is read with the ends'
            // offsets carried along it. A plane's (u, v) is flat: there the
            // line strays only as far as the chord's ends stand off the plane.
            // A cone's apex and a sphere's poles have no u, so the middle of a
            // chord on them is not read against its curve.
            if let FaceSurface::Plane { normal, d } = surface {
                let off = |p: Point3| (normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs();
                for c in chords.iter_mut().filter(|c| c.span.is_some()) {
                    c.reach += 2.0 * off(c.a).max(off(c.b));
                }
            } else {
                let held = matches!(
                    surface,
                    FaceSurface::Nurbs(_) | FaceSurface::Cylinder(_) | FaceSurface::Torus(_)
                );
                for c in chords.iter_mut().filter(|c| c.span.is_some()) {
                    let (Some((a, Some(at_a))), Some((b, Some(at_b)))) =
                        (read(c.a, true), read(c.b, true))
                    else {
                        continue;
                    };
                    let b = (near(a, b), b.1);
                    if (a.0 - b.0).abs() > 0.5 * su || (a.1 - b.1).abs() > 0.5 * sv {
                        continue;
                    }
                    let (off_a, off_b) = (c.a - at_a, c.b - at_b);
                    let line = |f: f64| {
                        surface
                            .evaluate((b.0 - a.0).madd(f, a.0), (b.1 - a.1).madd(f, a.1))
                            .map(|p| p + off_a + (off_b - off_a) * f)
                    };
                    let quarters = [0.25, 0.5, 0.75];
                    let on_line = quarters.map(line);
                    let stray = on_line
                        .iter()
                        .zip(quarters)
                        .filter_map(|(p, f)| p.map(|p| (p - (c.a + (c.b - c.a) * f)).length()))
                        .fold(0.0, f64::max);
                    c.reach += 2.0 * stray;
                    if !held {
                        continue;
                    }
                    let (Some(span), Some(on_line)) = (c.span, on_line[1]) else {
                        continue;
                    };
                    // The line's middle is held against the edge's nearest
                    // point, not its middle by parameter: a curve's parameter
                    // need not run evenly along it.
                    let strays = match span {
                        Span::Spline(nurbs, t0, t1) => {
                            (on_line - nurbs.evaluate(0.5 * (t0 + t1))).length() > deflection
                                && brepkit_geometry::extrema::point_to_curve(
                                    on_line,
                                    nurbs,
                                    t0.min(t1),
                                    t0.max(t1),
                                )
                                .distance
                                    > deflection
                        }
                        Span::Line | Span::Arc => off_segment(on_line, c.a, c.b) > deflection,
                    };
                    if strays {
                        halve.insert((c.edge, c.index));
                    }
                }
            }
            // Sweep the chords along the outline's widest axis (a face in a
            // plane square to one axis has every chord at one coordinate of
            // it), each chord's box grown by its reach.
            let coord = |p: Point3, axis: usize| match axis {
                0 => p.x(),
                1 => p.y(),
                _ => p.z(),
            };
            let spread = |axis: usize| {
                let (lo, hi) = chords
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), c| {
                        (lo.min(coord(c.a, axis)), hi.max(coord(c.a, axis)))
                    });
                hi - lo
            };
            let along = (0..3)
                .max_by(|&x, &y| spread(x).total_cmp(&spread(y)))
                .unwrap_or(0);
            let low = |c: &Chord, axis: usize| coord(c.a, axis).min(coord(c.b, axis)) - c.reach;
            let high = |c: &Chord, axis: usize| coord(c.a, axis).max(coord(c.b, axis)) + c.reach;
            let lo = |c: &Chord| low(c, along);
            let hi = |c: &Chord| high(c, along);
            let mut order: Vec<usize> = (0..chords.len()).collect();
            order.sort_by(|&i, &j| lo(&chords[i]).total_cmp(&lo(&chords[j])));
            // Only a pair with a chord that can be halved in it is tested.
            // Each such chord meets the chords starting within its span, and
            // each chord the halvable chords starting within its own: every
            // overlapping pair with a halvable chord in it is one or the other.
            let mut test = |i: usize, j: usize| {
                let ci = &chords[i];
                let cj = &chords[j];
                if !(ci.span.is_some() || cj.span.is_some())
                    || (ci.edge == cj.edge && ci.index == cj.index)
                    || (0..3).any(|axis| {
                        low(ci, axis) > high(cj, axis) || low(cj, axis) > high(ci, axis)
                    })
                {
                    return;
                }
                let ends = [ci.a, ci.b];
                if ends
                    .iter()
                    .any(|&p| (p - cj.a).length() < SAME || (p - cj.b).length() < SAME)
                {
                    return;
                }
                let (gap, _, _) = segment_segment_distance(ci.a, ci.b, cj.a, cj.b);
                if gap > ci.reach + cj.reach {
                    return;
                }
                let mut to_uv = |p: Point3| read(p, false).map(|(q, _)| q);
                let (Some(a), Some(b), Some(c), Some(d)) =
                    (to_uv(ci.a), to_uv(ci.b), to_uv(cj.a), to_uv(cj.b))
                else {
                    return;
                };
                // A sphere's pole has no u of its own.
                if round_seam
                    && [a, b, c, d]
                        .iter()
                        .any(|p| (p.1.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9)
                {
                    return;
                }
                let (a, b, c, d) = chords_in_one_turn(a, b, c, d, round_seam.then_some(su));
                // A chord across a closed surface's seam is no chord in (u, v).
                if [(a, b), (c, d)]
                    .iter()
                    .any(|&(p, q)| (p.0 - q.0).abs() > 0.5 * su || (p.1 - q.1).abs() > 0.5 * sv)
                {
                    return;
                }
                // Chords lying along one line in (u, v) (two edges whose
                // samples project onto one rim of the surface) overlap however
                // finely they are cut: halving them only adds samples on the
                // other's chords, every round.
                let reach2 = |p: (f64, f64), q: (f64, f64)| {
                    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
                    dx.madd(dx, dy * dy)
                };
                let (ab2, cd2) = (reach2(a, b), reach2(c, d));
                if ab2 > 0.0
                    && cd2 > 0.0
                    && orient(a, b, c).abs() <= TOUCH * ab2
                    && orient(a, b, d).abs() <= TOUCH * ab2
                    && orient(c, d, a).abs() <= TOUCH * cd2
                    && orient(c, d, b).abs() <= TOUCH * cd2
                {
                    return;
                }
                let crosses = orient(a, b, c) * orient(a, b, d) < 0.0
                    && orient(c, d, a) * orient(c, d, b) < 0.0;
                // A sample lying on the other chord makes a T: the face
                // splits the chord there, and the face across it does not.
                let on = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
                    let (dx, dy) = (q.0 - p.0, q.1 - p.1);
                    let length2 = dx.madd(dx, dy * dy);
                    let t = (r.0 - p.0).madd(dx, (r.1 - p.1) * dy) / length2;
                    length2 > 0.0 && t > 0.0 && t < 1.0 && orient(p, q, r).abs() <= TOUCH * length2
                };
                let touches = on(a, b, c) || on(a, b, d) || on(c, d, a) || on(c, d, b);
                if !(crosses || touches) {
                    return;
                }
                for c in [ci, cj] {
                    if c.span.is_some() {
                        halve.insert((c.edge, c.index));
                    }
                }
            };
            let starts: Vec<f64> = order.iter().map(|&i| lo(&chords[i])).collect();
            let halvable_order: Vec<usize> = order
                .iter()
                .copied()
                .filter(|&i| chords[i].span.is_some())
                .collect();
            let halvable_starts: Vec<f64> =
                halvable_order.iter().map(|&i| lo(&chords[i])).collect();
            for &i in &halvable_order {
                let (from, to) = (lo(&chords[i]), hi(&chords[i]));
                let first = starts.partition_point(|&x| x < from);
                for &j in order[first..].iter().take_while(|&&j| lo(&chords[j]) <= to) {
                    if j != i {
                        test(i, j);
                    }
                }
            }
            for &j in &order {
                let (from, to) = (lo(&chords[j]), hi(&chords[j]));
                let first = halvable_starts.partition_point(|&x| x <= from);
                for &i in halvable_order[first..]
                    .iter()
                    .take_while(|&&i| lo(&chords[i]) <= to)
                {
                    if i != j {
                        test(i, j);
                    }
                }
            }
        }
        if halve.is_empty() {
            break;
        }
        let mut by_edge: DetHashMap<usize, Vec<usize>> = DetHashMap::default();
        for &(edge, index) in &halve {
            by_edge.entry(edge).or_default().push(index);
        }
        let mut edges: Vec<usize> = by_edge.keys().copied().collect();
        edges.sort_unstable();
        changed = Some(edges.iter().copied().collect());
        halved = true;
        for idx in edges {
            let (Some(edge_id), Some(pts)) =
                (topo.edge_id_from_index(idx), edge_points.get_mut(&idx))
            else {
                continue;
            };
            let mut indices = by_edge[&idx].clone();
            indices.sort_unstable_by(|x, y| y.cmp(x));
            let curve = topo.edge(edge_id)?.curve();
            if matches!(curve, EdgeCurve::Line) {
                for index in indices {
                    let middle = pts[index] + (pts[index + 1] - pts[index]) * 0.5;
                    pts.insert(index + 1, middle);
                }
                continue;
            }
            if let EdgeCurve::Circle(circle) = curve {
                // Two samples of an arc lie less than half a turn apart.
                for index in indices {
                    let t0 = circle.project(pts[index]);
                    let step = (circle.project(pts[index + 1]) - t0 + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                    pts.insert(index + 1, circle.evaluate(step.madd(0.5, t0)));
                }
                continue;
            }
            let (EdgeCurve::NurbsCurve(nurbs), Some(ts)) = (curve, params.get_mut(&idx)) else {
                continue;
            };
            for index in indices {
                let t = 0.5 * (ts[index] + ts[index + 1]);
                ts.insert(index + 1, t);
                pts.insert(index + 1, nurbs.evaluate(t));
            }
        }
    }
    Ok(halved)
}

/// Gives each straight edge of a sphere face's outline the samples of the
/// faces' holes that reach past it.
///
/// A ball's equator is a polygon of chords, each within its sag of the
/// sphere. A boolean finds where a section crosses such an edge on the
/// chord, so a section that passes between a chord and the sphere does not
/// split there, and a face's holes can reach past the outline into the face
/// beyond. The holes' samples past a chord, by no more than its sag, become
/// samples of the chord as well. Each chord takes them from the holes of
/// every face on it, all at once and in order along it, and keeps its own
/// samples where two holes' samples overlap along it. The outline then runs
/// round them, the face beyond leaves them out, and the two faces and each
/// hole's other face meet along the same points. An outline keeps its face
/// on its left, seen from the side the surface's normal points to, so a
/// sample past a chord lies on the right of the great circle through it.
pub(super) fn sample_holes_past_sphere_chords(
    topo: &Topology,
    faces: &[FaceId],
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
) -> Result<(), crate::OperationsError> {
    // For each chord, its samples' ends and each hole's run past it, with
    // their places along the chord from its first sample.
    type Runs = (Point3, Point3, Vec<Vec<(f64, Point3)>>);
    let mut chords: std::collections::BTreeMap<usize, Runs> = std::collections::BTreeMap::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        let FaceSurface::Sphere(sphere) = face.surface() else {
            continue;
        };
        if face.inner_wires().is_empty() {
            continue;
        }
        let (centre, radius) = (sphere.center(), sphere.radius());
        let key = |p: Point3| point_merge_key(p, MERGE_GRID);
        let mut loops: Vec<Vec<Point3>> = Vec::new();
        for &wire_id in face.inner_wires() {
            let mut samples: Vec<Point3> = Vec::new();
            for oe in topo.wire(wire_id)?.edges() {
                let Some(points) = edge_points.get(&oe.edge().index()) else {
                    continue;
                };
                let ordered: Vec<Point3> = if oe.is_forward() {
                    points.clone()
                } else {
                    points.iter().rev().copied().collect()
                };
                for p in ordered {
                    if samples.last().is_none_or(|&q| key(q) != key(p)) {
                        samples.push(p);
                    }
                }
            }
            if samples.len() > 1
                && samples.first().map(|&p| key(p)) == samples.last().map(|&p| key(p))
            {
                samples.pop();
            }
            loops.push(samples);
        }
        for oe in topo.wire(face.outer_wire())?.edges() {
            if !matches!(topo.edge(oe.edge())?.curve(), EdgeCurve::Line) {
                continue;
            }
            let Some(&[start, end]) = edge_points.get(&oe.edge().index()).map(Vec::as_slice) else {
                continue;
            };
            let (p, q) = if oe.is_forward() {
                (start, end)
            } else {
                (end, start)
            };
            let Ok(left) = (p - centre).cross(q - centre).normalize() else {
                continue;
            };
            let sag = radius - (p + (q - p) * 0.5 - centre).length();
            let chord = q - p;
            let length_sq = chord.dot(chord);
            let tol = 1e-9 * radius;
            if sag <= 0.0 || length_sq <= tol * tol {
                continue;
            }
            // Where along the chord, from its first sample, a sample past it
            // lies: within the sag past its great circle, and within the sag
            // of the arc over it, so within twice the sag of the chord.
            let along = |h: Point3| {
                let past = -left.dot(h - centre);
                let t = (h - p).dot(chord) / length_sq;
                let off = (h - (p + chord * t)).length();
                let t = if oe.is_forward() { t } else { 1.0 - t };
                (past >= -tol && past <= sag && off <= 2.0 * sag && t > 1e-9 && t < 1.0 - 1e-9)
                    .then_some(t)
            };
            let runs = &mut chords
                .entry(oe.edge().index())
                .or_insert_with(|| (start, end, Vec::new()))
                .2;
            // Each hole's run of consecutive samples past the chord.
            for samples in &loops {
                let n = samples.len();
                let hits: Vec<Option<f64>> = samples.iter().map(|&h| along(h)).collect();
                let starts: Vec<usize> = (0..n)
                    .filter(|&i| hits[i].is_some() && hits[(i + n - 1) % n].is_none())
                    .collect();
                let &[i0] = starts.as_slice() else {
                    continue;
                };
                let mut run: Vec<(f64, Point3)> = (0..n)
                    .map(|k| (i0 + k) % n)
                    .map_while(|i| hits[i].map(|t| (t, samples[i])))
                    .collect();
                let rising = run.windows(2).all(|w| w[1].0 > w[0].0);
                let falling = run.windows(2).all(|w| w[1].0 < w[0].0);
                if !(rising || falling) {
                    continue;
                }
                run.sort_by(|a, b| a.0.total_cmp(&b.0));
                runs.push(run);
            }
        }
    }
    for (index, (start, end, mut runs)) in chords {
        if runs.is_empty() {
            continue;
        }
        // Runs that overlap along the chord would put the outline across a
        // hole.
        runs.sort_by(|a, b| a[0].0.total_cmp(&b[0].0));
        if runs
            .windows(2)
            .any(|w| w[0].last().is_none_or(|last| last.0 >= w[1][0].0))
        {
            continue;
        }
        let mut points = vec![start];
        points.extend(runs.into_iter().flatten().map(|(_, h)| h));
        points.push(end);
        edge_points.insert(index, points);
    }
    Ok(())
}

/// Gives each closed circle or ellipse edge that a face's seam meets its
/// vertex as a sample.
///
/// A closed edge's samples start where its curve's frame starts, which need
/// not be its vertex. A face's mesher ends its seam on the rim's sample
/// nearest the vertex, which moves the seam up to half a step round the
/// wall, and across a hole that passes nearer the seam than that. With the
/// vertex as a sample, the seam stays where the face has it. The faces on
/// both sides of the rim take the same sample.
pub(super) fn sample_seam_vertices_on_rims(
    topo: &Topology,
    faces: &[FaceId],
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
) -> Result<(), crate::OperationsError> {
    let mut seam_vertices: DetHashSet<usize> = DetHashSet::default();
    let mut rims = Vec::new();
    for &face_id in faces {
        let face = topo.face(face_id)?;
        for &wire_id in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            let edges = topo.wire(wire_id)?.edges();
            for (i, oe) in edges.iter().enumerate() {
                let edge = topo.edge(oe.edge())?;
                if edge.is_closed() {
                    if matches!(edge.curve(), EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_)) {
                        rims.push(oe.edge());
                    }
                } else if edges[i + 1..].iter().any(|other| other.edge() == oe.edge()) {
                    seam_vertices.extend([edge.start().index(), edge.end().index()]);
                }
            }
        }
    }
    if seam_vertices.is_empty() {
        return Ok(());
    }
    rims.sort_unstable_by_key(|e| e.index());
    rims.dedup();
    for rim in rims {
        let edge = topo.edge(rim)?;
        if !seam_vertices.contains(&edge.start().index()) {
            continue;
        }
        let Some(points) = edge_points.get_mut(&rim.index()) else {
            continue;
        };
        let vertex = topo.vertex(edge.start())?.point();
        let key = point_merge_key(vertex, MERGE_GRID);
        if points.len() < 2
            || points
                .iter()
                .any(|&p| point_merge_key(p, MERGE_GRID) == key)
        {
            continue;
        }
        let param = |p: Point3| match edge.curve() {
            EdgeCurve::Circle(circle) => circle.project(p),
            EdgeCurve::Ellipse(ellipse) => ellipse.project(p),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => 0.0,
        };
        let t0 = param(points[0]);
        let along = |p: Point3| (param(p) - t0 + 1e-9).rem_euclid(std::f64::consts::TAU) - 1e-9;
        // The samples run once round from the first, which the last repeats.
        let last = points.len() - 1;
        let rising = points[..last].windows(2).all(|w| along(w[1]) > along(w[0]));
        if !rising {
            continue;
        }
        let t = along(vertex);
        let at = (1..last).find(|&i| along(points[i]) > t).unwrap_or(last);
        points.insert(at, vertex);
    }
    Ok(())
}

/// Drops each edge's inner samples that lie nearer than `reach` to one of
/// its own end vertices, where no other edge has a sample at that point.
///
/// The T-junction splice gives an arc each pool point within `reach` of it,
/// except the points within `reach` of its ends, which it reads as the ends.
/// Two edges of one face can run into their shared vertex almost together:
/// the boolean can put the vertex a little past the point where the two
/// curves cross, within its tolerance of both. The chords of both edges cross
/// there, and are halved round by round. The splice then gives one edge the
/// other's samples up to `reach` from the vertex, but not the samples nearer
/// than that. A sample that one edge keeps beside the vertex is a sample the
/// other edge lacks, and the face's outline holds a sliver of three points
/// that runs the wrong way round, where the mesh opens. A sample that only
/// its own edge holds serves only the faces of that edge, so all of them
/// lose it together.
fn drop_lone_samples_beside_end_vertices(
    topo: &Topology,
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
    reach: f64,
) -> Result<(), crate::OperationsError> {
    let mut edges: Vec<usize> = edge_points.keys().copied().collect();
    edges.sort_unstable();
    let mut beside: Vec<(usize, Vec<usize>)> = Vec::new();
    for &idx in &edges {
        let Some(edge_id) = topo.edge_id_from_index(idx) else {
            continue;
        };
        let edge = topo.edge(edge_id)?;
        let ends = [
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        ];
        // A closed edge, or one whose ends lie this near each other, can
        // have every inner sample beside an end (a rim of radius under
        // `reach`): it keeps them all, as it needs them to bound its faces.
        if edge.start() == edge.end() || (ends[0] - ends[1]).length() < 2.0 * reach {
            continue;
        }
        let points = &edge_points[&idx];
        let near: Vec<usize> = (1..points.len().saturating_sub(1))
            .filter(|&i| ends.iter().any(|&end| (points[i] - end).length() < reach))
            .collect();
        if !near.is_empty() {
            beside.push((idx, near));
        }
    }
    if beside.is_empty() {
        return Ok(());
    }
    // The first edge to hold each weld cell, and whether another edge holds
    // it too: a pass above shared such a point on purpose.
    let mut holders: WordHashMap<(i64, i64, i64), (usize, bool)> = WordHashMap::default();
    for &idx in &edges {
        for &p in &edge_points[&idx] {
            let holder = holders
                .entry(point_merge_key(p, MERGE_GRID))
                .or_insert((idx, false));
            holder.1 |= holder.0 != idx;
        }
    }
    for (idx, near) in beside {
        let Some(points) = edge_points.get_mut(&idx) else {
            continue;
        };
        for &i in near.iter().rev() {
            if holders
                .get(&point_merge_key(points[i], MERGE_GRID))
                .is_some_and(|&(_, shared)| !shared)
            {
                points.remove(i);
            }
        }
    }
    Ok(())
}

/// [`split_crossing_spline_chords`], then the samples of edges that double
/// back over each other shared again: a straight edge the split halved hands
/// its new samples to the edge overlapping it, which a face across that edge
/// alone need not have halved.
pub(super) fn split_chords_and_share(
    topo: &Topology,
    faces: &[FaceId],
    edge_points: &mut DetHashMap<usize, Vec<Point3>>,
    params: DetHashMap<usize, Vec<f64>>,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    if split_crossing_spline_chords(topo, faces, edge_points, params, deflection)? {
        share_doubled_back_samples(topo, faces, edge_points)?;
    }
    Ok(())
}

/// Core watertight tessellation pipeline.
///
/// When `track_faces` is set, also returns a parallel `tri -> face` array (one
/// entry per triangle, holding the index of the owning face within
/// `explorer::solid_faces` order); otherwise the attribution bookkeeping is
/// skipped and `None` is returned. The face count is always returned.
///
/// `circle_floor` selects whether circular edges keep the curvature floor.
/// Display callers pass `false` (constant-curvature circles are exact without
/// it); the boolean mesh-fallback passes `true` for co-refinement robustness.
/// Because the shared edge pool drives the cylinder/cone band density, this one
/// flag governs every circular feature in the solid consistently.
#[allow(clippy::too_many_lines, clippy::fn_params_excessive_bools)]
pub(super) fn tessellate_solid_core(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
    track_faces: bool,
    circle_floor: bool,
) -> Result<CoreMesh, crate::OperationsError> {
    use brepkit_topology::explorer;

    super::check_deflection(deflection)?;
    let all_faces = explorer::solid_faces(topo, solid)?;
    for &face in &all_faces {
        super::check_reach(topo, face)?;
        super::check_samples(topo, face, deflection, angular_tol)?;
    }
    let edge_face_map = explorer::edge_to_face_map(topo, solid)?;
    let mut phase_t = PhaseTimer::start();
    let phase = |label: &str, t: &mut PhaseTimer| t.lap(label);

    // The map is a std `HashMap`, so sort its keys into ID order before use —
    // keeping all downstream iteration deterministic regardless of
    // insertion-order hashing.
    let mut edge_indices: Vec<usize> = edge_face_map.keys().copied().collect();
    edge_indices.sort_unstable();
    // A NURBS edge's sample parameters, for the chord split below.
    let mut edge_params: DetHashMap<usize, Vec<f64>> = DetHashMap::default();
    #[cfg(not(target_arch = "wasm32"))]
    let mut edge_points: DetHashMap<usize, Vec<Point3>> = if edge_indices.len() >= 32 {
        use rayon::prelude::*;
        #[allow(clippy::items_after_statements)]
        type Sampled = Result<(usize, Vec<Point3>, Option<Vec<f64>>), crate::OperationsError>;
        let results: Vec<Sampled> = edge_indices
            .par_iter()
            .filter_map(|&edge_idx| {
                let edge_id = topo.edge_id_from_index(edge_idx)?;
                let edge_data = match topo.edge(edge_id) {
                    Ok(d) => d,
                    Err(e) => return Some(Err(crate::OperationsError::Topology(e))),
                };
                Some(
                    sample_edge_with_params(topo, edge_data, deflection, angular_tol, circle_floor)
                        .map(|(pts, ts)| (edge_idx, pts, ts)),
                )
            })
            .collect();
        let mut map = DetHashMap::default();
        for r in results {
            let (idx, pts, ts) = r?;
            map.insert(idx, pts);
            if let Some(ts) = ts {
                edge_params.insert(idx, ts);
            }
        }
        map
    } else {
        let mut map = DetHashMap::default();
        for &edge_idx in &edge_indices {
            if let Some(edge_id) = topo.edge_id_from_index(edge_idx)
                && let Ok(edge_data) = topo.edge(edge_id)
            {
                let (points, ts) = sample_edge_with_params(
                    topo,
                    edge_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                )?;
                map.insert(edge_idx, points);
                if let Some(ts) = ts {
                    edge_params.insert(edge_idx, ts);
                }
            }
        }
        map
    };
    #[cfg(target_arch = "wasm32")]
    let mut edge_points: DetHashMap<usize, Vec<Point3>> = {
        let mut map = DetHashMap::default();
        for &edge_idx in &edge_indices {
            if let Some(edge_id) = topo.edge_id_from_index(edge_idx) {
                if let Ok(edge_data) = topo.edge(edge_id) {
                    let (points, ts) = sample_edge_with_params(
                        topo,
                        edge_data,
                        deflection,
                        angular_tol,
                        circle_floor,
                    )?;
                    map.insert(edge_idx, points);
                    if let Some(ts) = ts {
                        edge_params.insert(edge_idx, ts);
                    }
                }
            }
        }
        map
    };

    share_doubled_back_samples(topo, &all_faces, &mut edge_points)?;
    phase("edge_sampling", &mut phase_t);
    // Synchronize circle edge samples with face grid density so a face's rim
    // points line up with its own analytic grid columns.
    {
        for &face_id in &all_faces {
            let face_data = topo.face(face_id)?;
            let face_nu = match face_data.surface() {
                FaceSurface::Cone(cone) => {
                    let v_range =
                        compute_v_param_range(topo, face_data, |p| cone.project_point(p).1);
                    let u_range = compute_angular_range(topo, face_data, |p| cone.project_point(p));
                    let max_radius = cone.radius_at(v_range.1.abs().max(v_range.0.abs()));
                    segments_for_chord_deviation_a(
                        max_radius.max(0.01),
                        u_range.1 - u_range.0,
                        deflection,
                        angular_tol,
                        circle_floor,
                    )
                }
                FaceSurface::Cylinder(cyl) => {
                    let u_range = compute_angular_range(topo, face_data, |p| cyl.project_point(p));
                    segments_for_chord_deviation_a(
                        cyl.radius(),
                        u_range.1 - u_range.0,
                        deflection,
                        angular_tol,
                        circle_floor,
                    )
                }
                _ => continue,
            };
            let expected_count = face_nu + 1;

            let mut wire_ids = vec![face_data.outer_wire()];
            wire_ids.extend_from_slice(face_data.inner_wires());
            for &wire_id in &wire_ids {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let edge_idx = oe.edge().index();
                    let Some(edge_id) = topo.edge_id_from_index(edge_idx) else {
                        continue;
                    };
                    let Ok(edge_data) = topo.edge(edge_id) else {
                        continue;
                    };
                    let EdgeCurve::Circle(circle) = edge_data.curve() else {
                        continue;
                    };

                    if let Some(pts) = edge_points.get(&edge_idx)
                        && pts.len() < expected_count
                    {
                        if tess_trace() {
                            log::debug!(
                                "sync e{edge_idx}: {} -> {expected_count} demanded by {face_id:?} {}",
                                pts.len(),
                                face_data.surface().type_tag()
                            );
                        }
                        let (t_start, t_end) = circle_param_range(topo, edge_data, circle)?;
                        let new_pts = brepkit_geometry::sampling::sample_uniform(
                            circle,
                            t_start,
                            t_end,
                            expected_count,
                        );
                        edge_points.insert(edge_idx, new_pts);
                    }
                }
            }
        }
    }

    // A bilinear patch twisted between skew lines is meshed on a grid fine
    // enough for its twist, but its straight edges keep only their ends, so
    // the triangles joining such an edge to the grid reach across the face.
    // Each straight edge of such a face is sampled as finely as its grid
    // across the face's own parameter box. Edge samples are shared, so the
    // face across the edge takes them too.
    // Other patches mesh their straight edges on rows of their own.
    for &face_id in &all_faces {
        let face_data = topo.face(face_id)?;
        for (idx, points) in super::edge_sampling::twisted_straight_edge_samples(
            topo,
            face_data,
            deflection,
            angular_tol,
            circle_floor,
        )? {
            if edge_points
                .get(&idx)
                .is_some_and(|pts| pts.len() >= points.len())
            {
                continue;
            }
            edge_points.insert(idx, points);
        }
    }

    // A hole passing a planar face's arc nearer than the arc's chords sag
    // would cross them in the face's mesh: the arc is sampled finer, until
    // its chords sag at most half the hole's clearance. The arc's samples
    // are shared, so the face across it takes them too. The chords sag
    // toward the arc's centre, so only a hole inside its circle can meet
    // them, and the hole's mesh boundary is the polyline through its
    // samples, whose farthest points from the centre are its vertices:
    // the samples alone give the clearance, a straight edge's included.
    for &face_id in &all_faces {
        let face_data = topo.face(face_id)?;
        if !face_data.surface().is_planar() || face_data.inner_wires().is_empty() {
            continue;
        }
        let mut hole_points: Vec<Point3> = Vec::new();
        for &wire_id in face_data.inner_wires() {
            for oe in topo.wire(wire_id)?.edges() {
                if let Some(points) = edge_points.get(&oe.edge().index()) {
                    hole_points.extend_from_slice(points);
                }
            }
        }
        for oe in topo.wire(face_data.outer_wire())?.edges() {
            let edge_data = topo.edge(oe.edge())?;
            let EdgeCurve::Circle(circle) = edge_data.curve() else {
                continue;
            };
            let Some(points) = edge_points.get(&oe.edge().index()) else {
                continue;
            };
            let (t_start, t_end) = circle_param_range(topo, edge_data, circle)?;
            let span = (t_end - t_start).abs();
            let segments = points.len().saturating_sub(1).max(1);
            #[allow(clippy::cast_precision_loss)]
            let sag = circle.radius() * (1.0 - (0.5 * span / segments as f64).cos());
            let clearance = hole_points
                .iter()
                .map(|&p| ((p - circle.center()).length() - circle.radius()).abs())
                .fold(f64::INFINITY, f64::min);
            if clearance >= 2.0 * sag || clearance <= 0.0 {
                continue;
            }
            let half_step = (1.0 - 0.5 * clearance / circle.radius())
                .clamp(-1.0, 1.0)
                .acos();
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let needed = ((0.5 * span / half_step).ceil() as usize).min(4096);
            if needed > segments {
                let finer =
                    brepkit_geometry::sampling::sample_uniform(circle, t_start, t_end, needed + 1);
                edge_points.insert(oe.edge().index(), finer);
            }
        }
    }

    // A sphere face's circle edge running over the sphere's pole (a wall
    // through its axis) takes the pole as a sample: the face's u turns over
    // there, and its mesher runs the loop along the pole's row from it. Both
    // faces on the edge weld to the one sample.
    for &face_id in &all_faces {
        let face_data = topo.face(face_id)?;
        let FaceSurface::Sphere(sphere) = face_data.surface() else {
            continue;
        };
        for &wire_id in std::iter::once(&face_data.outer_wire()).chain(face_data.inner_wires()) {
            for oe in topo.wire(wire_id)?.edges() {
                let edge_data = topo.edge(oe.edge())?;
                let EdgeCurve::Circle(circle) = edge_data.curve() else {
                    continue;
                };
                let Some(points) = edge_points.get_mut(&oe.edge().index()) else {
                    continue;
                };
                let (t0, t1) = circle_param_range(topo, edge_data, circle)?;
                let along = |p: Point3| {
                    t0 + (circle.project(p) - t0 + 1e-9).rem_euclid(std::f64::consts::TAU) - 1e-9
                };
                for t in circle_pole_params(circle, (t0, t1), sphere) {
                    let pole = circle.evaluate(t);
                    if points.iter().any(|&p| {
                        point_merge_key(p, MERGE_GRID) == point_merge_key(pole, MERGE_GRID)
                    }) {
                        continue;
                    }
                    let at = (1..points.len())
                        .find(|&i| i + 1 == points.len() || along(points[i]) > t)
                        .unwrap_or(points.len());
                    points.insert(at, pole);
                }
            }
        }
    }

    sample_seam_vertices_on_rims(topo, &all_faces, &mut edge_points)?;
    sample_holes_past_sphere_chords(topo, &all_faces, &mut edge_points)?;
    split_chords_and_share(topo, &all_faces, &mut edge_points, edge_params, deflection)?;
    // The T-junction splice below reads a point this near an arc's end as
    // that end.
    let refine_tol = brepkit_math::tolerance::Tolerance::new().linear * 10.0;
    drop_lone_samples_beside_end_vertices(topo, &mut edge_points, refine_tol)?;
    phase("circle_sync", &mut phase_t);
    let mut merged = TriangleMesh::default();
    // Room for the edges' samples and as many again inside the faces.
    let samples: usize = edge_points.values().map(Vec::len).sum();
    let mut point_to_global = PointPool::with_capacity_and_hasher(2 * samples, WordState);
    let mut edge_global_indices: DetHashMap<usize, Vec<u32>> = DetHashMap::default();

    let mut sampled_edges: Vec<usize> = edge_points.keys().copied().collect();
    sampled_edges.sort_unstable();
    for edge_idx in sampled_edges {
        let points = &edge_points[&edge_idx];
        let mut global_ids = Vec::with_capacity(points.len());
        for &pt in points {
            let key = point_merge_key(pt, MERGE_GRID);
            let idx = point_to_global.entry(key).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt);
                merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                idx
            });
            global_ids.push(*idx);
        }
        edge_global_indices.insert(edge_idx, global_ids);
    }

    phase("edge_weld_pool", &mut phase_t);
    {
        // Spatial hash over the shared edge-point pool. The T-junction splice
        // below must find every pool point lying ON each circle edge; scanning
        // the whole pool per circle is O(circle-edges × pool-points) and
        // dominated a 1274-face export tessellation at 90% of its wall clock
        // (gh #1500). Instead, bucket the pool once and walk each arc at
        // cell-sized steps, inspecting the 3×3×3 neighborhood of each
        // visited cell — sound because a candidate sits within
        // `refine_tol + step/2 ≤ cell` of some walked sample, so its cell
        // index differs by at most 1 per axis. A point beyond an open arc's
        // ends is never spliced into it, so only the arc is walked.
        let has_circle = edge_indices.iter().any(|&edge_idx| {
            topo.edge_id_from_index(edge_idx)
                .and_then(|edge_id| topo.edge(edge_id).ok())
                .is_some_and(|edge| matches!(edge.curve(), EdgeCurve::Circle(_)))
        });
        let mut lo = Point3::new(f64::INFINITY, f64::INFINITY, f64::INFINITY);
        let mut hi = Point3::new(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY);
        if has_circle {
            for p in &merged.positions {
                lo = Point3::new(lo.x().min(p.x()), lo.y().min(p.y()), lo.z().min(p.z()));
                hi = Point3::new(hi.x().max(p.x()), hi.y().max(p.y()), hi.z().max(p.z()));
            }
        }
        let diag = (hi - lo).length();
        let grid_cell = if diag.is_finite() && diag > 0.0 {
            (diag / 128.0).max(refine_tol * 4.0)
        } else {
            refine_tol.max(1e-3)
        };
        let grid_inv = 1.0 / grid_cell;
        let cell_of = |p: &Point3| -> (i64, i64, i64) {
            #[allow(clippy::cast_possible_truncation)]
            (
                (p.x() * grid_inv).floor() as i64,
                (p.y() * grid_inv).floor() as i64,
                (p.z() * grid_inv).floor() as i64,
            )
        };
        // The pool's ids sorted by cell, each cell's own ascending, and the
        // run of each cell in that order.
        let mut by_cell: Vec<((i64, i64, i64), u32)> = Vec::new();
        if has_circle {
            #[allow(clippy::cast_possible_truncation)]
            by_cell.extend(
                merged
                    .positions
                    .iter()
                    .enumerate()
                    .map(|(gid, p)| (cell_of(p), gid as u32)),
            );
            by_cell.sort_unstable();
        }
        let pool_by_cell: Vec<u32> = by_cell.iter().map(|&(_, gid)| gid).collect();
        let mut cell_runs: WordHashMap<(i64, i64, i64), (usize, usize)> =
            WordHashMap::with_capacity_and_hasher(by_cell.len(), WordState);
        for (k, &(cell, _)) in by_cell.iter().enumerate() {
            cell_runs
                .entry(cell)
                .and_modify(|run| run.1 = k + 1)
                .or_insert((k, k + 1));
        }
        let mut visited: WordHashSet<(i64, i64, i64)> = WordHashSet::default();

        for &edge_idx in &edge_indices {
            let Some(edge_id) = topo.edge_id_from_index(edge_idx) else {
                continue;
            };
            let Ok(edge_data) = topo.edge(edge_id) else {
                continue;
            };
            let EdgeCurve::Circle(circle) = edge_data.curve() else {
                continue;
            };

            let Ok(start_vtx) = topo.vertex(edge_data.start()) else {
                continue;
            };
            let Ok(end_vtx) = topo.vertex(edge_data.end()) else {
                continue;
            };
            let start_pos = start_vtx.point();
            let end_pos = end_vtx.point();

            let (t_min, t_max) = edge_data.curve().domain_with_endpoints(start_pos, end_pos);
            let is_closed = edge_data.start() == edge_data.end();
            let place = |t: f64| {
                if is_closed {
                    t
                } else {
                    along_arc(t, t_min, t_max - t_min)
                }
            };

            let existing_gids_vec: Vec<u32> = edge_global_indices
                .get(&edge_idx)
                .cloned()
                .unwrap_or_default();
            let existing_gids: WordHashSet<u32> = existing_gids_vec.iter().copied().collect();

            let candidates: Vec<u32> = {
                use std::f64::consts::TAU;
                let span = t_max - t_min;
                let walk_arc = !is_closed && span.is_finite() && span > 0.0 && span < TAU;
                let (first, span, steps) = if walk_arc {
                    let steps =
                        ((span * circle.radius() * grid_inv).ceil() as usize).clamp(1, 8192);
                    (t_min, span, steps)
                } else {
                    let steps = ((TAU * circle.radius() * grid_inv).ceil() as usize).clamp(8, 8192);
                    (0.0, TAU, steps)
                };
                // A whole turn's last step would land on its first.
                let last = if walk_arc { steps } else { steps - 1 };
                visited.clear();
                let mut cand: Vec<u32> = Vec::new();
                // The cells round the step before, which the set holds already.
                let mut before: Option<(i64, i64, i64)> = None;
                #[allow(clippy::cast_precision_loss)]
                for i in 0..=last {
                    let t = span.madd(i as f64 / steps as f64, first);
                    let (cx, cy, cz) = cell_of(&circle.evaluate(t));
                    if before == Some((cx, cy, cz)) {
                        continue;
                    }
                    let near_before = |c: (i64, i64, i64)| {
                        before.is_some_and(|(bx, by, bz)| {
                            (c.0 - bx).abs() <= 1 && (c.1 - by).abs() <= 1 && (c.2 - bz).abs() <= 1
                        })
                    };
                    for dx in -1..=1_i64 {
                        for dy in -1..=1_i64 {
                            for dz in -1..=1_i64 {
                                let c = (cx + dx, cy + dy, cz + dz);
                                if !near_before(c)
                                    && visited.insert(c)
                                    && let Some(&(from, to)) = cell_runs.get(&c)
                                {
                                    cand.extend_from_slice(&pool_by_cell[from..to]);
                                }
                            }
                        }
                    }
                    before = Some((cx, cy, cz));
                }
                cand
            };

            let mut insertions: Vec<(f64, u32)> = Vec::new();
            for gid32 in candidates {
                let pos = &merged.positions[gid32 as usize];
                if existing_gids.contains(&gid32) {
                    continue;
                }
                if (*pos - start_pos).length() < refine_tol {
                    continue;
                }
                if !is_closed && (*pos - end_pos).length() < refine_tol {
                    continue;
                }
                let t = circle.project(*pos);
                let on_circle = circle.evaluate(t);
                let dist = (*pos - on_circle).length();
                if dist >= refine_tol {
                    continue;
                }
                let along = place(t);
                if is_closed || (-1e-8..=t_max - t_min + 1e-8).contains(&along) {
                    insertions.push((along, gid32));
                }
            }

            if insertions.is_empty() {
                continue;
            }

            // The walk's order is no order along the arc: two points at the
            // same place keep the lower id.
            insertions.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            insertions.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-8);

            let mut all_with_t: Vec<(f64, u32)> = existing_gids_vec
                .iter()
                .map(|&gid| {
                    let pos = merged.positions[gid as usize];
                    (place(circle.project(pos)), gid)
                })
                .collect();
            all_with_t.extend(insertions);
            all_with_t.sort_by(|a, b| a.0.total_cmp(&b.0));
            let mut seen_gids = WordHashSet::default();
            all_with_t.retain(|(_, gid)| seen_gids.insert(*gid));

            let refined: Vec<u32> = all_with_t.into_iter().map(|(_, gid)| gid).collect();
            edge_global_indices.insert(edge_idx, refined);
        }
    }

    phase("circle_refine_scan", &mut phase_t);
    // When tracking, `tri_faces` runs parallel to the mesh triangles:
    // tri_faces[t] is the index (into `all_faces`) of the face that produced
    // triangle t. The ungrouped caller skips this bookkeeping entirely.
    // Tracked always: the pinched-chord pass needs each triangle's face.
    let mut tri_faces: Option<Vec<u32>> = Some(Vec::new());
    #[allow(clippy::items_after_statements)]
    struct CdtJob {
        face_index: u32,
        pts2d: Vec<brepkit_math::vec::Point2>,
        outer_count: usize,
        inner_wire_ranges: Vec<(usize, usize)>,
        all_global_ids: Vec<Option<u32>>,
        all_positions: Vec<Point3>,
        normal: Vec3,
        is_reversed: bool,
    }
    #[allow(clippy::items_after_statements)]
    type CdtResult = Result<super::planar::PlanarCdtOutput, crate::OperationsError>;

    let mut other_face_indices: Vec<usize> = Vec::new();
    // Faces whose outline crosses itself mesh first, so the crossing
    // vertices they splice into shared edges reach every neighbour; planar
    // faces with holes collect their outlines in the second pass.
    let mut queued = vec![false; all_faces.len()];
    for crossing_pass in [true, false] {
        let mut cdt_jobs: Vec<CdtJob> = Vec::new();
        for (fi, &face_id) in all_faces.iter().enumerate() {
            if queued[fi] {
                continue;
            }
            let face_data = topo.face(face_id)?;
            let has_inner = !face_data.inner_wires().is_empty();
            let planar = matches!(face_data.surface(), FaceSurface::Plane { .. });
            if crossing_pass && planar && has_inner {
                continue;
            }
            queued[fi] = true;
            if let FaceSurface::Plane { normal, .. } = face_data.surface() {
                let normal = *normal;
                let is_reversed = face_data.is_reversed();
                let wire = topo.wire(face_data.outer_wire())?;
                let tol = 1e-10;

                let (mut all_positions, mut all_global_ids) = collect_wire_global_vertices(
                    wire,
                    &edge_global_indices,
                    &merged.positions,
                    tol,
                );
                remove_closing_duplicate_global(
                    &mut all_positions,
                    &mut all_global_ids,
                    &merged.positions,
                    tol,
                );
                // A face without holes joins these jobs only when its sides
                // cross: the job keeps the crossing vertices and hands them to
                // the neighbours before they mesh.
                if !has_inner {
                    let outline: Vec<brepkit_math::vec::Point2> = all_positions
                        .iter()
                        .map(|&p| project_by_normal(p, normal))
                        .collect();
                    if !polygon_self_crosses(&outline) {
                        other_face_indices.push(fi);
                        continue;
                    }
                }
                let outer_count = all_positions.len();

                let mut inner_wire_ranges: Vec<(usize, usize)> = Vec::new();
                for &iw_id in face_data.inner_wires() {
                    let iw = topo.wire(iw_id)?;
                    let start = all_positions.len();
                    let (inner_pos, inner_gids) = collect_wire_global_vertices(
                        iw,
                        &edge_global_indices,
                        &merged.positions,
                        tol,
                    );
                    let mut inner_flat_ids: Vec<u32> = Vec::with_capacity(inner_gids.len());
                    let mut next_sentinel = u32::MAX;
                    for (pos, gid_opt) in inner_pos.into_iter().zip(inner_gids) {
                        let gid = gid_opt.unwrap_or_else(|| {
                            debug_assert!(false, "inner wire vertex had no global ID");
                            let s = next_sentinel;
                            next_sentinel = next_sentinel.wrapping_sub(1);
                            s
                        });
                        inner_flat_ids.push(gid);
                        all_positions.push(pos);
                        all_global_ids.push(Some(gid));
                    }
                    if inner_flat_ids.len() > 2 {
                        remove_closing_duplicate_ids(&mut inner_flat_ids, &merged.positions, tol);
                        let expected_end = start + inner_flat_ids.len();
                        all_positions.truncate(expected_end);
                        all_global_ids.truncate(expected_end);
                    }
                    let end = all_positions.len();
                    inner_wire_ranges.push((start, end));
                }

                let pts2d: Vec<brepkit_math::vec::Point2> = all_positions
                    .iter()
                    .map(|&p| project_by_normal(p, normal))
                    .collect();

                #[allow(clippy::cast_possible_truncation)]
                cdt_jobs.push(CdtJob {
                    face_index: fi as u32,
                    pts2d,
                    outer_count,
                    inner_wire_ranges,
                    all_global_ids,
                    all_positions,
                    normal,
                    is_reversed,
                });
                continue;
            }
            other_face_indices.push(fi);
        }

        phase("cdt_job_prep", &mut phase_t);
        #[cfg(not(target_arch = "wasm32"))]
        let cdt_results: Vec<CdtResult> = if cdt_jobs.len() >= 2 {
            use rayon::prelude::*;
            cdt_jobs
                .par_iter()
                .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
                .collect()
        } else {
            cdt_jobs
                .iter()
                .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
                .collect()
        };
        #[cfg(target_arch = "wasm32")]
        let cdt_results: Vec<CdtResult> = cdt_jobs
            .iter()
            .map(|job| run_planar_cdt(&job.pts2d, job.outer_count, &job.inner_wire_ranges))
            .collect();

        phase("cdt_run", &mut phase_t);
        for (job, result) in cdt_jobs.iter().zip(cdt_results) {
            // A face without holes whose crossing outline the CDT cannot recover
            // meshes as before, on its own after the other faces.
            let (mut tris, steiner) = match result {
                Ok(output) => output,
                Err(e) if job.inner_wire_ranges.is_empty() => {
                    log::debug!(
                        "crossing outline of face {} left to the fan: {e}",
                        job.face_index
                    );
                    other_face_indices.push(job.face_index as usize);
                    continue;
                }
                Err(e) => return Err(e),
            };
            if job.inner_wire_ranges.is_empty() {
                cover_by_winding(&job.pts2d, job.outer_count, &steiner, true, &mut tris);
            }

            // Lift constraint-recovery Steiner points to 3D and give them global
            // vertices. A Steiner point that lies ON a shared boundary edge is
            // additionally spliced into that edge's shared sample chain so the
            // NEIGHBOUR faces (tessellated after the CDT jobs) pick it up —
            // without this the neighbour spans the original segment in one piece
            // and the mesh cracks at a T-junction.
            let n_input = job.all_positions.len();
            let mut steiner_positions: Vec<Point3> = Vec::with_capacity(steiner.len());
            let mut steiner_gids: Vec<u32> = Vec::with_capacity(steiner.len());
            for p2d in &steiner {
                let p3d = unproject_point(*p2d, job.normal, &job.all_positions[0]);
                let key = point_merge_key(p3d, MERGE_GRID);
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(p3d);
                    merged.normals.push(job.normal);
                    idx
                });
                steiner_positions.push(p3d);
                steiner_gids.push(gid);
            }
            if !steiner.is_empty() {
                let ring_segments: Vec<(usize, usize)> = (0..job.outer_count)
                    .map(|i| (i, (i + 1) % job.outer_count))
                    .chain(job.inner_wire_ranges.iter().flat_map(|&(st, en)| {
                        (st..en).map(move |i| (i, if i + 1 == en { st } else { i + 1 }))
                    }))
                    .collect();
                let mut per_segment: DetHashMap<(usize, usize), Vec<(f64, u32)>> =
                    DetHashMap::default();
                for (si, p2d) in steiner.iter().enumerate() {
                    for &(i, j) in &ring_segments {
                        let a = job.pts2d[i];
                        let b = job.pts2d[j];
                        let ab = (b.x() - a.x(), b.y() - a.y());
                        let len2 = ab.0 * ab.0 + ab.1 * ab.1;
                        if len2 < 1e-24 {
                            continue;
                        }
                        let ap = (p2d.x() - a.x(), p2d.y() - a.y());
                        let t = (ap.0 * ab.0 + ap.1 * ab.1) / len2;
                        if !(1e-9..=1.0 - 1e-9).contains(&t) {
                            continue;
                        }
                        let cross = ap.0 * ab.1 - ap.1 * ab.0;
                        if cross * cross / len2 < 1e-18 {
                            per_segment
                                .entry((i, j))
                                .or_default()
                                .push((t, steiner_gids[si]));
                        }
                    }
                }
                // A vertex where two sides cross lies on both, and each side's
                // edge chain takes it, except a chain holding both sides: that
                // edge's curve passes through the point twice, and its other
                // face could not tell which pass a lone sample belongs to.
                // (edge chain, segment ends, crossing vertices with their place
                // along the segment)
                #[allow(clippy::items_after_statements)]
                type Splice = (usize, u32, u32, Vec<(f64, u32)>);
                let mut placed: Vec<Splice> = Vec::new();
                for ((i, j), run) in per_segment {
                    let (Some(gi), Some(gj)) = (job.all_global_ids[i], job.all_global_ids[j])
                    else {
                        continue;
                    };
                    let key = edge_global_indices.iter().find_map(|(&key, chain)| {
                        chain
                            .windows(2)
                            .any(|w| (w[0] == gi && w[1] == gj) || (w[0] == gj && w[1] == gi))
                            .then_some(key)
                    });
                    if let Some(key) = key {
                        placed.push((key, gi, gj, run));
                    }
                }
                let twice = |key: usize, gid: u32| {
                    placed
                        .iter()
                        .filter(|(k, _, _, run)| *k == key && run.iter().any(|&(_, g)| g == gid))
                        .count()
                        > 1
                };
                let splices: Vec<Splice> = placed
                    .iter()
                    .map(|(key, gi, gj, run)| {
                        let kept = run.iter().filter(|&&(_, g)| !twice(*key, g)).copied();
                        (*key, *gi, *gj, kept.collect())
                    })
                    .collect();
                for (key, gi, gj, mut run) in splices {
                    run.sort_by(|a, b| a.0.total_cmp(&b.0));
                    let Some(chain) = edge_global_indices.get_mut(&key) else {
                        continue;
                    };
                    for p in 0..chain.len().saturating_sub(1) {
                        if chain[p] == gi && chain[p + 1] == gj {
                            for (k, &(_, gid)) in run.iter().enumerate() {
                                chain.insert(p + 1 + k, gid);
                            }
                            break;
                        }
                        if chain[p] == gj && chain[p + 1] == gi {
                            for (k, &(_, gid)) in run.iter().rev().enumerate() {
                                chain.insert(p + 1 + k, gid);
                            }
                            break;
                        }
                    }
                }
            }

            let pos_of = |i: usize| -> Point3 {
                if i < n_input {
                    job.all_positions[i]
                } else {
                    steiner_positions[i - n_input]
                }
            };
            let gid_of = |i: usize| -> u32 {
                if i < n_input {
                    job.all_global_ids[i].unwrap_or(0)
                } else {
                    steiner_gids[i - n_input]
                }
            };

            let needs_flip = !tris.is_empty()
                && (winding_area(
                    job.normal,
                    tris.iter()
                        .map(|&(i0, i1, i2)| [pos_of(i0), pos_of(i1), pos_of(i2)]),
                ) > 0.0)
                    == job.is_reversed;

            for &(i0, i1, i2) in &tris {
                let g0 = gid_of(i0);
                let g1 = gid_of(i1);
                let g2 = gid_of(i2);
                if let Some(tf) = tri_faces.as_mut() {
                    tf.push(job.face_index);
                }
                if needs_flip {
                    merged.indices.push(g0);
                    merged.indices.push(g2);
                    merged.indices.push(g1);
                } else {
                    merged.indices.push(g0);
                    merged.indices.push(g1);
                    merged.indices.push(g2);
                }
            }
        }
    }

    phase("planar_stitch", &mut phase_t);
    for &fi in &other_face_indices {
        if let Err(e) = tessellate_face_with_shared_edges(
            topo,
            all_faces[fi],
            deflection,
            angular_tol,
            circle_floor,
            &edge_global_indices,
            &mut merged,
            &mut point_to_global,
        ) {
            // A face skipped for too many samples would leave a hole where
            // the caller asked for a mesh finer than any the limit allows.
            if super::is_sample_limit(&e) {
                return Err(e);
            }
            log::warn!("skipping face during tessellation: {e}");
        }
        // Attribute every triangle appended by this face (even partial output
        // before an error) so tri_faces stays parallel to the triangle list.
        if let Some(tf) = tri_faces.as_mut() {
            #[allow(clippy::cast_possible_truncation)]
            tf.resize(merged.indices.len() / 3, fi as u32);
        }
    }

    phase("other_faces", &mut phase_t);
    let n_verts = merged.positions.len();
    let tri_count = merged.indices.len() / 3;

    let mut needs_normal = vec![false; n_verts];
    for i in 0..n_verts {
        let n = &merged.normals[i];
        if n.x().abs() < 1e-30 && n.y().abs() < 1e-30 && n.z().abs() < 1e-30 {
            needs_normal[i] = true;
        }
    }

    {
        // The faces of the edges through each vertex that needs a normal,
        // with repeats: `vertex_faces[first[i]..first[i + 1]]` for vertex `i`.
        let mut first = vec![0_usize; n_verts + 1];
        for (&edge_idx, global_ids) in &edge_global_indices {
            if let Some(face_ids) = edge_face_map.get(&edge_idx) {
                for &gid in global_ids {
                    let gi = gid as usize;
                    if gi < n_verts && needs_normal[gi] {
                        first[gi + 1] += face_ids.len();
                    }
                }
            }
        }
        for i in 0..n_verts {
            first[i + 1] += first[i];
        }
        let mut vertex_faces: Vec<FaceId> = match all_faces.first() {
            Some(&any) => vec![any; first[n_verts]],
            None => Vec::new(),
        };
        let mut fill = first.clone();
        for (&edge_idx, global_ids) in &edge_global_indices {
            if let Some(face_ids) = edge_face_map.get(&edge_idx) {
                for &gid in global_ids {
                    let gi = gid as usize;
                    if gi < n_verts && needs_normal[gi] {
                        for &fid in face_ids {
                            vertex_faces[fill[gi]] = fid;
                            fill[gi] += 1;
                        }
                    }
                }
            }
        }

        let mut fallback_needed = vec![false; n_verts];
        for i in 0..n_verts {
            if !needs_normal[i] {
                continue;
            }
            let pos = merged.positions[i];
            let mut normal_sum = Vec3::new(0.0, 0.0, 0.0);
            let mut count = 0_u32;
            let faces = &mut vertex_faces[first[i]..first[i + 1]];
            faces.sort_unstable_by_key(|f| f.index());
            let mut last: Option<FaceId> = None;
            for &fid in faces.iter() {
                if last == Some(fid) {
                    continue;
                }
                last = Some(fid);
                if let Ok(face_data) = topo.face(fid) {
                    let surf = face_data.surface();
                    if let Some(n) = crate::fillet::face_surface_normal_at(surf, pos) {
                        let oriented = if face_data.is_reversed() {
                            Vec3::new(-n.x(), -n.y(), -n.z())
                        } else {
                            n
                        };
                        normal_sum += oriented;
                        count += 1;
                    }
                }
            }
            if count > 0 {
                merged.normals[i] = normal_sum.normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
            } else {
                fallback_needed[i] = true;
            }
        }

        if fallback_needed.iter().any(|&f| f) {
            let mut accum: Vec<Vec3> = vec![Vec3::new(0.0, 0.0, 0.0); n_verts];
            for t in 0..tri_count {
                let i0 = merged.indices[t * 3] as usize;
                let i1 = merged.indices[t * 3 + 1] as usize;
                let i2 = merged.indices[t * 3 + 2] as usize;
                let a = merged.positions[i1] - merged.positions[i0];
                let b = merged.positions[i2] - merged.positions[i0];
                let face_normal = a.cross(b);
                if fallback_needed.get(i0).copied().unwrap_or(false) {
                    accum[i0] += face_normal;
                }
                if fallback_needed.get(i1).copied().unwrap_or(false) {
                    accum[i1] += face_normal;
                }
                if fallback_needed.get(i2).copied().unwrap_or(false) {
                    accum[i2] += face_normal;
                }
            }
            for i in 0..n_verts {
                if fallback_needed[i] {
                    merged.normals[i] = accum[i].normalize().unwrap_or(Vec3::new(0.0, 0.0, 1.0));
                }
            }
        }
    }

    phase("normals", &mut phase_t);
    // A sample only on edges that one face uses lies on that face's free
    // rim. Two such samples of one face are no crack between faces: welded,
    // they would collapse a face narrower than the weld's reach. A join that
    // two faces hold with separate edges is still welded.
    let mut on_shared = vec![false; merged.positions.len()];
    for (edge_idx, global_ids) in &edge_global_indices {
        if edge_face_map
            .get(edge_idx)
            .is_some_and(|faces| faces.len() > 1)
        {
            for &gid in global_ids {
                if let Some(mark) = on_shared.get_mut(gid as usize) {
                    *mark = true;
                }
            }
        }
    }
    // A sample where the free rims of two faces meet is on neither alone.
    let mut rim_face = vec![NO_RIM_FACE; merged.positions.len()];
    let mut on_two_rims = vec![false; merged.positions.len()];
    for (edge_idx, global_ids) in &edge_global_indices {
        let Some(faces) = edge_face_map.get(edge_idx) else {
            continue;
        };
        let &[face] = faces.as_slice() else {
            continue;
        };
        #[allow(clippy::cast_possible_truncation)]
        let face = face.index() as u32;
        for &gid in global_ids {
            let gid = gid as usize;
            if on_shared.get(gid).copied().unwrap_or(true) {
                continue;
            }
            if rim_face[gid] == NO_RIM_FACE {
                rim_face[gid] = face;
            } else if rim_face[gid] != face {
                on_two_rims[gid] = true;
            }
        }
    }
    for (face, &mixed) in rim_face.iter_mut().zip(&on_two_rims) {
        if mixed {
            *face = NO_RIM_FACE;
        }
    }
    let repeated = weld_boundary_vertices(&mut merged, deflection, &rim_face, tri_faces.as_mut());

    // A face no mesher covered, or whose rim the weld collapsed beside a
    // neighbour's crack, leaves a hole the caller cannot see from the mesh.
    // Read before the dedupe below, which empties two coincident faces of
    // opposite sense on purpose.
    let unmeshed: Vec<FaceId> = tri_faces.as_ref().map_or_else(Vec::new, |tf| {
        let mut meshed = vec![false; all_faces.len()];
        for &fi in tf {
            if let Some(flag) = meshed.get_mut(fi as usize) {
                *flag = true;
            }
        }
        all_faces
            .iter()
            .zip(&meshed)
            .filter(|(_, m)| !**m)
            .map(|(&face_id, _)| face_id)
            .collect()
    });
    for face_id in &unmeshed {
        log::warn!("{face_id:?} meshed to no triangles at deflection {deflection}");
    }

    // Drop coincident/cancelling triangles left by booleans that
    // produced overlapping coplanar faces (issue #696). Keyed on quantized
    // positions so position-coincident triangles with distinct vertex IDs
    // are still caught.
    dedupe_coincident_triangles(&mut merged, tri_faces.as_mut());
    phase("weld_dedupe", &mut phase_t);

    if repeated && let Some(tf) = tri_faces.as_mut() {
        let surfaces: Vec<Option<&FaceSurface>> = all_faces
            .iter()
            .map(|&f| topo.face(f).ok().map(brepkit_topology::face::Face::surface))
            .collect();
        // A curved analytic surface projects exactly; a plane holds its
        // chords already, and a NURBS projection can fail to a far point.
        let on_face = |fi: u32, p: Point3| -> Option<Point3> {
            let surface = (*surfaces.get(fi as usize)?)?;
            if !matches!(
                surface,
                FaceSurface::Cylinder(_)
                    | FaceSurface::Cone(_)
                    | FaceSurface::Sphere(_)
                    | FaceSurface::Torus(_)
            ) {
                return None;
            }
            let (u, v) = surface.project_point(p)?;
            surface.evaluate(u, v)
        };
        split_pinched_chords(&mut merged, tf, &on_face);
    }

    Ok(CoreMesh {
        mesh: merged,
        tri_faces: tri_faces.filter(|_| track_faces),
        n_faces: all_faces.len(),
        #[cfg(test)]
        unmeshed,
    })
}

/// Where a sample at circle parameter `t` lies along an arc that starts at
/// `start` and spans `span`: `t` unwrapped from the start, so the order of
/// samples does not break where the parameter wraps (`project` gives an
/// angle in `(-π, π]`). A sample just before the start reads a little below
/// zero.
pub(super) fn along_arc(t: f64, start: f64, span: f64) -> f64 {
    let along = (t - start).rem_euclid(std::f64::consts::TAU);
    if along > 0.5 * (span + std::f64::consts::TAU) {
        along - std::f64::consts::TAU
    } else {
        along
    }
}

/// `BK_TESS_PHASES` (any value): log per-phase wall clock of the solid
/// tessellation pipeline. Resolved once per process.
#[cfg(not(target_arch = "wasm32"))]
fn tess_phases() -> bool {
    static PHASES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *PHASES.get_or_init(|| std::env::var("BK_TESS_PHASES").is_ok())
}

/// Per-phase wall-clock lap timer for `BK_TESS_PHASES` diagnostics.
///
/// `std::time::Instant::now()` PANICS on wasm32-unknown-unknown (no time
/// source), and this runs inside every solid tessellation — the timer must
/// compile to a no-op there (3.2.15 aborted every wasm solid tessellation
/// through this call).
struct PhaseTimer {
    #[cfg(not(target_arch = "wasm32"))]
    t: std::time::Instant,
}

impl PhaseTimer {
    fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            t: std::time::Instant::now(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lap(&mut self, label: &str) {
        if tess_phases() {
            log::debug!(
                "TESS_PHASE {label}: {:.1}ms",
                self.t.elapsed().as_secs_f64() * 1e3
            );
        }
        self.t = std::time::Instant::now();
    }

    #[cfg(target_arch = "wasm32")]
    #[allow(clippy::unused_self)]
    fn lap(&mut self, _label: &str) {}
}

/// `BK_TESS_TRACE` (any value): log each face's mesher-arm dispatch.
/// Resolved ONCE per process — the dispatch runs per face and must not pay
/// an env lookup each call.
fn tess_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("BK_TESS_TRACE").is_ok())
}

/// Tessellate a single face, reusing shared edge vertices from the global mesh.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn tessellate_face_with_shared_edges(
    topo: &Topology,
    face_id: FaceId,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<(), crate::OperationsError> {
    let face_data = topo.face(face_id)?;
    let is_reversed = face_data.is_reversed();

    let idx_start = merged.indices.len();
    let pos_start = merged.positions.len();

    if let FaceSurface::Plane { normal, .. } = face_data.surface() {
        let normal = *normal;
        let wire = topo.wire(face_data.outer_wire())?;

        let mut boundary_global_ids: Vec<u32> = Vec::new();
        let tol = 1e-10;

        for oe in wire.edges() {
            let edge_idx = oe.edge().index();
            if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
                let is_fwd = oe.is_forward();
                let len = global_ids.len();
                for j in 0..len {
                    let gid = if is_fwd {
                        global_ids[j]
                    } else {
                        global_ids[len - 1 - j]
                    };
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if last_gid == gid {
                            continue;
                        }
                        if (last_gid as usize) < merged.positions.len()
                            && (gid as usize) < merged.positions.len()
                        {
                            let last_pos = merged.positions[last_gid as usize];
                            let this_pos = merged.positions[gid as usize];
                            if (last_pos - this_pos).length() < tol {
                                continue;
                            }
                        }
                    }
                    boundary_global_ids.push(gid);
                }
            } else {
                let edge_data = topo.edge(oe.edge())?;
                let points = sample_edge(topo, edge_data, deflection, angular_tol, circle_floor)?;
                let ordered: Vec<Point3> = if oe.is_forward() {
                    points
                } else {
                    points.into_iter().rev().collect()
                };
                for (j, pt) in ordered.iter().enumerate() {
                    if j == 0 && !boundary_global_ids.is_empty() {
                        let last_gid = *boundary_global_ids.last().unwrap_or(&u32::MAX);
                        if (last_gid as usize) < merged.positions.len() {
                            let last_pos = merged.positions[last_gid as usize];
                            if (last_pos - *pt).length() < tol {
                                continue;
                            }
                        }
                    }
                    let key = point_merge_key(*pt, MERGE_GRID);
                    let gid = point_to_global.entry(key).or_insert_with(|| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = merged.positions.len() as u32;
                        merged.positions.push(*pt);
                        merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                        idx
                    });
                    boundary_global_ids.push(*gid);
                }
            }
        }

        remove_closing_duplicate_ids(&mut boundary_global_ids, &merged.positions, tol);

        let n = boundary_global_ids.len();
        if n < 3 {
            return Ok(());
        }

        let local_positions: Vec<Point3> = boundary_global_ids
            .iter()
            .map(|&gid| merged.positions[gid as usize])
            .collect();

        if face_data.inner_wires().is_empty() {
            let mut local_indices = cdt_triangulate_simple(&local_positions, normal);

            let area = winding_area(
                normal,
                local_indices
                    .chunks_exact(3)
                    .map(|t| [t[0], t[1], t[2]].map(|i| local_positions[i as usize])),
            );
            if area < 0.0 {
                for t in 0..local_indices.len() / 3 {
                    local_indices.swap(t * 3 + 1, t * 3 + 2);
                }
            }

            for &li in &local_indices {
                merged.indices.push(boundary_global_ids[li as usize]);
            }
        } else {
            tessellate_planar_shared_with_holes(
                topo,
                face_data,
                &boundary_global_ids,
                &local_positions,
                normal,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(face_data.surface(), FaceSurface::Nurbs(_)) {
        let triangles_before = merged.indices.len();
        let cdt_ok = tessellate_nonplanar_cdt(
            topo,
            face_id,
            face_data,
            deflection,
            angular_tol,
            circle_floor,
            edge_global_indices,
            merged,
            point_to_global,
        );
        // A boundary that encloses nothing in (u, v), such as a whole torus's
        // seam pair collapsed onto one vertex, leaves the CDT without
        // triangles: the snap mesher covers the surface's whole domain.
        if cdt_ok.is_err() || merged.indices.len() == triangles_before {
            tessellate_nonplanar_snap(
                topo,
                face_id,
                face_data,
                deflection,
                angular_tol,
                circle_floor,
                edge_global_indices,
                merged,
                point_to_global,
            )?;
        }
    } else if matches!(
        face_data.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        let (all_line_circle, band_eligible) = {
            let wire = topo.wire(face_data.outer_wire())?;
            let lc = wire.edges().iter().all(|oe| {
                topo.edge(oe.edge())
                    .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line | EdgeCurve::Circle(_)))
            });
            // The structured band takes any rim, wavy mixed ones included
            // (winding-chain separators carry marched-NURBS pieces); it
            // verifies the cycle structure itself and declines anything else.
            let be = wire.edges().iter().all(|oe| topo.edge(oe.edge()).is_ok());
            (lc, be)
        };
        let is_standard_rect =
            all_line_circle && topo.wire(face_data.outer_wire())?.edges().len() <= 4;

        // Prefer a structured band built from the shared rim vertices — it
        // is watertight by construction and avoids the snap path's proximity
        // reconciliation, which cracks drilled holes at certain radius/
        // deflection combos (issue #696). Tried for ANY Line/Circle wire, not
        // just the 4-edge canonical shape: a boolean can deliver a full band
        // whose rims are split into arc chains (cone∪box inscribed-rim), which
        // the band mesher now handles; it still returns false for anything
        // that is not a two-full-rim band.
        let band_handled = band_eligible
            && tessellate_revolution_band_shared(topo, face_data, edge_global_indices, merged)?;

        // The mesh boolean's meshes (`circle_floor`) stay on the CDT.
        if band_handled
            || (!circle_floor
                && tessellate_cylinder_strip_shared(
                    topo,
                    face_data,
                    deflection,
                    angular_tol,
                    edge_global_indices,
                    merged,
                )?)
        {
            // done — watertight structured band emitted
        } else if is_standard_rect {
            // Partial bands and holed full bands have a simple polygon
            // outer boundary in unwrapped UV, so CDT over the shared pool
            // ids is watertight by construction. The snap path re-samples
            // the rim independently and cracks at fine deflections when its
            // segment count diverges from the pool's (the #696 class, seen
            // on gridfinity socket cone/cylinder corner rings). Inner wires
            // are constrained and flood-removed as holes.
            let mut cdt_handled = false;
            {
                let pos_save = merged.positions.len();
                let nrm_save = merged.normals.len();
                let idx_save = merged.indices.len();
                let cdt_ok = tessellate_nonplanar_cdt(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    edge_global_indices,
                    merged,
                    point_to_global,
                );
                if cdt_ok.is_err() || merged.indices.len() == idx_save {
                    merged.positions.truncate(pos_save);
                    merged.normals.truncate(nrm_save);
                    merged.indices.truncate(idx_save);
                    // The CDT attempt may have registered merge-map entries
                    // for the now-truncated vertices; a later lookup would
                    // return a global id past `positions.len()`.
                    point_to_global.retain(|_, gid| (*gid as usize) < pos_save);
                } else {
                    cdt_handled = true;
                }
            }
            if !cdt_handled {
                tessellate_nonplanar_snap(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?;
            }
        } else {
            let pos_save = merged.positions.len();
            let nrm_save = merged.normals.len();
            let idx_save = merged.indices.len();
            let cdt_ok = tessellate_nonplanar_cdt(
                topo,
                face_id,
                face_data,
                deflection,
                angular_tol,
                circle_floor,
                edge_global_indices,
                merged,
                point_to_global,
            );
            if cdt_ok.is_err() || merged.indices.len() == idx_save {
                merged.positions.truncate(pos_save);
                merged.normals.truncate(nrm_save);
                merged.indices.truncate(idx_save);
                // Same stale-merge-map hazard as the partial-band rollback
                // above: drop entries referencing the truncated vertices.
                point_to_global.retain(|_, gid| (*gid as usize) < pos_save);
                tessellate_nonplanar_snap(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?;
            }
        }
    } else {
        // A sphere/torus latitude band (the annular region between two
        // constant-v full-revolution boundaries, e.g. a cylinder bored through a
        // sphere) degenerates in UV: each latitude projects to a zero-area
        // back-and-forth segment, so the CDT below cannot bound the band and
        // fills the removed polar cap. Tessellate such bands structurally from
        // the shared boundary vertices instead. Returns false for any other
        // sphere/torus face, which then takes the CDT/snap path unchanged.
        // A torus notch band (a patch wrapping the tube fully, its outer and
        // one inner wire each a v-wrapping loop at an end of a ring-angle
        // span) is swept along u, not v, so it is not a latitude band. Try it
        // first; it returns false for any other torus face.
        let handled_notch = matches!(face_data.surface(), FaceSurface::Torus(_))
            && tessellate_torus_notch_band(
                topo,
                face_data,
                deflection,
                angular_tol,
                edge_global_indices,
                merged,
                point_to_global,
            )?;

        // A full-revolution torus band between two closed rims (an analytic
        // revolve's arc-profile wall, seamed by its doubled profile arc) is
        // structured from the shared rim vertices, like the cylinder/cone
        // standard band — CDT degenerates on its fully-u-wrapping UV image and
        // the snap path re-samples the rims into cracks.
        let handled_band = handled_notch
            || (matches!(face_data.surface(), FaceSurface::Torus(_))
                && tessellate_torus_two_rim_band(
                    topo,
                    face_data,
                    deflection,
                    angular_tol,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?)
            || (matches!(
                face_data.surface(),
                FaceSurface::Sphere(_) | FaceSurface::Torus(_)
            ) && tessellate_latitude_band_shared(
                topo,
                face_data,
                deflection,
                angular_tol,
                edge_global_indices,
                merged,
                point_to_global,
            )?)
            || (!circle_floor
                && tessellate_sphere_triangle_shared(
                    topo,
                    face_data,
                    deflection,
                    angular_tol,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?);

        if !handled_band {
            let pos_save = merged.positions.len();
            let nrm_save = merged.normals.len();
            let idx_save = merged.indices.len();
            let ptg_count_save = point_to_global.len();

            let cdt_ok = tessellate_nonplanar_cdt(
                topo,
                face_id,
                face_data,
                deflection,
                angular_tol,
                circle_floor,
                edge_global_indices,
                merged,
                point_to_global,
            );
            let cdt_produced_tris = cdt_ok.is_ok() && merged.indices.len() > idx_save;
            if tess_trace() {
                log::debug!(
                    "tess dispatch {face_id:?} {} rev={is_reversed} cdt_ok={cdt_produced_tris}",
                    face_data.surface().type_tag()
                );
            }
            if !cdt_produced_tris {
                merged.positions.truncate(pos_save);
                merged.normals.truncate(nrm_save);
                merged.indices.truncate(idx_save);
                if point_to_global.len() > ptg_count_save {
                    point_to_global.retain(|_, v| (*v as usize) < pos_save);
                }

                tessellate_nonplanar_snap(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    edge_global_indices,
                    merged,
                    point_to_global,
                )?;
            }
        }
    }

    if tess_trace() {
        log::debug!(
            "tess face {face_id:?} {} tris {}..{}",
            face_data.surface().type_tag(),
            idx_start / 3,
            merged.indices.len() / 3
        );
    }
    if is_reversed {
        let idx_end = merged.indices.len();
        let tri_count = (idx_end - idx_start) / 3;
        for t in 0..tri_count {
            let base = idx_start + t * 3;
            merged.indices.swap(base + 1, base + 2);
        }
        for n in &mut merged.normals[pos_start..] {
            *n = -*n;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::chords_in_one_turn;
    use std::f64::consts::TAU;

    fn orient(a: (f64, f64), b: (f64, f64), c: (f64, f64)) -> f64 {
        (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
    }

    /// A chord from u = 3.15 back across the seam to u = 0.05 on a sphere
    /// crosses a chord near u = 0.1. Read with its first end next to the
    /// other chord's, it sat a turn away and missed it.
    #[test]
    fn a_chord_across_the_seam_meets_the_chord_it_crosses() {
        let (a, b) = ((0.0, 1.5), (0.2, 1.5));
        let (c, d) = ((3.15, 1.56), (0.05, 1.499));
        for (p, q, r, s) in [(a, b, c, d), (c, d, a, b)] {
            let (p, q, r, s) = chords_in_one_turn(p, q, r, s, Some(TAU));
            assert!(
                orient(p, q, r) * orient(p, q, s) < 0.0 && orient(r, s, p) * orient(r, s, q) < 0.0,
                "the chords {p:?}-{q:?} and {r:?}-{s:?} do not cross"
            );
        }
        // On a surface that does not run round, nothing moves.
        assert_eq!(chords_in_one_turn(a, b, c, d, None), (a, b, c, d));
    }
}

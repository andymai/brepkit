//! Analytic sweep along a closed planar chain of line and circular-arc edges.
//!
//! The gridfinity stacking-lip class: a closed spine of straight runs joined
//! by tangent arcs (a rounded rectangle), swept by a planar all-line profile
//! perpendicular to the spine. Each straight run extrudes the profile (planar
//! quads); each arc corner revolves it about the arc axis (plane / cylinder /
//! cone bands via the revolve machinery). The result is one exact analytic
//! face per profile edge per spine segment — no fitted-NURBS faceting.
//!
//! Falls back (returns `Ok(None)`) whenever the configuration is outside the
//! gated class; the caller then uses the fitted-path sweep.

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::dot_normal_point;
use crate::revolve::{make_arc_curve, revolution_band_surface, rotate_point};

/// Angular alignment band for the gates below (unit-vector dot products).
const ALIGN_TOL: f64 = 1e-6;

/// One segment of the spine chain, oriented along the traversal.
enum SpineSeg {
    Line {
        dir: Vec3,
        len: f64,
    },
    Arc {
        center: Point3,
        axis: Vec3,
        angle: f64,
    },
}

impl SpineSeg {
    fn transport(&self, p: Point3) -> Point3 {
        match self {
            Self::Line { dir, len } => p + *dir * *len,
            Self::Arc {
                center,
                axis,
                angle,
            } => rotate_point(p, *center, *axis, *angle),
        }
    }

    fn start_tangent(&self, start: Point3) -> Vec3 {
        match self {
            Self::Line { dir, .. } => *dir,
            Self::Arc { center, axis, .. } => {
                let radial = start - *center;
                axis.cross(radial - *axis * axis.dot(radial))
                    .normalize()
                    .unwrap_or(*axis)
            }
        }
    }

    fn end_tangent(&self, end: Point3) -> Vec3 {
        match self {
            Self::Line { dir, .. } => *dir,
            Self::Arc { .. } => self.start_tangent(end),
        }
    }
}

/// Extract a closed, planar, G1 line/arc chain from the edges, orienting each
/// edge to continue the traversal. Returns the segments plus the chain start
/// point and start tangent, or `None` when any gate fails.
fn extract_closed_planar_spine(
    topo: &Topology,
    edges: &[EdgeId],
    lin: f64,
) -> Option<(Vec<SpineSeg>, Point3, Vec3)> {
    struct RawSeg {
        start: Point3,
        end: Point3,
        seg: SpineSeg,
    }

    if edges.len() < 2 {
        return None;
    }

    let mut raw: Vec<(Point3, Point3, EdgeCurve)> = Vec::with_capacity(edges.len());
    for &eid in edges {
        let edge = topo.edge(eid).ok()?;
        let sp = topo.vertex(edge.start()).ok()?.point();
        let ep = topo.vertex(edge.end()).ok()?.point();
        raw.push((sp, ep, edge.curve().clone()));
    }

    let make_seg = |sp: Point3, ep: Point3, curve: &EdgeCurve, flipped: bool| -> Option<SpineSeg> {
        match curve {
            EdgeCurve::Line => {
                let (from, to) = if flipped { (ep, sp) } else { (sp, ep) };
                let chord = to - from;
                let len = chord.length();
                if len < lin {
                    return None;
                }
                Some(SpineSeg::Line {
                    dir: chord.normalize().ok()?,
                    len,
                })
            }
            EdgeCurve::Circle(c) => {
                // A stored circle edge travels CCW about its normal from the
                // stored start to the stored end; a flipped traversal is CCW
                // about the negated axis.
                let (from, to) = if flipped { (ep, sp) } else { (sp, ep) };
                let t0 = c.project(from);
                let mut t1 = c.project(to);
                if t1 <= t0 + lin {
                    t1 += std::f64::consts::TAU;
                }
                let angle = t1 - t0;
                if !(1e-9..=std::f64::consts::FRAC_PI_2 + 1e-9).contains(&angle) {
                    return None;
                }
                let axis = if flipped { -c.normal() } else { c.normal() };
                Some(SpineSeg::Arc {
                    center: c.center(),
                    axis,
                    angle,
                })
            }
            EdgeCurve::NurbsCurve(_) | EdgeCurve::Ellipse(_) => None,
        }
    };

    // Chain the edges in order, flipping individual edges as needed.
    let mut segs: Vec<RawSeg> = Vec::with_capacity(raw.len());
    let (first_sp, first_ep, first_curve) = &raw[0];
    segs.push(RawSeg {
        start: *first_sp,
        end: *first_ep,
        seg: make_seg(*first_sp, *first_ep, first_curve, false)?,
    });
    for (sp, ep, curve) in raw.iter().skip(1) {
        let current = segs.last()?.end;
        let (start, end, flipped) = if (*sp - current).length() < lin * 100.0 {
            (*sp, *ep, false)
        } else if (*ep - current).length() < lin * 100.0 {
            (*ep, *sp, true)
        } else {
            return None; // not a contiguous chain
        };
        segs.push(RawSeg {
            start,
            end,
            seg: make_seg(*sp, *ep, curve, flipped)?,
        });
    }

    // Closed chain.
    if (segs.last()?.end - segs[0].start).length() > lin * 100.0 {
        return None;
    }

    // Planarity: all arc axes parallel to one normal, all line dirs
    // perpendicular to it.
    let plane_normal = segs.iter().find_map(|r| match &r.seg {
        SpineSeg::Arc { axis, .. } => Some(*axis),
        SpineSeg::Line { .. } => None,
    })?;
    for r in &segs {
        match &r.seg {
            SpineSeg::Arc { axis, .. } => {
                if axis.cross(plane_normal).length() > ALIGN_TOL {
                    return None;
                }
            }
            SpineSeg::Line { dir, .. } => {
                if dir.dot(plane_normal).abs() > ALIGN_TOL {
                    return None;
                }
            }
        }
    }

    // G1 continuity at every junction (including the closing one).
    for i in 0..segs.len() {
        let j = (i + 1) % segs.len();
        let t_out = segs[i].seg.end_tangent(segs[i].end);
        let t_in = segs[j].seg.start_tangent(segs[j].start);
        if t_out.dot(t_in) < 1.0 - ALIGN_TOL {
            return None;
        }
    }

    let start = segs[0].start;
    let tangent = segs[0].seg.start_tangent(start);
    Some((segs.into_iter().map(|r| r.seg).collect(), start, tangent))
}

/// Try the analytic spine sweep. Returns `Ok(None)` when the configuration is
/// outside the gated class (the caller falls back to the fitted-path sweep).
#[allow(clippy::too_many_lines)]
pub fn try_analytic_spine_sweep(
    topo: &mut Topology,
    profile: FaceId,
    edges: &[EdgeId],
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();

    let Some((segs, spine_start, start_tangent)) =
        extract_closed_planar_spine(topo, edges, tol.linear)
    else {
        return Ok(None);
    };

    // Profile gates: planar face, no inner wires, all-line closed outer wire,
    // plane perpendicular to the spine start tangent.
    let face_data = topo.face(profile)?;
    let FaceSurface::Plane {
        normal: profile_normal,
        ..
    } = *face_data.surface()
    else {
        return Ok(None);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(None);
    }
    if profile_normal.dot(start_tangent).abs() < 1.0 - ALIGN_TOL {
        return Ok(None);
    }
    let outer_wire_id = face_data.outer_wire();
    let wire = topo.wire(outer_wire_id)?;
    let mut input_oriented: Vec<OrientedEdge> = wire.edges().to_vec();
    if input_oriented.len() < 3 {
        return Ok(None);
    }
    for oe in &input_oriented {
        if !matches!(topo.edge(oe.edge())?.curve(), EdgeCurve::Line) {
            return Ok(None);
        }
    }

    // Normalize traversal to CCW about the spine direction so side-face
    // normals built as edge_dir x path_dir point out of the material.
    let positions_of = |topo: &Topology,
                        oriented: &[OrientedEdge]|
     -> Result<Vec<Point3>, crate::OperationsError> {
        oriented
            .iter()
            .map(|oe| -> Result<Point3, crate::OperationsError> {
                let edge = topo.edge(oe.edge())?;
                Ok(topo.vertex(oe.oriented_start(edge))?.point())
            })
            .collect()
    };
    let mut ring0_positions = positions_of(topo, &input_oriented)?;
    if crate::winding::is_cw_winding(&ring0_positions, &start_tangent) {
        input_oriented = input_oriented
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        ring0_positions = positions_of(topo, &input_oriented)?;
    }
    let n = input_oriented.len();
    let num_segs = segs.len();

    // The ring is transported junction-to-junction, so the profile must lie on
    // the perpendicular plane THROUGH the chain start (the brepjs sweepSketch
    // contract); a profile elsewhere along the spine would put every ring off
    // its junction.
    if (ring0_positions[0] - spine_start).dot(start_tangent).abs() > tol.linear * 100.0 {
        return Ok(None);
    }

    // Transport the ring through every junction and verify exact closure
    // before allocating anything (snapshot then allocate).
    let mut ring_positions: Vec<Vec<Point3>> = Vec::with_capacity(num_segs);
    ring_positions.push(ring0_positions.clone());
    for seg in segs.iter().take(num_segs - 1) {
        let prev = ring_positions
            .last()
            .ok_or_else(|| crate::OperationsError::InvalidInput {
                reason: "spine sweep ring underflow".into(),
            })?;
        let next: Vec<Point3> = prev.iter().map(|&p| seg.transport(p)).collect();
        ring_positions.push(next);
    }
    {
        let last_seg = &segs[num_segs - 1];
        let last_ring = &ring_positions[num_segs - 1];
        for (i, &p) in last_ring.iter().enumerate() {
            if (last_seg.transport(p) - ring0_positions[i]).length() > tol.linear * 100.0 {
                return Ok(None);
            }
        }
    }

    // Every profile edge must be coplanar with its corner's arc axis for the
    // corner bands to be exact surfaces of revolution (skew lines sweep
    // hyperboloids). Tested on the ring TRANSPORTED to that corner's junction.
    for (k, seg) in segs.iter().enumerate() {
        if let SpineSeg::Arc { center, axis, .. } = seg {
            for i in 0..n {
                let a = ring_positions[k][i];
                let b = ring_positions[k][(i + 1) % n];
                let e = b - a;
                let coplanarity = e.cross(*axis).dot(a - *center);
                if coplanarity.abs() > tol.linear * 100.0 {
                    return Ok(None);
                }
            }
        }
    }

    // Allocate junction rings: ring 0 reuses the profile's own vertices and
    // edges; rings 1..S get fresh vertices and line edges.
    let mut ring_verts: Vec<Vec<VertexId>> = Vec::with_capacity(num_segs);
    let ring0_verts: Vec<VertexId> = input_oriented
        .iter()
        .map(|oe| -> Result<VertexId, crate::OperationsError> {
            let edge = topo.edge(oe.edge())?;
            Ok(oe.oriented_start(edge))
        })
        .collect::<Result<_, _>>()?;
    ring_verts.push(ring0_verts);
    for ring in ring_positions.iter().skip(1) {
        let verts: Vec<VertexId> = ring
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
            .collect();
        ring_verts.push(verts);
    }

    let mut ring_edges: Vec<Vec<EdgeId>> = Vec::with_capacity(num_segs);
    ring_edges.push(
        input_oriented
            .iter()
            .map(brepkit_topology::wire::OrientedEdge::edge)
            .collect(),
    );
    for ring in ring_verts.iter().skip(1) {
        let edges_k: Vec<EdgeId> = (0..n)
            .map(|i| topo.add_edge(Edge::new(ring[i], ring[(i + 1) % n], EdgeCurve::Line)))
            .collect();
        ring_edges.push(edges_k);
    }

    // Path edges per segment: lines on straight runs, rational arcs on corners.
    let mut path_edges: Vec<Vec<EdgeId>> = Vec::with_capacity(num_segs);
    for (k, seg) in segs.iter().enumerate() {
        let next = (k + 1) % num_segs;
        let mut edges_k = Vec::with_capacity(n);
        for i in 0..n {
            let sv = ring_verts[k][i];
            let ev = ring_verts[next][i];
            let curve = match seg {
                SpineSeg::Line { .. } => EdgeCurve::Line,
                SpineSeg::Arc {
                    center,
                    axis,
                    angle,
                } => EdgeCurve::NurbsCurve(
                    make_arc_curve(
                        ring_positions[k][i],
                        ring_positions[next][i],
                        *center,
                        *axis,
                        *angle,
                    )
                    .map_err(crate::OperationsError::Math)?,
                ),
            };
            edges_k.push(topo.add_edge(Edge::new(sv, ev, curve)));
        }
        path_edges.push(edges_k);
    }

    // Side faces: one per profile edge per segment.
    let mut all_faces: Vec<FaceId> = Vec::with_capacity(num_segs * n);
    for (k, seg) in segs.iter().enumerate() {
        let next = (k + 1) % num_segs;
        for i in 0..n {
            let next_i = (i + 1) % n;
            let fwd_k = if k == 0 {
                input_oriented[i].is_forward()
            } else {
                true
            };
            let fwd_next = if next == 0 {
                input_oriented[i].is_forward()
            } else {
                true
            };

            let p0_start = ring_positions[k][i];
            let p0_end = ring_positions[next][i];
            let p1_start = ring_positions[k][next_i];
            let p1_end = ring_positions[next][next_i];

            let (surface, reversed) = match seg {
                SpineSeg::Line { dir, .. } => {
                    let edge_dir = p1_start - p0_start;
                    let normal = edge_dir
                        .cross(*dir)
                        .normalize()
                        .map_err(crate::OperationsError::Math)?;
                    (
                        FaceSurface::Plane {
                            normal,
                            d: dot_normal_point(normal, p0_start),
                        },
                        false,
                    )
                }
                SpineSeg::Arc {
                    center,
                    axis,
                    angle,
                } => revolution_band_surface(
                    &EdgeCurve::Line,
                    p0_start,
                    p0_end,
                    p1_start,
                    p1_end,
                    *center,
                    *axis,
                    *angle,
                )
                .map_err(crate::OperationsError::Math)?,
            };

            // A reversed face flips every edge's effective traversal, so its
            // wire is built reversed too (the revolve side-face idiom).
            let side_wire = if reversed {
                Wire::new(
                    vec![
                        OrientedEdge::new(path_edges[k][i], true),
                        OrientedEdge::new(ring_edges[next][i], fwd_next),
                        OrientedEdge::new(path_edges[k][next_i], false),
                        OrientedEdge::new(ring_edges[k][i], !fwd_k),
                    ],
                    true,
                )
            } else {
                Wire::new(
                    vec![
                        OrientedEdge::new(ring_edges[k][i], fwd_k),
                        OrientedEdge::new(path_edges[k][next_i], true),
                        OrientedEdge::new(ring_edges[next][i], !fwd_next),
                        OrientedEdge::new(path_edges[k][i], false),
                    ],
                    true,
                )
            }
            .map_err(crate::OperationsError::Topology)?;
            let side_wire_id = topo.add_wire(side_wire);
            let fid = if reversed {
                topo.add_face(Face::new_reversed(side_wire_id, vec![], surface))
            } else {
                topo.add_face(Face::new(side_wire_id, vec![], surface))
            };
            all_faces.push(fid);
        }
    }

    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(Some(topo.add_solid(Solid::new(shell_id, vec![]))))
}

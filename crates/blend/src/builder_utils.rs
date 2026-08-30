//! Shared utilities for fillet and chamfer builders.
//!
//! Functions used by both [`FilletBuilder`](crate::fillet_builder::FilletBuilder)
//! and [`ChamferBuilder`](crate::chamfer_builder::ChamferBuilder) for creating
//! blend faces and sampling contact curves.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::BlendError;
use crate::stripe::Stripe;

/// Sample the start and end points of a NURBS curve.
#[must_use]
pub fn sample_nurbs_endpoints(curve: &NurbsCurve) -> Vec<Point3> {
    let (t0, t1) = curve.domain();
    vec![curve.evaluate(t0), curve.evaluate(t1)]
}

/// Construct the exact circular cross-section arc between two contacts.
pub fn cross_section_curve(
    sec: &crate::section::CircSection,
    a: Point3,
    b: Point3,
) -> Option<EdgeCurve> {
    let normal = (a - sec.center).cross(b - sec.center).normalize().ok()?;
    let circle = brepkit_math::curves::Circle3D::new(sec.center, normal, sec.radius).ok()?;
    Some(EdgeCurve::Circle(circle))
}

/// Create the blend face for a periodic (closed rim) stripe.
///
/// The stripe is one full-revolution roll: both contacts are closed edges
/// (start == end vertex) and the two cross-sections are the same seam segment
/// traversed in opposite directions. Canonical solids store that seam as a
/// single edge entity used twice in the band wire (the source cylinder's own
/// seam does exactly this). Minting a twin arc per direction would leave each
/// arc with a single face use and open the shell at the seam.
///
/// # Errors
///
/// Returns [`BlendError`] if the contact boundaries have no materialized
/// edge, the seam vertices are degenerate, or wire/face construction fails.
pub fn create_periodic_blend_face(
    topo: &mut Topology,
    stripe: &Stripe,
    registry: &mut crate::boundary_registry::BoundaryRegistry,
    contact1: crate::boundary_registry::BoundaryHandle,
    contact2: crate::boundary_registry::BoundaryHandle,
) -> Result<BlendFaceInfo, BlendError> {
    let entry1 = registry
        .entry(contact1)
        .ok_or_else(|| BlendError::PlanningFailure {
            reason: format!("unknown periodic contact boundary {contact1}"),
        })?;
    let entry2 = registry
        .entry(contact2)
        .ok_or_else(|| BlendError::PlanningFailure {
            reason: format!("unknown periodic contact boundary {contact2}"),
        })?;
    if entry1.edge_id().is_none() {
        return Err(BlendError::PlanningFailure {
            reason: format!("periodic contact boundary {contact1:?} was not materialized"),
        });
    }
    if entry2.edge_id().is_none() {
        return Err(BlendError::PlanningFailure {
            reason: format!("periodic contact boundary {contact2:?} was not materialized"),
        });
    }
    let v1 = entry1.start.vertex;
    let v2 = entry2.start.vertex;
    if v1 == v2 {
        return Err(BlendError::PlanningFailure {
            reason: "periodic blend seam vertices coincide".to_owned(),
        });
    }
    let contact1_oriented = registry.oriented_edge(topo, contact1, 1)?;
    let contact2_oriented = registry.oriented_edge(topo, contact2, 1)?;
    let seam = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
    let wire = Wire::new(
        vec![
            contact1_oriented,
            OrientedEdge::new(seam, true),
            contact2_oriented,
            OrientedEdge::new(seam, false),
        ],
        true,
    )?;
    let wire_id = topo.add_wire(wire);
    let face = topo.add_face(Face::new(wire_id, Vec::new(), stripe.surface.clone()));
    Ok(BlendFaceInfo {
        face,
        cross_end: None,
        cross_start: None,
    })
}

/// Create a blend face from a stripe's surface and contact curves.
///
/// Builds a minimal quadrilateral wire from the four contact-curve endpoints
/// and associates the blend surface with it.
///
/// # Errors
///
/// Returns [`BlendError`] if wire or face construction fails.
/// [`create_blend_face`] that REUSES the trimmers' contact edges when they
/// span the same contacts. Minting fresh edges for curves the trimmed
/// neighbours already carry leaves two edge entities per contact — each used
/// by one face — opening the shell along every blend flank. A trimmer edge
/// is adopted (with its vertices) when its endpoints match the stripe's
/// contact endpoints within the weld band, in either orientation; otherwise
/// that side falls back to a fresh edge.
pub fn create_blend_face_with_contacts(
    topo: &mut Topology,
    stripe: &Stripe,
    contact1_edge: Option<brepkit_topology::edge::EdgeId>,
    contact2_edge: Option<brepkit_topology::edge::EdgeId>,
) -> Result<BlendFaceInfo, BlendError> {
    const WELD: f64 = 1e-5;
    let (t0_1, t1_1) = stripe.contact1.domain();
    let (t0_2, t1_2) = stripe.contact2.domain();

    let p1_start = stripe.contact1.evaluate(t0_1);
    let p1_end = stripe.contact1.evaluate(t1_1);
    let p2_start = stripe.contact2.evaluate(t0_2);
    let p2_end = stripe.contact2.evaluate(t1_2);

    // Adopt a trimmer contact edge when its endpoints match `(want_s, want_e)`
    // in either orientation: returns (edge, forward, start_vid, end_vid) in
    // the WIRE traversal direction.
    let adopt = |topo: &Topology,
                 eid: Option<brepkit_topology::edge::EdgeId>,
                 want_s: Point3,
                 want_e: Point3|
     -> Option<(brepkit_topology::edge::EdgeId, bool, VertexId, VertexId)> {
        let eid = eid?;
        let e = topo.edge(eid).ok()?;
        let (sv, ev) = (e.start(), e.end());
        let sp = topo.vertex(sv).ok()?.point();
        let ep = topo.vertex(ev).ok()?.point();
        if (sp - want_s).length() <= WELD && (ep - want_e).length() <= WELD {
            Some((eid, true, sv, ev))
        } else if (sp - want_e).length() <= WELD && (ep - want_s).length() <= WELD {
            Some((eid, false, ev, sv))
        } else {
            None
        }
    };
    let adopt1 = adopt(topo, contact1_edge, p1_start, p1_end);
    // Contact 2 traverses end -> start in the quad below.
    let adopt2 = adopt(topo, contact2_edge, p2_end, p2_start);

    // A variable-radius stripe can pinch to a point at an end: both
    // contact curves land on the same position. Detect it up front so the
    // pinched end SHARES one vertex entity between the two contact curves
    // (the cross edge is skipped below, and separate entities would leave
    // the wire closed only positionally, not at entity level — see the
    // closure tolerance note: validation treats vertices as coincident at
    // 1e-7, tighter than the 1e-5 weld distance used here).
    let end_degenerate = (p1_end - p2_end).length() < WELD;
    let start_degenerate = (p2_start - p1_start).length() < WELD;

    // Create/reuse vertices (snapshot then allocate).
    let (v1s, v1e) = adopt1.map_or_else(
        || {
            (
                topo.add_vertex(Vertex::new(p1_start, 1e-7)),
                topo.add_vertex(Vertex::new(p1_end, 1e-7)),
            )
        },
        |(_, _, s, e)| (s, e),
    );
    let (v2e, v2s) = adopt2.map_or_else(
        || {
            (
                if end_degenerate {
                    v1e
                } else {
                    topo.add_vertex(Vertex::new(p2_end, 1e-7))
                },
                if start_degenerate {
                    v1s
                } else {
                    topo.add_vertex(Vertex::new(p2_start, 1e-7))
                },
            )
        },
        |(_, _, s, e)| (s, e),
    );

    // Build quad: p1_start -> p1_end -> p2_end -> p2_start -> p1_start.
    // Use actual contact curves for e0 and e2 (the longitudinal edges along
    // the spine direction). Cross edges e1 and e3 are straight lines connecting
    // the two contact curves at the spine endpoints.
    let (e0, e0_fwd) = adopt1.map_or_else(
        || {
            (
                topo.add_edge(Edge::new(
                    v1s,
                    v1e,
                    EdgeCurve::NurbsCurve(stripe.contact1.clone()),
                )),
                true,
            )
        },
        |(eid, fwd, _, _)| (eid, fwd),
    );
    // Cross edges carry the true end cross-section arcs when the stripe has
    // sections: the fillet's end profile is a circular arc, and a straight
    // chord both misrepresents the surface boundary and can never be shared
    // with a notched end cap.
    let end_curve = stripe
        .sections
        .last()
        .and_then(|sec| {
            let r = cross_section_curve(sec, p1_end, p2_end);
            if r.is_none() {
                log::debug!(
                    "cross END line fallback: sec c={:?} r={:.5} a={p1_end:?} b={p2_end:?}",
                    sec.center,
                    sec.radius
                );
            }
            r
        })
        .unwrap_or(EdgeCurve::Line);
    let start_curve = stripe
        .sections
        .first()
        .and_then(|sec| {
            let r = cross_section_curve(sec, p2_start, p1_start);
            if r.is_none() {
                log::debug!(
                    "cross START line fallback: sec c={:?} r={:.5} a={p2_start:?} b={p1_start:?}",
                    sec.center,
                    sec.radius
                );
            }
            r
        })
        .unwrap_or(EdgeCurve::Line);
    // A pinched end's cross edge would be zero-length: minting it leaves a
    // degenerate use-1 edge no weld can pair; skip it.
    let e1 = if end_degenerate {
        Option::None
    } else {
        Some(topo.add_edge(Edge::new(v1e, v2e, end_curve)))
    };
    let (e2, e2_fwd) = adopt2.map_or_else(
        || {
            (
                topo.add_edge(Edge::new(
                    v2e,
                    v2s,
                    EdgeCurve::NurbsCurve(stripe.contact2.clone()),
                )),
                true,
            )
        },
        |(eid, fwd, _, _)| (eid, fwd),
    );
    let e3 = if start_degenerate {
        Option::None
    } else {
        Some(topo.add_edge(Edge::new(v2s, v1s, start_curve)))
    };

    let mut wire_edges = vec![OrientedEdge::new(e0, e0_fwd)];
    if let Some(e1) = e1 {
        wire_edges.push(OrientedEdge::new(e1, true));
    }
    wire_edges.push(OrientedEdge::new(e2, e2_fwd));
    if let Some(e3) = e3 {
        wire_edges.push(OrientedEdge::new(e3, true));
    }
    let wire = Wire::new(wire_edges, true)?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, Vec::new(), stripe.surface.clone());
    let face_id = topo.add_face(face);

    Ok(BlendFaceInfo {
        face: face_id,
        cross_end: e1.map(|e| (e, v1e, v2e)),
        cross_start: e3.map(|e| (e, v2s, v1s)),
    })
}

/// A created blend face plus its two cross edges (the end cross-section
/// arcs), each with its (from, to) vertices in the blend wire's traversal
/// direction — the handles the end-cap notch surgery needs to SHARE those
/// arcs instead of leaving both sides use-1.
pub struct BlendFaceInfo {
    /// The blend face.
    pub face: FaceId,
    /// Cross edge at the spine end: `(edge, from, to)`.
    pub cross_end: Option<(brepkit_topology::edge::EdgeId, VertexId, VertexId)>,
    /// Cross edge at the spine start: `(edge, from, to)`. `None` when the
    /// stripe pinches to a point at that end and no cross edge exists.
    pub cross_start: Option<(brepkit_topology::edge::EdgeId, VertexId, VertexId)>,
}

#[allow(dead_code)]
/// Build a face directly from canonical registry boundaries.
///
/// Every wire edge is obtained from
/// [`crate::boundary_registry::BoundaryRegistry::oriented_edge`], so this
/// path preserves one materialized edge and the planner's orientation on both
/// owners. It deliberately does not convert the boundaries to a point-list
/// `FaceSpec`.
///
/// # Errors
///
/// Returns an error when fewer than two boundaries are supplied, a boundary
/// handle/owner is invalid, or the wire cannot be constructed.
pub fn create_face_from_registry(
    topo: &mut Topology,
    registry: &mut crate::boundary_registry::BoundaryRegistry,
    surface: FaceSurface,
    boundaries: &[(crate::boundary_registry::BoundaryHandle, usize)],
) -> Result<FaceId, BlendError> {
    if boundaries.len() < 2 {
        return Err(BlendError::PlanningFailure {
            reason: format!(
                "registry face needs at least two boundaries, got {}",
                boundaries.len()
            ),
        });
    }
    let oriented = boundaries
        .iter()
        .map(|&(handle, owner)| registry.oriented_edge(topo, handle, owner))
        .collect::<Result<Vec<_>, _>>()?;
    let wire_id = topo.add_wire(Wire::new(oriented, true)?);
    Ok(topo.add_face(Face::new(wire_id, Vec::new(), surface)))
}

#[allow(dead_code)]
/// Create a stripe face from four canonical registry boundaries.
///
/// The boundaries are ordered as contact-1, end cross-section, contact-2,
/// start cross-section. Cross-section handles are optional for pinched ends.
/// Unlike the compatibility constructor below, this function performs no
/// endpoint matching or geometric welding.
///
/// # Errors
/// Returns an error when a registry boundary is invalid or the wire cannot be
/// constructed.
pub fn create_blend_face_from_registry(
    topo: &mut Topology,
    stripe: &Stripe,
    registry: &mut crate::boundary_registry::BoundaryRegistry,
    contact1: (crate::boundary_registry::BoundaryHandle, usize),
    contact2: (crate::boundary_registry::BoundaryHandle, usize),
    cross_end: Option<(crate::boundary_registry::BoundaryHandle, usize)>,
    cross_start: Option<(crate::boundary_registry::BoundaryHandle, usize)>,
) -> Result<BlendFaceInfo, BlendError> {
    let mut boundaries = Vec::with_capacity(4);
    boundaries.push(contact1);
    if let Some(boundary) = cross_end {
        boundaries.push(boundary);
    }
    boundaries.push(contact2);
    if let Some(boundary) = cross_start {
        boundaries.push(boundary);
    }
    let face = create_face_from_registry(topo, registry, stripe.surface.clone(), &boundaries)?;
    let edge_vertices =
        |handle: crate::boundary_registry::BoundaryHandle,
         owner: usize,
         topo: &Topology,
         registry: &crate::boundary_registry::BoundaryRegistry|
         -> Result<(brepkit_topology::edge::EdgeId, VertexId, VertexId), BlendError> {
            let entry = registry
                .entry(handle)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("unknown registry boundary handle {handle}"),
                })?;
            let edge_id = entry.edge_id().ok_or_else(|| BlendError::PlanningFailure {
                reason: format!("registry boundary {:?} was not materialized", entry.key),
            })?;
            let forward = entry
                .owners
                .get(owner)
                .ok_or_else(|| BlendError::PlanningFailure {
                    reason: format!("boundary {:?} has invalid owner index {owner}", entry.key),
                })?
                .forward;
            let edge = topo.edge(edge_id)?;
            let (from, to) = if forward {
                (edge.start(), edge.end())
            } else {
                (edge.end(), edge.start())
            };
            Ok((edge_id, from, to))
        };
    let end = cross_end
        .map(|(handle, owner)| edge_vertices(handle, owner, topo, registry))
        .transpose()?;
    let start = cross_start
        .map(|(handle, owner)| edge_vertices(handle, owner, topo, registry))
        .transpose()?;
    Ok(BlendFaceInfo {
        face,
        cross_end: end,
        cross_start: start,
    })
}

/// Replace a face's two-edge corner path `from -> corner -> to` with the
/// single cross-section arc `edge`, notching the fillet's end profile out of
/// an end cap so the cap and the blend share one edge entity. Both replaced
/// edges must be straight (the box corner sides); returns whether a
/// replacement happened.
pub fn notch_face_corner_with_arc(
    topo: &mut Topology,
    face_id: FaceId,
    arc: (brepkit_topology::edge::EdgeId, VertexId, VertexId),
) -> Result<Option<FaceId>, BlendError> {
    let (arc_eid, va, vb) = arc;
    let wire_id = topo.face(face_id)?.outer_wire();
    let oes = topo.wire(wire_id)?.edges().to_vec();
    let n = oes.len();
    if n < 3 {
        return Ok(None);
    }
    let ends = |oe: &OrientedEdge| -> Result<(VertexId, VertexId), BlendError> {
        let e = topo.edge(oe.edge())?;
        Ok((oe.oriented_start(e), oe.oriented_end(e)))
    };
    if std::env::var("BK_NOTCH_TRACE").is_ok() {
        let mut has_a = false;
        let mut has_b = false;
        for oe in &oes {
            let (s, e) = ends(oe)?;
            has_a |= s == va || e == va;
            has_b |= s == vb || e == vb;
        }
        if has_a || has_b {
            log::warn!("NOTCH-TRACE face={face_id:?} has_va={has_a} has_vb={has_b} wire_len={n}");
        }
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (s0, e0) = ends(&oes[i])?;
        let (s1, e1) = ends(&oes[j])?;
        if e0 != s1 || e0 == va || e0 == vb {
            continue;
        }
        let fwd = s0 == va && e1 == vb;
        let rev = s0 == vb && e1 == va;
        if !(fwd || rev) {
            continue;
        }
        let both_straight = [oes[i].edge(), oes[j].edge()].iter().all(|&eid| {
            topo.edge(eid)
                .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
        });
        if !both_straight {
            continue;
        }
        let mut new_oes: Vec<OrientedEdge> = Vec::with_capacity(n - 1);
        for (k, oe) in oes.iter().enumerate() {
            if k == i {
                new_oes.push(OrientedEdge::new(arc_eid, fwd));
            } else if k != j {
                new_oes.push(*oe);
            }
        }
        let new_wire = topo.add_wire(Wire::new(new_oes, true)?);
        let (surface, reversed, inners) = {
            let f = topo.face(face_id)?;
            (
                f.surface().clone(),
                f.is_reversed(),
                f.inner_wires().to_vec(),
            )
        };
        let new_face = if reversed {
            Face::new_reversed(new_wire, inners, surface)
        } else {
            Face::new(new_wire, inners, surface)
        };
        let nf = topo.add_face(new_face);
        return Ok(Some(nf));
    }
    Ok(None)
}

/// Adapter that provides [`ParametricSurface`] for a `FaceSurface::Plane`.
///
/// Planes store only a normal and signed distance `d`, with no parametric
/// frame.  This adapter builds an orthonormal UV frame from the normal so
/// that the walking engine can evaluate, project, and differentiate the
/// plane surface uniformly.
pub struct PlaneAdapter {
    /// Origin point on the plane (the point closest to the world origin).
    pub origin: Point3,
    /// U-direction tangent (unit vector in the plane).
    pub u_dir: Vec3,
    /// V-direction tangent (unit vector in the plane, orthogonal to `u_dir`).
    pub v_dir: Vec3,
    /// Outward-facing unit normal.
    pub norm: Vec3,
}

impl PlaneAdapter {
    /// Build a `PlaneAdapter` from a plane normal and signed distance.
    ///
    /// The UV frame is constructed by choosing a non-parallel reference vector
    /// and computing the cross products.
    #[must_use]
    pub fn from_normal_and_d(normal: Vec3, d: f64) -> Self {
        let origin = Point3::new(normal.x() * d, normal.y() * d, normal.z() * d);

        // Pick a reference vector that is not parallel to the normal.
        let ref_vec = if normal.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };

        let u_dir = normal
            .cross(ref_vec)
            .normalize()
            .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
        let v_dir = normal
            .cross(u_dir)
            .normalize()
            .unwrap_or(Vec3::new(0.0, 1.0, 0.0));

        Self {
            origin,
            u_dir,
            v_dir,
            norm: normal,
        }
    }
}

impl ParametricSurface for PlaneAdapter {
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.origin + self.u_dir * u + self.v_dir * v
    }

    fn normal(&self, _u: f64, _v: f64) -> Vec3 {
        self.norm
    }

    fn project_point(&self, point: Point3) -> (f64, f64) {
        let d = point - self.origin;
        (d.dot(self.u_dir), d.dot(self.v_dir))
    }

    fn partial_u(&self, _u: f64, _v: f64) -> Vec3 {
        self.u_dir
    }

    fn partial_v(&self, _u: f64, _v: f64) -> Vec3 {
        self.v_dir
    }
}

/// A [`ParametricSurface`] view that negates the wrapped surface's normal.
///
/// The walking engine's blend constraint places the rolling-ball centre on the
/// `+normal` side of each surface (`centre = p + r·normal`), so the surfaces
/// must present their **inward** (toward-material) normals. `PlaneAdapter`
/// flips a plane via its stored normal, but analytic/NURBS surfaces have an
/// intrinsic outward normal that can't be re-oriented in place — wrapping one
/// here flips it so a fillet against a curved neighbour solves the internal
/// (material-side) branch instead of the external common-tangent one.
pub struct FlippedNormalSurface<'a> {
    inner: &'a dyn ParametricSurface,
}

impl<'a> FlippedNormalSurface<'a> {
    /// Wrap a surface so its normal is negated.
    #[must_use]
    pub const fn new(inner: &'a dyn ParametricSurface) -> Self {
        Self { inner }
    }
}

impl ParametricSurface for FlippedNormalSurface<'_> {
    fn evaluate(&self, u: f64, v: f64) -> Point3 {
        self.inner.evaluate(u, v)
    }

    fn normal(&self, u: f64, v: f64) -> Vec3 {
        -self.inner.normal(u, v)
    }

    fn project_point(&self, point: Point3) -> (f64, f64) {
        self.inner.project_point(point)
    }

    fn partial_u(&self, u: f64, v: f64) -> Vec3 {
        self.inner.partial_u(u, v)
    }

    fn partial_v(&self, u: f64, v: f64) -> Vec3 {
        self.inner.partial_v(u, v)
    }
}

/// Extract a `&dyn ParametricSurface` from a `FaceSurface`, or build a
/// `PlaneAdapter` for plane faces.
///
/// Returns `Ok(adapter)` for planes and `Err(face_id)` for unsupported types.
/// For analytic and NURBS surfaces that already implement `ParametricSurface`,
/// the reference is extracted directly and the adapter is unused.
///
/// # Usage pattern
///
/// ```ignore
/// let mut adapter = None;
/// let surf: &dyn ParametricSurface = surface_ref_or_adapter(&face_surface, &mut adapter);
/// ```
#[must_use]
pub fn surface_ref_or_adapter<'a>(
    surface: &'a FaceSurface,
    adapter_slot: &'a mut Option<PlaneAdapter>,
) -> &'a dyn ParametricSurface {
    // For Plane faces, we need to populate the adapter_slot first,
    // then return a reference to it. For all other variants, we can
    // return a reference directly to the surface inside FaceSurface.
    if let FaceSurface::Plane { normal, d } = surface {
        let adapter = adapter_slot.insert(PlaneAdapter::from_normal_and_d(*normal, *d));
        return adapter as &dyn ParametricSurface;
    }
    match surface {
        FaceSurface::Plane { .. } => {
            // Already handled above; this arm is unreachable.
            adapter_slot.insert(PlaneAdapter::from_normal_and_d(
                Vec3::new(0.0, 0.0, 1.0),
                0.0,
            )) as &dyn ParametricSurface
        }
        FaceSurface::Cylinder(c) => c as &dyn ParametricSurface,
        FaceSurface::Cone(c) => c as &dyn ParametricSurface,
        FaceSurface::Sphere(s) => s as &dyn ParametricSurface,
        FaceSurface::Torus(t) => t as &dyn ParametricSurface,
        FaceSurface::Nurbs(n) => n as &dyn ParametricSurface,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::face::FaceSurface;
    use brepkit_topology::vertex::Vertex;

    #[test]
    fn registry_faces_reuse_edges_without_geometric_welding() {
        let mut topo = Topology::new();
        let vertices = [
            topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), 1e-7)),
        ];
        let mut registry = crate::boundary_registry::BoundaryRegistry::new();
        let mut handles = Vec::new();
        for (index, pair) in [(0, 1), (1, 2), (2, 3), (3, 0)].into_iter().enumerate() {
            handles.push(
                registry
                    .register(
                        crate::boundary_registry::BoundaryKey::contact(0, index, 0),
                        crate::boundary_registry::PlannedVertex::new(vertices[pair.0]),
                        crate::boundary_registry::PlannedVertex::new(vertices[pair.1]),
                        EdgeCurve::Line,
                        (0.0, 1.0),
                        [
                            crate::boundary_registry::BoundaryOwner::planned("face A", true),
                            crate::boundary_registry::BoundaryOwner::planned("face B", false),
                        ],
                    )
                    .unwrap(),
            );
        }
        let face_a = create_face_from_registry(
            &mut topo,
            &mut registry,
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
            &handles
                .iter()
                .map(|&handle| (handle, 0))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let face_b = create_face_from_registry(
            &mut topo,
            &mut registry,
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, -1.0),
                d: 0.0,
            },
            &handles
                .iter()
                .rev()
                .map(|&handle| (handle, 1))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        for &handle in &handles {
            registry.set_owner_face(handle, 0, face_a).unwrap();
            registry.set_owner_face(handle, 1, face_b).unwrap();
        }
        registry.preassembly_audit().unwrap();
        let report = registry
            .postassembly_audit(&topo, &[face_a, face_b])
            .unwrap();
        assert_eq!(report.len(), handles.len());
        assert!(report.iter().all(|incidence| incidence.uses == 2));
        assert!(report.iter().all(|incidence| incidence.key.is_some()));
    }
}

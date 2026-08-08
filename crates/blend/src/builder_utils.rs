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
                topo.add_vertex(Vertex::new(p2_end, 1e-7)),
                topo.add_vertex(Vertex::new(p2_start, 1e-7)),
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
    // with a notched end cap. The arc's plane normal comes from the two
    // contact endpoints and the section centre.
    let arc_curve =
        |sec: &crate::section::CircSection, a: Point3, b: Point3| -> Option<EdgeCurve> {
            let u = a - sec.center;
            let v = b - sec.center;
            let n = u.cross(v);
            let n = n.normalize().ok()?;
            let circle = brepkit_math::curves::Circle3D::new(sec.center, n, sec.radius).ok()?;
            Some(EdgeCurve::Circle(circle))
        };
    let end_curve = stripe
        .sections
        .last()
        .and_then(|sec| arc_curve(sec, p1_end, p2_end))
        .unwrap_or(EdgeCurve::Line);
    let start_curve = stripe
        .sections
        .first()
        .and_then(|sec| arc_curve(sec, p2_start, p1_start))
        .unwrap_or(EdgeCurve::Line);
    let e1 = topo.add_edge(Edge::new(v1e, v2e, end_curve));
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
    let e3 = topo.add_edge(Edge::new(v2s, v1s, start_curve));

    let wire = Wire::new(
        vec![
            OrientedEdge::new(e0, e0_fwd),
            OrientedEdge::new(e1, true),
            OrientedEdge::new(e2, e2_fwd),
            OrientedEdge::new(e3, true),
        ],
        true,
    )?;
    let wire_id = topo.add_wire(wire);

    let face = Face::new(wire_id, Vec::new(), stripe.surface.clone());
    let face_id = topo.add_face(face);

    Ok(BlendFaceInfo {
        face: face_id,
        cross_end: (e1, v1e, v2e),
        cross_start: (e3, v2s, v1s),
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
    pub cross_end: (brepkit_topology::edge::EdgeId, VertexId, VertexId),
    /// Cross edge at the spine start: `(edge, from, to)`.
    pub cross_start: (brepkit_topology::edge::EdgeId, VertexId, VertexId),
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

/// Weld pairs of free (use-1) edges that trace identical geometry.
///
/// Adjacent blend walls whose terminal sections coincide each mint their own
/// cross edge — same endpoints, same curve, two edge entities each used by
/// one face. Rewrite every wire of the faces to reference one edge per
/// geometric identity. Requires BOTH endpoints and the curve midpoint to
/// match at weld distance, so complementary arcs and genuinely distinct
/// co-endpoint edges are never merged; zero-length edges collapse away
/// entirely when their twin is also zero-length.
pub(crate) fn weld_coincident_free_edges(
    topo: &mut Topology,
    faces: &[FaceId],
) -> Result<(), BlendError> {
    use brepkit_topology::edge::EdgeId;
    use std::collections::HashMap;

    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            for oe in topo.wire(wid)?.edges() {
                *uses.entry(oe.edge()).or_insert(0) += 1;
            }
        }
    }

    const WELD: f64 = 1e-6;
    let q = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() / WELD).round() as i64,
            (p.y() / WELD).round() as i64,
            (p.z() / WELD).round() as i64,
        )
    };

    // Geometry key for every free edge: symmetric endpoint pair + midpoint.
    let mut groups: HashMap<
        ((i64, i64, i64), (i64, i64, i64), (i64, i64, i64)),
        Vec<(EdgeId, VertexId, VertexId)>,
    > = HashMap::new();
    let mut free_edges: Vec<EdgeId> = uses
        .iter()
        .filter(|&(_, &c)| c == 1)
        .map(|(&e, _)| e)
        .collect();
    free_edges.sort_unstable_by_key(|e| e.index());
    for eid in free_edges {
        let e = topo.edge(eid)?;
        let (sv, ev) = (e.start(), e.end());
        let sp = topo.vertex(sv)?.point();
        let ep = topo.vertex(ev)?.point();
        // The geometric identity slot. Stored-curve evaluation is
        // phase-dependent for circles (endpoints do not trim the raw
        // parameterization), so circle edges key on centre + radius + |axis|
        // instead; antipodal endpoint pairs stay unkeyed (minor/major arc
        // ambiguity — the merge-key lesson).
        let mid = match e.curve() {
            EdgeCurve::Circle(c) => {
                let chord_mid = Point3::new(
                    (sp.x() + ep.x()) * 0.5,
                    (sp.y() + ep.y()) * 0.5,
                    (sp.z() + ep.z()) * 0.5,
                );
                if (chord_mid - c.center()).length() < 1e-6 {
                    continue;
                }
                let ax = c.normal();
                c.center() + Vec3::new(ax.x().abs(), ax.y().abs(), ax.z().abs()) * c.radius()
            }
            _ => e.curve().evaluate_with_endpoints(0.5, sp, ep),
        };
        let (ks, ke) = (q(sp), q(ep));
        let key = if ks <= ke {
            (ks, ke, q(mid))
        } else {
            (ke, ks, q(mid))
        };
        groups.entry(key).or_default().push((eid, sv, ev));
    }

    // For each group, rewrite all wires to use the first edge.
    let mut replace: HashMap<EdgeId, (EdgeId, bool)> = HashMap::new();
    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        let (keep, keep_sv, _) = members[0];
        let keep_sp = topo.vertex(keep_sv)?.point();
        for &(dup, dup_sv, _) in &members[1..] {
            let dup_sp = topo.vertex(dup_sv)?.point();
            let same_dir = (dup_sp - keep_sp).length() < WELD;
            replace.insert(dup, (keep, same_dir));
        }
    }
    if replace.is_empty() {
        return Ok(());
    }

    for &fid in faces {
        let face = topo.face(fid)?;
        let mut wires = vec![face.outer_wire()];
        wires.extend_from_slice(face.inner_wires());
        for wid in wires {
            let wire = topo.wire(wid)?;
            let mut edges = wire.edges().to_vec();
            let mut changed = false;
            for oe in &mut edges {
                if let Some(&(keep, same_dir)) = replace.get(&oe.edge()) {
                    let fwd = if same_dir {
                        oe.is_forward()
                    } else {
                        !oe.is_forward()
                    };
                    *oe = OrientedEdge::new(keep, fwd);
                    changed = true;
                }
            }
            if changed {
                let closed = wire.is_closed();
                *topo.wire_mut(wid)? = Wire::new(edges, closed)?;
            }
        }
    }
    Ok(())
}

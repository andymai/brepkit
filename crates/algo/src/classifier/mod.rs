//! Face classification -- determines if a sub-face is inside/outside
//! the opposing solid.
//!
//! Two strategies:
//! - **Analytic**: O(1) point-in-solid for convex analytic solids.
//! - **Ray cast**: Multi-ray fallback for general solids.

mod analytic;
mod ray_cast;

pub use analytic::{AnalyticClassifier, classify_analytic, try_build_analytic_classifier};
pub use ray_cast::{
    RayCastGeoms, classify_ray_cast, classify_ray_cast_cached, compute_solid_bbox,
    planar_face_polygons, point_in_face_3d, point_in_planar_region, ray_cast_inside_votes,
    ray_cast_inside_votes_cached,
};
pub(crate) use ray_cast::{largest_u_gap, u_in_gap};

use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Classify a point relative to a solid -- dispatch to the best available
/// strategy.
///
/// Tries the analytic classifier first (O(1) for convex analytic solids),
/// then falls back to ray casting.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate.
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    if let Some(class) = classify_analytic(topo, solid, point) {
        return Ok(class);
    }

    classify_ray_cast(topo, solid, point)
}

/// Like [`classify_point`], but reuses pre-collected ray-cast geometry for the
/// solid when available (`Some`).
///
/// The analytic fast path is tried first exactly as in [`classify_point`]; only
/// the ray-cast fallback consults the cache. Passing `None` reproduces
/// [`classify_point`] verbatim (geometry collected per call), so a caller that
/// failed to build the cache degrades to identical behaviour.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate.
pub fn classify_point_cached(
    topo: &Topology,
    solid: SolidId,
    geoms: Option<&ray_cast::RayCastGeoms>,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    let analytic = match geoms {
        Some(g) => g
            .analytic()
            .and_then(|c| c.classify(point, brepkit_math::tolerance::Tolerance::new())),
        None => classify_analytic(topo, solid, point),
    };
    if let Some(class) = analytic {
        return Ok(class);
    }

    match geoms {
        Some(g) => ray_cast::classify_ray_cast_cached(g, point),
        None => classify_ray_cast(topo, solid, point),
    }
}

/// A face's loops sampled into its own 2D parameter space.
///
/// A plane's local frame, or the surface's `(u, v)` with `u` unwrapped along
/// each loop on a periodic surface. Built once per face and reused for
/// containment tests and for alternative interior samples.
pub struct FaceLoops2d {
    frame: Option<crate::builder::plane_frame::PlaneFrame>,
    surface: FaceSurface,
    periodic: bool,
    u_mean: f64,
    /// The outer loop, in traversal order.
    pub outer: Vec<brepkit_math::vec::Point2>,
    /// The inner loops (holes), each in traversal order.
    pub holes: Vec<Vec<brepkit_math::vec::Point2>>,
    /// Axis-aligned bounds of every sampled boundary point.
    pub aabb: [f64; 6],
}

impl FaceLoops2d {
    /// Sample `face_id`'s loops: two points per line, sixteen per curved edge,
    /// each edge walked in its traversal direction and excluding the traversal
    /// endpoint (the next edge supplies it).
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] on a topology lookup failure.
    pub fn new(
        topo: &Topology,
        face_id: brepkit_topology::face::FaceId,
    ) -> Result<Self, AlgoError> {
        use brepkit_math::vec::Point2;
        use brepkit_topology::edge::EdgeCurve;
        use std::f64::consts::{PI, TAU};

        let face = topo.face(face_id)?;
        let surface = face.surface().clone();
        let periodic = matches!(
            &surface,
            FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_)
        );
        let frame = if let FaceSurface::Plane { normal, .. } = &surface {
            let mut pts = Vec::new();
            for oe in topo.wire(face.outer_wire())?.edges() {
                let e = topo.edge(oe.edge())?;
                pts.push(topo.vertex(e.start())?.point());
            }
            Some(crate::builder::plane_frame::PlaneFrame::from_plane_face(
                *normal, &pts,
            ))
        } else {
            None
        };
        let mut aabb = [f64::MAX, f64::MAX, f64::MAX, f64::MIN, f64::MIN, f64::MIN];
        let mut sample_loop = |wid: brepkit_topology::wire::WireId,
                               u_ref: Option<f64>|
         -> Result<Vec<Point2>, AlgoError> {
            let wire = topo.wire(wid)?;
            let mut out: Vec<Point2> = Vec::new();
            for oe in wire.edges() {
                let e = topo.edge(oe.edge())?;
                let sp = topo.vertex(e.start())?.point();
                let ep = topo.vertex(e.end())?.point();
                let (t0, t1) = e.curve().domain_with_endpoints(sp, ep);
                let n = if matches!(e.curve(), EdgeCurve::Line) {
                    2
                } else {
                    16
                };
                for k in 0..n {
                    #[allow(clippy::cast_precision_loss)]
                    let f = k as f64 / n as f64;
                    // Traversal order: a forward edge walks t0 -> t1 and
                    // excludes t1; a reversed edge walks t1 -> t0 and
                    // excludes t0. The shared junction vertex is supplied
                    // exactly once, by the edge that starts there.
                    let t = if oe.is_forward() {
                        t0 + (t1 - t0) * f
                    } else {
                        t1 - (t1 - t0) * f
                    };
                    let p3 = e.curve().evaluate_with_endpoints(t, sp, ep);
                    for (a, v) in [p3.x(), p3.y(), p3.z()].iter().enumerate() {
                        aabb[a] = aabb[a].min(*v);
                        aabb[a + 3] = aabb[a + 3].max(*v);
                    }
                    let Some(q) = (match frame.as_ref() {
                        Some(f) => Some(f.project(p3)),
                        None => surface.project_point(p3).map(|(u, v)| Point2::new(u, v)),
                    }) else {
                        continue;
                    };
                    let anchor = if periodic {
                        out.last().map_or(u_ref, |prev| Some(prev.x()))
                    } else {
                        None
                    };
                    let q = if let Some(a) = anchor {
                        Point2::new(a + (q.x() - a + PI).rem_euclid(TAU) - PI, q.y())
                    } else {
                        q
                    };
                    out.push(q);
                }
            }
            Ok(out)
        };
        let outer = sample_loop(face.outer_wire(), None)?;
        #[allow(clippy::cast_precision_loss)]
        let u_mean = if outer.is_empty() {
            0.0
        } else {
            outer.iter().map(|q| q.x()).sum::<f64>() / outer.len() as f64
        };
        let mut holes = Vec::new();
        for &wid in face.inner_wires() {
            holes.push(sample_loop(wid, Some(u_mean))?);
        }
        Ok(Self {
            frame,
            surface,
            periodic,
            u_mean,
            outer,
            holes,
            aabb,
        })
    }

    /// `p` in this face's 2D space, `u` unwrapped next to the outer loop.
    #[must_use]
    pub fn to_uv(&self, p: Point3) -> Option<brepkit_math::vec::Point2> {
        use brepkit_math::vec::Point2;
        use std::f64::consts::{PI, TAU};
        let q = if let Some(f) = self.frame.as_ref() {
            f.project(p)
        } else {
            let (u, v) = self.surface.project_point(p)?;
            Point2::new(u, v)
        };
        if self.periodic {
            let u = self.u_mean + (q.x() - self.u_mean + PI).rem_euclid(TAU) - PI;
            Some(Point2::new(u, q.y()))
        } else {
            Some(q)
        }
    }

    /// The 3D point at `q`.
    #[must_use]
    pub fn to_3d(&self, q: brepkit_math::vec::Point2) -> Option<Point3> {
        match self.frame.as_ref() {
            Some(f) => Some(f.evaluate(q.x(), q.y())),
            None => self.surface.evaluate(q.x(), q.y()),
        }
    }

    /// True when `q` lies inside the outer loop and outside every hole.
    #[must_use]
    pub fn contains(&self, q: brepkit_math::vec::Point2) -> bool {
        use crate::builder::classify_2d::point_in_polygon_2d;
        self.outer.len() >= 3
            && point_in_polygon_2d(q, &self.outer)
            && !self
                .holes
                .iter()
                .any(|h| h.len() >= 3 && point_in_polygon_2d(q, h))
    }

    /// Distance from `p` to the face's supporting surface.
    #[must_use]
    pub fn distance_to_surface(&self, p: Point3) -> Option<f64> {
        match &self.surface {
            FaceSurface::Plane { normal, d } => {
                Some((normal.dot(Vec3::new(p.x(), p.y(), p.z())) - d).abs())
            }
            surface => {
                let (u, v) = surface.project_point(p)?;
                surface.evaluate(u, v).map(|q| (q - p).length())
            }
        }
    }
}

/// The trimmed boundary of a solid, sampled once, for on-boundary queries.
///
/// A classifier's verdict for a point on the boundary is a coin toss (a ray
/// cast grazes the face), so the builder re-samples instead of trusting it.
pub struct BoundaryProbe {
    faces: Vec<FaceLoops2d>,
}

impl BoundaryProbe {
    /// Sample every face of `solid`.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] on a topology lookup failure.
    pub fn new(topo: &Topology, solid: SolidId) -> Result<Self, AlgoError> {
        let mut faces = Vec::new();
        for fid in brepkit_topology::explorer::solid_faces(topo, solid)? {
            faces.push(FaceLoops2d::new(topo, fid)?);
        }
        Ok(Self { faces })
    }

    /// True when `point` lies within `tol` of a face: on its supporting
    /// surface and inside the face's trimmed region.
    #[must_use]
    pub fn point_on_boundary(&self, point: Point3, tol: f64) -> bool {
        let p = [point.x(), point.y(), point.z()];
        self.faces.iter().any(|face| {
            (0..3).all(|a| p[a] >= face.aabb[a] - tol && p[a] <= face.aabb[a + 3] + tol)
                && face.distance_to_surface(point).is_some_and(|d| d <= tol)
                && face.to_uv(point).is_some_and(|q| face.contains(q))
        })
    }
}

/// Classify a planar sub-face that is coincident-coplanar with a face of the
/// opposing solid by 2D containment, bypassing the unstable grazing ray-cast.
///
/// When a split sub-face's supporting plane is coincident (coplanar within
/// `tol`, ignoring normal sign) with a planar face of the opposing solid, the
/// sub-face's interior point necessarily lies *on* that opposing face's plane.
/// A cardinal ray-cast from such a point grazes the coincident cap and its wall
/// top-edges and can vote wrongly Inside (and a single interior sample is
/// itself unreliable on a thin corner wedge).
///
/// The override fires only for the *wholly-exterior wedge* signature: the
/// sub-face has at least one vertex strictly outside the opposing region and
/// **no** vertex strictly inside it (every vertex is outside or on the shared
/// boundary) — the clipped-away corner orphan whose only contact with the
/// opposing region is along the shared boundary.
///
/// To stay sound it additionally runs a *depth probe* at the wedge tip: a 2D
/// point outside the opposing face's region is outside the opposing *solid*
/// only when this coincident plane is the local outer boundary there. Stepping
/// off the plane to both sides of the tip and finding the solid absent on both
/// sides confirms the plane is a local boundary → the wedge is exterior
/// ([`FaceClass::Outside`]). If the solid persists on either side (a plane
/// shared with an interior feature, e.g. the honeycomb's stacked caps), the
/// genuinely-inside coincident face is left to the regular classifier.
///
/// Returns `None` when there is no coincident opposing face, the sub-face is
/// not a wholly-exterior wedge, or the depth probe finds the plane is internal.
///
/// # Errors
///
/// Returns [`AlgoError`] on a topology lookup failure.
#[allow(clippy::too_many_arguments)]
pub fn classify_coincident_coplanar(
    topo: &Topology,
    opposing_solid: SolidId,
    geoms: Option<&ray_cast::RayCastGeoms>,
    sub_face_id: brepkit_topology::face::FaceId,
    sub_normal: Vec3,
    sub_d: f64,
    interior: Point3,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<FaceClass>, AlgoError> {
    let plane_tol = tol.linear.max(1e-7);
    let n_tol = 1e-6_f64;
    // The sub-face's own material region: probes must sample it, not just
    // avoid the opposing region. An annular sub-face's vertex centroid falls
    // in its own hole, and the hole can be genuinely open space (the spacer
    // foot plate's ring around the open foot cavity, #1570) — probing there
    // reads air-air and wrongly declares the buried band Outside.
    let own_region = planar_face_polygons(topo, sub_face_id)?;
    let faces = brepkit_topology::explorer::solid_faces(topo, opposing_solid)?;
    for fid in faces {
        let face = topo.face(fid)?;
        let FaceSurface::Plane {
            normal: fn_raw,
            d: fd_raw,
        } = face.surface()
        else {
            continue;
        };
        // The stored (normal, d) define the plane regardless of face
        // orientation; coincidence is sign-agnostic.
        let fnv = *fn_raw;
        let coplanar_same =
            (fnv - sub_normal).length() < n_tol && (fd_raw - sub_d).abs() < plane_tol;
        let coplanar_flip =
            (fnv + sub_normal).length() < n_tol && (fd_raw + sub_d).abs() < plane_tol;
        if !(coplanar_same || coplanar_flip) {
            continue;
        }
        let Some((outer, holes, region_normal)) = planar_face_polygons(topo, fid)? else {
            continue;
        };
        let Some(sub_verts) = sub_face_outer_vertices(topo, sub_face_id)? else {
            return Ok(None);
        };

        // Classify each sub-face vertex against the opposing region with a
        // boundary band: a vertex on the shared boundary (within `plane_tol`)
        // is neither strictly inside nor strictly outside. Track the deepest
        // strictly-outside vertex (farthest from the opposing boundary) — that
        // is the wedge tip, the most reliable place to probe.
        let mut any_strictly_inside = false;
        let mut deepest_outside: Option<(f64, Point3)> = None;
        for &v in &sub_verts {
            let dist = dist_to_polygon_boundary(v, &outer, &region_normal);
            if dist <= plane_tol {
                continue;
            }
            if point_in_planar_region(v, &outer, &holes, &region_normal) {
                any_strictly_inside = true;
            } else if deepest_outside.is_none_or(|(d, _)| dist > d) {
                deepest_outside = Some((dist, v));
            }
        }

        // Wholly-exterior wedge: outside-or-on everywhere, with real exterior
        // extent. A straddler (any strictly-inside vertex) is deferred.
        let Some((depth, tip)) = deepest_outside else {
            return Ok(None);
        };
        if any_strictly_inside {
            return Ok(None);
        }

        // Depth probe: a 2D point outside the opposing face's region is outside
        // the opposing *solid* only if this coincident plane is the local outer
        // boundary there — i.e. stepping off the plane to *both* sides leaves
        // the solid. (A plane shared with an interior feature, e.g. the
        // honeycomb's stacked caps, has solid on one side → defer to ray-cast,
        // which correctly keeps the genuinely-inside coincident face.)
        //
        // The wedge tip sits at the sub-face's outermost corner, which lies on
        // the shared walls — ray-cast grazes there. Nudge the probe location
        // off the tip toward the wedge centroid so it clears the walls, while
        // keeping it strictly outside the opposing 2D region.
        let nlen = region_normal.length();
        if nlen < 1e-12 {
            return Ok(None);
        }
        let np = region_normal * (1.0 / nlen);
        let (mut cx, mut cy, mut cz) = (0.0, 0.0, 0.0);
        for &v in &sub_verts {
            cx += v.x();
            cy += v.y();
            cz += v.z();
        }
        let inv = 1.0 / sub_verts.len() as f64;
        let centroid = Point3::new(cx * inv, cy * inv, cz * inv);
        let probe = (100.0 * plane_tol).max(1e-3);

        // Candidate probe locations along tip → centroid. The centroid fractions
        // cover the wedge from rim to interior (a partially-internal coincident
        // plane can persist at either end — the honeycomb's stacked cap persists
        // near the RIM); the small ABSOLUTE nudges scaled to the wedge's own
        // outside-extent `depth` stay near the tip inside the band and are the
        // ONLY valid probes on a thin annulus (a ~1.2mm lip on a ~125mm face),
        // where every centroid fraction jumps clear across the band into the hole
        // (the opposing 2D region) and is rejected — without them the band face
        // found no valid probe and was dropped.
        let mut candidates: Vec<Point3> = Vec::with_capacity(7);
        // The sub-face's sampled interior point is the one probe guaranteed
        // to lie on its own material — for an annulus it is the only one.
        candidates.push(interior);
        for frac in [0.25_f64, 0.4, 0.55] {
            candidates.push(tip + (centroid - tip) * frac);
        }
        let dir = centroid - tip;
        let dl = dir.length();
        if dl > 1e-12 {
            let dir_unit = dir * (1.0 / dl);
            for scale in [0.5_f64, 0.25, 0.1] {
                candidates.push(tip + dir_unit * (depth * scale).min(0.9 * dl));
            }
        }

        // Decide from ALL valid probes, order-independently: if ANY strictly-
        // outside probe finds the opposing solid persisting on a side, the plane
        // is internal there → defer (keep). Only when at least one probe is valid
        // and NONE show persistence is the plane a genuine local outer boundary →
        // Outside. (A first-valid-wins scan was order-fragile: it could accept a
        // both-sides-empty rim probe before reaching the honeycomb cap's interior
        // persistence, or vice-versa.)
        let mut any_valid = false;
        for probe_xy in candidates {
            // Must still be strictly outside the opposing region and clear of
            // its boundary, else the probe is meaningless.
            if point_in_planar_region(probe_xy, &outer, &holes, &region_normal)
                || dist_to_polygon_boundary(probe_xy, &outer, &region_normal) <= probe
            {
                continue;
            }
            // And it must sample the sub-face's OWN material: a probe in the
            // sub-face's hole says nothing about the face itself.
            if let Some((own_outer, own_holes, own_normal)) = &own_region
                && !point_in_planar_region(probe_xy, own_outer, own_holes, own_normal)
            {
                continue;
            }
            any_valid = true;
            let probe_a = probe_xy + np * probe;
            let probe_b = probe_xy - np * probe;
            let (av, bv) = match geoms {
                Some(g) => (
                    ray_cast::ray_cast_inside_votes_cached(g, probe_a)?,
                    ray_cast::ray_cast_inside_votes_cached(g, probe_b)?,
                ),
                None => (
                    ray_cast_inside_votes(topo, opposing_solid, probe_a)?,
                    ray_cast_inside_votes(topo, opposing_solid, probe_b)?,
                ),
            };
            if std::env::var("BK_COP").is_ok() {
                log::debug!(
                    "COP sub={sub_face_id:?} vs {fid:?}: tip=({:.3},{:.3},{:.3}) depth={depth:.3} probe=({:.3},{:.3},{:.3}) av={av} bv={bv}",
                    tip.x(),
                    tip.y(),
                    tip.z(),
                    probe_xy.x(),
                    probe_xy.y(),
                    probe_xy.z()
                );
            }
            if av >= 2 || bv >= 2 {
                // Solid persists on a side: internal plane → keep (defer).
                return Ok(None);
            }
        }
        if std::env::var("BK_COP").is_ok() {
            log::debug!(
                "COP sub={sub_face_id:?} vs {fid:?}: any_valid={any_valid} -> {:?}",
                any_valid.then_some(FaceClass::Outside)
            );
        }
        return Ok(any_valid.then_some(FaceClass::Outside));
    }
    Ok(None)
}

/// Minimum distance from `p` to the closed polyline `poly` (edges + wrap).
fn dist_to_polygon_boundary(p: Point3, poly: &[Point3], _normal: &Vec3) -> f64 {
    let n = poly.len();
    if n < 2 {
        return f64::INFINITY;
    }
    let mut best = f64::INFINITY;
    for i in 0..n {
        let a = poly[i];
        let b = poly[(i + 1) % n];
        let ab = b - a;
        let len2 = ab.dot(ab);
        let t = if len2 > 1e-18 {
            ((p - a).dot(ab) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let proj = a + ab * t;
        best = best.min((p - proj).length());
    }
    best
}

/// Collect a planar sub-face's outer-wire vertices (3D), de-duplicated.
fn sub_face_outer_vertices(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
) -> Result<Option<Vec<Point3>>, AlgoError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut verts = Vec::new();
    for oe in wire.edges() {
        let e = topo.edge(oe.edge())?;
        verts.push(topo.vertex(e.start())?.point());
        verts.push(topo.vertex(e.end())?.point());
    }
    if verts.len() < 3 {
        return Ok(None);
    }
    Ok(Some(verts))
}

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
    if let Some(class) = classify_analytic(topo, solid, point) {
        return Ok(class);
    }

    match geoms {
        Some(g) => ray_cast::classify_ray_cast_cached(g, point),
        None => classify_ray_cast(topo, solid, point),
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
pub fn classify_coincident_coplanar(
    topo: &Topology,
    opposing_solid: SolidId,
    geoms: Option<&ray_cast::RayCastGeoms>,
    sub_face_id: brepkit_topology::face::FaceId,
    sub_normal: Vec3,
    sub_d: f64,
    tol: brepkit_math::tolerance::Tolerance,
) -> Result<Option<FaceClass>, AlgoError> {
    let plane_tol = tol.linear.max(1e-7);
    let n_tol = 1e-6_f64;
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

        // Candidate probe locations, tip → centroid, DEEP first. The historical
        // centroid fractions are tried first so a partially-internal coincident
        // plane (the honeycomb's stacked caps: outer-boundary near the rim, but
        // solid persists deeper toward the centroid) is still probed at its deep
        // internal region and correctly deferred. Only when EVERY deep fraction
        // is invalid — the thin-annulus overshoot, where a fraction of the
        // tip→centroid distance jumps clear across a ~1.2mm band into the hole
        // (the opposing 2D region) and is rejected — fall back to small ABSOLUTE
        // nudges (a fraction of the wedge's own outside-extent `depth`) that
        // stay near the tip inside the band. Without the fallback the band face
        // found no valid probe and was left unclassified and dropped.
        let mut candidates: Vec<Point3> = Vec::new();
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

        let mut decided: Option<FaceClass> = None;
        for probe_xy in candidates {
            // Must still be strictly outside the opposing region and clear of
            // its boundary, else the probe is meaningless.
            if point_in_planar_region(probe_xy, &outer, &holes, &region_normal)
                || dist_to_polygon_boundary(probe_xy, &outer, &region_normal) <= probe
            {
                continue;
            }
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
            decided = Some(if av < 2 && bv < 2 {
                FaceClass::Outside
            } else {
                // Solid persists on a side: internal plane → keep (defer).
                return Ok(None);
            });
            break;
        }
        return Ok(decided);
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

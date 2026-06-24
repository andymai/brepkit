//! Ray-cast point-in-solid classification (canonical implementation).
//!
//! Shoots rays from a sample point and counts boundary crossings
//! to determine inside/outside status.
//!
//! NOTE: `operations/boolean/classify.rs` contains a duplicate of this
//! logic. Bug fixes should be applied here first; the operations copy
//! will be deleted during the GFA step 5 switchover.

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::builder::FaceClass;
use crate::error::AlgoError;

/// Per-face geometry used for ray crossing tests.
enum FaceGeom {
    /// A planar (or planar-approximated) face: boundary polygon, hole
    /// polygons, and the supporting plane.
    Planar {
        verts: Vec<Point3>,
        holes: Vec<Vec<Point3>>,
        normal: Vec3,
        d: f64,
    },
    /// A full-period cylindrical face (e.g. a bore lateral). Crossings are
    /// computed analytically — a flat polygon approximation counts one
    /// crossing where the real surface has two, flipping the parity.
    ///
    /// `hole_bands` are full-circumference v-ranges carved out of the lateral
    /// (a flush-cap interaction can leave such a holed lateral). A crossing
    /// whose axial parameter falls inside a hole band is excluded.
    Cylinder {
        surface: brepkit_math::surfaces::CylindricalSurface,
        v_min: f64,
        v_max: f64,
        hole_bands: Vec<(f64, f64)>,
        /// For a partial-arc patch (e.g. a rounded-rect corner quarter), the
        /// angular range NOT covered by the face — a crossing whose `u`
        /// (circumferential parameter) falls in this gap is off the patch and
        /// excluded. `None` for a full-period lateral.
        u_gap: Option<(f64, f64)>,
    },
}

/// Classify a point by ray casting against the solid's faces.
///
/// Shoots 3 rays (+Z, +X, +Y) and uses majority vote. A point is
/// inside if 2+ rays report an odd crossing count.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if classification is
/// indeterminate after multiple ray directions.
pub fn classify_ray_cast(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    let inside_votes = ray_cast_inside_votes(topo, solid, point)?;
    if inside_votes >= 2 {
        Ok(FaceClass::Inside)
    } else {
        Ok(FaceClass::Outside)
    }
}

/// Number of cardinal rays (of three) reporting an odd crossing count.
///
/// A point is classified inside when 2+ of the three rays agree; the raw count
/// distinguishes a confident 3-vote verdict from a grazing 2-vote tie.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if no face geometry is collected.
pub fn ray_cast_inside_votes(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
) -> Result<u8, AlgoError> {
    let face_data = collect_face_geoms(topo, solid)?;
    votes_from_geoms(&face_data, point)
}

/// Pre-collected ray-cast geometry for a solid, built once and reused across
/// many point classifications.
///
/// [`collect_face_geoms`] samples every face's wire into a polygon (and chains
/// the polylines), which is the dominant cost of a single classification. When
/// classifying many points against the same solid — every sub-face of a boolean
/// against the opposing solid — rebuilding it per point is O(faces) × O(points).
/// Building it once turns that into a single O(faces) pass.
pub struct RayCastGeoms {
    faces: Vec<FaceGeom>,
}

impl RayCastGeoms {
    /// Collect the ray-cast geometry for `solid` once.
    ///
    /// # Errors
    ///
    /// Returns [`AlgoError`] if a topology lookup fails.
    pub fn new(topo: &Topology, solid: SolidId) -> Result<Self, AlgoError> {
        Ok(Self {
            faces: collect_face_geoms(topo, solid)?,
        })
    }
}

/// Inside-vote count for `point` using pre-collected geometry.
///
/// Identical to [`ray_cast_inside_votes`] but skips the per-call geometry
/// collection.
///
/// # Errors
///
/// Returns [`AlgoError`] if no face geometry was collected.
pub fn ray_cast_inside_votes_cached(geoms: &RayCastGeoms, point: Point3) -> Result<u8, AlgoError> {
    votes_from_geoms(&geoms.faces, point)
}

/// Ray-cast classification for `point` using pre-collected geometry.
///
/// # Errors
///
/// Returns [`AlgoError`] if no face geometry was collected.
pub fn classify_ray_cast_cached(
    geoms: &RayCastGeoms,
    point: Point3,
) -> Result<FaceClass, AlgoError> {
    if votes_from_geoms(&geoms.faces, point)? >= 2 {
        Ok(FaceClass::Inside)
    } else {
        Ok(FaceClass::Outside)
    }
}

/// Count the inside votes (of three cardinal rays) for a point against
/// pre-collected face geometry.
fn votes_from_geoms(face_data: &[FaceGeom], point: Point3) -> Result<u8, AlgoError> {
    if face_data.is_empty() {
        return Err(AlgoError::ClassificationFailed(
            "no face polygons collected for ray-cast".into(),
        ));
    }

    let tol = Tolerance::new();
    let ray_dirs = [
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
    ];

    let mut inside_votes = 0u8;
    for ray_dir in &ray_dirs {
        let mut crossings = 0i32;
        for geom in face_data {
            crossings += ray_geom_crossings(point, *ray_dir, geom, tol);
        }
        if crossings % 2 != 0 {
            inside_votes += 1;
        }
    }
    Ok(inside_votes)
}

/// Sample a wire into a polygon by geometrically chaining its edges.
///
/// Wires are not guaranteed to list edges in traversal order (primitive
/// builders store edge sets), so each edge is sampled into a polyline and
/// the polylines are chained by endpoint matching. Closed curved edges
/// (full circles) get dense sampling; open curved edges get interior
/// samples for better coverage.
fn wire_polygon(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<Vec<Point3>, AlgoError> {
    let wire = topo.wire(wire_id)?;

    let mut polylines: Vec<Vec<Point3>> = Vec::with_capacity(wire.edges().len());
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let raw_start = topo.vertex(edge.start())?.point();
        let raw_end = topo.vertex(edge.end())?.point();
        let mut pts = vec![raw_start];
        if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
            let (t0, t1) = edge.curve().domain_with_endpoints(raw_start, raw_end);
            let is_closed = (raw_start - raw_end).length() < 1e-9;
            let n_samples = if is_closed { 16_i32 } else { 3_i32 };
            for k in 1..=n_samples {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(n_samples + 1);
                pts.push(edge.curve().evaluate_with_endpoints(t, raw_start, raw_end));
            }
        }
        pts.push(raw_end);
        if !oe.is_forward() {
            pts.reverse();
        }
        polylines.push(pts);
    }

    let join_tol = 1e-6;
    let mut used = vec![false; polylines.len()];
    let mut verts: Vec<Point3> = Vec::new();
    let Some(first) = polylines.first() else {
        return Ok(verts);
    };
    verts.extend_from_slice(first);
    used[0] = true;
    for _ in 1..polylines.len() {
        let tail = match verts.last() {
            Some(p) => *p,
            None => break,
        };
        let next = polylines.iter().enumerate().find_map(|(i, pl)| {
            if used[i] {
                return None;
            }
            let s = *pl.first()?;
            let e = *pl.last()?;
            if (s - tail).length() < join_tol {
                Some((i, false))
            } else if (e - tail).length() < join_tol {
                Some((i, true))
            } else {
                None
            }
        });
        let Some((idx, rev)) = next else { break };
        used[idx] = true;
        let mut pl = polylines[idx].clone();
        if rev {
            pl.reverse();
        }
        verts.extend_from_slice(&pl[1..]);
    }
    // Append any unchained polylines so no geometry is silently lost
    // (matches the previous behavior of emitting all edge samples).
    for (i, pl) in polylines.iter().enumerate() {
        if !used[i] {
            verts.extend_from_slice(pl);
        }
    }
    // Drop the duplicated closing point.
    if verts.len() >= 2 {
        let first_pt = verts[0];
        if let Some(last) = verts.last()
            && (*last - first_pt).length() < join_tol
        {
            verts.pop();
        }
    }
    Ok(verts)
}

/// Collect per-face ray-cast geometry from a solid.
fn collect_face_geoms(topo: &Topology, solid: SolidId) -> Result<Vec<FaceGeom>, AlgoError> {
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    let mut result = Vec::with_capacity(faces.len());

    for fid in faces {
        let face = topo.face(fid)?;

        // Full-period cylindrical faces: the outer wire contains a closed
        // circle edge, so the face wraps the entire circumference and the
        // analytic crossing test applies. Inner wires are accepted only when
        // each is a full-circumference v-band (the shape a flush-cap
        // interaction carves out); any non-banded hole forces the polygon
        // fallback. Partial cylinder patches also fall through.
        if let brepkit_topology::face::FaceSurface::Cylinder(cyl) = face.surface() {
            let wire = topo.wire(face.outer_wire())?;
            let mut has_closed_circle = false;
            for oe in wire.edges() {
                let edge = topo.edge(oe.edge())?;
                if matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Circle(_))
                    && edge.start() == edge.end()
                {
                    has_closed_circle = true;
                    break;
                }
            }
            if has_closed_circle {
                let verts = wire_polygon(topo, face.outer_wire())?;
                let mut v_min = f64::INFINITY;
                let mut v_max = f64::NEG_INFINITY;
                for p in &verts {
                    let (_, v) = cyl.project_point(*p);
                    v_min = v_min.min(v);
                    v_max = v_max.max(v);
                }
                let hole_bands = cylinder_hole_bands(topo, face, cyl)?;
                let holes_banded = hole_bands.len() == face.inner_wires().len();
                if v_min.is_finite() && v_max > v_min && holes_banded {
                    result.push(FaceGeom::Cylinder {
                        surface: cyl.clone(),
                        v_min,
                        v_max,
                        hole_bands,
                        u_gap: None,
                    });
                    continue;
                }
            }

            // Partial-arc cylinder patch (e.g. a rounded-rect corner quarter):
            // no closed-circle edge, so the full-period path skipped it.
            // Collect it analytically with an angular trim rather than the
            // polygon fallback, whose non-planar boundary mis-counts crossings.
            if face.inner_wires().is_empty() {
                let verts = wire_polygon(topo, face.outer_wire())?;
                if verts.len() >= 3 {
                    let mut pv_min = f64::INFINITY;
                    let mut pv_max = f64::NEG_INFINITY;
                    let mut u_samples = Vec::with_capacity(verts.len());
                    for p in &verts {
                        let (u, v) = cyl.project_point(*p);
                        pv_min = pv_min.min(v);
                        pv_max = pv_max.max(v);
                        u_samples.push(u);
                    }
                    if pv_min.is_finite()
                        && pv_max > pv_min
                        && let Some(gap) = largest_u_gap(&u_samples)
                    {
                        result.push(FaceGeom::Cylinder {
                            surface: cyl.clone(),
                            v_min: pv_min,
                            v_max: pv_max,
                            hole_bands: Vec::new(),
                            u_gap: Some(gap),
                        });
                        continue;
                    }
                }
            }
        }

        let verts = wire_polygon(topo, face.outer_wire())?;
        if verts.len() < 3 {
            continue;
        }

        let mut holes = Vec::with_capacity(face.inner_wires().len());
        for &iw in face.inner_wires() {
            let hole = wire_polygon(topo, iw)?;
            if hole.len() >= 3 {
                holes.push(hole);
            }
        }

        let raw_normal =
            if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
                *normal
            } else {
                newell_normal(&verts)
            };
        let normal = if face.is_reversed() {
            -raw_normal
        } else {
            raw_normal
        };

        let d = dot_normal_point(normal, verts[0]);
        result.push(FaceGeom::Planar {
            verts,
            holes,
            normal,
            d,
        });
    }

    Ok(result)
}

/// Collect full-circumference v-band holes carved out of a cylindrical face.
///
/// Each inner wire is sampled and projected into `(u, v)`. A wire is treated
/// as a band only when its u-samples wrap the full circumference; the band's
/// `[v_lo, v_hi]` is the projected axial span. Returns one entry per qualifying
/// inner wire — a count short of `face.inner_wires().len()` signals a
/// non-banded hole, which the caller uses to force the polygon fallback.
fn cylinder_hole_bands(
    topo: &Topology,
    face: &brepkit_topology::face::Face,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) -> Result<Vec<(f64, f64)>, AlgoError> {
    use std::f64::consts::TAU;

    let mut bands = Vec::with_capacity(face.inner_wires().len());
    for &iw in face.inner_wires() {
        let pts = wire_polygon(topo, iw)?;
        if pts.len() < 3 {
            continue;
        }
        let mut v_lo = f64::INFINITY;
        let mut v_hi = f64::NEG_INFINITY;
        let mut u_min = f64::INFINITY;
        let mut u_max = f64::NEG_INFINITY;
        for p in &pts {
            let (u, v) = cyl.project_point(*p);
            v_lo = v_lo.min(v);
            v_hi = v_hi.max(v);
            u_min = u_min.min(u);
            u_max = u_max.max(u);
        }
        // Only full-circumference bands qualify: a partial-arc hole would be
        // over-excluded by a v-band test.
        let wraps_full = (u_max - u_min) >= TAU - 1e-3;
        if wraps_full && v_hi > v_lo {
            bands.push((v_lo, v_hi));
        }
    }
    Ok(bands)
}

/// Count ray crossings against a face geometry.
#[inline]
fn ray_geom_crossings(origin: Point3, ray_dir: Vec3, geom: &FaceGeom, tol: Tolerance) -> i32 {
    match geom {
        FaceGeom::Planar {
            verts,
            holes,
            normal,
            d,
        } => ray_face_crossing(origin, ray_dir, verts, holes, *normal, *d, tol),
        FaceGeom::Cylinder {
            surface,
            v_min,
            v_max,
            hole_bands,
            u_gap,
        } => ray_cylinder_crossings(
            origin,
            ray_dir,
            surface,
            (*v_min, *v_max),
            hole_bands,
            *u_gap,
            tol,
        ),
    }
}

/// Test a single face polygon against a ray for crossing parity.
///
/// Returns +1 for a crossing, 0 for no intersection. Hits inside a hole
/// polygon do not count.
#[inline]
fn ray_face_crossing(
    origin: Point3,
    ray_dir: Vec3,
    verts: &[Point3],
    holes: &[Vec<Point3>],
    normal: Vec3,
    d: f64,
    tol: Tolerance,
) -> i32 {
    let denom = normal.dot(ray_dir);
    if denom.abs() < tol.angular {
        return 0;
    }
    let numer = d - dot_normal_point(normal, origin);
    let t = numer / denom;
    if t <= tol.linear {
        return 0;
    }
    let hit = Point3::new(
        origin.x() + ray_dir.x() * t,
        origin.y() + ray_dir.y() * t,
        origin.z() + ray_dir.z() * t,
    );
    if !point_in_face_3d(hit, verts, &normal) {
        return 0;
    }
    if holes.iter().any(|h| point_in_face_3d(hit, h, &normal)) {
        return 0;
    }
    1
}

/// Count ray crossings with a bounded full-period cylindrical face.
///
/// Solves the ray/infinite-cylinder quadratic and counts roots whose axial
/// parameter falls within the face's v-range but outside any `hole_bands`
/// (full-circumference v-ranges carved out of the lateral). Tangent grazes
/// (discriminant ≈ 0) count as zero crossings, which preserves parity.
fn ray_cylinder_crossings(
    origin: Point3,
    ray_dir: Vec3,
    surface: &brepkit_math::surfaces::CylindricalSurface,
    v_range: (f64, f64),
    hole_bands: &[(f64, f64)],
    u_gap: Option<(f64, f64)>,
    tol: Tolerance,
) -> i32 {
    let (v_min, v_max) = v_range;
    let axis = surface.axis();
    let m = origin - surface.origin();
    let d_perp = ray_dir - axis * ray_dir.dot(axis);
    let m_perp = m - axis * m.dot(axis);

    let a = d_perp.dot(d_perp);
    if a < 1e-14 {
        return 0;
    }
    let b = 2.0 * m_perp.dot(d_perp);
    let c = surface
        .radius()
        .mul_add(-surface.radius(), m_perp.dot(m_perp));
    let disc = b.mul_add(b, -4.0 * a * c);
    // Treat near-tangent rays as misses: counting one graze flips parity.
    if disc < 1e-12 * a * surface.radius() * surface.radius() {
        return 0;
    }
    let sqrt_disc = disc.sqrt();
    let mut crossings = 0;
    for t in [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)] {
        if t <= tol.linear {
            continue;
        }
        let hit = Point3::new(
            origin.x() + ray_dir.x() * t,
            origin.y() + ray_dir.y() * t,
            origin.z() + ray_dir.z() * t,
        );
        let v = axis.dot(hit - surface.origin());
        if v < v_min - tol.linear || v > v_max + tol.linear {
            continue;
        }
        if hole_bands
            .iter()
            .any(|&(lo, hi)| v > lo + tol.linear && v < hi - tol.linear)
        {
            continue;
        }
        // Angular trim for a partial-arc patch: skip a hit on the off-patch
        // portion of the full cylinder (the rounded-rect corner quarter only
        // covers a 90° arc; the other 3/4 is not a real face).
        if let Some(gap) = u_gap {
            let (u, _) = surface.project_point(hit);
            if u_in_gap(u, gap) {
                continue;
            }
        }
        crossings += 1;
    }
    crossings
}

/// Whether circumferential parameter `u` lies in the excluded angular gap
/// `(lo, hi)` (CCW from `lo` to `hi`, possibly wrapping past 2π).
pub fn u_in_gap(u: f64, gap: (f64, f64)) -> bool {
    use std::f64::consts::TAU;
    let eps = 1e-6;
    let u = u.rem_euclid(TAU);
    let (lo, hi) = (gap.0.rem_euclid(TAU), gap.1.rem_euclid(TAU));
    if lo <= hi {
        u > lo + eps && u < hi - eps
    } else {
        u > lo + eps || u < hi - eps
    }
}

/// Largest angular gap between sorted circumferential samples — the arc the
/// partial-cylinder face does NOT cover. `None` for too-few samples or a gap
/// too small to be a genuine partial arc.
pub fn largest_u_gap(u_samples: &[f64]) -> Option<(f64, f64)> {
    use std::f64::consts::TAU;
    let mut us: Vec<f64> = u_samples.iter().map(|&u| u.rem_euclid(TAU)).collect();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    us.dedup_by(|a, b| (*a - *b).abs() < 1e-9);
    if us.len() < 2 {
        return None;
    }
    let mut best = 0.0_f64;
    let mut gap = (0.0, 0.0);
    for i in 0..us.len() {
        let lo = us[i];
        let hi = if i + 1 < us.len() {
            us[i + 1]
        } else {
            us[0] + TAU
        };
        if hi - lo > best {
            best = hi - lo;
            gap = (lo, hi.rem_euclid(TAU));
        }
    }
    if best > 0.2 { Some(gap) } else { None }
}

/// Test if a 3D point lies inside a planar face polygon by projecting to 2D.
#[must_use]
pub fn point_in_face_3d(point: Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    if polygon.len() < 3 {
        return false;
    }

    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (project_point, project_polygon): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
        (
            Point2::new(point.x(), point.y()),
            polygon.iter().map(|p| Point2::new(p.x(), p.y())).collect(),
        )
    } else if ay >= ax {
        (
            Point2::new(point.x(), point.z()),
            polygon.iter().map(|p| Point2::new(p.x(), p.z())).collect(),
        )
    } else {
        (
            Point2::new(point.y(), point.z()),
            polygon.iter().map(|p| Point2::new(p.y(), p.z())).collect(),
        )
    };

    point_in_polygon(project_point, &project_polygon)
}

/// Compute `n . p` treating a `Point3` as a direction vector.
fn dot_normal_point(n: Vec3, p: Point3) -> f64 {
    n.dot(Vec3::new(p.x(), p.y(), p.z()))
}

/// A planar face's sampled boundary: `(outer_polygon, hole_polygons, normal)`.
pub type FacePolygons = (Vec<Point3>, Vec<Vec<Point3>>, Vec3);

/// Build a planar face's boundary as `(outer_polygon, hole_polygons, normal)`.
///
/// The outer wire and inner wires are sampled into polylines (arcs densified
/// via `wire_polygon`) so a rounded-corner cap's true region is captured.
/// Returns `None` if the outer polygon is degenerate (< 3 points).
///
/// # Errors
///
/// Returns [`AlgoError`] on a topology lookup failure.
pub fn planar_face_polygons(
    topo: &Topology,
    face_id: brepkit_topology::face::FaceId,
) -> Result<Option<FacePolygons>, AlgoError> {
    let face = topo.face(face_id)?;
    let verts = wire_polygon(topo, face.outer_wire())?;
    if verts.len() < 3 {
        return Ok(None);
    }
    let raw_normal =
        if let brepkit_topology::face::FaceSurface::Plane { normal, .. } = face.surface() {
            *normal
        } else {
            newell_normal(&verts)
        };
    let normal = if face.is_reversed() {
        -raw_normal
    } else {
        raw_normal
    };
    let mut holes = Vec::with_capacity(face.inner_wires().len());
    for &iw in face.inner_wires() {
        let hole = wire_polygon(topo, iw)?;
        if hole.len() >= 3 {
            holes.push(hole);
        }
    }
    Ok(Some((verts, holes, normal)))
}

/// Test whether `point` lies inside the planar face's region (inside the outer
/// polygon and outside every hole), projecting along the face normal.
#[must_use]
pub fn point_in_planar_region(
    point: Point3,
    outer: &[Point3],
    holes: &[Vec<Point3>],
    normal: &Vec3,
) -> bool {
    if !point_in_face_3d(point, outer, normal) {
        return false;
    }
    !holes.iter().any(|h| point_in_face_3d(point, h, normal))
}

/// Compute the solid-level AABB from boundary vertices.
///
/// # Errors
///
/// Returns [`AlgoError::ClassificationFailed`] if the solid has no boundary
/// vertices.
pub fn compute_solid_bbox(
    topo: &Topology,
    solid: SolidId,
) -> Result<brepkit_math::aabb::Aabb3, AlgoError> {
    let mut points = Vec::new();
    let faces = brepkit_topology::explorer::solid_faces(topo, solid)?;
    for fid in faces {
        let face = topo.face(fid)?;
        let wire = topo.wire(face.outer_wire())?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            let start_pos = topo.vertex(edge.start())?.point();
            let end_pos = topo.vertex(edge.end())?.point();
            points.push(start_pos);
            points.push(end_pos);
            // Curved edges can bulge beyond their endpoints
            if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
                let (t0, t1) = edge.curve().domain_with_endpoints(start_pos, end_pos);
                let t_mid = 0.5_f64.mul_add(t1 - t0, t0);
                let mid = edge
                    .curve()
                    .evaluate_with_endpoints(t_mid, start_pos, end_pos);
                points.push(mid);
            }
        }
    }
    brepkit_math::aabb::Aabb3::try_from_points(points)
        .ok_or_else(|| AlgoError::ClassificationFailed("solid has no boundary vertices".into()))
}

/// Compute polygon normal via Newell's method.
fn newell_normal(verts: &[Point3]) -> Vec3 {
    let n = verts.len();
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    for i in 0..n {
        let curr = verts[i];
        let next = verts[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len > 1e-15 {
        Vec3::new(nx / len, ny / len, nz / len)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Build a degenerate solid where all faces have < 3 vertices
    /// (single-edge faces). This tests the empty polygon fallback.
    fn make_degenerate_solid(topo: &mut Topology) -> brepkit_topology::solid::SolidId {
        // Create a "solid" with a single face that has only 2 vertices
        // (a degenerate line edge). This will produce < 3 polygon vertices.
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let e01 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e10 = topo.add_edge(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(e01, true), OrientedEdge::new(e10, true)],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn empty_face_polygons_returns_error() {
        let mut topo = Topology::default();
        let solid = make_degenerate_solid(&mut topo);

        let result = classify_ray_cast(&topo, solid, Point3::new(0.5, 0.5, 0.5));
        assert!(
            result.is_err(),
            "ray-cast with no valid face polygons should return Err, got {result:?}"
        );
    }

    /// Build a unit box for classification tests.
    fn make_box(
        topo: &mut Topology,
        min: [f64; 3],
        max: [f64; 3],
    ) -> brepkit_topology::solid::SolidId {
        let [x0, y0, z0] = min;
        let [x1, y1, z1] = max;
        let v = [
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z0), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y0, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x1, y1, z1), 1e-7)),
            topo.add_vertex(Vertex::new(Point3::new(x0, y1, z1), 1e-7)),
        ];
        let mut edge = |a: usize, b: usize| -> brepkit_topology::edge::EdgeId {
            topo.add_edge(Edge::new(v[a], v[b], EdgeCurve::Line))
        };
        let e01 = edge(0, 1);
        let e12 = edge(1, 2);
        let e23 = edge(2, 3);
        let e30 = edge(3, 0);
        let e45 = edge(4, 5);
        let e56 = edge(5, 6);
        let e67 = edge(6, 7);
        let e74 = edge(7, 4);
        let e04 = edge(0, 4);
        let e15 = edge(1, 5);
        let e26 = edge(2, 6);
        let e37 = edge(3, 7);

        let fwd = |eid| OrientedEdge::new(eid, true);
        let rev = |eid| OrientedEdge::new(eid, false);
        let w_bot =
            topo.add_wire(Wire::new(vec![rev(e01), rev(e30), rev(e23), rev(e12)], true).unwrap());
        let w_top =
            topo.add_wire(Wire::new(vec![fwd(e45), fwd(e56), fwd(e67), fwd(e74)], true).unwrap());
        let w_front =
            topo.add_wire(Wire::new(vec![fwd(e01), fwd(e15), rev(e45), rev(e04)], true).unwrap());
        let w_back =
            topo.add_wire(Wire::new(vec![fwd(e23), fwd(e37), rev(e67), rev(e26)], true).unwrap());
        let w_left =
            topo.add_wire(Wire::new(vec![fwd(e30), fwd(e04), rev(e74), rev(e37)], true).unwrap());
        let w_right =
            topo.add_wire(Wire::new(vec![fwd(e12), fwd(e26), rev(e56), rev(e15)], true).unwrap());

        let mk_face =
            |w, n: Vec3, d: f64| Face::new(w, vec![], FaceSurface::Plane { normal: n, d });
        let faces = vec![
            topo.add_face(mk_face(w_bot, Vec3::new(0.0, 0.0, -1.0), -z0)),
            topo.add_face(mk_face(w_top, Vec3::new(0.0, 0.0, 1.0), z1)),
            topo.add_face(mk_face(w_front, Vec3::new(0.0, -1.0, 0.0), -y0)),
            topo.add_face(mk_face(w_back, Vec3::new(0.0, 1.0, 0.0), y1)),
            topo.add_face(mk_face(w_left, Vec3::new(-1.0, 0.0, 0.0), -x0)),
            topo.add_face(mk_face(w_right, Vec3::new(1.0, 0.0, 0.0), x1)),
        ];
        let shell = topo.add_shell(Shell::new(faces).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn ray_cast_classifies_inside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(1.0, 1.0, 1.0)).unwrap();
        assert_eq!(result, FaceClass::Inside, "center of box should be Inside");
    }

    #[test]
    fn ray_cast_classifies_outside_point() {
        let mut topo = Topology::default();
        let solid = make_box(&mut topo, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);

        let result = classify_ray_cast(&topo, solid, Point3::new(5.0, 5.0, 5.0)).unwrap();
        assert_eq!(
            result,
            FaceClass::Outside,
            "point far from box should be Outside"
        );
    }
}

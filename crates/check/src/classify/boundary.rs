//! UV boundary polygon construction and containment tests for face trimming.
//!
//! Provides the core algorithms for determining whether a ray-surface hit
//! point falls within a face's trimming boundary, using UV-space projection
//! for analytic surfaces and 3D polygon containment for surfaces with
//! pole singularities (spheres).

use std::f64::consts::PI;

use smallvec::SmallVec;

use brepkit_math::predicates::point_in_polygon;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::CheckError;
use crate::classify::ray_surface;
use crate::util::{face_polygon, point_in_polygon_3d};

/// Minimum positive ray parameter to count as a forward hit.
const RAY_T_MIN: f64 = 1e-12;

/// Threshold for half-space sign test (negative side rejection).
const HALF_SPACE_EPS: f64 = 1e-10;

/// Threshold for coincident vertex detection (squared distance).
const COINCIDENT_SQ: f64 = 1e-12;

/// Unwrap a step in a periodic (angular) coordinate so the difference
/// lies in `[-PI, PI)`.
///
/// Given the previous unwrapped value `prev` and the next raw value `next`,
/// returns the next value adjusted so the step is continuous.
#[inline]
fn unwrap_angle(prev: f64, next: f64) -> f64 {
    let tau = std::f64::consts::TAU;
    let diff = next - prev;
    prev + diff - tau * ((diff + PI) / tau).floor()
}

/// Build a UV boundary polygon from 3D face boundary vertices,
/// with proper unwrapping of periodic coordinates.
///
/// `v_periodic`: whether the v-coordinate is periodic (e.g. torus). Cylinder
/// and cone have linear v (height / distance), so only u is unwrapped for them.
fn build_uv_boundary<F>(verts: &[Point3], project: &F, v_periodic: bool) -> Vec<(f64, f64)>
where
    F: Fn(Point3) -> (f64, f64),
{
    let mut uv: Vec<(f64, f64)> = verts.iter().map(|&p| project(p)).collect();

    for i in 1..uv.len() {
        // u is always periodic (angular coordinate for all analytic surfaces).
        uv[i].0 = unwrap_angle(uv[i - 1].0, uv[i].0);

        // v is periodic only for doubly-periodic surfaces (torus).
        if v_periodic {
            uv[i].1 = unwrap_angle(uv[i - 1].1, uv[i].1);
        }
    }

    uv
}

/// Test if a (u,v) point is inside the UV boundary polygon.
///
/// Adjusts the test point's u coordinate (and v when periodic) to lie within
/// the unwrapped polygon's coordinate range before testing.
fn point_in_uv_boundary(
    hit_u: f64,
    hit_v: f64,
    uv_boundary: &[(f64, f64)],
    v_periodic: bool,
) -> bool {
    let u_min = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::INFINITY, f64::min);
    let u_max = uv_boundary
        .iter()
        .map(|(u, _)| *u)
        .fold(f64::NEG_INFINITY, f64::max);
    let u_center = (u_min + u_max) * 0.5;

    // Shift hit_u to be closest to the polygon's u center.
    let hu = unwrap_angle(u_center, hit_u);

    // For doubly-periodic surfaces (torus), also shift hit_v.
    let hv = if v_periodic {
        let v_min = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::INFINITY, f64::min);
        let v_max = uv_boundary
            .iter()
            .map(|(_, v)| *v)
            .fold(f64::NEG_INFINITY, f64::max);
        let v_center = (v_min + v_max) * 0.5;
        unwrap_angle(v_center, hit_v)
    } else {
        hit_v
    };

    let poly: Vec<Point2> = uv_boundary
        .iter()
        .map(|(u, v)| Point2::new(*u, *v))
        .collect();
    let test = Point2::new(hu, hv);
    point_in_polygon(test, &poly)
}

/// Compute the normal of a polygon via Newell's method.
///
/// Returns a unit-length normal, or `(0,0,1)` for degenerate polygons.
pub fn polygon_normal(verts: &[Point3]) -> Vec3 {
    crate::util::polygon_normal(verts)
}

/// Whether `point`, projected along `normal`, lies inside `polygon`
/// projected the same way.
fn point_in_polygon_along(point: &Point3, polygon: &[Point3], normal: Vec3) -> bool {
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(polygon[0], normal) else {
        return false;
    };
    let flat = |p: Point3| {
        let d = p - frame.origin;
        Point2::new(d.dot(frame.x), d.dot(frame.y))
    };
    let flat_poly: Vec<Point2> = polygon.iter().map(|&p| flat(p)).collect();
    point_in_polygon(flat(*point), &flat_poly)
}

/// Whether a hit on a sphere face lands in one of its holes. A hole in one
/// plane is the sphere's part beyond that plane, away from the face (whose
/// outer loop lies on the near side); any other hole is tested by polygon,
/// projected along the outer loop's `normal`.
fn hit_in_sphere_hole(
    topo: &Topology,
    face_id: FaceId,
    hit: Point3,
    outer: &[Point3],
    normal: Vec3,
) -> Result<bool, CheckError> {
    for &iw in topo.face(face_id)?.inner_wires() {
        let hole = crate::util::wire_polygon(topo, iw)?;
        if hole.len() < 3 {
            continue;
        }
        let hole_normal = polygon_normal(&hole);
        let in_hole = if loop_is_planar(&hole, hole_normal) {
            let at = hole[0];
            let near = outer
                .iter()
                .map(|p| (*p - at).dot(hole_normal))
                .fold(0.0_f64, |a, d| if d.abs() > a.abs() { d } else { a });
            let side = (hit - at).dot(hole_normal);
            near != 0.0 && side * near.signum() < -HALF_SPACE_EPS
        } else {
            point_in_polygon_along(&hit, &hole, normal)
        };
        if in_hole {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether a loop lies in the plane through its first point with `normal`.
fn loop_is_planar(pts: &[Point3], normal: Vec3) -> bool {
    let scale = pts
        .iter()
        .map(|p| (*p - pts[0]).length())
        .fold(0.0, f64::max);
    pts.iter()
        .all(|p| (*p - pts[0]).dot(normal).abs() <= 1e-9 * scale)
}

/// Whether a hit inside the outer wire actually lands in one of the face's
/// holes.
///
/// A ray leaving a solid through the mouth of a pocket passes through the hole
/// of the ring face around it. Without this test that hole counts as a
/// crossing, and the extra count flips the parity: an open pocket reads as
/// solid material.
fn hit_in_inner_wire_3d(
    topo: &Topology,
    face_id: FaceId,
    hit: Point3,
    normal: &Vec3,
) -> Result<bool, CheckError> {
    for &iw in topo.face(face_id)?.inner_wires() {
        let hole = crate::util::wire_polygon(topo, iw)?;
        if hole.len() >= 3 && point_in_polygon_3d(&hit, &hole, normal) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// UV-space counterpart of [`hit_in_inner_wire_3d`] for curved faces.
fn hit_in_inner_wire_uv<F>(
    topo: &Topology,
    face_id: FaceId,
    hit_u: f64,
    hit_v: f64,
    project: &F,
    v_periodic: bool,
) -> Result<bool, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    for &iw in topo.face(face_id)?.inner_wires() {
        let hole = crate::util::wire_polygon(topo, iw)?;
        if hole.len() < 3 {
            continue;
        }
        let uv_hole = build_uv_boundary(&hole, project, v_periodic);
        if point_in_uv_boundary(hit_u, hit_v, &uv_hole, v_periodic) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Count crossings for analytic (non-planar) faces using UV containment.
///
/// Given ray parameter roots (where the ray hits the infinite surface),
/// checks whether each hit point falls within the face's trimming boundary
/// by projecting to the surface's (u,v) parameter space.
///
/// If the face boundary is degenerate (all vertices coincide, as in a full
/// torus face with seam edges), every positive-t root outside the face's
/// holes is counted as a crossing.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
#[allow(clippy::too_many_arguments)]
fn count_analytic_crossings<F>(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &SmallVec<[f64; 4]>,
    project: F,
    v_periodic: bool,
    apex: Option<Point3>,
) -> Result<u32, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    if roots.is_empty() {
        return Ok(0);
    }

    let verts = face_polygon(topo, face_id)?;

    // Detect degenerate boundary: a "full-surface" face whose wire has fewer
    // than 3 distinct vertices. Every positive-t root is a crossing.
    let is_full_surface = verts.len() < 3 || {
        let ref_pt = verts[0];
        verts
            .iter()
            .all(|v| (*v - ref_pt).length_squared() < COINCIDENT_SQ)
    };
    if is_full_surface {
        let mut crossings = 0u32;
        for &t in roots.iter().filter(|&&t| t > RAY_T_MIN) {
            let (hit_u, hit_v) = project(origin + direction * t);
            if !hit_in_inner_wire_uv(topo, face_id, hit_u, hit_v, &project, v_periodic)? {
                crossings += 1;
            }
        }
        return Ok(crossings);
    }

    let mut uv_boundary = build_uv_boundary(&verts, &project, v_periodic);
    // A pointed cone's wire runs up its seam to the apex and straight back,
    // which bounds nothing in (u, v): its region is the rim's run closed
    // along the apex row, as a pole closes a sphere cap.
    // The wire may start anywhere on it, so the samples are turned to end at
    // the apex first.
    if let Some(apex) = apex
        && let Some(turn) = verts
            .iter()
            .position(|v| (*v - apex).length_squared() < COINCIDENT_SQ)
        && verts.len() >= 4
    {
        let mut rim = verts.clone();
        rim.rotate_left(turn + 1);
        rim.pop();
        uv_boundary = build_uv_boundary(&rim, &project, v_periodic);
        let (_, v_apex) = project(apex);
        let first_u = uv_boundary[0].0;
        let last_u = uv_boundary[uv_boundary.len() - 1].0;
        uv_boundary.push((last_u, v_apex));
        uv_boundary.push((first_u, v_apex));
    }

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;
        let (hit_u, hit_v) = project(hit);

        if point_in_uv_boundary(hit_u, hit_v, &uv_boundary, v_periodic)
            && !hit_in_inner_wire_uv(topo, face_id, hit_u, hit_v, &project, v_periodic)?
        {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Count crossings using 3D polygon containment (for faces with planar
/// boundaries, e.g. sphere hemispheres where UV projection has pole
/// singularities).
///
/// The polygon normal (from Newell's method) indicates which side of the
/// boundary plane the face extends into. A hit point must be on that side
/// AND project inside the boundary polygon.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
fn count_3d_polygon_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    roots: &SmallVec<[f64; 4]>,
) -> Result<u32, CheckError> {
    if roots.is_empty() {
        return Ok(0);
    }

    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }
    // The wire runs about the sphere's outward normal on a reversed face too,
    // so its polygon normal points to the face's side of the boundary plane.
    let normal = polygon_normal(&verts);
    let ref_pt = verts[0];
    // A loop in one plane bounds exactly the sphere's part on its side, at
    // any size; a polygon test would only add the chords' sagitta and miss a
    // cap larger than a hemisphere.
    let planar = loop_is_planar(&verts, normal);

    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let hit = origin + direction * t;

        // The hit must be on the face's side of the boundary plane.
        let side = (hit - ref_pt).dot(normal);
        if side < -HALF_SPACE_EPS {
            continue;
        }

        // Projected along the loop's own normal, not the nearest world axis:
        // a tilted face is not a graph over an axis plane, and the part of
        // it past the axis's silhouette projects outside its own boundary.
        let in_outer = planar || point_in_polygon_along(&hit, &verts, normal);
        if in_outer && !hit_in_sphere_hole(topo, face_id, hit, &verts, normal)? {
            crossings += 1;
        }
    }

    Ok(crossings)
}

/// Count ray crossings for a single face, dispatching by surface type.
///
/// For plane faces, uses direct ray-plane + 3D polygon containment.
/// For analytic curved faces, uses ray-surface intersection + UV containment.
/// For sphere faces, uses 3D polygon containment (avoids UV pole singularity).
/// For NURBS faces, uses line-surface intersection.
///
/// # Errors
///
/// Returns an error if topology lookups or intersection computations fail.
#[allow(clippy::too_many_lines)]
pub fn count_face_ray_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
) -> Result<u32, CheckError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, d } => {
            ray_plane_crossings(topo, face_id, origin, direction, *normal, *d)
        }
        FaceSurface::Cylinder(cyl) => {
            let cyl = cyl.clone();
            let roots = ray_surface::ray_cylinder(origin, direction, &cyl);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cyl.project_point(p),
                false,
                None,
            )
        }
        FaceSurface::Cone(cone) => {
            let cone = cone.clone();
            let roots = ray_surface::ray_cone(origin, direction, &cone);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| cone.project_point(p),
                false,
                Some(cone.apex()),
            )
        }
        FaceSurface::Sphere(sph) => {
            // Sphere boundaries are planar (equator, small circles), so
            // point_in_polygon_3d works. UV projection fails at poles.
            let sph = sph.clone();
            let roots = ray_surface::ray_sphere(origin, direction, &sph);
            count_3d_polygon_crossings(topo, face_id, origin, direction, &roots)
        }
        FaceSurface::Torus(tor) => {
            let tor = tor.clone();
            let roots = ray_surface::ray_torus(origin, direction, &tor);
            count_analytic_crossings(
                topo,
                face_id,
                origin,
                direction,
                &roots,
                |p| tor.project_point(p),
                true,
                None,
            )
        }
        FaceSurface::Nurbs(surface) => {
            ray_crossings_nurbs(topo, face_id, origin, direction, surface)
        }
    }
}

/// Ray-plane intersection with point-in-polygon boundary test.
fn ray_plane_crossings(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    normal: Vec3,
    d: f64,
) -> Result<u32, CheckError> {
    let t = match ray_surface::ray_plane(origin, direction, normal, d) {
        Some(t) => t,
        None => return Ok(0),
    };

    let hit = origin + direction * t;
    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        return Ok(0);
    }

    if point_in_polygon_3d(&hit, &verts, &normal)
        && !hit_in_inner_wire_3d(topo, face_id, hit, &normal)?
    {
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Count ray crossings for a NURBS face using ray-surface intersection.
fn ray_crossings_nurbs(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    direction: Vec3,
    surface: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<u32, CheckError> {
    let hits = ray_surface::ray_nurbs(origin, direction, surface, 20)?;
    if hits.is_empty() {
        return Ok(0);
    }

    let verts = face_polygon(topo, face_id)?;
    if verts.len() < 3 {
        // Full-surface face — every forward hit is a crossing.
        #[allow(clippy::cast_possible_truncation)]
        return Ok(hits.len() as u32);
    }

    let project = |p: Point3| -> (f64, f64) { surface.project_point(p) };
    let uv_boundary = build_uv_boundary(&verts, &project, false);

    let mut crossings = 0u32;
    for (_, hit_u, hit_v) in &hits {
        if point_in_uv_boundary(*hit_u, *hit_v, &uv_boundary, false)
            && !hit_in_inner_wire_uv(topo, face_id, *hit_u, *hit_v, &project, false)?
        {
            crossings += 1;
        }
    }

    Ok(crossings)
}

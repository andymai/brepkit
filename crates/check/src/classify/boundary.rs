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

fn loop_is_planar(pts: &[Point3], normal: Vec3) -> bool {
    let extent = loop_extent(pts);
    pts.iter()
        .all(|p| (*p - pts[0]).dot(normal).abs() <= 1e-9 * extent)
}

/// A wire whose every edge runs out and back as often (a seam, with no
/// rim) bounds nothing. A band's two rims can cancel each other's vector
/// area, so the area cannot tell; the wire's own edge uses can. An edge
/// closing on its start at a point (a pole) is skipped.
fn wire_runs_out_and_back(
    topo: &Topology,
    wire: brepkit_topology::wire::WireId,
) -> Result<bool, CheckError> {
    let mut runs: Vec<(brepkit_topology::edge::EdgeId, i32)> = Vec::new();
    for oe in topo.wire(wire)?.edges() {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?.point();
        if edge.start() == edge.end() {
            // A closed rim passes its vertex once, and one sample could
            // land there.
            let (t0, t1) = edge.curve().domain_with_endpoints(start, start);
            let at_vertex = [0.25, 0.5, 0.75].iter().all(|f| {
                let p =
                    edge.curve()
                        .evaluate_with_endpoints((t1 - t0).mul_add(*f, t0), start, start);
                (p - start).length() <= brepkit_math::tolerance::Tolerance::new().linear
            });
            if at_vertex {
                continue;
            }
        }
        let step = if oe.is_forward() { 1 } else { -1 };
        match runs.iter_mut().find(|(id, _)| *id == oe.edge()) {
            Some((_, n)) => *n += step,
            None => runs.push((oe.edge(), step)),
        }
    }
    Ok(!runs.is_empty() && runs.iter().all(|&(_, n)| n == 0))
}

fn loop_extent(pts: &[Point3]) -> f64 {
    pts.iter()
        .map(|p| (*p - pts[0]).length())
        .fold(0.0, f64::max)
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
    let region = uv_region(topo, face_id, &project, v_periodic, apex)?;
    let mut crossings = 0u32;
    for &t in roots {
        if t <= RAY_T_MIN {
            continue;
        }
        let (hit_u, hit_v) = project(origin + direction * t);
        if uv_region_contains(&region, topo, face_id, (hit_u, hit_v), &project, v_periodic)? {
            crossings += 1;
        }
    }
    Ok(crossings)
}

/// A curved face's outer region in its `(u, v)`.
enum UvRegion {
    /// The whole surface: a wire of fewer than three distinct points.
    Whole,
    /// The outer loop's samples.
    Bounded(Vec<(f64, f64)>),
}

/// The outer region of a cylinder, cone or torus face in its `(u, v)`.
fn uv_region<F>(
    topo: &Topology,
    face_id: FaceId,
    project: &F,
    v_periodic: bool,
    apex: Option<Point3>,
) -> Result<UvRegion, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    let verts = face_polygon(topo, face_id)?;

    // Detect degenerate boundary: a "full-surface" face whose wire has fewer
    // than 3 distinct vertices.
    let is_full_surface = verts.len() < 3 || {
        let ref_pt = verts[0];
        verts
            .iter()
            .all(|v| (*v - ref_pt).length_squared() < COINCIDENT_SQ)
    };
    if is_full_surface {
        return Ok(UvRegion::Whole);
    }

    let mut uv_boundary = build_uv_boundary(&verts, project, v_periodic);
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
        uv_boundary = build_uv_boundary(&rim, project, v_periodic);
        let (_, v_apex) = project(apex);
        let first_u = uv_boundary[0].0;
        let last_u = uv_boundary[uv_boundary.len() - 1].0;
        uv_boundary.push((last_u, v_apex));
        uv_boundary.push((first_u, v_apex));
    }
    Ok(UvRegion::Bounded(uv_boundary))
}

/// Whether a point at `(u, v)` lies in the region and outside the face's
/// holes.
fn uv_region_contains<F>(
    region: &UvRegion,
    topo: &Topology,
    face_id: FaceId,
    (u, v): (f64, f64),
    project: &F,
    v_periodic: bool,
) -> Result<bool, CheckError>
where
    F: Fn(Point3) -> (f64, f64),
{
    let in_outer = match region {
        UvRegion::Whole => true,
        UvRegion::Bounded(boundary) => point_in_uv_boundary(u, v, boundary, v_periodic),
    };
    Ok(in_outer && !hit_in_inner_wire_uv(topo, face_id, u, v, project, v_periodic)?)
}

/// A `(u, v)` loop's enclosed area, by the shoelace sum.
fn uv_area(loop_uv: &[(f64, f64)]) -> f64 {
    let n = loop_uv.len();
    (0..n)
        .map(|i| {
            let (a, b) = (loop_uv[i], loop_uv[(i + 1) % n]);
            a.0.mul_add(b.1, -(b.0 * a.1))
        })
        .sum::<f64>()
        .abs()
        / 2.0
}

/// The area of a `(u, v)` loop's bounding box.
fn uv_extent(loop_uv: &[(f64, f64)]) -> f64 {
    let (mut lo, mut hi) = (
        (f64::INFINITY, f64::INFINITY),
        (f64::NEG_INFINITY, f64::NEG_INFINITY),
    );
    for &(u, v) in loop_uv {
        lo = (lo.0.min(u), lo.1.min(v));
        hi = (hi.0.max(u), hi.1.max(v));
    }
    (hi.0 - lo.0).max(0.0) * (hi.1 - lo.1).max(0.0)
}

/// Whether `p`, a point on a face's surface, lies on the face: inside its
/// outer loop and outside its holes, read as the ray-cast classifier reads a
/// hit.
///
/// # Errors
///
/// Returns an error if topology lookups fail.
pub fn face_contains(topo: &Topology, face_id: FaceId, p: Point3) -> Result<bool, CheckError> {
    let face = topo.face(face_id)?;
    match face.surface() {
        FaceSurface::Plane { normal, .. } => {
            if let Some(inside) = plane_hit_inside(topo, face_id, p, *normal)? {
                return Ok(inside);
            }
            let verts = face_polygon(topo, face_id)?;
            Ok(verts.len() >= 3
                && point_in_polygon_3d(&p, &verts, normal)
                && !hit_in_inner_wire_3d(topo, face_id, p, normal)?)
        }
        FaceSurface::Cylinder(cyl) => {
            let project = |q: Point3| cyl.project_point(q);
            let region = uv_region(topo, face_id, &project, false, None)?;
            uv_region_contains(&region, topo, face_id, project(p), &project, false)
        }
        FaceSurface::Cone(cone) => {
            let project = |q: Point3| cone.project_point(q);
            let region = uv_region(topo, face_id, &project, false, Some(cone.apex()))?;
            uv_region_contains(&region, topo, face_id, project(p), &project, false)
        }
        FaceSurface::Torus(tor) => {
            let project = |q: Point3| tor.project_point(q);
            let region = uv_region(topo, face_id, &project, true, None)?;
            uv_region_contains(&region, topo, face_id, project(p), &project, true)
        }
        FaceSurface::Sphere(_) => match SphereRegion::of(topo, face_id)? {
            Some(region) => region.contains(topo, face_id, p),
            None => Ok(true),
        },
        FaceSurface::Nurbs(surface) => {
            let project = |q: Point3| -> (f64, f64) { surface.project_point(q) };
            let verts = face_polygon(topo, face_id)?;
            let (u, v) = project(p);
            let boundary = build_uv_boundary(&verts, &project, false);
            // A closed surface's seam copies project to one `u`, folding its
            // loop flat: such a face is the whole surface.
            let in_outer = verts.len() < 3
                || uv_area(&boundary) <= 1e-9 * uv_extent(&boundary)
                || point_in_uv_boundary(u, v, &boundary, false);
            Ok(in_outer && !hit_in_inner_wire_uv(topo, face_id, u, v, &project, false)?)
        }
    }
}

/// A sphere face's region, read in 3D from its outer loop (a sphere's
/// `(u, v)` is singular at its poles).
///
/// The outer loop's Newell normal points to the face's side of the loop. A
/// loop in one plane bounds exactly the sphere's part on that side; any
/// other loop of lines and circles is read by the parity of great-circle
/// arcs ([`SphereRims`]), and one with other curves bounds the points that
/// project inside it along that normal. A wire that only runs a seam out
/// and back bounds nothing, and the face is the whole sphere. Holes come
/// off by [`hit_in_sphere_hole`] where the parity does not count them.
pub struct SphereRegion {
    outer: Vec<Point3>,
    normal: Vec3,
    whole: bool,
    planar: bool,
    /// Set for a loop in no one plane whose edges are all lines and circles.
    rims: Option<SphereRims>,
}

impl SphereRegion {
    /// `None` when the outer loop samples to fewer than three points.
    pub fn of(topo: &Topology, face_id: FaceId) -> Result<Option<Self>, CheckError> {
        let outer = face_polygon(topo, face_id)?;
        if outer.len() < 3 {
            return Ok(None);
        }
        // The wire runs about the sphere's outward normal on a reversed face
        // too, so its polygon normal points to the face's side of the loop.
        let normal = polygon_normal(&outer);
        let whole = wire_runs_out_and_back(topo, topo.face(face_id)?.outer_wire())?;
        // A loop in one plane bounds exactly the sphere's part on its side,
        // at any size; a polygon test would only add the chords' sagitta and
        // miss a cap larger than a hemisphere.
        let planar = loop_is_planar(&outer, normal);
        let rims = if whole || planar {
            None
        } else {
            SphereRims::of(topo, face_id)?
        };
        Ok(Some(Self {
            outer,
            normal,
            whole,
            planar,
            rims,
        }))
    }

    /// Whether `p`, a point on the sphere, lies on the face.
    pub fn contains(
        &self,
        topo: &Topology,
        face_id: FaceId,
        p: Point3,
    ) -> Result<bool, CheckError> {
        if let Some(inside) = self.rims.as_ref().and_then(|rims| rims.contains(p)) {
            return Ok(inside);
        }
        // Projected along the loop's own normal, not the nearest world axis:
        // a tilted face is not a graph over an axis plane, and the part of it
        // past the axis's silhouette projects outside its own boundary.
        let in_outer = self.whole
            || ((p - self.outer[0]).dot(self.normal) >= -HALF_SPACE_EPS
                && (self.planar || point_in_polygon_along(&p, &self.outer, self.normal)));
        Ok(in_outer && !hit_in_sphere_hole(topo, face_id, p, &self.outer, self.normal)?)
    }
}

/// Angular band, in radians, within which a crossing touches a vertex, an
/// end of the test arc or a rim tangentially, leaving its parity unread.
const SPHERE_GRAZE: f64 = 1e-9;

/// A sphere face's edges read on the sphere, holes included, and points
/// just inside its outer wire. A circle is an arc of itself; a line is a
/// chord standing for the great-circle arc it projects to from the centre.
struct SphereRims {
    center: Point3,
    radius: f64,
    rims: Vec<SphereRim>,
    inside: Vec<Point3>,
}

enum SphereRim {
    /// Centre, in-plane axes and radius of the circle, and its span.
    Arc {
        center: Vec3,
        u: Vec3,
        v: Vec3,
        radius: f64,
        t0: f64,
        span: f64,
    },
    /// Ends, from the sphere's centre.
    Chord(Vec3, Vec3),
}

impl SphereRims {
    /// `None` when an edge is an ellipse or NURBS curve.
    fn of(topo: &Topology, face_id: FaceId) -> Result<Option<Self>, CheckError> {
        use brepkit_topology::edge::EdgeCurve;
        let face = topo.face(face_id)?;
        let FaceSurface::Sphere(sphere) = face.surface() else {
            return Ok(None);
        };
        let (center, radius) = (sphere.center(), sphere.radius());
        let on_sphere = |q: Point3| {
            let d = q - center;
            let len = d.length();
            (len > 0.0).then(|| center + d * (radius / len))
        };
        let mut rims = Vec::new();
        // (length, midpoint on the sphere, direction of travel) per outer edge.
        let mut outer_mids: Vec<(f64, Point3, Vec3)> = Vec::new();
        let wires = std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied());
        for (w, wid) in wires.enumerate() {
            for oe in topo.wire(wid)?.edges() {
                let edge = topo.edge(oe.edge())?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                let (mid, along, len) = match edge.curve() {
                    EdgeCurve::Line => {
                        rims.push(SphereRim::Chord(a - center, b - center));
                        (on_sphere(a + (b - a) * 0.5), b - a, (b - a).length())
                    }
                    EdgeCurve::Circle(c) => {
                        let (t0, t1) = edge.curve().domain_with_endpoints(a, b);
                        rims.push(SphereRim::Arc {
                            center: c.center() - center,
                            u: c.u_axis(),
                            v: c.v_axis(),
                            radius: c.radius(),
                            t0,
                            span: t1 - t0,
                        });
                        let tm = 0.5 * (t0 + t1);
                        (Some(c.evaluate(tm)), c.tangent(tm), c.radius() * (t1 - t0))
                    }
                    EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => return Ok(None),
                };
                if w == 0
                    && let Some(mid) = mid
                {
                    let along = if oe.is_forward() { along } else { -along };
                    outer_mids.push((len, mid, along));
                }
            }
        }
        let mut region = Self {
            center,
            radius,
            rims,
            inside: Vec::new(),
        };
        // The wire runs about the sphere's outward normal, so the face lies
        // to the left of each edge seen from outside. A point that far left
        // is inside only when the arc to it from as far right crosses the
        // edge alone: on a face narrower than the step it crosses the far
        // side too, and the step shrinks.
        outer_mids.sort_by(|x, y| y.0.total_cmp(&x.0));
        for (len, mid, along) in outer_mids.into_iter().take(4) {
            let n = mid - center;
            let Ok(left) = n.cross(along).normalize() else {
                continue;
            };
            let mut step = 1e-3 * len.min(radius);
            for _ in 0..3 {
                if let (Some(q), Some(out)) =
                    (on_sphere(mid + left * step), on_sphere(mid - left * step))
                    && region.crossings(out, q) == Some(1)
                {
                    region.inside.push(q);
                    break;
                }
                step /= 16.0;
            }
        }
        Ok(Some(region))
    }

    /// Whether `p`, on the sphere, lies on the face: the great-circle arc
    /// from `p` to a point inside crosses the edges an even number of
    /// times. The points inside vote, and at least two must carry it;
    /// `None` on a tie, or when fewer than two arcs clear the vertices.
    fn contains(&self, p: Point3) -> Option<bool> {
        let (mut on, mut off) = (0_u32, 0_u32);
        for &q in &self.inside {
            let Some(crossings) = self.crossings(p, q) else {
                continue;
            };
            if crossings % 2 == 0 {
                on += 1;
            } else {
                off += 1;
            }
            if on >= 2 && off == 0 {
                return Some(true);
            }
            if off >= 2 && on == 0 {
                return Some(false);
            }
        }
        match on.cmp(&off) {
            std::cmp::Ordering::Greater if on >= 2 => Some(true),
            std::cmp::Ordering::Less if off >= 2 => Some(false),
            _ => None,
        }
    }

    /// Crossings of the minor great-circle arc from `p` to `q` with the
    /// edges; `None` when it touches a vertex or an edge tangentially,
    /// or when `p` lies on an edge.
    fn crossings(&self, p: Point3, q: Point3) -> Option<u32> {
        let (pv, qv) = (p - self.center, q - self.center);
        let rr = self.radius * self.radius;
        let g = pv.cross(qv);
        if g.length() < SPHERE_GRAZE * rr {
            return None;
        }
        let g = g.normalize().ok()?;
        // Whether `x`, on the sphere and in the test arc's plane, lies on
        // the arc between `p` and `q`.
        let on_path = |x: Vec3| -> Option<bool> {
            let (s0, s1) = (pv.cross(x).dot(g) / rr, x.cross(qv).dot(g) / rr);
            if (s0.abs() < SPHERE_GRAZE && pv.dot(x) > 0.0)
                || (s1.abs() < SPHERE_GRAZE && qv.dot(x) > 0.0)
            {
                return None;
            }
            Some(s0 > 0.0 && s1 > 0.0)
        };
        let mut count = 0;
        for rim in &self.rims {
            match *rim {
                SphereRim::Arc {
                    center,
                    u,
                    v,
                    radius,
                    t0,
                    span,
                } => {
                    // g . x(t) = 0 along the circle x(t) = center + radius
                    // (cos t u + sin t v).
                    let (a, b, d) = (radius * g.dot(u), radius * g.dot(v), g.dot(center));
                    let m = a.hypot(b);
                    if m < SPHERE_GRAZE * self.radius {
                        if d.abs() < SPHERE_GRAZE * self.radius {
                            return None;
                        }
                        continue;
                    }
                    let c = -d / m;
                    if (c.abs() - 1.0).abs() <= SPHERE_GRAZE {
                        return None;
                    }
                    if c.abs() > 1.0 {
                        continue;
                    }
                    let (phi, w) = (b.atan2(a), c.acos());
                    let closed = span >= std::f64::consts::TAU - 1e-12;
                    for t in [phi + w, phi - w] {
                        let rel = (t - t0).rem_euclid(std::f64::consts::TAU);
                        if !closed
                            && (rel < SPHERE_GRAZE
                                || std::f64::consts::TAU - rel < SPHERE_GRAZE
                                || (rel - span).abs() < SPHERE_GRAZE)
                        {
                            return None;
                        }
                        if rel <= span && on_path(center + (u * t.cos() + v * t.sin()) * radius)? {
                            count += 1;
                        }
                    }
                }
                SphereRim::Chord(a, b) => {
                    let h = a.cross(b);
                    let Ok(h) = h.normalize() else {
                        continue;
                    };
                    let Ok(dir) = g.cross(h).normalize() else {
                        return None;
                    };
                    let x0 = dir * self.radius;
                    for x in [x0, -x0] {
                        let (s0, s1) = (
                            a.cross(x).dot(h) / (a.length() * self.radius),
                            x.cross(b).dot(h) / (b.length() * self.radius),
                        );
                        if (s0.abs() < SPHERE_GRAZE && a.dot(x) > 0.0)
                            || (s1.abs() < SPHERE_GRAZE && b.dot(x) > 0.0)
                        {
                            return None;
                        }
                        if s0 > 0.0 && s1 > 0.0 && on_path(x)? {
                            count += 1;
                        }
                    }
                }
            }
        }
        Some(count)
    }
}

/// Count a ray's crossings of a sphere face.
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
    let Some(region) = SphereRegion::of(topo, face_id)? else {
        return Ok(0);
    };
    let mut crossings = 0u32;
    for &t in roots {
        if t > RAY_T_MIN && region.contains(topo, face_id, origin + direction * t)? {
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
    if let Some(inside) = plane_hit_inside(topo, face_id, hit, normal)? {
        return Ok(u32::from(inside));
    }
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

/// Whether a plane hit lies inside its face, read on the face's own lines
/// and arcs rather than on chords of them. `None` when an edge is a NURBS
/// curve or the hit lies on the boundary.
///
/// # Errors
///
/// Returns an error if a topology lookup fails.
pub fn plane_hit_inside(
    topo: &Topology,
    face_id: FaceId,
    hit: Point3,
    normal: Vec3,
) -> Result<Option<bool>, CheckError> {
    let Ok(frame) = brepkit_math::frame::Frame3::from_normal(hit, normal) else {
        return Ok(None);
    };
    let Some(pieces) =
        brepkit_topology::planar::face_boundary_2d(topo, face_id, hit, frame.x, frame.y)?
    else {
        return Ok(None);
    };
    Ok(brepkit_math::region2d::point_in_region(
        &pieces,
        brepkit_math::vec::Point2::new(0.0, 0.0),
        brepkit_math::tolerance::Tolerance::new().linear,
    ))
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

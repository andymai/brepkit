//! Builder utilities for edges and wires.
//!
//! Provides ergonomic functions for creating topology from geometry.

use std::f64::consts::PI;

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::frame::Frame3;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};

use crate::Topology;
use crate::edge::{Edge, EdgeCurve, EdgeId};
use crate::face::{Face, FaceId, FaceSurface};
use crate::vertex::Vertex;
use crate::wire::{OrientedEdge, Wire, WireId};

/// Create a straight-line edge between two points.
///
/// Allocates vertices with the given `tolerance` and the connecting edge.
///
/// # Errors
///
/// Returns an error if the points are coincident.
pub fn make_line_edge(
    topo: &mut Topology,
    start: Point3,
    end: Point3,
    tolerance: f64,
) -> Result<EdgeId, crate::TopologyError> {
    let tol = Tolerance::new();
    if (end - start).length_squared() < tol.linear * tol.linear {
        return Err(crate::TopologyError::NonManifold {
            reason: "degenerate edge: start and end points coincide".into(),
        });
    }

    let v0 = topo.add_vertex(Vertex::new(start, tolerance));
    let v1 = topo.add_vertex(Vertex::new(end, tolerance));
    Ok(topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)))
}

/// Create a closed circular edge with a caller-supplied reference x-direction.
///
/// `ref_dir` is projected onto the plane perpendicular to `normal` to fix
/// the orientation of the circle's `u_axis` — and therefore the position
/// of the seam vertex at `circle.evaluate(0.0)`. The edge has parameter
/// domain `[0, 2π]`.
///
/// If `ref_dir` is zero or (nearly) parallel to `normal`, the underlying
/// [`Frame3::from_normal_and_ref`] falls back to an arbitrary perpendicular
/// axis — the seam orientation is *not* pinned in that case. The WASM
/// boundary rejects zero `ref_dir` outright; callers of the topology
/// layer should pass a non-zero, non-parallel direction to actually pin
/// the seam.
///
/// # Errors
///
/// Returns an error if `radius` is non-positive or `normal` is a zero vector.
pub fn make_circle_edge_with_ref(
    topo: &mut Topology,
    center: Point3,
    normal: Vec3,
    radius: f64,
    ref_dir: Vec3,
    tolerance: f64,
) -> Result<EdgeId, crate::TopologyError> {
    let circle = Circle3D::new_with_ref(center, normal, radius, ref_dir).map_err(|e| {
        crate::TopologyError::NonManifold {
            reason: format!("invalid circle: {e}"),
        }
    })?;
    let seam = circle.evaluate(0.0);
    let v = topo.add_vertex(Vertex::new(seam, tolerance));
    Ok(topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle))))
}

/// Create a closed circular edge.
///
/// Constructs a [`Circle3D`] from `center`, `normal`, and `radius`, then
/// creates a single closed edge whose start and end share a seam vertex
/// at `circle.evaluate(0.0)`. The edge has parameter domain `[0, 2π]`.
///
/// Delegates to [`make_circle_edge_with_ref`] using the default frame's
/// x-axis ([`Frame3::from_normal`]) as the reference direction.
///
/// # Errors
///
/// Returns an error if `radius` is non-positive or `normal` is a zero vector.
pub fn make_circle_edge(
    topo: &mut Topology,
    center: Point3,
    normal: Vec3,
    radius: f64,
    tolerance: f64,
) -> Result<EdgeId, crate::TopologyError> {
    let ref_dir = Frame3::from_normal(center, normal)
        .map_err(|e| crate::TopologyError::NonManifold {
            reason: format!("invalid circle frame: {e}"),
        })?
        .x;
    make_circle_edge_with_ref(topo, center, normal, radius, ref_dir, tolerance)
}

/// Create a closed elliptical edge with a caller-supplied reference major-axis.
///
/// `ref_dir` is projected onto the plane perpendicular to `normal` to fix
/// the orientation of the ellipse's major axis (`u_axis`, carrying the
/// `semi_major` extent). The edge has parameter domain `[0, 2π]`.
///
/// If `ref_dir` is zero or (nearly) parallel to `normal`, the underlying
/// [`Frame3::from_normal_and_ref`] falls back to an arbitrary perpendicular
/// axis — the major-axis orientation is *not* pinned in that case. The
/// WASM boundary rejects zero `ref_dir` outright; callers of the topology
/// layer should pass a non-zero, non-parallel direction to actually pin
/// the major axis.
///
/// # Errors
///
/// Returns an error if either semi-axis is non-positive, `semi_minor`
/// exceeds `semi_major`, or `normal` is a zero vector.
pub fn make_ellipse_edge_with_ref(
    topo: &mut Topology,
    center: Point3,
    normal: Vec3,
    semi_major: f64,
    semi_minor: f64,
    ref_dir: Vec3,
    tolerance: f64,
) -> Result<EdgeId, crate::TopologyError> {
    let ellipse = Ellipse3D::new_with_ref(center, normal, semi_major, semi_minor, ref_dir)
        .map_err(|e| crate::TopologyError::NonManifold {
            reason: format!("invalid ellipse: {e}"),
        })?;
    let seam = ellipse.evaluate(0.0);
    let v = topo.add_vertex(Vertex::new(seam, tolerance));
    Ok(topo.add_edge(Edge::new(v, v, EdgeCurve::Ellipse(ellipse))))
}

/// Create a closed elliptical edge.
///
/// Constructs an [`Ellipse3D`] from `center`, `normal`, and the two
/// semi-axis lengths, then creates a single closed edge whose start and
/// end share a seam vertex at `ellipse.evaluate(0.0)`. The edge has
/// parameter domain `[0, 2π]`.
///
/// Delegates to [`make_ellipse_edge_with_ref`] using the default frame's
/// x-axis ([`Frame3::from_normal`]) as the reference direction.
///
/// # Errors
///
/// Returns an error if either semi-axis is non-positive, `semi_minor`
/// exceeds `semi_major`, or `normal` is a zero vector.
pub fn make_ellipse_edge(
    topo: &mut Topology,
    center: Point3,
    normal: Vec3,
    semi_major: f64,
    semi_minor: f64,
    tolerance: f64,
) -> Result<EdgeId, crate::TopologyError> {
    let ref_dir = Frame3::from_normal(center, normal)
        .map_err(|e| crate::TopologyError::NonManifold {
            reason: format!("invalid ellipse frame: {e}"),
        })?
        .x;
    make_ellipse_edge_with_ref(
        topo, center, normal, semi_major, semi_minor, ref_dir, tolerance,
    )
}

/// Create a closed wire from an ordered list of points.
///
/// Each consecutive pair of points becomes a line edge, and the last
/// point connects back to the first. Vertices are created with the
/// given `tolerance`.
///
/// # Errors
///
/// Returns an error if fewer than 3 points are provided.
pub fn make_polygon_wire(
    topo: &mut Topology,
    points: &[Point3],
    tolerance: f64,
) -> Result<WireId, crate::TopologyError> {
    let n = points.len();
    if n < 3 {
        return Err(crate::TopologyError::Empty {
            entity: "polygon (need at least 3 points)",
        });
    }

    let verts: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tolerance)))
        .collect();

    let edges: Vec<_> = (0..n)
        .map(|i| {
            let next = (i + 1) % n;
            topo.add_edge(Edge::new(verts[i], verts[next], EdgeCurve::Line))
        })
        .collect();

    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();

    let wire = Wire::new(oriented, true)?;
    Ok(topo.add_wire(wire))
}

/// Create a regular polygon wire on the XY plane centered at the origin.
///
/// Returns the wire ID of a closed polygon with `n_sides` edges.
/// Vertices are created with the given `tolerance`.
///
/// # Errors
///
/// Returns an error if `n_sides < 3` or `radius` is non-positive.
pub fn make_regular_polygon_wire(
    topo: &mut Topology,
    radius: f64,
    n_sides: usize,
    tolerance: f64,
) -> Result<WireId, crate::TopologyError> {
    if n_sides < 3 {
        return Err(crate::TopologyError::Empty {
            entity: "polygon (need at least 3 sides)",
        });
    }
    if radius <= 0.0 {
        return Err(crate::TopologyError::NonManifold {
            reason: "polygon radius must be positive".into(),
        });
    }

    #[allow(clippy::cast_precision_loss)]
    let points: Vec<Point3> = (0..n_sides)
        .map(|i| {
            let angle = 2.0 * PI * (i as f64) / (n_sides as f64);
            Point3::new(radius * angle.cos(), radius * angle.sin(), 0.0)
        })
        .collect();

    make_polygon_wire(topo, &points, tolerance)
}

/// Create a face from a closed wire (general construction mode).
///
/// Samples points along the wire's edges, fits a candidate plane, and
/// verifies that every sample lies within tolerance of it. If verification
/// succeeds the face carries a planar surface; otherwise a non-planar
/// (bilinear NURBS) surface is attached so that a surface-type query never
/// reports `plane` for a wire whose geometry is not coplanar.
///
/// # Errors
///
/// Returns an error if the wire has no edges or fewer than 3 usable sample
/// points (the wire is degenerate or collinear).
pub fn make_face_from_wire(
    topo: &mut Topology,
    wire_id: WireId,
) -> Result<FaceId, crate::TopologyError> {
    let wire = topo.wire(wire_id)?;
    let edges = wire.edges();
    if edges.is_empty() {
        return Err(crate::TopologyError::Empty {
            entity: "wire (no edges)",
        });
    }

    let sampled = sample_wire_for_planarity(topo, edges)?;
    let surface = match verified_plane(&sampled) {
        Some((normal, d)) => FaceSurface::Plane { normal, d },
        None => FaceSurface::Nurbs(bilinear_surface(&sampled.points)?),
    };

    Ok(topo.add_face(Face::new(wire_id, vec![], surface)))
}

/// Create a strictly planar face from a closed wire.
///
/// Like [`make_face_from_wire`] but in planar-only mode: if the wire's
/// geometry does not lie within tolerance of a single plane, construction
/// fails with [`crate::TopologyError::NotPlanar`] instead of falling back to
/// a non-planar surface.
///
/// # Errors
///
/// Returns an error if the wire has no edges, has fewer than 3 usable sample
/// points, or its samples do not lie within tolerance of any single plane.
pub fn make_planar_face_from_wire(
    topo: &mut Topology,
    wire_id: WireId,
) -> Result<FaceId, crate::TopologyError> {
    let wire = topo.wire(wire_id)?;
    let edges = wire.edges();
    if edges.is_empty() {
        return Err(crate::TopologyError::Empty {
            entity: "wire (no edges)",
        });
    }

    let sampled = sample_wire_for_planarity(topo, edges)?;
    let (normal, d) = verified_plane(&sampled).ok_or(crate::TopologyError::NotPlanar)?;

    Ok(topo.add_face(Face::new(wire_id, vec![], FaceSurface::Plane { normal, d })))
}

/// Sample points and tolerance gathered from a wire for planarity testing.
struct WireSamples {
    points: Vec<Point3>,
    /// Boundary points in wire-traversal order, used to orient the plane
    /// normal consistently with the wire winding (right-hand rule). Unlike
    /// `points`, these are strictly ordered along the loop so Newell's method
    /// yields the correct winding sign.
    ordered: Vec<Point3>,
    /// Squared effective tolerance: `max(base linear, max edge tolerance)²`.
    tol_sq: f64,
    /// Plane derived directly from the first conic edge, if any. A conic is
    /// intrinsically planar, so its own axis-system plane is exact and is
    /// preferred over a scatter fit.
    conic_plane: Option<(Vec3, f64)>,
}

/// A spline lies in a plane iff all its control points do, so the poles are
/// the exact planarity witnesses; a conic is intrinsically planar and four
/// well-spaced points pin its inclusion; a line needs only its endpoints.
fn sample_wire_for_planarity(
    topo: &Topology,
    edges: &[OrientedEdge],
) -> Result<WireSamples, crate::TopologyError> {
    let mut points = Vec::with_capacity(edges.len() * 4);
    let mut ordered = Vec::with_capacity(edges.len() * 4);
    let mut tol = Tolerance::new().linear;
    let mut conic_plane = None;

    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let edge_tol = edge.tolerance().unwrap_or_else(|| {
            let a = topo.vertex(edge.start()).map_or(0.0, Vertex::tolerance);
            let b = topo.vertex(edge.end()).map_or(0.0, Vertex::tolerance);
            a.max(b)
        });
        tol = tol.max(edge_tol);

        let start_pt = topo.vertex(oe.oriented_start(edge))?.point();
        let end_pt = topo.vertex(oe.oriented_end(edge))?.point();

        match edge.curve() {
            EdgeCurve::Line => {
                points.push(start_pt);
                points.push(end_pt);
                points.push(start_pt + (end_pt - start_pt) * 0.5);
                ordered.push(start_pt);
            }
            EdgeCurve::Circle(c) => {
                if conic_plane.is_none() {
                    conic_plane = Some(plane_from(c.center(), c.normal()));
                }
                let before = points.len();
                sample_conic(
                    start_pt,
                    end_pt,
                    edge.is_closed(),
                    &mut points,
                    |t| c.evaluate(t),
                    |p| c.project(p),
                );
                ordered.extend_from_slice(&points[before..]);
            }
            EdgeCurve::Ellipse(e) => {
                if conic_plane.is_none() {
                    conic_plane = Some(plane_from(e.center(), e.normal()));
                }
                let before = points.len();
                sample_conic(
                    start_pt,
                    end_pt,
                    edge.is_closed(),
                    &mut points,
                    |t| e.evaluate(t),
                    |p| e.project(p),
                );
                ordered.extend_from_slice(&points[before..]);
            }
            EdgeCurve::NurbsCurve(nc) => {
                points.extend_from_slice(nc.control_points());
                ordered.push(start_pt);
                ordered.push(start_pt + (end_pt - start_pt) * 0.5);
            }
        }
    }

    Ok(WireSamples {
        points,
        ordered,
        tol_sq: tol * tol,
        conic_plane,
    })
}

/// Sample an oriented conic sub-arc at four points along the short way
/// between its endpoints, so samples lie on the real curve segment rather
/// than the antipodal arc.
fn sample_conic(
    start_pt: Point3,
    end_pt: Point3,
    closed: bool,
    out: &mut Vec<Point3>,
    evaluate: impl Fn(f64) -> Point3,
    project: impl Fn(Point3) -> f64,
) {
    let tau = std::f64::consts::TAU;
    if closed {
        for i in 0..4 {
            out.push(evaluate(tau * f64::from(i) / 4.0));
        }
        return;
    }
    let t_a = project(start_pt);
    let t_b = project(end_pt);
    let mut delta = t_b - t_a;
    if delta > PI {
        delta -= tau;
    } else if delta < -PI {
        delta += tau;
    }
    for i in 0..=3 {
        out.push(evaluate(t_a + delta * f64::from(i) / 3.0));
    }
}

fn plane_from(point: Point3, normal: Vec3) -> (Vec3, f64) {
    let n = normal.normalize().unwrap_or(normal);
    (n, n.dot(point - Point3::new(0.0, 0.0, 0.0)))
}

/// Obtain a candidate plane (conic-derived or scatter-fit) and verify that
/// every sample lies within tolerance of it. Returns `None` when no plane
/// can be formed or verification fails.
fn verified_plane(samples: &WireSamples) -> Option<(Vec3, f64)> {
    let (mut normal, mut d) = samples
        .conic_plane
        .or_else(|| fit_plane(&samples.points, samples.tol_sq))?;

    let origin = Point3::new(0.0, 0.0, 0.0);
    for p in &samples.points {
        let dist = normal.dot(*p - origin) - d;
        if dist * dist > samples.tol_sq {
            return None;
        }
    }

    // Orient the normal consistently with the wire winding (right-hand rule).
    // `fit_plane` and a conic's intrinsic plane both yield a normal whose sign
    // is independent of traversal order; downstream consumers (e.g. extrude)
    // require the normal to follow the boundary's CCW winding so cap and wall
    // orientations come out correct.
    if let Some(winding_normal) = newell_normal(&samples.ordered)
        && normal.dot(winding_normal) < 0.0
    {
        normal = -normal;
        d = -d;
    }
    Some((normal, d))
}

/// Newell's-method normal for an ordered loop of boundary points. The result
/// follows the winding (CCW gives the right-hand-rule normal). Returns `None`
/// for fewer than 3 points or a degenerate (collinear) loop.
fn newell_normal(ordered: &[Point3]) -> Option<Vec3> {
    if ordered.len() < 3 {
        return None;
    }
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    let n = ordered.len();
    for i in 0..n {
        let curr = ordered[i];
        let next = ordered[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    Vec3::new(nx, ny, nz).normalize().ok()
}

/// Fit a candidate plane from scattered sample points using the most
/// extreme, least-collinear triple. Returns `None` for degenerate
/// (coincident) or collinear point sets.
fn fit_plane(points: &[Point3], tol_sq: f64) -> Option<(Vec3, f64)> {
    if points.len() < 3 {
        return None;
    }
    let origin = Point3::new(0.0, 0.0, 0.0);
    let p0 = points[0];

    let (p1, far_sq) = points
        .iter()
        .map(|&p| (p, (p - p0).length_squared()))
        .fold((p0, 0.0), |acc, c| if c.1 > acc.1 { c } else { acc });
    if far_sq <= tol_sq {
        return None;
    }
    let v1 = p1 - p0;

    let (p2, off_sq) = points
        .iter()
        .map(|&p| (p, v1.cross(p - p0).length_squared()))
        .fold((p0, 0.0), |acc, c| if c.1 > acc.1 { c } else { acc });
    if (p2 - p0).length_squared() <= tol_sq {
        return None;
    }

    let v2 = p2 - p0;
    let angular = Tolerance::new().angular;
    if off_sq <= v1.length_squared() * v2.length_squared() * angular * angular {
        return None;
    }

    let normal = v1.cross(v2).normalize().ok()?;
    let d = normal.dot(p0 - origin);
    Some((normal, d))
}

/// Build a degree-1 bilinear NURBS patch spanning the extreme corners of the
/// sample set, used as a non-planar surface for wires that fail planarity
/// verification in general construction mode.
fn bilinear_surface(points: &[Point3]) -> Result<NurbsSurface, crate::TopologyError> {
    if points.len() < 3 {
        return Err(crate::TopologyError::NonManifold {
            reason: "need at least 3 sample points for a non-planar face".into(),
        });
    }
    let p0 = points[0];
    let p1 = *points
        .iter()
        .max_by(|a, b| {
            (**a - p0)
                .length_squared()
                .total_cmp(&(**b - p0).length_squared())
        })
        .unwrap_or(&p0);
    let v1 = p1 - p0;
    let p2 = *points
        .iter()
        .max_by(|a, b| {
            v1.cross(**a - p0)
                .length_squared()
                .total_cmp(&v1.cross(**b - p0).length_squared())
        })
        .unwrap_or(&p0);

    // p3 must be the actual sample farthest off the (p0, p1, p2) plane so the
    // resulting bilinear patch is genuinely twisted; a parallelogram corner
    // would be coplanar with p0/p1/p2 and recognized as a plane downstream.
    let plane_n = v1.cross(p2 - p0).normalize().unwrap_or(v1);
    let p3 = *points
        .iter()
        .max_by(|a, b| {
            plane_n
                .dot(**a - p0)
                .abs()
                .total_cmp(&plane_n.dot(**b - p0).abs())
        })
        .unwrap_or(&p2);

    let grid = vec![vec![p0, p2], vec![p1, p3]];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots = vec![0.0, 0.0, 1.0, 1.0];

    NurbsSurface::new(1, 1, knots.clone(), knots, grid, weights).map_err(|e| {
        crate::TopologyError::NonManifold {
            reason: format!("non-planar surface construction failed: {e}"),
        }
    })
}

/// Create a rectangular face on the XY plane centered at the origin.
///
/// Vertices are created with the given `tolerance`.
///
/// # Errors
///
/// Returns an error if `width` or `height` is non-positive.
pub fn make_rectangle_face(
    topo: &mut Topology,
    width: f64,
    height: f64,
    tolerance: f64,
) -> Result<FaceId, crate::TopologyError> {
    if width <= 0.0 || height <= 0.0 {
        return Err(crate::TopologyError::NonManifold {
            reason: "rectangle dimensions must be positive".into(),
        });
    }

    let hw = width / 2.0;
    let hh = height / 2.0;
    let points = [
        Point3::new(-hw, -hh, 0.0),
        Point3::new(hw, -hh, 0.0),
        Point3::new(hw, hh, 0.0),
        Point3::new(-hw, hh, 0.0),
    ];

    let wid = make_polygon_wire(topo, &points, tolerance)?;
    make_face_from_wire(topo, wid)
}

/// Create a circular polygon face on the XY plane centered at the origin.
///
/// The circle is approximated with `segments` straight edges. Vertices
/// are created with the given `tolerance`.
///
/// # Errors
///
/// Returns an error if `radius` is non-positive or `segments < 3`.
pub fn make_circle_face(
    topo: &mut Topology,
    radius: f64,
    segments: usize,
    tolerance: f64,
) -> Result<FaceId, crate::TopologyError> {
    let wid = make_regular_polygon_wire(topo, radius, segments, tolerance)?;
    make_face_from_wire(topo, wid)
}

/// Create a closed planar face from an ordered sequence of 3D points.
///
/// Each consecutive pair of points becomes a line edge, and the last point
/// connects back to the first. The face normal is computed via Newell's
/// method from the point polygon.
///
/// # Errors
///
/// Returns an error if fewer than 3 points are provided or the computed
/// normal is degenerate.
pub fn make_planar_face(
    topo: &mut Topology,
    points: &[Point3],
    tolerance: f64,
) -> Result<FaceId, crate::TopologyError> {
    let wid = make_polygon_wire(topo, points, tolerance)?;
    make_face_from_wire(topo, wid)
}

/// Create an edge from explicit start/end points and a NURBS curve.
///
/// Allocates vertices at `start` and `end` with the given `tolerance`,
/// then creates an edge with the provided `NurbsCurve` geometry.
pub fn make_nurbs_edge(
    topo: &mut Topology,
    start: Point3,
    end: Point3,
    curve: NurbsCurve,
    tolerance: f64,
) -> EdgeId {
    let v_start = topo.add_vertex(Vertex::new(start, tolerance));
    let v_end = topo.add_vertex(Vertex::new(end, tolerance));
    topo.add_edge(Edge::new(v_start, v_end, EdgeCurve::NurbsCurve(curve)))
}

/// Create an edge from a NURBS curve, evaluating its endpoints.
///
/// The start and end points are obtained by evaluating the curve at its
/// first and last knot values.
pub fn make_nurbs_edge_from_curve(
    topo: &mut Topology,
    curve: &NurbsCurve,
    tolerance: f64,
) -> EdgeId {
    let knots = curve.knots();
    let start = curve.evaluate(knots[0]);
    let end = curve.evaluate(knots[knots.len() - 1]);
    let v_start = topo.add_vertex(Vertex::new(start, tolerance));
    let v_end = topo.add_vertex(Vertex::new(end, tolerance));
    topo.add_edge(Edge::new(
        v_start,
        v_end,
        EdgeCurve::NurbsCurve(curve.clone()),
    ))
}

/// Create a face from a NURBS surface with a rectangular domain wire.
///
/// Evaluates the four corner points of the surface domain, creates line
/// edges between them forming a closed rectangular wire, and attaches
/// the `NurbsSurface` as the face geometry.
///
/// # Errors
///
/// Returns an error if the wire construction fails.
pub fn make_nurbs_face(
    topo: &mut Topology,
    surface: NurbsSurface,
    tolerance: f64,
) -> Result<FaceId, crate::TopologyError> {
    let (u_min, u_max) = surface.domain_u();
    let (v_min, v_max) = surface.domain_v();
    let corners = [
        surface.evaluate(u_min, v_min),
        surface.evaluate(u_max, v_min),
        surface.evaluate(u_max, v_max),
        surface.evaluate(u_min, v_max),
    ];
    let verts: Vec<_> = corners
        .iter()
        .map(|p| topo.add_vertex(Vertex::new(*p, tolerance)))
        .collect();
    let n = verts.len();
    let edges: Vec<_> = (0..n)
        .map(|i| topo.add_edge(Edge::new(verts[i], verts[(i + 1) % n], EdgeCurve::Line)))
        .collect();
    let oriented: Vec<_> = edges
        .iter()
        .map(|&eid| OrientedEdge::new(eid, true))
        .collect();
    let wire = Wire::new(oriented, true)?;
    let wid = topo.add_wire(wire);
    let face_id = topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));
    Ok(face_id)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::panic)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Point3;

    use super::*;
    use crate::Topology;

    const TOL: f64 = 1e-7;

    #[test]
    fn make_line_edge_basic() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
            TOL,
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        assert_ne!(edge.start(), edge.end());
    }

    #[test]
    fn make_line_edge_coincident_error() {
        let mut topo = Topology::new();
        let p = Point3::new(1.0, 2.0, 3.0);
        assert!(make_line_edge(&mut topo, p, p, TOL).is_err());
    }

    #[test]
    fn make_circle_edge_is_closed_with_circle_curve() {
        let mut topo = Topology::new();
        let eid = make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.5,
            TOL,
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        assert!(edge.is_closed(), "circle edge must be closed");
        assert_eq!(edge.curve().type_tag(), "circle");

        let seam = topo.vertex(edge.start()).unwrap().point();
        let (t_min, t_max) = edge.curve().domain_with_endpoints(seam, seam);
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn make_circle_edge_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(
            make_circle_edge(
                &mut topo,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                0.0,
                TOL,
            )
            .is_err()
        );
    }

    #[test]
    fn make_circle_edge_zero_normal_error() {
        let mut topo = Topology::new();
        assert!(
            make_circle_edge(
                &mut topo,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 0.0),
                1.0,
                TOL,
            )
            .is_err()
        );
    }

    #[test]
    fn make_ellipse_edge_is_closed_with_ellipse_curve() {
        let mut topo = Topology::new();
        let eid = make_ellipse_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            2.0,
            TOL,
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        assert!(edge.is_closed(), "ellipse edge must be closed");
        assert_eq!(edge.curve().type_tag(), "ellipse");

        let seam = topo.vertex(edge.start()).unwrap().point();
        let (t_min, t_max) = edge.curve().domain_with_endpoints(seam, seam);
        assert!((t_min - 0.0).abs() < 1e-12);
        assert!((t_max - std::f64::consts::TAU).abs() < 1e-12);
    }

    #[test]
    fn make_ellipse_edge_minor_exceeds_major_error() {
        let mut topo = Topology::new();
        assert!(
            make_ellipse_edge(
                &mut topo,
                Point3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                1.0,
                2.0,
                TOL,
            )
            .is_err()
        );
    }

    #[test]
    fn make_circle_edge_with_ref_seam_at_supplied_x() {
        // ref_dir=[1,0,0] with normal=[0,0,1]: u_axis should align with +x,
        // putting the seam vertex (circle.evaluate(0.0) = center + r·u_axis)
        // at (r, 0, 0). The default frame produces u_axis=[0,1,0] so the
        // seam lands at (0, r, 0) — verifies the ref_dir takes effect.
        let mut topo = Topology::new();
        let eid = make_circle_edge_with_ref(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            Vec3::new(1.0, 0.0, 0.0),
            TOL,
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        let seam = topo.vertex(edge.start()).unwrap().point();
        assert!(
            (seam.x() - 3.0).abs() < 1e-12,
            "seam.x should be 3.0, got {}",
            seam.x()
        );
        assert!(
            seam.y().abs() < 1e-12,
            "seam.y should be 0, got {}",
            seam.y()
        );
        assert!(
            seam.z().abs() < 1e-12,
            "seam.z should be 0, got {}",
            seam.z()
        );

        // Sanity: default-frame variant puts the seam at +y instead.
        let mut topo2 = Topology::new();
        let eid2 = make_circle_edge(
            &mut topo2,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            3.0,
            TOL,
        )
        .unwrap();
        let seam2 = topo2
            .vertex(topo2.edge(eid2).unwrap().start())
            .unwrap()
            .point();
        assert!(
            (seam2.y() - 3.0).abs() < 1e-12,
            "default-frame seam.y should be 3.0, got {}",
            seam2.y()
        );
    }

    #[test]
    fn make_ellipse_edge_with_ref_major_axis_along_supplied_dir() {
        // ref_dir=[1,0,0] with normal=[0,0,1]: u_axis aligns with +x, so
        // the seam vertex (ellipse.evaluate(0.0) = center + semi_major·u_axis)
        // lands at (semi_major, 0, 0). This is what brepjs's `sketchEllipse`
        // assumes; the default frame puts the major axis along +y, which
        // breaks the volume of `sketchEllipse(5,2).extrude(10)` because
        // the adapter falls back to a NURBS approximation.
        let mut topo = Topology::new();
        let eid = make_ellipse_edge_with_ref(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            2.0,
            Vec3::new(1.0, 0.0, 0.0),
            TOL,
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        let seam = topo.vertex(edge.start()).unwrap().point();
        assert!(
            (seam.x() - 5.0).abs() < 1e-12,
            "seam.x should be semi_major (5.0), got {}",
            seam.x()
        );
        assert!(
            seam.y().abs() < 1e-12,
            "seam.y should be 0, got {}",
            seam.y()
        );

        // Sanity: default-frame variant puts the major-axis seam along +y.
        let mut topo2 = Topology::new();
        let eid2 = make_ellipse_edge(
            &mut topo2,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            5.0,
            2.0,
            TOL,
        )
        .unwrap();
        let seam2 = topo2
            .vertex(topo2.edge(eid2).unwrap().start())
            .unwrap()
            .point();
        assert!(
            (seam2.y() - 5.0).abs() < 1e-12,
            "default-frame seam.y should be 5.0, got {}",
            seam2.y()
        );
    }

    #[test]
    fn make_polygon_wire_square() {
        let mut topo = Topology::new();
        let wid = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            TOL,
        )
        .unwrap();

        let wire = topo.wire(wid).unwrap();
        assert_eq!(wire.edges().len(), 4);
    }

    #[test]
    fn make_polygon_wire_too_few_points() {
        let mut topo = Topology::new();
        assert!(
            make_polygon_wire(
                &mut topo,
                &[Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                TOL,
            )
            .is_err()
        );
    }

    #[test]
    fn make_regular_polygon_wire_hexagon() {
        let mut topo = Topology::new();
        let wid = make_regular_polygon_wire(&mut topo, 1.0, 6, TOL).unwrap();
        let wire = topo.wire(wid).unwrap();
        assert_eq!(wire.edges().len(), 6);
    }

    #[test]
    fn make_face_from_wire_square() {
        let mut topo = Topology::new();
        let wid = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            TOL,
        )
        .unwrap();

        let fid = make_face_from_wire(&mut topo, wid).unwrap();
        let face = topo.face(fid).unwrap();
        let FaceSurface::Plane { normal, .. } = face.surface() else {
            panic!("square wire must produce a planar face");
        };
        let tol = Tolerance::new();
        // CCW square in the XY plane: right-hand rule normal must point +Z.
        // A sign-agnostic plane fit would (incorrectly) allow -Z here, which
        // flips extruded caps and collapses solid volume.
        assert!(
            tol.approx_eq(normal.z(), 1.0),
            "CCW square normal must be +Z"
        );
    }

    #[test]
    fn make_face_from_wire_cw_square_normal_is_minus_z() {
        let mut topo = Topology::new();
        // CW winding (reverse of the CCW square): normal must point -Z.
        let wid = make_polygon_wire(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
            ],
            TOL,
        )
        .unwrap();
        let fid = make_face_from_wire(&mut topo, wid).unwrap();
        let FaceSurface::Plane { normal, .. } = topo.face(fid).unwrap().surface() else {
            panic!("square wire must produce a planar face");
        };
        let tol = Tolerance::new();
        assert!(
            tol.approx_eq(normal.z(), -1.0),
            "CW square normal must be -Z"
        );
    }

    #[test]
    fn make_rectangle_face_basic() {
        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 2.0, 3.0, TOL).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(matches!(face.surface(), FaceSurface::Plane { .. }));
    }

    #[test]
    fn make_rectangle_face_zero_error() {
        let mut topo = Topology::new();
        assert!(make_rectangle_face(&mut topo, 0.0, 1.0, TOL).is_err());
    }

    #[test]
    fn make_circle_face_basic() {
        let mut topo = Topology::new();
        let fid = make_circle_face(&mut topo, 1.0, 16, TOL).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(matches!(face.surface(), FaceSurface::Plane { .. }));
    }

    #[test]
    fn make_circle_face_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(make_circle_face(&mut topo, 0.0, 16, TOL).is_err());
    }

    fn polygon_wire_3d(topo: &mut Topology, pts: &[Point3]) -> WireId {
        make_polygon_wire(topo, pts, TOL).unwrap()
    }

    #[test]
    fn make_planar_face_from_wire_rejects_noncoplanar_wire() {
        let mut topo = Topology::new();
        let wid = polygon_wire_3d(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(5.0, 5.0, 5.0),
            ],
        );

        let res = make_planar_face_from_wire(&mut topo, wid);
        assert!(matches!(res, Err(crate::TopologyError::NotPlanar)));
    }

    #[test]
    fn make_face_from_wire_noncoplanar_is_not_plane_surface() {
        let mut topo = Topology::new();
        let wid = polygon_wire_3d(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(5.0, 5.0, 5.0),
            ],
        );

        let fid = make_face_from_wire(&mut topo, wid).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(
            !matches!(face.surface(), FaceSurface::Plane { .. }),
            "non-coplanar wire must not produce a planar surface"
        );
    }

    #[test]
    fn make_planar_face_from_wire_accepts_square() {
        let mut topo = Topology::new();
        let wid = polygon_wire_3d(
            &mut topo,
            &[
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(10.0, 0.0, 0.0),
                Point3::new(10.0, 10.0, 0.0),
                Point3::new(0.0, 10.0, 0.0),
            ],
        );

        let fid = make_planar_face_from_wire(&mut topo, wid).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(
            matches!(face.surface(), FaceSurface::Plane { .. }),
            "square wire must classify as plane"
        );
        if let FaceSurface::Plane { normal, .. } = face.surface() {
            let tol = Tolerance::new();
            assert!(tol.approx_eq(normal.z().abs(), 1.0), "normal should be ±Z");
        }
    }

    #[test]
    fn make_planar_face_from_wire_accepts_single_circle() {
        let mut topo = Topology::new();
        let eid = make_circle_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            2.0,
            TOL,
        )
        .unwrap();
        let wire = Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap();
        let wid = topo.add_wire(wire);

        let fid = make_planar_face_from_wire(&mut topo, wid).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(
            matches!(face.surface(), FaceSurface::Plane { .. }),
            "single circle loop must classify as plane"
        );
    }

    #[test]
    fn make_planar_face_from_wire_near_miss_boundary() {
        let eps = 1e-6;
        let within = eps * 0.4;
        let beyond = eps * 4.0;

        let build = |z: f64| {
            let mut topo = Topology::new();
            let wid = make_polygon_wire(
                &mut topo,
                &[
                    Point3::new(0.0, 0.0, 0.0),
                    Point3::new(10.0, 0.0, 0.0),
                    Point3::new(10.0, 10.0, 0.0),
                    Point3::new(0.0, 10.0, z),
                ],
                eps,
            )
            .unwrap();
            make_planar_face_from_wire(&mut topo, wid)
        };

        assert!(
            build(within).is_ok(),
            "within-tolerance wire must be accepted"
        );
        assert!(
            matches!(build(beyond), Err(crate::TopologyError::NotPlanar)),
            "beyond-tolerance wire must be rejected"
        );
    }
}

//! Shared utility functions for the check crate.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::CheckError;

/// Compute the normal of a polygon via Newell's method.
///
/// Returns a unit-length normal, or `(0,0,1)` for degenerate polygons.
pub fn polygon_normal(verts: &[Point3]) -> Vec3 {
    let mut nx = 0.0;
    let mut ny = 0.0;
    let mut nz = 0.0;
    let n = verts.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let vi = verts[i];
        let vj = verts[j];
        nx += (vi.y() - vj.y()) * (vi.z() + vj.z());
        ny += (vi.z() - vj.z()) * (vi.x() + vj.x());
        nz += (vi.x() - vj.x()) * (vi.y() + vj.y());
    }
    let len = (nx.mul_add(nx, ny.mul_add(ny, nz * nz))).sqrt();
    if len < 1e-30 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(nx / len, ny / len, nz / len)
    }
}

/// Number of sample points for closed-curve edges.
pub const CLOSED_CURVE_SAMPLES: usize = 32;

/// Sample a closed-edge curve at `n` evenly spaced parameter values.
///
/// Returns an empty vector for `Line` edges (geometry determined by vertices).
pub fn sample_edge_curve(curve: &EdgeCurve, n: usize) -> Vec<Point3> {
    match curve {
        EdgeCurve::Circle(c) => (0..n)
            .map(|i| {
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                c.evaluate(t)
            })
            .collect(),
        EdgeCurve::Ellipse(e) => (0..n)
            .map(|i| {
                let t = std::f64::consts::TAU * (i as f64) / (n as f64);
                e.evaluate(t)
            })
            .collect(),
        EdgeCurve::NurbsCurve(nc) => {
            let (u0, u1) = nc.domain();
            let start_pt = nc.evaluate(u0);
            let end_pt = nc.evaluate(u1);
            let is_closed = (start_pt - end_pt).length() < 1e-6;
            let divisor = if is_closed { n } else { n - 1 };
            (0..n)
                .map(|i| {
                    let t = u0 + (u1 - u0) * (i as f64) / (divisor as f64);
                    nc.evaluate(t)
                })
                .collect()
        }
        EdgeCurve::Line => vec![],
    }
}

/// Build a polygon from the outer wire of a face by sampling vertex positions
/// and closed-edge curves.
///
/// # Errors
///
/// Returns an error if any topology entity referenced by the face is missing.
pub fn face_polygon(topo: &Topology, face_id: FaceId) -> Result<Vec<Point3>, CheckError> {
    let face = topo.face(face_id)?;
    wire_polygon(topo, face.outer_wire())
}

/// Build a polygon from a wire by sampling vertex positions and closed-edge
/// curves.
///
/// Wires store edges in loop order, but the per-edge orientation flags are
/// not guaranteed to chain head-to-tail; each edge's traversal direction is
/// re-derived from vertex connectivity with the previous edge so the polygon
/// follows the actual loop.
///
/// # Errors
///
/// Returns an error if any topology entity referenced by the wire is missing.
pub fn wire_polygon(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<Vec<Point3>, CheckError> {
    let wire = topo.wire(wire_id)?;
    let mut pts = Vec::new();
    let mut prev_end: Option<brepkit_topology::vertex::VertexId> = None;

    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let curve = edge.curve();
        let start_vid = edge.start();
        let end_vid = edge.end();
        let forward = match prev_end {
            Some(pe) if start_vid == pe && end_vid != pe => true,
            Some(pe) if end_vid == pe && start_vid != pe => false,
            _ => oe.is_forward(),
        };
        let is_closed_edge = start_vid == end_vid
            && matches!(
                curve,
                EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_)
            );
        if is_closed_edge {
            // Start sampling at the edge's seam vertex so the polygon chains
            // cleanly with adjacent edges; the curve's own parameter origin
            // is unrelated to the vertex.
            let seam_pt = topo.vertex(start_vid)?.point();
            let t0 = match curve {
                EdgeCurve::Circle(c) => Some(c.project(seam_pt)),
                EdgeCurve::Ellipse(e) => Some(e.project(seam_pt)),
                EdgeCurve::NurbsCurve(_) | EdgeCurve::Line => None,
            };
            // Traversal must start at the seam vertex in both directions:
            // forward covers [t0, t0 + TAU), reversed covers (t0, t0 + TAU]
            // walked backwards — the next edge supplies the closing point.
            #[allow(clippy::cast_precision_loss)]
            let params = |n: usize| -> Vec<f64> {
                if forward {
                    (0..n)
                        .map(|i| std::f64::consts::TAU.mul_add((i as f64) / (n as f64), 0.0))
                        .collect()
                } else {
                    (1..=n)
                        .rev()
                        .map(|i| std::f64::consts::TAU.mul_add((i as f64) / (n as f64), 0.0))
                        .collect()
                }
            };
            let sampled: Vec<Point3> = match (curve, t0) {
                (EdgeCurve::Circle(c), Some(t0)) => params(CLOSED_CURVE_SAMPLES)
                    .into_iter()
                    .map(|dt| c.evaluate(t0 + dt))
                    .collect(),
                (EdgeCurve::Ellipse(e), Some(t0)) => params(CLOSED_CURVE_SAMPLES)
                    .into_iter()
                    .map(|dt| e.evaluate(t0 + dt))
                    .collect(),
                _ => {
                    let mut s = sample_edge_curve(curve, CLOSED_CURVE_SAMPLES);
                    if !forward {
                        s.reverse();
                    }
                    s
                }
            };
            pts.extend(sampled);
            prev_end = Some(start_vid);
        } else {
            let vid = if forward { start_vid } else { end_vid };
            pts.push(topo.vertex(vid)?.point());
            prev_end = Some(if forward { end_vid } else { start_vid });
        }
    }

    Ok(pts)
}

/// Expand an AABB to account for surface curvature that may extend beyond
/// the wire vertices.
///
/// Plane and Cone surfaces are bounded by their vertices, so this is a no-op
/// for those types. Sphere, Cylinder, Torus, and NURBS surfaces can bulge
/// beyond the vertex-derived bounding box.
pub fn expand_aabb_for_surface(aabb: &mut Aabb3, surface: &FaceSurface) {
    match surface {
        FaceSurface::Sphere(s) => {
            let c = s.center();
            let r = s.radius();
            aabb_include(aabb, Point3::new(c.x() - r, c.y() - r, c.z() - r));
            aabb_include(aabb, Point3::new(c.x() + r, c.y() + r, c.z() + r));
        }
        FaceSurface::Cylinder(c) => {
            let origin = c.origin();
            let axis = c.axis();
            let r = c.radius();
            let rx = r * (1.0 - axis.x() * axis.x()).max(0.0).sqrt();
            let ry = r * (1.0 - axis.y() * axis.y()).max(0.0).sqrt();
            let rz = r * (1.0 - axis.z() * axis.z()).max(0.0).sqrt();
            for corner in [aabb.min, aabb.max] {
                let rel = Vec3::new(
                    corner.x() - origin.x(),
                    corner.y() - origin.y(),
                    corner.z() - origin.z(),
                );
                let t = axis.dot(rel);
                let coa = Point3::new(
                    origin.x() + axis.x() * t,
                    origin.y() + axis.y() * t,
                    origin.z() + axis.z() * t,
                );
                aabb_include(aabb, Point3::new(coa.x() - rx, coa.y() - ry, coa.z() - rz));
                aabb_include(aabb, Point3::new(coa.x() + rx, coa.y() + ry, coa.z() + rz));
            }
        }
        FaceSurface::Torus(t) => {
            let c = t.center();
            let outer_r = t.major_radius() + t.minor_radius();
            let axis = t.z_axis();
            let axial_offset = Vec3::new(
                axis.x() * t.minor_radius(),
                axis.y() * t.minor_radius(),
                axis.z() * t.minor_radius(),
            );
            aabb_include(
                aabb,
                Point3::new(
                    c.x() - outer_r + axial_offset.x().min(0.0),
                    c.y() - outer_r + axial_offset.y().min(0.0),
                    c.z() - outer_r + axial_offset.z().min(0.0),
                ),
            );
            aabb_include(
                aabb,
                Point3::new(
                    c.x() + outer_r + axial_offset.x().max(0.0),
                    c.y() + outer_r + axial_offset.y().max(0.0),
                    c.z() + outer_r + axial_offset.z().max(0.0),
                ),
            );
        }
        FaceSurface::Nurbs(nurbs) => {
            let (u_min, u_max) = nurbs.domain_u();
            let (v_min, v_max) = nurbs.domain_v();
            let n_samples = 8;
            for iu in 0..=n_samples {
                let u = u_min + (u_max - u_min) * (iu as f64) / (n_samples as f64);
                for iv in 0..=n_samples {
                    let v = v_min + (v_max - v_min) * (iv as f64) / (n_samples as f64);
                    aabb_include(aabb, nurbs.evaluate(u, v));
                }
            }
        }
        FaceSurface::Plane { .. } | FaceSurface::Cone(_) => {}
    }
}

/// Include a single point in an AABB.
fn aabb_include(aabb: &mut Aabb3, p: Point3) {
    *aabb = aabb.union(Aabb3 { min: p, max: p });
}

/// Compute the axis-aligned bounding box of a face.
///
/// Starts from the wire vertex positions and then expands for surface
/// curvature (spheres, cylinders, tori, NURBS).
///
/// # Errors
///
/// Returns an error if any topology entity referenced by the face is missing.
pub fn face_aabb(topo: &Topology, face_id: FaceId) -> Result<Aabb3, CheckError> {
    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let mut points = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        points.push(topo.vertex(edge.start())?.point());
        points.push(topo.vertex(edge.end())?.point());
    }
    let mut aabb = Aabb3::try_from_points(points.iter().copied())
        .ok_or_else(|| CheckError::ClassificationFailed("face has no vertices".into()))?;
    expand_aabb_for_surface(&mut aabb, face.surface());
    Ok(aabb)
}

/// Test whether a 3D point lies inside a 3D polygon by projecting onto the
/// dominant axis plane (the plane most aligned with the polygon normal).
pub fn point_in_polygon_3d(point: &Point3, polygon: &[Point3], normal: &Vec3) -> bool {
    use brepkit_math::predicates::point_in_polygon;

    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let (proj_pt, proj_poly): (Point2, Vec<Point2>) = if az >= ax && az >= ay {
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

    point_in_polygon(proj_pt, &proj_poly)
}

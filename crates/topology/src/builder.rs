//! Builder utilities for edges and wires.
//!
//! Provides ergonomic functions for creating topology from geometry,
//! equivalent to `BRepBuilderAPI_MakeEdge` and `BRepBuilderAPI_MakeWire`
//! in `OpenCascade`.

use std::f64::consts::PI;

use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};

use crate::Topology;
use crate::edge::{Edge, EdgeCurve, EdgeId};
use crate::face::{Face, FaceId, FaceSurface};
use crate::vertex::Vertex;
use crate::wire::{OrientedEdge, Wire, WireId};

/// Default vertex tolerance for builder operations.
const TOL: f64 = 1e-7;

/// Create a straight-line edge between two points.
///
/// Allocates vertices and the connecting edge.
///
/// # Errors
///
/// Returns an error if the points are coincident.
pub fn make_line_edge(
    topo: &mut Topology,
    start: Point3,
    end: Point3,
) -> Result<EdgeId, crate::TopologyError> {
    let tol = Tolerance::new();
    if (end - start).length_squared() < tol.linear * tol.linear {
        return Err(crate::TopologyError::NonManifold {
            reason: "degenerate edge: start and end points coincide".into(),
        });
    }

    let v0 = topo.add_vertex(Vertex::new(start, TOL));
    let v1 = topo.add_vertex(Vertex::new(end, TOL));
    Ok(topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line)))
}

/// Create a closed wire from an ordered list of points.
///
/// Each consecutive pair of points becomes a line edge, and the last
/// point connects back to the first.
///
/// # Errors
///
/// Returns an error if fewer than 3 points are provided.
pub fn make_polygon_wire(
    topo: &mut Topology,
    points: &[Point3],
) -> Result<WireId, crate::TopologyError> {
    let n = points.len();
    if n < 3 {
        return Err(crate::TopologyError::Empty {
            entity: "polygon (need at least 3 points)",
        });
    }

    let verts: Vec<_> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, TOL)))
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
///
/// # Errors
///
/// Returns an error if `n_sides < 3` or `radius` is non-positive.
pub fn make_regular_polygon_wire(
    topo: &mut Topology,
    radius: f64,
    n_sides: usize,
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

    make_polygon_wire(topo, &points)
}

/// Create a planar face from a closed wire.
///
/// Computes the face normal from the first three vertices of the wire.
///
/// # Errors
///
/// Returns an error if the wire has fewer than 3 edges or the normal
/// is degenerate.
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

    // Compute the plane normal from sample points along the wire.
    // For wires with fewer than 3 edges (e.g. a single circle), we cannot
    // rely on 3 distinct vertex positions. Instead, sample points along the
    // edge curves to obtain non-collinear positions.
    let sample_points = sample_wire_points(topo, edges)?;

    let normal = compute_plane_normal(&sample_points)?;
    let d = normal.x().mul_add(
        sample_points[0].x(),
        normal
            .y()
            .mul_add(sample_points[0].y(), normal.z() * sample_points[0].z()),
    );

    let face_id = topo.add_face(Face::new(wire_id, vec![], FaceSurface::Plane { normal, d }));

    Ok(face_id)
}

/// Sample at least 3 non-coincident points from a wire's edges.
///
/// For each edge, collects the start vertex and (for non-line curves) a
/// midpoint sample. This ensures that even a single closed circle edge
/// yields 3 well-spaced points for normal computation.
fn sample_wire_points(
    topo: &Topology,
    edges: &[OrientedEdge],
) -> Result<Vec<Point3>, crate::TopologyError> {
    let mut points = Vec::with_capacity(edges.len() * 2);

    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let start_vid = if oe.is_forward() {
            edge.start()
        } else {
            edge.end()
        };
        let start_pt = topo.vertex(start_vid)?.point();
        points.push(start_pt);

        // For curved edges, sample the midpoint to get a non-collinear point.
        match edge.curve() {
            EdgeCurve::Line => {}
            EdgeCurve::Circle(c) => {
                if edge.start() == edge.end() {
                    // Full circle: sample at 1/3 and 2/3 of parameter range
                    // in monotonic order so Newell's method gives correct
                    // normal sign.
                    let tau = std::f64::consts::TAU;
                    points.push(c.evaluate(tau / 3.0));
                    points.push(c.evaluate(2.0 * tau / 3.0));
                } else {
                    points.push(c.evaluate(PI));
                }
            }
            EdgeCurve::Ellipse(e) => {
                if edge.start() == edge.end() {
                    let tau = std::f64::consts::TAU;
                    points.push(e.evaluate(tau / 3.0));
                    points.push(e.evaluate(2.0 * tau / 3.0));
                } else {
                    points.push(e.evaluate(PI));
                }
            }
            EdgeCurve::NurbsCurve(nc) => {
                let knots = nc.knots();
                let mid_u = f64::midpoint(knots[0], knots[knots.len() - 1]);
                if edge.start() == edge.end() {
                    // Sample at 1/3 and 2/3 in monotonic order so Newell's
                    // method gives correct normal sign.
                    let third_u = knots[0] + (knots[knots.len() - 1] - knots[0]) / 3.0;
                    let two_third_u = knots[0] + 2.0 * (knots[knots.len() - 1] - knots[0]) / 3.0;
                    points.push(nc.evaluate(third_u));
                    points.push(nc.evaluate(two_third_u));
                } else {
                    points.push(nc.evaluate(mid_u));
                }
            }
        }
    }

    Ok(points)
}

/// Compute a unit plane normal from a set of sample points using Newell's
/// method. Works for any number of points >= 3.
fn compute_plane_normal(points: &[Point3]) -> Result<Vec3, crate::TopologyError> {
    if points.len() < 3 {
        return Err(crate::TopologyError::NonManifold {
            reason: "need at least 3 sample points for face normal".into(),
        });
    }

    // Newell's method: accumulate cross-product contributions from all
    // consecutive point pairs. More robust than using just 3 points when
    // the polygon has near-collinear segments.
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    let n = points.len();
    for i in 0..n {
        let curr = points[i];
        let next = points[(i + 1) % n];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }

    let len = (nx * nx + ny * ny + nz * nz).sqrt();
    if len < 1e-15 {
        return Err(crate::TopologyError::NonManifold {
            reason: "face normal is degenerate (collinear points)".into(),
        });
    }

    Ok(Vec3::new(nx / len, ny / len, nz / len))
}

/// Create a rectangular face on the XY plane centered at the origin.
///
/// # Errors
///
/// Returns an error if `width` or `height` is non-positive.
pub fn make_rectangle_face(
    topo: &mut Topology,
    width: f64,
    height: f64,
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

    let wid = make_polygon_wire(topo, &points)?;
    make_face_from_wire(topo, wid)
}

/// Create a circular polygon face on the XY plane centered at the origin.
///
/// The circle is approximated with `segments` straight edges.
///
/// # Errors
///
/// Returns an error if `radius` is non-positive or `segments < 3`.
pub fn make_circle_face(
    topo: &mut Topology,
    radius: f64,
    segments: usize,
) -> Result<FaceId, crate::TopologyError> {
    let wid = make_regular_polygon_wire(topo, radius, segments)?;
    make_face_from_wire(topo, wid)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::Point3;

    use super::*;
    use crate::Topology;

    #[test]
    fn make_line_edge_basic() {
        let mut topo = Topology::new();
        let eid = make_line_edge(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        )
        .unwrap();

        let edge = topo.edge(eid).unwrap();
        assert_ne!(edge.start(), edge.end());
    }

    #[test]
    fn make_line_edge_coincident_error() {
        let mut topo = Topology::new();
        let p = Point3::new(1.0, 2.0, 3.0);
        assert!(make_line_edge(&mut topo, p, p).is_err());
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
                &[Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)]
            )
            .is_err()
        );
    }

    #[test]
    fn make_regular_polygon_wire_hexagon() {
        let mut topo = Topology::new();
        let wid = make_regular_polygon_wire(&mut topo, 1.0, 6).unwrap();
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
        )
        .unwrap();

        let fid = make_face_from_wire(&mut topo, wid).unwrap();
        let face = topo.face(fid).unwrap();
        if let FaceSurface::Plane { normal, .. } = face.surface() {
            let tol = Tolerance::new();
            assert!(tol.approx_eq(normal.z().abs(), 1.0), "normal should be ±Z");
        }
    }

    #[test]
    fn make_rectangle_face_basic() {
        let mut topo = Topology::new();
        let fid = make_rectangle_face(&mut topo, 2.0, 3.0).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(matches!(face.surface(), FaceSurface::Plane { .. }));
    }

    #[test]
    fn make_rectangle_face_zero_error() {
        let mut topo = Topology::new();
        assert!(make_rectangle_face(&mut topo, 0.0, 1.0).is_err());
    }

    #[test]
    fn make_circle_face_basic() {
        let mut topo = Topology::new();
        let fid = make_circle_face(&mut topo, 1.0, 16).unwrap();
        let face = topo.face(fid).unwrap();
        assert!(matches!(face.surface(), FaceSurface::Plane { .. }));
    }

    #[test]
    fn make_circle_face_zero_radius_error() {
        let mut topo = Topology::new();
        assert!(make_circle_face(&mut topo, 0.0, 16).is_err());
    }
}

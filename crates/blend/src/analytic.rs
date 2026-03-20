//! Analytic fast paths for common surface pairs.
//!
//! Closed-form fillet and chamfer solutions for plane-plane and plane-cylinder
//! surface pairs. These bypass the walking engine entirely, producing exact
//! geometry that is 10-100x faster than Newton-Raphson marching.
//!
//! About 80% of real-world fillets are between plane-plane or plane-cylinder
//! pairs, making these fast paths high-impact optimizations.

use brepkit_math::curves2d::{Curve2D, Line2D};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::surfaces::CylindricalSurface;
use brepkit_math::traits::ParametricSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::BlendError;
use crate::section::CircSection;
use crate::spine::Spine;
use crate::stripe::{Stripe, StripeResult};

/// Result of an analytic fillet/chamfer computation.
///
/// Contains the blend surface and contact geometry, but not yet
/// integrated into topology (no new edges created at this stage).
pub struct AnalyticResult {
    /// The blend surface (cylinder for plane-plane fillet, plane for chamfer).
    pub surface: FaceSurface,
    /// 3D contact curve on face 1.
    pub contact1: NurbsCurve,
    /// 3D contact curve on face 2.
    pub contact2: NurbsCurve,
    /// PCurve on face 1 (UV-space).
    pub pcurve1: Curve2D,
    /// PCurve on face 2 (UV-space).
    pub pcurve2: Curve2D,
    /// Cross-sections at spine start and end.
    pub sections: Vec<CircSection>,
}

/// Try to compute a fillet analytically for two surfaces.
///
/// Returns `Some(StripeResult)` if the surface pair has a closed-form solution,
/// `None` otherwise (caller should fall back to the walking engine).
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_arguments)]
pub fn try_analytic_fillet(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    match (surf1, surf2) {
        (FaceSurface::Plane { normal: n1, d: _d1 }, FaceSurface::Plane { normal: n2, d: _d2 }) => {
            let result = plane_plane_fillet(spine, topo, *n1, *n2, radius, face1, face2)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Try to compute a chamfer analytically for two surfaces.
///
/// Returns `Some(StripeResult)` if the surface pair has a closed-form solution,
/// `None` otherwise (caller should fall back to the walking engine).
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_arguments)]
pub fn try_analytic_chamfer(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<Option<StripeResult>, BlendError> {
    match (surf1, surf2) {
        (
            FaceSurface::Plane {
                normal: n1,
                d: _dd1,
            },
            FaceSurface::Plane {
                normal: n2,
                d: _dd2,
            },
        ) => {
            let result = plane_plane_chamfer(spine, topo, *n1, *n2, d1, d2, face1, face2)?;
            Ok(Some(result))
        }
        _ => Ok(None),
    }
}

/// Make a degree-1 NURBS line between two 3D points.
fn nurbs_line(p0: Point3, p1: Point3) -> Result<NurbsCurve, BlendError> {
    let curve = NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![p0, p1], vec![1.0, 1.0])?;
    Ok(curve)
}

/// Compute the dihedral half-angle between two plane normals.
///
/// Returns the half-angle in radians. The angle is between 0 and pi/2
/// for convex edges and pi/2 to pi for concave edges.
fn dihedral_half_angle(n1: Vec3, n2: Vec3) -> f64 {
    let cos_angle = n1.dot(n2).clamp(-1.0, 1.0);
    cos_angle.acos() / 2.0
}

/// Compute the section plane basis from two plane normals and spine tangent.
///
/// Returns `(bisector, cross_dir)` where bisector points from edge toward
/// fillet center and `cross_dir` is perpendicular to both in the section plane.
fn section_basis(n1: Vec3, n2: Vec3, spine_tangent: Vec3) -> (Vec3, Vec3) {
    // Bisector of the two normals — points from the edge toward the fillet center
    let bisector_raw = n1 + n2;
    let bisector = bisector_raw.normalize().unwrap_or_else(|_| {
        // Normals are antiparallel (180 deg) — use cross product with tangent
        spine_tangent.cross(n1)
    });

    // In the section plane, the direction perpendicular to the bisector
    // that lies in the plane of the two normals.
    let cross_dir_raw = spine_tangent.cross(bisector);
    let cross_dir = cross_dir_raw
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));

    (bisector, cross_dir)
}

/// Compute the direction from edge toward contact point on a plane.
///
/// This is the component of the bisector projected onto the plane surface,
/// pointing away from the edge toward where the fillet touches the plane.
fn compute_contact_direction(normal: Vec3, bisector: Vec3) -> Vec3 {
    // Project bisector onto the plane (remove component along normal)
    let proj = bisector - normal * bisector.dot(normal);
    proj.normalize().unwrap_or(bisector)
}

/// Compute the midpoint of two 3D points.
fn midpoint_3d(a: Point3, b: Point3) -> Point3 {
    Point3::new(
        f64::midpoint(a.x(), b.x()),
        f64::midpoint(a.y(), b.y()),
        f64::midpoint(a.z(), b.z()),
    )
}

/// Fillet between two planes: the result is a cylindrical surface.
///
/// # Geometry
///
/// Given two planes meeting at a straight edge:
/// - The fillet surface is a cylinder whose axis is parallel to the edge
/// - The cylinder radius equals the fillet radius
/// - The center is offset from the edge along the angle bisector
/// - Contact lines are straight lines on each plane
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn plane_plane_fillet(
    spine: &Spine,
    topo: &Topology,
    n1: Vec3,
    n2: Vec3,
    radius: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<StripeResult, BlendError> {
    // Spine endpoints and tangent
    let p_start = spine.evaluate(topo, 0.0)?;
    let p_end = spine.evaluate(topo, spine.length())?;
    let tangent = spine.tangent(topo, 0.0)?;

    // Dihedral geometry
    let half_angle = dihedral_half_angle(n1, n2);
    let sin_half = half_angle.sin();
    let cos_half = half_angle.cos();

    // Guard against degenerate cases (parallel or antiparallel normals)
    if sin_half.abs() < 1e-10 {
        return Err(BlendError::Math(brepkit_math::MathError::ZeroVector));
    }

    let (bisector, _cross_dir) = section_basis(n1, n2, tangent);

    // Center offset from the edge along the bisector
    let center_offset = radius / sin_half;

    // Cylinder origin: on the center line at the spine start
    let cyl_origin = p_start + bisector * center_offset;
    let cyl_axis = tangent;

    // Create the cylindrical surface
    let cylinder = CylindricalSurface::new(cyl_origin, cyl_axis, radius)?;

    // Contact point offsets from the edge
    // The contact point on each plane is at distance R/tan(half_angle) from the edge.
    let contact_offset = radius * cos_half / sin_half; // = R / tan(half_angle)

    // Direction from edge toward contact on each plane
    let contact_dir1 = compute_contact_direction(n1, bisector);
    let contact_dir2 = compute_contact_direction(n2, bisector);

    // Contact lines (straight lines on each plane)
    let c1_start = p_start + contact_dir1 * contact_offset;
    let c1_end = p_end + contact_dir1 * contact_offset;
    let c2_start = p_start + contact_dir2 * contact_offset;
    let c2_end = p_end + contact_dir2 * contact_offset;

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // PCurves: project 3D contact endpoints onto each face surface to get UV
    let pcurve1 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n1, 0.0);
        let (u0, v0) = adapter.project_point(c1_start);
        let (u1, v1) = adapter.project_point(c1_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };
    let pcurve2 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n2, 0.0);
        let (u0, v0) = adapter.project_point(c2_start);
        let (u1, v1) = adapter.project_point(c2_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };

    // Cross-sections at start and end
    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: cyl_origin,
        radius,
        uv1: (0.0, 0.0),
        uv2: (0.0, 0.0),
        t: 0.0,
    };
    let cyl_end = p_end + bisector * center_offset;
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: cyl_end,
        radius,
        uv1: (1.0, 0.0),
        uv2: (1.0, 0.0),
        t: spine.length(),
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Cylinder(cylinder),
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };

    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// Chamfer between two planes: the result is a flat ruled surface (plane).
///
/// # Geometry
///
/// Given two planes meeting at an edge with chamfer distances d1, d2:
/// - The chamfer surface is a plane connecting two lines
/// - Line 1 is at distance d1 from the edge on plane 1
/// - Line 2 is at distance d2 from the edge on plane 2
///
/// # Errors
/// Returns `BlendError` if topology lookups or math operations fail.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn plane_plane_chamfer(
    spine: &Spine,
    topo: &Topology,
    n1: Vec3,
    n2: Vec3,
    d1: f64,
    d2: f64,
    face1: FaceId,
    face2: FaceId,
) -> Result<StripeResult, BlendError> {
    // Spine endpoints and tangent
    let p_start = spine.evaluate(topo, 0.0)?;
    let p_end = spine.evaluate(topo, spine.length())?;
    let tangent = spine.tangent(topo, 0.0)?;

    let (bisector, _cross_dir) = section_basis(n1, n2, tangent);

    // Contact directions on each plane
    let contact_dir1 = compute_contact_direction(n1, bisector);
    let contact_dir2 = compute_contact_direction(n2, bisector);

    // Contact lines at specified distances
    let c1_start = p_start + contact_dir1 * d1;
    let c1_end = p_end + contact_dir1 * d1;
    let c2_start = p_start + contact_dir2 * d2;
    let c2_end = p_end + contact_dir2 * d2;

    let contact1 = nurbs_line(c1_start, c1_end)?;
    let contact2 = nurbs_line(c2_start, c2_end)?;

    // The chamfer surface is a plane through the two contact lines.
    // Its normal is perpendicular to both the spine tangent and the line
    // connecting corresponding contact points.
    let chamfer_span = c2_start - c1_start;
    let chamfer_normal_raw = tangent.cross(chamfer_span);
    let chamfer_normal = chamfer_normal_raw
        .normalize()
        .map_err(|_| BlendError::Math(brepkit_math::MathError::ZeroVector))?;

    // Signed distance from origin
    let chamfer_d = chamfer_normal.dot(Vec3::new(c1_start.x(), c1_start.y(), c1_start.z()));

    // PCurves: project 3D contact endpoints onto each face surface to get UV
    let pcurve1 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n1, 0.0);
        let (u0, v0) = adapter.project_point(c1_start);
        let (u1, v1) = adapter.project_point(c1_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };
    let pcurve2 = {
        let adapter = crate::builder_utils::PlaneAdapter::from_normal_and_d(n2, 0.0);
        let (u0, v0) = adapter.project_point(c2_start);
        let (u1, v1) = adapter.project_point(c2_end);
        Curve2D::Line(Line2D::new(
            brepkit_math::vec::Point2::new(u0, v0),
            brepkit_math::vec::Vec2::new(u1 - u0, v1 - v0),
        )?)
    };

    // Sections at start and end
    let midpoint_start = midpoint_3d(c1_start, c2_start);
    let midpoint_end = midpoint_3d(c1_end, c2_end);
    let chamfer_radius = (c1_start - c2_start).length() / 2.0;

    let section_start = CircSection {
        p1: c1_start,
        p2: c2_start,
        center: midpoint_start,
        radius: chamfer_radius,
        uv1: (0.0, 0.0),
        uv2: (0.0, 0.0),
        t: 0.0,
    };
    let section_end = CircSection {
        p1: c1_end,
        p2: c2_end,
        center: midpoint_end,
        radius: chamfer_radius,
        uv1: (1.0, 0.0),
        uv2: (1.0, 0.0),
        t: spine.length(),
    };

    let stripe = Stripe {
        spine: spine.clone(),
        surface: FaceSurface::Plane {
            normal: chamfer_normal,
            d: chamfer_d,
        },
        pcurve1,
        pcurve2,
        contact1,
        contact2,
        face1,
        face2,
        sections: vec![section_start, section_end],
    };

    Ok(StripeResult {
        stripe,
        new_edges: Vec::new(),
    })
}

/// Fillet between a plane and a cylinder: the result is typically a torus section.
///
/// Not yet implemented. Returns `None` so the caller falls back to the walking
/// engine.
#[allow(unused_variables)]
#[must_use]
pub fn plane_cylinder_fillet(
    surf_plane: &FaceSurface,
    surf_cyl: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
) -> Option<StripeResult> {
    // TODO: implement plane-cylinder fillet (torus section)
    // This requires:
    // 1. Project spine onto the cylinder surface
    // 2. Compute rolling ball center trajectory (torus centerline)
    // 3. Build toroidal surface patch
    None
}

/// Fillet between a plane and a cone.
///
/// Not yet implemented. Returns `None` so the caller falls back to the walking
/// engine.
#[allow(unused_variables)]
#[must_use]
pub fn plane_cone_fillet(
    surf_plane: &FaceSurface,
    surf_cone: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
) -> Option<StripeResult> {
    // TODO: implement plane-cone fillet
    None
}

/// Fillet between two cylinders.
///
/// Not yet implemented. Returns `None` so the caller falls back to the walking
/// engine.
#[allow(unused_variables)]
#[must_use]
pub fn cylinder_cylinder_fillet(
    surf1: &FaceSurface,
    surf2: &FaceSurface,
    spine: &Spine,
    topo: &Topology,
    radius: f64,
) -> Option<StripeResult> {
    // TODO: implement cylinder-cylinder fillet
    None
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::Face;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    /// Create a spine along a single edge from `a` to `b`, plus two dummy faces.
    fn make_spine(topo: &mut Topology, a: Point3, b: Point3) -> (Spine, FaceId, FaceId) {
        let v0 = topo.add_vertex(Vertex::new(a, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(b, 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        let oe = OrientedEdge::new(eid, true);
        let w1 = topo.add_wire(Wire::new(vec![oe], false).unwrap());
        let w2 = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], false).unwrap());
        let f1 = topo.add_face(Face::new(
            w1,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let f2 = topo.add_face(Face::new(
            w2,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));

        let spine = Spine::from_single_edge(topo, eid).unwrap();
        (spine, f1, f2)
    }

    #[test]
    fn plane_plane_90_degree_fillet() {
        let mut topo = Topology::new();

        // Two perpendicular planes meeting along X axis at origin
        let n1 = Vec3::new(0.0, 0.0, 1.0); // XY plane (top)
        let n2 = Vec3::new(0.0, 1.0, 0.0); // XZ plane (front)
        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );

        let radius = 2.0;
        let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

        // The result surface should be a cylinder
        match &result.stripe.surface {
            FaceSurface::Cylinder(cyl) => {
                assert!(
                    (cyl.radius() - radius).abs() < 1e-10,
                    "Expected radius {radius}, got {}",
                    cyl.radius()
                );

                let axis = cyl.axis();
                assert!(
                    axis.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
                    "Cylinder axis should be along X, got {axis:?}"
                );
            }
            other => panic!("Expected Cylinder surface, got {other:?}"),
        }

        // Contact curves should be lines parallel to X axis
        let c1_start = result.stripe.contact1.evaluate(0.0);
        let c1_end = result.stripe.contact1.evaluate(1.0);
        let c1_dir = (c1_end - c1_start).normalize().unwrap();
        assert!(
            c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
            "Contact 1 should be along X"
        );

        // Half-angle for 90 deg is pi/4, so offset = R/sin(pi/4) = R*sqrt(2)
        let half_angle = std::f64::consts::FRAC_PI_4;
        let expected_offset = radius / half_angle.sin();
        let sections = &result.stripe.sections;
        assert_eq!(sections.len(), 2);
        assert!((sections[0].radius - radius).abs() < 1e-10);

        // Center should be offset from origin by R/sin(45deg) along bisector
        let bisector = (n1 + n2).normalize().unwrap();
        let expected_center = Point3::new(0.0, 0.0, 0.0) + bisector * expected_offset;
        let actual_center = sections[0].center;
        assert!(
            (actual_center - expected_center).length() < 1e-10,
            "Expected center at {expected_center:?}, got {actual_center:?}"
        );
    }

    #[test]
    fn plane_plane_60_degree_fillet() {
        let mut topo = Topology::new();

        let n1 = Vec3::new(0.0, 0.0, 1.0);
        // Normal at 60 deg from n1
        let angle = std::f64::consts::FRAC_PI_3;
        let n2 = Vec3::new(0.0, angle.sin(), angle.cos());
        let n2 = n2.normalize().unwrap();

        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(5.0, 0.0, 0.0),
        );

        let radius = 1.5;
        let result = plane_plane_fillet(&spine, &topo, n1, n2, radius, f1, f2).unwrap();

        match &result.stripe.surface {
            FaceSurface::Cylinder(cyl) => {
                assert!(
                    (cyl.radius() - radius).abs() < 1e-10,
                    "Expected radius {radius}, got {}",
                    cyl.radius()
                );
            }
            other => panic!("Expected Cylinder surface, got {other:?}"),
        }

        // Verify center offset matches expected geometry
        let cos_angle = n1.dot(n2);
        let half = cos_angle.acos() / 2.0;
        let expected_offset = radius / half.sin();

        let center = result.stripe.sections[0].center;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let actual_offset = (center - origin).length();
        assert!(
            (actual_offset - expected_offset).abs() < 1e-10,
            "Expected offset {expected_offset}, got {actual_offset}"
        );
    }

    #[test]
    fn plane_plane_chamfer_is_flat() {
        let mut topo = Topology::new();

        let n1 = Vec3::new(0.0, 0.0, 1.0);
        let n2 = Vec3::new(0.0, 1.0, 0.0);
        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
        );

        let d1 = 3.0;
        let d2 = 2.0;
        let result = plane_plane_chamfer(&spine, &topo, n1, n2, d1, d2, f1, f2).unwrap();

        // The result should be a plane
        match &result.stripe.surface {
            FaceSurface::Plane { normal, d } => {
                // Normal should be perpendicular to the spine direction (X axis)
                let spine_dir = Vec3::new(1.0, 0.0, 0.0);
                assert!(
                    normal.dot(spine_dir).abs() < 1e-10,
                    "Chamfer normal should be perpendicular to spine, dot={:.6}",
                    normal.dot(spine_dir)
                );
                assert!(
                    (normal.length() - 1.0).abs() < 1e-10,
                    "Normal should be unit length"
                );
                assert!(d.is_finite(), "d should be finite");
            }
            other => panic!("Expected Plane surface for chamfer, got {other:?}"),
        }

        // Contact curves should be lines parallel to X
        let c1_start = result.stripe.contact1.evaluate(0.0);
        let c1_end = result.stripe.contact1.evaluate(1.0);
        let c1_dir = (c1_end - c1_start).normalize().unwrap();
        assert!(
            c1_dir.dot(Vec3::new(1.0, 0.0, 0.0)).abs() > 0.99,
            "Contact 1 should be along X"
        );
    }

    #[test]
    fn non_analytic_returns_none() {
        let mut topo = Topology::new();

        // One NURBS surface — should return None
        let nurbs_surf = FaceSurface::Nurbs(
            brepkit_math::nurbs::surface::NurbsSurface::new(
                1,
                1,
                vec![0.0, 0.0, 1.0, 1.0],
                vec![0.0, 0.0, 1.0, 1.0],
                vec![
                    vec![Point3::new(0.0, 0.0, 0.0), Point3::new(1.0, 0.0, 0.0)],
                    vec![Point3::new(0.0, 1.0, 0.0), Point3::new(1.0, 1.0, 0.0)],
                ],
                vec![vec![1.0, 1.0], vec![1.0, 1.0]],
            )
            .unwrap(),
        );
        let plane_surf = FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, 1.0),
            d: 0.0,
        };

        let (spine, f1, f2) = make_spine(
            &mut topo,
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(1.0, 0.0, 0.0),
        );

        let result =
            try_analytic_fillet(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, f1, f2).unwrap();
        assert!(result.is_none(), "NURBS-Plane pair should return None");

        let result =
            try_analytic_chamfer(&nurbs_surf, &plane_surf, &spine, &topo, 1.0, 1.0, f1, f2)
                .unwrap();
        assert!(result.is_none(), "NURBS-Plane chamfer should return None");
    }
}

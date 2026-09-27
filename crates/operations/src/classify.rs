//! Point-in-solid classification by ray casting.
//!
//! Every classifier here reads the point with the check crate's ray cast:
//! rays from the point count their crossings with the solid's faces, each
//! face read on its own surface and edges, and three rays vote.

use brepkit_check::classify::{ClassifyOptions, PointClassification as CheckClass};
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::solid::SolidId;

use crate::OperationsError;

/// Result of classifying a point relative to a solid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PointClassification {
    /// The point is inside the solid.
    Inside,
    /// The point is outside the solid.
    Outside,
    /// The point is on the boundary (within tolerance).
    OnBoundary,
}

/// Classifies a point relative to a solid by ray casting.
///
/// `tolerance` is the distance within which the point reads
/// [`PointClassification::OnBoundary`], measured to the faces as trimmed.
/// `deflection` is unused: no face is tessellated.
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    _deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    let options = ClassifyOptions {
        tolerance,
        ..ClassifyOptions::default()
    };
    Ok(
        match brepkit_check::classify::classify_point(topo, solid, point, &options)? {
            CheckClass::Inside => PointClassification::Inside,
            CheckClass::Outside => PointClassification::Outside,
            CheckClass::OnBoundary => PointClassification::OnBoundary,
        },
    )
}

/// Classifies a point relative to a solid; the same ray cast as
/// [`classify_point`].
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_winding(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    classify_point(topo, solid, point, deflection, tolerance)
}

/// Classifies a point relative to a solid; the same ray cast as
/// [`classify_point`].
///
/// # Errors
/// Returns an error if the solid or its faces are invalid.
pub fn classify_point_robust(
    topo: &Topology,
    solid: SolidId,
    point: Point3,
    deflection: f64,
    tolerance: f64,
) -> Result<PointClassification, OperationsError> {
    classify_point(topo, solid, point, deflection, tolerance)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::primitives::{make_box, make_cone, make_cylinder, make_sphere, make_torus};
    use brepkit_math::vec::Vec3;
    use brepkit_topology::face::FaceSurface;
    use std::f64::consts::PI;

    #[test]
    fn point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_on_boundary_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(1.0, 1.0, 2.0), 0.1, 1e-3).unwrap();
        assert_eq!(result, PointClassification::OnBoundary);
    }

    #[test]
    fn point_outside_negative_direction() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point(&topo, solid, Point3::new(-5.0, -5.0, -5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    /// A point floating in an open pocket is outside the solid.
    ///
    /// The pocket makes the top face a ring with an inner wire, and the ray
    /// leaves through the middle of that hole. Counting the hole as a crossing
    /// flips the parity and reports the empty pocket as solid material.
    #[test]
    fn point_in_open_pocket_is_outside() {
        let mut topo = Topology::new();
        let plate = make_box(&mut topo, 100.0, 100.0, 10.0).unwrap();
        let tool = make_box(&mut topo, 60.0, 60.0, 4.0).unwrap();
        crate::transform::transform_solid(
            &mut topo,
            tool,
            &brepkit_math::mat::Mat4::translation(20.0, 20.0, 6.0),
        )
        .unwrap();
        let pocketed =
            crate::boolean::boolean(&mut topo, crate::boolean::BooleanOp::Cut, plate, tool)
                .unwrap();

        let result =
            classify_point(&topo, pocketed, Point3::new(50.0, 50.0, 8.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_near_corner() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.9, 0.9, 0.9), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_inside_cylinder() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_cylinder() {
        let mut topo = Topology::new();
        let solid = make_cylinder(&mut topo, 2.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_sphere() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 3.0, 32).unwrap();

        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_sphere() {
        let mut topo = Topology::new();
        let solid = make_sphere(&mut topo, 3.0, 32).unwrap();

        let result = classify_point(&topo, solid, Point3::new(5.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_cone() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 5.0).unwrap();

        // Point on the axis, inside the cone
        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_cone() {
        let mut topo = Topology::new();
        let solid = make_cone(&mut topo, 2.0, 1.0, 5.0).unwrap();

        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 2.5), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_inside_torus() {
        let mut topo = Topology::new();
        // major=3, minor=1 → tube center at distance 3 from origin
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point inside the tube (on the x-axis at distance 3 from origin)
        let result = classify_point(&topo, solid, Point3::new(3.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn point_outside_torus() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point at origin — in the hole of the torus
        let result = classify_point(&topo, solid, Point3::new(0.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn point_outside_torus_far() {
        let mut topo = Topology::new();
        let solid = make_torus(&mut topo, 3.0, 1.0, 32).unwrap();

        // Point far from torus
        let result = classify_point(&topo, solid, Point3::new(10.0, 0.0, 0.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    /// Build the partial-turn revolve of a circle profile: one trimmed torus
    /// band (wire = 2 closed rims + doubled seam) plus 2 planar disc caps.
    fn make_partial_torus(
        topo: &mut Topology,
        big_r: f64,
        rho: f64,
        angle: f64,
    ) -> brepkit_topology::solid::SolidId {
        use brepkit_math::curves::Circle3D;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let circ =
            Circle3D::new(Point3::new(big_r, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), rho).unwrap();
        let p0 = circ.evaluate(0.0);
        let v0 = topo.add_vertex(Vertex::new(p0, 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v0, EdgeCurve::Circle(circ)));
        let wire = Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        let face = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 1.0, 0.0),
                d: 0.0,
            },
        ));
        crate::revolve::revolve(
            topo,
            face,
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            angle,
        )
        .unwrap()
    }

    /// Regression: the trimmed-torus band of a partial-turn revolve. Two
    /// stacked defects made every interior point read Outside: the local
    /// Ferrari ray-torus quartic missed real roots and emitted off-surface
    /// spurious ones, and the UV boundary sampled closed rim circles from the
    /// curve's parameter origin, so the two rims entered the periodic unwrap
    /// at incoherent phases and the UV polygon rejected real band hits.
    #[test]
    fn partial_turn_torus_band_classification() {
        let (big_r, rho, angle) = (6.0_f64, 2.0_f64, 2.0 * PI / 3.0);
        let mut topo = Topology::new();
        let solid = make_partial_torus(&mut topo, big_r, rho, angle);

        let mid = angle / 2.0;
        let inside = [
            Point3::new(big_r * mid.cos(), big_r * mid.sin(), 0.0),
            Point3::new(big_r * mid.cos(), big_r * mid.sin(), 1.0),
            Point3::new(big_r * mid.cos(), big_r * mid.sin(), -1.0),
            Point3::new(big_r * 0.05f64.cos(), big_r * 0.05f64.sin(), 0.0),
            Point3::new(
                big_r * (angle - 0.05).cos(),
                big_r * (angle - 0.05).sin(),
                0.0,
            ),
            Point3::new((big_r - 1.5) * mid.cos(), (big_r - 1.5) * mid.sin(), 0.0),
            Point3::new((big_r + 1.5) * mid.cos(), (big_r + 1.5) * mid.sin(), 0.0),
        ];
        for p in inside {
            let result = classify_point(&topo, solid, p, 0.05, 1e-6).unwrap();
            assert_eq!(result, PointClassification::Inside, "probe {p:?}");
        }

        let outside = [
            Point3::new(big_r * mid.cos(), big_r * mid.sin(), 2.5),
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(-big_r, 0.0, 0.0),
            Point3::new(
                big_r * (angle + 0.1).cos(),
                big_r * (angle + 0.1).sin(),
                0.0,
            ),
            Point3::new(big_r * (-0.1f64).cos(), big_r * (-0.1f64).sin(), 0.0),
        ];
        for p in outside {
            let result = classify_point(&topo, solid, p, 0.05, 1e-6).unwrap();
            assert_eq!(result, PointClassification::Outside, "probe {p:?}");
        }
    }

    /// A full-turn revolve (single closed torus face, seam edges only) must
    /// keep classifying correctly alongside the partial-band fix.
    #[test]
    fn full_turn_torus_classification() {
        let (big_r, rho) = (6.0_f64, 2.0_f64);
        let mut topo = Topology::new();
        let solid = make_partial_torus(&mut topo, big_r, rho, 2.0 * PI);

        for theta in [0.0_f64, 1.0, 2.5, 4.0, 5.5] {
            let p = Point3::new(big_r * theta.cos(), big_r * theta.sin(), 0.0);
            let result = classify_point(&topo, solid, p, 0.05, 1e-6).unwrap();
            assert_eq!(result, PointClassification::Inside, "tube center {theta}");
        }
        for p in [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(big_r, 0.0, 2.5),
            Point3::new(2.0 * big_r, 0.0, 0.0),
        ] {
            let result = classify_point(&topo, solid, p, 0.05, 1e-6).unwrap();
            assert_eq!(result, PointClassification::Outside, "probe {p:?}");
        }
    }

    #[test]
    fn winding_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_winding(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn winding_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_winding(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }

    #[test]
    fn robust_point_inside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_robust(&topo, solid, Point3::new(1.0, 1.0, 1.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Inside);
    }

    #[test]
    fn robust_point_outside_box() {
        let mut topo = Topology::new();
        let solid = make_box(&mut topo, 2.0, 2.0, 2.0).unwrap();

        let result =
            classify_point_robust(&topo, solid, Point3::new(5.0, 5.0, 5.0), 0.1, 1e-6).unwrap();
        assert_eq!(result, PointClassification::Outside);
    }
}

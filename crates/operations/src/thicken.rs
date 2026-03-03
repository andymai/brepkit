//! Thicken a face into a solid by offsetting along its normal.
//!
//! A convenience operation that extrudes a face along its own normal
//! direction. Supports planar, NURBS, and analytic surface faces.
//! Equivalent to a specialized form of
//! `BRepOffsetAPI_MakeOffsetShape` in `OpenCascade`.

use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::extrude::extrude;

/// Thicken a face into a solid by extruding along its normal.
///
/// Positive `thickness` extrudes in the face normal direction;
/// negative extrudes in the opposite direction.
///
/// For planar faces, the normal is the plane's normal vector.
/// For NURBS and analytic faces, the normal is estimated at the
/// surface center (u=0.5, v=0.5).
///
/// # Errors
///
/// Returns an error if `thickness` is zero or the extrusion fails.
pub fn thicken(
    topo: &mut Topology,
    face: FaceId,
    thickness: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = brepkit_math::tolerance::Tolerance::new();

    if tol.approx_eq(thickness, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "thickness must be non-zero".into(),
        });
    }

    let face_data = topo.face(face)?;
    let normal = face_normal(face_data.surface())?;

    let (direction, distance) = if thickness > 0.0 {
        (normal, thickness)
    } else {
        (-normal, -thickness)
    };

    extrude(topo, face, direction, distance)
}

/// Extract or compute a representative outward normal for any surface type.
fn face_normal(surface: &FaceSurface) -> Result<Vec3, crate::OperationsError> {
    match surface {
        FaceSurface::Plane { normal, .. } => Ok(*normal),
        FaceSurface::Nurbs(nurbs) => {
            // Evaluate at the surface center.
            nurbs
                .normal(0.5, 0.5)
                .map_err(|e| crate::OperationsError::InvalidInput {
                    reason: format!("NURBS normal computation failed: {e}"),
                })
        }
        FaceSurface::Cylinder(cyl) => Ok(cyl.normal(0.0, 0.0)),
        FaceSurface::Cone(cone) => Ok(cone.normal(0.0, 0.0)),
        FaceSurface::Sphere(sphere) => Ok(sphere.normal(0.0, 0.0)),
        FaceSurface::Torus(torus) => Ok(torus.normal(0.0, 0.0)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    #[test]
    fn thicken_positive() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = thicken(&mut topo, face, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 2.0),
            "1×1 face thickened by 2 should have volume ~2.0, got {vol}"
        );
    }

    #[test]
    fn thicken_negative() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let solid = thicken(&mut topo, face, -1.5).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.5),
            "1×1 face thickened by -1.5 should have volume ~1.5, got {vol}"
        );
    }

    #[test]
    fn thicken_zero_error() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);
        assert!(thicken(&mut topo, face, 0.0).is_err());
    }

    #[test]
    fn thicken_nurbs_face() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_math::vec::Point3;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::{Face, FaceSurface};
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        // Build a flat NURBS face.
        let surface = NurbsSurface::new(
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
        .unwrap();

        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.edges.alloc(Edge::new(v3, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
                OrientedEdge::new(e3, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);
        let face = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

        let solid = thicken(&mut topo, face, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "thickened NURBS face should have positive volume, got {vol}"
        );
    }
}

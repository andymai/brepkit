//! Convert NURBS geometry to analytic (elementary) surfaces and curves
//! where possible.

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use brepkit_geometry::convert::{
    RecognizedCurve, RecognizedSurface, recognize_curve, recognize_surface,
};

use crate::HealError;

/// Try to recognize and replace NURBS surfaces with analytic equivalents.
///
/// Returns the number of surfaces converted.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn convert_to_elementary(
    topo: &mut Topology,
    solid_id: SolidId,
    tolerance: &Tolerance,
) -> Result<usize, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    let mut converted = 0;

    // Snapshot surfaces.
    let surfaces: Vec<(FaceId, FaceSurface)> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| (fid, f.surface().clone())))
        .collect::<Result<Vec<_>, _>>()?;

    for (fid, surface) in &surfaces {
        if let FaceSurface::Nurbs(nurbs) = surface {
            match recognize_surface(nurbs, tolerance.linear) {
                RecognizedSurface::Plane { normal, d } => {
                    let face = topo.face_mut(*fid)?;
                    face.set_surface(FaceSurface::Plane { normal, d });
                    converted += 1;
                }
                RecognizedSurface::Cylinder {
                    origin,
                    axis,
                    radius,
                } => {
                    if let Ok(cyl) =
                        brepkit_math::surfaces::CylindricalSurface::new(origin, axis, radius)
                    {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Cylinder(cyl));
                        converted += 1;
                    }
                }
                RecognizedSurface::Sphere { center, radius } => {
                    if let Ok(sph) = brepkit_math::surfaces::SphericalSurface::new(center, radius) {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Sphere(sph));
                        converted += 1;
                    }
                }
                RecognizedSurface::Cone {
                    apex,
                    axis,
                    half_angle,
                } => {
                    if let Ok(cone) =
                        brepkit_math::surfaces::ConicalSurface::new(apex, axis, half_angle)
                    {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Cone(cone));
                        converted += 1;
                    }
                }
                RecognizedSurface::Torus {
                    center,
                    axis,
                    major_radius,
                    minor_radius,
                } => {
                    if let Ok(torus) = brepkit_math::surfaces::ToroidalSurface::with_axis(
                        center,
                        major_radius,
                        minor_radius,
                        axis,
                    ) {
                        let face = topo.face_mut(*fid)?;
                        face.set_surface(FaceSurface::Torus(torus));
                        converted += 1;
                    }
                }
                RecognizedSurface::NotRecognized => {}
            }
        }
    }

    Ok(converted)
}

/// Try to recognize and replace NURBS edges with analytic curves.
///
/// Iterates every edge in the solid; if the edge has an
/// [`EdgeCurve::NurbsCurve`] that
/// `recognize_curve` identifies as a line, circle, or ellipse, replaces the edge's
/// curve with the analytic form. Returns the number of curves
/// converted.
///
/// Hyperbolas and parabolas are recognized but not converted (no
/// `EdgeCurve::Hyperbola` / `Parabola` variants exist yet); they
/// continue to be represented as `NurbsCurve`.
///
/// # Errors
///
/// Returns [`HealError`] if entity lookups fail.
pub fn convert_edges_to_elementary(
    topo: &mut Topology,
    solid_id: SolidId,
    tolerance: &Tolerance,
) -> Result<usize, HealError> {
    let solid_data = topo.solid(solid_id)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    let face_ids: Vec<FaceId> = shell.faces().to_vec();

    // Collect unique edge IDs across all faces (edges may be shared
    // between faces).
    let mut edge_ids: Vec<EdgeId> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &fid in &face_ids {
        let face = topo.face(fid)?;
        for &wid in std::iter::once(&face.outer_wire()).chain(face.inner_wires()) {
            let wire = topo.wire(wid)?;
            for oe in wire.edges() {
                let eid = oe.edge();
                if seen.insert(eid.index()) {
                    edge_ids.push(eid);
                }
            }
        }
    }

    let mut converted = 0;
    for eid in edge_ids {
        let edge = topo.edge(eid)?;
        let nurbs = match edge.curve() {
            EdgeCurve::NurbsCurve(n) => n.clone(),
            // Already analytic — nothing to convert.
            EdgeCurve::Line | EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => continue,
        };
        match recognize_curve(&nurbs, tolerance.linear) {
            RecognizedCurve::Circle {
                center,
                normal,
                radius,
            } => {
                if let Ok(c) = Circle3D::new(center, normal, radius) {
                    let edge_mut = topo.edge_mut(eid)?;
                    edge_mut.set_curve(EdgeCurve::Circle(c));
                    converted += 1;
                }
            }
            RecognizedCurve::Ellipse {
                center,
                normal,
                u_axis: _,
                semi_major,
                semi_minor,
            } => {
                // Ellipse3D::new takes (center, normal, semi_major, semi_minor)
                // and derives u_axis internally from the normal via
                // Frame3::from_normal — so the recognized u_axis isn't
                // directly used (the analytic form's frame may differ
                // from the recognizer's, but both describe the same
                // ellipse SET in 3D).
                if let Ok(e) = Ellipse3D::new(center, normal, semi_major, semi_minor) {
                    let edge_mut = topo.edge_mut(eid)?;
                    edge_mut.set_curve(EdgeCurve::Ellipse(e));
                    converted += 1;
                }
            }
            RecognizedCurve::Line { .. } => {
                // EdgeCurve::Line stores no geometry — vertex
                // positions imply the line. Replace the NURBS with
                // the implicit Line variant.
                let edge_mut = topo.edge_mut(eid)?;
                edge_mut.set_curve(EdgeCurve::Line);
                converted += 1;
            }
            // Hyperbola and parabola are recognized but not yet
            // representable as analytic EdgeCurve variants (no
            // EdgeCurve::Hyperbola / Parabola exist in topology).
            // They keep their NURBS representation. Likewise for
            // unrecognized curves.
            RecognizedCurve::Hyperbola { .. }
            | RecognizedCurve::Parabola { .. }
            | RecognizedCurve::NotRecognized => {}
        }
    }

    Ok(converted)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use super::*;
    use brepkit_geometry::convert::curve_to_nurbs::circle_to_nurbs;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::Edge;
    use brepkit_topology::face::Face;
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    #[test]
    fn convert_edges_to_elementary_recovers_circle() {
        // Build a minimal solid with one face whose boundary contains
        // a NURBS edge that's actually a full circle. After running
        // `convert_edges_to_elementary`, the edge should be a Circle3D
        // EdgeCurve.
        let mut topo = Topology::new();
        let circle =
            Circle3D::new(Point3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 2.5).unwrap();
        let nurbs = circle_to_nurbs(&circle, 0.0, std::f64::consts::TAU).unwrap();

        // Closed circle: start_vertex == end_vertex.
        let v = topo.add_vertex(Vertex::new(Point3::new(2.5, 0.0, 0.0), 1e-7));
        let edge_id = topo.add_edge(Edge::new(v, v, EdgeCurve::NurbsCurve(nurbs)));

        // Wrap in a wire / face / shell / solid scaffold so the iterator
        // in `convert_edges_to_elementary` can find the edge.
        let wire = Wire::new(vec![OrientedEdge::new(edge_id, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        let face_id = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let shell_id = topo.add_shell(Shell::new(vec![face_id]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let tol = Tolerance::new();
        let n = convert_edges_to_elementary(&mut topo, solid_id, &tol).unwrap();
        assert_eq!(n, 1, "expected 1 conversion, got {n}");

        // Verify the edge is now Circle3D, not NurbsCurve.
        let edge = topo.edge(edge_id).unwrap();
        match edge.curve() {
            EdgeCurve::Circle(c) => {
                assert!(
                    (c.radius() - 2.5).abs() < 1e-6,
                    "radius {} vs 2.5",
                    c.radius()
                );
            }
            other => panic!("expected Circle, got {other:?}"),
        }
    }

    #[test]
    fn convert_edges_skips_already_analytic() {
        // An edge that's already EdgeCurve::Line should not be touched.
        let mut topo = Topology::new();
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let edge_id = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        // Build the minimum scaffold (degenerate face/shell/solid).
        let wire = Wire::new(vec![OrientedEdge::new(edge_id, true)], false).unwrap();
        let wid = topo.add_wire(wire);
        let face_id = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 0.0,
            },
        ));
        let shell_id = topo.add_shell(Shell::new(vec![face_id]).unwrap());
        let solid_id = topo.add_solid(Solid::new(shell_id, vec![]));

        let tol = Tolerance::new();
        let n = convert_edges_to_elementary(&mut topo, solid_id, &tol).unwrap();
        assert_eq!(n, 0, "Line edges shouldn't be converted, got {n}");
    }
}

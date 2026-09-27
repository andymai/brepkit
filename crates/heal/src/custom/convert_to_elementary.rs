//! Convert NURBS geometry to analytic (elementary) surfaces and curves
//! where possible.

use brepkit_math::curves::{Circle3D, Ellipse3D};
use brepkit_math::tolerance::Tolerance;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::solid_faces;
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
    // Walk outer shell *and* inner (cavity) shells via the topology
    // explorer helper. Hollow solids (cavities from `shell_op` or
    // boolean cuts) hold faces in `Solid::inner_shells()`, and
    // visiting only the outer shell would silently leave those faces
    // unconverted.
    let face_ids: Vec<FaceId> = solid_faces(topo, solid_id)?;

    let mut converted = 0;

    let surfaces: Vec<(FaceId, FaceSurface)> = face_ids
        .iter()
        .map(|&fid| topo.face(fid).map(|f| (fid, f.surface().clone())))
        .collect::<Result<Vec<_>, _>>()?;

    for (fid, surface) in &surfaces {
        let FaceSurface::Nurbs(nurbs) = surface else {
            continue;
        };
        let replacement = match recognize_surface(nurbs, tolerance.linear) {
            RecognizedSurface::Plane { normal, d } => Some(FaceSurface::Plane { normal, d }),
            RecognizedSurface::Cylinder {
                origin,
                axis,
                radius,
            } => brepkit_math::surfaces::CylindricalSurface::new(origin, axis, radius)
                .ok()
                .map(FaceSurface::Cylinder),
            RecognizedSurface::Sphere { center, radius } => {
                brepkit_math::surfaces::SphericalSurface::new(center, radius)
                    .ok()
                    .map(FaceSurface::Sphere)
            }
            RecognizedSurface::Cone {
                apex,
                axis,
                half_angle,
            } => brepkit_math::surfaces::ConicalSurface::new(apex, axis, half_angle)
                .ok()
                .map(FaceSurface::Cone),
            RecognizedSurface::Torus {
                center,
                axis,
                major_radius,
                minor_radius,
            } => brepkit_math::surfaces::ToroidalSurface::with_axis(
                center,
                major_radius,
                minor_radius,
                axis,
            )
            .ok()
            .map(FaceSurface::Torus),
            RecognizedSurface::NotRecognized => None,
        };
        let Some(replacement) = replacement else {
            continue;
        };
        let opposed = normals_oppose(nurbs, &replacement);
        // The face's pcurves live in the NURBS parameter space.
        super::convert_to_bspline::drop_face_pcurves(topo, *fid)?;
        match replacement {
            // A plane stores its outward normal and is left unflagged: taken
            // along the patch's own normal, and turned over with the face
            // when the face is flagged.
            FaceSurface::Plane { normal, d } => {
                let flagged = topo.face(*fid)?.is_reversed();
                let sign = if opposed == flagged { 1.0 } else { -1.0 };
                topo.face_mut(*fid)?.set_surface(FaceSurface::Plane {
                    normal: normal * sign,
                    d: d * sign,
                });
                if flagged {
                    turn_over(topo, *fid)?;
                }
            }
            other => {
                topo.face_mut(*fid)?.set_surface(other);
                if opposed {
                    turn_over(topo, *fid)?;
                }
            }
        }
        converted += 1;
    }

    Ok(converted)
}

/// Whether a NURBS patch's normal (`Su x Sv`) and the recognized surface's
/// own normal point opposite ways, read at the patch's middle. A patch
/// parameterized inward (a mirrored one, whose transform flips its flag and
/// keeps its wire) faces the other way to the analytic surface.
fn normals_oppose(
    nurbs: &brepkit_math::nurbs::surface::NurbsSurface,
    replacement: &FaceSurface,
) -> bool {
    let ((u0, u1), (v0, v1)) = (nurbs.domain_u(), nurbs.domain_v());
    let (u, v) = (0.5 * (u0 + u1), 0.5 * (v0 + v1));
    let Ok(n) = nurbs.normal(u, v) else {
        return false;
    };
    let p = brepkit_math::traits::ParametricSurface::evaluate(nurbs, u, v);
    let m = match replacement {
        FaceSurface::Plane { normal, .. } => *normal,
        other => match other.project_point(p) {
            Some((pu, pv)) => other.normal(pu, pv),
            None => return false,
        },
    };
    n.dot(m) < 0.0
}

/// Turn a face over against its new surface: its flag flips, and each of its
/// wires runs the other way, so every edge keeps the sense it had in the
/// shell.
fn turn_over(topo: &mut Topology, face_id: FaceId) -> Result<(), HealError> {
    use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

    let (outer, inner, reversed) = {
        let face = topo.face(face_id)?;
        (
            face.outer_wire(),
            face.inner_wires().to_vec(),
            face.is_reversed(),
        )
    };
    let mut flip = |w: WireId| -> Result<WireId, HealError> {
        let wire = topo.wire(w)?;
        let closed = wire.is_closed();
        let edges: Vec<OrientedEdge> = wire
            .edges()
            .iter()
            .rev()
            .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
            .collect();
        Ok(topo.add_wire(Wire::new(edges, closed)?))
    };
    let new_outer = flip(outer)?;
    let new_inner = inner
        .into_iter()
        .map(&mut flip)
        .collect::<Result<Vec<_>, _>>()?;
    let face = topo.face_mut(face_id)?;
    face.set_outer_wire(new_outer);
    *face.inner_wires_mut() = new_inner;
    face.set_reversed(!reversed);
    Ok(())
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
    let face_ids: Vec<FaceId> = solid_faces(topo, solid_id)?;

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

    #[test]
    fn convert_walks_inner_shells() {
        // A solid with both an outer shell and an inner (cavity) shell
        // should have faces recognized on BOTH shells. Regression for
        // the prior outer-shell-only behavior, which silently left
        // cavity faces unconverted in hollow solids.
        use crate::construct::convert_surface::sphere_to_nurbs;
        use brepkit_math::surfaces::SphericalSurface;

        let mut topo = Topology::new();

        // Build two scaffolds — an outer "face" (planar) and an inner
        // "face" carrying a NURBS sphere surface that should be
        // recognized back as Sphere.
        let outer_face = {
            let v = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), 1e-7));
            let edge_id = topo.add_edge(Edge::new(v, v, EdgeCurve::Line));
            let wire = Wire::new(vec![OrientedEdge::new(edge_id, true)], true).unwrap();
            let wid = topo.add_wire(wire);
            topo.add_face(Face::new(
                wid,
                vec![],
                FaceSurface::Plane {
                    normal: Vec3::new(0.0, 0.0, 1.0),
                    d: 0.0,
                },
            ))
        };

        let sphere = SphericalSurface::new(Point3::new(5.0, 0.0, 0.0), 1.0).unwrap();
        let nurbs_sphere = sphere_to_nurbs(&sphere).unwrap();
        let inner_face = {
            let v = topo.add_vertex(Vertex::new(Point3::new(6.0, 0.0, 0.0), 1e-7));
            let edge_id = topo.add_edge(Edge::new(v, v, EdgeCurve::Line));
            let wire = Wire::new(vec![OrientedEdge::new(edge_id, true)], true).unwrap();
            let wid = topo.add_wire(wire);
            topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(nurbs_sphere)))
        };

        let outer_shell = topo.add_shell(Shell::new(vec![outer_face]).unwrap());
        let inner_shell = topo.add_shell(Shell::new(vec![inner_face]).unwrap());
        let solid_id = topo.add_solid(Solid::new(outer_shell, vec![inner_shell]));

        let tol = Tolerance::new();
        let converted = convert_to_elementary(&mut topo, solid_id, &tol).unwrap();
        assert_eq!(
            converted, 1,
            "should have converted the cavity-shell sphere face, got {converted}"
        );

        // The outer face was already analytic; the inner-shell face
        // should now be Sphere, not NURBS.
        assert!(matches!(
            topo.face(outer_face).unwrap().surface(),
            FaceSurface::Plane { .. }
        ));
        match topo.face(inner_face).unwrap().surface() {
            FaceSurface::Sphere(s) => {
                assert!(
                    (s.radius() - 1.0).abs() < 1e-6,
                    "recovered sphere radius {} should be ~1.0",
                    s.radius()
                );
            }
            other => panic!("expected inner-shell face to be Sphere, got {other:?}"),
        }
    }
}

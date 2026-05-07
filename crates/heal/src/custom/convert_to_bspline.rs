//! Convert analytic geometry to B-spline representation.
//!
//! Equivalent to OCCT's `BRepBuilderAPI_NurbsConvert` / `ShapeCustom::ConvertToBSpline`.
//! Replaces every analytic surface (Plane, Cylinder, Cone, Sphere, Torus) with a
//! NURBS surface and every analytic curve (Line, Circle, Ellipse) with a NURBS
//! curve.
//!
//! Surfaces use the rational NURBS representations exposed by
//! [`brepkit_geometry::convert`]. Curves use the rational quadratic arc form for
//! Circle/Ellipse and a degree-1 form for Line.
//!
//! # Limitation: pcurves are dropped
//!
//! Stored pcurves on the (edge, face) registry are removed for every face whose
//! surface is converted. The (u, v) coordinates of pcurves on an analytic
//! surface do not map linearly to the equivalent NURBS surface (e.g. cylindrical
//! `u` is angular, but the NURBS u is rational), so the stored pcurves would
//! silently misalign without re-projection. Callers that need pcurves should
//! recompute them after this op.

use std::f64::consts::TAU;

use brepkit_geometry::convert::curve_to_nurbs::{circle_to_nurbs, ellipse_to_nurbs, line_to_nurbs};
use brepkit_geometry::convert::surface_to_nurbs::{
    cone_to_nurbs, cylinder_to_nurbs, sphere_to_nurbs, torus_to_nurbs,
};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::explorer::{face_edges, solid_edges, solid_faces};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;

use crate::HealError;

/// Convert all analytic geometry in a solid to B-Spline representation.
///
/// Returns the total number of faces and edges that were converted (NURBS
/// faces/edges are skipped and not counted).
///
/// # Errors
///
/// Returns [`HealError`] if any topology lookup, NURBS construction, or face
/// surface replacement fails.
pub fn convert_solid_to_bspline(
    topo: &mut Topology,
    solid_id: SolidId,
) -> Result<usize, HealError> {
    let face_ids = solid_faces(topo, solid_id)?;
    let edge_ids = solid_edges(topo, solid_id)?;

    let mut converted = 0;

    for fid in face_ids {
        if convert_face_surface(topo, fid)? {
            converted += 1;
        }
    }

    for eid in edge_ids {
        if convert_edge_curve(topo, eid)? {
            converted += 1;
        }
    }

    Ok(converted)
}

fn convert_face_surface(topo: &mut Topology, fid: FaceId) -> Result<bool, HealError> {
    let surface = topo.face(fid)?.surface().clone();
    let nurbs = match surface {
        FaceSurface::Plane { normal, d } => plane_face_to_nurbs(topo, fid, normal, d)?,
        FaceSurface::Cylinder(c) => {
            let v_range = axial_v_range(topo, fid, c.origin(), c.axis())?;
            cylinder_to_nurbs(&c, v_range)?
        }
        FaceSurface::Cone(c) => {
            let mut v_range = axial_v_range(topo, fid, c.apex(), c.axis())?;
            // Cone has a parametric singularity at v=0 (the apex). Pull v_min
            // strictly positive to keep the rational NURBS construction stable.
            if v_range.0 < 1e-9 {
                v_range.0 = 1e-9;
            }
            if v_range.1 <= v_range.0 {
                v_range.1 = v_range.0 + 1.0;
            }
            cone_to_nurbs(&c, v_range)?
        }
        FaceSurface::Sphere(s) => sphere_to_nurbs(&s)?,
        FaceSurface::Torus(t) => torus_to_nurbs(&t)?,
        FaceSurface::Nurbs(_) => return Ok(false),
    };

    drop_face_pcurves(topo, fid)?;
    topo.face_mut(fid)?.set_surface(FaceSurface::Nurbs(nurbs));
    Ok(true)
}

fn convert_edge_curve(topo: &mut Topology, eid: EdgeId) -> Result<bool, HealError> {
    let edge = topo.edge(eid)?;
    let curve = edge.curve().clone();
    let start_v = edge.start();
    let end_v = edge.end();
    let start_pt = topo.vertex(start_v)?.point();
    let end_pt = topo.vertex(end_v)?.point();

    let nurbs = match curve {
        EdgeCurve::Line => {
            // Skip degenerate edges (start == end position) — line_to_nurbs
            // rejects them and we have no meaningful NURBS to substitute.
            if (end_pt - start_pt).length() < 1e-15 {
                return Ok(false);
            }
            line_to_nurbs(start_pt, end_pt)?
        }
        EdgeCurve::Circle(c) => {
            if start_v == end_v {
                circle_to_nurbs(&c, 0.0, TAU)?
            } else {
                let (t_start, t_end) = arc_param_range(c.project(start_pt), c.project(end_pt));
                circle_to_nurbs(&c, t_start, t_end)?
            }
        }
        EdgeCurve::Ellipse(e) => {
            if start_v == end_v {
                ellipse_to_nurbs(&e, 0.0, TAU)?
            } else {
                let (t_start, t_end) = arc_param_range(e.project(start_pt), e.project(end_pt));
                ellipse_to_nurbs(&e, t_start, t_end)?
            }
        }
        EdgeCurve::NurbsCurve(_) => return Ok(false),
    };

    topo.edge_mut(eid)?.set_curve(EdgeCurve::NurbsCurve(nurbs));
    Ok(true)
}

/// Pick the canonical CCW arc range from two unwrapped angular params.
///
/// `Circle3D::project` and `Ellipse3D::project` return values in `[0, 2π)`. An
/// arc from start to end that wraps past the seam ends up with `t_end < t_start`;
/// shift `t_end` up by 2π so the resulting span is positive.
fn arc_param_range(t_start: f64, t_end: f64) -> (f64, f64) {
    if t_end > t_start {
        (t_start, t_end)
    } else {
        (t_start, t_end + TAU)
    }
}

fn drop_face_pcurves(topo: &mut Topology, fid: FaceId) -> Result<(), HealError> {
    let edges = face_edges(topo, fid)?;
    for eid in edges {
        topo.pcurves_mut().remove(eid, fid);
    }
    Ok(())
}

/// Bounds of a face's wire vertices projected onto an axis through `origin`.
fn axial_v_range(
    topo: &Topology,
    face_id: FaceId,
    origin: Point3,
    axis: Vec3,
) -> Result<(f64, f64), HealError> {
    let face = topo.face(face_id)?;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                let pt = topo.vertex(vid)?.point();
                let to_pt = pt - origin;
                let v = axis.dot(to_pt);
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }

    if v_min < v_max {
        Ok((v_min, v_max))
    } else {
        Ok((-1.0, 1.0))
    }
}

/// Build a NURBS plane surface that comfortably contains every wire vertex
/// of `face_id`.
fn plane_face_to_nurbs(
    topo: &Topology,
    face_id: FaceId,
    normal: Vec3,
    d: f64,
) -> Result<NurbsSurface, HealError> {
    let (u_axis, v_axis) = plane_frame_axes(normal);
    let plane_origin = Point3::new(0.0, 0.0, 0.0) + normal * d;

    let face = topo.face(face_id)?;
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        let wire = topo.wire(wire_id)?;
        for oe in wire.edges() {
            let edge = topo.edge(oe.edge())?;
            for vid in [edge.start(), edge.end()] {
                let pt = topo.vertex(vid)?.point();
                let rel = pt - plane_origin;
                u_min = u_min.min(u_axis.dot(rel));
                u_max = u_max.max(u_axis.dot(rel));
                v_min = v_min.min(v_axis.dot(rel));
                v_max = v_max.max(v_axis.dot(rel));
            }
        }
    }

    if u_min >= u_max {
        u_min = -1.0;
        u_max = 1.0;
    }
    if v_min >= v_max {
        v_min = -1.0;
        v_max = 1.0;
    }
    let margin_u = 0.1 * (u_max - u_min);
    let margin_v = 0.1 * (v_max - v_min);

    let u_range = (u_min - margin_u, u_max + margin_u);
    let v_range = (v_min - margin_v, v_max + margin_v);

    let cp = vec![
        vec![
            plane_origin + u_axis * u_range.0 + v_axis * v_range.0,
            plane_origin + u_axis * u_range.0 + v_axis * v_range.1,
        ],
        vec![
            plane_origin + u_axis * u_range.1 + v_axis * v_range.0,
            plane_origin + u_axis * u_range.1 + v_axis * v_range.1,
        ],
    ];
    let weights = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
    let knots_u = vec![u_range.0, u_range.0, u_range.1, u_range.1];
    let knots_v = vec![v_range.0, v_range.0, v_range.1, v_range.1];

    Ok(NurbsSurface::new(1, 1, knots_u, knots_v, cp, weights)?)
}

fn plane_frame_axes(normal: Vec3) -> (Vec3, Vec3) {
    let seed = if normal.x().abs() < 0.9 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    let u_raw = normal.cross(seed);
    let u_axis = u_raw.normalize().unwrap_or(Vec3::new(1.0, 0.0, 0.0));
    let v_axis = normal.cross(u_axis);
    (u_axis, v_axis)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
mod tests {
    use std::f64::consts::PI;

    use brepkit_math::curves::Circle3D;
    use brepkit_math::nurbs::curve::NurbsCurve;
    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };
    use brepkit_math::traits::ParametricCurve;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::shell::Shell;
    use brepkit_topology::solid::Solid;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    fn x_axis() -> Vec3 {
        Vec3::new(1.0, 0.0, 0.0)
    }
    fn z_axis() -> Vec3 {
        Vec3::new(0.0, 0.0, 1.0)
    }

    /// Build a single-face solid with a degenerate-edge wire so we can convert
    /// arbitrary surfaces in isolation. This keeps the per-surface tests
    /// independent of `make_cylinder`/`make_sphere` topology details.
    fn single_face_solid(topo: &mut Topology, surface: FaceSurface, ring: &[Point3]) -> SolidId {
        assert!(ring.len() >= 3, "need at least 3 points for a ring");
        let n = ring.len();
        let vids: Vec<_> = ring
            .iter()
            .map(|&p| topo.add_vertex(Vertex::new(p, 1e-7)))
            .collect();
        let mut edges = Vec::new();
        for i in 0..n {
            let eid = topo.add_edge(Edge::new(vids[i], vids[(i + 1) % n], EdgeCurve::Line));
            edges.push(OrientedEdge::new(eid, true));
        }
        let wire = topo.add_wire(Wire::new(edges, true).unwrap());
        let fid = topo.add_face(Face::new(wire, vec![], surface));
        let shell = topo.add_shell(Shell::new(vec![fid]).unwrap());
        topo.add_solid(Solid::new(shell, vec![]))
    }

    #[test]
    fn box_solid_all_faces_become_nurbs() {
        let mut topo = Topology::default();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);

        let n = convert_solid_to_bspline(&mut topo, solid).unwrap();
        assert!(n > 0);

        for fid in solid_faces(&topo, solid).unwrap() {
            assert!(
                matches!(topo.face(fid).unwrap().surface(), FaceSurface::Nurbs(_)),
                "face {fid:?} should be NURBS after convert_to_bspline"
            );
        }
        for eid in solid_edges(&topo, solid).unwrap() {
            assert!(
                matches!(topo.edge(eid).unwrap().curve(), EdgeCurve::NurbsCurve(_)),
                "edge {eid:?} should be NURBS after convert_to_bspline"
            );
        }
    }

    #[test]
    fn idempotent_on_already_nurbs() {
        let mut topo = Topology::default();
        let solid = brepkit_topology::test_utils::make_unit_cube_manifold(&mut topo);

        let first = convert_solid_to_bspline(&mut topo, solid).unwrap();
        assert!(first > 0);
        let second = convert_solid_to_bspline(&mut topo, solid).unwrap();
        assert_eq!(second, 0);
    }

    #[test]
    fn cylinder_face_converts_with_axial_range() {
        let cyl = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z_axis(), 2.0).unwrap();
        let mut topo = Topology::default();
        let ring = [
            Point3::new(2.0, 0.0, 0.0),
            Point3::new(0.0, 2.0, 0.0),
            Point3::new(2.0, 0.0, 5.0),
            Point3::new(0.0, 2.0, 5.0),
        ];
        let solid = single_face_solid(&mut topo, FaceSurface::Cylinder(cyl), &ring);

        convert_solid_to_bspline(&mut topo, solid).unwrap();

        let fid = solid_faces(&topo, solid).unwrap()[0];
        let surf = topo.face(fid).unwrap().surface().clone();
        let nurbs = match surf {
            FaceSurface::Nurbs(n) => n,
            other => panic!("expected NURBS, got {:?}", other.type_tag()),
        };

        // Sample the NURBS and verify points lie at distance 2 from the z-axis
        // and within the v-range derived from the wire (0..5).
        let (u_min, u_max) = nurbs.domain_u();
        let (v_min, v_max) = nurbs.domain_v();
        for i in 0..=8 {
            for j in 0..=4 {
                let u = u_min + (u_max - u_min) * f64::from(i) / 8.0;
                let v = v_min + (v_max - v_min) * f64::from(j) / 4.0;
                let p = nurbs.evaluate(u, v);
                let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
                assert!((r - 2.0).abs() < 1e-6, "u={u}, v={v}: r={r}");
                assert!(
                    p.z() >= -1e-9 && p.z() <= 5.0 + 1e-9,
                    "z out of range: {}",
                    p.z()
                );
            }
        }
    }

    #[test]
    fn sphere_face_converts() {
        let sphere = SphericalSurface::new(Point3::new(0.0, 0.0, 0.0), 3.0).unwrap();
        let mut topo = Topology::default();
        let ring = [
            Point3::new(3.0, 0.0, 0.0),
            Point3::new(0.0, 3.0, 0.0),
            Point3::new(-3.0, 0.0, 0.0),
        ];
        let solid = single_face_solid(&mut topo, FaceSurface::Sphere(sphere), &ring);

        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let fid = solid_faces(&topo, solid).unwrap()[0];
        assert!(matches!(
            topo.face(fid).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ));
    }

    #[test]
    fn cone_face_converts_with_clamped_apex() {
        let cone = ConicalSurface::new(
            Point3::new(0.0, 0.0, 0.0),
            z_axis(),
            std::f64::consts::FRAC_PI_4,
        )
        .unwrap();
        let mut topo = Topology::default();
        let ring = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(2.0, 0.0, 2.0),
            Point3::new(0.0, 2.0, 2.0),
        ];
        let solid = single_face_solid(&mut topo, FaceSurface::Cone(cone), &ring);

        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let fid = solid_faces(&topo, solid).unwrap()[0];
        assert!(matches!(
            topo.face(fid).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ));
    }

    #[test]
    fn torus_face_converts() {
        let torus = ToroidalSurface::new(Point3::new(0.0, 0.0, 0.0), 4.0, 1.0).unwrap();
        let mut topo = Topology::default();
        let ring = [
            Point3::new(5.0, 0.0, 0.0),
            Point3::new(0.0, 5.0, 0.0),
            Point3::new(-5.0, 0.0, 0.0),
        ];
        let solid = single_face_solid(&mut topo, FaceSurface::Torus(torus), &ring);

        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let fid = solid_faces(&topo, solid).unwrap()[0];
        assert!(matches!(
            topo.face(fid).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ));
    }

    #[test]
    fn closed_circle_edge_becomes_full_nurbs() {
        let mut topo = Topology::default();
        let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z_axis(), 1.0).unwrap();
        let v = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), 1e-7));
        let eid = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(circle)));

        // Plug the closed edge into a one-edge wire on a planar face so the
        // solid traversal sees it.
        let wire = topo.add_wire(Wire::new(vec![OrientedEdge::new(eid, true)], true).unwrap());
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: z_axis(),
                d: 0.0,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        let solid = topo.add_solid(Solid::new(shell, vec![]));

        convert_solid_to_bspline(&mut topo, solid).unwrap();

        let nurbs = match topo.edge(eid).unwrap().curve().clone() {
            EdgeCurve::NurbsCurve(n) => n,
            other => panic!("expected NurbsCurve, got {}", other.type_tag()),
        };
        // Sample the closed NURBS and ensure points lie on the circle.
        for i in 0..16 {
            let t = ParametricCurve::domain(&nurbs).0
                + (ParametricCurve::domain(&nurbs).1 - ParametricCurve::domain(&nurbs).0)
                    * f64::from(i)
                    / 16.0;
            let p = nurbs.evaluate(t);
            let r = (p.x() * p.x() + p.y() * p.y()).sqrt();
            assert!(
                (r - 1.0).abs() < 1e-6,
                "circle radius drift at t={t}: r={r}"
            );
            assert!(
                p.z().abs() < 1e-9,
                "circle out-of-plane at t={t}: z={}",
                p.z()
            );
        }
    }

    #[test]
    fn arc_param_range_handles_wrap() {
        // No wrap: t_end > t_start.
        assert_eq!(arc_param_range(0.0, PI), (0.0, PI));
        // Wrap: t_end < t_start, shift by 2π.
        let (a, b) = arc_param_range(1.5 * PI, 0.5 * PI);
        assert!((a - 1.5 * PI).abs() < 1e-12);
        assert!((b - 2.5 * PI).abs() < 1e-12);
    }

    #[test]
    fn line_to_nurbs_preserves_endpoints() {
        let mut topo = Topology::default();
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(3.0, 4.0, 0.0);
        let v0 = topo.add_vertex(Vertex::new(p0, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(p1, 1e-7));
        let eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        // Embed in a (degenerate, unbounded) face so solid_edges finds it.
        let wire = topo.add_wire(
            Wire::new(
                vec![OrientedEdge::new(eid, true), OrientedEdge::new(eid, false)],
                true,
            )
            .unwrap(),
        );
        let face = topo.add_face(Face::new(
            wire,
            vec![],
            FaceSurface::Plane {
                normal: z_axis(),
                d: 0.0,
            },
        ));
        let shell = topo.add_shell(Shell::new(vec![face]).unwrap());
        let solid = topo.add_solid(Solid::new(shell, vec![]));

        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let curve = topo.edge(eid).unwrap().curve().clone();
        let nurbs: NurbsCurve = match curve {
            EdgeCurve::NurbsCurve(n) => n,
            other => panic!("expected NurbsCurve, got {}", other.type_tag()),
        };
        let (t0, t1) = ParametricCurve::domain(&nurbs);
        let q0 = nurbs.evaluate(t0);
        let q1 = nurbs.evaluate(t1);
        assert!((q0 - p0).length() < 1e-12);
        assert!((q1 - p1).length() < 1e-12);
    }

    #[test]
    fn x_axis_plane_picks_safe_uv_frame() {
        // Normal along +x triggers the alternate seed in plane_frame_axes.
        let mut topo = Topology::default();
        let normal = x_axis();
        let ring = [
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(0.0, 1.0, 0.0),
            Point3::new(0.0, 0.0, 1.0),
        ];
        let solid = single_face_solid(&mut topo, FaceSurface::Plane { normal, d: 0.0 }, &ring);
        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let fid = solid_faces(&topo, solid).unwrap()[0];
        assert!(matches!(
            topo.face(fid).unwrap().surface(),
            FaceSurface::Nurbs(_)
        ));
    }
}

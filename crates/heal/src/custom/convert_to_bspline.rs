//! Convert analytic geometry to B-spline representation.
//!
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

use crate::construct::convert_surface::{
    cone_to_nurbs, cylinder_to_nurbs, sphere_to_nurbs, torus_to_nurbs,
};
use brepkit_geometry::convert::curve_to_nurbs::{circle_to_nurbs, ellipse_to_nurbs, line_to_nurbs};
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
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
            let v_range = surface_v_range(topo, fid, |p| c.project_point(p).1)?;
            cylinder_to_nurbs(&c, v_range)?
        }
        FaceSurface::Cone(c) => {
            let (mut v_lo, v_hi) = surface_v_range(topo, fid, |p| c.project_point(p).1)?;
            // The rational form needs a nonzero radius row, so a face that
            // reaches the apex keeps a vanishing ring there.
            v_lo = v_lo.max((1e-9 * v_hi.abs().max(1.0)).min(0.1 * Tolerance::new().linear));
            cone_to_nurbs(&c, (v_lo, v_hi.max(v_lo * 2.0)))?
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

    let tol = Tolerance::new();
    let nurbs = match curve {
        EdgeCurve::Line => {
            // Skip near-degenerate edges. Use the topology linear tolerance so
            // we don't propagate a `line_to_nurbs` rejection (which would abort
            // the whole solid conversion) for edges that are noise-only-long.
            if (end_pt - start_pt).length() < tol.linear {
                return Ok(false);
            }
            line_to_nurbs(start_pt, end_pt)?
        }
        // A closed edge's curve starts and ends at its one vertex, so its full
        // turn begins at that vertex's angle, not at the frame's.
        EdgeCurve::Circle(c) => {
            if start_v == end_v {
                let t0 = c.project(start_pt);
                circle_to_nurbs(&c, t0, t0 + TAU)?
            } else {
                let Some((t_start, t_end)) =
                    arc_param_range(c.project(start_pt), c.project(end_pt), tol.angular)
                else {
                    return Ok(false);
                };
                circle_to_nurbs(&c, t_start, t_end)?
            }
        }
        EdgeCurve::Ellipse(e) => {
            if start_v == end_v {
                let t0 = e.project(start_pt);
                ellipse_to_nurbs(&e, t0, t0 + TAU)?
            } else {
                let Some((t_start, t_end)) =
                    arc_param_range(e.project(start_pt), e.project(end_pt), tol.angular)
                else {
                    return Ok(false);
                };
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
///
/// Returns `None` when the start and end project to the same angle (within
/// `tol_ang`). Two distinct vertices that share an angle imply a zero-span or
/// full-loop arc that the closed-edge branch above should have caught — handing
/// it back as `Some((t, t + 2π))` would silently turn a zero-length topological
/// edge into a complete circle.
fn arc_param_range(t_start: f64, t_end: f64, tol_ang: f64) -> Option<(f64, f64)> {
    let delta = t_end - t_start;
    if delta.abs() < tol_ang {
        None
    } else if delta > 0.0 {
        Some((t_start, t_end))
    } else {
        Some((t_start, t_end + TAU))
    }
}

fn drop_face_pcurves(topo: &mut Topology, fid: FaceId) -> Result<(), HealError> {
    let edges = face_edges(topo, fid)?;
    for eid in edges {
        topo.pcurves_mut().remove(eid, fid);
    }
    Ok(())
}

/// Points along a face's whole boundary, each edge sampled end to end: a
/// curved edge reaches past its vertices (a disc's one vertex spans nothing).
fn boundary_points(topo: &Topology, face_id: FaceId) -> Result<Vec<Point3>, HealError> {
    const SAMPLES: u32 = 32;
    let face = topo.face(face_id)?;
    let mut points = Vec::new();
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            for k in 0..=SAMPLES {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(SAMPLES);
                points.push(edge.curve().evaluate_with_endpoints(t, start, end));
            }
        }
    }
    Ok(points)
}

/// The span of a surface parameter over a face's boundary. The patch ends
/// exactly there, so an extreme between two samples (the crest of an
/// elliptical rim) is refined to its true value.
fn surface_v_range(
    topo: &Topology,
    face_id: FaceId,
    v_of: impl Fn(Point3) -> f64,
) -> Result<(f64, f64), HealError> {
    const SAMPLES: u32 = 32;
    let face = topo.face(face_id)?;
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let param = |k: usize| {
                #[allow(clippy::cast_precision_loss)]
                let f = k as f64 / f64::from(SAMPLES);
                t0 + (t1 - t0) * f
            };
            let v_at = |t: f64| v_of(edge.curve().evaluate_with_endpoints(t, start, end));
            let vs: Vec<f64> = (0..=SAMPLES as usize).map(|k| v_at(param(k))).collect();
            for (k, &v) in vs.iter().enumerate() {
                lo = lo.min(v);
                hi = hi.max(v);
                if k == 0 || k + 1 == vs.len() {
                    continue;
                }
                let (before, after) = (vs[k - 1], vs[k + 1]);
                let flat = 1e-12 * (v.abs() + 1.0);
                if v >= before && v >= after && v - before.min(after) > flat {
                    hi = hi.max(golden_max(v_at, param(k - 1), param(k + 1)));
                }
                if v <= before && v <= after && before.max(after) - v > flat {
                    lo = lo.min(-golden_max(|t| -v_at(t), param(k - 1), param(k + 1)));
                }
            }
        }
    }
    if lo < hi {
        Ok((lo, hi))
    } else {
        Ok((-1.0, 1.0))
    }
}

/// The maximum of `f` on `[a, b]`, where it has a single peak.
fn golden_max(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let r = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..64 {
        if fc > fd {
            b = d;
            d = c;
            fd = fc;
            c = b - r * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + r * (b - a);
            fd = f(d);
        }
    }
    fc.max(fd)
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

    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;

    for pt in boundary_points(topo, face_id)? {
        let rel = pt - plane_origin;
        u_min = u_min.min(u_axis.dot(rel));
        u_max = u_max.max(u_axis.dot(rel));
        v_min = v_min.min(v_axis.dot(rel));
        v_max = v_max.max(v_axis.dot(rel));
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

    /// A slanted rim's crest falls between boundary samples; the cylinder's
    /// patch still reaches it.
    #[test]
    fn slanted_rim_crest_stays_on_the_patch() {
        use brepkit_math::curves::Ellipse3D;
        use brepkit_math::nurbs::projection::project_point_to_surface;

        let cylinder = CylindricalSurface::new(Point3::new(0.0, 0.0, 0.0), z_axis(), 1.0).unwrap();
        let rim = Ellipse3D::new_with_ref(
            Point3::new(0.0, 0.0, 0.0),
            Vec3::new(-0.5, 0.0, 1.0),
            1.25_f64.sqrt(),
            1.0,
            Vec3::new(1.0, 0.0, 0.5),
        )
        .unwrap();
        let mut topo = Topology::default();
        let a = topo.add_vertex(Vertex::new(rim.evaluate(0.05), 1e-7));
        let b = topo.add_vertex(Vertex::new(rim.evaluate(0.05 + PI), 1e-7));
        let halves = [(a, b), (b, a)].map(|(from, to)| {
            let eid = topo.add_edge(Edge::new(from, to, EdgeCurve::Ellipse(rim.clone())));
            OrientedEdge::new(eid, true)
        });
        let wire = topo.add_wire(Wire::new(halves.to_vec(), true).unwrap());
        let fid = topo.add_face(Face::new(wire, vec![], FaceSurface::Cylinder(cylinder)));
        let shell = topo.add_shell(Shell::new(vec![fid]).unwrap());
        let solid = topo.add_solid(Solid::new(shell, vec![]));

        convert_solid_to_bspline(&mut topo, solid).unwrap();
        let FaceSurface::Nurbs(patch) = topo.face(fid).unwrap().surface() else {
            panic!("the cylinder face should be NURBS");
        };
        for crest in [Point3::new(1.0, 0.0, 0.5), Point3::new(-1.0, 0.0, -0.5)] {
            let on = project_point_to_surface(patch, crest, 1e-12).unwrap();
            assert!(on.distance < 1e-9, "{crest:?} is {} off", on.distance);
        }
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

    /// A closed conic edge's curve starts and ends at its vertex after the
    /// conversion, wherever that vertex sits on the conic.
    #[test]
    fn closed_conics_start_at_their_vertex() {
        let circle = Circle3D::new(Point3::new(0.0, 0.0, 0.0), z_axis(), 1.0).unwrap();
        let ellipse =
            brepkit_math::curves::Ellipse3D::new(Point3::new(0.0, 0.0, 0.0), z_axis(), 2.0, 1.0)
                .unwrap();
        for (curve, at) in [
            (EdgeCurve::Circle(circle.clone()), circle.evaluate(2.2)),
            (EdgeCurve::Ellipse(ellipse.clone()), ellipse.evaluate(4.0)),
        ] {
            let mut topo = Topology::default();
            let v = topo.add_vertex(Vertex::new(at, 1e-7));
            let eid = topo.add_edge(Edge::new(v, v, curve));
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
            let EdgeCurve::NurbsCurve(nurbs) = topo.edge(eid).unwrap().curve().clone() else {
                panic!("expected a NURBS curve");
            };
            let (t0, t1) = ParametricCurve::domain(&nurbs);
            for end in [nurbs.evaluate(t0), nurbs.evaluate(t1)] {
                assert!(
                    (end - at).length() < 1e-12,
                    "curve end {end:?}, vertex {at:?}"
                );
            }
        }
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
        let tol = 1e-12;
        // No wrap: t_end > t_start.
        assert_eq!(arc_param_range(0.0, PI, tol), Some((0.0, PI)));
        // Wrap: t_end < t_start, shift by 2π.
        let (a, b) = arc_param_range(1.5 * PI, 0.5 * PI, tol).unwrap();
        assert!((a - 1.5 * PI).abs() < 1e-12);
        assert!((b - 2.5 * PI).abs() < 1e-12);
    }

    #[test]
    fn arc_param_range_rejects_zero_span() {
        // Distinct vertices that project to the same angle ⇒ zero-span; must
        // not silently inflate to a full circle.
        assert_eq!(arc_param_range(1.0, 1.0, 1e-12), None);
        // Within angular tolerance ⇒ also rejected.
        assert_eq!(arc_param_range(1.0, 1.0 + 1e-15, 1e-12), None);
        // Just outside tolerance ⇒ accepted.
        assert!(arc_param_range(1.0, 1.0 + 1e-9, 1e-12).is_some());
    }

    #[test]
    fn near_degenerate_line_edge_is_skipped_not_errored() {
        // Edge with length below topology tolerance must skip cleanly, not
        // bubble a GeomError that aborts the whole solid conversion.
        let mut topo = Topology::default();
        let p0 = Point3::new(0.0, 0.0, 0.0);
        let p1 = Point3::new(1e-10, 0.0, 0.0);
        let v0 = topo.add_vertex(Vertex::new(p0, 1e-7));
        let v1 = topo.add_vertex(Vertex::new(p1, 1e-7));
        let degenerate_eid = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));

        // Embed in a face so solid_edges traversal sees it.
        let wire = topo.add_wire(
            Wire::new(
                vec![
                    OrientedEdge::new(degenerate_eid, true),
                    OrientedEdge::new(degenerate_eid, false),
                ],
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

        // Should succeed without converting the degenerate edge.
        convert_solid_to_bspline(&mut topo, solid).unwrap();
        assert!(matches!(
            topo.edge(degenerate_eid).unwrap().curve(),
            EdgeCurve::Line
        ));
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

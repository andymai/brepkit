//! Thicken a face into a solid of even thickness.
//!
//! A planar face is extruded along its normal. A face on a cylinder, cone,
//! sphere or torus is grown along its surface's normal: its far side lies on
//! the offset surface, each edge is carried to it along the normals, and
//! each side between an edge and its image is the ruled surface the normals
//! sweep.

use std::collections::HashMap;

use brepkit_math::curves::Circle3D;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::extrude::extrude;

/// Thicken a face into a solid by growing it `thickness` along its normal
/// (against it when negative).
///
/// A planar face, or a NURBS face that is a plane, is extruded. A face on a
/// cylinder or sphere grows to the offset surface by a radial scaling that
/// carries every edge exactly. A face on a cone or torus does too when each
/// of its edges is a line or a circle the offset carries to a line or a
/// circle.
///
/// # Errors
///
/// Returns an error if `thickness` is zero, the offset surface collapses,
/// the face lies on a curved NURBS surface, or an edge of a cone or torus
/// face is not carried to a line or circle.
pub fn thicken(
    topo: &mut Topology,
    face: FaceId,
    thickness: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(thickness, 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "thickness must be non-zero".into(),
        });
    }
    let face_data = topo.face(face)?;
    let reversed = face_data.is_reversed();
    let plane_normal = match face_data.surface() {
        FaceSurface::Plane { normal, .. } => Some(*normal),
        FaceSurface::Nurbs(nurbs) => {
            match brepkit_geometry::convert::recognize_surface(nurbs, tol.linear) {
                brepkit_geometry::convert::RecognizedSurface::Plane { .. } => {
                    Some(nurbs.normal(0.5, 0.5).map_err(|e| {
                        crate::OperationsError::InvalidInput {
                            reason: format!("NURBS normal computation failed: {e}"),
                        }
                    })?)
                }
                _ => {
                    return Err(crate::OperationsError::InvalidInput {
                        reason: "thicken does not take a face on a curved NURBS surface".into(),
                    });
                }
            }
        }
        FaceSurface::Cylinder(_)
        | FaceSurface::Cone(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => None,
    };
    if let Some(normal) = plane_normal {
        let normal = if reversed { -normal } else { normal };
        let (direction, distance) = if thickness > 0.0 {
            (normal, thickness)
        } else {
            (-normal, -thickness)
        };
        return extrude(topo, face, direction, distance);
    }
    thicken_curved(topo, face, thickness)
}

/// A boundary edge of the face being thickened, as its wire runs it.
struct Run {
    edge: EdgeId,
    forward: bool,
}

/// [`thicken`] for a face on a cylinder, cone, sphere or torus.
#[allow(clippy::too_many_lines)]
fn thicken_curved(
    topo: &mut Topology,
    face: FaceId,
    thickness: f64,
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let base = crate::copy::copy_face(topo, face)?;
    let base_data = topo.face(base)?;
    let surface = base_data.surface().clone();
    let reversed = base_data.is_reversed();
    // The face grows along `grow`, the surface's normal turned to its side,
    // by `depth`; `along` is `grow` read against the surface's own normal.
    let along = if reversed == (thickness > 0.0) {
        -1.0
    } else {
        1.0
    };
    let depth = thickness.abs();
    let grow = |p: Point3| -> Vec3 { surface_normal(&surface, p) * along };
    let offset = offset_surface(&surface, along * depth)?;
    keep_off_the_axis(topo, base, &surface, along * depth)?;
    let carry = |p: Point3| p + grow(p) * depth;
    let affine = affine_carry(&surface, along * depth);

    // The wires as the face runs them about `grow`: stored wires run about
    // the surface's own normal, so turned when the face grows against it.
    let wire_ids: Vec<_> = std::iter::once(base_data.outer_wire())
        .chain(base_data.inner_wires().iter().copied())
        .collect();
    let mut loops: Vec<Vec<Run>> = Vec::with_capacity(wire_ids.len());
    for wid in &wire_ids {
        let mut runs: Vec<Run> = topo
            .wire(*wid)?
            .edges()
            .iter()
            .map(|oe| Run {
                edge: oe.edge(),
                forward: oe.is_forward(),
            })
            .collect();
        if along < 0.0 {
            runs.reverse();
            for r in &mut runs {
                r.forward = !r.forward;
            }
        }
        loops.push(runs);
    }
    let mut uses: HashMap<EdgeId, usize> = HashMap::new();
    for r in loops.iter().flatten() {
        *uses.entry(r.edge).or_default() += 1;
    }

    // Each vertex and edge carried to the far side, and each vertex's rail.
    let mut far_vertex: HashMap<VertexId, VertexId> = HashMap::new();
    let mut far_edge: HashMap<EdgeId, EdgeId> = HashMap::new();
    let mut rails: HashMap<VertexId, EdgeId> = HashMap::new();
    for r in loops.iter().flatten() {
        if far_edge.contains_key(&r.edge) {
            continue;
        }
        let e = topo.edge(r.edge)?;
        let (vs, ve, curve) = (e.start(), e.end(), e.curve().clone());
        for v in [vs, ve] {
            if let std::collections::hash_map::Entry::Vacant(slot) = far_vertex.entry(v) {
                let p = topo.vertex(v)?.point();
                slot.insert(topo.add_vertex(Vertex::new(carry(p), tol.linear)));
            }
        }
        let (a, b) = (topo.vertex(vs)?.point(), topo.vertex(ve)?.point());
        let far_curve =
            carry_curve(&curve, a, b, &carry, affine.as_deref(), tol).ok_or_else(|| {
                crate::OperationsError::InvalidInput {
                    reason: format!(
                        "thicken: a {} edge of a {} face is not carried to a line or circle",
                        curve.type_tag(),
                        surface.type_tag()
                    ),
                }
            })?;
        let far = topo.add_edge(Edge::new(far_vertex[&vs], far_vertex[&ve], far_curve));
        far_edge.insert(r.edge, far);
    }

    let mut faces = Vec::new();
    // The near side faces into the solid, the far side out of it.
    let near_loops: Vec<Vec<OrientedEdge>> = loops
        .iter()
        .map(|runs| {
            runs.iter()
                .rev()
                .map(|r| OrientedEdge::new(r.edge, !r.forward))
                .collect()
        })
        .collect();
    faces.push(add_face(topo, near_loops, surface.clone(), along < 0.0)?);
    let far_loops: Vec<Vec<OrientedEdge>> = loops
        .iter()
        .map(|runs| {
            runs.iter()
                .map(|r| OrientedEdge::new(far_edge[&r.edge], r.forward))
                .collect()
        })
        .collect();
    faces.push(add_face(topo, far_loops, offset, along > 0.0)?);

    // A side for each edge the face does not cross as a seam.
    for r in loops.iter().flatten() {
        if uses[&r.edge] > 1 {
            continue;
        }
        let e = topo.edge(r.edge)?;
        let (vs, ve, curve) = (e.start(), e.end(), e.curve().clone());
        let (a, b) = (topo.vertex(vs)?.point(), topo.vertex(ve)?.point());
        let far = far_edge[&r.edge];
        let far_curve = topo.edge(far)?.curve().clone();
        let (fa, fb) = (carry(a), carry(b));
        // Out of the face: the run's tangent across the growth.
        let (t0, t1) = curve.domain_with_endpoints(a, b);
        let mid_t = f64::midpoint(t0, t1);
        let mid = curve.evaluate_with_endpoints(mid_t, a, b);
        let tangent =
            curve.tangent_with_endpoints(mid_t, a, b) * if r.forward { 1.0 } else { -1.0 };
        let out = tangent.cross(grow(mid));
        let (from, to) = if r.forward { (vs, ve) } else { (ve, vs) };
        let side = side_surface(&curve, a, b, &far_curve, fa, fb, mid, out, tol)?;
        if let SideSurface::Annulus(normal) = side {
            // Two closed circles in one plane: the larger bounds it.
            let near = OrientedEdge::new(r.edge, r.forward);
            let far_run = OrientedEdge::new(far, !r.forward);
            let (outer, inner) = if circle_radius(&far_curve) > circle_radius(&curve) {
                (far_run, near)
            } else {
                (near, far_run)
            };
            let wire = |topo: &mut Topology, oe| -> Result<_, crate::OperationsError> {
                Ok(topo.add_wire(Wire::new(vec![oe], true)?))
            };
            let (ow, iw) = (wire(topo, outer)?, wire(topo, inner)?);
            faces.push(topo.add_face(Face::new(
                ow,
                vec![iw],
                FaceSurface::Plane {
                    normal,
                    d: normal.dot(a - Point3::new(0.0, 0.0, 0.0)),
                },
            )));
            continue;
        }
        let mut rail = |topo: &mut Topology, v: VertexId| -> EdgeId {
            *rails
                .entry(v)
                .or_insert_with(|| topo.add_edge(Edge::new(v, far_vertex[&v], EdgeCurve::Line)))
        };
        let (rail_to, rail_from) = (rail(topo, to), rail(topo, from));
        let run = vec![
            OrientedEdge::new(r.edge, r.forward),
            OrientedEdge::new(rail_to, true),
            OrientedEdge::new(far, !r.forward),
            OrientedEdge::new(rail_from, false),
        ];
        let SideSurface::Ruled(surface, outward) = side else {
            continue;
        };
        faces.push(add_face(topo, vec![run], surface, outward)?);
    }

    // A closed face, every edge of it a seam, has no sides, so its two
    // copies bound the wall apart: one outside, one round a cavity. Growing
    // along a sphere's or torus's own normal moves it outward.
    if let &[near, far] = faces.as_slice() {
        let (outer, cavity) = if along > 0.0 {
            (far, near)
        } else {
            (near, far)
        };
        let outer = topo.add_shell(Shell::new(vec![outer])?);
        let cavity = topo.add_shell(Shell::new(vec![cavity])?);
        return Ok(topo.add_solid(Solid::new(outer, vec![cavity])));
    }
    let shell = topo.add_shell(Shell::new(faces)?);
    Ok(topo.add_solid(Solid::new(shell, vec![])))
}

/// A face whose loops run about its outward normal; when its surface's own
/// normal points inward it is reversed and its loops stored the other way.
fn add_face(
    topo: &mut Topology,
    loops: Vec<Vec<OrientedEdge>>,
    surface: FaceSurface,
    outward: bool,
) -> Result<FaceId, crate::OperationsError> {
    let mut wires = Vec::with_capacity(loops.len());
    for mut edges in loops {
        if !outward {
            edges.reverse();
            edges = edges
                .into_iter()
                .map(|oe| OrientedEdge::new(oe.edge(), !oe.is_forward()))
                .collect();
        }
        wires.push(topo.add_wire(Wire::new(edges, true)?));
    }
    let outer = wires.remove(0);
    Ok(topo.add_face(if outward {
        Face::new(outer, wires, surface)
    } else {
        Face::new_reversed(outer, wires, surface)
    }))
}

/// `surface` moved `distance` along its own normal.
fn offset_surface(
    surface: &FaceSurface,
    distance: f64,
) -> Result<FaceSurface, crate::OperationsError> {
    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };
    let collapses = |what: &str| crate::OperationsError::InvalidInput {
        reason: format!("thicken: the {what} collapses at this thickness"),
    };
    Ok(match surface {
        FaceSurface::Cylinder(c) => {
            let r = c.radius() + distance;
            if r <= 0.0 {
                return Err(collapses("cylinder"));
            }
            FaceSurface::Cylinder(CylindricalSurface::with_ref_dir(
                c.origin(),
                c.axis(),
                r,
                c.x_axis(),
            )?)
        }
        FaceSurface::Sphere(s) => {
            let r = s.radius() + distance;
            if r <= 0.0 {
                return Err(collapses("sphere"));
            }
            FaceSurface::Sphere(SphericalSurface::with_axis_and_ref_dir(
                s.center(),
                r,
                s.z_axis(),
                s.x_axis(),
            )?)
        }
        FaceSurface::Torus(t) => {
            let r = t.minor_radius() + distance;
            if r <= 0.0 {
                return Err(collapses("torus"));
            }
            FaceSurface::Torus(ToroidalSurface::with_axis_and_ref_dir(
                t.center(),
                t.major_radius(),
                r,
                t.z_axis(),
                t.x_axis(),
            )?)
        }
        FaceSurface::Cone(c) => {
            // The normal leans off the axis by the half angle's complement,
            // so the apex moves along the axis by `distance / cos`.
            let apex = c.apex() + c.axis() * (-distance / c.half_angle().cos());
            FaceSurface::Cone(ConicalSurface::with_ref_dir(
                apex,
                c.axis(),
                c.half_angle(),
                c.x_axis(),
            )?)
        }
        FaceSurface::Plane { normal, d } => FaceSurface::Plane {
            normal: *normal,
            d: d + distance,
        },
        FaceSurface::Nurbs(_) => {
            return Err(crate::OperationsError::InvalidInput {
                reason: "thicken does not take a face on a curved NURBS surface".into(),
            });
        }
    })
}

/// Refuses a thickness at which a torus face, or its image `distance` along
/// the normals, reaches the torus's axis: there every ring angle carries to
/// one point, so the far side pinches or runs through itself.
fn keep_off_the_axis(
    topo: &Topology,
    face: FaceId,
    surface: &FaceSurface,
    distance: f64,
) -> Result<(), crate::OperationsError> {
    let FaceSurface::Torus(t) = surface else {
        return Ok(());
    };
    // A point at tube angle v lies R + r cos v from the axis and its image
    // R + (r + distance) cos v, both rising with cos v. The face comes
    // nearest its inner equator on its boundary unless it holds that
    // equator; one no edge reaches lies wholly in the face or out of it, so
    // one point of it tells which.
    let inner_equator = t.evaluate(0.0, std::f64::consts::PI);
    let cos_v = if brepkit_check::classify::face_contains(topo, face, inner_equator)? {
        -1.0
    } else {
        least_on_boundary(topo, face, |p| t.project_point(p).1.cos())?
    };
    let (big, small) = (t.major_radius(), t.minor_radius());
    let nearest = small
        .mul_add(cos_v, big)
        .min((small + distance).mul_add(cos_v, big));
    if nearest <= Tolerance::new().linear {
        return Err(crate::OperationsError::InvalidInput {
            reason: "thicken: the torus wall reaches its axis at this thickness".into(),
        });
    }
    Ok(())
}

/// The least of `f` along the edges of `face`, sampled along each and
/// refined about each least sample: a rim's least can sit between samples.
fn least_on_boundary(
    topo: &Topology,
    face: FaceId,
    f: impl Fn(Point3) -> f64,
) -> Result<f64, crate::OperationsError> {
    const SAMPLES: usize = 32;
    let face_data = topo.face(face)?;
    let mut least = f64::INFINITY;
    for wid in
        std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
    {
        for oe in topo.wire(wid)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (a, b) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let curve = edge.curve();
            let (t0, t1) = curve.domain_with_endpoints(a, b);
            #[allow(clippy::cast_precision_loss)]
            let param = |k: usize| t0 + (t1 - t0) * (k as f64 / SAMPLES as f64);
            let at = |t: f64| f(curve.evaluate_with_endpoints(t, a, b));
            let values: Vec<f64> = (0..=SAMPLES).map(|k| at(param(k))).collect();
            for (k, &value) in values.iter().enumerate() {
                least = least.min(value);
                let (before, after) = (k.saturating_sub(1), (k + 1).min(SAMPLES));
                let (lo, hi) = (
                    values[before].min(values[after]),
                    values[before].max(values[after]),
                );
                if value <= lo && value < hi {
                    least = least.min(golden_min(&at, param(before), param(after)));
                }
            }
        }
    }
    Ok(least)
}

/// The least of `f` on `[a, b]`, where it has a single dip.
fn golden_min(f: &impl Fn(f64) -> f64, mut a: f64, mut b: f64) -> f64 {
    let r = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut c, mut d) = (b - r * (b - a), a + r * (b - a));
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..64 {
        if fc < fd {
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
    fc.min(fd)
}

/// The unit normal of `surface` at `p`, a point on it.
fn surface_normal(surface: &FaceSurface, p: Point3) -> Vec3 {
    surface
        .project_point(p)
        .map_or(Vec3::new(0.0, 0.0, 1.0), |(u, v)| surface.normal(u, v))
}

/// The offset of a cylinder or sphere by `distance` along its normal as an
/// affine map, which carries any curve on it: a scaling about the axis or
/// centre.
fn affine_carry(surface: &FaceSurface, distance: f64) -> Option<Box<dyn Fn(Point3) -> Point3>> {
    match surface {
        FaceSurface::Cylinder(c) => {
            let (o, a, k) = (c.origin(), c.axis(), (c.radius() + distance) / c.radius());
            Some(Box::new(move |p: Point3| {
                let d = p - o;
                let up = a * a.dot(d);
                o + up + (d - up) * k
            }))
        }
        FaceSurface::Sphere(s) => {
            let (c, k) = (s.center(), (s.radius() + distance) / s.radius());
            Some(Box::new(move |p: Point3| c + (p - c) * k))
        }
        FaceSurface::Plane { .. }
        | FaceSurface::Cone(_)
        | FaceSurface::Torus(_)
        | FaceSurface::Nurbs(_) => None,
    }
}

/// `curve` from `a` to `b` carried along the normals by `carry`: a line
/// stays a line and a circle a circle when the carried points say so, and
/// any curve is carried exactly by `affine`. `None` when neither holds.
fn carry_curve(
    curve: &EdgeCurve,
    a: Point3,
    b: Point3,
    carry: &dyn Fn(Point3) -> Point3,
    affine: Option<&dyn Fn(Point3) -> Point3>,
    tol: Tolerance,
) -> Option<EdgeCurve> {
    let close = 10.0 * tol.linear;
    match curve {
        EdgeCurve::Line if affine.is_some() => Some(EdgeCurve::Line),
        EdgeCurve::Line => {
            let (fa, fb) = (carry(a), carry(b));
            let fm = carry(a + (b - a) * 0.5);
            let chord = fb - fa;
            let off =
                (fm - fa) - chord * ((fm - fa).dot(chord) / chord.length_squared().max(1e-300));
            (off.length() <= close).then_some(EdgeCurve::Line)
        }
        EdgeCurve::Circle(c) => {
            let at = |k: f64| carry(c.evaluate(k * std::f64::consts::FRAC_PI_3));
            let (p0, p2, p4) = (at(0.0), at(2.0), at(4.0));
            let (x, y) = (p0 - p4, p2 - p4);
            let n = x.cross(y);
            let nn = n.length_squared();
            if nn <= 0.0 {
                return None;
            }
            let centre =
                p4 + (y * x.length_squared() - x * y.length_squared()).cross(n) * (0.5 / nn);
            let radius = (p0 - centre).length();
            let fits = [1.0, 3.0, 5.0]
                .into_iter()
                .all(|k| ((at(k) - centre).length() - radius).abs() <= close);
            if !fits {
                return None;
            }
            Circle3D::new(centre, c.normal(), radius)
                .ok()
                .map(EdgeCurve::Circle)
        }
        EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
            let affine = affine?;
            let nurbs = as_nurbs(curve, a, b)?;
            let points = nurbs.control_points().iter().map(|&p| affine(p)).collect();
            NurbsCurve::new(
                nurbs.degree(),
                nurbs.knots().to_vec(),
                points,
                nurbs.weights().to_vec(),
            )
            .ok()
            .map(EdgeCurve::NurbsCurve)
        }
    }
}

/// `curve` from `a` to `b` as a NURBS curve over the edge's span.
fn as_nurbs(curve: &EdgeCurve, a: Point3, b: Point3) -> Option<NurbsCurve> {
    let (t0, t1) = curve.domain_with_endpoints(a, b);
    match curve {
        EdgeCurve::Line => {
            NurbsCurve::new(1, vec![0.0, 0.0, 1.0, 1.0], vec![a, b], vec![1.0; 2]).ok()
        }
        EdgeCurve::Circle(c) => brepkit_geometry::convert::circle_to_nurbs(c, t0, t1).ok(),
        EdgeCurve::Ellipse(e) => brepkit_geometry::convert::ellipse_to_nurbs(e, t0, t1).ok(),
        EdgeCurve::NurbsCurve(n) => Some(n.clone()),
    }
}

fn circle_radius(curve: &EdgeCurve) -> f64 {
    match curve {
        EdgeCurve::Circle(c) => c.radius(),
        EdgeCurve::Line | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => 0.0,
    }
}

/// The surface a side sweeps between an edge and its far image.
enum SideSurface {
    /// A plane annulus between two closed circles, with its outward normal.
    Annulus(Vec3),
    /// A surface, and whether its own normal points out of the solid.
    Ruled(FaceSurface, bool),
}

/// The side swept by the normals from `curve` (from `a` to `b`) to
/// `far_curve` (from `fa` to `fb`), facing `out` at `mid`: a plane between
/// lines or circles in one plane, a cylinder or cone between coaxial
/// circles, else the ruled NURBS between the two curves.
#[allow(clippy::too_many_arguments)]
fn side_surface(
    curve: &EdgeCurve,
    a: Point3,
    b: Point3,
    far_curve: &EdgeCurve,
    fa: Point3,
    fb: Point3,
    mid: Point3,
    out: Vec3,
    tol: Tolerance,
) -> Result<SideSurface, crate::OperationsError> {
    let flat = |normal: Vec3| -> FaceSurface {
        FaceSurface::Plane {
            normal,
            d: normal.dot(mid - Point3::new(0.0, 0.0, 0.0)),
        }
    };
    let out = out.normalize()?;
    match (curve, far_curve) {
        (EdgeCurve::Line, EdgeCurve::Line) => {
            let n = (b - a).cross(fa - a);
            if let Ok(n) = n.normalize()
                && n.dot(fb - a).abs() <= 10.0 * tol.linear
            {
                let n = if n.dot(out) < 0.0 { -n } else { n };
                return Ok(SideSurface::Ruled(flat(n), true));
            }
        }
        (EdgeCurve::Circle(c0), EdgeCurve::Circle(c1)) => {
            let n = c0.normal().normalize()?;
            let lateral = (c1.center() - c0.center()) - n * (c1.center() - c0.center()).dot(n);
            let coplanar = (c1.center() - c0.center()).dot(n).abs() <= tol.linear
                && lateral.length() <= tol.linear
                && c1.normal().normalize()?.dot(n).abs() >= 1.0 - tol.angular;
            if coplanar {
                let n = if n.dot(out) < 0.0 { -n } else { n };
                if (a - b).length() <= tol.linear {
                    return Ok(SideSurface::Annulus(n));
                }
                return Ok(SideSurface::Ruled(flat(n), true));
            }
            // The band's cone opens from its apex toward the band when the
            // circles' normal runs from the near one to the far.
            let toward = if (c1.center() - c0.center()).dot(n) < 0.0 {
                -n
            } else {
                n
            };
            let near = Circle3D::new(c0.center(), toward, c0.radius())?;
            let far = Circle3D::new(c1.center(), toward, c1.radius())?;
            if let Some(surface) = crate::loft::coaxial_corner_surface(&near, &far, tol) {
                let outward = surface
                    .project_point(mid)
                    .is_some_and(|(u, v)| surface.normal(u, v).dot(out) > 0.0);
                return Ok(SideSurface::Ruled(surface, outward));
            }
        }
        _ => {}
    }
    let (near, far) = (as_nurbs(curve, a, b), as_nurbs(far_curve, fa, fb));
    let ruled = near.zip(far).and_then(|(near, far)| {
        (near.degree() == far.degree() && near.control_points().len() == far.control_points().len())
            .then(|| {
                NurbsSurface::new(
                    1,
                    near.degree(),
                    vec![0.0, 0.0, 1.0, 1.0],
                    near.knots().to_vec(),
                    vec![
                        near.control_points().to_vec(),
                        far.control_points().to_vec(),
                    ],
                    vec![near.weights().to_vec(), far.weights().to_vec()],
                )
                .ok()
            })
            .flatten()
    });
    let Some(ruled) = ruled else {
        return Err(crate::OperationsError::InvalidInput {
            reason: "thicken: no ruled surface between an edge and its image".into(),
        });
    };
    let outward = ruled.normal(0.5, 0.5).is_ok_and(|n| n.dot(out) > 0.0);
    Ok(SideSurface::Ruled(FaceSurface::Nurbs(ruled), outward))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

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

    /// The side of a cylinder of radius 10 and height 20, thickened by 2,
    /// is a tube wall of radii 10 and 12 holding π (12² − 10²) 20: its own
    /// face, the offset cylinder and the two annuli between their rims.
    /// Thickened by -2, the wall grows inward to radius 8.
    #[test]
    fn thicken_a_cylinders_side_into_a_wall() {
        use std::f64::consts::PI;
        for (thickness, inner, outer) in [(2.0, 10.0, 12.0), (-2.0, 8.0, 10.0)] {
            let mut topo = Topology::new();
            let cylinder = crate::primitives::make_cylinder(&mut topo, 10.0, 20.0).unwrap();
            let side = brepkit_topology::explorer::solid_faces(&topo, cylinder)
                .unwrap()
                .into_iter()
                .find(|&f| !topo.face(f).unwrap().surface().is_planar())
                .unwrap();
            let wall = thicken(&mut topo, side, thickness).unwrap();
            let faces = brepkit_topology::explorer::solid_faces(&topo, wall).unwrap();
            let kinds: Vec<_> = faces
                .iter()
                .map(|&f| topo.face(f).unwrap().surface().type_tag())
                .collect();
            assert_eq!(kinds.len(), 4, "{kinds:?}");
            let report = crate::validate::validate_solid(&topo, wall).unwrap();
            assert!(report.is_valid(), "{thickness}: {:?}", report.issues);
            let volume = crate::measure::solid_volume(&topo, wall, 1e-3).unwrap();
            let exact = PI * (outer * outer - inner * inner) * 20.0;
            assert!(
                (volume - exact).abs() < 1e-6 * exact,
                "{thickness}: {volume} of {exact}"
            );
            let b = crate::measure::solid_bounding_box(&topo, wall).unwrap();
            assert!((b.max.x() - outer).abs() < 1e-6 && (b.min.y() + outer).abs() < 1e-6);
        }
    }

    /// The side of a frustum of radii 10 and 5 and height 12 (slant 13),
    /// thickened by 1 or -1, is a wall whose cross-section, 13 by 1, turns
    /// about the axis with its centre at radius 7.5 ± 12/26: by Pappus,
    /// 2π (97.5 ± 6). Its rims carry to circles, joined by conical bands.
    #[test]
    fn thicken_a_frustums_side_into_a_wall() {
        use std::f64::consts::PI;
        for (thickness, exact) in [(1.0, 2.0 * PI * 103.5), (-1.0, 2.0 * PI * 91.5)] {
            let mut topo = Topology::new();
            let cone = crate::primitives::make_cone(&mut topo, 10.0, 5.0, 12.0).unwrap();
            let side = brepkit_topology::explorer::solid_faces(&topo, cone)
                .unwrap()
                .into_iter()
                .find(|&f| !topo.face(f).unwrap().surface().is_planar())
                .unwrap();
            let wall = thicken(&mut topo, side, thickness).unwrap();
            let faces = brepkit_topology::explorer::solid_faces(&topo, wall).unwrap();
            assert_eq!(faces.len(), 4);
            let report = crate::validate::validate_solid(&topo, wall).unwrap();
            assert!(report.is_valid(), "{thickness}: {:?}", report.issues);
            let volume = crate::measure::solid_volume(&topo, wall, 1e-3).unwrap();
            assert!(
                (volume - exact).abs() < 1e-4 * exact,
                "{thickness}: {volume} of {exact}"
            );
        }
    }

    /// The torus face of a circle of radius 3 revolved a quarter turn at
    /// radius 20: a bend covering the whole tube round.
    fn quarter_bend(topo: &mut Topology) -> FaceId {
        let c = Circle3D::new(Point3::new(0.0, 0.0, 30.0), Vec3::new(0.0, 0.0, 1.0), 3.0).unwrap();
        let v = topo.add_vertex(Vertex::new(c.evaluate(0.0), 1e-7));
        let e = topo.add_edge(Edge::new(v, v, EdgeCurve::Circle(c)));
        let w = topo.add_wire(Wire::new(vec![OrientedEdge::new(e, true)], true).unwrap());
        let disc = brepkit_topology::builder::make_face_from_wire(topo, w).unwrap();
        let bend = crate::revolve::revolve(
            topo,
            disc,
            Point3::new(20.0, 0.0, 30.0),
            Vec3::new(0.0, 1.0, 0.0),
            std::f64::consts::FRAC_PI_2,
        )
        .unwrap();
        brepkit_topology::explorer::solid_faces(topo, bend)
            .unwrap()
            .into_iter()
            .find(|&f| matches!(topo.face(f).unwrap().surface(), FaceSurface::Torus(_)))
            .unwrap()
    }

    /// The bend's torus face, thickened by 1, is a quarter of a tube wall
    /// of radii 3 and 4: 7π swept 10π along, 70π². Its end circles carry to
    /// concentric circles, so the wall's ends are annuli.
    #[test]
    fn thicken_a_bends_torus_face_into_a_wall() {
        let mut topo = Topology::new();
        let torus = quarter_bend(&mut topo);
        let wall = thicken(&mut topo, torus, 1.0).unwrap();
        assert_eq!(
            brepkit_topology::explorer::solid_faces(&topo, wall)
                .unwrap()
                .len(),
            4
        );
        let report = crate::validate::validate_solid(&topo, wall).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let volume = crate::measure::solid_volume(&topo, wall, 1e-3).unwrap();
        let exact = 70.0 * std::f64::consts::PI * std::f64::consts::PI;
        assert!((volume - exact).abs() < 1e-4 * exact, "{volume} of {exact}");
    }

    /// Thickened by t, a torus face of radii 20 and 3 covering the whole
    /// tube round grows to minor radius 3 + t, whose inner equator lies
    /// 17 − t from the axis. At 16 the bend is a quarter wall of radii 3
    /// and 19, 3520π²; at 17 its far side pinches on the axis and at 18 runs
    /// through itself. A whole torus's seams are lines of no length at one
    /// point of its outer equator, so only the face itself reaches its inner
    /// equator; it is refused at 18 too.
    #[test]
    fn thicken_refuses_a_torus_wall_reaching_its_axis() {
        use std::f64::consts::PI;
        let mut topo = Topology::new();
        let bend = quarter_bend(&mut topo);
        let wall = thicken(&mut topo, bend, 16.0).unwrap();
        let report = crate::validate::validate_solid(&topo, wall).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let volume = crate::measure::solid_volume(&topo, wall, 1e-3).unwrap();
        let exact = 3520.0 * PI * PI;
        assert!((volume - exact).abs() < 1e-4 * exact, "{volume} of {exact}");
        for thickness in [17.0, 18.0] {
            let mut topo = Topology::new();
            let bend = quarter_bend(&mut topo);
            assert!(thicken(&mut topo, bend, thickness).is_err(), "{thickness}");
        }

        let whole = |topo: &mut Topology| {
            let torus = crate::primitives::make_torus(topo, 20.0, 3.0, 16).unwrap();
            brepkit_topology::explorer::solid_faces(topo, torus).unwrap()[0]
        };
        let mut topo = Topology::new();
        let face = whole(&mut topo);
        let ring = thicken(&mut topo, face, 16.0).unwrap();
        let volume = crate::measure::solid_volume(&topo, ring, 1e-3).unwrap();
        let exact = 2.0 * PI * PI * 20.0 * (19.0 * 19.0 - 9.0);
        assert!((volume - exact).abs() < 1e-6 * exact, "{volume} of {exact}");
        let mut topo = Topology::new();
        let face = whole(&mut topo);
        assert!(thicken(&mut topo, face, 18.0).is_err());
    }

    /// A whole torus of radii 20 and 3 is one face, every edge a seam, so
    /// thickened by 1 it is a ring whose tube, of radius 4, holds a cavity
    /// of radius 3 in a shell of its own: 2π² 20 (4² − 3²). Thickened by -1
    /// the torus itself is outermost, round a cavity of radius 2.
    #[test]
    fn thicken_a_whole_torus_into_a_hollow_ring() {
        for (thickness, outer, inner) in [(1.0, 4.0, 3.0), (-1.0, 3.0, 2.0)] {
            let mut topo = Topology::new();
            let torus = crate::primitives::make_torus(&mut topo, 20.0, 3.0, 16).unwrap();
            let face = brepkit_topology::explorer::solid_faces(&topo, torus).unwrap()[0];
            let ring = thicken(&mut topo, face, thickness).unwrap();
            let solid = topo.solid(ring).unwrap();
            assert_eq!(solid.inner_shells().len(), 1, "{thickness}");
            let minor = |shell| {
                let faces = topo.shell(shell).unwrap().faces();
                assert_eq!(faces.len(), 1, "{thickness}");
                let face = topo.face(faces[0]).unwrap();
                let radius = match face.surface() {
                    FaceSurface::Torus(t) => t.minor_radius(),
                    _ => f64::NAN,
                };
                (radius, face.is_reversed())
            };
            assert_eq!(minor(solid.outer_shell()), (outer, false), "{thickness}");
            assert_eq!(minor(solid.inner_shells()[0]), (inner, true), "{thickness}");
            let report = crate::validate::validate_solid(&topo, ring).unwrap();
            assert!(report.is_valid(), "{thickness}: {:?}", report.issues);
            let volume = crate::measure::solid_volume(&topo, ring, 1e-3).unwrap();
            let exact = 2.0 * std::f64::consts::PI.powi(2) * 20.0 * (outer * outer - inner * inner);
            assert!(
                (volume - exact).abs() < 1e-6 * exact,
                "{thickness}: {volume} of {exact}"
            );
        }
    }

    /// A ball's upper half, thickened by 1, is a hemispherical shell of
    /// radii 10 and 11 holding about 2/3 π (11³ − 10³).
    #[test]
    fn thicken_a_hemisphere_into_a_shell() {
        let mut topo = Topology::new();
        let ball = crate::primitives::make_sphere(&mut topo, 10.0, 32).unwrap();
        let top = brepkit_topology::explorer::solid_faces(&topo, ball)
            .unwrap()
            .into_iter()
            .find(|&f| {
                let face = topo.face(f).unwrap();
                let FaceSurface::Sphere(s) = face.surface() else {
                    return false;
                };
                let p = s.evaluate(0.3, 1.0);
                let n = face.surface().normal(0.3, 1.0);
                let n = if face.is_reversed() { -n } else { n };
                p.z() > 0.0 && n.z() > 0.0
            })
            .unwrap();
        let shell = thicken(&mut topo, top, 1.0).unwrap();
        let report = crate::validate::validate_solid(&topo, shell).unwrap();
        assert!(report.is_valid(), "{:?}", report.issues);
        let volume = crate::measure::solid_volume(&topo, shell, 1e-3).unwrap();
        let exact = 2.0 / 3.0 * std::f64::consts::PI * (1331.0 - 1000.0);
        assert!((volume - exact).abs() < 1e-2 * exact, "{volume} of {exact}");
    }

    /// A face on a twisted NURBS patch is refused rather than extruded
    /// along one normal.
    #[test]
    fn thicken_refuses_a_curved_nurbs_face() {
        use brepkit_math::nurbs::surface::NurbsSurface;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};
        let mut topo = Topology::new();
        let p = Point3::new;
        let corners = [
            p(0.0, 0.0, 0.0),
            p(1.0, 0.0, 0.0),
            p(1.0, 1.0, 1.0),
            p(0.0, 1.0, 0.0),
        ];
        let surface = NurbsSurface::new(
            1,
            1,
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![vec![corners[0], corners[3]], vec![corners[1], corners[2]]],
            vec![vec![1.0; 2]; 2],
        )
        .unwrap();
        let vs = corners.map(|c| topo.add_vertex(Vertex::new(c, 1e-7)));
        let edges = (0..4)
            .map(|i| {
                let e = topo.add_edge(Edge::new(vs[i], vs[(i + 1) % 4], EdgeCurve::Line));
                OrientedEdge::new(e, true)
            })
            .collect();
        let w = topo.add_wire(Wire::new(edges, true).unwrap());
        let face = topo.add_face(Face::new(w, vec![], FaceSurface::Nurbs(surface)));
        assert!(thicken(&mut topo, face, 1.0).is_err());
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
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 0.0), tol));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 0.0), tol));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(1.0, 1.0, 0.0), tol));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(0.0, 1.0, 0.0), tol));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v3, EdgeCurve::Line));
        let e3 = topo.add_edge(Edge::new(v3, v0, EdgeCurve::Line));

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
        let wid = topo.add_wire(wire);
        let face = topo.add_face(Face::new(wid, vec![], FaceSurface::Nurbs(surface)));

        let solid = thicken(&mut topo, face, 2.0).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "thickened NURBS face should have positive volume, got {vol}"
        );
    }
}

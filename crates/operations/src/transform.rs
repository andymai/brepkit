//! Affine transforms applied to topological shapes.

use std::collections::HashSet;

use brepkit_math::mat::Mat4;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::Vec3;
use brepkit_topology::Topology;
use brepkit_topology::edge::{EdgeCurve, EdgeId};
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::solid::SolidId;
use brepkit_topology::vertex::VertexId;
use brepkit_topology::wire::{OrientedEdge, WireId};

/// Apply an affine transform to a solid, modifying vertex positions and
/// face surface geometry in place.
///
/// The transform matrix must be non-degenerate (non-zero determinant).
/// Every vertex, edge curve, and face surface reachable from the solid's
/// shells is mapped to its exact image:
///
/// - Analytic surfaces keep their full reference frame, so a rotated torus,
///   sphere, cone, or cylinder is the same surface in its new pose with the
///   same `(u, v)` parameterization. A map that would stop a surface being
///   circular (a non-uniform scale across a cylinder's axis, say) converts
///   that face to the NURBS image of the surface.
/// - A mirror (negative determinant) turns every boundary clockwise about
///   its face's outward normal. A face with an explicit normal (a plane or
///   quadric) keeps that normal outward and reverses its wires; a NURBS
///   face's Su × Sv turns inward, so it keeps its wires and flips its
///   `reversed` flag. Either way each stored outer wire still winds
///   counter-clockwise about its surface normal.
/// - Stored pcurves survive only on faces whose parameterization is carried
///   over exactly; the rest are dropped for consumers to recompute.
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
pub fn transform_solid(
    topo: &mut Topology,
    solid: SolidId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let inverse = checked_inverse(matrix)?;
    let (vertex_ids, edge_ids, face_ids) = collect_solid_entities(topo, solid)?;
    transform_topology(topo, &vertex_ids, &edge_ids, &face_ids, matrix, &inverse)
}

/// Checks the matrix is invertible and returns its inverse.
fn checked_inverse(matrix: &Mat4) -> Result<Mat4, crate::OperationsError> {
    let tol = Tolerance::new();
    if tol.approx_eq(matrix.determinant(), 0.0) {
        return Err(crate::OperationsError::InvalidInput {
            reason: "transform matrix is degenerate (zero determinant)".into(),
        });
    }
    Ok(matrix.inverse()?)
}

/// Transform a closed set of faces with the edges and vertices they use.
fn transform_topology(
    topo: &mut Topology,
    vertex_ids: &HashSet<VertexId>,
    edge_ids: &HashSet<EdgeId>,
    face_ids: &HashSet<FaceId>,
    matrix: &Mat4,
    inverse: &Mat4,
) -> Result<(), crate::OperationsError> {
    // Surfaces go first: the NURBS fallbacks read a face's parameter range
    // off its boundary vertices, which must still lie on the source surface.
    let mut stale_pcurves = HashSet::new();
    let mut flipped = HashSet::new();
    for &fid in face_ids {
        let (keeps_parameterization, flips_face) =
            transform_face_surface(topo, fid, matrix, inverse)?;
        if !keeps_parameterization {
            stale_pcurves.insert(fid);
        }
        if flips_face {
            flipped.insert(fid);
        }
    }
    for &vid in vertex_ids {
        let vertex = topo.vertex_mut(vid)?;
        let new_point = matrix.mul_point(vertex.point());
        vertex.set_point(new_point);
    }
    let moved_origins = transform_edges(topo, edge_ids, matrix)?;
    // A mirror turns every face's boundary clockwise about its outward
    // normal. A face with an explicit normal (plane, quadric) keeps that
    // normal outward, so its wires reverse. A NURBS image's Su × Sv turns
    // inward instead, which already leaves its wires counter-clockwise about
    // the surface normal: only its flag flips.
    if matrix.determinant() < 0.0 {
        let kept: HashSet<FaceId> = face_ids.difference(&flipped).copied().collect();
        separate_shared_wires(topo, &kept, &flipped)?;
        reverse_face_wires(topo, &kept)?;
    }
    topo.pcurves_mut().remove_faces(&stale_pcurves);
    topo.pcurves_mut().remove_edges(&moved_origins);
    Ok(())
}

/// Give each of `flipped` its own copy of any wire it shares with a face in
/// `kept`: the two groups need opposite senses of the same boundary.
fn separate_shared_wires(
    topo: &mut Topology,
    kept: &HashSet<FaceId>,
    flipped: &HashSet<FaceId>,
) -> Result<(), crate::OperationsError> {
    let mut kept_wires = HashSet::new();
    for &fid in kept {
        let face = topo.face(fid)?;
        kept_wires.insert(face.outer_wire());
        kept_wires.extend(face.inner_wires().iter().copied());
    }
    for &fid in flipped {
        let face = topo.face(fid)?;
        let (outer, inner) = (face.outer_wire(), face.inner_wires().to_vec());
        if kept_wires.contains(&outer) {
            let copy = topo.wire(outer)?.clone();
            let copy = topo.add_wire(copy);
            topo.face_mut(fid)?.set_outer_wire(copy);
        }
        for (k, wid) in inner.into_iter().enumerate() {
            if kept_wires.contains(&wid) {
                let copy = topo.wire(wid)?.clone();
                let copy = topo.add_wire(copy);
                topo.face_mut(fid)?.inner_wires_mut()[k] = copy;
            }
        }
    }
    Ok(())
}

/// Reverse every wire of `faces`: order and per-edge sense.
fn reverse_face_wires(
    topo: &mut Topology,
    faces: &HashSet<FaceId>,
) -> Result<(), crate::OperationsError> {
    let mut wire_ids = HashSet::new();
    for &fid in faces {
        let face = topo.face(fid)?;
        wire_ids.insert(face.outer_wire());
        wire_ids.extend(face.inner_wires().iter().copied());
    }
    for wid in wire_ids {
        let edges = topo.wire_mut(wid)?.edges_mut();
        edges.reverse();
        for oe in edges.iter_mut() {
            *oe = OrientedEdge::new(oe.edge(), !oe.is_forward());
        }
    }
    Ok(())
}

/// The latitude range `(v_min, v_max)` a sphere face covers.
///
/// The boundary's own latitudes bound it. A face whose outer wire is one loop
/// around the axis (no seam, no inner wires) is a cap, and which pole it
/// holds follows from the loop's winding: the outer wire runs
/// counter-clockwise about the surface's outward normal (a reversed face
/// keeps its wire), so it circles the sphere axis counter-clockwise exactly
/// when the cap holds the north pole. A
/// primitive hemisphere is bounded by the equator alone and has no pole
/// vertex, so neither its latitudes nor its vertices can tell the two apart.
fn sphere_face_v_range(
    topo: &Topology,
    face_id: FaceId,
    sph: &brepkit_math::surfaces::SphericalSurface,
) -> Result<(f64, f64), crate::OperationsError> {
    use std::f64::consts::FRAC_PI_2;

    let face = topo.face(face_id)?;
    let wire = topo.wire(face.outer_wire())?;
    let center = sph.center();
    let axis = sph.z_axis();

    let mut samples: Vec<brepkit_math::vec::Point3> = Vec::new();
    let mut uses: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for oe in wire.edges() {
        *uses.entry(oe.edge().index()).or_default() += 1;
        let edge = topo.edge(oe.edge())?;
        let (sp, ep) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
        let n = if matches!(edge.curve(), EdgeCurve::Line) {
            1
        } else {
            16
        };
        let (from, to) = if oe.is_forward() { (t0, t1) } else { (t1, t0) };
        samples.extend((0..n).map(|k| {
            let t = from + (to - from) * f64::from(k) / f64::from(n);
            edge.curve().evaluate_with_endpoints(t, sp, ep)
        }));
    }
    if samples.is_empty() {
        return Ok((-FRAC_PI_2, FRAC_PI_2));
    }
    let (v_lo, v_hi) = samples.iter().fold((f64::MAX, f64::MIN), |(lo, hi), &p| {
        let v = sph.project_point(p).1;
        (lo.min(v), hi.max(v))
    });

    let seamed = uses.values().any(|&n| n > 1);
    if seamed || !face.inner_wires().is_empty() {
        return Ok((v_lo, v_hi));
    }
    let winding: f64 = samples
        .iter()
        .zip(samples.iter().cycle().skip(1))
        .map(|(&a, &b)| (a - center).cross(b - center).dot(axis))
        .sum();
    let north = winding > 0.0;
    Ok(if north {
        (v_lo, FRAC_PI_2)
    } else {
        (-FRAC_PI_2, v_hi)
    })
}

/// The linear part of `matrix` applied to `v` (no translation).
fn linear(matrix: &Mat4, v: Vec3) -> Vec3 {
    let m = &matrix.0;
    Vec3::new(
        m[0][0].mul_add(v.x(), m[0][1].mul_add(v.y(), m[0][2] * v.z())),
        m[1][0].mul_add(v.x(), m[1][1].mul_add(v.y(), m[1][2] * v.z())),
        m[2][0].mul_add(v.x(), m[2][1].mul_add(v.y(), m[2][2] * v.z())),
    )
}

/// Relative tolerance for "these image vectors are orthogonal and equally
/// long", the test that a map keeps a circle a circle.
const SHAPE_REL_TOL: f64 = 1e-9;

fn nearly_orthogonal(a: Vec3, b: Vec3) -> bool {
    a.dot(b).abs() <= SHAPE_REL_TOL * a.length() * b.length()
}

fn nearly_equal(a: f64, b: f64) -> bool {
    (a - b).abs() <= SHAPE_REL_TOL * a.abs().max(b.abs())
}

/// The scale factor if `matrix` is a similarity (rotation, reflection, and
/// uniform scale, plus translation), else `None`.
fn similarity_scale(matrix: &Mat4) -> Option<f64> {
    let cols = [
        linear(matrix, Vec3::new(1.0, 0.0, 0.0)),
        linear(matrix, Vec3::new(0.0, 1.0, 0.0)),
        linear(matrix, Vec3::new(0.0, 0.0, 1.0)),
    ];
    let s = cols[0].length();
    let similar = nearly_equal(cols[1].length(), s)
        && nearly_equal(cols[2].length(), s)
        && nearly_orthogonal(cols[0], cols[1])
        && nearly_orthogonal(cols[0], cols[2])
        && nearly_orthogonal(cols[1], cols[2]);
    similar.then_some(s)
}

/// For a surface of revolution with frame images `x`, `y` (radial) and `z`
/// (axis), the radial scale factor if the image is still a surface of
/// revolution about `z`: the radial images orthogonal, equally long, and
/// both perpendicular to the axis image.
fn revolution_scale(x: Vec3, y: Vec3, z: Vec3) -> Option<f64> {
    let s = x.length();
    (nearly_equal(y.length(), s)
        && nearly_orthogonal(x, y)
        && nearly_orthogonal(x, z)
        && nearly_orthogonal(y, z))
    .then_some(s)
}

/// The image of one face's surface under an affine map.
pub(crate) struct SurfaceImage {
    /// The mapped surface.
    pub surface: FaceSurface,
    /// Whether the face's `(u, v)` parameterization carries over unchanged,
    /// i.e. whether its stored pcurves remain valid.
    pub keeps_parameterization: bool,
    /// Whether the face's `reversed` flag must flip to keep it outward: a
    /// NURBS normal is the cross product of its partials, which a mirror
    /// turns inward.
    pub flips_face: bool,
}

/// The exact image of face `fid`'s surface under `matrix`.
///
/// Reads the face's boundary in its untransformed position (the NURBS
/// fallbacks take their parameter range from it).
#[allow(clippy::too_many_lines)]
pub(crate) fn surface_image(
    topo: &Topology,
    fid: FaceId,
    matrix: &Mat4,
    inverse: &Mat4,
) -> Result<SurfaceImage, crate::OperationsError> {
    use brepkit_heal::construct::convert_surface::{
        cone_to_nurbs, cylinder_to_nurbs, sphere_band_to_nurbs, torus_to_nurbs,
    };
    use brepkit_math::surfaces::{
        ConicalSurface, CylindricalSurface, SphericalSurface, ToroidalSurface,
    };

    let mirrored = matrix.determinant() < 0.0;
    let heal_err = |what: &str, e: brepkit_heal::HealError| crate::OperationsError::InvalidInput {
        reason: format!("{what} failed: {e}"),
    };
    let face = topo.face(fid)?;
    let (surface, preserved) = match face.surface().clone() {
        FaceSurface::Plane { normal, d } => {
            let origin = brepkit_math::vec::Point3::new(0.0, 0.0, 0.0);
            let new_normal = linear(&inverse.transpose(), normal).normalize()?;
            let on_plane = origin + normal * (d / normal.dot(normal));
            let new_d = new_normal.dot(matrix.mul_point(on_plane) - origin);
            (
                FaceSurface::Plane {
                    normal: new_normal,
                    d: new_d,
                },
                false,
            )
        }
        FaceSurface::Nurbs(s) => (
            FaceSurface::Nurbs(transform_nurbs_surface(&s, matrix)?),
            true,
        ),
        FaceSurface::Cylinder(cyl) => {
            let (x, y, z) = (
                linear(matrix, cyl.x_axis()),
                linear(matrix, cyl.y_axis()),
                linear(matrix, cyl.axis()),
            );
            if let Some(s) = revolution_scale(x, y, z) {
                let image = CylindricalSurface::with_ref_dir(
                    matrix.mul_point(cyl.origin()),
                    z,
                    cyl.radius() * s,
                    x,
                )?;
                (
                    FaceSurface::Cylinder(image),
                    !mirrored && nearly_equal(z.length(), 1.0),
                )
            } else {
                let v_range = face_v_range(topo, fid, |pt| cyl.project_point(pt).1, None)?;
                let nurbs = cylinder_to_nurbs(&cyl, v_range)
                    .map_err(|e| heal_err("cylinder_to_nurbs", e))?;
                (
                    FaceSurface::Nurbs(transform_nurbs_surface(&nurbs, matrix)?),
                    false,
                )
            }
        }
        FaceSurface::Cone(cone) => {
            let (x, y, z) = (
                linear(matrix, cone.x_axis()),
                linear(matrix, cone.y_axis()),
                linear(matrix, cone.axis()),
            );
            if let Some(s) = revolution_scale(x, y, z) {
                // The generator cos(a)·radial + sin(a)·axis maps to
                // s·cos(a)·radial' + |z'|·sin(a)·axis'.
                let (sin_a, cos_a) = cone.half_angle().sin_cos();
                let half_angle = (z.length() * sin_a).atan2(s * cos_a);
                let image =
                    ConicalSurface::with_ref_dir(matrix.mul_point(cone.apex()), z, half_angle, x)?;
                (
                    FaceSurface::Cone(image),
                    !mirrored && nearly_equal(s, 1.0) && nearly_equal(z.length(), 1.0),
                )
            } else {
                let v_range = face_v_range(topo, fid, |pt| cone.project_point(pt).1, Some(0.0))?;
                let nurbs =
                    cone_to_nurbs(&cone, v_range).map_err(|e| heal_err("cone_to_nurbs", e))?;
                (
                    FaceSurface::Nurbs(transform_nurbs_surface(&nurbs, matrix)?),
                    false,
                )
            }
        }
        FaceSurface::Sphere(sph) => {
            if let Some(s) = similarity_scale(matrix) {
                let image = SphericalSurface::with_axis_and_ref_dir(
                    matrix.mul_point(sph.center()),
                    sph.radius() * s,
                    linear(matrix, sph.z_axis()),
                    linear(matrix, sph.x_axis()),
                )?;
                (FaceSurface::Sphere(image), !mirrored)
            } else {
                let (v_min, v_max) = sphere_face_v_range(topo, fid, &sph)?;
                let nurbs = sphere_band_to_nurbs(&sph, v_min, v_max)
                    .map_err(|e| heal_err("sphere_band_to_nurbs", e))?;
                (
                    FaceSurface::Nurbs(transform_nurbs_surface(&nurbs, matrix)?),
                    false,
                )
            }
        }
        FaceSurface::Torus(tor) => {
            if let Some(s) = similarity_scale(matrix) {
                let image = ToroidalSurface::with_axis_and_ref_dir(
                    matrix.mul_point(tor.center()),
                    tor.major_radius() * s,
                    tor.minor_radius() * s,
                    linear(matrix, tor.z_axis()),
                    linear(matrix, tor.x_axis()),
                )?;
                (FaceSurface::Torus(image), !mirrored)
            } else {
                let nurbs = torus_to_nurbs(&tor).map_err(|e| heal_err("torus_to_nurbs", e))?;
                (
                    FaceSurface::Nurbs(transform_nurbs_surface(&nurbs, matrix)?),
                    false,
                )
            }
        }
    };
    let flips_face = mirrored && matches!(surface, FaceSurface::Nurbs(_));
    Ok(SurfaceImage {
        surface,
        keeps_parameterization: preserved,
        flips_face,
    })
}

/// Transform a single face's surface to its exact image under `matrix`.
/// Returns whether the face's stored pcurves remain valid, and whether its
/// flag flipped with the image's normal.
fn transform_face_surface(
    topo: &mut Topology,
    fid: FaceId,
    matrix: &Mat4,
    inverse: &Mat4,
) -> Result<(bool, bool), crate::OperationsError> {
    let image = surface_image(topo, fid, matrix, inverse)?;
    let face = topo.face_mut(fid)?;
    if image.flips_face {
        let reversed = face.is_reversed();
        face.set_reversed(!reversed);
    }
    face.set_surface(image.surface);
    Ok((image.keeps_parameterization, image.flips_face))
}

/// The v-parameter range a face's boundary covers, sampled along every edge
/// of every wire and padded by a twentieth of its span on each side, so a
/// NURBS patch built on it reaches past the face everywhere. `floor` keeps the
/// padded range strictly above it (a cone's apex).
fn face_v_range(
    topo: &Topology,
    face_id: FaceId,
    project_v: impl Fn(brepkit_math::vec::Point3) -> f64,
    floor: Option<f64>,
) -> Result<(f64, f64), crate::OperationsError> {
    let face = topo.face(face_id)?;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for wire_id in std::iter::once(face.outer_wire()).chain(face.inner_wires().iter().copied()) {
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (sp, ep) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(sp, ep);
            let n = if matches!(edge.curve(), EdgeCurve::Line) {
                1
            } else {
                32
            };
            for k in 0..=n {
                let t = t0 + (t1 - t0) * f64::from(k) / f64::from(n);
                let v = project_v(edge.curve().evaluate_with_endpoints(t, sp, ep));
                v_min = v_min.min(v);
                v_max = v_max.max(v);
            }
        }
    }
    if v_min.partial_cmp(&v_max) != Some(std::cmp::Ordering::Less) {
        return Err(crate::OperationsError::InvalidInput {
            reason: format!("face {face_id:?} spans no parameter range along v"),
        });
    }
    let pad = 0.05 * (v_max - v_min);
    let low = floor.map_or(v_min - pad, |f| (v_min - pad).max(f64::midpoint(f, v_min)));
    Ok((low, v_max + pad))
}

/// Transform a NURBS surface's control points by a matrix.
fn transform_nurbs_surface(
    surface: &NurbsSurface,
    matrix: &Mat4,
) -> Result<NurbsSurface, crate::OperationsError> {
    let new_cps: Vec<Vec<_>> = surface
        .control_points()
        .iter()
        .map(|row| row.iter().map(|pt| matrix.mul_point(*pt)).collect())
        .collect();
    Ok(NurbsSurface::new(
        surface.degree_u(),
        surface.degree_v(),
        surface.knots_u().to_vec(),
        surface.knots_v().to_vec(),
        new_cps,
        surface.weights().to_vec(),
    )?)
}

/// The image of an edge curve under `matrix`, and whether its parameter
/// origin moved off the image of the old one (a conic re-expressed on new
/// principal axes). Line geometry lives in the vertices, so a line maps to
/// itself.
pub(crate) fn curve_image(
    curve: &EdgeCurve,
    matrix: &Mat4,
) -> Result<(EdgeCurve, bool), crate::OperationsError> {
    Ok(match curve {
        EdgeCurve::Line => (EdgeCurve::Line, false),
        EdgeCurve::NurbsCurve(c) => {
            let control_points: Vec<_> = c
                .control_points()
                .iter()
                .map(|pt| matrix.mul_point(*pt))
                .collect();
            (
                EdgeCurve::NurbsCurve(NurbsCurve::new(
                    c.degree(),
                    c.knots().to_vec(),
                    control_points,
                    c.weights().to_vec(),
                )?),
                false,
            )
        }
        EdgeCurve::Circle(c) => transform_conic(
            matrix,
            c.center(),
            c.u_axis() * c.radius(),
            c.v_axis() * c.radius(),
        )?,
        EdgeCurve::Ellipse(e) => transform_conic(
            matrix,
            e.center(),
            e.u_axis() * e.semi_major(),
            e.v_axis() * e.semi_minor(),
        )?,
    })
}

/// Transform a set of edge curves in place. Returns the closed edges whose
/// parameter origin moved: a pcurve of one no longer starts where the edge
/// does.
fn transform_edges(
    topo: &mut Topology,
    edge_ids: &HashSet<EdgeId>,
    matrix: &Mat4,
) -> Result<HashSet<EdgeId>, crate::OperationsError> {
    let mut moved = HashSet::new();
    for &eid in edge_ids {
        let edge = topo.edge(eid)?;
        if matches!(edge.curve(), EdgeCurve::Line) {
            continue;
        }
        let closed = edge.start() == edge.end();
        let (curve, origin_moved) = curve_image(edge.curve(), matrix)?;
        if closed && origin_moved {
            moved.insert(eid);
        }
        topo.edge_mut(eid)?.set_curve(curve);
    }
    Ok(moved)
}

/// The exact image of the conic `center + p·cos(t) + q·sin(t)`.
///
/// An affine map sends it to `center' + p'·cos(t) + q'·sin(t)` with `p'`,
/// `q'` conjugate semi-diameters; the result is re-expressed on its principal
/// axes, a phase shift of `t` that keeps the direction of travel. A circle
/// whose image axes are still orthogonal and equal keeps them unchanged, so a
/// rigid motion preserves its parameterization exactly.
fn transform_conic(
    matrix: &Mat4,
    center: brepkit_math::vec::Point3,
    p: Vec3,
    q: Vec3,
) -> Result<(EdgeCurve, bool), crate::OperationsError> {
    use brepkit_math::curves::{Circle3D, Ellipse3D};

    let (p, q) = (linear(matrix, p), linear(matrix, q));
    let (a1, a2) = if nearly_orthogonal(p, q) {
        if !nearly_equal(p.length(), q.length()) && q.length() > p.length() {
            (q, -p)
        } else {
            (p, q)
        }
    } else {
        let t0 = 0.5 * (2.0 * p.dot(q)).atan2(p.dot(p) - q.dot(q));
        let (sin_t, cos_t) = t0.sin_cos();
        (p * cos_t + q * sin_t, q * cos_t - p * sin_t)
    };
    let (l1, l2) = (a1.length(), a2.length());
    let center = matrix.mul_point(center);
    let normal = a1.cross(a2).normalize()?;
    let (u, v) = (a1.normalize()?, a2.normalize()?);
    let origin_moved = a1 != p;
    let curve = if nearly_equal(l1, l2) {
        EdgeCurve::Circle(Circle3D::with_axes(center, normal, l1, u, v)?)
    } else {
        EdgeCurve::Ellipse(Ellipse3D::with_axes(center, normal, l1, l2, u, v)?)
    };
    Ok((curve, origin_moved))
}

/// Apply an affine transform to a wire, modifying vertex positions and
/// edge curve geometry in place.
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
pub fn transform_wire(
    topo: &mut Topology,
    wire_id: WireId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    checked_inverse(matrix)?;
    let (vertex_ids, edge_ids) = collect_wire_entities(topo, wire_id)?;

    // Transform vertices.
    for vid in vertex_ids {
        let vertex = topo.vertex_mut(vid)?;
        let new_point = matrix.mul_point(vertex.point());
        vertex.set_point(new_point);
    }

    let moved_origins = transform_edges(topo, &edge_ids, matrix)?;
    topo.pcurves_mut().remove_edges(&moved_origins);
    Ok(())
}

/// Apply an affine transform to a face, modifying vertex positions, edge
/// curve geometry, and the face surface in place.
///
/// Transforms all vertices/edges in the face's outer and inner wires and the
/// face surface, with the same exact-image rules as [`transform_solid`].
///
/// # Errors
///
/// Returns an error if the matrix is degenerate or a referenced entity is missing.
pub fn transform_face(
    topo: &mut Topology,
    face_id: FaceId,
    matrix: &Mat4,
) -> Result<(), crate::OperationsError> {
    let inverse = checked_inverse(matrix)?;
    let (vertex_ids, edge_ids) = collect_face_entities(topo, face_id)?;
    transform_topology(
        topo,
        &vertex_ids,
        &edge_ids,
        &HashSet::from([face_id]),
        matrix,
        &inverse,
    )
}

/// Traverses face → wires → edges → vertices and returns deduplicated sets.
fn collect_face_entities(
    topo: &Topology,
    face_id: FaceId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let face = topo.face(face_id)?;
    let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
        .chain(face.inner_wires().iter().copied())
        .collect();

    for wid in wire_ids {
        let wire = topo.wire(wid)?;
        for oe in wire.edges() {
            let eid = oe.edge();
            edge_ids.insert(eid);
            let edge = topo.edge(eid)?;
            vertex_ids.insert(edge.start());
            vertex_ids.insert(edge.end());
        }
    }

    Ok((vertex_ids, edge_ids))
}

/// Traverses wire → edges → vertices and returns deduplicated sets.
fn collect_wire_entities(
    topo: &Topology,
    wire_id: WireId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let wire = topo.wire(wire_id)?;
    for oe in wire.edges() {
        let eid = oe.edge();
        edge_ids.insert(eid);
        let edge = topo.edge(eid)?;
        vertex_ids.insert(edge.start());
        vertex_ids.insert(edge.end());
    }
    Ok((vertex_ids, edge_ids))
}

/// Traverses solid → shells → faces → wires → edges → vertices and
/// returns deduplicated sets of vertex IDs, edge IDs, and face IDs.
#[allow(clippy::type_complexity)]
fn collect_solid_entities(
    topo: &Topology,
    solid: SolidId,
) -> Result<(HashSet<VertexId>, HashSet<EdgeId>, HashSet<FaceId>), crate::OperationsError> {
    let mut vertex_ids = HashSet::new();
    let mut edge_ids = HashSet::new();
    let mut face_ids = HashSet::new();
    let solid_data = topo.solid(solid)?;
    let shell_ids: Vec<_> = std::iter::once(solid_data.outer_shell())
        .chain(solid_data.inner_shells().iter().copied())
        .collect();

    for shell_id in shell_ids {
        let shell = topo.shell(shell_id)?;
        let fids: Vec<_> = shell.faces().to_vec();

        for face_id in fids {
            face_ids.insert(face_id);
            let face = topo.face(face_id)?;
            let wire_ids: Vec<_> = std::iter::once(face.outer_wire())
                .chain(face.inner_wires().iter().copied())
                .collect();

            for wire_id in wire_ids {
                let wire = topo.wire(wire_id)?;
                for oe in wire.edges() {
                    let eid = oe.edge();
                    edge_ids.insert(eid);
                    let edge = topo.edge(eid)?;
                    vertex_ids.insert(edge.start());
                    vertex_ids.insert(edge.end());
                }
            }
        }
    }

    Ok((vertex_ids, edge_ids, face_ids))
}

#[cfg(test)]
mod tests;

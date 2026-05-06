//! Bounding box computation for B-rep solids.

use brepkit_math::aabb::Aabb3;
use brepkit_math::vec::Point3;
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use super::helpers::collect_solid_vertex_points;

/// Compute the axis-aligned bounding box of a solid.
///
/// Uses vertex positions as the base AABB, then expands for non-planar
/// surfaces by sampling edge midpoints on the surface. This captures
/// curvature without over-expanding (unlike projecting the surface's
/// full theoretical extent).
///
/// # Errors
///
/// Returns an error if the solid has no vertices or a topology lookup fails.
pub fn solid_bounding_box(
    topo: &Topology,
    solid: SolidId,
) -> Result<Aabb3, crate::OperationsError> {
    let points = collect_solid_vertex_points(topo, solid)?;
    let mut aabb = Aabb3::try_from_points(points.iter().copied()).ok_or_else(|| {
        crate::OperationsError::InvalidInput {
            reason: "solid has no vertices".into(),
        }
    })?;

    // Expand AABB for non-planar faces by sampling edge midpoints on the
    // actual surface. This captures curvature (e.g., the arc midpoint of a
    // fillet cylinder) without over-expanding to the surface's full extent.
    let solid_data = topo.solid(solid)?;
    let shell = topo.shell(solid_data.outer_shell())?;
    for &fid in shell.faces() {
        if let Ok(face) = topo.face(fid) {
            expand_aabb_for_face(topo, &mut aabb, fid, face.surface());
        }
    }

    Ok(aabb)
}

/// Expand an AABB to include a point.
fn aabb_include(aabb: &mut Aabb3, p: Point3) {
    *aabb = aabb.union(Aabb3 { min: p, max: p });
}

/// Expand an AABB for a face, accounting for surface curvature.
///
/// Uses different strategies based on surface type:
/// - **Sphere/Torus**: analytic expansion (full surface extent)
/// - **Cylinder/Cone**: wire-bounded expansion (sample edge midpoints
///   to avoid over-expanding for partial arcs like fillets)
/// - **NURBS**: sparse interior grid sampling
/// - **Plane**: no expansion needed
#[allow(clippy::too_many_lines)]
fn expand_aabb_for_face(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    surface: &FaceSurface,
) {
    // Always sample wire midpoints — captures curvature of curved boundary
    // edges (Circle, Ellipse, NurbsCurve) regardless of surface type.
    // Critical for: cone base discs (Plane face with circle edge), partial
    // arcs whose extremes lie between vertices, and any curved edge on a
    // planar face.
    sample_face_wire_midpoints(topo, aabb, face_id);

    match surface {
        FaceSurface::Plane { .. } => {}

        // Sphere and torus: use analytic expansion (these are typically full
        // or near-full surfaces where the extremes can be far from vertices).
        FaceSurface::Sphere(s) => {
            let c = s.center();
            let r = s.radius();
            aabb_include(aabb, Point3::new(c.x() - r, c.y() - r, c.z() - r));
            aabb_include(aabb, Point3::new(c.x() + r, c.y() + r, c.z() + r));
        }
        FaceSurface::Torus(t) => {
            // Per-dim half-extent of a torus = R * sqrt(1 - axis.d²) + r.
            // The R*sqrt(1-axis.d²) term is the major-circle's extent in
            // world dim d (zero for the dim aligned with the torus axis);
            // the r term is the minor radius offset, which can extend
            // freely in any direction. Replaces the previous formula that
            // applied (R+r) to all dimensions and over-estimated the
            // axis-aligned dim by `R`.
            let c = t.center();
            let r_major = t.major_radius();
            let r_minor = t.minor_radius();
            let axis = t.z_axis();
            let hx = r_major * (1.0 - axis.x() * axis.x()).max(0.0).sqrt() + r_minor;
            let hy = r_major * (1.0 - axis.y() * axis.y()).max(0.0).sqrt() + r_minor;
            let hz = r_major * (1.0 - axis.z() * axis.z()).max(0.0).sqrt() + r_minor;
            aabb_include(aabb, Point3::new(c.x() - hx, c.y() - hy, c.z() - hz));
            aabb_include(aabb, Point3::new(c.x() + hx, c.y() + hy, c.z() + hz));
        }

        // Cylinder: expand radially at each face vertex's axis projection.
        // Unlike the old approach that used AABB corners (which over-expands
        // for fillet cylinders), this uses the face's own vertices to
        // constrain the expansion to the actual face extent.
        FaceSurface::Cylinder(c) => {
            expand_cylinder_at_vertices(topo, aabb, face_id, c);
        }

        // Cone: expand radially at each face vertex (the radius varies per
        // axial position). Uses the vertex's own distance-from-axis as the
        // local radius, then projects to a full circle at that axial slice.
        FaceSurface::Cone(c) => {
            expand_cone_at_vertices(topo, aabb, face_id, c);
        }

        // NURBS: sample the surface at a sparse interior grid.
        FaceSurface::Nurbs(nurbs) => {
            let (u_min, u_max) = nurbs.domain_u();
            let (v_min, v_max) = nurbs.domain_v();
            let n_samples = 4;
            #[allow(clippy::cast_precision_loss)]
            for iu in 1..n_samples {
                let u = u_min + (u_max - u_min) * (iu as f64) / (n_samples as f64);
                for iv in 1..n_samples {
                    let v = v_min + (v_max - v_min) * (iv as f64) / (n_samples as f64);
                    aabb_include(aabb, nurbs.evaluate(u, v));
                }
            }
        }
    }
}

/// Sample edge midpoints along a face's outer wire to expand the AABB.
///
/// Returns `true` if any curved (non-Line) edges were found. For curved
/// edges (Circle, Ellipse, NurbsCurve), sampling at 0.25, 0.5, 0.75
/// captures the curvature.
fn sample_face_wire_midpoints(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
) -> bool {
    let Ok(face) = topo.face(face_id) else {
        return false;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return false;
    };
    let mut has_curved = false;
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        if !matches!(edge.curve(), brepkit_topology::edge::EdgeCurve::Line) {
            has_curved = true;
        }
        let Ok(sv) = topo.vertex(edge.start()) else {
            continue;
        };
        let Ok(ev) = topo.vertex(edge.end()) else {
            continue;
        };
        let p_start = sv.point();
        let p_end = ev.point();
        let (t0, t1) = edge.curve().domain_with_endpoints(p_start, p_end);
        for &frac in &[0.25, 0.5, 0.75] {
            let t = t0 + (t1 - t0) * frac;
            let pt = edge.curve().evaluate_with_endpoints(t, p_start, p_end);
            aabb_include(aabb, pt);
        }
    }
    has_curved
}

/// Expand AABB for a cylinder face by projecting each vertex onto the
/// cylinder axis and adding the full radial extent at that axial position.
fn expand_cylinder_at_vertices(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    cyl: &brepkit_math::surfaces::CylindricalSurface,
) {
    let Ok(face) = topo.face(face_id) else {
        return;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return;
    };
    let axis = cyl.axis();
    let origin = cyl.origin();
    let r = cyl.radius();
    let rx = r * (1.0 - axis.x() * axis.x()).max(0.0).sqrt();
    let ry = r * (1.0 - axis.y() * axis.y()).max(0.0).sqrt();
    let rz = r * (1.0 - axis.z() * axis.z()).max(0.0).sqrt();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        for vid in [edge.start(), edge.end()] {
            let Ok(v) = topo.vertex(vid) else {
                continue;
            };
            let rel = brepkit_math::vec::Vec3::new(
                v.point().x() - origin.x(),
                v.point().y() - origin.y(),
                v.point().z() - origin.z(),
            );
            let t = axis.dot(rel);
            let coa = Point3::new(
                origin.x() + axis.x() * t,
                origin.y() + axis.y() * t,
                origin.z() + axis.z() * t,
            );
            aabb_include(aabb, Point3::new(coa.x() - rx, coa.y() - ry, coa.z() - rz));
            aabb_include(aabb, Point3::new(coa.x() + rx, coa.y() + ry, coa.z() + rz));
        }
    }
}

/// Expand AABB for a cone face by computing each face vertex's radial
/// distance from the axis (the local cone radius at that axial slice),
/// then including a full circle of that radius at that slice.
fn expand_cone_at_vertices(
    topo: &Topology,
    aabb: &mut Aabb3,
    face_id: brepkit_topology::face::FaceId,
    cone: &brepkit_math::surfaces::ConicalSurface,
) {
    use brepkit_math::vec::Vec3;
    let Ok(face) = topo.face(face_id) else {
        return;
    };
    let Ok(wire) = topo.wire(face.outer_wire()) else {
        return;
    };
    let axis = cone.axis();
    let apex = cone.apex();
    // Axis-perpendicular projection scales for a full ring at slice centre.
    let sx = (1.0 - axis.x() * axis.x()).max(0.0).sqrt();
    let sy = (1.0 - axis.y() * axis.y()).max(0.0).sqrt();
    let sz = (1.0 - axis.z() * axis.z()).max(0.0).sqrt();
    for oe in wire.edges() {
        let Ok(edge) = topo.edge(oe.edge()) else {
            continue;
        };
        for vid in [edge.start(), edge.end()] {
            let Ok(v) = topo.vertex(vid) else {
                continue;
            };
            let rel = Vec3::new(
                v.point().x() - apex.x(),
                v.point().y() - apex.y(),
                v.point().z() - apex.z(),
            );
            let t = axis.dot(rel);
            let coa = Point3::new(
                apex.x() + axis.x() * t,
                apex.y() + axis.y() * t,
                apex.z() + axis.z() * t,
            );
            // Local radius is the perpendicular distance from axis to vertex.
            let perp = Vec3::new(
                rel.x() - axis.x() * t,
                rel.y() - axis.y() * t,
                rel.z() - axis.z() * t,
            );
            let r = perp.length();
            aabb_include(
                aabb,
                Point3::new(coa.x() - r * sx, coa.y() - r * sy, coa.z() - r * sz),
            );
            aabb_include(
                aabb,
                Point3::new(coa.x() + r * sx, coa.y() + r * sy, coa.z() + r * sz),
            );
        }
    }
}

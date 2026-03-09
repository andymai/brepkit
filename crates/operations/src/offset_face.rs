//! Face offset: create a new face offset from an existing face by a given
//! distance along its surface normal.
//!
//! For planar faces, this is an exact operation (translate along normal).
//! For NURBS faces, the offset is approximated by sampling the surface
//! normal field and refitting via surface interpolation.

use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::{FaceId, FaceSurface};

use crate::OperationsError;

/// Create a new face that is offset from `face_id` by `distance` along
/// the outward surface normal.
///
/// Positive distance offsets outward (away from the solid interior),
/// negative distance offsets inward.
///
/// For planar faces, the operation is exact. For NURBS faces, the
/// surface is sampled at a grid of `samples × samples` points and
/// re-interpolated.
///
/// # Errors
///
/// Returns an error if:
/// - The face lookup fails
/// - NURBS surface normal computation fails at any sample point
/// - Surface re-interpolation fails
pub fn offset_face(
    topo: &mut Topology,
    face_id: FaceId,
    distance: f64,
    samples: usize,
) -> Result<FaceId, OperationsError> {
    let tol = Tolerance::new();
    if distance.abs() < tol.linear {
        // Zero offset: just copy the face.
        return copy_face(topo, face_id);
    }

    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    match surface {
        FaceSurface::Plane { normal, d } => {
            offset_planar_face(topo, outer_wire, &inner_wires, normal, d, distance)
        }
        FaceSurface::Nurbs(ref nurbs) => offset_nurbs_face(topo, face_id, nurbs, distance, samples),
        FaceSurface::Cylinder(ref cyl) => {
            offset_cylinder_face(topo, outer_wire, &inner_wires, cyl, distance)
        }
        FaceSurface::Cone(ref cone) => {
            offset_cone_face(topo, outer_wire, &inner_wires, cone, distance)
        }
        FaceSurface::Sphere(ref sphere) => {
            offset_sphere_face(topo, outer_wire, &inner_wires, sphere, distance)
        }
        FaceSurface::Torus(ref torus) => {
            offset_torus_face(topo, outer_wire, &inner_wires, torus, distance)
        }
    }
}

/// Offset a planar face: shift plane along its normal.
fn offset_planar_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    normal: Vec3,
    d: f64,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    // New plane: same normal, shifted d.
    let new_d = d + distance;

    // Offset all vertices in the wires.
    let offset_vec = Vec3::new(
        normal.x() * distance,
        normal.y() * distance,
        normal.z() * distance,
    );

    let new_outer = offset_wire_vertices(topo, outer_wire, offset_vec)?;

    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        let new_iw = offset_wire_vertices(topo, iw, offset_vec)?;
        new_inner.push(new_iw);
    }

    let new_surface = FaceSurface::Plane { normal, d: new_d };
    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        new_surface,
    ));
    Ok(face_id)
}

/// Offset a NURBS face by sampling and refitting with adaptive refinement.
///
/// Uses a two-pass approach:
/// 1. Coarse grid sampling with curvature estimation at each point
/// 2. Adaptive refinement: regions where `curvature × |distance|` is large
///    (indicating potential cusps in the offset surface) get denser sampling
///
/// This prevents the previous uniform-grid approach from missing cusps
/// at high-curvature regions of the input surface.
#[allow(clippy::too_many_lines)]
fn offset_nurbs_face(
    topo: &mut Topology,
    face_id: FaceId,
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    distance: f64,
    samples: usize,
) -> Result<FaceId, OperationsError> {
    let n = samples.max(4); // Minimum 4×4 grid.

    // Pass 1: Evaluate curvature at a coarse grid to identify high-curvature regions.
    let coarse = n.max(4);
    #[allow(clippy::cast_precision_loss)]
    let coarse_div = (coarse - 1) as f64;
    let mut max_curvature = 0.0_f64;
    let mut curvatures: Vec<Vec<f64>> = Vec::with_capacity(coarse);

    #[allow(clippy::cast_precision_loss)]
    for i in 0..coarse {
        let u = i as f64 / coarse_div;
        let mut row = Vec::with_capacity(coarse);
        for j in 0..coarse {
            let v = j as f64 / coarse_div;
            let kappa = estimate_curvature(nurbs, u, v);
            max_curvature = max_curvature.max(kappa);
            row.push(kappa);
        }
        curvatures.push(row);
    }

    // Pass 2: Build adaptive parameter grid. For each coarse cell, decide
    // the local subdivision level based on curvature × |distance|.
    // High curvature × distance → more samples (up to 4× coarse density).
    let threshold = max_curvature * distance.abs() * 0.25;
    let mut u_params: Vec<f64> = Vec::new();
    let mut v_params: Vec<f64> = Vec::new();

    // Collect u-parameters: for each coarse interval, subdivide if needed.
    #[allow(clippy::cast_precision_loss)]
    for i in 0..coarse {
        let u0 = i as f64 / coarse_div;
        u_params.push(u0);

        if i + 1 < coarse {
            // Check if any cell in this u-row has high curvature.
            let row_max = curvatures[i].iter().copied().fold(0.0_f64, f64::max);
            let cell_metric = row_max * distance.abs();
            if threshold > 1e-12 && cell_metric > threshold {
                // Add midpoints for this interval.
                let u1 = (i + 1) as f64 / coarse_div;
                let mid = 0.5 * (u0 + u1);
                u_params.push(mid);
                // If very high curvature, add quarter-points too.
                if cell_metric > threshold * 3.0 {
                    u_params.push(0.25_f64.mul_add(u1 - u0, u0));
                    u_params.push(0.75_f64.mul_add(u1 - u0, u0));
                }
            }
        }
    }
    // Ensure endpoint.
    if u_params.last().is_none_or(|&u| (u - 1.0).abs() > 1e-15) {
        u_params.push(1.0);
    }

    // Collect v-parameters similarly.
    #[allow(clippy::cast_precision_loss)]
    for j in 0..coarse {
        let v0 = j as f64 / coarse_div;
        v_params.push(v0);

        if j + 1 < coarse {
            let col_max = curvatures.iter().map(|row| row[j]).fold(0.0_f64, f64::max);
            let cell_metric = col_max * distance.abs();
            if threshold > 1e-12 && cell_metric > threshold {
                let v1 = (j + 1) as f64 / coarse_div;
                let mid = 0.5 * (v0 + v1);
                v_params.push(mid);
                if cell_metric > threshold * 3.0 {
                    v_params.push(0.25_f64.mul_add(v1 - v0, v0));
                    v_params.push(0.75_f64.mul_add(v1 - v0, v0));
                }
            }
        }
    }
    if v_params.last().is_none_or(|&v| (v - 1.0).abs() > 1e-15) {
        v_params.push(1.0);
    }

    // Sort and deduplicate parameter lists.
    u_params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    u_params.dedup_by(|a, b| (*a - *b).abs() < 1e-15);
    v_params.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v_params.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    // Sample the surface and offset each point along the normal.
    let nu = u_params.len();
    let nv = v_params.len();
    let mut offset_grid: Vec<Vec<Point3>> = Vec::with_capacity(nu);

    for &u in &u_params {
        let mut row = Vec::with_capacity(nv);
        for &v in &v_params {
            let pt = nurbs.evaluate(u, v);
            let normal = nurbs
                .normal(u, v)
                .map_err(|e| OperationsError::InvalidInput {
                    reason: format!("NURBS normal computation failed at ({u}, {v}): {e}"),
                })?;

            let offset_pt = Point3::new(
                normal.x().mul_add(distance, pt.x()),
                normal.y().mul_add(distance, pt.y()),
                normal.z().mul_add(distance, pt.z()),
            );
            row.push(offset_pt);
        }
        offset_grid.push(row);
    }

    // Fit a new NURBS surface through the offset points.
    let degree = nurbs.degree_u().min(nurbs.degree_v()).min(3);
    let raw_offset = interpolate_surface(&offset_grid, degree, degree).map_err(|e| {
        OperationsError::InvalidInput {
            reason: format!("offset surface interpolation failed: {e}"),
        }
    })?;

    // Attempt self-intersection detection and trimming. If trimming fails,
    // fall back to the raw offset (best-effort improvement).
    let offset_surface =
        crate::offset_trim::trim_offset_self_intersections(nurbs, &raw_offset, distance, 1e-7)
            .unwrap_or(raw_offset);

    // Copy the wire topology from the original face, offsetting vertices.
    let face = topo.face(face_id)?;
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    // Offset wire vertices along the NURBS normal at their positions.
    let new_outer = offset_wire_along_nurbs(topo, outer_wire, nurbs, distance)?;

    let mut new_inner = Vec::new();
    for &iw in &inner_wires {
        let new_iw = offset_wire_along_nurbs(topo, iw, nurbs, distance)?;
        new_inner.push(new_iw);
    }

    let new_surface = FaceSurface::Nurbs(offset_surface);
    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        new_surface,
    ));
    Ok(face_id)
}

/// Offset a cylindrical face: create a new cylinder with adjusted radius.
///
/// Cylinder offset: radius R → R + distance. The surface type is preserved.
fn offset_cylinder_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    cyl: &brepkit_math::surfaces::CylindricalSurface,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    let new_radius = cyl.radius() + distance;
    if new_radius <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "cylinder offset by {distance} would produce negative radius \
                 (original radius = {})",
                cyl.radius()
            ),
        });
    }

    let new_cyl =
        brepkit_math::surfaces::CylindricalSurface::new(cyl.origin(), cyl.axis(), new_radius)
            .map_err(OperationsError::Math)?;

    // Offset wire vertices radially outward from the cylinder axis.
    let radial_offset = |pt: Point3| -> Point3 {
        let to_axis = Vec3::new(
            pt.x() - cyl.origin().x(),
            pt.y() - cyl.origin().y(),
            pt.z() - cyl.origin().z(),
        );
        // Project out the axis component to get the radial direction.
        let along_axis = cyl.axis() * cyl.axis().dot(to_axis);
        let radial = to_axis - along_axis;
        if let Ok(dir) = radial.normalize() {
            pt + dir * distance
        } else {
            pt // Point is on the axis; can't determine radial direction.
        }
    };

    let new_outer = offset_wire_by_fn(topo, outer_wire, &radial_offset)?;
    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        new_inner.push(offset_wire_by_fn(topo, iw, &radial_offset)?);
    }

    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        FaceSurface::Cylinder(new_cyl),
    ));
    Ok(face_id)
}

/// Offset a spherical face: create a new sphere with adjusted radius.
///
/// Sphere offset: radius R → R + distance. The surface type is preserved.
fn offset_sphere_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    sphere: &brepkit_math::surfaces::SphericalSurface,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    let new_radius = sphere.radius() + distance;
    if new_radius <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "sphere offset by {distance} would produce negative radius \
                 (original radius = {})",
                sphere.radius()
            ),
        });
    }

    let new_sphere = brepkit_math::surfaces::SphericalSurface::new(sphere.center(), new_radius)
        .map_err(OperationsError::Math)?;

    // Offset vertices radially from sphere center.
    let radial_offset = |pt: Point3| -> Point3 {
        let to_center = pt - sphere.center();
        if let Ok(dir) = to_center.normalize() {
            pt + dir * distance
        } else {
            pt
        }
    };

    let new_outer = offset_wire_by_fn(topo, outer_wire, &radial_offset)?;
    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        new_inner.push(offset_wire_by_fn(topo, iw, &radial_offset)?);
    }

    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        FaceSurface::Sphere(new_sphere),
    ));
    Ok(face_id)
}

/// Offset a conical face: shift apex along axis, preserve half-angle.
///
/// Cone offset: the apex moves along the axis by distance / sin(half_angle),
/// preserving the half-angle. The new cone has the same shape but is larger
/// (positive offset) or smaller (negative offset).
fn offset_cone_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    cone: &brepkit_math::surfaces::ConicalSurface,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    // The offset of a cone is another cone with the same half-angle.
    // The apex shifts along the axis by d / sin(α).
    let sin_ha = cone.half_angle().sin();
    if sin_ha.abs() < 1e-12 {
        return Err(OperationsError::InvalidInput {
            reason: "cone half-angle is degenerate (sin ≈ 0)".into(),
        });
    }

    let apex_shift = distance / sin_ha;
    let new_apex = cone.apex() + cone.axis() * apex_shift;

    let new_cone =
        brepkit_math::surfaces::ConicalSurface::new(new_apex, cone.axis(), cone.half_angle())
            .map_err(OperationsError::Math)?;

    // Offset wire vertices along the cone's radial direction.
    let radial_offset = |pt: Point3| -> Point3 {
        let to_apex = Vec3::new(
            pt.x() - cone.apex().x(),
            pt.y() - cone.apex().y(),
            pt.z() - cone.apex().z(),
        );
        let along_axis = cone.axis() * cone.axis().dot(to_apex);
        let radial = to_apex - along_axis;
        if let Ok(dir) = radial.normalize() {
            pt + dir * distance
        } else {
            pt
        }
    };

    let new_outer = offset_wire_by_fn(topo, outer_wire, &radial_offset)?;
    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        new_inner.push(offset_wire_by_fn(topo, iw, &radial_offset)?);
    }

    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        FaceSurface::Cone(new_cone),
    ));
    Ok(face_id)
}

/// Offset a toroidal face: adjust the minor radius.
///
/// Torus offset: minor radius r → r + distance. Major radius unchanged.
fn offset_torus_face(
    topo: &mut Topology,
    outer_wire: brepkit_topology::wire::WireId,
    inner_wires: &[brepkit_topology::wire::WireId],
    torus: &brepkit_math::surfaces::ToroidalSurface,
    distance: f64,
) -> Result<FaceId, OperationsError> {
    let new_minor = torus.minor_radius() + distance;
    if new_minor <= 0.0 {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "torus offset by {distance} would produce negative minor radius \
                 (original minor = {})",
                torus.minor_radius()
            ),
        });
    }

    let new_torus = brepkit_math::surfaces::ToroidalSurface::new(
        torus.center(),
        torus.major_radius(),
        new_minor,
    )
    .map_err(OperationsError::Math)?;

    // Offset wire vertices radially from the torus center, in the tube direction.
    let z_axis = torus.z_axis();
    let center = torus.center();
    let radial_offset = |pt: Point3| -> Point3 {
        let to_center = Vec3::new(
            pt.x() - center.x(),
            pt.y() - center.y(),
            pt.z() - center.z(),
        );
        // Project to the plane perpendicular to the torus axis.
        let in_plane = to_center - z_axis * z_axis.dot(to_center);
        // Direction from the tube center circle to the point.
        if let Ok(ring_dir) = in_plane.normalize() {
            let tube_center = center + ring_dir * torus.major_radius();
            let to_tube = Vec3::new(
                pt.x() - tube_center.x(),
                pt.y() - tube_center.y(),
                pt.z() - tube_center.z(),
            );
            if let Ok(tube_dir) = to_tube.normalize() {
                pt + tube_dir * distance
            } else {
                pt
            }
        } else {
            pt
        }
    };

    let new_outer = offset_wire_by_fn(topo, outer_wire, &radial_offset)?;
    let mut new_inner = Vec::new();
    for &iw in inner_wires {
        new_inner.push(offset_wire_by_fn(topo, iw, &radial_offset)?);
    }

    let face_id = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer,
        new_inner,
        FaceSurface::Torus(new_torus),
    ));
    Ok(face_id)
}

/// Estimate surface curvature at parameter (u, v) using second derivatives.
///
/// Returns the maximum absolute principal curvature. Uses the second
/// fundamental form eigenvalues: `κ = (L·N - M²) / (E·G - F²)` for
/// Gaussian curvature, and `(E·N - 2·F·M + G·L) / (2·(E·G - F²))` for
/// mean curvature. The max principal curvature is `|H| + sqrt(H² - K)`.
fn estimate_curvature(nurbs: &brepkit_math::nurbs::NurbsSurface, u: f64, v: f64) -> f64 {
    let d = nurbs.derivatives(u, v, 2);
    if d.len() < 3 || d[0].len() < 3 {
        return 0.0;
    }

    let su = d[1][0]; // ∂S/∂u
    let sv = d[0][1]; // ∂S/∂v
    let suu = d[2][0]; // ∂²S/∂u²
    let suv = d[1][1]; // ∂²S/∂u∂v
    let svv = d[0][2]; // ∂²S/∂v²

    let n_raw = su.cross(sv);
    let n_len = n_raw.length();
    if n_len < 1e-20 {
        return 0.0;
    }
    let n = n_raw * (1.0 / n_len);

    // First fundamental form coefficients.
    let e_coeff = su.dot(su);
    let f_coeff = su.dot(sv);
    let g_coeff = sv.dot(sv);

    // Second fundamental form coefficients.
    let l_coeff = suu.dot(n);
    let m_coeff = suv.dot(n);
    let n_coeff = svv.dot(n);

    let denom = e_coeff * g_coeff - f_coeff * f_coeff;
    if denom.abs() < 1e-30 {
        return 0.0;
    }

    // Mean curvature H and Gaussian curvature K.
    let h = (e_coeff * n_coeff - 2.0 * f_coeff * m_coeff + g_coeff * l_coeff) / (2.0 * denom);
    let k = (l_coeff * n_coeff - m_coeff * m_coeff) / denom;

    // Max principal curvature: |H| + sqrt(max(H² - K, 0))
    let disc = (h * h - k).max(0.0).sqrt();
    (h.abs() + disc).abs()
}

/// Offset all vertices in a wire using a position-dependent function.
fn offset_wire_by_fn(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    offset_fn: &dyn Fn(Point3) -> Point3,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot then allocate.
    let mut snaps: Vec<(Point3, Point3, EdgeCurve, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start_pt = topo.vertex(edge.start())?.point();
        let end_pt = topo.vertex(edge.end())?.point();
        snaps.push((start_pt, end_pt, edge.curve().clone(), oe.is_forward()));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, end_pt, curve, forward) in snaps {
        let new_start = topo.vertices.alloc(Vertex::new(offset_fn(start_pt), 1e-7));
        let new_end = topo.vertices.alloc(Vertex::new(offset_fn(end_pt), 1e-7));
        let new_edge = topo.edges.alloc(Edge::new(new_start, new_end, curve));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Copy a face with new IDs.
fn copy_face(topo: &mut Topology, face_id: FaceId) -> Result<FaceId, OperationsError> {
    let face = topo.face(face_id)?;
    let surface = face.surface().clone();
    let outer_wire = face.outer_wire();
    let inner_wires: Vec<_> = face.inner_wires().to_vec();

    let new_outer = copy_wire(topo, outer_wire)?;
    let mut new_inner = Vec::new();
    for &iw in &inner_wires {
        new_inner.push(copy_wire(topo, iw)?);
    }

    let new_face = topo.faces.alloc(brepkit_topology::face::Face::new(
        new_outer, new_inner, surface,
    ));
    Ok(new_face)
}

/// Copy a wire with new vertex and edge IDs.
fn copy_wire(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::Edge;
    use brepkit_topology::edge::EdgeCurve;
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot edge data before allocating (borrow checker).
    let mut edge_snaps: Vec<(Point3, f64, Point3, f64, EdgeCurve, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start = topo.vertex(edge.start())?;
        let end = topo.vertex(edge.end())?;
        edge_snaps.push((
            start.point(),
            start.tolerance(),
            end.point(),
            end.tolerance(),
            edge.curve().clone(),
            oe.is_forward(),
        ));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, start_tol, end_pt, end_tol, curve, forward) in edge_snaps {
        let new_start = topo.vertices.alloc(Vertex::new(start_pt, start_tol));
        let new_end = topo.vertices.alloc(Vertex::new(end_pt, end_tol));
        let new_edge = topo.edges.alloc(Edge::new(new_start, new_end, curve));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset all vertices in a wire by a constant vector.
fn offset_wire_vertices(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    offset: Vec3,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot then allocate.
    let mut edge_snaps: Vec<(Point3, Point3, EdgeCurve, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start_pt = topo.vertex(edge.start())?.point();
        let end_pt = topo.vertex(edge.end())?.point();
        edge_snaps.push((start_pt, end_pt, edge.curve().clone(), oe.is_forward()));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, end_pt, curve, forward) in edge_snaps {
        let new_start = topo.vertices.alloc(Vertex::new(start_pt + offset, 1e-7));
        let new_end = topo.vertices.alloc(Vertex::new(end_pt + offset, 1e-7));
        let new_edge = topo.edges.alloc(Edge::new(new_start, new_end, curve));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset wire vertices along the NURBS surface normal at their closest
/// parametric position.
fn offset_wire_along_nurbs(
    topo: &mut Topology,
    wire_id: brepkit_topology::wire::WireId,
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    distance: f64,
) -> Result<brepkit_topology::wire::WireId, OperationsError> {
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    let wire = topo.wire(wire_id)?;
    let edges = wire.edges().to_vec();

    // Snapshot then compute offsets.
    let mut snaps: Vec<(Point3, Point3, bool)> = Vec::new();
    for oe in &edges {
        let edge = topo.edge(oe.edge())?;
        let start_pt = topo.vertex(edge.start())?.point();
        let end_pt = topo.vertex(edge.end())?.point();
        snaps.push((start_pt, end_pt, oe.is_forward()));
    }

    let mut new_oriented = Vec::new();
    for (start_pt, end_pt, forward) in snaps {
        let new_start_pt = offset_point_on_surface(nurbs, start_pt, distance)?;
        let new_end_pt = offset_point_on_surface(nurbs, end_pt, distance)?;

        let new_start = topo.vertices.alloc(Vertex::new(new_start_pt, 1e-7));
        let new_end = topo.vertices.alloc(Vertex::new(new_end_pt, 1e-7));
        let new_edge = topo
            .edges
            .alloc(Edge::new(new_start, new_end, EdgeCurve::Line));
        new_oriented.push(OrientedEdge::new(new_edge, forward));
    }

    let new_wire = topo.wires.alloc(Wire::new(new_oriented, true)?);
    Ok(new_wire)
}

/// Offset a single point along the surface normal at its closest parametric
/// position on the NURBS surface.
fn offset_point_on_surface(
    nurbs: &brepkit_math::nurbs::NurbsSurface,
    point: Point3,
    distance: f64,
) -> Result<Point3, OperationsError> {
    use brepkit_math::nurbs::projection::project_point_to_surface;

    let proj = project_point_to_surface(nurbs, point, 1e-7).map_err(|e| {
        OperationsError::InvalidInput {
            reason: format!("surface projection failed: {e}"),
        }
    })?;
    let u = proj.u;
    let v = proj.v;
    let normal = nurbs
        .normal(u, v)
        .map_err(|e| OperationsError::InvalidInput {
            reason: format!("NURBS normal at ({u}, {v}) failed: {e}"),
        })?;

    Ok(Point3::new(
        normal.x().mul_add(distance, point.x()),
        normal.y().mul_add(distance, point.y()),
        normal.z().mul_add(distance, point.z()),
    ))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_square_face;

    use super::*;

    #[test]
    fn offset_planar_face_outward() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 1.0, 10).unwrap();

        // The offset face should have a shifted plane.
        let offset_face = topo.face(offset).unwrap();
        match offset_face.surface() {
            FaceSurface::Plane { normal, d } => {
                // Unit square face is on z=0, normal=(0,0,1), d=0.
                // Offset by 1.0 should give d=1.0.
                assert!((normal.z() - 1.0).abs() < 1e-6);
                assert!((d - 1.0).abs() < 1e-6);
            }
            _ => panic!("expected planar surface"),
        }
    }

    #[test]
    fn offset_planar_face_inward() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, -0.5, 10).unwrap();

        let offset_face = topo.face(offset).unwrap();
        match offset_face.surface() {
            FaceSurface::Plane { d, .. } => {
                assert!((d - (-0.5)).abs() < 1e-6);
            }
            _ => panic!("expected planar surface"),
        }
    }

    #[test]
    fn offset_zero_returns_copy() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 0.0, 10).unwrap();

        // Should return a different face ID (it's a copy).
        assert_ne!(face, offset);

        // But same surface properties.
        let original = topo.face(face).unwrap();
        let copied = topo.face(offset).unwrap();
        match (original.surface(), copied.surface()) {
            (
                FaceSurface::Plane { normal: n1, d: d1 },
                FaceSurface::Plane { normal: n2, d: d2 },
            ) => {
                assert!((n1.x() - n2.x()).abs() < 1e-10);
                assert!((n1.y() - n2.y()).abs() < 1e-10);
                assert!((n1.z() - n2.z()).abs() < 1e-10);
                assert!((d1 - d2).abs() < 1e-10);
            }
            _ => panic!("expected both planar"),
        }
    }

    #[test]
    fn offset_face_preserves_vertex_count() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 2.0, 10).unwrap();

        // Count vertices in original vs offset.
        let orig_face = topo.face(face).unwrap();
        let offset_face = topo.face(offset).unwrap();

        let orig_wire = topo.wire(orig_face.outer_wire()).unwrap();
        let off_wire = topo.wire(offset_face.outer_wire()).unwrap();

        assert_eq!(orig_wire.edges().len(), off_wire.edges().len());
    }

    #[test]
    fn offset_vertices_are_shifted() {
        let mut topo = Topology::new();
        let face = make_unit_square_face(&mut topo);

        let offset = offset_face(&mut topo, face, 3.0, 10).unwrap();

        // Get a vertex from the offset face and check it's shifted.
        let off_face = topo.face(offset).unwrap();
        let off_wire = topo.wire(off_face.outer_wire()).unwrap();
        let first_edge = off_wire.edges()[0];
        let edge = topo.edge(first_edge.edge()).unwrap();
        let vert = topo.vertex(edge.start()).unwrap();

        // Original unit square has vertices at z=0.
        // Offset of 3.0 along +Z normal should give z=3.0.
        assert!(
            (vert.point().z() - 3.0).abs() < 1e-6,
            "expected z=3.0, got z={}",
            vert.point().z()
        );
    }

    // ── NURBS face helpers ────────────────────────────────────────────────────

    /// Build a flat bilinear NURBS face on the XY plane (z = `z_height`).
    ///
    /// Degree-1 in both u and v, 2×2 control points, clamped knot vectors.
    /// The wire has 4 line-edges forming a unit square at `z_height`.
    fn make_flat_nurbs_face(topo: &mut Topology, z_height: f64) -> FaceId {
        use brepkit_math::nurbs::NurbsSurface;
        use brepkit_math::vec::Point3 as P;
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        // 2×2 control-point grid spanning [0,1]×[0,1] at given z.
        let ctrl = vec![
            vec![P::new(0.0, 0.0, z_height), P::new(1.0, 0.0, z_height)],
            vec![P::new(0.0, 1.0, z_height), P::new(1.0, 1.0, z_height)],
        ];
        let weights = vec![vec![1.0_f64, 1.0], vec![1.0, 1.0]];
        // Clamped knot vector for degree 1, 2 control points: [0,0,1,1]
        let knots = vec![0.0, 0.0, 1.0, 1.0];
        let nurbs = NurbsSurface::new(1, 1, knots.clone(), knots, ctrl, weights).unwrap();

        // Wire: unit square in XY at z_height.
        let tol = 1e-7;
        let v0 = topo
            .vertices
            .alloc(Vertex::new(P::new(0.0, 0.0, z_height), tol));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(P::new(1.0, 0.0, z_height), tol));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(P::new(1.0, 1.0, z_height), tol));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(P::new(0.0, 1.0, z_height), tol));

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

        topo.faces
            .alloc(Face::new(wid, vec![], FaceSurface::Nurbs(nurbs)))
    }

    // ── NURBS face offset tests ───────────────────────────────────────────────

    #[test]
    fn offset_nurbs_face_outward_produces_nurbs_surface() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 1.0, 6).unwrap();

        // The result must carry a NURBS surface.
        let off_face = topo.face(offset_id).unwrap();
        assert!(
            matches!(off_face.surface(), FaceSurface::Nurbs(_)),
            "expected NURBS surface after NURBS offset"
        );
    }

    #[test]
    fn offset_nurbs_face_new_id_differs_from_original() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 0.5, 6).unwrap();

        assert_ne!(face, offset_id, "offset should return a new face ID");
    }

    #[test]
    fn offset_nurbs_face_wire_has_same_edge_count() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, 1.0, 6).unwrap();

        let orig_wire = topo.wire(topo.face(face).unwrap().outer_wire()).unwrap();
        let off_wire = topo
            .wire(topo.face(offset_id).unwrap().outer_wire())
            .unwrap();
        assert_eq!(
            orig_wire.edges().len(),
            off_wire.edges().len(),
            "offset wire should have the same edge count as the original"
        );
    }

    #[test]
    fn offset_nurbs_face_negative_distance() {
        let mut topo = Topology::new();
        // Place the surface at z=2 so a negative offset moves it toward z=1.
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        let offset_id = offset_face(&mut topo, face, -0.5, 6).unwrap();

        // Result is still a valid NURBS face.
        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_very_small_distance() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // A distance well above the linear tolerance (1e-7) but very small.
        let offset_id = offset_face(&mut topo, face, 1e-4, 6).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_zero_returns_copy() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // Exactly zero: should go through the copy_face path, keeping NURBS.
        let copy_id = offset_face(&mut topo, face, 0.0, 6).unwrap();

        assert_ne!(face, copy_id);
        let copy_face = topo.face(copy_id).unwrap();
        assert!(
            matches!(copy_face.surface(), FaceSurface::Nurbs(_)),
            "zero offset of NURBS face should still be NURBS"
        );
    }

    #[test]
    fn offset_cylinder_face_preserves_type() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3 as P, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        let tol = 1e-7;
        let v0 = topo.vertices.alloc(Vertex::new(P::new(1.0, 0.0, 0.0), tol));
        let v1 = topo.vertices.alloc(Vertex::new(P::new(0.0, 1.0, 0.0), tol));
        let v2 = topo.vertices.alloc(Vertex::new(P::new(0.0, 1.0, 1.0), tol));
        let v3 = topo.vertices.alloc(Vertex::new(P::new(1.0, 0.0, 1.0), tol));
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
        let cyl =
            CylindricalSurface::new(P::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        let face_id = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));

        // Offset should succeed and produce a cylinder with larger radius.
        let result = offset_face(&mut topo, face_id, 0.5, 6).unwrap();
        let off_face = topo.face(result).unwrap();
        match off_face.surface() {
            FaceSurface::Cylinder(cyl) => {
                assert!(
                    (cyl.radius() - 1.5).abs() < 1e-10,
                    "offset cylinder radius should be 1.5, got {}",
                    cyl.radius()
                );
            }
            _ => panic!("expected cylinder surface after offset"),
        }
    }

    #[test]
    fn offset_cylinder_negative_radius_error() {
        use brepkit_math::surfaces::CylindricalSurface;
        use brepkit_math::vec::{Point3 as P, Vec3};
        use brepkit_topology::edge::{Edge, EdgeCurve};
        use brepkit_topology::face::Face;
        use brepkit_topology::vertex::Vertex;
        use brepkit_topology::wire::{OrientedEdge, Wire};

        let mut topo = Topology::new();

        let tol = 1e-7;
        let v0 = topo.vertices.alloc(Vertex::new(P::new(0.5, 0.0, 0.0), tol));
        let v1 = topo.vertices.alloc(Vertex::new(P::new(0.0, 0.5, 0.0), tol));
        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v0, EdgeCurve::Line));
        let wire = Wire::new(
            vec![OrientedEdge::new(e0, true), OrientedEdge::new(e1, true)],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);
        let cyl =
            CylindricalSurface::new(P::new(0.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 1.0), 0.5).unwrap();
        let face_id = topo
            .faces
            .alloc(Face::new(wid, vec![], FaceSurface::Cylinder(cyl)));

        // Offset by -0.6 would produce negative radius.
        let result = offset_face(&mut topo, face_id, -0.6, 6);
        assert!(
            result.is_err(),
            "negative-radius cylinder offset should fail"
        );
    }

    #[test]
    fn offset_nurbs_face_large_distance() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // A large positive offset should still succeed (the surface is flat so
        // normals are uniform and well-defined everywhere).
        let offset_id = offset_face(&mut topo, face, 100.0, 8).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }

    #[test]
    fn offset_nurbs_face_minimum_samples_clamped() {
        let mut topo = Topology::new();
        let face = make_flat_nurbs_face(&mut topo, 0.0);

        // Passing samples=1 should be clamped to 4 internally and still succeed.
        let offset_id = offset_face(&mut topo, face, 1.0, 1).unwrap();

        let off_face = topo.face(offset_id).unwrap();
        assert!(matches!(off_face.surface(), FaceSurface::Nurbs(_)));
    }
}

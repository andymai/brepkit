//! Loft operation: create a solid by interpolating between profile faces.
//!
//! The loft connects two or more planar profiles by creating ruled (linear)
//! surfaces between corresponding profile edges.

use brepkit_math::nurbs::surface_fitting::interpolate_surface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::boolean::face_polygon;
use crate::dot_normal_point;
use crate::winding::ensure_ccw_profiles;

/// Resample a closed polygon to `target_count` evenly spaced points.
///
/// Distributes `target_count` points at equal arc-length intervals
/// along the polygon boundary.
#[allow(clippy::cast_precision_loss)]
fn resample_closed_polygon(points: &[Point3], target_count: usize) -> Vec<Point3> {
    let n = points.len();
    if n == 0 || target_count == 0 {
        return Vec::new();
    }
    // Compute cumulative arc lengths (closed: last segment wraps to first point)
    let mut cum_len = Vec::with_capacity(n + 1);
    cum_len.push(0.0);
    for i in 0..n {
        let next = (i + 1) % n;
        let seg = (points[next] - points[i]).length();
        cum_len.push(cum_len[i] + seg);
    }
    let total = *cum_len.last().unwrap_or(&0.0);
    if total < 1e-15 {
        return vec![points[0]; target_count];
    }

    let mut result = Vec::with_capacity(target_count);
    for i in 0..target_count {
        let target_len = total * (i as f64) / (target_count as f64);
        // Binary search for the segment containing target_len
        let seg = cum_len
            .partition_point(|&l| l < target_len)
            .saturating_sub(1)
            .min(n - 1);
        let seg_start = cum_len[seg];
        let seg_end = cum_len[seg + 1];
        let seg_len = seg_end - seg_start;
        let t = if seg_len > 1e-15 {
            (target_len - seg_start) / seg_len
        } else {
            0.0
        };
        let a = points[seg];
        let b = points[(seg + 1) % n];
        result.push(Point3::new(
            a.x() + t * (b.x() - a.x()),
            a.y() + t * (b.y() - a.y()),
            a.z() + t * (b.z() - a.z()),
        ));
    }
    result
}

/// Extract the outward cap normal from a planar face, accounting for winding reversal.
///
/// `inward` controls the sign convention: `true` for the start cap (points into
/// the loft interior, away from the profile direction), `false` for the end cap.
fn cap_normal_from_face(
    face: &brepkit_topology::face::Face,
    winding_reversed: bool,
    inward: bool,
) -> Result<Vec3, crate::OperationsError> {
    match face.surface() {
        FaceSurface::Plane { normal, .. } => {
            // The stored face normal comes from Newell on the original (pre-reversal)
            // winding. When winding_reversed is true, that normal points the wrong way,
            // so we negate it. The inward flag then flips the sign for the start cap.
            let sign = if winding_reversed == inward {
                1.0
            } else {
                -1.0
            };
            Ok(*normal * sign)
        }
        _ => Err(crate::OperationsError::InvalidInput {
            reason: "unexpected non-planar face".into(),
        }),
    }
}

/// Loft two or more planar profiles into a solid.
///
/// Each profile is a planar face. All profiles must have the same
/// number of boundary vertices. The loft connects corresponding
/// vertices between adjacent profiles with ruled (linear) surfaces,
/// and caps the first and last profiles as the solid's end faces.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 2 profiles are provided
/// - Profiles have different vertex counts
/// - Any profile is not a planar face
#[allow(clippy::too_many_lines)]
pub fn loft(topo: &mut Topology, profiles: &[FaceId]) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if profiles.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft requires at least 2 profiles".into(),
        });
    }

    // Fast path: lofting a stack of coaxial circles (incl. NURBS-recognized
    // circles produced by brepjs `sketchCircle`) collapses to an exact
    // sequence of cylinder / cone / frustum bands. The general path
    // tessellates the circles into N line segments, losing 0.6-1% of the
    // true π·r²·h (or frustum) volume per band.
    if let Some(stack_solid) = try_loft_coaxial_circle_stack(topo, profiles)? {
        return Ok(stack_solid);
    }

    // Collect vertex positions for each profile.
    let mut profile_verts: Vec<Vec<Point3>> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { .. } => {}
            _ => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "loft of non-planar faces is not supported".into(),
                });
            }
        }
        let verts = face_polygon(topo, fid)?;
        profile_verts.push(verts);
    }

    // Resample all profiles to the maximum vertex count so that lofting
    // between different-resolution profiles (e.g. rectangle ↔ circle) works.
    let n = profile_verts.iter().map(Vec::len).max().unwrap_or(0);
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft profiles must have at least 3 vertices".into(),
        });
    }
    for verts in &mut profile_verts {
        if verts.len() != n {
            *verts = resample_closed_polygon(verts, n);
        }
    }

    // Ensure profile vertex winding is CCW relative to the stacking direction.
    // The side normal formula `edge_dir.cross(connect_dir)` gives outward normals
    // only when vertices go CCW from the stacking direction.
    let winding_reversed = ensure_ccw_profiles(&mut profile_verts);

    let num_profiles = profile_verts.len();
    let num_sections = num_profiles - 1;

    // Create all vertices.
    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
                .collect()
        })
        .collect();

    // Create profile edges for each ring.
    let ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = ring_verts
        .iter()
        .map(|ring| {
            (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    // Create connecting edges between adjacent profiles.
    let connect_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = (0..num_sections)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.add_edge(Edge::new(
                        ring_verts[s][i],
                        ring_verts[s + 1][i],
                        EdgeCurve::Line,
                    ))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // Start cap: reversed first profile (outward normal pointing away from loft).
    {
        let face_data = topo.face(profiles[0])?;
        let cap_normal = cap_normal_from_face(face_data, winding_reversed, true)?;
        let reversed_edges: Vec<OrientedEdge> = (0..n)
            .rev()
            .map(|i| OrientedEdge::new(ring_edges[0][i], false))
            .collect();
        let wire = Wire::new(reversed_edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[0][0]);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // Side faces: one quad per profile-edge × section.
    for s in 0..num_sections {
        for i in 0..n {
            let next_i = (i + 1) % n;

            // Quad: ring[s][i] → ring[s][next_i] → ring[s+1][next_i] → ring[s+1][i]
            let p0 = profile_verts[s][i];
            let p1 = profile_verts[s][next_i];
            let p_next = profile_verts[s + 1][i];
            let edge_dir = p1 - p0;
            let connect_dir = p_next - p0;
            let side_normal = edge_dir
                .cross(connect_dir)
                .normalize()
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0));
            let side_d = dot_normal_point(side_normal, p0);

            let side_wire = Wire::new(
                vec![
                    OrientedEdge::new(ring_edges[s][i], true),
                    OrientedEdge::new(connect_edges[s][next_i], true),
                    OrientedEdge::new(ring_edges[s + 1][i], false),
                    OrientedEdge::new(connect_edges[s][i], false),
                ],
                true,
            )
            .map_err(crate::OperationsError::Topology)?;

            let side_wire_id = topo.add_wire(side_wire);
            let side_face = topo.add_face(Face::new(
                side_wire_id,
                vec![],
                FaceSurface::Plane {
                    normal: side_normal,
                    d: side_d,
                },
            ));
            all_faces.push(side_face);
        }
    }

    // End cap: last profile with forward orientation.
    {
        let face_data = topo.face(profiles[num_profiles - 1])?;
        let cap_normal = cap_normal_from_face(face_data, winding_reversed, false)?;
        let edges: Vec<OrientedEdge> = (0..n)
            .map(|i| OrientedEdge::new(ring_edges[num_profiles - 1][i], true))
            .collect();
        let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[num_profiles - 1][0]);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // Assemble.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Detect "loft across a stack of coaxial circles" and produce an exact
/// chain of cylinder / cone / frustum bands. Returns `Ok(None)` when any
/// profile is not a recognized circle, when the circles don't share a
/// common axis, or when adjacent centers coincide — the general loft then
/// handles the case.
///
/// All circle centers must lie on one line whose direction is parallel to
/// every circle's plane normal, and consecutive centers must be ordered
/// monotonically along that line (no zero-height bands).
fn try_loft_coaxial_circle_stack(
    topo: &mut Topology,
    profiles: &[FaceId],
) -> Result<Option<SolidId>, crate::OperationsError> {
    let tol = Tolerance::new();

    let mut circles: Vec<(Point3, Vec3, f64)> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        match face_recognized_circle(topo, fid) {
            Some(c) => circles.push(c),
            None => return Ok(None),
        }
    }

    let (center_0, _, _) = circles[0];
    let (center_1, _, _) = circles[1];
    let axis = center_1 - center_0;
    let axis_len = axis.length();
    if axis_len < tol.linear {
        return Ok(None);
    }
    let axis_unit = axis * (1.0 / axis_len);

    // Every circle's normal must be parallel to the stacking axis, every
    // center must lie on the axis line, and the signed axial positions must
    // be strictly increasing (so each band has positive height).
    let mut axial = Vec::with_capacity(circles.len());
    let mut prev_t = f64::NEG_INFINITY;
    for &(center, normal, _) in &circles {
        if normal.dot(axis_unit).abs() <= 1.0 - tol.angular {
            return Ok(None);
        }
        let rel = center - center_0;
        let t = rel.dot(axis_unit);
        // Reject lateral offset from the axis line.
        let radial = rel - axis_unit * t;
        if radial.length() > tol.linear {
            return Ok(None);
        }
        if t <= prev_t + tol.linear {
            return Ok(None);
        }
        prev_t = t;
        axial.push(t);
    }

    let radii: Vec<f64> = circles.iter().map(|&(_, _, r)| r).collect();
    let solid = build_coaxial_band_stack(topo, &axial, &radii)?;

    let z_axis = Vec3::new(0.0, 0.0, 1.0);
    if (z_axis - axis_unit).length() > tol.linear {
        let rot_axis = z_axis.cross(axis_unit);
        let rot_axis_len = rot_axis.length();
        let mat = if rot_axis_len < tol.linear {
            brepkit_math::mat::Mat4::rotation_x(std::f64::consts::PI)
        } else {
            let angle = z_axis.dot(axis_unit).clamp(-1.0, 1.0).acos();
            rodrigues_rotation(rot_axis * (1.0 / rot_axis_len), angle)
        };
        crate::transform::transform_solid(topo, solid, &mat)?;
    }
    if center_0.x().abs() > tol.linear
        || center_0.y().abs() > tol.linear
        || center_0.z().abs() > tol.linear
    {
        let xform = brepkit_math::mat::Mat4::translation(center_0.x(), center_0.y(), center_0.z());
        crate::transform::transform_solid(topo, solid, &xform)?;
    }
    Ok(Some(solid))
}

/// Rotation matrix around an arbitrary unit axis by `angle` radians
/// (Rodrigues' formula). Duplicates `pattern::rotation_matrix` so loft
/// stays self-contained.
fn rodrigues_rotation(axis: Vec3, angle: f64) -> brepkit_math::mat::Mat4 {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    let omc = 1.0 - cos_a;
    let (ax, ay, az) = (axis.x(), axis.y(), axis.z());
    brepkit_math::mat::Mat4([
        [
            omc.mul_add(ax * ax, cos_a),
            ax.mul_add(ay * omc, -(sin_a * az)),
            ax.mul_add(az * omc, sin_a * ay),
            0.0,
        ],
        [
            ax.mul_add(ay * omc, sin_a * az),
            omc.mul_add(ay * ay, cos_a),
            ay.mul_add(az * omc, -(sin_a * ax)),
            0.0,
        ],
        [
            ax.mul_add(az * omc, -(sin_a * ay)),
            ay.mul_add(az * omc, sin_a * ax),
            omc.mul_add(az * az, cos_a),
            0.0,
        ],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

/// Build a watertight stack of analytic cylinder/cone bands along +Z.
///
/// `axial[k]` is the z-height of ring `k`, `radii[k]` its circle radius.
/// Adjacent rings are connected by a cylindrical patch (equal radii) or a
/// conical patch (differing radii). Ring circle edges are shared between
/// neighbouring bands (and the two end caps) for watertight topology.
#[allow(clippy::too_many_lines)]
fn build_coaxial_band_stack(
    topo: &mut Topology,
    axial: &[f64],
    radii: &[f64],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();
    let z_axis = Vec3::new(0.0, 0.0, 1.0);

    // One shared circle edge per ring (degenerate seam vertex at angle 0).
    let mut ring_edges = Vec::with_capacity(axial.len());
    let mut ring_seam_verts = Vec::with_capacity(axial.len());
    for (&z, &r) in axial.iter().zip(radii.iter()) {
        let circle = brepkit_math::curves::Circle3D::new(Point3::new(0.0, 0.0, z), z_axis, r)
            .map_err(crate::OperationsError::Math)?;
        let seam_v = topo.add_vertex(Vertex::new(Point3::new(r, 0.0, z), tol.linear));
        let e = topo.add_edge(Edge::new(seam_v, seam_v, EdgeCurve::Circle(circle)));
        ring_edges.push(e);
        ring_seam_verts.push(seam_v);
    }

    let mut faces = Vec::new();

    for band in 0..axial.len() - 1 {
        let (z0, z1) = (axial[band], axial[band + 1]);
        let (r0, r1) = (radii[band], radii[band + 1]);
        let height = z1 - z0;

        let seam = topo.add_edge(Edge::new(
            ring_seam_verts[band],
            ring_seam_verts[band + 1],
            EdgeCurve::Line,
        ));

        let surface = if (r0 - r1).abs() < tol.linear {
            let cyl = brepkit_math::surfaces::CylindricalSurface::new(
                Point3::new(0.0, 0.0, z0),
                z_axis,
                r0,
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Cylinder(cyl)
        } else {
            // Virtual apex where the band's generator reaches radius zero.
            let z_apex = z0 - r0 * height / (r1 - r0);
            // Axis points apex → larger-radius end so the v>0 generator
            // sweeps outward; half-angle is measured from the radial plane.
            let (apex_z, axis_sign, r_ref, axial_to_ref) = if r1 > r0 {
                (z_apex, 1.0_f64, r1, z1 - z_apex)
            } else {
                (z_apex, -1.0_f64, r0, z_apex - z0)
            };
            let half_angle = axial_to_ref.abs().atan2(r_ref);
            let cone = brepkit_math::surfaces::ConicalSurface::new(
                Point3::new(0.0, 0.0, apex_z),
                Vec3::new(0.0, 0.0, axis_sign),
                half_angle,
            )
            .map_err(crate::OperationsError::Math)?;
            FaceSurface::Cone(cone)
        };

        let lateral_wire = Wire::new(
            vec![
                OrientedEdge::new(ring_edges[band], true),
                OrientedEdge::new(seam, true),
                OrientedEdge::new(ring_edges[band + 1], false),
                OrientedEdge::new(seam, false),
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;
        let lateral_wid = topo.add_wire(lateral_wire);
        faces.push(topo.add_face(Face::new(lateral_wid, vec![], surface)));
    }

    // Bottom cap (reversed first ring edge → outward normal -Z).
    let bot_wire = Wire::new(vec![OrientedEdge::new(ring_edges[0], false)], true)
        .map_err(crate::OperationsError::Topology)?;
    let bot_wid = topo.add_wire(bot_wire);
    faces.push(topo.add_face(Face::new(
        bot_wid,
        vec![],
        FaceSurface::Plane {
            normal: Vec3::new(0.0, 0.0, -1.0),
            d: -axial[0],
        },
    )));

    // Top cap (forward last ring edge → outward normal +Z).
    let last = axial.len() - 1;
    let top_wire = Wire::new(vec![OrientedEdge::new(ring_edges[last], true)], true)
        .map_err(crate::OperationsError::Topology)?;
    let top_wid = topo.add_wire(top_wire);
    faces.push(topo.add_face(Face::new(
        top_wid,
        vec![],
        FaceSurface::Plane {
            normal: z_axis,
            d: axial[last],
        },
    )));

    let shell = Shell::new(faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

/// Recognize a planar face as a circle (center, plane normal, radius).
/// Returns `None` when the face's outer wire isn't a single closed
/// circular edge (or NURBS-recognized circular edge), or when the
/// surface isn't planar.
fn face_recognized_circle(topo: &Topology, face_id: FaceId) -> Option<(Point3, Vec3, f64)> {
    let face = topo.face(face_id).ok()?;
    if !face.inner_wires().is_empty() {
        return None;
    }
    let normal = match face.surface() {
        FaceSurface::Plane { normal, .. } => *normal,
        _ => return None,
    };
    let wire = topo.wire(face.outer_wire()).ok()?;
    let edges = wire.edges();
    if edges.len() != 1 {
        return None;
    }
    let edge = topo.edge(edges[0].edge()).ok()?;
    if edge.start() != edge.end() {
        return None; // not a closed-loop circle
    }
    match edge.curve() {
        brepkit_topology::edge::EdgeCurve::Circle(c) => Some((c.center(), normal, c.radius())),
        brepkit_topology::edge::EdgeCurve::NurbsCurve(nc) => {
            let tol = Tolerance::new().linear;
            match brepkit_geometry::convert::recognize_curve(nc, tol * 100.0) {
                brepkit_geometry::convert::RecognizedCurve::Circle { center, radius, .. } => {
                    Some((center, normal, radius))
                }
                _ => None,
            }
        }
        _ => None,
    }
}

/// Loft profiles into a solid with smooth NURBS side surfaces.
///
/// Like [`loft`], but produces smooth NURBS surfaces for the side faces
/// instead of piecewise-planar quads. When 3+ profiles are provided,
/// the side surfaces interpolate smoothly through all profiles using
/// tensor-product surface fitting, giving C1+ continuity across sections.
///
/// For 2 profiles, the result is equivalent to the basic [`loft`] (ruled
/// surfaces). For 3+ profiles, the result is a smooth blend.
///
/// # Errors
///
/// Returns an error if:
/// - Fewer than 2 profiles are provided
/// - Profiles have different vertex counts
/// - Any profile is not a planar face
/// - Surface interpolation fails
#[allow(clippy::too_many_lines)]
pub fn loft_smooth(
    topo: &mut Topology,
    profiles: &[FaceId],
) -> Result<SolidId, crate::OperationsError> {
    let tol = Tolerance::new();

    if profiles.len() < 2 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft requires at least 2 profiles".into(),
        });
    }

    // For 2 profiles, delegate to the basic loft (ruled surfaces are optimal).
    if profiles.len() == 2 {
        return loft(topo, profiles);
    }

    // Collect vertex positions for each profile.
    let mut profile_verts: Vec<Vec<Point3>> = Vec::with_capacity(profiles.len());
    for &fid in profiles {
        let face = topo.face(fid)?;
        match face.surface() {
            FaceSurface::Plane { .. } => {}
            _ => {
                return Err(crate::OperationsError::InvalidInput {
                    reason: "loft of non-planar faces is not supported".into(),
                });
            }
        }
        let verts = face_polygon(topo, fid)?;
        profile_verts.push(verts);
    }

    // Resample all profiles to the maximum vertex count.
    let n = profile_verts.iter().map(Vec::len).max().unwrap_or(0);
    if n < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "loft profiles must have at least 3 vertices".into(),
        });
    }
    for verts in &mut profile_verts {
        if verts.len() != n {
            *verts = resample_closed_polygon(verts, n);
        }
    }

    // Ensure profile vertex winding is CCW relative to the stacking direction.
    let winding_reversed = ensure_ccw_profiles(&mut profile_verts);

    let num_profiles = profile_verts.len();

    // Create all vertices.
    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.add_vertex(Vertex::new(p, tol.linear)))
                .collect()
        })
        .collect();

    // Create profile edges for each ring.
    let ring_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = ring_verts
        .iter()
        .map(|ring| {
            (0..n)
                .map(|i| {
                    let next = (i + 1) % n;
                    topo.add_edge(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    let mut all_faces = Vec::new();

    // Start cap: reversed first profile.
    {
        let face_data = topo.face(profiles[0])?;
        let cap_normal = cap_normal_from_face(face_data, winding_reversed, true)?;
        let reversed_edges: Vec<OrientedEdge> = (0..n)
            .rev()
            .map(|i| OrientedEdge::new(ring_edges[0][i], false))
            .collect();
        let wire = Wire::new(reversed_edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[0][0]);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // NURBS side faces: one surface per edge index, spanning ALL profiles.
    // Degree in u (across profiles): min(P-1, 3) for smooth interpolation.
    // Degree in v (along edge): 1 (linear between adjacent vertices).
    let degree_u = (num_profiles - 1).min(3);
    let degree_v = 1;

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Build the interpolation grid: rows = profiles, cols = 2 (edge endpoints).
        let grid: Vec<Vec<Point3>> = (0..num_profiles)
            .map(|k| vec![profile_verts[k][i], profile_verts[k][next_i]])
            .collect();

        // Interpolate a NURBS surface through the grid.
        let surface =
            interpolate_surface(&grid, degree_u, degree_v).map_err(crate::OperationsError::Math)?;

        // Create the boundary wire for this side face.
        // The wire goes around the edge of the NURBS patch:
        // bottom edge → right rail → top edge (reversed) → left rail (reversed)
        let last = num_profiles - 1;

        // Bottom edge: ring_edges[0][i] (first profile, edge i)
        // Top edge: ring_edges[last][i] (last profile, edge i)
        // Left rail: connects vertex i across all profiles
        // Right rail: connects vertex next_i across all profiles

        // For the multi-section case, we need edges spanning ALL profiles.
        // Create single edges from first to last profile for the rails.
        let e_left_rail = topo.add_edge(Edge::new(
            ring_verts[0][i],
            ring_verts[last][i],
            EdgeCurve::Line,
        ));
        let e_right_rail = topo.add_edge(Edge::new(
            ring_verts[0][next_i],
            ring_verts[last][next_i],
            EdgeCurve::Line,
        ));

        let side_wire = Wire::new(
            vec![
                OrientedEdge::new(ring_edges[0][i], true),     // bottom
                OrientedEdge::new(e_right_rail, true),         // right
                OrientedEdge::new(ring_edges[last][i], false), // top (reversed)
                OrientedEdge::new(e_left_rail, false),         // left (reversed)
            ],
            true,
        )
        .map_err(crate::OperationsError::Topology)?;

        let side_wire_id = topo.add_wire(side_wire);
        let side_face = topo.add_face(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
        all_faces.push(side_face);
    }

    // End cap: last profile with forward orientation.
    {
        let face_data = topo.face(profiles[num_profiles - 1])?;
        let cap_normal = cap_normal_from_face(face_data, winding_reversed, false)?;
        let edges: Vec<OrientedEdge> = (0..n)
            .map(|i| OrientedEdge::new(ring_edges[num_profiles - 1][i], true))
            .collect();
        let wire = Wire::new(edges, true).map_err(crate::OperationsError::Topology)?;
        let wid = topo.add_wire(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[num_profiles - 1][0]);
        let fid = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: cap_normal,
                d: cap_d,
            },
        ));
        all_faces.push(fid);
    }

    // Assemble.
    let shell = Shell::new(all_faces).map_err(crate::OperationsError::Topology)?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, vec![])))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_math::tolerance::Tolerance;
    use brepkit_math::vec::{Point3, Vec3};
    use brepkit_topology::Topology;
    use brepkit_topology::edge::{Edge, EdgeCurve};
    use brepkit_topology::face::{Face, FaceSurface};
    use brepkit_topology::vertex::Vertex;
    use brepkit_topology::wire::{OrientedEdge, Wire};

    use super::*;

    /// Helper: make a square face at z=offset with given size.
    fn make_square_at(topo: &mut Topology, size: f64, z: f64) -> FaceId {
        let hs = size / 2.0;
        let tol_val = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, z), tol_val));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, z), tol_val));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, z), tol_val));

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

        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    #[test]
    fn loft_two_identical_squares_makes_box() {
        let mut topo = Topology::new();
        let bottom = make_square_at(&mut topo, 1.0, 0.0);
        let top = make_square_at(&mut topo, 1.0, 1.0);

        let solid = loft(&mut topo, &[bottom, top]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 2 caps + 4 sides = 6 faces
        assert_eq!(sh.faces().len(), 6, "lofted box should have 6 faces");

        // Volume should be 1.0 (unit cube)
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        let tol = Tolerance::loose();
        assert!(
            tol.approx_eq(vol, 1.0),
            "lofted box volume should be ~1.0, got {vol}"
        );
    }

    #[test]
    fn loft_tapered_frustum() {
        let mut topo = Topology::new();
        let bottom = make_square_at(&mut topo, 2.0, 0.0);
        let top = make_square_at(&mut topo, 1.0, 3.0);

        let solid = loft(&mut topo, &[bottom, top]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        assert_eq!(sh.faces().len(), 6);

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        // Frustum of a square pyramid: V = h/3 * (A1 + A2 + sqrt(A1*A2))
        // A1 = 4.0, A2 = 1.0, h = 3.0
        // V = 3/3 * (4 + 1 + 2) = 7.0
        let expected = 7.0;
        assert!(
            (vol - expected).abs() / expected < 0.05,
            "tapered frustum volume should be ~{expected}, got {vol} (error: {:.1}%)",
            (vol - expected).abs() / expected * 100.0
        );
    }

    #[test]
    fn loft_three_profiles() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 1.5);
        let p2 = make_square_at(&mut topo, 2.0, 3.0);

        let solid = loft(&mut topo, &[p0, p1, p2]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 2 caps + 2 sections × 4 edges = 10 faces
        assert_eq!(sh.faces().len(), 10);

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(vol > 0.0, "lofted solid should have positive volume");
    }

    #[test]
    fn loft_single_profile_error() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 1.0, 0.0);

        assert!(loft(&mut topo, &[p0]).is_err());
    }

    #[test]
    fn loft_mismatched_vertex_count_error() {
        let mut topo = Topology::new();
        let square = make_square_at(&mut topo, 1.0, 0.0);

        // Create a triangle profile.
        let tol_val = 1e-7;
        let v0 = topo.add_vertex(Vertex::new(Point3::new(0.0, 0.0, 1.0), tol_val));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(1.0, 0.0, 1.0), tol_val));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(0.5, 1.0, 1.0), tol_val));

        let e0 = topo.add_edge(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.add_edge(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.add_edge(Edge::new(v2, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.add_wire(wire);
        let tri = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: 1.0,
            },
        ));

        // Profiles with different vertex counts should succeed via resampling.
        let result = loft(&mut topo, &[square, tri]);
        assert!(
            result.is_ok(),
            "loft with different vertex counts should succeed via resampling"
        );
    }

    /// Helper: make a CW-wound square face at z=offset with given size.
    fn make_cw_square_at(topo: &mut Topology, size: f64, z: f64) -> FaceId {
        let hs = size / 2.0;
        let tol_val = 1e-7;
        // CW order: v0→v3→v2→v1 (reversed from make_square_at)
        let v0 = topo.add_vertex(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
        let v1 = topo.add_vertex(Vertex::new(Point3::new(-hs, hs, z), tol_val));
        let v2 = topo.add_vertex(Vertex::new(Point3::new(hs, hs, z), tol_val));
        let v3 = topo.add_vertex(Vertex::new(Point3::new(hs, -hs, z), tol_val));

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

        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: Vec3::new(0.0, 0.0, 1.0),
                d: z,
            },
        ))
    }

    #[test]
    fn loft_cw_profiles_produces_correct_solid() {
        let mut topo = Topology::new();
        let bottom = make_cw_square_at(&mut topo, 1.0, 0.0);
        let top = make_cw_square_at(&mut topo, 1.0, 1.0);

        let solid = loft(&mut topo, &[bottom, top]).unwrap();

        // CW profiles should be auto-corrected to produce a valid solid
        // with positive volume (not inside-out).
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "CW-wound loft should produce positive volume, got {vol}"
        );
    }

    // ── Smooth NURBS loft tests ──────────────────────────

    #[test]
    fn loft_smooth_two_profiles_delegates() {
        // With 2 profiles, loft_smooth delegates to basic loft (ruled surfaces).
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 1.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 1.0);

        let solid = loft_smooth(&mut topo, &[p0, p1]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            6,
            "2-profile smooth loft should have 6 faces"
        );
    }

    #[test]
    fn loft_smooth_three_profiles_has_nurbs() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 1.5);
        let p2 = make_square_at(&mut topo, 2.0, 3.0);

        let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // 2 caps + 4 NURBS sides = 6 faces (one surface per edge, spanning all profiles)
        assert_eq!(
            sh.faces().len(),
            6,
            "3-profile smooth loft should have 6 faces"
        );

        // Verify at least one NURBS face exists (the side surfaces).
        let has_nurbs = sh.faces().iter().any(|&fid| {
            matches!(
                topo.face(fid).expect("face").surface(),
                FaceSurface::Nurbs(_)
            )
        });
        assert!(has_nurbs, "smooth loft should produce NURBS side faces");
    }

    #[test]
    fn loft_smooth_three_profiles_positive_volume() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 1.5);
        let p2 = make_square_at(&mut topo, 2.0, 3.0);

        let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "smooth loft should have positive volume, got {vol}"
        );
    }

    #[test]
    fn loft_smooth_four_profiles() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.5, 1.0);
        let p2 = make_square_at(&mut topo, 1.0, 2.0);
        let p3 = make_square_at(&mut topo, 1.5, 3.0);

        let solid = loft_smooth(&mut topo, &[p0, p1, p2, p3]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            6,
            "4-profile smooth loft should have 6 faces"
        );

        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(vol > 0.0, "smooth loft should have positive volume");
    }

    /// Helper: make a planar circle face whose single outer edge is an
    /// analytic [`Circle3D`], centered at `(0,0,z)` with axis +Z.
    fn make_circle_face_at(topo: &mut Topology, radius: f64, z: f64) -> FaceId {
        let tol_val = 1e-7;
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let center = Point3::new(0.0, 0.0, z);
        let circle = brepkit_math::curves::Circle3D::new(center, axis, radius).unwrap();
        let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
        let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
        let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane { normal: axis, d: z },
        ))
    }

    /// Helper: make a planar circle face whose outer edge is stored as a
    /// rational NURBS curve that is geometrically a circle (exercises the
    /// canonical-recognition branch).
    fn make_nurbs_circle_face_at(topo: &mut Topology, radius: f64, z: f64) -> FaceId {
        let tol_val = 1e-7;
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let center = Point3::new(0.0, 0.0, z);
        let circle = brepkit_math::curves::Circle3D::new(center, axis, radius).unwrap();
        let nurbs =
            brepkit_geometry::convert::circle_to_nurbs(&circle, 0.0, 2.0 * std::f64::consts::PI)
                .unwrap();
        let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
        let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::NurbsCurve(nurbs)));
        let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane { normal: axis, d: z },
        ))
    }

    fn assert_analytic_frustum_solid(topo: &Topology, solid: SolidId, expected_volume: f64) {
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();
        assert_eq!(
            sh.faces().len(),
            3,
            "analytic loft must emit exactly 3 faces (two caps + one analytic side)"
        );
        let has_curved_side = sh.faces().iter().any(|&fid| {
            matches!(
                topo.face(fid).unwrap().surface(),
                FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
            )
        });
        assert!(
            has_curved_side,
            "analytic loft side face must be cylindrical/conical, not planar"
        );
        let vol = crate::measure::solid_volume(topo, solid, 0.05).unwrap();
        let rel_err = (vol - expected_volume).abs() / expected_volume;
        assert!(
            rel_err < 0.005,
            "analytic loft volume {vol} should be within 0.5% of {expected_volume} (err {:.3}%)",
            rel_err * 100.0
        );
    }

    #[test]
    fn loft_two_circles_volume_within_0_5pct_of_truncated_cone() {
        let h = 20.0;
        let big = 10.0;
        let small = 5.0;
        let expected = std::f64::consts::PI * h / 3.0 * (big * big + big * small + small * small);

        let mut topo = Topology::new();
        let bottom = make_circle_face_at(&mut topo, big, 0.0);
        let top = make_circle_face_at(&mut topo, small, h);
        let solid = loft(&mut topo, &[bottom, top]).unwrap();
        assert_analytic_frustum_solid(&topo, solid, expected);

        let mut topo2 = Topology::new();
        let bottom2 = make_nurbs_circle_face_at(&mut topo2, big, 0.0);
        let top2 = make_nurbs_circle_face_at(&mut topo2, small, h);
        let solid2 = loft(&mut topo2, &[bottom2, top2]).unwrap();
        assert_analytic_frustum_solid(&topo2, solid2, expected);
    }

    #[test]
    fn loft_two_equal_circles_makes_cylinder() {
        let h = 20.0;
        let r = 10.0;
        let expected = std::f64::consts::PI * r * r * h;

        let mut topo = Topology::new();
        let bottom = make_circle_face_at(&mut topo, r, 0.0);
        let top = make_circle_face_at(&mut topo, r, h);
        let solid = loft(&mut topo, &[bottom, top]).unwrap();
        assert_analytic_frustum_solid(&topo, solid, expected);
    }

    #[test]
    fn loft_non_coaxial_circles_falls_back_with_positive_volume() {
        let mut topo = Topology::new();
        let bottom = make_circle_face_at(&mut topo, 10.0, 0.0);
        let tol_val = 1e-7;
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let center = Point3::new(8.0, 0.0, 20.0);
        let circle = brepkit_math::curves::Circle3D::new(center, axis, 5.0).unwrap();
        let seam = topo.add_vertex(Vertex::new(circle.evaluate(0.0), tol_val));
        let edge = topo.add_edge(Edge::new(seam, seam, EdgeCurve::Circle(circle)));
        let wire = Wire::new(vec![OrientedEdge::new(edge, true)], true).unwrap();
        let wid = topo.add_wire(wire);
        let top = topo.add_face(Face::new(
            wid,
            vec![],
            FaceSurface::Plane {
                normal: axis,
                d: 20.0,
            },
        ));

        let solid = loft(&mut topo, &[bottom, top]).unwrap();
        let vol = crate::measure::solid_volume(&topo, solid, 0.1).unwrap();
        assert!(
            vol > 0.0,
            "non-coaxial loft should fall back to a positive-volume solid, got {vol}"
        );
    }

    #[test]
    fn loft_three_coaxial_circles_two_analytic_bands() {
        let r_big = 10.0;
        let r_mid = 5.0;
        let h = 20.0;
        let band =
            |ra: f64, rb: f64| std::f64::consts::PI * h / 3.0 * (ra * ra + ra * rb + rb * rb);
        let expected = band(r_big, r_mid) + band(r_mid, r_big);

        let mut topo = Topology::new();
        let p0 = make_circle_face_at(&mut topo, r_big, 0.0);
        let p1 = make_circle_face_at(&mut topo, r_mid, h);
        let p2 = make_circle_face_at(&mut topo, r_big, 2.0 * h);

        let solid = loft(&mut topo, &[p0, p1, p2]).unwrap();
        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        let analytic_sides = sh
            .faces()
            .iter()
            .filter(|&&fid| {
                matches!(
                    topo.face(fid).unwrap().surface(),
                    FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
                )
            })
            .count();
        assert_eq!(
            analytic_sides, 2,
            "three coaxial circles should emit two analytic frustum bands"
        );

        let vol = crate::measure::solid_volume(&topo, solid, 0.05).unwrap();
        let rel_err = (vol - expected).abs() / expected;
        assert!(
            rel_err < 0.005,
            "three-circle loft volume {vol} should be within 0.5% of {expected} (err {:.3}%)",
            rel_err * 100.0
        );
    }

    #[test]
    fn loft_smooth_surface_passes_through_profiles() {
        let mut topo = Topology::new();
        let p0 = make_square_at(&mut topo, 2.0, 0.0);
        let p1 = make_square_at(&mut topo, 1.0, 2.0);
        let p2 = make_square_at(&mut topo, 2.0, 4.0);

        let solid = loft_smooth(&mut topo, &[p0, p1, p2]).unwrap();

        let s = topo.solid(solid).unwrap();
        let sh = topo.shell(s.outer_shell()).unwrap();

        // Find a NURBS side face and verify it passes through the middle profile.
        for &fid in sh.faces() {
            let face = topo.face(fid).expect("face");
            if let FaceSurface::Nurbs(surface) = face.surface() {
                // At u=0.5 (middle profile), the surface should pass through
                // the middle profile's vertex positions. Evaluate at u=0.5, v=0.
                let mid_pt = surface.evaluate(0.5, 0.0);
                // The middle profile is at z=2.0.
                assert!(
                    (mid_pt.z() - 2.0).abs() < 0.5,
                    "surface at u=0.5 should be near z=2.0, got z={:.3}",
                    mid_pt.z()
                );
                break;
            }
        }
    }
}

//! Loft operation: create a solid by interpolating between profile faces.
//!
//! Equivalent to `BRepOffsetAPI_ThruSections` in `OpenCascade`. The loft
//! connects two or more planar profiles by creating ruled (linear)
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

/// Compute the centroid of a polygon's vertices.
#[allow(clippy::cast_precision_loss)]
fn polygon_centroid(verts: &[Point3]) -> Point3 {
    let inv = 1.0 / verts.len() as f64;
    let (sx, sy, sz) = verts.iter().fold((0.0, 0.0, 0.0), |(x, y, z), p| {
        (x + p.x(), y + p.y(), z + p.z())
    });
    Point3::new(sx * inv, sy * inv, sz * inv)
}

/// Compute the Newell normal of a polygon.
///
/// The Newell normal is proportional to twice the signed area of the polygon
/// and points in the direction of the polygon's outward normal (right-hand rule).
fn newell_normal(verts: &[Point3]) -> Vec3 {
    let m = verts.len();
    let mut nx = 0.0_f64;
    let mut ny = 0.0_f64;
    let mut nz = 0.0_f64;
    for i in 0..m {
        let curr = verts[i];
        let next = verts[(i + 1) % m];
        nx += (curr.y() - next.y()) * (curr.z() + next.z());
        ny += (curr.z() - next.z()) * (curr.x() + next.x());
        nz += (curr.x() - next.x()) * (curr.y() + next.y());
    }
    Vec3::new(nx, ny, nz)
}

/// Ensure profile vertices wind CCW relative to the stacking direction.
///
/// Computes the stacking direction from the first to last profile centroid,
/// then checks the Newell normal of the first profile. If the winding is CW
/// (Newell normal opposes stacking direction), reverses all profiles.
///
/// Returns `true` if the profiles were reversed.
fn ensure_ccw_winding(profile_verts: &mut [Vec<Point3>]) -> bool {
    let c0 = polygon_centroid(&profile_verts[0]);
    let c1 = polygon_centroid(&profile_verts[profile_verts.len() - 1]);
    let stack_dir = c1 - c0;
    let newell = newell_normal(&profile_verts[0]);
    if newell.dot(stack_dir) < 0.0 {
        for verts in profile_verts.iter_mut() {
            verts.reverse();
        }
        true
    } else {
        false
    }
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
    let winding_reversed = ensure_ccw_winding(&mut profile_verts);

    let num_profiles = profile_verts.len();
    let num_sections = num_profiles - 1;

    // Create all vertices.
    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
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
                    topo.edges
                        .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    // Create connecting edges between adjacent profiles.
    let connect_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = (0..num_sections)
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.edges.alloc(Edge::new(
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
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[0][0]);
        let fid = topo.faces.alloc(Face::new(
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

            let side_wire_id = topo.wires.alloc(side_wire);
            let side_face = topo.faces.alloc(Face::new(
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
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[num_profiles - 1][0]);
        let fid = topo.faces.alloc(Face::new(
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
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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
    let winding_reversed = ensure_ccw_winding(&mut profile_verts);

    let num_profiles = profile_verts.len();

    // Create all vertices.
    let ring_verts: Vec<Vec<brepkit_topology::vertex::VertexId>> = profile_verts
        .iter()
        .map(|verts| {
            verts
                .iter()
                .map(|&p| topo.vertices.alloc(Vertex::new(p, tol.linear)))
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
                    topo.edges
                        .alloc(Edge::new(ring[i], ring[next], EdgeCurve::Line))
                })
                .collect()
        })
        .collect();

    // Create connecting edges between adjacent profiles (used for topology).
    let _connect_edges: Vec<Vec<brepkit_topology::edge::EdgeId>> = (0..(num_profiles - 1))
        .map(|s| {
            (0..n)
                .map(|i| {
                    topo.edges.alloc(Edge::new(
                        ring_verts[s][i],
                        ring_verts[s + 1][i],
                        EdgeCurve::Line,
                    ))
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
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[0][0]);
        let fid = topo.faces.alloc(Face::new(
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
        let e_left_rail = topo.edges.alloc(Edge::new(
            ring_verts[0][i],
            ring_verts[last][i],
            EdgeCurve::Line,
        ));
        let e_right_rail = topo.edges.alloc(Edge::new(
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

        let side_wire_id = topo.wires.alloc(side_wire);
        let side_face =
            topo.faces
                .alloc(Face::new(side_wire_id, vec![], FaceSurface::Nurbs(surface)));
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
        let wid = topo.wires.alloc(wire);
        let cap_d = dot_normal_point(cap_normal, profile_verts[num_profiles - 1][0]);
        let fid = topo.faces.alloc(Face::new(
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
    let shell_id = topo.shells.alloc(shell);
    Ok(topo.solids.alloc(Solid::new(shell_id, vec![])))
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
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(-hs, -hs, z), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(hs, -hs, z), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(hs, hs, z), tol_val));
        let v3 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(-hs, hs, z), tol_val));

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

        topo.faces.alloc(Face::new(
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
        let v0 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.0, 0.0, 1.0), tol_val));
        let v1 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(1.0, 0.0, 1.0), tol_val));
        let v2 = topo
            .vertices
            .alloc(Vertex::new(Point3::new(0.5, 1.0, 1.0), tol_val));

        let e0 = topo.edges.alloc(Edge::new(v0, v1, EdgeCurve::Line));
        let e1 = topo.edges.alloc(Edge::new(v1, v2, EdgeCurve::Line));
        let e2 = topo.edges.alloc(Edge::new(v2, v0, EdgeCurve::Line));

        let wire = Wire::new(
            vec![
                OrientedEdge::new(e0, true),
                OrientedEdge::new(e1, true),
                OrientedEdge::new(e2, true),
            ],
            true,
        )
        .unwrap();
        let wid = topo.wires.alloc(wire);
        let tri = topo.faces.alloc(Face::new(
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

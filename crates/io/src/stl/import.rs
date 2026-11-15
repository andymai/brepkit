//! Mesh import: convert triangle meshes into B-Rep topology.
//!
//! Takes a [`TriangleMesh`] (from [`read_stl`](super::reader::read_stl) or
//! another mesh reader) and builds one planar face per triangle. Triangles
//! that meet at an edge share that edge, so a closed set of triangles is one
//! closed shell. [`import_mesh_parts`] returns each closed body as a solid
//! and each open set of triangles as a shell.

use std::collections::HashMap;

use brepkit_math::madd::Madd;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_operations::tessellate::TriangleMesh;
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::{Shell, ShellId};
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire};

use crate::IoError;

/// The bodies of an imported mesh.
#[derive(Debug, Clone, Default)]
pub struct MeshImport {
    /// One solid for each closed body. A closed set of triangles that faces
    /// inward, and lies in a body, is a cavity of that body.
    pub solids: Vec<SolidId>,
    /// One shell for each set of joined triangles that does not close.
    pub shells: Vec<ShellId>,
}

/// Import a [`TriangleMesh`] into topology, one entity for each body.
///
/// Each triangle becomes a planar face. Vertices at the same position
/// (within `tolerance`) are merged, and two triangles that use the same two
/// vertices share one edge. The triangles joined through shared edges are
/// one body:
///
/// - A body closes when each of its edges has two triangles that run it in
///   opposite directions. Triangles of a closed body that are wound against
///   their neighbours are turned.
/// - A closed body that faces outward is a solid, also when it lies in
///   another body.
/// - A closed body that faces inward is a cavity of the smallest outward
///   body round it. With no such body, it is turned and becomes a solid.
/// - Every other body is a shell: an open sheet, a set with an edge that
///   more than two triangles use, a one-sided surface, and a closed set
///   with no volume.
///
/// # Errors
///
/// Returns [`IoError`] if the mesh has no triangle with three separate
/// corners, or an index is past the end of the positions.
pub fn import_mesh_parts(
    topo: &mut Topology,
    mesh: &TriangleMesh,
    tolerance: f64,
) -> Result<MeshImport, IoError> {
    if mesh.indices.len() < 3 {
        return Err(IoError::InvalidTopology {
            reason: "mesh has no triangles".to_string(),
        });
    }
    if let Some(&bad) = mesh
        .indices
        .iter()
        .find(|&&i| i as usize >= mesh.positions.len())
    {
        return Err(IoError::InvalidTopology {
            reason: format!(
                "mesh index {bad} is past its {} positions",
                mesh.positions.len()
            ),
        });
    }

    let (points, remap) = merge_positions(&mesh.positions, tolerance);
    let mut tris = oriented_triangles(mesh, &remap);
    if tris.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: "no valid triangles in mesh".to_string(),
        });
    }

    let mut bodies = split_bodies(&mut tris, &points);
    assign_cavities(&mut bodies, &mut tris, &points);

    let vertex_ids: Vec<VertexId> = points
        .iter()
        .map(|&p| topo.add_vertex(Vertex::new(p, tolerance)))
        .collect();
    let mut edges: HashMap<(usize, usize), EdgeId> = HashMap::new();
    let mut shell_ids = Vec::with_capacity(bodies.len());
    for body in &bodies {
        let mut faces = Vec::with_capacity(body.tris.len());
        for &t in &body.tris {
            faces.push(build_triangle_face(
                topo,
                &points,
                &vertex_ids,
                &mut edges,
                tris[t],
            )?);
        }
        let shell = Shell::new(faces).map_err(|e| IoError::ParseError {
            reason: format!("failed to build shell from mesh: {e}"),
        })?;
        shell_ids.push(topo.add_shell(shell));
    }

    let mut out = MeshImport::default();
    for (i, body) in bodies.iter().enumerate() {
        match body.kind {
            BodyKind::Open => out.shells.push(shell_ids[i]),
            BodyKind::Cavity(_) => {}
            BodyKind::Solid => {
                let cavities = bodies
                    .iter()
                    .enumerate()
                    .filter(|(_, b)| b.kind == BodyKind::Cavity(i))
                    .map(|(j, _)| shell_ids[j])
                    .collect();
                out.solids
                    .push(topo.add_solid(Solid::new(shell_ids[i], cavities)));
            }
        }
    }
    Ok(out)
}

/// Import a closed [`TriangleMesh`] into topology as a single solid.
///
/// This is [`import_mesh_parts`] for a mesh of closed bodies. One body
/// comes back as its solid. Two or more bodies come back in one solid, with
/// the outside of each body in its outer shell and each cavity as an inner
/// shell.
///
/// # Errors
///
/// Returns [`IoError`] if:
/// - The mesh has no triangles
/// - A part of the mesh does not close; [`import_mesh_parts`] returns such
///   a part as a shell
pub fn import_mesh(
    topo: &mut Topology,
    mesh: &TriangleMesh,
    tolerance: f64,
) -> Result<SolidId, IoError> {
    let parts = import_mesh_parts(topo, mesh, tolerance)?;
    if !parts.shells.is_empty() {
        return Err(IoError::InvalidTopology {
            reason: format!(
                "the mesh does not close: {} of its {} bodies are open",
                parts.shells.len(),
                parts.shells.len() + parts.solids.len()
            ),
        });
    }
    if let [solid] = parts.solids[..] {
        return Ok(solid);
    }

    let mut outer = Vec::new();
    let mut cavities = Vec::new();
    for &sid in &parts.solids {
        let solid = topo.solid(sid).map_err(topo_err)?;
        outer.extend_from_slice(topo.shell(solid.outer_shell()).map_err(topo_err)?.faces());
        cavities.extend_from_slice(solid.inner_shells());
    }
    let shell = Shell::new(outer).map_err(|e| IoError::ParseError {
        reason: format!("failed to build shell from mesh: {e}"),
    })?;
    let shell_id = topo.add_shell(shell);
    Ok(topo.add_solid(Solid::new(shell_id, cavities)))
}

/// Merge positions within `tolerance` of each other.
///
/// Returns the kept points, and for each input position the index of its
/// kept point. A grid of `tolerance` cells finds the neighbours of a
/// position, so the cost is linear in the count of positions.
fn merge_positions(positions: &[Point3], tolerance: f64) -> (Vec<Point3>, Vec<usize>) {
    if !(tolerance.is_finite() && tolerance > 0.0) {
        // Only equal positions merge. `+ 0.0` gives the two zeros one key.
        let mut seen: HashMap<[u64; 3], usize> = HashMap::new();
        let mut points = Vec::new();
        let remap = positions
            .iter()
            .map(|&pos| {
                let key = [pos.x(), pos.y(), pos.z()].map(|c| (c + 0.0).to_bits());
                *seen.entry(key).or_insert_with(|| {
                    points.push(pos);
                    points.len() - 1
                })
            })
            .collect();
        return (points, remap);
    }
    let cell = tolerance;
    let tol_sq = tolerance * tolerance;
    #[allow(clippy::cast_possible_truncation)]
    let key = |p: Point3| -> [i64; 3] {
        [
            (p.x() / cell).floor() as i64,
            (p.y() / cell).floor() as i64,
            (p.z() / cell).floor() as i64,
        ]
    };

    let mut grid: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    let mut points: Vec<Point3> = Vec::new();
    let mut remap = Vec::with_capacity(positions.len());
    for &pos in positions {
        let k = key(pos);
        let mut found = None;
        'cells: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let near = [
                        k[0].saturating_add(dx),
                        k[1].saturating_add(dy),
                        k[2].saturating_add(dz),
                    ];
                    let Some(bucket) = grid.get(&near) else {
                        continue;
                    };
                    for &i in bucket {
                        let d = points[i] - pos;
                        let dist_sq = d.x().madd(d.x(), d.y().madd(d.y(), d.z() * d.z()));
                        if dist_sq < tol_sq || dist_sq == 0.0 {
                            found = Some(found.map_or(i, |f: usize| f.min(i)));
                            if dist_sq == 0.0 {
                                break 'cells;
                            }
                        }
                    }
                }
            }
        }
        let index = found.unwrap_or_else(|| {
            points.push(pos);
            grid.entry(k).or_default().push(points.len() - 1);
            points.len() - 1
        });
        remap.push(index);
    }
    (points, remap)
}

/// The triangles of the mesh on its merged points, less each triangle with
/// two corners on one point.
///
/// A stored normal that points against a triangle's winding turns the
/// triangle.
fn oriented_triangles(mesh: &TriangleMesh, remap: &[usize]) -> Vec<[usize; 3]> {
    let has_normals = mesh.normals.len() >= mesh.positions.len();
    let mut tris = Vec::with_capacity(mesh.indices.len() / 3);
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let mut t = [remap[i0], remap[i1], remap[i2]];
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue;
        }
        if has_normals {
            let p0 = mesh.positions[i0];
            let geo_normal = (mesh.positions[i1] - p0).cross(mesh.positions[i2] - p0);
            if geo_normal.dot(mesh.normals[i0]) < 0.0 {
                t.swap(1, 2);
            }
        }
        tris.push(t);
    }
    tris
}

/// What a set of joined triangles is in the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyKind {
    /// A shell: the triangles do not bound a volume.
    Open,
    /// A closed body that faces outward.
    Solid,
    /// A closed body that faces inward, in the body with this index.
    Cavity(usize),
}

/// A set of triangles joined through shared edges.
struct Body {
    tris: Vec<usize>,
    kind: BodyKind,
    /// Six times the signed volume; zero for an open body.
    volume6: f64,
}

const fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

/// Group the triangles into bodies, in the order of each body's first
/// triangle, and turn the triangles of a closed body to one side.
///
/// A closed body has [`BodyKind::Solid`] here, whichever way it faces;
/// [`assign_cavities`] reads the sign of its volume.
fn split_bodies(tris: &mut [[usize; 3]], points: &[Point3]) -> Vec<Body> {
    // Each edge's triangles, with `true` for a triangle that runs the edge
    // from its lower point to its higher one.
    let mut uses: HashMap<(usize, usize), Vec<(usize, bool)>> = HashMap::new();
    for (t, tri) in tris.iter().enumerate() {
        for k in 0..3 {
            let (a, b) = (tri[k], tri[(k + 1) % 3]);
            uses.entry(edge_key(a, b)).or_default().push((t, a < b));
        }
    }

    let mut body_of = vec![usize::MAX; tris.len()];
    let mut bodies: Vec<Body> = Vec::new();
    for seed in 0..tris.len() {
        if body_of[seed] != usize::MAX {
            continue;
        }
        let index = bodies.len();
        let mut members = vec![seed];
        body_of[seed] = index;
        let mut next = 0;
        while next < members.len() {
            let tri = tris[members[next]];
            next += 1;
            for k in 0..3 {
                for &(other, _) in &uses[&edge_key(tri[k], tri[(k + 1) % 3])] {
                    if body_of[other] == usize::MAX {
                        body_of[other] = index;
                        members.push(other);
                    }
                }
            }
        }
        members.sort_unstable();
        bodies.push(Body {
            tris: members,
            kind: BodyKind::Open,
            volume6: 0.0,
        });
    }

    for body in &mut bodies {
        let two_sided = body
            .tris
            .iter()
            .all(|&t| (0..3).all(|k| uses[&edge_key(tris[t][k], tris[t][(k + 1) % 3])].len() == 2));
        if !two_sided {
            continue;
        }
        let Some(turned) = consistent_turns(&body.tris, tris, &uses) else {
            continue;
        };
        for &t in &turned {
            tris[t].swap(1, 2);
        }
        let volume6 = signed_volume6(&body.tris, tris, points);
        let size = body_extent(&body.tris, tris, points);
        if volume6.abs() > 1e-12 * size * size * size {
            body.kind = BodyKind::Solid;
            body.volume6 = volume6;
        } else {
            // A closed set with no volume (two triangles back to back) is
            // a sheet; it keeps its winding.
            for &t in &turned {
                tris[t].swap(1, 2);
            }
        }
    }
    bodies
}

/// The triangles of a two-sided body to turn so that each edge's two
/// triangles run it in opposite directions.
///
/// Of the two answers, this is the one that turns fewer triangles. Returns
/// `None` for a one-sided surface, which has no such answer.
fn consistent_turns(
    members: &[usize],
    tris: &[[usize; 3]],
    uses: &HashMap<(usize, usize), Vec<(usize, bool)>>,
) -> Option<Vec<usize>> {
    let mut turn: HashMap<usize, bool> = HashMap::with_capacity(members.len());
    let mut queue = vec![members[0]];
    turn.insert(members[0], false);
    let mut next = 0;
    while next < queue.len() {
        let t = queue[next];
        next += 1;
        let t_turn = turn[&t];
        for k in 0..3 {
            let (a, b) = (tris[t][k], tris[t][(k + 1) % 3]);
            let forward = (a < b) != t_turn;
            for &(other, other_forward) in &uses[&edge_key(a, b)] {
                if other == t {
                    continue;
                }
                // The neighbour must run the edge the other way.
                let needs_turn = other_forward == forward;
                match turn.get(&other) {
                    Some(&known) if known != needs_turn => return None,
                    Some(_) => {}
                    None => {
                        turn.insert(other, needs_turn);
                        queue.push(other);
                    }
                }
            }
        }
    }
    let turned: Vec<usize> = members.iter().copied().filter(|t| turn[t]).collect();
    if turned.len() * 2 > members.len() {
        Some(members.iter().copied().filter(|t| !turn[t]).collect())
    } else {
        Some(turned)
    }
}

fn signed_volume6(members: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> f64 {
    // Measured from a point of the body, so a body far from the origin
    // keeps its digits.
    let origin = points[tris[members[0]][0]];
    members
        .iter()
        .map(|&t| {
            let [a, b, c] = tris[t].map(|i| points[i] - origin);
            a.dot(b.cross(c))
        })
        .sum()
}

fn body_box(members: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for &t in members {
        for i in tris[t] {
            let p = [points[i].x(), points[i].y(), points[i].z()];
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    (lo, hi)
}

fn body_extent(members: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> f64 {
    let (lo, hi) = body_box(members, tris, points);
    (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max)
}

/// Make each inward closed body a cavity of the smallest outward body
/// round it, or turn it into a solid when it has none.
fn assign_cavities(bodies: &mut [Body], tris: &mut [[usize; 3]], points: &[Point3]) {
    let closed: Vec<usize> = (0..bodies.len())
        .filter(|&i| bodies[i].kind == BodyKind::Solid)
        .collect();
    let boxes: Vec<_> = bodies
        .iter()
        .map(|b| body_box(&b.tris, tris, points))
        .collect();

    let mut owners: Vec<(usize, Option<usize>)> = Vec::new();
    for &i in &closed {
        if bodies[i].volume6 > 0.0 {
            continue;
        }
        // The smallest closed body round body `i` is the one it lies
        // directly in.
        let mut around: Option<usize> = None;
        for &j in &closed {
            if j == i || bodies[j].volume6.abs() <= bodies[i].volume6.abs() {
                continue;
            }
            let inside_box =
                (0..3).all(|k| boxes[i].0[k] >= boxes[j].0[k] && boxes[i].1[k] <= boxes[j].1[k]);
            if !inside_box || !lies_in(&bodies[i].tris, &bodies[j].tris, tris, points) {
                continue;
            }
            if around.is_none_or(|a| bodies[j].volume6.abs() < bodies[a].volume6.abs()) {
                around = Some(j);
            }
        }
        owners.push((i, around.filter(|&j| bodies[j].volume6 > 0.0)));
    }

    for (i, owner) in owners {
        if let Some(j) = owner {
            bodies[i].kind = BodyKind::Cavity(j);
        } else {
            for &t in &bodies[i].tris {
                tris[t].swap(1, 2);
            }
            bodies[i].volume6 = -bodies[i].volume6;
        }
    }
}

/// Whether the closed body `inner` lies in the closed body `outer`.
///
/// Two bodies of a mesh do not cross, so one point of `inner` decides. A
/// point on `outer` itself (the two bodies touch there) decides nothing,
/// and the next point is read.
fn lies_in(inner: &[usize], outer: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> bool {
    for &t in inner {
        for i in tris[t] {
            let turns = winding_number(points[i], outer, tris, points).abs();
            if (turns - turns.round()).abs() < 0.25 {
                return turns.round() >= 1.0;
            }
        }
    }
    false
}

/// The count of times the triangles of `members` wrap round `p`: the sum of
/// their solid angles at `p`, over a full sphere.
fn winding_number(p: Point3, members: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> f64 {
    let mut total = 0.0;
    for &t in members {
        let [a, b, c] = tris[t].map(|i| points[i] - p);
        let (la, lb, lc) = (a.length(), b.length(), c.length());
        let det = a.dot(b.cross(c));
        let denom = (la * lb).madd(lc, a.dot(b).madd(lc, b.dot(c).madd(la, c.dot(a) * lb)));
        total += 2.0 * det.atan2(denom);
    }
    total / (4.0 * std::f64::consts::PI)
}

/// Build a triangular planar face on three points, with the edges that
/// earlier triangles made on the same points.
fn build_triangle_face(
    topo: &mut Topology,
    points: &[Point3],
    vertex_ids: &[VertexId],
    edges: &mut HashMap<(usize, usize), EdgeId>,
    tri: [usize; 3],
) -> Result<FaceId, IoError> {
    let mut oriented = Vec::with_capacity(3);
    for k in 0..3 {
        let (a, b) = (tri[k], tri[(k + 1) % 3]);
        let (lo, hi) = edge_key(a, b);
        let edge = *edges.entry((lo, hi)).or_insert_with(|| {
            topo.add_edge(Edge::new(vertex_ids[lo], vertex_ids[hi], EdgeCurve::Line))
        });
        oriented.push(OrientedEdge::new(edge, a < b));
    }
    let wire = Wire::new(oriented, true).map_err(|e| IoError::ParseError {
        reason: format!("failed to build triangle wire: {e}"),
    })?;
    let wire_id = topo.add_wire(wire);

    let [p0, p1, p2] = tri.map(|i| points[i]);
    let normal = (p1 - p0)
        .cross(p2 - p0)
        .normalize()
        .unwrap_or(Vec3::new(0.0, 0.0, 1.0));
    let d = normal.dot(Vec3::new(p0.x(), p0.y(), p0.z()));

    let surface = FaceSurface::Plane { normal, d };
    Ok(topo.add_face(Face::new(wire_id, Vec::new(), surface)))
}

/// Convert a [`TopologyError`] into an [`IoError`].
fn topo_err(e: brepkit_topology::TopologyError) -> IoError {
    IoError::Operations(brepkit_operations::OperationsError::from(e))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::print_stderr)]

    use brepkit_topology::Topology;
    use brepkit_topology::test_utils::make_unit_cube_non_manifold;

    use super::*;
    use crate::stl::reader::read_stl;
    use crate::stl::writer::{self, StlFormat};

    #[test]
    fn vol_from_faces_8_vertex_box() {
        use brepkit_math::vec::Vec3;
        use brepkit_operations::tessellate::TriangleMesh;

        let positions = vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 0.0),
            Point3::new(0.0, 10.0, 0.0),
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(10.0, 0.0, 10.0),
            Point3::new(10.0, 10.0, 10.0),
            Point3::new(0.0, 10.0, 10.0),
        ];
        let normals = vec![Vec3::new(0.0, 0.0, 1.0); 8];
        let indices = vec![
            0u32, 2, 1, 0, 3, 2, // bottom -Z
            4, 5, 6, 4, 6, 7, // top +Z
            0, 1, 5, 0, 5, 4, // front -Y
            2, 3, 7, 2, 7, 6, // back +Y
            0, 4, 7, 0, 7, 3, // left -X
            1, 2, 6, 1, 6, 5, // right +X
        ];

        let mesh = TriangleMesh {
            positions,
            normals,
            indices,
        };
        let mut topo = Topology::new();
        let solid = import_mesh(&mut topo, &mesh, 1e-7).unwrap();

        let vol = brepkit_operations::measure::solid_volume_from_faces(&topo, solid, 0.01).unwrap();
        assert!(
            (vol - 1000.0).abs() < 10.0,
            "expected ~1000 from vol_from_faces, got {vol}"
        );
    }

    #[test]
    fn vol_from_faces_stl_roundtrip() {
        // This simulates the actual path: tessellate box → flat mesh → import_mesh
        let mut write_topo = Topology::new();
        let solid =
            brepkit_operations::primitives::make_box(&mut write_topo, 10.0, 10.0, 10.0).unwrap();

        let stl_bytes = writer::write_stl(&write_topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&stl_bytes).unwrap();

        let mut topo = Topology::new();
        let imported = import_mesh(&mut topo, &mesh, 1e-4).unwrap();

        let vol =
            brepkit_operations::measure::solid_volume_from_faces(&topo, imported, 0.01).unwrap();
        assert!(
            (vol - 1000.0).abs() < 10.0,
            "expected ~1000 from vol_from_faces, got {vol}"
        );
    }

    #[test]
    fn vol_from_faces_per_face_tessellation() {
        // Simulates the JS 3MF path: per-face tessellate → flat mesh → import_mesh
        // This is the path that produces 333.33 instead of 1000.
        use brepkit_operations::tessellate;

        let mut topo = Topology::new();
        let solid = brepkit_operations::primitives::make_box(&mut topo, 10.0, 10.0, 10.0).unwrap();

        // Per-face tessellation (same as JS meshSolid/tessellateFace)
        let solid_data = topo.solid(solid).unwrap();
        let shell = topo.shell(solid_data.outer_shell()).unwrap();
        let face_ids: Vec<_> = shell.faces().to_vec();

        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        let mut vert_offset = 0u32;

        for &fid in &face_ids {
            let mesh = tessellate::tessellate(&topo, fid, 0.1).unwrap();
            let n_verts = mesh.positions.len();
            positions.extend_from_slice(&mesh.positions);
            normals.extend_from_slice(&mesh.normals);
            for &idx in &mesh.indices {
                indices.push(idx + vert_offset);
            }
            vert_offset += n_verts as u32;
        }

        let mesh = brepkit_operations::tessellate::TriangleMesh {
            positions,
            normals,
            indices,
        };

        let mut import_topo = Topology::new();
        let imported = import_mesh(&mut import_topo, &mesh, 1e-4).unwrap();

        let vol_from_faces =
            brepkit_operations::measure::solid_volume_from_faces(&import_topo, imported, 0.01)
                .unwrap();
        let vol_standard =
            brepkit_operations::measure::solid_volume(&import_topo, imported, 0.01).unwrap();

        eprintln!("vol_from_faces = {vol_from_faces}");
        eprintln!("vol_standard   = {vol_standard}");

        assert!(
            (vol_from_faces - 1000.0).abs() < 10.0,
            "expected ~1000 from vol_from_faces, got {vol_from_faces}"
        );
        assert!(
            (vol_standard - 1000.0).abs() < 10.0,
            "expected ~1000 from vol_standard, got {vol_standard}"
        );
    }

    #[test]
    fn import_single_triangle() {
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
                Vec3::new(0.0, 0.0, 1.0),
            ],
            indices: vec![0, 1, 2],
        };

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();

        assert!(parts.solids.is_empty());
        assert_eq!(parts.shells.len(), 1);
        assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 1);
        assert!(import_mesh(&mut topo, &mesh, 1e-7).is_err());
    }

    #[test]
    fn import_two_triangles() {
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(1.0, 1.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-7).unwrap();

        // The two triangles share the edge from (1, 0, 0) to (0, 1, 0).
        assert!(parts.solids.is_empty());
        assert_eq!(parts.shells.len(), 1);
        assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 2);
        assert_eq!(topo.edges().len(), 5);
    }

    #[test]
    fn import_stl_roundtrip_unit_cube() {
        let mut write_topo = Topology::new();
        let solid = make_unit_cube_non_manifold(&mut write_topo);

        let stl_bytes = writer::write_stl(&write_topo, &[solid], 0.1, StlFormat::Binary).unwrap();
        let mesh = read_stl(&stl_bytes).unwrap();

        let mut read_topo = Topology::new();
        let imported = import_mesh(&mut read_topo, &mesh, 1e-4).unwrap();

        let read_solid = read_topo.solid(imported).unwrap();
        let shell = read_topo.shell(read_solid.outer_shell()).unwrap();
        // Unit cube: 12 triangles.
        assert_eq!(shell.faces().len(), 12);
    }

    #[test]
    fn vertex_merging() {
        // Two triangles sharing an edge — should merge 2 vertices.
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0),
                Point3::new(1.0, 0.0, 0.0), // Same as [1]
                Point3::new(2.0, 0.0, 0.0),
                Point3::new(0.5, 1.0, 0.0), // Same as [2]
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let _parts = import_mesh_parts(&mut topo, &mesh, 1e-6).unwrap();

        // Should have 4 unique vertices, not 6.
        assert_eq!(topo.vertices().len(), 4);
    }

    #[test]
    fn empty_mesh_error() {
        let mesh = TriangleMesh::default();
        let mut topo = Topology::new();
        let result = import_mesh(&mut topo, &mesh, 1e-7);
        assert!(result.is_err());
    }

    #[test]
    fn degenerate_triangles_skipped() {
        // Triangle with two coincident vertices should be skipped.
        let mesh = TriangleMesh {
            positions: vec![
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(0.0, 0.0, 0.0), // Same as [0]
                Point3::new(1.0, 1.0, 0.0),
                // Valid triangle
                Point3::new(0.0, 0.0, 0.0),
                Point3::new(1.0, 0.0, 0.0),
                Point3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::new(0.0, 0.0, 1.0); 6],
            indices: vec![0, 1, 2, 3, 4, 5],
        };

        let mut topo = Topology::new();
        let parts = import_mesh_parts(&mut topo, &mesh, 1e-6).unwrap();

        // Only the valid triangle should remain.
        assert_eq!(parts.shells.len(), 1);
        assert_eq!(topo.shell(parts.shells[0]).unwrap().faces().len(), 1);
    }
}

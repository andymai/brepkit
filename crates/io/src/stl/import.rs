//! Mesh import: convert triangle meshes into B-Rep topology.
//!
//! Takes a [`TriangleMesh`] (from [`read_stl`](super::reader::read_stl) or
//! another mesh reader) and builds one planar face per triangle. Triangles
//! that meet at an edge share that edge, so a closed set of triangles is one
//! closed shell. [`import_mesh_parts`] returns each closed body as a solid
//! and each open set of triangles as a shell.

use std::collections::{BinaryHeap, HashMap, HashSet};

use brepkit_math::madd::Madd;
use brepkit_math::predicates::{orient2d, orient3d};
use brepkit_math::vec::{Point2, Point3, Vec3};
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
    /// One shell for each set of joined triangles that does not bound a
    /// solid: a set that does not close, and a closed set that faces inward
    /// and crosses another body.
    pub shells: Vec<ShellId>,
}

/// Import a [`TriangleMesh`] into topology, one entity for each body.
///
/// Each triangle becomes a planar face. Vertices at the same position
/// (within `tolerance`) are merged, and two triangles that use the same two
/// vertices share one edge. The triangles joined through shared edges are
/// one body, with vertices of its own: two bodies that touch at a point
/// share no vertex.
///
/// - At an edge of more than two triangles, the triangles pair off by their
///   order round the edge, and each pair joins one body there. In each pair
///   the two triangles run the edge in opposite directions. When two ways
///   to pair them do that, each pair holds the material between its two
///   triangles, so two closed bodies that share an edge are two bodies.
/// - A body closes when each of its edges has two triangles that run it in
///   opposite directions. Triangles of a closed body that are wound against
///   their neighbours are turned.
/// - Two closed bodies cross when a triangle of one passes through a
///   triangle of the other. Two bodies that touch do not cross.
/// - A closed body that faces outward is a solid, also when it lies in or
///   crosses another body.
/// - A closed body that faces inward is a cavity of the outward body that
///   it lies directly in. When it crosses another body, it is a shell: it
///   bounds no solid. In any other place (in no body, or directly in
///   another inward body), it is turned and becomes a solid.
/// - Every other body is a shell: an open sheet, a set with an edge whose
///   triangles do not pair off, a one-sided surface, and a closed set with
///   no volume.
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
    let bodies = MeshBodies::read(mesh, tolerance)?;
    bodies.build(topo, tolerance)
}

/// The bodies of a mesh, before any of them is in a topology.
struct MeshBodies {
    points: Vec<Point3>,
    tris: Vec<[usize; 3]>,
    bodies: Vec<Body>,
    /// At an edge of more than two triangles, the triangle that each one
    /// pairs with.
    pairs: HashMap<(usize, (usize, usize)), usize>,
    placement: Placement,
}

impl MeshBodies {
    fn read(mesh: &TriangleMesh, tolerance: f64) -> Result<Self, IoError> {
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
        let (mut tris, votes) = mesh_triangles(mesh, &remap);
        if tris.is_empty() {
            return Err(IoError::InvalidTopology {
                reason: "no valid triangles in mesh".to_string(),
            });
        }

        let links = Links::read(&tris, &points, &votes, tolerance);
        let mut bodies = split_bodies(&mut tris, &votes, &points, &links);
        let placement = assign_cavities(&mut bodies, &mut tris, &points, &links, tolerance);
        Ok(Self {
            points,
            tris,
            bodies,
            pairs: links.pairs,
            placement,
        })
    }

    fn open_count(&self) -> usize {
        self.bodies
            .iter()
            .filter(|b| b.kind == BodyKind::Open)
            .count()
    }

    fn build(&self, topo: &mut Topology, tolerance: f64) -> Result<MeshImport, IoError> {
        let Self {
            points,
            tris,
            bodies,
            pairs,
            ..
        } = self;
        let mut shell_ids = Vec::with_capacity(bodies.len());
        for body in bodies {
            // The vertices and edges of one body. A point where two bodies
            // touch gets a vertex in each, so a transform of one body does not
            // move the other.
            let mut vertices: HashMap<usize, VertexId> = HashMap::new();
            let mut edges: HashMap<(usize, usize, usize), EdgeId> = HashMap::new();
            let mut faces = Vec::with_capacity(body.tris.len());
            for &t in &body.tris {
                let tri = tris[t];
                let corners = tri.map(|i| {
                    *vertices
                        .entry(i)
                        .or_insert_with(|| topo.add_vertex(Vertex::new(points[i], tolerance)))
                });
                // Each pair round an edge of more than two triangles has an
                // edge of its own, also when two pairs are in one body.
                let tags = [0, 1, 2].map(|k| {
                    pairs
                        .get(&(t, edge_key(tri[k], tri[(k + 1) % 3])))
                        .map_or(usize::MAX, |&other| other.min(t))
                });
                faces.push(build_triangle_face(
                    topo,
                    tri.map(|i| points[i]),
                    tri,
                    corners,
                    tags,
                    &mut edges,
                )?);
            }
            let shell = Shell::new(faces).map_err(|e| IoError::ParseError {
                reason: format!("failed to build shell from mesh: {e}"),
            })?;
            shell_ids.push(topo.add_shell(shell));
        }

        let mut cavities: HashMap<usize, Vec<ShellId>> = HashMap::new();
        for (i, body) in bodies.iter().enumerate() {
            if let BodyKind::Cavity(owner) = body.kind {
                cavities.entry(owner).or_default().push(shell_ids[i]);
            }
        }
        let mut out = MeshImport::default();
        for (i, body) in bodies.iter().enumerate() {
            match body.kind {
                BodyKind::Open => out.shells.push(shell_ids[i]),
                BodyKind::Cavity(_) => {}
                BodyKind::Solid => {
                    let inner = cavities.remove(&i).unwrap_or_default();
                    out.solids
                        .push(topo.add_solid(Solid::new(shell_ids[i], inner)));
                }
            }
        }
        Ok(out)
    }
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
/// - Two closed bodies of the mesh cross. The material of the two has no
///   boundary in the triangles of the mesh
/// - A part of the mesh does not close; [`import_mesh_parts`] returns such
///   a part as a shell
/// - A body lies in the material of another body. One solid cannot hold
///   the two; [`import_mesh_parts`] returns each as a solid
pub fn import_mesh(
    topo: &mut Topology,
    mesh: &TriangleMesh,
    tolerance: f64,
) -> Result<SolidId, IoError> {
    // Read the whole mesh first: a refused import adds nothing to `topo`.
    let bodies = MeshBodies::read(mesh, tolerance)?;
    if bodies.placement.crossing {
        return Err(IoError::InvalidTopology {
            reason: "two closed bodies of the mesh cross, so the mesh is not one solid".to_string(),
        });
    }
    let open = bodies.open_count();
    if open > 0 {
        return Err(IoError::InvalidTopology {
            reason: format!(
                "the mesh does not close: {open} of its {} bodies are open",
                bodies.bodies.len()
            ),
        });
    }
    if bodies.placement.overlap {
        return Err(IoError::InvalidTopology {
            reason: "a body of the mesh lies in the material of another body, \
                     so the mesh is not one solid"
                .to_string(),
        });
    }
    let parts = bodies.build(topo, tolerance)?;
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
/// two corners on one point, as they are wound.
///
/// Each triangle comes with the vote of its stored normals: the dot product
/// of their sum with the normal of the winding, or zero for a mesh with no
/// normals. A vote below zero is a normal that points against the winding.
fn mesh_triangles(mesh: &TriangleMesh, remap: &[usize]) -> (Vec<[usize; 3]>, Vec<f64>) {
    let has_normals = mesh.normals.len() >= mesh.positions.len();
    let mut tris = Vec::with_capacity(mesh.indices.len() / 3);
    let mut votes = Vec::with_capacity(mesh.indices.len() / 3);
    for tri in mesh.indices.chunks_exact(3) {
        let (i0, i1, i2) = (tri[0] as usize, tri[1] as usize, tri[2] as usize);
        let t = [remap[i0], remap[i1], remap[i2]];
        if t[0] == t[1] || t[1] == t[2] || t[0] == t[2] {
            continue;
        }
        let vote = if has_normals {
            let p0 = mesh.positions[i0];
            let geo_normal = (mesh.positions[i1] - p0).cross(mesh.positions[i2] - p0);
            geo_normal.dot(mesh.normals[i0] + mesh.normals[i1] + mesh.normals[i2])
        } else {
            0.0
        };
        tris.push(t);
        votes.push(if vote.is_finite() { vote } else { 0.0 });
    }
    (tris, votes)
}

/// What a set of joined triangles is in the result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyKind {
    /// A shell: the triangles do not bound a solid.
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

/// Whether the triangle `tri` runs the edge `(lo, hi)` from `lo` to `hi`.
fn runs_up(tri: [usize; 3], (lo, hi): (usize, usize)) -> bool {
    (0..3).any(|k| tri[k] == lo && tri[(k + 1) % 3] == hi)
}

/// How the triangles of a mesh meet at their edges.
struct Links {
    /// The triangles of each edge, with `true` for a triangle that runs the
    /// edge from its lower point to its higher one.
    uses: HashMap<(usize, usize), Vec<(usize, bool)>>,
    /// At an edge of more than two triangles that pair off round it, the
    /// triangle across the edge from each one.
    pairs: HashMap<(usize, (usize, usize)), usize>,
}

impl Links {
    fn read(tris: &[[usize; 3]], points: &[Point3], votes: &[f64], tolerance: f64) -> Self {
        let mut uses: HashMap<(usize, usize), Vec<(usize, bool)>> = HashMap::new();
        for (t, tri) in tris.iter().enumerate() {
            for k in 0..3 {
                let (a, b) = (tri[k], tri[(k + 1) % 3]);
                uses.entry(edge_key(a, b)).or_default().push((t, a < b));
            }
        }
        let mut pairs = HashMap::new();
        if uses.values().any(|list| list.len() > 2) {
            let turned = patch_turns(tris, &uses, votes);
            for (&key, list) in &uses {
                if list.len() <= 2 {
                    continue;
                }
                for [s, t] in pair_round_edge(key, list, &turned, tris, points, tolerance) {
                    pairs.insert((s, key), t);
                    pairs.insert((t, key), s);
                }
            }
        }
        Self { uses, pairs }
    }

    /// The triangle across the edge `key` from `t`: the other triangle of an
    /// edge of two, or the pair of `t` round an edge of more.
    fn across(&self, t: usize, key: (usize, usize)) -> Option<usize> {
        let list = &self.uses[&key];
        if list.len() == 2 {
            list.iter()
                .map(|&(other, _)| other)
                .find(|&other| other != t)
        } else {
            self.pairs.get(&(t, key)).copied()
        }
    }
}

/// For each triangle, whether it is wound against the side of the triangles
/// joined to it through edges of two triangles.
///
/// Such a set takes the side of most of its triangles as they are wound,
/// and turns over when its stored normals point against that side, as a
/// closed body does in [`split_bodies`].
fn patch_turns(
    tris: &[[usize; 3]],
    uses: &HashMap<(usize, usize), Vec<(usize, bool)>>,
    votes: &[f64],
) -> Vec<bool> {
    let mut turn = vec![false; tris.len()];
    let mut seen = vec![false; tris.len()];
    let mut patch = Vec::new();
    for seed in 0..tris.len() {
        if seen[seed] {
            continue;
        }
        patch.clear();
        patch.push(seed);
        seen[seed] = true;
        let mut next = 0;
        while next < patch.len() {
            let t = patch[next];
            next += 1;
            for k in 0..3 {
                let (a, b) = (tris[t][k], tris[t][(k + 1) % 3]);
                let list = &uses[&edge_key(a, b)];
                if list.len() != 2 {
                    continue;
                }
                let forward = (a < b) != turn[t];
                for &(other, other_forward) in list {
                    if !seen[other] {
                        seen[other] = true;
                        turn[other] = other_forward == forward;
                        patch.push(other);
                    }
                }
            }
        }
        if patch.iter().filter(|&&t| turn[t]).count() * 2 > patch.len() {
            for &t in &patch {
                turn[t] = !turn[t];
            }
        }
        let (mut vote, mut weight) = (0.0, 0.0);
        for &t in &patch {
            vote += if turn[t] { -votes[t] } else { votes[t] };
            weight += votes[t].abs();
        }
        if vote < -0.5 * weight {
            for &t in &patch {
                turn[t] = !turn[t];
            }
        }
    }
    turn
}

/// The pairs of triangles round an edge of more than two triangles, each
/// pair the two sides of one body at the edge. Empty when they do not pair
/// off.
///
/// Each triangle faces one of its two neighbours round the edge. Two
/// neighbours that run the edge the same way cannot be the two sides of one
/// body, so the pairs come from the rotation in which no pair does. When
/// both rotations fit, each pair takes the wedge that both of its triangles
/// face away from: the material of the body.
fn pair_round_edge(
    (lo, hi): (usize, usize),
    list: &[(usize, bool)],
    turned: &[bool],
    tris: &[[usize; 3]],
    points: &[Point3],
    tolerance: f64,
) -> Vec<[usize; 2]> {
    /// A triangle round the edge.
    struct Side {
        angle: f64,
        /// The distance from the edge to the far corner.
        reach: f64,
        /// Whether the triangle faces the way that the angle grows.
        up: bool,
        tri: usize,
    }

    let n = list.len();
    if n % 2 == 1 {
        return Vec::new();
    }
    let origin = points[lo];
    let Ok(axis) = (points[hi] - origin).normalize() else {
        return Vec::new();
    };
    let least = if axis.x().abs() <= axis.y().abs() && axis.x().abs() <= axis.z().abs() {
        Vec3::new(1.0, 0.0, 0.0)
    } else if axis.y().abs() <= axis.z().abs() {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let Ok(u) = axis.cross(least).normalize() else {
        return Vec::new();
    };
    // `u`, `v` and `axis` are right-handed, so a triangle that runs the
    // edge from `lo` to `hi` faces the way that the angle grows.
    let v = axis.cross(u);
    let mut round: Vec<Side> = list
        .iter()
        .map(|&(tri, forward)| {
            let far = tris[tri]
                .into_iter()
                .find(|&i| i != lo && i != hi)
                .unwrap_or(lo);
            let w = points[far] - origin;
            let (x, y) = (w.dot(u), w.dot(v));
            Side {
                angle: y.atan2(x),
                reach: x.hypot(y),
                up: forward != turned[tri],
                tri,
            }
        })
        .collect();
    round.sort_by(|a, b| a.angle.total_cmp(&b.angle));

    // The turn from one side to the next.
    let gap = |a: &Side, b: &Side| {
        let g = b.angle - a.angle;
        if g < 0.0 {
            g + std::f64::consts::TAU
        } else {
            g
        }
    };
    // Start after the widest gap, so that no run of sides at one angle
    // wraps round the end of the list.
    let widest = (0..n)
        .max_by(|&i, &j| {
            gap(&round[i], &round[(i + 1) % n]).total_cmp(&gap(&round[j], &round[(j + 1) % n]))
        })
        .unwrap_or(0);
    round.rotate_left((widest + 1) % n);
    // At one angle (within the tolerance at the far corners), the sides
    // that face up come first: two faces that touch hold no material
    // between them.
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n
            && gap(&round[end - 1], &round[end]) * round[end - 1].reach.min(round[end].reach)
                <= tolerance.max(0.0)
        {
            end += 1;
        }
        round[start..end].sort_by_key(|side| !side.up);
        start = end;
    }

    let fits = |shift: usize| {
        (0..n / 2).all(|m| round[(shift + 2 * m) % n].up != round[(shift + 2 * m + 1) % n].up)
    };
    let shift = match (fits(0), fits(1)) {
        (true, true) => usize::from(round[0].up),
        (true, false) => 0,
        (false, true) => 1,
        (false, false) => return Vec::new(),
    };
    (0..n / 2)
        .map(|m| {
            [
                round[(shift + 2 * m) % n].tri,
                round[(shift + 2 * m + 1) % n].tri,
            ]
        })
        .collect()
}

/// Group the triangles into bodies, in the order of each body's first
/// triangle, and turn the triangles of each body to one side.
///
/// - A closed body takes the side of most of its triangles as they are
///   wound. It is turned over when its stored normals, summed over the
///   body, point against that side. Normals that say nothing (one direction
///   for each triangle of a closed body sums to zero) leave the winding.
/// - In a body that does not close, a stored normal that points against a
///   triangle's winding turns the triangle.
///
/// A closed body has [`BodyKind::Solid`] here, whichever way it faces;
/// [`assign_cavities`] reads the sign of its volume.
fn split_bodies(
    tris: &mut [[usize; 3]],
    votes: &[f64],
    points: &[Point3],
    links: &Links,
) -> Vec<Body> {
    let mut body_of = vec![usize::MAX; tris.len()];
    let mut walked: HashSet<(usize, usize)> = HashSet::new();
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
            let t = members[next];
            next += 1;
            let tri = tris[t];
            for k in 0..3 {
                let key = edge_key(tri[k], tri[(k + 1) % 3]);
                if let Some(other) = links.across(t, key) {
                    if body_of[other] == usize::MAX {
                        body_of[other] = index;
                        members.push(other);
                    }
                } else if links.uses[&key].len() > 2 && walked.insert(key) {
                    // An edge whose triangles do not pair off joins them
                    // all, once, however many it has.
                    for &(other, _) in &links.uses[&key] {
                        if body_of[other] == usize::MAX {
                            body_of[other] = index;
                            members.push(other);
                        }
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
        let paired = body.tris.iter().all(|&t| {
            (0..3).all(|k| {
                links
                    .across(t, edge_key(tris[t][k], tris[t][(k + 1) % 3]))
                    .is_some()
            })
        });
        let turned = if paired {
            consistent_turns(&body.tris, tris, links)
        } else {
            None
        };
        let closed = turned.and_then(|mut turned| {
            // The body's stored normals against its one side.
            let is_turned: HashSet<usize> = turned.iter().copied().collect();
            let (mut vote, mut weight) = (0.0, 0.0);
            for &t in &body.tris {
                vote += if is_turned.contains(&t) {
                    -votes[t]
                } else {
                    votes[t]
                };
                weight += votes[t].abs();
            }
            if vote < -0.5 * weight {
                turned = body
                    .tris
                    .iter()
                    .copied()
                    .filter(|t| !is_turned.contains(t))
                    .collect();
            }
            for &t in &turned {
                tris[t].swap(1, 2);
            }
            // A closed set with no volume (two triangles back to back) is
            // a sheet. The sum of the sizes of the terms bounds the
            // rounding of their sum.
            let (volume6, terms) = signed_volume6(&body.tris, tris, points);
            if volume6.abs() > 1e-12 * terms {
                Some(volume6)
            } else {
                for &t in &turned {
                    tris[t].swap(1, 2);
                }
                None
            }
        });
        if let Some(volume6) = closed {
            body.kind = BodyKind::Solid;
            body.volume6 = volume6;
        } else {
            for &t in &body.tris {
                if votes[t] < 0.0 {
                    tris[t].swap(1, 2);
                }
            }
        }
    }
    bodies
}

/// The triangles of a closed body to turn so that each edge's two
/// triangles run it in opposite directions.
///
/// Of the two answers, this is the one that turns fewer triangles. Returns
/// `None` for a one-sided surface, which has no such answer.
fn consistent_turns(members: &[usize], tris: &[[usize; 3]], links: &Links) -> Option<Vec<usize>> {
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
            let key = edge_key(a, b);
            let forward = (a < b) != t_turn;
            let other = links.across(t, key)?;
            // The neighbour must run the edge the other way.
            let needs_turn = runs_up(tris[other], key) == forward;
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
    let turned: Vec<usize> = members.iter().copied().filter(|t| turn[t]).collect();
    if turned.len() * 2 > members.len() {
        Some(members.iter().copied().filter(|t| !turn[t]).collect())
    } else {
        Some(turned)
    }
}

/// Six times the signed volume of a closed body, and the sum of the sizes
/// of its terms.
fn signed_volume6(members: &[usize], tris: &[[usize; 3]], points: &[Point3]) -> (f64, f64) {
    // Measured from a point of the body, so a body far from the origin
    // keeps its digits.
    let origin = points[tris[members[0]][0]];
    let (mut sum, mut sizes) = (0.0, 0.0);
    for &t in members {
        let [a, b, c] = tris[t].map(|i| points[i] - origin);
        let term = a.dot(b.cross(c));
        sum += term;
        sizes += term.abs();
    }
    (sum, sizes)
}

/// An axis-aligned box: its low corner and its high corner.
type Bounds = ([f64; 3], [f64; 3]);

const fn coords(p: Point3) -> [f64; 3] {
    [p.x(), p.y(), p.z()]
}

fn point_bounds(points: impl IntoIterator<Item = Point3>) -> Bounds {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        let p = coords(p);
        for k in 0..3 {
            lo[k] = lo[k].min(p[k]);
            hi[k] = hi[k].max(p[k]);
        }
    }
    (lo, hi)
}

fn grown((lo, hi): Bounds, by: f64) -> Bounds {
    (lo.map(|c| c - by), hi.map(|c| c + by))
}

fn meets(a: &Bounds, b: &Bounds) -> bool {
    (0..3).all(|k| a.0[k] <= b.1[k] && b.0[k] <= a.1[k])
}

fn holds(outer: &Bounds, inner: &Bounds) -> bool {
    (0..3).all(|k| outer.0[k] <= inner.0[k] && inner.1[k] <= outer.1[k])
}

/// A tree of boxes. Each node splits its items at the middle one along
/// its longest side, so the tree is built in `n log n`.
struct BoxTree {
    nodes: Vec<BoxNode>,
    items: Vec<usize>,
    boxes: Vec<Bounds>,
}

#[derive(Clone, Copy)]
struct BoxNode {
    bounds: Bounds,
    /// A leaf holds `items[first..first + count]`; a node with no count has
    /// its two children at `first` and `first + 1`.
    first: usize,
    count: usize,
}

impl BoxTree {
    const LEAF: usize = 4;

    fn new(boxes: Vec<Bounds>) -> Self {
        let mut items: Vec<usize> = (0..boxes.len()).collect();
        let mut nodes = Vec::new();
        if boxes.is_empty() {
            return Self {
                nodes,
                items,
                boxes,
            };
        }
        let empty = BoxNode {
            bounds: boxes[0],
            first: 0,
            count: 0,
        };
        nodes.push(empty);
        let mut stack = vec![(0, 0, boxes.len())];
        while let Some((node, start, end)) = stack.pop() {
            let mut bounds = boxes[items[start]];
            for &i in &items[start + 1..end] {
                for k in 0..3 {
                    bounds.0[k] = bounds.0[k].min(boxes[i].0[k]);
                    bounds.1[k] = bounds.1[k].max(boxes[i].1[k]);
                }
            }
            if end - start <= Self::LEAF {
                nodes[node] = BoxNode {
                    bounds,
                    first: start,
                    count: end - start,
                };
                continue;
            }
            let side = |k: usize| bounds.1[k] - bounds.0[k];
            let axis = if side(0) >= side(1) && side(0) >= side(2) {
                0
            } else if side(1) >= side(2) {
                1
            } else {
                2
            };
            let mid = start + (end - start) / 2;
            items[start..end].select_nth_unstable_by(mid - start, |&a, &b| {
                (boxes[a].0[axis] + boxes[a].1[axis])
                    .total_cmp(&(boxes[b].0[axis] + boxes[b].1[axis]))
            });
            let first = nodes.len();
            nodes.extend([empty, empty]);
            nodes[node] = BoxNode {
                bounds,
                first,
                count: 0,
            };
            stack.push((first, start, mid));
            stack.push((first + 1, mid, end));
        }
        Self {
            nodes,
            items,
            boxes,
        }
    }

    /// Visit each item whose box passes `test`, while `visit` returns
    /// `true`. Returns `false` when `visit` stopped the walk.
    fn walk(&self, test: impl Fn(&Bounds) -> bool, mut visit: impl FnMut(usize) -> bool) -> bool {
        let mut stack = Vec::new();
        if !self.nodes.is_empty() {
            stack.push(0);
        }
        while let Some(n) = stack.pop() {
            let node = self.nodes[n];
            if !test(&node.bounds) {
                continue;
            }
            if node.count == 0 {
                stack.extend([node.first, node.first + 1]);
                continue;
            }
            for &item in &self.items[node.first..node.first + node.count] {
                if test(&self.boxes[item]) && !visit(item) {
                    return false;
                }
            }
        }
        true
    }

    /// Visit each item whose box meets `bounds`, while `visit` returns
    /// `true`.
    fn each_meeting(&self, bounds: &Bounds, visit: impl FnMut(usize) -> bool) -> bool {
        self.walk(|b| meets(b, bounds), visit)
    }

    /// Visit each item whose box the segment from `p` to `q` meets.
    fn each_along(&self, p: Point3, q: Point3, visit: impl FnMut(usize) -> bool) -> bool {
        let d = q - p;
        self.walk(|b| segment_entry(b, p, d).is_some(), visit)
    }
}

/// Where the segment from `p` to `p + d` enters the box `bounds`, as a part
/// of the segment, or `None` when it misses the box.
fn segment_entry((lo, hi): &Bounds, p: Point3, d: Vec3) -> Option<f64> {
    let (p, d) = (coords(p), [d.x(), d.y(), d.z()]);
    let (mut t0, mut t1) = (0.0_f64, 1.0_f64);
    for k in 0..3 {
        if d[k] == 0.0 {
            if p[k] < lo[k] || p[k] > hi[k] {
                return None;
            }
            continue;
        }
        let (a, b) = ((lo[k] - p[k]) / d[k], (hi[k] - p[k]) / d[k]);
        t0 = t0.max(a.min(b));
        t1 = t1.min(a.max(b));
        if t0 > t1 {
            return None;
        }
    }
    Some(t0)
}

/// How the closed bodies of a mesh sit together.
#[derive(Debug, Clone, Copy, Default)]
struct Placement {
    /// A solid of the result lies directly in an outward body: in its
    /// material, and not in a cavity of it.
    overlap: bool,
    /// Two closed bodies cross.
    crossing: bool,
}

/// Directions for the rays that find whether a point lies in a body. None
/// lies along an axis or a diagonal, where the edges of a mesh often are.
const RAYS: [[f64; 3]; 4] = [
    [1.0, 0.297_1, 0.261_3],
    [-0.267_9, 1.0, 0.141_4],
    [0.173_2, -0.223_6, 1.0],
    [-1.0, -0.414_2, -0.302_7],
];

/// Make each inward closed body a cavity of the outward body that it lies
/// directly in, a shell when it crosses another body, or turn it into a
/// solid when it lies in no outward body.
fn assign_cavities(
    bodies: &mut [Body],
    tris: &mut [[usize; 3]],
    points: &[Point3],
    links: &Links,
    tolerance: f64,
) -> Placement {
    let closed: Vec<usize> = (0..bodies.len())
        .filter(|&i| bodies[i].kind == BodyKind::Solid)
        .collect();
    let nesting = Nesting::read(&closed, bodies, tris, points, links, tolerance);

    // The cavities come from the winding of the mesh as it is.
    let outward: Vec<bool> = bodies.iter().map(|b| b.volume6 > 0.0).collect();
    for (c, &i) in closed.iter().enumerate() {
        if outward[i] {
            continue;
        }
        if nesting.crossed[c] {
            bodies[i].kind = BodyKind::Open;
        } else if let Some(j) = nesting.around[c].map(|d| closed[d]).filter(|&j| outward[j]) {
            bodies[i].kind = BodyKind::Cavity(j);
        } else {
            for &t in &bodies[i].tris {
                tris[t].swap(1, 2);
            }
            bodies[i].volume6 = -bodies[i].volume6;
        }
    }
    // The overlap comes from the result: a turned body is a solid too.
    let overlap = closed.iter().zip(&nesting.around).any(|(&i, around)| {
        bodies[i].kind == BodyKind::Solid
            && around.is_some_and(|d| bodies[closed[d]].kind == BodyKind::Solid)
    });
    Placement {
        overlap,
        crossing: nesting.crossed.contains(&true),
    }
}

/// The margins of the import at coordinates of a given size.
#[derive(Clone, Copy)]
struct Margins {
    tolerance: f64,
}

impl Margins {
    /// Two points within the merge tolerance are one point. With no
    /// tolerance, the margin is the rounding of a coordinate of size
    /// `reach`.
    fn on(self, reach: f64) -> f64 {
        if self.tolerance > 0.0 {
            self.tolerance
        } else {
            8.0 * f64::EPSILON * reach
        }
    }

    /// The margin of a box in a tree: [`Margins::on`], and the rounding of
    /// the box tests.
    fn pad(self, reach: f64) -> f64 {
        self.on(reach).max(64.0 * f64::EPSILON * reach)
    }
}

/// The largest size of a coordinate in `bounds`.
fn reach_of((lo, hi): &Bounds) -> f64 {
    lo.iter().chain(hi).fold(0.0_f64, |m, c| m.max(c.abs()))
}

/// Where the closed bodies of a mesh lie, each body by its place in the
/// list of closed bodies.
struct Nesting {
    /// The smallest closed body round each closed body: the one that it
    /// lies directly in.
    around: Vec<Option<usize>>,
    /// Whether each closed body crosses another.
    crossed: Vec<bool>,
}

impl Nesting {
    #[allow(clippy::too_many_lines)]
    fn read(
        closed: &[usize],
        bodies: &[Body],
        tris: &[[usize; 3]],
        points: &[Point3],
        links: &Links,
        tolerance: f64,
    ) -> Self {
        let n = closed.len();
        let mut out = Self {
            around: vec![None; n],
            crossed: vec![false; n],
        };
        if n < 2 {
            return out;
        }
        let margins = Margins { tolerance };
        let corners = |t: usize| tris[t].map(|i| points[i]);
        let boxes: Vec<Bounds> = closed
            .iter()
            .map(|&i| point_bounds(bodies[i].tris.iter().flat_map(|&t| corners(t))))
            .collect();
        let sizes: Vec<f64> = closed.iter().map(|&i| bodies[i].volume6.abs()).collect();
        let outward: Vec<bool> = closed.iter().map(|&i| bodies[i].volume6 > 0.0).collect();
        let body_tree = BoxTree::new(
            boxes
                .iter()
                .map(|b| grown(*b, margins.pad(reach_of(b))))
                .collect(),
        );

        // Only the triangles of a body whose box meets the box of another
        // can cross or hold another body.
        let near: Vec<bool> = (0..n)
            .map(|c| !body_tree.each_meeting(&body_tree.boxes[c], |d| d == c))
            .collect();
        let mut tri_of = Vec::new();
        let mut owner = Vec::new();
        for (c, &i) in closed.iter().enumerate() {
            if near[c] {
                tri_of.extend_from_slice(&bodies[i].tris);
                owner.resize(tri_of.len(), c);
            }
        }
        let tri_tree = BoxTree::new(
            tri_of
                .iter()
                .map(|&t| {
                    let b = point_bounds(corners(t));
                    grown(b, margins.pad(reach_of(&b)))
                })
                .collect(),
        );

        // A triangle, with the far corner of the triangle across each edge.
        let facet = |t: usize| {
            let tri = tris[t];
            Facet {
                corners: corners(t),
                beyond: [0, 1, 2].map(|k| {
                    let key = edge_key(tri[k], tri[(k + 1) % 3]);
                    let far = links
                        .across(t, key)
                        .and_then(|o| tris[o].into_iter().find(|&i| i != key.0 && i != key.1));
                    points[far.unwrap_or(tri[(k + 2) % 3])]
                }),
            }
        };
        let mut crossing: HashSet<(usize, usize)> = HashSet::new();
        for (item, &t) in tri_of.iter().enumerate() {
            let c = owner[item];
            let bounds = tri_tree.boxes[item];
            if body_tree.each_meeting(&bounds, |d| d == c) {
                continue;
            }
            tri_tree.each_meeting(&bounds, |other| {
                let d = owner[other];
                if d > c && !crossing.contains(&(c, d)) {
                    let reach = reach_of(&bounds).max(reach_of(&tri_tree.boxes[other]));
                    if triangles_cross(&facet(t), &facet(tri_of[other]), margins.on(reach)) {
                        crossing.insert((c, d));
                    }
                }
                true
            });
        }
        for &(c, d) in &crossing {
            out.crossed[c] = true;
            out.crossed[d] = true;
        }

        let mut scene = body_tree.boxes[0];
        for (c, b) in body_tree.boxes.iter().enumerate() {
            if near[c] {
                scene = point_bounds([
                    Point3::new(b.0[0], b.0[1], b.0[2]),
                    Point3::new(b.1[0], b.1[1], b.1[2]),
                    Point3::new(scene.0[0], scene.0[1], scene.0[2]),
                    Point3::new(scene.1[0], scene.1[1], scene.1[2]),
                ]);
            }
        }
        let rays = Rays {
            tri_tree: &tri_tree,
            tri_of: &tri_of,
            owner: &owner,
            corners: &corners,
            body_boxes: &body_tree.boxes,
            outward: &outward,
            margins,
        };
        for c in (0..n).filter(|&c| near[c]) {
            // A body that can hold this one: larger, with a box that holds
            // its box, and not across it.
            let holds_c = |d: usize| {
                d != c
                    && sizes[d] > sizes[c]
                    && holds(&body_tree.boxes[d], &boxes[c])
                    && !crossing.contains(&(c.min(d), c.max(d)))
            };
            if body_tree.each_meeting(&boxes[c], |d| !holds_c(d)) {
                continue;
            }
            let members = &bodies[closed[c]].tris;
            out.around[c] = match rays.first_round(c, members, &scene, &holds_c) {
                Some(Round::In(d)) => Some(d),
                Some(Round::Free) => None,
                // No ray decides: the bodies whose boxes hold this body's
                // box, smallest first, each with rays of its own.
                None => {
                    let mut candidates = BinaryHeap::new();
                    body_tree.each_meeting(&boxes[c], |d| {
                        if holds_c(d) {
                            candidates.push(std::cmp::Reverse((Size(sizes[d]), d)));
                        }
                        true
                    });
                    std::iter::from_fn(|| candidates.pop().map(|std::cmp::Reverse((_, d))| d))
                        .find(|&d| rays.lies_in(members, d))
                }
            };
        }
        out
    }
}

/// A size that orders by [`f64::total_cmp`].
#[derive(Clone, Copy, PartialEq)]
struct Size(f64);

impl Eq for Size {}

impl PartialOrd for Size {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Size {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// The rays that find whether a body lies in another.
struct Rays<'a, F: Fn(usize) -> [Point3; 3]> {
    tri_tree: &'a BoxTree,
    tri_of: &'a [usize],
    owner: &'a [usize],
    corners: &'a F,
    /// The box of each closed body, with the margin of the tree.
    body_boxes: &'a [Bounds],
    outward: &'a [bool],
    margins: Margins,
}

/// What a ray out of a closed body finds round it.
#[derive(Clone, Copy)]
enum Round {
    /// The body lies directly in this closed body.
    In(usize),
    /// The body lies in no closed body.
    Free,
}

/// A crossing of a ray with a triangle of a body.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Hit {
    /// The part of the ray before the crossing.
    at: Size,
    /// Two crossings closer than this may be read in the wrong order.
    tie: Size,
    body: usize,
    /// Whether the ray leaves the body here.
    exit: bool,
}

impl<F: Fn(usize) -> [Point3; 3]> Rays<'_, F> {
    /// The body round the closed body `c` (of the triangles `members`):
    /// the first body that a ray out of `c` leaves before it enters it.
    /// `holder` tells the bodies that can hold `c`.
    ///
    /// The ray starts at the corner of `c` furthest along it, so it meets
    /// no part of `c` and nothing in `c`. It reads the crossings in their
    /// order, so it stops at the first body that it leaves: the body round
    /// `c`. `None` when no ray decides.
    fn first_round(
        &self,
        c: usize,
        members: &[usize],
        scene: &Bounds,
        holder: &impl Fn(usize) -> bool,
    ) -> Option<Round> {
        for dir in RAYS {
            let dir = Vec3::new(dir[0], dir[1], dir[2]);
            let along = |p: &Point3| dir.dot(Vec3::new(p.x(), p.y(), p.z()));
            let p = members
                .iter()
                .flat_map(|&t| (self.corners)(t))
                .max_by(|a, b| along(a).total_cmp(&along(b)))?;
            if self.touches(p, None, c) {
                continue;
            }
            if let Some(found) = self.walk(c, p, dir, scene, holder) {
                return Some(found);
            }
        }
        None
    }

    /// The first body that the ray from `p` along `dir` leaves before it
    /// enters it, of the bodies that `holder` accepts. `None` when the ray
    /// meets an edge or a corner, or two crossings of one body are too
    /// close to read their order.
    fn walk(
        &self,
        c: usize,
        p: Point3,
        dir: Vec3,
        scene: &Bounds,
        holder: &impl Fn(usize) -> bool,
    ) -> Option<Round> {
        let q = p + dir * exit_along(p, dir, scene);
        let d = q - p;
        let length = d.length();
        if length == 0.0 {
            return None;
        }
        let tree = self.tri_tree;
        let mut nodes = BinaryHeap::new();
        if let Some(t) = tree
            .nodes
            .first()
            .and_then(|n| segment_entry(&n.bounds, p, d))
        {
            nodes.push(std::cmp::Reverse((Size(t), 0)));
        }
        let mut hits: BinaryHeap<std::cmp::Reverse<Hit>> = BinaryHeap::new();
        let mut crossings: HashMap<usize, Vec<f64>> = HashMap::new();
        let mut read: HashSet<usize> = HashSet::new();
        loop {
            let next_node = nodes.peek().map(|std::cmp::Reverse((Size(t), _))| *t);
            let next_hit = hits
                .peek()
                .map(|std::cmp::Reverse(hit)| hit.at.0 + hit.tie.0);
            match (next_node, next_hit) {
                // Each node that the ray enters before the next crossing
                // (and its tie) is read first, so the crossings come in
                // their order.
                (Some(t), h) if h.is_none_or(|h| t <= h) => {
                    let std::cmp::Reverse((_, n)) = nodes.pop()?;
                    let node = tree.nodes[n];
                    if node.count == 0 {
                        for child in [node.first, node.first + 1] {
                            if let Some(t) = segment_entry(&tree.nodes[child].bounds, p, d) {
                                nodes.push(std::cmp::Reverse((Size(t), child)));
                            }
                        }
                        continue;
                    }
                    for &item in &tree.items[node.first..node.first + node.count] {
                        let body = self.owner[item];
                        if body == c || segment_entry(&tree.boxes[item], p, d).is_none() {
                            continue;
                        }
                        let tri = (self.corners)(self.tri_of[item]);
                        if !segment_crosses(p, q, tri)? {
                            continue;
                        }
                        let [a, b, e] = tri;
                        let normal = (b - a).cross(e - a);
                        let across = normal.dot(d);
                        // A ray close to the plane of a triangle reads the
                        // place of its crossing poorly.
                        if across.abs() <= 1e-9 * normal.length() * length {
                            return None;
                        }
                        let at = normal.dot(a - p) / across;
                        let reach = reach_of(&tree.boxes[item]).max(reach_of(&point_bounds([p])));
                        crossings.entry(body).or_default().push(at);
                        hits.push(std::cmp::Reverse(Hit {
                            at: Size(at),
                            tie: Size(self.margins.pad(reach) / length),
                            body,
                            // `p` is behind the triangle, inside an outward
                            // body, when the sign is positive.
                            exit: (orient3d(a, b, e, p) > 0.0) == self.outward[body],
                        }));
                    }
                }
                (_, Some(_)) => {
                    let std::cmp::Reverse(hit) = hits.pop()?;
                    // The first crossing of a body tells whether it holds
                    // `p`.
                    if !read.insert(hit.body) {
                        continue;
                    }
                    let close = crossings[&hit.body]
                        .iter()
                        .filter(|&&t| (t - hit.at.0).abs() <= hit.tie.0)
                        .count();
                    if close > 1 {
                        return None;
                    }
                    if hit.exit && holder(hit.body) {
                        // The count of crossings of that body alone, with
                        // exact signs, has the last word.
                        return self
                            .crosses_odd(p, dir, hit.body)
                            .filter(|&odd| odd)
                            .map(|_| Round::In(hit.body));
                    }
                }
                _ => return Some(Round::Free),
            }
        }
    }

    /// Whether the closed body of the triangles `members` lies in the
    /// closed body `outer`.
    ///
    /// One point of `members` decides, from the count of times a ray from
    /// it crosses `outer`. A point within the margin of `outer` (the two
    /// bodies touch there) decides nothing, and the next point is read:
    /// each corner, then the centre of each triangle. A ray that meets an
    /// edge or a corner of `outer` decides nothing, and the next ray is
    /// cast.
    fn lies_in(&self, members: &[usize], outer: usize) -> bool {
        let mut read: HashSet<Point3Key> = HashSet::new();
        let corners = members.iter().flat_map(|&t| (self.corners)(t));
        let centres = members.iter().map(|&t| {
            let [a, b, c] = (self.corners)(t);
            Point3::new(
                (a.x() + b.x() + c.x()) / 3.0,
                (a.y() + b.y() + c.y()) / 3.0,
                (a.z() + b.z() + c.z()) / 3.0,
            )
        });
        for p in corners.chain(centres) {
            if !read.insert(Point3Key::of(p)) || self.touches(p, Some(outer), usize::MAX) {
                continue;
            }
            for dir in RAYS {
                if let Some(odd) = self.crosses_odd(p, Vec3::new(dir[0], dir[1], dir[2]), outer) {
                    return odd;
                }
            }
        }
        false
    }

    /// Whether `p` lies within the margin of a triangle of the body `only`,
    /// or of any body but `not` when `only` is `None`.
    fn touches(&self, p: Point3, only: Option<usize>, not: usize) -> bool {
        let reach = reach_of(&point_bounds([p]));
        let at = grown(point_bounds([p]), self.margins.pad(reach));
        !self.tri_tree.each_meeting(&at, |item| {
            let body = self.owner[item];
            if only.map_or(body == not, |o| body != o) {
                return true;
            }
            let [a, b, c] = (self.corners)(self.tri_of[item]);
            let on = self
                .margins
                .on(reach.max(reach_of(&self.tri_tree.boxes[item])));
            origin_to_triangle(a - p, b - p, c - p) > on
        })
    }

    /// Whether a ray from `p` along `dir` crosses the triangles of `outer`
    /// an odd count of times, or `None` when it meets an edge or a corner.
    ///
    /// The ray stops past the box of `outer`, so it reads only the
    /// triangles near that body.
    fn crosses_odd(&self, p: Point3, dir: Vec3, outer: usize) -> Option<bool> {
        let q = p + dir * exit_along(p, dir, &self.body_boxes[outer]);
        let mut odd = false;
        let mut sure = true;
        self.tri_tree.each_along(p, q, |item| {
            if self.owner[item] != outer {
                return true;
            }
            match segment_crosses(p, q, (self.corners)(self.tri_of[item])) {
                Some(cross) => odd ^= cross,
                None => sure = false,
            }
            sure
        });
        sure.then_some(odd)
    }
}

/// How far along `dir` the ray from `p` leaves the box `bounds`, which
/// holds `p`.
fn exit_along(p: Point3, dir: Vec3, (lo, hi): &Bounds) -> f64 {
    let (from, along) = (coords(p), [dir.x(), dir.y(), dir.z()]);
    (0..3)
        .map(|k| {
            let wall = if along[k] > 0.0 { hi[k] } else { lo[k] };
            (wall - from[k]) / along[k]
        })
        .fold(f64::INFINITY, f64::min)
        .max(0.0)
}

/// The bits of a point, to read each point once.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Point3Key([u64; 3]);

impl Point3Key {
    fn of(p: Point3) -> Self {
        Self(coords(p).map(f64::to_bits))
    }
}

/// Whether the segment `p q` passes through the triangle `a b c`, with
/// exact signs. `None` when the segment meets an edge or a corner of the
/// triangle, or runs in its plane: there the count of crossings has no
/// meaning. The two ends must be off the triangle.
fn segment_crosses(p: Point3, q: Point3, [a, b, c]: [Point3; 3]) -> Option<bool> {
    let (sp, sq) = (orient3d(a, b, c, p), orient3d(a, b, c, q));
    if sp == 0.0 && sq == 0.0 {
        let flat = |f: fn(Point3) -> Point2| orient2d(f(a), f(b), f(c)) == 0.0;
        let on_a_line = flat(|p| Point2::new(p.x(), p.y()))
            && flat(|p| Point2::new(p.y(), p.z()))
            && flat(|p| Point2::new(p.z(), p.x()));
        // A triangle with its corners on a line has no inside to cross.
        return if on_a_line { Some(false) } else { None };
    }
    // An end in the plane of the triangle is off the triangle, so the
    // segment meets the plane there and nowhere in the triangle.
    if sp == 0.0 || sq == 0.0 || (sp > 0.0) == (sq > 0.0) {
        return Some(false);
    }
    let sides = [
        orient3d(p, q, a, b),
        orient3d(p, q, b, c),
        orient3d(p, q, c, a),
    ];
    let above = sides.iter().any(|&s| s > 0.0);
    let below = sides.iter().any(|&s| s < 0.0);
    if above && below {
        Some(false)
    } else if sides.contains(&0.0) {
        None
    } else {
        Some(true)
    }
}

/// A triangle of a closed body, with the far corner of the triangle across
/// each of its edges: `beyond[k]` is across the edge from corner `k` to
/// corner `k + 1`.
struct Facet {
    corners: [Point3; 3],
    beyond: [Point3; 3],
}

/// Whether two triangles of two closed bodies cross: the surface of each
/// body passes through the plane of the other triangle there, and the two
/// meet along a length of more than `on`. Two triangles that touch at a
/// point, along an edge, or in one plane do not cross.
fn triangles_cross(t: &Facet, u: &Facet, on: f64) -> bool {
    let (Some(cut_t), Some(cut_u)) = (
        cut_by_plane(t, &u.corners, on),
        cut_by_plane(u, &t.corners, on),
    ) else {
        return false;
    };
    let [t0, t1, t2] = t.corners;
    let [u0, u1, u2] = u.corners;
    let n_t = (t1 - t0).cross(t2 - t0);
    let n_u = (u1 - u0).cross(u2 - u0);
    let Ok(line) = n_t.cross(n_u).normalize() else {
        return false;
    };
    let span = |(a, b): (Point3, Point3)| {
        let (x, y) = (line.dot(a - t0), line.dot(b - t0));
        (x.min(y), x.max(y))
    };
    let ((a0, a1), (b0, b1)) = (span(cut_t), span(cut_u));
    a1.min(b1) - a0.max(b0) > on
}

/// Where the surface of the body of `facet` passes through the plane of
/// the triangle `other`, in `facet`, by its two ends.
///
/// - When `facet` passes more than `on` to each side of the plane, the
///   part of `facet` in the plane.
/// - When an edge of `facet` lies in the plane, and `facet` and the
///   triangle across that edge are more than `on` to the two sides of the
///   plane, that edge.
fn cut_by_plane(facet: &Facet, other: &[Point3; 3], on: f64) -> Option<(Point3, Point3)> {
    let tri = facet.corners;
    let normal = (other[1] - other[0])
        .cross(other[2] - other[0])
        .normalize()
        .ok()?;
    let side = |p: Point3| normal.dot(p - other[0]);
    let d = tri.map(side);
    let (low, high) = (d[0].min(d[1]).min(d[2]), d[0].max(d[1]).max(d[2]));
    if high > on && low < -on {
        let mut ends = Vec::with_capacity(2);
        for k in 0..3 {
            let m = (k + 1) % 3;
            if d[k] == 0.0 {
                ends.push(tri[k]);
            } else if d[m] != 0.0 && (d[k] > 0.0) != (d[m] > 0.0) {
                ends.push(tri[k] + (tri[m] - tri[k]) * (d[k] / (d[k] - d[m])));
            }
        }
        return match ends[..] {
            [a, b] => Some((a, b)),
            _ => None,
        };
    }
    (0..3).find_map(|k| {
        let (m, far) = ((k + 1) % 3, (k + 2) % 3);
        let beyond = side(facet.beyond[k]);
        (d[k].abs() <= on
            && d[m].abs() <= on
            && d[far].abs() > on
            && beyond.abs() > on
            && (beyond > 0.0) != (d[far] > 0.0))
            .then_some((tri[k], tri[m]))
    })
}

/// The distance from the origin to the triangle `a b c`.
fn origin_to_triangle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let (ab, ac) = (b - a, c - a);
    // The region of the triangle's plane that holds the foot of the
    // origin: a corner, an edge, or the inside.
    let (d1, d2) = (-ab.dot(a), -ac.dot(a));
    if d1 <= 0.0 && d2 <= 0.0 {
        return a.length();
    }
    let (d3, d4) = (-ab.dot(b), -ac.dot(b));
    if d3 >= 0.0 && d4 <= d3 {
        return b.length();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return (a + ab * (d1 / (d1 - d3))).length();
    }
    let (d5, d6) = (-ab.dot(c), -ac.dot(c));
    if d6 >= 0.0 && d5 <= d6 {
        return c.length();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return (a + ac * (d2 / (d2 - d6))).length();
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w).length();
    }
    let denom = 1.0 / (va + vb + vc);
    (a + ab * (vb * denom) + ac * (vc * denom)).length()
}

/// Build a triangular planar face on three points, with the edges that
/// earlier triangles of the body made on the same points. `tags` tells
/// apart the edges of the pairs round an edge of more than two triangles.
fn build_triangle_face(
    topo: &mut Topology,
    [p0, p1, p2]: [Point3; 3],
    tri: [usize; 3],
    corners: [VertexId; 3],
    tags: [usize; 3],
    edges: &mut HashMap<(usize, usize, usize), EdgeId>,
) -> Result<FaceId, IoError> {
    let mut oriented = Vec::with_capacity(3);
    for k in 0..3 {
        let n = (k + 1) % 3;
        let forward = tri[k] < tri[n];
        let (lo, hi) = if forward { (k, n) } else { (n, k) };
        let edge = *edges
            .entry((tri[lo], tri[hi], tags[k]))
            .or_insert_with(|| topo.add_edge(Edge::new(corners[lo], corners[hi], EdgeCurve::Line)));
        oriented.push(OrientedEdge::new(edge, forward));
    }
    let wire = Wire::new(oriented, true).map_err(|e| IoError::ParseError {
        reason: format!("failed to build triangle wire: {e}"),
    })?;
    let wire_id = topo.add_wire(wire);

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

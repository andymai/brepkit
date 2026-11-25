//! Mesh validation and boundary operations.

use brepkit_math::det_hash::{DetHashMap, DetHashSet, WordHashMap, WordState};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::face::FaceSurface;
use brepkit_topology::solid::SolidId;

use super::TriangleMesh;
use super::edge_sampling::sample_edge;

/// 1µm position-quantization grid for coincident-triangle dedupe: tight
/// enough that legitimately distinct CAD features (down to ~10µm geometry
/// like thin plates) keep separate keys, while still merging post-merge
/// floating-point noise in coincident vertices that boundary-vertex welding
/// didn't catch. Shared with the mesh-boolean output self-check so both
/// measure manifoldness on the same weld grid.
pub const COINCIDENT_DEDUPE_GRID: f64 = 1e-6;

/// Check if a mesh is a closed 2-manifold.
///
/// Returns `true` iff every edge is shared by exactly 2 triangles: no gaps
/// (boundary edges), no branching (non-manifold edges). Useful for
/// validating that `tessellate_solid` produces watertight meshes suitable
/// for slicers and downstream geometric operations.
#[must_use]
pub fn is_watertight(mesh: &TriangleMesh) -> bool {
    boundary_edge_count(mesh) == 0 && non_manifold_edge_count(mesh) == 0
}

/// Count boundary (one-sided) edges in a mesh.
///
/// A boundary edge is one where the half-edge `(a, b)` exists but `(b, a)`
/// does not. Returns the number of such edges. A watertight mesh has 0.
#[must_use]
pub fn boundary_edge_count(mesh: &TriangleMesh) -> usize {
    let mut half_edges: DetHashSet<(u32, u32)> = DetHashSet::default();
    let tri_count = mesh.indices.len() / 3;

    for t in 0..tri_count {
        let i0 = mesh.indices[t * 3];
        let i1 = mesh.indices[t * 3 + 1];
        let i2 = mesh.indices[t * 3 + 2];
        half_edges.insert((i0, i1));
        half_edges.insert((i1, i2));
        half_edges.insert((i2, i0));
    }

    half_edges
        .iter()
        .filter(|&&(a, b)| !half_edges.contains(&(b, a)))
        .count()
}

/// Count non-manifold (branching) edges in a mesh.
///
/// An undirected edge `{a, b}` is non-manifold when 3 or more triangles
/// reference it. A 2-manifold mesh has 0 such edges. Distinct from
/// [`boundary_edge_count`], which counts 1-sided edges. Use both together
/// to validate that a tessellated solid is a closed 2-manifold.
#[must_use]
pub fn non_manifold_edge_count(mesh: &TriangleMesh) -> usize {
    let mut edge_count: DetHashMap<(u32, u32), u32> = DetHashMap::default();
    for tri in mesh.indices.chunks_exact(3) {
        let (a, b, c) = (tri[0], tri[1], tri[2]);
        for (p, q) in [(a, b), (b, c), (c, a)] {
            let key = if p < q { (p, q) } else { (q, p) };
            *edge_count.entry(key).or_default() += 1;
        }
    }

    edge_count.values().filter(|&&c| c > 2).count()
}

/// Split the chords two faces both lay between the same two vertices: where
/// the faces pass within the deflection of each other (a rod's wall 0.004
/// inside a ball, between the two curves where it leaves the ball), each
/// face's mesh bridges the narrow gap with the same straight edge, which then
/// bounds four triangles. One face's pair is split at its own surface point
/// over the edge's midpoint (`on_face`: the face index and a point to the
/// point of that face's surface over it), taking the face whose surface
/// stands farther off the chord, and the two sheets part there.
pub(super) fn split_pinched_chords(
    mesh: &mut TriangleMesh,
    tri_faces: &mut Vec<u32>,
    on_face: &dyn Fn(u32, Point3) -> Option<Point3>,
) {
    let key = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    // Each round splits the pinched chords whose triangles no earlier split
    // in the round touched, then the edges are read again. The edges are
    // found by sorting packed keys: a mesh without a pinch costs one sort.
    for _ in 0..8 {
        let mut keyed: Vec<(u64, usize)> = Vec::with_capacity(mesh.indices.len());
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            for k in 0..3 {
                let (a, b) = key(tri[k], tri[(k + 1) % 3]);
                keyed.push(((u64::from(a) << 32) | u64::from(b), t));
            }
        }
        keyed.sort_unstable();
        let mut pinched: Vec<((u32, u32), Vec<usize>)> = Vec::new();
        let mut i = 0;
        while i < keyed.len() {
            let j = i + keyed[i..].iter().take_while(|k| k.0 == keyed[i].0).count();
            if j - i == 4 {
                #[allow(clippy::cast_possible_truncation)]
                let edge = ((keyed[i].0 >> 32) as u32, keyed[i].0 as u32);
                pinched.push((edge, keyed[i..j].iter().map(|k| k.1).collect()));
            }
            i = j;
        }
        if pinched.is_empty() {
            break;
        }
        let mut touched: DetHashSet<usize> = DetHashSet::default();
        let mut split_any = false;
        for ((a, b), tris) in pinched {
            if tris.iter().any(|t| touched.contains(t)) {
                continue;
            }
            let faces: Vec<u32> = tris.iter().map(|&t| tri_faces[t]).collect();
            let f0 = faces[0];
            let Some(f1) = faces.iter().copied().find(|&f| f != f0) else {
                continue;
            };
            let count = |f: u32| faces.iter().filter(|&&g| g == f).count();
            if count(f0) != 2 || count(f1) != 2 {
                continue;
            }
            let (pa, pb) = (mesh.positions[a as usize], mesh.positions[b as usize]);
            let mid = Point3::new(
                0.5 * (pa.x() + pb.x()),
                0.5 * (pa.y() + pb.y()),
                0.5 * (pa.z() + pb.z()),
            );
            // A surface point farther off than half the chord is not the
            // face's own over it (a projection that failed).
            let len = (pb - pa).length();
            let Some((face, q)) = [f0, f1]
                .into_iter()
                .filter_map(|f| on_face(f, mid).map(|q| (f, (q - mid).length(), q)))
                .filter(|&(_, gap, _)| gap > 1e-9 * len && gap <= 0.5 * len)
                .max_by(|x, y| x.1.total_cmp(&y.1))
                .map(|(f, _, q)| (f, q))
            else {
                continue;
            };
            #[allow(clippy::cast_possible_truncation)]
            let m = mesh.positions.len() as u32;
            mesh.positions.push(q);
            let n = mesh.normals[a as usize] + mesh.normals[b as usize];
            mesh.normals
                .push(n.normalize().unwrap_or(mesh.normals[a as usize]));
            let own: Vec<usize> = tris
                .iter()
                .copied()
                .filter(|&t| tri_faces[t] == face)
                .collect();
            for t in own {
                let tri = [
                    mesh.indices[3 * t],
                    mesh.indices[3 * t + 1],
                    mesh.indices[3 * t + 2],
                ];
                // The edge runs from tri[k] to tri[k + 1] in the triangle's
                // own winding, which both halves keep.
                let Some(k) =
                    (0..3).find(|&k| key(tri[k], tri[(k + 1) % 3]) == (a.min(b), a.max(b)))
                else {
                    continue;
                };
                let (x, y, z) = (tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]);
                mesh.indices[3 * t..3 * t + 3].copy_from_slice(&[x, m, z]);
                mesh.indices.extend_from_slice(&[m, y, z]);
                tri_faces.push(face);
            }
            touched.extend(tris);
            split_any = true;
        }
        if !split_any {
            break;
        }
    }
}

/// Remove duplicate triangles, cancelling opposing pairs and dedup same-winding pairs.
///
/// Workaround for issue #696: when a boolean leaves overlapping coplanar faces
/// in its output (the GFA path can do this without breaking Euler), tessellating
/// each face independently produces multiple triangles on the same 3D positions.
/// Slicers see this as branching (an edge shared by 3+ triangles), then "repair"
/// it by dropping pieces — turning hollow baseplates into solid blocks.
///
/// Triangles are keyed by their **quantized vertex positions** (sorted), not
/// global vertex IDs — boundary-vertex welding only runs on edges that already
/// look like boundaries, so two coplanar interior overlaps can survive with
/// distinct IDs at coincident positions. Pairs with matching winding
/// (sort-permutation parity equal) deduplicate to one triangle; pairs with
/// opposite winding cancel (both removed) — that's the signature of two faces
/// tessellated from opposite sides of the same plane.
/// `tri_faces` is a parallel tri -> face attribution array (one entry per
/// triangle); entries for removed triangles are filtered alongside so group
/// offsets recomputed from it stay aligned.
pub(super) fn dedupe_coincident_triangles(
    mesh: &mut TriangleMesh,
    tri_faces: Option<&mut Vec<u32>>,
) {
    const POS_GRID: f64 = COINCIDENT_DEDUPE_GRID;

    type TriKey = [u32; 3];

    let tri_count = mesh.indices.len() / 3;
    if tri_count < 2 {
        return;
    }

    let s = 1.0 / POS_GRID;
    // Each vertex reads as the first vertex at its grid point, so a key of
    // three such ids is the key of three grid points.
    let mut first_at: WordHashMap<(i64, i64, i64), u32> =
        WordHashMap::with_capacity_and_hasher(mesh.positions.len(), WordState);
    #[allow(clippy::cast_possible_truncation)]
    let quantized: Vec<u32> = mesh
        .positions
        .iter()
        .enumerate()
        .map(|(v, p)| {
            let at = (
                (p.x() * s).round() as i64,
                (p.y() * s).round() as i64,
                (p.z() * s).round() as i64,
            );
            *first_at.entry(at).or_insert(v as u32)
        })
        .collect();

    // Each triangle's key, and the count of keys at each least grid point:
    // the triangles of one key share their least point.
    let mut keys: Vec<Option<TriKey>> = vec![None; tri_count];
    let mut parity = vec![true; tri_count];
    let mut start = vec![0_usize; mesh.positions.len() + 1];
    for t in 0..tri_count {
        let mut tri_pts = [0, 1, 2].map(|k| quantized[mesh.indices[t * 3 + k] as usize]);
        // Sort tri_pts ascending; track parity of the sort permutation.
        let mut parity_even = true;
        if tri_pts[0] > tri_pts[1] {
            tri_pts.swap(0, 1);
            parity_even = !parity_even;
        }
        if tri_pts[1] > tri_pts[2] {
            tri_pts.swap(1, 2);
            parity_even = !parity_even;
        }
        if tri_pts[0] > tri_pts[1] {
            tri_pts.swap(0, 1);
            parity_even = !parity_even;
        }
        // Skip degenerate triangles (collapsed to <3 distinct positions).
        if tri_pts[0] == tri_pts[1] || tri_pts[1] == tri_pts[2] {
            continue;
        }
        parity[t] = parity_even;
        keys[t] = Some(tri_pts);
        start[tri_pts[0] as usize + 1] += 1;
    }
    for v in 1..start.len() {
        start[v] += start[v - 1];
    }
    let mut by_least = vec![0_usize; start[start.len() - 1]];
    let mut fill = start.clone();
    for (t, key) in keys.iter().enumerate() {
        if let Some(key) = key {
            let at = &mut fill[key[0] as usize];
            by_least[*at] = t;
            *at += 1;
        }
    }
    // For each triangle the one before it with its key: the triangles of a
    // key form a chain, read back to front.
    let mut earlier = vec![usize::MAX; tri_count];
    let mut repeated = false;
    let key_of = |t: usize| keys[t].unwrap_or([u32::MAX; 3]);
    for (&from, &to) in start.iter().zip(&start[1..]) {
        let bucket = &mut by_least[from..to];
        if bucket.len() < 2 {
            continue;
        }
        bucket.sort_unstable_by_key(|&t| (key_of(t), t));
        for pair in bucket.windows(2) {
            if key_of(pair[0]) == key_of(pair[1]) {
                earlier[pair[1]] = pair[0];
                repeated = true;
            }
        }
    }
    if !repeated {
        return;
    }

    let mut keep = vec![true; tri_count];
    let mut grouped = vec![false; tri_count];
    let mut tris: Vec<usize> = Vec::new();
    for last in (0..tri_count).rev() {
        if earlier[last] == usize::MAX || grouped[last] {
            continue;
        }
        tris.clear();
        let mut t = last;
        loop {
            grouped[t] = true;
            tris.push(t);
            if earlier[t] == usize::MAX {
                break;
            }
            t = earlier[t];
        }
        tris.reverse();
        let (even, odd): (Vec<usize>, Vec<usize>) = tris.iter().partition(|&&t| parity[t]);
        let cancel_pairs = even.len().min(odd.len());
        for &t in even.iter().take(cancel_pairs) {
            keep[t] = false;
        }
        for &t in odd.iter().take(cancel_pairs) {
            keep[t] = false;
        }
        // Of the surviving same-winding triangles, keep only one.
        for &t in even.iter().skip(cancel_pairs).skip(1) {
            keep[t] = false;
        }
        for &t in odd.iter().skip(cancel_pairs).skip(1) {
            keep[t] = false;
        }
    }

    if keep.iter().all(|&k| k) {
        return;
    }

    let mut new_indices = Vec::with_capacity(mesh.indices.len());
    let mut new_tri_faces = Vec::with_capacity(tri_faces.as_ref().map_or(0, |tf| tf.len()));
    for (t, &k) in keep.iter().enumerate().take(tri_count) {
        if k {
            new_indices.extend_from_slice(&mesh.indices[t * 3..t * 3 + 3]);
            if let Some(&f) = tri_faces.as_ref().and_then(|tf| tf.get(t)) {
                new_tri_faces.push(f);
            }
        }
    }
    if let Some(tf) = tri_faces {
        *tf = new_tri_faces;
    }

    // Compact the position/normal buffers: drop any vertex no longer
    // referenced by a surviving triangle. Downstream consumers that iterate
    // `mesh.positions` directly (e.g. bbox passes, exporters that walk
    // vertices rather than triangle indices) would otherwise see phantom
    // vertices from removed triangles.
    let n_verts = mesh.positions.len();
    let mut remap: Vec<u32> = vec![u32::MAX; n_verts];
    let mut new_positions: Vec<Point3> = Vec::new();
    let mut new_normals: Vec<Vec3> = Vec::new();
    for idx in &mut new_indices {
        let old = *idx as usize;
        if remap[old] == u32::MAX {
            #[allow(clippy::cast_possible_truncation)]
            let new_id = new_positions.len() as u32;
            remap[old] = new_id;
            new_positions.push(mesh.positions[old]);
            if old < mesh.normals.len() {
                new_normals.push(mesh.normals[old]);
            }
        }
        *idx = remap[old];
    }
    mesh.indices = new_indices;
    mesh.positions = new_positions;
    mesh.normals = new_normals;
}

/// Edge polyline data for wireframe visualization.
///
/// Contains flattened position data for all edges in a solid, plus offsets
/// to identify where each edge's polyline starts.
#[derive(Debug, Clone, Default)]
pub struct EdgeLines {
    /// Vertex positions for all edge polylines (concatenated).
    pub positions: Vec<Point3>,
    /// Start index (in vertex count, not float count) of each edge polyline.
    /// The i-th edge's points are `positions[offsets[i]..offsets[i+1]]`
    /// (or `..positions.len()` for the last edge).
    pub offsets: Vec<usize>,
}

/// Check whether two face surfaces represent the same geometric surface.
fn surfaces_equivalent(a: &FaceSurface, b: &FaceSurface) -> bool {
    let tol = brepkit_math::tolerance::Tolerance::new();
    let lin = tol.linear;
    let ang = tol.angular;

    match (a, b) {
        (FaceSurface::Plane { normal: na, d: da }, FaceSurface::Plane { normal: nb, d: db }) => {
            let dot = na.dot(*nb);
            (dot.abs() - 1.0).abs() < ang && (da - db * dot.signum()).abs() < lin
        }
        (FaceSurface::Cylinder(ca), FaceSurface::Cylinder(cb)) => {
            (ca.radius() - cb.radius()).abs() < lin
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.origin() - ca.origin();
                    let cross = d.cross(ca.axis());
                    cross.dot(cross) < lin * lin
                }
        }
        (FaceSurface::Cone(ca), FaceSurface::Cone(cb)) => {
            (ca.half_angle() - cb.half_angle()).abs() < ang
                && ca.axis().dot(cb.axis()).abs() > 1.0 - ang
                && {
                    let d = cb.apex() - ca.apex();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Sphere(sa), FaceSurface::Sphere(sb)) => {
            (sa.radius() - sb.radius()).abs() < lin && {
                let d = sb.center() - sa.center();
                d.dot(d) < lin * lin
            }
        }
        (FaceSurface::Torus(ta), FaceSurface::Torus(tb)) => {
            (ta.major_radius() - tb.major_radius()).abs() < lin
                && (ta.minor_radius() - tb.minor_radius()).abs() < lin
                && ta.z_axis().dot(tb.z_axis()).abs() > 1.0 - ang
                && {
                    let d = tb.center() - ta.center();
                    d.dot(d) < lin * lin
                }
        }
        (FaceSurface::Nurbs(_), FaceSurface::Nurbs(_)) => false,
        _ => false,
    }
}

/// Sample all edges of a solid into polylines for wireframe rendering.
///
/// Each edge is sampled according to the given `deflection` tolerance.
/// Returns [`EdgeLines`] containing the polyline data for all unique edges.
///
/// # Errors
///
/// Returns an error if topology traversal or edge sampling fails.
pub fn sample_solid_edges(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
) -> Result<EdgeLines, crate::OperationsError> {
    sample_solid_edges_filtered(
        topo,
        solid,
        deflection,
        brepkit_math::chord::DEFAULT_ANGULAR_TOL,
        true,
    )
}

/// Sample edges of a solid, optionally filtering out smooth (co-surface) edges.
///
/// When `filter_smooth` is `true`, edges shared by two faces on the same
/// underlying geometric surface are omitted. These edges arise from boolean
/// face-splitting and add wireframe clutter without representing visible creases.
///
/// `angular_tol` caps the per-segment turn angle when discretizing curved
/// edges (circles, ellipses, NURBS); a smaller value yields smoother polylines.
/// Pass [`brepkit_math::chord::DEFAULT_ANGULAR_TOL`] for the historical default.
///
/// # Errors
///
/// Returns an error if topology traversal or edge sampling fails.
pub fn sample_solid_edges_filtered(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
    filter_smooth: bool,
) -> Result<EdgeLines, crate::OperationsError> {
    let edges = drawn_edge_samples(topo, solid, deflection, angular_tol, filter_smooth)?;
    let mut result = EdgeLines {
        positions: Vec::new(),
        offsets: Vec::with_capacity(edges.len()),
    };
    for (_, points) in edges {
        result.offsets.push(result.positions.len());
        result.positions.extend(points);
    }
    Ok(result)
}

/// Each edge of `solid` a drawing shows, with its samples: with
/// `filter_smooth`, an edge between two faces on one surface is left out.
pub fn drawn_edge_samples(
    topo: &Topology,
    solid: SolidId,
    deflection: f64,
    angular_tol: f64,
    filter_smooth: bool,
) -> Result<Vec<(brepkit_topology::edge::EdgeId, Vec<Point3>)>, crate::OperationsError> {
    let edges = brepkit_topology::explorer::solid_edges(topo, solid)?;
    let edge_face_map = if filter_smooth {
        Some(brepkit_topology::explorer::edge_to_face_map(topo, solid)?)
    } else {
        None
    };
    let mut out = Vec::with_capacity(edges.len());
    for edge_id in edges {
        if let Some(ref efm) = edge_face_map
            && let Some(faces) = efm.get(&edge_id.index())
            && faces.len() == 2
        {
            let fa = topo.face(faces[0])?;
            let fb = topo.face(faces[1])?;
            if surfaces_equivalent(fa.surface(), fb.surface()) {
                continue;
            }
        }
        let edge = topo.edge(edge_id)?;
        super::check_edge_samples(edge.curve(), deflection, angular_tol)?;
        out.push((
            edge_id,
            sample_edge(topo, edge, deflection, angular_tol, false)?,
        ));
    }
    Ok(out)
}

/// The `rim_face` of a vertex that lies on no face's free rim alone.
pub(super) const NO_RIM_FACE: u32 = u32::MAX;

/// Weld remaining boundary vertices by merging coincident positions.
///
/// Uses union-find over a spatial hash grid to merge boundary vertices that
/// are within `weld_tol` of each other. Rewrites triangle indices and removes
/// degenerate triangles (where merged indices create duplicate vertices).
/// A weld never joins two vertices on the free rim of one face, directly or
/// through other vertices: `rim_face` gives the face of each vertex that
/// lies only on that face's free rim, and [`NO_RIM_FACE`] (or a vertex past
/// its end) for any other vertex.
/// `tri_faces` is the parallel tri -> face attribution array; entries for
/// removed degenerate triangles are filtered alongside. Returns whether any
/// directed edge was used twice before the weld: two faces bridging the same
/// gap with one chord leave one (see [`split_pinched_chords`]).
pub(super) fn weld_boundary_vertices(
    mesh: &mut TriangleMesh,
    deflection: f64,
    rim_face: &[u32],
    tri_faces: Option<&mut Vec<u32>>,
) -> bool {
    let n_verts = mesh.positions.len();
    if n_verts == 0 || mesh.indices.is_empty() {
        return false;
    }

    // The half-edges leaving each vertex, sorted, in one array.
    if mesh.indices.iter().any(|&idx| idx as usize >= n_verts) {
        return false;
    }
    let mut first = vec![0usize; n_verts + 1];
    for tri in mesh.indices.chunks_exact(3) {
        for &a in tri {
            first[a as usize + 1] += 1;
        }
    }
    for v in 0..n_verts {
        first[v + 1] += first[v];
    }
    let mut targets = vec![0u32; first[n_verts]];
    let mut fill = first.clone();
    for tri in mesh.indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            targets[fill[a as usize]] = b;
            fill[a as usize] += 1;
        }
    }
    for v in 0..n_verts {
        targets[first[v]..first[v + 1]].sort_unstable();
    }
    let leaving = |v: u32| &targets[first[v as usize]..first[v as usize + 1]];
    let repeated = (0..n_verts).any(|v| {
        targets[first[v]..first[v + 1]]
            .windows(2)
            .any(|w| w[0] == w[1])
    });

    // Boundary vertices: incident on an edge only one triangle uses. An edge
    // two faces share the same way round is no crack: where a rim curls
    // back through itself, the cap and the wall both run its loop that way,
    // and welding its closely spaced samples would collapse the loop.
    let mut on_boundary = vec![false; n_verts];
    #[allow(clippy::cast_possible_truncation)]
    for a in 0..n_verts as u32 {
        let out = leaving(a);
        for (k, &b) in out.iter().enumerate() {
            let once = (k == 0 || out[k - 1] != b) && out.get(k + 1) != Some(&b);
            if once && leaving(b).binary_search(&a).is_err() {
                on_boundary[a as usize] = true;
                on_boundary[b as usize] = true;
            }
        }
    }
    if !on_boundary.iter().any(|&b| b) {
        return repeated;
    }

    // Ascending ids keep grid-cell contents and union order reproducible.
    #[allow(clippy::cast_possible_truncation)]
    let boundary_verts: Vec<u32> = (0..n_verts as u32)
        .filter(|&v| on_boundary[v as usize])
        .collect();

    #[allow(clippy::items_after_statements)]
    fn uf_find(parent: &mut [u32], mut x: u32) -> u32 {
        while parent[x as usize] != x {
            parent[x as usize] = parent[parent[x as usize] as usize];
            x = parent[x as usize];
        }
        x
    }
    let mut parent: Vec<u32> = (0..n_verts as u32).collect();

    let weld_tol = deflection.max(1e-6) * 2.0;
    let inv_cell = 1.0 / weld_tol;

    #[allow(clippy::cast_possible_truncation)]
    let cell_key = |p: Point3| -> (i64, i64, i64) {
        (
            (p.x() * inv_cell).floor() as i64,
            (p.y() * inv_cell).floor() as i64,
            (p.z() * inv_cell).floor() as i64,
        )
    };

    let mut grid: DetHashMap<(i64, i64, i64), Vec<u32>> = DetHashMap::default();
    for &vid in &boundary_verts {
        let p = mesh.positions[vid as usize];
        grid.entry(cell_key(p)).or_default().push(vid);
    }
    let positions = &mesh.positions;
    // The vertices within reach of `vid` that follow it in id order.
    let reach_of = |vid: u32, near: &mut Vec<(f64, u32)>| {
        near.clear();
        let p = positions[vid as usize];
        let (cx, cy, cz) = cell_key(p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(cell) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &other in cell {
                            let gap = (p - positions[other as usize]).length();
                            if other > vid && gap < weld_tol {
                                near.push((gap, other));
                            }
                        }
                    }
                }
            }
        }
    };
    let rim_of = |v: u32| rim_face.get(v as usize).copied().unwrap_or(NO_RIM_FACE);

    // A pair without a rim sample joins at once: no cluster holds a rim
    // sample yet, and the clusters such pairs make do not depend on the
    // order of the joins. Rooting at the smallest index makes each
    // cluster's representative its least member.
    let mut near: Vec<(f64, u32)> = Vec::new();
    let mut rim_pairs = false;
    for &vid in &boundary_verts {
        reach_of(vid, &mut near);
        for &(_, other) in &near {
            let (rim, other_rim) = (rim_of(vid), rim_of(other));
            if rim == NO_RIM_FACE && other_rim == NO_RIM_FACE {
                let (ra, rb) = (uf_find(&mut parent, vid), uf_find(&mut parent, other));
                parent[ra.max(rb) as usize] = ra.min(rb);
            } else if rim != other_rim {
                rim_pairs = true;
            }
        }
    }

    // A pair with a rim sample joins in rounds. In each round, each cluster
    // takes its nearest partner that it may join, and those joins run
    // nearest first: a crack's own pairs join before a pair across a narrow
    // face. No cluster takes two samples of one face's rim, not even through
    // a sample of another face. A round keeps one pair for each cluster, so
    // the weld's memory grows with its samples, not with their pairs, and
    // the nearest pair of a round always joins, so the rounds end.
    let mut cluster_rims: DetHashMap<u32, Vec<u32>> = DetHashMap::default();
    if rim_pairs {
        for &vid in &boundary_verts {
            if rim_of(vid) != NO_RIM_FACE {
                cluster_rims.entry(vid).or_default().push(rim_of(vid));
            }
        }
    }
    let clash = |rims: &DetHashMap<u32, Vec<u32>>, ra: u32, rb: u32| {
        rims.get(&ra)
            .zip(rims.get(&rb))
            .is_some_and(|(a, b)| a.iter().any(|face| b.contains(face)))
    };
    let nearer = |a: &(f64, u32, u32), b: &(f64, u32, u32)| {
        a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2))
    };
    let mut best: DetHashMap<u32, (f64, u32, u32)> = DetHashMap::default();
    let mut joining = rim_pairs;
    while joining {
        best.clear();
        for &vid in &boundary_verts {
            reach_of(vid, &mut near);
            for &(gap, other) in &near {
                if rim_of(vid) == rim_of(other) {
                    continue;
                }
                let (ra, rb) = (uf_find(&mut parent, vid), uf_find(&mut parent, other));
                if ra == rb || clash(&cluster_rims, ra, rb) {
                    continue;
                }
                let pair = (gap, vid, other);
                for root in [ra, rb] {
                    let held = best.entry(root).or_insert(pair);
                    if nearer(&pair, held).is_lt() {
                        *held = pair;
                    }
                }
            }
        }
        let mut round: Vec<(f64, u32, u32)> = best.values().copied().collect();
        joining = !round.is_empty();
        round.sort_by(nearer);
        round.dedup_by(|a, b| a.1 == b.1 && a.2 == b.2);
        for (_, a, b) in round {
            let (ra, rb) = (uf_find(&mut parent, a), uf_find(&mut parent, b));
            if ra == rb || clash(&cluster_rims, ra, rb) {
                continue;
            }
            let (root, child) = (ra.min(rb), ra.max(rb));
            parent[child as usize] = root;
            if let Some(rims) = cluster_rims.remove(&child) {
                cluster_rims.entry(root).or_default().extend(rims);
            }
        }
    }

    let mut changed = false;
    for idx in &mut mesh.indices {
        let root = uf_find(&mut parent, *idx);
        if root != *idx {
            *idx = root;
            changed = true;
        }
    }

    if changed {
        let mut new_indices = Vec::with_capacity(mesh.indices.len());
        let mut new_tri_faces = Vec::with_capacity(tri_faces.as_ref().map_or(0, |tf| tf.len()));
        for (t, tri) in mesh.indices.chunks_exact(3).enumerate() {
            let (i0, i1, i2) = (tri[0], tri[1], tri[2]);
            if i0 != i1 && i1 != i2 && i2 != i0 {
                new_indices.push(i0);
                new_indices.push(i1);
                new_indices.push(i2);
                if let Some(&f) = tri_faces.as_ref().and_then(|tf| tf.get(t)) {
                    new_tri_faces.push(f);
                }
            }
        }
        mesh.indices = new_indices;
        if let Some(tf) = tri_faces {
            *tf = new_tri_faces;
        }
    }
    repeated
}

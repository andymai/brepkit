//! Non-planar CDT and fallback paths for face tessellation.

use brepkit_math::det_hash::{DetHashMap, DetHashSet};
use brepkit_math::madd::Madd;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use std::f64::consts::TAU;

use super::edge_sampling::{sample_edge, segments_for_chord_deviation_a};
use super::{MERGE_GRID, PointPool, TriangleMesh, point_merge_key};

/// Maps a 3D point to its `(u, v)` surface parameters.
type ProjectFn = Box<dyn Fn(Point3) -> (f64, f64)>;
/// Maps `(u, v)` surface parameters to a 3D surface point.
type EvalFn = Box<dyn Fn(f64, f64) -> Point3>;
/// Maps `(u, v)` surface parameters to the outward surface normal.
type NormalFn = Box<dyn Fn(f64, f64) -> Vec3>;

/// Per-face variant of the cycle-rim structured band: rims are sampled
/// LOCALLY at the requested deflection instead of pulled from the solid
/// tessellation's shared edge pool, so the `tessellate(topo, face, defl)`
/// route (which feeds `classify_point`'s meshes) gets the same watertight
/// wavy-band handling as the solid path. Returns `Ok(None)` when the face is
/// not a two-full-winding-rim band; the caller falls back.
pub(super) fn tessellate_band_face_local(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
) -> Result<Option<super::TriangleMeshUV>, crate::OperationsError> {
    let Some(rim_wires) = band_rim_wires(topo, face_data)? else {
        return Ok(None);
    };
    let (project, surf_normal): (ProjectFn, NormalFn) = match face_data.surface() {
        FaceSurface::Cylinder(c) => {
            let (c1, c2) = (c.clone(), c.clone());
            (
                Box::new(move |p| c1.project_point(p)),
                Box::new(move |u, v| c2.normal(u, v)),
            )
        }
        FaceSurface::Cone(c) => {
            let (c1, c2) = (c.clone(), c.clone());
            (
                Box::new(move |p| c1.project_point(p)),
                Box::new(move |u, v| c2.normal(u, v)),
            )
        }
        _ => return Ok(None),
    };

    // Curved wire edges → endpoint-connected cycles (the pool version's
    // structure; a closed single-edge NURBS loop has no by-construction
    // winding, so decline).
    let mut curved: Vec<(
        brepkit_topology::edge::EdgeId,
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
    )> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for oe in rim_wires
        .iter()
        .map(|&w| topo.wire(w).map(brepkit_topology::wire::Wire::edges))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
    {
        let e = topo.edge(oe.edge())?;
        match e.curve() {
            EdgeCurve::NurbsCurve(_) if e.start() == e.end() => return Ok(None),
            EdgeCurve::Circle(_) | EdgeCurve::NurbsCurve(_) => {
                if seen.insert(oe.edge().index()) {
                    curved.push((oe.edge(), e.start(), e.end()));
                }
            }
            EdgeCurve::Line => {}
            EdgeCurve::Ellipse(_) => return Ok(None),
        }
    }
    let mut by_vertex: std::collections::HashMap<brepkit_topology::vertex::VertexId, Vec<usize>> =
        std::collections::HashMap::new();
    for (j, &(_, sv, ev)) in curved.iter().enumerate() {
        by_vertex.entry(sv).or_default().push(j);
        by_vertex.entry(ev).or_default().push(j);
    }
    let mut used = vec![false; curved.len()];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..curved.len() {
        if used[start] {
            continue;
        }
        let (_, origin, mut at) = curved[start];
        used[start] = true;
        let mut cycle = vec![start];
        let mut closed = curved[start].1 == curved[start].2 || at == origin;
        while !closed {
            let Some(&next) = by_vertex
                .get(&at)
                .and_then(|c| c.iter().find(|&&j| !used[j]))
            else {
                break;
            };
            used[next] = true;
            at = if curved[next].1 == at {
                curved[next].2
            } else {
                curved[next].1
            };
            cycle.push(next);
            closed = at == origin;
        }
        if !closed {
            return Ok(None);
        }
        cycles.push(cycle);
    }
    if cycles.len() != 2 {
        return Ok(None);
    }
    let wrap_pi = |d: f64| -> f64 { (d + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0 };
    for cycle in &cycles {
        let mut winding = 0.0_f64;
        let mut whole_turn = false;
        let mut at: Option<brepkit_topology::vertex::VertexId> = None;
        for &ci in cycle {
            let (_, sv, ev) = curved[ci];
            if sv == ev {
                whole_turn = true;
                continue;
            }
            let (from, to) = match at {
                None => (sv, ev),
                Some(v) if v == sv => (sv, ev),
                Some(_) => (ev, sv),
            };
            let (u0, _) = project(topo.vertex(from)?.point());
            let (u1, _) = project(topo.vertex(to)?.point());
            winding += wrap_pi(u1 - u0);
            at = Some(to);
        }
        if !whole_turn && (winding.abs() - TAU).abs() > 1e-6 {
            return Ok(None);
        }
    }

    // Sample each rim's edges locally, in the order the rim's cycle walks
    // them, turned to run up the angle around the axis.
    let mut rims: Vec<Vec<Point3>> = Vec::with_capacity(2);
    for cycle in &cycles {
        let Some(mut pts) = rim_in_walk_order(
            cycle,
            &curved,
            |edge| -> Result<_, crate::OperationsError> {
                let edge = topo.edge(edge)?;
                Ok(Some(sample_edge(
                    topo,
                    edge,
                    deflection,
                    angular_tol,
                    false,
                )?))
            },
            |p| point_merge_key(*p, MERGE_GRID),
        )?
        else {
            return Ok(None);
        };
        if pts.len() < 3 {
            return Ok(None);
        }
        ascending_from_least_angle(&mut pts, |p| project(*p).0);
        rims.push(pts);
    }

    // Assemble the vertex arrays: ring 0 then ring 1.
    let n = rims[0].len();
    let m = rims[1].len();
    let mut positions: Vec<Point3> = Vec::with_capacity(n + m);
    positions.extend_from_slice(&rims[0]);
    positions.extend_from_slice(&rims[1]);
    let mut normals: Vec<Vec3> = Vec::with_capacity(n + m);
    let mut uvs: Vec<[f64; 2]> = Vec::with_capacity(n + m);
    for p in &positions {
        let (u, v) = project(*p);
        normals.push(surf_normal(u, v));
        uvs.push([u, v]);
    }

    // Angular zipper (the pool version's sweep, on local indices). Rotate
    // ring 1 to start just after ring 0's start angle.
    let ang = |i: usize| -> f64 { uvs[i][0] };
    let base = ang(0);
    let start1 = (0..m)
        .min_by(|&a, &b| {
            let ka = (ang(n + a) - base).rem_euclid(TAU);
            let kb = (ang(n + b) - base).rem_euclid(TAU);
            ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    let ring0: Vec<usize> = (0..n).collect();
    let mut ring1: Vec<usize> = (n..n + m).collect();
    ring1.rotate_left(start1);
    let unwrap = |a: f64| (a - base).rem_euclid(TAU);
    let a0: Vec<f64> = ring0.iter().map(|&i| unwrap(ang(i))).collect();
    let a1: Vec<f64> = ring1.iter().map(|&i| unwrap(ang(i))).collect();
    let u0 = unwrapped_angles(&ring0.iter().map(|&i| ang(i)).collect::<Vec<_>>(), base);
    let u1 = unwrapped_angles(&ring1.iter().map(|&i| ang(i)).collect::<Vec<_>>(), base);
    let corner = |(ring, k): (usize, usize)| {
        let (at, u) = if ring == 0 {
            (ring0[k % n], u0[k])
        } else {
            (ring1[k % m], u1[k])
        };
        #[allow(clippy::cast_possible_truncation)]
        (at as u32, [u, uvs[at][1]])
    };
    let tris: Vec<([u32; 3], [[f64; 2]; 3])> = zip_rims(&a0, &a1)
        .into_iter()
        .map(|tri| {
            let [(a, pa), (b, pb), (c, pc)] = tri.map(corner);
            ([a, b, c], [pa, pb, pc])
        })
        .collect();
    if tris.is_empty() || !turns_one_way(tris.iter().map(|&(_, uv)| uv)) {
        return Ok(None);
    }

    let mut mesh = TriangleMesh {
        positions,
        normals,
        indices: tris.into_iter().flat_map(|(ids, _)| ids).collect(),
    };
    orient_triangle_run(&mut mesh, 0, project.as_ref(), surf_normal.as_ref());
    Ok(Some(super::TriangleMeshUV { mesh, uvs }))
}

/// The wires that hold a band's rims: its outer wire, or with no seam the
/// outer wire and its one hole, each wire then one rim. `None` for a face
/// with more holes, or with a hole and a straight edge in its outer wire.
fn band_rim_wires(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<Option<Vec<brepkit_topology::wire::WireId>>, crate::OperationsError> {
    let outer = face_data.outer_wire();
    Ok(match face_data.inner_wires() {
        [] => Some(vec![outer]),
        &[hole] => {
            let straight = topo.wire(outer)?.edges().iter().any(|oe| {
                topo.edge(oe.edge())
                    .is_ok_and(|e| matches!(e.curve(), EdgeCurve::Line))
            });
            (!straight).then(|| vec![outer, hole])
        }
        _ => None,
    })
}

/// Tessellate a cylinder/cone lateral "standard band" face directly from the
/// shared rim edge vertices, bypassing the snap path's proximity reconciliation.
///
/// The snap path tessellates the cylinder independently and snaps its rim
/// vertices to the shared edge pool by 1e-6 proximity; when the independent rim
/// sampling and the shared-edge sampling diverge by one segment (a radius/
/// deflection-dependent off-by-one) the rim vertices land at different angles,
/// fail the snap, and become near-coincident duplicates that crack the mesh
/// (issue #696: a drilled magnet hole). Reusing the shared rim vertices makes
/// the band watertight by construction.
///
/// Returns `Ok(true)` when the face is a simple two-rim band (or a pointed
/// cone, one rim and a seam line up to the apex) that was handled here,
/// `Ok(false)` when it is not (the caller then falls back to the snap or CDT
/// path). A "simple band" has no inner wires and exactly two rims
/// (everything else a seam line), or no seam, one rim its outer wire and the
/// other its one hole. Each rim is either one **closed** circle
/// edge or a CHAIN of open circle arcs at one constant `v` whose spans sum to
/// a full revolution — a boolean that splits a rim at tangency or crossing
/// points (e.g. the cone∪box inscribed-rim fuse, whose z=6 rim arrives as
/// four arcs each shared with a different corner face) still gets the
/// structured watertight band. Rims with equal shared-vertex counts sweep
/// index-paired exactly as before; unequal counts (each rim's sampling is
/// dictated by its own neighbours) are stitched with an angular zipper merge.
/// Each rim keeps the order its cycle walks it, and the band is emitted only
/// when every triangle turns one way in the unwrapped `(u, v)` strip. A notch
/// cut into a rim can fold the sweep (a side along a ruling, or a steep side
/// that the zipper joins to the other rim across the notch); such a face
/// returns `Ok(false)` for the CDT.
pub(super) fn tessellate_revolution_band_shared(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
) -> Result<bool, crate::OperationsError> {
    let Some(rim_wires) = band_rim_wires(topo, face_data)? else {
        return Ok(false);
    };

    let (project, surf_normal): (ProjectFn, NormalFn) = match face_data.surface() {
        FaceSurface::Cylinder(c) => {
            let (c1, c2) = (c.clone(), c.clone());
            (
                Box::new(move |p| c1.project_point(p)),
                Box::new(move |u, v| c2.normal(u, v)),
            )
        }
        FaceSurface::Cone(c) => {
            let (c1, c2) = (c.clone(), c.clone());
            (
                Box::new(move |p| c1.project_point(p)),
                Box::new(move |u, v| c2.normal(u, v)),
            )
        }
        _ => return Ok(false),
    };

    // Collect rim edges as endpoint-connected CYCLES of curved edges;
    // everything else must be a seam line. A rim is any cycle whose net
    // surface-u winding is a full revolution: one closed circle, a chain of
    // ring arcs, or a wavy mixed circle+NURBS chain (the winding-chain band
    // separator). A cycle that does not wind — a lens hole, a partial band
    // arc run bounded by non-seam generators — declines the structured
    // sweep, which would otherwise skin across the removed region.
    let wire = topo.wire(face_data.outer_wire())?;
    let mut curved: Vec<(
        usize,
        brepkit_topology::vertex::VertexId,
        brepkit_topology::vertex::VertexId,
    )> = Vec::new();
    let mut seen: std::collections::HashSet<usize> = std::collections::HashSet::new();
    for oe in rim_wires
        .iter()
        .map(|&w| topo.wire(w).map(brepkit_topology::wire::Wire::edges))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .flatten()
    {
        let e = topo.edge(oe.edge())?;
        match e.curve() {
            // A closed single-edge NURBS loop has no by-construction winding
            // (unlike a closed circle) — decline rather than guess.
            EdgeCurve::NurbsCurve(_) if e.start() == e.end() => return Ok(false),
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) | EdgeCurve::NurbsCurve(_) => {
                if seen.insert(oe.edge().index()) {
                    curved.push((oe.edge().index(), e.start(), e.end()));
                }
            }
            EdgeCurve::Line => {}
        }
    }
    // Walk cycles by shared vertices (vertex→edge adjacency built once).
    let mut by_vertex: std::collections::HashMap<brepkit_topology::vertex::VertexId, Vec<usize>> =
        std::collections::HashMap::new();
    for (j, &(_, sv, ev)) in curved.iter().enumerate() {
        by_vertex.entry(sv).or_default().push(j);
        by_vertex.entry(ev).or_default().push(j);
    }
    let mut used = vec![false; curved.len()];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..curved.len() {
        if used[start] {
            continue;
        }
        let (_, origin, mut at) = curved[start];
        used[start] = true;
        let mut cycle = vec![start];
        let mut closed = curved[start].1 == curved[start].2 || at == origin;
        while !closed {
            let Some(&next) = by_vertex
                .get(&at)
                .and_then(|c| c.iter().find(|&&j| !used[j]))
            else {
                break;
            };
            used[next] = true;
            at = if curved[next].1 == at {
                curved[next].2
            } else {
                curved[next].1
            };
            cycle.push(next);
            closed = at == origin;
        }
        if !closed {
            return Ok(false); // open curved run — not a rim structure
        }
        cycles.push(cycle);
    }
    let pointed_cone = matches!(face_data.surface(), FaceSurface::Cone(_)) && cycles.len() == 1;
    if cycles.len() != 2 && !pointed_cone {
        return Ok(false);
    }
    // Net winding per cycle from surface-projected endpoint deltas; a closed
    // single edge (start == end vertex) winds a full turn by construction.
    let wrap_pi = |d: f64| -> f64 { (d + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0 };
    for cycle in &cycles {
        let mut winding = 0.0_f64;
        let mut whole_turn = false;
        let mut at: Option<brepkit_topology::vertex::VertexId> = None;
        for &ci in cycle {
            let (_, sv, ev) = curved[ci];
            if sv == ev {
                whole_turn = true;
                continue;
            }
            let (from, to) = match at {
                None => (sv, ev),
                Some(v) if v == sv => (sv, ev),
                Some(_) => (ev, sv),
            };
            let (u0, _) = project(topo.vertex(from)?.point());
            let (u1, _) = project(topo.vertex(to)?.point());
            winding += wrap_pi(u1 - u0);
            at = Some(to);
        }
        if !whole_turn && (winding.abs() - TAU).abs() > 1e-6 {
            return Ok(false);
        }
        // The sweep pairs the rims by angle, so each must advance in u one
        // way all round: a pin's section through a bore's mouth runs back on
        // itself where the pin's wall is tangent to the bore's, and sorting
        // its samples by angle stitched the fold across itself.
        if !whole_turn {
            let mut at: Option<brepkit_topology::vertex::VertexId> = None;
            let mut prev: Option<f64> = None;
            for &ci in cycle {
                let (eid, sv, ev) = curved[ci];
                let Some(ids) = edge_global_indices.get(&eid) else {
                    return Ok(false);
                };
                let along = !matches!(at, Some(v) if v != sv);
                at = Some(if along { ev } else { sv });
                let ordered: Box<dyn Iterator<Item = &u32>> = if along {
                    Box::new(ids.iter())
                } else {
                    Box::new(ids.iter().rev())
                };
                for &gid in ordered {
                    let (u, _) = project(merged.positions[gid as usize]);
                    if let Some(p) = prev {
                        let du = wrap_pi(u - p);
                        if du * winding < -1e-9 {
                            return Ok(false);
                        }
                    }
                    prev = Some(u);
                }
            }
        }
    }

    // Pull each rim's shared global vertex IDs in the order its cycle walks
    // them, turned to run up the angle around the axis from its least angle.
    let angle_of = |gid: u32, merged: &TriangleMesh| project(merged.positions[gid as usize]).0;
    let mut rims: Vec<Vec<u32>> = Vec::with_capacity(2);
    for cycle in &cycles {
        let Some(mut ids) = rim_in_walk_order(
            cycle,
            &curved,
            |edge| Ok(edge_global_indices.get(&edge).cloned()),
            |&gid| gid,
        )?
        else {
            return Ok(false);
        };
        if ids.len() < 3 {
            return Ok(false);
        }
        ascending_from_least_angle(&mut ids, |&gid| angle_of(gid, merged));
        rims.push(ids);
    }
    let n = rims[0].len();

    // A pointed cone's only other boundary is its seam line up to the apex.
    // Every ruling is straight, so a fan from the rim's shared samples to the
    // apex deviates from the surface only by the rim's own chords.
    let apex_id = if pointed_cone {
        let FaceSurface::Cone(cone) = face_data.surface() else {
            return Ok(false);
        };
        let mut seam = None;
        for oe in wire.edges() {
            let e = topo.edge(oe.edge())?;
            if matches!(e.curve(), EdgeCurve::Line) {
                if seam.is_some_and(|s| s != oe.edge()) {
                    return Ok(false);
                }
                seam = Some(oe.edge());
            }
        }
        let Some(seam) = seam else {
            return Ok(false);
        };
        let seam_edge = topo.edge(seam)?;
        let on_rim = |v| {
            cycles[0]
                .iter()
                .any(|&ci| curved[ci].1 == v || curved[ci].2 == v)
        };
        let apex_end = if on_rim(seam_edge.start()) {
            seam_edge.end()
        } else if on_rim(seam_edge.end()) {
            seam_edge.start()
        } else {
            return Ok(false);
        };
        let apex_vertex = topo.vertex(apex_end)?;
        if (apex_vertex.point() - cone.apex()).length() > apex_vertex.tolerance() {
            return Ok(false);
        }
        // Only the apex sample is taken from the seam: a closed circle is
        // sampled from its frame's origin, so the rim's samples need not
        // include the seam's rim vertex.
        let Some(&apex) = edge_global_indices.get(&seam.index()).and_then(|ids| {
            ids.iter().find(|&&id| {
                (merged.positions[id as usize] - apex_vertex.point()).length()
                    <= apex_vertex.tolerance()
            })
        }) else {
            return Ok(false);
        };
        Some(apex)
    } else {
        None
    };

    // Each candidate triangle with its corners' (u, v), u unwrapped along
    // the rims from ring 0's first sample.
    let rim_uv = |rim: &[u32], merged: &TriangleMesh, base: f64| {
        let angles: Vec<f64> = rim.iter().map(|&g| angle_of(g, merged)).collect();
        let vs: Vec<f64> = rim
            .iter()
            .map(|&g| project(merged.positions[g as usize]).1)
            .collect();
        (unwrapped_angles(&angles, base), vs)
    };
    let base = angle_of(rims[0][0], merged);
    let (u0, v0) = rim_uv(&rims[0], merged, base);
    let mut tris: Vec<([u32; 3], [[f64; 2]; 3])> = Vec::with_capacity(2 * n + 2);

    if let Some(apex) = apex_id {
        let apex_v = project(merged.positions[apex as usize]).1;
        for i in 0..n {
            let ni = (i + 1) % n;
            tris.push((
                [rims[0][i], rims[0][ni], apex],
                [
                    [u0[i], v0[i]],
                    [u0[i + 1], v0[ni]],
                    [0.5 * (u0[i] + u0[i + 1]), apex_v],
                ],
            ));
        }
    } else {
        let m = rims[1].len();
        if n == m {
            // Equal counts: the historical index-paired sweep (kept
            // byte-identical for the calibrated closed-rim cases). A sample
            // on the surface's u seam projects to either end of [0, 2π), so
            // the two rings can start one sample out of phase; pairing
            // them as-is twists the band through the solid. Rotate ring 1 to
            // start at ring 0's angular partner, keeping index 0 unless
            // another sample is strictly closer.
            let circular = |a: f64, b: f64| {
                let d = (a - b).rem_euclid(TAU);
                d.min(TAU - d)
            };
            let mut phase = 0;
            let mut phase_gap = circular(angle_of(rims[1][0], merged), base);
            for k in 1..m {
                let gap = circular(angle_of(rims[1][k], merged), base);
                if gap < phase_gap - 1e-9 {
                    phase = k;
                    phase_gap = gap;
                }
            }
            rims[1].rotate_left(phase);
            let (u1, v1) = rim_uv(&rims[1], merged, base);
            for i in 0..n {
                let j = (i + 1) % n;
                let (b0, b1) = (rims[0][i], rims[0][j]);
                let (t0, t1) = (rims[1][i], rims[1][j]);
                let (pb0, pb1) = ([u0[i], v0[i]], [u0[i + 1], v0[j]]);
                let (pt0, pt1) = ([u1[i], v1[i]], [u1[i + 1], v1[j]]);
                tris.push(([b0, b1, t1], [pb0, pb1, pt1]));
                tris.push(([b0, t1, t0], [pb0, pt1, pt0]));
            }
        } else {
            // Unequal counts: angular zipper merge. Rotate ring 1 so its
            // start sits just after ring 0's start angle, keeping the initial
            // quad local instead of spanning the whole circle.
            let start1 = (0..m)
                .min_by(|&a, &b| {
                    let ka = (angle_of(rims[1][a], merged) - base).rem_euclid(TAU);
                    let kb = (angle_of(rims[1][b], merged) - base).rem_euclid(TAU);
                    ka.partial_cmp(&kb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            rims[1].rotate_left(start1);
            let (u1, v1) = rim_uv(&rims[1], merged, base);
            let unwrap = |a: f64| (a - base).rem_euclid(TAU);
            let a0: Vec<f64> = rims[0]
                .iter()
                .map(|&g| unwrap(angle_of(g, merged)))
                .collect();
            let a1: Vec<f64> = rims[1]
                .iter()
                .map(|&g| unwrap(angle_of(g, merged)))
                .collect();
            let corner = |(ring, k): (usize, usize)| {
                if ring == 0 {
                    (rims[0][k % n], [u0[k], v0[k % n]])
                } else {
                    (rims[1][k % m], [u1[k], v1[k % m]])
                }
            };
            for tri in zip_rims(&a0, &a1) {
                let [(a, pa), (b, pb), (c, pc)] = tri.map(corner);
                tris.push(([a, b, c], [pa, pb, pc]));
            }
        }
    }

    if tris.is_empty() || !turns_one_way(tris.iter().map(|&(_, uv)| uv)) {
        return Ok(false);
    }
    // The triangles all turn one way, so one flip decision orients the run
    // outward, the convention `tessellate_analytic` uses. The caller
    // (`tessellate_face_with_shared_edges`) applies the face's `is_reversed`
    // flip afterward, so it must NOT be applied here.
    let idx_start = merged.indices.len();
    for (ids, _) in tris {
        merged.indices.extend_from_slice(&ids);
    }
    orient_triangle_run(merged, idx_start, project.as_ref(), surf_normal.as_ref());
    Ok(true)
}

type RimEdge<K> = (
    K,
    brepkit_topology::vertex::VertexId,
    brepkit_topology::vertex::VertexId,
);

/// The samples of one rim cycle in the order the cycle walks them: an edge's
/// samples run backwards where the walk enters the edge at its end, and a
/// joint shared by consecutive edges, like the closing sample, appears once.
/// An angle sort cannot stand in for the walk: a rim that runs along a
/// ruling (the side of a notch cut into the rim) holds samples at one angle.
/// `None` when an edge has no samples or the walk meets a sample twice.
fn rim_in_walk_order<K: Copy, T, Q: Eq + std::hash::Hash>(
    cycle: &[usize],
    curved: &[RimEdge<K>],
    mut samples: impl FnMut(K) -> Result<Option<Vec<T>>, crate::OperationsError>,
    key: impl Fn(&T) -> Q,
) -> Result<Option<Vec<T>>, crate::OperationsError> {
    let mut rim: Vec<T> = Vec::new();
    let mut at = None;
    for &ci in cycle {
        let (edge, sv, ev) = curved[ci];
        let forward = at.is_none_or(|v| v == sv);
        let Some(mut pts) = samples(edge)? else {
            return Ok(None);
        };
        if !forward {
            pts.reverse();
        }
        for p in pts {
            if rim.last().is_none_or(|q| key(q) != key(&p)) {
                rim.push(p);
            }
        }
        at = Some(if forward { ev } else { sv });
    }
    while rim.len() > 1 && rim.first().map(&key) == rim.last().map(&key) {
        rim.pop();
    }
    let mut seen = DetHashSet::default();
    if !rim.iter().all(|p| seen.insert(key(p))) {
        return Ok(None);
    }
    Ok(Some(rim))
}

/// Turns a walked rim to run up the angle around the axis and starts it at
/// its least angle: the order an angle sort gives a rim that is a graph over
/// the angle.
fn ascending_from_least_angle<T>(rim: &mut [T], angle: impl Fn(&T) -> f64) {
    let n = rim.len();
    if n == 0 {
        return;
    }
    let turn: f64 = (0..n)
        .map(|k| wrap_to_half_turn(angle(&rim[(k + 1) % n]) - angle(&rim[k])))
        .sum();
    if turn < 0.0 {
        rim.reverse();
    }
    let least = (0..n)
        .min_by(|&a, &b| {
            angle(&rim[a])
                .partial_cmp(&angle(&rim[b]))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    rim.rotate_left(least);
}

fn wrap_to_half_turn(d: f64) -> f64 {
    (d + TAU / 2.0).rem_euclid(TAU) - TAU / 2.0
}

/// A rim's angles unwrapped along it, starting within half a turn of
/// `base`, with a closing entry for the first sample after one more step.
fn unwrapped_angles(angles: &[f64], base: f64) -> Vec<f64> {
    let Some(&first) = angles.first() else {
        return Vec::new();
    };
    let mut at = base + wrap_to_half_turn(first - base);
    let mut out = Vec::with_capacity(angles.len() + 1);
    out.push(at);
    for k in 1..=angles.len() {
        at += wrap_to_half_turn(angles[k % angles.len()] - angles[k - 1]);
        out.push(at);
    }
    out
}

/// The angular zipper between two rims that each run up the angle from
/// `a0[0]`: advance whichever rim's next sample comes first in angle, one
/// triangle per advance, until both close. Corners are `(rim, position)`;
/// position `len` is the rim's first sample again, one turn on.
fn zip_rims(a0: &[f64], a1: &[f64]) -> Vec<[(usize, usize); 3]> {
    let next = |a: &[f64], k: usize| {
        if k >= a.len() {
            f64::INFINITY
        } else if k + 1 < a.len() {
            a[k + 1]
        } else {
            a[0] + TAU
        }
    };
    let (mut i, mut j) = (0, 0);
    let mut tris = Vec::with_capacity(a0.len() + a1.len());
    while i < a0.len() || j < a1.len() {
        if next(a0, i) <= next(a1, j) {
            tris.push([(0, i), (1, j), (0, i + 1)]);
            i += 1;
        } else {
            tris.push([(0, i), (1, j), (1, j + 1)]);
            j += 1;
        }
    }
    tris
}

/// Whether band triangles, given by their corners' unwrapped `(u, v)`, all
/// turn one way. Such triangles cover the band between two rims that each
/// wind once exactly once; one that turns the other way, or has no area in
/// `(u, v)`, folds over a region a rim leaves out, such as a notch cut into
/// the rim.
fn turns_one_way(tris: impl IntoIterator<Item = [[f64; 2]; 3]>) -> bool {
    let mut way: Option<bool> = None;
    for [a, b, c] in tris {
        let turn = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
        let left = match turn.partial_cmp(&0.0) {
            Some(std::cmp::Ordering::Greater) => true,
            Some(std::cmp::Ordering::Less) => false,
            _ => return false,
        };
        if *way.get_or_insert(left) != left {
            return false;
        }
    }
    true
}

/// Tessellate a torus band bounded by two closed rims and seamed by a
/// doubled open arc (one edge, or a chain of them), in either orientation:
///   * constant-`v` rims (latitude circles wrapping the ring angle `u`): a
///     full analytic revolve of a profile arc, seamed by that arc; interior
///     full-`u` rows are swept along the tube angle;
///   * tube rims wrapping `v` (tube circles at constant `u`, a PARTIAL-turn
///     revolve of a full circle profile; or a plane's loops around the tube,
///     whose `u` wanders with `v`), seamed along a latitude; interior full-`v`
///     rings are swept along the ring angle between the rims, column by column.
///
/// The rims split their periodic direction into two arcs; the seam arc's
/// midpoint picks which one the band covers (sweeping the wrong one would skin
/// the band across the material). Both rims reuse their SHARED pool vertices,
/// so the band meets its neighbour caps/walls crack-free — the CDT path
/// degenerates on these fully-wrapping UV images and the snap path re-samples
/// the rims independently (the #696 crack class).
///
/// Returns `Ok(false)` (caller falls back to CDT/snap) for any other torus
/// face.
pub(super) fn tessellate_torus_two_rim_band(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<bool, crate::OperationsError> {
    use std::f64::consts::TAU;
    let FaceSurface::Torus(torus) = face_data.surface() else {
        return Ok(false);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }

    let wire = topo.wire(face_data.outer_wire())?;
    // The seam is run up and back once (it may be split into several
    // pieces); every other edge runs once, round one of the two rims (a
    // closed curve, or arcs joined end to end).
    let mut uses: Vec<(brepkit_topology::edge::EdgeId, usize)> = Vec::new();
    for oe in wire.edges() {
        if matches!(topo.edge(oe.edge())?.curve(), EdgeCurve::Ellipse(_)) {
            return Ok(false);
        }
        match uses.iter_mut().find(|(eid, _)| *eid == oe.edge()) {
            Some((_, n)) => *n += 1,
            None => uses.push((oe.edge(), 1)),
        }
    }
    if uses.iter().any(|&(_, n)| n > 2) {
        return Ok(false);
    }
    // A NURBS seam is the analytic revolve of a recognised NURBS-circle
    // profile arc, and a LINE seam is the rim-fillet band's degenerate chord
    // between its two contact circles: the seam is only midpoint-sampled (via
    // the EdgeCurve delegates) to pick the covered arc, so any open curve type
    // is safe here.
    let Some(seam_eid) = wire
        .edges()
        .iter()
        .map(brepkit_topology::wire::OrientedEdge::edge)
        .find(|eid| uses.iter().any(|&(e, n)| e == *eid && n == 2))
    else {
        return Ok(false);
    };
    // The once-run edges in two rims, joined by shared vertices.
    let mut rims_of: Vec<Vec<brepkit_topology::edge::EdgeId>> = Vec::new();
    let mut rim_vertices: Vec<Vec<brepkit_topology::vertex::VertexId>> = Vec::new();
    for &(eid, n) in &uses {
        if n != 1 {
            continue;
        }
        let e = topo.edge(eid)?;
        let ends = [e.start(), e.end()];
        let touching: Vec<usize> = (0..rims_of.len())
            .filter(|&r| ends.iter().any(|v| rim_vertices[r].contains(v)))
            .collect();
        match touching.as_slice() {
            [] => {
                rims_of.push(vec![eid]);
                rim_vertices.push(ends.to_vec());
            }
            [r] => {
                rims_of[*r].push(eid);
                rim_vertices[*r].extend(ends);
            }
            [r, rest @ ..] => {
                let r = *r;
                for &o in rest.iter().rev() {
                    let (edges, verts) = (rims_of.remove(o), rim_vertices.remove(o));
                    rims_of[r].extend(edges);
                    rim_vertices[r].extend(verts);
                }
                rims_of[r].push(eid);
                rim_vertices[r].extend(ends);
            }
        }
    }
    if rims_of.len() != 2 {
        return Ok(false);
    }

    let (t1, t2, t3) = (torus.clone(), torus.clone(), torus.clone());
    let project = move |p: Point3| t1.project_point(p);
    let surf_eval = move |u: f64, v: f64| t2.evaluate(u, v);
    let surf_normal = move |u: f64, v: f64| t3.normal(u, v);

    // Circular mean and max wrapped deviation of a set of angles.
    let circ_mean_spread = |angles: &[f64]| -> (f64, f64) {
        let (mut sx, mut sy) = (0.0_f64, 0.0_f64);
        for &a in angles {
            sx += a.cos();
            sy += a.sin();
        }
        let mean = sy.atan2(sx);
        let spread = angles
            .iter()
            .map(|&a| {
                let d = (a - mean + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
                d.abs()
            })
            .fold(0.0_f64, f64::max);
        (mean.rem_euclid(TAU), spread)
    };

    // Project each rim's shared pool vertices (wrap-safe: a rim at angle 0
    // projects samples on both sides of the period).
    let mut raw: Vec<Vec<(f64, f64, u32)>> = Vec::with_capacity(2);
    for rim in &rims_of {
        let mut seen: DetHashSet<u32> = DetHashSet::default();
        let mut pts: Vec<(f64, f64, u32)> = Vec::new();
        for eid in rim {
            let Some(gids) = edge_global_indices.get(&eid.index()) else {
                return Ok(false);
            };
            for &g in gids {
                if !seen.insert(g) {
                    continue;
                }
                let (u, v) = project(merged.positions[g as usize]);
                pts.push((u, v, g));
            }
        }
        if pts.len() < 3 {
            return Ok(false);
        }
        raw.push(pts);
    }

    // Both rims must be latitude rims (constant v, swept along the tube
    // angle) or both tube rims (swept along the ring angle).
    let spread_of = |pts: &[(f64, f64, u32)], pick_u: bool| -> (f64, f64) {
        let angles: Vec<f64> = pts
            .iter()
            .map(|&(u, v, _)| if pick_u { u } else { v })
            .collect();
        circ_mean_spread(&angles)
    };
    // Tube rims may also wander in u as they wind round the tube (a plane's
    // loops around it), as long as neither winds the ring.
    let (u_stats0, v_stats0) = (spread_of(&raw[0], true), spread_of(&raw[0], false));
    let (u_stats1, v_stats1) = (spread_of(&raw[1], true), spread_of(&raw[1], false));
    let lat_mode = if v_stats0.1 <= 1e-6 && v_stats1.1 <= 1e-6 {
        true
    } else if u_stats0.1 < std::f64::consts::FRAC_PI_2 && u_stats1.1 < std::f64::consts::FRAC_PI_2 {
        false
    } else {
        return Ok(false);
    };
    // Each tube rim's u along its tube angle, unwrapped about its mean and
    // read between its pool vertices.
    let wrap = |d: f64| (d + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
    let tracks: Vec<Vec<(f64, f64)>> = raw
        .iter()
        .zip([u_stats0.0, u_stats1.0])
        .map(|(pts, mean)| {
            let mut track: Vec<(f64, f64)> = pts
                .iter()
                .map(|&(u, v, _)| (v.rem_euclid(TAU), mean + wrap(u - mean)))
                .collect();
            track.sort_by(|a, b| a.0.total_cmp(&b.0));
            track
        })
        .collect();
    let u_at = |k: usize, a: f64| -> f64 {
        let track = &tracks[k];
        let a = a.rem_euclid(TAU);
        let hi = track.partition_point(|&(v, _)| v < a);
        let (first, last) = (track[0], track[track.len() - 1]);
        let (p0, p1) = match hi {
            0 => ((last.0 - TAU, last.1), first),
            h if h == track.len() => (last, (first.0 + TAU, first.1)),
            h => (track[h - 1], track[h]),
        };
        let gap = p1.0 - p0.0;
        if gap <= 1e-15 {
            p0.1
        } else {
            p0.1 + (p1.1 - p0.1) * (a - p0.0) / gap
        }
    };

    // Rings keyed by the wrapping parameter, sorted, covering its full circle.
    let mut rims: Vec<LatRing> = Vec::with_capacity(2);
    for pts in &raw {
        let mut ring: LatRing = pts
            .iter()
            .map(|&(u, v, g)| if lat_mode { (u, g) } else { (v, g) })
            .collect();
        ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let max_gap = ring
            .windows(2)
            .map(|w| w[1].0 - w[0].0)
            .chain(std::iter::once(ring[0].0 + TAU - ring[ring.len() - 1].0))
            .fold(0.0_f64, f64::max);
        if max_gap > std::f64::consts::PI {
            return Ok(false);
        }
        rims.push(ring);
    }

    // The seam arc's midpoint picks which of the two swept-parameter arcs
    // between the rims the band covers.
    let seam_edge = topo.edge(seam_eid)?;
    let sp = topo.vertex(seam_edge.start())?.point();
    let ep = topo.vertex(seam_edge.end())?.point();
    let (d0, d1) = seam_edge.curve().domain_with_endpoints(sp, ep);
    let seam_mid = seam_edge
        .curve()
        .evaluate_with_endpoints(f64::midpoint(d0, d1), sp, ep);
    let (mid_u, mid_v) = project(seam_mid);
    let mid = if lat_mode { mid_v } else { mid_u };
    let (lvl0, lvl1) = if lat_mode {
        (v_stats0.0, v_stats1.0)
    } else {
        (u_at(0, mid_v), u_at(1, mid_v))
    };
    let fwd_span = (lvl1 - lvl0).rem_euclid(TAU);
    if fwd_span < 1e-9 || (TAU - fwd_span) < 1e-9 {
        return Ok(false);
    }
    let mid_off = (mid - lvl0).rem_euclid(TAU);
    let sweep = if mid_off <= fwd_span {
        fwd_span
    } else {
        -(TAU - fwd_span)
    };

    // Interior rows along the swept parameter; each row wraps the other
    // parameter's full circle.
    let (sweep_radius, wrap_radius) = if lat_mode {
        (
            torus.minor_radius(),
            torus.major_radius() + torus.minor_radius(),
        )
    } else {
        (
            torus.major_radius() + torus.minor_radius(),
            torus.minor_radius(),
        )
    };
    let full_circle_cols =
        segments_for_chord_deviation_a(wrap_radius, TAU, deflection, angular_tol, true);
    let n_cols = rims[0].len().max(rims[1].len()).max(full_circle_cols);
    super::check_count("a band's columns", n_cols, deflection)?;
    // The ring angle a column sweeps between tube rims, the covered way.
    let span_at = |a: f64| {
        let fwd = (u_at(1, a) - u_at(0, a)).rem_euclid(TAU);
        if sweep > 0.0 { fwd } else { fwd - TAU }
    };
    let widest = if lat_mode {
        sweep.abs()
    } else {
        (0..n_cols)
            .map(|j| {
                #[allow(clippy::cast_precision_loss)]
                let a = TAU * (j as f64) / (n_cols as f64);
                span_at(a).abs()
            })
            .fold(0.0_f64, f64::max)
    };
    let n_rows =
        segments_for_chord_deviation_a(sweep_radius, widest, deflection, angular_tol, true).max(1);

    // Rims joined from several arcs turn corners where the arcs meet, and can
    // run nearly along a latitude near one (a lobe near its pinch). A row
    // column level with every rim vertex keeps each row turning with the rim;
    // a row sampled only between them would cut across a corner and fold the
    // stitch.
    #[allow(clippy::cast_precision_loss)]
    let mut cols: Vec<f64> = (0..n_cols)
        .map(|j| TAU * (j as f64) / (n_cols as f64))
        .collect();
    if !lat_mode && rims_of.iter().any(|rim| rim.len() > 1) {
        cols.extend(rims.iter().flatten().map(|&(a, _)| a.rem_euclid(TAU)));
        cols.sort_by(f64::total_cmp);
        cols.dedup_by(|a, b| (*a - *b).abs() < 1e-7);
    }
    super::check_grid(cols.len(), n_rows, deflection)?;

    // Orient each stitch in (u, v), where counterclockwise is outward on a
    // torus: a sliver between a nearly straight rim and a row just inside it
    // is too thin for its 3D normal to say which way it faces.
    let emit = |merged: &mut TriangleMesh, a: u32, b: u32, c: u32| {
        if a == b || b == c || a == c {
            return;
        }
        let wrap = |d: f64| (d + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
        let (ua, va) = project(merged.positions[a as usize]);
        let (ub, vb) = project(merged.positions[b as usize]);
        let (uc, vc) = project(merged.positions[c as usize]);
        // A horn torus's pinch lies on the axis, where `u` means nothing: a
        // corner there reads the `u` of the triangle's other corners.
        let on_axis = |i: u32| {
            let from_center = merged.positions[i as usize] - torus.center();
            (from_center - torus.z_axis() * from_center.dot(torus.z_axis())).length() <= 1e-9
        };
        let (bu, bv) = (wrap(ub - ua), wrap(vb - va));
        let (cu, cv) = (wrap(uc - ua), wrap(vc - va));
        let (bu, cu) = if on_axis(a) {
            (0.0, wrap(uc - ub))
        } else if on_axis(b) {
            (0.0, cu)
        } else if on_axis(c) {
            (bu, 0.0)
        } else {
            (bu, cu)
        };
        let area = bu.madd(cv, -(bv * cu));
        if area.abs() < 1e-18 {
            return;
        }
        let mut tri = [a, b, c];
        if area < 0.0 {
            tri.swap(1, 2);
        }
        merged.indices.extend_from_slice(&tri);
    };
    let mut prev_ring: LatRing = rims[0].clone();
    for i in 1..n_rows {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f64 / n_rows as f64;
        let level = lvl0 + sweep * t;
        let mut row: LatRing = Vec::with_capacity(cols.len());
        for &a in &cols {
            let (u, v) = if lat_mode {
                (a, level)
            } else {
                (span_at(a).madd(t, u_at(0, a)), a)
            };
            let p = surf_eval(u, v);
            let key = point_merge_key(p, MERGE_GRID);
            let gid = *point_to_global.entry(key).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(p);
                merged.normals.push(surf_normal(u, v));
                idx
            });
            row.push((a, gid));
        }
        stitch_rings(merged, &prev_ring, &row, &emit);
        prev_ring = row;
    }
    stitch_rings(merged, &prev_ring, &rims[1], &emit);
    Ok(true)
}

/// A boundary ring of a latitude band: each entry is `(u_angle, global_id)`,
/// with `u_angle ∈ [0, 2π)`. Sorted ascending by angle so two rings align by
/// longitude during stitching.
type LatRing = Vec<(f64, u32)>;

/// Collect a torus face wire's boundary as a ring of `(tube-angle v, shared gid)`
/// sorted by `v`, taking the SHARED global vertices (so the ring shares the
/// notch walls' vertices) and projecting to the torus `(u, v)`. Accepts edges of
/// any curve type (the notch seam arcs are NURBS). Returns `None` if any edge is
/// missing from the shared pool, or the ring does NOT wrap the tube — detected
/// as the largest gap between consecutive sorted `v` samples (including the
/// wrap-around gap) EXCEEDING half a turn (`π`): a ring that encircles the tube
/// has all its `v`-gaps below `π`, whereas a partial arc leaves one gap above it.
fn collect_torus_phi_ring(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    torus: &brepkit_math::surfaces::ToroidalSurface,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &TriangleMesh,
) -> Result<Option<Vec<(f64, u32)>>, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut gids: Vec<u32> = Vec::new();
    for oe in wire.edges() {
        let Some(edge_gids) = edge_global_indices.get(&oe.edge().index()) else {
            return Ok(None);
        };
        gids.extend_from_slice(edge_gids);
    }
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    let mut ring: Vec<(f64, u32)> = Vec::with_capacity(gids.len());
    for g in gids {
        if !seen.insert(g) {
            continue;
        }
        let (_, v) = torus.project_point(merged.positions[g as usize]);
        ring.push((v, g));
    }
    if ring.len() < 3 {
        return Ok(None);
    }
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    // Must wrap the tube once: largest v-gap (incl. wrap) under a full turn.
    let max_gap = ring
        .windows(2)
        .map(|w| w[1].0 - w[0].0)
        .chain(std::iter::once(
            ring[0].0 + std::f64::consts::TAU - ring[ring.len() - 1].0,
        ))
        .fold(0.0_f64, f64::max);
    if max_gap > std::f64::consts::PI {
        return Ok(None);
    }
    Ok(Some(ring))
}

/// Tessellate the `torus − box`-style notch band: a kept toroidal patch that
/// WRAPS the tube angle `v` fully and is bounded by TWO `v`-wrapping seam-arc
/// loops at the two ends of a ring-angle (`u`) span (the box notch's `±y` walls).
/// The band is swept structurally along `u` from one boundary loop to the other
/// the LONG way (through `u = π`, the 294° kept side), with full-`v` interior
/// rings; both boundary loops use their SHARED wall vertices, so the band and the
/// plane notch walls meet crack-free (watertight). Returns `false` (defer to the
/// CDT path) for any torus face that is not this two-`v`-loop notch band.
///
/// Distinct from [`tessellate_latitude_band_shared`]: there the two boundaries
/// are constant-`v` latitude circles swept along `v`; here they wrap `v` and the
/// sweep is along `u`.
pub(super) fn tessellate_torus_notch_band(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<bool, crate::OperationsError> {
    use std::f64::consts::{PI, TAU};
    let FaceSurface::Torus(torus) = face_data.surface() else {
        return Ok(false);
    };
    if face_data.inner_wires().len() != 1 {
        return Ok(false);
    }
    let t1 = torus.clone();
    let t2 = torus.clone();
    let project = move |p: Point3| t1.project_point(p);
    let surf_normal = move |u: f64, v: f64| t2.normal(u, v);

    // Both boundary loops wrap the tube (v) once, with their shared wall gids.
    let Some(ring_a) = collect_torus_phi_ring(
        topo,
        face_data.outer_wire(),
        torus,
        edge_global_indices,
        merged,
    )?
    else {
        return Ok(false);
    };
    let Some(ring_b) = collect_torus_phi_ring(
        topo,
        face_data.inner_wires()[0],
        torus,
        edge_global_indices,
        merged,
    )?
    else {
        return Ok(false);
    };

    // Ring-angle (u) of each loop: each loop sits at a u-BAND (the box wall's cut
    // varies in u with the tube angle), one near u_a, the other near u_b, the
    // kept band the LONG way between them. Take each loop's mean u (wrap-safe)
    // plus its half-u-spread, so the interior rows start at each loop's KEPT-SIDE
    // edge (mean ± spread toward the band midpoint), NOT its mean — otherwise the
    // first/last interior row sits INSIDE the loop's u-band and the stitch folds
    // back over the boundary strip, under-covering the band.
    let mean_u = |ring: &[(f64, u32)]| -> f64 {
        let (mut sx, mut sy) = (0.0, 0.0);
        for &(_, g) in ring {
            let (u, _) = project(merged.positions[g as usize]);
            sx += u.cos();
            sy += u.sin();
        }
        sy.atan2(sx).rem_euclid(TAU)
    };
    // Max signed u-offset of a ring's vertices from its mean (wrap into (-π,π]).
    let half_spread = |ring: &[(f64, u32)], mean: f64| -> f64 {
        ring.iter()
            .map(|&(_, g)| {
                let (u, _) = project(merged.positions[g as usize]);
                let d = (u - mean + PI).rem_euclid(TAU) - PI;
                d.abs()
            })
            .fold(0.0_f64, f64::max)
    };
    let u_a = mean_u(&ring_a);
    let u_b = mean_u(&ring_b);
    let spread_a = half_spread(&ring_a, u_a);
    let spread_b = half_spread(&ring_b, u_b);

    // Sweep the LONG way from ring_a toward ring_b (through the kept far side).
    let fwd_span = (u_b - u_a).rem_euclid(TAU); // a -> b increasing u
    // The interior must lie on the long arc; start just past each loop's
    // kept-side edge so no interior row overlaps a boundary loop's u-band.
    let (u_start, u_end) = if fwd_span >= PI {
        // a -> b the long way is INCREASING u: kept edge of a is u_a+spread_a,
        // of b is u_b-spread_b (i.e. u_a+fwd_span-spread_b).
        (u_a + spread_a, u_a + fwd_span - spread_b)
    } else {
        // a -> b the long way is DECREASING u.
        (u_a - spread_a, u_a - (TAU - fwd_span) + spread_b)
    };
    let span = (u_end - u_start).abs();
    if span < 1e-6 {
        return Ok(false);
    }

    // Interior rows: full-v circles at constant u, stepped along the sweep. Count
    // from chord deviation over the band's u-arc-length (radius ≈ R, the ring).
    let n_u =
        segments_for_chord_deviation_a(torus.major_radius(), span, deflection, angular_tol, true)
            .max(2);
    // v-resolution: a full tube circle.
    let n_v =
        segments_for_chord_deviation_a(torus.minor_radius(), TAU, deflection, angular_tol, true)
            .max(8);
    super::check_grid(n_u, n_v, deflection)?;

    // Build interior rings as `LatRing` (sorted by v) of fresh vertices.
    let build_u_ring =
        |u: f64, merged: &mut TriangleMesh, point_to_global: &mut PointPool| -> LatRing {
            let mut row: LatRing = Vec::with_capacity(n_v);
            for j in 0..n_v {
                #[allow(clippy::cast_precision_loss)]
                let v = TAU * (j as f64) / (n_v as f64);
                let p = torus.evaluate(u, v);
                let key = point_merge_key(p, MERGE_GRID);
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(p);
                    merged.normals.push(surf_normal(u, v));
                    idx
                });
                row.push((v, gid));
            }
            row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            row
        };

    let emit = make_band_emit(&project, &surf_normal);
    let idx_start = merged.indices.len();

    // Stitch ring_a -> interior rows -> ring_b. All rings sorted by v; `v` is the
    // ring parameter passed to `stitch_rings` (it walks the shared tube angle).
    let mut prev: LatRing = ring_a;
    for iu in 1..n_u {
        #[allow(clippy::cast_precision_loss)]
        let u = u_start + (u_end - u_start) * (iu as f64) / (n_u as f64);
        let row = build_u_ring(u.rem_euclid(TAU), merged, point_to_global);
        stitch_rings(merged, &prev, &row, &emit);
        prev = row;
    }
    stitch_rings(merged, &prev, &ring_b, &emit);

    // Orient the whole band once against the torus outward normal.
    orient_triangle_run(merged, idx_start, &project, &surf_normal);
    Ok(true)
}

/// Tessellate a sphere/torus latitude band (the annular region between two
/// constant-`v` full-revolution boundaries) as a structured UV grid.
///
/// The CDT path cannot bound this band: each constant-`v` latitude boundary
/// projects to a back-and-forth horizontal segment of zero UV area, so the
/// 2D polygon degenerates and the triangulation fills the removed polar cap
/// (the tunnel mouth on a bored sphere is skinned over). Like the cylinder/cone
/// `tessellate_revolution_band_shared`, this builds the band directly from the
/// shared boundary vertices instead.
///
/// Unlike the ruled cylinder/cone band (whose two rims connect directly because
/// the surface is straight in `v`), a sphere/torus band bulges between its two
/// latitudes, so intermediate latitude rows are inserted until the chord error
/// in `v` stays within `deflection`. The two boundary rows reuse the shared rim
/// global vertex IDs (watertight by construction); interior-row vertices are new
/// face-local points evaluated on the surface across the full `u` ring.
///
/// Returns `Ok(true)` when the face is such a band and was handled here, else
/// `Ok(false)` (the caller then takes the CDT/snap path). Detection is
/// deliberately conservative: a face qualifies only if its surface is a sphere
/// or torus, it has exactly one inner wire, and both the outer and inner wires
/// are closed full-revolution loops, each at a single constant `v`, built only
/// from `Line`/`Circle` edges, at two distinct `v` levels.
#[allow(clippy::too_many_lines)]
pub(super) fn tessellate_latitude_band_shared(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<bool, crate::OperationsError> {
    if face_data.inner_wires().len() != 1 {
        return Ok(false);
    }

    let (project, surf_eval, surf_normal): (ProjectFn, EvalFn, NormalFn) = match face_data.surface()
    {
        FaceSurface::Sphere(s) => {
            let (s1, s2, s3) = (s.clone(), s.clone(), s.clone());
            (
                Box::new(move |p| s1.project_point(p)),
                Box::new(move |u, v| s2.evaluate(u, v)),
                Box::new(move |u, v| s3.normal(u, v)),
            )
        }
        FaceSurface::Torus(t) => {
            let (t1, t2, t3) = (t.clone(), t.clone(), t.clone());
            (
                Box::new(move |p| t1.project_point(p)),
                Box::new(move |u, v| t2.evaluate(u, v)),
                Box::new(move |u, v| t3.normal(u, v)),
            )
        }
        _ => return Ok(false),
    };

    let band_radius = match face_data.surface() {
        FaceSurface::Sphere(s) => s.radius(),
        FaceSurface::Torus(t) => t.minor_radius(),
        _ => return Ok(false),
    };
    let emit = make_band_emit(project.as_ref(), surf_normal.as_ref());
    let full_circle_cols = segments_for_chord_deviation_a(
        band_radius,
        std::f64::consts::TAU,
        deflection,
        angular_tol,
        true,
    );

    let outer_wid = face_data.outer_wire();
    let inner_wid = face_data.inner_wires()[0];

    // Case 1 — both boundaries are single constant-v latitude circles (the
    // bored-quadric band, e.g. sphere − through-cylinder). Sweep constant-v
    // interior rows between them.
    let outer_const = collect_constant_v_ring(
        topo,
        outer_wid,
        project.as_ref(),
        edge_global_indices,
        merged,
    )?;
    let inner_const = collect_constant_v_ring(
        topo,
        inner_wid,
        project.as_ref(),
        edge_global_indices,
        merged,
    )?;

    if let (Some((v_outer, ring_outer)), Some((v_inner, ring_inner))) = (&outer_const, &inner_const)
    {
        let mut rings = [
            (*v_outer, ring_outer.clone()),
            (*v_inner, ring_inner.clone()),
        ];
        rings.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let (v_lo, ring_lo) = (&rings[0].0, &rings[0].1);
        let (v_hi, ring_hi) = (&rings[1].0, &rings[1].1);
        let (v_lo, v_hi) = (*v_lo, *v_hi);
        if (v_hi - v_lo).abs() < 1e-9 {
            return Ok(false);
        }
        let n_v =
            segments_for_chord_deviation_a(band_radius, v_hi - v_lo, deflection, angular_tol, true)
                .max(1);
        let n_u_interior = ring_lo.len().max(ring_hi.len()).max(full_circle_cols);
        super::check_grid(n_u_interior, n_v, deflection)?;
        let mut prev_ring: LatRing = ring_lo.clone();
        for iv in 1..n_v {
            #[allow(clippy::cast_precision_loss)]
            let t = iv as f64 / n_v as f64;
            let v = v_lo + (v_hi - v_lo) * t;
            let row = build_interior_row(
                v,
                n_u_interior,
                surf_eval.as_ref(),
                surf_normal.as_ref(),
                merged,
                point_to_global,
            );
            stitch_rings(merged, &prev_ring, &row, &emit);
            prev_ring = row;
        }
        stitch_rings(merged, &prev_ring, ring_hi, &emit);
        return Ok(true);
    }

    // Case 2 — a COLLAR: one wire is a constant-v cap circle, the other a
    // full-longitude-wrap "floor" at varying v (great-circle/seam arcs of a
    // box ∩ sphere patch below its cap, or a tilted plane's circle above a
    // hemisphere's equator). Sweep interior rows whose per-column v
    // interpolates from the floor to the cap.
    let ((v_cap, cap_ring), floor_wid) = match (outer_const, inner_const) {
        (None, Some(cap)) => (cap, outer_wid),
        (Some(cap), None) => (cap, inner_wid),
        _ => return Ok(false),
    };
    let Some(floor) = collect_var_v_ring(
        topo,
        floor_wid,
        face_data.surface(),
        project.as_ref(),
        edge_global_indices,
        merged,
    )?
    else {
        return Ok(false);
    };
    // The collar must straddle the cap (the floor sits on the far side of the
    // cap latitude). Reject a near-flat outer wire (would be Case 1).
    let floor_v_min = floor.iter().map(|r| r.1).fold(f64::INFINITY, f64::min);
    let floor_v_max = floor.iter().map(|r| r.1).fold(f64::NEG_INFINITY, f64::max);
    if (floor_v_max - floor_v_min) <= 1e-6 {
        return Ok(false); // constant-v outer — Case 1 already tried it
    }
    let floor_v_near = if (v_cap - floor_v_max).abs() >= (v_cap - floor_v_min).abs() {
        floor_v_max
    } else {
        floor_v_min
    };
    if (v_cap - floor_v_near).abs() < 1e-9 {
        return Ok(false);
    }

    // The rows run along the floor's own columns, so a wide gap between
    // them spans the band with chords that no row can bend back to the
    // surface (a floor through the pole leaves half a turn unsampled and cuts
    // through the ball); past four deflections of sag, the CDT takes it.
    if let FaceSurface::Sphere(sphere) = face_data.surface() {
        let mut us: Vec<f64> = floor.iter().map(|r| r.0).collect();
        us.sort_by(f64::total_cmp);
        let gap = us
            .windows(2)
            .map(|w| w[1] - w[0])
            .chain(std::iter::once(us[0] + TAU - us[us.len() - 1]))
            .fold(0.0_f64, f64::max);
        let (lo, hi) = (v_cap.min(floor_v_min), v_cap.max(floor_v_max));
        let widest = if lo <= 0.0 && hi >= 0.0 {
            1.0
        } else {
            lo.abs().min(hi.abs()).cos()
        };
        if sphere.radius() * widest * (1.0 - (0.5 * gap).cos()) > 4.0 * deflection {
            return Ok(false);
        }
    }
    // The outer (scalloped) ring is the lower boundary; sweep up to the cap.
    // Use the absolute band height: the floor can sit above the cap latitude
    // (a southern collar), and a negative range trips the chord-deviation
    // helper's `<= 0` fallback (a fixed count) instead of scaling with height.
    let n_v = segments_for_chord_deviation_a(
        band_radius,
        (v_cap - floor_v_near).abs(),
        deflection,
        angular_tol,
        true,
    )
    .max(1);
    super::check_grid(floor.len(), n_v, deflection)?;

    // Lower boundary ring as a LatRing (drop the v component; the gid carries
    // the shared scalloped-floor vertex).
    let floor_ring: LatRing = floor.iter().map(|&(u, _, g)| (u, g)).collect();

    // Emit the collar's triangles in the rings' consistent walk order WITHOUT a
    // per-triangle normal flip, then orient the whole collar once below. (The
    // per-triangle normal fix that the bored-band path uses is unstable for the
    // thin stitch triangles bridging the clustered floor to the even cap — it
    // flips neighbours inconsistently. A single decision keeps the collar a
    // coherent 2-manifold.)
    let collar_idx_start = merged.indices.len();
    let emit_raw = |merged: &mut TriangleMesh, a: u32, b: u32, c: u32| {
        if a == b || b == c || a == c {
            return;
        }
        let (pa, pb, pc) = (
            merged.positions[a as usize],
            merged.positions[b as usize],
            merged.positions[c as usize],
        );
        if (pb - pa).cross(pc - pa).length() < 1e-20 {
            return;
        }
        merged.indices.extend_from_slice(&[a, b, c]);
    };

    // Connect the floor to each interior row as COLUMN-ALIGNED quad strips
    // (same longitudes, same count), then zipper only the topmost interior row
    // to the cap (different longitude sampling) with `stitch_rings`.
    let mut prev_ring: LatRing = floor_ring;
    for iv in 1..n_v {
        #[allow(clippy::cast_precision_loss)]
        let t = iv as f64 / n_v as f64;
        let row = build_collar_row(
            &floor,
            v_cap,
            t,
            surf_eval.as_ref(),
            surf_normal.as_ref(),
            merged,
            point_to_global,
        );
        emit_aligned_quad_strip(merged, &prev_ring, &row, &emit_raw);
        prev_ring = row;
    }
    stitch_rings(merged, &prev_ring, &cap_ring, &emit_raw);

    // Orient the collar as a whole: pick the best-conditioned triangle (largest
    // area), compare its geometric normal to the surface outward normal at its
    // centroid, and flip every collar triangle's winding if they disagree.
    orient_triangle_run(
        merged,
        collar_idx_start,
        project.as_ref(),
        surf_normal.as_ref(),
    );

    Ok(true)
}

/// The samples a spherical triangle's rows take: its apex, `count + 1` for
/// each row's `count` segments, and `base` along its base. The sum is in
/// `u64`, held at its largest value, so a 32-bit `usize` cannot wrap it
/// under the limit.
fn triangle_row_samples(base: usize, counts: &[usize]) -> u64 {
    counts.iter().fold(1 + base as u64, |total, &count| {
        total.saturating_add((count as u64).saturating_add(1))
    })
}

/// A sphere face bounded by three arcs (a fillet's setback corner, a
/// sphere's octant), two of which the shared samples split into the same
/// number `m` of segments, is meshed in rows from the corner between those
/// two: row `s` runs from the `s`-th sample of one to the `s`-th of the
/// other in `s` segments, or fewer where the row is shorter than `s` side
/// segments, and the third side is the last row. A point
/// between is the mean of the three points that lines along the three
/// sides give it, read on the side samples, and pushed out onto the sphere.
/// Two rows join in order along them, each step by the shorter of its two
/// new edges. An edge inside the patch that turns by more than `SQRT_2`
/// times `angular_tol` is then halved. The rims are the shared samples
/// themselves, so the face closes with its neighbours.
///
/// Returns `Ok(false)` and leaves the mesh as it was, for the CDT to take
/// the face, in these cases:
/// - the face is not such a patch, or the wire does not run counterclockwise
///   round the patch about the sphere's outward normal;
/// - a chord turning by `SQRT_2` times `angular_tol` would stand off the
///   sphere by more than `deflection`;
/// - the patch reaches the hemisphere opposite its samples' mean direction;
/// - a triangle faces into the sphere (a fold, which a side bending far
///   into the patch gives), or stands off it by more than `deflection`.
pub(super) fn tessellate_sphere_triangle_shared(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<bool, crate::OperationsError> {
    let FaceSurface::Sphere(sphere) = face_data.surface() else {
        return Ok(false);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }
    let wire = topo.wire(face_data.outer_wire())?;
    if wire.edges().len() != 3 {
        return Ok(false);
    }
    let mut sides: Vec<Vec<u32>> = Vec::with_capacity(3);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        if !matches!(edge.curve(), EdgeCurve::Circle(_)) || edge.start() == edge.end() {
            return Ok(false);
        }
        let Some(gids) = edge_global_indices.get(&oe.edge().index()) else {
            return Ok(false);
        };
        if gids.len() < 2 {
            return Ok(false);
        }
        let mut side = gids.clone();
        if !oe.is_forward() {
            side.reverse();
        }
        sides.push(side);
    }
    if (0..3).any(|k| sides[k].last() != sides[(k + 1) % 3].first()) {
        return Ok(false);
    }
    // The corner whose two sides hold the same count of samples is the
    // apex: the sides then run apex to `b`, `b` to `c`, and `c` back.
    let Some(apex) = (0..3).find(|&k| sides[k].len() == sides[(k + 2) % 3].len()) else {
        return Ok(false);
    };
    sides.rotate_left(apex);
    let along_b = &sides[0];
    let base = &sides[1];
    let along_c: Vec<u32> = sides[2].iter().rev().copied().collect();
    let m = along_b.len() - 1;

    let (center, radius) = (sphere.center(), sphere.radius());
    // An edge inside the patch may turn by as much as the diagonal of a
    // grid cell whose sides turn by the angle tolerance, the bound the CDT's
    // interior grid keeps. Where the deflection holds edges shorter than
    // that, the rows' triangles would stand further off the sphere than
    // their edges' chords, and the CDT takes the face.
    let by_angle = if angular_tol > 0.0 {
        angular_tol * std::f64::consts::SQRT_2
    } else {
        std::f64::consts::PI
    };
    let by_sag = 2.0 * (1.0 - (deflection / radius).min(1.0)).acos();
    if by_sag < by_angle {
        return Ok(false);
    }
    let point_of = |gid: u32| merged.positions[gid as usize];
    let mut mean = Vec3::new(0.0, 0.0, 0.0);
    for side in &sides {
        for &gid in &side[1..] {
            mean += point_of(gid) - center;
        }
    }
    let Ok(axis) = mean.normalize() else {
        return Ok(false);
    };
    if sides
        .iter()
        .flatten()
        .any(|&gid| (point_of(gid) - center).dot(axis) <= 1e-9 * radius)
    {
        return Ok(false);
    }
    // The rows cover the patch that the arcs close within that hemisphere.
    // The face lies on the left of its wire about the sphere's outward
    // normal, whichever way the face itself faces: the patch is the face
    // only when the wire turns counterclockwise about `axis`.
    let rim: Vec<Point3> = sides
        .iter()
        .flat_map(|side| side[1..].iter().map(|&gid| point_of(gid) - center))
        .map(|d| Point3::new(d.x(), d.y(), d.z()))
        .collect();
    let origin = Point3::new(0.0, 0.0, 0.0);
    let turning: f64 = (0..rim.len())
        .map(|k| {
            (rim[k] - origin)
                .cross(rim[(k + 1) % rim.len()] - origin)
                .dot(axis)
        })
        .sum();
    if turning <= 0.0 {
        return Ok(false);
    }

    let b_points: Vec<Point3> = along_b.iter().map(|&gid| point_of(gid)).collect();
    let c_points: Vec<Point3> = along_c.iter().map(|&gid| point_of(gid)).collect();
    let base_points: Vec<Point3> = base.iter().map(|&gid| point_of(gid)).collect();
    let lerp = |a: Point3, b: Point3, f: f64| a + (b - a) * f;
    // A side's polyline at a fraction of its samples, `x` in `[0, 1]`.
    let side_at = |points: &[Point3], x: f64| -> Point3 {
        #[allow(clippy::cast_precision_loss)]
        let at = x.clamp(0.0, 1.0) * (points.len() - 1) as f64;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let k = (at.floor() as usize).min(points.len() - 2);
        #[allow(clippy::cast_precision_loss)]
        lerp(points[k], points[k + 1], at - k as f64)
    };
    let gap = |a: Point3, b: Point3| (a - b).length();
    #[allow(clippy::cast_precision_loss)]
    let mf = m as f64;

    let pos_start = merged.positions.len();
    let idx_start = merged.indices.len();
    let ptg_start = point_to_global.len();
    // Row `s` runs from the `s`-th sample towards `b` to the `s`-th towards
    // `c` in up to `s` segments, at the barycentric level `s / m` from the
    // apex. A row that turns through less than `s` of the side segments
    // beside it takes fewer segments, so a narrow patch takes triangles for
    // its area and not for its sides' length. An octant's rows turn through
    // at least `s` of them and keep `s`.
    let turn = |p: Point3, q: Point3| {
        let (u, v) = (p - center, q - center);
        u.cross(v).length().atan2(u.dot(v))
    };
    let counts: Vec<usize> = (1..m)
        .map(|s| {
            let spacing =
                turn(b_points[s - 1], b_points[s]).max(turn(c_points[s - 1], c_points[s]));
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            if spacing > 0.0 {
                ((turn(b_points[s], c_points[s]) / spacing).ceil() as usize).clamp(1, s)
            } else {
                s
            }
        })
        .collect();
    // The rows are counted as they will be cut, before any is built: a
    // narrow patch has short rows however many it has.
    super::check_count("a face grid's rows", m + 1, deflection)?;
    super::check_total(
        "a face's rows",
        triangle_row_samples(base.len(), &counts),
        deflection,
    )?;
    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(m + 1);
    rows.push(vec![along_b[0]]);
    for s in 1..m {
        let count = counts[s - 1];
        let mut row = Vec::with_capacity(count + 1);
        row.push(along_b[s]);
        #[allow(clippy::cast_precision_loss)]
        let (sf, cf) = (s as f64, count as f64);
        for j in 1..count {
            #[allow(clippy::cast_precision_loss)]
            let f = j as f64 / cf;
            // Barycentric weights of the apex, `b` and `c`.
            let (beta, gamma) = (sf / mf * (1.0 - f), sf / mf * f);
            let in_row = lerp(b_points[s], c_points[s], f);
            let across_b = lerp(
                side_at(&b_points, beta),
                side_at(&base_points, 1.0 - beta),
                gamma / (1.0 - beta),
            );
            let across_c = lerp(
                side_at(&c_points, gamma),
                side_at(&base_points, gamma),
                beta / (1.0 - gamma),
            );
            let blend = Point3::new(
                (in_row.x() + across_b.x() + across_c.x()) / 3.0,
                (in_row.y() + across_b.y() + across_c.y()) / 3.0,
                (in_row.z() + across_b.z() + across_c.z()) / 3.0,
            );
            let Ok(out) = (blend - center).normalize() else {
                rollback(merged, point_to_global, pos_start, idx_start, ptg_start);
                return Ok(false);
            };
            let point = center + out * radius;
            let key = point_merge_key(point, MERGE_GRID);
            let gid = *point_to_global.entry(key).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(point);
                merged.normals.push(out);
                idx
            });
            row.push(gid);
        }
        row.push(along_c[s]);
        rows.push(row);
    }
    rows.push(base.clone());
    // Each pair of rows is joined in order along them: each triangle runs
    // from one row to the next, and of the two that can come next, the one
    // with the shorter new edge does.
    for s in 0..m {
        let (upper, lower) = (&rows[s], &rows[s + 1]);
        let (mut i, mut l) = (0, 0);
        while i + 1 < upper.len() || l + 1 < lower.len() {
            let lower_next = l + 1 < lower.len()
                && (i + 1 == upper.len()
                    || gap(
                        position_of(merged, upper[i]),
                        position_of(merged, lower[l + 1]),
                    ) <= gap(
                        position_of(merged, lower[l]),
                        position_of(merged, upper[i + 1]),
                    ));
            if lower_next {
                merged
                    .indices
                    .extend_from_slice(&[upper[i], lower[l], lower[l + 1]]);
                l += 1;
            } else {
                merged
                    .indices
                    .extend_from_slice(&[upper[i], lower[l], upper[i + 1]]);
                i += 1;
            }
        }
    }

    // One winding for the whole patch, then every triangle must face out of
    // the sphere and keep within the deflection of it.
    let outward = |t: &[u32]| {
        let [a, b, c] = [0, 1, 2].map(|k| merged.positions[t[k] as usize]);
        let middle = Point3::new(
            (a.x() + b.x() + c.x()) / 3.0,
            (a.y() + b.y() + c.y()) / 3.0,
            (a.z() + b.z() + c.z()) / 3.0,
        );
        let normal = (b - a).cross(c - a);
        (normal.dot(middle - center), normal, a)
    };
    let facing: f64 = merged.indices[idx_start..]
        .chunks_exact(3)
        .map(|t| outward(t).0)
        .sum();
    if facing < 0.0 {
        for t in merged.indices[idx_start..].chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }
    // An edge inside the patch longer than that bound is halved, and so are
    // the two triangles on it.
    let halved = halve_long_sphere_edges(
        merged,
        point_to_global,
        idx_start,
        (center, radius),
        by_angle.cos(),
    );
    let outward = |t: &[u32]| {
        let [a, b, c] = [0, 1, 2].map(|k| merged.positions[t[k] as usize]);
        let middle = a + ((b - a) + (c - a)) * (1.0 / 3.0);
        let normal = (b - a).cross(c - a);
        (normal.dot(middle - center), normal, a)
    };
    let sound = halved
        && merged.indices[idx_start..].chunks_exact(3).all(|t| {
            let (facing, normal, corner) = outward(t);
            let Ok(unit) = normal.normalize() else {
                return false;
            };
            facing > 0.0 && radius - (corner - center).dot(unit).abs() <= deflection
        });
    if !sound {
        rollback(merged, point_to_global, pos_start, idx_start, ptg_start);
        return Ok(false);
    }
    Ok(true)
}

/// Halve each edge inside a patch of a sphere's mesh (from `idx_start` on)
/// whose ends lie further apart than the angle whose cosine is `min_cos`,
/// at its middle pushed out onto the sphere, longest first, with the two
/// triangles on it. An edge of the patch's rim, held by one triangle, is
/// never halved. Returns `false` when the halving runs past a bound.
fn halve_long_sphere_edges(
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
    idx_start: usize,
    (center, radius): (Point3, f64),
    min_cos: f64,
) -> bool {
    use brepkit_math::det_hash::{WordHashMap, WordState};

    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let cos_between = |merged: &TriangleMesh, a: u32, b: u32| {
        let (p, q) = (
            (position_of(merged, a) - center).at_unit_size(),
            (position_of(merged, b) - center).at_unit_size(),
        );
        p.dot(q) / (p.length() * q.length())
    };
    let long_edge = |merged: &TriangleMesh, t: &[u32]| {
        (0..3).any(|i| cos_between(merged, t[i], t[(i + 1) % 3]) < min_cos)
    };
    if !merged.indices[idx_start..]
        .chunks_exact(3)
        .any(|t| long_edge(merged, t))
    {
        return true;
    }
    let mut tris: Vec<[u32; 3]> = merged.indices[idx_start..]
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .collect();
    // The triangles on each edge: an edge of the rim has one, and an edge
    // with a third is never halved.
    let mut owners: WordHashMap<(u32, u32), Owners> =
        WordHashMap::with_capacity_and_hasher(tris.len() * 2, WordState);
    for (k, t) in tris.iter().enumerate() {
        for i in 0..3 {
            owners.entry(key(t[i], t[(i + 1) % 3])).or_default().add(k);
        }
    }
    // Longest first: the heap orders by `1 - cos`, then by the ends.
    let mut long: std::collections::BinaryHeap<(u64, u32, u32)> = owners
        .iter()
        .filter(|(_, held)| held.pair().is_some())
        .filter_map(|(&(a, b), _)| {
            let c = cos_between(merged, a, b);
            (c < min_cos).then(|| ((1.0 - c).to_bits(), a, b))
        })
        .collect();
    let bound = 4 * tris.len() + 64;
    let mut halvings = 0;
    while let Some((_, a, b)) = long.pop() {
        let Some(held) = owners.get(&key(a, b)).and_then(Owners::pair) else {
            continue;
        };
        owners.remove(&key(a, b));
        halvings += 1;
        if halvings > bound {
            return false;
        }
        let (pa, pb) = (position_of(merged, a), position_of(merged, b));
        let Ok(out) = (pa + (pb - pa) * 0.5 - center).normalize() else {
            return false;
        };
        let point = center + out * radius;
        let m = *point_to_global
            .entry(point_merge_key(point, MERGE_GRID))
            .or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(point);
                merged.normals.push(out);
                idx
            });
        let mut fresh = vec![key(a, m), key(m, b)];
        for t in held {
            // Turn the triangle to start at the edge's first end in its own
            // winding: (p, q, x) with {p, q} = {a, b}.
            let tri = tris[t];
            let Some(at) = (0..3).find(|&i| key(tri[i], tri[(i + 1) % 3]) == key(a, b)) else {
                return false;
            };
            let (p, q, x) = (tri[at], tri[(at + 1) % 3], tri[(at + 2) % 3]);
            let n = tris.len();
            tris[t] = [p, m, x];
            tris.push([m, q, x]);
            owners.entry(key(p, m)).or_default().add(t);
            owners.entry(key(m, q)).or_default().add(n);
            let middle = owners.entry(key(m, x)).or_default();
            middle.add(t);
            middle.add(n);
            if let Some(held) = owners.get_mut(&key(q, x)) {
                held.replace(t, n);
            }
            fresh.push(key(m, x));
        }
        for (u, v) in fresh {
            let c = cos_between(merged, u, v);
            if c < min_cos && owners.get(&(u, v)).and_then(Owners::pair).is_some() {
                long.push(((1.0 - c).to_bits(), u, v));
            }
        }
    }
    merged.indices.truncate(idx_start);
    for t in tris {
        merged.indices.extend_from_slice(&t);
    }
    true
}

/// A cylinder face bounded by two arcs and two straight sides along the
/// axis, each side holding only its two ends in the shared samples (a
/// fillet's stripe between two plane faces), is meshed as one strip between
/// the arcs: the arcs' samples join in order along them, each step by the
/// shorter of its two new edges. The rims are the shared samples themselves,
/// so the face closes with its neighbours.
///
/// Returns `Ok(false)` and leaves the mesh as it was when the face is not
/// such a strip, when a triangle faces into the cylinder, or when a triangle
/// spans more of the cylinder's turn than a chord may at `deflection` or
/// than `SQRT_2` times `angular_tol`. The CDT takes the face then.
pub(super) fn tessellate_cylinder_strip_shared(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
) -> Result<bool, crate::OperationsError> {
    let FaceSurface::Cylinder(cylinder) = face_data.surface() else {
        return Ok(false);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }
    let wire = topo.wire(face_data.outer_wire())?;
    if wire.edges().len() != 4 {
        return Ok(false);
    }
    let axis = cylinder.axis();
    let mut sides: Vec<(bool, Vec<u32>)> = Vec::with_capacity(4);
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let Some(gids) = edge_global_indices.get(&oe.edge().index()) else {
            return Ok(false);
        };
        let mut side = gids.clone();
        if !oe.is_forward() {
            side.reverse();
        }
        let arc = matches!(edge.curve(), EdgeCurve::Circle(_)) && edge.start() != edge.end();
        if arc {
            if side.len() < 2 {
                return Ok(false);
            }
        } else {
            if side.len() != 2 {
                return Ok(false);
            }
            let chord = position_of(merged, side[1]) - position_of(merged, side[0]);
            if chord.cross(axis).length() > 1e-9 * chord.length().max(1.0) {
                return Ok(false);
            }
        }
        sides.push((arc, side));
    }
    if (0..4).any(|k| sides[k].1.last() != sides[(k + 1) % 4].1.first()) {
        return Ok(false);
    }
    let Some(first) =
        (0..2).find(|&k| sides[k].0 && sides[k + 2].0 && !sides[k + 1].0 && !sides[(k + 3) % 4].0)
    else {
        return Ok(false);
    };
    sides.rotate_left(first);
    let lower = &sides[0].1;
    let upper: Vec<u32> = sides[2].1.iter().rev().copied().collect();

    let idx_start = merged.indices.len();
    // Across the axis only: the arcs' distance along it is the same for
    // every pair, and on a long strip it would drown the difference.
    let gap = |merged: &TriangleMesh, a: u32, b: u32| {
        let d = position_of(merged, a) - position_of(merged, b);
        (d - axis * d.dot(axis)).length()
    };
    let turn = |p: Point3| cylinder.project_point(p).0.rem_euclid(TAU);
    let upper_turns: Vec<f64> = upper
        .iter()
        .map(|&g| turn(position_of(merged, g)))
        .collect();
    let lower_turns: Vec<f64> = lower
        .iter()
        .map(|&g| turn(position_of(merged, g)))
        .collect();
    // The turns of each triangle's corners, in any order.
    let mut tri_turns: Vec<[f64; 3]> = Vec::with_capacity(upper.len() + lower.len());
    let (mut i, mut l) = (0, 0);
    while i + 1 < upper.len() || l + 1 < lower.len() {
        let lower_next = l + 1 < lower.len()
            && (i + 1 == upper.len()
                || gap(merged, upper[i], lower[l + 1]) <= gap(merged, lower[l], upper[i + 1]));
        if lower_next {
            merged
                .indices
                .extend_from_slice(&[upper[i], lower[l], lower[l + 1]]);
            tri_turns.push([upper_turns[i], lower_turns[l], lower_turns[l + 1]]);
            l += 1;
        } else {
            merged
                .indices
                .extend_from_slice(&[upper[i], lower[l], upper[i + 1]]);
            tri_turns.push([upper_turns[i], lower_turns[l], upper_turns[i + 1]]);
            i += 1;
        }
    }

    // One winding for the whole strip, then every triangle must face out of
    // the cylinder, and the arc of the turn it spans must be no longer than
    // a chord may span at the deflection, nor than `SQRT_2` times the angle
    // tolerance.
    let (origin, radius) = (cylinder.origin(), cylinder.radius());
    let outward = |merged: &TriangleMesh, t: &[u32]| {
        let [a, b, c] = [0, 1, 2].map(|k| position_of(merged, t[k]));
        let middle = a + ((b - a) + (c - a)) * (1.0 / 3.0);
        let off_axis = middle - origin;
        let radial = off_axis - axis * off_axis.dot(axis);
        (b - a).cross(c - a).dot(radial)
    };
    let facing: f64 = merged.indices[idx_start..]
        .chunks_exact(3)
        .map(|t| outward(merged, t))
        .sum();
    if facing < 0.0 {
        for t in merged.indices[idx_start..].chunks_exact_mut(3) {
            t.swap(1, 2);
        }
    }
    let by_angle = if angular_tol > 0.0 {
        angular_tol * std::f64::consts::SQRT_2
    } else {
        TAU
    };
    let reach = (2.0 * (1.0 - (deflection / radius).min(1.0)).acos()).min(by_angle);
    let sound = merged.indices[idx_start..]
        .chunks_exact(3)
        .zip(&tri_turns)
        .all(|(t, &turns)| {
            let mut turns = turns;
            turns.sort_by(f64::total_cmp);
            // The shortest arc holding all three: the turn less its widest gap.
            let widest = (turns[1] - turns[0])
                .max(turns[2] - turns[1])
                .max(TAU - (turns[2] - turns[0]));
            outward(merged, t) > 0.0 && TAU - widest <= reach
        });
    if !sound {
        merged.indices.truncate(idx_start);
        return Ok(false);
    }
    Ok(true)
}

/// The triangles on one edge of a patch: up to two, and a mark for more.
#[derive(Clone, Copy)]
struct Owners([usize; 2], u8);

impl Default for Owners {
    fn default() -> Self {
        Self([usize::MAX; 2], 0)
    }
}

impl Owners {
    fn add(&mut self, triangle: usize) {
        if let Some(slot) = self.0.get_mut(usize::from(self.1)) {
            *slot = triangle;
        }
        self.1 = self.1.saturating_add(1);
    }

    fn pair(&self) -> Option<[usize; 2]> {
        (self.1 == 2).then_some(self.0)
    }

    fn replace(&mut self, old: usize, new: usize) {
        for slot in &mut self.0 {
            if *slot == old {
                *slot = new;
            }
        }
    }
}

fn position_of(merged: &TriangleMesh, gid: u32) -> Point3 {
    merged.positions[gid as usize]
}

/// Take back what a structured mesher added from `pos_start`, `idx_start`
/// and `ptg_start` on.
fn rollback(
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
    pos_start: usize,
    idx_start: usize,
    ptg_start: usize,
) {
    merged.positions.truncate(pos_start);
    merged.normals.truncate(pos_start);
    merged.indices.truncate(idx_start);
    if point_to_global.len() > ptg_start {
        #[allow(clippy::cast_possible_truncation)]
        point_to_global.retain(|_, gid| (*gid as usize) < pos_start);
    }
}

/// Make a contiguous run of triangles (added from `idx_start` onward) wind
/// consistently outward. The run is already wound coherently (one orientation)
/// by construction; this only decides whether that single orientation needs a
/// global flip, using the largest-area triangle (most reliable normal) against
/// the surface outward normal at its centroid.
fn orient_triangle_run(
    merged: &mut TriangleMesh,
    idx_start: usize,
    project: &dyn Fn(Point3) -> (f64, f64),
    surf_normal: &dyn Fn(f64, f64) -> Vec3,
) {
    let mut best_area = 0.0_f64;
    let mut flip = false;
    let mut t = idx_start;
    while t + 3 <= merged.indices.len() {
        let (a, b, c) = (
            merged.indices[t],
            merged.indices[t + 1],
            merged.indices[t + 2],
        );
        let (pa, pb, pc) = (
            merged.positions[a as usize],
            merged.positions[b as usize],
            merged.positions[c as usize],
        );
        let geo = (pb - pa).cross(pc - pa);
        let area = geo.length();
        if area > best_area {
            best_area = area;
            let centroid = Point3::new(
                (pa.x() + pb.x() + pc.x()) / 3.0,
                (pa.y() + pb.y() + pc.y()) / 3.0,
                (pa.z() + pb.z() + pc.z()) / 3.0,
            );
            let (u, v) = project(centroid);
            flip = geo.dot(surf_normal(u, v)) < 0.0;
        }
        t += 3;
    }
    if flip {
        let mut t = idx_start;
        while t + 3 <= merged.indices.len() {
            merged.indices.swap(t + 1, t + 2);
            t += 3;
        }
    }
}

/// Connect two column-aligned rings (identical longitude order and count) as a
/// quad strip: column `i` of `lo` joins column `i` of `hi`. Each quad is split
/// into two triangles via the supplied `emit` closure. The collar path passes
/// `emit_raw` (no per-triangle winding correction — the whole run is oriented
/// once afterward by [`orient_triangle_run`], which is stable for the thin
/// stitch triangles). Watertight by construction when the rings share columns.
fn emit_aligned_quad_strip(
    merged: &mut TriangleMesh,
    lo: &LatRing,
    hi: &LatRing,
    emit: &impl Fn(&mut TriangleMesh, u32, u32, u32),
) {
    let n = lo.len();
    if n < 2 || hi.len() != n {
        // Counts diverged (a merged-away duplicate column) — fall back to the
        // longitude zipper, which tolerates unequal counts.
        stitch_rings(merged, lo, hi, emit);
        return;
    }
    for i in 0..n {
        let j = (i + 1) % n;
        let (l0, l1) = (lo[i].1, lo[j].1);
        let (h0, h1) = (hi[i].1, hi[j].1);
        emit(merged, l0, l1, h1);
        emit(merged, l0, h1, h0);
    }
}

/// Collect a wire's shared boundary vertices as a `(v_level, ring)` pair, or
/// `None` if the wire is not a closed full-revolution loop at a single constant
/// `v` (built only from `Line`/`Circle` edges).
fn collect_constant_v_ring(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    project: &dyn Fn(Point3) -> (f64, f64),
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &TriangleMesh,
) -> Result<Option<(f64, LatRing)>, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut gids: Vec<u32> = Vec::new();
    for oe in wire.edges() {
        let e = topo.edge(oe.edge())?;
        match e.curve() {
            EdgeCurve::Line | EdgeCurve::Circle(_) => {}
            _ => return Ok(None),
        }
        let Some(edge_gids) = edge_global_indices.get(&oe.edge().index()) else {
            return Ok(None);
        };
        for &g in edge_gids {
            gids.push(g);
        }
    }
    if gids.len() < 3 {
        return Ok(None);
    }

    // Deduplicate to unique global IDs and check they all sit at one constant v
    // while their longitudes cover the full circle (a full revolution).
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    let mut ring: LatRing = Vec::with_capacity(gids.len());
    let mut v_sum = 0.0;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for g in gids {
        if !seen.insert(g) {
            continue;
        }
        let p = merged.positions[g as usize];
        let (u, v) = project(p);
        v_sum += v;
        v_min = v_min.min(v);
        v_max = v_max.max(v);
        ring.push((u, g));
    }
    if ring.len() < 3 {
        return Ok(None);
    }
    if (v_max - v_min) > 1e-6 {
        return Ok(None);
    }
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Full-revolution check: the largest angular gap between consecutive
    // longitudes (including the wrap-around) must be well under a full turn —
    // otherwise this is a partial arc, not a closed latitude loop.
    let max_gap = ring
        .windows(2)
        .map(|w| w[1].0 - w[0].0)
        .chain(std::iter::once(
            ring[0].0 + std::f64::consts::TAU - ring[ring.len() - 1].0,
        ))
        .fold(0.0_f64, f64::max);
    if max_gap > std::f64::consts::PI {
        return Ok(None);
    }

    let v_level = v_sum / ring.len() as f64;
    Ok(Some((v_level, ring)))
}

/// A boundary ring whose latitude varies with longitude: `(u_angle, v, gid)`
/// sorted ascending by `u_angle`. Used for a collar's scalloped outer wire (the
/// great-circle/seam-arc "floor" of a box∩sphere patch), which encircles
/// longitude fully but at a non-constant `v`.
type VarRing = Vec<(f64, f64, u32)>;

/// Collect a wire's shared boundary vertices as a longitude-sorted [`VarRing`],
/// or `None` if the wire is not a closed full-revolution loop (built only from
/// `Line`/`Circle` edges). Unlike [`collect_constant_v_ring`], the latitude may
/// vary with longitude. A loop through a sphere's pole winds nothing, however
/// much of the turn its samples cover.
fn collect_var_v_ring(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
    surface: &FaceSurface,
    project: &dyn Fn(Point3) -> (f64, f64),
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &TriangleMesh,
) -> Result<Option<VarRing>, crate::OperationsError> {
    let wire = topo.wire(wire_id)?;
    let mut gids: Vec<u32> = Vec::new();
    for oe in wire.edges() {
        let e = topo.edge(oe.edge())?;
        match e.curve() {
            EdgeCurve::Line | EdgeCurve::Circle(_) => {}
            _ => return Ok(None),
        }
        let Some(edge_gids) = edge_global_indices.get(&oe.edge().index()) else {
            return Ok(None);
        };
        if oe.is_forward() {
            gids.extend_from_slice(edge_gids);
        } else {
            gids.extend(edge_gids.iter().rev());
        }
    }
    if gids.len() < 3 {
        return Ok(None);
    }
    let poles = SurfacePoles::of(surface);
    let us: Vec<Option<f64>> = gids
        .iter()
        .map(|&g| {
            let point = merged.positions[g as usize];
            poles.v_of(point).is_none().then(|| project(point).0)
        })
        .collect();
    let (_, winding) = unwrap_loop_u(&us, TAU);
    if (winding.abs() - TAU).abs() > 1e-6 {
        return Ok(None);
    }
    let mut seen: DetHashSet<u32> = DetHashSet::default();
    let mut ring: VarRing = Vec::with_capacity(gids.len());
    for g in gids {
        if !seen.insert(g) {
            continue;
        }
        let (u, v) = project(merged.positions[g as usize]);
        ring.push((u, v, g));
    }
    if ring.len() < 3 {
        return Ok(None);
    }
    ring.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Full-revolution check: the largest longitude gap (including wrap-around)
    // must be under a full turn — else it is a partial arc, not a closed loop.
    let max_gap = ring
        .windows(2)
        .map(|w| w[1].0 - w[0].0)
        .chain(std::iter::once(
            ring[0].0 + std::f64::consts::TAU - ring[ring.len() - 1].0,
        ))
        .fold(0.0_f64, f64::max);
    if max_gap > std::f64::consts::PI {
        return Ok(None);
    }
    Ok(Some(ring))
}

/// Build an interior latitude row of `n` evenly-spaced new vertices at constant
/// `v`, returning them as a ring sorted by longitude.
fn build_interior_row(
    v: f64,
    n: usize,
    surf_eval: &dyn Fn(f64, f64) -> Point3,
    surf_normal: &dyn Fn(f64, f64) -> Vec3,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> LatRing {
    let mut row: LatRing = Vec::with_capacity(n);
    for i in 0..n {
        let u = std::f64::consts::TAU * (i as f64) / (n as f64);
        let p = surf_eval(u, v);
        let key = point_merge_key(p, MERGE_GRID);
        let gid = *point_to_global.entry(key).or_insert_with(|| {
            let idx = merged.positions.len() as u32;
            merged.positions.push(p);
            merged.normals.push(surf_normal(u, v));
            idx
        });
        row.push((u, gid));
    }
    row
}

/// Build a collar interior row at the floor ring's exact longitudes — one
/// column per floor vertex — each column's `v` interpolated a fraction `t` from
/// that floor vertex's `v` up to the constant cap latitude `v_cap`. Keeping the
/// interior rows column-aligned with the scalloped floor lets them connect as
/// clean quad strips (no longitude zippering, so the scallop corners — where
/// the floor dips to the seam — produce no flipped slivers).
fn build_collar_row(
    floor: &VarRing,
    v_cap: f64,
    t: f64,
    surf_eval: &dyn Fn(f64, f64) -> Point3,
    surf_normal: &dyn Fn(f64, f64) -> Vec3,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> LatRing {
    let mut row: LatRing = Vec::with_capacity(floor.len());
    for &(u, v_floor, _) in floor {
        let v = v_floor + (v_cap - v_floor) * t;
        let p = surf_eval(u, v);
        let key = point_merge_key(p, MERGE_GRID);
        let gid = *point_to_global.entry(key).or_insert_with(|| {
            let idx = merged.positions.len() as u32;
            merged.positions.push(p);
            merged.normals.push(surf_normal(u, v));
            idx
        });
        row.push((u, gid));
    }
    row
}

/// Emit a default-oriented (non-reversed) triangle, mirroring the orientation
/// convention of [`tessellate_revolution_band_shared`]: the geometric normal is
/// flipped to match the surface outward normal. The caller applies the global
/// `is_reversed` winding flip afterward.
fn make_band_emit<'a>(
    project: &'a dyn Fn(Point3) -> (f64, f64),
    surf_normal: &'a dyn Fn(f64, f64) -> Vec3,
) -> impl Fn(&mut TriangleMesh, u32, u32, u32) + 'a {
    move |merged: &mut TriangleMesh, a: u32, b: u32, c: u32| {
        if a == b || b == c || a == c {
            return;
        }
        let (pa, pb, pc) = (
            merged.positions[a as usize],
            merged.positions[b as usize],
            merged.positions[c as usize],
        );
        let geo = (pb - pa).cross(pc - pa);
        if geo.length() < 1e-20 {
            return;
        }
        // Reference the outward normal at all three vertices (averaged), not just
        // `pa`: a thin stitch triangle bridging a clustered ring to an even one
        // can sit nearly tangent to the surface, where the single-vertex normal
        // makes `geo.dot(outward)` sign-unstable and flips the triangle relative
        // to its neighbours. The averaged normal is stable across the triangle.
        let n_at = |p: Point3| -> Vec3 {
            let (u, v) = project(p);
            surf_normal(u, v)
        };
        let outward = n_at(pa) + n_at(pb) + n_at(pc);
        let mut tri = [a, b, c];
        if geo.dot(outward) < 0.0 {
            tri.swap(1, 2);
        }
        merged.indices.extend_from_slice(&tri);
    }
}

/// Triangulate the band between two coaxial latitude rings, both sorted by
/// longitude in `[0, 2π)`, whose vertex counts/phases may differ. Walks both
/// rings forward in longitude, at each step advancing whichever ring's next
/// vertex has the smaller longitude (relative to a monotonically increasing
/// base) and emitting one triangle per advance. Watertight by construction:
/// every interior quad diagonal is shared by exactly two triangles, and after
/// `nl + nh` advances each ring has been traversed once back to its start.
fn stitch_rings(
    merged: &mut TriangleMesh,
    lo: &LatRing,
    hi: &LatRing,
    emit: &impl Fn(&mut TriangleMesh, u32, u32, u32),
) {
    if lo.len() < 2 || hi.len() < 2 {
        return;
    }
    let (nl, nh) = (lo.len(), hi.len());
    // Precompute the unwrapped (strictly increasing) longitude reached after
    // `k` forward steps on each ring, k = 0..=len. Step 0 is the ring's first
    // longitude; step len returns to it plus one full turn.
    let unwrap_ring = |ring: &LatRing| -> Vec<f64> {
        let mut acc = Vec::with_capacity(ring.len() + 1);
        let mut prev = ring[0].0;
        acc.push(prev);
        for k in 1..=ring.len() {
            let raw = ring[k % ring.len()].0;
            // Forward gap to the next vertex, in (0, 2π]: a full turn on the
            // wrap-around step (k == len), the spacing otherwise.
            let mut gap = (raw - prev).rem_euclid(std::f64::consts::TAU);
            if gap <= 0.0 {
                gap = std::f64::consts::TAU;
            }
            prev += gap;
            acc.push(prev);
        }
        acc
    };
    let lo_ang = unwrap_ring(lo);
    let hi_ang = unwrap_ring(hi);

    // Each ring is advanced exactly once around (nl + nh advances total). Once a
    // ring has completed its revolution (`i == nl` / `j == nh`) it must not
    // advance again, so its "next longitude" is treated as +inf.
    let (mut i, mut j) = (0usize, 0usize);
    for _ in 0..(nl + nh) {
        let li = lo[i % nl].1;
        let hj = hi[j % nh].1;
        let lo_next = if i < nl { lo_ang[i + 1] } else { f64::INFINITY };
        let hi_next = if j < nh { hi_ang[j + 1] } else { f64::INFINITY };
        // Advance whichever ring's next vertex comes first in longitude; the new
        // triangle's apex stays on the ring that did not advance.
        if lo_next <= hi_next {
            let li_next = lo[(i + 1) % nl].1;
            emit(merged, li, li_next, hj);
            i += 1;
        } else {
            let hj_next = hi[(j + 1) % nh].1;
            emit(merged, li, hj_next, hj);
            j += 1;
        }
    }
}

/// `BK_CDT_TRACE` (any value): log CDT boundary sourcing and UV mapping.
/// Resolved ONCE per process: the checks sit in per-edge loops.
fn cdt_trace() -> bool {
    static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *TRACE.get_or_init(|| std::env::var("BK_CDT_TRACE").is_ok())
}

/// CDT-based tessellation for non-planar faces with exact boundary constraints.
///
/// Projects shared edge points into (u,v) parameter space, generates interior
/// sample points, then runs Constrained Delaunay Triangulation. Boundary
/// vertices use their pre-existing global IDs (watertight by construction).
///
/// # Errors
///
/// Returns an error when the boundary has fewer than three vertices, when the
/// CDT itself fails, or when a developable fillet stripe cannot be brought
/// inside `deflection`. Every caller in `solid.rs` answers an error by
/// re-meshing the face with `tessellate_nonplanar_snap`, which is watertight
/// but honours no deflection bound, so the stripe case logs a warning first
/// and the sag leaves a trace.
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_nonplanar_cdt(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<(), crate::OperationsError> {
    // Every seam, pole and winding-loop rule below reads the closed direction
    // as u. A NURBS face closed along v alone (a revolved fillet whose profile
    // runs along u) is meshed on the same surface with its parameters
    // exchanged, which leaves every point and normal where it was.
    if let FaceSurface::Nurbs(nurbs) = face_data.surface()
        && nurbs.is_periodic_v()
        && !nurbs.is_periodic_u()
    {
        let (exchanged, mirror) = exchange_nurbs_parameters(nurbs)?;
        let mut local = face_data.clone();
        local.set_surface(FaceSurface::Nurbs(exchanged));
        let pcurve_uv = |edge: brepkit_topology::edge::EdgeId, pt: Point3| {
            topo.pcurves()
                .get(edge, face_id)
                .and_then(|pcurve| project_via_pcurve(pcurve, pt, face_data.surface()))
                .map(|(u, v)| (v, mirror - u))
        };
        return cdt_in_surface_frame(
            topo,
            face_id,
            &local,
            &pcurve_uv,
            deflection,
            angular_tol,
            circle_floor,
            edge_global_indices,
            merged,
            point_to_global,
        );
    }
    // The edges' pcurves lie in the face's own chart, so a face put on a new
    // one reads its samples' (u, v) by projection.
    if let Some(local) = sphere_face_recharted(topo, face_data)? {
        return cdt_in_surface_frame(
            topo,
            face_id,
            &local,
            &|_, _| None,
            deflection,
            angular_tol,
            circle_floor,
            edge_global_indices,
            merged,
            point_to_global,
        );
    }
    let pcurve_uv = |edge: brepkit_topology::edge::EdgeId, pt: Point3| {
        topo.pcurves()
            .get(edge, face_id)
            .and_then(|pcurve| project_via_pcurve(pcurve, pt, face_data.surface()))
    };
    cdt_in_surface_frame(
        topo,
        face_id,
        face_data,
        &pcurve_uv,
        deflection,
        angular_tol,
        circle_floor,
        edge_global_indices,
        merged,
        point_to_global,
    )
}

/// A NURBS surface with its parameters exchanged, `T(p, q) = S(mirror - q,
/// p)`, and that `mirror`. Running the old u backwards keeps `dT/dp x dT/dq`
/// equal to `dS/du x dS/dv`, so the normal does not flip, and keeps the new
/// v on the old u's domain.
fn exchange_nurbs_parameters(
    nurbs: &brepkit_math::nurbs::surface::NurbsSurface,
) -> Result<(brepkit_math::nurbs::surface::NurbsSurface, f64), crate::OperationsError> {
    let (u_lo, u_hi) = nurbs.domain_u();
    let mirror = u_lo + u_hi;
    let points = nurbs.control_points();
    let weights = nurbs.weights();
    let cols = points.first().map_or(0, Vec::len);
    let control_points = (0..cols)
        .map(|j| points.iter().rev().map(|row| row[j]).collect())
        .collect();
    let exchanged_weights = (0..cols)
        .map(|j| weights.iter().rev().map(|row| row[j]).collect())
        .collect();
    let knots_v = nurbs.knots_u().iter().rev().map(|k| mirror - k).collect();
    let exchanged = brepkit_math::nurbs::surface::NurbsSurface::new(
        nurbs.degree_v(),
        nurbs.degree_u(),
        nurbs.knots_v().to_vec(),
        knots_v,
        control_points,
        exchanged_weights,
    )
    .map_err(crate::OperationsError::Math)?;
    Ok((exchanged, mirror))
}

/// The CDT mesher proper, for a face whose closed direction (if any) is u.
/// `pcurve_uv` gives an edge sample's `(u, v)` from the edge's pcurve on the
/// face, in this frame.
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
fn cdt_in_surface_frame(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    pcurve_uv: &dyn Fn(brepkit_topology::edge::EdgeId, Point3) -> Option<(f64, f64)>,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<(), crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use brepkit_topology::edge::EdgeId;

    let wire = topo.wire(face_data.outer_wire())?;
    let tol_dup = 1e-10;

    let anchored = anchor_closed_edges_at_vertices(topo, wire, edge_global_indices, merged)?;

    // A pointed cone's wall is walked from a seam edge that follows another
    // edge: every other stretch then runs between seam samples, and no seam
    // run loses its samples to the loop's closure. Walked from a rim, the
    // stretch after the seam stays where the unwrap through the apex left
    // it, a period off the seam copy it starts from. A wall whose seam a
    // hole notches (more than one seam edge) is told apart below.
    let cone_seams = pointed_cone_seams(topo, face_data, wire)?;
    let notched_cone = cone_seams > 1;
    let mut edges = wire.edges().to_vec();
    if cone_seams > 0 {
        let seam = |i: usize| {
            edges
                .iter()
                .filter(|other| other.edge() == edges[i].edge())
                .count()
                > 1
        };
        let n = edges.len();
        if let Some(k) = (0..n).find(|&i| seam(i) && !seam((i + n - 1) % n)) {
            edges.rotate_left(k);
        }
    }

    // Fourth element: is_forward flag -- needed for seam UV assignment.
    let mut boundary_3d: Vec<(Point3, u32, EdgeId, bool)> = Vec::new();
    for oe in &edges {
        let edge_id_local = oe.edge();
        let edge_idx = edge_id_local.index();
        let is_fwd = oe.is_forward();
        if let Some(global_ids) = anchored
            .get(&edge_idx)
            .or_else(|| edge_global_indices.get(&edge_idx))
        {
            if cdt_trace() {
                log::debug!(
                    "cdt {face_id:?} edge e{edge_idx} SHARED n={} gids {}..{}",
                    global_ids.len(),
                    global_ids.first().copied().unwrap_or(0),
                    global_ids.last().copied().unwrap_or(0)
                );
            }
            let ordered: Vec<u32> = if is_fwd {
                global_ids.clone()
            } else {
                global_ids.iter().rev().copied().collect()
            };
            for (j, &gid) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (_, last_gid, _, _) = boundary_3d[boundary_3d.len() - 1];
                    if last_gid == gid
                        || (merged.positions[last_gid as usize] - merged.positions[gid as usize])
                            .length()
                            < tol_dup
                    {
                        continue;
                    }
                }
                boundary_3d.push((merged.positions[gid as usize], gid, edge_id_local, is_fwd));
            }
        } else {
            if cdt_trace() {
                log::debug!("cdt {face_id:?} edge e{edge_idx} RESAMPLED");
            }
            // Edge not in shared pool -- insert directly.
            let edge_data = topo.edge(oe.edge())?;
            let points = sample_edge(topo, edge_data, deflection, angular_tol, circle_floor)?;
            let ordered: Vec<Point3> = if is_fwd {
                points
            } else {
                points.into_iter().rev().collect()
            };
            for (j, &pt) in ordered.iter().enumerate() {
                if j == 0 && !boundary_3d.is_empty() {
                    let (last_pos, _, _, _) = boundary_3d[boundary_3d.len() - 1];
                    if (last_pos - pt).length() < tol_dup {
                        continue;
                    }
                }
                let key = point_merge_key(pt, MERGE_GRID);
                let gid = *point_to_global.entry(key).or_insert_with(|| {
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(pt);
                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                    idx
                });
                boundary_3d.push((pt, gid, edge_id_local, is_fwd));
            }
        }
    }

    if boundary_3d.len() > 2
        && let (Some(&(_, first_gid, _, _)), Some(&(_, last_gid, _, _))) =
            (boundary_3d.first(), boundary_3d.last())
        && (first_gid == last_gid
            || (merged.positions[first_gid as usize] - merged.positions[last_gid as usize])
                .length()
                < tol_dup)
    {
        boundary_3d.pop();
    }

    let ring = whole_ring_rectangle(
        topo,
        face_data,
        &boundary_3d,
        deflection,
        angular_tol,
        circle_floor,
        merged,
        point_to_global,
    )?;
    if let Some((_, samples)) = &ring {
        boundary_3d.clone_from(samples);
    }

    let n_boundary = boundary_3d.len();
    if n_boundary < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "non-planar face has fewer than 3 boundary vertices".to_string(),
        });
    }

    let mut boundary_uv: Vec<(f64, f64)> = if let Some((uvs, _)) = ring.as_ref() {
        uvs.clone()
    } else {
        boundary_3d
            .iter()
            .map(|(pt, _, edge_id_local, _)| {
                pcurve_uv(*edge_id_local, *pt)
                    .map_or_else(|| project_to_surface_uv(face_data.surface(), *pt), Ok)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

    // Step 2a: Unwrap periodic u across the seam for polyline boundaries.
    {
        let (u_period, v_period) = surface_periods(face_data.surface());
        if let Some((u_origin, u_period)) = u_period
            && !boundary_uv.is_empty()
        {
            // A point on the surface's degenerate locus has no meaningful u
            // (a horn torus pinches onto its axis at tube angle v = pi; the
            // projection returns an arbitrary ring angle there). Left as
            // projected, such a point can steer the consecutive unwrap the
            // LONG way around the period, leaving the loop unclosed by a
            // full turn — the UV polygon then self-overlaps and
            // remove_exterior eats the triangles along the boundary strip.
            // Give a degenerate point its predecessor's u so the unwrap
            // steps over it neutrally.
            let degenerate_u = |v: f64| -> bool {
                if let FaceSurface::Torus(t) = face_data.surface() {
                    (t.major_radius() + t.minor_radius() * v.cos()).abs() < t.minor_radius() * 1e-6
                } else {
                    false
                }
            };
            // The boundary is cyclic, so the anchor itself can sit on the
            // degenerate locus (the wire may start at the pinch); unwrap
            // from the first NON-degenerate point instead so an arbitrary
            // anchor u never steers the walk.
            let n = boundary_uv.len();
            let start = (0..n)
                .find(|&i| !degenerate_u(boundary_uv[i].1))
                .unwrap_or(0);
            for k in 1..n {
                let i = (start + k) % n;
                let prev = (start + k + n - 1) % n;
                let prev_u = boundary_uv[prev].0;
                if degenerate_u(boundary_uv[i].1) {
                    boundary_uv[i].0 = prev_u;
                    continue;
                }
                let mut u = boundary_uv[i].0;
                let diff = u - prev_u;
                let shifts = (diff / u_period + 0.5).floor();
                u -= shifts * u_period;
                boundary_uv[i].0 = u;
            }
            // A pole (a sphere's, or a u-closed NURBS cap's) has no u of its
            // own: a loop through it runs along the pole's row, which also
            // keeps a loop that opens a wedge at the pole from reading as a
            // cap over it.
            let points: Vec<Point3> = boundary_3d.iter().map(|b| b.0).collect();
            if let Some(routed) =
                route_through_poles(face_data.surface(), &boundary_uv, &points, u_period)
            {
                boundary_3d = routed.iter().map(|&(i, _, _)| boundary_3d[i]).collect();
                boundary_uv = routed.into_iter().map(|(_, u, v)| (u, v)).collect();
            }
            let first_u = boundary_uv[0].0;
            let last_u = boundary_uv.last().map_or(first_u, |p| p.0);
            let close_diff = first_u - last_u;
            if close_diff.abs() > u_period / 2.0 {
                let u_mid = boundary_uv.iter().map(|p| p.0).sum::<f64>() / boundary_uv.len() as f64;
                let target_mid = u_origin + u_period / 2.0;
                // Whole periods only: any other shift moves the samples off
                // the points their (u, v) evaluates to.
                let shift = ((target_mid - u_mid) / u_period).round() * u_period;
                for pt in &mut boundary_uv {
                    pt.0 += shift;
                }
            }
        }

        // The tube angle (v) is periodic on a torus too. A toroidal band (a rim
        // fillet) is bounded by two rims at distinct v, joined by a seam where v
        // jumps by nearly a full turn; without unwrapping, the v-bbox spans the
        // long arc (the bulging 270° of the tube) instead of the short fillet
        // arc, and the interior CDT samples cover the wrong side. Unwrap v the
        // same way u is unwrapped so consecutive boundary points stay within
        // half a turn, collapsing the band to its true (short-arc) v-extent.
        if let Some((_, v_period)) = v_period
            && !boundary_uv.is_empty()
        {
            for i in 1..boundary_uv.len() {
                let prev_v = boundary_uv[i - 1].1;
                let mut v = boundary_uv[i].1;
                let diff = v - prev_v;
                let shifts = (diff / v_period + 0.5).floor();
                v -= shifts * v_period;
                boundary_uv[i].1 = v;
            }
        }
    }

    close_loop_at_pole(
        topo,
        face_data,
        &mut boundary_uv,
        &mut boundary_3d,
        deflection,
        angular_tol,
        merged,
        point_to_global,
    )?;
    let n_boundary = boundary_3d.len();

    // Compute (u,v) bounding box from a set of UV pairs.
    #[allow(clippy::items_after_statements)]
    fn uv_bounds(uvs: &[(f64, f64)]) -> (f64, f64, f64, f64) {
        uvs.iter().fold(
            (
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ),
            |(u_lo, u_hi, v_lo, v_hi), &(u, v)| {
                (u_lo.min(u), u_hi.max(u), v_lo.min(v), v_hi.max(v))
            },
        )
    }
    let (u_min, u_max, v_min, v_max) = uv_bounds(&boundary_uv);

    // Step 2b: Detect and fix degenerate seam edges.
    let (u_min, u_max, v_min, v_max) = {
        let mut wire_edge_counts: DetHashMap<usize, usize> = DetHashMap::default();
        for oe in wire.edges() {
            *wire_edge_counts.entry(oe.edge().index()).or_default() += 1;
        }
        let seam_edge_indices: DetHashSet<usize> = wire_edge_counts
            .iter()
            .filter(|&(_, &c)| c > 1)
            .map(|(&idx, _)| idx)
            .collect();

        // A torus sector's seam runs along the ring (a latitude arc, or a
        // curve winding the tube), not along a meridian: its samples keep
        // their unwrapped (u, v), which the placement below would pin to one u.
        // Read along each run of seam samples: a meridian seam's two copies
        // lie a period apart in u, but neither run moves in u.
        let seam_along_ring = matches!(face_data.surface(), FaceSurface::Torus(_)) && {
            let on_seam = |i: usize| seam_edge_indices.contains(&boundary_3d[i].2.index());
            (0..n_boundary).any(|i| {
                let j = (i + 1) % n_boundary;
                on_seam(i)
                    && on_seam(j)
                    && boundary_3d[i].2 == boundary_3d[j].2
                    && (boundary_uv[j].0 - boundary_uv[i].0).abs() > 1e-6
            })
        };
        if !seam_edge_indices.is_empty() && ring.is_none() && !seam_along_ring {
            let off_seam = |i: usize| !seam_edge_indices.contains(&boundary_3d[i].2.index());
            let non_seam_uvs: Vec<(f64, f64)> = boundary_uv
                .iter()
                .enumerate()
                .filter(|&(i, _)| off_seam(i))
                .map(|(_, &uv)| uv)
                .collect();
            let (u_min_bnd, u_max_bnd, v_min_bnd, v_max_bnd) = if non_seam_uvs.is_empty() {
                (u_min, u_max, v_min, v_max)
            } else {
                uv_bounds(&non_seam_uvs)
            };
            // A wall that runs its seam both ways spans one period between
            // the seam's two copies, and the seam's own vertices place them:
            // a closed rim's samples stand in for its vertex only to within
            // half a step, so bounds read off the rims can fall short of the
            // period (and the wall's mesh short of its area). On a notched
            // pointed cone the halves' continuous unwrap, carried through the
            // apex's arbitrary u, can also sit a period off, and on a wall a
            // section winds round the off-seam samples would put a copy on
            // the section's sample nearest the seam.
            let seam_vertex_u = seam_vertex_u(topo, &edges, face_data.surface())?;
            let u_period = surface_periods(face_data.surface())
                .0
                .map_or(TAU, |(_, period)| period);
            let (u_min_bnd, u_max_bnd) = seam_vertex_u.map_or((u_min_bnd, u_max_bnd), |seam_u| {
                let lo = ((u_min_bnd - seam_u) / u_period)
                    .round()
                    .madd(u_period, seam_u);
                (lo, lo + u_period)
            });
            // A rim's seam vertex joins whichever run the wire reaches it in,
            // so when both rims hand theirs to the seam runs, the rims' span
            // stops a sample short of the period on that side and the seam
            // would be placed a sample inside the wall. The seam's own
            // projection fixes its u to within whole periods. A span well
            // short of the period is a face that does not wrap.
            let (u_min_bnd, u_max_bnd) = seam_span(
                face_data.surface(),
                &boundary_3d,
                &boundary_uv,
                &seam_edge_indices,
            )
            .map_or((u_min_bnd, u_max_bnd), |(seam_u, period)| {
                let slack = 1e-9 * period;
                let span = u_max_bnd - u_min_bnd;
                // A seam and a rim read their shared vertex through their
                // own pcurves, which can disagree by more than rounding (by
                // 2e-5 on a torus's fitted seam and loop): half the gap the
                // rims leave tells a seam copy at a rim's end from one a
                // sample beyond it.
                let reach = slack.max(0.5 * (period - span));
                let lo = seam_u - ((seam_u - u_min_bnd - reach) / period).ceil() * period;
                if span < period - slack && span > 0.75 * period && lo + period >= u_max_bnd - reach
                {
                    (lo, lo + period)
                } else {
                    (u_min_bnd, u_max_bnd)
                }
            });

            #[allow(clippy::items_after_statements)]
            struct SeamRun {
                indices: Vec<usize>,
                is_forward: bool,
            }
            let mut seam_runs: Vec<SeamRun> = Vec::new();
            let mut current_indices: Vec<usize> = Vec::new();
            let mut current_fwd: Option<bool> = None;
            for i in 0..n_boundary {
                let (_, _, edge_id, is_fwd) = boundary_3d[i];
                if seam_edge_indices.contains(&edge_id.index()) {
                    current_indices.push(i);
                    if current_fwd.is_none() {
                        current_fwd = Some(is_fwd);
                    }
                } else if !current_indices.is_empty() {
                    seam_runs.push(SeamRun {
                        indices: std::mem::take(&mut current_indices),
                        is_forward: current_fwd.unwrap_or(true),
                    });
                    current_fwd = None;
                }
            }
            if !current_indices.is_empty() {
                let tail_fwd = current_fwd.unwrap_or(true);
                if !seam_runs.is_empty()
                    && seam_edge_indices.contains(&boundary_3d[0].2.index())
                    && seam_runs[0].is_forward == tail_fwd
                {
                    current_indices.extend(seam_runs.remove(0).indices);
                }
                seam_runs.push(SeamRun {
                    indices: current_indices,
                    is_forward: tail_fwd,
                });
            }

            // A pointed cone's seam runs up to the apex and back: after its
            // halves are put on the seam's two sides, the apex row joining
            // them is inserted here, `(after index, from u, to u, v, id)`.
            let mut apex_rows: Vec<(usize, f64, f64, f64, u32)> = Vec::new();
            // A holed cone, one whose seam a hole notches (the seam is then
            // more than one edge), or a tip a section bounds: its wall takes
            // the refined developable metric below; a whole cone keeps the
            // snap mesher's grid.
            let section_tip = is_section_tip(topo, face_data)?;
            let apex = match face_data.surface() {
                FaceSurface::Cone(cone)
                    if !face_data.inner_wires().is_empty()
                        || seam_edge_indices.len() > 1
                        || section_tip =>
                {
                    Some(cone.apex())
                }
                _ => None,
            };
            let poles = SurfacePoles::of(face_data.surface());
            for run in &seam_runs {
                // A seam up to a sphere's or NURBS cap's pole and back is one
                // run, which `route_through_poles` has already laid out: each
                // half on its own copy of the seam, at the u its unwrap
                // reached, and the pole opened into the stretch of its row
                // between them. One u for the whole run would fold it onto a
                // line.
                if run
                    .indices
                    .iter()
                    .any(|&i| poles.v_of(boundary_3d[i].0).is_some())
                {
                    continue;
                }
                // The rim sample just before the run is the seam's own vertex
                // on that rim, already unwrapped to the seam's side of the
                // span; the edge's sense only says which side for one wire
                // orientation.
                let before = (run.indices[0] + n_boundary - 1) % n_boundary;
                // On a notched cone the samples beside a seam copy lie on the
                // wall's side of it: +u of the lower copy, -u of the upper.
                // Read from the wrapped offset to the seam, the side needs no
                // unwrap the apex could have thrown a period off. The sample
                // before the run is the seam's own vertex, which a closed rim
                // stands in for with its nearest sample, up to half a step
                // to either side.
                let beside = || -> Option<f64> {
                    let seam_u = seam_vertex_u?;
                    let mut i = (before + n_boundary - 1) % n_boundary;
                    while off_seam(i) {
                        let d = (boundary_uv[i].0 - seam_u + 0.5 * TAU).rem_euclid(TAU) - 0.5 * TAU;
                        if d.abs() > 1e-9 {
                            return Some(if d > 0.0 { u_min_bnd } else { u_max_bnd });
                        }
                        i = (i + n_boundary - 1) % n_boundary;
                        if i == before {
                            break;
                        }
                    }
                    None
                };
                let u_assign = if let Some(u) = notched_cone.then(beside).flatten() {
                    u
                } else if seam_edge_indices.contains(&boundary_3d[before].2.index()) {
                    if run.is_forward { u_max_bnd } else { u_min_bnd }
                } else if (boundary_uv[before].0 - u_min_bnd).abs()
                    < (boundary_uv[before].0 - u_max_bnd).abs()
                {
                    u_min_bnd
                } else {
                    u_max_bnd
                };
                let n_pts = run.indices.len();
                let at_apex = apex.and_then(|apex| {
                    run.indices
                        .iter()
                        .position(|&i| (boundary_3d[i].0 - apex).length() < 1e-9)
                });
                if let Some(turn) = at_apex {
                    // The seam's other copy is one period on, whichever rim
                    // sample the wire happened to start at.
                    let period = surface_periods(face_data.surface())
                        .0
                        .map_or(TAU, |(_, p)| p);
                    let u_other = if u_assign <= f64::midpoint(u_min_bnd, u_max_bnd) {
                        u_assign + period
                    } else {
                        u_assign - period
                    };
                    for (k, &i) in run.indices.iter().enumerate() {
                        boundary_uv[i].0 = if k <= turn { u_assign } else { u_other };
                    }
                    let i = run.indices[turn];
                    apex_rows.push((i, u_assign, u_other, boundary_uv[i].1, boundary_3d[i].1));
                    continue;
                }
                if n_pts == 1 {
                    // A lone sample is one of the seam's ends; its projected
                    // v is exact there.
                    let i = run.indices[0];
                    boundary_uv[i] = (u_assign, boundary_uv[i].1.clamp(v_min_bnd, v_max_bnd));
                    continue;
                }

                // A seam point's projected v is exact where the projection is
                // well behaved (only u is ambiguous across the seam), so the
                // run keeps it when it climbs steadily within the rims' span.
                // Spacing the run evenly by index assumes the run holds both
                // seam vertices, but a rim usually keeps the vertex it closes.
                let (v_lo, v_hi) = (v_min_bnd.min(v_max_bnd), v_min_bnd.max(v_max_bnd));
                let slack = 1e-9 * (v_hi - v_lo).max(1.0);
                let projected: Vec<f64> = run.indices.iter().map(|&i| boundary_uv[i].1).collect();
                let within = projected
                    .iter()
                    .all(|&v| v >= v_lo - slack && v <= v_hi + slack);
                // The rims' span comes from their samples, and the sample a
                // closed rim keeps for the seam's vertex can fall short of the
                // vertex's v; with no span at all, nothing is spaced across.
                let reach = (1e-3 * (v_hi - v_lo)).max(slack);
                let within = within
                    || v_hi - v_lo <= slack
                    || projected
                        .iter()
                        .all(|&v| v >= v_lo - reach && v <= v_hi + reach);
                let steady = projected.windows(2).all(|w| w[1] > w[0])
                    || projected.windows(2).all(|w| w[1] < w[0]);
                if within && steady {
                    for &i in &run.indices {
                        boundary_uv[i].0 = u_assign;
                    }
                    continue;
                }

                let v_first = boundary_uv[run.indices[0]].1;
                let (v_start, v_end) = if (v_first - v_min_bnd).abs() < (v_first - v_max_bnd).abs()
                {
                    (v_min_bnd, v_max_bnd)
                } else {
                    (v_max_bnd, v_min_bnd)
                };

                for (k, &i) in run.indices.iter().enumerate() {
                    let t = if n_pts > 1 {
                        k as f64 / (n_pts - 1) as f64
                    } else {
                        0.5
                    };
                    let v = v_start + t * (v_end - v_start);
                    boundary_uv[i] = (u_assign, v);
                }
            }

            // Hang each stretch off the seam point before it, each sample
            // within half a period of the one before: a stretch can run on
            // past the wire's first sample, where the unwrap above broke.
            if notched_cone && let Some((_, period)) = surface_periods(face_data.surface()).0 {
                for i in 0..n_boundary {
                    let prev = (i + n_boundary - 1) % n_boundary;
                    if !off_seam(i) || off_seam(prev) {
                        continue;
                    }
                    let mut last = boundary_uv[prev].0;
                    let mut j = i;
                    while off_seam(j) {
                        let u = boundary_uv[j].0;
                        last = ((last - u) / period).round().madd(period, u);
                        boundary_uv[j].0 = last;
                        j = (j + 1) % n_boundary;
                        if j == i {
                            break;
                        }
                    }
                }
            }
            // A closed rim stands in for the seam's vertex with its nearest
            // sample, up to half a step off the seam: where the rim reaches
            // it, it sits on the seam copy too.
            if let Some(seam_u) = seam_vertex_u {
                let on_seam: DetHashSet<u32> = seam_runs
                    .iter()
                    .flat_map(|run| run.indices.iter().map(|&i| boundary_3d[i].1))
                    .collect();
                let at_apex = |p: Point3| match face_data.surface() {
                    FaceSurface::Cone(cone) => (p - cone.apex()).length() < 1e-9,
                    _ => false,
                };
                for i in (0..n_boundary).filter(|&i| off_seam(i)) {
                    if on_seam.contains(&boundary_3d[i].1) && !at_apex(boundary_3d[i].0) {
                        let u = boundary_uv[i].0;
                        boundary_uv[i].0 = ((u - seam_u) / u_period).round().madd(u_period, seam_u);
                    }
                }
            }
            apex_rows.sort_by_key(|row| std::cmp::Reverse(row.0));
            let rim_spacing = if apex_rows.is_empty() {
                1.0
            } else {
                let mut gaps: Vec<f64> = boundary_3d
                    .windows(2)
                    .map(|w| (w[1].0 - w[0].0).length())
                    .filter(|&g| g > 0.0)
                    .collect();
                gaps.sort_by(f64::total_cmp);
                gaps.get(gaps.len() / 2).copied().unwrap_or(1.0)
            };
            for (i, from, to, v, id) in apex_rows {
                let (point, _, edge, forward) = boundary_3d[i];
                let before = (i + boundary_3d.len() - 1) % boundary_3d.len();
                let (v_rim, rim_point) = (boundary_uv[before].1, boundary_3d[before].0);
                // A lone apex sample leaves each side of the seam one ruling
                // long; sampled like the rim, the ruling's triangles stay
                // local instead of fanning round the cone.
                let side: Vec<(f64, Point3, u32)> = {
                    let slant = (rim_point - point).length();
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let n = ((slant / rim_spacing).ceil() as usize).max(2);
                    super::check_count("a cone's seam down to its apex", n, deflection)?;
                    (1..n)
                        .map(|k| {
                            #[allow(clippy::cast_precision_loss)]
                            let f = k as f64 / n as f64;
                            let p = point + (rim_point - point) * f;
                            let gid = *point_to_global
                                .entry(point_merge_key(p, MERGE_GRID))
                                .or_insert_with(|| {
                                    #[allow(clippy::cast_possible_truncation)]
                                    let idx = merged.positions.len() as u32;
                                    merged.positions.push(p);
                                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                                    idx
                                });
                            (v + (v_rim - v) * f, p, gid)
                        })
                        .collect()
                };
                // Down one side to the apex, along the apex row, and up the
                // other side back to the rim.
                let steps = 16;
                let row = (1..=steps).map(|k| {
                    let u = from + (to - from) * f64::from(k) / f64::from(steps);
                    ((u, v), (point, id, edge, forward))
                });
                let up = side
                    .iter()
                    .map(|&(sv, p, gid)| ((to, sv), (p, gid, edge, forward)));
                let (after_uv, after_3d): (Vec<_>, Vec<_>) = row.chain(up).unzip();
                let after = i + 1;
                boundary_uv.splice(after..after, after_uv);
                boundary_3d.splice(after..after, after_3d);
                let (down_uv, down_3d): (Vec<_>, Vec<_>) = side
                    .iter()
                    .rev()
                    .map(|&(sv, p, gid)| ((from, sv), (p, gid, edge, forward)))
                    .unzip();
                boundary_uv.splice(i..i, down_uv);
                boundary_3d.splice(i..i, down_3d);
            }

            // A straight seam on a NURBS face belongs to this face alone and
            // needs no samples between its vertices. Left in where the rulings
            // get no interior rows, each one would fan out across the face to
            // the rims.
            let mut drop: DetHashSet<usize> = DetHashSet::default();
            let nurbs = matches!(face_data.surface(), FaceSurface::Nurbs(_));
            for run in seam_runs.iter().filter(|_| nurbs) {
                let edge = topo.edge(boundary_3d[run.indices[0]].2)?;
                let (a, b) = (
                    topo.vertex(edge.start())?.point(),
                    topo.vertex(edge.end())?.point(),
                );
                let length = (b - a).length();
                if length <= 0.0 {
                    continue;
                }
                let dir = (b - a) * (1.0 / length);
                let off_line = |p: Point3| {
                    let r = p - a;
                    (r - dir * r.dot(dir)).length()
                };
                if run
                    .indices
                    .iter()
                    .all(|&i| off_line(boundary_3d[i].0) <= 1e-9 * length)
                {
                    for &i in &run.indices {
                        let p = boundary_3d[i].0;
                        if (p - a).length() > 1e-9 * length && (p - b).length() > 1e-9 * length {
                            drop.insert(i);
                        }
                    }
                }
            }
            if !drop.is_empty() {
                let keep: Vec<usize> = (0..boundary_uv.len())
                    .filter(|i| !drop.contains(i))
                    .collect();
                boundary_uv = keep.iter().map(|&i| boundary_uv[i]).collect();
                boundary_3d = keep.iter().map(|&i| boundary_3d[i]).collect();
            }
        }

        // Recompute UV bounding box after seam fix.
        uv_bounds(&boundary_uv)
    };

    // Inner wires in the outer loop's (u, v): each hole is unwrapped along
    // itself and shifted by whole periods into the outer loop's span, then
    // constrained and flood-removed below like a planar face's holes.
    let mut holes = inner_wire_uv_loops(
        topo,
        face_id,
        face_data,
        pcurve_uv,
        (u_min, u_max, v_min, v_max),
        deflection,
        angular_tol,
        circle_floor,
        edge_global_indices,
        merged,
        point_to_global,
    )?;
    let joined = join_winding_hole(
        face_data,
        &mut holes,
        &mut boundary_uv,
        &mut boundary_3d,
        deflection,
        angular_tol,
        merged,
        point_to_global,
    )?;
    let (u_min, u_max, v_min, v_max) = if joined {
        uv_bounds(&boundary_uv)
    } else {
        (u_min, u_max, v_min, v_max)
    };
    // A hole that touches a wall's rim without a vertex there (a rod's wall
    // tangent to a drum's end cap) has samples on the rim's line, or a
    // fitted curve's rounding past it, between the rim's own samples. Left
    // there, the wall leaves the sliver between the rim's chord and them
    // outside both its outline and its hole, and no face covers it. They
    // are moved just inside for the triangulation only: each keeps its
    // shared point, so the wall's mesh closes on the rim's chord.
    if matches!(
        face_data.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) && v_max > v_min
    {
        let span = v_max - v_min;
        let (reach, inset) = (1e-6 * span, 1e-9 * span);
        let on_outline: DetHashSet<u32> = boundary_3d.iter().map(|b| b.1).collect();
        for hole in &mut holes {
            for (_, v, gid) in hole.iter_mut() {
                if on_outline.contains(gid) {
                    continue;
                }
                if *v < v_min + inset && *v >= v_min - reach {
                    *v = v_min + inset;
                } else if *v > v_max - inset && *v <= v_max + reach {
                    *v = v_max - inset;
                }
            }
        }
    }
    let n_boundary = boundary_3d.len();
    if cdt_trace() {
        log::debug!("cdt {face_id:?} outer u[{u_min:.4},{u_max:.4}] v[{v_min:.4},{v_max:.4}]");
        for (hi, h) in holes.iter().enumerate() {
            let pts: Vec<String> = h
                .iter()
                .map(|(u, v, _)| format!("({u:.3},{v:.3})"))
                .collect();
            log::debug!("cdt {face_id:?} hole {hi}: {}", pts.join(" "));
        }
    }
    if cdt_trace() {
        for h in &holes {
            for w in h.windows(2) {
                let (a, b) = (
                    merged.positions[w[0].2 as usize],
                    merged.positions[w[1].2 as usize],
                );
                if (a - b).length() < 1e-4 {
                    log::debug!(
                        "cdt {face_id:?} NEAR gid {} -> {} gap {:.3e} uv ({:.6},{:.6})",
                        w[0].2,
                        w[1].2,
                        (a - b).length(),
                        w[1].0,
                        w[1].1
                    );
                }
            }
        }
    }
    let hole_polys: Vec<Vec<(f64, f64)>> = holes
        .iter()
        .map(|h| h.iter().map(|&(u, v, _)| (u, v)).collect())
        .collect();
    // A fillet stripe is a trimmed developable patch: two marched contacts
    // along the rulings and two transverse cross-section arcs, or, where a
    // mitered end consumed one contact, one contact and two arcs of which at
    // least one is the miter's elliptic crease. The class stays narrow because
    // every other curved trim feeds boolean volume and classification paths
    // whose triangulation is qualified separately.
    // A plane's conic section trims a cone the same way: marched sides, rim
    // arcs split wherever earlier cuts put vertices on them, and lines, which
    // on a cone are rulings and stay straight in the developed metric.
    let mut nurbs_boundaries = 0;
    let mut transverse_boundaries = 0;
    let mut ellipse_boundaries = 0;
    for oriented in wire.edges() {
        match topo.edge(oriented.edge())?.curve() {
            EdgeCurve::NurbsCurve(_) => nurbs_boundaries += 1,
            EdgeCurve::Circle(_) => transverse_boundaries += 1,
            EdgeCurve::Ellipse(_) => {
                transverse_boundaries += 1;
                ellipse_boundaries += 1;
            }
            EdgeCurve::Line => {}
        }
    }
    let is_fillet_stripe = match wire.edges().len() {
        4 => nurbs_boundaries == 2 && transverse_boundaries == 2,
        3 => nurbs_boundaries == 1 && transverse_boundaries == 2 && ellipse_boundaries >= 1,
        _ => false,
    } || (matches!(face_data.surface(), FaceSurface::Cone(_))
        && nurbs_boundaries + ellipse_boundaries > 0
        && transverse_boundaries > 0
        && face_data.inner_wires().is_empty());
    let stripe_radius = if circle_floor || !is_fillet_stripe {
        None
    } else {
        developable_stripe_radius(face_data.surface(), v_min, v_max)
    };
    // A holed cylinder or cone wall has no rim-to-rim ladder: Delaunay over
    // its boundary alone fans rim samples to hole samples radians away and
    // chords through the solid. Triangulate it in the developed
    // (radius * u, v) metric and refine by angular extent, like a stripe.
    // A hole straddling the seam is carried by the outer wire instead (the
    // seam cannot cross it): two closed rims, a seam of lines, and the hole's
    // edges (arcs, lines or marched pieces) notching the wire between the
    // seam's two copies. It leaves the same rim-to-hole fans. So does a wall
    // a section loop winds round, whose loop dips between its seam copies:
    // across the dip Delaunay joins the loop's two sides. And so does any
    // patch a section bounds in part: raw `(u, v)` mixes radians with
    // lengths, and Delaunay there joins rim samples to section samples far
    // round the wall, a sheared strip longer than the surface it spans. And
    // so does a wall whose ruling a neighbour's vertex splits: Delaunay fans
    // that vertex to rim samples a quarter turn away, a flat chord through
    // the solid. So does a wall stepped along its rim, whose reflex corner
    // Delaunay fans to the far rim.
    let notched_wall = is_notched_wall(topo, face_data)?
        || is_wound_wall(topo, face_data)?
        || is_section_bound(topo, face_data)?
        || is_split_ruling_wall(topo, face_data)?
        || is_stepped_wall(topo, face_data)?;
    let holed_wall_radius = if holes.is_empty() && !notched_wall {
        None
    } else {
        developable_stripe_radius(face_data.surface(), v_min, v_max)
    };

    let du = u_max - u_min;
    let dv = v_max - v_min;
    // The CDT picks its diagonals by Euclidean distance, and raw cylindrical
    // UV mixes radians with model units: a long stripe reads as arbitrarily
    // narrow, so the triangulation joins its opposite angular ends across the
    // curved interior and sags by the full sector. Scaling u by the radius
    // turns the angular coordinate into a length; scaling the rulings down to
    // one chord cell keeps the CDT from subdividing a direction the surface is
    // already exact along, which lands the stripe on two triangles per angular
    // division. Only the metric moves: boundary identity, the trimmed-domain
    // tests and every surface evaluation stay in the face's parameterization.
    // A cone face running up to its apex has rulings that converge rather
    // than span the face, so its slant (v, a length already) keeps its scale.
    let reaches_apex = matches!(face_data.surface(), FaceSurface::Cone(_))
        && v_min.abs().min(v_max.abs()) < 1e-9 * (v_max - v_min).abs().max(1.0);
    // A NURBS face's iso-line divisions size both its metric and its interior
    // grid: probed once.
    let mut lines = nurbs_grid_lines(
        face_data.surface(),
        (u_min, du),
        (v_min, dv),
        deflection,
        angular_tol,
    )?;
    let divisions = lines.as_ref().map(|(us, vs)| (us.len() - 1, vs.len() - 1));
    if lines.is_none() && du > 1e-15 && dv > 1e-15 {
        let (n_u, n_v) = interior_grid_resolution(
            face_data.surface(),
            du,
            dv,
            deflection,
            angular_tol,
            circle_floor,
        );
        super::check_grid(n_u.saturating_add(1), n_v.saturating_add(1), deflection)?;
    }
    let (cdt_u_scale, cdt_v_scale) = match stripe_radius.or(holed_wall_radius) {
        Some(radius) if reaches_apex => (radius, 1.0),
        Some(radius) if du > 1e-15 && dv > 1e-15 => {
            let divisions =
                segments_for_chord_deviation_a(radius, du, deflection, angular_tol, false);
            let cell = radius * du / divisions as f64;
            (radius, cell / dv)
        }
        _ => extrusion_metric(
            face_data.surface(),
            (u_min, du),
            (v_min, dv),
            deflection,
            angular_tol,
        )
        .or_else(|| {
            let speeds = nurbs_speeds(face_data.surface(), (u_min, du), (v_min, dv));
            ruled_metric(speeds, divisions, du, dv).or(speeds)
        })
        .unwrap_or((1.0, 1.0)),
    };
    let to_cdt = |point: Point2| Point2::new(point.x() * cdt_u_scale, point.y() * cdt_v_scale);
    let from_cdt = |point: Point2| Point2::new(point.x() / cdt_u_scale, point.y() / cdt_v_scale);
    let margin = 0.01;
    let bounds = (
        to_cdt(Point2::new(u_min - margin, v_min - margin)),
        to_cdt(Point2::new(u_max + margin, v_max + margin)),
    );
    let mut cdt = Cdt::with_capacity(bounds, n_boundary);

    let mut cdt_to_global: Vec<Option<u32>> = vec![None; 3]; // 3 super-triangle verts

    let boundary_pts: Vec<Point2> = boundary_uv
        .iter()
        .map(|&(u, v)| to_cdt(Point2::new(u, v)))
        .collect();
    let boundary_cdt_ids = cdt
        .insert_points_hilbert(&boundary_pts)
        .map_err(crate::OperationsError::Math)?;
    if cdt_trace() {
        for (i, &cid) in boundary_cdt_ids.iter().enumerate() {
            log::debug!(
                "cdt {face_id:?} bpt[{i}] gid={} cdtid={cid} uv=({:.5},{:.5})",
                boundary_3d[i].1,
                boundary_uv[i].0,
                boundary_uv[i].1
            );
        }
    }
    let max_cdt_idx = boundary_cdt_ids.iter().copied().max().unwrap_or(2);
    if cdt_to_global.len() <= max_cdt_idx {
        cdt_to_global.resize(max_cdt_idx + 1, None);
    }
    for (i, &cdt_idx) in boundary_cdt_ids.iter().enumerate() {
        cdt_to_global[cdt_idx] = Some(boundary_3d[i].1);
    }

    for i in 0..n_boundary {
        let v0 = boundary_cdt_ids[i];
        let v1 = boundary_cdt_ids[(i + 1) % n_boundary];
        cdt.insert_constraint(v0, v1)
            .map_err(crate::OperationsError::Math)?;
    }
    let constraints_added_steiner_vertices = cdt.vertices().len() > cdt_to_global.len();

    let mut hole_pairs: Vec<(usize, usize)> = Vec::new();
    let mut hole_seed_pts: Vec<Point2> = Vec::new();
    let mut hole_ranges: Vec<(usize, usize)> = Vec::new();
    for hole in &holes {
        let pts: Vec<Point2> = hole
            .iter()
            .map(|&(u, v, _)| to_cdt(Point2::new(u, v)))
            .collect();
        let ids = cdt
            .insert_points_hilbert(&pts)
            .map_err(crate::OperationsError::Math)?;
        let max_id = ids.iter().copied().max().unwrap_or(0);
        if cdt_to_global.len() <= max_id {
            cdt_to_global.resize(max_id + 1, None);
        }
        for (&cid, &(_, _, gid)) in ids.iter().zip(hole) {
            cdt_to_global[cid] = Some(gid);
        }
        for k in 0..ids.len() {
            let (a, b) = (ids[k], ids[(k + 1) % ids.len()]);
            if a != b {
                cdt.insert_constraint(a, b)
                    .map_err(crate::OperationsError::Math)?;
                hole_pairs.push((a, b));
            }
        }
        // The seed steps a fraction of its vertex's shorter edge inside the
        // loop; a near-duplicate sample (a loop closing on its own first
        // point) would shrink that step below what the flood can resolve.
        let (lo, hi) = pts.iter().fold(
            (
                Point2::new(f64::MAX, f64::MAX),
                Point2::new(f64::MIN, f64::MIN),
            ),
            |(lo, hi), p| {
                (
                    Point2::new(lo.x().min(p.x()), lo.y().min(p.y())),
                    Point2::new(hi.x().max(p.x()), hi.y().max(p.y())),
                )
            },
        );
        let merge = 1e-6 * (hi.x() - lo.x()).hypot(hi.y() - lo.y());
        let mut distinct: Vec<Point2> = Vec::with_capacity(pts.len());
        for p in pts {
            if distinct
                .last()
                .is_none_or(|q| (p.x() - q.x()).hypot(p.y() - q.y()) > merge)
            {
                distinct.push(p);
            }
        }
        while distinct.len() > 1
            && distinct
                .first()
                .zip(distinct.last())
                .is_some_and(|(a, b)| (a.x() - b.x()).hypot(a.y() - b.y()) <= merge)
        {
            distinct.pop();
        }
        let start = hole_seed_pts.len();
        hole_seed_pts.extend(distinct);
        hole_ranges.push((start, hole_seed_pts.len()));
    }

    if du > 1e-15 && dv > 1e-15 {
        // A NURBS face's grid lines follow its surface; the others are even.
        let (us, vs) = if let Some((mut us, mut vs)) = lines.take() {
            let surface = face_data.surface();
            untwist_grid_lines(surface, &mut us, &mut vs, deflection)?;
            (us, vs)
        } else {
            let (n_u, n_v) = interior_grid_resolution(
                face_data.surface(),
                du,
                dv,
                deflection,
                angular_tol,
                circle_floor,
            );
            let even = |lo: f64, d: f64, n: usize| -> Vec<f64> {
                (0..=n).map(|k| lo + d * (k as f64 / n as f64)).collect()
            };
            (even(u_min, du, n_u), even(v_min, dv, n_v))
        };

        let boundary_uv_ref = &boundary_uv;
        let hole_polys = &hole_polys;
        let inner = |lines: &[f64]| lines[1..lines.len().max(2) - 1].to_vec();
        let (us, vs) = (inner(&us), inner(&vs));
        // A grid point on a constrained edge (the outline or a hole, each
        // sampled once for the whole solid) would split that edge at a
        // vertex the face across it does not have, and the mesh would open
        // there: such points are left out.
        let on_constraint = {
            let vertices = cdt.vertices();
            let segments: Vec<(Point2, Point2)> = cdt
                .constraint_edges()
                .iter()
                .map(|&(a, b)| (vertices[a], vertices[b]))
                .collect();
            let xs: Vec<f64> = us
                .iter()
                .map(|&u| to_cdt(Point2::new(u, 0.0)).x())
                .collect();
            let ys: Vec<f64> = vs
                .iter()
                .map(|&v| to_cdt(Point2::new(0.0, v)).y())
                .collect();
            grid_points_on_segments(&xs, &ys, &segments)
        };
        let on_constraint = &on_constraint;
        let vs = &vs;
        // Each row's crossings of the boundary and the holes, found once.
        let rows: Vec<(RowCrossings, Vec<RowCrossings>)> = vs
            .iter()
            .map(|&v| {
                (
                    row_crossings(boundary_uv_ref, v),
                    hole_polys.iter().map(|h| row_crossings(h, v)).collect(),
                )
            })
            .collect();
        let rows = &rows;
        let interior_pts: Vec<Point2> = us
            .iter()
            .enumerate()
            .flat_map(|(i, &u)| {
                vs.iter()
                    .zip(rows)
                    .enumerate()
                    .filter_map(move |(j, (&v, (outer, holes)))| {
                        if on_constraint.contains(&(i, j)) {
                            return None;
                        }
                        let parameter = Point2::new(u, v);
                        // Nested inner wires alternate void and island, so a
                        // point is void at odd depth.
                        let in_hole = hole_polys
                            .iter()
                            .zip(holes)
                            .filter(|(h, row)| point_in_polygon_row(h, row, parameter))
                            .count()
                            % 2
                            == 1;
                        (point_in_polygon_row(boundary_uv_ref, outer, parameter) && !in_hole)
                            .then_some(to_cdt(parameter))
                    })
            })
            .collect();
        if !interior_pts.is_empty() {
            let interior_cdt_ids = cdt
                .insert_points_hilbert(&interior_pts)
                .map_err(crate::OperationsError::Math)?;
            let max_interior = interior_cdt_ids.iter().copied().max().unwrap_or(0);
            if cdt_to_global.len() <= max_interior {
                cdt_to_global.resize(max_interior + 1, None);
            }
        }
    }

    // The metric leaves the diagonals well conditioned but does not by itself
    // certify the requested tolerances, so every retained triangle is measured
    // and any that overshoots has its widest angular edge split at the middle,
    // halving the overshoot. Halving reaches the required extent in
    // `log2(span / extent)` passes; the pass cap only bounds a boundary
    // pathological enough to defeat that, and the caller's contract for the
    // resulting error is in this function's doc comment. A crossing constraint
    // mints an untracked Steiner vertex, which means the parameter boundary
    // self-intersects: there is no unambiguous trimmed interior left to
    // measure, so that class keeps the pre-existing recovery.
    if let Some(radius) = stripe_radius
        .filter(|_| !constraints_added_steiner_vertices)
        .or(holed_wall_radius)
    {
        // Halving reaches any tolerance from a full turn in a dozen passes,
        // but every split leaves its Delaunay neighbours to fit again, and a
        // front of splits through a wide band can need several more.
        const MAX_HALVING_PASSES: usize = 32;
        // A cone's rims are sampled at their own radius, so a triangle's
        // sag is bounded by its widest corner (the radius is linear in v),
        // not the face's widest end.
        let radius_at = |v: f64| {
            if let FaceSurface::Cone(cone) = face_data.surface() {
                cone.radius_at(v).abs().min(radius)
            } else {
                radius
            }
        };
        let mut converged = false;
        for _ in 0..MAX_HALVING_PASSES {
            let vertices = cdt.vertices();
            // A constrained edge is shared boundary sampled once for the whole
            // solid, and a triangle touching it sags at least as far as it
            // however that triangle is split: each vertex allows the sag of
            // its coarsest constrained edge. Measured as sag, an edge through
            // a cone's apex spans any `u` at no radius.
            let mut allowed: DetHashMap<usize, f64> = DetHashMap::default();
            for &(a, b) in cdt.constraint_edges() {
                let (pa, pb) = (from_cdt(vertices[a]), from_cdt(vertices[b]));
                let sag =
                    radius_at(pa.y().max(pb.y())) * (1.0 - (0.5 * (pb.x() - pa.x()).abs()).cos());
                for k in [a, b] {
                    let entry = allowed.entry(k).or_insert(0.0);
                    *entry = entry.max(sag);
                }
            }
            let mut splits = Vec::new();
            for (i0, i1, i2) in cdt.triangles() {
                if i0 < 3 || i1 < 3 || i2 < 3 {
                    continue;
                }
                let ids = [i0, i1, i2];
                let corners = ids.map(|id| from_cdt(vertices[id]));
                let (mut lo, mut hi) = (0, 0);
                for corner in 1..3 {
                    if corners[corner].x() < corners[lo].x() {
                        lo = corner;
                    }
                    if corners[corner].x() > corners[hi].x() {
                        hi = corner;
                    }
                }
                let local_radius = corners.iter().map(|c| radius_at(c.y())).fold(0.0, f64::max);
                let span = corners[hi].x() - corners[lo].x();
                if stripe_span_within_tolerance(local_radius, span, deflection, angular_tol)
                    || local_radius * (1.0 - (0.5 * span).cos())
                        <= ids
                            .iter()
                            .map(|k| allowed.get(k).copied().unwrap_or(0.0))
                            .fold(0.0, f64::max)
                {
                    continue;
                }
                // A constrained edge is a shared boundary sampled once for the
                // whole solid; splitting it would leave the neighbouring face
                // welded to a vertex it does not have, so such an extent is as
                // fine as this face can weld to.
                if cdt
                    .constraint_edges()
                    .contains(&(ids[lo].min(ids[hi]), ids[lo].max(ids[hi])))
                {
                    continue;
                }
                let centroid = Point2::new(
                    (corners[0].x() + corners[1].x() + corners[2].x()) / 3.0,
                    (corners[0].y() + corners[1].y() + corners[2].y()) / 3.0,
                );
                // Triangles outside the trimmed boundary or inside a hole are
                // dropped below, so their sag never ships.
                let in_hole = |p: Point2| {
                    hole_polys
                        .iter()
                        .filter(|h| point_in_polygon_2d(h, p))
                        .count()
                        % 2
                        == 1
                };
                if !point_in_polygon_2d(&boundary_uv, centroid) || in_hole(centroid) {
                    continue;
                }
                let split = Point2::new(
                    0.5 * (corners[lo].x() + corners[hi].x()),
                    0.5 * (corners[lo].y() + corners[hi].y()),
                );
                if point_in_polygon_2d(&boundary_uv, split) && !in_hole(split) {
                    splits.push(to_cdt(split));
                }
            }
            if cdt_trace() {
                log::debug!(
                    "cdt {face_id:?} refine pass: {} splits, {} vertices",
                    splits.len(),
                    cdt.vertices().len()
                );
            }
            if splits.is_empty() {
                converged = true;
                break;
            }
            let before = cdt.vertices().len();
            cdt.insert_points_hilbert(&splits)
                .map_err(crate::OperationsError::Math)?;
            if cdt.vertices().len() == before {
                break;
            }
        }
        if !converged {
            log::warn!(
                "{face_id:?}: developable CDT stayed outside deflection {deflection} after {MAX_HALVING_PASSES} passes; the face falls back to the snap mesher"
            );
            return Err(crate::OperationsError::InvalidInput {
                reason: "developable CDT did not reach the requested deflection".to_string(),
            });
        }
    }
    split_edge_ears(
        topo,
        face_data,
        &anchored,
        edge_global_indices,
        &mut cdt,
        &cdt_to_global,
        &boundary_uv,
        &hole_polys,
        (cdt_u_scale, cdt_v_scale),
    )?;
    let boundary_pairs: Vec<(usize, usize)> = (0..n_boundary)
        .map(|i| (boundary_cdt_ids[i], boundary_cdt_ids[(i + 1) % n_boundary]))
        .collect();
    cdt.remove_exterior(&boundary_pairs);
    if cdt_trace() {
        log::debug!(
            "cdt {face_id:?} after remove_exterior: {} tris",
            cdt.triangles().len()
        );
    }
    if !hole_pairs.is_empty() {
        // Recovery can split a constraint at a Steiner point or at a vertex
        // it passes through, so the barrier is the CDT's own constrained
        // edges rather than the pairs as inserted.
        let barrier: DetHashSet<(usize, usize)> = cdt
            .constraint_edges()
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        for seed in super::planar::hole_removal_seeds(&hole_seed_pts, &hole_ranges) {
            cdt.flood_remove_from_point(seed, &barrier);
            if cdt_trace() {
                log::debug!(
                    "cdt {face_id:?} flood from {:?}: {} tris left",
                    from_cdt(seed),
                    cdt.triangles().len()
                );
            }
        }
    }
    // A hole that reaches the outline at one point can come as a lobe of
    // the outer loop, closed at that vertex. The loop's own constraints wall
    // it in, so the exterior flood leaves it.
    let (lobe_pts, lobe_ranges) = pinched_hole_lobes(&boundary_cdt_ids, &boundary_pts);
    if !lobe_ranges.is_empty() {
        let barrier: DetHashSet<(usize, usize)> = cdt
            .constraint_edges()
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();
        for seed in super::planar::hole_removal_seeds(&lobe_pts, &lobe_ranges) {
            cdt.flood_remove_from_point(seed, &barrier);
            if cdt_trace() {
                log::debug!(
                    "cdt {face_id:?} lobe flood from {:?}: {} tris left",
                    from_cdt(seed),
                    cdt.triangles().len()
                );
            }
        }
    }

    let cdt_verts = cdt.vertices();
    let triangles = cdt.triangles();

    // Constraint recovery can mint Steiner vertices (crossing splits,
    // bisection backstop) whose ids the insert calls above never returned;
    // cover them so the lift below assigns them global ids.
    if cdt_to_global.len() < cdt_verts.len() {
        cdt_to_global.resize(cdt_verts.len(), None);
    }

    let mut final_global_ids: Vec<u32> = vec![0; cdt_to_global.len()];

    for i in 0..cdt_to_global.len() {
        if let Some(gid) = cdt_to_global[i] {
            final_global_ids[i] = gid;
        } else if i >= 3 {
            let pt2 = from_cdt(cdt_verts[i]);
            let surface = face_data.surface();
            let pt3 = eval_surface_point(surface, pt2.x(), pt2.y());
            let (u, v) = wrap_to_domain(surface, pt2.x(), pt2.y());
            let nrm = surface.normal(u, v);

            let key = point_merge_key(pt3, MERGE_GRID);
            let gid = *point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pt3);
                merged.normals.push(nrm);
                idx
            });
            final_global_ids[i] = gid;
        }
    }
    // The CDT's UV winding is internally consistent but can be inverted as
    // a whole against the surface: a pinched parameterization (a horn torus
    // corner patch, where the base arc's UV image degenerates) triangulates
    // cleanly yet winds against the outward normal. Decide ONE flip for the
    // whole face by an area-weighted vote of geometric-vs-surface normal
    // agreement, keeping internal consistency (per-triangle flips near the
    // pinch scatter, where the sampled normal is unreliable). Default
    // (non-reversed) orientation is emitted; the caller applies the
    // `is_reversed` flip afterward.
    let mut vote = 0.0;
    for &(i0, i1, i2) in &triangles {
        if i0 < 3 || i1 < 3 || i2 < 3 {
            continue;
        }
        let (p0, p1, p2) = (
            merged.positions[final_global_ids[i0] as usize],
            merged.positions[final_global_ids[i1] as usize],
            merged.positions[final_global_ids[i2] as usize],
        );
        let geo = (p1 - p0).cross(p2 - p0);
        let (uv0, uv1, uv2) = (
            from_cdt(cdt_verts[i0]),
            from_cdt(cdt_verts[i1]),
            from_cdt(cdt_verts[i2]),
        );
        let uc = (uv0.x() + uv1.x() + uv2.x()) / 3.0;
        let vc = (uv0.y() + uv1.y() + uv2.y()) / 3.0;
        let (uc, vc) = wrap_to_domain(face_data.surface(), uc, vc);
        let outward = face_data.surface().normal(uc, vc);
        vote += geo.dot(outward);
    }
    let flip_all = vote < 0.0;
    for (i0, i1, i2) in triangles {
        if i0 < 3 || i1 < 3 || i2 < 3 {
            continue; // Skip super-triangle vertices
        }
        let (g0, g1, g2) = (
            final_global_ids[i0],
            final_global_ids[i1],
            final_global_ids[i2],
        );
        if g0 == g1 || g1 == g2 || g0 == g2 {
            continue; // collapsed onto a pole row
        }
        if flip_all {
            merged.indices.push(final_global_ids[i0]);
            merged.indices.push(final_global_ids[i2]);
            merged.indices.push(final_global_ids[i1]);
        } else {
            merged.indices.push(final_global_ids[i0]);
            merged.indices.push(final_global_ids[i1]);
            merged.indices.push(final_global_ids[i2]);
        }
    }

    Ok(())
}

/// Gives each triangle whose three corners are samples of one edge a point
/// at its middle, on the surface.
///
/// Such a triangle spans the edge's chord, not the surface: its corners lie
/// on one curve, so it lies where the curve does. The face across the edge
/// can span the same chord (a plane face does, along a curve it holds), and
/// the chord then bounds four triangles. The point joins each corner to the
/// surface between them. Each pass splits the triangles left, until none is.
#[allow(clippy::too_many_arguments)]
fn split_edge_ears(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    anchored: &DetHashMap<usize, Vec<u32>>,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    cdt: &mut brepkit_math::cdt::Cdt,
    cdt_to_global: &[Option<u32>],
    boundary_uv: &[(f64, f64)],
    hole_polys: &[Vec<(f64, f64)>],
    (u_scale, v_scale): (f64, f64),
) -> Result<(), crate::OperationsError> {
    use brepkit_math::vec::Point2;
    const MAX_PASSES: usize = 8;
    // The edges, by index, whose samples hold each shared point.
    let mut edges_of: DetHashMap<u32, Vec<usize>> = DetHashMap::default();
    for &wire_id in std::iter::once(&face_data.outer_wire()).chain(face_data.inner_wires()) {
        for oe in topo.wire(wire_id)?.edges() {
            let idx = oe.edge().index();
            let Some(gids) = anchored.get(&idx).or_else(|| edge_global_indices.get(&idx)) else {
                continue;
            };
            for &gid in gids {
                let edges = edges_of.entry(gid).or_default();
                if !edges.contains(&idx) {
                    edges.push(idx);
                }
            }
        }
    }
    if edges_of.is_empty() {
        return Ok(());
    }
    let in_face = |p: (f64, f64)| {
        let p = Point2::new(p.0, p.1);
        point_in_polygon_2d(boundary_uv, p)
            && hole_polys
                .iter()
                .filter(|h| point_in_polygon_2d(h, p))
                .count()
                % 2
                == 0
    };
    for _ in 0..MAX_PASSES {
        let vertices = cdt.vertices();
        let mut splits: Vec<Point2> = Vec::new();
        for (i0, i1, i2) in cdt.triangles() {
            if i0 < 3 || i1 < 3 || i2 < 3 {
                continue;
            }
            let gid = |i: usize| cdt_to_global.get(i).copied().flatten();
            let (Some(a), Some(b), Some(c)) = (gid(i0), gid(i1), gid(i2)) else {
                continue;
            };
            if a == b || b == c || a == c {
                continue;
            }
            let (Some(ea), Some(eb), Some(ec)) =
                (edges_of.get(&a), edges_of.get(&b), edges_of.get(&c))
            else {
                continue;
            };
            if !ea.iter().any(|e| eb.contains(e) && ec.contains(e)) {
                continue;
            }
            let [p0, p1, p2] = [i0, i1, i2].map(|i| vertices[i]);
            // A triangle with no area in (u, v) has its middle on its
            // longest side, which can be a constrained edge.
            let area =
                (p1.x() - p0.x()).madd(p2.y() - p0.y(), -((p1.y() - p0.y()) * (p2.x() - p0.x())));
            let longest = [(p0, p1), (p1, p2), (p2, p0)]
                .iter()
                .map(|(p, q)| (q.x() - p.x()).hypot(q.y() - p.y()))
                .fold(0.0, f64::max);
            if area.abs() <= 1e-9 * longest * longest {
                continue;
            }
            let middle = Point2::new(
                (p0.x() + p1.x() + p2.x()) / 3.0,
                (p0.y() + p1.y() + p2.y()) / 3.0,
            );
            if in_face((middle.x() / u_scale, middle.y() / v_scale)) {
                splits.push(middle);
            }
        }
        if splits.is_empty() {
            break;
        }
        let before = cdt.vertices().len();
        cdt.insert_points_hilbert(&splits)
            .map_err(crate::OperationsError::Math)?;
        if cdt.vertices().len() == before {
            break;
        }
    }
    Ok(())
}

/// Project a 3D point onto a face surface, returning (u, v) parameters.
fn project_to_surface_uv(
    surface: &FaceSurface,
    pt: Point3,
) -> Result<(f64, f64), crate::OperationsError> {
    match surface {
        FaceSurface::Cylinder(cyl) => Ok(cyl.project_point(pt)),
        FaceSurface::Cone(cone) => Ok(cone.project_point(pt)),
        FaceSurface::Sphere(sphere) => Ok(sphere.project_point(pt)),
        FaceSurface::Torus(torus) => Ok(torus.project_point(pt)),
        FaceSurface::Nurbs(surface) => super::nurbs::project_boundary_point(surface, pt)
            .map(|proj| (proj.u, proj.v))
            .map_err(crate::OperationsError::Math),
        FaceSurface::Plane { .. } => Err(crate::OperationsError::InvalidInput {
            reason: "planar faces should not use CDT tessellation".to_string(),
        }),
    }
}

/// Try to find (u,v) coordinates for a 3D point using a PCurve.
fn project_via_pcurve(
    pcurve: &brepkit_topology::pcurve::PCurve,
    pt: Point3,
    surface: &FaceSurface,
) -> Option<(f64, f64)> {
    let t_start = pcurve.t_start();
    let t_end = pcurve.t_end();
    let n_samples = 16;

    let mut best_t = t_start;
    let mut best_dist = f64::MAX;

    for i in 0..=n_samples {
        let t = t_start + (t_end - t_start) * (i as f64) / (n_samples as f64);
        let uv = pcurve.evaluate(t);
        let p_surf = eval_surface_point(surface, uv.x(), uv.y());
        let d = (p_surf - pt).length();
        if d < best_dist {
            best_dist = d;
            best_t = t;
        }
    }

    // Refine with bisection around best_t.
    let dt = (t_end - t_start) / (n_samples as f64);
    let mut lo = (best_t - dt).max(t_start);
    let mut hi = (best_t + dt).min(t_end);
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        let uv_lo = pcurve.evaluate(lo);
        let uv_hi = pcurve.evaluate(hi);
        let d_lo = (eval_surface_point(surface, uv_lo.x(), uv_lo.y()) - pt).length();
        let d_hi = (eval_surface_point(surface, uv_hi.x(), uv_hi.y()) - pt).length();
        if d_lo < d_hi {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let t_final = 0.5 * (lo + hi);
    let uv = pcurve.evaluate(t_final);
    let p_final = eval_surface_point(surface, uv.x(), uv.y());

    if (p_final - pt).length() < brepkit_math::tolerance::Tolerance::default().linear {
        Some((uv.x(), uv.y()))
    } else {
        None
    }
}

/// The u of a straight seam on a periodic face, from one of its samples off
/// a cone's apex, with the face's u period.
fn seam_span(
    surface: &FaceSurface,
    boundary_3d: &[(Point3, u32, brepkit_topology::edge::EdgeId, bool)],
    boundary_uv: &[(f64, f64)],
    seam_edges: &DetHashSet<usize>,
) -> Option<(f64, f64)> {
    let (_, period) = surface_periods(surface).0?;
    let apex = match surface {
        FaceSurface::Cone(cone) => Some(cone.apex()),
        _ => None,
    };
    boundary_3d
        .iter()
        .zip(boundary_uv)
        .find(|((p, _, edge, _), _)| {
            seam_edges.contains(&edge.index()) && apex.is_none_or(|a| (*p - a).length() > 1e-9)
        })
        .map(|(_, &(u, _))| (u, period))
}

/// How many seam edges a pointed cone's wall runs each way: the edges its
/// wire runs both ways, when the wire reaches the apex (zero otherwise). A
/// wall whose seam a hole notches runs more than one.
fn pointed_cone_seams(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    wire: &brepkit_topology::wire::Wire,
) -> Result<usize, crate::OperationsError> {
    let FaceSurface::Cone(cone) = face_data.surface() else {
        return Ok(0);
    };
    let edges = wire.edges();
    let twice = edges
        .iter()
        .enumerate()
        .filter(|&(i, oe)| {
            edges[..i].iter().all(|other| other.edge() != oe.edge())
                && edges[i + 1..].iter().any(|other| other.edge() == oe.edge())
        })
        .count();
    if twice == 0 {
        return Ok(0);
    }
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        for v in [edge.start(), edge.end()] {
            if (topo.vertex(v)?.point() - cone.apex()).length() < 1e-9 {
                return Ok(twice);
            }
        }
    }
    Ok(0)
}

/// The u of a cylinder's, cone's or u-closed NURBS face's seam, read off a
/// vertex of an edge its wire runs both ways, away from a cone's apex or a
/// NURBS pole, where u means nothing.
fn seam_vertex_u(
    topo: &Topology,
    edges: &[brepkit_topology::wire::OrientedEdge],
    surface: &FaceSurface,
) -> Result<Option<f64>, crate::OperationsError> {
    let apex = match surface {
        FaceSurface::Cone(cone) => Some(cone.apex()),
        FaceSurface::Cylinder(_) => None,
        FaceSurface::Nurbs(_) if surface_periods(surface).0.is_some() => None,
        _ => return Ok(None),
    };
    let poles = SurfacePoles::of(surface);
    for (i, oe) in edges.iter().enumerate() {
        if edges[i + 1..].iter().all(|other| other.edge() != oe.edge()) {
            continue;
        }
        let edge = topo.edge(oe.edge())?;
        for v in [edge.start(), edge.end()] {
            let p = topo.vertex(v)?.point();
            if apex.is_some_and(|a| (p - a).length() < 1e-9) || poles.v_of(p).is_some() {
                continue;
            }
            if let Ok((u, _)) = project_to_surface_uv(surface, p) {
                return Ok(Some(u));
            }
        }
    }
    Ok(None)
}

/// Re-sequence a wire's closed edges so each cycle starts and ends at the
/// sample nearest its vertex, and end the wire's seam edges on those samples.
///
/// A closed edge's pool samples run from its curve's own parametric origin,
/// which need not be its vertex: an ellipse's origin is always a major-axis
/// end, and a converted or transformed conic keeps whatever origin its new
/// frame gives it. Walked as-is, the rim leaves the boundary loop at the
/// origin while the seam joins it at the vertex, and the loop crosses
/// itself. A seam edge appears twice in this face's wire and in no other
/// face, so moving its ends along the rim by less than one sample spacing
/// opens no crack against a neighbour. Returns only the edges it changed.
fn anchor_closed_edges_at_vertices(
    topo: &Topology,
    wire: &brepkit_topology::wire::Wire,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &TriangleMesh,
) -> Result<DetHashMap<usize, Vec<u32>>, crate::OperationsError> {
    let mut uses: DetHashMap<usize, usize> = DetHashMap::default();
    for oe in wire.edges() {
        *uses.entry(oe.edge().index()).or_default() += 1;
    }
    let mut anchors: DetHashMap<brepkit_topology::vertex::VertexId, u32> = DetHashMap::default();
    let mut changed: DetHashMap<usize, Vec<u32>> = DetHashMap::default();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        if !edge.is_closed() || changed.contains_key(&oe.edge().index()) {
            continue;
        }
        let Some(gids) = edge_global_indices.get(&oe.edge().index()) else {
            continue;
        };
        let cycle = match gids.split_last() {
            Some((last, rest)) if !rest.is_empty() && rest.first() == Some(last) => rest,
            _ => gids.as_slice(),
        };
        let vertex = topo.vertex(edge.start())?.point();
        let Some(k) = (0..cycle.len()).min_by(|&a, &b| {
            let da = (merged.positions[cycle[a] as usize] - vertex).length();
            let db = (merged.positions[cycle[b] as usize] - vertex).length();
            da.total_cmp(&db)
        }) else {
            continue;
        };
        anchors.insert(edge.start(), cycle[k]);
        if k != 0 {
            let mut rotated = cycle.to_vec();
            rotated.rotate_left(k);
            rotated.push(cycle[k]);
            changed.insert(oe.edge().index(), rotated);
        }
    }
    if anchors.is_empty() {
        return Ok(changed);
    }
    for oe in wire.edges() {
        let idx = oe.edge().index();
        let edge = topo.edge(oe.edge())?;
        if edge.is_closed() || uses.get(&idx).copied() != Some(2) || changed.contains_key(&idx) {
            continue;
        }
        let Some(gids) = edge_global_indices.get(&idx) else {
            continue;
        };
        let mut snapped = gids.clone();
        if let (Some(&a), Some(first)) = (anchors.get(&edge.start()), snapped.first_mut()) {
            *first = a;
        }
        if let (Some(&a), Some(last)) = (anchors.get(&edge.end()), snapped.last_mut()) {
            *last = a;
        }
        if &snapped != gids {
            changed.insert(idx, snapped);
        }
    }
    Ok(changed)
}

/// Mesh one curved face with holes on its own, the way the solid mesher does:
/// every edge of the face is sampled into a local pool first, so closed rims
/// are anchored at their vertices exactly as they are against the solid's
/// shared pool. A cylinder or cone wall goes through the constrained CDT. A
/// sphere face goes through the latitude-band mesher when it is a band
/// between two latitude rims, and through the constrained CDT otherwise. A
/// torus face tries the solid mesher's structured bands (notch, two-rim,
/// latitude) and then its constrained CDT. When no mesher takes the face, it
/// comes back without triangles.
pub(super) fn tessellate_holed_face_local(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> Result<super::TriangleMeshUV, crate::OperationsError> {
    let mut merged = TriangleMesh::default();
    let mut point_to_global = PointPool::default();
    let mut pool: DetHashMap<usize, Vec<u32>> = DetHashMap::default();
    // A twisted face's straight edges are sampled as finely as its grid, as
    // the solid mesher samples them.
    let mut straight: DetHashMap<usize, Vec<Point3>> =
        super::edge_sampling::twisted_straight_edge_samples(
            topo,
            face_data,
            deflection,
            angular_tol,
            circle_floor,
        )?
        .into_iter()
        .collect();
    let mut edge_points: DetHashMap<usize, Vec<Point3>> = DetHashMap::default();
    let mut params: DetHashMap<usize, Vec<f64>> = DetHashMap::default();
    let mut order: Vec<usize> = Vec::new();
    for wire_id in
        std::iter::once(face_data.outer_wire()).chain(face_data.inner_wires().iter().copied())
    {
        for oe in topo.wire(wire_id)?.edges() {
            if edge_points.contains_key(&oe.edge().index()) {
                continue;
            }
            let edge = topo.edge(oe.edge())?;
            let points = if let Some(points) = straight.remove(&oe.edge().index()) {
                points
            } else {
                let (points, ts) = super::edge_sampling::sample_edge_with_params(
                    topo,
                    edge,
                    deflection,
                    angular_tol,
                    circle_floor,
                )?;
                if let Some(ts) = ts {
                    params.insert(oe.edge().index(), ts);
                }
                points
            };
            edge_points.insert(oe.edge().index(), points);
            order.push(oe.edge().index());
        }
    }
    // The face's outline is read in its (u, v) as the solid mesher reads it.
    super::solid::sample_seam_vertices_on_rims(topo, &[face_id], &mut edge_points)?;
    super::solid::sample_holes_past_sphere_chords(topo, &[face_id], &mut edge_points)?;
    super::solid::split_crossing_spline_chords(
        topo,
        &[face_id],
        &mut edge_points,
        params,
        deflection,
    )?;
    for idx in order {
        let gids = edge_points
            .remove(&idx)
            .unwrap_or_default()
            .into_iter()
            .map(|pt| {
                *point_to_global
                    .entry(point_merge_key(pt, MERGE_GRID))
                    .or_insert_with(|| {
                        #[allow(clippy::cast_possible_truncation)]
                        let idx = merged.positions.len() as u32;
                        merged.positions.push(pt);
                        merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                        idx
                    })
            })
            .collect();
        pool.insert(idx, gids);
    }
    let meshed = match face_data.surface() {
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Nurbs(_) => {
            tessellate_nonplanar_cdt(
                topo,
                face_id,
                face_data,
                deflection,
                angular_tol,
                circle_floor,
                &pool,
                &mut merged,
                &mut point_to_global,
            )?;
            true
        }
        FaceSurface::Sphere(_) => {
            tessellate_latitude_band_shared(
                topo,
                face_data,
                deflection,
                angular_tol,
                &pool,
                &mut merged,
                &mut point_to_global,
            )? || {
                // Any other sphere face takes the constrained CDT, as it
                // does in the solid mesher.
                tessellate_nonplanar_cdt(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    &pool,
                    &mut merged,
                    &mut point_to_global,
                )?;
                true
            }
        }
        FaceSurface::Torus(_) => {
            let banded = tessellate_torus_notch_band(
                topo,
                face_data,
                deflection,
                angular_tol,
                &pool,
                &mut merged,
                &mut point_to_global,
            )? || tessellate_torus_two_rim_band(
                topo,
                face_data,
                deflection,
                angular_tol,
                &pool,
                &mut merged,
                &mut point_to_global,
            )? || tessellate_latitude_band_shared(
                topo,
                face_data,
                deflection,
                angular_tol,
                &pool,
                &mut merged,
                &mut point_to_global,
            )?;
            // The snap mesher re-enters the per-face mesher, so a face the
            // CDT cannot take comes back empty for the caller's grid.
            banded || {
                let before = merged.indices.len();
                let cdt = super::pass_on_sample_limit(tessellate_nonplanar_cdt(
                    topo,
                    face_id,
                    face_data,
                    deflection,
                    angular_tol,
                    circle_floor,
                    &pool,
                    &mut merged,
                    &mut point_to_global,
                ))?;
                cdt.is_ok() && merged.indices.len() > before
            }
        }
        FaceSurface::Plane { .. } => false,
    };
    if !meshed {
        merged.indices.clear();
    }
    let surface = face_data.surface();
    let mut uvs = Vec::with_capacity(merged.positions.len());
    for (i, &p) in merged.positions.iter().enumerate() {
        let (u, v) = project_to_surface_uv(surface, p)?;
        merged.normals[i] = surface.normal(u, v);
        uvs.push([u, v]);
    }
    let (u_period, v_period) = surface_periods(surface);
    if let Some((_, period)) = u_period {
        split_uv_seam(&mut merged, &mut uvs, 0, period);
    }
    if let Some((_, period)) = v_period {
        split_uv_seam(&mut merged, &mut uvs, 1, period);
    }
    Ok(super::TriangleMeshUV { mesh: merged, uvs })
}

/// Give every triangle a continuous parameter `axis` (0 for u, 1 for v). A
/// vertex welded on the seam carries one principal value, so a triangle
/// beside it can span nearly a period; such a triangle takes copies of its
/// low-side vertices one period on.
fn split_uv_seam(mesh: &mut TriangleMesh, uvs: &mut Vec<[f64; 2]>, axis: usize, period: f64) {
    let mut copies: DetHashMap<u32, u32> = DetHashMap::default();
    for t in 0..mesh.indices.len() / 3 {
        let tri = [0, 1, 2].map(|k| mesh.indices[3 * t + k]);
        let us = tri.map(|i| uvs[i as usize][axis]);
        let high = us.iter().copied().fold(f64::MIN, f64::max);
        if high - us.iter().copied().fold(f64::MAX, f64::min) <= period / 2.0 {
            continue;
        }
        for (k, &i) in tri.iter().enumerate() {
            if high - us[k] <= period / 2.0 {
                continue;
            }
            let copy = *copies.entry(i).or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let j = mesh.positions.len() as u32;
                let mut uv = uvs[i as usize];
                uv[axis] += period;
                mesh.positions.push(mesh.positions[i as usize]);
                mesh.normals.push(mesh.normals[i as usize]);
                uvs.push(uv);
                j
            });
            mesh.indices[3 * t + k] = copy;
        }
    }
}

/// A closed `(u, v, global id)` sample loop.
type HoleLoop = Vec<(f64, f64, u32)>;

/// Each inner wire of a curved face as a closed `(u, v, global id)` loop in
/// the frame of the face's unwrapped outer boundary `(u_min, u_max, v_min,
/// v_max)`.
///
/// Samples come from the shared edge pool (resampled and registered when an
/// edge is missing from it), so the hole's rim is welded to the faces on the
/// other side. Along a periodic direction the loop is unwrapped sample to
/// sample and then shifted by whole periods until its centre falls inside
/// the outer span: a hole straddling the surface's parameter origin keeps
/// one contiguous image.
#[allow(clippy::too_many_arguments)]
fn inner_wire_uv_loops(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    pcurve_uv: &dyn Fn(brepkit_topology::edge::EdgeId, Point3) -> Option<(f64, f64)>,
    (u_min, u_max, v_min, v_max): (f64, f64, f64, f64),
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<Vec<HoleLoop>, crate::OperationsError> {
    let tol_dup = 1e-10;
    let (u_period, v_period) = surface_periods(face_data.surface());
    let mut holes = Vec::new();
    for &wire_id in face_data.inner_wires() {
        let wire = topo.wire(wire_id)?;
        let mut samples: Vec<(u32, brepkit_topology::edge::EdgeId)> = Vec::new();
        for oe in wire.edges() {
            let gids: Vec<u32> = if let Some(pool) = edge_global_indices.get(&oe.edge().index()) {
                pool.clone()
            } else {
                let edge = topo.edge(oe.edge())?;
                sample_edge(topo, edge, deflection, angular_tol, circle_floor)?
                    .into_iter()
                    .map(|pt| {
                        *point_to_global
                            .entry(point_merge_key(pt, MERGE_GRID))
                            .or_insert_with(|| {
                                #[allow(clippy::cast_possible_truncation)]
                                let idx = merged.positions.len() as u32;
                                merged.positions.push(pt);
                                merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                                idx
                            })
                    })
                    .collect()
            };
            let ordered: Vec<u32> = if oe.is_forward() {
                gids
            } else {
                gids.into_iter().rev().collect()
            };
            for gid in ordered {
                let repeats = samples.last().is_some_and(|&(last, _)| {
                    last == gid
                        || (merged.positions[last as usize] - merged.positions[gid as usize])
                            .length()
                            < tol_dup
                });
                if !repeats {
                    samples.push((gid, oe.edge()));
                }
            }
        }
        if samples.len() > 2
            && let (Some(&(first, _)), Some(&(last, _))) = (samples.first(), samples.last())
            && (first == last
                || (merged.positions[first as usize] - merged.positions[last as usize]).length()
                    < tol_dup)
        {
            samples.pop();
        }
        if samples.len() < 3 {
            return Err(crate::OperationsError::InvalidInput {
                reason: format!("inner wire of {face_id:?} has fewer than 3 samples"),
            });
        }

        let mut loop_uv: Vec<(f64, f64, u32)> = Vec::with_capacity(samples.len());
        for (gid, edge) in samples {
            let pt = merged.positions[gid as usize];
            let (mut u, mut v) = pcurve_uv(edge, pt)
                .map_or_else(|| project_to_surface_uv(face_data.surface(), pt), Ok)?;
            if let Some(&(pu, pv, _)) = loop_uv.last() {
                if let Some((_, period)) = u_period {
                    u -= ((u - pu) / period).round() * period;
                }
                if let Some((_, period)) = v_period {
                    v -= ((v - pv) / period).round() * period;
                }
            }
            loop_uv.push((u, v, gid));
        }
        if let Some((_, period)) = u_period {
            let uv: Vec<(f64, f64)> = loop_uv.iter().map(|&(u, v, _)| (u, v)).collect();
            let points: Vec<Point3> = loop_uv
                .iter()
                .map(|&(_, _, gid)| merged.positions[gid as usize])
                .collect();
            if let Some(routed) = route_through_poles(face_data.surface(), &uv, &points, period) {
                loop_uv = routed
                    .into_iter()
                    .map(|(i, u, v)| (u, v, loop_uv[i].2))
                    .collect();
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let count = loop_uv.len() as f64;
        let centre_u = loop_uv.iter().map(|p| p.0).sum::<f64>() / count;
        let centre_v = loop_uv.iter().map(|p| p.1).sum::<f64>() / count;
        let shift = |centre: f64, lo: f64, hi: f64, period: Option<(f64, f64)>| {
            period.map_or(0.0, |(_, p)| {
                ((f64::midpoint(lo, hi) - centre) / p).round() * p
            })
        };
        let du = shift(centre_u, u_min, u_max, u_period);
        let dv = shift(centre_v, v_min, v_max, v_period);
        for p in &mut loop_uv {
            p.0 += du;
            p.1 += dv;
        }
        holes.push(loop_uv);
    }
    Ok(holes)
}

/// Close a boundary loop that winds a NURBS or sphere face's periodic u
/// direction once with no seam: such a face is a cap over a pole, a v-domain
/// edge that collapses to one point (a hemisphere on its equator). The loop
/// runs counter-clockwise in (u, v) about the surface's normal, a reversed
/// face included (the boolean builders flip the flag and keep the wire), so
/// the cap lies on its left, at the far v edge for a loop run toward +u. That edge is
/// appended as virtual boundary samples, all welded to the pole, so the
/// region closes in (u, v); a loop that winds toward a non-degenerate edge is
/// left alone. So is a NURBS face with inner wires, and a sphere face with a
/// hole around the pole: that makes it a band, which the pole row would cap
/// over. A sphere's other holes (a drill's entry) are carved as usual, and
/// its closing meridian is sampled like an edge.
/// A surface's point at `(u, v)`.
type SurfacePoint<'a> = Box<dyn Fn(f64, f64) -> Point3 + 'a>;

#[allow(clippy::too_many_arguments)]
fn close_loop_at_pole(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    boundary_uv: &mut Vec<(f64, f64)>,
    boundary_3d: &mut Vec<(Point3, u32, brepkit_topology::edge::EdgeId, bool)>,
    deflection: f64,
    angular_tol: f64,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<(), crate::OperationsError> {
    // The closing meridian's radius, for a surface whose seam sides are
    // sampled along it.
    let mut meridian_radius = None;
    let (evaluate, (v_lo, v_hi)): (SurfacePoint<'_>, (f64, f64)) = match face_data.surface() {
        FaceSurface::Nurbs(nurbs) if face_data.inner_wires().is_empty() => {
            let surface = face_data.surface();
            (
                Box::new(move |u, v| eval_surface_point(surface, u, v)),
                nurbs.domain_v(),
            )
        }
        FaceSurface::Sphere(sphere) => {
            let Some(spans) = hole_u_spans(topo, face_data, sphere)? else {
                return Ok(());
            };
            start_loop_clear_of_holes(boundary_uv, boundary_3d, &spans);
            meridian_radius = Some(sphere.radius());
            (
                Box::new(|u, v| sphere.evaluate(u, v)),
                (-std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2),
            )
        }
        _ => return Ok(()),
    };
    let (Some((_, period)), _) = surface_periods(face_data.surface()) else {
        return Ok(());
    };
    let wire = topo.wire(face_data.outer_wire())?;
    let mut uses: DetHashMap<usize, usize> = DetHashMap::default();
    for oe in wire.edges() {
        *uses.entry(oe.edge().index()).or_default() += 1;
    }
    if uses.values().any(|&n| n > 1) {
        return Ok(());
    }
    let (Some(&(first_u, _)), Some(&(last_u, last_v)), Some(&(_, _, edge, _))) =
        (boundary_uv.first(), boundary_uv.last(), boundary_3d.last())
    else {
        return Ok(());
    };
    // Net winding of the closed loop: the unwrapped run plus the wrapped
    // closing step back to the first sample.
    let progress = last_u - first_u;
    let closing = (first_u - last_u + period / 2.0).rem_euclid(period) - period / 2.0;
    let winding = progress + closing;
    if (winding.abs() - period).abs() > 1e-6 * period {
        return Ok(());
    }
    let v_far = if winding > 0.0 { v_hi } else { v_lo };
    let pole = evaluate(first_u, v_far);
    let scale = (evaluate(last_u, last_v) - pole).length().max(1e-12);
    let degenerate = (0..8).all(|k| {
        let u = first_u + winding * f64::from(k) / 8.0;
        (evaluate(u, v_far) - pole).length() <= 1e-9 * scale
    });
    if !degenerate {
        return Ok(());
    }
    let gid = *point_to_global
        .entry(point_merge_key(pole, MERGE_GRID))
        .or_insert_with(|| {
            #[allow(clippy::cast_possible_truncation)]
            let idx = merged.positions.len() as u32;
            merged.positions.push(pole);
            merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
            idx
        });
    // The loop continues into the first sample's image one winding on, like
    // the far side of a seam; the pole row then runs back to its column.
    let (first_pt, first_gid, _, _) = boundary_3d[0];
    let first_v = boundary_uv[0].1;
    boundary_uv.push((first_u + winding, first_v));
    boundary_3d.push((first_pt, first_gid, edge, true));
    // The closing meridian is a real curve on the surface: sampled like an
    // edge, one side's points welding to the other's, so no triangle chords
    // it from rim to pole.
    let along: Vec<f64> = match meridian_radius {
        Some(radius) => {
            let n = segments_for_chord_deviation_a(
                radius,
                (v_far - first_v).abs(),
                deflection,
                angular_tol,
                false,
            )
            .max(2);
            super::check_count("a pole's meridian", n.saturating_add(1), deflection)?;
            #[allow(clippy::cast_precision_loss)]
            (1..n).map(|k| k as f64 / n as f64).collect()
        }
        None => surface_path_parameters(
            &|t| evaluate(first_u, first_v + (v_far - first_v) * t),
            deflection,
            angular_tol,
            1,
        )?,
    };
    let side: Vec<(f64, Point3, u32)> = along
        .iter()
        .map(|&t| {
            let v = first_v + (v_far - first_v) * t;
            let p = evaluate(first_u, v);
            let id = *point_to_global
                .entry(point_merge_key(p, MERGE_GRID))
                .or_insert_with(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(p);
                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                    idx
                });
            (v, p, id)
        })
        .collect();
    for &(v, p, id) in &side {
        boundary_uv.push((first_u + winding, v));
        boundary_3d.push((p, id, edge, true));
    }
    let steps = boundary_uv.len().max(8);
    for k in 0..=steps {
        #[allow(clippy::cast_precision_loss)]
        let u = first_u + winding * (1.0 - k as f64 / steps as f64);
        boundary_uv.push((u, v_far));
        boundary_3d.push((pole, gid, edge, true));
    }
    for &(v, p, id) in side.iter().rev() {
        boundary_uv.push((first_u, v));
        boundary_3d.push((p, id, edge, true));
    }
    Ok(())
}

/// The boundary of a whole torus ring with holes, whose own wire is its seam
/// pair collapsed onto one vertex and so bounds nothing in `(u, v)`: the
/// period rectangle, started at a `u` and a `v` clear of every hole, as
/// `(uv, samples)`. Opposite sides carry the same 3D points, so they weld to
/// the same pool ids and the mesh closes across both seams. `None` for any
/// other face.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn whole_ring_rectangle(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    boundary_3d: &[(Point3, u32, brepkit_topology::edge::EdgeId, bool)],
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<
    Option<(
        Vec<(f64, f64)>,
        Vec<(Point3, u32, brepkit_topology::edge::EdgeId, bool)>,
    )>,
    crate::OperationsError,
> {
    let FaceSurface::Torus(torus) = face_data.surface() else {
        return Ok(None);
    };
    let (Some(&(anchor, _, seam, _)), false) =
        (boundary_3d.first(), face_data.inner_wires().is_empty())
    else {
        return Ok(None);
    };
    if boundary_3d
        .iter()
        .any(|&(p, ..)| (p - anchor).length() > 1e-9)
    {
        return Ok(None);
    }
    let wrap = |d: f64| (d + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
    // Each hole's span in u and in v, as (middle, half width).
    let mut spans: (Vec<(f64, f64)>, Vec<(f64, f64)>) = (Vec::new(), Vec::new());
    for &wire_id in face_data.inner_wires() {
        let mut walk: Vec<(f64, f64)> = Vec::new();
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            // A NURBS edge's knot span can run from its end vertex back.
            let at_t0 = edge.curve().evaluate_with_endpoints(t0, start, end);
            let (t0, t1) = if (at_t0 - start).length() <= (at_t0 - end).length() {
                (t0, t1)
            } else {
                (t1, t0)
            };
            for k in 0..=16 {
                let f = f64::from(k) / 16.0;
                let f = if oe.is_forward() { f } else { 1.0 - f };
                let t = t0 + (t1 - t0) * f;
                let (u, v) =
                    torus.project_point(edge.curve().evaluate_with_endpoints(t, start, end));
                walk.push(
                    walk.last()
                        .map_or((u, v), |&(lu, lv)| (lu + wrap(u - lu), lv + wrap(v - lv))),
                );
            }
        }
        let bounds = |coord: fn(&(f64, f64)) -> f64| {
            let (lo, hi) = walk
                .iter()
                .map(coord)
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| {
                    (lo.min(x), hi.max(x))
                });
            (f64::midpoint(lo, hi), 0.5 * (hi - lo))
        };
        spans.0.push(bounds(|p| p.0));
        spans.1.push(bounds(|p| p.1));
    }
    let clear_of = |spans: &[(f64, f64)]| {
        (0..64)
            .map(|k| TAU * f64::from(k) / 64.0)
            .max_by(|&a, &b| {
                let clearance = |x: f64| {
                    spans
                        .iter()
                        .map(|&(middle, half)| wrap(x - middle).abs() - half)
                        .fold(f64::INFINITY, f64::min)
                };
                clearance(a).total_cmp(&clearance(b))
            })
            .unwrap_or(0.0)
    };
    let (major, minor) = (torus.major_radius(), torus.minor_radius());
    // A horn torus's inner equator is one point, the pinch: seamed there, the
    // rectangle's sides would close on it and bound nothing.
    if major - minor <= brepkit_math::tolerance::Tolerance::new().linear {
        spans.1.push((std::f64::consts::PI, 0.0));
    }
    let (u0, v0) = (clear_of(&spans.0), clear_of(&spans.1));
    let nu =
        segments_for_chord_deviation_a(major + minor, TAU, deflection, angular_tol, circle_floor)
            .max(8);
    let nv =
        segments_for_chord_deviation_a(minor, TAU, deflection, angular_tol, circle_floor).max(8);
    super::check_count("a ring round its tube", nu.saturating_add(1), deflection)?;
    super::check_count("a ring's tube", nv.saturating_add(1), deflection)?;
    #[allow(clippy::cast_precision_loss)]
    let (du, dv) = (TAU / nu as f64, TAU / nv as f64);
    let mut id_of = |p: Point3| {
        *point_to_global
            .entry(point_merge_key(p, MERGE_GRID))
            .or_insert_with(|| {
                #[allow(clippy::cast_possible_truncation)]
                let idx = merged.positions.len() as u32;
                merged.positions.push(p);
                merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                idx
            })
    };
    #[allow(clippy::cast_precision_loss)]
    let along_u: Vec<(f64, Point3)> = (0..nu)
        .map(|k| {
            let u = u0 + du * k as f64;
            (u, torus.evaluate(u, v0))
        })
        .collect();
    #[allow(clippy::cast_precision_loss)]
    let along_v: Vec<(f64, Point3)> = (0..nv)
        .map(|k| {
            let v = v0 + dv * k as f64;
            (v, torus.evaluate(u0, v))
        })
        .collect();
    let mut uvs = Vec::with_capacity(2 * (nu + nv));
    let mut samples = Vec::with_capacity(2 * (nu + nv));
    let mut push = |uv: (f64, f64), p: Point3, samples: &mut Vec<_>| {
        uvs.push(uv);
        samples.push((p, id_of(p), seam, true));
    };
    for &(u, p) in &along_u {
        push((u, v0), p, &mut samples);
    }
    for &(v, p) in &along_v {
        push((u0 + TAU, v), p, &mut samples);
    }
    for k in 0..nu {
        let (u, p) = if k == 0 {
            (u0 + TAU, along_u[0].1)
        } else {
            along_u[nu - k]
        };
        push((u, v0 + TAU), p, &mut samples);
    }
    for k in 0..nv {
        let (v, p) = if k == 0 {
            (v0 + TAU, along_v[0].1)
        } else {
            along_v[nv - k]
        };
        push((u0, v), p, &mut samples);
    }
    Ok(Some((uvs, samples)))
}

/// Rotate a loop that winds u once to start at the sample whose meridian
/// toward the pole the face holds stays farthest in u from every hole span
/// and from every boundary sample on the way: the pole closure runs a
/// virtual seam up that meridian, which must not cross a hole, nor run along
/// or beside a boundary edge heading for the pole (a wall through the axis,
/// or a steep wall arc beside the start), where its samples would interleave
/// with the edge's. The samples after the old start shift by the loop's
/// winding so u stays continuous.
fn start_loop_clear_of_holes(
    boundary_uv: &mut [(f64, f64)],
    boundary_3d: &mut [(Point3, u32, brepkit_topology::edge::EdgeId, bool)],
    spans: &[(f64, f64)],
) {
    let wrap = |d: f64| (d + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI;
    let n = boundary_uv.len();
    if n < 2 {
        return;
    }
    let (first_u, last_u) = (boundary_uv[0].0, boundary_uv[n - 1].0);
    let closing = wrap(first_u - last_u);
    let winding = last_u - first_u + closing;
    // Toward the pole the face holds: higher v for a loop run toward +u.
    let toward = if winding > 0.0 { 1.0 } else { -1.0 };
    // Each sample's clearance, visiting samples from the pole down: the
    // set holds the u of every sample strictly nearer the pole, so the
    // nearest in u (around the turn) is a lookup, not a scan.
    let key = |u: f64| u.rem_euclid(TAU).to_bits();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| (boundary_uv[b].1 * toward).total_cmp(&(boundary_uv[a].1 * toward)));
    let mut beyond = std::collections::BTreeSet::new();
    let mut clearance = vec![0.0; n];
    let mut next = 0;
    for &i in &order {
        let (u, v) = boundary_uv[i];
        while next < n && boundary_uv[order[next]].1 * toward > v.madd(toward, 1e-12) {
            beyond.insert(key(boundary_uv[order[next]].0));
            next += 1;
        }
        let at = key(u);
        let nearest = [
            beyond.range(..=at).next_back(),
            beyond.range(at..).next(),
            beyond.first(),
            beyond.last(),
        ]
        .into_iter()
        .flatten()
        .map(|&b| wrap(f64::from_bits(b) - u).abs())
        .fold(f64::INFINITY, f64::min);
        clearance[i] = spans
            .iter()
            .map(|&(middle, half)| wrap(u - middle).abs() - half)
            .fold(nearest, f64::min);
    }
    let Some(best) = (0..n).max_by(|&a, &b| clearance[a].total_cmp(&clearance[b])) else {
        return;
    };
    // Without holes a start already as clear stays put.
    if best == 0 || (spans.is_empty() && clearance[best] <= clearance[0]) {
        return;
    }
    for point in &mut boundary_uv[..best] {
        point.0 += winding;
    }
    boundary_uv.rotate_left(best);
    boundary_3d.rotate_left(best);
}

/// Each inner wire's span in a sphere face's u, as its middle and half
/// width; `None` when one winds u (a hole around a pole).
fn hole_u_spans(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> Result<Option<Vec<(f64, f64)>>, crate::OperationsError> {
    const SAMPLES: u32 = 16;
    let poles = SurfacePoles::of(face_data.surface());
    let mut spans = Vec::new();
    for &wire_id in face_data.inner_wires() {
        let mut us: Vec<Option<f64>> = Vec::new();
        for oe in topo.wire(wire_id)?.edges() {
            let edge = topo.edge(oe.edge())?;
            let (start, end) = (
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            );
            let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
            let mut fractions: Vec<f64> = (0..=SAMPLES)
                .map(|k| f64::from(k) / f64::from(SAMPLES))
                .collect();
            if let EdgeCurve::Circle(circle) = edge.curve() {
                fractions.extend(
                    circle_pole_params(circle, (t0, t1), sphere)
                        .into_iter()
                        .map(|t| (t - t0) / (t1 - t0)),
                );
                fractions.sort_by(f64::total_cmp);
            }
            for f in fractions {
                let f = if oe.is_forward() { f } else { 1.0 - f };
                let point = edge
                    .curve()
                    .evaluate_with_endpoints(t0 + (t1 - t0) * f, start, end);
                us.push(
                    poles
                        .v_of(point)
                        .is_none()
                        .then(|| sphere.project_point(point).0),
                );
            }
        }
        if us.iter().all(Option::is_none) {
            continue;
        }
        let (unwrapped, winding) = unwrap_loop_u(&us, TAU);
        if winding.abs() > std::f64::consts::PI {
            return Ok(None);
        }
        let (lo, hi) = unwrapped
            .iter()
            .flat_map(|&(u_in, u_out)| [u_in, u_out])
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), u| {
                (lo.min(u), hi.max(u))
            });
        spans.push((f64::midpoint(lo, hi), 0.5 * (hi - lo)));
    }
    Ok(Some(spans))
}

/// The poles of a surface whose u closes: each `(v, point)` at which a whole
/// u iso-line collapses to one point (a sphere's two, a revolved NURBS
/// profile's ends on the axis), and how near a point must lie to one to be
/// on it, where its u means nothing.
struct SurfacePoles {
    poles: Vec<(f64, Point3)>,
    tol: f64,
}

impl SurfacePoles {
    fn of(surface: &FaceSurface) -> Self {
        match surface {
            FaceSurface::Sphere(sphere) => Self {
                poles: [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2]
                    .into_iter()
                    .map(|v| (v, sphere.evaluate(0.0, v)))
                    .collect(),
                tol: MERGE_GRID.max(1e-9 * sphere.radius()),
            },
            FaceSurface::Nurbs(nurbs) if surface_periods(surface).0.is_some() => {
                // A vertex a file puts on the pole sits there only to its
                // own tolerance, so the pole is read geometrically.
                let reach = nurbs
                    .control_points()
                    .iter()
                    .flatten()
                    .fold(0.0_f64, |m, p| {
                        m.max(p.x().abs()).max(p.y().abs()).max(p.z().abs())
                    });
                let tol = 1e-6 * (1.0 + reach);
                let (u_lo, u_hi) = nurbs.domain_u();
                let (v_lo, v_hi) = nurbs.domain_v();
                let poles = [v_lo, v_hi]
                    .into_iter()
                    .filter_map(|v| {
                        let at = nurbs.evaluate(u_lo, v);
                        (1..8)
                            .all(|k| {
                                let u = u_lo + (u_hi - u_lo) * f64::from(k) / 8.0;
                                (nurbs.evaluate(u, v) - at).length() <= tol
                            })
                            .then_some((v, at))
                    })
                    .collect();
                Self { poles, tol }
            }
            _ => Self {
                poles: Vec::new(),
                tol: 0.0,
            },
        }
    }

    fn v_of(&self, point: Point3) -> Option<f64> {
        self.poles
            .iter()
            .find(|(_, pole)| (point - *pole).length() <= self.tol)
            .map(|&(v, _)| v)
    }
}

/// The parameters strictly inside a circle edge's domain `(t0, t1)` at which
/// it runs over one of a sphere's poles (a wall through the sphere's axis).
pub(super) fn circle_pole_params(
    circle: &brepkit_math::curves::Circle3D,
    (t0, t1): (f64, f64),
    sphere: &brepkit_math::surfaces::SphericalSurface,
) -> Vec<f64> {
    let tol = MERGE_GRID.max(1e-9 * sphere.radius());
    [std::f64::consts::FRAC_PI_2, -std::f64::consts::FRAC_PI_2]
        .into_iter()
        .filter_map(|v| {
            let pole = sphere.evaluate(0.0, v);
            let t = t0 + (circle.project(pole) - t0).rem_euclid(TAU);
            ((circle.evaluate(t) - pole).length() <= tol && t > t0 + 1e-9 && t < t1 - 1e-9)
                .then_some(t)
        })
        .collect()
}

/// A closed loop's u unwrapped sample to sample, and its net winding. A
/// `None` sample lies on a pole, where u is undefined: the loop runs along
/// the pole's row there, from the u it arrives at to the u it leaves at,
/// and that step can be any width (the wedge the loop opens at the pole),
/// so the loop starts just past a pole and the step across it is the one
/// that closes the loop. Each sample gets the u it is reached at and the u
/// it is left at, the two equal away from a pole.
fn unwrap_loop_u(us: &[Option<f64>], period: f64) -> (Vec<(f64, f64)>, f64) {
    let n = us.len();
    let wrap = |d: f64| d - (d / period).round() * period;
    let past_pole = (0..n).find(|&i| us[i].is_some() && us[(i + n - 1) % n].is_none());
    let Some(start) = past_pole.or_else(|| us.iter().position(Option::is_some)) else {
        return (vec![(0.0, 0.0); n], 0.0);
    };
    let mut out = vec![(0.0, 0.0); n];
    let mut prev: Option<f64> = None;
    for k in 0..n {
        let i = (start + k) % n;
        if let Some(u) = us[i] {
            let u = prev.map_or(u, |p| p + wrap(u - p));
            out[i] = (u, u);
            prev = Some(u);
        }
    }
    for k in 1..n {
        let i = (start + k) % n;
        if us[i].is_none() {
            let before = out[(i + n - 1) % n].1;
            let after = (k + 1..=n)
                .map(|j| (start + j) % n)
                .find(|&j| us[j].is_some())
                .map_or(before, |j| out[j].0);
            out[i] = (before, after);
        }
    }
    let (first, last) = (out[start].0, out[(start + n - 1) % n].1);
    (out, last - first + wrap(first - last))
}

/// A loop through a pole of a sphere or a u-closed NURBS cap (see
/// [`SurfacePoles`]), unwrapped by `unwrap_loop_u` with each pole sample
/// opened into the stretch of the pole's row the loop runs along, sampled as
/// densely as the pole closure's row: `(source sample, u, v)` per routed
/// sample. `None` when no sample, or every sample, lies on a pole.
fn route_through_poles(
    surface: &FaceSurface,
    uv: &[(f64, f64)],
    points: &[Point3],
    period: f64,
) -> Option<Vec<(usize, f64, f64)>> {
    let poles = SurfacePoles::of(surface);
    let pole_v: Vec<Option<f64>> = points.iter().map(|&p| poles.v_of(p)).collect();
    if pole_v.iter().all(Option::is_none) || pole_v.iter().all(Option::is_some) {
        return None;
    }
    let us: Vec<Option<f64>> = uv
        .iter()
        .zip(&pole_v)
        .map(|(&(u, _), pole)| pole.is_none().then_some(u))
        .collect();
    let (unwrapped, _) = unwrap_loop_u(&us, period);
    #[allow(clippy::cast_precision_loss)]
    let steps = uv.len().max(8) as f64;
    let mut routed = Vec::with_capacity(uv.len() + 16);
    for (i, (&(u_in, u_out), pole)) in unwrapped.iter().zip(&pole_v).enumerate() {
        let Some(v) = *pole else {
            routed.push((i, u_in, uv[i].1));
            continue;
        };
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let n = (((u_out - u_in).abs() / period * steps).ceil() as usize).min(MAX_PATH_SPANS);
        routed.extend((0..=n).map(|k| {
            #[allow(clippy::cast_precision_loss)]
            let f = if n == 0 { 0.0 } else { k as f64 / n as f64 };
            (i, u_in + (u_out - u_in) * f, v)
        }));
    }
    Some(routed)
}

/// The lobes of a loop that touches itself at a vertex and run the other way
/// round from the whole loop: holes it holds, as points with the range of
/// each lobe among them. `ids` names each point's CDT vertex, the same at
/// each visit of the vertex.
fn pinched_hole_lobes(
    ids: &[usize],
    points: &[brepkit_math::vec::Point2],
) -> (Vec<brepkit_math::vec::Point2>, Vec<(usize, usize)>) {
    let twice_area = |pts: &[brepkit_math::vec::Point2]| {
        (0..pts.len()).fold(0.0, |sum, i| {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            a.x().madd(b.y(), sum) - b.x() * a.y()
        })
    };
    let whole = twice_area(points);
    // The walk parts at each vertex it revisits into simple cycles: a stack
    // of the points since each vertex's last visit pops a cycle at the
    // revisit, and what is left at the end closes the last one. Wherever the
    // walk starts, and however many lobes meet at one vertex, each lobe is
    // one cycle; those that run the other way round from the whole are holes.
    let mut stack: Vec<usize> = Vec::with_capacity(ids.len());
    let mut at_depth: DetHashMap<usize, usize> = DetHashMap::default();
    let mut cycles: Vec<Vec<brepkit_math::vec::Point2>> = Vec::new();
    for (j, &id) in ids.iter().enumerate() {
        if let Some(&depth) = at_depth.get(&id) {
            cycles.push(stack[depth..].iter().map(|&k| points[k]).collect());
            for &k in &stack[depth + 1..] {
                at_depth.remove(&ids[k]);
            }
            stack.truncate(depth + 1);
            continue;
        }
        at_depth.insert(id, stack.len());
        stack.push(j);
    }
    cycles.push(stack.iter().map(|&k| points[k]).collect());
    let mut lobe_pts = Vec::new();
    let mut ranges = Vec::new();
    for cycle in cycles {
        if cycle.len() < 3 {
            continue;
        }
        let area = twice_area(&cycle);
        if area * whole < 0.0 && area.abs() > 1e-12 * whole.abs() {
            let start = lobe_pts.len();
            lobe_pts.extend_from_slice(&cycle);
            ranges.push((start, lobe_pts.len()));
        }
    }
    (lobe_pts, ranges)
}

/// A sphere face bounded by one loop with no seam, on the same sphere with
/// its poles moved, when its own chart reads the loop the wrong way round.
///
/// The face lies to the left of its loop, which runs counter-clockwise about
/// the sphere's normal. The mesher meshes what a loop bounds in `(u, v)`, so
/// a loop that winds round no pole must run counter-clockwise there. One
/// that runs clockwise holds both poles, and one through a pole can read as
/// the lune on its other side. The new chart puts both poles outside a face
/// that lies within one hemisphere, else one pole at the face's middle and
/// the other in the rest of the sphere, each clear of the loop. `None` when
/// the face's own chart reads it right, or no new chart does.
fn sphere_face_recharted(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<Option<brepkit_topology::face::Face>, crate::OperationsError> {
    const SAMPLES: u32 = 16;
    let FaceSurface::Sphere(sphere) = face_data.surface() else {
        return Ok(None);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(None);
    }
    let wire = topo.wire(face_data.outer_wire())?;
    let mut edges = DetHashSet::default();
    if !wire.edges().iter().all(|oe| edges.insert(oe.edge())) {
        return Ok(None);
    }
    let mut points = Vec::new();
    for oe in wire.edges() {
        let edge = topo.edge(oe.edge())?;
        let (start, end) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (t0, t1) = edge.curve().domain_with_endpoints(start, end);
        let along = |f: f64| if oe.is_forward() { f } else { 1.0 - f };
        // From the edge's start in the loop up to, not at, its end there,
        // which the next edge starts at.
        let mut fractions: Vec<f64> = (0..SAMPLES)
            .map(|k| f64::from(k) / f64::from(SAMPLES))
            .collect();
        if let EdgeCurve::Circle(circle) = edge.curve() {
            fractions.extend(
                circle_pole_params(circle, (t0, t1), sphere)
                    .into_iter()
                    .map(|t| along((t - t0) / (t1 - t0))),
            );
            fractions.sort_by(f64::total_cmp);
        }
        for f in fractions {
            points.push(edge.curve().evaluate_with_endpoints(
                t0 + (t1 - t0) * along(f),
                start,
                end,
            ));
        }
    }
    if points.len() < 3 || sphere_loop_reads_right(sphere, &points) != Some(false) {
        return Ok(None);
    }
    let center = sphere.center();
    let mut area = Vec3::new(0.0, 0.0, 0.0);
    for (i, &p) in points.iter().enumerate() {
        let q = points[(i + 1) % points.len()];
        area += (p - center).cross(q - center);
    }
    let Ok(middle) = area.normalize() else {
        return Ok(None);
    };
    let directions: Vec<Vec3> = points
        .iter()
        .filter_map(|&p| (p - center).normalize().ok())
        .collect();
    // Both poles stand at least this angle off the loop.
    let margin = 1e-3_f64;
    let clear = margin.cos();
    let within_hemisphere = directions.iter().all(|d| d.dot(middle) > margin.sin());
    let axis = if within_hemisphere {
        let aside = if middle.x().abs() < 0.9 {
            Vec3::new(1.0, 0.0, 0.0)
        } else {
            Vec3::new(0.0, 1.0, 0.0)
        };
        match middle.cross(aside).normalize() {
            Ok(axis) => axis,
            Err(_) => return Ok(None),
        }
    } else {
        middle
    };
    if directions.iter().any(|d| d.dot(axis).abs() >= clear) {
        return Ok(None);
    }
    let Ok(turned) =
        brepkit_math::surfaces::SphericalSurface::with_axis(center, sphere.radius(), axis)
    else {
        return Ok(None);
    };
    if sphere_loop_reads_right(&turned, &points) != Some(true) {
        return Ok(None);
    }
    let mut local = face_data.clone();
    local.set_surface(FaceSurface::Sphere(turned));
    Ok(Some(local))
}

/// How a sphere's chart reads a loop of points on it, the face to its left:
/// `Some(true)` when the mesher meshes that face (the loop winds once round a
/// pole, or round none and counter-clockwise in `(u, v)`), `Some(false)` when
/// it winds round none clockwise (the face holds both poles, or the loop
/// passes a pole on the wrong side of the face), `None` when it cannot tell.
pub fn sphere_loop_reads_right(
    sphere: &brepkit_math::surfaces::SphericalSurface,
    points: &[Point3],
) -> Option<bool> {
    use std::f64::consts::FRAC_PI_2;
    let tol = MERGE_GRID.max(1e-9 * sphere.radius());
    let (north, south) = (
        sphere.evaluate(0.0, FRAC_PI_2),
        sphere.evaluate(0.0, -FRAC_PI_2),
    );
    let pole_v = |p: Point3| {
        if (p - north).length() <= tol {
            Some(FRAC_PI_2)
        } else if (p - south).length() <= tol {
            Some(-FRAC_PI_2)
        } else {
            None
        }
    };
    let uv: Vec<(f64, f64)> = points.iter().map(|&p| sphere.project_point(p)).collect();
    let us: Vec<Option<f64>> = points
        .iter()
        .zip(&uv)
        .map(|(&p, &(u, _))| pole_v(p).is_none().then_some(u))
        .collect();
    if us.iter().all(Option::is_none) {
        return None;
    }
    let (unwrapped, winding) = unwrap_loop_u(&us, TAU);
    if (winding.abs() - TAU).abs() < 1e-6 {
        return Some(true);
    }
    if winding.abs() > 1e-6 {
        return None;
    }
    let mut corners = Vec::with_capacity(points.len() + 4);
    for (i, &p) in points.iter().enumerate() {
        let (u_in, u_out) = unwrapped[i];
        match pole_v(p) {
            Some(v) => corners.extend([(u_in, v), (u_out, v)]),
            None => corners.push((u_in, uv[i].1)),
        }
    }
    let mut twice_area = 0.0;
    for (i, &(u0, v0)) in corners.iter().enumerate() {
        let (u1, v1) = corners[(i + 1) % corners.len()];
        twice_area += u0 * v1 - u1 * v0;
    }
    let handed = sphere
        .x_axis()
        .cross(sphere.y_axis())
        .dot(sphere.z_axis())
        .signum();
    let signed = twice_area * handed;
    (signed.abs() > 1e-12).then_some(signed > 0.0)
}

/// A NURBS face's average speeds along u and v over its parameter box. Its
/// knot values carry no common scale (a converted cylinder's u spans 1 around
/// a circumference its v spans 4 along), so the CDT measures in lengths:
/// raw, it would pick diagonals running far round the surface and chord
/// through the solid.
/// Which parameter of a NURBS surface runs along straight rulings that are
/// all one vector: `Some(true)` for `u` (two control rows, the second the
/// first moved by that vector), `Some(false)` for `v`.
fn extrusion_direction(surface: &FaceSurface) -> Option<bool> {
    let FaceSurface::Nurbs(nurbs) = surface else {
        return None;
    };
    let cps = nurbs.control_points();
    let weights = nurbs.weights();
    let translated = |pairs: &mut dyn Iterator<Item = ((Point3, f64), (Point3, f64))>| {
        let pairs: Vec<_> = pairs.collect();
        let first = pairs.first()?;
        let along = first.1.0 - first.0.0;
        let scale = along.length().max(1.0);
        (along.length() > 1e-9
            && pairs.iter().all(|((a, wa), (b, wb))| {
                ((*b - *a) - along).length() <= 1e-9 * scale && (wa - wb).abs() <= 1e-12
            }))
        .then_some(())
    };
    if nurbs.degree_u() == 1 && cps.len() == 2 {
        translated(
            &mut cps[0]
                .iter()
                .copied()
                .zip(weights[0].iter().copied())
                .zip(cps[1].iter().copied().zip(weights[1].iter().copied())),
        )
        .map(|()| true)
    } else if nurbs.degree_v() == 1 && cps.iter().all(|row| row.len() == 2) {
        translated(
            &mut cps
                .iter()
                .zip(weights)
                .map(|(row, w)| ((row[0], w[0]), (row[1], w[1]))),
        )
        .map(|()| false)
    } else {
        None
    }
}

/// The triangulation metric of a trimmed NURBS extrusion: the curved
/// direction at its own speed, the rulings shrunk so their whole span is one
/// chord cell of the curve. Delaunay over raw rulings tens of cells long
/// joins a corner on one long side to curve samples far along it, a chord
/// through the surface; one cell wide, it ladders the two sides.
fn extrusion_metric(
    surface: &FaceSurface,
    (u_min, du): (f64, f64),
    (v_min, dv): (f64, f64),
    deflection: f64,
    angular_tol: f64,
) -> Option<(f64, f64)> {
    let along_u = extrusion_direction(surface)?;
    let (speed_u, speed_v) = nurbs_speeds(surface, (u_min, du), (v_min, dv))?;
    let at = |u: f64, v: f64| eval_surface_point(surface, u, v);
    let normal_at = |u: f64, v: f64| {
        let (u, v) = wrap_to_domain(surface, u, v);
        surface.normal(u, v)
    };
    let (u_span, v_span) = ((u_min, u_min + du), (v_min, v_min + dv));
    if along_u {
        let divisions = super::nurbs::iso_divisions_over(
            &at,
            &normal_at,
            false,
            (v_span, u_span),
            deflection,
            angular_tol,
            (1024, super::MAX_DIVISIONS),
        )
        .ok()?
        .max(1);
        let cell = speed_v * dv / divisions as f64;
        Some((cell / du, speed_v))
    } else {
        let divisions = super::nurbs::iso_divisions_over(
            &at,
            &normal_at,
            true,
            (u_span, v_span),
            deflection,
            angular_tol,
            (1024, super::MAX_DIVISIONS),
        )
        .ok()?
        .max(1);
        let cell = speed_u * du / divisions as f64;
        Some((speed_u, cell / dv))
    }
}

/// The triangulation metric of a NURBS face whose iso-lines run straight one
/// way without being one vector (a fillet band whose radius changes along
/// its edge): as for an extrusion, the straight direction shrunk to one chord
/// cell of the curved one. At their own speeds Delaunay joins the face's two
/// long sides across the curve whenever those run along the straight way.
fn ruled_metric(
    speeds: Option<(f64, f64)>,
    divisions: Option<(usize, usize)>,
    du: f64,
    dv: f64,
) -> Option<(f64, f64)> {
    let (speed_u, speed_v) = speeds?;
    match divisions? {
        (n_u, n_v) if n_u > 1 && n_v <= 1 => Some((speed_u, speed_u * du / n_u as f64 / dv)),
        (n_u, n_v) if n_u <= 1 && n_v > 1 => Some((speed_v * dv / n_v as f64 / du, speed_v)),
        _ => None,
    }
}

/// A face's grid lines, `u` then `v`.
type GridLines = (Vec<f64>, Vec<f64>);

/// The grid lines of a NURBS face across its parameter box, `u` then `v`:
/// each direction's spans keep the chords of the face's own iso-lines within
/// half the deflection and the angular tolerance, so the lines stand close
/// where the surface turns and far apart where it runs straight. `None` for a
/// face on another surface.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if a direction needs
/// more lines than the sample limit, or the grid more cells.
fn nurbs_grid_lines(
    surface: &FaceSurface,
    (u_min, du): (f64, f64),
    (v_min, dv): (f64, f64),
    deflection: f64,
    angular_tol: f64,
) -> Result<Option<GridLines>, crate::OperationsError> {
    let FaceSurface::Nurbs(nurbs) = surface else {
        return Ok(None);
    };
    let at = |u: f64, v: f64| eval_surface_point(surface, u, v);
    let normal_at = |u: f64, v: f64| {
        let (u, v) = wrap_to_domain(surface, u, v);
        surface.normal(u, v)
    };
    let (u_span, v_span) = ((u_min, u_min + du), (v_min, v_min + dv));
    let along = |along_u: bool, spans, knots: &[f64]| {
        super::nurbs::iso_breaks_over(
            &at,
            &normal_at,
            along_u,
            spans,
            knots,
            deflection,
            angular_tol,
            super::MAX_DIVISIONS,
        )
    };
    let (us, vs) = (
        along(true, (u_span, v_span), nurbs.knots_u())?,
        along(false, (v_span, u_span), nurbs.knots_v())?,
    );
    super::check_grid(us.len(), vs.len(), deflection)?;
    Ok(Some((us, vs)))
}

/// Cuts a NURBS face's grid lines where a cell between them is twisted
/// ([`super::nurbs::untwist_lines`]).
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the cut grid is
/// past the sample limit.
fn untwist_grid_lines(
    surface: &FaceSurface,
    us: &mut Vec<f64>,
    vs: &mut Vec<f64>,
    deflection: f64,
) -> Result<(), crate::OperationsError> {
    let at = |u: f64, v: f64| eval_surface_point(surface, u, v);
    super::nurbs::untwist_lines(&at, us, vs, deflection)
}

fn nurbs_speeds(
    surface: &FaceSurface,
    (u_min, du): (f64, f64),
    (v_min, dv): (f64, f64),
) -> Option<(f64, f64)> {
    const ROWS: u32 = 5;
    let FaceSurface::Nurbs(nurbs) = surface else {
        return None;
    };
    if du <= 1e-15 || dv <= 1e-15 {
        return None;
    }
    let (mut speed_u, mut speed_v) = (0.0, 0.0);
    for i in 0..ROWS {
        for j in 0..ROWS {
            let (u, v) = wrap_to_domain(
                surface,
                u_min + du * (f64::from(i) + 0.5) / f64::from(ROWS),
                v_min + dv * (f64::from(j) + 0.5) / f64::from(ROWS),
            );
            let d = nurbs.derivatives(u, v, 1);
            speed_u += d[1][0].length();
            speed_v += d[0][1].length();
        }
    }
    (speed_u > 0.0 && speed_v > 0.0).then_some((speed_u, speed_v))
}

/// A NURBS or sphere band bounded by two loops that each wind once around
/// the periodic u direction (a sphere zone left by a coaxial bore, a ball's
/// hemisphere less a box over its pole) encloses nothing in the unwrapped
/// `(u, v)` plane. The hole is joined to the outer loop along a virtual seam
/// from an outer sample to the hole sample nearest it in u: the outer loop
/// continues one winding on, climbs the seam, runs the hole the other way
/// round, and descends the seam's copy one period back, whose samples it
/// shares. The face's other holes stay holes: of the outer samples tried
/// (about 32 around the loop) whose seam crosses none of them, the seam
/// starts at the one keeping farthest from every boundary sample, and the
/// other holes move by whole periods into the rotated loop's span.
/// Returns whether the loops were joined, or the refusal of a seam that
/// would take too many samples.
#[allow(clippy::too_many_arguments)]
fn join_winding_hole(
    face_data: &brepkit_topology::face::Face,
    holes: &mut Vec<HoleLoop>,
    boundary_uv: &mut Vec<(f64, f64)>,
    boundary_3d: &mut Vec<(Point3, u32, brepkit_topology::edge::EdgeId, bool)>,
    deflection: f64,
    angular_tol: f64,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<bool, crate::OperationsError> {
    let surface = face_data.surface();
    if !matches!(surface, FaceSurface::Nurbs(_) | FaceSurface::Sphere(_)) || boundary_uv.len() < 3 {
        return Ok(false);
    }
    let (Some((_, period)), _) = surface_periods(surface) else {
        return Ok(false);
    };
    let winding = |us: &[f64]| {
        let (first, last) = (us[0], us[us.len() - 1]);
        last - first + (first - last + period / 2.0).rem_euclid(period) - period / 2.0
    };
    let once = |w: f64| (w.abs() - period).abs() <= 1e-6 * period;
    let hole_winding = |h: &HoleLoop| winding(&h.iter().map(|p| p.0).collect::<Vec<_>>());
    let outer_us: Vec<f64> = boundary_uv.iter().map(|p| p.0).collect();
    let w_outer = winding(&outer_us);
    let winding_holes: Vec<usize> = (0..holes.len())
        .filter(|&i| holes[i].len() >= 3 && once(hole_winding(&holes[i])))
        .collect();
    let [which] = winding_holes[..] else {
        return Ok(false);
    };
    if !once(w_outer) {
        return Ok(false);
    }
    let Some(&(_, _, edge, _)) = boundary_3d.last() else {
        return Ok(false);
    };

    // The hole runs against the outer loop, starting nearest the seam's
    // outer sample, from one winding on back down to it.
    let w_hole = hole_winding(&holes[which]);
    let mut hole = holes[which].clone();
    if w_hole.signum() == w_outer.signum() {
        hole.reverse();
    }
    let others: Vec<((f64, f64), (f64, f64))> = holes
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != which)
        .flat_map(|(_, h)| {
            h.iter()
                .zip(h.iter().cycle().skip(1))
                .map(|(a, b)| ((a.0, a.1), (b.0, b.1)))
                .collect::<Vec<_>>()
        })
        .collect();
    let clear = |a: (f64, f64), b: (f64, f64)| {
        others.iter().all(|&(p, q)| {
            (-2..=2).all(|k| {
                let shift = f64::from(k) * period;
                !segments_cross(a, b, (p.0 + shift, p.1), (q.0 + shift, q.1))
            })
        })
    };
    // How far a seam keeps from every boundary sample but its own ends: a
    // seam leaving a corner beside a steep wall arc interleaves its samples
    // with the arc's and leaves the mesh open.
    let clearance = |a: (f64, f64), b: (f64, f64), rotate: usize, start: usize| {
        let mid = f64::midpoint(a.0, b.0);
        let (du, dv) = (b.0 - a.0, b.1 - a.1);
        let len2 = du.madd(du, dv * dv).max(1e-300);
        let away = |(u, v): (f64, f64)| {
            let u = u + ((mid - u) / period).round() * period;
            let f = ((u - a.0).madd(du, (v - a.1) * dv) / len2).clamp(0.0, 1.0);
            (u - f.madd(du, a.0)).hypot(v - f.madd(dv, a.1))
        };
        let outer = boundary_uv
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != rotate)
            .map(|(_, &p)| away(p));
        let inner = hole
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != start)
            .map(|(_, &(u, v, _))| away((u, v)));
        let beside = others.iter().map(|&(p, _)| away(p));
        outer
            .chain(inner)
            .chain(beside)
            .fold(f64::INFINITY, f64::min)
    };
    let n = boundary_uv.len();
    let mut found: Option<(f64, usize, usize)> = None;
    for rotate in (0..n).step_by((n / 32).max(1)) {
        let (u0, v0) = boundary_uv[rotate];
        let offset = |u: f64| (u - u0 + period / 2.0).rem_euclid(period) - period / 2.0;
        let Some(start) = (0..hole.len())
            .min_by(|&a, &b| offset(hole[a].0).abs().total_cmp(&offset(hole[b].0).abs()))
        else {
            return Ok(false);
        };
        let from = (u0 + w_outer, v0);
        let target = (u0 + w_outer + offset(hole[start].0), hole[start].1);
        if clear(from, target) {
            let score = clearance(from, target, rotate, start);
            if found.is_none_or(|(best, _, _)| score > best) {
                found = Some((score, rotate, start));
            }
        }
    }
    let Some((_, rotate, start)) = found else {
        return Ok(false);
    };
    holes.remove(which);
    if rotate > 0 {
        let head: Vec<(f64, f64)> = boundary_uv
            .drain(..rotate)
            .map(|(u, v)| (u + w_outer, v))
            .collect();
        boundary_uv.extend(head);
        boundary_3d.rotate_left(rotate);
        // The loop now spans a turn from its new start, so the other holes
        // move by whole periods to lie within it (the seam crosses none).
        let lo = boundary_uv[0].0.min(boundary_uv[0].0 + w_outer);
        for other in holes.iter_mut() {
            #[allow(clippy::cast_precision_loss)]
            let centre = other.iter().map(|p| p.0).sum::<f64>() / other.len().max(1) as f64;
            let shift = ((centre - lo) / period).floor() * period;
            for p in other.iter_mut() {
                p.0 -= shift;
            }
        }
    }
    let (u0, v0) = boundary_uv[0];
    let offset = |u: f64| (u - u0 + period / 2.0).rem_euclid(period) - period / 2.0;
    hole.rotate_left(start);
    let mut hole_uv: Vec<(f64, f64, u32)> = Vec::with_capacity(hole.len() + 1);
    let mut prev_u = u0 + w_outer + offset(hole[0].0);
    for &(u, v, gid) in &hole {
        let u = u - ((u - prev_u) / period).round() * period;
        hole_uv.push((u, v, gid));
        prev_u = u;
    }

    // Seam samples, spaced like the outer loop's, and as finely as the seam
    // bends: across a tightly curved band (a torus's tube) the rims' spacing
    // leaves chords that cut far inside the surface.
    let (_, first_gid, _, _) = boundary_3d[0];
    let spacing = boundary_3d
        .windows(2)
        .map(|w| (w[1].0 - w[0].0).length())
        .sum::<f64>()
        / (boundary_3d.len() - 1) as f64;
    let (top_u, top_v) = (u0 + w_outer, v0);
    let (hole_u, hole_v, hole_gid) = hole_uv[0];
    let reach =
        (merged.positions[hole_gid as usize] - merged.positions[first_gid as usize]).length();
    let along = surface_path_parameters(
        &|t| {
            eval_surface_point(
                surface,
                top_u + (hole_u - top_u) * t,
                top_v + (hole_v - top_v) * t,
            )
        },
        deflection,
        angular_tol,
        seam_spans(reach, spacing),
    )?;
    let seam: Vec<(f64, f64, u32)> = along
        .iter()
        .map(|&t| {
            let (u, v) = (top_u + (hole_u - top_u) * t, top_v + (hole_v - top_v) * t);
            let pt = eval_surface_point(surface, u, v);
            let gid = *point_to_global
                .entry(point_merge_key(pt, MERGE_GRID))
                .or_insert_with(|| {
                    #[allow(clippy::cast_possible_truncation)]
                    let idx = merged.positions.len() as u32;
                    merged.positions.push(pt);
                    merged.normals.push(Vec3::new(0.0, 0.0, 0.0));
                    idx
                });
            (u, v, gid)
        })
        .collect();

    let mut push = |u: f64, v: f64, gid: u32| {
        boundary_uv.push((u, v));
        boundary_3d.push((merged.positions[gid as usize], gid, edge, true));
    };
    push(top_u, top_v, first_gid);
    for &(u, v, gid) in &seam {
        push(u, v, gid);
    }
    for &(u, v, gid) in &hole_uv {
        push(u, v, gid);
    }
    push(hole_u - w_outer, hole_v, hole_gid);
    for &(u, v, gid) in seam.iter().rev() {
        push(u - w_outer, v, gid);
    }
    Ok(true)
}

/// The most spans a virtual boundary the mesher runs across a face is cut
/// into, whatever its geometry asks: a degenerate or hostile face (rims whose
/// samples all but coincide, a loop that winds without end) must not size
/// the samples it allocates and evaluates from a ratio without bound.
const MAX_PATH_SPANS: usize = 4096;

/// Equal spans for a seam `reach` long beside boundary samples `spacing`
/// apart: as many as the spacing asks, at least two and at most
/// [`MAX_PATH_SPANS`].
fn seam_spans(reach: f64, spacing: f64) -> usize {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let spans = (reach / spacing.max(1e-12)).ceil() as usize;
    spans.clamp(2, MAX_PATH_SPANS)
}

/// Where to sample a path `point(t)` across a surface, as the parameters
/// strictly inside `(0, 1)`, to the bounds an edge is sampled to: each chord
/// within `deflection` of the path and turning by at most `angular_tol`. A
/// virtual boundary the mesher runs across a face (a pole closure's
/// meridian, a seam joining two loops) borders triangles like an edge does.
///
/// The path starts as `min_spans` equal spans and each span that fails is
/// halved. A span is read at its quarter points: an inflected path can cross
/// its chord at the middle and still stand far off it either side. A crease
/// (a cornered or degree-one profile) keeps its turn however finely it is
/// halved, so a span already within the deflection and shorter than it,
/// whose halves keep most of its turn, stands as it is; the halving it took
/// to get there stays beside the crease. It starts from at most
/// [`MAX_PATH_SPANS`] spans, and is cut as finely as the deflection needs.
///
/// # Errors
///
/// Returns [`crate::OperationsError::TooManySamples`] if the path would
/// take more samples than the tessellator allows: one that no sampling
/// brings within the deflection does.
fn surface_path_parameters(
    point: &dyn Fn(f64) -> Point3,
    deflection: f64,
    angular_tol: f64,
    min_spans: usize,
) -> Result<Vec<f64>, crate::OperationsError> {
    // The half-chords of an arc meet at half its turn.
    let turn =
        |a: Point3, mid: Point3, b: Point3| match ((mid - a).normalize(), (b - mid).normalize()) {
            (Ok(x), Ok(y)) => 2.0 * x.dot(y).clamp(-1.0, 1.0).acos(),
            _ => 0.0,
        };
    let off_chord = |p: Point3, a: Point3, b: Point3| {
        let chord = b - a;
        let len2 = chord.dot(chord);
        let f = if len2 > 0.0 {
            ((p - a).dot(chord) / len2).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (p - (a + chord * f)).length()
    };
    let spans = min_spans.clamp(1, MAX_PATH_SPANS);
    let mut pending: Vec<(f64, Point3, f64, Point3, u32)> = Vec::new();
    for k in (0..spans).rev() {
        #[allow(clippy::cast_precision_loss)]
        let (t0, t1) = (k as f64 / spans as f64, (k + 1) as f64 / spans as f64);
        pending.push((t0, point(t0), t1, point(t1), 0));
    }
    let mut along = Vec::new();
    while let Some((t0, p0, t1, p1, depth)) = pending.pop() {
        let quarters = [1.0, 2.0, 3.0].map(|k| point(t0 + (t1 - t0) * k / 4.0));
        let chord = (p1 - p0).length();
        let deviation = quarters
            .iter()
            .map(|&q| off_chord(q, p0, p1))
            .fold(0.0, f64::max);
        let bend = turn(p0, quarters[1], p1);
        let within = deviation <= deflection;
        let bends = angular_tol > 0.0 && bend > angular_tol;
        let crease = within
            && bends
            && chord <= deflection
            && turn(p0, quarters[0], quarters[1]).max(turn(quarters[1], quarters[2], p1))
                > 0.75 * bend;
        let satisfied = within && (!bends || crease);
        if satisfied {
            if t1 < 1.0 {
                along.push(t1);
            }
            continue;
        }
        super::check_count(
            "a path across a face",
            along.len() + pending.len() + 3,
            deflection,
        )?;
        let tm = 0.5 * (t0 + t1);
        if tm.to_bits() == t0.to_bits() || tm.to_bits() == t1.to_bits() {
            return Err(super::endless_samples("a path across a face", deflection));
        }
        pending.push((tm, quarters[1], t1, p1, depth + 1));
        pending.push((t0, p0, tm, quarters[1], depth + 1));
    }
    Ok(along)
}

/// Whether the open segments `a b` and `c d` cross in the plane.
fn segments_cross(a: (f64, f64), b: (f64, f64), c: (f64, f64), d: (f64, f64)) -> bool {
    let side = |p: (f64, f64), q: (f64, f64), r: (f64, f64)| {
        (q.0 - p.0).madd(r.1 - p.1, -((q.1 - p.1) * (r.0 - p.0)))
    };
    let (d1, d2) = (side(a, b, c), side(a, b, d));
    let (d3, d4) = (side(c, d, a), side(c, d, b));
    d1 * d2 < 0.0 && d3 * d4 < 0.0
}

/// Whether a cylinder or cone wall's outer wire carries a notch (a hole that
/// straddles its seam): it wraps round, its seam line used twice, between
/// rims at both ends of its axial extent (a pointed cone's apex at one end),
/// closed or cut into arcs, and some other edge runs between them.
pub(super) fn is_notched_wall(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    let surface = face_data.surface();
    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
        return Ok(false);
    }
    let edges = topo.wire(face_data.outer_wire())?.edges();
    if edges.len() <= 4 {
        return Ok(false);
    }
    let mut ends: Vec<(f64, f64)> = Vec::with_capacity(edges.len());
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let v = |id| -> Result<f64, crate::OperationsError> {
            let p = topo.vertex(id)?.point();
            Ok(surface.project_point(p).map_or(f64::NAN, |(_, v)| v))
        };
        ends.push((v(edge.start())?, v(edge.end())?));
    }
    let v_min = ends
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .fold(f64::INFINITY, f64::min);
    let v_max = ends
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .fold(f64::NEG_INFINITY, f64::max);
    let tol = 1e-7 * (1.0 + v_max.abs().max(v_min.abs()));
    let (mut rim_low, mut rim_high, mut seam_twice, mut between) = (false, false, false, false);
    for (oe, &(va, vb)) in edges.iter().zip(&ends) {
        let edge = topo.edge(oe.edge())?;
        let repeated = edges
            .iter()
            .filter(|other| other.edge() == oe.edge())
            .count()
            > 1;
        match edge.curve() {
            EdgeCurve::Line if repeated => seam_twice = true,
            _ if repeated => return Ok(false),
            EdgeCurve::Circle(_) if (va - vb).abs() <= tol && (va - v_min).abs() <= tol => {
                rim_low = true;
            }
            EdgeCurve::Circle(_) if (va - vb).abs() <= tol && (va - v_max).abs() <= tol => {
                rim_high = true;
            }
            _ => between = true,
        }
    }
    // A pointed cone's seam runs up to its apex, which stands in for a rim.
    let apex_end = match surface {
        FaceSurface::Cone(cone) => {
            let v_apex = surface
                .project_point(cone.apex())
                .map_or(f64::NAN, |(_, v)| v);
            (v_apex - v_min).abs() <= tol || (v_apex - v_max).abs() <= tol
        }
        _ => false,
    };
    Ok(seam_twice && between && ((rim_low && rim_high) || (apex_end && (rim_low || rim_high))))
}

/// Whether a cylinder or cone wall wraps round, its seam line used twice,
/// with a NURBS edge in its outer wire (most often a marched or traced
/// section winding the wall between the seam's two copies). A pointed cone's
/// wall reaching its apex keeps its own meshers.
pub(super) fn is_wound_wall(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    let apex = match face_data.surface() {
        FaceSurface::Cone(cone) => Some(cone.apex()),
        FaceSurface::Cylinder(_) => None,
        _ => return Ok(false),
    };
    let edges = topo.wire(face_data.outer_wire())?.edges();
    let (mut seam_twice, mut section) = (false, false);
    for (i, oe) in edges.iter().enumerate() {
        let edge = topo.edge(oe.edge())?;
        if let Some(apex) = apex {
            for v in [edge.start(), edge.end()] {
                if (topo.vertex(v)?.point() - apex).length() < 1e-9 {
                    return Ok(false);
                }
            }
        }
        match edge.curve() {
            EdgeCurve::Line => {
                seam_twice |= edges[i + 1..].iter().any(|other| other.edge() == oe.edge());
            }
            EdgeCurve::NurbsCurve(_) => section = true,
            EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_) => {}
        }
    }
    Ok(seam_twice && section)
}

/// Whether a cone face is a pointed cone's tip that a section bounds: its
/// wire reaches the apex and carries a NURBS or ellipse edge.
pub(super) fn is_section_tip(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    let FaceSurface::Cone(cone) = face_data.surface() else {
        return Ok(false);
    };
    let (mut apex, mut section) = (false, false);
    for oe in topo.wire(face_data.outer_wire())?.edges() {
        let edge = topo.edge(oe.edge())?;
        for v in [edge.start(), edge.end()] {
            apex |= (topo.vertex(v)?.point() - cone.apex()).length() < 1e-9;
        }
        section |= matches!(
            edge.curve(),
            EdgeCurve::NurbsCurve(_) | EdgeCurve::Ellipse(_)
        );
    }
    Ok(apex && section)
}

/// Whether a cylinder or cone wall's outer wire puts a vertex part way along
/// a ruling: a line edge ending strictly between the wall's lowest and
/// highest rim, as when same-domain faces stacked along the axis merge and
/// keep the vertices their side neighbours are still split at.
pub(super) fn is_split_ruling_wall(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    let surface = face_data.surface();
    if !matches!(surface, FaceSurface::Cylinder(_) | FaceSurface::Cone(_)) {
        return Ok(false);
    }
    let edges = topo.wire(face_data.outer_wire())?.edges();
    if edges.len() <= 4 {
        return Ok(false);
    }
    let v_at = |id| -> Result<f64, crate::OperationsError> {
        let p = topo.vertex(id)?.point();
        Ok(surface.project_point(p).map_or(f64::NAN, |(_, v)| v))
    };
    let (mut v_min, mut v_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut ruling_ends = Vec::new();
    for oe in edges {
        let edge = topo.edge(oe.edge())?;
        let (va, vb) = (v_at(edge.start())?, v_at(edge.end())?);
        v_min = v_min.min(va).min(vb);
        v_max = v_max.max(va).max(vb);
        if matches!(edge.curve(), EdgeCurve::Line) {
            ruling_ends.extend([va, vb]);
        }
    }
    // Scaled by the wall's own span: `v` is measured from the surface's
    // origin, which can sit anywhere along the axis.
    let tol = 1e-7 * (1.0 + (v_max - v_min));
    Ok(ruling_ends
        .iter()
        .any(|&v| v > v_min + tol && v < v_max - tol))
}

/// Whether a cylinder or cone face is bounded in part by a section: a NURBS
/// or ellipse edge in its outer wire.
pub(super) fn is_section_bound(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    if !matches!(
        face_data.surface(),
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_)
    ) {
        return Ok(false);
    }
    for oe in topo.wire(face_data.outer_wire())?.edges() {
        if matches!(
            topo.edge(oe.edge())?.curve(),
            EdgeCurve::NurbsCurve(_) | EdgeCurve::Ellipse(_)
        ) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether a cylinder or cone wall's outer loop has rim circles at more
/// than two heights: a wall stepped along its rim (a C-shaped band whose
/// floor drops partway round), which is not the (u, v) box between two rims
/// that the grid fills.
pub(super) fn is_stepped_wall(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
) -> Result<bool, crate::OperationsError> {
    let (origin, axis, scale) = match face_data.surface() {
        FaceSurface::Cylinder(c) => (c.origin(), c.axis(), c.radius()),
        FaceSurface::Cone(c) => (c.apex(), c.axis(), 1.0),
        _ => return Ok(false),
    };
    let tol = 1e-7 * scale.max(1.0);
    let mut heights: Vec<f64> = Vec::new();
    for oe in topo.wire(face_data.outer_wire())?.edges() {
        if let EdgeCurve::Circle(circle) = topo.edge(oe.edge())?.curve() {
            let h = axis.dot(circle.center() - origin);
            if heights.iter().all(|&k| (k - h).abs() > tol) {
                heights.push(h);
            }
        }
    }
    Ok(heights.len() > 2)
}

/// Evaluate a non-planar surface at `(u, v)` and return a 3D point.
fn eval_surface_point(surface: &FaceSurface, u: f64, v: f64) -> Point3 {
    let (u, v) = wrap_to_domain(surface, u, v);
    surface.evaluate(u, v).unwrap_or(Point3::new(0.0, 0.0, 0.0))
}

/// The `(origin, period)` of each periodic parameter direction.
///
/// Analytic surfaces are periodic in their angles over `[0, 2π)` (the torus
/// in both). A NURBS surface is periodic in a direction when it closes on
/// itself there, with its knot domain as the period: a cylinder or cone
/// converted to NURBS wraps its u domain exactly like the analytic angle.
type Period = Option<(f64, f64)>;
fn surface_periods(surface: &FaceSurface) -> (Period, Period) {
    let turn = Some((0.0, std::f64::consts::TAU));
    match surface {
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) | FaceSurface::Sphere(_) => (turn, None),
        FaceSurface::Torus(_) => (turn, turn),
        FaceSurface::Nurbs(n) => {
            let period =
                |closed: bool, (lo, hi): (f64, f64)| (closed && hi > lo).then_some((lo, hi - lo));
            (
                period(n.is_periodic_u(), n.domain_u()),
                period(n.is_periodic_v(), n.domain_v()),
            )
        }
        FaceSurface::Plane { .. } => (None, None),
    }
}

/// Map an unwrapped `(u, v)` back into a NURBS surface's knot domain along
/// its periodic directions. Analytic surfaces evaluate any angle directly.
fn wrap_to_domain(surface: &FaceSurface, u: f64, v: f64) -> (f64, f64) {
    let FaceSurface::Nurbs(_) = surface else {
        return (u, v);
    };
    let (u_period, v_period) = surface_periods(surface);
    let wrap = |x: f64, period: Period| {
        period.map_or(x, |(origin, len)| origin + (x - origin).rem_euclid(len))
    };
    (wrap(u, u_period), wrap(v, v_period))
}

/// Estimate the effective radius of a surface for sample density calculation.
fn estimate_surface_radius(surface: &FaceSurface) -> f64 {
    match surface {
        FaceSurface::Cylinder(cyl) => cyl.radius(),
        FaceSurface::Cone(_) => 1.0,
        FaceSurface::Sphere(sphere) => sphere.radius(),
        FaceSurface::Torus(torus) => torus.major_radius() + torus.minor_radius(),
        FaceSurface::Nurbs(_) | FaceSurface::Plane { .. } => 1.0,
    }
}

/// The circular radius a developable stripe's angular coordinate spans, or
/// `None` for a surface that is not a cylinder or a cone.
fn developable_stripe_radius(surface: &FaceSurface, v_min: f64, v_max: f64) -> Option<f64> {
    let radius = match surface {
        FaceSurface::Cylinder(cylinder) => cylinder.radius().abs(),
        // A cone's rulings change radius; its widest end drives both the
        // metric and the chord bound.
        FaceSurface::Cone(cone) => cone.radius_at(v_min).abs().max(cone.radius_at(v_max).abs()),
        FaceSurface::Plane { .. }
        | FaceSurface::Nurbs(_)
        | FaceSurface::Sphere(_)
        | FaceSurface::Torus(_) => return None,
    };
    (radius > 0.0).then_some(radius)
}

/// Whether a triangle spanning `u_span` radians of a developable stripe of
/// radius `radius` stays inside both requested tolerances.
///
/// The surface is straight along its rulings, so the deviation only depends on
/// the angular extent: a chord across `u_span` radians departs the circle by
/// `radius * (1 - cos(u_span / 2))` at its midpoint, and the normal turns by
/// exactly `u_span` across the triangle.
fn stripe_span_within_tolerance(
    radius: f64,
    u_span: f64,
    deflection: f64,
    angular_tol: f64,
) -> bool {
    // The transverse arcs are sampled at exactly whichever tolerance binds, so
    // a triangle one division wide sits on the limit and the rounding of
    // parameters recovered by projection decides the comparison. The slack
    // keeps that rounding from demanding a subdivision the shared boundary
    // itself does not carry.
    const SLACK: f64 = 1.0 + 1e-9;
    if angular_tol > 0.0 && u_span > angular_tol * SLACK {
        return false;
    }
    radius * (1.0 - (u_span / 2.0).cos()) <= deflection * SLACK
}

/// Compute interior grid resolution for `tessellate_nonplanar_cdt`.
fn interior_grid_resolution(
    surface: &FaceSurface,
    du: f64,
    dv: f64,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
) -> (usize, usize) {
    // This is the non-standard-boundary CDT fallback (boolean-result faces).
    // Watertightness comes from the explicit boundary samples, not from these
    // interior grid counts, and one radius drives both directions. Doubly
    // curved surfaces keep the curvature floor unconditionally (the nominal
    // radius understates the tightest curvature); developable surfaces
    // thread the caller's `circle_floor` so the display/export path is
    // tolerance-driven while the boolean path stays bit-identical.
    match surface {
        FaceSurface::Sphere(sphere) => {
            let r = sphere.radius();
            let n_u = segments_for_chord_deviation_a(r, du, deflection, angular_tol, true).max(2);
            let n_v = segments_for_chord_deviation_a(r, dv, deflection, angular_tol, true).max(2);
            (n_u, n_v)
        }
        FaceSurface::Torus(torus) => {
            let n_u = segments_for_chord_deviation_a(
                torus.major_radius(),
                du,
                deflection,
                angular_tol,
                true,
            )
            .max(2);
            let n_v = segments_for_chord_deviation_a(
                torus.minor_radius(),
                dv,
                deflection,
                angular_tol,
                true,
            )
            .max(2);
            (n_u, n_v)
        }
        FaceSurface::Cylinder(_) | FaceSurface::Cone(_) => {
            // u is the periodic direction (radians): curvature-driven. v runs
            // along the straight rulings (a length, not an angle): zero chord
            // sag, so feeding it to the chord formula would treat millimeters
            // as radians and emit hundreds of interior rows on a tall wall.
            // Two rows suffice for CDT quality on a developable band. With
            // the curvature floor off (display/export), skip interior points
            // entirely: the surface is exact along the rulings and the rim
            // samples already carry the u density, so interior points only
            // inflate the mesh.
            if !circle_floor {
                return (2, 1);
            }
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation_a(r, du, deflection, angular_tol, true).max(2);
            (n_u, 2)
        }
        // A NURBS face's parameters are knot values, not angles: its grid
        // lines come from `nurbs_grid_lines`.
        FaceSurface::Nurbs(_) => (1, 1),
        FaceSurface::Plane { .. } => {
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation_a(r, du, deflection, angular_tol, true).max(2);
            let n_v = segments_for_chord_deviation_a(r, dv, deflection, angular_tol, true).max(2);
            (n_u, n_v)
        }
    }
}

/// The points `(xs[i], ys[j])` of a grid that lie on one of `segments`, by
/// the exact orientation test the triangulation locates a point with.
fn grid_points_on_segments(
    xs: &[f64],
    ys: &[f64],
    segments: &[(brepkit_math::vec::Point2, brepkit_math::vec::Point2)],
) -> DetHashSet<(usize, usize)> {
    use brepkit_math::vec::Point2;
    let sorted = |values: &[f64]| {
        let mut order: Vec<usize> = (0..values.len()).collect();
        order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
        order
    };
    let (x_order, y_order) = (sorted(xs), sorted(ys));
    let within = |values: &[f64], order: &[usize], lo: f64, hi: f64| {
        let first = order.partition_point(|&k| values[k] < lo);
        let last = order.partition_point(|&k| values[k] <= hi);
        order[first..last].to_vec()
    };
    let mut on = DetHashSet::default();
    for &(a, b) in segments {
        let columns = within(xs, &x_order, a.x().min(b.x()), a.x().max(b.x()));
        if columns.is_empty() {
            continue;
        }
        for j in within(ys, &y_order, a.y().min(b.y()), a.y().max(b.y())) {
            for &i in &columns {
                let point = Point2::new(xs[i], ys[j]);
                if brepkit_math::predicates::orient2d(a, b, point) == 0.0 {
                    on.insert((i, j));
                }
            }
        }
    }
    on
}

/// Check if a 2D point is inside a polygon defined by (u, v) coordinates.
/// Uses the winding number algorithm for robustness.
/// A polygon's edges a grid row crosses: each edge's first vertex, and
/// whether it runs upward.
type RowCrossings = Vec<(usize, bool)>;

/// The edges of `polygon` that [`point_in_polygon_2d`] counts for a point at
/// height `y`, each as the index of its first vertex and whether it runs
/// upward: a grid row's points share their height, so its crossings are
/// found once for the row.
fn row_crossings(polygon: &[(f64, f64)], y: f64) -> RowCrossings {
    let n = polygon.len();
    (0..n)
        .filter_map(|i| {
            let (yi, yj) = (polygon[i].1, polygon[(i + 1) % n].1);
            if yi <= y {
                (yj > y).then_some((i, true))
            } else {
                (yj <= y).then_some((i, false))
            }
        })
        .collect()
}

/// [`point_in_polygon_2d`] of `pt`, from the crossings [`row_crossings`]
/// found at its height: the same edges, each read as it reads them.
fn point_in_polygon_row(
    polygon: &[(f64, f64)],
    row: &[(usize, bool)],
    pt: brepkit_math::vec::Point2,
) -> bool {
    let n = polygon.len();
    let mut winding = 0i32;
    for &(i, up) in row {
        let j = (i + 1) % n;
        let yi = polygon[i].1;
        let yj = polygon[j].1;
        let cross =
            (polygon[j].0 - polygon[i].0) * (pt.y() - yi) - (pt.x() - polygon[i].0) * (yj - yi);
        if up {
            if cross > 0.0 {
                winding += 1;
            }
        } else if cross < 0.0 {
            winding -= 1;
        }
    }
    winding != 0
}

pub(super) fn point_in_polygon_2d(polygon: &[(f64, f64)], pt: brepkit_math::vec::Point2) -> bool {
    let n = polygon.len();
    let mut winding = 0i32;
    for i in 0..n {
        let j = (i + 1) % n;
        let yi = polygon[i].1;
        let yj = polygon[j].1;
        if yi <= pt.y() {
            if yj > pt.y() {
                let cross = (polygon[j].0 - polygon[i].0) * (pt.y() - yi)
                    - (pt.x() - polygon[i].0) * (yj - yi);
                if cross > 0.0 {
                    winding += 1;
                }
            }
        } else if yj <= pt.y() {
            let cross =
                (polygon[j].0 - polygon[i].0) * (pt.y() - yi) - (pt.x() - polygon[i].0) * (yj - yi);
            if cross < 0.0 {
                winding -= 1;
            }
        }
    }
    winding != 0
}

/// Snap-based fallback tessellation for non-planar faces.
#[allow(clippy::too_many_arguments)]
pub(super) fn tessellate_nonplanar_snap(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut PointPool,
) -> Result<(), crate::OperationsError> {
    let mut face_mesh = super::face::tessellate_with_uvs_floor(
        topo,
        face_id,
        deflection,
        angular_tol,
        circle_floor,
    )
    .map(|uv| uv.mesh)?;

    // `tessellate()` already applies the `is_reversed` flip. The caller
    // `tessellate_face_with_shared_edges` will apply its own flip, so undo
    // the one from `tessellate()` to avoid a double-flip.
    if face_data.is_reversed() {
        let tri_count = face_mesh.indices.len() / 3;
        for t in 0..tri_count {
            face_mesh.indices.swap(t * 3 + 1, t * 3 + 2);
        }
        for n in &mut face_mesh.normals {
            *n = -*n;
        }
    }

    let mut local_to_global: Vec<u32> = Vec::with_capacity(face_mesh.positions.len());

    let wire = topo.wire(face_data.outer_wire())?;
    let mut snap_targets: Vec<(Point3, u32)> = Vec::new();
    for oe in wire.edges() {
        if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
            for &gid in global_ids {
                if (gid as usize) < merged.positions.len() {
                    snap_targets.push((merged.positions[gid as usize], gid));
                }
            }
        }
    }
    for &inner_wire_id in face_data.inner_wires() {
        if let Ok(inner_wire) = topo.wire(inner_wire_id) {
            for oe in inner_wire.edges() {
                if let Some(global_ids) = edge_global_indices.get(&oe.edge().index()) {
                    for &gid in global_ids {
                        if (gid as usize) < merged.positions.len() {
                            snap_targets.push((merged.positions[gid as usize], gid));
                        }
                    }
                }
            }
        }
    }

    // Build spatial hash for O(1) snap lookups.
    let snap_tol = 1e-6;
    let inv_cell = 1.0 / snap_tol;
    let mut snap_grid: DetHashMap<(i64, i64, i64), Vec<u32>> =
        DetHashMap::with_capacity_and_hasher(snap_targets.len(), brepkit_math::det_hash::DetState);
    for &(target_pos, gid) in &snap_targets {
        let cx = (target_pos.x() * inv_cell).round() as i64;
        let cy = (target_pos.y() * inv_cell).round() as i64;
        let cz = (target_pos.z() * inv_cell).round() as i64;
        snap_grid.entry((cx, cy, cz)).or_default().push(gid);
    }

    for (i, &pos) in face_mesh.positions.iter().enumerate() {
        let cx = (pos.x() * inv_cell).round() as i64;
        let cy = (pos.y() * inv_cell).round() as i64;
        let cz = (pos.z() * inv_cell).round() as i64;
        let mut best_gid = None;
        let mut best_dist = snap_tol;
        // Check 3x3x3 neighborhood for snap matches.
        for dx in -1_i64..=1 {
            for dy in -1_i64..=1 {
                for dz in -1_i64..=1 {
                    if let Some(gids) = snap_grid.get(&(cx + dx, cy + dy, cz + dz)) {
                        for &gid in gids {
                            let target_pos = merged.positions[gid as usize];
                            let dist = (pos - target_pos).length();
                            if dist < best_dist {
                                best_dist = dist;
                                best_gid = Some(gid);
                            }
                        }
                    }
                }
            }
        }

        if let Some(gid) = best_gid {
            local_to_global.push(gid);
        } else {
            let key = point_merge_key(pos, MERGE_GRID);
            let gid = point_to_global.entry(key).or_insert_with(|| {
                let idx = merged.positions.len() as u32;
                merged.positions.push(pos);
                merged.normals.push(
                    face_mesh
                        .normals
                        .get(i)
                        .copied()
                        .unwrap_or(Vec3::new(0.0, 0.0, 1.0)),
                );
                idx
            });
            local_to_global.push(*gid);
        }
    }

    for &li in &face_mesh.indices {
        merged.indices.push(local_to_global[li as usize]);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]

    use super::*;

    /// An octant whose sides hold 92,682 segments keeps a segment for each
    /// level on every row: its rows take 4,295,115,586 samples, past what a
    /// 32-bit `usize` holds. Summed in a 32-bit `usize`, they wrap to 148,290
    /// and pass the limit; summed here, they are refused.
    #[test]
    fn an_octant_past_a_32_bit_count_is_refused() {
        let m = 92_682_usize;
        let counts: Vec<usize> = (1..m).collect();
        let samples = triangle_row_samples(m + 1, &counts);
        assert_eq!(samples, 4_295_115_586);
        assert_eq!(samples % (1 << 32), 148_290);
        assert!(matches!(
            super::super::check_total("a face's rows", samples, 1.0),
            Err(crate::OperationsError::TooManySamples { .. })
        ));
    }

    /// A grid row's crossings give every point of the row the answer the
    /// whole polygon does, points level with vertices and on edges included.
    #[test]
    fn row_crossings_answer_as_the_whole_polygon() {
        use brepkit_math::vec::Point2;
        // A star with vertices at the grid's own heights and a horizontal
        // edge, walked one way and then the other.
        let mut star: Vec<(f64, f64)> = (0..14)
            .map(|k| {
                let a = f64::from(k) * std::f64::consts::TAU / 14.0;
                let r = if k % 2 == 0 { 1.0 } else { 0.45 };
                (
                    (r * a.cos() * 8.0).round() / 8.0,
                    (r * a.sin() * 8.0).round() / 8.0,
                )
            })
            .collect();
        star.insert(3, (0.5, star[2].1));
        for polygon in [star.clone(), star.into_iter().rev().collect()] {
            for iv in -10..=10 {
                let y = f64::from(iv) / 8.0;
                let row = row_crossings(&polygon, y);
                for iu in -100..=100 {
                    let pt = Point2::new(f64::from(iu) / 80.0, y);
                    assert_eq!(
                        point_in_polygon_row(&polygon, &row, pt),
                        point_in_polygon_2d(&polygon, pt),
                        "{pt:?}"
                    );
                }
            }
        }
    }

    /// Grid points on a segment along a grid line, or on a slanted one
    /// through grid points, are found; the others are not.
    #[test]
    fn grid_points_on_a_segment_are_found_exactly() {
        use brepkit_math::vec::Point2;
        let xs = [0.5, 1.0, 1.5, 2.0];
        let ys = [1.0, 0.25, 0.5, 0.75];
        let along = (Point2::new(0.75, 0.5), Point2::new(2.0, 0.5));
        let slanted = (Point2::new(0.0, 0.0), Point2::new(1.0, 1.0));
        let near = (
            Point2::new(0.0, 0.25 + 1e-12),
            Point2::new(3.0, 0.25 + 1e-12),
        );
        let on = grid_points_on_segments(&xs, &ys, &[along, slanted, near]);
        let mut found: Vec<(usize, usize)> = on.into_iter().collect();
        found.sort_unstable();
        // Along y = 0.5 from x = 0.75: x = 1, 1.5 and 2. On the slant: (0.5,
        // 0.5) and (1, 1). Beside y = 0.25 by 1e-12: none.
        assert_eq!(found, vec![(0, 2), (1, 0), (1, 2), (2, 2), (3, 2)]);
    }

    #[test]
    fn a_seam_beside_coinciding_samples_takes_bounded_spans() {
        assert_eq!(seam_spans(10.0, 4.0), 3);
        assert_eq!(seam_spans(1e-9, 4.0), 2);
        assert_eq!(seam_spans(10.0, 0.0), MAX_PATH_SPANS);
        assert_eq!(seam_spans(10.0, f64::NAN), MAX_PATH_SPANS);
        assert_eq!(seam_spans(f64::INFINITY, 1.0), MAX_PATH_SPANS);
        assert_eq!(seam_spans(f64::NAN, 1.0), 2);
    }

    /// A path starts from at most [`MAX_PATH_SPANS`] spans, and one that no
    /// sampling brings within the deflection, so wild no span ever sits
    /// within it, is refused, not cut short of it.
    #[test]
    fn a_path_no_sampling_satisfies_is_refused() {
        let started = std::time::Instant::now();
        // Straight but started from as many spans as a degenerate seam could
        // ask for.
        let straight =
            surface_path_parameters(&|t| Point3::new(t, 0.0, 0.0), 1e-3, 0.35, usize::MAX).unwrap();
        assert_eq!(straight.len(), MAX_PATH_SPANS - 1);
        let wild = surface_path_parameters(
            &|t| Point3::new(t, (1e9 * t).sin(), (3e9 * t).cos()),
            1e-12,
            1e-6,
            MAX_PATH_SPANS,
        );
        assert!(
            matches!(wild, Err(crate::OperationsError::TooManySamples { .. })),
            "{wild:?}"
        );
        // A path that is not finite reads as straight.
        let unknown =
            surface_path_parameters(&|_| Point3::new(f64::NAN, 0.0, 0.0), 1e-3, 0.35, 1).unwrap();
        assert!(unknown.len() < MAX_PATH_SPANS);
        assert!(started.elapsed() < std::time::Duration::from_secs(30));
    }

    #[test]
    fn a_creased_path_is_not_halved_far_below_its_deflection() {
        // A degree-one profile's corner turns as sharply however finely the
        // span across it is halved, and sits off every dyadic parameter.
        let creased = |t: f64| {
            if t <= 1.0 / 3.0 {
                Point3::new(3.0 * t, 0.0, 0.0)
            } else {
                Point3::new(1.0, 1.5 * (t - 1.0 / 3.0), 0.0)
            }
        };
        let deflection = 0.01;
        let along = surface_path_parameters(&creased, deflection, 0.35, 1).unwrap();
        let ts: Vec<f64> = std::iter::once(0.0)
            .chain(along)
            .chain(std::iter::once(1.0))
            .collect();
        for w in ts.windows(2) {
            let (a, b) = (creased(w[0]), creased(w[1]));
            let chord = (b - a).length();
            assert!(chord > deflection / 8.0, "span {w:?} halved to {chord}");
            for k in 1..8 {
                let p = creased(w[0] + (w[1] - w[0]) * f64::from(k) / 8.0);
                let f = ((p - a).dot(b - a) / chord.powi(2)).clamp(0.0, 1.0);
                let off = (p - (a + (b - a) * f)).length();
                assert!(off <= deflection, "span {w:?} stands {off} off its chord");
            }
        }
    }

    /// A square whose loop touches itself at a vertex of its bottom side, to
    /// run a triangular hole round from it. Wherever the loop starts, the
    /// hole is the one lobe that runs the other way round.
    #[test]
    fn a_pinched_hole_is_found_wherever_the_loop_starts() {
        use brepkit_math::vec::Point2;
        let p = |x: f64, y: f64| Point2::new(x, y);
        let points = [
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.5, 1.0),
            p(2.5, 1.0),
            p(2.0, 0.0),
            p(4.0, 0.0),
            p(4.0, 4.0),
            p(0.0, 4.0),
        ];
        let ids = [0, 1, 2, 3, 1, 4, 5, 6];
        for start in 0..points.len() {
            let turned_points: Vec<Point2> = (0..points.len())
                .map(|k| points[(start + k) % points.len()])
                .collect();
            let turned_ids: Vec<usize> = (0..ids.len())
                .map(|k| ids[(start + k) % ids.len()])
                .collect();
            let (lobe_pts, ranges) = pinched_hole_lobes(&turned_ids, &turned_points);
            assert_eq!(ranges.len(), 1, "start {start}: lobes {ranges:?}");
            let (a, b) = ranges[0];
            assert_eq!(b - a, 3, "start {start}: the hole has 3 corners");
            let corners = &lobe_pts[a..b];
            for c in [p(2.0, 0.0), p(1.5, 1.0), p(2.5, 1.0)] {
                assert!(
                    corners.iter().any(|q| (*q - c).length() < 1e-12),
                    "start {start}: corner {c:?} missing from {corners:?}"
                );
            }
        }
    }

    /// Two triangular holes that touch the outline at one vertex: wherever
    /// the loop starts, each is a hole of its own.
    #[test]
    fn two_holes_pinched_at_one_vertex_are_two_holes() {
        use brepkit_math::vec::Point2;
        let p = |x: f64, y: f64| Point2::new(x, y);
        let points = [
            p(0.0, 0.0),
            p(2.0, 0.0),
            p(1.0, 1.0),
            p(1.8, 1.0),
            p(2.0, 0.0),
            p(2.2, 1.0),
            p(3.0, 1.0),
            p(2.0, 0.0),
            p(4.0, 0.0),
            p(4.0, 4.0),
            p(0.0, 4.0),
        ];
        let ids = [0, 1, 2, 3, 1, 4, 5, 1, 6, 7, 8];
        for start in 0..points.len() {
            let turned_points: Vec<Point2> = (0..points.len())
                .map(|k| points[(start + k) % points.len()])
                .collect();
            let turned_ids: Vec<usize> = (0..ids.len())
                .map(|k| ids[(start + k) % ids.len()])
                .collect();
            let (_, ranges) = pinched_hole_lobes(&turned_ids, &turned_points);
            assert_eq!(ranges.len(), 2, "start {start}: lobes {ranges:?}");
            for &(a, b) in &ranges {
                assert_eq!(b - a, 3, "start {start}: a hole has 3 corners, {ranges:?}");
            }
        }
    }
}

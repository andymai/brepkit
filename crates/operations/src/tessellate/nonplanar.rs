//! Non-planar CDT and fallback paths for face tessellation.

use brepkit_math::det_hash::{DetHashMap, DetHashSet};
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::edge_sampling::{sample_edge, segments_for_chord_deviation_a};
use super::{MERGE_GRID, TriangleMesh, point_merge_key};

/// Maps a 3D point to its `(u, v)` surface parameters.
type ProjectFn = Box<dyn Fn(Point3) -> (f64, f64)>;
/// Maps `(u, v)` surface parameters to a 3D surface point.
type EvalFn = Box<dyn Fn(f64, f64) -> Point3>;
/// Maps `(u, v)` surface parameters to the outward surface normal.
type NormalFn = Box<dyn Fn(f64, f64) -> Vec3>;

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
/// Returns `Ok(true)` when the face is a simple two-rim band that was handled
/// here, `Ok(false)` when it is not (the caller then falls back to the snap or
/// CDT path). A "simple band" has no inner wires, exactly two **closed**
/// rim-circle edges (everything else a seam line), and matching shared-vertex
/// counts on the two rims.
pub(super) fn tessellate_revolution_band_shared(
    topo: &Topology,
    face_data: &brepkit_topology::face::Face,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
) -> Result<bool, crate::OperationsError> {
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }

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

    // Collect the two closed rim-circle edges; everything else must be a seam line.
    let wire = topo.wire(face_data.outer_wire())?;
    let mut rim_edge_ids: Vec<usize> = Vec::new();
    for oe in wire.edges() {
        let e = topo.edge(oe.edge())?;
        let closed = e.start() == e.end();
        match e.curve() {
            // Only closed circles are rims here. The caller gates this path on
            // `is_standard_rect` (Line | Circle edges only), so ellipse rims
            // never reach it — they take the CDT path instead.
            EdgeCurve::Circle(_) if closed => {
                let idx = oe.edge().index();
                if !rim_edge_ids.contains(&idx) {
                    rim_edge_ids.push(idx);
                }
            }
            EdgeCurve::Line => {}
            // An open arc rim, a NURBS boundary, or an open circle is not a
            // simple full-revolution band — let the caller handle it.
            _ => return Ok(false),
        }
    }
    if rim_edge_ids.len() != 2 {
        return Ok(false);
    }

    // Pull each rim's shared global vertex IDs, drop the closing duplicate, and
    // require matching counts so the rings connect index-for-index.
    let mut rims: Vec<Vec<u32>> = Vec::with_capacity(2);
    for &re in &rim_edge_ids {
        let Some(ids) = edge_global_indices.get(&re) else {
            return Ok(false);
        };
        let mut ids = ids.clone();
        if ids.len() > 1 && ids.first() == ids.last() {
            ids.pop();
        }
        if ids.len() < 3 {
            return Ok(false);
        }
        rims.push(ids);
    }
    if rims[0].len() != rims[1].len() {
        return Ok(false);
    }
    let n = rims[0].len();

    // Sort each rim by angle around the axis so the two rings align by index.
    let angle_of = |gid: u32, merged: &TriangleMesh| project(merged.positions[gid as usize]).0;
    for rim in &mut rims {
        rim.sort_by(|&a, &b| {
            angle_of(a, merged)
                .partial_cmp(&angle_of(b, merged))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // Emit default-oriented (non-reversed) triangles: the geometric normal
    // matches the surface outward normal, the convention `tessellate_analytic`
    // uses. The caller (`tessellate_face_with_shared_edges`) applies the global
    // `is_reversed` winding flip afterward, so we must NOT apply it here.
    let emit = |merged: &mut TriangleMesh, a: u32, b: u32, c: u32| {
        let (pa, pb, pc) = (
            merged.positions[a as usize],
            merged.positions[b as usize],
            merged.positions[c as usize],
        );
        // Skip degenerate triangles (two rim points at the same position).
        let geo = (pb - pa).cross(pc - pa);
        if geo.length() < 1e-20 {
            return;
        }
        let (u, v) = project(pa);
        let outward = surf_normal(u, v);
        let mut tri = [a, b, c];
        if geo.dot(outward) < 0.0 {
            tri.swap(1, 2);
        }
        merged.indices.extend_from_slice(&tri);
    };

    for i in 0..n {
        let j = (i + 1) % n;
        let (b0, b1) = (rims[0][i], rims[0][j]);
        let (t0, t1) = (rims[1][i], rims[1][j]);
        emit(merged, b0, b1, t1);
        emit(merged, b0, t1, t0);
    }

    Ok(true)
}

/// Tessellate a torus band bounded by two closed rim circles and seamed by ONE
/// doubled open arc edge, in either orientation:
///   * constant-`v` rims (latitude circles wrapping the ring angle `u`) — a
///     full analytic revolve of a profile arc, seamed by that arc; interior
///     full-`u` rows are swept along the tube angle;
///   * constant-`u` rims (tube circles wrapping `v`) — a PARTIAL-turn revolve
///     of a full circle profile, seamed by the vertex's sweep arc; interior
///     full-`v` rings are swept along the ring angle.
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
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
) -> Result<bool, crate::OperationsError> {
    use std::f64::consts::TAU;
    let FaceSurface::Torus(torus) = face_data.surface() else {
        return Ok(false);
    };
    if !face_data.inner_wires().is_empty() {
        return Ok(false);
    }

    let wire = topo.wire(face_data.outer_wire())?;
    let mut rim_edge_ids: Vec<usize> = Vec::new();
    let mut seam: Option<(brepkit_topology::edge::EdgeId, usize)> = None;
    for oe in wire.edges() {
        let e = topo.edge(oe.edge())?;
        let closed = e.start() == e.end();
        match e.curve() {
            EdgeCurve::Circle(_) if closed => {
                let idx = oe.edge().index();
                if !rim_edge_ids.contains(&idx) {
                    rim_edge_ids.push(idx);
                }
            }
            EdgeCurve::Circle(_) => match &mut seam {
                None => seam = Some((oe.edge(), 1)),
                Some((eid, uses)) if *eid == oe.edge() => *uses += 1,
                Some(_) => return Ok(false),
            },
            _ => return Ok(false),
        }
    }
    let Some((seam_eid, 2)) = seam else {
        return Ok(false);
    };
    if rim_edge_ids.len() != 2 {
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
    for &re in &rim_edge_ids {
        let Some(gids) = edge_global_indices.get(&re) else {
            return Ok(false);
        };
        let mut seen: DetHashSet<u32> = DetHashSet::default();
        let mut pts: Vec<(f64, f64, u32)> = Vec::with_capacity(gids.len());
        for &g in gids {
            if !seen.insert(g) {
                continue;
            }
            let (u, v) = project(merged.positions[g as usize]);
            pts.push((u, v, g));
        }
        if pts.len() < 3 {
            return Ok(false);
        }
        raw.push(pts);
    }

    // Both rims must be constant in the SAME parameter: constant-v (latitude
    // rims, swept along the tube angle) or constant-u (tube rims, swept along
    // the ring angle).
    let spread_of = |pts: &[(f64, f64, u32)], pick_u: bool| -> (f64, f64) {
        let angles: Vec<f64> = pts
            .iter()
            .map(|&(u, v, _)| if pick_u { u } else { v })
            .collect();
        circ_mean_spread(&angles)
    };
    let (u_stats0, v_stats0) = (spread_of(&raw[0], true), spread_of(&raw[0], false));
    let (u_stats1, v_stats1) = (spread_of(&raw[1], true), spread_of(&raw[1], false));
    let lat_mode = if v_stats0.1 <= 1e-6 && v_stats1.1 <= 1e-6 {
        true
    } else if u_stats0.1 <= 1e-6 && u_stats1.1 <= 1e-6 {
        false
    } else {
        return Ok(false);
    };
    let (lvl0, lvl1) = if lat_mode {
        (v_stats0.0, v_stats1.0)
    } else {
        (u_stats0.0, u_stats1.0)
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
    let n_rows =
        segments_for_chord_deviation_a(sweep_radius, sweep.abs(), deflection, angular_tol, true)
            .max(1);
    let full_circle_cols =
        segments_for_chord_deviation_a(wrap_radius, TAU, deflection, angular_tol, true);
    let n_cols = rims[0].len().max(rims[1].len()).max(full_circle_cols);

    let emit = make_band_emit(&project, &surf_normal);
    let mut prev_ring: LatRing = rims[0].clone();
    for i in 1..n_rows {
        #[allow(clippy::cast_precision_loss)]
        let t = i as f64 / n_rows as f64;
        let level = lvl0 + sweep * t;
        let mut row: LatRing = Vec::with_capacity(n_cols);
        for j in 0..n_cols {
            #[allow(clippy::cast_precision_loss)]
            let a = TAU * (j as f64) / (n_cols as f64);
            let (u, v) = if lat_mode { (a, level) } else { (level, a) };
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
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
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

    // Build interior rings as `LatRing` (sorted by v) of fresh vertices.
    let build_u_ring = |u: f64,
                        merged: &mut TriangleMesh,
                        point_to_global: &mut DetHashMap<(i64, i64, i64), u32>|
     -> LatRing {
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
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
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

    // Case 2 — a COLLAR: the inner wire is a constant-v cap circle, the outer
    // wire is a full-longitude-wrap "floor" at varying v (great-circle/seam
    // arcs, e.g. a box ∩ sphere patch). Sweep interior rows whose per-column v
    // interpolates from the scalloped floor up to the cap.
    let Some((v_cap, cap_ring)) = inner_const else {
        return Ok(false);
    };
    let Some(floor) = collect_var_v_ring(
        topo,
        outer_wid,
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
/// vary with longitude.
fn collect_var_v_ring(
    topo: &Topology,
    wire_id: brepkit_topology::wire::WireId,
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
        gids.extend_from_slice(edge_gids);
    }
    if gids.len() < 3 {
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
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
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
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
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

/// CDT-based tessellation for non-planar faces with exact boundary constraints.
///
/// Projects shared edge points into (u,v) parameter space, generates interior
/// sample points, then runs Constrained Delaunay Triangulation. Boundary
/// vertices use their pre-existing global IDs (watertight by construction).
#[allow(clippy::too_many_lines, clippy::too_many_arguments)]
pub(super) fn tessellate_nonplanar_cdt(
    topo: &Topology,
    face_id: FaceId,
    face_data: &brepkit_topology::face::Face,
    deflection: f64,
    angular_tol: f64,
    circle_floor: bool,
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    use brepkit_math::cdt::Cdt;
    use brepkit_math::vec::Point2;
    use brepkit_topology::edge::EdgeId;

    let wire = topo.wire(face_data.outer_wire())?;
    let tol_dup = 1e-10;

    // Fourth element: is_forward flag -- needed for seam UV assignment.
    let mut boundary_3d: Vec<(Point3, u32, EdgeId, bool)> = Vec::new();
    for oe in wire.edges() {
        let edge_id_local = oe.edge();
        let edge_idx = edge_id_local.index();
        let is_fwd = oe.is_forward();
        if let Some(global_ids) = edge_global_indices.get(&edge_idx) {
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

    let n_boundary = boundary_3d.len();
    if n_boundary < 3 {
        return Err(crate::OperationsError::InvalidInput {
            reason: "non-planar face has fewer than 3 boundary vertices".to_string(),
        });
    }

    let mut boundary_uv: Vec<(f64, f64)> = boundary_3d
        .iter()
        .map(|(pt, _, edge_id_local, _)| {
            if let Some(pcurve) = topo.pcurves().get(*edge_id_local, face_id) {
                let uv = project_via_pcurve(pcurve, *pt, face_data.surface());
                if let Some(uv) = uv {
                    return Ok(uv);
                }
            }
            project_to_surface_uv(face_data.surface(), *pt)
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Step 2a: Unwrap periodic u across the seam for polyline boundaries.
    {
        let is_periodic = matches!(
            face_data.surface(),
            FaceSurface::Cylinder(_)
                | FaceSurface::Cone(_)
                | FaceSurface::Sphere(_)
                | FaceSurface::Torus(_)
        );
        if is_periodic && !boundary_uv.is_empty() {
            for i in 1..boundary_uv.len() {
                let prev_u = boundary_uv[i - 1].0;
                let mut u = boundary_uv[i].0;
                let diff = u - prev_u;
                let shifts = (diff / std::f64::consts::TAU + 0.5).floor();
                u -= shifts * std::f64::consts::TAU;
                boundary_uv[i].0 = u;
            }
            let first_u = boundary_uv[0].0;
            let last_u = boundary_uv.last().map_or(first_u, |p| p.0);
            let close_diff = first_u - last_u;
            if close_diff.abs() > std::f64::consts::PI {
                let u_mid = boundary_uv.iter().map(|p| p.0).sum::<f64>() / boundary_uv.len() as f64;
                let target_mid = std::f64::consts::PI;
                let shift = target_mid - u_mid;
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
        if matches!(face_data.surface(), FaceSurface::Torus(_)) && !boundary_uv.is_empty() {
            for i in 1..boundary_uv.len() {
                let prev_v = boundary_uv[i - 1].1;
                let mut v = boundary_uv[i].1;
                let diff = v - prev_v;
                let shifts = (diff / std::f64::consts::TAU + 0.5).floor();
                v -= shifts * std::f64::consts::TAU;
                boundary_uv[i].1 = v;
            }
        }
    }

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

        if !seam_edge_indices.is_empty() {
            let non_seam_uvs: Vec<(f64, f64)> = boundary_uv
                .iter()
                .enumerate()
                .filter(|(i, _)| !seam_edge_indices.contains(&boundary_3d[*i].2.index()))
                .map(|(_, &uv)| uv)
                .collect();
            let (u_min_bnd, u_max_bnd, v_min_bnd, v_max_bnd) = if non_seam_uvs.is_empty() {
                (u_min, u_max, v_min, v_max)
            } else {
                uv_bounds(&non_seam_uvs)
            };

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

            for run in &seam_runs {
                let u_assign = if run.is_forward { u_max_bnd } else { u_min_bnd };
                let n_pts = run.indices.len();

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
        }

        // Recompute UV bounding box after seam fix.
        uv_bounds(&boundary_uv)
    };

    let margin = 0.01;
    let bounds = (
        Point2::new(u_min - margin, v_min - margin),
        Point2::new(u_max + margin, v_max + margin),
    );
    let mut cdt = Cdt::with_capacity(bounds, n_boundary);

    let mut cdt_to_global: Vec<Option<u32>> = vec![None; 3]; // 3 super-triangle verts

    let boundary_pts: Vec<Point2> = boundary_uv
        .iter()
        .map(|&(u, v)| Point2::new(u, v))
        .collect();
    let boundary_cdt_ids = cdt
        .insert_points_hilbert(&boundary_pts)
        .map_err(crate::OperationsError::Math)?;
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

    let du = u_max - u_min;
    let dv = v_max - v_min;
    if du > 1e-15 && dv > 1e-15 {
        let (n_u, n_v) =
            interior_grid_resolution(face_data.surface(), du, dv, deflection, angular_tol);

        let boundary_uv_ref = &boundary_uv;
        let interior_pts: Vec<Point2> = (1..n_u)
            .flat_map(|iu| {
                (1..n_v).filter_map(move |iv| {
                    let u = u_min + du * (iu as f64 / n_u as f64);
                    let v = v_min + dv * (iv as f64 / n_v as f64);
                    let pt2 = Point2::new(u, v);
                    point_in_polygon_2d(boundary_uv_ref, pt2).then_some(pt2)
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

    let boundary_pairs: Vec<(usize, usize)> = (0..n_boundary)
        .map(|i| (boundary_cdt_ids[i], boundary_cdt_ids[(i + 1) % n_boundary]))
        .collect();
    cdt.remove_exterior(&boundary_pairs);

    let cdt_verts = cdt.vertices();
    let triangles = cdt.triangles();

    let mut final_global_ids: Vec<u32> = vec![0; cdt_to_global.len()];

    for i in 0..cdt_to_global.len() {
        if let Some(gid) = cdt_to_global[i] {
            final_global_ids[i] = gid;
        } else if i >= 3 {
            let pt2 = cdt_verts[i];
            let surface = face_data.surface();
            let pt3 = eval_surface_point(surface, pt2.x(), pt2.y());
            let nrm = surface.normal(pt2.x(), pt2.y());

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

    for (i0, i1, i2) in triangles {
        if i0 < 3 || i1 < 3 || i2 < 3 {
            continue; // Skip super-triangle vertices
        }
        merged.indices.push(final_global_ids[i0]);
        merged.indices.push(final_global_ids[i1]);
        merged.indices.push(final_global_ids[i2]);
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
        FaceSurface::Nurbs(surface) => {
            brepkit_math::nurbs::projection::project_point_to_surface(surface, pt, 1e-6)
                .map(|proj| (proj.u, proj.v))
                .map_err(crate::OperationsError::Math)
        }
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

/// Evaluate a non-planar surface at `(u, v)` and return a 3D point.
fn eval_surface_point(surface: &FaceSurface, u: f64, v: f64) -> Point3 {
    surface.evaluate(u, v).unwrap_or(Point3::new(0.0, 0.0, 0.0))
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

/// Compute interior grid resolution for `tessellate_nonplanar_cdt`.
fn interior_grid_resolution(
    surface: &FaceSurface,
    du: f64,
    dv: f64,
    deflection: f64,
    angular_tol: f64,
) -> (usize, usize) {
    // This is the non-standard-boundary CDT fallback (boolean-result faces).
    // Watertightness comes from the explicit boundary samples, not from these
    // interior grid counts, and one radius drives both directions; keep the
    // curvature floor on for all surfaces here as the conservative default.
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
            // Two rows suffice for CDT quality on a developable band.
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation_a(r, du, deflection, angular_tol, true).max(2);
            (n_u, 2)
        }
        FaceSurface::Plane { .. } | FaceSurface::Nurbs(_) => {
            let r = estimate_surface_radius(surface);
            let n_u = segments_for_chord_deviation_a(r, du, deflection, angular_tol, true).max(2);
            let n_v = segments_for_chord_deviation_a(r, dv, deflection, angular_tol, true).max(2);
            (n_u, n_v)
        }
    }
}

/// Check if a 2D point is inside a polygon defined by (u, v) coordinates.
/// Uses the winding number algorithm for robustness.
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
    edge_global_indices: &DetHashMap<usize, Vec<u32>>,
    merged: &mut TriangleMesh,
    point_to_global: &mut DetHashMap<(i64, i64, i64), u32>,
) -> Result<(), crate::OperationsError> {
    let mut face_mesh = super::tessellate_with_tolerance(topo, face_id, deflection, angular_tol)?;

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

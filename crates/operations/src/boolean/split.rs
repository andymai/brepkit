//! Phase 3: Face splitting by intersection chords.
//!
//! Splits face polygons along intersection chords, producing fragments for
//! classification. Supports both iterative (small chord count) and CDT-based
//! batch (large chord count) approaches.

use std::collections::HashMap;

use brepkit_math::predicates::orient3d_sos;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point2, Point3, Vec3};
use brepkit_topology::face::FaceId;

use super::types::{CDT_CHORD_THRESHOLD, CDT_SNAP_FACTOR, FaceFragment, Source};

/// Split a face polygon along intersection chords, producing fragments.
///
/// For faces with many chords (≥ `CDT_CHORD_THRESHOLD`), uses CDT-based batch
/// splitting which is O(N log N). For faces with fewer chords, uses the simpler
/// iterative approach. Falls back to iterative on any CDT error.
pub(super) fn split_face(
    fid: FaceId,
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    chord_map: &HashMap<usize, Vec<(Point3, Point3)>>,
    tol: Tolerance,
) -> Vec<FaceFragment> {
    let Some(chords) = chord_map.get(&fid.index()).filter(|c| !c.is_empty()) else {
        return vec![FaceFragment {
            vertices: verts.to_vec(),
            normal,
            d,
            source,
        }];
    };

    if chords.len() >= CDT_CHORD_THRESHOLD {
        if let Ok(regions) = split_face_cdt_inner(verts, normal, d, chords, tol) {
            return regions
                .into_iter()
                .filter(|v| polygon_area_2x(v, &normal) > tol.linear * tol.linear)
                .map(|vertices| FaceFragment {
                    vertices,
                    normal,
                    d,
                    source,
                })
                .collect();
        }
    }

    split_face_iterative(verts, normal, d, source, chords, tol)
}

/// Iterative face splitting: apply each chord sequentially.
///
/// Simple and correct for small chord counts. For N chords, each chord splits
/// all existing fragments, so total work is O(N · F) where F grows with each
/// split.
pub(super) fn split_face_iterative(
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    source: Source,
    chords: &[(Point3, Point3)],
    tol: Tolerance,
) -> Vec<FaceFragment> {
    let mut frags: Vec<Vec<Point3>> = vec![verts.to_vec()];

    for &(c0, c1) in chords {
        let mut new_frags = Vec::new();
        for poly in &frags {
            let (left, right) = split_polygon_by_chord(poly, c0, c1, &normal);
            if left.len() >= 3 {
                new_frags.push(left);
            }
            if right.len() >= 3 {
                new_frags.push(right);
            }
        }
        if !new_frags.is_empty() {
            frags = new_frags;
        }
    }

    frags
        .into_iter()
        .filter(|v| polygon_area_2x(v, &normal) > tol.linear * tol.linear)
        .map(|vertices| FaceFragment {
            vertices,
            normal,
            d,
            source,
        })
        .collect()
}

/// CDT-based batch face splitting — O(N log N) for N chords.
///
/// Each chord defines a splitting LINE (not a finite segment). The chord
/// is extended to the polygon boundary before CDT insertion, matching the
/// semantics of the iterative `split_polygon_by_chord` which classifies
/// vertices relative to the chord plane.
///
/// After extension, chord-chord crossings are computed and the constraints
/// are split at intersection points for safe CDT insertion.
#[allow(clippy::too_many_lines)]
pub(super) fn split_face_cdt_inner(
    verts: &[Point3],
    normal: Vec3,
    d: f64,
    chords: &[(Point3, Point3)],
    tol: Tolerance,
) -> Result<Vec<Vec<Point3>>, crate::OperationsError> {
    use brepkit_math::cdt::Cdt;

    let nv = verts.len();
    if nv < 3 {
        return Ok(vec![verts.to_vec()]);
    }

    // --- Projection: drop the dominant normal axis ---
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let project = |p: Point3| -> Point2 {
        if az >= ax && az >= ay {
            Point2::new(p.x(), p.y())
        } else if ay >= ax {
            Point2::new(p.x(), p.z())
        } else {
            Point2::new(p.y(), p.z())
        }
    };

    let unproject = |p2: Point2| -> Point3 {
        if az >= ax && az >= ay {
            let z = (d - normal.x() * p2.x() - normal.y() * p2.y()) / normal.z();
            Point3::new(p2.x(), p2.y(), z)
        } else if ay >= ax {
            let y = (d - normal.x() * p2.x() - normal.z() * p2.y()) / normal.y();
            Point3::new(p2.x(), y, p2.y())
        } else {
            let x = (d - normal.y() * p2.x() - normal.z() * p2.y()) / normal.x();
            Point3::new(x, p2.x(), p2.y())
        }
    };

    let poly_2d: Vec<Point2> = verts.iter().map(|v| project(*v)).collect();

    // --- Extend each chord LINE to the polygon boundary ---
    // The iterative approach treats chords as infinite splitting planes.
    // We match this by extending each chord segment to the polygon boundary.
    let mut chords_2d: Vec<(Point2, Point2)> = Vec::with_capacity(chords.len());
    for &(c0, c1) in chords {
        let p0 = project(c0);
        let p1 = project(c1);
        let extended = extend_chord_to_polygon(p0, p1, &poly_2d);
        chords_2d.push(extended);
    }

    // Normalize chord direction (c0 < c1 lexicographically) so that
    // two chords defining the same line but in opposite directions dedup.
    // 1e-12: coordinate comparison tolerance — below f64 ULP for typical
    // model coordinates (~1e3), so this is effectively an exact equality
    // test that avoids NaN/rounding-induced flip-flop.
    for chord in &mut chords_2d {
        let (c0, c1) = *chord;
        if c0.x() > c1.x() + 1e-12 || ((c0.x() - c1.x()).abs() < 1e-12 && c0.y() > c1.y() + 1e-12) {
            *chord = (c1, c0);
        }
    }

    // Deduplicate identical chords (same line, same boundary endpoints).
    chords_2d.sort_by(|a, b| {
        a.0.x()
            .partial_cmp(&b.0.x())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                a.0.y()
                    .partial_cmp(&b.0.y())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
            .then(
                a.1.x()
                    .partial_cmp(&b.1.x())
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    chords_2d.dedup_by(|a, b| {
        let d0 = (a.0.x() - b.0.x()).powi(2) + (a.0.y() - b.0.y()).powi(2);
        let d1 = (a.1.x() - b.1.x()).powi(2) + (a.1.y() - b.1.y()).powi(2);
        d0 < tol.linear * tol.linear && d1 < tol.linear * tol.linear
    });

    // --- Compute chord-chord intersections (on extended segments) ---
    let mut chord_cross: Vec<Vec<(f64, Point2)>> = vec![Vec::new(); chords_2d.len()];

    for i in 0..chords_2d.len() {
        for j in (i + 1)..chords_2d.len() {
            if let Some((ti, tj, pt)) = seg_seg_cross_2d(
                chords_2d[i].0,
                chords_2d[i].1,
                chords_2d[j].0,
                chords_2d[j].1,
            ) {
                chord_cross[i].push((ti, pt));
                chord_cross[j].push((tj, pt));
            }
        }
    }

    for pts in &mut chord_cross {
        pts.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    }

    // --- Compute CDT bounds ---
    let (mut min_x, mut min_y) = (f64::INFINITY, f64::INFINITY);
    let (mut max_x, mut max_y) = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for &p in &poly_2d {
        min_x = min_x.min(p.x());
        min_y = min_y.min(p.y());
        max_x = max_x.max(p.x());
        max_y = max_y.max(p.y());
    }
    let bounds = (Point2::new(min_x, min_y), Point2::new(max_x, max_y));

    let n_cross: usize = chord_cross.iter().map(Vec::len).sum();
    let n_pts = nv + chords_2d.len() * 2 + n_cross;
    let mut cdt = Cdt::with_capacity(bounds, n_pts);

    // --- Insert polygon vertices ---
    let mut poly_vidxs: Vec<usize> = Vec::with_capacity(nv);
    for &pt in &poly_2d {
        poly_vidxs.push(cdt.insert_point(pt)?);
    }

    // --- Insert chord endpoints (on polygon boundary after extension) ---
    let mut chord_vidxs: Vec<(usize, usize)> = Vec::with_capacity(chords_2d.len());
    for &(c0, c1) in &chords_2d {
        let v0 = cdt.insert_point(c0)?;
        let v1 = cdt.insert_point(c1)?;
        chord_vidxs.push((v0, v1));
    }

    // --- Insert chord-chord intersection points and build per-chord splits ---
    // Track the CDT index assigned to each crossing point so we can
    // construct chord sub-segments directly (avoiding the expensive O(V*C)
    // scan of all CDT vertices).
    let mut chord_splits: Vec<Vec<(f64, usize)>> = (0..chords_2d.len())
        .map(|i| {
            let (v0, v1) = chord_vidxs[i];
            vec![(0.0, v0), (1.0, v1)]
        })
        .collect();

    for (chord_idx, pts) in chord_cross.iter().enumerate() {
        for &(t, pt) in pts {
            let vidx = cdt.insert_point(pt)?;
            chord_splits[chord_idx].push((t, vidx));
        }
    }

    // Also check for T-junctions: chord endpoints from OTHER chords that
    // lie on this chord's segment. Only scan chord endpoints (~2*C vertices),
    // not all CDT vertices (~15K+).
    // Chord endpoints are computed by line-edge intersection, which accumulates
    // floating-point error on the order of ~10× tol.linear. Use 100× as the
    // snap threshold to reliably capture all on-chord/on-boundary vertices
    // without pulling in nearby-but-off-chord polygon vertices.
    let snap_dist = tol.linear * CDT_SNAP_FACTOR;
    {
        let all_cdt_verts = cdt.vertices();
        // Collect unique chord endpoint CDT indices and their 2D positions.
        let mut endpoint_set: Vec<(usize, Point2)> = Vec::with_capacity(chord_vidxs.len() * 2);
        for &(cv0, cv1) in &chord_vidxs {
            endpoint_set.push((cv0, all_cdt_verts[cv0]));
            if cv1 != cv0 {
                endpoint_set.push((cv1, all_cdt_verts[cv1]));
            }
        }
        endpoint_set.sort_unstable_by_key(|&(vi, _)| vi);
        endpoint_set.dedup_by_key(|e| e.0);

        for (i, &(v0, v1)) in chord_vidxs.iter().enumerate() {
            if v0 == v1 {
                continue;
            }
            let c0 = chords_2d[i].0;
            let c1 = chords_2d[i].1;
            for &(vidx, pt) in &endpoint_set {
                if vidx == v0 || vidx == v1 {
                    continue;
                }
                let (t, dist) = point_segment_param_dist_2d(pt, c0, c1);
                // 1e-10: parametric boundary guard — exclude points at chord
                // endpoints (t ≈ 0 or t ≈ 1) which are already in the split list.
                if dist < snap_dist && t > 1e-10 && t < 1.0 - 1e-10 {
                    chord_splits[i].push((t, vidx));
                }
            }
        }
    }

    // Sort each chord's split points by parameter.
    for splits in &mut chord_splits {
        splits.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        splits.dedup_by(|a, b| a.1 == b.1);
    }

    // --- Build boundary constraints ---
    // Extended chord endpoints lie on polygon edges. Split boundary edges at
    // these points so CDT constraints don't cross.
    let mut boundary_edges: Vec<(usize, usize)> = Vec::new();

    for i in 0..nv {
        let j = (i + 1) % nv;
        let vi = poly_vidxs[i];
        let vj = poly_vidxs[j];
        let edge_a = poly_2d[i];
        let edge_b = poly_2d[j];

        // Find chord endpoints (CDT indices) that lie on this polygon edge.
        let mut on_edge: Vec<(f64, usize)> = Vec::new();
        for &(cv0, cv1) in &chord_vidxs {
            for &cv in &[cv0, cv1] {
                if cv == vi || cv == vj {
                    continue;
                }
                let pt = cdt.vertices()[cv];
                let (t, dist) = point_segment_param_dist_2d(pt, edge_a, edge_b);
                // 1e-12: parametric boundary guard — tighter than chord T-junction
                // check (1e-10) because boundary edges must not duplicate their
                // own endpoints. Still well above f64 ULP for unit-scale segments.
                if dist < snap_dist && t > 1e-12 && t < 1.0 - 1e-12 {
                    on_edge.push((t, cv));
                }
            }
        }

        on_edge.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        on_edge.dedup_by(|a, b| a.1 == b.1);

        let mut prev = vi;
        for &(_t, cv) in &on_edge {
            if cv != prev {
                boundary_edges.push((prev, cv));
                prev = cv;
            }
        }
        if vj != prev {
            boundary_edges.push((prev, vj));
        }
    }

    for &(a, b) in &boundary_edges {
        cdt.insert_constraint(a, b)?;
    }

    // --- Insert chord constraints (already split at crossings + T-junctions) ---
    let mut chord_separators: Vec<(usize, usize)> = Vec::new();
    for splits in &chord_splits {
        if splits.len() < 2 || splits[0].1 == splits[1].1 {
            continue;
        }
        let mut prev = splits[0].1;
        for &(_, vidx) in &splits[1..] {
            if vidx != prev {
                cdt.insert_constraint(prev, vidx)?;
                chord_separators.push((prev, vidx));
                prev = vidx;
            }
        }
    }

    // --- Remove exterior triangles ---
    cdt.remove_exterior(&boundary_edges);

    // --- Extract regions and unproject to 3D ---
    let regions_2d = cdt.extract_regions(&chord_separators);

    let regions_3d: Vec<Vec<Point3>> = regions_2d
        .into_iter()
        .map(|poly| poly.into_iter().map(&unproject).collect())
        .collect();

    Ok(regions_3d)
}

/// Extend a chord segment to the polygon boundary.
///
/// The chord defines a LINE through `c0` and `c1`. This function finds
/// where that line enters and exits the polygon, returning the boundary
/// intersection points. If the line doesn't cross the polygon (parallel
/// to an edge and outside), returns the original segment.
///
/// Hardcoded epsilons:
/// - `1e-15`: guards against degenerate zero-length chords and parallel-edge
///   denominator checks (numerical zero for f64 with coordinates up to ~1e7).
/// - `1e-10`: edge parameter boundary snap — accepts slight overshoot in the
///   `u ∈ [0, 1]` range due to floating-point arithmetic.
fn extend_chord_to_polygon(c0: Point2, c1: Point2, polygon: &[Point2]) -> (Point2, Point2) {
    let dx = c1.x() - c0.x();
    let dy = c1.y() - c0.y();

    if dx.abs() < 1e-15 && dy.abs() < 1e-15 {
        return (c0, c1);
    }

    let n = polygon.len();
    let mut t_vals: Vec<f64> = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let ex = polygon[j].x() - polygon[i].x();
        let ey = polygon[j].y() - polygon[i].y();

        let denom = dx * ey - dy * ex;
        if denom.abs() < 1e-15 {
            continue; // parallel
        }

        let fx = polygon[i].x() - c0.x();
        let fy = polygon[i].y() - c0.y();

        let t = (fx * ey - fy * ex) / denom;
        let u = (fx * dy - fy * dx) / denom;

        // u must be within polygon edge [0, 1]
        if (-1e-10..=1.0 + 1e-10).contains(&u) {
            t_vals.push(t);
        }
    }

    if t_vals.len() < 2 {
        return (c0, c1);
    }

    t_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let t_min = t_vals[0];
    let t_max = t_vals[t_vals.len() - 1];

    let p_min = Point2::new(dx.mul_add(t_min, c0.x()), dy.mul_add(t_min, c0.y()));
    let p_max = Point2::new(dx.mul_add(t_max, c0.x()), dy.mul_add(t_max, c0.y()));

    (p_min, p_max)
}

/// Strict interior intersection of two 2D line segments.
///
/// Returns `(t_on_ab, t_on_cd, intersection_point)` where both parameters
/// are strictly in (ε, 1−ε). Endpoint-touching segments are NOT considered
/// crossing — only true interior crossings are reported.
fn seg_seg_cross_2d(a: Point2, b: Point2, c: Point2, d_pt: Point2) -> Option<(f64, f64, Point2)> {
    let dx_ab = b.x() - a.x();
    let dy_ab = b.y() - a.y();
    let dx_cd = d_pt.x() - c.x();
    let dy_cd = d_pt.y() - c.y();

    let denom = dx_ab * dy_cd - dy_ab * dx_cd;
    // Numerical-zero guard: 1e-15 protects against degenerate parallel/collinear
    // segments. For unit-scale coordinates, denom = |AB|*|CD|*sin(angle); at
    // ~1e-15 the angle is ~1e-15 rad — well below any meaningful crossing.
    if denom.abs() < 1e-15 {
        return None; // parallel or collinear
    }

    let dx_ac = c.x() - a.x();
    let dy_ac = c.y() - a.y();

    let t = (dx_ac * dy_cd - dy_ac * dx_cd) / denom;
    let u = (dx_ac * dy_ab - dy_ac * dx_ab) / denom;

    // Parametric interior guard: 1e-10 excludes endpoint-touching segments so
    // only true interior crossings are reported. This prevents T-junction
    // false positives from endpoints that coincide within floating-point noise.
    let eps = 1e-10;
    if t > eps && t < 1.0 - eps && u > eps && u < 1.0 - eps {
        let px = dx_ab.mul_add(t, a.x());
        let py = dy_ab.mul_add(t, a.y());
        Some((t, u, Point2::new(px, py)))
    } else {
        None
    }
}

/// Parameter and distance from a 2D point to a line segment.
///
/// Returns `(t, distance)` where `t ∈ [0, 1]` is the closest parameter
/// along segment `(a, b)`.
fn point_segment_param_dist_2d(p: Point2, a: Point2, b: Point2) -> (f64, f64) {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx.mul_add(dx, dy * dy);
    // Numerical-zero guard: 1e-30 ≈ (1e-15)^2 catches degenerate zero-length
    // segments. Using len_sq avoids a sqrt; 1e-30 is far below any meaningful
    // geometric length squared, so this only triggers for truly collapsed segments.
    if len_sq < 1e-30 {
        let dist = ((p.x() - a.x()).powi(2) + (p.y() - a.y()).powi(2)).sqrt();
        return (0.0, dist);
    }
    let t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len_sq;
    let t_clamped = t.clamp(0.0, 1.0);
    let closest_x = dx.mul_add(t_clamped, a.x());
    let closest_y = dy.mul_add(t_clamped, a.y());
    let dist = ((p.x() - closest_x).powi(2) + (p.y() - closest_y).powi(2)).sqrt();
    (t_clamped, dist)
}

/// Compute twice the area of a 3D polygon projected along its normal.
///
/// Used for filtering degenerate (zero-area) fragments. Returns the
/// magnitude of the cross-product sum (Newell's method), which equals
/// `2 * area`.
#[inline]
pub(super) fn polygon_area_2x(vertices: &[Point3], normal: &Vec3) -> f64 {
    if vertices.len() < 3 {
        return 0.0;
    }
    // Project to the dominant axis plane and compute the shoelace area.
    let ax = normal.x().abs();
    let ay = normal.y().abs();
    let az = normal.z().abs();

    let mut area = 0.0;
    let n = vertices.len();
    for i in 0..n {
        let j = (i + 1) % n;
        let (ui, vi, uj, vj) = if az >= ax && az >= ay {
            (
                vertices[i].x(),
                vertices[i].y(),
                vertices[j].x(),
                vertices[j].y(),
            )
        } else if ay >= ax {
            (
                vertices[i].x(),
                vertices[i].z(),
                vertices[j].x(),
                vertices[j].z(),
            )
        } else {
            (
                vertices[i].y(),
                vertices[i].z(),
                vertices[j].y(),
                vertices[j].z(),
            )
        };
        area += ui.mul_add(vj, -(uj * vi));
    }
    area.abs()
}

/// Split a polygon into two sub-polygons along a chord line defined by
/// two points `c0, c1` on the plane.
///
/// Vertices are classified as left/right of the chord using `orient3d`.
/// Edge-chord intersection points are inserted at sign changes.
fn split_polygon_by_chord(
    polygon: &[Point3],
    c0: Point3,
    c1: Point3,
    normal: &Vec3,
) -> (Vec<Point3>, Vec<Point3>) {
    let n = polygon.len();
    if n < 3 {
        return (polygon.to_vec(), Vec::new());
    }

    // The classification plane is defined by (c0, c1, c0 + normal).
    let c_top = Point3::new(
        c0.x() + normal.x(),
        c0.y() + normal.y(),
        c0.z() + normal.z(),
    );

    // Use SoS perturbation so coplanar vertices get a deterministic side
    // assignment (never duplicated into both halves).
    #[allow(clippy::cast_precision_loss)]
    let signs: Vec<f64> = polygon
        .iter()
        .enumerate()
        .map(|(i, v)| {
            orient3d_sos(
                c0,
                c1,
                c_top,
                *v,
                usize::MAX - 2,
                usize::MAX - 1,
                usize::MAX,
                i,
            )
        })
        .collect();

    let mut left = Vec::new();
    let mut right = Vec::new();

    for i in 0..n {
        let j = (i + 1) % n;
        let si = signs[i];
        let sj = signs[j];

        // Classify using SoS sign — guaranteed non-zero, so each vertex
        // goes to exactly one side (no duplicates).
        if si > 0.0 {
            left.push(polygon[i]);
        } else {
            right.push(polygon[i]);
        }

        // Check for sign change (edge crossing).
        if (si > 0.0 && sj < 0.0) || (si < 0.0 && sj > 0.0) {
            // Interpolate the intersection point.
            let t = si / (si - sj);
            let pi = polygon[i];
            let pj = polygon[j];
            let ix = Point3::new(
                (pj.x() - pi.x()).mul_add(t, pi.x()),
                (pj.y() - pi.y()).mul_add(t, pi.y()),
                (pj.z() - pi.z()).mul_add(t, pi.z()),
            );
            left.push(ix);
            right.push(ix);
        }
    }

    (left, right)
}

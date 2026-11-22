//! A torus face that holds the whole tube over part of a turn about the
//! axis (a half ring), split by paths of sections that each run from one of
//! its two end meridians to the other.

use brepkit_math::vec::Point3;
use brepkit_topology::edge::EdgeCurve;
use brepkit_topology::face::{FaceId, FaceSurface};

use super::super::split_types::{OrientedPCurveEdge, SectionEdge, SplitSubFace};
use super::conversion::uv_endpoints_from_pcurve;
use crate::ds::Rank;

/// Split a tube band at section paths from one end meridian to the other.
///
/// The band is bounded by two meridian circles, each whole and in pieces,
/// and the latitude arcs of its seam. The paths do not cross, so they lie
/// in one order round each meridian. Each piece runs along one path from
/// the first meridian to the second, round the second to the next path,
/// back along that path, and round the first meridian to its start. The
/// seam's arcs bound no piece: each lies inside the piece the paths leave
/// round it.
///
/// `None` (the caller falls through) unless the boundary is two meridians
/// and latitude arcs, every section joins a path from one meridian to the
/// other, there are two paths or more, and the paths' order round one
/// meridian is the reverse of their order round the other.
#[allow(clippy::too_many_lines)]
pub(super) fn split_tube_band(
    surface: &FaceSurface,
    boundary_edges: &[OrientedPCurveEdge],
    sections: &[SectionEdge],
    rank: Rank,
    reversed: bool,
    face_id: FaceId,
    tol: f64,
) -> Option<Vec<SplitSubFace>> {
    let FaceSurface::Torus(torus) = surface else {
        return None;
    };
    if sections.len() < 2 || torus.minor_radius() >= torus.major_radius() {
        return None;
    }
    let join = tol * 100.0;
    let axis = torus.z_axis();
    let centre = torus.center();
    let azimuth = |c: Point3| {
        let w = c - centre;
        torus.y_axis().dot(w).atan2(torus.x_axis().dot(w))
    };
    // Each boundary edge is a meridian piece (and which meridian, by the
    // azimuth of its circle's centre) or a latitude arc.
    let mut meridians: Vec<(f64, Vec<OrientedPCurveEdge>)> = Vec::new();
    for e in boundary_edges {
        let EdgeCurve::Circle(c) = &e.curve_3d else {
            return None;
        };
        if c.normal().cross(axis).length() <= 1e-9 {
            continue;
        }
        let w = c.center() - centre;
        let meridian = (c.radius() - torus.minor_radius()).abs() <= tol
            && c.normal().dot(axis).abs() <= 1e-9
            && axis.dot(w).abs() <= tol
            && ((w - axis * axis.dot(w)).length() - torus.major_radius()).abs() <= tol;
        if !meridian {
            return None;
        }
        let at = azimuth(c.center());
        let same = |a: f64| {
            let d = (at - a).rem_euclid(std::f64::consts::TAU);
            d.min(std::f64::consts::TAU - d) <= 1e-9
        };
        match meridians.iter_mut().find(|(a, _)| same(*a)) {
            Some((_, pieces)) => pieces.push(e.clone()),
            None => meridians.push((at, vec![e.clone()])),
        }
    }
    // A latitude arc is the seam, run once each way.
    let latitudes: Vec<&OrientedPCurveEdge> = boundary_edges
        .iter()
        .filter(|e| matches!(&e.curve_3d, EdgeCurve::Circle(c) if c.normal().cross(axis).length() <= 1e-9))
        .collect();
    if !latitudes.iter().all(|e| {
        latitudes.iter().any(|f| {
            (f.start_3d - e.end_3d).length() <= join && (f.end_3d - e.start_3d).length() <= join
        })
    }) {
        return None;
    }
    let [(_, first), (_, second)] = <[_; 2]>::try_from(meridians).ok()?;
    let chain_a = closed_chain(first, join)?;
    let chain_b = closed_chain(second, join)?;
    let vertex_on = |chain: &[OrientedPCurveEdge], p: Point3| {
        chain.iter().position(|e| (e.start_3d - p).length() <= join)
    };

    // The sections joined end to end into paths from one meridian to the
    // other, each run from the first meridian.
    let mut pool: Vec<OrientedPCurveEdge> = sections
        .iter()
        .map(|s| section_edge(s, surface, rank))
        .collect();
    let mut paths: Vec<(usize, usize, Vec<OrientedPCurveEdge>)> = Vec::new();
    while let Some(i) = pool.iter().position(|e| {
        vertex_on(&chain_a, e.start_3d).is_some() || vertex_on(&chain_a, e.end_3d).is_some()
    }) {
        let mut e = pool.remove(i);
        if vertex_on(&chain_a, e.start_3d).is_none() {
            e = flipped(&e);
        }
        let from = vertex_on(&chain_a, e.start_3d)?;
        let mut path = vec![e];
        loop {
            let tail = path[path.len() - 1].end_3d;
            if let Some(to) = vertex_on(&chain_b, tail) {
                paths.push((from, to, path));
                break;
            }
            if vertex_on(&chain_a, tail).is_some() {
                return None;
            }
            let j = pool.iter().position(|e| {
                (e.start_3d - tail).length() <= join || (e.end_3d - tail).length() <= join
            })?;
            let next = pool.remove(j);
            path.push(if (next.start_3d - tail).length() <= join {
                next
            } else {
                flipped(&next)
            });
        }
    }
    if !pool.is_empty() || paths.len() < 2 {
        return None;
    }

    // Round the second meridian the paths come in one order; round the
    // first, the piece between two paths runs from the later one back to
    // the earlier, so that order must be the reverse.
    let n = paths.len();
    paths.sort_by_key(|&(_, to, _)| to);
    if paths.windows(2).any(|w| w[0].1 == w[1].1) {
        return None;
    }
    let mut by_a: Vec<usize> = (0..n).collect();
    by_a.sort_by_key(|&k| paths[k].0);
    if by_a.windows(2).any(|w| paths[w[0]].0 == paths[w[1]].0) {
        return None;
    }
    let rank_on_a: Vec<usize> = {
        let mut r = vec![0; n];
        for (pos, &k) in by_a.iter().enumerate() {
            r[k] = pos;
        }
        r
    };
    if (0..n).any(|k| (rank_on_a[(k + 1) % n] + 1) % n != rank_on_a[k]) {
        return None;
    }

    let run = |chain: &[OrientedPCurveEdge], from: usize, to: usize| {
        let m = chain.len();
        let len = (to + m - from) % m;
        (0..len)
            .map(|i| chain[(from + i) % m].clone())
            .collect::<Vec<_>>()
    };
    let mut pieces = Vec::with_capacity(n);
    for k in 0..n {
        let (a0, b0, ref here) = paths[k];
        let (a1, b1, ref next) = paths[(k + 1) % n];
        let mut wire: Vec<OrientedPCurveEdge> = here.clone();
        wire.extend(run(&chain_b, b0, b1));
        wire.extend(next.iter().rev().map(flipped));
        wire.extend(run(&chain_a, a1, a0));
        pieces.push(SplitSubFace {
            surface: surface.clone(),
            outer_wire: wire,
            inner_wires: Vec::new(),
            reversed,
            parent: face_id,
            rank,
            precomputed_interior: None,
        });
    }
    Some(pieces)
}

/// The pieces of one closed curve chained end to start from the first,
/// `None` when they do not close up or one is left over.
fn closed_chain(mut pieces: Vec<OrientedPCurveEdge>, join: f64) -> Option<Vec<OrientedPCurveEdge>> {
    let mut chain = vec![pieces.remove(0)];
    while !pieces.is_empty() {
        let tail = chain[chain.len() - 1].end_3d;
        let i = pieces
            .iter()
            .position(|e| (e.start_3d - tail).length() <= join)?;
        chain.push(pieces.remove(i));
    }
    ((chain[chain.len() - 1].end_3d - chain[0].start_3d).length() <= join).then_some(chain)
}

/// An edge run the other way.
fn flipped(e: &OrientedPCurveEdge) -> OrientedPCurveEdge {
    OrientedPCurveEdge {
        curve_3d: e.curve_3d.clone(),
        pcurve: e.pcurve.clone(),
        start_uv: e.end_uv,
        end_uv: e.start_uv,
        start_3d: e.end_3d,
        end_3d: e.start_3d,
        forward: !e.forward,
        source_edge_idx: e.source_edge_idx,
        pave_block_id: e.pave_block_id,
    }
}

/// A section on this face as an edge run from its start.
fn section_edge(section: &SectionEdge, surface: &FaceSurface, rank: Rank) -> OrientedPCurveEdge {
    let pcurve = match rank {
        Rank::A => &section.pcurve_a,
        Rank::B => &section.pcurve_b,
    };
    let (start_uv, end_uv) = match rank {
        Rank::A => section.start_uv_a.zip(section.end_uv_a),
        Rank::B => section.start_uv_b.zip(section.end_uv_b),
    }
    .unwrap_or_else(|| uv_endpoints_from_pcurve(pcurve, section.start, section.end, surface, &[]));
    OrientedPCurveEdge {
        curve_3d: section.curve_3d.clone(),
        pcurve: pcurve.clone(),
        start_uv,
        end_uv,
        start_3d: section.start,
        end_3d: section.end,
        forward: true,
        source_edge_idx: None,
        pave_block_id: section.pave_block_id,
    }
}

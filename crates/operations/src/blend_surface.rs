//! Surfaces built with continuity control at their edges: a bridge between two
//! edge chains, one surface's edge moved on to another's, and fairing.
//!
//! A blend or a match ends in a polynomial tensor-product spline. A boundary
//! row of control points and the two beside it fix a surface's position,
//! tangent plane and cross-boundary curvature along that boundary, so each
//! continuity is built by choosing those rows from the neighbouring surface's
//! fundamental forms. Nothing here reads a tessellation.
//!
//! Two surfaces meeting along a curve with one tangent plane at every point of
//! it agree in curvature along the curve and in the twist across it, so they
//! are curvature continuous once they agree in normal curvature in any one
//! direction across the curve. That one number is what the G2 row sets.

mod chain;
mod fair;
mod match_edge;
mod neighbour;
mod rows;
mod spline;

pub use fair::fair_surface;
pub use match_edge::{match_surface, match_surface_on};

use brepkit_geometry::extrema::segment_segment_distance;
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::curve_split;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::vertex::{Vertex, VertexId};
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use crate::OperationsError;
use chain::Chain;
use neighbour::Side;
use rows::{Ramps, dedup_breaks, row_tolerance, smoother};
use spline::{Space, fit_rows};

/// The degree along a chain. High enough that a fit converges fast, so the
/// derivatives the seam's continuity rests on converge too.
const ALONG_DEGREE: usize = 5;

/// Vertices closer than this are one vertex.
const WELD: f64 = 1e-9;

/// The share of a chain's length its arc-length quadrature may be out by.
const ARC_ROUNDING: f64 = 1e-12;

/// Pairs of chain pieces found near each other past this many are refused as
/// touching: the chains run that close for too long to prove apart.
const MOST_NEAR_PAIRS: usize = 1 << 16;

/// The least room an end the chains share takes along them, as a share of
/// the longer chain's length, and the most, as a share of the shorter's.
const SHARED_END_LEAST: f64 = 1e-6;
const SHARED_END_MOST: f64 = 1e-3;

/// The continuity a blend surface keeps with the face beside each chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendContinuity {
    /// Shared tangent plane along each chain.
    G1,
    /// Shared tangent plane and curvature along each chain.
    G2,
}

/// The continuity a matched edge keeps with its target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchContinuity {
    /// The edge lies on the target edge.
    G0,
    /// As `G0`, with a shared tangent plane along the edge.
    G1,
    /// As `G1`, with shared curvature along the edge.
    G2,
}

/// How a blend surface leaves its two chains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BlendSurfaceOptions {
    /// The continuity with the face beside each chain.
    pub continuity: BlendContinuity,
    /// How far the surface carries on along face A's tangent plane before
    /// turning, as a multiple of the distance between the chains. Positive.
    pub weight_a: f64,
    /// As `weight_a`, for face B.
    pub weight_b: f64,
}

impl BlendSurfaceOptions {
    /// Options for `continuity` with both tangent weights 1.
    #[must_use]
    pub const fn new(continuity: BlendContinuity) -> Self {
        Self {
            continuity,
            weight_a: 1.0,
            weight_b: 1.0,
        }
    }
}

/// Bridge two edge chains with a surface continuous to the face beside each.
///
/// `chain_a` lies on the boundary of `face_a` and `chain_b` on that of
/// `face_b`. Each chain lists its edges in running order, and each edge may be
/// stored either way along it. Lines, arcs, ellipses and splines mix freely,
/// and the two chains need not have the same edges. Both must be open or both
/// closed. Either face may be a sheet's or a solid's, on any surface type.
///
/// The surface runs in `u` along the chains and in `v` from chain A (`v = 0`)
/// to chain B (`v = 1`), and its normal agrees with face A's along chain A.
/// Which end of chain B meets which end of chain A is whichever keeps the
/// surface least twisted, and a closed pair starts at the least twisted place.
///
/// Chains with the same number of edges pair edge by edge; otherwise they pair
/// by arc length, so that corners meet where the arc length says they do.
///
/// At a corner in a chain the cross-boundary tangent runs along the bisector,
/// tilting back to square over the edges either side: it has to lie in the
/// tangent plane to be continuous and transverse to both edges.
///
/// If the two chains meet at an end, the surface closes to a point there, and
/// has no normal of its own at that one point.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if a weight is not positive and
/// finite, either chain is empty, breaks, repeats an edge, doubles back or
/// leaves its face, an edge is a seam of its face or not on its boundary, one
/// chain is closed and the other open, or the chains touch away from their
/// ends. A topology lookup can also fail.
pub fn blend_surface(
    topo: &mut Topology,
    chain_a: &[EdgeId],
    face_a: FaceId,
    chain_b: &[EdgeId],
    face_b: FaceId,
    options: &BlendSurfaceOptions,
) -> Result<FaceId, OperationsError> {
    for (name, weight) in [("A", options.weight_a), ("B", options.weight_b)] {
        if !weight.is_finite() || weight <= 0.0 {
            return Err(OperationsError::InvalidInput {
                reason: format!(
                    "the tangent weight of chain {name} must be positive, got {weight}"
                ),
            });
        }
    }
    let mut a = Side::new(topo, chain_a, face_a)?;
    let mut b = Side::new(topo, chain_b, face_b)?;
    if a.chain.closed != b.chain.closed {
        return Err(OperationsError::InvalidInput {
            reason: "one chain is closed and the other open, so no surface runs along both".into(),
        });
    }
    face_a_upward(&mut a)?;
    if pair_chains(&a.chain, &mut b.chain) {
        b.sign = -b.sign;
    }

    let reach = chain_gap(&a.chain, &b.chain)?;
    let degree_across = match options.continuity {
        BlendContinuity::G1 => 3,
        BlendContinuity::G2 => 5,
    };
    let tolerance = row_tolerance(reach.0, reach.1);

    let mut breaks = vec![0.0, 1.0];
    breaks.extend(a.chain.joints());
    breaks.extend(b.chain.joints());
    breaks.extend(a.chain.knot_breaks());
    breaks.extend(b.chain.knot_breaks());
    dedup_breaks(&mut breaks);
    let crossings: Vec<f64> = a
        .surface_knot_breaks(&breaks)?
        .into_iter()
        .chain(b.surface_knot_breaks(&breaks)?)
        .collect();
    breaks.extend(crossings);
    dedup_breaks(&mut breaks);

    let closed = a.chain.closed;
    let bridge =
        |tilts_a: &[(f64, f64)], tilts_b: &[(f64, f64)], lo: f64, hi: f64, piece: usize, s: f64| {
            bridge_rows(
                &a,
                &b,
                options,
                (tilts_a[piece], tilts_b[piece]),
                (lo, hi),
                s,
            )
        };

    let mut tilts_a = a.tilts(&breaks)?;
    let mut tilts_b = b.tilts(&breaks)?;
    let mut ramps = Ramps::measure(&breaks, closed, tolerance, |piece, lo, hi, s| {
        bridge(&tilts_a, &tilts_b, lo, hi, piece, s)
    })?;
    if ramps.needs_bands() {
        breaks.extend(ramps.band_breaks(&breaks));
        dedup_breaks(&mut breaks);
        tilts_a = a.tilts(&breaks)?;
        tilts_b = b.tilts(&breaks)?;
        ramps = Ramps::measure(&breaks, closed, tolerance, |piece, lo, hi, s| {
            bridge(&tilts_a, &tilts_b, lo, hi, piece, s)
        })?;
    }

    let start = Space::c0_over(ALONG_DEGREE, &breaks);
    let (space, mut fitted) = fit_rows(
        start,
        degree_across + 1,
        tolerance,
        "a blend surface along its chains",
        |s| {
            let piece = rows::piece_of(&breaks, s);
            let (lo, hi) = (breaks[piece], breaks[piece + 1]);
            let mut values = bridge(&tilts_a, &tilts_b, lo, hi, piece, s)?;
            ramps.apply(piece, lo, hi, s, &mut values);
            Ok(values)
        },
    )?;
    if closed {
        for row in &mut fitted {
            let first = row[0];
            let last = row.len() - 1;
            row[last] = first;
        }
    }

    let surface = surface_from_rows(&space, &fitted, degree_across)?;
    let bottom = a.chain.edge_joints();
    let top = b.chain.edge_joints();
    boundary_face(
        topo,
        surface,
        &SideJoints {
            bottom: &bottom,
            right: &[],
            top: &top,
            left: &[],
        },
        closed,
        Vec::new(),
    )
}

fn bridge_rows(
    a: &Side,
    b: &Side,
    options: &BlendSurfaceOptions,
    (tilt_a, tilt_b): ((f64, f64), (f64, f64)),
    piece: (f64, f64),
    s: f64,
) -> Result<Vec<Vec3>, OperationsError> {
    let along = (s - piece.0) / (piece.1 - piece.0);
    let at_a = a.station(piece, s)?;
    let at_b = b.station(piece, s)?;
    let across = (at_b.point - at_a.point).length();
    let pa = at_a.point - Point3::new(0.0, 0.0, 0.0);
    let pb = at_b.point - Point3::new(0.0, 0.0, 0.0);
    let da = tilted(at_a.off, at_a.tangent, tilt_a, along) * (options.weight_a * across);
    let db = tilted(at_b.off, at_b.tangent, tilt_b, along) * (options.weight_b * across);
    Ok(match options.continuity {
        BlendContinuity::G1 => vec![pa, pa + da * (1.0 / 3.0), pb + db * (1.0 / 3.0), pb],
        BlendContinuity::G2 => {
            let curl_a = at_a.normal * (at_a.form.second_form(da) / 20.0);
            let curl_b = at_b.normal * (at_b.form.second_form(db) / 20.0);
            vec![
                pa,
                pa + da * 0.2,
                pa + da * 0.4 + curl_a,
                pb + db * 0.4 + curl_b,
                pb + db * 0.2,
                pb,
            ]
        }
    })
}

/// `off` turned towards `tangent` by the piece's tilt at `along` through it.
fn tilted(off: Vec3, tangent: Vec3, (from, to): (f64, f64), along: f64) -> Vec3 {
    if from == 0.0 && to == 0.0 {
        return off;
    }
    let w = smoother(along);
    let (sin, cos) = (from * (1.0 - w) + to * w).sin_cos();
    off * cos + tangent * sin
}

/// Run chain A so that its tangent, crossed with the way off face A, is the
/// way the face presents.
fn face_a_upward(a: &mut Side) -> Result<(), OperationsError> {
    let piece = {
        let first = &a.chain.segments[0];
        (first.s0, first.s1)
    };
    let station = a.station(piece, 0.5 * (piece.0 + piece.1))?;
    if station.tangent.cross(station.off).dot(station.form.logical) < 0.0 {
        a.reverse();
    }
    Ok(())
}

/// Run chain `b`, and start it if it is closed, so that it lies least twisted
/// against chain `a`, and bring together the joins of chains of equal edge
/// counts. Returns whether `b` now runs the other way.
fn pair_chains(a: &Chain, b: &mut Chain) -> bool {
    const SAMPLES: usize = 128;
    let along = |chain: &Chain| -> Vec<Point3> {
        let last = if chain.closed { SAMPLES } else { SAMPLES - 1 };
        (0..SAMPLES)
            .map(|i| chain.point(i as f64 / last as f64))
            .collect()
    };
    let from_a = along(a);
    let twist = |points: &[Point3], offset: usize| -> f64 {
        from_a
            .iter()
            .enumerate()
            .map(|(i, p)| (*p - points[(i + offset) % SAMPLES]).length_squared())
            .sum()
    };
    let forward = along(b);
    let back_chain = b.reversed();
    let backward = along(&back_chain);
    let mut run_back = false;
    if a.closed {
        let same_counts = a.segments.len() == b.segments.len() && a.segments.len() > 1;
        let starts = |chain: &Chain| -> Vec<f64> {
            if same_counts {
                std::iter::once(0.0).chain(chain.joints()).collect()
            } else {
                (0..SAMPLES).map(|i| i as f64 / SAMPLES as f64).collect()
            }
        };
        let best = |points: &[Point3], chain: &Chain| -> (f64, f64) {
            starts(chain)
                .into_iter()
                .map(|s| {
                    let offset = (s * SAMPLES as f64).round() as usize % SAMPLES;
                    (twist(points, offset), s)
                })
                .fold((f64::INFINITY, 0.0), |x, y| if y.0 < x.0 { y } else { x })
        };
        let (cost_f, start_f) = best(&forward, b);
        let (cost_b, start_b) = best(&backward, &back_chain);
        let start = if cost_b < cost_f {
            run_back = true;
            *b = back_chain;
            start_b
        } else {
            start_f
        };
        *b = b.started_at(start);
    } else if twist(&backward, 0) < twist(&forward, 0) {
        run_back = true;
        *b = back_chain;
    }

    let joints_a = a.joints();
    let joints_b = b.joints();
    if joints_a.len() == joints_b.len() {
        b.set_joints(&joints_a);
    } else {
        let snapped: Vec<f64> = joints_b
            .iter()
            .map(|&s| {
                joints_a
                    .iter()
                    .copied()
                    .find(|j| (*j - s).abs() < 1e-3)
                    .unwrap_or(s)
            })
            .collect();
        if snapped.windows(2).all(|pair| pair[1] - pair[0] > 1e-3) {
            b.set_joints(&snapped);
        }
    }
    run_back
}

/// The smallest distance between the chains at the same `s` away from any end
/// they share, and the largest coordinate on either. Chains that touch, at the
/// same `s` or not, are refused.
fn chain_gap(a: &Chain, b: &Chain) -> Result<(f64, f64), OperationsError> {
    let mut smallest = f64::INFINITY;
    let mut largest = 0.0_f64;
    let mut widest = 0.0_f64;
    for i in 0..=128 {
        let s = f64::from(i) / 128.0;
        let (p, q) = (a.point(s), b.point(s));
        let gap = (p - q).length();
        widest = widest.max(gap);
        largest = largest
            .max(p.x().abs().max(p.y().abs()).max(p.z().abs()))
            .max(q.x().abs().max(q.y().abs()).max(q.z().abs()));
        if i != 0 && i != 128 {
            if gap <= WELD {
                return Err(OperationsError::InvalidInput {
                    reason: format!("the two chains touch at {p:?}, away from their ends"),
                });
            }
            smallest = smallest.min(gap);
        }
    }
    if widest <= WELD {
        return Err(OperationsError::InvalidInput {
            reason: "the two chains lie on top of each other".into(),
        });
    }
    if let Some(p) = chains_touch(a, b) {
        return Err(OperationsError::InvalidInput {
            reason: format!("the two chains touch at {p:?}, away from their ends"),
        });
    }
    Ok((smallest.max(1e-6 * widest), largest))
}

/// A piece of a chain between two of its points.
#[derive(Clone, Copy)]
struct Piece {
    from: f64,
    to: f64,
    start: Point3,
    end: Point3,
    arc: f64,
    /// How far the chain can stray from the chord: no point of the piece is
    /// farther from its two ends together than the arc length, so the piece
    /// lies in the ellipse round them.
    reach: f64,
}

impl Piece {
    fn new(chain: &Chain, (from, to): (f64, f64), (start, end): (Point3, Point3)) -> Self {
        let arc = chain.length_between(from, to);
        let chord = (end - start).length();
        Self {
            from,
            to,
            start,
            end,
            arc,
            reach: 0.5 * (arc * arc - chord * chord).max(0.0).sqrt()
                + rounding_reach(arc, chain.length),
        }
    }

    fn halves(&self, chain: &Chain) -> [Self; 2] {
        let middle = 0.5 * (self.from + self.to);
        let at = chain.point(middle);
        [
            Self::new(chain, (self.from, middle), (self.start, at)),
            Self::new(chain, (middle, self.to), (at, self.end)),
        ]
    }
}

/// Where the chains come within [`WELD`] of each other, each read at its own
/// `s`, away from an end they share: chains can cross at different shares of
/// their lengths. Pairs of pieces are halved until their chords and reaches
/// prove them apart, or they meet.
fn chains_touch(a: &Chain, b: &Chain) -> Option<Point3> {
    const STEPS: usize = 128;
    let pieces = |chain: &Chain| -> Vec<Piece> {
        let at = |i: usize| i as f64 / STEPS as f64;
        let points: Vec<Point3> = (0..=STEPS).map(|i| chain.point(at(i))).collect();
        (0..STEPS)
            .map(|i| Piece::new(chain, (at(i), at(i + 1)), (points[i], points[i + 1])))
            .collect()
    };
    let (along_a, along_b) = (pieces(a), pieces(b));
    let last = STEPS - 1;
    let open = !a.closed;
    let first_shared = open && (along_a[0].start - along_b[0].start).length() <= WELD;
    let last_shared = open && (along_a[last].end - along_b[last].end).length() <= WELD;
    let first_room = if first_shared {
        shared_end_room(a, b, false)
    } else {
        0.0
    };
    let last_room = if last_shared {
        shared_end_room(a, b, true)
    } else {
        0.0
    };
    let at_the_end = |piece_a: &Piece, piece_b: &Piece| {
        let near_first =
            |chain: &Chain, piece: &Piece| chain.length_between(0.0, piece.to) <= first_room;
        let near_last =
            |chain: &Chain, piece: &Piece| chain.length_between(piece.from, 1.0) <= last_room;
        (first_shared && near_first(a, piece_a) && near_first(b, piece_b))
            || (last_shared && near_last(a, piece_a) && near_last(b, piece_b))
    };
    let mut pairs = Vec::with_capacity(STEPS * STEPS);
    for piece_a in &along_a {
        for piece_b in &along_b {
            pairs.push((*piece_a, *piece_b));
        }
    }
    let mut near = 0;
    while let Some((piece_a, piece_b)) = pairs.pop() {
        if at_the_end(&piece_a, &piece_b) {
            continue;
        }
        let (gap, _, _) =
            segment_segment_distance(piece_a.start, piece_a.end, piece_b.start, piece_b.end);
        if gap > piece_a.reach + piece_b.reach + WELD {
            continue;
        }
        let both_first = first_shared && piece_a.from <= 0.0 && piece_b.from <= 0.0;
        let both_last = last_shared && piece_a.to >= 1.0 && piece_b.to >= 1.0;
        let (closest, at) = [
            (piece_a.start, piece_b.start, both_first),
            (piece_a.start, piece_b.end, false),
            (piece_a.end, piece_b.start, false),
            (piece_a.end, piece_b.end, both_last),
        ]
        .into_iter()
        .filter(|&(_, _, shared_end)| !shared_end)
        .map(|(p, q, _)| ((p - q).length(), p))
        .fold((f64::INFINITY, piece_a.start), |x, y| {
            if y.0 < x.0 { y } else { x }
        });
        near += 1;
        // Pieces no longer than the weld and not proved apart are as good as
        // touching.
        let unresolved = near > MOST_NEAR_PAIRS || piece_a.arc.max(piece_b.arc) <= WELD;
        if closest <= WELD || unresolved {
            return Some(at);
        }
        for half_a in piece_a.halves(a) {
            for half_b in piece_b.halves(b) {
                pairs.push((half_a, half_b));
            }
        }
    }
    None
}

/// How far along the chains an end they share reaches: out to where the two,
/// read at equal arc lengths in from it, first stand apart by more than a few
/// welds and the rounding a piece that long can hide. Chains meeting at an
/// angle `q` stay within the weld of each other out to about `WELD / q`, and
/// chains meeting tangentially part only quadratically.
fn shared_end_room(a: &Chain, b: &Chain, from_end: bool) -> f64 {
    let longer = a.length.max(b.length);
    let most = SHARED_END_MOST * a.length.min(b.length);
    let mut along = SHARED_END_LEAST * longer;
    for _ in 0..64 {
        if along >= most {
            break;
        }
        let p = a.point(a.s_in_from(along, from_end));
        let q = b.point(b.s_in_from(along, from_end));
        if (p - q).length() > 4.0 * (WELD + rounding_reach(along, longer)) {
            return along;
        }
        along *= 2.0;
    }
    most
}

/// How much of a piece's reach an arc length `arc` on a chain `length` long
/// can hide: one short by `d` hides up to `sqrt(arc d)`.
fn rounding_reach(arc: f64, length: f64) -> f64 {
    (arc * ARC_ROUNDING * length).sqrt()
}

fn surface_from_rows(
    space: &Space,
    rows: &[Vec<Vec3>],
    degree_across: usize,
) -> Result<NurbsSurface, OperationsError> {
    let columns = space.count();
    let net: Vec<Vec<Point3>> = (0..columns)
        .map(|i| {
            rows.iter()
                .map(|row| Point3::new(0.0, 0.0, 0.0) + row[i])
                .collect()
        })
        .collect();
    let across = Space::clamped(degree_across, &[(0.0, 0), (1.0, 0)]);
    surface_from_net(space, &across, net)
}

fn surface_from_net(
    u: &Space,
    v: &Space,
    net: Vec<Vec<Point3>>,
) -> Result<NurbsSurface, OperationsError> {
    let weights = vec![vec![1.0; v.count()]; u.count()];
    Ok(NurbsSurface::new(
        u.degree,
        v.degree,
        u.knots.clone(),
        v.knots.clone(),
        net,
        weights,
    )?)
}

/// Where a face's boundary curves are cut into edges: the bottom (`v = 0`) and
/// top in `u`, the right (`u = 1`) and left in `v`.
struct SideJoints<'a> {
    bottom: &'a [f64],
    right: &'a [f64],
    top: &'a [f64],
    left: &'a [f64],
}

/// A new face on `surface`, bounded by its four parameter edges and running
/// counter-clockwise about its normal. A side at one point gets no edge, and a
/// surface `closed_u` has one seam edge, used both ways.
fn boundary_face(
    topo: &mut Topology,
    surface: NurbsSurface,
    joints: &SideJoints<'_>,
    closed_u: bool,
    inner_wires: Vec<WireId>,
) -> Result<FaceId, OperationsError> {
    let net = surface.control_points();
    let (columns, rows) = (net.len(), net[0].len());
    let along_u = |j: usize| -> Result<NurbsCurve, OperationsError> {
        let points: Vec<Point3> = net.iter().map(|column| column[j]).collect();
        Ok(NurbsCurve::new(
            surface.degree_u(),
            surface.knots_u().to_vec(),
            points,
            vec![1.0; columns],
        )?)
    };
    let along_v = |i: usize| -> Result<NurbsCurve, OperationsError> {
        Ok(NurbsCurve::new(
            surface.degree_v(),
            surface.knots_v().to_vec(),
            net[i].clone(),
            vec![1.0; rows],
        )?)
    };

    let mut vertices: Vec<(Point3, VertexId)> = Vec::new();
    let bottom = make_edges(topo, &mut vertices, along_u(0)?, joints.bottom)?;
    let right = make_edges(topo, &mut vertices, along_v(columns - 1)?, joints.right)?;
    let top = make_edges(topo, &mut vertices, along_u(rows - 1)?, joints.top)?;
    let left = if closed_u {
        right.clone()
    } else {
        make_edges(topo, &mut vertices, along_v(0)?, joints.left)?
    };

    let mut oriented = Vec::with_capacity(bottom.len() + right.len() + top.len() + left.len());
    oriented.extend(bottom.iter().map(|&e| OrientedEdge::new(e, true)));
    oriented.extend(right.iter().map(|&e| OrientedEdge::new(e, true)));
    oriented.extend(top.iter().rev().map(|&e| OrientedEdge::new(e, false)));
    oriented.extend(left.iter().rev().map(|&e| OrientedEdge::new(e, false)));
    let wire = topo.add_wire(Wire::new(oriented, true)?);
    Ok(topo.add_face(Face::new(wire, inner_wires, FaceSurface::Nurbs(surface))))
}

/// Edges along `curve`, cut at `joints`; none if it stays at one point.
fn make_edges(
    topo: &mut Topology,
    vertices: &mut Vec<(Point3, VertexId)>,
    curve: NurbsCurve,
    joints: &[f64],
) -> Result<Vec<EdgeId>, OperationsError> {
    let first = curve.control_points()[0];
    if curve
        .control_points()
        .iter()
        .all(|p| (*p - first).length() <= WELD)
    {
        return Ok(Vec::new());
    }
    let mut pieces = Vec::with_capacity(joints.len() + 1);
    let mut rest = curve;
    for &joint in joints {
        let (head, tail) = curve_split(&rest, joint)?;
        pieces.push(head);
        rest = tail;
    }
    pieces.push(rest);

    let mut edges = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let points = piece.control_points();
        let start = vertex_at(topo, vertices, points[0]);
        let end = vertex_at(topo, vertices, points[points.len() - 1]);
        edges.push(topo.add_edge(Edge::new(start, end, EdgeCurve::NurbsCurve(piece))));
    }
    Ok(edges)
}

fn vertex_at(
    topo: &mut Topology,
    vertices: &mut Vec<(Point3, VertexId)>,
    point: Point3,
) -> VertexId {
    if let Some(&(_, id)) = vertices.iter().find(|(p, _)| (*p - point).length() <= WELD) {
        return id;
    }
    let id = topo.add_vertex(Vertex::new(point, 1e-7));
    vertices.push((point, id));
    id
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use brepkit_geometry::convert::circle_to_nurbs;
    use brepkit_math::curves::Circle3D;

    use super::*;

    fn arc(circle: &Circle3D, from: f64, to: f64) -> Chain {
        let curve = circle_to_nurbs(circle, from, to).unwrap();
        Chain::from_curve(curve, circle.evaluate(from), circle.evaluate(to)).unwrap()
    }

    fn bezier(points: &[Point3]) -> Chain {
        let degree = points.len() - 1;
        let mut knots = vec![0.0; degree + 1];
        knots.extend(vec![1.0; degree + 1]);
        let curve =
            NurbsCurve::new(degree, knots, points.to_vec(), vec![1.0; points.len()]).unwrap();
        Chain::from_curve(curve, points[0], points[degree]).unwrap()
    }

    fn at(x: f64, y: f64) -> Point3 {
        Point3::new(x, y, 0.0)
    }

    /// `A(s) = (s, 0)` and `B(t) = (2t, t (t - 0.003))` share their start and
    /// cross again at `(0.006, 0)`, inside the first piece of each.
    #[test]
    fn a_crossing_beside_a_shared_end_touches() {
        let a = bezier(&[at(0.0, 0.0), at(1.0, 0.0)]);
        let b = bezier(&[at(0.0, 0.0), at(1.0, -0.0015), at(2.0, 0.997)]);
        for (a, b) in [(a.clone(), b.clone()), (a.reversed(), b.reversed())] {
            let found = chains_touch(&a, &b);
            assert!(
                found.is_some_and(|p| (p - at(0.006, 0.0)).length() < 1e-6),
                "{found:?}"
            );
        }
    }

    /// Chains that only meet where they end, at an angle or tangentially and
    /// at either end, do not touch.
    #[test]
    fn chains_meeting_only_at_a_shared_end_do_not_touch() {
        let a = bezier(&[at(0.0, 0.0), at(1.0, 0.0)]);
        for b in [
            bezier(&[at(0.0, 0.0), at(1.0, 0.3), at(2.0, 1.0)]),
            bezier(&[at(0.0, 0.0), at(1.0, 0.0), at(2.0, 1.0)]),
        ] {
            assert_eq!(chains_touch(&a, &b), None);
            assert_eq!(chains_touch(&a.reversed(), &b.reversed()), None);
        }
    }

    /// Two arcs that meet out of each other's plane, at different shares of
    /// their lengths and between samples on both: no chord between samples
    /// meets another, so only halving the pieces finds where the arcs meet.
    /// Lifted 1e-6 apart, they do not touch.
    #[test]
    fn arcs_meeting_out_of_plane_touch_and_lifted_apart_do_not() {
        let z = Vec3::new(0.0, 0.0, 1.0);
        let origin = Point3::new(0.0, 0.0, 0.0);
        let rim = Circle3D::new_with_ref(origin, z, 5.0, Vec3::new(1.0, 0.0, 0.0)).unwrap();
        let a = arc(&rim, 0.0, 2.1);
        let meet = rim.evaluate(0.5);
        for (lift, touches) in [(0.0, true), (1e-6, false)] {
            let hoop =
                Circle3D::new_with_ref(meet + z * (4.0 + lift), rim.tangent(0.5), 4.0, -z).unwrap();
            let b = arc(&hoop, -2.4, 0.6);
            let found = chains_touch(&a, &b);
            assert_eq!(found.is_some(), touches, "lifted {lift}: {found:?}");
            if let Some(p) = found {
                assert!((p - meet).length() < 1e-6, "{p:?}");
            }
        }
    }
}

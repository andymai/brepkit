//! Moving one surface's edge on to another's.
//!
//! The source becomes a polynomial spline `G(s, t)` with the edge at `t = 0`,
//! then gains a displacement built from quintic Hermite functions: it sets the
//! value and the first two `t` derivatives at `t = 0`, and has all three at
//! zero at `t = tau`, so past `tau` the surface is exactly what it was, and
//! joins it curvature continuously.

use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::EdgeId;
use brepkit_topology::face::{FaceId, FaceSurface};
use brepkit_topology::wire::WireId;

use super::chain::Chain;
use super::neighbour::{Neighbour, Side};
use super::rows::{Ramps, dedup_breaks, piece_of, row_tolerance};
use super::spline::{Space, fit_grid, fit_rows, solve_grid};
use super::{
    ALONG_DEGREE, MatchContinuity, SideJoints, boundary_face, pair_chains, surface_from_net, tilted,
};
use crate::OperationsError;

/// The narrowest share of the surface's width the displacement fades over.
const NARROWEST_BAND: f64 = 0.25;

/// The largest slope a displacement may put into the surface over its band,
/// before the band has to widen; `H0'` peaks at 1.875 and `H1''` at 5.77.
const H0_SLOPE_PEAK: f64 = 1.875;
const H1_BEND_PEAK: f64 = 5.77;

/// How close a hole may come to the edge, as a share of the width: the band
/// has to fit between them.
const THINNEST_BAND: f64 = 0.02;

/// How much of the cross-boundary speed has to point off the target face for
/// the source's own tangent to serve, once flattened into the tangent plane.
const LEAVES_FACE: f64 = 0.2;

/// Breaks within this of a knot of the source land on the knot.
const NEAR_KNOT: f64 = 1e-7;

/// How far inside a piece its last derivatives are read.
const INSIDE_PIECE: f64 = 1e-13;

/// `1 - 10x^3 + 15x^4 - 6x^5`: 1 at 0, and 0 at 1, flat to second order at
/// both.
fn h0(x: f64) -> f64 {
    1.0 - x * x * x * (10.0 + x * (6.0 * x - 15.0))
}

/// `x - 6x^3 + 8x^4 - 3x^5`: slope 1 at 0, and otherwise flat to second order
/// at both ends.
fn h1(x: f64) -> f64 {
    x - x * x * x * (6.0 + x * (3.0 * x - 8.0))
}

/// `(x^2 - 3x^3 + 3x^4 - x^5) / 2`: second derivative 1 at 0, and otherwise
/// flat to second order at both ends.
fn h2(x: f64) -> f64 {
    0.5 * x * x * (1.0 - x * (3.0 + x * (x - 3.0)))
}

/// `h0'`: `-30 x^2 (1 - x)^2`.
fn h0_slope(x: f64) -> f64 {
    -30.0 * x * x * (1.0 - x) * (1.0 - x)
}

/// `h1'`: `1 - 18x^2 + 32x^3 - 15x^4`.
fn h1_slope(x: f64) -> f64 {
    1.0 - x * x * (18.0 - x * (32.0 - 15.0 * x))
}

/// `h2'`: `x - 4.5x^2 + 6x^3 - 2.5x^4`.
fn h2_slope(x: f64) -> f64 {
    x * (1.0 - x * (4.5 - x * (6.0 - 2.5 * x)))
}

/// Move `edge` of `face` on to `target_edge`, keeping `continuity` with the
/// face that edge bounds, and return a new face. Neither input face changes.
///
/// For [`MatchContinuity::G0`] the target edge need bound no face. For the
/// others it must bound exactly one besides `face`; when it bounds more, as a
/// solid's edges do, [`match_surface_on`] names the one meant.
///
/// The source keeps its shape past a band beside the edge and is unchanged
/// there, to the fit's tolerance where its surface had to be refitted as a
/// polynomial spline. The band is as narrow as the size of the move allows:
/// a quarter of the surface's width for a small move, and all of it for one
/// large against that width. The surface's far side is always held, with its
/// tangent and curvature. The two sides next to the edge move with its ends.
///
/// The result runs in `u` along the edge and in `v` away from it, with the
/// edge at `v = 0`, and its normal agrees with the source's. Which end of the
/// target meets which end of the edge is whichever moves it least, and they
/// pair by arc length along each.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if `edge` is not on the outer
/// boundary of `face` or is its seam, `target_edge` is on `face`, the target
/// is closed and the edge open or the other way round, the target bounds no
/// face (or several) when one is needed, `edge` is the face's whole boundary
/// and winds round its surface with no hole inside it, the face's outline folds
/// when mapped to a rectangle, a hole in it comes too close to the edge to fit
/// a band between, or the move runs along the face farther than the band can
/// take without folding the face back over itself. A topology lookup can also
/// fail.
pub fn match_surface(
    topo: &mut Topology,
    face: FaceId,
    edge: EdgeId,
    target_edge: EdgeId,
    continuity: MatchContinuity,
) -> Result<FaceId, OperationsError> {
    check_edges(topo, face, edge, target_edge)?;
    let target_face = if continuity == MatchContinuity::G0 {
        None
    } else {
        Some(owner_of(topo, target_edge, face)?)
    };
    match_onto(topo, face, edge, target_edge, target_face, continuity)
}

/// As [`match_surface`], naming the face the target edge bounds.
///
/// # Errors
///
/// As [`match_surface`], and if `target_edge` is not on `target_face` or
/// `target_face` is `face`.
pub fn match_surface_on(
    topo: &mut Topology,
    face: FaceId,
    edge: EdgeId,
    target_edge: EdgeId,
    target_face: FaceId,
    continuity: MatchContinuity,
) -> Result<FaceId, OperationsError> {
    check_edges(topo, face, edge, target_edge)?;
    if !face_has_edge(topo, target_face, target_edge)? {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "edge {} is not on the face named for it",
                target_edge.index()
            ),
        });
    }
    match_onto(topo, face, edge, target_edge, Some(target_face), continuity)
}

fn face_has_edge(topo: &Topology, face: FaceId, edge: EdgeId) -> Result<bool, OperationsError> {
    let data = topo.face(face)?;
    for wire in std::iter::once(data.outer_wire()).chain(data.inner_wires().iter().copied()) {
        if topo.wire(wire)?.edges().iter().any(|oe| oe.edge() == edge) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The one face besides `not` that `edge` bounds.
fn owner_of(topo: &Topology, edge: EdgeId, not: FaceId) -> Result<FaceId, OperationsError> {
    let mut found = None;
    for (id, _) in topo.faces().iter() {
        if id != not && face_has_edge(topo, id, edge)? {
            if found.is_some() {
                return Err(OperationsError::InvalidInput {
                    reason: format!(
                        "edge {} bounds several faces, so a match to it has to name one",
                        edge.index()
                    ),
                });
            }
            found = Some(id);
        }
    }
    found.ok_or_else(|| OperationsError::InvalidInput {
        reason: format!(
            "edge {} bounds no other face, so there is no surface to match it to beyond G0",
            edge.index()
        ),
    })
}

fn check_edges(
    topo: &Topology,
    face: FaceId,
    edge: EdgeId,
    target_edge: EdgeId,
) -> Result<(), OperationsError> {
    if edge == target_edge {
        return Err(OperationsError::InvalidInput {
            reason: "an edge cannot be matched to itself".into(),
        });
    }
    if face_has_edge(topo, face, target_edge)? {
        return Err(OperationsError::InvalidInput {
            reason: "a surface cannot be matched to an edge of its own face".into(),
        });
    }
    Ok(())
}

fn match_onto(
    topo: &mut Topology,
    face: FaceId,
    edge: EdgeId,
    target_edge: EdgeId,
    target_face: Option<FaceId>,
    continuity: MatchContinuity,
) -> Result<FaceId, OperationsError> {
    let patch = Patch::of(topo, face, edge)?;
    let g = &patch.surface;

    let first = g.control_points()[0][0];
    let last = g.control_points()[g.control_points().len() - 1][0];
    let bottom_curve = NurbsCurve::new(
        g.degree_u(),
        g.knots_u().to_vec(),
        g.control_points().iter().map(|column| column[0]).collect(),
        vec![1.0; g.control_points().len()],
    )?;
    let bottom = Chain::from_curve(bottom_curve, first, last)?;

    // The target's chain, and with a surface wanted, the face it is on.
    let mut side = target_face
        .map(|target| Side::new(topo, &[target_edge], target))
        .transpose()?;
    let mut chain_alone = if side.is_none() {
        Some(Chain::from_edges(topo, &[target_edge])?)
    } else {
        None
    };
    if let Some(side) = &mut side {
        if bottom.closed != side.chain.closed {
            return Err(shape_mismatch());
        }
        if pair_chains(&bottom, &mut side.chain) {
            side.sign = -side.sign;
        }
    }
    if let Some(chain) = &mut chain_alone {
        if bottom.closed != chain.closed {
            return Err(shape_mismatch());
        }
        pair_chains(&bottom, chain);
    }
    let chain: &Chain = match (&side, &chain_alone) {
        (Some(side), _) => &side.chain,
        (None, Some(chain)) => chain,
        (None, None) => {
            return Err(OperationsError::InvalidInput {
                reason: "a match needs a target".into(),
            });
        }
    };

    // Breaks along the target, in arc-length fraction.
    let mut breaks_x = vec![0.0, 1.0];
    breaks_x.extend(chain.joints());
    breaks_x.extend(chain.knot_breaks());
    dedup_breaks(&mut breaks_x);
    if let Some(side) = &side {
        let crossings = side.surface_knot_breaks(&breaks_x)?;
        breaks_x.extend(crossings);
        dedup_breaks(&mut breaks_x);
    }
    let tilts = match &side {
        Some(side) => side.tilts(&breaks_x)?,
        None => vec![(0.0, 0.0); breaks_x.len() - 1],
    };

    let reads = Reads {
        g,
        bottom: &bottom,
        chain,
        side: side.as_ref(),
        breaks_x: &breaks_x,
        tilts: &tilts,
        continuity,
        own_tangent: true,
    };
    let reads = Reads {
        own_tangent: reads.source_tangent_serves()?,
        ..reads
    };

    let (mut largest, mut widest) = (0.0_f64, 0.0_f64);
    for column in g.control_points() {
        for p in column {
            largest = largest.max(p.x().abs()).max(p.y().abs()).max(p.z().abs());
            widest = widest.max((*p - first).length());
        }
    }
    let tolerance = row_tolerance(widest, largest);

    // Breaks in `s`: the target's, and where the source itself has only value
    // continuity.
    let q_s = g.degree_u().max(ALONG_DEGREE);
    let mut knots_s = elevated_knots(g.knots_u(), g.degree_u(), q_s);
    let mut breaks_s: Vec<f64> = breaks_x
        .iter()
        .map(|&x| bottom.parameter_of_fraction(x))
        .collect();
    breaks_s.extend(knots_s.iter().filter(|(_, m)| *m >= q_s).map(|(k, _)| *k));
    dedup_breaks(&mut breaks_s);
    // A break a hair from one of the source's knots lands on it: the span
    // between them could not be fitted, and nothing lost in so short a step
    // can show.
    for b in &mut breaks_s {
        match knots_s
            .iter_mut()
            .find(|(k, _)| (*k - *b).abs() < NEAR_KNOT)
        {
            Some(known) => {
                known.1 = q_s;
                *b = known.0;
            }
            None => knots_s.push((*b, q_s)),
        }
    }
    breaks_s.dedup();
    knots_s.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut ramps = Ramps::measure(&breaks_s, bottom.closed, tolerance, |_, lo, hi, s| {
        reads.displacements((lo, hi), s)
    })?;
    if ramps.needs_bands() {
        let bands = ramps.band_breaks(&breaks_s);
        for &b in &bands {
            knots_s.push((b, q_s));
        }
        knots_s.sort_by(|a, b| a.0.total_cmp(&b.0));
        breaks_s.extend(bands);
        dedup_breaks(&mut breaks_s);
        ramps = Ramps::measure(&breaks_s, bottom.closed, tolerance, |_, lo, hi, s| {
            reads.displacements((lo, hi), s)
        })?;
    }

    let rows = match continuity {
        MatchContinuity::G0 => 1,
        MatchContinuity::G1 => 2,
        MatchContinuity::G2 => 3,
    };
    let (s_space, mut moves) = fit_rows(
        Space::clamped(q_s, &knots_s),
        rows,
        tolerance,
        "a matched edge along its target",
        |s| {
            let piece = piece_of(&breaks_s, s);
            let (lo, hi) = (breaks_s[piece], breaks_s[piece + 1]);
            let mut values = reads.displacements((lo, hi), s)?;
            ramps.apply(piece, lo, hi, s, &mut values);
            Ok(values)
        },
    )?;
    if bottom.closed {
        for row in &mut moves {
            let start = row[0];
            let end = row.len() - 1;
            row[end] = start;
        }
    }

    let band = band_width(g, &moves, &s_space, patch.hole_clear)?;
    check_fade_keeps_direction(g, &moves, &s_space, band)?;
    let q_t = g.degree_v().max(5);
    let mut knots_t = elevated_knots(g.knots_v(), g.degree_v(), q_t);
    if band < 1.0 - 1e-9 {
        match knots_t.iter_mut().find(|(k, _)| (*k - band).abs() < 1e-9) {
            Some(known) => known.1 = known.1.max(q_t - 2),
            None => knots_t.push((band, q_t - 2)),
        }
        knots_t.sort_by(|a, b| a.0.total_cmp(&b.0));
    }
    let t_space = Space::clamped(q_t, &knots_t);

    let net = displaced_net(g, &s_space, &t_space, &moves, band)?;
    let net: Vec<Vec<Point3>> = net
        .into_iter()
        .map(|column| {
            column
                .into_iter()
                .map(|v| Point3::new(0.0, 0.0, 0.0) + v)
                .collect()
        })
        .collect();
    let surface = surface_from_net(&s_space, &t_space, net)?;

    let inner: Vec<WireId> = patch
        .holes
        .iter()
        .map(|&wire| crate::copy::copy_wire(topo, wire))
        .collect::<Result<_, _>>()?;
    boundary_face(
        topo,
        surface,
        &SideJoints {
            bottom: &[],
            right: &patch.right_joints,
            top: &patch.top_joints,
            left: &patch.left_joints,
        },
        bottom.closed,
        inner,
    )
}

fn shape_mismatch() -> OperationsError {
    OperationsError::InvalidInput {
        reason: "one of the edge and its target is closed and the other open".into(),
    }
}

/// What the displacement of the source's edge is at each point along it.
#[derive(Clone, Copy)]
struct Reads<'a> {
    g: &'a NurbsSurface,
    bottom: &'a Chain,
    chain: &'a Chain,
    side: Option<&'a Side>,
    breaks_x: &'a [f64],
    tilts: &'a [(f64, f64)],
    continuity: MatchContinuity,
    /// Whether the source's own cross-boundary tangent, flattened into the
    /// target's tangent plane, leaves the target's face as it should.
    own_tangent: bool,
}

impl Reads<'_> {
    /// The source's cross-boundary derivatives at `s`, read inside `piece`.
    fn across(&self, piece: (f64, f64), s: f64) -> (Vec3, Vec3, Vec3) {
        // A knot reads as the span it starts, and this piece may end at one.
        let read = if s >= piece.1 && piece.1 < 1.0 {
            piece.1 - INSIDE_PIECE
        } else {
            s
        };
        let d = self.g.derivatives(read, 0.0, 2);
        let point = self.g.evaluate(s, 0.0) - Point3::new(0.0, 0.0, 0.0);
        (point, d[0][1], d[0][2])
    }

    fn source_tangent_serves(&self) -> Result<bool, OperationsError> {
        let Some(side) = self.side else {
            return Ok(true);
        };
        for i in 0..=32 {
            let s = f64::from(i) / 32.0;
            let x = self.bottom.fraction_of_parameter(s);
            let piece = piece_of(self.breaks_x, x);
            let range = (self.breaks_x[piece], self.breaks_x[piece + 1]);
            let station = side.station(range, x)?;
            let (_, g_t, _) = self.across((0.0, 1.0), s);
            let flat = g_t - station.normal * g_t.dot(station.normal);
            let speed = g_t.length();
            if speed > 0.0 && flat.dot(station.off) < LEAVES_FACE * speed {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `D0`, `D1`, `D2` at `s`, as many as the continuity asks for.
    fn displacements(&self, piece: (f64, f64), s: f64) -> Result<Vec<Vec3>, OperationsError> {
        let x = self.bottom.fraction_of_parameter(s);
        let (position, g_t, g_tt) = self.across(piece, s);
        let Some(side) = self.side else {
            return Ok(vec![
                self.chain.point(x) - Point3::new(0.0, 0.0, 0.0) - position,
            ]);
        };
        let mid = 0.5 * (piece.0 + piece.1);
        let x_piece = piece_of(self.breaks_x, self.bottom.fraction_of_parameter(mid));
        let range = (self.breaks_x[x_piece], self.breaks_x[x_piece + 1]);
        let station = side.station(range, x)?;
        let d0 = station.point - Point3::new(0.0, 0.0, 0.0) - position;
        if self.continuity == MatchContinuity::G0 {
            return Ok(vec![d0]);
        }
        let v = if self.own_tangent {
            g_t - station.normal * g_t.dot(station.normal)
        } else {
            let along = (x - range.0) / (range.1 - range.0);
            tilted(station.off, station.tangent, self.tilts[x_piece], along) * g_t.length()
        };
        let d1 = v - g_t;
        if self.continuity == MatchContinuity::G1 {
            return Ok(vec![d0, d1]);
        }
        let w = g_tt + station.normal * (station.form.second_form(v) - station.normal.dot(g_tt));
        Ok(vec![d0, d1, w - g_tt])
    }
}

/// A spline's knots raised from degree `p` to `q`: `(value, multiplicity)`,
/// with continuity kept.
fn elevated_knots(knots: &[f64], p: usize, q: usize) -> Vec<(f64, usize)> {
    let mut found: Vec<(f64, usize)> = Vec::new();
    for &k in knots {
        match found.last_mut() {
            Some(last) if (last.0 - k).abs() < 1e-9 => last.1 += 1,
            _ => found.push((k, 1)),
        }
    }
    for entry in &mut found {
        entry.1 += q - p;
    }
    found
}

fn band_width(
    g: &NurbsSurface,
    moves: &[Vec<Vec3>],
    s_space: &Space,
    hole_clear: f64,
) -> Result<f64, OperationsError> {
    let mut ask = NARROWEST_BAND;
    for i in 0..=32 {
        let s = f64::from(i) / 32.0;
        let d = g.derivatives(s, 0.0, 1);
        let speed = d[0][1].length().max(1e-12);
        let d0 = s_space.evaluate(&moves[0], s).length();
        ask = ask.max(H0_SLOPE_PEAK * d0 / speed);
        if let Some(row) = moves.get(1) {
            let d1 = s_space.evaluate(row, s).length();
            ask = ask.max(H1_BEND_PEAK * d1 / speed);
        }
    }
    let band = if hole_clear < 1.0 {
        ask.min(0.9 * hole_clear)
    } else {
        ask.min(1.0)
    };
    if band < THINNEST_BAND {
        return Err(OperationsError::InvalidInput {
            reason: format!(
                "a hole in the face comes within {:.1}% of its width of the edge, \
                 leaving no room to fade the match out",
                100.0 * hole_clear
            ),
        });
    }
    Ok(band)
}

/// Refuse a move that the band cannot fade out without folding: across the
/// band its position term must not turn the surface's own way across, with
/// the tangent and curvature terms added, back on itself. A band held to the
/// surface's width, or by a hole, can be too narrow for a move along it.
fn check_fade_keeps_direction(
    g: &NurbsSurface,
    moves: &[Vec<Vec3>],
    s_space: &Space,
    band: f64,
) -> Result<(), OperationsError> {
    const ACROSS: u32 = 64;
    let zero = Vec3::new(0.0, 0.0, 0.0);
    for i in 0..=32 {
        let s = f64::from(i) / 32.0;
        let row = |k: usize| moves.get(k).map_or(zero, |row| s_space.evaluate(row, s));
        let (d0, d1, d2) = (row(0), row(1), row(2));
        for j in 1..ACROSS {
            let x = f64::from(j) / f64::from(ACROSS);
            let rest =
                g.derivatives(s, x * band, 1)[0][1] + d1 * h1_slope(x) + d2 * (band * h2_slope(x));
            let moved = rest + d0 * (h0_slope(x) / band);
            if moved.dot(rest) <= 0.0 {
                return Err(OperationsError::InvalidInput {
                    reason: format!(
                        "the move runs along the face farther than a band {:.0}% of its width \
                         can take, so it would fold the face back over itself",
                        100.0 * band
                    ),
                });
            }
        }
    }
    Ok(())
}

fn displaced_net(
    g: &NurbsSurface,
    s_space: &Space,
    t_space: &Space,
    moves: &[Vec<Vec3>],
    band: f64,
) -> Result<Vec<Vec<Vec3>>, OperationsError> {
    let s_sites = s_space.sites();
    let t_sites = t_space.sites();
    let mut grid = Vec::with_capacity(s_sites.len());
    for &s in &s_sites {
        let mut column = Vec::with_capacity(t_sites.len());
        for &t in &t_sites {
            column.push(g.evaluate(s, t) - Point3::new(0.0, 0.0, 0.0));
        }
        grid.push(column);
    }
    let mut net = solve_grid(s_space, t_space, grid)?;

    let along_t = t_space.collocation()?;
    let fade = |shape: &dyn Fn(f64) -> f64, scale: f64| -> Vec<Vec3> {
        let mut values: Vec<Vec3> = t_sites
            .iter()
            .map(|&t| {
                let x = t / band;
                Vec3::new(if x < 1.0 { scale * shape(x) } else { 0.0 }, 0.0, 0.0)
            })
            .collect();
        along_t.solve(&mut values);
        values
    };
    let shapes = [fade(&h0, 1.0), fade(&h1, band), fade(&h2, band * band)];
    for (row, shape) in moves.iter().zip(&shapes) {
        for (i, column) in net.iter_mut().enumerate() {
            for (j, entry) in column.iter_mut().enumerate() {
                *entry += row[i] * shape[j].x();
            }
        }
    }
    Ok(net)
}

/// A face read as a polynomial spline over a rectangle, the matched edge along
/// its `t = 0` side.
struct Patch {
    surface: NurbsSurface,
    right_joints: Vec<f64>,
    top_joints: Vec<f64>,
    left_joints: Vec<f64>,
    holes: Vec<WireId>,
    /// The least `t` of any hole, 1 if there are none.
    hole_clear: f64,
}

/// One side of the rectangle: its curve and its walk in `(u, v)`.
struct SideTrace {
    chain: Option<Chain>,
    trace: Vec<(f64, (f64, f64))>,
}

impl SideTrace {
    fn of(
        neighbour: &Neighbour,
        chain: Option<Chain>,
        from: (f64, f64),
    ) -> Result<Self, OperationsError> {
        let Some(chain) = chain else {
            return Ok(Self {
                chain: None,
                trace: vec![(0.0, from), (1.0, from)],
            });
        };
        let steps = 24 * chain.segments.len();
        let mut trace = Vec::with_capacity(steps + 1);
        let mut near = from;
        for k in 0..=steps {
            let x = k as f64 / steps as f64;
            let at = neighbour.unwrap_near(neighbour.locate(chain.point(x))?, near);
            let jet = neighbour.jet(at.0, at.1);
            // A pole's `u` is rounding; the side keeps the `u` it came with.
            let at = if jet.su.length_squared() < 1e-18 {
                (near.0, at.1)
            } else if jet.sv.length_squared() < 1e-18 {
                (at.0, near.1)
            } else {
                at
            };
            trace.push((x, at));
            near = at;
        }
        Ok(Self {
            chain: Some(chain),
            trace,
        })
    }

    fn start(&self) -> (f64, f64) {
        self.trace[0].1
    }

    fn end(&self) -> (f64, f64) {
        self.trace[self.trace.len() - 1].1
    }

    fn at(&self, neighbour: &Neighbour, x: f64) -> Result<(f64, f64), OperationsError> {
        let Some(chain) = &self.chain else {
            return Ok(self.start());
        };
        let k = ((x * (self.trace.len() - 1) as f64).round() as usize).min(self.trace.len() - 1);
        Ok(neighbour.unwrap_near(neighbour.locate(chain.point(x))?, self.trace[k].1))
    }

    fn reversed(self) -> Self {
        let mut trace: Vec<(f64, (f64, f64))> = self
            .trace
            .iter()
            .rev()
            .map(|&(x, uv)| (1.0 - x, uv))
            .collect();
        trace.sort_by(|a, b| a.0.total_cmp(&b.0));
        Self {
            chain: self.chain.map(|chain| chain.reversed()),
            trace,
        }
    }

    fn joints(&self) -> Vec<f64> {
        self.chain
            .as_ref()
            .map_or_else(Vec::new, Chain::edge_joints)
    }
}

impl Patch {
    fn of(topo: &Topology, face: FaceId, edge: EdgeId) -> Result<Self, OperationsError> {
        let data = topo.face(face)?;
        let neighbour = Neighbour::of(topo, face)?;
        let holes: Vec<WireId> = data.inner_wires().to_vec();
        let walk: Vec<(EdgeId, bool)> = topo
            .wire(data.outer_wire())?
            .edges()
            .iter()
            .map(|oe| (oe.edge(), oe.is_forward()))
            .collect();
        let uses = walk.iter().filter(|(id, _)| *id == edge).count();
        if uses != 1 {
            return Err(OperationsError::InvalidInput {
                reason: if uses == 0 {
                    format!(
                        "edge {} is not on the outer boundary of the face to move",
                        edge.index()
                    )
                } else {
                    format!(
                        "edge {} is a seam of the face to move, which lies on both sides of it",
                        edge.index()
                    )
                },
            });
        }
        let at = walk.iter().position(|(id, _)| *id == edge).unwrap_or(0);
        let mut turned = walk;
        turned.rotate_left(at);

        let sides = four_sides(topo, &turned);
        let bottom = SideTrace::of(
            &neighbour,
            Some(Chain::from_walk(topo, &turned[..1], sides.closed_across)?),
            neighbour.locate(chain_start(topo, turned[0])?)?,
        )?;
        let ruled = turned.len() == 1;
        let (right, top, left, holes) = if ruled {
            let (top, holes) = far_side_of_loop(topo, &neighbour, &bottom, holes)?;
            let right = SideTrace::of(&neighbour, None, bottom.end())?;
            let left = SideTrace::of(&neighbour, None, bottom.start())?;
            (right, top, left, holes)
        } else {
            let right = SideTrace::of(&neighbour, sides.right_chain(topo)?, bottom.end())?;
            let top_walk = SideTrace::of(&neighbour, sides.top_chain(topo)?, right.end())?;
            let left_walk = SideTrace::of(&neighbour, sides.left_chain(topo)?, top_walk.end())?;
            (right, top_walk.reversed(), left_walk.reversed(), holes)
        };

        let map = if ruled {
            None
        } else {
            Rectangle::read(&bottom, &right, &top, &left)
        };
        let scale = surface_extent(&neighbour, &bottom, &top);
        let tolerance = (1e-9 * scale.0.min(1.0)).max(1e-12 * scale.1);

        let rectangle_joint =
            |trace: &SideTrace, along_s: bool| -> Result<Vec<f64>, OperationsError> {
                let mut found = Vec::new();
                for x in trace.joints() {
                    let uv = trace.at(&neighbour, x)?;
                    found.push(match &map {
                        Some(rectangle) => rectangle.param(uv, along_s),
                        None => x,
                    });
                }
                Ok(found)
            };
        let top_joints = rectangle_joint(&top, true)?;
        let right_joints = rectangle_joint(&right, false)?;
        let left_joints = rectangle_joint(&left, false)?;

        let source_uv = |s: f64, t: f64| -> Result<(f64, f64), OperationsError> {
            if let Some(rectangle) = &map {
                return Ok(rectangle.uv(s, t));
            }
            if ruled {
                let (a, c) = (bottom.at(&neighbour, s)?, top.at(&neighbour, s)?);
                return Ok((a.0 + (c.0 - a.0) * t, a.1 + (c.1 - a.1) * t));
            }
            coons_uv(&neighbour, &bottom, &right, &top, &left, s, t)
        };
        if map.is_none() {
            check_no_fold(&source_uv)?;
        }

        let mut surface = if let Some(whole) = map
            .as_ref()
            .and_then(|rectangle| whole_polynomial_net(&neighbour, rectangle))
        {
            whole
        } else {
            let (s_breaks, t_breaks) = source_breaks(
                &neighbour,
                map.as_ref(),
                &top_joints,
                &right_joints,
                &left_joints,
            );
            let degree = 5;
            let (s_space, t_space, net) = fit_grid(
                Space::c0_over(degree, &s_breaks),
                Space::c0_over(degree, &t_breaks),
                tolerance,
                "the surface to move",
                |s, t| {
                    let (u, v) = source_uv(s, t)?;
                    Ok(neighbour.point(u, v) - Point3::new(0.0, 0.0, 0.0))
                },
            )?;
            let net: Vec<Vec<Point3>> = net
                .into_iter()
                .map(|column| {
                    column
                        .into_iter()
                        .map(|v| Point3::new(0.0, 0.0, 0.0) + v)
                        .collect()
                })
                .collect();
            surface_from_net(&s_space, &t_space, net)?
        };

        // The result keeps the source's facing by running `s` the way that
        // gives it.
        let middle = surface.derivatives(0.5, 0.5, 1);
        let at_middle = surface.evaluate(0.5, 0.5);
        let facing = neighbour.form_at(at_middle)?.logical;
        let flip = middle[1][0].cross(middle[0][1]).dot(facing) < 0.0;
        let (right_joints, left_joints, top_joints) = if flip {
            surface = mirror_s(&surface)?;
            (
                left_joints,
                right_joints,
                top_joints.iter().rev().map(|j| 1.0 - j).collect(),
            )
        } else {
            (right_joints, left_joints, top_joints)
        };

        let hole_clear = hole_clearance(topo, &neighbour, &holes, &source_uv)?;

        Ok(Self {
            surface,
            right_joints,
            top_joints,
            left_joints,
            holes,
            hole_clear,
        })
    }
}

fn chain_start(
    topo: &Topology,
    (edge, forward): (EdgeId, bool),
) -> Result<Point3, OperationsError> {
    let data = topo.edge(edge)?;
    let vertex = if forward { data.start() } else { data.end() };
    Ok(topo.vertex(vertex)?.point())
}

/// How a wire falls into the four sides of a rectangle, the first edge alone
/// on the bottom.
struct Sides {
    right: Vec<(EdgeId, bool)>,
    top: Vec<(EdgeId, bool)>,
    left: Vec<(EdgeId, bool)>,
    /// The left and right sides are the same edges, or there are none because
    /// the bottom is a loop: the surface closes across them.
    closed_across: bool,
}

impl Sides {
    fn right_chain(&self, topo: &Topology) -> Result<Option<Chain>, OperationsError> {
        walk_chain(topo, &self.right, false)
    }

    fn top_chain(&self, topo: &Topology) -> Result<Option<Chain>, OperationsError> {
        walk_chain(topo, &self.top, self.closed_across)
    }

    fn left_chain(&self, topo: &Topology) -> Result<Option<Chain>, OperationsError> {
        walk_chain(topo, &self.left, false)
    }
}

fn walk_chain(
    topo: &Topology,
    walk: &[(EdgeId, bool)],
    closed: bool,
) -> Result<Option<Chain>, OperationsError> {
    if walk.is_empty() {
        return Ok(None);
    }
    match Chain::from_walk(topo, walk, closed) {
        Ok(chain) => Ok(Some(chain)),
        Err(OperationsError::InvalidInput { .. }) => Ok(None),
        Err(other) => Err(other),
    }
}

fn four_sides(topo: &Topology, walk: &[(EdgeId, bool)]) -> Sides {
    let n = walk.len();
    if n <= 3 {
        let (right, top, left): (&[_], &[_], &[_]) = match n {
            1 => (&[], &[], &[]),
            2 => (&[], &walk[1..], &[]),
            _ => (&walk[1..2], &[], &walk[2..]),
        };
        return Sides {
            right: right.to_vec(),
            top: top.to_vec(),
            left: left.to_vec(),
            closed_across: n == 1,
        };
    }
    let (mut cut_a, mut cut_b) = (1, 2);
    if n > 4 {
        let mut turns = Vec::with_capacity(n);
        for k in 0..n {
            turns.push(turn_between(topo, walk[k], walk[(k + 1) % n]));
        }
        let mut best = -1.0;
        for a in 1..n - 2 {
            for b in a + 1..n - 1 {
                let sum = turns[a] + turns[b];
                if sum > best {
                    best = sum;
                    (cut_a, cut_b) = (a, b);
                }
            }
        }
    }
    let right = walk[1..=cut_a].to_vec();
    let top = walk[cut_a + 1..=cut_b].to_vec();
    let left = walk[cut_b + 1..].to_vec();
    let closed_across =
        right.len() == 1 && left.len() == 1 && right[0].0 == left[0].0 && right[0].1 != left[0].1;
    Sides {
        right,
        top,
        left,
        closed_across,
    }
}

/// The far side of a face whose outer boundary is one closed edge: its only
/// hole, turned to lie least twisted against the edge, or failing that a point
/// in the middle. Returns the holes left over.
fn far_side_of_loop(
    topo: &Topology,
    neighbour: &Neighbour,
    bottom: &SideTrace,
    holes: Vec<WireId>,
) -> Result<(SideTrace, Vec<WireId>), OperationsError> {
    if let &[hole] = holes.as_slice() {
        let walk: Vec<(EdgeId, bool)> = topo
            .wire(hole)?
            .edges()
            .iter()
            .map(|oe| (oe.edge(), oe.is_forward()))
            .collect();
        let mut far = Chain::from_walk(topo, &walk, true)?;
        if let Some(along) = &bottom.chain {
            pair_chains(along, &mut far);
        }
        let from = neighbour.unwrap_near(neighbour.locate(far.point(0.0))?, bottom.start());
        return Ok((SideTrace::of(neighbour, Some(far), from)?, Vec::new()));
    }
    let (pu, pv) = neighbour.periods();
    let (a, b) = (bottom.start(), bottom.end());
    let winds = pu.is_some_and(|p| (b.0 - a.0).abs() > 0.5 * p)
        || pv.is_some_and(|p| (b.1 - a.1).abs() > 0.5 * p);
    if winds {
        return Err(OperationsError::InvalidInput {
            reason: "the edge to move winds round its surface, with no hole to be its far side"
                .into(),
        });
    }
    let once_round = &bottom.trace[..bottom.trace.len() - 1];
    let count = once_round.len() as f64;
    let middle = once_round
        .iter()
        .fold((0.0, 0.0), |sum, &(_, uv)| (sum.0 + uv.0, sum.1 + uv.1));
    let middle = (middle.0 / count, middle.1 / count);
    Ok((SideTrace::of(neighbour, None, middle)?, holes))
}

fn turn_between(topo: &Topology, a: (EdgeId, bool), b: (EdgeId, bool)) -> f64 {
    let tangent =
        |(edge, forward): (EdgeId, bool), at_end: bool| -> Result<Vec3, OperationsError> {
            let chain = Chain::from_walk(topo, &[(edge, forward)], false)?;
            let s = if at_end { 1.0 } else { 0.0 };
            Ok(chain.point_and_tangent((0.0, 1.0), s)?.1)
        };
    let (Ok(from), Ok(to)) = (tangent(a, true), tangent(b, false)) else {
        return 0.0;
    };
    from.dot(to).clamp(-1.0, 1.0).acos()
}

/// A face whose sides all lie along parameter lines of its surface.
struct Rectangle {
    u0: f64,
    u1: f64,
    v0: f64,
    v1: f64,
    /// `s` runs along `v` and `t` along `u`.
    swapped: bool,
}

impl Rectangle {
    fn read(
        bottom: &SideTrace,
        right: &SideTrace,
        top: &SideTrace,
        left: &SideTrace,
    ) -> Option<Self> {
        let reach = 1e-7;
        let flat = |trace: &SideTrace, read: usize| {
            let value = |uv: (f64, f64)| if read == 0 { uv.0 } else { uv.1 };
            let first = value(trace.start());
            trace
                .trace
                .iter()
                .all(|&(_, uv)| (value(uv) - first).abs() <= reach)
        };
        let swapped = flat(bottom, 0);
        if !swapped && !flat(bottom, 1) {
            return None;
        }
        let (along, across) = if swapped { (1, 0) } else { (0, 1) };
        if !flat(top, across) || !flat(left, along) || !flat(right, along) {
            return None;
        }
        let (a, b) = (bottom.start(), bottom.end());
        let c = left.end();
        if (top.start().0 - c.0).abs() > reach || (top.start().1 - c.1).abs() > reach {
            return None;
        }
        let d = top.end();
        if (right.end().0 - d.0).abs() > reach || (right.end().1 - d.1).abs() > reach {
            return None;
        }
        Some(if swapped {
            Self {
                u0: a.0,
                u1: c.0,
                v0: a.1,
                v1: b.1,
                swapped,
            }
        } else {
            Self {
                u0: a.0,
                u1: b.0,
                v0: a.1,
                v1: c.1,
                swapped,
            }
        })
    }

    fn uv(&self, s: f64, t: f64) -> (f64, f64) {
        let (x, y) = if self.swapped { (t, s) } else { (s, t) };
        (
            self.u0 + (self.u1 - self.u0) * x,
            self.v0 + (self.v1 - self.v0) * y,
        )
    }

    /// The `s` (or `t`) of a parameter point.
    fn param(&self, uv: (f64, f64), along_s: bool) -> f64 {
        let x = (uv.0 - self.u0) / (self.u1 - self.u0);
        let y = (uv.1 - self.v0) / (self.v1 - self.v0);
        if along_s == self.swapped { y } else { x }
    }
}

fn coons_uv(
    neighbour: &Neighbour,
    bottom: &SideTrace,
    right: &SideTrace,
    top: &SideTrace,
    left: &SideTrace,
    s: f64,
    t: f64,
) -> Result<(f64, f64), OperationsError> {
    let a = bottom.at(neighbour, s)?;
    let c = top.at(neighbour, s)?;
    let d = left.at(neighbour, t)?;
    let b = right.at(neighbour, t)?;
    let (p00, p10) = (bottom.start(), bottom.end());
    let (p01, p11) = (top.start(), top.end());
    let blend = |pick: fn((f64, f64)) -> f64| {
        (1.0 - t) * pick(a) + t * pick(c) + (1.0 - s) * pick(d) + s * pick(b)
            - (1.0 - s) * (1.0 - t) * pick(p00)
            - s * (1.0 - t) * pick(p10)
            - (1.0 - s) * t * pick(p01)
            - s * t * pick(p11)
    };
    Ok((blend(|p| p.0), blend(|p| p.1)))
}

fn check_no_fold<F>(uv: &F) -> Result<(), OperationsError>
where
    F: Fn(f64, f64) -> Result<(f64, f64), OperationsError>,
{
    const GRID: usize = 16;
    const STEP: f64 = 1e-3;
    let mut sign = 0.0;
    for i in 0..=GRID {
        for j in 0..=GRID {
            let s = (i as f64 / GRID as f64).clamp(STEP, 1.0 - STEP);
            let t = (j as f64 / GRID as f64).clamp(STEP, 1.0 - STEP);
            let (a, b, c, d) = (
                uv(s - STEP, t)?,
                uv(s + STEP, t)?,
                uv(s, t - STEP)?,
                uv(s, t + STEP)?,
            );
            let jacobian = (b.0 - a.0) * (d.1 - c.1) - (b.1 - a.1) * (d.0 - c.0);
            if jacobian.abs() < 1e-14 {
                continue;
            }
            if sign == 0.0 {
                sign = jacobian.signum();
            } else if jacobian * sign < 0.0 {
                return Err(OperationsError::InvalidInput {
                    reason: "the face to move folds when mapped to a rectangle, its outline \
                             too far from four convex sides"
                        .into(),
                });
            }
        }
    }
    Ok(())
}

/// A polynomial spline surface whose rectangle is its whole domain, with its
/// net turned to run `s` along the edge and `t` away from it.
fn whole_polynomial_net(neighbour: &Neighbour, map: &Rectangle) -> Option<NurbsSurface> {
    let FaceSurface::Nurbs(surface) = &neighbour.surface else {
        return None;
    };
    let clamped = |knots: &[f64], degree: usize| {
        knots[degree] - knots[0] <= 0.0
            && knots[knots.len() - 1] - knots[knots.len() - degree - 1] <= 0.0
    };
    if surface.is_rational()
        || !clamped(surface.knots_u(), surface.degree_u())
        || !clamped(surface.knots_v(), surface.degree_v())
    {
        return None;
    }
    let (du0, du1) = surface.domain_u();
    let (dv0, dv1) = surface.domain_v();
    let close = |a: f64, b: f64| (a - b).abs() <= 1e-7;
    let covers = |a: f64, b: f64, lo: f64, hi: f64| {
        (close(a, lo) && close(b, hi)) || (close(a, hi) && close(b, lo))
    };
    if !covers(map.u0, map.u1, du0, du1) || !covers(map.v0, map.v1, dv0, dv1) {
        return None;
    }
    let forward_u = map.u1 > map.u0;
    let forward_v = map.v1 > map.v0;
    let knots_of = |knots: &[f64], from: f64, to: f64| -> Vec<f64> {
        let mut mapped: Vec<f64> = knots.iter().map(|k| (k - from) / (to - from)).collect();
        mapped.sort_by(f64::total_cmp);
        mapped
    };
    let knots_u = knots_of(surface.knots_u(), map.u0, map.u1);
    let knots_v = knots_of(surface.knots_v(), map.v0, map.v1);
    let net = surface.control_points();
    let (n, m) = (net.len(), net[0].len());
    let read = |i: usize, j: usize| {
        net[if forward_u { i } else { n - 1 - i }][if forward_v { j } else { m - 1 - j }]
    };
    let (degree_s, degree_t, knots_s, knots_t, turned): (
        usize,
        usize,
        Vec<f64>,
        Vec<f64>,
        Vec<Vec<Point3>>,
    ) = if map.swapped {
        (
            surface.degree_v(),
            surface.degree_u(),
            knots_v,
            knots_u,
            (0..m)
                .map(|j| (0..n).map(|i| read(i, j)).collect())
                .collect(),
        )
    } else {
        (
            surface.degree_u(),
            surface.degree_v(),
            knots_u,
            knots_v,
            (0..n)
                .map(|i| (0..m).map(|j| read(i, j)).collect())
                .collect(),
        )
    };
    let weights = vec![vec![1.0; turned[0].len()]; turned.len()];
    NurbsSurface::new(degree_s, degree_t, knots_s, knots_t, turned, weights).ok()
}

/// The breaks a refit of the source needs: its sides' joints, and on a
/// rectangle, the lines where its own spline surface loses smoothness.
fn source_breaks(
    neighbour: &Neighbour,
    map: Option<&Rectangle>,
    top_joints: &[f64],
    right_joints: &[f64],
    left_joints: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut s_breaks = vec![0.0, 1.0];
    let mut t_breaks = vec![0.0, 1.0];
    s_breaks.extend_from_slice(top_joints);
    t_breaks.extend_from_slice(right_joints);
    t_breaks.extend_from_slice(left_joints);
    if let (Some(rectangle), FaceSurface::Nurbs(surface)) = (map, &neighbour.surface) {
        let (across_u, across_v): (&mut Vec<f64>, &mut Vec<f64>) = if rectangle.swapped {
            (&mut t_breaks, &mut s_breaks)
        } else {
            (&mut s_breaks, &mut t_breaks)
        };
        across_u.extend(
            surface
                .knots_u()
                .iter()
                .map(|k| (k - rectangle.u0) / (rectangle.u1 - rectangle.u0)),
        );
        across_v.extend(
            surface
                .knots_v()
                .iter()
                .map(|k| (k - rectangle.v0) / (rectangle.v1 - rectangle.v0)),
        );
    }
    dedup_breaks(&mut s_breaks);
    dedup_breaks(&mut t_breaks);
    (s_breaks, t_breaks)
}

/// The largest distance across the two long sides, and the largest coordinate
/// on them.
fn surface_extent(neighbour: &Neighbour, bottom: &SideTrace, top: &SideTrace) -> (f64, f64) {
    let (mut widest, mut largest) = (0.0_f64, 0.0_f64);
    for trace in [bottom, top] {
        for &(_, (u, v)) in &trace.trace {
            let p = neighbour.point(u, v);
            largest = largest.max(p.x().abs()).max(p.y().abs()).max(p.z().abs());
            let at = neighbour.point(bottom.start().0, bottom.start().1);
            widest = widest.max((p - at).length());
        }
    }
    (widest.max(1e-6), largest)
}

fn mirror_s(surface: &NurbsSurface) -> Result<NurbsSurface, OperationsError> {
    let mut knots: Vec<f64> = surface.knots_u().iter().map(|k| 1.0 - k).collect();
    knots.reverse();
    let net: Vec<Vec<Point3>> = surface.control_points().iter().rev().cloned().collect();
    let weights = surface.weights().iter().rev().cloned().collect();
    Ok(NurbsSurface::new(
        surface.degree_u(),
        surface.degree_v(),
        knots,
        surface.knots_v().to_vec(),
        net,
        weights,
    )?)
}

/// The least `t` of any point on a hole's wire, by Newton on the map from the
/// rectangle.
fn hole_clearance<F>(
    topo: &Topology,
    neighbour: &Neighbour,
    holes: &[WireId],
    uv: &F,
) -> Result<f64, OperationsError>
where
    F: Fn(f64, f64) -> Result<(f64, f64), OperationsError>,
{
    let mut least = 1.0_f64;
    for &hole in holes {
        let center = uv(0.5, 0.5)?;
        let trace = neighbour.chart_loop(topo, hole)?;
        for &point in &trace {
            let target = neighbour.unwrap_near(point, center);
            let (mut s, mut t) = (0.5, 0.5);
            for _ in 0..12 {
                let (a, b, c, d) = (
                    uv(f64::max(s - 1e-4, 0.0), t)?,
                    uv(f64::min(s + 1e-4, 1.0), t)?,
                    uv(s, f64::max(t - 1e-4, 0.0))?,
                    uv(s, f64::min(t + 1e-4, 1.0))?,
                );
                let (ds, dt) = (2e-4, 2e-4);
                let (j00, j10) = ((b.0 - a.0) / ds, (b.1 - a.1) / ds);
                let (j01, j11) = ((d.0 - c.0) / dt, (d.1 - c.1) / dt);
                let det = j00 * j11 - j01 * j10;
                if det.abs() < 1e-14 {
                    break;
                }
                let at = uv(s, t)?;
                let (rx, ry) = (target.0 - at.0, target.1 - at.1);
                s = (s + (j11 * rx - j01 * ry) / det).clamp(0.0, 1.0);
                t = (t + (j00 * ry - j10 * rx) / det).clamp(0.0, 1.0);
            }
            least = least.min(t);
        }
    }
    Ok(least)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use super::*;

    #[test]
    fn hermite_slopes_are_the_shapes_derivatives() {
        let h = 1e-6;
        for i in 0..=20 {
            let x = f64::from(i) / 20.0;
            for (shape, slope) in [
                (h0 as fn(f64) -> f64, h0_slope as fn(f64) -> f64),
                (h1, h1_slope),
                (h2, h2_slope),
            ] {
                let want = (shape(x + h) - shape(x - h)) / (2.0 * h);
                assert!((slope(x) - want).abs() < 1e-8, "{x}: {} {want}", slope(x));
            }
        }
    }

    #[test]
    fn hermite_shapes_have_their_end_conditions() {
        let h = 1e-5;
        let first = |f: fn(f64) -> f64, x: f64| (f(x + h) - f(x - h)) / (2.0 * h);
        let second = |f: fn(f64) -> f64, x: f64| (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h);
        assert!((h0(0.0) - 1.0).abs() < 1e-12);
        assert!(h0(1.0).abs() < 1e-12);
        assert!(first(h0, 0.0).abs() < 1e-6 && second(h0, 0.0).abs() < 1e-4);
        assert!(first(h0, 1.0).abs() < 1e-6 && second(h0, 1.0).abs() < 1e-4);

        assert!(h1(0.0).abs() < 1e-12 && h1(1.0).abs() < 1e-12);
        assert!((first(h1, 0.0) - 1.0).abs() < 1e-6 && second(h1, 0.0).abs() < 1e-4);
        assert!(first(h1, 1.0).abs() < 1e-6 && second(h1, 1.0).abs() < 1e-4);

        assert!(h2(0.0).abs() < 1e-12 && h2(1.0).abs() < 1e-12);
        assert!(first(h2, 0.0).abs() < 1e-6 && (second(h2, 0.0) - 1.0).abs() < 1e-4);
        assert!(first(h2, 1.0).abs() < 1e-6 && second(h2, 1.0).abs() < 1e-4);
    }
}

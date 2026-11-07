//! Lofts through a network of profiles and guide rails, leaving their end
//! profiles under an end condition.
//!
//! One face runs along each pair of paired profile edges, across every
//! profile. Writing `u` along the edges and `v` across the profiles, with `v`
//! taking a value `v_s` at each profile, the face is the boolean sum
//!
//! ```text
//! S(u, v) = P(u, v) + (1 - u) D_a(v) + u D_b(v)
//! ```
//!
//! `P` is the skin: the tensor-product spline through the profile curves at
//! the `v_s`, its derivative across an end profile the condition there. `D_a`
//! and `D_b` are what the rails at the edge's two ends add to it, each rail
//! less the spline of the skin's space through the rail's own points at the
//! `v_s` (and its own derivative at a conditioned end). That is the Gordon
//! surface `S_profiles + S_rails - S_tensor`, with the deviations vanishing at
//! every `v_s`: at `u = 0` and `u = 1` the sum is the rail, and at `v = v_s`
//! it is the profile. A vertex no rail runs through takes its neighbouring
//! rails' deviations blended by the distance round the profile between them.
//!
//! The sum is taken on control nets, so it is exact for polynomial curves. A
//! rational curve in `u` stays rational when every profile's edge has its
//! weights; the sum then raises `u`'s degree by one and multiplies the
//! deviation through by the weight function. Rational rails, and rational
//! edges paired with different weights, are replaced by polynomial splines
//! within [`POLYNOMIAL_TOLERANCE`], since the weights of a sum of rational
//! curves are the product of theirs.

use brepkit_geometry::extrema::segment_segment_distance;
use brepkit_math::nurbs::basis::{basis_funs, ders_basis_funs, find_span};
use brepkit_math::nurbs::curve::NurbsCurve;
use brepkit_math::nurbs::knot_ops::{curve_knot_insert, curve_split};
use brepkit_math::nurbs::projection::project_point_to_curve;
use brepkit_math::nurbs::surface::NurbsSurface;
use brepkit_math::tolerance::Tolerance;
use brepkit_math::vec::{Point3, Vec3};
use brepkit_topology::Topology;
use brepkit_topology::edge::{Edge, EdgeCurve, EdgeId};
use brepkit_topology::face::{Face, FaceId, FaceSurface};
use brepkit_topology::shell::Shell;
use brepkit_topology::solid::{Solid, SolidId};
use brepkit_topology::vertex::Vertex;
use brepkit_topology::wire::{OrientedEdge, Wire, WireId};

use super::{ProfileEdgeGeom, edge_distance, mean_chord_params, wire_oriented_edges};
use crate::{OperationsError, dot_normal_point};

/// How far a rail may pass from a profile's vertex and still run through it.
pub const RAIL_VERTEX_TOLERANCE: f64 = 1e-7;

/// How far a profile may stand from a `tangent` face and still lie on it.
pub const TANGENT_FACE_TOLERANCE: f64 = 1e-6;

/// How far off an end condition, in radians, a rail may leave its profile.
pub const RAIL_CONDITION_ANGLE: f64 = 1e-4;

/// How far a rational curve's polynomial stand-in may pass from it.
pub const POLYNOMIAL_TOLERANCE: f64 = 1e-8;

/// The most a surface's derivative across an end may lean off the tangent
/// plane it is fitted to, in radians, wherever the plane turns along the edge.
pub const TANGENT_FIT_ANGLE: f64 = 1e-6;

/// The degree of a rational curve's polynomial stand-in: its error falls with
/// the eighth power of the span, so an arc needs few spans at this tolerance.
const POLYNOMIAL_DEGREE: usize = 7;

/// How much a fitted end derivative may differ from the one asked for, as a
/// fraction of its size, in the magnitude and direction the condition leaves
/// free.
const FIELD_SHAPE_TOLERANCE: f64 = 1e-3;

/// How sharply, in radians, the surface must turn across a full knot for the
/// knot to count as a crease.
const CREASE_ANGLE: f64 = 1e-3;

/// The widest gap a wire's consecutive edges may leave and still connect, as
/// a sweep's path reads them.
const WIRE_GAP: f64 = 1e-5;

/// Knots within this of each other on a unit domain are one.
const KNOT_MERGE: f64 = 1e-9;

/// Vertices closer than this share a place along a run of edges, as a
/// fraction of its length.
const POSITION_MERGE: f64 = 1e-6;

/// A curve the section vertices follow from one profile to the next.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoftRail {
    /// A rail of one edge.
    Edge(EdgeId),
    /// A rail of a wire's chain of edges.
    Wire(WireId),
}

/// How a loft's surface leaves its first or last profile.
///
/// A `weight` scales the condition's derivative from its natural size, the
/// chord to the matching vertex of the next profile per unit of the surface's
/// parameter. A negative weight leaves to the other side of the profile. A
/// rail fixes the derivative at its own vertex, so there the weight applies to
/// the span between rails.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LoftEnd {
    /// The surface takes whatever direction the profiles give it.
    #[default]
    Free,
    /// Square to the profile's plane, heading toward the next profile. With the
    /// next profile in that plane, it leaves along the profile face's own normal.
    Normal {
        /// Scale on the derivative's natural size; finite and nonzero.
        weight: f64,
    },
    /// Within the tangent plane of a face the profile lies on, square to the
    /// profile: the direction of the profile's offset, which at a corner is
    /// the mitre between the corner's two edges. It heads toward the next
    /// profile, and with the next profile no nearer one side than the other,
    /// away from the profile's inside.
    Tangent {
        /// The face the profile lies on.
        face: FaceId,
        /// Scale on the derivative's natural size; finite and nonzero.
        weight: f64,
    },
}

impl LoftEnd {
    const fn is_free(&self) -> bool {
        matches!(self, Self::Free)
    }

    const fn weight(&self) -> Option<f64> {
        match *self {
            Self::Free => None,
            Self::Normal { weight } | Self::Tangent { weight, .. } => Some(weight),
        }
    }
}

/// What [`loft_with_options`] lofts along and how it leaves its ends.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LoftOptions {
    /// Rails, each through one vertex of every profile.
    pub rails: Vec<LoftRail>,
    /// How the surface leaves the first profile.
    pub start: LoftEnd,
    /// How the surface leaves the last profile.
    pub end: LoftEnd,
}

/// Loft `profiles` along rails and under end conditions.
///
/// Without rails or conditions this is [`super::loft_smooth`]. With them, one
/// face runs along each edge of the profiles, passing through every profile
/// and every rail, and vertices pair up along the rails: between two rails,
/// profiles of equal edge counts pair edge by edge, and others are cut at one
/// another's vertices by distance along the run, which leaves every profile's
/// curves in place. Without rails, the profiles pair from their first vertices
/// the same way. Holes in profiles are ignored, as [`super::loft`] ignores
/// them. A closed profile edge has one vertex for a rail, at its seam.
///
/// A rail at a conditioned end leaves the condition's direction or is refused,
/// since the surface's edge is the rail. Where a rail kinks, or where rails
/// meet an intermediate profile at different fractions of their parameters, the
/// surface passes through them but its derivative jumps: the faces end there,
/// on an edge along the crease.
///
/// # Errors
///
/// Returns [`OperationsError::InvalidInput`] if there are fewer than two
/// profiles; a rail misses a vertex of a profile by more than
/// [`RAIL_VERTEX_TOLERANCE`], meets one profile at two vertices, or runs
/// through the profiles out of order; rails outnumber a profile's vertices,
/// share a vertex, or come round the profiles in different orders; a rail
/// leaves a conditioned end more than [`RAIL_CONDITION_ANGLE`] off the
/// condition; a weight is zero or not finite; a `normal` profile is not planar;
/// a `tangent` profile stands off its face by more than
/// [`TANGENT_FACE_TOLERANCE`] or a corner of it doubles back; the profiles do
/// not say which way they wind; or a line across a side, or the loop round the
/// sides at one parameter, meets itself, as a weight too large for the distance
/// to the next profile makes it. Sides that pass through each other otherwise
/// are not looked for.
#[allow(clippy::too_many_lines)]
pub fn loft_with_options(
    topo: &mut Topology,
    profiles: &[FaceId],
    options: &LoftOptions,
) -> Result<SolidId, OperationsError> {
    if options.rails.is_empty() && options.start.is_free() && options.end.is_free() {
        return super::loft_smooth(topo, profiles);
    }
    if profiles.len() < 2 {
        return Err(refuse("loft requires at least 2 profiles"));
    }
    for end in [&options.start, &options.end] {
        if let Some(weight) = end.weight()
            && (!weight.is_finite() || weight == 0.0)
        {
            return Err(refuse(format!(
                "an end condition's weight must be finite and nonzero, got {weight}"
            )));
        }
    }

    let mut rings = Vec::with_capacity(profiles.len());
    let mut corners = Vec::with_capacity(profiles.len());
    for &face in profiles {
        let (ring, vertices) = read_ring(topo, face)?;
        rings.push(ring);
        corners.push(vertices);
    }
    let rails = read_rails(topo, &options.rails, &corners)?;

    orient_rings(topo, &mut rings, &rails, profiles, options)?;
    if rails.is_empty() {
        align_lone_seams(&mut rings, &corners)?;
    } else {
        for (s, ring) in rings.iter_mut().enumerate() {
            let at = ring_index(ring, rails[0].through[s])?;
            ring.rotate_left(at);
        }
    }
    let rail_at_vertex = equalize_rings(&mut rings, &rails)?;
    let n = rings[0].len();
    let last = rings.len() - 1;

    let params = section_params(&rings, &rails)?;
    let skin = SkinSpace::new(params, !options.start.is_free(), !options.end.is_free())?;
    let rail_curves = rails
        .iter()
        .map(|rail| rail.on_params(&skin.params))
        .collect::<Result<Vec<_>, _>>()?;

    let spans: Vec<Vec<NurbsCurve>> = rings
        .iter()
        .map(|ring| ring.iter().map(unit_span).collect())
        .collect::<Result<_, _>>()?;
    let rail_end_tangent = |rail_index: Option<usize>, at: f64| -> Option<Vec3> {
        rail_index.map(|j| rail_curves[j].derivatives(at, 1)[1])
    };

    let first_field = end_field(
        topo,
        &EndRequest {
            condition: options.start,
            profile: profiles[0],
            ring: &rings[0],
            near_ring: &rings[1],
            spans: &spans[0],
            step: skin.params[1] - skin.params[0],
            along: 1.0,
            rail_tangents: &(0..n)
                .map(|i| rail_end_tangent(rail_at_vertex[i], 0.0))
                .collect::<Vec<_>>(),
        },
    )?;
    let last_field = end_field(
        topo,
        &EndRequest {
            condition: options.end,
            profile: profiles[last],
            ring: &rings[last],
            near_ring: &rings[last - 1],
            spans: &spans[last],
            step: skin.params[last] - skin.params[last - 1],
            along: -1.0,
            rail_tangents: &(0..n)
                .map(|i| rail_end_tangent(rail_at_vertex[i], 1.0))
                .collect::<Vec<_>>(),
        },
    )?;

    // The rails' deviations from the skin, in the space that holds them both.
    let mut spaces: Vec<(usize, &[f64])> = vec![(skin.degree, skin.knots.as_slice())];
    spaces.extend(rail_curves.iter().map(|c| (c.degree(), c.knots())));
    let merged = merged_knots(&spaces, 0);
    let scale = rings
        .iter()
        .flatten()
        .map(|e| (e.start - rings[0][0].start).length())
        .fold(0.0, f64::max)
        .max(1.0);
    let rail_deviations = rail_curves
        .iter()
        .map(|curve| rail_deviation(curve, &skin, &merged))
        .collect::<Result<Vec<_>, _>>()?;
    let any_deviation = rail_deviations
        .iter()
        .flatten()
        .any(|d| d.iter().any(|x| x.abs() > 1e-11 * scale));
    let (v_degree, v_knots) = if any_deviation {
        merged
    } else {
        (skin.degree, skin.knots.clone())
    };
    let deviations =
        any_deviation.then(|| vertex_deviations(&rings, &rail_at_vertex, &rail_deviations));

    // The vertical rows both faces at a vertex share, so that they meet on
    // identical control points.
    let mut rows = Vec::with_capacity(n);
    for i in 0..n {
        let points: Vec<[f64; 3]> = rings.iter().map(|ring| arr(ring[i].start)).collect();
        let t0 = first_field.as_ref().map(|f| vec_arr(f.corners[i]));
        let t1 = last_field.as_ref().map(|f| vec_arr(f.corners[i]));
        let coefs = skin.interpolate(&points, t0, t1);
        let mut row = represent((skin.degree, &skin.knots), &coefs, (v_degree, &v_knots))?;
        if let Some(deviations) = &deviations {
            for (c, d) in row.iter_mut().zip(&deviations[i]) {
                *c = add(*c, *d);
            }
        }
        rows.push(row);
    }

    let mut sides = Vec::with_capacity(n);
    for i in 0..n {
        let edge_spans: Vec<NurbsCurve> = spans.iter().map(|ring| ring[i].clone()).collect();
        sides.push(side_net(&SideRequest {
            edge: i,
            spans: &edge_spans,
            start: first_field.as_ref(),
            end: last_field.as_ref(),
            skin: &skin,
            v_space: (v_degree, &v_knots),
            rows: &rows,
            deviations: deviations.as_deref(),
        })?);
    }
    let whole_sides = sides
        .iter()
        .map(|side| side.surface(0, side.net[0].len() - 1, v_degree, &v_knots))
        .collect::<Result<Vec<_>, _>>()?;
    if whole_sides
        .iter()
        .any(|s| doubles_back(s) || turns_inside_out(s))
        || sections_cross(&whole_sides)
    {
        return Err(refuse(
            "the loft's surface runs back over itself: a weight or a rail carries it past where it comes back through",
        ));
    }
    let bands = v_bands(v_degree, &v_knots, &sides);
    let mut band_surfaces = Vec::with_capacity(bands.len());
    for band in &bands {
        let surfaces: Vec<FaceSurface> = sides
            .iter()
            .map(|side| side.band_surface(band, v_degree))
            .collect::<Result<_, _>>()?;
        band_surfaces.push(surfaces);
    }

    // A ring of vertices and edges stands at each end profile and between each
    // two bands; a vertical edge runs up each band at each vertex.
    let tol = Tolerance::new().linear;
    let level_points = |level: usize, i: usize| -> Point3 {
        if level == 0 {
            rings[0][i].start
        } else if level == bands.len() {
            rings[last][i].start
        } else {
            point(rows[i][bands[level].rows.0])
        }
    };
    let mut level_vertices = Vec::with_capacity(bands.len() + 1);
    for level in 0..=bands.len() {
        let vertices: Vec<_> = (0..n)
            .map(|i| topo.add_vertex(Vertex::new(level_points(level, i), tol)))
            .collect();
        level_vertices.push(vertices);
    }
    let mut level_edges: Vec<Vec<EdgeId>> = Vec::with_capacity(bands.len() + 1);
    for (level, vertices) in level_vertices.iter().enumerate() {
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let curve = if level == 0 {
                rings[0][i].curve.clone()
            } else if level == bands.len() {
                rings[last][i].curve.clone()
            } else {
                sides[i].across(bands[level].rows.0)?
            };
            edges.push(topo.add_edge(Edge::new(vertices[i], vertices[(i + 1) % n], curve)));
        }
        level_edges.push(edges);
    }
    let mut band_rails: Vec<Vec<EdgeId>> = Vec::with_capacity(bands.len());
    for (level, band) in bands.iter().enumerate() {
        let mut edges = Vec::with_capacity(n);
        for i in 0..n {
            let curve = row_curve(&rows[i][band.rows.0..=band.rows.1], v_degree, &band.knots)?;
            edges.push(topo.add_edge(Edge::new(
                level_vertices[level][i],
                level_vertices[level + 1][i],
                curve,
            )));
        }
        band_rails.push(edges);
    }

    let mut faces = Vec::with_capacity(n * bands.len() + 2);
    faces.push(cap_face(
        topo,
        &level_edges[0],
        &rings[0],
        profiles[0],
        true,
    )?);
    for (level, surfaces) in band_surfaces.into_iter().enumerate() {
        for (i, surface) in surfaces.into_iter().enumerate() {
            let next = (i + 1) % n;
            let wire = topo.add_wire(Wire::new(
                vec![
                    OrientedEdge::new(level_edges[level][i], true),
                    OrientedEdge::new(band_rails[level][next], true),
                    OrientedEdge::new(level_edges[level + 1][i], false),
                    OrientedEdge::new(band_rails[level][i], false),
                ],
                true,
            )?);
            faces.push(topo.add_face(Face::new(wire, vec![], surface)));
        }
    }
    faces.push(cap_face(
        topo,
        &level_edges[bands.len()],
        &rings[last],
        profiles[last],
        false,
    )?);

    let shell = topo.add_shell(Shell::new(faces)?);
    Ok(topo.add_solid(Solid::new(shell, vec![])))
}

fn refuse(reason: impl Into<String>) -> OperationsError {
    OperationsError::InvalidInput {
        reason: reason.into(),
    }
}

const fn arr(p: Point3) -> [f64; 3] {
    [p.x(), p.y(), p.z()]
}

const fn vec_arr(v: Vec3) -> [f64; 3] {
    [v.x(), v.y(), v.z()]
}

const fn point(a: [f64; 3]) -> Point3 {
    Point3::new(a[0], a[1], a[2])
}

const fn vector(a: [f64; 3]) -> Vec3 {
    Vec3::new(a[0], a[1], a[2])
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// ── Spline spaces ───────────────────────────────────────────────────────────

fn spline_value<const K: usize>(
    degree: usize,
    knots: &[f64],
    coefs: &[[f64; K]],
    u: f64,
) -> [f64; K] {
    let span = find_span(coefs.len(), degree, u, knots);
    let mut out = [0.0; K];
    for (k, b) in basis_funs(span, u, degree, knots).into_iter().enumerate() {
        let c = &coefs[span - degree + k];
        for (o, x) in out.iter_mut().zip(c) {
            *o += b * x;
        }
    }
    out
}

#[allow(clippy::cast_precision_loss)]
fn greville(degree: usize, knots: &[f64]) -> Vec<f64> {
    let n = knots.len() - degree - 1;
    (0..n)
        .map(|i| knots[i + 1..=i + degree].iter().sum::<f64>() / degree as f64)
        .collect()
}

/// The spline of the space `(degree, knots)` taking `values` at its Greville
/// abscissae, which make its collocation matrix totally positive and banded,
/// so it eliminates stably without pivoting.
fn spline_through_greville<const K: usize>(
    degree: usize,
    knots: &[f64],
    mut values: Vec<[f64; K]>,
) -> Result<Vec<[f64; K]>, OperationsError> {
    let n = values.len();
    let sites = greville(degree, knots);
    let mut rows = vec![vec![0.0; n]; n];
    for (row, &u) in rows.iter_mut().zip(&sites) {
        let span = find_span(n, degree, u, knots);
        for (k, b) in basis_funs(span, u, degree, knots).into_iter().enumerate() {
            row[span - degree + k] = b;
        }
    }
    for col in 0..n {
        if rows[col][col].abs() < 1e-14 {
            return Err(refuse("a spline's collocation system is singular"));
        }
        for r in col + 1..n {
            let f = rows[r][col] / rows[col][col];
            if f == 0.0 {
                continue;
            }
            let reach = (col + degree + 1).min(n);
            for c in col..reach {
                rows[r][c] -= f * rows[col][c];
            }
            let pivot = values[col];
            for (x, p) in values[r].iter_mut().zip(pivot) {
                *x -= f * p;
            }
        }
    }
    for r in (0..n).rev() {
        let reach = (r + degree + 1).min(n);
        for c in r + 1..reach {
            let (f, known) = (rows[r][c], values[c]);
            for (x, k) in values[r].iter_mut().zip(known) {
                *x -= f * k;
            }
        }
        let pivot = rows[r][r];
        for x in &mut values[r] {
            *x /= pivot;
        }
    }
    Ok(values)
}

/// The coefficients in the space `to` of the spline `coefs` in the space
/// `from`. When `to` holds `from`, as raising a degree or adding knots makes
/// it, this is the same curve to rounding.
fn represent<const K: usize>(
    from: (usize, &[f64]),
    coefs: &[[f64; K]],
    to: (usize, &[f64]),
) -> Result<Vec<[f64; K]>, OperationsError> {
    if from.0 == to.0 && from.1 == to.1 {
        return Ok(coefs.to_vec());
    }
    let values = greville(to.0, to.1)
        .into_iter()
        .map(|u| spline_value(from.0, from.1, coefs, u))
        .collect();
    spline_through_greville(to.0, to.1, values)
}

/// The smallest clamped space holding every one of `spaces`, all on one
/// domain, of degree at least `at_least`: raising a spline's degree raises
/// each knot's multiplicity with it.
fn merged_knots(spaces: &[(usize, &[f64])], at_least: usize) -> (usize, Vec<f64>) {
    let degree = spaces.iter().map(|s| s.0).max().unwrap_or(1).max(at_least);
    let (lo, hi) = (spaces[0].1[0], spaces[0].1[spaces[0].1.len() - 1]);
    let mut runs: Vec<(f64, usize)> = Vec::new();
    for &(d, knots) in spaces {
        for chunk in knots.chunk_by(|a, b| (b - a).abs() <= KNOT_MERGE * (hi - lo)) {
            let k = chunk[0];
            if k > lo + KNOT_MERGE * (hi - lo) && k < hi - KNOT_MERGE * (hi - lo) {
                runs.push((k, chunk.len() + degree - d));
            }
        }
    }
    runs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = vec![lo; degree + 1];
    let mut merged: Vec<(f64, usize)> = Vec::new();
    for (k, m) in runs {
        match merged.last_mut() {
            Some(last) if (k - last.0).abs() <= KNOT_MERGE * (hi - lo) => last.1 = last.1.max(m),
            _ => merged.push((k, m)),
        }
    }
    for (k, m) in merged {
        out.extend(std::iter::repeat_n(k, m.min(degree)));
    }
    out.extend(std::iter::repeat_n(hi, degree + 1));
    (degree, out)
}

/// The curve `curve` as a non-rational spline on its own domain, within
/// [`POLYNOMIAL_TOLERANCE`] of it, cut at each of its knots, across which a
/// rational curve need not be smooth.
fn polynomial_curve(curve: &NurbsCurve) -> Result<NurbsCurve, OperationsError> {
    let w0 = curve.weights()[0];
    if curve
        .weights()
        .iter()
        .all(|w| (w - w0).abs() <= 1e-12 * w0.abs())
    {
        return Ok(NurbsCurve::new(
            curve.degree(),
            curve.knots().to_vec(),
            curve.control_points().to_vec(),
            vec![1.0; curve.weights().len()],
        )?);
    }
    let (d0, d1) = curve.domain();
    let mut cuts: Vec<f64> = curve
        .knots()
        .iter()
        .copied()
        .filter(|&k| k >= d0 && k <= d1)
        .collect();
    cuts.dedup_by(|a, b| (*a - *b).abs() <= KNOT_MERGE * (d1 - d0));
    let pieces = cuts
        .windows(2)
        .map(|w| fit_polynomial(curve, w[0], w[1]))
        .collect::<Result<Vec<_>, _>>()?;
    join_spans(&pieces, &cuts)
}

/// A clamped degree-[`POLYNOMIAL_DEGREE`] spline on `[a, b]` in equal spans
/// through `curve` at its Greville abscissae, halving the spans until it lies
/// within tolerance of the curve between them.
#[allow(clippy::cast_precision_loss)]
fn fit_polynomial(curve: &NurbsCurve, a: f64, b: f64) -> Result<NurbsCurve, OperationsError> {
    let degree = POLYNOMIAL_DEGREE;
    let target = |u: f64| arr(curve.evaluate(u));
    let mut spans = 1_usize;
    while spans <= 1024 {
        let step = (b - a) / spans as f64;
        let mut knots = vec![a; degree + 1];
        knots.extend((1..spans).map(|k| (k as f64).mul_add(step, a)));
        knots.extend(std::iter::repeat_n(b, degree + 1));
        let values = greville(degree, &knots).into_iter().map(target).collect();
        let coefs: Vec<[f64; 3]> = spline_through_greville(degree, &knots, values)?;
        let worst = (0..spans * 4)
            .map(|k| ((k as f64 + 0.5) / 4.0).mul_add(step, a))
            .map(|u| {
                let at = spline_value(degree, &knots, &coefs, u);
                (point(at) - point(target(u))).length()
            })
            .fold(0.0, f64::max);
        if worst <= POLYNOMIAL_TOLERANCE {
            let points = coefs.into_iter().map(point).collect::<Vec<_>>();
            let weights = vec![1.0; points.len()];
            return Ok(NurbsCurve::new(degree, knots, points, weights)?);
        }
        spans *= 2;
    }
    Err(refuse(
        "a rational curve has no polynomial spline within tolerance",
    ))
}

/// Non-rational `pieces` joined end to end, piece `i` running from `params[i]`
/// to `params[i + 1]` at a common degree. Each piece's end control point is
/// the next's start.
fn join_spans(pieces: &[NurbsCurve], params: &[f64]) -> Result<NurbsCurve, OperationsError> {
    let degree = pieces.iter().map(NurbsCurve::degree).max().unwrap_or(1);
    let mut knots = vec![params[0]; degree + 1];
    let mut points: Vec<Point3> = Vec::new();
    for (i, piece) in pieces.iter().enumerate() {
        let (d0, d1) = piece.domain();
        let (_, raised_knots) = merged_knots(&[(piece.degree(), piece.knots())], degree);
        let coefs: Vec<[f64; 3]> = piece.control_points().iter().map(|&p| arr(p)).collect();
        let raised = represent(
            (piece.degree(), piece.knots()),
            &coefs,
            (degree, &raised_knots),
        )?;
        let map = |k: f64| params[i] + (k - d0) / (d1 - d0) * (params[i + 1] - params[i]);
        if i > 0 {
            knots.extend(std::iter::repeat_n(params[i], degree));
        }
        knots.extend(
            raised_knots[degree + 1..raised_knots.len() - degree - 1]
                .iter()
                .map(|&k| map(k)),
        );
        let skip = usize::from(i > 0);
        points.extend(raised.into_iter().skip(skip).map(point));
    }
    knots.extend(std::iter::repeat_n(params[params.len() - 1], degree + 1));
    let weights = vec![1.0; points.len()];
    Ok(NurbsCurve::new(degree, knots, points, weights)?)
}

// ── Reading profiles ────────────────────────────────────────────────────────

/// A profile's boundary as edges running start to end, with the positions of
/// its vertices. A closed edge is cut into four, which adds no vertex: a
/// surface along a whole closed edge would meet itself.
fn read_ring(
    topo: &Topology,
    face: FaceId,
) -> Result<(Vec<ProfileEdgeGeom>, Vec<Point3>), OperationsError> {
    let outer = topo.face(face)?.outer_wire();
    let edges =
        wire_oriented_edges(topo, outer).ok_or_else(|| refuse("a profile's wire has no edges"))?;
    let count = edges.len();
    for (k, e) in edges.iter().enumerate() {
        if (e.end - edges[(k + 1) % count].start).length() > WIRE_GAP {
            return Err(refuse("a profile's wire does not close"));
        }
    }
    let edges = edges
        .into_iter()
        .map(|e| match e.curve {
            EdgeCurve::NurbsCurve(_) => Ok(ProfileEdgeGeom {
                curve: EdgeCurve::NurbsCurve(clamped_span(&e)?),
                ..e
            }),
            _ => Ok(e),
        })
        .collect::<Result<Vec<_>, OperationsError>>()?;
    let vertices: Vec<Point3> = edges.iter().map(|e| e.start).collect();
    let pieces = if count == 1 {
        split_closed(&edges[0])?
    } else {
        edges
    };
    let mut ring = Vec::with_capacity(pieces.len());
    for piece in &pieces {
        ring.extend(cut_at_kinks(piece)?);
    }
    Ok((ring, vertices))
}

/// A NURBS edge cut at each full knot it turns at: a side running across a
/// kink would hide a crease inside its face. Its pieces add no vertex a rail
/// could run through.
fn cut_at_kinks(e: &ProfileEdgeGeom) -> Result<Vec<ProfileEdgeGeom>, OperationsError> {
    let EdgeCurve::NurbsCurve(curve) = &e.curve else {
        return Ok(vec![e.clone()]);
    };
    let (d0, d1) = curve.domain();
    let degree = curve.degree();
    let turns_at = |run: &[f64]| -> bool {
        let t = run[0];
        if t <= d0 || t >= d1 || run.len() < degree {
            return false;
        }
        let at = curve.knots().partition_point(|&k| k < t) - 1;
        let points = curve.control_points();
        let before = (points[at] - points[at - 1]).normalize();
        let after = (points[at + 1] - points[at]).normalize();
        match (before, after) {
            (Ok(a), Ok(b)) => a.dot(b) < 0.0 || a.cross(b).length() > CREASE_ANGLE.sin(),
            _ => false,
        }
    };
    let cuts: Vec<f64> = curve
        .knots()
        .chunk_by(|a, b| a.to_bits() == b.to_bits())
        .filter(|run| turns_at(run))
        .map(|run| run[0])
        .collect();
    let mut out = Vec::with_capacity(cuts.len() + 1);
    let mut rest = curve.clone();
    let mut from = e.start;
    for t in cuts {
        let (piece, tail) = curve_split(&rest, t)?;
        let to = tail.evaluate(tail.domain().0);
        out.push(ProfileEdgeGeom {
            curve: EdgeCurve::NurbsCurve(piece),
            start: from,
            end: to,
        });
        rest = tail;
        from = to;
    }
    out.push(ProfileEdgeGeom {
        curve: EdgeCurve::NurbsCurve(rest),
        start: from,
        end: e.end,
    });
    Ok(out)
}

fn clamped_span(e: &ProfileEdgeGeom) -> Result<NurbsCurve, OperationsError> {
    crate::sweep::clamped_edge_span(&e.curve, true, (e.start, e.end))?
        .ok_or_else(|| refuse("a NURBS edge on an unclamped knot vector cannot be lofted"))
}

/// The edge's curve from its start to its end on the parameter range `[0, 1]`.
fn unit_span(e: &ProfileEdgeGeom) -> Result<NurbsCurve, OperationsError> {
    crate::sheet::on_unit_domain(&clamped_span(e)?)
}

/// A closed edge in four pieces from its start, each its own curve.
#[allow(clippy::cast_precision_loss)]
fn split_closed(e: &ProfileEdgeGeom) -> Result<Vec<ProfileEdgeGeom>, OperationsError> {
    let (a0, quarter) = match &e.curve {
        EdgeCurve::Circle(c) => (c.project(e.start), std::f64::consts::FRAC_PI_2),
        EdgeCurve::Ellipse(c) => (c.project(e.start), std::f64::consts::FRAC_PI_2),
        EdgeCurve::NurbsCurve(c) => {
            let (d0, d1) = c.domain();
            let mut rest = c.clone();
            let mut out = Vec::with_capacity(4);
            for k in 1..4 {
                let t = (k as f64).mul_add((d1 - d0) / 4.0, d0);
                let (piece, tail) = curve_split(&rest, t)?;
                out.push(piece);
                rest = tail;
            }
            out.push(rest);
            return Ok(out
                .into_iter()
                .map(|piece| {
                    let (t0, t1) = piece.domain();
                    let (start, end) = (piece.evaluate(t0), piece.evaluate(t1));
                    ProfileEdgeGeom {
                        curve: EdgeCurve::NurbsCurve(piece),
                        start,
                        end,
                    }
                })
                .collect());
        }
        EdgeCurve::Line => return Err(refuse("a straight edge cannot close on itself")),
    };
    let at = |k: usize| -> Point3 {
        let t = (k as f64).mul_add(quarter, a0);
        match &e.curve {
            EdgeCurve::Circle(c) => c.evaluate(t),
            EdgeCurve::Ellipse(c) => c.evaluate(t),
            _ => e.start,
        }
    };
    Ok((0..4)
        .map(|k| ProfileEdgeGeom {
            curve: e.curve.clone(),
            start: if k == 0 { e.start } else { at(k) },
            end: if k == 3 { e.start } else { at(k + 1) },
        })
        .collect())
}

fn edge_point(e: &ProfileEdgeGeom, fraction: f64) -> Point3 {
    let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
    e.curve
        .evaluate_with_endpoints((t1 - t0).mul_add(fraction, t0), e.start, e.end)
}

/// Points round a ring, each curved edge sampled along its span.
#[allow(clippy::cast_precision_loss)]
fn ring_samples(ring: &[ProfileEdgeGeom]) -> Vec<Point3> {
    const CURVE_SAMPLES: usize = 16;
    let mut points = Vec::new();
    for e in ring {
        let samples = if matches!(e.curve, EdgeCurve::Line) {
            1
        } else {
            CURVE_SAMPLES
        };
        points.extend((0..samples).map(|k| edge_point(e, k as f64 / samples as f64)));
    }
    points
}

/// Points along each edge of a ring, its straight edges too: an edge can leave a
/// face between two vertices that lie on it, across a hole or off a curved surface.
#[allow(clippy::cast_precision_loss)]
fn along_edges(ring: &[ProfileEdgeGeom]) -> Vec<Point3> {
    const SAMPLES: usize = 16;
    ring.iter()
        .flat_map(|e| (0..SAMPLES).map(move |k| edge_point(e, k as f64 / SAMPLES as f64)))
        .collect()
}

fn centre_of(points: &[Point3]) -> Point3 {
    crate::winding::polygon_centroid(points)
}

/// An edge's length, and the parameter fractions at equal lengths along it.
#[allow(clippy::cast_precision_loss)]
fn edge_lengths(e: &ProfileEdgeGeom) -> Vec<f64> {
    const STEPS: usize = 64;
    if matches!(e.curve, EdgeCurve::Line | EdgeCurve::Circle(_)) {
        let length = match &e.curve {
            EdgeCurve::Circle(c) => {
                let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
                (t1 - t0).abs() * c.radius()
            }
            _ => (e.end - e.start).length(),
        };
        return (0..=STEPS)
            .map(|k| length * k as f64 / STEPS as f64)
            .collect();
    }
    let mut total = 0.0;
    let mut out = Vec::with_capacity(STEPS + 1);
    let mut previous = edge_point(e, 0.0);
    out.push(0.0);
    for k in 1..=STEPS {
        let at = edge_point(e, k as f64 / STEPS as f64);
        total += (at - previous).length();
        out.push(total);
        previous = at;
    }
    out
}

/// `e` cut at the lengths `at` along it, in increasing order.
#[allow(clippy::cast_precision_loss)]
fn cut_edge(e: &ProfileEdgeGeom, at: &[f64]) -> Result<Vec<ProfileEdgeGeom>, OperationsError> {
    if at.is_empty() {
        return Ok(vec![e.clone()]);
    }
    let table = edge_lengths(e);
    let steps = (table.len() - 1) as f64;
    let fraction = |length: f64| {
        let k = table
            .partition_point(|&l| l < length)
            .clamp(1, table.len() - 1);
        let run = table[k] - table[k - 1];
        let within = if run > 0.0 {
            (length - table[k - 1]) / run
        } else {
            0.0
        };
        ((k - 1) as f64 + within) / steps
    };
    let (t0, t1) = e.curve.domain_with_endpoints(e.start, e.end);
    if let EdgeCurve::NurbsCurve(curve) = &e.curve {
        let mut out = Vec::with_capacity(at.len() + 1);
        let mut rest = curve.clone();
        let mut from = e.start;
        for &length in at {
            let t = (t1 - t0).mul_add(fraction(length), t0);
            let (piece, tail) = curve_split(&rest, t)?;
            let to = tail.evaluate(tail.domain().0);
            out.push(ProfileEdgeGeom {
                curve: EdgeCurve::NurbsCurve(piece),
                start: from,
                end: to,
            });
            rest = tail;
            from = to;
        }
        out.push(ProfileEdgeGeom {
            curve: EdgeCurve::NurbsCurve(rest),
            start: from,
            end: e.end,
        });
        return Ok(out);
    }
    let mut points = vec![e.start];
    points.extend(at.iter().map(|&length| edge_point(e, fraction(length))));
    points.push(e.end);
    Ok(points
        .windows(2)
        .map(|w| ProfileEdgeGeom {
            curve: e.curve.clone(),
            start: w[0],
            end: w[1],
        })
        .collect())
}

/// The ring edge starting at a vertex read earlier, which a reversal may have
/// taken from the neighbouring edge's end: a wire's closing gap can lie between.
fn ring_index(ring: &[ProfileEdgeGeom], at: Point3) -> Result<usize, OperationsError> {
    let distance = |e: &ProfileEdgeGeom| (e.start - at).length();
    (0..ring.len())
        .min_by(|&a, &b| distance(&ring[a]).total_cmp(&distance(&ring[b])))
        .filter(|&i| distance(&ring[i]) <= WIRE_GAP)
        .ok_or_else(|| refuse("a rail's vertex is not on its profile's ring"))
}

/// Turn every ring to run counterclockwise about the way the loft goes there,
/// which makes each side's `Su × Sv` point out of the solid. That way is the
/// chord between the neighbouring profiles, unless a profile's plane holds the
/// chord, as a U-bend's does: then the rails say, or a `normal` condition's
/// profile by its own normal.
fn orient_rings(
    topo: &Topology,
    rings: &mut [Vec<ProfileEdgeGeom>],
    rails: &[RailThrough],
    profiles: &[FaceId],
    options: &LoftOptions,
) -> Result<(), OperationsError> {
    let last = rings.len() - 1;
    let samples: Vec<Vec<Point3>> = rings.iter().map(|r| ring_samples(r)).collect();
    let centres: Vec<Point3> = samples.iter().map(|s| centre_of(s)).collect();
    for (s, ring) in rings.iter_mut().enumerate() {
        let normal = crate::winding::newell_normal(&samples[s])
            .normalize()
            .map_err(|_| refuse("a profile has no area"))?;
        let reads = |heading: Vec3, floor: f64| -> Option<f64> {
            let c = normal.dot(heading.normalize().ok()?);
            (c.abs() > floor).then(|| c.signum())
        };
        let chord = centres[(s + 1).min(last)] - centres[s.saturating_sub(1)];
        let along_rails = rails.iter().fold(Vec3::new(0.0, 0.0, 0.0), |sum, rail| {
            sum + rail
                .tangent_at(s)
                .normalize()
                .unwrap_or(Vec3::new(0.0, 0.0, 0.0))
        });
        let from_condition = || -> Result<Option<f64>, OperationsError> {
            let (end, along) = match s {
                0 => (&options.start, 1.0),
                _ if s == last => (&options.end, -1.0),
                _ => return Ok(None),
            };
            let LoftEnd::Normal { weight } = *end else {
                return Ok(None);
            };
            let side = profile_outward(topo, profiles[s], normal)? * (along * weight.signum());
            Ok(reads(side, 1e-6))
        };
        let sign = reads(chord, 1e-3)
            .or_else(|| reads(along_rails, 1e-6))
            .map_or_else(from_condition, |sign| Ok(Some(sign)))?
            .or_else(|| reads(chord, 1e-9))
            .ok_or_else(|| {
                refuse("a profile's neighbour lies in its plane, which leaves its winding open")
            })?;
        if sign < 0.0 {
            *ring = reversed_ring(ring)?;
        }
    }
    Ok(())
}

/// The way a profile's own face looks: its plane's normal if it has one,
/// otherwise `fallback`.
fn profile_outward(topo: &Topology, face: FaceId, fallback: Vec3) -> Result<Vec3, OperationsError> {
    let data = topo.face(face)?;
    Ok(match data.surface() {
        FaceSurface::Plane { normal, .. } => {
            let sign = if data.is_reversed() { -1.0 } else { 1.0 };
            normal.normalize()? * sign
        }
        _ => fallback,
    })
}

fn reversed_ring(ring: &[ProfileEdgeGeom]) -> Result<Vec<ProfileEdgeGeom>, OperationsError> {
    ring.iter()
        .rev()
        .map(|e| {
            Ok(ProfileEdgeGeom {
                curve: e.curve.reversed()?,
                start: e.end,
                end: e.start,
            })
        })
        .collect()
}

/// Move the seam of each closed circle or ellipse to face the first vertex of
/// the first ring that has fixed ones, corners or a closed spline's, since such
/// a seam is no corner: it would twist the loft by however far it happens to
/// lie round. With no fixed ones anywhere, the seams face the way the loft
/// leans, so the result does not hang on how the first curve's frame happens to
/// lie; a coaxial stack leans no way and keeps the first ring's seam.
fn align_lone_seams(
    rings: &mut [Vec<ProfileEdgeGeom>],
    corners: &[Vec<Point3>],
) -> Result<(), OperationsError> {
    let tol = Tolerance::new().linear;
    let movable: Vec<bool> = rings
        .iter()
        .zip(corners)
        .map(|(ring, c)| {
            c.len() == 1 && matches!(ring[0].curve, EdgeCurve::Circle(_) | EdgeCurve::Ellipse(_))
        })
        .collect();
    let toward = if let Some(reference) = movable.iter().position(|&m| !m) {
        rings[reference][0].start - centre_of(&ring_samples(&rings[reference]))
    } else {
        let centres: Vec<Point3> = rings.iter().map(|r| centre_of(&ring_samples(r))).collect();
        let samples = ring_samples(&rings[0]);
        let normal = crate::winding::newell_normal(&samples).normalize()?;
        let lean = centres[centres.len() - 1] - centres[0];
        let lean = lean - normal * lean.dot(normal);
        if lean.length() > tol {
            lean
        } else {
            rings[0][0].start - centres[0]
        }
    };
    if toward.length() <= tol {
        return Ok(());
    }
    for (ring, _) in rings.iter_mut().zip(&movable).filter(|(_, m)| **m) {
        let e = &ring[0];
        let seam = match &e.curve {
            EdgeCurve::Circle(c) => c.evaluate(c.project(c.center() + toward)),
            EdgeCurve::Ellipse(c) => c.evaluate(c.project(c.center() + toward)),
            EdgeCurve::Line | EdgeCurve::NurbsCurve(_) => continue,
        };
        *ring = split_closed(&ProfileEdgeGeom {
            curve: e.curve.clone(),
            start: seam,
            end: seam,
        })?;
    }
    Ok(())
}

/// Cut every ring at the others' vertices between rails where their edge
/// counts differ, so every ring has one edge count and the rails' vertices one
/// index. With no rails the whole ring is one run from its first vertex. Which
/// rail runs through each ring index.
fn equalize_rings(
    rings: &mut [Vec<ProfileEdgeGeom>],
    rails: &[RailThrough],
) -> Result<Vec<Option<usize>>, OperationsError> {
    // Each ring's rail indices, in the first ring's order round it.
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(rings.len());
    for (s, ring) in rings.iter().enumerate() {
        indices.push(
            rails
                .iter()
                .map(|rail| ring_index(ring, rail.through[s]))
                .collect::<Result<_, _>>()?,
        );
    }
    let mut order: Vec<usize> = (0..rails.len()).collect();
    if let Some(first) = indices.first() {
        order.sort_by_key(|&j| first[j]);
    }
    for at in &indices {
        if order.windows(2).any(|w| at[w[0]] >= at[w[1]]) {
            return Err(refuse(
                "the rails come round the profiles in different orders",
            ));
        }
    }

    // The runs of each ring between consecutive rails.
    let runs_of = |s: usize, ring: &[ProfileEdgeGeom]| -> Vec<Vec<ProfileEdgeGeom>> {
        if rails.is_empty() {
            return vec![ring.to_vec()];
        }
        (0..order.len())
            .map(|k| {
                let from = indices[s][order[k]];
                let to = if k + 1 == order.len() {
                    ring.len()
                } else {
                    indices[s][order[k + 1]]
                };
                ring[from..to].to_vec()
            })
            .collect()
    };
    let runs: Vec<Vec<Vec<ProfileEdgeGeom>>> = rings
        .iter()
        .enumerate()
        .map(|(s, ring)| runs_of(s, ring))
        .collect();
    let run_count = runs[0].len();
    let mut equalized: Vec<Vec<ProfileEdgeGeom>> = vec![Vec::new(); rings.len()];
    let mut rail_at_vertex = Vec::new();
    for k in 0..run_count {
        let per_ring: Vec<&Vec<ProfileEdgeGeom>> = runs.iter().map(|ring| &ring[k]).collect();
        let cut = equalize_run(&per_ring)?;
        if !rails.is_empty() {
            rail_at_vertex.push(Some(order[k]));
        }
        rail_at_vertex.extend(std::iter::repeat_n(
            None,
            cut[0].len() - usize::from(!rails.is_empty()),
        ));
        for (ring, edges) in equalized.iter_mut().zip(cut) {
            ring.extend(edges);
        }
    }
    for (ring, edges) in rings.iter_mut().zip(equalized) {
        *ring = edges;
    }
    Ok(rail_at_vertex)
}

/// One run of edges per ring, cut so that all have one edge count. Runs of one
/// count stay as they are, pairing edge by edge. Others are cut at every place
/// any of them has a vertex, by fraction of their length.
fn equalize_run(
    runs: &[&Vec<ProfileEdgeGeom>],
) -> Result<Vec<Vec<ProfileEdgeGeom>>, OperationsError> {
    if runs.iter().all(|run| run.len() == runs[0].len()) {
        return Ok(runs.iter().map(|run| (*run).clone()).collect());
    }
    let places_in = |run: &[ProfileEdgeGeom]| -> (Vec<f64>, f64) {
        let lengths: Vec<f64> = run
            .iter()
            .map(|e| *edge_lengths(e).last().unwrap_or(&0.0))
            .collect();
        let total: f64 = lengths.iter().sum();
        let mut sum = 0.0;
        let places = lengths[..lengths.len() - 1]
            .iter()
            .map(|l| {
                sum += l;
                sum / total
            })
            .collect();
        (places, total)
    };
    let mut every: Vec<f64> = runs.iter().flat_map(|run| places_in(run).0).collect();
    every.sort_by(f64::total_cmp);
    every.dedup_by(|a, b| (*a - *b).abs() <= 2.0 * POSITION_MERGE);

    let mut out = Vec::with_capacity(runs.len());
    for run in runs {
        let (own, total) = places_in(run);
        let new_places: Vec<f64> = every
            .iter()
            .copied()
            .filter(|&place| {
                own.iter()
                    .all(|&p| (p - place).abs() > 2.0 * POSITION_MERGE)
            })
            .collect();
        let mut cut = Vec::new();
        let mut from = 0.0;
        for (i, e) in run.iter().enumerate() {
            let to = own.get(i).copied().unwrap_or(1.0);
            let at: Vec<f64> = new_places
                .iter()
                .filter(|&&place| place > from && place < to)
                .map(|&place| (place - from) * total)
                .collect();
            cut.extend(cut_edge(e, &at)?);
            from = to;
        }
        out.push(cut);
    }
    if out.iter().any(|run| run.len() != out[0].len()) {
        return Err(refuse(
            "two vertices of a profile are too close together to pair with another's",
        ));
    }
    Ok(out)
}

/// The surface's parameter at each profile: where the rails put it on average,
/// or the mean chord length along the vertices.
fn section_params(
    rings: &[Vec<ProfileEdgeGeom>],
    rails: &[RailThrough],
) -> Result<Vec<f64>, OperationsError> {
    let count = rings.len();
    let params: Vec<f64> = if rails.is_empty() {
        let rows: Vec<Vec<Point3>> = (0..rings[0].len())
            .map(|i| rings.iter().map(|ring| ring[i].start).collect())
            .collect();
        mean_chord_params(&rows)
    } else {
        let mut sums = vec![0.0; count];
        for rail in rails {
            let (first, last) = (rail.at[0], rail.at[count - 1]);
            for (sum, &t) in sums.iter_mut().zip(&rail.at) {
                *sum += (t - first) / (last - first);
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let rails_count = rails.len() as f64;
        sums.iter().map(|sum| sum / rails_count).collect()
    };
    if params.windows(2).any(|w| w[1] - w[0] <= 1e-9) {
        return Err(refuse("two neighbouring profiles coincide"));
    }
    let mut params = params;
    params[0] = 0.0;
    params[count - 1] = 1.0;
    Ok(params)
}

// ── Rails ───────────────────────────────────────────────────────────────────

/// A rail as a non-rational spline running from the first profile to the
/// last, with where it meets each.
struct RailThrough {
    curve: NurbsCurve,
    /// The rail's parameter at each profile, increasing.
    at: Vec<f64>,
    /// The vertex it runs through on each profile.
    through: Vec<Point3>,
}

impl RailThrough {
    fn tangent_at(&self, s: usize) -> Vec3 {
        self.curve.derivatives(self.at[s], 1)[1]
    }

    /// The rail from its first profile to its last, on the unit domain, with
    /// each profile's point at `params`. Its parameter is rescaled over each
    /// stretch between profiles, which leaves it the same curve: it takes a
    /// full knot only where the rescaling changes, the one place its
    /// derivative jumps.
    fn on_params(&self, params: &[f64]) -> Result<NurbsCurve, OperationsError> {
        let last = self.at.len() - 1;
        let mut curve = trimmed(&self.curve, self.at[0], self.at[last])?;
        let slope = |s: usize| (params[s + 1] - params[s]) / (self.at[s + 1] - self.at[s]);
        for s in 1..last {
            if (slope(s) / slope(s - 1) - 1.0).abs() > 1e-9 {
                curve = curve_knot_insert(&curve, self.at[s], curve.degree())?;
            }
        }
        let degree = curve.degree();
        let knots = curve.knots();
        let map = |t: f64| {
            let s = self
                .at
                .partition_point(|&x| x <= t)
                .saturating_sub(1)
                .min(last - 1);
            (t - self.at[s]).mul_add(slope(s), params[s])
        };
        let mapped: Vec<f64> = knots
            .iter()
            .enumerate()
            .map(|(i, &t)| {
                if i <= degree {
                    0.0
                } else if i >= knots.len() - degree - 1 {
                    1.0
                } else {
                    map(t)
                }
            })
            .collect();
        Ok(NurbsCurve::new(
            degree,
            mapped,
            curve.control_points().to_vec(),
            curve.weights().to_vec(),
        )?)
    }
}

fn trimmed(curve: &NurbsCurve, a: f64, b: f64) -> Result<NurbsCurve, OperationsError> {
    let (d0, d1) = curve.domain();
    let eps = KNOT_MERGE * (d1 - d0);
    let mut out = curve.clone();
    if a > d0 + eps {
        out = curve_split(&out, a)?.1;
    }
    if b < d1 - eps {
        out = curve_split(&out, b)?.0;
    }
    Ok(out)
}

/// A rail's edges as one non-rational spline in the order they chain, each
/// edge taking a stretch of the parameter in proportion to its length.
fn rail_spline(topo: &Topology, rail: LoftRail) -> Result<NurbsCurve, OperationsError> {
    let runs: Vec<OrientedEdge> = match rail {
        LoftRail::Edge(edge) => vec![OrientedEdge::new(edge, true)],
        LoftRail::Wire(wire) => topo.wire(wire)?.edges().to_vec(),
    };
    if runs.is_empty() {
        return Err(refuse("a rail has no edges"));
    }
    let mut pieces = Vec::with_capacity(runs.len());
    let mut params = vec![0.0];
    let mut reach: Option<Point3> = None;
    for oe in &runs {
        let edge = topo.edge(oe.edge())?;
        let (a, b) = (
            topo.vertex(edge.start())?.point(),
            topo.vertex(edge.end())?.point(),
        );
        let (from, to) = if oe.is_forward() { (a, b) } else { (b, a) };
        if let Some(reach) = reach
            && (from - reach).length() > WIRE_GAP
        {
            return Err(refuse("a rail's edges do not connect end to end"));
        }
        let span = crate::sweep::clamped_edge_span(edge.curve(), oe.is_forward(), (from, to))?
            .ok_or_else(|| refuse("a NURBS rail on an unclamped knot vector cannot be followed"))?;
        let span = polynomial_curve(&span)?;
        params.push(params[params.len() - 1] + span.arc_length(64));
        pieces.push(span);
        reach = Some(to);
    }
    if pieces.len() == 1 {
        return Ok(pieces.remove(0));
    }
    let total = params[params.len() - 1];
    if total <= 0.0 {
        return Err(refuse("a rail has no length"));
    }
    let params: Vec<f64> = params.into_iter().map(|p| p / total).collect();
    join_spans(&pieces, &params)
}

/// Each rail through its vertex of every profile, in order from the first
/// profile to the last.
fn read_rails(
    topo: &Topology,
    rails: &[LoftRail],
    corners: &[Vec<Point3>],
) -> Result<Vec<RailThrough>, OperationsError> {
    if let Some(fewest) = corners.iter().map(Vec::len).min()
        && rails.len() > fewest
    {
        return Err(refuse(format!(
            "{} rails cannot pair with a profile of {fewest} vertices",
            rails.len()
        )));
    }
    let mut out: Vec<RailThrough> = Vec::with_capacity(rails.len());
    for (j, &rail) in rails.iter().enumerate() {
        let curve = rail_spline(topo, rail)?;
        let mut at = Vec::with_capacity(corners.len());
        let mut through = Vec::with_capacity(corners.len());
        for (s, vertices) in corners.iter().enumerate() {
            let mut hits = Vec::new();
            for &vertex in vertices {
                let found = project_point_to_curve(&curve, vertex, 1e-12)?;
                if found.distance <= RAIL_VERTEX_TOLERANCE {
                    hits.push((vertex, found.parameter));
                }
            }
            let [(vertex, t)] = hits.as_slice() else {
                return Err(refuse(if hits.is_empty() {
                    format!("rail {j} does not pass through a vertex of profile {s}")
                } else {
                    format!("rail {j} passes through more than one vertex of profile {s}")
                }));
            };
            at.push(snap_to_knot(&curve, *t));
            through.push(*vertex);
        }
        let (curve, at) = if at.windows(2).all(|w| w[1] > w[0]) {
            (curve, at)
        } else if at.windows(2).all(|w| w[1] < w[0]) {
            let (d0, d1) = curve.domain();
            (
                crate::revolve::reversed_nurbs(&curve)?,
                at.iter().map(|t| d0 + d1 - t).collect(),
            )
        } else {
            return Err(refuse(format!(
                "rail {j} does not run through the profiles in order"
            )));
        };
        for earlier in &out {
            if earlier
                .through
                .iter()
                .zip(&through)
                .any(|(a, b)| (*a - *b).length() <= RAIL_VERTEX_TOLERANCE)
            {
                return Err(refuse("two rails run through one vertex of a profile"));
            }
        }
        out.push(RailThrough { curve, at, through });
    }
    Ok(out)
}

/// `t`, moved onto a knot it lies next to: a span of the width of a rounding
/// error is one no later step can use.
fn snap_to_knot(curve: &NurbsCurve, t: f64) -> f64 {
    let (d0, d1) = curve.domain();
    let reach = 1e-7 * (d1 - d0);
    curve
        .knots()
        .iter()
        .copied()
        .filter(|k| (k - t).abs() <= reach)
        .min_by(|a, b| (a - t).abs().total_cmp(&(b - t).abs()))
        .unwrap_or(t)
}

/// The rail less the skin's spline through its own points and, at a
/// conditioned end, its own derivative: zero there, so adding it moves neither
/// the profiles nor the derivative across an end. In the space `to`.
fn rail_deviation(
    rail: &NurbsCurve,
    skin: &SkinSpace,
    to: &(usize, Vec<f64>),
) -> Result<Vec<[f64; 3]>, OperationsError> {
    let points: Vec<[f64; 3]> = skin.params.iter().map(|&v| arr(rail.evaluate(v))).collect();
    let t0 = skin.start.then(|| vec_arr(rail.derivatives(0.0, 1)[1]));
    let t1 = skin.end.then(|| vec_arr(rail.derivatives(1.0, 1)[1]));
    let interpolant = skin.interpolate(&points, t0, t1);
    let interpolant = represent((skin.degree, &skin.knots), &interpolant, (to.0, &to.1))?;
    let coefs: Vec<[f64; 3]> = rail.control_points().iter().map(|&p| arr(p)).collect();
    let on_merged = represent((rail.degree(), rail.knots()), &coefs, (to.0, &to.1))?;
    Ok(on_merged
        .iter()
        .zip(&interpolant)
        .map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]])
        .collect())
}

/// Each ring vertex's deviation: its rail's, or between two rails their blend
/// by the distance round the run between them, the mean over the rings'.
fn vertex_deviations(
    rings: &[Vec<ProfileEdgeGeom>],
    rail_at_vertex: &[Option<usize>],
    rail_deviations: &[Vec<[f64; 3]>],
) -> Vec<Vec<[f64; 3]>> {
    let n = rail_at_vertex.len();
    #[allow(clippy::cast_precision_loss)]
    let lengths: Vec<f64> = (0..n)
        .map(|i| {
            rings
                .iter()
                .map(|ring| *edge_lengths(&ring[i]).last().unwrap_or(&0.0))
                .sum::<f64>()
                / rings.len() as f64
        })
        .collect();
    let railed: Vec<usize> = (0..n).filter(|&i| rail_at_vertex[i].is_some()).collect();
    let mut out = vec![vec![[0.0; 3]; rail_deviations[0].len()]; n];
    for (k, &from) in railed.iter().enumerate() {
        let to = railed[(k + 1) % railed.len()];
        let (Some(a), Some(b)) = (rail_at_vertex[from], rail_at_vertex[to]) else {
            continue;
        };
        let count = if to > from { to - from } else { n - from + to };
        let total: f64 = (0..count).map(|m| lengths[(from + m) % n]).sum();
        let mut along = 0.0;
        for m in 0..count {
            let i = (from + m) % n;
            let w = if total > 0.0 { along / total } else { 0.0 };
            for (c, (da, db)) in out[i]
                .iter_mut()
                .zip(rail_deviations[a].iter().zip(&rail_deviations[b]))
            {
                for axis in 0..3 {
                    c[axis] = (1.0 - w).mul_add(da[axis], w * db[axis]);
                }
            }
            along += lengths[i];
        }
    }
    out
}

// ── The skin across the profiles ────────────────────────────────────────────

/// The spline space across the profiles, and its interpolation: through a
/// point at each profile's parameter and, at a conditioned end, a derivative.
struct SkinSpace {
    degree: usize,
    knots: Vec<f64>,
    params: Vec<f64>,
    start: bool,
    end: bool,
    /// The inverse of the interpolation system, whose rows run through the
    /// profiles in order with each end's derivative next to its point.
    inverse: Vec<Vec<f64>>,
}

impl SkinSpace {
    fn new(params: Vec<f64>, start: bool, end: bool) -> Result<Self, OperationsError> {
        let mut sites = Vec::with_capacity(params.len() + 2);
        if start {
            sites.push(params[0]);
        }
        sites.extend_from_slice(&params);
        if end {
            sites.push(params[params.len() - 1]);
        }
        let n = sites.len();
        let degree = (n - 1).min(3);
        let mut knots = vec![params[0]; degree + 1];
        for j in 1..n - degree {
            #[allow(clippy::cast_precision_loss)]
            knots.push(sites[j..j + degree].iter().sum::<f64>() / degree as f64);
        }
        knots.extend(std::iter::repeat_n(params[params.len() - 1], degree + 1));

        let mut matrix = vec![vec![0.0; n]; n];
        for (r, row) in matrix.iter_mut().enumerate() {
            let derivative = (start && r == 0) || (end && r == n - 1);
            let u = sites[r];
            let span = find_span(n, degree, u, &knots);
            let values = if derivative {
                ders_basis_funs(span, u, degree, 1, &knots).swap_remove(1)
            } else {
                basis_funs(span, u, degree, &knots)
            };
            for (k, value) in values.into_iter().enumerate() {
                row[span - degree + k] = value;
            }
        }
        let inverse = inverted(matrix).ok_or_else(|| refuse("the loft's profiles are singular"))?;
        Ok(Self {
            degree,
            knots,
            params,
            start,
            end,
            inverse,
        })
    }

    /// The spline's coefficients through `points`, one per profile, with the
    /// given derivatives across its ends where they are conditioned.
    fn interpolate(
        &self,
        points: &[[f64; 3]],
        t0: Option<[f64; 3]>,
        t1: Option<[f64; 3]>,
    ) -> Vec<[f64; 3]> {
        let mut rhs = Vec::with_capacity(self.inverse.len());
        if self.start {
            rhs.push(t0.unwrap_or([0.0; 3]));
        }
        rhs.extend_from_slice(points);
        if self.end {
            rhs.push(t1.unwrap_or([0.0; 3]));
        }
        self.inverse
            .iter()
            .map(|row| {
                let mut out = [0.0; 3];
                for (&f, r) in row.iter().zip(&rhs) {
                    for (o, x) in out.iter_mut().zip(r) {
                        *o += f * x;
                    }
                }
                out
            })
            .collect()
    }
}

/// `matrix`'s inverse by Gauss-Jordan elimination with partial pivoting.
/// `None` for a singular one.
fn inverted(mut matrix: Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = matrix.len();
    let mut out: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
        .collect();
    for col in 0..n {
        let pivot =
            (col..n).max_by(|&a, &b| matrix[a][col].abs().total_cmp(&matrix[b][col].abs()))?;
        if matrix[pivot][col].abs() < 1e-13 {
            return None;
        }
        matrix.swap(col, pivot);
        out.swap(col, pivot);
        let scale = 1.0 / matrix[col][col];
        for j in 0..n {
            matrix[col][j] *= scale;
            out[col][j] *= scale;
        }
        for r in 0..n {
            if r == col || matrix[r][col] == 0.0 {
                continue;
            }
            let f = matrix[r][col];
            for j in 0..n {
                matrix[r][j] -= f * matrix[col][j];
                out[r][j] -= f * out[col][j];
            }
        }
    }
    Some(out)
}

// ── End conditions ──────────────────────────────────────────────────────────

/// What an end field's direction comes from.
enum Direction {
    /// Square to the profile's plane.
    Normal(Vec3),
    /// In the tangent plane of a surface, square to the profile.
    Tangent(FaceSurface),
}

/// The surface's derivative across an end profile, along its ring.
struct EndField {
    direction: Direction,
    /// Which side of the unsigned direction the field heads.
    side: f64,
    /// The natural size at each vertex, weighted.
    sizes: Vec<f64>,
    /// The derivative at each vertex.
    corners: Vec<Vec3>,
    /// The end ring's edges on `[0, 1]`.
    spans: Vec<NurbsCurve>,
}

impl EndField {
    /// The unit direction, signed, along edge `i` at `u`.
    fn square(&self, i: usize, u: f64) -> Result<Vec3, OperationsError> {
        match &self.direction {
            Direction::Normal(n) => Ok(*n * self.side),
            Direction::Tangent(surface) => {
                Ok(square_in_tangent_plane(surface, &self.spans[i], u)? * self.side)
            }
        }
    }

    /// The derivative asked for along edge `i` at `u`: the two vertices'
    /// derivatives blended, plus the direction's own turn along the edge, plus
    /// a bump in the middle of the edge where a rail set a vertex's size apart
    /// from the condition's.
    fn desired(&self, i: usize, u: f64) -> Result<Vec3, OperationsError> {
        let next = (i + 1) % self.corners.len();
        let lerp = |a: f64, b: f64| (1.0 - u).mul_add(a, u * b);
        let (at_start, at_end) = (self.square(i, 0.0)?, self.square(i, 1.0)?);
        let blend = self.corners[i] * (1.0 - u) + self.corners[next] * u;
        let here = self.square(i, u)?;
        let shortfall = lerp(
            self.sizes[i] - self.corners[i].dot(at_start),
            self.sizes[next] - self.corners[next].dot(at_end),
        );
        let bump = 4.0 * u * (1.0 - u) * shortfall;
        let wanted = match &self.direction {
            Direction::Normal(_) => blend + here * bump,
            Direction::Tangent(surface) => {
                let turn = here - at_start * (1.0 - u) - at_end * u;
                let t = blend + turn * lerp(self.sizes[i], self.sizes[next]) + here * bump;
                let n = surface_normal_at(surface, self.spans[i].evaluate(u))?;
                t - n * t.dot(n)
            }
        };
        Ok(wanted)
    }

    /// The sine of the angle `t` makes with the tangent plane along edge `i`
    /// at `u`. Zero for a `normal` field, whose direction every control vector
    /// holds.
    fn off_plane(&self, i: usize, u: f64, t: Vec3) -> Result<f64, OperationsError> {
        match &self.direction {
            Direction::Normal(_) => Ok(0.0),
            Direction::Tangent(surface) => {
                let n = surface_normal_at(surface, self.spans[i].evaluate(u))?;
                Ok(t.normalize().map_or(1.0, |unit| unit.dot(n).abs()))
            }
        }
    }
}

fn surface_normal_at(surface: &FaceSurface, p: Point3) -> Result<Vec3, OperationsError> {
    let n = match surface {
        FaceSurface::Plane { normal, .. } => *normal,
        other => {
            let (u, v) = other
                .project_point(p)
                .ok_or_else(|| refuse("a point of a profile does not project onto its face"))?;
            other.normal(u, v)
        }
    };
    Ok(n.normalize()?)
}

/// The unit vector square to `span` at `u` within the surface's tangent plane,
/// on the side `normal × tangent` gives. Where a span's parameter stands still
/// at an end its tangent is read just inside.
fn square_in_tangent_plane(
    surface: &FaceSurface,
    span: &NurbsCurve,
    u: f64,
) -> Result<Vec3, OperationsError> {
    let mut tangent = span.derivatives(u, 1)[1];
    if tangent.length() <= 1e-12 {
        let inside = if u < 0.5 { u + 1e-3 } else { u - 1e-3 };
        tangent = span.derivatives(inside, 1)[1];
    }
    let n = surface_normal_at(surface, span.evaluate(u))?;
    Ok(n.cross(tangent).normalize()?)
}

/// The inputs to one end's field.
struct EndRequest<'a> {
    condition: LoftEnd,
    profile: FaceId,
    ring: &'a [ProfileEdgeGeom],
    near_ring: &'a [ProfileEdgeGeom],
    spans: &'a [NurbsCurve],
    /// The parameter between this profile and the next one in.
    step: f64,
    /// 1 where `v` runs from this profile to the near one, -1 where it runs in.
    along: f64,
    rail_tangents: &'a [Option<Vec3>],
}

/// One end's field, or `None` for a free end.
#[allow(clippy::too_many_lines)]
fn end_field(
    topo: &Topology,
    request: &EndRequest<'_>,
) -> Result<Option<EndField>, OperationsError> {
    let n = request.ring.len();
    let samples = ring_samples(request.ring);
    let centre = centre_of(&samples);
    let weight = match request.condition.weight() {
        Some(weight) => weight,
        None => return Ok(None),
    };

    let chords: Vec<f64> = (0..n)
        .map(|i| (request.near_ring[i].start - request.ring[i].start).length() / request.step)
        .collect();
    let mean = chords.iter().sum::<f64>() / chords.len() as f64;
    if mean <= Tolerance::new().linear {
        return Err(refuse("an end profile coincides with the next one"));
    }
    let sizes: Vec<f64> = chords
        .iter()
        .map(|&c| weight.abs() * if c > Tolerance::new().linear { c } else { mean })
        .collect();

    // The direction at each vertex, on one side of the profile.
    let (direction, unsigned, tie_side): (Direction, Vec<Vec3>, f64) = match request.condition {
        LoftEnd::Normal { .. } => {
            let normal = crate::winding::newell_normal(&samples)
                .normalize()
                .map_err(|_| refuse("a profile has no area"))?;
            let tol = Tolerance::new().linear;
            if samples
                .iter()
                .any(|&p| (p - samples[0]).dot(normal).abs() > tol)
            {
                return Err(refuse("a normal end condition needs a planar profile"));
            }
            let own = profile_outward(topo, request.profile, normal)?;
            let tie = own.dot(normal).signum() * request.along;
            (Direction::Normal(normal), vec![normal; n], tie)
        }
        LoftEnd::Tangent { face, .. } => {
            let surface = lying_on(topo, face, &along_edges(request.ring))?;
            let mut mitres = Vec::with_capacity(n);
            for i in 0..n {
                let previous = (i + n - 1) % n;
                let before = square_in_tangent_plane(&surface, &request.spans[previous], 1.0)?;
                let after = square_in_tangent_plane(&surface, &request.spans[i], 0.0)?;
                let c = before.dot(after);
                if c < -0.99 {
                    return Err(refuse("a tangent profile's corner doubles back on itself"));
                }
                mitres.push((before + after) * (1.0 / (1.0 + c)));
            }
            let out: f64 = (0..n)
                .map(|i| mitres[i].dot(request.ring[i].start - centre))
                .sum();
            let tie = out.signum() * request.along;
            (Direction::Tangent(surface), mitres, tie)
        }
        LoftEnd::Free => return Ok(None),
    };

    // Toward the next profile, the rails reading first where they run.
    let (mut heading, mut total) = (0.0, 0.0);
    for i in 0..n {
        let toward = request.rail_tangents[i].unwrap_or_else(|| {
            (request.near_ring[i].start - request.ring[i].start) * request.along
        });
        heading += toward.dot(unsigned[i]);
        total += toward.length() * unsigned[i].length();
    }
    let side = if heading.abs() > 1e-9 * total {
        heading.signum()
    } else {
        tie_side
    } * weight.signum();

    let mut corners = Vec::with_capacity(n);
    for i in 0..n {
        let wanted = unsigned[i] * side;
        let corner = match request.rail_tangents[i] {
            None => wanted * sizes[i],
            Some(tangent) => {
                check_rail_leaves(&direction, tangent, wanted, request.spans[i].evaluate(0.0))?;
                tangent
            }
        };
        corners.push(corner);
    }
    Ok(Some(EndField {
        direction,
        side,
        sizes,
        corners,
        spans: request.spans.to_vec(),
    }))
}

fn check_rail_leaves(
    direction: &Direction,
    tangent: Vec3,
    wanted: Vec3,
    at: Point3,
) -> Result<(), OperationsError> {
    let unit = tangent
        .normalize()
        .map_err(|_| refuse("a rail stands still where it leaves an end profile"))?;
    let wanted = wanted.normalize()?;
    let (off, same_side) = match direction {
        Direction::Normal(_) => (unit.cross(wanted).length(), unit.dot(wanted) > 0.0),
        Direction::Tangent(surface) => (
            unit.dot(surface_normal_at(surface, at)?).abs(),
            unit.dot(wanted) > 0.0,
        ),
    };
    if off > RAIL_CONDITION_ANGLE.sin() || !same_side {
        return Err(refuse(format!(
            "a rail leaves an end profile {:.4} degrees off its end condition",
            off.clamp(0.0, 1.0).asin().to_degrees()
        )));
    }
    Ok(())
}

/// The surface of a `tangent` face, once each of `points` is shown to lie on
/// it: within [`TANGENT_FACE_TOLERANCE`] of its surface, inside its boundary
/// or on it.
fn lying_on(
    topo: &Topology,
    face: FaceId,
    points: &[Point3],
) -> Result<FaceSurface, OperationsError> {
    let data = topo.face(face)?;
    let surface = data.surface().clone();
    let stand_off = |p: Point3| -> f64 {
        match &surface {
            FaceSurface::Plane { normal, d } => {
                let length = normal.length();
                (dot_normal_point(*normal, p) - d).abs() / length
            }
            other => other
                .project_point(p)
                .and_then(|(u, v)| other.evaluate(u, v))
                .map_or(f64::INFINITY, |q| (q - p).length()),
        }
    };
    let off = points.iter().map(|&p| stand_off(p)).fold(0.0, f64::max);
    if off > TANGENT_FACE_TOLERANCE {
        return Err(refuse(format!(
            "a profile stands {off:.3e} off the face its end is tangent to"
        )));
    }
    let wires: Vec<WireId> = std::iter::once(data.outer_wire())
        .chain(data.inner_wires().iter().copied())
        .collect();
    let mut bounds = Vec::new();
    for wire in wires {
        for oe in topo.wire(wire)?.edges() {
            let edge = topo.edge(oe.edge())?;
            bounds.push((
                edge.curve().clone(),
                topo.vertex(edge.start())?.point(),
                topo.vertex(edge.end())?.point(),
            ));
        }
    }
    let inside = brepkit_check::classify::face_contains_each(topo, face, points)?;
    for (&p, &within) in points.iter().zip(&inside) {
        if !within
            && !bounds
                .iter()
                .any(|(curve, a, b)| edge_distance(curve, *a, *b, p) <= TANGENT_FACE_TOLERANCE)
        {
            return Err(refuse("a profile runs off the face its end is tangent to"));
        }
    }
    Ok(surface)
}

// ── Side faces ──────────────────────────────────────────────────────────────

/// The space along one edge of the profiles and its curves on it: one degree,
/// one knot vector and one set of weights, so the skin needs no weights of its
/// own in `v`.
#[derive(Clone)]
struct EdgeSpace {
    degree: usize,
    knots: Vec<f64>,
    weights: Vec<f64>,
    /// One row of control points per profile.
    rows: Vec<Vec<Point3>>,
    rational: bool,
}

impl EdgeSpace {
    /// `spans` on one space, or `None` if their weights differ.
    fn of(spans: &[NurbsCurve]) -> Result<Option<Self>, OperationsError> {
        let spaces: Vec<(usize, &[f64])> = spans.iter().map(|c| (c.degree(), c.knots())).collect();
        let (degree, knots) = merged_knots(&spaces, 0);
        let rational = spans.iter().any(NurbsCurve::is_rational);
        let mut rows = Vec::with_capacity(spans.len());
        let mut weights: Vec<f64> = Vec::new();
        for span in spans {
            let coefs: Vec<[f64; 4]> = span
                .control_points()
                .iter()
                .zip(span.weights())
                .map(|(p, &w)| [p.x() * w, p.y() * w, p.z() * w, w])
                .collect();
            let onto = represent((span.degree(), span.knots()), &coefs, (degree, &knots))?;
            let scale = onto[0][3];
            let own: Vec<f64> = onto.iter().map(|c| c[3] / scale).collect();
            if weights.is_empty() {
                weights.clone_from(&own);
            } else if weights.iter().zip(&own).any(|(a, b)| (a - b).abs() > 1e-9) {
                return Ok(None);
            }
            rows.push(
                onto.iter()
                    .map(|c| Point3::new(c[0] / c[3], c[1] / c[3], c[2] / c[3]))
                    .collect(),
            );
        }
        if !rational {
            weights.fill(1.0);
        }
        Ok(Some(Self {
            degree,
            knots,
            weights,
            rows,
            rational,
        }))
    }

    fn weight_at(&self, u: f64) -> f64 {
        let coefs: Vec<[f64; 1]> = self.weights.iter().map(|&w| [w]).collect();
        spline_value(self.degree, &self.knots, &coefs, u)[0]
    }

    /// The same curves in a space holding this one.
    fn onto(&self, degree: usize, knots: Vec<f64>) -> Result<Self, OperationsError> {
        let from = (self.degree, self.knots.as_slice());
        let mut weights = Vec::new();
        let mut rows = Vec::with_capacity(self.rows.len());
        for row in &self.rows {
            let coefs: Vec<[f64; 4]> = row
                .iter()
                .zip(&self.weights)
                .map(|(p, &w)| [p.x() * w, p.y() * w, p.z() * w, w])
                .collect();
            let onto = represent(from, &coefs, (degree, &knots))?;
            weights = onto.iter().map(|c| c[3]).collect();
            rows.push(
                onto.iter()
                    .map(|c| Point3::new(c[0] / c[3], c[1] / c[3], c[2] / c[3]))
                    .collect(),
            );
        }
        Ok(Self {
            degree,
            knots,
            weights,
            rows,
            rational: self.rational,
        })
    }

    /// A finer space for a field that does not fit this one: cubic first, then
    /// halved spans.
    fn refined(&self) -> Result<Self, OperationsError> {
        if self.degree < 3 {
            let (degree, knots) = merged_knots(&[(self.degree, &self.knots)], 3);
            return self.onto(degree, knots);
        }
        let mut knots = self.knots.clone();
        for w in self.knots.windows(2) {
            if w[1] > w[0] {
                knots.push(f64::midpoint(w[0], w[1]));
            }
        }
        knots.sort_by(f64::total_cmp);
        self.onto(self.degree, knots)
    }

    fn field_coefficients(
        &self,
        field: &EndField,
        edge: usize,
    ) -> Result<Vec<[f64; 3]>, OperationsError> {
        let values = greville(self.degree, &self.knots)
            .into_iter()
            .map(|u| Ok(vec_arr(field.desired(edge, u)? * self.weight_at(u))))
            .collect::<Result<Vec<_>, OperationsError>>()?;
        let mut coefs = spline_through_greville(self.degree, &self.knots, values)?;
        for (c, w) in coefs.iter_mut().zip(&self.weights) {
            for x in c.iter_mut() {
                *x /= w;
            }
        }
        let last = coefs.len() - 1;
        coefs[0] = vec_arr(field.corners[edge]);
        coefs[last] = vec_arr(field.corners[(edge + 1) % field.corners.len()]);
        Ok(coefs)
    }

    /// Whether the rational spline of `coefs` holds the field to its tolerances
    /// between the sites it was fitted at.
    #[allow(clippy::cast_precision_loss)]
    fn fits(
        &self,
        field: &EndField,
        edge: usize,
        coefs: &[[f64; 3]],
    ) -> Result<bool, OperationsError> {
        let weighted: Vec<[f64; 3]> = coefs
            .iter()
            .zip(&self.weights)
            .map(|(c, w)| [c[0] * w, c[1] * w, c[2] * w])
            .collect();
        for span in self.knots.windows(2).filter(|w| w[1] > w[0]) {
            for k in 1..=4 {
                let u = (k as f64 / 5.0).mul_add(span[1] - span[0], span[0]);
                let t = vector(spline_value(self.degree, &self.knots, &weighted, u))
                    * (1.0 / self.weight_at(u));
                let wanted = field.desired(edge, u)?;
                if (t - wanted).length() > FIELD_SHAPE_TOLERANCE * wanted.length()
                    || field.off_plane(edge, u, t)? > TANGENT_FIT_ANGLE.sin()
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }
}

/// The inputs to one side face.
struct SideRequest<'a> {
    edge: usize,
    spans: &'a [NurbsCurve],
    start: Option<&'a EndField>,
    end: Option<&'a EndField>,
    skin: &'a SkinSpace,
    v_space: (usize, &'a [f64]),
    rows: &'a [Vec<[f64; 3]>],
    deviations: Option<&'a [Vec<[f64; 3]>]>,
}

/// One side's control net: the skin across this edge's profile curves, plus
/// the deviations at its two vertices blended along it.
fn side_net(request: &SideRequest<'_>) -> Result<SideNet, OperationsError> {
    let mut space = if let Some(space) = EdgeSpace::of(request.spans)? {
        space
    } else {
        let polynomials = request
            .spans
            .iter()
            .map(polynomial_curve)
            .collect::<Result<Vec<_>, _>>()?;
        EdgeSpace::of(&polynomials)?
            .ok_or_else(|| refuse("a side's profile curves share no spline space"))?
    };

    let fit = |space: &EdgeSpace,
               field: Option<&EndField>|
     -> Result<Option<Vec<[f64; 3]>>, OperationsError> {
        let Some(field) = field else {
            return Ok(None);
        };
        let coefs = space.field_coefficients(field, request.edge)?;
        Ok(space.fits(field, request.edge, &coefs)?.then_some(coefs))
    };
    let mut first_coefs = None;
    let mut last_coefs = None;
    for attempt in 0..=8 {
        let start = fit(&space, request.start)?;
        let end = fit(&space, request.end)?;
        if start.is_some() == request.start.is_some() && end.is_some() == request.end.is_some() {
            first_coefs = start;
            last_coefs = end;
            break;
        }
        if attempt == 8 {
            return Err(refuse(
                "an end condition cannot be held along a profile edge to tolerance",
            ));
        }
        space = space.refined()?;
    }

    let (v_degree, v_knots) = request.v_space;
    let skin = request.skin;
    let mut net = Vec::with_capacity(space.weights.len());
    for a in 0..space.weights.len() {
        let points: Vec<[f64; 3]> = space.rows.iter().map(|row| arr(row[a])).collect();
        let coefs = skin.interpolate(
            &points,
            first_coefs.as_ref().map(|f| f[a]),
            last_coefs.as_ref().map(|f| f[a]),
        );
        net.push(represent(
            (skin.degree, &skin.knots),
            &coefs,
            (v_degree, v_knots),
        )?);
    }

    let next = (request.edge + 1) % request.rows.len();
    let (degree, knots, weights, mut net) = match request.deviations {
        None => (space.degree, space.knots, space.weights, net),
        Some(deviations) => {
            with_deviation(&space, net, &deviations[request.edge], &deviations[next])?
        }
    };
    let last_u = net.len() - 1;
    net[0].clone_from(&request.rows[request.edge]);
    net[last_u].clone_from(&request.rows[next]);
    Ok(SideNet {
        degree,
        knots,
        weights,
        net,
    })
}

/// One side's control net: `net[a][b]`, `a` along the profile edges and `b`
/// across the profiles.
struct SideNet {
    degree: usize,
    knots: Vec<f64>,
    weights: Vec<f64>,
    net: Vec<Vec<[f64; 3]>>,
}

impl SideNet {
    /// The surface on the rows `first..=last` across the profiles.
    fn surface(
        &self,
        first: usize,
        last: usize,
        v_degree: usize,
        v_knots: &[f64],
    ) -> Result<NurbsSurface, OperationsError> {
        let points: Vec<Vec<Point3>> = self
            .net
            .iter()
            .map(|column| column[first..=last].iter().map(|&c| point(c)).collect())
            .collect();
        let weights = self
            .weights
            .iter()
            .map(|&w| vec![w; last - first + 1])
            .collect();
        Ok(NurbsSurface::new(
            self.degree,
            v_degree,
            self.knots.clone(),
            v_knots.to_vec(),
            points,
            weights,
        )?)
    }

    /// One band of the side: its plane when all of its net lies in one, and
    /// otherwise its NURBS.
    fn band_surface(&self, band: &Band, v_degree: usize) -> Result<FaceSurface, OperationsError> {
        let surface = self.surface(band.rows.0, band.rows.1, v_degree, &band.knots)?;
        Ok(flat_side(surface.control_points()).unwrap_or(FaceSurface::Nurbs(surface)))
    }

    /// Whether the surface turns across the full knot at row `r`, read along
    /// the side as well as at its ends: two rails that each run smoothly through
    /// it, at different speeds, still leave a crease between them.
    #[allow(clippy::cast_precision_loss)]
    fn creases_at(&self, r: usize) -> bool {
        const ALONG: usize = 8;
        let legs = |from: usize, to: usize| -> Vec<[f64; 3]> {
            self.net
                .iter()
                .zip(&self.weights)
                .map(|(column, w)| {
                    let leg = vector(column[to]) - vector(column[from]);
                    vec_arr(leg * *w)
                })
                .collect()
        };
        let (before, after) = (legs(r - 1, r), legs(r, r + 1));
        let (u0, u1) = (self.knots[0], self.knots[self.knots.len() - 1]);
        (0..=ALONG).any(|k| {
            let u = (u1 - u0).mul_add(k as f64 / ALONG as f64, u0);
            let leg = |coefs: &[[f64; 3]]| {
                vector(spline_value(self.degree, &self.knots, coefs, u)).normalize()
            };
            match (leg(&before), leg(&after)) {
                (Ok(a), Ok(b)) => a.dot(b) < 0.0 || a.cross(b).length() > CREASE_ANGLE.sin(),
                _ => false,
            }
        })
    }

    /// The curve along the side at row `b` across the profiles, a line when
    /// its control points all lie on its chord.
    fn across(&self, b: usize) -> Result<EdgeCurve, OperationsError> {
        let row: Vec<Point3> = self.net.iter().map(|column| point(column[b])).collect();
        if is_straight(&row) {
            return Ok(EdgeCurve::Line);
        }
        Ok(EdgeCurve::NurbsCurve(NurbsCurve::new(
            self.degree,
            self.knots.clone(),
            row,
            self.weights.clone(),
        )?))
    }
}

/// A stretch of the sides across the profiles with no crease inside it.
struct Band {
    /// Its first and last rows of control points.
    rows: (usize, usize),
    /// Its own clamped knots.
    knots: Vec<f64>,
}

/// The stretches between the rows where the surface creases across the
/// profiles: a kink in a rail, or an intermediate profile the rails meet at
/// different fractions of their parameters. A crease inside a NURBS face has
/// no edge to tell it by, and no mesh of the face that follows the deflection
/// settles around it. A full knot the sides run smoothly through, as a fitted
/// rail's joints have, stays inside its band.
fn v_bands(degree: usize, knots: &[f64], sides: &[SideNet]) -> Vec<Band> {
    let mut cuts = Vec::new();
    let mut first = degree + 1;
    let interior_end = knots.len() - degree - 1;
    while first < interior_end {
        let count = knots[first..interior_end]
            .iter()
            .take_while(|&&k| k.to_bits() == knots[first].to_bits())
            .count();
        if count >= degree {
            let r = first + count - degree - 1;
            if sides.iter().any(|side| side.creases_at(r)) {
                cuts.push((first, r));
            }
        }
        first += count;
    }
    let row_count = knots.len() - degree - 1;
    let mut bands = Vec::with_capacity(cuts.len() + 1);
    let (mut from_row, mut from_knot) = (0, degree + 1);
    for cut in cuts.iter().copied().chain([(interior_end, row_count - 1)]) {
        let (to_knot, to_row) = cut;
        let mut band_knots = vec![knots[from_knot - 1]; degree + 1];
        band_knots.extend_from_slice(&knots[from_knot..to_knot]);
        band_knots.extend(std::iter::repeat_n(knots[to_knot], degree + 1));
        bands.push(Band {
            rows: (from_row, to_row),
            knots: band_knots,
        });
        from_row = to_row;
        from_knot = to_knot + degree;
    }
    bands
}

/// The skin `net` with `(1 - u) a(v) + u b(v)` added. A polynomial space holds
/// that as it is, its coefficients those of `u` at the Greville abscissae. In a
/// rational one the sum is over the weight function, so the numerators take
/// `W(u) (1 - u)` and `W(u) u`, one degree up.
#[allow(clippy::type_complexity)]
fn with_deviation(
    space: &EdgeSpace,
    net: Vec<Vec<[f64; 3]>>,
    a: &[[f64; 3]],
    b: &[[f64; 3]],
) -> Result<(usize, Vec<f64>, Vec<f64>, Vec<Vec<[f64; 3]>>), OperationsError> {
    if !space.rational {
        let sites = greville(space.degree, &space.knots);
        let mut net = net;
        for (column, &g) in net.iter_mut().zip(&sites) {
            for (c, (da, db)) in column.iter_mut().zip(a.iter().zip(b)) {
                for axis in 0..3 {
                    c[axis] += (1.0 - g).mul_add(da[axis], g * db[axis]);
                }
            }
        }
        return Ok((
            space.degree,
            space.knots.clone(),
            space.weights.clone(),
            net,
        ));
    }
    let (degree, knots) = merged_knots(&[(space.degree, &space.knots)], space.degree + 1);
    let from = (space.degree, space.knots.as_slice());
    let count_v = net[0].len();
    let count_u = knots.len() - degree - 1;
    let mut raised = vec![vec![[0.0; 3]; count_v]; count_u];
    let mut weights = vec![0.0; count_u];
    for b_index in 0..count_v {
        let homogeneous: Vec<[f64; 4]> = net
            .iter()
            .zip(&space.weights)
            .map(|(column, &w)| {
                let c = column[b_index];
                [c[0] * w, c[1] * w, c[2] * w, w]
            })
            .collect();
        let onto = represent(from, &homogeneous, (degree, &knots))?;
        for (a_index, c) in onto.iter().enumerate() {
            raised[a_index][b_index] = [c[0], c[1], c[2]];
            weights[a_index] = c[3];
        }
    }
    let values: Vec<[f64; 1]> = greville(degree, &knots)
        .into_iter()
        .map(|u| [space.weight_at(u) * u])
        .collect();
    let along = spline_through_greville(degree, &knots, values)?;
    for (a_index, column) in raised.iter_mut().enumerate() {
        let (w, c1) = (weights[a_index], along[a_index][0]);
        let c0 = w - c1;
        for (c, (da, db)) in column.iter_mut().zip(a.iter().zip(b)) {
            for axis in 0..3 {
                c[axis] = (c0.mul_add(da[axis], c1 * db[axis]) + c[axis]) / w;
            }
        }
    }
    Ok((degree, knots, weights, raised))
}

/// The side's plane when all of its control net lies in one, facing as
/// `Su × Sv` does.
fn flat_side(net: &[Vec<Point3>]) -> Option<FaceSurface> {
    const FLAT: f64 = 1e-9;
    let last_u = net.len() - 1;
    let last_v = net[0].len() - 1;
    let corners = [
        net[0][0],
        net[last_u][0],
        net[last_u][last_v],
        net[0][last_v],
    ];
    let normal = crate::winding::newell_normal(&corners).normalize().ok()?;
    net.iter()
        .flatten()
        .all(|&p| (p - corners[0]).dot(normal).abs() <= FLAT)
        .then(|| FaceSurface::Plane {
            normal,
            d: dot_normal_point(normal, corners[0]),
        })
}

/// Whether two segments of `line` that are not neighbours meet.
fn line_meets_itself(line: &[Point3], closed: bool) -> bool {
    let count = line.len() - usize::from(!closed);
    let at = |k: usize| line[k % line.len()];
    let length: f64 = (0..count).map(|k| (at(k + 1) - at(k)).length()).sum();
    (0..count).any(|a| {
        (a + 2..count).any(|b| {
            !(closed && a == 0 && b == count - 1)
                && segment_segment_distance(at(a), at(a + 1), at(b), at(b + 1)).0 <= 1e-6 * length
        })
    })
}

/// Whether a line across the profiles runs back over itself anywhere along the
/// side: a weight too large for the distance to the next profile throws the
/// surface past where it comes back through.
#[allow(clippy::cast_precision_loss)]
fn doubles_back(surface: &NurbsSurface) -> bool {
    const ACROSS: usize = 8;
    const ALONG: usize = 48;
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    (0..=ACROSS).any(|i| {
        let u = (u1 - u0).mul_add(i as f64 / ACROSS as f64, u0);
        let line: Vec<Point3> = (0..=ALONG)
            .map(|k| surface.evaluate(u, (v1 - v0).mul_add(k as f64 / ALONG as f64, v0)))
            .collect();
        line_meets_itself(&line, false)
    })
}

/// Whether the surface's normal reverses within a stretch of `v` with no full
/// knot in it, where a crease could turn it: the section has shrunk through a
/// point there, as a weight that carries every corner through the axis does.
#[allow(clippy::cast_precision_loss)]
fn turns_inside_out(surface: &NurbsSurface) -> bool {
    const ACROSS: usize = 8;
    const ALONG: usize = 32;
    let (u0, u1) = surface.domain_u();
    let (v0, v1) = surface.domain_v();
    let mut stops = vec![v0];
    for run in surface
        .knots_v()
        .chunk_by(|a, b| a.to_bits() == b.to_bits())
    {
        if run[0] > v0 && run[0] < v1 && run.len() >= surface.degree_v() {
            stops.push(run[0]);
        }
    }
    stops.push(v1);
    let normal = |u: f64, v: f64| {
        let d = surface.derivatives(u, v, 1);
        d[1][0].cross(d[0][1])
    };
    (0..=ACROSS).any(|i| {
        let u = (u1 - u0).mul_add(i as f64 / ACROSS as f64, u0);
        stops.windows(2).any(|w| {
            // A normal read on a full knot is the far side's, across the crease.
            let end = w[1] - 1e-9 * (w[1] - w[0]);
            let at = |k: usize| (end - w[0]).mul_add(k as f64 / ALONG as f64, w[0]);
            (0..ALONG).any(|k| normal(u, at(k)).dot(normal(u, at(k + 1))) < 0.0)
        })
    })
}

/// Whether the loop round the sides at one value of `v` crosses itself: its
/// corners thrown past one another, as a weight or rail that carries opposite
/// sides through each other does.
#[allow(clippy::cast_precision_loss)]
fn sections_cross(sides: &[NurbsSurface]) -> bool {
    const LEVELS: usize = 32;
    const ALONG: usize = 8;
    (1..LEVELS).any(|level| {
        let ring: Vec<Point3> = sides
            .iter()
            .flat_map(|side| {
                let (u0, u1) = side.domain_u();
                let (v0, v1) = side.domain_v();
                let v = (v1 - v0).mul_add(level as f64 / LEVELS as f64, v0);
                (0..ALONG)
                    .map(move |k| side.evaluate((u1 - u0).mul_add(k as f64 / ALONG as f64, u0), v))
            })
            .collect();
        line_meets_itself(&ring, true)
    })
}

/// Whether control points all lie on the chord between their ends.
fn is_straight(row: &[Point3]) -> bool {
    const STRAIGHT: f64 = 1e-9;
    let (a, b) = (row[0], row[row.len() - 1]);
    let chord = b - a;
    chord.length() > 0.0
        && row.iter().all(|&c| {
            let off = c - a;
            let along = off.dot(chord) / chord.dot(chord);
            (off - chord * along).length() <= STRAIGHT
        })
}

/// A vertical row of a side's net as an edge's curve: a line when its control
/// points all lie on the chord between its ends.
fn row_curve(row: &[[f64; 3]], degree: usize, knots: &[f64]) -> Result<EdgeCurve, OperationsError> {
    let points: Vec<Point3> = row.iter().map(|&c| point(c)).collect();
    if is_straight(&points) {
        return Ok(EdgeCurve::Line);
    }
    let weights = vec![1.0; points.len()];
    Ok(EdgeCurve::NurbsCurve(NurbsCurve::new(
        degree,
        knots.to_vec(),
        points,
        weights,
    )?))
}

/// One end's cap: its plane when the ring lies in one, and otherwise the
/// profile's own surface, which the ring bounds.
fn cap_face(
    topo: &mut Topology,
    edges: &[EdgeId],
    ring: &[ProfileEdgeGeom],
    profile: FaceId,
    start: bool,
) -> Result<FaceId, OperationsError> {
    let samples = ring_samples(ring);
    let normal = crate::winding::newell_normal(&samples).normalize()?;
    let outward = if start { -normal } else { normal };
    let tol = Tolerance::new().linear;
    let flat = samples
        .iter()
        .all(|&p| (p - samples[0]).dot(normal).abs() <= tol);
    let surface = topo.face(profile)?.surface().clone();
    let (reversed, surface) = if flat || surface.is_planar() {
        (
            false,
            FaceSurface::Plane {
                normal: outward,
                d: dot_normal_point(outward, samples[0]),
            },
        )
    } else {
        let centre = centre_of(&samples);
        let (u, v) = surface
            .project_point(centre)
            .ok_or_else(|| refuse("a profile's boundary does not project onto its surface"))?;
        (surface.normal(u, v).dot(outward) < 0.0, surface)
    };
    let runs: Vec<OrientedEdge> = if start == reversed {
        edges.iter().map(|&e| OrientedEdge::new(e, true)).collect()
    } else {
        edges
            .iter()
            .rev()
            .map(|&e| OrientedEdge::new(e, false))
            .collect()
    };
    let wire = topo.add_wire(Wire::new(runs, true)?);
    let mut face = Face::new(wire, vec![], surface);
    if reversed {
        face.set_reversed(true);
    }
    Ok(topo.add_face(face))
}

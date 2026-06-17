//! Robust 2D boolean operations on simple polygons (union, intersection,
//! difference).
//!
//! # Approach
//!
//! Rather than weaving a doubly-linked list through coincident vertices
//! (the classic Greiner–Hormann hazard), this implementation builds a planar
//! arrangement and classifies *directed sub-edges* by their **midpoints**:
//!
//! 1. Every edge of `A` is split at all points where it meets `B` (proper
//!    crossings, vertex-on-edge T-junctions, and the endpoints of any
//!    collinear-overlap), and vice versa.
//! 2. All split points are *snapped* to a tolerance grid so that points
//!    arising independently from `A` and `B` collapse to bit-identical
//!    coordinates. This is what eliminates sliver artifacts from
//!    near-coincident edges.
//! 3. Each resulting sub-edge is classified by sampling its midpoint against
//!    the *other* polygon: `Outside`, `Inside`, or `OnBoundary` (with the
//!    relative direction of the shared boundary recorded).
//! 4. Sub-edges are selected per the operation, then traced into closed loops
//!    by following snapped coordinates.
//! 5. Loops are classified as outer (CCW) or hole (CW) by signed area and
//!    assembled into a [`PolygonBooleanResult`].
//!
//! Deciding inside/outside on a midpoint — a point in the *relative interior*
//! of a sub-edge, away from the singular intersection vertices — is what makes
//! the degenerate cases (collinear overlap, T-junctions, shared edges, corner
//! touches) robust: the classification never has to disambiguate behaviour
//! *at* a shared vertex.
//!
//! # Tolerance model
//!
//! `tol` is an absolute linear tolerance in the polygons' coordinate units.
//! Two points within `tol` of each other are treated as identical (snapped to
//! a shared grid cell of size `tol`); a point within `tol` of an edge is
//! treated as lying on it; an edge pair whose overlap exceeds `tol` in length
//! is treated as collinear-shared. Pass the same `tol` you use elsewhere for
//! the geometry in question (e.g. `Tolerance::default().linear`, or a looser
//! value for coarse data).

use crate::predicates::winding_number;
use crate::vec::Point2;

/// Which boolean operation to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp {
    /// `A ∪ B` — points in either polygon.
    Union,
    /// `A ∩ B` — points in both polygons.
    Intersection,
    /// `A \ B` — points in `A` but not `B`.
    Difference,
}

/// The result of a polygon boolean operation.
///
/// Winding convention: every `outer` loop is counter-clockwise (positive
/// signed area) and every `hole` loop is clockwise (negative signed area).
/// A point is "in" the result when it is inside an odd nesting of these loops
/// per the even-odd rule; equivalently, inside some `outer` and not inside any
/// `hole` contained by that outer.
///
/// A disjoint union yields multiple `outer` loops and no holes; a union that
/// encloses a void yields one `outer` and one `hole`; a fully-degenerate or
/// empty result yields both vectors empty.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PolygonBooleanResult {
    /// Counter-clockwise outer boundary loops.
    pub outer: Vec<Vec<Point2>>,
    /// Clockwise hole loops (voids).
    pub holes: Vec<Vec<Point2>>,
}

impl PolygonBooleanResult {
    /// `true` when the operation produced no geometry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outer.is_empty() && self.holes.is_empty()
    }

    /// Total signed area: sum of outer-loop areas minus hole-loop areas.
    ///
    /// Because outers are CCW (positive) and holes CW (negative), this is the
    /// plain sum of every loop's signed area, and equals the covered area.
    #[must_use]
    pub fn area(&self) -> f64 {
        let mut total = 0.0;
        for loop_pts in &self.outer {
            total += signed_area(loop_pts);
        }
        for loop_pts in &self.holes {
            total += signed_area(loop_pts);
        }
        total
    }
}

/// Union of two simple polygons.
///
/// Both inputs should be simple (non-self-intersecting) polygons; orientation
/// is normalized internally, so either winding is accepted. Returns the outer
/// loop(s) of the union; holes (if the union encloses a void) are dropped from
/// this convenience wrapper — use [`polygon_boolean`] if you need them.
///
/// Returns an empty `Vec` if either input is degenerate (fewer than 3
/// non-collinear vertices) or the arrangement could not be traced.
#[must_use]
pub fn polygon_union(a: &[Point2], b: &[Point2], tol: f64) -> Vec<Vec<Point2>> {
    polygon_boolean(a, b, BooleanOp::Union, tol).outer
}

/// General boolean of two simple polygons.
///
/// Orientation of the inputs is normalized internally (either winding is
/// accepted). For [`BooleanOp::Difference`] the operation is `A \ B`.
///
/// Returns an empty [`PolygonBooleanResult`] if an input is degenerate or the
/// arrangement could not be traced into closed loops; it never panics and
/// never returns a silently-wrong partial result.
#[must_use]
#[allow(clippy::too_many_lines)]
pub fn polygon_boolean(
    a: &[Point2],
    b: &[Point2],
    op: BooleanOp,
    tol: f64,
) -> PolygonBooleanResult {
    let tol = if tol > 0.0 && tol.is_finite() {
        tol
    } else {
        return PolygonBooleanResult::default();
    };

    let poly_a = match Polygon::normalized(a, tol) {
        Some(p) => p,
        None => return degenerate_fallback(a, b, op, tol),
    };
    let poly_b = match Polygon::normalized(b, tol) {
        Some(p) => p,
        None => return degenerate_fallback(a, b, op, tol),
    };

    // Split each polygon's edges at every interaction with the other, snapping
    // all split coordinates to a shared grid so coincident points merge.
    let snapper = Snapper::new(tol);
    let edges_a = split_polygon(&poly_a, &poly_b, &snapper, tol);
    let edges_b = split_polygon(&poly_b, &poly_a, &snapper, tol);

    // Classify and select directed sub-edges per the operation.
    let mut selected: Vec<DirectedEdge> = Vec::new();
    select_edges(
        &edges_a,
        &poly_b,
        op,
        EdgeSource::A,
        &snapper,
        tol,
        &mut selected,
    );
    select_edges(
        &edges_b,
        &poly_a,
        op,
        EdgeSource::B,
        &snapper,
        tol,
        &mut selected,
    );

    if selected.is_empty() {
        return PolygonBooleanResult::default();
    }

    let loops = trace_loops(selected, &snapper, tol);
    classify_loops(loops, tol)
}

// ===========================================================================
// Geometry helpers
// ===========================================================================

/// Signed area via the shoelace formula. Positive for CCW, negative for CW.
#[must_use]
pub fn signed_area(polygon: &[Point2]) -> f64 {
    let n = polygon.len();
    if n < 3 {
        return 0.0;
    }
    let mut sum = 0.0;
    for i in 0..n {
        let p = polygon[i];
        let q = polygon[(i + 1) % n];
        sum += p.x().mul_add(q.y(), -(q.x() * p.y()));
    }
    sum * 0.5
}

fn dist_sq(a: Point2, b: Point2) -> f64 {
    let dx = a.x() - b.x();
    let dy = a.y() - b.y();
    dx.mul_add(dx, dy * dy)
}

/// Distance squared from `p` to the *segment* `[a, b]` (clamped to the
/// segment, unlike the infinite-line variant in `polygon2d`).
fn point_segment_dist_sq(p: Point2, a: Point2, b: Point2) -> f64 {
    let abx = b.x() - a.x();
    let aby = b.y() - a.y();
    let len_sq = abx.mul_add(abx, aby * aby);
    if len_sq < f64::MIN_POSITIVE {
        return dist_sq(p, a);
    }
    let t = (((p.x() - a.x()) * abx) + ((p.y() - a.y()) * aby)) / len_sq;
    let t = t.clamp(0.0, 1.0);
    let proj = Point2::new(a.x() + t * abx, a.y() + t * aby);
    dist_sq(p, proj)
}

/// Parameter of the projection of `p` onto the infinite line through `[a, b]`,
/// expressed in `[0, 1]` over the segment (may fall outside `[0, 1]`).
fn project_param(p: Point2, a: Point2, b: Point2) -> f64 {
    let abx = b.x() - a.x();
    let aby = b.y() - a.y();
    let len_sq = abx.mul_add(abx, aby * aby);
    if len_sq < f64::MIN_POSITIVE {
        return 0.0;
    }
    (((p.x() - a.x()) * abx) + ((p.y() - a.y()) * aby)) / len_sq
}

fn lerp(a: Point2, b: Point2, t: f64) -> Point2 {
    Point2::new(a.x() + t * (b.x() - a.x()), a.y() + t * (b.y() - a.y()))
}

// ===========================================================================
// Coordinate snapping
// ===========================================================================

/// Snaps coordinates to a grid of cell size `tol` so that points arising
/// independently from the two polygons collapse to identical values. This is
/// the mechanism that removes spurious micro-edges (slivers): two vertices
/// closer than `tol` round to the same grid cell and therefore the same key.
#[derive(Clone, Copy)]
struct Snapper {
    inv: f64,
    cell: f64,
}

impl Snapper {
    fn new(tol: f64) -> Self {
        // Use a cell a touch larger than tol so that two points up to `tol`
        // apart reliably land in the same cell after rounding.
        let cell = tol.max(f64::MIN_POSITIVE);
        Self {
            inv: 1.0 / cell,
            cell,
        }
    }

    /// Integer grid key for a coordinate (used for equality / adjacency).
    fn key(&self, p: Point2) -> (i64, i64) {
        // round half away from zero, deterministic for finite inputs
        let kx = (p.x() * self.inv).round();
        let ky = (p.y() * self.inv).round();
        (kx as i64, ky as i64)
    }

    /// Canonical snapped position for a coordinate.
    fn snap(&self, p: Point2) -> Point2 {
        let (kx, ky) = self.key(p);
        Point2::new(kx as f64 * self.cell, ky as f64 * self.cell)
    }
}

// ===========================================================================
// Normalized polygon
// ===========================================================================

/// A CCW, duplicate-free simple polygon ready for arrangement.
struct Polygon {
    /// Vertices in CCW order, no two consecutive within `tol`.
    verts: Vec<Point2>,
}

impl Polygon {
    /// Clean and orient an input ring. Returns `None` if it is degenerate
    /// (fewer than 3 distinct vertices or zero signed area).
    fn normalized(input: &[Point2], tol: f64) -> Option<Self> {
        if input.len() < 3 {
            return None;
        }
        // Drop consecutive (and wrap-around) near-duplicates.
        let mut verts: Vec<Point2> = Vec::with_capacity(input.len());
        for &p in input {
            if let Some(&last) = verts.last()
                && dist_sq(p, last) <= tol * tol
            {
                continue;
            }
            verts.push(p);
        }
        while verts.len() >= 2 {
            let first = verts[0];
            let last = verts[verts.len() - 1];
            if dist_sq(first, last) <= tol * tol {
                verts.pop();
            } else {
                break;
            }
        }
        if verts.len() < 3 {
            return None;
        }

        let area = signed_area(&verts);
        if area.abs() <= tol * tol {
            return None;
        }
        if area < 0.0 {
            verts.reverse();
        }
        Some(Self { verts })
    }

    fn len(&self) -> usize {
        self.verts.len()
    }

    fn vert(&self, i: usize) -> Point2 {
        self.verts[i % self.verts.len()]
    }

    fn as_slice(&self) -> &[Point2] {
        &self.verts
    }
}

// ===========================================================================
// Edge splitting
// ===========================================================================

/// A directed sub-edge produced by splitting, before classification.
struct SubEdge {
    start: Point2,
    end: Point2,
}

/// Split every edge of `subject` at all parameters where it interacts with any
/// edge of `other` (crossings, T-junctions, collinear-overlap endpoints).
/// Endpoints are snapped; zero-length results are dropped.
fn split_polygon(subject: &Polygon, other: &Polygon, snapper: &Snapper, tol: f64) -> Vec<SubEdge> {
    let mut out = Vec::new();
    let n = subject.len();
    for i in 0..n {
        let a1 = subject.vert(i);
        let a2 = subject.vert(i + 1);

        // Collect split parameters in (0, 1) along this edge.
        let mut params: Vec<f64> = Vec::new();
        let m = other.len();
        for j in 0..m {
            let b1 = other.vert(j);
            let b2 = other.vert(j + 1);
            collect_edge_split_params(a1, a2, b1, b2, tol, &mut params);
        }

        // Snap-deduplicate parameters and clamp to the open interval.
        params.retain(|&t| t > 0.0 && t < 1.0);
        params.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        dedup_params(&mut params, a1, a2, tol);

        // Emit sub-edges between consecutive break points.
        let mut prev = snapper.snap(a1);
        let mut cuts: Vec<Point2> = Vec::with_capacity(params.len());
        for &t in &params {
            cuts.push(snapper.snap(lerp(a1, a2, t)));
        }
        cuts.push(snapper.snap(a2));
        for pt in cuts {
            if dist_sq(prev, pt) > tol * tol {
                out.push(SubEdge {
                    start: prev,
                    end: pt,
                });
            }
            prev = pt;
        }
    }
    out
}

/// Append the parameters along edge `[a1, a2]` (in `[0, 1]`) at which it should
/// be split because of edge `[b1, b2]`: proper crossings, the projection of
/// each `b` endpoint that lies on the segment (T-junction), and the endpoints
/// of a collinear overlap.
fn collect_edge_split_params(
    a1: Point2,
    a2: Point2,
    b1: Point2,
    b2: Point2,
    tol: f64,
    params: &mut Vec<f64>,
) {
    let tol_sq = tol * tol;

    // Endpoints of B that lie on segment A → T-junctions / shared vertices.
    for &bp in &[b1, b2] {
        if point_segment_dist_sq(bp, a1, a2) <= tol_sq {
            let t = project_param(bp, a1, a2);
            if t > 0.0 && t < 1.0 {
                params.push(t);
            }
        }
    }

    // Collinear overlap: if both B endpoints are on the *line* of A, the
    // overlap interval's interior endpoints become split points. (Endpoint
    // T-junctions above already cover the shared-vertex case; this adds the
    // case where A extends past the overlap on one or both sides.)
    let d_b1 = point_line_dist_sq(b1, a1, a2);
    let d_b2 = point_line_dist_sq(b2, a1, a2);
    if d_b1 <= tol_sq && d_b2 <= tol_sq {
        // Both endpoints collinear with A; their projections clip the overlap.
        let tb1 = project_param(b1, a1, a2);
        let tb2 = project_param(b2, a1, a2);
        for t in [tb1, tb2] {
            if t > 0.0 && t < 1.0 {
                params.push(t);
            }
        }
        return;
    }

    // Proper (transversal) intersection in the interior of both segments.
    if let Some((ta, _tb)) = segment_intersection_params(a1, a2, b1, b2)
        && ta > 0.0
        && ta < 1.0
    {
        params.push(ta);
    }
}

/// Squared distance from `p` to the *infinite line* through `[a, b]`.
fn point_line_dist_sq(p: Point2, a: Point2, b: Point2) -> f64 {
    let dx = b.x() - a.x();
    let dy = b.y() - a.y();
    let len_sq = dx.mul_add(dx, dy * dy);
    if len_sq < f64::MIN_POSITIVE {
        return dist_sq(p, a);
    }
    let cross = (p.x() - a.x()).mul_add(dy, -((p.y() - a.y()) * dx));
    (cross * cross) / len_sq
}

/// Parameters `(ta, tb)` of a proper line-line intersection, or `None` when
/// the segments are parallel/collinear (handled separately).
fn segment_intersection_params(
    a1: Point2,
    a2: Point2,
    b1: Point2,
    b2: Point2,
) -> Option<(f64, f64)> {
    let dax = a2.x() - a1.x();
    let day = a2.y() - a1.y();
    let dbx = b2.x() - b1.x();
    let dby = b2.y() - b1.y();
    let denom = dax.mul_add(dby, -(day * dbx));
    if denom.abs() < f64::MIN_POSITIVE {
        return None;
    }
    let rx = b1.x() - a1.x();
    let ry = b1.y() - a1.y();
    let ta = rx.mul_add(dby, -(ry * dbx)) / denom;
    let tb = rx.mul_add(day, -(ry * dax)) / denom;
    if (0.0..=1.0).contains(&tb) {
        Some((ta, tb))
    } else {
        None
    }
}

/// Remove parameters whose snapped 3D positions coincide within `tol`.
fn dedup_params(params: &mut Vec<f64>, a1: Point2, a2: Point2, tol: f64) {
    if params.is_empty() {
        return;
    }
    let tol_sq = tol * tol;
    let mut kept: Vec<f64> = Vec::with_capacity(params.len());
    for &t in params.iter() {
        let pt = lerp(a1, a2, t);
        let is_dup = kept
            .last()
            .is_some_and(|&pt_t| dist_sq(lerp(a1, a2, pt_t), pt) <= tol_sq);
        if !is_dup {
            kept.push(t);
        }
    }
    *params = kept;
}

// ===========================================================================
// Edge classification + selection
// ===========================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum EdgeSource {
    A,
    B,
}

/// A selected directed edge feeding the loop tracer.
struct DirectedEdge {
    start: Point2,
    end: Point2,
}

/// Position of a sub-edge midpoint relative to the other polygon.
enum MidClass {
    Inside,
    Outside,
    /// The sub-edge lies on the other polygon's boundary; `same_dir` is true
    /// when both boundaries run in the same direction along this sub-edge.
    OnBoundary {
        same_dir: bool,
    },
}

#[allow(clippy::too_many_arguments)]
fn select_edges(
    edges: &[SubEdge],
    other: &Polygon,
    op: BooleanOp,
    source: EdgeSource,
    snapper: &Snapper,
    tol: f64,
    out: &mut Vec<DirectedEdge>,
) {
    for e in edges {
        let class = classify_midpoint(e, other, snapper, tol);
        let keep = match (op, source, &class) {
            // --- Union: boundary of A∪B = parts of each outside the other,
            // plus each shared edge once (counted on A, same direction). ---
            (BooleanOp::Union, _, MidClass::Outside) => Keep::Forward,
            (BooleanOp::Union, EdgeSource::A, MidClass::OnBoundary { same_dir: true }) => {
                Keep::Forward
            }
            (BooleanOp::Union, _, _) => Keep::Drop,

            // --- Intersection: boundary = parts of each inside the other,
            // plus each shared (same-direction) edge once (on A). ---
            (BooleanOp::Intersection, _, MidClass::Inside) => Keep::Forward,
            (BooleanOp::Intersection, EdgeSource::A, MidClass::OnBoundary { same_dir: true }) => {
                Keep::Forward
            }
            (BooleanOp::Intersection, _, _) => Keep::Drop,

            // --- Difference A\B: A's parts outside B (forward), B's parts
            // inside A (reversed, so the hole winds CW), plus opposite-
            // direction shared edges (on A, forward). ---
            (BooleanOp::Difference, EdgeSource::A, MidClass::Outside) => Keep::Forward,
            (BooleanOp::Difference, EdgeSource::B, MidClass::Inside) => Keep::Reverse,
            (BooleanOp::Difference, EdgeSource::A, MidClass::OnBoundary { same_dir: false }) => {
                Keep::Forward
            }
            (BooleanOp::Difference, _, _) => Keep::Drop,
        };

        match keep {
            Keep::Forward => out.push(DirectedEdge {
                start: e.start,
                end: e.end,
            }),
            Keep::Reverse => out.push(DirectedEdge {
                start: e.end,
                end: e.start,
            }),
            Keep::Drop => {}
        }
    }
}

enum Keep {
    Forward,
    Reverse,
    Drop,
}

/// Classify a sub-edge by its midpoint against `other`.
fn classify_midpoint(e: &SubEdge, other: &Polygon, snapper: &Snapper, tol: f64) -> MidClass {
    let mid = Point2::new(
        f64::midpoint(e.start.x(), e.end.x()),
        f64::midpoint(e.start.y(), e.end.y()),
    );

    // On-boundary test: is the midpoint within tol of some edge of `other`,
    // collinear with it? If so the whole sub-edge is a shared boundary segment
    // (it was split precisely so that it does not straddle a boundary vertex).
    let tol_sq = tol * tol;
    let edir = e.end - e.start;
    let mut on_boundary: Option<bool> = None;
    let m = other.len();
    for j in 0..m {
        let b1 = other.vert(j);
        let b2 = other.vert(j + 1);
        if point_segment_dist_sq(mid, b1, b2) <= tol_sq {
            // Same direction iff the dot of edge directions is positive.
            let bdir = b2 - b1;
            let dot = edir.x().mul_add(bdir.x(), edir.y() * bdir.y());
            on_boundary = Some(dot >= 0.0);
            break;
        }
    }
    if let Some(same_dir) = on_boundary {
        return MidClass::OnBoundary { same_dir };
    }

    // Interior test by winding number on the snapped ring (consistent keys).
    let snapped: Vec<Point2> = other.as_slice().iter().map(|&p| snapper.snap(p)).collect();
    if winding_number(snapper.snap(mid), &snapped) != 0 {
        MidClass::Inside
    } else {
        MidClass::Outside
    }
}

// ===========================================================================
// Loop tracing
// ===========================================================================

/// Trace selected directed edges into closed loops by following snapped
/// coordinates. At a junction shared by several outgoing edges (a pinch
/// vertex, e.g. two polygons touching at a corner) the next edge is chosen by
/// the most counter-clockwise turn from the incoming direction; this hugs one
/// face at a time and separates the faces meeting at the pinch instead of
/// weaving them into a figure-eight.
fn trace_loops(edges: Vec<DirectedEdge>, snapper: &Snapper, tol: f64) -> Vec<Vec<Point2>> {
    use std::collections::HashMap;

    // Adjacency: snapped start key → list of edge indices leaving it.
    let mut adjacency: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (idx, e) in edges.iter().enumerate() {
        adjacency.entry(snapper.key(e.start)).or_default().push(idx);
    }

    let mut used = vec![false; edges.len()];
    let mut loops: Vec<Vec<Point2>> = Vec::new();

    for start_idx in 0..edges.len() {
        if used[start_idx] {
            continue;
        }
        let mut loop_pts: Vec<Point2> = Vec::new();
        let mut current = start_idx;
        let mut guard = 0usize;
        let max_steps = edges.len() + 1;

        loop {
            if used[current] {
                break;
            }
            used[current] = true;
            let e = &edges[current];
            loop_pts.push(e.start);
            let end_key = snapper.key(e.end);

            // Find the best unused outgoing edge from `end`.
            let Some(candidates) = adjacency.get(&end_key) else {
                break;
            };
            let incoming_dir = e.end - e.start;
            let mut best: Option<usize> = None;
            let mut best_score = f64::NEG_INFINITY;
            for &cand in candidates {
                if used[cand] {
                    continue;
                }
                let ce = &edges[cand];
                let out_dir = ce.end - ce.start;
                let score = turn_score(incoming_dir, out_dir);
                if score > best_score {
                    best_score = score;
                    best = Some(cand);
                }
            }

            match best {
                Some(next) => current = next,
                None => break,
            }

            guard += 1;
            if guard > max_steps {
                break;
            }

            // Closed the loop: arrived back at the first edge of this walk.
            if current == start_idx {
                break;
            }
        }

        // Accept only genuinely closed, non-degenerate loops.
        if loop_pts.len() >= 3 {
            let area = signed_area(&loop_pts);
            if area.abs() > tol * tol {
                loops.push(loop_pts);
            }
        }
    }

    loops
}

/// Signed turn angle (radians, in `(-pi, pi]`) from the `incoming` direction to
/// the `outgoing` direction. Larger = more counter-clockwise; the tracer picks
/// the maximum so it consistently takes the leftmost branch at a junction.
fn turn_score(incoming: crate::vec::Vec2, outgoing: crate::vec::Vec2) -> f64 {
    let inx = incoming.x();
    let iny = incoming.y();
    let outx = outgoing.x();
    let outy = outgoing.y();
    let dot = inx.mul_add(outx, iny * outy);
    let cross = inx.mul_add(outy, -(iny * outx));
    cross.atan2(dot)
}

// ===========================================================================
// Loop classification
// ===========================================================================

/// Split traced loops into CCW outers and CW holes per their signed area.
fn classify_loops(loops: Vec<Vec<Point2>>, tol: f64) -> PolygonBooleanResult {
    let mut result = PolygonBooleanResult::default();
    for loop_pts in loops {
        let area = signed_area(&loop_pts);
        if area.abs() <= tol * tol {
            continue;
        }
        if area > 0.0 {
            result.outer.push(loop_pts);
        } else {
            result.holes.push(loop_pts);
        }
    }
    result
}

// ===========================================================================
// Degenerate fallbacks
// ===========================================================================

/// When one input is degenerate (collapses to < 3 vertices / zero area), the
/// boolean reduces to a trivial case rather than failing outright.
fn degenerate_fallback(
    a: &[Point2],
    b: &[Point2],
    op: BooleanOp,
    tol: f64,
) -> PolygonBooleanResult {
    let pa = Polygon::normalized(a, tol);
    let pb = Polygon::normalized(b, tol);
    match (pa, pb) {
        (None, None) => PolygonBooleanResult::default(),
        (Some(p), None) => {
            // B is empty: A∪∅ = A, A∩∅ = ∅, A\∅ = A.
            match op {
                BooleanOp::Union | BooleanOp::Difference => single_outer(p),
                BooleanOp::Intersection => PolygonBooleanResult::default(),
            }
        }
        (None, Some(p)) => {
            // A is empty: ∅∪B = B, ∅∩B = ∅, ∅\B = ∅.
            match op {
                BooleanOp::Union => single_outer(p),
                BooleanOp::Intersection | BooleanOp::Difference => PolygonBooleanResult::default(),
            }
        }
        // Both valid: caller should not have routed here, but be safe.
        (Some(_), Some(_)) => PolygonBooleanResult::default(),
    }
}

fn single_outer(p: Polygon) -> PolygonBooleanResult {
    PolygonBooleanResult {
        outer: vec![p.verts],
        holes: Vec::new(),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::float_cmp)]
mod tests {
    use super::*;

    fn sq(x0: f64, y0: f64, s: f64) -> Vec<Point2> {
        vec![
            Point2::new(x0, y0),
            Point2::new(x0 + s, y0),
            Point2::new(x0 + s, y0 + s),
            Point2::new(x0, y0 + s),
        ]
    }

    fn rect(x0: f64, y0: f64, w: f64, h: f64) -> Vec<Point2> {
        vec![
            Point2::new(x0, y0),
            Point2::new(x0 + w, y0),
            Point2::new(x0 + w, y0 + h),
            Point2::new(x0, y0 + h),
        ]
    }

    const TOL: f64 = 1e-9;

    fn assert_area_close(got: f64, expected: f64, eps: f64) {
        assert!(
            (got - expected).abs() <= eps,
            "area mismatch: got {got}, expected {expected}"
        );
    }

    #[test]
    fn overlapping_squares_union_area() {
        // A = [0,2]^2, B = [1,3]^2; overlap = [1,2]^2 (area 1).
        let a = sq(0.0, 0.0, 2.0);
        let b = sq(1.0, 1.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1, "expected one merged outer loop");
        assert!(res.holes.is_empty(), "no holes expected");
        assert_area_close(res.area(), 4.0 + 4.0 - 1.0, 1e-7);
    }

    #[test]
    fn overlapping_squares_intersection_area() {
        let a = sq(0.0, 0.0, 2.0);
        let b = sq(1.0, 1.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Intersection, TOL);
        assert_eq!(res.outer.len(), 1);
        assert_area_close(res.area(), 1.0, 1e-7);
    }

    #[test]
    fn overlapping_squares_difference_area() {
        let a = sq(0.0, 0.0, 2.0);
        let b = sq(1.0, 1.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Difference, TOL);
        // A minus the overlap [1,2]^2 → area 4 - 1 = 3, L-shaped, no hole.
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 3.0, 1e-7);
    }

    #[test]
    fn disjoint_squares_union_two_loops() {
        let a = sq(0.0, 0.0, 1.0);
        let b = sq(5.0, 5.0, 1.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 2, "disjoint union → two outer loops");
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 2.0, 1e-7);
    }

    #[test]
    fn disjoint_squares_intersection_empty() {
        let a = sq(0.0, 0.0, 1.0);
        let b = sq(5.0, 5.0, 1.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Intersection, TOL);
        assert!(res.is_empty(), "disjoint intersection is empty");
    }

    #[test]
    fn nested_union_is_outer() {
        // B fully inside A; union = A.
        let a = sq(0.0, 0.0, 10.0);
        let b = sq(3.0, 3.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1);
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 100.0, 1e-6);
    }

    #[test]
    fn nested_intersection_is_inner() {
        let a = sq(0.0, 0.0, 10.0);
        let b = sq(3.0, 3.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Intersection, TOL);
        assert_eq!(res.outer.len(), 1);
        assert_area_close(res.area(), 4.0, 1e-7);
    }

    #[test]
    fn nested_difference_makes_hole() {
        // A with B punched out → outer = A, hole = B, net area 96.
        let a = sq(0.0, 0.0, 10.0);
        let b = sq(3.0, 3.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Difference, TOL);
        assert_eq!(res.outer.len(), 1, "outer boundary preserved");
        assert_eq!(res.holes.len(), 1, "punched void is a hole");
        assert_area_close(res.area(), 96.0, 1e-6);
    }

    #[test]
    fn shared_partial_edge_sliver_no_artifacts() {
        // The snapClip case: two rectangles sharing a partial edge with a
        // ~0.01 overlap. A = [0,10]x[0,5]; B = [0,10]x[5,8] but lifted down by
        // 0.01 so its bottom edge y=4.99 overlaps A's top region by a sliver.
        // The union must be a single clean polygon with no micro-edges.
        let a = rect(0.0, 0.0, 10.0, 5.0);
        let b = rect(0.0, 4.99, 10.0, 3.01); // top at y=8.0
        let res = polygon_boolean(&a, &b, BooleanOp::Union, 0.02);
        assert_eq!(res.outer.len(), 1, "sliver overlap → one merged rectangle");
        assert!(res.holes.is_empty(), "no sliver holes");
        // Merged rectangle is [0,10]x[0,8] = 80; overlap area ~0.1 removed once.
        assert_area_close(res.area(), 80.0, 0.2);
        // No degenerate micro-edges in the output.
        for loop_pts in &res.outer {
            for i in 0..loop_pts.len() {
                let p = loop_pts[i];
                let q = loop_pts[(i + 1) % loop_pts.len()];
                assert!(
                    dist_sq(p, q) > (0.02 * 0.02),
                    "found a sliver micro-edge of length {}",
                    dist_sq(p, q).sqrt()
                );
            }
        }
    }

    #[test]
    fn shared_full_edge_union() {
        // Two unit squares sharing a full edge (x=1) exactly → merged 1x2.
        let a = sq(0.0, 0.0, 1.0);
        let b = sq(1.0, 0.0, 1.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1, "shared-edge union is one rectangle");
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 2.0, 1e-7);
    }

    #[test]
    fn t_junction_vertex_on_edge() {
        // B's bottom edge midpoint vertex sits on A's top edge (T-junction):
        // A = [0,4]x[0,2]; B = [1,3]x[2,4] shares the segment y=2, x∈[1,3].
        let a = rect(0.0, 0.0, 4.0, 2.0);
        let b = rect(1.0, 2.0, 2.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1, "T-junction union is one polygon");
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 8.0 + 4.0, 1e-7);
    }

    #[test]
    fn touching_at_corner_union() {
        // Squares meeting only at the corner (2,2). Topologically they join
        // at a pinch point; the covered area is simply the sum, and no
        // spurious hole is introduced at the pinch.
        let a = sq(0.0, 0.0, 2.0);
        let b = sq(2.0, 2.0, 2.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        assert_area_close(res.area(), 8.0, 1e-7);
        assert!(
            res.holes.is_empty(),
            "corner pinch must not fabricate a hole"
        );
        // Every returned outer must enclose positive area (no zero-area pinch
        // loops leaking through).
        for loop_pts in &res.outer {
            assert!(
                signed_area(loop_pts) > 1e-7,
                "degenerate outer loop emitted"
            );
        }
    }

    #[test]
    fn identical_polygons_union_is_same() {
        let a = sq(0.0, 0.0, 3.0);
        let res = polygon_boolean(&a, &a, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1, "self-union is the polygon");
        assert!(res.holes.is_empty());
        assert_area_close(res.area(), 9.0, 1e-7);
    }

    #[test]
    fn identical_polygons_intersection_is_same() {
        let a = sq(0.0, 0.0, 3.0);
        let res = polygon_boolean(&a, &a, BooleanOp::Intersection, TOL);
        assert_eq!(res.outer.len(), 1);
        assert_area_close(res.area(), 9.0, 1e-7);
    }

    #[test]
    fn identical_polygons_difference_is_empty() {
        let a = sq(0.0, 0.0, 3.0);
        let res = polygon_boolean(&a, &a, BooleanOp::Difference, TOL);
        assert!(res.is_empty(), "A \\ A is empty");
    }

    #[test]
    fn degenerate_input_too_few_points() {
        let a = vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)];
        let b = sq(0.0, 0.0, 1.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, TOL);
        // A is empty → union is just B.
        assert_eq!(res.outer.len(), 1);
        assert_area_close(res.area(), 1.0, 1e-9);
    }

    #[test]
    fn degenerate_zero_tolerance_rejected() {
        let a = sq(0.0, 0.0, 1.0);
        let b = sq(0.5, 0.5, 1.0);
        let res = polygon_boolean(&a, &b, BooleanOp::Union, 0.0);
        assert!(res.is_empty(), "non-positive tolerance returns empty");
    }

    #[test]
    fn union_wrapper_returns_outer_only() {
        let a = sq(0.0, 0.0, 2.0);
        let b = sq(1.0, 1.0, 2.0);
        let loops = polygon_union(&a, &b, TOL);
        assert_eq!(loops.len(), 1);
        assert_area_close(signed_area(&loops[0]), 7.0, 1e-7);
    }

    #[test]
    fn cw_input_is_normalized() {
        // A clockwise square should be accepted (orientation normalized).
        let a_cw = vec![
            Point2::new(0.0, 0.0),
            Point2::new(0.0, 2.0),
            Point2::new(2.0, 2.0),
            Point2::new(2.0, 0.0),
        ];
        let b = sq(1.0, 1.0, 2.0);
        let res = polygon_boolean(&a_cw, &b, BooleanOp::Union, TOL);
        assert_eq!(res.outer.len(), 1);
        assert_area_close(res.area(), 7.0, 1e-7);
    }

    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Union area never exceeds the sum of the parts (overlap is not
        /// double-counted) and is at least the larger of the two.
        #[test]
        fn prop_union_area_bounded(
            ax in -3.0f64..3.0, ay in -3.0f64..3.0, asz in 0.5f64..4.0,
            bx in -3.0f64..3.0, by in -3.0f64..3.0, bsz in 0.5f64..4.0,
        ) {
            let a = sq(ax, ay, asz);
            let b = sq(bx, by, bsz);
            let area_a = asz * asz;
            let area_b = bsz * bsz;
            let res = polygon_boolean(&a, &b, BooleanOp::Union, 1e-9);
            if !res.is_empty() {
                let u = res.area();
                prop_assert!(
                    u <= area_a + area_b + 1e-6,
                    "union {u} exceeds sum {}", area_a + area_b
                );
                prop_assert!(
                    u >= area_a.max(area_b) - 1e-6,
                    "union {u} smaller than larger part {}", area_a.max(area_b)
                );
            }
        }

        /// Intersection area never exceeds either part.
        #[test]
        fn prop_intersection_area_bounded(
            ax in -3.0f64..3.0, ay in -3.0f64..3.0, asz in 0.5f64..4.0,
            bx in -3.0f64..3.0, by in -3.0f64..3.0, bsz in 0.5f64..4.0,
        ) {
            let a = sq(ax, ay, asz);
            let b = sq(bx, by, bsz);
            let area_a = asz * asz;
            let area_b = bsz * bsz;
            let res = polygon_boolean(&a, &b, BooleanOp::Intersection, 1e-9);
            let i = res.area();
            prop_assert!(
                i <= area_a.min(area_b) + 1e-6,
                "intersection {i} exceeds smaller part {}", area_a.min(area_b)
            );
        }

        /// Inclusion–exclusion: |A∪B| + |A∩B| == |A| + |B| for axis-aligned
        /// squares (areas are exact regardless of overlap topology).
        #[test]
        fn prop_inclusion_exclusion(
            ax in -3.0f64..3.0, ay in -3.0f64..3.0, asz in 0.5f64..4.0,
            bx in -3.0f64..3.0, by in -3.0f64..3.0, bsz in 0.5f64..4.0,
        ) {
            let a = sq(ax, ay, asz);
            let b = sq(bx, by, bsz);
            let area_a = asz * asz;
            let area_b = bsz * bsz;
            let u = polygon_boolean(&a, &b, BooleanOp::Union, 1e-9);
            let i = polygon_boolean(&a, &b, BooleanOp::Intersection, 1e-9);
            if !u.is_empty() {
                let lhs = u.area() + i.area();
                prop_assert!(
                    (lhs - (area_a + area_b)).abs() <= 1e-4,
                    "inclusion-exclusion off: {lhs} vs {}", area_a + area_b
                );
            }
        }

        /// Difference partition: |A\B| + |A∩B| == |A| for axis-aligned squares.
        #[test]
        fn prop_difference_partitions_a(
            ax in -3.0f64..3.0, ay in -3.0f64..3.0, asz in 0.5f64..4.0,
            bx in -3.0f64..3.0, by in -3.0f64..3.0, bsz in 0.5f64..4.0,
        ) {
            let a = sq(ax, ay, asz);
            let b = sq(bx, by, bsz);
            let area_a = asz * asz;
            let d = polygon_boolean(&a, &b, BooleanOp::Difference, 1e-9);
            let i = polygon_boolean(&a, &b, BooleanOp::Intersection, 1e-9);
            let lhs = d.area() + i.area();
            prop_assert!(
                (lhs - area_a).abs() <= 1e-4,
                "difference partition off: {lhs} vs {area_a}"
            );
        }
    }

    #[test]
    fn diagonal_triangle_square_intersection() {
        // A triangle overlapping a square with genuinely diagonal edges, so the
        // arrangement must cut edges off the snapping grid (not just at integer
        // coordinates). Square [0,4]^2; triangle (2,-1)-(6,3)-(2,7) — a
        // rightward wedge whose left vertex sits inside the square.
        let square = sq(0.0, 0.0, 4.0);
        let tri = vec![
            Point2::new(2.0, -1.0),
            Point2::new(6.0, 3.0),
            Point2::new(2.0, 7.0),
        ];
        let inter = polygon_boolean(&square, &tri, BooleanOp::Intersection, TOL);
        assert!(!inter.is_empty(), "diagonal overlap must intersect");
        // Cross-check against the convex-clip result (both convex here).
        let clipped = crate::polygon2d::sutherland_hodgman_clip(&square, &tri);
        let expected = signed_area(&clipped).abs();
        assert!(expected > 0.0, "sanity: clip area positive");
        assert_area_close(inter.area(), expected, 1e-6);
    }

    #[test]
    fn rotated_square_overlap_union_intersection() {
        // A 45-degree diamond overlapping an axis-aligned square: every
        // intersection lands at a non-integer coordinate. Verify inclusion-
        // exclusion holds, proving the off-grid arrangement is exact.
        let square = sq(0.0, 0.0, 4.0);
        let diamond = vec![
            Point2::new(2.0, -1.0),
            Point2::new(5.0, 2.0),
            Point2::new(2.0, 5.0),
            Point2::new(-1.0, 2.0),
        ];
        let area_sq = 16.0;
        let area_di = signed_area(&diamond).abs();
        let u = polygon_boolean(&square, &diamond, BooleanOp::Union, TOL);
        let i = polygon_boolean(&square, &diamond, BooleanOp::Intersection, TOL);
        assert!(!u.is_empty() && !i.is_empty());
        assert_area_close(u.area() + i.area(), area_sq + area_di, 1e-6);
    }
}
